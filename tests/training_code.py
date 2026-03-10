"""
training_code.py
===================

This module provides a robust starting point for fine‑tuning a quantized
large language model (LLM) using Low‑Rank Adaptation (LoRA).  It shows
how to perform classic supervised fine‑tuning (SFT) and illustrates how to
adapt the same setup for reinforcement learning with verifiable rewards
(e.g. Group Relative Policy Optimisation, GRPO).

UNDERSTANDING TOOL-CALLING IN THIS CONTEXT
==========================================

This code implements an "implicit tool-calling" pattern where the model learns
to generate structured outputs that YOUR CODE then verifies against real
simulations. Here's how it works:

1. TRAINING (SFT Phase):
   - Model sees examples with structured output (SNAPSHOT, INPUT_VECTOR, etc.)
   - Model learns to generate valid test vectors and predict simulation results
   - This is like teaching the model "how to call" a simulation tool

2. REINFORCEMENT (GRPO Phase):
   - Model generates test vectors (the "tool call arguments")
   - YOUR CODE runs ACTUAL fault simulation with model's predicted inputs
   - Rewards are based on whether the simulation confirms the model's predictions
   - Model improves at generating inputs that ACTUALLY detect faults

KEY INSIGHT: The model never runs simulations itself! It learns to:
   - Predict what inputs to use (like filling tool arguments)
   - Predict what the simulation will return (like anticipating tool results)
   - The reward function runs real simulations to verify predictions

This is functionally equivalent to HuggingFace's native tool-calling API
(with apply_chat_template(tools=[...])), but uses structured text tags
(SNAPSHOT:, INPUT_VECTOR:, etc.) instead of JSON tool_calls.

Dataset Structure
-----------------
The code assumes a dataset organised as an `IterableDatasetDict` with
records containing:

    * `system_content`: instructions or general system prompt for the model.
    * `user_content`: the user's query or problem description.
    * `reasoning_content`: an optional chain‑of‑thought describing how the
      assistant should reason before producing an answer.  It may include
      placeholder variables (e.g. `{fault}`) that will be resolved per example.
    * `answer_content`: the expected final answer.

Additionally, the dataset may include fields such as `fault`, `netlist`,
`input_vector`, `expected_output`, `snapshot` and `detected_faults`.  These
can be used to compute a reward during reinforcement learning via an
external verifier (e.g. `snapshot_verifier`).

The training routines here utilise the HuggingFace `transformers` and
`peft` libraries.  The base model is loaded in 4‑bit quantised form
(`BitsAndBytesConfig`) to reduce GPU memory, and LoRA adapters are
inserted into the attention and feed‑forward layers.  By default the
script uses gradient checkpointing, paged optimisers and gradient
accumulation to further reduce memory usage, following the QLoRA
guidelines.

Example usage (SFT):

```bash
python training_code.py \\
  --model_name meta-llama/Llama-3-8B-Instruct \\
  --train_split "path/to/your/dataset.jsonl" \\
  --output_dir ./finetuned_lora
```

Example usage (GRPO after SFT):

```bash
python training_code.py \\
  --method grpo \\
  --resume_from ./finetuned_lora \\
  --dataset "path/to/your/dataset.jsonl" \\
  --output_dir ./finetuned_grpo
```

The RL (GRPO) example uses a reward function that:
1. Parses the model's generated INPUT_VECTOR
2. Runs actual fault simulation using that input
3. Compares model's predicted SNAPSHOT with real simulation
4. Returns higher rewards for inputs that correctly detect faults

Version Requirements:
    - transformers == 4.57.3
    - trl == 0.26.1
    - peft == 0.13.2
    - bitsandbytes == 0.49.0
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Literal

# Add the parent directory of atpgllm to sys.path to allow importing from data_preprocessing
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'data_preprocessing'))


# =====================================================================
# MIG (Multi-Instance GPU) visibility fix — MUST run before any
# torch / CUDA import so that each DDP rank sees exactly one device.
# =====================================================================
def _setup_mig_visibility() -> None:
    """Restrict each DDP rank to its own MIG instance.

    When ``CUDA_VISIBLE_DEVICES`` contains multiple MIG UUIDs, CUDA
    still exposes only **one** device per process (``cuda:0``).  Without
    this fix the process with ``LOCAL_RANK ≥ 1`` would try to open
    ``cuda:<LOCAL_RANK>`` and crash with *"invalid device ordinal"*.

    By slicing ``CUDA_VISIBLE_DEVICES`` **before** ``import torch`` we
    guarantee every rank sees a single, unique MIG instance as
    ``cuda:0``.

    We must also reset ``LOCAL_RANK`` to ``0`` because libraries like
    ``accelerate`` use it as a CUDA device index (e.g.
    ``torch.distributed.barrier(device_ids=[local_rank])``).  The global
    ``RANK`` and ``WORLD_SIZE`` are left untouched so distributed
    communication is unaffected.
    """
    local_rank = os.environ.get("LOCAL_RANK")
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if local_rank is not None and cuda_devices:
        uuids = [u.strip() for u in cuda_devices.split(",") if u.strip()]
        if any(u.startswith("MIG-") for u in uuids) and len(uuids) > 1:
            rank = int(local_rank)
            if rank < len(uuids):
                os.environ["CUDA_VISIBLE_DEVICES"] = uuids[rank]
                # Each process now sees exactly one device (cuda:0).
                # LOCAL_RANK must reflect this so that accelerate /
                # torch.distributed don't try to address cuda:1+.
                os.environ["LOCAL_RANK"] = "0"

_setup_mig_visibility()


from datasets import load_dataset
from transformers import AutoTokenizer

from tools import TOOLS, fault_simulation_tool_handler
from reward_function_factory import RewardFunctionFactory
from callbacks import (
    ThroughputMetricsCallback,
    ContextLengthHistogramCallback,
    SFTStoppingCallback,
)
from git_utils import get_git_info
import wandb

# =====================================================================
# Re-export public names so that existing imports like
#     from training_code import ConversationExample, TrainingMode
# continue to work.
# =====================================================================
from conversation import ConversationExample                                # noqa: F401
from dataset_utils import TrainingMode, buffer_streaming_dataset, format_dataset_for_training  # noqa: F401
from model_utils import (                                                   # noqa: F401
    load_quantised_model,
    get_lora_config,
    prepare_lora_model,
    load_unsloth_model,
    prepare_unsloth_lora_model,
    load_unsloth_model_from_adapter,
    smart_sync_model_config,
    load_model_from_adapter,
    _require_unsloth,
)


# =====================================================================
# DDP helpers
# =====================================================================

def _get_device_map(use_ddp: bool) -> str | dict:
    """
    Return the appropriate ``device_map`` for model loading.

    * **DDP mode** (``use_ddp=True``): each ``accelerate`` / ``torchrun``
      process must load the full model on its *own* GPU.  We read the
      ``LOCAL_RANK`` environment variable (set automatically by the
      launcher) and pin to that device.
    * **Single-process mode** (``use_ddp=False``): fall back to
      ``"auto"`` which lets HuggingFace ``accelerate`` spread the model
      across all visible GPUs (useful when the model doesn't fit on one).

    On **MIG** nodes each DDP rank has already been restricted to a
    single MIG instance by :func:`_setup_mig_visibility`, so
    ``torch.cuda.device_count()`` returns 1 and we always use
    ``cuda:0``.
    """
    if not use_ddp:
        return "auto"
    import torch
    # On MIG (or any setup where each rank sees exactly one device),
    # LOCAL_RANK may be >0 but the only valid device is cuda:0.
    if torch.cuda.device_count() <= 1:
        return {"": "cuda:0"}
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return {"": f"cuda:{local_rank}"}


# =====================================================================
# Training orchestration
# =====================================================================

def train_with_sft(
    model_name: str,
    dataset_path: str,
    output_dir: str,
    resume_from: str = None,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    max_steps: int = -1,
    report_to: str = "wandb",
    use_vllm: bool = False,
    vllm_server_url: str = None,
    eval_buffer_size: int = 30,
    max_model_len: int = 16384,
    use_unsloth: bool = False,
    use_ddp: bool = False,
    **kwargs,
) -> None:
    """
    Perform supervised fine‑tuning (SFT) on a quantised LLM using LoRA
    adapters.  The dataset must be an IterableDataset or JSONL file
    containing the necessary fields.  The resulting LoRA weights will be
    saved in *output_dir*.

    An ``SFTStoppingCallback`` is automatically attached.  It monitors:

    1. **Format compliance** – can the reward function parse ≥ 95 % of
       validation outputs?
    2. **Output diversity** – are 10 completions from a single prompt
       sufficiently distinct?

    When both criteria are met the trainer stops, and the checkpoint is
    ready to be used as the starting point for GRPO.

    Parameters
    ----------
    model_name : str
        HuggingFace Hub identifier of the base model (e.g.
        ``"Qwen/Qwen2.5-72B-Instruct"``).
    dataset_path : str
        Path to a local JSONL file or dataset identifier.
    output_dir : str
        Directory where the LoRA adapter weights and training artefacts
        will be saved.
    resume_from : str, optional
        Path to a previously saved adapter directory to resume training
        from.
    use_vllm : bool
        If *True*, validation generation in the stopping callback uses
        a **persistent vLLM server** for faster batch inference.  Start
        the server on a spare GPU before training with dynamic LoRA
        loading enabled::

            VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \\
                CUDA_VISIBLE_DEVICES=2 vllm serve <model_name> \\
                --enable-lora --max-lora-rank 64 --port 8000

    vllm_server_url : str, optional
        Base URL of the running vLLM server (default
        ``"http://localhost:8000"`` when ``use_vllm=True``).
    eval_buffer_size : int
        Number of examples to buffer from the test split for the
        stopping callback validation (default 30).
    max_model_len : int
        Maximum model length for filtering dataset (default 16384).
    use_unsloth : bool
        If *True*, use unsloth's ``FastLanguageModel`` for model loading
        and LoRA injection.
    use_ddp : bool
        If *True*, use Distributed Data Parallel (DDP) mode.  Each
        ``accelerate`` / ``torchrun`` process loads the full model on its
        own GPU (via ``LOCAL_RANK``).  When *False* the model is loaded
        with ``device_map="auto"`` which spreads it across all visible
        GPUs (pipeline parallelism, no data parallelism).  DDP requires
        the model to fit on a single GPU in 4-bit (e.g. 72B ≈ 36 GB
        fits on H100 80 GB but not A100 40 GB).
    """
    # Lazy-import SFTTrainer and SFTConfig
    from trl import SFTTrainer, SFTConfig
    import torch
    print(f"Available GPUs: {torch.cuda.is_available()}, Num GPUs: {torch.cuda.device_count()}")

    # Validate unsloth availability early
    if use_unsloth:
        _require_unsloth()
        print("[Unsloth] Enabled – using FastLanguageModel for optimised training")

    # Determine device_map: per-GPU for DDP, "auto" otherwise
    device_map = _get_device_map(use_ddp)
    if use_ddp:
        print(f"[DDP] Enabled – loading model with device_map={device_map}")

    # Load model and tokenizer — either from saved adapter or fresh
    if resume_from:
        print(f"Resuming training from: {resume_from}")
        if use_unsloth:
            model, tokenizer = load_unsloth_model_from_adapter(resume_from, max_seq_length=max_model_len, fast_inference=True, device_map=device_map)
        else:
            model, tokenizer = load_model_from_adapter(resume_from, device_map=device_map)
    else:
        if use_unsloth:
            model, tokenizer = load_unsloth_model(model_name)
            model = prepare_unsloth_lora_model(model)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if not tokenizer.eos_token:
                tokenizer.add_special_tokens({"eos_token": "</s>"})
            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token
            base_model = load_quantised_model(model_name, device_map=device_map)
            model = prepare_lora_model(base_model)

    # Synchronize the model's config with the tokenizer's special token IDs
    model = smart_sync_model_config(model, tokenizer)

    # Load dataset (streaming for memory efficiency)
    data = load_dataset(dataset_path, split="train", streaming=True)

    # Format dataset into chat prompts
    train_dataset = format_dataset_for_training(data, tokenizer, TrainingMode.SFT)

    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        max_steps=max_steps,
        logging_steps=5,
        save_steps=10,
        ddp_find_unused_parameters=False if use_ddp else None,
        gradient_checkpointing=True,
        # DDP + gradient checkpointing + LoRA requires non-reentrant
        # checkpointing to avoid "parameter marked ready twice" errors.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        report_to=report_to,
        dataset_text_field="text",
        max_length=8192,
        # Enable token counting so ThroughputMetricsCallback can compute tokens/sec
        include_num_input_tokens_seen=True,
        # DDP with IterableDataset: each process must fetch its own batch
        # independently.  The default dispatch_batches=True tries to
        # concatenate batches from all workers on the main process, which
        # fails when sequences have different lengths.
        **({"accelerator_config": {"dispatch_batches": False}} if use_ddp else {}),
    )

    shared_callbacks = [
        ThroughputMetricsCallback(),
        ContextLengthHistogramCallback(pad_token_id=tokenizer.pad_token_id, tokenizer=tokenizer),
        SFTStoppingCallback(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            eval_buffer_size=eval_buffer_size,
            tool_functions={"fault_simulation_tool": fault_simulation_tool_handler},
            tools_schema=TOOLS,
            format_threshold=0.95,
            diversity_threshold=0.2,
            diversity_num_generations=10,
            use_vllm=use_vllm,
            vllm_server_url=vllm_server_url,
            max_new_tokens=8192,
            vllm_max_context=32768,
            min_steps=50,
            patience=1,
            temperature=0.7,
            generation_batch_size=50 if use_vllm else 8,
        ),
    ]

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        args=training_args,
        callbacks=shared_callbacks,
    )
    trainer.train()
    trainer.save_model(output_dir)


def train_with_grpo(
    model_name: str,
    dataset_path: str,
    output_dir: str,
    resume_from: str = None,
    buffer_size: int = 10000,
    per_device_train_batch_size: int = 8,
    gradient_accumulation_steps: int = 1,
    max_steps: int = -1,
    report_to: str = "wandb",
    use_vllm: bool = False,
    vllm_mode: Optional[Literal["colocate", "server"]] = None,
    vllm_server_url: str = None,
    num_generations: int = 8,
    steps_per_generation: int = 1,
    max_model_len: int = 8192,
    max_completion_length: int = 4096,
    max_prompt_length: int = 4096,
    use_dual_adapter: bool = True,
    use_unsloth: bool = False,
    use_ddp: bool = False,
    **kwargs,
) -> None:
    """
    Group Relative Policy Optimisation (GRPO) training.

    GRPO is an online reinforcement learning algorithm that requires
    reward functions to evaluate generated responses.

    Parameters
    ----------
    model_name : str
        Identifier of the base model.
    dataset_path : str
        Path to the JSONL dataset used for exploration.
    output_dir : str
        Directory to store the resulting model.
    resume_from : str, optional
        Path to a previously saved adapter directory to resume from.
        Required for dual-adapter mode.
    buffer_size : int
        Maximum examples to buffer from the streaming dataset into
        memory.  ``GRPOTrainer`` doesn't support streaming datasets.
    per_device_train_batch_size : int
    gradient_accumulation_steps : int
    max_steps : int
    report_to : str
    use_vllm : bool
    vllm_mode : Optional[Literal["colocate", "server"]]
    vllm_server_url : str
    num_generations : int
    steps_per_generation : int
    max_model_len : int
        Maximum model length (default 8192).
    max_completion_length : int
        Maximum completion length (default 4096).
    max_prompt_length : int
        Maximum prompt length (default 4096).
    use_dual_adapter : bool
        If *True* (default), uses ``DualAdapterGRPOTrainer`` which keeps
        the SFT adapter isolated and avoids ``merge_and_unload``.
    use_unsloth : bool
        If *True*, use unsloth's ``FastLanguageModel`` for model loading.
    use_ddp : bool
        If *True*, use Distributed Data Parallel (DDP) mode.  See
        :func:`train_with_sft` for details.
    """
    # Validate unsloth availability early
    if use_unsloth:
        _require_unsloth()
        print("[Unsloth] Enabled – using FastLanguageModel for optimised GRPO training")

    # Lazy-import GRPO trainers and GRPOConfig to avoid pulling in
    # trl.GRPOTrainer (and its vllm dependency) during SFT-only runs.
    from trl import GRPOConfig
    from tool_calling_grpo_trainer import ToolCallingGRPOTrainer
    from dual_adapter_grpo_trainer import DualAdapterGRPOTrainer

    # Determine device_map: per-GPU for DDP, "auto" otherwise
    device_map = _get_device_map(use_ddp)
    if use_ddp:
        print(f"[DDP] Enabled – loading model with device_map={device_map}")

    # Define LoRA config — needed for both fresh start and resume
    lora_config = get_lora_config(use_unsloth=use_unsloth)

    # Load model and tokenizer
    if resume_from:
        print(f"Resuming GRPO training from SFT checkpoint: {resume_from}")
        if use_unsloth:
            model, tokenizer = load_unsloth_model_from_adapter(resume_from, max_seq_length=max_model_len, fast_inference=True, device_map=device_map)
        else:
            model, tokenizer = load_model_from_adapter(resume_from, device_map=device_map)

        if use_dual_adapter:
            print("Using dual-adapter mode (DualAdapterGRPOTrainer)")
            print("  - SFT adapter will be kept isolated (no merging)")
            print("  - Reference model: SFT adapter only")
            print("  - Policy model: new policy adapter")
            peft_config_for_trainer = None
            policy_lora_config = lora_config
        else:
            print("Using standard mode (GRPOTrainer with merge_and_unload)")
            print("  - WARNING: SFT adapter will be merged into base weights")
            print("  - This may cause precision loss with 4-bit quantization")
            peft_config_for_trainer = lora_config
            policy_lora_config = None
    else:
        if use_unsloth:
            model, tokenizer = load_unsloth_model(model_name)
            model = prepare_unsloth_lora_model(model, lora_config)
            peft_config_for_trainer = None
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if not tokenizer.eos_token:
                tokenizer.add_special_tokens({"eos_token": "</s>"})
            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token
            model = load_quantised_model(model_name, device_map=device_map)
            peft_config_for_trainer = lora_config

        if use_dual_adapter:
            print("Note: Dual-adapter mode requires --resume_from with an SFT checkpoint")
            print("  Falling back to standard mode for fresh start")
        policy_lora_config = None
        use_dual_adapter = False

    # Synchronize model config with tokenizer special tokens
    model = smart_sync_model_config(model, tokenizer)

    # Load and format dataset
    print(f"Loading dataset from: {dataset_path}")
    data = load_dataset(dataset_path, split="train", streaming=True)
    formatted_data = format_dataset_for_training(data, tokenizer, TrainingMode.GRPO)

    print("Buffering streaming dataset (GRPOTrainer requires non-streaming Dataset)...")
    train_dataset = buffer_streaming_dataset(
        formatted_data, buffer_size=buffer_size, shuffle=True, seed=42, tokenizer=tokenizer, max_prompt_length=max_prompt_length,
    )

    # =========================================================================
    # REWARD FUNCTION SETUP
    # =========================================================================
    print("Initializing reward function factory...")
    reward_factory = RewardFunctionFactory(config_path='sim_config.json')
    reward_fn = reward_factory.create_reward_function()
    
    # Set up training arguments using GRPOConfig.  RL typically requires more
    # exploration, so use a smaller learning rate and more steps.
    # Note: batch_size must be divisible by num_generations (default=8)
    #
    # IMPORTANT: steps_per_generation controls how many accumulation steps worth of
    # samples are generated at once. Default is gradient_accumulation_steps, which
    # can cause OOM for large models. Set it to 1 to generate only per_device_train_batch_size
    # samples at a time, trading off generation throughput for memory efficiency.
    #
    # Example with per_device_train_batch_size=2, gradient_accumulation_steps=64:
    #   - steps_per_generation=64 (default): generates 2*64=128 samples at once (OOM!)
    #   - steps_per_generation=1: generates 2*1=2 samples at a time (memory efficient)
    training_args = GRPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=5e-5,
        max_steps=max_steps,
        logging_steps=1,
        save_steps=10,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=report_to,
        # GRPO-specific options
        loss_type="dapo", # "grpo", "dr_grpo", "dapo", "bnpo", "cispo", default is "dapo"
        num_generations=max(num_generations, 2),
        steps_per_generation=steps_per_generation,
        max_completion_length=max_completion_length,
        use_vllm=use_vllm,
        vllm_mode=vllm_mode,
        vllm_server_base_url=vllm_server_url,
        importance_sampling_level="sequence", # "token" or "sequence" : sequence provides more stable training and better alignment with sequence-level rewards
    )

    shared_callbacks = [
        ThroughputMetricsCallback(),
        ContextLengthHistogramCallback(pad_token_id=tokenizer.pad_token_id, tokenizer=tokenizer),
    ]

    if use_dual_adapter:
        trainer = DualAdapterGRPOTrainer(
            model=model,
            reward_funcs=[reward_fn],
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            policy_lora_config=policy_lora_config,
            tool_functions={'fault_simulation_tool': fault_simulation_tool_handler},
            callbacks=shared_callbacks,
            tools=TOOLS,
        )
    else:
        trainer = ToolCallingGRPOTrainer(
            model=model,
            reward_funcs=[reward_fn],
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            peft_config=peft_config_for_trainer,  # Triggers merge_and_unload if model is PeftModel
            tool_functions={'fault_simulation_tool': fault_simulation_tool_handler},
            callbacks=shared_callbacks,
        )

    trainer.train()
    trainer.save_model(output_dir)


# =====================================================================
# CLI helpers (environment variable defaults)
# =====================================================================

def _model_name():
    return os.environ.get('MODEL', None)

def _train_dataset():
    return os.environ.get('TRAIN_DATASET', None)

def _output_dir():
    return os.environ.get('OUTPUT_DIR', "./finetuned_model")

def _method():
    return os.environ.get('METHOD', "sft")

def _resume_from():
    return os.environ.get('RESUME_FROM', None)

def _buffer_size():
    return int(os.environ.get('BUFFER_SIZE', 10000))

def _per_device_train_batch_size():
    return int(os.environ.get('PER_DEVICE_TRAIN_BATCH_SIZE', 2))

def _gradient_accumulation_steps():
    return int(os.environ.get('GRADIENT_ACCUMULATION_STEPS', 1))

def _max_steps():
    return int(os.environ.get('MAX_STEPS', 1))

def _report_to():
    return os.environ.get('REPORT_TO', "wandb")

def _num_generations():
    return int(os.environ.get('NUM_GENERATIONS', 8))

def _steps_per_generation():
    return int(os.environ.get('STEPS_PER_GENERATION', 2))

def _max_model_len():
    return int(os.environ.get('MAX_MODEL_LEN', 8192))

def _max_completion_length():
    return int(os.environ.get('MAX_COMPLETION_LENGTH', 4096))

def _max_prompt_length():
    return int(os.environ.get('MAX_PROMPT_LENGTH', 4096))

def _use_unsloth():
    return os.environ.get('USE_UNSLOTH', '0').lower() in ('1', 'true', 'yes')

def _vllm_mode():
    return os.environ.get('VLLM_MODE', "server")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine‑tune a quantised LLM with LoRA")
    parser.add_argument("--model_name", type=str, default=_model_name(),
                        help="Base model name on the HF hub (not required if --resume_from is provided)")
    parser.add_argument("--dataset", type=str, default=_train_dataset(),
                        help="Path to JSONL dataset with training records")
    parser.add_argument("--output_dir", type=str, default=_output_dir(),
                        help="Output directory")
    parser.add_argument("--method", type=str, default=_method(), choices=["sft", "grpo"],
                        help="Training method: sft or grpo")
    parser.add_argument("--resume_from", type=str, default=_resume_from(),
                        help="Path to a saved adapter directory to resume training from")
    parser.add_argument("--buffer_size", type=int, default=_buffer_size(),
                        help="Number of examples to buffer from streaming dataset for GRPO")
    parser.add_argument("--max_steps", type=int, default=_max_steps(),
                        help="Maximum number of training steps")
    parser.add_argument("--per_device_train_batch_size", type=int,
                        default=_per_device_train_batch_size(),
                        help="Per device train batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int,
                        default=_gradient_accumulation_steps(),
                        help="Gradient accumulation steps")
    parser.add_argument("--report_to", type=str, default=_report_to(),
                        help="Report to: wandb or none")
    parser.add_argument("--use_vllm", action="store_true", help="Use VLLM")
    parser.add_argument("--vllm_server_url", type=str, default=None,
                        help="URL of a running vLLM server for SFT eval generation "
                             "(e.g. http://localhost:8000). Used when --use_vllm is set.")
    parser.add_argument("--vllm_mode", type=str, default=_vllm_mode(),
                        help="VLLM mode: 'colocate' or 'server'")
    parser.add_argument("--num_generations", type=int, default=_num_generations(),
                        help="Number of generations per prompt to sample.")
    parser.add_argument("--steps_per_generation", type=int, default=_steps_per_generation(),
                        help="Steps per generation")
    parser.add_argument("--max_model_len", type=int, default=_max_model_len(),
                        help="Maximum model length")
    parser.add_argument("--max_completion_length", type=int, default=_max_completion_length(),
                        help="Maximum completion length")
    parser.add_argument("--max_prompt_length", type=int, default=_max_prompt_length(),
                        help="Maximum prompt length")
    parser.add_argument("--use_dual_adapter", action="store_true", default=True,
                        help="Use dual-adapter mode (keeps SFT adapter isolated, avoids merge). Default: True")
    parser.add_argument("--use_unsloth", action="store_true", default=_use_unsloth(),
                        help="Use unsloth's FastLanguageModel for optimised training "
                             "(2x faster, 80%% less VRAM). Requires: pip install unsloth")
    parser.add_argument("--use_ddp", action="store_true", default=False,
                        help="Use Distributed Data Parallelization. It's recommended for SFT + unsloth training.")
    args = parser.parse_args()

    # Validate integer arguments
    for field in ('buffer_size', 'per_device_train_batch_size',
                  'gradient_accumulation_steps', 'max_steps',
                  'num_generations', 'steps_per_generation'):
        try:
            setattr(args, field, int(getattr(args, field)))
        except ValueError:
            parser.error(f"Invalid {field}: {getattr(args, field)}")

    # Convert args to dict and fix parameter naming
    args_dict = vars(args)
    args_dict['dataset_path'] = args_dict.pop('dataset')
    method = args_dict.pop('method')

    # 1. Combine CLI args with Git info for the config
    run_config = {"method": method, **args_dict, **get_git_info()}

    # 2. Initialize wandb explicitly to capture everything from this point forward
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "huggingface"),
        config=run_config,
        name=f"{args_dict.get('output_dir', 'run')}" if method in args_dict.get('output_dir') else f"{method}-{args_dict.get('output_dir', 'run')}",
        settings=wandb.Settings(console="wrap") # Forces capture of Python stdout/stderr
    )

    # 3. Tell wandb to live-stream the Slurm bash log file to the cloud
    slurm_log = os.environ.get("SLURM_LOG_FILE")
    if slurm_log and os.path.exists(slurm_log):
        # policy="live" continuously uploads the file as Slurm writes to it
        wandb.save(os.path.abspath(slurm_log), base_path=os.getcwd(), policy="live")
    
    # 4. Tell wandb to live-stream the Slurm error file to the cloud
    slurm_error = os.environ.get("SLURM_ERROR_FILE")
    if slurm_error and os.path.exists(slurm_error):
        # policy="live" continuously uploads the file as Slurm writes to it
        wandb.save(os.path.abspath(slurm_error), base_path=os.getcwd(), policy="live")
    # -----------------------

    if method == "sft":
        train_with_sft(**args_dict)
    elif method == "grpo":
        train_with_grpo(**args_dict)
    else:
        raise ValueError(f"Unknown training method: {method}")


if __name__ == "__main__":
    main()
