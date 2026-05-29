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
    TrainingStateCheckpointCallback,
    read_cumulative_skip_from_checkpoint,
    read_launch_skip_from_checkpoint,
    validate_training_state_checkpoint,
)
from git_utils import get_git_info
import wandb

# =====================================================================
# Re-export public names so that existing imports like
#     from training_code import ConversationExample, TrainingMode
# continue to work.
# =====================================================================
from conversation import ConversationExample                                # noqa: F401
from dataset_utils import (  # noqa: F401
    TrainingMode,
    buffer_streaming_dataset,
    filter_streaming_dataset_by_prompt_length,
    format_dataset_for_training,
)
from model_utils import (                                                   # noqa: F401
    load_quantised_model,
    get_lora_config,
    prepare_lora_model,
    smart_sync_model_config,
    load_model_from_adapter,
    patch_qwen_chat_template_for_assistant_mask,
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
    resume_training_state: bool = False,
    skip_buffer_size: int = 0,
    skip_batch_size: int = 128,
    skip_num_workers: int | None = None,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    max_steps: int = -1,
    report_to: str = "wandb",
    max_model_len: int = 16384,
    max_prompt_length: int = 4096,
    use_ddp: bool = False,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_target_modules: list[str] | None = None,
    assistant_only_loss: bool = True,
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
        from.  Loads the LoRA adapter weights only — training starts
        from step 0 with a fresh optimizer unless
        ``resume_training_state`` is also set.
    resume_training_state : bool
        When *True* **and** ``resume_from`` points to a Trainer
        checkpoint directory (one that contains ``trainer_state.json``,
        ``optimizer.pt``, ``scheduler.pt``, etc.), the full training
        state is restored: optimizer weights, LR schedule position,
        global step / epoch counters, and RNG seeds.  Training resumes
        exactly where it left off.

        Combine the two flags for crash recovery::

            python training_code.py \\
                --resume_from ./output_dir/checkpoint-50 \\
                --resume_training_state \\
                --output_dir ./output_dir ...

        Without ``--resume_training_state``, ``--resume_from`` loads
        only the adapter weights and starts a fresh training run.
    skip_buffer_size : int
        Skip this many training examples from the start of the streaming
        dataset (stream order, after gate filter, chat formatting, and
        ``max_prompt_length`` filtering when using ``messages`` format).
        Use with ``--resume_from`` when restarting SFT on
        the same dataset so already-seen samples are not trained again.
        Independent of ``--resume_training_state`` (which restores step
        counters but does not advance the data stream).
    skip_batch_size : int
        Raw rows per batch for the fast streaming skip/filter pipeline.
    skip_num_workers : int, optional
        Parallel workers for skip/filter/format batches (default: min(16, CPUs)).
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
        Maximum sequence length passed to ``SFTConfig(max_length=...)``.
    max_prompt_length : int
        For ``messages`` format (``assistant_only_loss=True``): drop
        examples whose system+user chat-template length is ``>=`` this
        value. Not applied in legacy ``text`` format.
    use_ddp : bool
        If *True*, use Distributed Data Parallel (DDP) mode.  Each
        ``accelerate`` / ``torchrun`` process loads the full model on its
        own GPU (via ``LOCAL_RANK``).  When *False* the model is loaded
        with ``device_map="auto"`` which spreads it across all visible
        GPUs (pipeline parallelism, no data parallelism).  DDP requires
        the model to fit on a single GPU in 4-bit (e.g. 72B ≈ 36 GB
        fits on H100 80 GB but not A100 40 GB).
    lora_rank, lora_alpha, lora_target_modules
        LoRA hyper-parameters (see :func:`model_utils.get_lora_config`).
        Ignored when ``resume_from`` loads an existing adapter (architecture
        comes from the checkpoint).
    """
    lora_config = get_lora_config(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
    )

    # Lazy-import SFTTrainer and SFTConfig
    from trl import SFTTrainer, SFTConfig
    import torch
    print(f"Available GPUs: {torch.cuda.is_available()}, Num GPUs: {torch.cuda.device_count()}")

    # -- Validate resume flags -----------------------------------------
    resume_checkpoint = None
    if resume_training_state:
        if not resume_from:
            raise ValueError(
                "--resume_training_state requires --resume_from to point "
                "to a Trainer checkpoint directory."
            )
        validate_training_state_checkpoint(resume_from)
        resume_checkpoint = resume_from
        print(f"[Resume] Will restore full training state from: {resume_from}")

    # Determine device_map: per-GPU for DDP, "auto" otherwise
    device_map = _get_device_map(use_ddp)
    if use_ddp:
        print(f"[DDP] Enabled – loading model with device_map={device_map}")

    # Load model and tokenizer — either from saved adapter or fresh
    if resume_from:
        print(f"Resuming training from: {resume_from}")
        model, tokenizer = load_model_from_adapter(resume_from, device_map=device_map)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if not tokenizer.eos_token:
            tokenizer.add_special_tokens({"eos_token": "</s>"})
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        base_model = load_quantised_model(model_name, device_map=device_map)
        model = prepare_lora_model(base_model, lora_config)

    # Synchronize the model's config with the tokenizer's special token IDs
    model = smart_sync_model_config(model, tokenizer)

    # ------------------------------------------------------------------
    # Loss-mode selection: assistant-only (new, recommended) vs legacy LM.
    # ------------------------------------------------------------------
    if assistant_only_loss:
        # Patch the tokenizer's chat template so SFTTrainer can produce
        # assistant_masks. For Qwen2/2.5/3 the stock template lacks
        # {% generation %} markers; this injects them around the two assistant
        # rendering paths. No-op for templates that already mark assistant
        # blocks (e.g. Llama-3.1+ Instruct).
        if patch_qwen_chat_template_for_assistant_mask(tokenizer):
            print("[SFT] Patched tokenizer chat_template with {% generation %} "
                  "markers for assistant_only_loss=True.")
        sft_format = "messages"
        print("[SFT] Loss mode: assistant-only (system / user / tool-response "
              "tokens are masked out).")
    else:
        sft_format = "text"
        print("[SFT] Loss mode: legacy language modeling (loss over ALL "
              "non-pad tokens, incl. system / user / tool-response).")

    # Load dataset (streaming for memory efficiency)
    data = load_dataset(dataset_path, split="train", streaming=True)

    # Format dataset into the chosen SFT layout
    train_dataset = format_dataset_for_training(
        data,
        tokenizer,
        TrainingMode.SFT,
        sft_format=sft_format,
        max_prompt_length=max_prompt_length if sft_format == "messages" else None,
        skip_buffer_size=skip_buffer_size if sft_format == "messages" else 0,
        skip_batch_size=skip_batch_size,
        skip_num_workers=skip_num_workers,
    )

    if resume_training_state and skip_buffer_size:
        print(
            "[SFT] Warning: --resume_training_state restores trainer step/checkpoint "
            "state; non-zero --skip_buffer_size also skips a dataset prefix. "
            "Combine only if intentional."
        )

    # ------------------------------------------------------------------
    # Build SFTConfig kwargs.  The base set is shared by both loss modes;
    # the loss-specific keys are injected below so the two paths are
    # explicit and easy to diff.
    # ------------------------------------------------------------------
    sft_kwargs: dict = dict(
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
        gradient_checkpointing_kwargs={"use_reentrant": False} if use_ddp else None,
        bf16=True,
        report_to=report_to,
        max_length=max_model_len,
        # Enable token counting so ThroughputMetricsCallback can compute tokens/sec
        include_num_input_tokens_seen=True,
    )
    if assistant_only_loss:
        # SFTTrainer detects the conversational layout from the "messages"
        # column and applies the chat template internally with
        # return_assistant_tokens_mask=True.
        sft_kwargs["assistant_only_loss"] = True
    else:
        # Legacy: pre-rendered "text" column, full LM loss over the chat.
        sft_kwargs["dataset_text_field"] = "text"

    if use_ddp:
        # DDP with IterableDataset: each process must fetch its own batch
        # independently. The default dispatch_batches=True tries to
        # concatenate batches from all workers on the main process, which
        # fails when sequences have different lengths.
        sft_kwargs["accelerator_config"] = {"dispatch_batches": False}

    training_args = SFTConfig(**sft_kwargs)

    shared_callbacks = [
        ThroughputMetricsCallback(),
        ContextLengthHistogramCallback(pad_token_id=tokenizer.pad_token_id, tokenizer=tokenizer),
        TrainingStateCheckpointCallback(
            method="sft",
            launch_skip_buffer_size=skip_buffer_size,
        ),
    ]

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        args=training_args,
        callbacks=shared_callbacks,
    )
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(output_dir)


def train_with_grpo(
    model_name: str,
    dataset_path: str,
    output_dir: str,
    resume_from: str = None,
    resume_training_state: bool = False,
    buffer_size: int = 10000,
    skip_buffer_size: int = 0,
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
    use_ddp: bool = False,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_target_modules: list[str] | None = None,
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
        Required for dual-adapter mode.  Loads adapter weights only —
        training starts from step 0 with a fresh optimizer unless
        ``resume_training_state`` is also set.
    resume_training_state : bool
        When *True* **and** ``resume_from`` points to a Trainer
        checkpoint directory (one that contains ``trainer_state.json``,
        ``optimizer.pt``, ``scheduler.pt``, etc.), the full training
        state is restored: optimizer weights, LR schedule position,
        global step / epoch counters, and RNG seeds.  Training resumes
        exactly where it left off.

        Combine the two flags for crash recovery::

            python training_code.py --method grpo \\
                --resume_from ./output_dir/checkpoint-20 \\
                --resume_training_state \\
                --output_dir ./output_dir ...

        Without ``--resume_training_state``, ``--resume_from`` loads
        only the adapter weights and starts a fresh training run.
    buffer_size : int
        Maximum examples to buffer from the streaming dataset into
        memory.  ``GRPOTrainer`` doesn't support streaming datasets.
    skip_buffer_size : int
        First advance the stream past this many **valid** examples (same
        dedupe / length rules as buffering). Then collect up to
        ``buffer_size`` examples for training. The two counts are independent:
        e.g. ``skip_buffer_size=5000`` and ``buffer_size=10000`` yields up to
        10000 buffered rows taken from positions 5000 onward (ignored when 0).
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
    use_ddp : bool
        If *True*, use Distributed Data Parallel (DDP) mode.  See
        :func:`train_with_sft` for details.
    lora_rank, lora_alpha, lora_target_modules
        LoRA hyper-parameters for new adapters / policy adapter
        (:func:`model_utils.get_lora_config`).  When loading from
        ``resume_from``, existing adapter shapes still apply where relevant.
    """
    # -- Validate resume flags -----------------------------------------
    resume_checkpoint = None
    if resume_training_state:
        if not resume_from:
            raise ValueError(
                "--resume_training_state requires --resume_from to point "
                "to a Trainer checkpoint directory."
            )
        validate_training_state_checkpoint(resume_from)
        resume_checkpoint = resume_from
        print(f"[Resume] Will restore full training state from: {resume_from}")

    # Lazy-import GRPO trainers and GRPOConfig to avoid pulling in
    # trl.GRPOTrainer (and its vllm dependency) during SFT-only runs.
    from trl import GRPOConfig
    from tool_calling_grpo_trainer import ToolCallingGRPOTrainer
    from dual_adapter_grpo_trainer import (
        DualAdapterGRPOTrainer,
        is_dual_adapter_checkpoint,
        load_dual_adapter_checkpoint,
    )

    # Determine device_map: per-GPU for DDP, "auto" otherwise
    device_map = _get_device_map(use_ddp)
    if use_ddp:
        print(f"[DDP] Enabled – loading model with device_map={device_map}")

    # Define LoRA config — needed for both fresh start and resume
    lora_config = get_lora_config(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
    )

    # Load model and tokenizer
    if resume_from:
        # GRPO → GRPO resume: the checkpoint already contains BOTH the frozen
        # SFT reference adapter AND the trained policy adapter.  Detect that
        # layout and reload both so the DualAdapterGRPOTrainer can pick up
        # where it left off without re-initialising the policy from the
        # reference (which would discard all GRPO progress).
        is_grpo_ckpt = use_dual_adapter and is_dual_adapter_checkpoint(resume_from)
        if is_grpo_ckpt:
            print(f"Resuming GRPO training from dual-adapter checkpoint: {resume_from}")
            print("  - Reference (frozen SFT) adapter: loaded from reference/")
            print("  - Policy (trainable) adapter: loaded from policy/")
            model, tokenizer = load_dual_adapter_checkpoint(resume_from, device_map=device_map)
            peft_config_for_trainer = None
            # Signal to DualAdapterGRPOTrainer: adapters are already set up.
            policy_lora_config = None
        else:
            # Transition SFT → GRPO: only a single (SFT) adapter is on disk.
            print(f"Resuming GRPO training from SFT checkpoint: {resume_from}")
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
    if resume_training_state and skip_buffer_size:
        print(
            "[GRPO] Warning: --resume_training_state restores trainer step/checkpoint state; "
            "non-zero --skip_buffer_size also skips a dataset prefix. Combine only if intentional."
        )
    if skip_buffer_size:
        print(
            f"[GRPO] Stream: skip first {skip_buffer_size} valid example(s), "
            f"then buffer up to {buffer_size} (independent quotas; skip does not shrink the buffer cap)."
        )
    train_dataset = buffer_streaming_dataset(
        formatted_data,
        buffer_size=buffer_size,
        shuffle=True,
        seed=42,
        tokenizer=tokenizer,
        max_prompt_length=max_prompt_length,
        skip_buffer_size=skip_buffer_size,
    )

    # =========================================================================
    # REWARD FUNCTION SETUP
    # =========================================================================
    print("Initializing reward function factory...")
    reward_factory = RewardFunctionFactory(config_path='sim_config.json')
    reward_fn = reward_factory.create_reward_function(return_component_dicts=True)
    
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
        learning_rate=2e-5,
        max_steps=max_steps,
        logging_steps=5,
        save_steps=10,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=report_to,
        lr_scheduler_type="cosine",
        lr_scheduler_kwargs={'num_cycles': 0.4},
        warmup_steps=10,
        # GRPO-specific options
        loss_type="dapo", # "grpo", "dr_grpo", "dapo", "bnpo", "cispo", default is "dapo"
        num_generations=max(num_generations, 2),
        steps_per_generation=steps_per_generation,
        max_completion_length=max_completion_length,
        use_vllm=use_vllm,
        vllm_mode=vllm_mode,
        vllm_server_base_url=vllm_server_url,
        importance_sampling_level="sequence", # "token" or "sequence" : sequence provides more stable training and better alignment with sequence-level rewards
        # Whether to compute importance sampling ratios at the `"token"` or `"sequence"` level.
        # `"token"`: keeps raw per-token log-probability ratios. 
        # `"sequence"`: averages them across valid tokens into a single ratio per sequence — generally more stable (see GSPO paper).
        scale_rewards=False, 
        # - `True` or `"group"` (default): rewards are scaled by the standard deviation within each group, ensuring unit variance within a group.
        # - `"batch"`: rewards are scaled by the standard deviation across the entire batch
        # - `False` or `"none"`: no scaling is applied. The [Dr. GRPO paper] recommends not scaling rewards, as scaling by the standard deviation introduces a question-level difficulty bias.
        # Logging options
        log_completions=True,
        num_completions_to_print=10,
        log_unique_prompts=True,
    )

    shared_callbacks = [
        ThroughputMetricsCallback(),
        ContextLengthHistogramCallback(pad_token_id=tokenizer.pad_token_id, tokenizer=tokenizer),
        TrainingStateCheckpointCallback(
            method="grpo",
            launch_skip_buffer_size=skip_buffer_size,
            buffer_size=buffer_size,
        ),
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
            vllm_max_model_len=max_model_len if use_vllm else None,
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

    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(output_dir)


# =====================================================================
# CLI helpers (environment variable defaults)
# =====================================================================

def _env(key: str, default: str =None):
    return os.environ.get(key, default)


# Keys uploaded to wandb: shared by SFT and GRPO, plus GRPO-only tuning knobs.
_WANDB_CONFIG_KEYS_SFT = frozenset({
    "model_name",
    "dataset_path",
    "output_dir",
    "resume_from",
    "resume_training_state",
    "skip_buffer_size",
    "auto_skip_from_resume",
    "skip_batch_size",
    "skip_num_workers",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "max_steps",
    "report_to",
    "max_model_len",
    "max_prompt_length",
    "use_ddp",
    "lora_rank",
    "lora_alpha",
    "lora_target_modules",
    "assistant_only_loss",
})
_WANDB_CONFIG_KEYS_GRPO_ONLY = frozenset({
    "buffer_size",
    "num_generations",
    "steps_per_generation",
    "max_completion_length",
    "use_dual_adapter",
    "use_vllm",
    "vllm_mode",
    "vllm_server_url",
})


def _wandb_run_config(method: str, args_dict: dict, git_info: dict) -> dict:
    """Subset of CLI args that actually tune the active training path (SFT vs GRPO)."""
    keys = _WANDB_CONFIG_KEYS_SFT | (
        _WANDB_CONFIG_KEYS_GRPO_ONLY if method == "grpo" else frozenset()
    )
    body = {k: args_dict[k] for k in sorted(keys) if k in args_dict}
    return {"method": method, **body, **git_info}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine‑tune an LLM with quantised LoRA")
    parser.add_argument("--model_name", type=str, default=_env('MODEL', None),
                        help="Base model name on the HF hub (not required if --resume_from is provided)")
    parser.add_argument("--dataset", type=str, default=_env('TRAIN_DATASET', None),
                        help="Path to JSONL dataset with training records")
    parser.add_argument("--output_dir", type=str, default=_env('OUTPUT_DIR', "./finetuned_model"),
                        help="Output directory")
    parser.add_argument("--method", type=str, default=_env('METHOD', "sft"), choices=["sft", "grpo"],
                        help="Training method: sft or grpo")
    parser.add_argument("--resume_from", type=str, default=_env('RESUME_FROM', None),
                        help="Path to a saved adapter directory to resume training from. "
                             "Loads LoRA adapter weights only (training starts from step 0 "
                             "with a fresh optimizer). Combine with --resume_training_state "
                             "to also restore optimizer, LR schedule, step counter, and RNG "
                             "seeds for full crash recovery.")
    parser.add_argument("--resume_training_state", action="store_true",
                        default=_env('RESUME_TRAINING_STATE', '0').lower() in ('1', 'true', 'yes'),
                        help="Restore the full training state (optimizer weights, LR schedule, "
                             "step/epoch counters, RNG seeds) from the checkpoint specified by "
                             "--resume_from. Without this flag, --resume_from only loads adapter "
                             "weights. Requires --resume_from to point to a Trainer checkpoint "
                             "directory (e.g., output_dir/checkpoint-50/) containing "
                             "trainer_state.json and optimizer state files.")
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=_env('BUFFER_SIZE', 10000),
        help="GRPO: max examples to collect **after** any --skip_buffer_size offset (skip does not "
             "reduce this count). Env: BUFFER_SIZE",
    )
    parser.add_argument(
        "--skip_buffer_size",
        type=int,
        default=_env("SKIP_BUFFER_SIZE", 0),
        help="Skip this many training examples from the start of the stream (after "
             "formatting). SFT: skips formatted rows in stream order. GRPO: skips "
             "valid buffered rows (dedupe/length filters) before collecting "
             "--buffer_size rows — the two GRPO limits are independent. "
             "Env: SKIP_BUFFER_SIZE (default: 0)",
    )
    parser.add_argument(
        "--auto_skip_from_resume",
        action=argparse.BooleanOptionalAction,
        default=_env("AUTO_SKIP_FROM_RESUME", "1").lower() in ("1", "true", "yes"),
        help="Default ON. When --resume_from is set and --skip_buffer_size is left "
             "at 0, auto-load it from the checkpoint's training_state_summary.json: "
             "(a) without --resume_training_state -> use cumulative_skip_buffer_size "
             "(fresh run on new data, advances past everything the previous run consumed); "
             "(b) with --resume_training_state -> use launch_skip_buffer_size (crash "
             "recovery, matches the stream offset the optimizer was trained against). "
             "An explicit non-zero --skip_buffer_size always wins. Disable with "
             "--no-auto_skip_from_resume. Env: AUTO_SKIP_FROM_RESUME (1/0).",
    )
    parser.add_argument(
        "--skip_batch_size",
        type=int,
        default=_env("SKIP_BATCH_SIZE", 256),
        help="SFT streaming: raw rows per batch for fast skip/filter/format. "
             "Env: SKIP_BATCH_SIZE (default: 128)",
    )
    parser.add_argument(
        "--skip_num_workers",
        type=int,
        default=_env("SKIP_NUM_WORKERS", None),
        help="SFT streaming: parallel workers for skip/filter batches "
             "(default: min(16, CPU count)). Env: SKIP_NUM_WORKERS",
    )
    parser.add_argument("--max_steps", type=int, default=_env('MAX_STEPS', 1),
                        help="Maximum number of training steps")
    parser.add_argument("--per_device_train_batch_size", type=int, default=_env('PER_DEVICE_TRAIN_BATCH_SIZE', 1),
                        help="Per device train batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=_env('GRADIENT_ACCUMULATION_STEPS', 2),
                        help="Gradient accumulation steps")
    parser.add_argument("--report_to", type=str, default=_env('REPORT_TO', "wandb"),
                        help="Report to: wandb or none")
    parser.add_argument("--use_vllm", action="store_true", default=_env('USE_VLLM', '0').lower() in ('1', 'true', 'yes'),
                        help="Use VLLM for faster inference. Ideal for GRPO training. It can be used for SFT training as well. Requires: pip install vllm")
    parser.add_argument("--vllm_server_url", type=str, default=_env('VLLM_SERVER_URL', None),
                        help="URL of a running vLLM server for SFT eval generation "
                             "(e.g. http://localhost:8000). Used when --use_vllm is set.")
    parser.add_argument("--vllm_mode", type=str, default=_env('VLLM_MODE', "server"), choices=["colocate", "server"],
                        help="VLLM mode")
    parser.add_argument("--num_generations", type=int, default=_env('NUM_GENERATIONS', 8),
                        help="Number of generations per prompt to sample.")
    parser.add_argument("--steps_per_generation", type=int, default=_env('STEPS_PER_GENERATION', 2),
                        help="Steps per generation")
    parser.add_argument("--max_model_len", type=int, default=_env('MAX_MODEL_LEN', 8192),
                        help="Maximum model length")
    parser.add_argument("--max_completion_length", type=int, default=_env('MAX_COMPLETION_LENGTH', 4096),
                        help="Maximum completion length")
    parser.add_argument("--max_prompt_length", type=int, default=_env('MAX_PROMPT_LENGTH', 4096),
                        help="Max system+user prompt tokens (chat template). SFT messages: "
                             "drop longer examples. GRPO: drop during buffering. "
                             "Ignored for SFT legacy text format.")
    parser.add_argument("--use_dual_adapter", action="store_true", default=_env('USE_DUAL_ADAPTER', '1').lower() in ('1', 'true', 'yes'),
                        help="Use dual-adapter mode (keeps SFT adapter isolated, avoids merge). Default: True")
    parser.add_argument("--use_ddp", action="store_true", default=_env('USE_DDP', '0').lower() in ('1', 'true', 'yes'),
                        help="Use Distributed Data Parallelization. It's recommended for SFT training.")
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=int(_env("LORA_RANK", "8")),
        help="LoRA rank (env: LORA_RANK, default: 8)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=int(_env("LORA_ALPHA", "16")),
        help="LoRA alpha (env: LORA_ALPHA, default: 16)",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default=_env("LORA_TARGET_MODULES", "") or "",
        help="Comma-separated module names (e.g. q_proj,v_proj). Empty = default attention+MLP set. Env: LORA_TARGET_MODULES",
    )
    parser.add_argument(
        "--assistant_only_loss",
        action=argparse.BooleanOptionalAction,
        default=_env("ASSISTANT_ONLY_LOSS", "1").lower() in ("1", "true", "yes"),
        help="SFT only. If set (default), compute loss only on assistant tokens "
             "(content + tool-call JSON); system / user / tool-response tokens are "
             "masked out. The dataset is emitted as `messages` + `tools` and the "
             "Qwen2/2.5/3 chat template is patched in-place to add {%% generation %%} "
             "markers required by TRL. Use --no-assistant_only_loss to revert to the "
             "legacy behavior (pre-rendered `text` field, language-modeling loss over "
             "every non-pad token). Env: ASSISTANT_ONLY_LOSS (1/0).",
    )
    args = parser.parse_args()

    _ltm = (args.lora_target_modules or "").strip()
    args.lora_target_modules = (
        [m.strip() for m in _ltm.split(",") if m.strip()] if _ltm else None
    )

    # Validate integer arguments
    if args.skip_num_workers is not None:
        args.skip_num_workers = int(args.skip_num_workers)

    for field in ('buffer_size', 'skip_buffer_size', 'skip_batch_size',
                  'per_device_train_batch_size',
                  'gradient_accumulation_steps', 'max_steps',
                  'num_generations', 'steps_per_generation',
                  'max_model_len', 'max_completion_length', 'max_prompt_length',
                  'lora_rank', 'lora_alpha'):
        try:
            setattr(args, field, int(getattr(args, field)))
        except ValueError:
            parser.error(f"Invalid {field}: {getattr(args, field)}")

    if args.skip_buffer_size < 0:
        parser.error("skip_buffer_size must be >= 0")

    # ------------------------------------------------------------------
    # Auto-derive --skip_buffer_size from a resumed checkpoint.
    #
    # Triggered when:
    #   * --auto_skip_from_resume is on (default), AND
    #   * --resume_from points at a checkpoint, AND
    #   * --skip_buffer_size was left at the default 0 (any non-zero
    #     value is treated as an explicit override and respected).
    #
    # Which field we read depends on --resume_training_state:
    #   * off (fresh run on new data) -> cumulative_skip_buffer_size
    #     (advance past everything consumed by the previous run).
    #   * on  (crash recovery)        -> launch_skip_buffer_size
    #     (same stream offset the checkpoint optimizer was trained on,
    #     so HF Trainer's built-in step-resume skip lands on the right rows).
    # ------------------------------------------------------------------
    if (
        args.auto_skip_from_resume
        and args.resume_from
        and args.skip_buffer_size == 0
    ):
        if args.resume_training_state:
            derive_kind = "launch_skip_buffer_size"
            derived = read_launch_skip_from_checkpoint(args.resume_from)
            mode_label = "crash recovery, matches checkpoint optimizer state"
        else:
            derive_kind = "cumulative_skip_buffer_size"
            derived = read_cumulative_skip_from_checkpoint(args.resume_from)
            mode_label = "fresh run on new data, advances past consumed rows"

        if derived is not None:
            print(
                f"[auto_skip_from_resume] Derived --skip_buffer_size {derived} "
                f"({derive_kind}; {mode_label}) from "
                f"{args.resume_from}/training_state_summary.json. "
                f"Override with --skip_buffer_size <N> or disable with "
                f"--no-auto_skip_from_resume."
            )
            args.skip_buffer_size = derived
        else:
            print(
                f"[auto_skip_from_resume] No {derive_kind} found in "
                f"{args.resume_from}/training_state_summary.json — keeping "
                f"--skip_buffer_size 0. (Expected for checkpoints saved before "
                f"this feature was added; pass --skip_buffer_size <N> manually "
                f"or disable with --no-auto_skip_from_resume.)"
            )
    elif (
        args.auto_skip_from_resume
        and args.resume_from
        and args.skip_buffer_size != 0
    ):
        print(
            f"[auto_skip_from_resume] Explicit --skip_buffer_size "
            f"{args.skip_buffer_size} provided; not auto-deriving."
        )

    # Convert args to dict and fix parameter naming
    args_dict = vars(args)
    args_dict['dataset_path'] = args_dict.pop('dataset')
    method = args_dict.pop('method')
    # auto_skip_from_resume is a launch-time concern; train_with_* absorb it via **kwargs.

    # name wandb run according to the output directory and lora rank and alpha
    wandb_name = f"r{args_dict.get('lora_rank', 8)}-alpha{args_dict.get('lora_alpha', 16)}" 
    if method in args_dict.get('output_dir'):
        wandb_name = f"{args_dict.get('output_dir', 'run')}-{wandb_name}" 
    else:
        wandb_name = f"{method}-{args_dict.get('output_dir', 'run')}-{wandb_name}" 

    # 1. Wandb config: only hyperparameters that apply to the chosen method (plus git metadata)
    run_config = _wandb_run_config(method, args_dict, get_git_info())

    # 2. Initialize wandb explicitly to capture everything from this point forward
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", f"{method}-training"),
        config=run_config,
        name=wandb_name,
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
