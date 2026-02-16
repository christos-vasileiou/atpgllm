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
import json
import sys
import time
from pathlib import Path

# Add the parent directory of atpgllm to sys.path to allow importing from data_preprocessing
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'data_preprocessing'))

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Literal
import regex as re
import torch
import os
from datasets import load_dataset, Dataset, IterableDataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers.utils import get_json_schema
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig, GRPOTrainer, GRPOConfig
from transformers import StoppingCriteria, StoppingCriteriaList, TrainerCallback
from tool_calling_grpo_trainer import ToolCallingGRPOTrainer
from dual_adapter_grpo_trainer import DualAdapterGRPOTrainer
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend – safe for headless servers
import matplotlib.pyplot as plt
import pandas as pd
from tools import FAULT_SIMULATION_TOOL, TOOLS, fault_simulation_tool
from reward_function_factory import RewardFunctionFactory
from callbacks import (
    ThroughputMetricsCallback,
    ContextLengthHistogramCallback,
    SFTStoppingCallback,
)

# =====================================================================
# UNSLOTH SUPPORT (conditional – graceful fallback when not installed)
# =====================================================================
_UNSLOTH_AVAILABLE = False
try:
    from unsloth import FastLanguageModel as _FastLM
    _UNSLOTH_AVAILABLE = True
except ImportError:
    _FastLM = None


def _require_unsloth() -> None:
    """Raise a clear error when unsloth is requested but not installed."""
    if not _UNSLOTH_AVAILABLE:
        raise ImportError(
            "unsloth is not installed.  Install it with:\n"
            "    pip install unsloth\n"
            "Or remove --use_unsloth / set USE_UNSLOTH=0 to use standard training."
        )

class TrainingMode:
    """Training mode enumeration."""
    SFT = "sft"
    GRPO = "grpo"


@dataclass
class ConversationExample:
    """Internal container for a single training example."""
    
    messages: List[Dict[str, str]]  # list of {"role": ..., "content": ...}
    
    @staticmethod
    def from_record(record: Dict[str, Any], use_tools: bool = False) -> "ConversationExample":
        """
        Create a ConversationExample from a dataset record.  This method
        combines the system, user, reasoning and answer fields into a
        structured chat conversation.  It uses placeholders contained in
        the record to fill the reasoning template.
        
        Parameters
        ----------
        record: Dict[str, Any]
            A dictionary from the dataset with keys such as
            `system_content`, `user_content`, `reasoning_content`,
            `answer_content`, `fault`, `netlist`, `input_vector`, etc.
        
        Returns
        -------
        ConversationExample
        
        Template Placeholder Derivation
        -------------------------------
        The reasoning templates use the following placeholders that must be
        derived from the stored dataset fields:
        
        1. module_name         <- direct from 'module_name'
        2. fault_net           <- parsed from 'fault' (e.g., "sa0 net_name" -> "net_name")
        3. fault_model_long    <- parsed from 'fault' (e.g., "sa0" -> "stuck-at-0")
        4. fault_model_short   <- parsed from 'fault' (e.g., "sa0" -> "SA0")
        5. excitation_value    <- parsed from 'fault' (sa0 needs 1 to excite, sa1 needs 0)
        6. propagation_gates   <- from 'fault_propagation_gates'
        7. primary_observation_nets <- keys from 'expected_output' JSON
        8. backtrack_gates     <- from 'backtrack_gates'
        9. primary_controlling_nets <- from 'backtrack_nets'
        10. expected_output    <- formatted from 'expected_output' JSON
        11. input_vector       <- formatted from 'input_vector' JSON
        12. detected_faults    <- from 'detected_faults'
        13. non_controlling_nets <- from 'backtrack_nets' (sensitizing inputs)
        14. snapshot           <- from 'snapshot'
        """
        
        # =================================================================
        # Parse fault string to derive fault-related placeholders
        # Fault format: "sa0 net_name" or "sa1 net_name"
        # =================================================================
        fault = record.get('fault', '')
        if fault:
            fault_match = re.match(r'(sa)(\d)\s+(.+)', fault, re.IGNORECASE)
            if fault_match:
                fault_value = int(fault_match.group(2))  # 0 or 1
                fault_net = fault_match.group(3).strip()  # net name
                
                record['fault_net'] = fault_net
                record['fault_model_short'] = f"SA{fault_value}"  # "SA0" or "SA1"
                record['fault_model_long'] = f"stuck-at-{fault_value}"  # "stuck-at-0" or "stuck-at-1"
                # To excite a SA0 fault, drive the net to 1 (opposite of stuck value)
                # To excite a SA1 fault, drive the net to 0 (opposite of stuck value)
                record['excitation_value'] = str(1 - fault_value)
        
        # =================================================================
        # Parse JSON fields and derive additional placeholders
        # =================================================================
        # Parse expected_output to get primary_observation_nets (output net names)
        expected_output_dict = json.loads(record.get('expected_output', '{}'))
        record['primary_observation_nets'] = ', '.join(expected_output_dict.keys())
        
        # Format vectors as "net: value, net: value, ..."
        input_vector_dict = json.loads(record.get('input_vector', '{}'))
        record['input_vector'] = ", ".join(f"{net}: {value}" for net, value in input_vector_dict.items())
        record['expected_output'] = ", ".join(f"{net}: {value}" for net, value in expected_output_dict.items())
        
        # =================================================================
        # Map stored field names to template placeholder names
        # =================================================================
        # propagation_gates: gates whose outputs are on the fault propagation path
        record['propagation_gates'] = record.get('fault_propagation_gates', '')
        
        # primary_controlling_nets & non_controlling_nets: sensitizing inputs
        # Both map to backtrack_nets (the inputs that control fault propagation)
        record['primary_controlling_nets'] = ', '.join(set(input_vector_dict.keys()) & set(record.get('backtrack_nets', '').split(', ')))
        record['non_controlling_nets'] = record.get('backtrack_nets', '')
        
        # Extract the raw fields
        system_content = record.get("system_content", "").format(**record)
        user_content = record.get("user_content", "").format(**record)
        reasoning_content = record.get("reasoning_content", "").format(**record)
        answer_content = record.get("answer_content", "").format(**record)
        snapshot = record.get("snapshot", "")
        
        # Compose the messages sequence.  We include the chain‑of‑thought in a
        # separate assistant message tagged as "assistant" reasoning.  During
        # RL training we can evaluate the reasoning chain separately from the
        # final answer.
        messages: List[Dict[str, str]] = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        if user_content:
            messages.append({"role": "user", "content": user_content})
        if reasoning_content:
            if use_tools:
                arguments = dict.fromkeys(FAULT_SIMULATION_TOOL['function']['parameters']['properties'].keys())
                arguments['input_vector'] = input_vector_dict
                arguments['output_vector'] = expected_output_dict
                arguments['fault'] = fault
                arguments['netlist'] = record.get('netlist', 'netlist is unknown')
                
                tool_call_json = {
                    "name": FAULT_SIMULATION_TOOL['function']['name'],
                    "arguments": arguments,
                }
                
                # For SFT we include the reasoning verbatim.  At inference time you
                # may instruct the model to produce its chain of thought using
                # special tags (e.g. <think>...</think>) or tool calls.
                tool_call_content = (
                    "<think>" + reasoning_content + "</think>\n\n" +
                    "I have to verify if the fault is detected by the input and output vectors. I need to call the fault simulation tool.\n\n"
                )
                messages.append({"role": "assistant", "content": tool_call_content, "tool_calls": [{"type": "function", "function": tool_call_json}]})
                messages.append({"role": "tool", "name": FAULT_SIMULATION_TOOL['function']['name'], "content": snapshot})
            else:
                messages.append({
                    "role": "assistant",
                    "content": "<think>" + reasoning_content + "\n" +
                                "I'll perform a fault simulation by myself to verify if the fault is detected by the input and output vectors\n\n" +
                                snapshot +
                                "</think>\n\n"
                })
        if answer_content:
            messages.append({"role": "assistant", "content": "I can see the difference between Good/Bad Machine. The fault has been sensitized. To sum up, the vectors are:\n" + answer_content})
        return ConversationExample(messages=messages)


def buffer_streaming_dataset(
    streaming_dataset: IterableDataset, 
    buffer_size: int = 10000,
    shuffle: bool = True,
    seed: int = 42,
    unique_by: Optional[Literal["text", "prompt", "module_name", "netlist"]] = None,
) -> Dataset:
    """
    Convert a streaming IterableDataset to a regular Dataset by buffering examples.
    
    This is necessary for trainers that don't support streaming datasets (like GRPOTrainer).
    The buffer_size controls how many examples are loaded into memory.
    
    Parameters
    ----------
    streaming_dataset: IterableDataset
        The streaming dataset to buffer.
    buffer_size: int
        Maximum number of examples to load into memory. Set to -1 to load all examples
        (only use if you know the dataset fits in memory).
    shuffle: bool
        Whether to shuffle the buffered dataset. Recommended for training.
    seed: int
        Random seed for shuffling.
    unique_by: str, optional
        The field to use for uniqueness. Can be "text", "prompt", "module_name", "netlist".

    Returns
    -------
    Dataset
        A regular Dataset containing the buffered examples.
    """
    examples = []

    if unique_by is None:
        unique_by = 'text' if 'text' in next(iter(streaming_dataset)).keys() else None
    if unique_by is None:
        unique_by = 'prompt' if 'prompt' in next(iter(streaming_dataset)).keys() else None
    if unique_by not in next(iter(streaming_dataset)).keys():
        raise ValueError(f"Unique by field {unique_by} not found in dataset")

    desc = f"Buffering dataset (max {buffer_size} examples) searching unique samples by \'{unique_by}\'" if buffer_size > 0 else "Buffering entire dataset"

    unique_values = set()
    for example in tqdm(streaming_dataset, desc=desc, file=sys.stdout):
        if buffer_size > 0 and len(examples) >= buffer_size:
            break
        unique_value = example[unique_by]
        if unique_value in unique_values:
            continue
        unique_values.add(unique_value)
        examples.append(example)
    
    if not examples:
        raise ValueError("No examples were buffered from the streaming dataset. Check your dataset path.")
    
    dataset = Dataset.from_list(examples)
    
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    
    print(f"Buffered {len(dataset)} examples into memory")
    return dataset


def format_dataset_for_training(dataset, tokenizer: AutoTokenizer, training_mode: str):
    """
    Convert a dataset of records into a format accepted by SFTTrainer or GRPOTrainer.
    
    Each record is converted into a single chat conversation using
    ConversationExample.from_record().  The resulting conversation is
    rendered to a plain prompt using the tokenizer's chat template via
    `apply_chat_template()`.  The final output is a
    dictionary with keys `text` (the formatted prompt) and optional
    metadata fields.
    
    This function supports both regular Dataset and streaming IterableDataset.
    For streaming datasets, processing is done lazily on-the-fly using .map(),
    which is memory-efficient for large datasets that cannot fit in memory.
    
    Parameters
    ----------
    dataset
        The dataset containing raw records (can be Dataset or IterableDataset).
    tokenizer: AutoTokenizer
        The tokenizer whose chat template will be applied.
    training_mode: str
        The training mode to use (TrainingMode.SFT or TrainingMode.GRPO).
    
    Returns
    -------
    Dataset or IterableDataset
        A dataset ready for SFTTrainer or GRPOTrainer. For SFT, the field is `text`. 
        For GRPO, the field is `prompt`. Returns the same type as the input dataset.
    
    """
    is_streaming = isinstance(dataset, IterableDataset)
    use_tools = 'tools' in tokenizer.chat_template or 'tool' in tokenizer.chat_template
    
    if training_mode == TrainingMode.SFT:
        def format_fn(record):
            convo = ConversationExample.from_record(record, use_tools=use_tools)
            # Use the tokenizer's chat template to render the conversation.
            prompt = tokenizer.apply_chat_template(convo.messages, tokenize=False, tools=TOOLS if use_tools else None)
            return {"text": prompt}
        
        # Use .map() for lazy on-the-fly processing (works for both Dataset and IterableDataset)
        return dataset.map(format_fn, remove_columns=dataset.column_names if not is_streaming else None)
        
    elif training_mode == TrainingMode.GRPO:
        def format_fn(record):
            convo = ConversationExample.from_record(record, use_tools=use_tools)
            # Drop assistant messages (reasoning/answer) for RL training
            prompt_messages = [m for m in convo.messages if m["role"] != "assistant" and m["role"] != "tool"]
            # GRPOTrainer expects 'prompt' field with the formatted prompt
            prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, tools=TOOLS if use_tools else None, add_generation_prompt=True)
            return {"prompt": prompt, **record}
        
        # Use .map() for lazy on-the-fly processing
        # For GRPO we keep original columns since we need metadata for reward functions
        return dataset.map(format_fn)
    else:
        raise ValueError(f"Unknown training mode: {training_mode}")


def load_quantised_model(model_name: str) -> AutoModelForCausalLM:
    """
    Load a base causal language model in 4‑bit quantised form using
    `BitsAndBytesConfig`.  Gradient checkpointing is enabled to save
    memory.
    """
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    # Use explicit device mapping for compatibility with accelerate/trainer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    return model


def get_lora_config(use_unsloth: bool = False) -> LoraConfig:
    """
    Return the standard LoRA configuration used for both SFT and GRPO training.
    This ensures consistency when continuing from SFT to GRPO.
    
    When *use_unsloth* is ``True`` the dropout is forced to 0 because
    unsloth's fused kernels do not support non-zero LoRA dropout.
    """
    return LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0 if use_unsloth else 0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


def prepare_lora_model(base_model: AutoModelForCausalLM, lora_config: LoraConfig = None) -> AutoModelForCausalLM:
    """
    Inject LoRA adapters into the quantised base model.  We target all
    linear layers ("all-linear") to mirror the configuration used in the
    DeepSeek‑R1 experiments.  Feel free to adjust the
    rank (r) and alpha according to your compute budget.
    
    Parameters
    ----------
    base_model: AutoModelForCausalLM
        The base model to add LoRA adapters to.
    lora_config: LoraConfig, optional
        The LoRA configuration to use. If None, uses get_lora_config().
    """
    # Prepare the model for k-bit (4-bit/8-bit) training - this enables gradients
    # for the input embeddings and sets up the model for QLoRA training
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
    
    if lora_config is None:
        lora_config = get_lora_config()
    
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()  # Print trainable parameters info
    return peft_model


# =====================================================================
# UNSLOTH MODEL LOADING HELPERS
# =====================================================================

def load_unsloth_model(
    model_name: str,
    max_seq_length: int = 8192,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a base model and tokenizer using unsloth's ``FastLanguageModel``.

    This replaces both :func:`load_quantised_model` **and** the separate
    tokenizer loading step, providing up to 2× faster training and up to
    80 % less VRAM through hand-written Triton kernels.

    Parameters
    ----------
    model_name : str
        HuggingFace Hub identifier of the base model.
    max_seq_length : int
        Maximum sequence length the model will see (default 8192).

    Returns
    -------
    tuple[AutoModelForCausalLM, AutoTokenizer]
    """
    _require_unsloth()
    model, tokenizer = _FastLM.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,           # auto-detect best dtype for the GPU
        load_in_4bit=True,    # 4-bit QLoRA
    )
    # Ensure tokenizer has required special tokens
    if not tokenizer.eos_token:
        tokenizer.add_special_tokens({"eos_token": "</s>"})
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def prepare_unsloth_lora_model(
    model,
    lora_config: LoraConfig | None = None,
) -> AutoModelForCausalLM:
    """
    Apply LoRA adapters via unsloth's optimised ``get_peft_model``.

    Replaces :func:`prepare_lora_model` when running in unsloth mode.
    Uses unsloth's own gradient-checkpointing implementation
    (``"unsloth"`` mode) and forces ``lora_dropout=0`` as required by the
    fused Triton kernels.

    Parameters
    ----------
    model
        Base model returned by :func:`load_unsloth_model`.
    lora_config : LoraConfig, optional
        LoRA hyper-parameters.  If *None*, :func:`get_lora_config` is used
        with ``use_unsloth=True``.
    """
    _require_unsloth()
    if lora_config is None:
        lora_config = get_lora_config(use_unsloth=True)

    model = _FastLM.get_peft_model(
        model,
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        target_modules=list(lora_config.target_modules),
        lora_dropout=0,                             # required by unsloth
        bias="none",
        use_gradient_checkpointing="unsloth",       # long-context optimised
        random_state=42,
    )
    model.print_trainable_parameters()
    return model


def load_unsloth_model_from_adapter(
    adapter_path: str,
    max_seq_length: int = 8192,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a previously saved LoRA adapter on top of an unsloth-optimised
    base model.

    Reads ``adapter_config.json`` to determine the base model, loads it
    via :func:`load_unsloth_model`, then loads the LoRA weights through
    PEFT's ``PeftModel.from_pretrained``.

    .. note::
       The base model benefits from unsloth's fused Triton kernels.
       The LoRA matrices themselves are standard PEFT parameters, which
       is sufficient because the compute-heavy attention / MLP layers
       are already optimised.

    Parameters
    ----------
    adapter_path : str
        Directory containing ``adapter_config.json`` and the adapter
        weight files.
    max_seq_length : int
        Maximum sequence length (default 8192).

    Returns
    -------
    tuple[AutoModelForCausalLM, AutoTokenizer]
    """
    _require_unsloth()

    adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
    with open(adapter_config_path, 'r') as f:
        adapter_config = json.load(f)

    base_model_name = adapter_config.get("base_model_name_or_path")
    if not base_model_name:
        raise ValueError(
            f"Could not find base_model_name_or_path in {adapter_config_path}"
        )

    print(f"[Unsloth] Loading base model: {base_model_name}")
    model, tokenizer = _FastLM.from_pretrained(
        model_name=base_model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    # Ensure tokenizer has required special tokens
    if not tokenizer.eos_token:
        tokenizer.add_special_tokens({"eos_token": "</s>"})
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the LoRA adapter on top of the unsloth-optimised base
    print(f"[Unsloth] Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    model.print_trainable_parameters()

    return model, tokenizer


def smart_sync_model_config(model, tokenizer):
    """
    Synchronizes the model's configuration (and generation config) with the 
    tokenizer's special token IDs.
    
    This prevents the 'The tokenizer has new PAD/BOS/EOS tokens...' warning 
    and ensures the model uses the correct tokens during training/generation.
    
    Args:
        model: The AutoModelForCausalLM (or similar) object.
        tokenizer: The AutoTokenizer object.
        
    Returns:
        model: The updated model with synced configuration.
    """
    # The keys to check synchronization for
    token_keys = ["pad_token_id", "bos_token_id", "eos_token_id"]
    
    for key in token_keys:
        # 1. Get the token ID from the tokenizer (The Source of Truth)
        tokenizer_token_id = getattr(tokenizer, key, None)
        
        # 2. Get the current token ID from the model config
        model_config_id = getattr(model.config, key, None)
        
        # 3. SYNC CONDITION: 
        #    Tokenizer HAS a value AND it is DIFFERENT from the Model's value
        if tokenizer_token_id is not None and model_config_id != tokenizer_token_id:
            
            # Update Model Config
            setattr(model.config, key, tokenizer_token_id)
            print(f"Synced {key}: Model config updated to {tokenizer_token_id}")
            
            # Update Generation Config (if the model has one)
            if hasattr(model, "generation_config") and model.generation_config is not None:
                setattr(model.generation_config, key, tokenizer_token_id)

    return model


def load_model_from_adapter(adapter_path: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a previously fine-tuned model from a saved adapter directory.
    
    This function reads the adapter configuration to determine the base model,
    loads the quantised base model, and then loads the LoRA adapters on top.
    The tokenizer is also loaded from the adapter directory if available,
    otherwise from the base model.
    
    Parameters
    ----------
    adapter_path: str
        Path to the directory containing the saved adapter files
        (adapter_config.json, adapter_model.safetensors, tokenizer files, etc.)
    
    Returns
    -------
    tuple[AutoModelForCausalLM, AutoTokenizer]
        The model with loaded LoRA adapters and the tokenizer.
    """
    import os
    
    # Read adapter config to get base model name
    adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
    with open(adapter_config_path, 'r') as f:
        adapter_config = json.load(f)
    
    base_model_name = adapter_config.get("base_model_name_or_path")
    if not base_model_name:
        raise ValueError(f"Could not find base_model_name_or_path in {adapter_config_path}")
    
    print(f"Loading base model: {base_model_name}")
    
    # Load the tokenizer from the adapter directory (contains saved tokenizer files)
    # Fall back to base model if tokenizer files are not in adapter directory
    tokenizer_config_path = os.path.join(adapter_path, "tokenizer_config.json")
    if os.path.exists(tokenizer_config_path):
        print(f"Loading tokenizer from adapter directory: {adapter_path}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    else:
        print(f"Loading tokenizer from base model: {base_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    
    # Ensure tokenizer has EOS and PAD tokens
    if not tokenizer.eos_token:
        tokenizer.add_special_tokens({"eos_token": "</s>"})
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load the quantised base model
    base_model = load_quantised_model(base_model_name)
    
    # Prepare for k-bit training before loading adapter
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
    
    # Load the LoRA adapter weights
    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)
    model.print_trainable_parameters()
    
    return model, tokenizer


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
    eval_buffer_size: int = 30,
    use_unsloth: bool = False,
    **kwargs
) -> None:
    """
    Perform supervised fine‑tuning (SFT) on a quantised LLM using LoRA
    adapters.  The dataset must be an IterableDataset or JSONL file
    containing the necessary fields.  The resulting LoRA weights will be
    saved in `output_dir`.
    
    An ``SFTStoppingCallback`` is automatically attached.  It monitors:
    
    1. **Format compliance** – can the reward function parse ≥ 95 % of
       validation outputs?
    2. **Output diversity** – are 10 completions from a single prompt
       sufficiently distinct?
    
    When both criteria are met the trainer stops, and the checkpoint is
    ready to be used as the starting point for GRPO.
    
    Parameters
    ----------
    model_name: str
        HuggingFace Hub identifier of the base model (e.g.
        "Qwen/Qwen2.5-72B-Instruct").
    dataset_path: str
        Path to a local JSONL file or dataset identifier.  The file must
        contain JSON objects with the keys described in the module doc.
    output_dir: str
        Directory where the LoRA adapter weights and training artefacts will
        be saved.
    resume_from: str, optional
        Path to a previously saved adapter directory to resume training from.
        If provided, the model and tokenizer will be loaded from this directory
        instead of initializing new LoRA adapters.
    use_vllm: bool
        If True, validation generation in the stopping callback uses vLLM
        for faster batch inference.  Otherwise falls back to model.generate().
    eval_buffer_size: int
        Number of examples to buffer from the test split for the stopping
        callback validation (default 30).
    use_unsloth: bool
        If True, use unsloth's ``FastLanguageModel`` for model loading and
        LoRA injection.  Provides up to 2× faster training and up to 80 %
        less VRAM through fused Triton kernels.  Requires the ``unsloth``
        package to be installed.
    """
    # Validate unsloth availability early
    if use_unsloth:
        _require_unsloth()
        print("[Unsloth] Enabled – using FastLanguageModel for optimised training")

    # Load model and tokenizer - either from saved adapter or fresh
    if resume_from:
        print(f"Resuming training from: {resume_from}")
        if use_unsloth:
            model, tokenizer = load_unsloth_model_from_adapter(resume_from)
        else:
            model, tokenizer = load_model_from_adapter(resume_from)
    else:
        if use_unsloth:
            model, tokenizer = load_unsloth_model(model_name)
            model = prepare_unsloth_lora_model(model)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            # Make sure the tokenizer has an EOS token and PAD token
            if not tokenizer.eos_token:
                tokenizer.add_special_tokens({"eos_token": "</s>"})
            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token
            # Load quantised model and inject LoRA
            base_model = load_quantised_model(model_name)
            model = prepare_lora_model(base_model)
    
    # Synchronize the model's configuration with the tokenizer's special token IDs
    model = smart_sync_model_config(model, tokenizer)

    # Load dataset (non-streaming for compatibility with TRL's SFTTrainer)
    data = load_dataset(dataset_path, split="train", streaming=True)
    
    # Format dataset into chat prompts
    train_dataset = format_dataset_for_training(data, tokenizer, TrainingMode.SFT)
    # Define training arguments; we use gradient accumulation and paged
    # optimisers as recommended by QLoRA.  SFTConfig is the TRL-specific
    # config that extends TrainingArguments with SFT-specific options.
    # Determine actual max_steps: use provided value or default
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,  # Keep at 1 for stability with small datasets
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        max_steps=max_steps,
        logging_steps=5,
        save_steps=5,
        gradient_checkpointing=True,
        bf16=True,
        report_to=report_to,  # Disable wandb by default for testing
        # SFT-specific options (moved from SFTTrainer constructor)
        dataset_text_field="text",
        max_length=8192,
        # Enable token counting so ThroughputMetricsCallback can compute tokens/sec
        include_num_input_tokens_seen=True,
    )
    
    # Shared callbacks for logging throughput and context-length distribution
    shared_callbacks = [
            ThroughputMetricsCallback(),
            ContextLengthHistogramCallback(pad_token_id=tokenizer.pad_token_id),
            SFTStoppingCallback(
                tokenizer=tokenizer,
                dataset_path=dataset_path,
                eval_buffer_size=eval_buffer_size,
                tool_functions={"fault_simulation_tool": fault_simulation_tool},
                tools_schema=TOOLS,
                format_threshold=0.95,
                diversity_threshold=0.3,
                diversity_num_generations=10,
                use_vllm=use_vllm,
                max_new_tokens=4096,
                min_steps=5,
                patience=1,
                temperature=0.7,
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
    steps_per_generation: int = 1,
    use_dual_adapter: bool = True,
    use_unsloth: bool = False,
    **kwargs
) -> None:
    """
    Skeleton implementation of Group Relative Policy Optimisation (GRPO)
    training.  GRPO is an online reinforcement learning algorithm that
    requires reward functions to evaluate generated responses.  This
    function shows how to load a quantised model with LoRA and set up
    GRPOTrainer from the TRL library.  You must implement your own
    reward function(s) to suit your dataset and external verifier.
    
    Parameters
    ----------
    model_name: str
        Identifier of the base model.
    dataset_path: str
        Path to your JSONL dataset used for exploration. For GRPO, the dataset
        typically contains prompts only (no completions), since the model will
        generate its own outputs and receive a reward.
    output_dir: str
        Directory to store the resulting model.
    resume_from: str, optional
        Path to a previously saved adapter directory to resume training from.
        Required for dual-adapter mode.
    buffer_size: int, optional
        Maximum number of examples to buffer from the streaming dataset into memory.
        GRPOTrainer doesn't support streaming datasets. Default is 10000.
    use_dual_adapter: bool, optional
        If True (default), uses DualAdapterGRPOTrainer which keeps the SFT adapter
        isolated and avoids merge_and_unload. This is recommended for 4-bit
        quantized models where merging causes precision loss.
        
        **Dual-adapter mode Reference Model Handling:**
        When resuming from an SFT checkpoint with use_dual_adapter=True:
        1. SFT adapter is kept as "sft" (frozen, never merged)
        2. Policy adapter is added as "policy" (trainable, stacked on SFT)
        3. Reference computation uses SFT adapter only
        4. Policy computation uses SFT + policy adapters
        
        This ensures:
        - Reference model = SFT adapter (preserved at full LoRA precision)
        - Policy model = SFT + policy adapters (combined effect)
        - No quantization precision loss from merging
    use_unsloth: bool, optional
        If True, use unsloth's ``FastLanguageModel`` for model loading.
        The base model benefits from fused Triton kernels (2× faster,
        80 % less VRAM).  Adapter operations (dual-adapter, freeze,
        ``add_adapter``) remain standard PEFT calls.
    """
    # Validate unsloth availability early
    if use_unsloth:
        _require_unsloth()
        print("[Unsloth] Enabled – using FastLanguageModel for optimised GRPO training")

    # Define LoRA config - needed for both fresh start and resume scenarios
    lora_config = get_lora_config(use_unsloth=use_unsloth)
    
    # Load model and tokenizer - either from saved adapter or fresh
    if resume_from:
        print(f"Resuming GRPO training from SFT checkpoint: {resume_from}")
        if use_unsloth:
            model, tokenizer = load_unsloth_model_from_adapter(resume_from)
        else:
            model, tokenizer = load_model_from_adapter(resume_from)
        
        if use_dual_adapter:
            print("Using dual-adapter mode (DualAdapterGRPOTrainer)")
            print("  - SFT adapter will be kept isolated (no merging)")
            print("  - Reference model: SFT adapter only")
            print("  - Policy model: new policy adapter")
            # For dual-adapter mode, we'll pass policy_lora_config instead of peft_config
            peft_config_for_trainer = None
            policy_lora_config = lora_config
        else:
            print("Using standard mode (GRPOTrainer with merge_and_unload)")
            print("  - WARNING: SFT adapter will be merged into base weights")
            print("  - This may cause precision loss with 4-bit quantization")
            # Standard mode: pass peft_config to trigger merge_and_unload
            peft_config_for_trainer = lora_config
            policy_lora_config = None
    else:
        if use_unsloth:
            # Unsloth: load optimised base, apply LoRA now (not via trainer),
            # pass peft_config=None so GRPOTrainer won't merge_and_unload.
            model, tokenizer = load_unsloth_model(model_name)
            model = prepare_unsloth_lora_model(model, lora_config)
            peft_config_for_trainer = None  # LoRA already applied by unsloth
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if not tokenizer.eos_token:
                tokenizer.add_special_tokens({"eos_token": "</s>"})
            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token
            # Load base model (don't apply LoRA here - let GRPOTrainer do it)
            model = load_quantised_model(model_name)
            peft_config_for_trainer = lora_config
        
        # Fresh start: must use standard mode (no SFT adapter to preserve)
        if use_dual_adapter:
            print("Note: Dual-adapter mode requires --resume_from with an SFT checkpoint")
            print("  Falling back to standard mode for fresh start")
        policy_lora_config = None
        use_dual_adapter = False  # Force standard mode for fresh start
    
    # Synchronize the model's configuration with the tokenizer's special token IDs
    model = smart_sync_model_config(model, tokenizer)
    
    # Load prompts only – for RL we typically provide the system and user
    # content as the initial context and let the model generate reasoning
    # and answers.  We convert each record into a conversation and keep
    # only the "system" and "user" messages.
    #
    # Note: GRPOTrainer does NOT support streaming/iterable datasets.
    # We load as streaming to handle large datasets, then buffer a portion
    # into memory as a regular Dataset.
    print(f"Loading dataset from: {dataset_path}")
    data = load_dataset(dataset_path, split="train", streaming=True)
    
    # Format the streaming dataset (still streaming at this point)
    formatted_data = format_dataset_for_training(data, tokenizer, TrainingMode.GRPO)
    
    # Buffer the streaming dataset into a regular Dataset for GRPOTrainer
    # This loads buffer_size examples into memory
    print(f"Buffering streaming dataset (GRPOTrainer requires non-streaming Dataset)...")
    train_dataset = buffer_streaming_dataset(formatted_data, buffer_size=buffer_size, shuffle=True, seed=42)
    
    # =========================================================================
    # REWARD FUNCTION SETUP
    # =========================================================================
    # Create the reward function using the factory. This:
    # 1. Loads and parses gate functions from sim_config.json
    # 2. Provides netlist caching for efficient simulation
    # 3. Returns a reward function that runs actual fault simulation
    #
    # HOW THIS IMPLEMENTS "IMPLICIT TOOL-CALLING":
    # - Model generates structured output (INPUT_VECTOR, SNAPSHOT, etc.)
    # - Reward function PARSES the model's predicted input vector
    # - We RUN ACTUAL SIMULATION with the model's predicted inputs
    # - We COMPARE model's predicted snapshot with real simulation
    # - Rewards flow back to train the model to generate inputs that WORK
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
        per_device_train_batch_size=max(per_device_train_batch_size, 2),  # Must be divisible by num_generations
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=5e-5,
        max_steps=max_steps,
        logging_steps=5,
        save_steps=50,
        bf16=True,  
        report_to=report_to,
        # GRPO-specific options
        loss_type="dapo", # "grpo", "dr_grpo", "dapo", "bnpo", "cispo", default is "dapo"
        max_completion_length=tokenizer.model_max_length,
        num_generations=max(per_device_train_batch_size, 2),  # Number of completions to generate per prompt
        steps_per_generation=steps_per_generation, # Avoid OOM with large models
        use_vllm=use_vllm,
        vllm_mode=vllm_mode,
    )
    
    # Shared callbacks for logging throughput and context-length distribution
    shared_callbacks = [
        ThroughputMetricsCallback(),
        ContextLengthHistogramCallback(pad_token_id=tokenizer.pad_token_id),
    ]
    
    # Instantiate the appropriate trainer based on mode
    if use_dual_adapter:
        # Dual-adapter mode: uses DualAdapterGRPOTrainer which keeps SFT adapter isolated
        # Reference = SFT adapter, Policy = SFT + policy adapter
        trainer = DualAdapterGRPOTrainer(
            model=model,
            reward_funcs=[reward_fn],
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            policy_lora_config=policy_lora_config,  # For the new policy adapter
            tool_functions=[fault_simulation_tool],
            callbacks=shared_callbacks,
            # Note: don't pass peft_config - DualAdapterGRPOTrainer handles adapters manually
        )
    else:
        # Standard mode: uses merge_and_unload (may lose precision with 4-bit)
        # Reference = merged SFT, Policy = merged SFT + new LoRA
        trainer = ToolCallingGRPOTrainer(
        # trainer = GRPOTrainer(
            model=model,
            reward_funcs=[reward_fn],
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            peft_config=peft_config_for_trainer,  # Triggers merge_and_unload if model is PeftModel
            tool_functions=[fault_simulation_tool],
            callbacks=shared_callbacks,
        )
    
    trainer.train()
    trainer.save_model(output_dir)


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

def _steps_per_generation():
    return int(os.environ.get('STEPS_PER_GENERATION', 1))

def _use_unsloth():
    return os.environ.get('USE_UNSLOTH', '0').lower() in ('1', 'true', 'yes')

def main() -> None:
    parser = argparse.ArgumentParser(description="Fine‑tune a quantised LLM with LoRA")
    parser.add_argument("--model_name", type=str, default=_model_name(), help="Base model name on the HF hub (not required if --resume_from is provided)")
    parser.add_argument("--dataset", type=str, default=_train_dataset(), help="Path to JSONL dataset with training records")
    parser.add_argument("--output_dir", type=str, default=_output_dir(), help="Output directory")
    parser.add_argument("--method", type=str, default=_method(), choices=["sft", "grpo"], help="Training method: sft or grpo")
    parser.add_argument("--resume_from", type=str, default=_resume_from(), help="Path to a saved adapter directory to resume training from")
    parser.add_argument("--buffer_size", type=int, default=_buffer_size(), help="Number of examples to buffer from streaming dataset for GRPO")
    parser.add_argument("--max_steps", type=int, default=_max_steps(), help="Maximum number of training steps")
    parser.add_argument("--per_device_train_batch_size", type=int, default=_per_device_train_batch_size(), help="Per device train batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=_gradient_accumulation_steps(), help="Gradient accumulation steps")
    parser.add_argument("--report_to", type=str, default=_report_to(), help="Report to: wandb or none")
    parser.add_argument("--use_vllm", action="store_true", help="Use VLLM")
    parser.add_argument("--steps_per_generation", type=int, default=_steps_per_generation(), help="Steps per generation")
    parser.add_argument("--use_dual_adapter", action="store_true", default=True, 
                        help="Use dual-adapter mode (keeps SFT adapter isolated, avoids merge). Default: True")
    parser.add_argument("--no_dual_adapter", action="store_false", dest="use_dual_adapter",
                        help="Disable dual-adapter mode (uses standard merge_and_unload)")
    parser.add_argument("--use_unsloth", action="store_true", default=_use_unsloth(),
                        help="Use unsloth's FastLanguageModel for optimised training "
                             "(2x faster, 80%% less VRAM). Requires: pip install unsloth")
    args = parser.parse_args()
    
    try:
        args.buffer_size = int(args.buffer_size)
    except ValueError as e:
        parser.error(f"Invalid buffer size: {args.buffer_size}")

    try:
        args.per_device_train_batch_size = int(args.per_device_train_batch_size)
    except ValueError as e:
        parser.error(f"Invalid per device train batch size: {args.per_device_train_batch_size}")

    try:
        args.gradient_accumulation_steps = int(args.gradient_accumulation_steps)
    except ValueError as e:
        parser.error(f"Invalid gradient accumulation steps: {args.gradient_accumulation_steps}")
    
    try:
        args.max_steps = int(args.max_steps)
    except ValueError as e:
        parser.error(f"Invalid max steps: {args.max_steps}")
    
    try:
        args.steps_per_generation = int(args.steps_per_generation)
    except ValueError as e:
        parser.error(f"Invalid steps per generation: {args.steps_per_generation}")
    
    # Convert args to dict and fix parameter naming
    args_dict = vars(args)
    # Rename 'dataset' to 'dataset_path' (expected by training functions)
    args_dict['dataset_path'] = args_dict.pop('dataset')
    # Remove 'method' as it's not a training function parameter
    method = args_dict.pop('method')
    
    if method == "sft":
        train_with_sft(**args_dict)
    elif method == "grpo":
        train_with_grpo(**args_dict)
    else:
        raise ValueError(f"Unknown training method: {method}")


if __name__ == "__main__":
    main()
    