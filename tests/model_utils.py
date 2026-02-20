"""
model_utils.py
==============

Model loading, LoRA injection and configuration helpers for both
standard (``BitsAndBytes`` + ``PEFT``) and *unsloth* training modes.

Public API
----------
* :func:`load_quantised_model` — load a 4-bit quantised causal LM.
* :func:`get_lora_config` — standard LoRA hyper-parameters.
* :func:`prepare_lora_model` — inject LoRA adapters into a quantised model.
* :func:`load_unsloth_model` — load model + tokenizer via unsloth.
* :func:`prepare_unsloth_lora_model` — apply LoRA via unsloth's fused kernels.
* :func:`load_unsloth_model_from_adapter` — resume from a saved adapter
  with an unsloth-optimised base.
* :func:`smart_sync_model_config` — synchronise model/generation config
  with the tokenizer's special-token IDs.
* :func:`load_model_from_adapter` — standard adapter loading (no unsloth).
"""

from __future__ import annotations

import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel


# =====================================================================
# UNSLOTH SUPPORT (conditional – graceful fallback when not installed)
# =====================================================================
_UNSLOTH_AVAILABLE = False
try:
    # from unsloth import FastLanguageModel as _FastLM
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


# =====================================================================
# Standard (BitsAndBytes + PEFT) model loading
# =====================================================================

def load_quantised_model(model_name: str) -> AutoModelForCausalLM:
    """
    Load a base causal language model in 4-bit quantised form using
    ``BitsAndBytesConfig``.  Gradient checkpointing is enabled to save
    memory.
    """
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    return model


def get_lora_config(use_unsloth: bool = False) -> LoraConfig:
    """
    Return the standard LoRA configuration used for both SFT and GRPO
    training.  This ensures consistency when continuing from SFT to GRPO.

    When *use_unsloth* is ``True`` the dropout is forced to 0 because
    unsloth's fused kernels do not support non-zero LoRA dropout.
    """
    return LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0 if use_unsloth else 0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


def prepare_lora_model(
    base_model: AutoModelForCausalLM,
    lora_config: LoraConfig = None,
) -> AutoModelForCausalLM:
    """
    Inject LoRA adapters into a quantised base model.

    Parameters
    ----------
    base_model : AutoModelForCausalLM
        The base model to add LoRA adapters to.
    lora_config : LoraConfig, optional
        The LoRA configuration to use.  If *None*, :func:`get_lora_config`
        is called.
    """
    base_model = prepare_model_for_kbit_training(
        base_model, use_gradient_checkpointing=True,
    )

    if lora_config is None:
        lora_config = get_lora_config()

    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()
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
    (``"unsloth"`` mode) and forces ``lora_dropout=0`` as required by
    the fused Triton kernels.

    Parameters
    ----------
    model
        Base model returned by :func:`load_unsloth_model`.
    lora_config : LoraConfig, optional
        LoRA hyper-parameters.  If *None*, :func:`get_lora_config` is
        used with ``use_unsloth=True``.
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

    if not tokenizer.eos_token:
        tokenizer.add_special_tokens({"eos_token": "</s>"})
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[Unsloth] Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    model.print_trainable_parameters()

    return model, tokenizer


# =====================================================================
# Model / tokenizer synchronisation
# =====================================================================

def smart_sync_model_config(model, tokenizer):
    """
    Synchronise the model's configuration (and generation config) with
    the tokenizer's special-token IDs.

    This prevents the *"The tokenizer has new PAD/BOS/EOS tokens…"*
    warning and ensures the model uses the correct tokens during
    training/generation.

    Parameters
    ----------
    model
        The ``AutoModelForCausalLM`` (or similar) object.
    tokenizer
        The ``AutoTokenizer`` object.

    Returns
    -------
    model
        The updated model with synced configuration.
    """
    token_keys = ["pad_token_id", "bos_token_id", "eos_token_id"]

    for key in token_keys:
        tokenizer_token_id = getattr(tokenizer, key, None)
        model_config_id = getattr(model.config, key, None)

        if tokenizer_token_id is not None and model_config_id != tokenizer_token_id:
            setattr(model.config, key, tokenizer_token_id)
            print(f"Synced {key}: Model config updated to {tokenizer_token_id}")

            if hasattr(model, "generation_config") and model.generation_config is not None:
                setattr(model.generation_config, key, tokenizer_token_id)

    return model


# =====================================================================
# Standard adapter loading (no unsloth)
# =====================================================================

def load_model_from_adapter(
    adapter_path: str,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a previously fine-tuned model from a saved adapter directory.

    This function reads the adapter configuration to determine the base
    model, loads the quantised base model, and then loads the LoRA
    adapters on top.  The tokenizer is loaded from the base model.

    Parameters
    ----------
    adapter_path : str
        Path to the directory containing the saved adapter files
        (``adapter_config.json``, ``adapter_model.safetensors``,
        tokenizer files, etc.)

    Returns
    -------
    tuple[AutoModelForCausalLM, AutoTokenizer]
        The model with loaded LoRA adapters and the tokenizer.
    """
    adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
    with open(adapter_config_path, 'r') as f:
        adapter_config = json.load(f)

    base_model_name = adapter_config.get("base_model_name_or_path")
    if not base_model_name:
        raise ValueError(
            f"Could not find base_model_name_or_path in {adapter_config_path}"
        )

    print(f"Loading base model: {base_model_name}")

    tokenizer_config_path = os.path.join(adapter_path, "tokenizer_config.json")
    if os.path.exists(tokenizer_config_path):
        print(f"Loading tokenizer from adapter directory: {adapter_path}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    else:
        print(f"Loading tokenizer from base model: {base_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    if not tokenizer.eos_token:
        tokenizer.add_special_tokens({"eos_token": "</s>"})
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = load_quantised_model(base_model_name)
    base_model = prepare_model_for_kbit_training(
        base_model, use_gradient_checkpointing=True,
    )

    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)
    model.print_trainable_parameters()

    return model, tokenizer
