"""
evaluate_model.py
=================

Evaluation script for dual-adapter models (SFT + GRPO) using pass@k metrics.

This module evaluates a model trained through SFT → GRPO by:
1. Loading the dual-adapter model via HF PEFT or vLLM natively.
2. Loading the eval split of `chrivasileiou/asap7-language-of-test`
3. For batches of prompts, generating num_completions completions each, where
   each completion is one independent application of the chosen sampling
   strategy (with tool-calling support).
4. Executing tool calls (fault simulation) when the model requests them
5. Computing rewards via RewardFunctionFactory
6. Calculating pass@k metrics (pass@1, pass@5, pass@10, etc.)

The pass@k metric (from the Codex paper, Chen et al. 2021) estimates:
    pass@k = E[1 - C(n-c, k) / C(n, k)]
where n = num_completions (completions per problem), c = correct completions.
The estimator is unbiased only when the num_completions completions are i.i.d.,
which is why every strategy returns one completion per independent application.

Usage:
    python evaluate_model.py \
        --backend vllm \
        --tp_size 2 \
        --adapter ./finetuned_model/combined/policy/ \
        --dataset chrivasileiou/asap7-language-of-test \
        --num_completions 10 --k 1 5 10 \
        --temperature 0.6

Environment Variables:
    ADAPTER_CHECKPOINT: Path to SFT/GRPO adapter checkpoint
    WANDB_RUN_NAME: Optional override for the Weights & Biases run display name
    EVAL_DATASET: Dataset identifier (default: chrivasileiou/asap7-language-of-test)
    NUM_COMPLETIONS: Completions per prompt for pass@k (default: 10)
    MAX_EVAL_SAMPLES: Maximum number of eval samples (default: -1 for all)
    EVAL_PROMPT_BATCH_SIZE: Prompts per fused generation batch (default: 8)
    GENERATION_MICRO_BATCH_SIZE: HF generate micro-batch (default: 8)
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import math
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset, Dataset
from tqdm import tqdm

# Add the parent directory of atpgllm to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'data_preprocessing'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from transformers import AutoTokenizer, GenerationConfig
from peft import PeftModel

from dual_adapter_grpo_trainer import load_dual_adapter_model, REFERENCE_ADAPTER_NAME, POLICY_ADAPTER_NAME
from training_code import ConversationExample, TrainingMode
from dataset_utils import buffer_streaming_dataset
from reward_function_factory import RewardFunctionFactory
from tools import TOOLS, FAULT_SIMULATION_TOOL, fault_simulation_tool, fault_simulation_tool_handler, ToolHelper
from revert_template import revert_chat_template
from sampling_strategies import (
    Verifier,
    list_available_strategies,
    make_hf_generator,
    make_strategy,
    make_vllm_generator,
    run_strategy_batch,
)

warnings.filterwarnings("ignore")


# =============================================================================
# PASS@K COMPUTATION
# =============================================================================

def estimate_pass_at_k(
    num_samples: np.ndarray,
    num_correct: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Estimate pass@k using the unbiased estimator from Chen et al. (2021).
    
    pass@k = 1 - C(n-c, k) / C(n, k)
    
    Numerically stable version using log-space computation to avoid overflow
    with large n and k values.
    
    Parameters
    ----------
    num_samples : np.ndarray
        Number of total completions per problem (n).
    num_correct : np.ndarray 
        Number of correct completions per problem (c).
    k : int
        The k in pass@k.
    
    Returns
    -------
    np.ndarray
        pass@k estimates for each problem.
    """
    def _estimator(n: int, c: int, k: int) -> float:
        """Compute pass@k for a single problem."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
    
    if isinstance(num_samples, int):
        num_samples = np.array([num_samples])
    if isinstance(num_correct, int):
        num_correct = np.array([num_correct])
    
    assert len(num_samples) == len(num_correct), "num_samples and num_correct must have the same length"
    
    return np.array([
        _estimator(int(n), int(c), k)
        for n, c in zip(num_samples, num_correct)
    ])


# =============================================================================
# TOOL CALLING UTILITIES
# =============================================================================

def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse a tool call from model completion text.
    
    Expected format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    
    Parameters
    ----------
    text : str
        The completion text to parse.
    
    Returns
    -------
    Optional[Dict]
        Parsed tool call dict with at least ``name`` (``arguments`` may be
        missing; use :func:`ensure_tool_call_arguments_dict` before mutating).
    """
    import regex as re
    match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL)
    if match:
        try:
            tool_call_data = json.loads(match.group(1))
            if "name" in tool_call_data:
                return tool_call_data
        except json.JSONDecodeError:
            pass
    return None


def ensure_tool_call_arguments_dict(tool_call: Dict[str, Any]) -> None:
    """
    Ensure ``tool_call['arguments']`` is a dict so callers can ``.update()`` netlist.

    The model may emit OpenAI-style ``parameters`` / ``params`` / ``args``, omit
    ``arguments``, or set it to null. ``parse_tool_call`` only requires ``name``,
    so without this, ``tool_call['arguments'].update(...)`` raises KeyError.
    """
    args = tool_call.get("arguments")
    if isinstance(args, dict):
        return
    if isinstance(args, str) and args.strip():
        try:
            parsed = json.loads(args)
            if isinstance(parsed, dict):
                tool_call["arguments"] = parsed
                return
        except json.JSONDecodeError:
            pass
    for alt in ("parameters", "params", "args"):
        val = tool_call.get(alt)
        if isinstance(val, dict):
            tool_call["arguments"] = dict(val)
            return
    tool_call["arguments"] = {}


def execute_tool_call(tool_call: Dict[str, Any]) -> str:
    """
    Execute a parsed tool call synchronously.
    
    Parameters
    ----------
    tool_call : dict
        Dict with 'name' and 'arguments' keys.
    
    Returns
    -------
    str
        Tool execution result as a string.
    """
    tool_name = tool_call.get("name")
    tool_args = tool_call.get("arguments", {})
    
    if tool_name == "fault_simulation_tool":
        try:
            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(fault_simulation_tool_handler(**tool_args))
            loop.close()
            return str(result)
        except Exception as e:
            return f"Tool execution failed: {e}"
    else:
        return f"Unknown tool: {tool_name}"


# =============================================================================
# GENERATION WITH TOOL CALLING
# =============================================================================

def generate_with_tools(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    generation_config: GenerationConfig,
    max_tool_rounds: int = 1,
) -> str:
    """
    Generate a completion with optional tool calling support.
    
    The model generates text; if a tool call is detected, the tool is executed,
    its result is appended to the conversation, and the model continues generating.
    
    Parameters
    ----------
    model : PeftModel
        The model to generate from.
    tokenizer : AutoTokenizer
        The tokenizer.
    prompt_text : str
        The formatted prompt string.
    generation_config : GenerationConfig
        Generation configuration.
    max_tool_rounds : int
        Maximum number of tool call rounds.
    
    Returns
    -------
    str
        The full completion text (including tool results if any).
    """
    device = next(model.parameters()).device
    full_completion = ""
    current_input = prompt_text
    
    for _round in range(max_tool_rounds + 1):
        # Tokenize current input
        inputs = tokenizer(
            current_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length,
        ).to(device)
        
        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                generation_config=generation_config,
            )
        
        # Decode only the new tokens
        prompt_length = inputs["input_ids"].shape[1]
        completion_ids = output_ids[0, prompt_length:]
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        full_completion += completion_text
        
        # Check for tool call
        if _round < max_tool_rounds:
            tool_call = parse_tool_call(completion_text)
            if tool_call is not None:
                try:
                    ensure_tool_call_arguments_dict(tool_call)
                    tool_call["arguments"].update(
                        {"netlist": ToolHelper.get_netlist(prompt_text)}
                    )
                    tool_result = execute_tool_call(tool_call)
                    messages = revert_chat_template(
                        current_input, tokenizer=tokenizer
                    )
                    messages.append(
                        {"role": "assistant", "content": completion_text}
                    )
                    messages.append({
                        "role": "tool",
                        "name": tool_call.get("name", "fault_simulation_tool"),
                        "content": tool_result,
                    })
                    current_input = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        tools=TOOLS,
                        add_generation_prompt=True,
                    )
                    full_completion += (
                        f"\n<tool_response>\n{tool_result}\n</tool_response>\n"
                    )
                    continue
                except Exception as e:
                    full_completion += (
                        f"\n<tool_response>\nTool round failed: {e}\n</tool_response>\n"
                    )
                    break
        
        break  # No tool call or max rounds reached
    
    return full_completion


def generate_n_completions(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    n: int,
    generation_config: GenerationConfig,
    max_tool_rounds: int = 1,
    micro_batch_size: int = 8,
) -> List[str]:
    """
    Generate n completions for a single prompt (delegates to batched HF generation).
    """
    return generate_batch_n_completions_hf(
        model=model,
        tokenizer=tokenizer,
        prompt_texts=[prompt_text],
        n=n,
        generation_config=generation_config,
        max_tool_rounds=max_tool_rounds,
        micro_batch_size=micro_batch_size,
    )


# =============================================================================
# GENERATION WITH TOOL CALLING (vLLM)
# =============================================================================

def generate_batch_n_completions_vllm(
    llm,  # type: vllm.LLM
    tokenizer: AutoTokenizer,
    prompt_texts: List[str],
    n: int,
    sampling_params,  # type: vllm.SamplingParams
    lora_request,  # type: vllm.lora.request.LoRARequest
    max_tool_rounds: int = 1,
) -> List[str]:
    """
    For B prompts, generate n independent completions each (B * n total paths).
    
    Each round batches all active paths in one ``llm.generate`` call so vLLM can
    schedule work across prompts, not just across samples of a single prompt.
    """
    states = [
        {
            "current_input": p,
            "full_completion": "",
            "done": False,
        }
        for p in prompt_texts
        for _ in range(n)
    ]
    
    for _round in range(max_tool_rounds + 1):
        active_indices = [i for i, state in enumerate(states) if not state["done"]]
        if not active_indices:
            break
        
        active_inputs = [states[i]["current_input"] for i in active_indices]
        
        outputs = llm.generate(
            active_inputs,
            sampling_params=sampling_params,
            lora_request=lora_request,
            use_tqdm=False,
        )
        
        if len(outputs) != len(active_inputs):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} outputs for {len(active_inputs)} "
                "requests; refusing to continue with misaligned tool-round state."
            )
        
        for i, output in zip(active_indices, outputs):
            completion_text = output.outputs[0].text
            states[i]["full_completion"] += completion_text
            
            if _round < max_tool_rounds:
                tool_call = parse_tool_call(completion_text)
                if tool_call is not None:
                    try:
                        ensure_tool_call_arguments_dict(tool_call)
                        tool_call["arguments"].update(
                            {
                                "netlist": ToolHelper.get_netlist(
                                    states[i]["current_input"]
                                )
                            }
                        )
                        tool_result = execute_tool_call(tool_call)
                        
                        messages = revert_chat_template(
                            states[i]["current_input"], tokenizer=tokenizer
                        )
                        messages.append({"role": "assistant", "content": completion_text})
                        messages.append({
                            "role": "tool",
                            "name": tool_call.get("name", "fault_simulation_tool"),
                            "content": tool_result,
                        })
                        
                        states[i]["current_input"] = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            tools=TOOLS,
                            add_generation_prompt=True,
                        )
                        states[i]["full_completion"] += (
                            f"\n<tool_response>\n{tool_result}\n</tool_response>\n"
                        )
                    except Exception as e:
                        err = f"Tool round failed: {e}"
                        states[i]["full_completion"] += (
                            f"\n<tool_response>\n{err}\n</tool_response>\n"
                        )
                        states[i]["done"] = True
                else:
                    states[i]["done"] = True
            else:
                states[i]["done"] = True
    
    return [state["full_completion"] for state in states]


def generate_n_completions_vllm(
    llm,  # type: vllm.LLM
    tokenizer: AutoTokenizer,
    prompt_text: str,
    n: int,
    sampling_params,  # type: vllm.SamplingParams
    lora_request,  # type: vllm.lora.request.LoRARequest
    max_tool_rounds: int = 1,
) -> List[str]:
    """Generate n completions for one prompt via :func:`generate_batch_n_completions_vllm`."""
    return generate_batch_n_completions_vllm(
        llm,
        tokenizer,
        [prompt_text],
        n,
        sampling_params,
        lora_request,
        max_tool_rounds=max_tool_rounds,
    )


@torch.no_grad()
def generate_batch_n_completions_hf(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    prompt_texts: List[str],
    n: int,
    generation_config: GenerationConfig,
    max_tool_rounds: int = 1,
    micro_batch_size: int = 8,
) -> List[str]:
    """
    For B prompts, generate n completions each using batched ``model.generate``.

    Expands to B * n sequences and micro-batches forward passes to limit memory.
    Tool rounds follow the same state machine as the vLLM batch path.
    """
    device = next(model.parameters()).device
    orig_pad_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    try:
        expanded_prompts: List[str] = []
        for p in prompt_texts:
            expanded_prompts.extend([p] * n)

        states = [
            {"current_input": p, "full_completion": "", "done": False}
            for p in expanded_prompts
        ]

        for _round in range(max_tool_rounds + 1):
            active_indices = [i for i, s in enumerate(states) if not s["done"]]
            if not active_indices:
                break

            for start in range(0, len(active_indices), micro_batch_size):
                batch_indices = active_indices[start : start + micro_batch_size]
                batch_inputs = [states[i]["current_input"] for i in batch_indices]

                enc = tokenizer(
                    batch_inputs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                ).to(device)

                out = model.generate(**enc, generation_config=generation_config)
                prompt_len = enc["input_ids"].shape[1]
                decoded = tokenizer.batch_decode(
                    out[:, prompt_len:], skip_special_tokens=True
                )

                if len(decoded) != len(batch_indices):
                    raise RuntimeError(
                        f"HF generate returned {len(decoded)} decodes for "
                        f"{len(batch_indices)} inputs."
                    )

                for idx, completion_text in zip(batch_indices, decoded):
                    states[idx]["full_completion"] += completion_text

                    if _round < max_tool_rounds:
                        tool_call = parse_tool_call(completion_text)
                        if tool_call is not None:
                            try:
                                ensure_tool_call_arguments_dict(tool_call)
                                tool_call["arguments"].update(
                                    {
                                        "netlist": ToolHelper.get_netlist(
                                            states[idx]["current_input"]
                                        )
                                    }
                                )
                                tool_result = execute_tool_call(tool_call)
                                messages = revert_chat_template(
                                    states[idx]["current_input"], tokenizer=tokenizer
                                )
                                messages.append(
                                    {"role": "assistant", "content": completion_text}
                                )
                                messages.append({
                                    "role": "tool",
                                    "name": tool_call.get(
                                        "name", "fault_simulation_tool"
                                    ),
                                    "content": tool_result,
                                })
                                states[idx]["current_input"] = (
                                    tokenizer.apply_chat_template(
                                        messages,
                                        tokenize=False,
                                        tools=TOOLS,
                                        add_generation_prompt=True,
                                    )
                                )
                                states[idx]["full_completion"] += (
                                    f"\n<tool_response>\n{tool_result}\n</tool_response>\n"
                                )
                            except Exception as e:
                                err = f"Tool round failed: {e}"
                                states[idx]["full_completion"] += (
                                    f"\n<tool_response>\n{err}\n</tool_response>\n"
                                )
                                states[idx]["done"] = True
                        else:
                            states[idx]["done"] = True
                    else:
                        states[idx]["done"] = True

        return [s["full_completion"] for s in states]
    finally:
        tokenizer.padding_side = orig_pad_side

# =============================================================================
# PROMPT FORMATTING
# =============================================================================

def _eval_record_as_dict(record: Any) -> Dict[str, Any]:
    """Normalize a single dataset row to a plain dict (HuggingFace row or mapping)."""
    if isinstance(record, dict):
        return record
    try:
        return dict(record)
    except (TypeError, ValueError) as e:
        raise TypeError(
            f"Each eval sample must be a mapping with conversation fields; "
            f"got {type(record).__name__!r}. "
            f"When batching, index the dataset with integers (e.g. ds[i]), "
            f"not by iterating a slice that yields column names."
        ) from e


def _eval_record_fault_meta(record: Any) -> Tuple[str, str]:
    """Safe fault / module_name for error reports when formatting fails."""
    if isinstance(record, dict):
        return str(record.get("fault", "")), str(record.get("module_name", ""))
    return "", ""


def format_eval_prompt(record: Dict[str, Any], tokenizer: AutoTokenizer) -> str:
    """
    Format a dataset record into a prompt for evaluation.
    
    Uses ConversationExample.from_record() to create a structured conversation,
    then extracts only system + user messages (no assistant) as the prompt.
    
    Parameters
    ----------
    record : Dict[str, Any]
        A dataset record with fields like system_content, user_content, etc.
    tokenizer : AutoTokenizer
        The tokenizer with chat template.
    
    Returns
    -------
    str
        The formatted prompt string.
    """
    use_tools = 'tools' in tokenizer.chat_template or 'tool' in tokenizer.chat_template
    convo = ConversationExample.from_record(record, use_tools=use_tools)
    
    # Keep only system and user messages for the prompt
    prompt_messages = [
        m for m in convo.messages 
        if m["role"] not in ("assistant", "tool")
    ]
    
    prompt = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        tools=TOOLS if use_tools else None,
        add_generation_prompt=True,
    )
    
    return prompt


# =============================================================================
# REWARD EVALUATION
# =============================================================================

def evaluate_completions(
    reward_factory: RewardFunctionFactory,
    prompts: List[str],
    completions: List[str],
    records: List[Dict[str, Any]],
) -> List[Dict[str, float]]:
    """
    Evaluate completions using the reward function factory.
    
    Returns per-component reward dictionaries for detailed analysis.
    
    Parameters
    ----------
    reward_factory : RewardFunctionFactory
        The reward function factory instance.
    prompts : List[str]
        The prompt strings.
    completions : List[str]
        The completion strings.
    records : List[Dict]
        The original dataset records (for netlist, fault, etc.).
    
    Returns
    -------
    List[Dict[str, float]]
        Per-completion reward dictionaries with component scores.
    """
    from atpgllm.llm.reward_funcs import test_generation_reward
    from fault_sim import resolve_fault_sim_runner
    
    # Build kwargs
    netlists = []
    for prompt, record in zip(prompts, records):
        rec = _eval_record_as_dict(record)
        netlist_str = rec.get('netlist', '')
        try:
            netlists.append(reward_factory.validate_and_get_netlist_from_prompt(prompt, netlist_str))
        except Exception:
            netlists.append(netlist_str)
    
    # Build fault kwargs from records
    faults = [_eval_record_as_dict(r).get('fault', '') for r in records]
    
    reward_kwargs = {
        "fault_fn": lambda x, **kw: RewardFunctionFactory.fault_fn(x, **kw),
        "simulation_fn": RewardFunctionFactory.simulation_fn,
        "input_vector_fn": RewardFunctionFactory.input_vector_fn,
        "expected_output_fn": RewardFunctionFactory.expected_output_fn,
        "detected_faults_fn": RewardFunctionFactory.detected_faults_fn,
        "eval_mode": True,
        "lib_gate_funcs": reward_factory.gate_funcs,
        "fault_sim": resolve_fault_sim_runner(),
        "netlists": netlists,
        "fault": faults,
        "module_name": [_eval_record_as_dict(r).get("module_name", "") for r in records],
    }
    
    try:
        rewards = test_generation_reward(prompts, completions, **reward_kwargs)
    except Exception as e:
        print(f"Warning: Reward computation failed: {e}")
        import traceback
        traceback.print_exc()
        rewards = [{'format': 0, 'pred_simulation': 0, 'fault_simulation': 0,
                     'input_vector': 0, 'expected_output': 0, 'detected_faults': 0,
                     'fault_detect_inpvector': 0, 'pred_vs_fault_sim_acc': 0,
                     'fault_detected_by_pred_input_vector_acc': 0,
                     'expected_output_acc': 0, 'input_vector_acc': 0,
                     'detected_faults_acc': 0}] * len(prompts)
    
    return rewards


def is_completion_correct(reward: Dict[str, float], threshold_mode: str = "fault_detected") -> bool:
    """
    Determine if a completion is "correct" for pass@k purposes.
    
    Parameters
    ----------
    reward : Dict[str, float]
        The reward dictionary for a completion.
    threshold_mode : str
        How to determine correctness:
        - "fault_detected": The fault was correctly detected by the predicted input vector
        - "positive_reward": Sum of all rewards is positive
        - "full_accuracy": All accuracy metrics are 1.0
    
    Returns
    -------
    bool
        True if the completion passes the threshold.
    """
    def _get_acc(key: str) -> float:
        """Read an _acc field with fallback to its _logonly variant.

        ``test_generation_grpo_reward`` currently emits only ``..._logonly``
        keys for accuracy metrics (they're informational, not part of the
        GRPO loss). Keep reading the bare ``..._acc`` first so any future
        refactor that re-introduces them stays compatible.
        """
        return float(reward.get(key, reward.get(f"{key}_logonly", 0)))

    if threshold_mode == "fault_detected":
        return _get_acc("fault_detected_by_pred_input_vector_acc") == 1
    elif threshold_mode == "positive_reward":
        return sum(reward.values()) > 0
    elif threshold_mode == "full_accuracy":
        return (
            _get_acc("fault_detected_by_pred_input_vector_acc") == 1 and
            _get_acc("input_vector_acc") == 1 and
            _get_acc("expected_output_acc") == 1 and
            _get_acc("detected_faults_acc") == 1
        )
    else:
        raise ValueError(f"Unknown threshold mode: {threshold_mode}")


def wandb_run_name_from_adapter(adapter: Path) -> str:
    """
    Build a short Weights & Biases run *name* from the adapter path so runs are
    identifiable by experiment folder and checkpoint (e.g. SFT adapter dir vs
    GRPO ``.../checkpoint-N/policy`` or ``.../checkpoint-N/combined/policy``).
    """
    try:
        p = adapter.resolve()
    except (OSError, RuntimeError):
        p = Path(os.path.abspath(str(adapter)))

    parts = p.parts
    tokens: Tuple[str, ...]

    if len(parts) >= 2 and parts[-1] == "policy":
        if parts[-2] == "combined" and len(parts) >= 3:
            ckpt_token = parts[-3]
            exp_token = parts[-4] if len(parts) >= 4 else ""
        elif parts[-2].startswith("checkpoint-"):
            ckpt_token = parts[-2]
            exp_token = parts[-3] if len(parts) >= 3 else ""
        else:
            ckpt_token = ""
            exp_token = ""
        if ckpt_token:
            tokens = (
                (exp_token, ckpt_token, "policy")
                if exp_token
                else (ckpt_token, "policy")
            )
        else:
            tokens = (p.parent.name, p.name) if p.parent.name else (p.name,)
    elif p.name.startswith("checkpoint-"):
        exp = p.parent.name
        tokens = (exp, p.name) if exp else (p.name,)
    else:
        tokens = (p.parent.name, p.name) if p.parent.name else (p.name,)

    base = "_".join(t for t in tokens if t)
    base = base.replace(" ", "_")
    if len(base) > 128:
        base = base[-128:]
    return base or "atpg_eval"


def should_disable_custom_all_reduce(tp_size: int) -> bool:
    """
    Decide whether vLLM's custom all-reduce kernel must be disabled for TP>1.

    On NVLink-less nodes (e.g. A100-PCIE) the kernel's GPU P2P transfers can
    silently corrupt tensors, producing garbage completions at TP>1 while
    TP=1 is fine.  On NVLink topologies (H100/H200 SXM) it is safe and
    faster, so we keep it.  Detection is per-node via NVML NVLink state;
    on any detection failure we disable the kernel (correctness over speed —
    the NCCL fallback is always correct).
    """
    if tp_size <= 1:
        return False
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            for link in range(18):  # NVML max NVLink links per GPU
                try:
                    if pynvml.nvmlDeviceGetNvLinkState(handle, link) == pynvml.NVML_FEATURE_ENABLED:
                        print("[vllm] NVLink detected: keeping custom all-reduce enabled.")
                        return False
                except pynvml.NVMLError:
                    break
            print("[vllm] No NVLink (PCIe-only topology): disabling custom all-reduce.")
            return True
        finally:
            pynvml.nvmlShutdown()
    except Exception as e:
        print(f"[vllm] NVLink detection failed ({e}): disabling custom all-reduce.")
        return True


def _merged_export_complete(merged_dir: Path) -> bool:
    """True if *merged_dir* holds a complete HF model export (config present
    and every shard listed in the safetensors index on disk)."""
    if not (merged_dir / "config.json").exists():
        return False
    index_path = merged_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        return all((merged_dir / fn).exists() for fn in set(weight_map.values()))
    return (merged_dir / "model.safetensors").exists()


def resolve_merged_bf16_dir(adapter: Path) -> Path:
    """
    Return the cached QLoRA-faithful merged bf16 export for *adapter*,
    creating it via ``model_utils.export_qlora_merged_bf16`` if missing.

    The export materialises ``dequantize_4bit(W_nf4) + B·A·scaling`` — the
    exact weights GRPO training pushed to its vLLM server — so vLLM can serve
    it as a plain bf16 model without ``enable_lora``.  Cached as a sibling
    directory ``<adapter>_merged_bf16/`` (e.g. ``checkpoint-30/policy_merged_bf16/``).

    The export runs in a subprocess: dequantisation needs a CUDA context that
    would otherwise stay resident in this process and starve vLLM's GPU
    memory reservation (`del` + ``empty_cache()`` cannot release the context).
    """
    merged_dir = adapter.parent / f"{adapter.name}_merged_bf16"
    if _merged_export_complete(merged_dir):
        print(f"[merge_dequant] Reusing cached merged bf16 export: {merged_dir}")
        return merged_dir

    import subprocess

    print(f"[merge_dequant] No complete cached export found, building: {merged_dir}")
    tests_dir = str(Path(__file__).resolve().parent)
    code = (
        "import sys; "
        f"sys.path.insert(0, {tests_dir!r}); "
        "from model_utils import export_qlora_merged_bf16; "
        f"export_qlora_merged_bf16({str(adapter)!r}, {str(merged_dir)!r})"
    )
    result = subprocess.run([sys.executable, "-c", code])
    if result.returncode != 0:
        raise RuntimeError(
            f"merged bf16 export subprocess failed (exit {result.returncode}) "
            f"for adapter: {adapter}"
        )
    if not _merged_export_complete(merged_dir):
        raise RuntimeError(
            f"merged bf16 export finished but {merged_dir} is incomplete "
            "(missing config.json or weight shards)"
        )
    return merged_dir


# =============================================================================
# MAIN EVALUATION PIPELINE
# =============================================================================

def evaluate(
    adapter: Path,
    dataset_path: str = "chrivasileiou/asap7-language-of-test-v2",
    num_completions: int = 10,
    k_values: List[int] = None,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_new_tokens: int = 16384,
    max_prompt_length: int = 4096,
    max_eval_samples: int = -1,
    max_tool_rounds: int = 1,
    threshold_mode: str = "fault_detected",
    config_path: str = "sim_config.json",
    seed: int = 42,
    report_to: str = "none",
    output_file: Optional[str] = None,
    backend: str = "transformers",
    tp_size: int = 2,
    gpu_memory_utilization: float = 0.9,
    qlora: bool = False,
    eval_prompt_batch_size: int = 8,
    generation_micro_batch_size: int = 8,
    wandb_run_name: Optional[str] = None,
    sampling_method: str = "random",
    merge_dequant: bool = False,
    budget: Optional[int] = None,
    best_of_n_width: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Main evaluation function.
    
    Parameters
    ----------
    adapter : str
        Path to SFT/GRPO adapter checkpoint.
    dataset_path : str
        HuggingFace dataset identifier.
    num_completions : int
        Completions per problem for pass@k estimation (the pass@k pool, N).
        Each completion is one independent application of the sampling
        strategy.
    k_values : List[int]
        List of k values to compute pass@k for.
    temperature : float
        Generation temperature. Higher = more diverse.
    top_p : float
        Nucleus sampling threshold.
    max_new_tokens : int
        Maximum new tokens to generate per completion.
    max_eval_samples : int
        Maximum number of eval samples (-1 for all).
    max_tool_rounds : int
        Maximum tool call rounds per generation.
    threshold_mode : str
        Correctness threshold mode for pass@k.
    config_path : str
        Path to sim_config.json for reward computation.
    seed : int
        Random seed.
    report_to : str
        Reporting backend ("wandb", "none").
    output_file : str, optional
        Path to save detailed results as JSON.
    eval_prompt_batch_size : int
        How many dataset prompts to run through generation together. Each batch
        issues one fused multi-path generation of ``batch_size * num_completions`` sequences
        per tool round (vLLM), improving GPU utilization vs one prompt at a time.
    generation_micro_batch_size : int
        Transformers backend only: cap on parallel sequences per ``generate`` call.
    wandb_run_name : str, optional
        Weights & Biases run display name. If omitted, derived from *adapter*.
    merge_dequant : bool
        vLLM backend only: export the QLoRA-faithful merged bf16 model
        (``dequantize_4bit(W_nf4) + B·A·scaling``, exactly what GRPO training
        pushed to its vLLM server) and serve it without dynamic LoRA.  The
        export is cached next to the adapter as ``<adapter>_merged_bf16/``.
    budget : int, optional
        Per-completion search width (B) for ``mcts`` / ``evolutionary`` only:
        scored rollouts (mcts) or completions evaluated (evolutionary) per
        independent search. Orthogonal to ``num_completions``.
    best_of_n_width : int, optional
        Per-completion width (B) for ``best_of_n`` only: i.i.d. samples drawn
        per completion, of which the best is kept (``1`` = i.i.d. baseline).

    Returns
    -------
    Dict[str, Any]
        Results dictionary with pass@k scores and detailed metrics.
    """
    if k_values is None:
        k_values = [1, 5, 10]
    
    # Validate k values
    for k in k_values:
        if k > num_completions:
            raise ValueError(
                f"k={k} > num_completions={num_completions}. Cannot compute "
                f"pass@{k} with only {num_completions} completions per problem."
            )

    # ``budget`` (mcts/evolutionary) and ``best_of_n_width`` (best_of_n) are the
    # same concept — the per-completion search width (B) — under different flag
    # names; ``random`` takes neither. Normalize to a single ``width``.
    if sampling_method in ("mcts", "evolutionary"):
        if budget is None:
            raise ValueError(
                f"--budget is required when --sampling_method={sampling_method}"
            )
        if budget < 1:
            raise ValueError(f"--budget must be >= 1 (got {budget})")
        if best_of_n_width is not None:
            raise ValueError(
                "--n applies to best_of_n only; use --budget for "
                f"{sampling_method}"
            )
        width = budget
    elif sampling_method == "best_of_n":
        if best_of_n_width is None:
            raise ValueError("--n is required when --sampling_method=best_of_n")
        if best_of_n_width < 1:
            raise ValueError(f"--n must be >= 1 (got {best_of_n_width})")
        if budget is not None:
            raise ValueError(
                "--budget applies to mcts/evolutionary only; use --n for "
                "best_of_n"
            )
        width = best_of_n_width
    else:  # random
        if budget is not None or best_of_n_width is not None:
            raise ValueError(
                "--budget / --n do not apply to the random sampling method"
            )
        width = None
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # =========================================================================
    # 1. Load Model
    # =========================================================================
    print("=" * 70)
    print("LOADING MODEL")
    print("=" * 70)
    
    # Load tokenizer from the checkpoint's base model
    with open(adapter / "adapter_config.json", "r") as f:
        adapter_config = json.load(f)
    base_model_name = adapter_config["base_model_name_or_path"]
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # For generation
    
    if backend == "vllm":
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        if merge_dequant:
            # QLoRA-faithful serving: the adapter was trained against the
            # NF4-quantised base, so serve dequant(W_nf4) + B·A·s as plain
            # bf16 weights instead of mounting the LoRA on the clean Hub base.
            merged_dir = resolve_merged_bf16_dir(adapter)
            vllm_model_path = str(merged_dir)
            lora_kwargs = {}
        else:
            vllm_model_path = base_model_name
            lora_kwargs = {
                "enable_lora": True,
                "max_lora_rank": adapter_config.get("r", 8),
            }
            if qlora:
                # Activates handling for bitsandbytes quantized base models natively.
                lora_kwargs["quantization"] = "bitsandbytes"
                lora_kwargs["load_format"] = "bitsandbytes"

        try:
            model = LLM(
                model=vllm_model_path,
                tensor_parallel_size=tp_size,
                gpu_memory_utilization=gpu_memory_utilization,
                dtype="bfloat16",
                trust_remote_code=True,
                disable_custom_all_reduce=should_disable_custom_all_reduce(tp_size),
                **lora_kwargs
            )
        except (RuntimeError, ValueError) as e:
            em = str(e).lower()
            if "memory" in em or "free memory" in em or "gpu" in em:
                raise RuntimeError(
                    f"{e}\n\n"
                    f"vLLM could not reserve GPU memory (utilization target={gpu_memory_utilization}). "
                    "Lower --gpu_memory_utilization, reduce --tp_size if misconfigured, or free VRAM "
                    "from other processes."
                ) from e
            raise
        lora_request = None if merge_dequant else LoRARequest("active_adapter", 1, str(adapter))
        
        # In vLLM, n=1 here because the generation_vllm function manually submits 'n' active prompts
        generation_config = SamplingParams(
            n=1,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            stop_token_ids=[tokenizer.eos_token_id, tokenizer.pad_token_id],
        )
        
        print(f"vLLM Engine initialized on {tp_size} GPUs.")
        if merge_dequant:
            print(f"Serving QLoRA-faithful merged bf16 model: {vllm_model_path}")
        else:
            print(f"Base Model: {base_model_name}")
            print(f"Adapter dynamically mounted from: {adapter.as_posix()}")

    else:
        if merge_dequant:
            print(
                "[merge_dequant] Note: flag ignored for the transformers backend — "
                "load_dual_adapter_model already loads the NF4-quantised base, "
                "which is QLoRA-faithful by construction."
            )
        # Standard HF PEFT
        model = load_dual_adapter_model(adapter=adapter)
        model.eval()
        
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        print(f"Model loaded via Transformers/PEFT from: {adapter.as_posix()}")

    # =========================================================================
    # 2. Load Dataset
    # =========================================================================
    print("\n" + "=" * 70)
    print("LOADING DATASET")
    print("=" * 70)
    
    eval_dataset = load_dataset(dataset_path, split="test", streaming=True)
    
    # Buffer the streaming dataset
    eval_records = buffer_streaming_dataset(eval_dataset, buffer_size=max_eval_samples, shuffle=False, unique_by="netlist", tokenizer=tokenizer, max_prompt_length=max_prompt_length)
    
    print(f"Loaded {len(eval_records)} eval samples from '{dataset_path}' (eval split)")
    
    # =========================================================================
    # 3. Initialize Reward Factory
    # =========================================================================
    print("\n" + "=" * 70)
    print("INITIALIZING REWARD FACTORY")
    print("=" * 70)
    
    reward_factory = RewardFunctionFactory(config_path=config_path)
    print(f"Loaded gate functions from: {config_path}")
    
    # =========================================================================
    # 4. Optional WandB Setup
    # =========================================================================
    wandb_run = None
    if report_to == "wandb":
        try:
            import wandb
            wb_name = wandb_run_name or wandb_run_name_from_adapter(adapter)
            wandb_run = wandb.init(
                project="atpg-eval",
                name=wb_name,
                tags=["passatk"],
                config={
                    "adapter": str(adapter),
                    "dataset": dataset_path,
                    "num_completions": num_completions,
                    "sampling_method": sampling_method,
                    "width": width,
                    "k_values": k_values,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_new_tokens": max_new_tokens,
                    "max_tool_rounds": max_tool_rounds,
                    "threshold_mode": threshold_mode,
                    "num_eval_samples": len(eval_records),
                    "seed": seed,
                    "eval_prompt_batch_size": eval_prompt_batch_size,
                    "generation_micro_batch_size": generation_micro_batch_size,
                },
            )
        except ImportError:
            print("wandb not installed; skipping wandb logging")
    
    # =========================================================================
    # 5. Generation & Evaluation Loop
    # =========================================================================
    print("\n" + "=" * 70)
    print(
        f"EVALUATING: num_completions={num_completions}, k={k_values}, "
        f"temperature={temperature}, prompt_batch_size={eval_prompt_batch_size}"
    )
    print(f"Threshold mode: {threshold_mode}")
    print("=" * 70)
    
    all_results = []  # Per-problem results
    all_num_correct = []  # Number of correct completions per problem
    all_rewards = []  # Detailed rewards per problem per completion
    
    eval_start_time = time.time()
    # =========================================================================
    # 4.5. Build the search strategy.
    # When max_tool_rounds >= 1, the search runs the same fault-simulation tool
    # loop as the random path (the model can call the tool mid-generation);
    # otherwise tool calls are disabled and the simulator is used only as the
    # external verifier / oracle.
    # =========================================================================
    strategy = None
    if sampling_method != "random":
        verifier = Verifier(reward_factory)
        if backend == "vllm":
            generator = make_vllm_generator(
                model, tokenizer, lora_request, generation_config,
            )
        else:
            generator = make_hf_generator(
                model, tokenizer, generation_config,
                micro_batch_size=generation_micro_batch_size,
            )
        strategy_kwargs = {}
        if sampling_method == "mcts":
            strategy_kwargs["branching"] = 3
            strategy_kwargs["chunk_tokens"] = 256
            strategy_kwargs["rollout_max_tokens"] = None
            # PUCT (AlphaZero / Silver 2017) selection knobs: c_puct trades off
            # exploration vs exploitation; prior_temperature (tau) sharpens or
            # flattens the LM policy prior P(s, a) over sampled chunks.
            strategy_kwargs["c_puct"] = 1.25
            strategy_kwargs["prior_temperature"] = 1.0
            strategy_kwargs["chunk_temperature"] = 0.9
            strategy_kwargs["rollout_temperature"] = 0.7
        elif sampling_method == "evolutionary":
            strategy_kwargs["population_size"] = 6
            strategy_kwargs["elite_fraction"] = 0.5
            strategy_kwargs["mutation_temperature"] = 1.0
            strategy_kwargs["crossover_temperature"] = 0.7
            strategy_kwargs["crossover_max_tokens"] = 2048
        
        use_tools = max_tool_rounds >= 1
        strategy = make_strategy(
            sampling_method,
            generator,
            verifier,
            num_completions=num_completions,
            width=width,
            use_tools=use_tools,
            max_tool_rounds=max_tool_rounds,
            **strategy_kwargs,
        )
        width_label = "best_of_n_width" if sampling_method == "best_of_n" else "search_budget"
        print(
            f"Sampling strategy: {sampling_method} "
            f"(num_completions={num_completions}, {width_label}={width}, "
            f"tool_calling={'on' if use_tools else 'off'}, "
            f"max_tool_rounds={max_tool_rounds})"
        )

    n_records = len(eval_records)
    batch_starts = list(range(0, n_records, max(1, eval_prompt_batch_size)))
    
    for batch_start in tqdm(batch_starts, desc="Evaluating"):
        batch_end = min(batch_start + eval_prompt_batch_size, n_records)
        global_indices = list(range(batch_start, batch_end))
        # Index rows by int: slicing ``Dataset[start:end]`` and iterating can yield
        # column names (strings) on some versions, not one dict per row.
        chunk = [eval_records[i] for i in global_indices]
        
        ok_prompts: List[str] = []
        ok_records: List[Dict[str, Any]] = []
        ok_slot: List[int] = []  # index within this chunk (0 .. len(chunk)-1)
        
        chunk_num_correct: List[Optional[int]] = [None] * len(chunk)
        chunk_rewards: List[Optional[List[Dict[str, float]]]] = [None] * len(chunk)
        chunk_time_seconds: List[Optional[float]] = [None] * len(chunk)
        chunk_problem_result: List[Optional[Dict[str, Any]]] = [None] * len(chunk)
        
        for slot, (idx, record) in enumerate(zip(global_indices, chunk)):
            try:
                prompt_text = format_eval_prompt(record.copy(), tokenizer)
                ok_prompts.append(prompt_text)
                ok_records.append(record)
                ok_slot.append(slot)
            except Exception as e:
                print(f"Warning: Failed to format prompt for sample {idx}: {e}")
                fault_s, mod_s = _eval_record_fault_meta(record)
                chunk_num_correct[slot] = 0
                chunk_rewards[slot] = []
                chunk_time_seconds[slot] = 0.0
                chunk_problem_result[slot] = {
                    "idx": idx,
                    "fault": fault_s,
                    "module_name": mod_s,
                    "error": f"prompt_format_error: {e}",
                    "num_correct": 0,
                    "num_completions": num_completions,
                }
        
        if ok_prompts:
            gen_t0 = time.time()
            if strategy is not None:
                # Non-random search: strategy owns both generation and scoring.
                flat_completions, rewards_flat, _ = run_strategy_batch(
                    strategy=strategy,
                    prompts=ok_prompts,
                    records=ok_records,
                    num_completions=num_completions,
                )
                gen_dt = time.time() - gen_t0
                prompts_rep = [p for p in ok_prompts for _ in range(num_completions)]
                records_rep = [r for r in ok_records for _ in range(num_completions)]
            else:
                if backend == "vllm":
                    flat_completions = generate_batch_n_completions_vllm(
                        llm=model,
                        tokenizer=tokenizer,
                        prompt_texts=ok_prompts,
                        n=num_completions,
                        sampling_params=generation_config,
                        lora_request=lora_request,
                        max_tool_rounds=max_tool_rounds,
                    )
                else:
                    flat_completions = generate_batch_n_completions_hf(
                        model=model,
                        tokenizer=tokenizer,
                        prompt_texts=ok_prompts,
                        n=num_completions,
                        generation_config=generation_config,
                        max_tool_rounds=max_tool_rounds,
                        micro_batch_size=generation_micro_batch_size,
                    )
                gen_dt = time.time() - gen_t0

                expected_flat = len(ok_prompts) * num_completions
                if len(flat_completions) != expected_flat:
                    print(
                        f"Warning: expected {expected_flat} completions "
                        f"({len(ok_prompts)} prompts × num_completions={num_completions}), "
                        f"got {len(flat_completions)}; padding or truncating to match."
                    )
                    if len(flat_completions) < expected_flat:
                        flat_completions = flat_completions + [""] * (
                            expected_flat - len(flat_completions)
                        )
                    else:
                        flat_completions = flat_completions[:expected_flat]

                prompts_rep = [p for p in ok_prompts for _ in range(num_completions)]
                records_rep = [r for r in ok_records for _ in range(num_completions)]
                try:
                    rewards_flat = evaluate_completions(
                        reward_factory=reward_factory,
                        prompts=prompts_rep,
                        completions=flat_completions,
                        records=records_rep,
                    )
                except Exception as e:
                    print(f"Warning: Batch reward computation failed: {e}")
                    import traceback
                    traceback.print_exc()
                    zero_r = {
                        'format': 0, 'pred_simulation': 0, 'fault_simulation': 0,
                        'input_vector': 0, 'expected_output': 0, 'detected_faults': 0,
                        'fault_detect_inpvector': 0, 'pred_vs_fault_sim_acc': 0,
                        'fault_detected_by_pred_input_vector_acc': 0,
                        'expected_output_acc': 0, 'input_vector_acc': 0,
                        'detected_faults_acc': 0,
                    }
                    rewards_flat = [zero_r] * (len(ok_prompts) * num_completions)
            
            n_reward = len(rewards_flat)
            if n_reward != len(prompts_rep):
                print(
                    f"Warning: reward list length {n_reward} != "
                    f"{len(prompts_rep)} (prompts × completions); adjusting."
                )
                if n_reward < len(prompts_rep):
                    zero_r = {
                        'format': 0, 'pred_simulation': 0, 'fault_simulation': 0,
                        'input_vector': 0, 'expected_output': 0, 'detected_faults': 0,
                        'fault_detect_inpvector': 0, 'pred_vs_fault_sim_acc': 0,
                        'fault_detected_by_pred_input_vector_acc': 0,
                        'expected_output_acc': 0, 'input_vector_acc': 0,
                        'detected_faults_acc': 0,
                    }
                    rewards_flat = rewards_flat + [zero_r] * (
                        len(prompts_rep) - n_reward
                    )
                else:
                    rewards_flat = rewards_flat[: len(prompts_rep)]
            
            per_problem_time = gen_dt / max(len(ok_prompts), 1)
            
            for j, slot in enumerate(ok_slot):
                problem_rewards = rewards_flat[j * num_completions : (j + 1) * num_completions]
                num_correct = sum(
                    1
                    for r in problem_rewards
                    if is_completion_correct(r, threshold_mode=threshold_mode)
                )
                idx = global_indices[slot]
                record = ok_records[j]
                
                chunk_num_correct[slot] = num_correct
                chunk_rewards[slot] = problem_rewards
                chunk_time_seconds[slot] = round(per_problem_time, 2)
                chunk_problem_result[slot] = {
                    "idx": idx,
                    "fault": record.get("fault", ""),
                    "module_name": record.get("module_name", ""),
                    "num_correct": num_correct,
                    "num_completions": num_completions,
                    "time_seconds": chunk_time_seconds[slot],
                    "rewards_summary": {
                        "mean_total_reward": np.mean(
                            [sum(r.values()) for r in problem_rewards]
                        ),
                        "fault_detection_rate": num_correct / num_completions,
                        "mean_format_reward": np.mean(
                            [r.get('format', 0) for r in problem_rewards]
                        ),
                        "mean_fault_sim_reward": np.mean(
                            [r.get('fault_simulation', 0) for r in problem_rewards]
                        ),
                    },
                }
        
        for slot in range(len(chunk)):
            if chunk_problem_result[slot] is None:
                idx = global_indices[slot]
                rec = chunk[slot]
                fault_s, mod_s = _eval_record_fault_meta(rec)
                chunk_num_correct[slot] = chunk_num_correct[slot] or 0
                chunk_rewards[slot] = chunk_rewards[slot] or []
                chunk_problem_result[slot] = {
                    "idx": idx,
                    "fault": fault_s,
                    "module_name": mod_s,
                    "error": "internal_error: incomplete batch slot",
                    "num_correct": 0,
                    "num_completions": num_completions,
                }
            all_num_correct.append(chunk_num_correct[slot])
            all_rewards.append(chunk_rewards[slot])
            all_results.append(chunk_problem_result[slot])
        
        last_idx = global_indices[-1]
        if wandb_run and (last_idx + 1) % 10 == 0:
            running_pass_at_k = {}
            for k in k_values:
                if k <= num_completions:
                    pass_k = estimate_pass_at_k(
                        np.array([num_completions] * len(all_num_correct)),
                        np.array(all_num_correct),
                        k,
                    )
                    running_pass_at_k[f"running_pass@{k}"] = np.mean(pass_k)
            _rates = [
                r["rewards_summary"]["fault_detection_rate"]
                for r in all_results
                if "error" not in r and "rewards_summary" in r
            ]
            wandb.log({
                "eval_step": last_idx + 1,
                **running_pass_at_k,
                "running_fault_detection_rate": float(np.mean(_rates)) if _rates else 0.0,
            })
        
        if (last_idx + 1) % 10 == 0:
            _rates = [
                r["rewards_summary"]["fault_detection_rate"]
                for r in all_results
                if "error" not in r and "rewards_summary" in r
            ]
            running_rate = float(np.mean(_rates)) if _rates else 0.0
            print(
                f"  [{last_idx+1}/{n_records}] "
                f"Running fault detection rate: {running_rate:.3f}"
            )
    
    eval_time = time.time() - eval_start_time
    
    # =========================================================================
    # 6. Compute pass@k Metrics
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPUTING PASS@K METRICS")
    print("=" * 70)
    
    num_samples_arr = np.array([num_completions] * len(all_num_correct))
    num_correct_arr = np.array(all_num_correct)
    
    pass_at_k_results = {}
    for k in k_values:
        if k <= num_completions:
            pass_k = estimate_pass_at_k(num_samples_arr, num_correct_arr, k)
            mean_pass_k = np.mean(pass_k)
            pass_at_k_results[f"pass@{k}"] = float(mean_pass_k)
            print(f"  pass@{k} = {mean_pass_k:.4f}")
    
    # =========================================================================
    # 7. Compute Aggregate Metrics
    # =========================================================================
    valid_results = [r for r in all_results if "error" not in r]
    
    aggregate_metrics = {
        "num_eval_samples": len(eval_records),
        "num_valid_samples": len(valid_results),
        "num_errors": len(all_results) - len(valid_results),
        "total_eval_time_seconds": round(eval_time, 2),
        "avg_time_per_sample_seconds": round(eval_time / max(len(eval_records), 1), 2),
        "num_completions": num_completions,
        "sampling_method": sampling_method,
        "search_width": width,
        "temperature": temperature,
        "threshold_mode": threshold_mode,
    }
    
    # Per-component accuracy averages across all problems
    component_metrics = defaultdict(list)
    for problem_rewards in all_rewards:
        for reward in problem_rewards:
            for key, value in reward.items():
                component_metrics[key].append(value)
    
    avg_component_metrics = {
        f"avg_{key}": float(np.mean(values))
        for key, values in component_metrics.items()
    }
    
    # Accuracy metrics (only the _acc fields, which are 0 or 1)
    accuracy_metrics = {
        key: float(np.mean(values))
        for key, values in component_metrics.items()
        if key.endswith("_acc")
    }
    
    print("\nComponent Accuracy Metrics:")
    for key, val in accuracy_metrics.items():
        print(f"  {key}: {val:.4f}")
    
    print(f"\nOverall fault detection rate: {accuracy_metrics.get('fault_detected_by_pred_input_vector_acc', 0):.4f}")
    print(f"Total evaluation time: {eval_time:.1f}s ({eval_time/60:.1f}m)")
    
    # =========================================================================
    # 8. Compile Results
    # =========================================================================
    results = {
        "pass_at_k": pass_at_k_results,
        "aggregate_metrics": aggregate_metrics,
        "avg_component_metrics": avg_component_metrics,
        "accuracy_metrics": accuracy_metrics,
        "config": {
            "adapter": str(adapter),
            "dataset": dataset_path,
            "num_completions": num_completions,
            "search_width": width,
            "k_values": k_values,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "max_tool_rounds": max_tool_rounds,
            "threshold_mode": threshold_mode,
            "seed": seed,
            "backend": backend,
            "tp_size": tp_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "qlora": qlora,
            "eval_prompt_batch_size": eval_prompt_batch_size,
            "generation_micro_batch_size": generation_micro_batch_size,
            "sampling_method": sampling_method,
        },
    }
    
    # Save detailed results if output file specified
    if output_file:
        detailed_output = {
            **results,
            "per_problem_results": all_results,
        }
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(detailed_output, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {output_file}")
    
    # Log final metrics to wandb
    if wandb_run:
        wandb.log({
            **pass_at_k_results,
            **accuracy_metrics,
            **aggregate_metrics,
        })
        wandb.finish()
    
    # =========================================================================
    # 9. Print Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Dataset: {dataset_path} (eval split)")
    print(f"Samples: {len(eval_records)}")
    print(f"Completions per sample: {num_completions}")
    print(f"Temperature: {temperature}")
    print(f"Threshold mode: {threshold_mode}")
    print()
    for k, v in pass_at_k_results.items():
        print(f"  {k}: {v:.4f}")
    print(f"\n  Fault detection rate: {accuracy_metrics.get('fault_detected_by_pred_input_vector_acc', 0):.4f}")
    print(f"  Input vector accuracy: {accuracy_metrics.get('input_vector_acc', 0):.4f}")
    print(f"  Expected output accuracy: {accuracy_metrics.get('expected_output_acc', 0):.4f}")
    print(f"  Detected faults accuracy: {accuracy_metrics.get('detected_faults_acc', 0):.4f}")
    print("=" * 70)
    
    return results


# =============================================================================
# SFT STOPPING CRITERIA EVALUATION
# =============================================================================

def discover_checkpoints(output_dir: Path) -> List[Tuple[int, Path]]:
    """
    Find checkpoint-N directories containing adapter_config.json under
    *output_dir*.  Returns a list of ``(step, path)`` sorted by step.
    """
    checkpoints = []
    if not output_dir.is_dir():
        return checkpoints
    for entry in output_dir.iterdir():
        if not entry.is_dir() or not entry.name.startswith("checkpoint-"):
            continue
        try:
            step = int(entry.name.split("-", 1)[1])
        except (ValueError, IndexError):
            continue
        if (entry / "adapter_config.json").exists():
            checkpoints.append((step, entry))
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def load_sft_eval_prompts(
    tokenizer: AutoTokenizer,
    dataset_path: str,
    eval_buffer_size: int,
    max_prompt_length: int,
) -> List[str]:
    """
    Load eval prompts using the same pipeline as SFTStoppingCallback:
    streaming test split -> GRPO format -> buffer -> extract ``prompt`` field.
    """
    from dataset_utils import (
        TrainingMode,
        buffer_streaming_dataset,
        format_dataset_for_training,
    )

    print(
        f"[SFTStopEval] Loading eval dataset from {dataset_path} "
        f"(buffer={eval_buffer_size})..."
    )
    raw = load_dataset(dataset_path, split="test", streaming=True)
    formatted = format_dataset_for_training(raw, tokenizer, TrainingMode.GRPO)
    buffered = buffer_streaming_dataset(
        formatted,
        buffer_size=eval_buffer_size,
        shuffle=False,
        tokenizer=tokenizer,
        max_prompt_length=max_prompt_length,
    )
    prompts = [ex["prompt"] for ex in buffered]
    print(f"[SFTStopEval] Loaded {len(prompts)} eval prompts")
    return prompts


def sft_check_format_compliance(
    completions: List[str],
) -> Tuple[float, Dict[str, Any]]:
    """
    Score format compliance over *completions*.
    Mirrors ``SFTStoppingCallback._check_format_compliance``.

    Returns ``(score, details)`` where *score* is the fraction of fully
    parseable completions.
    """
    from callbacks import (
        THINK_RE, TOOL_CALL_RE, INPUT_VECTOR_RE,
        EXPECTED_OUTPUT_RE, DETECTED_FAULTS_RE,
    )

    total = len(completions)
    details: Dict[str, int] = {
        "total": total,
        "think_ok": 0,
        "tool_call_ok": 0,
        "tool_call_json_ok": 0,
        "input_vector_ok": 0,
        "expected_output_ok": 0,
        "detected_faults_ok": 0,
        "fully_parseable": 0,
    }

    for comp in completions:
        has_think = bool(THINK_RE.search(comp))
        tc_match = TOOL_CALL_RE.search(comp)
        has_tc = bool(tc_match)
        has_tc_json = False
        if tc_match:
            try:
                json.loads(tc_match.group(1))
                has_tc_json = True
            except (json.JSONDecodeError, ValueError):
                pass
        has_iv = bool(INPUT_VECTOR_RE.search(comp))
        has_eo = bool(EXPECTED_OUTPUT_RE.search(comp))
        has_df = bool(DETECTED_FAULTS_RE.search(comp))

        details["think_ok"] += int(has_think)
        details["tool_call_ok"] += int(has_tc)
        details["tool_call_json_ok"] += int(has_tc_json)
        details["input_vector_ok"] += int(has_iv)
        details["expected_output_ok"] += int(has_eo)
        details["detected_faults_ok"] += int(has_df)
        if has_tc_json and has_iv and has_eo and has_df:
            details["fully_parseable"] += 1

    score = details["fully_parseable"] / total if total > 0 else 0.0
    return score, details


def sft_check_diversity(
    completions: List[str],
) -> Tuple[float, Dict[str, Any]]:
    """
    Score output diversity on completions generated for the *same* prompt.
    Mirrors ``SFTStoppingCallback._check_diversity``.

    Returns ``(score, details)`` where
    ``score = effective_unique / total_completions``.
    """
    from callbacks import INPUT_VECTOR_RE

    vectors: List[Optional[str]] = []
    for comp in completions:
        m = INPUT_VECTOR_RE.search(comp)
        vectors.append(m.group(1).strip() if m else None)

    parseable = [v for v in vectors if v is not None]
    unique_count = len(set(parseable)) if parseable else 0
    none_count = sum(1 for v in vectors if v is None)
    effective_unique = unique_count + min(none_count, 1)
    score = effective_unique / len(completions) if completions else 0.0

    return score, {
        "total_generations": len(completions),
        "parseable_count": len(parseable),
        "unique_input_vectors": unique_count,
        "unique_texts": len(set(c.strip() for c in completions)),
        "unparseable_count": none_count,
        "effective_unique": effective_unique,
        "sample_vectors": parseable[:3],
    }


def sft_check_loss_plateau(
    output_dir: Path,
    up_to_step: int,
    loss_window: int = 10,
    loss_delta: float = 0.01,
) -> bool:
    """
    Read ``trainer_state.json`` and check whether the training loss has
    plateaued up to *up_to_step*.
    Mirrors ``SFTStoppingCallback._check_loss_plateau``.
    """
    candidates = [
        output_dir / "trainer_state.json",
        output_dir / f"checkpoint-{up_to_step}" / "trainer_state.json",
    ]
    state_data = None
    for path in candidates:
        if path.exists():
            try:
                with open(path) as f:
                    state_data = json.load(f)
                break
            except (json.JSONDecodeError, IOError):
                continue

    if state_data is None:
        return False

    losses: List[float] = []
    for entry in state_data.get("log_history", []):
        loss_val = entry.get("loss")
        if loss_val is not None:
            if entry.get("step", 0) > up_to_step:
                break
            losses.append(float(loss_val))

    if len(losses) < loss_window:
        return False

    recent = losses[-loss_window:]
    older_start = max(0, len(losses) - 2 * loss_window)
    older_end = len(losses) - loss_window
    older = losses[older_start:older_end]
    if not older:
        return False

    improvement = float(np.mean(older)) - float(np.mean(recent))
    return improvement < loss_delta


# -----------------------------------------------------------------------------
# Batched generation helpers for SFT stop eval
# -----------------------------------------------------------------------------

def _sft_eval_generate_vllm(
    llm,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    n_per_prompt: int,
    sampling_params,
    lora_request,
    max_tool_rounds: int = 1,
) -> List[str]:
    """
    Batched multi-turn generation via vLLM for SFT stopping evaluation.
    Returns ``len(prompts) * n_per_prompt`` completions.
    """
    return generate_batch_n_completions_vllm(
        llm,
        tokenizer,
        prompts,
        n_per_prompt,
        sampling_params,
        lora_request,
        max_tool_rounds=max_tool_rounds,
    )


@torch.no_grad()
def _sft_eval_generate_hf(
    model,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    n_per_prompt: int,
    max_new_tokens: int = 8192,
    temperature: float = 0.7,
    batch_size: int = 8,
    max_tool_rounds: int = 1,
) -> List[str]:
    """
    Batched two-turn generation via HF ``model.generate()`` for SFT stop eval.
    Mirrors ``SFTStoppingCallback._generate_hf``.
    """
    device = next(model.parameters()).device
    orig_pad_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    try:
        expanded = []
        for p in prompts:
            expanded.extend([p] * n_per_prompt)

        # -- Turn 1 --
        first_turns: List[str] = []
        for start in range(0, len(expanded), batch_size):
            batch = expanded[start : start + batch_size]
            enc = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=max_new_tokens,
            ).to(device)
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
            )
            first_turns.extend(
                tokenizer.batch_decode(
                    out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True,
                )
            )

        # -- Tool execution --
        full: List[str] = list(first_turns)
        turn2_items: List[tuple] = []
        for idx, ft in enumerate(first_turns):
            tc = parse_tool_call(ft)
            if not tc:
                continue
            ensure_tool_call_arguments_dict(tc)
            tc["arguments"].update(
                {"netlist": ToolHelper.get_netlist(expanded[idx])}
            )
            result = execute_tool_call(tc)
            try:
                msgs = revert_chat_template(expanded[idx], tokenizer=tokenizer)
                msgs.append({"role": "assistant", "content": ft})
                msgs.append({
                    "role": "tool",
                    "name": tc.get("name", "fault_simulation_tool"),
                    "content": result,
                })
                cont = tokenizer.apply_chat_template(
                    msgs, tokenize=False, tools=TOOLS,
                    add_generation_prompt=True,
                )
                turn2_items.append((idx, ft, result, cont))
            except Exception:
                pass

        # -- Turn 2 --
        if turn2_items:
            conts = [item[3] for item in turn2_items]
            turn2_texts: List[str] = []
            for start in range(0, len(conts), batch_size):
                batch = conts[start : start + batch_size]
                enc = tokenizer(
                    batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=max_new_tokens // 2,
                ).to(device)
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens // 2,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                )
                turn2_texts.extend(
                    tokenizer.batch_decode(
                        out[:, enc["input_ids"].shape[1] :],
                        skip_special_tokens=True,
                    )
                )
            for j, (orig_idx, ft, result, _) in enumerate(turn2_items):
                full[orig_idx] = (
                    ft
                    + f"\n<tool_response>\n{result}\n</tool_response>\n"
                    + turn2_texts[j]
                )

        return full
    finally:
        tokenizer.padding_side = orig_pad_side


# -----------------------------------------------------------------------------
# Main SFT stopping evaluation pipeline
# -----------------------------------------------------------------------------

def evaluate_sft_stop(
    adapter: Path,
    dataset_path: str = "chrivasileiou/asap7-language-of-test-v2",
    eval_buffer_size: int = 30,
    format_threshold: float = 0.95,
    diversity_threshold: float = 0.30,
    diversity_num_generations: int = 10,
    patience: int = 1,
    min_steps: int = 50,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_new_tokens: int = 8192,
    max_prompt_length: int = 4096,
    generation_batch_size: int = 8,
    loss_delta: float = 0.01,
    loss_window: int = 10,
    max_tool_rounds: int = 1,
    backend: str = "vllm",
    tp_size: int = 2,
    gpu_memory_utilization: float = 0.9,
    qlora: bool = False,
    output_file: Optional[str] = None,
    report_to: str = "none",
    seed: int = 42,
    wandb_run_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate saved SFT checkpoints against the stopping criteria from
    ``SFTStoppingCallback`` (format compliance + output diversity + loss
    plateau).

    Supports two modes depending on what *adapter* points to:

    * **Training output directory** — evaluates every ``checkpoint-N/``
      subdirectory in order, tracks consecutive passes, and reports which
      checkpoint (if any) first satisfies the stopping criteria.
    * **Single checkpoint** — evaluates one checkpoint and reports whether
      it meets the thresholds.

    Hard criteria (both must pass for *patience* consecutive evaluations):
        1. Format compliance  >= *format_threshold*  (default 95 %)
        2. Output diversity   >= *diversity_threshold* (default 30 %)

    Soft metric (informational):
        3. Loss plateau (read from ``trainer_state.json``)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    adapter = Path(adapter)

    # ----- discover checkpoints -----
    if (adapter / "adapter_config.json").exists():
        step_str = adapter.name.split("-", 1)[1] if adapter.name.startswith("checkpoint-") else "0"
        try:
            step_val = int(step_str)
        except ValueError:
            step_val = 0
        checkpoints = [(step_val, adapter)]
        output_dir = adapter.parent
    else:
        checkpoints = discover_checkpoints(adapter)
        output_dir = adapter

    if not checkpoints:
        raise FileNotFoundError(
            f"No valid checkpoints found under {adapter}. "
            f"Expected checkpoint-N/ directories with adapter_config.json."
        )

    print("=" * 70)
    print("SFT STOPPING CRITERIA EVALUATION")
    print("=" * 70)
    print(f"  Output directory  : {output_dir}")
    print(f"  Checkpoints found : {len(checkpoints)}")
    print(f"  Steps             : {[s for s, _ in checkpoints]}")
    print(f"  Backend           : {backend}")
    print(f"  Format threshold  : {format_threshold:.0%}")
    print(f"  Diversity thresh  : {diversity_threshold:.0%}")
    print(f"  Patience          : {patience}")
    print(f"  Min steps         : {min_steps}")
    print("=" * 70)

    # ----- tokenizer -----
    first_cfg_path = checkpoints[0][1] / "adapter_config.json"
    with open(first_cfg_path) as f:
        first_cfg = json.load(f)
    base_model_name = first_cfg["base_model_name_or_path"]

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # ----- eval prompts -----
    eval_prompts = load_sft_eval_prompts(
        tokenizer, dataset_path, eval_buffer_size, max_prompt_length,
    )
    if not eval_prompts:
        raise RuntimeError(
            "No eval prompts loaded. Check dataset_path and eval_buffer_size."
        )

    # ----- backend initialisation -----
    llm = None
    base_model_hf = None
    peft_model_hf = None
    sampling_config = None

    if backend == "vllm":
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        max_lora_rank = 8
        for _, ckpt in checkpoints:
            try:
                with open(ckpt / "adapter_config.json") as f:
                    r = json.load(f).get("r", 8)
                max_lora_rank = max(max_lora_rank, r)
            except Exception:
                pass

        quant_kwargs: Dict[str, Any] = {}
        if qlora:
            quant_kwargs = {"quantization": "bitsandbytes", "load_format": "bitsandbytes"}

        llm = LLM(
            model=base_model_name,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="bfloat16",
            enable_lora=True,
            max_lora_rank=max_lora_rank,
            trust_remote_code=True,
            disable_custom_all_reduce=should_disable_custom_all_reduce(tp_size),
            **quant_kwargs,
        )
        sampling_config = SamplingParams(
            n=1,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            stop_token_ids=[tokenizer.eos_token_id, tokenizer.pad_token_id],
        )
        print(
            f"[SFTStopEval] vLLM engine ready  (base={base_model_name}, "
            f"tp={tp_size}, max_lora_rank={max_lora_rank})"
        )
    else:
        from transformers import AutoModelForCausalLM

        print(f"[SFTStopEval] Loading base model: {base_model_name}")
        base_model_hf = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    # ----- wandb -----
    wandb_run = None
    if report_to == "wandb":
        try:
            import wandb
            wb_name = wandb_run_name or (
                wandb_run_name_from_adapter(adapter) + "_sft_stop"
            )
            wandb_run = wandb.init(
                project="atpg-sft-stop-eval",
                name=wb_name,
                tags=["sft_stop"],
                config={
                    "adapter": str(adapter),
                    "dataset": dataset_path,
                    "format_threshold": format_threshold,
                    "diversity_threshold": diversity_threshold,
                    "diversity_num_generations": diversity_num_generations,
                    "patience": patience,
                    "min_steps": min_steps,
                    "temperature": temperature,
                    "backend": backend,
                    "num_checkpoints": len(checkpoints),
                    "eval_buffer_size": eval_buffer_size,
                },
            )
        except ImportError:
            print("wandb not installed; skipping.")

    # =====================================================================
    # Evaluate each checkpoint
    # =====================================================================
    consecutive_passes = 0
    stop_checkpoint: Optional[int] = None
    all_results: List[Dict[str, Any]] = []
    eval_start = time.time()
    prev_hf_adapter: Optional[str] = None

    for ckpt_idx, (step, ckpt_path) in enumerate(checkpoints):
        print(f"\n{'=' * 60}")
        print(
            f"[SFTStopEval] Checkpoint {ckpt_idx + 1}/{len(checkpoints)}: "
            f"step {step}  ({ckpt_path.name})"
        )
        print(f"{'=' * 60}")

        if step < min_steps:
            msg = f"step {step} < min_steps {min_steps}"
            print(f"  Skipping: {msg}")
            all_results.append({
                "step": step, "checkpoint": str(ckpt_path),
                "skipped": True, "reason": msg,
            })
            continue

        ckpt_start = time.time()

        try:
            # ----- generate format-compliance completions -----
            print(
                f"  [Format] Generating 1 completion for "
                f"{len(eval_prompts)} prompts..."
            )
            if backend == "vllm":
                lora_req = LoRARequest(
                    f"ckpt_{step}", ckpt_idx + 1, str(ckpt_path),
                )
                fmt_completions = _sft_eval_generate_vllm(
                    llm, tokenizer, eval_prompts, 1,
                    sampling_config, lora_req, max_tool_rounds,
                )
            else:
                adapter_name = f"ckpt_{step}"
                if peft_model_hf is None:
                    peft_model_hf = PeftModel.from_pretrained(
                        base_model_hf, str(ckpt_path),
                        adapter_name=adapter_name,
                    )
                else:
                    peft_model_hf.load_adapter(
                        str(ckpt_path), adapter_name=adapter_name,
                    )
                    peft_model_hf.set_adapter(adapter_name)
                    if prev_hf_adapter is not None:
                        try:
                            peft_model_hf.delete_adapter(prev_hf_adapter)
                        except Exception:
                            pass
                prev_hf_adapter = adapter_name
                peft_model_hf.eval()

                fmt_completions = _sft_eval_generate_hf(
                    peft_model_hf, tokenizer, eval_prompts, 1,
                    max_new_tokens, temperature,
                    generation_batch_size, max_tool_rounds,
                )

            format_score, format_details = sft_check_format_compliance(
                fmt_completions,
            )

            # ----- generate diversity completions -----
            print(
                f"  [Diversity] Generating {diversity_num_generations} "
                f"completions for 1 prompt..."
            )
            if backend == "vllm":
                div_completions = _sft_eval_generate_vllm(
                    llm, tokenizer, [eval_prompts[0]],
                    diversity_num_generations,
                    sampling_config, lora_req, max_tool_rounds,
                )
            else:
                div_completions = _sft_eval_generate_hf(
                    peft_model_hf, tokenizer, [eval_prompts[0]],
                    diversity_num_generations,
                    max_new_tokens, temperature,
                    generation_batch_size, max_tool_rounds,
                )

            diversity_score, diversity_details = sft_check_diversity(
                div_completions,
            )

            # ----- loss plateau (soft) -----
            loss_plateaued = sft_check_loss_plateau(
                output_dir, step, loss_window, loss_delta,
            )

            ckpt_elapsed = time.time() - ckpt_start

            # ----- print results -----
            format_passed = format_score >= format_threshold
            diversity_passed = diversity_score >= diversity_threshold

            print(f"\n  Results at step {step}:")
            print(
                f"    Format compliance : {format_score:6.1%}  "
                f"{'PASS' if format_passed else 'FAIL'}  "
                f"(threshold: {format_threshold:.0%})"
            )
            t = format_details["total"]
            for key in [
                "think_ok", "tool_call_ok", "tool_call_json_ok",
                "input_vector_ok", "expected_output_ok",
                "detected_faults_ok", "fully_parseable",
            ]:
                label = key.replace("_ok", "").replace("_", " ").title()
                print(f"      {label:.<28s} {format_details[key]:>3d}/{t}")

            print(
                f"    Output diversity  : {diversity_score:6.1%}  "
                f"{'PASS' if diversity_passed else 'FAIL'}  "
                f"(threshold: {diversity_threshold:.0%})"
            )
            dd = diversity_details
            print(f"      Generations ......... {dd['total_generations']}")
            print(f"      Parseable ........... {dd['parseable_count']}")
            print(f"      Unique vectors ...... {dd['unique_input_vectors']}")
            print(f"      Unique texts ........ {dd['unique_texts']}")

            print(
                f"    Loss plateau      : "
                f"{'Yes' if loss_plateaued else 'No'}"
            )
            print(f"    Eval time         : {ckpt_elapsed:.1f}s")

            # ----- stopping criteria -----
            if format_passed and diversity_passed:
                consecutive_passes += 1
            else:
                consecutive_passes = 0

            criteria_met = consecutive_passes >= patience

            if criteria_met and stop_checkpoint is None:
                stop_checkpoint = step
                print(f"\n  {'*' * 56}")
                print(f"  * SFT STOPPING CRITERIA MET at step {step}")
                print(
                    f"  *   Format    : {format_score:.1%} "
                    f">= {format_threshold:.0%}"
                )
                print(
                    f"  *   Diversity : {diversity_score:.1%} "
                    f">= {diversity_threshold:.0%}"
                )
                print(f"  *   Patience  : {consecutive_passes}/{patience}")
                print(
                    f"  *   Loss plat.: "
                    f"{'Yes' if loss_plateaued else 'No'}"
                )
                print(f"  {'*' * 56}")
            elif format_passed and diversity_passed:
                remaining = patience - consecutive_passes
                print(
                    f"\n  Criteria passed ({consecutive_passes}/{patience}). "
                    f"Waiting for {remaining} more."
                )
            else:
                reasons = []
                if not format_passed:
                    reasons.append(
                        f"Format {format_score:.1%} < {format_threshold:.0%}"
                    )
                if not diversity_passed:
                    reasons.append(
                        f"Diversity {diversity_score:.1%} "
                        f"< {diversity_threshold:.0%}"
                    )
                print(f"\n  Not yet: {'; '.join(reasons)}")

            # ----- per-checkpoint result -----
            result: Dict[str, Any] = {
                "step": step,
                "checkpoint": str(ckpt_path),
                "format_score": round(format_score, 4),
                "format_threshold": format_threshold,
                "format_details": format_details,
                "format_passed": format_passed,
                "diversity_score": round(diversity_score, 4),
                "diversity_threshold": diversity_threshold,
                "diversity_details": {
                    k: v for k, v in diversity_details.items()
                    if k != "sample_vectors"
                },
                "diversity_passed": diversity_passed,
                "loss_plateaued": loss_plateaued,
                "consecutive_passes": consecutive_passes,
                "criteria_met": criteria_met,
                "eval_time_seconds": round(ckpt_elapsed, 2),
            }
            all_results.append(result)

            ckpt_result_path = ckpt_path / "sft_stopping_results.json"
            with open(ckpt_result_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"  Saved to {ckpt_result_path}")

            if wandb_run:
                import wandb
                wandb.log({
                    "step": step,
                    "sft_stop/format_score": format_score,
                    "sft_stop/diversity_score": diversity_score,
                    "sft_stop/loss_plateaued": int(loss_plateaued),
                    "sft_stop/format_passed": int(format_passed),
                    "sft_stop/diversity_passed": int(diversity_passed),
                    "sft_stop/consecutive_passes": consecutive_passes,
                    "sft_stop/criteria_met": int(criteria_met),
                })

        except Exception as exc:
            import traceback
            print(f"\n  ERROR evaluating checkpoint-{step}: {exc}")
            traceback.print_exc()
            consecutive_passes = 0
            all_results.append({
                "step": step,
                "checkpoint": str(ckpt_path),
                "error": str(exc),
            })

    total_time = time.time() - eval_start

    # ----- cleanup -----
    if llm is not None:
        del llm
    if peft_model_hf is not None:
        del peft_model_hf
    if base_model_hf is not None:
        del base_model_hf
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =====================================================================
    # Summary
    # =====================================================================
    evaluated = sum(
        1 for r in all_results
        if "skipped" not in r and "error" not in r
    )
    print(f"\n{'=' * 70}")
    print("SFT STOPPING EVALUATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Checkpoints evaluated : {evaluated}/{len(checkpoints)}")
    print(f"  Total eval time       : {total_time:.1f}s ({total_time / 60:.1f}m)")

    if stop_checkpoint is not None:
        print(f"\n  RECOMMENDED STOP: checkpoint-{stop_checkpoint}")
        print(
            f"    First checkpoint where format >= {format_threshold:.0%} AND "
            f"diversity >= {diversity_threshold:.0%}"
        )
        print(f"    for {patience} consecutive evaluation(s).")
    else:
        print("\n  NO STOPPING POINT FOUND")
        print(
            f"    No checkpoint met both thresholds for "
            f"{patience} consecutive eval(s)."
        )
        print(
            "    Consider: longer training, lower thresholds, or patience=1."
        )

    # Summary table
    print(
        f"\n  {'Step':>6s}  {'Format':>8s}  {'Divers':>8s}  "
        f"{'Plateau':>8s}  {'Consec':>6s}  Status"
    )
    print(
        f"  {'-' * 6}  {'-' * 8}  {'-' * 8}  "
        f"{'-' * 8}  {'-' * 6}  {'-' * 14}"
    )
    for r in all_results:
        if "skipped" in r:
            print(f"  {r['step']:>6d}  {'--':>8s}  {'--':>8s}  "
                  f"{'--':>8s}  {'--':>6s}  skipped")
        elif "error" in r:
            print(f"  {r['step']:>6d}  {'--':>8s}  {'--':>8s}  "
                  f"{'--':>8s}  {'--':>6s}  ERROR")
        else:
            loss_str = "Yes" if r["loss_plateaued"] else "No"
            status = "PASS" if r["criteria_met"] else "FAIL"
            marker = " <-- STOP" if (
                r["criteria_met"] and r["step"] == stop_checkpoint
            ) else ""
            print(
                f"  {r['step']:>6d}  {r['format_score']:>7.1%}  "
                f"{r['diversity_score']:>7.1%}  "
                f"{loss_str:>8s}  "
                f"{r['consecutive_passes']:>6d}  "
                f"{status}{marker}"
            )
    print("=" * 70)

    # ----- compile final results -----
    results: Dict[str, Any] = {
        "config": {
            "adapter": str(adapter),
            "dataset": dataset_path,
            "format_threshold": format_threshold,
            "diversity_threshold": diversity_threshold,
            "diversity_num_generations": diversity_num_generations,
            "patience": patience,
            "min_steps": min_steps,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "backend": backend,
            "eval_buffer_size": eval_buffer_size,
            "seed": seed,
        },
        "checkpoints": all_results,
        "stop_checkpoint_step": stop_checkpoint,
        "total_eval_time_seconds": round(total_time, 2),
    }

    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_file}")

    if wandb_run:
        import wandb
        wandb.log({"sft_stop/stop_checkpoint_step": stop_checkpoint or -1})
        wandb.finish()

    return results


# =============================================================================
# CLI
# =============================================================================

def _env(key: str, default=None):
    return os.environ.get(key, default)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate dual-adapter model (SFT + GRPO) using pass@k metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation with pass@1 (random i.i.d. baseline)
  python evaluate_model.py --adapter ./sft_finetuned_model/ --num_completions 1 --k 1

  # Full evaluation with pass@1,5,10 (random i.i.d. baseline)
  python evaluate_model.py \\
      --adapter ./finetuned_model/combined/policy/ \\
      --backend vllm \\
      --num_completions 10 --k 1 5 10 \\
      --temperature 0.6 \\
      --output eval_results.json

  # best_of_n: each of 20 completions is the best of 3 i.i.d. samples
  python evaluate_model.py \\
      --adapter ./finetuned_model/combined/policy/ --backend vllm \\
      --sampling_method best_of_n --num_completions 20 --n 3 --k 1 5 10

  # mcts: each of 20 completions is one MCTS search of budget 3
  python evaluate_model.py \\
      --adapter ./finetuned_model/combined/policy/ --backend vllm \\
      --sampling_method mcts --num_completions 20 --budget 3 --k 1 5 10

  # SFT stopping criteria evaluation (all checkpoints in a training dir)
  python evaluate_model.py \\
      --sft_stop \\
      --adapter ./sft_7b_exper1/ \\
      --backend vllm --tp_size 2 \\
      --format_threshold 0.95 --diversity_threshold 0.30 \\
      --patience 1 --min_steps 50 \\
      --output sft_stop_results.json

  # SFT stopping criteria evaluation (single checkpoint)
  python evaluate_model.py \\
      --sft_stop \\
      --adapter ./sft_7b_exper1/checkpoint-100/ \\
      --backend vllm --tp_size 2
        """,
    )
    
    parser.add_argument(
        "--adapter", 
        type=str, 
        default=_env("ADAPTER_CHECKPOINT"), 
        required=True,
        help="Path to the adapter checkpoint",
    )
    parser.add_argument( 
        "--dataset", 
        type=str, 
        default=_env("EVAL_DATASET", "chrivasileiou/asap7-language-of-test"), 
        help="HuggingFace dataset identifier",
    )
    parser.add_argument(
        "--num_completions",
        type=int,
        default=int(_env("NUM_COMPLETIONS", _env("NUM_SAMPLES", "10"))),
        help="Completions per problem for pass@k (the pass@k pool, N). Each "
             "completion is one independent application of --sampling_method.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="best_of_n only: i.i.d. samples drawn per completion, of which "
             "the best is kept (the 'n' in best-of-n; per-completion width).",
    )
    parser.add_argument(
        "--k", 
        type=int, 
        nargs="+", 
        default=[1, 5, 10], 
        help="k values for pass@k (e.g., --k 1 5 10)",
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=float(_env("TEMPERATURE", "0.6")), 
        help="Generation temperature",
    )
    parser.add_argument(
        "--top_p", 
        type=float, 
        default=float(_env("TOP_P", "0.95")), 
        help="Nucleus sampling threshold"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=int(_env("MAX_NEW_TOKENS", "16384")), 
        help="Maximum new tokens per generation"
    )
    parser.add_argument(
        "--max_eval_samples", 
        type=int, 
        default=int(_env("MAX_EVAL_SAMPLES", "-1")), 
        help="Maximum eval samples (-1 for all)",
    )
    parser.add_argument(
        "--max_tool_rounds", 
        type=int, 
        default=int(_env("MAX_TOOL_ROUNDS", "1")), 
        help="Maximum tool call rounds per generation"
    )
    parser.add_argument(
        "--threshold_mode", 
        type=str, 
        default=_env("THRESHOLD_MODE", "fault_detected"), 
        choices=["fault_detected", "positive_reward", "full_accuracy"], 
        help="Correctness threshold mode for pass@k"
    )
    parser.add_argument(
        "--config_path", 
        type=str, 
        default=_env("SIM_CONFIG", "sim_config.json"), 
        help="Path to sim_config.json"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=int(_env("SEED", "42")), 
        help="Random seed"
    )
    parser.add_argument(
        "--report_to", 
        type=str, 
        default=_env("REPORT_TO", "none"), 
        choices=["wandb", "none"], 
        help="Reporting backend"
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=_env("WANDB_RUN_NAME"),
        help="Weights & Biases run display name (default: derived from --adapter).",
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default=_env("OUTPUT_FILE"), 
        help="Output file for detailed results (JSON)"
    )
    
    # Backend Arguments
    parser.add_argument(
        "--backend", 
        type=str, 
        default="transformers", 
        choices=["transformers", "vllm"], 
        help="Inference engine backend to use."
    )
    parser.add_argument(
        "--tp_size", 
        type=int, 
        default=len(_env("CUDA_VISIBLE_DEVICES", "1").split(",")), 
        help="Tensor parallel size (number of GPUs) for vLLM."
    )
    parser.add_argument(
        "--gpu_memory_utilization", 
        type=float, 
        default=0.7, 
        help="GPU memory utilization for vLLM."
    )
    parser.add_argument(
        "--qlora", 
        action="store_true", 
        help="Use if the base model is a BitsAndBytes quantized model (4-bit/8-bit)."
    )
    parser.add_argument(
        "--merge_dequant",
        action="store_true",
        help="vLLM backend: serve the QLoRA-faithful merged bf16 model "
             "(dequantize_4bit(W_nf4) + B*A*scaling, exactly what GRPO training "
             "pushed to its vLLM server) instead of mounting the LoRA on the "
             "clean bf16 Hub base. The export is built once and cached as "
             "<adapter>_merged_bf16/ next to the adapter.",
    )
    parser.add_argument(
        "--eval_prompt_batch_size",
        type=int,
        default=int(_env("EVAL_PROMPT_BATCH_SIZE", "8")),
        help="Number of dataset prompts to generate in each fused batch (default: 8). "
             "Use 1 to mimic the old one-prompt-at-a-time behavior.",
    )
    parser.add_argument(
        "--generation_micro_batch_size",
        type=int,
        default=int(_env("GENERATION_MICRO_BATCH_SIZE", "8")),
        help="Transformers backend: max parallel sequences per generate() call "
             "when expanding to batch_size * num_completions paths (default: 8).",
    )
    parser.add_argument(
        "--sampling_method",
        type=str,
        default=_env("SAMPLING_METHOD", "random"),
        choices=["random"] + list_available_strategies(),
        help=(
            "Inference-time search strategy. 'random' (default) uses the "
            "existing tool-calling pipeline. Other strategies bypass model "
            "tool calls and use the external fault simulator as the "
            "verifier / oracle (see sampling_strategies.py)."
        ),
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=int(_env("SEARCH_BUDGET")) if _env("SEARCH_BUDGET") else None,
        help="mcts/evolutionary only: per-completion search width (B). Scored "
             "rollouts (mcts) or completions evaluated (evolutionary) per "
             "independent search. Orthogonal to --num_completions.",
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=4096,
        help="Max prompt token length for eval buffering (default: 4096).",
    )

    # SFT Stopping Criteria Arguments
    sft_group = parser.add_argument_group(
        "SFT stopping criteria",
        "Evaluate checkpoints against SFTStoppingCallback criteria "
        "(format compliance + output diversity). Activated by --sft_stop.",
    )
    sft_group.add_argument(
        "--sft_stop",
        action="store_true",
        help="Evaluate SFT stopping criteria across saved checkpoints "
             "instead of pass@k. --adapter should point to the training "
             "output directory (containing checkpoint-N/ subdirs) or a "
             "single checkpoint.",
    )
    sft_group.add_argument(
        "--format_threshold",
        type=float,
        default=0.95,
        help="Min format compliance fraction (default: 0.95).",
    )
    sft_group.add_argument(
        "--diversity_threshold",
        type=float,
        default=0.30,
        help="Min output diversity fraction (default: 0.30).",
    )
    sft_group.add_argument(
        "--diversity_num_generations",
        type=int,
        default=10,
        help="Number of completions for the diversity check (default: 10).",
    )
    sft_group.add_argument(
        "--patience",
        type=int,
        default=1,
        help="Consecutive passing evaluations required before declaring "
             "stop (default: 1).",
    )
    sft_group.add_argument(
        "--min_steps",
        type=int,
        default=50,
        help="Skip checkpoints before this training step (default: 50).",
    )
    sft_group.add_argument(
        "--eval_buffer_size",
        type=int,
        default=30,
        help="Number of eval prompts to buffer from the test split "
             "(default: 30).",
    )
    sft_group.add_argument(
        "--generation_batch_size",
        type=int,
        default=8,
        help="Batch size for generation during eval (default: 8).",
    )
    sft_group.add_argument(
        "--loss_delta",
        type=float,
        default=0.01,
        help="Minimum loss improvement to not flag plateau (default: 0.01).",
    )
    sft_group.add_argument(
        "--loss_window",
        type=int,
        default=10,
        help="Number of log entries for plateau detection (default: 10).",
    )

    args = parser.parse_args()
    if not args.config_path.startswith("/"):
        args.config_path = Path(__file__).resolve().parent / args.config_path
    else:
        args.config_path = Path(args.config_path)

    if args.sft_stop:
        evaluate_sft_stop(
            adapter=Path(args.adapter),
            dataset_path=args.dataset,
            eval_buffer_size=args.eval_buffer_size,
            format_threshold=args.format_threshold,
            diversity_threshold=args.diversity_threshold,
            diversity_num_generations=args.diversity_num_generations,
            patience=args.patience,
            min_steps=args.min_steps,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            max_prompt_length=args.max_prompt_length,
            generation_batch_size=args.generation_batch_size,
            loss_delta=args.loss_delta,
            loss_window=args.loss_window,
            max_tool_rounds=args.max_tool_rounds,
            backend=args.backend,
            tp_size=args.tp_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            qlora=args.qlora,
            output_file=args.output_file,
            report_to=args.report_to,
            seed=args.seed,
            wandb_run_name=args.wandb_run_name or None,
        )
    else:
        evaluate(
            adapter=Path(args.adapter),
            dataset_path=args.dataset,
            num_completions=args.num_completions,
            k_values=args.k,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            max_prompt_length=args.max_prompt_length,
            max_eval_samples=args.max_eval_samples,
            max_tool_rounds=args.max_tool_rounds,
            threshold_mode=args.threshold_mode,
            config_path=args.config_path,
            seed=args.seed,
            report_to=args.report_to,
            output_file=args.output_file,
            backend=args.backend,
            tp_size=args.tp_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            qlora=args.qlora,
            eval_prompt_batch_size=args.eval_prompt_batch_size,
            generation_micro_batch_size=args.generation_micro_batch_size,
            wandb_run_name=args.wandb_run_name or None,
            sampling_method=args.sampling_method,
            merge_dequant=args.merge_dequant,
            budget=args.budget,
            best_of_n_width=args.n,
        )

if __name__ == "__main__":
    main()