"""
evaluate_model.py
=================

Evaluation script for dual-adapter models (SFT + GRPO) using pass@k metrics.

This module evaluates a model trained through SFT → GRPO by:
1. Loading the dual-adapter model via load_dual_adapter_model()
2. Loading the eval split of `chrivasileiou/asap7-language-of-test`
3. For each prompt, generating N completions (with tool-calling support)
4. Executing tool calls (fault simulation) when the model requests them
5. Computing rewards via RewardFunctionFactory
6. Calculating pass@k metrics (pass@1, pass@5, pass@10, etc.)

The pass@k metric (from the Codex paper, Chen et al. 2021) estimates:
    pass@k = E[1 - C(n-c, k) / C(n, k)]
where n = total completions per problem, c = correct completions.

Usage:
    python evaluate_model.py \\
        --sft_checkpoint ./sft_finetuned_model \\
        --policy_checkpoint ./grpo_finetuned_model/policy_adapter \\
        --dataset chrivasileiou/asap7-language-of-test \\
        --n 10 --k 1 5 10 \\
        --temperature 0.6

Environment Variables:
    SFT_CHECKPOINT: Path to SFT adapter checkpoint
    POLICY_CHECKPOINT: Path to policy adapter checkpoint
    EVAL_DATASET: Dataset identifier (default: chrivasileiou/asap7-language-of-test)
    NUM_SAMPLES: Number of completions per prompt (default: 10)
    BATCH_SIZE: Batch size for generation (default: 4)
    MAX_EVAL_SAMPLES: Maximum number of eval samples (default: -1 for all)
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
from reward_function_factory import RewardFunctionFactory
from tools import TOOLS, FAULT_SIMULATION_TOOL, fault_simulation_tool
from revert_template import revert_qwen2_5_template

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
        Parsed tool call dict with 'name' and 'arguments', or None.
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
            result = loop.run_until_complete(fault_simulation_tool(**tool_args))
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
                # Execute the tool
                tool_result = execute_tool_call(tool_call)
                
                # Build the continuation with tool result
                # Parse the current conversation, add tool result, and prepare for next generation
                messages = revert_qwen2_5_template(current_input)
                
                # Add the assistant message (with tool call)
                messages.append({"role": "assistant", "content": completion_text})
                
                # Add tool result
                messages.append({
                    "role": "tool",
                    "name": tool_call.get("name", "fault_simulation_tool"),
                    "content": tool_result,
                })
                
                # Re-format with chat template
                current_input = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    tools=TOOLS,
                    add_generation_prompt=True,
                )
                
                # Add tool response to full completion for reward evaluation
                full_completion += f"\n<tool_response>\n{tool_result}\n</tool_response>\n"
                continue
        
        break  # No tool call or max rounds reached
    
    return full_completion


def generate_n_completions(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    n: int,
    generation_config: GenerationConfig,
    max_tool_rounds: int = 1,
) -> List[str]:
    """
    Generate n completions for a single prompt.
    
    Parameters
    ----------
    model : PeftModel
        The model.
    tokenizer : AutoTokenizer
        The tokenizer.
    prompt_text : str
        The formatted prompt.
    n : int
        Number of completions to generate.
    generation_config : GenerationConfig
        Generation configuration.
    max_tool_rounds : int
        Maximum tool call rounds per completion.
    
    Returns
    -------
    List[str]
        List of n completion strings.
    """
    completions = []
    for _ in range(n):
        completion = generate_with_tools(
            model, tokenizer, prompt_text, generation_config, max_tool_rounds
        )
        completions.append(completion)
    return completions


# =============================================================================
# PROMPT FORMATTING
# =============================================================================

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
    convo = ConversationExample.from_record(dict(record), use_tools=use_tools)
    
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
    from fault_sim import fast_fault_sim
    
    # Build kwargs
    netlists = []
    for record in records:
        netlist_str = record.get('netlist', '')
        try:
            netlists.append(reward_factory.get_or_create_netlist(netlist_str))
        except Exception:
            netlists.append(netlist_str)
    
    # Build fault kwargs from records
    faults = [record.get('fault', '') for record in records]
    
    reward_kwargs = {
        "fault_fn": lambda x, **kw: RewardFunctionFactory.fault_fn(x, **kw),
        "simulation_fn": RewardFunctionFactory.simulation_fn,
        "input_vector_fn": RewardFunctionFactory.input_vector_fn,
        "expected_output_fn": RewardFunctionFactory.expected_output_fn,
        "detected_faults_fn": RewardFunctionFactory.detected_faults_fn,
        "eval_mode": True,
        "lib_gate_funcs": reward_factory.gate_funcs,
        "fault_sim": fast_fault_sim,
        "netlists": netlists,
        "fault": faults,
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
    if threshold_mode == "fault_detected":
        return reward.get('fault_detected_by_pred_input_vector_acc', 0) == 1
    elif threshold_mode == "positive_reward":
        return sum(reward.values()) > 0
    elif threshold_mode == "full_accuracy":
        return (
            reward.get('fault_detected_by_pred_input_vector_acc', 0) == 1 and
            reward.get('input_vector_acc', 0) == 1 and
            reward.get('expected_output_acc', 0) == 1 and
            reward.get('detected_faults_acc', 0) == 1
        )
    else:
        raise ValueError(f"Unknown threshold mode: {threshold_mode}")


# =============================================================================
# MAIN EVALUATION PIPELINE
# =============================================================================

def evaluate(
    sft_checkpoint: str,
    policy_checkpoint: Optional[str] = None,
    dataset_path: str = "chrivasileiou/asap7-language-of-test",
    n: int = 10,
    k_values: List[int] = None,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_new_tokens: int = 4096,
    max_eval_samples: int = -1,
    max_tool_rounds: int = 1,
    threshold_mode: str = "fault_detected",
    config_path: str = "sim_config.json",
    seed: int = 42,
    report_to: str = "none",
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main evaluation function.
    
    Parameters
    ----------
    sft_checkpoint : str
        Path to SFT adapter checkpoint.
    policy_checkpoint : str, optional
        Path to policy adapter checkpoint.
    dataset_path : str
        HuggingFace dataset identifier.
    n : int
        Number of completions per prompt (for pass@k estimation).
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
    
    Returns
    -------
    Dict[str, Any]
        Results dictionary with pass@k scores and detailed metrics.
    """
    if k_values is None:
        k_values = [1, 5, 10]
    
    # Validate k values
    for k in k_values:
        if k > n:
            raise ValueError(f"k={k} > n={n}. Cannot compute pass@{k} with only {n} samples per prompt.")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # =========================================================================
    # 1. Load Model
    # =========================================================================
    print("=" * 70)
    print("LOADING MODEL")
    print("=" * 70)
    
    model = load_dual_adapter_model(
        sft_checkpoint_path=sft_checkpoint,
        policy_checkpoint_path=policy_checkpoint,
    )
    model.eval()
    
    # Load tokenizer from the SFT checkpoint's base model
    with open(os.path.join(sft_checkpoint, "adapter_config.json"), "r") as f:
        adapter_config = json.load(f)
    base_model_name = adapter_config["base_model_name_or_path"]
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # For generation
    
    print(f"Model loaded from: SFT={sft_checkpoint}, Policy={policy_checkpoint}")
    print(f"Base model: {base_model_name}")
    print(f"Active adapter: {model.active_adapter}")
    model.print_trainable_parameters()
    
    # Set up generation config
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # =========================================================================
    # 2. Load Dataset
    # =========================================================================
    print("\n" + "=" * 70)
    print("LOADING DATASET")
    print("=" * 70)
    
    eval_dataset = load_dataset(dataset_path, split="eval", streaming=True)
    
    # Buffer the streaming dataset
    eval_records = []
    for i, record in enumerate(tqdm(eval_dataset, desc="Loading eval data")):
        if max_eval_samples > 0 and i >= max_eval_samples:
            break
        eval_records.append(record)
    
    if not eval_records:
        raise ValueError("No eval records loaded. Check dataset path and split name.")
    
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
            wandb_run = wandb.init(
                project="atpg-eval",
                config={
                    "sft_checkpoint": sft_checkpoint,
                    "policy_checkpoint": policy_checkpoint,
                    "dataset": dataset_path,
                    "n_completions": n,
                    "k_values": k_values,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_new_tokens": max_new_tokens,
                    "threshold_mode": threshold_mode,
                    "num_eval_samples": len(eval_records),
                },
            )
        except ImportError:
            print("wandb not installed; skipping wandb logging")
    
    # =========================================================================
    # 5. Generation & Evaluation Loop
    # =========================================================================
    print("\n" + "=" * 70)
    print(f"EVALUATING: n={n}, k={k_values}, temperature={temperature}")
    print(f"Threshold mode: {threshold_mode}")
    print("=" * 70)
    
    all_results = []  # Per-problem results
    all_num_correct = []  # Number of correct completions per problem
    all_rewards = []  # Detailed rewards per problem per completion
    
    eval_start_time = time.time()
    
    for idx, record in enumerate(tqdm(eval_records, desc="Evaluating")):
        problem_start = time.time()
        
        # Format the prompt
        try:
            prompt_text = format_eval_prompt(record, tokenizer)
        except Exception as e:
            print(f"Warning: Failed to format prompt for sample {idx}: {e}")
            all_num_correct.append(0)
            all_results.append({
                "idx": idx,
                "fault": record.get("fault", ""),
                "module_name": record.get("module_name", ""),
                "error": f"prompt_format_error: {e}",
                "num_correct": 0,
                "n": n,
            })
            continue
        
        # Generate n completions
        completions = generate_n_completions(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            n=n,
            generation_config=generation_config,
            max_tool_rounds=max_tool_rounds,
        )
        
        # Evaluate each completion
        problem_rewards = []
        num_correct = 0
        
        for comp_idx, completion in enumerate(completions):
            try:
                rewards = evaluate_completions(
                    reward_factory=reward_factory,
                    prompts=[prompt_text],
                    completions=[completion],
                    records=[record],
                )
                reward = rewards[0]
            except Exception as e:
                reward = {
                    'format': 0, 'pred_simulation': 0, 'fault_simulation': 0,
                    'input_vector': 0, 'expected_output': 0, 'detected_faults': 0,
                    'fault_detect_inpvector': 0, 'pred_vs_fault_sim_acc': 0,
                    'fault_detected_by_pred_input_vector_acc': 0,
                    'expected_output_acc': 0, 'input_vector_acc': 0,
                    'detected_faults_acc': 0
                }
            
            problem_rewards.append(reward)
            if is_completion_correct(reward, threshold_mode=threshold_mode):
                num_correct += 1
        
        all_num_correct.append(num_correct)
        all_rewards.append(problem_rewards)
        
        problem_time = time.time() - problem_start
        
        # Store per-problem result
        problem_result = {
            "idx": idx,
            "fault": record.get("fault", ""),
            "module_name": record.get("module_name", ""),
            "num_correct": num_correct,
            "n": n,
            "time_seconds": round(problem_time, 2),
            "rewards_summary": {
                "mean_total_reward": np.mean([sum(r.values()) for r in problem_rewards]),
                "fault_detection_rate": num_correct / n,
                "mean_format_reward": np.mean([r.get('format', 0) for r in problem_rewards]),
                "mean_fault_sim_reward": np.mean([r.get('fault_simulation', 0) for r in problem_rewards]),
            },
        }
        all_results.append(problem_result)
        
        # Log to wandb periodically
        if wandb_run and (idx + 1) % 10 == 0:
            running_pass_at_k = {}
            for k in k_values:
                if k <= n:
                    pass_k = estimate_pass_at_k(
                        np.array([n] * len(all_num_correct)),
                        np.array(all_num_correct),
                        k,
                    )
                    running_pass_at_k[f"running_pass@{k}"] = np.mean(pass_k)
            wandb.log({
                "eval_step": idx + 1,
                **running_pass_at_k,
                "running_fault_detection_rate": np.mean([r["rewards_summary"]["fault_detection_rate"] for r in all_results if "error" not in r]),
            })
        
        # Print progress every 10 samples
        if (idx + 1) % 10 == 0:
            running_rate = np.mean([r["rewards_summary"]["fault_detection_rate"] for r in all_results if "error" not in r])
            print(f"  [{idx+1}/{len(eval_records)}] Running fault detection rate: {running_rate:.3f}")
    
    eval_time = time.time() - eval_start_time
    
    # =========================================================================
    # 6. Compute pass@k Metrics
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPUTING PASS@K METRICS")
    print("=" * 70)
    
    num_samples_arr = np.array([n] * len(all_num_correct))
    num_correct_arr = np.array(all_num_correct)
    
    pass_at_k_results = {}
    for k in k_values:
        if k <= n:
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
        "n_completions_per_prompt": n,
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
            "sft_checkpoint": sft_checkpoint,
            "policy_checkpoint": policy_checkpoint,
            "dataset": dataset_path,
            "n": n,
            "k_values": k_values,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "max_tool_rounds": max_tool_rounds,
            "threshold_mode": threshold_mode,
            "seed": seed,
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
    print(f"Completions per sample: {n}")
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
  # Basic evaluation with pass@1
  python evaluate_model.py --sft_checkpoint ./sft_model --n 1 --k 1

  # Full evaluation with pass@1,5,10
  python evaluate_model.py \\
      --sft_checkpoint ./sft_model \\
      --policy_checkpoint ./grpo_model/policy_adapter \\
      --n 10 --k 1 5 10 \\
      --temperature 0.6 \\
      --output eval_results.json

  # Quick evaluation on a subset
  python evaluate_model.py \\
      --sft_checkpoint ./sft_model \\
      --max_eval_samples 50 --n 5 --k 1 5
        """,
    )
    
    parser.add_argument(
        "--sft_checkpoint", type=str, 
        default=_env("SFT_CHECKPOINT"),
        help="Path to SFT adapter checkpoint",
    )
    parser.add_argument(
        "--policy_checkpoint", type=str, 
        default=_env("POLICY_CHECKPOINT"),
        help="Path to policy adapter checkpoint (optional, for GRPO-trained models)",
    )
    parser.add_argument(
        "--dataset", type=str,
        default=_env("EVAL_DATASET", "chrivasileiou/asap7-language-of-test"),
        help="HuggingFace dataset identifier",
    )
    parser.add_argument(
        "--n", type=int, 
        default=int(_env("NUM_SAMPLES", "10")),
        help="Number of completions per prompt",
    )
    parser.add_argument(
        "--k", type=int, nargs="+", 
        default=[1, 5, 10],
        help="k values for pass@k (e.g., --k 1 5 10)",
    )
    parser.add_argument(
        "--temperature", type=float, 
        default=float(_env("TEMPERATURE", "0.6")),
        help="Generation temperature",
    )
    parser.add_argument(
        "--top_p", type=float, 
        default=float(_env("TOP_P", "0.95")),
        help="Nucleus sampling threshold",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, 
        default=int(_env("MAX_NEW_TOKENS", "4096")),
        help="Maximum new tokens per generation",
    )
    parser.add_argument(
        "--max_eval_samples", type=int, 
        default=int(_env("MAX_EVAL_SAMPLES", "-1")),
        help="Maximum eval samples (-1 for all)",
    )
    parser.add_argument(
        "--max_tool_rounds", type=int, 
        default=int(_env("MAX_TOOL_ROUNDS", "1")),
        help="Maximum tool call rounds per generation",
    )
    parser.add_argument(
        "--threshold_mode", type=str,
        default=_env("THRESHOLD_MODE", "fault_detected"),
        choices=["fault_detected", "positive_reward", "full_accuracy"],
        help="Correctness threshold mode for pass@k",
    )
    parser.add_argument(
        "--config_path", type=str,
        default=_env("SIM_CONFIG", "sim_config.json"),
        help="Path to sim_config.json",
    )
    parser.add_argument(
        "--seed", type=int, 
        default=int(_env("SEED", "42")),
        help="Random seed",
    )
    parser.add_argument(
        "--report_to", type=str,
        default=_env("REPORT_TO", "none"),
        choices=["wandb", "none"],
        help="Reporting backend",
    )
    parser.add_argument(
        "--output", type=str,
        default=_env("OUTPUT_FILE"),
        help="Output file for detailed results (JSON)",
    )
    
    args = parser.parse_args()
    
    if args.sft_checkpoint is None:
        parser.error("--sft_checkpoint is required (or set SFT_CHECKPOINT env var)")
    
    results = evaluate(
        sft_checkpoint=args.sft_checkpoint,
        policy_checkpoint=args.policy_checkpoint,
        dataset_path=args.dataset,
        n=args.n,
        k_values=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        max_eval_samples=args.max_eval_samples,
        max_tool_rounds=args.max_tool_rounds,
        threshold_mode=args.threshold_mode,
        config_path=args.config_path,
        seed=args.seed,
        report_to=args.report_to,
        output_file=args.output,
    )
    
    return results


if __name__ == "__main__":
    main()
