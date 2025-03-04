import os
import torch.distributed as dist
import torch
import time
import torch.nn as nn
import itertools
import argparse
import wandb
import gc
import pandas as pd
import numpy as np
import regex as re
import multiprocessing as mp
from peft import PeftModel
from tqdm.auto import tqdm
from tabulate import tabulate
from collections import defaultdict
from transformers import set_seed, AutoTokenizer, get_cosine_schedule_with_warmup
from sentence_transformers import SentenceTransformer
from accelerate.utils import gather
from peft import get_peft_model, LoraConfig, TaskType, get_peft_model_state_dict
from torch.optim import AdamW
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from contextlib import contextmanager
from atpgllm import (
  AttrDict,
  parse_arguments, 
  hyperparameters,
  models_causal,
  load_model,
  load_tokenizer,
  load_raw_dataset,
  prepare_objects_for_training,
  initialize_training_environment,
  get_targets,
  cot_reward,
  test_generation_reward,
  get_user_prompt,
  train_layers,
  is_main_process,
  convert_bytes,
  model_size_in_bytes,
  infer, 
  save_model,
  MyCollate,
  get_trainable_parameters,
)
from typing import List, Union, Callable, Dict, Any
from transformers import Trainer

DEBUG = False

def sft(dataloader, model, hps, desc:str = "SFT Training...", training_loop:bool = True):
  # Retrieve optimizer from hyperparameters dict
  optimizer = hps.get('optimizer', torch.optim.AdamW(model.parameters(), lr=hps.lr))
  scheduler = hps.get('scheduler', None)
  device    = hps.get('device', torch.device(hps.free_gpu_id))
  criterion = hps.get('criterion', nn.CrossEntropyLoss(ignore_index=hps.tokenizer.pad_token_id if hps.is_causal else -100).to(device))
  tokenizer = hps.get('tokenizer', AutoTokenizer.from_pretrained(hps.model_name))
  is_causal = hps.get('is_causal', True)

  # TODO: add it as a hyperparameter
  logging_steps = 10
  # logging iterator 
  logging_step = 0
  # Set collate function's flags to supervised fine-tuning
  dataloader.collate_fn.set_right_padding()

  # Log hyperparameters to wandb
  if hps.wandb:
    wandb.log({'sft/hps/lr': hps.lr, 'sft/hps/micro_batch_size': hps.micro_batch_size, 'sft/hps/gradient_accumulation_steps': hps.gradient_accumulation_steps})

  if is_main_process():
    pbar = tqdm(total=len(dataloader), desc=f"[{hps.local_rank}]: {desc}", disable=hps.disable_tqdm)
  # Get dataloader iterator
  data_iterator = iter(itertools.cycle(dataloader))

  metrics = defaultdict(list)
  if is_main_process():
    print(f"Optimizer's step every: {hps.gradient_accumulation_steps}")
  
  for epoch in range(hps.epochs):
    if is_main_process():
      print(f"Epoch {epoch+1}/{hps.epochs}:")
      pbar.reset()
    for i in range(0, len(dataloader)):
      # Initialize gradient accumulation state
      if i % hps.gradient_accumulation_steps == 0:
        batch_loss = 0
        # zero the parameter gradients
        optimizer.zero_grad()

      # Get next data
      data = next(data_iterator)

      # IDs and Attention Mask
      ids  = data['input_ids'].to(device, non_blocking=True)
      mask = data['attention_mask'].to(device, non_blocking=True)

      # Get targets
      targets = get_targets(data, tokenizer, is_causal, device)

      # Forward Pass
      outputs = model(input_ids=ids, attention_mask=mask)

      # Compute loss
      micro_batch_loss = criterion(outputs.logits.transpose(2, 1).to(torch.bfloat16), targets)    
      micro_batch_loss /= hps.gradient_accumulation_steps
      
      batch_loss += micro_batch_loss.item()
      if training_loop:
        if torch.isnan(micro_batch_loss).any():
          raise ValueError(f"The gradients are vanished/exploded. The loss is {micro_batch_loss.item()}")
        
        # Calculate/Accumulate the gradients
        hps.accelerator.backward(micro_batch_loss) if hps.deepspeed_kernel else micro_batch_loss.backward()
      
        # Log metrics
        if (i+1) % hps.gradient_accumulation_steps == 0 or i==len(dataloader)-1:
          # Gradient accumulation
          optimizer.step()
          # schedule the learning rate based on the model loss
          if scheduler:
            scheduler.step() if hps.deepspeed_kernel else scheduler.step(batch_loss)

      if (i+1) % hps.gradient_accumulation_steps == 0:
        metrics["epoch"].append(str(epoch+1))
        metrics["batch"].append(str(int(i//hps.gradient_accumulation_steps)+1))
        metrics['step'].append(str(i+1))
        metrics['batch_loss'].append(smart_round(batch_loss))
        x = pd.DataFrame(metrics)

        if hps.wandb: 
          # Log to wandb as both metrics and table
          wandb.log({
              **{f"sft/{k}": float(eval(metrics[k][-1])) for k in metrics.keys()}
          })

        if is_main_process():
          pbar.set_postfix(x.iloc[-1].to_dict())
          if logging_step % logging_steps == 0:
            print(tabulate(x.iloc[-logging_steps:], headers='keys', tablefmt='psql', showindex=False))
            logging_step=0
          logging_step+=1
      if DEBUG and i>10:
        break
      
      if is_main_process():
        pbar.update(1)
    
  x = pd.DataFrame(metrics)
  # Log final complete table
  if hps.wandb: 
    x = x.astype(float)
    wandb.log({"sft_complete_history": wandb.Table(dataframe=x)})
  if is_main_process():
    print(tabulate(x, headers='keys', tablefmt='psql', showindex=False))
    pbar.close()

def parse_output_to_dict(output_str):
  sections = ["CHAIN_OF_THOUGHT", "SNAPSHOT", "INPUT_VECTOR", "EXPECTED_OUTPUT", "DETECTED_FAULTS"]
  output_dict = {}  
  for i, section in enumerate(sections):
    start_idx = output_str.find(section)
    if start_idx != -1:
      end_idx = output_str.find(sections[i+1], start_idx) if i != len(sections) - 1 else -1
      if end_idx == -1:
        end_idx = len(output_str)
      content = output_str[start_idx + len(section) + 1:end_idx].strip()
      output_dict[section] = content
  return output_dict

# Get the per-token log probabilities for the completions for the model and the reference model
def get_per_token_logps(model, input_ids, attention_mask, logits_to_keep):
  # NOTE: Be careful with num_logits_to_keep. 
  # It should be num_logits_to_keep=logits_to_keep+1 only if transformers <= 4.48. 
  # It should be logits_to_keep=logits_to_keep+1 only if transformers >= 4.49. 
  logits = model(input_ids=input_ids, attention_mask=attention_mask, num_logits_to_keep=logits_to_keep+1).logits # (B, L, V)
  logits = logits[:, :-1, :] # (B, L-1, V) exclude the last logit: it corresponds to the next token pred  
  input_ids = input_ids[:, -logits_to_keep:] # Keep completion ids
  
  # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
  per_token_logps = []
  for logits_row, input_ids_row in zip(logits, input_ids):
    log_probs = logits_row.log_softmax(dim=-1)
    token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(-1)).squeeze(-1)
    per_token_logps.append(token_log_prob)
  return torch.stack(per_token_logps)

def prepare_reward_kwargs(hps):
  cot_block_re = re.compile(r'CHAIN_OF_THOUGHT:\n(.*?)SNAPSHOT', re.DOTALL)
  thought_pattern_re = re.compile(r'(\d+)\.(.*?)(?=\d+\.|$)', re.DOTALL)
  fault_re = re.compile(r"(sa\d)\s+(_\d+_)", re.DOTALL)
  simulation_re = re.compile(r"SNAPSHOT:\n```\n(.*?)```\s+INPUT_VECTOR", re.DOTALL)
  input_vector_re = re.compile(r"INPUT_VECTOR:\s\"(.*?)\"", re.DOTALL)
  expected_output_re = re.compile(r"EXPECTED_OUTPUT:\s\"(.*?)\"", re.DOTALL)
  detected_faults_re = re.compile(r"DETECTED_FAULTS:\s\"(.*?)\"", re.DOTALL)

  # Prepare the reward function's arguments
  sentence_transformer = SentenceTransformer('paraphrase-MiniLM-L6-v2').to(hps.device)
  reward_kwargs = [{"model": sentence_transformer, "cot_block_re": cot_block_re, "thought_pattern_re": thought_pattern_re, "fault_re": fault_re},
                   {"fault_re": fault_re, "simulation_re": simulation_re, "input_vector_re": input_vector_re, "expected_output_re": expected_output_re, "detected_faults_re": detected_faults_re}]
  return reward_kwargs

def smart_round(value, sig_figs=3):
  return f"{value:.{sig_figs}g}" if value != 0 else "0"

@contextmanager
def unwrap_ddp(model):
  yield model.module if hasattr(model, "module") else model

@contextmanager
def disable_ref_adapter(model):
  for name, param in model.named_parameters():
    param.requires_grad = False
  yield model

def apply_lora_distributed(model, peft_model, lora_config, adapter_name="default"):
  # For distributed training, apply LoRA to the module of the DDP-wrapped model
  model.module = peft_model
  # Add a new LoRA adapter to the model
  model.module.add_adapter(peft_config=lora_config, adapter_name=adapter_name)
  # Activate the newly added LoRA adapter
  model.module.set_adapter(adapter_name)
  return model

def apply_lora_non_distributed(model, peft_model, lora_config, adapter_name="default"):
  # For non-distributed training, apply LoRA directly to the model
  model = peft_model
  # Add a new LoRA adapter to the model
  model.add_adapter(peft_config=lora_config, adapter_name=adapter_name)
  # Activate the newly added LoRA adapter
  model.set_adapter(adapter_name)
  return model


def compute_loss(model, data, reward_funcs, reward_kwargss, hps):
  """
  Compute the GRPO loss for a batch of inputs.
  
  Args:
      model: The model to compute loss for
      data: Dictionary containing input_ids, attention_mask, netlist, etc.
      reward_funcs: List of reward functions to use
      reward_kwargss: List of kwargs for each reward function
      hps: Hyperparameters object
      
  Returns:
      tuple containing:
      - loss: The computed loss value
      - metrics: Dictionary of metrics to log
  """
  device = hps.device
  grpo_beta = hps.grpo_beta
  num_generations = hps.num_generations
  tokenizer = hps.tokenizer
  
  metrics = {}
  base_model = model.module if hasattr(model, "module") else model
  error_flag = torch.zeros(1, device=hps.device)

  try:
    # Activate the GRPO adapter
    base_model.set_adapter("grpo_adapter")
    
    # Get ids and mask from data
    ids = data['input_ids'].to(device, non_blocking=True)
    mask = data['attention_mask'].to(device, non_blocking=True)

    # Generate completions
    gen_start = time.time()
    with unwrap_ddp(model) as unwrapped_model:
      prompt_completion_ids = unwrapped_model.generate(
        input_ids=ids,
        attention_mask=mask, 
        max_length=hps.model_max_length,
        num_return_sequences=num_generations,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
      )
    gen_end = time.time()
    metrics['generation_time'] = gen_end - gen_start
    
    prompt_length = ids.size(1)
    prompt_ids = ids.repeat_interleave(num_generations, dim=0)
    prompt_mask = mask.repeat_interleave(num_generations, dim=0)
    completion_ids = prompt_completion_ids[:, prompt_length:]

    # Mask everything after the first EOS token
    is_eos = completion_ids == tokenizer.eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
    eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
    completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

    # Prepare inputs for logit computation
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    logits_to_keep = completion_ids.size(1)

    # Compute log probabilities
    per_token_logps = get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
    
    # Activate the reference adapter
    base_model.set_adapter("ref_adapter")

    # Compute reference model log probabilities
    with torch.inference_mode():
      with unwrap_ddp(model) as unwrapped_model:
        with disable_ref_adapter(unwrapped_model) as disabled_ref_model:
          ref_per_token_logps = get_per_token_logps(
            disabled_ref_model, 
            prompt_completion_ids,
            attention_mask,
            logits_to_keep
          )

    # Activate the GRPO adapter again
    base_model.set_adapter("grpo_adapter")

    # Compute KL divergence
    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
    metrics['kl'] = per_token_kl.sum(dim=1).mean().item()

    # Decode completions and get prompts
    completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    text_inputs = tokenizer.batch_decode(ids)
    prompts = get_user_prompt(text_inputs)
    prompts = [prompt for prompt in prompts for _ in range(num_generations)]
    netlists = [netlist for netlist in data['netlist'] for _ in range(num_generations)]

    # Compute rewards
    rewards_per_func = torch.zeros(len(prompts), len(reward_funcs), device=device)
    for reward_idx, (reward_func, reward_kwargs) in enumerate(zip(reward_funcs, reward_kwargss)):
      if reward_func.__name__ == "test_generation_reward":
        reward_kwargs.update({"netlists": netlists})
      rewards_per_func[:, reward_idx] = torch.tensor(
        reward_func(prompts, completions, **reward_kwargs),
        dtype=torch.float32,
        device=device
      )

    # Gather rewards and compute advantages
    rewards_per_func = gather(rewards_per_func)
    rewards = rewards_per_func.sum(dim=1)
    
    mean_grouped_rewards = rewards.view(-1, num_generations).mean(dim=1)
    std_grouped_rewards = rewards.view(-1, num_generations).std(dim=1)
    
    mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
    std_grouped_rewards = std_grouped_rewards.repeat_interleave(num_generations, dim=0)
    advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-6)

    # Handle parallel processing
    if hps.parallel:
      process_slice = slice(
        hps.local_rank * len(prompts),
        (hps.local_rank + 1) * len(prompts)
      )
      advantages = advantages[process_slice]

    # Compute final loss
    per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
    per_token_loss = -(per_token_loss - grpo_beta * per_token_kl)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

    # Add metrics
    metrics.update({
      'per_token_loss': per_token_loss.sum(dim=1).mean().item(),
      'per_token_kl': per_token_kl.sum(dim=1).mean().item(),
      'reward': rewards.mean().item(),
      'reward_std': std_grouped_rewards.mean().item()
    })

    reward_per_func = rewards_per_func.mean(0)
    for i, reward_func in enumerate(reward_funcs):
      metrics[f"{reward_func.__name__}"] = reward_per_func[i].item()
  except torch.cuda.OutOfMemoryError as e:
    import traceback
    print(f"[{hps.local_rank}]: Out of memory error. Exiting training loop. Error is being handled by the training loop.\n{traceback.print_exc()}\n{e}")
    error_flag.fill_(1)
    if hps.parallel:
      dist.broadcast(error_flag, src=hps.local_rank)
    # Return a zero loss to avoid breaking the training loop
    loss = torch.tensor(0, device=hps.device, requires_grad=False)

  torch.cuda.empty_cache()
  gc.collect()

  return loss, metrics, error_flag.item()

def update_ref_adapter_ema(model, tau=0.9, ref_adapter_name="ref_adapter", grpo_adapter_name="grpo_adapter"):
  """
  Performs an exponential moving average (EMA) update of the reference adapter weights.
  The reference adapter is updated using the weights from the GRPO adapter.
  
  Formula: new_ref_weights = tau * grpo_weights + (1 - tau) * old_ref_weights
  
  Args:
      model: The model containing both adapters
      tau: The EMA decay rate (default: 0.9)
      ref_adapter_name: Name of the reference adapter to update
      grpo_adapter_name: Name of the source GRPO adapter
  """
  base_model = model.module if hasattr(model, "module") else model
  
  # Process parameters layer by layer to minimize memory usage
  # Map source parameter names to their corresponding target parameter names
  param_mapping = {}
  
  # Build the mapping between GRPO and reference parameters
  for name, _ in base_model.named_parameters():
    if grpo_adapter_name in name:
      ref_name = name.replace(grpo_adapter_name, ref_adapter_name)
      param_mapping[name] = ref_name
  
  # Update parameters one by one without storing all in memory
  for grpo_name, ref_name in param_mapping.items():
    # Get parameters by name to avoid storing all parameters in memory
    grpo_param = dict(base_model.named_parameters())[grpo_name]
    ref_param = dict(base_model.named_parameters())[ref_name]
    
    # Apply EMA update directly: ref = (1-tau)*ref + tau*grpo
    ref_param.data.mul_(1 - tau).add_(grpo_param.data, alpha=tau)
    
    # Free memory after each update
    if len(param_mapping) > 10:  # Only clear cache periodically for large models
      torch.cuda.empty_cache()
  # Final cleanup
  del param_mapping
  torch.cuda.empty_cache()
  gc.collect()

def copy_adapter_weights(model, source_adapter_name="ref_adapter", target_adapter_name="grpo_adapter"):
  """
  Creates a hard copy of adapter weights from source adapter to target adapter.
  Memory-optimized implementation that processes one layer at a time.
  
  Args:
      model: The model containing the adapters
      source_adapter_name: Name of the source adapter (e.g., "grpo_adapter")
      target_adapter_name: Name of the target adapter (e.g., "ref_adapter")
  """
  base_model = model.module if hasattr(model, "module") else model
  
  # Map source parameter names to their corresponding target parameter names
  source_to_target_map = {}
  
  # Build the mapping between source and target parameters
  for name, _ in base_model.named_parameters():
    if source_adapter_name in name:
      target_name = name.replace(source_adapter_name, target_adapter_name)
      source_to_target_map[name] = target_name
  
  # Copy parameters directly without storing intermediate copies
  for source_name, target_name in source_to_target_map.items():
    # Get parameters by name to avoid storing all parameters in memory
    source_param = dict(base_model.named_parameters())[source_name]
    target_param = dict(base_model.named_parameters())[target_name]
    # Copy data directly without creating additional clones
    target_param.data.copy_(source_param.data)
    # Free memory after each copy
    torch.cuda.empty_cache()
  torch.cuda.empty_cache()
  gc.collect()

def get_ref_update_freq(ref_update_step, total_updates, initial_ref_update_freq=10, final_ref_update_freq=1):
  """
  Calculate how often to update reference adapter based on training progress.
  Args:
      ref_update_step: Current update step (after gradient accumulation) across all epochs
      total_updates: Total updates across all epochs
      initial_ref_update_freq: Initial update frequency
      final_ref_update_freq: Final update frequency
  Returns:
      Number of gradient steps to wait before next reference update
  """
  progress = ref_update_step / total_updates
  # Exponentially decrease update interval
  current_freq = max(
      final_ref_update_freq,
      int(initial_ref_update_freq * (1 - progress)**2)
  )
  return current_freq

# Below rlft() function for RL Fine-Tuning
def rlft(dataloader, model, hps: AttrDict, reward_funcs: Union[Callable, list[Callable]], desc:str = "GRPO-RL Training...", training_loop: bool = True):
  # Retrieve optimizer from hyperparameters dict
  optimizer = hps.get('optimizer', torch.optim.AdamW(model.parameters(), lr=hps.lr))
  scheduler = hps.get('scheduler', None)
  
  # TODO: I have to add them in hyperparameters
  logging_steps = 10 

  # Set collate function's flags to supervised fine-tuning
  dataloader.collate_fn.set_left_padding()

  # Ensure model is in training mode for gradient computation
  model.train()

  if hps.wandb:
    wandb.log({'rlft/hps/lr': hps.lr, 'rlft/hps/micro_batch_size': hps.micro_batch_size, 'rlft/hps/gradient_accumulation_steps': hps.gradient_accumulation_steps})
    wandb.watch(model, log="gradients", log_freq=1)
  
  if is_main_process():
    pbar = tqdm(total=len(dataloader), desc=f"[{hps.local_rank}]: {desc}", disable=hps.disable_tqdm)
  # Get dataloader iterator
  data_iterator = iter(itertools.cycle(dataloader))
  # Gather statistics
  metrics = defaultdict(list)

  # Reward functions 
  if not isinstance(reward_funcs, list):
    reward_funcs = [reward_funcs]

  # Regular expressions used during Chain-of-Thoughts (COTs) rewards calculation.
  reward_kwargss = prepare_reward_kwargs(hps)
  # Initialize logging step
  logging_step = 0
  # Start training
  if is_main_process():
    print(f"Optimizer's step every: {hps.gradient_accumulation_steps}")
  # Initialize best_avg_reward to a negative value
  hps.best_avg_reward = -10.0
  
  # Parameters for dynamic reference adapter updates
  updates_per_epoch = len(dataloader) // hps.gradient_accumulation_steps
  total_updates = hps.epochs * updates_per_epoch  # Total updates across all epochs
  initial_ref_update_freq = hps.initial_ref_update_freq  # Update every 10 gradient steps at start
  final_ref_update_freq = hps.final_ref_update_freq      # Update every gradient step at end
  
  ref_update_step = 0  # Counter for actual parameter updates across all epochs
  next_ref_update = get_ref_update_freq(0, total_updates, initial_ref_update_freq, final_ref_update_freq)  # Steps until next ref update
  
  # Initialize accumulators for metrics
  accumulated_metrics = defaultdict(list)
  for epoch in range(hps.epochs):
    if is_main_process():
      print(f"Epoch {epoch+1}/{hps.epochs}:")
      pbar.reset()
  
    for i in range(0, len(dataloader)):
      if i % hps.gradient_accumulation_steps == 0:
        batch_loss = 0
        # Reset metric accumulators at the start of each accumulation step
        accumulated_metrics.clear()
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Count skipped micro batches due to OOM error
        count_skip_micro_batch = 0
        # Initialize last_grad_norm to None
        last_grad_norm = None
      # Get next data
      data = next(data_iterator)

      # Compute loss and get metrics
      micro_batch_loss, micro_batch_metrics, error_flag = compute_loss(
          model=model,
          data=data,
          reward_funcs=reward_funcs,
          reward_kwargss=reward_kwargss,
          hps=hps
      )
      if error_flag:
        if is_main_process():
          pbar.update(1)
        count_skip_micro_batch += 1
        continue
      # Scale loss for gradient accumulation
      micro_batch_loss = micro_batch_loss / (hps.gradient_accumulation_steps - count_skip_micro_batch)
      batch_loss += micro_batch_loss.item()
      # Accumulate metrics for each micro-batch
      for key, value in micro_batch_metrics.items():
        accumulated_metrics[key].append(float(value))

      # Compute/Accumulate the gradients
      if training_loop:
        micro_batch_loss.backward()

        # Apply the gradients when the accumulation step is reached
        if (i+1) % hps.gradient_accumulation_steps == 0 or i == len(dataloader)-1:
          # Clip the gradients
          last_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
          # Update the model parameters
          optimizer.step()
          # Update the learning rate scheduler
          if scheduler:
            scheduler.step() if hps.deepspeed_kernel else scheduler.step(batch_loss)

      if (i+1) % hps.gradient_accumulation_steps == 0:
        # Dynamic reference model update
        ref_update_step += 1
        if ref_update_step % next_ref_update == 0:
          # Update the reference adapter
          update_ref_adapter_ema(model, tau=hps.grpo_tau)
          # Update the next reference update step
          next_ref_update = get_ref_update_freq(ref_update_step, total_updates)
        
        # Log the metrics
        metrics["epoch"].append(str(epoch+1))
        metrics["batch"].append(str(int(i//hps.gradient_accumulation_steps)+1))
        metrics["step"].append(str(i+1))
        metrics["batch_loss"].append(smart_round(batch_loss))
        metrics["ref_update_step"].append(f"{ref_update_step}/{next_ref_update}")
        # Log the gradient norm metric (defaulting to 0.0 if not available)
        metrics["grad_norm"].append(smart_round(last_grad_norm) if last_grad_norm is not None else smart_round(0.0))
        # Add averaged metrics for each micro-batch
        for key in accumulated_metrics.keys():
          avg_value = sum(accumulated_metrics[key]) / len(accumulated_metrics[key])
          metrics[f"avg_{key}"].append(smart_round(avg_value))
        # Track and save the best model based on average reward
        current_avg_reward = float(eval(metrics["avg_reward"][-1]))

        # Save model when we get a new best average reward
        if current_avg_reward > hps.best_avg_reward:
          hps.best_avg_reward = current_avg_reward
          metrics["best_avg_reward"].append(smart_round(hps.best_avg_reward))
          if is_main_process():
            print(f"\n🔥 New best average reward: {current_avg_reward:.4f}")
          if not DEBUG:
            best_model_path = os.path.join(hps.log_dir, "models", "best_reward_model", hps.save_in_repo.split("/")[-1])
            best_model_file_path = os.path.join(best_model_path, hps.file_path.split("/")[-1])
            os.makedirs(best_model_path, exist_ok=True)
            infer(ddp_model=model, dataloader=dataloader, tokenizer=hps.tokenizer, model_max_length=hps.model_max_length, epoch=1, file_path=best_model_file_path, parallel=hps.parallel, new_file=not os.path.exists(best_model_file_path))
            save_model(model, best_model_path, push_to_hub=False, save_embedding_layers=True, hps=hps, delete_previous=True, commit_message=f"Best reward model with avg_reward={current_avg_reward:.4f}")
            if is_main_process():
              print(f"✅ Saved best model to {best_model_path}")        
        else:
          metrics["best_avg_reward"].append(smart_round(hps.best_avg_reward))

        if hps.wandb: 
          # Log to wandb both metrics and table, including the gradient norm along with other metrics
          wandb.log({
              **{f"rlft/{k}": float(eval(metrics[k][-1])) for k in metrics.keys()},
          })

        if is_main_process():
          x = pd.DataFrame(metrics)
          # Show averaged metrics in progress bar
          pbar.set_postfix(x.loc[x.index[-1], ['batch', 'batch_loss', 'avg_per_token_loss', 'avg_per_token_kl', 'avg_reward', 'best_avg_reward']].to_dict())
          if logging_step % logging_steps == 0:
            print(tabulate(x.loc[x.index[-logging_steps:], ['batch', 'batch_loss', 'avg_per_token_loss', 'avg_per_token_kl', 'avg_reward', 'avg_cot_reward', 'avg_test_generation_reward', 'best_avg_reward']], headers='keys', tablefmt='psql', showindex=False))
            logging_step = 0
          logging_step += 1
        
      if is_main_process():
        pbar.update(1)

      if DEBUG and i > 500:
        break

  x = pd.DataFrame(metrics)
  # Log final complete table
  if hps.wandb:
    x = x.astype(float)
    wandb.log({"rlft_complete_history": wandb.Table(dataframe=x)})
  if is_main_process():
    print(tabulate(x, headers='keys', tablefmt='psql', showindex=False))
    pbar.close()

def fine_tuning(dataloader, validation_loader, model, hps, training_loop=True):
  # Fine-tune with Supevised Fine-Tuning (SFT)
  # 1. Train new embeddings tokens and head
  if hps.new_tokens:
    sft(dataloader, model, hps, desc="SFT Embeddings Training...", training_loop=training_loop)
    # Apply inference on some random samples of validation set
    if not DEBUG:
      infer(ddp_model=model, dataloader=validation_loader, tokenizer=hps.tokenizer, model_max_length=hps.model_max_length, epoch=1, file_path=hps.file_path, parallel=hps.parallel, new_file=True)
      save_model(model, hps.save_in_repo, push_to_hub=True, save_embedding_layers=True, hps=hps, commit_message="Model trained with new embeddings")

  hps.lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM if hps.is_causal else TaskType.SEQ_2_SEQ_LM, r=hps.lora_r, lora_alpha=hps.lora_alpha, lora_dropout=hps.lora_dropout)
  if not hps.adapter_name or not hps.adapter_repo:
    # Add adapter (LoRA matrices) 
    base_model = model.module if hasattr(model, "module") and hps.parallel else model
    # Check if the base model has the 'peft_config' attribute to determine if LoRA configuration needs to be set
    if not hasattr(base_model, 'peft_config'):
      # Set LoRA configuration: LoRA (Low-Rank Adaptation) is used to efficiently fine-tune large models by adding low-rank matrices to the model's weights.
      peft_model = get_peft_model(base_model, hps.lora_config, adapter_name="ref_adapter")
    else:
      # TODO: Check if this is correct
      peft_model = model.module if hasattr(model, "module") and hps.parallel else model 

    if is_main_process():
      print(f"Applying LoRA to the model...")
    # Activate the newly added LoRA adapter to use the LoRA weights during training
    if hps.parallel:
      model = apply_lora_distributed(model, peft_model, hps.lora_config, adapter_name="ref_adapter")
    else:
      model = apply_lora_non_distributed(model, peft_model, hps.lora_config, adapter_name="ref_adapter")
    # Load the adapter
  elif hps.adapter_name and hps.adapter_repo:
    if hps.parallel:
      model.module = PeftModel.from_pretrained(model.module, hps.adapter_repo, subfolder=hps.adapter_name, adapter_name=hps.adapter_name).to(torch.bfloat16)
    else:
      model = PeftModel.from_pretrained(model, hps.adapter_repo, subfolder=hps.adapter_name, adapter_name=hps.adapter_name).to(torch.bfloat16)

  # Set find_unused_parameters to True to avoid OOM error
  model.find_unused_parameters = True

  # Configure which parts of the model to train in step 2
  train_layers(model, train_embeddings=True, train_head=True, train_lora=True, train_base_model=False)

  # Get trainable parameters with their specific learning rates
  # trainable_params = get_trainable_parameters(model)
  
  # Initialize the optimizer with parameter groups
  if hps.parallel:
    hps.optimizer = ZeroRedundancyOptimizer(
        model.parameters(),  # Contains parameter groups with their learning rates
        optimizer_class=AdamW,
        lr=hps.lr
    )
  else:
    hps.optimizer = AdamW(
        model.parameters(),  # Contains parameter groups with their learning rates
        lr=hps.lr
    )

  # Calculate the total number of training steps
  total_training_steps = (hps.epochs * len(dataloader)) // hps.gradient_accumulation_steps
  # Set up the learning rate scheduler. Uses a cosine schedule with warmup
  hps.scheduler = get_cosine_schedule_with_warmup(hps.optimizer, num_warmup_steps=10, num_training_steps=total_training_steps, num_cycles=3/20)

  if hps.train_lora:
    if is_main_process():
      print(f"{model}\nSFT embedding, lora and head:\nModel training parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,d}\nTrainable Model size: {convert_bytes(model_size_in_bytes(model))}\n")
    # 2. Train adapter (LoRA weights) as well
    sft(dataloader, model, hps, desc="SFT Lora Training...", training_loop=training_loop) 
    # Apply inference on some random samples of validation set
    if not DEBUG:
      infer(ddp_model=model, dataloader=validation_loader, tokenizer=hps.tokenizer, model_max_length=hps.model_max_length, epoch=1, file_path=hps.file_path, parallel=hps.parallel, new_file=True)
      save_model(model, hps.save_in_repo, push_to_hub=True, save_embedding_layers=True, hps=hps, commit_message="Model trained with LoRA")

  # Configure which parts of the model to train in step 3
  train_layers(model, train_embeddings=False, train_head=False, train_lora=True, train_base_model=False)

  # Add GRPO adapter
  if hps.parallel:
    model.module.base_model.add_adapter(hps.lora_config, adapter_name="grpo_adapter")
    model.module.set_adapter("grpo_adapter")
  else:
    model.base_model.add_adapter(hps.lora_config, adapter_name="grpo_adapter")
    model.set_adapter("grpo_adapter")
  # Copy the weights from the reference adapter to the GRPO adapter
  copy_adapter_weights(model, source_adapter_name="ref_adapter", target_adapter_name="grpo_adapter")
  # Set the flags for distributed systems. Data samplers and Data Loaders
  use_sampler = hps.parallel==True and hps.deepspeed_kernel==False
  # Shuffle is handled by the sampler
  hps.shuffle = not use_sampler
  # Replicate the sampler across all processes
  sampler = DistributedSampler(dataloader.sampler.dataset, rank=dataloader.sampler.rank, num_replicas=dataloader.sampler.num_replicas) if use_sampler else None  

  # Adjust gradient accumulation steps. GRPO is slower than SFT. lower the number of gradient accumulation steps.
  hps.gradient_accumulation_steps = max(1, hps.gradient_accumulation_steps//min(2, hps.num_generations))
  hps.epochs = 1

  # Subset the dataset to 200,000 samples to shorten the training time
  dataset_subset = dataloader.dataset.select(range(min(200_000, len(dataloader.dataset))))
  # Change batch size. Due to number of generations there might be OOM cuda error.
  hps.micro_batch_size = max(1, hps.micro_batch_size//hps.num_generations)
  dataloader = DataLoader(dataset=dataset_subset, batch_size=hps.micro_batch_size, shuffle=hps.shuffle, collate_fn=MyCollate(tokenizer=hps.tokenizer, is_causal=hps.is_causal, lora=hps.lora), sampler=sampler, num_workers=dataloader.num_workers, pin_memory=dataloader.pin_memory, drop_last=dataloader.drop_last)#, multiprocessing_context='fork', worker_init_fn=worker_init_fn)
  torch.cuda.empty_cache()
  gc.collect()

  # When adding GRPO adapter, update optimizer similarly:
  # after_grpo_params = get_trainable_parameters(model)

  hps.lr = min(5e-6, hps.lr)
  if hps.parallel:
    hps.optimizer = ZeroRedundancyOptimizer(
      model.parameters(),
      optimizer_class=AdamW,
      lr=hps.lr
    )
  else:
    hps.optimizer = AdamW(
      model.parameters(),
      weight_decay=0.01,
      eps=1e-8,
      betas=(0.9, 0.999),
      lr=hps.lr
    )

  # Configure which parts of the model to train
  # train_layers(model, train_embeddings=True, train_head=True, train_lora=True, train_base_model=False)
  # Calculate the total number of training steps
  total_training_steps = (hps.epochs * len(dataloader)) // hps.gradient_accumulation_steps
  hps.scheduler = get_cosine_schedule_with_warmup(hps.optimizer, num_warmup_steps=1, num_training_steps=total_training_steps, num_cycles=3/20)

  if is_main_process():
    print(f"{model}\nGRPO-RLFT train embedding, lora and head:\nTrainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,d}\nTrainable Model size: {convert_bytes(model_size_in_bytes(model))}")
  
  # 3, Fine-tune with Reinforcement Learning (RL)
  rlft(dataloader, model, hps, reward_funcs=[cot_reward, test_generation_reward], training_loop=training_loop)
  # Apply inference on some random samples of validation set
  if not DEBUG: 
    model.set_adapter("grpo_adapter")
    infer(ddp_model=model, dataloader=validation_loader, tokenizer=hps.tokenizer, model_max_length=hps.model_max_length, epoch=1, file_path=hps.file_path, parallel=hps.parallel, new_file=True)
    save_model(model, hps.save_in_repo, push_to_hub=True, save_embedding_layers=True, hps=hps, commit_message="Final Model trained with GRPO-RL")
    final_model_path = os.path.join(hps.log_dir, "models", "final_model", hps.save_in_repo.split("/")[-1])
    os.makedirs(final_model_path, exist_ok=True)
    save_model(model, final_model_path, push_to_hub=False, save_embedding_layers=True, hps=hps, delete_previous=False)


def main():
  # Arguments
  parser = argparse.ArgumentParser(description='Training arguments parser')

  # Initialize hyperparameters using custom AttrDict()
  hps = hyperparameters(parse_arguments(parser))
  
  # Set debug mode
  hps.debug = DEBUG

  # Set the model name
  hps.model_name = models_causal[hps.model_name]
  hps.adapter_repo = models_causal[hps.adapter_repo] if hps.adapter_repo else None
  hps.save_in_repo = models_causal[hps.save_model]

  # Initialize the W&B logger
  if is_main_process() and hps.wandb:
    wandb.init(project=f"RL Fine-Tuning", entity="chrivasileiou", config=hps)

  # Initialize the training environment based on the GPU availability and parallelization
  initialize_training_environment(hps)
  
  # set a seed for proper synchronization
  set_seed(hps.seed + hps.local_rank + np.random.randint(1, 2**32 - 1))
  
  # Model Loading
  model = load_model(hps)

  # Load tokenizer
  model = load_tokenizer(model, hps)
  
  # Load dataset
  dataset = load_raw_dataset(hps.data_file)
  
  # Prepare objects for training and validation methods (optimizer, scheduler, dataloader, etc...)
  model, training_loader, validation_loader, testing_loader = prepare_objects_for_training(model, dataset, hps)

  # Fine-Tune model
  fine_tuning(training_loader, validation_loader, model, hps, training_loop=True)


if __name__ == '__main__':
  main()

