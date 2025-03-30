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
from transformers import AutoModelForCausalLM
from transformers import set_seed, AutoTokenizer, get_cosine_schedule_with_warmup
from sentence_transformers import SentenceTransformer
from accelerate.utils import gather
from peft import get_peft_model, LoraConfig, TaskType, get_peft_model_state_dict
from torch.optim import AdamW
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
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
  print_prompt_completions_sample,
)
from typing import List, Union, Callable, Dict, Any
from trl.trainer.utils import print_rich_table
import random

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
            print_rich_table(x.iloc[-logging_steps:])
            # print(tabulate(x.iloc[-logging_steps:], headers='keys', tablefmt='psql', showindex=False))
            logging_step=0
          logging_step+=1
      if DEBUG and i>10:
        break
      
      if is_main_process():
        pbar.update(1)
    
  x = pd.DataFrame(metrics)
  # Log final complete table
  if hps.wandb: 
    # Convert string values to float, handling any non-numeric values and None values
    x = x.fillna(0).applymap(lambda val: float(eval(val)) if isinstance(val, str) else val)
    wandb.log({"sft_complete_history": wandb.Table(dataframe=x)})
  if is_main_process():
    print_rich_table(x)
    # print(tabulate(x, headers='keys', tablefmt='psql', showindex=False))
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
  # It should be num_logits_to_keep=logits_to_keep+1 only if transformers <= 4.48 
  # It should be logits_to_keep=logits_to_keep+1 only if transformers >= 4.49 
  # logits_to_keep keeps the last predicted tokens. Starts counting from the last token.
  logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep+1).logits # (B, L, V)
  logits = logits[:, :-1, :] # (B, L-1, V) exclude the last logit: it corresponds to the next token prediction
  input_ids = input_ids[:, -logits_to_keep:] # Keep completion ids

  # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
  per_token_logps = []
  for logits_row, input_ids_row in zip(logits, input_ids):
    log_probs = logits_row.log_softmax(dim=-1)
    token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(-1)).squeeze(-1)
    per_token_logps.append(token_log_prob)
  return torch.stack(per_token_logps)

def prepare_reward_kwargs(hps, reward_funcs):
  cot_block_re = re.compile(r'CHAIN_OF_THOUGHT:\n(.*?)SNAPSHOT', re.DOTALL)
  thought_pattern_re = re.compile(r'(\d+)\.(.*?)(?=\d+\.|$)', re.DOTALL)
  fault_re = re.compile(r"(sa\d)\s+(_\d+_)", re.DOTALL)
  simulation_re = re.compile(r"SNAPSHOT:\n```\n(.*?)```\s+INPUT_VECTOR", re.DOTALL)
  input_vector_re = re.compile(r"INPUT_VECTOR:\s\"(.*?)\"", re.DOTALL)
  expected_output_re = re.compile(r"EXPECTED_OUTPUT:\s\"(.*?)\"", re.DOTALL)
  detected_faults_re = re.compile(r"DETECTED_FAULTS:\s\"(.*?)\"", re.DOTALL)

  # Prepare the reward function's arguments
  sentence_transformer = SentenceTransformer('paraphrase-MiniLM-L6-v2').to(hps.device)
  reward_kwargs = []
  for reward_func in reward_funcs:
    if reward_func.__name__ == "cot_reward":
      reward_kwargs.append({"model": sentence_transformer, "cot_block_re": cot_block_re, "thought_pattern_re": thought_pattern_re, "fault_re": fault_re})
    elif reward_func.__name__ == "test_generation_reward":
      reward_kwargs.append({"fault_re": fault_re, "simulation_re": simulation_re, "input_vector_re": input_vector_re, "expected_output_re": expected_output_re, "detected_faults_re": detected_faults_re})
  return reward_kwargs

def smart_round(value, sig_figs=3):
  return f"{value:.{sig_figs}g}" if value != 0 else "0"

@contextmanager
def unwrap_model(model):
  if isinstance(model, DDP):
    yield model.module
  elif isinstance(model, FSDP): 
    # Set up a context where we have the full weights
    with FSDP.summon_full_params(model, recurse=True, writeback=False, with_grads=False):
      yield model._fsdp_wrapped_module
  else:
    yield model

def get_base_model(model):
  if isinstance(model, DDP):
    return model.module
  elif isinstance(model, FSDP):
    return model._fsdp_wrapped_module
  else:
    return model
  
@contextmanager
def disable_ref_adapter(model):
  for _, param in model.named_parameters():
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


def compute_loss(model, data_iterator, reward_funcs, reward_kwargss, hps, step):
  """
  Compute the GRPO loss for a batch of inputs.
  
  Args:
      model: The model to compute loss for
      data_iterator: Iterator over the data
      reward_funcs: List of reward functions to use
      reward_kwargss: List of kwargs for each reward function
      hps: Hyperparameters object
      step: Current step number

  Returns:
      tuple containing:
      - loss: The computed loss value
      - metrics: Dictionary of metrics to log
  """
  device = hps.device
  grpo_beta = hps.grpo_beta
  num_generations = hps.num_generations
  tokenizer = hps.tokenizer
  epsilon = 0.2
  # Get next data
  data = next(data_iterator)

  metrics = {}
  base_model = get_base_model(model)
  error_flag = torch.zeros(1, device=hps.device)
  generate_kwargs = {
    "max_length": hps.model_max_length,
    "num_return_sequences": num_generations,
    "do_sample": True,
    "temperature": 0.6,
    "top_p": 0.75,
    "pad_token_id": tokenizer.eos_token_id
  }

  try:
    # Activate the GRPO adapter
    base_model.set_adapter("grpo_adapter")
    
    # Get ids and mask from data
    ids = data['input_ids'].to(device, non_blocking=True).contiguous()
    mask = data['attention_mask'].to(device, non_blocking=True).contiguous()

    # Generate completions
    gen_start = time.time()
    with unwrap_model(model) as unwrapped_model:        
      # Run generation with proper error handling
      prompt_completion_ids = unwrapped_model.generate(input_ids=ids, attention_mask=mask, **generate_kwargs)
    gen_end = time.time()
    metrics['generation_time'] = gen_end - gen_start

    # Prepare inputs for logit computation
    prompt_length = ids.size(1)
    prompt_ids = ids.repeat_interleave(num_generations, dim=0) # equivalent to prompt_completion_ids[:, :prompt_length]
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

    # Compute log probabilities of the training model
    with torch.no_grad(): # don't track gradients
      # Get log probabilities for the training model
      per_token_logps = get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
    
    # Activate the reference adapter
    base_model.set_adapter("ref_adapter")

    # Compute reference model log probabilities with gradients disabled. KL divergence from the reference model to the training model.
    with torch.no_grad(): # don't track gradients
      with unwrap_model(model) as unwrapped_model: # unwrap_model() is used to handle and disable the adapter
        with disable_ref_adapter(unwrapped_model) as ref_model: # disable_ref_adapter() is used to disable the adapter
          # Get log probabilities for the reference model
          ref_per_token_logps = get_per_token_logps(
            ref_model, 
            prompt_completion_ids,
            attention_mask,
            logits_to_keep
          )

    # Activate the GRPO adapter again
    base_model.set_adapter("grpo_adapter")

    # Compute KL divergence between the model and the reference model
    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
    metrics['kl'] = per_token_kl.sum(dim=1).mean().item()

    # Decode completions and get prompts
    completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    text_inputs = tokenizer.batch_decode(ids)
    prompts = get_user_prompt(text_inputs)
    prompts = [prompt for prompt in prompts for _ in range(num_generations)]
    netlists = [netlist for netlist in data['netlist'] for _ in range(num_generations)]

    # Compute rewards
    rewards_per_func = torch.zeros(len(prompts), len(reward_funcs), device=device, dtype=torch.bfloat16)
    for reward_idx, (reward_func, reward_kwargs) in enumerate(zip(reward_funcs, reward_kwargss)):
      if reward_func.__name__ == "test_generation_reward":
        reward_kwargs.update({"netlists": netlists})
      returned_rewards_list = reward_func(prompts, completions, **reward_kwargs)
      
      # Initialize rewards array and process metrics in one pass
      returned_rewards_per_func = torch.zeros(len(returned_rewards_list), dtype=torch.bfloat16, device=device)
      # Initialize pass@k metrics
      if reward_func.__name__ == "test_generation_reward":
        pass_key = f'pass@{hps.num_generations}'
        pass_at_k = {pass_key: []}
      # Pre-calculate divisor for averaging
      divisor = 1.0 / len(returned_rewards_list)
      for r_i, returned_rewards in enumerate(returned_rewards_list):
        if reward_func.__name__ == "test_generation_reward":
          if r_i//hps.num_generations == len(pass_at_k[pass_key]):
            pass_at_k[pass_key].append([])
          pass_at_k[pass_key][r_i//hps.num_generations].append(returned_rewards['fault_detect_inpvector'])
        # Sum all reward components and update metrics
        returned_rewards_per_func[r_i] = sum(returned_rewards.values())
        # Update metrics dictionary efficiently
        for k, v in returned_rewards.items():
          # Construct metrics key efficiently - only append suffix if k is not empty
          metrics_key = f"{reward_func.__name__}" + (f"/{k}" if k else "")
          metrics[metrics_key] = metrics.get(metrics_key, 0) + v * divisor
      rewards_per_func[:, reward_idx] = returned_rewards_per_func
    if 'pass_at_k' in locals():
      pass_at_k[pass_key] = torch.tensor(pass_at_k[pass_key], device=device, dtype=torch.bool)
      pass_at_k[pass_key] = gather(pass_at_k[pass_key])
      metrics[pass_key] = pass_at_k[pass_key].cpu().numpy()
    # Print a sample of the prompt, completion, and reward
    if step % hps.gradient_accumulation_steps == 0:
      print_prompt_completions_sample(prompts, completions, rewards_per_func.sum(dim=1).clone().cpu(), step)

    # Gather rewards and compute advantages
    rewards_per_func = gather(rewards_per_func)
    rewards = rewards_per_func.sum(dim=1)
    
    # Size: (B*N) -> Size: (B, N) -> Size: (B, 1)
    mean_grouped_rewards = rewards.view(-1, num_generations).mean(dim=1)
    std_grouped_rewards = rewards.view(-1, num_generations).std(dim=1)
    
    # Size: (B, 1) -> Size: (B*N, 1)
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

    old_per_token_logps = per_token_logps if num_generations > 1 else per_token_logps.detach()
    
    # Get log probabilities for the training model with the gradients enabled
    per_token_logps = get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

    # Compute surrogate loss with clipped ratio
    ratio = torch.exp(per_token_logps - old_per_token_logps)
    clipped_ratio = torch.clamp(ratio, min=1-epsilon, max=1+epsilon)
    
    # Compute the surrogate loss
    per_token_loss1 = ratio * advantages.unsqueeze(1)
    per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
    per_token_loss = -torch.minimum(per_token_loss1, per_token_loss2)

    # Add the KL divergence to the loss 
    per_token_loss = per_token_loss + grpo_beta * per_token_kl

    # Compute the loss
    loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
    
    # Add metrics
    is_clipped = (ratio < 1-epsilon).float() + (ratio > 1+epsilon).float()
    clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
    kl_div = (per_token_kl * completion_mask).sum() / completion_mask.sum()

    metrics.update({
      'clip_ratio': clip_ratio.item(),
      'kl': kl_div.item(),
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
    # Ensure all processes know about the OOM error
    if dist.is_available() and dist.is_initialized():
      dist.all_reduce(error_flag, op=dist.ReduceOp.MAX)
    # Return a zero loss to avoid breaking the training loop
    loss = torch.tensor(0.0, device=hps.device, dtype=torch.bfloat16, requires_grad=True)
    return loss, metrics, error_flag

  # Synchronize error flags one last time before returning
  if dist.is_available() and dist.is_initialized():
    dist.all_reduce(error_flag, op=dist.ReduceOp.MAX)

  torch.cuda.empty_cache()
  gc.collect()

  return loss, metrics, error_flag




def compute_aggregated_loss(model, data_iterator, reward_funcs, reward_kwargss, hps, step, num_micro_batches=4):
    """
    Compute the GRPO loss aggregated across multiple micro-batches.
    
    Args:
        model: The model to compute loss for
        data_iterator: Iterator providing batches of data
        reward_funcs: List of reward functions to use
        reward_kwargss: List of kwargs for each reward function
        hps: Hyperparameters object
        step: Current step number
        num_micro_batches: Number of micro-batches to process before computing loss
        
    Returns:
        tuple containing:
        - loss: The computed aggregated loss value
        - metrics: Dictionary of metrics to log
        - error_flag: Tensor indicating if an error occurred
    """
    device = hps.device
    grpo_beta = hps.grpo_beta
    num_generations = hps.num_generations
    tokenizer = hps.tokenizer
    epsilon = 0.2
    
    # Initialize storage for aggregated data across micro-batches
    all_input_ids = []
    all_attention_masks = []
    all_completion_masks = []
    all_per_token_logps = []
    all_ref_per_token_logps = []
    all_per_token_kl = []
    all_prompts = []
    all_completions = []
    all_netlists = []
    all_generation_times = []
    all_logits_to_keep = []
    aggregated_metrics = {}
    base_model = get_base_model(model)
    error_flag = torch.zeros(1, device=device)

    # Set up generation parameters
    generate_kwargs = {
        "max_length": hps.model_max_length,
        "num_return_sequences": num_generations,
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.75,
        "pad_token_id": tokenizer.eos_token_id,
    }

    # Process multiple micro-batches sequentially
    for micro_batch_idx in range(num_micro_batches):
      # Skip processing more micro-batches if an error occurred
      if error_flag.item() > 0:
        break
      
      try:
        # Get next batch of data
        data = next(data_iterator)
        
        # Activate the GRPO adapter
        base_model.set_adapter("grpo_adapter")
        
        # Get ids and mask from data
        ids = data['input_ids'].to(device, non_blocking=True).contiguous()
        mask = data['attention_mask'].to(device, non_blocking=True).contiguous()
        
        # Generate completions with timing
        gen_start = time.time()
        with unwrap_model(model) as unwrapped_model:
          prompt_completion_ids = unwrapped_model.generate(input_ids=ids, attention_mask=mask, **generate_kwargs)
        gen_end = time.time()
        all_generation_times.append(gen_end - gen_start)
        
        # Prepare inputs for logit computation
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
        
        # Store for later processing
        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_completion_masks.append(completion_mask)
        all_logits_to_keep.append(logits_to_keep)

        # Compute log probabilities of the training model for this micro-batch
        with torch.no_grad():  # don't track gradients for initial logp computation
          per_token_logps = get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
          all_per_token_logps.append(per_token_logps)
        
        # Activate the reference adapter
        base_model.set_adapter("ref_adapter")
        
        # Compute reference model log probabilities with gradients disabled
        with torch.no_grad():
          with unwrap_model(model) as unwrapped_model:
            with disable_ref_adapter(unwrapped_model) as ref_model:
              ref_per_token_logps = get_per_token_logps(
                  ref_model,
                  input_ids,
                  attention_mask,
                  logits_to_keep
              )
              all_ref_per_token_logps.append(ref_per_token_logps)
        
        # Activate the GRPO adapter again
        base_model.set_adapter("grpo_adapter")
        
        # Compute KL divergence between the model and the reference model
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        all_per_token_kl.append(per_token_kl)
        
        # Decode completions and get prompts
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        text_inputs = tokenizer.batch_decode(ids)
        prompts = get_user_prompt(text_inputs)
        prompts = [prompt for prompt in prompts for _ in range(num_generations)]
        netlists = [netlist for netlist in data['netlist'] for _ in range(num_generations)]
        
        # Store for later processing
        all_prompts.extend(prompts)
        all_completions.extend(completions)
        all_netlists.extend(netlists)
        
        # Clear CUDA cache after each micro-batch to prevent OOM
        torch.cuda.empty_cache()
          
      except torch.cuda.OutOfMemoryError as e:
        import traceback
        print(f"[{hps.local_rank}]: Out of memory error during micro-batch {micro_batch_idx}. Error is being handled.\n{traceback.print_exc()}\n{e}")
        error_flag.fill_(1)
        # Ensure all processes know about the OOM error
        if dist.is_available() and dist.is_initialized():
          dist.all_reduce(error_flag, op=dist.ReduceOp.MAX)
        break
    
    # If an error occurred, return early with zero loss
    if error_flag.item() > 0:
      loss = torch.tensor(0.0, device=device, dtype=torch.bfloat16, requires_grad=True)
      return loss, aggregated_metrics, error_flag
    
    # Process rewards for all completions across micro-batches
    rewards_per_func = torch.zeros(len(all_prompts), len(reward_funcs), device=device, dtype=torch.bfloat16)
    
    for reward_idx, (reward_func, reward_kwargs) in enumerate(zip(reward_funcs, reward_kwargss)):
      if reward_func.__name__ == "test_generation_reward":
        reward_kwargs.update({"netlists": all_netlists})

      # Calculate rewards
      returned_rewards_list = reward_func(all_prompts, all_completions, **reward_kwargs)
      returned_rewards_per_func = torch.zeros(len(returned_rewards_list), dtype=torch.bfloat16, device=device)
      divisor = 1.0 / len(returned_rewards_list) if len(returned_rewards_list) > 0 else 0
      for r_i, returned_rewards in enumerate(returned_rewards_list):
        returned_rewards_per_func[r_i] = sum(returned_rewards.values())
        for k, v in returned_rewards.items():
          metrics_key = f"{reward_func.__name__}" + (f"/{k}" if k else "")
          aggregated_metrics[metrics_key] = aggregated_metrics.get(metrics_key, 0) + v * divisor
      
      rewards_per_func[:, reward_idx] = returned_rewards_per_func
    
    # Print sample of prompts, completions and rewards
    if step % hps.gradient_accumulation_steps == 0:
      print_prompt_completions_sample(
          all_prompts[:5], 
          all_completions[:5], 
          rewards_per_func[:5].sum(dim=1).clone().cpu(), 
          step
      )
    
    # Gather rewards from all processes
    rewards_per_func = gather(rewards_per_func)
    rewards = rewards_per_func.sum(dim=1)
    
    # TODO: Test the following 3 lines of code
    # Compute advantages across all micro-batches
    # We need to reshape rewards to (num_prompts_total, num_generations) to compute mean and std per prompt group
    batch_size_per_micro_batch = len(all_input_ids[0]) // num_generations
    total_prompts = batch_size_per_micro_batch * num_micro_batches * hps.world_size
    rewards_reshaped = rewards.view(total_prompts, num_generations)
    
    # Compute mean and std per prompt group
    mean_grouped_rewards = rewards_reshaped.mean(dim=1)
    std_grouped_rewards = rewards_reshaped.std(dim=1)
    
    # Expand back to original shape
    mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
    std_grouped_rewards = std_grouped_rewards.repeat_interleave(num_generations, dim=0)
    
    # Compute advantages
    advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-6)
    
    # Handle parallel processing - slice to get only this process's data
    if hps.parallel:
      process_start = hps.local_rank * len(all_prompts)
      process_end = (hps.local_rank + 1) * len(all_prompts)
      process_slice = slice(process_start, process_end)
      advantages = advantages[process_slice]
    
    # Compute the final loss in memory-efficient chunks
    total_loss = torch.tensor(0.0, device=device, dtype=torch.bfloat16, requires_grad=True)
    total_token_count = 0
    
    # Detach old logps for stable gradient computation
    detached_per_token_logps = [logps.detach() for logps in all_per_token_logps]
    
    # Split processing into chunks to avoid OOM
    for mb_idx in range(len(all_input_ids)):
      input_ids = all_input_ids[mb_idx]
      attention_mask = all_attention_masks[mb_idx]
      completion_mask = all_completion_masks[mb_idx]
      logits_to_keep = all_logits_to_keep[mb_idx]
      old_per_token_logps = detached_per_token_logps[mb_idx]
      per_token_kl = all_per_token_kl[mb_idx]
      mb_advantages = advantages[mb_idx * input_ids.size(0):(mb_idx + 1) * input_ids.size(0)]

      # Compute current log probabilities for this chunk (with gradients enabled)
      current_per_token_logps = get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
      
      # Compute PPO ratio
      ratio = torch.exp(current_per_token_logps - old_per_token_logps)
      clipped_ratio = torch.clamp(ratio, min=1-epsilon, max=1+epsilon)
      
      # Compute surrogate loss
      per_token_loss1 = ratio * mb_advantages.unsqueeze(1)
      per_token_loss2 = clipped_ratio * mb_advantages.unsqueeze(1)
      per_token_loss = -torch.minimum(per_token_loss1, per_token_loss2)
          
      # Add KL penalty
      per_token_loss = per_token_loss + grpo_beta * per_token_kl
      
      # Sum loss and update metrics
      token_count = completion_mask.sum()
      loss = (per_token_loss * completion_mask).sum() / token_count
      loss.backward()
      total_loss = total_loss + loss * token_count
      total_token_count += token_count
      
      # Calculate clip ratio for metrics
      is_clipped = (ratio < 1-epsilon).float() + (ratio > 1+epsilon).float()
      clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
      kl_div = (per_token_kl * completion_mask).sum() / completion_mask.sum()
      
      # Update metrics
      aggregated_metrics.update({
          'clip_ratio': aggregated_metrics.get('clip_ratio', 0) + clip_ratio.item(),
          'kl': aggregated_metrics.get('kl', 0) + kl_div.item(),
      })
      del current_per_token_logps
      del old_per_token_logps
      del per_token_kl
      del mb_advantages
      del completion_mask
      del input_ids
      
      # Clear cache after each chunk
      torch.cuda.empty_cache()
      gc.collect()

    # Normalize the total loss and metrics
    if total_token_count > 0:
      total_loss = total_loss / total_token_count
    
    # Average metrics over all chunks
    num_chunks = len(all_input_ids)
    if num_chunks > 0:
      for key in ['clip_ratio', 'kl']:
        if key in aggregated_metrics:
          aggregated_metrics[key] /= num_chunks
    
    # Add reward metrics
    aggregated_metrics.update({
        'reward': rewards.mean().item(),
        'reward_std': std_grouped_rewards.mean().item(),
        'generation_time': sum(all_generation_times) / len(all_generation_times) if all_generation_times else 0
    })
    
    # Add per-function reward metrics
    reward_per_func = rewards_per_func.mean(0)
    for i, reward_func in enumerate(reward_funcs):
        aggregated_metrics[f"{reward_func.__name__}"] = reward_per_func[i].item()
    
    # Synchronize error flags one last time before returning
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(error_flag, op=dist.ReduceOp.MAX)
    
    # Final cleanup
    torch.cuda.empty_cache()
    gc.collect()
    
    return total_loss, aggregated_metrics, error_flag




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
  base_model = get_base_model(model)

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
  base_model = get_base_model(model)
  
  # Map source parameter names to their corresponding target parameter names
  source_to_target_map = {}
  
  # Build the mapping between source and target parameters
  for name, _ in base_model.named_parameters():
    if source_adapter_name in name:
      target_name = name.replace(source_adapter_name, target_adapter_name)
      source_to_target_map[name] = target_name
  
  # Copy parameters directly without storing intermediate copies
  base_model_named_parameters = dict(base_model.named_parameters())
  for source_name, target_name in source_to_target_map.items():
    # Get parameters by name to avoid storing all parameters in memory
    source_param = base_model_named_parameters[source_name]
    target_param = base_model_named_parameters[target_name]
    # Copy data directly without creating additional clones
    if source_param.data.numel() == target_param.data.numel():
      # Make sure shapes match without trying to reshape empty tensors
      if source_param.data.shape == target_param.data.shape:
        target_param.data.copy_(source_param.data)
      else:
        print(f"Warning: Shape mismatch. Source: {source_param.data.shape}, Target: {target_param.data.shape}")
        # Only reshape if source has data
        target_param.data.copy_(source_param.data.reshape(target_param.data.shape))
    # Free memory after each copy
    torch.cuda.empty_cache()
  
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

  # Regular expressions used during Chain-of-Thoughts (COTs) rewards calculation and fault simulation.
  reward_kwargss = prepare_reward_kwargs(hps, reward_funcs)
  # Initialize logging step
  logging_step = 0
  # Start training
  if is_main_process():
    print(f"Optimizer's step every: {hps.gradient_accumulation_steps}")
  # Initialize best_avg_reward to a negative value
  hps.best_avg_reward = -np.inf
  
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
        # Initialize pass@k metrics
        pass_key = f'pass@{hps.num_generations}'
        pass_at_k = dict()

      # Compute loss and get metrics
      micro_batch_loss, micro_batch_metrics, error_flag = compute_loss(
          model=model,
          data_iterator=data_iterator,
          reward_funcs=reward_funcs,
          reward_kwargss=reward_kwargss,
          hps=hps,
          step=epoch*len(dataloader) + i
      )

      # Handle errors during compute_loss
      if error_flag.item() > 0:
        if is_main_process():
          pbar.update(1)
        count_skip_micro_batch += 1
        # Ensure all processes are synchronized after an error
        if dist.is_available() and dist.is_initialized():
          dist.barrier()
        continue

      if pass_key not in pass_at_k.keys():
        pass_at_k[pass_key] = micro_batch_metrics.pop(pass_key)
      else:
        pass_at_k[pass_key] = np.concatenate([pass_at_k[pass_key], micro_batch_metrics.pop(pass_key)], axis=0)
      # Scale loss for gradient accumulation
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
          torch.cuda.empty_cache()
          gc.collect()
        
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
        metrics[pass_key].append(smart_round(pass_at_k[pass_key].any(1).mean()))
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
            infer(wrapped_model=model, dataloader=dataloader, tokenizer=hps.tokenizer, model_max_length=hps.model_max_length, epoch=1, file_path=best_model_file_path, parallel=hps.parallel, new_file=not os.path.exists(best_model_file_path))
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
          # Dynamically create a list of columns to display
          display_columns = ['batch', 'batch_loss', 'avg_clip_ratio', 'avg_kl', 'avg_reward', 'avg_reward_std']
          # Show averaged metrics in progress bar
          pbar.set_postfix(x.loc[x.index[-1], display_columns + ['best_avg_reward']].to_dict())
          if logging_step % logging_steps == 0:
            # Add any reward function metrics that exist in the dataframe
            reward_metrics = [col for col in x.columns if col.startswith('avg_') and col.endswith('_reward')]
            display_columns.extend(reward_metrics)
            print_rich_table(x.loc[x.index[-logging_steps:], display_columns + ['best_avg_reward']])
            # print(tabulate(x.loc[x.index[-logging_steps:], display_columns + ['best_avg_reward']], headers='keys', tablefmt='psql', showindex=False))
            logging_step = 0
          logging_step += 1
      
      if is_main_process():
        pbar.update(1)

      if DEBUG and i > 500:
        break

  x = pd.DataFrame(metrics)
  # Log final complete table
  if hps.wandb:
    # Convert string values to float, handling any non-numeric values and None values
    x = x.fillna(0).applymap(lambda val: float(eval(val)) if isinstance(val, str) else val)
    wandb.log({"rlft_complete_history": wandb.Table(dataframe=x)})
  if is_main_process():
    print_rich_table(x)
    # print(tabulate(x, headers='keys', tablefmt='psql', showindex=False))
    pbar.close()


def validate_model(model, validation_loader, hps, reward_funcs=[test_generation_reward], num_prompts_to_validate=4_000):
  generate_kwargs = {"max_length": hps.model_max_length, "num_return_sequences": 1, "temperature": 0.6, "top_p": 0.6, "top_k": 5, "pad_token_id": hps.tokenizer.eos_token_id, "num_beams": 4}
  print(generate_kwargs)
  validation_loader.collate_fn.set_left_padding()
  validation_loader_iterator = iter(validation_loader)
  total_steps = max(1, min(1000, round(num_prompts_to_validate // (validation_loader.batch_size * hps.world_size))))
  if is_main_process():
    pbar = tqdm(total=total_steps, desc=f"[{hps.local_rank}]: Validation...", disable=hps.disable_tqdm)
  base_model = get_base_model(model)
  base_model.set_adapter("grpo_adapter")
  if not isinstance(reward_funcs, list):
    reward_funcs = [reward_funcs]
  reward_kwargss = prepare_reward_kwargs(hps, reward_funcs)
  total_completions = []
  total_prompts = []
  total_netlists = []
  for i in range(total_steps): 
    data = next(validation_loader_iterator)
    with torch.inference_mode():
      with unwrap_model(model) as unwrapped_model:
        generate_kwargs.update({"input_ids": data.input_ids.to(hps.device), "attention_mask": data.attention_mask.to(hps.device)})
        completions = unwrapped_model.generate(**generate_kwargs)
        prompt_length = data.input_ids.size(1)
        completions = completions[:, prompt_length:]
        completions = hps.tokenizer.batch_decode(completions, skip_special_tokens=True)
        total_completions.extend(completions)
        prompts = get_user_prompt(hps.tokenizer.batch_decode(data.input_ids))
        prompts = [prompt for prompt in prompts for _ in range(hps.num_generations)]
        netlists = [data.netlist for _ in range(hps.num_generations)]
        total_prompts.extend(prompts)
        total_netlists.extend(netlists)
        if is_main_process():
          pbar.update(1)
        torch.cuda.empty_cache()
        gc.collect()
  metrics = {}

  rewards_per_func = torch.zeros(len(total_prompts), len(reward_funcs), device=hps.device, dtype=torch.bfloat16)
  for reward_idx, (reward_func, reward_kwargs) in enumerate(zip(reward_funcs, reward_kwargss)):
    if reward_func.__name__ == "test_generation_reward":
      reward_kwargs.update({"netlists": total_netlists})
    returned_rewards_list = reward_func(total_prompts, total_completions, **reward_kwargs)
    returned_rewards_per_func = torch.zeros(len(returned_rewards_list), dtype=torch.bfloat16, device=hps.device)
    divisor = 1.0 / len(returned_rewards_list)
    for r_i, returned_rewards in enumerate(returned_rewards_list):
      returned_rewards_per_func[r_i] = sum(returned_rewards.values())
      for k, v in returned_rewards.items():
        metrics_key = f"{reward_func.__name__}" + (f"/{k}" if k else "")
        metrics[metrics_key] = metrics.get(metrics_key, 0) + v * divisor      
    rewards_per_func[:, reward_idx] = returned_rewards_per_func
  rewards_per_func = gather(rewards_per_func).mean(dim=0)
  if is_main_process():
    print(rewards_per_func)
    print(pd.DataFrame([metrics]))
  if hps.wandb:
    wandb.log({"validation_metrics": wandb.Table(dataframe=pd.DataFrame([metrics]))})


def fine_tuning(dataloader, validation_loader, model, hps, training_loop=True):
  # Fine-tune with Supevised Fine-Tuning (SFT)
  # 1. Train new embeddings tokens and head
  if hps.new_tokens:
    if hps.wandb:
      hps.run.tags += ('train_new_embeddings',)
    sft(dataloader, model, hps, desc="SFT Embeddings Training...", training_loop=training_loop)
    # Apply inference on some random samples of validation set
    if not DEBUG:
      infer(wrapped_model=model, dataloader=validation_loader, tokenizer=hps.tokenizer, model_max_length=hps.model_max_length, epoch=1, file_path=hps.file_path, parallel=hps.parallel, new_file=True)
      save_model(model, hps.save_in_repo, push_to_hub=True, save_embedding_layers=True, hps=hps, commit_message="Model trained with new embeddings")

  hps.lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM if hps.is_causal else TaskType.SEQ_2_SEQ_LM, r=hps.lora_r, lora_alpha=hps.lora_alpha, lora_dropout=hps.lora_dropout)
  if not hps.adapter_name or not hps.adapter_repo:
    # This block of code is for adding lora weights when either fsdp or ddp is used
    # Check for DDP or FSDP wrapped modules using attribute names
    base_model = model
    if hps.parallel:
      base_model = get_base_model(model)

    # Check if the model is already a PeftModel
    is_peft_model = hasattr(base_model, 'peft_config') or isinstance(base_model, PeftModel)

    if not is_peft_model:
      # Set LoRA configuration: LoRA (Low-Rank Adaptation) is used to efficiently fine-tune large models by adding low-rank matrices to the model's weights.
      peft_model = get_peft_model(base_model, hps.lora_config, adapter_name="ref_adapter")
    else:
      # Model is already a PeftModel, no need to convert it
      peft_model = base_model

    if is_main_process():
      print(f"Applying LoRA to the model...")
    # Activate the newly added LoRA adapter to use the LoRA weights during training
    if hps.parallel:
      model = apply_lora_distributed(model, peft_model, hps.lora_config, adapter_name="ref_adapter")
    else:
      model = apply_lora_non_distributed(model, peft_model, hps.lora_config, adapter_name="ref_adapter")  
  elif hps.adapter_name and hps.adapter_repo:
    if hps.parallel:
      if isinstance(model, DDP):
        model.module = PeftModel.from_pretrained(model.module, hps.adapter_repo, subfolder=hps.adapter_name, adapter_name=hps.adapter_name).to(torch.bfloat16)
      elif isinstance(model, FSDP):
        # We need to completely unwrap the model first
        # Get the base model without FSDP wrapping
        with FSDP.summon_full_params(model, recurse=True, writeback=False):
          # Create a deep copy of the unwrapped module to avoid FSDP references
          base_model_config = model._fsdp_wrapped_module.config
          base_model_state = {k: v.clone().detach().cpu() for k, v in model._fsdp_wrapped_module.state_dict().items()}
        del model
        torch.cuda.empty_cache()
        gc.collect()

        print("Creating clean model from config. It might take a while...")
        # Create a model on meta device to avoid redundant memory allocation
        clean_model = AutoModelForCausalLM.from_pretrained(hps.model_name, 
                                                           attn_implementation="flash_attention_2", 
                                                           config=base_model_config)
        
        # Move model to CPU first for efficient memory management
        clean_model = clean_model.to('cpu')
        
        # Load state dict with non-blocking transfers and pin memory for faster GPU transfer
        for key, param in base_model_state.items():
          if key in clean_model.state_dict():
            clean_model.state_dict()[key].copy_(param.to('cpu', non_blocking=True))
        
        # Clear CUDA cache to free up memory before next operations
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Ensure all operations complete
        # Now apply PEFT to the clean model
        peft_model = PeftModel.from_pretrained(
            clean_model,
            hps.adapter_repo,
            subfolder=hps.adapter_name,
            adapter_name=hps.adapter_name
        ).to(torch.bfloat16)
        
        # Re-wrap with FSDP
        model = FSDP(
            peft_model,
            # device_id=hps.local_rank,
            auto_wrap_policy=hps.my_auto_wrap_policy,
            mixed_precision=hps.mixed_precision,
            use_orig_params=True,
            # ignored_modules=[m for name, m in peft_model.named_modules() if 'lora' in name.lower()]
        )
    else:
      model = PeftModel.from_pretrained(model, hps.adapter_repo, subfolder=hps.adapter_name, adapter_name=hps.adapter_name).to(torch.bfloat16)

  # Handle unused parameters based on model type
  if hps.parallel:
    if isinstance(model, DDP):
      model.find_unused_parameters = True  # Set find_unused_parameters to True to avoid OOM error

  # Configure which parts of the model to train in step 2
  train_layers(model, train_embeddings=True, train_head=True, train_lora=True, train_base_model=False)

  # Initialize the optimizer with parameter groups
  # Get only trainable parameters to optimize memory usage and training efficiency
  trainable_params = [p for p in model.parameters() if p.requires_grad]
  if hps.parallel and not hps.fsdp:
    hps.optimizer = ZeroRedundancyOptimizer(
      trainable_params,  # Only parameters that require gradients
      optimizer_class=AdamW,
      lr=hps.lr
    )
  else:
    hps.optimizer = AdamW(
      trainable_params,  # Only parameters that require gradients
      lr=hps.lr
    )

  # Calculate the total number of training steps
  total_training_steps = (hps.epochs * len(dataloader)) // hps.gradient_accumulation_steps
  # Set up the learning rate scheduler. Uses a cosine schedule with warmup
  hps.scheduler = get_cosine_schedule_with_warmup(hps.optimizer, num_warmup_steps=10, num_training_steps=total_training_steps, num_cycles=3/20)

  if hps.train_lora:
    dataloader.collate_fn.set_train_lora(True)
    if is_main_process():
      print(f"{model}\nSFT embedding, lora and head:\nModel training parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,d}\nTrainable Model size: {convert_bytes(model_size_in_bytes(model))}\n")
    if hps.wandb:
      hps.run.tags += ('train_lora',)
    # 2. Train adapter (LoRA weights) as well
    sft(dataloader, model, hps, desc="SFT Lora Training...", training_loop=training_loop) 
    # Apply inference on some random samples of validation set
    if not DEBUG:
      infer(wrapped_model=model, dataloader=validation_loader, tokenizer=hps.tokenizer, model_max_length=hps.model_max_length, epoch=1, file_path=hps.file_path, parallel=hps.parallel, new_file=True)
      save_model(model, hps.save_in_repo, push_to_hub=True, save_embedding_layers=True, hps=hps, commit_message="Model trained with LoRA")

  # Configure which parts of the model to train in step 3
  train_layers(model, train_embeddings=False, train_head=False, train_lora=True, train_base_model=False)

  # Add GRPO adapter
  if hps.parallel:
    if isinstance(model, DDP):
      # Check if the model already has ref_adapter or grpo_adapter
      available_adapters = model.module.base_model.peft_config.keys() if hasattr(model.module.base_model, 'peft_config') else []
      
      if "ref_adapter" in available_adapters:
        model.module.base_model.add_adapter(hps.lora_config, adapter_name="grpo_adapter")
        # Copy the weights from the reference adapter to the GRPO adapter
        copy_adapter_weights(model, source_adapter_name="ref_adapter", target_adapter_name="grpo_adapter")
        model.module.set_adapter("grpo_adapter")
      elif "grpo_adapter" in available_adapters:
        model.module.base_model.add_adapter(hps.lora_config, adapter_name="ref_adapter")
        # Copy the weights from the GRPO adapter to the reference adapter
        copy_adapter_weights(model, source_adapter_name="grpo_adapter", target_adapter_name="ref_adapter")
        model.module.set_adapter("grpo_adapter")
    elif isinstance(model, FSDP):
      model._fsdp_wrapped_module.base_model.add_adapter(hps.lora_config, adapter_name="grpo_adapter")
      # Copy the weights from the reference adapter to the GRPO adapter
      copy_adapter_weights(model, source_adapter_name="ref_adapter", target_adapter_name="grpo_adapter")
      model._fsdp_wrapped_module.set_adapter("grpo_adapter")
  else:
    model.base_model.add_adapter(hps.lora_config, adapter_name="grpo_adapter")
    # Copy the weights from the reference adapter to the GRPO adapter
    copy_adapter_weights(model, source_adapter_name="ref_adapter", target_adapter_name="grpo_adapter")
    model.set_adapter("grpo_adapter")
  torch.cuda.empty_cache()
  gc.collect()

  # Set the flags for distributed systems. Data samplers and Data Loaders
  use_sampler = hps.parallel==True and hps.deepspeed_kernel==False
  # Shuffle is handled by the sampler
  hps.shuffle = not use_sampler
  # Randomly select 200,000 samples from the dataset to shorten the training time
  random_indices = random.sample(range(len(dataloader.dataset)), min(200_000, len(dataloader.dataset)))
  dataset_subset = dataloader.dataset.select(random_indices)
  # Replicate the sampler across all processes
  sampler = DistributedSampler(dataset_subset, rank=dataloader.sampler.rank, num_replicas=dataloader.sampler.num_replicas, shuffle=True) if use_sampler else None  

  # Adjust gradient accumulation steps. GRPO is slower than SFT. lower the number of gradient accumulation steps.
  hps.gradient_accumulation_steps = max(1, int(hps.gradient_accumulation_steps//hps.num_generations))
  hps.epochs = 1

  # Change batch size. Due to number of generations there might be OOM cuda error.
  hps.micro_batch_size = max(1, hps.micro_batch_size//hps.num_generations)
  dataloader = DataLoader(dataset=dataset_subset, batch_size=hps.micro_batch_size, shuffle=hps.shuffle, collate_fn=MyCollate(tokenizer=hps.tokenizer, is_causal=hps.is_causal, lora=hps.lora), sampler=sampler, num_workers=dataloader.num_workers, pin_memory=dataloader.pin_memory, drop_last=dataloader.drop_last)
  torch.cuda.empty_cache()
  gc.collect()

  hps.lr = max(1e-7, min(1e-5, hps.lr))
  trainable_params = [p for p in model.parameters() if p.requires_grad]
  if hps.parallel and not hps.fsdp:
    hps.optimizer = ZeroRedundancyOptimizer(
      trainable_params,
      optimizer_class=AdamW,
      lr=hps.lr
    )
  else:
    hps.optimizer = AdamW(
      trainable_params,
      lr=hps.lr
    )

  # Configure which parts of the model to train
  # Calculate the total number of training steps
  total_training_steps = (hps.epochs * len(dataloader)) // hps.gradient_accumulation_steps
  hps.scheduler = get_cosine_schedule_with_warmup(hps.optimizer, num_warmup_steps=1, num_training_steps=total_training_steps, num_cycles=3/20)

  if is_main_process():
    print(f"{model}\nGRPO-RLFT train embedding, lora and head:\nTrainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,d}\nTrainable Model size: {convert_bytes(model_size_in_bytes(model))}")
  
  # 3, Fine-tune with Reinforcement Learning (RL)
  if hps.wandb:
    hps.run.tags += ('train_rl_grpo',)
  rlft(dataloader, model, hps, reward_funcs=[test_generation_reward], training_loop=training_loop)
  # Apply inference on some random samples of validation set
  if not DEBUG: 
    base_model = get_base_model(model)
    base_model.set_adapter("grpo_adapter")
    # Apply inference on some random samples of validation set
    infer(wrapped_model=model, dataloader=validation_loader, tokenizer=hps.tokenizer, model_max_length=hps.model_max_length, epoch=1, file_path=hps.file_path, parallel=hps.parallel, new_file=True)
    # Save the model in the repository
    save_model(model, hps.save_in_repo, push_to_hub=True, save_embedding_layers=True, hps=hps, commit_message="Final Model trained with GRPO-RL")
    # Save the model in the local directory
    final_model_path = os.path.join(hps.log_dir, "models", "final_model", hps.save_in_repo.split("/")[-1])
    os.makedirs(final_model_path, exist_ok=True)
    save_model(model, final_model_path, push_to_hub=False, save_embedding_layers=True, hps=hps, delete_previous=False)
  
  torch.cuda.empty_cache()
  gc.collect()
  dist.barrier()

  # Validate the model
  validate_model(model, validation_loader, hps, reward_funcs=[test_generation_reward])
  

def main():
  # Arguments
  parser = argparse.ArgumentParser(description='Training arguments parser')

  # Initialize hyperparameters using custom AttrDict()
  hps = hyperparameters(parse_arguments(parser))
  
  # Set debug mode
  hps.debug = DEBUG

  # Set the model name
  hps.model_name = models_causal[hps.model_name]
  hps.adapter_repo = models_causal[hps.adapter_repo] if hps.adapter_repo is not None and hps.adapter_repo in models_causal.keys() else hps.adapter_repo
  hps.save_in_repo = models_causal[hps.save_model]

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

