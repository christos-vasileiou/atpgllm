# Helper Functions
import logging.handlers
import torch
import argparse
import os
import logging
import copy
import wandb
import gc
import shutil
import ast
import numpy as np
import regex as re
import matplotlib.pyplot as plt
import subprocess
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, BackwardPrefetch, ShardingStrategy, CPUOffload, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, enable_wrap, wrap
from torch.distributed.optim import ZeroRedundancyOptimizer
from peft import get_peft_model, LoraConfig, TaskType
from tqdm.auto import tqdm
from collections import Counter
from pathlib import Path
from accelerate import Accelerator
from huggingface_hub import upload_folder
from itertools import product
from datetime import datetime
from datasets import DatasetDict, Dataset, load_dataset
from sklearn.metrics import accuracy_score, precision_score, f1_score
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from transformers import (
  AutoModelForCausalLM,
  AutoModelForSeq2SeqLM,
  BitsAndBytesConfig,
  AutoTokenizer,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
def is_rich_available():
  """Check if the rich library is available."""
  try:
    import rich
    return True
  except ImportError:
    return False

if is_rich_available():
  from rich.table import Table
  from rich.console import Console
  from rich.panel import Panel
  from rich.text import Text

class AttrDict(dict):
  """
  A dictionary subclass that allows attribute-style access to dictionary keys.
  
  This class enables dot notation access to dictionary items, making it more
  convenient to work with configuration and parameter dictionaries.
  
  Example:
      config = AttrDict({"learning_rate": 0.01})
      print(config.learning_rate)  # Access via attribute
      print(config["learning_rate"])  # Access via key
  """
  def __init__(self, initial_dict=None, **kwargs):
    # Initialize as a regular dictionary
    super(AttrDict, self).__init__()
    # If an initial dictionary is provided, update with its contents
    if initial_dict is not None:
      self.update(initial_dict)
    # Update with any keyword arguments provided
    self.update(kwargs)

  def __getattr__(self, key):
    # Allow dictionary keys to be accessed as attributes
    try:
      # Try to get the key from the dictionary
      return self[key]
    except KeyError:
      # If the key doesn't exist, raise an AttributeError (standard behavior for attribute access)
      raise AttributeError(f"No such attribute: {key}")

  def __setattr__(self, key, value=None):
    # Allow setting dictionary items using attribute syntax
    if isinstance(key, str):
      # Store the key-value pair in the dictionary
      self[key] = value
    else:
      # Ensure keys are strings to maintain expected behavior
      raise ValueError(f'Invalid attribute assignment. You passed as key: {key} and value: {value}')

  def __dir__(self):
    # Enhance dir() output to include dictionary keys for better IDE autocompletion
    return super().__dir__() + list(self.keys())

def patterns_lengths(batch, max_freq):
  patterns_lengths = [len(example.split('\n')[0].split()) for example in batch['patterns']]
  batch['patterns_lengths'] = [l if l == max_freq else None for l in patterns_lengths]
  return batch


def initial_data_preprocessing(raw_dataset: DatasetDict):
    """
    Perform initial data preprocessing steps such as tokenizing, cleaning, and filtering.

    Args:
        raw_dataset (DatasetDict): The raw dataset containing the prompts and patterns.

    Returns:
        DatasetDict: The preprocessed dataset.
    """
    lengths = []
    for i in tqdm(range(0, len(raw_dataset['train']['patterns']), 1_000)):
        batch = raw_dataset['train']['patterns'][ i : i+1_000 ]
        lengths.extend([len(example.split('\n')[0].split()) for example in batch])
    freq = Counter(lengths)
    max_freq = max(freq, key=freq.get)
    print(freq)
    print(max_freq)

    raw_dataset = raw_dataset.map(
        patterns_lengths,
        batched=True,
        fn_kwargs={'max_freq': max_freq}
    )
    raw_dataset = raw_dataset.filter(lambda batch: batch['patterns_lengths'] is not None)
    return raw_dataset, max_freq


def remove_trailing_tokens(term, token=0):
  reversed_term = list(reversed(term))
  temp_term = reversed_term.copy()
  #print('reversed:\n', reversed_term[0], '\ntoken:\n', token)
  for i in reversed_term:
    if (i == token).all() and (reversed_term[0] == token).all():
      temp_term.remove(i)
  #print('temp_term:\n', temp_term, '\n\n')
  return len(temp_term)


def length_less_than_model_max_len(tokenized, tokenizer, train_test=None, key=None, token=0):
  batch = tokenized if train_test is None else tokenized[train_test]
  batch = batch if key is None else batch[key]
  non_zero_batch_lengths = [remove_trailing_tokens(input_ids, token=token) for input_ids in batch]
  non_zero_batch_lengths = torch.tensor([non_zero_length if non_zero_length < tokenizer.model_max_length else -1 for non_zero_length in non_zero_batch_lengths])
  return non_zero_batch_lengths


def tokenize_less_modelmaxlen_fn(batch, tokenizer):
  tokenized_inputs = tokenizer(batch['prompts'], padding=True, truncation=True, return_tensors='pt')
  tokenized_inputs['input_ids_lengths'] = length_less_than_model_max_len(tokenized_inputs, tokenizer, train_test=None, key='input_ids', token=0)
  answers_sorted_byline = ['\n'.join(sorted(patterns.split('\n'))) for patterns in batch['patterns']]
  tokenized_targets = tokenizer(text_target=answers_sorted_byline, padding=True, truncation=True, return_tensors='pt') 
  tokenized_inputs['labels'] = tokenized_targets['input_ids'] if type(tokenizer).__name__ == 'ATPGTokenizer' else tokenized_targets['input_ids'][:, 1:-1]
  return tokenized_inputs


def collate_fn_dict(batch):
  keys = batch[0].keys()
  collated_batch = {key: torch.stack([torch.tensor(item[key], dtype=torch.float) for item in batch]) for key in keys}
  return collated_batch


def get_patterns_info_mapping(max_freq):
  id2possible_labels = {int(i): l for i, l in enumerate([' '.join(p) for p in product('01', repeat=max_freq)])}
  possible_labels2id = {l: int(i) for i, l in id2possible_labels.items()}
  return id2possible_labels, possible_labels2id


def patterns_contains_special_tokens(tokenizer, pattern):
  if type(tokenizer).__name__ == 'BertTokenizerFast':
    if tokenizer.pad_token_id in pattern or tokenizer.sep_token_id in pattern:
      return True
  elif type(tokenizer).__name__ == 'ATPGTokenizer':
    if tokenizer.stoi[tokenizer.eos[0]] in pattern or tokenizer.stoi[tokenizer.pad[0]] in pattern:
      return True
  elif type(tokenizer).__name__ == 'GraphTokenizer':
    if tokenizer.stoi[tokenizer.eos[0]] in pattern or tokenizer.stoi[tokenizer.pad[0]] in pattern:
      return True
  return False


def align_labels_into_matrix(batch, max_freq, max_num_of_patterns_per_circuit, tokenizer, possible_labels2id):
  tokenized = batch.copy()
  # Re assign them to tensors
  for k in tokenized.keys():
    tokenized[k] = torch.tensor(tokenized[k])
  #print(f"{type(tokenizer).__name__}: tokenized['labels']: {tokenized['labels'].shape}")
  tokenized['labels'] = tokenized['labels'].view(-1, tokenized['labels'].shape[-1]//max_freq, max_freq)
  # Create that many samples as required to match the max number of patterns. Not only in the batch!
  temp       = torch.zeros(tuple(dim if i!=1 else max_num_of_patterns_per_circuit-tokenized['labels'].shape[1] for i, dim in enumerate(tokenized['labels'].shape)))
  # Trace the indices of the labels.
  bool_true  = torch.ones_like(tokenized['labels'], dtype=torch.bool)
  bool_false = torch.zeros_like(temp, dtype=torch.bool)
  # Concatenate the booleans and the labels-padding
  valid_indices       = torch.cat([bool_true, bool_false], dim=1)
  tokenized['labels'] = torch.cat((tokenized['labels'], temp), dim=1)

  assert tokenized['labels'].shape == valid_indices.shape

  #print(tokenized['labels'].shape)
  # Create 2D representation of the labels.
  matrix = [torch.full((2**max_freq,), 0.)  for _ in range(tokenized['labels'].shape[0])]
  # matrix = [torch.full((2**max_freq, max_freq), -1) for _ in range(tokenized['labels'].shape[0])]
  for b in range(tokenized['labels'].shape[0]):
    for pattern, _idx in zip(tokenized['labels'][b], valid_indices[b]):
      if patterns_contains_special_tokens(tokenizer, pattern):
        continue
      if _idx.all().item():
        matrix[b][possible_labels2id[tokenizer.decode(pattern.to(int))]] = 1.
  tokenized['labels'] = torch.stack(matrix)
  # filter out samples longer than model max length
  valid_samples = tokenized['input_ids_lengths'] != -1
  for key in tokenized:
    tokenized[key] = tokenized[key][valid_samples]
  return tokenized


def model_size_in_bytes(model) -> int:
  total_size = 0
  for p in model.parameters():
    if p.requires_grad:
      total_size += sizeof_tensor(p)
  return total_size


def convert_bytes(num_bytes: int) -> str:
  for unit in ['bytes', 'KB', 'MB', 'GB', 'TB']:
    if num_bytes < 1024:
      return f"{num_bytes:.4f} {unit}"
    num_bytes /= 1024


def get_pos_weight(possible_labels2id, raw_dataset, max_freq):
  patterns         = [possible_labels2id[pattern] for patterns in raw_dataset['train']['patterns'] for pattern in patterns.split('\n')]
  counter          = Counter(patterns)
  population       = np.array([(i, p, p/len(patterns)) for i, p in counter.most_common()])
  population[:, 2] = population[:, 2][::-1]
  population[:, 2] = (population[:, 2] - population[:, 2].mean()) / (population[:, 2].std()+.001)
  population[:, 2] = population[:, 2] + (population[:, 2].max() - population[:, 2].min())
  pos_weight       = torch.ones([2**max_freq])
  for i, n, w in population:
    pos_weight[int(i)] = w
  return pos_weight


def create_data_loader(dataset, batch_size=4, shuffle=False, collate_fn=None, sampler=None, num_workers=0):
  return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, sampler=sampler, num_workers=num_workers)


def save_checkpoint(model, optimizer, epoch, local_rank, train_loss_to_print, val_loss_to_print, filename='model_checkpoint.pth'):
  """
  Save model, optimizer, and epoch number in a date and time-specific folder.
  """
  print(f"\n{model}")
  # Get the current date and time as a string (e.g., '2023_04_01_15_30')
  current_datetime = datetime.now().strftime('%Y_%m_%d_%H_%M')
  # Create a directory with the current date and time
  save_dir = os.path.join(f'models/{type(model).__name__}', current_datetime)
  os.makedirs(save_dir, exist_ok=True)
  # Save the model under the date-time-specific folder
  save_path_checkpoint = os.path.join(save_dir, f"{local_rank}_{filename}")
  save_path_pics       = os.path.join(save_dir, f"train_val_plt.png")
  checkpoint = {
    'epoch': epoch,
    'model': model,
    'optimizer_state_dict': optimizer.state_dict(),
    'hyperparameters': {
      'learning_rate': optimizer.param_groups[0]['lr'],
      'model_architecture': str(model),
    }
  }
  
  plt.plot(train_loss_to_print, label='train')
  plt.plot(val_loss_to_print, label='validation')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.savefig(save_path_pics)

  torch.save(checkpoint, save_path_checkpoint)
  print(f"Model saved at {save_path_checkpoint}")


def load_checkpoint(date_time=None, local_rank=0, filename='model_checkpoint.pth', model=None, optimizer=None):
  """
  Load model, optimizer, and epoch number from a date and time-specific folder.
  """
  if date_time is None:
    # If no date or time provided, load the latest
    model_folders = sorted([d for d in os.listdir('models') if os.path.isdir(os.path.join(f"models/{type(model).__name__}", d))])
    datetime_str = model_folders[-1]  # Last folder (latest date and time)
  else:
    datetime_str = f"{date_time}"
  load_path = os.path.join(f"models/{type(model).__name__}", datetime_str, f"{local_rank}_{filename}")
  #print(f"load path: {load_path}")
  checkpoint = torch.load(load_path, map_location=torch.device('cpu'))
  if 'model_state_dict' in checkpoint.keys():
    model.load_state_dict(checkpoint['model_state_dict'])
  elif 'model' in checkpoint.keys():
    model = checkpoint['model']
  if optimizer:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  epoch = checkpoint['epoch']
  hyperparameters = checkpoint.get('hyperparameters', {})
  return model, optimizer, epoch, hyperparameters


def seconds_to_dhms(seconds):
  days, remainder  = divmod(seconds, 86400)
  hours, remainder = divmod(remainder, 3600)
  minutes, seconds = divmod(remainder, 60)
  if days>0:
    return f"{int(days)} days, {int(hours):02}:{int(minutes):02}:{int(seconds):02}"
  else:
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"


def setup_logging(hps: AttrDict):
  if not is_main_process():
    return None

  from logging.handlers import RotatingFileHandler
  import uuid
  # Ensure the logs directory exists
  hps.log_dir = "logs"
  os.makedirs(hps.log_dir, exist_ok=True)

  # Create a logger
  logger = logging.getLogger(__name__)
  logger.setLevel(logging.DEBUG)
  # Create a unique identifier for log files
  unique_id = uuid.uuid4().hex[:8]

  # create log filename
  log_info_filename = os.path.join(hps.log_dir, f"info.log")
  log_debug_filename = os.path.join(hps.log_dir, f"debug.log")

  # Create a formatter
  formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

  # Create a file handlers
  # Set INFO handler
  info_fh = RotatingFileHandler(log_info_filename, maxBytes=10**6, backupCount=20, encoding="utf-8")
  # Set Level
  info_fh.setLevel(logging.INFO)
  # Add formatter to file handlers
  info_fh.setFormatter(formatter)
  # Add handler into the logger
  logger.addHandler(info_fh)

  # Set DEBUG handler
  debug_fh = RotatingFileHandler(log_debug_filename, maxBytes=10**6, backupCount=20, encoding="utf-8")
  # Set Level
  debug_fh.setLevel(logging.DEBUG)
  # Add formatter to file handlers
  debug_fh.setFormatter(formatter)
  # Add handler into the logger
  logger.addHandler(debug_fh)

  console_handler = logging.StreamHandler()
  console_handler.setLevel(logging.INFO)
  console_handler.setFormatter(formatter)
  logger.addHandler(console_handler)

  return logger


def move_optimizer_to_device(optimizer, device):
  """
  Move an optimizer to the specified device.

  Args:
  - optimizer (torch.optim.Optimizer): The optimizer to be moved.
  - device (torch.device): The target device.

  Returns:
  - None
  """
  for state in optimizer.state.values():
    for key, value in state.items():
      if isinstance(value, torch.Tensor):
        state[key] = value.to(device)


def labels_filtering_loss(outputs, labels, criterion, llambda=1, vllambda=1, epoch=1, loss_ratio=None):
  # Define special values based on patterns
  special_value = 0
  
  # Filter valid and invalid samples
  valid_samples    = labels != special_value
  invalid_samples  = ~valid_samples # Inverse of valid_samples
  
  # Calculate valid and invalid outputs and labels
  valid_outputs,   valid_labels    = outputs[valid_samples],   labels[valid_samples]
  invalid_outputs, invalid_labels  = outputs[invalid_samples], labels[invalid_samples]	
  
  # Calculate base, valid, and invalid losses
  loss             = llambda * criterion[0](outputs, labels)
  # Binary CrossEntropy Loss for multilabel - sigmoid-based
  valid_loss       = criterion[0](valid_outputs, valid_labels)
  invalid_loss     = criterion[0](invalid_outputs, invalid_labels)
  
  # Regularization terms
  if loss_ratio is None or epoch%5 == 0:
    vld_invld_loss_ratio = valid_loss/invalid_loss

  # Final loss calculation
  loss             += valid_loss + vld_invld_loss_ratio * invalid_loss
  return loss


def penalize(outputs, labels, criterion, labels_filtering, llambda, vllambda, train_type, epoch, loss_ratio=None):
  """
  Calculate the penalty of the model based on the outputs and labels.

  Args:
  - outputs (torch.Tensor): Model predictions.
  - labels (torch.Tensor): True labels or targets.
  - criterion (callable): Loss function (e.g., torch.nn.CrossEntropyLoss).
  - labels_filtering (bool): Flag to apply special handling for labels.
  - lambda_loss_weight (float): Weight for the overall loss.
  - var_lambda_loss_weight (float): Weight for the valid/invalid loss.
  - train_type (str): Type of training ('multilabel' or 'multiclass').
 
  Returns:
  - torch.Tensor: Computed loss.
  """
  
  if train_type == 'multilabel':
    if labels_filtering:
      loss = labels_filtering_loss(outputs, labels, criterion, llambda, vllambda, epoch, loss_ratio)
    else:
      loss = criterion(outputs, labels)
  elif train_type == 'multiclass':
    if labels_filtering:
      raise ValueError(f"Doesn't make sense to labels_filtering for multiclass classification")
    else:
      loss = criterion(outputs, labels)
    return loss, None
  else:
    if labels_filtering:
      loss_sigmoid = labels_filtering_loss(outputs, labels, criterion, llambda, vllambda, epoch, loss_ratio)
    else:
      # Binary CrossEntropy Loss for multilabel - sigmoid-based
      loss_sigmoid = criterion[0](outputs, labels)

    # Crossentropy Loss
    loss_softmax = criterion[1](outputs, labels)
    if loss_ratio is None or epoch%5 == 0:
      loss_ratio = (loss_softmax / loss_sigmoid).item()
      print(loss_ratio)
    loss = loss_softmax + loss_ratio * loss_sigmoid
    return loss, loss_ratio


def compute_metrics(val_outputs, val_labels, train_type):
  if train_type == 'multilabel':
    val_outputs = (val_outputs > .5).int()
    accuracy          = accuracy_score(val_labels, val_outputs)
    non_norm_accuracy = accuracy_score(val_labels, val_outputs, normalize=False)
    precision         = precision_score(val_labels, val_outputs)
    f1_score_macro    = f1_score(val_labels, val_outputs, average='macro')
    return accuracy, non_norm_accuracy, precision, None, None, None, f1_score_macro
  elif train_type == 'multiclass':
    k                                      = val_labels[val_labels == 1].size(0)
    _, arg_topk_val_outputs                = val_outputs.topk(k)
    topk_predictions                       = torch.zeros_like(val_outputs)
    topk_predictions[arg_topk_val_outputs] = 1
    accuracy                               = accuracy_score(val_labels, topk_predictions)
    non_norm_accuracy                      = accuracy_score(val_labels, topk_predictions, normalize=False)
    precision                              = precision_score(val_labels, topk_predictions)
    f1_score_macro                         = f1_score(val_labels, topk_predictions, average='macro')
    return accuracy, non_norm_accuracy, precision, None, None, None, f1_score_macro
  else:
    k                                      = val_labels[val_labels == 1].size(0)
    _, arg_topk_val_outputs                = val_outputs.topk(k)
    topk_predictions                       = torch.zeros_like(val_outputs)
    topk_predictions[arg_topk_val_outputs] = 1
    val_outputs                            = (val_outputs > .5).int()
    accuracy                               = accuracy_score(val_labels, val_outputs)
    non_norm_accuracy                      = accuracy_score(val_labels, val_outputs, normalize=False)
    precision                              = precision_score(val_labels, val_outputs)
    accuracy_topk                          = accuracy_score(val_labels, topk_predictions)
    non_norm_accuracy_topk                 = accuracy_score(val_labels, topk_predictions, normalize=False)
    precision_topk                         = precision_score(val_labels, topk_predictions)
    f1_score_macro_topk                    = f1_score(val_labels, topk_predictions, average='macro')
    return accuracy, non_norm_accuracy, precision, accuracy_topk, non_norm_accuracy_topk, precision_topk, f1_score_macro_topk 


def get_free_gpu():
  try:
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    
    # Run the nvidia-smi command
    command = "nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader"
    memory_free_info = _output_to_list(subprocess.check_output(command.split())) 
    memory_free_values = [int(x) for i, x in enumerate(memory_free_info)]
  
  except Exception as e:
    print("Could not run nvidia-smi:", e)
    return None
  all_available_devices = ",".join([str(c) for c in range(len(memory_free_values))])
  visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', all_available_devices)

  # filter out devices
  for i in all_available_devices.split(','):
    if i not in visible_devices: 
      memory_free_values[int(i)] = -1

  # Remove undesired devices
  while -1 in memory_free_values: memory_free_values.remove(-1)

  # Get the GPU with the maximum free memory
  free_gpu = memory_free_values.index(max(memory_free_values))
  return free_gpu
  
  

def parse_arguments(parser):  
  # Add arguments
  # Training Arguments
  parser.add_argument('--batch_size', '--batch-size', type=int, default=64, help='Batch size for training. Number of training samples needs to be processed for the optimizer\'s step. A gradient accumulation technique is applied. The number of gradient accumulation steps should be an integer. gradient_accumulation_steps == batch_size / (#gpus * micro_batch_size) ')
  parser.add_argument('--micro_batch_size', '--micro-batch-size', type=int, default=2, help='Micro Batch size. Number of samples that should fit into a single GPU so that the gradients can be calculated')
  parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
  parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
  parser.add_argument('--num_workers', '--num-workers', type=int, default=0, help='how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process')
  parser.add_argument('--test_size', '--test-size', type=float, default=0.3, help='Test size. The test size is the ratio from the original dataset size')
  parser.add_argument('--dropout_p', '--dropout-p', type=float, default=0.1, help='Dropout probability')
  parser.add_argument('--model_name', '--model-name', '--model_checkpoint', '--model-checkpoint', '--load_checkpoint', '--load-checkpoint', type=str, default='llama-2', help='Based Model checkpoint for training.')
  parser.add_argument('--revision', type=str, default='main', help='Set the commit hash of huggingface\'s repository to download and train. Huggingface repo is set by `--model-name`.')
  parser.add_argument('--save_model', '--save-model', type=str, default='llama-2-combin-atpg', help='Save the trained model')
  parser.add_argument('--is_causal', '--is-causal', action='store_true', help='Specify if the you want to have a causal model specified')  
  
  # Quantization
  parser.add_argument('--use_4bit', '--use-4bit', action='store_true', help='Apply 4-bit quantization on the model. Be advised that the model will be loaded on float32 but the training will take place on a device i.e. gpu, will be then quantized to the specified bit precision')
  parser.add_argument('--use_8bit', '--use-8bit',action='store_true', help='Apply 8-bit quantization on the model. Be advised that the model will be loaded on float32 but the training will take place on a device i.e. gpu, will be then quantized to the specified bit precision')
  parser.add_argument('--bf16', action='store_true', help='Store the model as dtype torch.bfloat16')

  # LoRA hyperparameters
  parser.add_argument('--lora', action='store_true', help='Apply Parametric-Efficient Fine-Tuning (PEFT) with the use of LoRA technique. Specify the appropriate lora hyperparameters.')  
  parser.add_argument('--lora_alpha', '--lora-alpha', type=int, default=16, help="It's used  for scaling the weights of lora layers. Low-Rank Adaptation (LoRA) alpha parameter.")
  parser.add_argument('--lora_r', '--lora-r', type=int, default=64, help="It's the size of A and B matrices. The intermediate size of the 2 matrices. i.e. (B_dim, r) x (r, A_dim) = (B_dim, A_dim). Low-Rank Adaptation (LoRA) rank parameter.")
  parser.add_argument('--lora_dropout', '--lora-dropout', type=float, default=0.1, help='Dropout probability')

  # Tokenizer
  parser.add_argument('--tokenizer', type=str, default='custom', help='Which Tokenizer will be used.')
  parser.add_argument('--new_tokens', '--new-tokens', action='store_true', help='Parse given dataset and add new tokens')
  parser.add_argument('--model_max_length', '--model-max-length', '--mml', type=int, default=4096, help="Specify the maximum model's length that the tokenizer can tokenize.")
  parser.add_argument('--atpg_collate', '--atpg-collate', action='store_true', help='Use Custom ATPG Collate so that have more efficient training with less tokens. Tokenization will take place during training process.')
  
  # Data
  parser.add_argument('--data_file', '--data-file', type=str, default=None, required=True, help='Data file for training')
  parser.add_argument('--vocab_file', '--vocab-file', type=str, default=None, help='Use vocabulary for the custom tokenizer')
  
  # GPU parallelization
  parser.add_argument('--parallel', action='store_true', help='Parallel training using Distributed Data Parallelization')  
  parser.add_argument('--deepspeed_kernel', '--deepspeed-kernel', action='store_true', help='Use DeepSpeed transformer kernel to accelerate. Should be used together with --parallel')
  parser.add_argument('--fsdp', action='store_true', help='Fully Sharded Data Parallelization (FSDP). Should be enabled together with --parallel')
  parser.add_argument('--local_rank', '--local-rank', type=int, default=-1, help='Local rank passed from distributed launcher')

  # Accelerator arguments used for ZeRO acceleration
  parser.add_argument('--zero_stage', '--zero-stage', type=int, default=2, help="Enable ZeRO memory optimizations, compatible with FP16/BF16/FP32 and the Adam Optimizer. zero_stage: Chooses different of ZeRO Optimizer. Stage 0, 1, 2, and 3 refer to disabled, optimizer, state partioning, and optimizer+gradient state partitioning, and optimizer+gradient+parameter partitioning, respectively.")
  parser.add_argument('--seed', type=int, default=27, help="Seed for initialization of the accelerator used for ZeRO acceleration")

  # GRPO-RL hyperparameters
  parser.add_argument('--grpo_beta', '--beta', type=float, default=0.04, help="KL coefficient for Group Relative Policy Optimization (GRPO)")
  parser.add_argument('--num_generations', '--num-generations', type=int, default=2, help="Number of generations per prompt to sample.")
  parser.add_argument('--initial_ref_update_freq', '--initial-ref-update-freq', type=int, default=1, help='Initial frequency for reference model updates')
  parser.add_argument('--final_ref_update_freq', '--final-ref-update-freq', type=int, default=1, help='Final frequency for reference model updates')
  parser.add_argument('--grpo_tau', '--grpo-tau', type=float, default=0.9, help='EMA coefficient for reference model updates')

  # Weights and Biases
  parser.add_argument('--wandb', action='store_true', help='Use Weights and Biases for logging')
  parser.add_argument('--tags', type=str, default='baseline', help='Tags for Weights and Biases. For multiple tags, use comma to separate them.')

  # Add adapter loading arguments
  parser.add_argument('--adapter_name', '--adapter-name', type=str, default='ref_adapter', help='Name of the adapter to load')
  parser.add_argument('--adapter_repo', '--adapter-repo', type=str, default=None, help='Repository containing the adapter to load')

  # Add train_lora flag
  parser.add_argument('--train_lora', '--train-lora', action='store_true', help='Whether to train LoRA layers during training')
  parser.add_argument('--train_rl', '--train-rl', action='store_true', help='Whether to train RL layers during training. Group Relative Policy Optimization (GRPO) is used.')

  # Add filename parameter
  parser.add_argument('--filename', type=str, default=None, help='Custom filename for output files (logs, generated text, etc.)')

  # Parse arguments
  args = parser.parse_args()
  if args.deepspeed_kernel == True:
    # parser = deepspeed.add_config_arguments(parser)
    args   = parser.parse_args()
  return args


def get_gpu_memory_info():
  if torch.cuda.is_available():
    gpu_info = []
    for i in range(torch.cuda.device_count()):
      torch.cuda.set_device(i)
      gpu_info.append({
        'device': torch.cuda.get_device_name(i),
        'memory_allocated': torch.cuda.memory_allocated(i) / 1e9,  # in GB
        'memory_cached': torch.cuda.memory_reserved(i) / 1e9,  # in GB
      })
    return gpu_info
  else:
    return "No CUDA-compatible GPU detected."


def get_patterns_criterion(hps):
  """
  Goal: Ignore any index rather the ids of the tokens '0' and '1'.
  """

  weight = torch.zeros(len(hps.tokenizer), dtype=torch.bfloat16).to(hps.device)
  # weight only the vocabulary's digits 
  for i in range(10):
    weight[hps.tokenizer.convert_tokens_to_ids(str(i))] = 1
  criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=-100) 
  return criterion 


def get_code_files_to_version():
  """
  Find important code files in the transformers_atpg directory structure for versioning.
  
  Returns:
      tuple: (transformers_atpg_path, code_files_to_version) where:
          - transformers_atpg_path is the path to the transformers_atpg directory or None if not found
          - code_files_to_version is a list of file paths to version
  """
  # Version control for code files
  code_files_to_version = []
  
  # Find the transformers_atpg directory
  transformers_atpg_path = None
  current_dir = os.path.abspath(os.getcwd())
  
  # Search for transformers_atpg directory
  while current_dir != os.path.dirname(current_dir):  # Stop at root directory
    if os.path.basename(current_dir) == "transformers_atpg":
      transformers_atpg_path = current_dir
      break
    current_dir = os.path.dirname(current_dir)
  
  # If transformers_atpg directory found, look for specific files
  if transformers_atpg_path:
    if os.path.exists(os.path.join(transformers_atpg_path, "libatpgllm")):
      atpgllm_dir = os.path.join(transformers_atpg_path, "libatpgllm")
    else:
      atpgllm_dir = os.path.join(transformers_atpg_path, "atpgllm")

    if os.path.exists(atpgllm_dir) and os.path.isdir(atpgllm_dir):
      inner_atpgllm_dir = os.path.join(atpgllm_dir, "atpgllm")
      llm_dir = os.path.join(inner_atpgllm_dir, "llm")
      tests_dir = os.path.join(atpgllm_dir, "tests")
      
      # Check for utils.py in inner_atpgllm_dir
      if os.path.exists(inner_atpgllm_dir) and os.path.isdir(inner_atpgllm_dir):
        utils_path = os.path.join(inner_atpgllm_dir, "utils.py")
        if os.path.exists(utils_path) and os.path.isfile(utils_path):
          code_files_to_version.append(utils_path)
      
      # Check for files in llm directory
      if os.path.exists(llm_dir) and os.path.isdir(llm_dir):
        for file_name in ["tokenizer.py", "collate.py", "fault_coverage_calc.py", "reward_funcs.py"]:
          file_path = os.path.join(llm_dir, file_name)
          if os.path.exists(file_path) and os.path.isfile(file_path):
            code_files_to_version.append(file_path)

      # Check for ft_train_with_rl.py in tests_dir
      if os.path.exists(tests_dir) and os.path.isdir(tests_dir):
        file_path = os.path.join(tests_dir, "ft_train_with_rl.py")
        if os.path.exists(file_path) and os.path.isfile(file_path):
          code_files_to_version.append(file_path)
  
  return transformers_atpg_path, code_files_to_version


def distributed_env_init(hps):
  """
  Initialize the distributed training environment.
  
  Args:
      hps: Hyperparameters object containing training configuration
  
  This function:
  1. Sets up distributed process ranks and world size from environment variables
  2. Initializes the PyTorch distributed process group
  3. Configures the appropriate CUDA device for each process
  4. Sets random seed for reproducibility
  5. Adjusts gradient accumulation steps based on world size
  """
  # Extract distributed training information from environment variables
  hps.rank             = int(os.environ['RANK'])             # Global rank of current process
  hps.local_rank       = int(os.environ['LOCAL_RANK'])       # Local rank within the current node
  hps.local_world_size = int(os.environ['LOCAL_WORLD_SIZE']) # Number of processes in the current node
  hps.world_size       = int(os.environ['WORLD_SIZE'])       # Total number of processes across all nodes
  
  # Initialize the process group with NCCL backend (optimized for GPU communication)
  dist.init_process_group(backend='nccl', rank=hps.local_rank, world_size=hps.world_size)
  
  # Set the device for the current process
  hps.device = torch.device(f'cuda:{hps.local_rank}' if torch.cuda.is_available() else 'cpu')

  # Initialize the accelerator
  hps.accelerator = Accelerator()

  # Log distributed training configuration
  hps.info += f"Backend Configuration: {dist.get_backend_config()}, local_rank: {hps.local_rank}, local_world_size: {hps.local_world_size}, free-est gpu: {hps.free_gpu_id}, world_size: {hps.world_size}, device: {hps.device}\n"
  
  # Set random seed for reproducibility
  torch.manual_seed(hps.seed)
  
  # Set the current CUDA device to match local rank
  torch.cuda.set_device(hps.local_rank)
  
  # Scale down gradient accumulation steps by world size to maintain effective batch size
  hps.gradient_accumulation_steps = int(hps.gradient_accumulation_steps//hps.world_size)
  
  # Only print from the main process (rank 0) to avoid duplicate messages
  if hps.local_rank == 0:
    print(f"Accumulated gradient steps: {hps.gradient_accumulation_steps}")

def accelerator_init(hps):
  import datasets
  import transformers
  from accelerate.utils import set_seed
  from accelerate.utils.dataclasses import DeepSpeedPlugin

  # TODO: test mixed_precision='bf16'
  accelerator = Accelerator(deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=hps.deepspeed_config))
  hps.disable_tqdm = not accelerator.is_local_main_process
  if accelerator.is_local_main_process:
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()
  else:
    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()
  set_seed(hps.seed)
  return accelerator

def initialize_training_environment(hps):
  hps.info = ''
  # Verify the training type has been set
  verify_training_type(hps)
  if hps.parallel==True:
    if hps.deepspeed_kernel:
      # Initialize the accelerator
      hps.accelerator = accelerator_init(hps)
    else:
      # Initialize the distributed environment to apply:
      # 1. Distributed Data Parallel (DDP)
      # 2. Fully Sharded Data Parallelization (FSDP)
      distributed_env_init(hps)
  else:
    # Initialize single GPU/CPU
    hps.device = torch.device(f'cuda:{hps.free_gpu_id}' if torch.cuda.is_available() else 'cpu')
    hps.local_rank = hps.free_gpu_id
    hps.world_size = 1
    hps.info += f'Free detected GPU: {hps.device}\n' if torch.cuda.is_available() else 'No GPU is detected'
    hps.logger.info(f"Accumulated gradient steps: {hps.gradient_accumulation_steps}")

  # Initialize the W&B logger
  if hps.wandb:
    # Create a consistent group name across all distributed processes
    # For distributed training, we want the same group name for all ranks
    if hps.parallel:
      # Initialize group_name for all processes
      group_name = ""
      
      # Only create timestamp on rank 0
      if is_main_process():
        timestamp = datetime.now().strftime("%Y_%m_%d_%Hh_%Mm_%Ss")
        group_name = f"{timestamp}"
        
      # Synchronize the group name across all processes if using distributed
      if hps.world_size > 1:
        # Create a tensor buffer for broadcasting
        if is_main_process():
          group_name_tensor = torch.tensor([ord(c) for c in group_name] + [0] * (100 - len(group_name)), 
                                          dtype=torch.long, device=hps.device)
        else:
          # Non-main processes need to initialize the tensor to receive the broadcast
          group_name_tensor = torch.zeros(100, dtype=torch.long, device=hps.device)
          
        # Broadcast from rank 0 to all processes
        dist.broadcast(group_name_tensor, src=0)
        
        # Convert back to string on all ranks
        if not is_main_process():
          group_name = ''.join([chr(c) for c in group_name_tensor.cpu().tolist() if c > 0])
    else:
      # For non-distributed training, add timestamp
      timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
      group_name = f"{timestamp}"
        
    # Get code files for versioning
    transformers_atpg_path, code_files_to_version = get_code_files_to_version()
    
    # Initialize wandb with versioned code files
    hps.run = wandb.init(
        group=group_name, 
        project=f"RL Fine-Tuning", 
        entity="chrivasileiou", 
        config=hps, 
        save_code=True, 
        tags=hps.tags.split(","),
    )
    
    # Log specific files if found
    if code_files_to_version and hps.run:
      artifact = wandb.Artifact(name=f"code-{group_name}", type="code")
      for file_path in code_files_to_version:
        artifact.add_file(file_path)
      hps.run.log_artifact(artifact)


def get_trainable_parameters(model):
  """
  Get trainable parameters based on what we want to train.
  
  Args:
      model: The model containing the parameters
  
  Returns:
      List of parameter dictionaries with name and params
  """
  trainable_params = []
  base_model = model.module if hasattr(model, "module") else model
  
  # Group parameters by type
  for name, param in base_model.named_parameters():
    # Skip frozen parameters
    if not param.requires_grad:
      continue
        
    # Handle embeddings
    if "embed" in name:
      trainable_params.append({
          "name": f"embeddings: {name}",
          "params": param,
          "lr": 1e-5  # Lower learning rate for embeddings
      })
    # Handle LoRA parameters
    elif any(adapter in name for adapter in ["lora", "adapter"]):
      trainable_params.append({
          "name": f"lora: {name}",
          "params": param,
          "lr": 1e-4  # Higher learning rate for LoRA
      })
    # Handle output head
    elif "head" in name or "output" in name:
      trainable_params.append({
          "name": f"head: {name}",
          "params": param,
          "lr": 5e-5  # Medium learning rate for head
      })
  return trainable_params

def prepare_objects_for_training(model, dataset, hps):
  from torch.optim import AdamW
  from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
  from atpgllm.llm.collate import MyCollate
  from atpgllm.llm.tokenizer import tokenize_fn
  from atpgllm.llm.metrics import ATPGAccuracy
  from accelerate.utils import DummyOptim, DummyScheduler
  from accelerate import DistributedType
  from transformers import get_constant_schedule_with_warmup, get_inverse_sqrt_schedule, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
  from torchinfo import summary
  from functools import partial
  import multiprocessing as mp
  import math
  import sys

  # Set the distributed systems if necessary
  if hps.parallel==True:
    if hps.deepspeed_kernel:
      # Deepspeed Kernel

      # Tokenize the dataset
      with hps.accelerator.main_process_first():
        tokenized_dataset = dataset
        # tokenized_dataset = dataset.map(tokenize_fn, batched=True, num_proc=16, load_from_cache_file=True, remove_columns=['text'], fn_kwargs={'tokenizer': hps.tokenizer, 'is_causal': hps.is_causal})
      hps.collate_fn = MyCollate(tokenizer=hps.tokenizer, is_causal=hps.is_causal, lora=hps.lora, instruction_training=hps.train_lora or hps.new_tokens)
      
      if hps.accelerator.is_local_main_process:
        hps.info += f"{tokenized_dataset}\n" + \
                    f"{model}\n"
        hps.logger.info(f"\n{hps.info}")

      # Creates Dummy Optimizer if `optimizer` was spcified in the config file else creates Adam Optimizer
      optimizer_cls = (
          torch.optim.AdamW
          if hps.accelerator.state.deepspeed_plugin is None or "optimizer" not in hps.accelerator.state.deepspeed_plugin.deepspeed_config
          else DummyOptim
      )
      hps.optimizer = optimizer_cls(model.parameters(), lr=hps.lr) if sys.argv[0] != 'sft.py' else None

      # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
      if hps.accelerator.distributed_type == DistributedType.TPU:
          model.tie_weights()
    
      # Scheduler and math around the number of training steps.
      # Get gradient accumulation steps from deepspeed config if available
      if hps.accelerator.state.deepspeed_plugin is not None:
          hps.gradient_accumulation_steps = int(hps.accelerator.state.deepspeed_plugin.deepspeed_config["gradient_accumulation_steps"])
    elif not hps.deepspeed_kernel: 
      # Initialize the distributed environment to apply in NCCL:
      os.environ["NCCL_TIMEOUT"] = "3000"
      # os.environ["NCCL_DEBUG"] = "INFO"
      # os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"

      # 1. Distributed Data Parallel (DDP)
      # 2. Fully Sharded Data Parallelization (FSDP)

      # Tokenize the dataset
      # tokenized_dataset = dataset.map(tokenize_fn, batched=True, num_proc=16, load_from_cache_file=True, remove_columns=['text'], fn_kwargs={'tokenizer': hps.tokenizer, 'is_causal': hps.is_causal})
      tokenized_dataset = dataset
      hps.collate_fn = MyCollate(tokenizer=hps.tokenizer, is_causal=hps.is_causal, lora=hps.lora, instruction_training=hps.train_lora or hps.new_tokens)
      
      if hps.fsdp:
        total_model_parameters = sum(p.numel() for p in model.parameters())
        num_params_per_gpu = (total_model_parameters // torch.cuda.device_count())
        
        # Create a custom wrap policy for FSDP
        hps.my_auto_wrap_policy = partial(
          transformer_auto_wrap_policy, recurse=True, transformer_layer_cls={LlamaDecoderLayer}
        )
        hps.mixed_precision = MixedPrecision(
          param_dtype=torch.bfloat16, 
          reduce_dtype=torch.bfloat16, 
          buffer_dtype=torch.bfloat16
        )
        # Standard FSDP wrapping for non-LoRA models
        # Check if all parameters have requires_grad=True
        all_params_require_grad = all(p.requires_grad for p in model.parameters())
        model = FSDP(model,
                      # device_id=hps.device,
                      auto_wrap_policy=hps.my_auto_wrap_policy,
                      use_orig_params=not all_params_require_grad,  # Set to True if some params don't require grad
                      sharding_strategy=ShardingStrategy.FULL_SHARD,
                      # Avoid wrapping LoRA modules to prevent issues
                      ignored_modules=[m for name, m in model.named_modules() if 'lora' in name.lower()],
                      mixed_precision=hps.mixed_precision
                    ) if sys.argv[0] != 'sft.py' else model
        torch.cuda.set_device(hps.local_rank)
      else:
        # Move model to the correct device before wrapping with DDP
        model = model.to(f"cuda:{hps.local_rank}")
        train_layers(model, train_embeddings=True, train_head=True, train_lora=True, train_base_model=False)
        model = DDP(model, device_ids=[hps.local_rank], output_device=hps.local_rank, find_unused_parameters=False) if sys.argv[0] != 'sft.py' else model
        # model.module = torch.compile(model.module, backend='inductor', fullgraph=True)

      if is_main_process():
        hps.info += f"{tokenized_dataset}\n" + \
                    f"{model}\n"
        hps.logger.info(f"\n{hps.info}")
        
  else:
    # Move the model to the GPU only if necessary
    if hps.use_4bit == False and hps.use_8bit == False:
      model  = model.to(hps.device) if sys.argv[0] != 'sft.py' else model

    # Tokenize the dataset
    # tokenized_dataset = dataset.map(tokenize_fn, batched=True, num_proc=32, load_from_cache_file=True, remove_columns=['text'], fn_kwargs={'tokenizer': hps.tokenizer, 'is_causal': hps.is_causal})
    tokenized_dataset = dataset
    hps.collate_fn = MyCollate(tokenizer=hps.tokenizer, is_causal=hps.is_causal, lora=hps.lora, instruction_training=hps.train_lora or hps.new_tokens)
    hps.info += f"{tokenized_dataset}\n" + \
                f"{model}\n"
    hps.logger.info(f"\n{hps.info}")
    
  # Set the flags for distributed systems. Data samplers and Data Loaders
  use_sampler = hps.parallel==True and hps.deepspeed_kernel==False
  # Shuffle is handled by the Data Sampler in distributed systems
  hps.shuffle = not use_sampler
  training_dataset = sorted(tokenized_dataset['train'], key=lambda x: x['gates_count'])

  if use_sampler:
    training_sampler   = DistributedSampler(training_dataset, rank=hps.local_rank, num_replicas=hps.local_world_size, shuffle=True)
    validation_sampler = DistributedSampler(tokenized_dataset['validation'], rank=hps.local_rank, num_replicas=hps.local_world_size, shuffle=hps.shuffle)
    testing_sampler    = DistributedSampler(tokenized_dataset['test'], rank=hps.local_rank, num_replicas=hps.local_world_size, shuffle=hps.shuffle)
  else:
    training_sampler, validation_sampler, testing_sampler = None, None, None

  # Create the Data Loaders
  training_loader   = DataLoader(dataset=training_dataset, batch_size=hps.micro_batch_size, collate_fn=hps.collate_fn, sampler=training_sampler, num_workers=hps.num_workers, pin_memory=True, drop_last=True)
  validation_loader = DataLoader(dataset=tokenized_dataset['validation'], batch_size=hps.micro_batch_size, collate_fn=hps.collate_fn, sampler=validation_sampler, num_workers=hps.num_workers, pin_memory=True, drop_last=True)
  testing_loader    = DataLoader(dataset=tokenized_dataset['test'], batch_size=hps.micro_batch_size, collate_fn=hps.collate_fn, sampler=testing_sampler, num_workers=hps.num_workers, pin_memory=True, drop_last=True)

  if not hps.deepspeed_kernel:
    total_training_steps = (hps.epochs * len(training_loader)) // hps.gradient_accumulation_steps
    hps.optimizer = ZeroRedundancyOptimizer(model.parameters(), optimizer_class=AdamW, lr=hps.lr) if hps.parallel and not hps.fsdp else AdamW(model.parameters(), lr=hps.lr) if sys.argv[0] != 'sft.py' else None
    if hps.new_tokens:
      hps.scheduler = get_cosine_schedule_with_warmup(hps.optimizer, num_warmup_steps=10, num_training_steps=total_training_steps, num_cycles=3/20)
    else:
      hps.scheduler = get_cosine_schedule_with_warmup(hps.optimizer, num_warmup_steps=10, num_training_steps=total_training_steps, num_cycles=3/20) if sys.argv[0] != 'sft.py' else None # when cosine scheduler is used, we need to set num_training_steps to 10 
      
  # Prepare whatever requires for ZeRO acceleration training (deepspeed_kernel is activated)
  if hps.deepspeed_kernel and hps.parallel:
    num_update_steps_per_epoch = math.ceil(len(training_loader) / hps.gradient_accumulation_steps)
    hps.max_train_steps = hps.epochs * num_update_steps_per_epoch
    if (hps.accelerator.state.deepspeed_plugin is None
        or "scheduler" not in hps.accelerator.state.deepspeed_plugin.deepspeed_config):

      # Select: ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]
      if hps.new_tokens:
        hps.scheduler = get_cosine_schedule_with_warmup(hps.optimizer, num_warmup_steps=10, num_training_steps=total_training_steps, num_cycles=3/20) if sys.argv[0] != 'sft.py' else None
      else:
        hps.scheduler = get_cosine_schedule_with_warmup(hps.optimizer, num_warmup_steps=10, num_training_steps=total_training_steps) # when cosine scheduler is used, we need to set num_training_steps to 10
    else:
      hps.scheduler = DummyScheduler(
        hps.optimizer, total_num_steps=hps.max_train_steps, warmup_num_steps=10 # hps.num_warmup_steps
      ) if sys.argv[0] != 'sft.py' else None
    model, hps.optimizer, training_loader, validation_loader, testing_loader, hps.scheduler = hps.accelerator.prepare(model, hps.optimizer, training_loader, validation_loader, testing_loader, hps.scheduler) if sys.argv[0] != 'sft.py' else model, None, training_loader, validation_loader, testing_loader, None
    hps.device = model.device
    num_update_steps_per_epoch = math.ceil(len(training_loader) / hps.gradient_accumulation_steps)
    hps.max_train_steps = hps.epochs * num_update_steps_per_epoch
    if hps.accelerator.is_local_main_process:
      hps.logger.info(f"{model}")
  
  # Set the loss functions
  hps.criterion = nn.CrossEntropyLoss(ignore_index=-100).to(hps.device)
  hps.patterns_criterion = get_patterns_criterion(hps)
  # hps.accuracy = ATPGAccuracy(model, hps.tokenizer)

  return model, training_loader, validation_loader, testing_loader


def dataset_formation(raw_dataset, hps):
  """
  raw
  DatasetDict({
      train: Dataset({
          features: ['module_name', 'prompts', 'answers', 'patterns', 'faults', 'atpg', 'netlist'],
          num_rows: 150000
      })
  })
  """
  test_size = hps.test_size
  goal = {"test coverage": 7, "fault coverage": 8} # the number shows the line that needs to be cropped.
  system_prompt = "<<SYS>> You are a Test Pattern Generation tool is applied to integrated circuits (ICs) for testing. <</SYS>>\n\n"
  prompts = ["[INST] " + system_prompt + "Your task is to write test vectors that can achieve " + coverage.split('\n')[goal['test coverage']].split()[-1] + " test coverage for the design:\n\n```\n" + netlist + "```\n\nPlease wrap the test vectors in ```. Please think carefully. [/INST]" for netlist, coverage in zip(raw_dataset['train']['netlist_only_gates'], raw_dataset['train']['atpg'])]
  answers = ["The test vectors for the provided design are:\n\n```\n" + '\n'.join([pat for pat in patterns.replace(' ', '').split('\n')]) + "\n```" for patterns in raw_dataset['train']['patterns']]
  if hps.is_causal:
    # o_dataset = [{'prompt':chat, 'netlist':netlist} for chat, netlist in tqdm(zip(formatted_chats, raw_dataset['train']['netlist'])) if len(tokenizer.tokenize(chat)) < tokenizer.model_max_length ]
    o_dataset = [prompt + "\n" + answer for prompt, answer in zip(prompts, answers)]
  else:
    o_dataset = [{'prompts': prompt, 'answers': answer} for prompt, answer in zip(prompts, answers)]
  
  # Compute splits ratio
  val_size   = (1 - test_size) * 0.15 
  train_size = (1 - test_size) * 0.85 
  assert train_size + val_size + test_size == 1

  # Compute splits
  train_size = int(train_size * len(o_dataset))
  val_size   = int(val_size * len(o_dataset))
  test_size  = len(o_dataset) - train_size - val_size
  assert train_size + val_size + test_size == len(o_dataset)

  # Drop samples that make the datasets uneven
  train_size = (train_size//hps.batch_size) * hps.batch_size if not hps.parallel else (((train_size//hps.local_world_size) * hps.local_world_size) // hps.batch_size) * hps.batch_size
  val_size   = (val_size//hps.batch_size) * hps.batch_size   if not hps.parallel else (((val_size//hps.local_world_size) * hps.local_world_size) // hps.batch_size) * hps.batch_size
  test_size  = (test_size//hps.batch_size) * hps.batch_size  if not hps.parallel else (((test_size//hps.local_world_size) * hps.local_world_size) // hps.batch_size) * hps.batch_size

  # Create Datasets
  # Take correct indices and split to text and netlist. (netlist is used for RL training)
  train_dataset = Dataset.from_dict( {'text': o_dataset[ : train_size ], 'netlist': raw_dataset['train']['netlist'][ :train_size]} )
  val_dataset   = Dataset.from_dict( {'text': o_dataset[ train_size : train_size+val_size], 'netlist': raw_dataset['train']['netlist'][ train_size : train_size+val_size ]} )
  test_dataset  = Dataset.from_dict( {'text': o_dataset[ train_size+val_size : train_size+val_size+test_size ], 'netlist': raw_dataset['train']['netlist'][ train_size+val_size : train_size+val_size+test_size ]} )
  
  # Final Dataset
  dataset = DatasetDict({'train': train_dataset, 'validation': val_dataset, 'test': test_dataset})

  return dataset 

def dataset_formation_using_chat_template(raw_dataset, hps):
  from random import randint
  from . import _system_prompts, _details, _extras, _training_prompts_coverage, chat_template
  remove_whitespaces = re.compile(r' {2,}')
  tokenizer = hps.tokenizer
  is_causal = hps.is_causal
  test_size = hps.test_size
  coverage_types = ["test", "fault"]
  goal = {"test": 7, "fault": 8} # the number shows the line that needs to be cropped.
  messages = []
  max_netlist_length = 0
  max_prompt_length = 0
  for i, (netlist, coverage, patterns, faults) in enumerate(zip(raw_dataset['train']['netlist_only_gates'], raw_dataset['train']['atpg'], raw_dataset['train']['patterns'], raw_dataset['train']['faults'])):
    _faults = sorted([tuple(_fault.split()) for _fault in faults.split('\n') if len(_fault) > 1], key=lambda k: k[2])
    _faults = ', '.join([' '.join(_fault[2:]) for _fault in _faults])
    
    coverage_type = coverage_types[randint(0, len(coverage_types)-1)]
    system_prompt = _system_prompts[randint(0, len(_system_prompts)-1)]
    detail = _details[randint(0, len(_details)-1)]
    extra = _extras[randint(0, len(_extras)-1)]
    user_prompt = _training_prompts_coverage[randint(0, len(_training_prompts_coverage)-1)] + coverage.split('\n')[goal[coverage_type]].split()[-1] + " " + coverage_type + " coverage for the design:\n\n```\n" + netlist + "\n```" + detail + extra
    answer = "The test vectors for the provided design are:\n\n```\n" + '\n'.join([pat for pat in patterns.replace(' ', '').split('\n')]) + "\n```"
    message = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": user_prompt},
      {"role": "assistant", "content": answer},
    ]
    messages.append(message)

  formatted_chats = [remove_whitespaces.sub(' ', chat).replace('<s>', '').strip()+'</s>' for chat in tokenizer.apply_chat_template(messages, tokenize=False, chat_template=chat_template).split('</s>\n      ')]


  if is_causal:
    o_dataset = [{'prompt':chat, 'netlist':netlist} for chat, netlist in tqdm(zip(formatted_chats, raw_dataset['train']['netlist'])) if len(tokenizer.tokenize(chat)) < tokenizer.model_max_length ]
  else:
    o_dataset = []
    for chat in formatted_chats:
      split = chat.split('[/INST]')
      prompt, answer = split[0], split[1]
      if len(tokenizer.tokenize(prompt)) < tokenizer.model_max_length:
        o_dataset.append({'prompt': {'prompts': prompt+'[/INST]', 'answers': answer}, 'netlist': netlist})
  
  # Compute splits ratio
  val_size   = (1 - test_size) * 0.15 
  train_size = (1 - test_size) * 0.85 
  assert train_size + val_size + test_size == 1, "the ratios are not correct"
  # Compute splits
  train_size = int(train_size * len(o_dataset))
  val_size   = int(val_size * len(o_dataset))
  test_size  = len(o_dataset) - train_size - val_size
  assert train_size + val_size + test_size == len(o_dataset)
  # Drop samples that make the datasets uneven. Huge problem for multi-GPU training...
  if not hps.parallel:
    train_size = (train_size//hps.batch_size) * hps.batch_size if train_size > hps.batch_size else train_size
    val_size   = (val_size//hps.batch_size) * hps.batch_size   if val_size > hps.batch_size else val_size
    test_size  = (test_size//hps.batch_size) * hps.batch_size  if test_size > hps.batch_size else test_size
  else:
    train_size = (train_size//(hps.local_world_size * hps.batch_size)) * (hps.local_world_size * hps.batch_size) if train_size > hps.local_world_size * hps.batch_size else train_size
    val_size   = (val_size//(hps.local_world_size * hps.batch_size)) * (hps.local_world_size * hps.batch_size)   if val_size > hps.local_world_size * hps.batch_size else val_size
    test_size  = (test_size//(hps.local_world_size * hps.batch_size)) * (hps.local_world_size * hps.batch_size)  if test_size > hps.local_world_size * hps.batch_size else test_size

  # Create Datasets
  # Take correct indices
  train_dataset = Dataset.from_list(o_dataset[ : train_size ])
  val_dataset   = Dataset.from_list(o_dataset[ train_size : train_size+val_size ])
  test_dataset  = Dataset.from_list(o_dataset[ train_size+val_size : train_size+val_size+test_size ])
  # Split to text and netlist. (netlist is used for RL training)
  train_dataset = Dataset.from_dict( {'text': train_dataset['prompt'], 'netlist': train_dataset['netlist']} )
  val_dataset   = Dataset.from_dict( {'text': val_dataset['prompt'],   'netlist': val_dataset['netlist']} )
  test_dataset  = Dataset.from_dict( {'text': test_dataset['prompt'],  'netlist': test_dataset['netlist']} )
  # Final Dataset
  dataset       = DatasetDict({'train': train_dataset, 'validation': val_dataset, 'test': test_dataset})
  return dataset 


def generate_all_possible_binary_combination(starting_point=1, max_binary_length=5):
  def helper(current, max_binary_length):
    if len(current) == max_binary_length:
      yield current
    else:
      yield from helper(current + '0', max_binary_length)
      yield from helper(current + '1', max_binary_length)
  comb = []
  for i in range(starting_point, max_binary_length + 1):
    comb += list(helper('', i))
  return comb


def load_raw_dataset(data_file, test_size=.3):
  """
  Load the dataset given at the data_file argument
  """
  import os
  from glob import glob
  if os.path.exists(data_file) or all(os.path.exists(file) for file in glob(data_file)):
    from datasets import Features, Value
    features = Features({'text': Value(dtype='string'), 'netlist': Value(dtype='string')})
    # It's a local file
    raw_dataset = load_dataset('csv', data_files=data_file)
    if 'Unnamed: 0' in raw_dataset.column_names:
      raw_dataset = raw_dataset.remove_columns('Unnamed: 0')
  else:
    # It's a Hugging Face dataset repository
    raw_dataset = load_dataset(data_file)
  
  train_test_split = raw_dataset['train'].train_test_split(test_size=test_size)
  train_val_split = train_test_split['train'].train_test_split(test_size=.15)
  train_dataset = balance_dataset_by_gates_count(train_val_split['train'])
  dataset = DatasetDict({'train': train_dataset, 'validation': train_val_split['test'], 'test': train_test_split['test']})
  return dataset

def sizeof_tensor(tensor):
  return tensor.element_size() * tensor.nelement()

def calculate_memory(model=None, optimizer=None, ids=None, mask=None, targets=None):
  total_memory = 0  
  # Model parameters
  if model is not None:
    for param in model.parameters():
      total_memory += sizeof_tensor(param)
      if param.grad is not None and param.requires_grad == True:
        total_memory += sizeof_tensor(param.grad)
  # Optimizer states
  if optimizer is not None:
    for v in optimizer.param_groups[0]['params']:
      total_memory += sizeof_tensor(v)
    for state_k, state in optimizer.state.items():
      total_memory += sizeof_tensor(state_k)
      for k, v in state.items():
          if isinstance(v, torch.Tensor):
            total_memory += sizeof_tensor(v)  
  # ids + mask + targets
  total_memory += sizeof_tensor(ids) if ids is not None else 0
  total_memory += sizeof_tensor(mask) if mask is not None else 0
  total_memory += sizeof_tensor(targets) if targets is not None else 0
  
  return convert_bytes(total_memory)


def hyperparameters(args):
  from huggingface_hub import HfApi
  from atpgllm import deepspeed_config
  deepspeed_config = AttrDict(deepspeed_config)

  # args
  hps = AttrDict({})
  # training
  hps.epochs      = args.epochs 
  hps.lr          = args.lr
  hps.batch_size  = args.batch_size
  hps.micro_batch_size            = args.micro_batch_size # micro batch size, it's hardcoded to make sure it fits in GPU memory
  hps.gradient_accumulation_steps = int(args.batch_size // args.micro_batch_size)
  hps.test_size   = args.test_size
  hps.num_workers = args.num_workers
  # Low-Rank Adaptation (LoRA) - Parametric-Efficient Fine-Tuning (PEFT)
  hps.lora_alpha   = args.lora_alpha if isinstance(args.lora_alpha, int) else ast.literal_eval(args.lora_alpha)
  hps.lora_r       = args.lora_r
  hps.lora_dropout = args.dropout_p
  # model
  hps.dropout    = args.dropout_p
  hps.data_file  = args.data_file
  hps.vocab_file = args.vocab_file
  hps.model_name = args.model_name
  hps.save_model = args.save_model
  hps.bf16       = args.bf16
  hps.is_causal  = args.is_causal
  hps.lora       = args.lora 
  hps.revision   = args.revision
  # tokenizer
  hps.tokenizer        = args.tokenizer
  hps.model_max_length = args.model_max_length
  # collate
  hps.collate = args.atpg_collate
  # gpu usage
  hps.free_gpu_id      = get_free_gpu()
  hps.parallel         = args.parallel
  hps.deepspeed_kernel = args.deepspeed_kernel
  hps.fsdp             = args.fsdp
  hps.world_size       = int(len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))) if hps.deepspeed_kernel == True else 1
  hps.device_map       = None
  hps.use_4bit         = args.use_4bit
  hps.use_8bit         = args.use_8bit

  # ZeRO
  deepspeed_config.train_batch_size               = args.batch_size
  deepspeed_config.train_micro_batch_size_per_gpu = args.micro_batch_size
  deepspeed_config.zero_optimization['stage']     = args.zero_stage if hps.deepspeed_kernel == True else 0
  deepspeed_config.optimizer['params']['lr']      = hps.lr
  hps.deepspeed_config = deepspeed_config if hps.deepspeed_kernel == True else None
  hps.seed     = args.seed

  # Add adapter loading parameters
  hps.adapter_name = args.adapter_name
  hps.adapter_repo = args.adapter_repo

  # Add train_lora flag
  hps.new_tokens = args.new_tokens
  hps.train_lora = args.train_lora
  hps.train_rl   = args.train_rl

  # GRPO
  hps.grpo_beta = args.grpo_beta
  hps.num_generations = args.num_generations
  hps.initial_ref_update_freq = args.initial_ref_update_freq
  hps.final_ref_update_freq = args.final_ref_update_freq
  hps.grpo_tau = args.grpo_tau

  hps.disable_tqdm = False

  # Create an api to communicate with huggingface hub
  hps.api = HfApi()

  # Weights and Biases
  hps.wandb = args.wandb
  hps.tags = args.tags

  # Create a logger
  hps.logger = setup_logging(hps)
  
  # Use custom filename if provided, otherwise use default
  hps.filename = args.filename if args.filename is not None else "generated_text_embs_grpo.md"
  hps.file_path = os.path.join(hps.log_dir, hps.filename)

  return hps


def plot_training_plots(train_losses, val_losses, batch_train_losses, learning_rates, filename: str = "losses_plot.html", parallel=False):
  if parallel:
    if not is_main_process():
      return
  
  # Assuming all lists are of the same length
  epochs = list(range(1, len(train_losses) + 1))
  batches = list(range(1, len(batch_train_losses) + 1))

  # Create traces
  trace1 = go.Scatter(x=epochs, y=train_losses, mode='lines+markers', name='Training')
  trace2 = go.Scatter(x=epochs, y=val_losses, mode='lines+markers', name='Validation')
  trace3 = go.Scatter(x=batches, y=learning_rates, mode='lines+markers', name='Learning Rate')
  trace4 = go.Scatter(x=batches, y=batch_train_losses, mode='lines+markers', name='Training')
  
  # Create the figure
  fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("Train/Val Loss per Epoch", "Train Loss per Batch"))

  # Adding traces
  fig.add_trace(trace1, row=1, col=1)
  fig.add_trace(trace2, row=1, col=1)
  fig.add_trace(trace3, row=1, col=2)
  fig.add_trace(trace4, row=1, col=2)
  
  # Update layout
  fig.update_layout(title=dict(text='Training and Validation Losses', font=dict(size=25)),
                    yaxis_title='Loss',
                    legend_title='Loss Type')

  # Update x-axis titles for each subplot
  fig.update_xaxes(title_text='Epochs', row=1, col=1)
  fig.update_xaxes(title_text='Batches', row=1, col=2)
  
  # Save the figure as an interactive HTML
  fig.write_html(filename)


def cleanup(hps):
  # for key in hps.keys():
  #   hps[key] = None
  #del hps
  import gc
  gc.collect()
  gc.collect()

def verify_training_type(hps):
  hps.info += '=' * 36 + ' Model Loading ' + '=' * 36 + '\n'
  try:
    # Can't activate both 4-bit and 8-bit quantization types
    assert not (hps.use_4bit and hps.use_8bit) == True
  except AssertionError: 
    # There is a conflict when both types of quantization have been activated
    raise ValueError(f"You can't have activated both 4-bit and 8-bit quantization")
  finally:
    if hps.use_4bit:
      hps.info += f"4-Bit quantization is applied\n"
    elif hps.use_8bit:
      hps.info += f"8-Bit quantization is applied\n"
    else:
      hps.info += f"No quantization is applied\n"

  if hps.parallel:
    try:
      assert hps.deepspeed_kernel != hps.fsdp or (hps.deepspeed_kernel == False and hps.fsdp == False)
    except:
      raise ValueError(f"DeepSpeed kernel and FSDP cannot be used together (deepspeed_kernel={hps.deepspeed_kernel} and fsdp={hps.fsdp}). If you haven't activated any, it will be defaulted to DDP when parallel is activated (parallel={hps.parallel}).")
    finally:
      hps.info += "Parallelization is activated\n"
  elif not hps.parallel:
    hps.info += "No Parallelization\n"

  if hps.lora:
    hps.info += 'Low-Rank Adaptation (LoRA) is used as Parametric-Efficient Fine-Tuning (PEFT) technique\n'
  else:
    hps.info += 'No Parametric-Efficient Fine-Tuning (PEFT) technique is used\n'
  hps.info += '=' * 87 + '\n'
  
def load_model(hps: AttrDict) -> torch.nn.Module:
  """
  Load the base model.

  Args:
      hps (AttrDict): Hyperparameter object.

  Returns:
      torch.nn.Module: The base model.
  """
  from peft import PeftModel

  # Load base model
  if hps.use_4bit:
    # Compute dtype for 4-bit base model
    hps.bnb_4bit_compute_dtype = "bfloat16"
    # Quantization type (fp4 or nf4)
    hps.bnb_4bit_quant_type = "nf4"
    # Activate nested quantization for 4-bit base models (double quantization)
    hps.use_nested_quant = False
    hps.compute_dtype = getattr(torch, hps.bnb_4bit_compute_dtype)
    # Load the entire model on an available GPU
    hps.device_map = {"": 0 if torch.cuda.is_available() else 'cpu'}
    hps.logger.info(hps.device_map)
    # Quantization
    hps.bnb_config = BitsAndBytesConfig(
      load_in_4bit=hps.use_4bit,
      bnb_4bit_quant_type=hps.bnb_4bit_quant_type,
      bnb_4bit_compute_dtype=hps.compute_dtype,
      bnb_4bit_use_double_quant=hps.use_nested_quant,
    )
    model = AutoModelForCausalLM.from_pretrained(
      hps.model_name, quantization_config=hps.bnb_config, device_map=hps.device_map, # attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
    ) if hps.is_causal == True else AutoModelForSeq2SeqLM.from_pretrained(
      hps.model_name, quantization_config=hps.bnb_config, device_map=hps.device_map, # attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
    )

  elif hps.use_8bit:
    # Load the entire model on an available GPU
    hps.device_map = {"": hps.free_gpu_id if torch.cuda.is_available() else 'cpu'}
    # Quantization
    hps.bnb_config = BitsAndBytesConfig(
      load_in_8bit=hps.use_8bit
    )
    model = AutoModelForCausalLM.from_pretrained(
      hps.model_name, quantization_config=hps.bnb_config, device_map=hps.device_map, # attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
    ) if hps.is_causal == True else AutoModelForSeq2SeqLM.from_pretrained(
      hps.model_name, quantization_config=hps.bnb_config, device_map=hps.device_map, # attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
    )
  else:
    model = AutoModelForCausalLM.from_pretrained(hps.model_name, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, revision=hps.revision) if hps.is_causal == True else AutoModelForSeq2SeqLM.from_pretrained(hps.model_name, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, revision=hps.revision)
    model = model.to(torch.bfloat16)
  
  if hps.lora:
    hps.lora_config = LoraConfig(
      task_type=TaskType.CAUSAL_LM if hps.is_causal else TaskType.SEQ_2_SEQ_LM, r=hps.lora_r, lora_alpha=hps.lora_alpha, lora_dropout=hps.lora_dropout
    )
    if hps.adapter_name and hps.adapter_repo:
      model = PeftModel.from_pretrained(model, hps.adapter_repo, subfolder=hps.adapter_name, adapter_name=hps.adapter_name)
    else:
      model = get_peft_model(model, hps.lora_config) 
    model.to(torch.bfloat16) 

    train_layers(model, train_embeddings=True, train_head=True, train_lora=True, train_base_model=False) 
    hps.info += f"{model.print_trainable_parameters()}\n" 

  if hps.new_tokens:
    train_layers(model, train_embeddings=True, train_head=True, train_lora=False, train_base_model=False)
    # train_tokens_embeddings_and_head(model)
  else:
    if not hps.lora:
      train_layers(model, train_embeddings=True, train_head=True, train_lora=False, train_base_model=False) 

  hps.info += f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,d}\n"
  hps.info += f"Trainable Model size: {convert_bytes(model_size_in_bytes(model))}\n"

  model.config.use_cache = False
  model.config.pretraining_tp = 1
  torch.set_float32_matmul_precision('medium')
  return model

def train_tokens_embeddings_and_head(model, trainable_layers=["embed_tokens", "lm_head"]):
  for name, params in model.named_parameters():
    if any(layer in name for layer in trainable_layers):
      params.requires_grad = True
    else:
      params.requires_grad = False

def train_layers(model, train_embeddings=False, train_head=False, train_lora=False, train_base_model=False):
  for name, param in model.named_parameters():
    if 'lm_head' in name:
      param.requires_grad = train_head  # Train head
    elif 'embed_tokens' in name:
      param.requires_grad = train_embeddings  # Train embeddings
    elif 'lora' in name:
      param.requires_grad = train_lora  # Train LoRA layers
    else:
      param.requires_grad = train_base_model  # Freeze base model


def load_tokenizer(model, hps:AttrDict):
  """
  Load tokenizer from pre-trained model.
  Args:
      model (`obj:pytorch_model`):
          The model to be used for inference.
      hps (`obj:AttrDict`):
          Hyperparameter object.
  Returns:
      `obj:AutoTokenizer`:
          Tokenizer object.
  """
  from atpgllm.tokenization import get_new_tokens
  tokenizer = AutoTokenizer.from_pretrained(hps.model_name, model_max_length=hps.model_max_length, use_fast=True if hps.parallel else False)
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
  if not hps.is_causal:
    tokenizer.add_special_tokens({"cls_token": "<s>"})
  if hps.new_tokens:
    # Get new tokens
    hps.added_new_tokens = get_new_tokens(hps)
    # for every token
    for token in hps.added_new_tokens.copy(): 
      # check if token exist in vocab
      if tokenizer.convert_tokens_to_ids(token.content) != 0: 
        # remove existing tokens
        hps.added_new_tokens.remove(token) 
    hps.info += f"Tokenizer vocabulary: {len(tokenizer)}. New added tokens: {len(hps.added_new_tokens)}. "
    # Resize the Embeddings
    if len(hps.added_new_tokens) > 0:
      prev_size_tokenizer = len(tokenizer)
      tokenizer.add_tokens(hps.added_new_tokens)
      hps.tokenizer = tokenizer
      # Initialize the new embeddings from a multivariate that has old embeddings' mean and covariance. As described in the article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html.
      # To disable this, use `mean_resizing=False`
      model.resize_token_embeddings(len(tokenizer), mean_resizing=True)
      # initialize_new_embeddings(model, prev_size_tokenizer)
      hps.info += f"New Tokenizer vocabulary: {len(tokenizer)}.\n"
    else:
      hps.tokenizer = tokenizer
      hps.info += "\n"
  else:
    hps.tokenizer = tokenizer
    hps.info += f"Tokenizer vocabulary: {len(tokenizer)}. New added tokens: 0\n"
  return model

def initialize_new_embeddings(model, prev_size_tokenizer, noise_std=0.0001):
  mean_embedding = model.model.embed_tokens.weight[:prev_size_tokenizer].mean(0)
  new_added_tokens = model.model.embed_tokens.weight.shape[0] - prev_size_tokenizer
  embedding_dim = model.model.embed_tokens.weight.shape[1]
  with torch.no_grad():
    random_noise = torch.randn(new_added_tokens, embedding_dim) * noise_std
    model.model.embed_tokens.weight[prev_size_tokenizer:] = mean_embedding + random_noise

def save_model(model, save_directory: str, push_to_hub: bool = True, **kwargs):
  """
  Save the model to a directory.

  Args:
      model (`obj:pytorch_model`):
          The model to be saved.
      save_directory (`str`):
          The directory where the model should be saved.
      push_to_hub (`bool`, optional, default=False):
          Whether to push the model to the model hub.
      kwargs (`Dict[str, Any]`, optional):
          Additional keyword arguments used during saving.

  Returns:
      `None`
  """
  import threading
  save_embedding_layers = kwargs.get('save_embedding_layers', False)
  hps = kwargs.get('hps', AttrDict({}))
  model.eval()
  # Save model to directory
  if hps.parallel:
    if hps.deepspeed_kernel:
      if hps.accelerator.is_main_process:
        # push to the hub!
        push_full_model(model, save_directory=save_directory, **kwargs)  
      hps.accelerator.wait_for_everyone()
    elif not hps.deepspeed_kernel:
      if is_main_process():
        # push to the hub!
        push_full_model(model, save_directory=save_directory, **kwargs)
  else:
    # push to the hub!
    push_full_model(model, save_directory=save_directory, **kwargs)


def push_full_model(model, save_directory, hps, **kwargs):
  """
  Push the full model to the Hugging Face Hub.
  """
  from peft import PeftModelForCausalLM
  commit_message = kwargs.get('commit_message', "")
  private = kwargs.get('private', True)
  merge_lora = kwargs.get('merge_lora', False)
  tmp_dir = kwargs.get('tmp_dir', './tmp_dir')
  tmp_dir = os.path.join(tmp_dir, save_directory)
  accelerator = Accelerator()

  peft_model = accelerator.unwrap_model(model)

  if merge_lora:
    model_to_push = peft_model.merge_and_unload()
  else:
    model_to_push = peft_model
  
  tmp_dir = Path(tmp_dir)
  tmp_dir.mkdir(parents=True, exist_ok=True)

  if hps.train_lora or hps.train_rl:
    clean_base = load_model(hps)
    clean_sd = {}
    if isinstance(peft_model, PeftModelForCausalLM):
      for k, v in peft_model.base_model.model.state_dict().items():
        if k.replace(".base_layer.", ".") not in clean_base.state_dict().keys():          # drop LoRA params
          continue
        clean_sd[k.replace(".base_layer.", ".")] = v 
    missing, unexpected = clean_base.load_state_dict(clean_sd, strict=False)
    assert not unexpected, unexpected
    assert not missing, missing
    model_to_push.save_pretrained(tmp_dir, safe_serialization=True)
    clean_base.save_pretrained(tmp_dir, 
                                safe_serialization=True, 
                                save_embedding_layers=True, 
                                save_transformers_model=True,
                                max_shard_size="10GB",
                                )
  elif hps.new_tokens:
    model_to_push.save_pretrained(tmp_dir, 
                                  safe_serialization=True, 
                                  save_embedding_layers=True, 
                                  save_transformers_model=True,
                                  max_shard_size="5GB",
                                  )
  else:
    model_to_push.save_pretrained(tmp_dir, safe_serialization=True, max_shard_size="5GB")

  if hps.tokenizer is not None:
    hps.tokenizer.save_pretrained(tmp_dir)

  # Copy the file
  try:
    shutil.copy2(hps.file_path, os.path.join(tmp_dir, hps.filename))
    print(f"✅ Copied {hps.file_path} to {os.path.join(tmp_dir, hps.filename)}")
  except Exception as e:
    print(f"❌ Failed to copy file: {e}")

  try:
    hps.api.upload_folder(repo_id=save_directory, 
                          folder_path=str(tmp_dir), 
                          commit_message=commit_message, 
                          repo_type="model", 
                          )
    print(f"✅ pushed to https://huggingface.co/{save_directory}")
  except Exception as e:
    print(f"❌ Failed to upload folder to https://huggingface.co/{save_directory}: \n{e}")


def upload_model(model, save_directory, push_to_hub, hps, **kwargs):
  """
  Upload or save a model to the Hugging Face Hub or local directory.
  
  Args:
      model: The model to be uploaded or saved
      save_directory: Directory path where the model will be saved
      push_to_hub: Boolean flag to determine if model should be pushed to HF Hub
      hps: Hyperparameters object containing configuration settings
      **kwargs: Additional keyword arguments
          - commit_message: Custom commit message for Hub uploads
          - delete_previous: Whether to delete previous local checkpoints
  
  Returns:
      None
  """
  # Extract commit message from kwargs or use empty string as default
  commit_message = kwargs.get('commit_message', "")
  private = kwargs.get('private', True)
  
  # Push model to Hugging Face Hub if specified
  if push_to_hub:
    # First upload the source file to the repository for reference
    hps.api.upload_file(
        repo_id=save_directory, 
        path_or_fileobj=hps.file_path, 
        path_in_repo=hps.filename, 
        commit_message=f"Upload {hps.filename} before upload {commit_message}"
    )

    print(f"Pushing model to Hugging Face Hub: {save_directory}")
    
    # Handle differently based on whether we're using LoRA or not
    if hps.lora or hps.train_lora:
      # For LoRA models, we need to push tokenizer and base model separately
      hps.tokenizer.push_to_hub(save_directory, private=True)
      model.push_to_hub(
          save_directory, 
          private=private, 
          safe_serialization=True,  # Use safetensors format
          commit_message=f"LoRA adapters {commit_message}"
      )
      model.base_model.model.push_to_hub(
          save_directory, 
          private=private, 
          safe_serialization=True,  # Use safetensors format
          commit_message=f"Base model {commit_message}"
      )
    else:
      # For full models, push tokenizer and model
      hps.tokenizer.push_to_hub(save_directory, private=True)
      model.push_to_hub(
          save_directory, 
          private=private, 
          safe_serialization=True,  # Use safetensors format
          commit_message=f"LoRA adapters {commit_message}"
      )
      model.base_model.model.push_to_hub(
          save_directory, 
          private=private, 
          safe_serialization=True,  # Use safetensors format
          commit_message=f"Base model {commit_message}"
      )
  else:
    # Save model locally instead of pushing to Hub
    print(f"Saving model locally to {save_directory}")
    
    # Check if we should delete previous checkpoints before saving
    delete_previous = kwargs.get('delete_previous', False)
    if delete_previous and os.path.exists(save_directory):
      print(f"Deleting previous checkpoint at {save_directory}")
      try:
        # Remove the entire directory and recreate it
        shutil.rmtree(save_directory)
        os.makedirs(save_directory, exist_ok=True)
        print(f"✅ Previous checkpoint deleted successfully")
      except Exception as e:
        print(f"⚠️ Error deleting previous checkpoint: {e}")
    
    # Save the model based on architecture type
    if hps.lora or hps.train_lora:
      # For LoRA models, save tokenizer and base model with adapters
      hps.tokenizer.save_pretrained(save_directory)
      model.base_model.model.save_pretrained(save_directory, safe_serialization=True)
    else:
      # For full models, save tokenizer and complete model
      hps.tokenizer.save_pretrained(save_directory)
      model.save_pretrained(save_directory, safe_serialization=True)
    
    print(f"✅ Model saved locally to {save_directory}")


def is_main_process():
  return not dist.is_initialized() or dist.get_rank() == 0

def is_distributed_execution():
  return 'WORLD_SIZE' in os.environ

def print_peak_memory(prefix, device):
  print(f"[{device}]: {prefix}: {convert_bytes(torch.cuda.max_memory_allocated(device))}")

def zero_grad_old_embeddings(model, tunable_ids = 32000):
  for module in model.modules():
    if isinstance(module, nn.Embedding):
      # Create a mask for embeddings to freeze
      mask = torch.ones_like(module.weight.grad, dtype=torch.bool)
      mask[tunable_ids] = False  # Do not freeze tunable embeddings
      module.weight.grad[mask].zero_()
      return

def infer(wrapped_model, dataloader, tokenizer, model_max_length=4096, epoch=1, file_path='generated_text.txt', parallel=False, new_file=True):
  """
  Creates a copy of the model, merges LoRA weights, and runs inference using token_ids.
  
  Args:
    ddp_model: The DDP-wrapped model during training.
    data_iterator: An iterator that returns token_ids from the validation dataset.
    tokenizer: The tokenizer for the model.
    model_max_length: The maximum length of the generated text.
    epoch: The current epoch number.
    file_path: The path to the file where the generated text will be saved.
    parallel: Whether to run inference in parallel.
    new_file: Whether to create a new file for the generated text.

  Returns:
    None
  """
  import itertools
  # To achieve efficient completion set left padding. Left padding comes with user prompt cropping so that model generates the completion
  dataloader.collate_fn.set_left_padding()
  data_iterator = iter(itertools.cycle(dataloader))
  # Ensure we only do this on rank 0 (main process) to avoid redundant work on other GPUs
  if is_main_process():
    print(f"Generate text. Please wait...")
    # Store the current device of the original DDP model to restore it later
    original_device = next(wrapped_model.parameters()).device

    try:
      
      # Don't use gradients
      wrapped_model.eval()
      
      # Iterate over the data iterator and accumulate token_ids until the stop_token sequence is found
      data = next(data_iterator)

      # Open the file based on the epoch
      mode = 'w' if new_file else 'a'
      # Open the file in the specified mode
      with open(file_path, mode) as file:
        file.write(f"\n---\nEpoch: {epoch}\n\n---\n")

        # Move to GPU
        ids = data.input_ids[:2].to(original_device).contiguous() # get 2 prompts from the micro_batch
        mask = data.attention_mask[:2].to(original_device).contiguous() # get 2 prompts from the micro_batch

        # Generate completion
        generate_model = wrapped_model.module if isinstance(wrapped_model, FSDP) or isinstance(wrapped_model, DDP) else wrapped_model 
        with torch.no_grad():
          generated_ids = generate_model.generate(input_ids=ids, attention_mask=mask, num_return_sequences=2, max_length=model_max_length, do_sample=True, temperature=0.6, top_p=0.95)
        
        torch.cuda.empty_cache()

        # Decode and flush the generated text to a file
        generated_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for i, generated_output in enumerate(generated_outputs):
          # postproc_generated_text = postprocess_generated_text(generated_output)
          file.write(f"Generated Sample {i+1}:\n{generated_output}\n\n")
        file.write("-" * 80)
        file.write('\n\n')
    except Exception as e:
      print(f"Error generating text: {e}")
    finally:
      # Empty cache
      torch.cuda.empty_cache()

      gc.collect()
      gc.collect()

      # After inference, move the original DDP-wrapped model back to its original device (e.g., GPU)
      print(f"Check {file_path} to estimate progress. Epoch {epoch} is completed.")
  if parallel:
    # Ensure all processes stay in sync
    dist.barrier()

# Function to postprocess the generated text
def postprocess_generated_text(generated_text):
  generated_text = re.sub(r"(_\d+_)\s([,\"\);])|(IBUF|XNR|XOR)\s(\d)", lambda m: f"{m.group(1) or m.group(3)}{m.group(2) or m.group(4)}", generated_text)
  return re.sub(r" (\[\/INST\]) ", r"\n\1\n", generated_text)


def align_tokens_length(tokenizer, x):
  """
  Aligns all inputs in a batch to the maximum length in the batch.

  Args:
    tokenizer: The tokenizer for the model.
    x: A list of tensors, each of shape (B, T, Embeddings) or (B, T).
  Returns:
    A tensor of shape (B, max_length, Dims) or (B, max_length).
  """
  max_size = max(x[i].size(1) for i in range(len(x)))
  # Find out the padding is needed for each tensor of the list
  padding = [max_size - x[i].size(1) for i in range(len(x))]
  # case where size is (B, T, Embeddings)
  if len(x[0].size()) == 3:
    # Randomize the paddings uniformly random in range of [0, 1]
    padding = [torch.rand(x[0].size(0), pad, x[0].size(2)) for pad in padding]
    # Assign 100 to the index of the end of string (eos_token_id)
    # It's set to 100 due to low ranomized values. It has to be the maximum number among embeddings. It will return eos_token_id later
    for pad in padding:
      pad[:, :, tokenizer.eos_token_id] = 100
    x = [torch.cat([x[i], pad], dim=1) for i, pad in enumerate(padding)]
    x = torch.cat(x)
    return x
  else:
    # Pad the targets with the eos_token_id
    x = torch.cat([F.pad(xx, (0, padding[i]), "constant", tokenizer.eos_token_id) for i, xx in enumerate(x)])
    return x


def get_user_prompt(text_inputs, system_tag="<</SYS>>", instr_tag="[/INST]"):
  prompts = []
  for text_input in text_inputs:
    if system_tag:
      sys_split = text_input.split(system_tag)
      if len(sys_split) > 2:
        raise ValueError(f"There are multiple {system_tag}") 
      sys_split = sys_split[1]
    else:
      sys_split = text_input
    if instr_tag:
      instr_split = sys_split.split(instr_tag)
      if len(instr_split) > 2:
        prompts.append(instr_tag.join(instr_split[:2]).strip())
      else:
        prompts.append(instr_split[0].strip())
    else:
      prompts.append(sys_split)
  return prompts


def print_prompt_completions_sample(prompts: list[str], completions: list[str], rewards: list[int], step: int) -> None:
  """
  Print out a sample of model completions to the console.

  This function creates a nicely formatted table showing prompt-completion pairs, useful for monitoring model outputs
  during training. It requires the `rich` library to be installed.

  Args:
      prompts (`list[str]`):
          List of prompts.
      completions (`list[str]`):
          List of completions corresponding to the prompts.
      reward (`list[float]`):
          List of rewards corresponding to the completions.
      step (`int`):
          Current training step number, used in the output title.

  Example:
  ```python
  >>> from trl.trainer.utils import print_prompt_completions_sample
  >>> prompts = ["The sky is", "The sun is"]
  >>> completions = [" blue.", " in the sky."]
  >>> rewards = [0.12345, 0.68789]
  >>> print_prompt_completions_sample(prompts, completions, rewards, 42)
  ╭─────────────── Step 42 ────────────────╮
  │ ┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━┓ │
  │ ┃ Prompt     ┃ Completion   ┃ Reward ┃ │
  │ ┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━┩ │
  │ │ The sky is │  blue.       │   0.12 │ │
  │ ├────────────┼──────────────┼────────┤ │
  │ │ The sun is │  in the sky. │   0.68 │ │
  │ └────────────┴──────────────┴────────┘ │
  ╰────────────────────────────────────────╯
  ```
  """
  if not is_rich_available():
    raise ImportError("This feature requires `rich` to be installed. Please install it first: `pip install rich`")

  console = Console(highlight=True)
  table = Table(show_header=True, header_style="bold", title_style="bold", expand=True)

  # Add columns
  table.add_column("[bold]Prompt[/bold]")
  table.add_column("[bold]Completion[/bold]")
  table.add_column("[bold]Reward[/bold]", justify="right")

  for prompt, completion, reward in zip(prompts, completions, rewards, strict=True):
    if reward > 0:
      table.add_row(Text(prompt), Text(completion), Text(f"{reward:.2f}", style="bold green"))  # Formatting reward to 2 decimal places
    else:
      table.add_row(Text(prompt), Text(completion), Text(f"{reward:.2f}", style="bold red"))  # Formatting reward to 2 decimal places
    table.add_section()  # Adds a separator between rows

  # Create a panel with bold border
  panel = Panel(table, title=f"[bold]Step {step}[/bold]")
  # Print with bold lines
  console.print(panel)
  
def balance_dataset_by_gates_count(dataset):
  """
  Balances a dataset by ensuring each gate count has the same number of examples.
  Instead of using the most frequent count as target, it allows for more flexible balancing.
  
  Args:
      dataset: The dataset to balance
      
  Returns:
      A balanced dataset with equal representation of each gate count
  """
  import pandas as pd
  # Convert to pandas for more efficient grouping and sampling
  df = dataset.to_pandas()
  
  # Group by gates_count and get counts
  gate_count_groups = df.groupby('gates_count')
  counts = {count: len(group) for count, group in gate_count_groups}
  
  # Find the target count (using the most frequent by default)
  target_count = max(counts.values())
  
  # Create balanced samples for each gate count
  balanced_dfs = []
  for count, group in gate_count_groups:
    # Set a fixed random seed based on the count to ensure all processes sample the same data
    random_seed = hash(f"gate_count_{count}") % 10000
    
    if len(group) < target_count:
      # If we have fewer examples than target, sample with replacement
      # Use a fixed random state to ensure all processes get the same samples
      balanced_group = group.sample(n=target_count, replace=True, random_state=random_seed).reset_index(drop=True)
    elif len(group) > target_count:
      # If we have more, sample without replacement
      # Use a fixed random state to ensure all processes get the same samples
      balanced_group = group.sample(n=target_count, replace=False, random_state=random_seed).reset_index(drop=True)
    else:
      balanced_group = group
    
    balanced_dfs.append(balanced_group)

  # Combine all balanced groups
  balanced_df = pd.concat(balanced_dfs, ignore_index=True)
  
  # Convert back to datasets format
  from datasets import Dataset
  balanced_dataset = Dataset.from_pandas(balanced_df)
  
  # Create a visualization comparing original vs balanced distribution
  import plotly.graph_objects as go
  
  # Get counts for original dataset
  original_counts = df['gates_count'].value_counts().sort_index()
  
  # Get counts for balanced dataset
  balanced_counts = balanced_df['gates_count'].value_counts().sort_index()
  
  # If we're in a distributed environment, make sure all processes are synced
  # Your balance_dataset_by_gates_count reshuffles, oversamples, and resamples the data per-process.
  # random seed fixes based on gate_count, but:
  # - Tiny differences in local state (like numpy random generator or pandas sampling behavior) can still cause drift.
  # - Even if sampling identical, shuffling or order during to_pandas() or from_pandas() can slightly differ.
  # Move the balancing to rank 0 only, and broadcast to all other ranks.
  if dist.is_initialized():
    if dist.get_rank() == 0:
      balanced_dataset.save_to_disk('/tmp/train_dataset')
    dist.barrier()
    # Now all ranks have the same train_dataset
    balanced_dataset = Dataset.load_from_disk('/tmp/train_dataset')
    
  return balanced_dataset
