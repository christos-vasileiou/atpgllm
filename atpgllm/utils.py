# Helper Functions
import torch
import argparse
import os
import logging
import numpy as np
import deepspeed
import matplotlib.pyplot as plt
import subprocess
import torch.distributed as dist
import torch.nn as nn
from torch.optim import AdamW
from tqdm.auto import tqdm
from collections import Counter
from itertools import product
from torch.utils.data import DataLoader
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, f1_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from datasets import DatasetDict, Dataset, load_dataset
from atpgllm import deepspeed_config
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from transformers import (
  AutoModelForCausalLM,
  AutoModelForSeq2SeqLM,
  BitsAndBytesConfig,
)

def patterns_lengths(batch, max_freq):
  patterns_lengths = [len(example.split('\n')[0].split()) for example in batch['patterns']]
  batch['patterns_lengths'] = [l if l == max_freq else None for l in patterns_lengths]
  return batch


def initial_data_preprocessing(raw_dataset):
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


def model_size_in_bytes(model):
  total_size = 0
  for p in model.parameters():
    if p.dtype == torch.bfloat16:
      total_size += p.numel() * 2
    elif p.dtype == torch.float16:
      total_size += p.numel() * 2
    elif p.dtype == torch.int8:
      total_size += p.numel() * 1
    elif p.dtype == torch.int32:  # assuming int is 32-bit
      total_size += p.numel() * 4
    elif p.dtype == torch.float32:
      total_size += p.numel() * 4
    elif p.dtype == torch.float64:
      total_size += p.numel() * 8
  return total_size


def convert_bytes(num_bytes):
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


def setup_logging(rank):
  # Ensure the logs directory exists
  log_dir = 'logs'
  if not os.path.exists(log_dir):
    print(log_dir)
    os.makedirs(log_dir, exist_ok=True)

  # Create a logger
  logger = logging.getLogger(__name__)
  logger.setLevel(logging.INFO)
  
  # Create a console handler and set level to info
  #ch = logging.StreamHandler()
  #ch.setLevel(logging.INFO)
  #logger.addHandler(ch)
 
  # Check if the file exists and try different filenames if it does
  base_filename = f"process_{rank}"
  extension     = '.log'
  counter       = 0
  filename      = os.path.join(log_dir, f"{base_filename}_{counter}{extension}")
  while os.path.exists(filename) and os.path.isfile(filename):
    counter  += 1
    filename  = os.path.join(log_dir, f"{base_filename}_{counter}{extension}")
  # Create a file handler and set level to info
  fh = logging.FileHandler(filename)
  fh.setLevel(logging.INFO)
  logger.addHandler(fh)

  # Create a formatter
  formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
  
  # Add formatter to ch and fh
  #ch.setFormatter(formatter)
  fh.setFormatter(formatter)
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
  #valid_loss_reg   = (invalid_outputs.size(0) / (outputs.size(0) * max(valid_outputs.size(0), 1))) 
  #invalid_loss_reg = (  valid_outputs.size(0) / (outputs.size(0) * max(invalid_outputs.size(0), 1)))
  if loss_ratio is None or epoch%5 == 0:
    vld_invld_loss_ratio = valid_loss/invalid_loss

  # Final loss calculation
  loss             += valid_loss + vld_invld_loss_ratio * invalid_loss
  #loss             += valid_loss_reg * valid_loss + invalid_loss_reg * invalid_loss
  #loss             += valid_loss + invalid_loss
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
      #print("sigmoid", loss_sigmoid)
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
    
    # Get the GPU with the maximum free memory
    return memory_free_values.index(max(memory_free_values))
  
  except Exception as e:
    print("Could not run nvidia-smi:", e)
    return None


def parse_arguments(parser):  
  # Add arguments
  parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
  parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
  parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
  parser.add_argument('--dropout_p', type=float, default=0.1, help='Dropout probability')
  parser.add_argument('--model_checkpoint', type=str, default='bert', help='Based Model checkpoint for training. Values can take: "t5-small", "t5-base", "t5-v1_1-base", "t5-large", "t5-xl", "t5-xxl", "mt5-base", "m2m100", "t5-finetuned", "led-base", "distilbert", "bert"')
  parser.add_argument('--is_causal', action='store_true', help='Specify if the you want to have a causal model specified')  
  parser.add_argument('--use_4bit', action='store_true', help='Apply 4-bit quantization on the model. Be advised that the model will be loaded on float32 but the training will take place on a device i.e. gpu, will be then quantized to the specified bit precision')
  parser.add_argument('--use_8bit', action='store_true', help='Apply 8-bit quantization on the model. Be advised that the model will be loaded on float32 but the training will take place on a device i.e. gpu, will be then quantized to the specified bit precision')
  
  parser.add_argument('--peft', action='store_true', help='Apply Parametric-Efficient Fine-Tuning (PEFT) with the use of LoRA technique. Specify the appropriate lora hyperparameters.')  
  parser.add_argument('--data_file', type=str, default=None, required=True, help='Data file for training')
  parser.add_argument('--vocab_file', type=str, default=None, help='Use vocabulary for the custom tokenizer')
  parser.add_argument('--parallel', action='store_true', help='Parallel training using Distributed Data Parallelization')  
  parser.add_argument('--deepspeed_kernel', action='store_true', help='Use DeepSpeed transformer kernel to accelerate')
  parser.add_argument('--fp16', action='store_true', help='Store the model as dtype torch.bfloat16')
  parser.add_argument('--tokenizer', type=str, default='custom', help='Which Tokenizer will be used. Values can take: "t5-small", "t5-base", "t5-v1_1-base", "t5-large", "t5-xl", "t5-xxl", "mt5-base", "m2m100", "t5-finetuned", "led-base", "distilbert", "bert"')
  parser.add_argument('--zero_stage', type=int, default=2, help="Enable ZeRO memory optimizations, compatible with FP16/BF16/FP32 and the Adam Optimizer. zero_stage: Chooses different of ZeRO Optimizer. Stage 0, 1, 2, and 3 refer to disabled, optimizer, state partioning, and optimizer+gradient state partitioning, and optimizer+gradient+parameter partitioning, respectively.")
  parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
  parser.add_argument('--load_checkpoint', type=str, default=None, help='Pre-Trained weights checkpoint from which a model will be loaded')
  parser.add_argument('--atpg_collate', action='store_true', help='Use Custom ATPG Collate so that have more efficient training with less tokens. Tokenization will take place during training process.')
  
  parser.add_argument('--lora_alpha', type=int, default=16, help="Low-Rank Adaptation (LoRA) alpha parameter. It's used  for scaling the weights of lora layers")
  parser.add_argument('--lora_r', type=int, default=64, help="Low-Rank Adaptation (LoRA) rank parameter. it's the size of A and B matrices. The intermediate size of the 2 matrices. i.e. (B_dim, r) x (r, A_dim) = (B_dim, A_dim)")

  args = parser.parse_args()
  if args.deepspeed_kernel == True:
    parser = deepspeed.add_config_arguments(parser)
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
  weight = torch.zeros(len(hps.tokenizer), dtype=torch.bfloat16)
  # weight only the vocabulary's digits 
  for i in range(10):
    weight[hps.tokenizer.convert_tokens_to_ids(str(i))] = 1
  criterion = nn.CrossEntropyLoss(weight=weight) 
  return criterion 


def set_training_environment(model, tokenized_dataset, hps):
  # Set optimizer
  hps.optimizer = AdamW(model.parameters(), lr=hps.lr)
  logger = None

  # Set the distributed systems if necessary
  if hps.parallel==True:
    if hps.deepspeed_config is not None:
      
      model, hps.optimizer, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=hps.deepspeed_config)
      local_rank = model.local_rank
      
      hps.device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
      # logger = setup_logging(local_rank)
      # logger.info(f"{hps.device}\n{model}\n{hps.optimizer}\nModel parameters: {sum(p.numel() for p in model.parameters())}, \nModel size: {convert_bytes(model_size_in_bytes(model))}\n")
      print(f"{hps.device}\n{model}\n{hps.optimizer}\nModel parameters: {sum(p.numel() for p in model.parameters())}, \nModel size: {convert_bytes(model_size_in_bytes(model))}\n")

    elif hps.deepspeed_config is None:
      local_rank       = int(os.environ['LOCAL_RANK'])
      rank             = int(os.environ['RANK'])
      group_rank       = int(os.environ['GROUP_RANK'])
      role_rank        = int(os.environ['ROLE_RANK'])
      local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
      world_size       = int(os.environ['WORLD_SIZE'])
      dist.init_process_group(backend='nccl', rank=local_rank, world_size=world_size)

      hps.device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
      print(f"local_rank: {local_rank}, rank: {rank}, group_rank: {group_rank}, role_rank: {role_rank}, local_world_size: {local_world_size} world_size: {world_size}")
      print(f"free gpu: {hps.free_gpu_id}")
      print(f"Backend Configuration: {dist.get_backend_config()}")
      torch.manual_seed(27)
      torch.cuda.set_device(local_rank)
      model = model.to(hps.device)
      model = DDP(model, device_ids=[local_rank], output_device=local_rank)
      # move_optimizer_to_device(hps.optimizer, hps.device)
      # logger = setup_logging(local_rank)
      # logger.info(f"{hps.device}\n{model}\n{hps.optimizer}\nModel parameters: {sum(p.numel() for p in model.parameters())}, \nModel size: {convert_bytes(model_size_in_bytes(model))}\n")

      # print(f"{hps.device}\n{model}\n{hps.optimizer}\nModel parameters: {sum(p.numel() for p in model.parameters())}, \nModel size: {convert_bytes(model_size_in_bytes(model))}\n")
  else:
    hps.device = torch.device(f'cuda:{hps.free_gpu_id}' if torch.cuda.is_available() else 'cpu')
    # logger = setup_logging(hps.free_gpu_id)
    if hps.use_4bit == False and hps.use_8bit == False:
      model  = model.to(hps.device)

  # Set the loss functions
  hps.criterion = nn.CrossEntropyLoss(ignore_index=hps.tokenizer.pad_token_id if hps.is_causal else -100).to(hps.device)
  hps.patterns_criterion = get_patterns_criterion(hps).to(hps.device)

  # Set the flags for distributed systems. Data samplers and Data Loaders
  use_sampler = hps.parallel and hps.deepspeed_config is None
  hps.shuffle = not use_sampler
  
  if use_sampler:
    training_sampler   = DistributedSampler(tokenized_dataset['train'], rank=local_rank, num_replicas=local_world_size, drop_last=True)
    validation_sampler = DistributedSampler(tokenized_dataset['validation'], rank=local_rank, num_replicas=local_world_size, drop_last=True)
    testing_sampler    = DistributedSampler(tokenized_dataset['test'], rank=local_rank, num_replicas=local_world_size, drop_last=True)
  else:
    training_sampler, validation_sampler, testing_sampler = None, None, None
  
  # Create the Data Loaders
  training_loader   = DataLoader(dataset=tokenized_dataset['train'], batch_size=hps.batch_size, shuffle=hps.shuffle, collate_fn=hps.collate_fn, sampler=training_sampler, num_workers=hps.num_workers)
  validation_loader = DataLoader(dataset=tokenized_dataset['validation'], batch_size=hps.batch_size, shuffle=hps.shuffle, collate_fn=hps.collate_fn, sampler=validation_sampler, num_workers=hps.num_workers)
  testing_loader    = DataLoader(dataset=tokenized_dataset['test'], batch_size=hps.batch_size, shuffle=hps.shuffle, collate_fn=hps.collate_fn, sampler=testing_sampler, num_workers=hps.num_workers)

  return model, logger, training_loader, validation_loader, testing_loader


class AttrDict(dict):
  def __init__(self, initial_dict=None, **kwargs):
    super(AttrDict, self).__init__()
    if initial_dict is not None:
      self.update(initial_dict)
    self.update(kwargs)

  def __getattr__(self, key):
    try:
      return self[key]
    except KeyError:
      raise AttributeError(f"No such attribute: {key}")

  def __setattr__(self, key, value=None):
    if isinstance(key, str):
      self[key] = value
    else:
      raise ValueError(f'Invalid attribute assignment. You passed as key: {key} and value: {value}')

deepspeed_config = AttrDict(deepspeed_config)

def dataset_formation(raw_dataset, is_causal):
  """
  DatasetDict({
      train: Dataset({
          features: ['module_name', 'prompts', 'answers', 'patterns', 'faults', 'atpg', 'netlist'],
          num_rows: 150000
      })
  })
  """
  goal = {"test coverage": 7, "fault coverage": 8}
  system_prompt = "<<SYS>> You are a Test Pattern Generation tool able to create test vectors that are applied to integrated circuits to check and detect manufacturing defects. <</SYS>>\n\n"
  prompts = ["[INST] " + system_prompt + "Your task is to write test vectors that can achieve " + coverage.split('\n')[goal['test coverage']].split()[-1] + " test coverage for the design:\n\n```\n" + netlist + "```\n\nPlease wrap the test vectors in ```. Please think carefully the steps you need to go over in order to come to a conclusion. [/INST]" for netlist, coverage in zip(raw_dataset['train']['netlist_only_gates'], raw_dataset['train']['atpg'])]
  answers = ["The test vectors for the provided design are:\n\n```\n" + '\n'.join([pat for pat in patterns.replace(' ', '').split('\n')]) + "\n```" for patterns in raw_dataset['train']['patterns']]
  if is_causal:
    o_dataset = [prompt + "\n" + answer for prompt, answer in zip(prompts, answers)]
  else:
    o_dataset = [{'prompts': prompt, 'answers': answer} for prompt, answer in zip(prompts, answers)]
  
  test_size  = .3
  val_size   = (1 - test_size) * 0.15 
  train_size = (1 - test_size) * 0.85 
  assert train_size + val_size + test_size == 1

  train_size = int(train_size * len(o_dataset))
  val_size   = int(val_size * len(o_dataset))
  test_size  = len(o_dataset) - train_size - val_size
  assert train_size + val_size + test_size == len(o_dataset)

  train_dataset = Dataset.from_dict( {'text': o_dataset[ :train_size], 'netlist': raw_dataset['train']['netlist'][ :train_size]} )
  val_dataset   = Dataset.from_dict( {'text': o_dataset[ train_size : train_size+val_size], 'netlist': raw_dataset['train']['netlist'][ train_size : train_size+val_size]} )
  test_dataset  = Dataset.from_dict( {'text': o_dataset[ train_size+val_size: ], 'netlist': raw_dataset['train']['netlist'][ train_size+val_size: ]} )
  
  dataset = DatasetDict({'train': train_dataset, 'validation': val_dataset, 'test': test_dataset})
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


def load_raw_dataset(data_file):
  """
  Load the dataset given at the data_file argument
  """
  raw_dataset = load_dataset('csv', data_files=data_file).remove_columns('Unnamed: 0')
  return raw_dataset


def sizeof_tensor(tensor):
  return tensor.element_size() * tensor.nelement()


def calculate_memory(model, optimizer, ids, mask, targets):
  total_memory = 0
  # Model parameters
  for param in model.parameters():
      total_memory += sizeof_tensor(param)
      if param.grad is not None:
          total_memory += sizeof_tensor(param.grad)
  # Optimizer states
  for state in optimizer.state.values():
      for k, v in state.items():
          if isinstance(v, torch.Tensor):
              total_memory += sizeof_tensor(v)
  # ids + mask + targets
  total_memory += sizeof_tensor(ids) + sizeof_tensor(mask) + sizeof_tensor(targets)
  return convert_bytes(total_memory)


def hyperparameters(args):
  # args
  hps = AttrDict({})
  # training
  hps.epochs      = args.epochs 
  hps.lr          = args.lr
  hps.batch_size  = args.batch_size
  hps.num_workers = 0
  # Low-Rank Adaptation (LoRA) - Parametric-Efficient Fine-Tuning (PEFT)
  hps.lora_alpha   = args.lora_alpha
  hps.lora_r       = args.lora_r
  hps.lora_dropout = args.dropout_p
  # model
  hps.dropout    = args.dropout_p
  hps.data_file  = args.data_file
  hps.vocab_file = args.vocab_file
  hps.model_name = args.model_checkpoint
  hps.fp16       = args.fp16
  hps.is_causal  = args.is_causal
  hps.peft       = args.peft 
  hps.load_ckpt  = args.load_checkpoint
  # tokenizer
  hps.tokenizer = args.tokenizer
  # collate
  hps.collate = args.atpg_collate
  # gpu usage
  hps.free_gpu_id      = get_free_gpu()
  hps.parallel         = args.parallel
  hps.deepspeed_kernel = args.deepspeed_kernel
  hps.world_size       = int(len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))) if hps.deepspeed_kernel == True else 1
  hps.device_map       = None
  hps.use_4bit         = args.use_4bit
  hps.use_8bit         = args.use_8bit
  deepspeed_config.train_batch_size               = hps.batch_size
  deepspeed_config.train_micro_batch_size_per_gpu = hps.batch_size // hps.world_size
  deepspeed_config.zero_optimization['stage']     = args.zero_stage if hps.deepspeed_kernel == True else 0
  deepspeed_config.optimizer['params']['lr']      = hps.lr
  hps.deepspeed_config = deepspeed_config if hps.deepspeed_kernel == True else None
  
  return hps


def plot_training_plots(train_losses, val_losses):
  # Assuming all lists are of the same length
  epochs = list(range(1, len(train_losses) + 1))

  # Create traces
  trace1 = go.Scatter(x=epochs, y=train_losses, mode='lines+markers', name='Training Loss')
  trace2 = go.Scatter(x=epochs, y=val_losses, mode='lines+markers', name='Validation Loss')
  
  # Create the figure
  fig = make_subplots(specs=[[{"secondary_y": True}]])

  # Adding traces
  fig.add_trace(trace1)
  fig.add_trace(trace2)
  
  # Update layout
  fig.update_layout(title='Training, and Validation Losses',
                    xaxis_title='Epoch',
                    yaxis_title='Loss',
                    legend_title='Loss Type')
  
  # Save the figure as an interactive HTML
  fig.write_html("losses_plot.html")


def cleanup(hps):
  # for key in hps.keys():
  #   hps[key] = None
  #del hps
  import gc
  gc.collect()
  gc.collect()


def load_model(hps):
  print('*' * 12 + ' Model Loading ' + '*' * 12)
  if hps.parallel:
    try: 
      # When any quantization type is activated you can't work on distributed systems
      assert hps.use_4bit == False and hps.use_8bit == False
    except AssertionError:
      raise ValueError(f"When hyperparameter 'parallel' is acticated (parallel={hps.parallel}). Both quantization types should be de-activated. You set use_4bit={hps.use_4bit} and use_8bit={hps.use_8bit}")
    finally:
      print(f"No quantization is applied. Parallelization is activated")
  elif not hps.parallel:
    try:
      # Can't activate both 4-bit and 8-bit quantization
      assert (not (hps.use_4bit and hps.use_8bit)) == True
    except AssertionError: 
      # There is a conflict when both types of quantization have been activated
      raise ValueError(f"You can't have activated both 4-bit and 8-bit quantization")
    finally:
      if hps.use_4bit:
        print(f"4-Bit quantization is applied")
      elif hps.use_8bit:
        print(f"8-Bit quantization is applied")
      else:
        print(f"No quantization is applied")
  print('*' * 39)

  # Load base model
  if hps.use_4bit:
    # Compute dtype for 4-bit base model
    hps.bnb_4bit_compute_dtype = "float16"
    # Quantization type (fp4 or nf4)
    hps.bnb_4bit_quant_type = "nf4"
    # Activate nested quantization for 4-bit base models (double quantization)
    hps.use_nested_quant = False
    hps.compute_dtype = getattr(torch, hps.bnb_4bit_compute_dtype)
    # Load the entire model on an available GPU
    hps.device_map = {"": hps.free_gpu_id if torch.cuda.is_available() else 'cpu'}

    # Quantization
    hps.bnb_config = BitsAndBytesConfig(
      load_in_4bit=hps.use_4bit,
      bnb_4bit_quant_type=hps.bnb_4bit_quant_type,
      bnb_4bit_compute_dtype=hps.compute_dtype,
      bnb_4bit_use_double_quant=hps.use_nested_quant,
    )
    model = AutoModelForCausalLM.from_pretrained(hps.model_name, quantization_config=hps.bnb_config, device_map=hps.device_map) if hps.is_causal == True else AutoModelForSeq2SeqLM.from_pretrained(hps.model_name, quantization_config=hps.bnb_config, device_map=hps.device_map)
  elif hps.use_8bit:
    # Load the entire model on an available GPU
    hps.device_map = {"": hps.free_gpu_id if torch.cuda.is_available() else 'cpu'}
    # Quantization
    hps.bnb_config = BitsAndBytesConfig(
      load_in_8bit=hps.use_8bit
    )
    model = AutoModelForCausalLM.from_pretrained(hps.model_name, quantization_config=hps.bnb_config, device_map=hps.device_map) if hps.is_causal == True else AutoModelForSeq2SeqLM.from_pretrained(hps.model_name, quantization_config=hps.bnb_config, device_map=hps.device_map)
  else:
    model = AutoModelForCausalLM.from_pretrained(hps.model_name) if hps.is_causal == True else AutoModelForSeq2SeqLM.from_pretrained(hps.model_name)
    model = model.to(torch.bfloat16)

  model.config.use_cache = False
  model.config.pretraining_tp = 1
  return model
