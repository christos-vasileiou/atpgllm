from transformers import AutoTokenizer
from glob import glob
from pprint import pprint
from tqdm.auto import tqdm
from datasets import load_dataset, load_from_disk
from atpgllm.llm.tokenizer import *
from atpgllm.llm.models import *
from atpgllm.llm.train import *
from atpgllm.utils import *
from atpgllm.llm.collate import *
import argparse
import pandas as pd
import torch.distributed as dist
import time
import sys
import warnings
warnings.filterwarnings('ignore')

parser                         = argparse.ArgumentParser(description='Training arguments parser')
args                           = parse_arguments(parser)
print(args)
batch_size                     = args.batch_size
epochs                         = args.epochs
lr                             = args.lr
dropout_p                      = args.dropout_p
model_checkpoint               = args.model_checkpoint
data_file                      = args.data_file
vocab_file                     = args.vocab_file
parallel                       = args.parallel
deepspeed_kernel               = args.deepspeed_kernel
fp16                           = args.fp16
tokenizer                      = args.tokenizer
zero_stage                     = args.zero_stage
load_checkpoint                = args.load_checkpoint
collate                        = args.atpg_collate
train_type                     = sys.argv[0].split('_')[1]
world_size                     = int(len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))) if deepspeed_kernel == True else None
free_gpu_id                    = get_free_gpu()

if __name__ == '__main__':
  if deepspeed_kernel == True:
    deepspeed_config = {
      "train_batch_size": batch_size,
      "train_micro_batch_size_per_gpu": batch_size//world_size,
      "steps_per_print": 10,
      "gradient_accumulation_steps": 1,
      "optimizer": {
        "type": "Adam",
        "params": {
          "lr": lr,
          "betas": [0.9, 0.999],
          "eps": 1e-8,
          "weight_decay": 3e-7,
          "torch_adam": False, # Use torch’s implementation of adam instead of DeepSpeed's fused adam implementation
          "adam_w_mode": True  # Apply L2 regularization (also known as AdamW)
        },
      },
      "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
      },
      "zero_optimization": {
        "stage": zero_stage,
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "offload_optimizer": {
          "device": "cpu",
          "pin_memory": True
        },
        "offload_param": {
          "device": "cpu",
          "pin_memory": True
        }
      }
    }   
  else:
    deepspeed_config = None
  print(f"Load Tokenizer...")
  tokenizer = ATPGTokenizer(pd.read_csv(vocab_file, sep=',', index_col=0)) if tokenizer == 'custom' else AutoTokenizer.from_pretrained(models[model_checkpoint]) 
  print(data_file)
  if data_file[-3:] != 'csv':
    tokenized_datasets = load_from_disk(data_file, keep_in_memory=True)
    max_freq           = 10 # is hardcoded......................... 
  else:
    print(f"Load Dataset...")
    raw_dataset                            = load_dataset(data_files=data_file, keep_in_memory=True).remove_columns('Unnamed: 0')
    max_num_of_patterns_per_circuit        = max(set([len(patterns.split('\n')) for patterns in raw_dataset['train']['patterns']]))

    print(f"Data Initial Preprocessing...")
    raw_dataset, max_freq = initial_data_preprocessing(raw_dataset)
    id2possible_labels, possible_labels2id = get_patterns_info_mapping(max_freq)
    
    if collate:
      atpg_collate = ATPGCollate(tokenizer, max_freq, possible_labels2id, max_num_of_patterns_per_circuit)
    else:
      print(f"Data Tokenization...")
      atpg_collate       = None
      num_proc           = (len(raw_dataset['train'])//1_000) if len(raw_dataset['train']) % 1_000 == 0 else (len(raw_dataset['train'])//1_000)+1
      num_proc           = 32 if num_proc>32 else num_proc
      tokenized_datasets = raw_dataset.map(
        tokenize_less_modelmaxlen_fn,
        batched=True,
        num_proc=num_proc,
        remove_columns=raw_dataset['train'].column_names,
        fn_kwargs={'tokenizer': tokenizer},
        keep_in_memory=True,
      )
      del raw_dataset
      print(f"{tokenized_datasets}\n\n")
    
      print(f"Labels Alignment...")
      num_proc = (len(tokenized_datasets['train'])//1_000) if len(tokenized_datasets['train']) % 1_000 == 0 else (len(tokenized_datasets['train'])//1_000)+1
      num_proc = 32 if num_proc>32 else num_proc
      tokenized_datasets = tokenized_datasets.map(
        align_labels_into_matrix,
        batched=True,
        num_proc=num_proc,
        fn_kwargs={
            "max_freq": max_freq,
            "max_num_of_patterns_per_circuit": max_num_of_patterns_per_circuit,
            "tokenizer": tokenizer,
            "possible_labels2id": possible_labels2id},
        keep_in_memory=True,
      )
      # Create a folder and stores inside the data
      #tokenized_datasets.save_to_disk(os.path.join(data_path, data_file)[:-4])
      print(f"{tokenized_datasets}\n\n")

  model = BERT(model_checkpoint=model_checkpoint, dropout_p=dropout_p, patterns_len=max_freq)

  train_kwargs = {
    'epochs': epochs,
    'model': model,
    'dataset': tokenized_datasets if atpg_collate is None else raw_dataset,
    'lr': lr,
    'batch_size': batch_size,
    'labels_filtering': True,
    'reduction': 'mean',
    'parallel': parallel,
    'deepspeed_config': deepspeed_config if parallel == True and deepspeed_kernel == True else None,
    'fp16': fp16,
    'stored_checkpoint': load_checkpoint,
    'atpg_collate': atpg_collate,
    'pos_weight': None, #get_pos_weight(possible_labels2id, raw_dataset, max_freq)
    'train_type': train_type,
    'free_gpu_id': free_gpu_id,
  }
  
  print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")
  if parallel==True and deepspeed_kernel==False:
    dist.init_process_group(backend='nccl')
  if train_type != 'multilabel':
    raise ValueError("model_type: should be set to 'multilabel'")
  model_to_save, optimizer, epochs, local_rank, train_loss_to_print, val_loss_to_print = train_cls_task(**train_kwargs)
  if free_gpu_id == local_rank:
    model_to_save = model_to_save.cpu()
    save_checkpoint(model=model_to_save, optimizer=optimizer, epoch=epochs, local_rank=local_rank, train_loss_to_print=train_loss_to_print, val_loss_to_print=val_loss_to_print) 


