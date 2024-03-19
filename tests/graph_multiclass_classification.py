from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from atpgllm import (
  models,
  deepspeed_config,
  parse_arguments,
  hyperparameters,
  GraphTokenizer,
  GraphCollate,
  initial_data_preprocessing,
  get_patterns_info_mapping,
  tokenize_less_modelmaxlen_fn,
  train_graph,
  align_labels_into_matrix,
  GAT,
  GPT,
  T5,
  MyModel,
  save_checkpoint,
)
import argparse
import pandas as pd
import torch.nn.functional as F
import torch.distributed as dist
import time
import os
import warnings
warnings.filterwarnings('ignore')

parser                         = argparse.ArgumentParser(description='Training arguments parser')
args                           = parse_arguments(parser)
hps                            = hyperparameters(args)
print(hps)
batch_size                     = hps.batch_size
epochs                         = hps.epochs
lr                             = hps.lr
dropout_p                      = hps.dropout
model_checkpoint               = hps.model_name
data_file                      = hps.data_file
vocab_file                     = hps.vocab_file
parallel                       = hps.parallel
deepspeed_kernel               = hps.deepspeed_kernel
fp16                           = hps.fp16
tokenizer                      = hps.tokenizer
load_checkpoint                = hps.load_ckpt
collate                        = hps.collate
world_size                     = hps.world_size
free_gpu_id                    = hps.free_gpu_id
train_type                     = 'multiclass' 

if __name__ == '__main__':
  if deepspeed_kernel == False:
    deepspeed_config = None
  print(f"Load Tokenizer...")
  tokenizer = GraphTokenizer(pd.read_csv(vocab_file, sep=',', index_col=0)) if tokenizer == 'custom' else AutoTokenizer.from_pretrained(models[model_checkpoint]) 
  print(data_file)
  if data_file[-3:] != 'csv':
    tokenized_datasets = load_from_disk(data_file, keep_in_memory=True)
    max_freq           = 5 # is hardcoded.........................
  elif data_file[-3:] == 'csv':
    print(f"Load Dataset...")
    raw_dataset                            = load_dataset(data_files=data_file, keep_in_memory=True).remove_columns('Unnamed: 0')
    max_num_of_patterns_per_circuit        = max(set([len(patterns.split('\n')) for patterns in raw_dataset['train']['patterns']]))

    print(f"Data Initial Preprocessing...")
    raw_dataset, max_freq = initial_data_preprocessing(raw_dataset)
    id2possible_labels, possible_labels2id = get_patterns_info_mapping(max_freq)
    
    if collate:
      print("GraphCollate loading...")
      graph_collate = GraphCollate(tokenizer, max_freq, possible_labels2id, max_num_of_patterns_per_circuit)
    else:
      print(f"Data Tokenization...")
      graph_collate       = None
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

  model = GAT(model_checkpoint=model_checkpoint, dropout_p=dropout_p, patterns_len=max_freq)
  model = GPT(model_checkpoint=model_checkpoint, dropout_p=dropout_p, patterns_len=max_freq)
  model = T5(model_checkpoint=model_checkpoint, dropout_p=dropout_p, patterns_len=max_freq)
  model = MyModel(patterns_len=max_freq)

  world_size = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
  
  train_kwargs = {
    'epochs': epochs,
    'model': model,
    'dataset': tokenized_datasets if graph_collate is None else raw_dataset,
    'lr': lr,
    'batch_size': batch_size,
    'labels_filtering': False,
    'reduction': 'mean',
    'parallel': parallel,
    'deepspeed_config': deepspeed_config if parallel == True and deepspeed_kernel == True else None,
    'fp16': fp16,
    'stored_checkpoint': load_checkpoint,
    'graph_collate': graph_collate,
    'pos_weight': None, #get_pos_weight(possible_labels2id, raw_dataset, max_freq)
    'train_type': train_type,
    'free_gpu_id': free_gpu_id,
    'args' : args,
    #'world_size': world_size
  }
  
  print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")
  if parallel==True and deepspeed_kernel==False:
    dist.init_process_group(backend='nccl')
  model_to_save, optimizer, epochs, local_rank, train_loss_to_print, val_loss_to_print = train_graph(**train_kwargs)
  #print(free_gpu_id, local_rank)
  if local_rank == 0:
    model_to_save = model_to_save.cpu()
    save_checkpoint(model=model_to_save, optimizer=optimizer, epoch=epochs, local_rank=local_rank, train_loss_to_print=train_loss_to_print, val_loss_to_print=val_loss_to_print)
  else:
    print('model was not saved!')


