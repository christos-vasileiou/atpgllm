import torch
import argparse
import os
import sys
import json
import itertools
import torch.nn as nn
import numpy as np
import pandas as pd
from collections import OrderedDict
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from transformers import AutoTokenizer, set_seed
from huggingface_hub import login
import wandb
from atpgllm import (
  parse_arguments, 
  load_raw_dataset, 
  hyperparameters,
  plot_training_plots,
  load_model,
  load_tokenizer,
  save_model, 
  get_dec_ids_and_mask, 
  get_targets, 
  convert_bytes,
  is_main_process,
  initialize_training_environment,
  prepare_objects_for_training,
  zero_grad_old_embeddings,
  dist,
  models_causal, 
  infer,
  compute_bleu,
  compute_rouge, 
  compute_repetition_rate, 
  compute_distinct_n, 
  compute_fault_coverage,
)

import warnings
warnings.filterwarnings('ignore')

DEBUG = False

def epoch_loop(dataloader, desc, model, hps, training_loop:bool = True, epoch=1) -> list:
  """
  A function that loops over the dataloader for one epoch.

  Args:
      dataloader (DataLoader): The dataloader to iterate over.
      desc (str): A description of the loop.
      model: The model to train.
      hps (AttrDict): The hyperparameters for the training.
      accelerator: The DeepSpeed accelerator object.
      training_loop (bool, optional): Whether to run the training loop. Defaults to True.

  Returns:
      list: The list of losses.
  """
  global DEBUG

  # Use the hyperparemeters for the training
  optimizer = hps.get('optimizer', torch.optim.AdamW(model.parameters(), lr=hps.lr))
  scheduler = hps.get('scheduler', None)
  device    = hps.get('device', torch.device(hps.free_gpu_id))
  criterion = hps.get('criterion', nn.CrossEntropyLoss(ignore_index=hps.tokenizer.pad_token_id if hps.is_causal else -100).to(device))
  tokenizer = hps.get('tokenizer', AutoTokenizer.from_pretrained(hps.model_name))
  is_causal = hps.get('is_causal', True)

  memory_consumption = False

  # get the dataloader iterator
  if is_main_process():
    # If parallel training is used, we set the tqdm progress bar to the main process
    pbar = tqdm(total=len(dataloader), desc=f"[{hps.local_rank}]: {desc}", disable=hps.disable_tqdm)
  data_iterator = iter(itertools.cycle(dataloader))

  if not training_loop and is_main_process():
    stop_token = tokenizer.encode("[/INST]", return_tensors='pt')[:, 1:] # return BxT: where B is 1, and T is <s>+token_ids. Get rid of <s> with '1:'
    windows_size = stop_token.size(1)

  metrics_df = pd.DataFrame([{}])
  learning_rates = []
  epoch_batch_losses = []
  pat_losses = []
  predictions = []
  references = []
  info = dict()
  metrics = dict()
  generated_texts_for_metrics = []
  target_texts_for_metrics = []
  netlists = []
  # Main loop
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

    # Get decoder input_ids and decoder attention mask in the case where you have a seq2seq model
    dec_input, dec_mask = get_dec_ids_and_mask(targets, tokenizer, is_causal, device)

    # Forward Pass
    outputs = model(input_ids=ids, attention_mask=mask) if hps.is_causal else model(input_ids=ids, attention_mask=mask, decoder_input_ids=dec_input, decoder_attention_mask=dec_mask)

    # loss
    micro_batch_loss = criterion(outputs.logits.transpose(2, 1).to(torch.bfloat16), targets)    
    micro_batch_loss /= hps.gradient_accumulation_steps
    
    batch_loss += micro_batch_loss.item()
    if training_loop:
      total_micro_batch_loss = micro_batch_loss
      if torch.isnan(total_micro_batch_loss).any():
        raise ValueError(f"The gradients are vanished/exploded. The loss is {total_micro_batch_loss.item()}")
      
      # Calculate the gradients
      hps.accelerator.backward(total_micro_batch_loss) if hps.deepspeed_kernel else total_micro_batch_loss.backward()

      # Gather outputs and targets 
      if is_main_process():
        # Make sure both will be loaded on CPU RAM
        # Generate the token ids
        _, generated_ids = torch.topk(outputs.logits.detach().cpu(), k=1)
        # Convert generated ids to text
        generated_texts = tokenizer.batch_decode(generated_ids.squeeze(-1), skip_special_tokens=True)
        # Convert target ids to text
        target_texts = tokenizer.batch_decode(targets.cpu(), skip_special_tokens=True)
        # Collect texts
        generated_texts_for_metrics.extend(generated_texts)
        target_texts_for_metrics.extend(target_texts)
        netlists.extend(data["netlist"])
      
      # gradient accumulation completes the batch size 
      if (i+1) % hps.gradient_accumulation_steps == 0:
        if hps.new_tokens:
          zero_grad_old_embeddings(model, tunable_ids = list(tokenizer.added_tokens_decoder.keys()))

        # Update the weights
        optimizer.step()

        # schedule the learning rate based on the model validation loss
        if scheduler:
          scheduler.step() if hps.deepspeed_kernel else scheduler.step(batch_loss)
          learning_rates.append(scheduler.get_lr()[0])
        
        if is_main_process():
          # Collect metrics
          # metrics.update(compute_bleu(references=target_texts_for_metrics, completions=generated_texts_for_metrics))
          # metrics.update(compute_rouge(references=target_texts_for_metrics, completions=generated_texts_for_metrics))
          # metrics.update(compute_distinct_n(completions=generated_texts_for_metrics, n=3))
          # metrics.update(compute_repetition_rate(completions=generated_texts_for_metrics))
          # metrics.update(compute_fault_coverage(completions=generated_texts_for_metrics, netlists=netlists))
          info.update(dict(micro_batch_loss=micro_batch_loss.item()))
          info.update(metrics)
          generated_texts_for_metrics = []
          target_texts_for_metrics = []
          netlists = []
        dist.barrier()

      if memory_consumption:
        # Calculate the Memory Consumption per GPU
        memory_consumption_per_gpu = convert_bytes(torch.cuda.max_memory_allocated(device))
        memory_consumption = False

    else:
      # if is_main_process():
      #   # Take token ids up to the "[/INST]"
      #   ids = ids.cpu()
      #   windows = ids.unfold(1, windows_size, 1)
      #   matches = (windows == stop_token).all(dim=2)
      #   # Find the first occurrence in each sequence
      #   # Initialize indices with sequence length for sequence without a match
      #   indices = torch.full((ids.size(0),), ids.size(1), dtype=torch.long)

      #   # For sequence where a match is found, update the index
      #   for j in range(ids.size(0)):
      #     match_indices = torch.nonzero(matches[j], as_tuple=True)[0]
      #     if match_indices.numel() > 0:
      #       indices[j] = match_indices[0].item()
      #   user_prompt_ids = [ids[j, :idx+stop_token.size(1)] for j, idx in enumerate(indices)]

      #   # Create the attention mask
      #   attention_mask_list = [torch.ones_like(ids) for ids in user_prompt_ids]

      #   # Pad sequences to the maximum length
      #   input_ids = pad_sequence(user_prompt_ids, batch_first=True, padding_value=tokenizer.eos_token_id).to(model.module.device)
      #   attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0).to(model.module.device)

      #   # Generate completion
      #   generated_ids = model.module.generate(input_ids=input_ids, attention_mask=attention_mask, num_return_sequences=1, top_k=1, max_length=1024)
      #   # Convert generated ids to text
      #   generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
      #   # Convert target ids to text
      #   references = tokenizer.batch_decode(targets, skip_special_tokens=True)
      #   # Update the metrics dictionary 
      #   metrics.update(compute_bleu(references=references, completions=generated_texts))
      #   metrics.update(compute_rouge(references=references, completions=generated_texts))
      #   metrics.update(compute_distinct_n(completions=generated_texts, n=3))
      #   metrics.update(compute_repetition_rate(completions=generated_texts))
      #   metrics.update(compute_fault_coverage(completions=generated_texts, netlists=data["netlist"]))
        
      #   metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics])]).dropna().reset_index(drop=True)
      #   # Update the info dictionary 
      #   info.update(metrics)
      #   # Update the progress bar 
      #   pbar.set_postfix(info)
      dist.barrier()

    if (i+1) % hps.gradient_accumulation_steps == 0:
      epoch_batch_losses.append(batch_loss)
      info.update(dict(epoch_est=np.mean(epoch_batch_losses), b_loss=batch_loss))

    if is_main_process():
      pbar.update(1)
      pbar.set_postfix(info)
    
    if DEBUG == True and i>10 and training_loop == True:
      break

    if training_loop == False and (i+1) % hps.gradient_accumulation_steps == 0:
      break
  
  gathered_dfs = [None for _ in range(hps.world_size)]
  dist.all_gather_object(gathered_dfs, metrics_df)

  if is_main_process():
    gathered_dfs = pd.concat(gathered_dfs, ignore_index=True)
    avg_metrics_gpus = gathered_dfs.mean(0)
    pbar.set_postfix(info)
    if not training_loop and DEBUG == False:
      # Flush the metrics into a logfile 
      with open("logs/metrics.json", mode='w' if epoch == 1 else 'a', encoding='utf-8') as logfile:
        json.dump(avg_metrics_gpus.to_dict(into=OrderedDict), logfile)
        logfile.write("\n") # Ensure each JSON object is on a separate line
        logfile.flush()     # Flush the buffer to ensure data is written to disk
        os.fsync(logfile.fileno())  # Force write to disk
      
    # Close progress bar 
    pbar.close()
  dist.barrier()

  torch.cuda.empty_cache()

  return epoch_batch_losses, learning_rates


def train(model, training_loader, validation_loader, hps) -> list:
  """
  Train the model using the given hyperparameters.

  Args:
      model (obj): The model to be trained.
      logger (obj): The logger to log the training results.
      training_loader (obj): The DataLoader for the training set.
      validation_loader (obj): The DataLoader for the validation set.
      hps (obj): The Hyperparameters object containing the configuration for the training.

  Returns:
      list: The list of training losses.
      list: The list of validation losses.
  """
  global DEBUG

  # Use the hyperparemeters for the training
  scheduler          = hps.get('scheduler', None)
  epochs             = hps.get('epochs', 1)

  learning_rates = []
  batch_train_losses = []
  train_losses = []
  val_losses   = []
  for epoch in range(1, epochs+1):
    if is_main_process():
      hps.logger.info(f"\nEpoch: [{epoch}/{epochs}]")
      if DEBUG == False:
        with open("logs/losses.log", 'w' if epoch == 1 else 'a') as file:
          file.write(f"Epoch: [{epoch}/{epochs}]\n")

    model.train()
    train_loss, _learning_rates = epoch_loop(dataloader=training_loader, desc=f"Training...", model=model, hps=hps, training_loop=True, epoch=epoch)
    learning_rates.extend(_learning_rates)

    # Get train loss
    _train_loss = np.mean(train_loss)
    if is_main_process():
      hps.logger.info(f"Training loss: {_train_loss}")
    # Save losses
    train_losses.append(_train_loss)
    batch_train_losses.extend(train_loss)
    
    val_loss = []
    model.eval()
    # Validation
    with torch.no_grad():
      val_loss, _ = epoch_loop(dataloader=validation_loader, desc=f"Validation...", model=model, hps=hps, training_loop=False, epoch=epoch)

    # Get eval loss
    _val_loss = np.mean(val_loss)
    if is_main_process():
      hps.logger.info(f"Validation loss: {_val_loss}")
    # schedule the learning rate based on the model validation loss
    scheduler.step() if hps.deepspeed_kernel else scheduler.step(_val_loss)

    # Save losses
    val_losses.append(_val_loss)

    if DEBUG:
      continue
    
    hps.log_dir = "logs"
    hps.save_in_repo = models_causal[hps.save_model]
    hps.filename = "generated_text_lora.md" if hps.lora else "generated_text.md"
    hps.file_path = os.path.join(hps.log_dir, hps.filename)
    # print examples of trained model and in-memory model.
    infer(ddp_model=model, dataloader=validation_loader, tokenizer=hps.tokenizer, model_max_length=hps.model_max_length, epoch=epoch, file_path=hps.file_path, parallel=hps.parallel)

    if is_main_process():
      hps.api.upload_file(repo_id=hps.save_in_repo, path_or_fileobj=hps.file_path, path_in_repo=hps.filename)

    # store model
    save_model(model, hps.save_in_repo, push_to_hub=True, save_embedding_layers=True, hps=hps)

      
  return train_losses, val_losses, batch_train_losses, learning_rates


def evaluate(model, testing_loader, hps) -> None:
  """
  Evaluate the model performance on the testing set.

  Args:
      model (obj): The model to be evaluated.
      logger (obj): The logger to log the evaluation results.
      testing_loader (obj): The DataLoader for the testing set.
      hps (obj): The Hyperparameters object containing the configuration for the training.

  Returns:
      float: The average loss over the testing set.
  """
  model.eval()
  with torch.no_grad():
    test_loss, _ = epoch_loop(dataloader=testing_loader, desc=f"Testing...", model=model, hps=hps, training_loop=False)

  # Get eval loss
  test_loss = np.mean(test_loss)
  if is_main_process():
    hps.logger.info(f"Testing loss: {test_loss}")

def train_and_evaluate(model, training_loader, validation_loader, testing_loader, hps) -> None:
  """
  Trains and evaluates a model using the given data loaders and hyperparameters.

  Args:
      model (obj): The model to be trained and evaluated.
      training_loader (obj): The DataLoader for the training set.
      validation_loader (obj): The DataLoader for the validation set.
      testing_loader (obj): The DataLoader for the testing set.
      hps (obj): The Hyperparameters object containing the configuration for the training and evaluation.

  Returns:
      None
  """
  global DEBUG

  # Train the model
  train_losses, val_losses, batch_train_losses, learning_rates = train(model, training_loader, validation_loader, hps)

  # Evaluate the model performance
  evaluate(model, testing_loader, hps)

  if DEBUG:
    return 

  save_in_repo = models_causal[hps.save_model]

  # Since the model is uploaded after validation
  # it doesn't make sense to update it again

  # # store model
  # save_model(model, save_in_repo, push_to_hub=True, save_embedding_layers=True, hps=hps)
  
  if is_main_process():
    # Plot training and validation plots
    filename = "losses_plot.html" if hps.new_tokens == True else "losses_plot_lora.html"
    filepath = f"logs/{filename}"
    plot_training_plots(train_losses, val_losses, batch_train_losses, learning_rates, filename=filepath, parallel=hps.parallel)
    # upload logs
    hps.api.upload_file(repo_id=save_in_repo, path_or_fileobj=filepath, path_in_repo=filename)
  dist.barrier()

def main():
  # Arguments
  parser = argparse.ArgumentParser(description='Training arguments parser')

  # Initialize hyperparameters using custom AttrDict()
  hps = hyperparameters(parse_arguments(parser))
  
  # set a seed for proper synchronization
  set_seed(hps.seed)

  # Set debug mode
  hps.debug = DEBUG

  # Initialize weights & bias environment for training recording
  # wandb.init(project=f"Supervised Fine-Tuning for {hps.model_name}", entity="chrivasileiou", config=hps)

  # Set the model name
  hps.model_name = models_causal[hps.model_name]

  # Initialize the training environment based on the GPU availability and parallelization
  initialize_training_environment(hps)

  # Model Loading
  model = load_model(hps)

  # Load tokenizer
  model = load_tokenizer(model, hps)

  # load directly the dataset
  dataset = load_raw_dataset(hps.data_file)
  
  # Prepare objects for training and validation methods (optimizer, scheduler, dataloader, etc...)
  model, training_loader, validation_loader, testing_loader = prepare_objects_for_training(model, dataset, hps)

  try:
    # Train
    train_and_evaluate(model, training_loader, validation_loader, testing_loader, hps)
  finally:
    if is_main_process():
      from logging.handlers import RotatingFileHandler
      save_in_repo = models_causal[hps.save_model]
      # store model
      # if hps.debug == False:
      #   save_model(model, save_in_repo, push_to_hub=True, save_embedding_layers=True, hps=hps)

      # Save logs
      for handler in hps.logger.handlers:
        if isinstance(handler, RotatingFileHandler):
          handler.doRollover()
      # upload logs
      hps.api.upload_file(repo_id=save_in_repo, path_or_fileobj=r"logs/info.log.1", path_in_repo="info.log")
      hps.api.upload_file(repo_id=save_in_repo, path_or_fileobj=r"logs/losses.log", path_in_repo="losses.log")
      hps.api.upload_file(repo_id=save_in_repo, path_or_fileobj=r"logs/metrics.json", path_in_repo="metrics.json")
      # hps.api.upload_file(repo_id=save_in_repo, path_or_fileobj=r"logs/debug.log.1", path_in_repo="logs")

  # Ends weights & bias recording
  # wandb.finish()

if __name__ == '__main__':
  main()
