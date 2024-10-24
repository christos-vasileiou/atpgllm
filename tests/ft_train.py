import torch
import argparse
import os
import sys
import itertools
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer, set_seed
from huggingface_hub import HfApi, login
# import wandb
from atpgllm import (
  parse_arguments, 
  load_raw_dataset, 
  dataset_formation, 
  dataset_formation_using_chat_template, 
  calculate_memory,
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
  get_patterns_criterion,
  dist,
  models_causal, 
  models_seq2seq,
  infer
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
  optimizer          = hps.get('optimizer', torch.optim.AdamW(model.parameters(), lr=hps.lr))
  scheduler          = hps.get('scheduler', None)
  device             = hps.get('device', torch.device(hps.free_gpu_id))
  criterion          = hps.get('criterion', nn.CrossEntropyLoss(ignore_index=hps.tokenizer.pad_token_id if hps.is_causal else -100).to(device))
  patterns_criterion = hps.get('patterns_criterion', get_patterns_criterion(hps))
  tokenizer          = hps.get('tokenizer', AutoTokenizer.from_pretrained(hps.model_name))
  is_causal          = hps.get('is_causal', True)
  # accuracy           = hps.get('accuracy', ATPGAccuracy(model, tokenizer))

  # lambda variable to penalize the loss of the patterns
  lambda_pat_loss = 1

  memory_consumption = False
  model_memory_consumption_per_gpu = 0
  data_memory_consumption_per_gpu = 0

  # If parallel training is used, we set the tqdm progress bar to the main process
  if is_main_process():
    pbar = tqdm(total=len(dataloader), desc=f"[{hps.local_rank}]: {desc}", disable=hps.disable_tqdm)
    data_iterator = iter(itertools.cycle(dataloader))
  else:
    data_iterator = iter(itertools.cycle(dataloader))

  learning_rates = []
  epoch_batch_losses = []
  pat_losses = []
  info = dict()
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
    
    # micro_batch_pat_loss = patterns_criterion(outputs.logits.transpose(2,1).to(torch.bfloat16), targets)
    # micro_batch_pat_loss /= hps.gradient_accumulation_steps

    batch_loss += micro_batch_loss.item()
    # batch_pat_loss += micro_batch_pat_loss.item()
    if training_loop:
      #NOTE: Penalty `lambda_pat_loss` is applied only during training loop. It is not recorded at the `losses` variable.
      # total_micro_batch_loss = micro_batch_loss + (lambda_pat_loss * micro_batch_pat_loss)
      total_micro_batch_loss = micro_batch_loss
      if torch.isnan(total_micro_batch_loss).any():
        raise ValueError(f"The gradients are vanished/exploded. The loss is {total_micro_batch_loss.item()}")
      
      # Calculate the gradients
      hps.accelerator.backward(total_micro_batch_loss) if hps.deepspeed_kernel else total_micro_batch_loss.backward()

      if (i+1) % hps.gradient_accumulation_steps == 0:
        if hps.new_tokens:
          zero_grad_old_embeddings(model, tunable_ids = list(tokenizer.added_tokens_decoder.keys()))

        # Update the weights
        optimizer.step()

        # schedule the learning rate based on the model validation loss
        if scheduler:
          scheduler.step() if hps.deepspeed_kernel else scheduler.step(batch_loss)
          learning_rates.append(scheduler.get_lr()[0])

      if memory_consumption:
        # Calculate the Memory Consumption per GPU
        memory_consumption_per_gpu = convert_bytes(torch.cuda.max_memory_allocated(device))
        memory_consumption = False
    
    info.update(dict(micro_batch_loss=micro_batch_loss.item()))
    if (i+1) % hps.gradient_accumulation_steps == 0:
      epoch_batch_losses.append(batch_loss)
      info.update(dict(epoch_estimate=np.mean(epoch_batch_losses), batch_loss=batch_loss))

    if is_main_process():
      pbar.update(1)
      pbar.set_postfix(info)
    
    if DEBUG and i>50:
      break

  if is_main_process():
    pbar.close()
  
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
  # for param in model.parameters():
  #   param.register_hook(lambda x: print(x))
  for epoch in range(1, epochs+1):
    if is_main_process():
      print(f"\nEpoch: [{epoch}/{epochs}]")
      with open("logs/losses.log", 'w' if epoch == 1 else 'a') as file:
        file.write(f"Epoch: [{epoch}/{epochs}]\n")

    model.train()
    train_loss, _learning_rates = epoch_loop(dataloader=training_loader, desc=f"Training...", model=model, hps=hps, training_loop=True, epoch=epoch)
    learning_rates.extend(_learning_rates)

    # Get train loss
    _train_loss = np.mean(train_loss)
    if is_main_process():
      print(f"Training loss: {_train_loss}")
      with open("logs/losses.log", 'a') as file:
        file.write(f"Training loss: {_train_loss}\n")
    
    # Save losses
    train_losses.append(_train_loss)
    batch_train_losses.extend(train_loss)
    
    val_loss = []
    model.eval()
    # Validation
    with torch.no_grad():
      val_loss, _ = epoch_loop(dataloader=validation_loader, desc=f"Validation...", model=model, hps=hps, training_loop=False)

    # Get eval loss
    _val_loss = np.mean(val_loss)
    if is_main_process():
      print(f"Validation loss: {_val_loss}")
      with open("logs/losses.log", 'a') as file:
        file.write(f"Validation loss: {_val_loss}\n\n")
    
    # schedule the learning rate based on the model validation loss
    scheduler.step() if hps.deepspeed_kernel else scheduler.step(_val_loss)

    # Save losses
    val_losses.append(_val_loss)

    if DEBUG:
      continue
    
    save_directory = models_causal[hps.save_model]
    # print examples of trained model and in-memory model.
    infer(ddp_model=model, 
          data_iterator=iter(itertools.cycle(validation_loader)), 
          tokenizer=hps.tokenizer, 
          stop_token="[/INST]", 
          model_max_length=hps.model_max_length,
          epoch=epoch,
          file_path='logs/generated_text_lora.md' if hps.lora else 'logs/generated_text.md')

    if is_main_process():
      hps.api.upload_folder(repo_id=save_directory, folder_path="logs")
    dist.barrier()

    # store model
    save_model(model, 
                save_directory, 
                push_to_hub=True, 
                save_embedding_layers=True, 
                hps=hps)

      
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
    print(f"Testing loss: {test_loss}")
    with open("logs/losses.log", 'a') as file:
      file.write(f"Testing loss: {test_loss}\n")
  

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
  # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
  train_losses, val_losses, batch_train_losses, learning_rates = train(model, training_loader, validation_loader, hps)

  # Evaluate the model performance
  evaluate(model, testing_loader, hps)

  if DEBUG:
    return 
  
  save_directory = models_causal[hps.save_model]
  
  # store model
  save_model(model,
              save_directory, 
              push_to_hub=True, 
              save_embedding_layers=True, 
              hps=hps)
  
  if is_main_process():
    # Plot training and validation plots
    filename = "logs/losses_plot.html" if hps.new_tokens == True else "logs/losses_plot_lora.html"
    plot_training_plots(train_losses, val_losses, batch_train_losses, learning_rates, filename=filename, parallel=hps.parallel)
    # upload logs
    hps.api.upload_folder(repo_id=save_directory, folder_path="logs/")
  dist.barrier()

def main():
  # Arguments
  parser = argparse.ArgumentParser(description='Training arguments parser')
  args   = parse_arguments(parser)
  
  # set a seed for proper synchronization
  set_seed(args.seed)

  # Initialize hyperparameters using custom AttrDict()
  hps = hyperparameters(args)

  # Create an api to communicate with huggingface hub
  hps.api = HfApi()

  # Initialize weights & bias environment for training recording
  # wandb.init(project=f"Supervised Fine-Tuning for {hps.model_name}", entity="chrivasileiou", config=hps)

  # Set the model name
  hps.model_name = models_causal[hps.model_name] if hps.is_causal == True else models_seq2seq[hps.model_name]

  # Initialize the training environment based on the GPU availability and parallelization
  initialize_training_environment(hps)

  # Model Loading
  model = load_model(hps)

  # Load tokenizer
  model = load_tokenizer(model, hps)

  # Load and Form the Dataset appropriately
  # raw_dataset = load_raw_dataset(hps.data_file)
  
  # Updated: load directly the dataset
  dataset = load_raw_dataset(hps.data_file)
  
  # NOTE: WORK for parallel=True
  # dataset = dataset_formation(raw_dataset, hps=hps)
  
  # NOTE: DOES NOT WORK for parallel=True
  # dataset = dataset_formation_using_chat_template(raw_dataset, hps=hps) 

  # Prepare objects for training and validation methods (optimizer, scheduler, dataloader, etc...)
  model, training_loader, validation_loader, testing_loader = prepare_objects_for_training(model, dataset, hps)

  # Train
  train_and_evaluate(model, training_loader, validation_loader, testing_loader, hps)

  # Ends weights & bias recording
  # wandb.finish()

if __name__ == '__main__':
  main()
