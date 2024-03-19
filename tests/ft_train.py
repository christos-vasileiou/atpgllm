import torch
import argparse
import numpy as np
import os
import sys
from pprint import pprint
from tqdm.auto import tqdm
from atpgllm import (
  parse_arguments, 
  load_raw_dataset, 
  dataset_formation, 
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
  models_causal, 
  models_seq2seq
)

PROFILING = False
DEBUG = False

def epoch_loop(dataloader, desc, model, hps, training_loop:bool = True, losses:list = []) -> list:
  """
  A function that loops over the dataloader for one epoch.

  Args:
      dataloader (DataLoader): The dataloader to iterate over.
      desc (str): A description of the loop.
      model: The model to train.
      hps (AttrDict): The hyperparameters for the training.
      accelerator: The DeepSpeed accelerator object.
      training_loop (bool, optional): Whether to run the training loop. Defaults to True.
      losses (list, optional): A list to store the losses. Defaults to [].

  Returns:
      list: The list of losses.
  """
  global PROFILING
  global DEBUG

  # Use the hyperparemeters for the training
  optimizer          = hps.optimizer
  criterion          = hps.criterion
  patterns_criterion = hps.patterns_criterion
  tokenizer          = hps.tokenizer
  device             = hps.device
  is_causal          = hps.is_causal
  
  # lambda variable to penalize the loss of the patterns
  lambda_patterns_loss = 1

  memory_consumption = True
  model_memory_consumption_per_gpu = 0
  data_memory_consumption_per_gpu = 0

  # If parallel training is used, we set the tqdm progress bar to the main process
  if hps.parallel:
    if is_main_process():
      t = tqdm(dataloader, file=sys.stdout, desc=desc, disable=hps.disable_tqdm)
    else:
      t = dataloader
  else:
    t = tqdm(dataloader, file=sys.stdout, desc=desc, disable=hps.disable_tqdm)

  # Main loop
  for data in t:
    # IDs and Attention Mask
    ids  = data['input_ids'].to(device)
    mask = data['attention_mask'].to(device)

    # Get targets
    targets = get_targets(data, tokenizer, is_causal, device)

    # Get decoder input_ids and decoder attention mask in the case where you have a seq2seq model
    dec_input, dec_mask = get_dec_ids_and_mask(targets, tokenizer, is_causal, device)
  
    # Forward Pass
    outputs = model(input_ids=ids, attention_mask=mask) if hps.is_causal else model(input_ids=ids, attention_mask=mask, decoder_input_ids=dec_input, decoder_attention_mask=dec_mask)

    # loss
    loss          = criterion(outputs.logits.transpose(2, 1).to(torch.bfloat16), targets)
    patterns_loss = patterns_criterion(outputs.logits.transpose(2,1).to(torch.bfloat16), targets)
    
    if training_loop:
      total_loss    = (loss + (lambda_patterns_loss * patterns_loss)) / hps.gradient_accumulation_steps if hps.deepspeed_kernel else loss + (lambda_patterns_loss * patterns_loss)
      if torch.isnan(total_loss).any():
        raise ValueError(f"The gradients are vanished/exploded. The loss is {total_loss.item()}")

      # Calculate the gradients
      hps.accelerator.backward(total_loss) if hps.deepspeed_kernel else total_loss.backward()

      # Update the weights
      optimizer.step()
      
      if memory_consumption:
        # Calculate the Memory Consumption per GPU
        model_memory_consumption_per_gpu = calculate_memory(model=model, optimizer=optimizer)
        data_memory_consumption_per_gpu  = calculate_memory(ids=ids, mask=mask, targets=targets)
        memory_consumption = False
      
      # zero the parameter gradients for the next loop
      optimizer.zero_grad()

    total_loss = (loss + patterns_loss) / hps.gradient_accumulation_steps if hps.deepspeed_kernel else loss + patterns_loss
    losses.append(total_loss.item())

    # Print out info
    if hps.parallel:
      if is_main_process():
        t.set_postfix(loss=loss.item(), patterns_loss=patterns_loss.item(), model_memory_consumption_per_gpu=model_memory_consumption_per_gpu, data_memory_consumption_per_gpu=data_memory_consumption_per_gpu)
    else:
      t.set_postfix(loss=loss.item(), patterns_loss=patterns_loss.item(), model_memory_consumption_per_gpu=model_memory_consumption_per_gpu, data_memory_consumption_per_gpu=data_memory_consumption_per_gpu)

    if DEBUG:
      break
  
  if hps.parallel:
    if is_main_process():
      t.close()
  else:
    t.close()

  return losses


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
  global PROFILING
  global DEBUG

  # Use the hyperparemeters for the training
  scheduler          = hps.scheduler
  epochs             = hps.epochs

  train_losses = []
  val_losses   = []
  # for param in model.parameters():
  #   param.register_hook(lambda x: print(x))
  for epoch in range(1, epochs+1):
    if hps.parallel:
      if is_main_process():
        print(f"\nEpoch: [{epoch}/{epochs}]")
    else:
      print(f"\nEpoch: [{epoch}/{epochs}]")
    model.train()
    train_loss = epoch_loop(dataloader=training_loader, desc=f"Training...", model=model, hps=hps, training_loop=True)    
  
    # Get train loss
    train_loss = np.mean(train_loss)
    if hps.parallel:
      if is_main_process():
        print(f"Training loss: {train_loss}")
    else:
      print(f"Training loss: {train_loss}")
    
    # Save losses
    train_losses.append(train_loss)
    
    val_loss = []
    model.eval()
    # Validation
    with torch.no_grad():
      val_loss = epoch_loop(dataloader=validation_loader, desc=f"Validation...", model=model, hps=hps, training_loop=False)

    # Get eval loss
    val_loss = np.mean(val_loss)
    if hps.parallel:
      if is_main_process():
        print(f"Validation loss: {val_loss}")
    else:
      print(f"Validation loss: {val_loss}")
    
    # schedule the learning rate based on the model validation loss
    scheduler.step() if hps.deepspeed_kernel else scheduler.step(val_loss)

    # Save losses
    val_losses.append(val_loss)

    # store model
    if not DEBUG:
      save_model(model, 
                 'LlamaModelForCausalLM-ATPG' if hps.new_tokens == True else 'LlamaModelForCausalLM-ATPG-LoRA', 
                 push_to_hub=True, 
                 save_embedding_layers=True, 
                 hps=hps)

    if DEBUG:
      break
  return train_losses, val_losses


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
    test_loss = epoch_loop(dataloader=testing_loader, desc=f"Testing...", model=model, hps=hps, training_loop=False)
      
  # Get eval loss
  test_loss = np.mean(test_loss)
  if hps.parallel:
    if is_main_process():
      print(f"Testing loss: {test_loss}")
  else:
    print(f"Testing loss: {test_loss}")
  


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
  global PROFILING
  global DEBUG

  # Train the model
  # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
  train_losses, val_losses = train(model, training_loader, validation_loader, hps)

  # Evaluate the model performance
  evaluate(model, testing_loader, hps)

  if not DEBUG:
    # store model
    save_model(model, 
               'LlamaModelForCausalLM-ATPG' if hps.new_tokens == True else 'LlamaModelForCausalLM-ATPG-LoRA', 
               push_to_hub=True, 
               save_embedding_layers=True, 
               hps=hps)
  
    # Plot training and validation plots
    plot_training_plots(train_losses, val_losses, filename="losses_plot.html" if hps.new_tokens == True else "losses_plot_lora.html")


if __name__ == '__main__':
  from huggingface_hub import login
  # import wandb
  #login(os.environ['HF_TOKEN'], add_to_git_credential=True)

  # Arguments
  parser = argparse.ArgumentParser(description='Training arguments parser')
  args   = parse_arguments(parser)
  hps    = hyperparameters(args)

  # Initialize weights & bias environment for training recording
  # wandb.init(project=f"Supervised Fine-Tuning for {hps.model_name}", entity="chrivasileiou", config=hps)

  # Set the model name
  hps.model_name = models_causal[hps.model_name] if hps.is_causal == True else models_seq2seq[hps.model_name]
  
  # Initialize the training environment based on the GPU availability and parallelization
  initialize_training_environment(hps)

  # Model Loading
  model = load_model(hps)
  hps.info += str(type(model)) + '\n'
  
  # Load tokenizer
  model, tokenizer = load_tokenizer(model, hps)
  hps.tokenizer = tokenizer

  # Load and Form the Dataset appropriately
  raw_dataset = load_raw_dataset(hps.data_file)
  dataset = dataset_formation(raw_dataset, is_causal=hps.is_causal)

  # Prepare objects for training and validation methods (optimizer, scheduler, dataloader, etc...)
  model, training_loader, validation_loader, testing_loader = prepare_objects_for_training(model, dataset, hps)

  # Train
  train_and_evaluate(model, training_loader, validation_loader, testing_loader, hps)

  # Ends weights & bias recording
  # wandb.finish()
