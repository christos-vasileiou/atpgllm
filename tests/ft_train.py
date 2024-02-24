import torch
# import deepspeed
import argparse
import numpy as np
from pprint import pprint
from peft import get_peft_model, LoraConfig, TaskType
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from atpgllm.utils import (
  set_training_environment, 
  parse_arguments, 
  load_raw_dataset, 
  dataset_formation, 
  generate_all_possible_binary_combination, 
  convert_bytes, 
  model_size_in_bytes, 
  calculate_memory,
  hyperparameters,
  plot_training_plots,
  load_model,
)
from atpgllm.llm.collate import MyCollate
from atpgllm.llm.fine_tuning import get_dec_ids_and_mask, get_targets
from atpgllm.llm.tokenizer import tokenize_fn

def train(hps):
  # Use the hyperparemeters for the training
  optimizer          = hps.optimizer
  criterion          = hps.criterion
  patterns_criterion = hps.patterns_criterion
  epochs             = hps.epochs
  tokenizer          = hps.tokenizer
  device             = hps.device
  is_causal          = hps.is_causal

  train_losses = []
  val_losses   = []
  for epoch in range(1, epochs+1):
    train_loss = []
    model.train()
    print(f"Epoch: [{epoch}/{epochs}]")
    # Training
    with tqdm(training_loader, desc=f"Training...") as t:
      for data in t:
        ids  = data['input_ids'].to(device)
        mask = data['attention_mask'].to(device)
        
        # Get targets
        targets = get_targets(data, tokenizer, is_causal, device)
        # get decoder input_ids and decoder attention mask in the case where you have a causal model
        dec_input, dec_mask = get_dec_ids_and_mask(targets, tokenizer, is_causal, device)

        # zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward Pass
        outputs = model(ids, mask) if hps.is_causal else model(ids, mask, dec_input, dec_mask)

        # loss
        loss          = criterion(outputs.logits.transpose(2, 1).to(torch.bfloat16), targets)
        patterns_loss = patterns_criterion(outputs.logits.transpose(2,1).to(torch.bfloat16), targets)
        total_loss    = loss + patterns_loss
        
        # Backward and optimize
        total_loss.backward()
        optimizer.step()
        train_loss.append(total_loss.item())

        #print out info
        t.set_postfix(loss=loss.item(), patterns_loss=patterns_loss.item(), memory_consumption=calculate_memory(model, optimizer, ids, mask, targets))

    # Get train loss
    train_loss = np.mean(train_loss)
    print(f"Training loss: {train_loss}")
    
    # Save losses
    train_losses.append(train_loss)
    
    val_loss = []
    model.eval()
    # Validation
    with torch.no_grad():
      with tqdm(validation_loader, desc=f"Validation...") as t:
        for data in t:
          # move data to GPU
          ids  = data['input_ids'].to(device)
          mask = data['attention_mask'].to(device)
          
          # shift targets backwards if causal model
          targets = get_targets(data, tokenizer, is_causal, device)
          # shift targets forwards if seq2seq model. Get decoder input_ids and decoder attention mask in the case where you have a causal model
          dec_input, dec_mask = get_dec_ids_and_mask(targets, tokenizer, is_causal, device)
          
          # Forward Pass
          outputs = model(ids, mask) if hps.is_causal else model(ids, mask, dec_input, dec_mask)

          # loss calculation
          loss          = criterion(outputs.logits.transpose(2, 1), targets)
          patterns_loss = patterns_criterion(outputs.logits.transpose(2,1), targets)
          total_loss    = loss + patterns_loss
          val_loss.append(total_loss.item())
          t.set_postfix(loss=loss.item(), patterns_loss=patterns_loss.item())
          
    # Get eval loss
    val_loss = np.mean(val_loss)
    print(f"Validation loss: {val_loss}")

  # Save losses
  val_losses.append(val_loss)
  return train_losses, val_losses

def evaluate(hps):
  # Use the hyperparemeters for the training
  criterion          = hps.criterion
  patterns_criterion = hps.patterns_criterion
  tokenizer          = hps.tokenizer
  device             = hps.device
  is_causal          = hps.is_causal
  
  test_loss = []
  model.eval()
  with torch.no_grad():
    with tqdm(testing_loader, desc=f"Testing...") as t:
      for data in t:
        # move data to GPU
        print(data)
        ids  = data['input_ids'].to(device)
        mask = data['attention_mask'].to(device)
        # shift targets backwards if causal model
        targets = get_targets(data, tokenizer, is_causal, device)
        # shift targets forwards if seq2seq model. Get decoder input_ids and decoder attention mask in the case where you have a causal model
        dec_input, dec_mask = get_dec_ids_and_mask(targets, tokenizer, is_causal, device)
        # Forward Pass
        outputs = model(ids, mask) if hps.is_causal else model(ids, mask, dec_input, dec_mask)
        # loss calculation
        loss          = criterion(outputs.logits.transpose(2, 1), targets)
        patterns_loss = patterns_criterion(outputs.logits.transpose(2,1), targets)
        total_loss    = loss + patterns_loss
        test_loss.append(total_loss.item())
        t.set_postfix(loss=loss.item(), patters_loss=patterns_loss.item())
  # Get eval loss
  test_loss = np.mean(test_loss)
  print(f"Testing loss: {test_loss}")
  return test_loss


def train_and_evaluate(model, logger, training_loader, validation_loader, testing_loader, hps):
  # Train the model
  train_losses, val_losses = train(hps)

  # Evaluate the model performance
  evaluate(hps)

  # Plot training and validation plots
  plot_training_plots(train_losses, val_losses)




if __name__ == '__main__':
  # Arguments
  parser = argparse.ArgumentParser(description='Training arguments parser')
  args   = parse_arguments(parser)
  hps    = hyperparameters(args)

  # hardcoded hps values. For code testing
  hps.max_new_binary_tokens_length = 0
  hps.model_name = "openaccess-ai-collective/tiny-mistral" if hps.is_causal == True else "google/t5-efficient-tiny"
  
  # Model Loading
  model = load_model(hps)
  pprint(hps)
  print(str(type(model)))

  # Load tokenizer
  tokenizer = AutoTokenizer.from_pretrained(hps.model_name, model_max_length=4096)
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training
  if not hps.is_causal:
    tokenizer.add_special_tokens({"cls_token": "<s>"})

  # Load and Form the Dataset appropriately
  raw_dataset = load_raw_dataset(hps.data_file)
  dataset = dataset_formation(raw_dataset, is_causal=hps.is_causal)
 
  # Create new tokens. Binary combinations to interpret the generated patterns.
  new_tokens = list(generate_all_possible_binary_combination(starting_point=1, max_binary_length=hps.max_new_binary_tokens_length)) if hps.max_new_binary_tokens_length > 0 else []
  print(f"Tokenizer vocabulary: {len(tokenizer)}. New added tokens: {len(new_tokens)}")

  # Resize the Embeddings
  tokenizer.add_tokens(new_tokens)
  model.resize_token_embeddings(len(tokenizer))
  
  if hps.peft:
    peft_config = LoraConfig(
      task_type=TaskType.CAUSAL_LM, r=hps.lora_r, lora_alpha=hps.lora_alpha, lora_dropout=hps.lora_dropout
    )
    model = get_peft_model(model, peft_config)
    print(f"{model.print_trainable_parameters()}\n")
    del peft_config
  else:
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
  print(f"Model size: {convert_bytes(model_size_in_bytes(model))}")
  # Tokenize the dataset
  tokenized_dataset = dataset.map(tokenize_fn, batched=True, num_proc=16, remove_columns=['text'], fn_kwargs={'tokenizer': tokenizer, 'is_causal': hps.is_causal})
  hps.collate_fn = MyCollate(tokenizer=tokenizer, is_causal=hps.is_causal)
  print(tokenized_dataset)
  hps.tokenizer = tokenizer

  # Set the environment based on the GPU availability
  model, logger, training_loader, validation_loader, testing_loader = set_training_environment(model, tokenized_dataset, hps)
  print(model)

  # Train
  train_and_evaluate(model, logger, training_loader, validation_loader, testing_loader, hps)





