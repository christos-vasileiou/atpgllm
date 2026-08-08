# Imports for Supervised Fine-Tuning of Llama 2
from transformers import (
  TrainingArguments,
  pipeline,
  logging,
)
from trl import SFTTrainer
import torch
import argparse

if __name__ == "__main__":
  from atpgllm import (
    hyperparameters, 
    parse_arguments, 
    models_causal, 
    models_seq2seq, 
    load_model, 
    initialize_training_environment, 
    load_tokenizer, 
    load_raw_dataset, 
    dataset_formation,
    dataset_formation_using_chat_template,
    prepare_objects_for_training, 
    tokenize_fn,
    is_distributed_execution,
    MyCollate,
    ATPGAccuracy,
    MetricsCallback,
    MetricsLoggingCallback,
    ATPGTrainer,
  )

  # Arguments
  parser = argparse.ArgumentParser(description='Training arguments parser')
  args   = parse_arguments(parser)
  hps    = hyperparameters(args)
  print(hps)
  
  # Set the model name
  hps.model_name = models_causal[hps.model_name] if hps.is_causal == True else models_seq2seq[hps.model_name]

  # Initialize the training environment based on the GPU availability and parallelization
  initialize_training_environment(hps)

  # Model Loading
  model = load_model(hps)

  # Load tokenizer
  model, tokenizer = load_tokenizer(model, hps)
  hps.tokenizer = tokenizer

  # Load and Form the Dataset appropriately
  dataset = load_raw_dataset(hps.data_file)

  #dataset = dataset_formation(raw_dataset, is_causal=hps.is_causal, test_size=hps.test_size)
  # dataset = dataset_formation_using_chat_template(raw_dataset, hps=hps)

  # Prepare objects for training and validation methods (optimizer, scheduler, dataloader, etc...)
  model, training_loader, validation_loader, testing_loader = prepare_objects_for_training(model, dataset, hps)

  ################################################################################
  # TrainingArguments parameters
  ################################################################################

  # Output directory where the model predictions and checkpoints will be stored
  output_dir = "./results" 

  # Enable fp16/bf16 training (set bf16 to True with an A100)
  fp16 = not hps.bf16
  bf16 = hps.bf16

  # Enable Fully Sharded Data Parallel (FSDP) training if cuda is available and num of GPUs > 1
  if torch.cuda.is_available() and is_distributed_execution() and torch.cuda.device_count() > 1:
    fsdp = ["full_shard", "offload"]
  else:
    fsdp = False

  # Number of update steps to accumulate the gradients for
  gradient_accumulation_steps = hps.batch_size // (hps.micro_batch_size * hps.local_world_size)

  # Enable gradient checkpointing
  gradient_checkpointing = True

  # Maximum gradient normal (gradient clipping)
  max_grad_norm = 0.3 

  # Weight decay to apply to all layers except bias/LayerNorm weights
  weight_decay = 0.001 
  
  # Select if model's repo is private
  hub_private_repo = True
  push_to_hub = True

  # Optimizer to use
  optim = "paged_adamw_32bit" 

  # Learning rate schedule
  lr_scheduler_type = "cosine_with_restarts" #cosine_with_restarts or cosine

  # Number of training steps (overrides num_train_epochs)
  max_steps = -1 

  # Ratio of steps for a linear warmup (from 0 to learning rate)
  warmup_steps = 2000

  # Group sequences into batches with same length
  # Saves memory and speeds up training considerably
  group_by_length = True

  # Save checkpoint every X updates steps
  save_steps = 20000

  # Log every X updates steps
  logging_steps = 500
  
  # Set training parameters
  training_args = TrainingArguments(
    log_level='debug', # Possible choices : 'debug', 'info', 'warning', 'error' and 'critical', plus a 'passive' level which doesn't set anything and keeps the current log level for the Transformers library (which will be `"warning"` by default).
    output_dir=output_dir,
    # do_train=True,
    # do_eval=True,
    # evaluation_strategy="epoch",
    num_train_epochs=hps.epochs,
    per_device_train_batch_size=hps.batch_size,
    per_device_eval_batch_size=hps.batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    learning_rate=hps.lr,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    fsdp=fsdp,
    max_grad_norm=max_grad_norm,
    save_steps=save_steps,
    logging_steps=logging_steps,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    lr_scheduler_kwargs={'num_cycles': 10},
    # warmup_ratio=warmup_ratio,
    warmup_steps=warmup_steps,
    report_to="wandb", # "azure_ml", "clearml", "codecarbon", "comet_ml", "dagshub", "dvclive", "flyte", "mlflow", "neptune", "tensorboard", and "wandb"
    hub_private_repo=hub_private_repo,
    push_to_hub=push_to_hub,
    # hub_model_id=models_causal['llama-2-atpg-lora'],
    # hub_strategy="every_save",
    # hub_private_repo=True
  )

  ################################################################################
  # SFT parameters
  ################################################################################

  # Maximum sequence length to use
  max_seq_length = hps.model_max_length
  
  # Pack multiple short examples in the same input sequence to increase efficiency
  packing = True

  dataset['train'] = dataset['train'].remove_columns('netlist')
  dataset['validation'] = dataset['validation'].remove_columns('netlist')
  dataset['test'] = dataset['test'].remove_columns('netlist')

  # Set supervised fine-tuning parameters
  trainer = SFTTrainer(
    model=model,
    args=training_args,
    data_collator=MyCollate(tokenizer=hps.tokenizer, is_causal=hps.is_causal, lora=hps.lora, sft=True),
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    tokenizer=tokenizer,
    # callbacks=[MetricsLoggingCallback(hps.accuracy, logging_steps)], #MetricsCallback(hps.accuracy), 
    peft_config=hps.lora_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    packing=packing
  )
  trainer.accelerate.print(f"{trainer.model}")
  trainer.model.print_trainable_parameters()
  if getattr(trainer.accelerate.state, "fsdp_plugin", None):
    from peft.utils.other import fsdp_auto_wrap_policy
    fsdp_plugin = trainer.accelerator.state.fsdp_plugin
    fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)
  
  # Train model
  trainer.train()

  # Save trained model
  if trainer.is_fsdp_enabled:
    trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
  trainer.save_model()
  # trainer.model.save_pretrained(models_causal['llama-2-atpg-lora'], save_embedding_layers=True)
