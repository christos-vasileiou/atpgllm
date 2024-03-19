# @title # Imports from Fine-Tuning of Llama 2
import os
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

# Function to plot data
def plot_data(elements, title, figfile):
    # Separate elements and frequencies
    binaries = [item[0] for item in elements]
    frequencies = [item[1] for item in elements]
    decimals = [int(b, 2) for b in binaries]

    # Create a bar plot
    plt.figure(figsize=(10, 5))
    plt.bar(binaries, frequencies, tick_label=decimals)

    # Add titles and labels
    plt.title(title)
    plt.xlabel('Pattern (Decimal)')
    plt.ylabel('Frequency')

    # Show plot
    plt.savefig(figfile)

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


if __name__ == "__main__":
  #Load ATPG dataset
  # data file name
  data_file = '../../data/atpg_data_5pis_only.csv'

  # Load Dataset
  raw_dataset = load_dataset('csv', data_files=data_file).remove_columns('Unnamed: 0')

  import matplotlib.pyplot as plt
  from collections import Counter
  from pprint import pprint

  all_patterns = [pat for patterns in raw_dataset['train']['patterns'] for pat in patterns.replace(' ', '').split('\n')]
  unique_patterns = Counter(all_patterns)
  pprint(unique_patterns)
  print(list(int(p, 2) for p in unique_patterns.keys()))

  # Convert to list of tuples and sort by frequency
  sorted_data = sorted(unique_patterns.items(), key=lambda x: x[1])
  # Select top and bottom N elements
  N = 6
  top_elements = sorted_data[-N:]
  bottom_elements = sorted_data[:N]

  # Print data info
  plt.bar(list(int(p, 2) for p in unique_patterns.keys()), unique_patterns.values())
  plt.xlabel('Pattern')
  plt.ylabel('Count')
  plt.title('Pattern Frequency')
  plt.savefig('pattern_frequency.png')

  # Plot for top elements
  plot_data(top_elements, f'Top {N} Most Frequent Patterns', 'most_freq.png')

  # Plot for bottom elements
  plot_data(bottom_elements, f'Top {N} Least Frequent Patterns', 'least_freq.png')

  # prompt engineering - task specification
  prompts = ["[INST] Your task is to write test vectors that can achieve " + answers.split('\n')[7].split()[-1] + " coverage for the design:\n\n```\n" + prompts + "```\n\nPlease wrap the test vectors in ```. [/INST]" for prompts, answers in zip(raw_dataset['train']['prompts'], raw_dataset['train']['answers'])]
  print(prompts[0])

  # response specification
  answers = ["The test vectors for the provided design are:\n\n```\n" + '\n'.join([pat for pat in patterns.replace(' ', '').split('\n')]) + "\n```" for patterns in raw_dataset['train']['patterns']]
  print(answers[0])

  o_dataset = [prompt + "\n" + answer for prompt, answer in zip(prompts, answers)]
  print(o_dataset[0])

  del raw_dataset
  del prompts
  del answers

  dataset_size = len(o_dataset) // 10
  dataset = Dataset.from_dict({'text': o_dataset[:dataset_size]})

  # The model that you want to train from the Hugging Face hub
  model_name = "openaccess-ai-collective/tiny-mistral" # @param ["NousResearch/Llama-2-7b-chat-hf", "openaccess-ai-collective/tiny-mistral"]

  # Fine-tuned model name
  new_model = "llama-2-7b-atpg" 

  ################################################################################
  # QLoRA parameters
  ################################################################################

  # LoRA attention dimension
  lora_r = 64 

  # Alpha parameter for LoRA scaling
  lora_alpha = 16 

  # Dropout probability for LoRA layers
  lora_dropout = 0.1 

  ################################################################################
  # bitsandbytes parameters
  ################################################################################

  # Activate 4-bit precision base model loading
  use_4bit = True 

  # Compute dtype for 4-bit base models
  bnb_4bit_compute_dtype = "float16" 

  # Quantization type (fp4 or nf4)
  bnb_4bit_quant_type = "nf4" 

  # Activate nested quantization for 4-bit base models (double quantization)
  use_nested_quant = False

  ################################################################################
  # TrainingArguments parameters
  ################################################################################

  # Output directory where the model predictions and checkpoints will be stored
  output_dir = "./results" 

  # Number of training epochs
  num_train_epochs = 5

  # Enable fp16/bf16 training (set bf16 to True with an A100)
  fp16 = False 
  bf16 = True 

  # Batch size per GPU for training
  per_device_train_batch_size = 32

  # Batch size per GPU for evaluation
  per_device_eval_batch_size = 32

  # Number of update steps to accumulate the gradients for
  gradient_accumulation_steps = 1 

  # Enable gradient checkpointing
  gradient_checkpointing = True 

  # Maximum gradient normal (gradient clipping)
  max_grad_norm = 0.3 

  # Initial learning rate (AdamW optimizer)
  learning_rate = 2e-4 

  # Weight decay to apply to all layers except bias/LayerNorm weights
  weight_decay = 0.001 

  # Optimizer to use
  optim = "paged_adamw_32bit" 

  # Learning rate schedule
  lr_scheduler_type = "cosine"

  # Number of training steps (overrides num_train_epochs)
  max_steps = -1 

  # Ratio of steps for a linear warmup (from 0 to learning rate)
  warmup_ratio = 0.03

  # Group sequences into batches with same length
  # Saves memory and speeds up training considerably
  group_by_length = True

  # Save checkpoint every X updates steps
  save_steps = 20000

  # Log every X updates steps
  logging_steps = 20000

  ################################################################################
  # SFT parameters
  ################################################################################

  # Maximum sequence length to use
  max_seq_length = None 

  # Pack multiple short examples in the same input sequence to increase efficiency
  packing = False 

  # Load the entire model on the GPU 0
  device_map = {"": 0 if torch.cuda.is_available() else 'cpu'} 

  ################################################################################
  # Input ports maximum length. Parameters for extra vocabulary tokens
  ################################################################################

  # Maximum input ports the tokenizer can handle.
  max_in_ports_num = 5 

  # Load tokenizer and model with QLoRA configuration
  compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

  #bnb_config = BitsAndBytesConfig(
  #    load_in_4bit=use_4bit,
  #    bnb_4bit_quant_type=bnb_4bit_quant_type,
  #    bnb_4bit_compute_dtype=compute_dtype,
  #    bnb_4bit_use_double_quant=use_nested_quant,
  #)

  # Check GPU compatibility with bfloat16
  if torch.cuda.is_available():
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

  # Load base model
  model = AutoModelForCausalLM.from_pretrained(
      model_name,
      #quantization_config=bnb_config,
      device_map=device_map,
  )
  model.config.use_cache = False
  model.config.pretraining_tp = 1

  # Load LLaMA tokenizer
  tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
  tokenizer.pad_token    = tokenizer.eos_token
  tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

  # Load LoRA configuration
  peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
  )

  # Set training parameters
  training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard"
  )

  # Set supervised fine-tuning parameters
  trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    packing=packing,
  )


  # Generate all possible binary representation
  combs = list(generate_all_possible_binary_combination(starting_point=1, max_binary_length=max_in_ports_num))

  # add the new tokens in the tokenizer. Results to new vocabulary
  tokenizer.add_tokens(combs)

  # resize the token embedding layer w.r.t new vocabulary
  model.resize_token_embeddings(len(tokenizer))

  # Train model
  trainer.train()

  # Save trained model
  trainer.model.save_pretrained(new_model, save_embedding_layers=True)




