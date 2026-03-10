#%%
from datasets import load_dataset
from transformers import AutoTokenizer
from dataset_utils import TrainingMode, buffer_streaming_dataset, format_dataset_for_training, plot_max_gates_by_prompt_length
import os

#%%
# Get model name from environment variable
model_name = os.environ["MODEL"]

# Path to the HuggingFace dataset to analyze
dataset_path = "chrivasileiou/asap7-language-of-test"
# Buffer size (not actually used directly in this script, but possibly referenced by dataset_utils functions)
buffer_size = 50_000

#%%
# Load the tokenizer for the specified model
print(f"Loading tokenizer for model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the streaming dataset split (train)
print(f"Loading dataset from path: {dataset_path}")
data = load_dataset(dataset_path, split="train", streaming=True)

# Format the dataset using the GRPO training mode (sets 'prompt' fields, applies chat template, etc)
print("Formatting dataset for GRPO training mode...")
formatted_data = format_dataset_for_training(data, tokenizer, TrainingMode.GRPO)

# Analyze the relation between prompt token length and the maximum number of gates found in the text (netlist)
# Writes output to a PNG, can adjust prompt_lengths or batch size if needed
print("Plotting max gates by prompt length...")
plot_max_gates_by_prompt_length(formatted_data, tokenizer, nested_text_key="netlist")
print("Done.")