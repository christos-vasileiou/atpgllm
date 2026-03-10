"""
dataset_utils.py
================

Utilities for loading, buffering and formatting datasets for SFT and
GRPO training pipelines.

* :func:`buffer_streaming_dataset` — materialise a streaming
  ``IterableDataset`` into a regular ``Dataset`` (required by
  ``GRPOTrainer``).
* :func:`format_dataset_for_training` — convert raw records into the
  chat-prompt format expected by ``SFTTrainer`` or ``GRPOTrainer``.
"""

from __future__ import annotations

import sys
from typing import Optional, Literal

from datasets import Dataset, IterableDataset
from tqdm import tqdm
from transformers import AutoTokenizer

from conversation import ConversationExample
from tools import TOOLS
import sys
import itertools
import concurrent.futures
from typing import Optional, Literal
from datasets import Dataset, IterableDataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm

import re
import matplotlib.pyplot as plt

# =====================================================================
# Training mode enumeration
# =====================================================================

class TrainingMode:
    """Training mode enumeration."""
    SFT = "sft"
    GRPO = "grpo"


# =====================================================================
# Dataset helpers
# =====================================================================

def process_batch(batch: list, tokenizer: AutoTokenizer, unique_by: str, max_prompt_length: int) -> list:
    """Helper function to tokenize and filter a batch of examples."""
    texts = [ex[unique_by] for ex in batch]
    
    # Batched tokenization: This drops the GIL and multi-threads natively in Rust
    encodings = tokenizer(
        texts, 
        add_special_tokens=False,     # We only need the length, not special tokens
        truncation=False,             # We want to measure the true length
        return_attention_mask=False,  # Saves memory/compute
        return_token_type_ids=False   # Saves memory/compute
    )
    
    valid_examples = []
    for ex, length in zip(batch, map(len, encodings["input_ids"])):
        if length < max_prompt_length:
            valid_examples.append(ex)
            
    return valid_examples

def buffer_streaming_dataset(
    streaming_dataset: IterableDataset,
    buffer_size: int = 10000,
    shuffle: bool = True,
    seed: int = 42,
    unique_by: Optional[Literal["text", "prompt", "module_name", "netlist"]] = None,
    tokenizer: AutoTokenizer = None,
    max_prompt_length: int = 16384,
    batch_size: int = 1000,
    num_workers: int = 4,
) -> Dataset:
    
    if tokenizer is None:
        raise ValueError("A tokenizer must be provided.")

    dataset_iter = iter(streaming_dataset)
    
    # 1. Safely peek at the first element WITHOUT losing it
    try:
        first_example = next(dataset_iter)
    except StopIteration:
        raise ValueError("The streaming dataset is empty. Check your dataset path.")

    if unique_by is None:
        if 'text' in first_example:
            unique_by = 'text'
        elif 'prompt' in first_example:
            unique_by = 'prompt'
            
    if unique_by is None or unique_by not in first_example:
        raise ValueError(f"Unique by field '{unique_by}' not found in dataset keys: {list(first_example.keys())}")

    # Reconstruct the iterator so the first example gets processed
    dataset_iter = itertools.chain([first_example], dataset_iter)

    examples = []
    unique_values = set()
    
    desc = (
        f"Buffering dataset (max {buffer_size}, unique by '{unique_by}')"
        if buffer_size > 0
        else "Buffering entire dataset"
    )
    pbar = tqdm(total=buffer_size if buffer_size > 0 else None, desc=desc, file=sys.stdout)

    # Helper generator to yield perfectly sized batches of unique items
    def unique_batch_generator():
        current_batch = []
        for example in dataset_iter:
            val = example[unique_by]
            if val not in unique_values:
                unique_values.add(val)
                current_batch.append(example)
                if len(current_batch) == batch_size:
                    yield current_batch
                    current_batch = []
        if current_batch:
            yield current_batch

    is_fast_tokenizer = getattr(tokenizer, "is_fast", False)

    # 2. Process Data
    if is_fast_tokenizer:
        # FAST PATH: Rely on Rust's internal multithreading via batching
        for batch in unique_batch_generator():
            valid_batch = process_batch(batch, tokenizer, unique_by, max_prompt_length)
            
            # Append only what we strictly need to hit buffer_size
            needed = buffer_size - len(examples) if buffer_size > 0 else len(valid_batch)
            examples.extend(valid_batch[:needed])
            pbar.update(len(valid_batch[:needed]))
            
            if buffer_size > 0 and len(examples) >= buffer_size:
                break
    else:
        # SLOW PATH: Fallback to Python Multiprocessing for legacy tokenizers
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(process_batch, batch, tokenizer, unique_by, max_prompt_length)
                for batch in unique_batch_generator()
            ]
            
            for future in concurrent.futures.as_completed(futures):
                valid_batch = future.result()
                needed = buffer_size - len(examples) if buffer_size > 0 else len(valid_batch)
                examples.extend(valid_batch[:needed])
                pbar.update(len(valid_batch[:needed]))
                
                if buffer_size > 0 and len(examples) >= buffer_size:
                    # Cancel remaining tasks to free up CPU
                    for f in futures: f.cancel()
                    break

    pbar.close()

    if not examples:
        raise ValueError("No examples were buffered. Check max_prompt_length or dataset content.")

    dataset = Dataset.from_list(examples)

    if shuffle:
        dataset = dataset.shuffle(seed=seed)

    print(f"Buffered {len(dataset)} examples into memory.")
    return dataset

def format_dataset_for_training(dataset, tokenizer: AutoTokenizer, training_mode: str):
    """
    Convert a dataset of records into a format accepted by SFTTrainer or
    GRPOTrainer.

    Each record is converted into a single chat conversation using
    :meth:`ConversationExample.from_record`.  The resulting conversation
    is rendered to a plain prompt using the tokenizer's chat template via
    ``apply_chat_template()``.

    This function supports both regular ``Dataset`` and streaming
    ``IterableDataset``.  For streaming datasets, processing is done
    lazily on-the-fly using ``.map()``, which is memory-efficient for
    large datasets that cannot fit in memory.

    Parameters
    ----------
    dataset
        The dataset containing raw records (``Dataset`` or
        ``IterableDataset``).
    tokenizer : AutoTokenizer
        The tokenizer whose chat template will be applied.
    training_mode : str
        ``TrainingMode.SFT`` or ``TrainingMode.GRPO``.

    Returns
    -------
    Dataset or IterableDataset
        A dataset ready for ``SFTTrainer`` or ``GRPOTrainer``.  For SFT
        the field is ``text``; for GRPO the field is ``prompt``.  Returns
        the same type as the input dataset.
    """
    is_streaming = isinstance(dataset, IterableDataset)
    use_tools = 'tools' in tokenizer.chat_template or 'tool' in tokenizer.chat_template

    if training_mode == TrainingMode.SFT:
        def format_fn(record):
            convo = ConversationExample.from_record(record, use_tools=use_tools)
            prompt = tokenizer.apply_chat_template(
                convo.messages, tokenize=False, tools=TOOLS if use_tools else None,
            )
            return {"text": prompt}

        return dataset.map(
            format_fn,
            remove_columns=dataset.column_names if not is_streaming else None,
        )

    elif training_mode == TrainingMode.GRPO:
        def format_fn(record):
            convo = ConversationExample.from_record(record, use_tools=use_tools)
            prompt_messages = [
                m for m in convo.messages
                if m["role"] != "assistant" and m["role"] != "tool"
            ]
            prompt = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False,
                tools=TOOLS if use_tools else None,
                add_generation_prompt=True,
            )
            return {"prompt": prompt, **record}

        return dataset.map(format_fn)

    else:
        raise ValueError(f"Unknown training mode: {training_mode}")


def plot_max_gates_by_prompt_length(
    dataset, 
    tokenizer, 
    text_key="netlist",           # Change this if your text is nested, e.g., ex["netlist"]["netlist"]
    nested_text_key=None,         # Set to "netlist" if your structure is x["netlist"]["netlist"]
    prompt_lengths=[(i+1)*1024 for i in range(16)],
    batch_size=1000,
    image_filename="max_gates_vs_length.png"
):
    """
    Passes through the dataset ONCE to compute token lengths and gate counts,
    then evaluates max gates across multiple prompt length thresholds and plots it.
    """
    # 1. Compile regex once
    regex_instances = re.compile(r"\s*\w+\s+\w+\s*\(\s*\.\w+\(\s*\w+")
    
    # Store tuples of (token_length, gate_count)
    stats = []
    
    # 2. Process dataset in batches (drops the GIL for fast tokenization)
    dataset_iter = iter(dataset)
    
    # Try to get the total length for ETA calculation, fallback to None if it's a streaming dataset
    try:
        total_items = len(dataset)
    except (TypeError, AttributeError):
        total_items = None
        
    pbar = tqdm(total=total_items, desc="Extracting tokens & gates", unit="ex")

    # 3. Extract token lengths and gate counts in a single pass
    print("Extracting token lengths and gate counts in a single pass...")
    while True:
        # Fetch a batch
        batch = []
        try:
            for _ in range(batch_size):
                batch.append(next(dataset_iter))
        except StopIteration:
            pass # End of dataset
            
        if not batch:
            break
            
        # Extract texts for this batch
        texts = list(set([ex[text_key][nested_text_key] for ex in batch]))
        
        # Fast Rust tokenization (only grabbing lengths, skipping masks/type_ids for speed)
        encodings = tokenizer(
            texts, 
            add_special_tokens=False, 
            truncation=False, 
            return_attention_mask=False, 
            return_token_type_ids=False
        )
        batch_token_lengths = [len(ids) for ids in encodings["input_ids"]]
        
        # Fast list comprehension for regex matching
        batch_gate_counts = [len(regex_instances.findall(text)) for text in texts]
        
        # Store the pairs
        stats.extend(zip(batch_token_lengths, batch_gate_counts))

        # Update progress bar by the number of examples processed in this batch
        pbar.update(len(batch))

    # Close the progress bar once the loop finishes
    pbar.close()

    if not stats:
        raise ValueError("Dataset was empty or texts could not be extracted.")
    
    # 3. Calculate max lengths for each target threshold instantly
    print("\nCalculating maximums for each threshold...")
    results = {}
    max_gate_list = []
    
    # Sort prompt lengths to ensure plot is in order
    prompt_lengths = sorted(prompt_lengths)
    
    for p_len in prompt_lengths:
        # Filter all gate counts where the token length is STRICTLY LESS than p_len
        valid_gate_counts = [gates for tokens, gates in stats if tokens < p_len]
        
        max_gates = max(valid_gate_counts) if valid_gate_counts else 0
        results[p_len] = max_gates
        max_gate_list.append(max_gates)
        print(f"Max Prompt Length: {p_len:<5} | Max Gates: {max_gates}")
    
    # 4. Generate and save the plot
    plt.figure(figsize=(10, 6))
    plt.plot(prompt_lengths, max_gate_list, marker='o', linestyle='-', color='#1f77b4', linewidth=2)
    plt.title('Maximum Gate Count vs. Maximum Prompt Length', fontsize=14, pad=15)
    plt.xlabel('Max Prompt Length (Tokens)', fontsize=12)
    plt.ylabel('Maximum Gate Count Filtered', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(prompt_lengths)
    
    # Add data labels slightly offset from the points
    for i, txt in enumerate(max_gate_list):
        plt.annotate(txt, (prompt_lengths[i], max_gate_list[i]), 
                     textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.tight_layout()
    plt.savefig(image_filename, dpi=300)
    plt.close()
    
    print(f"\nPlot saved successfully to: {image_filename}")
    return results
