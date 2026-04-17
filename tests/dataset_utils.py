"""
dataset_utils.py
================

Utilities for loading, buffering and formatting datasets for SFT and
GRPO training pipelines.

* :func:`buffer_streaming_dataset` — materialise a streaming
  ``IterableDataset`` into a regular ``Dataset`` (required by
  ``GRPOTrainer``). ``skip_buffer_size`` and ``buffer_size`` are
  **independent**: the stream pointer advances past ``skip_buffer_size``
  valid rows first, then up to ``buffer_size`` rows are collected for the
  buffer (skips do not count toward ``buffer_size``).
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
    skip_buffer_size: int = 0,
) -> Dataset:
    """Materialise a streaming iterable into a :class:`datasets.Dataset`.

    **Two-phase stream consumption (independent quotas).** Valid examples are
    those passing uniqueness on ``unique_by`` and ``max_prompt_length``.

    1. Advance the logical stream pointer: discard the first
       ``skip_buffer_size`` valid examples. This count does **not** reduce
       ``buffer_size``.
    2. **Then** append at most ``buffer_size`` further valid examples into the
       returned dataset (``buffer_size == 0`` means keep reading until the
       stream ends).

    Order is the streaming iterator **before** ``shuffle``.

    Parameters
    ----------
    skip_buffer_size : int, default 0
        Skip this many **valid** examples (unique by ``unique_by``, under
        ``max_prompt_length``) from the start of the stream before filling the
        buffer. Use when resuming GRPO on the same stream to avoid retraining on
        prompts already covered in a prior run. Order follows the streaming
        iterator **before** ``shuffle``.
    """
    try:
        skip_buffer_size = int(skip_buffer_size)
        buffer_size = int(buffer_size)
    except (TypeError, ValueError) as e:
        raise ValueError("buffer_size and skip_buffer_size must be integers") from e
    if skip_buffer_size < 0:
        raise ValueError("skip_buffer_size must be >= 0")
    if buffer_size < 0:
        raise ValueError("buffer_size must be >= 0 (0 means buffer the rest of the stream)")
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
        f"Buffering dataset (max {buffer_size}, unique by '{unique_by}', skip {skip_buffer_size})"
        if buffer_size > 0
        else f"Buffering entire dataset (skip {skip_buffer_size})"
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
    skipped_valid = 0  # valid examples consumed in the skip phase only

    def consume_valid_batch(valid_batch: list) -> bool:
        """Apply skip phase then buffer phase; return True when buffer quota is met."""
        nonlocal skipped_valid
        for ex in valid_batch:
            # Phase 1 — stream offset: skip_buffer_size valid items (not counted toward buffer_size).
            if skipped_valid < skip_buffer_size:
                skipped_valid += 1
                continue
            # Phase 2 — buffer: up to buffer_size samples (0 = unbounded until EOF).
            if buffer_size > 0 and len(examples) >= buffer_size:
                return True
            examples.append(ex)
            pbar.update(1)
        return buffer_size > 0 and len(examples) >= buffer_size

    # 2. Process Data
    if is_fast_tokenizer:
        # FAST PATH: Rely on Rust's internal multithreading via batching
        for batch in unique_batch_generator():
            valid_batch = process_batch(batch, tokenizer, unique_by, max_prompt_length)
            if consume_valid_batch(valid_batch):
                break
    else:
        # SLOW PATH: Fallback to Python Multiprocessing for legacy tokenizers
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(process_batch, batch, tokenizer, unique_by, max_prompt_length)
                for batch in unique_batch_generator()
            ]
            # Iterate in submission order so skip_buffer_size matches stream order.
            for future in futures:
                valid_batch = future.result()
                if consume_valid_batch(valid_batch):
                    for f in futures:
                        f.cancel()
                    break

    pbar.close()

    if skipped_valid < skip_buffer_size:
        print(
            f"WARNING: stream ended after skipping {skipped_valid} valid example(s); "
            f"requested skip_buffer_size={skip_buffer_size}. "
            "Increase dataset size or lower skip_buffer_size."
        )

    if not examples:
        raise ValueError(
            "No examples were buffered after filtering. "
            f"(skip_buffer_size={skip_buffer_size}, skipped_valid={skipped_valid}, "
            "max_prompt_length, or empty stream). Check dataset path and filters."
        )

    dataset = Dataset.from_list(examples)

    if shuffle:
        dataset = dataset.shuffle(seed=seed)

    if skip_buffer_size > 0 and buffer_size > 0 and len(dataset) < buffer_size:
        print(
            f"WARNING: after skip_buffer_size={skip_buffer_size}, only {len(dataset)} example(s) "
            f"were collected; buffer_size={buffer_size} was the target (skip does not reduce that quota)."
        )

    print(
        f"Buffered {len(dataset)} example(s) into memory "
        f"(skip phase: skipped_valid={skipped_valid}/{skip_buffer_size}, buffer cap={buffer_size or 'none'})."
    )
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
    import os
    is_streaming = isinstance(dataset, IterableDataset)

    # Matches Verilog gate/module instantiations: <cell_type> <instance_name> ( ... )
    # - instance_name may be a normal identifier or an escaped identifier (starts with '\')
    # - escaped identifiers can contain '/', '*', '[', ']', etc. up to the first whitespace
    GATE_INGREDIENT = r"(?m)^(?!\s*(?:module|endmodule|input|output|inout|wire|wand|wor|tri|tri0|tri1|trireg|reg|logic|assign|parameter|localparam|specify|endspecify|genvar|generate|endgenerate|always|always_ff|always_comb|always_latch|initial|begin|end|if|else|case|endcase|for|while|repeat|forever)\b)\s*" \
                  r"[^\s(]+\s+(?:\\\S+|[A-Za-z_][A-Za-z0-9_$]*)\s*\("
    regex_instances = re.compile(GATE_INGREDIENT)
    def filter_fn(record):
        gates_cnt = len(regex_instances.findall(record['netlist']))
        return gates_cnt not in (1, 5, 84)
    dataset = dataset.filter(filter_fn)

    use_tools = hasattr(tokenizer, 'chat_template') and tokenizer.chat_template and ('tools' in tokenizer.chat_template or 'tool' in tokenizer.chat_template)

    if training_mode == TrainingMode.SFT:
        def format_fn(record):
            convo = ConversationExample.from_record(record, use_tools=use_tools)
            prompt = tokenizer.apply_chat_template(
                convo.messages, tokenize=False, tools=TOOLS if use_tools else None,
            )
            return {"text": prompt}

        return dataset.map(format_fn)

    elif training_mode == TrainingMode.GRPO:
        def format_fn(record):
            convo = ConversationExample.from_record(record, use_tools=use_tools)
            prompt_messages = [
                m for m in convo.messages
                if m["role"] not in ("assistant", "tool")
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


def plot_max_gates_by_context_length(
    dataset, 
    tokenizer, 
    context_lengths=[(i+1)*1024 for i in range(16)],
    batch_size=1000,
    image_filename="max_gates_vs_length.png"
):
    """
    Passes through the dataset ONCE to compute token lengths and gate counts,
    then evaluates max gates across multiple context length thresholds and plots it.
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
            
        # Extract texts for this batch. 
        # ex -> {"netlist": {"id": ..., "netlist": ...}}, derived from format_dataset_for_training() and GRPO vs SFT training mode
        netlists = list(set([ex["netlist"]["netlist"] for ex in batch]))
        prompts = [ex["prompt"] for ex in batch]

        # Fast Rust tokenization (only grabbing lengths, skipping masks/type_ids for speed)
        prompt_encodings = tokenizer(
            prompts, 
            add_special_tokens=False, 
            truncation=False, 
            return_attention_mask=False, 
            return_token_type_ids=False
        )
        batch_token_lengths = [len(ids) for ids in prompt_encodings["input_ids"]]
        
        # Fast list comprehension for regex matching
        batch_gate_counts = [len(regex_instances.findall(netlist)) for netlist in netlists]
        
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
    context_lengths = sorted(context_lengths)
    
    for p_len in context_lengths:
        # Filter all gate counts where the token length is STRICTLY LESS than p_len
        valid_gate_counts = [gates for tokens, gates in stats if tokens < p_len]
        
        max_gates = max(valid_gate_counts) if valid_gate_counts else 0
        results[p_len] = max_gates
        max_gate_list.append(max_gates)
        print(f"Max Prompt Length: {p_len:<5} | Max Gates: {max_gates}")
    
    # 4. Generate and save the plot
    plt.figure(figsize=(10, 6))
    plt.plot(context_lengths, max_gate_list, marker='o', linestyle='-', color='#1f77b4', linewidth=2)
    plt.title('Maximum Gate Count vs. Maximum Prompt Length', fontsize=14, pad=15)
    plt.xlabel('Max Prompt Length (Tokens)', fontsize=12)
    plt.ylabel('Maximum Gate Count Filtered', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(context_lengths)
    
    # Add data labels slightly offset from the points
    for i, txt in enumerate(max_gate_list):
        plt.annotate(txt, (context_lengths[i], max_gate_list[i]), 
                     textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.tight_layout()
    plt.savefig(image_filename, dpi=300)
    plt.close()
    
    print(f"\nPlot saved successfully to: {image_filename}")
    return results


def plot_max_tokens_by_gate_count(
    dataset, 
    tokenizer, 
    batch_size=1000,
    image_filename="max_tokens_vs_gates.png"
):
    """
    Passes through the dataset ONCE to compute token lengths and gate counts,
    then evaluates the maximum token length for each unique gate count and plots it.
    """
    # 1. Compile regex once
    regex_instances = re.compile(r"\s*\w+\s+\w+\s*\(\s*\.\w+\(\s*\w+")
    stats = []
    
    dataset_iter = iter(dataset)
    
    # 2. Setup the smart progress bar
    try:
        total_items = len(dataset)
    except (TypeError, AttributeError):
        total_items = None
        
    pbar = tqdm(total=total_items, desc="Extracting tokens & gates", unit="ex")

    # 3. Process dataset in batches
    while True:
        batch = []
        try:
            for _ in range(batch_size):
                batch.append(next(dataset_iter))
        except StopIteration:
            pass # End of dataset
            
        if not batch:
            break
            
        # ex -> {"netlist": ...}, derived from format_dataset_for_training() and GRPO vs SFT training mode
        netlists = [ex["netlist"] for ex in batch]
        texts = [ex["text"] for ex in batch]

        # Fast Rust tokenization
        text_encodings = tokenizer(
            texts, 
            add_special_tokens=False, 
            truncation=False, 
            return_attention_mask=False, 
            return_token_type_ids=False
        )

        batch_token_lengths = [len(ids) for ids in text_encodings["input_ids"]]
        batch_gate_counts = [len(regex_instances.findall(netlist)) for netlist in netlists]
        
        stats.extend(zip(batch_token_lengths, batch_gate_counts))
        
        # Update progress bar
        pbar.update(len(batch))

    pbar.close()

    if not stats:
        raise ValueError("Dataset was empty or texts could not be extracted.")

    # 4. Calculate maximum tokens for each gate count
    print("\nCalculating maximum context window for each netlist size...")
    gate_to_max_tokens = {}
    
    for tokens, gates in stats:
        # If we haven't seen this gate count yet, or if this token length is larger, update it
        if gates not in gate_to_max_tokens or tokens > gate_to_max_tokens[gates]:
            gate_to_max_tokens[gates] = tokens

    # Sort the dictionary by gate count so the X-axis plots in chronological order
    sorted_gate_counts = sorted(gate_to_max_tokens.keys())
    max_token_lengths = [gate_to_max_tokens[g] for g in sorted_gate_counts]

    # Print a quick summary to the console
    print(f"Found {len(sorted_gate_counts)} unique gate counts (min: {min(sorted_gate_counts)}, max: {max(sorted_gate_counts)})")

    # 5. Generate and save the plot
    plt.figure(figsize=(12, 6))
    
    # Using a slightly smaller marker '.' since we might have ~100 points
    plt.plot(sorted_gate_counts, max_token_lengths, marker='.', linestyle='-', color='#d62728', linewidth=1.5, markersize=8)
    
    plt.title('Maximum Token Length vs. Netlist Size', fontsize=14, pad=15)
    plt.xlabel('Netlist Size (Gate Count)', fontsize=12)
    plt.ylabel('Maximum Context Window (Tokens)', fontsize=12)
    
    # Add gridlines for readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set X-axis ticks to step by 5 or 10 so it isn't completely crowded
    step = 10 if max(sorted_gate_counts) > 50 else 5
    plt.xticks(range(0, max(sorted_gate_counts) + step, step))

    plt.tight_layout()
    plt.savefig(image_filename, dpi=300)
    plt.close()
    
    print(f"\nPlot saved successfully to: {image_filename}")
    
    return gate_to_max_tokens