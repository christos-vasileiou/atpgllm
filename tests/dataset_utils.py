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

def buffer_streaming_dataset(
    streaming_dataset: IterableDataset,
    buffer_size: int = 10000,
    shuffle: bool = True,
    seed: int = 42,
    unique_by: Optional[Literal["text", "prompt", "module_name", "netlist"]] = None,
) -> Dataset:
    """
    Convert a streaming IterableDataset to a regular Dataset by buffering examples.

    This is necessary for trainers that don't support streaming datasets
    (like GRPOTrainer).  The *buffer_size* controls how many examples are
    loaded into memory.

    Parameters
    ----------
    streaming_dataset : IterableDataset
        The streaming dataset to buffer.
    buffer_size : int
        Maximum number of examples to load into memory.  Set to -1 to
        load all examples (only use if you know the dataset fits in
        memory).
    shuffle : bool
        Whether to shuffle the buffered dataset.  Recommended for training.
    seed : int
        Random seed for shuffling.
    unique_by : str, optional
        The field to use for uniqueness.  Can be ``"text"``, ``"prompt"``,
        ``"module_name"`` or ``"netlist"``.

    Returns
    -------
    Dataset
        A regular Dataset containing the buffered examples.
    """
    examples = []

    if unique_by is None:
        unique_by = 'text' if 'text' in next(iter(streaming_dataset)).keys() else None
    if unique_by is None:
        unique_by = 'prompt' if 'prompt' in next(iter(streaming_dataset)).keys() else None
    if unique_by not in next(iter(streaming_dataset)).keys():
        raise ValueError(f"Unique by field {unique_by} not found in dataset")

    desc = (
        f"Buffering dataset (max {buffer_size} examples) searching unique samples by '{unique_by}'"
        if buffer_size > 0
        else "Buffering entire dataset"
    )

    unique_values = set()
    for example in tqdm(streaming_dataset, desc=desc, file=sys.stdout):
        if buffer_size > 0 and len(examples) >= buffer_size:
            break
        unique_value = example[unique_by]
        if unique_value in unique_values:
            continue
        unique_values.add(unique_value)
        examples.append(example)

    if not examples:
        raise ValueError(
            "No examples were buffered from the streaming dataset. "
            "Check your dataset path."
        )

    dataset = Dataset.from_list(examples)

    if shuffle:
        dataset = dataset.shuffle(seed=seed)

    print(f"Buffered {len(dataset)} examples into memory")
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
