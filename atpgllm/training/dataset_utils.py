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
* :func:`skip_streaming_dataset` — lazy skip on an already-formatted
  stream (tests / legacy). Streaming SFT should pass ``skip_buffer_size``
  into :func:`format_dataset_for_training` for the fast fused path.
* :func:`filter_streaming_dataset_by_prompt_length` — drop SFT ``messages``
  rows whose system+user prompt exceeds ``max_prompt_length`` tokens.
* :func:`format_dataset_for_training` — convert raw records into the
  chat-prompt format expected by ``SFTTrainer`` or ``GRPOTrainer``.
"""

from __future__ import annotations

import copy
import itertools
import os
import random
import re
import sys
import concurrent.futures
from collections import deque
from typing import Iterator, Literal, Optional

import matplotlib.pyplot as plt
from datasets import Dataset, IterableDataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from atpgllm.training.conversation import ConversationExample
from atpgllm.training.tools import TOOLS

# =====================================================================
# Training mode enumeration
# =====================================================================

class TrainingMode:
    """Training mode enumeration."""
    SFT = "sft"
    GRPO = "grpo"


# Gate-count filter (matches format_dataset_for_training).
_GATE_INGREDIENT = (
    r"(?m)^(?!\s*(?:module|endmodule|input|output|inout|wire|wand|wor|tri|tri0|tri1|"
    r"trireg|reg|logic|assign|parameter|localparam|specify|endspecify|genvar|generate|"
    r"endgenerate|always|always_ff|always_comb|always_latch|initial|begin|end|if|else|"
    r"case|endcase|for|while|repeat|forever)\b)\s*"
    r"[^\s(]+\s+(?:\\\S+|[A-Za-z_][A-Za-z0-9_$]*)\s*\("
)
_GATE_REGEX = re.compile(_GATE_INGREDIENT)


def _default_skip_num_workers() -> int:
    n = os.cpu_count() or 8
    return max(1, min(16, n))


def _gate_filter_passes(record: dict) -> bool:
    gates_cnt = len(_GATE_REGEX.findall(record["netlist"]))
    return gates_cnt not in (1, 5, 84)


def _format_sft_messages_record(record: dict, use_tools: bool) -> dict:
    convo = ConversationExample.from_record(record, use_tools=use_tools)
    out: dict = {"messages": convo.messages}
    if use_tools:
        out["tools"] = TOOLS
    return out


def _format_sft_messages_batch(records: list[dict], use_tools: bool) -> list[dict]:
    formatted = []
    for raw in records:
        rec = copy.deepcopy(raw)
        formatted.append(_format_sft_messages_record(rec, use_tools))
    return formatted


def _classify_sft_messages_batch(
    batch: list[dict],
    tokenizer: AutoTokenizer,
    use_tools: bool,
    max_prompt_length: Optional[int],
) -> list[str]:
    """Per-row status in stream order: ``gate``, ``length``, or ``ok``."""
    statuses: list[str] = ["gate"] * len(batch)
    tools = TOOLS if use_tools else None
    candidates: list[tuple[int, list[dict]]] = []

    for i, raw in enumerate(batch):
        if not _gate_filter_passes(raw):
            continue
        rec = copy.deepcopy(raw)
        try:
            prompt_messages = ConversationExample.prompt_messages_from_record(
                rec, use_tools=use_tools,
            )
        except Exception:
            statuses[i] = "length"
            continue
        if not prompt_messages:
            statuses[i] = "length"
            continue
        candidates.append((i, prompt_messages))

    if not candidates or max_prompt_length is None:
        for idx, _ in candidates:
            statuses[idx] = "ok"
        return statuses

    prompt_texts = [
        tokenizer.apply_chat_template(
            msgs,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
        )
        for _, msgs in candidates
    ]
    encodings = tokenizer(
        prompt_texts,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    for (idx, _), token_ids in zip(candidates, encodings["input_ids"]):
        if len(token_ids) < max_prompt_length:
            statuses[idx] = "ok"
        else:
            statuses[idx] = "length"
    return statuses


def _classify_sft_messages_batch_worker(
    batch: list[dict],
    tokenizer: AutoTokenizer,
    use_tools: bool,
    max_prompt_length: Optional[int],
) -> list[str]:
    return _classify_sft_messages_batch(batch, tokenizer, use_tools, max_prompt_length)


def _sft_messages_stream(
    raw_iter: Iterator[dict],
    tokenizer: AutoTokenizer,
    use_tools: bool,
    max_prompt_length: Optional[int],
    skip_count: int,
    batch_size: int,
    num_workers: int,
) -> Iterator[dict]:
    """Fused gate filter, prompt-length filter, optional skip, and SFT formatting."""
    skip_count = int(skip_count)
    batch_size = max(1, int(batch_size))
    num_workers = max(1, int(num_workers))

    skipped_valid = 0
    skip_done = skip_count == 0
    stats = {"raw": 0, "gate": 0, "length": 0, "skipped": 0, "yielded": 0}

    skip_pbar = (
        tqdm(
            total=skip_count,
            desc="[SFT] Skipping training examples",
            unit="ex",
            file=sys.stdout,
        )
        if skip_count > 0
        else None
    )

    def format_records(records: list[dict]) -> list[dict]:
        # Full SFT format stays on the main process to avoid pool deadlocks
        # (classify jobs already occupy worker_pool during pipelined skip).
        return _format_sft_messages_batch(records, use_tools)

    def consume_classified(batch: list[dict], statuses: list[str]) -> Iterator[dict]:
        nonlocal skipped_valid, skip_done
        if not batch:
            return
        stats["raw"] += len(batch)
        to_format: list[dict] = []
        for raw, status in zip(batch, statuses):
            if status == "gate":
                stats["gate"] += 1
                continue
            if status == "length":
                stats["length"] += 1
                continue
            if not skip_done:
                skipped_valid += 1
                stats["skipped"] += 1
                if skip_pbar is not None:
                    skip_pbar.update(1)
                if skipped_valid >= skip_count:
                    skip_done = True
                    if skip_pbar is not None:
                        skip_pbar.close()
                    print(
                        f"[SFT] Skip complete: {skipped_valid} training example(s) "
                        f"(scanned {stats['raw']} raw row(s), "
                        f"dropped gate={stats['gate']} length={stats['length']})."
                    )
                continue
            to_format.append(raw)

        if not to_format:
            return

        for ex in format_records(to_format):
            stats["yielded"] += 1
            yield ex

    pending: deque = deque()
    worker_pool: concurrent.futures.ProcessPoolExecutor | None = None
    if num_workers > 1:
        worker_pool = concurrent.futures.ProcessPoolExecutor(max_workers=num_workers)

    def classify_batch(batch: list[dict]) -> list[str]:
        if worker_pool is not None:
            return worker_pool.submit(
                _classify_sft_messages_batch_worker,
                batch,
                tokenizer,
                use_tools,
                max_prompt_length,
            ).result()
        return _classify_sft_messages_batch(
            batch, tokenizer, use_tools, max_prompt_length,
        )

    try:
        batch: list[dict] = []
        for raw in raw_iter:
            batch.append(raw)
            if len(batch) < batch_size:
                continue
            if worker_pool is not None:
                pending.append((batch, worker_pool.submit(
                    _classify_sft_messages_batch_worker,
                    batch,
                    tokenizer,
                    use_tools,
                    max_prompt_length,
                )))
                if len(pending) > num_workers:
                    b, fut = pending.popleft()
                    yield from consume_classified(b, fut.result())
            else:
                yield from consume_classified(batch, classify_batch(batch))
            batch = []

        if batch:
            if worker_pool is not None:
                pending.append((batch, worker_pool.submit(
                    _classify_sft_messages_batch_worker,
                    batch,
                    tokenizer,
                    use_tools,
                    max_prompt_length,
                )))
            else:
                yield from consume_classified(batch, classify_batch(batch))

        while pending:
            b, fut = pending.popleft()
            yield from consume_classified(b, fut.result())
    finally:
        if worker_pool is not None:
            worker_pool.shutdown(wait=True)
        if skip_pbar is not None:
            skip_pbar.close()

    if skip_count > 0 and skipped_valid < skip_count:
        print(
            f"WARNING: stream ended after skipping {skipped_valid} valid example(s); "
            f"requested skip_count={skip_count}."
        )


def _streaming_sft_messages_dataset(
    raw_dataset: IterableDataset,
    tokenizer: AutoTokenizer,
    use_tools: bool,
    max_prompt_length: Optional[int],
    skip_buffer_size: int,
    skip_batch_size: int,
    skip_num_workers: int,
) -> IterableDataset:
    """IterableDataset with fused filter/format/skip for streaming SFT ``messages``."""
    skip_buffer_size = int(skip_buffer_size)
    skip_batch_size = max(1, int(skip_batch_size))
    skip_num_workers = max(1, int(skip_num_workers))

    if skip_buffer_size > 0:
        print(
            f"[SFT] Fast stream pipeline: skip first {skip_buffer_size} training example(s) "
            f"(batch_size={skip_batch_size}, num_workers={skip_num_workers}, "
            "system+user-only checks during skip)."
        )
    elif max_prompt_length is not None:
        print(
            f"[SFT] Fast stream pipeline: batched prompt filter "
            f"(batch_size={skip_batch_size}, num_workers={skip_num_workers})."
        )

    def _generator():
        yield from _sft_messages_stream(
            iter(raw_dataset),
            tokenizer=tokenizer,
            use_tools=use_tools,
            max_prompt_length=max_prompt_length,
            skip_count=skip_buffer_size,
            batch_size=skip_batch_size,
            num_workers=skip_num_workers,
        )

    return IterableDataset.from_generator(_generator)


# =====================================================================
# Dataset helpers
# =====================================================================

def process_batch(batch: list, tokenizer: AutoTokenizer, unique_by: str, max_prompt_length: int) -> list:
    """Helper function to tokenize and filter a batch of examples."""
    texts = [ex[unique_by] for ex in batch]

    encodings = tokenizer(
        texts,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )

    valid_examples = []
    for ex, length in zip(batch, map(len, encodings["input_ids"])):
        if length < max_prompt_length:
            valid_examples.append(ex)

    return valid_examples


def _example_field_key(example: dict, field: str) -> str:
    """Diversity key for an example.

    The GRPO ``netlist`` field is a ``{"doc_id": ..., "netlist": ...}`` payload
    (see :func:`conversation.convert_netlist_to_json_payload`), so prefer its
    ``doc_id``; otherwise stringify the field value.
    """
    val = example.get(field)
    if isinstance(val, dict):
        return str(val.get("doc_id") or val.get("id") or val.get("netlist") or val)
    return str(val)


DiversityStrategy = Literal["random", "round_robin", "even_spacing"]


def _grouped_shuffled(examples: list, field: str, seed: int) -> list:
    """Group examples by ``field`` key, shuffle within each group (seeded), and
    return the groups as a list of item-lists ordered largest-first (with the
    key as a deterministic tie-break)."""
    rng = random.Random(seed)
    groups: dict = {}
    for ex in examples:
        groups.setdefault(_example_field_key(ex, field), []).append(ex)
    ordered_keys = sorted(groups.keys(), key=lambda k: (-len(groups[k]), k))
    group_lists: list = []
    for k in ordered_keys:
        items = groups[k]
        rng.shuffle(items)  # randomise fault order within a netlist (seeded)
        group_lists.append(items)
    return group_lists


def _round_robin_interleave(group_lists: list) -> list:
    """One item per group in rotation (largest groups first). Maximises early
    diversity but a dominant group's tail clusters once smaller groups exhaust."""
    queues: deque = deque(deque(g) for g in group_lists)
    result: list = []
    while queues:
        q = queues.popleft()
        result.append(q.popleft())
        if q:
            queues.append(q)
    return result


def _even_spacing_interleave(group_lists: list) -> list:
    """Assign each item an evenly spaced fractional position ``(rank+0.5)/n`` in
    ``[0, 1)`` and sort globally, so every group's items are spread uniformly
    across the whole sequence. Keeps per-window diversity uniformly high (best
    average *and* minimum)."""
    decorated: list = []
    for gi, items in enumerate(group_lists):
        n = len(items)
        for rank, ex in enumerate(items):
            decorated.append(((rank + 0.5) / n, gi, ex))
    decorated.sort(key=lambda t: (t[0], t[1]))
    return [ex for _, _, ex in decorated]


def interleave_for_field_diversity(
    examples: list,
    field: str,
    strategy: DiversityStrategy = "even_spacing",
    seed: int = 42,
) -> list:
    """Deterministically reorder ``examples`` to control ``field`` diversity in
    every contiguous window. Pure function of ``(examples, field, strategy, seed)``.

    Strategies
    ----------
    ``"random"``
        Seeded global shuffle. Baseline; diversity per window is left to chance
        (≈ what you get today). No structural guarantee.
    ``"round_robin"``
        Cycle through netlist groups one item at a time (largest first). Early
        windows are maximally diverse, but once small groups run out a dominant
        netlist's remaining faults cluster at the tail → low-diversity windows
        there. Good only when group sizes are similar.
    ``"even_spacing"`` (default, recommended)
        Spread every netlist's faults uniformly across the epoch (fractional
        positions). Any window of length ``W`` then holds ~``W * group_size / N``
        of each netlist — as many distinct netlists as the class balance allows —
        consistently from start to end. Best average and minimum diversity.

    For GRPO this prevents effective batches dominated by 1–3 netlists' worth of
    faults, which otherwise make proxy/reward hacking and policy collapse easier.
    """
    if not examples:
        return examples

    if strategy == "random":
        result = list(examples)
        random.Random(seed).shuffle(result)
        return result

    group_lists = _grouped_shuffled(examples, field, seed)
    if strategy == "round_robin":
        return _round_robin_interleave(group_lists)
    if strategy == "even_spacing":
        return _even_spacing_interleave(group_lists)
    raise ValueError(
        f"Unknown diversity strategy {strategy!r}; "
        "choose 'random', 'round_robin', or 'even_spacing'."
    )


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
    maximize_diversity_by: Optional[Literal["netlist", "module_name"]] = None,
    diversity_strategy: DiversityStrategy = "even_spacing",
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
    maximize_diversity_by : {"netlist", "module_name"}, optional
        When set, deterministically reorder the buffered examples (via
        :func:`interleave_for_field_diversity`) so consecutive samples cycle
        through different values of this field — maximising netlist diversity
        within each effective batch. Replaces the random ``shuffle`` (the
        interleave already randomises within-group order using ``seed``). Pair
        with ``GRPOConfig(shuffle_dataset=False)`` so the trainer's sampler
        preserves this order.
    diversity_strategy : {"random", "round_robin", "even_spacing"}, default "even_spacing"
        Reordering algorithm used when ``maximize_diversity_by`` is set. See
        :func:`interleave_for_field_diversity` for the trade-offs.
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

    if maximize_diversity_by is not None:
        examples = interleave_for_field_diversity(
            examples, maximize_diversity_by, strategy=diversity_strategy, seed=seed
        )
        dataset = Dataset.from_list(examples)
        print(
            f"Reordered buffer for '{maximize_diversity_by}' diversity per window "
            f"(strategy='{diversity_strategy}', seed={seed}); random shuffle skipped. "
            "Set GRPOConfig(shuffle_dataset=False) to preserve this order."
        )
    else:
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


def skip_streaming_dataset(
    dataset: IterableDataset,
    skip_count: int = 0,
) -> IterableDataset:
    """Advance past the first ``skip_count`` examples in a streaming dataset.

    Each skipped row is one example the trainer would otherwise consume
    (after formatting and any upstream filters such as prompt-length).
    Use when resuming SFT on the
    same stream so training does not revisit samples from a prior run.

    For SFT ``messages`` training, prefer passing ``skip_buffer_size`` into
    :func:`format_dataset_for_training` (fast fused skip). This helper remains
    for already-formatted iterables and tests.

    Parameters
    ----------
    dataset : IterableDataset
        Formatted training stream (e.g. output of
        :func:`format_dataset_for_training`).
    skip_count : int, default 0
        Number of leading examples to drop. ``0`` returns ``dataset`` unchanged.
    """
    try:
        skip_count = int(skip_count)
    except (TypeError, ValueError) as e:
        raise ValueError("skip_count must be an integer") from e
    if skip_count < 0:
        raise ValueError("skip_count must be >= 0")
    if skip_count == 0:
        return dataset
    if not isinstance(dataset, IterableDataset):
        raise TypeError(
            f"skip_streaming_dataset requires an IterableDataset, got {type(dataset)!r}"
        )
    print(
        f"[SFT] Warning: lazy .skip({skip_count}) on a formatted stream is slow. "
        "Pass skip_buffer_size into format_dataset_for_training for the fast path."
    )
    return dataset.skip(skip_count)


def extract_prompt_messages(messages: list[dict]) -> list[dict]:
    """Return system + user turns only (exclude assistant / tool)."""
    return [m for m in messages if m.get("role") not in ("assistant", "tool")]


def count_prompt_tokens(
    tokenizer: AutoTokenizer,
    messages: list[dict],
    *,
    tools=None,
    add_generation_prompt: bool = True,
) -> int:
    """Token length of the system+user prompt, aligned with GRPO prompt sizing."""
    prompt_messages = extract_prompt_messages(messages)
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
    )
    return len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])


def filter_streaming_dataset_by_prompt_length(
    dataset: IterableDataset,
    tokenizer: AutoTokenizer,
    max_prompt_length: int,
) -> IterableDataset:
    """Drop ``messages`` examples whose system+user prompt is too long.

    Keeps rows with ``count_prompt_tokens(...) < max_prompt_length`` (same
    strict comparison as :func:`buffer_streaming_dataset` / GRPO).
    """
    try:
        max_prompt_length = int(max_prompt_length)
    except (TypeError, ValueError) as e:
        raise ValueError("max_prompt_length must be an integer") from e
    if max_prompt_length <= 0:
        raise ValueError("max_prompt_length must be > 0")
    if not isinstance(dataset, IterableDataset):
        raise TypeError(
            "filter_streaming_dataset_by_prompt_length requires an IterableDataset, "
            f"got {type(dataset)!r}"
        )

    def passes(example: dict) -> bool:
        tools = example.get("tools")
        return count_prompt_tokens(tokenizer, example["messages"], tools=tools) < max_prompt_length

    print(
        f"[SFT] Filtering stream: keep examples with system+user prompt "
        f"< {max_prompt_length} tokens (assistant / tool excluded)."
    )
    return dataset.filter(passes)


def format_dataset_for_training(
    dataset,
    tokenizer: AutoTokenizer,
    training_mode: str,
    sft_format: Literal["messages", "text"] = "messages",
    max_prompt_length: Optional[int] = None,
    skip_buffer_size: int = 0,
    skip_batch_size: int = 128,
    skip_num_workers: Optional[int] = None,
):
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
    sft_format : {"messages", "text"}
        Only used when ``training_mode == TrainingMode.SFT``:

        * ``"messages"`` (default) — emit a *conversational* dataset with
          ``messages`` (and ``tools`` when applicable). Required for
          ``SFTConfig(assistant_only_loss=True)``: SFTTrainer applies the
          chat template internally and produces ``assistant_masks`` so that
          loss is computed only on assistant tokens.
        * ``"text"`` — emit a pre-rendered ``text`` field. Legacy
          language-modeling format; loss is computed on every non-pad token
          (system / user / tool responses included). Use only when reverting
          to the old behavior for ablation.
    max_prompt_length : int, optional
        SFT ``messages`` only. After formatting, drop examples whose
        system+user chat-template token length is ``>= max_prompt_length``.
        Matches GRPO prompt sizing (see :func:`count_prompt_tokens`). Ignored
        for ``sft_format="text"`` — see module docstring on the text path.
    skip_buffer_size : int, default 0
        Streaming SFT ``messages`` only. Skip this many training examples
        (post gate filter and prompt-length filter) before yielding. Uses a
        batched multiprocessing pipeline when ``skip_buffer_size > 0`` or when
        ``max_prompt_length`` is set on a streaming dataset.
    skip_batch_size : int, default 128
        Raw rows per batch for the fast streaming SFT pipeline.
    skip_num_workers : int, optional
        Worker processes for batch classify/format. Default: ``min(16, cpu_count)``.

    Returns
    -------
    Dataset or IterableDataset
        A dataset ready for ``SFTTrainer`` or ``GRPOTrainer``. For GRPO the
        field is ``prompt``; for SFT it is ``messages`` or ``text`` per
        ``sft_format``. Same iterable/eager type as the input dataset.
    """
    is_streaming = isinstance(dataset, IterableDataset)

    use_tools = hasattr(tokenizer, 'chat_template') and tokenizer.chat_template and ('tools' in tokenizer.chat_template or 'tool' in tokenizer.chat_template)

    if skip_num_workers is None:
        skip_num_workers = _default_skip_num_workers()

    if training_mode == TrainingMode.SFT:
        if sft_format == "messages":
            use_fast_stream = is_streaming and (
                skip_buffer_size > 0 or max_prompt_length is not None
            )
            if use_fast_stream:
                return _streaming_sft_messages_dataset(
                    dataset,
                    tokenizer,
                    use_tools,
                    max_prompt_length,
                    skip_buffer_size=skip_buffer_size,
                    skip_batch_size=skip_batch_size,
                    skip_num_workers=skip_num_workers,
                )

            def filter_fn(record):
                return _gate_filter_passes(record)

            dataset = dataset.filter(filter_fn)

            def format_fn(record):
                convo = ConversationExample.from_record(record, use_tools=use_tools)
                out = {"messages": convo.messages}
                if use_tools:
                    out["tools"] = TOOLS
                return out

            mapped = dataset.map(format_fn)
            if max_prompt_length is not None:
                mapped = filter_streaming_dataset_by_prompt_length(
                    mapped, tokenizer, max_prompt_length,
                )
            if skip_buffer_size > 0:
                mapped = skip_streaming_dataset(mapped, skip_buffer_size)
            return mapped

        elif sft_format == "text":
            if max_prompt_length is not None:
                print(
                    "[SFT] max_prompt_length is not applied for sft_format='text'. "
                    "Use sft_format='messages' (assistant_only_loss=True) or pre-filter "
                    "the dataset."
                )

            def filter_fn(record):
                return _gate_filter_passes(record)

            dataset = dataset.filter(filter_fn)

            def format_fn(record):
                convo = ConversationExample.from_record(record, use_tools=use_tools)
                prompt = tokenizer.apply_chat_template(
                    convo.messages, tokenize=False,
                    tools=TOOLS if use_tools else None,
                )
                return {"text": prompt}

            return dataset.map(format_fn)

        else:
            raise ValueError(
                f"Unknown sft_format: {sft_format!r}. Expected 'messages' or 'text'."
            )

    elif training_mode == TrainingMode.GRPO:
        def filter_fn(record):
            return _gate_filter_passes(record)

        dataset = dataset.filter(filter_fn)

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