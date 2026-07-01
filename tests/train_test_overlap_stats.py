#!/usr/bin/env python3
"""Train vs test overlap statistics for buffered SFT / GRPO / eval runs.

Scans the streaming dataset and applies the same gate / prompt-length filters
used during training (via :func:`format_dataset_for_training` logic). Window
sizes default to the 7B exper1 setup:

* SFT — first 204_800 valid train examples after gate + ``max_prompt_length``
  (``cumulative_skip_buffer_size`` from ``sft_7b_exper1/checkpoint-200``)
* GRPO — either skip SFT-consumed valid prompts then buffer 10_000 (resume
  after SFT), or buffer 10_000 from stream start with a different prompt limit
  (``--no-grpo-skip-sft-prompts``)
* Test eval — first 512 valid test examples with prompt below ``test_max_prompt_size``

GRPO uniqueness and skip semantics match :func:`buffer_streaming_dataset``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
import sys
from collections import deque
from collections.abc import Iterator
from typing import Optional

from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from conversation import ConversationExample
from dataset_utils import (
    _default_skip_num_workers,
    _gate_filter_passes,
)
from tools import TOOLS

ProblemKey = tuple[str, str]
ClassifiedRow = tuple[str, str, bool, bool, str]  # netlist, fault, sft_ok, grpo_ok, prompt


# Worker globals — tokenizer is loaded once per process, not pickled per batch.
_worker_tokenizer: AutoTokenizer | None = None
_worker_use_tools: bool = False
_worker_sft_max_prompt_size: int = 0
_worker_grpo_max_prompt_size: int = 0


def _init_overlap_worker(
    model_name: str,
    use_tools: bool,
    sft_max_prompt_size: int,
    grpo_max_prompt_size: int,
) -> None:
    global _worker_tokenizer, _worker_use_tools
    global _worker_sft_max_prompt_size, _worker_grpo_max_prompt_size
    _worker_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    _worker_use_tools = use_tools
    _worker_sft_max_prompt_size = sft_max_prompt_size
    _worker_grpo_max_prompt_size = grpo_max_prompt_size


def _overlap_classify_batch_worker(batch: list[dict]) -> list[Optional[ClassifiedRow]]:
    assert _worker_tokenizer is not None
    return _overlap_classify_batch(
        batch,
        _worker_tokenizer,
        _worker_use_tools,
        _worker_sft_max_prompt_size,
        _worker_grpo_max_prompt_size,
    )


def problem_key(record: dict) -> ProblemKey:
    return record["netlist"], record["fault"]


def unique_netlists(keys: set[ProblemKey]) -> set[str]:
    return {netlist for netlist, _ in keys}


def _overlap_classify_batch(
    batch: list[dict],
    tokenizer: AutoTokenizer,
    use_tools: bool,
    sft_max_prompt_size: int,
    grpo_max_prompt_size: int,
) -> list[Optional[ClassifiedRow]]:
    """Gate-filter and batch-tokenize one chunk of raw rows.

    Builds the system+user prompt once per row (same template as
    ``format_dataset_for_training``) and applies SFT / GRPO length thresholds
    from a single tokenization pass.
    """
    tools = TOOLS if use_tools else None
    results: list[Optional[ClassifiedRow]] = [None] * len(batch)
    candidates: list[tuple[int, str, str, list]] = []

    for index, raw in enumerate(batch):
        if not _gate_filter_passes(raw):
            continue
        record = copy.deepcopy(raw)
        try:
            prompt_messages = ConversationExample.prompt_messages_from_record(
                record, use_tools=use_tools,
            )
        except Exception:
            continue
        if not prompt_messages:
            continue
        candidates.append((index, raw["netlist"], raw["fault"], prompt_messages))

    if not candidates:
        return results

    prompt_texts = [
        tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
        )
        for _, _, _, messages in candidates
    ]
    encodings = tokenizer(
        prompt_texts,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    for (index, netlist, fault, _), token_ids, prompt_text in zip(
        candidates, encodings["input_ids"], prompt_texts,
    ):
        token_len = len(token_ids)
        results[index] = (
            netlist,
            fault,
            token_len < sft_max_prompt_size,
            token_len < grpo_max_prompt_size,
            prompt_text,
        )
    return results


def _batched_stream_classify(
    split,
    *,
    model_name: str,
    tokenizer: AutoTokenizer,
    use_tools: bool,
    sft_max_prompt_size: int,
    grpo_max_prompt_size: int,
    batch_size: int,
    num_workers: int,
) -> Iterator[tuple[list[dict], list[Optional[ClassifiedRow]]]]:
    """Yield ``(batch, classifications)`` in stream order with optional worker pool."""
    batch_size = max(1, int(batch_size))
    num_workers = max(1, int(num_workers))

    pending: deque[
        tuple[list[dict], concurrent.futures.Future[list[Optional[ClassifiedRow]]]]
    ] = deque()
    worker_pool: concurrent.futures.ProcessPoolExecutor | None = None
    if num_workers > 1:
        worker_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_overlap_worker,
            initargs=(model_name, use_tools, sft_max_prompt_size, grpo_max_prompt_size),
        )

    def submit(batch: list[dict]) -> concurrent.futures.Future[list[Optional[ClassifiedRow]]]:
        if worker_pool is not None:
            return worker_pool.submit(_overlap_classify_batch_worker, batch)
        fut: concurrent.futures.Future[list[Optional[ClassifiedRow]]] = (
            concurrent.futures.Future()
        )
        fut.set_result(
            _overlap_classify_batch(
                batch, tokenizer, use_tools, sft_max_prompt_size, grpo_max_prompt_size,
            )
        )
        return fut

    def drain_one() -> Iterator[tuple[list[dict], list[Optional[ClassifiedRow]]]]:
        batch, future = pending.popleft()
        yield batch, future.result()

    try:
        batch: list[dict] = []
        for record in split:
            batch.append(record)
            if len(batch) < batch_size:
                continue
            pending.append((batch, submit(batch)))
            batch = []
            if len(pending) > num_workers:
                yield from drain_one()

        if batch:
            pending.append((batch, submit(batch)))

        while pending:
            yield from drain_one()
    finally:
        if worker_pool is not None:
            worker_pool.shutdown(wait=True)


def _prompt_fingerprint(prompt: str) -> bytes:
    return hashlib.sha256(prompt.encode("utf-8")).digest()


def collect_filtered_train_keys(
    split,
    *,
    model_name: str,
    tokenizer: AutoTokenizer,
    use_tools: bool,
    sft_rows: int,
    grpo_rows: int,
    sft_max_prompt_size: int,
    grpo_max_prompt_size: int,
    grpo_skip_sft_prompts: bool,
    batch_size: int,
    num_workers: int,
) -> dict[str, set[ProblemKey]]:
    """Collect unique ``(netlist, fault)`` keys for SFT / GRPO training windows."""
    sft_keys: set[ProblemKey] = set()
    grpo_keys: set[ProblemKey] = set()
    train_keys: set[ProblemKey] = set()

    sft_valid = 0
    grpo_skip_left = sft_rows if grpo_skip_sft_prompts else 0
    grpo_buffered = 0
    seen_grpo_prompts: set[bytes] = set()

    pbar = tqdm(desc="Train split", unit="raw", file=sys.stdout)
    try:
        for _batch, classifications in _batched_stream_classify(
            split,
            model_name=model_name,
            tokenizer=tokenizer,
            use_tools=use_tools,
            sft_max_prompt_size=sft_max_prompt_size,
            grpo_max_prompt_size=grpo_max_prompt_size,
            batch_size=batch_size,
            num_workers=num_workers,
        ):
            pbar.update(len(_batch))
            for row in classifications:
                if row is None:
                    continue
                netlist, fault, sft_ok, grpo_ok, prompt = row
                key = (netlist, fault)

                if sft_valid < sft_rows and sft_ok:
                    sft_keys.add(key)
                    train_keys.add(key)
                    sft_valid += 1

                if grpo_buffered < grpo_rows and grpo_ok:
                    prompt_id = _prompt_fingerprint(prompt)
                    if prompt_id in seen_grpo_prompts:
                        continue
                    seen_grpo_prompts.add(prompt_id)
                    if grpo_skip_left > 0:
                        grpo_skip_left -= 1
                    else:
                        grpo_keys.add(key)
                        train_keys.add(key)
                        grpo_buffered += 1

            if sft_valid >= sft_rows and grpo_buffered >= grpo_rows:
                break

            pbar.set_postfix(sft=sft_valid, grpo=grpo_buffered, refresh=False)
    finally:
        pbar.close()

    return {"sft": sft_keys, "grpo": grpo_keys, "train": train_keys}


def collect_filtered_test_keys(
    split,
    *,
    model_name: str,
    tokenizer: AutoTokenizer,
    use_tools: bool,
    test_rows: int,
    test_max_prompt_size: int,
    batch_size: int,
    num_workers: int,
) -> set[ProblemKey]:
    """First ``test_rows`` valid examples with prompt below ``test_max_prompt_size``."""
    keys: set[ProblemKey] = set()
    valid = 0

    pbar = tqdm(desc="Test split", unit="raw", file=sys.stdout)
    try:
        for _batch, classifications in _batched_stream_classify(
            split,
            model_name=model_name,
            tokenizer=tokenizer,
            use_tools=use_tools,
            sft_max_prompt_size=test_max_prompt_size,
            grpo_max_prompt_size=test_max_prompt_size,
            batch_size=batch_size,
            num_workers=num_workers,
        ):
            pbar.update(len(_batch))
            for row in classifications:
                if row is None:
                    continue
                netlist, fault, _sft_ok, grpo_ok, _prompt = row
                if not grpo_ok:
                    continue
                keys.add((netlist, fault))
                valid += 1
                if valid >= test_rows:
                    pbar.set_postfix(valid=valid, refresh=False)
                    return keys
            pbar.set_postfix(valid=valid, refresh=False)
    finally:
        pbar.close()

    return keys


def read_cumulative_skip(checkpoint_dir: str) -> int | None:
    summary_path = os.path.join(checkpoint_dir, "training_state_summary.json")
    if not os.path.isfile(summary_path):
        return None
    with open(summary_path, encoding="utf-8") as handle:
        return json.load(handle).get("cumulative_skip_buffer_size")


def pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{100.0 * numerator / denominator:.1f}%"


def print_overlap_report(
    *,
    sft: set[ProblemKey],
    grpo: set[ProblemKey],
    train: set[ProblemKey],
    test: set[ProblemKey],
    sft_rows: int,
    grpo_rows: int,
    test_rows: int,
    sft_max_prompt_size: int,
    grpo_max_prompt_size: int,
    test_max_prompt_size: int,
    grpo_skip_sft_prompts: bool,
) -> None:
    sft_netlists = unique_netlists(sft)
    grpo_netlists = unique_netlists(grpo)
    train_netlists = unique_netlists(train)
    test_netlists = unique_netlists(test)

    common_netlists = test_netlists & train_netlists
    common_problems = test & train
    test_only_in_sft = test & sft
    test_only_in_grpo = test & grpo

    print("=== Train / test overlap (unique netlist + fault) ===")
    print("Training windows (valid post-filter examples):")
    print(f"  SFT train:  first {sft_rows:,} valid rows (prompt < {sft_max_prompt_size} tokens)")
    if grpo_skip_sft_prompts:
        print(
            f"  GRPO train: skip {sft_rows:,} valid unique prompts, "
            f"then {grpo_rows:,} more (prompt < {grpo_max_prompt_size} tokens)"
        )
    else:
        print(
            f"  GRPO train: first {grpo_rows:,} valid unique prompts from stream start "
            f"(prompt < {grpo_max_prompt_size} tokens; no SFT skip)"
        )
    print(
        f"  Test eval:  first {test_rows:,} valid rows "
        f"(prompt < {test_max_prompt_size} tokens)"
    )
    print()
    print(f"{'Split':<14} {'Problems':>10} {'Netlists':>10}")
    print(f"{'SFT':<14} {len(sft):>10,} {len(sft_netlists):>10,}")
    print(f"{'GRPO':<14} {len(grpo):>10,} {len(grpo_netlists):>10,}")
    print(f"{'Train total':<14} {len(train):>10,} {len(train_netlists):>10,}")
    print(f"{'Test':<14} {len(test):>10,} {len(test_netlists):>10,}")
    print()
    print("Overlap with test:")
    print(f"  Problems also in train: {len(common_problems):>6}  ({pct(len(common_problems), len(test))} of test)")
    print(f"  Netlists also in train: {len(common_netlists):>6}  ({pct(len(common_netlists), len(test_netlists))} of test netlists)")
    print(f"  Problems also in SFT:   {len(test_only_in_sft):>6}  ({pct(len(test_only_in_sft), len(test))} of test)")
    print(f"  Problems also in GRPO:  {len(test_only_in_grpo):>6}  ({pct(len(test_only_in_grpo), len(test))} of test)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report train vs test overlap for buffered training windows.",
    )
    parser.add_argument(
        "--dataset",
        default=os.environ.get("TRAIN_DATASET", "chrivasileiou/asap7-language-of-test-v2"),
        help="HuggingFace dataset id (env: TRAIN_DATASET)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        help="Tokenizer model for prompt-length checks (env: MODEL)",
    )
    parser.add_argument(
        "--sft-rows",
        type=int,
        default=204_800,
        help="Valid SFT training examples to include (default: 204_800)",
    )
    parser.add_argument(
        "--grpo-rows",
        type=int,
        default=10_000,
        help="Valid GRPO training examples to buffer (default: 10_000)",
    )
    parser.add_argument(
        "--test-rows",
        type=int,
        default=512,
        help="Valid test examples to include after filtering (default: 512)",
    )
    parser.add_argument(
        "--sft-max-prompt-size",
        type=int,
        default=2048,
        help="SFT max_prompt_length (default: 2048)",
    )
    parser.add_argument(
        "--grpo-max-prompt-size",
        type=int,
        default=4096,
        help="GRPO max_prompt_length (default: 4096)",
    )
    parser.add_argument(
        "--test-max-prompt-size",
        type=int,
        default=4096,
        help="Test eval max_prompt_length (default: 4096)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Raw rows per tokenization batch (default: 1024)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Worker processes for batch classification (default: min(16, cpu_count))",
    )
    parser.add_argument(
        "--sft-checkpoint",
        metavar="DIR",
        help="Use cumulative_skip_buffer_size from DIR/training_state_summary.json as --sft-rows",
    )
    parser.add_argument(
        "--grpo-skip-sft-prompts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="GRPO: skip valid prompts already consumed by SFT before buffering "
        "(matches GRPO with --skip_buffer_size from SFT checkpoint). "
        "Use --no-grpo-skip-sft-prompts when GRPO starts at stream offset 0 "
        "with a different max_prompt_length.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sft_rows = args.sft_rows
    if args.sft_checkpoint:
        derived = read_cumulative_skip(args.sft_checkpoint)
        if derived is None:
            raise SystemExit(
                f"No cumulative_skip_buffer_size in {args.sft_checkpoint}/training_state_summary.json"
            )
        sft_rows = derived
        print(f"Using SFT valid-example window {sft_rows:,} from {args.sft_checkpoint}")

    num_workers = args.num_workers if args.num_workers is not None else _default_skip_num_workers()

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    use_tools = bool(
        getattr(tokenizer, "chat_template", None)
        and ("tools" in tokenizer.chat_template or "tool" in tokenizer.chat_template)
    )

    ds = load_dataset(args.dataset, streaming=True)
    scan_kwargs = dict(
        model_name=args.model,
        tokenizer=tokenizer,
        use_tools=use_tools,
        batch_size=args.batch_size,
        num_workers=num_workers,
    )
    print(
        "Scanning train split "
        f"(batch={args.batch_size}, workers={num_workers}, "
        f"SFT prompt < {args.sft_max_prompt_size}, GRPO prompt < {args.grpo_max_prompt_size}, "
        f"GRPO skip SFT prompts={'yes' if args.grpo_skip_sft_prompts else 'no'})..."
    )
    train_keys = collect_filtered_train_keys(
        ds["train"],
        sft_rows=sft_rows,
        grpo_rows=args.grpo_rows,
        sft_max_prompt_size=args.sft_max_prompt_size,
        grpo_max_prompt_size=args.grpo_max_prompt_size,
        grpo_skip_sft_prompts=args.grpo_skip_sft_prompts,
        **scan_kwargs,
    )

    print(
        f"Scanning test split (prompt < {args.test_max_prompt_size})..."
    )
    test_keys = collect_filtered_test_keys(
        ds["test"],
        test_rows=args.test_rows,
        test_max_prompt_size=args.test_max_prompt_size,
        **scan_kwargs,
    )

    print_overlap_report(
        sft=train_keys["sft"],
        grpo=train_keys["grpo"],
        train=train_keys["train"],
        test=test_keys,
        sft_rows=sft_rows,
        grpo_rows=args.grpo_rows,
        test_rows=args.test_rows,
        sft_max_prompt_size=args.sft_max_prompt_size,
        grpo_max_prompt_size=args.grpo_max_prompt_size,
        test_max_prompt_size=args.test_max_prompt_size,
        grpo_skip_sft_prompts=args.grpo_skip_sft_prompts,
    )


if __name__ == "__main__":
    main()
