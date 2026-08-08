#!/usr/bin/env python3
"""Dump the per-netlist fault-count distribution of the GRPO training buffer.

This reuses the *exact* GRPO data path used by ``training_code.train_with_grpo``:

    load_dataset(streaming) -> format_dataset_for_training(..., GRPO)
                            -> buffer_streaming_dataset(...)

so the resulting set matches what the model actually trains on:
  * the gate-count filter (``format_dataset_for_training`` GRPO branch),
  * the tokenized system+user **prompt-length** filter (``< max_prompt_length``),
  * uniqueness by ``prompt`` and the ``skip_buffer_size`` / ``buffer_size`` quotas.

The output is a JSON list of fault-counts per netlist, sorted descending — the
input format expected by ``benchmark_netlist_diversity.py --real``.

Example (matches grpo_7b_exper1 settings):
    python dump_netlist_distribution.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --dataset chrivasileiou/asap7-language-of-test-v2 \
        --buffer_size 10000 --max_prompt_length 4096 --skip_buffer_size 0 \
        --output netlist_dist_filtered.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter

from datasets import load_dataset
from transformers import AutoTokenizer

from atpgllm.training.dataset_utils import (
    TrainingMode,
    buffer_streaming_dataset,
    format_dataset_for_training,
)


def _netlist_doc_id(netlist_field) -> str:
    """Match conversation.convert_netlist_to_json_payload: doc_id is the netlist
    identity (sha256[:16] of the netlist text, or the carried doc_id)."""
    if isinstance(netlist_field, dict):
        doc_id = netlist_field.get("doc_id") or netlist_field.get("id")
        if doc_id:
            return str(doc_id)
        netlist_field = netlist_field.get("netlist", "")
    if not isinstance(netlist_field, str):
        netlist_field = str(netlist_field)
    return hashlib.sha256(netlist_field.encode("utf-8")).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                    help="Tokenizer source (must match training for prompt-length parity). "
                         "Can also point at the SFT checkpoint dir.")
    ap.add_argument("--dataset", default="chrivasileiou/asap7-language-of-test-v2")
    ap.add_argument("--split", default="train")
    ap.add_argument("--buffer_size", type=int, default=10000)
    ap.add_argument("--max_prompt_length", type=int, default=4096)
    ap.add_argument("--skip_buffer_size", type=int, default=0)
    ap.add_argument("--output", default="netlist_dist_filtered.json")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    data = load_dataset(args.dataset, split=args.split, streaming=True)
    formatted = format_dataset_for_training(data, tokenizer, TrainingMode.GRPO)

    # Identical call to training_code.train_with_grpo (diversity reorder omitted:
    # it changes order, not the buffered *set*).
    ds = buffer_streaming_dataset(
        formatted,
        buffer_size=args.buffer_size,
        shuffle=False,
        seed=42,
        tokenizer=tokenizer,
        max_prompt_length=args.max_prompt_length,
        skip_buffer_size=args.skip_buffer_size,
    )

    counts = Counter()
    for ex in ds:
        counts[_netlist_doc_id(ex.get("netlist"))] += 1

    dist = sorted(counts.values(), reverse=True)
    json.dump(dist, open(args.output, "w"))

    N = sum(dist)
    print(f"\nWrote {args.output}")
    print(f"buffered prompts (post gate+length filter): {N}")
    print(f"unique netlists: {len(dist)}")
    print(f"top-15 fault counts/netlist: {dist[:15]}")
    if dist:
        print(f"max={dist[0]}  median={dist[len(dist)//2]}  min={dist[-1]}")
        print(f"largest share: {dist[0]/N*100:.1f}%   "
              f"top-3: {sum(dist[:3])/N*100:.1f}%   top-10: {sum(dist[:10])/N*100:.1f}%")


if __name__ == "__main__":
    main()
