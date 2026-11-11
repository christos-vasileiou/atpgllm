#!/usr/bin/env python
"""Circuit kinds per optimizer step, measured on the real training order (dataset v5).

    python scripts/dataset/check_kind_mix.py --dataset ../data/freeset/dataset.freeset.asap7sc7p5t_28.rvt.tt.v5 \\
        --sft_steps 20 --out runs/v5_schedule/eval/kind_mix.json

SFT: the train stream in ShardPermutation order (what SFT_CIRCUIT_VALIDATION=True
reads), with the SFT filters (gate filter, prompt below the window, whole
conversation within max_model_len), grouped into steps of ``--sft_step_rows``.
GRPO: the whole first chunk of the real rolling buffer (``_rolling_grpo_buffer``, with
the config's difficulty weights), grouped into steps of ``--grpo_step_prompts``.
Reports the per-step share of each kind and the smallest count per step.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def summary(steps):
    kinds = sorted({k for s in steps for k in s})
    return {
        "steps": len(steps),
        "rows_per_step": [sum(s.values()) for s in steps][:5],
        "min_per_step": {k: min(s.get(k, 0) for s in steps) for k in kinds},
        "mean_share": {k: sum(s.get(k, 0) / max(1, sum(s.values())) for s in steps) / len(steps) for k in kinds},
        "steps_missing_a_kind": sum(any(s.get(k, 0) == 0 for k in kinds) for s in steps),
    }


def sft_steps(dataset, tokenizer, steps, step_rows, max_prompt_length, max_model_len):
    from atpgllm.training.conversation import ConversationExample
    from atpgllm.training.dataset_utils import _gate_filter_passes
    from atpgllm.training.kinds import row_kind
    from atpgllm.training.sft_validation import load_circuit_train
    from atpgllm.training.tools import TOOLS
    kept, dropped, out, current = 0, Counter(), [], Counter()
    for row in load_circuit_train(dataset):
        if not _gate_filter_passes(row):
            dropped[(row_kind(row), "gate")] += 1
            continue
        messages = ConversationExample.from_record(dict(row), use_tools=True).messages
        prompt = [m for m in messages if m["role"] in ("system", "user")]
        n_prompt = len(tokenizer.apply_chat_template(prompt, tools=TOOLS, tokenize=True, add_generation_prompt=True))
        if n_prompt >= max_prompt_length:
            dropped[(row_kind(row), "prompt")] += 1
            continue
        if len(tokenizer.apply_chat_template(messages, tools=TOOLS, tokenize=True)) > max_model_len:
            dropped[(row_kind(row), "too_long")] += 1
            continue
        current[row_kind(row)] += 1
        kept += 1
        if kept % step_rows == 0:
            out.append(current)
            current = Counter()
            if len(out) == steps:
                break
    return summary(out), {f"{k}/{why}": n for (k, why), n in sorted(dropped.items())}


def grpo_steps(dataset, tokenizer, buffer_size, step_prompts, max_prompt_length, weights):
    import importlib.util
    spec = importlib.util.spec_from_file_location("training_code", REPO / "scripts/train/training_code.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from atpgllm.training.kinds import row_kind
    from atpgllm.training.stream_buffer import parse_difficulty_weights
    with tempfile.TemporaryDirectory() as tmp:
        buffer, _ = module._rolling_grpo_buffer(
            dataset, tokenizer, output_dir=tmp, resume_checkpoint=None, buffer_size=buffer_size,
            max_prompt_length=max_prompt_length, netlist_diversity_strategy="even_spacing",
            held_out_examples=[], prompts_per_step=step_prompts, max_steps=40, is_main_process=True,
            difficulty_weights=parse_difficulty_weights(weights) if weights else None)
        rows = buffer.chunk(0)
    out = [Counter(row_kind(r) for r in rows[i:i + step_prompts]) for i in range(0, len(rows), step_prompts)]
    levels = Counter((row_kind(r), r.get("difficulty")) for r in rows)
    return summary(out), {f"{k}/{lvl}": n for (k, lvl), n in sorted(levels.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tokenizer", default="ibm-granite/granite-4.2-8b")
    parser.add_argument("--sft_steps", type=int, default=20)
    parser.add_argument("--sft_step_rows", type=int, default=1024, help="4 GPUs x 1 x 256 accumulation")
    parser.add_argument("--sft_max_model_len", type=int, default=24576)
    parser.add_argument("--grpo_buffer_size", type=int, default=20000, help="BUFFER_SIZE; every step of the first chunk is measured")
    parser.add_argument("--grpo_step_prompts", type=int, default=72, help="2 ranks x 576 / 16 generations")
    parser.add_argument("--max_prompt_length", type=int, default=14336)
    parser.add_argument("--difficulty_weights", default="easy=1,medium=13,hard=16,very_hard=40,random_never_detects=40")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    result = {"dataset": str(Path(args.dataset).resolve())}
    if args.sft_steps:
        result["sft"], result["sft_dropped"] = sft_steps(args.dataset, tokenizer, args.sft_steps, args.sft_step_rows,
                                                         args.max_prompt_length, args.sft_max_model_len)
        print(json.dumps({"sft": result["sft"], "sft_dropped": result["sft_dropped"]}, indent=1), flush=True)
    if args.grpo_buffer_size:
        result["grpo"], result["grpo_levels"] = grpo_steps(args.dataset, tokenizer, args.grpo_buffer_size,
                                                           args.grpo_step_prompts, args.max_prompt_length,
                                                           args.difficulty_weights)
        print(json.dumps({"grpo": result["grpo"], "grpo_levels": result["grpo_levels"]}, indent=1), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
