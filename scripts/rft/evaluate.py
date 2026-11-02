#!/usr/bin/env python
"""Greedy detection of one checkpoint on a fixed problem set, through the evaluator path.

Two sets are supported, and both are scored by TetraMAX on the final answer:

* ``--manifest``: the GDPO fixed evaluation (``fixed_eval_manifest.json``,
  72 validation faults with rendered prompts), so RFT rounds, SFT and GDPO
  checkpoints are compared on the same faults with the same verifier.
* ``--problems_file``: a stored selection from ``rationalize.py --split
  validation --write_problems``, split by difficulty level.

Greedy decoding and a 32,768 token context with a 24,576 token budget follow
the GDPO fixed-evaluation protocol. ``--shard`` processes (one per GPU)
write ``slots.shard<i>.jsonl``; ``--summarize`` merges them into
``metrics.json``. A rerun continues where the last one stopped.

    python scripts/rft/evaluate.py --adapter <checkpoint> --manifest <json> --out <dir> --shard 0 --num_shards 4
    python scripts/rft/evaluate.py --out <dir> --summarize
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "star"))
sys.path.insert(0, str(REPO / "scripts" / "eval"))


def manifest_items(path):
    """``(example id, level, rendered prompt, record)`` for a GDPO fixed-eval manifest."""
    manifest = json.loads(Path(path).read_text())
    # A stratified manifest says each problem's level in a sidecar file.
    sidecar = Path(str(path) + ".reference.json")
    reference = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    for example in manifest["examples"]:
        record = dict(example)
        level = reference.get(example["_fixed_eval_id"], {}).get("level", "fixed")
        yield example["_fixed_eval_id"], level, example["prompt"], record


def problem_items(path, tokenizer):
    """``(example id, level, rendered prompt, record)`` for a stored selection."""
    import pyarrow.parquet as pq
    from atpgllm.training.conversation import ConversationExample
    from atpgllm.training.rationalization import problem_key
    from atpgllm.training.tools import TOOLS
    for row in pq.read_table(path).to_pylist():
        messages = ConversationExample.prompt_messages_from_record(dict(row), use_tools=True)
        prompt = tokenizer.apply_chat_template(messages, tools=TOOLS, tokenize=False, add_generation_prompt=True)
        yield "|".join(problem_key(row)), row["difficulty"], prompt, {**row, "_search_messages": messages}


def _line_id(line):
    try:
        return json.loads(line)["example_id"]
    except (json.JSONDecodeError, KeyError):
        return None


def detected(slot):
    return bool((slot.get("reward_components") or {}).get("detection", 0) >= 1.0)


def summarize(out):
    slots = []
    for path in sorted(Path(out).glob("slots*.jsonl")):
        for line in open(path):
            try:
                slots.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    by_level = defaultdict(list)
    for slot in slots:
        by_level[slot["level"]].append(slot)

    def rates(group):
        n = len(group)
        return {"n": n, "detection": sum(map(detected, group)) / n,
                "final": sum(s.get("status") == "FINAL" for s in group) / n,
                "truncated": sum(s.get("status") == "EXHAUSTED" for s in group) / n}

    metrics = {"all": rates(slots), "by_level": {level: rates(g) for level, g in sorted(by_level.items())}}
    levels = [m["detection"] for m in metrics["by_level"].values()]
    # Mean over levels, so a stratified set is not dominated by any one level.
    metrics["level_mean_detection"] = sum(levels) / len(levels)
    run = json.loads((Path(out) / "run.json").read_text()) if (Path(out) / "run.json").exists() else {}
    metrics["adapter"] = run.get("adapter")
    (Path(out) / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps(metrics, indent=1))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", help="Checkpoint to evaluate (an SFT checkpoint or a GDPO policy/ adapter)")
    parser.add_argument("--manifest", help="GDPO fixed_eval_manifest.json")
    parser.add_argument("--problems_file", help="Stored validation selection (parquet)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--max_tool_rounds", type=int, default=16)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--max_generated_tokens", type=int, default=24576)
    parser.add_argument("--batch_problems", type=int, default=128)
    parser.add_argument("--tool_workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--num_samples", type=int, default=1, help="Slots per problem (direct sampling above 1)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None, help="Only the first N examples (smoke tests)")
    parser.add_argument("--summarize", action="store_true", help="Merge the shard files into metrics.json")
    args = parser.parse_args()
    out = Path(args.out)
    if args.summarize:
        summarize(out)
        return
    if not args.adapter or bool(args.manifest) == bool(args.problems_file):
        parser.error("--adapter and exactly one of --manifest or --problems_file are required")
    out.mkdir(parents=True, exist_ok=True)

    from rationalize import build_strategy
    strategy, tokenizer, info = build_strategy(
        args.adapter, num_samples=args.num_samples, temperature=args.temperature, top_p=args.top_p,
        max_tool_rounds=args.max_tool_rounds,
        max_model_len=args.max_model_len, max_generated_tokens=args.max_generated_tokens, seed=args.seed,
        tool_workers=args.tool_workers)
    items = list(manifest_items(args.manifest) if args.manifest else problem_items(args.problems_file, tokenizer))
    items = items[:args.limit][args.shard::args.num_shards]
    name = out / ("slots.jsonl" if args.num_shards == 1 else f"slots.shard{args.shard}.jsonl")
    seen = set()
    if name.exists():
        counts = defaultdict(int)
        for line in open(name):
            try:
                counts[json.loads(line)["example_id"]] += 1
            except json.JSONDecodeError:
                pass
        # A problem is done only with all its slots; drop a torn last line and a partial problem.
        seen = {example_id for example_id, count in counts.items() if count >= args.num_samples}
        lines = [line for line in open(name) if line.endswith("\n") and _line_id(line) in seen]
        name.write_text("".join(lines))
    items = [item for item in items if item[0] not in seen]
    if args.shard == 0:
        (out / "run.json").write_text(json.dumps({"args": vars(args), "adapter": str(Path(args.adapter).resolve()),
                                                  **info}, indent=1, default=str))
    print(f"[eval] {len(items)} examples left in shard {args.shard}", flush=True)
    with open(name, "a") as sink:
        for start in range(0, len(items), args.batch_problems):
            batch = items[start:start + args.batch_problems]
            t = time.time()
            results = strategy.sample_batch([p for _, _, p, _ in batch], [r for _, _, _, r in batch])
            for (example_id, level, _, _), result in zip(batch, results):
                for slot in result.slots:
                    sink.write(json.dumps({"example_id": example_id, "level": level,
                                           "completion_slot": slot.get("completion_slot", 0),
                                           "status": slot.get("status"), "stop_reason": slot.get("stop_reason"),
                                           "reward_components": slot.get("reward_components"),
                                           "vector": slot.get("vector"), "final_answer": slot.get("final_answer"),
                                           "usage": slot.get("usage"), "messages": slot.get("messages"),
                                           "observations": slot.get("observations")},
                                          default=str) + "\n")
            sink.flush()
            print(f"[eval] {start + len(batch)}/{len(items)} in {time.time() - t:.0f} s", flush=True)
    (out / f"{name.name}.done").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))


if __name__ == "__main__":
    main()
