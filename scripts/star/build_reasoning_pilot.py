#!/usr/bin/env python
"""Rows for the reasoning pilot: gate by gate demonstrations and real retries.

    python scripts/star/build_reasoning_pilot.py --out runs/reasoning_pilot/sft.jsonl

Writes ``{"messages", "tools"}`` rows for ``SFT_EXTRA_MESSAGES``.

* Demonstrations: train rows with the dataset's reasoning replaced by
  :func:`reasoning_trace.render_trace`, balanced over difficulty levels.
* Retries: the model's own failed attempt from a stored RFT generation, kept as
  it was with its real simulator result, then a turn that reads the result and
  reasons again, a tool call on the dataset's vector, and the final answer. The
  failure is the model's own, so these rows teach what to do after a negative
  result without adding wrong vectors that the model would not already write.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "star"))
sys.path.insert(0, str(REPO / "scripts" / "eval"))
os.environ["FAULT_SIM_BACKEND"] = "fast"

LEVELS = ("easy", "medium", "hard", "very_hard", "random_never_detects")
HEAD = re.compile(r'\{"problem": \["([^"]+)", "((?:[^"\\]|\\.)*)"\], "difficulty": "(\w+)"')


def with_reasoning(content, text):
    """``content`` with its thinking block holding ``text``; what follows the block is kept."""
    end = content.find("</think>")
    return f"<think>\n{text}\n{content[end:]}" if end >= 0 else f"<think>\n{text}\n</think>\n{content}"


def gold_messages(row):
    """The dataset conversation for ``row``, as SFT renders it."""
    from atpgllm.training.conversation import ConversationExample
    from atpgllm.training.dataset_utils import _vectors_to_text
    return _vectors_to_text(ConversationExample.from_record(copy.deepcopy(row), use_tools=True).messages)


def demonstration(row, simulator, seed):
    from atpgllm.training.reasoning_trace import render_trace
    label = json.loads(row["input_vector"]) if isinstance(row["input_vector"], str) else row["input_vector"]
    split = simulator.decisive_split(row["netlist"], row["fault"], label)
    if split is None:
        return None
    text = render_trace(row, seed=seed, split=split)
    if text is None:
        return None
    messages = gold_messages(row)
    first = next(m for m in messages if m["role"] == "assistant")
    first["content"] = with_reasoning(first["content"], text)
    return messages


def retry(row, slot, simulator, seed):
    """Model's failed turn and its real result, then the reading, the derivation and the dataset's vector."""
    from atpgllm.training.rationalization import sft_example
    from atpgllm.training.reasoning_trace import read_result, render_trace
    from atpgllm.training.tools import TOOLS
    generated = [m for m in slot["messages"] if m["role"] in ("assistant", "tool")]
    if len(generated) < 2 or generated[1]["role"] != "tool" or not generated[0].get("tool_calls"):
        return None
    table = json.loads(generated[1]["content"])
    label = json.loads(row["input_vector"]) if isinstance(row["input_vector"], str) else row["input_vector"]
    split = simulator.decisive_split(row["netlist"], row["fault"], label)
    text = render_trace(row, seed=seed, split=split) if split is not None else None
    if text is None:
        return None
    # Keep the failed turn and result in the layout of an RFT row, without its final answer.
    kept = dict(slot, messages=[*slot["prompt_messages"], generated[0], generated[1]])
    messages = sft_example(kept, slot["prompt_messages"], TOOLS)["messages"]
    gold = gold_messages(row)
    call_turn = next(m for m in gold if m["role"] == "assistant")
    tool_turn = next(m for m in gold if m["role"] == "tool")
    final_turn = [m for m in gold if m["role"] == "assistant"][-1]
    call_turn = dict(call_turn, content=with_reasoning(call_turn["content"], read_result(table, row["fault"], row["netlist"]) + "\n" + text))
    return messages + [call_turn, tool_turn, final_turn]


def failed_slots(run_dir, rounds, want, seed):
    """Unhinted single call attempts that did not detect, spread evenly over levels."""
    rng = random.Random(seed)
    chosen = []
    per_level = {level: want // len(LEVELS) for level in LEVELS}
    for n in rng.sample(rounds, len(rounds)):
        for path in sorted((Path(run_dir) / f"round_{n}" / "gen").glob("slots*.jsonl")):
            for line in open(path):
                head = HEAD.match(line)
                if not head or per_level.get(head.group(3), 0) <= 0 or '"hint_mode": "none"' not in line[:400]:
                    continue
                if rng.random() > 0.05:
                    continue
                try:
                    slot = json.loads(line)
                except json.JSONDecodeError:
                    continue
                facts = slot["facts"]
                if slot["hint_mode"] == "none" and facts["final"] and not facts["detected"] and facts["tool_calls"] == 1 and facts["single_user_turn"]:
                    slot["_round"] = n
                    chosen.append(slot)
                    per_level[head.group(3)] -= 1
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset", default="../data/freeset/dataset.freeset.asap7sc7p5t_28.rvt.tt.v4")
    parser.add_argument("--run_dir", default="runs/rft_v4_a100")
    parser.add_argument("--demonstrations", type=int, default=12288)
    parser.add_argument("--retries", type=int, default=3072)
    parser.add_argument("--retry_rounds", default="3,4,5")
    parser.add_argument("--max_model_len", type=int, default=14336)
    parser.add_argument("--max_prompt_length", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from rationalize import read_selected
    from atpgllm.training.dataset_utils import _gate_filter_passes
    from atpgllm.training.eval_report import Simulator
    from atpgllm.training.tools import TOOLS

    tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-4.2-8b", trust_remote_code=True)
    simulator = Simulator()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    written = Counter()

    def fits(messages):
        length = len(tokenizer.apply_chat_template(messages, tools=TOOLS, tokenize=True))
        return length <= args.max_model_len, length

    with open(out, "w") as sink:
        quota = {level: args.demonstrations // len(LEVELS) for level in LEVELS}
        rows = read_selected(args.dataset, quota, args.seed)
        for index, row in enumerate(rows):
            level = row["difficulty"]
            if quota[level] <= 0:
                continue
            if not _gate_filter_passes(row):
                counts["demo_gate_filter"] += 1
                continue
            try:
                messages = demonstration(row, simulator, index)
            except Exception as exc:
                counts[f"demo_error_{type(exc).__name__}"] += 1
                continue
            if messages is None:
                counts["demo_no_trace"] += 1
                continue
            ok, length = fits(messages)
            if not ok:
                counts["demo_too_long"] += 1
                continue
            sink.write(json.dumps({"messages": messages, "tools": TOOLS, "pilot": {
                "kind": "demonstration", "difficulty": level, "fault": row["fault"], "tokens": length}}) + "\n")
            quota[level] -= 1
            written["demonstration", level] += 1
            if index % 500 == 0:
                sink.flush()
                print(f"[pilot] {index}/{len(rows)} demonstrations scanned, {dict(quota)} left", flush=True)

        rounds = [int(n) for n in args.retry_rounds.split(",")]
        slots = failed_slots(args.run_dir, rounds, args.retries, args.seed)
        print(f"[pilot] {len(slots)} failed attempts chosen", flush=True)
        by_round = {}
        for index, slot in enumerate(slots):
            n = slot["_round"]
            if n not in by_round:
                by_round[n] = {(str(r.get("netlist_id") or r.get("netlist")), str(r["fault"])): r
                               for r in pq.read_table(Path(args.run_dir) / f"round_{n}" / "problems.parquet").to_pylist()}
            row = by_round[n].get(tuple(map(str, slot["problem"])))
            if row is None:
                counts["retry_no_row"] += 1
                continue
            try:
                messages = retry(row, slot, simulator, index)
            except Exception as exc:
                counts[f"retry_error_{type(exc).__name__}"] += 1
                continue
            if messages is None:
                counts["retry_no_trace"] += 1
                continue
            ok, length = fits(messages)
            if not ok:
                counts["retry_too_long"] += 1
                continue
            sink.write(json.dumps({"messages": messages, "tools": TOOLS, "pilot": {
                "kind": "retry", "difficulty": row["difficulty"], "fault": row["fault"], "tokens": length}}) + "\n")
            written["retry", row["difficulty"]] += 1
    report = {"written": {f"{k[0]}/{k[1]}": v for k, v in sorted(written.items())}, "skipped": dict(counts)}
    Path(str(out) + ".report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
