#!/usr/bin/env python
"""Report a sampling run and write the trajectories to train on as SFT examples.

    python scripts/star/build_sft.py runs/star_pilot [--require_path_mention] [--max_per_problem 2]

Reads every ``slots*.jsonl`` in the run directory (one per shard) and writes
``report.json`` (per hint mode and difficulty level) and ``sft.jsonl``
(hint-free ``{"messages", "tools"}`` examples). For each problem, kept
unhinted attempts (rejection sampling) are used when there are any, and hinted
ones (rationalization) only otherwise, least revealing hint first; see
:func:`atpgllm.training.rationalization.choose_for_training`. Examples longer
than ``--max_model_len`` tokens are dropped, as the SFT stream drops them, so
no answer is truncated away.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def read_slots(run):
    """Every slot line of the run's shard files; a torn last line (crash) is skipped."""
    for path in sorted(Path(run).glob("slots*.jsonl")):
        for line in open(path):
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"[build_sft] skipping a partial line in {path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="Directory with slots*.jsonl from rationalize.py")
    parser.add_argument("--require_path_mention", action="store_true",
                        help="Keep only trajectories that name an internal net on the propagation path")
    parser.add_argument("--max_per_problem", type=int, default=2,
                        help="Kept trajectories per problem, with distinct answer vectors")
    parser.add_argument("--hinted_only", action="store_true",
                        help="Write only rationalizations (the old STaR-only set), no unhinted successes")
    parser.add_argument("--hinted_even_if_solved", action="store_true",
                        help="Also use hinted attempts for problems an unhinted attempt solved")
    parser.add_argument("--max_model_len", type=int, default=14336,
                        help="SFT sequence budget (MAX_MODEL_LEN of the SFT config)")
    parser.add_argument("--tokenizer", default=None, help="Defaults to the run's base model")
    args = parser.parse_args()

    from atpgllm.training.rationalization import choose_for_training, keep, sft_example
    from atpgllm.training.tools import TOOLS

    run = Path(args.run)
    stats = defaultdict(lambda: defaultdict(float))
    solved = defaultdict(set)
    problems = defaultdict(set)
    by_problem = defaultdict(list)
    for slot in read_slots(run):
        facts, key = slot["facts"], (slot["hint_mode"], slot["difficulty"])
        problem = tuple(slot["problem"])
        s = stats[key]
        problems[key].add(problem)
        s["slots"] += 1
        s["final"] += facts["final"]
        s["detected"] += facts["detected"]
        s["truncated"] += facts.get("truncated", False)
        s["tool_calls"] += facts["tool_calls"]
        if facts["detected"]:
            solved[key].add(problem)
            s["leak"] += facts["leak"]
            s["first_call_copies_cube"] += facts["first_call_copies_cube"]
            s["first_call_follows_hint"] += facts.get("first_call_follows_hint", False)
            s["names_internal_path_net"] += facts["names_internal_path_net"]
            s["has_internal_path"] += facts["has_internal_path"]
        if keep(facts, args.require_path_mention):
            s["kept"] += 1
        if not (args.hinted_only and slot["hint_mode"] == "none"):
            by_problem[problem].append(slot)

    report = {}
    for (mode, level), s in sorted(stats.items()):
        n, d = s["slots"], max(s["detected"], 1)
        report[f"{mode}/{level}"] = {
            "problems": len(problems[(mode, level)]), "slots": int(n),
            "final_rate": s["final"] / n, "detect_rate_per_slot": s["detected"] / n,
            "truncated_rate": s["truncated"] / n,
            "problems_solved": len(solved[(mode, level)]) / len(problems[(mode, level)]),
            "mean_tool_calls": s["tool_calls"] / n,
            # Shares among detecting slots:
            "leak_rate": s["leak"] / d, "first_call_copies_cube": s["first_call_copies_cube"] / d,
            "first_call_follows_hint": s["first_call_follows_hint"] / d,
            "names_internal_path_net": s["names_internal_path_net"] / max(s["has_internal_path"], 1),
            "kept_slots": int(s["kept"]),
        }
    from transformers import AutoTokenizer
    base = args.tokenizer or json.loads((run / "run.json").read_text())["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    examples, too_long = [], 0
    trained = defaultdict(lambda: defaultdict(int))
    for problem, slots in sorted(by_problem.items()):
        for slot in choose_for_training(slots, args.max_per_problem,
                                        require_path_mention=args.require_path_mention,
                                        hinted_only_if_unsolved=not args.hinted_even_if_solved):
            example = sft_example(slot, slot["prompt_messages"], TOOLS)
            length = len(tokenizer.apply_chat_template(example["messages"], tools=TOOLS, tokenize=True))
            if length > args.max_model_len:
                too_long += 1
                continue
            example["star"] = {"problem": list(problem), "difficulty": slot["difficulty"],
                               "hint_mode": slot["hint_mode"], "tokens": length}
            trained[slot["difficulty"]][slot["hint_mode"]] += 1
            examples.append(example)
    with open(run / "sft.jsonl", "w") as sink:
        for example in examples:
            sink.write(json.dumps(example) + "\n")
    summary = {"by_mode_and_level": report, "sft_examples": len(examples),
               "sft_examples_by_level_and_mode": {k: dict(v) for k, v in sorted(trained.items())},
               "dropped_over_max_model_len": too_long, "max_model_len": args.max_model_len,
               "sft_problems": len({tuple(e["star"]["problem"]) for e in examples}),
               "require_path_mention": args.require_path_mention, "hinted_only": args.hinted_only,
               "hinted_even_if_solved": args.hinted_even_if_solved}
    (run / "report.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
