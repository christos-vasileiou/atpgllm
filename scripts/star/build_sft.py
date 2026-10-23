#!/usr/bin/env python
"""Report a rationalization run and write its kept trajectories as SFT examples.

    python scripts/star/build_sft.py runs/star_pilot [--require_path_mention] [--max_per_problem 2]

Writes ``report.json`` (per hint mode and difficulty level) and ``sft.jsonl``
(hint-free ``{"messages", "tools"}`` examples) into the run directory. Only
hinted slots are kept unless ``--include_unhinted``; unhinted successes are
ordinary rejection-sampling data.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="Directory with slots.jsonl from rationalize.py")
    parser.add_argument("--require_path_mention", action="store_true",
                        help="Keep only trajectories that name an internal net on the propagation path")
    parser.add_argument("--max_per_problem", type=int, default=2,
                        help="Kept trajectories per problem, preferring distinct answer vectors")
    parser.add_argument("--include_unhinted", action="store_true")
    args = parser.parse_args()

    from atpgllm.training.rationalization import keep, sft_example
    from atpgllm.training.tools import TOOLS

    run = Path(args.run)
    stats = defaultdict(lambda: defaultdict(float))
    solved = defaultdict(set)
    problems = defaultdict(set)
    kept = defaultdict(list)
    for line in open(run / "slots.jsonl"):
        slot = json.loads(line)
        facts, key = slot["facts"], (slot["hint_mode"], slot["difficulty"])
        problem = tuple(slot["problem"])
        s = stats[key]
        problems[key].add(problem)
        s["slots"] += 1
        s["final"] += facts["final"]
        s["detected"] += facts["detected"]
        s["tool_calls"] += facts["tool_calls"]
        if facts["detected"]:
            solved[key].add(problem)
            s["leak"] += facts["leak"]
            s["first_call_copies_cube"] += facts["first_call_copies_cube"]
            s["names_internal_path_net"] += facts["names_internal_path_net"]
            s["has_internal_path"] += facts["has_internal_path"]
        if keep(facts, args.require_path_mention) and (args.include_unhinted or slot["hint_mode"] != "none"):
            s["kept"] += 1
            kept[problem].append(slot)

    report = {}
    for (mode, level), s in sorted(stats.items()):
        n, d = s["slots"], max(s["detected"], 1)
        report[f"{mode}/{level}"] = {
            "problems": len(problems[(mode, level)]), "slots": int(n),
            "final_rate": s["final"] / n, "detect_rate_per_slot": s["detected"] / n,
            "problems_solved": len(solved[(mode, level)]) / len(problems[(mode, level)]),
            "mean_tool_calls": s["tool_calls"] / n,
            # Shares among detecting slots:
            "leak_rate": s["leak"] / d, "first_call_copies_cube": s["first_call_copies_cube"] / d,
            "names_internal_path_net": s["names_internal_path_net"] / max(s["has_internal_path"], 1),
            "kept_slots": int(s["kept"]),
        }
    examples = []
    for problem, slots in kept.items():
        chosen, vectors = [], set()
        for slot in sorted(slots, key=lambda x: (x["hint_mode"], x.get("completion_slot", 0))):
            vector = json.dumps(slot.get("vector"), sort_keys=True)
            if vector in vectors:
                continue
            vectors.add(vector)
            chosen.append(slot)
            if len(chosen) == args.max_per_problem:
                break
        for slot in chosen:
            example = sft_example(slot, slot["prompt_messages"], TOOLS)
            example["star"] = {"problem": list(problem), "difficulty": slot["difficulty"],
                               "hint_mode": slot["hint_mode"]}
            examples.append(example)
    with open(run / "sft.jsonl", "w") as sink:
        for example in examples:
            sink.write(json.dumps(example) + "\n")
    summary = {"by_mode_and_level": report, "sft_examples": len(examples),
               "sft_problems": len({tuple(e["star"]["problem"]) for e in examples}),
               "require_path_mention": args.require_path_mention}
    (run / "report.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
