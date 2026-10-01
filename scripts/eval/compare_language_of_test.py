#!/usr/bin/env python3
"""Recompute paired pilot comparisons and pre-feedback proposal diagnostics."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from lot_metrics import assignments, interval, outcomes, summarize
from execute_language_of_test import file_hash, write


def first_proposals(rows):
    result, unchanged = [], 0
    for row in rows:
        text = row["raw_answer"].split("</tool_call>", 1)[0]
        parsed = {"input": None, "expected": None}
        for parameter, key in (("input_vector", "input"), ("output_vector", "expected")):
            match = re.search(r'<parameter=' + parameter + r'>\s*(.*?)\s*</parameter>', text, re.S)
            if match:
                try:
                    names = row["parsed"]["input"] if key == "input" else row["reference"].get("good")
                    if names is not None:
                        parsed[key] = assignments(match[1], names)
                except (SyntaxError, ValueError):
                    pass
        good = bad = None
        if parsed["input"] is not None and parsed["input"] == row["parsed"]["input"]:
            unchanged += 1
            # Reuse independently replayed values only for precisely the same
            # stimulus, circuit and target. A changed proposal stays unknown.
            good, bad = row["reference"].get("good"), row["reference"].get("bad")
        result.append({"example_id": row["example_id"], "circuit_id": row["circuit_id"],
                       "parsed": parsed, **outcomes(parsed, good, bad)})
    return {"scope": "first tool arguments before feedback, not a fresh tool-free-prompt experiment",
            "same_stimulus_as_final": unchanged, "metrics": summarize(result), "rows": result}


def paired(left, right):
    l = {r["example_id"]: r for r in left}
    r = {r["example_id"]: r for r in right}
    if len(l) != len(left) or len(r) != len(right) or set(l) != set(r):
        raise ValueError("Paired pilot requires exactly one matching slot per problem")
    for key in l:
        if (l[key]["circuit_id"], l[key]["fault"]) != (r[key]["circuit_id"], r[key]["fault"]):
            raise ValueError("Paired problem semantics differ")
    ls = summarize([{**x, **x["reference"]["outcomes"]} for x in left])
    rs = summarize([{**x, **x["reference"]["outcomes"]} for x in right])
    result = {}
    for metric in ("D", "S", "U"):
        diffs = [rs["per_circuit"][c][metric] - vals[metric] for c, vals in ls["per_circuit"].items()]
        result[metric] = {"macro_difference_pp": 100 * sum(diffs) / len(diffs),
            "circuit_bootstrap_95_pp": [100 * v for v in interval(diffs)],
            "circuit_wins_ties_losses": [sum(d > 0 for d in diffs), sum(d == 0 for d in diffs), sum(d < 0 for d in diffs)]}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    runs = {}
    for name in ("sft90_fast_reference", "grpo_fast5_reference", "sft90_reference"):
        path = args.root / name
        runs[name] = [json.loads(line) for line in (path / "slots.jsonl").read_text().splitlines()]
    comparison = {"source_sha256": {name: file_hash(args.root / name / "slots.jsonl") for name in runs},
        "paired_GRPO_fast5_minus_SFT90_same_fast_tool_policy": paired(runs["sft90_fast_reference"], runs["grpo_fast5_reference"]),
        "first_model_proposal": {name: first_proposals(rows) for name, rows in runs.items()},
        "limitations": ["retrospective validation; not checkpoint selection or a locked test claim",
            "circuit bootstrap does not establish family independence or training-seed stability",
            "single deterministic slot: only pass@1, no pooling identical reruns",
            "TetraMAX-trained checkpoint-1 has no saved post-training evaluation"]}
    write(args.output, comparison)
    print(json.dumps({"paired": comparison["paired_GRPO_fast5_minus_SFT90_same_fast_tool_policy"],
        "first_proposal": {k: {"unchanged": v["same_stimulus_as_final"], "D": v["metrics"]["D"]["micro"],
                                "U": v["metrics"]["U"]["micro"]} for k, v in comparison["first_model_proposal"].items()}}, indent=2))


if __name__ == "__main__":
    main()
