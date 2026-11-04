#!/usr/bin/env python
"""Check the gate by gate renderer on dataset rows, CPU only.

    python scripts/star/check_reasoning_trace.py --rows 1000 --out analysis/reasoning_axes_<date>/trace_check.json

Every rendered statement must pass the reasoning audit, a corrupted trace must
score lower, and the report gives the token cost and how many conversations
still fit the SFT budget with the rendered reasoning in place of the dataset's.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics as st
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ["FAULT_SIM_BACKEND"] = "fast"


def corrupt(text, rng):
    """Flip the value of half the stated nets."""
    def flip(match):
        return f"{match.group(1)}{1 - int(match.group(2))}" if rng.random() < 0.5 else match.group(0)
    return re.sub(r"(\b[\w\[\]/\\]+ = )([01])\b", flip, text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="../data/freeset/dataset.freeset.asap7sc7p5t_28.rvt.tt.v4")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--max_model_len", type=int, default=14336)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from atpgllm.training.conversation import ConversationExample
    from atpgllm.training.reasoning_audit import audit_text
    from atpgllm.training.reasoning_trace import render_trace
    from atpgllm.training.tools import TOOLS

    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-4.2-8b", trust_remote_code=True)
    paths = sorted((Path(args.dataset) / args.split).glob("*.parquet"))
    rng.shuffle(paths)
    rows = []
    for path in paths:
        table = pq.read_table(path).to_pylist()
        rng.shuffle(table)
        rows.extend(table[:max(1, args.rows // 50)])
        if len(rows) >= args.rows:
            break
    rows = rows[:args.rows]
    results = Counter()
    claim, implication, corrupted_claim, corrupted_implication, costs, old_lengths, new_lengths, failures = ([] for _ in range(8))
    for index, row in enumerate(rows):
        text = render_trace(dict(row), seed=index)
        if text is None:
            results["no_trace"] += 1
            continue
        results["rendered"] += 1
        snapshot = json.loads(row["snapshot"]) if isinstance(row["snapshot"], str) else row["snapshot"]
        good, bad = snapshot["Good Machine"], snapshot["Bad Machine"]
        audit = audit_text(text, row["netlist"], row["fault"], good, bad)
        claim.append(audit.get("claim_truth"))
        implication.append(audit.get("implication_truth"))
        if audit.get("claim_truth", 1) < 1 or audit.get("implication_truth", 1) < 1:
            failures.append({"fault": row["fault"], "module": row["module_name"], "audit": audit, "text": text[:1500]})
        worse = audit_text(corrupt(text, rng), row["netlist"], row["fault"], good, bad)
        corrupted_claim.append(worse.get("claim_truth"))
        corrupted_implication.append(worse.get("implication_truth"))
        tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        convo = ConversationExample.from_record(dict(row), use_tools=True)
        old_think = next((m.get("content") or "" for m in convo.messages if m["role"] == "assistant"), "")
        old_tokens = len(tokenizer(old_think, add_special_tokens=False)["input_ids"])
        costs.append(tokens - old_tokens)
        full = len(tokenizer.apply_chat_template(convo.messages, tools=TOOLS, tokenize=True))
        old_lengths.append(full)
        new_lengths.append(full + tokens - old_tokens)
    known = lambda values: [v for v in values if v is not None]
    fits = lambda lengths: sum(v <= args.max_model_len for v in lengths) / max(len(lengths), 1)
    summary = {
        "rows": len(rows), **results,
        "claim_truth_mean": st.mean(known(claim)), "claim_truth_all_one": sum(v == 1 for v in known(claim)) / len(known(claim)),
        "implication_truth_mean": st.mean(known(implication)),
        "implication_truth_all_one": sum(v == 1 for v in known(implication)) / len(known(implication)),
        "implication_covered": len(known(implication)) / max(results["rendered"], 1),
        "corrupted_claim_truth_mean": st.mean(known(corrupted_claim)),
        "corrupted_implication_truth_mean": st.mean(known(corrupted_implication)),
        "added_tokens_median": st.median(costs), "added_tokens_p90": sorted(costs)[int(0.9 * len(costs))],
        "fits_budget_now": fits(old_lengths), "fits_budget_with_trace": fits(new_lengths),
        "audit_failures": len(failures), "failure_examples": failures[:5]}
    Path(args.out).write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "failure_examples"}, indent=1))
    for failure in failures[:3]:
        print(failure["fault"], failure["module"], failure["audit"])
        print(failure["text"][:600])


if __name__ == "__main__":
    main()
