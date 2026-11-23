#!/usr/bin/env python
"""Does the next call fix what the last answer showed? The model's edits against random edits.

For every call that missed and was followed by another call (evaluate.py
slots, or any slots with ``observations``), the edit is the set of inputs the
next call changed. The script reads both answers and asks:

* after a miss that never activated the fault: does the next call activate it,
  and does it detect;
* after a miss that was activated but blocked: does the next call detect, and
  does it carry the effect further (staged propagation ratio up, activation
  kept);
* the same questions for ``--random`` random edits of the same size from the
  same failed vector (inputs drawn uniformly, a 0/1 input flipped, an x input
  set to 0 or 1), simulated with the fast simulator;
* how many of the changed inputs lie in the inputs that can fix the reading
  (the fault site's fan-in after a not-activated miss, the blocking side
  inputs' fan-in after a blocked one), against the share a random edit gets.

An edit that beats random edits of its own size on these questions reads the
answer; one that does not is guessing. CPU only (fast simulator, which agreed
with TetraMAX on 200 of 200 stored RFT failures).

    python scripts/retry/edit_quality.py runs/retry_sft_v4/gate/retry_ckpt20 --out edit_quality.json
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import random
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def _literal(value):
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return None
    return value


def _calls(slot):
    """[(vector, outputs, fault, table)] of a slot's executed calls, in order."""
    out = []
    for observation in slot.get("observations") or []:
        vector = _literal(observation.get("vector"))
        arguments = _literal(observation.get("arguments")) or {}
        outputs = _literal(arguments.get("output_vector")) if isinstance(arguments, dict) else None
        fault = arguments.get("fault") if isinstance(arguments, dict) else None
        if not isinstance(vector, dict) or not isinstance(outputs, dict) or not fault:
            return None
        out.append((vector, list(outputs), str(fault).strip(), observation.get("result")))
    return out


def _bit(value):
    text = str(value).strip()
    return text if text in ("0", "1") else None


def analyse(job):
    slot, samples, seed = job
    os.environ.setdefault("FAULT_SIM_BACKEND", "fast")
    from atpgllm.training.retry_repair import RepairSearch, fast_simulator, fast_structure
    from atpgllm.training.tools import ToolHelper
    calls = _calls(slot)
    if not calls or len(calls) < 2:
        return []
    user = next((m["content"] for m in slot.get("messages") or [] if m.get("role") == "user"), None)
    document = ToolHelper.get_document(user) if user else None
    if not document:
        return []
    fault, outputs = calls[0][2], calls[0][1]
    structure = fast_structure(document["netlist"])
    simulate = fast_simulator(structure, fault, outputs)
    inputs = list(calls[0][0])
    search = RepairSearch(structure, simulate, fault, inputs, outputs)
    rng = random.Random(seed)
    pairs = []
    for (vector, _, _, table), (following, _, _, next_table) in zip(calls, calls[1:]):
        try:
            before = search.explain(vector, table)
            after = search.explain(following, next_table)
        except Exception:
            continue
        if before.reading.detected:
            continue
        changed = [net for net in following if str(following[net]) != str(vector.get(net))]
        kind = "not_activated" if not before.reading.activated else "blocked"
        if kind == "not_activated":
            pool = set([fault.split(None, 1)[1]] if fault.split(None, 1)[1] in inputs
                       else search.fanin_inputs([before.reading.fault_net]))
        else:
            sides = sorted({side for _, _, side, _ in before.blockers})
            pool = set(search.fanin_inputs(sides)) if sides else set()

        def outcome(observation):
            return {"activates": observation.reading.activated, "detects": observation.reading.detected,
                    "further": observation.reading.activated and observation.propagation > before.propagation}

        record = {"level": slot.get("level"), "kind": kind, "size": len(changed),
                  "model": outcome(after), "repeat": not changed,
                  "in_pool": (sum(net in pool for net in changed) / len(changed)) if changed else None,
                  "pool_share": len(pool) / len(inputs) if inputs else None}
        if changed:
            totals = defaultdict(float)
            for _ in range(samples):
                edit = dict(vector)
                for net in rng.sample(inputs, min(len(changed), len(inputs))):
                    bit = _bit(edit.get(net))
                    edit[net] = 1 - int(bit) if bit is not None else rng.randint(0, 1)
                try:
                    result = outcome(search.observe(edit))
                except Exception:
                    continue
                for key, value in result.items():
                    totals[key] += value / samples
            record["random"] = dict(totals)
        pairs.append(record)
    return pairs


def summary(pairs):
    by = defaultdict(list)
    for pair in pairs:
        by[pair["kind"]].append(pair)
        by["all"].append(pair)
    out = {}
    for kind, group in sorted(by.items()):
        edited = [p for p in group if "random" in p]
        row = {"pairs": len(group), "repeats": sum(p["repeat"] for p in group),
               "mean_edit_size": sum(p["size"] for p in edited) / len(edited) if edited else None}
        for key in ("activates", "detects", "further"):
            row[f"model_{key}"] = sum(p["model"][key] for p in edited) / len(edited) if edited else None
            row[f"random_{key}"] = sum(p["random"].get(key, 0) for p in edited) / len(edited) if edited else None
        shares = [p for p in edited if p["in_pool"] is not None and p["pool_share"] is not None]
        row["changed_inputs_in_fix_pool"] = sum(p["in_pool"] for p in shares) / len(shares) if shares else None
        row["random_expected_in_fix_pool"] = sum(p["pool_share"] for p in shares) / len(shares) if shares else None
        out[kind] = row
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="Directories (or files) with slots*.jsonl")
    parser.add_argument("--random", type=int, default=20, help="Random edits per model edit")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out")
    args = parser.parse_args()
    result = {}
    for run in args.runs:
        paths = [run] if Path(run).is_file() else sorted(glob.glob(str(Path(run) / "slots*.jsonl")))
        slots = [json.loads(line) for path in paths for line in open(path) if line.strip()]
        jobs = [(slot, args.random, args.seed + i) for i, slot in enumerate(slots)]
        with Pool(args.workers) as pool:
            pairs = [pair for group in pool.map(analyse, jobs, chunksize=4) for pair in group]
        result[str(run)] = {"slots": len(slots), "summary": summary(pairs)}
    text = json.dumps(result, indent=1)
    if args.out:
        Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
