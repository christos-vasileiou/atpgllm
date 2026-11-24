#!/usr/bin/env python
"""Does the model use the simulator's answer? Gate before relaunching GDPO.

Reads evaluate.py slot files (greedy, several tool rounds allowed) and reports,
overall and per level:

* ``first_call``: the first call detects the fault.
* ``within_k``: some call among the first k detects (k = 1 .. max calls).
* ``answer``: the final answer's vector detects (the evaluator's detection).
* ``retry_after_miss``: share of slots whose first call missed that call again.
* ``next_call_after_miss``: share of all missed calls followed by another call.
* ``duplicate_calls``: share of later calls that repeat an earlier vector.
* ``stop_after_detect``: share of slots with a detecting call that stop calling.
* ``false_claim``: share of slots whose answer does not detect but whose final
  turn says the fault is detected.
* ``no_final_answer``, ``hit_call_limit``, ``mean_calls_never_detected``,
  ``call_detected_answer_not``: runaway and hand-off failures (GDPO's tool loop
  has no call cap).

Pass bar fixed before the retry arms were evaluated (2026-11-23): retry after a
miss 0.8 or more (baseline 0), duplicate calls 0.1 or less, detection within the
call budget above the baseline with gained > lost and McNemar p < 0.05, first
call no more than about 3 points below the baseline, false claims far below 0.96.

With ``--baseline`` the two runs are paired by example id (McNemar counts on
``answer`` and ``first_call``).

    python scripts/retry/gate_report.py runs/retry_sft_v4/gate/retry --baseline runs/retry_sft_v4/gate/round11
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

CLAIM_RE = re.compile(r"has been sensitized|differ at \d+ primary output|The pattern detects|is detected",
                      re.IGNORECASE)
# A reading turn that says the call missed (checked before CLAIM_RE).
MISS_RE = re.compile(r"agree on every primary output|No primary output has a 0|does not detect|do not differ|"
                     r"are identical|has no effect|is not detected", re.IGNORECASE)


def reading_claims(messages):
    """What the turn after each tool answer says: 'detected', 'miss' or None, in call order."""
    claims = []
    for i, message in enumerate(messages):
        if message.get("role") != "tool":
            continue
        following = next((m for m in messages[i + 1:] if m.get("role") == "assistant"), None)
        text = (following or {}).get("content") or ""
        head = text.split("To sum up", 1)[0]
        claims.append("miss" if MISS_RE.search(head) else "detected" if CLAIM_RE.search(head) else None)
    return claims


def _vector(observation):
    vector = observation.get("vector")
    if isinstance(vector, str):
        try:
            vector = ast.literal_eval(vector)
        except (ValueError, SyntaxError):
            return vector
    return json.dumps(vector, sort_keys=True) if isinstance(vector, dict) else str(vector)


def _outputs(observation):
    arguments = observation.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = ast.literal_eval(arguments)
        except (ValueError, SyntaxError):
            return None
    if not isinstance(arguments, dict):
        return None
    outputs = arguments.get("output_vector")
    if isinstance(outputs, str):
        try:
            outputs = json.loads(outputs)
        except json.JSONDecodeError:
            return None
    return list(outputs) if isinstance(outputs, dict) else None


def slot_facts(slot, fault):
    from atpgllm.training.tool_feedback import read_snapshot
    observations = slot.get("observations") or []
    detected = []
    for observation in observations:
        outputs = _outputs(observation)
        try:
            detected.append(bool(outputs) and read_snapshot(observation["result"], fault, outputs).detected)
        except Exception:
            detected.append(False)
    vectors = [_vector(o) for o in observations]
    final = slot.get("final_answer") or ""
    answer = (slot.get("reward_components") or {}).get("detection", 0) >= 1
    first_hit = next((i for i, d in enumerate(detected) if d), None)
    claims = reading_claims(slot.get("messages") or [])
    return {
        "readings": [(claim, "detected" if truth else "miss") for claim, truth in zip(claims, detected)],
        "calls": len(observations), "detected": detected, "answer": answer,
        "first_call": bool(detected[:1] and detected[0]),
        "duplicates": sum(v in vectors[:i] for i, v in enumerate(vectors) if i > 0),
        "later_calls": max(len(observations) - 1, 0),
        "first_hit": first_hit,
        "claims": bool(CLAIM_RE.search(final)),
        "final": slot.get("status") == "FINAL",
    }


def faults_of(manifest):
    data = json.loads(Path(manifest).read_text())
    return {e["_fixed_eval_id"]: e["fault"] for e in data["examples"]}


def fault_of(example_id, faults=None):
    """The fault of a slot: from the manifest, else the last field of an RFT problem id."""
    if faults and example_id in faults:
        return faults[example_id]
    return str(example_id).rsplit("|", 1)[-1] if "|" in str(example_id) else None


def load(run, faults=None):
    slots = {}
    for path in sorted(glob.glob(str(Path(run) / "slots*.jsonl"))):
        for line in open(path):
            try:
                slot = json.loads(line)
            except json.JSONDecodeError:
                continue
            fault = fault_of(slot["example_id"], faults)
            if fault is None:
                continue
            key = (slot["example_id"], slot.get("completion_slot", 0))
            slots[key if key[1] else slot["example_id"]] = (slot["level"], slot_facts(slot, fault))
    return slots


def summary(facts, max_k):
    n = len(facts)
    if not n:
        return {"n": 0}
    misses_first = [f for f in facts if f["calls"] and not f["detected"][0]]
    missed_calls = sum(sum(not d for d in f["detected"]) for f in facts)
    followed = sum(sum(1 for i, d in enumerate(f["detected"][:-1]) if not d) for f in facts)
    hits = [f for f in facts if f["first_hit"] is not None]
    wrong = [f for f in facts if not f["answer"]]
    later = sum(f["later_calls"] for f in facts)
    readings = [pair for f in facts for pair in f.get("readings", [])]
    misses_read = [claim for claim, truth in readings if truth == "miss"]
    detects_read = [claim for claim, truth in readings if truth == "detected"]
    return {
        "n": n,
        # Reading the tool answer: every turn after a tool answer, its verdict against the snapshot.
        "readings": len(readings),
        "reading_accuracy": sum(c == t for c, t in readings) / len(readings) if readings else None,
        "reading_miss_recall": sum(c == "miss" for c in misses_read) / len(misses_read) if misses_read else None,
        "reading_detect_recall": (sum(c == "detected" for c in detects_read) / len(detects_read)
                                  if detects_read else None),
        "first_call": sum(f["first_call"] for f in facts) / n,
        "within_k": {k: sum(any(f["detected"][:k]) for f in facts) / n for k in range(1, max_k + 1)},
        "answer": sum(f["answer"] for f in facts) / n,
        "mean_calls": sum(f["calls"] for f in facts) / n,
        "retry_after_miss": (sum(f["calls"] > 1 for f in misses_first) / len(misses_first)) if misses_first else None,
        "next_call_after_miss": followed / missed_calls if missed_calls else None,
        "duplicate_calls": sum(f["duplicates"] for f in facts) / later if later else None,
        "stop_after_detect": (sum(f["first_hit"] == f["calls"] - 1 for f in hits) / len(hits)) if hits else None,
        "false_claim": (sum(f["claims"] for f in wrong) / len(wrong)) if wrong else None,
        # Runaway risk for GDPO, whose tool loop has no call cap.
        "no_final_answer": sum(not f["final"] for f in facts) / n,
        "hit_call_limit": sum(f["calls"] >= max_k for f in facts) / n,
        "mean_calls_never_detected": (sum(f["calls"] for f in facts if f["first_hit"] is None)
                                      / max(1, sum(f["first_hit"] is None for f in facts))),
        "call_detected_answer_not": sum(f["first_hit"] is not None and not f["answer"] for f in facts) / n,
    }


def report(slots, max_k):
    by_level = defaultdict(list)
    for level, facts in slots.values():
        by_level[level].append(facts)
    return {"all": summary([f for _, f in slots.values()], max_k),
            "by_level": {level: summary(group, max_k) for level, group in sorted(by_level.items())}}


def paired(a, b, key):
    common = sorted(set(a) & set(b))
    gained = sum(1 for i in common if a[i][1][key] and not b[i][1][key])
    lost = sum(1 for i in common if b[i][1][key] and not a[i][1][key])
    from math import comb
    n = gained + lost
    p = min(1.0, 2 * sum(comb(n, j) for j in range(0, min(gained, lost) + 1)) / 2 ** n) if n else 1.0
    return {"common": len(common), "gained": gained, "lost": lost, "mcnemar_p": p}


PASS_BAR = {
    "retry_after_miss": 0.8,   # at least
    "duplicate_calls": 0.1,    # at most
    "first_call_drop": 0.03,   # at most, against the baseline
    "false_claim": 0.5,        # at most
    "within_k_p": 0.05,        # McNemar p of the within-budget gain, gained > lost
}


def check(result, max_k):
    """The pass bar, item by item; ``passed`` is True only when every item holds."""
    run, base = result["report"]["all"], result["baseline"]["report"]["all"]
    within = result["paired"]["within_k"]
    items = {
        "retry_after_miss": (run["retry_after_miss"] or 0.0) >= PASS_BAR["retry_after_miss"],
        "duplicate_calls": (run["duplicate_calls"] or 0.0) <= PASS_BAR["duplicate_calls"],
        "first_call": run["first_call"] >= base["first_call"] - PASS_BAR["first_call_drop"],
        "false_claim": (run["false_claim"] or 0.0) <= PASS_BAR["false_claim"],
        "within_k": within["gained"] > within["lost"] and within["mcnemar_p"] < PASS_BAR["within_k_p"],
    }
    return {"passed": all(items.values()), "items": items, "bar": PASS_BAR, "max_k": max_k}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run")
    parser.add_argument("--baseline")
    parser.add_argument("--manifest", help="fixed_eval_manifest.json used by both runs (RFT evaluation slots "
                        "carry the fault in their id and need none)")
    parser.add_argument("--max_k", type=int, default=8)
    parser.add_argument("--out")
    parser.add_argument("--require", action="store_true",
                        help="Exit 1 unless the run passes the bar against --baseline")
    args = parser.parse_args()
    faults = faults_of(args.manifest) if args.manifest else None
    run = load(args.run, faults)
    result = {"run": str(args.run), "report": report(run, args.max_k)}
    if args.baseline:
        base = load(args.baseline, faults)
        result["baseline"] = {"run": str(args.baseline), "report": report(base, args.max_k)}
        result["paired"] = {key: paired(run, base, key) for key in ("answer", "first_call")}
        for slots in (run, base):
            for _, facts in slots.values():
                facts["within_k"] = any(facts["detected"][:args.max_k])
        result["paired"]["within_k"] = paired(run, base, "within_k")
        result["gate"] = check(result, args.max_k)
    text = json.dumps(result, indent=1)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    if args.require:
        if "gate" not in result:
            parser.error("--require needs --baseline")
        sys.exit(0 if result["gate"]["passed"] else 1)


def scalars(run, max_k=8):
    """Flat reading, retry and detection metrics of one evaluation directory, for W&B and state.json."""
    full = report(load(run), max_k)
    keys = ("reading_accuracy", "reading_miss_recall", "reading_detect_recall", "retry_after_miss",
            "next_call_after_miss", "duplicate_calls", "false_claim", "first_call", "answer", "mean_calls",
            "no_final_answer", "hit_call_limit")
    out = {key: full["all"].get(key) for key in keys}
    out[f"within_{max_k}"] = full["all"].get("within_k", {}).get(max_k)
    by_kind = defaultdict(list)
    for path in sorted(glob.glob(str(Path(run) / "slots*.jsonl"))):
        for line in open(path):
            try:
                slot = json.loads(line)
            except json.JSONDecodeError:
                continue
            fault = fault_of(slot["example_id"])
            if fault:
                by_kind[slot.get("kind") or "combinational"].append(slot_facts(slot, fault))
    for kind, facts in by_kind.items():
        part = summary(facts, max_k)
        for key in ("reading_accuracy", "reading_miss_recall", "retry_after_miss"):
            out[f"{kind}_{key}"] = part.get(key)
        out[f"{kind}_within_{max_k}"] = part.get("within_k", {}).get(max_k)
    return out


if __name__ == "__main__":
    main()
