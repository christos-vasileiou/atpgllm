#!/usr/bin/env python
"""Rank candidate GDPO reward axes on stored rollouts. CPU only, nothing is trained.

    python scripts/star/score_reward_axes.py --run_dir runs/rft_v4_a100 --rounds 1,2,3,4,5 --out <dir>

Each unhinted problem of a round's generation holds a group of attempts, the same
group GDPO normalizes over. For every axis the report says whether it gives a
signal inside a group, whether it tracks detection and near misses, whether a
detecting attempt is at its maximum, how much it costs, whether synthetic
exploits can beat real attempts, and whether it pays for long or uniform text.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ["FAULT_SIM_BACKEND"] = "fast"

LEVELS = ("easy", "medium", "hard", "very_hard", "random_never_detects")
HARD = LEVELS[2:]
# The start of a slot line, so that lines of other problems need no JSON parse.
HEAD = re.compile(r'\{"problem": \["([^"]+)", "((?:[^"\\]|\\.)*)"\], "difficulty": "(\w+)"')


def load_signals():
    """The staged reward's propagation and controlling signals."""
    from atpgllm.llm import propagation_signals
    return propagation_signals


def rows_of(run_dir, n, per_level, seed):
    """Groups of attempts of round ``n``: problem key to its dataset row and unhinted slots."""
    import pyarrow.parquet as pq
    problems = {}
    for row in pq.read_table(run_dir / f"round_{n}" / "problems.parquet").to_pylist():
        problems[(str(row.get("netlist_id") or row.get("netlist")), str(row["fault"]))] = row
    # Generation is time boxed, so only sample among the problems it reached.
    rng = random.Random(seed + n)
    by_level = defaultdict(set)
    paths = sorted((run_dir / f"round_{n}" / "gen").glob("slots*.jsonl"))
    for path in paths:
        for line in open(path):
            head = HEAD.match(line)
            if head:
                by_level[head.group(3)].add((head.group(1), json.loads(f'"{head.group(2)}"')))
    chosen = set()
    for keys in by_level.values():
        keys = sorted(keys)
        rng.shuffle(keys)
        chosen.update(keys[:per_level])
    groups = defaultdict(list)
    for path in paths:
        for line in open(path):
            head = HEAD.match(line)
            if not head or (head.group(1), json.loads(f'"{head.group(2)}"')) not in chosen:
                continue
            try:
                slot = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = tuple(map(str, slot["problem"]))
            if slot.get("hint_mode") == "none" and key in chosen:
                slot.pop("prompt_messages", None)
                slot.pop("trace", None)
                groups[key].append(slot)
    return {key: (problems[key], slots) for key, slots in groups.items() if len(slots) >= 2}


class Scorer:
    def __init__(self, signals):
        from atpgllm.training.eval_report import Simulator
        from atpgllm.training.reasoning_audit import audit_slot
        self.simulator, self.audit_slot, self.signals = Simulator(), audit_slot, signals
        self.graphs = {}
        self.seconds = defaultdict(float)

    def tables(self, text, fault, vector):
        net_list = self.simulator.compile(text)
        full = {net: (int(vector[net]) if str(vector.get(net)) in ("0", "1") else "x") for net in net_list.input_nets}
        frame = self.simulator.fs.fast_fault_sim(full, {o: 0 for o in net_list.output_nets}, fault, net_list)
        return frame["Good Machine"].to_dict(), frame["Bad Machine"].to_dict(), net_list

    def distances(self, text, fault, vector):
        """Whether one input change, x to 0 or 1 included, activates or detects."""
        net_list = self.simulator.compile(text)
        activate = detect = False
        for net in net_list.input_nets:
            current = str(vector.get(net, "x"))
            for value in (0, 1):
                if str(value) == current:
                    continue
                a, d = self.simulator.outcome(text, fault, {**vector, net: value})
                activate, detect = activate or a, detect or d
                if detect:
                    return True, True
        return activate, detect

    def score(self, row, slot, distances=True):
        text, fault = row["netlist"], row["fault"]
        vector = slot.get("vector") or {}
        out = {}
        reference = {"netlist": text, "fault": fault}
        t = time.time()
        out.update(self.audit_slot(slot, reference))
        self.seconds["audit"] += time.time() - t
        calls = len(slot.get("observations") or [])
        out["real_call"] = float(calls >= 1)
        out["detected"] = float((slot.get("reward_components") or {}).get("fault_detected_by_pred_input_vector_acc_logonly", 0) == 1)
        if not vector:
            for axis in ("activated", "propagation", "controlling", "activate_distance", "detect_distance"):
                out[axis] = 0.0
            return out
        t = time.time()
        good, bad, net_list = self.tables(text, fault, vector)
        site, stuck = fault.split(" ", 1)[1], fault[2]
        activated = str(good.get(site)) in ("0", "1") and str(good.get(site)) != stuck
        detected = any(str(good.get(o)) in ("0", "1") and str(bad.get(o)) in ("0", "1") and str(good[o]) != str(bad[o])
                       for o in net_list.output_nets)
        out["activated"], out["simulated_detect"] = float(activated), float(detected)
        self.seconds["simulate"] += time.time() - t
        t = time.time()
        if text not in self.graphs:
            self.graphs[text] = self.signals.SignalGraph(net_list)
        ratios = self.signals.attempt_signals(self.graphs[text], good, bad, site, activated, detected)
        out["propagation"] = self.signals.sensitize(ratios["propagation"])
        out["controlling"] = self.signals.sensitize(ratios["controlling"])
        self.seconds["signals"] += time.time() - t
        if distances:
            t = time.time()
            if detected:
                out["activate_distance"], out["detect_distance"] = 1.0, 1.0
            else:
                a, d = (activated, False)
                if not activated or True:
                    a, d = self.distances(text, fault, vector)
                out["activate_distance"] = 1.0 if activated else 0.5 if a else 0.0
                out["detect_distance"] = 0.5 if d else 0.0
            self.seconds["distance"] += time.time() - t
        return out


def _ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2
        i = j + 1
    return ranks


def _corr(pairs):
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sx = sum((x - mx) ** 2 for x in xs)
    sy = sum((y - my) ** 2 for y in ys)
    if not sx or not sy:
        return None
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(sx * sy)


def axis_statistics(groups, axis):
    """One row of the report. ``groups`` is a list of ``(level, round, attempts)``."""
    varying = usable = 0
    within, near_miss, lengths = [], [], []
    at_max = detecting = 0
    hard_values, by_round = [], defaultdict(list)
    for level, round_n, attempts in groups:
        values = [a.get(axis) for a in attempts]
        known = [(v, a) for v, a in zip(values, attempts) if v is not None]
        if len(known) < 2:
            continue
        usable += 1
        scores = [v for v, _ in known]
        varying += len(set(scores)) > 1
        top = max(scores)
        for v, a in known:
            by_round[round_n].append(v)
            if level in HARD:
                hard_values.append(v)
            if a.get("detected"):
                detecting += 1
                at_max += v == top
        mean = sum(scores) / len(scores)
        for v, a in known:
            within.append((v - mean, a["detected"] - sum(x["detected"] for _, x in known) / len(known)))
            if a.get("reasoning_chars") is not None:
                lengths.append((v - mean, a["reasoning_chars"] - sum(x.get("reasoning_chars", 0) for _, x in known) / len(known)))
        failures = [(v, a) for v, a in known if not a["detected"] and a.get("detect_distance") is not None]
        if len(failures) >= 2:
            fm = sum(v for v, _ in failures) / len(failures)
            nm = sum(a["detect_distance"] for _, a in failures) / len(failures)
            near_miss.extend((v - fm, a["detect_distance"] - nm) for v, a in failures)
    # Where GDPO has no detection to learn from: signal among groups in which every attempt fails,
    # and whether failures that score higher sit on problems another attempt could solve. That
    # second test is independent of the near miss label, which detect_distance defines itself.
    all_fail = all_fail_signal = 0
    closeness = defaultdict(list)
    for level, _round_n, attempts in groups:
        known = [a for a in attempts if a.get(axis) is not None]
        failed = [a for a in known if not a["detected"]]
        if len(known) >= 2 and len(failed) == len(known):
            all_fail += 1
            all_fail_signal += len({a[axis] for a in known}) > 1
        if failed:
            closeness[level].append((sum(a[axis] for a in failed) / len(failed), float(len(failed) < len(known))))
    pooled = []
    for pairs in closeness.values():
        mx = sum(x for x, _ in pairs) / len(pairs)
        my = sum(y for _, y in pairs) / len(pairs)
        pooled.extend((x - mx, y - my) for x, y in pairs)
    return {"axis": axis, "groups": usable, "group_coverage": usable / max(len(groups), 1),
            "all_fail_groups": all_fail, "all_fail_groups_with_signal": all_fail_signal / max(all_fail, 1),
            "failure_score_predicts_solved": _corr(pooled),
            "groups_with_signal": varying / max(usable, 1), "within_group_corr_detection": _corr(within),
            "within_group_corr_near_miss": _corr(near_miss), "detecting_at_group_max": at_max / max(detecting, 1),
            "within_group_corr_length": _corr(lengths),
            "hard_mean": sum(hard_values) / len(hard_values) if hard_values else None,
            "hard_std": (lambda v: math.sqrt(sum((x - sum(v) / len(v)) ** 2 for x in v) / len(v)))(hard_values) if hard_values else None,
            "by_round": {str(k): sum(v) / len(v) for k, v in sorted(by_round.items())}}


def exploits(row, slot, rng, scorer):
    """Synthetic attempts on one problem: what a policy could learn to do instead of testing."""
    from atpgllm.training.reasoning_audit import structure
    s = structure(row["netlist"])
    inputs = sorted(s["inputs"])
    fault = row["fault"]
    site = fault.split(" ", 1)[1]
    internal = sorted(s["nets"] - s["inputs"])

    def attempt(text, vector, outputs=None, call=True):
        messages = [{"role": "assistant", "content": text, **({"tool_calls": [{"function": {
            "name": "fault_simulation_tool", "arguments": {"input_vector": vector, "output_vector": outputs or {}}}}]} if call else {})}]
        if call:  # the table the simulator would have returned, so that claims can be checked
            good, bad, _ = scorer.tables(row["netlist"], fault, vector)
            messages.append({"role": "tool", "content": json.dumps({
                "Good Machine": {k: str(v) for k, v in good.items()}, "Bad Machine": {k: str(v) for k, v in bad.items()}})})
        return {"vector": vector, "messages": messages, "observations": [{"vector": vector}] if call else [],
                "final_answer": f'DETECTED_FAULTS: "{fault}"', "reward_components": {}}

    zeros = {n: 0 for n in inputs}
    plain = (slot.get("messages") or [{}])
    first = next((m for m in plain if m["role"] == "assistant"), {"content": ""})
    return {
        "all_zeros": attempt("Set everything to 0.", zeros),
        "all_x": attempt("Leave the inputs open.", {n: "x" for n in inputs}),
        "copy_netlist": attempt(row["netlist"], zeros),
        "name_spam": attempt(" ".join(sorted(s["gates"]) + internal), zeros),
        "value_spam": attempt(" ".join(f"{n} = {rng.randint(0, 1)}." for n in internal), zeros),
        "both_values_spam": attempt(" ".join(f"{n} = 0. {n} = 1." for n in internal), zeros),
        "long_reasoning": attempt((first.get("content") or "") * 5, slot.get("vector") or zeros),
        "claim_without_call": attempt(f"{site} = 1", zeros, call=False),
        "predict_all_zero_outputs": attempt("Outputs stay 0.", zeros, {o: 0 for o in s["outputs"]}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", default="runs/rft_v4_a100")
    parser.add_argument("--rounds", default="1,3,5")
    parser.add_argument("--per_level", type=int, default=40, help="Problems per level and round")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", required=True)
    parser.add_argument("--no_distances", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "attempts.jsonl").write_text("")
    scorer = Scorer(load_signals())
    rng = random.Random(args.seed)
    groups, attempts = [], 0
    exploit_scores = defaultdict(lambda: defaultdict(list))
    honest = defaultdict(list)
    diversity = defaultdict(list)
    for n in map(int, args.rounds.split(",")):
        for key, (row, slots) in rows_of(Path(args.run_dir), n, args.per_level, args.seed).items():
            try:
                scored = [scorer.score(row, slot, distances=not args.no_distances) for slot in slots]
            except Exception as exc:
                print(f"[axes] skipped {key[1]}: {exc!r}", flush=True)
                continue
            groups.append((row["difficulty"], n, scored))
            with open(out / "attempts.jsonl", "a") as sink:
                for attempt in scored:
                    sink.write(json.dumps({"round": n, "level": row["difficulty"], "fault": row["fault"],
                                           "module": row["module_name"], **attempt}) + "\n")
            attempts += len(scored)
            for attempt, slot in zip(scored, slots):
                attempt["distinct_vector"] = json.dumps(slot.get("vector"), sort_keys=True)
            if rng.random() < 0.25:
                for name, fake in exploits(row, slots[0], rng, scorer).items():
                    exploit_scores[name]["__group__"].append(1)
                    # Distances are costly and depend only on the vector.
                    for axis, value in scorer.score(row, fake, distances=name in ("all_zeros", "all_x")).items():
                        if isinstance(value, (int, float)):
                            exploit_scores[name][axis].append(value)
                for attempt in scored:
                    for axis, value in attempt.items():
                        if isinstance(value, (int, float)):
                            honest[axis].append(value)
        print(f"[axes] round {n}: {attempts} attempts so far, seconds {dict(scorer.seconds)}", flush=True)

    axes = ["activated", "propagation", "controlling", "activate_distance", "detect_distance", "output_prediction",
            "output_prediction_accuracy", "claim_truth", "claim_score", "structural_validity", "names_exist",
            "implication_truth", "verdict_truth", "site_target_correct", "site_target_realized"]
    table = [axis_statistics(groups, axis) for axis in axes]
    # Favouring an axis: how many distinct vectors stay among the top half of each group.
    for row in table:
        kept, baseline = [], []
        for _level, _round, group in groups:
            known = [a for a in group if a.get(row["axis"]) is not None]
            if len(known) < 4:
                continue
            ordered = sorted(known, key=lambda a: -a[row["axis"]])[:len(known) // 2]
            kept.append(len({a["distinct_vector"] for a in ordered}) / len(ordered))
            baseline.append(len({a["distinct_vector"] for a in known}) / len(known))
        row["distinct_share_top_half"] = sum(kept) / len(kept) if kept else None
        row["distinct_share_all"] = sum(baseline) / len(baseline) if baseline else None
    exploit_table = {name: {axis: sum(v) / len(v) for axis, v in values.items() if axis != "__group__" and v}
                     for name, values in exploit_scores.items()}
    honest_table = {axis: {"mean": sum(v) / len(v), "n": len(v)} for axis, v in honest.items()}
    per_attempt = {k: v / max(attempts, 1) for k, v in scorer.seconds.items()}
    result = {"rounds": args.rounds, "per_level": args.per_level, "attempts": attempts, "groups": len(groups),
              "axes": table, "exploits": exploit_table, "honest": honest_table, "seconds_per_attempt": per_attempt}
    (out / "axes.json").write_text(json.dumps(result, indent=1))
    fmt = lambda v: "-" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)
    print("axis | coverage | signal | all-fail signal | predicts solved | corr det | corr near miss | detecting at max | corr length | hard mean | std")
    for row in table:
        print(" | ".join(fmt(row[k]) for k in ("axis", "group_coverage", "groups_with_signal", "all_fail_groups_with_signal",
                                                "failure_score_predicts_solved", "within_group_corr_detection",
                                                "within_group_corr_near_miss", "detecting_at_group_max",
                                                "within_group_corr_length", "hard_mean", "hard_std")))
    print(json.dumps(per_attempt))


if __name__ == "__main__":
    main()
