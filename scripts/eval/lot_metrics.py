"""CPU-only Language-of-Test metrics. No reward or model imports.

Missing slots are failures in end-to-end metrics; unknown verification is
retained separately. Bootstrap units must be declared by the caller.
"""
from __future__ import annotations

import ast
from collections import defaultdict
import math
import random
import re


def assignments(value, names):
    if not isinstance(value, str):
        raise ValueError("Missing assignment field")
    if value.strip().startswith("{"):
        node = ast.parse(value.strip(), mode="eval").body
        if not isinstance(node, ast.Dict):
            raise ValueError("Expected assignment object")
        pairs = [(ast.literal_eval(k), ast.literal_eval(v)) for k, v in zip(node.keys, node.values)]
    else:
        pairs = [tuple(x.strip() for x in item.rsplit(":", 1)) for item in value.split(",")]
    result = {}
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError("Expected net: bit")
        key, bit = pair
        if not isinstance(key, str) or key in result or type(bit) not in (str, int) or bit not in (0, 1, "0", "1"):
            raise ValueError("Duplicate, nonbinary, or invalid assignment")
        result[key] = int(bit)
    if set(result) != set(names):
        raise ValueError("Assignment must cover exactly the canonical ports")
    return {n: result[n] for n in names}


def parse_answer(text, inputs, outputs):
    # Match the deployed explicit-field interface; never mine a tool table.
    text = text.rsplit("</tool_response>", 1)[-1].rsplit("</think>", 1)[-1]
    fields = defaultdict(list)
    for name, value in re.findall(r'\b(INPUT_VECTOR|EXPECTED_OUTPUT|DETECTED_FAULTS):\s*"([^"]*)"', text):
        fields[name].append(value)
    result = {"input": None, "expected": None, "errors": []}
    for field, key, names in (("INPUT_VECTOR", "input", inputs), ("EXPECTED_OUTPUT", "expected", outputs)):
        try:
            if len(fields[field]) != 1:
                raise ValueError("Missing or repeated " + field)
            result[key] = assignments(fields[field][0], names)
        except (ValueError, TypeError, SyntaxError) as exc:
            result["errors"].append(str(exc))
    return result


def outcomes(parsed, good, bad):
    vx = parsed["input"] is not None
    good_known = good is not None and bool(good) and all(type(v) is int and v in (0, 1) for v in good.values())
    known = good_known and bad is not None and set(good) == set(bad)
    known = known and all(type(v) is int and v in (0, 1) for v in bad.values())
    detection = vx and known and good != bad
    sound = vx and good_known and parsed["expected"] == good
    return {"Vx": vx, "D": bool(detection), "S": bool(sound), "U": bool(detection and sound),
            "verification_known": bool(known)}


def pass_at_k(n, c, k):
    if not all(type(v) is int for v in (n, c, k)) or not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError("Require 0 <= c <= n and 1 <= k <= n")
    return 1.0 if n - c < k else 1 - math.prod((n - c - j) / (n - j) for j in range(k))


def interval(values, *, seed=1729, repeats=2000):
    if not values:
        return None
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(repeats))
    return [means[int(.025 * repeats)], means[min(repeats - 1, int(.975 * repeats))]]


def summarize(rows, k_values=(1,), *, seed=1729):
    if not rows:
        raise ValueError("Empty benchmark")
    problems = defaultdict(list)
    circuits = defaultdict(list)
    for row in rows:
        problems[row["example_id"]].append(row)
        circuits[row["circuit_id"]].append(row)
    result = {"slots": len(rows), "problems": len(problems), "circuits": len(circuits),
              "interval_unit": "circuit; family independence not established",
              "unknown_verification_slots": sum(not r["verification_known"] for r in rows),
              "per_circuit": {}}
    for metric in ("Vx", "D", "S", "U"):
        per_circuit = defaultdict(list)
        for group in problems.values():
            per_circuit[group[0]["circuit_id"]].append(sum(r[metric] for r in group) / len(group))
        means = {c: sum(v) / len(v) for c, v in per_circuit.items()}
        result[metric] = {"micro": sum(r[metric] for r in rows) / len(rows),
                          "macro": sum(means.values()) / len(means),
                          "macro_circuit_bootstrap_95": interval(list(means.values()), seed=seed)}
        for c, value in means.items():
            result["per_circuit"].setdefault(c, {})[metric] = value
    for metric in ("D", "U"):
        for k in k_values:
            if any(len(g) < k for g in problems.values()):
                raise ValueError(f"pass@{k} exceeds saved slots; do not invent or pool deterministic repeats")
            by_circuit = defaultdict(list)
            for group in problems.values():
                by_circuit[group[0]["circuit_id"]].append(pass_at_k(len(group), sum(r[metric] for r in group), k))
            means = [sum(v) / len(v) for v in by_circuit.values()]
            values = [x for v in by_circuit.values() for x in v]
            result[f"{metric}_pass@{k}"] = {"macro": sum(means) / len(means),
                "micro": sum(values) / len(values), "macro_circuit_bootstrap_95": interval(means, seed=seed)}
    return result


def coverage_curve(detected_sets, universe):
    universe = set(universe)
    if not universe:
        raise ValueError("Empty fault universe")
    covered, curve = set(), []
    for i, detected in enumerate(detected_sets, 1):
        if not set(detected) <= universe:
            raise ValueError("Detection outside frozen fault universe")
        covered.update(detected)
        curve.append({"patterns": i, "detected": len(covered), "FC_all": len(covered) / len(universe)})
    return curve


def compact(detected_sets):
    """Deterministic greedy set cover at the original stream's final coverage."""
    remaining = set().union(*map(set, detected_sets)) if detected_sets else set()
    chosen = []
    while remaining:
        best = max(range(len(detected_sets)), key=lambda i: (len(set(detected_sets[i]) & remaining), -i))
        chosen.append(best)
        remaining.difference_update(detected_sets[best])
    return chosen
