"""Paper style analysis of one evaluation, and figures over all evaluations of a run.

An evaluation is the ``slots*.jsonl`` written by ``scripts/rft/evaluate.py``.
:func:`analyze` turns it into tables per difficulty bin:

* effort: pass@k against simulator executions, with the uniform random curve;
  "strict" counts only answers whose vector a real tool call verified;
* mechanism: activation and propagation given activation, per problem and then
  per bin, against the random reference;
* inputs: decisive and other care bits;
* outcomes: what happened to the slots;
* reasoning: the log only audit of :mod:`atpgllm.training.reasoning_audit`.

Nothing here needs a GPU or TetraMAX. Decisive inputs come from the Python
simulator, which handles x, and every report counts where it disagrees with the
detection stored in the slot.
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

LEVELS = ("easy", "medium", "hard", "very_hard", "random_never_detects")
BINS = LEVELS + ("all", "hard_three")
CALL_BUCKETS = ("0", "1", "2", "3", "4+")
GROUPS = ("site_input", "site_behind_gates", "family_seen", "family_unseen")
K_VALUES = (1, 2, 4, 8, 16)
NAMES = {"easy": "Easy", "medium": "Medium", "hard": "Hard", "very_hard": "Very hard",
         "random_never_detects": "Random never detects", "all": "All problems", "hard_three": "Hard three levels",
         "site_input": "Site at an input", "site_behind_gates": "Site behind gates",
         "family_seen": "Family seen in training", "family_unseen": "Family unseen in training"}


def pass_at_k(n: int, c: int, k: int) -> float:
    """Chance that k of the n answers, drawn without replacement, include one of the c detecting ones."""
    if k > n:
        return float("nan")
    return 1.0 if n - c < k else 1 - comb(n - c, k) / comb(n, k)


def _mean(values):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(values) / len(values) if values else float("nan")


def load_slots(directory) -> list[dict]:
    """Every whole line of a run's shard files."""
    slots = []
    for path in sorted(Path(directory).glob("slots*.jsonl")):
        for line in open(path):
            if line.endswith("\n"):
                try:
                    slots.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return slots


def slot_problem_id(slot) -> str:
    for key in ("example_id", "evaluation_problem_id"):
        if slot.get(key):
            return slot[key]
    return "|".join(map(str, slot["problem"]))


def answer_detected(slot) -> bool:
    """The final answer detects the fault, as the paper counts it."""
    r = slot.get("reward_components") or {}
    if r.get("search_failure_logonly", 0) > 0 or r.get("simulator_error_logonly", 0) > 0:
        return False
    return r.get("fault_detected_by_pred_input_vector_acc", r.get("fault_detected_by_pred_input_vector_acc_logonly", 0)) == 1


def answer_activated(slot) -> bool:
    r = slot.get("reward_components") or {}
    return r.get("fault_site_activated_acc_logonly", r.get("activation", 0)) == 1


def real_calls(slot) -> int:
    if slot.get("observations") is not None:
        return len(slot["observations"])
    if slot.get("messages"):
        return sum(m["role"] == "tool" for m in slot["messages"])
    return (slot.get("usage") or {}).get("tool_calls", 0)


def _care(vector) -> dict:
    return {net: str(v) for net, v in (vector or {}).items() if str(v) in ("0", "1")}


def _as_vector(raw) -> dict:
    return json.loads(raw) if isinstance(raw, str) else dict(raw or {})


def verified_by_call(slot) -> bool:
    """A real tool call in the conversation simulated the answer's care bits."""
    answer = _care(slot.get("vector"))
    if not answer:
        return False
    if slot.get("observations") is not None:
        return any(_care(o.get("vector")) == answer for o in slot["observations"])
    return real_calls(slot) > 0


# ------------------------------------------------------------------ simulation


class Simulator:
    """Three valued Python fault simulation of many vectors, with compiled netlists cached."""

    def __init__(self):
        os.environ.setdefault("FAULT_SIM_BACKEND", "fast")
        from atpgllm.training._paths import ensure_data_preprocessing_on_path
        root = ensure_data_preprocessing_on_path()
        import fault_sim
        from atpgllm.training.reward_function_factory import RewardFunctionFactory as RF
        self.fs, self.rf = fault_sim, RF
        self.gates = json.loads((Path(fault_sim.__file__).parent / "sim_config.json").read_text())["gate_funcs"]
        self.cache = {}

    def compile(self, text):
        if text not in self.cache:
            if len(self.cache) > 256:
                self.cache.clear()
            self.cache[text] = self.fs.OptimizedNetlist(text, self.gates, self.rf.DECL_RE, self.rf.NAME_RE)
        return self.cache[text]

    def outcome(self, text, fault, vector):
        """``(activated, detected)``; a missing input reads as x."""
        net_list = self.compile(text)
        full = {net: (int(vector[net]) if str(vector.get(net)) in ("0", "1") else "x") for net in net_list.input_nets}
        frame = self.fs.fast_fault_sim(full, {o: 0 for o in net_list.output_nets}, fault, net_list)
        good, bad = frame["Good Machine"], frame["Bad Machine"]
        site, stuck = fault.split(" ", 1)[1], fault[2]
        activated = site in good.index and str(good[site]) in ("0", "1") and str(good[site]) != stuck
        detected = any(o in good.index and str(good[o]) in ("0", "1") and str(bad[o]) in ("0", "1")
                       and str(good[o]) != str(bad[o]) for o in net_list.output_nets)
        return activated, detected

    def decisive(self, text, fault, vector):
        """Care bits whose single flip loses detection, or None when the vector does not detect."""
        split = self.decisive_split(text, fault, vector)
        return None if split is None else split["activating"] + split["propagating"]

    def decisive_split(self, text, fault, vector):
        """Why each decisive bit matters, or None when the vector does not detect.

        ``activating``: its flip takes the fault site off the excited value.
        ``propagating``: the site stays excited but the effect no longer reaches an output.
        """
        if not self.outcome(text, fault, vector)[1]:
            return None
        split = {"activating": [], "propagating": []}
        for net, value in _care(vector).items():
            activated, detected = self.outcome(text, fault, {**vector, net: 1 - int(value)})
            if not detected:
                split["propagating" if activated else "activating"].append(net)
        return split


def site_depth(net_list, net):
    """Longest path in gates from an input to ``net``, and the shortest from ``net`` to an output."""
    deps = {a: set(b) for a, b in net_list.net_dependencies.items()}
    seen, frontier, depth, longest = {net}, {net}, 0, 0
    while frontier:
        nxt = {u for x in frontier for u in deps.get(x, ()) if u not in seen}
        seen |= nxt
        depth += 1
        if nxt:
            longest = depth
        frontier = nxt
    fanout = defaultdict(set)
    for output, inputs in deps.items():
        for u in inputs:
            fanout[u].add(output)
    outputs = set(net_list.output_nets)
    seen, frontier, depth, to_output = {net}, {net}, 0, None
    while frontier and to_output is None:
        if frontier & outputs:
            to_output = depth
            break
        nxt = {u for x in frontier for u in fanout.get(x, ()) if u not in seen}
        seen |= nxt
        frontier = nxt
        depth += 1
    return longest, to_output


def reference_entry(row: dict, simulator: Simulator, *, seen_families=None, analyzer=None) -> dict:
    """What the analysis needs to know about one problem, from a dataset row."""
    from atpgllm.training.fixed_eval import family_of
    text, fault = row["netlist"], row["fault"]
    net_list = simulator.compile(text)
    site = fault.split(" ", 1)[1]
    depth, to_output = site_depth(net_list, site)
    label = _as_vector(row.get("input_vector"))
    split = simulator.decisive_split(text, fault, label) if label else None
    decisive = None if split is None else split["activating"] + split["propagating"]
    family = family_of(row.get("module_name"))
    zeros = simulator.outcome(text, fault, {net: 0 for net in net_list.input_nets})
    ones = simulator.outcome(text, fault, {net: 1 for net in net_list.input_nets})
    entry = {
        "level": row.get("difficulty"), "family": family,
        "family_in_train": None if seen_families is None else family in seen_families,
        "module_name": row.get("module_name"), "netlist_id": row.get("netlist_id"),
        "circuit_id": row.get("circuit_id"), "fault": fault, "p": row.get("random_detection_probability"),
        "site": ("input" if site in net_list.input_nets else "output" if site in net_list.output_nets else "internal"),
        "site_depth": depth, "site_to_output": to_output, "num_inputs": len(net_list.input_nets),
        "num_gates": len(net_list.net_dependencies), "label": label,
        "label_detects": decisive is not None, "care": list(_care(label)), "decisive": decisive,
        "decisive_activating": None if split is None else split["activating"],
        "decisive_propagating": None if split is None else split["propagating"],
        "dont_care": [net for net in net_list.input_nets if net not in _care(label)],
        "all_zeros_detects": zeros[1], "all_ones_detects": ones[1],
        "netlist": text,
    }
    if analyzer is not None:
        random_ref = analyzer.analyze(row)
        entry["random_activation"] = random_ref.get("activation_probability")
        entry["random_observability"] = random_ref.get("conditional_observability")
    return entry


# ------------------------------------------------------------------ analysis


def _members(reference: dict) -> dict:
    """Problem ids of every bin and group."""
    members = defaultdict(list)
    for pid, ref in reference.items():
        members["all"].append(pid)
        if ref.get("level") in LEVELS:
            members[ref["level"]].append(pid)
            if ref["level"] in LEVELS[2:]:
                members["hard_three"].append(pid)
        if ref.get("site") == "input":
            members["site_input"].append(pid)
        elif ref.get("site") is not None:
            members["site_behind_gates"].append(pid)
        if ref.get("family_in_train") is True:
            members["family_seen"].append(pid)
        elif ref.get("family_in_train") is False:
            members["family_unseen"].append(pid)
    return members


def _bootstrap(values_by_pid, pids, reference, statistic, repeats=1000, seed=0):
    """95% interval over resampled circuits of ``statistic`` applied to per problem values."""
    by_circuit = defaultdict(list)
    for pid in pids:
        if pid in values_by_pid:
            by_circuit[reference[pid].get("circuit_id") or pid].append(values_by_pid[pid])
    circuits = list(by_circuit)
    if not circuits:
        return None, None
    rng = random.Random(seed)
    results = sorted(statistic([v for c in rng.choices(circuits, k=len(circuits)) for v in by_circuit[c]])
                     for _ in range(repeats))
    return results[int(0.025 * repeats)], results[int(0.975 * repeats) - 1]


def analyze(slots: list[dict], reference: dict, simulator: Simulator | None = None, *, tag: str = "",
            source: str = "benchmark", audit=None, flips_per_problem: int = 16) -> dict:
    """All tables of one evaluation. ``reference`` maps problem ids to :func:`reference_entry` dicts."""
    by_calls = defaultdict(lambda: [0, 0])  # (problem, calls bucket) -> slots, detected
    per = defaultdict(lambda: dict(n=0, c=0, strict=0, vec=0, act=0, actdet=0, calls=0, execs=0, novec=0, nocall=0,
                                   retry=0, claims=0, nondetecting=0, truncated=0, vectors=set(), detecting=[],
                                   answers=[]))
    unknown = 0
    for slot in slots:
        pid = slot_problem_id(slot)
        if pid not in reference:
            unknown += 1
            continue
        x = per[pid]
        x["n"] += 1
        detected = answer_detected(slot)
        x["c"] += detected
        x["strict"] += detected and verified_by_call(slot)
        usage = slot.get("usage") or {}
        x["execs"] += usage.get("simulator_executions", 0)
        calls = real_calls(slot)
        x["calls"] += calls
        by_calls[pid, str(calls) if calls < 4 else "4+"][0] += 1
        by_calls[pid, str(calls) if calls < 4 else "4+"][1] += detected
        x["retry"] += calls >= 2
        x["truncated"] += slot.get("status") == "EXHAUSTED"
        vector = slot.get("vector")
        if not vector:
            x["novec"] += 1
            continue
        x["nocall"] += calls == 0
        x["vec"] += 1
        activated = answer_activated(slot)
        x["act"] += activated
        x["actdet"] += activated and detected
        care = tuple(sorted(_care(vector).items()))
        x["vectors"].add(care)
        answer = dict(vector)
        x["answers"].append(answer)
        if detected:
            x["detecting"].append(answer)
        else:
            x["nondetecting"] += 1
            m = re.search(r'DETECTED_FAULTS:\s*"([^"]*)"', slot.get("final_answer") or "")
            x["claims"] += bool(m and reference[pid].get("fault") in m.group(1))

    members = _members(reference)
    # A resumed generation can leave a problem with extra slots; k follows the usual count.
    usual = Counter(x["n"] for x in per.values()).most_common(1)
    result = {"tag": tag, "source": source, "problems": len(per), "slots": len(slots), "unknown_problem_slots": unknown,
              "slots_per_problem": usual[0][0] if usual else 0,
              "effort": [], "mechanism": [], "inputs": [], "outcomes": [], "calls": [], "reward": [], "reasoning": []}
    ks = [k for k in K_VALUES if k <= result["slots_per_problem"]]

    for name in BINS + GROUPS:
        pids = [p for p in members.get(name, []) if p in per]
        if not pids:
            continue
        execs_per_slot = sum(per[p]["execs"] for p in pids) / max(sum(per[p]["n"] for p in pids), 1)
        for k in ks:
            values = {p: pass_at_k(per[p]["n"], per[p]["c"], k) for p in pids}
            strict = {p: pass_at_k(per[p]["n"], per[p]["strict"], k) for p in pids}
            low, high = _bootstrap(values, pids, reference, _mean)
            m = k * execs_per_slot
            ps = [reference[p].get("p") for p in pids]
            result["effort"].append({
                "tag": tag, "bin": name, "k": k, "problems": len(pids), "pass_at_k": _mean(values.values()),
                "ci_low": low, "ci_high": high, "strict_pass_at_k": _mean(strict.values()),
                "executions_per_slot": execs_per_slot, "executions": m,
                "uniform_at_same_executions": _mean([1 - (1 - p) ** m for p in ps if p is not None]),
                "uniform_at_k_vectors": _mean([1 - (1 - p) ** k for p in ps if p is not None]),
                # Two constant vectors, as a reference for what needs no reasoning at all.
                "all_zeros": _mean([float(reference[p]["all_zeros_detects"]) for p in pids if "all_zeros_detects" in reference[p]]),
                "all_zeros_or_ones": _mean([float(reference[p]["all_zeros_detects"] or reference[p]["all_ones_detects"])
                                            for p in pids if "all_zeros_detects" in reference[p]])})

        def act_value(p):
            return per[p]["act"] / per[p]["vec"] if per[p]["vec"] else None

        def prop_value(p):
            return per[p]["actdet"] / per[p]["act"] if per[p]["act"] else None

        result["mechanism"].append({
            "tag": tag, "bin": name, "problems": len(pids),
            "activation": _mean([act_value(p) for p in pids]),
            "activation_random": _mean([reference[p].get("random_activation") for p in pids]),
            # As in the paper: problems with an activating answer and a finite random reference.
            "propagation": _mean([prop_value(p) for p in pids if reference[p].get("random_observability") is not None]),
            "propagation_random": _mean([reference[p].get("random_observability") for p in pids
                                         if prop_value(p) is not None]),
            # Every problem with an activating answer, including those random vectors never activate.
            "propagation_all": _mean([prop_value(p) for p in pids]),
            "propagation_problems": sum(prop_value(p) is not None for p in pids)})

        slots_here = max(sum(per[p]["n"] for p in pids), 1)
        chosen = set(pids)
        tally = defaultdict(lambda: [0, 0])
        for (pid, bucket), (count, hits) in by_calls.items():
            if pid in chosen:
                tally[bucket][0] += count
                tally[bucket][1] += hits
        detected_total = sum(hits for _, hits in tally.values())
        result["calls"].append({
            "tag": tag, "bin": name, "problems": len(pids), "no_detection": 1 - detected_total / slots_here,
            **{f"attempts_{bucket}_calls": tally[bucket][0] / slots_here for bucket in CALL_BUCKETS},
            **{f"detected_with_{bucket}_calls": tally[bucket][1] / slots_here for bucket in CALL_BUCKETS}})
        answers = max(sum(per[p]["vec"] for p in pids), 1)
        nondetecting = sum(per[p]["nondetecting"] for p in pids)
        result["outcomes"].append({
            "tag": tag, "bin": name, "problems": len(pids),
            "no_vector": sum(per[p]["novec"] for p in pids) / slots_here,
            "no_real_call": sum(per[p]["nocall"] for p in pids) / answers,
            "truncated": sum(per[p]["truncated"] for p in pids) / slots_here,
            "calls_per_slot": sum(per[p]["calls"] for p in pids) / slots_here,
            "retried": sum(per[p]["retry"] for p in pids) / slots_here,
            "distinct_vectors_per_problem": _mean([len(per[p]["vectors"]) for p in pids]),
            "false_detection_claims": sum(per[p]["claims"] for p in pids) / nondetecting if nondetecting else float("nan"),
            "executions_per_slot": execs_per_slot})

    # Decisive and other inputs, and the simulator cross check.
    disagreements = checked = 0
    decisive_rows = {}
    if simulator is not None:
        for pid, x in per.items():
            ref = reference[pid]
            text = ref.get("netlist")
            if not text:
                continue
            label = ref.get("label") or {}
            label_decisive = set(ref.get("decisive") or [])
            label_other = set(ref.get("care") or []) - label_decisive
            row = dict(match_decisive=[], match_other=[], x_decisive=[], x_other=[], share=[], agree_decisive=[], agree_other=[],
                       match_activating=[], match_propagating=[], x_activating=[], x_propagating=[], x_dont_care=[],
                       share_activating=[], share_propagating=[])
            label_activating = set(ref.get("decisive_activating") or [])
            label_propagating = set(ref.get("decisive_propagating") or [])
            dont_care = ref.get("dont_care") or []
            for answer in x["answers"]:
                for kind, nets in (("activating", label_activating), ("propagating", label_propagating)):
                    for net in nets:
                        stated = str(answer.get(net, "x")) in ("0", "1")
                        row[f"x_{kind}"].append(not stated)
                        if stated:
                            row[f"match_{kind}"].append(str(answer[net]) == str(label[net]))
                row["x_dont_care"].extend(str(answer.get(net, "x")) not in ("0", "1") for net in dont_care)
                for net in label_decisive:
                    row["x_decisive"].append(str(answer.get(net, "x")) not in ("0", "1"))
                    if str(answer.get(net, "x")) in ("0", "1"):
                        row["match_decisive"].append(str(answer[net]) == str(label[net]))
                for net in label_other:
                    row["x_other"].append(str(answer.get(net, "x")) not in ("0", "1"))
                    if str(answer.get(net, "x")) in ("0", "1"):
                        row["match_other"].append(str(answer[net]) == str(label[net]))
            distinct = {tuple(sorted(_care(v).items())): v for v in x["detecting"]}
            for vector in list(distinct.values())[:flips_per_problem]:
                checked += 1
                split = simulator.decisive_split(text, ref["fault"], vector)
                if split is None:
                    disagreements += 1
                    continue
                found = split["activating"] + split["propagating"]
                care = _care(vector)
                if care:
                    row["share"].append(len(found) / len(care))
                    row["share_activating"].append(len(split["activating"]) / len(care))
                    row["share_propagating"].append(len(split["propagating"]) / len(care))
                decisive_set = set(found)
                for other in x["answers"]:
                    if other is vector:
                        continue
                    for net in care:
                        same = str(other.get(net, "x")) == care[net]
                        row["agree_decisive" if net in decisive_set else "agree_other"].append(same)
            decisive_rows[pid] = row
        # Answers graded as failing must fail in the Python simulator too.
        for pid, x in per.items():
            text = reference[pid].get("netlist")
            if not text:
                continue
            detecting = {tuple(sorted(_care(v).items())) for v in x["detecting"]}
            for vector in x["answers"]:
                key = tuple(sorted(_care(vector).items()))
                if key not in detecting:
                    detecting.add(key)
                    checked += 1
                    disagreements += simulator.outcome(text, reference[pid]["fault"], vector)[1]

        for name in BINS + GROUPS:
            pids = [p for p in members.get(name, []) if p in decisive_rows]
            if not pids:
                continue
            per_problem = lambda key: _mean([_mean(decisive_rows[p][key]) for p in pids])
            result["inputs"].append({
                "tag": tag, "bin": name, "problems": len(pids),
                "label_decisive_share": _mean([len(reference[p]["decisive"] or []) / len(reference[p]["care"])
                                               for p in pids if reference[p].get("care") and reference[p].get("label_detects")]),
                "answer_decisive_share": per_problem("share"),
                "label_match_decisive": per_problem("match_decisive"), "label_match_other": per_problem("match_other"),
                "x_rate_decisive": per_problem("x_decisive"), "x_rate_other": per_problem("x_other"),
                "label_activating_share": _mean([len(reference[p]["decisive_activating"] or []) / len(reference[p]["care"])
                                                 for p in pids if reference[p].get("care") and reference[p].get("label_detects")
                                                 and "decisive_activating" in reference[p]]),
                "label_propagating_share": _mean([len(reference[p]["decisive_propagating"] or []) / len(reference[p]["care"])
                                                  for p in pids if reference[p].get("care") and reference[p].get("label_detects")
                                                  and "decisive_propagating" in reference[p]]),
                "answer_activating_share": per_problem("share_activating"),
                "answer_propagating_share": per_problem("share_propagating"),
                "label_match_activating": per_problem("match_activating"),
                "label_match_propagating": per_problem("match_propagating"),
                "x_rate_activating": per_problem("x_activating"), "x_rate_propagating": per_problem("x_propagating"),
                "dont_care_left_x": per_problem("x_dont_care"),
                "agreement_decisive": per_problem("agree_decisive"), "agreement_other": per_problem("agree_other")})
    # The training objectives as the reward in force scored these attempts.
    objectives = ("detection", "activation", "propagation", "controlling", "fidelity", "format", "uniqueness")
    scored = defaultdict(list)
    for slot in slots:
        pid = slot_problem_id(slot)
        if pid in reference:
            scored[pid].append(slot.get("reward_components") or {})
    for name in BINS + GROUPS:
        pids = [p for p in members.get(name, []) if p in scored]
        if pids:
            present = [k for k in objectives if any(k in c for p in pids for c in scored[p])]
            if present:
                result["reward"].append({"tag": tag, "bin": name, "problems": len(pids), **{
                    key: _mean([_mean([c.get(key, 0.0) for c in scored[p]]) for p in pids]) for key in present}})
    result["simulator_checks"] = checked
    result["simulator_disagreements"] = disagreements

    if audit is not None:
        scored = defaultdict(list)
        for slot in slots:
            pid = slot_problem_id(slot)
            if pid in reference:
                scored[pid].append(audit(slot, reference[pid]))
        keys = sorted({k for rows in scored.values() for row in rows for k in row})
        for name in BINS + GROUPS:
            pids = [p for p in members.get(name, []) if p in scored]
            if pids:
                result["reasoning"].append({"tag": tag, "bin": name, "problems": len(pids), **{
                    key: _mean([_mean([row.get(key) for row in scored[p]]) for p in pids]) for key in keys}})
    return result


TABLES = ("effort", "mechanism", "inputs", "outcomes", "calls", "reward", "reasoning")


def write_tables(result: dict, directory) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metrics.json").write_text(json.dumps(result, indent=1, default=str))
    for table in TABLES:
        rows = result.get(table) or []
        if rows:
            with open(directory / f"{table}.csv", "w", newline="") as sink:
                writer = csv.DictWriter(sink, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)


def headline(result: dict, name: str = "all") -> dict:
    """pass@1 and pass@16 of one bin, for ``round/*`` style scalars."""
    out = {}
    for row in result["effort"]:
        if row["bin"] == name and row["k"] in (1, 16):
            out[f"pass_at_{row['k']}"] = row["pass_at_k"]
            out[f"strict_pass_at_{row['k']}"] = row["strict_pass_at_k"]
    return out


# ------------------------------------------------------------------ figures

# One hue, light to dark, so an earlier model reads as lighter than a later one.
INK, INK2, GRID, GRAY = "#0b0b0b", "#52514e", "#e4e3df", "#7a7974"
ACCENT = "#2a78d6"


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "Nimbus Roman No9 L", "STIXGeneral"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": INK2,
        "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02})
    return plt


def _grid(ax, axis="y"):
    ax.grid(True, axis=axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def _ramp(count):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#a9c9f0", "#2a78d6", "#0d366b"])
    return [cmap(i / max(count - 1, 1)) for i in range(count)]


def effort_figure(history: list[dict], reference: dict, path) -> None:
    """Detection probability against simulator executions per problem, one line per evaluated model."""
    plt = _style()
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.1), sharey=True)
    colors = _ramp(len(history))
    members = _members(reference)
    for ax, name in zip(axes.flat, LEVELS + ("all",)):
        ps = [reference[p].get("p") for p in members.get(name, [])]
        ps = [p for p in ps if p is not None]
        ms = [2 ** (i / 20) for i in range(-20, 121)]
        if ps:
            ax.plot(ms, [sum(1 - (1 - p) ** m for p in ps) / len(ps) for m in ms], color=GRAY, lw=1.2, ls=(0, (4, 2)),
                    label="Uniform random", zorder=2)
        constants = [reference[p] for p in members.get(name, []) if "all_zeros_detects" in reference[p]]
        if constants:
            zero = sum(c["all_zeros_detects"] for c in constants) / len(constants)
            either = sum(c["all_zeros_detects"] or c["all_ones_detects"] for c in constants) / len(constants)
            ax.plot([1, 2], [zero, either], color=INK2, lw=0.9, ls=":", marker="D", ms=3.5, mfc="white", mec=INK2, mew=0.8,
                    label="All zeros, then all ones", zorder=2)
        for color, result in zip(colors, history):
            rows = [r for r in result["effort"] if r["bin"] == name]
            if rows:
                ax.plot([r["executions"] for r in rows], [r["pass_at_k"] for r in rows], color=color, lw=1.5, marker="o",
                        ms=3.5, mec="white", mew=0.5, label=result["tag"], zorder=3)
        ax.set_xscale("log", base=2)
        ax.set_xlim(0.5, 48)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([0.5, 1, 2, 4, 8, 16, 32])
        ax.set_xticklabels(["0.5", "1", "2", "4", "8", "16", "32"])
        ax.set_title(f"{NAMES[name]} ($n$ = {len(members.get(name, []))})", loc="left", pad=3)
        _grid(ax, "both")
    for ax in axes[1]:
        ax.set_xlabel("Simulator executions per problem")
    for ax in axes[:, 0]:
        ax.set_ylabel("Detection probability, pass@$k$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 8), frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(h_pad=1.0, w_pad=0.8)
    _save(fig, path)


def mechanism_figure(history: list[dict], path) -> None:
    """Activation, propagation and decisive inputs over evaluated models, small multiples by level."""
    plt = _style()
    fig, axes = plt.subplots(4, 5, figsize=(7.16, 5.8), sharex=True, sharey=True)
    tags = [r["tag"] for r in history]
    xs = list(range(len(tags)))

    def series(table, name, key):
        out = []
        for result in history:
            rows = [r for r in result.get(table, []) if r["bin"] == name]
            out.append(rows[0].get(key, float("nan")) if rows else float("nan"))
        return out

    rows_spec = (("mechanism", "activation", "activation_random", "Activation", "Answers", "Random vector"),
                 ("mechanism", "propagation", "propagation_random", "Propagation, if activated", "Answers", "Random vector"),
                 ("inputs", "label_match_activating", "label_match_propagating", "Decisive bits matching the label",
                  "Activating", "Propagating"),
                 ("inputs", "x_rate_decisive", "dont_care_left_x", "Left as x", "Decisive bits", "Don't-care bits"))
    for row, (table, key, baseline, ylabel, label, base_label) in enumerate(rows_spec):
        for column, name in enumerate(LEVELS):
            ax = axes[row, column]
            ax.plot(xs, series(table, name, baseline), color=GRAY, lw=1.2, ls=(0, (4, 2)), label=base_label, zorder=2)
            ax.plot(xs, series(table, name, key), color=ACCENT, lw=1.5, marker="o", ms=3.5, mec="white", mew=0.5,
                    label=label, zorder=3)
            ax.set_ylim(0, 1)
            _grid(ax)
            if row == 0:
                ax.set_title(NAMES[name], loc="left", pad=3)
            if column == 0:
                ax.set_ylabel(ylabel)
                if row != 1:  # the same two lines as the row above
                    ax.legend(frameon=False, fontsize=6, loc="upper left")
            if row == len(rows_spec) - 1:
                ax.set_xticks(xs)
                ax.set_xticklabels(tags, rotation=60, ha="right", fontsize=6)
    fig.tight_layout(h_pad=0.8, w_pad=0.4)
    _save(fig, path)


def calls_figure(history: list[dict], path) -> None:
    """How attempts ended, by the number of real tool calls, one bar per evaluated model and level."""
    plt = _style()
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 3.9), sharex=True, sharey=True)
    tags = [r["tag"] for r in history]
    segments = (("detected_with_0_calls", "Detected, no real call", "#b7d3f6"),
                ("detected_with_1_calls", "Detected with 1 call", "#2a78d6"),
                ("detected_with_2_calls", "Detected with 2 calls", "#184f95"),
                ("detected_with_3_calls", "Detected with 3 calls", "#104281"),
                ("detected_with_4+_calls", "Detected with 4 or more", "#0d366b"),
                ("no_detection", "No detection", GRID))
    for ax, name in zip(axes.flat, LEVELS + ("all",)):
        bottoms = [0.0] * len(history)
        for key, label, color in segments:
            values = []
            for result in history:
                rows = [r for r in result.get("calls", []) if r["bin"] == name]
                values.append(rows[0].get(key, 0.0) if rows else 0.0)
            ax.bar(range(len(tags)), values, bottom=bottoms, color=color, width=0.72, edgecolor="white", linewidth=0.6, label=label)
            bottoms = [b + v for b, v in zip(bottoms, values)]
        ax.set_ylim(0, 1)
        ax.set_title(NAMES[name], loc="left", pad=3)
        ax.set_xticks(range(len(tags)))
        ax.set_xticklabels(tags, rotation=60, ha="right", fontsize=6)
        _grid(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("Share of attempts")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(h_pad=0.8, w_pad=0.4)
    _save(fig, path)


def _save(fig, path):
    import matplotlib.pyplot as plt
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)
