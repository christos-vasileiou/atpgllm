"""Soft targets for the BRIDGES re-test (ports-only prompt, graph carries the circuit).

One v4 problem is a (circuit, fault) pair with one or more detecting patterns.
A single pattern's care bits are one solution among many, so the targets here
are frequencies over all patterns of the problem:

- ``care_freq[pi]``: share of patterns where input ``pi`` is a care bit (not ``x``)
- ``one_freq[pi]``: among those patterns, share where ``pi`` is 1
- ``path_freq[gate]``: share of patterns whose propagation path uses ``gate``
- ``obs_freq[po]``: share of patterns whose fault effect reaches output ``po``

Signal probability per net comes from COP over the compiled netlist (inputs at
1/2, independent inputs per gate). The fact text names only ports and values,
so it can be read without the netlist.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
from torch_geometric.data import Data

from .netlist_parser import ParsedNetlistGraph

PI_CELL = "<PI>"
PO_CELL = "<PO>"


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, Mapping) else {}
    return {}


def _names(value: Any) -> List[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [n.strip() for n in str(value or "").split(",") if n.strip()]


@dataclass
class ProblemTargets:
    """Frequencies over the detecting patterns of one (circuit, fault)."""

    fault: str
    inputs: List[str]
    outputs: List[str]
    patterns: int = 0
    care_freq: Dict[str, float] = field(default_factory=dict)
    one_freq: Dict[str, float] = field(default_factory=dict)
    path_freq: Dict[str, float] = field(default_factory=dict)
    obs_freq: Dict[str, float] = field(default_factory=dict)


def problem_targets(rows: Sequence[Mapping[str, Any]]) -> ProblemTargets:
    """Aggregate the rows of one problem into soft targets.

    Rows must share the circuit and fault. Rows without an input vector are
    skipped, and a problem with no usable row has ``patterns == 0``.
    """
    if not rows:
        raise ValueError("problem_targets needs at least one row")
    faults = {str(r.get("fault", "")).strip() for r in rows}
    if len(faults) != 1:
        raise ValueError(f"rows mix faults: {sorted(faults)}")
    care = defaultdict(int)
    ones = defaultdict(int)
    path = defaultdict(int)
    obs = defaultdict(int)
    inputs: List[str] = []
    outputs: List[str] = []
    patterns = 0
    for row in rows:
        vector = _mapping(row.get("input_vector"))
        if not vector:
            continue
        expected = _mapping(row.get("expected_output"))
        if not inputs:
            inputs = list(vector)
            outputs = list(expected)
        elif list(vector) != inputs:
            raise ValueError("rows of one problem disagree on the input order")
        patterns += 1
        for net, bit in vector.items():
            text = str(bit).strip().lower()
            if text in ("0", "1"):
                care[net] += 1
                ones[net] += text == "1"
        for gate in set(_names(row.get("fault_propagation_gates"))):
            path[gate] += 1
        output_set = set(outputs)
        for net in set(_names(row.get("fault_propagation_nets"))):
            if net in output_set:
                obs[net] += 1
    targets = ProblemTargets(fault=faults.pop(), inputs=inputs, outputs=outputs, patterns=patterns)
    if patterns:
        targets.care_freq = {n: care[n] / patterns for n in inputs}
        targets.one_freq = {n: (ones[n] / care[n]) if care[n] else 0.5 for n in inputs}
        targets.path_freq = {g: c / patterns for g, c in path.items()}
        targets.obs_freq = {n: obs[n] / patterns for n in outputs}
    return targets


def group_problems(rows: Iterable[Mapping[str, Any]]) -> Dict[tuple, List[Mapping[str, Any]]]:
    """Rows keyed by (netlist text, fault), in first-seen order.

    In v4 neither ``circuit_id`` nor ``netlist_id`` is unique per netlist
    text, so a problem is a (netlist text, fault) pair, as on the v4 card."""
    groups: Dict[tuple, List[Mapping[str, Any]]] = {}
    for row in rows:
        netlist = row.get("netlist")
        if isinstance(netlist, Mapping):
            netlist = netlist.get("netlist")
        key = (str(netlist or row.get("netlist_id", "")), str(row.get("fault", "")).strip())
        groups.setdefault(key, []).append(row)
    return groups


def _truth_table(lut_or_fn: Any, arity: int) -> List[int]:
    if isinstance(lut_or_fn, list):
        return [int(v) for v in lut_or_fn]
    table = []
    for idx in range(1 << arity):
        bits = [(idx >> j) & 1 for j in range(arity)]
        table.append(int(bool(lut_or_fn(*bits))))
    return table


def cop_signal_probability(compiled: Any, inputs: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """P(net = 1) under uniform independent inputs, COP style.

    ``compiled`` is a fast-simulator ``OptimizedNetlist`` (topologically
    sorted ``instructions``). Reconvergence is ignored, as in COP.
    """
    prob: Dict[str, float] = {}
    for net in inputs if inputs is not None else getattr(compiled, "input_nets", []):
        prob[net] = 0.5

    def value(net: str) -> float:
        if isinstance(net, str) and net.isdigit():
            return float(int(net))
        return prob.get(net, 0.5)

    for instr in compiled.instructions:
        kind = instr[0]
        if kind == "assign_const":
            prob[instr[1]] = float(instr[2])
        elif kind == "assign_net":
            prob[instr[1]] = value(instr[2])
        elif kind == "gate":
            _, out_net, lut_or_fn, sym_names, input_map, _ = instr
            ps = [value(input_map.get(sym, "")) for sym in sym_names]
            table = _truth_table(lut_or_fn, len(sym_names))
            total = 0.0
            for idx, out in enumerate(table):
                if not out:
                    continue
                weight = 1.0
                for j, p in enumerate(ps):
                    weight *= p if (idx >> j) & 1 else 1.0 - p
                total += weight
            prob[out_net] = total
    return prob


def attach_retest_targets(
    data: Data,
    parsed: ParsedNetlistGraph,
    targets: ProblemTargets,
    signal_probability: Optional[Mapping[str, float]] = None,
) -> Data:
    """Node-level soft targets and masks on a schema-v2 graph.

    PI nodes carry care and value targets, gate nodes the path target, PO
    nodes the observation target. ``sp_target`` is the signal probability of
    each node's driven net (PO nodes take their net's value).
    """
    n = len(parsed.gates)
    care = torch.zeros(n)
    one = torch.full((n,), 0.5)
    path = torch.zeros(n)
    obs = torch.zeros(n)
    pi_mask = torch.zeros(n, dtype=torch.bool)
    po_mask = torch.zeros(n, dtype=torch.bool)
    gate_mask = torch.zeros(n, dtype=torch.bool)
    sp = torch.zeros(n)
    sp_mask = torch.zeros(n, dtype=torch.bool)
    driven: Dict[int, List[str]] = defaultdict(list)
    for net, nodes in parsed.net_drivers.items():
        for node in nodes:
            driven[node].append(net)
    for i, gate in enumerate(parsed.gates):
        if gate.cell == PI_CELL:
            pi_mask[i] = True
            care[i] = targets.care_freq.get(gate.inst, 0.0)
            one[i] = targets.one_freq.get(gate.inst, 0.5)
            net = gate.inst
        elif gate.cell == PO_CELL:
            po_mask[i] = True
            obs[i] = targets.obs_freq.get(gate.inst, 0.0)
            net = gate.inst
        else:
            gate_mask[i] = True
            path[i] = targets.path_freq.get(gate.inst, 0.0)
            nets = driven.get(i, [])
            net = nets[0] if nets else None
        if signal_probability is not None and net is not None and net in signal_probability:
            sp[i] = float(signal_probability[net])
            sp_mask[i] = True
    data.care_target = care
    data.one_target = one
    data.path_target = path
    data.obs_target = obs
    data.pi_node_mask = pi_mask
    data.po_target_mask = po_mask
    data.gate_target_mask = gate_mask
    data.sp_target = sp
    data.sp_mask = sp_mask
    data.num_patterns = torch.tensor([targets.patterns], dtype=torch.long)
    return data


def fact_text(targets: ProblemTargets, *, difficulty: Optional[str] = None, threshold: float = 0.5) -> str:
    """Port-only facts about one problem, for graph-text alignment.

    Inputs that are care bits in at least ``threshold`` of the patterns are
    listed with their majority value. No internal gate or net is named.
    """
    parts = [f"Fault {targets.fault}."]
    fixed = [
        f"{net}={1 if targets.one_freq.get(net, 0.5) >= 0.5 else 0}"
        for net in targets.inputs
        if targets.care_freq.get(net, 0.0) >= threshold
    ]
    free = len(targets.inputs) - len(fixed)
    parts.append(("Set " + ", ".join(fixed) + ".") if fixed else "No input is always needed.")
    parts.append(f"{free} of {len(targets.inputs)} inputs are free.")
    seen = [net for net in targets.outputs if targets.obs_freq.get(net, 0.0) >= threshold]
    if seen:
        parts.append("Observed at " + ", ".join(seen) + ".")
    if difficulty:
        parts.append(f"Difficulty {difficulty}.")
    return " ".join(parts)
