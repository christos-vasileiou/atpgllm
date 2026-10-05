"""Fault-aware graph inputs and non-leaking ATPG supervision labels."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping

import torch
from torch_geometric.data import Data

from .netlist_parser import GraphRejected, ParsedNetlistGraph


FAULT_FEATURE_NAMES = (
    "is_fault_site",
    "drives_fault_net",
    "reads_fault_net",
    "stuck_at_zero",
    "stuck_at_one",
)
NUM_FAULT_FEATURES = len(FAULT_FEATURE_NAMES)

_FAULT_RE = re.compile(r"^\s*s(?:tuck[-_\s]*)?a(?:t[-_\s]*)?([01])\s+(.+?)\s*$", re.I)


@dataclass(frozen=True)
class FaultSpec:
    stuck_at: int
    net: str


def parse_fault(value: str) -> FaultSpec:
    """Parse the dataset's ``sa0 <net>`` / ``sa1 <net>`` fault contract."""
    match = _FAULT_RE.match(str(value or ""))
    if match is None:
        raise ValueError(
            f"Unsupported fault {value!r}; expected 'sa0 <net>' or 'sa1 <net>'."
        )
    return FaultSpec(stuck_at=int(match.group(1)), net=match.group(2).strip())


def _name_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {item.strip() for item in value.split(",") if item.strip()}
    if isinstance(value, Iterable):
        return {str(item).strip() for item in value if str(item).strip()}
    return set()


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(value)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
        if isinstance(parsed, Mapping):
            return parsed
    return {}


def _snapshot_states(value: Any) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    snapshot = _mapping(value)
    good = snapshot.get("Good Machine", snapshot.get("good", {}))
    bad = snapshot.get("Bad Machine", snapshot.get("bad", {}))
    return (
        good if isinstance(good, Mapping) else {},
        bad if isinstance(bad, Mapping) else {},
    )


def _instance_mask(
    parsed: ParsedNetlistGraph,
    names: set[str],
) -> torch.Tensor:
    return torch.tensor(
        [
            not gate.cell.startswith("<") and gate.inst in names
            for gate in parsed.gates
        ],
        dtype=torch.bool,
    )


def _discrepancy_mask(
    parsed: ParsedNetlistGraph,
    snapshot: Any,
) -> tuple[torch.Tensor, bool]:
    good, bad = _snapshot_states(snapshot)
    has_label = bool(good) and bool(bad)
    driven: Dict[int, list[str]] = {}
    for net, (node,) in parsed.net_drivers.items():
        driven.setdefault(node, []).append(net)
    values = []
    for node in range(len(parsed.gates)):
        output_nets = driven.get(node, [])
        values.append(
            has_label
            and any(
                net in good and net in bad and good[net] != bad[net]
                for net in output_nets
            )
        )
    return torch.tensor(values, dtype=torch.bool), has_label


def attach_atpg_context(
    data: Data,
    parsed: ParsedNetlistGraph,
    gate_funcs: Dict[str, Dict[str, str]],
    record: Mapping[str, Any],
) -> Data:
    """Attach target-fault inputs and auxiliary labels to a parsed graph.

    Only the netlist and target fault contribute to ``fault_feats`` and
    ``fault_edge`` (1.0 on every edge carrying the faulty net). Snapshot,
    propagation, and backtrack fields are attached as labels for encoder
    pretraining and are never consumed by the inference encoder.

    Raises :class:`GraphRejected` when the record's ``input_vector`` bits are
    not the netlist's primary inputs in order (the record describes another
    circuit), or when a fault is given but cannot be parsed or names a net
    absent from the graph.
    """
    num_nodes = len(parsed.gates)
    fault_feats = torch.zeros((num_nodes, NUM_FAULT_FEATURES), dtype=torch.float32)
    target_mask = torch.zeros(num_nodes, dtype=torch.bool)
    fault_edge = torch.zeros(len(parsed.edges), dtype=torch.float32)

    vector = _mapping(record.get("input_vector"))
    if vector and list(vector) != parsed.inputs:
        raise GraphRejected("vector_port_mismatch")

    fault = None
    fault_text = str(record.get("fault", "") or "")
    if fault_text.strip():
        try:
            fault = parse_fault(fault_text)
        except ValueError as exc:
            raise GraphRejected("fault_unparsable", fault_text) from exc
        if fault.net not in parsed.net_drivers:
            raise GraphRejected("fault_net_unknown", fault.net)

    if fault is not None:
        for i, edge in enumerate(parsed.edges):
            if edge[4] == fault.net:
                fault_edge[i] = 1.0
        drivers = set(parsed.net_drivers.get(fault.net, ()))
        sinks = set(parsed.net_sinks.get(fault.net, ()))
        sites = drivers | sinks
        for idx in sites:
            target_mask[idx] = True
            fault_feats[idx, 0] = 1.0
            fault_feats[idx, 3 + fault.stuck_at] = 1.0
        for idx in drivers:
            fault_feats[idx, 1] = 1.0
        for idx in sinks:
            fault_feats[idx, 2] = 1.0

    propagation_names = _name_set(record.get("fault_propagation_gates"))
    backtrack_names = _name_set(record.get("backtrack_gates"))
    propagation_mask = _instance_mask(parsed, propagation_names)
    backtrack_mask = _instance_mask(parsed, backtrack_names)
    discrepancy_mask, has_discrepancy = _discrepancy_mask(
        parsed, record.get("snapshot")
    )

    data.fault_feats = fault_feats
    data.fault_edge = fault_edge
    data.target_node_mask = target_mask
    data.propagation_mask = propagation_mask
    data.backtrack_mask = backtrack_mask
    data.discrepancy_mask = discrepancy_mask
    data.has_propagation_labels = torch.tensor(
        [bool(propagation_names)], dtype=torch.bool
    )
    data.has_backtrack_labels = torch.tensor(
        [bool(backtrack_names)], dtype=torch.bool
    )
    data.has_discrepancy_labels = torch.tensor(
        [has_discrepancy], dtype=torch.bool
    )
    return data
