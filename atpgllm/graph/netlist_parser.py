from __future__ import annotations

"""
Lightweight Verilog netlist → graph parser for ASAP7-style gate-level
netlists using DGL as the graph backend.

This module reuses the same regex patterns as ``OptimizedNetlist`` in
``fault_sim.py`` but only extracts a graph structure:

- one node per gate instance
- directed edges from driver gates to sink gates via nets

The parser expects a ``gate_func`` dictionary (as produced by
``data_preprocessing/gate_funcs_extraction.py``), so that it can
distinguish output pins from input pins for each cell.

The returned object can be converted into a graph suitable for
PyTorch Geometric with:

- one node per gate instance
- directed edges from driver gates to sink gates via nets

For multi-GPU training, graphs can be batched via a PyG
``DataLoader`` and trained with standard DDP/FSDP setups.
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math
import regex as re
import torch
from torch_geometric.data import Data

from .gate_features import GateAttributeVocab, NUM_ATTRIBUTES


# ---------------------------------------------------------------------
# SINGLE SOURCE OF TRUTH for continuous structural features.
#
# To disable a structural feature for debugging / ablation, comment
# out its name below.  ``compute_structural_features`` and the PyG
# ``Data.structural_feats`` tensor will automatically drop that column.
# ---------------------------------------------------------------------
STRUCTURAL_FEATURE_NAMES: List[str] = [
    "forward_depth",
    "backward_depth",
    "log_in_degree",
    "log_out_degree",
]
NUM_STRUCTURAL_FEATURES = len(STRUCTURAL_FEATURE_NAMES)

_ALL_STRUCTURAL_FEATURES: Tuple[str, ...] = (
    "forward_depth",
    "backward_depth",
    "log_in_degree",
    "log_out_degree",
)

_unknown_sf = set(STRUCTURAL_FEATURE_NAMES) - set(_ALL_STRUCTURAL_FEATURES)
if _unknown_sf:
    raise ValueError(
        f"STRUCTURAL_FEATURE_NAMES contains unknown name(s): {sorted(_unknown_sf)}. "
        f"Valid options: {list(_ALL_STRUCTURAL_FEATURES)}"
    )


@dataclass
class ParsedGate:
    cell: str
    inst: str
    # port name -> net name
    connections: Dict[str, str]


@dataclass
class ParsedNetlistGraph:
    """
    Lightweight representation of a structural netlist.

    Attributes
    ----------
    gates:
        List of gate instances with cell type, instance name and
        port→net connections.
    net_drivers:
        Mapping: net name → list of node indices that drive the net.
    net_sinks:
        Mapping: net name → list of node indices that read the net.
    inputs / outputs / wires:
        Expanded lists of primary input / output / internal wire nets.
        Bus and array declarations are expanded to scalar net names.
    """

    gates: List[ParsedGate]
    net_drivers: Dict[str, List[int]]
    net_sinks: Dict[str, List[int]]
    inputs: List[str]
    outputs: List[str]
    wires: List[str]


# Regex patterns adapted from ``OptimizedNetlist._parse_and_compile``
GATE_INGREDIENT = r"((?:\\[^\s]+|\w+))\s+((?:\\[^\s]+|\w+))\s*\(\s*([\s\S]+?)\s*\)\s*;"
GATE_CONNECTIONS = r"\.((?:\\[^\s]+|\w+))\s*\(\s*([^)]+?)\s*\)"
LOGIC_VALUE = r"\*Logic(?P<val>[01]+)\*\s*"

_GATE_INGREDIENT_RE = re.compile(GATE_INGREDIENT)
_GATE_CONNECTIONS_RE = re.compile(GATE_CONNECTIONS)
_LOGIC_VALUE_RE = re.compile(LOGIC_VALUE)
_COMMON_OUTPUT_PINS = frozenset({
    "Y", "Z", "ZN", "Q", "QN", "O", "OUT", "CO", "CON", "S", "SN",
})

# Declaration regex adapted to support ASAP7-style synthesised netlists.
# Matches lines like:
#   input a, b;
#   output [3:0] sum;
DECL_RE = re.compile(
    r"(?P<kind>input|output|inout|wire|reg|tri)\s*"
    r"(?P<packed>\[[^]]+\])?\s*"
    r"(?P<rest>[^;]+);",
    re.IGNORECASE,
)

NAME_RE = re.compile(
    r"(?P<name>[A-Za-z_\\][\w\\]*)"
    r"(?P<unpacked>\[[^]]+\])?\s*$"
)


def _parse_range(rng: str | None) -> int:
    """Convert [msb:lsb] into integer width. If None, return 1."""
    if not rng:
        return 1
    nums = re.findall(r"\d+", rng)
    if len(nums) != 2:
        return 1
    msb, lsb = map(int, nums)
    return abs(msb - lsb) + 1


def _expand_nets_from_decl(verilog_text: str, keyword: str) -> List[str]:
    """
    Expand net declarations for a given keyword (input/output/wire…).

    This mirrors the behaviour of ``netlist_utils.expand_nets`` but is
    self-contained to avoid tight coupling. It is robust to both scalar
    and bus/array style declarations.
    """
    keyword = keyword.lower()
    assert keyword in {"input", "output", "inout", "wire", "reg", "tri"}

    expanded: List[str] = []
    for m in DECL_RE.finditer(verilog_text):
        kind = m.group("kind").lower()
        if kind != keyword:
            continue
        packed = m.group("packed")
        rest = m.group("rest")
        bus_width = _parse_range(packed)

        for token in rest.split(","):
            token = token.strip()
            if not token:
                continue
            nm = NAME_RE.match(token)
            if not nm:
                continue
            base = nm.group("name")
            unpacked = nm.group("unpacked")
            array_len = _parse_range(unpacked)

            # Scalar net
            if bus_width == 1 and array_len == 1 and not packed and not unpacked:
                expanded.append(base)
                continue

            # Expand bus and/or array dimensions
            for i in range(bus_width):
                for j in range(max(1, array_len)):
                    suffix = ""
                    if packed:
                        suffix += f"[{i}]"
                    if unpacked:
                        suffix += f"[{j}]"
                    expanded.append(f"{base}{suffix}")

    return expanded


def _parse_gates(verilog_text: str, gate_func: Dict) -> List[ParsedGate]:
    """Parse gate instances from the Verilog netlist text."""
    gates: List[ParsedGate] = []

    for match in _GATE_INGREDIENT_RE.finditer(verilog_text):
        gate_type, instance, connections_str = match.groups()

        # Skip module headers. Unknown cells are retained and encoded through
        # the vocabulary's explicit UNKNOWN values; their output pins use the
        # conservative common-name fallback in _build_driver_sink_maps.
        if gate_type.startswith("module"):
            continue

        raw_conns = dict(_GATE_CONNECTIONS_RE.findall(connections_str))

        # Clean any encoded logic values in connections (e.g. *Logic1*)
        connections: Dict[str, str] = {}
        for port, net in raw_conns.items():
            m = _LOGIC_VALUE_RE.search(net)
            if m:
                connections[port] = m.group("val")
            else:
                connections[port] = net.strip()

        gates.append(ParsedGate(cell=gate_type, inst=instance, connections=connections))

    return gates


def _build_driver_sink_maps(
    gates: List[ParsedGate],
    gate_func: Dict,
) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    """
    Build driver / sink maps per net using ``gate_func`` to
    identify output vs input ports for each cell.
    """
    net_drivers: Dict[str, List[int]] = {}
    net_sinks: Dict[str, List[int]] = {}

    for node_idx, gate in enumerate(gates):
        cell_type = gate.cell
        cell_outputs = set(gate_func.get(cell_type, {}).keys())
        instance_ports = set(gate.connections.keys())
        if not cell_outputs:
            cell_outputs = {
                port
                for port in instance_ports
                if port.upper() in _COMMON_OUTPUT_PINS
            }

        out_ports = instance_ports & cell_outputs
        in_ports = instance_ports - cell_outputs

        # Drivers: each output port drives its connected net
        for p in out_ports:
            out_net = gate.connections[p]
            if not out_net:
                continue
            net_drivers.setdefault(out_net, []).append(node_idx)

        # Sinks: each input port reads its connected net
        for p in in_ports:
            in_net = gate.connections[p]
            if not in_net:
                continue
            net_sinks.setdefault(in_net, []).append(node_idx)

    return net_drivers, net_sinks


def parse_verilog_to_graph(
    verilog_text: str,
    gate_func: Dict,
) -> ParsedNetlistGraph:
    """
    Parse a structural Verilog netlist into a driver/sink representation.

    Args
    ----
    verilog_text:
        Full Verilog netlist text for a single module.
    gate_func:
        Normalised gate function dict:
        ``{cell_name: {output_pin: {...}}}`` as produced by
        ``gate_funcs_extraction.py``.
    """
    gates = _parse_gates(verilog_text, gate_func)
    net_drivers, net_sinks = _build_driver_sink_maps(gates, gate_func)

    inputs = _expand_nets_from_decl(verilog_text, "input")
    outputs = _expand_nets_from_decl(verilog_text, "output")
    wires = _expand_nets_from_decl(verilog_text, "wire")

    return ParsedNetlistGraph(
        gates=gates,
        net_drivers=net_drivers,
        net_sinks=net_sinks,
        inputs=inputs,
        outputs=outputs,
        wires=wires,
    )


def compute_structural_features(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Compute DAG-native structural features for each node.

    Replaces RWPE (which is degenerate on DAGs — random walks never
    return) with features that directly encode SCOAP-like controllability
    and observability.

    Returns
    -------
    Tensor[num_nodes, NUM_STRUCTURAL_FEATURES] with the columns listed
    in ``STRUCTURAL_FEATURE_NAMES`` (in declared order):
        - **forward_depth** (normalised 0–1): longest path from any
           source node.  Approximates combinational *controllability
           difficulty* — deeper gates are harder to control.
        - **backward_depth** (normalised 0–1): longest path to any
           sink node.  Approximates *observability difficulty* — gates
           farther from outputs are harder to observe.
        - **log_in_degree**: ``log₂(1 + in_degree)``.  Fanin size
           directly affects sensitisation difficulty.
        - **log_out_degree**: ``log₂(1 + out_degree)``.  High-fanout
           nodes create reconvergent structures that complicate ATPG.
    """
    if num_nodes == 0:
        return torch.zeros((0, NUM_STRUCTURAL_FEATURES), dtype=torch.float32)

    # Build adjacency lists from edge_index
    successors: List[List[int]] = [[] for _ in range(num_nodes)]
    predecessors: List[List[int]] = [[] for _ in range(num_nodes)]
    in_deg = [0] * num_nodes
    out_deg = [0] * num_nodes

    E = edge_index.size(1)
    ei_cpu = edge_index.cpu()
    for e in range(E):
        src = ei_cpu[0, e].item()
        dst = ei_cpu[1, e].item()
        successors[src].append(dst)
        predecessors[dst].append(src)
        in_deg[dst] += 1
        out_deg[src] += 1

    # ----- Forward depth (Kahn's algorithm + longest-path relaxation) -----
    fwd_depth = [0] * num_nodes
    remaining_in = list(in_deg)
    queue: deque[int] = deque()
    for v in range(num_nodes):
        if remaining_in[v] == 0:
            queue.append(v)

    while queue:
        v = queue.popleft()
        for w in successors[v]:
            fwd_depth[w] = max(fwd_depth[w], fwd_depth[v] + 1)
            remaining_in[w] -= 1
            if remaining_in[w] == 0:
                queue.append(w)

    # ----- Backward depth (reverse Kahn's) -----
    bwd_depth = [0] * num_nodes
    remaining_out = list(out_deg)
    queue.clear()
    for v in range(num_nodes):
        if remaining_out[v] == 0:
            queue.append(v)

    while queue:
        v = queue.popleft()
        for u in predecessors[v]:
            bwd_depth[u] = max(bwd_depth[u], bwd_depth[v] + 1)
            remaining_out[u] -= 1
            if remaining_out[u] == 0:
                queue.append(u)

    # ----- Normalise depths per graph -----
    max_fwd = max(fwd_depth) if fwd_depth else 1
    max_bwd = max(bwd_depth) if bwd_depth else 1
    norm_fwd = [d / max_fwd if max_fwd > 0 else 0.0 for d in fwd_depth]
    norm_bwd = [d / max_bwd if max_bwd > 0 else 0.0 for d in bwd_depth]

    # ----- Log-scaled degrees -----
    log_in = [math.log2(1.0 + d) for d in in_deg]
    log_out = [math.log2(1.0 + d) for d in out_deg]

    # Compute all columns first, then select only the enabled ones.
    _all_cols: Dict[str, List[float]] = {
        "forward_depth":  norm_fwd,
        "backward_depth": norm_bwd,
        "log_in_degree":  log_in,
        "log_out_degree": log_out,
    }
    selected = [_all_cols[name] for name in STRUCTURAL_FEATURE_NAMES]
    if not selected:
        return torch.zeros((num_nodes, 0), dtype=torch.float32)
    feats = torch.tensor(list(zip(*selected)), dtype=torch.float32)
    return feats


def parsed_to_pyg(
    parsed: ParsedNetlistGraph,
    vocab: Optional[GateAttributeVocab] = None,
) -> Data:
    """Convert a :class:`ParsedNetlistGraph` into a PyG ``Data`` object.

    When *vocab* is provided, the returned ``Data`` includes:

    - ``gate_attrs`` : ``[N, NUM_ATTRIBUTES]`` int64 — decomposed attribute indices
    - ``structural_feats`` : ``[N, NUM_STRUCTURAL_FEATURES]`` float32 — depth / degree features

    These are consumed by :class:`AttributeDecompositionEncoder` and
    :class:`DAGGINEncoder`.

    A dummy ``x`` of shape ``[N, 1]`` is still attached for backward
    compatibility with code that expects it.
    """
    num_nodes = len(parsed.gates)

    # Build edges: for each net, connect all drivers to all sinks
    edge_src: List[int] = []
    edge_dst: List[int] = []

    for net, drivers in parsed.net_drivers.items():
        sinks = parsed.net_sinks.get(net, [])
        if not drivers or not sinks:
            continue
        for d in drivers:
            for s in sinks:
                edge_src.append(d)
                edge_dst.append(s)

    if edge_src:
        src = torch.tensor(edge_src, dtype=torch.int64)
        dst = torch.tensor(edge_dst, dtype=torch.int64)
        edge_index = torch.stack([src, dst], dim=0)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.int64)

    # Dummy ``x`` for backward compatibility
    x = torch.zeros((num_nodes, 1), dtype=torch.float32)

    # Cell types and instance names (local integer IDs, kept for debug)
    cell_types = [g_.cell for g_ in parsed.gates]
    inst_names = [g_.inst for g_ in parsed.gates]

    cell2id: Dict[str, int] = {}
    cell_ids: List[int] = []
    for ct in cell_types:
        if ct not in cell2id:
            cell2id[ct] = len(cell2id)
        cell_ids.append(cell2id[ct])

    inst2id: Dict[str, int] = {}
    inst_ids: List[int] = []
    for nm in inst_names:
        if nm not in inst2id:
            inst2id[nm] = len(inst2id)
        inst_ids.append(inst2id[nm])

    cell_type = torch.tensor(cell_ids, dtype=torch.int64)
    inst_id = torch.tensor(inst_ids, dtype=torch.int64)

    # ----- Attribute decomposition & structural features -----
    if vocab is not None:
        if num_nodes:
            gate_attrs = torch.tensor(
                [vocab.encode(ct) for ct in cell_types],
                dtype=torch.int64,
            )
        else:
            gate_attrs = torch.empty(
                (0, vocab.num_attributes), dtype=torch.int64
            )
        structural_feats = compute_structural_features(edge_index, num_nodes)
    else:
        gate_attrs = torch.zeros((num_nodes, NUM_ATTRIBUTES), dtype=torch.int64)
        structural_feats = torch.zeros(
            (num_nodes, NUM_STRUCTURAL_FEATURES), dtype=torch.float32
        )

    return Data(
        x=x,
        edge_index=edge_index,
        cell_type=cell_type,
        inst_id=inst_id,
        gate_attrs=gate_attrs,
        structural_feats=structural_feats,
    )


def parse_verilog_to_pyg(
    verilog_text: str,
    gate_func: Dict,
    vocab: Optional[GateAttributeVocab] = None,
) -> Data:
    """Convenience wrapper: directly return a PyG ``Data`` graph from
    Verilog text and gate function dict.

    When *vocab* is given, attribute decomposition and structural
    features are computed.
    """
    parsed = parse_verilog_to_graph(verilog_text, gate_func)
    return parsed_to_pyg(parsed, vocab=vocab)


netlist_to_pyg = parse_verilog_to_pyg


