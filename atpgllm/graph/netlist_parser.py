from __future__ import annotations

"""
Structural Verilog netlist → graph parser for ASAP7-style gate-level
netlists, built on the fault simulator's own parse.

The netlist is parsed by ``OptimizedNetlist`` (``fault_sim.py``), so the
graph uses exactly the simulator's view of the circuit: declared bus
indices, ``assign`` pass-throughs, constants, and multi-output cells.
Graph nodes are:

- one node per gate instance
- one ``<PI>`` node per primary-input bit and ``<PO>`` node per output bit,
  in the simulator's (and therefore the input/output vector's) bit order
- one ``<ASSIGN>`` node per ``assign`` statement (a fault on an alias net
  only affects that alias's readers, as in the simulator)
- ``<CONST0>`` / ``<CONST1>`` source nodes for tied inputs

Directed edges run from the node driving a net to each pin reading it and
carry the (driver output pin, sink input pin) roles. Netlists the schema
cannot represent faithfully raise :class:`GraphRejected` with a reason.
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math
import regex as re
import torch
from torch_geometric.data import Data

from .gate_features import (
    GateAttributeVocab,
    NUM_ATTRIBUTES,
    PSEUDO_IN_PIN,
    PSEUDO_OUT_PIN,
)


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
    "port_position",
]
NUM_STRUCTURAL_FEATURES = len(STRUCTURAL_FEATURE_NAMES)

_ALL_STRUCTURAL_FEATURES: Tuple[str, ...] = (
    "forward_depth",
    "backward_depth",
    "log_in_degree",
    "log_out_degree",
    "port_position",
)

_unknown_sf = set(STRUCTURAL_FEATURE_NAMES) - set(_ALL_STRUCTURAL_FEATURES)
if _unknown_sf:
    raise ValueError(
        f"STRUCTURAL_FEATURE_NAMES contains unknown name(s): {sorted(_unknown_sf)}. "
        f"Valid options: {list(_ALL_STRUCTURAL_FEATURES)}"
    )


class GraphRejected(ValueError):
    """A netlist or fault that the graph schema cannot represent faithfully."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


@dataclass
class ParsedGate:
    """One graph node: a gate instance or a ``<PI>``/``<PO>``/``<ASSIGN>``/
    ``<CONST*>`` pseudo-cell."""

    cell: str
    inst: str
    # port name -> net name
    connections: Dict[str, str]


@dataclass
class ParsedNetlistGraph:
    """
    Node/edge representation of a structural netlist.

    Attributes
    ----------
    gates:
        Graph nodes (gate instances and pseudo-cells) with cell type,
        instance name and port→net connections.
    net_drivers:
        Mapping: net name → [index of the single node driving the net].
    net_sinks:
        Mapping: net name → indices of nodes reading the net (one entry
        per reading pin).
    inputs / outputs / wires:
        Simulator-ordered primary input / output / wire bits.
    edges:
        ``(driver, sink, driver_pin, sink_pin, net)`` per reading pin.
    """

    gates: List[ParsedGate]
    net_drivers: Dict[str, List[int]]
    net_sinks: Dict[str, List[int]]
    inputs: List[str]
    outputs: List[str]
    wires: List[str]
    edges: List[Tuple[int, int, str, str, str]]


# Same instance pattern as ``OptimizedNetlist._parse_and_compile``; used to
# find instances the simulator silently skips.
GATE_INGREDIENT = r"((?:\\[^\s]+|\w+))\s+((?:\\[^\s]+|\w+))\s*\(\s*([\s\S]+?)\s*\)\s*;"
_GATE_INGREDIENT_RE = re.compile(GATE_INGREDIENT)
_ASSIGN_RE = re.compile(r"\bassign\b")
_STATE_VAR_RE = re.compile(r"\bIQN?\b")
_FUNC_VAR_RE = re.compile(r"\b([A-Z][A-Z0-9]*)\b")
_NON_INPUT_VARS = frozenset({"IQ", "IQN", "VDD", "VSS"})
_CONST_NETS = {"0": "<CONST0>", "1": "<CONST1>"}
_LOGIC_VALUE_RE = re.compile(r"^\\?\*Logic(?P<val>[01])\*$")


def _net(name: str) -> str:
    """Normalise ``*Logic0*``/``\\*Logic1*`` constants to ``"0"``/``"1"``, as
    the simulator does when evaluating."""
    name = name.strip()
    match = _LOGIC_VALUE_RE.match(name)
    return match.group("val") if match else name


def _simulator_netlist(verilog_text: str, gate_func: Dict):
    from atpgllm.training.reward_function_factory import RewardFunctionFactory
    from fault_sim import OptimizedNetlist  # on sys.path via the import above

    try:
        return OptimizedNetlist(
            verilog_text,
            gate_func,
            RewardFunctionFactory.DECL_RE,
            RewardFunctionFactory.NAME_RE,
        )
    except ValueError as exc:
        raise GraphRejected("unsupported_declaration", str(exc)) from exc


def parse_verilog_to_graph(
    verilog_text: str,
    gate_func: Dict,
) -> ParsedNetlistGraph:
    """
    Parse a structural combinational netlist into graph nodes and edges.

    Args
    ----
    verilog_text:
        Full Verilog netlist text for a single module.
    gate_func:
        ``{cell_name: {output_pin: boolean_expression}}`` as in
        ``sim_config.json["gate_funcs"]``.

    Raises
    ------
    GraphRejected
        For constructs the simulator skips or the schema cannot represent:
        unknown or sequential cells, unparsed ``assign`` statements,
        unconnected function pins, multiply-driven or undriven nets.
    """
    instances = [
        (cell, inst)
        for cell, inst, _ in _GATE_INGREDIENT_RE.findall(verilog_text)
        if not cell.startswith("module")
    ]
    for cell, inst in instances:
        if cell not in gate_func:
            raise GraphRejected("unknown_cell", f"{cell} {inst}")
        if any(_STATE_VAR_RE.search(expr) for expr in gate_func[cell].values()):
            raise GraphRejected("sequential_cell", f"{cell} {inst}")
    if len({inst for _, inst in instances}) != len(instances):
        raise GraphRejected("duplicate_instance")

    sim = _simulator_netlist(verilog_text, gate_func)

    nodes: List[ParsedGate] = []
    driver: Dict[str, Tuple[int, str]] = {}
    reads: List[Tuple[str, int, str]] = []

    def add_node(cell: str, inst: str, connections: Dict[str, str]) -> int:
        nodes.append(ParsedGate(cell=cell, inst=inst, connections=connections))
        return len(nodes) - 1

    def add_driver(net: str, node: int, pin: str) -> None:
        if net in driver:
            raise GraphRejected("multi_driven_net", net)
        driver[net] = (node, pin)

    for net in sim.input_nets:
        add_driver(net, add_node("<PI>", net, {PSEUDO_OUT_PIN: net}), PSEUDO_OUT_PIN)

    gate_index: Dict[str, int] = {}
    for cell, inst, out_pin, out_net, input_map in sim.gate_metadata:
        out_net = _net(out_net)
        input_map = {pin: _net(net) for pin, net in input_map.items()}
        if inst not in gate_index:
            gate_index[inst] = add_node(cell, inst, dict(input_map))
            for pin, net in input_map.items():
                reads.append((net, gate_index[inst], pin))
        node = gate_index[inst]
        nodes[node].connections[out_pin] = out_net
        add_driver(out_net, node, out_pin)
        used = {
            m.group(1)
            for m in _FUNC_VAR_RE.finditer(gate_func[cell][out_pin])
        } - _NON_INPUT_VARS
        missing = used - set(input_map)
        if missing:
            raise GraphRejected("unconnected_pin", f"{inst} {sorted(missing)}")
    if len(gate_index) != len(instances):
        raise GraphRejected("unparsed_gate")

    assigns = 0
    for instr in sim.instructions:
        if instr[0] not in ("assign_net", "assign_const"):
            continue
        net, src = instr[1], str(instr[2])
        if instr[0] == "assign_const" and net in driver:
            continue  # TIEHI/TIELO output, already a gate node
        assigns += 1
        node = add_node(
            "<ASSIGN>", net, {PSEUDO_IN_PIN: src, PSEUDO_OUT_PIN: net}
        )
        add_driver(net, node, PSEUDO_OUT_PIN)
        reads.append((src, node, PSEUDO_IN_PIN))
    if assigns != len(_ASSIGN_RE.findall(verilog_text)):
        raise GraphRejected("unsupported_assign")

    for net in sim.output_nets:
        reads.append((net, add_node("<PO>", net, {PSEUDO_IN_PIN: net}), PSEUDO_IN_PIN))

    edges: List[Tuple[int, int, str, str, str]] = []
    net_sinks: Dict[str, List[int]] = {}
    for net, sink, sink_pin in reads:
        if net not in driver and net in _CONST_NETS:
            cell = _CONST_NETS[net]
            add_driver(net, add_node(cell, cell, {PSEUDO_OUT_PIN: net}), PSEUDO_OUT_PIN)
        if net not in driver:
            raise GraphRejected("undriven_net", net)
        src, src_pin = driver[net]
        edges.append((src, sink, src_pin, sink_pin, net))
        net_sinks.setdefault(net, []).append(sink)

    from netlist_utils import declared_nets
    from atpgllm.training.reward_function_factory import RewardFunctionFactory

    return ParsedNetlistGraph(
        gates=nodes,
        net_drivers={net: [node] for net, (node, _) in driver.items()},
        net_sinks=net_sinks,
        inputs=list(sim.input_nets),
        outputs=list(sim.output_nets),
        wires=declared_nets(
            verilog_text,
            "wire",
            RewardFunctionFactory.DECL_RE,
            RewardFunctionFactory.NAME_RE,
        ),
        edges=edges,
    )


def compute_structural_features(
    edge_index: torch.Tensor,
    num_nodes: int,
    port_position: Optional[List[float]] = None,
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
        - **port_position**: a ``<PI>``/``<PO>`` node's bit index in the
           input/output vector order, normalised to 0–1; 0 elsewhere.

    Raises :class:`GraphRejected` (``cycle``) when the graph is not a DAG.
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

    visited = 0
    while queue:
        v = queue.popleft()
        visited += 1
        for w in successors[v]:
            fwd_depth[w] = max(fwd_depth[w], fwd_depth[v] + 1)
            remaining_in[w] -= 1
            if remaining_in[w] == 0:
                queue.append(w)
    if visited != num_nodes:
        raise GraphRejected("cycle")

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
        "port_position":  port_position or [0.0] * num_nodes,
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
    - ``edge_pins`` : ``[E, 2]`` int64 — (driver output pin, sink input pin) roles

    Always included:

    - ``edge_index`` : ``[2, E]`` — one driver → sink edge per reading pin
    - ``structural_feats`` : ``[N, NUM_STRUCTURAL_FEATURES]`` float32 — depth / degree / port features
    - ``gate_node_mask`` / ``po_node_mask`` / ``pi_node_mask`` : ``[N]`` bool — node kinds
    - ``node_names`` : instance names, or net names for ``<PI>``/``<PO>`` nodes

    These are consumed by :class:`AttributeDecompositionEncoder` and
    :class:`DAGGINEncoder`.

    A dummy ``x`` of shape ``[N, 1]`` is still attached for backward
    compatibility with code that expects it.
    """
    num_nodes = len(parsed.gates)

    edge_index = torch.tensor(
        [[e[0] for e in parsed.edges], [e[1] for e in parsed.edges]],
        dtype=torch.int64,
    ).reshape(2, -1)

    pi_pos = {net: i for i, net in enumerate(parsed.inputs)}
    po_pos = {net: i for i, net in enumerate(parsed.outputs)}
    port_position = [
        pi_pos[g_.inst] / max(1, len(pi_pos) - 1) if g_.cell == "<PI>"
        else po_pos[g_.inst] / max(1, len(po_pos) - 1) if g_.cell == "<PO>"
        else 0.0
        for g_ in parsed.gates
    ]
    structural_feats = compute_structural_features(
        edge_index, num_nodes, port_position
    )

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
        edge_pins = torch.tensor(
            [[vocab.encode_pin(e[2]), vocab.encode_pin(e[3])] for e in parsed.edges],
            dtype=torch.int64,
        ).reshape(-1, 2)
    else:
        gate_attrs = torch.zeros((num_nodes, NUM_ATTRIBUTES), dtype=torch.int64)
        edge_pins = torch.zeros((len(parsed.edges), 2), dtype=torch.int64)

    return Data(
        x=x,
        edge_index=edge_index,
        cell_type=cell_type,
        inst_id=inst_id,
        gate_attrs=gate_attrs,
        structural_feats=structural_feats,
        edge_pins=edge_pins,
        gate_node_mask=torch.tensor(
            [not ct.startswith("<") for ct in cell_types], dtype=torch.bool
        ),
        po_node_mask=torch.tensor(
            [ct == "<PO>" for ct in cell_types], dtype=torch.bool
        ),
        pi_node_mask=torch.tensor(
            [ct == "<PI>" for ct in cell_types], dtype=torch.bool
        ),
        node_names=inst_names,
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


