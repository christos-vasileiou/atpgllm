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

Full-scan sequential netlists (dataset v5) go through
:func:`parse_scan_to_graph`: every flop is cut into a state input node, named
by its QN net, and a state output node, named by its D net. The graph stays a
DAG. Which state output feeds which state input is kept beside the graph
(``state_outputs`` / ``state_inputs``), never as an edge.
:func:`parse_netlist` picks the right parse for a netlist.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

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
    flops / state_inputs / state_outputs / state_inverted:
        Scan netlists only, one entry per flop, in scan chain order when the
        chain table is given (netlist order otherwise). ``state_inputs[k]`` is
        the ``<PI>`` node of flop ``k``'s QN net and ``state_outputs[k]`` the
        ``<PO>`` node of its D net. ``state_inverted[k]`` is true when the
        value captured at D appears inverted at QN after the clock (every v5
        flop, since only QN is wired). This table is the only link from a
        state output back to a state input: the edge set never closes a loop.
    """

    gates: List[ParsedGate]
    net_drivers: Dict[str, List[int]]
    net_sinks: Dict[str, List[int]]
    inputs: List[str]
    outputs: List[str]
    wires: List[str]
    edges: List[Tuple[int, int, str, str, str]]
    flops: List[str] = field(default_factory=list)
    state_inputs: List[int] = field(default_factory=list)
    state_outputs: List[int] = field(default_factory=list)
    state_inverted: List[bool] = field(default_factory=list)

    @property
    def is_scan(self) -> bool:
        return bool(self.flops)


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


# ---------------------------------------------------------------------
# Full-scan sequential netlists
# ---------------------------------------------------------------------

_CHAIN_ROW_RE = re.compile(r"^\s*(\d+)\s+(\S+)\s+D=(\S+)\s+QN=(\S+)\s*$", re.M)


def chain_order(chain_table: str) -> List[str]:
    """Flop instances from test_si to test_so, as a v5 row's ``scan_chain``
    column lists them (``  0 d_reg  D=n11  QN=d``)."""
    rows = _CHAIN_ROW_RE.findall(chain_table or "")
    return [inst for _, inst, _, _ in sorted(rows, key=lambda row: int(row[0]))]


_MODULE_PORTS_RE = re.compile(r"\bmodule\s+\S+\s*\(([^)]*)\)\s*;")


def has_scan_ports(verilog_text: str) -> bool:
    """True when the module has a scan chain to load and unload its flops
    through (``test_si`` and ``test_so`` ports). Only then is one capture
    cycle a combinational test of the cut view."""
    header = _MODULE_PORTS_RE.search(verilog_text)
    ports = {p.strip() for p in header.group(1).split(",")} if header else set()
    return {"test_si", "test_so"} <= ports


def _scan_view():
    """``scan_view`` of the paired data_preprocessing checkout (branch seq-v5
    or later), which owns the cut-view rules the dataset and the simulator use."""
    from atpgllm.training._paths import ensure_data_preprocessing_on_path

    ensure_data_preprocessing_on_path()
    try:
        import scan_view
    except ImportError as exc:
        raise GraphRejected(
            "scan_view_missing",
            "the data_preprocessing checkout has no scan_view; point "
            "DATA_PREPROCESSING_ROOT at branch seq-v5 or later",
        ) from exc
    return scan_view


def parse_scan_to_graph(
    verilog_text: str,
    gate_func: Dict,
    chain_table: str = "",
) -> ParsedNetlistGraph:
    """Parse a full-scan netlist into the graph of one capture cycle.

    The dataset's cut view removes every flop, drives its QN net from a new
    input and observes its D net at a new output; scan ports and the clock
    leave the port list. That view is parsed as a combinational netlist and
    its synthetic buffers are folded away, so the graph speaks the names a
    scan test uses:

    - a ``<PI>`` node per primary input, then one per flop named by its QN net;
    - a ``<PO>`` node per primary output, then one per distinct D net. A D
      net that is also a primary output keeps one node, among the D nets, as
      in the row's ``expected_output``.

    ``<PI>`` nodes come first, in the order of the row's ``input_vector``.
    State bits follow the scan chain (test_si first) when ``chain_table`` is
    the row's ``scan_chain``; the netlist alone does not give that order for
    a flop stitched through a mux. Without the table they keep netlist order,
    which is the simulator's.

    Raises :class:`GraphRejected` ``scan_cut_failed`` for a circuit the cut
    view refuses (a clock that feeds logic, a non-scan sequential cell) and
    ``chain_table_mismatch`` when the table names other flops.
    """
    scan_view = _scan_view()
    try:
        cells, clocks = scan_view.scan_cells_from_netlist(verilog_text)
        cut = scan_view.cut_view(verilog_text, cells, clocks)
    except ValueError as exc:
        raise GraphRejected("scan_cut_failed", str(exc)) from exc
    order = {inst: i for i, inst in enumerate(chain_order(chain_table))}
    if order:
        if set(order) != {cell["instance"] for cell in cells}:
            raise GraphRejected("chain_table_mismatch")
        cells = sorted(cells, key=lambda cell: order[cell["instance"]])
    g = parse_verilog_to_graph(cut.netlist, gate_func)

    qn_nets = list(dict.fromkeys(cell["QN"] for cell in cells))
    d_nets = list(dict.fromkeys(cell["D"] for cell in cells))
    qn_set, d_set = set(qn_nets), set(d_nets)

    # Old nodes of the cut graph, by what they become.
    pseudo_in: Dict[int, str] = {}    # <PI> v5_ppi_k        -> QN net
    pseudo_out: Dict[int, str] = {}   # <PO> v5_ppo_k        -> D net
    buffer_in: Dict[int, str] = {}    # buffer v5_ppi_k -> QN -> QN net
    buffer_out: Dict[int, str] = {}   # buffer D -> v5_ppo_k  -> D net
    merged_out: set = set()           # primary <PO> of a net that is also a D net
    for i, gate in enumerate(g.gates):
        if gate.cell == "<PI>" and gate.inst in cut.ppi:
            pseudo_in[i] = cut.ppi[gate.inst]
        elif gate.cell == "<PO>" and gate.inst in cut.ppo:
            pseudo_out[i] = cut.ppo[gate.inst]
        elif gate.cell == "<PO>" and gate.inst in d_set:
            merged_out.add(i)
        elif not gate.cell.startswith("<"):
            reads = [net for pin, net in gate.connections.items() if net in cut.ppi]
            drives = [net for pin, net in gate.connections.items() if net in cut.ppo]
            if reads:
                buffer_in[i] = cut.ppi[reads[0]]
            elif drives:
                buffer_out[i] = cut.ppo[drives[0]]
    if len(pseudo_in) != len(cells) or len(buffer_in) != len(cells) \
            or len(pseudo_out) != len(cells) or len(buffer_out) != len(cells):
        raise GraphRejected("scan_cut_failed", "cut cells do not match the flops")

    nodes: List[ParsedGate] = []
    new_index: Dict[int, int] = {}
    for i, gate in enumerate(g.gates):
        if gate.cell == "<PI>" and i not in pseudo_in:
            new_index[i] = len(nodes)
            nodes.append(gate)
    state_in: Dict[str, int] = {}
    for net in qn_nets:
        state_in[net] = len(nodes)
        nodes.append(ParsedGate("<PI>", net, {PSEUDO_OUT_PIN: net}))
    dropped = set(pseudo_in) | set(pseudo_out) | set(buffer_in) | set(buffer_out) | merged_out
    for i, gate in enumerate(g.gates):
        if i in dropped or gate.cell in ("<PI>", "<PO>"):
            continue
        new_index[i] = len(nodes)
        nodes.append(gate)
    for i, gate in enumerate(g.gates):
        if gate.cell == "<PO>" and i not in dropped:
            new_index[i] = len(nodes)
            nodes.append(gate)
    state_out: Dict[str, int] = {}
    for net in d_nets:
        state_out[net] = len(nodes)
        nodes.append(ParsedGate("<PO>", net, {PSEUDO_IN_PIN: net}))

    edges: List[Tuple[int, int, str, str, str]] = []
    observed: set = set()
    for src, dst, src_pin, dst_pin, net in g.edges:
        if dst in buffer_in or src in buffer_out or dst in merged_out:
            continue                      # pseudo port to its buffer, or a merged output
        if src in buffer_in:
            new_src, src_pin = state_in[buffer_in[src]], PSEUDO_OUT_PIN
        else:
            new_src = new_index[src]
        if dst in buffer_out:
            if net in observed:           # flops sharing a D net share its node
                continue
            observed.add(net)
            new_dst, dst_pin = state_out[buffer_out[dst]], PSEUDO_IN_PIN
        else:
            new_dst = new_index[dst]
        edges.append((new_src, new_dst, src_pin, dst_pin, net))

    net_drivers: Dict[str, List[int]] = {}
    for net, (node,) in g.net_drivers.items():
        if cut.synthetic(net):
            continue
        net_drivers[net] = [state_in[buffer_in[node]] if node in buffer_in else new_index[node]]
    net_sinks: Dict[str, List[int]] = {}
    for _, dst, _, _, net in edges:
        net_sinks.setdefault(net, []).append(dst)

    return ParsedNetlistGraph(
        gates=nodes,
        net_drivers=net_drivers,
        net_sinks=net_sinks,
        inputs=[n for n in g.inputs if not cut.synthetic(n) and n not in qn_set] + qn_nets,
        outputs=[n for n in g.outputs if not cut.synthetic(n) and n not in d_set] + d_nets,
        wires=[w for w in g.wires if not cut.synthetic(w)],
        edges=edges,
        flops=[cell["instance"] for cell in cells],
        state_inputs=[state_in[cell["QN"]] for cell in cells],
        state_outputs=[state_out[cell["D"]] for cell in cells],
        state_inverted=[
            any("IQN" in expr for expr in gate_func.get(cell["cell"], {}).values())
            for cell in cells
        ],
    )


def parse_netlist(
    verilog_text: str,
    gate_func: Dict,
    chain_table: str = "",
) -> ParsedNetlistGraph:
    """Graph of a combinational netlist, or of one capture cycle of a
    full-scan netlist (:func:`parse_scan_to_graph`). ``chain_table`` is a v5
    row's ``scan_chain`` and is ignored for combinational netlists.

    A netlist with flops and no scan chain stays rejected as
    ``sequential_cell``: its state cannot be set or read by a test, so the
    cut would describe another circuit."""
    try:
        return parse_verilog_to_graph(verilog_text, gate_func)
    except GraphRejected as exc:
        if exc.reason != "sequential_cell" or not has_scan_ports(verilog_text):
            raise
    return parse_scan_to_graph(verilog_text, gate_func, chain_table)


def next_state(parsed: ParsedNetlistGraph, captured: Mapping[str, Any]) -> Dict[str, Any]:
    """State inputs of the next cycle from the outputs of this one.

    ``captured`` maps output names to 0, 1 or ``"x"`` (a row's
    ``expected_output``, or a simulated response). Returns ``{QN net: value}``
    for every flop, using the pairing kept beside the graph. This is the only
    place a state output reaches a state input, so a multi-cycle test is a
    chain of single-cycle graphs, never a cyclic one.
    """
    out: Dict[str, Any] = {}
    for d_node, qn_node, inverted in zip(parsed.state_outputs, parsed.state_inputs, parsed.state_inverted):
        value = captured.get(parsed.gates[d_node].inst, "x")
        text = str(value).strip().lower()
        if text in ("0", "1"):
            value = int(text) ^ int(inverted)
        else:
            value = "x"
        out[parsed.gates[qn_node].inst] = value
    return out


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
    - ``state_in_mask`` / ``state_out_mask`` : ``[N]`` bool — the ``<PI>`` /
      ``<PO>`` nodes that are state bits of a scan circuit (all false otherwise)
    - ``state_pair_index`` : ``[2, F]`` — per flop, its state output node
      (row 0, D side) and its state input node (row 1, QN side). A side table,
      not edges: batching offsets it like ``edge_index``, message passing
      never reads it. ``state_pair_inverted`` : ``[F]`` bool.
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

    state_in = set(parsed.state_inputs)
    state_out = set(parsed.state_outputs)
    state_nodes = state_in | state_out

    # ----- Attribute decomposition & structural features -----
    if vocab is not None:
        if num_nodes:
            gate_attrs = torch.tensor(
                [vocab.encode_node(ct, i in state_nodes) for i, ct in enumerate(cell_types)],
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
        state_in_mask=torch.tensor(
            [i in state_in for i in range(num_nodes)], dtype=torch.bool
        ),
        state_out_mask=torch.tensor(
            [i in state_out for i in range(num_nodes)], dtype=torch.bool
        ),
        state_pair_index=torch.tensor(
            [parsed.state_outputs, parsed.state_inputs], dtype=torch.int64
        ).reshape(2, -1),
        state_pair_inverted=torch.tensor(parsed.state_inverted, dtype=torch.bool),
        node_names=inst_names,
    )


def parse_verilog_to_pyg(
    verilog_text: str,
    gate_func: Dict,
    vocab: Optional[GateAttributeVocab] = None,
    chain_table: str = "",
) -> Data:
    """Convenience wrapper: directly return a PyG ``Data`` graph from
    Verilog text and gate function dict (:func:`parse_netlist`).

    When *vocab* is given, attribute decomposition and structural
    features are computed.
    """
    parsed = parse_netlist(verilog_text, gate_func, chain_table)
    return parsed_to_pyg(parsed, vocab=vocab)


netlist_to_pyg = parse_verilog_to_pyg


