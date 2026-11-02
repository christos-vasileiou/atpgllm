"""Continuous propagation and controlling signals read from one Good/Bad net table.

A simulation shows, for every net, its good and its bad (faulty) value. A *D-net*
is a net whose two definite values differ. The fault effect starts at the fault
site and travels along D-nets toward the primary outputs, and a gate passes it on
only when its other (side) inputs hold values that do not block it.

``propagation``: share of the PO-reaching fanout cone of the fault that carries the
fault effect. ``controlling``: share of the side inputs of the gates the effect
has reached that hold a value compatible with passing it. Both are 0 when the
fault is not activated and 1 when the vector detects it, and both are smoothed by
:func:`sensitize`.

Neither needs TetraMAX to flag anything. The values come from the table and the
structure from the compiled netlist (``OptimizedNetlist.instructions``).
"""
import math
from collections import deque
from itertools import product
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

# Gates with more side inputs than this are skipped (cells have far fewer).
MAX_SIDE_INPUTS = 10


def _bit(value: Any) -> Optional[int]:
    """0 or 1 for a definite value, None for x, NaN or anything unreadable."""
    if isinstance(value, str):
        value = value.strip()
        return int(value) if value in ("0", "1") else None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number in (0.0, 1.0) else None


def sensitize(ratio: float) -> float:
    """``(1 + ratio)^3`` shifted to zero and scaled to [0, 1].

    A convex map: the last stretch toward a full path is worth more than the
    first. Zero stays zero so an attempt with no signal scores like no attempt.
    """
    ratio = min(1.0, max(0.0, ratio))
    return ((1.0 + ratio) ** 3 - 1.0) / 7.0


def _is_constant(net: str) -> bool:
    return isinstance(net, str) and net.isdigit()


class SignalGraph:
    """Fanout structure of a compiled netlist, built once per netlist.

    ``records[i]`` is ``(out_net, lut, syms, input_map, input_nets)``; aliases
    (``assign_net``) have no LUT, no symbols and a single input.
    """

    def __init__(self, netlist: Any, output_nets: Optional[List[str]] = None):
        self.records: List[Tuple[str, Any, Tuple[str, ...], Optional[Dict[str, str]], List[str]]] = []
        self.consumers: Dict[str, List[int]] = {}
        producer: Dict[str, int] = {}
        for instruction in netlist.instructions:
            kind = instruction[0]
            if kind == "gate":
                _, out_net, lut, syms, input_map, _deps = instruction
                inputs = [input_map[s] for s in syms if s in input_map]
                record = (out_net, lut, tuple(syms), dict(input_map), inputs)
            elif kind == "assign_net":
                _, out_net, source = instruction
                inputs = [source]
                record = (out_net, None, (), None, inputs)
            else:
                continue
            index = len(self.records)
            self.records.append(record)
            producer[out_net] = index
            for net in set(inputs):
                if not _is_constant(net):
                    self.consumers.setdefault(net, []).append(index)

        # Gate levels to the nearest PO; the keys are the nets that reach a PO.
        self.dist: Dict[str, int] = {}
        queue = deque()
        for net in output_nets if output_nets is not None else getattr(netlist, "output_nets", None) or []:
            self.dist[net] = 0
            queue.append(net)
        while queue:
            net = queue.popleft()
            index = producer.get(net)
            if index is None:
                continue
            for source in self.records[index][4]:
                if source not in self.dist and not _is_constant(source):
                    self.dist[source] = self.dist[net] + 1
                    queue.append(source)
        self._cones: Dict[str, FrozenSet[str]] = {}

    def cone(self, fault_net: str) -> FrozenSet[str]:
        """The fault net and every net it feeds that can still reach a PO."""
        cone = self._cones.get(fault_net)
        if cone is None:
            seen: Set[str] = {fault_net}
            queue = deque([fault_net])
            while queue:
                net = queue.popleft()
                for index in self.consumers.get(net, ()):
                    out_net = self.records[index][0]
                    if out_net not in seen and out_net in self.dist:
                        seen.add(out_net)
                        queue.append(out_net)
            cone = self._cones[fault_net] = frozenset(seen)
        return cone


def signal_graph(netlist: Any, gate_funcs: Any = None) -> Optional[SignalGraph]:
    """The cached :class:`SignalGraph` of a netlist, None when its structure is unknown.

    The fast simulator's ``OptimizedNetlist`` already holds the compiled gates. The
    TetraMAX backend's ``SimulationNetlist`` keeps ports only, so its Verilog text is
    compiled once with the gate library into an ``OptimizedNetlist`` that serves as
    the structure (net names are the same, the values still come from TetraMAX).
    """
    graph = getattr(netlist, "_signal_graph", None)
    if graph is None:
        structure = netlist
        if not getattr(netlist, "instructions", None):
            text = getattr(netlist, "netlist", None)
            if not isinstance(text, str) or not gate_funcs:
                return None
            from fault_sim import OptimizedNetlist
            from atpgllm.training.reward_function_factory import RewardFunctionFactory
            try:
                structure = OptimizedNetlist(text, gate_funcs, RewardFunctionFactory.DECL_RE,
                                             RewardFunctionFactory.NAME_RE)
            except Exception:
                return None
            if not structure.instructions:
                return None
        graph = SignalGraph(structure, getattr(netlist, "output_nets", None))
        try:
            netlist._signal_graph = graph
        except AttributeError:
            pass
    return graph


def _evaluate(lut: Any, args: List[int]) -> int:
    if isinstance(lut, list):
        index = 0
        for position, value in enumerate(args):
            index |= value << position
        return lut[index]
    return int(bool(lut(*args)))


def _side_score(record, good: Dict[str, Any], bad: Dict[str, Any],
                d_nets: Set[str]) -> Tuple[int, int]:
    """(side inputs, correctly assigned side inputs) of one gate the effect reached.

    A side input is correct when its value is 0 or 1 and some completion of the
    gate's other side inputs lets the fault effect through. That is the exact
    cell-level condition, so complex cells (AOI, OAI) need no controlling-value
    table, and a gate that already passes the effect has every side input correct.
    """
    _, lut, syms, input_map, _ = record
    if lut is None or input_map is None:
        return 0, 0
    d_values: Dict[int, Tuple[int, int]] = {}
    side_values: Dict[int, Optional[int]] = {}
    for position, sym in enumerate(syms):
        net = input_map.get(sym)
        if net is None:
            return 0, 0
        if _is_constant(net):
            side_values[position] = int(net)
        elif net in d_nets:
            d_values[position] = (_bit(good.get(net)), _bit(bad.get(net)))
        else:
            g, b = _bit(good.get(net)), _bit(bad.get(net))
            side_values[position] = g if g is not None and g == b else None
    sides = sorted(side_values)
    if not d_values or not sides or len(sides) > MAX_SIDE_INPUTS:
        return 0, 0

    compatible: Dict[int, Set[int]] = {position: set() for position in sides}
    for assignment in product((0, 1), repeat=len(sides)):
        good_args = [0] * len(syms)
        bad_args = [0] * len(syms)
        for position, (g, b) in d_values.items():
            good_args[position], bad_args[position] = g, b
        for position, value in zip(sides, assignment):
            good_args[position] = bad_args[position] = value
        if _evaluate(lut, good_args) != _evaluate(lut, bad_args):
            for position, value in zip(sides, assignment):
                compatible[position].add(value)
    correct = sum(side_values[p] is not None and side_values[p] in compatible[p] for p in sides)
    return len(sides), correct


def attempt_signals(graph: SignalGraph, good: Dict[str, Any], bad: Dict[str, Any],
                    fault_net: str, activated: bool, detected: bool) -> Dict[str, float]:
    """Raw ratios in [0, 1] for one simulated vector.

    ``propagation``: D-nets / nets of the PO-reaching cone present in the table.
    ``controlling``: correctly assigned side inputs / side inputs, over the gates
    that read a D-net. ``depth``: how far the deepest D-net is from the fault
    site toward the nearest PO, a diagnostic. A detecting vector scores 1 on all
    three, so a broad dead-end region can never outrank it; a vector that does
    not activate the fault scores 0.
    """
    if detected:
        return {"propagation": 1.0, "controlling": 1.0, "depth": 1.0}
    if not activated:
        return {"propagation": 0.0, "controlling": 0.0, "depth": 0.0}

    cone = graph.cone(fault_net)
    if len(cone) == 1 and graph.dist.get(fault_net) != 0:
        # The fault net feeds nothing the structure can follow (a tie net, whose
        # consumers read a constant): a lone D-net would otherwise score as a full path.
        return {"propagation": 0.0, "controlling": 0.0, "depth": 0.0}
    present = [net for net in cone if net in good and net in bad]
    d_nets = {net for net in present
              if _bit(good[net]) is not None and _bit(bad[net]) is not None
              and _bit(good[net]) != _bit(bad[net])}
    propagation = len(d_nets) / len(present) if present else 0.0

    depth = 0.0
    start = graph.dist.get(fault_net)
    if d_nets and start is not None:
        nearest = min(graph.dist[net] for net in d_nets if net in graph.dist)
        depth = 1.0 if start == 0 else min(1.0, max(0.0, 1.0 - nearest / start))

    total = correct = 0
    visited: Set[int] = set()
    for net in d_nets:
        for index in graph.consumers.get(net, ()):
            record = graph.records[index]
            if index in visited or record[0] not in graph.dist:
                continue
            visited.add(index)
            sides, right = _side_score(record, good, bad, d_nets)
            total += sides
            correct += right
    if total:
        controlling = correct / total
    else:
        # No gate to pass: the effect either already sits on a PO or never moved.
        controlling = 1.0 if any(graph.dist.get(net) == 0 for net in d_nets) else 0.0
    return {"propagation": propagation, "controlling": controlling, "depth": depth}
