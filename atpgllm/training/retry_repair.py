"""Repair paths for retry SFT: from a failed call to a detecting one, step by step.

A retry trajectory starts from a call that did not detect the fault, usually
the policy's own failed attempt, and continues with new calls until one
detects. Every new call changes a few primary inputs of the previous one, and
the change answers what the previous snapshot showed:

* the fault was not activated, so inputs in the fan-in of the fault site change
  until the site carries the opposite of the stuck-at value;
* the fault was activated but blocked, so inputs in the fan-in of the side
  inputs that stop the difference change until a primary output differs. When
  no small change detects, a change that carries the difference further is
  taken and the next snapshot is read again.

The search tries the fewest flips first (one, then two, then three), so a path
is a sequence of small, readable edits. It runs on any simulator that returns
the tool's ``{"Good Machine": ..., "Bad Machine": ...}`` table; the builder
uses the fast simulator for the search and the tool's own backend for the
snapshots shown in the conversation.
"""

from __future__ import annotations

import itertools
import json
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from atpgllm.training.tool_feedback import SimulationReading, parse_fault, read_snapshot


Blocker = Tuple[str, str, str, Optional[str]]  # (gate output, D input, side input, side value)


@dataclass(frozen=True)
class Observation:
    """One call: the vector, what the tool returned and what that shows."""

    vector: Dict[str, Any]
    snapshot: str
    reading: SimulationReading
    blockers: Tuple[Blocker, ...] = ()
    propagation: float = 0.0
    controlling: float = 0.0

    @property
    def good(self) -> Dict[str, Any]:
        return json.loads(self.snapshot)["Good Machine"]

    def score(self) -> Tuple[float, ...]:
        return (float(self.reading.detected), float(self.reading.activated),
                self.propagation, self.controlling)


@dataclass(frozen=True)
class RepairStep:
    """A new call: the inputs that changed and the observation it produced."""

    changes: Tuple[Tuple[str, Any], ...]
    observation: Observation


def _bit(value: Any) -> Optional[str]:
    text = str(value).strip()
    return text if text in ("0", "1") else None


def _evaluate(lut: Any, args: List[int]) -> int:
    if isinstance(lut, list):
        index = 0
        for position, value in enumerate(args):
            index |= value << position
        return lut[index]
    return int(bool(lut(*args)))


def blocking_side_inputs(graph: Any, good: Dict[str, Any], bad: Dict[str, Any],
                         fault_net: str) -> Tuple[List[Blocker], float, float]:
    """Gates that read the fault effect but stop it, and the side inputs that do.

    Returns ``(blockers, propagation, controlling)``. A side input blocks when
    its value is unknown or no completion of the gate's other side inputs lets
    the difference through with it (the exact cell condition, as in the staged
    reward's controlling objective). ``propagation`` and ``controlling`` are the
    staged reward's raw ratios for the same table.
    """
    from atpgllm.llm.propagation_signals import attempt_signals

    activated = any(
        _bit(good.get(n)) is not None and _bit(bad.get(n)) is not None
        and _bit(good[n]) != _bit(bad.get(n))
        for n in good
    )
    signals = attempt_signals(graph, good, bad, fault_net, activated, False)
    cone = graph.cone(fault_net)
    d_nets = {n for n in cone if n in good and n in bad
              and _bit(good[n]) is not None and _bit(bad[n]) is not None
              and _bit(good[n]) != _bit(bad[n])}
    blockers: List[Blocker] = []
    seen: Set[int] = set()
    for net in sorted(d_nets):
        for index in graph.consumers.get(net, ()):
            if index in seen:
                continue
            seen.add(index)
            out_net, lut, syms, input_map, _ = graph.records[index]
            if out_net not in graph.dist or out_net in d_nets or lut is None or input_map is None:
                continue
            d_pos, sides = {}, {}
            for position, sym in enumerate(syms):
                source = input_map.get(sym)
                if source is None:
                    break
                if source.isdigit():
                    sides[position] = (source, int(source))
                elif source in d_nets:
                    d_pos[position] = (int(_bit(good[source])), int(_bit(bad[source])))
                else:
                    g, b = _bit(good.get(source)), _bit(bad.get(source))
                    sides[position] = (source, int(g) if g is not None and g == b else None)
            else:
                if not d_pos or not sides or len(sides) > 6:
                    continue
                order = sorted(sides)
                compatible = {p: set() for p in order}
                for assignment in itertools.product((0, 1), repeat=len(order)):
                    good_args = [0] * len(syms)
                    bad_args = [0] * len(syms)
                    for p, (g, b) in d_pos.items():
                        good_args[p], bad_args[p] = g, b
                    for p, value in zip(order, assignment):
                        good_args[p] = bad_args[p] = value
                    if _evaluate(lut, good_args) != _evaluate(lut, bad_args):
                        for p, value in zip(order, assignment):
                            compatible[p].add(value)
                d_input = input_map[syms[min(d_pos)]]
                for p in order:
                    source, value = sides[p]
                    if source.isdigit():
                        continue
                    if value is None or value not in compatible[p]:
                        blockers.append((out_net, d_input, source,
                                         None if value is None else str(value)))
    return blockers, signals["propagation"], signals["controlling"]


def fast_structure(netlist_text: str) -> Any:
    """The fast simulator's compiled netlist, whatever ``FAULT_SIM_BACKEND`` says.

    A full-scan (v5 sequential) netlist becomes a ``ScanModel`` over the cut
    view: inputs are the primary inputs plus each flop's QN net, outputs the
    primary outputs plus each flop's D net, as in the tool's vectors, and its
    ``instructions`` see no path through a flop.
    """
    from atpgllm.training._paths import ensure_data_preprocessing_on_path, resolve_sim_config_path
    ensure_data_preprocessing_on_path()
    from fault_sim import OptimizedNetlist
    from scan_view import ScanModel, is_scan_netlist
    from atpgllm.training.reward_function_factory import RewardFunctionFactory as RF

    with resolve_sim_config_path().open(encoding="utf-8") as handle:
        gates = json.load(handle)
    if is_scan_netlist(netlist_text):
        return ScanModel(netlist_text, lambda cut: OptimizedNetlist(cut, gates, RF.DECL_RE, RF.NAME_RE))
    return OptimizedNetlist(netlist_text, gates, RF.DECL_RE, RF.NAME_RE)


def fast_simulator(structure: Any, fault: str,
                   output_nets: Sequence[str]) -> Callable[[Dict[str, Any]], str]:
    """``vector -> tool table`` on the fast simulator (search only, never shown)."""
    from fault_sim import fast_fault_sim

    expected = {net: 0 for net in output_nets}

    def simulate(vector: Dict[str, Any]) -> str:
        frame = fast_fault_sim(vector, expected, fault, structure)
        if "error" in frame.columns:
            raise ValueError(str(frame.loc[0, "error"]))
        return frame[["Good Machine", "Bad Machine"]].to_json()

    return simulate


class RepairSearch:
    """Fewest-flip repairs of failed calls on one netlist and fault."""

    def __init__(self, structure: Any, simulate: Callable[[Dict[str, Any]], str], fault: str,
                 input_nets: Sequence[str], output_nets: Sequence[str], *,
                 seed: int = 0, max_flips: int = 3, single_cap: int = 96,
                 pair_pool: int = 12, triple_pool: int = 8):
        from atpgllm.llm.propagation_signals import SignalGraph

        self.structure = structure
        self.graph = SignalGraph(structure, list(output_nets))
        self.simulate_table = simulate
        self.fault = fault
        self.stuck_at, self.fault_net = parse_fault(fault)
        self.input_nets = list(input_nets)
        self.output_nets = list(output_nets)
        self.rng = random.Random(seed)
        self.max_flips = max_flips
        self.single_cap, self.pair_pool, self.triple_pool = single_cap, pair_pool, triple_pool
        self.producer: Dict[str, int] = {}
        for index, record in enumerate(self.graph.records):
            self.producer[record[0]] = index
        self._fanin: Dict[str, Tuple[str, ...]] = {}
        self._cache: Dict[Tuple[Tuple[str, str], ...], Observation] = {}
        self.simulations = 0

    # -- structure ----------------------------------------------------------------
    def fanin_inputs(self, nets: Sequence[str]) -> List[str]:
        """Primary inputs that can change any of ``nets``, in declared order."""
        found: Set[str] = set()
        inputs = set(self.input_nets)
        for net in nets:
            if net not in self._fanin:
                seen, stack, hits = {net}, [net], set()
                while stack:
                    current = stack.pop()
                    if current in inputs:
                        hits.add(current)
                    index = self.producer.get(current)
                    if index is None:
                        continue
                    for source in self.graph.records[index][4]:
                        if source not in seen and not source.isdigit():
                            seen.add(source)
                            stack.append(source)
                self._fanin[net] = tuple(hits)
            found.update(self._fanin[net])
        return [net for net in self.input_nets if net in found]

    # -- simulation ---------------------------------------------------------------
    def observe(self, vector: Dict[str, Any]) -> Observation:
        key = tuple(sorted((net, str(value)) for net, value in vector.items()))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        snapshot = self.simulate_table(dict(vector))
        self.simulations += 1
        observation = self.explain(vector, snapshot)
        self._cache[key] = observation
        return observation

    def explain(self, vector: Dict[str, Any], snapshot: str) -> Observation:
        """The reading of ``snapshot`` plus the side inputs that block the effect."""
        reading = read_snapshot(snapshot, self.fault, self.output_nets)
        data = json.loads(snapshot)
        blockers: List[Blocker] = []
        propagation = controlling = 0.0
        if reading.activated and not reading.detected:
            blockers, propagation, controlling = blocking_side_inputs(
                self.graph, data["Good Machine"], data["Bad Machine"], self.fault_net)
        elif reading.detected:
            propagation = controlling = 1.0
        return Observation(dict(vector), snapshot, reading, tuple(blockers), propagation, controlling)

    # -- search -------------------------------------------------------------------
    def _candidates(self, vector: Dict[str, Any], pool: Sequence[str]) -> List[Tuple[str, Any]]:
        """Single-input changes: a 0/1 input flips, an x input tries 0 and 1."""
        changes = []
        for net in pool:
            value = _bit(vector.get(net))
            if value is None:
                changes += [(net, 0), (net, 1)]
            else:
                changes.append((net, 1 - int(value)))
        return changes

    def next_step(self, current: Observation) -> Optional[RepairStep]:
        """The fewest-flip change that fixes what ``current`` shows, or None."""
        reading = current.reading
        if reading.detected:
            return None
        if not reading.activated:
            pool = ([self.fault_net] if self.fault_net in self.input_nets
                    else self.fanin_inputs([self.fault_net]))

            def goal(o: Observation) -> bool:
                return o.reading.activated
        else:
            sides = sorted({side for _, _, side, _ in current.blockers})
            pool = self.fanin_inputs(sides) if sides else []
            if not pool:
                pool = self.fanin_inputs(sorted(self.graph.cone(self.fault_net)))

            def goal(o: Observation) -> bool:
                return o.reading.detected
        singles = self._candidates(current.vector, pool)
        if len(singles) > self.single_cap:
            singles = self.rng.sample(singles, self.single_cap)
        if not singles:
            return None

        def apply(changes: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
            vector = dict(current.vector)
            for net, value in changes:
                vector[net] = value
            return vector

        tried: List[Tuple[Tuple[Tuple[str, Any], ...], Observation]] = []
        for size in range(1, self.max_flips + 1):
            if size == 1:
                combos = [(c,) for c in singles]
            else:
                width = self.pair_pool if size == 2 else self.triple_pool
                ranked = [c[0] for c, _ in sorted(
                    ((c, o) for c, o in tried if len(c) == 1),
                    key=lambda item: item[1].score(), reverse=True)][:width]
                combos = [combo for combo in itertools.combinations(ranked, size)
                          if len({net for net, _ in combo}) == size]
            hits = []
            for combo in combos:
                observation = self.observe(apply(combo))
                tried.append((combo, observation))
                if goal(observation):
                    hits.append((combo, observation))
            if hits:
                combo, observation = max(hits, key=lambda item: item[1].score())
                return RepairStep(tuple(combo), observation)
        if reading.activated:
            # No small change detects: take the one that carries the effect furthest,
            # if it beats the current table, and read the next snapshot.
            better = [(c, o) for c, o in tried
                      if o.reading.activated and o.score() > current.score()]
            if better:
                best = max(o.score() for _, o in better)
                combo, observation = min(
                    ((c, o) for c, o in better if o.score() == best), key=lambda item: len(item[0]))
                return RepairStep(tuple(combo), observation)
        return None

    def path(self, start: Dict[str, Any], *, max_steps: int = 4,
             start_snapshot: Optional[str] = None) -> Optional[List[RepairStep]]:
        """Steps from ``start`` to a detecting call, or None within ``max_steps``."""
        current = (self.explain(start, start_snapshot) if start_snapshot is not None
                   else self.observe(start))
        if current.reading.detected:
            return None
        steps: List[RepairStep] = []
        seen = {tuple(sorted((n, str(v)) for n, v in start.items()))}
        for _ in range(max_steps):
            step = self.next_step(current)
            if step is None:
                return None
            key = tuple(sorted((n, str(v)) for n, v in step.observation.vector.items()))
            if key in seen:
                return None
            seen.add(key)
            steps.append(step)
            current = step.observation
            if current.reading.detected:
                return steps
        return None
