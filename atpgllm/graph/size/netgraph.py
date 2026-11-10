"""Net-level view of a compiled netlist, with COP controllability and observability.

Built from the fast simulator's ``OptimizedNetlist`` (topologically sorted
``instructions``), so every net name matches what the simulator reports.
One gate instruction drives one net. Cost: one pass forward and one pass
backward, O(V + E) times the truth-table size of each cell.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set, Tuple

from ..retest_labels import _truth_table

_LINE_COMMENT = re.compile(r"//[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)


def clean_netlist(text: str) -> str:
    """Drop Verilog comments. Raw FreeSet files open with a Design Compiler
    banner ("Created by: Synopsys DC Ultra(TM) ...") that the cell regex of
    both parsers would read as an instance. v4 netlists are already clean."""
    return _LINE_COMMENT.sub("", _BLOCK_COMMENT.sub("", text))


def _prob(sp: Dict[str, float], net: str) -> float:
    if net.isdigit():
        return float(int(net))
    return sp.get(net, 0.5)


def _p_one(table: Sequence[int], probs: Sequence[float]) -> float:
    return sum(
        math.prod(p if (idx >> k) & 1 else 1.0 - p for k, p in enumerate(probs))
        for idx, v in enumerate(table) if v
    )


@dataclass
class NetGraph:
    inputs: List[str]
    outputs: List[str]
    # net -> (lut, input nets in argument order); PIs and constants have none
    gates: Dict[str, Tuple[List[int], List[str]]] = field(default_factory=dict)
    fanout: Dict[str, List[str]] = field(default_factory=dict)  # net -> nets it feeds
    order: List[str] = field(default_factory=list)              # topological
    level: Dict[str, int] = field(default_factory=dict)
    sp: Dict[str, float] = field(default_factory=dict)          # P(net = 1)
    obs: Dict[str, float] = field(default_factory=dict)         # P(net observable)
    alias: Dict[str, str] = field(default_factory=dict)         # assign target -> source net

    def resolve(self, net: str) -> str:
        """The net an ``assign`` alias stands for (outputs are often aliases)."""
        seen = 0
        while net in self.alias and seen < 32:
            net = self.alias[net]
            seen += 1
        return net

    @property
    def depth(self) -> int:
        return max(self.level.values(), default=0)


def _sensitivity(table: Sequence[int], probs: Sequence[float], j: int) -> float:
    """P(output changes when argument ``j`` flips), other arguments independent."""
    total = 0.0
    n = len(probs)
    for idx in range(1 << n):
        if (idx >> j) & 1:
            continue
        if table[idx] == table[idx | (1 << j)]:
            continue
        w = 1.0
        for i, p in enumerate(probs):
            if i != j:
                w *= p if (idx >> i) & 1 else 1.0 - p
        total += w
    return total


def build_netgraph(compiled) -> NetGraph:
    g = NetGraph(inputs=list(compiled.input_nets), outputs=list(compiled.output_nets))
    alias = g.alias
    for net in g.inputs:
        g.level[net] = 0
        g.sp[net] = 0.5
        g.order.append(net)

    def resolve(net: str) -> str:
        seen = 0
        while net in alias and seen < 32:
            net = alias[net]
            seen += 1
        return net

    for instr in compiled.instructions:
        kind = instr[0]
        if kind == "assign_const":
            g.sp[instr[1]] = float(instr[2])
            g.level[instr[1]] = 0
            g.order.append(instr[1])
        elif kind == "assign_net":
            alias[instr[1]] = instr[2]
        elif kind == "gate":
            _, out, lut_or_fn, syms, input_map, _ = instr
            ins = [resolve(input_map.get(s, "")) for s in syms]
            table = _truth_table(lut_or_fn, len(syms))
            g.gates[out] = (table, ins)
            g.level[out] = 1 + max((g.level.get(i, 0) for i in ins), default=0)
            g.sp[out] = _p_one(table, [_prob(g.sp, i) for i in ins])
            g.order.append(out)
            for i in ins:
                g.fanout.setdefault(i, []).append(out)
    # Outputs that are aliases of internal nets observe those nets directly.
    g.outputs = [resolve(o) for o in g.outputs]
    po = set(g.outputs)
    for net in reversed(g.order):
        if net in po:
            g.obs[net] = 1.0
            continue
        miss = 1.0
        for out in g.fanout.get(net, ()):
            table, ins = g.gates[out]
            ps = [_prob(g.sp, i) for i in ins]
            for j, i in enumerate(ins):
                if i == net:
                    miss *= 1.0 - g.obs.get(out, 0.0) * _sensitivity(table, ps, j)
        g.obs[net] = 1.0 - miss
    return g


def transitive_fanin(g: NetGraph, net: str) -> Set[str]:
    seen, stack = set(), [net]
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        if n in g.gates:
            stack.extend(g.gates[n][1])
    return seen
