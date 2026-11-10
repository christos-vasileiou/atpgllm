"""S1 fault view and S2 side tokens, at net level.

The view keeps, at full detail and within ``cap`` nets:

- the fault net and its fanin cone, nearest levels first, up to
  ``fanin_share * cap`` nets (deeper fanin is cut, its boundary nets become
  side tokens);
- ``k`` propagation paths from the fault toward outputs, each step taking
  the fanout net with the highest COP observability, with nets already used
  by an earlier path discounted so paths spread out.

Side tokens are the other inputs of gates on the kept paths, nearest to
the fault first, then the cut boundary of the fanin, up to ``side_cap``.
In the full stack each side token is the trunk vector of that net, which
already summarizes its whole fanin cone.

Cost per fault: O(cap + side_cap) net visits, independent of the netlist.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

from .netgraph import NetGraph


@dataclass
class View:
    fault_net: str
    nets: List[str]
    paths: List[List[str]] = field(default_factory=list)
    side: List[str] = field(default_factory=list)
    fanin_truncated: bool = False
    paths_truncated: bool = False

    @property
    def net_set(self) -> Set[str]:
        return set(self.nets)

    @property
    def tokens(self) -> int:
        return len(self.nets) + len(self.side)


def build_view(g: NetGraph, fault_net: str, *, cap: int = 256, k: int = 4, side_cap: int = 128,
               fanin_share: float = 0.25, discount: float = 0.25, mode: str = "cone") -> View:
    """``mode="cone"`` (default): the fanout cone breadth first, nearest levels
    first, until the cap. ``mode="paths"``: ``k`` greedy observability paths.

    Defaults come from Check 1 (11-06, 81 random-detectable faults on 1k to
    10k gate FreeSet netlists): the effect chain was inside the view for 60%
    to 64% of faults with greedy paths, 86% with the cone at cap 256, fanin
    share 0.25 and side cap 128, and 96% at cap 512 with no side cap."""
    if fault_net not in g.level:
        raise KeyError(f"net {fault_net!r} is not in the netlist")
    nets: List[str] = [fault_net]
    in_view: Set[str] = {fault_net}
    fanin_budget = max(1, int(cap * fanin_share))
    boundary: List[str] = []
    queue = deque([fault_net])
    truncated = False
    while queue:
        net = queue.popleft()
        for i in g.gates.get(net, ((), ()))[1]:
            if i in in_view or i.isdigit():
                continue
            if len(nets) >= fanin_budget:
                truncated = True
                boundary.append(i)
                continue
            nets.append(i)
            in_view.add(i)
            queue.append(i)
    view = View(fault_net=fault_net, nets=nets, fanin_truncated=truncated)
    outputs = set(g.outputs)
    if mode == "cone":
        q = deque([fault_net])
        reached = {fault_net}
        while q and len(view.nets) < cap:
            n = q.popleft()
            for o in g.fanout.get(n, ()):
                if o in reached:
                    continue
                reached.add(o)
                if o not in in_view:
                    if len(view.nets) >= cap:
                        view.paths_truncated = True
                        break
                    view.nets.append(o)
                    in_view.add(o)
                q.append(o)
        if q:
            view.paths_truncated = True
        cone = [n for n in view.nets if n in reached and n != fault_net]
        view.paths = [[fault_net] + cone]
        k = 0
    elif mode != "paths":
        raise ValueError("mode must be 'paths' or 'cone'")
    used: Dict[str, int] = {}
    for _ in range(k):
        path = [fault_net]
        net = fault_net
        steps = 0
        while net not in outputs and steps < len(g.order):
            steps += 1
            nxt = g.fanout.get(net, [])
            if not nxt:
                break
            net = max(nxt, key=lambda n: g.obs.get(n, 0.0) * (discount ** used.get(n, 0)))
            path.append(net)
        for n in path[1:]:
            used[n] = used.get(n, 0) + 1
        kept = [fault_net]
        for n in path[1:]:
            if n not in in_view:
                if len(view.nets) >= cap:
                    view.paths_truncated = True
                    break
                view.nets.append(n)
                in_view.add(n)
            kept.append(n)
        view.paths.append(kept)
    # Side inputs along the kept paths, nearest to the fault first.
    side: List[str] = []
    seen: Set[str] = set(in_view)
    frontier = []
    for path in view.paths:
        for depth, n in enumerate(path[1:], start=1):
            for i in g.gates.get(n, ((), ()))[1]:
                if i not in seen and not i.isdigit():
                    frontier.append((depth, i))
                    seen.add(i)
    frontier.sort(key=lambda t: t[0])
    side.extend(n for _, n in frontier)
    for b in boundary:
        if b not in seen:
            side.append(b)
            seen.add(b)
    view.side = side[:side_cap]
    return view


def effect_chain_covered(g: NetGraph, view: View, differing: Sequence[str],
                         detecting_outputs: Sequence[str]) -> Optional[bool]:
    """Check 1 for one detecting pattern.

    True when some chain of nets that differ between the good and faulty
    machine runs from the fault to a detecting output inside the view, and
    every other input of the gates on that chain is in the view or among
    the side tokens. None when the pattern does not detect.
    """
    if not detecting_outputs:
        return None
    # The simulator names outputs by their port; the net graph by the net an
    # ``assign`` makes them stand for.
    differing = [g.resolve(n) for n in differing]
    detecting_outputs = [g.resolve(n) for n in detecting_outputs]
    diff = set(differing) | {view.fault_net}
    allowed = view.net_set & diff
    targets = set(detecting_outputs) & allowed
    if not targets:
        return False
    parent: Dict[str, Optional[str]] = {view.fault_net: None}
    queue = deque([view.fault_net])
    covered_by = view.net_set | set(view.side)
    while queue:
        net = queue.popleft()
        if net in targets:
            chain = []
            n: Optional[str] = net
            while n is not None:
                chain.append(n)
                n = parent[n]
            chain_set = set(chain)
            ok = all(
                i in chain_set or i in covered_by or i.isdigit()
                for c in chain if c != view.fault_net
                for i in g.gates.get(c, ((), ()))[1]
            )
            if ok:
                return True
            targets.discard(net)
            if not targets:
                return False
        for out in g.fanout.get(net, ()):
            if out in allowed and out not in parent:
                parent[out] = net
                queue.append(out)
    return False
