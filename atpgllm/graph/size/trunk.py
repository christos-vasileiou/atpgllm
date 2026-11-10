"""S10 trunk: one forward and one backward topological sweep, fault-agnostic.

Each node is updated once per direction, in level order: a GRU cell takes the
node's own features as input and the mean of messages from its fanin
(forward) or fanout (backward) as state. The update gate starts biased
toward keeping that state, so a value can survive hundreds of levels. All nodes of one
level are updated together (level batching), so the sweep takes 2L steps of
O(E_level·d + V_level·d²) work: O(E·d + V·d²) in total, O(V·d) memory.

Node features are local only. The Stage A features divide depth by the
graph's maximum depth and port position by the port count, so the same gate
changes as the netlist grows. Here depths are log-scaled hop counts (S9),
which keeps the trunk local: a circuit embedded in a host that does not feed
it, or read from it, gets the same vectors.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn
from torch_geometric.utils import scatter

from ..gate_features import GateAttributeVocab
from ..node_encoder import AttributeDecompositionEncoder

NUM_LOCAL_FEATURES = 6


def levelize(edge_index: Tensor, num_nodes: int) -> Tuple[Tensor, Tensor]:
    """Longest-path level from sources (``level``) and to sinks (``rlevel``)."""
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    succ: List[List[int]] = [[] for _ in range(num_nodes)]
    pred: List[List[int]] = [[] for _ in range(num_nodes)]
    for s, d in zip(src, dst):
        succ[s].append(d)
        pred[d].append(s)

    def sweep(nexts, prevs):
        lvl = [0] * num_nodes
        remaining = [len(p) for p in prevs]
        q = deque(v for v in range(num_nodes) if remaining[v] == 0)
        seen = 0
        while q:
            v = q.popleft()
            seen += 1
            for w in nexts[v]:
                lvl[w] = max(lvl[w], lvl[v] + 1)
                remaining[w] -= 1
                if remaining[w] == 0:
                    q.append(w)
        if seen != num_nodes:
            raise ValueError("graph has a cycle")
        return torch.tensor(lvl, dtype=torch.long)

    return sweep(succ, pred), sweep(pred, succ)


def local_features(edge_index: Tensor, num_nodes: int, level: Tensor, rlevel: Tensor,
                   is_pi: Tensor, is_po: Tensor) -> Tensor:
    """Six local, size-free features per node."""
    ones = torch.ones(edge_index.size(1))
    in_deg = scatter(ones, edge_index[1], dim=0, dim_size=num_nodes, reduce="sum")
    out_deg = scatter(ones, edge_index[0], dim=0, dim_size=num_nodes, reduce="sum")
    return torch.stack([
        torch.log2(1 + level.float()), torch.log2(1 + rlevel.float()),
        torch.log2(1 + in_deg), torch.log2(1 + out_deg),
        is_pi.float(), is_po.float(),
    ], dim=-1)


class TopoSweepTrunk(nn.Module):
    def __init__(self, vocab: GateAttributeVocab, dim: int = 256, per_attr_dim: int = 16,
                 attr_hidden: int = 128) -> None:
        super().__init__()
        self.dim = dim
        self.attr = AttributeDecompositionEncoder.from_vocab(
            vocab, per_attr_dim=per_attr_dim, structural_dim=NUM_LOCAL_FEATURES,
            hidden_dim=attr_hidden, out_dim=dim)
        self.driver_pin = nn.Embedding(vocab.num_pins, dim)
        self.sink_pin = nn.Embedding(vocab.num_pins, dim)
        self.fwd_msg = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU())
        self.bwd_msg = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU())
        self.fwd_cell = nn.GRUCell(dim, dim)
        self.bwd_cell = nn.GRUCell(dim, dim)
        # Normalized state: residual messages would otherwise grow with depth.
        self.fwd_norm = nn.LayerNorm(dim)
        self.bwd_norm = nn.LayerNorm(dim)
        self.out = nn.Sequential(nn.Linear(2 * dim, dim), nn.LayerNorm(dim))
        # Long memory: the GRU state is the fanin's message, the input is the
        # node's own features, and the update gate starts biased toward
        # keeping the state (z about 0.88), so a value decays slowly across
        # hundreds of levels until training decides what to pass on.
        for cell in (self.fwd_cell, self.bwd_cell):
            with torch.no_grad():
                cell.bias_hh[dim:2 * dim].fill_(2.0)

    @staticmethod
    def prepare(data) -> None:
        """Attach levels and local features to a schema-v2 graph (once per circuit)."""
        n = data.num_nodes
        level, rlevel = levelize(data.edge_index, n)
        is_po = data.po_node_mask
        is_pi = ~data.gate_node_mask & ~is_po
        data.level = level
        data.rlevel = rlevel
        data.local_feats = local_features(data.edge_index, n, level, rlevel, is_pi, is_po)

    def _sweep(self, x: Tensor, order: Tensor, src: Tensor, dst: Tensor, edge: Tensor,
               msg: nn.Module, cell: nn.GRUCell, norm: nn.Module) -> Tensor:
        """Update nodes level by level; ``src`` feeds ``dst``, ``order`` is dst's level.

        Nodes and edges are sorted by level once, so each step touches only
        its own level: O(V + E) work over all levels plus L small steps. (A
        per-level mask over every node and edge made 3938_top, 8,934 levels,
        take 764 s on CPU.) With gradients, each step copies the state
        (O(V·d)), so train on crops or level windows of deep circuits."""
        n = x.size(0)
        h = cell(x, torch.zeros_like(x))          # level-0 nodes: empty state
        if n == 0:
            return h
        max_level = int(order.max())
        edge_level = order[dst]
        node_order = torch.argsort(order, stable=True)
        edge_order = torch.argsort(edge_level, stable=True)
        node_bounds = torch.cumsum(torch.bincount(order, minlength=max_level + 1), 0).tolist()
        edge_bounds = torch.cumsum(torch.bincount(edge_level, minlength=max_level + 1), 0).tolist()
        # Position of each node inside its own level, for per-level scatter.
        starts = torch.tensor([0] + node_bounds[:-1], device=order.device)
        pos = torch.empty(n, dtype=torch.long, device=order.device)
        pos[node_order] = torch.arange(n, device=order.device) - starts[order[node_order]]
        grad = torch.is_grad_enabled()
        for lvl in range(1, max_level + 1):
            n0, n1 = node_bounds[lvl - 1], node_bounds[lvl]
            if n1 == n0:
                continue
            nodes = node_order[n0:n1]
            sel = edge_order[edge_bounds[lvl - 1]:edge_bounds[lvl]]
            hs = h[src[sel]]
            m = hs + msg(torch.cat([hs, edge[sel]], dim=-1))   # residual: the state passes through
            agg = norm(scatter(m, pos[dst[sel]], dim=0, dim_size=n1 - n0, reduce="mean"))
            new = cell(x[nodes], agg)
            if grad:
                h = h.index_copy(0, nodes, new)
            else:
                h.index_copy_(0, nodes, new)
        return h

    def forward(self, data) -> Dict[str, Tensor]:
        x = self.attr(data.gate_attrs, data.local_feats, None)
        e = self.driver_pin(data.edge_pins[:, 0]) + self.sink_pin(data.edge_pins[:, 1])
        src, dst = data.edge_index[0], data.edge_index[1]
        fwd = self._sweep(x, data.level, src, dst, e, self.fwd_msg, self.fwd_cell, self.fwd_norm)
        bwd = self._sweep(fwd, data.rlevel, dst, src, e, self.bwd_msg, self.bwd_cell, self.bwd_norm)
        return {"node_embs": self.out(torch.cat([fwd, bwd], dim=-1)), "fwd": fwd, "bwd": bwd}
