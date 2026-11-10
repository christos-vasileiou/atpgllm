"""Fault head: a small transformer over one fault's view, run per fault.

Tokens are the trunk vectors of the view nets and side nets (read from the
per-circuit cache) plus four gist tokens from the circuit spectrum. Each
token also gets its role (fault site, fanin, path, side input, gist), the
stuck value on the fault token, and a log-scaled hop distance to the fault
(S9). Cost per fault is O(T²·d + T·d²) with T about 324, whatever the
netlist size. The output is one vector ``z`` per view and side net, which the
anchored fusion hook adds to the prompt tokens naming that net.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import torch
from torch import Tensor, nn

ROLES = ("fault", "fanin", "path", "side", "gist")
GIST_TOKENS = 4


class FaultHead(nn.Module):
    def __init__(self, dim: int, spectrum_dim: int, *, layers: int = 2, heads: int = 4) -> None:
        super().__init__()
        self.dim = dim
        self.role = nn.Embedding(len(ROLES), dim)
        self.stuck = nn.Embedding(2, dim)
        self.dist = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        self.gist = nn.Linear(spectrum_dim, GIST_TOKENS * dim)
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.out = nn.LayerNorm(dim)

    def forward(self, node_vecs: Tensor, roles: Tensor, hops: Tensor, stuck_at: int,
                spectrum: Tensor) -> Tensor:
        """``node_vecs [T, d]`` trunk vectors of view and side nets, ``roles [T]``,
        ``hops [T]`` distance to the fault. Returns ``z [T, d]`` (gist dropped)."""
        x = node_vecs + self.role(roles) + self.dist(torch.log2(1 + hops.float()).unsqueeze(-1))
        x = x + (roles == ROLES.index("fault")).float().unsqueeze(-1) * self.stuck.weight[stuck_at]
        gist = self.gist(spectrum).view(GIST_TOKENS, self.dim) + self.role.weight[ROLES.index("gist")]
        h = self.encoder(torch.cat([x, gist], dim=0).unsqueeze(0)).squeeze(0)
        return self.out(h[: x.size(0)])


def view_tokens(view, g) -> Dict[str, List]:
    """Nets of a view in token order, with roles and hop distance to the fault."""
    nets, roles, hops = [], [], []
    path_nets = {n for p in view.paths for n in p[1:]}
    fanin_depth = {}
    for n in view.nets:
        if n == view.fault_net:
            fanin_depth[n] = 0
    for n in view.nets:
        if n in path_nets or n == view.fault_net:
            continue
        fanin_depth[n] = max(0, g.level.get(view.fault_net, 0) - g.level.get(n, 0))
    path_pos = {}
    for p in view.paths:
        for i, n in enumerate(p):
            path_pos[n] = min(path_pos.get(n, i), i)
    for n in view.nets:
        nets.append(n)
        if n == view.fault_net:
            roles.append(ROLES.index("fault")); hops.append(0)
        elif n in path_nets:
            roles.append(ROLES.index("path")); hops.append(path_pos.get(n, 0))
        else:
            roles.append(ROLES.index("fanin")); hops.append(fanin_depth.get(n, 0))
    for n in view.side:
        nets.append(n)
        roles.append(ROLES.index("side"))
        hops.append(abs(g.level.get(n, 0) - g.level.get(view.fault_net, 0)))
    return {"nets": nets, "roles": roles, "hops": hops}
