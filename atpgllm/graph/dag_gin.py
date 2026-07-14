"""
DAG-aware bidirectional GIN (Graph Isomorphism Network) encoder for
circuit netlists.

Key design choices and their rationale:

1. **GIN over GraphConv/GCN.**  GIN with sum aggregation is provably
   maximally expressive among 1-WL GNNs.  For fault propagation analysis,
   we need to distinguish reconvergent fanout structures and different
   fanin/fanout counts — GraphConv (mean aggregation) literally cannot
   tell 2 identical neighbours apart from 5.

2. **Bidirectional (forward + backward) message passing.**  Netlists are
   DAGs.  A gate's output value depends on its *fanin* (forward /
   controllability), while fault *observability* depends on its *fanout*
   (backward).  Separate GIN aggregations per direction capture both.

3. **JumpingKnowledge (JK) concatenation.**  Shallow layers capture local
   gate-level patterns; deep layers see multi-level cones of influence.
   JK preserves information from all depths.

4. **No graph transformer / global attention.**  Full self-attention is
   O(N²) and kills VRAM for 10 k+ gate netlists.  GIN is O(|E|) per
   layer.

Compatible with PyG batched graphs (``DataLoader`` + ``Batch``).
"""

from __future__ import annotations

from typing import Dict, Literal

import torch
from torch import nn, Tensor
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool, global_add_pool


# =====================================================================
# Single bidirectional GIN layer
# =====================================================================


def _gin_mlp(dim: int) -> nn.Sequential:
    """2-layer MLP with BatchNorm, as prescribed by the original GIN paper."""
    return nn.Sequential(
        nn.Linear(dim, dim),
        nn.BatchNorm1d(dim),
        nn.ReLU(),
        nn.Linear(dim, dim),
        nn.BatchNorm1d(dim),
        nn.ReLU(),
    )


class DAGGINLayer(nn.Module):
    """One layer of bidirectional GIN message passing.

    For each node *v*:

    .. math::

        h_v^{\\text{fwd}} = \\text{MLP}_{\\text{fwd}}\\bigl(
            (1+\\varepsilon_{\\text{fwd}})\\,h_v
            + \\sum_{u \\in \\text{pred}(v)} h_u
        \\bigr)

        h_v^{\\text{bwd}} = \\text{MLP}_{\\text{bwd}}\\bigl(
            (1+\\varepsilon_{\\text{bwd}})\\,h_v
            + \\sum_{w \\in \\text{succ}(v)} h_w
        \\bigr)

        h_v' = \\text{MLP}_{\\text{fuse}}\\bigl(
            [h_v^{\\text{fwd}} \\| h_v^{\\text{bwd}}]
        \\bigr)

    ``pred(v)`` = fanin  gates (drivers), ``succ(v)`` = fanout gates (sinks).
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.gin_fwd = GINConv(nn=_gin_mlp(hidden_dim), train_eps=True)
        self.gin_bwd = GINConv(nn=_gin_mlp(hidden_dim), train_eps=True)

        self.fuse = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        edge_index_fwd: Tensor,
        edge_index_bwd: Tensor,
    ) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor[N, H]
        edge_index_fwd : Tensor[2, E]
            Forward edges: ``edge_index_fwd[0]`` = predecessor (driver),
            ``edge_index_fwd[1]`` = successor (sink).  Each node aggregates
            from its *fanin*.
        edge_index_bwd : Tensor[2, E]
            Reversed edges (``edge_index_fwd.flip(0)``).  Each node
            aggregates from its *fanout*.
        """
        h_fwd = self.gin_fwd(x, edge_index_fwd)
        h_bwd = self.gin_bwd(x, edge_index_bwd)
        h = self.fuse(torch.cat([h_fwd, h_bwd], dim=-1))
        return self.dropout(h)


# =====================================================================
# Full DAG-GIN encoder
# =====================================================================


class DAGGINEncoder(nn.Module):
    """Multi-layer DAG-aware GIN with JumpingKnowledge and multi-pool readout.

    Parameters
    ----------
    in_dim : int
        Input node feature dimension (output of ``AttributeDecompositionEncoder``).
    hidden_dim : int
        Hidden / output dimensionality per node.
    num_layers : int
        Number of bidirectional GIN layers.
    dropout : float
        Dropout probability applied after each layer.
    num_pools : int
        Number of pooling functions for graph-level readout (up to 4:
        mean / max / sum / min).
    jk_mode : ``"cat"`` | ``"last"``
        JumpingKnowledge strategy.  ``"cat"`` concatenates all layer
        outputs (including the initial projection) and projects back.
        ``"last"`` uses only the final layer.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        dropout: float = 0.1,
        num_pools: int = 4,
        jk_mode: Literal["cat", "last"] = "cat",
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.num_pools = min(num_pools, 4)
        self.jk_mode = jk_mode

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.layers = nn.ModuleList([
            DAGGINLayer(hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        if jk_mode == "cat":
            # +1 for the initial projection
            self.jk_proj = nn.Linear(hidden_dim * (num_layers + 1), hidden_dim)
        else:
            self.jk_proj = None

        self.out_dim = hidden_dim

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Parameters
        ----------
        x : Tensor[N_total, in_dim]
            Node features (from ``AttributeDecompositionEncoder``).
        edge_index : Tensor[2, E_total]
            **Forward** directed edges (driver → sink, as produced by
            ``netlist_parser``).
        batch : Tensor[N_total]
            Graph membership per node (from PyG ``DataLoader``).

        Returns
        -------
        dict
            ``node_embs`` : Tensor[N_total, hidden_dim]
                Per-node embeddings (for Q-Former cross-attention).
            ``graph_embs`` : Tensor[B, hidden_dim × num_pools]
                Graph-level embeddings (for contrastive / matching losses).
        """
        edge_index_bwd = edge_index.flip(0)

        h = self.input_proj(x)
        layer_outputs = [h]

        for layer in self.layers:
            h_new = layer(h, edge_index, edge_index_bwd)
            h = h + h_new  # residual
            layer_outputs.append(h)

        # JumpingKnowledge
        if self.jk_mode == "cat":
            h_jk = torch.cat(layer_outputs, dim=-1)
            node_embs = self.jk_proj(h_jk)
        else:
            node_embs = layer_outputs[-1]

        # Graph-level multi-pool readout
        pools = [
            global_mean_pool(node_embs, batch),
            global_max_pool(node_embs, batch),
            global_add_pool(node_embs, batch),
            -global_max_pool(-node_embs, batch),  # min pool
        ][: self.num_pools]
        graph_embs = torch.cat(pools, dim=-1)

        return {"node_embs": node_embs, "graph_embs": graph_embs}
