"""Fault-seeded role queries for the BRIDGES re-test.

The Stage B Q-Former used 32 global learned queries and trained text
generation on their mean, so the queries collapsed (cosine 0.93). Here:

- every query starts from a role embedding, a slot embedding, the fault
  site's node vector and the stuck value, so queries differ per fault;
- four roles with four slots each: ``gist`` (circuit and fault summary for
  text alignment), ``care`` (which inputs to set, and to what), ``path``
  (gates that carry the effect) and ``obs`` (outputs that see it);
- each non-gist role has a pointer head over nodes, trained on soft targets
  from :mod:`atpgllm.graph.retest_labels`, so a query must point at
  specific nodes and cannot copy its neighbours;
- variance, covariance and within-graph decorrelation terms keep the
  queries apart.

The module keeps the ``forward(node_embs, batch) -> [B, Q, H]`` contract of
:class:`~atpgllm.graph.models_stage1.GraphQFormer`, with an optional
``fault_feats`` argument, so the Stage C wrapper can use it unchanged.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.utils import scatter, to_dense_batch

from .models_stage1 import QFormerBlock

ROLES = ("gist", "care", "path", "obs")
POINTER_ROLES = ("care", "path", "obs")


class RoleQueryFormer(nn.Module):
    def __init__(
        self,
        d_node: int,
        hidden_dim: int = 512,
        slots_per_role: int = 4,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: Optional[int] = None,
        cross_attn_every_n: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.slots_per_role = slots_per_role
        self.num_queries = len(ROLES) * slots_per_role
        self.uses_fault = True
        ffn_dim = ffn_dim or 4 * hidden_dim
        # Unit-scale role and slot codes: with 0.02 the fault term swamps
        # them and every query of a graph starts as the same vector.
        self.role_emb = nn.Parameter(torch.randn(len(ROLES), hidden_dim))
        self.slot_emb = nn.Parameter(torch.randn(self.num_queries, hidden_dim) * 0.5)
        # Each role reads the fault through its own projection: one shared
        # fault vector added to every query pulls them together.
        self.stuck_emb = nn.Embedding(2, len(ROLES) * hidden_dim)
        nn.init.normal_(self.stuck_emb.weight, std=0.1)
        self.fault_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, len(ROLES) * hidden_dim),
        )
        self.fault_norm = nn.LayerNorm(hidden_dim)
        self.graph_in_proj = nn.Linear(d_node, hidden_dim)
        self.graph_in_norm = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList([
            QFormerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                use_cross=((i + 1) % cross_attn_every_n == 0),
                dropout=dropout,
            )
            for i in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.ptr_q = nn.ModuleDict({r: nn.Linear(hidden_dim, hidden_dim) for r in POINTER_ROLES})
        self.ptr_k = nn.ModuleDict({r: nn.Linear(hidden_dim, hidden_dim) for r in POINTER_ROLES})
        self.value_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        self.last: Dict[str, Tensor] = {}

    # Read only by the R3 losses and the R4 vector decoder. Stage C passes the
    # queries to Granite and never calls these, so it freezes them (with two
    # GPUs, DDP fails on trainable parameters that get no gradient).
    DECODER_HEADS = ("ptr_q", "ptr_k", "value_head")

    def role_slice(self, role: str) -> slice:
        i = ROLES.index(role)
        return slice(i * self.slots_per_role, (i + 1) * self.slots_per_role)

    def forward(
        self,
        node_embs: Tensor,
        batch: Tensor,
        fault_feats: Optional[Tensor] = None,
    ) -> Tensor:
        node_h = self.graph_in_norm(self.graph_in_proj(node_embs))
        dense, mask = to_dense_batch(node_h, batch)
        B = dense.size(0)
        q = (
            self.role_emb.repeat_interleave(self.slots_per_role, dim=0) + self.slot_emb
        ).unsqueeze(0).expand(B, -1, -1)
        if fault_feats is not None:
            site = fault_feats[:, 0].to(node_h.dtype)
            num = scatter(node_h * site.unsqueeze(-1), batch, dim=0, dim_size=B, reduce="sum")
            den = scatter(site, batch, dim=0, dim_size=B, reduce="sum").clamp(min=1.0)
            fault_vec = num / den.unsqueeze(-1)
            stuck = scatter(fault_feats[:, 4].to(node_h.dtype), batch, dim=0, dim_size=B, reduce="max")
            R, H = len(ROLES), self.hidden_dim
            per_role = self.fault_norm(self.fault_proj(fault_vec).view(B, R, H))
            per_role = per_role + self.stuck_emb(stuck.long().clamp(0, 1)).view(B, R, H)
            q = q + per_role.repeat_interleave(self.slots_per_role, dim=1)
        q = q.contiguous()
        for block in self.blocks:
            q = block(q, graph_kv=dense, graph_kv_mask=mask)
        q = self.out_norm(q)
        self.last = {"queries": q, "nodes": dense, "node_mask": mask}
        return q

    def pointer_scores(self, role: str, queries: Tensor, nodes: Tensor) -> Tensor:
        """Per-slot scores ``[B, S, N]`` of the role's queries against nodes."""
        qs = self.ptr_q[role](queries[:, self.role_slice(role)])
        ks = self.ptr_k[role](nodes)
        return torch.einsum("bsh,bnh->bsn", qs, ks) / math.sqrt(self.hidden_dim)

    def pointer_logits(self, role: str, queries: Tensor, nodes: Tensor) -> Tensor:
        """One logit per node ``[B, N]``: the best slot's score."""
        return self.pointer_scores(role, queries, nodes).max(dim=1).values

    def value_logits(self, queries: Tensor, nodes: Tensor) -> Tensor:
        """Logit of value 1 for each node ``[B, N]``, read through the care slots."""
        scores = self.pointer_scores("care", queries, nodes)       # [B, S, N]
        attn = scores.softmax(dim=1)                                # over slots
        ctx = torch.einsum("bsn,bsh->bnh", attn, queries[:, self.role_slice("care")])
        return self.value_head(torch.cat([nodes, ctx], dim=-1)).squeeze(-1)


class QueryGroundedTextGenerator(nn.Module):
    """Small causal decoder whose memory is all queries ``[B, Q, D]``.

    The Stage B generator conditioned on the mean of the queries, which gave
    the queries no reason to differ. Here every text token can attend to
    every query, as in BLIP-2."""

    def __init__(self, vocab_size: int, d_model: int, *, query_dim: int,
                 num_layers: int = 2, num_heads: int = 8, max_len: int = 256) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.memory_proj = nn.Linear(query_dim, d_model)
        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, queries: Tensor, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        tgt = self.token_emb(input_ids) + self.pos_emb(pos)
        causal = torch.triu(torch.ones(T, T, device=input_ids.device, dtype=torch.bool), diagonal=1)
        out = self.decoder(tgt=tgt, memory=self.memory_proj(queries), tgt_mask=causal,
                           tgt_key_padding_mask=attention_mask == 0)
        return self.lm_head(out)


# ---------------------------------------------------------------------
# Losses and collapse metrics
# ---------------------------------------------------------------------


def soft_pointer_bce(logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """BCE with soft targets over masked nodes, positives reweighted so a
    few true nodes among many still matter."""
    valid = mask.bool()
    if not valid.any():
        return logits.sum() * 0.0
    x = logits[valid]
    y = target[valid].to(x.dtype)
    pos = y.sum()
    neg = (1.0 - y).sum()
    pos_weight = (neg / pos.clamp(min=1.0)).clamp(max=50.0)
    return F.binary_cross_entropy_with_logits(x, y, pos_weight=pos_weight)


def weighted_value_bce(logits: Tensor, one_target: Tensor, weight: Tensor, mask: Tensor) -> Tensor:
    """BCE on the value of each input, weighted by how often it is a care bit."""
    w = weight * mask.to(weight.dtype)
    if w.sum() <= 0:
        return logits.sum() * 0.0
    loss = F.binary_cross_entropy_with_logits(logits, one_target.to(logits.dtype), reduction="none")
    return (loss * w).sum() / w.sum()


def anti_collapse(queries: Tensor, eps: float = 1e-4) -> Dict[str, Tensor]:
    """Variance and covariance across the batch (per slot), plus mean squared
    cosine between different queries of the same graph."""
    B, Q, H = queries.shape
    z = queries.reshape(B * Q, H)
    z = z - z.mean(dim=0)
    std = torch.sqrt(z.var(dim=0) + eps)
    variance = F.relu(1.0 - std).mean()
    if z.size(0) > 1:
        cov = (z.T @ z) / (z.size(0) - 1)
        off = cov - torch.diag(torch.diag(cov))
        covariance = (off ** 2).sum() / H
    else:
        covariance = queries.sum() * 0.0
    qn = F.normalize(queries, dim=-1)
    cos = qn @ qn.transpose(1, 2)
    eye = torch.eye(Q, device=queries.device, dtype=torch.bool)
    within = (cos[:, ~eye] ** 2).mean()
    return {"variance": variance, "covariance": covariance, "within": within}


@torch.no_grad()
def collapse_metrics(queries: Tensor) -> Dict[str, float]:
    """``query_cosine``: mean cosine between different queries of one graph
    (0.93 in Stage B). ``circuit_variance_share``: share of query variance
    that changes with the graph, for a fixed slot (6.7% after the old
    projector)."""
    B, Q, H = queries.shape
    qn = F.normalize(queries.float(), dim=-1)
    cos = qn @ qn.transpose(1, 2)
    eye = torch.eye(Q, device=queries.device, dtype=torch.bool)
    query_cosine = cos[:, ~eye].mean().item()
    flat = queries.float().reshape(B * Q, H)
    total = flat.var(dim=0, unbiased=False).sum()
    per_slot = queries.float().var(dim=0, unbiased=False).sum(dim=-1).mean()  # across graphs
    share = (per_slot / total.clamp(min=1e-12)).item() if B > 1 else 0.0
    return {"query_cosine": query_cosine, "circuit_variance_share": share}
