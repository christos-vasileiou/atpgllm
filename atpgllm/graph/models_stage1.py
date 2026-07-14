from __future__ import annotations

"""
Stage 1 graph-text representation-learning model (BRIDGES-style).

Pipeline
--------
Netlist (Verilog) --(netlist_parser + GateAttributeVocab)--> PyG Data with
    - ``gate_attrs``          [N, NUM_ATTRIBUTES]   (long)
    - ``structural_feats``    [N, NUM_STRUCTURAL_FEATURES]  (float)
    - ``edge_index``          [2, E]
    - ``batch``               [N]

AttributeDecompositionEncoder(gate_attrs, structural_feats) -> [N, d_node]
DAGGINEncoder(x, edge_index, batch) -> {node_embs [N, d_node],
                                        graph_embs [B, d_node * num_pools]}

Cross-modal projector (Q-Former, standalone pre-norm blocks):
    Q learnable queries (``num_queries`` x d_q) attend via cross-attention
    to the *per-node* graph embeddings h_v (not a single pooled token).
    This is strictly more expressive than pooled cross-attention and is
    what BLIP-2 / BRIDGES actually do for per-token modalities.

Text encoder: ``answerdotai/ModernBERT-base`` — the current SOTA
encoder-only transformer (RoPE, GeGLU, 8k context, local/global alternating
attention). We treat it as a black-box sentence encoder that outputs a
[B, d_text] vector from ``(input_ids, attention_mask)``.

Both projected graph repr and projected text repr live in a shared
``proj_dim``-D contrastive space.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torch_geometric.utils import to_dense_batch


# ----------------------------------------------------------------------
# Flash-Attention safety shim.
#
# ModernBERT's modeling file unconditionally imports from ``flash_attn``
# when ``transformers.utils.is_flash_attn_2_available()`` returns True
# — even though flash_attn's C extension may have a torch-ABI mismatch
# with the installed torch wheel, which raises ``ImportError`` at
# *module import* time, before any ``attn_implementation`` argument can
# help. We probe flash_attn once and, if broken, patch the detector so
# transformers falls back to SDPA / eager implementations.
# ----------------------------------------------------------------------


def _patch_broken_flash_attn() -> None:
    try:
        import flash_attn  # noqa: F401
        import flash_attn_2_cuda  # noqa: F401
        return  # flash_attn imports cleanly
    except Exception:
        pass

    try:
        import transformers.utils as _hf_utils
        _hf_utils.is_flash_attn_2_available = lambda: False  # type: ignore[assignment]
        import transformers.utils.import_utils as _hf_iu
        _hf_iu.is_flash_attn_2_available = lambda: False  # type: ignore[assignment]
    except Exception:
        pass


_patch_broken_flash_attn()

from transformers import AutoConfig, AutoModel  # noqa: E402

from .dag_gin import DAGGINEncoder
from .gate_features import GateAttributeVocab
from .netlist_parser import NUM_STRUCTURAL_FEATURES
from .node_encoder import AttributeDecompositionEncoder


DEFAULT_TEXT_MODEL = "answerdotai/ModernBERT-base"


# =====================================================================
# Standalone Q-Former block: self-attn + cross-attn (graph nodes as K,V)
# + FFN, pre-norm, GELU. Learnable query tokens are passed in from the
# outer module.
# =====================================================================


class QFormerBlock(nn.Module):
    """One pre-norm Q-Former block.

    - Self-attention over the Q query tokens (bidirectional).
    - Cross-attention with graph node embeddings as K, V (only if
      ``use_cross`` is True; corresponds to the "every alternate layer"
      cross-attention pattern from BLIP-2 / BRIDGES).
    - FFN (GEGLU variant via SiLU * Linear-gate, same as ModernBERT).

    Residuals + pre-norm on every sub-layer.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        use_cross: bool,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_cross = use_cross

        # --- Self-attention ---
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # --- Cross-attention (optional) ---
        if use_cross:
            self.cross_q_norm = nn.LayerNorm(hidden_dim)
            self.cross_kv_norm = nn.LayerNorm(hidden_dim)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )

        # --- FFN: GEGLU (gate * SiLU(x)) like ModernBERT / LLaMA ---
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn_gate = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.ffn_up = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.ffn_down = nn.Linear(ffn_dim, hidden_dim, bias=False)
        self.ffn_act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: Tensor,                     # [B, Q, H]
        graph_kv: Optional[Tensor] = None,   # [B, N_max, H]
        graph_kv_mask: Optional[Tensor] = None,  # [B, N_max] True = valid
    ) -> Tensor:
        # --- Self-attention ---
        h = self.self_norm(queries)
        h, _ = self.self_attn(h, h, h, need_weights=False)
        queries = queries + self.dropout(h)

        # --- Cross-attention on graph nodes ---
        if self.use_cross:
            assert graph_kv is not None, "cross-attn layer needs graph_kv"
            q = self.cross_q_norm(queries)
            kv = self.cross_kv_norm(graph_kv)
            # key_padding_mask: True = mask. to_dense_batch gives True = valid.
            key_padding_mask = None
            if graph_kv_mask is not None:
                key_padding_mask = ~graph_kv_mask
            h, _ = self.cross_attn(
                q, kv, kv,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            queries = queries + self.dropout(h)

        # --- FFN (GEGLU) ---
        h = self.ffn_norm(queries)
        h = self.ffn_down(self.ffn_act(self.ffn_gate(h)) * self.ffn_up(h))
        queries = queries + self.dropout(h)
        return queries


class GraphQFormer(nn.Module):
    """Standalone Q-Former that attends from learnable query tokens to
    per-node graph embeddings.

    Parameters
    ----------
    d_node : int
        Per-node embedding dimensionality (output of DAGGINEncoder).
    hidden_dim : int
        Q-Former hidden size.
    num_queries : int
        Number of learnable query tokens ``Q``.
    num_layers : int
        Number of transformer blocks.
    num_heads : int
        Attention heads.
    cross_attn_every_n : int
        Insert cross-attention on every n-th block (1-based, so 2 means
        on layers 1, 3, 5, …). Matches the BRIDGES / BLIP-2 recipe.
    """

    def __init__(
        self,
        d_node: int,
        hidden_dim: int = 512,
        num_queries: int = 32,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: Optional[int] = None,
        cross_attn_every_n: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        ffn_dim = ffn_dim or 4 * hidden_dim

        self.query_embeddings = nn.Parameter(
            torch.randn(1, num_queries, hidden_dim) * 0.02
        )

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

    def forward(
        self,
        node_embs: Tensor,     # [N_total, d_node]
        batch: Tensor,         # [N_total]
    ) -> Tensor:
        """Return per-batch query embeddings ``[B, Q, hidden_dim]``."""
        # Densify node embeddings: [B, N_max, hidden_dim] + mask [B, N_max]
        node_h = self.graph_in_norm(self.graph_in_proj(node_embs))
        node_dense, node_mask = to_dense_batch(node_h, batch)

        B = node_dense.size(0)
        q = self.query_embeddings.expand(B, -1, -1).contiguous()

        for block in self.blocks:
            q = block(q, graph_kv=node_dense, graph_kv_mask=node_mask)

        return self.out_norm(q)


# =====================================================================
# Graph encoder wrapper: AttributeDecompositionEncoder + DAGGINEncoder
# =====================================================================


class NetlistGraphEncoder(nn.Module):
    """Attribute decomposition + DAG-GIN, driven by PyG ``Data``.

    Returns both per-node embeddings (needed for node-level cross-attn)
    and a pooled graph embedding (useful for ablation / matching).
    """

    def __init__(
        self,
        vocab: GateAttributeVocab,
        node_dim: int = 256,
        gin_hidden_dim: int = 256,
        gin_num_layers: int = 6,
        per_attr_dim: int = 16,
        attr_mlp_hidden: int = 128,
        gin_num_pools: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attr_encoder = AttributeDecompositionEncoder.from_vocab(
            vocab,
            per_attr_dim=per_attr_dim,
            structural_dim=NUM_STRUCTURAL_FEATURES,
            hidden_dim=attr_mlp_hidden,
            out_dim=node_dim,
        )
        self.dag_gin = DAGGINEncoder(
            in_dim=node_dim,
            hidden_dim=gin_hidden_dim,
            num_layers=gin_num_layers,
            dropout=dropout,
            num_pools=gin_num_pools,
        )
        self.out_dim = gin_hidden_dim
        self.graph_dim = gin_hidden_dim * gin_num_pools

    def forward(self, data) -> dict[str, Tensor]:
        x = self.attr_encoder(data.gate_attrs, data.structural_feats)
        return self.dag_gin(x, data.edge_index, data.batch)


# =====================================================================
# Text encoder (ModernBERT)
# =====================================================================


class TextEncoder(nn.Module):
    """ModernBERT-based sentence encoder.

    Returns a single [CLS]-pooled vector per example.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_TEXT_MODEL,
        freeze: bool = False,
        attn_implementation: str = "sdpa",
    ) -> None:
        super().__init__()
        # ``attn_implementation="sdpa"`` sidesteps environments where
        # flash_attn has a torch-ABI mismatch (the default for ModernBERT
        # in newer transformers tries flash_attn). "eager" is the
        # safest fallback.
        self.model = AutoModel.from_pretrained(
            model_name,
            attn_implementation=attn_implementation,
        )
        self.hidden_size = AutoConfig.from_pretrained(model_name).hidden_size
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pool over non-pad tokens for robustness (works for any
        # encoder, not only those with a [CLS] token).
        last = out.last_hidden_state  # [B, T, H]
        mask = attention_mask.unsqueeze(-1).to(last.dtype)
        pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled  # [B, H]


# =====================================================================
# Stage 1 end-to-end model
# =====================================================================


@dataclass
class Stage1Outputs:
    node_embs: Tensor         # [N_total, d_node]
    graph_embs: Tensor        # [B, d_node * num_pools]
    query_embs: Tensor        # [B, Q, d_q]
    text_embs: Tensor         # [B, d_text]


class Stage1GraphTextModel(nn.Module):
    """Full Stage 1 model = graph encoder + Q-Former + text encoder.

    Parameters
    ----------
    vocab : GateAttributeVocab
        Vocabulary for attribute decomposition (required; controls the
        dimensionality of all embedding tables in the node encoder).
    text_model_name : str
        HF id of the text encoder. Defaults to ModernBERT-base.
    proj_dim : int
        Dimensionality of the shared contrastive / matching space.
    freeze_text : bool
        If True, the ModernBERT weights are frozen (only projectors
        and Q-Former are trained).
    """

    def __init__(
        self,
        vocab: GateAttributeVocab,
        text_model_name: str = DEFAULT_TEXT_MODEL,
        node_dim: int = 256,
        gin_hidden_dim: int = 256,
        gin_num_layers: int = 6,
        qformer_hidden_dim: int = 512,
        qformer_layers: int = 6,
        qformer_heads: int = 8,
        qformer_cross_every_n: int = 2,
        num_queries: int = 32,
        proj_dim: int = 512,
        freeze_text: bool = False,
        text_attn_implementation: str = "sdpa",
    ) -> None:
        super().__init__()

        self.graph_encoder = NetlistGraphEncoder(
            vocab=vocab,
            node_dim=node_dim,
            gin_hidden_dim=gin_hidden_dim,
            gin_num_layers=gin_num_layers,
        )

        self.q_former = GraphQFormer(
            d_node=self.graph_encoder.out_dim,
            hidden_dim=qformer_hidden_dim,
            num_queries=num_queries,
            num_layers=qformer_layers,
            num_heads=qformer_heads,
            cross_attn_every_n=qformer_cross_every_n,
        )

        self.text_encoder = TextEncoder(
            text_model_name,
            freeze=freeze_text,
            attn_implementation=text_attn_implementation,
        )

        self.graph_proj = nn.Linear(qformer_hidden_dim, proj_dim)
        self.text_proj = nn.Linear(self.text_encoder.hidden_size, proj_dim)

        self.proj_dim = proj_dim

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def encode_graph(self, data) -> tuple[Tensor, Tensor, Tensor]:
        g_out = self.graph_encoder(data)
        node_embs = g_out["node_embs"]     # [N_total, d_node]
        graph_embs = g_out["graph_embs"]   # [B, d_node * pools]
        query_embs = self.q_former(node_embs, data.batch)  # [B, Q, d_q]
        return node_embs, graph_embs, query_embs

    def encode_text(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        return self.text_encoder(input_ids, attention_mask)

    def forward(
        self,
        data,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Stage1Outputs:
        node_embs, graph_embs, query_embs = self.encode_graph(data)
        text_embs = self.encode_text(input_ids, attention_mask)
        return Stage1Outputs(
            node_embs=node_embs,
            graph_embs=graph_embs,
            query_embs=query_embs,
            text_embs=text_embs,
        )

    # -----------------------------------------------------------------
    # Projections into the shared contrastive space
    # -----------------------------------------------------------------

    def projected_graph_repr_per_query(self, outputs: Stage1Outputs) -> Tensor:
        """Per-query graph projections ``[B, Q, proj_dim]``.

        BLIP-2 / BRIDGES use these directly: GTC computes a Q-by-B
        similarity matrix and reduces by ``max_q``; GTM applies the
        binary classifier per-query and averages logits. Pooling Q
        before projection — the previous behaviour — collapsed all
        queries into one vector, defeating the purpose of having Q
        learnable queries.
        """
        return self.graph_proj(outputs.query_embs)   # [B, Q, proj_dim]

    def projected_graph_repr(self, outputs: Stage1Outputs) -> Tensor:
        """Mean-pool the per-query projections to ``[B, proj_dim]``.

        Used as the conditioning vector for GTG (a single vector per
        graph). Mean is a stable pooling for the soft-prompt context;
        max would amplify whichever query happens to fire strongly.
        """
        return self.projected_graph_repr_per_query(outputs).mean(dim=1)

    def projected_text_repr(self, outputs: Stage1Outputs) -> Tensor:
        return self.text_proj(outputs.text_embs)     # [B, proj_dim]
