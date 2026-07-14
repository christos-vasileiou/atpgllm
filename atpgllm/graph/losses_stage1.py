from __future__ import annotations

"""
Stage 1 cross-modal losses for BRIDGES-style pre-training:

- Graph–Text Contrastive Learning (GTC)
- Graph–Text Matching (GTM)
- Graph-Grounded Text Generation (GTG, simplified)

These are implemented on top of the ``Stage1GraphTextModel`` outputs.
"""

from typing import Tuple

import torch
from torch import nn

from .models_stage1 import Stage1Outputs, Stage1GraphTextModel


def _l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


class GraphTextContrastiveLoss(nn.Module):
    """
    InfoNCE-style contrastive loss between graph and text
    representations in a shared space.

    Following BRIDGES (§V.A) / BLIP-2: the graph side keeps its Q query
    tokens, and the alignment score for a (graph_i, text_j) pair is the
    *maximum* cosine similarity over the Q queries. This lets different
    queries specialise on different aspects of the graph and is the
    reason for having Q > 1 in the first place.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        model: Stage1GraphTextModel,
        outputs: Stage1Outputs,
    ) -> torch.Tensor:
        g_q = model.projected_graph_repr_per_query(outputs)  # [B, Q, D]
        t = model.projected_text_repr(outputs)               # [B, D]

        g_q = _l2_normalize(g_q, dim=-1)
        t = _l2_normalize(t, dim=-1)

        # Per-query, per-pair similarity: sim[i, q, j] = g_q[i, q] . t[j]
        sim = torch.einsum("iqd,jd->iqj", g_q, t)            # [B, Q, B]
        # Reduce over queries: keep the best-aligned query per pair.
        sim, _ = sim.max(dim=1)                              # [B, B]
        logits = sim / self.temperature

        labels = torch.arange(logits.size(0), device=logits.device)
        loss_g2t = nn.functional.cross_entropy(logits, labels)
        loss_t2g = nn.functional.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_g2t + loss_t2g)


class GraphTextMatchingHead(nn.Module):
    """Binary classifier for graph-text matching (GTM).

    Two input shapes are supported:

    - ``g_repr`` ``[N, D]``, ``t_repr`` ``[N, D]`` (legacy / pooled): one
      logit per pair.
    - ``g_repr`` ``[N, Q, D]``, ``t_repr`` ``[N, D]`` (BRIDGES path):
      classifier is applied per query and the *logits* are averaged over
      Q to yield one matching score per pair, matching the paper's
      "matching score averaged across all queries" rule.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )

    def forward(
        self,
        g_repr: torch.Tensor,
        t_repr: torch.Tensor,
    ) -> torch.Tensor:
        if g_repr.dim() == 3:
            # [N, Q, D] graph + [N, D] text -> [N] logit (mean over Q)
            t_expanded = t_repr.unsqueeze(1).expand(-1, g_repr.size(1), -1)
            x = torch.cat([g_repr, t_expanded], dim=-1)        # [N, Q, 2D]
            per_q_logits = self.classifier(x).squeeze(-1)       # [N, Q]
            return per_q_logits.mean(dim=1)                     # [N]
        # Legacy 2D path
        x = torch.cat([g_repr, t_repr], dim=-1)
        return self.classifier(x).squeeze(-1)  # [N]


class GraphTextMatchingLoss(nn.Module):
    """
    Graph–Text Matching (GTM): predict whether a graph–text pair
    is matched or not.

    This implementation assumes the caller constructs a batch with
    both positive and negative pairs and provides binary labels.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.head = GraphTextMatchingHead(dim)

    def forward(
        self,
        model: Stage1GraphTextModel,
        outputs: Stage1Outputs,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args
        ----
        labels:
            [B] tensor with 1 for matched pairs, 0 for unmatched.
        """
        g = model.projected_graph_repr(outputs)
        t = model.projected_text_repr(outputs)
        logits = self.head(g, t)  # [B]
        return nn.functional.binary_cross_entropy_with_logits(logits, labels.float())


class GraphGroundedTextGenerator(nn.Module):
    """
    Lightweight decoder for Graph-Grounded Text Generation (GTG).

    This module treats the pooled query embeddings as a single
    conditioning vector per example and trains a small Transformer
    decoder to generate text tokens autoregressively (teacher
    forcing) conditioned on that vector.

    This is intentionally lightweight and separate from the large
    LLM used in Stage 2.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        num_layers: int = 4,
        max_len: int = 4096,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Project graph-side conditioning vector into decoder space
        self.cond_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        cond: torch.Tensor,        # [B, D] conditioning vector from queries
        input_ids: torch.Tensor,   # [B, T] teacher-forced tokens (shifted right)
        attention_mask: torch.Tensor,  # [B, T], 1 = keep, 0 = pad
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = input_ids.shape
        device = input_ids.device
        if T > self.pos_emb.num_embeddings:
            raise ValueError(
                f"GTG sequence length {T} exceeds max_len={self.pos_emb.num_embeddings} "
                "(increase max_len / gtg_max_seq_len to match tokenizer max_length)."
            )

        # Decoder input embeddings
        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        tgt = self.token_emb(input_ids) + self.pos_emb(pos)  # [B, T, D]

        # Conditioning as a "memory" sequence of length 1
        memory = self.cond_proj(cond).unsqueeze(1)  # [B, 1, D]
        
        # Causal mask for autoregressive decoding
        causal_mask = torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
        )

        key_padding_mask = attention_mask == 0  # [B, T]
        decoded = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=key_padding_mask,
        )  # [B, T, D]

        logits = self.lm_head(decoded)  # [B, T, V]
        return logits, decoded


class GraphGroundedTextGenLoss(nn.Module):
    """
    Cross-entropy loss for Graph-Grounded Text Generation (GTG).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        *,
        max_seq_len: int = 4096,
    ) -> None:
        super().__init__()
        self.decoder = GraphGroundedTextGenerator(
            vocab_size=vocab_size,
            d_model=d_model,
            max_len=max_seq_len,
        )

    def forward(
        self,
        model: Stage1GraphTextModel,
        outputs: Stage1Outputs,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args
        ----
        input_ids:
            [B, T] decoder inputs (typically labels shifted right with BOS).
        attention_mask:
            [B, T] 1 = keep, 0 = pad.
        labels:
            [B, T] target token IDs, with -100 for positions to ignore.
        """
        # Condition on projected graph repr (same space as contrastive; d_model = proj_dim)
        cond = model.projected_graph_repr(outputs)  # [B, proj_dim]

        logits, _ = self.decoder(cond, input_ids, attention_mask)  # [B, T, V]

        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
        return loss

