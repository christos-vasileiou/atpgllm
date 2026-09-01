from __future__ import annotations

"""
Stage 1 trainer for the graph-text representation-learning model.

Loss objectives (BRIDGES / BLIP-2 inspired):

- **GTC** — Graph-Text Contrastive: align projected graph and text
  embeddings across the batch via InfoNCE.
- **GTM** — Graph-Text Matching: binary classifier fed with positive
  pairs (i, i) and in-batch negatives produced by a cyclic shift.
  Negatives come for free from the contrastive batch, no extra forward
  pass needed.
- **GTG** — Graph-Grounded Text Generation: a tiny autoregressive
  decoder conditioned on the pooled query embedding reconstructs the
  paired text.

Each loss is independently weighted (``LossWeights``). By default the
trainer uses **full bfloat16** weights and activations on CUDA when
``torch.cuda.is_bf16_supported()``; pass ``use_bf16=False`` for float32.
It also supports optional gradient accumulation (see
:meth:`Stage1Trainer.train_step`) and gradient clipping over the main
model plus the GTM / GTG heads — it does not wrap DDP/FSDP (callers do
that around the wrapped model).
"""

from dataclasses import dataclass
from itertools import chain
from typing import Any, Dict, Iterable, Optional

import torch
from torch import nn

from .losses_stage1 import (
    GraphGroundedTextGenLoss,
    GraphTextContrastiveLoss,
    GraphTextMatchingLoss,
)
from .models_stage1 import Stage1GraphTextModel, Stage1Outputs


@dataclass
class LossWeights:
    gtc: float = 1.0
    gtm: float = 1.0
    gtg: float = 1.0


@dataclass
class Stage1Losses:
    gtc: torch.Tensor
    gtm: torch.Tensor
    gtg: torch.Tensor

    @property
    def total(self) -> torch.Tensor:
        return self.gtc + self.gtm + self.gtg

    def as_dict(self) -> Dict[str, float]:
        return {
            "gtc": float(self.gtc.detach()),
            "gtm": float(self.gtm.detach()),
            "gtg": float(self.gtg.detach()),
        }


# ---------------------------------------------------------------------
# In-batch GTM negative construction
# ---------------------------------------------------------------------


def _build_gtm_batch(
    outputs: Stage1Outputs,
    model: Stage1GraphTextModel,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Produce (graph_repr, text_repr, labels) for GTM.

    Positives: (graph_i, text_i) for every i in the batch (label 1).
    Negatives: (graph_i, text_{i+1 mod B}) for every i (label 0).

    The graph side is kept *per-query* ``[2B, Q, D]`` so the matching
    head can apply its classifier to each of the Q queries and average
    the logits per pair (BRIDGES §V.A "matching score averaged across
    all queries").
    """
    g_q = model.projected_graph_repr_per_query(outputs)  # [B, Q, D]
    t = model.projected_text_repr(outputs)               # [B, D]
    B = g_q.size(0)

    t_shift = torch.roll(t, shifts=1, dims=0)            # [B, D]

    g_pair = torch.cat([g_q, g_q], dim=0)                # [2B, Q, D]
    t_pair = torch.cat([t, t_shift], dim=0)              # [2B, D]
    lab_dtype = g_q.dtype
    labels = torch.cat(
        [
            torch.ones(B, device=device, dtype=lab_dtype),
            torch.zeros(B, device=device, dtype=lab_dtype),
        ],
        dim=0,
    )
    return g_pair, t_pair, labels


def _resolve_param_dtype(device: torch.device, use_bf16: bool) -> torch.dtype:
    """Float dtype for model + GTM/GTG heads (full bf16 when supported)."""
    if not use_bf16:
        return torch.float32
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    is_cpu_bf16 = getattr(torch.cpu, "is_bf16_supported", None)
    if device.type == "cpu" and is_cpu_bf16 is not None and is_cpu_bf16():
        return torch.bfloat16
    return torch.float32


class Stage1Trainer:
    """Orchestrates the three Stage 1 losses for ``Stage1GraphTextModel``.

    The trainer does *not* own the optimiser or the data loader — those
    stay with the caller so that DDP, schedulers, logging, etc., remain
    the caller's responsibility.

    Usage::

        trainer = Stage1Trainer(
            model,
            vocab_size=len(tokenizer),
            gtg_max_seq_len=max_answer_len,
        )
        for batch in loader:
            losses = trainer.train_step(batch, optimizer)
            print(losses.as_dict())
    """

    def __init__(
        self,
        model: Stage1GraphTextModel,
        vocab_size: int,
        weights: Optional[LossWeights] = None,
        grad_clip: float = 1.0,
        device: torch.device | str = "cuda",
        *,
        gtg_max_seq_len: int = 4096,
        use_bf16: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.weights = weights or LossWeights()
        self.grad_clip = grad_clip

        self.param_dtype = _resolve_param_dtype(self.device, use_bf16)
        self.model = model.to(self.device)
        if self.param_dtype == torch.bfloat16:
            self.model = self.model.to(torch.bfloat16)

        self.gtc_loss = GraphTextContrastiveLoss().to(self.device)
        self.gtm_loss = GraphTextMatchingLoss(dim=model.proj_dim).to(
            self.device, dtype=self.param_dtype
        )
        self.gtg_loss = GraphGroundedTextGenLoss(
            vocab_size=vocab_size,
            d_model=model.proj_dim,
            max_seq_len=gtg_max_seq_len,
        ).to(self.device, dtype=self.param_dtype)

    def _clip_params(self) -> Iterable[nn.Parameter]:
        return chain(
            self.model.parameters(),
            self.gtm_loss.parameters(),
            self.gtg_loss.parameters(),
        )

    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device; cast graph floats to ``param_dtype`` when bf16."""
        g = batch["g"].to(self.device)
        if self.param_dtype != torch.float32:
            if getattr(g, "structural_feats", None) is not None:
                if g.structural_feats.is_floating_point():
                    g.structural_feats = g.structural_feats.to(self.param_dtype)
            if getattr(g, "fault_feats", None) is not None:
                if g.fault_feats.is_floating_point():
                    g.fault_feats = g.fault_feats.to(self.param_dtype)
            if getattr(g, "x", None) is not None and g.x.is_floating_point():
                g.x = g.x.to(self.param_dtype)
        return {
            "g": g,
            "input_ids": batch["input_ids"].to(self.device),
            "attention_mask": batch["attention_mask"].to(self.device),
            "decoder_input_ids": batch["decoder_input_ids"].to(self.device),
            "decoder_attention_mask": batch["decoder_attention_mask"].to(
                self.device
            ),
            "decoder_labels": batch["decoder_labels"].to(self.device),
        }

    # -----------------------------------------------------------------
    # Forward + backward on a single batch
    # -----------------------------------------------------------------

    def train_step(
        self,
        batch: Dict[str, Any],
        optimizer: torch.optim.Optimizer,
        *,
        accum_index: int = 0,
        grad_accum_steps: int = 1,
    ) -> Stage1Losses:
        """Forward, backward, and optionally optimizer step.

        For gradient accumulation, call once per micro-batch with
        ``accum_index`` in ``0 .. grad_accum_steps - 1`` (cycling). The
        weighted loss is divided by ``grad_accum_steps`` before
        ``backward`` so accumulated gradients match one large batch.

        **Contrastive caveat:** GTC (and GTM negatives) use in-batch
        pairs; that batch is still each *forward*'s physical batch, not
        ``per_device_batch * grad_accum_steps``. Accumulation matches the
        *optimizer* effective batch, not the number of InfoNCE
        negatives.
        """
        if grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be >= 1")
        if not (0 <= accum_index < grad_accum_steps):
            raise ValueError(
                "accum_index must satisfy 0 <= accum_index < grad_accum_steps"
            )

        self.model.train()
        self.gtm_loss.train()
        self.gtg_loss.train()
        if accum_index == 0:
            optimizer.zero_grad(set_to_none=True)

        losses = self.compute_losses(batch)
        total = (
            self.weights.gtc * losses.gtc
            + self.weights.gtm * losses.gtm
            + self.weights.gtg * losses.gtg
        ) / float(grad_accum_steps)

        is_last = accum_index == grad_accum_steps - 1

        total.backward()
        if is_last:
            nn.utils.clip_grad_norm_(self._clip_params(), self.grad_clip)
            optimizer.step()

        return losses

    @torch.no_grad()
    def eval_step(self, batch: Dict[str, Any]) -> Stage1Losses:
        self.model.eval()
        self.gtm_loss.eval()
        self.gtg_loss.eval()
        return self.compute_losses(batch)

    # -----------------------------------------------------------------
    # Compute losses only (no optimizer step). Shared by train/eval.
    # -----------------------------------------------------------------

    def compute_losses(self, batch: Dict[str, Any]) -> Stage1Losses:
        b = self._prepare_batch(batch)
        g = b["g"]
        input_ids = b["input_ids"]
        attention_mask = b["attention_mask"]
        outputs = self.model(g, input_ids, attention_mask)

        # --- GTC ---
        loss_gtc = self.gtc_loss(self.model, outputs)

        # --- GTM: positives + in-batch negatives ---
        g_pair, t_pair, labels = _build_gtm_batch(outputs, self.model, self.device)
        logits = self.gtm_loss.head(g_pair, t_pair)
        loss_gtm = nn.functional.binary_cross_entropy_with_logits(logits, labels)

        # --- GTG ---
        loss_gtg = self.gtg_loss(
            self.model,
            outputs,
            b["decoder_input_ids"],
            b["decoder_attention_mask"],
            b["decoder_labels"],
        )

        return Stage1Losses(gtc=loss_gtc, gtm=loss_gtm, gtg=loss_gtg)
