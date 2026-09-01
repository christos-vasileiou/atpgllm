from __future__ import annotations

"""
Stage 2 — Graph-Text-to-Text multimodal model.

Combines a (Stage-1-pretrained) graph stack with a causal language
model. The Q-Former output ``[B, Q, d_q]`` is linearly projected into
the LLM embedding space and prepended as *soft-prompt tokens* to the
tokenised prompt:

    inputs_embeds = [graph_tokens_0 … graph_tokens_{Q-1},
                     tok_embed(prompt_0), …, tok_embed(prompt_{T-1})]

Supervision: causal-LM loss on the *assistant answer* span only. The
graph-token positions and the user-prompt tokens are masked with ``-100``
so gradient only flows through answer tokens.

The graph stack (AttributeDecompositionEncoder + DAGGINEncoder +
GraphQFormer) can be frozen entirely, fine-tuned end-to-end, or
fine-tuned only partially; the projector itself is always trainable.

This module is intentionally LLM-agnostic: any HF causal LM that
supports ``inputs_embeds`` works (Qwen2.5, LLaMA, Mistral, ...).

Typical wiring::

    model = Stage2GraphTextLM.from_stage1(
        stage1_model,                  # loaded Stage1GraphTextModel
        llm_name="Qwen/Qwen2.5-7B-Instruct",
        freeze_graph=True,
        freeze_llm=False,              # usually wrap this with LoRA
    )

    batch = build_stage2_inputs(records, gate_funcs, vocab, tokenizer, ...)
    out = model(**batch)
    out.loss.backward()
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch

from .dataset import record_to_pyg, render_prompt_and_answer
from .gate_features import GateAttributeVocab
from .models_stage1 import Stage1GraphTextModel


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------


@dataclass
class Stage2Output:
    loss: Optional[Tensor]
    logits: Tensor
    inputs_embeds: Tensor


class Stage2GraphTextLM(nn.Module):
    """Graph + text → text model.

    Parameters
    ----------
    stage1 : Stage1GraphTextModel
        Graph encoder + Q-Former (text encoder is unused here).
    llm : transformers.PreTrainedModel
        A HF causal LM (must expose ``get_input_embeddings()`` and
        accept ``inputs_embeds``).
    num_queries : int
        ``stage1.q_former.num_queries`` — cached for convenience.
    freeze_graph : bool
        Freeze all parameters below the Q-Former output (graph encoder
        + Q-Former). The projector stays trainable.
    """

    def __init__(
        self,
        stage1: Stage1GraphTextModel,
        llm: nn.Module,
        freeze_graph: bool = True,
    ) -> None:
        super().__init__()
        self.stage1 = stage1
        self.llm = llm

        # Infer LLM embed dim from its input embedding table
        self.llm_embed_dim = self.llm.get_input_embeddings().weight.shape[1]
        self.num_queries = stage1.q_former.num_queries

        # Project Q-Former hidden -> LLM embed dim
        q_hidden = stage1.q_former.hidden_dim
        self.graph_to_llm = nn.Sequential(
            nn.Linear(q_hidden, self.llm_embed_dim),
            nn.GELU(),
            nn.Linear(self.llm_embed_dim, self.llm_embed_dim),
        )

        if freeze_graph:
            for p in self.stage1.graph_encoder.parameters():
                p.requires_grad = False
            for p in self.stage1.q_former.parameters():
                p.requires_grad = False

    # -----------------------------------------------------------------
    # Constructors
    # -----------------------------------------------------------------

    @classmethod
    def from_stage1(
        cls,
        stage1: Stage1GraphTextModel,
        llm,
        freeze_graph: bool = True,
    ) -> "Stage2GraphTextLM":
        return cls(stage1=stage1, llm=llm, freeze_graph=freeze_graph)

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def _graph_soft_prompt(self, g_batch) -> Tensor:
        """[B, Q, llm_embed_dim] — graph soft-prompt tokens."""
        _, _, query_embs = self.stage1.encode_graph(g_batch)
        return self.graph_to_llm(query_embs)

    def forward(
        self,
        g: Batch,
        input_ids: Tensor,              # [B, T] tokenised (prompt + answer)
        attention_mask: Tensor,         # [B, T]
        labels: Optional[Tensor] = None,  # [B, T] with -100 where loss masked
        return_inputs_embeds: bool = False,
    ) -> Stage2Output:
        tok_embeds = self.llm.get_input_embeddings()(input_ids)  # [B, T, H]
        graph_embeds = self._graph_soft_prompt(g)                 # [B, Q, H]

        B, Q, _ = graph_embeds.shape
        T = tok_embeds.size(1)

        inputs_embeds = torch.cat([graph_embeds, tok_embeds], dim=1)  # [B, Q+T, H]

        graph_mask = torch.ones(B, Q, dtype=attention_mask.dtype, device=attention_mask.device)
        full_mask = torch.cat([graph_mask, attention_mask], dim=1)   # [B, Q+T]

        full_labels = None
        if labels is not None:
            # -100 on graph-prompt positions so they never contribute to loss.
            prefix_labels = torch.full(
                (B, Q), fill_value=-100, dtype=labels.dtype, device=labels.device,
            )
            full_labels = torch.cat([prefix_labels, labels], dim=1)  # [B, Q+T]

        llm_out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            labels=full_labels,
        )
        return Stage2Output(
            loss=getattr(llm_out, "loss", None),
            logits=llm_out.logits,
            inputs_embeds=inputs_embeds if return_inputs_embeds else None,
        )

    # -----------------------------------------------------------------
    # Generation helper
    # -----------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        g: Batch,
        input_ids: Tensor,
        attention_mask: Tensor,
        **gen_kwargs,
    ) -> Tensor:
        tok_embeds = self.llm.get_input_embeddings()(input_ids)
        graph_embeds = self._graph_soft_prompt(g)
        inputs_embeds = torch.cat([graph_embeds, tok_embeds], dim=1)

        B, Q, _ = graph_embeds.shape
        graph_mask = torch.ones(B, Q, dtype=attention_mask.dtype, device=attention_mask.device)
        full_mask = torch.cat([graph_mask, attention_mask], dim=1)

        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            **gen_kwargs,
        )


# ---------------------------------------------------------------------
# Batch builder for Stage-2 SFT (prompt + answer)
# ---------------------------------------------------------------------


def _build_prompt_answer_ids(
    tokenizer,
    prompt: str,
    answer: str,
    max_len: int,
) -> tuple[List[int], int]:
    """Return ``(input_ids, answer_start_idx)``.

    ``answer_start_idx`` is the position in ``input_ids`` where the
    answer begins; everything before it is prompt / graph context and
    should be masked out of the loss.
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)

    # End with EOS so the model learns to stop.
    eos = tokenizer.eos_token_id
    if eos is not None:
        answer_ids = answer_ids + [eos]

    # Strategy: keep the answer; if it alone exceeds the budget, hard-truncate
    # it (keep the head — this preserves the INPUT_VECTOR / EXPECTED_OUTPUT
    # prefix in ASAP7 answers). Then trim the prompt from the left to fit.
    if len(answer_ids) > max_len:
        answer_ids = answer_ids[: max_len]
        prompt_ids = []
    else:
        budget = max_len - len(answer_ids)
        if len(prompt_ids) > budget:
            prompt_ids = prompt_ids[len(prompt_ids) - budget:]

    input_ids = prompt_ids + answer_ids
    answer_start = len(prompt_ids)
    return input_ids, answer_start


def _render_prompt_and_answer(
    record: Dict[str, Any],
    tokenizer,
    tests_dir: Optional[Path],
) -> tuple[str, str]:
    """Backward-compatible alias for the shared renderer."""
    return render_prompt_and_answer(record, tokenizer, tests_dir=tests_dir)


def build_stage2_inputs(
    records: List[Dict[str, Any]],
    gate_funcs: Dict[str, Dict[str, str]],
    vocab: GateAttributeVocab,
    tokenizer,
    max_seq_len: int = 2048,
    tests_dir: Optional[Path] = None,
    compact_netlist: bool = False,
) -> Dict[str, Any]:
    """Build a Stage-2 SFT batch from raw dataset records.

    Each record must expose ``netlist``, ``user_content``,
    ``answer_content`` (or equivalent) and friends.  Placeholder
    rendering is handled by
    :class:`atpgllm.training.conversation.ConversationExample`
    so the prompt and target match the main SFT pipeline exactly.

    Returns a dict compatible with :meth:`Stage2GraphTextLM.forward`.
    """
    graphs = []
    input_ids_list: List[List[int]] = []
    ans_starts: List[int] = []

    for rec in records:
        data = record_to_pyg(rec, gate_funcs, vocab)
        if data is None:
            continue

        prompt_text, answer_text = render_prompt_and_answer(
            rec,
            tokenizer,
            tests_dir=tests_dir,
            compact_netlist=compact_netlist,
        )

        ids, start = _build_prompt_answer_ids(
            tokenizer, prompt_text, answer_text, max_seq_len,
        )
        graphs.append(data)
        input_ids_list.append(ids)
        ans_starts.append(start)

    if not graphs:
        raise ValueError("No valid records: all netlists failed to parse.")

    # Right-pad to the same length.
    max_len = max(len(x) for x in input_ids_list)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    B = len(input_ids_list)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long)
    labels = torch.full((B, max_len), -100, dtype=torch.long)

    for i, (ids, start) in enumerate(zip(input_ids_list, ans_starts)):
        n = len(ids)
        input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, :n] = 1
        # Supervise only answer tokens.
        labels[i, start:n] = torch.tensor(ids[start:n], dtype=torch.long)

    g_batch = Batch.from_data_list(graphs)
    return {
        "g": g_batch,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
