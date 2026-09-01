"""Shared SFT/GRPO batches for graph-conditioned ATPG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from torch import Tensor
from torch_geometric.data import Batch

from atpgllm.graph.dataset import record_to_pyg, render_prompt_and_answer
from atpgllm.graph.gate_features import GateAttributeVocab
from atpgllm.graph.stage2_model import build_stage2_inputs


def build_multimodal_sft_batch(
    records: List[Dict[str, Any]],
    gate_funcs: Dict[str, Dict[str, str]],
    vocab: GateAttributeVocab,
    tokenizer,
    *,
    max_seq_len: int,
) -> Dict[str, Any]:
    """Build answer-only SFT labels while replacing netlist text by graph context."""
    batch = build_stage2_inputs(
        records,
        gate_funcs,
        vocab,
        tokenizer,
        max_seq_len=max_seq_len,
        compact_netlist=True,
    )
    batch["graph"] = batch.pop("g")
    return batch


@dataclass
class GraphPromptExample:
    graph: Batch
    prompt: str
    prompt_ids: Tensor
    prompt_mask: Tensor
    record: Dict[str, Any]


def build_graph_prompt_example(
    record: Dict[str, Any],
    gate_funcs: Dict[str, Dict[str, str]],
    vocab: GateAttributeVocab,
    tokenizer,
    *,
    max_prompt_length: int,
) -> Optional[GraphPromptExample]:
    """Build one GRPO prompt and preserve all verifier fields in ``record``."""
    graph = record_to_pyg(record, gate_funcs, vocab)
    if graph is None:
        return None
    prompt, _ = render_prompt_and_answer(
        record,
        tokenizer,
        compact_netlist=True,
    )
    tokenized = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )
    if tokenized["input_ids"].size(1) > max_prompt_length:
        return None
    return GraphPromptExample(
        graph=Batch.from_data_list([graph]),
        prompt=prompt,
        prompt_ids=tokenized["input_ids"],
        prompt_mask=tokenized["attention_mask"],
        record=record,
    )
