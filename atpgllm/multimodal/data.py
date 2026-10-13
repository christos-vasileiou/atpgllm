"""Shared SFT/GRPO batches for graph-conditioned ATPG."""

from __future__ import annotations

import collections
import copy
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
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


# ---------------------------------------------------------------------
# Tool-calling SFT (chat template, per-message loss masks)
# ---------------------------------------------------------------------

GRAPH_CONTEXT = "<GRAPH_CONTEXT>"


def _compact_netlist_payload(record: Dict[str, Any]) -> tuple[str, str]:
    """The netlist payload as the prompt prints it, and its graph stand-in;
    the ``doc_id`` stays so tool calls can name the circuit."""
    netlist = record.get("netlist")
    if isinstance(netlist, dict):
        netlist = netlist.get("netlist")
    netlist = netlist or ""
    doc_id = hashlib.sha256(netlist.encode("utf-8")).hexdigest()[:16]
    return (
        str({"doc_id": doc_id, "netlist": netlist}),
        str({"doc_id": doc_id, "netlist": GRAPH_CONTEXT}),
    )


def _template_ids(tokenizer, messages, *, generation_prompt: bool = False) -> List[int]:
    from atpgllm.training.tools import TOOLS

    ids = tokenizer.apply_chat_template(
        messages,
        tools=TOOLS,
        tokenize=True,
        add_generation_prompt=generation_prompt,
    )
    return list(ids["input_ids"] if isinstance(ids, dict) else ids)


_THINK_OPEN, _THINK_CLOSE = "<think>", "</think>"


def template_opens_thinking(tokenizer) -> bool:
    """Whether the chat template's generation prompt opens ``<think>`` (Granite
    4.2: ``<|im_start|>assistant\n<think>\n``), so the model writes each turn
    from inside a thinking block."""
    cached = getattr(tokenizer, "_atpg_opens_thinking", None)
    if cached is None:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "u"}], tokenize=False, add_generation_prompt=True
        )
        cached = prompt.rstrip().endswith(_THINK_OPEN)
        tokenizer._atpg_opens_thinking = cached
    return cached


def template_turns(messages: Sequence[Dict[str, Any]], tokenizer) -> List[Dict[str, Any]]:
    """The conversation with each assistant turn written as the template's
    generation prompt begins it.

    When the template opens thinking, a turn becomes ``<think>\n`` reasoning
    ``\n</think>\n`` text (Granite 4.2's own layout); a turn without reasoning
    (the final answer) gets an empty block. The opening ``<think>\n`` is then
    part of the prompt in SFT and in rollouts alike, and the model learns to
    write the rest. Other templates get the messages unchanged.
    """
    messages = [dict(message) for message in messages]
    if not template_opens_thinking(tokenizer):
        return messages
    for message in messages:
        if message["role"] != "assistant":
            continue
        content = message.get("content") or ""
        if content.startswith(_THINK_OPEN) and _THINK_CLOSE in content:
            reasoning, rest = content[len(_THINK_OPEN):].split(_THINK_CLOSE, 1)
            reasoning = reasoning.strip("\n")
        else:
            reasoning, rest = "", content
        message["content"] = (
            _THINK_OPEN + "\n" + (reasoning + "\n" if reasoning else "")
            + _THINK_CLOSE + "\n" + rest.lstrip("\n")
        )
    return messages


def tokenize_tool_conversation(
    messages: Sequence[Dict[str, Any]],
    trainable: Optional[Sequence[bool]],
    tokenizer,
) -> tuple[List[int], List[int]]:
    """Token ids of the whole conversation and labels that keep only the
    trainable assistant turns (``trainable=None``: every assistant turn).

    Each turn's span is found by rendering the conversation up to it, so the
    labels cover exactly what the model writes after the template's
    generation prompt. Raises ValueError if the chat template is not
    prefix-preserving, which would misplace those spans.
    """
    # Tool-call arguments stay dicts: Granite 4.2's template iterates them
    # into XML <parameter=...> blocks (and Qwen2.5's writes a JSON object);
    # a pre-serialized string would be written again as an escaped string.
    messages = copy.deepcopy(list(messages))
    ids = _template_ids(tokenizer, messages)
    labels = [-100] * len(ids)
    for index, message in enumerate(messages):
        target = (
            message["role"] == "assistant" if trainable is None else trainable[index]
        )
        if not target:
            continue
        before = _template_ids(tokenizer, messages[:index], generation_prompt=True)
        through = _template_ids(tokenizer, messages[: index + 1])
        if ids[: len(before)] != before or ids[: len(through)] != through:
            raise ValueError("The chat template is not prefix-preserving.")
        labels[len(before): len(through)] = ids[len(before): len(through)]
    return ids, labels


@dataclass
class ToolSFTExample:
    graph: Any
    input_ids: List[int]
    labels: List[int]
    retry: bool


def build_tool_sft_example(
    record: Dict[str, Any],
    gate_funcs: Dict[str, Dict[str, str]],
    vocab: GateAttributeVocab,
    tokenizer,
    *,
    max_seq_len: int,
    failed_attempt: Any = None,
    rejections: Optional[collections.Counter] = None,
    tool_view: str = "po_fault",
) -> Optional[ToolSFTExample]:
    """One graph-conditioned tool-calling SFT example, or None when the record
    is rejected, its snapshot does not detect the fault, or it is longer than
    ``max_seq_len`` (dropped whole: truncation could cut through a tool turn).

    The prompt's netlist is replaced by :data:`GRAPH_CONTEXT`; the graph
    reaches the model as the soft prefix instead. ``tool_view`` sets what tool
    messages show (``atpgllm.training.tool_feedback.TOOL_VIEWS``); RL rollouts
    of the model must use the same view.
    """
    from atpgllm.training.conversation import ConversationExample

    graph = record_to_pyg(record, gate_funcs, vocab, rejections=rejections)
    if graph is None:
        return None
    full, compact = _compact_netlist_payload(record)
    try:
        conversation = ConversationExample.from_record(
            copy.deepcopy(record),
            use_tools=True,
            failed_attempt=failed_attempt,
            interpret_results=True,
            tool_view=tool_view,
        )
    except ValueError:
        if rejections is not None:
            rejections["snapshot_not_detecting"] += 1
        return None
    messages = template_turns(conversation.messages, tokenizer)
    for message in messages:
        if message["role"] == "user":
            message["content"] = message["content"].replace(full, compact)
    ids, labels = tokenize_tool_conversation(messages, conversation.trainable, tokenizer)
    if len(ids) > max_seq_len:
        if rejections is not None:
            rejections["too_long_retry" if failed_attempt else "too_long"] += 1
        return None
    return ToolSFTExample(
        graph=graph, input_ids=ids, labels=labels, retry=failed_attempt is not None
    )


def collate_tool_sft(examples: Sequence[ToolSFTExample], pad_token_id: int) -> Dict[str, Any]:
    """Right-padded batch for :class:`GraphConditionedCausalLM.forward`."""
    width = max(len(example.input_ids) for example in examples)
    input_ids = torch.full((len(examples), width), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), width), dtype=torch.long)
    labels = torch.full((len(examples), width), -100, dtype=torch.long)
    for row, example in enumerate(examples):
        n = len(example.input_ids)
        input_ids[row, :n] = torch.tensor(example.input_ids)
        attention_mask[row, :n] = 1
        labels[row, :n] = torch.tensor(example.labels)
    return {
        "graph": Batch.from_data_list([example.graph for example in examples]),
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
