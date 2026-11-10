"""Held-out probes of tool use for graph-conditioned SFT.

Two questions, each answered by running the model and simulating its tool
call with the same simulator the tool uses:

- ``first_try_detects``: from the prompt alone, does the model's first
  fault-simulation call carry a pattern that detects the fault?
- ``recovery_detects``: shown a real failed first call and the simulator's
  negative answer, does the model's next call detect the fault?

The recovery probe needs no natural failures: it starts from an injected
one, so it measures reading a negative result and repairing the pattern even
when the model rarely fails on its own.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch
from torch_geometric.data import Batch

from atpgllm.graph.dataset import record_to_pyg
from atpgllm.multimodal.data import (
    _compact_netlist_payload,
    _template_ids,
    collate_anchors,
    graph_position,
    port_anchors,
    ports_extra,
    template_turns,
)
from atpgllm.training.conversation import ConversationExample
from atpgllm.training.revert_template import parse_tool_call
from atpgllm.training.tool_feedback import read_snapshot
from atpgllm.training.tool_trajectories import simulate


@dataclass
class ProbeItem:
    record: Dict[str, Any]
    failed_attempt: Any


def _bits(vector: Any) -> Optional[Dict[str, Any]]:
    """The model's input_vector as a dict, or None when it cannot be read.

    A model early in training writes things like ``a:'0000'}``. That is a
    miss, the same as an empty vector, and must not end an evaluation (it
    ended the R5 smoke 504231 at its first probe)."""
    if isinstance(vector, str):
        try:
            vector = json.loads(vector)
        except json.JSONDecodeError:
            from fault_sim import convert_string_to_dict

            try:
                vector = convert_string_to_dict(vector, sep=":" if ":" in vector else "=")
            except (ValueError, TypeError, AttributeError):
                return None
    return vector if isinstance(vector, dict) else None


def _call_detects(record: Dict[str, Any], text: str) -> tuple[bool, bool]:
    """(the text has a parsable tool call, its pattern detects the fault)."""
    call = parse_tool_call(text)
    if not call or not isinstance(call.get("arguments"), dict):
        return False, False
    vector = _bits(call["arguments"].get("input_vector"))
    if not vector:
        return True, False
    try:
        snapshot = simulate(record, vector)
    except Exception:
        return True, False
    outputs = list(json.loads(record["expected_output"]))
    return True, read_snapshot(snapshot, record["fault"], outputs).detected


@torch.no_grad()
def run_tool_probe(
    model,
    tokenizer,
    items: List[ProbeItem],
    gate_funcs,
    vocab,
    *,
    device: torch.device,
    max_new_tokens: int = 1024,
    tool_view: str = "po_fault",
    netlist_view: str = "graph",
    port_bits: bool = False,
    hints: Optional[Dict[tuple, str]] = None,
) -> Dict[str, Any]:
    """Greedy decoding of one assistant turn per probe and question.
    ``port_bits`` and ``hints`` are the prompt additions the model trained
    with (:func:`atpgllm.multimodal.data.ports_extra`)."""
    was_training = model.training
    model.eval()
    stop = [
        token
        for token in (
            tokenizer.convert_tokens_to_ids("<|im_end|>"),
            tokenizer.eos_token_id,
        )
        if isinstance(token, int) and token >= 0
    ]
    counts = {name: 0 for name in (
        "first_try_called", "first_try_detects", "recovery_called", "recovery_detects",
    )}
    samples: List[Dict[str, str]] = []
    for item in items:
        record = item.record
        graph = record_to_pyg(record, gate_funcs, vocab)
        if graph is None:
            continue
        extra = ports_extra(record, graph, port_bits=port_bits, hints=hints)
        if extra is None:
            continue
        full, compact = _compact_netlist_payload(record, netlist_view, extra)
        conversation = ConversationExample.from_record(
            copy.deepcopy(record), use_tools=True,
            failed_attempt=item.failed_attempt, interpret_results=True,
            tool_view=tool_view,
        )
        messages = template_turns(conversation.messages, tokenizer)
        for message in messages:
            if message["role"] == "user":
                message["content"] = message["content"].replace(full, compact)
        prompt_end = next(i for i, m in enumerate(messages) if m["role"] == "assistant")
        # First try: system + user. Recovery: + the failed call and its result.
        for name, context in (("first_try", messages[:prompt_end]),
                              ("recovery", messages[: prompt_end + 2])):
            context_ids = _template_ids(tokenizer, context, generation_prompt=True)
            ids = torch.tensor([context_ids], device=device)
            graph_at = graph_position(tokenizer, context_ids)
            anchors = None
            if port_bits:
                anchors = collate_anchors([SimpleNamespace(
                    graph=graph, anchors=port_anchors(tokenizer, context_ids, graph_at, graph))])
            output = model.generate(
                Batch.from_data_list([graph]).to(device),
                ids,
                torch.ones_like(ids),
                graph_at=graph_at,
                anchors=anchors,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=stop,
                pad_token_id=tokenizer.pad_token_id,
            )
            text = tokenizer.decode(output[0], skip_special_tokens=True)
            called, detects = _call_detects(record, text)
            counts[f"{name}_called"] += called
            counts[f"{name}_detects"] += detects
            if len(samples) < 4:
                samples.append({"probe": name, "fault": record["fault"], "text": text[-1500:]})
    if was_training:
        model.train()
    probes = max(1, len(items))
    return {
        **{key: value / probes for key, value in counts.items()},
        "probes": len(items),
        "samples": samples,
    }
