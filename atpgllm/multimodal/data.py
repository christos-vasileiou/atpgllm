"""Shared SFT/GRPO batches for graph-conditioned ATPG."""

from __future__ import annotations

import collections
import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
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
# What the prompt keeps of the netlist as text: nothing (``graph``), or the
# module header and its input/output declarations (``ports``), so the model
# reads the names it must write and the graph carries the cells.
NETLIST_VIEWS = ("graph", "ports")


# What the assistant turns train on: everything (``full``), or with the
# ``<think>`` reasoning removed (``answer``), so the turns keep only the tool
# call with its vector and the final INPUT_VECTOR and EXPECTED_OUTPUT block.
# With the ports view the reasoning names internal gates the prompt no longer
# shows. The BRIDGES re-test trains on ``answer``.
SFT_TARGETS = ("full", "answer")
_THINK = re.compile(r"<think>.*?</think>\s*", re.S)


def drop_reasoning(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Copies of ``messages`` with ``<think>`` blocks removed from assistant turns."""
    out = []
    for message in messages:
        message = dict(message)
        if message.get("role") == "assistant" and isinstance(message.get("content"), str):
            message["content"] = _THINK.sub("", message["content"]).lstrip()
        out.append(message)
    return out


def port_declarations(netlist: str, drop: Sequence[str] = ()) -> str:
    """The module header and input/output declarations of a netlist, without
    the ports named in ``drop`` (see :func:`scan_procedure_ports`)."""
    statements = (s.strip() for s in netlist.split(";"))
    kept = [s for s in statements if re.match(r"(module|input|output)\b", s)]
    if not drop:
        return "".join(f"{s};\n" for s in kept)
    gone = set(drop)
    out = []
    for s in kept:
        header = re.match(r"(module\s+\S+\s*\()(.*)\)\s*$", s, re.S)
        decl = re.match(r"((?:input|output)\s*(?:\[[^\]]+\]\s*)?)(.*)$", s, re.S)
        head, names = (header or decl).groups()
        names = [n.strip() for n in names.split(",") if n.strip() and n.strip() not in gone]
        if header:
            out.append(f"{head} {', '.join(names)} );\n")
        elif names:
            out.append(f"{head.rstrip()} {', '.join(names)};\n")
    return "".join(out)


def scan_procedure_ports(netlist: str) -> List[str]:
    """The ports of a full-scan netlist that a test never sets or reads: the
    clock and the scan ports (the scan procedure drives them). Empty for a
    combinational netlist. The ports view leaves them out, so every declared
    port is one the answer may name."""
    from atpgllm.graph.netlist_parser import _scan_view

    try:
        scan_view = _scan_view()
    except ValueError:
        return []
    if not scan_view.is_scan_netlist(netlist):
        return []
    try:
        _, clocks = scan_view.scan_cells_from_netlist(netlist)
    except ValueError:      # a scan netlist the graph rejects as well
        return []
    return list(scan_view.SCAN_PORTS) + [re.sub(r"\[\d+\]$", "", clock) for clock in clocks]


# Lines the ports view can add after the graph slot. Bus ports are declared
# as ``input [3:0] a;``, so the declarations never spell a bit such as
# ``a[3]``, the name the model must write.
#   PORT_BITS  every input bit by name, in the order of the answer's input
#              vector: primary inputs, then for a scan circuit each flop's QN
#              net in chain order (most are not ports, so the declarations
#              do not name them). Anchored fusion adds each bit's graph
#              vector to the tokens of its name here.
#   HINT       the graph model's decoded suggestion as plain text (the hint
#              arm): the inputs its care head marks, with the value head's
#              values, for example ``a[3]=1 sel=1``.
# The ``rich`` hint style adds how sure the graph model is and where it expects
# the fault to show (:func:`rich_hint_text`):
#   // suggested inputs: a[3]=1 sel=0?
#   // possibly needed: b=1 c=0?
#   // observe at: y n10
PORT_BITS = "// input bits:"
PORT_BITS_END = " ;"
HINT = "// suggested inputs:"
HINT_MAYBE = "// possibly needed:"
HINT_OBSERVE = "// observe at:"
HINT_STYLES = ("plain", "rich")
# Rich hint thresholds, from the calibration of 2026-11-12 on v5 validation.
# Care probability at or above RICH_SURE: 64% to 95% of those inputs are set
# in most of the dataset's patterns; between RICH_MAYBE and RICH_SURE, 9% to
# 35%. A value within RICH_VALUE_MARGIN of 0.5 is right 57% to 74% of the
# time and gets a question mark; a surer one is right 85% to 98%.
RICH_SURE = 0.8
RICH_MAYBE = 0.35
RICH_VALUE_MARGIN = 0.3
RICH_MAYBE_CAP = 12
RICH_OBSERVE_CAP = 6


def netlist_key(record: Dict[str, Any]) -> str:
    """The prompt's ``doc_id``, and the key of the re-test labels and hints."""
    netlist = record.get("netlist")
    if isinstance(netlist, dict):
        netlist = netlist.get("netlist")
    return hashlib.sha256((netlist or "").encode("utf-8")).hexdigest()[:16]


def hint_text(vector: Dict[str, Any]) -> str:
    """``a[3]=1 sel=1`` for the chosen inputs of ``vector``; ``none`` when it chooses none."""
    chosen = [f"{name}={value}" for name, value in vector.items() if str(value).lower() != "x"]
    return " ".join(chosen) or "none"


def rich_hint_text(rich: Dict[str, Any]) -> str:
    """The rich hint of one problem, as the text after :data:`HINT`.

    ``rich`` is a row's ``rich`` field from ``retest_hints``. Inputs the
    graph model is sure need a value go on the first line; a ``?`` after a
    value says the value itself is close to a coin flip. Inputs it is less
    sure about follow on :data:`HINT_MAYBE`, most likely first. The outputs
    where it expects the fault effect go on :data:`HINT_OBSERVE`; when it is
    sure of none, the most likely one is listed with a ``?``."""
    def assignment(name: str, one: float) -> str:
        return f"{name}={int(one >= 0.5)}" + ("?" if abs(one - 0.5) < RICH_VALUE_MARGIN else "")

    inputs = rich.get("inputs") or {}
    sure = [assignment(name, one) for name, (care, one) in inputs.items() if care >= RICH_SURE]
    maybe = sorted(((care, name, one) for name, (care, one) in inputs.items()
                    if RICH_MAYBE <= care < RICH_SURE), key=lambda item: -item[0])[:RICH_MAYBE_CAP]
    outputs = rich.get("outputs") or {}
    seen = [name for name, p in outputs.items() if p >= RICH_SURE][:RICH_OBSERVE_CAP]
    if not seen and outputs:
        seen = [max(outputs, key=outputs.get) + "?"]
    lines = [" ".join(sure) or "none"]
    if maybe:
        lines.append(f"{HINT_MAYBE} " + " ".join(assignment(name, one) for _, name, one in maybe))
    if seen:
        lines.append(f"{HINT_OBSERVE} " + " ".join(seen))
    return "\n".join(lines)


def load_hints(path, style: str = "plain") -> Dict[tuple, str]:
    """``(netlist key, fault) -> hint text`` from the JSONL files that
    ``atpgllm.graph.scripts.retest_hints`` writes (one file or a directory).
    ``style="rich"`` needs rows written with their ``rich`` field."""
    if style not in HINT_STYLES:
        raise ValueError(f"hint style must be one of {HINT_STYLES}")
    path = Path(path)
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"no hint files under {path}")
    hints: Dict[tuple, str] = {}
    for file in files:
        with file.open() as fh:
            for line in fh:
                row = json.loads(line)
                if style == "rich":
                    if "rich" not in row:
                        raise ValueError(f"{file} has no rich hints; build it again with retest_hints")
                    text = rich_hint_text(row["rich"])
                else:
                    text = hint_text(row["hint"])
                hints[(row["netlist_key"], row["fault"])] = text
    return hints


def ports_extra(record: Dict[str, Any], graph, *, port_bits: bool = False,
                hints: Optional[Dict[tuple, str]] = None) -> Optional[str]:
    """The lines to add after the graph slot, or None when a hint is asked
    for and the record's problem has none."""
    extra = ""
    if port_bits:
        names = [name for name, is_pi in zip(graph.node_names, graph.pi_node_mask.tolist()) if is_pi]
        extra += f"{PORT_BITS} {' '.join(names)}{PORT_BITS_END}\n"
    if hints is not None:
        hint = hints.get((netlist_key(record), str(record["fault"]).strip()))
        if hint is None:
            return None
        extra += f"{HINT} {hint}\n"
    return extra


def port_anchors(tokenizer, ids: Sequence[int], graph_at: int, graph) -> List[tuple]:
    """``(token index, node index, weight)`` for the tokens that spell each
    input bit in the :data:`PORT_BITS` line, which follows the graph slot.

    The weights of one name's tokens sum to 1. The line is read back from
    the tokens themselves (decoded one by one from ``graph_at``), so the map
    holds however the chat template escapes the payload."""
    from atpgllm.multimodal.anchored import token_net_map

    start = max(0, graph_at - 1)
    text, offsets, found = "", [], -1
    for token in ids[start:]:
        piece = tokenizer.decode([token], clean_up_tokenization_spaces=False)
        offsets.append((len(text), len(text) + len(piece)))
        text += piece
        found = text.find(PORT_BITS)
        if found >= 0 and text.find(PORT_BITS_END, found + len(PORT_BITS)) >= 0:
            break
    else:
        raise ValueError("The prompt has no input-bits line after the graph slot.")
    end = text.find(PORT_BITS_END, found + len(PORT_BITS))
    spans, at = [], found + len(PORT_BITS)
    for node, is_pi in enumerate(graph.pi_node_mask.tolist()):
        if not is_pi:
            continue
        name = graph.node_names[node]
        where = text.find(name, at, end)
        if where < 0:        # a name the template escaped: no anchor
            continue
        spans.append((where, where + len(name), node))
        at = where + len(name)
    return [(start + token, node, weight) for token, node, weight in token_net_map(offsets, spans)]


def _compact_netlist_payload(record: Dict[str, Any], view: str = "graph",
                             extra: str = "") -> tuple[str, str]:
    """The netlist payload as the prompt prints it, and its graph stand-in;
    the ``doc_id`` stays so tool calls can name the circuit. ``extra``
    (:func:`ports_extra`) goes after the graph slot of the ports view."""
    if view not in NETLIST_VIEWS:
        raise ValueError(f"netlist view must be one of {NETLIST_VIEWS}")
    if extra and view != "ports":
        raise ValueError("Input-bit and hint lines need --netlist-view ports.")
    netlist = record.get("netlist")
    if isinstance(netlist, dict):
        netlist = netlist.get("netlist")
    netlist = netlist or ""
    doc_id = netlist_key(record)
    stand_in = GRAPH_CONTEXT
    if view == "ports":
        stand_in = (port_declarations(netlist, scan_procedure_ports(netlist))
                    + GRAPH_CONTEXT + "\n" + extra + "endmodule\n")
    return (
        str({"doc_id": doc_id, "netlist": netlist}),
        str({"doc_id": doc_id, "netlist": stand_in}),
    )


def graph_position(tokenizer, ids: Sequence[int]) -> int:
    """Index of the first token after the ``<GRAPH_CONTEXT>`` placeholder:
    where ``prefix_at="placeholder"`` puts the graph tokens. Found on the
    decoded text, so it holds however the tokenizer merges the placeholder
    with its neighbours."""
    if GRAPH_CONTEXT not in tokenizer.decode(list(ids)):
        raise ValueError("The prompt has no graph placeholder.")
    low, high = 1, len(ids)
    while low < high:
        middle = (low + high) // 2
        if GRAPH_CONTEXT in tokenizer.decode(list(ids[:middle])):
            high = middle
        else:
            low = middle + 1
    return low


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
    graph_at: int = 0
    # (token index, node index, weight): see :func:`port_anchors`.
    anchors: Sequence[tuple] = ()


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
    netlist_view: str = "graph",
    target: str = "full",
    port_bits: bool = False,
    hints: Optional[Dict[tuple, str]] = None,
) -> Optional[ToolSFTExample]:
    """One graph-conditioned tool-calling SFT example, or None when the record
    is rejected, its snapshot does not detect the fault, or it is longer than
    ``max_seq_len`` (dropped whole: truncation could cut through a tool turn).

    The prompt's netlist is replaced by :data:`GRAPH_CONTEXT` (with
    ``netlist_view="ports"``, all but its port declarations); the graph
    reaches the model as the soft prefix instead. ``tool_view`` sets what tool
    messages show (``atpgllm.training.tool_feedback.TOOL_VIEWS``); RL rollouts
    of the model must use the same view. ``port_bits`` lists every input bit
    after the graph slot and returns the example's ``anchors``; ``hints``
    adds the problem's text hint (a problem without one is rejected).
    """
    from atpgllm.training.conversation import ConversationExample

    graph = record_to_pyg(record, gate_funcs, vocab, rejections=rejections)
    if graph is None:
        return None
    extra = ports_extra(record, graph, port_bits=port_bits, hints=hints)
    if extra is None:
        if rejections is not None:
            rejections["no_hint"] += 1
        return None
    full, compact = _compact_netlist_payload(record, netlist_view, extra)
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
    if target not in SFT_TARGETS:
        raise ValueError(f"target must be one of {SFT_TARGETS}")
    if target == "answer":
        conversation.messages = drop_reasoning(conversation.messages)
    messages = template_turns(conversation.messages, tokenizer)
    for message in messages:
        if message["role"] == "user":
            message["content"] = message["content"].replace(full, compact)
    ids, labels = tokenize_tool_conversation(messages, conversation.trainable, tokenizer)
    if len(ids) > max_seq_len:
        if rejections is not None:
            rejections["too_long_retry" if failed_attempt else "too_long"] += 1
        return None
    graph_at = graph_position(tokenizer, ids)
    return ToolSFTExample(
        graph=graph, input_ids=ids, labels=labels, retry=failed_attempt is not None,
        graph_at=graph_at,
        anchors=port_anchors(tokenizer, ids, graph_at, graph) if port_bits else (),
    )


def collate_anchors(examples: Sequence[ToolSFTExample]) -> tuple[Tensor, Tensor]:
    """The anchors of a batch: ``[M, 3]`` (row, token, node index in the
    batched graph) and their weights ``[M]``."""
    index, weight, first_node = [], [], 0
    for row, example in enumerate(examples):
        for token, node, w in example.anchors:
            index.append((row, token, first_node + node))
            weight.append(w)
        first_node += int(example.graph.num_nodes)
    return (torch.tensor(index, dtype=torch.long).reshape(-1, 3),
            torch.tensor(weight, dtype=torch.float32))


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
    batch = {
        "graph": Batch.from_data_list([example.graph for example in examples]),
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "graph_at": torch.tensor([example.graph_at for example in examples]),
    }
    if any(example.anchors for example in examples):
        batch["anchors"] = collate_anchors(examples)
    return batch
