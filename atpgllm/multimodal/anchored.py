"""Anchored fusion: graph vectors added to the prompt tokens that name each net.

For a prompt token ``t`` that is part of the name of net ``v``::

    embedding(t) + tanh(alpha) · W · z(v) / pieces(v)

``alpha`` starts at 0, so at step 0 the LLM input is bit-identical to the
text model's. ``pieces(v)`` is how many subword tokens spell the name, so a
long name does not get a larger push. Tokens that name no net are unchanged.

The token map is built on the final rendered prompt (after the chat
template), from the tokenizer's offset mapping, so it holds however the
template quotes or escapes the netlist.

This module is additive. The BRIDGES re-test path (soft prefix in
:class:`~atpgllm.multimodal.model.GraphConditionedCausalLM`) is unchanged.
"""

from __future__ import annotations

import bisect
import re
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn


def name_spans(text: str, names: Sequence[str], region: Optional[Tuple[int, int]] = None) -> List[Tuple[int, int, int]]:
    """(start, end, name index) of every whole-word occurrence of each name.

    A name matches only when not glued to another identifier character, so
    ``n4`` does not match inside ``n42`` and ``a`` not inside ``a[3]``.
    ``region`` limits matching to the netlist part of the prompt, so a net
    called ``a`` does not anchor on the English word in "a test pattern"."""
    lo, hi = region if region is not None else (0, len(text))
    spans = []
    for idx, name in enumerate(names):
        pattern = re.compile(r"(?<![\w\\\[\]])" + re.escape(name) + r"(?![\w\[])")
        spans += [(m.start(), m.end(), idx) for m in pattern.finditer(text, lo, hi)]
    return sorted(spans)


def netlist_region(prompt: str, marker: str = "'netlist':") -> Tuple[int, int]:
    """From the netlist payload to the end of that message."""
    start = prompt.find(marker)
    if start < 0:
        return (0, len(prompt))
    end = prompt.find("<|end_of_text|>", start)
    return (start, end if end >= 0 else len(prompt))


def token_net_map(offsets: Sequence[Tuple[int, int]], spans: Sequence[Tuple[int, int, int]]) -> List[Tuple[int, int, float]]:
    """(token index, name index, weight) for tokens that overlap a name span."""
    ends = [b for _, b in offsets]
    out: List[Tuple[int, int, float]] = []
    for start, end, name in spans:
        first = bisect.bisect_right(ends, start)   # first token ending after the span starts
        tokens = []
        t = first
        while t < len(offsets) and offsets[t][0] < end:
            if offsets[t][1] > offsets[t][0]:
                tokens.append(t)
            t += 1
        for t in tokens:
            out.append((t, name, 1.0 / len(tokens)))
    return out


class AnchoredFusion(nn.Module):
    def __init__(self, z_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(z_dim, llm_dim)
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, embeds: Tensor, z: Tensor, mapping: Sequence[Tuple[int, int, float]]) -> Tensor:
        """``embeds [L, D]`` token embeddings of one prompt, ``z [N, dz]`` one
        vector per named net, ``mapping`` from :func:`token_net_map`."""
        if not mapping:
            return embeds
        tok = torch.tensor([m[0] for m in mapping], device=embeds.device)
        net = torch.tensor([m[1] for m in mapping], device=embeds.device)
        w = torch.tensor([m[2] for m in mapping], device=embeds.device, dtype=embeds.dtype)
        add = self.proj(z[net]).to(embeds.dtype) * w.unsqueeze(-1) * torch.tanh(self.alpha).to(embeds.dtype)
        return embeds.index_add(0, tok, add)


def render_view_text(compiled, view, *, fault: str, module: str, ports: str) -> str:
    """The view as netlist text: the module header and port declarations,
    the view's gates in topological order, then the side inputs by name."""
    by_out: Dict[str, Tuple[str, str, Dict[str, str], str, str]] = {}
    for cell, inst, out_pin, out_net, pins in compiled.gate_metadata:
        by_out[out_net] = (cell, inst, pins, out_pin, out_net)
    keep = set(view.nets)
    lines = [ports.rstrip()]
    seen_inst = set()
    for instr in compiled.instructions:
        out = instr[1]
        if out not in keep or out not in by_out:
            continue
        cell, inst, pins, out_pin, out_net = by_out[out]
        if inst in seen_inst:
            continue
        seen_inst.add(inst)
        conns = ", ".join(f".{p}({n})" for p, n in pins.items())
        lines.append(f"{cell} {inst} ( {conns}, .{out_pin}({out_net}) );")
    lines.append("endmodule")
    text = "\n".join(lines)
    if view.side:
        text += "\nSide inputs from outside the view: " + ", ".join(view.side)
    return f"Fault {fault} in {module}, view of {len(view.nets)} nets:\n{text}"
