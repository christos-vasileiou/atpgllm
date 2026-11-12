"""The size-invariant first stack wired to a causal LM (Figure 1 on the size page).

Per circuit, once and cached: parse, levels, COP, trunk sweep (S10), cut
spectrum (S4). Per fault: view and side tokens (S1, S2), fault head, view
text with a token-to-net map, and anchored fusion into the LLM's input
embeddings. The LLM reads text and writes the same tool calls and answers
as the text model. With ``alpha = 0`` its input is exactly the text model's.

This is additive. :class:`~atpgllm.multimodal.model.GraphConditionedCausalLM`
(the BRIDGES re-test) is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch

from atpgllm.graph.netlist_parser import parse_netlist, parsed_to_pyg
from atpgllm.graph.size.fault_head import FaultHead, view_tokens
from atpgllm.graph.size.fovea import View, build_view
from atpgllm.graph.size.netgraph import NetGraph, build_netgraph, clean_netlist
from atpgllm.graph.size.spectrum import circuit_spectrum, spectrum_dim
from atpgllm.graph.size.trunk import TopoSweepTrunk
from atpgllm.multimodal.anchored import (
    AnchoredFusion,
    name_spans,
    netlist_region,
    render_view_text,
    token_net_map,
)
from atpgllm.multimodal.data import port_declarations, scan_procedure_ports


@dataclass
class CircuitCache:
    """Everything computed once per circuit."""

    netlist: str
    compiled: Any
    graph: NetGraph
    pyg: Any
    node_of_net: Dict[str, int]
    spectrum: Tensor
    trunk_vectors: Optional[Tensor] = None


def build_circuit_cache(netlist: str, gate_funcs, vocab, chain_table: str = "") -> CircuitCache:
    """``chain_table`` (a v5 row's ``scan_chain``) orders the state bits of a
    scan netlist. Both views of a scan netlist are its cut view: the net
    graph through the simulator's model, the node graph through the parser."""
    from atpgllm.training.tools import simulation_model

    text = clean_netlist(netlist)
    compiled = simulation_model(text)
    graph = build_netgraph(compiled)
    parsed = parse_netlist(text, gate_funcs, chain_table)
    pyg = parsed_to_pyg(parsed, vocab=vocab)
    TopoSweepTrunk.prepare(pyg)
    node_of_net: Dict[str, int] = {}
    for i, gate in enumerate(parsed.gates):
        if gate.cell == "<PI>":
            node_of_net[gate.inst] = i
    for net, nodes in parsed.net_drivers.items():
        if nodes and parsed.gates[nodes[0]].cell != "<PI>":
            node_of_net.setdefault(net, nodes[0])
    spectrum = torch.from_numpy(circuit_spectrum(graph))
    return CircuitCache(text, compiled, graph, pyg, node_of_net, spectrum)


@dataclass
class FaultPrompt:
    view: View
    nets: List[str]
    roles: List[int]
    hops: List[int]
    prompt: str
    input_ids: List[int]
    mapping: List[Tuple[int, int, float]]


def build_fault_prompt(cache: CircuitCache, fault: str, tokenizer, *, module: str, user_template: str,
                       cap: int = 256, k: int = 4, side_cap: int = 64) -> FaultPrompt:
    """The per-fault prompt: only the view's netlist text, plus the token map.

    ``user_template`` is formatted with ``module``, ``fault`` and ``netlist``
    (the view text), like the v4 prompt."""
    stuck, net = fault.split(None, 1)
    view = build_view(cache.graph, net, cap=cap, k=k, side_cap=side_cap)
    tokens = view_tokens(view, cache.graph)
    view_text = render_view_text(cache.compiled, view, fault=fault, module=module,
                                 ports=port_declarations(cache.netlist, scan_procedure_ports(cache.netlist)))
    user = user_template.format(module=module, fault=fault, netlist=str({"netlist": view_text}))
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                           add_generation_prompt=True)
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
    spans = name_spans(prompt, tokens["nets"], netlist_region(prompt))
    return FaultPrompt(view, tokens["nets"], tokens["roles"], tokens["hops"], prompt,
                       enc["input_ids"], token_net_map(enc["offset_mapping"], spans))


class SizeStackLM(nn.Module):
    def __init__(self, vocab, llm: nn.Module, *, dim: int = 256, head_layers: int = 2) -> None:
        super().__init__()
        self.trunk = TopoSweepTrunk(vocab, dim=dim)
        self.head = FaultHead(dim, spectrum_dim(), layers=head_layers)
        self.llm = llm
        self.fusion = AnchoredFusion(dim, llm.get_input_embeddings().weight.shape[1])

    def circuit_vectors(self, cache: CircuitCache) -> Tensor:
        """Trunk vectors for every node, computed once per circuit and kept."""
        if cache.trunk_vectors is None or self.training:
            out = self.trunk(Batch.from_data_list([cache.pyg]))["node_embs"]
            if not self.training:
                cache.trunk_vectors = out.detach()
            return out
        return cache.trunk_vectors

    def fused_embeddings(self, cache: CircuitCache, fp: FaultPrompt, stuck_at: int) -> Tensor:
        vectors = self.circuit_vectors(cache)
        index = [cache.node_of_net.get(n, -1) for n in fp.nets]
        safe = torch.tensor([i if i >= 0 else 0 for i in index], device=vectors.device)
        known = torch.tensor([i >= 0 for i in index], device=vectors.device).unsqueeze(-1)
        node_vecs = vectors[safe] * known
        z = self.head(node_vecs, torch.tensor(fp.roles, device=vectors.device),
                      torch.tensor(fp.hops, device=vectors.device), stuck_at,
                      cache.spectrum.to(vectors.device))
        ids = torch.tensor(fp.input_ids, device=vectors.device)
        embeds = self.llm.get_input_embeddings()(ids)
        return self.fusion(embeds, z, fp.mapping)

    def forward(self, cache: CircuitCache, fp: FaultPrompt, stuck_at: int, labels: Optional[Tensor] = None, **kw):
        embeds = self.fused_embeddings(cache, fp, stuck_at).unsqueeze(0)
        return self.llm(inputs_embeds=embeds, labels=labels, **kw)
