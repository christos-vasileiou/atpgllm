"""Fault head and anchored fusion."""

import torch

from atpgllm.graph.size.fault_head import FaultHead, view_tokens
from atpgllm.graph.size.fovea import build_view
from atpgllm.graph.size.netgraph import build_netgraph
from atpgllm.multimodal.anchored import AnchoredFusion, name_spans, netlist_region, render_view_text, token_net_map

TWO = """module m(a, b, c, y);
input a, b, c;
output y;
wire n1;
AND2x2_ASAP7_75t_R U1 (.A(a), .B(b), .Y(n1));
OR2x2_ASAP7_75t_R U2 (.A(n1), .B(c), .Y(y));
endmodule"""


def test_names_match_whole_words_inside_an_escaped_prompt():
    prompt = str({"netlist": "AND2 U1 (.A(n4), .B(n42), .Y(a[3]))\nOR2 U10 (.A(a))"})
    spans = name_spans(prompt, ["n4", "n42", "a", "U1"])
    found = {(prompt[s:e], i) for s, e, i in spans}
    assert ("n4", 0) in found and ("n42", 1) in found and ("U1", 3) in found
    assert sum(1 for s, e, i in spans if i == 0) == 1        # n4 not inside n42
    assert sum(1 for s, e, i in spans if i == 2) == 1        # a, not inside a[3]
    assert sum(1 for s, e, i in spans if i == 3) == 1        # U1, not inside U10


def test_token_map_splits_weight_over_pieces():
    offsets = [(0, 2), (2, 4), (4, 5), (5, 8)]
    mapping = token_net_map(offsets, [(0, 4, 0), (5, 8, 1)])
    assert mapping == [(0, 0, 0.5), (1, 0, 0.5), (3, 1, 1.0)]


def test_gate_at_zero_is_bit_identical():
    torch.manual_seed(0)
    fusion = AnchoredFusion(8, 12)
    embeds = torch.randn(5, 12)
    out = fusion(embeds, torch.randn(2, 8), [(1, 0, 1.0), (3, 1, 0.5)])
    assert torch.equal(out, embeds)
    with torch.no_grad():
        fusion.alpha.fill_(0.5)
    moved = fusion(embeds, torch.randn(2, 8), [(1, 0, 1.0)])
    assert not torch.equal(moved[1], embeds[1]) and torch.equal(moved[0], embeds[0])


def test_fault_head_and_view_text(monkeypatch):
    monkeypatch.setenv("FAULT_SIM_BACKEND", "fast")
    from atpgllm.training.tools import simulation_model

    compiled = simulation_model(TWO)
    g = build_netgraph(compiled)
    view = build_view(g, "n1", cap=16, k=1)
    tokens = view_tokens(view, g)
    assert tokens["nets"][0] == "n1" and "c" in tokens["nets"]
    head = FaultHead(16, 266, layers=1, heads=2)
    z = head(torch.randn(len(tokens["nets"]), 16), torch.tensor(tokens["roles"]),
             torch.tensor(tokens["hops"]), 0, torch.randn(266))
    assert z.shape == (len(tokens["nets"]), 16)
    text = render_view_text(compiled, view, fault="sa0 n1", module="m",
                            ports="module m(a, b, c, y);\ninput a, b, c;\noutput y;")
    assert "U1" in text and "U2" in text and "Side inputs from outside the view: c" in text


def test_names_anchor_only_inside_the_netlist():
    prompt = "I need a test pattern for " + str({"doc_id": "x", "netlist": "AND2 U1 (.A(a))"})
    spans = name_spans(prompt, ["a"], netlist_region(prompt))
    assert len(spans) == 1
    start, end, _ = spans[0]
    assert prompt[start:end] == "a" and start > prompt.find("'netlist'")


def test_size_stack_end_to_end_with_a_tiny_lm(monkeypatch):
    """Record to fused embeddings: at alpha = 0 the LLM input equals the
    text model's embeddings of the view prompt."""
    import json
    from pathlib import Path

    from atpgllm.graph.gate_features import GateAttributeVocab
    from atpgllm.multimodal.size_stack import SizeStackLM, build_circuit_cache, build_fault_prompt
    from atpgllm.training._paths import DEFAULT_SIM_CONFIG

    monkeypatch.setenv("FAULT_SIM_BACKEND", "fast")
    gate_funcs = json.loads(Path(DEFAULT_SIM_CONFIG).read_text())["gate_funcs"]
    vocab = GateAttributeVocab(gate_funcs)

    class Tok:
        """Character-level stand-in with offsets and a trivial chat template."""
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            return "<|user|>" + msgs[0]["content"] + "<|end_of_text|><|assistant|>"

        def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False):
            return {"input_ids": [ord(ch) % 97 for ch in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}

    class LM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(97, 12)

        def get_input_embeddings(self):
            return self.emb

        def forward(self, inputs_embeds=None, labels=None):
            return inputs_embeds

    torch.manual_seed(0)
    cache = build_circuit_cache(TWO, gate_funcs, vocab)
    fp = build_fault_prompt(cache, "sa0 n1", Tok(), module="m",
                            user_template='Test "{module}" for "{fault}": {netlist}', cap=16, k=1)
    assert fp.mapping and "U1" in fp.prompt
    model = SizeStackLM(vocab, LM(), dim=16, head_layers=1).eval()
    fused = model.fused_embeddings(cache, fp, 0)
    plain = model.llm.get_input_embeddings()(torch.tensor(fp.input_ids))
    assert torch.equal(fused, plain)
    with torch.no_grad():
        model.fusion.alpha.fill_(1.0)
    moved = model.fused_embeddings(cache, fp, 0)
    touched = {t for t, _, _ in fp.mapping}
    assert all(torch.equal(moved[t], plain[t]) for t in range(len(plain)) if t not in touched)
    assert any(not torch.equal(moved[t], plain[t]) for t in touched)
