"""The two arms added to the BRIDGES re-test on 11-10: input-bit anchors and text hints."""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Batch

from atpgllm.graph.dataset import record_to_pyg
from atpgllm.graph.gate_features import GateAttributeVocab
from atpgllm.graph.models_stage1 import NetlistGraphEncoder
from atpgllm.graph.retest_qformer import RoleQueryFormer
from atpgllm.multimodal.data import (
    GRAPH_CONTEXT,
    HINT,
    PORT_BITS,
    _compact_netlist_payload,
    collate_anchors,
    hint_text,
    load_hints,
    netlist_key,
    port_anchors,
    ports_extra,
)
from atpgllm.multimodal.model import GraphConditionedCausalLM
from atpgllm.training._paths import DEFAULT_SIM_CONFIG

GATE_FUNCS = json.loads(Path(DEFAULT_SIM_CONFIG).read_text())["gate_funcs"]
VOCAB = GateAttributeVocab(GATE_FUNCS)

# A bus port: the declarations never spell a[1], the name the model writes.
BUS = """module m(a, c, y);
input [1:0] a;
input c;
output y;
wire n1;
AND2x2_ASAP7_75t_R U1 (.A(a[0]), .B(a[1]), .Y(n1));
OR2x2_ASAP7_75t_R U2 (.A(n1), .B(c), .Y(y));
endmodule"""


class _Pieces:
    """A tokenizer stand-in: words, single symbols and runs of spaces."""

    def __init__(self):
        self.pieces = []

    def encode(self, text):
        ids = []
        for piece in re.findall(r"\w+|\s+|.", text, flags=re.S):
            if piece not in self.pieces:
                self.pieces.append(piece)
            ids.append(self.pieces.index(piece))
        return ids

    def decode(self, ids, clean_up_tokenization_spaces=False):
        return "".join(self.pieces[i] for i in ids)


class _LLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(400, 12)

    def get_input_embeddings(self):
        return self.emb


def _graph(fault="sa0 n1"):
    return record_to_pyg({"netlist": BUS, "fault": fault}, GATE_FUNCS, VOCAB)


def _stack():
    torch.manual_seed(0)
    encoder = NetlistGraphEncoder(VOCAB, node_dim=16, gin_hidden_dim=16, gin_num_layers=2,
                                  per_attr_dim=4, attr_mlp_hidden=16, dropout=0.0).eval()
    former = RoleQueryFormer(d_node=16, hidden_dim=32, slots_per_role=2, num_layers=2, num_heads=4).eval()
    return encoder, former


def _prompt(graph, tokenizer, record, **extras):
    """Token ids of a ports prompt as the chat template would escape it
    (``str`` of a dict), and the index after the graph placeholder."""
    _, compact = _compact_netlist_payload(record, "ports", ports_extra(record, graph, **extras))
    text = "Find a test for the fault. " + compact
    ids = tokenizer.encode(text)
    at = len(tokenizer.encode(text[: text.index(GRAPH_CONTEXT) + len(GRAPH_CONTEXT)]))
    return ids, at, text


def test_port_bits_line_names_every_input_bit_and_anchors_its_tokens():
    graph, tokenizer = _graph(), _Pieces()
    record = {"netlist": BUS, "fault": "sa0 n1"}
    ids, at, text = _prompt(graph, tokenizer, record, port_bits=True)
    assert f"{PORT_BITS} a[0] a[1] c ;" in text
    anchors = port_anchors(tokenizer, ids, at, graph)
    by_node = {}
    for token, node, weight in anchors:
        by_node.setdefault(node, []).append((token, weight))
    spelled = {graph.node_names[node]: tokenizer.decode([ids[t] for t, _ in tokens])
               for node, tokens in by_node.items()}
    assert spelled == {"a[0]": "a[0]", "a[1]": "a[1]", "c": "c"}
    for tokens in by_node.values():
        assert sum(w for _, w in tokens) == pytest.approx(1.0)
    # Only the bit line is anchored: the declaration's "a" and "c" are not.
    line_start = len(tokenizer.encode(text[: text.index(PORT_BITS)]))
    assert min(token for token, _, _ in anchors) >= line_start


def test_port_anchors_need_the_bit_line():
    graph, tokenizer = _graph(), _Pieces()
    ids, at, _ = _prompt(graph, tokenizer, {"netlist": BUS, "fault": "sa0 n1"})
    with pytest.raises(ValueError):
        port_anchors(tokenizer, ids, at, graph)


def test_hint_line_and_missing_hint(tmp_path):
    record = {"netlist": BUS, "fault": "sa0 n1"}
    (tmp_path / "validation_000.jsonl").write_text(json.dumps(
        {"netlist_key": netlist_key(record), "fault": "sa0 n1", "hint": {"a[0]": 1, "a[1]": 1, "c": 0}}) + "\n")
    hints = load_hints(tmp_path)
    assert ports_extra(record, _graph(), hints=hints) == f"{HINT} a[0]=1 a[1]=1 c=0\n"
    assert ports_extra({"netlist": BUS, "fault": "sa1 y"}, _graph(), hints=hints) is None
    assert hint_text({"a": "x"}) == "none"
    with pytest.raises(ValueError):
        _compact_netlist_payload(record, "graph", "extra")


def _anchored_inputs(fusion):
    graph, tokenizer = _graph(), _Pieces()
    ids, at, _ = _prompt(graph, tokenizer, {"netlist": BUS, "fault": "sa0 n1"}, port_bits=True)
    example = SimpleNamespace(graph=graph, anchors=port_anchors(tokenizer, ids, at, graph))
    encoder, former = _stack()
    model = GraphConditionedCausalLM(encoder, former, _LLM(), prefix_at="placeholder", fusion=fusion)
    ids = torch.tensor([ids])
    return model, Batch.from_data_list([graph]), ids, [at], collate_anchors([example])


def test_anchor_gate_starts_at_zero_then_moves_only_the_named_tokens():
    model, batch, ids, at, anchors = _anchored_inputs("anchored")
    plain = model.llm.emb(ids)
    embeds, mask = model._prefix_and_token_embeddings(batch, ids, torch.ones_like(ids),
                                                      graph_at=at, anchors=anchors)
    slot = slice(at[0], at[0] + model.num_queries)
    text = torch.cat([embeds[0, : at[0]], embeds[0, slot.stop:]])
    assert torch.equal(text, plain[0]) and torch.count_nonzero(embeds[0, slot]) == 0
    assert model.anchor_gate() == 0.0 and mask.shape[1] == ids.shape[1] + model.num_queries
    with torch.no_grad():
        model.port_anchor.alpha.fill_(1.0)
    embeds, _ = model._prefix_and_token_embeddings(batch, ids, torch.ones_like(ids),
                                                   graph_at=at, anchors=anchors)
    text = torch.cat([embeds[0, : at[0]], embeds[0, slot.stop:]])
    moved = (text != plain[0]).any(dim=-1).nonzero().flatten().tolist()
    assert moved == sorted(set(anchors[0][:, 1].tolist()))


@pytest.mark.parametrize("fusion", ["anchored", "both"])
def test_anchored_trainable_graph_parameters_all_get_gradients(fusion):
    """As for the prefix arm: two-GPU training fails on a trainable parameter
    without a gradient. With anchors only, the slot's projector is unused."""
    model, batch, ids, at, anchors = _anchored_inputs(fusion)
    model.set_graph_policy("full")
    embeds, _ = model._prefix_and_token_embeddings(batch, ids, torch.ones_like(ids),
                                                   graph_at=at, anchors=anchors)
    embeds.square().sum().backward()
    stack = {"graph_encoder": model.graph_encoder, "q_former": model.q_former,
             "graph_to_llm": model.graph_to_llm, "port_anchor": model.port_anchor}
    missing = [f"{part}.{name}" for part, module in stack.items()
               for name, p in module.named_parameters() if p.requires_grad and p.grad is None]
    assert missing == []
    assert any(p.requires_grad for p in model.graph_to_llm.parameters()) == (fusion == "both")


def test_anchors_need_the_retest_qformer_and_rollouts_refuse_them():
    from atpgllm.graph.models_stage1 import GraphQFormer

    encoder, former = _stack()
    plain = GraphQFormer(d_node=16, hidden_dim=32, num_queries=4, num_layers=2, num_heads=4)
    with pytest.raises(ValueError):
        GraphConditionedCausalLM(encoder, plain, _LLM(), fusion="anchored")
    model = GraphConditionedCausalLM(encoder, former, _LLM(), fusion="anchored")
    with pytest.raises(NotImplementedError):
        model.generate_turns(torch.zeros(1, model.num_queries, 12), [[1, 2]], pad_token_id=0)


def test_the_anchor_gate_has_its_own_learning_rate():
    """At the adapters' rate the gate cannot open within a 300 step run."""
    import importlib.util
    import sys

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "scripts" / "train"))
    spec = importlib.util.spec_from_file_location(
        "multimodal_training", root / "scripts" / "train" / "multimodal_training.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    model, *_ = _anchored_inputs("anchored")
    model.set_graph_policy("full")
    args = SimpleNamespace(graph_encoder_lr=5e-5, graph_lr=1e-4, lr=2e-4, weight_decay=0.0,
                           anchor_gate_lr=1e-2)
    optimizer = trainer._optimizer(model, args)
    gate = [g for g in optimizer.param_groups if any(p is model.port_anchor.alpha for p in g["params"])]
    assert len(gate) == 1 and gate[0]["lr"] == 1e-2 and len(gate[0]["params"]) == 1
    assert optimizer.param_groups[-1]["lr"] == 2e-4
    held = [p for g in optimizer.param_groups for p in g["params"]]
    assert any(p is model.port_anchor.proj.weight for p in held)
    assert len(held) == len({id(p) for p in held})
