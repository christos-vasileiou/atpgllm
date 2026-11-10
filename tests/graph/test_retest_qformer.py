"""Fault-seeded role queries for the BRIDGES re-test."""

import json
from pathlib import Path

import torch
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch

from atpgllm.graph.dataset import record_to_pyg
from atpgllm.graph.gate_features import GateAttributeVocab
from atpgllm.graph.models_stage1 import NetlistGraphEncoder
from atpgllm.graph.retest_qformer import (
    RoleQueryFormer,
    anti_collapse,
    collapse_metrics,
    soft_pointer_bce,
    weighted_value_bce,
)
from atpgllm.training._paths import DEFAULT_SIM_CONFIG

GATE_FUNCS = json.loads(Path(DEFAULT_SIM_CONFIG).read_text())["gate_funcs"]
VOCAB = GateAttributeVocab(GATE_FUNCS)

TWO_GATES = """module m(a, b, c, y);
input a, b, c;
output y;
wire n1;
AND2x2_ASAP7_75t_R U1 (.A(a), .B(b), .Y(n1));
OR2x2_ASAP7_75t_R U2 (.A(n1), .B(c), .Y(y));
endmodule"""


def _setup(faults=("sa0 n1", "sa1 n1")):
    torch.manual_seed(0)
    graphs = [record_to_pyg({"netlist": TWO_GATES, "fault": f}, GATE_FUNCS, VOCAB) for f in faults]
    batch = Batch.from_data_list(graphs)
    encoder = NetlistGraphEncoder(VOCAB, node_dim=16, gin_hidden_dim=16, gin_num_layers=2,
                                  per_attr_dim=4, attr_mlp_hidden=16, dropout=0.0).eval()
    former = RoleQueryFormer(d_node=16, hidden_dim=32, slots_per_role=2, num_layers=2, num_heads=4).eval()
    return batch, encoder, former


def test_queries_shape_and_contract():
    batch, encoder, former = _setup()
    nodes = encoder(batch)["node_embs"]
    q = former(nodes, batch.batch, batch.fault_feats)
    assert q.shape == (2, former.num_queries, 32)
    assert former.num_queries == 8
    # The GraphQFormer call shape (no fault) still works.
    assert former(nodes, batch.batch).shape == q.shape


def test_stuck_value_changes_the_queries():
    batch, encoder, former = _setup()
    with torch.no_grad():
        q = former(encoder(batch)["node_embs"], batch.batch, batch.fault_feats)
    assert not torch.allclose(q[0], q[1])


def test_pointer_and_value_losses_train():
    batch, encoder, former = _setup()
    former.train()
    nodes = encoder(batch)["node_embs"]
    q = former(nodes, batch.batch, batch.fault_feats)
    dense, mask = former.last["nodes"], former.last["node_mask"]
    is_pi = ~batch.gate_node_mask & ~batch.po_node_mask
    pi_mask, _ = to_dense_batch(is_pi, batch.batch)
    assert pi_mask.sum() == 6  # a, b, c in each of the two graphs
    target = torch.zeros_like(pi_mask, dtype=torch.float)
    target[:, 0] = 1.0
    logits = former.pointer_logits("care", q, dense)
    loss = soft_pointer_bce(logits, target, pi_mask & mask)
    vloss = weighted_value_bce(former.value_logits(q, dense), target, target, pi_mask & mask)
    (loss + vloss).backward()
    assert former.ptr_q["care"].weight.grad is not None
    assert torch.isfinite(loss) and torch.isfinite(vloss)


def test_anti_collapse_and_metrics():
    copies = torch.ones(4, 8, 16)
    m = collapse_metrics(copies)
    assert abs(m["query_cosine"] - 1.0) < 1e-5 and m["circuit_variance_share"] == 0.0
    terms = anti_collapse(copies)
    assert terms["variance"] > 0.9 and terms["within"] > 0.9
    spread = torch.randn(64, 8, 16)
    m = collapse_metrics(spread)
    assert m["query_cosine"] < 0.3 and m["circuit_variance_share"] > 0.5


def test_queries_of_one_graph_start_apart():
    batch, encoder, former = _setup()
    with torch.no_grad():
        q = former(encoder(batch)["node_embs"], batch.batch, batch.fault_feats)
    assert collapse_metrics(q)["query_cosine"] < 0.5


def test_stage_c_loads_a_retest_checkpoint(tmp_path):
    from atpgllm.graph.scripts.train_retest import RetestGraphModel
    from atpgllm.multimodal.loading import load_aligned_graph_stack
    from atpgllm.multimodal.model import query_tokens

    args = {"node_dim": 16, "gin_layers": 2, "q_hidden": 32, "slots": 2, "q_layers": 2}
    model = RetestGraphModel(VOCAB, node_dim=16, gin_layers=2, q_hidden=32, slots=2, q_layers=2)
    path = tmp_path / "best.pt"
    torch.save({"format": "bridges_retest_v1", "model": model.state_dict(), "args": args,
                "vocab": VOCAB.to_dict(), "step": 1}, path)
    encoder, former, meta = load_aligned_graph_stack(path, VOCAB)
    assert meta["architecture"]["num_queries"] == 8
    batch, _, _ = _setup()
    encoder.eval(); former.eval()
    q = query_tokens(former, encoder(batch), batch)
    assert q.shape == (2, 8, 32)


def test_answer_target_drops_reasoning():
    from atpgllm.multimodal.data import drop_reasoning

    msgs = [{"role": "user", "content": "<think>keep</think> q"},
            {"role": "assistant", "content": "<think>U17 then n42</think>\n\nI call the tool."}]
    out = drop_reasoning(msgs)
    assert out[0]["content"] == msgs[0]["content"]
    assert out[1]["content"] == "I call the tool."


def test_graph_off_gives_a_zero_prefix():
    from atpgllm.multimodal.model import GraphConditionedCausalLM

    class _LLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(10, 12)

        def get_input_embeddings(self):
            return self.emb

    batch, encoder, former = _setup()
    model = GraphConditionedCausalLM(encoder, former, _LLM())
    model.graph_off = True
    prefix = model.graph_soft_prompt(batch)
    assert prefix.shape == (2, former.num_queries, 12) and torch.count_nonzero(prefix) == 0


def test_stage_c_trainable_graph_parameters_all_get_gradients():
    """DDP fails when a trainable parameter gets no gradient. The R3 decoder
    heads are not on the path to Granite, so Stage C must freeze them."""
    from atpgllm.multimodal.model import GraphConditionedCausalLM

    class _LLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(10, 12)

        def get_input_embeddings(self):
            return self.emb

    batch, encoder, former = _setup()
    model = GraphConditionedCausalLM(encoder, former, _LLM())
    model.set_graph_policy("full")
    model.graph_soft_prompt(batch).square().sum().backward()
    stack = {"graph_encoder": model.graph_encoder, "q_former": model.q_former,
             "graph_to_llm": model.graph_to_llm}
    missing = [f"{part}.{name}" for part, module in stack.items()
               for name, p in module.named_parameters() if p.requires_grad and p.grad is None]
    assert missing == []
    for name in RoleQueryFormer.DECODER_HEADS:
        assert not any(p.requires_grad for p in getattr(model.q_former, name).parameters())
