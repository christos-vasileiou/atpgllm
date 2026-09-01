from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch_geometric.data import Batch

from atpgllm.graph.checkpoints import (
    STAGE_GRAPH_PRETRAIN,
    STAGE_GRAPH_TEXT_ALIGNMENT,
    load_stage_checkpoint,
    save_stage_checkpoint,
)
from atpgllm.graph.dataset import record_to_pyg, render_prompt_and_answer
from atpgllm.graph.gate_features import GateAttributeVocab
from atpgllm.graph.models_stage1 import GraphQFormer, NetlistGraphEncoder
from atpgllm.graph.netlist_parser import parse_verilog_to_pyg
from atpgllm.graph.pretrain import GraphPretrainingModel
from atpgllm.multimodal.grpo import (
    graph_grpo_loss,
    group_relative_advantages,
)
from atpgllm.multimodal.loading import load_aligned_graph_stack
from atpgllm.multimodal.model import GraphConditionedCausalLM


GATE_FUNCS = {
    "AND2x1_ASAP7_75t_R": {"Y": "A & B"},
    "INVx1_ASAP7_75t_R": {"Y": "~A"},
}

NETLIST = """
module tiny(input a, input b, output y);
wire n1;
AND2x1_ASAP7_75t_R U1 (.A(a), .B(b), .Y(n1));
INVx1_ASAP7_75t_R U2 (.A(n1), .Y(y));
endmodule
"""


def _record():
    return {
        "netlist": NETLIST,
        "fault": "sa0 n1",
        "fault_propagation_gates": "U1, U2",
        "backtrack_gates": "U1",
        "snapshot": str({
            "Good Machine": {"n1": 1, "y": 0},
            "Bad Machine": {"n1": 0, "y": 1},
        }),
    }


def test_vocab_round_trip_has_explicit_unknowns():
    vocab = GateAttributeVocab(GATE_FUNCS)
    known = vocab.encode("AND2x1_ASAP7_75t_R")
    unknown = vocab.encode("UNSEENx1_OTHER")
    assert len(known) == vocab.num_attributes
    assert unknown == (0,) * vocab.num_attributes
    assert known[vocab.attribute_names.index("is_sequential")] == 1

    restored = GateAttributeVocab.from_dict(vocab.to_dict())
    assert restored.fingerprint == vocab.fingerprint
    assert restored.encode("AND2x1_ASAP7_75t_R") == known

    unknown_graph = parse_verilog_to_pyg(
        """
        module unknown_cell(input a, input b, output y);
        wire n1;
        MYSTERYx1 U0 (.A(a), .Y(n1));
        AND2x1_ASAP7_75t_R U1 (.A(n1), .B(b), .Y(y));
        endmodule
        """,
        GATE_FUNCS,
        vocab,
    )
    assert unknown_graph.num_nodes == 2
    assert unknown_graph.gate_attrs[0].tolist() == list(unknown)


def test_fault_context_keeps_labels_out_of_inputs():
    vocab = GateAttributeVocab(GATE_FUNCS)
    graph = record_to_pyg(_record(), GATE_FUNCS, vocab)
    assert graph is not None
    assert graph.fault_feats.shape == (2, 5)
    assert graph.fault_feats[:, 0].tolist() == [1.0, 1.0]
    assert graph.fault_feats[:, 3].tolist() == [1.0, 1.0]
    assert graph.propagation_mask.tolist() == [True, True]
    assert graph.backtrack_mask.tolist() == [True, False]
    assert graph.discrepancy_mask.tolist() == [True, True]

    changed = _record()
    changed["snapshot"] = ""
    no_snapshot = record_to_pyg(changed, GATE_FUNCS, vocab)
    assert torch.equal(graph.fault_feats, no_snapshot.fault_feats)
    assert not no_snapshot.has_discrepancy_labels.item()


def test_multimodal_prompt_replaces_duplicate_netlist():
    class PlainTokenizer:
        chat_template = None

    record = {
        **_record(),
        "module_name": "tiny",
        "system_content": "ATPG",
        "user_content": "Target {fault} with {netlist}",
        "reasoning_content": "",
        "answer_content": "done",
        "input_vector": "{}",
        "expected_output": "{}",
        "detected_faults": "sa0 n1",
        "backtrack_nets": "a, b",
    }
    prompt, answer = render_prompt_and_answer(
        record,
        PlainTokenizer(),
        compact_netlist=True,
    )
    assert "<GRAPH_CONTEXT>" in prompt
    assert "AND2x1_ASAP7_75t_R" not in prompt
    assert "doc_id" in prompt
    assert "done" in answer


def test_graph_pretraining_shapes_and_losses():
    vocab = GateAttributeVocab(GATE_FUNCS)
    graph = record_to_pyg(_record(), GATE_FUNCS, vocab)
    batch = Batch.from_data_list([graph, graph])
    model = GraphPretrainingModel(
        vocab,
        node_dim=8,
        gin_hidden_dim=8,
        gin_num_layers=1,
        dropout=0.0,
    )
    outputs = model(batch)
    losses = model.compute_losses(batch, outputs)
    assert outputs.node_embs.shape == (4, 8)
    assert torch.isfinite(losses.total)


def test_qformer_masks_padded_nodes():
    torch.manual_seed(7)
    q_former = GraphQFormer(
        d_node=8,
        hidden_dim=8,
        num_queries=2,
        num_layers=2,
        num_heads=2,
        cross_attn_every_n=2,
        dropout=0.0,
    ).eval()
    short = torch.randn(2, 8)
    long = torch.randn(5, 8)
    alone = q_former(short, torch.zeros(2, dtype=torch.long))[0]
    together = q_former(
        torch.cat([short, long]),
        torch.tensor([0, 0, 1, 1, 1, 1, 1]),
    )[0]
    assert torch.allclose(alone, together, atol=1e-6)


def test_checkpoint_rejects_vocab_mismatch(tmp_path):
    vocab = GateAttributeVocab(GATE_FUNCS)
    path = tmp_path / "graph.pt"
    save_stage_checkpoint(
        path,
        stage=STAGE_GRAPH_PRETRAIN,
        parent_stage=None,
        vocab=vocab,
        architecture={"graph_encoder": {"hidden": 8}},
        states={"graph_encoder": {}},
        step=3,
    )
    loaded = load_stage_checkpoint(
        path,
        expected_stages=[STAGE_GRAPH_PRETRAIN],
        vocab=vocab,
        expected_architecture={"graph_encoder": {"hidden": 8}},
    )
    assert loaded["step"] == 3

    other = GateAttributeVocab(
        {**GATE_FUNCS, "INVx2_ASAP7_75t_R": {"Y": "~A"}}
    )
    with pytest.raises(ValueError, match="vocabulary mismatch"):
        load_stage_checkpoint(
            path,
            expected_stages=[STAGE_GRAPH_PRETRAIN],
            vocab=other,
        )


def test_alignment_checkpoint_loads_without_text_encoder(tmp_path):
    vocab = GateAttributeVocab(GATE_FUNCS)
    graph_encoder = NetlistGraphEncoder(
        vocab,
        node_dim=8,
        gin_hidden_dim=8,
        gin_num_layers=1,
        per_attr_dim=2,
        attr_mlp_hidden=8,
        dropout=0.0,
    )
    q_former = GraphQFormer(
        d_node=8,
        hidden_dim=8,
        num_queries=2,
        num_layers=2,
        num_heads=2,
        cross_attn_every_n=2,
        dropout=0.0,
    )
    model_state = {
        **{
            f"graph_encoder.{name}": value
            for name, value in graph_encoder.state_dict().items()
        },
        **{
            f"q_former.{name}": value
            for name, value in q_former.state_dict().items()
        },
    }
    path = tmp_path / "alignment.pt"
    save_stage_checkpoint(
        path,
        stage=STAGE_GRAPH_TEXT_ALIGNMENT,
        parent_stage=STAGE_GRAPH_PRETRAIN,
        vocab=vocab,
        architecture={
            "graph_encoder": graph_encoder.config,
            "qformer_hidden_dim": 8,
            "qformer_layers": 2,
            "qformer_heads": 2,
            "qformer_cross_every_n": 2,
            "num_queries": 2,
        },
        states={"model": model_state},
        step=1,
    )
    loaded_graph, loaded_qformer, _ = load_aligned_graph_stack(path, vocab)
    assert loaded_graph.out_dim == 8
    assert loaded_qformer.num_queries == 2


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size=17, hidden_size=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, attention_mask=None, labels=None, **kwargs):
        logits = self.head(inputs_embeds)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(logits=logits, loss=loss)


def test_multimodal_prefix_sft_and_grpo_wiring():
    vocab = GateAttributeVocab(GATE_FUNCS)
    graph = record_to_pyg(_record(), GATE_FUNCS, vocab)
    batch = Batch.from_data_list([graph, graph])
    graph_encoder = NetlistGraphEncoder(
        vocab,
        node_dim=8,
        gin_hidden_dim=8,
        gin_num_layers=1,
        per_attr_dim=2,
        attr_mlp_hidden=8,
        dropout=0.0,
    )
    q_former = GraphQFormer(
        d_node=8,
        hidden_dim=8,
        num_queries=2,
        num_layers=2,
        num_heads=2,
        ffn_dim=16,
        cross_attn_every_n=2,
    )
    model = GraphConditionedCausalLM(
        graph_encoder,
        q_former,
        TinyCausalLM(),
    )
    input_ids = torch.tensor([[1, 2, 3], [1, 4, 5]])
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, :2] = -100
    output = model(batch, input_ids, attention_mask, labels)
    assert output.logits.shape == (2, 5, 17)
    assert torch.isfinite(output.loss)

    completion_ids = torch.tensor([[6, 7], [8, 9]])
    completion_mask = torch.ones_like(completion_ids)
    log_probs = model.completion_log_probs(
        batch,
        input_ids,
        attention_mask,
        completion_ids,
        completion_mask,
    )
    assert log_probs.shape == completion_ids.shape

    policy = log_probs.detach().clone().requires_grad_(True)
    advantages = group_relative_advantages(torch.tensor([0.0, 1.0]))
    loss = graph_grpo_loss(
        policy,
        log_probs.detach(),
        log_probs.detach(),
        completion_mask,
        advantages,
    )
    loss.backward()
    assert policy.grad is not None
