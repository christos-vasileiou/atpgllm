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
from atpgllm.graph.netlist_parser import GraphRejected, parse_verilog_to_pyg
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
module tiny(a, b, y);
input a, b;
output y;
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

    # The simulator silently skips unknown cells, so the graph rejects them.
    with pytest.raises(GraphRejected) as rejected:
        parse_verilog_to_pyg(
            """
            module unknown_cell(a, b, y);
            input a, b;
            output y;
            wire n1;
            MYSTERYx1 U0 (.A(a), .Y(n1));
            AND2x1_ASAP7_75t_R U1 (.A(n1), .B(b), .Y(y));
            endmodule
            """,
            GATE_FUNCS,
            vocab,
        )
    assert rejected.value.reason == "unknown_cell"


def test_fault_context_keeps_labels_out_of_inputs():
    vocab = GateAttributeVocab(GATE_FUNCS)
    graph = record_to_pyg(_record(), GATE_FUNCS, vocab)
    assert graph is not None
    # Nodes: <PI> a, <PI> b, U1, U2, <PO> y.
    assert graph.node_names == ["a", "b", "U1", "U2", "y"]
    assert graph.fault_feats.shape == (5, 5)
    assert graph.fault_feats[:, 0].tolist() == [0.0, 0.0, 1.0, 1.0, 0.0]
    assert graph.fault_feats[:, 3].tolist() == [0.0, 0.0, 1.0, 1.0, 0.0]
    assert graph.propagation_mask.tolist() == [False, False, True, True, False]
    assert graph.backtrack_mask.tolist() == [False, False, True, False, False]
    assert graph.discrepancy_mask.tolist() == [False, False, True, True, False]

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
    assert outputs.node_embs.shape == (10, 8)
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


class TinyDecoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.mix = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs_embeds, attention_mask=None, **kwargs):
        return SimpleNamespace(last_hidden_state=torch.tanh(self.mix(inputs_embeds)))


class TinyCausalLM(nn.Module):
    """Decoder, head and logit scaling laid out like transformers' Granite."""

    def __init__(self, vocab_size=17, hidden_size=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.model = TinyDecoder(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.config = SimpleNamespace(logits_scaling=2.0)

    def get_input_embeddings(self):
        return self.embedding

    def get_decoder(self):
        return self.model

    def get_output_embeddings(self):
        return self.head

    def forward(self, inputs_embeds, attention_mask=None, labels=None, **kwargs):
        hidden = self.model(inputs_embeds).last_hidden_state
        logits = self.head(hidden) / self.config.logits_scaling
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
    labels[1, 1] = 4  # rows train 1 and 2 tokens
    output = model(batch, input_ids, attention_mask, labels)
    # The loss is the mean of each example's own causal LM loss, also when
    # trained tokens span several loss chunks.
    embeds, _ = model._prefix_and_token_embeddings(batch, input_ids, attention_mask)
    full_labels = torch.cat([torch.full((2, 2), -100), labels], dim=1)
    expected = torch.stack([
        model.llm(inputs_embeds=embeds[row:row + 1], labels=full_labels[row:row + 1]).loss
        for row in range(2)
    ]).mean()
    assert torch.allclose(output.loss, expected)
    model.LOSS_CHUNK = 2
    chunked = model(batch, input_ids, attention_mask, labels)
    assert torch.allclose(chunked.loss, expected)
    chunked.loss.backward()

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


def test_generate_turns_keeps_the_stop_token_when_it_is_also_padding():
    # Granite 4.2 pads with its stop token <|im_end|>: a finished row must
    # keep the stop, and the padding after it must go.
    stop = 7

    class FakeLLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(16, 4)

        def get_input_embeddings(self):
            return self.embed

        def generate(self, inputs_embeds, attention_mask, **kwargs):
            return torch.tensor([[3, 4, stop, stop, stop], [5, 6, 8, 9, 10]])

    fake = SimpleNamespace(llm=FakeLLM())
    turns = GraphConditionedCausalLM.generate_turns(
        fake, torch.zeros(1, 2, 4), [[1, 2], [1]],
        pad_token_id=stop, eos_token_id=stop, max_new_tokens=5,
    )
    assert turns == [[3, 4, stop], [5, 6, 8, 9, 10]]


def _tiny_model(**kwargs):
    vocab = GateAttributeVocab(GATE_FUNCS)
    graph_encoder = NetlistGraphEncoder(
        vocab, node_dim=8, gin_hidden_dim=8, gin_num_layers=1,
        per_attr_dim=2, attr_mlp_hidden=8, dropout=0.0,
    )
    q_former = GraphQFormer(
        d_node=8, hidden_dim=8, num_queries=2, num_layers=2, num_heads=2,
        ffn_dim=16, cross_attn_every_n=2,
    )
    model = GraphConditionedCausalLM(graph_encoder, q_former, TinyCausalLM(), **kwargs)
    graph = record_to_pyg(_record(), GATE_FUNCS, vocab)
    return model, Batch.from_data_list([graph, graph])


def test_ports_view_keeps_declarations_and_drops_cells():
    from atpgllm.multimodal.data import GRAPH_CONTEXT, _compact_netlist_payload

    full, graph_only = _compact_netlist_payload(_record())
    _, ports = _compact_netlist_payload(_record(), "ports")
    assert full != graph_only and GRAPH_CONTEXT in graph_only and "input" not in graph_only
    assert "module tiny(a, b, y);" in ports
    assert "input a, b;" in ports and "output y;" in ports
    assert GRAPH_CONTEXT in ports and "endmodule" in ports
    assert "AND2x1" not in ports and "wire" not in ports
    # Same doc_id, so tool calls name the same circuit.
    assert full.split("'netlist'")[0] == ports.split("'netlist'")[0]


def test_graph_position_is_the_token_after_the_placeholder():
    from atpgllm.multimodal.data import graph_position

    class CharTokenizer:
        pieces = ["sys", " {'netlist': '", "<GRAPH", "_CONTEXT>", "'}", " user", " end"]

        def decode(self, ids):
            return "".join(self.pieces[i] for i in ids)

    assert graph_position(CharTokenizer(), list(range(7))) == 4
    with pytest.raises(ValueError):
        graph_position(CharTokenizer(), [0, 1, 2])


def test_placeholder_prefix_goes_inside_the_text():
    model, batch = _tiny_model(prefix_at="placeholder")
    assert model.architecture_config["prefix_at"] == "placeholder"
    assert "prefix_at" not in _tiny_model()[0].architecture_config
    input_ids = torch.tensor([[1, 2, 3, 4], [1, 4, 5, 6]])
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, :3] = -100
    graph_at = torch.tensor([2, 1])
    with pytest.raises(ValueError):
        model(batch, input_ids, attention_mask, labels)
    output = model(batch, input_ids, attention_mask, labels, graph_at=graph_at)

    prefix = model.graph_soft_prompt(batch)
    tokens = model.llm.get_input_embeddings()(input_ids)
    expected = []
    for row, at in enumerate(graph_at.tolist()):
        embeds = torch.cat([tokens[row, :at], prefix[row], tokens[row, at:]]).unsqueeze(0)
        row_labels = torch.cat([labels[row, :at], torch.full((2,), -100), labels[row, at:]])
        expected.append(model.llm(inputs_embeds=embeds, labels=row_labels.unsqueeze(0)).loss)
    assert torch.allclose(output.loss, torch.stack(expected).mean())

    # The completion still starts num_queries + prompt_len positions in.
    completion = torch.tensor([[7, 8], [9, 10]])
    log_probs = model.completion_log_probs(
        batch, input_ids, attention_mask, completion, torch.ones_like(completion),
        graph_at=graph_at,
    )
    embeds = torch.cat([tokens[0, :2], prefix[0], tokens[0, 2:],
                        model.llm.get_input_embeddings()(completion[0])]).unsqueeze(0)
    logits = model.llm(inputs_embeds=embeds).logits[0]
    manual = logits[2 + 4 - 1:2 + 4 + 1].log_softmax(-1).gather(-1, completion[0][:, None])[:, 0]
    assert torch.allclose(log_probs[0], manual)


def test_generate_turns_puts_the_graph_at_the_same_prompt_position():
    captured = {}

    class FakeLLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(16, 4)

        def get_input_embeddings(self):
            return self.embed

        def generate(self, inputs_embeds, attention_mask, **kwargs):
            captured["embeds"], captured["mask"] = inputs_embeds, attention_mask
            return torch.tensor([[3, 7], [5, 7]])

    fake = SimpleNamespace(llm=FakeLLM(), prefix_at="placeholder")
    prefix = torch.full((1, 2, 4), 9.0)
    GraphConditionedCausalLM.generate_turns(
        fake, prefix, [[1, 2, 3], [1, 2, 3, 4]], pad_token_id=7, eos_token_id=7,
        graph_at=2, max_new_tokens=2,
    )
    embeds, mask = captured["embeds"], captured["mask"]
    # Row 0 is left-padded by one token: pad, 1, 2, graph, graph, 3.
    assert mask[0].tolist() == [0, 1, 1, 1, 1, 1]
    assert torch.all(embeds[0, 3:5] == 9.0) and torch.all(embeds[1, 2:4] == 9.0)
    assert torch.equal(embeds[0, 5], fake.llm.embed(torch.tensor(3)))


def test_frozen_graph_batchnorm_stays_in_eval_mode():
    model, _ = _tiny_model(freeze_graph_batchnorm=True)
    model.train()
    norms = [m for m in model.graph_encoder.modules()
             if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    assert norms and not any(m.training for m in norms)
    assert model.q_former.training
    unfrozen, _ = _tiny_model()
    unfrozen.train()
    assert all(m.training for m in unfrozen.graph_encoder.modules()
               if isinstance(m, nn.modules.batchnorm._BatchNorm))
