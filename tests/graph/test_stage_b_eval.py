"""Stage-B evaluation: per-circuit retrieval, the held-out set, frozen modes."""

import pytest
import torch
from torch import nn

from atpgllm.graph.hpo.metrics import grouped_retrieval_metrics
from atpgllm.graph.stage_b_eval import build_alignment_validation_set
from atpgllm.graph.train_stage1 import Stage1Trainer


def test_grouped_retrieval_separates_circuit_and_fault():
    groups = ["a", "a", "b", "b"]
    perfect = grouped_retrieval_metrics(torch.eye(4), groups)
    assert perfect["circuit_r1"] == 1.0 and perfect["within_circuit_r1"] == 1.0
    assert perfect["within_circuit_chance_r1"] == 0.5
    assert perfect["chance_r1"] == 0.25

    # Right circuit, but the two faults of circuit a are swapped.
    swapped = torch.tensor([
        [0.5, 0.9, 0.0, 0.0],
        [0.9, 0.5, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    metrics = grouped_retrieval_metrics(swapped, groups)
    assert metrics["circuit_r1"] == 1.0
    assert metrics["within_circuit_r1"] == pytest.approx(0.5)


def test_grouped_retrieval_counts_ties_as_misses_and_skips_singletons():
    # Identical scores inside circuit a: no encoder could tell them apart.
    tied = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    metrics = grouped_retrieval_metrics(tied, ["a", "a", "b"])
    assert metrics["within_circuit_r1"] == 0.0
    assert metrics["within_circuit_pairs"] == 2  # circuit b has one pair
    assert grouped_retrieval_metrics(torch.eye(2), ["a", "b"])["within_circuit_r1"] is None


def test_validation_set_keeps_one_pair_per_fault():
    examples = [
        {"circuit": "a", "fault": "sa0 n1"},
        {"circuit": "a", "fault": "sa0 n1"},  # another pattern, same graph
        {"circuit": "a", "fault": "sa1 n1"},
        {"circuit": "a", "fault": "sa0 n2"},
        {"circuit": "b", "fault": "sa0 n1"},
    ]
    items = build_alignment_validation_set(examples, per_circuit=2, limit=10)
    assert [(i["circuit"], i["fault"]) for i in items] == [
        ("a", "sa0 n1"), ("a", "sa1 n1"), ("b", "sa0 n1"),
    ]
    assert len(build_alignment_validation_set(examples, per_circuit=2, limit=2)) == 2


def test_frozen_encoders_stay_in_eval_mode():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.graph_encoder = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
            self.text_encoder = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.5))

    trainer = Stage1Trainer.__new__(Stage1Trainer)
    trainer.model = Model()
    for parameter in trainer.model.graph_encoder.parameters():
        parameter.requires_grad = False
    trainer.model.train()
    trainer._keep_frozen_modules_in_eval()
    assert not trainer.model.graph_encoder.training
    assert trainer.model.text_encoder.training  # still trainable

    # Partly trainable: only the frozen BatchNorm is held in eval mode.
    trainer.model.graph_encoder[0].weight.requires_grad = True
    trainer.model.train()
    trainer._keep_frozen_modules_in_eval()
    assert trainer.model.graph_encoder.training
    assert not trainer.model.graph_encoder[1].training
