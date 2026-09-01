from pathlib import Path

import pytest
import torch

from atpgllm.graph.hpo.config import load_hpo_config
from atpgllm.graph.hpo.manifests import (
    build_trial_manifest,
    canonical_hash,
    checkpoint_filename,
    design_hash,
)
from atpgllm.graph.hpo.metrics import (
    binary_average_precision,
    binary_iou,
    binary_roc_auc,
    graph_pretrain_score,
    retrieval_metrics,
)
from atpgllm.graph.hpo.promotion import (
    aggregate_seed_rows,
    select_top_k,
)
from atpgllm.graph.hpo.spaces import (
    assert_graph_architecture_locked,
    suggest_graph_pretrain,
    suggest_graph_text_alignment,
)
from atpgllm.graph.hpo.splits import (
    DesignSplitManifest,
    build_design_split_manifest,
)
from atpgllm.graph.scripts.search_graph_pipeline import (
    DEFAULT_CONFIG,
    build_parser,
    main,
)


class FakeTrial:
    def __init__(self, selected=None):
        self.selected = dict(selected or {})
        self.names = []

    def suggest_categorical(self, name, choices):
        self.names.append(name)
        value = self.selected.get(name, choices[0])
        assert value in choices
        return value

    def suggest_int(self, name, low, high, step=1, log=False):
        self.names.append(name)
        return int(self.selected.get(name, low))

    def suggest_float(self, name, low, high, step=None, log=False):
        self.names.append(name)
        return float(self.selected.get(name, low))


def test_conditional_spaces_and_normalized_weights():
    config = load_hpo_config(DEFAULT_CONFIG, profile="smoke")
    stage_a = suggest_graph_pretrain(
        FakeTrial(),
        config.stage("graph_pretrain").search,
    )
    assert sum(stage_a["loss_weights"].values()) == pytest.approx(3.0)

    trial = FakeTrial({"qformer_hidden": 256})
    stage_b = suggest_graph_text_alignment(
        trial,
        config.stage("graph_text_alignment").search,
    )
    assert stage_b["qformer_hidden"] == 256
    assert stage_b["qformer_heads"] in {4, 8}
    assert "qformer_heads_h256" in trial.names
    assert sum(stage_b["loss_weights"].values()) == pytest.approx(3.0)


def test_graph_architecture_locking():
    parent = {"gin_hidden_dim": 256, "gin_num_layers": 6}
    assert_graph_architecture_locked(parent, dict(parent))
    with pytest.raises(ValueError, match="exactly match"):
        assert_graph_architecture_locked(
            parent,
            {"gin_hidden_dim": 128, "gin_num_layers": 6},
        )


def test_semantic_metric_edge_cases():
    assert binary_average_precision([0.1, 0.2], [0, 0]) is None
    assert binary_average_precision([0.1, 0.2], [1, 1]) == pytest.approx(1.0)
    assert binary_roc_auc([0.1, 0.2], [1, 1]) is None
    assert binary_roc_auc([0.1, 0.9], [0, 1]) == pytest.approx(1.0)
    assert binary_iou([0.1, 0.2], [0, 0]) == pytest.approx(1.0)
    score = graph_pretrain_score({
        "propagation_ap": 0.8,
        "discrepancy_ap": None,
        "backtrack_ap": 0.6,
    })
    assert 0.6 <= score <= 0.8

    retrieval = retrieval_metrics(torch.eye(3))
    assert retrieval["mean_r1"] == pytest.approx(1.0)
    assert retrieval["mean_mrr"] == pytest.approx(1.0)


def test_design_disjoint_manifest_is_deterministic(tmp_path):
    records = [
        {"netlist": "module a; endmodule", "fault": "sa0 n1"},
        {"netlist": "module a; endmodule", "fault": "sa1 n1"},
        {"netlist": "module b; endmodule", "fault": "sa0 n2"},
        {"netlist": "module c; endmodule", "fault": "sa0 n3"},
    ]
    first = build_design_split_manifest(
        records,
        dataset="fixture",
        revision="v1",
        source_split="train",
        seed=42,
        max_records=100,
    )
    second = build_design_split_manifest(
        records,
        dataset="fixture",
        revision="v1",
        source_split="train",
        seed=42,
        max_records=100,
    )
    assert first.to_dict() == second.to_dict()
    assert first.split_for_record(records[0]) == first.split_for_record(records[1])
    path = first.save(tmp_path / "split.json")
    assert DesignSplitManifest.load(path).to_dict() == first.to_dict()
    assert len(first.designs) == 3


def test_hashes_and_artifact_names_are_deterministic():
    payload = {"b": [2, 1], "a": {"x": True}}
    assert canonical_hash(payload) == canonical_hash(
        {"a": {"x": True}, "b": [2, 1]}
    )
    assert design_hash({"netlist": "abc"}) == design_hash("abc")
    name = checkpoint_filename(
        "study/name",
        "graph_pretrain",
        7,
        42,
        500,
        canonical_hash(payload),
    )
    assert name.startswith("study_name_graph_pretrain_t00007_s42_step500_")


def test_trial_manifest_is_deterministic(tmp_path):
    split = tmp_path / "split.json"
    split.write_text('{"version": 1}\n', encoding="utf-8")
    repository = Path(__file__).resolve().parents[2]
    kwargs = {
        "study_name": "study",
        "stage": "graph_pretrain",
        "trial_number": 3,
        "seed": 42,
        "fidelities": [1, 3],
        "params": {"gin_layers": 4},
        "fixed": {"batch_size": 2},
        "split_manifest": split,
        "repository": repository,
        "parent_checkpoint": None,
    }
    first = build_trial_manifest(**kwargs)
    second = build_trial_manifest(**kwargs)
    assert first == second
    assert first["config_hash"] == second["config_hash"]
    assert first["split_manifest"]["sha256"] == second["split_manifest"]["sha256"]


def test_top_k_and_seed_aggregation():
    rows = [
        {
            "trial_number": index,
            "seed": seed,
            "config_hash": config,
            "semantic_score": score,
            "checkpoint_path": f"/tmp/{index}.pt",
            "params": {"x": config},
        }
        for index, (config, seed, score) in enumerate([
            ("a", 17, 0.6),
            ("a", 42, 0.8),
            ("a", 73, 0.7),
            ("b", 17, 0.9),
            ("b", 42, 0.4),
            ("b", 73, 0.5),
        ])
    ]
    aggregated = aggregate_seed_rows(
        rows,
        metric="semantic_score",
        min_seeds=3,
    )
    selected = select_top_k(
        aggregated,
        metric="semantic_score",
        top_k=1,
    )
    assert selected[0]["config_hash"] == "a"
    assert selected[0]["promotion_score"] == pytest.approx(0.7)
    assert selected[0]["seed_count"] == 3


def test_config_profiles_and_cli_parsing():
    production = load_hpo_config(DEFAULT_CONFIG)
    smoke = load_hpo_config(DEFAULT_CONFIG, profile="smoke")
    assert production.stage("graph_pretrain").fidelities == (
        500,
        1500,
        4500,
        13500,
    )
    assert smoke.stage("graph_pretrain").fidelities == (1, 2, 3)
    args = build_parser().parse_args([
        "worker",
        "--stage",
        "graph_pretrain",
        "--split-manifest",
        "split.json",
    ])
    assert args.command == "worker"
    assert args.trials == 1


def test_deterministic_mock_optuna_study(tmp_path):
    storage = f"sqlite:///{tmp_path / 'study.db'}"
    assert main([
        "mock-study",
        "--config",
        str(DEFAULT_CONFIG),
        "--profile",
        "smoke",
        "--stage",
        "graph_pretrain",
        "--storage",
        storage,
        "--trials",
        "4",
    ]) == 0
    assert (tmp_path / "study.db").is_file()
