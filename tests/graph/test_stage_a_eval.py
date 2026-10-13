"""Stage-A validation: size buckets, the structural rule, grouped metrics."""

import json
from pathlib import Path

import torch
from torch_geometric.data import Batch

from atpgllm.graph.dataset import record_to_pyg
from atpgllm.graph.gate_features import GateAttributeVocab
from atpgllm.graph.pretrain import GraphPretrainingModel
from atpgllm.graph.stage_a_eval import (
    build_validation_set,
    evaluate_stage_a,
    rule_baseline,
    size_bucket,
    structural_rule_scores,
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


def _record(fault="sa0 n1", netlist=TWO_GATES, difficulty="easy", propagation="U1, U2"):
    return {
        "netlist": netlist,
        "fault": fault,
        "fault_propagation_gates": propagation,
        "backtrack_gates": "U1",
        "snapshot": str({
            "Good Machine": {"a": 1, "b": 1, "c": 0, "n1": 1, "y": 1},
            "Bad Machine": {"a": 1, "b": 1, "c": 0, "n1": 0, "y": 0},
        }),
        "difficulty": difficulty,
    }


def test_structural_rule_follows_the_fault_cone():
    graph = record_to_pyg(_record(), GATE_FUNCS, VOCAB)
    names = graph.node_names
    rule = structural_rule_scores(graph)
    downstream = {n for n, s in zip(names, rule["propagation"].tolist()) if s > 0}
    upstream = {n for n, s in zip(names, rule["backtrack"].tolist()) if s > 0}
    assert downstream == {"U1", "U2", "y"}
    assert upstream == set(names)
    # Closer to the fault ranks higher.
    score = dict(zip(names, rule["propagation"].tolist()))
    assert score["U1"] > score["U2"] > score["y"]


def test_validation_set_respects_bucket_and_design_caps():
    other = TWO_GATES.replace("module m(", "module m2(").replace("OR2x2", "NOR2xp33")
    records = [_record("sa0 n1"), _record("sa1 n1"), _record("sa0 y"), _record("sa0 n1", other)]
    items, rejections, scanned = build_validation_set(
        records, GATE_FUNCS, VOCAB, per_bucket=10, per_design=2, scan_limit=100
    )
    assert scanned == 4 and not rejections
    assert len(items) == 3  # the third fault on TWO_GATES exceeds per_design
    assert {item.size for item in items} == {"<=50 gates"}
    assert size_bucket(items[0].graph) == "<=50 gates"

    items, _, scanned = build_validation_set(
        records, GATE_FUNCS, VOCAB, per_bucket=1, per_design=2, scan_limit=100
    )
    # Scanning continues while the other size buckets are still empty.
    assert len(items) == 1 and scanned == 4


def test_validation_set_caps_per_difficulty():
    records = [_record("sa0 n1", difficulty="easy"), _record("sa1 n1", difficulty="easy"),
               _record("sa0 y", difficulty="hard")]
    # Size-bucket cap of 1: the easy majority would take the only slot.
    items, _, _ = build_validation_set(
        records, GATE_FUNCS, VOCAB, per_bucket=1, per_design=2, scan_limit=100
    )
    assert [item.difficulty for item in items] == ["easy"]
    # Per (size, difficulty) cap of 1: the hard fault gets its own slot.
    items, _, _ = build_validation_set(
        records, GATE_FUNCS, VOCAB, per_bucket=1, per_design=1, scan_limit=100,
        per_difficulty=1,
    )
    assert [item.difficulty for item in items] == ["easy", "hard"]


def test_grouped_metrics_and_lift_over_rule():
    items, _, _ = build_validation_set(
        # Only U2 labelled: AP is undefined when every gate is positive.
        [_record(difficulty="easy", propagation="U2"),
         _record("sa1 n1", difficulty="hard", propagation="U2")],
        GATE_FUNCS, VOCAB, per_bucket=10, per_design=2, scan_limit=10,
    )
    baseline = rule_baseline(items)
    assert baseline["by_size"]["<=50 gates"]["graphs"] == 2
    assert set(baseline["by_difficulty"]) == {"easy", "hard"}
    # The rule ranks U1 (distance 0) above the labelled U2 (distance 1).
    assert baseline["overall"]["propagation_ap"] == 0.5

    torch.manual_seed(0)
    model = GraphPretrainingModel(
        VOCAB, node_dim=8, gin_hidden_dim=8, gin_num_layers=1, dropout=0.0
    )
    report = evaluate_stage_a(
        model, items, device=torch.device("cpu"), batch_size=2, baseline=baseline
    )
    assert set(report) >= {"overall", "by_size", "by_difficulty", "score", "lift_vs_rule"}
    # Two graphs is below MIN_BUCKET_GRAPHS, so no balanced score yet.
    assert report["score"] is None
    lift = report["lift_vs_rule"]["<=50 gates"]["propagation"]
    assert lift == report["by_size"]["<=50 gates"]["propagation_ap"] - 0.5
    assert model.training  # evaluation restores train mode


def test_circuit_sampling_caps_and_mixes(tmp_path):
    import collections
    import pyarrow as pa
    import pyarrow.parquet as pq
    from atpgllm.graph.circuit_sampling import choose_rows, load_rows

    # Two files, rows grouped by circuit; circuit "big" dominates.
    rows = [("big", i) for i in range(90)] + [("a", i) for i in range(5)] + [("b", i) for i in range(5)]
    files = []
    for part, chunk in enumerate((rows[:60], rows[60:])):
        path = tmp_path / f"train-{part}.parquet"
        pq.write_table(pa.table({
            "circuit_id": [c for c, _ in chunk],
            "netlist": [f"netlist-{c}" for c, _ in chunk],
            "fault": [f"sa0 n{i}" for _, i in chunk],
        }), path)
        files.append(str(path))

    picks = choose_rows(files, per_circuit=10, total=None, seed=0)
    assert len(picks) == 20 and len(set(picks)) == 20
    records = load_rows(files, picks, columns=("circuit_id", "netlist", "fault"))
    per_circuit = collections.Counter(r["circuit_id"] for r in records)
    assert per_circuit == {"big": 10, "a": 5, "b": 5}
    # Random within the big circuit, not its first rows.
    big_faults = {r["fault"] for r in records if r["circuit_id"] == "big"}
    assert big_faults != {f"sa0 n{i}" for i in range(10)}
    # One netlist object per circuit.
    big = [r["netlist"] for r in records if r["circuit_id"] == "big"]
    assert all(n is big[0] for n in big)
    assert choose_rows(files, per_circuit=10, total=7, seed=0) == picks[:7]

    # Grouping by (circuit, difficulty) keeps a circuit's rare label.
    path = tmp_path / "graded.parquet"
    pq.write_table(pa.table({
        "circuit_id": ["c"] * 12,
        "difficulty": ["easy"] * 10 + ["hard"] * 2,
    }), path)
    by_circuit = choose_rows([str(path)], per_circuit=2, total=None, seed=0)
    graded = choose_rows([str(path)], per_circuit=2, total=None, seed=0,
                         group_by=("circuit_id", "difficulty"))
    assert len(by_circuit) == 2 and len(graded) == 4
    assert {row for _, row in graded} >= {10, 11}


def test_sampled_columns_cover_every_record_field_the_graph_reads():
    # Fields attach_atpg_context reads; a missing column silently skips a check.
    from atpgllm.graph.circuit_sampling import GRAPH_COLUMNS

    assert {"netlist", "fault", "input_vector", "fault_propagation_gates",
            "backtrack_gates", "snapshot", "difficulty"} <= set(GRAPH_COLUMNS)


def test_split_files_reads_a_local_dataset_directory(tmp_path):
    import pyarrow as pa
    import pytest
    import pyarrow.parquet as pq

    from atpgllm.graph.circuit_sampling import split_files

    for split in ("train", "validation"):
        (tmp_path / split).mkdir()
        for index in (1, 0):
            pq.write_table(pa.table({"circuit_id": ["c"]}),
                           tmp_path / split / f"data-{index:05d}.parquet")
    files = split_files(str(tmp_path), "validation", None)
    assert [f.rsplit("/", 2)[-2:] for f in files] == [
        ["validation", "data-00000.parquet"], ["validation", "data-00001.parquet"],
    ]
    with pytest.raises(ValueError):
        split_files(str(tmp_path), "train", "some-revision")
    with pytest.raises(FileNotFoundError):
        split_files(str(tmp_path), "test", None)


def test_rows_keep_their_own_netlist_within_a_circuit(tmp_path):
    # A circuit_id groups structurally equal netlists with different port
    # names; each row must keep the netlist its vector was written for.
    import pyarrow as pa
    import pyarrow.parquet as pq

    from atpgllm.graph.circuit_sampling import load_rows

    path = tmp_path / "data-00000.parquet"
    pq.write_table(pa.table({
        "circuit_id": ["c", "c", "c"],
        "netlist": ["module m(a, y);", "module m(b, y);", "module m(a, y);"],
    }), path)
    rows = load_rows([str(path)], [(0, 0), (0, 1), (0, 2)], columns=("circuit_id", "netlist"))
    assert [r["netlist"] for r in rows] == ["module m(a, y);", "module m(b, y);", "module m(a, y);"]
    assert rows[0]["netlist"] is rows[2]["netlist"]   # identical text is still shared
