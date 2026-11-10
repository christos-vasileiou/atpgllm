"""R6 report logic and offline model resolution, without GPUs or TetraMAX."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def r6():
    spec = importlib.util.spec_from_file_location("retest_r6", ROOT / "scripts" / "eval" / "retest_r6.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["retest_r6"] = module
    spec.loader.exec_module(module)
    return module


def record(fault="sa0 n1"):
    return {"fault": fault, "circuit_id": 1,
            "input_vector": json.dumps({"a": 1, "b": "x", "c": 0}),
            "expected_output": json.dumps({"y": 1})}


def test_location_classes(r6):
    assert r6.location(record("sa0 a")) == "pi_stem"
    assert r6.location(record("sa1 y")) == "po_stem"
    assert r6.location(record("sa0 n1")) == "internal"


def test_fill_bits_are_shared_per_row_and_only_replace_x(r6):
    rec = record()
    assert r6.fill_bits(rec, 7) == r6.fill_bits(rec, 7)
    bits = r6.fill_bits(rec, 7)
    out = r6.filled({"a": 1, "b": "x", "c": "X"}, bits)
    assert out["a"] == 1 and out["b"] == bits["b"] and out["c"] == bits["c"]


def test_shuffled_partner_is_another_circuit_and_seeded(r6):
    recs = [{"circuit_id": c} for c in (1, 1, 2, 3)]
    for seed in range(4):
        p = r6.shuffled_partner(recs, 0, seed)
        assert recs[p]["circuit_id"] != 1
        assert p == r6.shuffled_partner(recs, 0, seed)


def _loss_row(row, real, shuffled, zero=1.0):
    conditions = {"real": {"mean": real}, "zero": {"mean": zero}}
    conditions.update({f"shuffled_{s}": {"mean": v} for s, v in enumerate(shuffled)})
    return {"row": row, "circuit_id": row // 4, "conditions": conditions}


def test_loss_gate_passes_only_with_a_clear_gap(r6):
    rows = [_loss_row(i, 0.10 + 0.001 * i, [0.20 + 0.001 * i + 0.0005 * s for s in range(5)])
            for i in range(40)]
    gate = r6.loss_gate({"retest": rows})
    assert gate["pass"] and gate["gap"] == pytest.approx(0.101, abs=1e-6)
    assert gate["circuits"] == 10 and gate["circuit_bootstrap_95"][0] > 0
    # A gap carried by one circuit is not clear of zero across circuits.
    one = [_loss_row(i, 0.2, [(0.9 if i < 4 else 0.19) for _ in range(5)]) for i in range(40)]
    assert not r6.loss_gate({"retest": one})["pass"]
    same = [_loss_row(i, 0.2, [0.2 + 0.0001 * ((i + s) % 3 - 1) for s in range(5)]) for i in range(40)]
    assert not r6.loss_gate({"retest": same})["pass"]


def _scored(model, condition, row, loc, hit):
    return {"model": model, "condition": condition, "row": row, "circuit_id": row // 2, "location": loc,
            "difficulty": "easy", "parsed": True, "detects_kept": hit, "detects_filled": hit}


def test_report_detection_gate_and_matched_rows(r6):
    scored = []
    for row in range(40):
        scored.append(_scored("retest", "real", row, "internal", row % 10 < 6))
        scored.append(_scored("ports_only", "real", row, "internal", row % 10 < 3))
        if row < 4:
            scored.append(_scored("text", "full", row, "internal", True))
    report = r6.build_report(scored, {}, 0.10)
    gap = report["G3"]["internal_detection"]["kept"]
    assert gap["difference"] == pytest.approx(0.3) and gap["circuits"] == 20
    assert report["G3"]["detection_pass"] and not report["G3"]["pass"]  # no loss rows yet
    assert report["matched_rows"] == 4
    assert report["matched"]["retest/real"]["n"] == 4
    assert "retest/real" in r6.markdown(report)


def test_local_pretrained_offline_and_online(monkeypatch, tmp_path):
    from atpgllm.multimodal.loading import local_pretrained

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    assert local_pretrained("org/model") == "org/model"
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert local_pretrained(str(tmp_path)) == str(tmp_path)
    calls = {}
    import huggingface_hub

    def fake(name, local_files_only):
        calls[name] = local_files_only
        return "/snapshots/abc"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    assert local_pretrained("org/model") == "/snapshots/abc" and calls == {"org/model": True}


def test_dealt_inputs_gives_every_input_another_inputs_vector(r6):
    import torch

    nodes = torch.arange(6, dtype=torch.float32).unsqueeze(-1).repeat(1, 2)
    anchors = (torch.tensor([[0, 5, 1], [0, 6, 1], [0, 8, 3], [0, 9, 4]]), torch.ones(4))
    out = r6.dealt_inputs(nodes, anchors, seed=3)
    assert torch.equal(out[[0, 2, 5]], nodes[[0, 2, 5]])            # not inputs
    assert sorted(out[[1, 3, 4], 0].tolist()) == [1.0, 3.0, 4.0]    # the same vectors
    assert all(out[i, 0] != nodes[i, 0] for i in (1, 3, 4))         # none in place
    assert torch.equal(out, r6.dealt_inputs(nodes, anchors, seed=3))
    one = (torch.tensor([[0, 5, 1]]), torch.ones(1))
    assert torch.equal(r6.dealt_inputs(nodes, one, seed=0), nodes)


def test_report_gates_the_added_arms(r6):
    scored = []
    for row in range(40):
        scored.append(_scored("ports_only", "real", row, "internal", row % 10 < 3))
        scored.append(_scored("ports_only", "hint", row, "internal", row % 10 < 5))
        scored.append(_scored("anchored", "real", row, "internal", row % 10 < 7))
        scored.append(_scored("hint", "real", row, "internal", row % 10 < 6))
    losses = {"anchored": [_loss_row(row, 0.5, [0.6, 0.62]) for row in range(40)]}
    report = r6.build_report(scored, losses, 0.10)
    arms = report["arms"]
    assert set(arms) == {"anchored", "hint"}
    assert arms["anchored"]["internal_detection"]["kept"]["difference"] == pytest.approx(0.4)
    assert arms["anchored"]["pass"]
    assert arms["hint"]["detection_pass"] and not arms["hint"]["pass"]   # no graph to shuffle
    text = r6.markdown(report)
    assert "G3 anchored" in text and "ports_only/hint" in text


def test_value_token_means_and_their_gate(r6):
    nll = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0]]
    parts = [[None, "input_value", "input_value", "value"], [None, "value"]]
    means = r6.part_means(nll, parts)
    assert means["input_value"] == {"n": 2, "mean": 2.5}
    assert means["value"] == {"n": 4, "mean": pytest.approx(3.75)}
    assert r6.part_means([[1.0]], [[None]])["value"] == {"n": 0, "mean": None}

    def row(i, real, shuffled, value_real, value_shuffled):
        out = _loss_row(i, real, [shuffled, shuffled])
        out["conditions"]["real"]["value"] = {"n": 3, "mean": value_real}
        out["conditions"]["zero"]["value"] = {"n": 3, "mean": 2.0}
        for s in (0, 1):
            out["conditions"][f"shuffled_{s}"]["value"] = {"n": 3, "mean": value_shuffled}
        return out

    # The same loss over all tokens, a clear gap on the value tokens.
    losses = {"retest": [row(i, 0.5, 0.5, 0.2, 0.6) for i in range(40)]}
    assert not r6.loss_gate(losses)["pass"]
    value = r6.loss_gate(losses, part="value")
    assert value["pass"] and value["gap"] == pytest.approx(0.4)
    assert r6.loss_gate(losses, part="input_value") == {}       # no such tokens recorded
    report = r6.build_report([], losses, 0.10)
    assert report["G3"]["loss_value"]["pass"] and not report["G3"]["pass"]
    assert "value tokens only" in r6.markdown(report)
