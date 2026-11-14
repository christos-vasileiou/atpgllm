"""The latent alignment test end to end on CPU, with a tiny language model in Granite's place."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
ROW = json.loads((ROOT / "tests" / "unit" / "fixtures" / "v5_scan_row.json").read_text())
COMB = """module m(a, b, c, y);
input [1:0] a;
input b, c;
output y;
wire n1;
AND2x2_ASAP7_75t_R U1 (.A(a[0]), .B(a[1]), .Y(n1));
AOI21xp5_ASAP7_75t_R U2 (.A1(n1), .A2(b), .B(c), .Y(y));
endmodule
"""


@pytest.fixture(scope="module")
def la():
    spec = importlib.util.spec_from_file_location("latent_alignment", ROOT / "scripts" / "train" / "latent_alignment.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["latent_alignment"] = module
    spec.loader.exec_module(module)
    return module


def test_hint_agreement_measures(la):
    assert la.assignments("a=1 b[2]=0 none") == {"a=1", "b[2]=0"}
    assert la.f1({"a=1", "b=0"}, {"a=1", "c=0"}) == pytest.approx(0.5)
    assert la.f1(set(), set()) == 1.0 and la.f1({"a=1"}, set()) == 0.0


def _tables(root: Path):
    """Two circuits, three faults each, the same rows for train and validation."""
    labels, hints = root / "labels", root / "hints"
    labels.mkdir(), hints.mkdir()
    circuits = [{"netlist_key": "0a", "netlist": ROW["netlist"], "scan_chain": ROW["scan_chain"]},
                {"netlist_key": "0b", "netlist": COMB, "scan_chain": ""}]
    scan_vector, comb_vector = ROW["input_vector"], json.dumps({"a[0]": 1, "a[1]": 1, "b": 0, "c": 0})
    problems = [("0a", "sa1 n16", scan_vector, {"txreset": 0, "tx1": 0, "s": 0}),
                ("0a", "sa0 n10", scan_vector, {"txreset": 0, "tx0": 1}),
                ("0a", "sa1 \\add/n13", scan_vector, {"tx1": 1}),       # not a net: rejected by the graph
                ("0b", "sa0 n1", comb_vector, {"a[0]": 1, "a[1]": 1, "c": 0}),
                ("0b", "sa1 n1", comb_vector, {"a[0]": 0, "c": 0}),
                ("0b", "sa0 y", comb_vector, {})]
    for split in ("train", "validation"):
        (labels / f"circuits_{split}.jsonl").write_text("".join(json.dumps(c) + "\n" for c in circuits))
        (labels / f"problems_{split}.jsonl").write_text("".join(
            json.dumps({"netlist_key": key, "netlist_id": "n", "circuit_id": "c", "fault": fault,
                        "example_input_vector": vector}) + "\n" for key, fault, vector, _ in problems))
        (hints / f"{split}_000.jsonl").write_text("".join(
            json.dumps({"netlist_key": key, "fault": fault, "hint": hint}) + "\n"
            for key, fault, _, hint in problems[:5]))            # the last problem has no hint
    return labels, hints


@pytest.mark.parametrize("fusion,train", [("prefix", "projector"), ("anchored", "projector"), ("both", "graph")])
def test_alignment_runs_and_scores_real_wrong_and_no_graph(la, tmp_path, fusion, train):
    from atpgllm.training._paths import ensure_data_preprocessing_on_path

    ensure_data_preprocessing_on_path()
    pytest.importorskip("scan_view", reason="needs a data_preprocessing checkout with scan_view")
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    from atpgllm.graph.gate_features import GateAttributeVocab
    from atpgllm.graph.models_stage1 import NetlistGraphEncoder
    from atpgllm.graph.retest_qformer import RoleQueryFormer
    from atpgllm.multimodal import GraphConditionedCausalLM
    from atpgllm.multimodal.loading import local_pretrained
    from atpgllm.training._paths import DEFAULT_SIM_CONFIG

    try:
        tokenizer = AutoTokenizer.from_pretrained(local_pretrained("ibm-granite/granite-4.2-8b"))
    except Exception as exc:      # no local copy of the tokenizer
        pytest.skip(f"Granite tokenizer not available: {exc}")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    gate_funcs = json.loads(Path(DEFAULT_SIM_CONFIG).read_text())["gate_funcs"]
    vocab = GateAttributeVocab(gate_funcs, state_cells=True)

    def factory(args):
        torch.manual_seed(0)
        llm = LlamaForCausalLM(LlamaConfig(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
                                           num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2))
        for parameter in llm.parameters():
            parameter.requires_grad = False
        encoder = NetlistGraphEncoder(vocab, node_dim=16, gin_hidden_dim=16, gin_num_layers=2)
        former = RoleQueryFormer(d_node=16, hidden_dim=32, slots_per_role=2, num_layers=2)
        model = GraphConditionedCausalLM(encoder, former, llm, prefix_at="placeholder",
                                         freeze_graph_batchnorm=True, fusion=args.fusion)
        model.set_graph_policy("full" if args.train == "graph" else "frozen")
        return model, tokenizer, gate_funcs, vocab, torch.device("cpu")

    labels, hints = _tables(tmp_path)
    out = tmp_path / "out"
    la.main(["--alignment", "unused", "--labels", str(labels), "--hints", str(hints), "--out", str(out),
             "--fusion", fusion, "--train", train, "--steps", "2", "--batch-size", "2", "--eval-every", "1",
             "--max-eval", "8", "--decode", "2", "--warmup", "1"], model_factory=factory)
    rows = [json.loads(line) for line in (out / "metrics.jsonl").open()]
    start = rows[0]
    # The problem without a hint is not loaded; the one the graph rejects stays out of every batch.
    assert start["train_problems"] == 5 and start["eval_problems"] >= 3 and start["trainable_parameters"] > 0
    evals = [r["eval"] for r in rows if "eval" in r]
    assert len(evals) == 3 and all(e["rows"] >= 3 for e in evals)
    last = evals[-1]
    for key in ("loss_real", "loss_wrong_fault", "loss_none", "gap_wrong_minus_real"):
        assert last[key] == last[key]                      # a number
    assert "decode_real" not in evals[0]
    assert last["decode_real"]["rows"] == 2 and 0.0 <= last["decode_real"]["own"] <= 1.0
    if fusion == "prefix":
        # With no graph the slot is zeros: the loss differs from the real graph's at once.
        assert evals[0]["loss_none"] != evals[0]["loss_real"]
    else:
        assert "anchor_gate" in last and last["anchor_gate"] != 0.0     # the gate moved
    saved = torch.load(out / "graph_side.pt", weights_only=False)
    assert ("q_former" in saved) == (train == "graph")
    assert ("port_anchor" in saved) == (fusion != "prefix")
