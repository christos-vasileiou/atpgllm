"""Full-scan (v5) netlists in the graph modality: cut-view graph, state bits, pairing table.

Needs a data_preprocessing checkout with ``scan_view`` (branch seq-v5 or
later): run with ``DATA_PREPROCESSING_ROOT`` pointing at it.
"""

import json
import re
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch

from atpgllm.graph.dataset import record_to_pyg
from atpgllm.graph.gate_features import STATE_CELLS, GateAttributeVocab
from atpgllm.graph.models_stage1 import NetlistGraphEncoder
from atpgllm.graph.netlist_parser import (
    GraphRejected,
    chain_order,
    compute_structural_features,
    next_state,
    parse_netlist,
    parse_scan_to_graph,
    parse_verilog_to_graph,
    parsed_to_pyg,
)
from atpgllm.multimodal.data import (
    PORT_BITS,
    _compact_netlist_payload,
    port_declarations,
    ports_extra,
    scan_procedure_ports,
)
from atpgllm.training._paths import DEFAULT_SIM_CONFIG, ensure_data_preprocessing_on_path

ensure_data_preprocessing_on_path()
pytest.importorskip("scan_view", reason="data_preprocessing checkout without scan_view (needs seq-v5)")

GATE_FUNCS = json.loads(Path(DEFAULT_SIM_CONFIG).read_text())["gate_funcs"]
VOCAB = GateAttributeVocab(GATE_FUNCS)
STATE_VOCAB = GateAttributeVocab(GATE_FUNCS, state_cells=True)
# The same two rows as on branch seq-v5 (tests/unit/fixtures).
DATA = Path(__file__).parents[1] / "unit" / "fixtures"
ROW = json.loads((DATA / "v5_scan_row.json").read_text())
BUS_CLOCK_ROW = json.loads((DATA / "v5_scan_row_bus_clock.json").read_text())

COMBINATIONAL = """module m(a, b, y);
input a, b;
output y;
NAND2xp33_ASAP7_75t_R U1 (.A(a), .B(b), .Y(y));
endmodule
"""

# Both flops capture the same net, and that net is also a primary output.
SHARED_D = """module m ( clk, a, y, test_si, test_se, test_so );
  input clk, a, test_si, test_se;
  output y, test_so;
  wire q1, q2, n1;
  assign test_so = q2;
  SDFHx1_ASAP7_75t_R r1 ( .D(y), .SI(test_si), .SE(test_se), .CLK(clk), .QN(q1) );
  SDFHx1_ASAP7_75t_R r2 ( .D(y), .SI(q1), .SE(test_se), .CLK(clk), .QN(q2) );
  NAND2xp33_ASAP7_75t_R U1 ( .A(a), .B(q1), .Y(n1) );
  NAND2xp33_ASAP7_75t_R U2 ( .A(n1), .B(q2), .Y(y) );
endmodule
"""


def _names(parsed, nodes):
    return [parsed.gates[i].inst for i in nodes]


def test_scan_netlist_becomes_the_graph_of_one_capture_cycle():
    with pytest.raises(GraphRejected) as rejected:
        parse_verilog_to_graph(ROW["netlist"], GATE_FUNCS)
    assert rejected.value.reason == "sequential_cell"

    parsed = parse_netlist(ROW["netlist"], GATE_FUNCS, ROW["scan_chain"])
    # The row's own vectors name the graph's inputs and outputs, in order.
    assert parsed.inputs == list(json.loads(ROW["input_vector"]))
    assert parsed.outputs == list(json.loads(ROW["expected_output"]))
    assert parsed.is_scan and parsed.flops == ["d_reg", "s_reg"]
    assert _names(parsed, parsed.state_inputs) == ["d", "s"]        # QN nets
    assert _names(parsed, parsed.state_outputs) == ["n11", "n10"]   # D nets
    cells = {gate.cell for gate in parsed.gates}
    assert not any(cell.startswith(("SDF", "DFF")) for cell in cells)
    # Nothing synthetic and no scan or clock port is left.
    names = {gate.inst for gate in parsed.gates} | set(parsed.net_drivers)
    assert not any(n.startswith("v5_") or n in ("txclk", "test_si", "test_se", "test_so") for n in names)
    # <PI> nodes come first, in vector order: the input-bit line relies on it.
    assert [g.inst for g in parsed.gates[:len(parsed.inputs)]] == parsed.inputs
    assert all(g.cell == "<PI>" for g in parsed.gates[:len(parsed.inputs)])


def test_flops_without_a_scan_chain_stay_rejected():
    no_chain = """module m(d, clk, q);
input d, clk;
output q;
DFFHQNx1_ASAP7_75t_R U1 (.D(d), .CLK(clk), .QN(q));
endmodule
"""
    with pytest.raises(GraphRejected) as rejected:
        parse_netlist(no_chain, GATE_FUNCS)
    assert rejected.value.reason == "sequential_cell"
    assert record_to_pyg({"netlist": no_chain, "fault": "sa0 q"}, GATE_FUNCS, VOCAB) is None


def test_state_bits_follow_the_chain_table():
    netlist, table = BUS_CLOCK_ROW["netlist"], BUS_CLOCK_ROW["scan_chain"]
    assert chain_order(table) == ["out_reg_0_", "out_reg_1_"]
    with_table = parse_scan_to_graph(netlist, GATE_FUNCS, table)
    assert with_table.inputs == list(json.loads(BUS_CLOCK_ROW["input_vector"]))
    assert with_table.outputs == list(json.loads(BUS_CLOCK_ROW["expected_output"]))
    # The netlist lists the flops the other way round.
    alone = parse_scan_to_graph(netlist, GATE_FUNCS)
    assert alone.flops == ["out_reg_1_", "out_reg_0_"]
    assert alone.inputs[-2:] == ["out[1]", "out[0]"]
    with pytest.raises(GraphRejected) as rejected:
        parse_scan_to_graph(netlist, GATE_FUNCS, ROW["scan_chain"])
    assert rejected.value.reason == "chain_table_mismatch"


def test_pairing_is_a_side_table_and_the_graph_stays_acyclic():
    parsed = parse_netlist(ROW["netlist"], GATE_FUNCS, ROW["scan_chain"])
    data = parsed_to_pyg(parsed, vocab=VOCAB)
    assert data.state_pair_index.tolist() == [parsed.state_outputs, parsed.state_inputs]
    assert data.state_pair_inverted.tolist() == [True, True]
    assert data.state_in_mask.sum() == 2 and data.state_out_mask.sum() == 2
    assert bool((data.state_in_mask <= data.pi_node_mask).all())
    assert bool((data.state_out_mask <= data.po_node_mask).all())
    # State outputs are sinks and state inputs are sources: no edge closes the loop.
    src, dst = data.edge_index
    assert not data.state_out_mask[src].any()
    assert not data.state_in_mask[dst].any()
    compute_structural_features(data.edge_index, data.num_nodes)     # raises on a cycle
    # The next cycle's state comes from the table (QN is the inverse of the stored bit).
    captured = {"d": "x", "s": 0, "n11": 0, "n10": 1}
    assert next_state(parsed, captured) == {"d": 1, "s": 0}
    assert next_state(parsed, {"n10": "x"}) == {"d": "x", "s": "x"}


def test_shared_d_net_and_output_keep_one_node():
    parsed = parse_scan_to_graph(SHARED_D, GATE_FUNCS)
    assert parsed.inputs == ["a", "q1", "q2"]
    assert parsed.outputs == ["y"]
    outputs = [i for i, g in enumerate(parsed.gates) if g.cell == "<PO>"]
    assert _names(parsed, outputs) == ["y"]
    assert parsed.state_outputs == [outputs[0], outputs[0]]
    assert len(set(parsed.state_inputs)) == 2
    assert [e for e in parsed.edges if e[1] == outputs[0]] == [
        (parsed.net_drivers["y"][0], outputs[0], "Y", "<IN>", "y")]


def test_records_of_both_kinds_batch_together():
    scan = record_to_pyg(dict(ROW), GATE_FUNCS, VOCAB)
    comb = record_to_pyg({"netlist": COMBINATIONAL, "fault": "sa0 y"}, GATE_FUNCS, VOCAB)
    assert scan is not None and comb is not None
    assert scan.target_node_mask.any()
    assert comb.state_pair_index.shape == (2, 0) and not comb.state_in_mask.any()
    batch = Batch.from_data_list([comb, scan])
    offset = comb.num_nodes
    assert batch.state_pair_index.tolist() == (scan.state_pair_index + offset).tolist()
    assert bool((batch.batch[batch.state_pair_index] == 1).all())
    # A row whose vector names the state bits in another order is another circuit.
    wrong = dict(ROW, input_vector=json.dumps({"txreset": 0, "tx1": 0, "tx0": 0, "s": 0, "d": "x"}))
    assert record_to_pyg(wrong, GATE_FUNCS, VOCAB) is None


def test_state_cells_are_their_own_bin_without_moving_any_index():
    assert not VOCAB.has_state_cells and STATE_VOCAB.has_state_cells
    assert VOCAB.fingerprint == GateAttributeVocab(GATE_FUNCS).fingerprint
    assert VOCAB.with_state_cells().fingerprint == STATE_VOCAB.fingerprint
    assert STATE_VOCAB.fingerprint != VOCAB.fingerprint
    assert STATE_VOCAB.vocab_sizes == VOCAB.vocab_sizes
    assert STATE_VOCAB.function_labels == VOCAB.function_labels
    for cell in GATE_FUNCS:
        assert STATE_VOCAB.encode(cell) == VOCAB.encode(cell)
    column = STATE_VOCAB.attribute_names.index("is_sequential")
    for port, state in STATE_CELLS.items():
        differs = [i for i, (a, b) in enumerate(zip(STATE_VOCAB.encode(port), STATE_VOCAB.encode(state))) if a != b]
        assert differs == [column]
    restored = GateAttributeVocab.from_dict(STATE_VOCAB.to_dict())
    assert restored.has_state_cells and restored.fingerprint == STATE_VOCAB.fingerprint

    parsed = parse_netlist(ROW["netlist"], GATE_FUNCS, ROW["scan_chain"])
    plain = parsed_to_pyg(parsed, vocab=VOCAB)
    marked = parsed_to_pyg(parsed, vocab=STATE_VOCAB)
    state = plain.state_in_mask | plain.state_out_mask
    assert torch.equal(plain.gate_attrs[~state], marked.gate_attrs[~state])
    changed = (plain.gate_attrs[state] != marked.gate_attrs[state]).nonzero()[:, 1].unique().tolist()
    assert changed == [column]
    # A vocabulary without state cells reads a state bit as the port it acts as.
    assert plain.gate_attrs[parsed.state_inputs[0]].tolist() == list(VOCAB.encode("<PI>"))


def test_weights_trained_without_state_cells_read_them_unchanged_after_seeding():
    parsed = parse_netlist(ROW["netlist"], GATE_FUNCS, ROW["scan_chain"])
    torch.manual_seed(0)
    old = NetlistGraphEncoder(VOCAB, node_dim=16, gin_hidden_dim=16, gin_num_layers=2).eval()
    new = NetlistGraphEncoder(STATE_VOCAB, node_dim=16, gin_hidden_dim=16, gin_num_layers=2).eval()
    new.load_state_dict(old.state_dict())          # same tables, same shapes

    def encode(model, vocab):
        record = dict(ROW)
        with torch.no_grad():
            return model(Batch.from_data_list([record_to_pyg(record, GATE_FUNCS, vocab)]))["node_embs"]

    before = encode(old, VOCAB)
    assert not torch.allclose(before, encode(new, STATE_VOCAB), atol=1e-5)
    assert new.attr_encoder.seed_state_rows()
    assert torch.allclose(before, encode(new, STATE_VOCAB), atol=1e-6)
    assert parsed.is_scan


def test_stage_c_takes_the_vocabulary_of_a_state_cell_checkpoint(tmp_path):
    from atpgllm.graph.scripts.train_retest import RetestGraphModel
    from atpgllm.multimodal.loading import load_aligned_graph_stack

    args = {"node_dim": 16, "gin_layers": 2, "q_hidden": 32, "slots": 2, "q_layers": 2}
    model = RetestGraphModel(STATE_VOCAB, node_dim=16, gin_layers=2, q_hidden=32, slots=2, q_layers=2)
    path = tmp_path / "best.pt"
    torch.save({"format": "bridges_retest_v1", "model": model.state_dict(), "args": args,
                "vocab": STATE_VOCAB.to_dict(), "step": 1}, path)
    _, _, meta = load_aligned_graph_stack(path, VOCAB)
    assert meta["vocab"].has_state_cells
    torch.save({"format": "bridges_retest_v1", "model": model.state_dict(), "args": args,
                "vocab": VOCAB.to_dict(), "step": 1}, path)
    _, _, meta = load_aligned_graph_stack(path, VOCAB)
    assert not meta["vocab"].has_state_cells


def test_ports_view_names_what_a_scan_test_sets():
    netlist = ROW["netlist"]
    assert scan_procedure_ports(COMBINATIONAL) == []
    assert scan_procedure_ports(netlist) == ["test_si", "test_se", "test_so", "txclk"]
    assert port_declarations(netlist, scan_procedure_ports(netlist)) == (
        "module tx_ds_se ( txreset, tx1, tx0, d, s );\n"
        "input txreset, tx1, tx0;\n"
        "output d, s;\n"
    )
    # A one-bit bus clock leaves with its declaration.
    bus = BUS_CLOCK_ROW["netlist"]
    declared = port_declarations(bus, scan_procedure_ports(bus))
    assert "clk" not in declared and "test_" not in declared
    assert "input [1:0] in_;" in declared and "input [0:0] reset;" in declared
    # Combinational prompts are unchanged, byte for byte.
    old = "".join(f"{s.strip()};\n" for s in COMBINATIONAL.split(";")
                  if re.match(r"(module|input|output)\b", s.strip()))
    assert port_declarations(COMBINATIONAL, scan_procedure_ports(COMBINATIONAL)) == old

    graph = record_to_pyg(dict(ROW), GATE_FUNCS, VOCAB)
    extra = ports_extra(ROW, graph, port_bits=True)
    # Primary inputs, then QN nets in chain order: the order of the answer.
    assert extra == f"{PORT_BITS} txreset tx1 tx0 d s ;\n"
    _, compact = _compact_netlist_payload(ROW, "ports", extra)
    assert "txclk" not in compact and "test_se" not in compact and "SDFH" not in compact


def test_label_tables_of_both_datasets_load_as_one(tmp_path):
    from atpgllm.graph.retest_labels import fact_text, problem_targets
    from atpgllm.graph.scripts.train_retest import ProblemGraphs, load_split

    def table(directory, key, netlist, row, chain=""):
        directory.mkdir()
        t = problem_targets([row])
        problem = {"netlist_key": key, "fault": row["fault"], "patterns": t.patterns, "inputs": t.inputs,
                   "outputs": t.outputs, "care_freq": t.care_freq, "one_freq": t.one_freq,
                   "path_freq": t.path_freq, "obs_freq": t.obs_freq, "fact_text": fact_text(t),
                   "example_input_vector": row["input_vector"]}
        (directory / "problems_validation.jsonl").write_text(json.dumps(problem) + "\n")
        (directory / "circuits_validation.jsonl").write_text(
            json.dumps({"netlist_key": key, "netlist": netlist, "scan_chain": chain}) + "\n")

    comb_row = {"fault": "sa0 y", "input_vector": json.dumps({"a": 1, "b": 1}),
                "expected_output": json.dumps({"y": 0}), "fault_propagation_gates": "U1",
                "fault_propagation_nets": "y"}
    table(tmp_path / "v4", "comb", COMBINATIONAL, comb_row)
    table(tmp_path / "v5", "scan", ROW["netlist"], ROW, ROW["scan_chain"])
    problems, circuits = load_split([tmp_path / "v4", tmp_path / "v5"], "validation", 0, 0)
    assert len(problems) == 2 and set(circuits) == {"comb", "scan"}
    builder = ProblemGraphs(circuits, GATE_FUNCS, STATE_VOCAB)
    graphs = {p["netlist_key"]: builder.build(p) for p in problems}
    assert builder.rejected == 0
    scan = graphs["scan"]
    names = scan.node_names
    care = {names[i]: float(scan.care_target[i]) for i in scan.pi_node_mask.nonzero().flatten().tolist()}
    assert care == {"txreset": 1.0, "tx1": 1.0, "tx0": 1.0, "d": 0.0, "s": 1.0}
    observed = {names[i]: float(scan.obs_target[i]) for i in scan.po_target_mask.nonzero().flatten().tolist()}
    assert observed == {"d": 0.0, "s": 0.0, "n11": 0.0, "n10": 1.0}   # seen at a state output
    Batch.from_data_list(list(graphs.values()))


def test_size_stack_reads_a_scan_netlist_through_the_same_cut():
    from atpgllm.multimodal.size_stack import build_circuit_cache

    cache = build_circuit_cache(ROW["netlist"], GATE_FUNCS, STATE_VOCAB, ROW["scan_chain"])
    # Net graph (simulator model) and node graph (parser) name the same test.
    assert sorted(cache.graph.inputs) == sorted(json.loads(ROW["input_vector"]))
    assert sorted(cache.graph.outputs) == sorted(json.loads(ROW["expected_output"]))
    names = cache.pyg.node_names
    assert [names[i] for i in cache.pyg.state_in_mask.nonzero().flatten().tolist()] == ["d", "s"]
    assert cache.node_of_net["s"] == names.index("s")           # the state input drives its QN net
    assert int(cache.pyg.level.max()) > 0 and cache.pyg.local_feats.shape[0] == cache.pyg.num_nodes
    assert all(net in cache.node_of_net for net in ("n16", "n10", "n11"))
