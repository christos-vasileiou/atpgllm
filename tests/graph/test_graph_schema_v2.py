"""Graph schema v2: circuit meaning survives parsing and encoding.

Counterexamples are from docs/GRAPH_MODALITY_REVIEW_20260919.md (#1, #2).
"""

import collections
import json
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch

from atpgllm.graph.dataset import record_to_pyg
from atpgllm.graph.gate_features import GateAttributeVocab
from atpgllm.graph.models_stage1 import NetlistGraphEncoder
from atpgllm.graph.netlist_parser import GraphRejected, parse_verilog_to_graph
from atpgllm.graph.pretrain import GraphPretrainingModel
from atpgllm.training._paths import DEFAULT_SIM_CONFIG, ensure_data_preprocessing_on_path


GATE_FUNCS = json.loads(Path(DEFAULT_SIM_CONFIG).read_text())["gate_funcs"]
VOCAB = GateAttributeVocab(GATE_FUNCS)

AO21 = """module m(a, b, c, y);
input a, b, c;
output y;
AO21x1_ASAP7_75t_R U1 (.A1(a), .A2(b), .B(c), .Y(y));
endmodule"""

TWO_GATES = """module m(a, b, c, y);
input a, b, c;
output y;
wire n1;
AND2x2_ASAP7_75t_R U1 (.A(a), .B(b), .Y(n1));
OR2x2_ASAP7_75t_R U2 (.A(n1), .B(c), .Y(y));
endmodule"""


def _graph(netlist, fault="sa0 y", rejections=None):
    return record_to_pyg(
        {"netlist": netlist, "fault": fault}, GATE_FUNCS, VOCAB,
        rejections=rejections,
    )


def _embed(*graphs):
    torch.manual_seed(0)
    encoder = NetlistGraphEncoder(
        VOCAB, node_dim=16, gin_hidden_dim=16, gin_num_layers=2,
        per_attr_dim=4, attr_mlp_hidden=16, dropout=0.0,
    ).eval()
    with torch.no_grad():
        return [encoder(Batch.from_data_list([g]))["graph_embs"] for g in graphs]


def test_library_cells_with_equal_features_have_equal_functions():
    for cells in VOCAB.unique_combinations().values():
        assert len({VOCAB.get_attrs(cell).function for cell in cells}) == 1, cells
    assert VOCAB.encode("MAJx2_ASAP7_75t_R") != VOCAB.encode("MAJIxp5_ASAP7_75t_R")
    assert VOCAB.encode("AO211x2_ASAP7_75t_R") != VOCAB.encode("AO22x1_ASAP7_75t_R")
    # Drive variants of one function still share features.
    assert VOCAB.encode("AND2x2_ASAP7_75t_R") == VOCAB.encode("AND2x4_ASAP7_75t_R")


def test_vocab_round_trip_and_old_schema_rejected():
    payload = VOCAB.to_dict()
    assert GateAttributeVocab.from_dict(payload).fingerprint == VOCAB.fingerprint
    with pytest.raises(ValueError):
        GateAttributeVocab.from_dict({**payload, "schema_version": 1})


def test_pin_roles_and_input_order_change_the_encoding():
    # (a&b)|c vs (a&c)|b: with a=0,b=0,c=1 only the first exposes sa0 y.
    swapped = AO21.replace(".A2(b), .B(c)", ".A2(c), .B(b)")
    first, second = _embed(_graph(AO21), _graph(swapped))
    assert not torch.allclose(first, second)


def test_primary_input_faults_are_distinct():
    sa0_a, sa0_b = _embed(_graph(AO21, "sa0 a"), _graph(AO21, "sa0 b"))
    assert not torch.allclose(sa0_a, sa0_b)


def test_encoding_is_invariant_to_statement_order():
    lines = TWO_GATES.splitlines()
    reordered = "\n".join(lines[:5] + [lines[6], lines[5]] + lines[7:])
    assert reordered != TWO_GATES
    first, second = _embed(_graph(TWO_GATES), _graph(reordered))
    assert torch.allclose(first, second, atol=1e-6)


def test_ports_follow_simulator_bit_order():
    ensure_data_preprocessing_on_path()
    from fault_sim import OptimizedNetlist
    from atpgllm.training.reward_function_factory import RewardFunctionFactory as RF

    netlist = """module m(a, y);
input [7:4] a;
output [1:0] y;
AND2x2_ASAP7_75t_R U1 (.A(a[4]), .B(a[7]), .Y(y[0]));
INVx1_ASAP7_75t_R U2 (.A(a[5]), .Y(y[1]));
endmodule"""
    parsed = parse_verilog_to_graph(netlist, GATE_FUNCS)
    sim = OptimizedNetlist(netlist, GATE_FUNCS, RF.DECL_RE, RF.NAME_RE)
    assert parsed.inputs == sim.input_nets == ["a[4]", "a[5]", "a[6]", "a[7]"]
    assert parsed.outputs == sim.output_nets
    graph = _graph(netlist, "sa0 y[0]")
    names = graph.node_names
    position = graph.structural_feats[:, -1].tolist()
    assert [position[names.index(n)] for n in sim.input_nets] == pytest.approx(
        [0.0, 1 / 3, 2 / 3, 1.0]
    )


def test_assign_alias_and_constants_keep_connections():
    netlist = """module m(a, y, z);
input a;
output y, z;
wire n, t;
INVx1_ASAP7_75t_R U1 (.A(a), .Y(n));
assign t = n;
NAND2xp5_ASAP7_75t_R U2 (.A(t), .B(*Logic1*), .Y(y));
assign z = *Logic0*;
endmodule"""
    graph = _graph(netlist, "sa0 t")
    names = graph.node_names
    edges = {(names[s], names[d]) for s, d in graph.edge_index.t().tolist()}
    assert {("a", "U1"), ("U1", "t"), ("t", "U2"), ("<CONST1>", "U2"),
            ("U2", "y"), ("<CONST0>", "z")} <= edges
    # A fault on the alias marks only the alias's own fanout edge.
    marked = [
        (names[s], names[d])
        for (s, d), flag in zip(graph.edge_index.t().tolist(), graph.fault_edge.tolist())
        if flag
    ]
    assert marked == [("t", "U2")]


@pytest.mark.parametrize(
    ("netlist", "fault", "reason"),
    [
        (AO21.replace("AO21x1_ASAP7_75t_R", "MYSTERYx1"), "sa0 y", "unknown_cell"),
        ("""module m(d, clk, q);
input d, clk;
output q;
DFFHQNx1_ASAP7_75t_R U1 (.D(d), .CLK(clk), .QN(q));
endmodule""", "sa0 q", "sequential_cell"),
        ("""module m(a, b, y);
input a, b;
output y;
wire n2;
NAND2xp5_ASAP7_75t_R U1 (.A(a), .B(n2), .Y(y));
NAND2xp5_ASAP7_75t_R U2 (.A(b), .B(y), .Y(n2));
endmodule""", "sa0 y", "cycle"),
        (AO21.replace(".B(c)", ".B(floating)"), "sa0 y", "undriven_net"),
        (AO21.replace("endmodule", "assign y = a & b;\nendmodule"), "sa0 y",
         "unsupported_assign"),
        (AO21, "sa0 nowhere", "fault_net_unknown"),
        (AO21, "stuck somewhere", "fault_unparsable"),
    ],
)
def test_unrepresentable_records_are_rejected_with_a_reason(netlist, fault, reason):
    rejections = collections.Counter()
    assert _graph(netlist, fault, rejections) is None
    assert rejections == {reason: 1}


def test_stage_a_heads_are_scored_on_meaningful_nodes():
    graph = Batch.from_data_list([_graph(TWO_GATES, "sa0 n1")])
    names = graph.node_names[0]
    gates = [names[i] for i in GraphPretrainingModel.scored_nodes("propagation", graph).nonzero().flatten()]
    drivers = [names[i] for i in GraphPretrainingModel.scored_nodes("discrepancy", graph).nonzero().flatten()]
    assert gates == ["U1", "U2"]
    assert drivers == ["a", "b", "c", "U1", "U2"]


def test_record_for_another_circuit_is_rejected():
    rejections = collections.Counter()
    good = {"netlist": AO21, "fault": "sa0 y", "input_vector": '{"a": 0, "b": 0, "c": 1}'}
    assert record_to_pyg(good, GATE_FUNCS, VOCAB, rejections=rejections) is not None
    swapped = {**good, "input_vector": '{"ina[0]": 0, "inb[0]": 1, "c": 1}'}
    reordered = {**good, "input_vector": '{"b": 0, "a": 0, "c": 1}'}
    assert record_to_pyg(swapped, GATE_FUNCS, VOCAB, rejections=rejections) is None
    assert record_to_pyg(reordered, GATE_FUNCS, VOCAB, rejections=rejections) is None
    assert rejections == {"vector_port_mismatch": 2}
