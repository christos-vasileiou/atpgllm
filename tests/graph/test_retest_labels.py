"""Soft targets for the BRIDGES re-test, on the v4 mux example."""

import json
from pathlib import Path

import pytest

from atpgllm.graph.netlist_parser import parse_verilog_to_graph, parsed_to_pyg
from atpgllm.graph.retest_labels import (
    attach_retest_targets,
    cop_signal_probability,
    fact_text,
    group_problems,
    problem_targets,
)
from atpgllm.training._paths import DEFAULT_SIM_CONFIG

GATE_FUNCS = json.loads(Path(DEFAULT_SIM_CONFIG).read_text())["gate_funcs"]

MUX = """module mux ( o, i1, i2, s );
  input i1, i2, s;
  output o;
  wire   n2;
  INVxp33_ASAP7_75t_R U3 ( .A(s), .Y(n2) );
  AO22x1_ASAP7_75t_R U4 ( .A1(s), .A2(i2), .B1(n2), .B2(i1), .Y(o) );
endmodule
"""


def _row(vector, prop_gates="U4", prop_nets="i1, o", fault="sa0 i1", cid="c1"):
    return {
        "circuit_id": "c0",
        "netlist_id": cid,
        "fault": fault,
        "input_vector": json.dumps(vector),
        "expected_output": json.dumps({"o": 1}),
        "fault_propagation_gates": prop_gates,
        "fault_propagation_nets": prop_nets,
    }


def test_frequencies_over_patterns():
    rows = [
        _row({"i1": 1, "i2": "x", "s": 0}),
        _row({"i1": 1, "i2": 0, "s": 0}),
    ]
    t = problem_targets(rows)
    assert t.patterns == 2
    assert t.care_freq == {"i1": 1.0, "i2": 0.5, "s": 1.0}
    assert t.one_freq["i1"] == 1.0 and t.one_freq["s"] == 0.0 and t.one_freq["i2"] == 0.0
    assert t.path_freq == {"U4": 1.0}
    assert t.obs_freq == {"o": 1.0}


def test_rows_must_share_fault():
    with pytest.raises(ValueError):
        problem_targets([_row({"i1": 1, "i2": 0, "s": 0}), _row({"i1": 1, "i2": 0, "s": 0}, fault="sa1 i1")])


def test_grouping_keys_by_circuit_and_fault():
    rows = [_row({"i1": 1, "i2": 0, "s": 0}), _row({"i1": 1, "i2": 0, "s": 0}, cid="c2")]
    assert len(group_problems(rows)) == 2


class _Compiled:
    """Minimal stand-in for the fast simulator's compiled netlist."""

    input_nets = ["i1", "i2", "s"]
    instructions = [
        ("gate", "n2", [1, 0], ["A"], {"A": "s"}, {"s"}),
        # AO22: (A1 & A2) | (B1 & B2), bit j of the index is argument j.
        ("gate", "o", lambda a1, a2, b1, b2: (a1 and a2) or (b1 and b2),
         ["A1", "A2", "B1", "B2"], {"A1": "s", "A2": "i2", "B1": "n2", "B2": "i1"}, set()),
    ]


def test_cop_signal_probability():
    sp = cop_signal_probability(_Compiled())
    assert sp["n2"] == pytest.approx(0.5)
    # COP treats s and n2 as independent: 1 - (1 - 1/4)(1 - 1/4)
    assert sp["o"] == pytest.approx(1 - 0.75 * 0.75)


def test_attach_targets_on_schema_v2_graph():
    parsed = parse_verilog_to_graph(MUX, GATE_FUNCS)
    data = parsed_to_pyg(parsed)
    t = problem_targets([_row({"i1": 1, "i2": "x", "s": 0})])
    attach_retest_targets(data, parsed, t, cop_signal_probability(_Compiled()))
    names = [g.inst for g in parsed.gates]
    i1 = names.index("i1")
    i2 = names.index("i2")
    u4 = names.index("U4")
    assert data.pi_node_mask[i1] and data.care_target[i1] == 1.0 and data.one_target[i1] == 1.0
    assert data.care_target[i2] == 0.0
    assert data.gate_target_mask[u4] and data.path_target[u4] == 1.0
    po = [i for i, g in enumerate(parsed.gates) if g.cell == "<PO>"]
    assert po and data.obs_target[po[0]] == 1.0
    assert data.sp_mask.sum() >= 4


def test_fact_text_names_only_ports():
    t = problem_targets([_row({"i1": 1, "i2": "x", "s": 0})])
    text = fact_text(t, difficulty="easy")
    assert "i1=1" in text and "s=0" in text and "U4" not in text and "n2" not in text
    assert "1 of 3 inputs are free" in text


def test_cop_on_the_fast_simulator_netlist(monkeypatch):
    monkeypatch.setenv("FAULT_SIM_BACKEND", "fast")
    from atpgllm.training.tools import simulation_model

    sp = cop_signal_probability(simulation_model(MUX))
    assert sp["n2"] == pytest.approx(0.5)
    assert sp["o"] == pytest.approx(1 - 0.75 * 0.75)
