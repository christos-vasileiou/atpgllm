"""S1 view, S2 side tokens and COP observability on tiny netlists."""

import pytest

from atpgllm.graph.size.fovea import build_view, effect_chain_covered
from atpgllm.graph.size.netgraph import build_netgraph, clean_netlist, transitive_fanin

TWO_GATES = """// Created by: Synopsys DC Ultra(TM) in wire load mode
module m(a, b, c, y);
input a, b, c;
output y;
wire n1;
AND2x2_ASAP7_75t_R U1 (.A(a), .B(b), .Y(n1));
OR2x2_ASAP7_75t_R U2 (.A(n1), .B(c), .Y(y));
endmodule"""


@pytest.fixture()
def g(monkeypatch):
    monkeypatch.setenv("FAULT_SIM_BACKEND", "fast")
    from atpgllm.training.tools import simulation_model

    return build_netgraph(simulation_model(clean_netlist(TWO_GATES)))


def test_clean_netlist_drops_the_banner():
    assert "DC Ultra" not in clean_netlist(TWO_GATES)


def test_cop_controllability_and_observability(g):
    assert g.sp["n1"] == pytest.approx(0.25)
    assert g.sp["y"] == pytest.approx(1 - 0.75 * 0.5)
    assert g.obs["y"] == 1.0
    assert g.obs["n1"] == pytest.approx(0.5)      # OR passes n1 when c = 0
    assert g.obs["a"] == pytest.approx(0.25)      # and AND passes a when b = 1
    assert g.level["y"] == 2 and g.depth == 2
    assert transitive_fanin(g, "y") == {"y", "n1", "c", "a", "b"}


def test_view_has_fanin_path_and_side_inputs(g):
    v = build_view(g, "n1", cap=16, k=2, mode="paths", fanin_share=0.5)
    assert {"n1", "a", "b", "y"} <= v.net_set
    assert v.paths[0] == ["n1", "y"]
    assert v.side == ["c"]
    assert not v.fanin_truncated and v.tokens == 5


def test_fanin_cap_moves_boundary_to_side_tokens(g):
    v = build_view(g, "y", cap=2, k=1, fanin_share=0.5, mode="paths")
    assert v.fanin_truncated
    assert set(v.side) <= {"n1", "c"} and v.side


def test_check1_needs_side_inputs_covered(g):
    full = build_view(g, "n1", cap=16, k=1)
    assert effect_chain_covered(g, full, ["n1", "y"], ["y"]) is True
    blind = build_view(g, "n1", cap=16, k=1, side_cap=0)
    assert effect_chain_covered(g, blind, ["n1", "y"], ["y"]) is False
    assert effect_chain_covered(g, full, ["n1"], []) is None


ALIASED = """module m(a, b, c, y);
input a, b, c;
output y;
wire n1, n2;
AND2x2_ASAP7_75t_R U1 (.A(a), .B(b), .Y(n1));
OR2x2_ASAP7_75t_R U2 (.A(n1), .B(c), .Y(n2));
assign y = n2;
endmodule"""


def test_check1_resolves_outputs_that_are_aliases(monkeypatch):
    monkeypatch.setenv("FAULT_SIM_BACKEND", "fast")
    from atpgllm.training.tools import simulation_model

    g = build_netgraph(simulation_model(ALIASED))
    assert g.resolve("y") == "n2" and "n2" in g.outputs
    v = build_view(g, "n1", cap=16, k=1)
    assert effect_chain_covered(g, v, ["n1", "n2", "y"], ["y"]) is True


def test_cone_mode_takes_the_nearest_fanout_first(g):
    v = build_view(g, "n1", cap=16, mode="cone")
    assert "y" in v.net_set and v.side == ["c"]
