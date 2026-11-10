"""S10 trunk: locality, reach and gradients."""

import json
from pathlib import Path

import torch
from torch_geometric.data import Batch

from atpgllm.graph.gate_features import GateAttributeVocab
from atpgllm.graph.netlist_parser import parse_verilog_to_graph, parsed_to_pyg
from atpgllm.graph.size.trunk import TopoSweepTrunk, levelize
from atpgllm.training._paths import DEFAULT_SIM_CONFIG

GATE_FUNCS = json.loads(Path(DEFAULT_SIM_CONFIG).read_text())["gate_funcs"]
VOCAB = GateAttributeVocab(GATE_FUNCS)

GUEST = """module m(a, b, c, y);
input a, b, c;
output y;
wire n1;
AND2x2_ASAP7_75t_R U1 (.A(a), .B(b), .Y(n1));
OR2x2_ASAP7_75t_R U2 (.A(n1), .B(c), .Y(y));
endmodule"""

HOST_DISJOINT = """module m(a, b, c, d, e, y, z);
input a, b, c, d, e;
output y, z;
wire n1, n9;
AND2x2_ASAP7_75t_R U1 (.A(a), .B(b), .Y(n1));
OR2x2_ASAP7_75t_R U2 (.A(n1), .B(c), .Y(y));
NAND2xp33_ASAP7_75t_R U7 (.A(d), .B(e), .Y(n9));
INVx1_ASAP7_75t_R U8 (.A(n9), .Y(z));
endmodule"""

HOST_READS_GUEST = """module m(a, b, c, d, y, z);
input a, b, c, d;
output y, z;
wire n1;
AND2x2_ASAP7_75t_R U1 (.A(a), .B(b), .Y(n1));
OR2x2_ASAP7_75t_R U2 (.A(n1), .B(c), .Y(y));
NAND2xp33_ASAP7_75t_R U7 (.A(n1), .B(d), .Y(z));
endmodule"""


def _graph(text):
    data = parsed_to_pyg(parse_verilog_to_graph(text, GATE_FUNCS), vocab=VOCAB)
    TopoSweepTrunk.prepare(data)
    return data


def _vectors(trunk, text):
    data = _graph(text)
    with torch.no_grad():
        out = trunk(Batch.from_data_list([data]))["node_embs"]
    return {name: out[i] for i, name in enumerate(data.node_names)}


def _chain(first_cell, n=40):
    lines = ["module m(a, y);", "input a;", "output y;", "wire " + ", ".join(f"w{i}" for i in range(n)) + ";",
             f"{first_cell} U0 (.A(a), .Y(w0));"]
    lines += [f"BUFx2_ASAP7_75t_R U{i} (.A(w{i-1}), .Y(w{i}));" for i in range(1, n)]
    lines += ["BUFx2_ASAP7_75t_R UOUT (.A(w%d), .Y(y));" % (n - 1), "endmodule"]
    return "\n".join(lines)


def test_levels():
    level, rlevel = levelize(torch.tensor([[0, 1], [1, 2]]), 3)
    assert level.tolist() == [0, 1, 2] and rlevel.tolist() == [2, 1, 0]


def test_guest_vectors_do_not_change_inside_a_host_that_ignores_it():
    torch.manual_seed(0)
    trunk = TopoSweepTrunk(VOCAB, dim=16, per_attr_dim=4, attr_hidden=16).eval()
    alone = _vectors(trunk, GUEST)
    host = _vectors(trunk, HOST_DISJOINT)
    for name in ("a", "b", "c", "U1", "U2", "y"):
        assert torch.allclose(alone[name], host[name], atol=1e-6), name


def test_a_host_that_reads_the_guest_changes_its_observability_side():
    torch.manual_seed(0)
    trunk = TopoSweepTrunk(VOCAB, dim=16, per_attr_dim=4, attr_hidden=16).eval()
    alone = _vectors(trunk, GUEST)
    host = _vectors(trunk, HOST_READS_GUEST)
    assert not torch.allclose(alone["U1"], host["U1"], atol=1e-6)


def test_one_sweep_reaches_far_with_bounded_state():
    """40 levels away the output still sees which cell the chain starts with,
    already at random init (a 6-layer GIN gives exactly equal outputs past 6
    levels). At init the signal decays about 8% per level, so carrying it
    hundreds of levels is learned, by the gates. The state stays bounded at
    any depth."""
    torch.manual_seed(0)
    trunk = TopoSweepTrunk(VOCAB, dim=16, per_attr_dim=4, attr_hidden=16).eval()

    def run(text):
        data = _graph(text)
        with torch.no_grad():
            out = trunk(Batch.from_data_list([data]))["fwd"]
        return out[list(data.node_names).index("y")]

    inv, buf = run(_chain("INVx1_ASAP7_75t_R", 40)), run(_chain("BUFx2_ASAP7_75t_R", 40))
    assert (inv - buf).abs().max() > 1e-3
    deep = run(_chain("INVx1_ASAP7_75t_R", 300))
    assert inv.norm() < 3 * 16 ** 0.5 and deep.norm() < 3 * 16 ** 0.5


def test_gradients_flow_through_the_sweep():
    torch.manual_seed(0)
    trunk = TopoSweepTrunk(VOCAB, dim=16, per_attr_dim=4, attr_hidden=16)
    batch = Batch.from_data_list([_graph(GUEST), _graph(HOST_DISJOINT)])
    trunk(batch)["node_embs"].pow(2).mean().backward()
    assert trunk.fwd_cell.weight_ih.grad is not None and trunk.bwd_cell.weight_ih.grad.abs().sum() > 0
