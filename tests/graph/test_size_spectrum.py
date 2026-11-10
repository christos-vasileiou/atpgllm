"""S4 spectrum: NPN classes and size-free shares."""

import numpy as np
import pytest

from atpgllm.graph.size.netgraph import build_netgraph
from atpgllm.graph.size.spectrum import circuit_spectrum, npn_classes, spectrum_dim


def test_four_input_functions_form_222_npn_classes():
    cls, n = npn_classes()
    assert n == 222 and cls.shape == (65536,)
    # AND and NOR of two inputs are NPN equivalent, AND and XOR are not.
    and2, nor2, xor2 = 0x8888, 0x1111, 0x6666
    assert cls[and2] == cls[nor2] and cls[and2] != cls[xor2]


def _adder(bits):
    lines = [f"module add(a, b, s);", f"input [{bits-1}:0] a, b;", f"output [{bits-1}:0] s;",
             "wire " + ", ".join(f"c{i}" for i in range(bits + 1)) + ";", "assign c0 = 1'b0;"]
    for i in range(bits):
        lines.append(f"FAx1_ASAP7_75t_R U{i} (.A(a[{i}]), .B(b[{i}]), .CI(c{i}), .CON(c{i+1}), .SN(s[{i}]));")
    lines.append("endmodule")
    return "\n".join(lines)


@pytest.fixture()
def model(monkeypatch):
    monkeypatch.setenv("FAULT_SIM_BACKEND", "fast")
    from atpgllm.training.tools import simulation_model
    return simulation_model


def test_spectrum_has_fixed_length_and_unit_shares(model):
    for bits in (4, 16):
        v = circuit_spectrum(build_netgraph(model(_adder(bits))))
        assert v.shape == (spectrum_dim(),)
        assert v[:222].sum() == pytest.approx(1.0, abs=1e-5)


def test_widths_of_one_family_stay_close(model):
    a, b = (circuit_spectrum(build_netgraph(model(_adder(n))))[:222] for n in (8, 32))
    assert np.abs(a - b).sum() < 0.5
