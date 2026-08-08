"""Unit tests for the model-free ``random`` eval sampling helpers.

``random`` is the model-free PI/PO bitvector baseline (no LLM). It is not
the model-based ``greedy`` tool-calling path in ``evaluate_model.py``.
"""

from __future__ import annotations

import random

from atpgllm.training._paths import ensure_data_preprocessing_on_path

ensure_data_preprocessing_on_path()

from netlist_utils import parse_range  # noqa: E402
from atpgllm.training.sampling_strategies import (  # noqa: E402
    format_nets_bit_assignment,
    format_random_answer,
    sample_bit_string,
)


def test_parse_range_bus_width_inclusive():
    """``[5:0]`` is 6 bits (abs(msb-lsb)+1), matching OptimizedNetlist."""
    assert parse_range("[5:0]") == 6
    assert parse_range("[15:0]") == 16
    assert parse_range(None) == 1


def test_sample_bit_string_length_and_alphabet():
    rng = random.Random(0)
    for n in (0, 1, 8, 64):
        s = sample_bit_string(n, rng)
        assert len(s) == n
        assert set(s) <= {"0", "1"}


def test_sample_bit_string_large_n_no_enumerate():
    """Large widths must use getrandbits, not 2**n enumeration."""
    rng = random.Random(1)
    s = sample_bit_string(256, rng)
    assert len(s) == 256
    assert set(s) <= {"0", "1"}


def test_format_nets_bit_assignment_order():
    nets = ["clk", "reset", "inputA[0]", "inputA[1]"]
    assert format_nets_bit_assignment(nets, "1010") == (
        "clk: 1, reset: 0, inputA[0]: 1, inputA[1]: 0"
    )


def test_format_random_answer_minimal_fields():
    rng = random.Random(42)
    text = format_random_answer(
        ["a", "b"],
        ["y"],
        "sa0 n1",
        rng,
    )
    assert 'INPUT_VECTOR: "' in text
    assert 'EXPECTED_OUTPUT: "' in text
    assert 'DETECTED_FAULTS: "sa0 n1"' in text
    assert "a:" in text and "b:" in text and "y:" in text
