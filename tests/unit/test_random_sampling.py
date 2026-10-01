"""Unit tests for the model-free ``random`` eval sampling helpers.

``random`` is the model-free PI/PO bitvector baseline (no LLM). It is not
the model-based ``single_completion`` tool-calling path in ``evaluate_model.py``.
"""

from __future__ import annotations

import random
from types import SimpleNamespace
from unittest.mock import patch

from atpgllm.training._paths import ensure_data_preprocessing_on_path

ensure_data_preprocessing_on_path()

from netlist_utils import parse_range  # noqa: E402
from atpgllm.training.sampling_strategies import (  # noqa: E402
    format_nets_bit_assignment,
    format_random_answer,
    make_strategy,
    sample_bit_string,
)
from atpgllm.training.search_types import CompletionScore, stable_seed  # noqa: E402
from atpgllm.training.search_verifier import assignment, final_fields  # noqa: E402


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


class ModelFreeVerifier:
    def __init__(self, input_nets=("a", "b", "c")):
        self.input_nets = input_nets
        self.scored = []
        self.simulated = []

    def problem(self, prompt, record):
        return SimpleNamespace(problem_id="packed-draw", input_nets=self.input_nets,
                               output_nets=("y", "z"), fault="sa0 y")

    @staticmethod
    def failure(status):
        return CompletionScore(status=status)

    def score_state(self, problem, state, context):
        self.scored.append(state.final_answer)
        vector = assignment(final_fields(state.final_answer)["INPUT_VECTOR"], problem.input_nets)
        return CompletionScore(), vector

    def simulate(self, problem, vector, context):
        self.simulated.append(dict(vector))
        return SimpleNamespace(loc={("y", "Good Machine"): vector["a"],
                                    ("z", "Good Machine"): 1 - vector["a"]}), {}


def test_random_strategy_uses_one_packed_draw_per_vector_and_slot():
    verifier = ModelFreeVerifier()
    expected = [format_random_answer(
        verifier.input_nets, ("y", "z"), "sa0 y",
        random.Random(stable_seed(42, "packed-draw", slot)),
    ) for slot in range(2)]
    widths = []

    class CountingRandom(random.Random):
        def getrandbits(self, n):
            widths.append(n)
            return super().getrandbits(n)

    strategy = make_strategy("random", None, verifier, num_completions=2, width=10)
    with patch("atpgllm.training.search_types.random.Random", CountingRandom):
        result, = strategy.sample_batch(["prompt"], [{}])
    assert result.completions == expected == verifier.scored
    assert widths == [3, 2, 3, 2]
    assert verifier.simulated == []
    assert [slot["usage"]["attempts"] for slot in result.slots] == [1, 1]
    assert [slot["unique_vectors"] for slot in result.slots] == [1, 1]
    assert all(slot["answer_source"] == "random" for slot in result.slots)


def test_vector_evolutionary_initialization_draws_only_packed_inputs():
    verifier = ModelFreeVerifier()
    widths = []

    class CountingRandom(random.Random):
        def getrandbits(self, n):
            widths.append(n)
            return super().getrandbits(n)

    strategy = make_strategy("vector_evolutionary", None, verifier, num_completions=1, width=2)
    with patch("atpgllm.training.search_types.random.Random", CountingRandom):
        result, = strategy.sample_batch(["prompt"], [{}])
    assert widths == [3, 3]
    assert verifier.simulated
    for text, vector in zip(verifier.scored, verifier.simulated):
        fields = final_fields(text)
        assert assignment(fields["INPUT_VECTOR"], verifier.input_nets) == vector
        assert assignment(fields["EXPECTED_OUTPUT"], ("y", "z")) == {
            "y": vector["a"], "z": 1 - vector["a"],
        }
    assert result.slots[0]["answer_source"] == "simulator"


def test_vector_evolutionary_mutates_without_drawing_an_unused_vector():
    verifier = ModelFreeVerifier(input_nets=("a",))
    strategy = make_strategy("vector_evolutionary", None, verifier, num_completions=1,
                             width=2, search_config={"population_size": 1})
    with patch("atpgllm.training.sampling_strategies.sample_bit_string",
               wraps=sample_bit_string) as draw:
        result, = strategy.sample_batch(["prompt"], [{}])
    assert draw.call_count == 1
    assert len(verifier.simulated) == 2
    assert verifier.simulated[1]["a"] == 1 - verifier.simulated[0]["a"]
    assert result.slots[0]["unique_vectors"] == 2
    assert result.slots[0]["usage"]["attempts"] == 2
