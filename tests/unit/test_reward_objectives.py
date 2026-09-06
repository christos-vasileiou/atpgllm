from types import SimpleNamespace

import pandas as pd
import pytest

from atpgllm.llm.reward_funcs import (
    ATPG_GDPO_OBJECTIVE_KEYS,
    _fault_detected_at_pos,
    test_generation_grpo_reward as score_generation,
    train_scalar_from_reward_components,
)
from fault_sim import fast_fault_sim


def _extract(label):
    def extract(text):
        marker = f'{label}:"'
        if marker not in text:
            return []
        return [text.split(marker, 1)[1].split('"', 1)[0]]

    return extract


def _simulation(*, detected=True):
    return pd.DataFrame(
        {
            "Good Machine": [1, 0, 1, 1],
            "Bad Machine": [1, 0, 0, 0 if detected else 1],
            "PIs": [True, True, False, False],
            "POs": [False, False, False, True],
            "Fault Propagation Path": [False, False, True, detected],
            "Backtrack Sensitizing Inputs": [True, True, False, False],
        },
        index=["a", "b", "n_fault", "y"],
    )


def _score(completion, *, detected=True):
    calls = []

    def simulator(input_vector, expected_output, fault, netlist, gate_func, **kwargs):
        calls.append((input_vector, expected_output, fault))
        return _simulation(detected=detected), {}

    netlist = SimpleNamespace(input_nets=["a", "b"], output_nets=["y"])
    rewards = score_generation(
        ["target sa0 n_fault"],
        [completion],
        netlists=[netlist],
        fault_fn=lambda _: [("sa0", "n_fault")],
        simulation_fn=lambda _: [],
        input_vector_fn=_extract("INPUT_VECTOR"),
        expected_output_fn=_extract("EXPECTED_OUTPUT"),
        detected_faults_fn=_extract("DETECTED_FAULTS"),
        lib_gate_funcs={},
        fault_sim=simulator,
    )
    return rewards[0], calls


def test_reward_exposes_only_four_training_objectives():
    reward, _ = _score(
        'INPUT_VECTOR:"a:1" EXPECTED_OUTPUT:"y:1" '
        'DETECTED_FAULTS:"sa0 n_fault"'
    )

    training_keys = [key for key in reward if not key.endswith("_logonly")]
    assert tuple(training_keys) == ATPG_GDPO_OBJECTIVE_KEYS
    assert reward["detection"] == 1.0
    assert reward["activation"] == 1.0
    assert reward["fidelity"] == 1.0
    assert reward["format"] == pytest.approx(0.75)
    assert reward["pi_completeness_logonly"] == 0.5


def test_easy_auxiliaries_are_zero_when_fault_is_not_detected():
    reward, _ = _score(
        'INPUT_VECTOR:"a:1,b:0" EXPECTED_OUTPUT:"y:1" '
        'DETECTED_FAULTS:"sa0 n_fault"',
        detected=False,
    )

    assert reward["detection"] == 0.0
    assert reward["activation"] == 1.0
    assert reward["fidelity"] == 0.0
    assert reward["format"] == 0.0
    assert reward["fault_mention_logonly"] == 1.0


def test_detection_uses_canonical_outputs_not_claimed_values():
    reward, calls = _score(
        'INPUT_VECTOR:"a:1,b:0" EXPECTED_OUTPUT:"y:1" '
        'DETECTED_FAULTS:"sa0 n_fault"'
    )

    assert reward["detection"] == 1.0
    assert reward["fidelity"] == 1.0
    # The claimed value was 1; the zero is only an ignored PO-name placeholder.
    assert calls[0][1] == {"y": 0}

    wrong_claim, _ = _score(
        'INPUT_VECTOR:"a:1,b:0" EXPECTED_OUTPUT:"y:0" '
        'DETECTED_FAULTS:"sa0 n_fault"'
    )
    assert wrong_claim["detection"] == 1.0
    assert wrong_claim["fidelity"] == 0.0


def test_missing_expected_output_does_not_block_detection():
    reward, calls = _score(
        'INPUT_VECTOR:"a:1,b:0" DETECTED_FAULTS:"sa0 n_fault"'
    )

    assert calls
    assert reward["detection"] == 1.0
    assert reward["fidelity"] == 0.0
    assert 0.0 < reward["format"] < 1.0


def test_logonly_diagnostics_do_not_enter_legacy_scalar():
    reward = {
        "detection": 1.0,
        "activation": 1.0,
        "fidelity": 1.0,
        "format": 1.0,
        "fault_mention_logonly": 100.0,
    }
    assert train_scalar_from_reward_components(reward) == pytest.approx(1.5)


def test_fast_simulator_never_trusts_claimed_output_value():
    netlist = SimpleNamespace(
        input_nets=["a"],
        output_nets=["y"],
        instructions=[("assign_net", "y", "a")],
        net_dependencies={},
    )

    claimed_zero = fast_fault_sim(
        {"a": 1}, {"y": 0}, "sa0 y", optimized_netlist=netlist
    )
    claimed_one = fast_fault_sim(
        {"a": 1}, {"y": 1}, "sa0 y", optimized_netlist=netlist
    )

    assert claimed_zero.loc["y", "Good Machine"] == 1
    assert claimed_one.loc["y", "Good Machine"] == 1
    assert _fault_detected_at_pos(claimed_zero)
    assert _fault_detected_at_pos(claimed_one)


def test_incomplete_input_cannot_fabricate_output_detection():
    netlist = SimpleNamespace(
        input_nets=["a"],
        output_nets=["y"],
        instructions=[("assign_net", "y", "a")],
        net_dependencies={},
    )
    result = fast_fault_sim({}, {"y": 0}, "sa0 y", optimized_netlist=netlist)
    assert not _fault_detected_at_pos(result)
