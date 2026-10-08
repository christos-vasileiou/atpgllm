"""Trajectory-level ATPG reward: attempts, fidelity to the detecting call, protocol format.

Earlier calls are scored from what the tool returned; only the last generation
that called the tool is simulated again, and the answer never is.
"""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

import atpgllm.llm.reward_funcs as reward_funcs
from atpgllm.llm.reward_funcs import (
    Trajectory,
    vector_key,
    prompt_opens_thinking,
    test_generation_grpo_reward as score_generation,
)

# Granite 4.2: the generation prompt opens the thinking block.
GRANITE_PROMPT = "target sa0 n_fault<|im_start|>assistant\n<think>\n"
QWEN_PROMPT = "target sa0 n_fault<|im_start|>assistant\n"

DETECTS = "a:1,b:0"   # activates n_fault and propagates to y
BLOCKED = "a:0,b:0"   # activates n_fault, y unchanged
INACTIVE = "a:0,b:1"  # n_fault already 0 in the good machine
ALSO_DETECTS = "a:1,b:1"

TABLE = json.dumps({"Good Machine": {"y": 1}, "Bad Machine": {"y": 0}})


def _frame(vector):
    values = dict(vector_key(vector) or ())
    detected = values.get("a") == "1"
    activated = detected or values.get("b") == "0"
    return pd.DataFrame(
        {
            "Good Machine": [int(values.get("a", 0)), int(values.get("b", 0)),
                             1 if activated else 0, 1],
            "Bad Machine": [int(values.get("a", 0)), int(values.get("b", 0)),
                            0, 0 if detected else 1],
            "PIs": [True, True, False, False],
            "POs": [False, False, False, True],
        },
        index=["a", "b", "n_fault", "y"],
    )


def _shown(vector):
    """What the tool returns for ``vector``."""
    return _frame(vector)[["Good Machine", "Bad Machine"]].to_json()


def _xml_call(vector, doc_id="d"):
    return ("<tool_call>\n<function=fault_simulation_tool>\n"
            f"<parameter=input_vector>\n{vector}\n</parameter>\n"
            "<parameter=output_vector>\ny=1\n</parameter>\n"
            "<parameter=fault>\nsa0 n_fault\n</parameter>\n"
            f"<parameter=doc_id>\n{doc_id}\n</parameter>\n</function>\n</tool_call>")


def _call(vector, result=None):
    result = _shown(vector) if result is None else result
    return {
        "arguments": {"input_vector": vector, "output_vector": "y=1",
                      "fault": "sa0 n_fault", "doc_id": "d"},
        "result": result,
    }


def _call_turn(reasoning="Set a to propagate the fault."):
    return (f"{reasoning}\n</think>\n\nI need to call the fault simulation tool.\n"
            "<tool_call>\n<function=fault_simulation_tool>\n</function>\n</tool_call>")


def _answer(vector, output="y:1", think="</think>"):
    return (f'{think}The fault is detected. To sum up, the vectors are:\n'
            f'INPUT_VECTOR:"{vector}" EXPECTED_OUTPUT:"{output}" '
            'DETECTED_FAULTS:"sa0 n_fault"')


def _trace(call_vectors, answer=None, turns=None):
    calls = [_call(v) if isinstance(v, str) else v for v in call_vectors]
    turns = turns or [_call_turn() for _ in calls]
    if answer is not None:
        turns = turns + [answer]
    return {"turns": turns, "calls": calls}


def _extract(label):
    def extract(text):
        marker = f'{label}:"'
        return [part.split('"', 1)[0] for part in text.split(marker)[1:]]
    return extract


def _score_many(traces, prompt=GRANITE_PROMPT, completions=None):
    simulated = []

    def simulator(input_vector, expected_output, fault, netlist, gate_func, **kwargs):
        simulated.append(input_vector)
        return _frame(input_vector), {}

    netlist = SimpleNamespace(input_nets=["a", "b"], output_nets=["y"])
    n = len(traces) if traces is not None else len(completions)
    kwargs = {"tool_trace": traces} if traces is not None else {}
    rows = score_generation(
        [prompt] * n,
        completions or [""] * n,
        netlists=[netlist] * n,
        fault_fn=lambda *_args, **_kw: [("sa0", "n_fault")],
        simulation_fn=lambda _: [],
        input_vector_fn=_extract("INPUT_VECTOR"),
        expected_output_fn=_extract("EXPECTED_OUTPUT"),
        detected_faults_fn=_extract("DETECTED_FAULTS"),
        lib_gate_funcs={},
        fault_sim=simulator,
        **kwargs,
    )
    return rows, simulated


def _score(trace, **kwargs):
    rows, simulated = _score_many([trace], **kwargs)
    return rows[0], simulated


def test_first_try_success_is_best():
    reward, _ = _score(_trace([DETECTS], _answer(DETECTS)))
    assert reward["detection"] == 1.0
    assert reward["activation"] == 1.0
    assert reward["fidelity"] == 1.0
    assert reward["format"] == 1.0
    assert reward["uniqueness"] == 1.0


def test_a_repeated_vector_is_another_attempt():
    reward, _ = _score(_trace([INACTIVE, INACTIVE, DETECTS], _answer(DETECTS)))
    assert reward["detection"] == pytest.approx(1 / 3)
    assert reward["activation"] == pytest.approx(1 / 3)
    assert reward["uniqueness"] == pytest.approx(2 / 3)
    assert reward["duplicate_calls_logonly"] == 1.0
    fresh, _ = _score(_trace([INACTIVE, BLOCKED, DETECTS], _answer(DETECTS)))
    assert fresh["detection"] == reward["detection"]
    assert fresh["uniqueness"] == 1.0 > reward["uniqueness"]
    assert fresh["format"] > reward["format"]


def test_the_reward_simulates_each_vector_once():
    turns = [_call_turn(),
             "Retry.\n</think>\n\n" + _xml_call(BLOCKED) + _xml_call(BLOCKED) + _xml_call(DETECTS),
             _answer(DETECTS)]
    trace = {"turns": turns, "calls": [_call(INACTIVE), _call(BLOCKED)]}
    reward, simulated = _score(trace)
    assert simulated == [BLOCKED, DETECTS]
    assert reward["detection"] == pytest.approx(1 / 4)
    assert reward["uniqueness"] == pytest.approx(3 / 4)
    assert reward["simulated_calls_logonly"] == 2.0


def test_identical_rejected_calls_are_repeats():
    failed = _call("a=2", result="Tool execution failed: input_vector has non-binary values")
    reward, _ = _score(_trace([failed, failed, DETECTS], _answer(DETECTS)))
    assert reward["detection"] == pytest.approx(1 / 3)
    assert reward["uniqueness"] == pytest.approx(2 / 3)


def test_detection_and_activation_average_over_attempts():
    reward, _ = _score(_trace([INACTIVE, DETECTS], _answer(DETECTS)))
    assert reward["detection"] == 0.5
    assert reward["activation"] == 0.5
    assert reward["fidelity"] == 1.0
    assert reward["format"] == 1.0
    # Coverage metrics read the final answer's binary detection, not the objective.
    assert reward["fault_detected_by_pred_input_vector_acc_logonly"] == 1.0


def test_repeating_the_detecting_call_gains_nothing_and_breaks_format():
    once, _ = _score(_trace([INACTIVE, DETECTS], _answer(DETECTS)))
    again, simulated = _score(_trace([INACTIVE, DETECTS, DETECTS], _answer(DETECTS)))
    assert again["detection"] == once["detection"]
    assert again["activation"] == once["activation"]
    assert again["format"] < once["format"]
    assert again["duplicate_calls_logonly"] == 1.0
    assert again["calls_after_detection_logonly"] == 1.0
    assert simulated == [DETECTS]  # only the last generation's call


def test_another_detecting_call_after_success_gains_nothing():
    once, _ = _score(_trace([INACTIVE, DETECTS], _answer(DETECTS)))
    more, _ = _score(_trace([INACTIVE, DETECTS, ALSO_DETECTS], _answer(DETECTS)))
    assert more["detection"] == once["detection"]
    assert more["format"] < once["format"]


def test_padding_with_blocked_vectors_lowers_detection():
    padded, _ = _score(_trace([BLOCKED, "a:0,b:0,c:1", DETECTS], _answer(DETECTS)))
    direct, _ = _score(_trace([DETECTS], _answer(DETECTS)))
    assert padded["detection"] == pytest.approx(1 / 3)
    assert padded["detection"] < direct["detection"]


def test_answer_that_departs_from_the_detecting_call():
    reward, simulated = _score(_trace([DETECTS], _answer(INACTIVE)))
    # Detection scores the calls; the answer is held to them by fidelity.
    assert reward["detection"] == 1.0
    # Inputs disagree on both nets; y:1 still matches the call's good machine.
    assert reward["fidelity"] == pytest.approx(0.5)
    assert reward["unverified_answer_logonly"] == 1.0
    assert reward["fault_detected_by_pred_input_vector_acc_logonly"] == 0.0
    assert simulated == [DETECTS]


def test_fidelity_counts_expected_output_against_the_detecting_call():
    reward, _ = _score(_trace([DETECTS], _answer(DETECTS, output="y:0")))
    assert reward["detection"] == 1.0
    assert reward["fidelity"] == pytest.approx(0.5)


def test_fidelity_uses_the_last_detecting_call():
    reward, _ = _score(_trace([DETECTS, ALSO_DETECTS], _answer(ALSO_DETECTS)))
    assert reward["fidelity"] == 1.0


def test_detection_needs_a_tool_call():
    reward, simulated = _score(_trace([], _answer(DETECTS)))
    assert reward["detection"] == 0.0
    assert reward["uniqueness"] == 0.0
    assert reward["activation"] == 0.0
    assert reward["fidelity"] == 0.0
    assert reward["format"] == pytest.approx(5 / 6)
    assert simulated == []


def test_answer_copying_a_failed_vector_scores_nothing():
    reward, _ = _score(_trace([INACTIVE, BLOCKED], _answer(BLOCKED)))
    assert reward["detection"] == 0.0
    assert reward["fidelity"] == 0.0
    assert reward["activation"] == 0.5
    # Answered after a negative result instead of calling again.
    assert reward["format"] < 1.0


def test_the_answer_is_never_simulated():
    reward, simulated = _score(_trace([INACTIVE, BLOCKED], _answer(DETECTS)))
    assert reward["detection"] == 0.0
    assert reward["fidelity"] == 0.0
    assert simulated == [BLOCKED]


def test_earlier_calls_are_scored_from_the_tool_result():
    reward, simulated = _score(_trace([INACTIVE, BLOCKED, DETECTS], _answer(DETECTS)))
    assert simulated == [DETECTS]
    assert reward["detection"] == pytest.approx(1 / 3)
    assert reward["activation"] == pytest.approx(2 / 3)
    assert reward["simulated_calls_logonly"] == 1.0


def test_all_calls_listed_in_the_last_generation_are_simulated():
    turns = [_call_turn(), "Two candidates.\n</think>\n\n" + _xml_call(BLOCKED) + _xml_call(DETECTS),
             _answer(DETECTS)]
    # The environment ran only the first listed call.
    trace = {"turns": turns, "calls": [_call(INACTIVE), _call(BLOCKED)]}
    reward, simulated = _score(trace)
    assert simulated == [BLOCKED, DETECTS]
    assert reward["detection"] == pytest.approx(1 / 3)
    assert reward["fidelity"] == 1.0


def test_calls_left_unexecuted_in_the_final_turn_are_simulated():
    turns = [_call_turn(), "Retry.\n</think>\n\n" + _xml_call(DETECTS)]
    reward, simulated = _score({"turns": turns, "calls": [_call(INACTIVE)]})
    assert simulated == [DETECTS]
    assert reward["detection"] == 0.5
    # No answer followed the call.
    assert reward["format"] < 1.0


def test_failed_tool_execution_is_a_failed_attempt():
    failed = _call("a=2", result="Tool execution failed: input_vector has non-binary values")
    reward, simulated = _score(_trace([failed, DETECTS], _answer(DETECTS)))
    assert reward["detection"] == 0.5
    assert "a=2" not in simulated
    assert reward["format"] < 1.0


def test_truncated_trajectory_fails_the_final_turn():
    reward, _ = _score(_trace([INACTIVE]))
    assert reward["detection"] == 0.0
    assert reward["format"] == pytest.approx(2 / 8)


def test_retry_turn_must_reason_before_calling():
    reasoned, _ = _score(_trace([INACTIVE, DETECTS], _answer(DETECTS)))
    turns = [_call_turn(), _call_turn(reasoning=""), _answer(DETECTS)]
    silent, _ = _score({"turns": turns, "calls": [_call(INACTIVE), _call(DETECTS)]})
    assert silent["format"] < reasoned["format"]
    assert silent["reasoned_call_turns_logonly"] == 0.5


def test_unclosed_thinking_in_the_answer_turn_breaks_format():
    closed, _ = _score(_trace([DETECTS], _answer(DETECTS)))
    open_, _ = _score(_trace([DETECTS], _answer(DETECTS, think="")))
    assert open_["format"] < closed["format"]
    assert open_["detection"] == closed["detection"] == 1.0


def test_thinking_opened_in_the_completion_needs_no_prompt_tag():
    turns = ["<think>Set a.</think>\n\n" + _xml_call(DETECTS), _answer(DETECTS, think="")]
    reward, _ = _score({"turns": turns, "calls": [_call(DETECTS)]}, prompt=QWEN_PROMPT)
    assert reward["format"] == 1.0
    assert not prompt_opens_thinking(QWEN_PROMPT)
    assert prompt_opens_thinking(GRANITE_PROMPT)


def test_template_opening_is_required_when_the_turn_has_none():
    turns = ["Set a.\n</think>\n\n" + _xml_call(DETECTS), _answer(DETECTS, think="")]
    reward, _ = _score({"turns": turns, "calls": [_call(DETECTS)]}, prompt=QWEN_PROMPT)
    assert reward["reasoned_call_turns_logonly"] == 0.0


def test_only_the_final_turn_answers():
    turns = [_call_turn() + '\nINPUT_VECTOR:"a:0,b:1"', _answer(DETECTS)]
    reward, _ = _score({"turns": turns, "calls": [_call(DETECTS)]})
    assert reward["detection"] == 1.0
    assert reward["fidelity"] == 1.0


def _granite_text(trace):
    """The decoded completion of a Granite tool trajectory (special tokens dropped)."""
    parts = []
    for k, turn in enumerate(trace["turns"]):
        if k:
            parts.append("\n\nassistant\n<think>\n")
        parts.append(turn)
        if k < len(trace["calls"]):
            call = trace["calls"][k]
            args = "".join(f"<parameter={key}>\n{value}\n</parameter>\n"
                           for key, value in call["arguments"].items())
            parts[-1] = parts[-1].replace(
                "<function=fault_simulation_tool>\n",
                f"<function=fault_simulation_tool>\n{args}",
            )
            parts.append(f"\n\nuser\n<tool_response>\n{call['result']}\n</tool_response>")
    return "".join(parts)


def test_text_fallback_matches_the_recorded_trace():
    trace = _trace([INACTIVE, DETECTS], _answer(DETECTS))
    from_trace, _ = _score(trace)
    rows, _ = _score_many(None, completions=[_granite_text(trace)])
    for key in ("detection", "activation", "fidelity", "format"):
        assert rows[0][key] == pytest.approx(from_trace[key]), key


def test_written_tool_response_is_not_a_call():
    fake = (_call_turn().split("<tool_call>")[0]
            + "\n<tool_response>\n" + TABLE + "\n</tool_response>"
            + "\n" + _answer(DETECTS))
    with_trace, _ = _score({"turns": [fake], "calls": []})
    honest, _ = _score(_trace([DETECTS], _answer(DETECTS)))
    assert with_trace["fidelity"] == 0.0
    assert with_trace["format"] < honest["format"]

    rows, _ = _score_many(None, completions=[fake])
    assert rows[0]["fidelity"] == 0.0
    assert rows[0]["detection"] < 1.0


def test_trajectory_from_text_pairs_calls_with_responses():
    trajectory = Trajectory.from_text(_granite_text(_trace([INACTIVE], _answer(DETECTS))))
    assert len(trajectory.calls) == 1
    assert trajectory.calls[0]["arguments"]["input_vector"] == INACTIVE
    assert trajectory.final_turn.endswith('DETECTED_FAULTS:"sa0 n_fault"')


def test_threaded_native_scoring_keeps_each_trace(monkeypatch):
    monkeypatch.setattr(reward_funcs, "using_native_simulator", lambda: True)
    monkeypatch.setenv("TMAX_REWARD_PROFILE", "full")
    traces = [_trace([DETECTS], _answer(DETECTS)),
              _trace([INACTIVE, DETECTS], _answer(DETECTS)),
              _trace([INACTIVE], _answer(INACTIVE))]
    rows, _ = _score_many(traces)
    assert [row["detection"] for row in rows] == [1.0, 0.5, 0.0]


def test_unexecuted_calls_must_pass_the_tool_checks():
    document = "{'doc_id': 'd', 'netlist': 'module m; endmodule'}"
    prompt = f"target sa0 n_fault {document}<|im_start|>assistant\n<think>\n"
    turns = [_call_turn(), "Retry.\n</think>\n\n" + _xml_call(DETECTS, doc_id="other")]
    reward, simulated = _score({"turns": turns, "calls": [_call(INACTIVE)]}, prompt=prompt)
    assert simulated == []
    assert reward["detection"] == 0.0
    turns[1] = "Retry.\n</think>\n\n" + _xml_call(DETECTS)
    reward, simulated = _score({"turns": turns, "calls": [_call(INACTIVE)]}, prompt=prompt)
    assert simulated == [DETECTS]
    assert reward["detection"] == 0.5


def test_calls_without_a_readable_vector_differ_by_their_arguments():
    from atpgllm.llm.reward_funcs import call_identity
    assert call_identity({"arguments": {"fault": "sa0 a"}}) != call_identity({"arguments": {"fault": "sa1 b"}})
    assert call_identity({"arguments": None}) == call_identity({"arguments": None})
    assert call_identity({"arguments": {"input_vector": "a=1, b=0"}}) == \
        call_identity({"arguments": {"input_vector": "b:0,a:1", "fault": "sa1 x"}})
