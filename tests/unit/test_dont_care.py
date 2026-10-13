"""Don't-care (x) protocol: tool calls carry x, final answers state care bits only.

A missing input in the final answer is read as x and simulated 3-valued, so a
detection holds for every fill. The reward wants the answer to repeat the
detecting call's care bits and omit its don't-cares.
"""

import copy
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from atpgllm.llm.reward_funcs import (
    care_bits,
    test_generation_grpo_reward as score_generation,
    vector_key,
)
from atpgllm.training.conversation import ConversationExample
from atpgllm.training.search_verifier import assignment, final_assignment

GRANITE_PROMPT = "target sa0 n_fault<|im_start|>assistant\n<think>\n"
# n_fault = a; y = n_fault OR b. With a=1, b=0 the fault reaches y; with b
# unknown the good y is 1 but the bad y is unknown, so only b=0 detects.
CALL = "a:1,b:0,c:x"


def _frame(vector):
    v = dict(vector_key(vector) or ())
    a, b = v.get("a", "x"), v.get("b", "x")
    good_n = a if a != "x" else "x"
    good_y = "1" if "1" in (good_n, b) else ("0" if good_n == b == "0" else "x")
    bad_y = "1" if b == "1" else ("0" if b == "0" else "x")
    as_cell = lambda x: x if x == "x" else int(x)
    return pd.DataFrame(
        {"Good Machine": [as_cell(a), as_cell(b), as_cell(v.get("c", "x")), as_cell(good_n), as_cell(good_y)],
         "Bad Machine": [as_cell(a), as_cell(b), as_cell(v.get("c", "x")), 0, as_cell(bad_y)],
         "PIs": [True, True, True, False, False],
         "POs": [False, False, False, False, True]},
        index=["a", "b", "c", "n_fault", "y"],
    )


def _score(answer_vector, output="y:1"):
    call = {"arguments": {"input_vector": CALL, "output_vector": "y=1",
                          "fault": "sa0 n_fault", "doc_id": "d"},
            "result": _frame(CALL)[["Good Machine", "Bad Machine"]].to_json()}
    answer = ('</think>Detected. To sum up, the vectors are:\n'
              f'INPUT_VECTOR:"{answer_vector}" EXPECTED_OUTPUT:"{output}" '
              'DETECTED_FAULTS:"sa0 n_fault"')
    trace = {"turns": ["Set a and b.\n</think>\n<tool_call>\n</tool_call>", answer],
             "calls": [call]}

    def extract(label):
        return lambda text: [p.split('"', 1)[0] for p in text.split(f'{label}:"')[1:]]

    return score_generation(
        [GRANITE_PROMPT], [""], tool_trace=[trace],
        netlists=[SimpleNamespace(input_nets=["a", "b", "c"], output_nets=["y"])],
        fault_fn=lambda *_a, **_k: [("sa0", "n_fault")], simulation_fn=lambda _: [],
        input_vector_fn=extract("INPUT_VECTOR"), expected_output_fn=extract("EXPECTED_OUTPUT"),
        detected_faults_fn=extract("DETECTED_FAULTS"), lib_gate_funcs={},
        fault_sim=lambda vector, *a, **k: (_frame(vector), {}),
    )[0]


def test_care_bits_answer_is_the_ideal_final_answer():
    reward = _score("a:1,b:0")
    assert reward["detection"] == 1.0
    assert reward["fidelity"] == 1.0
    assert reward["format"] == 1.0
    assert reward["answer_x_free_logonly"] == 1.0
    assert reward["fault_detected_by_pred_input_vector_acc_logonly"] == 1.0
    assert reward["pi_completeness_logonly"] == 1.0


def test_writing_x_in_the_answer_costs_fidelity_and_format():
    explicit = _score("a:1,b:0,c:x")
    assert explicit["fidelity"] < 1.0
    assert explicit["format"] < 1.0
    assert explicit["answer_x_free_logonly"] == 0.0
    # Filling the don't-care is a valid test but costs tokens: not the care bits.
    filled = _score("a:1,b:0,c:1")
    assert filled["fidelity"] < 1.0
    assert filled["format"] == 1.0


def test_dropping_a_care_bit_loses_fidelity_and_empty_answer_scores_zero():
    assert _score("a:1")["fidelity"] < 1.0
    assert _score("")["fidelity"] == pytest.approx(0.5)  # PO agreement only


def test_care_bits_strip_dont_cares():
    assert care_bits(vector_key("a:1,b:x")) == (("a", "1"),)
    assert care_bits(None) is None


def test_final_answer_omits_dont_cares_but_tool_call_keeps_them():
    record = {
        "system_content": "s", "user_content": "u {fault}", "module_name": "m",
        "netlist": "module m(a,b,y); endmodule", "fault": "sa0 a",
        "input_vector": json.dumps({"a": 1, "b": "x"}),
        "expected_output": json.dumps({"y": 1, "z": "x"}),
        "snapshot": "{}", "detected_faults": "sa0 a",
        "fault_propagation_gates": "", "backtrack_gates": "", "backtrack_nets": "",
        "reasoning_content": "1. **Step**: {fault_net}",
        "answer_content": 'INPUT_VECTOR: "{input_vector}"\nEXPECTED_OUTPUT: "{expected_output}"\n',
    }
    messages = ConversationExample.from_record(copy.deepcopy(record), use_tools=True).messages
    call = messages[2]["tool_calls"][0]["function"]["arguments"]
    assert call["input_vector"] == {"a": 1, "b": "x"}
    assert call["output_vector"] == {"y": 1, "z": "x"}
    assert 'INPUT_VECTOR: "a: 1"' in messages[-1]["content"]
    assert 'EXPECTED_OUTPUT: "y: 1"' in messages[-1]["content"]

    binary = dict(record, input_vector=json.dumps({"a": 1, "b": 0}),
                  expected_output=json.dumps({"y": 1, "z": 0}))
    answer = ConversationExample.from_record(binary, use_tools=True).messages[-1]["content"]
    assert 'INPUT_VECTOR: "a: 1, b: 0"' in answer
    assert 'EXPECTED_OUTPUT: "y: 1, z: 0"' in answer


def test_final_assignment_reads_missing_inputs_as_dont_care():
    names = ["a", "b", "c"]
    assert final_assignment("a: 1, c: 0", names) == {"a": 1, "b": "x", "c": 0}
    assert final_assignment("a: 1, b: x", names) == {"a": 1, "b": "x", "c": "x"}
    assert final_assignment("", names) == {"a": "x", "b": "x", "c": "x"}
    assert final_assignment({"b": 1}, names) == {"a": "x", "b": 1, "c": "x"}
    assert final_assignment("{'b': 0}", names) == {"a": "x", "b": 0, "c": "x"}
    for bad in ("d: 1", "a: 1, a: 0", "a: 2", "{'a': 1, 'a': 0}"):
        with pytest.raises(ValueError):
            final_assignment(bad, names)


def test_tool_calls_accept_x_only_where_allowed():
    assert assignment("a: x, b: 1", ["a", "b"], allow_x=True) == {"a": "x", "b": 1}
    with pytest.raises(ValueError):
        assignment("a: x, b: 1", ["a", "b"])
    with pytest.raises(ValueError):
        assignment("a: 1", ["a", "b"], allow_x=True)  # Tool calls list every input.


def test_tool_guard_accepts_dont_cares():
    from atpgllm.training.tools import _require_bit_mapping
    _require_bit_mapping("input_vector", {"a": "x", "b": 1})
    _require_bit_mapping("input_vector", "a: x, b: 1")
    with pytest.raises(ValueError):
        _require_bit_mapping("input_vector", {"a": 2})
