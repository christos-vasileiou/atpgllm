"""Retry SFT examples: assembly, grounded text, per-turn loss masks."""

import json
import os

import pytest

from atpgllm.training.conversation import ConversationExample
from atpgllm.training.dataset_utils import assistant_spans, tokenize_with_turn_masks
from atpgllm.training.retry_repair import RepairSearch, fast_simulator, fast_structure
from atpgllm.training.retry_trajectories import (
    STOCK_CALL_LINE,
    call_message,
    ground_example,
    retry_example,
    tool_message,
)
from atpgllm.training.tool_feedback import read_snapshot
from atpgllm.training.tools import TOOLS

NETLIST = """module tiny(a, b, c, y);
input a, b, c;
output y;
wire n1;
AND2x2_ASAP7_75t_R U1 (.A(a), .B(b), .Y(n1));
OR2x2_ASAP7_75t_R U2 (.A(n1), .B(c), .Y(y));
endmodule"""
FAULT = "sa0 n1"
PROMPT = [{"role": "system", "content": "ATPG assistant."},
          {"role": "user", "content": "Target sa0 n1 in tiny."}]


def _path(start):
    structure = fast_structure(NETLIST)
    simulate = fast_simulator(structure, FAULT, ["y"])
    search = RepairSearch(structure, simulate, FAULT, ["a", "b", "c"], ["y"])
    first = search.observe(start)
    steps, previous = [], first
    for step in search.path(start):
        steps.append({"vector": step.observation.vector, "changes": list(step.changes),
                      "snapshot": step.observation.snapshot, "prev_reading": previous.reading,
                      "prev_blockers": list(previous.blockers)})
        previous = step.observation
    return first, steps


def _example():
    first, steps = _path({"a": 0, "b": 1, "c": 1})
    failed = call_message("my own attempt", first.vector, {"y": 1}, FAULT, "d",
                          post=STOCK_CALL_LINE + "\n")
    return retry_example(PROMPT, failed, first.snapshot, steps, FAULT, "d", ["y"], TOOLS)


def test_retry_example_reads_each_answer_and_ends_on_a_detection():
    example = _example()
    roles = [m["role"] for m in example["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool",
                     "assistant", "tool", "assistant"]
    assert example["assistant_train"] == [False, True, True, True]
    second, third, final = [m["content"] for m in example["messages"] if m["role"] == "assistant"][1:]
    assert "never activated: n1 must be 1" in second and "I'll set a to 1" in second
    assert "blocked before every primary output" in third
    assert "the gate driving y reads the difference on n1" in third
    assert "c to 0" in third
    assert "differ at 1 primary output: y" in final
    assert 'INPUT_VECTOR: "a: 1, b: 1, c: 0"' in final
    assert 'DETECTED_FAULTS: "sa0 n1"' in final
    # Each call reports the Good Machine outputs its own snapshot shows.
    calls = [m for m in example["messages"] if m.get("tool_calls")]
    tables = [m["content"] for m in example["messages"] if m["role"] == "tool"]
    for call, table in zip(calls[1:], tables[1:]):
        good = json.loads(table)["Good Machine"]
        assert json.loads(call["tool_calls"][0]["function"]["arguments"]["output_vector"]) == {"y": good["y"]}
    assert STOCK_CALL_LINE not in json.dumps(example["messages"][4:])


def test_retry_example_must_end_on_a_detection():
    first, steps = _path({"a": 0, "b": 1, "c": 1})
    failed = call_message("x", first.vector, {"y": 1}, FAULT, "d")
    with pytest.raises(ValueError):
        retry_example(PROMPT, failed, first.snapshot, steps[:1], FAULT, "d", ["y"], TOOLS)


def test_ground_example_replaces_both_stock_lines():
    first, steps = _path({"a": 1, "b": 1, "c": 1})
    hit = steps[-1]
    example = {"messages": PROMPT + [
        call_message("reasoning", hit["vector"], {"y": 1}, FAULT, "d",
                     post="I have to verify if the fault sa0 n1 is detected by the input and output "
                          "vectors. I need to call the fault simulation tool.\n"),
        tool_message(hit["snapshot"]),
        {"role": "assistant", "content": "<think>\nThe fault sa0 n1 has been sensitized. I can see the "
                                         "difference between Good/Bad Machine. To sum up, X\n</think>\n\n"
                                         "The fault sa0 n1 has been sensitized. I can see the difference "
                                         "between Good/Bad Machine. To sum up, X"},
    ], "tools": TOOLS}
    grounded = ground_example(example, FAULT, ["y"])
    call, final = grounded["messages"][2]["content"], grounded["messages"][4]["content"]
    assert "I have to verify" not in call and call.endswith("</think>\n\n")
    assert "has been sensitized" not in final
    assert final.count("The Good and Bad Machine differ at 1 primary output: y.") == 2
    # A last answer that does not detect cannot be grounded as a detection.
    example["messages"][3] = tool_message(first.snapshot)
    assert ground_example(example, FAULT, ["y"]) is None


def _record(snapshot):
    return {
        "netlist": NETLIST, "module_name": "tiny", "circuit_id": "tiny", "fault": FAULT,
        "input_vector": json.dumps({"a": 1, "b": 1, "c": 0}), "expected_output": json.dumps({"y": 1}),
        "fault_propagation_gates": "U1, U2", "backtrack_gates": "U1, U2", "backtrack_nets": "a, b, c",
        "detected_faults": "sa0 n1", "system_content": "ATPG assistant.",
        "user_content": "Target {fault} in {module_name}: {netlist}",
        "reasoning_content": "Drive n1 to 1 and keep c at 0.",
        "answer_content": 'INPUT_VECTOR: "{input_vector}"\n', "snapshot": snapshot,
    }


def test_grounded_conversation_is_opt_in(monkeypatch):
    _, steps = _path({"a": 1, "b": 1, "c": 1})
    record = _record(steps[-1]["snapshot"])
    monkeypatch.delenv("SFT_GROUNDED_TEXT", raising=False)
    plain = ConversationExample.from_record(dict(record), use_tools=True).messages
    assert "I have to verify" in plain[2]["content"]
    assert plain[-1]["content"].startswith("The fault sa0 n1 has been sensitized.")
    monkeypatch.setenv("SFT_GROUNDED_TEXT", "1")
    grounded = ConversationExample.from_record(dict(record), use_tools=True).messages
    assert "I have to verify" not in grounded[2]["content"]
    assert grounded[-1]["content"].startswith(
        "<think>\nThe Good and Bad Machine differ at 1 primary output: y.")
    assert "To sum up, the vectors are:" in grounded[-1]["content"]
    # The default path is byte-identical to before when the switch is off.
    monkeypatch.setenv("SFT_GROUNDED_TEXT", "0")
    assert ConversationExample.from_record(dict(record), use_tools=True).messages == plain


def test_assistant_spans():
    assert assistant_spans([0, 1, 1, 0, 0, 1, 0, 1]) == [(1, 3), (5, 6), (7, 8)]
    assert assistant_spans([]) == []


@pytest.fixture(scope="module")
def granite_tokenizer():
    from transformers import AutoTokenizer
    from atpgllm.training.model_utils import patch_chat_template_for_assistant_mask
    try:
        tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-4.2-8b", local_files_only=True)
    except Exception as exc:  # no cached tokenizer on this host
        pytest.skip(f"Granite 4.2 tokenizer not cached: {exc}")
    patch_chat_template_for_assistant_mask(tokenizer)
    return tokenizer


def test_turn_masks_leave_the_failed_call_out_of_the_loss(granite_tokenizer):
    tokenizer = granite_tokenizer
    example = _example()
    tokens = tokenize_with_turn_masks(example, tokenizer)
    everything = tokenize_with_turn_masks({k: v for k, v in example.items() if k != "assistant_train"},
                                          tokenizer)
    assert tokens["input_ids"] == everything["input_ids"]
    spans = assistant_spans(everything["assistant_masks"])
    assert len(spans) == 4
    trained = assistant_spans(tokens["assistant_masks"])
    assert trained == spans[1:]
    masked = tokenizer.decode(tokens["input_ids"][spans[0][0]:spans[0][1]])
    assert "my own attempt" in masked
    assert "never activated" in tokenizer.decode(tokens["input_ids"][trained[0][0]:trained[0][1]])
    bad = dict(example, assistant_train=[True, False])
    with pytest.raises(ValueError):
        tokenize_with_turn_masks(bad, tokenizer)
