"""Retry trajectories: reading tool results, failed attempts, rendering."""

import copy
import json
import random

import pytest

from atpgllm.training.conversation import ConversationExample
from atpgllm.training.tool_feedback import (
    detected_sentence,
    read_snapshot,
    revision_sentence,
    undetected_sentence,
)
from atpgllm.training.tool_trajectories import failed_attempt, simulate

# y = (a & b) | c. For sa0 n1: a=b=1 activates it, and c must be 0 or the
# OR gate blocks it.
NETLIST = """module tiny(a, b, c, y);
input a, b, c;
output y;
wire n1;
AND2x2_ASAP7_75t_R U1 (.A(a), .B(b), .Y(n1));
OR2x2_ASAP7_75t_R U2 (.A(n1), .B(c), .Y(y));
endmodule"""


def _record():
    record = {
        "netlist": NETLIST,
        "module_name": "tiny",
        "circuit_id": "tiny",
        "fault": "sa0 n1",
        "input_vector": json.dumps({"a": 1, "b": 1, "c": 0}),
        "expected_output": json.dumps({"y": 1}),
        "fault_propagation_gates": "U1, U2",
        "backtrack_gates": "U1, U2",
        "backtrack_nets": "a, b, c",
        "detected_faults": "sa0 n1",
        "system_content": "ATPG assistant.",
        "user_content": "Target {fault} in {module_name}: {netlist}",
        "reasoning_content": "Drive n1 to 1 and keep c at 0.",
        "answer_content": 'INPUT_VECTOR: "{input_vector}"\n',
    }
    record["snapshot"] = simulate(record, {"a": 1, "b": 1, "c": 0})
    return record


def test_snapshot_readings_name_the_failure():
    record = _record()
    hit = read_snapshot(record["snapshot"], "sa0 n1", ["y"])
    assert hit.detected and hit.differing_outputs == ["y"]
    assert "1 primary output: y" in detected_sentence(hit)

    unexcited = read_snapshot(simulate(record, {"a": 0, "b": 1, "c": 0}), "sa0 n1", ["y"])
    assert not unexcited.detected and not unexcited.activated
    assert "never activated: n1 must be 1" in undetected_sentence(unexcited)

    blocked = read_snapshot(simulate(record, {"a": 1, "b": 1, "c": 1}), "sa0 n1", ["y"])
    assert not blocked.detected and blocked.activated
    assert blocked.differing_internal == ["n1"]
    assert "blocked before every primary output" in undetected_sentence(blocked)
    assert "c to 0" in revision_sentence(blocked, [("c", 0)])


def test_failed_attempt_is_a_confirmed_miss_that_the_changes_fix():
    record = _record()
    for seed in range(10):
        attempt = failed_attempt(record, random.Random(seed))
        assert attempt is not None
        assert not read_snapshot(attempt.snapshot, "sa0 n1", ["y"]).detected
        fixed = dict(attempt.input_vector, **dict(attempt.changes))
        assert fixed == json.loads(record["input_vector"])
        # The failed call reports what the Good Machine really outputs.
        good = json.loads(attempt.snapshot)["Good Machine"]
        assert attempt.output_vector == {"y": good["y"]}


def test_default_conversation_is_unchanged_and_retry_is_opt_in():
    record = _record()
    plain = ConversationExample.from_record(copy.deepcopy(record), use_tools=True)
    assert plain.trainable is None
    assert [m["role"] for m in plain.messages] == [
        "system", "user", "assistant", "tool", "assistant",
    ]
    assert "has been sensitized" in plain.messages[-1]["content"]

    attempt = failed_attempt(record, random.Random(0))
    retry = ConversationExample.from_record(
        copy.deepcopy(record), use_tools=True, failed_attempt=attempt
    )
    roles = [m["role"] for m in retry.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"]
    # The injected failed call is context, not a target.
    assert retry.trainable == [False, False, False, False, True, False, True]
    first, second = (
        m["tool_calls"][0]["function"]["arguments"]
        for m in retry.messages if m.get("tool_calls")
    )
    assert first["input_vector"] == attempt.input_vector
    assert second["input_vector"] == json.loads(record["input_vector"])
    assert retry.messages[3]["content"] == attempt.snapshot
    assert "does not detect sa0 n1" in retry.messages[4]["content"]
    assert "INPUT_VECTOR" not in retry.messages[4]["content"]
    assert retry.messages[-1]["content"].startswith("The Good and Bad Machine differ")

    with pytest.raises(ValueError):
        ConversationExample.from_record(copy.deepcopy(record), failed_attempt=attempt)


def test_tool_conversation_labels_only_trainable_turns():
    from transformers import AutoTokenizer

    from atpgllm.multimodal.data import tokenize_tool_conversation

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True
        )
    except OSError:
        pytest.skip("Qwen2.5 tokenizer not cached")
    record = _record()
    attempt = failed_attempt(record, random.Random(0))
    retry = ConversationExample.from_record(
        copy.deepcopy(record), use_tools=True, failed_attempt=attempt
    )
    ids, labels = tokenize_tool_conversation(retry.messages, retry.trainable, tokenizer)
    trained = tokenizer.decode([t for t, l in zip(ids, labels) if l != -100])
    assert "does not detect sa0 n1" in trained
    assert "differ at 1 primary output" in trained
    assert "Drive n1 to 1" not in trained          # the failed turn's reasoning
    assert "\"Good Machine\":" not in trained      # tool responses (JSON)
    assert trained.count("<tool_call>") == 1       # only the corrected call


def test_po_fault_view_keeps_outputs_and_fault_site():
    from atpgllm.training.tool_feedback import shown_snapshot

    record = _record()
    blocked = simulate(record, {"a": 1, "b": 1, "c": 1})
    shown = json.loads(shown_snapshot(blocked, "sa0 n1", ["y"], "po_fault"))
    assert list(shown["Good Machine"]) == ["n1", "y"]       # table order kept
    assert shown["Bad Machine"] == {"n1": 0, "y": 1}
    assert shown_snapshot(blocked, "sa0 n1", ["y"], "full") == blocked
    with pytest.raises(ValueError):
        shown_snapshot(blocked, "sa0 n1", ["y"], "pis_only")


def test_live_tool_and_sft_show_the_same_view(monkeypatch):
    import asyncio

    from atpgllm.training.tool_feedback import shown_snapshot
    from atpgllm.training.tools import fault_simulation_tool_handler

    record = _record()
    vector = {"a": 1, "b": 1, "c": 1}
    monkeypatch.setenv("FAULT_SIM_TOOL_VIEW", "po_fault")
    live = asyncio.run(fault_simulation_tool_handler(
        vector, {"y": 1}, "sa0 n1", "tiny", record["netlist"]
    ))
    assert live == shown_snapshot(simulate(record, vector), "sa0 n1", ["y"], "po_fault")


def test_retry_under_po_fault_view_cites_only_what_is_shown():
    record = _record()
    attempt = next(
        a for a in (failed_attempt(record, random.Random(s)) for s in range(50))
        if a.reading.activated
    )
    retry = ConversationExample.from_record(
        copy.deepcopy(record), use_tools=True, failed_attempt=attempt,
        tool_view="po_fault",
    )
    tool_messages = [m["content"] for m in retry.messages if m["role"] == "tool"]
    assert all(set(json.loads(t)["Good Machine"]) == {"n1", "y"} for t in tool_messages)
    diagnosis = retry.messages[4]["content"]
    assert "The fault is activated (n1 is 1 in the Good Machine and 0 in the Bad Machine), but" in diagnosis
    with pytest.raises(ValueError):
        ConversationExample.from_record(copy.deepcopy(record), tool_view="po_fault")
