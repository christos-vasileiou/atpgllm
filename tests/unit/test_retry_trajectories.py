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


def test_strip_assignment_lines_keeps_the_plan_and_drops_the_vector():
    from atpgllm.training.retry_trajectories import strip_assignment_lines
    content = ("<think>1. **Excitation**: set inputs via a, b so n1 is 1.\n"
               "6. **Output Prediction**: Expected output: y: 1.\n"
               "7. **Simulation**: Execute the input vector a: 1, b: 1, c: 0 and snapshot.</think>\n\n"
               "I'll call the tool.")
    stripped = strip_assignment_lines(content)
    assert "set inputs via a, b so n1 is 1." in stripped
    assert "a: 1, b: 1" not in stripped and "Expected output" not in stripped
    assert stripped.endswith("</think>\n\nI'll call the tool.")


def test_strip_assignment_lines_drops_compact_vectors_and_single_outputs():
    """v5 plans as conversation renders them: wide buses compact, one-output circuits."""
    from atpgllm.training.retry_trajectories import strip_assignment_lines
    content = (
        "<think>1. **Initialization**: Execute the input vector a: 1, b: 0, c: 1 first.\n"
        "2. **Backtrace to PIs**: Trace backward from n2 through U4, U3 to satisfy n2 = 0. "
        "Controlling nets: i1, i2, s.\n"
        "3. **Logic Justification**: To excite the stuck-at-1 fault, we need n19 at 0.\n"
        "4. **Output Verification**: The fault is sensitized if out1 differs from out1: 0.\n"
        "5. **Fault Simulation**: Compare the good machine (expected output: o: 0) with the faulty one.\n"
        "6. **Final Vector Synthesis**: The input vector in1[0..31] ones={2..3, 26}.\n"
        "7. **Data Snapshot**: Snapshot with a[0..7] bits_lsb0=01111111, d[0]: x. "
        "Detected faults: sa1 n19, sa1 out1.</think>\n\n")
    stripped = strip_assignment_lines(content)
    # The first line goes with its values; the thinking opener stays.
    assert stripped.startswith("<think>2. **Backtrace to PIs**")
    assert "satisfy n2 = 0" in stripped and "we need n19 at 0" in stripped
    for gone in ("a: 1", "out1: 0", "o: 0", "ones={", "bits_lsb0", "Detected faults"):
        assert gone not in stripped
    assert stripped.endswith("we need n19 at 0.</think>\n\n")
    # Applying it again changes nothing, so a built set can be brought up to date.
    assert strip_assignment_lines(stripped) == stripped


def test_restrip_restores_the_opener_the_old_filter_dropped():
    import importlib
    import sys
    from collections import Counter
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "retry"))
    restrip_plans = importlib.import_module("restrip_plans")
    old_output = "2. **Path Selection**: from n10 to y.\n6. **Vector**: in1[0..31] ones={1}.</think>\n\n"
    row = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                        {"role": "assistant", "content": old_output, "tool_calls": []},
                        {"role": "tool", "content": "{}"},
                        {"role": "assistant", "content": "<think>\nreading\n</think>\n\nanswer"}],
           "assistant_train": [False, True]}
    stats = Counter()
    out = restrip_plans.restrip_row(row, stats)
    assert out["messages"][2]["content"] == "<think>2. **Path Selection**: from n10 to y.</think>\n\n"
    assert out["messages"][4]["content"] == "<think>\nreading\n</think>\n\nanswer"
    assert stats["opener_restored"] == 1 and stats["changed"] == 1
    trained = dict(row, assistant_train=[True, True])
    import pytest
    with pytest.raises(SystemExit):
        restrip_plans.restrip_row(trained, Counter())


def _builder():
    import importlib
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "retry"))
    return importlib.import_module("build_retry_sft")


def _slot(vectors, final=True):
    structure = fast_structure(NETLIST)
    simulate = fast_simulator(structure, FAULT, ["y"])
    user = "Target sa0 n1 in tiny.\n\n" + repr({"doc_id": "d", "netlist": NETLIST})
    messages = []
    for vector in vectors:
        messages.append({"role": "assistant", "content": "<think>\ntry\n</think>\n\n", "tool_calls": [
            {"type": "function", "function": {"name": "fault_simulation_tool", "arguments": {
                "input_vector": dict(vector), "output_vector": {"y": 1}, "fault": FAULT, "doc_id": "d"}}}]})
        messages.append({"role": "tool", "name": "fault_simulation_tool", "content": simulate(vector)})
    if final:
        messages.append({"role": "assistant", "content": "The fault sa0 n1 has been sensitized."})
    return {"prompt_messages": [{"role": "system", "content": "ATPG"}, {"role": "user", "content": user}],
            "messages": messages, "problem": ["tiny", FAULT], "difficulty": "easy", "_round": "r",
            "seed": 1, "completion_slot": 0, "fault": FAULT, "hint_mode": "none", "status": "FINAL",
            "reward_components": {"detection": 0.0}}


def test_slot_calls_reads_multi_call_trajectories():
    b = _builder()
    slot = _slot([{"a": 0, "b": 1, "c": 1}, {"a": 0, "b": 0, "c": 1}])
    user, calls = b._slot_calls(slot)
    assert len(calls) == 2 and calls[-1][1]["input_vector"] == {"a": 0, "b": 0, "c": 1}
    # RFT slots keep the prompt in front of the generated turns.
    with_prompt = dict(slot, messages=slot["prompt_messages"] + slot["messages"])
    assert len(b._slot_calls(with_prompt)[1]) == 2
    assert b._slot_parts(slot)[2]["input_vector"] == {"a": 0, "b": 0, "c": 1}
    # A controller message between calls is not a plain call trajectory.
    slot["messages"].insert(2, {"role": "user", "content": "The previous action was invalid"})
    assert b._slot_calls(slot) is None


def test_search_continues_from_the_last_failed_call_within_the_call_budget():
    b = _builder()
    slot = _slot([{"a": 0, "b": 1, "c": 1}, {"a": 0, "b": 0, "c": 1}])
    row = b._search_one((slot, 4, 3, 8))
    assert row["status"] == "path"
    assert row["steps"][-1]["vector"] == {"a": 1, "b": 1, "c": 0}
    # The first failed vector is never proposed again.
    assert all(step["vector"] != {"a": 0, "b": 1, "c": 1} for step in row["steps"])
    # No call left: two failed calls under a cap of two.
    assert b._search_one((slot, 4, 3, 2))["status"] == "no_path"


def test_example_from_context_masks_every_failed_call():
    from atpgllm.training.retry_trajectories import retry_example_from_context
    structure = fast_structure(NETLIST)
    simulate = fast_simulator(structure, FAULT, ["y"])
    search = RepairSearch(structure, simulate, FAULT, ["a", "b", "c"], ["y"])
    tried = [{"a": 0, "b": 1, "c": 1}, {"a": 0, "b": 0, "c": 1}]
    context = [(call_message("x", v, {"y": 1}, FAULT, "d"), simulate(v)) for v in tried]
    previous = search.explain(tried[-1], context[-1][1])
    steps = []
    for step in search.path(tried[-1], start_snapshot=context[-1][1], avoid=tried[:-1]):
        steps.append({"vector": step.observation.vector, "changes": list(step.changes),
                      "snapshot": step.observation.snapshot, "prev_reading": previous.reading,
                      "prev_blockers": list(previous.blockers)})
        previous = step.observation
    example = retry_example_from_context(PROMPT, context, steps, FAULT, "d", ["y"], TOOLS)
    assert example["assistant_train"][:2] == [False, False]
    assert all(example["assistant_train"][2:])
    calls = [m for m in example["messages"] if m.get("tool_calls")]
    assert len(calls) == 2 + len(steps)
    assert "does not detect" in [m["content"] for m in example["messages"] if m["role"] == "assistant"][2]
