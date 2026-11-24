"""v5r data design: output pairs before every reading, plan-only call thinking, reading metrics."""
import json
import sys
from pathlib import Path

from atpgllm.training.conversation import ConversationExample
from atpgllm.training.retry_repair import RepairSearch, fast_simulator, fast_structure
from atpgllm.training.retry_trajectories import call_message, retry_example
from atpgllm.training.tool_feedback import output_pairs, with_outputs
from atpgllm.training.tools import TOOLS

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "retry"))

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
    failed = call_message("my own attempt", first.vector, {"y": 1}, FAULT, "d")
    return retry_example(PROMPT, failed, first.snapshot, steps, FAULT, "d", ["y"], TOOLS)


def test_output_pairs_keep_the_given_order_and_show_unknowns():
    snapshot = json.dumps({"Good Machine": {"q": "x", "y": 1, "z": 0}, "Bad Machine": {"q": 1, "y": 0, "z": 0}})
    assert output_pairs(snapshot, ["z", "y", "q"]) == "Primary outputs, Good/Bad: z 0/0, y 1/0, q x/1."
    assert with_outputs("Verdict.", snapshot, ["y"], enabled=False) == "Verdict."
    assert with_outputs("Verdict.", snapshot, ["y"], enabled=True) == "Primary outputs, Good/Bad: y 1/0. Verdict."
    assert with_outputs("Verdict.", None, ["y"], enabled=True) == "Verdict."


def test_every_reading_lists_the_outputs_of_the_answer_it_reads(monkeypatch):
    monkeypatch.setenv("SFT_READ_OUTPUTS", "1")
    example = _example()
    turns = [m["content"] for m in example["messages"] if m["role"] == "assistant"][1:]
    tables = [m["content"] for m in example["messages"] if m["role"] == "tool"]
    # Turn k reads tool answer k: its pairs are that answer's y values, then the verdict.
    for turn, table in zip(turns, tables):
        machines = json.loads(table)
        pairs = f"Primary outputs, Good/Bad: y {machines['Good Machine']['y']}/{machines['Bad Machine']['y']}."
        assert turn.startswith("<think>\n" + pairs + " The Good and Bad Machine ")
    assert "agree on every primary output" in turns[0] and "I'll set a to 1" in turns[0]
    assert "differ at 1 primary output: y" in turns[-1]
    monkeypatch.delenv("SFT_READ_OUTPUTS")
    assert "Primary outputs, Good/Bad" not in json.dumps(_example()["messages"])


def _record(snapshot):
    return {
        "netlist": NETLIST, "module_name": "tiny", "circuit_id": "tiny", "fault": FAULT,
        "input_vector": json.dumps({"a": 1, "b": 1, "c": 0}), "expected_output": json.dumps({"y": 1}),
        "fault_propagation_gates": "U1, U2", "backtrack_gates": "U1, U2", "backtrack_nets": "a, b, c",
        "detected_faults": "sa0 n1", "system_content": "ATPG assistant.",
        "user_content": "Target {fault} in {module_name}: {netlist}",
        "reasoning_content": "Drive n1 to 1 and keep c at 0.\nSet a: 1, b: 1, c: 0.\nExpected output y: 1.",
        "answer_content": 'INPUT_VECTOR: "{input_vector}"\n', "snapshot": snapshot,
    }


def test_plan_think_and_output_pairs_in_ordinary_rows(monkeypatch):
    _, steps = _path({"a": 1, "b": 1, "c": 1})
    record = _record(steps[-1]["snapshot"])
    monkeypatch.setenv("SFT_GROUNDED_TEXT", "1")
    full = ConversationExample.from_record(dict(record), use_tools=True).messages
    assert "a: 1, b: 1" in full[2]["content"]
    monkeypatch.setenv("SFT_PLAN_THINK", "1")
    monkeypatch.setenv("SFT_READ_OUTPUTS", "1")
    messages = ConversationExample.from_record(dict(record), use_tools=True).messages
    call, final = messages[2]["content"], messages[-1]["content"]
    assert "Drive n1 to 1" in call and "a: 1, b: 1" not in call and "Expected output" not in call
    assert final.startswith("<think>\nPrimary outputs, Good/Bad: y 1/0. The Good and Bad Machine differ at 1")


def test_reading_claims_and_fault_ids():
    from gate_report import fault_of, reading_claims
    messages = [
        {"role": "tool", "content": "{}"},
        {"role": "assistant", "content": "<think>\nPrimary outputs, Good/Bad: y 1/1. The Good and Bad Machine "
                                         "agree on every primary output, so this pattern does not detect sa0 n1."},
        {"role": "tool", "content": "{}"},
        {"role": "assistant", "content": "<think>\nThe Good and Bad Machine differ at 1 primary output: y."
                                         "\n</think>\n\nTo sum up, the vectors are: x"},
        {"role": "tool", "content": "{}"},
        {"role": "assistant", "content": "To sum up, the vectors are: y"},
    ]
    assert reading_claims(messages) == ["miss", "detected", None]
    assert fault_of("netlist123|sa0 n1") == "sa0 n1"
    assert fault_of("abc", {"abc": "sa1 q"}) == "sa1 q"
    assert fault_of("abc") is None
