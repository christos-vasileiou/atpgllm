"""Repair paths from a failed call to a detecting one."""

import json

from atpgllm.training.retry_repair import (
    RepairSearch,
    blocking_side_inputs,
    fast_simulator,
    fast_structure,
)
from atpgllm.training.tool_feedback import (
    blocked_detail,
    read_snapshot,
    repair_sentence,
    undetected_sentence,
)

# y = (a & b) | c. For sa0 n1: a=b=1 activates it, and c must be 0 or the
# OR gate blocks it.
NETLIST = """module tiny(a, b, c, y);
input a, b, c;
output y;
wire n1;
AND2x2_ASAP7_75t_R U1 (.A(a), .B(b), .Y(n1));
OR2x2_ASAP7_75t_R U2 (.A(n1), .B(c), .Y(y));
endmodule"""


def _search(fault="sa0 n1"):
    structure = fast_structure(NETLIST)
    simulate = fast_simulator(structure, fault, ["y"])
    return RepairSearch(structure, simulate, fault, ["a", "b", "c"], ["y"]), simulate


def test_fanin_follows_the_gates_back_to_inputs():
    search, _ = _search()
    assert search.fanin_inputs(["n1"]) == ["a", "b"]
    assert search.fanin_inputs(["y"]) == ["a", "b", "c"]


def test_blocked_reading_names_the_side_input():
    search, simulate = _search()
    table = simulate({"a": 1, "b": 1, "c": 1})
    data = json.loads(table)
    blockers, propagation, controlling = blocking_side_inputs(
        search.graph, data["Good Machine"], data["Bad Machine"], "n1")
    assert blockers == [("y", "n1", "c", "1")]
    assert 0 < propagation < 1 and controlling == 0
    text = blocked_detail(blockers)
    assert "the gate driving y reads the difference on n1" in text
    assert "c = 1" in text


def test_path_activates_first_then_unblocks():
    search, _ = _search()
    steps = search.path({"a": 0, "b": 1, "c": 1})
    assert steps is not None and len(steps) == 2
    first, second = steps
    assert first.changes == (("a", 1),)
    assert first.observation.reading.activated and not first.observation.reading.detected
    assert first.observation.blockers == (("y", "n1", "c", "1"),)
    assert second.changes == (("c", 0),)
    assert second.observation.reading.detected
    assert second.observation.vector == {"a": 1, "b": 1, "c": 0}


def test_one_step_when_a_single_change_detects():
    search, _ = _search()
    steps = search.path({"a": 1, "b": 1, "c": 1})
    assert [s.changes for s in steps] == [(("c", 0),)]


def test_detecting_start_and_unreachable_fault_give_no_path():
    search, _ = _search()
    assert search.path({"a": 1, "b": 1, "c": 0}) is None
    # Two flips are needed to activate from a=b=0, so a one-flip search fails.
    narrow = RepairSearch(search.structure, search.simulate_table, "sa0 n1",
                          ["a", "b", "c"], ["y"], max_flips=1)
    assert narrow.path({"a": 0, "b": 0, "c": 0}, max_steps=1) is None


def test_start_snapshot_is_read_as_given():
    search, simulate = _search()
    table = simulate({"a": 0, "b": 1, "c": 1})
    steps = search.path({"a": 0, "b": 1, "c": 1}, start_snapshot=table)
    assert steps is not None and steps[-1].observation.reading.detected


def test_repair_sentences_follow_the_reading():
    _, simulate = _search()
    unexcited = read_snapshot(simulate({"a": 0, "b": 1, "c": 1}), "sa0 n1", ["y"])
    assert "never activated: n1 must be 1" in undetected_sentence(unexcited)
    assert repair_sentence(unexcited, [("a", 1)]).startswith("I'll set a to 1 to drive n1 to 1")
    blocked = read_snapshot(simulate({"a": 1, "b": 1, "c": 1}), "sa0 n1", ["y"])
    text = repair_sentence(blocked, [("c", 0)], unblocked=[("c", 0)])
    assert "to bring c to 0 so the fault effect can pass" in text
