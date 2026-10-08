"""A repeated tool call is answered from the earlier observation, not simulated again."""

from atpgllm.training import tools
from atpgllm.training.tools import ToolScheduler, repeat_key


def _call(vector, outputs="y=1", fault="sa0 n1"):
    return {"name": "fault_simulation_tool",
            "arguments": {"input_vector": vector, "output_vector": outputs,
                          "fault": fault, "doc_id": "d"}}


def test_repeated_call_reuses_the_observation(monkeypatch):
    executed = []

    def execute(item, functions, cancelled):
        call, _ = item
        executed.append(call["arguments"]["input_vector"])
        return f"observation {len(executed)}", False

    monkeypatch.setattr(tools, "_execute_tool", execute)
    with ToolScheduler({}) as scheduler:
        _, _, first = scheduler.take(None, [_call("a=1, b=0")], [0], ["p"])
        # Same vector written in another order, different claimed output value.
        _, _, again = scheduler.take(None, [_call("b=0,a=1", outputs="y=0")], [0], ["p"])
        # Another trajectory with the same call is simulated on its own.
        _, _, other = scheduler.take(None, [_call("a=1, b=0")], [1], ["p", "p"])
    assert executed == ["a=1, b=0", "a=1, b=0"]
    assert again == first
    assert other == [("observation 2", False)]
    assert scheduler.reused == 1


def test_repeat_key_separates_what_changes_the_observation():
    base = repeat_key(_call("a=1, b=0"))
    assert repeat_key(_call("a:1,b:0", outputs="y=0")) == base
    assert repeat_key(_call("a=1, b=1")) != base
    assert repeat_key(_call("a=1, b=0", outputs="z=1")) != base
    assert repeat_key(_call("a=1, b=0", fault="sa1 n1")) != base
