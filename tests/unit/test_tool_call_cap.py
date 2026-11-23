"""GRPO_MAX_TOOL_CALLS: executed tool calls allowed per GRPO completion."""

import pytest

from atpgllm.training.tools import executed_tool_calls, max_tool_calls, may_call_again


def _completion(calls):
    messages = [{"role": "assistant", "content": "x"}]
    for _ in range(calls):
        messages += [{"role": "tool", "content": "{}"}, {"role": "assistant", "content": "y"}]
    return messages


def test_cap_setting(monkeypatch):
    monkeypatch.delenv("GRPO_MAX_TOOL_CALLS", raising=False)
    assert max_tool_calls() is None
    monkeypatch.setenv("GRPO_MAX_TOOL_CALLS", "0")
    assert max_tool_calls() is None
    monkeypatch.setenv("GRPO_MAX_TOOL_CALLS", "8")
    assert max_tool_calls() == 8
    monkeypatch.setenv("GRPO_MAX_TOOL_CALLS", "-1")
    with pytest.raises(ValueError):
        max_tool_calls()


def test_only_executed_calls_count():
    assert executed_tool_calls("plain text completion") == 0
    assert executed_tool_calls(_completion(3)) == 3


def test_eighth_call_executes_and_ninth_does_not():
    assert may_call_again(_completion(7), 8)
    assert not may_call_again(_completion(8), 8)
    assert may_call_again(_completion(50), None)
