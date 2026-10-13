"""Stage-D rollouts reproduce the Stage-C SFT rendering token for token."""

import copy
import random

import pytest

from atpgllm.training.conversation import ConversationExample
from atpgllm.training.tool_trajectories import failed_attempt
from test_tool_trajectories import _record, granite_tokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return granite_tokenizer()


def _replay(tokenizer, conversation):
    """Feed the conversation's assistant turns to the rollout as the model's
    generations and its tool messages as the tool's answers."""
    from atpgllm.multimodal.data import _template_ids, template_turns, tokenize_tool_conversation
    from atpgllm.multimodal.tool_rollout import rollout_group

    messages = template_turns(conversation.messages, tokenizer)
    ids, labels = tokenize_tool_conversation(messages, None, tokenizer)
    first = next(i for i, m in enumerate(messages) if m["role"] == "assistant")
    prompt = _template_ids(tokenizer, messages[:first], generation_prompt=True)
    turns, start = [], None
    for position, label in enumerate(labels + [-100]):
        if label != -100 and start is None:
            start = position
        elif label == -100 and start is not None:
            turns.append(ids[start:position])
            start = None
    answers = [m["content"] for m in messages if m["role"] == "tool"]
    calls = []

    def generate_turn(contexts, max_new_tokens):
        assert len(contexts) == 1
        assert contexts[0] == ids[: len(contexts[0])]   # same context as SFT
        return [turns.pop(0)]

    def run_tools(batch):
        calls.extend(batch)
        return [answers.pop(0) for _ in batch]

    trajectory = rollout_group(
        generate_turn, tokenizer, prompt, run_tools,
        num_generations=1, max_tool_calls=3, max_completion_tokens=10**6,
    )[0]
    return ids, labels, prompt, trajectory, calls


def _expected_mask(tokenizer, ids, labels, prompt):
    """SFT labels, except the template token after each turn's stop token,
    which the rollout appends as context rather than generating."""
    from atpgllm.multimodal.tool_rollout import turn_stop

    _, stop = turn_stop(tokenizer)
    mask = [int(label != -100) for label in labels]
    for position in range(1, len(ids)):
        if ids[position - 1] == stop:
            mask[position] = 0
    return mask[len(prompt): len(ids) - 1]


@pytest.mark.parametrize("retry", [False, True])
def test_rollout_matches_sft_tokens_and_masks(tokenizer, retry):
    record = _record()
    attempt = failed_attempt(record, random.Random(3)) if retry else None
    conversation = ConversationExample.from_record(
        copy.deepcopy(record), use_tools=True, failed_attempt=attempt,
        interpret_results=True, tool_view="po_fault",
    )
    ids, labels, prompt, trajectory, calls = _replay(tokenizer, conversation)
    # SFT ends with the template's newline after the last stop token.
    assert prompt + trajectory.completion_ids == ids[:-1]
    assert trajectory.loss_mask == _expected_mask(tokenizer, ids, labels, prompt)
    assert len(calls) == (2 if retry else 1)
    assert calls[-1]["arguments"]["input_vector"] == {"a": 1, "b": 1, "c": 0}
    assert not trajectory.truncated and not trajectory.call_limit_hit
    # Granite's generation prompt opens <think>, so every turn the model writes
    # starts inside it: call turns close non-empty reasoning, the answer turn
    # closes an empty block. That is what the GRPO format objective checks.
    from atpgllm.llm.reward_funcs import _reasoned, prompt_opens_thinking

    assert prompt_opens_thinking(tokenizer.decode(prompt))
    assert all(_reasoned(turn, True) for turn in trajectory.turns[:-1])
    assert trajectory.turns[-1].startswith("</think>\nThe Good and Bad Machine differ")


def test_call_limit_and_budget_end_trajectories(tokenizer):
    from atpgllm.multimodal.tool_rollout import rollout_group

    call = tokenizer.encode(
        '<tool_call>\n{"name": "fault_simulation_tool", "arguments": {}}\n</tool_call><|im_end|>',
        add_special_tokens=False,
    )
    always_call = lambda contexts, n: [list(call) for _ in contexts]
    answer = lambda batch: ["{}" for _ in batch]
    limited = rollout_group(always_call, tokenizer, [1, 2], answer, num_generations=2,
                            max_tool_calls=1, max_completion_tokens=10**6)
    assert all(t.call_limit_hit and len(t.tool_results) == 1 for t in limited)
    assert all(len(t.turns) == 2 for t in limited)

    short = rollout_group(always_call, tokenizer, [1, 2], answer, num_generations=1,
                          max_tool_calls=3, max_completion_tokens=len(call) - 3)[0]
    assert short.truncated and not short.tool_results


def test_reward_reads_only_the_final_report():
    from atpgllm.multimodal.tool_rollout import Trajectory, reward_text

    trajectory = Trajectory(
        turns=['Trying INPUT_VECTOR: "a: 0"', 'Fixed. INPUT_VECTOR: "a: 1"'],
        tool_results=['{"Good Machine":{}}'],
    )
    text = reward_text(trajectory)
    assert text.count("INPUT_VECTOR:") == 1 and 'INPUT_VECTOR: "a: 1"' in text
    assert "<tool_response>" in text


def test_malformed_model_arguments_come_back_as_tool_errors():
    import threading

    from atpgllm.training.tools import (
        ToolContext,
        _execute_tool,
        fault_simulation_tool_handler,
    )

    record = _record()
    context = ToolContext(netlist=record["netlist"], expected_doc_id="d",
                          expected_fault="sa0 n1", view="po_fault")
    functions = {"fault_simulation_tool": fault_simulation_tool_handler}
    good = {"input_vector": {"a": 1, "b": 1, "c": 0}, "output_vector": {"y": 1},
            "fault": "sa0 n1", "doc_id": "d"}
    reply, failed = _execute_tool(
        ({"name": "fault_simulation_tool", "arguments": good}, context),
        functions, threading.Event(),
    )
    assert not failed and reply == '{"Good Machine":{"n1":1,"y":1},"Bad Machine":{"n1":0,"y":0}}'
    bad = dict(good, fault={"net": "n1"})
    reply, failed = _execute_tool(
        ({"name": "fault_simulation_tool", "arguments": bad}, context),
        functions, threading.Event(),
    )
    assert not failed and reply == (
        "Tool execution failed: fault must be a string such as 'sa0 <net>'"
    )


def test_reward_trace_pairs_each_executed_call_with_its_turn():
    from atpgllm.multimodal.tool_rollout import Trajectory, reward_trace

    call = '<tool_call>\n{"name": "fault_simulation_tool", "arguments": {"input_vector": "a=1"}}\n</tool_call>'
    trajectory = Trajectory(
        turns=["<think>r</think>\n" + call, "<think>r</think>\n" + call],
        tool_results=["{}"],
        call_limit_hit=True,
    )
    trace = reward_trace(trajectory)
    assert trace["calls"] == [{"arguments": {"input_vector": "a=1"}, "result": "{}"}]
    # The second call was not run, so its turn is the (unanswered) final turn.
    assert len(trace["turns"]) == 2



def test_repeated_call_reuses_the_earlier_answer(tokenizer):
    import json

    from atpgllm.multimodal.tool_rollout import reward_trace, rollout_group

    def turn(vector):
        arguments = {"input_vector": vector, "output_vector": {"y": 0},
                     "fault": "U1/A sa0", "doc_id": "d"}
        text = json.dumps({"name": "fault_simulation_tool", "arguments": arguments})
        return tokenizer.encode(f"<tool_call>\n{text}\n</tool_call><|im_end|>",
                                add_special_tokens=False)

    a, b = {"a": 1, "b": 0}, {"a": 0, "b": 0}
    plan = [[turn(a), turn(a)], [turn(a), turn(a)], [turn(b), turn(a)]]
    simulated = []

    def generate_turn(contexts, n):
        assert len(contexts) == 2
        if plan:
            return plan.pop(0)
        return [tokenizer.encode("done<|im_end|>", add_special_tokens=False)] * 2

    def run_tools(batch):
        answers = []
        for call in batch:
            simulated.append(call["arguments"]["input_vector"])
            answers.append(f"answer {len(simulated)}")
        return answers

    first, second = rollout_group(generate_turn, tokenizer, [1, 2], run_tools,
                                  num_generations=2, max_tool_calls=3,
                                  max_completion_tokens=10**6)
    # Each trajectory runs its own first call; nothing is shared between them.
    assert simulated == [a, a, b]
    assert first.tool_results == ["answer 1", "answer 1", "answer 3"]
    assert second.tool_results == ["answer 2"] * 3
    assert (first.reused_calls, second.reused_calls) == (1, 2)
    assert len(reward_trace(second)["calls"]) == 3   # repeats are still attempts
