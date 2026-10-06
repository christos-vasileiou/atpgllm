"""Multi-turn tool-calling rollouts for graph-conditioned GRPO.

A trajectory alternates model turns and fault-simulation results: the model
writes a turn; if the turn ends with a tool call, the call is executed and the
tool's answer is appended in the chat template's tool format; the model then
writes the next turn. A trajectory ends with a turn that calls no tool, at
``max_tool_calls``, or when ``max_completion_tokens`` is spent.

Every trajectory keeps two masks over its completion tokens: all of them are
attended to, but only the tokens the model wrote are trained (``loss_mask``);
the tool's answers are context. The rendering matches Stage C's SFT
conversations token for token (``tests/unit/test_tool_rollout.py``), so the
policy is rolled out in the format it was fine-tuned on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

from atpgllm.training.revert_template import parse_tool_call


@dataclass
class Trajectory:
    completion_ids: List[int] = field(default_factory=list)
    loss_mask: List[int] = field(default_factory=list)
    turns: List[str] = field(default_factory=list)         # the model's turns
    tool_results: List[str] = field(default_factory=list)  # what the tool returned
    truncated: bool = False       # ran out of tokens mid-turn or before a reply
    call_limit_hit: bool = False  # asked for a call past max_tool_calls


def turn_stop(tokenizer) -> Tuple[str, int]:
    """The token that ends a model turn (Qwen2.5: ``<|im_end|>``)."""
    text = tokenizer.eos_token
    return text, tokenizer.convert_tokens_to_ids(text)


def tool_turn_tokens(tokenizer, content: str) -> List[int]:
    """Tokens between a model turn that ended on its stop token and the next
    model turn: what the chat template writes there in a whole conversation
    (the rest of the closed turn, the tool answer, the next generation
    prompt). The text after a special token tokenizes the same on its own,
    so this matches the SFT rendering token for token."""
    stop, _ = turn_stop(tokenizer)
    base = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]
    closed = tokenizer.apply_chat_template(base, tokenize=False)
    full = tokenizer.apply_chat_template(
        base + [{"role": "tool", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not full.startswith(closed) or stop not in closed:
        raise ValueError("The chat template is not prefix-preserving.")
    after_stop = closed[closed.rindex(stop) + len(stop):]
    return tokenizer.encode(after_stop + full[len(closed):], add_special_tokens=False)


def rollout_group(
    generate_turn: Callable[[List[List[int]], int], List[List[int]]],
    tokenizer,
    prompt_ids: List[int],
    run_tools: Callable[[List[Dict[str, Any]]], List[str]],
    *,
    num_generations: int,
    max_tool_calls: int,
    max_completion_tokens: int,
) -> List[Trajectory]:
    """Roll out ``num_generations`` trajectories for one prompt.

    ``generate_turn(contexts, max_new_tokens)`` returns the new tokens for each
    context (one batched generation per turn); ``run_tools(calls)`` returns the
    tool's answer for each parsed call.
    """
    stop_text, stop = turn_stop(tokenizer)
    trajectories = [Trajectory() for _ in range(num_generations)]
    calls = [0] * num_generations
    active = list(range(num_generations))
    while active:
        budgets = [max_completion_tokens - len(trajectories[i].completion_ids) for i in active]
        outputs = generate_turn(
            [prompt_ids + trajectories[i].completion_ids for i in active], max(budgets)
        )
        pending: List[Tuple[int, Dict[str, Any]]] = []
        for i, new, budget in zip(active, outputs, budgets):
            trajectory = trajectories[i]
            new = list(new)
            if stop in new:
                new = new[: new.index(stop) + 1]
            new = new[:budget]
            trajectory.completion_ids += new
            trajectory.loss_mask += [1] * len(new)
            text = tokenizer.decode(new, skip_special_tokens=False)
            trajectory.turns.append(text.removesuffix(stop_text))
            if not new or new[-1] != stop:
                trajectory.truncated = True
                continue
            call = parse_tool_call(text)
            if call is None:
                continue
            if calls[i] >= max_tool_calls:
                trajectory.call_limit_hit = True
                continue
            pending.append((i, call))
        results = run_tools([call for _, call in pending]) if pending else []
        active = []
        for (i, _), content in zip(pending, results):
            trajectory = trajectories[i]
            calls[i] += 1
            trajectory.tool_results.append(content)
            block = tool_turn_tokens(tokenizer, content)
            if len(trajectory.completion_ids) + len(block) >= max_completion_tokens:
                trajectory.truncated = True
                continue
            trajectory.completion_ids += block
            trajectory.loss_mask += [0] * len(block)
            active.append(i)
    return trajectories


_REPORT = re.compile(r"\b(INPUT_VECTOR|EXPECTED_OUTPUT|DETECTED_FAULTS):")


def reward_text(trajectory: Trajectory) -> str:
    """The completion the reward reads. The reward scores the first
    ``INPUT_VECTOR:`` it finds, so reports written in earlier turns are
    renamed: only the final turn's answer is scored."""
    parts: List[str] = []
    last = len(trajectory.turns) - 1
    for index, turn in enumerate(trajectory.turns):
        parts.append(
            turn if index == last
            else _REPORT.sub(lambda m: m.group(1).lower() + " (draft):", turn)
        )
        if index < len(trajectory.tool_results):
            parts.append(
                "<tool_response>\n" + trajectory.tool_results[index] + "\n</tool_response>"
            )
    return "\n".join(parts)
