"""Hint-conditioned rationalization (STaR) for faults random patterns rarely detect.

A model that never samples a detecting vector for a fault gets no reward signal
from it, in GRPO or in rejection sampling. Rationalization gives the model part
of a known test (from the dataset's TetraMAX pattern) as a hint, samples tool
conversations, keeps the ones whose final answer TetraMAX finds detecting, and
trains on them with the hint removed from the prompt.

This module holds the pure parts: choosing problems, writing the hint, judging
a trajectory, and turning it into a hint-free SFT example in the same message
layout as :func:`atpgllm.training.dataset_utils._format_sft_messages_record`.
``scripts/star/rationalize.py`` generates and ``scripts/star/build_sft.py``
filters.
"""
from __future__ import annotations

import json
import random
import re
from copy import deepcopy
from typing import Iterable, Optional

from .conversation import is_dont_care

HINT_MODES = ("none", "path", "partial", "cube")
HARD_LEVELS = ("hard", "very_hard", "random_never_detects")

# Assistant text that refers to the hint instead of deriving the test.
LEAK_RE = re.compile(
    r"\bhints?\b|\bprivate\b|\b(?:the|this|that) note\b|"
    r"\b(?:I|we) (?:was|were|am|are|have been|had been) (?:given|provided|told)\b|"
    r"\b(?:given|provided|known|verified) (?:test|cube|pattern|vector|assignment|bits?)\b",
    re.IGNORECASE)


def problem_key(row: dict) -> tuple:
    """A problem is a (netlist, fault) pair; several rows (patterns) share one."""
    return str(row.get("netlist_id") or row.get("netlist")), str(row.get("fault"))


def select_problems(rows: Iterable[dict], per_level: dict, seed: int = 0) -> list[dict]:
    """One seeded row per problem, ``per_level[level]`` problems per difficulty level.

    The row kept for a problem supplies its hint, so it is drawn uniformly
    among that problem's rows.
    """
    by_level: dict[str, dict[tuple, list[dict]]] = {level: {} for level in per_level}
    for row in rows:
        level = row.get("difficulty")
        if level in by_level:
            by_level[level].setdefault(problem_key(row), []).append(row)
    rng = random.Random(seed)
    chosen = []
    for level in per_level:
        problems = sorted(by_level[level])
        rng.shuffle(problems)
        for key in problems[:per_level[level]]:
            chosen.append(rng.choice(by_level[level][key]))
    return chosen


def _vector(row: dict, field: str) -> dict:
    value = row.get(field) or {}
    return json.loads(value) if isinstance(value, str) else value


def care_bits(row: dict) -> dict:
    return {net: value for net, value in _vector(row, "input_vector").items() if not is_dont_care(value)}


def _nets(raw) -> list[str]:
    return [token.strip() for token in str(raw or "").split(",") if token.strip()]


def hint_bits(row: dict, mode: str, seed: int = 0) -> dict:
    """The care bits a ``partial`` or ``cube`` hint reveals; empty for other modes."""
    if mode not in ("partial", "cube"):
        return {}
    bits = care_bits(row)
    names = sorted(bits)
    if mode == "partial" and names:
        rng = random.Random(f"{seed}:{problem_key(row)}")
        names = sorted(rng.sample(names, max(1, len(names) // 2)))
    return {name: bits[name] for name in names}


def hint_text(row: dict, mode: str, seed: int = 0) -> Optional[str]:
    """The hint appended to the user turn, or ``None`` for ``mode="none"``.

    ``path`` names the nets the dataset pattern sensitizes, without values.
    ``partial`` gives a seeded half of the pattern's care bits (at least one),
    ``cube`` all of them.
    """
    if mode not in HINT_MODES:
        raise ValueError(f"Unknown hint mode {mode!r}; expected one of {HINT_MODES}")
    if mode == "none":
        return None
    if mode == "path":
        propagate, control = _nets(row.get("fault_propagation_nets")), _nets(row.get("backtrack_nets"))
        if not propagate and not control:
            return None
        parts = []
        if propagate:
            parts.append(f"the fault effect can reach an output through {', '.join(propagate)}")
        if control:
            parts.append(f"the nets to justify include {', '.join(control)}")
        detail = "; ".join(parts)
    else:
        bits = hint_bits(row, mode, seed)
        if not bits:
            return None
        detail = ("one detecting test assigns " + ", ".join(f"{n}: {v}" for n, v in bits.items())
                  + ("" if mode == "cube" else " (other inputs are yours to choose)"))
    return ("\n\nPrivate note for this attempt: " + detail + ". Work out the test "
            "yourself from the netlist, check it with the fault simulation tool, and "
            "never mention this note.")


def hinted_messages(messages: list[dict], hint: Optional[str]) -> list[dict]:
    """``messages`` with ``hint`` appended to the last user turn."""
    out = deepcopy(messages)
    if hint:
        last_user = max(i for i, m in enumerate(out) if m["role"] == "user")
        out[last_user]["content"] += hint
    return out


def _assistant_text(messages: list[dict]) -> str:
    return "\n".join(str(m.get("content") or "") for m in messages if m["role"] == "assistant")


def judge(slot: dict, row: dict, hinted: Optional[dict] = None) -> dict:
    """Facts about one generated trajectory, used for filtering and the pilot report.

    ``slot`` is a search diagnostics entry (``messages``, ``status``,
    ``reward_components``, ``observations``) from
    :class:`atpgllm.training.search_policies.SamplingStrategy`. ``hinted`` is
    the care bits the hint revealed (:func:`hint_bits`).
    """
    messages = slot.get("messages") or []
    components = slot.get("reward_components") or {}
    text = _assistant_text(messages)
    calls = [o.get("vector") or {} for o in slot.get("observations") or []]
    bits = care_bits(row)
    first = calls[0] if calls else {}
    # Internal nets past the fault site: the prompt already names the inputs,
    # the outputs and the fault net, so mentioning those shows nothing.
    named = set(_vector(row, "input_vector")) | set(_vector(row, "expected_output"))
    named.add(str(row.get("fault", "")).split()[-1] if row.get("fault") else "")
    path = [n for n in _nets(row.get("fault_propagation_nets")) if n not in named]
    return {
        "final": slot.get("status") == "FINAL",
        "detected": bool(components.get("detection", 0) >= 1.0),
        "single_user_turn": sum(m["role"] == "user" for m in messages) == 1,
        "leak": bool(LEAK_RE.search(text)),
        "first_call_copies_cube": bool(first) and all(
            str(first.get(net)) == str(value) for net, value in bits.items()),
        "first_call_follows_hint": bool(first) and bool(hinted) and all(
            str(first.get(net)) == str(value) for net, value in hinted.items()),
        "names_internal_path_net": any(re.search(rf"(?<![\w\[]){re.escape(n)}(?![\w\]])", text) for n in path),
        "has_internal_path": bool(path),
        "tool_calls": len(calls),
    }


def keep(facts: dict, require_path_mention: bool = False) -> bool:
    """A trajectory is kept when its answer detects the fault, it took no
    controller repair turn, and no assistant turn refers to the hint."""
    ok = facts["final"] and facts["detected"] and facts["single_user_turn"] and not facts["leak"]
    if require_path_mention and facts["has_internal_path"]:
        ok = ok and facts["names_internal_path_net"]
    return ok


def sft_example(slot: dict, prompt_messages: list[dict], tools) -> dict:
    """The trajectory with the hint-free prompt, in the SFT ``messages`` layout.

    Tool-call vectors become JSON text and generation-only fields (call ids)
    are dropped, matching ``_format_sft_messages_record`` with vectors as text.
    """
    generated = [m for m in slot["messages"] if m["role"] in ("assistant", "tool")]
    messages = deepcopy(prompt_messages)
    for m in generated:
        if m["role"] == "tool":
            messages.append({"role": "tool", "name": m.get("name"), "content": m["content"]})
            continue
        entry = {"role": "assistant", "content": m.get("content") or ""}
        if m.get("tool_calls"):
            calls = []
            for call in m["tool_calls"]:
                arguments = deepcopy(call["function"]["arguments"])
                for key in ("input_vector", "output_vector"):
                    if isinstance(arguments.get(key), dict):
                        arguments[key] = json.dumps(arguments[key], ensure_ascii=False)
                calls.append({"type": "function",
                              "function": {"name": call["function"]["name"], "arguments": arguments}})
            entry["tool_calls"] = calls
        messages.append(entry)
    return {"messages": messages, "tools": tools}
