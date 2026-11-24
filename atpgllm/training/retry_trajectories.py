"""Retry SFT examples: a failed call, the simulator's answer, and calls until one detects.

Each example starts with a call that did not detect the fault (the policy's own
failed attempt, or a perturbed dataset pattern) and the tool's real answer for
it. Every later assistant turn reads the last answer in its thinking block
(not activated, or activated and blocked at named side inputs), changes the
inputs the reading points to, and calls the tool again, until a call detects.
The final turn names the primary outputs where the two machines differ and
reports the detecting pattern.

The failed first call is context: ``assistant_train`` marks it False, so SFT
does not teach the model to write it. Every turn after it is a target.

The same module grounds existing single-call examples: the stock sentence after
the thinking block of a call ("I have to verify ...") is dropped and the stock
final lead ("The fault ... has been sensitized ...") is replaced by what the
last tool answer shows.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple

from atpgllm.training.tool_feedback import (
    SimulationReading,
    blocked_detail,
    detected_sentence,
    read_snapshot,
    repair_sentence,
    undetected_sentence,
    with_outputs,
)

TOOL_NAME = "fault_simulation_tool"
STOCK_CALL_LINE = (
    "I have to verify if the fault is detected by the input and output vectors. "
    "I need to call the fault simulation tool."
)
# The policy also writes the line with the fault named ("... if the fault sa0 n27 is ...").
STOCK_CALL_RE = re.compile(
    r"I have to verify if the fault (?:.{1,200}? )?is detected by the input and output vectors\. "
    r"I need to call the fault simulation tool\.\s*"
)
STOCK_LEAD_RE = re.compile(
    r"The fault .{1,400}? has been sensitized\. I can see the difference between Good/Bad Machine\.\s*"
)


# A "net: value" or "net=value" assignment inside reasoning text.
_ASSIGNMENT = re.compile(r"[\w\\\[\]/.$:-]+?\s*[:=]\s*[01xX]\b")
# The value the plan needs on the fault site or a net of its path ("satisfy n2 = 0")
# is the same for every detecting vector, so that one assignment stays.
_OBJECTIVE = re.compile(r"\bsatisfy\s+\S+\s*=\s*[01xX]\b")
# Wide vectors are written compactly (conversation.compact_binary_assignment_dict):
# "in1[0..31] ones={2..3, 26}" and "a[0..7] bits_lsb0=01111111".
_COMPACT_VECTOR = re.compile(r"\b(?:ones|zeros)\s*=\s*\{|\bbits_lsb0\s*=")
_THINK_OPEN = "<think>"


def _carries_values(line: str) -> bool:
    if _COMPACT_VECTOR.search(line) or "expected output" in line.lower():
        return True
    return _ASSIGNMENT.search(_OBJECTIVE.sub("", line)) is not None


def strip_assignment_lines(content: str) -> str:
    """The thinking block without the lines that state input or output values.

    A synthetic retry example starts from a dataset row whose reasoning spells
    out the detecting vector, while its first call uses a perturbed one. Kept as
    context, those lines would let the next call copy the answer instead of
    reading the simulator's table, and a plan whose vector disagrees with its
    call would mark the call as a failure before the table is read. So only the
    structural plan (net and gate names, and the value the fault site needs)
    stays, in every row that uses this filter. A line goes when it holds a
    plain assignment, a compact vector or an expected output. The thinking
    opener stays when the first line goes, and text after the thinking block is
    unchanged.
    """
    head, sep, tail = content.partition("</think>")
    opener = ""
    if head.startswith(_THINK_OPEN):
        opener, head = _THINK_OPEN, head[len(_THINK_OPEN):]
    kept = [line for line in head.split("\n") if not _carries_values(line)]
    return opener + "\n".join(kept) + sep + tail


def _care(value: Any) -> bool:
    return str(value).strip() in ("0", "1")


def answer_text(vector: Dict[str, Any], good: Dict[str, Any], output_nets: Sequence[str],
                fault: str) -> str:
    """The final report: care bits of the detecting call and its Good Machine outputs."""
    inputs = ", ".join(f"{net}: {value}" for net, value in vector.items() if _care(value))
    outputs = ", ".join(f"{net}: {good[net]}" for net in output_nets
                        if net in good and _care(good[net]))
    return (
        "To sum up, the vectors are:\n"
        f'INPUT_VECTOR: "{inputs}"\n'
        f'EXPECTED_OUTPUT: "{outputs}"\n'
        f'DETECTED_FAULTS: "{fault}" by propagating the fault forward.\n'
    )


def call_message(think: str, vector: Dict[str, Any], output_vector: Dict[str, Any],
                 fault: str, doc_id: str, post: str = "") -> Dict[str, Any]:
    """An assistant tool call in the SFT layout (vectors as JSON text, no call id)."""
    arguments = {
        "input_vector": json.dumps(vector, ensure_ascii=False),
        "output_vector": json.dumps(output_vector, ensure_ascii=False),
        "fault": fault,
        "doc_id": doc_id,
    }
    return {
        "role": "assistant",
        "content": "<think>\n" + think.strip() + "\n</think>\n\n" + post,
        "tool_calls": [{"type": "function", "function": {"name": TOOL_NAME, "arguments": arguments}}],
    }


def tool_message(snapshot: str) -> Dict[str, Any]:
    return {"role": "tool", "name": TOOL_NAME, "content": snapshot}


def outputs_of(snapshot: str, output_nets: Sequence[str]) -> Dict[str, Any]:
    """Good Machine values at the primary outputs, in declared order."""
    good = json.loads(snapshot)["Good Machine"]
    return {net: good[net] for net in output_nets if net in good}


def unblocked_sides(blockers: Sequence[Tuple[str, str, str, Optional[str]]],
                    next_snapshot: str, changed_inputs: Sequence[str] = ()) -> List[Tuple[str, Any]]:
    """Blocking side inputs whose Good Machine value the next call changes.

    Inputs the call sets directly are left out: the sentence already names them.
    """
    good = json.loads(next_snapshot)["Good Machine"]
    changed, seen = [], set(changed_inputs)
    for _, _, side, value in blockers:
        if side in seen or side not in good:
            continue
        seen.add(side)
        new = str(good[side])
        if _care(new) and new != value:
            changed.append((side, new))
    return changed


def retry_think(reading: SimulationReading, blockers, changes, next_snapshot: str,
                snapshot: Optional[str] = None, output_nets: Sequence[str] = ()) -> str:
    """Reasoning for the call after an undetected answer.

    ``snapshot`` is the undetected answer itself: with ``SFT_READ_OUTPUTS`` the
    reading lists its output pairs before the verdict.
    """
    parts = [with_outputs(undetected_sentence(reading), snapshot, output_nets)]
    if reading.activated and blockers:
        parts.append(blocked_detail(blockers))
    unblocked = (unblocked_sides(blockers, next_snapshot, [net for net, _ in changes])
                 if reading.activated else [])
    parts.append(repair_sentence(reading, changes, unblocked))
    return " ".join(part for part in parts if part)


def final_message(reading: SimulationReading, vector: Dict[str, Any], snapshot: str,
                  output_nets: Sequence[str], fault: str) -> Dict[str, Any]:
    good = json.loads(snapshot)["Good Machine"]
    return {
        "role": "assistant",
        "content": ("<think>\n" + with_outputs(detected_sentence(reading), snapshot, output_nets)
                    + "\n</think>\n\n" + answer_text(vector, good, output_nets, fault)),
    }


def retry_example(prompt_messages: List[Dict[str, Any]], first_call: Dict[str, Any],
                  first_snapshot: str, steps: Sequence[Dict[str, Any]], fault: str,
                  doc_id: str, output_nets: Sequence[str], tools: Any) -> Dict[str, Any]:
    """The SFT example for one repair path after a single failed call.

    ``first_call`` is the failed assistant call in the SFT layout and
    ``first_snapshot`` the tool's answer for it. Each step holds ``vector``,
    ``changes``, ``snapshot`` (the tool's answer, as GDPO's tool returns it)
    and ``reading``/``blockers`` of the snapshot *before* the step
    (``prev_reading``, ``prev_blockers``). The last step must detect.
    """
    return retry_example_from_context(prompt_messages, [(first_call, first_snapshot)], steps,
                                      fault, doc_id, output_nets, tools)


def retry_example_from_context(prompt_messages: List[Dict[str, Any]],
                               context: Sequence[Tuple[Dict[str, Any], str]],
                               steps: Sequence[Dict[str, Any]], fault: str, doc_id: str,
                               output_nets: Sequence[str], tools: Any) -> Dict[str, Any]:
    """The SFT example for a repair path after one or more failed calls.

    ``context`` holds the failed calls in order, each with the tool's answer
    for it. They are kept as context and left out of the loss: the model learns
    to read the last answer and recover, not to make those calls. Every step
    after them is a target, as in :func:`retry_example`.
    """
    messages = deepcopy(prompt_messages)
    for call, snapshot in context:
        messages += [deepcopy(call), tool_message(snapshot)]
    train = [False] * len(context)
    previous = context[-1][1] if context else None
    for step in steps:
        think = retry_think(step["prev_reading"], step["prev_blockers"], step["changes"], step["snapshot"],
                            snapshot=previous, output_nets=output_nets)
        previous = step["snapshot"]
        messages.append(call_message(think, step["vector"], outputs_of(step["snapshot"], output_nets),
                                     fault, doc_id))
        messages.append(tool_message(step["snapshot"]))
        train.append(True)
    last = steps[-1]
    reading = read_snapshot(last["snapshot"], fault, output_nets)
    if not reading.detected:
        raise ValueError("The last call of a retry example must detect the fault")
    messages.append(final_message(reading, last["vector"], last["snapshot"], output_nets, fault))
    train.append(True)
    return {"messages": messages, "tools": tools, "assistant_train": train}


def sft_first_call(slot_message: Dict[str, Any]) -> Dict[str, Any]:
    """A slot's assistant call in the SFT layout, as build_sft writes it."""
    entry = {"role": "assistant", "content": slot_message.get("content") or ""}
    calls = []
    for call in slot_message.get("tool_calls") or []:
        arguments = deepcopy(call["function"]["arguments"])
        for key in ("input_vector", "output_vector"):
            if isinstance(arguments.get(key), dict):
                arguments[key] = json.dumps(arguments[key], ensure_ascii=False)
        calls.append({"type": "function", "function": {"name": call["function"]["name"],
                                                       "arguments": arguments}})
    entry["tool_calls"] = calls
    return entry


def ground_example(example: Dict[str, Any], fault: str,
                   output_nets: Sequence[str]) -> Optional[Dict[str, Any]]:
    """A detecting single-call example with the two stock lines replaced.

    The stock line after a call's thinking block is dropped. The final turn's
    stock lead, inside and after its thinking block, becomes the reading of the
    last tool answer. None when the last answer does not detect the fault.
    """
    out = deepcopy(example)
    messages = out["messages"]
    tools = [m for m in messages if m["role"] == "tool"]
    if not tools:
        return None
    reading = read_snapshot(tools[-1]["content"], fault, output_nets)
    if not reading.detected:
        return None
    lead = with_outputs(detected_sentence(reading), tools[-1]["content"], output_nets)
    for message in messages:
        if message["role"] != "assistant":
            continue
        content = message.get("content") or ""
        if message.get("tool_calls"):
            if STOCK_CALL_RE.search(content):
                message["content"] = STOCK_CALL_RE.sub("", content).rstrip() + "\n\n"
            continue
        if STOCK_LEAD_RE.search(content):
            # A function, so backslashes in net names are not read as escapes.
            message["content"] = STOCK_LEAD_RE.sub(lambda _: lead + " ", content)
    return out
