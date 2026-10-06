"""
conversation.py
===============

Defines :class:`ConversationExample`, which converts a raw dataset record
into a structured chat conversation suitable for SFT or GRPO training.

The ``from_record`` factory parses fault strings, derives placeholder
values, and conditionally renders the reasoning template (via
:func:`template_rendering.render_reasoning_template`) so that steps
referencing empty optional fields are replaced with generic explanations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import regex as re

from atpgllm.training.template_rendering import render_reasoning_template
from atpgllm.training.tools import FAULT_SIMULATION_TOOL

FAULT_PATTERN = re.compile(r"(sa)(\d)\s+(.+)", re.IGNORECASE)
INDEXED_BRACKET_PATTERN = re.compile(r"^(.+)\[(\d+)\]$")
INDEXED_SUFFIX_PATTERN = re.compile(r"^(.+/[A-Za-z_]+)(\d+)$")


def _parse_csv_tokens(raw: str) -> List[str]:
    """Split a comma-separated list into normalized tokens."""
    if not raw:
        return []
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


def _split_indexed_token(token: str) -> tuple[str, int, str] | None:
    """
    Parse indexed signals like:
      - dataa[12]
      - \DP_OP_15J1_122_8723/n73
    """
    m = INDEXED_BRACKET_PATTERN.match(token)
    if m:
        return m.group(1), int(m.group(2)), "bracket"
    m = INDEXED_SUFFIX_PATTERN.match(token)
    if m:
        return m.group(1), int(m.group(2)), "suffix"
    return None


def _format_int_ranges(values: List[int]) -> str:
    """Convert sorted integers to compact range syntax."""
    if not values:
        return ""
    ranges: List[str] = []
    start = values[0]
    prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append(f"{start}..{prev}" if start != prev else str(start))
        start = value
        prev = value
    ranges.append(f"{start}..{prev}" if start != prev else str(start))
    return ", ".join(ranges)


def _parse_binary_value(value: Any) -> int | None:
    """Normalize a value to binary int (0/1) when possible."""
    if value in (0, 1):
        return int(value)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        v = value.strip()
        if v == "0":
            return 0
        if v == "1":
            return 1
    return None


def compact_binary_assignment_dict(values: Dict[str, Any], min_compact_size: int = 24) -> str:
    """
    Compact binary dictionaries like {"y[0]": 0, "y[1]": 1, ...}.

    - Small dicts keep the explicit "k: v" format for readability.
    - Large binary vectors are compacted as range summaries:
      y[0..127] ones={14..15, 26..44, 63}
    """
    if not values:
        return ""

    if len(values) < min_compact_size:
        return ", ".join(f"{k}: {v}" for k, v in values.items())

    grouped: Dict[str, Dict[int, int]] = {}
    group_order: List[str] = []
    passthrough: List[str] = []

    for key, value in values.items():
        m = INDEXED_BRACKET_PATTERN.match(key)
        bit = _parse_binary_value(value)
        if not m or bit is None:
            passthrough.append(f"{key}: {value}")
            continue

        base = m.group(1)
        index = int(m.group(2))
        if base not in grouped:
            grouped[base] = {}
            group_order.append(base)
        grouped[base][index] = bit

    compacted: List[str] = []
    for base in group_order:
        index_to_bit = grouped[base]
        sorted_indices = sorted(index_to_bit.keys())
        if not sorted_indices:
            continue

        min_idx = sorted_indices[0]
        max_idx = sorted_indices[-1]
        contiguous = (
            len(sorted_indices) == (max_idx - min_idx + 1)
        )
        ones = [idx for idx in sorted_indices if index_to_bit[idx] == 1]
        one_ranges = _format_int_ranges(ones) if ones else "-"

        # For compact low-width buses, include exact bitstring (LSB to MSB).
        if contiguous and min_idx == 0 and len(sorted_indices) <= 16:
            bits_lsb0 = "".join(str(index_to_bit[i]) for i in range(max_idx + 1))
            compacted.append(
                f"{base}[0..{max_idx}] bits_lsb0={bits_lsb0}"
            )
            continue

        if contiguous:
            compacted.append(f"{base}[{min_idx}..{max_idx}] ones={{{one_ranges}}}")
        else:
            idx_ranges = _format_int_ranges(sorted_indices)
            compacted.append(f"{base}[{idx_ranges}] ones={{{one_ranges}}}")

    return ", ".join(compacted + passthrough)


def compact_signal_list(raw: str) -> str:
    """
    Compact comma-separated signal names by grouping indexed families.

    Example:
      dataa[0], dataa[1], dataa[3] -> dataa[0..1, 3]
      \\DP_OP_.../n7, \\DP_OP_.../n8 -> \\DP_OP_.../n[7..8]
    """
    tokens = _parse_csv_tokens(raw)
    return compact_signal_tokens(tokens)


def compact_signal_tokens(tokens: List[str]) -> str:
    """Compact a normalized list of tokens without csv parsing overhead."""
    if not tokens:
        return ""

    families: Dict[str, List[int]] = {}
    family_styles: Dict[str, str] = {}
    family_order: List[str] = []
    plain_tokens: List[str] = []

    for token in tokens:
        parsed = _split_indexed_token(token)
        if not parsed:
            if token not in plain_tokens:
                plain_tokens.append(token)
            continue
        family, index, style = parsed
        if family not in families:
            families[family] = []
            family_order.append(family)
            family_styles[family] = style
        families[family].append(index)

    compacted: List[str] = []
    for family in family_order:
        indices = sorted(set(families[family]))
        style = family_styles.get(family, "bracket")
        if style == "suffix" and len(indices) == 1:
            compacted.append(f"{family}{indices[0]}")
            continue
        range_text = _format_int_ranges(indices)
        compacted.append(f"{family}[{range_text}]")

    return ", ".join(compacted + plain_tokens)


def convert_netlist_to_json_payload(record: Dict[str, Any]) -> None:
    import hashlib
    netlist_content = record["netlist"]
    if isinstance(netlist_content, dict) and "doc_id" in netlist_content and "netlist" in netlist_content:
        return
    doc_id = hashlib.sha256(netlist_content.encode("utf-8")).hexdigest()[:16]
    json_payload = {"doc_id": doc_id, "netlist": netlist_content}
    record["netlist"] = json_payload

@dataclass
class ConversationExample:
    """Internal container for a single training example."""

    messages: List[Dict[str, str]]  # list of {"role": ..., "content": ...}
    # Per message: is it a training target? None means every assistant
    # message is. Set for retry trajectories, whose injected failed first
    # attempt is context the model must not learn to produce.
    trainable: Optional[List[bool]] = None

    @staticmethod
    def _prepare_record(record: Dict[str, Any]) -> Dict[str, Any]:
        """Derive placeholders and formatted system/user strings (mutates *record*)."""
        fault = record.get('fault', '')
        if fault:
            fault_match = FAULT_PATTERN.match(fault)
            if fault_match:
                fault_value = int(fault_match.group(2))
                fault_net = fault_match.group(3).strip()

                record['fault_net'] = fault_net
                record['fault_model_short'] = f"SA{fault_value}"
                record['fault_model_long'] = f"stuck-at-{fault_value}"
                record['excitation_value'] = str(1 - fault_value)

        expected_output_dict = json.loads(record.get('expected_output', '{}'))
        record['primary_observation_nets'] = compact_signal_tokens(
            list(expected_output_dict.keys())
        )

        input_vector_dict = json.loads(record.get('input_vector', '{}'))
        record['input_vector_json'] = ", ".join(
            f"{net}: {value}" for net, value in input_vector_dict.items()
        )
        record['expected_output_json'] = ", ".join(
            f"{net}: {value}" for net, value in expected_output_dict.items()
        )
        record['input_vector'] = compact_binary_assignment_dict(input_vector_dict)
        record['expected_output'] = compact_binary_assignment_dict(expected_output_dict)

        record['propagation_gates'] = compact_signal_list(
            record.get('fault_propagation_gates', '')
        )

        backtrack_tokens = _parse_csv_tokens(record.get('backtrack_nets', ''))
        input_keys = set(input_vector_dict.keys())
        controlling_tokens = [
            tok for tok in backtrack_tokens if tok in input_keys
        ]
        record['primary_controlling_nets'] = compact_signal_tokens(
            controlling_tokens
        )
        record['non_controlling_nets'] = compact_signal_tokens(
            backtrack_tokens
        )

        system_content = record.get("system_content", "")
        user_content = record.get("user_content", "")

        convert_netlist_to_json_payload(record)

        system_content = system_content.format(**record)
        user_content = user_content.format(**record)

        return {
            "system_content": system_content,
            "user_content": user_content,
            "input_vector_dict": input_vector_dict,
            "expected_output_dict": expected_output_dict,
            "fault": fault,
        }

    @staticmethod
    def prompt_messages_from_record(
        record: Dict[str, Any], use_tools: bool = False,
    ) -> List[Dict[str, str]]:
        """System + user messages only (for prompt-length checks without full SFT format)."""
        prep = ConversationExample._prepare_record(record)
        messages: List[Dict[str, str]] = []
        if prep["system_content"]:
            messages.append({"role": "system", "content": prep["system_content"]})
        if prep["user_content"]:
            messages.append({"role": "user", "content": prep["user_content"]})
        return messages

    @staticmethod
    def from_record(
        record: Dict[str, Any],
        use_tools: bool = False,
        *,
        failed_attempt: Any = None,
        interpret_results: bool = False,
        tool_view: str = "full",
    ) -> "ConversationExample":
        """
        Create a ConversationExample from a dataset record.  This method
        combines the system, user, reasoning and answer fields into a
        structured chat conversation.  It uses placeholders contained in
        the record to fill the reasoning template.

        Parameters
        ----------
        record : Dict[str, Any]
            A dictionary from the dataset with keys such as
            ``system_content``, ``user_content``, ``reasoning_content``,
            ``answer_content``, ``fault``, ``netlist``, ``input_vector``, etc.
        use_tools : bool
            If *True*, format the assistant turn with explicit tool-call
            JSON and a separate ``tool`` message containing the snapshot.
        failed_attempt : FailedAttempt, optional
            Requires ``use_tools``. Turn the conversation into a retry: the
            first tool call uses this non-detecting pattern and gets its real
            snapshot; the next assistant turn reads why it failed, calls the
            tool again with the record's pattern, and reads the success. The
            first assistant turn is marked not trainable (``trainable``).
            See :mod:`atpgllm.training.tool_trajectories`.
        interpret_results : bool
            Start the final answer by naming the primary outputs where the
            Good and Bad Machine differ, instead of a generic sentence.
            Implied by ``failed_attempt``. Both default off, so the default
            conversation is unchanged.
        tool_view : str
            What the tool messages show (``tool_feedback.TOOL_VIEWS``):
            ``full`` (default) is the simulator's whole table; ``po_fault``
            keeps the primary outputs and the fault site. Interpretations
            are read from the same view, so they cite only what is shown.

        Returns
        -------
        ConversationExample

        Template Placeholder Derivation
        -------------------------------
        The reasoning templates use the following placeholders that must be
        derived from the stored dataset fields:

        1.  module_name         <- direct from 'module_name'
        2.  fault_net           <- parsed from 'fault' (e.g. "sa0 net_name" → "net_name")
        3.  fault_model_long    <- parsed from 'fault' (e.g. "sa0" → "stuck-at-0")
        4.  fault_model_short   <- parsed from 'fault' (e.g. "sa0" → "SA0")
        5.  excitation_value    <- parsed from 'fault' (sa0 needs 1 to excite, sa1 needs 0)
        6.  propagation_gates   <- from 'fault_propagation_gates'
        7.  primary_observation_nets <- keys from 'expected_output' JSON
        8.  backtrack_gates     <- from 'backtrack_gates'
        9.  primary_controlling_nets <- from 'backtrack_nets'
        10. expected_output     <- formatted from 'expected_output' JSON
        11. input_vector        <- formatted from 'input_vector' JSON
        12. detected_faults     <- from 'detected_faults'
        13. non_controlling_nets <- from 'backtrack_nets' (sensitizing inputs)
        14. snapshot            <- from 'snapshot'
        """
        if failed_attempt is not None and not use_tools:
            raise ValueError("A retry trajectory needs use_tools=True.")
        if tool_view != "full" and not use_tools:
            raise ValueError("A tool view needs use_tools=True.")
        prep = ConversationExample._prepare_record(record)
        system_content = prep["system_content"]
        user_content = prep["user_content"]
        input_vector_dict = prep["input_vector_dict"]
        expected_output_dict = prep["expected_output_dict"]
        fault = prep["fault"]

        reasoning_content = record.get("reasoning_content", "")
        answer_content = record.get("answer_content", "")
        snapshot = record.get("snapshot", "")

        def shown(table: str) -> str:
            if tool_view == "full" or not table:
                return table
            from atpgllm.training.tool_feedback import shown_snapshot

            return shown_snapshot(table, fault, list(expected_output_dict), tool_view)

        # Lazy replacement of the placeholders
        answer_content = answer_content.replace('input_vector', 'input_vector_json')
        answer_content = answer_content.replace('expected_output', 'expected_output_json')
        answer_content = answer_content.format(**record)

        # Conditionally render the reasoning template.  Steps whose
        # optional placeholders (propagation_gates, backtrack_gates, etc.)
        # are empty are automatically replaced with generic explanations
        # instead of producing garbled text like "through the  to".
        reasoning_content = render_reasoning_template(reasoning_content, record)

        # Compose the messages sequence.  We include the chain-of-thought in a
        # separate assistant message tagged as "assistant" reasoning.  During
        # RL training we can evaluate the reasoning chain separately from the
        # final answer.
        messages: List[Dict[str, str]] = []
        trainable: List[bool] = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        if user_content:
            messages.append({"role": "user", "content": user_content})
        if reasoning_content:
            if use_tools:
                def tool_call(input_vector, output_vector):
                    arguments = dict.fromkeys(
                        FAULT_SIMULATION_TOOL['function']['parameters']['properties'].keys()
                    )
                    arguments['input_vector'] = input_vector
                    arguments['output_vector'] = output_vector
                    arguments['fault'] = fault
                    arguments['doc_id'] = record.get('netlist', 'netlist is unknown').get('doc_id', 'netlist is unknown')
                    return {
                        "name": FAULT_SIMULATION_TOOL['function']['name'],
                        "arguments": arguments,
                    }

                if failed_attempt is not None:
                    tool_call_json = tool_call(
                        failed_attempt.input_vector, failed_attempt.output_vector
                    )
                else:
                    tool_call_json = tool_call(input_vector_dict, expected_output_dict)

                # For SFT we include the reasoning verbatim.  At inference time
                # you may instruct the model to produce its chain of thought
                # using special tags (e.g. <think>...</think>) or tool calls.
                tool_call_content = (
                    "<think>" + reasoning_content + "</think>\n\n"
                    "I have to verify if the fault is detected by the input and "
                    "output vectors. I need to call the fault simulation tool.\n"
                )
                messages.append({
                    "role": "assistant",
                    "content": tool_call_content,
                    "tool_calls": [{"type": "function", "function": tool_call_json}],
                })
                if failed_attempt is not None:
                    from atpgllm.training.tool_feedback import (
                        read_snapshot,
                        revision_sentence,
                        undetected_sentence,
                    )

                    trainable = [False] * len(messages)
                    failed_table = shown(failed_attempt.snapshot)
                    reading = read_snapshot(
                        failed_table, fault, list(expected_output_dict)
                    )
                    messages.append({
                        "role": "tool",
                        "name": FAULT_SIMULATION_TOOL['function']['name'],
                        "content": failed_table,
                    })
                    messages.append({
                        "role": "assistant",
                        "content": (
                            undetected_sentence(reading) + " "
                            + revision_sentence(reading, failed_attempt.changes)
                            + "\n"
                        ),
                        "tool_calls": [{
                            "type": "function",
                            "function": tool_call(input_vector_dict, expected_output_dict),
                        }],
                    })
                    trainable += [False, True]
                messages.append({
                    "role": "tool",
                    "name": FAULT_SIMULATION_TOOL['function']['name'],
                    "content": shown(snapshot),
                })
            else:
                messages.append({
                    "role": "assistant",
                    "content": (
                        "<think>" + reasoning_content + "\n"
                        "I'll perform a fault simulation by myself to verify if "
                        "the fault is detected by the input and output vectors\n\n"
                        + snapshot
                        + "</think>\n\n"
                    ),
                })
        if answer_content:
            lead = (
                f"The fault {record['fault']} has been sensitized. I can see the "
                "difference between Good/Bad Machine. "
            )
            if (interpret_results or failed_attempt is not None) and snapshot:
                from atpgllm.training.tool_feedback import (
                    detected_sentence,
                    read_snapshot,
                )

                reading = read_snapshot(
                    shown(snapshot), fault, list(expected_output_dict)
                )
                if not reading.detected:
                    raise ValueError(
                        f"The record's snapshot does not detect {fault!r}."
                    )
                lead = detected_sentence(reading) + " "
            messages.append({
                "role": "assistant",
                "content": lead + "To sum up, the vectors are:\n" + answer_content,
            })
        if failed_attempt is not None:
            trainable += [m["role"] == "assistant" for m in messages[len(trainable):]]
        return ConversationExample(
            messages=messages,
            trainable=trainable if failed_attempt is not None else None,
        )
