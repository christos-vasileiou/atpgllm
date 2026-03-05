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
from typing import List, Dict, Any

import regex as re

from template_rendering import render_reasoning_template
from tools import FAULT_SIMULATION_TOOL

def convert_netlist_to_json_payload(record: Dict[str, Any]) -> None:
    import hashlib
    netlist_content = record["netlist"]
    doc_id = hashlib.sha256(netlist_content.encode("utf-8")).hexdigest()[:16]
    json_payload = {"doc_id": doc_id, "netlist": netlist_content}
    record["netlist"] = json_payload

@dataclass
class ConversationExample:
    """Internal container for a single training example."""

    messages: List[Dict[str, str]]  # list of {"role": ..., "content": ...}

    @staticmethod
    def from_record(record: Dict[str, Any], use_tools: bool = False) -> "ConversationExample":
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

        # =================================================================
        # Parse fault string to derive fault-related placeholders
        # Fault format: "sa0 net_name" or "sa1 net_name"
        # =================================================================
        fault = record.get('fault', '')
        if fault:
            fault_match = re.match(r'(sa)(\d)\s+(.+)', fault, re.IGNORECASE)
            if fault_match:
                fault_value = int(fault_match.group(2))  # 0 or 1
                fault_net = fault_match.group(3).strip()  # net name

                record['fault_net'] = fault_net
                record['fault_model_short'] = f"SA{fault_value}"  # "SA0" or "SA1"
                record['fault_model_long'] = f"stuck-at-{fault_value}"  # "stuck-at-0" or "stuck-at-1"
                # To excite a SA0 fault, drive the net to 1 (opposite of stuck value)
                # To excite a SA1 fault, drive the net to 0 (opposite of stuck value)
                record['excitation_value'] = str(1 - fault_value)

        # =================================================================
        # Parse JSON fields and derive additional placeholders
        # =================================================================
        # Parse expected_output to get primary_observation_nets (output net names)
        expected_output_dict = json.loads(record.get('expected_output', '{}'))
        record['primary_observation_nets'] = ', '.join(expected_output_dict.keys())

        # Format vectors as "net: value, net: value, ..."
        input_vector_dict = json.loads(record.get('input_vector', '{}'))
        record['input_vector'] = ", ".join(f"{net}: {value}" for net, value in input_vector_dict.items())
        record['expected_output'] = ", ".join(f"{net}: {value}" for net, value in expected_output_dict.items())

        # =================================================================
        # Map stored field names to template placeholder names
        # =================================================================
        # propagation_gates: gates whose outputs are on the fault propagation path
        record['propagation_gates'] = record.get('fault_propagation_gates', '')

        # primary_controlling_nets & non_controlling_nets: sensitizing inputs
        # Both map to backtrack_nets (the inputs that control fault propagation)
        record['primary_controlling_nets'] = ', '.join(
            set(input_vector_dict.keys()) & set(record.get('backtrack_nets', '').split(', '))
        )
        record['non_controlling_nets'] = record.get('backtrack_nets', '')

        # Extract the raw fields
        system_content = record.get("system_content", "")
        user_content = record.get("user_content", "")
        reasoning_content = record.get("reasoning_content", "")
        answer_content = record.get("answer_content", "")
        snapshot = record.get("snapshot", "")

        # Convert the netlist to a json payload including the doc_id and the netlist content
        convert_netlist_to_json_payload(record)

        # Format the system, user and answer content with the record
        system_content = system_content.format(**record)
        user_content = user_content.format(**record)
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
        if system_content:
            messages.append({"role": "system", "content": system_content})
        if user_content:
            messages.append({"role": "user", "content": user_content})
        if reasoning_content:
            if use_tools:
                arguments = dict.fromkeys(
                    FAULT_SIMULATION_TOOL['function']['parameters']['properties'].keys()
                )
                arguments['input_vector'] = input_vector_dict
                arguments['output_vector'] = expected_output_dict
                arguments['fault'] = fault
                arguments['doc_id'] = record.get('netlist', 'netlist is unknown').get('doc_id', 'netlist is unknown')

                tool_call_json = {
                    "name": FAULT_SIMULATION_TOOL['function']['name'],
                    "arguments": arguments,
                }

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
                messages.append({
                    "role": "tool",
                    "name": FAULT_SIMULATION_TOOL['function']['name'],
                    "content": snapshot,
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
            messages.append({
                "role": "assistant",
                "content": (
                    "I can see the difference between Good/Bad Machine. "
                    "The fault has been sensitized. To sum up, the vectors are:\n"
                    + answer_content
                ),
            })
        return ConversationExample(messages=messages)
