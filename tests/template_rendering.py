"""
template_rendering.py
=====================

Conditional template rendering for reasoning steps.

The reasoning templates use numbered steps like::

    3. **Fault Propagation**: ... through {propagation_gates} to ...

Some placeholders (``propagation_gates``, ``backtrack_gates``, etc.) are
frequently empty.  Rather than format-then-regex-clean, we inspect each
step's placeholders *before* rendering: if any optional field is empty,
the entire step is replaced with a generic explanation.

This is analogous to a Jinja2 ``{% if field %}...{% endif %}`` guard
around each step, but implemented via lightweight introspection of the
Python format-string placeholders.
"""

from __future__ import annotations

from typing import List

import regex as re


# =====================================================================
# Constants
# =====================================================================

# Placeholder regex: matches "{name}" but not "{{escaped}}"
_PLACEHOLDER_RE = re.compile(r'\{(\w+)\}')

# Fields that may legitimately be empty in the dataset.  When a step
# references one of these and its value is blank, the step is swapped
# for a generic domain explanation instead of producing garbled text.
_OPTIONAL_FIELDS = frozenset({
    'propagation_gates',
    'backtrack_gates',
    'primary_controlling_nets',
    'non_controlling_nets',
})

# Generic replacement sentences keyed by lowercase substrings found in
# the step title.  Order matters – first match wins.
_GENERIC_REPLACEMENTS: List[tuple] = [
    # Excitation / setup / activation / input justification / value assignment
    (lambda t: any(k in t for k in ('excitation', 'setup', 'activation',
                                     'input justification', 'value assignment',
                                     'logic justification')),
     "The {fault_model_long} fault on {fault_net} is excited by driving the "
     "net to {excitation_value} through the combined primary input assignments."),

    # Propagation / sensitiz / path selection / d-frontier drive/maintenance
    (lambda t: any(k in t for k in ('propagation', 'sensitiz', 'path selection',
                                     'd-frontier drive', 'forward propagation',
                                     'd-frontier maintenance', 'gate-level',
                                     'gate constraint')),
     "The fault effect at {fault_net} propagates directly to the primary "
     "outputs ({primary_observation_nets}) without requiring intermediate "
     "gate-level path sensitization."),

    # Backtrack / side-input / justification of / recursive / conditioning
    (lambda t: any(k in t for k in ('backtrack', 'side-input', 'justification of',
                                     'recursive', 'conditioning')),
     "No intermediate gate-level backtracking or side-input conditioning "
     "is needed — the fault path from {fault_net} to the outputs is direct."),

    # Implication / conflict / consistency / resolution
    (lambda t: any(k in t for k in ('implication', 'conflict', 'consistency',
                                     'resolution')),
     "The primary input assignments are logically consistent with no "
     "conflicts along the fault propagation path."),
]

# Regex to split a template into numbered steps.
# Each step starts with "N. **Title**" on a new line.
_STEP_SPLIT_RE = re.compile(r'(?=\d+\.\s+\*\*)')
# Regex to extract the step number and bold title.
_STEP_HEADER_RE = re.compile(r'^(\d+)\.\s+\*\*([^*]+)\*\*')


# =====================================================================
# Public API
# =====================================================================

def render_reasoning_template(template_str: str, record: dict) -> str:
    """Conditionally render a reasoning template, replacing steps that
    reference empty optional placeholders with generic explanations.

    How it works (Jinja-style logic without Jinja):

    1. Split the raw template into numbered steps.
    2. For each step, extract every ``{placeholder}`` it references.
    3. If *all* referenced optional fields have non-empty values
       → format the step normally with ``str.format(**record)``.
    4. If *any* referenced optional field is empty
       → replace the step with a domain-appropriate generic sentence
         (looked up by the step's bold title).

    This avoids garbled output ("through the  to", "gates . For")
    without brittle regex post-processing.
    """
    steps = _STEP_SPLIT_RE.split(template_str.strip())
    rendered: List[str] = []

    for step in steps:
        if not step.strip():
            continue

        # Which {placeholder} names does this step reference?
        referenced = set(_PLACEHOLDER_RE.findall(step))

        # Are any of the *optional* fields referenced AND empty?
        has_empty_optional = any(
            f in _OPTIONAL_FIELDS and not str(record.get(f, '')).strip()
            for f in referenced
        )

        if not has_empty_optional:
            # All values present — render the step normally.
            try:
                rendered.append(step.rstrip().format(**record))
            except KeyError:
                rendered.append(step.rstrip())
            continue

        # At least one optional placeholder is empty.
        # → Swap the whole step for a generic explanation.
        header = _STEP_HEADER_RE.match(step)
        if not header:
            # Can't parse the header — keep as-is rather than drop.
            try:
                rendered.append(step.rstrip().format(**record))
            except KeyError:
                rendered.append(step.rstrip())
            continue

        step_num = header.group(1)
        step_title = header.group(2).strip().rstrip(':')
        title_lower = step_title.lower()

        # Look up a generic replacement by title keywords.
        body = None
        for predicate, tmpl in _GENERIC_REPLACEMENTS:
            if predicate(title_lower):
                body = tmpl
                break

        if body is None:
            body = (
                "This step is handled implicitly for the "
                "{fault_model_long} fault on {fault_net} in module "
                "{module_name}, as the fault path is direct."
            )

        try:
            body = body.format(**record)
        except KeyError:
            pass  # leave the raw template — still better than garbled text

        rendered.append(f"{step_num}. **{step_title}**: {body}")

    return '\n'.join(rendered)
