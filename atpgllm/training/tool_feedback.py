"""Read a fault-simulation tool result and say what it means.

The tool returns ``{"Good Machine": {net: value}, "Bad Machine": {net: value}}``
for every net. A pattern detects the fault when the two machines differ at a
primary output. When they do not, the snapshot also shows why: either the
fault net already carries the stuck-at value in the Good Machine (the fault is
never activated), or the machines differ only at internal nets (the fault is
activated but blocked before every output).

Everything here uses only what the model sees at inference time: the tool's
snapshot, the fault and the circuit's primary outputs. SFT trajectories built
from these sentences therefore teach a reading the model can reproduce.

What the model sees is set by the tool view (:data:`TOOL_VIEWS`): ``full`` is
the simulator's whole table; ``po_fault`` keeps only the primary outputs and
the fault site, the site first and the outputs in declared order (median 511
instead of 1,929 tokens on v3 train). The inputs are left out because they repeat the call's own vector;
the fault site stays so activation remains readable. Rewards run their own
full simulation, so they do not depend on the view.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from atpgllm.training.conversation import FAULT_PATTERN, compact_signal_tokens


TOOL_VIEWS = ("full", "po_fault")


def shown_snapshot(
    snapshot: str,
    fault: str,
    output_nets: Sequence[str],
    view: str,
) -> str:
    """The part of the simulator's table the tool returns under ``view``."""
    if view == "full":
        return snapshot
    if view != "po_fault":
        raise ValueError(f"Unknown tool view {view!r}; expected one of {TOOL_VIEWS}")
    good, bad = _machines(snapshot)
    _, net = parse_fault(fault)
    # A fixed order, whatever order the simulator listed its table in (the
    # stored v4 snapshots and live TetraMAX frames differ): the fault site
    # unless it is an output, then the outputs in the order given (declared).
    order = ([] if net in output_nets else [net]) + list(output_nets)
    names = [name for name in order if name in good]
    return json.dumps(
        {
            "Good Machine": {name: good[name] for name in names},
            "Bad Machine": {name: bad.get(name) for name in names},
        },
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class SimulationReading:
    """What one fault-simulation snapshot shows."""

    fault: str
    fault_net: str
    stuck_at: int
    fault_net_good: Optional[str]
    detected: bool
    activated: bool
    differing_outputs: List[str]
    differing_internal: List[str]
    # Primary outputs that are x in either machine (three-valued simulation).
    unknown_outputs: Tuple[str, ...] = ()


def _machines(snapshot: str | Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    data = json.loads(snapshot) if isinstance(snapshot, str) else snapshot
    return dict(data["Good Machine"]), dict(data["Bad Machine"])


def _bit(value: Any) -> Optional[str]:
    """``"0"`` or ``"1"`` for a known value, None for x or a missing net."""
    text = str(value).strip()
    return text if text in ("0", "1") else None


def _flag(name: str) -> bool:
    import os

    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def read_outputs_enabled() -> bool:
    """``SFT_READ_OUTPUTS``: a reading lists every primary output's Good/Bad values first.

    The retry SFT of 2026-11-23 left the detect-or-miss choice to one token
    after the tool answer, and the model never learned to make it from the
    answer (teacher-forced AUC 0.63, juno 530313). Copying each output's two
    values first turns the verdict into a local comparison of what was just
    written.
    """
    return _flag("SFT_READ_OUTPUTS")


def output_pairs(snapshot: str | Mapping[str, Any], output_nets: Sequence[str]) -> str:
    """``Primary outputs, Good/Bad: y[0] 0/0, y[1] 1/0.`` in the given output order (x unknown)."""
    good, bad = _machines(snapshot)

    def shown(value: Any) -> str:
        bit = _bit(value)
        return bit if bit is not None else "x"

    pairs = [f"{net} {shown(good.get(net))}/{shown(bad.get(net))}" for net in output_nets]
    return "Primary outputs, Good/Bad: " + ", ".join(pairs) + "."


def with_outputs(sentence: str, snapshot: str | Mapping[str, Any] | None,
                 output_nets: Sequence[str], enabled: Optional[bool] = None) -> str:
    """``sentence`` after the output pairs when ``SFT_READ_OUTPUTS`` is on, else unchanged."""
    if enabled is None:
        enabled = read_outputs_enabled()
    if not enabled or snapshot is None or not output_nets:
        return sentence
    return output_pairs(snapshot, output_nets) + " " + sentence


def parse_fault(fault: str) -> Tuple[int, str]:
    match = FAULT_PATTERN.match(fault.strip())
    if not match:
        raise ValueError(f"Unsupported fault {fault!r}")
    return int(match.group(2)), match.group(3).strip()


def read_snapshot(
    snapshot: str | Mapping[str, Any],
    fault: str,
    output_nets: Sequence[str],
) -> SimulationReading:
    """Detection, activation and where the fault effect is visible."""
    good, bad = _machines(snapshot)
    stuck_at, net = parse_fault(fault)
    outputs = set(output_nets)
    # Three-valued: only a 0 against a 1 is a difference. A net that is x in
    # either machine is unknown, as TetraMAX and the reward read it.
    differing = [
        n for n in good
        if _bit(good[n]) is not None and _bit(bad.get(n)) is not None
        and _bit(good[n]) != _bit(bad.get(n))
    ]
    differing_outputs = [n for n in differing if n in outputs]
    fault_net_good = None if good.get(net) is None else str(good[net])
    return SimulationReading(
        fault=fault.strip(),
        fault_net=net,
        stuck_at=stuck_at,
        fault_net_good=fault_net_good,
        detected=bool(differing_outputs),
        activated=bool(differing),
        differing_outputs=differing_outputs,
        differing_internal=[n for n in differing if n not in outputs],
        unknown_outputs=tuple(
            n for n in good
            if n in outputs and (_bit(good[n]) is None or _bit(bad.get(n)) is None)
        ),
    )


def _nets(names: Sequence[str], limit: int = 8) -> str:
    text = compact_signal_tokens(list(names))
    parts = text.split(", ")
    if len(parts) > limit:
        text = ", ".join(parts[:limit]) + f" and {len(parts) - limit} more"
    return text


def detected_sentence(reading: SimulationReading) -> str:
    """Interpretation of a snapshot that detects the fault."""
    count = len(reading.differing_outputs)
    outputs = "output" if count == 1 else "outputs"
    return (
        f"The Good and Bad Machine differ at {count} primary {outputs}: "
        f"{_nets(reading.differing_outputs)}. The pattern detects {reading.fault}."
    )


def undetected_sentence(reading: SimulationReading) -> str:
    """Interpretation of a snapshot that does not detect the fault."""
    head = (
        "The Good and Bad Machine agree on every primary output, so this "
        f"pattern does not detect {reading.fault}."
    )
    if reading.unknown_outputs:
        head = (
            "No primary output has a 0 in one machine and a 1 in the other "
            f"({_nets(list(reading.unknown_outputs))} unknown), so this pattern "
            f"does not detect {reading.fault}."
        )
    excitation = 1 - reading.stuck_at
    if not reading.activated:
        value = reading.fault_net_good if reading.fault_net_good is not None else "unknown"
        if _bit(value) is None:
            return (
                f"{head} In the Good Machine {reading.fault_net} is {value}, so "
                f"the fault is not activated: {reading.fault_net} must be {excitation}."
            )
        return (
            f"{head} In the Good Machine {reading.fault_net} is {value}, the "
            f"value the stuck-at-{reading.stuck_at} fault forces, so the fault "
            f"is never activated: {reading.fault_net} must be {excitation}."
        )
    beyond_site = [n for n in reading.differing_internal if n != reading.fault_net]
    reaches = (
        f" and the difference reaches {_nets(beyond_site)}," if beyond_site else ","
    )
    return (
        f"{head} The fault is activated ({reading.fault_net} is {excitation} in "
        f"the Good Machine and {reading.stuck_at} in the Bad Machine){reaches} "
        "but it is blocked before every primary output."
    )


def blocked_detail(blockers: Sequence[Tuple[str, str, str, Optional[str]]], limit: int = 2) -> str:
    """Where the difference stops: gates that read it and the side inputs that block it.

    ``blockers`` holds ``(gate output, D input, side input, side value)``, as
    :func:`atpgllm.training.retry_repair.blocking_side_inputs` returns them.
    """
    by_gate: Dict[str, Tuple[str, List[str]]] = {}
    for out_net, d_input, side, value in blockers:
        shown = "unknown" if value is None else value
        entry = by_gate.setdefault(out_net, (d_input, []))
        text = f"{side} = {shown}"
        if text not in entry[1]:
            entry[1].append(text)
    if not by_gate:
        return ""
    parts = []
    for out_net, (d_input, sides) in list(by_gate.items())[:limit]:
        parts.append(
            f"the gate driving {out_net} reads the difference on {d_input}, "
            f"but its side {'input' if len(sides) == 1 else 'inputs'} {' and '.join(sides)} "
            f"{'keeps' if len(sides) == 1 else 'keep'} {out_net} equal in both machines"
        )
    more = len(by_gate) - limit
    tail = f" ({more} more gates are blocked the same way)" if more > 0 else ""
    return "The difference stops there: " + ", and ".join(parts) + tail + "."


def repair_sentence(
    reading: SimulationReading,
    changes: Sequence[Tuple[str, Any]],
    unblocked: Sequence[Tuple[str, Any]] = (),
) -> str:
    """The next call for an undetected pattern: which inputs change, and why.

    ``unblocked`` lists side inputs (net, new value) the change sets, so a
    blocked reading can say which input it fixes.
    """
    assignments = ", ".join(f"{net} to {value}" for net, value in changes)
    if not reading.activated:
        why = f"to drive {reading.fault_net} to {1 - reading.stuck_at}"
    elif unblocked:
        fixed = ", ".join(f"{net} to {value}" for net, value in unblocked)
        why = f"to bring {fixed} so the fault effect can pass toward a primary output"
    else:
        why = "so the fault effect can pass toward a primary output"
    return (
        f"I'll set {assignments} {why}, keep the other inputs, and simulate again."
    )


def revision_sentence(
    reading: SimulationReading,
    changes: Sequence[Tuple[str, Any]],
) -> str:
    """The fix for an undetected pattern: which inputs change, and why."""
    assignments = ", ".join(f"{net} to {value}" for net, value in changes)
    if not reading.activated:
        why = f"to drive {reading.fault_net} to {1 - reading.stuck_at}"
    else:
        why = "so the fault effect can pass through to a primary output"
    return (
        f"I'll set {assignments} {why}, keep the other inputs, and simulate again."
    )
