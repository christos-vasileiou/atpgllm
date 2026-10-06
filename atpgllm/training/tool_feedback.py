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
the fault site, in the table's order (median 511 instead of 1,929 tokens on v3
train). The inputs are left out because they repeat the call's own vector;
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
    keep = set(output_nets) | {net}
    names = [name for name in good if name in keep]
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


def _machines(snapshot: str | Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    data = json.loads(snapshot) if isinstance(snapshot, str) else snapshot
    return dict(data["Good Machine"]), dict(data["Bad Machine"])


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
    differing = [n for n in good if str(good[n]) != str(bad.get(n))]
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
    excitation = 1 - reading.stuck_at
    if not reading.activated:
        value = reading.fault_net_good if reading.fault_net_good is not None else "unknown"
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
