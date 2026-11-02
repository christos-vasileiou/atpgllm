"""Gate by gate reasoning for a dataset row, written from its simulated values.

The dataset's reasoning lists net and gate names but no values, so it never does
the computation it describes. This walks back from the fault site to primary
inputs and forward along the path to an output, stating each gate's inputs and
output. Every statement is true of the row's vector, and the phrasing varies so
that a demonstration does not become a format the model has to copy.
"""
from __future__ import annotations

import json
import random
from typing import Optional

from atpgllm.training.reasoning_audit import structure

JUSTIFY = (
    "{cell} {gate} drives {out}: with {inputs}, {out} = {value}.",
    "For {out} = {value}, look at {gate} ({cell}). Setting {inputs} gives it.",
    "{gate} is a {cell}. {inputs}, so {out} = {value}.",
)
PROPAGATE = (
    "The effect reaches {gate} ({cell}) on {net}. With {sides}, {out} = {good} in the good machine and {bad} in the faulty one.",
    "At {cell} {gate}, {net} carries the difference. {sides} lets it through: {out} = {good} good, {bad} faulty.",
    "{gate} ({cell}) passes it on, since {sides}. Good {out} = {good}, faulty {out} = {bad}.",
)


ACTIVATING = (
    "So the activating inputs are {inputs}: flipping any one of them takes {site} off {value}.",
    "{inputs} are what set {site} = {value}. These bits decide activation.",
    "Activation rests on {inputs}, since changing one of them loses {site} = {value}.",
)
PROPAGATING = (
    "The propagating inputs are {inputs}: with the site still excited, flipping one of them stops the effect.",
    "{inputs} keep the path open. These bits decide propagation.",
    "Propagation rests on {inputs}, which hold the side inputs at values that let the effect through.",
)
DONT_CARE = (
    "The other {count} inputs can stay x, since the result does not depend on them.",
    "Leave the remaining {count} inputs at x. Neither the activation nor the path needs them.",
    "That leaves {count} inputs that do not matter here, so they stay x.",
)


def _bit(value) -> Optional[str]:
    return str(value) if str(value) in ("0", "1") else None


def _is_x_free(net, table):
    return _bit(table.get(net)) is not None


def render_trace(row: dict, *, seed: int = 0, max_lines: int = 32, split: Optional[dict] = None) -> Optional[str]:
    """The reasoning text for ``row``, or None when its snapshot does not show a detection.

    The derivation comes first and input values are named as its result: a
    vector written early in the reasoning would make the rest decoration and the
    tool call a copy. ``split`` is ``Simulator.decisive_split`` of the row's
    label, which names the activating and propagating inputs. Without it those
    two conclusions are left out.
    """
    snapshot = row["snapshot"]
    snapshot = json.loads(snapshot) if isinstance(snapshot, str) else snapshot
    good, bad = snapshot.get("Good Machine") or {}, snapshot.get("Bad Machine") or {}
    fault, netlist = row["fault"], row["netlist"]
    s = structure(netlist)
    stuck, site = fault[2], fault.split(" ", 1)[1]
    required = "1" if stuck == "0" else "0"
    if str(good.get(site)) != required:
        return None
    gate_of = {out: (gate, cell, inputs) for gate, (cell, out, inputs) in s["gates"].items()}
    rng = random.Random(seed)
    pick = lambda options: options[rng.randrange(len(options))]
    lines = [f"A stuck-at-{stuck} on {site} shows only when {site} = {required}, so {site} has to be driven to {required}."]

    # Justification: breadth first from the site, only through definite values.
    queue, seen = [site], {site}
    budget = max(4, max_lines // 2)
    while queue and budget > 0:
        net = queue.pop(0)
        driver = gate_of.get(net)
        if driver is None or net in s["inputs"]:
            continue
        gate, cell, inputs = driver
        known = {n: _bit(good.get(n)) for n in inputs.values() if _bit(good.get(n)) is not None}
        if not known:
            continue
        stated = ", ".join(f"{n} = {v}" for n, v in known.items())
        lines.append(pick(JUSTIFY).format(cell=cell, gate=gate, out=net, inputs=stated, value=good.get(net)))
        budget -= 1
        for n in known:
            if n not in seen:
                seen.add(n)
                queue.append(n)

    def assigned(nets):
        return ", ".join(f"{n} = {good[n]}" for n in nets if _bit(good.get(n)))

    if split and split.get("activating"):
        lines.append(pick(ACTIVATING).format(inputs=assigned(split["activating"]), site=site, value=required))

    # Propagation: follow nets whose good and faulty values differ, toward an output.
    d_nets = {n for n in good if _bit(good.get(n)) and _bit(bad.get(n)) and good[n] != bad[n]}
    frontier, visited = [site], {site}
    budget = max(4, max_lines // 2)
    reached = None
    while frontier and budget > 0 and reached is None:
        net = frontier.pop(0)
        if net in s["outputs"] and net in d_nets:
            reached = net
            break
        for gate, (cell, out, inputs) in s["gates"].items():
            if net in inputs.values() and out in d_nets and out not in visited:
                sides = {n: _bit(good.get(n)) for n in inputs.values() if n != net and _bit(good.get(n)) is not None}
                stated = ", ".join(f"{n} = {v}" for n, v in sides.items()) or "no other input"
                lines.append(pick(PROPAGATE).format(gate=gate, cell=cell, net=net, sides=stated, out=out,
                                                    good=good[out], bad=bad[out]))
                visited.add(out)
                frontier.append(out)
                budget -= 1
                if budget == 0:
                    break
    outputs = [o for o in s["outputs"] if o in d_nets]
    if outputs:
        lines.append(f"{outputs[0]} differs between the two machines ({good[outputs[0]]} against {bad[outputs[0]]}), so this should detect the fault.")
    if split and split.get("propagating"):
        lines.append(pick(PROPAGATING).format(inputs=assigned(split["propagating"])))
    free = [n for n in s["inputs"] if _bit(good.get(n)) is None]
    if free:
        lines.append(pick(DONT_CARE).format(count=len(free)))
    return "\n".join(lines)


def read_result(table: dict, fault: str, netlist: str) -> str:
    """What a simulator result that does not detect the fault says, from its own values."""
    good, bad = table.get("Good Machine") or {}, table.get("Bad Machine") or {}
    s = structure(netlist)
    stuck, site = fault[2], fault.split(" ", 1)[1]
    required = "1" if stuck == "0" else "0"
    value = _bit(good.get(site))
    if value is None:
        return f"The simulator shows {site} = x in the good machine, so that vector left the site undetermined and the fault was not excited."
    if value != required:
        return f"The simulator shows {site} = {value} in the good machine, the same as the stuck value, so the fault was not excited."
    d_nets = {n for n in good if _bit(good.get(n)) and _bit(bad.get(n)) and good[n] != bad[n]}
    for gate, (cell, out, inputs) in sorted(s["gates"].items()):
        carried = [n for n in inputs.values() if n in d_nets]
        if carried and out not in d_nets:
            sides = {n: good.get(n) for n in inputs.values() if n not in d_nets}
            stated = ", ".join(f"{n} = {v}" for n, v in sides.items()) or "no other input"
            return (f"The fault is excited, {site} = {required}, and the effect reaches {carried[0]}, but {gate} ({cell}) stops it "
                    f"with {stated}. No output differs, so that vector does not detect the fault.")
    return f"The fault is excited, {site} = {required}, but no output differs between the two machines, so that vector does not detect the fault."
