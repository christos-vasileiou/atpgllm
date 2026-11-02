"""Check what a model's free text reasoning says against the netlist and the simulator.

Deterministic, no template and no LLM judge. Whatever can be checked is checked,
and a sentence with nothing checkable in it is ignored, neither rewarded nor
penalized, so the model can phrase its reasoning however it likes.

Scores for one attempt, each None when it has nothing to look at:

``claim_truth``
    ``net = v``, ``net to v``, ``net is v`` about internal nets, compared with the
    good machine under the vector the model then simulated. Claims about primary
    inputs are ignored, because restating your own vector is always true.
``implication_truth``
    A line naming a gate, its output value and at least one input value is judged
    by that cell's truth table, whatever the final vector does.
``structural_validity``
    Share of named nets and gates that exist and lie on the fault's cones.
``output_prediction``
    The tool call's ``output_vector`` against the simulated good values. It is
    written before the result is seen, so it cannot be copied from the tool.
``verdict_truth``
    The final claim of detection against what the simulator showed.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from itertools import product

TOKEN = re.compile(r"\\?[A-Za-z_][\w/$.*]*(?:\[\d+\])?")
VALUE = re.compile(r"\s*(?:=+|:|->|\bto\b|\bis\b|\bbeing\b|\bat\b|\bequals\b)\s*(?:logic\s+|value\s+)?([01])(?![\w\[]|\.\.)",
                   re.IGNORECASE)


def _compile(text):
    from atpgllm.training.eval_report import Simulator
    if not hasattr(_compile, "simulator"):
        _compile.simulator = Simulator()
    return _compile.simulator.compile(text)


@lru_cache(maxsize=512)
def structure(text: str):
    """What the checks need about one netlist."""
    net_list = _compile(text)
    gates, drivers, readers = {}, {}, {}
    for cell, instance, _port, out_net, inputs in net_list.gate_metadata:
        gates[instance] = (cell.split("_")[0], out_net, dict(inputs))
    for instruction in net_list.instructions:
        if instruction[0] == "gate":
            _, out_net, lut, syms, input_map, _deps = instruction
            drivers[out_net] = (lut, tuple(syms), dict(input_map))
            for net in input_map.values():
                readers.setdefault(net, set()).add(out_net)
    nets = set(net_list.input_nets) | set(net_list.output_nets) | set(net_list.net_dependencies)
    for _cell, _out, inputs in gates.values():
        nets |= set(inputs.values())
    return dict(inputs=set(net_list.input_nets), outputs=set(net_list.output_nets), nets=nets, gates=gates,
                drivers=drivers, readers=readers, deps={k: set(v) for k, v in net_list.net_dependencies.items()})


def cones(s, site):
    """Nets a correct argument about ``site`` could mention: fan-in, fanout, and the side inputs and fan-in of fanout gates."""
    def walk(start, step):
        seen, frontier = set(start), set(start)
        while frontier:
            frontier = {u for x in frontier for u in step(x)} - seen
            seen |= frontier
        return seen
    fanin = walk({site}, lambda n: s["deps"].get(n, ()))
    fanout = walk({site}, lambda n: s["readers"].get(n, ()))
    sides = {u for out in fanout if out != site for u in s["deps"].get(out, ())} - fanout
    return fanin | fanout | walk(sides, lambda n: s["deps"].get(n, ()))


def claims_in(text: str, nets: set) -> list:
    """``(net, value, position)`` for every statement that a known net holds a value."""
    found = []
    for match in TOKEN.finditer(text):
        name = match.group()
        if name in nets:
            value = VALUE.match(text, match.end())
            if value:
                found.append((name, value.group(1), match.start()))
    return found


def _evaluate(lut, args):
    """One cell output; a lookup table is indexed with the first input as the low bit."""
    if isinstance(lut, list):
        return lut[sum(value << position for position, value in enumerate(args))]
    return int(bool(lut(*args)))


def forced(driver, assignment: dict):
    """Set of output values the gate can produce given claimed input values."""
    lut, syms, input_map = driver
    choices = []
    for sym in syms:
        net = input_map.get(sym)
        if net is not None and str(net).isdigit():
            choices.append((int(net),))
        elif net in assignment:
            choices.append((int(assignment[net]),))
        else:
            choices.append((0, 1))
    return {_evaluate(lut, list(args)) for args in product(*choices)}


def _first_tool(slot):
    messages = slot.get("messages") or []
    turn = next((m for m in messages if m["role"] == "assistant"), None)
    tool = next((m for m in messages if m["role"] == "tool"), None)
    table = None
    if tool:
        try:
            table = json.loads(tool["content"])
        except (json.JSONDecodeError, TypeError):
            table = None
    call = None
    if turn and turn.get("tool_calls"):
        call = turn["tool_calls"][0]["function"].get("arguments")
    return turn, call, table


def _vector(raw):
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return dict(part.strip().rsplit(":", 1) for part in raw.split(",") if ":" in part)
    return {}


def audit_text(text: str, netlist: str, fault: str, good: dict | None, bad: dict | None = None) -> dict:
    """Claim and structure scores of reasoning ``text`` under the machine values ``good`` and ``bad``.

    A claim right after "faulty" or "bad" is about the faulty machine.
    """
    s = structure(netlist)
    site = fault.split(" ", 1)[1]
    out = {"reasoning_chars": len(text)}
    claims = claims_in(text, s["nets"])
    checkable = [claim for claim in claims if claim[0] not in s["inputs"]]
    out["claims"] = len(checkable)
    if good is not None and checkable:
        def holds(net, value, position):
            about_fault = re.search(r"(fault\w*|bad)\s*$", text[max(0, position - 12):position], re.IGNORECASE)
            return str((bad or {}).get(net) if about_fault else good.get(net)) == value

        hits = sum(holds(*claim) for claim in checkable)
        out["claim_truth"] = hits / len(checkable)
        # Bounded coverage: one true sentence should not outrank a real argument.
        out["claim_score"] = out["claim_truth"] * min(1.0, len(checkable) / 3)

    # Implications, line by line.
    judged = []
    for line in re.split(r"[\n;]|\.\s", text):
        named = [t for t in TOKEN.findall(line) if t in s["gates"]]
        if not named:
            continue
        values = {net: value for net, value, _ in claims_in(line, s["nets"])}
        for instance in named:
            _cell, out_net, inputs = s["gates"][instance]
            driver = s["drivers"].get(out_net)
            given = {net: values[net] for net in inputs.values() if net in values}
            if driver is None or out_net not in values or not given:
                continue
            possible = forced(driver, given)
            want = int(values[out_net])
            judged.append(1.0 if possible == {want} else 0.5 if want in possible else 0.0)
    if judged:
        out["implication_truth"] = sum(judged) / len(judged)
        out["implications"] = len(judged)

    # Names: do they exist, and do they lie where an argument about this fault can go?
    gate_like = re.compile(r"^U\d+$|^n\d+$")
    named = [t for t in TOKEN.findall(text) if t in s["nets"] or t in s["gates"] or gate_like.match(t)]
    if named:
        relevant_nets = cones(s, site)
        relevant_gates = {g for g, (_c, out_net, _i) in s["gates"].items() if out_net in relevant_nets}
        known = [t for t in named if t in s["nets"] or t in s["gates"]]
        on_cone = [t for t in known if t in relevant_nets or t in relevant_gates]
        out["names_exist"] = len(known) / len(named)
        out["structural_validity"] = len(on_cone) / len(named)
        out["cone_share_of_circuit"] = len(relevant_nets) / max(len(s["nets"]), 1)
    return out


def reasoning_window(text: str, call, s) -> dict:
    """Where in the reasoning the model commits to input values, and whether the call copies them.

    An assignment stated early, before any internal net has been reasoned about,
    is not the result of reasoning: the rest of the window only leads up to a
    vector already chosen, which the tool call then copies.
    """
    out = {}
    length = max(len(text), 1)
    inputs = [(net, value, position) for net, value, position in claims_in(text, s["inputs"])]
    internal = [position for net, _value, position in claims_in(text, s["nets"]) if net not in s["inputs"]]
    if not inputs:
        out["inputs_stated_in_reasoning"] = 0.0
        return out
    out["inputs_stated_in_reasoning"] = 1.0
    out["first_input_value_position"] = inputs[0][2] / length
    out["inputs_before_internal_nets"] = float(not internal or inputs[0][2] < internal[0])
    if call:
        care = {net: str(value) for net, value in _vector(call.get("input_vector")).items() if str(value) in ("0", "1")}
        if care:
            stated = {}
            for net, value, _position in inputs:
                stated[net] = value
            out["call_bits_stated_in_reasoning"] = sum(net in stated for net in care) / len(care)
            agreed = [net for net in care if net in stated]
            if agreed:
                out["call_bits_copy_reasoning"] = sum(stated[net] == care[net] for net in agreed) / len(agreed)
    return out


def audit_slot(slot: dict, ref: dict) -> dict:
    """All log only scores of one attempt; ``ref`` holds the problem's netlist and fault."""
    netlist, fault = ref.get("netlist"), ref.get("fault")
    if not netlist:
        return {}
    turn, call, table = _first_tool(slot)
    if turn is None:
        return {}
    good = table.get("Good Machine") if isinstance(table, dict) else None
    bad = table.get("Bad Machine") if isinstance(table, dict) else None
    text = turn.get("content") or ""
    out = audit_text(text, netlist, fault, good, bad)

    site, stuck = fault.split(" ", 1)[1], fault[2]
    required = "1" if stuck == "0" else "0"
    s = structure(netlist)
    out.update(reasoning_window(text, call, s))
    stated = [value for net, value, _ in claims_in(text, {site})]
    if stated:
        out["site_target_correct"] = float(stated[0] == required)
        if good is not None and stated[0] == required:
            out["site_target_realized"] = float(str(good.get(site)) == required)

    if good is not None and call:
        predicted = _vector(call.get("output_vector"))
        definite = [o for o in s["outputs"] if str(good.get(o)) in ("0", "1")]
        stated_outputs = [o for o in definite if str(predicted.get(o)) in ("0", "1")]
        if definite:
            coverage = len(stated_outputs) / len(definite)
            agree = (sum(str(predicted[o]) == str(good[o]) for o in stated_outputs) / len(stated_outputs)
                     if stated_outputs else 0.0)
            out["output_prediction_accuracy"] = agree
            out["output_prediction"] = agree * coverage

    final = slot.get("final_answer")
    if final and good is not None and bad is not None:
        detected = any(str(good.get(o)) in ("0", "1") and str(bad.get(o)) in ("0", "1") and str(good.get(o)) != str(bad.get(o))
                       for o in s["outputs"])
        m = re.search(r'DETECTED_FAULTS:\s*"([^"]*)"', final)
        claimed = bool(m and fault in m.group(1))
        out["verdict_truth"] = float(claimed == detected)
    return out
