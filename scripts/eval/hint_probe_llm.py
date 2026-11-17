"""Does a stronger language model use the simulator's feedback better?

A probe without any training. A served model (OpenAI-compatible endpoint,
for example vLLM with gpt-oss-120b) gets the rows of ``hint_arms_eval``:
the fault, the circuit and the graph model's rich hint, and may call the
fault simulator up to ``--max-tool-calls`` times before it answers. The
rows it writes have the layout of ``hint_arms_eval``'s rollouts, so the
same report scores them next to the trained arms and the graph decoder.

Two views of the circuit:

- ``ports``: the port declarations and the hint, which is what the hint
  arms of Granite see. The model can only work from the hint and from what
  the simulator reports.
- ``netlist``: the full netlist and the hint, so the model can reason over
  the gates.

The model replies in plain text, one line ``CALL: {...}`` or ``ANSWER:
{...}`` after its reasoning, so no tool-call parser of the server is needed.

    PYTHONPATH=$PWD python scripts/eval/hint_probe_llm.py --out runs/hint_arms_eval_probe \\
        --name gptoss_ports --view ports --hints runs/bridges_retest_seq/hints \\
        --base-url http://localhost:8000/v1 --model gpt-oss-120b
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import hint_arms_eval as ev  # noqa: E402  (also puts scripts/train on the path)

r6 = ev.r6

SYSTEM = """You write test patterns for stuck-at faults in digital circuits (ATPG).

For one fault you get the circuit and a suggestion from a graph model. The suggestion is often right and sometimes wrong. A fault simulator is available. A pattern detects the fault when the fault changes the value of at least one output.

How to reply. Think first if you need to, then end your reply with exactly one line of one of these two forms:
CALL: {{"input name": 0 or 1, ...}}
ANSWER: {{"input name": 0 or 1, ...}}

CALL runs the simulator on that pattern and you get its report. ANSWER gives your final pattern and ends the task. Name only the inputs you set. Every input you do not name stays unknown (x), and an unknown value can block detection. You may make at most {calls} calls. Answer as soon as a call detects the fault. When a call does not detect it, use the report to change the pattern, and do not repeat a pattern you already tried."""

HINT_NOTES = """How to read the suggestion lines in the circuit text:
- "// suggested inputs:" lists inputs the graph model would set, as name=value. A "?" after a value means the model is unsure of that value.
- "// possibly needed:" lists further inputs that may have to be set, most likely first.
- "// observe at:" lists outputs where the fault effect is expected."""

PLAN_NOTES = """The graph model's complete first pattern:
{first}
Use exactly this pattern for your first CALL.

If it does not detect the fault, these single changes to that pattern are the graph model's best guesses, most promising first:
{changes}
Each one means: keep the first pattern and set that one input to that value. Try them, and use the simulator's reports to choose among them, to combine them or to go beyond them."""

USER = """Fault: {fault}

{notes}

Circuit:
```
{circuit}
```"""

_LINE = re.compile(r"(CALL|ANSWER)\s*:\s*(\{.*?\})", re.S)


def circuit_text(record, hints, view: str) -> Optional[str]:
    """The circuit as the prompt prints it, with the hint lines, or None
    when the row has no hint."""
    from atpgllm.multimodal.data import GRAPH_CONTEXT, _compact_netlist_payload, ports_extra

    extra = ports_extra(record, None, hints=hints)
    if extra is None:
        return None
    if view == "ports":
        _, compact = _compact_netlist_payload(record, "ports", extra)
        return ast.literal_eval(compact)["netlist"].replace(GRAPH_CONTEXT + "\n", "").replace(GRAPH_CONTEXT, "")
    netlist = record["netlist"]["netlist"] if isinstance(record["netlist"], dict) else record["netlist"]
    return extra + netlist


def row_hints(path: Path, records) -> Dict[tuple, str]:
    """The rich hint of each evaluation row. Only the validation files are
    read and only the rows' own problems are kept: the full hint tables
    hold every training problem."""
    from atpgllm.multimodal.data import netlist_key, rich_hint_text

    wanted = {(netlist_key(r), str(r["fault"]).strip()) for r in records}
    hints: Dict[tuple, str] = {}
    for file in sorted(Path(path).glob("validation_*.jsonl")):
        with file.open() as fh:
            for line in fh:
                row = json.loads(line)
                key = (row["netlist_key"], row["fault"])
                if key in wanted:
                    hints[key] = rich_hint_text(row["rich"])
    return hints


def row_rich(path: Path, records) -> Dict[tuple, dict]:
    """The graph model's probabilities for each evaluation row (the ``rich``
    field of its hint row), from the validation files only."""
    from atpgllm.multimodal.data import netlist_key

    wanted = {(netlist_key(r), str(r["fault"]).strip()) for r in records}
    rich: Dict[tuple, dict] = {}
    for file in sorted(Path(path).glob("validation_*.jsonl")):
        with file.open() as fh:
            for line in fh:
                row = json.loads(line)
                key = (row["netlist_key"], row["fault"])
                if key in wanted:
                    rich[key] = row["rich"]
    return rich


def plan_text(rich: Dict[str, Any], inputs: List[str], changes: int = 15) -> str:
    """The graph decoder's first pattern and its ranked single changes, as
    the scripted retry of ``hint_arms_eval`` would try them."""
    vectors = ev.decoder_attempts(rich, inputs, changes + 1)
    first = {name: value for name, value in vectors[0].items() if value != "x"}
    steps = []
    for vector in vectors[1:]:
        name = next(n for n in inputs if vector[n] != vectors[0][n])
        steps.append(f"{name}={vector[name]}")
    return PLAN_NOTES.format(first=json.dumps(first), changes=", ".join(steps) or "none")


def parse_reply(text: str, inputs: List[str]) -> tuple:
    """``(kind, vector)`` of the last CALL or ANSWER line, or ``(None, None)``.
    The vector names every input; the ones the model left out are x."""
    found = _LINE.findall(text or "")
    for kind, body in reversed(found):
        try:
            named = json.loads(body)
        except json.JSONDecodeError:
            try:
                named = ast.literal_eval(body)
            except (ValueError, SyntaxError):
                continue
        if not isinstance(named, dict):
            continue
        vector: Dict[str, Any] = {name: "x" for name in inputs}
        for name, value in named.items():
            if name in vector and str(value).strip() in ("0", "1"):
                vector[name] = int(str(value).strip())
        return kind, vector
    return None, None


def final_text(message) -> str:
    """The reply without the model's hidden reasoning, whichever way the
    server returns it."""
    text = message.content or ""
    if "<|message|>" in text:        # channel markers left in the text
        text = text.rsplit("<|message|>", 1)[1]
    return text


def probe_row(client, args, record, hints, index: int, rich=None) -> Dict[str, Any]:
    import multimodal_training as mt
    from atpgllm.multimodal.data import netlist_key
    from atpgllm.training.tool_feedback import read_snapshot

    base = {"row": index, "name": args.name, "circuit_id": record["circuit_id"], "fault": record["fault"],
            "kind": ev.kind(record), "location": ev.location(record), "difficulty": record.get("difficulty")}
    circuit = circuit_text(record, hints, args.view)
    if circuit is None:
        return {**base, "rejected": True}
    if len(circuit) > args.max_circuit_chars:
        return {**base, "rejected": True, "reason": "circuit text too long for the context"}
    inputs, outputs = r6.inputs_of(record), r6.outputs_of(record)
    run_tools = mt._tool_runner(copy.deepcopy(record), "po_fault")
    notes = HINT_NOTES
    if rich is not None:
        probabilities = rich.get((netlist_key(record), str(record["fault"]).strip()))
        if probabilities is None:
            return {**base, "rejected": True}
        notes += "\n\n" + plan_text(probabilities, inputs)
    messages = [{"role": "system", "content": SYSTEM.format(calls=args.max_tool_calls)},
                {"role": "user", "content": USER.format(fault=record["fault"], notes=notes, circuit=circuit)}]
    calls, turns, tokens = [], [], 0
    final, truncated, limit_hit, repeats, seen = None, False, False, 0, {}
    while True:
        try:
            reply = client.chat.completions.create(
                model=args.model, messages=messages, max_tokens=args.max_turn_tokens,
                temperature=0.0, extra_body={"reasoning_effort": args.reasoning_effort})
        except Exception as exc:     # the context is full, or the server refused the request
            truncated = True
            turns.append(f"request failed: {type(exc).__name__}: {str(exc)[:300]}")
            break
        choice = reply.choices[0]
        tokens += reply.usage.completion_tokens if reply.usage else 0
        text = final_text(choice.message)
        turns.append(text[-1200:])
        kind, vector = parse_reply(text, inputs)
        if kind is None:
            truncated = choice.finish_reason == "length"
            break
        if kind == "ANSWER":
            final = vector
            break
        if len(calls) >= args.max_tool_calls:
            limit_hit = True
            break
        key = json.dumps(vector, sort_keys=True)
        if key in seen:
            repeats += 1
            content = seen[key]
        else:
            call = {"name": "fault_simulation_tool",
                    "arguments": {"input_vector": vector, "output_vector": {name: "x" for name in outputs},
                                  "fault": record["fault"], "doc_id": netlist_key(record)}}
            content = seen[key] = run_tools([call])[0]
        try:
            detected = bool(read_snapshot(content, record["fault"], outputs).detected)
        except (ValueError, KeyError, TypeError):
            detected = False
        calls.append({"vector": vector, "detected": detected,
                      "malformed": content.startswith("Tool execution failed")})
        messages += [{"role": "assistant", "content": text},
                     {"role": "user", "content": f"Simulator report for call {len(calls)} of {args.max_tool_calls}:\n"
                                                 f"{content}\nFault detected: {'yes' if detected else 'no'}."}]
    return {**base, "calls": calls, "final_vector": final,
            "final_detected": bool(final) and r6._simulate(record, final)[0],
            "truncated": truncated, "call_limit_hit": limit_hit, "reused_calls": repeats,
            "tokens": tokens, "turns": turns}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="Folder with rows.jsonl of hint_arms_eval.")
    parser.add_argument("--name", required=True, help="The probe's label in the report.")
    parser.add_argument("--view", choices=("ports", "netlist"), default="ports")
    parser.add_argument("--hints", type=Path, required=True)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--max-tool-calls", type=int, default=8)
    parser.add_argument("--max-turn-tokens", type=int, default=8192)
    parser.add_argument("--max-circuit-chars", type=int, default=200_000)
    parser.add_argument("--plan", action="store_true",
                        help="Also give the graph decoder's complete first pattern and its ranked "
                             "single changes (what the scripted retry would try).")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    from openai import OpenAI

    client = OpenAI(base_url=args.base_url, api_key="none", timeout=1800)
    records = ev.load_rows(args.out)
    hints = row_hints(args.hints, records)
    rich = row_rich(args.hints, records) if args.plan else None
    out_path = args.out / f"rollout_{args.name}.jsonl"
    done = r6._done(out_path, lambda r: r["row"])
    todo = [i for i in range(len(records)) if i not in done]
    if args.limit:
        todo = todo[:args.limit]
    lock, started, count = threading.Lock(), time.time(), [0]

    def work(index: int) -> None:
        row = probe_row(client, args, records[index], hints, index, rich)
        with lock, out_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
            count[0] += 1
            print(f"probe {args.name} {count[0]}/{len(todo)} {time.time() - started:.0f}s", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in [pool.submit(work, index) for index in todo]:
            future.result()


if __name__ == "__main__":
    main()
