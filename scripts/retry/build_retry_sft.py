#!/usr/bin/env python
"""Build retry SFT examples from the policy's own failed calls.

Input: RFT generation slots (``slots*.jsonl`` from rationalize.py). A slot is a
candidate when it is unhinted, made exactly one tool call, and that call did
not detect the fault. For each candidate:

1. ``search``: the fast simulator looks for the fewest input flips that fix
   what the failed call's snapshot shows (activation, then propagation), one
   readable step at a time (atpgllm.training.retry_repair).
2. ``confirm``: every call of the path, the failed one included, is simulated
   again through GDPO's own tool handler, so each tool message is exactly what
   the GDPO tool returns. The readings and blocking side inputs are taken from
   those snapshots. A path whose detection or activation differs is dropped.
3. ``assemble``: the conversation (failed call masked, then reading, new call,
   answer, ..., final report), length check, and the output files.

``--kept`` also rewrites an existing single-call sft.jsonl with the stock lines
replaced (atpgllm.training.retry_trajectories.ground_example).

Run ``confirm`` under the TetraMAX service, like RFT generation::

    scripts/rft/with_tetramax.sh python scripts/retry/build_retry_sft.py \\
        --slots 'runs/rft_v4_a100/round_1[01]/gen/slots.shard*.jsonl' \\
        --kept runs/rft_v4_a100/round_11/gen/sft.jsonl --out runs/retry_sft_v4/data

Every phase writes its own file and a rerun continues from it.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import hashlib
import itertools
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def _slot_parts(slot):
    """(user text, first call message, call arguments, tool content) of a one-call slot."""
    user = next(m for m in slot["prompt_messages"] if m["role"] == "user")["content"]
    calls = [m for m in slot["messages"] if m["role"] == "assistant" and m.get("tool_calls")]
    tools = [m for m in slot["messages"] if m["role"] == "tool"]
    if len(calls) != 1 or len(tools) != 1 or len(calls[0]["tool_calls"]) != 1:
        return None
    arguments = calls[0]["tool_calls"][0]["function"]["arguments"]
    if not isinstance(arguments, dict) or not isinstance(arguments.get("input_vector"), dict) \
            or not isinstance(arguments.get("output_vector"), dict):
        return None
    return user, calls[0], arguments, tools[0]["content"]


def select(patterns, per_level, max_per_problem, seed):
    """Unhinted one-call failures, at most ``max_per_problem`` per problem and ``per_level`` per level."""
    from atpgllm.training.tool_feedback import read_snapshot
    by_level = defaultdict(lambda: defaultdict(list))
    seen = Counter()
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            round_name = Path(path).parent.parent.name
            for line in open(path):
                slot = json.loads(line)
                seen["slots"] += 1
                if slot.get("hint_mode") != "none" or slot.get("status") != "FINAL":
                    continue
                if (slot.get("reward_components") or {}).get("detection", 0) >= 1:
                    continue
                parts = _slot_parts(slot)
                if parts is None:
                    seen["not_one_call"] += 1
                    continue
                _, _, arguments, tool = parts
                try:
                    reading = read_snapshot(tool, slot["fault"], list(arguments["output_vector"]))
                except Exception:
                    seen["tool_unreadable"] += 1
                    continue
                if reading.detected:
                    seen["call_detected_answer_wrong"] += 1
                    continue
                slot["_round"] = round_name
                by_level[slot["difficulty"]][tuple(slot["problem"])].append(slot)
    rng = random.Random(seed)
    chosen = []
    for level, problems in sorted(by_level.items()):
        keys = sorted(problems)
        rng.shuffle(keys)
        taken = []
        for key in keys:
            slots = problems[key]
            rng.shuffle(slots)
            vectors = set()
            for slot in slots:
                vector = json.dumps(_slot_parts(slot)[2]["input_vector"], sort_keys=True)
                if vector in vectors:
                    continue
                vectors.add(vector)
                taken.append(slot)
                if len(vectors) >= max_per_problem:
                    break
            if len(taken) >= per_level:
                break
        chosen += taken[:per_level]
        seen[f"candidates/{level}"] = len(taken[:per_level])
    return chosen, dict(seen)


DATASET_COLUMNS = ("system_content", "user_content", "assistant_content", "module_name", "netlist",
                   "input_vector", "expected_output", "snapshot", "detected_faults", "fault",
                   "pattern_index", "fault_propagation_gates", "fault_propagation_nets",
                   "backtrack_gates", "backtrack_nets", "reasoning_content", "answer_content",
                   "difficulty", "netlist_id", "circuit_id", "circuit_kind")


def synthetic_failure(record, simulate, rng, *, max_flips=3, tries=24):
    """A confirmed miss near the record's detecting pattern: 1-3 care bits flipped.

    Flips prefer the backtrace inputs (they decide activation and propagation)
    and aim half the time for each kind of miss, so both readings are taught.
    Returns ``(vector, snapshot)`` or None.
    """
    from atpgllm.training.tool_feedback import read_snapshot
    gold = json.loads(record["input_vector"])
    outputs = list(json.loads(record["expected_output"]))
    care = [net for net in gold if str(gold[net]) in ("0", "1")]
    if not care or not outputs:
        return None
    backtrace = {n.strip() for n in str(record.get("backtrack_nets") or "").split(",") if n.strip()}
    preferred = [net for net in care if net in backtrace]
    want_activated = rng.random() < 0.5
    fallback = None
    for _ in range(tries):
        if fallback is not None and not want_activated:
            break
        pool = preferred if preferred and rng.random() < 0.8 else care
        count = rng.choice([1, 1, 2, 3][: max_flips + 1])
        vector = dict(gold)
        for net in rng.sample(pool, min(len(pool), count)):
            vector[net] = 1 - int(vector[net])
        try:
            snapshot = simulate(vector)
        except (ValueError, KeyError, TypeError):
            return fallback
        reading = read_snapshot(snapshot, record["fault"], outputs)
        if reading.detected:
            continue
        if reading.activated == want_activated:
            return vector, snapshot
        fallback = fallback or (vector, snapshot)
    return fallback


def _synthetic_slot(job):
    """A pseudo RFT slot for one dataset row: prompt, a failed call and its snapshot."""
    record, seed = job
    os.environ.setdefault("FAULT_SIM_BACKEND", "fast")
    from atpgllm.training.conversation import ConversationExample
    from atpgllm.training.retry_repair import fast_simulator, fast_structure
    from atpgllm.training.retry_trajectories import outputs_of, strip_assignment_lines
    try:
        messages = ConversationExample.from_record(dict(record), use_tools=True, grounded=True).messages
        outputs = list(json.loads(record["expected_output"]))
        structure = fast_structure(record["netlist"])
        simulate = fast_simulator(structure, record["fault"], outputs)
        failure = synthetic_failure(record, simulate, random.Random(seed))
    except Exception as exc:
        return {"status": f"error: {type(exc).__name__}: {exc}"[:300]}
    if failure is None:
        return {"status": "no_failure"}
    vector, snapshot = failure
    call = dict(messages[2])
    # The dataset reasoning spells out the detecting vector; keep only the plan.
    call["content"] = strip_assignment_lines(call.get("content") or "")
    function = dict(call["tool_calls"][0]["function"])
    function["arguments"] = dict(function["arguments"], input_vector=vector,
                                 output_vector=outputs_of(snapshot, outputs))
    call["tool_calls"] = [{"type": "function", "function": function}]
    return {"status": "slot", "slot": {
        "problem": [record.get("netlist_id") or record.get("circuit_id"), record["fault"]],
        "difficulty": record.get("difficulty") or "unknown", "kind": record.get("circuit_kind") or "combinational",
        "seed": seed, "completion_slot": int(record.get("pattern_index") or 0), "fault": record["fault"],
        "hint_mode": "none", "status": "FINAL", "reward_components": {"detection": 0.0},
        "prompt_messages": messages[:2],
        "messages": [call, {"role": "tool", "name": "fault_simulation_tool", "content": snapshot}],
        "_round": "dataset"}}


def select_dataset(directory, split, per_group, seed, workers):
    """Synthetic failures from dataset rows, ``per_group`` per (kind, level)."""
    import pyarrow.parquet as pq
    from atpgllm.training.kinds import dataset_kinds, shard_kinds
    rng = random.Random(seed)
    files = sorted(glob.glob(str(Path(directory) / split / "*.parquet")))
    rng.shuffle(files)
    kinds = sorted(dataset_kinds(directory)) or ["combinational"]
    levels = ["easy", "medium", "hard", "very_hard", "random_never_detects"]
    expected = {(kind, level) for kind in kinds for level in levels}
    file_kinds = shard_kinds([Path(f) for f in files]) or [None] * len(files)
    if any(file_kinds):
        # Alternate kinds, so both fill early (a v5 shard holds one kind).
        by_kind = defaultdict(list)
        for path, kind in zip(files, file_kinds):
            by_kind[kind].append((path, kind))
        files, file_kinds = [], []
        for pair in itertools.zip_longest(*by_kind.values()):
            for item in pair:
                if item is not None:
                    files.append(item[0])
                    file_kinds.append(item[1])
    groups, seen = defaultdict(list), Counter()
    want = per_group * 2  # rows tried per group; about half yield a usable failure

    def full(kind=None):
        return all(len(groups[key]) >= want for key in expected if kind is None or key[0] == kind)

    for path, file_kind in zip(files, file_kinds):
        if full():
            break
        if file_kind is not None and full(file_kind):
            continue
        handle = pq.ParquetFile(path)
        # Every column: the prompt templates of a kind may name more fields (v5 scan_chain).
        for batch in handle.iter_batches(batch_size=4096):
            rows = batch.to_pylist()
            rng.shuffle(rows)
            for row in rows:
                key = (row.get("circuit_kind") or "combinational", row.get("difficulty") or "unknown")
                if len(groups[key]) < want:
                    groups[key].append(row)
            seen["rows_read"] += len(rows)
            if full(file_kind):
                break
    jobs = [(row, rng.randrange(2 ** 31)) for rows in groups.values() for row in rows]
    chosen, per_group_count, errors = [], Counter(), []
    with Pool(workers) as pool:
        for result in pool.imap_unordered(_synthetic_slot, jobs, chunksize=8):
            seen[result["status"] if not result["status"].startswith("error") else "error"] += 1
            if result["status"].startswith("error") and len(errors) < 5:
                errors.append(result["status"])
            if result["status"] != "slot":
                continue
            slot = result["slot"]
            key = (slot["kind"], slot["difficulty"])
            if per_group_count[key] < per_group:
                per_group_count[key] += 1
                chosen.append(slot)
    for (kind, level), count in sorted(per_group_count.items()):
        seen[f"candidates/{kind}/{level}"] = count
    if errors:
        seen["error_examples"] = errors
    return chosen, dict(seen)


def _search_one(job):
    slot, max_steps, max_flips = job
    os.environ.setdefault("FAULT_SIM_BACKEND", "fast")
    from atpgllm.training.retry_repair import RepairSearch, fast_simulator, fast_structure
    from atpgllm.training.tools import ToolHelper
    user, _, arguments, tool = _slot_parts(slot)
    key = {"problem": slot["problem"], "difficulty": slot["difficulty"], "round": slot["_round"],
           "seed": slot.get("seed"), "completion_slot": slot.get("completion_slot")}
    try:
        netlist = ToolHelper.get_document(user)["netlist"]
        outputs = list(arguments["output_vector"])
        structure = fast_structure(netlist)
        search = RepairSearch(structure, fast_simulator(structure, slot["fault"], outputs), slot["fault"],
                              list(arguments["input_vector"]), outputs,
                              seed=int(hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:8], 16),
                              max_flips=max_flips)
        steps = search.path(dict(arguments["input_vector"]), start_snapshot=tool, max_steps=max_steps)
    except Exception as exc:
        return {**key, "status": f"error: {type(exc).__name__}: {exc}"[:300]}
    if steps is None:
        return {**key, "status": "no_path", "simulations": search.simulations}
    return {**key, "status": "path", "simulations": search.simulations,
            "steps": [{"vector": s.observation.vector, "changes": [list(c) for c in s.changes]} for s in steps]}


def search_phase(chosen, out, workers, max_steps, max_flips):
    path = out / "paths.jsonl"
    done = set()
    if path.exists():
        for line in open(path):
            try:
                row = json.loads(line)
                done.add((tuple(row["problem"]), row["round"], row["seed"], row["completion_slot"]))
            except json.JSONDecodeError:
                pass
    jobs = [(slot, max_steps, max_flips) for slot in chosen
            if (tuple(slot["problem"]), slot["_round"], slot.get("seed"), slot.get("completion_slot")) not in done]
    print(f"[search] {len(jobs)} slots to search ({len(done)} done)", flush=True)
    t = time.time()
    with open(path, "a") as sink, Pool(workers) as pool:
        for i, row in enumerate(pool.imap_unordered(_search_one, jobs, chunksize=4), 1):
            sink.write(json.dumps(row) + "\n")
            if i % 500 == 0:
                sink.flush()
                print(f"[search] {i}/{len(jobs)} in {time.time() - t:.0f} s", flush=True)
    return [json.loads(line) for line in open(path)]


def _tool(vector, output_vector, fault, doc_id, netlist):
    from atpgllm.training.tools import fault_simulation_tool_handler
    return asyncio.run(fault_simulation_tool_handler(vector, output_vector, fault, doc_id, netlist))


def _confirm_one(item):
    """Simulate every call of a path with GDPO's tool handler and re-read each snapshot."""
    slot, row = item
    from atpgllm.training.retry_repair import RepairSearch, fast_simulator, fast_structure
    from atpgllm.training.retry_trajectories import outputs_of
    from atpgllm.training.tools import ToolHelper
    user, call, arguments, stored = _slot_parts(slot)
    document = ToolHelper.get_document(user)
    netlist, doc_id = document["netlist"], document["doc_id"]
    fault = slot["fault"]
    outputs = list(arguments["output_vector"])
    structure = fast_structure(netlist)
    search = RepairSearch(structure, fast_simulator(structure, fault, outputs), fault,
                          list(arguments["input_vector"]), outputs)
    first = _tool(arguments["input_vector"], arguments["output_vector"], fault, doc_id, netlist)
    if first.startswith("Tool "):
        return {"status": "first_call_tool_error", "detail": first[:200]}
    previous = search.explain(arguments["input_vector"], first)
    if previous.reading.detected:
        return {"status": "first_call_detects_on_tool"}
    steps = []
    for step in row["steps"]:
        fast = search.observe(step["vector"])
        snapshot = _tool(step["vector"], outputs_of(fast.snapshot, outputs), fault, doc_id, netlist)
        if snapshot.startswith("Tool "):
            return {"status": "step_tool_error", "detail": snapshot[:200]}
        observed = search.explain(step["vector"], snapshot)
        if (observed.reading.detected, observed.reading.activated) != \
                (fast.reading.detected, fast.reading.activated):
            return {"status": "tool_disagrees_with_search"}
        steps.append({"vector": step["vector"], "changes": [tuple(c) for c in step["changes"]],
                      "snapshot": snapshot, "prev_reading": previous.reading,
                      "prev_blockers": list(previous.blockers)})
        previous = observed
    if not previous.reading.detected:
        return {"status": "last_call_does_not_detect"}
    return {"status": "confirmed", "first_snapshot": first, "first_matches_stored": first == stored,
            "steps": steps, "doc_id": doc_id, "outputs": outputs, "first_call": call,
            "start_activated": search.explain(arguments["input_vector"], first).reading.activated}


def _key(problem, round_name, seed, completion_slot):
    return json.dumps([list(problem), round_name, seed, completion_slot])


def confirm_and_assemble(chosen, rows, out, tool_workers, tokenizer, max_model_len, confirm, cap=None):
    """Confirm each path with the tool, assemble its example; one line per path in confirm.jsonl."""
    from atpgllm.training.retry_trajectories import retry_example, sft_first_call
    from atpgllm.training.tools import TOOLS
    if confirm and os.environ.get("FAULT_SIM_BACKEND") != "tetramax":
        raise SystemExit("confirm needs FAULT_SIM_BACKEND=tetramax (run under scripts/rft/with_tetramax.sh)")
    by_key = {_key(s["problem"], s["_round"], s.get("seed"), s.get("completion_slot")): s for s in chosen}
    log = out / "confirm.jsonl"
    done = {}
    if log.exists():
        for line in open(log):
            try:
                entry = json.loads(line)
                done[entry["key"]] = entry
            except json.JSONDecodeError:
                pass
    items = []
    for r in rows:
        key = _key(r["problem"], r["round"], r["seed"], r["completion_slot"])
        if r["status"] == "path" and key in by_key and key not in done:
            items.append((by_key[key], r))
    print(f"[confirm] {len(items)} paths to confirm ({len(done)} done)", flush=True)

    def finish(slot, result):
        level = slot["difficulty"]
        entry = {"key": _key(slot["problem"], slot["_round"], slot.get("seed"), slot.get("completion_slot")),
                 "level": level, "status": result["status"]}
        if result["status"] != "confirmed":
            return entry
        entry["first_matches_stored"] = result["first_matches_stored"]
        prompt = [m for m in slot["prompt_messages"] if m["role"] in ("system", "user")]
        example = retry_example(prompt, sft_first_call(result["first_call"]), result["first_snapshot"],
                                result["steps"], slot["fault"], result["doc_id"], result["outputs"], TOOLS)
        length = len(tokenizer.apply_chat_template(example["messages"], tools=TOOLS, tokenize=True))
        if length > max_model_len:
            entry["status"] = "too_long"
            return entry
        example["retry"] = {"problem": slot["problem"], "difficulty": level, "round": slot["_round"],
                            "kind": slot.get("kind", "combinational"),
                            "steps": len(result["steps"]),
                            "flips": [len(s["changes"]) for s in result["steps"]],
                            "start": "activated" if result["start_activated"] else "not_activated",
                            "tokens": length}
        entry["example"] = example
        return entry

    t = time.time()
    with open(log, "a") as sink, ThreadPoolExecutor(max_workers=tool_workers if confirm else 1) as pool:
        for i, ((slot, _), result) in enumerate(zip(items, pool.map(_confirm_one, items)), 1):
            entry = finish(slot, result)
            done[entry["key"]] = entry
            sink.write(json.dumps(entry) + "\n")
            if i % 100 == 0:
                sink.flush()
                print(f"[confirm] {i}/{len(items)} in {time.time() - t:.0f} s", flush=True)
    stats, by_level, examples = Counter(), defaultdict(Counter), []
    for entry in done.values():
        stats[entry["status"]] += 1
        stats["first_matches_stored"] += entry.get("first_matches_stored", 0)
        by_level[entry["level"]][entry["status"]] += 1
        if "example" in entry:
            examples.append(entry["example"])
    examples.sort(key=lambda e: json.dumps(e["retry"]["problem"]))
    if cap:
        # Dataset mode: a rerun can confirm more rows of a group than asked
        # (pool results arrive unordered); keep the kinds and levels balanced.
        kept, count = [], Counter()
        for example in examples:
            key = (example["retry"].get("kind", "combinational"), example["retry"]["difficulty"])
            if count[key] < cap:
                count[key] += 1
                kept.append(example)
        stats["capped_out"] = len(examples) - len(kept)
        examples = kept
    with open(out / "retry.jsonl", "w") as sink:
        for example in examples:
            sink.write(json.dumps(example) + "\n")
    return examples, dict(stats), {k: dict(v) for k, v in sorted(by_level.items())}


def ground_kept(path, out, tokenizer, max_model_len):
    from atpgllm.training.retry_trajectories import ground_example
    from atpgllm.training.tools import TOOLS
    kept, stats = [], Counter()
    for line in open(path):
        example = json.loads(line)
        call = next((m for m in example["messages"] if m.get("tool_calls")), None)
        if call is None:
            stats["no_call"] += 1
            continue
        arguments = call["tool_calls"][0]["function"]["arguments"]
        outputs = arguments["output_vector"]
        outputs = list(json.loads(outputs) if isinstance(outputs, str) else outputs)
        grounded = ground_example(example, arguments["fault"], outputs)
        if grounded is None:
            stats["last_call_does_not_detect"] += 1
            continue
        grounded.pop("star", None)
        length = len(tokenizer.apply_chat_template(grounded["messages"], tools=TOOLS, tokenize=True))
        if length > max_model_len:
            stats["too_long"] += 1
            continue
        stats["examples"] += 1
        kept.append(grounded)
    with open(out / "kept_grounded.jsonl", "w") as sink:
        for example in kept:
            sink.write(json.dumps(example) + "\n")
    return kept, dict(stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slots", action="append", help="Glob of slots*.jsonl (repeatable)")
    parser.add_argument("--dataset", help="Dataset directory: synthetic failures from its rows instead of slots")
    parser.add_argument("--split", default="train")
    parser.add_argument("--per_group", type=int, default=600, help="Dataset mode: examples per (kind, level)")
    parser.add_argument("--kept", help="Single-call sft.jsonl to ground and mix in")
    parser.add_argument("--out", required=True)
    parser.add_argument("--per_level", type=int, default=2000)
    parser.add_argument("--max_per_problem", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=4)
    parser.add_argument("--max_flips", type=int, default=3)
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--tool_workers", type=int, default=16)
    parser.add_argument("--max_model_len", type=int, default=14336)
    parser.add_argument("--tokenizer", default="ibm-granite/granite-4.2-8b")
    parser.add_argument("--no_confirm", action="store_true",
                        help="Use the stored first snapshot and fast-simulator steps (tests only)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if bool(args.slots) == bool(args.dataset):
        parser.error("give --slots or --dataset")
    if args.dataset:
        chosen, selection = select_dataset(args.dataset, args.split, args.per_group, args.seed, args.workers)
    else:
        chosen, selection = select(args.slots, args.per_level, args.max_per_problem, args.seed)
    print(f"[select] {len(chosen)} failed calls: {selection}", flush=True)
    rows = search_phase(chosen, out, args.workers, args.max_steps, args.max_flips)
    search_stats = Counter(r["status"] if not r["status"].startswith("error") else "error" for r in rows)
    examples, confirm_stats, by_level = confirm_and_assemble(
        chosen, rows, out, args.tool_workers, tokenizer, args.max_model_len, not args.no_confirm,
        cap=args.per_group if args.dataset else None)
    report = {"selection": selection, "search": dict(search_stats), "confirm": confirm_stats,
              "by_level": by_level, "retry_examples": len(examples),
              "steps": dict(Counter(e["retry"]["steps"] for e in examples)),
              "start": dict(Counter(e["retry"]["start"] for e in examples)),
              "flips": dict(Counter(f for e in examples for f in e["retry"]["flips"])),
              "args": vars(args)}
    if args.kept:
        kept, kept_stats = ground_kept(args.kept, out, tokenizer, args.max_model_len)
        report["kept"] = kept_stats
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
