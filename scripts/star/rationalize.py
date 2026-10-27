#!/usr/bin/env python
"""Sample tool conversations for v4 train faults, unhinted and with hints.

Each selected train problem is attempted ``--num_samples`` times per hint mode
with the evaluator's conversation runner and TetraMAX verifier. Every slot is
written to ``<out>/slots[.shard<i>].jsonl`` with its messages, the hint-free
prompt, the judged facts and the source row's identifiers; ``build_sft.py``
filters them.

    FAULT_SIM_BACKEND=tetramax python scripts/star/rationalize.py \
        --adapter runs/sft_granite_4.2_8b_v4/checkpoint-250 \
        --dataset ../data/freeset/dataset.freeset.asap7sc7p5t_28.rvt.tt.v4 \
        --per_level hard=100,very_hard=100,random_never_detects=100 \
        --hint_modes none,partial --num_samples 4 --out runs/star_pilot

With ``--escalate`` a hint mode is tried only on problems that every earlier
mode failed, which is how STaR uses rationalization. ``--write_problems``
stores the selection (CPU only) so several ``--shard`` processes, one per GPU,
can share it through ``--problems_file``. A rerun into the same ``--out``
continues where the last one stopped. ``--chunk_problems`` with ``--deadline``
bounds the wall time: every mode finishes on a chunk before the next one, and
no chunk starts after the deadline.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "eval"))


def parse_counts(spec):
    counts = {}
    for item in filter(None, (p.strip() for p in spec.split(","))):
        level, _, n = item.partition("=")
        counts[level.strip()] = int(n)
    return counts


def local_model(name):
    """The cached snapshot of a Hub model id, so offline loads never query the Hub.

    transformers 4.57 asks the Hub about a model id while loading its
    tokenizer even with HF_HUB_OFFLINE=1 (juno job 473238); a local path does not.
    """
    if Path(name).exists():
        return str(name)
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(name, local_files_only=True)
    except Exception:
        return name


def problem_keys(paths):
    """Problem keys of earlier selections (``--write_problems`` files)."""
    import pyarrow.parquet as pq
    keys = set()
    for path in paths:
        for row in pq.read_table(path, columns=["netlist_id", "fault"]).to_pylist():
            keys.add((str(row["netlist_id"]), str(row["fault"])))
    return keys


def read_selected(dataset, counts, seed, split="train", exclude=()):
    """One seeded row per problem, ``3 * counts[level]`` problems per level.

    Selection reads only the key columns; full rows (netlists) are read for the
    chosen problems alone. The surplus covers the gate and prompt-length filters.
    """
    import pyarrow.parquet as pq
    from atpgllm.training.rationalization import select_problems
    paths = sorted((Path(dataset) / split).glob("*.parquet"))
    keys = []
    for index, path in enumerate(paths):
        table = pq.read_table(path, columns=["difficulty", "netlist_id", "fault"])
        for offset, row in enumerate(table.to_pylist()):
            if row["difficulty"] in counts:
                keys.append(dict(row, _file=index, _offset=offset))
    chosen = select_problems(keys, {k: 3 * v for k, v in counts.items()}, seed=seed, exclude=exclude)
    by_file = {}
    for order, key in enumerate(chosen):
        by_file.setdefault(key["_file"], []).append((order, key["_offset"]))
    rows = [None] * len(chosen)
    for index, wanted in by_file.items():
        table = pq.read_table(paths[index]).take([offset for _, offset in wanted])
        for (order, _), row in zip(wanted, table.to_pylist()):
            rows[order] = row
    return rows


def prepare_problems(dataset, counts, seed, tokenizer, max_prompt_length, *, split="train",
                     exclude=(), rows=None):
    """``[(row, record, clean prompt messages)]`` and the count kept per level.

    Keeps the first ``counts[level]`` candidates per level that pass the gate
    filter and whose hint-free prompt fits the GRPO prompt budget. ``rows``
    (a stored selection) replaces reading the dataset; all of them are kept.
    """
    from atpgllm.training.conversation import ConversationExample
    from atpgllm.training.dataset_utils import _gate_filter_passes
    from atpgllm.training.tools import TOOLS
    pool = read_selected(dataset, counts, seed, split, exclude) if rows is None else rows
    if rows is not None:
        counts = Counter(row["difficulty"] for row in rows)
    problems, taken = [], dict.fromkeys(counts, 0)
    for row in pool:
        if taken[row["difficulty"]] >= counts[row["difficulty"]] or not _gate_filter_passes(row):
            continue
        # The prompt and the verifier both derive doc_id = sha256(netlist)[:16].
        record = dict(row)
        clean = ConversationExample.prompt_messages_from_record(dict(record), use_tools=True)
        prompt = tokenizer.apply_chat_template(clean, tools=TOOLS, tokenize=False, add_generation_prompt=True)
        if len(tokenizer.encode(prompt, add_special_tokens=False)) >= max_prompt_length:
            continue
        taken[row["difficulty"]] += 1
        problems.append((row, record, clean))
    print(f"[star] {len(pool)} candidates -> {len(problems)} problems: {taken}", flush=True)
    return problems, taken


def load_progress(path, num_samples):
    """Completed (problem, mode) groups of an earlier run of this shard.

    Returns ``(done, solved)``: the (problem, mode) pairs with all
    ``num_samples`` slots written, and the problems some slot detected. A
    group cut short by a crash is removed from the file so it is redone whole.
    """
    path = Path(path)
    if not path.exists():
        return set(), set()
    groups = {}
    for line in open(path):
        try:
            slot = json.loads(line)
        except json.JSONDecodeError:
            continue  # a line torn by the crash
        groups.setdefault((tuple(slot["problem"]), slot["hint_mode"]), []).append(line)
    complete = {key: lines for key, lines in groups.items() if len(lines) == num_samples}
    if len(complete) != len(groups):
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as sink:
            for lines in complete.values():
                sink.writelines(line if line.endswith("\n") else line + "\n" for line in lines)
        os.replace(tmp, path)
        print(f"[star] resume: dropped {len(groups) - len(complete)} incomplete groups", flush=True)
    solved = set()
    for (problem, _), lines in complete.items():
        if any(json.loads(line)["facts"]["detected"] for line in lines):
            solved.add(problem)
    return set(complete), solved


def sample_and_write(strategy, problems, modes, tokenizer, sink, *, seed, batch_problems,
                     escalate=False, done=(), solved=(), chunk_problems=None, deadline=None):
    """Attempt every ``(row, record, clean prompt)`` problem under each hint mode
    and append one JSON line per completion slot to ``sink``.

    ``done`` holds (problem, mode) pairs already written; with ``escalate`` a
    mode after the first skips problems in ``solved`` or solved by an earlier
    mode. A hinted mode skips problems for which it has no hint.

    With ``chunk_problems`` the problems are taken that many at a time, and
    every mode finishes on a chunk before the next chunk starts, so hinted
    attempts keep pace with unhinted ones. No chunk starts after ``deadline``
    (seconds since the epoch). Returns False when the deadline cut it short.
    """
    done, solved = set(done), set(solved)
    size = chunk_problems or len(problems) or 1
    for start in range(0, len(problems), size):
        if deadline is not None and time.time() >= deadline:
            print(f"[star] deadline reached after {start} of {len(problems)} problems", flush=True)
            return False
        _sample_modes(strategy, problems[start:start + size], modes, tokenizer, sink, seed=seed,
                      batch_problems=batch_problems, escalate=escalate, done=done, solved=solved)
    return True


def _sample_modes(strategy, problems, modes, tokenizer, sink, *, seed, batch_problems, escalate, done, solved):
    from atpgllm.training.rationalization import hint_bits, hint_text, hinted_messages, judge, problem_key
    from atpgllm.training.tools import TOOLS
    for index, mode in enumerate(modes):
        todo = []
        for problem in problems:
            key = problem_key(problem[0])
            if (key, mode) in done or (escalate and index and key in solved):
                continue
            hint = hint_text(problem[0], mode, seed=seed)
            if mode != "none" and hint is None:
                continue
            todo.append((problem, hint))
        print(f"[star] {mode}: {len(todo)} problems to sample", flush=True)
        for start in range(0, len(todo), batch_problems):
            batch = todo[start:start + batch_problems]
            prompts, records = [], []
            for (row, record, clean), hint in batch:
                messages = hinted_messages(clean, hint)
                prompts.append(tokenizer.apply_chat_template(
                    messages, tools=TOOLS, tokenize=False, add_generation_prompt=True))
                records.append({**record, "_search_messages": messages})
            t = time.time()
            results = strategy.sample_batch(prompts, records)
            for ((row, _, clean), hint), result in zip(batch, results):
                for slot in result.slots:
                    facts = judge(slot, row, hint_bits(row, mode, seed=seed))
                    if facts["detected"]:
                        solved.add(problem_key(row))
                    sink.write(json.dumps({
                        "problem": list(problem_key(row)), "difficulty": row["difficulty"],
                        "fault": row["fault"], "circuit_id": row["circuit_id"],
                        "pattern_index": row["pattern_index"],
                        "random_detection_probability": row["random_detection_probability"],
                        "hint_mode": mode, "hint": hint, "prompt_messages": clean,
                        "facts": facts, **slot,
                    }, default=str) + "\n")
            sink.flush()
            print(f"[star] {mode}: problems {start}-{start + len(batch)} of {len(todo)} in "
                  f"{time.time() - t:.0f} s", flush=True)


def build_strategy(adapter, *, num_samples, temperature, top_p, max_tool_rounds, max_model_len,
                   max_generated_tokens, seed, tp_size=1, gpu_memory_utilization=0.9,
                   merge_dequant=True, tool_workers=16):
    """The evaluator's single-completion strategy on vLLM with the TetraMAX verifier.

    Returns ``(strategy, tokenizer, info)``; ``info`` records the serving mode
    and search configuration for ``run.json``.
    """
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from evaluate_model import resolve_merged_bf16_dir
    from atpgllm.training.reward_function_factory import RewardFunctionFactory
    from atpgllm.training._paths import resolve_sim_config_path
    from atpgllm.training.sampling_strategies import make_strategy
    from atpgllm.training.search_backends import make_vllm_generator
    from atpgllm.training.search_types import SearchConfig
    from atpgllm.training.search_verifier import Verifier

    adapter = Path(adapter).resolve()
    base = local_model(json.loads((adapter / "adapter_config.json").read_text())["base_model_name_or_path"])
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    serving = "lora"
    if merge_dequant:
        try:
            merged = resolve_merged_bf16_dir(adapter)
            serving = "merged_bf16"
        except Exception as exc:  # the export has its own CUDA subprocess; fall back to LoRA
            print(f"[star] merged bf16 export failed ({exc}); serving the adapter with LoRA", flush=True)
    if serving == "merged_bf16":
        llm = LLM(model=str(merged), tokenizer=base, tensor_parallel_size=tp_size,
                  gpu_memory_utilization=gpu_memory_utilization, dtype="bfloat16",
                  max_model_len=max_model_len, trust_remote_code=True, seed=seed)
        lora = None
    else:
        rank = json.loads((adapter / "adapter_config.json").read_text()).get("r", 8)
        llm = LLM(model=base, tensor_parallel_size=tp_size, enable_lora=True, max_lora_rank=rank,
                  gpu_memory_utilization=gpu_memory_utilization, dtype="bfloat16",
                  max_model_len=max_model_len, trust_remote_code=True, seed=seed)
        lora = LoRARequest("active_adapter", 1, str(adapter))
    params = SamplingParams(n=1, temperature=temperature, top_p=top_p,
                            max_tokens=max_generated_tokens,
                            stop_token_ids=[tokenizer.eos_token_id, tokenizer.pad_token_id])
    generator = make_vllm_generator(llm, tokenizer, lora, params)
    verifier = Verifier(RewardFunctionFactory(config_path=str(resolve_sim_config_path())))
    config = SearchConfig(max_generated_tokens=max_generated_tokens)
    strategy = make_strategy("single_completion", generator, verifier, num_completions=num_samples,
                             use_tools=True, max_tool_rounds=max_tool_rounds, seed=seed,
                             search_config=config, tool_workers=tool_workers)
    return strategy, tokenizer, {"base_model": base, "serving": serving, "search_config": asdict(config)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", required=True, help="SFT checkpoint (adapter_config.json inside)")
    parser.add_argument("--dataset", required=True, help="Local v4 dataset directory with train/")
    parser.add_argument("--split", default="train", help="Split to select from (validation for an eval set)")
    parser.add_argument("--per_level", default="hard=100,very_hard=100,random_never_detects=100")
    parser.add_argument("--hint_modes", default="none,partial", help="Comma list of none,path,partial,cube")
    parser.add_argument("--escalate", action="store_true",
                        help="Try each mode after the first only on problems all earlier modes failed")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tool_rounds", type=int, default=4)
    parser.add_argument("--max_prompt_length", type=int, default=8192)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--max_generated_tokens", type=int, default=24576)
    parser.add_argument("--batch_problems", type=int, default=64)
    parser.add_argument("--chunk_problems", type=int, default=None,
                        help="Finish every hint mode on this many problems before taking more")
    parser.add_argument("--deadline", type=float, default=None,
                        help="Start no chunk after this time (seconds since the epoch)")
    parser.add_argument("--tool_workers", type=int, default=16,
                        help="Concurrent TetraMAX simulations (tool calls and final answers)")
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--select_only", action="store_true",
                        help="Write the chosen problems and their hints, then stop before loading a model")
    parser.add_argument("--write_problems", default=None,
                        help="Also store the selected rows as parquet for --problems_file")
    parser.add_argument("--problems_file", default=None,
                        help="Sample a stored selection instead of selecting from --dataset")
    parser.add_argument("--exclude_problems", default="",
                        help="Comma list of earlier --write_problems files whose problems are never chosen")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--merge_dequant", action=argparse.BooleanOptionalAction, default=True,
                        help="Serve the QLoRA-faithful merged bf16 export (as the evaluator does)")
    args = parser.parse_args()
    if not 0 <= args.shard < args.num_shards:
        parser.error("--shard must be in [0, --num_shards)")

    from atpgllm.training.rationalization import hint_text, problem_key
    from transformers import AutoTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    counts = parse_counts(args.per_level)
    modes = [m.strip() for m in args.hint_modes.split(",") if m.strip()]
    adapter = Path(args.adapter).resolve()
    base = local_model(json.loads((adapter / "adapter_config.json").read_text())["base_model_name_or_path"])
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)

    t0 = time.time()
    rows = None
    if args.problems_file:
        import pyarrow.parquet as pq
        rows = pq.read_table(args.problems_file).to_pylist()
    exclude = problem_keys([p for p in args.exclude_problems.split(",") if p])
    problems, taken = prepare_problems(args.dataset, counts, args.seed, tokenizer, args.max_prompt_length,
                                       split=args.split, exclude=exclude, rows=rows)
    print(f"[star] selection took {time.time() - t0:.0f} s", flush=True)
    if args.write_problems:
        import pyarrow as pa
        import pyarrow.parquet as pq
        target = Path(args.write_problems)
        pq.write_table(pa.Table.from_pylist([row for row, _, _ in problems]), target.with_suffix(".tmp"))
        os.replace(target.with_suffix(".tmp"), target)
    if args.shard == 0:
        with open(out / "problems.jsonl", "w") as sink:
            for row, _, _ in problems:
                sink.write(json.dumps({"problem": list(problem_key(row)), "difficulty": row["difficulty"],
                                       "random_detection_probability": row["random_detection_probability"],
                                       "hints": {m: hint_text(row, m, seed=args.seed) for m in modes}}) + "\n")
    if args.select_only:
        return
    problems = problems[args.shard::args.num_shards]
    if args.deadline is not None:
        # A deadline may cut the list short; mix the levels so the cut is even.
        random.Random(args.seed + args.shard).shuffle(problems)
    name = "slots.jsonl" if args.num_shards == 1 else f"slots.shard{args.shard}.jsonl"
    done, solved = load_progress(out / name, args.num_samples)

    from atpgllm.training.simulator_provenance import runtime_provenance
    strategy, tokenizer, info = build_strategy(
        adapter, num_samples=args.num_samples, temperature=args.temperature, top_p=args.top_p,
        max_tool_rounds=args.max_tool_rounds, max_model_len=args.max_model_len,
        max_generated_tokens=args.max_generated_tokens, seed=args.seed, tp_size=args.tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization, merge_dequant=args.merge_dequant,
        tool_workers=args.tool_workers)
    if args.shard == 0:
        (out / "run.json").write_text(json.dumps({
            "args": vars(args), "problems": taken, "simulator": runtime_provenance(), **info},
            indent=1, default=str))

    with open(out / name, "a") as sink:
        complete = sample_and_write(strategy, problems, modes, tokenizer, sink, seed=args.seed,
                                    batch_problems=args.batch_problems, escalate=args.escalate,
                                    done=done, solved=solved, chunk_problems=args.chunk_problems,
                                    deadline=args.deadline)
    (out / f"{name}.done").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + ("" if complete else " deadline") + "\n")


if __name__ == "__main__":
    main()
