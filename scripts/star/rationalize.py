#!/usr/bin/env python
"""Sample hinted (and unhinted control) tool conversations for hard v4 faults.

Each selected train problem is attempted ``--num_samples`` times per hint mode
with the evaluator's conversation runner and TetraMAX verifier. Every slot is
written to ``<out>/slots.jsonl`` with its messages, the hint-free prompt, the
judged facts and the source row's identifiers; ``build_sft.py`` filters them.

    FAULT_SIM_BACKEND=tetramax python scripts/star/rationalize.py \
        --adapter runs/sft_granite_4.2_8b_v4/checkpoint-250 \
        --dataset ../data/freeset/dataset.freeset.asap7sc7p5t_28.rvt.tt.v4 \
        --per_level hard=100,very_hard=100,random_never_detects=100 \
        --hint_modes none,partial --num_samples 4 --out runs/star_pilot
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
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


def read_selected(dataset, counts, seed):
    """One seeded row per problem, ``3 * counts[level]`` problems per level.

    Selection reads only the key columns; full rows (netlists) are read for the
    chosen problems alone. The surplus covers the gate and prompt-length filters.
    """
    import pyarrow.parquet as pq
    from atpgllm.training.rationalization import select_problems
    paths = sorted((Path(dataset) / "train").glob("*.parquet"))
    keys = []
    for index, path in enumerate(paths):
        table = pq.read_table(path, columns=["difficulty", "netlist_id", "fault"])
        for offset, row in enumerate(table.to_pylist()):
            if row["difficulty"] in counts:
                keys.append(dict(row, _file=index, _offset=offset))
    chosen = select_problems(keys, {k: 3 * v for k, v in counts.items()}, seed=seed)
    by_file = {}
    for order, key in enumerate(chosen):
        by_file.setdefault(key["_file"], []).append((order, key["_offset"]))
    rows = [None] * len(chosen)
    for index, wanted in by_file.items():
        table = pq.read_table(paths[index]).take([offset for _, offset in wanted])
        for (order, _), row in zip(wanted, table.to_pylist()):
            rows[order] = row
    return rows


def prepare_problems(dataset, counts, seed, tokenizer, max_prompt_length):
    """``[(row, record, clean prompt messages)]`` and the count kept per level.

    Keeps the first ``counts[level]`` candidates per level that pass the gate
    filter and whose hint-free prompt fits the GRPO prompt budget.
    """
    from atpgllm.training.conversation import ConversationExample
    from atpgllm.training.dataset_utils import _gate_filter_passes
    from atpgllm.training.tools import TOOLS
    pool = read_selected(dataset, counts, seed)
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


def sample_and_write(strategy, problems, modes, tokenizer, sink, *, seed, batch_problems):
    """Attempt every ``(row, record, clean prompt)`` problem under each hint mode
    and append one JSON line per completion slot to ``sink``."""
    from atpgllm.training.rationalization import hint_bits, hint_text, hinted_messages, judge, problem_key
    from atpgllm.training.tools import TOOLS
    for mode in modes:
        for start in range(0, len(problems), batch_problems):
            batch = problems[start:start + batch_problems]
            prompts, records, hints = [], [], []
            for row, record, clean in batch:
                hint = hint_text(row, mode, seed=seed)
                messages = hinted_messages(clean, hint)
                prompts.append(tokenizer.apply_chat_template(
                    messages, tools=TOOLS, tokenize=False, add_generation_prompt=True))
                records.append({**record, "_search_messages": messages})
                hints.append(hint)
            t = time.time()
            results = strategy.sample_batch(prompts, records)
            for (row, _, clean), hint, result in zip(batch, hints, results):
                for slot in result.slots:
                    sink.write(json.dumps({
                        "problem": list(problem_key(row)), "difficulty": row["difficulty"],
                        "fault": row["fault"], "circuit_id": row["circuit_id"],
                        "pattern_index": row["pattern_index"],
                        "random_detection_probability": row["random_detection_probability"],
                        "hint_mode": mode, "hint": hint, "prompt_messages": clean,
                        "facts": judge(slot, row, hint_bits(row, mode, seed=seed)), **slot,
                    }, default=str) + "\n")
            sink.flush()
            print(f"[star] {mode}: problems {start}-{start + len(batch)} in {time.time() - t:.0f} s",
                  flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", required=True, help="SFT checkpoint (adapter_config.json inside)")
    parser.add_argument("--dataset", required=True, help="Local v4 dataset directory with train/")
    parser.add_argument("--per_level", default="hard=100,very_hard=100,random_never_detects=100")
    parser.add_argument("--hint_modes", default="none,partial", help="Comma list of none,path,partial,cube")
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
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--select_only", action="store_true",
                        help="Write the chosen problems and their hints, then stop before loading a model")
    parser.add_argument("--merge_dequant", action=argparse.BooleanOptionalAction, default=True,
                        help="Serve the QLoRA-faithful merged bf16 export (as the evaluator does)")
    args = parser.parse_args()

    from atpgllm.training.rationalization import hint_text, problem_key
    from transformers import AutoTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    counts = parse_counts(args.per_level)
    modes = [m.strip() for m in args.hint_modes.split(",") if m.strip()]
    adapter = Path(args.adapter).resolve()
    base = json.loads((adapter / "adapter_config.json").read_text())["base_model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)

    t0 = time.time()
    problems, taken = prepare_problems(args.dataset, counts, args.seed, tokenizer, args.max_prompt_length)
    print(f"[star] selection took {time.time() - t0:.0f} s", flush=True)
    with open(out / "problems.jsonl", "w") as sink:
        for row, _, _ in problems:
            sink.write(json.dumps({"problem": list(problem_key(row)), "difficulty": row["difficulty"],
                                   "random_detection_probability": row["random_detection_probability"],
                                   "hints": {m: hint_text(row, m, seed=args.seed) for m in modes}}) + "\n")
    if args.select_only:
        return

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from evaluate_model import resolve_merged_bf16_dir
    from atpgllm.training.reward_function_factory import RewardFunctionFactory
    from atpgllm.training._paths import resolve_sim_config_path
    from atpgllm.training.sampling_strategies import make_strategy
    from atpgllm.training.search_backends import make_vllm_generator
    from atpgllm.training.search_types import SearchConfig
    from atpgllm.training.search_verifier import Verifier
    from atpgllm.training.simulator_provenance import runtime_provenance

    serving = "lora"
    if args.merge_dequant:
        try:
            merged = resolve_merged_bf16_dir(adapter)
            serving = "merged_bf16"
        except Exception as exc:  # the export has its own CUDA subprocess; fall back to LoRA
            print(f"[star] merged bf16 export failed ({exc}); serving the adapter with LoRA", flush=True)
    if serving == "merged_bf16":
        llm = LLM(model=str(merged), tensor_parallel_size=args.tp_size,
                  gpu_memory_utilization=args.gpu_memory_utilization, dtype="bfloat16",
                  max_model_len=args.max_model_len, trust_remote_code=True)
        lora = None
    else:
        rank = json.loads((adapter / "adapter_config.json").read_text()).get("r", 8)
        llm = LLM(model=base, tensor_parallel_size=args.tp_size, enable_lora=True, max_lora_rank=rank,
                  gpu_memory_utilization=args.gpu_memory_utilization, dtype="bfloat16",
                  max_model_len=args.max_model_len, trust_remote_code=True)
        lora = LoRARequest("active_adapter", 1, str(adapter))
    params = SamplingParams(n=1, temperature=args.temperature, top_p=args.top_p,
                            max_tokens=args.max_generated_tokens,
                            stop_token_ids=[tokenizer.eos_token_id, tokenizer.pad_token_id])
    generator = make_vllm_generator(llm, tokenizer, lora, params)
    verifier = Verifier(RewardFunctionFactory(config_path=str(resolve_sim_config_path())))
    config = SearchConfig(max_generated_tokens=args.max_generated_tokens)
    strategy = make_strategy("single_completion", generator, verifier, num_completions=args.num_samples,
                             use_tools=True, max_tool_rounds=args.max_tool_rounds, seed=args.seed,
                             search_config=config)
    (out / "run.json").write_text(json.dumps({
        "args": vars(args), "base_model": base, "problems": taken, "serving": serving,
        "simulator": runtime_provenance(), "search_config": asdict(config)}, indent=1, default=str))

    with open(out / "slots.jsonl", "a") as sink:
        sample_and_write(strategy, problems, modes, tokenizer, sink, seed=args.seed,
                         batch_problems=args.batch_problems)


if __name__ == "__main__":
    main()
