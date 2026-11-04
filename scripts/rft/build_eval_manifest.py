#!/usr/bin/env python
"""Freeze the evaluation problem set shared by RFT rounds and GDPO.

    python scripts/rft/build_eval_manifest.py --out runs/rft_v4_a100/eval/robust_eval_manifest.json

Writes a fixed-evaluation manifest (``fixed_eval.load_or_create_manifest``
accepts it when the GDPO config states the same protocol) and a sidecar
``<manifest>.reference.json``, keyed by example id, with what the analysis
needs and the trainer must not see as dataset columns: difficulty, family,
whether training holds that family, where the fault sits, the random vector
reference and the decisive inputs of the dataset's label.

The whole split is considered through its key columns, and full rows are read
only for the problems the selection reaches. The test split is left for final
claims.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "eval"))

LEVELS = ("easy", "medium", "hard", "very_hard", "random_never_detects")
EXAMPLE_KEYS = ("prompt", "fault", "netlist", "module_name", "doc_id")


def train_families(dataset):
    """Generator families of every module in the train split."""
    import pyarrow.parquet as pq
    from atpgllm.training.fixed_eval import family_of
    modules = set()
    for path in sorted((Path(dataset) / "train").glob("*.parquet")):
        modules.update(pq.read_table(path, columns=["module_name"])["module_name"].unique().to_pylist())
    return {family_of(m) for m in modules}


def split_keys(dataset, split, seed):
    """One seeded row per problem of the split, as key columns and its parquet position."""
    import pyarrow.parquet as pq
    rng = random.Random(seed)
    rows = {}
    seen = Counter()
    columns = ["difficulty", "module_name", "netlist_id", "circuit_id", "fault", "random_detection_probability"]
    for index, path in enumerate(sorted((Path(dataset) / split).glob("*.parquet"))):
        for offset, row in enumerate(pq.read_table(path, columns=columns).to_pylist()):
            key = (str(row["netlist_id"]), str(row["fault"]))
            seen[key] += 1
            # Reservoir of one: every row of a problem is equally likely to supply it.
            if rng.randrange(seen[key]) == 0:
                rows[key] = dict(row, _file=path, _offset=offset)
    return list(rows.values())


def read_row(row, cache={}):
    import pyarrow.parquet as pq
    path = row["_file"]
    if path not in cache:
        cache.clear()
        cache[path] = pq.read_table(path).to_pylist()
    return dict(cache[path][row["_offset"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset", default="../data/freeset/dataset.freeset.asap7sc7p5t_28.rvt.tt.v4")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--per_level", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--tokenizer", default="ibm-granite/granite-4.2-8b")
    # The protocol of the GDPO config that will reuse this manifest.
    parser.add_argument("--max_prompt_length", type=int, default=8192)
    parser.add_argument("--max_completion_length", type=int, default=24576)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--backend", default="tetramax", help="FAULT_SIM_BACKEND of the GDPO config")
    parser.add_argument("--analyzer_samples", type=int, default=65536)
    args = parser.parse_args()

    os.environ["FAULT_SIM_BACKEND"] = args.backend
    from transformers import AutoTokenizer
    from atpgllm.training._paths import ensure_data_preprocessing_on_path
    from atpgllm.training.dataset_utils import _gate_filter_passes, chat_template_uses_tools, format_grpo_record
    from atpgllm.training.eval_report import Simulator, reference_entry
    from atpgllm.training.fixed_eval import circuit_ids, digest, fixed_eval_protocol, select_stratified, write_json_atomic
    ensure_data_preprocessing_on_path()
    from fault_difficulty import Analyzer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    use_tools = chat_template_uses_tools(tokenizer)
    seen_families = train_families(args.dataset)
    keys = split_keys(args.dataset, args.split, args.seed)
    print(f"[manifest] {len(keys)} problems in {args.split}, {len(seen_families)} train families", flush=True)

    rendered, rejected = {}, Counter()

    def eligible(key_row):
        raw = read_row(key_row)
        if not _gate_filter_passes(raw):
            rejected["gate_filter"] += 1
            return False
        record = format_grpo_record(dict(raw), tokenizer, use_tools)
        length = len(tokenizer(record["prompt"], add_special_tokens=False)["input_ids"])
        if length >= args.max_prompt_length:
            rejected["prompt_length"] += 1
            return False
        rendered[(str(key_row["netlist_id"]), str(key_row["fault"]))] = (raw, record)
        return True

    chosen = select_stratified(keys, {level: args.per_level for level in LEVELS}, args.seed,
                               seen_families=seen_families, eligible=eligible)
    print(f"[manifest] selected {len(chosen)}, rejected {dict(rejected)}", flush=True)

    # The reference always uses the Python simulator, whatever the GDPO config scores with.
    os.environ["FAULT_SIM_BACKEND"] = "fast"
    simulator = Simulator()
    analyzer = Analyzer(config=Path(simulator.fs.__file__).parent / "sim_config.json",
                        samples=args.analyzer_samples, exact_max_inputs=16, seed=1729)
    examples, reference = [], {}
    for key_row in chosen:
        raw, record = rendered[(str(key_row["netlist_id"]), str(key_row["fault"]))]
        net_id = next(k for k in circuit_ids(record) if k.startswith("net:"))
        example_id = digest([net_id, record["fault"]])
        examples.append({**{k: record[k] for k in EXAMPLE_KEYS if k in record}, "_fixed_eval_id": example_id})
        reference[example_id] = reference_entry(raw, simulator, seen_families=seen_families, analyzer=analyzer)

    size = len(examples)
    protocol = fixed_eval_protocol(
        dataset=args.dataset, split=args.split, size=size, seed=args.seed, tokenizer=tokenizer,
        max_prompt_length=args.max_prompt_length, max_completion_length=args.max_completion_length,
        max_model_len=args.max_model_len, generations=1, per_device_batch_size=args.per_device_batch_size,
        world_size=args.world_size, holdout_policy="exclude_evaluation_circuits", rolling_buffer=True,
        selection="stratified_family_v2")
    protocol["fault_sim_backend"] = args.backend
    manifest = {"version": 1, "protocol": protocol, "examples": examples, "examples_sha256": digest(examples)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out, manifest)
    write_json_atomic(str(out) + ".reference.json", reference)

    for level in LEVELS:
        group = [r for r in reference.values() if r["level"] == level]
        families = Counter(r["family"] for r in group)
        print(f"[manifest] {level}: {len(group)} problems, {len({r['netlist_id'] for r in group})} netlists, "
              f"{len(families)} families, {sum(r['family_in_train'] for r in group)} from seen families, "
              f"largest family {families.most_common(1)[0]}, "
              f"label detects {sum(r['label_detects'] for r in group)}, "
              f"mean p {sum(r['p'] for r in group) / len(group):.4f}, "
              f"sites {dict(Counter(r['site'] for r in group))}", flush=True)
    print(f"[manifest] sha256 {manifest['examples_sha256']} -> {out}")


if __name__ == "__main__":
    main()
