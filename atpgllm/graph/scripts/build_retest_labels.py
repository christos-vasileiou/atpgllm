"""Build the BRIDGES re-test label table from the v4 parquet shards.

Writes, per split, ``problems_<split>.jsonl`` (one line per (netlist, fault)
with soft targets and port-only fact text) and ``circuits_<split>.jsonl``
(netlist text once, keyed by the sha256 of the text). Reads only the columns it needs.

Usage::

    python -m atpgllm.graph.scripts.build_retest_labels \
        --data ../data/freeset/v4_hf_verify/repo/data \
        --out ../data/freeset/retest_labels_v4 --splits validation train
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import time
from collections import OrderedDict
from typing import Dict, List

import pyarrow.parquet as pq

from atpgllm.graph.retest_labels import fact_text, problem_targets

COLUMNS = [
    "circuit_id", "netlist_id", "module_name", "fault", "difficulty", "netlist",
    "input_vector", "expected_output", "fault_propagation_gates", "fault_propagation_nets",
]


def _netlist(value):
    if isinstance(value, dict):
        return value.get("netlist") or ""
    return value or ""


def build_split(data_dir: str, split: str, out_dir: str) -> Dict[str, int]:
    shards = sorted(glob.glob(os.path.join(data_dir, f"{split}-*.parquet")))
    if not shards:
        raise FileNotFoundError(f"no {split} shards under {data_dir}")
    problems: "OrderedDict[tuple, List[dict]]" = OrderedDict()
    circuits: Dict[str, dict] = {}
    rows_seen = 0
    for shard in shards:
        table = pq.read_table(shard, columns=COLUMNS)
        for row in table.to_pylist():
            rows_seen += 1
            text = _netlist(row.get("netlist"))
            # netlist_id is not unique per netlist text in v4, so key on the text.
            nid = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
            if nid not in circuits:
                circuits[nid] = {
                    "netlist_key": nid,
                    "circuit_id": row.get("circuit_id"),
                    "netlist_id": row.get("netlist_id"),
                    "module_name": row.get("module_name"),
                    "netlist": text,
                }
            row.pop("netlist", None)
            problems.setdefault((nid, str(row["fault"]).strip()), []).append(row)
    os.makedirs(out_dir, exist_ok=True)
    written = 0
    with open(os.path.join(out_dir, f"problems_{split}.jsonl"), "w", encoding="utf-8") as f:
        for (nid, fault), rows in problems.items():
            t = problem_targets(rows)
            if not t.patterns:
                continue
            difficulty = rows[0].get("difficulty")
            f.write(json.dumps({
                "netlist_key": nid,
                "netlist_id": rows[0].get("netlist_id"),
                "circuit_id": rows[0].get("circuit_id"),
                "fault": fault,
                "difficulty": difficulty,
                "patterns": t.patterns,
                "inputs": t.inputs,
                "outputs": t.outputs,
                "care_freq": t.care_freq,
                "one_freq": t.one_freq,
                "path_freq": t.path_freq,
                "obs_freq": t.obs_freq,
                "fact_text": fact_text(t, difficulty=difficulty),
                "example_input_vector": rows[0]["input_vector"],
                "example_expected_output": rows[0]["expected_output"],
            }) + "\n")
            written += 1
    with open(os.path.join(out_dir, f"circuits_{split}.jsonl"), "w", encoding="utf-8") as f:
        for c in circuits.values():
            f.write(json.dumps(c) + "\n")
    return {"rows": rows_seen, "problems": written, "circuits": len(circuits)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["validation", "test", "train"])
    args = ap.parse_args()
    summary = {}
    for split in args.splits:
        start = time.time()
        summary[split] = build_split(args.data, split, args.out)
        summary[split]["seconds"] = round(time.time() - start, 1)
        print(split, summary[split], flush=True)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
