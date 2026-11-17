"""Compact the simulator search into the graph model: targets found by search.

The graph model is trained on frequencies over all the detecting patterns of
a fault (how often each input is set, and to what). The vector decoded from
those averages is not always a pattern that detects: the vote of the
dataset's own patterns detects 82.9% of the held-out rows, and the model's
first attempt 73.1%. The scripted search (``hint_arms_eval.decoder_attempts``:
the first pattern, then one changed decision per attempt, least sure first)
finds a detecting pattern for 92.9% within 16 attempts.

This script runs that search on training problems and writes label tables in
which a problem's targets are the pattern the search found: one concrete
pattern that detects, and the one closest to what the model already says.
Training on them (``train_retest --init <model> --labels <out>``) asks the
model to give that pattern on its first attempt.

    search  one shard of the hint files: simulate the attempts of each
            problem until one detects, write the problem row with what was found
    merge   the shard files into a label directory for ``train_retest``

    PYTHONPATH=$PWD python scripts/eval/search_labels.py search --hints runs/bridges_retest_seq/hints \\
        --labels ../data/freeset/retest_labels_v5 --out runs/search_labels/r1 --shard 0 --limit 4000
    PYTHONPATH=$PWD python scripts/eval/search_labels.py merge --labels ../data/freeset/retest_labels_v5 \\
        --out runs/search_labels/r1 --blend 1.0 --into runs/search_labels/r1_hard
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import hint_arms_eval as ev  # noqa: E402  (also puts scripts/train on the path)

r6 = ev.r6

_ROW_HEAD = re.compile(r'\{"netlist_key": "([0-9a-f]+)", .{0,400}?"fault": "((?:[^"\\]|\\.)*)"')


def wanted_problems(labels: Path, split: str, wanted: Dict[tuple, dict]) -> Iterator[dict]:
    """The label rows of the wanted problems, in file order. Only those rows
    are parsed: the train table is several GB, and every row starts with its
    netlist key and fault."""
    with (labels / f"problems_{split}.jsonl").open() as fh:
        for line in fh:
            match = _ROW_HEAD.match(line)
            if match is not None and (match.group(1), json.loads(f'"{match.group(2)}"')) not in wanted:
                continue
            row = json.loads(line)
            if (row["netlist_key"], row["fault"]) in wanted:
                yield row


def search(record: Dict[str, Any], rich: Dict[str, Any], inputs: List[str], attempts: int,
           simulate=None) -> Dict[str, Any]:
    """Try the decoder's attempts in order until one detects.
    ``found`` is the index of the detecting attempt (0 is the first pattern),
    or -1. ``vector`` holds the inputs that pattern sets."""
    simulate = simulate or (lambda vector: r6._simulate(record, vector)[0])
    vectors = ev.decoder_attempts(rich, inputs, attempts)
    for index, vector in enumerate(vectors):
        if simulate(vector):
            return {"found": index, "tried": index + 1, "candidates": len(vectors),
                    "vector": {name: value for name, value in vector.items() if value != "x"}}
    return {"found": -1, "tried": len(vectors), "candidates": len(vectors), "vector": None}


def run_search(args) -> None:
    hints: Dict[tuple, dict] = {}
    path = args.hints / f"{args.split}_{args.shard:03d}.jsonl"
    with path.open() as fh:
        for line in fh:
            row = json.loads(line)
            hints[(row["netlist_key"], row["fault"])] = row["rich"]
            if args.limit and len(hints) >= args.limit:
                break
    circuits = {}
    with (args.labels / f"circuits_{args.split}.jsonl").open() as fh:
        for line in fh:
            circuit = json.loads(line)
            circuits[circuit["netlist_key"]] = circuit
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"rows.s{args.shard:03d}.jsonl"
    done = r6._done(out_path, lambda r: (r["netlist_key"], r["fault"]))
    counts, started = collections.Counter(), time.time()
    with out_path.open("a") as fh:
        for problem in wanted_problems(args.labels, args.split, hints):
            key = (problem["netlist_key"], problem["fault"])
            if key in done:
                continue
            circuit = circuits[problem["netlist_key"]]
            record = {"netlist": circuit["netlist"], "fault": problem["fault"],
                      "input_vector": problem["example_input_vector"],
                      "expected_output": problem["example_expected_output"],
                      "scan_chain": circuit.get("scan_chain") or ""}
            try:
                result = search(record, hints[key], list(problem["inputs"]), args.attempts)
            except Exception as exc:      # a circuit the simulator cannot take
                result = {"found": -1, "tried": 0, "candidates": 0, "vector": None,
                          "error": f"{type(exc).__name__}: {exc}"[:200]}
            counts["problems"] += 1
            counts["first"] += result["found"] == 0
            counts["later"] += result["found"] > 0
            fh.write(json.dumps({**problem, "search": result}) + "\n")
            fh.flush()
            if counts["problems"] % 200 == 0:
                print(f"shard {args.shard}: {counts['problems']} problems, first {counts['first']}, "
                      f"later {counts['later']}, {time.time() - started:.0f}s", flush=True)
    print(f"shard {args.shard} done: {dict(counts)} in {time.time() - started:.0f}s", flush=True)


def relabel(row: Dict[str, Any], blend: float) -> Optional[Dict[str, Any]]:
    """The label row with the found pattern as its target, or None when the
    search found none. ``blend`` is the weight of the found pattern, the rest
    stays on the dataset's frequencies (1.0: the found pattern alone)."""
    found = row["search"]
    if found["found"] < 0 or found["vector"] is None:
        return None
    vector = found["vector"]
    care, one = {}, {}
    for name in row["inputs"]:
        old_care = float(row["care_freq"].get(name, 0.0))
        old_one = float(row["one_freq"].get(name, 0.5))
        care[name] = blend * float(name in vector) + (1.0 - blend) * old_care
        # An input the pattern leaves unknown has no value of its own.
        one[name] = blend * float(vector[name]) + (1.0 - blend) * old_one if name in vector else old_one
    out = {key: value for key, value in row.items() if key != "search"}
    out.update(care_freq=care, one_freq=one, search_found=found["found"])
    return out


def run_merge(args) -> None:
    args.into.mkdir(parents=True, exist_ok=True)
    counts, keys = collections.Counter(), set()
    with (args.into / f"problems_{args.split}.jsonl").open("w") as out:
        for path in sorted(args.out.glob("rows.s*.jsonl")):
            for line in path.open():
                row = json.loads(line)
                counts["searched"] += 1
                counts["first"] += row["search"]["found"] == 0
                counts["later"] += row["search"]["found"] > 0
                if args.only_later and row["search"]["found"] == 0:
                    continue
                new = relabel(row, args.blend)
                if new is None:
                    continue
                out.write(json.dumps(new) + "\n")
                keys.add(new["netlist_key"])
                counts["written"] += 1
    with (args.labels / f"circuits_{args.split}.jsonl").open() as fh, \
            (args.into / f"circuits_{args.split}.jsonl").open("w") as out:
        for line in fh:
            if json.loads(line)["netlist_key"] in keys:
                out.write(line)
    # The other split is read as it is (train_retest evaluates on it).
    for name in (f"circuits_{args.eval_split}.jsonl", f"problems_{args.eval_split}.jsonl"):
        link = args.into / name
        if not link.exists():
            os.symlink((args.labels / name).resolve(), link)
    summary = {**counts, "circuits": len(keys), "blend": args.blend, "only_later": args.only_later}
    (args.into / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("pass_", metavar="pass", choices=("search", "merge"))
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--hints", type=Path, help="search: hint files with the model's probabilities (rich).")
    parser.add_argument("--out", type=Path, required=True, help="Folder of the searched shard files.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--shard", type=int, default=0, help="search: which hint file of the split.")
    parser.add_argument("--limit", type=int, default=0, help="search: first N problems of the hint file.")
    parser.add_argument("--attempts", type=int, default=16)
    parser.add_argument("--into", type=Path, help="merge: the label directory to write.")
    parser.add_argument("--blend", type=float, default=1.0,
                        help="merge: weight of the found pattern in the targets (1.0: the pattern alone).")
    parser.add_argument("--only-later", action="store_true",
                        help="merge: keep only problems whose first attempt missed.")
    args = parser.parse_args()
    {"search": run_search, "merge": run_merge}[args.pass_](args)


if __name__ == "__main__":
    main()
