"""Circuit-balanced row sampling straight from a Hub dataset's parquet files.

Some datasets (asap7-language-of-test-v3) store rows grouped by circuit, with
one circuit holding a third of the training rows. A streaming shuffle buffer
cannot mix that, and an Arrow cache of the whole split is tens of GB on
network storage. Instead: read only ``circuit_id`` from the cached parquet
files, choose rows at random (at most ``per_circuit`` per circuit, chosen at
random within it), then load just those rows and the columns the graph code
uses, keeping one copy of each circuit's netlist.
"""

from __future__ import annotations

import collections
import glob
import random
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import pyarrow.parquet as pq

GRAPH_COLUMNS: Tuple[str, ...] = (
    "netlist",
    "fault",
    "fault_propagation_gates",
    "backtrack_gates",
    "snapshot",
    "difficulty",
    "circuit_id",
    # Read by the vector_port_mismatch check, not by the encoder.
    "input_vector",
)


def split_files(dataset: str, split: str, revision: Optional[str]) -> List[str]:
    """Local parquet files of ``split`` (downloaded once into the HF cache)."""
    from huggingface_hub import snapshot_download

    root = snapshot_download(
        dataset,
        repo_type="dataset",
        revision=revision,
        allow_patterns=[f"data/{split}-*.parquet"],
    )
    files = sorted(glob.glob(f"{root}/data/{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files for {dataset}:{split} at {root}.")
    return files


def choose_rows(
    files: Sequence[str],
    *,
    per_circuit: int,
    total: Optional[int],
    seed: int,
) -> List[Tuple[int, int]]:
    """``(file index, row index)`` picks in a random order: at most
    ``per_circuit`` random rows per circuit (0 = all), at most ``total``."""
    by_circuit: Dict[str, List[Tuple[int, int]]] = collections.defaultdict(list)
    for file_index, path in enumerate(files):
        ids = pq.read_table(path, columns=["circuit_id"]).column("circuit_id").to_pylist()
        for row, circuit in enumerate(ids):
            by_circuit[circuit].append((file_index, row))
    rng = random.Random(seed)
    picks: List[Tuple[int, int]] = []
    for circuit in sorted(by_circuit):
        rows = by_circuit[circuit]
        picks.extend(rng.sample(rows, per_circuit) if 0 < per_circuit < len(rows) else rows)
    rng.shuffle(picks)
    return picks[:total] if total else picks


def load_rows(
    files: Sequence[str],
    picks: Sequence[Tuple[int, int]],
    columns: Sequence[str] = GRAPH_COLUMNS,
) -> List[Dict[str, Any]]:
    """Rows for ``picks``, in pick order, reading one file at a time."""
    wanted: Dict[int, List[int]] = collections.defaultdict(list)
    for file_index, row in picks:
        wanted[file_index].append(row)
    loaded: Dict[Tuple[int, int], Dict[str, Any]] = {}
    netlists: Dict[str, str] = {}
    for file_index, rows in wanted.items():
        table = pq.read_table(files[file_index], columns=list(columns)).take(rows)
        for row, record in zip(rows, table.to_pylist()):
            if "netlist" in record:
                record["netlist"] = netlists.setdefault(record["circuit_id"], record["netlist"])
            loaded[(file_index, row)] = record
    return [loaded[pick] for pick in picks]


def sampled_records(
    dataset: str,
    split: str,
    *,
    revision: Optional[str],
    per_circuit: int,
    total: Optional[int],
    seed: int,
    repeat: bool = False,
) -> Iterator[Dict[str, Any]]:
    """Yield the sampled rows; with ``repeat``, again in a new order per epoch."""
    files = split_files(dataset, split, revision)
    records = load_rows(
        files, choose_rows(files, per_circuit=per_circuit, total=total, seed=seed)
    )
    print(f"{dataset}:{split} sampled {len(records)} rows", flush=True)
    epoch = 0
    while True:
        yield from records
        if not repeat:
            return
        epoch += 1
        random.Random(seed + epoch).shuffle(records)
