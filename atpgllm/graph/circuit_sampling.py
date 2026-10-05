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
    group_by: Sequence[str] = ("circuit_id",),
) -> List[Tuple[int, int]]:
    """``(file index, row index)`` picks in a random order: at most
    ``per_circuit`` random rows per group of ``group_by`` values (0 = all),
    at most ``total``. Grouping by ``("circuit_id", "difficulty")`` keeps the
    rare difficulty labels of every circuit."""
    groups: Dict[Tuple[Any, ...], List[Tuple[int, int]]] = collections.defaultdict(list)
    for file_index, path in enumerate(files):
        table = pq.read_table(path, columns=list(group_by))
        keys = zip(*(table.column(name).to_pylist() for name in group_by))
        for row, key in enumerate(keys):
            groups[key].append((file_index, row))
    rng = random.Random(seed)
    picks: List[Tuple[int, int]] = []
    for key in sorted(groups, key=lambda k: tuple(map(str, k))):
        rows = groups[key]
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
    group_by: Sequence[str] = ("circuit_id",),
    columns: Optional[Sequence[str]] = GRAPH_COLUMNS,
) -> Iterator[Dict[str, Any]]:
    """The sampled rows; with ``repeat``, again in a new order per epoch.
    ``columns=None`` loads every column (captions fill placeholders from any
    of them). Rows are loaded when this is called, so DataLoader workers
    forked afterwards share them instead of each loading its own copy."""
    files = split_files(dataset, split, revision)
    if columns is None:
        columns = pq.read_schema(files[0]).names
    records = load_rows(
        files,
        choose_rows(
            files, per_circuit=per_circuit, total=total, seed=seed, group_by=group_by
        ),
        columns=columns,
    )
    print(f"{dataset}:{split} sampled {len(records)} rows", flush=True)
    return _epochs(records, repeat=repeat, seed=seed)


def _epochs(
    records: List[Dict[str, Any]], *, repeat: bool, seed: int
) -> Iterator[Dict[str, Any]]:
    epoch = 0
    while True:
        yield from records
        if not repeat:
            return
        epoch += 1
        random.Random(seed + epoch).shuffle(records)
