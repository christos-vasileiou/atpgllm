"""Circuit kinds (dataset v5: combinational and full-scan sequential) in training order and selection.

v5 rows carry ``circuit_kind``. Training must see both kinds in every optimizer
step, so wherever rows are ordered (the SFT stream, each GRPO chunk) the kinds
are merged by even spacing: kind ``k`` with ``n_k`` of ``N`` rows takes the
fractional positions ``(j + 0.5) / n_k``. Every window of ``L`` consecutive rows
then holds ``L * n_k / N`` rows of kind ``k``, give or take one, so a window of
``L >= 2 * N / min(n_k)`` rows holds both kinds. The order inside a kind is kept.
Datasets with one kind (v4) are left exactly as they were.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

KINDS = ("combinational", "sequential")
DEFAULT_KIND = "combinational"

# The prompt window for v5 (dataset card, Granite 4.2 tokenizer, system + user with
# chat template): share of problems whose prompt exceeds the window and is skipped.
V5_PROMPT_WINDOW = 14336
V5_OVER_WINDOW = {"combinational": 0.029, "sequential": 0.068}


def row_kind(row: dict) -> str:
    return row.get("circuit_kind") or DEFAULT_KIND


def even_spacing_order(kinds: Sequence[int]):
    """Positions of ``kinds`` (one small int per item) merged by even spacing, order kept per kind."""
    import numpy as np
    kinds = np.asarray(kinds)
    if kinds.size == 0:
        return np.zeros(0, dtype=np.int64)
    keys = np.empty(kinds.size, dtype=np.float64)
    for k in np.unique(kinds):
        where = np.flatnonzero(kinds == k)
        keys[where] = (np.arange(where.size) + 0.5) / where.size
    # lexsort: last key is primary; ties (equal fractions) go to the lower kind first.
    return np.lexsort((kinds, keys))


def interleave_by_kind(rows: list, kind_of: Callable[[dict], str] = row_kind) -> list:
    """``rows`` with the kinds evenly spaced and each kind's own order kept."""
    names = [kind_of(r) for r in rows]
    if len(set(names)) < 2:
        return rows
    codes = {name: i for i, name in enumerate(sorted(set(names)))}
    return [rows[i] for i in even_spacing_order([codes[n] for n in names])]


def parse_kind_shares(spec: Optional[str]) -> Optional[dict]:
    """``"combinational=0.5,sequential=0.5"`` -> shares that sum to 1 (None when empty)."""
    if not spec:
        return None
    shares = {}
    for part in spec.split(","):
        if part.strip():
            kind, value = part.split("=")
            if kind.strip() not in KINDS:
                raise ValueError(f"Unknown circuit kind {kind!r}; expected one of {KINDS}")
            shares[kind.strip()] = float(value)
    total = sum(shares.values())
    if total <= 0:
        raise ValueError(f"Kind shares must be positive: {spec}")
    return {k: v / total for k, v in shares.items()}


def split_counts(counts: dict, shares: Optional[dict]) -> dict:
    """Per-level counts split across kinds: ``{(kind, level): n}``; rounding keeps each level's total."""
    if not shares:
        return {(None, level): n for level, n in counts.items()}
    out = {}
    kinds = list(shares)
    for level, n in counts.items():
        given = 0
        for i, kind in enumerate(kinds):
            share = n - given if i == len(kinds) - 1 else int(round(n * shares[kind]))
            out[(kind, level)] = share
            given += share
    return out


def dataset_kinds(directory) -> dict:
    """``{kind: train rows}`` from the split manifest (``kind_row_counts``), or {} when absent."""
    manifest = Path(directory) / "split_manifest.json"
    if not manifest.is_file():
        return {}
    counts = json.loads(manifest.read_text()).get("kind_row_counts", {})
    return dict(counts.get("train", {}))


def prompt_window_notice(directory, max_prompt_length: int) -> tuple[int, Optional[str]]:
    """The prompt window to use for ``directory`` and a line to print, or (unchanged, None).

    A v5 dataset needs at least :data:`V5_PROMPT_WINDOW` tokens: below that the
    filter drops a quarter of the sequential problems (27% exceed 8,192).
    """
    manifest = Path(directory) / "split_manifest.json"
    if not manifest.is_file() or json.loads(manifest.read_text()).get("dataset_version") != "v5":
        return max_prompt_length, None
    over = ", ".join(f"{V5_OVER_WINDOW[k]:.1%} of {k}" for k in KINDS)
    if max_prompt_length < V5_PROMPT_WINDOW:
        return V5_PROMPT_WINDOW, (
            f"[v5] Changing the prompt window from {max_prompt_length} to {V5_PROMPT_WINDOW} tokens: "
            f"v5 sequential prompts carry the scan netlist and chain table. {over} problems still "
            f"exceed {V5_PROMPT_WINDOW} and are skipped. Set MAX_PROMPT_LENGTH={V5_PROMPT_WINDOW} in the config.")
    return max_prompt_length, (
        f"[v5] Prompt window {max_prompt_length} tokens ({over} problems exceed {V5_PROMPT_WINDOW} tokens).")


def shard_kinds(paths: Iterable[Path]) -> list:
    """The ``circuit_kind`` of each parquet shard (one kind per shard), from footer statistics.

    Falls back to reading the column when a footer has no statistics; None when
    the shards have no ``circuit_kind`` column. A shard mixing kinds is refused.
    """
    import pyarrow.parquet as pq
    kinds = []
    for path in paths:
        pf = pq.ParquetFile(path)
        names = pf.schema_arrow.names
        if "circuit_kind" not in names:
            return None
        column = names.index("circuit_kind")
        seen = set()
        for g in range(pf.metadata.num_row_groups):
            stats = pf.metadata.row_group(g).column(column).statistics
            if stats is not None and stats.has_min_max:
                seen |= {stats.min, stats.max}
            else:
                seen |= set(pf.read_row_group(g, columns=["circuit_kind"]).column(0).to_pylist())
        seen = {s.decode() if isinstance(s, bytes) else s for s in seen if s is not None}
        if len(seen) > 1:
            raise ValueError(f"{path} mixes circuit kinds {sorted(seen)}")
        kinds.append(seen.pop() if seen else DEFAULT_KIND)
    return kinds
