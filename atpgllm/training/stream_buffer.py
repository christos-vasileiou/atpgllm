"""Seeded, resumable reading of a local split across all of its circuits.

The parquet shards of a dataset split are written circuit by circuit, so
reading them in file order trains on one circuit after another.
:class:`ShardPermutation` instead visits every row of the split in a seeded
random order (one permutation per pass over the split), reading a block of
positions at a time and only the shards that block touches.

:class:`RollingPromptBuffer` turns that order into the map-style dataset
GRPOTrainer needs. It holds ``chunk_size`` prompts at a time and loads the
next chunk when training reaches it. A ledger records where each chunk starts
in the permutation and a digest of its rows, so an interrupted run rebuilds
the exact chunk it stopped in and continues from there.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np

LEDGER_NAME = "grpo_stream_ledger.json"


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


class ShardPermutation:
    """Every row of ``<directory>/<split>/*.parquet``, in a seeded order per pass.

    Row counts come from ``split_manifest.json`` when it lists the shards, so
    no shard is opened to build the order; otherwise from parquet metadata.
    """

    def __init__(self, directory, split: str = "train", seed: int = 0, block_size: int = 20000):
        import pyarrow.parquet as pq

        self.directory = Path(directory)
        self.split = split
        self.seed = int(seed)
        self.block_size = int(block_size)
        manifest_path = self.directory / "split_manifest.json"
        shards = []
        if manifest_path.is_file():
            listed = json.loads(manifest_path.read_text()).get("shards", {})
            shards = sorted((name, int(info["rows"])) for name, info in listed.items()
                            if name.startswith(split + "/"))
        if not shards:
            shards = [(str(path.relative_to(self.directory)), pq.ParquetFile(path).metadata.num_rows)
                      for path in sorted((self.directory / split).glob("*.parquet"))]
        if not shards:
            raise ValueError(f"No parquet shards for split {split!r} under {self.directory}")
        self.paths = [self.directory / name for name, _ in shards]
        counts = np.array([rows for _, rows in shards], dtype=np.int64)
        self.starts = np.concatenate([[0], np.cumsum(counts)])
        self.total = int(self.starts[-1])
        self._order = (None, None)

    def order(self, pass_index: int) -> np.ndarray:
        if self._order[0] != pass_index:
            rng = np.random.default_rng([self.seed, int(pass_index)])
            self._order = (pass_index, rng.permutation(self.total))
        return self._order[1]

    def rows(self, pass_index: int, start: int, stop: int) -> Iterator[tuple[int, dict]]:
        """``(position, row)`` for permutation positions ``start <= p < stop`` of one pass."""
        import pyarrow.parquet as pq

        order = self.order(pass_index)
        for block_start in range(start, min(stop, self.total), self.block_size):
            block_stop = min(block_start + self.block_size, stop, self.total)
            ids = order[block_start:block_stop]
            shard = np.searchsorted(self.starts, ids, side="right") - 1
            local = ids - self.starts[shard]
            rows: dict[int, dict] = {}
            for s in np.unique(shard):
                mask = shard == s
                table = pq.read_table(self.paths[s]).take(local[mask])
                for gid, row in zip(ids[mask].tolist(), table.to_pylist()):
                    rows[gid] = row
            for offset, gid in enumerate(ids.tolist()):
                yield block_start + offset, rows[gid]

    def stream(self, pass_index: int = 0) -> Iterator[dict]:
        """Rows forever: pass ``pass_index`` from its start, then the next pass."""
        while True:
            for _, row in self.rows(pass_index, 0, self.total):
                yield row
            pass_index += 1


def streaming_split(directory, split: str = "train", seed: int = 0):
    """The split as a HuggingFace ``IterableDataset`` in :class:`ShardPermutation` order.

    The order is a pure function of the shards and ``seed``, so counting
    consumed rows (``AUTO_SKIP_FROM_RESUME``) resumes at the same place.
    """
    from datasets import IterableDataset

    def generate(directory, split, seed):
        yield from ShardPermutation(directory, split, seed).stream()

    return IterableDataset.from_generator(
        generate, gen_kwargs={"directory": str(directory), "split": split, "seed": seed})


class RollingPromptBuffer:
    """A map-style GRPO dataset that holds one chunk of prompts at a time.

    Index ``i`` lies in chunk ``i // chunk_size``. GRPOTrainer reads indices in
    order (``shuffle_dataset=False``), so chunk ``k + 1`` is built from the
    permutation, continuing where chunk ``k`` ended, when training first
    reaches it, and the previous chunk is kept for indices that straddle the
    boundary. ``prepare`` turns a block of raw rows into training rows (format,
    filters, holdout), returning ``None`` for each row it drops; prompts
    repeated within a chunk are dropped too. Each chunk is reordered for netlist diversity.

    The ledger (``<output_dir>/grpo_stream_ledger.json``, also copied into
    every checkpoint) records each chunk's start and end positions and the
    digest of its rows. On resume the chunk the run stopped in is rebuilt from
    its recorded start; a different digest means the data or the filters
    changed, and the run stops instead of training on other rows.
    """

    def __init__(self, source: ShardPermutation, prepare: Callable[[list], list], *,
                 chunk_size: int, length: int, ledger_path, settings: dict,
                 reorder: Optional[Callable[[list, int], list]] = None,
                 unique_by: str = "prompt", writer: bool = True, resume_ledger=None,
                 read_block: int = 4096):
        if chunk_size < 1 or length < chunk_size:
            raise ValueError("RollingPromptBuffer needs chunk_size >= 1 and length >= chunk_size")
        self.source = source
        self.prepare = prepare
        self.chunk_size = int(chunk_size)
        self.length = int(length)
        self.ledger_path = Path(ledger_path)
        self.reorder = reorder
        self.unique_by = unique_by
        self.writer = writer
        self.read_block = int(read_block)
        self.settings = dict(settings, chunk_size=self.chunk_size, seed=source.seed,
                             rows_in_split=source.total)
        self.ledger = {"settings": self.settings, "chunks": {}}
        self._cache: OrderedDict[int, list] = OrderedDict()
        for path in (resume_ledger, self.ledger_path):
            if path and Path(path).is_file():
                self._load(Path(path))
                break

    def _load(self, path: Path):
        ledger = json.loads(path.read_text())
        if ledger.get("settings") != self.settings:
            changed = sorted(k for k in set(ledger.get("settings", {})) | set(self.settings)
                             if ledger.get("settings", {}).get(k) != self.settings.get(k))
            raise ValueError(
                f"GRPO stream ledger {path} was written with different data settings "
                f"({', '.join(changed)}). Resuming would train on different prompts than the "
                "checkpoint saw: restore those settings or start a new OUTPUT_DIR.")
        self.ledger = ledger
        print(f"[GRPO buffer] Resuming with {len(ledger['chunks'])} recorded chunk(s) from {path}")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if not 0 <= index < self.length:
            raise IndexError(index)
        chunk, offset = divmod(int(index), self.chunk_size)
        return self.chunk(chunk)[offset]

    def chunk(self, k: int) -> list:
        if k not in self._cache:
            self._cache[k] = self._build(k)
            while len(self._cache) > 2:
                self._cache.popitem(last=False)
        return self._cache[k]

    def _start(self, k: int):
        recorded = self.ledger["chunks"].get(str(k))
        if recorded:
            return tuple(recorded["start"])
        if k == 0:
            return (0, 0)
        previous = self.ledger["chunks"].get(str(k - 1))
        if previous is None:
            self.chunk(k - 1)
            previous = self.ledger["chunks"][str(k - 1)]
        return tuple(previous["end"])

    def _build(self, k: int) -> list:
        pass_index, position = start = self._start(k)
        rows, seen = [], set()
        while len(rows) < self.chunk_size:
            if position >= self.source.total:
                if pass_index > start[0] and not rows:
                    raise ValueError("A full pass over the split produced no training prompt: "
                                     "every row was filtered out (prompt length, gate filter or holdout)")
                pass_index, position = pass_index + 1, 0
                print(f"[GRPO buffer] Split exhausted; starting pass {pass_index} in a new order")
            block = list(self.source.rows(pass_index, position, position + self.read_block))
            prepared = self.prepare([row for _, row in block])
            if len(prepared) != len(block):
                raise ValueError("prepare must return one entry (or None) per raw row")
            position = block[-1][0] + 1
            for (p, _), row in zip(block, prepared):
                if row is None:
                    continue
                key = _digest(row.get(self.unique_by))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
                if len(rows) == self.chunk_size:
                    position = p + 1
                    break
        if self.reorder is not None:
            rows = self.reorder(rows, k)
        entry = {"start": list(start), "end": [pass_index, position], "rows": len(rows),
                 "rows_sha256": _digest([[r.get("prompt"), r.get("fault")] for r in rows]),
                 "circuits": len({str(r.get("circuit_id") or r.get("module_name")) for r in rows})}
        recorded = self.ledger["chunks"].get(str(k))
        if recorded and recorded["rows_sha256"] != entry["rows_sha256"]:
            raise ValueError(
                f"GRPO buffer chunk {k} differs from the one this run trained on "
                f"(recorded {recorded['rows_sha256'][:12]}, rebuilt {entry['rows_sha256'][:12]}). "
                "The dataset, tokenizer, prompt filters or evaluation holdout changed since the "
                "checkpoint; restore them or start a new OUTPUT_DIR.")
        self.ledger["chunks"][str(k)] = entry
        if self.writer:
            self.save()
        print(f"[GRPO buffer] chunk {k}: {len(rows)} prompts from pass {start[0]} positions "
              f"[{start[1]}, {position}) of {self.source.total}, {entry['circuits']} circuits", flush=True)
        return rows

    def save(self, path=None):
        path = Path(path or self.ledger_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".tmp{os.getpid()}")
        temporary.write_text(json.dumps(self.ledger, indent=1, sort_keys=True))
        temporary.replace(path)


def ledger_checkpoint_callback(buffer: RollingPromptBuffer):
    """Copy the buffer ledger into each checkpoint, so a resume from that
    checkpoint finds the chunk positions even if OUTPUT_DIR moves."""
    from transformers import TrainerCallback

    class LedgerCheckpointCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            if state.is_world_process_zero:
                checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                if checkpoint.is_dir():
                    buffer.save(checkpoint / LEDGER_NAME)
            return control

    return LedgerCheckpointCallback()
