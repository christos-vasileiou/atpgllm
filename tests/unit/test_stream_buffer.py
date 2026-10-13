"""Seeded split order and the resumable rolling GRPO buffer."""
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from atpgllm.training.stream_buffer import RollingPromptBuffer, ShardPermutation, streaming_split


def _split(tmp_path, circuits=6, rows=10):
    folder = tmp_path / "train"
    folder.mkdir()
    shards = {}
    for c in range(circuits):
        table = [{"circuit_id": f"c{c}", "fault": f"sa0 n{i}", "prompt_text": f"c{c} n{i}", "n": i}
                 for i in range(rows)]
        name = f"train/data-{c:05d}.parquet"
        pq.write_table(pa.Table.from_pylist(table), tmp_path / name)
        shards[name] = {"rows": rows}
    (tmp_path / "split_manifest.json").write_text(json.dumps({"shards": shards}))
    return tmp_path


def _prepare(rows, drop=lambda r: False):
    return [None if drop(r) else {"prompt": r["prompt_text"], "fault": r["fault"],
                                   "circuit_id": r["circuit_id"]} for r in rows]


def test_permutation_visits_every_row_once_per_pass_and_mixes_circuits(tmp_path):
    source = ShardPermutation(_split(tmp_path), seed=3, block_size=7)
    first = [row["prompt_text"] for _, row in source.rows(0, 0, source.total)]
    assert sorted(first) == sorted(f"c{c} n{i}" for c in range(6) for i in range(10))
    assert len({r.split()[0] for r in first[:10]}) > 2  # not one circuit at a time
    again = [row["prompt_text"] for _, row in ShardPermutation(tmp_path, seed=3).rows(0, 0, 60)]
    assert again == first
    assert [row["prompt_text"] for _, row in source.rows(1, 0, 60)] != first
    stream = streaming_split(tmp_path, seed=3)
    assert [r["prompt_text"] for _, r in zip(range(60), stream)] == first


def _buffer(tmp_path, ledger, **kw):
    source = ShardPermutation(tmp_path, seed=1)
    return RollingPromptBuffer(source, kw.pop("prepare", _prepare), chunk_size=8, length=80,
                               ledger_path=ledger, settings={"max_prompt_length": 100},
                               reorder=lambda rows, k: rows, read_block=5, **kw)


def test_chunks_roll_over_and_a_restart_rebuilds_the_same_rows(tmp_path):
    data = _split(tmp_path)
    ledger = tmp_path / "run" / "grpo_stream_ledger.json"
    buffer = _buffer(data, ledger)
    seen = [buffer[i]["prompt"] for i in range(len(buffer))]  # 80 > 60 rows: wraps to pass 1
    assert len(set(seen[:56])) == 56  # pass 0 rows are new until the split is exhausted
    chunks = json.loads(ledger.read_text())["chunks"]
    assert chunks["1"]["start"] == chunks["0"]["end"]
    assert chunks["9"]["end"][0] == 1
    # A new process resumes at chunk 6 from the ledger alone.
    resumed = _buffer(data, tmp_path / "other" / "ledger.json", resume_ledger=ledger)
    assert [resumed[i]["prompt"] for i in range(48, 56)] == seen[48:56]


def test_resume_refuses_different_data(tmp_path):
    data = _split(tmp_path)
    ledger = tmp_path / "ledger.json"
    buffer = _buffer(data, ledger)
    buffer[0]
    # Same positions, different prompt text (for example a new chat template).
    reworded = lambda rows: [dict(r, prompt=r["prompt"] + "!") for r in _prepare(rows)]
    changed = _buffer(data, ledger, prepare=reworded)
    with pytest.raises(ValueError, match="differs from the one this run trained on"):
        changed[0]
    source = ShardPermutation(data, seed=1)
    with pytest.raises(ValueError, match="different data settings"):
        RollingPromptBuffer(source, _prepare, chunk_size=4, length=8, ledger_path=ledger,
                            settings={"max_prompt_length": 100})


def test_filters_and_duplicates_are_skipped(tmp_path):
    data = _split(tmp_path)
    buffer = _buffer(data, tmp_path / "l.json",
                     prepare=lambda rows: _prepare(rows, lambda r: r["circuit_id"] == "c0"))
    prompts = [buffer[i]["prompt"] for i in range(40)]
    assert not any(p.startswith("c0 ") for p in prompts)
    assert len(set(prompts[:8])) == 8
    with pytest.raises(ValueError, match="no training prompt"):
        _buffer(data, tmp_path / "m.json", prepare=lambda rows: [None] * len(rows))[0]
