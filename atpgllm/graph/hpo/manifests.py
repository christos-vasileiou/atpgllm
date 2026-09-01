"""Deterministic trial, result, and checkpoint lineage manifests."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

import torch


TRIAL_MANIFEST_VERSION = 1


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def canonical_hash(value: Any, length: int | None = None) -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return digest[:length] if length else digest


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def design_hash(record_or_netlist: Any) -> str:
    if isinstance(record_or_netlist, Mapping):
        netlist = record_or_netlist.get("netlist", "")
    else:
        netlist = record_or_netlist
    if isinstance(netlist, Mapping):
        netlist = netlist.get("netlist", "")
    return hashlib.sha256(str(netlist or "").encode("utf-8")).hexdigest()


def write_json_atomic(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def append_jsonl(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(payload) + "\n")


def checkpoint_identity(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path).resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return {
        "path": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "stage": payload.get("stage"),
        "format": payload.get("format"),
        "format_version": payload.get("format_version"),
        "vocab_fingerprint": payload.get("vocab_fingerprint"),
        "architecture_hash": canonical_hash(payload.get("architecture", {})),
        "step": int(payload.get("step", 0)),
    }


def git_state(repository: str | Path) -> dict[str, Any]:
    root = Path(repository)

    def run(*args: str) -> bytes:
        return subprocess.check_output(
            ["git", *args],
            cwd=root,
            stderr=subprocess.DEVNULL,
        )

    try:
        commit = run("rev-parse", "HEAD").decode().strip()
        diff = run("diff", "--binary", "HEAD")
        status = run("status", "--porcelain").decode().splitlines()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "diff_sha256": None}
    return {
        "commit": commit,
        "dirty": bool(status),
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def trial_artifact_dir(
    root: str | Path,
    study_name: str,
    stage: str,
    trial_number: int,
    seed: int,
) -> Path:
    return (
        Path(root)
        / study_name
        / stage
        / f"trial-{trial_number:05d}"
        / f"seed-{seed}"
    )


def checkpoint_filename(
    study_name: str,
    stage: str,
    trial_number: int,
    seed: int,
    step: int,
    config_hash: str,
) -> str:
    safe_study = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in study_name
    )
    return (
        f"{safe_study}_{stage}_t{trial_number:05d}_s{seed}_"
        f"step{step}_{config_hash[:8]}.pt"
    )


def build_trial_manifest(
    *,
    study_name: str,
    stage: str,
    trial_number: int,
    seed: int,
    fidelities: list[int] | tuple[int, ...],
    params: Mapping[str, Any],
    fixed: Mapping[str, Any],
    split_manifest: str | Path,
    repository: str | Path,
    parent_checkpoint: str | Path | None,
) -> dict[str, Any]:
    parent = (
        checkpoint_identity(parent_checkpoint)
        if parent_checkpoint is not None
        else None
    )
    configuration = {
        "params": dict(params),
        "fixed": dict(fixed),
        "parent_checkpoint_sha256": (
            parent["sha256"] if parent is not None else None
        ),
    }
    return {
        "version": TRIAL_MANIFEST_VERSION,
        "study": study_name,
        "stage": stage,
        "trial_number": int(trial_number),
        "seed": int(seed),
        "fidelities": [int(value) for value in fidelities],
        "configuration": configuration,
        "config_hash": canonical_hash(configuration),
        "split_manifest": {
            "path": str(Path(split_manifest).resolve()),
            "sha256": sha256_file(split_manifest),
        },
        "parent_checkpoint": parent,
        "git": git_state(repository),
        "slurm": {
            key: os.environ.get(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURM_ARRAY_JOB_ID",
                "SLURM_ARRAY_TASK_ID",
                "SLURM_NODELIST",
            )
            if os.environ.get(key) is not None
        },
    }
