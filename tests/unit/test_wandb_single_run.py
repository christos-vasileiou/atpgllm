"""training_code._init_wandb: one W&B run per job, continued only on a training-state resume.

Each rank is a separate process (as under accelerate/torchrun), with W&B
offline so nothing leaves the machine.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("wandb")

REPO = Path(__file__).resolve().parents[2]

CHILD = r"""
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("training_code_under_test", sys.argv[1])
tc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tc)
args = {"output_dir": sys.argv[2], "resume_training_state": sys.argv[3] == "1"}
run = tc._init_wandb("sft", args, {"x": 1}, "default-name")
tc.wandb.define_metric("eval/fixed/global_step")
tc.wandb.log({"loss": 1.0})
print(json.dumps({"mode": run.settings.mode, "id": run.id, "name": run.name}))
run.finish()
"""


def _launch(tmp_path, out, resume, ranks=4):
    procs = []
    for rank in range(ranks):
        env = {**os.environ, "RANK": str(rank), "WANDB_MODE": "offline", "WANDB_DIR": str(tmp_path),
               "WANDB_SILENT": "true", "WANDB_NAME": "rft_test/round_1/sft", "PYTHONPATH": str(REPO)}
        env.pop("WANDB_RUN_ID", None)
        procs.append(subprocess.Popen(
            [sys.executable, "-c", CHILD, str(REPO / "scripts/train/training_code.py"), str(out),
             "1" if resume else "0"],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(tmp_path)))
    results = []
    for proc in procs:
        stdout, stderr = proc.communicate(timeout=600)
        assert proc.returncode == 0, stderr[-3000:]
        results.append(json.loads(stdout.strip().splitlines()[-1]))
    return results


def test_only_rank_zero_reports_and_resume_continues_the_run(tmp_path):
    out = tmp_path / "sft"
    first = _launch(tmp_path, out, resume=False)
    assert first[0]["mode"] == "offline" and first[0]["name"] == "rft_test/round_1/sft"
    assert all(r["mode"] == "disabled" for r in first[1:])
    run_dirs = list((tmp_path / "wandb").glob("offline-run-*"))
    assert len(run_dirs) == 1, run_dirs
    stored = (out / "wandb_run_id").read_text().strip()
    assert stored == first[0]["id"]

    # A restart that resumes the training state continues the same run.
    resumed = _launch(tmp_path, out, resume=True, ranks=2)
    assert resumed[0]["id"] == stored and resumed[1]["mode"] == "disabled"
    # A fresh launch into the same directory starts a new run.
    fresh = _launch(tmp_path, out, resume=False, ranks=1)
    assert fresh[0]["id"] != stored
    assert (out / "wandb_run_id").read_text().strip() == fresh[0]["id"]
