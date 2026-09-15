"""Exercise the shell launchers without importing the GPU evaluation stack."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "eval"


@pytest.fixture
def launcher(tmp_path):
    repo = tmp_path / "repo with spaces"
    scripts = repo / "scripts" / "eval"
    scripts.mkdir(parents=True)
    for script in SCRIPTS.glob("*.sh"):
        shutil.copy(script, scripts / script.name)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # Capture exact argv through the real execution branch, including redirection.
    python = bin_dir / "python"
    python.write_text('#!/usr/bin/env python3\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n')
    python.chmod(0o755)

    def run(kind, *args, **overrides):
        env = {key: value for key, value in os.environ.items() if key in (
            "PATH", "HOME", "LANG", "TMPDIR",
        )}
        env.update(PATH=f"{bin_dir}:{env['PATH']}", TP_SIZE="1",
                   VIRTUAL_ENV=str(tmp_path), EVAL_RESULTS_DIR=str(tmp_path / "results"))
        env.update(overrides)
        result = subprocess.run(
            ["bash", str(scripts / f"eval_{kind}_policy_checkpoints.sh"), *map(str, args)],
            cwd=tmp_path, env=env, text=True, capture_output=True,
        )
        return result

    return repo, run


def adapter(path, model="ibm-granite/granite-4.2-8b"):
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": model}))
    return path


def captured_args(tmp_path):
    return [json.loads(log.read_text()) for log in sorted((tmp_path / "results").glob("*.log"))]


@pytest.mark.parametrize("model", ["ibm-granite/granite-4.2-8b", "Qwen/Qwen2.5-32B-Instruct"])
@pytest.mark.parametrize("path_mode", ["absolute", "repo_relative", "runs_relative", "environment"])
def test_sft_checkpoint_paths_and_model_independence(launcher, tmp_path, model, path_mode):
    repo, run = launcher
    checkpoint = adapter(repo / "runs" / "experiment" / "checkpoint-200", model)
    supplied = {
        "absolute": checkpoint,
        "repo_relative": "runs/experiment/checkpoint-200",
        "runs_relative": "experiment/checkpoint-200",
    }
    result = (run("sft", CHECKPOINT=str(checkpoint)) if path_mode == "environment"
              else run("sft", supplied[path_mode]))
    assert result.returncode == 0, result.stderr
    args, = captured_args(tmp_path)
    assert args[args.index("--adapter") + 1] == str(checkpoint)
    assert "--merge_dequant" in args


@pytest.mark.parametrize("layout", ["policy", "combined/policy"])
@pytest.mark.parametrize("direct", [False, True])
def test_grpo_checkpoint_and_explicit_policy(launcher, tmp_path, layout, direct):
    repo, run = launcher
    checkpoint = repo / "runs" / "experiment" / "checkpoint-40"
    policy = adapter(checkpoint / layout)
    if layout == "combined/policy" and direct:
        adapter(checkpoint / "policy")  # Explicit legacy path must still win.
    result = run("grpo", policy if direct else checkpoint)
    assert result.returncode == 0, result.stderr
    args, = captured_args(tmp_path)
    assert args[args.index("--adapter") + 1] == str(policy)


@pytest.mark.parametrize("kind", ["sft", "grpo"])
def test_experiment_order_skips_missing_and_merged_exports(launcher, tmp_path, kind):
    repo, run = launcher
    experiment = repo / "runs" / "experiment"
    for name in ["checkpoint-10", "checkpoint-2", "checkpoint-2_merged_bf16"]:
        path = experiment / name
        adapter(path / "policy" if kind == "grpo" else path)
    (experiment / "checkpoint-3").mkdir()
    result = run(kind, experiment)
    assert result.returncode == 0, result.stderr
    assert result.stdout.index("checkpoint-2") < result.stdout.index("checkpoint-10")
    assert "skip:" in result.stderr
    assert len(captured_args(tmp_path)) == 2


@pytest.mark.parametrize("kind", ["sft", "grpo"])
def test_invalid_checkpoint_and_missing_argument_fail(launcher, kind):
    repo, run = launcher
    checkpoint = repo / "checkpoint-1"
    checkpoint.mkdir()
    assert run(kind, checkpoint).returncode != 0
    assert run(kind).returncode != 0
    assert run(kind, "--help").returncode == 0


@pytest.mark.parametrize("method,flag,width", [("mcts", "--budget", "50"),
                                                ("evolutionary", "--budget", "50"),
                                                ("vector_evolutionary", "--budget", "50"),
                                                ("best_of_n", "--n", "4")])
def test_sampling_and_logging_overrides(launcher, tmp_path, method, flag, width):
    repo, run = launcher
    checkpoint = adapter(repo / "checkpoint-1")
    result = run("sft", checkpoint, SAMPLING_METHOD=method, MERGE_DEQUANT="0",
                 WANDB_RUN_NAME="custom run", REPORT_TO="none")
    assert result.returncode == 0, result.stderr
    args, = captured_args(tmp_path)
    assert args[args.index(flag) + 1] == width
    assert args[args.index("--wandb_run_name") + 1] == "custom run"
    assert args[args.index("--report_to") + 1] == "none"
    assert "--merge_dequant" not in args
    assert 'csv1' in args[args.index('--output_file') + 1]


def test_dry_run_and_validation(launcher, tmp_path):
    repo, run = launcher
    checkpoint = adapter(repo / "checkpoint-1")
    result = run("sft", checkpoint, DRY_RUN="1", TP_SIZE="", CUDA_VISIBLE_DEVICES="0,1")
    assert result.returncode == 0, result.stderr
    assert "--tp_size 2" in result.stdout
    assert not (tmp_path / "results").exists()
    for overrides in ({"NUM_COMPLETIONS": "2"}, {"SEARCH_BUDGET": "3"},
                      {"BEST_OF_N_WIDTH": "3"}, {"PASS_AT_K": "0"},
                      {"SAMPLING_METHOD": "invalid"}):
        assert run("sft", checkpoint, **overrides).returncode != 0
