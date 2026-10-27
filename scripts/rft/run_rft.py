#!/usr/bin/env python
"""Rejection sampling fine tuning rounds with STaR rationalization, then GDPO.

    python scripts/rft/run_rft.py --plan scripts/rft/configs/rft_v4_a100.json

Each round r samples the current model on fresh train faults, keeps what
TetraMAX verifies, fine tunes, and evaluates:

1. ``select``: ``per_level`` problems per difficulty level, disjoint from
   earlier rounds.
2. ``generate``: ``num_samples`` unhinted attempts per problem on every GPU
   (``rationalize.py --escalate``). Problems all of them fail get hinted
   attempts, the least revealing hint (path) first, then partial.
3. ``build``: per problem, unhinted successes when there are any, otherwise
   rationalizations with the hint removed (``build_sft.py``).
4. ``sft``: ``epochs`` passes over those examples as ``extra_fraction`` of an
   SFT stream whose other part is unseen ordinary v4 rows, starting from the
   previous round's checkpoint.
5. ``eval``: greedy detection on the GDPO fixed evaluation (72 faults) and
   on a stratified validation set, scored the same way for every round.

Rounds stop at ``rounds``, when a round fails to beat the best validation
score, or when the next round would end after ``deadline_hours`` (measured
from the first launch). GDPO then starts from the best checkpoint (round 0 is
the starting SFT checkpoint) and runs to the end of its config.

Every stage writes a marker when it finishes, so a rerun (after a reboot or a
crash) skips finished stages and continues an interrupted one where it
stopped. A stage that fails ``max_failures`` times halts the driver.
``scripts/rft/watchdog.sh`` reruns it from cron. The plan is copied to
``<run_dir>/plan.json`` at the first launch, and that copy is read before
every stage, so it can be edited while the run goes on.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
TERMINATING = False
CHILD = None


class Halt(Exception):
    """A stage failed too often; the watchdog must not restart the driver."""


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message):
    print(f"[rft {now()}] {message}", flush=True)


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, default=str) + "\n")
    os.replace(tmp, path)


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


class Run:
    def __init__(self, plan_path, run_dir=None):
        plan = json.loads(Path(plan_path).read_text())
        self.dir = Path(run_dir or plan["run_dir"])
        if not self.dir.is_absolute():
            self.dir = REPO / self.dir
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "logs").mkdir(exist_ok=True)
        if not (self.dir / "plan.json").exists():
            write_json(self.dir / "plan.json", plan)
        self.state_path = self.dir / "state.json"
        state = read_json(self.state_path, {})
        state.setdefault("started", time.time())
        state.setdefault("failures", {})
        state.setdefault("done", {})
        state.setdefault("durations", {})
        self.state = state
        self.save()

    @property
    def plan(self):
        return read_json(self.dir / "plan.json")

    def save(self):
        write_json(self.state_path, self.state)

    def path(self, *parts):
        return self.dir.joinpath(*parts)

    def stage(self, name, function, *args, **kwargs):
        """Run ``function`` once; skip it when it already finished."""
        if name in self.state["done"]:
            return self.state["done"][name].get("result")
        failures = self.state["failures"].get(name, 0)
        if failures >= self.plan.get("max_failures", 3):
            raise Halt(f"stage {name} failed {failures} times; fix it and reset state.json failures")
        log(f"stage {name} starts (earlier failures: {failures})")
        start = time.time()
        try:
            result = function(*args, **kwargs)
        except Exception as exc:
            if TERMINATING:
                log(f"stage {name} interrupted by shutdown")
                raise SystemExit(143)
            self.state["failures"][name] = failures + 1
            exc.rft_stage_failure = True  # capped here, not as a driver crash
            self.save()
            log(f"stage {name} FAILED: {exc}")
            raise
        self.state["done"][name] = {"at": now(), "seconds": time.time() - start, "result": result}
        self.state["failures"].pop(name, None)
        self.save()
        log(f"stage {name} done in {(time.time() - start) / 3600:.2f} h")
        return result


def run(cmd, logfile, env=None, cwd=REPO):
    """Run ``cmd`` in its own process group, appending its output to ``logfile``."""
    global CHILD
    logfile = Path(logfile)
    logfile.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(part) for part in cmd]
    with open(logfile, "a") as sink:
        sink.write(f"\n=== {now()} {' '.join(cmd)}\n")
        sink.flush()
        CHILD = subprocess.Popen(cmd, cwd=cwd, env={**os.environ, **(env or {})}, stdout=sink,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        code = CHILD.wait()
        CHILD = None
    if code != 0:
        raise RuntimeError(f"exit {code}: {' '.join(cmd[:4])} ... (log {logfile})")


def on_signal(signum, _):
    global TERMINATING
    TERMINATING = True
    if CHILD is not None and CHILD.poll() is None:
        os.killpg(CHILD.pid, signal.SIGTERM)


def wait_for_free_gpus(plan):
    """Block until no compute process holds a planned GPU (another run, a stale server)."""
    gpus = plan["gpus"]
    while True:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
        if not out:
            return
        log(f"waiting for GPUs {gpus} to be free: {out.splitlines()[:4]}")
        time.sleep(300)


def gpu_list(plan):
    return ",".join(str(g) for g in plan["gpus"])


# ------------------------------------------------------------------ stages


def select_validation(r):
    plan = r.plan
    target = r.path("eval", "val_problems.parquet")
    target.parent.mkdir(exist_ok=True)
    run([PYTHON, "scripts/star/rationalize.py", "--adapter", plan["start_adapter"], "--dataset", plan["dataset"],
         "--split", "validation", "--per_level", plan["eval"]["val_per_level"], "--seed", plan["seed"],
         "--select_only", "--write_problems", target, "--out", r.path("eval", "val_select")],
        r.path("logs", "select_validation.log"))
    return str(target)


def merge_export(r, adapter, name):
    # One export per model before the shards start; four at once would race.
    code = ("import sys; from pathlib import Path; sys.path.insert(0, 'scripts/eval'); "
            "from evaluate_model import resolve_merged_bf16_dir; "
            f"print(resolve_merged_bf16_dir(Path({str(adapter)!r})))")
    wait_for_free_gpus(r.plan)
    run([PYTHON, "-c", code], r.path("logs", f"{name}.log"),
        env={"CUDA_VISIBLE_DEVICES": str(r.plan["gpus"][0]), "PYTHONPATH": str(REPO)})
    return str(Path(adapter).parent / f"{Path(adapter).name}_merged_bf16")


def evaluate(r, adapter, out, *, manifest=None, problems=None):
    plan = r.plan
    out = Path(out)
    what = ["--manifest", manifest] if manifest else ["--problems_file", problems]
    wait_for_free_gpus(plan)
    run(["bash", "scripts/rft/with_tetramax.sh", PYTHON, "scripts/rft/fanout.py", "--gpus", gpu_list(plan),
         "--log_dir", out / "logs", "--", PYTHON, "scripts/rft/evaluate.py", "--adapter", adapter, *what,
         "--out", out, "--shard", "{shard}", "--num_shards", "{num_shards}",
         "--max_tool_rounds", plan["eval"]["max_tool_rounds"],
         *(["--limit", plan["eval"]["limit"]] if plan["eval"].get("limit") else [])],
        r.path("logs", f"eval_{out.parent.name}_{out.name}.log"), env={"PYTHONPATH": str(REPO)})
    run([PYTHON, "scripts/rft/evaluate.py", "--out", out, "--summarize"],
        r.path("logs", f"eval_{out.parent.name}_{out.name}.log"))
    return read_json(out / "metrics.json")


def evaluate_model(r, tag, adapter):
    plan = r.plan
    r.stage(f"{tag}/export", merge_export, r, adapter, f"{tag.replace('/', '_')}_export")
    fixed = r.stage(f"{tag}/eval_fixed", evaluate, r, adapter, r.path(tag, "eval_fixed"),
                    manifest=plan["eval"]["fixed_manifest"])
    val = r.stage(f"{tag}/eval_val", evaluate, r, adapter, r.path(tag, "eval_val"),
                  problems=str(r.path("eval", "val_problems.parquet")))
    return {"adapter": str(adapter), "fixed": fixed["all"]["detection"],
            "val": val["level_mean_detection"],
            "val_by_level": {k: v["detection"] for k, v in val["by_level"].items()}}


def select_round(r, n):
    plan = r.plan
    target = r.path(f"round_{n}", "problems.parquet")
    target.parent.mkdir(exist_ok=True)
    earlier = [str(r.path(f"round_{i}", "problems.parquet")) for i in range(1, n)]
    run([PYTHON, "scripts/star/rationalize.py", "--adapter", plan["start_adapter"], "--dataset", plan["dataset"],
         "--per_level", plan["per_level"], "--seed", plan["seed"] + n, "--select_only",
         "--write_problems", target, "--exclude_problems", ",".join(earlier),
         "--hint_modes", plan["sample"]["hint_modes"], "--out", r.path(f"round_{n}", "gen")],
        r.path("logs", f"round_{n}_select.log"))
    return str(target)


def generate(r, n, adapter):
    plan, sample = r.plan, r.plan["sample"]
    out = r.path(f"round_{n}", "gen")
    wait_for_free_gpus(plan)
    run(["bash", "scripts/rft/with_tetramax.sh", PYTHON, "scripts/rft/fanout.py", "--gpus", gpu_list(plan),
         "--log_dir", out / "logs", "--", PYTHON, "scripts/star/rationalize.py", "--adapter", adapter,
         "--dataset", plan["dataset"], "--problems_file", r.path(f"round_{n}", "problems.parquet"),
         "--hint_modes", sample["hint_modes"], "--escalate", "--num_samples", sample["num_samples"],
         "--seed", plan["seed"] + n, "--temperature", sample["temperature"],
         "--max_tool_rounds", sample["max_tool_rounds"], "--max_model_len", sample["max_model_len"],
         "--max_generated_tokens", sample["max_generated_tokens"],
         "--batch_problems", sample["batch_problems"], "--out", out,
         "--shard", "{shard}", "--num_shards", "{num_shards}"],
        r.path("logs", f"round_{n}_generate.log"), env={"PYTHONPATH": str(REPO)})
    return str(out)


def build(r, n):
    plan = r.plan
    out = r.path(f"round_{n}", "gen")
    run([PYTHON, "scripts/star/build_sft.py", out, "--max_per_problem", plan["build"]["max_per_problem"],
         "--max_model_len", plan["sft"]["max_model_len"]], r.path("logs", f"round_{n}_build.log"),
        env={"PYTHONPATH": str(REPO)})
    report = read_json(out / "report.json")
    if report["sft_examples"] < plan["build"].get("min_examples", 1):
        raise RuntimeError(f"only {report['sft_examples']} SFT examples")
    return {"sft_examples": report["sft_examples"], "by_level_and_mode": report["sft_examples_by_level_and_mode"]}


def sft_steps(plan, examples):
    sft = plan["sft"]
    per_step = sft["gpus"] * sft.get("gradient_accumulation_steps", 256)
    steps = math.ceil(sft["epochs"] * examples / (sft["extra_fraction"] * per_step))
    steps = min(max(steps, sft.get("min_steps", 10)), sft.get("max_steps", 60))
    return int(math.ceil(steps / 10) * 10)  # SFT saves every 10 steps


def complete_checkpoint(directory, step):
    summary = Path(directory) / f"checkpoint-{step}" / "training_state_summary.json"
    return summary.exists() and read_json(summary).get("global_step") == step


def train(r, n, start_adapter, examples):
    plan, sft = r.plan, r.plan["sft"]
    out = r.path(f"round_{n}", "sft")
    steps = r.state.setdefault("sft_steps", {}).setdefault(str(n), sft_steps(plan, examples))
    r.save()
    if complete_checkpoint(out, steps):
        return str(out / f"checkpoint-{steps}")
    has_checkpoint = subprocess.run([PYTHON, "atpgllm/training/checkpoints.py", str(out)], cwd=REPO,
                                    capture_output=True).returncode == 0 if out.exists() else False
    env = {"OUTPUT_DIR": str(out), "MAX_STEPS": str(steps),
           "SFT_EXTRA_MESSAGES": str(r.path(f"round_{n}", "gen", "sft.jsonl")),
           "SFT_EXTRA_FRACTION": str(sft["extra_fraction"]), "SFT_LEARNING_RATE": str(sft["learning_rate"]),
           "GRADIENT_ACCUMULATION_STEPS": str(sft.get("gradient_accumulation_steps", 256)),
           "SFT_CIRCUIT_VALIDATION": str(sft.get("circuit_validation", True)),
           "RESUME_FROM": "latest" if has_checkpoint else str(start_adapter),
           "RESUME_TRAINING_STATE": "True" if has_checkpoint else "False"}
    wait_for_free_gpus(plan)
    log(f"round {n} SFT: {examples} examples, {steps} steps, from {env['RESUME_FROM']}")
    run(["bash", "scripts/train/run_training_code.sh", sft["conf"]], r.path("logs", f"round_{n}_sft.log"), env=env)
    if not complete_checkpoint(out, steps):
        raise RuntimeError(f"SFT ended without checkpoint-{steps}")
    return str(out / f"checkpoint-{steps}")


def gdpo(r, adapter):
    plan, g = r.plan, r.plan["gdpo"]
    out = Path(g["output_dir"])
    out = out if out.is_absolute() else REPO / out
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "fixed_eval_manifest.json"
    if not manifest.exists():
        shutil.copy2(plan["eval"]["fixed_manifest"], manifest)
    resumable = subprocess.run([PYTHON, "atpgllm/training/checkpoints.py", str(out)], cwd=REPO,
                               capture_output=True).returncode == 0
    env = {"OUTPUT_DIR": str(out), "RESUME_FROM": "latest" if resumable else str(adapter),
           "RESUME_TRAINING_STATE": "True" if resumable else "False"}
    wait_for_free_gpus(plan)
    log(f"GDPO from {env['RESUME_FROM']} into {out}")
    run(["bash", "scripts/train/run_training_code.sh", g["conf"]], r.path("logs", "gdpo.log"), env=env)
    return str(out)


# ------------------------------------------------------------------ driver


def rounds(r):
    plan = r.plan
    r.stage("eval/select_validation", select_validation, r)
    results = r.state.setdefault("results", {})
    results["0"] = r.stage("round_0/evaluate", evaluate_model, r, "round_0", plan["start_adapter"])
    r.save()
    for tag, adapter in plan["eval"].get("references", {}).items():
        r.state.setdefault("references", {})[tag] = r.stage(f"{tag}/evaluate", evaluate_model, r, tag, adapter)
        r.save()
    best = max(results, key=lambda k: (results[k]["val"], results[k]["fixed"]))
    for n in range(1, plan["rounds"] + 1):
        if r.state.get("rounds_stopped"):
            break
        if f"round_{n}/evaluate" not in r.state["done"]:
            spent = time.time() - r.state["started"]
            previous = [v for k, v in r.state["durations"].items()]
            estimate = previous[-1] if previous else plan["round_hours_estimate"] * 3600
            if spent + estimate > plan["deadline_hours"] * 3600:
                log(f"round {n} would end after the {plan['deadline_hours']} h deadline; stopping rounds")
                r.state["rounds_stopped"] = f"deadline before round {n}"
                break
        started = r.state.setdefault("round_started", {}).setdefault(str(n), time.time())
        r.save()
        model = results[str(n - 1)]["adapter"]
        r.stage(f"round_{n}/select", select_round, r, n)
        r.stage(f"round_{n}/export_policy", merge_export, r, model, f"round_{n}_export_policy")
        r.stage(f"round_{n}/generate", generate, r, n, model)
        built = r.stage(f"round_{n}/build", build, r, n)
        adapter = r.stage(f"round_{n}/sft", train, r, n, model, built["sft_examples"])
        results[str(n)] = r.stage(f"round_{n}/evaluate", evaluate_model, r, f"round_{n}", adapter)
        r.state["durations"].setdefault(str(n), time.time() - started)
        r.save()
        log(f"round {n}: fixed {results[str(n)]['fixed']:.3f} val {results[str(n)]['val']:.3f} "
            f"(best so far round {best}: val {results[best]['val']:.3f})")
        if (results[str(n)]["val"], results[str(n)]["fixed"]) <= (results[best]["val"], results[best]["fixed"]):
            r.state["rounds_stopped"] = f"round {n} did not beat round {best}"
            r.save()
            break
        best = str(n)
    best = max(results, key=lambda k: (results[k]["val"], results[k]["fixed"]))
    r.state["best"] = {"round": best, **results[best]}
    r.save()
    write_json(r.path("best.json"), r.state["best"])
    log(f"best: round {best} {results[best]}")
    return results[best]["adapter"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", default=str(REPO / "scripts/rft/configs/rft_v4_a100.json"))
    parser.add_argument("--run_dir", default=None, help="Overrides the plan's run_dir")
    parser.add_argument("--no_gdpo", action="store_true", help="Stop after choosing the best round")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGHUP, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    os.chdir(REPO)
    r = Run(args.plan, args.run_dir)
    (r.path("driver.pid")).write_text(f"{os.getpid()}\n")
    log(f"driver pid {os.getpid()} in {r.dir}")
    try:
        best = rounds(r)
        if args.no_gdpo or not r.plan.get("gdpo"):
            r.state["finished"] = now()
            r.save()
            return
        r.stage("gdpo", gdpo, r, best)
        r.state["finished"] = now()
        r.save()
        log("GDPO finished; the plan is complete")
    except Halt as exc:
        r.state["halted"] = f"{now()} {exc}"
        r.save()
        log(f"HALTED: {exc}")
        sys.exit(3)
    except SystemExit:
        raise
    except Exception as exc:
        if TERMINATING:
            raise SystemExit(143)
        # Stage failures are capped per stage; anything else is capped here.
        if not getattr(exc, "rft_stage_failure", False):
            r.state["driver_crashes"] = r.state.get("driver_crashes", 0) + 1
        if r.state.get("driver_crashes", 0) >= 2 * r.plan.get("max_failures", 3):
            r.state["halted"] = f"{now()} driver crashed {r.state['driver_crashes']} times: {exc}"
        r.save()
        raise


if __name__ == "__main__":
    main()
