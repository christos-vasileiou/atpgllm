#!/usr/bin/env python
"""Mirror an RFT run directory (``scripts/rft/run_rft.py``) to Weights & Biases.

    python scripts/rft/wandb_log.py --run_dir runs/rft_v4_a100 [--project rft-training] [--every 900]

It reads only what the driver writes (``state.json``, ``plan.json``, the
round directories) and never touches the GPUs, so it runs beside the driver
and can be stopped or restarted at any time. One W&B run per run directory
(its id is kept in ``<run_dir>/wandb_run_id``) gets:

* ``round/*`` against ``round``: greedy detection on the fixed 72 faults and
  on the validation set by level, generation and build statistics, SFT steps
  and round hours. Round 0 is the starting checkpoint; the reference
  checkpoints go to the run summary and the ``evaluations`` table.
* ``live/*`` against ``hours`` since the run started: the current round's
  generation progress.
* ``sft/*`` against ``sft_step`` (counted across rounds): the SFT losses from
  each round's ``trainer_state.json``.
* ``completions/round_<n>``: once round n is built, 3 trajectories it trains
  on and 3 it dropped, with the reason each was dropped.

``<run_dir>/wandb_logged.json`` records what was sent, so a restart does not
log a round twice. With ``--every`` it repeats until the run finishes or
halts, then syncs once more and exits.
"""
from __future__ import annotations

import argparse
import html
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

LEVELS = ("easy", "medium", "hard", "very_hard", "random_never_detects")


def read_json(path, default=None):
    path = Path(path)
    try:
        return json.loads(path.read_text()) if path.exists() else default
    except json.JSONDecodeError:
        return default


def write_json(path, value):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, indent=1) + "\n")
    tmp.replace(path)


def read_slots(directory):
    for path in sorted(Path(directory).glob("slots*.jsonl")):
        for line in open(path):
            if line.endswith("\n"):
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass


class SlotTail:
    """Generation counters for one round, reading only lines added since the last call."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.offsets = {}
        self.slots = defaultdict(int)
        self.detected = defaultdict(int)
        self.problems = defaultdict(set)
        self.solved = defaultdict(set)
        self.tokens = 0

    def update(self):
        for path in sorted(self.directory.glob("slots*.jsonl")):
            with open(path, "rb") as handle:
                handle.seek(self.offsets.get(path, 0))
                for raw in handle:
                    if not raw.endswith(b"\n"):
                        break
                    self.offsets[path] = self.offsets.get(path, 0) + len(raw)
                    try:
                        slot = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    mode, problem = slot["hint_mode"], tuple(slot["problem"])
                    self.slots[mode] += 1
                    self.problems[mode].add(problem)
                    self.tokens += (slot.get("usage") or {}).get("generated_tokens", 0)
                    if slot["facts"]["detected"]:
                        self.detected[mode] += 1
                        self.solved[mode].add(problem)
        return self


# ------------------------------------------------------------------ completions


def chosen_keys(slots, max_per_problem):
    """``(problem, hint mode, slot)`` of every trajectory build_sft.py trains on."""
    from atpgllm.training.rationalization import choose_for_training
    by_problem = defaultdict(list)
    for slot in slots:
        by_problem[tuple(slot["problem"])].append(slot)
    keys = set()
    for problem, group in by_problem.items():
        for slot in choose_for_training(group, max_per_problem):
            keys.add((problem, slot["hint_mode"], slot.get("completion_slot", 0)))
    return keys, by_problem


def drop_reason(slot, solved_unhinted):
    from atpgllm.training.rationalization import keep
    facts = slot["facts"]
    if not facts["final"]:
        return f"no final answer ({slot.get('status')}: {slot.get('stop_reason') or 'stopped'})"
    if not facts["detected"]:
        return "the answer does not detect the fault"
    if not facts["single_user_turn"]:
        return "detected, but needed a protocol repair turn"
    if facts["leak"]:
        return "detected, but the reasoning refers to the hint"
    if keep(facts) and slot["hint_mode"] != "none" and solved_unhinted:
        return "detected with a hint, but an unhinted attempt solved the problem"
    return "detected, but the answer repeats a kept one or the per-problem cap was reached"


def pick_completions(slots, max_per_problem, seed, count=3):
    """``count`` kept and ``count`` dropped slots, spread over levels and outcomes."""
    keys, by_problem = chosen_keys(slots, max_per_problem)
    rng = random.Random(seed)
    harder_first = {level: i for i, level in enumerate(reversed(LEVELS))}
    kept, dropped = defaultdict(list), defaultdict(list)
    for problem, group in by_problem.items():
        solved_unhinted = any(s["hint_mode"] == "none" and s["facts"]["detected"] for s in group)
        for slot in group:
            if (problem, slot["hint_mode"], slot.get("completion_slot", 0)) in keys:
                # A rationalization first, then unhinted ones from the hardest levels.
                bucket = ("hinted",) if slot["hint_mode"] != "none" else ("unhinted", slot["difficulty"])
                kept[bucket].append((slot, "trained on"))
            else:
                reason = drop_reason(slot, solved_unhinted)
                kind = ("wrong" if reason.startswith("the answer") else
                        "no_answer" if reason.startswith("no final") else "not_used")
                dropped[kind].append((slot, reason))

    def take(buckets, order):
        out = []
        for bucket in order:
            if buckets.get(bucket):
                out.append(buckets[bucket].pop(rng.randrange(len(buckets[bucket]))))
            if len(out) == count:
                return out
        rest = [item for bucket in order for item in buckets.get(bucket, [])]
        rng.shuffle(rest)
        return out + rest[:count - len(out)]

    kept_order = [("hinted",)] + [("unhinted", level) for level in sorted(LEVELS, key=harder_first.get)]
    return take(kept, kept_order), take(dropped, ["wrong", "no_answer", "not_used"])


def clip(text, limit):
    text = str(text)
    return text if len(text) <= limit else text[:limit] + f"\n... [{len(text) - limit} more characters]"


def transcript(slot):
    """The generated part of the trajectory as plain text (the prompt is the dataset's)."""
    parts = []
    if slot.get("hint"):
        parts.append(f"[hint appended to the prompt, removed before SFT]\n{slot['hint']}")
    for m in (slot.get("messages") or [])[len(slot.get("prompt_messages") or []):]:
        body = str(m.get("content") or "")
        if m.get("tool_calls"):
            body += "\n[tool call] " + json.dumps([c["function"] for c in m["tool_calls"]])
        parts.append(f"[{m['role']}]\n{clip(body, 3000 if m['role'] == 'tool' else 12000)}")
    return "\n\n".join(parts)


def render(slot, outcome, reason):
    """The trajectory as HTML: prompt folded, hint, every generated turn, verdict."""
    esc = html.escape
    usage = slot.get("usage") or {}
    head = (f"<p><b>{esc(outcome.upper())}</b> ({esc(reason)})<br>"
            f"<b>fault</b> {esc(str(slot.get('fault')))} &nbsp; <b>level</b> {esc(slot['difficulty'])} &nbsp; "
            f"<b>random detection p</b> {slot.get('random_detection_probability')} &nbsp; "
            f"<b>hint mode</b> {esc(slot['hint_mode'])}<br>"
            f"<b>status</b> {esc(str(slot.get('status')))} &nbsp; <b>detected</b> {slot['facts']['detected']} "
            f"&nbsp; <b>tool calls</b> {slot['facts']['tool_calls']} &nbsp; "
            f"<b>generated tokens</b> {usage.get('generated_tokens')}</p>")
    prompt = "\n\n".join(f"[{m['role']}]\n{m.get('content')}" for m in slot.get("prompt_messages") or [])
    parts = [head, f"<details><summary>prompt ({len(prompt)} characters)</summary>"
                   f"<pre style='white-space:pre-wrap'>{esc(clip(prompt, 20000))}</pre></details>"]
    if slot.get("hint"):
        parts.append(f"<p><b>hint appended to the prompt (removed before SFT)</b></p>"
                     f"<pre style='white-space:pre-wrap'>{esc(slot['hint'])}</pre>")
    messages = slot.get("messages") or []
    for m in messages[len(slot.get("prompt_messages") or []):]:
        body = str(m.get("content") or "")
        if m.get("tool_calls"):
            body += "\n\n[tool call]\n" + json.dumps([c["function"] for c in m["tool_calls"]], indent=1)
        limit = 3000 if m["role"] == "tool" else 12000
        parts.append(f"<p><b>{esc(m['role'])}</b></p><pre style='white-space:pre-wrap'>{esc(clip(body, limit))}</pre>")
    return "<div style='font-family:monospace;font-size:12px'>" + "".join(parts) + "</div>"


# ------------------------------------------------------------------ sync


def round_metrics(run_dir, n, result, durations, sft_steps):
    metrics = {"round": n, "round/fixed_detection": result["fixed"], "round/val_level_mean": result["val"]}
    for level, value in (result.get("val_by_level") or {}).items():
        metrics[f"round/val_{level}"] = value
    if n == 0:
        return metrics
    if str(n) in durations:
        metrics["round/hours"] = durations[str(n)] / 3600
    if str(n) in sft_steps:
        metrics["round/sft_steps"] = sft_steps[str(n)]
    report = read_json(run_dir / f"round_{n}" / "gen" / "report.json")
    if report:
        metrics["round/sft_examples"] = report["sft_examples"]
        metrics["round/sft_problems"] = report["sft_problems"]
        by_mode = defaultdict(int)
        for level, modes in report["sft_examples_by_level_and_mode"].items():
            for mode, count in modes.items():
                by_mode[mode] += count
                metrics[f"round/sft_examples_{level}"] = metrics.get(f"round/sft_examples_{level}", 0) + count
        for mode, count in by_mode.items():
            metrics[f"round/sft_examples_mode_{mode}"] = count
        for key, row in report["by_mode_and_level"].items():
            mode, level = key.split("/", 1)
            metrics[f"round/solved_{mode}_{level}"] = row["problems_solved"]
            metrics[f"round/detect_per_attempt_{mode}_{level}"] = row["detect_rate_per_slot"]
    return metrics


def sync(run, run_dir, logged, tails, completions):
    import wandb
    state = read_json(run_dir / "state.json", {})
    plan = read_json(run_dir / "plan.json", {})
    if plan and not run.config.get("plan"):  # the logger may start before the driver copies the plan
        run.config.update({"plan": plan}, allow_val_change=True)
    done, results = state.get("done", {}), state.get("results", {})
    durations, sft_steps = state.get("durations", {}), state.get("sft_steps", {})

    # Rounds, once evaluated (round 0 is the starting checkpoint).
    for n in sorted(results, key=int):
        if n in logged["rounds"] or (n != "0" and f"round_{n}/evaluate" not in done):
            continue
        wandb.log(round_metrics(run_dir, int(n), results[n], durations, sft_steps))
        logged["rounds"].append(n)
    for tag, ref in (state.get("references") or {}).items():
        run.summary[f"reference/{tag}/fixed_detection"] = ref["fixed"]
        run.summary[f"reference/{tag}/val_level_mean"] = ref["val"]
    table_rows = len(logged["rounds"]) + len(state.get("references") or {})
    if results and table_rows != logged.get("eval_table_rows"):
        columns = ["model", "adapter", "fixed_detection", "val_level_mean", *[f"val_{lv}" for lv in LEVELS]]
        table = wandb.Table(columns=columns)
        rows = [(f"round {n}", r) for n, r in sorted(results.items(), key=lambda kv: int(kv[0]))
                if n in logged["rounds"]]
        rows += [(f"reference {tag}", r) for tag, r in (state.get("references") or {}).items()]
        for name, r in rows:
            table.add_data(name, r["adapter"], r["fixed"], r["val"],
                           *[(r.get("val_by_level") or {}).get(lv) for lv in LEVELS])
        wandb.log({"evaluations": table})
        logged["eval_table_rows"] = table_rows

    # Completions, once a round is built.
    for n in range(1, plan.get("rounds", 0) + 1):
        if str(n) in logged["completions"] or f"round_{n}/build" not in done:
            continue
        kept, dropped = pick_completions(list(read_slots(run_dir / f"round_{n}" / "gen")),
                                         plan["build"]["max_per_problem"], plan.get("seed", 0) + n,
                                         completions)
        table = wandb.Table(columns=["round", "outcome", "reason", "level", "hint_mode", "fault",
                                     "random_detection_p", "status", "detected", "tool_calls",
                                     "generated_tokens", "conversation"])
        pages = {}
        for outcome, picks in (("kept", kept), ("dropped", dropped)):
            for index, (slot, reason) in enumerate(picks, 1):
                table.add_data(n, outcome, reason, slot["difficulty"], slot["hint_mode"], slot.get("fault"),
                               slot.get("random_detection_probability"), slot.get("status"),
                               slot["facts"]["detected"], slot["facts"]["tool_calls"],
                               (slot.get("usage") or {}).get("generated_tokens"), transcript(slot))
                pages[f"completions/round_{n}_{outcome}_{index}"] = wandb.Html(render(slot, outcome, reason))
        # Tables hold text only; each conversation is also a formatted panel.
        wandb.log({f"completions/round_{n}": table, **pages})
        logged["completions"].append(str(n))

    # SFT losses, counted across rounds.
    offset = 0
    for n in range(1, plan.get("rounds", 0) + 1):
        sft_dir = run_dir / f"round_{n}" / "sft"
        checkpoints = sorted((p for p in sft_dir.glob("checkpoint-*") if p.name.split("-")[1].isdigit()),
                             key=lambda p: int(p.name.split("-")[1])) if sft_dir.exists() else []
        history = (read_json(checkpoints[-1] / "trainer_state.json", {}) or {}).get("log_history", []) \
            if checkpoints else []
        last = logged["sft"].get(str(n), 0)
        for entry in history:
            if entry.get("step", 0) <= last or not any(k in entry for k in ("loss", "eval_loss")):
                continue
            values = {f"sft/{k}": v for k, v in entry.items()
                      if k in ("loss", "eval_loss", "grad_norm", "learning_rate", "mean_token_accuracy",
                               "eval_mean_token_accuracy", "entropy")}
            wandb.log({"sft_step": offset + entry["step"], "sft/round": n, **values})
            logged["sft"][str(n)] = max(logged["sft"].get(str(n), 0), entry["step"])
        offset += int(sft_steps.get(str(n), 0))

    # Live: the round in progress.
    hours = (time.time() - state.get("started", time.time())) / 3600
    current = None if state.get("rounds_stopped") else next(
        (n for n in range(1, plan.get("rounds", 0) + 1) if f"round_{n}/evaluate" not in done), None)
    live = {"hours": hours, "live/rounds_evaluated": len([n for n in results if n != "0"]),
            "live/stages_done": len(done), "live/failures": sum(state.get("failures", {}).values())}
    if current is not None:
        gen = run_dir / f"round_{current}" / "gen"
        if gen.exists():
            tail = tails.setdefault(current, SlotTail(gen)).update()
            started = (state.get("generate_deadline") or {}).get(str(current))
            started = started - plan["sample"].get("max_hours", 0) * 3600 if started else None
            live.update({"live/round": current,
                         "live/unhinted_problems": len(tail.problems["none"]),
                         "live/unhinted_solved": len(tail.solved["none"]),
                         "live/unhinted_detect_per_attempt": tail.detected["none"] / max(tail.slots["none"], 1),
                         "live/hinted_attempts": sum(v for k, v in tail.slots.items() if k != "none"),
                         "live/rationalized_problems": len(set().union(
                             *[s for k, s in tail.solved.items() if k != "none"]) - tail.solved["none"]),
                         "live/attempts": sum(tail.slots.values()),
                         "live/generated_tokens": tail.tokens})
            if started and f"round_{current}/generate" not in done:
                live["live/attempts_per_hour"] = sum(tail.slots.values()) / max((time.time() - started) / 3600, 1e-6)
    wandb.log(live)
    stage = "finished" if state.get("finished") else "halted" if state.get("halted") else \
        (f"round {current}" if current else "gdpo" if "gdpo" not in done else "")
    run.summary["status"] = stage
    if state.get("best"):
        run.summary["best_round"] = state["best"]["round"]
        run.summary["best_adapter"] = state["best"]["adapter"]
    write_json(run_dir / "wandb_logged.json", logged)
    return bool(state.get("finished") or state.get("halted"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--project", default="rft-training")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--every", type=int, default=0, help="Seconds between syncs; 0 syncs once")
    parser.add_argument("--completions", type=int, default=3, help="Kept and dropped trajectories per round")
    args = parser.parse_args()
    import wandb

    run_dir = Path(args.run_dir).resolve()
    id_path = run_dir / "wandb_run_id"
    if not id_path.exists():
        id_path.write_text(wandb.util.generate_id() + "\n")
    (run_dir / "wandb_logger.pid").write_text(f"{__import__('os').getpid()}\n")
    run = wandb.init(project=args.project, entity=args.entity, id=id_path.read_text().strip(),
                     resume="allow", name=run_dir.name, dir=str(run_dir),
                     config={"plan": read_json(run_dir / "plan.json", {}), "run_dir": str(run_dir)})
    for axis, pattern in (("round", "round/*"), ("hours", "live/*"), ("sft_step", "sft/*")):
        wandb.define_metric(axis)
        wandb.define_metric(pattern, step_metric=axis)
    logged = read_json(run_dir / "wandb_logged.json", None) or {"rounds": [], "completions": [], "sft": {}}
    tails = {}
    while True:
        try:
            ended = sync(run, run_dir, logged, tails, args.completions)
        except Exception as exc:  # a half-written file; the next pass sees it whole
            print(f"[wandb_log] sync failed: {exc!r}", flush=True)
            write_json(run_dir / "wandb_logged.json", logged)  # what did go out is not sent again
            ended = False
        print(f"[wandb_log] {time.strftime('%F %T')} synced", flush=True)
        if ended or not args.every:
            break
        time.sleep(args.every)
    run.finish()


if __name__ == "__main__":
    main()
