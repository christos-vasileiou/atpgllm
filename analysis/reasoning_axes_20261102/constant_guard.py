"""Score the ladder guarded against constant vectors, from the saved attempts. CPU only.
Run from the libatpgllm_hard-faults root; writes constant_guard.json beside this file."""
import collections, json, os, sys
from pathlib import Path
os.environ["FAULT_SIM_BACKEND"] = "fast"
sys.path[:0] = [".", "scripts/star"]
import pyarrow.parquet as pq
import score_reward_axes as S
here = Path(__file__).parent
groups = collections.defaultdict(list)
for line in open(here / "attempts.jsonl"):
    a = json.loads(line)
    a["ladder"] = (a.get("activate_distance", 0) + a.get("detect_distance", 0)) / 2
    groups[a["round"], a["module"], a["fault"]].append(a)
scorer = S.Scorer(S.load_signals())
rows = {}
for n in sorted({k[0] for k in groups}):
    wanted = {(k[1], k[2]) for k in groups if k[0] == n}
    table = pq.read_table(f"runs/rft_v4_a100/round_{n}/problems.parquet", columns=["module_name", "fault", "netlist", "difficulty"]).to_pylist()
    for r in table:
        if (r["module_name"], r["fault"]) in wanted:
            rows[n, r["module_name"], r["fault"]] = r
out_groups, constant_detects, skipped = [], 0, 0
cache = {}
for key, attempts in groups.items():
    row = rows.get(key)
    if row is None:
        skipped += 1
        continue
    ck = (row["netlist"], row["fault"])
    if ck not in cache:
        net_list = scorer.simulator.compile(row["netlist"])
        best = 0.0
        detects = False
        for value in (0, 1):
            fake = {"vector": {p: value for p in net_list.input_nets}, "messages": [], "observations": [], "reward_components": {}}
            scored = scorer.score(row, fake)
            best = max(best, (scored["activate_distance"] + scored["detect_distance"]) / 2)
            detects = detects or scored.get("simulated_detect") == 1
        cache[ck] = (best, detects)
    best, detects = cache[ck]
    constant_detects += detects
    for a in attempts:
        a["constant_best"] = best
        a["guarded_ladder"] = max(0.0, a["ladder"] - best) / (1 - best) if best < 1 else 0.0
    out_groups.append((row["difficulty"], key[0], attempts))
result = {"groups": len(out_groups), "skipped": skipped, "constant_detects_share": constant_detects / max(len(out_groups), 1)}
for axis in ("ladder", "guarded_ladder", "constant_best"):
    stats = S.axis_statistics(out_groups, axis)
    result[axis] = {k: v for k, v in stats.items() if k in ("group_coverage", "groups_with_signal", "all_fail_groups", "all_fail_groups_with_signal", "failure_score_predicts_solved", "within_group_corr_detection", "detecting_at_group_max", "within_group_corr_length", "hard_mean", "hard_std", "by_round")}
by = collections.defaultdict(list)
for level, _, attempts in out_groups:
    by[level].extend(a for a in attempts)
result["by_level"] = {level: dict(n=len(v), ladder=sum(a["ladder"] for a in v) / len(v), constant=sum(a["constant_best"] for a in v) / len(v), guarded=sum(a["guarded_ladder"] for a in v) / len(v), beats_constant=sum(a["ladder"] > a["constant_best"] for a in v) / len(v), detected=sum(a["detected"] for a in v) / len(v)) for level, v in by.items()}
(here / "constant_guard.json").write_text(json.dumps(result, indent=1))
print(json.dumps(result, indent=1))
