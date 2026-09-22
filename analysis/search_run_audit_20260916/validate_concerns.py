"""Recheck the six named artifacts and test three circuits with hand-derived logic.

Run: /usr/bin/python3 validate_concerns.py [--scan-training]
No model, production parser, or production simulator is imported. Historical
uniform probes and constant-vector results are explicitly read from the prior
audit, not represented as newly independent measurements. Writes a separate
concerns_validation.json, preserving all original audit artifacts.
"""
import argparse
import ast
import collections
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import re

import numpy as np
import yaml

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
RUNS = {
    "mcts": "run-20260915_104906-8rvdaqde",
    "best_of_n": "run-20260915_160758-snqbdxoi",
    "evolutionary": "run-20260915_163135-ipt1fjf7",
    "greedy": "run-20260915_234409-dms52ruj",
    "random": "run-20260916_124725-johktkni",
    "vector_evolutionary": "run-20260916_125121-fkmhdl8w",
}
BAD_ROWS = {108, 321, 395, 474}


def passk(c, k):
    return 1 - math.comb(16-c, k)/math.comb(16, k) if 16-c >= k else 1.0


def fields(text):
    text = text.rsplit("</tool_response>", 1)[-1].rsplit("</think>", 1)[-1]
    return dict(re.findall(r'\b(INPUT_VECTOR|EXPECTED_OUTPUT|DETECTED_FAULTS):\s*"([^"]*)"', text))


def assignment(text):
    pairs = [item.rsplit(":", 1) for item in text.split(",")]
    result = {k.strip(): int(v) for k, v in pairs}
    assert len(result) == len(pairs) and set(result.values()) <= {0, 1}
    return result


def raw_netlist(row):
    prompt = next(m["content"] for m in row["search_slots"][0]["messages"] if m["role"] == "user")
    match = re.search(r"['\"]netlist['\"]\s*:\s*('(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\")", prompt)
    return ast.literal_eval(match.group(1))


# These equations were derived directly from the three saved Verilog netlists.
# They share neither parsing nor gate truth tables with the production simulator.
BUS_CONES = {
    "s3": {f"r{i}out" for i in range(8, 16)},
    "s2": {f"r{i}out" for i in [4, 5, 6, 7, 12, 13, 14, 15]} | {"cout", "mdrout", "inportout", "pcout"},
    "s1": {f"r{i}out" for i in [2, 3, 6, 7, 10, 11, 14, 15]} | {"cout", "zlowout", "zhighout", "inportout"},
    "s0": {f"r{i}out" for i in [1, 3, 5, 7, 9, 11, 13, 15]} | {"cout", "zlowout", "lowout", "mdrout"},
    "s4": {"cout", "mdrout", "inportout", "pcout", "zlowout", "lowout", "zhighout", "hiout"},
}


def logic(idx, v):
    if idx == 508:
        return {f"p{i}y": int(not all(v[f"p{i}{k}"] for k in "abcd")) for i in [1, 2]}
    if idx == 419:
        return {out: int(any(v[k] for k in cone)) for out, cone in BUS_CONES.items()}
    if idx == 259:
        addr = sum(v[f"addr[{i}]"] << i for i in range(4))
        outputs = {f"rd_ram_or_io_dat[{i}]": int((addr == 2 and v[f"rd_ram_dat[{i}]"])
                   or (addr == 4 and v[f"rd_io_dat[{i}]"])) for i in range(32)}
        outputs.update(ram_wr=int(addr == 2 and v["ram_or_io_wr"]),
                       io_wr=int(addr == 4 and v["ram_or_io_wr"]))
        return outputs
    raise ValueError(idx)


def exact_bus_probability():
    # r15out must be 1 and at least one of its four OR cones must have
    # every other input 0. Inclusion-exclusion handles overlapping cones.
    cones = [c - {"r15out"} for c in BUS_CONES.values() if "r15out" in c]
    return .5 * sum((-1)**(n+1) * 2.0**(-len(set.union(*subset)))
                   for n in range(1, 5) for subset in itertools.combinations(cones, n))


def interval(delta, rng):
    delta = np.asarray(delta)
    draws = rng.integers(0, len(delta), (10000, len(delta)))
    return {"delta_pp": float(delta.mean()*100),
            "ci95_pp": (np.quantile(delta[draws].mean(axis=1), [.025, .975])*100).tolist(),
            "wins_ties_losses": [int((delta > 0).sum()), int((delta == 0).sum()), int((delta < 0).sum())]}


def main(scan_training=False):
    prior = json.loads((OUT/"analysis.json").read_text())
    detail = list(csv.DictReader((OUT/"per_problem.csv").open()))
    result = {"runs": {}, "independent_examples": {}, "comparisons": {}, "strata": {}}
    ids, counts, eval_hashes = None, {}, {}
    manifest_path = ROOT/"runs/grpo_granite_4.2_8b/checkpoint-50/fixed_eval_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    checksum = hashlib.sha256(json.dumps(manifest["examples"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    assert checksum == manifest["examples_sha256"]
    heldout = {hashlib.sha256(e["netlist"]["netlist"].encode()).hexdigest(): e["fault"] for e in manifest["examples"]}
    for method, run in RUNS.items():
        folder = ROOT/"wandb"/run/"files"
        config = yaml.safe_load((folder/"config.yaml").read_text())
        meta = json.loads((folder/"wandb-metadata.json").read_text())
        summary = json.loads((folder/"wandb-summary.json").read_text())
        args = meta["args"]
        data_path = ROOT/"runs/eval_results_grpo_granite_4.2_8b_policy"/Path(args[args.index("--output_file")+1]).name
        raw = data_path.read_bytes()
        data = json.loads(raw)
        rows = data["per_problem_results"]
        assert data["config"]["sampling_method"] == config["sampling_method"]["value"] == method
        row_ids = [r["search_slots"][0]["problem_id"] for r in rows]
        if ids is None:
            ids = row_ids
        assert len(rows) == len(set(row_ids)) == 512 and row_ids == ids
        counts[method] = [r["num_correct"] for r in rows]
        slots = [s for r in rows for s in r["search_slots"]]
        assert len(slots) == 8192
        for i, r in enumerate(rows):
            assert r["num_correct"] == sum(s["reward_components"].get("detection", 0) for s in r["search_slots"])
            assert r["num_correct"]/16 == float(detail[i][method])
        metrics = {f"pass@{k}": sum(passk(c, k) for c in counts[method])/512 for k in [1, 2, 4, 8, 16]}
        for k, val in metrics.items():
            assert abs(val-data["pass_at_k"][k]) < 1e-12
            assert abs(val-summary[k]) < 1e-12
            assert abs(val-prior["runs"][method][k]) < 1e-12
        usage = {k: sum(s["usage"][k] for s in slots) for k in slots[0]["usage"]}
        assert usage == data["aggregate_metrics"]["search_usage"]
        result["runs"][method] = dict(run=run, artifact_sha256=hashlib.sha256(raw).hexdigest(),
            config=data["config"], pass_at_k=metrics, successes=sum(counts[method]),
            solved=sum(c > 0 for c in counts[method]), usage=usage,
            time_seconds=data["aggregate_metrics"]["total_eval_time_seconds"],
            full_accuracy=sum(all(s["reward_components"].get(k, 0) >= 1 for k in
                ["detection", "input_vector_acc_logonly", "expected_output_acc_logonly", "detected_faults_acc_logonly"])
                for s in slots)/8192,
            unique_vectors_at_four_attempts=dict(collections.Counter(s["unique_vectors"] for s in slots if s["usage"]["attempts"] == 4)))
        for idx in [259, 419, 508]:
            row = rows[idx]
            target = row["fault"].split()[1]
            stuck = int(row["fault"][2])
            stats = collections.Counter()
            witness = None
            for j, slot in enumerate(row["search_slots"]):
                if slot["status"] != "FINAL":
                    continue
                f = fields(slot.get("final_answer", row["completions"][j]))
                v = assignment(f["INPUT_VECTOR"])
                good, bad = logic(idx, v), logic(idx, {**v, target: stuck})
                detected = good != bad
                assert int(detected) == slot["reward_components"]["detection"]
                stats["checked"] += 1
                stats["detected"] += int(detected)
                stats["expected_output_correct"] += int(assignment(f["EXPECTED_OUTPUT"]) == good)
                if detected and witness is None:
                    witness = dict(vector=v, good_outputs=good, faulty_outputs=bad,
                                   reported_outputs=assignment(f["EXPECTED_OUTPUT"]))
            result["independent_examples"].setdefault(str(idx), {})[method] = dict(stats=stats, witness=witness)
        if method == "greedy":
            for i, row in enumerate(rows):
                raw_text = raw_netlist(row)
                eval_hashes[hashlib.sha256(raw_text.encode()).hexdigest()] = (i, row["fault"])
        print(f"Validated {method}: counts, W&B metrics, costs, three independent circuit equations", flush=True)
        del data, rows, slots, raw
    rng = np.random.default_rng(20260916)
    holdout_indices = sorted(i for h, (i, fault) in eval_hashes.items() if h in heldout)
    holdout_pair_indices = sorted(i for h, (i, fault) in eval_hashes.items() if heldout.get(h) == fault)
    result["checkpoint_holdout"] = dict(protocol=manifest["protocol"],
        manifest_checksum_verified=True, matching_circuit_indices=holdout_indices,
        matching_circuit_fault_indices=holdout_pair_indices,
        caveat="Recorded GRPO exclusion policy plus matching source code; actual buffer fingerprint and prior SFT exposure not reconstructed.")
    for a, b in [("greedy", "random"), ("best_of_n", "vector_evolutionary"),
                 ("evolutionary", "vector_evolutionary"), ("mcts", "best_of_n")]:
        for k in [1, 4, 16]:
            delta = [passk(ca, k)-passk(cb, k) for ca, cb in zip(counts[a], counts[b])]
            result["comparisons"][f"{a}-{b}@{k}"] = interval(delta, rng)
    for name, select in {
        "nonbus": lambda d: int(d["idx"]) not in BAD_ROWS,
        "hard_nonbus": lambda d: int(d["idx"]) not in BAD_ROWS and float(d["uniform_detection"]) <= .1,
        "hard_neither_constant_nonbus": lambda d: int(d["idx"]) not in BAD_ROWS and float(d["uniform_detection"]) <= .1 and d["zero_or_one"] == "0",
        "hard_neither_constant_nonpi_nonbus": lambda d: int(d["idx"]) not in BAD_ROWS and float(d["uniform_detection"]) <= .1 and d["zero_or_one"] == "0" and d["target_is_pi"] == "False",
        "neither_constant_nonbus": lambda d: int(d["idx"]) not in BAD_ROWS and d["zero_or_one"] == "0",
        "recorded_grpo_holdout_nonbus": lambda d: int(d["idx"]) in holdout_indices and int(d["idx"]) not in BAD_ROWS,
        "outside_recorded_grpo_holdout_nonbus": lambda d: int(d["idx"]) not in holdout_indices and int(d["idx"]) not in BAD_ROWS,
    }.items():
        subset = [d for d in detail if select(d)]
        result["strata"][name] = dict(n=len(subset),
            pass_at_1={m:sum(float(d[m]) for d in subset)/len(subset) for m in RUNS},
            greedy_minus_random=interval([float(d["greedy"])-float(d["random"]) for d in subset], rng))
    result["independent_exact_uniform_probabilities"] = {"259": 1/16, "419": exact_bus_probability(), "508": 1/16}
    if scan_training:
        import pyarrow as pa
        cache = Path('/home/eng/c/cxv200006/.cache/huggingface/datasets/chrivasileiou___asap7-language-of-test-v2/default/0.0.0/d35cfea64eadf30fb3b39735b0e8d20bffcc3345')
        files = sorted(cache.glob('*-train-*.arrow'))
        assert files
        matched, pairs, unique, total = set(), set(), set(), 0
        for fi, path in enumerate(files):
            with pa.memory_map(str(path), 'r') as source:
                for batch in pa.ipc.open_stream(source):
                    nets = batch.column(batch.schema.get_field_index('netlist')).to_pylist()
                    faults = batch.column(batch.schema.get_field_index('fault')).to_pylist()
                    for raw, fault in zip(nets, faults):
                        h = hashlib.sha256(raw.encode()).hexdigest()
                        total += 1
                        unique.add(h)
                        if h in eval_hashes:
                            matched.add(h)
                            if fault == eval_hashes[h][1]:
                                pairs.add(h)
            print(f"Training scan {fi+1}/{len(files)}", flush=True)
        result['cached_training_overlap'] = dict(rows=total, unique_netlists=len(unique),
            matching_netlists=len(matched), matching_netlist_fault_pairs=len(pairs),
            caveat='Local cached training split; actual checkpoint exposure remains unverified.')
    (OUT/"concerns_validation.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ['runs', 'independent_examples']}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-training", action="store_true")
    main(parser.parse_args().scan_training)
