#!/usr/bin/env python3
"""Replay saved fixed-evaluation slots under the Language-of-Test protocol.

Does not load an LLM. Native reference checks never fall back to Python.
Outputs are exclusive, self-contained, and sufficient to recompute summaries.
Run --help for the audit and replay commands. See docs/LANGUAGE_OF_TEST_EXECUTION.md.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time

from lot_metrics import compact, coverage_curve, outcomes, parse_answer, summarize, interval

ROOT = Path(__file__).resolve().parents[2]
PREPROCESSING = ROOT.parent / "data_preprocessing"
sys.path.insert(0, str(PREPROCESSING))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def raw_netlist(example):
    net = example["netlist"]
    return net["netlist"] if isinstance(net, dict) else net


def load_manifest(path):
    m = json.loads(Path(path).read_text())
    if digest(m["examples"]) != m["examples_sha256"]:
        raise ValueError("Fixed evaluation manifest checksum mismatch")
    if len({e["_fixed_eval_id"] for e in m["examples"]}) != len(m["examples"]):
        raise ValueError("Duplicate example IDs")
    return m


class Simulator:
    def __init__(self):
        import regex
        from fault_sim import OptimizedNetlist, fast_fault_sim
        from tetramax_backend import SimulationNetlist
        self.metadata = SimulationNetlist
        self.optimized = OptimizedNetlist
        self.fast = fast_fault_sim
        # Read the existing parser declarations without importing the ML package.
        source = ROOT / "atpgllm/training/reward_function_factory.py"
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RewardFunctionFactory")
        scope = {"re": regex}
        for node in cls.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("DECL_RE", "NAME_RE") for t in node.targets):
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
        self.decl, self.name = scope["DECL_RE"], scope["NAME_RE"]
        self.config = ROOT / "atpgllm/training/data/sim_config.json"
        self.gates = json.loads(self.config.read_text())["gate_funcs"]
        self.models, self.cache = {}, {}
        self.requests = self.executions = self.cache_hits = 0

    def model(self, text):
        key = digest(text)
        if key not in self.models:
            meta = self.metadata(text)
            fast = self.optimized(text, self.gates, self.decl, self.name)
            if set(meta.input_nets) != set(fast.input_nets) or set(meta.output_nets) != set(fast.output_nets):
                raise ValueError("Native/Python canonical port sets disagree")
            self.models[key] = meta, fast
        return self.models[key]

    def run(self, text, vector, fault):
        self.requests += 1
        key = digest([text, vector, fault])
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        self.executions += 1
        meta, model = self.model(text)
        frame = self.fast(vector, dict.fromkeys(meta.output_nets, 0), fault, model, self.gates)
        if frame is None or "error" in frame.columns:
            raise ValueError("Python simulator failed")
        good, bad = {}, {}
        for name in meta.output_nets:
            g, b = frame.loc[name, ["Good Machine", "Bad Machine"]]
            if g not in (0, 1) or b not in (0, 1):
                raise ValueError("Nonbinary canonical output")
            good[name], bad[name] = int(g), int(b)
        self.cache[key] = good, bad
        return good, bad


def reference(text, vector, fault, output_names):
    from tetramax_backend import make_request, simulate
    started = time.monotonic()
    try:
        result = simulate(make_request(text, vector, dict.fromkeys(output_names, 0), fault))
        if result["status"] != "ok" or result.get("indeterminate"):
            raise ValueError("Unknown/indeterminate native verification")
        good = {n: result["values"][n][0] for n in output_names}
        bad = {n: result["values"][n][1] for n in output_names}
        return {"status": "verified", "good": good, "bad": bad,
                "provenance": {k: result.get(k) for k in ("backend", "schema", "tool_version", "identity", "native_fault", "cache_hit", "fault_status")},
                "seconds": time.monotonic() - started}
    except Exception as exc:
        return {"status": "unknown", "error": str(exc), "seconds": time.monotonic() - started}


def freeze_faults(manifest, sim):
    circuits = {}
    for example in manifest["examples"]:
        text = raw_netlist(example)
        cid = digest(text)
        if cid in circuits:
            continue
        model = sim.metadata(text)
        # Full declared-net stems, both polarities; aliases are explicitly raw
        # labeled faults, not claimed independent physical representatives.
        circuits[cid] = {"module_name": example.get("module_name"), "netlist": text,
            "inputs": model.input_nets, "outputs": model.output_nets,
            "faults": [f"sa{bit} {n}" for n in sorted(model.all_nets) for bit in (0, 1)],
            "alias_groups": model.net_aliases}
    return {"fault_model": "single stuck-at; all declared-net stems; sa0+sa1; uncollapsed labeled faults; no branches",
            "constraints": "all binary primary inputs; all primary outputs observed",
            "circuits": circuits}


def replay(args):
    manifest = load_manifest(args.manifest)
    payload = json.loads(args.records.read_text())
    if payload["examples_sha256"] != manifest["examples_sha256"] or payload["protocol"] != manifest["protocol"]:
        raise ValueError("Saved records and manifest use different protocols")
    examples = {e["_fixed_eval_id"]: e for e in manifest["examples"]}
    groups = defaultdict(list)
    for row in payload["records"]:
        if row["example_id"] not in examples:
            raise ValueError("Unknown example in saved records")
        groups[row["example_id"]].append(row)
    n = manifest["protocol"]["generations"]
    if any(len(g) > n for g in groups.values()):
        raise ValueError("Duplicated completion slots")
    args.output.mkdir(parents=True, exist_ok=False)
    sim = Simulator()
    frozen = freeze_faults(manifest, sim)
    write(args.output / "fault_manifest.json", frozen)
    write(args.output / "source_manifest.json", manifest)
    write(args.output / "source_records.json", payload)
    provenance = {"manifest_sha256": file_hash(args.manifest), "records_sha256": file_hash(args.records),
        "fault_manifest_sha256": digest(frozen), "gate_config_sha256": file_hash(sim.config),
        "code_sha256": {str(p.relative_to(ROOT.parent)): file_hash(p) for p in
            [Path(__file__), Path(__file__).with_name("lot_metrics.py"), PREPROCESSING / "fault_sim.py", PREPROCESSING / "tetramax_backend.py"]},
        "protocol": manifest["protocol"], "checkpoint_step": payload["step"],
        "initial_checkpoint": payload.get("initial_checkpoint"), "reference": args.reference,
        "retry_policy": "zero retries; unknown retained; no fallback",
        "generation_cost": "unavailable in historical records; replay costs are offline audit only",
        "generation_terminal_status": "unavailable in historical records; field validity is not a truncation audit",
        "seed": args.seed, "random_vectors_per_circuit": args.random_vectors,
        "ci_scope": "exploratory circuit bootstrap, not proven independent families or training seeds"}
    checkpoint = ROOT / payload["initial_checkpoint"] if payload.get("initial_checkpoint") else None
    if payload["step"] != 0:
        checkpoint = args.records.parent.parent / f"checkpoint-{payload['step']}"
    if payload.get("evaluated_checkpoint"):
        checkpoint = Path(payload["evaluated_checkpoint"])
    provenance["checkpoint"] = str(checkpoint) if checkpoint else None
    provenance["checkpoint_files"] = {str(p.relative_to(checkpoint)): file_hash(p) for p in
        sorted(checkpoint.glob("**/adapter*")) if p.is_file()} if checkpoint else {}
    write(args.output / "provenance.json", provenance)
    rows, reference_rows = [], []
    started = time.monotonic()
    with (args.output / "slots.jsonl").open("x") as stream:
        for eid, example in examples.items():
            text = raw_netlist(example)
            cid = digest(text)
            meta = sim.metadata(text)
            for slot in range(n):
                saved = groups[eid][slot] if slot < len(groups[eid]) else None
                completion = saved["completion"] if saved else ""
                if not isinstance(completion, str):
                    raise ValueError("Expected a saved text completion")
                parsed = parse_answer(completion, meta.input_nets, meta.output_nets)
                if saved and saved.get("terminal_status", "stop") != "stop":
                    parsed = {"input": None, "expected": None, "errors": [saved["terminal_status"]]}
                row = {"example_id": eid, "circuit_id": cid, "module_name": example.get("module_name"),
                    "fault": example["fault"], "slot": slot, "raw_answer": completion, "parsed": parsed,
                    "slot_status": saved.get("terminal_status", "saved") if saved else "missing",
                    "candidate_origin": "model_no_feedback" if manifest["protocol"].get("prompt_pipeline") == "lot-no-feedback-v1" else "historical_model_with_tools",
                    "tool_calls": completion.count("<tool_call>"), "tool_responses": completion.count("<tool_response>"),
                    "saved_components": saved.get("components", {}) if saved else {}}
                good = bad = None
                if parsed["input"] is not None:
                    try:
                        good, bad = sim.run(text, parsed["input"], example["fault"])
                    except Exception as exc:
                        row["internal_error"] = str(exc)
                row.update(outcomes(parsed, good, bad))
                row.update(good=good, bad=bad)
                ref = {"status": "not_requested"}
                if args.reference == "tetramax":
                    ref = (reference(text, parsed["input"], example["fault"], meta.output_nets)
                           if parsed["input"] is not None else {"status": "invalid_input"})
                    scored = outcomes(parsed, ref.get("good"), ref.get("bad"))
                    reference_rows.append({**row, **scored})
                    ref["outcomes"] = scored
                row["reference"] = ref
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
            if len(rows) % 8 == 0:
                print(f"Replayed {len(rows)}/{len(examples) * n} slots", flush=True)
    result = {"status": "complete" if sum(map(len, groups.values())) == len(examples) * n else "incomplete_source",
        "split": manifest["protocol"]["split"], "primary_claim": "retrospective validation pilot; not a locked test benchmark",
        "internal": summarize(rows, args.k, seed=args.seed), "reference": None,
        "reference_statuses": dict(Counter(r["reference"]["status"] for r in rows))}
    if reference_rows:
        result["reference"] = summarize(reference_rows, args.k, seed=args.seed)
        comparable = [(r, s) for r, s in zip(rows, reference_rows) if r["verification_known"] and s["verification_known"]]
        confusion = Counter(f"internal_{int(r['D'])}_reference_{int(s['D'])}" for r, s in comparable)
        accepted = sum(r["D"] for r, s in comparable)
        result["audit"] = {"detection_confusion": dict(confusion), "comparable_slots": len(comparable),
            "good_output_disagreements": sum(r["good"] != r["reference"]["good"] for r, s in comparable),
            "detection_disagreements": sum(r["D"] != s["D"] for r, s in comparable),
            "false_acceptance_among_reference_known_internal_accepts": confusion["internal_1_reference_0"] / accepted if accepted else None,
            "internally_accepted_reference_unknown": sum(r["D"] and not s["verification_known"] for r, s in zip(rows, reference_rows)),
            "independence": ("Python vs TetraMAX implementations; TetraMAX also supplied tool feedback, so native replay alone is not independent of that feedback"
                if manifest["protocol"].get("fault_sim_backend") == "tetramax" else
                "Native TetraMAX replay independent of the Python simulator used for original tool feedback and reward")}
    result["offline_replay_seconds"] = time.monotonic() - started
    if args.coverage:
        result["coverage"] = coverage(args, frozen, rows, sim)
    result["offline_total_seconds"] = time.monotonic() - started
    result["internal_simulator"] = {"requests": sim.requests, "executions": sim.executions, "cache_hits": sim.cache_hits}
    write(args.output / "summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "coverage"}, indent=2), flush=True)


def coverage(args, frozen, rows, sim):
    """Internal all-fault replay and uniform control. Never calls this independent FC."""
    reports = {}
    for cid, circuit in frozen["circuits"].items():
        text, faults = circuit["netlist"], circuit["faults"]
        rng = random.Random(int(digest([args.seed, cid])[:16], 16))
        vectors = [{n: rng.randrange(2) for n in circuit["inputs"]} for _ in range(args.random_vectors)]
        circuit_rows = [r for r in rows if r["circuit_id"] == cid]
        model_vectors = [r["parsed"]["input"] for r in circuit_rows]
        patterns = {digest(v): v for v in model_vectors + vectors if v is not None}
        matrix, failures = {}, []
        for vid, vector in patterns.items():
            detected = []
            for fault in faults:
                try:
                    good, bad = sim.run(text, vector, fault)
                    if good != bad:
                        detected.append(fault)
                except Exception as exc:
                    failures.append({"vector_id": vid, "fault": fault, "error": str(exc)})
            matrix[vid] = detected
        model_sets = [matrix[digest(v)] if v is not None else [] for v in model_vectors]
        deployable = [det if r["S"] else [] for r, det in zip(circuit_rows, model_sets)]
        random_sets = [matrix[digest(v)] for v in vectors]
        witnessed = set().union(*map(set, matrix.values()))
        streams = {}
        for method, sets, vecs in (("model_stimulus", model_sets, model_vectors),
                                    ("model_raw_deployable", deployable, model_vectors),
                                    ("uniform_stimulus", random_sets, vectors)):
            curve = coverage_curve(sets, faults)
            chosen = compact(sets)
            used = {digest(v) for v, det in zip(vecs, sets) if v is not None}
            distinct_sets = [set(matrix[vid]) for vid in used]
            if method == "model_raw_deployable":
                used = {digest(v) for v, r in zip(vecs, circuit_rows) if v is not None and r["S"]}
                distinct_sets = [set(matrix[vid]) for vid in used]
            streams[method] = {"ordered_vector_ids": [digest(v) if v is not None else None for v in vecs],
                "curve": curve, "unique_vectors": len(used), "total_slots": len(vecs),
                "compacted_indices": chosen, "compacted_patterns": len(chosen),
                "compaction_policy": "deterministic greedy at this stream's final coverage; cross-method counts not equal-coverage claims",
                "Ndetect": {str(n): sum(sum(f in ds for ds in distinct_sets) >= n for f in faults) / len(faults) for n in (1, 2, 4, 8)}}
        # Independent uniform sample estimates per-target difficulty, not a
        # matched wall-time control and never a requirement to guess PO bits.
        difficulty = []
        for r in circuit_rows:
            hits = sum(r["fault"] in ds for ds in random_sets)
            difficulty.append({"example_id": r["example_id"], "fault": r["fault"],
                "random_hits": hits, "random_n": len(vectors), "D_probability": hits / len(vectors)})
        report = {"module_name": circuit["module_name"], "faults": len(faults),
            "verification": "Python only; provisional pending full independent matrix replay",
            "T_witnessed_internal": len(witnessed), "Z_proven": 0, "R_unresolved": len(faults) - len(witnessed),
            "patterns": patterns, "detected_fault_matrix": matrix, "simulation_failures": failures,
            "streams": streams, "random_target_difficulty": difficulty}
        write(args.output / f"coverage-{cid}.json", report)
        reports[cid] = {k: v for k, v in report.items() if k not in ("patterns", "detected_fault_matrix", "streams", "simulation_failures")}
        reports[cid]["simulation_failure_count"] = len(failures)
        reports[cid]["final_coverage"] = {m: s["curve"][-1]["FC_all"] if s["curve"] else 0 for m, s in streams.items()}
        reports[cid]["model_minus_uniform_expected_D"] = sum(r["D"] - d["D_probability"] for r, d in zip(circuit_rows, difficulty)) / len(circuit_rows)
        print(f"Coverage {circuit['module_name']}: {len(faults)} faults, {len(patterns)} unique audit vectors, {len(failures)} unknown cells", flush=True)
        sim.cache.clear()
    deltas = [r["model_minus_uniform_expected_D"] for r in reports.values()]
    return {"circuits": reports, "random_comparison_scope": "exploratory stimulus-only; matched pattern prefixes, no matched generation costs; Monte Carlo difficulty uncertainty not included in circuit CI",
            "model_minus_uniform_expected_D_macro": sum(deltas) / len(deltas),
            "difference_circuit_bootstrap_95": interval(deltas, seed=args.seed),
            "macro_FC_all": {m: sum(r["final_coverage"][m] for r in reports.values()) / len(reports)
                for m in ("model_stimulus", "model_raw_deployable", "uniform_stimulus")}}


def audit_split(args):
    """Audit recorded source identities; do not infer equivalence from text hashes."""
    manifest = json.loads((args.dataset / "split_manifest.json").read_text())
    accepted = [c for c in manifest["circuits"] if c.get("split")]
    splits = sorted({c["split"] for c in accepted})
    result = {"dataset": str(args.dataset.resolve()), "manifest_sha256": file_hash(args.dataset / "split_manifest.json"),
        "scope": "source split manifest; actual parquet content audited below",
        "splits": {}, "overlap": {}, "family_disjointness": "not established by circuit/text identities",
        "locked_test_split_available": "test" in splits}
    for split in splits:
        members = [c for c in accepted if c["split"] == split]
        result["splits"][split] = {"source_designs": len(members), "rows_recorded": sum(c["rows"] for c in members),
            "unique_circuit_ids": len({c["circuit_id"] for c in members}), "unique_netlist_ids": len({c["netlist_id"] for c in members})}
    for field in ("circuit_id", "netlist_id", "module_name", "source_module_name"):
        train = {c[field] for c in accepted if c["split"] == "train"}
        other = {c[field] for c in accepted if c["split"] != "train"}
        result["overlap"][field] = sorted(train & other)
    if args.read_parquet:
        import pyarrow.parquet as pq
        raw = defaultdict(set)
        counts = Counter()
        for split in splits:
            for i, path in enumerate(sorted((args.dataset / split).glob("*.parquet"))):
                for batch in pq.ParquetFile(path).iter_batches(columns=["netlist"], batch_size=1024):
                    nets = batch.column(0).to_pylist()
                    counts[split] += len(nets)
                    raw[split].update(hashlib.sha256(n.strip().replace("\r\n", "\n").encode()).hexdigest() for n in set(nets))
                if i % 250 == 0:
                    print(f"Split audit {split}: {i + 1} shards, {counts[split]} rows", flush=True)
        result["actual_parquet"] = {"row_counts": dict(counts), "unique_text_hashes": {k: len(v) for k, v in raw.items()},
            "train_validation_text_overlap": sorted(raw["train"] & raw["validation"]),
            "limitation": "text hashes do not exclude renamed/resynthesized/family duplicates"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, result)
    print(json.dumps(result, indent=2))


def audit_coverage(args):
    """Independently replay every fault for model and matched-prefix controls."""
    from concurrent.futures import ThreadPoolExecutor
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = json.loads((args.replay / "fault_manifest.json").read_text())
    slots = [json.loads(line) for line in (args.replay / "slots.jsonl").read_text().splitlines()]
    prior = {}
    if args.reuse_reference:
        previous = json.loads((args.reuse_reference / "provenance.json").read_text())
        if previous["fault_manifest_sha256"] != digest(frozen):
            raise ValueError("Cannot reuse reference matrix for a different fault manifest")
        for line in (args.reuse_reference / "reference_matrix.jsonl").read_text().splitlines():
            row = json.loads(line)
            prior[row["circuit_id"], row["vector_id"], row["fault"]] = row
    provenance = {"source": str(args.replay.resolve()), "fault_manifest_sha256": digest(frozen),
        "workers": args.workers, "backend": "tetramax", "retry_policy": "zero retries; unknown retained; no fallback",
        "budget_match": "uniform prefix has same emitted slot count as each circuit's model stream; generation time unavailable",
        "code_sha256": file_hash(Path(__file__)), "reference_function_sha256": file_hash(PREPROCESSING / "tetramax_backend.py")}
    if args.reuse_reference:
        provenance["reuse_reference"] = {"path": str(args.reuse_reference.resolve()),
            "matrix_sha256": file_hash(args.reuse_reference / "reference_matrix.jsonl"),
            "meaning": "Previously measured outcomes, including unknowns; not fresh verification or a retry"}
    write(args.output / "provenance.json", provenance)
    summaries = {}
    started = time.monotonic()
    with (args.output / "reference_matrix.jsonl").open("x") as stream, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for cid, circuit in frozen["circuits"].items():
            coverage = json.loads((args.replay / f"coverage-{cid}.json").read_text())
            model_ids = coverage["streams"]["model_stimulus"]["ordered_vector_ids"]
            random_ids = coverage["streams"]["uniform_stimulus"]["ordered_vector_ids"][:len(model_ids)]
            if len(random_ids) != len(model_ids):
                raise ValueError("Insufficient uniform vectors for matched pattern comparison")
            vector_ids = sorted(set(model_ids + random_ids) - {None})
            jobs = [(vid, fault) for vid in vector_ids for fault in circuit["faults"]]
            def execute(job):
                vid, fault = job
                if (cid, vid, fault) in prior:
                    return {**prior[cid, vid, fault], "reused_previous_measurement": True}
                return {"circuit_id": cid, "vector_id": vid, "fault": fault,
                    **reference(circuit["netlist"], coverage["patterns"][vid], fault, circuit["outputs"])}
            detected = {vid: [] for vid in vector_ids}
            unknown = {vid: [] for vid in vector_ids}
            good_outputs, errors, counts = {}, [], Counter()
            false_accepted, disagreement = [], []
            for i, result in enumerate(pool.map(execute, jobs), 1):
                stream.write(json.dumps(result) + "\n")
                stream.flush()
                counts[result["status"]] += 1
                vid, fault = result["vector_id"], result["fault"]
                if result["status"] == "verified":
                    if vid in good_outputs and good_outputs[vid] != result["good"]:
                        raise ValueError("Fault injection changed reference good-machine outputs")
                    good_outputs[vid] = result["good"]
                    reference_detects = result["good"] != result["bad"]
                    internal_detects = fault in coverage["detected_fault_matrix"][vid]
                    if reference_detects:
                        detected[vid].append(fault)
                    if reference_detects != internal_detects:
                        disagreement.append({"vector_id": vid, "fault": fault, "internal": internal_detects, "reference": reference_detects})
                        if internal_detects:
                            false_accepted.append(disagreement[-1])
                else:
                    errors.append(result)
                    unknown[vid].append(fault)
                if i % 100 == 0:
                    print(f"Reference matrix {circuit['module_name']}: {i}/{len(jobs)} cells", flush=True)
            circuit_slots = [s for s in slots if s["circuit_id"] == cid]
            model_sets = [detected[vid] if vid is not None else [] for vid in model_ids]
            usable_sets = [ds if s["parsed"]["expected"] is not None and s["parsed"]["expected"] == good_outputs.get(vid) else []
                           for ds, s, vid in zip(model_sets, circuit_slots, model_ids)]
            curves = {"model_stimulus": coverage_curve(model_sets, circuit["faults"]),
                      "model_raw_deployable": coverage_curve(usable_sets, circuit["faults"]),
                      "uniform_stimulus": coverage_curve([detected[vid] for vid in random_ids], circuit["faults"])}
            model_possible = [detected[vid] + unknown[vid] if vid is not None else [] for vid in model_ids]
            deployable_possible = [ds if s["parsed"]["expected"] is not None and
                (vid not in good_outputs or s["parsed"]["expected"] == good_outputs[vid]) else []
                for ds, s, vid in zip(model_possible, circuit_slots, model_ids)]
            upper_curves = {"model_stimulus": coverage_curve(model_possible, circuit["faults"]),
                "model_raw_deployable": coverage_curve(deployable_possible, circuit["faults"]),
                "uniform_stimulus": coverage_curve([detected[vid] + unknown[vid] for vid in random_ids], circuit["faults"])}
            compacted = {"model_stimulus": compact(model_sets), "model_raw_deployable": compact(usable_sets),
                         "uniform_stimulus": compact([detected[vid] for vid in random_ids])}
            witnessed = set().union(*map(set, detected.values()))
            model_covered = set().union(*map(set, model_sets))
            summaries[cid] = {"module_name": circuit["module_name"], "faults": len(circuit["faults"]),
                "curve_axis": "completion slots, including invalid slots; not physical test-application cycles",
                "compaction_scope": "known detections at each stream's own final coverage; not an equal-coverage comparison",
                "reference_counts": dict(counts), "curves": curves, "compacted_indices": compacted,
                "final_FC_all": {k: v[-1]["FC_all"] for k, v in curves.items()},
                "FC_all_interpretation": "verified lower bound when any matrix outcomes are unknown",
                "FC_all_bounds": {k: [v[-1]["FC_all"], upper_curves[k][-1]["FC_all"]] for k, v in curves.items()},
                "upper_curves": upper_curves,
                "T_witnessed": len(witnessed), "Z_proven": 0, "R_unresolved": len(circuit["faults"]) - len(witnessed),
                "model_testable_coverage_bounds": None if errors else [len(model_covered) / len(circuit["faults"]), len(model_covered) / len(witnessed) if witnessed else None],
                "errors": errors, "disagreements": disagreement, "internal_false_accepts": false_accepted,
                "complete_matrix": not errors}
            print(f"Reference coverage complete: {circuit['module_name']}, {dict(counts)}, {len(disagreement)} disagreements", flush=True)
    write(args.output / "summary.json", {"circuits": summaries,
        "status": "complete" if all(s["complete_matrix"] for s in summaries.values()) else "incomplete_reference_lower_bounds",
        "offline_audit_seconds": time.monotonic() - started,
        "macro_FC_all": {m: sum(s["final_FC_all"][m] for s in summaries.values()) / len(summaries)
                         for m in ("model_stimulus", "model_raw_deployable", "uniform_stimulus")},
        "macro_FC_all_bounds": {m: [sum(s["FC_all_bounds"][m][i] for s in summaries.values()) / len(summaries) for i in (0, 1)]
                                for m in ("model_stimulus", "model_raw_deployable", "uniform_stimulus")}})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("replay")
    r.add_argument("--manifest", type=Path, required=True)
    r.add_argument("--records", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--reference", choices=("none", "tetramax"), default="none")
    r.add_argument("--coverage", action="store_true")
    r.add_argument("--random-vectors", type=int, default=64)
    r.add_argument("--seed", type=int, default=1729)
    r.add_argument("--k", nargs="+", type=int, default=[1])
    r.set_defaults(func=replay)
    a = sub.add_parser("audit-split")
    a.add_argument("--dataset", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--read-parquet", action="store_true")
    a.set_defaults(func=audit_split)
    c = sub.add_parser("audit-coverage")
    c.add_argument("--replay", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--workers", type=int, choices=(1, 2), default=2)
    c.add_argument("--reuse-reference", type=Path, help="Explicitly reuse prior measured matrix cells, retaining their provenance and unknowns")
    c.set_defaults(func=audit_coverage)
    args = p.parse_args()
    if getattr(args, "random_vectors", 1) < 1:
        p.error("--random-vectors must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
