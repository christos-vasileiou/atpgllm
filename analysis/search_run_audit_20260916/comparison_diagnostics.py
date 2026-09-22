"""Post-evaluation diagnostic strata and response-copying measurements.

Run after the full matrix finishes. This reads raw answers without repairing
them. Nonconstant-output strata were selected from simulation, before scores.
"""
import argparse
import ast
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval"))
from prepare_circuit_comparison import parser_and_tools


def assignment(value):
    if isinstance(value, dict):
        return {k: int(v) for k, v in value.items()}
    try:
        return {k.strip(): int(v.strip()) for k, v in
                (item.rsplit(":", 1) for item in value.split(","))}
    except (ValueError, AttributeError):
        return {}


def main(output):
    plan = json.loads((output / "plan.json").read_text())
    if any(not Path(j["output"]).exists() for j in plan["jobs"]):
        raise SystemExit("Comparison incomplete; no final diagnostic aggregation written")
    command = plan["jobs"][0]["command"]
    manifest = json.loads(Path(command[command.index("--eval_manifest")+1]).read_text())
    rows = {r["comparison_id"]: r for r in manifest["examples"]}
    baselines = {r["source_circuit_id"]: r for r in json.loads(
        (OUT / "comparison_output_baselines.json").read_text())["circuits"]}
    parse, _ = parser_and_tools()
    circuits = {key: parse(row["netlist"]["netlist"]) for key, row in rows.items()}
    tree = ast.parse((OUT / "audit.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "simulate")
    scope = {"np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "audit_simulate", "exec"), scope)
    results, disagreements = {}, []
    for job in plan["jobs"]:
        data = json.loads(Path(job["output"]).read_text())
        if data["config"]["eval_manifest_sha256"] != manifest["examples_sha256"]:
            raise ValueError("Manifest mismatch")
        groups = defaultdict(Counter)
        for result in data["per_problem_results"]:
            row = rows[result["comparison_id"]]
            circuit = circuits[row["comparison_id"]]
            prop = baselines[row["source_circuit_id"]]
            strata = ["all", "observed_variable_outputs" if prop["observed_variable_output_bits"] else "observed_constant_outputs"]
            for slot in result["search_slots"]:
                comps = slot.get("reward_components", {})
                counts = Counter(slots=1, detection=int(comps.get("detection", 0) >= 1),
                    expected_output=int(comps.get("expected_output_acc_logonly", 0) >= 1),
                    valid_final=int(slot["status"] == "FINAL"))
                text = slot.get("final_answer", "").rsplit("</think>", 1)[-1]
                fields = dict(re.findall(r'\b(INPUT_VECTOR|EXPECTED_OUTPUT):\s*"([^"]*)"', text))
                vector = assignment(fields.get("INPUT_VECTOR", ""))
                expected = assignment(fields.get("EXPECTED_OUTPUT", ""))
                if slot["status"] == "FINAL" and set(vector) == set(circuit.input_nets):
                    inputs = np.asarray([[vector[k] for k in circuit.input_nets]], dtype=np.int8)
                    _, _, good, _ = scope["simulate"](circuit, row["fault"], inputs)
                    correct = {k: int(good[k][0]) for k in circuit.output_nets}
                    bits = sum(expected.get(k) == v for k, v in correct.items())
                    counts.update(bit_scored_finals=1, output_bits=len(correct), correct_bits=bits,
                        pattern_bit_accuracy_sum=bits/len(correct))
                    if int(expected == correct) != counts["expected_output"]:
                        disagreements.append(dict(job=job["name"], comparison_id=row["comparison_id"],
                            slot=slot["completion_slot"], simulated_exact=expected == correct,
                            recorded_exact=bool(counts["expected_output"])))
                    observations = slot.get("observations", [])
                    if observations and vector == observations[-1]["vector"]:
                        requested = assignment(observations[-1]["arguments"]["output_vector"])
                        counts.update(final_matches_tool_input=1,
                            copied_requested_output=int(expected == requested),
                            wrong_requested_output=int(requested != correct),
                            corrected_wrong_request=int(requested != correct and expected == correct),
                            copied_wrong_request=int(requested != correct and expected == requested))
                for group in strata:
                    groups[group].update(counts)
        results[job["name"]] = dict(groups)
    final = dict(groups=results, score_disagreements=disagreements,
        caveat="Five circuits with observed varying outputs and four with constant outputs; source circuits, not variants, are the independent units. Bit scores condition on valid final inputs. Copy/correction denominators additionally require final input equal to the last tool input. Production parsing and gate tables are shared with the offline executor.")
    (output / "diagnostics.json").write_text(json.dumps(final, indent=2)+"\n")
    print(f"Wrote diagnostics for {len(results)} jobs; {len(disagreements)} output-score disagreements")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    main(p.parse_args().output)
