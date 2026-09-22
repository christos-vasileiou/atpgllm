"""Check three pilot circuits using hand-derived Boolean equations.

No production Verilog parser, cell tables, or simulator is imported. Includes
original and renamed/reordered variants, scored finals, and tool observations.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

OUT = Path(__file__).resolve().parent


def logic(module, vector):
    if module == "half_wave":
        sign = vector["sine[7]"]
        good = {f"out[{i}]": vector[f"sine[{i}]"] | (1-sign) for i in range(7)}
        good["out[7]"] = sign
        bad = {**{f"out[{i}]": 1 for i in range(7)}, "out[7]": sign}
    elif module == "adder":
        good = {f"sum[{i}]": vector[f"a[{i}]"] ^ vector[f"b[{i}]"] for i in range(4)}
        good.update({f"cout[{i}]": vector[f"a[{i}]"] & vector[f"b[{i}]"] for i in range(4)})
        bad = {**good, "sum[3]": 1-good["cout[3]"]}
    elif module == "multiple_module_opt2":
        good, bad = {"y": 0}, {"y": 1}
    else:
        raise ValueError(module)
    return good, bad


def main(output):
    manifest = json.loads((OUT / "circuit_comparison_explicit_manifest.json").read_text())
    originals = {r["source_circuit_id"]: r["module_name"] for r in manifest["examples"] if r["variant"] == "original"}
    records = {r["comparison_id"]: r for r in manifest["examples"]}
    plan = json.loads((output / "plan.json").read_text())
    counters, errors = Counter(), []
    for job in plan["jobs"]:
        data = json.loads(Path(job["output"]).read_text())
        if data["config"]["eval_manifest_sha256"] != manifest["examples_sha256"]:
            raise ValueError("Unexpected manifest")
        for result in data["per_problem_results"]:
            row = records[result["comparison_id"]]
            module = originals[row["source_circuit_id"]]
            if module not in ("half_wave", "adder", "multiple_module_opt2"):
                continue
            inverse = {v: k for k, v in row["renaming"].items()}
            def original(values):
                return {re.sub(r"^[^\[]+", lambda m: inverse.get(m[0], m[0]), k): int(v)
                        for k, v in values.items()}
            for slot in result["search_slots"]:
                identity = dict(job=job["name"], comparison_id=row["comparison_id"], slot=slot["completion_slot"])
                if slot["status"] == "FINAL":
                    good, bad = logic(module, original(slot["vector"]))
                    predicted = {}
                    match = re.search(r'EXPECTED_OUTPUT:\s*"([^"]*)"', slot["final_answer"].rsplit("</think>", 1)[-1])
                    if match:
                        try:
                            predicted = original({k.strip(): int(v) for k, v in
                                (item.rsplit(":", 1) for item in match[1].split(","))})
                        except ValueError:
                            pass
                    detected = any(good[k] != bad[k] for k in good)
                    expected = predicted == good
                    rewards = slot["reward_components"]
                    counters["final_detection_labels_checked"] += 1
                    counters["final_output_labels_checked"] += 1
                    counters["correct_final_outputs"] += expected
                    counters["incorrect_final_outputs"] += not expected
                    if detected != bool(rewards.get("detection", 0)) or expected != bool(rewards.get("expected_output_acc_logonly", 0)):
                        errors.append(dict(**identity, kind="final_label"))
                for obs in slot["observations"]:
                    good, bad = logic(module, original(obs["vector"]))
                    table = json.loads(obs["result"])
                    for column, truth in [("Good Machine", good), ("Bad Machine", bad)]:
                        # Tables also include internal nets; compare only POs.
                        reverse_names = {k: v for k, v in row["renaming"].items()}
                        actual = {}
                        for k in truth:
                            renamed = re.sub(r"^[^\[]+", lambda m: reverse_names.get(m[0], m[0]), k)
                            actual[k] = table[column].get(renamed)
                        counters["tool_output_tables_checked"] += 1
                        if actual != truth:
                            errors.append(dict(**identity, kind=column))
    result = dict(counts=counters, mismatches=errors,
        scope="Hand-derived half_wave (INV/OR), adder (four independent half adders), and multiple_module_opt2 (tie low). Checks all completed variants and checkpoints; does not validate every circuit in the pilot.")
    (OUT / "checkpoint_independent_checks.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit("Independent circuit checks failed")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    main(p.parse_args().output)
