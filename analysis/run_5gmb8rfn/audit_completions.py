"""CPU simulator audit of saved evaluation responses; no model inference."""
import ast
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
sys.path.insert(0, str(REPO.parent / "data_preprocessing"))
from fault_sim import OptimizedNetlist, fast_fault_sim, convert_string_to_dict
import regex

# Load just the regex definitions from the factory, without importing ML models.
tree = ast.parse((REPO / "atpgllm/training/reward_function_factory.py").read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RewardFunctionFactory")
scope = {"re": regex}
for node in cls.body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("DECL_RE", "NAME_RE") for t in node.targets):
        exec(compile(ast.Module(body=[node], type_ignores=[]), "factory-regex", "exec"), scope)
gates = json.loads((REPO / "atpgllm/training/data/sim_config.json").read_text())["gate_funcs"]
manifest = json.loads((REPO / "runs/grpo_granite_4.2_8b/fixed_eval_manifest.json").read_text())
examples = {r["_fixed_eval_id"]: r for r in manifest["examples"]}
netlists = {i: OptimizedNetlist(e["netlist"]["netlist"], gates, scope["DECL_RE"], scope["NAME_RE"]) for i, e in examples.items()}

def final_vector(text, label):
    matches = re.findall(label + r':\s+"(.*?)"', text, re.S)
    return convert_string_to_dict(matches[0], sep=":" if ":" in matches[0] else "=") if matches else None

summaries = {}
for path in sorted((REPO / "runs/grpo_granite_4.2_8b/fixed_eval").glob("step-*.json")):
    payload = json.loads(path.read_text())
    count = Counter()
    groups = defaultdict(list)
    failures = []
    for r in payload["records"]:
        e = examples[r["example_id"]]
        netlist = netlists[r["example_id"]]
        text = r["completion"]
        count["completions"] += 1
        count["tool_calls"] += text.count("<tool_call>")
        count["multiple_tool_calls"] += text.count("<tool_call>") > 1
        try:
            iv, ov = final_vector(text, "INPUT_VECTOR"), final_vector(text, "EXPECTED_OUTPUT")
        except Exception:
            iv = ov = None
        groups[r["example_id"]].append((text, json.dumps(iv, sort_keys=True)))
        toolout = re.search(r'<parameter=output_vector>\s*(.*?)\s*</parameter>', text, re.S)
        toolin = re.search(r'<parameter=input_vector>\s*(.*?)\s*</parameter>', text, re.S)
        if toolout and toolin and iv is not None and ov is not None:
            try:
                guess, tool_iv = json.loads(toolout[1]), json.loads(toolin[1])
                count["comparable_tool_final"] += 1
                count["final_output_equals_pretool_guess"] += guess == ov
                count["final_input_equals_tool_input"] += tool_iv == iv
                count["unchanged_wrong_output"] += guess == ov and r["components"]["expected_output_acc_logonly"] == 0
            except Exception:
                count["tool_vector_parse_failure"] += 1
        if not iv:
            continue
        try:
            frame, extra = fast_fault_sim(iv, dict.fromkeys(netlist.output_nets, 0), e["fault"], netlist, gates, module_name=e.get("module_name"), return_rewards=True)
            if "error" in frame.columns:
                count["replay_simulator_error"] += 1
                continue
            count["replayed"] += 1
            pos = frame.loc[frame["POs"]]
            detected = any(a in (0, 1) and b in (0, 1) and a != b for a,b in zip(pos["Good Machine"], pos["Bad Machine"]))
            count["detection_mismatches"] += float(detected) != r["components"]["detection"]
            expected = {str(k): v for k,v in pos["Good Machine"].items()}
            correct = bool(expected) and ov is not None and all(ov.get(k) == v and v in (0,1) for k,v in expected.items())
            count["exact_output_mismatches"] += float(correct) != r["components"]["expected_output_acc_logonly"]
            count["missing_some_output_names"] += ov is not None and not set(expected).issubset(ov)
            count["wrong_supplied_output_values"] += ov is not None and any(k in ov and ov[k] != v for k,v in expected.items())
            count["detection_and_exact_output"] += detected and correct
            count["detection_exact_output_complete_inputs"] += detected and correct and set(netlist.input_nets).issubset(iv)
            if detected and not correct and len(failures) < 4:
                failures.append({"example_id":r["example_id"], "fault":e["fault"], "doc_id":e["netlist"]["doc_id"], "predicted":ov,"simulated_good":expected})
        except Exception as exc:
            count["replay_exception"] += 1
            if len(failures) < 4:
                failures.append({"error":str(exc)})
    count["groups_all_completions_identical"] = sum(len(set(t for t,v in rs)) == 1 for rs in groups.values())
    count["groups_all_input_vectors_identical"] = sum(len(set(v for t,v in rs)) == 1 for rs in groups.values())
    count["distinct_completions_within_groups"] = sum(len(set(t for t,v in rs)) for rs in groups.values())
    summaries[payload["step"]] = {"counts": count, "examples": failures}
(OUT / "completion_audit.json").write_text(json.dumps(summaries, indent=2, default=str))
print(json.dumps({k:v["counts"] for k,v in summaries.items()}, indent=2))

dataset = json.loads((OUT / "dataset_audit.json").read_text())
token_info = json.loads((OUT / "token_audit.json").read_text())
short_ids = {r["id"] for r in token_info["prompts"] if r["below_sft_prompt_limit"]}
reference_counts = Counter()
reference_failures = []
for i, rows in dataset["problem_matches"].items():
    for row in rows:
        iv = json.loads(row["input_vector"])
        ov = json.loads(row["expected_output"])
        netlist = netlists[i]
        e = examples[i]
        frame, _ = fast_fault_sim(iv, dict.fromkeys(netlist.output_nets, 0), e["fault"], netlist, gates, module_name=e.get("module_name"), return_rewards=True)
        reference_counts["labels"] += 1
        if "error" in frame.columns:
            reference_counts["simulator_error"] += 1
            continue
        pos = frame.loc[frame["POs"]]
        reference_counts["detected"] += any(a in (0,1) and b in (0,1) and a != b for a,b in zip(pos["Good Machine"],pos["Bad Machine"]))
        reference_counts["all_expected_outputs_correct"] += not pos.empty and all(ov.get(k) == v for k,v in pos["Good Machine"].items())
        reference_counts["missing_output_names"] += not set(pos.index).issubset(ov)
        reference_counts["wrong_supplied_output_values"] += any(k in ov and ov[k] != v for k,v in pos["Good Machine"].items())
        if any(k in ov and ov[k] != v for k,v in pos["Good Machine"].items()):
            reference_failures.append({"example_id":i,"row":row["row"],"fault":e["fault"],"input_vector":iv,"expected_output_label":ov,"simulated_good":dict(pos["Good Machine"].items()),"simulated_bad":dict(pos["Bad Machine"].items())})
reference_counts["eval_circuit_overlap_below_sft_limit"] = len(short_ids.intersection(dataset["circuit_matches"]))
reference_counts["eval_task_overlap_below_sft_limit"] = len(short_ids.intersection(dataset["problem_matches"]))
(OUT / "reference_label_audit.json").write_text(json.dumps(reference_counts,indent=2))
(OUT / "reference_label_discrepancies.json").write_text(json.dumps(reference_failures,indent=2,default=str))
print("Reference-label audit", json.dumps(reference_counts))
