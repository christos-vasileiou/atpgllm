"""Reproduce the local six-run audit without loading a language model.

Run with the project's Python environment. Reads saved evaluation JSON only.
The vectorized simulator is an independent execution loop over the production
parser's compiled gates; it is not independent validation of parsing/cell logic.
"""
from pathlib import Path
import collections
import csv
import hashlib
import json
import math
import os
import random
import sys
import ast
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/atpg-audit-matplotlib")
# Avoid eager model-stack imports in atpgllm/__init__.py. The unchanged parser,
# gate factory and verifier.problem are sufficient for this offline audit.
for name, path in [("atpgllm", ROOT/"atpgllm"), ("atpgllm.training", ROOT/"atpgllm/training")]:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
factory_path = ROOT/"atpgllm/training/reward_function_factory.py"
tree = ast.parse(factory_path.read_text())
tree.body = [node for node in tree.body if not
    (isinstance(node, ast.ImportFrom) and node.module == "atpgllm.llm.reward_funcs")]
module = types.ModuleType("atpgllm.training.reward_function_factory")
module.__file__ = str(factory_path)
sys.modules[module.__name__] = module
exec(compile(tree, str(factory_path), "exec"), module.__dict__)
from atpgllm.training.reward_function_factory import (
    RewardFunctionFactory, parse_doc_id_and_netlist_from_prompt,
)
from atpgllm.training.search_verifier import Verifier, final_fields, assignment
from atpgllm.training.search_types import stable_seed

METHODS = ["random", "greedy", "mcts", "best_of_n", "evolutionary", "vector_evolutionary"]
DATA = ROOT / "runs/eval_results_grpo_granite_4.2_8b_policy"


def passk(c, k, n=16):
    return 1 - math.comb(n-c, k)/math.comb(n, k) if n-c >= k else 1.0


def simulate(netlist, fault, vectors):
    """Independent NumPy gate execution: clamp only the target fault net."""
    count = len(vectors)
    stuck, target = int(fault[2]), fault.split(maxsplit=1)[1]
    good = {k: vectors[:, i] for i, k in enumerate(netlist.input_nets)}
    bad = dict(good)
    bad[target] = np.full(count, stuck, dtype=np.int8)
    def resolve(machine, name):
        if isinstance(name, int) or (isinstance(name, str) and name.isdigit()):
            return np.full(count, int(name), dtype=np.int8)
        return machine.get(name)
    def run(machine, faulty):
        pending = list(netlist.instructions)
        for _ in range(len(pending)+1):
            remaining = []
            for ins in pending:
                kind, dest = ins[:2]
                if faulty and dest == target:
                    continue
                if kind == "assign_const":
                    machine[dest] = np.full(count, ins[2], dtype=np.int8)
                elif kind == "assign_net":
                    value = resolve(machine, ins[2])
                    if value is None:
                        remaining.append(ins)
                    else:
                        machine[dest] = value
                else:
                    lut, names, mapping = ins[2:5]
                    args = [resolve(machine, mapping.get(k)) for k in names]
                    if any(a is None for a in args):
                        remaining.append(ins)
                    elif isinstance(lut, list):
                        index = np.zeros(count, dtype=np.int64)
                        for j, a in enumerate(args):
                            index |= a.astype(np.int64) << j
                        machine[dest] = np.asarray(lut, dtype=np.int8)[index]
                    else:
                        machine[dest] = np.asarray([int(bool(lut(*a))) for a in zip(*args)], dtype=np.int8)
            if not remaining or len(remaining) == len(pending):
                break
            pending = remaining
    run(good, False)
    run(bad, True)
    detected = np.zeros(count, dtype=bool)
    for po in netlist.output_nets:
        if po in good and po in bad:
            detected |= good[po] != bad[po]
    activation = good[target] != stuck if target in good else np.zeros(count, dtype=bool)
    return detected, activation, good, bad


def main():
    runs = {}
    for path in DATA.glob("*.json"):
        d = json.loads(path.read_text())
        method = d.get("config", {}).get("sampling_method")
        if method in METHODS:
            runs[method] = d
    assert set(runs) == set(METHODS)
    rows = {m: runs[m]["per_problem_results"] for m in METHODS}
    ids = [[r["search_slots"][0]["problem_id"] for r in rows[m]] for m in METHODS]
    assert all(x == ids[0] for x in ids)
    assert len(set(ids[0])) == len(ids[0]) == 512
    summary = {}
    for m in METHODS:
        slots = [s for r in rows[m] for s in r["search_slots"]]
        d = runs[m]
        c = np.array([r["num_correct"] for r in rows[m]])
        for k in [1, 2, 4, 8, 16]:
            assert abs(np.mean([passk(int(x), k) for x in c])-d["pass_at_k"][f"pass@{k}"]) < 1e-12
        for r in rows[m]:
            assert r["num_correct"] == sum(s["reward_components"].get("detection", 0) for s in r["search_slots"])
        summary[m] = {
            **d["pass_at_k"], **d["aggregate_metrics"],
            "accuracy_metrics": d["accuracy_metrics"],
            "successes": int(c.sum()), "solved_problems": int((c > 0).sum()),
            "status": dict(collections.Counter(s["status"] for s in slots)),
            "stop_reason": dict(collections.Counter(s["stop_reason"] for s in slots)),
            "attempts_distribution": dict(collections.Counter(s["usage"]["attempts"] for s in slots)),
            "mean_unique_vectors": float(np.mean([s["unique_vectors"] for s in slots])),
            "full_accuracy_selected": sum(all(s["reward_components"].get(k, 0) >= 1 for k in
                ["detection", "input_vector_acc_logonly", "expected_output_acc_logonly", "detected_faults_acc_logonly"])
                for s in slots)/len(slots),
        }
    rng = np.random.default_rng(20260916)
    bootstrap = rng.integers(0, 512, size=(10000, 512))
    comparisons = []
    for a, b in [("greedy", "random"), ("best_of_n", "vector_evolutionary"),
                 ("evolutionary", "vector_evolutionary"), ("mcts", "best_of_n")]:
        for k in [1, 4, 16]:
            x = np.array([passk(r["num_correct"], k) for r in rows[a]])
            y = np.array([passk(r["num_correct"], k) for r in rows[b]])
            delta = x-y
            ci = np.quantile(delta[bootstrap].mean(axis=1), [0.025, 0.975])
            comparisons.append(dict(a=a,b=b,k=k,delta=float(delta.mean()),ci95=ci.tolist(),
                wins=int((delta>0).sum()),ties=int((delta==0).sum()),losses=int((delta<0).sum())))
    rf = RewardFunctionFactory()
    from fault_sim import fast_fault_sim
    verifier = Verifier.__new__(Verifier)
    verifier.reward_factory = rf
    verifier.fault_sim = fast_fault_sim
    details, replay_mismatches, missing_po, net_hashes = [], collections.Counter(), [], []
    problems = []
    hybrid_counts, hybrid_cost = [], 0
    for i, r in enumerate(rows["greedy"]):
        prompt = next(x["content"] for x in r["search_slots"][0]["messages"] if x["role"] == "user")
        doc, raw = parse_doc_id_and_netlist_from_prompt(prompt)
        problem = verifier.problem(prompt, dict(netlist=dict(doc_id=doc, netlist=raw), fault=r["fault"], module_name=r["module_name"]))
        assert problem.problem_id == ids[0][i], "Backend/config/source identity mismatch"
        net_hashes.append(hashlib.sha256(raw.encode()).hexdigest())
        problems.append(problem)
        n = len(problem.input_nets)
        # Exact for <=16 PI bits; deterministic Monte Carlo for larger circuits.
        if n <= 16:
            vectors = ((np.arange(2**n, dtype=np.uint64)[:, None] >> np.arange(n, dtype=np.uint64)) & 1).astype(np.int8)
            exact = True
        else:
            vectors = rng.integers(0, 2, size=(8192, n), dtype=np.int8)
            exact = False
        det, act, gm, bm = simulate(problem.netlist, problem.fault, vectors)
        p = float(det.mean())
        absent = [k for k in problem.output_nets if k not in gm or k not in bm]
        if absent:
            missing_po.append(dict(idx=i,module=r["module_name"],outputs=absent))
        entry = dict(idx=i,problem_id=problem.problem_id,module=r["module_name"],fault=r["fault"],
            pi_bits=n,po_bits=len(problem.output_nets),gates=len(problem.netlist.instructions),
            exact=exact,probe_vectors=len(vectors),uniform_detection=p,uniform_activation=float(act.mean()),
            expected_random_best4=1-(1-p)**4,target_is_pi=problem.fault.split()[1] in problem.input_nets,
            target_is_po=problem.fault.split()[1] in problem.output_nets)
        control = np.array([[0]*n,[1]*n], dtype=np.int8)
        dc,_,_,_ = simulate(problem.netlist,problem.fault,control)
        entry.update(all_zero=int(dc[0]), all_one=int(dc[1]),zero_or_one=int(dc.any()))
        # A cheap, prespecified four-candidate control: zeros, ones, two random
        # vectors; stop on detection. Sixteen independent slots per problem.
        if dc.any():
            hybrid_counts.append(16)
            hybrid_cost += 16 * (1 if dc[0] else 2)
        else:
            trial = []
            for slot in range(16):
                control_rng = random.Random(stable_seed(42, problem.problem_id, slot))
                trial.extend([[control_rng.getrandbits(1) for _ in range(n)] for _ in range(2)])
            hits,_,_,_ = simulate(problem.netlist,problem.fault,np.array(trial,dtype=np.int8))
            hits = hits.reshape(16,2)
            hybrid_counts.append(int(hits.any(axis=1).sum()))
            hybrid_cost += int((3 + ~hits[:,0]).sum())
        entry['constant2_random2_observed'] = hybrid_counts[-1]/16
        for m in METHODS:
            entry[m] = rows[m][i]["num_correct"]/16
            extracted, labels = [], []
            for j, s in enumerate(rows[m][i]["search_slots"]):
                if s["status"] != "FINAL":
                    continue
                text = s.get("final_answer", rows[m][i]["completions"][j])
                vector = assignment(final_fields(text)["INPUT_VECTOR"], problem.input_nets)
                extracted.append([vector[k] for k in problem.input_nets])
                labels.append(int(s["reward_components"]["detection"]))
            if extracted:
                actual,_,_,_ = simulate(problem.netlist,problem.fault,np.array(extracted,dtype=np.int8))
                replay_mismatches[m] += int((actual != np.array(labels)).sum())
        details.append(entry)
        if i % 64 == 0:
            print(f"Validated {i+1}/512 circuits", flush=True)
    strata = []
    for title, select in [
        ("zero hits in uniform probe (not proof of p=0)",lambda d:d["uniform_detection"]==0),
        ("uniform_p<=0.1",lambda d:0<d["uniform_detection"]<=.1),
        ("0.1<uniform_p<=0.5",lambda d:.1<d["uniform_detection"]<=.5),
        ("uniform_p>0.5",lambda d:d["uniform_detection"]>.5)]:
        subset=[d for d in details if select(d)]
        strata.append(dict(stratum=title,n=len(subset),**{m:float(np.mean([d[m] for d in subset])) for m in METHODS}))
    out = dict(runs=summary,comparisons=comparisons,strata=strata,replay_mismatches=dict(replay_mismatches),
        missing_outputs=missing_po,unique_netlist_hashes=len(set(net_hashes)),
        uniform_probe_mean=float(np.mean([d["uniform_detection"] for d in details])),
        uniform_best4_mean=float(np.mean([d["expected_random_best4"] for d in details])),
        constant2_random2_expected=float(np.mean([1 if d['zero_or_one'] else
            1-(1-d['uniform_detection'])**2 for d in details])),
        constant2_random2_observed={**{f'pass@{k}':float(np.mean([passk(c,k) for c in hybrid_counts]))
            for k in [1,2,4,8,16]},'simulator_evaluations':hybrid_cost,
            'successes':sum(hybrid_counts),'solved_problems':sum(c>0 for c in hybrid_counts)},
        exact_circuits=sum(d["exact"] for d in details),
        constants={k:float(np.mean([d[k] for d in details])) for k in ["all_zero","all_one","zero_or_one"]},
        pi_bits=dict(collections.Counter(d["pi_bits"] for d in details)),
        target_is_pi=sum(d["target_is_pi"] for d in details),target_is_po=sum(d["target_is_po"] for d in details))
    (OUT/"analysis.json").write_text(json.dumps(out,indent=2)+"\n")
    with (OUT/"per_problem.csv").open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(details[0]));w.writeheader();w.writerows(details)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
    for m in METHODS:
        axes[0].plot([1,2,4,8,16],[summary[m][f"pass@{k}"]*100 for k in [1,2,4,8,16]],marker="o",label=m)
    axes[0].set(xlabel="Returned search completions k",ylabel="Fault detection pass@k (%)",xticks=[1,2,4,8,16])
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    axes[1].scatter([d["uniform_detection"]*100 for d in details],[d["greedy"]*100 for d in details],s=12,alpha=.35)
    axes[1].plot([0,100],[0,100],"--",color="black",linewidth=1)
    axes[1].set(xlabel="Uniform-vector detection probability (%)",ylabel="Greedy observed detection (%)")
    axes[1].grid(alpha=.2)
    fig.suptitle("512 matched circuits; 16 slots per circuit; detection-only scoring")
    fig.savefig(OUT/"performance.png",dpi=180)
    print(json.dumps({k:v for k,v in out.items() if k!='runs'},indent=2),flush=True)


if __name__ == "__main__":
    main()
