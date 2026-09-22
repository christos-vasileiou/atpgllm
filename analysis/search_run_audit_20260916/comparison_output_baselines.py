"""Measure output predictability under uniform input vectors in the pilot.

These are reference distributions, not output predictions repaired for models.
The vectorized executor shares the production parser and cell truth tables.
"""
import ast
import json
from pathlib import Path
import sys

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval"))
from prepare_circuit_comparison import parser_and_tools


def main():
    tree = ast.parse((OUT / "audit.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "simulate")
    scope = {"np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "audit_simulate", "exec"), scope)
    parse, _ = parser_and_tools()
    manifest = json.loads((OUT / "circuit_comparison_manifest.json").read_text())
    rng = np.random.default_rng(20260916)
    results = []
    for row in manifest["examples"]:
        if row["variant"] != "original":
            continue
        circuit = parse(row["netlist"]["netlist"])
        width = len(circuit.input_nets)
        exact = width <= 12
        vectors = (((np.arange(1 << width)[:, None] >> np.arange(width)) & 1).astype(np.int8)
                   if exact else rng.integers(0, 2, (8192, width), dtype=np.int8))
        detected, _, good, _ = scope["simulate"](circuit, row["fault"], vectors)
        outputs = np.stack([good[k] for k in circuit.output_nets], axis=1)
        patterns, counts = np.unique(outputs, axis=0, return_counts=True)
        mode = patterns[counts.argmax()]
        marginal = outputs.mean(axis=0)
        results.append(dict(source_circuit_id=row["source_circuit_id"], module=row["module_name"],
            exact=exact, input_vectors=len(vectors), output_bits=len(circuit.output_nets),
            observed_variable_output_bits=int(np.sum((marginal > 0) & (marginal < 1))),
            distinct_observed_output_patterns=len(patterns), uniform_detection=float(detected.mean()),
            best_constant_output_exact_accuracy=float(counts.max()/len(vectors)),
            best_constant_bit_accuracy=float(np.maximum(marginal, 1-marginal).mean()),
            most_frequent_output=dict(zip(circuit.output_nets, mode.tolist()))))
    result = dict(circuits=results,
        caveat="Uniform inputs; exhaustive for seven circuits, sampled for two. Model-selected inputs need not have this distribution. These references neither repair model answers nor constitute an independently validated simulator.")
    (OUT / "comparison_output_baselines.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps([{k: v for k, v in r.items() if k != 'most_frequent_output'} for r in results], indent=2))


if __name__ == "__main__":
    main()
