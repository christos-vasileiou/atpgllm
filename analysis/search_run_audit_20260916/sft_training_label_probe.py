"""Check output labels in the same prespecified SFT truncation-risk sample.

This audits cached labels under the corrected simulator, not actual checkpoint
exposure. Unresolved simulations and missing inputs are reported separately.
"""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pyarrow as pa

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval"))
from prepare_circuit_comparison import parser_and_tools


def reverse_bus_values(values):
    result, buses = dict(values), {}
    for key in values:
        match = re.fullmatch(r'([A-Za-z_]\w*)\[(-?\d+)\]', key)
        if match:
            buses.setdefault(match[1], []).append((int(match[2]), key))
    for group in buses.values():
        keys = [k for _, k in sorted(group)]
        for target, source in zip(keys, reversed(keys)):
            result[target] = values[source]
    return result


def main():
    probe = json.loads((OUT / 'sft_truncation_probe.json').read_text())
    eligible = {(r['netlist_sha256'], r['fault']) for r in probe['examples'] if 'netlist_sha256' in r}
    cache = Path('/home/eng/c/cxv200006/.cache/huggingface/datasets/chrivasileiou___asap7-language-of-test-v2/default/0.0.0/d35cfea64eadf30fb3b39735b0e8d20bffcc3345')
    parse, _ = parser_and_tools()
    tree = ast.parse((OUT / 'audit.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'simulate')
    scope = {'np': np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'audit_simulate', 'exec'), scope)
    rng = np.random.default_rng(probe['seed'])
    counts, problems, sample_rows = Counter(), [], []
    for path in sorted(cache.glob('*-train-*.arrow')):
        with pa.memory_map(str(path), 'r') as source:
            for batch in pa.ipc.open_stream(source):
                chosen = np.flatnonzero(rng.random(batch.num_rows) < probe['sample_probability'])
                for row in batch.take(pa.array(chosen)).to_pylist():
                    key = hashlib.sha256(row['netlist'].encode()).hexdigest(), row['fault']
                    if key not in eligible:
                        continue
                    counts['eligible_rows'] += 1
                    problem = dict(netlist_sha256=key[0], fault=key[1])
                    sample_rows.append({k: row[k] for k in ['netlist', 'fault', 'module_name', 'input_vector', 'expected_output', 'snapshot']})
                    try:
                        circuit = parse(row['netlist'])
                        vector = json.loads(row['input_vector'])
                        expected = json.loads(row['expected_output'])
                        if set(vector) != set(circuit.input_nets):
                            counts['input_name_mismatch'] += 1
                            problems.append(dict(**problem, reason='input_name_mismatch'))
                            continue
                        inputs = np.asarray([[int(vector[k]) for k in circuit.input_nets]], dtype=np.int8)
                        detected, _, good, bad = scope['simulate'](circuit, row['fault'], inputs)
                        if any(k not in good for k in circuit.output_nets):
                            counts['unresolved_outputs'] += 1
                            problems.append(dict(**problem, reason='unresolved_outputs'))
                            continue
                        actual = {k: int(good[k][0]) for k in circuit.output_nets}
                        counts['resolved_rows'] += 1
                        counts['detecting_vectors'] += bool(detected[0])
                        match = {k: int(v) for k, v in expected.items()} == actual
                        counts['correct_expected_output_labels'] += match
                        stored = row['snapshot']
                        if isinstance(stored, str):
                            try:
                                stored = json.loads(stored)
                            except ValueError:
                                parsed = ast.parse(stored, mode='eval')
                                class ReplaceNaN(ast.NodeTransformer):
                                    def visit_Name(self, node):
                                        return ast.copy_location(ast.Constant(None), node) if node.id in ('nan', 'NaN') else node
                                stored = ast.literal_eval(ReplaceNaN().visit(parsed))
                        saved_good = {k: stored['Good Machine'].get(k) for k in circuit.output_nets}
                        saved_bad = {k: stored['Bad Machine'].get(k) for k in circuit.output_nets}
                        counts['stored_good_matches_circuit'] += saved_good == actual
                        counts['stored_good_matches_expected_label'] += saved_good == expected
                        counts['stored_bad_matches_circuit'] += saved_bad == {k: int(bad[k][0]) for k in circuit.output_nets}
                        for flip_inputs in [False, True]:
                            v = reverse_bus_values(vector) if flip_inputs else vector
                            a = np.asarray([[int(v[k]) for k in circuit.input_nets]], dtype=np.int8)
                            det, _, gm, _ = scope['simulate'](circuit, row['fault'], a)
                            counts[f'detecting_vectors_reverse_inputs_{flip_inputs}'] += bool(det[0])
                            for flip_outputs in [False, True]:
                                e = reverse_bus_values(expected) if flip_outputs else expected
                                counts[f'correct_labels_reverse_inputs_{flip_inputs}_outputs_{flip_outputs}'] += e == {k: int(gm[k][0]) for k in circuit.output_nets}
                        if not match:
                            problems.append(dict(**problem, reason='incorrect_expected_output', expected=expected, actual=actual,
                                module=row['module_name'], netlist=row['netlist'], input_vector=vector,
                                snapshot=row.get('snapshot'), detected_faults=row.get('detected_faults')))
                    except Exception as exc:
                        counts['parse_or_simulation_errors'] += 1
                        problems.append(dict(**problem, reason=str(exc)))
    if counts['eligible_rows'] != probe['counts']['eligible_rows']:
        raise ValueError('Failed to reproduce the eligible SFT sample')
    result = dict(counts=counts, problems=problems,
        caveat='Same cached Bernoulli sample and prompt eligibility as the truncation probe. Production parser and cell truth tables are shared with the vectorized executor. Neither this sample nor its labels establish actual training exposure or correctness of every training row or reasoning explanation.')
    (OUT / 'sft_training_label_probe.json').write_text(json.dumps(result, indent=2)+'\n')
    (OUT / 'sft_training_label_sample.json').write_text(json.dumps(sample_rows, indent=2)+'\n')
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()
