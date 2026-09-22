"""Bounded independent replay of repaired labels and source STIL association."""
import ast
import json
import random
import sys
from collections import Counter
from pathlib import Path
import regex
import pyarrow.parquet as pq

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
sys.path.insert(0, str(REPO.parent / 'data_preprocessing'))
from fault_sim import OptimizedNetlist, fast_fault_sim
from pattern_mapping import read_pattern_mapping
from circuit_split import netlist_identity

root = REPO.parent / 'data/freeset/dataset.freeset.asap7sc7p5t_28.rvt.tt.stil_repaired_v1'
manifest = json.loads((root/'split_manifest.json').read_text())
tree = ast.parse((REPO/'atpgllm/training/reward_function_factory.py').read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RewardFunctionFactory')
scope = {'re': regex}
for n in cls.body:
    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('DECL_RE','NAME_RE') for t in n.targets):
        exec(compile(ast.Module(body=[n],type_ignores=[]),'factory-regex','exec'),scope)
gates = json.loads((REPO/'atpgllm/training/data/sim_config.json').read_text())['gate_funcs']
rng = random.Random(1729)
train_paths = sorted((root/'train').glob('*.parquet'))
chosen = rng.sample(train_paths,64)
counts, failures, sample, sources = Counter(), [], [], []
for path in sorted((root/'validation').glob('*.parquet')) + chosen:
    rows = pq.read_table(path).to_pylist()
    if path.parent.name == 'train':
        rows = rng.sample(rows,min(4,len(rows)))
    sources.append(rows[0]['source_module_name'])
    net = OptimizedNetlist(rows[0]['netlist'],gates,scope['DECL_RE'],scope['NAME_RE'])
    stil = read_pattern_mapping((REPO.parent/'data/freeset/out.freeset.asap7sc7p5t_28.rvt.tt'/rows[0]['source_module_name']/'simulation.stil').read_text())
    for row in rows:
        counts[path.parent.name+'_rows'] += 1
        iv, ov = json.loads(row['input_vector']), json.loads(row['expected_output'])
        pi,po = stil.patterns[row['pattern_index']]
        mapped_iv,mapped_ov = stil.vectors(row['pattern_index'],pi,po,net.input_nets,net.output_nets)
        frame,_ = fast_fault_sim(iv,dict.fromkeys(net.output_nets,0),row['fault'],net,gates,return_rewards=True)
        snapshot = json.loads(row['snapshot'])
        checks = {
            'stil_inputs_match': iv == mapped_iv,
            'stil_outputs_match': ov == mapped_ov,
            'binary_complete_inputs': set(iv)==set(net.input_nets) and all(v in (0,1) for v in iv.values()),
            'exact_outputs': set(ov)==set(net.output_nets) and all(frame.loc[k,'Good Machine']==v for k,v in ov.items()),
            'target_detected': any(frame.loc[k,'Good Machine'] != frame.loc[k,'Bad Machine'] for k in net.output_nets),
            'snapshot_matches': all(k in frame.index and frame.loc[k,column]==v for column in ('Good Machine','Bad Machine') for k,v in snapshot[column].items()),
        }
        for k,v in checks.items(): counts[k] += bool(v)
        if not all(checks.values()): failures.append(dict(source=row['source_module_name'],fault=row['fault'],checks=checks))
        if len(sample)<3: sample.append({k:row[k] for k in ['source_module_name','fault','input_vector','expected_output']})

historical = json.loads((REPO/'runs/grpo_granite_4.2_8b/fixed_eval_manifest.json').read_text())
overlap = {}
for split in ('train','validation'):
    members=[r for r in manifest['circuits'] if r.get('split')==split]
    names={r['module_name'] for r in members}
    structures={r['netlist_id'] for r in members}
    matches=[e['_fixed_eval_id'] for e in historical['examples'] if e.get('module_name') in names or netlist_identity(e['netlist']['netlist']) in structures]
    overlap[split]=dict(count=len(matches),example_ids=matches)
report=dict(counts=counts, failures=failures, source_circuits=len(set(sources)), sampling='All 432 validation rows plus up to four rows in each of 64 uniformly sampled training source shards, seed 1729; not row-uniform', historical_eval_overlap=overlap, sample=sample)
(OUT/'repaired_replay.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(counts=counts,failures=failures,source_circuits=len(set(sources)),historical_eval_overlap={k:v['count'] for k,v in overlap.items()}),indent=2))
