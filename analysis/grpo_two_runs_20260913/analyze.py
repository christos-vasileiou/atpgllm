"""Read local W&B binary logs and frozen evaluations; never mutate source evidence."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
RUNS = ['run-20260909_135227-5gmb8rfn', 'run-20260911_135505-f9uuxsrt']

def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2) + '\n')

all_train, runs = [], {}
for name in RUNS:
    run_id = name.split('-')[-1]
    reader = DataStore()
    reader.open_for_scan(str(REPO / 'wandb' / name / f'run-{run_id}.wandb'))
    counts, config, rows = Counter(), {}, []
    tail_error = None
    while True:
        try:
            data = reader.scan_data()
        except Exception as exc:
            tail_error = str(exc)
            break
        if data is None:
            break
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        kind = record.WhichOneof('record_type')
        counts[kind] += 1
        if kind == 'history':
            row = {x.key or '/'.join(x.nested_key): json.loads(x.value_json) for x in record.history.item}
            if 'train/reward' in row or any(k.startswith('eval_fixed/') for k in row):
                rows.append(row)
        elif kind in ('config', 'run'):
            items = record.config.update if kind == 'config' else record.run.config.update
            config.update({x.key or '/'.join(x.nested_key): json.loads(x.value_json) for x in items})
    train = [dict(r, run_id=run_id) for r in rows if 'train/reward' in r]
    all_train.extend(train)
    runs[run_id] = dict(record_counts=counts, tail_error=tail_error, config=config, metrics=rows)
    write(f'{run_id}_snapshot.json', runs[run_id])

keys = sorted(set().union(*(r.keys() for r in all_train)))
with (OUT / 'training_metrics.csv').open('w') as stream:
    writer = csv.DictWriter(stream, fieldnames=keys)
    writer.writeheader()
    writer.writerows(all_train)

manifest = json.loads((REPO / 'runs/grpo_granite_4.2_8b/fixed_eval_manifest.json').read_text())
evals, groups = {}, {}
for path in sorted((REPO / 'runs/grpo_granite_4.2_8b/fixed_eval').glob('step-*.json')):
    payload = json.loads(path.read_text())
    assert payload['examples_sha256'] == manifest['examples_sha256']
    step = payload['step']
    groups[step] = defaultdict(list)
    for r in payload['records']:
        groups[step][r['example_id']].append(r)
    assert len(groups[step]) == 72 and all(len(rs) == 3 for rs in groups[step].values())
    evals[step] = dict(payload['metrics'], initial_checkpoint=payload['initial_checkpoint'], protocol=payload['protocol'])
    evals[step]['usable_logged'] = sum(r['components']['detection'] == 1 and r['components']['expected_output_acc_logonly'] == 1 and r['components']['pi_completeness_logonly'] == 1 for r in payload['records']) / 216

ids = sorted(groups[0])
paired = {}
for base_step, step in [(0,s) for s in evals] + [(34,65)]:
    base = np.array([np.mean([r['components']['detection'] for r in groups[base_step][i]]) for i in ids])
    current = np.array([np.mean([r['components']['detection'] for r in groups[step][i]]) for i in ids])
    delta = current-base
    rng = np.random.default_rng(1729)
    boot = delta[rng.integers(0,72,size=(20000,72))].mean(axis=1)
    paired[f'{base_step}_to_{step}'] = dict(delta=float(delta.mean()), ci95=np.quantile(boot,[.025,.975]).tolist(), improved=int((delta>0).sum()), worsened=int((delta<0).sum()), unchanged=int((delta==0).sum()))

selected = ['train/reward', 'train/rewards/reward_fn/raw_mean', 'train/rewards/reward_fn/component_mean/detection', 'train/rewards/reward_fn/component_mean/expected_output_acc', 'train/rewards/reward_fn/component_mean/fidelity', 'train/entropy', 'train/kl', 'train/grad_norm', 'train/gdpo/objective_active_fraction/detection', 'train/frac_reward_zero_std', 'train/loss/accumulation_scale', 'train/completions/clipped_ratio', 'train/completions/mean_length', 'train/tools/call_frequency', 'train/learning_rate', 'train/sampling/sampling_logp_difference/mean']
windows = {}
for low, high in [(1,5),(1,34),(25,34),(35,44),(35,67),(58,67)]:
    rows = [r for r in all_train if low <= r['train/global_step'] <= high]
    windows[f'{low}-{high}'] = {k: dict(mean=float(np.mean([r[k] for r in rows if k in r])), min=float(min(r[k] for r in rows if k in r)), max=float(max(r[k] for r in rows if k in r))) for k in selected if any(k in r for r in rows)}
    windows[f'{low}-{high}']['n'] = len(rows)
run_summary = {}
for run_id, run in runs.items():
    rows = [r for r in all_train if r['run_id'] == run_id]
    run_summary[run_id] = dict(steps=[r['train/global_step'] for r in rows], mean_minutes_between_updates=float(np.mean(np.diff([r['_timestamp'] for r in rows]))/60), history_records=run['record_counts']['history'], tail_error=run['tail_error'])
write('summary.json', dict(runs=run_summary, windows=windows, evaluations=evals, paired_detection=paired, manifest_sha256=manifest['examples_sha256']))
print(json.dumps(dict(runs=run_summary, evaluations={s:{k:v for k,v in e.items() if k not in ('protocol','initial_checkpoint')} for s,e in evals.items()}, paired_final=paired['0_to_65']),indent=2))
