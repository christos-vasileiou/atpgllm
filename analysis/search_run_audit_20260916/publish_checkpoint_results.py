"""Preserve completed comparison aggregates and plot circuit-bootstrap intervals."""
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
RUN = ROOT / 'runs/circuit_comparison_explicit_20260916'


def main():
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/atpg-checkpoint-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plan = json.loads((RUN / 'plan.json').read_text())
    summary = json.loads((RUN / 'summary.json').read_text())
    if len(plan['jobs']) != 12 or any(d['infrastructure_errors'] for d in summary['diagnostics']):
        raise ValueError('Unexpected or failed comparison matrix')
    receipt, observations = [], {}
    for job in plan['jobs']:
        path = Path(job['output'])
        raw = path.read_bytes()
        data = json.loads(raw)
        rows = data['per_problem_results']
        if data['config']['eval_manifest_sha256'] != plan['manifest_sha256']:
            raise ValueError('Manifest mismatch')
        scores = {}
        for row in rows:
            scores.setdefault(row['source_circuit_id'], []).extend(
                s.get('reward_components', {}).get('expected_output_acc_logonly', 0) >= 1 for s in row['search_slots'])
        observations[job['model'], job['context'], job['tool_rounds']] = np.array([np.mean(scores[k]) for k in sorted(scores)])
        receipt.append(dict(job=job['name'], path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
            problems=len(rows), slots=sum(len(r['search_slots']) for r in rows)))
    (OUT / 'checkpoint_comparison_receipt.json').write_text(json.dumps(dict(
        manifest_sha256=plan['manifest_sha256'], jobs=receipt, slots=sum(r['slots'] for r in receipt)), indent=2)+'\n')
    for source, target in [('summary.json', 'checkpoint_comparison_summary.json'),
                           ('summary.csv', 'checkpoint_comparison_summary.csv'),
                           ('diagnostics.json', 'checkpoint_comparison_diagnostics.json')]:
        shutil.copyfile(RUN / source, OUT / target)
    rng = np.random.default_rng(20260917)
    draws = rng.integers(0, 9, (10000, 9))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for rounds, ax in enumerate(axes):
        for i, (model, label, color) in enumerate([('base', 'Base', '#2463A6'), ('sft', 'SFT 200', '#B45C22'), ('grpo', 'GRPO 50', '#27805A')]):
            means, intervals = [], []
            for context in [8192, 16384]:
                values = observations[model, context, rounds]
                means.append(values.mean()*100)
                intervals.append(np.quantile(values[draws].mean(axis=1), [.025, .975])*100)
            means, intervals = np.array(means), np.array(intervals).T
            ax.errorbar(np.array([0, 1])+(i-1)*.035, means,
                yerr=np.maximum(0, np.array([means-intervals[0], intervals[1]-means])),
                color=color, marker='o', capsize=4, linewidth=1.8, label=label)
        ax.set(title='No simulator feedback' if rounds == 0 else 'At most one simulator call',
               xticks=[0, 1], xticklabels=['8K', '16K'], xlabel='Inference context cap', ylim=(0, 100), xlim=(-.18, 1.18))
        ax.grid(axis='y', alpha=.22)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Exact expected-output accuracy (%)')
    axes[1].legend(frameon=False, loc='upper left')
    fig.suptitle('Longer context helps the base model much more than the adapted checkpoints')
    fig.text(.5, .01, 'Nine circuits · four equivalent variants · four slots each · one seed · shared bf16 backbone\n95% intervals resample source circuits; small diagnostic pilot', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, .94))
    fig.savefig(OUT / 'checkpoint_output_accuracy.png', dpi=180)
    fig.savefig(OUT / 'checkpoint_output_accuracy.svg')
    plt.close(fig)
    print(f'Preserved {len(receipt)} jobs / {sum(r["slots"] for r in receipt)} slots and the accuracy figure')


if __name__ == '__main__':
    main()
