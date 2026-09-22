"""Extract this run's logged optimizer metrics and plot the completed steps."""
import ast
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
SOURCE = REPO / 'wandb/run-20260908_144532-58vu7m6x/files/output.log'
rows = [ast.literal_eval(line) for line in SOURCE.read_text().splitlines()
        if line.startswith("{'loss':")]
(OUT / 'metrics.json').write_text(json.dumps(rows, indent=2) + '\n')
with (OUT / 'metrics.csv').open('w') as stream:
    fields = ['optimizer_step'] + sorted(set().union(*(row.keys() for row in rows)))
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows({'optimizer_step': i, **row} for i, row in enumerate(rows, 1))

x = list(range(1, len(rows) + 1))
def values(key, scale=1):
    return [scale * r[key] for r in rows]

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.facecolor': 'white'})
fig, axes = plt.subplots(2, 2, figsize=(12, 7.4), constrained_layout=True)
ax = axes[0, 0]
ax.plot(x, values('rewards/reward_fn/component_mean/detection', 100),
        'o-', color='#1764ab', label='Raw detection')
ax.plot(x, values('rewards/reward_fn/component_mean/activation', 100),
        '.--', color='#888888', label='Raw activation')
ax.set(title='Changing training prompts; no fixed evaluation', ylabel='Completions (%)', ylim=(45, 80))
ax.legend(loc='lower right', frameon=False)

ax = axes[0, 1]
ax.semilogy(x, values('grad_norm'), 'o-', color='#b33f32')
ax.axhline(1, color='#777777', linestyle=':', label='Existing clipping threshold = 1')
for step in (9, 11, 13):
    if step <= len(rows):
        ax.annotate(f'{rows[step-1]["grad_norm"]:g}',
                    (step, rows[step-1]['grad_norm']), xytext=(4, 5), textcoords='offset points')
ax.set(title='Intermittent gradient spikes', ylabel='Gradient norm before clipping', ylim=(0.8, 190))
ax.legend(loc='upper left', frameon=False)

ax = axes[1, 0]
ax.plot(x, values('kl'), 'o-', color='#7658a5')
ax.set(title='Divergence from the fixed SFT reference', ylabel='Mean sampled KL')
right = ax.twinx()
right.plot(x, values('learning_rate', 1e6), '--', color='#666666')
right.set_ylabel('Learning rate (×10⁻⁶)', color='#666666')

ax = axes[1, 1]
ax.plot(x, values('step_time', 1/60), 'o-', color='#217f71')
ax.set(title='About 91 minutes per optimizer step', ylabel='Minutes', ylim=(50, 120))
for ax in axes.flat:
    ax.set_xlabel('Optimizer step')
    ax.set_xticks([1, 3, 5, 7, 9, 11, 14])
    ax.grid(alpha=0.16)
fig.suptitle(f'58vu7m6x · Slurm 383805 · {len(rows)} completed steps', fontsize=16)
fig.savefig(OUT / 'training_diagnostics.png', dpi=170)
fig.savefig(OUT / 'training_diagnostics.pdf')
print(f'Exported {len(rows)} metric rows and PNG/PDF plots to {OUT}')
