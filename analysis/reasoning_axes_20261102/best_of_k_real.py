"""Expected best stage score of k real attempts per problem, from the stored pilot_start slots.
Slots are independent samples of the policy with no feedback between them, so this is a
stand in for k attempts, not a measurement of real retries. CPU only."""
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
HERE = Path(__file__).resolve().parent
RUN = HERE.parents[1] / 'runs/reasoning_pilot/bench_pilot_start'
STAGES = ('activation', 'propagation', 'controlling')
problems = defaultdict(list)
for path in sorted(RUN.glob('slots.shard*.jsonl')):
    for line in open(path):
        s = json.loads(line)
        c = s.get('reward_components') or {}
        problems[(s['level'], s['example_id'])].append({k: float(c.get(k) or 0.0) for k in STAGES})
out = defaultdict(dict)
for k in (1, 2, 3, 5):
    table = defaultdict(lambda: defaultdict(list))
    for (level, _), slots in problems.items():
        for key in STAGES:
            values = sorted(s[key] for s in slots)
            n = len(values)
            combos = list(combinations(values, k)) if k <= 3 else None
            if combos is None:
                from math import comb
                best = sum(v * comb(i, k - 1) for i, v in enumerate(values)) / comb(n, k)
            else:
                best = sum(max(c) for c in combos) / len(combos)
            for b in (level, 'all'):
                table[b][key].append(best)
    for b, row in table.items():
        out[b][f'k{k}'] = {key: sum(v) / len(v) for key, v in row.items()}
constants = json.loads((HERE / 'best_attempt_check.json').read_text())['constants']
for b in ('easy', 'medium', 'hard', 'very_hard', 'random_never_detects', 'all'):
    print(b)
    for key in STAGES:
        print(f"  {key:12s} real k1 {out[b]['k1'][key]:.3f} k2 {out[b]['k2'][key]:.3f} k3 {out[b]['k3'][key]:.3f} k5 {out[b]['k5'][key]:.3f}"
              f" | zeros+ones best {constants[b]['best_of_both_' + key]:.3f}")
(HERE / 'best_of_k_real.json').write_text(json.dumps(out, indent=1))
