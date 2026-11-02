"""Detection by number of tool calls, for RFT rounds and GDPO steps, from stored evaluations. CPU only.
Run from the libatpgllm_hard-faults root; writes tool_calls.json beside this file."""
import collections, glob, json, re
from pathlib import Path
R = 'runs/rft_v4_a100/'
G = '../libatpgllm/runs/grpo_granite_4.2_8b_v4_a100_tetramax/'
HEAD = re.compile(r'"hint_mode": "(\w+)"')
def bucket(n): return str(n) if n < 4 else '4+'
def table(items):
    n = len(items); c = collections.Counter(); d = collections.Counter()
    for calls, det in items:
        c[bucket(calls)] += 1; d[bucket(calls)] += det
    return dict(n=n, detected=sum(d.values()) / max(n, 1), no_detection=1 - sum(d.values()) / max(n, 1),
                mean_calls=sum(x for x, _ in items) / max(n, 1),
                share={k: c[k] / n for k in sorted(c)}, detect_share={k: d[k] / n for k in sorted(d)},
                detect_given={k: d[k] / c[k] for k in sorted(c)}, counts=dict(c))
def slots(directory):
    for p in sorted(glob.glob(directory + '/slots*.jsonl')):
        for line in open(p):
            try: yield json.loads(line)
            except json.JSONDecodeError: pass
def det(s): return (s.get('reward_components') or {}).get('fault_detected_by_pred_input_vector_acc_logonly', 0) == 1
out = {}
for tag in ['round_%d' % n for n in range(0, 7)]:
    if Path(R + tag + '/eval_val/metrics.json').exists():
        out['val_greedy/' + tag] = table([(sum(m['role'] == 'tool' for m in s.get('messages') or []), det(s)) for s in slots(R + tag + '/eval_val')])
if Path(R + 'gdpo_step2/eval_val/metrics.json').exists():
    out['val_greedy/gdpo_step2'] = table([(sum(m['role'] == 'tool' for m in s.get('messages') or []), det(s)) for s in slots(R + 'gdpo_step2/eval_val')])
for n in range(1, 7):
    if Path(R + f'round_{n}/gen/report.json').exists():
        out[f'gen_T1/model_{n-1}'] = table([(len(s.get('observations') or []), det(s)) for s in slots(R + f'round_{n}/gen') if s.get('hint_mode') == 'none'])
for f in sorted(glob.glob(G + 'fixed_eval/step-*.json')):
    if 'repeat' in f: continue
    d = json.load(open(f))
    out['gdpo_fixed_greedy/step_%02d' % d['step']] = table([(int(r['components'].get('tool_calls_logonly', 0)), r['components'].get('fault_detected_by_pred_input_vector_acc_logonly', r['components']['detection']) == 1) for r in d['records']])
s = json.load(open(G + 'checkpoint-14/trainer_state.json'))
train = {}
for e in s['log_history']:
    if 'rewards/reward_fn/component_mean/tool_calls' in e:
        train[e['step']] = {k.split('/')[-1]: round(e[k], 4) for k in e if k.startswith('rewards/reward_fn/component_mean/') and k.split('/')[-1] in ('tool_calls', 'attempts', 'simulated_calls', 'duplicate_calls', 'calls_after_detection', 'detection')}
        train[e['step']].update({k: round(e[k], 4) for k in ('tools/call_frequency', 'tools/call_count') if k in e})
out['gdpo_training_means'] = train
p = Path('../libatpgllm/runs/eval_results_grpo_granite_4.2_8b_repaired_a100_bon32_policy/checkpoint-25_passatk_single_completion_csv1_nc16_t0.7_topp0.95_b64.slots.jsonl')
if p.exists():
    items = []
    for line in open(p):
        x = json.loads(line)
        if 'evaluation_problem_id' in x:
            r = x.get('reward_components') or {}
            items.append((len(x.get('observations') or []), r.get('fault_detected_by_pred_input_vector_acc_logonly', 0) == 1))
    out['paper_ckpt25_direct_T0.7'] = table(items)
Path(__file__).with_name('tool_calls.json').write_text(json.dumps(out, indent=1))
print('%-32s %6s %7s %6s | share of attempts by calls | detected AND that many calls | no detection' % ('source', 'n', 'det', 'calls'))
for k, v in out.items():
    if k == 'gdpo_training_means': continue
    keys = ['0', '1', '2', '3', '4+']
    print('%-32s %6d %7.3f %6.2f | %s | %s | %.3f' % (k, v['n'], v['detected'], v['mean_calls'], ' '.join('%5.3f' % v['share'].get(x, 0) for x in keys), ' '.join('%5.3f' % v['detect_share'].get(x, 0) for x in keys), v['no_detection']))
print('GDPO training (mean per trajectory, retained best-of-32):')
for step, row in sorted(train.items()): print(' ', step, row)
