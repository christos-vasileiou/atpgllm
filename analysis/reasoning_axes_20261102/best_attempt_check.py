"""Two checks of best attempt scoring under the staged reward. CPU only.

1. Stored pilot_start slots, rescored the way evaluation scores them (one call, the
   answer's vector), must match the stored reward components.
2. On the benchmark problems, what constant vectors score: one call, and zeros then
   ones under the best attempt, against the mean the reward used before.

    PYTHONPATH=<worktree> TMAX_REWARD_PROFILE=staged python best_attempt_check.py
"""
import ast
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

os.environ['FAULT_SIM_BACKEND'] = 'fast'
os.environ['TMAX_REWARD_PROFILE'] = 'staged'
HERE = Path(__file__).resolve().parent
H = HERE.parents[1]
import atpgllm.llm.reward_funcs as RF
from atpgllm.training.eval_report import Simulator
from atpgllm.training.reward_function_factory import RewardFunctionFactory as F

STAGES = ('activation', 'propagation', 'controlling')
OBJECTIVES = ('detection',) + STAGES + ('fidelity', 'format', 'uniqueness')
sim = Simulator()
manifest = json.loads((H / 'runs/rft_v4_a100/eval/robust_eval_manifest.json').read_text())
reference = json.loads((H / 'runs/rft_v4_a100/eval/robust_eval_manifest.json.reference.json').read_text())
problems = {e['_fixed_eval_id']: e for e in manifest['examples']}


def netlist_text(example):
    value = example['netlist']
    if isinstance(value, dict):
        return value['netlist']
    return ast.literal_eval(value)['netlist'] if value.startswith('{') else value


def score(example, calls, answer):
    trace = {'turns': [''] * len(calls) + [answer], 'calls': calls}
    return RF.test_generation_grpo_reward(
        [example['prompt']], [answer], netlists=[sim.compile(netlist_text(example))], tool_trace=[trace],
        fault=[example['fault']], module_name=[example['module_name']], lib_gate_funcs=sim.gates,
        fault_sim=sim.fs.fast_fault_sim, fault_fn=F.fault_fn, simulation_fn=F.simulation_fn,
        input_vector_fn=F.input_vector_fn, expected_output_fn=F.expected_output_fn,
        detected_faults_fn=F.detected_faults_fn, eval_mode=True)[0]


def call(vector):
    return {'arguments': {'input_vector': vector}, 'result': '{"Good Machine": {}, "Bad Machine": {}}'}


def stored_check(sample=400):
    from atpgllm.training.search_verifier import final_assignment, final_fields
    slots = []
    for path in sorted((H / 'runs/reasoning_pilot/bench_pilot_start').glob('slots.shard*.jsonl')):
        slots += [json.loads(line) for line in open(path)]
    slots = [s for s in slots if s.get('status') == 'FINAL' and s.get('reward_components')]
    calls_seen = sum(len(s.get('observations') or []) > 1 for s in slots)
    random.Random(5).shuffle(slots)
    compared = differences = skipped = 0
    largest = 0.0
    for slot in slots[:sample]:
        example = problems[slot['example_id']]
        try:
            fields = final_fields(slot['final_answer'])
            vector = final_assignment(fields['INPUT_VECTOR'], sim.compile(netlist_text(example)).input_nets)
        except (ValueError, TypeError, SyntaxError, KeyError):
            skipped += 1
            continue
        text = '\n'.join(f'{name}: "{value}"' for name, value in fields.items())
        row = score(example, [call(vector)], text)
        for key in OBJECTIVES:
            old, new = slot['reward_components'].get(key), row.get(key)
            if old is None or new is None or (isinstance(old, float) and math.isnan(old)):
                continue
            compared += 1
            gap = abs(float(old) - float(new))
            largest = max(largest, gap)
            differences += gap > 1e-9
    return {'final_slots': len(slots), 'slots_with_more_than_one_call': calls_seen, 'rescored': sample,
            'skipped_unparsed': skipped, 'values_compared': compared, 'differences': differences,
            'largest_gap': largest}


def stored_means():
    by_level = defaultdict(lambda: defaultdict(list))
    for path in sorted((H / 'runs/reasoning_pilot/bench_pilot_start').glob('slots.shard*.jsonl')):
        for line in open(path):
            slot = json.loads(line)
            components = slot.get('reward_components') or {}
            for key in STAGES + ('detection',):
                value = components.get(key)
                for level in (slot['level'], 'all'):
                    by_level[level][key].append(0.0 if value is None else float(value))
    return {level: {k: sum(v) / len(v) for k, v in row.items()} for level, row in by_level.items()}


def constants():
    table = defaultdict(lambda: defaultdict(list))
    answer = 'INPUT_VECTOR: ""\nEXPECTED_OUTPUT: ""\nDETECTED_FAULTS: ""'
    for example_id, example in problems.items():
        inputs = sim.compile(netlist_text(example)).input_nets
        zeros = {net: 0 for net in inputs}
        ones = {net: 1 for net in inputs}
        a = score(example, [call(zeros)], answer)
        b = score(example, [call(ones)], answer)
        # The reward reads earlier calls from what the tool returned, so that must be real.
        net_list = sim.compile(netlist_text(example))
        frame = sim.fs.fast_fault_sim(zeros, {o: 0 for o in net_list.output_nets}, example['fault'], net_list)
        shown = json.dumps({side: frame[side].to_dict() for side in ('Good Machine', 'Bad Machine')}, default=int)
        both = score(example, [dict(call(zeros), result=shown), call(ones)], answer)
        for key in STAGES:
            assert abs(both[key] - max(a[key], b[key])) < 1e-9 or a['detection'] or b['detection'] or math.isnan(both[key]), (example_id, key)
        level = reference[example_id]['level']
        for key in STAGES:
            for bin_name in (level, 'all'):
                table[bin_name][f'zeros_{key}'].append(a[key])
                table[bin_name][f'ones_{key}'].append(b[key])
                table[bin_name][f'best_of_both_{key}'].append(both[key])
                table[bin_name][f'mean_of_both_{key}'].append((a[key] + b[key]) / 2)
    return {level: {k: sum(v) / len(v) for k, v in row.items()} for level, row in table.items()}


result = {'stored_slots': stored_check()}
print(json.dumps(result['stored_slots'], indent=1), flush=True)
result['real_attempts_pilot_start'] = stored_means()
result['constants'] = constants()
for level in ('easy', 'medium', 'hard', 'very_hard', 'random_never_detects', 'all'):
    real, constant = result['real_attempts_pilot_start'][level], result['constants'][level]
    print(level, ' '.join(f"{key}: real {real[key]:.3f} zeros {constant[f'zeros_{key}']:.3f} "
                          f"best {constant[f'best_of_both_{key}']:.3f} mean {constant[f'mean_of_both_{key}']:.3f} |"
                          for key in STAGES))
(HERE / 'best_attempt_check.json').write_text(json.dumps(result, indent=1))
