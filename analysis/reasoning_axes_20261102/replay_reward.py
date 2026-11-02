"""Score stored trajectories and pilot rows with the reward code on PYTHONPATH. CPU only.
    PYTHONPATH=<worktree> python replay_reward.py <out.json> [--pilot]"""
import json, os, random, sys
from pathlib import Path
os.environ['FAULT_SIM_BACKEND'] = 'fast'
H = Path('/proj/trela/christos/transformers_atpg/libatpgllm_hard-faults')
import atpgllm.llm.reward_funcs as RF
from atpgllm.training.reward_function_factory import RewardFunctionFactory as F
from atpgllm.training.eval_report import Simulator
sim = Simulator()
PROMPT = "target<|im_start|>assistant\n<think>\n"
def strip(text):
    return text[len("<think>\n"):] if text.startswith("<think>\n") else text[len("<think>"):] if text.startswith("<think>") else text
def vec(args):
    out = dict(args)
    for k in ("input_vector", "output_vector"):
        if isinstance(out.get(k), str):
            try: out[k] = json.loads(out[k])
            except json.JSONDecodeError: pass
    return out
def trace_of(messages):
    turns, calls = [], []
    gen = [m for m in messages if m["role"] in ("assistant", "tool")]
    for i, m in enumerate(gen):
        if m["role"] == "assistant":
            turns.append(strip(m.get("content") or ""))
            if m.get("tool_calls"):
                result = gen[i + 1]["content"] if i + 1 < len(gen) and gen[i + 1]["role"] == "tool" else None
                calls.append({"arguments": vec(m["tool_calls"][0]["function"]["arguments"]), "result": result})
    return {"turns": turns, "calls": calls}
def score(trace, netlist, fault, module):
    return RF.test_generation_grpo_reward([PROMPT], [""], netlists=[sim.compile(netlist)], tool_trace=[trace], fault=[fault], module_name=[module],
        lib_gate_funcs=sim.gates, fault_sim=sim.fs.fast_fault_sim, fault_fn=F.fault_fn, simulation_fn=F.simulation_fn, input_vector_fn=F.input_vector_fn,
        expected_output_fn=F.expected_output_fn, detected_faults_fn=F.detected_faults_fn)[0]
import pyarrow.parquet as pq
out = {}
rows = {'|'.join([str(r.get('netlist_id') or r.get('netlist')), str(r['fault'])]): r for r in pq.read_table(H / 'runs/rft_v4_a100/eval/val_problems.parquet').to_pylist()}
n = 0
for path in sorted((H / 'runs/rft_v4_a100/round_5/eval_val').glob('slots*.jsonl')):
    for line in open(path):
        s = json.loads(line); r = rows[s['example_id']]
        if not s.get('messages'): continue
        out['val|' + s['example_id']] = {"level": r["difficulty"], **score(trace_of(s['messages']), r['netlist'], r['fault'], r['module_name'])}
        n += 1
print('val scored', n, flush=True)
if '--pilot' in sys.argv:
    rng = random.Random(3)
    lines = open(H / 'runs/reasoning_pilot/sft.jsonl').readlines()
    demo = [l for l in lines if '"kind": "demonstration"' in l[-200:]]; retry = [l for l in lines if '"kind": "retry"' in l[-200:]]
    print(len(demo), len(retry), flush=True)
    pr = {}
    for path in ('runs/rft_v4_a100/round_5/problems.parquet',):
        pass
    import re
    for kind, pool in (('demonstration', rng.sample(demo, 250)), ('retry', rng.sample(retry, 150))):
        for i, line in enumerate(pool):
            d = json.loads(line)
            user = next(m['content'] for m in d['messages'] if m['role'] == 'user')
            m = re.search(r"'netlist': '(.*?)'\}", user, re.S)
            netlist = m.group(1).encode().decode('unicode_escape')
            module = re.search(r"module\s+(\S+)", netlist).group(1)
            out[f'{kind}|{i}'] = {"level": d['pilot']['difficulty'], **score(trace_of(d['messages']), netlist, d['pilot']['fault'], module)}
    print('pilot scored', flush=True)
Path(sys.argv[1]).write_text(json.dumps(out))
