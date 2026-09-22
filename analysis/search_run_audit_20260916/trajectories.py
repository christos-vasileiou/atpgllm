"""Inspect selected trajectories and cached training split without model imports."""
from pathlib import Path
import ast
import collections
import hashlib
import json
import re
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
DATA=ROOT/'runs/eval_results_grpo_granite_4.2_8b_policy'

def fields(text):
    text=text.rsplit('</tool_response>',1)[-1].rsplit('</think>',1)[-1]
    return dict(re.findall(r'\b(INPUT_VECTOR|EXPECTED_OUTPUT|DETECTED_FAULTS):\s*"([^"]*)"',text))

def vector(text):
    try:return {k.strip():int(v.strip()) for k,v in (item.rsplit(':',1) for item in text.split(','))}
    except (ValueError,AttributeError):return {}

def main():
    result={}
    greedy=json.loads(next(DATA.glob('*_greedy_*.json')).read_text())
    rows=greedy['per_problem_results']
    hashes={}
    for i,r in enumerate(rows):
        prompt=next(m['content'] for m in r['search_slots'][0]['messages'] if m['role']=='user')
        match=re.search(r"['\"]netlist['\"]\s*:\s*('(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\")",prompt)
        raw=ast.literal_eval(match.group(1))
        hashes[hashlib.sha256(raw.encode()).hexdigest()]=(i,r['fault'])
    result['module_counts']=dict(collections.Counter(r['module_name'] for r in rows))
    for path in DATA.glob('*.json'):
        d=json.loads(path.read_text());method=d['config']['sampling_method']
        if method in ('random','vector_evolutionary'):continue
        totals=collections.Counter()
        for r in d['per_problem_results']:
            for s in r['search_slots']:
                totals['slots']+=1
                f=fields(s.get('final_answer',''))
                expected=vector(f.get('EXPECTED_OUTPUT',''))
                v=vector(f.get('INPUT_VECTOR',''))
                if s['status']=='FINAL':
                    totals['valid_finals']+=1
                    totals['valid_successes']+=int(s['reward_components'].get('detection',0))
                    if v:
                        totals['all_zero_final']+=int(not any(v.values()))
                        totals['all_one_final']+=int(all(v.values()))
                obs=s.get('observations',[])
                if not obs:continue
                totals['with_tool_observation']+=1
                last=obs[-1]
                gm=json.loads(last['result'])['Good Machine']
                bm=json.loads(last['result'])['Bad Machine']
                outputs=last['arguments']['output_vector']
                if isinstance(outputs,str):
                    try:outputs=json.loads(outputs)
                    except ValueError:outputs=vector(outputs)
                po=list(outputs)
                totals['tool_vector_detects']+=int(any(gm.get(k) in (0,1) and bm.get(k) in (0,1) and gm[k]!=bm[k] for k in po))
                totals['final_vector_matches_tool']+=int(v==last['vector'])
                if expected:
                    totals['expected_matches_tool_request']+=int(expected==outputs)
                    totals['expected_matches_good_table']+=int(set(expected)==set(po) and all(expected[k]==gm.get(k) for k in po))
                    totals['expected_matches_bad_table']+=int(set(expected)==set(po) and all(expected[k]==bm.get(k) for k in po))
        result[method]=dict(totals)
    # Read cached Arrow data without loading datasets or contacting Hugging Face.
    import pyarrow as pa
    cached=Path('/home/eng/c/cxv200006/.cache/huggingface/datasets/chrivasileiou___asap7-language-of-test-v2/default/0.0.0/d35cfea64eadf30fb3b39735b0e8d20bffcc3345')
    matched=set();pairs=set();count=0;unique=set()
    files=sorted(cached.glob('*-train-*.arrow'))
    for fi,path in enumerate(files):
        with pa.memory_map(str(path),'r') as source:
            reader=pa.ipc.open_stream(source)
            for batch in reader:
                nets=batch.column(batch.schema.get_field_index('netlist')).to_pylist()
                faults=batch.column(batch.schema.get_field_index('fault')).to_pylist()
                for nl,fault in zip(nets,faults):
                    h=hashlib.sha256(nl.encode()).hexdigest()
                    count+=1;unique.add(h)
                    if h in hashes:
                        matched.add(h)
                        if fault==hashes[h][1]:pairs.add(h)
        print(f'Cached train scan {fi+1}/{len(files)}; matching netlists {len(matched)}',flush=True)
    result['cached_training_overlap']=dict(cache_revision=cached.name,rows=count,unique_netlists=len(unique),
        evaluation_netlists_found=len(matched),evaluation_netlist_fault_pairs_found=len(pairs),
        caveat='Cached dataset revision; exact training buffer/checkpoint exposure not established.')
    (OUT/'trajectory_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='module_counts'},indent=2))

if __name__=='__main__':main()
