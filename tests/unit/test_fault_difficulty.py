"""Independent truth-table answers, sampling and denominator contracts."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/eval/fault_difficulty.py'
spec = importlib.util.spec_from_file_location('fault_difficulty',SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
GATES = {'AND':{'Y':'A & B'}, 'XOR':{'Y':'(~A & B) | (A & ~B)'}, 'INV':{'Y':'~A'}, 'BUF':{'Y':'A'},
         'OR':{'Y':'A | B'}, 'MUX':{'Y':'(A & ~S) | (B & S)'}}


def circuit(body,inputs='a,b',outputs='y',wires=''):
    return m.Circuit(f'module top({inputs},{outputs}); input {inputs}; output {outputs}; '+
                     (f'wire {wires}; ' if wires else '')+body+' endmodule',GATES)


def measure(c,fault,**kwargs):
    return c.characterize(fault,samples=2048,exact_max_inputs=12,seed=7,**kwargs)


def test_and_polarity_and_conditional_observation():
    c = circuit('AND g(.A(a),.B(b),.Y(y));')
    assert c.cc['y'] == (2,3)
    assert c.co['a'] == 2
    sa0,sa1 = measure(c,'sa0 y'),measure(c,'sa1 y')
    assert sa0['random_detection_probability'] == .25
    assert sa1['random_detection_probability'] == .75
    assert sa0['activation_cost'] == 3 and sa1['activation_cost'] == 2
    pi = measure(c,'sa0 a')
    assert pi['site_depth'] == 0 and pi['activation_probability'] == .5
    assert pi['conditional_observability'] == .5 and pi['random_detection_probability'] == .25


def test_reconvergence_can_make_a_shallow_fault_undetectable():
    c = circuit('BUF g1(.A(a),.Y(n)); XOR g2(.A(n),.B(a),.Y(y));',inputs='a',wires='n')
    r = measure(c,'sa0 a')
    assert r['scoap_cost'] is not None  # finite heuristic, but cancellation
    assert r['random_detection_probability'] == 0
    assert r['zero_detection_status'] == 'proven_undetectable_exhaustive'
    assert r['reconvergent_gates'] == 1


def test_alias_physical_stems_and_constant_bus_indices():
    c = circuit("assign n=a; BUF g(.A(n),.Y(y));",inputs='a',wires='n')
    assert measure(c,'sa0 n')['site_depth'] == 0
    assert measure(c,'sa0 n')['random_detection_probability'] == measure(c,'sa0 a')['random_detection_probability']
    c = circuit("assign y=1'b0;",inputs='a')
    assert measure(c,'sa0 y')['structural_unreachable']
    assert measure(c,'sa1 y')['random_detection_probability'] == 1
    c = m.Circuit('module top(a,y); input [6:2] a; output y; BUF g(.A(a[6]),.Y(y)); endmodule',GATES)
    assert len(c.inputs) == 5 and 'a[6]' in c.inputs and 'a[0]' not in c.inputs


def test_mux_and_tied_inputs():
    c = circuit('MUX g(.A(a),.B(b),.S(s),.Y(y));',inputs='a,b,s')
    assert c.cc['y'] == (3,3)
    assert c.co['a'] == 2
    assert c.co['s'] == 3
    c = circuit('XOR g(.A(a),.B(a),.Y(y));',inputs='a')
    assert measure(c,'sa0 y')['random_detection_probability'] == 0
    assert measure(c,'sa1 y')['random_detection_probability'] == 1


@pytest.mark.parametrize('body',[
    'UNKNOWN g(.A(a),.Y(y));', 'AND g(.A(a),.Y(y));',
    'BUF g(.A(y),.Y(y));', 'BUF g(.A(z),.Y(y));',
    'BUF g1(.A(a),.Y(y)); BUF g2(.A(b),.Y(y));', 'assign y=a & b;'])
def test_unknowns_fail_closed(body):
    with pytest.raises(ValueError):
        circuit(body)


def test_monte_carlo_zero_hits_are_not_proven_untestable():
    c = circuit('XOR g(.A(a),.B(a),.Y(y));',inputs='a')
    r = c.characterize('sa0 y',samples=100,exact_max_inputs=0,seed=7)
    assert r['zero_detection_status'] == 'unresolved_zero_hits'
    assert r['random_detection_ci95'][1] > 0


def test_reservoir_is_reproducible_deduplicated_and_not_a_prefix():
    rows = [dict(netlist=f'net{i//2}',fault=f'sa{i%2} a') for i in range(100)]
    selected,audit = m.select_records(rows+rows,10,seed=17)
    assert len(selected) == 10
    assert audit['counts']['eligible_unique_problems'] == 100
    assert audit['counts']['duplicate_rows'] == 100
    assert selected == m.select_records(rows+rows,10,seed=17)[0]
    assert any(int(r['netlist'][3:]) > 20 for r in selected)
    legacy,_ = m.select_records(rows,10,mode='legacy_prefix')
    assert len({r['netlist'] for r in legacy}) == 10
    all_rows,_ = m.select_records(rows,-1)
    assert len(all_rows) == 100


def test_manifest_roundtrip_and_tamper(tmp_path):
    path = tmp_path/'manifest.json'
    rows = [dict(netlist='net',fault='sa0 a')]
    m.save_manifest(path,rows,{'mode':'test'})
    assert m.load_manifest(path)[0] == rows
    data = json.loads(path.read_text()); data['examples'][0]['fault']='sa1 a'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        m.load_manifest(path)


def test_counts_do_not_use_threshold_mode_and_keep_missing_slots():
    rewards = [dict(input_vector_acc=1,expected_output_acc=0,fault_detected_by_pred_input_vector_acc=1)]
    counts = m.outcome_counts(rewards,2)
    assert counts == dict(D=1,failed_slots=1)
    rows = [dict(problem_id='a',circuit_id='a',status='unsupported',fault='sa0 a',num_completions=2,outcomes=counts)]
    summary = m.summarize(rows,[1,2])
    assert summary['overall']['D_pass@1']['problem_mean'] == .5
    assert 'U_pass@1' not in summary['overall']
    assert summary['overall']['D_pass@2']['problem_mean'] == 1
    assert summary['overall']['failed_slots'] == 1


def test_packed_simulation_matches_scalar_truth_table_exhaustively():
    c = circuit('AND g1(.A(a),.B(b),.Y(n)); OR g2(.A(n),.B(a),.Y(y));',wires='n')
    for site in ('a','b','n','y'):
        for stuck in (0,1):
            detections = 0
            for a,b in ((0,0),(0,1),(1,0),(1,1)):
                good = (a & b) | a
                aa,bb = (stuck if site=='a' else a),(stuck if site=='b' else b)
                n = stuck if site=='n' else aa & bb
                bad = stuck if site=='y' else n | aa
                detections += good != bad
            assert measure(c,f'sa{stuck} {site}')['random_detection_probability'] == detections/4


def test_stratification_keeps_rare_hard_faults_and_audits_pool():
    class Analyzer:
        def analyze(self,r):
            return dict(status='ok',fault_type='sa0',location_class='internal',
                        random_detection_probability=.5 if r['netlist'] != 'hard' else 1/1024)
    rows = [dict(netlist=str(i),fault='sa0 a') for i in range(99)]+[dict(netlist='hard',fault='sa0 a')]
    chosen,audit = m.select_records(rows,10,mode='stratified',pool_size=100,analyzer=Analyzer())
    assert len(chosen) == 10 and any(r['netlist']=='hard' for r in chosen)
    assert sum(audit['candidate_strata'].values()) == 100
    assert not audit['population_estimate']


def test_circuit_macro_does_not_overweight_many_faults_and_costs_are_retained():
    good = dict(status='unsupported',fault='sa0 x',num_completions=1,
                outcomes=dict(D=1,Vx=1,S=1,U=1,failed_slots=0),saved_search_usage=dict(attempts=4))
    rows = [dict(good,problem_id=str(i),circuit_id='large') for i in range(9)]
    rows.append(dict(good,problem_id='bad',circuit_id='small',outcomes=dict(D=0,Vx=0,S=0,U=0,failed_slots=1)))
    out = m.summarize(rows)['overall']
    assert out['D_pass@1']['problem_mean'] == .9
    assert out['D_pass@1']['circuit_macro'] == .5
    assert out['search_usage']['attempts'] == 40


def test_declared_net_identifiers_are_not_confused_with_cell_pins():
    c = m.Circuit('module top(a,y); input a; output y; wire \\a[1] ; assign \\a[1] = a; BUF g(.A(\\a[1] ),.Y(y)); endmodule',GATES)
    assert measure(c,'sa0 \\a[1]')['random_detection_probability'] == .5


def test_evaluator_integration_saves_metrics_and_reuses_manifest(tmp_path,monkeypatch):
    """Execute the real evaluate() body with fake model/dataset boundaries."""
    import ast
    from collections import defaultdict
    from dataclasses import dataclass,asdict
    import os
    import random
    import time
    import types
    from typing import Any,Dict,List,Optional
    import numpy as np
    @dataclass
    class Config:
        flag: int = 1
        @classmethod
        def load(cls,_):
            return cls()
    class Tokenizer:
        pad_token='pad'; eos_token='eos'
        def encode(self,text,**kwargs):
            return [1]
    config = tmp_path/'gates.json'
    config.write_text(json.dumps({'gate_funcs':GATES}))
    paths = types.ModuleType('atpgllm.training._paths')
    paths.resolve_sim_config_path=lambda _:config
    provenance = types.ModuleType('atpgllm.training.simulator_provenance')
    provenance.runtime_provenance=lambda:dict(backend='test')
    monkeypatch.setitem(sys.modules,'fault_difficulty',m)
    monkeypatch.setitem(sys.modules,paths.__name__,paths)
    monkeypatch.setitem(sys.modules,provenance.__name__,provenance)
    rows = [dict(netlist='module top(a,y); input a; output y; BUF g(.A(a),.Y(y)); endmodule',
                 fault='sa0 a',module_name='top')]
    def run_strategy(strategy,prompts,records,n):
        rewards = [dict(input_vector_acc=1,expected_output_acc=0,fault_detected_by_pred_input_vector_acc=1),
                   dict(search_failure_logonly=1)]
        result = types.SimpleNamespace(completions=['answer',''],slots=[dict(status='FINAL',usage={}),dict(status='INFRA_ERROR',usage={})])
        return ['answer',''],rewards,[result]
    ns=dict(Path=Path,Optional=Optional,List=List,Dict=Dict,Any=Any,np=np,os=os,json=json,
            py_random=random,time=time,asdict=asdict,defaultdict=defaultdict,
            SearchConfig=Config,torch=types.SimpleNamespace(manual_seed=lambda _:None),
            AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**k:Tokenizer()),
            MODEL_FREE_STRATEGY_NAMES={'random'},load_dataset=lambda *a,**k:rows,
            format_eval_prompt=lambda r,t,return_messages=False:('prompt',[]) if return_messages else 'prompt',
            RewardFunctionFactory=lambda **k:object(),Verifier=lambda _:object(),
            make_strategy=lambda *a,**k:object(),run_strategy_batch=run_strategy,
            is_completion_correct=lambda r,threshold_mode:False,PROTOCOL_VERSION='test',
            tqdm=lambda iterable,**kwargs:iterable)
    tree = ast.parse(SCRIPT.with_name('evaluate_model.py').read_text())
    functions = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('evaluate','estimate_pass_at_k')]
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(SCRIPT),'exec'),ns)
    adapter = tmp_path/'adapter'; adapter.mkdir()
    (adapter/'adapter_config.json').write_text(json.dumps({'base_model_name_or_path':'fake'}))
    output=tmp_path/'result.json'; manifest=tmp_path/'manifest.json'
    args=dict(adapter=adapter,sampling_method='random',num_completions=2,k_values=[1,2],
              max_eval_samples=1,output_file=str(output),threshold_mode='full_accuracy')
    manifest=tmp_path/'result.manifest.json'
    result=ns['evaluate'](**args)
    assert result['pass_at_k']['pass@1'] == 0  # chosen threshold
    assert result['difficulty_analysis']['overall']['D_pass@1']['problem_mean'] == .5
    assert 'U_pass@1' not in result['difficulty_analysis']['overall']
    assert manifest.exists() and (tmp_path/'result.difficulty.problems.csv').exists()
    saved=json.loads(output.read_text())['per_problem_results'][0]
    assert saved['rewards_summary']['fault_detection_rate'] == .5
    assert saved['difficulty']['outcomes']['failed_slots'] == 1
    ns['load_dataset']=lambda *a,**k:pytest.fail('Frozen manifest must not reload source')
    ns['evaluate'](**args)


def test_uniform_limit_is_independent_of_stratification_pool_size():
    rows=[dict(netlist=str(i),fault='sa0 a') for i in range(20)]
    selected,_=m.select_records(rows,10,pool_size=2)
    assert len(selected)==10
    with pytest.raises(ValueError):
        m.select_records(rows,10,pool_size=2,mode='stratified')


def test_interface_counts_include_each_output_port_and_group_difficulty():
    c = circuit('BUF g(.A(a),.Y(y)); assign z=y;', inputs='a', outputs='y,z')
    row = measure(c, 'sa0 a')
    assert row['num_inputs'] == 1
    assert row['num_outputs'] == 2
    assert row['num_output_signals'] == 1
    row.update(problem_id='easy', circuit_id='one', num_completions=2,
               outcomes=dict(D=1,Vx=2,S=1,U=1,failed_slots=0))
    hard = dict(row, problem_id='hard', num_inputs=32, num_outputs=8,
                random_detection_probability=1/128)
    unknown = dict(problem_id='unknown', circuit_id='two', fault='sa0 a', status='unsupported')
    report = m.summarize([row, hard, unknown])['by']
    assert report['num_outputs']['2']['D_pass@1']['problem_mean'] == .5
    assert report['num_inputs_difficulty']['1|easy_p>=0.25']['problems'] == 1
    assert report['num_outputs_difficulty']['5..8|hard_0.00390625<=p<0.0625']['problems'] == 1
    assert report['input_output_counts']['17..32|5..8']['problems'] == 1
    assert report['num_outputs']['unknown_or_infinite']['unsupported'] == 1
    assert sum(g['problems'] for g in report['num_outputs'].values()) == 3


def test_detection_only_replay_requires_propagation_to_any_output(tmp_path):
    import subprocess
    # a=1 activates sa0 a in both vectors; b=0 masks it, b=1 exposes it on y.
    # z never changes, so requiring all outputs to differ would be incorrect.
    config = tmp_path/'gates.json'
    config.write_text(json.dumps({'gate_funcs':GATES}))
    record = dict(netlist='module top(a,b,y,z); input a,b; output y,z; '
                  'AND g(.A(a),.B(b),.Y(y)); assign z=b; endmodule', fault='sa0 a')
    manifest = tmp_path/'manifest.json'
    m.save_manifest(manifest,[record],{})
    results = tmp_path/'saved.json'
    results.write_text(json.dumps({'per_problem_results':[dict(problem_id=m.identity(record),
        fault=record['fault'],num_completions=3,
        completions=['INPUT_VECTOR: "a: 1, b: 0"',
                     'INPUT_VECTOR: "a: 1, b: 1"',
                     'INPUT_VECTOR: "a: 1, b: 1"\nEXPECTED_OUTPUT: "y: 0, z: 0"'],
        search_slots=[dict(status='FINAL') for _ in range(3)])]}))
    output=tmp_path/'report.json'
    run=subprocess.run([sys.executable,str(SCRIPT.with_name('analyze_fault_difficulty.py')),
        '--evaluation-results','--input',str(results),'--problem-source',str(manifest),
        '--config',str(config),'--output',str(output)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    report=json.loads(output.read_text())
    assert report['overall']['D_pass@1']['problem_mean']==pytest.approx(2/3)
    assert not any(k.startswith(('U_','S_','Vx_')) for k in report['overall'])
    rows=[json.loads(l) for l in output.with_suffix('.problems.jsonl').read_text().splitlines()]
    assert rows[0]['outcomes']==dict(D=2,failed_slots=0)
    assert rows[0]['num_inputs']==2 and rows[0]['num_outputs']==2
    assert 'U_rate' not in output.with_suffix('.problems.csv').read_text()


def test_detection_report_strips_legacy_outcomes_without_mutating_source(tmp_path):
    row=dict(problem_id='p',circuit_id='c',fault='sa0 a',status='unsupported',
             num_completions=1,outcomes=dict(D=1,U=0,S=0,Vx=1,failed_slots=0))
    m.write_report(tmp_path/'report.json',[row],{})
    saved=json.loads((tmp_path/'report.problems.jsonl').read_text())
    assert saved['outcomes']==dict(D=1,failed_slots=0)
    assert row['outcomes']['U']==0
