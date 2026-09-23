"""License-free tool/reward contract and pipeline tests."""
import asyncio
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import time

import pandas as pd
import pytest
import torch
from atpgllm.training.tools import ToolHelper, ToolScheduler, execute_tool_batch
from atpgllm.llm.reward_funcs import test_generation_grpo_reward as score, active_reward_objectives
from atpgllm.training.simulator_provenance import SimulatorProvenanceCallback

DOCUMENT = {'doc_id':'doc','netlist':'module design(input a,output y); BUF U0(.A(a),.Y(y)); endmodule'}
PROMPT = 'Generate a test vector targeting the "sa0 y" using the netlist '+repr(DOCUMENT)
CALL = {'name':'fault_simulation_tool','arguments':{'input_vector':{'a':1},'output_vector':{'y':0},'fault':'sa0 y','doc_id':'doc'}}


def test_tool_binding_uses_user_context():
    context=ToolHelper.context([{'role':'system','content':'Example: sa1 other'}, {'role':'user','content':PROMPT}])
    assert context=={'netlist':DOCUMENT['netlist'],'expected_doc_id':'doc','expected_fault':'sa0 y'}
    from atpgllm.training.tools import fault_simulation_tool_handler
    with pytest.raises(ValueError,match='document'):
        asyncio.run(fault_simulation_tool_handler(**dict(CALL['arguments'],doc_id='spoof'),**context))


@pytest.mark.parametrize('field,value',[
    ('input_vector',[1]), ('input_vector',None), ('input_vector',{'a':2}),
    ('input_vector','a: 2'), ('input_vector',{'a':'2'}), ('output_vector',[0]),
    ('fault','sa2 y'),
])
def test_malformed_model_vectors_are_tool_errors_not_infrastructure(field,value):
    from atpgllm.training.tools import fault_simulation_tool_handler
    call=dict(CALL,arguments=dict(CALL['arguments'],**{field:value}))
    [(message,infrastructure)]=execute_tool_batch([call],[PROMPT],{'fault_simulation_tool':fault_simulation_tool_handler})
    assert message.startswith('Tool execution failed') and infrastructure is False


def test_tool_batch_preserves_order_and_classifies_failures():
    from tetramax_seats import SimulationError
    def handler(**kwargs):
        if kwargs['doc_id']=='invalid': raise ValueError('invalid')
        if kwargs['doc_id']=='failure': raise SimulationError('offline')
        if kwargs['doc_id']=='simulator_bug': raise IndexError('list index out of range')
        return kwargs['expected_fault']
    docs=('doc','failure','invalid','simulator_bug')
    calls=[dict(CALL,arguments=dict(CALL['arguments'],doc_id=doc)) for doc in docs]
    results=execute_tool_batch(calls,[PROMPT]*len(docs),{'fault_simulation_tool':handler})
    assert results[0]==('sa0 y',False)
    assert results[1][1] is True
    assert results[2][1] is False
    assert results[3]==('Tool execution failed: list index out of range',False)


def test_pipeline_ready_trajectory_advances_while_another_waits(monkeypatch):
    monkeypatch.setenv('TMAX_PIPELINED_TOOLS','1')
    blocked=threading.Event()
    def handler(**kwargs):
        if kwargs['doc_id']=='slow':
            assert blocked.wait(3)
        return kwargs['doc_id']
    trainer=SimpleNamespace(accelerator=SimpleNamespace(device=torch.device('cpu'),num_processes=1))
    slow=dict(CALL,arguments=dict(CALL['arguments'],doc_id='slow'))
    with ToolScheduler({'fault_simulation_tool':handler}) as scheduler:
        try:
            calls,indices,outcomes=scheduler.take(trainer,[slow,CALL],[0,1],[PROMPT,PROMPT])
            assert indices==[1] and outcomes==[('doc',False)]
            assert set(scheduler.pending)=={0}
        finally:
            blocked.set()
        calls,indices,outcomes=scheduler.take(trainer,[],[],[PROMPT,PROMPT])
        assert indices==[0] and outcomes==[('slow',False)]
        assert not scheduler.pending


def test_native_rewards_mask_unknowns_and_exclude_internal_objective(monkeypatch):
    monkeypatch.setenv('FAULT_SIM_BACKEND','tetramax')
    monkeypatch.setenv('TMAX_REWARD_PROFILE','po')
    frame=pd.DataFrame({'Good Machine':[1,1,'x'],'Bad Machine':[1,0,'x'],
                        'PIs':[True,False,False],'POs':[False,True,True]},index=['a','y','z'])
    def simulator(*args,**kw):
        return frame,{'tetramax_available':True,'tetramax_detected':True}
    result=score(['sa0 y']*2,['answer']*2,
        netlists=[SimpleNamespace(input_nets=['a'],output_nets=['y','z'])]*2,
        fault_fn=lambda *a,**k:[('sa0','y')], simulation_fn=lambda _:[],
        input_vector_fn=lambda _:['a:1'],expected_output_fn=lambda _:['y:1,z:0'],
        detected_faults_fn=lambda _:['sa0 y'],fault_sim=simulator)
    assert active_reward_objectives()[0]==('detection','fidelity','format')
    assert len(result)==2
    for reward in result:
        assert reward['detection']==1 and reward['fidelity']==1
        assert reward['known_po_fraction_logonly']==0.5
        assert 'activation' not in reward
        assert not reward['simulator_error_logonly']


def test_reward_profile_resume_guard(tmp_path,monkeypatch):
    from atpgllm.training import simulator_provenance as module
    provenance={'backend':'tetramax','profile':'po'}
    monkeypatch.setattr(module,'runtime_provenance',lambda:provenance)
    with pytest.raises(ValueError,match='differs'):
        SimulatorProvenanceCallback(tmp_path)
    (tmp_path/'simulator_provenance.json').write_text(json.dumps(provenance))
    callback=SimulatorProvenanceCallback(tmp_path)
    callback.on_save(SimpleNamespace(output_dir=str(tmp_path)),SimpleNamespace(global_step=1,is_world_process_zero=True),None)
    assert json.loads((tmp_path/'checkpoint-1/simulator_provenance.json').read_text())==provenance


@pytest.mark.parametrize('trainer_module,class_name',[
    ('tool_calling_grpo_trainer','ToolCallingGRPOTrainer'),
    ('dual_adapter_grpo_trainer','DualAdapterGRPOTrainer'),
])
def test_trainer_loop_preserves_observations_masks_and_indices(monkeypatch,trainer_module,class_name):
    import importlib
    from collections import defaultdict
    module=importlib.import_module('atpgllm.training.'+trainer_module)
    cls=getattr(module,class_name)
    monkeypatch.setenv('TMAX_PIPELINED_TOOLS','1')
    monkeypatch.setattr(module,'restore_generation_prefix',lambda text,*a:text)
    monkeypatch.setattr(module,'revert_assistant_completion',lambda text,**kw:{'role':'assistant','content':text})
    release=threading.Event()
    observed=[]
    def handler(**kwargs):
        if kwargs['doc_id']=='slow': assert release.wait(3)
        return kwargs['doc_id']
    tokenizer=SimpleNamespace(apply_chat_template=lambda *a,**kw:[1,2,3],
                              batch_decode=lambda ids,**kw:['done']*len(ids))
    trainer=SimpleNamespace(accelerator=SimpleNamespace(device=torch.device('cpu'),num_processes=1),
        processing_class=tokenizer,tools=[],chat_template_kwargs={},
        model=SimpleNamespace(training=True),_vllm_max_model_len=100,max_completion_length=20,
        tool_functions={'fault_simulation_tool':handler},_metrics={'train':defaultdict(list)},
        _parse_tool_call=lambda text:dict(CALL,arguments=dict(CALL['arguments'],doc_id=text)) if text in ('slow','fast') else None)
    def generate(conversations):
        observed.extend(conv[-1]['content'] for conv in conversations)
        if observed==['fast']:
            release.set()  # Slow simulation overlaps this generation boundary.
        return [[1,2,3] for _ in conversations],[[4] for _ in conversations],[[-0.3] for _ in conversations],{}
    trainer._generate_tool_continuation=generate
    result=cls._custom_tool_call_loop_impl(trainer,[PROMPT]*2,[[1],[1]],[[2],[2]],['slow','fast'],[[-0.1],[-0.2]])
    masks,completions,ids,logprobs,count,failures=result
    assert observed==['fast','slow']
    assert masks==[[1,0,1],[1,0,1]] and ids==[[2,3,4],[2,3,4]]
    assert logprobs==[[-0.1,0.0,-0.3],[-0.2,0.0,-0.3]]
    assert count==2 and failures==0
    assert [[m['role'] for m in conv] for conv in completions]==[['assistant','tool','assistant']]*2
    assert [conv[1]['content'] for conv in completions]==['slow','fast']
