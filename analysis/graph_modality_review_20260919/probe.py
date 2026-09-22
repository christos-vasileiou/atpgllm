"""Offline CPU counterexamples for the graph-modality implementation review.

Run from the repository root with PYTHONPATH=. and the project Python.
This records current behavior; it does not modify implementation or checkpoints.
"""
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('PYG_HOME', '/tmp/graph-review-pyg')
print('Importing graph stack...', flush=True)
import torch
from torch import nn
from torch_geometric.data import Batch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from atpgllm.graph.dataset import record_to_pyg, render_prompt_and_answer
from atpgllm.graph.gate_features import GateAttributeVocab
from atpgllm.graph.models_stage1 import GraphQFormer, NetlistGraphEncoder, Stage1Outputs
from atpgllm.graph.netlist_parser import parse_verilog_to_graph
from atpgllm.graph.stage2_model import _build_prompt_answer_ids
from atpgllm.graph.train_stage1 import _build_gtm_batch
from atpgllm.multimodal.model import GraphConditionedCausalLM

torch.set_num_threads(1)
torch.manual_seed(42)
ROOT = Path(__file__).resolve().parents[2]
gate_funcs = json.loads((ROOT / 'atpgllm/training/data/sim_config.json').read_text())['gate_funcs']
vocab = GateAttributeVocab(gate_funcs)
results = {'versions': {'torch': torch.__version__}, 'probes': {}}
out = results['probes']

def graph(body, fault='sa0 y'):
    return record_to_pyg({'netlist': body, 'fault': fault}, gate_funcs, vocab)

def consumed_equal(a, b):
    return all(torch.equal(getattr(a, k), getattr(b, k)) for k in
               ('gate_attrs', 'structural_feats', 'fault_feats', 'edge_index'))

def model(dropout=0.0):
    encoder = NetlistGraphEncoder(vocab, node_dim=8, gin_hidden_dim=8,
                                 gin_num_layers=1, per_attr_dim=2,
                                 attr_mlp_hidden=8, dropout=dropout)
    qformer = GraphQFormer(8, hidden_dim=8, num_queries=2, num_layers=2,
                          num_heads=2, ffn_dim=16, dropout=0.0)
    lm = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16,
                                    intermediate_size=32, num_hidden_layers=1,
                                    num_attention_heads=2, num_key_value_heads=2,
                                    attention_dropout=0.0))
    return GraphConditionedCausalLM(encoder, qformer, lm)

print('Running counterexamples...', flush=True)
out['cell_collisions'] = [
    {'cells': [a, b], 'same_features': vocab.encode(a) == vocab.encode(b),
     'functions': [gate_funcs[a], gate_funcs[b]]}
    for a, b in [('MAJx2_ASAP7_75t_R', 'MAJIxp5_ASAP7_75t_R'),
                 ('AO211x2_ASAP7_75t_R', 'AO22x1_ASAP7_75t_R')]
]
net_a = 'module m(a,b,c,y); input a,b,c; output y; AO21x1_ASAP7_75t_R U1 (.A1(a),.A2(b),.B(c),.Y(y)); endmodule'
net_b = net_a.replace('.A2(b),.B(c)', '.A2(c),.B(b)')
ga, gb = graph(net_a), graph(net_b)
m = model().eval()
with torch.no_grad():
    pa = m.graph_soft_prompt(Batch.from_data_list([ga]))
    pb = m.graph_soft_prompt(Batch.from_data_list([gb]))
out['pin_role_collision'] = {
    'encoder_inputs_equal': consumed_equal(ga, gb),
    'prefix_max_abs_difference': float((pa-pb).abs().max()),
    'input_assignment': {'a': 0, 'b': 0, 'c': 1},
    'correct_good_outputs': [1, 0],
    'sa0_y_detected': [True, False],
}
out['primary_input_fault_collision'] = {
    'sa0_a_vs_sa0_b_same_encoder_inputs': consumed_equal(graph(net_a, 'sa0 a'), graph(net_a, 'sa0 b')),
}
parsed = parse_verilog_to_graph('module m(a,y); input [7:4] a; output y; endmodule', gate_funcs)
out['bus_indices'] = {'declared': '[7:4] a', 'parsed': parsed.inputs}
out['ansi_ports'] = {'parsed_inputs': parse_verilog_to_graph(net_a.replace('module m(a,b,c,y); input a,b,c; output y;', 'module m(input a, input b, input c, output y);'), gate_funcs).inputs}
alias_net = 'module m(a,y); input a; output y; wire n,t; INVx1_ASAP7_75t_R U1 (.A(a),.Y(n)); assign t=n; INVx1_ASAP7_75t_R U2 (.A(t),.Y(y)); endmodule'
out['assign_alias'] = {'edge_index': graph(alias_net).edge_index.tolist()}

two = graph('module m(a,b,y); input a,b; output y; wire n; AND2x2_ASAP7_75t_R U1 (.A(a),.B(b),.Y(n)); INVx1_ASAP7_75t_R U2 (.A(n),.Y(y)); endmodule')
batch = Batch.from_data_list([two])
frozen = model(dropout=0.1)
frozen.set_graph_policy('frozen')
before = {n: v.clone() for n,v in frozen.graph_encoder.named_buffers()}
with torch.no_grad():
    p1 = frozen.graph_soft_prompt(batch)
    p2 = frozen.graph_soft_prompt(batch)
out['frozen_module_drift'] = {
    'encoder_training_mode': frozen.graph_encoder.training,
    'trainable_encoder_parameters': sum(p.numel() for p in frozen.graph_encoder.parameters() if p.requires_grad),
    'changed_buffers': [n for n,v in frozen.graph_encoder.named_buffers() if not torch.equal(before[n],v)],
    'prefix_max_abs_difference': float((p1-p2).abs().max()),
}
try:
    model().graph_soft_prompt(Batch.from_data_list([ga]))
    out['single_gate_training'] = {'error': None}
except ValueError as exc:
    out['single_gate_training'] = {'error': str(exc)}

# Prove the bridge can carry answer-loss gradients through a real causal decoder.
positive = model().eval()
ids = torch.tensor([[1,2,3,4]])
labels = torch.tensor([[-100,-100,3,4]])
loss = positive(batch, ids, torch.ones_like(ids), labels=labels).loss
loss.backward()
out['real_causal_decoder_gradient'] = {
    'loss': float(loss.detach()),
    'projector_gradient_l1': sum(float(p.grad.abs().sum()) for p in positive.graph_to_llm.parameters() if p.grad is not None),
    'encoder_gradient_l1': sum(float(p.grad.abs().sum()) for p in positive.graph_encoder.parameters() if p.grad is not None),
}

class Tokens:
    eos_token_id = 9
    def encode(self, text, **kwargs):
        return [1,2] if text == 'prompt' else [3,4,5]

out['exact_answer_budget'] = dict(zip(('ids','answer_start'), _build_prompt_answer_ids(Tokens(), 'prompt','answer',4)))
out['truncated_answer_budget'] = dict(zip(('ids','answer_start'), _build_prompt_answer_ids(Tokens(), 'prompt','answer',3)))
fake = SimpleNamespace(projected_graph_repr_per_query=lambda o: torch.ones(1,2,3),
                       projected_text_repr=lambda o: torch.ones(1,3))
g,t,labels = _build_gtm_batch(None, fake, torch.device('cpu'))
out['singleton_matching'] = {'same_positive_negative': bool(torch.equal(g[0],g[1]) and torch.equal(t[0],t[1])), 'labels': labels.tolist()}

# A configurable Q-Former can accidentally have no graph-attention layers.
q = GraphQFormer(8, hidden_dim=8, num_queries=2, num_layers=1,
                 num_heads=2, cross_attn_every_n=2).eval()
with torch.no_grad():
    a=q(torch.randn(3,8),torch.zeros(3,dtype=torch.long))
    b=q(torch.randn(3,8),torch.zeros(3,dtype=torch.long))
out['zero_cross_attention_configuration'] = {'outputs_identical': bool(torch.equal(a,b))}
tokenizer = AutoTokenizer.from_pretrained(ROOT / 'runs/sft_granite_4.2_8b_repaired/checkpoint-90', local_files_only=True)
prompt, answer = render_prompt_and_answer({
    'netlist': net_a, 'fault': 'sa0 y', 'module_name': 'm',
    'system_content': 'ATPG', 'user_content': 'Target {fault} with {netlist}',
    'reasoning_content': 'Find a vector.', 'answer_content': 'done',
    'input_vector': '{}', 'expected_output': '{}',
}, tokenizer, compact_netlist=True)
out['granite_chat_boundary'] = {'prompt_tail': prompt[-50:], 'answer_head': answer[:50],
    'opens_think_twice': prompt.rstrip().endswith('<think>') and answer.startswith('<think>')}
destination = Path(__file__).with_name('probe_results.json')
destination.write_text(json.dumps(results, indent=2)+'\n')
print(json.dumps(results, indent=2), flush=True)
