"""CPU-only reproductions; no model loading, network, or training changes.

Run with /work/cxv200006/myenv/bin/python analysis/run_58vu7m6x/reproduce_findings.py
from the atpgllm repository root. Extracts the installed TRL loss verbatim via AST
and supplies a tiny synthetic forward pass to isolate accumulation scaling.
"""

import ast
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch


REPO = Path(__file__).resolve().parents[2]
TRL = Path('/work/cxv200006/myenv/lib/python3.11/site-packages/trl/trainer/grpo_trainer.py')


def reproduce_loss_scaling():
    tree = ast.parse(TRL.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GRPOTrainer')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_compute_loss')
    fn.decorator_list = []
    scope = {'torch': torch, 'nanmin': lambda x: x.min(), 'nanmax': lambda x: x.max()}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(TRL), 'exec'), scope)
    outputs = {}
    for generation_steps in (16, 384):
        parameter = torch.tensor(0.0, requires_grad=True)
        fake = SimpleNamespace(
            _get_per_token_logps_and_entropies=lambda *a, **kw: (
                parameter.expand(1, 2), torch.zeros(1, 2)),
            top_entropy_quantile=1.0, tools=None,
            importance_sampling_level='sequence', beta=0.0,
            loss_type='dapo', epsilon_low=0.2, epsilon_high=0.2,
            args=SimpleNamespace(delta=None, use_bias_correction_kl=False),
            use_vllm=False, current_gradient_accumulation_steps=384,
            accelerator=SimpleNamespace(num_processes=3, gather=lambda x: x),
            model=SimpleNamespace(training=True),
            _metrics={'train': defaultdict(list)},
        )
        inputs = {
            'prompt_ids': torch.ones(1, 1, dtype=torch.long),
            'prompt_mask': torch.ones(1, 1),
            'completion_ids': torch.ones(1, 2, dtype=torch.long),
            'completion_mask': torch.ones(1, 2),
            'advantages': torch.ones(1),
            'old_per_token_logps': torch.zeros(1, 2),
            'num_items_in_batch': torch.tensor(3 * generation_steps * 2),
        }
        micro_loss = scope['_compute_loss'](fake, fake.model, inputs)
        # Identical synthetic microbatches isolate the normalization coefficient.
        (micro_loss * 384).backward()
        outputs[generation_steps] = parameter.grad.item()
    ratio = outputs[16] / outputs[384]
    assert abs(ratio - 24.0) < 1e-6, outputs
    return {'accumulated_gradients': outputs, 'ratio': ratio,
            'scope': 'Synthetic loss scaling; not a prediction of Adam update magnitude.'}


def reproduce_prompt_roundtrip():
    from transformers import AutoTokenizer

    source = REPO / 'atpgllm/training/revert_template.py'
    spec = importlib.util.spec_from_file_location('revert_template_probe', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tokenizer = AutoTokenizer.from_pretrained(
        str(REPO / 'runs/sft_granite_4.2_8b/checkpoint-200'), local_files_only=True)
    prompt = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': 'Generate a test for sa0 y[16].'}],
        tokenize=False, add_generation_prompt=True)
    parsed = module.revert_chat_template(prompt, tokenizer=tokenizer)
    rendered = tokenizer.apply_chat_template(parsed, tokenize=False, add_generation_prompt=True)
    assert prompt.count('<|im_start|>assistant') == 1
    assert rendered.count('<|im_start|>assistant') == 2
    return {'original_suffix': prompt[-140:], 'rerendered_suffix': rendered[-180:]}


if __name__ == '__main__':
    torch.set_num_threads(1)
    print(json.dumps({'loss_scaling': reproduce_loss_scaling(),
                      'prompt_roundtrip': reproduce_prompt_roundtrip()}, indent=2))
