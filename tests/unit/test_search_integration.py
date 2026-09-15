"""Real reward/simulator contracts, tokenizer histories and backend adapters."""
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from atpgllm.training.search_types import (
    SearchConfig, SearchContext, ConversationState, GenerationRequest, BudgetExceeded, accepted,
)
from atpgllm.training.search_verifier import Verifier, ProblemContext


@pytest.fixture
def verifier(monkeypatch):
    monkeypatch.setenv('FAULT_SIM_BACKEND', 'fast')
    from atpgllm.training.reward_function_factory import RewardFunctionFactory
    verifier = Verifier(RewardFunctionFactory())
    calls = []

    def simulator(vector, outputs, fault, netlist, gates, **kwargs):
        calls.append((dict(vector), dict(outputs), fault))
        good = vector['a'] & vector['b']
        frame = pd.DataFrame({'Good Machine': [vector['a'], vector['b'], good],
                              'Bad Machine': [vector['a'], vector['b'], 0],
                              'PIs': [True, True, False], 'POs': [False, False, True]},
                             index=['a', 'b', 'y'])
        return frame, {}

    verifier.fault_sim = simulator
    verifier.recorded_calls = calls
    return verifier


@pytest.fixture
def problem():
    return ProblemContext('problem', {}, 'problem',
        SimpleNamespace(input_nets=['a', 'b'], output_nets=['y']),
        'sa0 y', 'doc', 'module', ('a', 'b'), ('y',), 'identity')


def tool(vector, **changes):
    return {'name': 'fault_simulation_tool', 'arguments': {
        'input_vector': vector, 'output_vector': {'y': 0}, 'fault': 'sa0 y', 'doc_id': 'doc', **changes}}


def state(vector='a: 1, b: 1', output='y: 1', observations=()):
    text = f'INPUT_VECTOR: "{vector}"\nEXPECTED_OUTPUT: "{output}"\nDETECTED_FAULTS: "sa0 y"'
    return ConversationState(status='FINAL', final_answer=text, readable=text, observations=observations)


def test_tool_and_final_verifier_share_simulation_but_rescore_output(verifier, problem):
    ctx = SearchContext(1, SearchConfig(), 4)
    obs = verifier.tool(tool({'a': 1, 'b': 1}), problem, ctx)
    wrong, _ = verifier.score_state(problem, state(output='y: 0', observations=(obs,)), ctx)
    correct, _ = verifier.score_state(problem, state(observations=(obs,)), ctx)
    assert wrong.detected and not accepted(wrong.components, 'full_accuracy')
    assert accepted(correct.components, 'full_accuracy')
    assert len(verifier.recorded_calls) == 1
    assert ctx.usage.simulator_requests == 3
    assert ctx.usage.simulator_executions == 1
    assert ctx.usage.cache_hits == 2
    assert verifier.recorded_calls[0][1] == {'y': 0}


def test_changed_final_vector_runs_new_simulation_and_ignores_old_observation(verifier, problem):
    ctx = SearchContext(1, SearchConfig(), 4)
    old = verifier.tool(tool({'a': 1, 'b': 0}), problem, ctx)
    final = state(observations=(old,))
    score, vector = verifier.score_state(problem, final, ctx)
    assert score.detected and vector == {'a': 1, 'b': 1}
    assert score.components['tool_response_fidelity_logonly'] == 0
    assert len(verifier.recorded_calls) == 2


def test_missing_or_extra_outputs_do_not_pass_full_accuracy(verifier, problem):
    score, _ = verifier.score_state(problem, state(output='y: 1, invented: 1'),
                                    SearchContext(1, SearchConfig(), 1))
    assert score.detected
    assert not accepted(score.components, 'full_accuracy')


@pytest.mark.parametrize('change', [{'fault': 'sa1 y'}, {'doc_id': 'other'},
                                  {'input_vector': {'a': 1}}, {'output_vector': {'z': 1}},
                                  {'netlist': 'injected'}])
def test_wrong_tool_binding_never_executes(verifier, problem, change):
    with pytest.raises(ValueError):
        verifier.tool(tool({'a': 1, 'b': 1}, **change), problem, SearchContext(1, SearchConfig(), 1))
    assert not verifier.recorded_calls


def test_cache_identity_and_budget_reservations(verifier, problem):
    ctx = SearchContext(1, SearchConfig(max_simulator_executions=2, max_simulator_requests=3), 5)
    verifier.simulate(problem, {'a': 1, 'b': 1}, ctx)
    verifier.simulate(problem, {'b': 1, 'a': 1}, ctx)  # Canonical order hits cache.
    verifier.simulate(replace(problem, simulation_identity='different'), {'a': 1, 'b': 1}, ctx)
    with pytest.raises(BudgetExceeded):
        verifier.simulate(problem, {'a': 1, 'b': 1}, ctx)
    assert len(verifier.recorded_calls) == 2
    assert ctx.usage.simulator_requests == 3


def test_transient_simulator_failure_is_not_cached(verifier, problem):
    real = verifier.fault_sim
    verifier.fault_sim = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('temporary'))
    ctx = SearchContext(1, SearchConfig(), 2)
    failed, _ = verifier.score_state(problem, state(), ctx)
    assert failed.status == 'INFRA_ERROR'
    assert not ctx.cache
    verifier.fault_sim = real
    successful, _ = verifier.score_state(problem, state(), ctx)
    assert successful.detected
    assert ctx.usage.simulator_executions == 2


def test_vector_only_baseline_uses_simulator_outputs_without_generator(verifier, problem):
    from atpgllm.training.sampling_strategies import make_strategy
    verifier.problem = lambda prompt, record: problem
    strategy = make_strategy('vector_evolutionary', None, verifier, width=8,
                              num_completions=2, threshold_mode='full_accuracy')
    result, = strategy.sample_batch(['problem'], [{}])
    assert len(result.completions) == 2
    assert result.generator_calls == 0
    assert all(slot['answer_source'] == 'simulator' for slot in result.slots)
    assert all(slot['usage']['generated_tokens'] == 0 for slot in result.slots)
    assert all(slot['usage']['attempts'] <= 8 for slot in result.slots)


def test_vector_baseline_reports_infrastructure_failure_separately(verifier, problem):
    from atpgllm.training.sampling_strategies import make_strategy
    verifier.problem = lambda prompt, record: problem
    verifier.fault_sim = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('simulator unavailable'))
    strategy = make_strategy('vector_evolutionary', None, verifier, width=8,
                             num_completions=1, search_config=SearchConfig(infrastructure_retry_limit=0))
    result, = strategy.sample_batch(['problem'], [{}])
    slot, = result.slots
    assert slot['status'] == 'INFRA_ERROR'
    assert slot['stop_reason'] == 'infrastructure_retry_limit'
    assert slot['usage']['simulator_executions'] == 1


def test_fast_backend_on_real_netlist(verifier):
    from fault_sim import fast_fault_sim
    verifier.fault_sim = fast_fault_sim
    record = {'netlist': {'doc_id': 'test', 'netlist':
              'module design(a, b, y);\ninput a, b;\noutput y;\nassign y = a;\nendmodule'},
              'fault': 'sa0 y', 'module_name': 'design'}
    problem = verifier.problem('problem', record)
    score, _ = verifier.score_state(problem, state(), SearchContext(1, SearchConfig(), 1))
    assert score.detected


def test_vllm_adapter_preserves_individual_settings_and_metadata(monkeypatch):
    from atpgllm.training.search_backends import VLLMGenerator
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(SamplingParams=lambda **kw: SimpleNamespace(**kw)))
    class LLM:
        llm_engine = SimpleNamespace(model_config=SimpleNamespace(max_model_len=100))
        def generate(self, prompts, sampling_params, **kwargs):
            self.params = sampling_params
            return [SimpleNamespace(prompt_token_ids=[4, 5], outputs=[SimpleNamespace(
                text='done', token_ids=[7, 8], finish_reason='length', cumulative_logprob=-4.0)]) for _ in prompts]
    llm = LLM()
    gen = VLLMGenerator(llm, SimpleNamespace(model_max_length=500, encode=lambda s, **kw: [4, 5]), None,
                       SimpleNamespace(max_tokens=20, temperature=0.7, top_p=0.95, stop_token_ids=[9]))
    requests = [GenerationRequest('a', 3, 0.4, 0.9, 11, True), GenerationRequest('b', 7, 1.1, 1.0, 22)]
    results = gen.generate_requests(requests)
    assert [p.seed for p in llm.params] == [11, 22]
    assert [p.max_tokens for p in llm.params] == [3, 7]
    assert [p.temperature for p in llm.params] == [0.4, 1.1]
    assert all(p.include_stop_str_in_output for p in llm.params)
    assert results[0].mean_logprob == -2
    assert results[0].finish_reason == 'length'


def test_hf_adapter_real_tiny_cpu_model_preserves_rng():
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, GenerationConfig
    from atpgllm.training.search_backends import HFGenerator
    tok = Tokenizer(WordLevel({'<unk>': 0, '<eos>': 1, 'a': 2, 'b': 3}, unk_token='<unk>'))
    tok.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token='<unk>', eos_token='<eos>', pad_token='<eos>', model_max_length=64)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=4, n_positions=64, n_embd=8, n_layer=1, n_head=1)).eval()
    gen = HFGenerator(model, tokenizer, GenerationConfig(max_new_tokens=4, temperature=0.7, top_p=0.95, eos_token_id=1))
    request = GenerationRequest('a b', 4, 0.7, 0.95, 99, True)
    before = torch.random.get_rng_state().clone()
    first, second = gen.generate_requests([request, request])
    assert first.error is None and second.error is None
    assert first.token_ids == second.token_ids
    assert len(first.token_ids) <= 4
    assert first.prompt_tokens == 2
    assert torch.equal(before, torch.random.get_rng_state())
    deterministic = HFGenerator(model, tokenizer, GenerationConfig(
        max_new_tokens=4, temperature=0.0, top_p=0.95, eos_token_id=0))
    assert deterministic.temperature == 0.0
    observed_configs = []
    original_generate = model.generate
    def recording_generate(*args, **kwargs):
        observed_configs.append(kwargs['generation_config'])
        return original_generate(*args, **kwargs)
    model.generate = recording_generate
    zero_result, = deterministic.generate_requests([GenerationRequest('a b', 4, 0.0, 0.95, 99)])
    assert zero_result.error is None
    assert observed_configs[0].eos_token_id == 0
    assert observed_configs[0].do_sample is False


@pytest.mark.parametrize('relative', ['runs/sft_granite_4.2_8b/checkpoint-200', 'runs/grpo_7b_h100/checkpoint-10'])
def test_real_local_tokenizer_renders_complete_tool_history(relative):
    from transformers import AutoTokenizer
    from atpgllm.training.completion_runner import CompletionRunner
    from atpgllm.training.revert_template import restore_generation_prefix
    from atpgllm.training.tools import TOOLS
    path = Path(__file__).resolve().parents[2] / relative
    if not (path / 'tokenizer_config.json').exists():
        pytest.skip('Local tokenizer unavailable')
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    runner = CompletionRunner(SimpleNamespace(tokenizer=tokenizer), None)
    messages = ({'role': 'user', 'content': 'Test sa0 y'},
                {'role': 'assistant', 'content': restore_generation_prefix('Check it.</think>', tokenizer),
                 'tool_calls': [{'type': 'function', 'function': tool({'a': 1, 'b': 1})}]},
                {'role': 'tool', 'name': 'fault_simulation_tool', 'content': 'actual result'})
    state = ConversationState(messages=messages)
    rendered = runner.render(state)
    assert 'actual result' in rendered and 'fault_simulation_tool' in rendered
    assert isinstance(state.messages[1]['tool_calls'][0]['function']['arguments'], dict)
    original = tokenizer.apply_chat_template([messages[0]], tokenize=False, tools=TOOLS, add_generation_prompt=True)
    assert runner.render(runner.initial(original, [messages[0]])) == original


@pytest.mark.parametrize('method', ['greedy', 'mcts', 'evolutionary', 'vector_evolutionary'])
def test_evaluator_writes_versioned_metrics_and_all_slots(tmp_path, monkeypatch, method):
    import importlib.util
    from test_conversation_search import Generator, Tokenizer, PROMPT, answer
    path = Path(__file__).resolve().parents[2] / 'scripts/eval/evaluate_model.py'
    spec = importlib.util.spec_from_file_location('eval_search_integration', path)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    with pytest.raises(ValueError, match='k >= 1'):
        evaluator.evaluate(tmp_path / 'missing-checkpoint', num_completions=2, k_values=[0])
    monkeypatch.setenv('FAULT_SIM_BACKEND', 'fast')
    tokenizer = Tokenizer()
    tokenizer.pad_token, tokenizer.eos_token = '<pad>', '<eos>'
    tokenizer.pad_token_id, tokenizer.eos_token_id = 0, 1
    monkeypatch.setattr(evaluator.AutoTokenizer, 'from_pretrained', lambda *a, **k: tokenizer)
    monkeypatch.setattr(evaluator, 'load_eval_adapter', lambda *a, **k: SimpleNamespace(eval=lambda: None))
    monkeypatch.setattr(evaluator, 'make_hf_generator', lambda *a, **k: Generator(lambda r, i: answer()))
    record = {'netlist': {'doc_id': 'doc', 'netlist':
              'module design(a,b,y);\ninput a,b;\noutput y;\nassign y = a;\nendmodule'},
              'fault': 'sa0 y', 'module_name': 'design'}
    rows = [record, {**record, 'broken': True}]
    monkeypatch.setattr(evaluator, 'load_dataset', lambda *a, **k: rows)
    monkeypatch.setattr(evaluator, 'buffer_streaming_dataset', lambda *a, **k: rows)
    def prompt(record, tok, **kwargs):
        if record.get('broken'):
            raise ValueError('bad input')
        return PROMPT, [{'role': 'user', 'content': 'Find sa0 y'}]
    monkeypatch.setattr(evaluator, 'format_eval_prompt', prompt)
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'adapter_config.json').write_text(json.dumps({'base_model_name_or_path': 'local-test'}))
    output = tmp_path / 'metrics.json'
    evaluator.evaluate(checkpoint, num_completions=2, k_values=[1, 2], sampling_method=method,
                       budget=None if method == 'greedy' else 8, max_tool_rounds=0,
                       output_file=str(output))
    metrics = json.loads(output.read_text())
    slots = [json.loads(line) for line in output.with_suffix('.slots.jsonl').read_text().splitlines()]
    assert metrics['config']['search_protocol_version'] == 'conversation-search-v1'
    assert metrics['aggregate_metrics']['failed_completion_slots'] == 2
    assert len(slots) == 4
    assert len(metrics['per_problem_results']) == 2
    assert all(s['status'] == 'INFRA_ERROR' for s in slots[2:])
    assert all(s['usage']['attempts'] == 0 for s in slots[2:])
    # Missing/failed slots must not disappear from component denominators.
    assert metrics['accuracy_metrics']['fault_detected_by_pred_input_vector_acc'] <= 0.5
