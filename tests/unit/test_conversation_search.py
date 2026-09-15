"""Behavior tests for tool-aware inference search, without model downloads."""
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from atpgllm.training.completion_runner import CompletionRunner
from atpgllm.training.search_types import (
    ConversationState, GenerationResult, SearchConfig, SearchContext, CompletionScore, accepted,
)
from atpgllm.training.search_verifier import assignment, final_fields
from atpgllm.training.search_policies import MCTSStrategy, EvolutionaryStrategy, BestOfNStrategy


def answer(a=1, b=1, y=1):
    return f'INPUT_VECTOR: "a: {a}, b: {b}"\nEXPECTED_OUTPUT: "y: {y}"\nDETECTED_FAULTS: "sa0 y"'


def call(a=1, b=1):
    return '<tool_call>' + json.dumps({"name": "fault_simulation_tool", "arguments": {
        "input_vector": {"a": a, "b": b}, "output_vector": {"y": 1}, "fault": "sa0 y", "doc_id": "doc",
    }}) + '</tool_call>'


class Tokenizer:
    chat_template = "<|im_start|>"
    model_max_length = 20000

    def encode(self, text, **kwargs):
        return list(text.encode())

    def apply_chat_template(self, messages, **kwargs):
        return ''.join('<|im_start|>' + m['role'] + '\n' + m.get('content', '')
                       + (json.dumps(m['tool_calls']) if m.get('tool_calls') else '')
                       + '<|im_end|>\n' for m in messages) + '<|im_start|>assistant\n'


class Generator:
    tokenizer = Tokenizer()
    temperature, top_p, max_tokens, context_limit = 0.7, 0.95, 2048, 20000

    def __init__(self, response):
        self.response, self.requests = response, []

    def generate_requests(self, requests):
        output = []
        for request in requests:
            self.requests.append(request)
            result = self.response(request, len(self.requests))
            if isinstance(result, GenerationResult):
                output.append(result)
            else:
                text = result[:request.max_tokens]
                output.append(GenerationResult(text, tuple(text.encode()),
                    'length' if len(result) > request.max_tokens else 'stop',
                    prompt_tokens=len(request.prompt.encode())))
        return output


class Verifier:
    @staticmethod
    def failure(status, *args):
        return CompletionScore(components={'search_failure_logonly': 1}, status=status)

    def __init__(self):
        self.tool_vectors, self.scored = [], []

    def problem(self, prompt, record):
        return SimpleNamespace(prompt=prompt, problem_id=record.get('id', 'problem'),
                               input_nets=('a', 'b'), output_nets=('y',))

    def tool(self, call, problem, context):
        vector = call['arguments']['input_vector']
        self.tool_vectors.append(dict(vector))
        return {'vector': vector, 'result': 'OBS:' + json.dumps(vector), 'identity': 'sim'}

    def score_state(self, problem, state, context):
        self.scored.append(state)
        if state.status != 'FINAL':
            return self.failure(state.status), None
        fields = final_fields(state.final_answer)
        vector = assignment(fields['INPUT_VECTOR'], problem.input_nets)
        detected = bool(vector['a'] and vector['b'])
        correct_output = fields.get('EXPECTED_OUTPUT') == 'y: 1'
        components = {'detection': float(detected), 'activation': float(vector['a']),
                      'simulation_valid_logonly': 1,
                      'fault_detected_by_pred_input_vector_acc_logonly': float(detected),
                      'expected_output_acc_logonly': float(correct_output),
                      'input_vector_acc_logonly': 1, 'detected_faults_acc_logonly': 1}
        return CompletionScore(detected, float(detected), components), vector


PROMPT = '<|im_start|>user\nFind sa0 y<|im_end|>\n<|im_start|>assistant\n'


def job(runner, config=None, state=None):
    ctx = SearchContext(17, config or SearchConfig(), 8)
    return (state or runner.initial(PROMPT), runner.verifier.problem(PROMPT, {}), ctx, 0.7)


def test_three_exchanges_preserve_all_history_and_call_identity():
    gen = Generator(lambda req, i: call(i % 2, 1) if i <= 3 else answer())
    runner = CompletionRunner(gen, Verifier(), max_tool_rounds=3)
    j = job(runner)
    (state, checkpoints), = runner.complete_many([j])
    assert state.status == 'FINAL'
    assert len(checkpoints) == 3
    assert gen.requests[-1].prompt.count('OBS:') == 3
    assert len([m for m in state.messages if m['role'] == 'tool']) == 3
    assert len({o['call_id'] for o in state.observations}) == 3
    assert j[2].usage.generation_requests == 4
    assert j[2].usage.generated_tokens == sum(len(x) for x in [call(1, 1), call(0, 1), call(1, 1), answer()])


def test_optional_trace_retains_discarded_candidate_conversations():
    gen = Generator(lambda req, i: answer(a=0) if i == 1 else answer())
    policy = MCTSStrategy(gen, Verifier(), budget=3, search_config=SearchConfig(save_trace=True))
    result, = policy.sample_batch([PROMPT], [{}])
    events = [event for event in result.slots[0]['trace'] if event['kind'] == 'candidate']
    assert len(events) == 2
    assert events[0]['final_answer'] == answer(a=0)
    assert events[1]['final_answer'] == result.slots[0]['final_answer']
    assert events[0]['state_id'] != events[1]['state_id']
    assert events[0]['parent_state_id'] == events[1]['parent_state_id']
    assert events[0]['messages'][-1]['content'] == answer(a=0)


def test_pending_call_runs_before_generation_and_discards_speculative_tail():
    gen = Generator(lambda req, i: answer())
    verifier = Verifier()
    runner = CompletionRunner(gen, verifier)
    initial = runner.initial(PROMPT)
    pending = replace(initial, open_text=call() + 'unobserved guess', readable=call() + 'unobserved guess')
    j = job(runner, state=pending)
    state, = runner.advance_many([j])
    assert not gen.requests
    assert verifier.tool_vectors == [{'a': 1, 'b': 1}]
    assert 'unobserved guess' not in state.readable
    assert j[2].usage.discarded_tokens == len('unobserved guess')


@pytest.mark.parametrize('rounds', [0, 1])
def test_prefix_is_present_in_actual_backend_prompt(rounds):
    gen = Generator(lambda req, i: answer())
    runner = CompletionRunner(gen, Verifier(), max_tool_rounds=rounds)
    state = replace(runner.initial(PROMPT), open_text='PREFIX ', readable='PREFIX ')
    runner.complete_many([job(runner, state=state)])
    assert gen.requests[0].prompt.endswith('PREFIX ')
    assert gen.requests[0].prompt.count('PREFIX ') == 1


def test_tool_round_limit_is_inherited_by_branch():
    gen = Generator(lambda req, i: call())
    runner = CompletionRunner(gen, Verifier(), max_tool_rounds=1)
    j = job(runner)
    first, = runner.advance_many([j])
    (last, _), = runner.complete_many([(first, *j[1:])])
    assert last.status == 'EXHAUSTED'
    assert last.tool_rounds == 1
    assert len(runner.verifier.tool_vectors) == 1


def test_multiple_calls_fail_explicitly_without_execution():
    gen = Generator(lambda req, i: call() + call())
    runner = CompletionRunner(gen, Verifier())
    state, = runner.advance_many([job(runner, SearchConfig(repair_limit=0))])
    assert state.status == 'INVALID'
    assert not runner.verifier.tool_vectors


def test_incomplete_xml_can_continue_after_token_limit():
    prefix = '<tool_call><function=fault_simulation_tool><parameter=input_vector>{"a":1,"b":1}</parameter>'
    tail = '<parameter=output_vector>{"y":1}</parameter><parameter=fault>sa0 y</parameter><parameter=doc_id>doc</parameter></function></tool_call>'
    def response(req, i):
        if i == 1:
            return GenerationResult(prefix, tuple(prefix.encode()), 'length')
        return tail if i == 2 else answer()
    runner = CompletionRunner(Generator(response), Verifier())
    (state, _), = runner.complete_many([job(runner)])
    assert state.status == 'FINAL'
    assert runner.verifier.tool_vectors == [{'a': 1, 'b': 1}]


def test_nonempty_eos_final_is_not_extended():
    gen = Generator(lambda req, i: answer())
    runner = CompletionRunner(gen, Verifier())
    (state, _), = runner.complete_many([job(runner)])
    runner.complete_many([job(runner, state=state)])
    assert len(gen.requests) == 1


def test_token_budget_stops_partial_generation_without_silent_context_truncation():
    gen = Generator(lambda req, i: 'x' * 100)
    runner = CompletionRunner(gen, Verifier())
    j = job(runner, SearchConfig(max_generated_tokens=30, finalization_tokens=5, action_tokens=12))
    (state, _), = runner.complete_many([j])
    assert state.status == 'EXHAUSTED'
    assert j[2].usage.generated_tokens == 30


def test_output_cardinality_mismatch_raises():
    gen = Generator(lambda req, i: answer())
    gen.generate_requests = lambda requests: []
    runner = CompletionRunner(gen, Verifier())
    with pytest.raises(RuntimeError, match='number of results'):
        runner.complete_many([job(runner)])


@pytest.mark.parametrize('budget', range(1, 9))
def test_evolution_generates_exact_admitted_count_at_small_budgets(budget):
    gen = Generator(lambda req, i: answer(0, 0, 0))
    strategy = EvolutionaryStrategy(gen, Verifier(), budget=budget, num_completions=1,
                                    use_tools=False)
    result, = strategy.sample_batch([PROMPT], [{}])
    assert len(result.completions) == 1
    assert len(gen.requests) == budget
    assert result.slots[0]['usage']['attempts'] == budget


def test_full_accuracy_repairs_answer_after_detection():
    gen = Generator(lambda req, i: answer(y=0) if i == 1 else answer())
    strategy = EvolutionaryStrategy(gen, Verifier(), budget=2, num_completions=1,
                                    threshold_mode='full_accuracy', use_tools=False,
                                    search_config={'population_size': 1})
    result, = strategy.sample_batch([PROMPT], [{}])
    assert len(gen.requests) == 2
    assert 'Keep INPUT_VECTOR unchanged' in gen.requests[1].prompt
    assert accepted(result.scores[0].components, 'full_accuracy')


def test_evolution_vector_edits_have_no_stale_observation():
    def response(req, i):
        if 'OBS:' in req.prompt:
            return answer(1, 0, 0)
        return call(1, 0)
    gen = Generator(response)
    strategy = EvolutionaryStrategy(gen, Verifier(), budget=2, num_completions=1,
                                    search_config={'population_size': 1, 'operator_weights': [0, 1, 0, 0]})
    strategy.sample_batch([PROMPT], [{}])
    proposals = [r for r in gen.requests if 'Consider this candidate' in r.prompt and 'OBS:' not in r.prompt]
    assert proposals
    assert all('tool_call_id' not in r.prompt for r in proposals)


def test_mcts_can_branch_after_actual_observation():
    def response(req, i):
        if 'OBS:' not in req.prompt:
            return call(1, 0)
        return answer(1, 0, 0)
    gen = Generator(response)
    strategy = MCTSStrategy(gen, Verifier(), budget=6, num_completions=1)
    result, = strategy.sample_batch([PROMPT], [{}])
    assert sum('OBS:' in r.prompt for r in gen.requests) >= 2
    assert result.slots[0]['usage']['attempts'] <= 6
    assert not result.detected_any


def test_closed_duplicate_mcts_leaves_do_not_repeat_verification_or_loop_forever():
    verifier = Verifier()
    gen = Generator(lambda req, i: answer(0, 0, 0))
    strategy = MCTSStrategy(gen, verifier, budget=20, num_completions=1,
                            search_config={'duplicate_limit': 2})
    result, = strategy.sample_batch([PROMPT], [{}])
    assert len(verifier.scored) == 1
    assert result.slots[0]['stop_reason'] == 'frontier_closed'
    assert len(gen.requests) == 3


def test_independent_slots_keep_seed_streams_when_problem_order_changes():
    def run(records):
        gen = Generator(lambda req, i: answer(0, 0, 0))
        strategy = BestOfNStrategy(gen, Verifier(), width=2, num_completions=2)
        results = strategy.sample_batch([PROMPT] * len(records), records)
        return {rec['id']: [s['seed'] for s in result.slots] for rec, result in zip(records, results)}
    assert run([{'id': 'a'}, {'id': 'b'}]) == run([{'id': 'b'}, {'id': 'a'}])


def test_answer_extraction_ignores_history_and_rejects_ambiguity():
    text = answer(0, 0, 0) + '<tool_response>old</tool_response>' + answer()
    assert final_fields(text)['INPUT_VECTOR'] == 'a: 1, b: 1'
    with pytest.raises(ValueError, match='Ambiguous'):
        final_fields(answer() + answer())


@pytest.mark.parametrize('value', ['a:0,a:1,b:0', '{"a":0,"a":1,"b":0}', 'a:0', 'a:2,b:0'])
def test_assignment_rejects_duplicates_missing_and_nonbinary_inputs(value):
    with pytest.raises(ValueError):
        assignment(value, ('a', 'b'))


@pytest.mark.parametrize('config', [{'wrong_key': 1}, {'max_children': 0}, {'c_puct': float('nan')},
                                   {'operator_weights': [1, 2]}, {'controller_edits': 'yes'}])
def test_search_config_rejects_invalid_settings(config):
    with pytest.raises(ValueError):
        SearchConfig.load(config)
