#!/usr/bin/env python3
"""Analyze local examples and optionally saved strict Language-of-Test slots.

No model load or inference. Accepts a JSON examples/manifest file or JSONL rows.
Saved slot joins require exact circuit/fault identity; positional joins forbidden.
"""
import argparse
import ast
import re
from collections import defaultdict
import json
from pathlib import Path

from fault_difficulty import Analyzer, Circuit, DEFAULT_CONFIG, digest, identity, raw_netlist, write_report
from lot_metrics import parse_answer, outcomes


def saved_identity(result):
    keys = {slot['problem_id'] for slot in result.get('search_slots',[]) if 'problem_id' in slot}
    if result.get('problem_id'):
        return result['problem_id']
    if len(keys) != 1:
        raise ValueError('No unique saved problem identity')
    return keys.pop()


def extract_record(result):
    netlists = set()
    for slot in result.get('search_slots',[]):
        for message in slot.get('messages',[]):
            if message['role'] == 'user':
                match = re.search(r"['\"]netlist['\"]\s*:\s*('(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\")",message['content'])
                if match:
                    netlists.add(ast.literal_eval(match[1]))
    if len(netlists) != 1:
        raise ValueError('Expected exactly one circuit identity in saved prompts; provide --problem-source')
    return dict(netlist=netlists.pop(),fault=result['fault'],module_name=result.get('module_name',''))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--evaluation-results', action='store_true',
                        help='Input is evaluate_model.py JSON; extract saved prompts and replay final answers on CPU')
    parser.add_argument('--problem-source', type=Path,
                        help='Checksummed manifest or saved evaluator JSON with prompts; join by problem ID, never position')
    parser.add_argument('--slots', type=Path, help='Strict replay slots.jsonl containing D and circuit_id/fault')
    parser.add_argument('--output', type=Path, required=True, help='Output prefix (JSON, CSV and JSONL)')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--random-samples', type=int, default=4096)
    parser.add_argument('--exact-max-inputs', type=int, default=12)
    parser.add_argument('--seed', type=int, default=1729)
    parser.add_argument('--k', type=int, nargs='+', default=[1])
    args = parser.parse_args()
    if min(args.k) < 1:
        parser.error('k must be positive')
    saved_results = None
    expected_slots = None
    if args.evaluation_results:
        if args.slots:
            parser.error('--slots and --evaluation-results are mutually exclusive')
        payload = json.loads(args.input.read_text())
        saved_results = payload['per_problem_results']
        source_records = {}
        if args.problem_source:
            source = json.loads(args.problem_source.read_text())
            if 'examples' in source:
                if digest(source['examples']) != source.get('examples_sha256'):
                    raise ValueError('Problem-source checksum mismatch')
                source_records = {identity(r):r for r in source['examples']}
            else:
                source_records = {saved_identity(r):extract_record(r) for r in source['per_problem_results']}
        records = []
        for result in saved_results:
            if source_records:
                record = source_records[saved_identity(result)]
                if record['fault'] != result['fault']:
                    raise ValueError('Problem source fault mismatch')
            else:
                record = extract_record(result)
            records.append(record)
    elif args.input.suffix == '.jsonl':
        records = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    else:
        payload = json.loads(args.input.read_text())
        records = payload if isinstance(payload,list) else payload['examples']
        if isinstance(payload,dict):
            expected_slots = payload.get('protocol',{}).get('generations')
        if isinstance(payload,dict) and 'examples_sha256' in payload and digest(records) != payload['examples_sha256']:
            raise ValueError('Source manifest checksum mismatch')
    by_id = {identity(r):r for r in records}
    if len(by_id) != len(records):
        raise ValueError('Duplicate circuit/fault pairs in input')
    slots = defaultdict(list)
    if args.slots:
        lookup = {(digest(raw_netlist(r)),r['fault']):identity(r) for r in records}
        seen = set()
        for line in args.slots.read_text().splitlines():
            row = json.loads(line)
            key = (row['circuit_id'],row['fault'])
            if key not in lookup:
                raise ValueError('Slot circuit/fault not in source examples')
            slot_key = (lookup[key],row['slot'])
            if slot_key in seen:
                raise ValueError('Duplicate slot')
            seen.add(slot_key)
            if type(row['D']) is not bool:
                raise ValueError('Strict Boolean metrics required')
            slots[lookup[key]].append(row)
        if set(slots) != set(by_id):
            raise ValueError('Slots must cover every source problem; replay missing slots first')
        for group in slots.values():
            n = expected_slots if expected_slots is not None else len(group)
            if {s['slot'] for s in group} != set(range(n)):
                raise ValueError('Missing/out-of-range slots; run strict replay to fill failures first')
    analyzer = Analyzer(args.config,args.random_samples,args.exact_max_inputs,args.seed)
    rows = []
    for i, record in enumerate(records):
        row = analyzer.analyze(record)
        group = slots.get(identity(record))
        if saved_results is not None:
            saved = saved_results[i]
            n = saved['num_completions']
            counts = dict(D=0,failed_slots=0)
            model = Circuit(raw_netlist(record),analyzer.gates) if row['status'] == 'ok' else None
            completions = saved.get('completions',[])
            if len(completions) > n:
                raise ValueError('Too many completion slots')
            for slot in range(n):
                text = completions[slot] if slot < len(completions) else ''
                parsed = parse_answer(text,model.meta.input_nets,model.meta.output_nets) if model else dict(input=None,expected=None)
                terminal = saved.get('search_slots',[])
                if slot < len(terminal) and terminal[slot].get('status') not in (None,'FINAL'):
                    parsed = dict(input=None,expected=None)
                good = bad = None
                if model and parsed['input'] is not None:
                    values = {model.alias[k]:v for k,v in parsed['input'].items()}
                    site = model.alias[record['fault'].split(maxsplit=1)[1]]
                    good_values = model.simulate(values,1)
                    bad_values = model.simulate(values,1,(site,int(record['fault'][2])))
                    good = {p:good_values[model.alias[p]] for p in model.meta.output_nets}
                    bad = {p:bad_values[model.alias[p]] for p in model.meta.output_nets}
                measured = outcomes(parsed,good,bad)
                for metric in ('D',):
                    counts[metric] += int(measured[metric])
                counts['failed_slots'] += int(not measured['verification_known'])
            row['num_completions'],row['outcomes'] = n,counts
            row['saved_search_usage'] = {k:sum(s.get('usage',{}).get(k,0) for s in saved.get('search_slots',[]))
                for k in ('attempts','simulator_requests','simulator_executions','generated_tokens')}
        if group:
            row['num_completions'] = len(group)
            row['outcomes'] = {m:sum(s[m] for s in group) for m in ('D',)}
            row['outcomes']['failed_slots'] = sum(not s.get('verification_known',False) for s in group)
        rows.append(row)
        if (i+1) % 100 == 0:
            print(f'Analyzed {i+1}/{len(records)}',flush=True)
    provenance = {**analyzer.provenance(), 'source':str(args.input),
                  'source_sha256':digest(records), 'slots':str(args.slots) if args.slots else None,
                  'slots_sha256':__import__('hashlib').sha256(args.slots.read_bytes()).hexdigest() if args.slots else None,
                  'analysis_code_sha256':__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest(),
                  'saved_evaluation_config':payload.get('config') if args.evaluation_results else None,
                  'scoring':'strict CPU replay of saved FINAL answers; other terminal statuses are failed slots' if saved_results is not None else 'strict saved replay outcomes' if args.slots else 'no model outcomes',
                  'source_file_sha256':__import__('hashlib').sha256(args.input.read_bytes()).hexdigest(),
                  'problem_source':str(args.problem_source) if args.problem_source else None,
                  'problem_source_sha256':__import__('hashlib').sha256(args.problem_source.read_bytes()).hexdigest() if args.problem_source else None}
    if args.slots and (args.slots.parent/'provenance.json').exists():
        source_provenance = args.slots.parent/'provenance.json'
        provenance['source_replay_provenance'] = json.loads(source_provenance.read_text())
    report = write_report(args.output,rows,provenance,args.k)
    print(json.dumps(report['overall'],indent=2))


if __name__ == '__main__':
    main()
