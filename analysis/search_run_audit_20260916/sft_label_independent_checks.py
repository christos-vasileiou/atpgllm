"""Confirm cached-label errors directly from simple wiring and source bit order."""
import json
from pathlib import Path
import re

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]


def main():
    records = json.loads((OUT / 'sft_training_label_sample.json').read_text())
    examples = []
    for row in records:
        module, raw = row['module_name'], row['netlist']
        vector, expected = json.loads(row['input_vector']), json.loads(row['expected_output'])
        if module == 'binary_to_ascii_bin':
            assert 'assign ascii[0] = binary;' in raw
            for i in [4, 5]:
                assert f'assign ascii[{i}] = \\*Logic1* ;' in raw
            for i in [1, 2, 3, 6, 7]:
                assert f'assign ascii[{i}] = \\*Logic0* ;' in raw
            assert 'TIEHI' in raw and 'TIELO' in raw
            good = {f'ascii[{i}]': int(i in [4, 5]) for i in range(8)}
            good['ascii[0]'] = vector['binary']
        elif module in ['sl2', 'slt2']:
            assert 'TIELO' in raw
            for i in [0, 1]:
                assert f'assign y[{i}] = \\*Logic0* ;' in raw
            for i in range(2, 32):
                assert f'assign y[{i}] = a[{i-2}];' in raw
            good = {f'y[{i}]': 0 if i < 2 else vector[f'a[{i-2}]'] for i in range(32)}
        else:
            continue
        differences = {k: {'label': expected.get(k), 'wiring': v} for k, v in good.items() if expected.get(k) != v}
        assert differences, 'Selected cached label unexpectedly agrees with wiring'
        examples.append(dict(module=module, fault=row['fault'], wrong_bits=differences,
            expected=expected, good_from_wiring=good))
    assert len(examples) >= 3
    # Separate, retained TetraMAX artifact: demonstrates the source-order
    # hazard, but is not claimed as the exact provenance of these cached rows.
    path = ROOT.parent / 'data/out.freeset.asap7sc7p5t_28.rvt.tt/1180_sl2/simulation.stil'
    stil = path.read_text()
    groups = {}
    for group in ['_pi', '_po']:
        match = re.search(r'"'+group+r'"\s*=\s*\x27([^\x27]*)\x27', stil, re.S)
        groups[group] = re.findall(r'"([^\"]+)"', match[1])
    assert groups['_pi'] == [f'a[{i}]' for i in range(31, -1, -1)]
    assert groups['_po'] == [f'y[{i}]' for i in range(31, -1, -1)]
    result = dict(confirmed_wrong_label_examples=examples,
        source_order_example=dict(path=str(path), groups=groups),
        caveat='Hand-derived truth on three simple cached circuits. The retained STIL file is a separate source-order example, not an exact provenance reconstruction for the cached examples. No labels or model outputs were modified.')
    (OUT / 'sft_label_independent_checks.json').write_text(json.dumps(result, indent=2)+'\n')
    print(f'Confirmed {len(examples)} incorrect cached labels directly from wiring; source STIL uses descending bus order')


if __name__ == '__main__':
    main()
