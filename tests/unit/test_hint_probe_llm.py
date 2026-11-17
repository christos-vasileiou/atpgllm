"""The reply parser and the circuit text of the language-model probe."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def probe():
    sys.path.insert(0, str(ROOT / "scripts" / "eval"))
    spec = importlib.util.spec_from_file_location("hint_probe_llm", ROOT / "scripts" / "eval" / "hint_probe_llm.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_last_call_or_answer_line_counts(probe):
    inputs = ["a[0]", "a[1]", "sel"]
    text = 'I could try CALL: {"a[0]": 0}.\nBetter:\nCALL: {"a[0]": 1, "sel": "0", "zz": 1, "a[1]": 2}'
    assert probe.parse_reply(text, inputs) == ("CALL", {"a[0]": 1, "a[1]": "x", "sel": 0})
    assert probe.parse_reply("ANSWER: {'sel': 1}", inputs) == ("ANSWER", {"a[0]": "x", "a[1]": "x", "sel": 1})
    assert probe.parse_reply("no line here", inputs) == (None, None)
    assert probe.parse_reply("CALL: {broken", inputs) == (None, None)


def test_hidden_reasoning_is_dropped_and_the_prompt_formats(probe):
    message = SimpleNamespace(content='<|channel|>analysis<|message|>CALL: {"a": 0}<|end|>'
                                      '<|channel|>final<|message|>ANSWER: {"a": 1}')
    assert probe.final_text(message) == 'ANSWER: {"a": 1}'
    assert probe.final_text(SimpleNamespace(content=None)) == ""
    assert "at most 8 calls" in probe.SYSTEM.format(calls=8) and 'CALL: {"input name"' in probe.SYSTEM.format(calls=8)


def test_circuit_text_has_the_hint_in_both_views(probe):
    from atpgllm.multimodal.data import netlist_key

    netlist = "module m (a, b, y);\n  input a, b;\n  output y;\n  AND2x2_ASAP7_75t_R U1 (.A(a), .B(b), .Y(y));\nendmodule\n"
    record = {"netlist": netlist, "fault": "sa0 y"}
    hints = {(netlist_key(record), "sa0 y"): "a=1 b=1?"}
    ports = probe.circuit_text(record, hints, "ports")
    assert "// suggested inputs: a=1 b=1?" in ports and "AND2x2" not in ports and "GRAPH" not in ports.upper()
    full = probe.circuit_text(record, hints, "netlist")
    assert full.startswith("// suggested inputs: a=1 b=1?") and "AND2x2" in full
    assert probe.circuit_text(record, {}, "ports") is None


def test_plan_gives_the_decoder_pattern_and_its_ranked_changes(probe):
    # care and one probabilities: a is set to 1 for sure, b to 0 with doubt, c is left out but likely.
    rich = {"inputs": {"a": [0.99, 0.98], "b": [0.9, 0.45], "c": [0.4, 0.9], "d": [0.05, 0.5]}}
    text = probe.plan_text(rich, ["a", "b", "c", "d"], changes=3)
    assert 'first pattern:\n{"a": 1, "b": 0}' in text
    # Least sure first: flip b (a coin flip), then set c (care 0.4), then d or the flip of a.
    assert "most promising first:\nb=1, c=1, " in text
    assert probe.plan_text({"inputs": {}}, ["a"]).rstrip().endswith("none\nEach one means: keep the first pattern and set that one input to that value. Try them, and use the simulator's reports to choose among them, to combine them or to go beyond them.")
