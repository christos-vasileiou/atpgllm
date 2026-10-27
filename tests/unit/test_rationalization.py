"""Hint construction, trajectory filters and the hint-free SFT layout for STaR."""
import json
import re
from pathlib import Path

import pytest

from atpgllm.training.rationalization import (
    hint_text, hinted_messages, judge, keep, select_problems, sft_example,
)

ROW = {"netlist_id": "n1", "fault": "sa0 n2", "difficulty": "hard",
       "input_vector": json.dumps({"a": 1, "b": "x", "c": 0, "d": 1}),
       "expected_output": json.dumps({"y": 1}),
       "fault_propagation_nets": "n2, n5, y", "backtrack_nets": "a, c, n3"}


def test_selection_keeps_one_row_per_problem_and_the_requested_counts():
    # Difficulty belongs to the problem; each problem has several rows (patterns).
    rows = [dict(ROW, netlist_id=f"{level}{i % 7}", fault=f"sa{i % 2} x", difficulty=level, pattern_index=i)
            for i, level in enumerate(["hard", "random_never_detects", "easy"] * 30)]
    chosen = select_problems(rows, {"hard": 3, "random_never_detects": 50}, seed=1)
    keys = [(r["netlist_id"], r["fault"]) for r in chosen]
    assert len(keys) == len(set(keys))
    assert sum(r["difficulty"] == "hard" for r in chosen) == 3
    assert {r["difficulty"] for r in chosen} == {"hard", "random_never_detects"}
    assert chosen == select_problems(rows, {"hard": 3, "random_never_detects": 50}, seed=1)


def test_hint_modes():
    assert hint_text(ROW, "none") is None
    cube = hint_text(ROW, "cube")
    assert "a: 1, c: 0, d: 1" in cube and "b:" not in cube
    partial = hint_text(ROW, "partial", seed=3)
    assert len(re.findall(r"\b[acd]: [01]", partial)) == 1 and "yours to choose" in partial  # half of 3 care bits
    assert partial == hint_text(ROW, "partial", seed=3)
    path = hint_text(ROW, "path")
    assert "n2, n5, y" in path and "a, c, n3" in path and ": 1" not in path
    with pytest.raises(ValueError):
        hint_text(ROW, "full")
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    assert hinted_messages(messages, cube)[1]["content"] == "u" + cube
    assert messages[1]["content"] == "u"


def _slot(text, *, detection=1.0, vector=None, users=1):
    messages = [{"role": "system", "content": "s"}] + [{"role": "user", "content": "u"}] * users
    messages += [{"role": "assistant", "content": "<think>\n" + text + "</think>\n", "tool_calls": [{
                     "id": "call_0", "type": "function", "function": {"name": "fault_simulation_tool",
                     "arguments": {"input_vector": vector or {"a": 1, "b": "x", "c": 0, "d": 1},
                                   "output_vector": {"y": "x"}, "fault": "sa0 n2", "doc_id": "d"}}}]},
                 {"role": "tool", "name": "fault_simulation_tool", "tool_call_id": "call_0", "content": "{}"},
                 {"role": "assistant", "content": "<think>\n</think>\nINPUT_VECTOR: \"a: 1\""}]
    return {"status": "FINAL", "messages": messages, "reward_components": {"detection": detection},
            "observations": [{"vector": vector or {"a": 1, "b": "x", "c": 0, "d": 1}}]}


def test_judge_and_keep():
    clean = judge(_slot("Setting a=1 activates n2; n5 carries it to y."), ROW)
    assert clean["detected"] and not clean["leak"] and clean["names_internal_path_net"]
    assert clean["first_call_copies_cube"] and keep(clean, require_path_mention=True)
    assert not judge(_slot("Setting a=1 activates the fault.", vector={"a": 0}), ROW)["first_call_copies_cube"]
    for leaking in ("The hint says a=1.", "As the private note suggests", "I was given a test",
                    "Using the provided cube", "Following this note, a=1"):
        assert judge(_slot(leaking), ROW)["leak"], leaking
    assert not keep(judge(_slot("ok", detection=0.0), ROW))
    assert not keep(judge(_slot("ok", users=2), ROW))  # a protocol-repair turn
    # The fault net n2 and the output y are in the prompt; only n5 counts.
    assert not keep(judge(_slot("a=1 activates n2 and sets y."), ROW), require_path_mention=True)


def test_sft_example_uses_the_clean_prompt_and_the_sft_message_layout():
    prompt = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    example = sft_example(_slot("x"), prompt, tools=["T"])
    roles = [m["role"] for m in example["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant"] and example["tools"] == ["T"]
    call = example["messages"][2]["tool_calls"][0]
    assert set(call) == {"type", "function"} and "id" not in call
    assert json.loads(call["function"]["arguments"]["input_vector"]) == {"a": 1, "b": "x", "c": 0, "d": 1}
    assert example["messages"][3] == {"role": "tool", "name": "fault_simulation_tool", "content": "{}"}


V4 = Path(__file__).resolve().parents[3] / "data/freeset/dataset.freeset.asap7sc7p5t_28.rvt.tt.v4/train/data-00001.parquet"


@pytest.mark.skipif(not V4.is_file(), reason="needs the local v4 dataset")
def test_sft_render_reproduces_the_generation_conditioning():
    """What SFT trains on equals what the model saw while generating, hint aside."""
    pq = pytest.importorskip("pyarrow.parquet")
    transformers = pytest.importorskip("transformers")
    from atpgllm.training.completion_runner import CompletionRunner
    from atpgllm.training.conversation import ConversationExample
    from atpgllm.training.search_types import GenerationResult, SearchConfig, SearchContext
    from atpgllm.training.tools import TOOLS
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained("ibm-granite/granite-4.2-8b")
    except OSError:
        pytest.skip("Granite tokenizer unavailable")
    row = pq.read_table(V4).slice(0, 1).to_pylist()[0]
    record = dict(row, netlist={"netlist": row["netlist"], "doc_id": "doc"})
    clean = ConversationExample.prompt_messages_from_record(dict(record), use_tools=True)
    hint = hint_text(row, "partial")
    vector = json.dumps(json.loads(row["input_vector"]))
    outputs = iter([
        "Reasoning about the cone.</think>\n<tool_call>\n<function=fault_simulation_tool>\n"
        f"<parameter=input_vector>\n{vector}\n</parameter>\n<parameter=output_vector>\n"
        f"{row['expected_output']}\n</parameter>\n<parameter=fault>\n{row['fault']}\n</parameter>\n"
        "<parameter=doc_id>\ndoc\n</parameter>\n</function>\n</tool_call>",
        "The output differs.</think>\n\nINPUT_VECTOR: \"a: 1\"\nEXPECTED_OUTPUT: \"y: 1\"\n"
        f"DETECTED_FAULTS: \"{row['fault']}\"",
    ])
    seen = []

    class Scripted:
        temperature, top_p, max_tokens, context_limit = 1.0, 1.0, 4096, 32768

        def __init__(self):
            self.tokenizer = tokenizer

        def generate_requests(self, requests):
            seen.extend(r.prompt for r in requests)
            text = next(outputs)
            return [GenerationResult(text, tuple(tokenizer.encode(text, add_special_tokens=False)),
                                     "stop", None, 0)]

    class Tool:
        def tool(self, call, problem, context):
            return {"vector": json.loads(row["input_vector"]), "result": row["snapshot"], "identity": "i"}

    runner = CompletionRunner(Scripted(), Tool(), max_tool_rounds=4)
    state = runner.initial(None, hinted_messages(clean, hint))
    (final, _), = runner.complete_many([(state, None, SearchContext(0, SearchConfig(), 1), 1.0)])
    assert final.status == "FINAL"
    slot = {"messages": list(final.messages)}
    trained = tokenizer.apply_chat_template(sft_example(slot, clean, TOOLS)["messages"], tools=TOOLS,
                                            tokenize=False)
    hinted = tokenizer.apply_chat_template(sft_example(slot, hinted_messages(clean, hint), TOOLS)["messages"],
                                           tools=TOOLS, tokenize=False)
    assert hint not in trained and trained == hinted.replace(hint, "")
    # Every prompt the model was given is a prefix of the hinted training text,
    # followed by exactly the text it produced.
    assert hinted.startswith(seen[0]) and hinted.startswith(seen[1])
    produced = hinted[len(seen[1]):]
    assert produced.startswith("The output differs.</think>")


def test_extra_examples_are_mixed_at_the_requested_fraction(tmp_path):
    from datasets import IterableDataset
    from atpgllm.training.dataset_utils import load_messages_jsonl, mix_in_examples
    path = tmp_path / "star.jsonl"
    path.write_text("".join(json.dumps({"messages": [{"role": "user", "content": f"star {i}"}],
                                        "tools": [], "star": {"problem": i}}) + "\n" for i in range(5)))
    extra = load_messages_jsonl(path)
    assert set(extra[0]) == {"messages", "tools"}
    base = IterableDataset.from_generator(
        lambda: ({"messages": [{"role": "user", "content": f"base {i}"}], "tools": []} for i in range(2000)))
    mixed = [m["messages"][0]["content"] for m in mix_in_examples(base, extra, 0.2, seed=7)]
    assert sum(t.startswith("base") for t in mixed) == 2000  # the base stream ends the mix
    share = sum(t.startswith("star") for t in mixed) / len(mixed)
    assert 0.17 < share < 0.23
    assert mixed == [m["messages"][0]["content"] for m in mix_in_examples(base, extra, 0.2, seed=7)]


def _judged(mode, slot_index, vector, *, detection=1.0, text="Setting a=1 activates n2."):
    slot = _slot(text, detection=detection, vector=vector)
    return {**slot, "hint_mode": mode, "completion_slot": slot_index, "vector": vector,
            "facts": judge(slot, ROW)}


def test_choose_for_training_prefers_unhinted_then_the_least_revealing_hint():
    from atpgllm.training.rationalization import choose_for_training
    a, b, c = {"a": 1, "c": 0, "d": 1}, {"a": 1, "c": 0, "d": 0}, {"a": 1, "b": 1, "c": 0, "d": 1}
    hinted = [_judged("partial", 0, a), _judged("path", 1, b), _judged("cube", 0, c)]
    # No unhinted success: path, then partial, then cube (not alphabetical).
    assert [s["hint_mode"] for s in choose_for_training(hinted, 3)] == ["path", "partial", "cube"]
    # An unhinted success excludes every hinted attempt of that problem.
    slots = hinted + [_judged("none", 3, a), _judged("none", 1, a), _judged("none", 2, b, detection=0.0)]
    chosen = choose_for_training(slots, 2)
    assert [(s["hint_mode"], s["completion_slot"]) for s in chosen] == [("none", 1)]  # same answer twice
    assert len(choose_for_training(slots, 2, hinted_only_if_unsolved=False)) == 2


def test_selection_excludes_earlier_problems():
    rows = [dict(ROW, netlist_id=f"n{i}", fault="sa0 x", difficulty="hard") for i in range(10)]
    first = select_problems(rows, {"hard": 4}, seed=1)
    second = select_problems(rows, {"hard": 10}, seed=2, exclude=[(r["netlist_id"], r["fault"]) for r in first])
    assert len(second) == 6 and not {r["netlist_id"] for r in first} & {r["netlist_id"] for r in second}


def test_truncated_fact():
    slot = dict(_slot("x"), status="EXHAUSTED")
    assert judge(slot, ROW)["truncated"] and not judge(_slot("x"), ROW)["truncated"]
