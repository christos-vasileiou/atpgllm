"""Tests for streaming dataset skip / prompt-length helpers."""

from datasets import IterableDataset

from atpgllm.training.dataset_utils import (
    _classify_sft_messages_batch,
    _gate_filter_passes,
    count_prompt_tokens,
    extract_prompt_messages,
    filter_streaming_dataset_by_prompt_length,
    format_dataset_for_training,
    skip_streaming_dataset,
    TrainingMode,
)


class _FakeTokenizer:
    """Minimal tokenizer: one token per whitespace-separated word."""

    def apply_chat_template(self, messages, **kwargs):
        return " ".join(m["content"] for m in messages)

    def __call__(self, text, add_special_tokens=False, **kwargs):
        if isinstance(text, list):
            return {"input_ids": [t.split() for t in text]}
        return {"input_ids": text.split()}


def _counter_dataset(n: int = 10) -> IterableDataset:
    def _gen():
        for i in range(n):
            yield {"idx": i}

    return IterableDataset.from_generator(_gen)


def test_skip_streaming_dataset_zero_is_noop():
    ds = _counter_dataset()
    out = skip_streaming_dataset(ds, 0)
    assert out is ds
    assert [ex["idx"] for ex in out] == list(range(10))


def test_skip_streaming_dataset_advances_stream():
    ds = _counter_dataset()
    out = skip_streaming_dataset(ds, 3)
    assert [ex["idx"] for ex in out] == [3, 4, 5, 6, 7, 8, 9]


def test_extract_prompt_messages_excludes_assistant_and_tool():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
        {"role": "assistant", "content": "asst"},
        {"role": "tool", "content": "tool"},
    ]
    assert extract_prompt_messages(messages) == messages[:2]


def test_count_prompt_tokens_system_user_only():
    tok = _FakeTokenizer()
    messages = [
        {"role": "system", "content": "one two"},
        {"role": "user", "content": "three"},
        {"role": "assistant", "content": "four five six seven eight"},
    ]
    assert count_prompt_tokens(tok, messages) == 3


def test_gate_filter_rejects_small_netlists():
    one_gate = "INV_X1 u0 ( .A(a) );\n"
    many_gates = "BUF_X1 u0 ( .A(a) );\n" * 10
    assert not _gate_filter_passes({"netlist": one_gate})
    assert _gate_filter_passes({"netlist": many_gates})


def test_classify_sft_messages_batch_prompt_length():
    tok = _FakeTokenizer()

    def _raw(user_words: str) -> dict:
        return {
            "netlist": "BUF_X1 u0 ( .A(a) );\n" * 10,
            "system_content": "sys",
            "user_content": user_words,
            "reasoning_content": "",
            "answer_content": "",
            "fault": "sa0 n1",
            "input_vector": "{}",
            "expected_output": "{}",
        }

    batch = [_raw("a b"), _raw("a b c d e")]
    statuses = _classify_sft_messages_batch(batch, tok, use_tools=False, max_prompt_length=4)
    assert statuses == ["ok", "length"]


def test_fast_stream_skip_matches_lazy_skip():
    tok = _FakeTokenizer()

    def _raw(i: int) -> dict:
        return {
            "netlist": "BUF_X1 u0 ( .A(a) );\n" * 10,
            "system_content": "s",
            "user_content": f"u{i}",
            "reasoning_content": "",
            "answer_content": "ans {fault}",
            "fault": "sa0 n1",
            "input_vector": "{}",
            "expected_output": "{}",
            "module_name": "m",
        }

    def _gen():
        for i in range(6):
            yield _raw(i)

    raw = IterableDataset.from_generator(_gen)
    fast = format_dataset_for_training(
        raw,
        tok,
        TrainingMode.SFT,
        sft_format="messages",
        max_prompt_length=100,
        skip_buffer_size=2,
        skip_batch_size=2,
        skip_num_workers=1,
    )
    fast_list = list(fast)
    assert len(fast_list) == 4
    assert all("messages" in ex for ex in fast_list)


def test_filter_streaming_dataset_by_prompt_length():
    tok = _FakeTokenizer()

    def _gen():
        yield {"messages": [{"role": "user", "content": "a b"}]}  # 2 tokens
        yield {"messages": [{"role": "user", "content": "a b c d e"}]}  # 5 tokens

    ds = IterableDataset.from_generator(_gen)
    out = filter_streaming_dataset_by_prompt_length(ds, tok, max_prompt_length=4)
    kept = list(out)
    assert len(kept) == 1
    assert kept[0]["messages"][0]["content"] == "a b"
