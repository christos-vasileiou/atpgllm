"""The probe's vector parser must turn unreadable model output into a miss."""

import pytest

from atpgllm.multimodal.tool_probe import _bits


def test_readable_vectors_parse():
    assert _bits('{"a": 1, "b": "x"}') == {"a": 1, "b": "x"}
    assert _bits("a:1,b:0") == {"a": 1, "b": 0}
    assert _bits("a=1,b=x") == {"a": 1, "b": "x"}
    assert _bits({"a": 0}) == {"a": 0}


@pytest.mark.parametrize("text", [
    "a:'0000'}",        # what ended the R5 smoke 504231
    "a:1,b",            # a pair without a value
    "a:1:0",            # too many separators
    "a:yes",
])
def test_unreadable_vectors_are_none(text):
    assert _bits(text) is None


def test_non_string_non_dict_is_none():
    assert _bits(None) is None
    assert _bits([1, 0]) is None
