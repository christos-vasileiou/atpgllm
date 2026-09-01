"""Granite vs ChatML template patching and revert parsers."""

from pathlib import Path

from atpgllm.training.model_utils import (
    patch_chat_template_for_assistant_mask,
    patch_granite42_chat_template_for_assistant_mask,
    patch_granite_chat_template_for_assistant_mask,
    patch_qwen_chat_template_for_assistant_mask,
)
from atpgllm.training.revert_template import (
    get_generation_prompt_suffix,
    parse_tool_call,
    revert_assistant_completion,
    revert_chat_template,
    revert_qwen2_5_template,
    stringify_tool_arguments_for_template,
    wrap_assistant_for_revert,
)


class _Tok:
    def __init__(self, chat_template: str):
        self.chat_template = chat_template


# Minimal fragments that the format-specific patchers search for.
_QWEN_STUB = (
    '{%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}\n'
    "        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n"
    '{%- elif message.role == "assistant" %}\n'
    "        {{- '<|im_start|>' + message.role }}\n"
    "        {%- if message.content %}\n"
    "            {{- '\\n' + message.content }}\n"
    "        {%- endif %}\n"
    "        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}"
)

_GRANITE_STUB = (
    "    {%- elif message.role == 'assistant' %}\n"
    "        {{- '<|start_of_role|>' + message.role + '<|end_of_role|>' + content.val }}\n"
    "        {%- if message.tool_calls %}\n"
    "            {%- for tool_call in message.tool_calls %}\n"
    "            {%- endfor %}\n"
    "        {%- endif %}\n"
    "        {{- '<|end_of_text|>\\n' }}\n"
    "    {%- elif message.role == 'tool' %}\n"
)

# Fragments the Granite 4.2 patcher searches for, plus dispatcher fingerprints.
_GRANITE42_STUB = (
    "{%- set enable_thinking = enable_thinking if enable_thinking is defined else True %}\n"
    "            {{- '<|im_start|>assistant\\n' }}\n"
    "                {%- set include_content = not (truncate_history_thinking and loop.index0 < ns.last_user_idx) %}\n"
    "                    {{- '<tool_call>\\n<function=' ~ tool_call.name ~ '>\\n' -}}\n"
    "                {{- '<|im_end|>\\n' }}\n"
    "        {%- else %}\n"
    "            {# Assistant message doesn't have tool calls. #}\n"
    "                {{- '<|im_start|>assistant\\n' ~ (content | default('', true) | string | trim) ~ '<|im_end|>\\n' }}\n"
    "                    {{- '<|im_start|>assistant\\n' ~ c ~ '<|im_end|>\\n' }}\n"
    "                    {{- '<|im_start|>assistant\\n<|im_end|>\\n' }}\n"
    "        {{- '<|im_start|>assistant\\n<think>\\n' }}\n"
)


def test_revert_qwen2_5_chatml_still_parses():
    rendered = (
        "<|im_start|>system\nYou are an ATPG assistant<|im_end|>\n"
        "<|im_start|>user\nGenerate a pattern<|im_end|>\n"
        "<|im_start|>assistant\n<think>reason</think>\n\nanswer<|im_end|>\n"
    )
    msgs = revert_qwen2_5_template(rendered)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[0]["content"] == "You are an ATPG assistant"
    assert "<think>reason</think>" in msgs[2]["content"]


def test_revert_granite_system_user_assistant():
    rendered = (
        "<|start_of_role|>system<|end_of_role|>You are an ATPG assistant<|end_of_text|>\n"
        "<|start_of_role|>user<|end_of_role|>Generate a pattern for sa0 n15<|end_of_text|>\n"
        "<|start_of_role|>assistant<|end_of_role|><think>reason</think>\n\nanswer<|end_of_text|>\n"
    )
    msgs = revert_chat_template(rendered)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[0]["content"] == "You are an ATPG assistant"
    assert msgs[1]["content"] == "Generate a pattern for sa0 n15"
    assert msgs[2]["content"].startswith("<think>reason</think>")


def test_revert_granite_strips_tools_boilerplate():
    rendered = (
        "<|start_of_role|>system<|end_of_role|>You are an ATPG assistant\n\n"
        "You are a helpful assistant with access to the following tools. "
        "You may call one or more tools.<|end_of_text|>\n"
        "<|start_of_role|>user<|end_of_role|>hi<|end_of_text|>\n"
    )
    msgs = revert_chat_template(rendered, format_hint="granite")
    assert msgs[0] == {"role": "system", "content": "You are an ATPG assistant"}
    assert msgs[1]["role"] == "user"


def test_revert_granite_tool_call_and_response():
    rendered = (
        "<|start_of_role|>user<|end_of_role|>weather?<|end_of_text|>\n"
        '<|start_of_role|>assistant<|end_of_role|><tool_call>\n'
        '{"name": "fault_simulation_tool", "arguments": {"fault": "sa0 n15"}}\n'
        "</tool_call><|end_of_text|>\n"
        "<|start_of_role|>user<|end_of_role|>\n<tool_response>\n"
        "detected\n</tool_response><|end_of_text|>\n"
    )
    msgs = revert_chat_template(rendered, format_hint="granite")
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["tool_calls"][0]["function"]["name"] == "fault_simulation_tool"
    assert msgs[2] == {"role": "tool", "content": "detected"}


def test_granite_generation_suffix_and_wrap():
    tok = _Tok(_GRANITE_STUB)
    assert get_generation_prompt_suffix(tokenizer=tok) == (
        "<|start_of_role|>assistant<|end_of_role|>"
    )
    wrapped = wrap_assistant_for_revert("hello", tokenizer=tok)
    assert wrapped.startswith("<|start_of_role|>assistant<|end_of_role|>hello")
    assert wrapped.endswith("<|end_of_text|>")
    msgs = revert_assistant_completion("hello", tokenizer=tok)
    assert msgs == [{"role": "assistant", "content": "hello"}]


def test_patch_granite_injects_generation_markers():
    tok = _Tok(_GRANITE_STUB)
    assert patch_granite_chat_template_for_assistant_mask(tok) is True
    assert "{%- generation %}" in tok.chat_template
    assert "{%- endgeneration %}" in tok.chat_template
    assert "<|start_of_role|>' + message.role + '<|end_of_role|>' + content.val" not in tok.chat_template
    # Role header stays outside the generation block.
    assert (
        "{{- '<|start_of_role|>' + message.role + '<|end_of_role|>' }}"
        in tok.chat_template
    )
    assert patch_granite_chat_template_for_assistant_mask(tok) is False


def test_patch_qwen_stub_still_works():
    tok = _Tok(_QWEN_STUB)
    assert patch_qwen_chat_template_for_assistant_mask(tok) is True
    assert tok.chat_template.count("{%- generation %}") == 2
    assert patch_qwen_chat_template_for_assistant_mask(tok) is False


def test_dispatcher_routes_granite_and_qwen():
    g = _Tok(_GRANITE_STUB)
    q = _Tok(_QWEN_STUB)
    g42 = _Tok(_GRANITE42_STUB)
    assert patch_chat_template_for_assistant_mask(g) is True
    assert patch_chat_template_for_assistant_mask(q) is True
    assert patch_chat_template_for_assistant_mask(g42) is True
    assert g42.chat_template.count("{%- generation %}") == 4
    already = _Tok("{%- generation %}already{% endgeneration %}")
    assert patch_chat_template_for_assistant_mask(already) is False


def test_patch_granite42_injects_generation_markers():
    tok = _Tok(_GRANITE42_STUB)
    assert patch_granite42_chat_template_for_assistant_mask(tok) is True
    assert tok.chat_template.count("{%- generation %}") == 4
    assert tok.chat_template.count("{%- endgeneration %}") == 4
    # Role header stays outside the generation block on every assistant path.
    assert "{%- generation %}" in tok.chat_template
    assert "'<|im_start|>assistant\\n' ~ (content | default('', true) | string | trim)" not in tok.chat_template
    assert patch_granite42_chat_template_for_assistant_mask(tok) is False


def test_granite42_generation_suffix_includes_think():
    tok = _Tok(_GRANITE42_STUB)
    assert get_generation_prompt_suffix(tokenizer=tok) == "<|im_start|>assistant\n<think>\n"
    wrapped = wrap_assistant_for_revert("reason</think>\nanswer", tokenizer=tok)
    assert wrapped.startswith("<|im_start|>assistant\nreason</think>\nanswer")
    msgs = revert_assistant_completion("reason</think>\nanswer", tokenizer=tok)
    assert msgs[0]["role"] == "assistant"
    assert "reason</think>" in msgs[0]["content"]


def test_revert_granite42_xml_tool_call_and_tools_boilerplate():
    rendered = (
        "<|im_start|>system\nYou are an ATPG assistant\n\n"
        "# Tools\n\nYou have access to the following functions:\n\n<tools>\n"
        '{"name": "fault_simulation_tool"}\n</tools><|im_end|>\n'
        "<|im_start|>user\nGenerate a pattern<|im_end|>\n"
        "<|im_start|>assistant\n<think>reason</think>\n"
        "<tool_call>\n<function=fault_simulation_tool>\n"
        "<parameter=fault>\nsa0 n15\n</parameter>\n"
        '<parameter=input_vector>\n{"n1": "1"}\n</parameter>\n'
        "</function>\n</tool_call>\n<|im_end|>\n"
        "<|im_start|>user\n<tool_response>\ndetected\n</tool_response><|im_end|>\n"
    )
    msgs = revert_chat_template(rendered, format_hint="chatml")
    assert msgs[0] == {"role": "system", "content": "You are an ATPG assistant"}
    assert msgs[1]["role"] == "user"
    assert msgs[2]["role"] == "assistant"
    fn = msgs[2]["tool_calls"][0]["function"]
    assert fn["name"] == "fault_simulation_tool"
    assert fn["arguments"]["fault"] == "sa0 n15"
    assert fn["arguments"]["input_vector"] == {"n1": "1"}
    assert msgs[3] == {"role": "tool", "content": "detected"}


def test_parse_tool_call_json_and_xml():
    json_blob = '<tool_call>{"name": "fault_simulation_tool", "arguments": {"fault": "sa0 n15"}}</tool_call>'
    xml_blob = (
        "<tool_call>\n<function=fault_simulation_tool>\n"
        "<parameter=fault>\nsa0 n15\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    js = parse_tool_call(json_blob)
    xml = parse_tool_call(xml_blob)
    assert js["name"] == "fault_simulation_tool"
    assert js["arguments"] == {"fault": "sa0 n15"}
    assert xml["name"] == "fault_simulation_tool"
    assert xml["arguments"] == {"fault": "sa0 n15"}


def test_stringify_tool_arguments_skips_granite42():
    messages = [{
        "role": "assistant",
        "tool_calls": [{
            "type": "function",
            "function": {"name": "t", "arguments": {"fault": "sa0 n15"}},
        }],
    }]
    stringify_tool_arguments_for_template(messages, _QWEN_STUB)
    assert isinstance(messages[0]["tool_calls"][0]["function"]["arguments"], str)
    messages42 = [{
        "role": "assistant",
        "tool_calls": [{
            "type": "function",
            "function": {"name": "t", "arguments": {"fault": "sa0 n15"}},
        }],
    }]
    stringify_tool_arguments_for_template(messages42, _GRANITE42_STUB)
    assert messages42[0]["tool_calls"][0]["function"]["arguments"] == {"fault": "sa0 n15"}


def test_patch_real_granite42_jinja_if_cached():
    roots = [
        Path.home() / ".cache/huggingface/hub/models--ibm-granite--granite-4.2-8b",
        Path("/proj/trela/christos/.cache/huggingface/hub/models--ibm-granite--granite-4.2-8b"),
    ]
    jinja_paths = []
    for root in roots:
        if root.exists():
            jinja_paths.extend(root.glob("**/chat_template.jinja"))
    if not jinja_paths:
        return
    tok = _Tok(jinja_paths[0].read_text(encoding="utf-8"))
    assert patch_granite42_chat_template_for_assistant_mask(tok) is True
    assert tok.chat_template.count("{%- generation %}") == 4
    assert get_generation_prompt_suffix(tokenizer=tok) == "<|im_start|>assistant\n<think>\n"
    assert patch_chat_template_for_assistant_mask(tok) is False
