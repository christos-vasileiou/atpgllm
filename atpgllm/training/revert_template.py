"""
Revert chat template: parse a formatted chat string back into a list of messages.

Since transformers 4.57.3 does not provide response parsing in tokenizers/processors,
this module implements format-specific parsers for common chat templates.

Supported formats:
- ChatML (Qwen2, Qwen2.5, Qwen3, Qwen3.5, Zephyr, Granite 4.2): <|im_start|>role\\ncontent<|im_end|>
- Granite 3.x / 4.1: <|start_of_role|>role<|end_of_role|>content<|end_of_text|>
- Llama/Mistral: [INST] content [/INST] with <<SYS>> for system
- DeepSeek-R1-Distill-Qwen: <｜User｜ / <｜Assistant｜ / <｜end▁of▁sentence｜ (full-width)

Usage:
    messages = revert_chat_template(chat_string)
    messages = revert_chat_template(chat_string, tokenizer=tokenizer)
    messages = revert_assistant_completion(completion, tokenizer=tokenizer)
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

# -----------------------------------------------------------------------------
# Format detection
# -----------------------------------------------------------------------------

CHATML_IM_START = "<|im_start|>"
CHATML_IM_END = "<|im_end|>"
GRANITE_START_ROLE = "<|start_of_role|>"
GRANITE_END_ROLE = "<|end_of_role|>"
GRANITE_EOT = "<|end_of_text|>"
# Appended by Granite's jinja when tools= is passed (see ibm-granite chat_template.jinja).
GRANITE_TOOL_SYSTEM_PREFIX = (
    "You are a helpful assistant with access to the following tools."
)
LLAMA_INST_START = "[INST]"
LLAMA_INST_END = "[/INST]"
LLAMA_SYS_START = "<<SYS>>"
LLAMA_SYS_END = "<</SYS>>"
# DeepSeek-R1-Distill-Qwen: full-width ｜ (U+FF5C), ▁ (U+2581)
DEEPSEEK_USER = "<｜User｜"
DEEPSEEK_ASSISTANT = "<｜Assistant｜"
DEEPSEEK_EOS = "<｜end▁of▁sentence｜"


def _detect_format_from_string(chat_string: str) -> str:
    """Detect chat format from the string content."""
    if GRANITE_START_ROLE in chat_string and GRANITE_END_ROLE in chat_string:
        return "granite"
    if CHATML_IM_START in chat_string and CHATML_IM_END in chat_string:
        return "chatml"
    if LLAMA_INST_START in chat_string and LLAMA_INST_END in chat_string:
        return "llama"
    if DEEPSEEK_USER in chat_string or DEEPSEEK_ASSISTANT in chat_string or DEEPSEEK_EOS in chat_string:
        return "deepseek_r1"
    # Fallback: try ChatML if im_start present (some models use only start)
    if CHATML_IM_START in chat_string:
        return "chatml"
    if GRANITE_START_ROLE in chat_string:
        return "granite"
    raise ValueError(
        f"Cannot detect chat format from string. "
        f"Expected ChatML, Granite, Llama, or DeepSeek-R1. First 200 chars: {repr(chat_string[:200])}"
    )


def _detect_format_from_tokenizer(tokenizer: "PreTrainedTokenizerBase") -> Optional[str]:
    """Infer format from tokenizer's chat_template (Jinja2 string)."""
    template = getattr(tokenizer, "chat_template", None)
    if template is None:
        return None
    template_str = template if isinstance(template, str) else ""
    # DeepSeek-R1 uses full-width ｜ - check before ChatML (both may have "User")
    if "User｜" in template_str or "Assistant｜" in template_str or "end▁of▁sentence" in template_str:
        return "deepseek_r1"
    if "start_of_role" in template_str and "end_of_role" in template_str:
        return "granite"
    if "im_start" in template_str or "im_end" in template_str:
        return "chatml"
    if "[INST]" in template_str or "[/INST]" in template_str:
        return "llama"
    return None


# -----------------------------------------------------------------------------
# ChatML parser (Qwen, Qwen2, Qwen2.5, Zephyr, etc.)
# -----------------------------------------------------------------------------

TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
TOOL_RESPONSE_PATTERN = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)
# Qwen and Granite 4.2 both splice tools after this heading (wording after it differs).
TOOL_SYSTEM_MARKER = "\n\n# Tools\n\n"
TOOL_SYSTEM_HEADING = "# Tools\n\n"
_XML_FUNCTION_RE = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.DOTALL)
_XML_PARAMETER_RE = re.compile(r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
# Granite 4.2 default thinking generation prompt (jinja source uses escaped newlines).
_GRANITE42_THINK_PROMPT_JINJA = "'<|im_start|>assistant\\n<think>\\n'"


def _coerce_jsonish(value: str) -> Any:
    text = value.strip()
    if text[:1] in "{[":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value
    return value


def _tool_call_from_inner(inner: str) -> Optional[Dict[str, Any]]:
    """Parse JSON or Granite 4.2 XML inside ``<tool_call>``."""
    blob = inner.strip()
    if blob.startswith("{"):
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict) and "name" in data:
            args = data.get("arguments", {})
            if isinstance(args, str):
                try:
                    parsed = json.loads(args)
                    if isinstance(parsed, dict):
                        args = parsed
                except json.JSONDecodeError:
                    pass
            if not isinstance(args, dict):
                args = {}
            return {"name": data["name"], "arguments": args}
    match = _XML_FUNCTION_RE.search(blob)
    if not match:
        return None
    args: Dict[str, Any] = {}
    for pm in _XML_PARAMETER_RE.finditer(match.group(2)):
        args[pm.group(1)] = _coerce_jsonish(pm.group(2))
    return {"name": match.group(1), "arguments": args}


def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Return the first ``<tool_call>`` as ``{name, arguments}``, or None.

    Accepts Qwen-style JSON (``{"name": ..., "arguments": {...}}``) and
    Granite 4.2 XML (``<function=name><parameter=k>v</parameter></function>``).
    ``arguments`` is always a dict when parsing succeeds.
    """
    if not text:
        return None
    match = TOOL_CALL_PATTERN.search(text)
    if not match:
        return None
    return _tool_call_from_inner(match.group(1))


def stringify_tool_arguments_for_template(
    messages,
    chat_template: Optional[str] = None,
) -> None:
    """JSON-stringify tool-call argument dicts in place, except Granite 4.2.

    Qwen and Granite 4.1 jinja accept a JSON string. Granite 4.2 iterates
    ``arguments|items`` and requires a dict.
    """
    tpl = chat_template or ""
    if (
        "im_start" in tpl
        and "enable_thinking" in tpl
        and "<function=" in tpl
        and "start_of_role" not in tpl
    ):
        return
    for message in messages:
        if not isinstance(message, dict) or "tool_calls" not in message:
            continue
        for call in message["tool_calls"]:
            fn = call.get("function") if isinstance(call, dict) else None
            if not isinstance(fn, dict):
                continue
            args = fn.get("arguments")
            if isinstance(args, dict):
                fn["arguments"] = json.dumps(args)


def _assistant_tool_calls_and_text(content: str) -> tuple[List[Dict[str, Any]], str]:
    tool_calls: List[Dict[str, Any]] = []
    for inner in TOOL_CALL_PATTERN.findall(content):
        parsed = _tool_call_from_inner(inner)
        if parsed:
            tool_calls.append({
                "type": "function",
                "function": {
                    "name": parsed["name"],
                    "arguments": parsed["arguments"],
                },
            })
    text_content = TOOL_CALL_PATTERN.sub("", content).strip()
    return tool_calls, text_content


def _strip_tools_system(content: str) -> str:
    """Drop tokenizer-appended tools boilerplate from a system turn."""
    if TOOL_SYSTEM_MARKER in content:
        return content.split(TOOL_SYSTEM_MARKER)[0]
    if content.startswith(TOOL_SYSTEM_HEADING):
        return ""
    return content


def _parse_chatml(chat_string: str) -> List[Dict[str, Any]]:
    """Parse ChatML-formatted string into messages."""
    segments = [s.strip() for s in chat_string.split(CHATML_IM_END) if s.strip()]
    block_pattern = re.compile(r"<\|im_start\|>(.*?)\n(.*)", re.DOTALL)
    messages: List[Dict[str, Any]] = []

    for segment in segments:
        match = block_pattern.search(segment)
        if not match:
            continue

        role = match.group(1).strip()
        content = match.group(2)

        if role == "system":
            if TOOL_SYSTEM_MARKER in content or content.startswith(TOOL_SYSTEM_HEADING):
                content = _strip_tools_system(content)
            if content:
                messages.append({"role": "system", "content": content})

        elif role == "assistant":
            tool_calls, text_content = _assistant_tool_calls_and_text(content)
            msg_obj: Dict[str, Any] = {"role": "assistant"}
            if text_content:
                msg_obj["content"] = text_content
            if tool_calls:
                msg_obj["tool_calls"] = tool_calls
            messages.append(msg_obj)

        elif role == "user":
            tool_responses = list(TOOL_RESPONSE_PATTERN.finditer(content))
            if tool_responses:
                for tr in tool_responses:
                    messages.append({"role": "tool", "content": tr.group(1).strip()})
            else:
                messages.append({"role": "user", "content": content.strip()})

    return messages


# -----------------------------------------------------------------------------
# Llama/Mistral parser
# -----------------------------------------------------------------------------


def _parse_llama(chat_string: str) -> List[Dict[str, Any]]:
    """
    Parse Llama/Mistral [INST] format into messages.

    Format: [INST] user_content [/INST] assistant_content [INST] ... [/INST]
    System: <<SYS>>\\nsystem_content\\n<</SYS>>\\n\\n (before first [INST])
    """
    messages: List[Dict[str, Any]] = []

    # Extract system prompt if present
    if LLAMA_SYS_START in chat_string and LLAMA_SYS_END in chat_string:
        sys_match = re.search(
            rf"{re.escape(LLAMA_SYS_START)}\s*(.*?)\s*{re.escape(LLAMA_SYS_END)}",
            chat_string,
            re.DOTALL,
        )
        if sys_match:
            messages.append({"role": "system", "content": sys_match.group(1).strip()})
            # Remove system block for further parsing
            chat_string = (
                chat_string[: sys_match.start()]
                + chat_string[sys_match.end() :]
            ).strip()

    # Split by [INST] ... [/INST] - user content is inside, assistant is between [/INST] and next [INST]
    # Pattern: [INST] user_content [/INST] assistant_content
    inst_pattern = re.compile(
        rf"{re.escape(LLAMA_INST_START)}\s*(.*?)\s*{re.escape(LLAMA_INST_END)}\s*(.*?)(?={re.escape(LLAMA_INST_START)}|$)",
        re.DOTALL,
    )

    for m in inst_pattern.finditer(chat_string):
        user_content = m.group(1).strip()
        assistant_content = m.group(2).strip() if m.group(2) else ""

        # Remove leading BOS/special tokens from user content if present
        if user_content:
            messages.append({"role": "user", "content": user_content})

        if assistant_content:
            tool_calls, text_content = _assistant_tool_calls_and_text(assistant_content)
            msg_obj: Dict[str, Any] = {"role": "assistant"}
            if text_content:
                msg_obj["content"] = text_content
            if tool_calls:
                msg_obj["tool_calls"] = tool_calls
            messages.append(msg_obj)

    return messages


# -----------------------------------------------------------------------------
# DeepSeek-R1-Distill-Qwen parser
# -----------------------------------------------------------------------------


def _parse_deepseek_r1(chat_string: str) -> List[Dict[str, Any]]:
    """
    Parse DeepSeek-R1-Distill-Qwen format.

    Uses full-width tokens: <｜User｜, <｜Assistant｜, <｜end▁of▁sentence｜
    System prompt is at the start (before first User block).
    """
    messages: List[Dict[str, Any]] = []

    remaining = chat_string

    # Extract system from start (before first User or Assistant)
    first_user = remaining.find(DEEPSEEK_USER)
    first_assistant = remaining.find(DEEPSEEK_ASSISTANT)
    first_block = min(
        first_user if first_user >= 0 else len(remaining),
        first_assistant if first_assistant >= 0 else len(remaining),
    )
    if first_block > 0:
        system_content = remaining[:first_block].strip()
        if system_content:
            messages.append({"role": "system", "content": system_content})
        remaining = remaining[first_block:]

    # Parse alternating User/Assistant blocks
    pos = 0
    while pos < len(remaining):
        if remaining[pos:].startswith(DEEPSEEK_USER):
            prefix = DEEPSEEK_USER
            role = "user"
        elif remaining[pos:].startswith(DEEPSEEK_ASSISTANT):
            prefix = DEEPSEEK_ASSISTANT
            role = "assistant"
        else:
            pos += 1
            continue

        start = pos + len(prefix)
        # Find next block start
        next_user = remaining.find(DEEPSEEK_USER, start)
        next_assistant = remaining.find(DEEPSEEK_ASSISTANT, start)
        next_block = min(
            next_user if next_user >= 0 else len(remaining),
            next_assistant if next_assistant >= 0 else len(remaining),
        )
        content = remaining[start:next_block].strip()
        # Remove trailing EOS if present
        if content.endswith(DEEPSEEK_EOS):
            content = content[: -len(DEEPSEEK_EOS)].strip()

        if role == "assistant":
            tool_calls, text_content = _assistant_tool_calls_and_text(content)
            msg_obj: Dict[str, Any] = {"role": "assistant"}
            if text_content:
                msg_obj["content"] = text_content
            if tool_calls:
                msg_obj["tool_calls"] = tool_calls
            messages.append(msg_obj)
        elif role == "user":
            tool_responses = list(TOOL_RESPONSE_PATTERN.finditer(content))
            if tool_responses:
                for tr in tool_responses:
                    messages.append({"role": "tool", "content": tr.group(1).strip()})
            else:
                messages.append({"role": "user", "content": content.strip()})

        pos = next_block if next_block < len(remaining) else len(remaining)

    return messages


# -----------------------------------------------------------------------------
# Granite 3.x / 4.1 parser (ibm-granite start_of_role chat_template.jinja)
# -----------------------------------------------------------------------------


def _parse_granite(chat_string: str) -> List[Dict[str, Any]]:
    """
    Parse Granite ``<|start_of_role|>role<|end_of_role|>content<|end_of_text|>``.

    Tool results are rendered as a user turn wrapping ``<tool_response>``;
    those are recovered as ``role="tool"``. Assistant ``<tool_call>`` JSON is
    recovered as ``tool_calls``, matching the ChatML parser.
    """
    segments = [s.strip() for s in chat_string.split(GRANITE_EOT) if s.strip()]
    block_pattern = re.compile(
        r"<\|start_of_role\|>(.*?)<\|end_of_role\|>(.*)",
        re.DOTALL,
    )
    messages: List[Dict[str, Any]] = []

    for segment in segments:
        match = block_pattern.search(segment)
        if not match:
            continue

        role = match.group(1).strip()
        content = match.group(2)

        if role == "system":
            if GRANITE_TOOL_SYSTEM_PREFIX in content:
                content = content.split(GRANITE_TOOL_SYSTEM_PREFIX)[0].rstrip()
            if content:
                messages.append({"role": "system", "content": content})

        elif role == "assistant":
            tool_calls, text_content = _assistant_tool_calls_and_text(content)
            msg_obj: Dict[str, Any] = {"role": "assistant"}
            if text_content:
                msg_obj["content"] = text_content
            if tool_calls:
                msg_obj["tool_calls"] = tool_calls
            messages.append(msg_obj)

        elif role == "user":
            tool_responses = list(TOOL_RESPONSE_PATTERN.finditer(content))
            if tool_responses:
                for tr in tool_responses:
                    messages.append({"role": "tool", "content": tr.group(1).strip()})
            else:
                messages.append({"role": "user", "content": content.strip()})

    return messages


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

PARSERS = {
    "chatml": _parse_chatml,
    "granite": _parse_granite,
    "llama": _parse_llama,
    "deepseek_r1": _parse_deepseek_r1,
}


def get_generation_prompt_suffix(
    tokenizer: Optional["PreTrainedTokenizerBase"] = None,
    format_hint: Optional[str] = None,
) -> str:
    """
    Return the format-specific suffix added by add_generation_prompt=True.
    Used to strip trailing prompts before parsing.
    """
    fmt = format_hint
    if fmt is None and tokenizer is not None:
        fmt = _detect_format_from_tokenizer(tokenizer)
    fmt = fmt or "chatml"
    if fmt == "chatml":
        tpl = ""
        if tokenizer is not None:
            raw = getattr(tokenizer, "chat_template", None)
            tpl = raw if isinstance(raw, str) else ""
        if "enable_thinking" in tpl and _GRANITE42_THINK_PROMPT_JINJA in tpl:
            return f"{CHATML_IM_START}assistant\n<think>\n"
        return f"{CHATML_IM_START}assistant\n"
    if fmt == "granite":
        return f"{GRANITE_START_ROLE}assistant{GRANITE_END_ROLE}"
    if fmt == "llama":
        return " "  # Llama often adds a space after [/INST]
    if fmt == "deepseek_r1":
        return f"{DEEPSEEK_ASSISTANT}"
    return ""


def wrap_assistant_for_revert(
    completion: str,
    tokenizer: Optional["PreTrainedTokenizerBase"] = None,
    format_hint: Optional[str] = None,
) -> str:
    """
    Wrap a raw completion string (model output after generation prompt) so
    revert_chat_template can parse it. Returns format-specific wrapped string.
    """
    fmt = format_hint
    if fmt is None and tokenizer is not None:
        fmt = _detect_format_from_tokenizer(tokenizer)
    fmt = fmt or "chatml"
    if fmt == "chatml":
        return f"{CHATML_IM_START}assistant\n{completion}{CHATML_IM_END}"
    if fmt == "granite":
        return f"{GRANITE_START_ROLE}assistant{GRANITE_END_ROLE}{completion}{GRANITE_EOT}"
    if fmt == "llama":
        return f"{LLAMA_INST_START} \n{LLAMA_INST_END} {completion}"
    if fmt == "deepseek_r1":
        return f"{DEEPSEEK_ASSISTANT}{completion}{DEEPSEEK_EOS}"
    return f"{CHATML_IM_START}assistant\n{completion}{CHATML_IM_END}"  # fallback ChatML


def revert_assistant_completion(
    completion: str,
    tokenizer: Optional["PreTrainedTokenizerBase"] = None,
    format_hint: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Parse a raw completion string (model output after generation prompt) into
    message dicts. Wraps the completion in format-specific markers and reverts.
    Returns list of messages (typically one assistant message).
    """
    wrapped = wrap_assistant_for_revert(completion, tokenizer=tokenizer, format_hint=format_hint)
    msgs = revert_chat_template(wrapped, tokenizer=tokenizer, format_hint=format_hint)
    # Filter to assistant messages only (for llama we get [user, assistant])
    return [m for m in msgs if m.get("role") == "assistant"]


def revert_chat_template(
    chat_string: str,
    tokenizer: Optional["PreTrainedTokenizerBase"] = None,
    format_hint: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Parse a formatted chat string back into a list of message dicts.

    Format is auto-detected from the string, or from the tokenizer's chat_template
    if provided. Use format_hint to override detection.

    Parameters
    ----------
    chat_string : str
        The formatted chat string (e.g. from apply_chat_template).
    tokenizer : PreTrainedTokenizerBase, optional
        Tokenizer with chat_template. Used to infer format if string detection fails.
    format_hint : str, optional
        Override: "chatml", "granite", "llama", or "deepseek_r1".

    Returns
    -------
    List[Dict[str, Any]]
        List of messages with "role" and "content" (and optionally "tool_calls").
    """
    fmt = format_hint
    if fmt is None:
        try:
            fmt = _detect_format_from_string(chat_string)
        except ValueError:
            if tokenizer is not None:
                fmt = _detect_format_from_tokenizer(tokenizer)
            if fmt is None:
                raise ValueError(
                    "Could not detect chat format. Pass format_hint='chatml', "
                    "'granite', or 'llama', or ensure the string contains format markers."
                )

    parser = PARSERS.get(fmt)
    if parser is None:
        raise ValueError(f"Unknown format: {fmt}. Supported: {list(PARSERS.keys())}")

    return parser(chat_string)


def revert_qwen2_5_template(chat_string: str) -> List[Dict[str, Any]]:
    """
    Parse a Qwen 2.5 Instruct (ChatML) string back into messages.

    Kept for backward compatibility. Prefer revert_chat_template() for
    multi-model support.
    """
    return revert_chat_template(chat_string, format_hint="chatml")
