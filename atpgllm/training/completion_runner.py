"""Batched assistant/tool transitions shared by inference sampling policies."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import re

from .search_types import (BudgetExceeded, ConversationState, FINAL_STATUSES,
                           GenerationRequest, stable_seed)
from .search_verifier import final_fields
from .revert_template import (parse_tool_call, revert_generation_prompt,
                              restore_generation_prefix, stringify_tool_arguments_for_template)

CALL_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)


class CompletionRunner:
    def __init__(self, generator, verifier, *, max_tool_rounds=1, temperature=0.7, top_p=0.95):
        if max_tool_rounds < 0:
            raise ValueError("max_tool_rounds must be nonnegative")
        self.gen, self.verifier = generator, verifier
        self.max_tool_rounds = max_tool_rounds
        self.temperature, self.top_p = temperature, top_p

    def initial(self, prompt, messages=None):
        if messages is None:
            messages = revert_generation_prompt(prompt, tokenizer=self.gen.tokenizer)
        if not messages or not any(m.get("role") == "user" for m in messages):
            raise ValueError("Cannot recover problem messages from generation prompt")
        return ConversationState(messages=tuple(deepcopy(messages)))

    def render(self, state):
        from .tools import TOOLS
        messages = deepcopy(list(state.messages))
        stringify_tool_arguments_for_template(messages, self.gen.tokenizer.chat_template)
        return self.gen.tokenizer.apply_chat_template(
            messages, tokenize=False, tools=TOOLS if self.max_tool_rounds else None,
            add_generation_prompt=True, truncate_history_thinking=False,
        ) + state.open_text

    @staticmethod
    def instruct(state, instruction):
        """Append an explicit controller message at a saved action boundary."""
        return replace(state, messages=state.messages + ({"role": "user", "content": instruction},),
                       open_text="", final_answer="", status="READY", reason="",
                       mean_logprob=None, open_logprob_sum=0.0, open_logprob_tokens=0,
                       open_logprob_missing=False,
                       readable=state.readable + "\n<controller_instruction>" + instruction + "</controller_instruction>\n")

    def _invalid(self, state, context, reason):
        context.usage.invalid_actions += 1
        if state.repairs >= context.config.repair_limit:
            return replace(state, status="INVALID", reason=reason)
        # Protocol repair is visible in the actual model conversation. It does
        # not claim to be simulator feedback.
        base = replace(state, messages=state.messages + ({"role": "assistant", "content": restore_generation_prefix(state.open_text, self.gen.tokenizer)},),
                       open_text="", repairs=state.repairs + 1)
        return self.instruct(base, f"The previous action was invalid: {reason}. "
                             "Return one complete valid tool call, or one final answer.")

    def _tool_boundary(self, state, problem, context):
        matches = list(CALL_RE.finditer(state.open_text))
        if len(matches) > 1 or state.open_text.count("<tool_call>") > 1:
            return self._invalid(state, context, "Only one tool call is allowed per action")
        if not matches:
            return None
        if state.tool_rounds >= self.max_tool_rounds:
            return replace(state, status="EXHAUSTED", reason="tool_round_limit")
        match = matches[0]
        call = parse_tool_call(match.group())
        if call is None:
            return self._invalid(state, context, "Malformed tool call")
        # Never let text sampled before the observation masquerade as a reply
        # conditioned on that observation. All sampled tokens are still charged.
        suffix = state.open_text[match.end():]
        context.usage.discarded_tokens += len(self.gen.tokenizer.encode(suffix, add_special_tokens=False)) if suffix else 0
        keep = state.open_text[:match.end()]
        readable = state.readable[:-len(suffix)] if suffix else state.readable
        call_id = f"call_{state.tool_rounds}_{stable_seed(context.seed, state.messages, keep):08x}"
        assistant = {"role": "assistant", "content": restore_generation_prefix(keep[:match.start()], self.gen.tokenizer), "tool_calls": [{
            "id": call_id, "type": "function", "function": deepcopy(call)}]}
        try:
            observation = self.verifier.tool(call, problem, context)
        except BudgetExceeded as exc:
            return replace(state, status="EXHAUSTED", reason=str(exc))
        except (ValueError, TypeError, SyntaxError) as exc:
            return self._invalid(state, context, str(exc))
        except Exception as exc:
            context.usage.infrastructure_errors += 1
            context.event(kind="tool_error", error=str(exc))
            return replace(state, status="INFRA_ERROR", reason=str(exc))
        context.usage.tool_calls += 1
        observation = {**observation, "call_id": call_id, "arguments": deepcopy(call["arguments"])}
        tool = {"role": "tool", "name": call["name"], "tool_call_id": call_id,
                "content": observation["result"]}
        context.event(kind="tool", call_id=call_id, vector=observation["vector"],
                      state_messages=len(state.messages))
        return replace(state, messages=state.messages + (assistant, tool), open_text="",
                       readable=readable + "\n<tool_response>\n" + tool["content"] + "\n</tool_response>\n",
                       tool_rounds=state.tool_rounds + 1, actions=state.actions + 1,
                       observations=state.observations + (observation,), status="READY",
                       open_logprob_sum=0.0, open_logprob_tokens=0, open_logprob_missing=False)

    def advance_many(self, jobs):
        """One complete action per job; batch only independent slot ledgers.

        jobs: (state, problem, context, temperature). Partial calls and token
        truncations remain internal to this transition.
        """
        states = [job[0] for job in jobs]
        pending = list(range(len(jobs)))
        while pending:
            requests, indices = [], []
            for i in pending:
                state = states[i]
                _, problem, context, temperature = jobs[i]
                if state.status in FINAL_STATUSES:
                    continue
                boundary = self._tool_boundary(state, problem, context)
                if boundary is not None:
                    states[i] = boundary
                    continue
                if state.actions >= context.config.max_actions or context.remaining_tokens() <= 0:
                    states[i] = replace(state, status="EXHAUSTED", reason="token_or_action_limit")
                    continue
                try:
                    prompt = self.render(state)
                    size = len(self.gen.tokenizer.encode(prompt, add_special_tokens=False))
                    available = self.gen.context_limit - size
                    cap = min(context.config.action_tokens, context.remaining_tokens(), available,
                              self.gen.max_tokens)
                    # Before a call, reserve enough to write the answer after it.
                    if not state.observations and context.remaining_tokens() > context.config.finalization_tokens:
                        cap = min(cap, context.remaining_tokens() - context.config.finalization_tokens)
                    if cap <= 0:
                        states[i] = replace(state, status="EXHAUSTED", reason="context_limit")
                        continue
                    requests.append(GenerationRequest(prompt, cap, temperature, self.top_p,
                                                       context.request_seed(), context.config.prior == "lm"))
                    indices.append(i)
                except Exception as exc:
                    states[i] = replace(state, status="INFRA_ERROR", reason=str(exc))
                    context.usage.infrastructure_errors += 1
            if not requests:
                break
            outputs = self.gen.generate_requests(requests)
            if len(outputs) != len(requests):
                raise RuntimeError("Generator returned a different number of results than requests")
            pending = []
            for i, request, result in zip(indices, requests, outputs):
                state = states[i]
                _, problem, context, _ = jobs[i]
                count = len(result.token_ids)
                if count > request.max_tokens:
                    raise RuntimeError("Backend exceeded reserved generation budget")
                context.usage.generated_tokens += count
                context.usage.prompt_tokens += result.prompt_tokens
                context.event(kind="generation", seed=request.seed, max_tokens=request.max_tokens,
                              finish_reason=result.finish_reason, tokens=count)
                if result.error:
                    context.usage.infrastructure_errors += 1
                    context.usage.generation_usage_unknown += 1
                    # A failed backend may already have spent tokens. Reserve
                    # its entire allowance rather than permitting hidden work
                    # to evade the hard cap; mark the measurement as unknown.
                    context.usage.generated_tokens += request.max_tokens - count
                    states[i] = replace(state, status="INFRA_ERROR", reason=result.error)
                    continue
                missing = state.open_logprob_missing or result.mean_logprob is None
                log_sum = state.open_logprob_sum + (result.mean_logprob or 0.0) * count
                log_count = state.open_logprob_tokens + count
                readable_delta = result.text if state.open_text else restore_generation_prefix(result.text, self.gen.tokenizer)
                state = replace(state, open_text=state.open_text + result.text,
                                readable=state.readable + readable_delta,
                                mean_logprob=log_sum / log_count if log_count and not missing else None,
                                open_logprob_sum=log_sum, open_logprob_tokens=log_count, open_logprob_missing=missing,
                                actions=state.actions + 1, status="GENERATING")
                boundary = self._tool_boundary(state, problem, context)
                if boundary is not None:
                    states[i] = boundary
                    continue
                if result.finish_reason == "length" and count:
                    states[i] = state
                    pending.append(i)
                elif "<tool_call>" in state.open_text:
                    states[i] = self._invalid(state, context, "Incomplete tool call at end of assistant turn")
                else:
                    try:
                        final_fields(state.open_text)
                        states[i] = replace(state, status="FINAL", final_answer=state.open_text,
                                            messages=state.messages + ({"role": "assistant", "content": restore_generation_prefix(state.open_text, self.gen.tokenizer)},),
                                            open_text="")
                    except ValueError as exc:
                        states[i] = self._invalid(state, context, str(exc))
        return states

    def complete_many(self, jobs):
        states = [job[0] for job in jobs]
        checkpoints = [[] for _ in jobs]
        while True:
            active = [i for i, state in enumerate(states) if state.status not in FINAL_STATUSES]
            if not active:
                break
            next_states = self.advance_many([(states[i], *jobs[i][1:]) for i in active])
            for i, state in zip(active, next_states):
                states[i] = state
                if state.status == "READY" and state.observations:
                    checkpoints[i].append(state)
        return list(zip(states, map(tuple, checkpoints)))
