"""
tool_calling_grpo_trainer.py
============================

Custom GRPOTrainer with multi-turn tool calling and the same generation / vLLM /
reward paths as ``dual_adapter_grpo_trainer.py``, but **without** the dual-adapter
(reference + policy LoRA) setup.

``DualAdapterGRPOTrainer`` keeps two LoRA adapters (frozen SFT as reference,
trainable policy) and switches adapters for reference log-probs. This module
uses the standard ``GRPOTrainer`` model lifecycle (whatever you pass in
``**kwargs``—including optional ``peft_config`` / separate ``ref_model``) and
only adds the tool-calling pipeline on top.

For tool execution details and DDP-safe phases, see ``DualAdapterGRPOTrainer``.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import copy
import json
import re
import threading
import time
import warnings
from typing import Any, Callable, List, Optional, Union

import torch
from accelerate.utils import broadcast_object_list, gather, gather_object
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from trl import GRPOTrainer
from trl.data_utils import (
    apply_chat_template,
    is_conversational,
    prepare_multimodal_messages_vllm,
)
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.models import unwrap_model_for_generation
from vllm import SamplingParams
from vllm.sampling_params import GuidedDecodingParams

from contextlib import nullcontext

from revert_template import revert_qwen2_5_template
from tools import TOOLS, ToolHelper


class ToolCallingGRPOTrainer(GRPOTrainer):
    """
    GRPOTrainer with multi-turn ``<tool_call>...</tool_call>`` support.

    Same generation stack as ``DualAdapterGRPOTrainer`` (vLLM server/colocate,
    Transformers paths, DDP-safe tool continuations, optional async rewards) but
    **no** dual-adapter reference/policy split—use the parent class for PEFT and
    reference modeling.
    """

    def __init__(
        self,
        model,
        reward_funcs: Union[Callable, List[Callable]],
        args=None,
        train_dataset: Optional[Any] = None,
        processing_class: Optional[Any] = None,
        tools: list[Callable] | None = None,
        tool_functions: dict | list | None = None,
        vllm_max_model_len: int | None = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        tool_functions : dict or list of callables, optional
            If a list, it is mapped ``{fn.__name__: fn}``. Executed by name from
            parsed tool calls.
        tools : optional
            Passed to chat templates / vLLM (schemas), same as dual trainer.
        vllm_max_model_len : int, optional
            Hard context limit for the tool loop when using vLLM server.
        **kwargs
            Forwarded to ``GRPOTrainer`` (``model``, ``peft_config``, ``ref_model``, etc.).
        """
        if tool_functions is not None and not isinstance(tool_functions, dict):
            tool_functions = {
                fn.__name__: fn for fn in tool_functions
            }

        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            processing_class=processing_class,
            tools=None,
            **kwargs,
        )

        self._has_async_reward_funcs = any(
            asyncio.iscoroutinefunction(func) for func in self.reward_funcs
        )
        if self._has_async_reward_funcs:
            (
                self.async_reward_loop_thread,
                self.async_reward_loop,
                self.async_reward_loop_ready_event,
            ) = start_event_loop_in_daemon(name="ToolCallingGRPOTrainer-AsyncRewardLoop")
            self.async_reward_loop_ready_event.wait()
            atexit.register(
                shutdown_event_loop_in_daemon,
                self.async_reward_loop_thread,
                self.async_reward_loop,
            )

        self.tools = tools if tools is not None else TOOLS
        self.tool_functions = tool_functions or {}
        self._vllm_max_model_len = vllm_max_model_len
        self.print = False

    @profiling_decorator
    def _move_model_to_vllm(self):
        """
        Sync merged (base + active LoRA) weights to the vLLM server.

        Same per-layer merge as ``DualAdapterGRPOTrainer``; with a single active
        adapter, only that adapter is folded in.
        """
        import bitsandbytes as bnb

        with torch.no_grad():
            for name, module in self.model.named_modules():
                if hasattr(module, "lora_A") and hasattr(module, "base_layer"):
                    base_layer = module.base_layer

                    if hasattr(base_layer.weight, "quant_state"):
                        merged_weight = bnb.functional.dequantize_4bit(
                            base_layer.weight.data,
                            base_layer.weight.quant_state,
                        ).to(torch.bfloat16)
                    else:
                        merged_weight = base_layer.weight.data.clone().to(torch.bfloat16)

                    for adapter_name in module.active_adapters:
                        if adapter_name in module.lora_A:
                            lora_A = module.lora_A[adapter_name].weight.to(torch.bfloat16)
                            lora_B = module.lora_B[adapter_name].weight.to(torch.bfloat16)
                            scaling = module.scaling[adapter_name]
                            merged_weight += (lora_B @ lora_A) * scaling

                    vllm_name = name.replace("base_model.model.", "") + ".weight"
                    merged_weight = merged_weight.to("cuda:0")
                    if hasattr(self, "vllm_client") and self.vllm_client is not None:
                        self.vllm_client.update_named_param(vllm_name, merged_weight)

    @profiling_decorator
    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)

        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
        reward_kwargs["trainer_state"] = self.state

        async_funcs_info = []

        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names, strict=True)
        ):
            if isinstance(reward_func, nn.Module):
                with profiling_context(self, reward_func_name):
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions, strict=True)]
                        texts = [
                            apply_chat_template(
                                x, reward_processing_class, tools=self.tools, **self.chat_template_kwargs
                            )["text"]
                            for x in messages
                        ]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions, strict=True)]
                    reward_inputs = reward_processing_class(
                        text=texts,
                        return_tensors="pt",
                        padding=True,
                        padding_side="right",
                        add_special_tokens=False,
                    )
                    reward_inputs = super(GRPOTrainer, self)._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]

            elif asyncio.iscoroutinefunction(reward_func):
                async_funcs_info.append((i, reward_func, reward_func_name))

            else:
                with profiling_context(self, reward_func_name):
                    completions_decoded = self.processing_class.batch_decode(
                        completion_ids_list, skip_special_tokens=True
                    )
                    output_reward_func = reward_func(
                        prompts=prompts,
                        completions=completions_decoded,
                        completion_ids=completion_ids_list,
                        **reward_kwargs,
                    )
                    output_reward_func = [
                        reward if reward is not None else torch.nan for reward in output_reward_func
                    ]
                    rewards_per_func[:, i] = torch.tensor(
                        output_reward_func, dtype=torch.float32, device=device
                    )

        if self.print:
            for prompt, completion in zip(prompts, completions):
                print(prompt)
                print(completion)
                print("-" * 100)
                self.print = False

        if async_funcs_info:
            completions_decoded = self.processing_class.batch_decode(
                completion_ids_list, skip_special_tokens=True
            )

            async def _invoke_async_reward(index, func, func_name):
                with profiling_context(self, func_name):
                    output = await func(
                        prompts=prompts,
                        completions=completions_decoded,
                        completion_ids=completion_ids_list,
                        **reward_kwargs,
                    )
                    output = [r if r is not None else torch.nan for r in output]
                    return index, output

            async def _run_async_funcs():
                coros = [
                    _invoke_async_reward(i, func, func_name)
                    for (i, func, func_name) in async_funcs_info
                ]
                return await asyncio.gather(*coros)

            async_results = asyncio.run_coroutine_threadsafe(
                _run_async_funcs(), self.async_reward_loop
            ).result()
            for idx, output_reward_func in async_results:
                rewards_per_func[:, idx] = torch.tensor(
                    output_reward_func, dtype=torch.float32, device=device
                )

        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {
                key: value[nan_row_idx]
                for key, value in reward_kwargs.items()
                if key != "trainer_state"
            }
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            print(
                f"All reward functions returned None for the following kwargs:\n{row_reward_kwargs}\n"
                "Please ensure that at least one reward function returns a valid reward."
            )

        rewards_per_func = gather(rewards_per_func)
        return rewards_per_func

    @profiling_decorator
    def _generate(self, prompts: list):
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        generation_start_time = time.perf_counter()

        prompts = [revert_qwen2_5_template(prompt) for prompt in prompts]
        prompts = copy.deepcopy(prompts)

        original_padding_side = self.processing_class.padding_side
        self.processing_class.padding_side = "left"

        try:
            prompt_ids, completion_ids, logprobs, extra_fields = self._generate_single_turn(prompts)

            if is_conversational({"prompt": prompts[0]}):
                contents = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
                completions = [[{"role": "assistant", "content": content}] for content in contents]
            else:
                completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
                completions = [
                    revert_qwen2_5_template("<|im_start|>assistant\n" + completion + "<|im_end|>")
                    for completion in completions
                ]

            tool_mask, completions, completion_ids, logprobs, tool_call_count, tool_failure_count = (
                self._custom_tool_call_loop(prompts, prompt_ids, completion_ids, completions, logprobs)
            )

            prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids], device=device)
            if tool_mask is not None:
                completion_lengths = torch.tensor([sum(mask) for mask in tool_mask], device=device)
            else:
                completion_lengths = torch.tensor([len(ids) for ids in completion_ids], device=device)

            agg_prompt_lengths = self.accelerator.gather(prompt_lengths)
            agg_completion_lengths = self.accelerator.gather(completion_lengths)
            total_completion_tokens = agg_completion_lengths.sum()

            if mode == "train":
                self.state.num_input_tokens_seen += (
                    agg_prompt_lengths.sum() + total_completion_tokens
                ).item()
            self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

            self._metrics[mode]["completions/mean_length"].append(
                agg_completion_lengths.float().mean().item()
            )
            self._metrics[mode]["completions/min_length"].append(
                agg_completion_lengths.float().min().item()
            )
            self._metrics[mode]["completions/max_length"].append(
                agg_completion_lengths.float().max().item()
            )

            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor(
                [ids[-1] not in eos_and_pad for ids in completion_ids], device=device
            )
            agg_is_truncated = self.accelerator.gather(is_truncated)
            self._metrics[mode]["completions/clipped_ratio"].append(
                agg_is_truncated.float().mean().item()
            )
            term_completion_lengths = agg_completion_lengths[~agg_is_truncated]
            if len(term_completion_lengths) == 0:
                term_completion_lengths = torch.zeros(1, device=device)
            self._metrics[mode]["completions/mean_terminated_length"].append(
                term_completion_lengths.float().mean().item()
            )
            self._metrics[mode]["completions/min_terminated_length"].append(
                term_completion_lengths.float().min().item()
            )
            self._metrics[mode]["completions/max_terminated_length"].append(
                term_completion_lengths.float().max().item()
            )

            if self.tool_functions:
                agg_tool_call_count = self.accelerator.gather(
                    torch.tensor(tool_call_count, device=device)
                ).sum()
                tool_call_frequency = (agg_tool_call_count / len(agg_prompt_lengths)).item()
                self._metrics[mode]["tools/call_frequency"].append(tool_call_frequency)
                agg_tool_failure_count = self.accelerator.gather(
                    torch.tensor(tool_failure_count, device=device)
                ).sum()
                failure_frequency = (
                    (agg_tool_failure_count / agg_tool_call_count).item()
                    if agg_tool_call_count > 0
                    else 0.0
                )
                self._metrics[mode]["tools/failure_frequency"].append(failure_frequency)

            generation_elapsed = time.perf_counter() - generation_start_time
            if generation_elapsed > 0 and total_completion_tokens.item() > 0:
                gen_tokens_per_sec = total_completion_tokens.item() / generation_elapsed
                self._metrics[mode]["generation/tokens_per_sec"].append(round(gen_tokens_per_sec, 2))
                self._metrics[mode]["generation/total_time_sec"].append(round(generation_elapsed, 3))

            return (
                prompt_ids,
                completion_ids,
                tool_mask,
                completions,
                total_completion_tokens,
                logprobs,
                extra_fields,
            )
        finally:
            self.processing_class.padding_side = original_padding_side

    def _custom_tool_call_loop(self, prompts, prompt_ids, completion_ids, completions, logprobs):
        original_padding_side = self.processing_class.padding_side
        self.processing_class.padding_side = "left"
        try:
            return self._custom_tool_call_loop_impl(
                prompts, prompt_ids, completion_ids, completions, logprobs
            )
        finally:
            self.processing_class.padding_side = original_padding_side

    def _custom_tool_call_loop_impl(self, prompts, prompt_ids, completion_ids, completions, logprobs):
        tool_mask = [[1] * len(ids) for ids in completion_ids]
        tool_call_count = 0
        tool_failure_count = 0

        device = self.accelerator.device

        tool_calls = [self._parse_tool_call(completion) for completion in completions]
        idxs_with_tool = [idx for idx, tc in enumerate(tool_calls) if tc is not None]
        tool_calls = [tool_calls[idx] for idx in idxs_with_tool]

        if self._vllm_max_model_len is not None:
            max_model_len = self._vllm_max_model_len
        elif self.use_vllm and self.vllm_mode == "colocate":
            max_model_len = getattr(self.llm.llm_engine.model_config, "max_model_len", 4096)
        else:
            max_model_len = getattr(self.model.config, "max_position_embeddings", 4096)

        while True:
            prompts_for_gen = []

            if idxs_with_tool:
                prompt_completion_tools = []
                for i, idx in enumerate(idxs_with_tool):
                    if is_conversational({"prompt": prompts[idx]}):
                        conv = copy.deepcopy(prompts[idx])
                    else:
                        conv = [{"role": "user", "content": prompts[idx]}]

                    if isinstance(completions[idx], list):
                        for msg in completions[idx]:
                            conv.append(msg)
                    else:
                        conv.append({"role": "assistant", "content": completions[idx]})

                    prompt_completion_tools.append(conv)

                for i, idx in enumerate(idxs_with_tool):
                    tool_call = tool_calls[i]
                    tool_name = tool_call.get("name")
                    tool_args = tool_call.get("arguments", {})
                    tool_args["netlist"] = ToolHelper.get_netlist(prompts[idx][1]["content"])

                    if tool_name in self.tool_functions:
                        tool_call_count += 1
                        try:
                            func = self.tool_functions[tool_name]
                            if asyncio.iscoroutinefunction(func):
                                if self._has_async_reward_funcs:
                                    future = asyncio.run_coroutine_threadsafe(
                                        func(**tool_args), self.async_reward_loop
                                    )
                                    result = future.result(timeout=60)
                                else:
                                    result = asyncio.get_event_loop().run_until_complete(
                                        func(**tool_args)
                                    )
                            else:
                                result = func(**tool_args)
                        except Exception as e:
                            tool_failure_count += 1
                            result = f"Tool execution failed: {e}"
                    else:
                        tool_failure_count += 1
                        result = (
                            f"Unknown tool: {tool_name}. "
                            f"Available tools: {list(self.tool_functions.keys())}"
                        )

                    tool_message = {"role": "tool", "name": tool_name, "content": str(result)}
                    prompt_completion_tools[i].append(tool_message)

                    if isinstance(completions[idx], list):
                        completions[idx].append(tool_message)

                tokenized_convs = [
                    self.processing_class.apply_chat_template(
                        conv,
                        tokenize=True,
                        tools=self.tools,
                        add_generation_prompt=True,
                    )
                    for conv in prompt_completion_tools
                ]
                pct_ids = [t if isinstance(t, list) else t["input_ids"] for t in tokenized_convs]

                overlong = [len(pct) >= max_model_len for pct in pct_ids]

                for i, idx in enumerate(idxs_with_tool):
                    if overlong[i]:
                        prompt_length = len(prompt_ids[idx])
                        ct = pct_ids[i][prompt_length : prompt_length + self.max_completion_length]
                        completion_ids[idx] = ct
                        current_mask_len = len(tool_mask[idx])
                        if len(ct) > current_mask_len:
                            tool_mask[idx] += [0] * (len(ct) - current_mask_len)
                        elif len(ct) < current_mask_len:
                            tool_mask[idx] = tool_mask[idx][: len(ct)]
                        if logprobs is not None:
                            current_logprobs_len = len(logprobs[idx])
                            if len(ct) > current_logprobs_len:
                                logprobs[idx] += [0.0] * (len(ct) - current_logprobs_len)
                            elif len(ct) < current_logprobs_len:
                                logprobs[idx] = logprobs[idx][: len(ct)]

                surviving_indices = [i for i, o in enumerate(overlong) if not o]
                idxs_with_tool = [idxs_with_tool[i] for i in surviving_indices]
                prompt_completion_tools = [prompt_completion_tools[i] for i in surviving_indices]
                pct_ids = [pct_ids[i] for i in surviving_indices]

                if idxs_with_tool:
                    prompts_for_gen = prompt_completion_tools

            local_has_gen = len(prompts_for_gen) > 0
            if self.accelerator.num_processes > 1:
                sync_tensor = torch.tensor([1 if local_has_gen else 0], device=device)
                any_has_gen = self.accelerator.gather(sync_tensor).sum().item() > 0
            else:
                any_has_gen = local_has_gen

            if not any_has_gen:
                break

            prompt_completion_tool_ids, post_tool_ids, post_tool_logprobs, _ = (
                self._generate_tool_continuation(prompts_for_gen)
            )

            if not idxs_with_tool:
                continue

            for i, idx in enumerate(idxs_with_tool):
                pct = prompt_completion_tool_ids[i]
                orig_prompt = prompt_ids[idx]
                if pct[: len(orig_prompt)] != orig_prompt:
                    warnings.warn(
                        "The chat template may not be prefix-preserving. "
                        "This could affect training quality."
                    )
                    break

            for i, idx in enumerate(idxs_with_tool):
                prompt_len = len(prompt_ids[idx])
                completion_tool_ids = prompt_completion_tool_ids[i][prompt_len:]
                excess_length = (
                    len(completion_tool_ids) + len(post_tool_ids[i]) - self.max_completion_length
                )

                if excess_length > 0:
                    if len(post_tool_ids[i]) > excess_length:
                        post_tool_ids[i] = post_tool_ids[i][:-excess_length]
                        if post_tool_logprobs is not None and post_tool_logprobs[i]:
                            post_tool_logprobs[i] = post_tool_logprobs[i][:-excess_length]
                    else:
                        remaining_excess = excess_length - len(post_tool_ids[i])
                        post_tool_ids[i] = []
                        if post_tool_logprobs is not None:
                            post_tool_logprobs[i] = []
                        if remaining_excess > 0:
                            prompt_completion_tool_ids[i] = prompt_completion_tool_ids[i][
                                :-remaining_excess
                            ]

            for i, idx in enumerate(idxs_with_tool):
                prompt_length = len(prompt_ids[idx])
                old_completion_length = len(completion_ids[idx])

                new_completion = prompt_completion_tool_ids[i][prompt_length:] + post_tool_ids[i]

                pct_completion_len = len(prompt_completion_tool_ids[i]) - prompt_length
                tool_result_length = pct_completion_len - old_completion_length
                post_tool_length = len(post_tool_ids[i])

                tool_mask[idx] = (
                    tool_mask[idx] + [0] * tool_result_length + [1] * post_tool_length
                )
                completion_ids[idx] = new_completion

                if logprobs is not None:
                    logprobs[idx] = logprobs[idx] + [0.0] * tool_result_length
                    if post_tool_logprobs is not None and post_tool_logprobs[i]:
                        logprobs[idx] = logprobs[idx] + post_tool_logprobs[i]
                    else:
                        logprobs[idx] = logprobs[idx] + [0.0] * post_tool_length

            post_tool_texts = self.processing_class.batch_decode(
                post_tool_ids, skip_special_tokens=True
            )

            for i, idx in enumerate(idxs_with_tool):
                if post_tool_texts[i]:
                    post_tool_msg = revert_qwen2_5_template(
                        "<|im_start|>assistant\n" + post_tool_texts[i] + "<|im_end|>"
                    )
                    if isinstance(completions[idx], list):
                        if isinstance(post_tool_msg, list) and isinstance(post_tool_msg[0], dict):
                            completions[idx] += post_tool_msg
                        elif isinstance(post_tool_msg, dict):
                            completions[idx].append(post_tool_msg)
                    else:
                        completions[idx] = [
                            {"role": "assistant", "content": completions[idx]},
                            post_tool_msg,
                        ]

            new_tool_calls = [self._parse_tool_call(text) for text in post_tool_texts]
            new_idxs_with_tool = []
            new_tool_calls_filtered = []
            for i, idx in enumerate(idxs_with_tool):
                if new_tool_calls[i] is not None:
                    new_idxs_with_tool.append(idx)
                    new_tool_calls_filtered.append(new_tool_calls[i])

            idxs_with_tool = new_idxs_with_tool
            tool_calls = new_tool_calls_filtered

        mode = "train" if self.model.training else "eval"
        if tool_call_count > 0:
            self._metrics[mode]["tools/call_count"].append(tool_call_count)
            self._metrics[mode]["tools/failure_count"].append(tool_failure_count)
            self._metrics[mode]["tools/failure_rate"].append(
                tool_failure_count / tool_call_count if tool_call_count > 0 else 0.0
            )

        for i in range(len(tool_mask)):
            ids_len = len(completion_ids[i])

            if len(tool_mask[i]) != ids_len:
                if len(tool_mask[i]) > ids_len:
                    tool_mask[i] = tool_mask[i][:ids_len]
                else:
                    tool_mask[i] = tool_mask[i] + [1] * (ids_len - len(tool_mask[i]))

            if logprobs is not None and len(logprobs[i]) != ids_len:
                if len(logprobs[i]) > ids_len:
                    logprobs[i] = logprobs[i][:ids_len]
                else:
                    logprobs[i] = logprobs[i] + [0.0] * (ids_len - len(logprobs[i]))

        return tool_mask, completions, completion_ids, logprobs, tool_call_count, tool_failure_count

    def _parse_tool_call(self, completion):
        if isinstance(completion, list):
            text = completion[-1].get("content", "") if completion else ""
        elif isinstance(completion, dict):
            text = completion.get("content", "")
        else:
            text = str(completion)

        match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
        if match:
            try:
                tool_call_data = json.loads(match.group(1))
                if "name" in tool_call_data:
                    return tool_call_data
            except json.JSONDecodeError:
                pass
        return None

    def _generate_tool_continuation(self, prompts: list, max_tokens_override: int = None):
        device = self.accelerator.device
        effective_max_tokens = max_tokens_override or self.max_completion_length

        if not (self.use_vllm and self.vllm_mode == "server"):
            if not prompts:
                return [], [], [], {}
            return self._generate_single_turn(prompts, max_tokens_override=max_tokens_override)

        local_count = len(prompts)

        counts_tensor = torch.tensor([local_count], device=device)
        all_counts = self.accelerator.gather(counts_tensor)

        all_prompts = gather_object(prompts)

        if self.accelerator.is_main_process and all_prompts:
            for prompt in all_prompts:
                if is_conversational({"prompt": prompt}):
                    for message in prompt:
                        if "tool_calls" in message:
                            for call in message["tool_calls"]:
                                args = call["function"]["arguments"]
                                if isinstance(args, dict):
                                    call["function"]["arguments"] = json.dumps(args)

            if is_conversational({"prompt": all_prompts[0]}):
                formatted_prompts = [
                    self.processing_class.apply_chat_template(
                        conversation=conv,
                        tools=self.tools,
                        chat_template=self.chat_template,
                        add_generation_prompt=True,
                        tokenize=False,
                        **(self.chat_template_kwargs or {}),
                    )
                    for conv in all_prompts
                ]
            else:
                formatted_prompts = all_prompts

            sampling_params = {
                "n": 1,
                "repetition_penalty": self.repetition_penalty,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": -1 if self.top_k is None else self.top_k,
                "min_p": 0.0 if self.min_p is None else self.min_p,
                "max_tokens": effective_max_tokens,
                "guided_decoding_regex": self.guided_decoding_regex,
                "generation_kwargs": self.args.generation_kwargs,
            }
            output = self.vllm_client.generate(prompts=formatted_prompts, **sampling_params)
            payload = (
                output["prompt_ids"],
                output["completion_ids"],
                output["logprobs"],
            )
        elif self.accelerator.is_main_process:
            payload = ([], [], [])
        else:
            payload = None

        obj_list = [payload]
        broadcast_object_list(obj_list, from_process=0)
        all_prompt_ids, all_completion_ids, all_logprobs = obj_list[0]

        counts = [int(c) for c in all_counts.tolist()]
        offset = sum(counts[: self.accelerator.process_index])
        prompt_ids = all_prompt_ids[offset : offset + local_count]
        completion_ids = all_completion_ids[offset : offset + local_count]
        logprobs_out = all_logprobs[offset : offset + local_count]

        return prompt_ids, completion_ids, logprobs_out, {}

    @profiling_decorator
    def _generate_single_turn(self, prompts: list, max_tokens_override: int = None):
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        if self.use_vllm:
            if self.vllm_mode == "colocate" and self.args.vllm_enable_sleep_mode:
                torch.cuda.empty_cache()
                self.llm.wake_up(tags=["weights"])
                self.llm.collective_rpc("reload_weights")

            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            if is_conversational({"prompt": prompts[0]}):
                prompts = [prepare_multimodal_messages_vllm(prompt) for prompt in prompts]

            for prompt in prompts:
                if is_conversational({"prompt": prompt}):
                    for message in prompt:
                        if "tool_calls" in message:
                            for call in message["tool_calls"]:
                                args = call["function"]["arguments"]
                                if isinstance(args, dict):
                                    call["function"]["arguments"] = json.dumps(args)

            if self.vllm_mode == "server":
                all_prompts = gather_object(prompts)
                num_generations = self.num_generations if mode == "train" else self.num_generations_eval

                if self.accelerator.is_main_process:
                    ordered_set_of_prompts = all_prompts[::num_generations]

                    effective_max_tokens = (
                        max_tokens_override
                        if max_tokens_override is not None
                        else self.max_completion_length
                    )
                    sampling_params = {
                        "n": num_generations,
                        "repetition_penalty": self.repetition_penalty,
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "top_k": -1 if self.top_k is None else self.top_k,
                        "min_p": 0.0 if self.min_p is None else self.min_p,
                        "max_tokens": effective_max_tokens,
                        "guided_decoding_regex": self.guided_decoding_regex,
                        "generation_kwargs": self.args.generation_kwargs,
                    }
                    with profiling_context(self, "vLLM.generate"):
                        if self.rollout_func is not None:
                            rollout_prompts = ordered_set_of_prompts
                            if rollout_prompts and is_conversational({"prompt": rollout_prompts[0]}):
                                rollout_prompts = [
                                    apply_chat_template(
                                        {"prompt": p}, self.processing_class, **self.chat_template_kwargs
                                    )["prompt"]
                                    for p in rollout_prompts
                                ]
                            output = self.rollout_func(rollout_prompts, self)
                        else:
                            if is_conversational({"prompt": ordered_set_of_prompts[0]}):
                                formatted_prompts = [
                                    self.processing_class.apply_chat_template(
                                        conversation=conv,
                                        tools=self.tools,
                                        chat_template=self.chat_template,
                                        add_generation_prompt=True,
                                        tokenize=False,
                                        **(self.chat_template_kwargs if self.chat_template_kwargs else {}),
                                    )
                                    for conv in ordered_set_of_prompts
                                ]
                                output = self.vllm_client.generate(
                                    prompts=formatted_prompts, **sampling_params
                                )
                            else:
                                output = self.vllm_client.generate(
                                    prompts=ordered_set_of_prompts, **sampling_params
                                )
                        required_keys = {"prompt_ids", "completion_ids", "logprobs"}
                        extra_fields = {k: v for k, v in output.items() if k not in required_keys}
                        payload = (
                            output["prompt_ids"],
                            output["completion_ids"],
                            output["logprobs"],
                            extra_fields,
                        )
                else:
                    payload = None

                obj_list = [payload]
                broadcast_object_list(obj_list, from_process=0)
                all_prompt_ids, all_completion_ids, all_logprobs, all_extra_fields = obj_list[0]

                all_prompt_ids = [ids for ids in all_prompt_ids for _ in range(num_generations)]

                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                prompt_ids = all_prompt_ids[process_slice]
                completion_ids = all_completion_ids[process_slice]
                logprobs = all_logprobs[process_slice]

                extra_fields = {}
                for key, values in all_extra_fields.items():
                    if isinstance(values, list):
                        extra_fields[key] = values[process_slice]
                    else:
                        extra_fields[key] = values

            elif self.vllm_mode == "colocate":
                if self.rollout_func is not None:
                    rollout_prompts = prompts
                    if rollout_prompts and is_conversational({"prompt": rollout_prompts[0]}):
                        rollout_prompts = [
                            apply_chat_template(
                                {"prompt": prompt}, self.processing_class, **self.chat_template_kwargs
                            )["prompt"]
                            for prompt in rollout_prompts
                        ]
                    output = self.rollout_func(rollout_prompts, self)
                    required_keys = {"prompt_ids", "completion_ids", "logprobs"}
                    extra_fields = {k: v for k, v in output.items() if k not in required_keys}
                    prompt_ids = output["prompt_ids"]
                    completion_ids = output["completion_ids"]
                    logprobs = output["logprobs"]
                else:
                    if self.guided_decoding_regex:
                        guided_decoding = GuidedDecodingParams(regex=self.guided_decoding_regex)
                    else:
                        guided_decoding = None

                    effective_max_tokens = (
                        max_tokens_override
                        if max_tokens_override is not None
                        else self.max_completion_length
                    )
                    generation_kwargs = {
                        "n": 1,
                        "repetition_penalty": self.repetition_penalty,
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "top_k": -1 if self.top_k is None else self.top_k,
                        "min_p": 0.0 if self.min_p is None else self.min_p,
                        "max_tokens": effective_max_tokens,
                        "guided_decoding": guided_decoding,
                        "logprobs": 0,
                    }
                    if self.args.generation_kwargs is not None:
                        generation_kwargs.update(self.args.generation_kwargs)
                    sampling_params = SamplingParams(**generation_kwargs)

                    if self.vllm_tensor_parallel_size > 1:
                        orig_size = len(prompts)
                        gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                        torch.distributed.all_gather_object(
                            gathered_prompts, prompts, group=self.tp_group
                        )
                        all_prompts = [p for sublist in gathered_prompts for p in sublist]
                    else:
                        all_prompts = prompts

                    if self.args.vllm_enable_sleep_mode:
                        self.llm.wake_up(tags=["kv_cache"])

                    with profiling_context(self, "vLLM.generate"):
                        if is_conversational({"prompt": prompts[0]}):
                            formatted_prompts = [
                                self.processing_class.apply_chat_template(
                                    conversation=conv,
                                    tools=self.tools,
                                    chat_template=self.chat_template,
                                    add_generation_prompt=True,
                                    tokenize=False,
                                    **(self.chat_template_kwargs if self.chat_template_kwargs else {}),
                                )
                                for conv in all_prompts
                            ]
                            all_outputs = self.llm.generate(
                                formatted_prompts, sampling_params=sampling_params, use_tqdm=False
                            )
                        else:
                            all_outputs = self.llm.generate(
                                all_prompts, sampling_params=sampling_params, use_tqdm=False
                            )

                    all_prompt_ids = [output.prompt_token_ids for output in all_outputs]
                    all_completion_ids = [
                        output.token_ids
                        for outputs in all_outputs
                        for output in outputs.outputs
                    ]
                    all_logprobs = [
                        [next(iter(lp.values())).logprob for lp in output.logprobs]
                        for outputs in all_outputs
                        for output in outputs.outputs
                    ]

                    if self.vllm_tensor_parallel_size > 1:
                        local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                        tp_slice = slice(
                            local_rank_in_group * orig_size,
                            (local_rank_in_group + 1) * orig_size,
                        )
                        prompt_ids = all_prompt_ids[tp_slice]
                        completion_ids = all_completion_ids[tp_slice]
                        logprobs = all_logprobs[tp_slice]
                    else:
                        prompt_ids = all_prompt_ids
                        completion_ids = all_completion_ids
                        logprobs = all_logprobs

                    extra_fields = {}

                    if self.args.vllm_enable_sleep_mode:
                        self.llm.sleep(level=2)

        elif self.use_transformers_paged:
            if is_conversational({"prompt": prompts[0]}):
                processor_outputs = self.processing_class.apply_chat_template(
                    conversation=prompts,
                    tools=self.tools,
                    chat_template=self.chat_template,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    **self.chat_template_kwargs,
                )
            else:
                processor_outputs = self.processing_class(text=prompts)

            with (
                profiling_context(self, "transformers.generate_batch"),
                unwrap_model_for_generation(
                    self.model_wrapped,
                    self.accelerator,
                    gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False)
                if self.is_fsdp_enabled
                else nullcontext(),
            ):
                if self.args.bf16:
                    unwrapped_model.to(torch.bfloat16)
                elif self.args.fp16:
                    unwrapped_model.to(torch.float16)
                if self.args.cast_lm_head_to_fp32:
                    unwrapped_model.lm_head.to(torch.float32)
                with torch.inference_mode():
                    all_outputs = unwrapped_model.generate_batch(
                        processor_outputs["input_ids"],
                        generation_config=self.generation_config,
                        progress_bar=False,
                    )
                    unwrapped_model.train()
            completion_ids = [output.generated_tokens for output in all_outputs.values()]
            prompt_ids = processor_outputs["input_ids"]
            logprobs = None
            extra_fields = {}

        else:
            if is_conversational({"prompt": prompts[0]}):
                generate_inputs = self.processing_class.apply_chat_template(
                    conversation=prompts,
                    tools=self.tools,
                    chat_template=self.chat_template,
                    add_generation_prompt=True,
                    tokenize=True,
                    padding=True,
                    padding_side="left",
                    return_tensors="pt",
                    return_dict=True,
                    **self.chat_template_kwargs,
                )
            else:
                generate_inputs = self.processing_class(
                    text=prompts, padding=True, padding_side="left", return_tensors="pt"
                )
            generate_inputs = super(GRPOTrainer, self)._prepare_inputs(generate_inputs)
            with (
                profiling_context(self, "transformers.generate"),
                unwrap_model_for_generation(
                    self.model_wrapped,
                    self.accelerator,
                    gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False)
                if self.is_fsdp_enabled
                else nullcontext(),
            ):
                prompt_completion_ids = unwrapped_model.generate(
                    **generate_inputs,
                    generation_config=self.generation_config,
                    disable_compile=True,
                )
            prompt_ids, prompt_mask = generate_inputs["input_ids"], generate_inputs["attention_mask"]
            prompt_length = prompt_ids.size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]

            is_eos = completion_ids == self.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
            prompt_ids = [p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool(), strict=True)]
            completion_ids = [
                c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool(), strict=True)
            ]
            logprobs = None
            extra_fields = {}

        return prompt_ids, completion_ids, logprobs, extra_fields


def start_event_loop_in_daemon(
    name: str | None = None,
) -> tuple[threading.Thread, asyncio.AbstractEventLoop, threading.Event]:
    loop = asyncio.new_event_loop()
    loop_ready_event = threading.Event()

    def run_loop():
        asyncio.set_event_loop(loop)
        loop_ready_event.set()
        loop.run_forever()

    thread = threading.Thread(target=run_loop, name=name, daemon=True)
    thread.start()
    return thread, loop, loop_ready_event


def shutdown_event_loop_in_daemon(
    thread: threading.Thread | None,
    loop: asyncio.AbstractEventLoop | None,
) -> None:
    if loop is None or thread is None:
        return
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
