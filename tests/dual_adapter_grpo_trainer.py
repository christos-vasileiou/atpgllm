"""
dual_adapter_grpo_trainer.py
============================

A custom GRPOTrainer that keeps the SFT adapter isolated (without merging) and uses
a dual-adapter approach for reference model computation.

The Problem:
------------
Standard GRPOTrainer with PEFT uses `merge_and_unload()` when you pass a PeftModel
with a peft_config. This permanently fuses LoRA weights into the base model, which
causes precision loss with 4-bit quantization:
  - Before merge: Base (4-bit) + LoRA (higher precision) computed separately
  - After merge: Combined weights re-quantized to 4-bit → loss of fidelity

The Solution:
-------------
Use PEFT's multi-adapter capability:
  1. Keep SFT adapter as "sft" (frozen, never merged)
  2. Add policy adapter as "policy" (trainable, stacked on SFT)
  3. For reference computation: switch to "sft" adapter only
  4. For policy computation: use both "sft" + "policy" adapters

This preserves the SFT adapter's full-precision behavior while allowing GRPO training.

Usage:
------
```python
from dual_adapter_grpo_trainer import DualAdapterGRPOTrainer

# Load your SFT model (already has LoRA adapter)
model, tokenizer = load_model_from_adapter("./sft_checkpoint")

# Create trainer - it will handle adapter setup automatically
trainer = DualAdapterGRPOTrainer(
    model=model,
    reward_funcs=[reward_fn],
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    # Pass policy_lora_config instead of peft_config
    policy_lora_config=LoraConfig(r=8, lora_alpha=16, ...),
)
```
"""

from __future__ import annotations

import asyncio
import contextlib
import atexit
import json
import re
import time
import threading
import warnings
from typing import Any, Callable, List, Optional, Union
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.data_utils import is_conversational, apply_chat_template
import torch
from torch import nn
from peft import LoraConfig, PeftModel
from peft.tuners.lora import LoraLayer
from trl import GRPOTrainer, GRPOConfig
from accelerate.utils import gather
import copy
from tools import TOOLS
from revert_template import revert_qwen2_5_template

# Adapter name constants
REFERENCE_ADAPTER_NAME = "reference"
POLICY_ADAPTER_NAME = "policy"


class DualAdapterGRPOTrainer(GRPOTrainer):
    """
    GRPOTrainer variant that uses a dual-adapter approach to avoid merging.
    
    Key differences from standard GRPOTrainer:
    1. Does NOT merge the SFT adapter - keeps it isolated
    2. Adds a separate "policy" adapter for GRPO training
    3. Overrides reference computation to use SFT adapter instead of disabling all
    
    This preserves the SFT model's behavior with quantization while allowing
    continued training with GRPO.
    """
    
    def __init__(
        self,
        model: PeftModel,
        reward_funcs: Union[Callable, List[Callable]],
        args: GRPOConfig = None,
        train_dataset: Optional[Any] = None,
        processing_class: Optional[Any] = None,
        policy_lora_config: Optional[LoraConfig] = None,
        ref_adapter_name: str = None,
        tools: list[Callable] | None = None,
        tool_functions: list = None,
        **kwargs,
    ):
        """
        Initialize the DualAdapterGRPOTrainer.
        
        Parameters
        ----------
        model : PeftModel
            A PeftModel with the SFT adapter already loaded. This adapter will be
            preserved (not merged) and used as the reference model.
        policy_lora_config : LoraConfig, optional
            Configuration for the policy adapter. If None, will use the same
            configuration as the SFT adapter.
        ref_adapter_name : str, optional
            Name of the existing SFT adapter. If None, will use the active adapter
            name or default to "default".
        **kwargs
            Additional arguments passed to GRPOTrainer.
        """
        # Validate that we have a PeftModel
        if not isinstance(model, PeftModel):
            raise ValueError(
                "DualAdapterGRPOTrainer requires a PeftModel with an existing adapter. "
                "If you're starting from scratch, use the standard GRPOTrainer instead."
            )
        
        # Store the original SFT adapter name before any modifications
        if ref_adapter_name is None:
            ref_adapter_name = model.active_adapter
            if isinstance(ref_adapter_name, list):
                ref_adapter_name = ref_adapter_name[0]
        
        self._ref_adapter_name = ref_adapter_name
        self._policy_adapter_name = POLICY_ADAPTER_NAME
        
        # Rename the SFT adapter if it has the default name to avoid confusion
        if ref_adapter_name == "default" and REFERENCE_ADAPTER_NAME != "default":
            print(f"Renaming SFT adapter from 'default' to '{REFERENCE_ADAPTER_NAME}'")
            self._rename_adapter(model, "default", REFERENCE_ADAPTER_NAME)
            self._ref_adapter_name = REFERENCE_ADAPTER_NAME
        
        # Get or create policy LoRA config
        if policy_lora_config is None:
            # Use the same config as the SFT adapter
            policy_lora_config = self._get_adapter_config(model, self._ref_adapter_name)
            print(f"Using SFT adapter config for policy adapter: r={policy_lora_config.r}")
        
        # Add the policy adapter (this stacks on top of the SFT adapter)
        print(f"Adding policy adapter '{self._policy_adapter_name}' next to SFT adapter '{self._ref_adapter_name}'")
        model.add_adapter(adapter_name=self._policy_adapter_name, peft_config=policy_lora_config)
        
        # Update grpo adapter weight to reference adapter weights (initialization step)
        self._update_adapter_weights(
            model=model, 
            source_adapter_name=self._ref_adapter_name, 
            target_adapter_name=self._policy_adapter_name, 
            tau=1.,
        )

        # Set which adapters are active and trainable
        # We want both adapters active, but only policy is trainable
        self._setup_adapter_training(model)
        
        # Print adapter info
        print(f"Active adapters: {model.active_adapter}")
        model.print_trainable_parameters()
        
        # Call parent __init__ but WITHOUT peft_config to prevent merge_and_unload
        # We've already set up the adapters manually above
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            processing_class=processing_class,
            tools=None,        # CRITICAL: Don't pass tools! Current transformers and trl versions don't support it.
            peft_config=None,  # CRITICAL: Don't pass peft_config to prevent merge!
            **kwargs,
        )
        
        self._has_async_reward_funcs = any(asyncio.iscoroutinefunction(func) for func in self.reward_funcs)
        if self._has_async_reward_funcs:
            self.async_reward_loop_thread, self.async_reward_loop, self.async_reward_loop_ready_event = (
                start_event_loop_in_daemon(name="GRPOTrainer-AsyncRewardLoop")
            )
            # wait until the event loop is running in the daemon thread
            self.async_reward_loop_ready_event.wait()
            atexit.register(shutdown_event_loop_in_daemon, self.async_reward_loop_thread, self.async_reward_loop)
        
        # Initialize tools and tool functions
        self.tools = tools
        self.tool_functions = {tool_function.__name__: tool_function for tool_function in tool_functions} or {}
        # Override ref_model - we use adapter switching instead
        self.ref_model = None
        
    def _update_adapter_weights(self, model: PeftModel, source_adapter_name: str, target_adapter_name: str, tau: float = 1.):
        """
        Performs an exponential moving average (EMA) update of the reference adapter weights.
        The reference adapter is updated using the weights from the GRPO adapter.
        
        Formula: new_target_weights = tau * source_weights + (1 - tau) * old_target_weights
        
        Args:
            model: The model containing both adapters
            source_adapter_name: Name of the source GRPO adapter
            target_adapter_name: Name of the reference adapter to update
            tau: The EMA decay rate (default: 1.)
        """
        # Process parameters layer by layer to minimize memory usage
        # Map source parameter names to their corresponding target parameter names
        param_mapping = {}
        
        # Build the mapping between Policy and Reference adapters
        for name, _ in model.named_parameters():
            if target_adapter_name in name:
                source_name = name.replace(target_adapter_name, source_adapter_name)
                param_mapping[name] = source_name
        
        # Update parameters one by one without storing in memory
        for target_name, source_name in param_mapping.items():
            # Get parameters by name to avoid storing all parameters in memory
            target_param = dict(model.named_parameters())[target_name]
            source_param = dict(model.named_parameters())[source_name]
            
            # Apply EMA update directly: ref = (1-tau)*ref + tau*policy
            target_param.data.mul_(1 - tau).add_(source_param.data, alpha=tau)
    
    def _rename_adapter(self, model: PeftModel, old_name: str, new_name: str):
        """Rename an adapter in the PeftModel."""
        if old_name not in model.peft_config:
            raise ValueError(f"Adapter '{old_name}' not found. Available: {list(model.peft_config.keys())}")
        
        # Copy config with new name
        model.peft_config[new_name] = model.peft_config.pop(old_name)
        
        # Rename in all LoRA layers
        for module in model.modules():
            if isinstance(module, LoraLayer):
                if old_name in module.lora_A:
                    module.lora_A[new_name] = module.lora_A.pop(old_name)
                if old_name in module.lora_B:
                    module.lora_B[new_name] = module.lora_B.pop(old_name)
                if hasattr(module, 'scaling') and old_name in module.scaling:
                    module.scaling[new_name] = module.scaling.pop(old_name)
                if old_name in module.lora_dropout:
                    module.lora_dropout[new_name] = module.lora_dropout.pop(old_name)
        
        model.set_adapter(new_name)
    
    def _get_adapter_config(self, model: PeftModel, adapter_name: str) -> LoraConfig:
        """Get the LoraConfig for an existing adapter."""
        if adapter_name not in model.peft_config:
            raise ValueError(f"Adapter '{adapter_name}' not found. Available: {list(model.peft_config.keys())}")
        return model.peft_config[adapter_name]
    
    def _setup_adapter_training(self, model: PeftModel):
        """
        Configure which adapters are active and trainable.
        
        Strategy:
        - Both SFT and policy adapters are active (their effects are combined)
        - Only policy adapter parameters require gradients
        - SFT adapter parameters are frozen
        """
        # Set both adapters as active - their effects combine additively
        model.set_adapter(self._policy_adapter_name)
        
        # Freeze SFT adapter, make policy adapter trainable
        for name, param in model.named_parameters():
            if self._ref_adapter_name in name and 'lora' in name.lower():
                param.requires_grad = False
            elif self._policy_adapter_name in name and 'lora' in name.lower():
                param.requires_grad = True
    
    @contextlib.contextmanager
    def _use_sft_adapter_only(self, model: PeftModel):
        """
        Context manager to temporarily use only the SFT adapter (for reference computation).
        
        This replaces the standard `disable_adapter()` which would disable ALL adapters.
        Instead, we only disable the policy adapter while keeping SFT active.
        """
        # Store current active adapters
        original_adapter = model.active_adapter
        
        try:
            # Switch to SFT adapter only
            model.set_adapter(self._ref_adapter_name)
            yield
        finally:
            # Restore original adapter configuration
            model.set_adapter(original_adapter)
    
    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        **forward_kwargs,
    ):
        """
        Get per-token log probabilities and entropies.
        
        This method is called for both policy and reference model computation.
        We override the parent to ensure we're using the correct adapter context.
        """
        # Call parent implementation - the adapter context is set externally
        return super()._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep, **forward_kwargs
        )
    
    def _compute_reference_logps(
        self, 
        prompt_completion_ids, 
        attention_mask, 
        logits_to_keep,
        **forward_kwargs
    ):
        """
        Compute reference model log probabilities using SFT adapter only.
        
        This is the key override - instead of disable_adapter(), we use
        _use_sft_adapter_only() to keep the SFT behavior intact.
        """
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        with self._use_sft_adapter_only(unwrapped_model):
            ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                self.model,
                prompt_completion_ids,
                attention_mask,
                logits_to_keep,
                **forward_kwargs,
            )
        
        return ref_per_token_logps

    def _generate_and_score_completions(
        self,
        inputs
    ):
        """
        Override to use our custom reference computation.
        
        This method orchestrates generation and scoring. We intercept it to
        ensure reference log-probs use the SFT adapter instead of disabling all adapters.
        """
        # The parent's _generate_and_score_completions has a section like:
        #   with self.accelerator.unwrap_model(self.model).disable_adapter():
        #       ref_per_token_logps, _ = ...
        #
        # We need to override this behavior. The cleanest way is to override
        # the method and replace the context manager.
        
        # For now, we call the parent and let the monkey-patching handle it
        # A cleaner solution would be to override the full method, but that's
        # maintenance-heavy as TRL updates frequently.
        
        return super()._generate_and_score_completions(inputs)

    def _patch_disable_adapter(self):
        """
        Patch the model's disable_adapter to use our SFT-only context instead.
        
        This is a temporary patch applied during training to ensure reference
        computation uses SFT adapter instead of disabling all adapters.
        """
        model = self.accelerator.unwrap_model(self.model)
        original_disable_adapter = model.disable_adapter
        
        @contextlib.contextmanager
        def patched_disable_adapter():
            """Patched version that switches to SFT adapter instead of disabling all."""
            with self._use_sft_adapter_only(model):
                yield
        
        model.disable_adapter = patched_disable_adapter
        return original_disable_adapter
    
    def train(self, *args, **kwargs):
        """
        Override train to patch disable_adapter behavior.
        """
        model = self.accelerator.unwrap_model(self.model)
        original_disable_adapter = self._patch_disable_adapter()
        
        try:
            return super().train(*args, **kwargs)
        finally:
            # Restore original disable_adapter
            model.disable_adapter = original_disable_adapter
    
    def save_model(self, output_dir: str = None, **kwargs):
        """
        Save the policy adapter (and optionally the SFT adapter).
        
        By default, only saves the policy adapter since SFT adapter hasn't changed.
        """
        if output_dir is None:
            output_dir = self.args.output_dir
        
        model = self.accelerator.unwrap_model(self.model)
        
        # Save the policy adapter
        policy_dir = f"{output_dir}/policy_adapter"
        print(f"Saving policy adapter to: {policy_dir}")
        model.save_pretrained(policy_dir, selected_adapters=[self._policy_adapter_name])
        
        # Save tokenizer
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        
        # Also save a combined checkpoint for easy loading
        combined_dir = f"{output_dir}/combined"
        print(f"Saving combined model (SFT + policy) to: {combined_dir}")
        model.save_pretrained(combined_dir, selected_adapters=[self._ref_adapter_name, self._policy_adapter_name])
        
        # Save adapter names mapping for later loading
        import json
        adapter_info = {
            "ref_adapter_name": self._ref_adapter_name,
            "policy_adapter_name": self._policy_adapter_name,
            "active_adapters": list(model.active_adapter) if isinstance(model.active_adapter, (list, tuple)) else [model.active_adapter],
        }
        with open(f"{output_dir}/adapter_info.json", "w") as f:
            json.dump(adapter_info, f, indent=2)

    @profiling_decorator
    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)

        # Repeat all input columns (but "prompt", "completion", and "completion_ids") to match the num of generations
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

        # This allows for dynamic reward shaping based on training progress.
        reward_kwargs["trainer_state"] = self.state

        async_funcs_info = []  # async custom functions for asyncio.gather
        
        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names, strict=True)
        ):
            if isinstance(reward_func, nn.Module):  # Module (no PretrainedModel) for compat with compiled models
                with profiling_context(self, reward_func_name):
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions, strict=True)]
                        texts = [
                            apply_chat_template(x, reward_processing_class, tools=self.tools, **self.chat_template_kwargs)["text"]
                            for x in messages
                        ]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions, strict=True)]
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            elif asyncio.iscoroutinefunction(reward_func):  # Separate async reward funcs to run them in parallel later
                async_funcs_info.append((i, reward_func, reward_func_name))
            else:
                # Run synchronous reward function
                with profiling_context(self, reward_func_name):
                    completions = self.processing_class.batch_decode(completion_ids_list, skip_special_tokens=True)
                    output_reward_func = reward_func(
                        prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs
                    )
                    # Convert None values to NaN
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Execute async custom functions in parallel using asyncio.gather
        if async_funcs_info:
            completions = self.processing_class.batch_decode(completion_ids_list, skip_special_tokens=True)
            async def _invoke_async_reward(index, func, func_name):
                with profiling_context(self, func_name):
                    output = await func(
                        prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs
                    )
                    output = [r if r is not None else torch.nan for r in output]
                    return index, output

            async def _run_async_funcs():
                coros = [_invoke_async_reward(i, func, func_name) for (i, func, func_name) in async_funcs_info]
                return await asyncio.gather(*coros)

            async_results = asyncio.run_coroutine_threadsafe(_run_async_funcs(), self.async_reward_loop).result()
            for idx, output_reward_func in async_results:
                rewards_per_func[:, idx] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {
                key: value[nan_row_idx] for key, value in reward_kwargs.items() if key != "trainer_state"
            }
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            print(
                f"All reward functions returned None for the following kwargs:\n{row_reward_kwargs}\n"
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)
        return rewards_per_func

    def _generate(self, prompts: list):
        """
        Override the generation function to implement custom tool calling.
        
        This method is called by _generate_and_score_completions and must return:
        - prompt_ids: list of lists of token IDs for prompts
        - completion_ids: list of lists of token IDs for completions
        - tool_mask: list of lists (1 for model tokens, 0 for tool result tokens) or None
        - completions: list of decoded completions (strings or message dicts)
        - total_completion_tokens: total tokens across all completions (for DAPO loss)
        - logprobs: list of lists of log probabilities (or None if not using vLLM IS)
        - extra_fields: dict of extra fields to pass to reward functions
        """
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        generation_start_time = time.perf_counter()
        
        # Copy the prompts to avoid modifying the original list
        prompts = [revert_qwen2_5_template(prompt) for prompt in prompts]
        prompts = copy.deepcopy(prompts)
        
        # Ensure left-padding for decoder-only models during generation
        original_padding_side = self.processing_class.padding_side
        self.processing_class.padding_side = "left"
        
        try:
            # Step 1: Initial generation (reuse parent's single-turn generation)
            prompt_ids, completion_ids, logprobs, extra_fields = self._generate_single_turn(prompts)
            
            # Step 2: Decode completions
            if is_conversational({"prompt": prompts[0]}):
                contents = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
                completions = [[{"role": "assistant", "content": content}] for content in contents]
            else:
                completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
                completions = [revert_qwen2_5_template("<|im_start|>assistant\n" + completion + "<|im_end|>") for completion in completions]
            
            # Step 3: Implement your tool calling loop here
            # Parse tool calls from completions, execute tools, regenerate if needed
            tool_mask, completions, completion_ids, logprobs, tool_call_count, tool_failure_count = self._custom_tool_call_loop(
                prompts, 
                prompt_ids, 
                completion_ids, 
                completions, 
                logprobs
            )
            
            # Step 4: Compute metrics (copied from parent)
            prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids], device=device)
            if tool_mask is not None:
                completion_lengths = torch.tensor([sum(mask) for mask in tool_mask], device=device)
            else:
                completion_lengths = torch.tensor([len(ids) for ids in completion_ids], device=device)
            
            agg_prompt_lengths = self.accelerator.gather(prompt_lengths)
            agg_completion_lengths = self.accelerator.gather(completion_lengths)
            total_prompt_tokens = agg_prompt_lengths.sum()
            total_completion_tokens = agg_completion_lengths.sum()  # = num_items_in_batch, required for the DAPO loss
            
            # Log metrics
            if mode == "train":
                self.state.num_input_tokens_seen += (agg_prompt_lengths.sum() + total_completion_tokens).item()
            self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

            # Log completion lengths, mean, min, max
            self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
            self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
            self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())
            
            # Check for truncated sequences
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids], device=device)
            agg_is_truncated = self.accelerator.gather(is_truncated)
            self._metrics[mode]["completions/clipped_ratio"].append(agg_is_truncated.float().mean().item())
            term_completion_lengths = agg_completion_lengths[~agg_is_truncated]
            if len(term_completion_lengths) == 0:  # edge case where no terminated sequences are found
                term_completion_lengths = torch.zeros(1, device=device)
            self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
            self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
            self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())
            
            if self.tool_functions:
                agg_tool_call_count = self.accelerator.gather(torch.tensor(tool_call_count, device=device)).sum()
                tool_call_frequency = (agg_tool_call_count / len(agg_prompt_lengths)).item()
                self._metrics[mode]["tools/call_frequency"].append(tool_call_frequency)
                agg_tool_failure_count = self.accelerator.gather(torch.tensor(tool_failure_count, device=device)).sum()
                failure_frequency = (
                    (agg_tool_failure_count / agg_tool_call_count).item() if agg_tool_call_count > 0 else 0.0
                )
                self._metrics[mode]["tools/failure_frequency"].append(failure_frequency)

            # Generation throughput: tokens produced per second during inference
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
            # Restore original padding side
            self.processing_class.padding_side = original_padding_side

    def _custom_tool_call_loop(self, prompts, prompt_ids, completion_ids, completions, logprobs):
        """
        Multi-turn tool calling loop for custom <tool_call>...</tool_call> format.
        
        This method:
        1. Parses tool calls from model completions (using <tool_call>{...}</tool_call> tags)
        2. Executes the tools (supports both sync and async functions)
        3. Appends tool results to the conversation
        4. Regenerates completions after tool execution
        5. Repeats until no more tool calls or max length exceeded
        
        Modifies completion_ids, completions, and logprobs in-place.
        
        Returns:
            tool_mask: list of lists where 1 = model token, 0 = tool result token
                       Returns None if no tool calling was performed
        """
        # Ensure left-padding for decoder-only models during generation
        original_padding_side = self.processing_class.padding_side
        self.processing_class.padding_side = "left"
        
        try:
            return self._custom_tool_call_loop_impl(prompts, prompt_ids, completion_ids, completions, logprobs)
        finally:
            # Restore original padding side
            self.processing_class.padding_side = original_padding_side
    
    def _custom_tool_call_loop_impl(self, prompts, prompt_ids, completion_ids, completions, logprobs):
        """Implementation of the tool call loop (called with left-padding set)."""
        # Initialize tool_mask - all 1s initially (all model tokens)
        tool_mask = [[1] * len(ids) for ids in completion_ids]
        tool_call_count = 0
        tool_failure_count = 0
        
        # Parse initial tool calls from completions
        tool_calls = [self._parse_tool_call(completion) for completion in completions]
        idxs_with_tool = [idx for idx, tc in enumerate(tool_calls) if tc is not None]
        tool_calls = [tool_calls[idx] for idx in idxs_with_tool]
        
        # Get max model length for truncation
        if self.use_vllm and self.vllm_mode == "colocate":
            max_model_len = getattr(self.llm.llm_engine.model_config, 'max_model_len', 4096)
        elif not self.use_vllm:
            max_model_len = getattr(self.model.config, 'max_position_embeddings', 4096)
        else:
            raise NotImplementedError(
                f"Unsupported mode detected: use_vllm={self.use_vllm}, vllm_mode={self.vllm_mode}"
            )
        
        while idxs_with_tool:
            # Build conversations with tool calls for samples that need tool execution
            prompt_completion_tools = []
            
            for i, idx in enumerate(idxs_with_tool):
                # Start with the original prompt
                if is_conversational({"prompt": prompts[idx]}):
                    conv = copy.deepcopy(prompts[idx])
                else:
                    # Convert non-conversational to conversational format
                    conv = [{"role": "user", "content": prompts[idx]}]
                
                # Append the assistant's response (which contains the tool call)
                if isinstance(completions[idx], list):
                    # Already in message format
                    for msg in completions[idx]:
                        conv.append(msg)
                else:
                    # String format - wrap in assistant message
                    conv.append({"role": "assistant", "content": completions[idx]})
                
                prompt_completion_tools.append(conv)
            
            # Execute tools and append results to conversations
            for i, idx in enumerate(idxs_with_tool):
                tool_call = tool_calls[i]
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("arguments", {})
                
                if tool_name in self.tool_functions:
                    tool_call_count += 1
                    try:
                        func = self.tool_functions[tool_name]
                        # Handle async functions
                        if asyncio.iscoroutinefunction(func):
                            if self._has_async_reward_funcs:
                                # Use existing event loop
                                future = asyncio.run_coroutine_threadsafe(
                                    func(**tool_args), self.async_reward_loop
                                )
                                result = future.result(timeout=60)  # 60s timeout
                            else:
                                # Create new event loop for this call
                                result = asyncio.get_event_loop().run_until_complete(func(**tool_args))
                        else:
                            result = func(**tool_args)
                    except Exception as e:
                        tool_failure_count += 1
                        result = f"Tool execution failed: {e}"
                else:
                    tool_failure_count += 1
                    result = f"Unknown tool: {tool_name}. Available tools: {list(self.tool_functions.keys())}"
                
                # Append tool result to conversation
                tool_message = {"role": "tool", "name": tool_name, "content": str(result)}
                prompt_completion_tools[i].append(tool_message)
                
                # Also track in completions for the final output
                if isinstance(completions[idx], list):
                    completions[idx].append(tool_message)
            
            # Tokenize to check lengths and prepare for next generation
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
            
            # Check for overlong sequences
            overlong = [len(pct) >= max_model_len for pct in pct_ids]
            
            # Handle overlong sequences - truncate and remove from further processing
            for i, idx in enumerate(idxs_with_tool):
                if overlong[i]:
                    prompt_length = len(prompt_ids[idx])
                    # Truncate to max_completion_length
                    ct = pct_ids[i][prompt_length : prompt_length + self.max_completion_length]
                    completion_ids[idx] = ct
                    # Extend tool_mask for the truncated portion
                    current_mask_len = len(tool_mask[idx])
                    if len(ct) > current_mask_len:
                        tool_mask[idx] += [0] * (len(ct) - current_mask_len)  # Tool result tokens = 0
                    if logprobs is not None:
                        current_logprobs_len = len(logprobs[idx])
                        if len(ct) > current_logprobs_len:
                            logprobs[idx] += [0.0] * (len(ct) - current_logprobs_len)
            
            # Keep only non-overlong items for further processing
            surviving_indices = [i for i, o in enumerate(overlong) if not o]
            idxs_with_tool = [idxs_with_tool[i] for i in surviving_indices]
            prompt_completion_tools = [prompt_completion_tools[i] for i in surviving_indices]
            pct_ids = [pct_ids[i] for i in surviving_indices]
            
            if not idxs_with_tool:
                break  # All overlong, exit tool loop
            
            # Generate new completions after tool execution
            prompt_completion_tool_ids, post_tool_ids, post_tool_logprobs, _ = self._generate_single_turn(
                prompt_completion_tools
            )
            
            # Sanity check: ensure chat template is prefix-preserving
            for i, idx in enumerate(idxs_with_tool):
                pct = prompt_completion_tool_ids[i]
                orig_prompt = prompt_ids[idx]
                if pct[:len(orig_prompt)] != orig_prompt:
                    warnings.warn(
                        "The chat template may not be prefix-preserving. This could affect training quality."
                    )
                    break
            
            # Truncate so that pct[len(prompt_ids[idx]) :] + post_tool does not exceed max_completion_length
            for i, idx in enumerate(idxs_with_tool):
                prompt_len = len(prompt_ids[idx])
                completion_tool_ids = prompt_completion_tool_ids[i][prompt_len:]
                excess_length = len(completion_tool_ids) + len(post_tool_ids[i]) - self.max_completion_length
                
                if excess_length > 0:
                    # First try truncating post_tool_ids
                    if len(post_tool_ids[i]) > excess_length:
                        post_tool_ids[i] = post_tool_ids[i][:-excess_length]
                        if post_tool_logprobs is not None and post_tool_logprobs[i]:
                            post_tool_logprobs[i] = post_tool_logprobs[i][:-excess_length]
                    else:
                        # Need to also truncate completion_tool_ids
                        remaining_excess = excess_length - len(post_tool_ids[i])
                        post_tool_ids[i] = []
                        if post_tool_logprobs is not None:
                            post_tool_logprobs[i] = []
                        if remaining_excess > 0:
                            prompt_completion_tool_ids[i] = prompt_completion_tool_ids[i][:-remaining_excess]
            
            # Update tool_mask, completion_ids, and logprobs
            for i, idx in enumerate(idxs_with_tool):
                prompt_length = len(prompt_ids[idx])
                old_completion_length = len(completion_ids[idx])
                
                # New completion = everything after prompt in pct + post_tool
                new_completion = prompt_completion_tool_ids[i][prompt_length:] + post_tool_ids[i]
                
                # Tool result length = (new completion length) - (old completion length) - (post_tool length)
                pct_completion_len = len(prompt_completion_tool_ids[i]) - prompt_length
                tool_result_length = pct_completion_len - old_completion_length
                post_tool_length = len(post_tool_ids[i])
                
                # Update tool_mask: keep existing, add 0s for tool result, add 1s for post-tool model output
                tool_mask[idx] = tool_mask[idx] + [0] * tool_result_length + [1] * post_tool_length
                
                # Update completion_ids
                completion_ids[idx] = new_completion
                
                # Update logprobs
                if logprobs is not None:
                    logprobs[idx] = logprobs[idx] + [0.0] * tool_result_length
                    if post_tool_logprobs is not None and post_tool_logprobs[i]:
                        logprobs[idx] = logprobs[idx] + post_tool_logprobs[i]
                    else:
                        logprobs[idx] = logprobs[idx] + [0.0] * post_tool_length
            
            # Decode post-tool completions and add to completions list
            post_tool_texts = self.processing_class.batch_decode(post_tool_ids, skip_special_tokens=True)
            
            for i, idx in enumerate(idxs_with_tool):
                if post_tool_texts[i]:
                    post_tool_msg = revert_qwen2_5_template("<|im_start|>assistant\n" + post_tool_texts[i] + "<|im_end|>")
                    if isinstance(completions[idx], list):
                        if isinstance(post_tool_msg, list) and isinstance(post_tool_msg[0], dict):
                            completions[idx] += post_tool_msg
                        elif isinstance(post_tool_msg, dict):
                            completions[idx].append(post_tool_msg)
                    else:
                        # Convert to list format
                        completions[idx] = [
                            {"role": "assistant", "content": completions[idx]},
                            post_tool_msg
                        ]
            
            # Check for further tool calls in post-tool completions
            new_tool_calls = [self._parse_tool_call(text) for text in post_tool_texts]
            new_idxs_with_tool = []
            new_tool_calls_filtered = []
            for i, idx in enumerate(idxs_with_tool):
                if new_tool_calls[i] is not None:
                    new_idxs_with_tool.append(idx)
                    new_tool_calls_filtered.append(new_tool_calls[i])
            
            idxs_with_tool = new_idxs_with_tool
            tool_calls = new_tool_calls_filtered
        
        # Log tool call metrics
        mode = "train" if self.model.training else "eval"
        if tool_call_count > 0:
            self._metrics[mode]["tools/call_count"].append(tool_call_count)
            self._metrics[mode]["tools/failure_count"].append(tool_failure_count)
            self._metrics[mode]["tools/failure_rate"].append(
                tool_failure_count / tool_call_count if tool_call_count > 0 else 0.0
            )
        
        # Return tool_mask (indicates which tokens are from model vs tool results)
        return tool_mask, completions, completion_ids, logprobs, tool_call_count, tool_failure_count

    def _parse_tool_call(self, completion):
        """
        Parse a tool call from model completion.
        
        Expected format: <tool_call>{"name": "function_name", "arguments": {...}}</tool_call>
        
        Override this method for different tool call formats.
        
        Args:
            completion: Either a string or a list of message dicts (conversational format)
            
        Returns:
            Dict with "name" and "arguments" keys if tool call found, None otherwise
        """
        # Extract text content from completion
        if isinstance(completion, list):
            # Conversational format - get content from last message
            text = completion[-1].get("content", "") if completion else ""
        elif isinstance(completion, dict):
            # Single message dict
            text = completion.get("content", "")
        else:
            # Plain string
            text = str(completion)
        
        # Parse <tool_call>{"name": ..., "arguments": ...}</tool_call>
        match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL)
        if match:
            try:
                tool_call_data = json.loads(match.group(1))
                # Validate that it has required fields
                if "name" in tool_call_data:
                    return tool_call_data
            except json.JSONDecodeError:
                pass
        return None

def load_dual_adapter_model(
    sft_checkpoint_path: str,
    policy_checkpoint_path: str = None,
    device_map: str = "auto",
) -> PeftModel:
    """
    Load a model with dual adapters (SFT + policy).
    
    Parameters
    ----------
    sft_checkpoint_path : str
        Path to the SFT adapter checkpoint.
    policy_checkpoint_path : str, optional
        Path to the policy adapter checkpoint. If None, only SFT adapter is loaded.
    device_map : str
        Device map for model loading.
    
    Returns
    -------
    PeftModel
        Model with both adapters loaded.
    """
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    import json
    import os
    
    # Load adapter config to get base model
    with open(os.path.join(sft_checkpoint_path, "adapter_config.json"), "r") as f:
        adapter_config = json.load(f)
    base_model_name = adapter_config["base_model_name_or_path"]
    
    # Load quantized base model
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quant_config,
        device_map=device_map,
        trust_remote_code=True,
    )
    
    # Load SFT adapter
    model = PeftModel.from_pretrained(base_model, sft_checkpoint_path, adapter_name=REFERENCE_ADAPTER_NAME)
    
    # Load policy adapter if provided
    if policy_checkpoint_path is not None:
        model.load_adapter(policy_checkpoint_path, adapter_name=POLICY_ADAPTER_NAME)
        model.set_adapter(POLICY_ADAPTER_NAME)
    
    return model


def start_event_loop_in_daemon(
    name: str | None = None,
) -> tuple[threading.Thread, asyncio.AbstractEventLoop, threading.Event]:
    """
    This function creates a new daemon thread that runs the provided event loop.

    Args:
        name (`str`, *optional*):
            Name of the thread. If `None`, the default thread naming will be used.

    Returns:
        `threading.Thread`:
            The thread running the event loop.
        `asyncio.AbstractEventLoop`:
            The event loop being run in the thread.
        `threading.Event`:
            An event that is set when the loop is ready.
    """
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
    """
    Shutdown an asyncio event loop running in a separate thread.

    This function stops the event loop and waits for the associated thread to finish execution.

    Args:
        thread (`threading.Thread`):
            The thread running the event loop.
        loop (`asyncio.AbstractEventLoop`):
            The asyncio event loop to shut down.
    """
    if loop is None or thread is None:
        return
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
