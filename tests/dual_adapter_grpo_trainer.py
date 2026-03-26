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
from tools import TOOLS, ToolHelper
from revert_template import revert_qwen2_5_template, revert_chat_template

# Adapter name constants
REFERENCE_ADAPTER_NAME = "reference"
POLICY_ADAPTER_NAME = "policy"

from contextlib import nullcontext
from accelerate.utils import broadcast_object_list, gather, gather_object
from trl.data_utils import (
    apply_chat_template,
    is_conversational,
    prepare_multimodal_messages_vllm,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from trl.models import unwrap_model_for_generation
from vllm import SamplingParams
from vllm.sampling_params import GuidedDecodingParams

class DualAdapterGRPOTrainer(GRPOTrainer):
    """
    GRPOTrainer variant that uses a dual-adapter (LoRA-on-LoRA) approach.

    Standard GRPOTrainer calls merge_and_unload() when given a PeftModel,
    permanently fusing the SFT LoRA into the 4-bit base weights.  This causes
    precision loss because the merged result gets re-quantised to 4-bit.

    Instead we keep *two* LoRA adapters side by side:
      - "reference"  (frozen SFT adapter)  — acts as the reference model
      - "policy"     (trainable adapter)    — updated by GRPO

    During forward:
      policy logps    = base + reference + policy   (both adapters active)
      reference logps = base + reference            (only SFT adapter active)

    This avoids any merge and preserves the SFT adapter's full-precision
    contribution throughout training.

    Multi-turn tool calling:
      After the initial generation, completions are scanned for <tool_call> tags.
      Matched tool calls are executed, results appended to the conversation, and
      the model is asked to continue.  This loop repeats until no tool calls
      remain or the context window is exhausted.  The loop is DDP-safe: all
      ranks synchronise before each vLLM generation call so collective ops
      never deadlock.
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
        tool_functions: dict = None,
        vllm_max_model_len: int = None,
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
        tool_functions: dict, optional
            Dictionary of tool functions to use for the tool calling. Key is the tool name, value is the tool function.
        **kwargs
            Additional arguments passed to GRPOTrainer.
        """
        # ── Validate input ──
        if not isinstance(model, PeftModel):
            raise ValueError(
                "DualAdapterGRPOTrainer requires a PeftModel with an existing adapter. "
                "If you're starting from scratch, use the standard GRPOTrainer instead."
            )

        # ── Phase 1: Resolve the SFT (reference) adapter name ──
        # The model already has an SFT adapter loaded.  Figure out its name
        # and normalise it to REFERENCE_ADAPTER_NAME for clarity.
        if ref_adapter_name is None:
            ref_adapter_name = model.active_adapter
            if isinstance(ref_adapter_name, list):
                ref_adapter_name = ref_adapter_name[0]
        
        self._ref_adapter_name = ref_adapter_name
        self._policy_adapter_name = POLICY_ADAPTER_NAME
        
        if ref_adapter_name == "default" and REFERENCE_ADAPTER_NAME != "default":
            print(f"Renaming SFT adapter from 'default' to '{REFERENCE_ADAPTER_NAME}'")
            self._rename_adapter(model, "default", REFERENCE_ADAPTER_NAME)
            self._ref_adapter_name = REFERENCE_ADAPTER_NAME

        # ── Phase 2: Create the policy adapter ──
        # A second LoRA adapter is added alongside the frozen SFT one.
        # Its weights are initialised as a copy of the SFT adapter (tau=1)
        # so training starts from the SFT checkpoint.
        if policy_lora_config is None:
            policy_lora_config = self._get_adapter_config(model, self._ref_adapter_name)
            print(f"Using SFT adapter config for policy adapter: r={policy_lora_config.r}")
        
        print(f"Adding policy adapter '{self._policy_adapter_name}' next to SFT adapter '{self._ref_adapter_name}'")
        model.add_adapter(adapter_name=self._policy_adapter_name, peft_config=policy_lora_config)
        
        self._update_adapter_weights(
            model=model, 
            source_adapter_name=self._ref_adapter_name, 
            target_adapter_name=self._policy_adapter_name, 
            tau=1.,
        )

        # ── Phase 3: Freeze SFT, make policy trainable ──
        self._setup_adapter_training(model)
        print(f"Active adapters: {model.active_adapter}")
        model.print_trainable_parameters()

        # ── Phase 4: Initialise the GRPOTrainer base class ──
        # peft_config=None prevents the parent from calling merge_and_unload().
        # tools=None because TRL's built-in tool support is incompatible with
        # our custom multi-turn tool loop; we handle tools ourselves below.
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

        # ── Phase 5: Set up async reward infrastructure & tool state ──
        # If any reward function is an async coroutine, spin up a dedicated
        # daemon event loop so they can run concurrently via asyncio.gather().
        self._has_async_reward_funcs = any(asyncio.iscoroutinefunction(func) for func in self.reward_funcs)
        if self._has_async_reward_funcs:
            self.async_reward_loop_thread, self.async_reward_loop, self.async_reward_loop_ready_event = (
                start_event_loop_in_daemon(name="GRPOTrainer-AsyncRewardLoop")
            )
            self.async_reward_loop_ready_event.wait()
            atexit.register(shutdown_event_loop_in_daemon, self.async_reward_loop_thread, self.async_reward_loop)
        
        self.tools = tools
        self.tool_functions = tool_functions
        self._vllm_max_model_len = vllm_max_model_len
        # We use adapter switching for ref logps — no separate ref_model needed.
        self.ref_model = None
        self.print = False
        
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
        batch_size=None,
        compute_entropy=False,
        pixel_values=None,
        image_grid_thw=None,
        num_images=None,
        pixel_attention_mask=None,
        image_sizes=None,
        token_type_ids=None,
        ):
        """
        Get per-token log probabilities and entropies.

        Called for both policy and reference forward passes.  The active adapter
        at call time determines which set of logps we compute.

        DDP bypass: when computing reference logps the policy adapter is
        disabled, so no policy LoRA parameters participate in the forward pass.
        DDP expects ALL parameters that require gradients to be used, and would
        hang waiting for gradient buckets that never arrive.  Bypassing the DDP
        wrapper for the reference pass avoids this.
        """
        unwrapped_model = self.accelerator.unwrap_model(model)
        active = getattr(unwrapped_model, "active_adapter", "")
        active_list = [active] if isinstance(active, str) else (active if isinstance(active, list) else [])
        
        if self._policy_adapter_name not in active_list:
            model_to_use = unwrapped_model
        else:
            model_to_use = model

        return super()._get_per_token_logps_and_entropies(
            model=model_to_use, 
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            logits_to_keep=logits_to_keep, 
            batch_size=batch_size, 
            compute_entropy=compute_entropy, 
            pixel_values=pixel_values, 
            image_grid_thw=image_grid_thw, 
            num_images=num_images, 
            pixel_attention_mask=pixel_attention_mask, 
            image_sizes=image_sizes, 
            token_type_ids=token_type_ids
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
        Monkey-patch model.disable_adapter() for the duration of training.

        The parent GRPOTrainer computes reference logps inside a
        ``with model.disable_adapter():`` block, which would disable BOTH
        adapters (SFT + policy).  We need it to keep the SFT adapter active.

        Rather than copy-pasting the entire parent method just to change one
        context manager, we swap disable_adapter with a version that switches
        to the SFT-only adapter instead of disabling everything.
        """
        model = self.accelerator.unwrap_model(self.model)
        original_disable_adapter = model.disable_adapter
        
        @contextlib.contextmanager
        def patched_disable_adapter():
            with self._use_sft_adapter_only(model):
                yield
        
        model.disable_adapter = patched_disable_adapter
        return original_disable_adapter
    
    def train(self, *args, **kwargs):
        """Install the disable_adapter patch before training, restore after."""
        model = self.accelerator.unwrap_model(self.model)
        original_disable_adapter = self._patch_disable_adapter()
        
        try:
            return super().train(*args, **kwargs)
        finally:
            model.disable_adapter = original_disable_adapter

    @profiling_decorator
    def _move_model_to_vllm(self):
        """
        Sync the merged (base + SFT + policy) weights to the vLLM server.

        The vLLM server holds the original base-model weights.  After each
        training step we need to push the updated weights so generation
        reflects the latest policy.

        Per-layer pipeline:
          1. Dequantise the 4-bit base weight → bfloat16
          2. Add ΔW from every active LoRA adapter  (W' = W + B·A·s)
          3. Map the PEFT parameter name back to the HuggingFace name that
             vLLM expects (strip "base_model.model." prefix)
          4. Move the tensor to cuda:0 (the vLLM NCCL communicator lives there)
          5. Push via vllm_client.update_named_param()

        Biases are skipped — LoRA never touches them and vLLM already has the
        correct base-model biases.
        """
        import inspect
        
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if hasattr(module, "lora_A") and hasattr(module, "base_layer"):
                    base_layer = module.base_layer

                    # Step 1: dequantise base weight
                    if hasattr(base_layer.weight, "quant_state"):
                        import bitsandbytes as bnb
                        merged_weight = bnb.functional.dequantize_4bit(
                            base_layer.weight.data, 
                            base_layer.weight.quant_state
                        ).to(torch.bfloat16)
                    else:
                        merged_weight = base_layer.weight.data.clone().to(torch.bfloat16)

                    # Step 2: fold in each active LoRA  (W' = W + Σ B_i · A_i · s_i)
                    for adapter_name in module.active_adapters:
                        if adapter_name in module.lora_A:
                            lora_A = module.lora_A[adapter_name].weight.to(torch.bfloat16)
                            lora_B = module.lora_B[adapter_name].weight.to(torch.bfloat16)
                            scaling = module.scaling[adapter_name]
                            merged_weight += (lora_B @ lora_A) * scaling

                    # Steps 3-5: rename, move to cuda:0, push to vLLM server
                    vllm_name = name.replace("base_model.model.", "") + ".weight"
                    merged_weight = merged_weight.to("cuda:0")
                    if hasattr(self, "vllm_client") and self.vllm_client is not None:
                        self.vllm_client.update_named_param(vllm_name, merged_weight)
    
    def save_model(self, output_dir: str = None, **kwargs):
        """
        Save the policy adapter (and optionally the SFT adapter).
        
        By default, only saves the policy adapter since SFT adapter hasn't changed.
        """
        if output_dir is None:
            output_dir = self.args.output_dir
        
        model = self.accelerator.unwrap_model(self.model)
        
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
        """
        Score every (prompt, completion) pair with all registered reward functions.

        Three kinds of reward function are supported, processed in order:
          1. nn.Module  — a learned reward model; gets tokenised text, returns logits
          2. async def  — collected first, then run concurrently via asyncio.gather
                          on a dedicated daemon event loop
          3. regular def — called synchronously one by one

        Returns a (B*G, num_reward_funcs) tensor gathered across all DDP ranks.
        """
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)

        # Build kwargs dict from all non-standard input columns for custom reward fns
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
        reward_kwargs["trainer_state"] = self.state

        async_funcs_info = []
        
        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names, strict=True)
        ):
            # Path 1: Neural reward model — tokenise, forward, take logits[:,0]
            if isinstance(reward_func, nn.Module):
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
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]

            # Path 2: Async function — defer to batch execution below
            elif asyncio.iscoroutinefunction(reward_func):
                async_funcs_info.append((i, reward_func, reward_func_name))

            # Path 3: Synchronous callable
            else:
                with profiling_context(self, reward_func_name):
                    completions = self.processing_class.batch_decode(completion_ids_list, skip_special_tokens=True)
                    output_reward_func = reward_func(
                        prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs
                    )
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        if self.print:
            for prompt, completion in zip(prompts, completions):
                print(prompt)
                print(completion)
                print("-" * 100)
                self.print = False

        # Run all async reward functions concurrently on the daemon event loop
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

        # Warn if every reward function returned None for any sample
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

        # Gather across DDP ranks — rewards must be global for per-group normalisation
        rewards_per_func = gather(rewards_per_func)
        return rewards_per_func

    def _generate(self, prompts: list):
        """
        Full generation pipeline: initial completion → tool loop → metrics.

        Called once per training step by _generate_and_score_completions.
        All DDP ranks enter and exit this method together.

        Pipeline:
          1. _generate_single_turn  — produce initial completions via vLLM
          2. Decode token IDs → text / message dicts
          3. _custom_tool_call_loop — parse tool calls, execute tools,
             re-generate; repeat until no more tool calls or context full
          4. Gather global metrics across DDP ranks (prompt/completion
             lengths, truncation ratios, tool call counts, throughput)

        Returns the tuple expected by the parent class:
          (prompt_ids, completion_ids, tool_mask, completions,
           total_completion_tokens, logprobs, extra_fields)
        """
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        generation_start_time = time.perf_counter()
        
        prompts = [revert_qwen2_5_template(prompt) for prompt in prompts]
        prompts = copy.deepcopy(prompts)
        
        original_padding_side = self.processing_class.padding_side
        self.processing_class.padding_side = "left"
        
        try:
            # ── Step 1: Initial single-turn generation via vLLM ──
            prompt_ids, completion_ids, logprobs, extra_fields = self._generate_single_turn(prompts)

            # ── Step 2: Decode raw token IDs into text / message dicts ──
            if is_conversational({"prompt": prompts[0]}):
                contents = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
                completions = [[{"role": "assistant", "content": content}] for content in contents]
            else:
                completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
                completions = [revert_qwen2_5_template("<|im_start|>assistant\n" + completion + "<|im_end|>") for completion in completions]

            # ── Step 3: Multi-turn tool calling loop ──
            # Scans completions for <tool_call> tags, executes matched tools,
            # appends results to the conversation, and re-generates.  Modifies
            # completion_ids/completions/logprobs in place and returns a
            # tool_mask (1=model token, 0=injected tool-result token).
            tool_mask, completions, completion_ids, logprobs, tool_call_count, tool_failure_count = self._custom_tool_call_loop(
                prompts, prompt_ids, completion_ids, completions, logprobs,
            )

            # ── Step 4: Aggregate metrics across DDP ranks ──
            prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids], device=device)
            if tool_mask is not None:
                completion_lengths = torch.tensor([sum(mask) for mask in tool_mask], device=device)
            else:
                completion_lengths = torch.tensor([len(ids) for ids in completion_ids], device=device)
            
            agg_prompt_lengths = self.accelerator.gather(prompt_lengths)
            agg_completion_lengths = self.accelerator.gather(completion_lengths)
            total_prompt_tokens = agg_prompt_lengths.sum()
            total_completion_tokens = agg_completion_lengths.sum()

            if mode == "train":
                self.state.num_input_tokens_seen += (agg_prompt_lengths.sum() + total_completion_tokens).item()
            self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

            self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
            self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
            self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids], device=device)
            agg_is_truncated = self.accelerator.gather(is_truncated)
            self._metrics[mode]["completions/clipped_ratio"].append(agg_is_truncated.float().mean().item())
            term_completion_lengths = agg_completion_lengths[~agg_is_truncated]
            if len(term_completion_lengths) == 0:
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
        """
        Core tool-call loop.  Iterates until no rank has pending tool calls.

        Each iteration has four DDP-safe phases:

        Phase 1 (local per rank):
            Build the full conversation (prompt + completion + tool result) for
            each sample that contains a <tool_call>.  Execute the matched tool
            function and append the result.  Tokenise the conversation and drop
            samples that now exceed max_model_len (overlong check).

        Phase 2 (DDP collective):
            All ranks vote on whether *any* rank still has prompts that need a
            continuation.  If no rank has work, the loop exits.

        Phase 3 (DDP collective):
            _generate_tool_continuation sends prompts to vLLM with n=1 (one
            completion per unique prompt, no deduplication) and
            max_tokens=max_completion_length.  vLLM internally caps each
            completion to min(max_tokens, max_model_len - prompt_length),
            so short prompts get the full budget while long prompts are
            naturally limited.  Handles variable prompt counts per rank
            via gather/broadcast.

        Phase 4 (local per rank):
            Stitch the new completion onto the existing token sequence.  Update
            tool_mask (1=model token, 0=tool-injected token), completion_ids,
            and logprobs.  Parse the new completion for further tool calls —
            if found, the loop repeats.
        """
        tool_mask = [[1] * len(ids) for ids in completion_ids]
        tool_call_count = 0
        tool_failure_count = 0
        
        device = self.accelerator.device

        tool_calls = [self._parse_tool_call(completion) for completion in completions]
        idxs_with_tool = [idx for idx, tc in enumerate(tool_calls) if tc is not None]
        tool_calls = [tool_calls[idx] for idx in idxs_with_tool]

        # Determine the hard context-window limit.  For vLLM server mode this
        # is the server's --max-model-len (not the model's max_position_embeddings).
        if self._vllm_max_model_len is not None:
            max_model_len = self._vllm_max_model_len
        elif self.use_vllm and self.vllm_mode == "colocate":
            max_model_len = getattr(self.llm.llm_engine.model_config, 'max_model_len', 4096)
        else:
            max_model_len = getattr(self.model.config, 'max_position_embeddings', 4096)

        while True:
            # ==================================================================
            # Phase 1 (local): execute tools, check overlong
            # ==================================================================
            prompts_for_gen = []
            local_max_tokens = self.max_completion_length

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
                                    result = asyncio.get_event_loop().run_until_complete(func(**tool_args))
                            else:
                                result = func(**tool_args)
                        except Exception as e:
                            tool_failure_count += 1
                            result = f"Tool execution failed: {e}"
                    else:
                        tool_failure_count += 1
                        result = f"Unknown tool: {tool_name}. Available tools: {list(self.tool_functions.keys())}"

                    tool_message = {"role": "tool", "name": tool_name, "content": str(result)}
                    prompt_completion_tools[i].append(tool_message)

                    if isinstance(completions[idx], list):
                        completions[idx].append(tool_message)

                tokenized_convs = [
                    self.processing_class.apply_chat_template(
                        conv, tokenize=True, tools=self.tools, add_generation_prompt=True,
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

            # ==================================================================
            # Phase 2 (DDP sync): decide whether to continue
            # ==================================================================
            # All ranks vote: does *any* rank still have prompts needing a
            # continuation?  If not, every rank breaks out together.
            local_has_gen = len(prompts_for_gen) > 0
            if self.accelerator.num_processes > 1:
                sync_tensor = torch.tensor([1 if local_has_gen else 0], device=device)
                any_has_gen = self.accelerator.gather(sync_tensor).sum().item() > 0
            else:
                any_has_gen = local_has_gen

            if not any_has_gen:
                break

            # ==================================================================
            # Phase 3 (DDP collective): generate tool-call continuations
            # ==================================================================
            # All ranks enter _generate_tool_continuation together (even those
            # with empty prompt lists) so the internal gather/broadcast ops
            # don't deadlock.
            #
            # max_tokens is set to max_completion_length for all prompts.
            # vLLM internally caps each completion to
            # min(max_tokens, max_model_len - prompt_length), so short prompts
            # get the full budget while long prompts are naturally limited.
            # Phase 1's overlong check already removed prompts that exceed
            # max_model_len, so no request can cause a vLLM rejection.
            prompt_completion_tool_ids, post_tool_ids, post_tool_logprobs, _ = (
                self._generate_tool_continuation(prompts_for_gen)
            )

            # ==================================================================
            # Phase 4 (local): stitch results, parse next tool calls
            # ==================================================================
            # Ranks that had no prompts in this iteration skip straight to the
            # top of the loop (their idxs_with_tool stays empty for the next
            # Phase 2 vote).
            if not idxs_with_tool:
                continue

            # 4a. Sanity check: the re-tokenised conversation should still
            #     start with the original prompt tokens (prefix-preserving).
            for i, idx in enumerate(idxs_with_tool):
                pct = prompt_completion_tool_ids[i]
                orig_prompt = prompt_ids[idx]
                if pct[:len(orig_prompt)] != orig_prompt:
                    warnings.warn(
                        "The chat template may not be prefix-preserving. This could affect training quality."
                    )
                    break

            # 4b. Truncate if (old_completion + tool_result + new_completion)
            #     exceeds max_completion_length.  Trim from the tail of the
            #     new model output first, then the tool-result prefix.
            for i, idx in enumerate(idxs_with_tool):
                prompt_len = len(prompt_ids[idx])
                completion_tool_ids = prompt_completion_tool_ids[i][prompt_len:]
                excess_length = len(completion_tool_ids) + len(post_tool_ids[i]) - self.max_completion_length

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
                            prompt_completion_tool_ids[i] = prompt_completion_tool_ids[i][:-remaining_excess]

            # 4c. Stitch: concatenate old_completion + tool_tokens + new_output
            #     and update tool_mask / logprobs to match.
            #     tool_mask: 0 for injected tool-result tokens, 1 for model tokens.
            #     logprobs:  0.0 placeholders for non-model tokens.
            for i, idx in enumerate(idxs_with_tool):
                prompt_length = len(prompt_ids[idx])
                old_completion_length = len(completion_ids[idx])

                new_completion = prompt_completion_tool_ids[i][prompt_length:] + post_tool_ids[i]

                pct_completion_len = len(prompt_completion_tool_ids[i]) - prompt_length
                tool_result_length = pct_completion_len - old_completion_length
                post_tool_length = len(post_tool_ids[i])

                tool_mask[idx] = tool_mask[idx] + [0] * tool_result_length + [1] * post_tool_length
                completion_ids[idx] = new_completion

                if logprobs is not None:
                    logprobs[idx] = logprobs[idx] + [0.0] * tool_result_length
                    if post_tool_logprobs is not None and post_tool_logprobs[i]:
                        logprobs[idx] = logprobs[idx] + post_tool_logprobs[i]
                    else:
                        logprobs[idx] = logprobs[idx] + [0.0] * post_tool_length

            # 4d. Decode the new model output and check for further tool calls.
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
        
        # Log tool call metrics
        mode = "train" if self.model.training else "eval"
        if tool_call_count > 0:
            self._metrics[mode]["tools/call_count"].append(tool_call_count)
            self._metrics[mode]["tools/failure_count"].append(tool_failure_count)
            self._metrics[mode]["tools/failure_rate"].append(
                tool_failure_count / tool_call_count if tool_call_count > 0 else 0.0
            )

        # Ensure tool_mask, logprobs, and completion_ids all have identical lengths per sequence.
        # Mismatches cause shape errors when these are padded into tensors downstream.
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
    
    def _generate_tool_continuation(self, prompts: list, max_tokens_override: int = None):
        """
        Generate continuations for the tool-call loop.

        Unlike _generate_single_turn, this method:
        - Uses n=1 (each prompt gets exactly one completion, no deduplication)
        - Handles DDP with variable-length prompt lists per rank
        - Does not sync model weights (already done by the initial generation)

        All ranks MUST call this together even if some have empty prompt lists,
        because it uses DDP collective operations internally.
        """
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
            output = self.vllm_client.generate(
                prompts=formatted_prompts, **sampling_params
            )
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

    def _generate_single_turn(self, prompts: list, max_tokens_override: int = None):
        """
        Produce completions for the *initial* generation (one turn, no tools).

        Used only by _generate() for the first round.  Tool-call continuations
        use _generate_tool_continuation() instead, which avoids the n=num_generations
        deduplication and handles variable prompt counts across DDP ranks.

        Three backend paths:
          1. vLLM server  — gather prompts to rank 0, generate via HTTP,
                            broadcast results back, slice per rank.
          2. vLLM colocate — each GPU runs its own vLLM engine locally.
          3. Transformers  — standard HF generate() or generate_batch().
        """
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        
        if self.use_vllm:
            # Wake colocated vLLM if it was sleeping to free memory
            if self.vllm_mode == "colocate" and self.args.vllm_enable_sleep_mode:
                torch.cuda.empty_cache()
                self.llm.wake_up(tags=["weights"])
                self.llm.collective_rpc("reload_weights")

            # Push the latest merged weights to the vLLM server (once per step)
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            if is_conversational({"prompt": prompts[0]}):
                prompts = [prepare_multimodal_messages_vllm(prompt) for prompt in prompts]

            # vLLM requires tool_call arguments to be JSON strings, not dicts
            for prompt in prompts:
                if is_conversational({"prompt": prompt}):
                    for message in prompt:
                        if "tool_calls" in message:
                            for call in message["tool_calls"]:
                                args = call["function"]["arguments"]
                                if isinstance(args, dict):
                                    call["function"]["arguments"] = json.dumps(args)

            # ── Path 1: vLLM server mode (DDP collective) ──
            # Prompts arrive duplicated num_generations times.  We deduplicate,
            # generate n=num_generations completions per unique prompt on rank 0,
            # then broadcast and slice results back to each rank.
            if self.vllm_mode == "server":
                all_prompts = gather_object(prompts)
                num_generations = self.num_generations if mode == "train" else self.num_generations_eval

                if self.accelerator.is_main_process:
                    # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                    # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                    # prompt individually.
                    ordered_set_of_prompts = all_prompts[::num_generations]

                    effective_max_tokens = max_tokens_override if max_tokens_override is not None else self.max_completion_length
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
                                # ==============================================================
                                # FIX STARTS HERE: Bypass vllm_client.chat() tool limitation
                                # ==============================================================
                                # We manually render the conversational messages (and tool schemas) 
                                # into a raw string using the tokenizer, then use .generate() instead of .chat()
                                formatted_prompts = [
                                    self.processing_class.apply_chat_template(
                                        conversation=conv,
                                        tools=self.tools,
                                        chat_template=self.chat_template,
                                        add_generation_prompt=True,
                                        tokenize=False,
                                        **(self.chat_template_kwargs if self.chat_template_kwargs else {})
                                    ) for conv in ordered_set_of_prompts
                                ]
                                output = self.vllm_client.generate(prompts=formatted_prompts, **sampling_params)
                                # ==============================================================
                            else:
                                output = self.vllm_client.generate(prompts=ordered_set_of_prompts, **sampling_params)
                        # Extract required fields and collect any extra fields for reward functions
                        required_keys = {"prompt_ids", "completion_ids", "logprobs"}
                        extra_fields = {k: v for k, v in output.items() if k not in required_keys}
                        payload = (output["prompt_ids"], output["completion_ids"], output["logprobs"], extra_fields)
                else:
                    payload = None

                # Broadcast the completions from the main process to all processes, ensuring each process receives its corresponding slice.
                obj_list = [payload]
                broadcast_object_list(obj_list, from_process=0)
                all_prompt_ids, all_completion_ids, all_logprobs, all_extra_fields = obj_list[0]

                # At this point, we only get 1 copy of each prompt, so we need to repeat them num_generations times
                all_prompt_ids = [ids for ids in all_prompt_ids for _ in range(num_generations)]

                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                prompt_ids = all_prompt_ids[process_slice]
                completion_ids = all_completion_ids[process_slice]
                logprobs = all_logprobs[process_slice]

                # Slice extra fields dict-of-lists per process (extra fields are per-completion, like completion_ids)
                extra_fields = {}
                for key, values in all_extra_fields.items():
                    if isinstance(values, list):
                        extra_fields[key] = values[process_slice]
                    else:
                        extra_fields[key] = values

            # ── Path 2: vLLM colocate mode (local per GPU) ──
            # Each GPU has its own vLLM engine and generates locally with n=1.
            # No cross-rank communication needed for generation itself.
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

                    effective_max_tokens = max_tokens_override if max_tokens_override is not None else self.max_completion_length
                    generation_kwargs = {
                        "n": 1,  # vLLM on each GPU generates only 1 in colocate mode
                        "repetition_penalty": self.repetition_penalty,
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "top_k": -1 if self.top_k is None else self.top_k,
                        "min_p": 0.0 if self.min_p is None else self.min_p,
                        "max_tokens": effective_max_tokens,
                        "guided_decoding": guided_decoding,
                        "logprobs": 0,  # enable returning log probabilities; 0 means for the sampled tokens only
                    }
                    if self.args.generation_kwargs is not None:
                        generation_kwargs.update(self.args.generation_kwargs)
                    sampling_params = SamplingParams(**generation_kwargs)

                    if self.vllm_tensor_parallel_size > 1:
                        # Gather prompts from all ranks in the TP group and flatten.
                        # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                        orig_size = len(prompts)
                        gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                        torch.distributed.all_gather_object(gathered_prompts, prompts, group=self.tp_group)
                        all_prompts = [p for sublist in gathered_prompts for p in sublist]
                    else:
                        all_prompts = prompts

                    if self.args.vllm_enable_sleep_mode:
                        self.llm.wake_up(tags=["kv_cache"])

                    with profiling_context(self, "vLLM.generate"):
                        if is_conversational({"prompt": prompts[0]}):
                            # ==============================================================
                            # FIX STARTS HERE: Bulletproof colocate mode against strict tools
                            # ==============================================================
                            formatted_prompts = [
                                self.processing_class.apply_chat_template(
                                    conversation=conv,
                                    tools=self.tools,
                                    chat_template=self.chat_template,
                                    add_generation_prompt=True,
                                    tokenize=False,
                                    **(self.chat_template_kwargs if self.chat_template_kwargs else {})
                                ) for conv in all_prompts
                            ]
                            all_outputs = self.llm.generate(
                                formatted_prompts, sampling_params=sampling_params, use_tqdm=False
                            )
                            # ==============================================================
                        else:
                            all_outputs = self.llm.generate(
                                all_prompts, sampling_params=sampling_params, use_tqdm=False
                            )

                    all_prompt_ids = [output.prompt_token_ids for output in all_outputs]
                    all_completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]
                    all_logprobs = [
                        [next(iter(lp.values())).logprob for lp in output.logprobs]
                        for outputs in all_outputs
                        for output in outputs.outputs
                    ]

                    if self.vllm_tensor_parallel_size > 1:
                        # Slice completions for this rank within its TP group.
                        # Each rank generates all outputs — we keep only our share.
                        local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                        tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                        prompt_ids = all_prompt_ids[tp_slice]
                        completion_ids = all_completion_ids[tp_slice]
                        logprobs = all_logprobs[tp_slice]
                    else:
                        prompt_ids = all_prompt_ids
                        completion_ids = all_completion_ids
                        logprobs = all_logprobs

                    extra_fields = {}  # No extra fields for colocate mode

                    if self.args.vllm_enable_sleep_mode:
                        self.llm.sleep(level=2)

        # ── Path 3a: Transformers paged / continuous batching ──
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
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                # Cast to the appropriate dtype based on training configuration
                if self.args.bf16:
                    unwrapped_model.to(torch.bfloat16)
                elif self.args.fp16:
                    unwrapped_model.to(torch.float16)
                if self.args.cast_lm_head_to_fp32:
                    unwrapped_model.lm_head.to(torch.float32)
                with torch.inference_mode():
                    # Continuous batching API expects 'inputs' arg only
                    all_outputs = unwrapped_model.generate_batch(
                        processor_outputs["input_ids"], generation_config=self.generation_config, progress_bar=False
                    )
                    unwrapped_model.train()  # restore training mode, as generate_batch forces eval mode
            completion_ids = [output.generated_tokens for output in all_outputs.values()]
            prompt_ids = processor_outputs["input_ids"]
            logprobs = None  # not used in this case
            extra_fields = {}  # No extra fields for paged mode

        # ── Path 3b: Transformers standard generate() ──
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
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                prompt_completion_ids = unwrapped_model.generate(
                    **generate_inputs, generation_config=self.generation_config, disable_compile=True
                )
            # Compute prompt length and extract completion ids
            prompt_ids, prompt_mask = generate_inputs["input_ids"], generate_inputs["attention_mask"]
            prompt_length = prompt_ids.size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]

            # Mask everything after the first EOS token
            is_eos = completion_ids == self.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
            prompt_ids = [p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool(), strict=True)]
            completion_ids = [c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool(), strict=True)]
            logprobs = None  # not used in this case
            extra_fields = {}  # No extra fields for non-rollout_func paths

        return prompt_ids, completion_ids, logprobs, extra_fields

from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────
# Utility: load a checkpoint saved by DualAdapterGRPOTrainer
# ──────────────────────────────────────────────────────────────────────────

def load_dual_adapter_model(
    adapter: Path,
    device_map: str = "auto",
) -> PeftModel:
    """
    Load a model with dual adapters (SFT + policy).
    
    Parameters
    ----------
    adapter : str
        Path to the SFT/GRPO adapter checkpoint.
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
    with open(adapter / "adapter_config.json", "r") as f:
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
    model = PeftModel.from_pretrained(base_model, adapter.as_posix(), adapter_name=REFERENCE_ADAPTER_NAME)
    
    # Create a mapping to toggle between POLICY and REFERENCE adapter names.
    toggle_map = {POLICY_ADAPTER_NAME: REFERENCE_ADAPTER_NAME, REFERENCE_ADAPTER_NAME: POLICY_ADAPTER_NAME}
    # Get the adapter's name from the stem of the adapter path.
    adapter_name = adapter.stem
    # If the current adapter is recognized (either POLICY or REFERENCE), toggle and load the other adapter.
    if adapter_name in toggle_map:
        toggled_adapter_name = toggle_map[adapter_name]
        # Load the adapter with the toggled name from its parent directory.
        model.load_adapter(adapter.parent / toggled_adapter_name, adapter_name=toggled_adapter_name)
    
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
