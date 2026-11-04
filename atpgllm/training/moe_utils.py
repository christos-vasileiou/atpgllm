"""
Mixture-of-experts helpers shared by SFT and GRPO.

* :func:`moe_layout` — read the expert count and top-k from a model config.
* :func:`resolve_moe_lora_targets` — pick LoRA target names from the modules
  the loaded model actually has, so one setting covers OLMoE, Qwen-MoE,
  Mixtral (``w1``/``w2``/``w3``) and hybrid Mamba checkpoints.
* :class:`ExpertLoadCallback` — log how evenly each router spreads tokens,
  to catch expert collapse during fine-tuning.

Expert weights stored as one fused 3-D parameter (GraniteMoE, and many
models under transformers v5) are not ``nn.Linear`` modules, so PEFT
``target_modules`` cannot reach them. :func:`resolve_moe_lora_targets`
reports such parameters instead of silently skipping them.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from transformers import TrainerCallback

# Attention, dense or expert MLP, and Mamba mixer projections, by leaf name.
_ATTENTION_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
_MLP_TARGETS = ("gate_proj", "up_proj", "down_proj", "w1", "w2", "w3")
_MAMBA_TARGETS = ("in_proj", "out_proj")
_ROUTER_TARGETS = ("gate",)

_EXPERT_COUNT_KEYS = ("num_experts", "num_local_experts", "n_routed_experts")


def moe_layout(config) -> tuple[int, int] | None:
    """Return ``(num_experts, experts_per_token)``, or None for a dense model."""
    config = getattr(config, "text_config", None) or config
    for key in _EXPERT_COUNT_KEYS:
        num_experts = getattr(config, key, None)
        if isinstance(num_experts, int) and num_experts > 1:
            top_k = getattr(config, "num_experts_per_tok", None) or 1
            return num_experts, int(top_k)
    return None


def _linear_leaf_names(model: torch.nn.Module) -> set[str]:
    # bitsandbytes Linear4bit subclasses nn.Linear, so NF4 models match too.
    # Under PEFT a targeted layer lives at ``<name>.base_layer``.
    return {
        name.removesuffix(".base_layer").rsplit(".", 1)[-1]
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }


def fused_expert_parameters(model: torch.nn.Module) -> list[str]:
    """Names of 3-D parameters, which hold every expert of a layer at once."""
    return [name for name, param in model.named_parameters() if param.dim() == 3]


def resolve_moe_lora_targets(
    model: torch.nn.Module,
    tune_router: bool = False,
) -> tuple[list[str], list[str]]:
    """
    Choose LoRA ``target_modules`` from the model's own ``nn.Linear`` names.

    Returns ``(targets, fused)``. ``fused`` lists expert parameters that LoRA
    cannot reach through ``target_modules``. The router (``gate``) is added
    only when ``tune_router`` is set.
    """
    present = _linear_leaf_names(model)
    wanted: Iterable[str] = _ATTENTION_TARGETS + _MLP_TARGETS + _MAMBA_TARGETS
    if tune_router:
        wanted = _ROUTER_TARGETS + tuple(wanted)
    targets = [name for name in wanted if name in present]
    if not targets:
        raise ValueError(
            f"No known LoRA target among the model's Linear layers: {sorted(present)}"
        )
    return targets, fused_expert_parameters(model)


def _router_modules(model: torch.nn.Module, num_experts: int):
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        # A tuned router is a PEFT wrapper (not nn.Linear) around ``gate.base_layer``.
        is_router = (
            leaf == "gate"
            or name.endswith(".gate.base_layer")
            or name.endswith("router.layer")
        )
        if is_router and isinstance(module, torch.nn.Linear) and module.out_features == num_experts:
            yield name, module


class ExpertLoadCallback(TrainerCallback):
    """
    Log the share of top-k routing slots each expert receives.

    Hooks every router and counts its top-k choices per layer. At each log
    step it adds, over layers, the mean and minimum normalised routing
    entropy (1.0 means a uniform load), the largest share any single expert
    took, and the fraction of experts that received no token. Counts reset
    after each log. Recomputation under gradient checkpointing counts some
    forwards twice, which leaves the shares unchanged. Routers that are not
    ``nn.Linear`` (for example Nemotron-H's top-k router) are not hooked, and
    ``handles`` is then empty.
    """

    def __init__(self, model: torch.nn.Module, num_experts: int, top_k: int):
        self.num_experts = num_experts
        self.top_k = top_k
        self.counts: dict[str, torch.Tensor] = {}
        self.handles = []
        for name, module in _router_modules(model, num_experts):
            self.counts[name] = torch.zeros(num_experts, dtype=torch.long)
            self.handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name: str):
        def record(_module, _inputs, logits):
            if not torch.is_grad_enabled():
                return  # skip evaluation and generation passes
            chosen = logits.detach().float().topk(self.top_k, dim=-1).indices.flatten()
            self.counts[name] += torch.bincount(chosen, minlength=self.num_experts).cpu()
        return record

    def summary(self) -> dict[str, float]:
        entropies, max_shares, dead = [], [], []
        for counts in self.counts.values():
            total = int(counts.sum())
            if total == 0:
                continue
            share = counts.double() / total
            nonzero = share[share > 0]
            entropies.append(float(-(nonzero * nonzero.log()).sum()) / math.log(self.num_experts))
            max_shares.append(float(share.max()))
            dead.append(float((counts == 0).double().mean()))
        if not entropies:
            return {}
        return {
            "moe/load_entropy_mean": sum(entropies) / len(entropies),
            "moe/load_entropy_min": min(entropies),
            "moe/max_expert_share": max(max_shares),
            "moe/dead_expert_fraction": sum(dead) / len(dead),
        }

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            logs.update(self.summary())
        for counts in self.counts.values():
            counts.zero_()
