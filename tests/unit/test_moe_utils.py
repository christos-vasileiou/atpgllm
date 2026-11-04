"""MoE LoRA target resolution, expert-load logging and the bf16 base path."""

import pytest
import torch
from peft import get_peft_model
from transformers import (
    AutoModelForCausalLM,
    GraniteMoeConfig,
    LlamaConfig,
    MixtralConfig,
    OlmoeConfig,
)

from atpgllm.training.model_utils import base_precision, get_lora_config, prepare_lora_model
from atpgllm.training.moe_utils import (
    ExpertLoadCallback,
    moe_layout,
    resolve_moe_lora_targets,
)

_SMALL = dict(
    hidden_size=32, intermediate_size=16, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
)


def _tiny(config_cls, **extra):
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(config_cls(**_SMALL, **extra))


def _olmoe():
    return _tiny(OlmoeConfig, num_experts=4, num_experts_per_tok=2)


def test_moe_layout_reads_expert_count_and_top_k():
    assert moe_layout(_olmoe().config) == (4, 2)
    assert moe_layout(_tiny(MixtralConfig, num_local_experts=4, num_experts_per_tok=2).config) == (4, 2)
    assert moe_layout(_tiny(LlamaConfig).config) is None


def test_olmoe_targets_exclude_router_unless_tuned():
    targets, fused = resolve_moe_lora_targets(_olmoe())
    assert targets == ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    assert fused == []
    tuned, _ = resolve_moe_lora_targets(_olmoe(), tune_router=True)
    assert tuned[0] == "gate"


def test_mixtral_experts_use_w_names():
    model = _tiny(MixtralConfig, num_local_experts=4, num_experts_per_tok=2)
    targets, _ = resolve_moe_lora_targets(model)
    assert {"w1", "w2", "w3"} <= set(targets)
    assert "gate_proj" not in targets


def test_fused_experts_are_reported():
    model = _tiny(GraniteMoeConfig, num_local_experts=4, num_experts_per_tok=2)
    targets, fused = resolve_moe_lora_targets(model)
    assert targets == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert fused and all("block_sparse_moe" in name for name in fused)


def test_targets_resolve_through_a_peft_wrapper():
    model = get_peft_model(_olmoe(), get_lora_config(r=2, lora_alpha=4))
    targets, _ = resolve_moe_lora_targets(model)
    assert "q_proj" in targets and "down_proj" in targets


def test_expert_load_counts_only_training_forwards():
    model = _olmoe()
    callback = ExpertLoadCallback(model, *moe_layout(model.config))
    assert len(callback.handles) == 2
    ids = torch.randint(0, 64, (2, 8))
    with torch.no_grad():
        model(ids)
    assert all(int(c.sum()) == 0 for c in callback.counts.values())

    model(ids).logits.sum().backward()
    # 16 tokens x top-2 per layer.
    assert all(int(c.sum()) == 32 for c in callback.counts.values())
    logs = {}
    callback.on_log(None, None, None, logs=logs)
    assert 0.0 < logs["moe/load_entropy_min"] <= logs["moe/load_entropy_mean"] <= 1.0
    assert 0.25 <= logs["moe/max_expert_share"] <= 1.0
    assert all(int(c.sum()) == 0 for c in callback.counts.values())


def test_expert_load_follows_a_lora_tuned_router():
    model = _olmoe()
    lora = get_lora_config(r=2, lora_alpha=4, target_modules=["gate", "q_proj"])
    model = get_peft_model(model, lora)
    callback = ExpertLoadCallback(model, 4, 2)
    assert len(callback.handles) == 2


def test_base_precision_env(monkeypatch):
    monkeypatch.delenv("BASE_PRECISION", raising=False)
    assert base_precision() == "nf4"
    monkeypatch.setenv("BASE_PRECISION", "BF16")
    assert base_precision() == "bf16"
    monkeypatch.setenv("BASE_PRECISION", "fp8")
    with pytest.raises(ValueError):
        base_precision()


def test_bf16_base_is_not_upcast_by_lora_preparation():
    model = _olmoe().to(torch.bfloat16)
    peft_model = prepare_lora_model(model, get_lora_config(r=2, lora_alpha=4))
    frozen = [p for p in peft_model.parameters() if not p.requires_grad]
    assert frozen and all(p.dtype == torch.bfloat16 for p in frozen)
    assert peft_model.base_model.model.is_gradient_checkpointing
