"""Central conditional search spaces and architecture-locking rules."""

from __future__ import annotations

from typing import Any, Mapping


def _suggest(trial: Any, name: str, spec: Mapping[str, Any]) -> Any:
    kind = spec.get("type")
    if kind == "categorical":
        choices = list(spec["choices"])
        if not choices:
            raise ValueError(f"Search parameter {name!r} has no choices.")
        return trial.suggest_categorical(name, choices)
    if kind == "int":
        return trial.suggest_int(
            name,
            int(spec["low"]),
            int(spec["high"]),
            step=int(spec.get("step", 1)),
            log=bool(spec.get("log", False)),
        )
    if kind == "float":
        kwargs: dict[str, Any] = {"log": bool(spec.get("log", False))}
        if "step" in spec:
            kwargs["step"] = float(spec["step"])
        return trial.suggest_float(
            name,
            float(spec["low"]),
            float(spec["high"]),
            **kwargs,
        )
    raise ValueError(f"Unsupported search parameter type {kind!r} for {name!r}.")


def suggest_graph_pretrain(
    trial: Any,
    search: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    names = (
        "node_dim",
        "gin_hidden",
        "gin_layers",
        "per_attr_dim",
        "attr_mlp_hidden",
        "dropout",
        "lr",
        "weight_decay",
        "grad_clip",
        "backtrack_ratio",
        "discrepancy_ratio",
    )
    values = {name: _suggest(trial, name, search[name]) for name in names}
    raw_weights = {
        "propagation": 1.0,
        "backtrack": float(values.pop("backtrack_ratio")),
        "discrepancy": float(values.pop("discrepancy_ratio")),
    }
    scale = 3.0 / sum(raw_weights.values())
    values["loss_weights"] = {
        name: value * scale for name, value in raw_weights.items()
    }
    return values


def suggest_graph_text_alignment(
    trial: Any,
    search: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    hidden = int(_suggest(trial, "qformer_hidden", search["qformer_hidden"]))
    configured_heads = list(search["qformer_heads"]["choices"])
    compatible_heads = [
        int(head) for head in configured_heads if hidden % int(head) == 0
    ]
    if not compatible_heads:
        raise ValueError(
            f"No configured Q-Former head count divides hidden size {hidden}."
        )
    heads = trial.suggest_categorical(
        f"qformer_heads_h{hidden}",
        compatible_heads,
    )
    values = {
        "qformer_hidden": hidden,
        "qformer_heads": int(heads),
    }
    for name in (
        "qformer_layers",
        "num_queries",
        "cross_attn_every_n",
        "qformer_dropout",
        "ffn_multiplier",
        "proj_dim",
        "graph_policy",
        "lr",
        "weight_decay",
        "grad_clip",
        "gtm_ratio",
        "gtg_ratio",
    ):
        values[name] = _suggest(trial, name, search[name])
    raw_weights = {
        "gtc": 1.0,
        "gtm": float(values.pop("gtm_ratio")),
        "gtg": float(values.pop("gtg_ratio")),
    }
    scale = 3.0 / sum(raw_weights.values())
    values["loss_weights"] = {
        name: value * scale for name, value in raw_weights.items()
    }
    return values


def assert_graph_architecture_locked(
    parent_graph_architecture: Mapping[str, Any],
    candidate_graph_architecture: Mapping[str, Any],
) -> None:
    if dict(parent_graph_architecture) != dict(candidate_graph_architecture):
        raise ValueError(
            "Stage-B graph architecture must exactly match its Stage-A parent; "
            f"parent={dict(parent_graph_architecture)}, "
            f"candidate={dict(candidate_graph_architecture)}."
        )


def downstream_screening_configs(
    stage: str,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Small deterministic Stage-C/D grids; selection remains held-out-eval based."""
    if stage == "multimodal_sft":
        values = [
            {
                "graph_policy": "frozen",
                "lora_r": 16,
                "lora_alpha_ratio": 2,
                "lr": 1e-4,
                "graph_lr": 3e-6,
            },
            {
                "graph_policy": "qformer",
                "lora_r": 16,
                "lora_alpha_ratio": 2,
                "lr": 1e-4,
                "graph_lr": 1e-5,
            },
            {
                "graph_policy": "last_layer",
                "lora_r": 32,
                "lora_alpha_ratio": 2,
                "lr": 2e-4,
                "graph_lr": 3e-6,
            },
        ]
        return values[:max(0, limit)]
    if stage == "multimodal_grpo":
        values = [
            {
                "graph_policy": "frozen",
                "lr": 3e-6,
                "graph_lr": 3e-7,
                "beta": 0.01,
                "temperature": 0.8,
                "clip_epsilon": 0.2,
                "num_generations": 4,
            },
            {
                "graph_policy": "qformer",
                "lr": 3e-6,
                "graph_lr": 1e-6,
                "beta": 0.03,
                "temperature": 1.0,
                "clip_epsilon": 0.2,
                "num_generations": 8,
            },
            {
                "graph_policy": "frozen",
                "lr": 5e-6,
                "graph_lr": 3e-7,
                "beta": 0.03,
                "temperature": 1.0,
                "clip_epsilon": 0.2,
                "num_generations": 8,
            },
        ]
        return values[:max(0, limit)]
    raise ValueError(f"No downstream screening space for stage {stage!r}.")
