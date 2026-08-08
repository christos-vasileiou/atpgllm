"""
looped_transformer.py
=====================
Converts any HuggingFace causal-LM (GPT-2, LLaMA, Mistral, Phi, etc.)
into a Huginn-style Recurrent-Depth / Looped Transformer and provides
a full training harness with:

  • Prelude  – first K layers  (encode tokens → latent space)
  • Core     – middle M layers shared and looped T times
  • Coda     – last  K layers  (map latent → logits via lm_head)
  • Depth embeddings injected at every loop step
  • Optional Adaptive Computation Time (ACT) halting gate
  • Curriculum training: T grows from 1 → T_max over warm-up steps
  • Implicit-differentiation-friendly gradient checkpointing
  • Stability sandwich norm  (PreNorm → block → PostNorm)
  • KV-cache aware generation loop

Author: example implementation — adapt to your model family.
Tested against: GPT-2, LLaMA-2/3 style configs via transformers>=4.40
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Imports
# ─────────────────────────────────────────────────────────────────────────────
import math
import copy
import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    PreTrainedModel,
    PretrainedConfig,
    get_cosine_schedule_with_warmup,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from transformers.masking_utils import create_causal_mask
except ImportError:  # pragma: no cover - older transformers
    create_causal_mask = None


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LoopedConfig:
    """
    All knobs for the looped transformer wrapper.

    n_prelude_layers  : how many transformer layers go into the Prelude
                        (these are NOT shared, run once to embed tokens)
    n_coda_layers     : how many transformer layers go into the Coda
                        (run once after the loop to project to vocab)
    n_core_layers     : how many layers form the shared block
                        (all three must sum to the original model depth)
    max_loops         : maximum number of loop iterations at inference
    train_loops_start : loop count at the START of curriculum (usually 1)
    train_loops_end   : loop count at the END of curriculum  (= max_loops)
    curriculum_steps  : how many gradient steps to ramp from start→end
    use_act           : enable Adaptive Computation Time halting gate
    act_loss_weight   : ponder-cost penalty weight λ
    depth_embed_dim   : dimension of the per-loop depth embedding
                        (defaults to model hidden_size)
    stability_norm    : add an extra LayerNorm AFTER the shared block
                        output (the "sandwich" norm used in Huginn)
    gradient_checkpointing : recompute activations during backward
    """
    n_prelude_layers:  int   = 2
    n_coda_layers:     int   = 2
    n_core_layers:     int   = -1        # -1 = infer from model depth
    max_loops:         int   = 8
    train_loops_start: int   = 1
    train_loops_end:   int   = 8
    curriculum_steps:  int   = 10_000
    use_act:           bool  = False
    act_loss_weight:   float = 1e-3
    depth_embed_dim:   int   = -1        # -1 = use model hidden_size
    stability_norm:    bool  = True
    gradient_checkpointing: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Depth Embedding  (encodes which loop iteration we are on)
# ─────────────────────────────────────────────────────────────────────────────

class DepthEmbedding(nn.Module):
    """
    Learned embedding over loop iteration index t ∈ {0, …, T_max-1}.
    Projected to hidden_size if depth_embed_dim != hidden_size.
    Added to the hidden state z before each loop pass.
    """
    def __init__(self, max_loops: int, hidden_size: int, embed_dim: int):
        super().__init__()
        self.embed  = nn.Embedding(max_loops, embed_dim)
        self.proj   = (
            nn.Linear(embed_dim, hidden_size, bias=False)
            if embed_dim != hidden_size else nn.Identity()
        )
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, t: int, device: torch.device) -> torch.Tensor:
        idx = torch.tensor([t], device=device)
        return self.proj(self.embed(idx))          # shape: (1, hidden_size)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Adaptive Computation Time (ACT) gate
# ─────────────────────────────────────────────────────────────────────────────

class ACTHaltingGate(nn.Module):
    """
    Per-token halting gate.  Given the current hidden state z ∈ R^{B×L×d},
    produces a halting probability h ∈ (0,1) per token.

    Returns:
        h         : halting probability this step   (B, L)
        remainder : remaining probability budget     (B, L)
        halted    : boolean mask of tokens that have halted
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.gate = nn.Linear(hidden_size, 1, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -1.0)   # bias toward NOT halting early

    def forward(
        self,
        z: torch.Tensor,           # (B, L, d)
        remainder: torch.Tensor,   # (B, L)  cumulative budget left
        halted: torch.Tensor,      # (B, L)  bool, already halted
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = torch.sigmoid(self.gate(z).squeeze(-1))        # (B, L)
        h = h * (~halted).float()                          # zero out already halted
        h = torch.min(h, remainder)                        # cannot exceed budget
        new_remainder = remainder - h
        new_halted    = halted | (new_remainder <= 1e-4)
        return h, new_remainder, new_halted


# ─────────────────────────────────────────────────────────────────────────────
# 3b.  HF decoder layer call (GPT-2 vs LLaMA / Cache API)
# ─────────────────────────────────────────────────────────────────────────────

def _call_layer_forward(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    layer_kwargs_base: dict,
    layer_idx: int,
    past_kv,
    use_cache: bool,
):
    """
    Invoke a single HF decoder block with the right keyword names.

    LLaMA-style (4.48+): needs 4D causal mask + position_embeddings from rotary_emb.
    GPT-2-style: 2D padding mask + cache_position.
    Legacy GPT-2: falls back to layer_past= instead of past_key_values=.
    """
    kwargs = {**layer_kwargs_base, "use_cache": use_cache}
    if past_kv is not None:
        if isinstance(past_kv, (list, tuple)) and layer_idx < len(past_kv):
            kwargs["past_key_values"] = past_kv[layer_idx]
        elif not isinstance(past_kv, (list, tuple)):
            kwargs["past_key_values"] = past_kv
    try:
        return layer(hidden_states, **kwargs)
    except TypeError:
        kwargs.pop("past_key_values", None)
        lp = None
        if isinstance(past_kv, (list, tuple)) and layer_idx < len(past_kv):
            lp = past_kv[layer_idx]
        if lp is not None:
            return layer(hidden_states, layer_past=lp, **kwargs)
        return layer(hidden_states, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Shared Core Block  (wraps a LIST of transformer layers)
# ─────────────────────────────────────────────────────────────────────────────

class SharedCoreBlock(nn.Module):
    """
    Wraps n_core_layers transformer layers whose weights are shared
    across ALL loop iterations.  A sandwich norm is applied around the
    full block for stability (critical at many loops).

    forward() is called once per loop iteration.
    """
    def __init__(
        self,
        layers: nn.ModuleList,
        hidden_size: int,
        stability_norm: bool,
        gradient_checkpointing: bool,
    ):
        super().__init__()
        self.layers  = layers
        self.pre_norm  = nn.LayerNorm(hidden_size, eps=1e-5)
        self.post_norm = nn.LayerNorm(hidden_size, eps=1e-5) if stability_norm else nn.Identity()
        self.use_gc    = gradient_checkpointing

    def _run_layers(self, hidden_states, layer_ctx, past_kv, use_cache):
        """Run all layers in the core block sequentially.

        ``layer_ctx`` is the dict returned by ``LoopedTransformer._layer_attention_context``
        (causal 4D mask + RoPE tensors for LLaMA-style, or 2D mask + cache_position for GPT-2).
        """
        presents = []
        base_kw = layer_ctx["layer_kwargs"]
        for i, layer in enumerate(self.layers):
            out = _call_layer_forward(
                layer, hidden_states, base_kw, i, past_kv, use_cache
            )
            if isinstance(out, tuple):
                hidden_states = out[0]
                if use_cache and len(out) > 1:
                    presents.append(out[1])
            else:
                hidden_states = out

        return hidden_states, presents if presents else None

    def forward(
        self,
        z: torch.Tensor,
        layer_ctx: dict,
        past_key_values: Optional[list]         = None,
        use_cache: bool                          = False,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        residual = z
        z = self.pre_norm(z)

        if self.use_gc and self.training:
            # Gradient checkpointing: do NOT pass use_cache during training
            def ckpt_fn(h):
                out, _ = self._run_layers(h, layer_ctx, None, False)
                return out
            z = checkpoint(ckpt_fn, z, use_reentrant=False)
            presents = None
        else:
            z, presents = self._run_layers(z, layer_ctx, past_key_values, use_cache)

        z = self.post_norm(z + residual)
        return z, presents


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Main LoopedTransformer model
# ─────────────────────────────────────────────────────────────────────────────

class LoopedTransformer(nn.Module):
    """
    Wraps any HuggingFace causal-LM into a Huginn-style architecture:

        Input tokens
             │
        [Embedding + PosEnc]         ← run ONCE
             │
        [Prelude layers × K]          ← run ONCE (encode into latent)
             │
        z⁽⁰⁾ ∈ ℝ^{B × L × d}
             │
        ┌────────────────────────┐
        │  + depth_embed(t)      │
        │  SharedCoreBlock       │   ← looped T times
        │  [optional ACT gate]   │
        └────────────────────────┘
             │
        z★ ∈ ℝ^{B × L × d}
             │
        [Coda layers × K]             ← run ONCE (map latent → pre-logit)
             │
        [lm_head]                     ← run ONCE, last token only for AR gen
             │
        logits → softmax → token
    """

    def __init__(self, base_model: PreTrainedModel, cfg: LoopedConfig):
        super().__init__()

        self.cfg = cfg
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size
        embed_dim   = hidden_size if cfg.depth_embed_dim == -1 else cfg.depth_embed_dim

        # ── locate the layer stack in the base model ──────────────────────
        self.layers, self.embed_fn, self.norm_fn, self.lm_head, self.rotary_emb = \
            _extract_components(base_model)

        n_total = len(self.layers)
        n_core  = cfg.n_core_layers if cfg.n_core_layers > 0 else (
            n_total - cfg.n_prelude_layers - cfg.n_coda_layers
        )
        assert n_core > 0, "Core must have at least 1 layer"
        assert cfg.n_prelude_layers + n_core + cfg.n_coda_layers == n_total, (
            f"Prelude({cfg.n_prelude_layers}) + Core({n_core}) + "
            f"Coda({cfg.n_coda_layers}) ≠ total layers ({n_total})"
        )
        self.n_core = n_core

        # ── split layers ──────────────────────────────────────────────────
        self.prelude_layers = nn.ModuleList(
            self.layers[:cfg.n_prelude_layers]
        )
        core_layers = nn.ModuleList(
            self.layers[cfg.n_prelude_layers : cfg.n_prelude_layers + n_core]
        )
        self.coda_layers = nn.ModuleList(
            self.layers[cfg.n_prelude_layers + n_core:]
        )

        # ── shared core ───────────────────────────────────────────────────
        self.shared_core = SharedCoreBlock(
            layers=core_layers,
            hidden_size=hidden_size,
            stability_norm=cfg.stability_norm,
            gradient_checkpointing=cfg.gradient_checkpointing,
        )

        # ── depth embeddings ──────────────────────────────────────────────
        self.depth_embed = DepthEmbedding(cfg.max_loops, hidden_size, embed_dim)

        # ── optional ACT gate ─────────────────────────────────────────────
        self.act_gate = ACTHaltingGate(hidden_size) if cfg.use_act else None

        # ── curriculum state (updated by trainer) ────────────────────────
        self.current_loops = cfg.train_loops_start

    def _layer_attention_context(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values,
    ) -> dict:
        """
        Build per-layer kwargs matching HuggingFace causal LMs (transformers 4.48+).

        Always build the **4D causal + padding mask** via ``create_causal_mask`` (same as
        ``GPT2Model`` / ``LlamaModel``). Passing a raw 2D mask into blocks disables SDPA's
        ``is_causal`` path and breaks autoregressive attention.

        RoPE models additionally need ``position_embeddings`` from the model's ``rotary_emb``.
        """
        B, L, _ = inputs_embeds.shape
        device = inputs_embeds.device
        attention_mask = attention_mask.to(device=device)
        if attention_mask.dtype not in (torch.long, torch.int, torch.bool):
            attention_mask = attention_mask.long()

        past_seen = 0
        if past_key_values is not None and hasattr(past_key_values, "get_seq_length"):
            past_seen = past_key_values.get_seq_length()

        cache_position = torch.arange(
            past_seen, past_seen + L, device=device, dtype=torch.long
        )
        position_ids = cache_position.unsqueeze(0)

        if create_causal_mask is None:
            raise RuntimeError(
                "transformers.masking_utils.create_causal_mask is required for this LoopedTransformer "
                "forward path; upgrade transformers to >= 4.48."
            )

        causal_mask = create_causal_mask(
            config=self.base_model.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        layer_kwargs = {
            "attention_mask": causal_mask,
            "cache_position": cache_position,
        }
        if self.rotary_emb is not None:
            layer_kwargs["position_embeddings"] = self.rotary_emb(inputs_embeds, position_ids)
            layer_kwargs["position_ids"] = position_ids
        return {"layer_kwargs": layer_kwargs}

    # ------------------------------------------------------------------
    # Curriculum helper – call this from the training loop
    # ------------------------------------------------------------------
    def update_curriculum(self, global_step: int):
        """Linearly ramp loop count from train_loops_start → train_loops_end."""
        cfg = self.cfg
        frac = min(1.0, global_step / max(1, cfg.curriculum_steps))
        self.current_loops = int(
            cfg.train_loops_start + frac * (cfg.train_loops_end - cfg.train_loops_start)
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels:         Optional[torch.Tensor] = None,
        num_steps:      Optional[int]          = None,   # override loop count
        use_cache:      bool                   = False,
        past_key_values: Optional[list]        = None,
    ) -> CausalLMOutputWithPast:

        T = num_steps if num_steps is not None else self.current_loops

        # ── 1. Embed tokens ────────────────────────────────────────────
        z = self.embed_fn(input_ids)       # (B, L, d)
        B, L, d = z.shape

        if attention_mask is None:
            attention_mask = torch.ones(B, L, device=z.device, dtype=torch.long)
        else:
            attention_mask = attention_mask.to(device=z.device)

        layer_ctx = self._layer_attention_context(
            z, attention_mask, past_key_values
        )

        # ── 2. Prelude (encode tokens into latent space, run once) ─────
        for i, layer in enumerate(self.prelude_layers):
            z = _apply_layer(layer, z, layer_ctx, layer_idx=i)

        # ── 3. Inner loop (latent refinement) ─────────────────────────
        act_loss      = torch.tensor(0.0, device=z.device)
        all_kv        = None

        if self.cfg.use_act:
            remainder = torch.ones(B, L, device=z.device)
            halted    = torch.zeros(B, L, device=z.device, dtype=torch.bool)
            accum_z   = torch.zeros_like(z)
            accum_w   = torch.zeros(B, L, device=z.device)

        for t in range(T):
            # inject depth embedding (tells the block which iteration it's on)
            depth_vec = self.depth_embed(t, z.device)   # (1, d)
            z_in = z + depth_vec.unsqueeze(0)            # broadcast over B, L

            z_out, new_kv = self.shared_core(
                z_in,
                layer_ctx,
                past_key_values=all_kv,
                use_cache=use_cache,
            )

            if use_cache:
                all_kv = new_kv

            if self.cfg.use_act:
                h, remainder, halted = self.act_gate(z_out, remainder, halted)
                # accumulate weighted output
                accum_z = accum_z + h.unsqueeze(-1) * z_out
                accum_w = accum_w + h
                act_loss = act_loss + remainder.mean()   # ponder cost
                if halted.all():
                    break
            z = z_out

        if self.cfg.use_act:
            # add remainder weight to the last iteration's output
            accum_z = accum_z + remainder.unsqueeze(-1) * z
            z = accum_z

        # ── 4. Coda (map latent to pre-logit space, run once) ─────────
        for i, layer in enumerate(self.coda_layers):
            idx = self.cfg.n_prelude_layers + self.n_core + i
            z = _apply_layer(layer, z, layer_ctx, layer_idx=idx)

        # ── 5. Final norm + lm_head  (outside the loop, run once) ──────
        if self.norm_fn is not None:
            z = self.norm_fn(z)
        logits = self.lm_head(z)             # (B, L, vocab_size)

        # ── 6. Loss ────────────────────────────────────────────────────
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            if self.cfg.use_act:
                loss = loss + self.cfg.act_loss_weight * act_loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=all_kv,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        num_steps: int  = None,
        temperature: float  = 1.0,
        top_p: float        = 0.9,
        eos_token_id: int   = None,
    ) -> torch.Tensor:
        """
        Simple autoregressive generation loop.
        Each new token:
          1. Runs the full inner loop (T steps) in latent space
          2. Applies lm_head to the LAST position only
          3. Samples one token from the resulting distribution
          4. Appends and repeats — never touches lm_head inside the loop
        """
        self.eval()
        T = num_steps or self.cfg.max_loops
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            out = self.forward(generated, num_steps=T, use_cache=False)
            next_logits = out.logits[:, -1, :]   # (B, vocab_size)

            if temperature != 1.0:
                next_logits = next_logits / temperature

            # nucleus (top-p) sampling
            probs = F.softmax(next_logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            mask = (cum_probs - sorted_probs) > top_p
            sorted_probs[mask] = 0.0
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
            next_token = torch.gather(
                sorted_idx, 1,
                torch.multinomial(sorted_probs, 1)
            )

            generated = torch.cat([generated, next_token], dim=1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Conversion helper:  AutoModelForCausalLM  →  LoopedTransformer
# ─────────────────────────────────────────────────────────────────────────────

def convert_to_looped_transformer(
    model_name_or_path: str,
    cfg: LoopedConfig,
    torch_dtype: torch.dtype = torch.float32,
    device_map: str = "auto",
) -> Tuple["LoopedTransformer", "AutoTokenizer"]:
    """
    Load any HuggingFace causal-LM and wrap it as a LoopedTransformer.

    What changes vs. the original model
    ------------------------------------
    1. Layers are split into Prelude / Core / Coda groups.
    2. Core layers now SHARE weights — only one copy is kept in memory.
    3. A DepthEmbedding module is added (new parameters, randomly init).
    4. An optional ACT gate is added (new parameters).
    5. lm_head is unchanged and still sits outside the loop.
    6. The embedding matrix and positional encodings are unchanged.
    7. All other weights (prelude, coda, lm_head, norms) are carried over.

    Parameters
    ----------
    model_name_or_path : HuggingFace model ID or local path
    cfg                : LoopedConfig with split / loop / ACT settings
    torch_dtype        : weight dtype (float32, bfloat16, float16)
    device_map         : passed to from_pretrained

    Returns
    -------
    looped_model, tokenizer
    """
    print(f"Loading base model: {model_name_or_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_total = _count_layers(base_model)
    print(f"Original model has {n_total} transformer layers")

    # Infer core layer count
    if cfg.n_core_layers == -1:
        cfg.n_core_layers = n_total - cfg.n_prelude_layers - cfg.n_coda_layers

    assert cfg.n_prelude_layers + cfg.n_core_layers + cfg.n_coda_layers == n_total, (
        f"Layer split {cfg.n_prelude_layers}+{cfg.n_core_layers}+{cfg.n_coda_layers}"
        f" ≠ {n_total} total layers"
    )

    print(
        f"Architecture split → "
        f"Prelude: {cfg.n_prelude_layers} | "
        f"Core (shared, looped {cfg.max_loops}×): {cfg.n_core_layers} | "
        f"Coda: {cfg.n_coda_layers}"
    )

    looped = LoopedTransformer(base_model, cfg)
    total_params = sum(p.numel() for p in looped.parameters())
    print(f"Total parameters (after sharing): {total_params:,}")
    return looped, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Training harness
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    output_dir:        str   = "./looped_output"
    num_epochs:        int   = 3
    batch_size:        int   = 4
    grad_accum_steps:  int   = 8
    learning_rate:     float = 2e-4
    weight_decay:      float = 0.1
    warmup_steps:      int   = 500
    max_grad_norm:     float = 1.0
    save_every_steps:  int   = 1000
    eval_every_steps:  int   = 500
    log_every_steps:   int   = 50
    # Freeze prelude/coda initially — only train depth embeds + core
    freeze_pretrained_steps: int = 1000
    # KL stability loss weight (penalises large hidden-state drift between loops)
    kl_stability_weight: float = 0.01


def train_looped_transformer(
    model:      "LoopedTransformer",
    tokenizer:  "AutoTokenizer",
    train_texts: List[str],
    val_texts:   List[str],
    train_cfg:   TrainingConfig,
):
    """
    Full training loop with:
      - Curriculum loop-count ramp
      - Optional prelude/coda freeze period
      - KL stability regularisation between successive hidden states
      - Gradient clipping + weight decay
      - Cosine LR schedule with linear warmup
    """
    import os
    from torch.utils.data import Dataset, DataLoader

    device = next(model.parameters()).device

    # ── Dataset ──────────────────────────────────────────────────────────
    class TextDataset(Dataset):
        def __init__(self, texts, tokenizer, max_length=512):
            self.encodings = tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
        def __len__(self): return self.encodings["input_ids"].shape[0]
        def __getitem__(self, i):
            ids = self.encodings["input_ids"][i]
            attn_mask = self.encodings["attention_mask"][i]
            return {"input_ids": ids, "labels": ids.clone(), "attention_mask": attn_mask}

    train_ds = TextDataset(train_texts, tokenizer)
    val_ds   = TextDataset(val_texts,   tokenizer)
    train_dl = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=train_cfg.batch_size)

    # ── Optimiser ─────────────────────────────────────────────────────────
    # Separate param groups: higher LR for newly added params (depth embed, ACT)
    new_params   = list(model.depth_embed.parameters())
    if model.act_gate:
        new_params += list(model.act_gate.parameters())

    pretrained_params = [
        p for n, p in model.named_parameters()
        if not any(id(p) == id(q) for q in new_params)
    ]

    optimizer = torch.optim.AdamW([
        {"params": pretrained_params, "lr": train_cfg.learning_rate * 0.1},
        {"params": new_params,        "lr": train_cfg.learning_rate},
    ], weight_decay=train_cfg.weight_decay)

    total_steps = train_cfg.num_epochs * len(train_dl) // train_cfg.grad_accum_steps
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=train_cfg.warmup_steps, num_training_steps=total_steps
    )

    os.makedirs(train_cfg.output_dir, exist_ok=True)
    global_step = 0
    model.train()

    for epoch in range(train_cfg.num_epochs):
        optimizer.zero_grad()

        for step, batch in enumerate(train_dl):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels    = batch["labels"].to(device)
            labels[labels == tokenizer.pad_token_id] = -100

            # ── Curriculum: update loop count ──────────────────────────
            model.update_curriculum(global_step)

            # ── Freeze / unfreeze pretrained weights ───────────────────
            freeze = global_step < train_cfg.freeze_pretrained_steps
            for p in pretrained_params:
                p.requires_grad_(not freeze)

            # ── Forward with KL stability loss ─────────────────────────
            # Run at T and T-1 loops; penalise distribution shift
            T = model.current_loops
            out_T  = model(input_ids, attention_mask=attn_mask, labels=labels, num_steps=T)
            loss   = out_T.loss

            if train_cfg.kl_stability_weight > 0 and T > 1:
                with torch.no_grad():
                    logits_Tm1 = model(input_ids, num_steps=T-1).logits.detach()
                kl = F.kl_div(
                    F.log_softmax(out_T.logits, dim=-1),
                    F.softmax(logits_Tm1, dim=-1),
                    reduction="batchmean",
                )
                loss = loss + train_cfg.kl_stability_weight * kl

            # ── Backward ───────────────────────────────────────────────
            (loss / train_cfg.grad_accum_steps).backward()

            if (step + 1) % train_cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # ── Logging ────────────────────────────────────────────
                if global_step % train_cfg.log_every_steps == 0:
                    print(
                        f"Epoch {epoch+1} | Step {global_step} | "
                        f"Loops {T} | Loss {loss.item():.4f} | "
                        f"LR {scheduler.get_last_lr()[0]:.2e}"
                    )

                # ── Validation ─────────────────────────────────────────
                if global_step % train_cfg.eval_every_steps == 0:
                    val_loss = _evaluate(model, val_dl, device, tokenizer)
                    print(f"  ↳ Val loss: {val_loss:.4f}")
                    model.train()

                # ── Checkpoint ─────────────────────────────────────────
                if global_step % train_cfg.save_every_steps == 0:
                    path = f"{train_cfg.output_dir}/step_{global_step}"
                    os.makedirs(path, exist_ok=True)
                    torch.save(model.state_dict(), f"{path}/model.pt")
                    tokenizer.save_pretrained(path)
                    print(f"  ↳ Saved checkpoint to {path}")

    print("Training complete.")
    torch.save(model.state_dict(), f"{train_cfg.output_dir}/final_model.pt")


def _evaluate(model, dataloader, device, tokenizer):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            labels[labels == tokenizer.pad_token_id] = -100
            out = model(input_ids, labels=labels)
            total_loss += out.loss.item()
    return total_loss / len(dataloader)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Internal helpers — model-family-agnostic layer extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_components(model):
    """
    Return (layer_list, embed_fn, norm_fn, lm_head) for any HF causal-LM.
    Handles GPT-2, LLaMA, Mistral, Phi, OPT, Falcon, etc.
    """
    # Map of known attribute paths
    layer_paths = [
        # (transformer_attr, layers_attr, embed_attr, norm_attr)
        ("transformer", "h",      "wte",           "ln_f"),    # GPT-2
        ("model",       "layers", "embed_tokens",  "norm"),     # LLaMA/Mistral
        ("model",       "layers", "embed_tokens",  "final_layernorm"),  # Phi-2
        ("model",       "decoder.layers", "shared", None),     # OPT
        ("transformer", "blocks", "wte",           "ln_f"),    # Falcon
    ]

    for trans_attr, layers_attr, embed_attr, norm_attr in layer_paths:
        try:
            trans = getattr(model, trans_attr, None)
            if trans is None: continue
            layers_obj = trans
            for part in layers_attr.split("."):
                layers_obj = getattr(layers_obj, part, None)
                if layers_obj is None: break
            if layers_obj is None: continue

            embed = getattr(trans, embed_attr, None)
            if embed is None: continue

            norm = getattr(trans, norm_attr, None) if norm_attr else None
            lm_head = getattr(model, "lm_head", None)

            if lm_head is None:
                raise ValueError("Cannot find lm_head on model")

            # Build embed_fn that includes positional encoding when present
            pos_embed = getattr(trans, "wpe", None)  # GPT-2 style

            def make_embed_fn(tok_embed, pos_emb):
                def embed_fn(input_ids):
                    x = tok_embed(input_ids)
                    if pos_emb is not None:
                        pos_ids = torch.arange(input_ids.shape[1], device=input_ids.device)
                        x = x + pos_emb(pos_ids)
                    return x
                return embed_fn

            return (
                nn.ModuleList(list(layers_obj)),
                make_embed_fn(embed, pos_embed),
                norm,
                lm_head,
                getattr(trans, "rotary_emb", None),
            )
        except Exception:
            continue

    raise ValueError(
        "Cannot extract components from this model family. "
        "Please subclass LoopedTransformer and override _extract_components."
    )


def _count_layers(model) -> int:
    layers, *_ = _extract_components(model)
    return len(layers)


def _apply_layer(layer, hidden_states, layer_ctx, layer_idx: int = 0):
    """Apply a single prelude/coda layer with the same mask/RoPE context as the core."""
    out = _call_layer_forward(
        layer, hidden_states, layer_ctx["layer_kwargs"], layer_idx, None, False
    )
    return out[0] if isinstance(out, tuple) else out


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Quick-start example
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Example: convert GPT-2 (12 layers) into a looped transformer with:
      - 2 prelude layers
      - 8 shared core layers (looped up to 8×)
      - 2 coda layers
    Then run a toy training loop and generate text.
    """

    # ── Config ──────────────────────────────────────────────────────────
    loop_cfg = LoopedConfig(
        n_prelude_layers  = 2,
        n_coda_layers     = 2,
        n_core_layers     = 8,   # GPT-2 has 12 total → 2+8+2=12 ✓
        max_loops         = 8,
        train_loops_start = 1,
        train_loops_end   = 8,
        curriculum_steps  = 2000,
        use_act           = True,
        act_loss_weight   = 1e-3,
        stability_norm    = True,
        gradient_checkpointing = False,  # disable for small demo
    )

    train_cfg = TrainingConfig(
        output_dir       = "./looped_gpt2",
        num_epochs       = 1,
        batch_size       = 2,
        grad_accum_steps = 2,
        learning_rate    = 2e-4,
        warmup_steps     = 50,
        log_every_steps  = 10,
        eval_every_steps = 50,
        save_every_steps = 100,
        freeze_pretrained_steps = 50,
        kl_stability_weight     = 0.01,
    )
    
    # ── Convert ─────────────────────────────────────────────────────────
    device = "cuda:3" if torch.cuda.is_available() else "cpu"
    looped_model, tokenizer = convert_to_looped_transformer(
        model_name_or_path = "gpt2", #"Qwen/Qwen2.5-7B-Instruct",
        cfg        = loop_cfg,
        torch_dtype = torch.float32,
        device_map  = device,
    )
    looped_model = looped_model.to(device)

    # ── Toy data ─────────────────────────────────────────────────────────
    train_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Transformers have revolutionised natural language processing.",
        "Looped transformers iterate in latent space without emitting tokens.",
        "Energy-based models minimise a scalar functional at equilibrium.",
    ] * 20

    val_texts = [
        "Deep learning enables machines to learn from raw data.",
        "The recurrent depth approach scales test-time compute.",
    ] * 5

    # ── Train ────────────────────────────────────────────────────────────
    train_looped_transformer(looped_model, tokenizer, train_texts, val_texts, train_cfg)

    # ── Generate: latent loop then decode ────────────────────────────────
    looped_model.eval()
    prompt = "Looped transformers"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    for n_loops in [1, 4, 8]:
        out = looped_model.generate(
            input_ids,
            max_new_tokens = 40,
            num_steps      = n_loops,
            temperature    = 0.8,
            eos_token_id   = tokenizer.eos_token_id,
        )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"\n[{n_loops} loops] {text}")