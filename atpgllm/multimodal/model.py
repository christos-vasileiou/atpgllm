"""Graph-conditioned causal LM with soft-prefix SFT and rollout primitives."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from atpgllm.graph.models_stage1 import GraphQFormer, NetlistGraphEncoder


class GraphConditionedCausalLM(nn.Module):
    """Inject DAG/Q-Former outputs into the text tokens of a causal LM.

    ``prefix_at="start"`` puts the graph tokens before the whole chat;
    ``"placeholder"`` puts them where the prompt names the netlist (each
    row's ``graph_at``: the index of the first token after the
    ``<GRAPH_CONTEXT>`` placeholder, see :func:`atpgllm.multimodal.data.graph_position`).
    """

    PREFIX_POSITIONS = ("start", "placeholder")

    # Trained tokens per chunk of the SFT loss (2,048 x 100k float32 logits
    # with their gradient is about 1.6 GiB).
    LOSS_CHUNK = 2048

    def __init__(
        self,
        graph_encoder: NetlistGraphEncoder,
        q_former: GraphQFormer,
        llm: nn.Module,
        *,
        prefix_at: str = "start",
        freeze_graph_batchnorm: bool = False,
    ) -> None:
        super().__init__()
        if prefix_at not in self.PREFIX_POSITIONS:
            raise ValueError(f"prefix_at must be one of {self.PREFIX_POSITIONS}")
        self.prefix_at = prefix_at
        # Keep the encoder's BatchNorm on its running statistics in train
        # mode: micro-batches hold 1-8 graphs, while evaluation and rollouts
        # use the running statistics.
        self.freeze_graph_batchnorm = freeze_graph_batchnorm
        self.graph_encoder = graph_encoder
        self.q_former = q_former
        self.llm = llm
        self.num_queries = q_former.num_queries
        self.llm_embed_dim = llm.get_input_embeddings().weight.shape[1]
        self.graph_to_llm = nn.Sequential(
            nn.Linear(q_former.hidden_dim, self.llm_embed_dim),
            nn.GELU(),
            nn.Linear(self.llm_embed_dim, self.llm_embed_dim),
            nn.LayerNorm(self.llm_embed_dim),
        )
        self.architecture_config = {
            "graph_encoder": dict(graph_encoder.config),
            "qformer_hidden_dim": q_former.hidden_dim,
            "num_queries": q_former.num_queries,
            "llm_embed_dim": self.llm_embed_dim,
        }
        # Only when not the default, so earlier checkpoints still match.
        if prefix_at != "start":
            self.architecture_config["prefix_at"] = prefix_at

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.freeze_graph_batchnorm:
            for module in self.graph_encoder.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def set_graph_policy(self, policy: str) -> None:
        """Apply an explicit graph-stack freeze policy; projector stays trainable."""
        if policy not in {"frozen", "qformer", "last_layer", "full"}:
            raise ValueError(
                "graph policy must be frozen, qformer, last_layer, or full"
            )
        for parameter in self.graph_encoder.parameters():
            parameter.requires_grad = policy == "full"
        for parameter in self.q_former.parameters():
            parameter.requires_grad = policy in {"qformer", "last_layer", "full"}
        if policy == "last_layer":
            for parameter in self.graph_encoder.dag_gin.layers[-1].parameters():
                parameter.requires_grad = True
            if self.graph_encoder.dag_gin.jk_proj is not None:
                for parameter in self.graph_encoder.dag_gin.jk_proj.parameters():
                    parameter.requires_grad = True
        for parameter in self.graph_to_llm.parameters():
            parameter.requires_grad = True

    def graph_soft_prompt(self, graph) -> Tensor:
        encoded = self.graph_encoder(graph)
        queries = self.q_former(encoded["node_embs"], graph.batch)
        return self.graph_to_llm(queries)

    def _graph_at(self, graph_at, rows: int, device) -> Optional[Tensor]:
        """Each row's insertion index, or None to put the graph first."""
        if getattr(self, "prefix_at", "start") == "start":
            return None
        if graph_at is None:
            raise ValueError("prefix_at='placeholder' needs each row's graph_at.")
        graph_at = torch.as_tensor(graph_at, device=device).reshape(-1)
        if graph_at.numel() == 1 and rows > 1:
            graph_at = graph_at.expand(rows)
        return graph_at

    @staticmethod
    def _insert(sequence: Tensor, inserted: Tensor, graph_at: Optional[Tensor]) -> Tensor:
        """``inserted`` (``[B, Q, ...]``) placed in each row of ``sequence``
        (``[B, L, ...]``) before index ``graph_at[row]`` (first if None)."""
        if graph_at is None:
            return torch.cat([inserted, sequence], dim=1)
        return torch.stack([
            torch.cat([sequence[row, :at], inserted[row], sequence[row, at:]], dim=0)
            for row, at in enumerate(graph_at.tolist())
        ])

    def _prefix_and_token_embeddings(
        self,
        graph,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        prefix: Optional[Tensor] = None,
        graph_at=None,
    ) -> tuple[Tensor, Tensor]:
        token_embeddings = self.llm.get_input_embeddings()(input_ids)
        prefix = self.graph_soft_prompt(graph) if prefix is None else prefix
        prefix = prefix.to(
            device=token_embeddings.device,
            dtype=token_embeddings.dtype,
        )
        if prefix.size(0) != token_embeddings.size(0):
            raise ValueError(
                f"Graph batch ({prefix.size(0)}) and text batch "
                f"({token_embeddings.size(0)}) differ."
            )
        prefix_mask = torch.ones(
            prefix.shape[:2],
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        at = self._graph_at(graph_at, input_ids.size(0), input_ids.device)
        return (
            self._insert(token_embeddings, prefix, at),
            self._insert(attention_mask, prefix_mask, at),
        )

    def forward(
        self,
        graph,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Optional[Tensor] = None,
        *,
        prefix: Optional[Tensor] = None,
        graph_at=None,
        **kwargs,
    ):
        inputs_embeds, full_mask = self._prefix_and_token_embeddings(
            graph,
            input_ids,
            attention_mask,
            prefix=prefix,
            graph_at=graph_at,
        )
        # Without padding the mask adds nothing to causal attention, and
        # dropping it lets SDPA use a fused causal kernel instead of
        # materializing a [seq, seq] mask (out of memory at 8k tokens).
        if bool(full_mask.all()):
            full_mask = None
        if labels is None:
            return self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=full_mask,
                **kwargs,
            )
        # Logits only for each row's trained tokens (position t predicts token
        # t + 1). Full-vocabulary logits of every position, their float32 copy
        # and its gradient are the largest tensors of a long conversation,
        # most of which is untrained prompt and tool output. The head is
        # applied to the decoder's last hidden states as the causal LM does
        # (Granite divides by logits_scaling).
        targets = self._insert(
            labels,
            torch.full(
                (labels.size(0), self.num_queries),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            ),
            self._graph_at(graph_at, labels.size(0), labels.device),
        )[:, 1:]
        trained = targets != -100
        base = self.llm.get_base_model() if hasattr(self.llm, "get_base_model") else self.llm
        hidden = base.get_decoder()(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            **kwargs,
        ).last_hidden_state[:, :-1]
        head = base.get_output_embeddings()
        scale = getattr(base.config, "logits_scaling", 1.0)

        def chunk_losses(states: Tensor, chunk_targets: Tensor) -> Tensor:
            logits = head(states).float() / scale
            return nn.functional.cross_entropy(logits, chunk_targets, reduction="none")

        # In chunks whose logits are recomputed in the backward pass, so
        # memory does not grow with the number of trained tokens.
        states, kept = hidden[trained], targets[trained]
        token_losses = torch.cat([
            checkpoint(chunk_losses, states[i:i + self.LOSS_CHUNK],
                       kept[i:i + self.LOSS_CHUNK], use_reentrant=False)
            if torch.is_grad_enabled() else
            chunk_losses(states[i:i + self.LOSS_CHUNK], kept[i:i + self.LOSS_CHUNK])
            for i in range(0, states.size(0), self.LOSS_CHUNK)
        ])
        # Mean over examples of each example's token mean: with one example,
        # the causal LM's own loss; with more, padding and length do not
        # change an example's weight.
        rows = trained.nonzero(as_tuple=True)[0]
        sums = torch.zeros(
            labels.size(0), dtype=token_losses.dtype, device=token_losses.device
        ).index_add_(0, rows, token_losses)
        loss = (sums / trained.sum(dim=1).clamp(min=1)).mean()
        return SimpleNamespace(loss=loss)

    def completion_log_probs(
        self,
        graph,
        prompt_ids: Tensor,
        prompt_mask: Tensor,
        completion_ids: Tensor,
        completion_mask: Tensor,
        *,
        prefix: Optional[Tensor] = None,
        graph_at=None,
    ) -> Tensor:
        """Return per-token completion log probabilities ``[B, L]``. The
        graph sits inside the prompt either way, so the completion starts
        ``num_queries + prompt_len`` positions in."""
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        output = self(
            graph,
            input_ids,
            attention_mask,
            prefix=prefix,
            graph_at=graph_at,
            use_cache=False,
        )
        prompt_len = prompt_ids.size(1)
        completion_len = completion_ids.size(1)
        start = self.num_queries + prompt_len - 1
        logits = output.logits[:, start:start + completion_len]
        log_probs = logits.log_softmax(dim=-1)
        return log_probs.gather(
            dim=-1,
            index=completion_ids.unsqueeze(-1),
        ).squeeze(-1)

    @torch.no_grad()
    def generate(self, graph, input_ids: Tensor, attention_mask: Tensor, *,
                 graph_at=None, **kwargs) -> Tensor:
        """``llm.generate`` after the graph prefix, with a KV cache; returns
        only the new tokens."""
        inputs_embeds, full_mask = self._prefix_and_token_embeddings(
            graph, input_ids, attention_mask, graph_at=graph_at
        )
        return self.llm.generate(
            inputs_embeds=inputs_embeds, attention_mask=full_mask, **kwargs
        )

    @torch.no_grad()
    def generate_turns(
        self,
        prefix: Tensor,
        contexts: list[list[int]],
        *,
        pad_token_id: int,
        graph_at: int = 0,
        **kwargs,
    ) -> list[list[int]]:
        """One batched generation over several token contexts that share one
        graph prefix (``[1, Q, D]``) and one prompt, so the graph goes in at
        the same ``graph_at`` of every context (with ``prefix_at="start"``,
        first). Each row is left-padded, so its positions run contiguously;
        returns the new tokens of each row, unpadded."""
        if getattr(self, "prefix_at", "start") == "start":
            graph_at = 0
        embed = self.llm.get_input_embeddings()
        width = max(len(ids) for ids in contexts)
        rows, masks = [], []
        prefix = prefix.to(device=embed.weight.device, dtype=embed.weight.dtype)
        for ids in contexts:
            pad = width - len(ids)
            tokens = torch.tensor(
                [pad_token_id] * pad + list(ids), device=embed.weight.device
            )
            embeddings = embed(tokens)
            at = pad + graph_at
            rows.append(torch.cat([embeddings[:at], prefix[0], embeddings[at:]], dim=0))
            masks.append(torch.tensor(
                [0] * pad + [1] * (prefix.size(1) + len(ids)), device=embed.weight.device
            ))
        output = self.llm.generate(
            inputs_embeds=torch.stack(rows),
            attention_mask=torch.stack(masks),
            pad_token_id=pad_token_id,
            use_cache=True,
            **kwargs,
        )
        # Finished rows are padded after their stop token. Cut there rather
        # than drop pad ids: Granite's pad token is its stop token <|im_end|>.
        stops = kwargs.get("eos_token_id")
        stops = set(stops if isinstance(stops, (list, tuple)) else [stops]) - {None}
        turns = []
        for row in output.tolist():
            end = next((i + 1 for i, token in enumerate(row) if token in stops), None)
            if end is None:
                while row and row[-1] == pad_token_id:
                    row = row[:-1]
                end = len(row)
            turns.append(row[:end])
        return turns

    @torch.no_grad()
    def sample_group(
        self,
        graph,
        prompt_ids: Tensor,
        prompt_mask: Tensor,
        *,
        num_generations: int,
        max_new_tokens: int,
        eos_token_id: Optional[int],
        pad_token_id: int,
        temperature: float = 1.0,
        graph_at=None,
    ) -> tuple[Tensor, Tensor]:
        """Sample a GRPO group for one graph/prompt without vLLM."""
        if prompt_ids.size(0) != 1:
            raise ValueError("sample_group currently accepts one prompt at a time.")
        if num_generations < 2:
            raise ValueError("GRPO requires num_generations >= 2.")

        prefix = self.graph_soft_prompt(graph).repeat(num_generations, 1, 1)
        ids = prompt_ids.repeat(num_generations, 1)
        mask = prompt_mask.repeat(num_generations, 1)
        completions = []
        masks = []
        finished = torch.zeros(
            num_generations,
            dtype=torch.bool,
            device=ids.device,
        )

        for _ in range(max_new_tokens):
            output = self(
                graph,
                ids,
                mask,
                prefix=prefix,
                graph_at=graph_at,
                use_cache=False,
            )
            logits = output.logits[:, -1] / max(float(temperature), 1e-5)
            next_ids = torch.multinomial(logits.softmax(dim=-1), num_samples=1)
            next_ids = next_ids.squeeze(1)
            active = ~finished
            next_ids = torch.where(
                active,
                next_ids,
                torch.full_like(next_ids, pad_token_id),
            )
            completions.append(next_ids)
            masks.append(active.to(mask.dtype))
            ids = torch.cat([ids, next_ids.unsqueeze(1)], dim=1)
            mask = torch.cat([mask, active.to(mask.dtype).unsqueeze(1)], dim=1)
            if eos_token_id is not None:
                finished |= active & next_ids.eq(eos_token_id)
                if finished.all():
                    break

        return torch.stack(completions, dim=1), torch.stack(masks, dim=1)
