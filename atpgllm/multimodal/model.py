"""Graph-conditioned causal LM with soft-prefix SFT and rollout primitives."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from atpgllm.graph.models_stage1 import GraphQFormer, NetlistGraphEncoder


class GraphConditionedCausalLM(nn.Module):
    """Inject DAG/Q-Former outputs before text tokens in a causal LM."""

    def __init__(
        self,
        graph_encoder: NetlistGraphEncoder,
        q_former: GraphQFormer,
        llm: nn.Module,
    ) -> None:
        super().__init__()
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

    def _prefix_and_token_embeddings(
        self,
        graph,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        prefix: Optional[Tensor] = None,
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
        return (
            torch.cat([prefix, token_embeddings], dim=1),
            torch.cat([prefix_mask, attention_mask], dim=1),
        )

    def forward(
        self,
        graph,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Optional[Tensor] = None,
        *,
        prefix: Optional[Tensor] = None,
        **kwargs,
    ):
        inputs_embeds, full_mask = self._prefix_and_token_embeddings(
            graph,
            input_ids,
            attention_mask,
            prefix=prefix,
        )
        full_labels = None
        if labels is not None:
            prefix_labels = torch.full(
                (labels.size(0), self.num_queries),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            full_labels = torch.cat([prefix_labels, labels], dim=1)
        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            labels=full_labels,
            **kwargs,
        )

    def completion_log_probs(
        self,
        graph,
        prompt_ids: Tensor,
        prompt_mask: Tensor,
        completion_ids: Tensor,
        completion_mask: Tensor,
        *,
        prefix: Optional[Tensor] = None,
    ) -> Tensor:
        """Return per-token completion log probabilities ``[B, L]``."""
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        output = self(
            graph,
            input_ids,
            attention_mask,
            prefix=prefix,
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
