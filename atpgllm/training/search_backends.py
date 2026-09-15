"""Inference adapters with per-request seeds, stop reasons, and token usage."""
from __future__ import annotations

from .search_types import GenerationResult


class VLLMGenerator:
    def __init__(self, llm, tokenizer, lora_request, base_sampling_params):
        from vllm import SamplingParams
        self._SamplingParams = SamplingParams
        self.llm, self.tokenizer, self.lora_request = llm, tokenizer, lora_request
        self._base = base_sampling_params
        self.max_tokens = int(base_sampling_params.max_tokens)
        self.temperature = float(base_sampling_params.temperature)
        self.top_p = float(base_sampling_params.top_p)
        config = getattr(getattr(llm, "llm_engine", None), "model_config", None)
        self.context_limit = int(getattr(config, "max_model_len", min(tokenizer.model_max_length, 32768)))

    def generate_requests(self, requests):
        params = [self._SamplingParams(
            n=1, temperature=float(r.temperature), top_p=float(r.top_p), max_tokens=int(r.max_tokens),
            seed=int(r.seed), stop_token_ids=list(self._base.stop_token_ids or []),
            stop=["</tool_call>"], include_stop_str_in_output=True,
            logprobs=0 if r.logprobs else None,
        ) for r in requests]
        try:
            prompts = [{"prompt_token_ids": self.tokenizer.encode(r.prompt, add_special_tokens=False)} for r in requests]
            outputs = self.llm.generate(prompts, sampling_params=params,
                                       lora_request=self.lora_request, use_tqdm=False)
        except Exception as exc:
            return [GenerationResult("", error=str(exc)) for _ in requests]
        if len(outputs) != len(requests) or any(len(o.outputs) != 1 for o in outputs):
            raise RuntimeError("vLLM output cardinality mismatch")
        results = []
        for output in outputs:
            item = output.outputs[0]
            ids = tuple(item.token_ids)
            logp = getattr(item, "cumulative_logprob", None)
            results.append(GenerationResult(item.text, ids, item.finish_reason or "stop",
                                           logp / len(ids) if logp is not None and ids else None,
                                           len(output.prompt_token_ids)))
        return results


class HFGenerator:
    def __init__(self, model, tokenizer, base_generation_config, micro_batch_size=8):
        self.model, self.tokenizer, self._base = model, tokenizer, base_generation_config
        self.micro_batch_size = micro_batch_size
        self.max_tokens = int(base_generation_config.max_new_tokens)
        self.temperature = float(0.7 if base_generation_config.temperature is None else base_generation_config.temperature)
        self.top_p = float(1.0 if base_generation_config.top_p is None else base_generation_config.top_p)
        self.context_limit = min(int(tokenizer.model_max_length),
                                 int(getattr(model.config, "max_position_embeddings", 32768)))

    def generate_requests(self, requests):
        import torch
        from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList

        tokenizer = self.tokenizer
        stop_window = max(32, len(tokenizer.encode("</tool_call>", add_special_tokens=False)) + 8)
        class ToolStop(StoppingCriteria):
            def __init__(self, prompt_length):
                self.prompt_length = prompt_length
            def __call__(self, input_ids, scores, **kwargs):
                start = max(self.prompt_length, input_ids.shape[1] - stop_window)
                return "</tool_call>" in tokenizer.decode(input_ids[0, start:], skip_special_tokens=True)

        device = next(self.model.parameters()).device
        devices = list(range(torch.cuda.device_count())) if device.type == "cuda" else []
        results = []
        # HF generate has a shared RNG. Scoped single-request execution preserves
        # per-slot seeds without leaking RNG changes into other application code.
        for req in requests:
            try:
                enc = tokenizer(req.prompt, return_tensors="pt", add_special_tokens=False).to(device)
                prompt_len = enc["input_ids"].shape[1]
                cfg = GenerationConfig(
                    max_new_tokens=req.max_tokens, do_sample=req.temperature > 0,
                    temperature=req.temperature if req.temperature > 0 else 1.0,
                    top_p=req.top_p, pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id if self._base.eos_token_id is None else self._base.eos_token_id,
                    return_dict_in_generate=True, output_scores=req.logprobs,
                )
                with torch.random.fork_rng(devices=devices), torch.inference_mode():
                    torch.random.default_generator.manual_seed(req.seed)
                    if devices:
                        torch.cuda.manual_seed_all(req.seed)
                    output = self.model.generate(**enc, generation_config=cfg,
                        stopping_criteria=StoppingCriteriaList([ToolStop(prompt_len)]))
                ids = tuple(output.sequences[0, prompt_len:].tolist())
                text = tokenizer.decode(ids, skip_special_tokens=True)
                eos = cfg.eos_token_id
                eos = eos if isinstance(eos, list) else [eos]
                finished = bool(ids and ids[-1] in eos) or "</tool_call>" in text
                logp = None
                if req.logprobs and output.scores:
                    transitions = self.model.compute_transition_scores(output.sequences, output.scores, normalize_logits=True)
                    logp = float(transitions[0].mean().item())
                results.append(GenerationResult(text, ids, "stop" if finished else "length", logp, prompt_len))
            except Exception as exc:
                results.append(GenerationResult("", error=str(exc)))
        return results


def make_vllm_generator(llm, tokenizer, lora_request, sampling_params):
    return VLLMGenerator(llm, tokenizer, lora_request, sampling_params)


def make_hf_generator(model, tokenizer, generation_config, micro_batch_size=8):
    return HFGenerator(model, tokenizer, generation_config, micro_batch_size)
