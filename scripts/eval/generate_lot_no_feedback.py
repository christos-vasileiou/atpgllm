#!/usr/bin/env python3
"""Frozen, no-feedback HF/PEFT inference for base, SFT, or GRPO policy models.

Use --prepare-only on CPU to freeze the actual formatted prompts and budgets.
Use --prepared with a fresh output directory on an idle GPU to generate them.
The opposite-fault control is scored against the ORIGINAL target fault.
This is a validation pilot unless a genuinely locked test manifest is supplied.
"""
import argparse
import json
import os
from pathlib import Path
import re
import time

from execute_language_of_test import digest, file_hash, load_manifest, raw_netlist, write


def resolve_adapter(path):
    if (path / "policy/adapter_config.json").exists():
        return path / "policy"
    if (path / "combined/policy/adapter_config.json").exists():
        return path / "combined/policy"
    if (path / "adapter_config.json").exists():
        return path
    raise ValueError(f"No adapter configuration in {path}")


def freeze(args, tokenizer):
    source = load_manifest(args.manifest)
    examples = []
    for original in source["examples"]:
        row = dict(original)
        target = row["fault"]
        if not re.fullmatch(r"sa[01] \S+", target):
            raise ValueError("Expected a single named stuck-at target")
        prompt_fault = f"sa{1 - int(target[2])} {target[4:]}" if args.condition == "opposite-fault" else target
        messages = [{"role": "system", "content": "Generate a test for a combinational circuit. No tools are available. "
            "Return complete binary assignments for every primary input and output using exactly these fields:\n"
            'INPUT_VECTOR: "net: bit, ..."\nEXPECTED_OUTPUT: "net: bit, ..."\n'
            "EXPECTED_OUTPUT must describe the fault-free circuit."},
            {"role": "user", "content": f'Target fault: {prompt_fault}\nNetlist:\n```verilog\n{raw_netlist(row)}\n```'}]
        row["prompt"] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        row["prompt_fault"] = prompt_fault
        # Preserve target and source example IDs for paired original/control scoring.
        row["prompt_tokens"] = len(tokenizer.encode(row["prompt"], add_special_tokens=False))
        examples.append(row)
    protocol = {**source["protocol"], "generations": args.n, "seed": args.seed,
        "temperature": args.temperature, "top_p": .95, "max_completion_length": args.max_new_tokens,
        "max_model_len": args.context_length, "fault_sim_backend": "none_during_generation",
        "sampling_method": "independent_no_feedback", "prompt_pipeline": "lot-no-feedback-v1",
        "condition": args.condition, "per_device_batch_size": 1, "world_size": 1,
        "tokenizer_path": str(args.tokenizer.resolve()), "chat_template_sha256": digest(tokenizer.chat_template),
        "tokenizer_vocab_sha256": digest(tokenizer.get_vocab()),
        "source_manifest_sha256": file_hash(args.manifest)}
    return {"version": 1, "protocol": protocol, "examples": examples, "examples_sha256": digest(examples)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--prepared", type=Path, help="Previously frozen no-feedback manifest; budgets come from that file")
    p.add_argument("--tokenizer", type=Path, required=True, help="Same SFT tokenizer for every model")
    p.add_argument("--checkpoint", type=Path, help="SFT checkpoint or GRPO root/policy; omit for base model")
    p.add_argument("--base-model", default="ibm-granite/granite-4.2-8b")
    p.add_argument("--condition", choices=("original", "opposite-fault"), default="original")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--temperature", type=float, default=.6)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--context-length", type=int, default=32768)
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()
    if bool(args.manifest) == bool(args.prepared):
        p.error("Supply exactly one of --manifest and --prepared")
    if args.n < 1 or args.max_new_tokens < 1 or args.context_length < 1 or args.temperature < 0:
        p.error("Invalid sample or token budget")
    if args.n > 1 and args.temperature == 0:
        p.error("Do not manufacture pass@k by repeating deterministic greedy decoding")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=True)
    manifest = load_manifest(args.prepared) if args.prepared else freeze(args, tokenizer)
    protocol = manifest["protocol"]
    if protocol.get("prompt_pipeline") != "lot-no-feedback-v1":
        p.error("Prepared manifest is not a frozen no-feedback experiment")
    if digest(tokenizer.chat_template) != protocol["chat_template_sha256"]:
        p.error("Tokenizer template differs from frozen protocol")
    if digest(tokenizer.get_vocab()) != protocol["tokenizer_vocab_sha256"]:
        p.error("Tokenizer vocabulary differs from frozen protocol")
    n = protocol["generations"]
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "manifest.json", manifest)
    if args.prepare_only:
        print(json.dumps({"status": "prepared_not_generated", "problems": len(manifest["examples"]),
            "slots": n * len(manifest["examples"]), "condition": protocol["condition"],
            "max_prompt_tokens": max(e["prompt_tokens"] for e in manifest["examples"]),
            "context_ineligible": sum(e["prompt_tokens"] + protocol["max_completion_length"] > protocol["max_model_len"] for e in manifest["examples"])}))
        return
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, set_seed
    if not torch.cuda.is_available():
        raise RuntimeError("A working idle CUDA GPU is required; prepared manifest was saved")
    free, total = torch.cuda.mem_get_info()
    if free < 12 * 1024**3:
        raise RuntimeError("Less than 12 GiB GPU memory free; leave the active training job undisturbed")
    adapter = resolve_adapter(args.checkpoint) if args.checkpoint else None
    base_name = json.loads((adapter / "adapter_config.json").read_text())["base_model_name_or_path"] if adapter else args.base_model
    provenance = {"checkpoint": str(adapter.resolve()) if adapter else None, "base_model": base_name,
        "checkpoint_sha256": {p.name: file_hash(p) for p in adapter.glob("adapter*") if p.is_file()} if adapter else {},
        "model_kind": "adapter" if adapter else "base", "precision": "NF4 double quantization; bfloat16 compute",
        "model_eval_mode": True, "tools": False, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(), "torch_version": torch.__version__,
        "script_sha256": file_hash(Path(__file__)), "manifest_sha256": file_hash(args.output / "manifest.json")}
    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(base_name, local_files_only=True, trust_remote_code=True,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16), device_map={"": 0})
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter), adapter_name="evaluated", is_trainable=False)
        model.set_adapter("evaluated")
    model.eval()
    provenance["cold_model_load_seconds"] = time.monotonic() - started
    provenance["base_model_commit"] = getattr(model.config, "_commit_hash", None)
    write(args.output / "generation_provenance.json", provenance)
    records, generated_tokens, prompt_tokens, model_requests = [], 0, 0, 0
    started = time.monotonic()
    with (args.output / "generation_events.jsonl").open("x") as stream:
        for row in manifest["examples"]:
            for slot in range(n):
                seed = int(digest([protocol["seed"], row["_fixed_eval_id"], slot])[:8], 16)
                set_seed(seed)
                record = {"example_id": row["_fixed_eval_id"], "slot": slot, "seed": seed,
                          "completion": "", "components": {}, "terminal_status": "context_exceeded"}
                t0 = time.monotonic()
                if row["prompt_tokens"] + protocol["max_completion_length"] <= protocol["max_model_len"]:
                    try:
                        inputs = tokenizer(row["prompt"], return_tensors="pt", add_special_tokens=False).to("cuda")
                        options = {"max_new_tokens": protocol["max_completion_length"],
                            "do_sample": protocol["temperature"] > 0,
                            "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id}
                        if options["do_sample"]:
                            options.update(temperature=protocol["temperature"], top_p=protocol["top_p"])
                        with torch.inference_mode():
                            model_requests += 1
                            output = model.generate(**inputs, **options)
                        tokens = output[0, inputs["input_ids"].shape[1]:]
                        record["completion"] = tokenizer.decode(tokens, skip_special_tokens=False)
                        eos = model.generation_config.eos_token_id
                        eos = eos if isinstance(eos, list) else [eos]
                        record["terminal_status"] = "stop" if len(tokens) and int(tokens[-1]) in eos else "length"
                        record["generated_tokens"] = len(tokens)
                        generated_tokens += len(tokens)
                        prompt_tokens += row["prompt_tokens"]
                    except Exception as exc:
                        record.update(terminal_status="generation_error", error=str(exc))
                        # Continue every reserved slot; never silently shrink n.
                        torch.cuda.empty_cache()
                record.update(seconds=time.monotonic() - t0, cumulative_seconds=time.monotonic() - started,
                    cumulative_generated_tokens=generated_tokens, cumulative_prompt_tokens=prompt_tokens,
                    cumulative_model_requests=model_requests,
                    simulator_requests=0, candidate_origin="model")
                records.append(record)
                stream.write(json.dumps(record) + "\n")
                stream.flush()
            print(f"Generated {len(records)}/{n * len(manifest['examples'])} slots", flush=True)
    write(args.output / "records.json", {"step": 0, "evaluated_checkpoint": str(adapter.resolve()) if adapter else None,
        "initial_checkpoint": str(adapter.resolve()) if adapter else None, "examples_sha256": manifest["examples_sha256"],
        "protocol": protocol, "records": records, "generation_provenance": provenance})


if __name__ == "__main__":
    main()
