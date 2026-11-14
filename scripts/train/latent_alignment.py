"""Can Granite read the graph model's latent state? A go or no-go test of the latent channel.

Stage C trained the projector, the LoRA adapters and the graph stack at once
on the answer loss. Granite could lower that loss from the ports prompt
alone, so nothing made it read the graph vectors, and at step 40 a wrong
graph or none gave the same answers (11-12). This script isolates the
channel:

- Granite is frozen and has no adapter. Only the graph side learns: the
  projector (``--train projector``), or the projector and the graph stack
  (``--train graph``).
- The task cannot be solved without the graph: write the graph model's own
  hint (``a[3]=1 sel=0``, the inputs its care head marks with its values)
  for a fault, given the port names and the graph vectors. The prompt holds
  no hint. The target is the text the hint arm reads, so a channel that
  passes here carries what the text hint carries.

Every evaluation scores held-out circuits three ways: the real graph, the
graph of another fault of the same circuit (``wrong_fault``), and no graph
(``none``). The channel is open when the real graph gives a clearly lower
loss than the wrong fault's, and the decoded hint matches the real fault's
hint and not the other's. A wrong fault of the same circuit keeps every name
and position, so only fault-specific content can tell the two apart.

    PYTHONPATH=$PWD python scripts/train/latent_alignment.py --fusion anchored --train projector \\
        --alignment runs/bridges_retest_seq/r3s/best.pt --labels ../data/freeset/retest_labels_v5 \\
        --hints runs/bridges_retest_seq/hints --out runs/latent_alignment/anchored_projector
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))

SYSTEM = "You are an ATPG assistant. You name the inputs a test must set to detect a fault."
USER = ('Which inputs must be set, and to what value, to detect the "{fault}" in this circuit?\n'
        "```\n{netlist}\n```")


_ROW_HEAD = re.compile(r'\{"netlist_key": "([0-9a-f]+)", .{0,400}?"fault": "((?:[^"\\]|\\.)*)"')


def hint_rows(hints: Path, split: str, shards: List[int]) -> Dict[tuple, dict]:
    """``(netlist key, fault) -> hint row`` of the named shard files."""
    rows: Dict[tuple, dict] = {}
    for shard in shards:
        path = hints / f"{split}_{shard:03d}.jsonl"
        for line in path.open():
            row = json.loads(line)
            rows[(row["netlist_key"], row["fault"])] = row
    return rows


def load_problems(labels: List[Path], split: str, wanted: Dict[tuple, dict]):
    """Circuits and the problems that have a hint, in file order."""
    circuits, problems = {}, []
    for directory in labels:
        for line in (Path(directory) / f"circuits_{split}.jsonl").open():
            c = json.loads(line)
            circuits[c["netlist_key"]] = c
        for line in (Path(directory) / f"problems_{split}.jsonl").open():
            # Parse only the rows with a hint: the train table is several GB.
            # Every row starts with its netlist key, ids and fault.
            match = _ROW_HEAD.match(line)
            if match is not None and (match.group(1), json.loads(f'"{match.group(2)}"')) not in wanted:
                continue
            p = json.loads(line)
            if (p["netlist_key"], p["fault"]) in wanted:
                problems.append({"netlist_key": p["netlist_key"], "fault": p["fault"],
                                 "example_input_vector": p["example_input_vector"]})
    return circuits, problems


def assignments(text: str) -> set:
    """``{"a=1", "b=0"}`` from a hint text."""
    return {item for item in text.split() if "=" in item}


def f1(predicted: set, target: set) -> float:
    if not predicted and not target:
        return 1.0
    hit = len(predicted & target)
    return 2 * hit / max(1, len(predicted) + len(target))


class Examples:
    """Tokenized (prompt, hint) pairs with their graphs."""

    def __init__(self, tokenizer, gate_funcs, vocab, circuits, hints, *, anchors: bool, max_len: int) -> None:
        from atpgllm.multimodal.data import template_opens_thinking

        self.tokenizer, self.gate_funcs, self.vocab = tokenizer, gate_funcs, vocab
        self.circuits, self.hints, self.anchors, self.max_len = circuits, hints, anchors, max_len
        self.rejected = collections.Counter()
        # Granite 4.2's generation prompt opens a thinking block: close it.
        self.lead = "</think>\n" if template_opens_thinking(tokenizer) else ""

    def record(self, problem: dict, fault: Optional[str] = None) -> dict:
        circuit = self.circuits[problem["netlist_key"]]
        return {"netlist": circuit["netlist"], "fault": fault or problem["fault"],
                "input_vector": problem["example_input_vector"],
                "scan_chain": circuit.get("scan_chain") or ""}

    def target(self, problem: dict) -> str:
        from atpgllm.multimodal.data import hint_text

        return hint_text(self.hints[(problem["netlist_key"], problem["fault"])]["hint"])

    def build(self, problem: dict, graph_fault: Optional[str] = None):
        """One example. ``graph_fault`` swaps in the graph of another fault
        of the same circuit; the prompt and the target stay this problem's."""
        from atpgllm.graph.dataset import record_to_pyg
        from atpgllm.multimodal.data import (ToolSFTExample, _compact_netlist_payload, _template_ids,
                                             graph_position, port_anchors, ports_extra)

        record = self.record(problem)
        graph = record_to_pyg(self.record(problem, graph_fault), self.gate_funcs, self.vocab,
                              rejections=self.rejected)
        if graph is None:
            return None
        _, compact = _compact_netlist_payload(record, "ports", ports_extra(record, graph, port_bits=True))
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": USER.format(fault=record["fault"], netlist=compact)}]
        prompt = _template_ids(self.tokenizer, messages, generation_prompt=True)
        target = self.tokenizer.encode(self.lead + self.target(problem) + self.tokenizer.eos_token,
                                       add_special_tokens=False)
        ids = prompt + target
        if len(ids) > self.max_len:
            self.rejected["too_long"] += 1
            return None
        graph_at = graph_position(self.tokenizer, ids)
        return ToolSFTExample(
            graph=graph, input_ids=ids, labels=[-100] * len(prompt) + target, retry=False, graph_at=graph_at,
            anchors=port_anchors(self.tokenizer, ids, graph_at, graph) if self.anchors else ())


def build_model(args):
    import compare_stage_c_text_sft as cs
    import multimodal_training as mt
    from atpgllm.multimodal import GraphConditionedCausalLM

    llm_args = cs._llm_args(alignment_ckpt=str(args.alignment))
    gate_funcs, vocab, graph_encoder, q_former, _, tokenizer = mt._load_common(llm_args)
    llm = mt._base_llm(llm_args)
    for parameter in llm.parameters():
        parameter.requires_grad = False
    model = GraphConditionedCausalLM(graph_encoder, q_former, llm, prefix_at="placeholder",
                                     freeze_graph_batchnorm=True, fusion=args.fusion)
    model.set_graph_policy("full" if args.train == "graph" else "frozen")
    device = mt._graph_device(model)
    for module in mt._graph_modules(model).values():
        module.to(device)
    return model, tokenizer, gate_funcs, vocab, device


def optimizer_for(model, args):
    import torch

    groups = []
    if model.fusion != "anchored":
        groups.append({"params": list(model.graph_to_llm.parameters()), "lr": args.lr})
    if model.fusion != "prefix":
        groups.append({"params": list(model.port_anchor.proj.parameters()), "lr": args.lr})
        groups.append({"params": [model.port_anchor.alpha], "lr": args.gate_lr})
    if args.train == "graph":
        groups.append({"params": [p for p in model.q_former.parameters() if p.requires_grad], "lr": args.graph_lr})
        groups.append({"params": [p for p in model.graph_encoder.parameters() if p.requires_grad],
                       "lr": args.graph_lr / 2})
    for group in groups:
        for parameter in group["params"]:
            parameter.requires_grad = True
    return torch.optim.AdamW(groups, weight_decay=0.0)


def batch_loss(model, examples, tokenizer, device):
    from atpgllm.multimodal.data import collate_tool_sft

    batch = collate_tool_sft(examples, tokenizer.pad_token_id)
    batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
    return model(**batch, use_cache=False).loss


def evaluate(model, tokenizer, data: Examples, problems, partner, device, *, decode: int = 0) -> Dict[str, Any]:
    """Held-out loss with the real graph, another fault's graph and none;
    with ``decode``, greedy hints for that many problems."""
    import torch
    from torch_geometric.data import Batch
    from types import SimpleNamespace

    from atpgllm.multimodal.data import collate_anchors

    was_training = model.training
    model.eval()
    losses = collections.defaultdict(list)
    decoded = collections.defaultdict(list)
    stop = [t for t in (tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id)
            if isinstance(t, int) and t >= 0]
    with torch.no_grad():
        for index, problem in enumerate(problems):
            other = partner.get(index)
            real = data.build(problem)
            wrong = data.build(problem, graph_fault=other["fault"]) if other else None
            if real is None or wrong is None:
                continue
            circuit = problem["netlist_key"]
            for name, example in (("real", real), ("wrong_fault", wrong)):
                losses[name].append((circuit, float(batch_loss(model, [example], tokenizer, device))))
            model.graph_off = True
            losses["none"].append((circuit, float(batch_loss(model, [real], tokenizer, device))))
            model.graph_off = False
            if len(decoded["real"]) >= decode:
                continue
            own, others = assignments(data.target(problem)), assignments(data.target(other))
            prompt_len = real.labels.index(next(x for x in real.labels if x != -100))
            ids = torch.tensor([real.input_ids[:prompt_len]], device=device)
            for name, example in (("real", real), ("wrong_fault", wrong)):
                anchors = (collate_anchors([SimpleNamespace(graph=example.graph, anchors=example.anchors)])
                           if example.anchors else None)
                output = model.generate(
                    Batch.from_data_list([example.graph]).to(device), ids, torch.ones_like(ids),
                    graph_at=example.graph_at, anchors=anchors, do_sample=False, eos_token_id=stop,
                    max_new_tokens=len(real.input_ids) - prompt_len + 16, pad_token_id=tokenizer.pad_token_id)
                written = assignments(tokenizer.decode(output[0], skip_special_tokens=True))
                decoded[name].append({"own": f1(written, own), "other": f1(written, others),
                                      "exact": written == own})
    if was_training:
        set_mode(model, "graph" if any(p.requires_grad for p in model.q_former.parameters()) else "projector")

    def mean(rows):
        return sum(v for _, v in rows) / max(1, len(rows))

    out: Dict[str, Any] = {"rows": len(losses["real"]), **{f"loss_{k}": mean(v) for k, v in losses.items()}}
    gaps = [(c, w - r) for (c, r), (_, w) in zip(losses["real"], losses["wrong_fault"])]
    by_circuit = collections.defaultdict(list)
    for circuit, gap in gaps:
        by_circuit[circuit].append(gap)
    rng, groups, means = random.Random(0), list(by_circuit.values()), []
    for _ in range(1000 if groups else 0):
        sample = [x for _ in groups for x in rng.choice(groups)]
        means.append(sum(sample) / len(sample))
    means.sort()
    out["gap_wrong_minus_real"] = mean(gaps)
    out["gap_95"] = [means[25], means[974]] if means else None
    for name, rows in decoded.items():
        if not rows:
            continue
        out[f"decode_{name}"] = {key: sum(float(r[key]) for r in rows) / len(rows) for key in ("own", "other", "exact")}
        out[f"decode_{name}"]["rows"] = len(rows)
    if model.fusion != "prefix":
        out["anchor_gate"] = model.anchor_gate()
    return out


def set_mode(model, train: str) -> None:
    """Train mode only for what learns: Granite is frozen, and so is the
    graph stack unless ``--train graph`` (its dropout would add noise)."""
    model.train()
    model.llm.eval()
    if train != "graph":
        model.graph_encoder.eval()
        model.q_former.eval()


def main(argv=None, model_factory=build_model) -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--labels", type=Path, nargs="+", required=True)
    parser.add_argument("--hints", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fusion", choices=("prefix", "anchored", "both"), default="anchored")
    parser.add_argument("--train", choices=("projector", "graph"), default="projector")
    parser.add_argument("--train-shards", type=int, nargs="+", default=[0],
                        help="Hint shard files of the train split to learn from.")
    parser.add_argument("--eval-shards", type=int, nargs="+", default=[0])
    parser.add_argument("--max-eval", type=int, default=400)
    parser.add_argument("--decode", type=int, default=150)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gate-lr", type=float, default=1e-2)
    parser.add_argument("--graph-lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    model, tokenizer, gate_funcs, vocab, device = model_factory(args)
    sets = {}
    for split, shards in (("train", args.train_shards), ("validation", args.eval_shards)):
        hints = hint_rows(args.hints, split, shards)
        circuits, problems = load_problems(args.labels, split, hints)
        sets[split] = (Examples(tokenizer, gate_funcs, vocab, circuits, hints,
                                anchors=args.fusion != "prefix", max_len=args.max_len), problems)
    train_data, train_problems = sets["train"]
    eval_data, eval_problems = sets["validation"]
    rng.shuffle(eval_problems)
    # Another fault of the same circuit whose hint differs: its graph is the wrong one.
    by_circuit = collections.defaultdict(list)
    for index, problem in enumerate(eval_problems):
        by_circuit[problem["netlist_key"]].append(index)
    partner, kept = {}, []
    for index, problem in enumerate(eval_problems):
        choices = [eval_problems[j] for j in by_circuit[problem["netlist_key"]]
                   if j != index and eval_data.target(eval_problems[j]) != eval_data.target(problem)]
        if choices and len(kept) < args.max_eval:
            partner[len(kept)] = rng.choice(choices)
            kept.append(problem)
    eval_problems = kept
    optimizer = optimizer_for(model, args)
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    trainable = sum(p.numel() for group in optimizer.param_groups for p in group["params"])
    log = (args.out / "metrics.jsonl").open("a")
    log.write(json.dumps({"event": "start", "args": {k: str(v) for k, v in vars(args).items()},
                          "train_problems": len(train_problems), "eval_problems": len(eval_problems),
                          "trainable_parameters": trainable}) + "\n")
    log.flush()
    print(f"train {len(train_problems)} problems, eval {len(eval_problems)}, trainable {trainable:,}", flush=True)

    def check(step: int, decode: int = 0) -> None:
        metrics = evaluate(model, tokenizer, eval_data, eval_problems, partner, device, decode=decode)
        log.write(json.dumps({"step": step, "eval": metrics}) + "\n")
        log.flush()
        print(step, json.dumps(metrics), flush=True)

    check(0)
    set_mode(model, args.train)
    order, cursor, started = list(range(len(train_problems))), 0, time.time()
    rng.shuffle(order)
    for step in range(1, args.steps + 1):
        examples = []
        while len(examples) < args.batch_size:
            if cursor >= len(order):
                rng.shuffle(order)
                cursor = 0
            example = train_data.build(train_problems[order[cursor]])
            cursor += 1
            if example is not None:
                examples.append(example)
        factor = min(1.0, step / max(1, args.warmup)) * 0.5 * (1.0 + math.cos(math.pi * step / args.steps))
        for group, base in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base * factor
        loss = batch_loss(model, examples, tokenizer, device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], 1.0)
        optimizer.step()
        if step % 25 == 0 or step == 1:
            log.write(json.dumps({"step": step, "loss": float(loss), "seconds": round(time.time() - started, 1)}) + "\n")
            log.flush()
        if step % args.eval_every == 0 and step != args.steps:
            check(step)
    check(args.steps, decode=args.decode)
    modules = {"graph_to_llm": model.graph_to_llm}
    if model.fusion != "prefix":
        modules["port_anchor"] = model.port_anchor
    if args.train == "graph":
        modules.update(graph_encoder=model.graph_encoder, q_former=model.q_former)
    torch.save({name: module.state_dict() for name, module in modules.items()}, args.out / "graph_side.pt")
    log.close()


if __name__ == "__main__":
    main()
