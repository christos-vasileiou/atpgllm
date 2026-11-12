"""Train the BRIDGES re-test graph side (plan phases R2 and R3).

R2 (encoder only): ``--query-weight 0 --text-weight 0``. The encoder learns
signal probability plus the existing Stage A heads where labels exist.
R3 (queries): ``--query-weight 1`` adds the fault-seeded role queries with
pointer and value heads and the anti-collapse terms. ``--text-weight 1``
adds GTC, GTM and GTG on the port-only fact text, with ModernBERT as the
text encoder, as in BRIDGES.

Every eval logs the G2 collapse metrics (query cosine within a graph, share
of query variance that changes with the graph) and pointer quality.

Example (CPU smoke)::

    python -m atpgllm.graph.scripts.train_retest --labels ../data/freeset/retest_labels_v4 \
        --train-split validation --eval-split test --max-train 64 --max-eval 32 \
        --steps 4 --batch-size 8 --out runs/retest_smoke --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch

from atpgllm.graph.dataset import _circuit_graph
from atpgllm.graph.fault_context import attach_atpg_context
from atpgllm.graph.gate_features import GateAttributeVocab
from atpgllm.graph.losses_stage1 import GraphTextMatchingHead
from atpgllm.graph.models_stage1 import NetlistGraphEncoder
from atpgllm.graph.netlist_parser import GraphRejected
from atpgllm.graph.retest_labels import ProblemTargets, attach_retest_targets, cop_signal_probability
from atpgllm.graph.retest_qformer import (
    QueryGroundedTextGenerator,
    RoleQueryFormer,
    anti_collapse,
    collapse_metrics,
    soft_pointer_bce,
    weighted_value_bce,
)
from atpgllm.training._paths import DEFAULT_SIM_CONFIG


def load_split(labels_dir, split: str, limit: int, seed: int):
    """Problems and circuits of one split. ``labels_dir`` is one label
    directory or several (v4 and v5 tables together); the problems are
    shuffled as one set."""
    dirs = [labels_dir] if isinstance(labels_dir, (str, os.PathLike)) else list(labels_dir)
    circuits = {}
    problems = []
    for directory in dirs:
        with open(os.path.join(directory, f"circuits_{split}.jsonl"), encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                circuits[c["netlist_key"]] = c
        with open(os.path.join(directory, f"problems_{split}.jsonl"), encoding="utf-8") as f:
            for line in f:
                problems.append(json.loads(line))
    random.Random(seed).shuffle(problems)
    if limit:
        problems = problems[:limit]
    return problems, circuits


class ProblemGraphs:
    """Builds one graph per problem with fault inputs and soft targets."""

    def __init__(self, circuits: Dict[str, dict], gate_funcs, vocab) -> None:
        self.circuits = circuits
        self.gate_funcs = gate_funcs
        self.vocab = vocab
        self._sp: Dict[str, Dict[str, float]] = {}
        self.rejected = 0

    def signal_probability(self, key: str) -> Optional[Dict[str, float]]:
        if key not in self._sp:
            os.environ.setdefault("FAULT_SIM_BACKEND", "fast")
            from atpgllm.training.tools import simulation_model
            try:
                self._sp[key] = cop_signal_probability(simulation_model(self.circuits[key]["netlist"]))
            except Exception:
                self._sp[key] = None
        return self._sp[key]

    def build(self, p: dict):
        circuit = self.circuits[p["netlist_key"]]
        record = {"netlist": circuit["netlist"], "fault": p["fault"], "input_vector": p["example_input_vector"]}
        try:
            parsed, base = _circuit_graph(circuit["netlist"], self.gate_funcs, self.vocab,
                                          circuit.get("scan_chain") or "")
            data = base.clone()
            attach_atpg_context(data, parsed, self.gate_funcs, record)
        except GraphRejected:
            self.rejected += 1
            return None
        t = ProblemTargets(fault=p["fault"], inputs=p["inputs"], outputs=p["outputs"], patterns=p["patterns"],
                           care_freq=p["care_freq"], one_freq=p["one_freq"],
                           path_freq=p["path_freq"], obs_freq=p["obs_freq"])
        attach_retest_targets(data, parsed, t, self.signal_probability(p["netlist_key"]))
        data.fact_text = p["fact_text"]
        return data


class RetestGraphModel(nn.Module):
    def __init__(self, vocab, *, node_dim: int, gin_layers: int, q_hidden: int, slots: int,
                 q_layers: int, text_dim: Optional[int] = None, proj_dim: int = 256) -> None:
        super().__init__()
        self.encoder = NetlistGraphEncoder(vocab, node_dim=node_dim, gin_hidden_dim=node_dim,
                                           gin_num_layers=gin_layers)
        self.sp_head = nn.Sequential(nn.Linear(node_dim, node_dim), nn.GELU(), nn.Linear(node_dim, 1))
        self.q_former = RoleQueryFormer(d_node=node_dim, hidden_dim=q_hidden, slots_per_role=slots,
                                        num_layers=q_layers)
        self.graph_proj = nn.Linear(q_hidden, proj_dim)
        self.text_proj = nn.Linear(text_dim, proj_dim) if text_dim else None

    def forward(self, batch):
        enc = self.encoder(batch)
        queries = self.q_former(enc["node_embs"], batch.batch, batch.fault_feats)
        return enc, queries


def dense(batch, name, dtype=None):
    value, _ = to_dense_batch(getattr(batch, name), batch.batch)
    return value if dtype is None else value.to(dtype)


def compute_losses(model, batch, enc, queries, args, text=None) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    sp_pred = model.sp_head(enc["node_embs"]).squeeze(-1)
    m = batch.sp_mask
    out["sp"] = F.mse_loss(torch.sigmoid(sp_pred[m]), batch.sp_target[m]) if m.any() else sp_pred.sum() * 0
    if args.query_weight > 0:
        f = model.q_former
        nodes, node_mask = f.last["nodes"], f.last["node_mask"]
        pi = dense(batch, "pi_node_mask") & node_mask
        out["care"] = soft_pointer_bce(f.pointer_logits("care", queries, nodes), dense(batch, "care_target"), pi)
        out["value"] = weighted_value_bce(f.value_logits(queries, nodes), dense(batch, "one_target"),
                                          dense(batch, "care_target"), pi)
        out["path"] = soft_pointer_bce(f.pointer_logits("path", queries, nodes), dense(batch, "path_target"),
                                       dense(batch, "gate_target_mask") & node_mask)
        out["obs"] = soft_pointer_bce(f.pointer_logits("obs", queries, nodes), dense(batch, "obs_target"),
                                      dense(batch, "po_target_mask") & node_mask)
        ac = anti_collapse(queries)
        out["anti_collapse"] = ac["variance"] + 0.04 * ac["covariance"] + ac["within"]
    if text is not None and args.text_weight > 0:
        gist = queries[:, model.q_former.role_slice("gist")]
        g = F.normalize(model.graph_proj(gist), dim=-1)                       # [B, S, D]
        t = F.normalize(model.text_proj(text["pooled"]), dim=-1)              # [B, D]
        sim = torch.einsum("isd,jd->isj", g, t).max(dim=1).values / 0.07
        labels = torch.arange(sim.size(0), device=sim.device)
        out["gtc"] = 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))
        # GTM: positives and hardest in-batch negatives, per query, averaged.
        with torch.no_grad():
            neg = sim.detach().clone()
            neg.fill_diagonal_(-1e4)
            hard = neg.argmax(dim=1)
        pos_logits = model.gtm(model.graph_proj(gist), model.text_proj(text["pooled"]))
        neg_logits = model.gtm(model.graph_proj(gist), model.text_proj(text["pooled"])[hard])
        out["gtm"] = F.binary_cross_entropy_with_logits(
            torch.cat([pos_logits, neg_logits]),
            torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)]))
        logits = model.gtg(queries, text["input_ids"][:, :-1], text["attention_mask"][:, :-1])
        target = text["input_ids"][:, 1:].masked_fill(text["attention_mask"][:, 1:] == 0, -100)
        out["gtg"] = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=-100)
    return out


def total_loss(losses, args):
    w = {"sp": args.sp_weight, "care": args.query_weight, "value": args.query_weight,
         "path": args.query_weight, "obs": args.query_weight, "anti_collapse": args.collapse_weight,
         "gtc": args.text_weight, "gtm": args.text_weight, "gtg": args.text_weight}
    return sum(w[k] * v for k, v in losses.items())


def average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
    if labels.sum() == 0 or labels.sum() == labels.numel():
        return float("nan")
    order = scores.argsort(descending=True)
    y = labels[order].float()
    hits = y.cumsum(0)
    precision = hits / torch.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


@torch.no_grad()
def evaluate(model, graphs: List, args, device) -> Dict[str, float]:
    model.eval()
    scores = {"care": [], "path": [], "obs": []}
    labels = {"care": [], "path": [], "obs": []}
    # Per node: is it a state bit, and is its circuit a scan circuit.
    state = {"care": [], "path": [], "obs": []}
    scan = {"care": [], "path": [], "obs": []}
    value_hits = value_n = 0
    sp_err = []
    qs = []
    for i in range(0, len(graphs), args.batch_size):
        batch = Batch.from_data_list(graphs[i:i + args.batch_size]).to(device)
        enc, queries = model(batch)
        m = batch.sp_mask
        if m.any():
            sp_err.append((torch.sigmoid(model.sp_head(enc["node_embs"]).squeeze(-1)[m]) - batch.sp_target[m]).abs().cpu())
        qs.append(queries.cpu())
        f = model.q_former
        nodes, node_mask = f.last["nodes"], f.last["node_mask"]
        for role, target, mask in (("care", "care_target", "pi_node_mask"),
                                   ("path", "path_target", "gate_target_mask"),
                                   ("obs", "obs_target", "po_target_mask")):
            valid = dense(batch, mask) & node_mask
            scores[role].append(f.pointer_logits(role, queries, nodes)[valid].cpu())
            labels[role].append((dense(batch, target)[valid] >= 0.5).cpu())
            bits = dense(batch, "state_in_mask") | dense(batch, "state_out_mask")
            state[role].append(bits[valid].cpu())
            scan[role].append(bits.any(dim=1, keepdim=True).expand_as(valid)[valid].cpu())
        care = (dense(batch, "care_target") >= 0.5) & dense(batch, "pi_node_mask") & node_mask
        if care.any():
            pred = f.value_logits(queries, nodes)[care] > 0
            value_hits += int((pred == (dense(batch, "one_target")[care] >= 0.5)).sum())
            value_n += int(care.sum())
    model.train()
    out = {f"ap_{r}": average_precision(torch.cat(scores[r]), torch.cat(labels[r])) for r in scores}
    # Scan circuits apart (NaN when the eval set has none), and inside them
    # state bits against primary ports.
    for r in scores:
        s, y, is_state, is_scan = (torch.cat(v[r]) for v in (scores, labels, state, scan))
        if not is_scan.any():
            continue
        out[f"ap_{r}_comb"] = average_precision(s[~is_scan], y[~is_scan])
        out[f"ap_{r}_scan"] = average_precision(s[is_scan], y[is_scan])
        if r != "path":
            out[f"ap_{r}_state"] = average_precision(s[is_state], y[is_state])
            out[f"ap_{r}_port"] = average_precision(s[is_scan & ~is_state], y[is_scan & ~is_state])
    out["value_acc"] = value_hits / max(1, value_n)
    out["sp_mae"] = float(torch.cat(sp_err).mean()) if sp_err else float("nan")
    out.update(collapse_metrics(torch.cat(qs)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--labels", required=True, nargs="+",
                    help="One or more label directories (build_retest_labels), for example v4 and v5.")
    ap.add_argument("--state-cells", action="store_true",
                    help="Give the state bits of scan circuits their own attribute bin. With --init from a "
                         "checkpoint trained without them, the new bin starts as a copy of the port bin.")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--eval-split", default="validation")
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-eval", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--text-lr", type=float, default=1e-5)
    ap.add_argument("--node-dim", type=int, default=256)
    ap.add_argument("--gin-layers", type=int, default=6)
    ap.add_argument("--q-hidden", type=int, default=512)
    ap.add_argument("--q-layers", type=int, default=6)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--sp-weight", type=float, default=1.0)
    ap.add_argument("--query-weight", type=float, default=1.0)
    ap.add_argument("--collapse-weight", type=float, default=1.0)
    ap.add_argument("--text-weight", type=float, default=0.0)
    ap.add_argument("--text-model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--text-max-len", type=int, default=96)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--init", help="Checkpoint whose matching weights start this run (R3 from R2).")
    ap.add_argument("--select", default="ap_care",
                    help="Eval metric that picks best.pt (sp_mae is lower-is-better; others higher).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    gate_funcs = json.loads(open(DEFAULT_SIM_CONFIG, encoding="utf-8").read())["gate_funcs"]
    vocab = GateAttributeVocab(gate_funcs, state_cells=args.state_cells)
    device = torch.device(args.device)

    train_p, train_c = load_split(args.labels, args.train_split, args.max_train, args.seed)
    eval_p, eval_c = load_split(args.labels, args.eval_split, args.max_eval, args.seed)
    eval_builder = ProblemGraphs(eval_c, gate_funcs, vocab)
    eval_graphs = [g for g in (eval_builder.build(p) for p in eval_p) if g is not None]
    train_builder = ProblemGraphs(train_c, gate_funcs, vocab)

    text_dim = None
    tokenizer = text_encoder = None
    if args.text_weight > 0:
        from transformers import AutoTokenizer
        from atpgllm.graph.models_stage1 import TextEncoder
        tokenizer = AutoTokenizer.from_pretrained(args.text_model)
        text_encoder = TextEncoder(args.text_model).to(device)
        text_dim = text_encoder.hidden_size
    model = RetestGraphModel(vocab, node_dim=args.node_dim, gin_layers=args.gin_layers, q_hidden=args.q_hidden,
                             slots=args.slots, q_layers=args.q_layers, text_dim=text_dim).to(device)
    params = [{"params": model.parameters(), "lr": args.lr}]
    if text_encoder is not None:
        model.gtm = GraphTextMatchingHead(256).to(device)
        model.gtg = QueryGroundedTextGenerator(len(tokenizer), 256, query_dim=args.q_hidden,
                                               max_len=args.text_max_len).to(device)
        params = [{"params": model.parameters(), "lr": args.lr},
                  {"params": text_encoder.parameters(), "lr": args.text_lr}]
    # After the text heads exist, so a checkpoint trained with text brings them too.
    if args.init:
        payload = torch.load(args.init, map_location="cpu", weights_only=False)
        init = payload["model"]
        own = model.state_dict()
        loaded = {k: v for k, v in init.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(loaded, strict=False)
        print(f"init: {len(loaded)} of {len(own)} tensors from {args.init}", flush=True)
        if args.state_cells and not GateAttributeVocab.from_dict(payload["vocab"]).has_state_cells:
            seeded = model.encoder.attr_encoder.seed_state_rows()
            print(f"init: state bin seeded from the port bin: {seeded}", flush=True)
    opt = torch.optim.AdamW(params, weight_decay=0.01)

    log = open(os.path.join(args.out, "metrics.jsonl"), "a", encoding="utf-8")
    log.write(json.dumps({"event": "start", "args": vars(args), "train_problems": len(train_p),
                          "eval_graphs": len(eval_graphs), "eval_rejected": eval_builder.rejected,
                          "eval_scan_graphs": sum(bool(g.state_in_mask.any()) for g in eval_graphs)}) + "\n")
    order = list(range(len(train_p)))
    cursor = len(order)
    start = time.time()
    best = None
    for step in range(1, args.steps + 1):
        items = []
        while len(items) < args.batch_size:
            if cursor >= len(order):
                random.shuffle(order)
                cursor = 0
            g = train_builder.build(train_p[order[cursor]])
            cursor += 1
            if g is not None:
                items.append(g)
        batch = Batch.from_data_list(items).to(device)
        text = None
        if text_encoder is not None:
            tok = tokenizer(list(batch.fact_text), padding=True, truncation=True,
                            max_length=args.text_max_len, return_tensors="pt").to(device)
            text = {"pooled": text_encoder(tok["input_ids"], tok["attention_mask"]),
                    "input_ids": tok["input_ids"], "attention_mask": tok["attention_mask"]}
        enc, queries = model(batch)
        losses = compute_losses(model, batch, enc, queries, args, text)
        loss = total_loss(losses, args)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 10 == 0 or step == 1:
            log.write(json.dumps({"step": step, "loss": float(loss), **{k: float(v) for k, v in losses.items()},
                                  "seconds": round(time.time() - start, 1)}) + "\n")
            log.flush()
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, eval_graphs, args, device)
            log.write(json.dumps({"step": step, "eval": metrics}) + "\n")
            log.flush()
            print(step, json.dumps(metrics), flush=True)
            score = metrics.get(args.select, float("nan"))
            if args.select == "sp_mae":
                score = -score
            state = {"format": "bridges_retest_v1", "model": model.state_dict(), "args": vars(args),
                     "vocab": vocab.to_dict(), "step": step,
                     "metrics": metrics}
            torch.save(state, os.path.join(args.out, "last.pt"))
            if best is None or (score == score and score > best):
                best = score
                torch.save(state, os.path.join(args.out, "best.pt"))
    log.close()


if __name__ == "__main__":
    main()
