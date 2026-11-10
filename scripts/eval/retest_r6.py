"""BRIDGES re-test R6: does Granite use the graph? (Notion plan phase R6, gate G3)

All passes read the text SFT's 704 validation rows (8 per held-out circuit,
the rows ``compare_stage_c_text_sft.py`` uses) and write into ``--out``.
Every pass can be stopped and run again: finished rows are skipped.

Models (``--model``): ``retest`` (16 graph queries in the slot),
``ports_only`` (no graph), ``anchored`` (each input bit's graph vector
added to the tokens of its name) and ``hint`` (no graph; the graph model's
decoded care inputs as text in the prompt).

``loss`` (GPU)
    Mean NLL of the trained tokens (tool call and final answer, the
    answer-only target) of every row. A model with a graph runs with its
    real graph, with none (``zero``), and with ``--shuffle-seeds`` wrong
    ones: for the slot another circuit's graph, for the anchors the row's
    own input vectors dealt to other inputs. The ports-only control and the
    hint arm have no graph (the slot is zero by construction).
``generate`` (GPU)
    Greedy first assistant turn (the tool call) on up to ``--per-circuit``
    rows per circuit: real, zero and one wrong graph for a model with a
    graph, the plain prompt otherwise. With ``--hints`` on a model that was
    not trained with hints (the ports-only control), the extra condition
    ``hint`` adds the hint line to its prompt: the untrained hint check.
``graph_only`` (CPU or GPU)
    The R4 decoders on the same rows, so the language models and the graph
    alone are scored alike: ``graph_decoder`` (care and value heads of the R3
    checkpoint), ``answer_key`` (the vote of the dataset's TetraMAX patterns)
    and ``random`` (no input chosen). Inputs a decoder leaves free are x.
``score`` (TetraMAX through ``FAULT_SIM_BACKEND``)
    Simulates every vector two ways. ``x kept``: x stays unknown, as the
    tool sees a model's call. ``x filled``: every x takes a coin flip that is
    the same for every method on that row, as R4 did. Then writes
    ``report.md`` and ``report.json`` with gate G3 and tables by location and
    difficulty. Text SFT rows come from ``generate_text.jsonl`` when present
    (copy it from the October 21st comparison, which used the same rows).

    PYTHONPATH=$PWD python scripts/eval/retest_r6.py loss --model retest \\
        --checkpoint runs/multimodal_sft_retest/multimodal_sft_best.pt \\
        --adapter runs/multimodal_sft_retest/adapter_best --out runs/retest_r6
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
import compare_stage_c_text_sft as cs  # noqa: E402  (also puts scripts/train on the path)

LABELS = ROOT.parent / "data" / "freeset" / "retest_labels_v4"
ALIGNMENT = ROOT / "runs" / "bridges_retest" / "r3" / "best.pt"
LOCATIONS = ("internal", "pi_stem", "po_stem")
DIFFICULTIES = ("easy", "medium", "hard", "very_hard", "random_never_detects")
RULES = ("kept", "filled")
MODELS = ("retest", "ports_only", "anchored", "hint")


# ---------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------

def inputs_of(record: Dict[str, Any]) -> List[str]:
    return list((cs._vector(record["input_vector"]) or {}).keys())


def outputs_of(record: Dict[str, Any]) -> List[str]:
    return list(json.loads(record["expected_output"]))


def location(record: Dict[str, Any]) -> str:
    net = str(record["fault"]).strip().split(None, 1)[1]
    if net in inputs_of(record):
        return "pi_stem"
    if net in outputs_of(record):
        return "po_stem"
    return "internal"


def fill_bits(record: Dict[str, Any], row: int) -> Dict[str, int]:
    """One coin flip per input, the same for every method on this row."""
    rng = random.Random(row)
    return {net: rng.randint(0, 1) for net in inputs_of(record)}


def filled(vector: Dict[str, Any], bits: Dict[str, int]) -> Dict[str, Any]:
    return {k: (bits.get(k, 0) if str(v).lower() == "x" else v) for k, v in vector.items()}


def shuffled_partner(records, index: int, seed: int) -> int:
    rng = random.Random(1_000_003 * seed + index)
    own = records[index]["circuit_id"]
    return rng.choice([i for i, r in enumerate(records) if r["circuit_id"] != own])


def _done(path: Path, key) -> set:
    return {key(json.loads(line)) for line in path.open()} if path.exists() else set()


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

def load_model(checkpoint: Path, adapter: Path):
    """A Stage C checkpoint of the re-test (either arm), built the way its
    run trained it: alignment stack, prompt view, target and graph-off flag
    come from the checkpoint's session."""
    import multimodal_training as mt
    from peft import PeftModel

    from atpgllm.graph.checkpoints import load_stage_checkpoint
    from atpgllm.graph.gate_features import GateAttributeVocab
    from atpgllm.multimodal import GraphConditionedCausalLM

    vocab = GateAttributeVocab(json.loads(cs.SIM_CONFIG.read_text())["gate_funcs"])
    payload = load_stage_checkpoint(checkpoint, expected_stages=["multimodal_sft"], vocab=vocab)
    session = payload["session"]
    args = cs._llm_args(alignment_ckpt=session["alignment_ckpt"])
    gate_funcs, vocab, graph_encoder, q_former, _, tokenizer = mt._load_common(args)
    llm = PeftModel.from_pretrained(mt._base_llm(args), adapter)
    architecture = payload["architecture"]["multimodal"]
    model = GraphConditionedCausalLM(graph_encoder, q_former, llm,
                                     prefix_at=architecture.get("prefix_at", "start"),
                                     fusion=architecture.get("fusion", "prefix"))
    if model.architecture_config != architecture:
        raise ValueError("The checkpoint's architecture differs from the model built.")
    device = mt._graph_device(model)
    for module in mt._graph_modules(model).values():
        module.to(device)
    mt._load_module_state(model, payload["states"])
    model.graph_off = bool(session.get("graph_off"))
    print(f"checkpoint {checkpoint}: step {payload.get('step')} target={session.get('target')} "
          f"netlist_view={session.get('netlist_view')} graph_off={model.graph_off} "
          f"fusion={model.fusion} hints={session.get('hints')}", flush=True)
    return model.eval(), tokenizer, gate_funcs, vocab, session


def hint_tables(args, session) -> tuple:
    """(the hints the model trained with, the hints of the untrained check).

    A hint arm always reads hints (``--hints``, else the path it trained
    with). For a model trained without them, ``--hints`` adds the ``hint``
    condition."""
    from atpgllm.multimodal.data import load_hints

    trained = session.get("hints")
    if trained:
        return load_hints(args.hints or trained), None
    return None, (load_hints(args.hints) if args.hints else None)


def example(record, tokenizer, gate_funcs, vocab, session, hints=None) -> Optional[Dict[str, Any]]:
    from types import SimpleNamespace

    from atpgllm.multimodal.data import build_tool_sft_example, collate_anchors

    ex = build_tool_sft_example(
        copy.deepcopy(record), gate_funcs, vocab, tokenizer, max_seq_len=1 << 30,
        tool_view=session.get("tool_view", "po_fault"),
        netlist_view=session.get("netlist_view", "graph"),
        target=session.get("target", "full"),
        port_bits=session.get("fusion", "prefix") != "prefix", hints=hints,
    )
    if ex is None:
        return None
    anchors = collate_anchors([SimpleNamespace(graph=ex.graph, anchors=ex.anchors)]) if ex.anchors else None
    return {"input_ids": ex.input_ids, "turns": cs._spans(label != -100 for label in ex.labels),
            "graph": ex.graph, "graph_at": ex.graph_at, "anchors": anchors}


PARTS = ("value", "input_value")


def value_parts(tokenizer, record, shown) -> List[List[Optional[str]]]:
    """Per turn and trained token: ``input_value`` for a 0, 1 or x of the
    first call's input vector (what first-try detection scores), ``value``
    for any other 0, 1 or x, else None. About 7% of the trained tokens are
    values. The rest are names and fixed text that every arm copies, so the
    mean over all tokens can hide whether a model uses its graph."""
    names = cs.netlist_names(cs.netlist_text(record))
    out = []
    for turn, (start, end) in enumerate(shown["turns"]):
        pieces = [tokenizer.decode([i]) for i in shown["input_ids"][start:end]]
        tags = cs.tag_tokens(pieces, names)
        text = "".join(pieces)
        cut = text.find("output_vector") if turn == 0 else 0
        cut = len(text) if cut < 0 else cut
        position, parts = 0, []
        for piece, tag in zip(pieces, tags):
            parts.append(None if tag != "value" else "input_value" if position < cut else "value")
            position += len(piece)
        out.append(parts)
    return out


def part_means(nll: List[List[float]], parts: List[List[Optional[str]]]) -> Dict[str, Any]:
    """``value`` covers every value token, ``input_value`` only the first call's inputs."""
    flat = [(x, part) for turn, tags in zip(nll, parts) for x, part in zip(turn, tags)]
    out = {}
    for name, keep in (("value", lambda p: p is not None), ("input_value", lambda p: p == "input_value")):
        xs = [x for x, part in flat if keep(part)]
        out[name] = {"n": len(xs), "mean": sum(xs) / len(xs) if xs else None}
    return out


def conditions_for(model, args, generate: bool, untrained_hint: bool = False) -> List[str]:
    conditions = ["real"]
    if not model.graph_off:
        seeds = 1 if generate else args.shuffle_seeds
        conditions += ["zero"] + [f"shuffled_{s}" for s in range(seeds)]
    return conditions + (["hint"] if untrained_hint else [])


def dealt_inputs(node_vectors, anchors, seed: int):
    """``node_vectors`` with every anchored input holding another anchored
    input's vector (unchanged when the row has one input)."""
    inputs = sorted(set(anchors[0][:, 2].tolist()))
    if len(inputs) < 2:
        return node_vectors
    rng = random.Random(seed)
    rng.shuffle(inputs)
    turn = rng.randrange(1, len(inputs))
    out = node_vectors.clone()
    out[inputs] = node_vectors[inputs[turn:] + inputs[:turn]]
    return out


def _context(model, records, index, condition, own, cache, tokenizer, gate_funcs, vocab, session):
    """(graph slot tokens, node vectors for the anchors or None) of one
    condition, or None when the wrong graph cannot be built."""
    import torch
    from torch_geometric.data import Batch

    device = model.llm.get_input_embeddings().weight.device
    context = lambda graph: model.graph_context(Batch.from_data_list([graph]).to(device))
    prefix, nodes = context(own["graph"])
    if condition == "zero":
        return torch.zeros_like(prefix), None
    if condition.startswith("shuffled_"):
        seed = int(condition.split("_")[1])
        if model.fusion != "anchored":
            partner = shuffled_partner(records, index, seed)
            if partner not in cache:
                ex = example(records[partner], tokenizer, gate_funcs, vocab, session)
                cache[partner] = ex["graph"] if ex else None
            if cache[partner] is None:
                return None
            prefix = context(cache[partner])[0]
        if nodes is not None and own["anchors"] is not None:
            nodes = dealt_inputs(nodes, own["anchors"], 1_000_003 * seed + index)
    return prefix, nodes


# ---------------------------------------------------------------------
# Passes
# ---------------------------------------------------------------------

def run_loss(args) -> None:
    import torch

    records = cs.load_records(args.limit)
    model, tokenizer, gate_funcs, vocab, session = load_model(args.checkpoint, args.adapter)
    out_path = args.out / f"loss_{args.model}.jsonl"
    done = _done(out_path, lambda r: r["row"])
    hints, untrained = hint_tables(args, session)
    conditions = conditions_for(model, args, generate=False, untrained_hint=untrained is not None)
    cache: Dict[int, Any] = {}
    started = time.time()
    with torch.no_grad(), out_path.open("a") as fh:
        for index, record in enumerate(records):
            if index in done:
                continue
            row = {"row": index, "circuit_id": record["circuit_id"], "fault": record["fault"],
                   "difficulty": record.get("difficulty"), "location": location(record),
                   "conditions": {}}
            own = example(record, tokenizer, gate_funcs, vocab, session, hints)
            if own is None:
                row["rejected"] = True
            else:
                for condition in conditions:
                    shown = own
                    if condition == "hint":
                        shown = example(record, tokenizer, gate_funcs, vocab, session, untrained)
                        if shown is None:
                            continue
                    context = _context(model, records, index, condition, shown, cache,
                                       tokenizer, gate_funcs, vocab, session)
                    if context is None:
                        continue
                    nll = cs.sequence_nll(model, shown["input_ids"], shown["turns"],
                                          prefix=context[0], graph_at=shown["graph_at"],
                                          anchors=shown["anchors"], node_vectors=context[1])
                    tokens = [x for turn in nll for x in turn]
                    if "parts" not in shown:
                        shown["parts"] = value_parts(tokenizer, record, shown)
                    row["conditions"][condition] = {
                        "n": len(tokens), "mean": sum(tokens) / max(1, len(tokens)),
                        "turns": [{"n": len(t), "mean": sum(t) / max(1, len(t))} for t in nll],
                        **part_means(nll, shown["parts"]),
                    }
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if index % 50 == 0:
                print(f"loss {args.model} row {index}/{len(records)} {time.time() - started:.0f}s",
                      flush=True)


def run_generate(args) -> None:
    import torch

    records = cs.load_records(args.limit)
    rows = cs._subset(records, args.per_circuit)
    model, tokenizer, gate_funcs, vocab, session = load_model(args.checkpoint, args.adapter)
    out_path = args.out / f"generate_{args.model}.jsonl"
    done = _done(out_path, lambda r: (r["row"], r["condition"]))
    hints, untrained = hint_tables(args, session)
    conditions = conditions_for(model, args, generate=True, untrained_hint=untrained is not None)
    stop = [t for t in (tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id)
            if isinstance(t, int) and t >= 0]
    device = model.llm.get_input_embeddings().weight.device
    cache: Dict[int, Any] = {}
    started = time.time()
    with torch.no_grad(), out_path.open("a") as fh:
        for count, index in enumerate(rows):
            record = records[index]
            own = None
            for condition in conditions:
                if (index, condition) in done:
                    continue
                own = own or example(record, tokenizer, gate_funcs, vocab, session, hints)
                if own is None:
                    break
                shown = own
                if condition == "hint":
                    shown = example(record, tokenizer, gate_funcs, vocab, session, untrained)
                    if shown is None:
                        continue
                context = _context(model, records, index, condition, shown, cache,
                                   tokenizer, gate_funcs, vocab, session)
                if context is None:
                    continue
                start, end = shown["turns"][0]
                prompt = torch.tensor([shown["input_ids"][:start]], device=device)
                budget = min(args.max_new_tokens, int(1.5 * (end - start)) + 64)
                # The anchored tokens are in the first user message, before the turn.
                embeds, mask = model._prefix_and_token_embeddings(
                    None, prompt, torch.ones_like(prompt), prefix=context[0],
                    graph_at=[shown["graph_at"]], anchors=shown["anchors"], node_vectors=context[1])
                output = model.llm.generate(inputs_embeds=embeds, attention_mask=mask,
                                            max_new_tokens=budget, do_sample=False,
                                            eos_token_id=stop, pad_token_id=tokenizer.pad_token_id)[0]
                fh.write(json.dumps({
                    "row": index, "condition": condition, "circuit_id": record["circuit_id"],
                    "fault": record["fault"], "difficulty": record.get("difficulty"),
                    "text": tokenizer.decode(output, skip_special_tokens=True),
                    "new_tokens": int(output.numel()), "budget": budget,
                }) + "\n")
                fh.flush()
            print(f"generate {args.model} {count + 1}/{len(rows)} {time.time() - started:.0f}s",
                  flush=True)


def run_graph_only(args) -> None:
    """R4's decoders on the generation rows; free inputs are x."""
    import torch
    from torch_geometric.data import Batch

    from atpgllm.graph.gate_features import GateAttributeVocab
    from atpgllm.graph.scripts.build_retest_labels import _netlist
    from atpgllm.graph.scripts.train_retest import ProblemGraphs, RetestGraphModel, load_split
    from atpgllm.training._paths import DEFAULT_SIM_CONFIG

    records = cs.load_records(args.limit)
    rows = cs._subset(records, args.per_circuit)
    out_path = args.out / "graph_only.jsonl"
    done = _done(out_path, lambda r: r["row"])
    state = torch.load(args.alignment, map_location="cpu", weights_only=False)
    a = state["args"]
    gate_funcs = json.loads(open(DEFAULT_SIM_CONFIG, encoding="utf-8").read())["gate_funcs"]
    vocab = GateAttributeVocab.from_dict(state["vocab"])
    model = RetestGraphModel(vocab, node_dim=a["node_dim"], gin_layers=a["gin_layers"],
                             q_hidden=a["q_hidden"], slots=a["slots"], q_layers=a["q_layers"])
    missing, _ = model.load_state_dict(state["model"], strict=False)
    if any(not k.startswith("text_proj") for k in missing):
        raise RuntimeError(f"checkpoint is missing weights: {missing}")
    device = torch.device(args.device)
    model.to(device).eval()
    problems, circuits = load_split(str(args.labels), "validation", 0, 0)
    by_key = {(p["netlist_key"], p["fault"]): p for p in problems}
    builder = ProblemGraphs(circuits, gate_funcs, vocab)
    with out_path.open("a") as fh:
        for index in rows:
            if index in done:
                continue
            record = records[index]
            key = hashlib.sha256(_netlist(record["netlist"]).encode("utf-8")).hexdigest()[:16]
            p = by_key.get((key, str(record["fault"]).strip()))
            graph = builder.build(p) if p else None
            if graph is None:
                fh.write(json.dumps({"row": index, "missing": True}) + "\n")
                continue
            with torch.no_grad():
                _, q = model(Batch.from_data_list([graph]).to(device))
                f = model.q_former
                nodes = f.last["nodes"]
                care = torch.sigmoid(f.pointer_logits("care", q, nodes))[0]
                value = torch.sigmoid(f.value_logits(q, nodes))[0]
            names = list(graph.node_names)
            pi = {names[i]: i for i in range(len(names)) if bool(graph.pi_node_mask[i])}
            nets = inputs_of(record)
            vectors = {
                "graph_decoder": {n: (int(value[pi[n]] >= 0.5) if n in pi and care[pi[n]] >= 0.5 else "x")
                                  for n in nets},
                "answer_key": {n: (int(p["one_freq"].get(n, 0) >= 0.5) if p["care_freq"].get(n, 0) >= 0.5
                                   else "x") for n in nets},
                "random": {n: "x" for n in nets},
            }
            fh.write(json.dumps({"row": index, "vectors": vectors}) + "\n")
            fh.flush()


def _simulate(record, vector) -> tuple:
    from atpgllm.training.tool_feedback import read_snapshot
    from atpgllm.training.tool_trajectories import simulate

    try:
        snapshot = simulate(record, vector)
        return bool(read_snapshot(snapshot, record["fault"], outputs_of(record)).detected), None
    except Exception as exc:  # an invalid vector is a miss, as in the tool probe
        return False, f"{type(exc).__name__}: {exc}"[:200]


def score_vector(record, row: int, vector: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = {"parsed": vector is not None, "keys_exact": False, "detects_kept": False,
           "detects_filled": False, "sim_error": None}
    if not vector:
        out["parsed"] = False
        return out
    out["keys_exact"] = set(vector) == set(inputs_of(record))
    out["detects_kept"], out["sim_error"] = _simulate(record, vector)
    full = filled(vector, fill_bits(record, row))
    out["detects_filled"] = (out["detects_kept"] if full == vector
                             else _simulate(record, full)[0])
    return out


def _sources(out: Path):
    """(method label, row, condition, vector or None, truncated)."""
    from atpgllm.training.revert_template import parse_tool_call

    for model in MODELS + ("text",):
        path = out / f"generate_{model}.jsonl"
        if not path.exists():
            continue
        for g in map(json.loads, path.open()):
            call = parse_tool_call(g["text"])
            vector = (cs._vector(call["arguments"].get("input_vector"))
                      if call and isinstance(call.get("arguments"), dict) else None)
            yield model, g["row"], g["condition"], vector, g["new_tokens"] >= g["budget"]
    path = out / "graph_only.jsonl"
    if path.exists():
        for g in map(json.loads, path.open()):
            for method, vector in (g.get("vectors") or {}).items():
                yield method, g["row"], "", vector, False


def run_score(args) -> None:
    records = cs.load_records(args.limit)
    scored_path = args.out / "scored.jsonl"
    done = _done(scored_path, lambda r: (r["model"], r["row"], r["condition"]))
    with scored_path.open("a") as fh:
        for model, row, condition, vector, truncated in _sources(args.out):
            if (model, row, condition) in done or row >= len(records):  # --limit
                continue
            record = records[row]
            fh.write(json.dumps({
                "model": model, "row": row, "condition": condition,
                "circuit_id": record["circuit_id"], "difficulty": record.get("difficulty"),
                "location": location(record), "truncated": truncated,
                **score_vector(record, row, vector),
            }) + "\n")
            fh.flush()
    scored = list(map(json.loads, scored_path.open()))
    losses = {m: [r for r in map(json.loads, (args.out / f"loss_{m}.jsonl").open())
                  if not r.get("rejected")]
              for m in MODELS if (args.out / f"loss_{m}.jsonl").exists()}
    report = build_report(scored, losses, args.g3_margin)
    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    (args.out / "report.md").write_text(markdown(report))
    print(markdown(report))


# ---------------------------------------------------------------------
# Report and gate G3
# ---------------------------------------------------------------------

def _rate(rows, rule) -> Optional[float]:
    return sum(r[f"detects_{rule}"] for r in rows) / len(rows) if rows else None


def detection_table(scored, rows_kept: Optional[set] = None) -> Dict[str, Any]:
    groups = collections.defaultdict(list)
    for r in scored:
        if rows_kept is not None and r["row"] not in rows_kept:
            continue
        name = r["model"] + (f"/{r['condition']}" if r["condition"] else "")
        groups[name].append(r)
    table = {}
    for name, rows in sorted(groups.items()):
        entry = {"n": len(rows), "parsed": sum(r["parsed"] for r in rows) / len(rows)}
        for rule in RULES:
            entry[rule] = {"all": _rate(rows, rule)}
            for loc in LOCATIONS:
                sub = [r for r in rows if r["location"] == loc]
                entry[rule][loc] = _rate(sub, rule)
                entry.setdefault("n_by", {})[loc] = len(sub)
            for d in DIFFICULTIES:
                sub = [r for r in rows if r["difficulty"] == d]
                entry[rule][d] = _rate(sub, rule)
                entry.setdefault("n_by", {})[d] = len(sub)
        table[name] = entry
    return table


def circuit_bootstrap(pairs, resamples: int = 2000, seed: int = 0) -> List[float]:
    """95% interval of the mean paired difference, resampling whole circuits:
    rows of one circuit share its netlist and are not independent."""
    by = collections.defaultdict(list)
    for circuit, diff in pairs:
        by[circuit].append(diff)
    groups = list(by.values())
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        draw = [groups[rng.randrange(len(groups))] for _ in groups]
        values = [d for g in draw for d in g]
        means.append(sum(values) / len(values))
    means.sort()
    return [means[int(0.025 * resamples)], means[int(0.975 * resamples)]]


def loss_gate(losses: Dict[str, list], model: str = "retest", part: Optional[str] = None) -> Dict[str, Any]:
    """Real graph against wrong graphs, on the rows of ``model`` (an arm
    with a graph). ``part`` (:data:`PARTS`) reads the loss of the value
    tokens only, on the rows that have them.

    Pass: the circuit-bootstrap 95% interval of (shuffled minus real) lies
    above zero. The spread of the shuffle draws (``seed_sd``) is reported
    too, but each draw averages hundreds of rows, so it is small and a
    "3 sd" rule on it alone would pass almost any gap."""
    def view(row):
        """The row with each condition reduced to the mean this gate reads."""
        conditions = {c: (v if part is None else v.get(part) or {}) for c, v in row["conditions"].items()}
        return {**row, "conditions": {c: v for c, v in conditions.items() if v.get("mean") is not None}}

    control_rows = [view(r) for r in losses.get("ports_only") or []]
    rows = [view(r) for r in losses.get(model) or []]
    rows = [r for r in rows if "real" in r["conditions"]]
    seeds = sorted({c for r in rows for c in r["conditions"] if c.startswith("shuffled_")})
    if not rows or not seeds:
        return {}
    rows = [r for r in rows if all(s in r["conditions"] for s in seeds)]
    mean = lambda xs: sum(xs) / len(xs)
    real = [r["conditions"]["real"]["mean"] for r in rows]
    per_seed = {s: mean([r["conditions"][s]["mean"] for r in rows]) for s in seeds}
    shuffled_row = [mean([r["conditions"][s]["mean"] for s in seeds]) for r in rows]
    gap = mean(shuffled_row) - mean(real)
    sd = (math.sqrt(sum((v - mean(list(per_seed.values()))) ** 2 for v in per_seed.values())
                    / (len(seeds) - 1)) if len(seeds) > 1 else None)
    low, high = circuit_bootstrap([(r["circuit_id"], s - x) for r, s, x in zip(rows, shuffled_row, real)])
    out = {"rows": len(rows), "circuits": len({r["circuit_id"] for r in rows}), "real": mean(real),
           "shuffled_by_seed": per_seed, "shuffled_mean": mean(shuffled_row), "gap": gap,
           "seed_sd": sd, "gap_in_seed_sd": (gap / sd) if sd else None, "circuit_bootstrap_95": [low, high],
           "zero": mean([r["conditions"]["zero"]["mean"] for r in rows if "zero" in r["conditions"]])}
    control = {r["row"]: r["conditions"]["real"]["mean"] for r in control_rows
               if "real" in r["conditions"]}
    shared = [r for r in rows if r["row"] in control]
    if shared:
        out["ports_only_same_rows"] = mean([control[r["row"]] for r in shared])
        out["real_same_rows"] = mean([r["conditions"]["real"]["mean"] for r in shared])
    out["pass"] = bool(low > 0)
    return out


def detection_gap(scored, rule: str, model: str = "retest") -> Dict[str, Any]:
    """Internal faults: the arm ``model`` (as trained) minus ports only,
    paired by row. The keys keep the first arm's name."""
    pick = lambda name: {r["row"]: r for r in scored if r["model"] == name and r["condition"] == "real"
                         and r["location"] == "internal"}
    graph, control = pick(model), pick("ports_only")
    rows = sorted(set(graph) & set(control))
    if not rows:
        return {"rows": 0, "retest_real_internal": None, "ports_only_internal": None, "difference": None}
    g = sum(graph[i][f"detects_{rule}"] for i in rows) / len(rows)
    c = sum(control[i][f"detects_{rule}"] for i in rows) / len(rows)
    ci = circuit_bootstrap([(graph[i]["circuit_id"],
                             int(graph[i][f"detects_{rule}"]) - int(control[i][f"detects_{rule}"])) for i in rows])
    return {"rows": len(rows), "circuits": len({graph[i]["circuit_id"] for i in rows}),
            "retest_real_internal": g, "ports_only_internal": c, "difference": g - c,
            "circuit_bootstrap_95": ci}


def build_report(scored, losses, margin: float) -> Dict[str, Any]:
    by_method = collections.defaultdict(set)
    for r in scored:
        by_method[r["model"] + (f"/{r['condition']}" if r["condition"] else "")].add(r["row"])
    matched = set.intersection(*by_method.values()) if by_method else set()
    full = detection_table(scored)

    def gate(model: str) -> Dict[str, Any]:
        detect = {rule: detection_gap(scored, rule, model) for rule in RULES}
        loss = loss_gate(losses, model)
        # The plan's rule (at least ``margin`` points), and the gap must also
        # be clear of zero across circuits.
        kept = detect["kept"]
        detect_pass = bool(kept["difference"] is not None and kept["difference"] >= margin
                           and kept["circuit_bootstrap_95"][0] > 0)
        # Reported, not part of the pass rule: the same gap on the value tokens.
        parts = {f"loss_{part}": loss_gate(losses, model, part) for part in PARTS}
        return {"loss": loss, **parts, "internal_detection": detect, "margin": margin,
                "detection_pass": detect_pass, "pass": bool(loss.get("pass") and detect_pass)}

    # The same gate for the arms added on 11-10. The hint arm has no graph,
    # so only its detection half can pass.
    present = {r["model"] for r in scored} | set(losses)
    return {"matched_rows": len(matched), "all_rows": full,
            "matched": detection_table(scored, matched) if matched else {},
            "G3": gate("retest"),
            "arms": {model: gate(model) for model in ("anchored", "hint") if model in present}}


def _pct(x) -> str:
    return "" if x is None else f"{100 * x:.1f}%"


def _gate_lines(arm: str, g3) -> List[str]:
    lines: List[str] = []
    for key, what in (("loss", "the trained answer tokens"),
                      ("loss_value", "the 0, 1 and x value tokens only"),
                      ("loss_input_value", "the first call's input values only")):
        loss = g3.get(key)
        if not loss:
            continue
        lines += [f"## G3 {arm}, loss (mean NLL of {what})", "",
                  f"- rows {loss['rows']}: real {loss['real']:.4f}, wrong graph {loss['shuffled_mean']:.4f} "
                  f"(by seed {', '.join(f'{v:.4f}' for v in loss['shuffled_by_seed'].values())}), "
                  f"zero {loss['zero']:.4f}",
                  f"- gap {loss['gap']:.4f}, circuit bootstrap 95% "
                  f"[{loss['circuit_bootstrap_95'][0]:.4f}, {loss['circuit_bootstrap_95'][1]:.4f}] "
                  f"over {loss['circuits']} circuits; shuffle-draw sd {loss['seed_sd']}",
                  f"- ports-only model on the same rows {loss.get('ports_only_same_rows')}",
                  f"- pass: {loss['pass']}", ""]
    d = g3["internal_detection"]
    lines += [f"## G3 {arm}, internal-fault first-try detection", "",
              f"- x kept, {d['kept']['rows']} paired rows: {arm} {_pct(d['kept']['retest_real_internal'])}, "
              f"ports only {_pct(d['kept']['ports_only_internal'])}, circuit bootstrap 95% of the difference "
              f"{[_pct(x) for x in d['kept'].get('circuit_bootstrap_95') or []]}, margin needed "
              f"{_pct(g3['margin'])}, pass {g3['detection_pass']}",
              f"- x filled: {arm} {_pct(d['filled']['retest_real_internal'])}, ports only "
              f"{_pct(d['filled']['ports_only_internal'])}",
              "", f"**G3 {arm} pass: {g3['pass']}**", ""]
    return lines


def markdown(report) -> str:
    lines = ["# BRIDGES re-test R6", ""]
    for arm, g3 in [("re-test", report["G3"])] + list(report.get("arms", {}).items()):
        lines += _gate_lines(arm, g3)
    for title, table in (("All generated rows", report["all_rows"]),
                         (f"Matched rows ({report['matched_rows']}, every method answered)",
                          report["matched"])):
        for rule in RULES:
            lines += [f"## {title}, x {rule}", "",
                      "| method | n | parsed | all | " + " | ".join(LOCATIONS + DIFFICULTIES[:4]) + " |",
                      "|---" * (4 + len(LOCATIONS) + 4) + "|"]
            for name, e in table.items():
                cells = [_pct(e[rule].get(k)) for k in LOCATIONS + DIFFICULTIES[:4]]
                lines.append(f"| {name} | {e['n']} | {_pct(e['parsed'])} | {_pct(e[rule]['all'])} | "
                             + " | ".join(cells) + " |")
            lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pass_", metavar="pass", choices=("loss", "generate", "graph_only", "score"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--hints", type=Path,
                        help="Hint files (retest_hints). A hint arm reads them in place of the "
                             "path it trained with; any other model gets the extra condition hint.")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--alignment", type=Path, default=ALIGNMENT)
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--limit", type=int, default=None, help="First N validation rows only.")
    parser.add_argument("--per-circuit", type=int, default=8)
    parser.add_argument("--shuffle-seeds", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--g3-margin", type=float, default=0.10)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.pass_ in ("loss", "generate") and not (args.model and args.checkpoint and args.adapter):
        parser.error(f"{args.pass_} needs --model, --checkpoint and --adapter")
    args.out.mkdir(parents=True, exist_ok=True)
    {"loss": run_loss, "generate": run_generate, "graph_only": run_graph_only,
     "score": run_score}[args.pass_](args)


if __name__ == "__main__":
    main()
