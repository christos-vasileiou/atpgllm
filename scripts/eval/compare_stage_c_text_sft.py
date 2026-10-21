"""Compare the graph-conditioned Stage C SFT with the text SFT on the same
held-out rows.

The rows are the text SFT's own validation examples
(``<text run>/sft_validation_examples.json``), single-call conversations
only. Three passes, each writing into ``--out``:

``loss`` (GPU)
    Per-token NLL of every trained token, for one model under several
    conditions. Text SFT: the prompt shows the full netlist (``full``; and
    ``cached``, the run's own cached validation messages, whose tool-call
    arguments come in another order, as its logged eval loss saw them), only
    the module's port declarations (``ports``), or the placeholder Stage C
    sees (``none``). Stage C: the real graph prefix (``real``), a zero prefix
    (``zero``), another circuit's graph (``shuffled``), or the encoder's
    BatchNorm statistics from Stage B instead of the trained ones
    (``bn_stage_b``).
``generate`` (GPU)
    Greedy first assistant turn under the same conditions, on a subset of
    rows (at most ``--per-circuit`` per circuit).
``score`` (CPU, TetraMAX)
    Parses each generated tool call and simulates it with the tool's
    simulator (``FAULT_SIM_BACKEND``), then writes ``report.md`` and
    ``report.json`` with the loss comparison too.

The two pipelines format the conversation slightly differently (where the
newlines around ``<think>`` fall, and the final turn's wording), so the
first-turn comparison uses the tokens both sequences share, aligned by a
diff. Each trained token is tagged as a port name, gate instance name,
internal net name, 0/1/x value or other, from the record's netlist.

    PYTHONPATH=$PWD python scripts/eval/compare_stage_c_text_sft.py loss \
        --out runs/compare_stage_c_v4/20261021 --model text
"""

from __future__ import annotations

import argparse
import collections
import copy
import difflib
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "train"))

MONOREPO = ROOT.parent
TEXT_RUN = MONOREPO / "libatpgllm" / "runs" / "sft_granite_4.2_8b_v4"
STAGE_C_RUN = MONOREPO / "libatpgllm_graph-modality" / "runs" / "multimodal_sft_v4"
STAGE_B = MONOREPO / "libatpgllm_graph-modality" / "runs" / "graph_alignment_v3_pretrained_seed42" / "stage1_best.pt"
DATASET = MONOREPO / "data" / "freeset" / "dataset.freeset.asap7sc7p5t_28.rvt.tt.v4"
SIM_CONFIG = ROOT / "atpgllm" / "training" / "data" / "sim_config.json"
LLM = "ibm-granite/granite-4.2-8b"

TEXT_CONDITIONS = ("full", "cached", "ports", "none")
STAGE_C_CONDITIONS = ("real", "zero", "shuffled", "bn_stage_b")
BUCKETS = ("port", "instance", "net", "value", "other")


# ---------------------------------------------------------------------
# Rows and conversations
# ---------------------------------------------------------------------

def load_records(limit: Optional[int]) -> List[Dict[str, Any]]:
    """The text SFT's validation rows, in its order."""
    import pyarrow.parquet as pq

    from atpgllm.graph.circuit_sampling import split_files

    examples = json.loads((TEXT_RUN / "sft_validation_examples.json").read_text())
    # A circuit groups structurally equal netlists whose port names differ,
    # so the netlist is part of the key.
    keys = [(e["circuit_id"], e["netlist_id"], e["fault"], int(e["pattern_index"]))
            for e in examples["report"]["examples"]]
    if limit:
        keys = keys[:limit]
    circuits = sorted({key[0] for key in keys})
    wanted = set(keys)
    found: Dict[tuple, Dict[str, Any]] = {}
    for path in split_files(str(DATASET), "validation", None):
        table = pq.read_table(path, filters=[("circuit_id", "in", circuits)])
        for record in table.to_pylist():
            key = (record["circuit_id"], record["netlist_id"], record["fault"],
                   int(record["pattern_index"]))
            if key in wanted:
                found.setdefault(key, record)
    missing = [key for key in keys if key not in found]
    if missing:
        raise RuntimeError(f"{len(missing)} validation rows not found, e.g. {missing[0]}")
    return [found[key] for key in keys]


def netlist_text(record: Dict[str, Any]) -> str:
    netlist = record.get("netlist")
    return (netlist.get("netlist") if isinstance(netlist, dict) else netlist) or ""


def port_declarations(netlist: str) -> str:
    """What Stage C v4b's prompt keeps of the netlist, without the graph
    placeholder."""
    from atpgllm.multimodal.data import port_declarations as declarations

    return declarations(netlist) + "endmodule\n"


def netlist_names(netlist: str) -> Dict[str, set]:
    """Port (bits and buses), cell instance and internal net names."""
    ports = set()
    for _, width, names in re.findall(
        r"\b(input|output)\s*(\[\s*\d+\s*:\s*\d+\s*\])?\s*([^;]+);", netlist
    ):
        for name in (n.strip().lstrip("\\") for n in names.split(",")):
            if not name:
                continue
            ports.add(name)
            if width:
                a, b = map(int, re.findall(r"\d+", width))
                ports.update(f"{name}[{i}]" for i in range(min(a, b), max(a, b) + 1))
    instances = {m.strip().lstrip("\\") for m in
                 re.findall(r"^\s*\w+ASAP7\w*\s+(\\?\S+)\s*\(", netlist, re.M)}
    nets = {m.strip().lstrip("\\") for m in
            re.findall(r"\.\w+\(\s*(\\?[^\s\)]+)\s*\)", netlist)} - ports
    return {"port": ports, "instance": instances, "net": nets}


def tag_tokens(pieces: List[str], names: Dict[str, set]) -> List[str]:
    """One bucket per decoded token: the most common label of its
    non-punctuation characters."""
    text = "".join(pieces)
    labels = ["other"] * len(text)
    for bucket in ("port", "instance", "net"):
        for name in sorted(names[bucket], key=len, reverse=True):
            for m in re.finditer(r"(?<![\w\[\]])" + re.escape(name) + r"(?![\w\[])", text):
                for j in range(m.start(), m.end()):
                    if labels[j] == "other":
                        labels[j] = bucket
    for m in re.finditer(r'(?<=[:"] )[01x](?=[,}"\s])|(?<=": )[01]|(?<=": ")x(?=")', text):
        for j in range(m.start(), m.end()):
            if labels[j] == "other":
                labels[j] = "value"
    out, position = [], 0
    for piece in pieces:
        span = labels[position:position + len(piece)]
        chars = [lab for lab, ch in zip(span, piece)
                 if not ch.isspace() and ch not in '",:{}[]'] or span or ["other"]
        out.append(collections.Counter(chars).most_common(1)[0][0])
        position += len(piece)
    return out


def _spans(flags: Iterable[bool]) -> List[tuple]:
    spans, start = [], None
    flags = list(flags)
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        if not flag and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(flags)))
    return spans


def cached_messages() -> List[Dict[str, Any]]:
    return [row["messages"] for row in
            json.loads((TEXT_RUN / "sft_validation_examples.json").read_text())["rows"]]


def text_example(record, tokenizer, condition: str, cached=None) -> Dict[str, Any]:
    """The text SFT's own tokenization (TRL assistant mask of the patched
    template saved with the checkpoint), with the prompt's netlist replaced
    for the ``ports`` and ``none`` conditions."""
    from atpgllm.multimodal.data import GRAPH_CONTEXT
    from atpgllm.training.dataset_utils import _format_sft_messages_record, tool_vectors_as_text

    formatted = _format_sft_messages_record(
        copy.deepcopy(record), True, tool_vectors_as_text(tokenizer)
    )
    if condition == "cached":
        formatted["messages"] = cached
    netlist = netlist_text(record)
    if condition in ("ports", "none"):
        stand_in = port_declarations(netlist) if condition == "ports" else GRAPH_CONTEXT
        for message in formatted["messages"]:
            if message["role"] == "user":
                # The prompt prints the payload with repr(), which escapes.
                message["content"] = message["content"].replace(repr(netlist)[1:-1], stand_in)
                if stand_in not in message["content"]:
                    raise ValueError("netlist not found in the user message")
    out = tokenizer.apply_chat_template(
        formatted["messages"], tools=formatted["tools"], tokenize=True,
        return_dict=True, return_assistant_tokens_mask=True,
    )
    ids = list(out["input_ids"])
    turns = _spans(out["assistant_masks"])
    return {"input_ids": ids, "turns": turns}


def stage_c_example(record, tokenizer, gate_funcs, vocab,
                    netlist_view: str = "graph") -> Optional[Dict[str, Any]]:
    from atpgllm.multimodal.data import build_tool_sft_example

    example = build_tool_sft_example(
        copy.deepcopy(record), gate_funcs, vocab, tokenizer, max_seq_len=1 << 30,
        tool_view="po_fault", netlist_view=netlist_view,
    )
    if example is None:
        return None
    return {"input_ids": example.input_ids,
            "turns": _spans(label != -100 for label in example.labels),
            "graph": example.graph, "graph_at": example.graph_at}


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

def _llm_args(**extra) -> argparse.Namespace:
    return argparse.Namespace(
        llm=LLM, local_rank=0, world_size=1, attn_implementation="flash_attention_2",
        load_4bit=True, sim_config=SIM_CONFIG, **extra,
    )


def load_text_model(adapter: Path):
    import multimodal_training as mt
    from peft import PeftModel

    model = PeftModel.from_pretrained(mt._base_llm(_llm_args()), adapter)
    return model.eval()


def load_stage_c_model(checkpoint: Path, adapter: Path):
    """The model, gate functions, vocabulary and the checkpoint's netlist
    view; the graph goes where the checkpoint's architecture says."""
    import multimodal_training as mt
    from peft import PeftModel

    from atpgllm.graph.checkpoints import load_stage_checkpoint
    from atpgllm.multimodal import GraphConditionedCausalLM

    args = _llm_args(alignment_ckpt=STAGE_B)
    gate_funcs, vocab, graph_encoder, q_former, _, _ = mt._load_common(args)
    payload = load_stage_checkpoint(checkpoint, expected_stages=["multimodal_sft"], vocab=vocab)
    llm = PeftModel.from_pretrained(mt._base_llm(args), adapter)
    prefix_at = payload["architecture"]["multimodal"].get("prefix_at", "start")
    model = GraphConditionedCausalLM(graph_encoder, q_former, llm, prefix_at=prefix_at)
    if model.architecture_config != payload["architecture"]["multimodal"]:
        raise ValueError("The Stage C checkpoint's architecture differs from the model built.")
    device = mt._graph_device(model)
    for module in (model.graph_encoder, model.q_former, model.graph_to_llm):
        module.to(device)
    mt._load_module_state(model, payload["states"])
    netlist_view = payload["session"].get("netlist_view", "graph")
    print(f"Stage C checkpoint: prefix_at={prefix_at} netlist_view={netlist_view}", flush=True)
    return model.eval(), gate_funcs, vocab, netlist_view


def stage_b_batchnorm_buffers() -> Dict[str, Any]:
    import torch

    states = torch.load(STAGE_B, map_location="cpu", weights_only=False)["states"]["model"]
    prefix = "graph_encoder."
    return {k[len(prefix):]: v for k, v in states.items()
            if k.startswith(prefix) and ("running_" in k or "num_batches" in k)}


def _token_nll(decoder, head, scale, inputs_embeds, targets_at, positions):
    """NLL of ``targets_at`` predicted from ``positions`` (one sequence)."""
    import torch

    hidden = decoder(inputs_embeds=inputs_embeds).last_hidden_state[0]
    losses = []
    for i in range(0, len(positions), 2048):
        states = hidden[positions[i:i + 2048]]
        logits = head(states).float() / scale
        losses.append(torch.nn.functional.cross_entropy(
            logits, targets_at[i:i + 2048], reduction="none"))
    return torch.cat(losses).tolist()


def sequence_nll(model, input_ids, turns, *, prefix=None, graph_at: int = 0) -> List[List[float]]:
    """Per-turn lists of per-token NLL. ``prefix`` (``[1, Q, D]``) is the
    graph soft prompt for Stage C, placed by the model itself (before the
    chat, or before token ``graph_at``); None for the text model."""
    import torch

    llm = model.llm if prefix is not None else model
    base = llm.get_base_model() if hasattr(llm, "get_base_model") else llm
    embed = base.get_input_embeddings()
    device = embed.weight.device
    ids = torch.tensor([input_ids], device=device)
    at, width = 0, 0
    if prefix is None:
        embeds = embed(ids)
    else:
        embeds, _ = model._prefix_and_token_embeddings(
            None, ids, torch.ones_like(ids), prefix=prefix, graph_at=[graph_at])
        width = prefix.size(1)
        at = graph_at if model.prefix_at == "placeholder" else 0
    # Token t sits at t, or t + Q once past the graph; position p predicts p + 1.
    place = lambda t: t + (width if t >= at else 0)
    positions = [place(t) - 1 for start, end in turns for t in range(start, end)]
    targets = torch.tensor([input_ids[t] for start, end in turns for t in range(start, end)],
                           device=device)
    losses = _token_nll(base.get_decoder(), base.get_output_embeddings(),
                        getattr(base.config, "logits_scaling", 1.0), embeds, targets,
                        torch.tensor(positions, device=device))
    out, i = [], 0
    for start, end in turns:
        out.append(losses[i:i + end - start])
        i += end - start
    return out


# ---------------------------------------------------------------------
# Passes
# ---------------------------------------------------------------------

def _graph_prefix(model, graph, device):
    from torch_geometric.data import Batch

    return model.graph_soft_prompt(Batch.from_data_list([graph]).to(device))


def _shuffled_partner(records, index):
    """A row of another circuit (fixed per row)."""
    rng = random.Random(index)
    own = records[index]["circuit_id"]
    others = [i for i, r in enumerate(records) if r["circuit_id"] != own]
    if not others:
        raise ValueError("The shuffled condition needs rows from two circuits (raise --limit).")
    return rng.choice(others)


def run_loss(args) -> None:
    import torch
    from transformers import AutoTokenizer

    records = load_records(args.limit)
    out_path = args.out / f"loss_{args.model}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(line)["row"] for line in out_path.open()}
    if args.model == "text":
        tokenizer = AutoTokenizer.from_pretrained(args.text_adapter)
        model = load_text_model(args.text_adapter)
        cached = cached_messages()
    else:
        tokenizer = AutoTokenizer.from_pretrained(LLM)
        model, gate_funcs, vocab, netlist_view = load_stage_c_model(
            args.stage_c_checkpoint, args.stage_c_adapter)
        trained_bn = {k: v.clone() for k, v in model.graph_encoder.state_dict().items()
                      if "running_" in k or "num_batches" in k}
        stage_b_bn = stage_b_batchnorm_buffers()
        device = model.llm.get_input_embeddings().weight.device
        graphs: Dict[int, Any] = {}
    started = time.time()
    with torch.no_grad(), out_path.open("a") as fh:
        for index, record in enumerate(records):
            if index in done:
                continue
            names = netlist_names(netlist_text(record))
            row: Dict[str, Any] = {"row": index, "circuit_id": record["circuit_id"],
                                   "fault": record["fault"],
                                   "difficulty": record.get("difficulty"), "conditions": {}}
            if args.model == "text":
                for condition in TEXT_CONDITIONS:
                    example = text_example(record, tokenizer, condition, cached[index])
                    nll = sequence_nll(model, example["input_ids"], example["turns"])
                    row["conditions"][condition] = _turn_rows(example, nll, tokenizer, names)
                    row["prompt_tokens"] = row.get("prompt_tokens") or example["turns"][0][0]
            else:
                example = stage_c_example(record, tokenizer, gate_funcs, vocab, netlist_view)
                if example is None:
                    row["rejected"] = True
                    fh.write(json.dumps(row) + "\n")
                    continue
                partner_index = _shuffled_partner(records, index)
                if partner_index not in graphs:
                    partner = stage_c_example(records[partner_index], tokenizer, gate_funcs, vocab,
                                              netlist_view)
                    graphs[partner_index] = partner["graph"] if partner else example["graph"]
                for condition in STAGE_C_CONDITIONS:
                    if condition == "bn_stage_b":
                        model.graph_encoder.load_state_dict(stage_b_bn, strict=False)
                    graph = graphs[partner_index] if condition == "shuffled" else example["graph"]
                    prefix = _graph_prefix(model, graph, device)
                    if condition == "zero":
                        prefix = torch.zeros_like(prefix)
                    if condition == "bn_stage_b":
                        model.graph_encoder.load_state_dict(trained_bn, strict=False)
                    nll = sequence_nll(model, example["input_ids"], example["turns"],
                                       prefix=prefix, graph_at=example["graph_at"])
                    row["conditions"][condition] = _turn_rows(example, nll, tokenizer, names)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if index % 25 == 0:
                print(f"loss {args.model} row {index}/{len(records)} "
                      f"{time.time() - started:.0f}s", flush=True)


def _turn_rows(example, nll, tokenizer, names) -> List[Dict[str, Any]]:
    turns = []
    for (start, end), losses in zip(example["turns"], nll):
        ids = example["input_ids"][start:end]
        pieces = [tokenizer.decode([i]) for i in ids]
        turns.append({"ids": ids, "nll": [round(x, 5) for x in losses],
                      "bucket": tag_tokens(pieces, names)})
    return turns


def _subset(records, per_circuit: int) -> List[int]:
    seen: collections.Counter = collections.Counter()
    picked = []
    for index, record in enumerate(records):
        if seen[record["circuit_id"]] < per_circuit:
            seen[record["circuit_id"]] += 1
            picked.append(index)
    return picked


def run_generate(args) -> None:
    import torch
    from transformers import AutoTokenizer

    records = load_records(args.limit)
    rows = _subset(records, args.per_circuit)
    out_path = args.out / f"generate_{args.model}.jsonl"
    done = set()
    if out_path.exists():
        done = {(json.loads(l)["row"], json.loads(l)["condition"]) for l in out_path.open()}
    if args.model == "text":
        tokenizer = AutoTokenizer.from_pretrained(args.text_adapter)
        model = load_text_model(args.text_adapter)
        cached = cached_messages()
        conditions = args.conditions or ["full", "ports", "none"]
    else:
        tokenizer = AutoTokenizer.from_pretrained(LLM)
        model, gate_funcs, vocab, netlist_view = load_stage_c_model(
            args.stage_c_checkpoint, args.stage_c_adapter)
        device = model.llm.get_input_embeddings().weight.device
        conditions = args.conditions or ["real", "zero"]
    stop = [t for t in (tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id)
            if isinstance(t, int) and t >= 0]
    started = time.time()
    with torch.no_grad(), out_path.open("a") as fh:
        for count, index in enumerate(rows):
            record = records[index]
            for condition in conditions:
                if (index, condition) in done:
                    continue
                if args.model == "text":
                    example = text_example(record, tokenizer, condition, cached[index])
                else:
                    example = stage_c_example(record, tokenizer, gate_funcs, vocab, netlist_view)
                    if example is None:
                        continue
                start, end = example["turns"][0]
                prompt = torch.tensor([example["input_ids"][:start]])
                budget = min(args.max_new_tokens, int(1.5 * (end - start)) + 64)
                if args.model == "text":
                    prompt = prompt.to(model.get_input_embeddings().weight.device)
                    output = model.generate(
                        input_ids=prompt, attention_mask=torch.ones_like(prompt),
                        max_new_tokens=budget, do_sample=False, eos_token_id=stop,
                        pad_token_id=tokenizer.pad_token_id,
                    )[0, prompt.size(1):]
                else:
                    prompt = prompt.to(device)
                    prefix = _graph_prefix(model, example["graph"], device)
                    if condition == "zero":
                        prefix = torch.zeros_like(prefix)
                    # The prompt ends before the first trained token, so it
                    # holds the placeholder.
                    embeds, mask = model._prefix_and_token_embeddings(
                        None, prompt, torch.ones_like(prompt), prefix=prefix,
                        graph_at=[example["graph_at"]])
                    output = model.llm.generate(
                        inputs_embeds=embeds,
                        attention_mask=mask,
                        max_new_tokens=budget, do_sample=False, eos_token_id=stop,
                        pad_token_id=tokenizer.pad_token_id,
                    )[0]
                text = tokenizer.decode(output, skip_special_tokens=True)
                fh.write(json.dumps({
                    "row": index, "condition": condition, "circuit_id": record["circuit_id"],
                    "fault": record["fault"], "difficulty": record.get("difficulty"),
                    "text": text, "new_tokens": int(output.numel()), "budget": budget,
                    "reference_tokens": end - start,
                }) + "\n")
                fh.flush()
            print(f"generate {args.model} {count + 1}/{len(rows)} {time.time() - started:.0f}s",
                  flush=True)


# ---------------------------------------------------------------------
# Scoring and report
# ---------------------------------------------------------------------

def _vector(value) -> Optional[Dict[str, Any]]:
    from atpgllm.multimodal.tool_probe import _bits

    return _bits(value)


def score_call(record, text) -> Dict[str, Any]:
    from atpgllm.training.revert_template import parse_tool_call
    from atpgllm.training.tool_feedback import read_snapshot
    from atpgllm.training.tool_trajectories import simulate

    out = {"parsed": False, "keys_exact": False, "missing_keys": None, "extra_keys": None,
           "care_bit_match": None, "detects": False, "sim_error": None}
    call = parse_tool_call(text)
    if not call or not isinstance(call.get("arguments"), dict):
        return out
    vector = _vector(call["arguments"].get("input_vector"))
    if not vector:
        return out
    out["parsed"] = True
    reference = _vector(record["input_vector"]) or {}
    keys, ref_keys = set(vector), set(reference)
    out["keys_exact"] = keys == ref_keys
    out["missing_keys"] = len(ref_keys - keys)
    out["extra_keys"] = len(keys - ref_keys)
    care = [k for k, v in reference.items() if str(v) in ("0", "1")]
    if care:
        out["care_bit_match"] = sum(str(vector.get(k)) == str(reference[k]) for k in care) / len(care)
    try:
        snapshot = simulate(record, vector)
        outputs = list(json.loads(record["expected_output"]))
        out["detects"] = bool(read_snapshot(snapshot, record["fault"], outputs).detected)
    except Exception as exc:  # an invalid call is a miss, as in tool_probe
        out["sim_error"] = f"{type(exc).__name__}: {exc}"[:200]
    return out


def _mean(values) -> Optional[float]:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _loss_rows(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.exists():
        return {}
    return {row["row"]: row for row in map(json.loads, path.open()) if not row.get("rejected")}


def loss_summary(text_rows, stage_rows) -> Dict[str, Any]:
    """Example-weighted mean NLL (mean over rows of each row's token mean)."""
    summary: Dict[str, Any] = {}

    def add(name, rows, condition, turn):
        whole, per_bucket = [], collections.defaultdict(list)
        for row in rows.values():
            turns = row["conditions"][condition]
            if turn >= len(turns):
                continue
            t = turns[turn]
            whole.append(_mean(t["nll"]))
            by = collections.defaultdict(list)
            for nll, bucket in zip(t["nll"], t["bucket"]):
                by[bucket].append(nll)
            for bucket, values in by.items():
                per_bucket[bucket].append(_mean(values))
        summary[name] = {"rows": len(whole), "mean_nll": _mean(whole),
                         **{f"{b}_nll": _mean(per_bucket[b]) for b in BUCKETS}}

    for condition in TEXT_CONDITIONS:
        if text_rows:
            add(f"text/{condition}/first_turn", text_rows, condition, 0)
            add(f"text/{condition}/final_turn", text_rows, condition, 1)
    for condition in STAGE_C_CONDITIONS:
        if stage_rows:
            add(f"stage_c/{condition}/first_turn", stage_rows, condition, 0)
            add(f"stage_c/{condition}/final_turn", stage_rows, condition, 1)

    # Shared tokens of the first turn, aligned by a diff of the token ids.
    common = sorted(set(text_rows) & set(stage_rows))
    for text_condition in TEXT_CONDITIONS:
        for stage_condition in ("real", "zero"):
            pairs = collections.defaultdict(lambda: ([], []))
            shares = []
            for index in common:
                a = text_rows[index]["conditions"][text_condition][0]
                b = stage_rows[index]["conditions"][stage_condition][0]
                matcher = difflib.SequenceMatcher(None, a["ids"], b["ids"], autojunk=False)
                by = collections.defaultdict(lambda: ([], []))
                matched = 0
                for block in matcher.get_matching_blocks():
                    for k in range(block.size):
                        bucket = b["bucket"][block.b + k]
                        by[bucket][0].append(a["nll"][block.a + k])
                        by[bucket][1].append(b["nll"][block.b + k])
                        by["all"][0].append(a["nll"][block.a + k])
                        by["all"][1].append(b["nll"][block.b + k])
                        matched += 1
                shares.append(matched / max(1, len(b["ids"])))
                for bucket, (x, y) in by.items():
                    pairs[bucket][0].append(_mean(x))
                    pairs[bucket][1].append(_mean(y))
            if common:
                summary[f"shared/text_{text_condition}_vs_stage_c_{stage_condition}"] = {
                    "rows": len(common), "shared_token_share": _mean(shares),
                    **{bucket: {"text": _mean(x), "stage_c": _mean(y)}
                       for bucket, (x, y) in pairs.items()},
                }
    # Trained-token mix of the Stage C first turn.
    counts = collections.Counter(
        bucket for row in stage_rows.values()
        for bucket in row["conditions"]["real"][0]["bucket"]
    )
    total = sum(counts.values())
    if total:
        summary["stage_c_first_turn_token_mix"] = {b: counts[b] / total for b in BUCKETS}
    return summary


def run_score(args) -> None:
    records = load_records(args.limit)
    scored_path = args.out / "scored.jsonl"
    done = set()
    if scored_path.exists():
        done = {(r["model"], r["row"], r["condition"]) for r in map(json.loads, scored_path.open())}
    with scored_path.open("a") as fh:
        for model in ("text", "stage_c"):
            path = args.out / f"generate_{model}.jsonl"
            if not path.exists():
                continue
            for generated in map(json.loads, path.open()):
                key = (model, generated["row"], generated["condition"])
                if key in done:
                    continue
                result = score_call(records[generated["row"]], generated["text"])
                fh.write(json.dumps({"model": model, "row": generated["row"],
                                     "condition": generated["condition"],
                                     "difficulty": generated.get("difficulty"),
                                     "truncated": generated["new_tokens"] >= generated["budget"],
                                     **result}) + "\n")
                fh.flush()

    scored = list(map(json.loads, scored_path.open())) if scored_path.exists() else []
    task: Dict[str, Any] = {}
    groups = collections.defaultdict(list)
    for row in scored:
        groups[f"{row['model']}/{row['condition']}"].append(row)
        groups[f"{row['model']}/{row['condition']}/{row.get('difficulty')}"].append(row)
    for name, rows in sorted(groups.items()):
        task[name] = {
            "rows": len(rows),
            "parsed": _mean([float(r["parsed"]) for r in rows]),
            "keys_exact": _mean([float(r["keys_exact"]) for r in rows]),
            "care_bit_match": _mean([r["care_bit_match"] for r in rows]),
            "first_try_detects": _mean([float(r["detects"]) for r in rows]),
            "truncated": _mean([float(r["truncated"]) for r in rows]),
        }
    losses = loss_summary(_loss_rows(args.out / "loss_text.jsonl"),
                          _loss_rows(args.out / "loss_stage_c.jsonl"))
    report = {"loss": losses, "task": task}
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    (args.out / "report.md").write_text(_markdown(report))
    print(_markdown(report))


def _fmt(value) -> str:
    return "" if value is None else f"{value:.4f}"


def _markdown(report) -> str:
    lines = ["# Stage C v4 vs text SFT v4", "", "## Loss (mean NLL per example)", "",
             "| run / condition / turn | rows | all | port | instance | net | value | other |",
             "|---|---|---|---|---|---|---|---|"]
    for name, row in report["loss"].items():
        if name.startswith(("text/", "stage_c/")):
            lines.append(f"| {name} | {row['rows']} | {_fmt(row['mean_nll'])} | "
                         + " | ".join(_fmt(row[f'{b}_nll']) for b in BUCKETS) + " |")
    lines += ["", "## First turn, tokens both sequences share", "",
              "| comparison | rows | shared | all text / stage C | port | value | other |",
              "|---|---|---|---|---|---|---|"]
    for name, row in report["loss"].items():
        if name.startswith("shared/"):
            cell = lambda b: (f"{_fmt(row[b]['text'])} / {_fmt(row[b]['stage_c'])}"
                              if b in row else "")
            lines.append(f"| {name[7:]} | {row['rows']} | {_fmt(row['shared_token_share'])} | "
                         f"{cell('all')} | {cell('port')} | {cell('value')} | {cell('other')} |")
    if "stage_c_first_turn_token_mix" in report["loss"]:
        mix = report["loss"]["stage_c_first_turn_token_mix"]
        lines += ["", "Stage C first-turn trained-token mix: "
                  + ", ".join(f"{b} {mix[b]:.3f}" for b in BUCKETS)]
    lines += ["", "## Greedy first tool call (TetraMAX)", "",
              "| model / condition | rows | parsed | PI keys exact | care bits | detects | truncated |",
              "|---|---|---|---|---|---|---|"]
    for name, row in report["task"].items():
        lines.append(f"| {name} | {row['rows']} | {_fmt(row['parsed'])} | {_fmt(row['keys_exact'])} | "
                     f"{_fmt(row['care_bit_match'])} | {_fmt(row['first_try_detects'])} | "
                     f"{_fmt(row['truncated'])} |")
    return "\n".join(lines) + "\n"


def run_check(args) -> None:
    """CPU: rows resolve, the text tokenization reproduces the text SFT's
    cached validation messages, and the token tagging runs."""
    from transformers import AutoTokenizer

    from atpgllm.graph.gate_features import GateAttributeVocab
    from atpgllm.training.dataset_utils import _format_sft_messages_record, tool_vectors_as_text

    records = load_records(args.limit)
    cached = json.loads((TEXT_RUN / "sft_validation_examples.json").read_text())["rows"]
    tokenizer = AutoTokenizer.from_pretrained(args.text_adapter)
    graph_tokenizer = AutoTokenizer.from_pretrained(LLM)
    gate_funcs = json.loads(SIM_CONFIG.read_text())["gate_funcs"]
    vocab = GateAttributeVocab(gate_funcs)
    same = 0
    for index, record in enumerate(records):
        mine = _format_sft_messages_record(copy.deepcopy(record), True, tool_vectors_as_text(tokenizer))
        ids = tokenizer.apply_chat_template(mine["messages"], tools=mine["tools"], tokenize=True)
        theirs = tokenizer.apply_chat_template(cached[index]["messages"], tools=mine["tools"],
                                               tokenize=True)
        unwrap = lambda out: list(out["input_ids"] if hasattr(out, "keys") else out)
        same += unwrap(ids) == unwrap(theirs)
        for condition in TEXT_CONDITIONS:
            text_example(record, tokenizer, condition, cached[index]["messages"])
        example = stage_c_example(record, graph_tokenizer, gate_funcs, vocab)
        if example is not None and index < 3:
            start, end = example["turns"][0]
            pieces = [graph_tokenizer.decode([i]) for i in example["input_ids"][start:end]]
            print(collections.Counter(tag_tokens(pieces, netlist_names(netlist_text(record)))))
    print(f"{same}/{len(records)} rows tokenize exactly as the text SFT's cached validation "
          "(tool-call arguments are reordered there; 'cached' scores those)")
    print(port_declarations(netlist_text(records[0]))[:600])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pass_", metavar="pass", choices=("check", "loss", "generate", "score"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", choices=("text", "stage_c"))
    parser.add_argument("--text-adapter", type=Path, default=TEXT_RUN / "checkpoint-250")
    parser.add_argument("--stage-c-checkpoint", type=Path,
                        default=STAGE_C_RUN / "multimodal_sft_best.pt")
    parser.add_argument("--stage-c-adapter", type=Path, default=STAGE_C_RUN / "adapter_best")
    parser.add_argument("--limit", type=int, default=None, help="First N validation rows only.")
    parser.add_argument("--per-circuit", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--conditions", nargs="*")
    args = parser.parse_args()
    if args.pass_ in ("loss", "generate") and not args.model:
        parser.error(f"{args.pass_} needs --model")
    args.out.mkdir(parents=True, exist_ok=True)
    {"check": run_check, "loss": run_loss, "generate": run_generate, "score": run_score}[args.pass_](args)


if __name__ == "__main__":
    main()
