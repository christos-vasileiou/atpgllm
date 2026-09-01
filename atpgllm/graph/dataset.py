from __future__ import annotations

"""
HuggingFace dataset adapter for ``chrivasileiou/asap7-language-of-test``.

Produces batches consumable by:

- :class:`Stage1Trainer` (graph, caption) pairs for contrastive +
  matching + generative pre-training.
- :class:`Stage2GraphTextLM` (graph, prompt, answer) triples for
  soft-prompt LLM fine-tuning.

Placeholder rendering
---------------------
Raw dataset fields (``user_content``, ``reasoning_content``,
``answer_content``) contain Python-style ``{placeholder}`` tokens
(``{module_name}``, ``{fault_net}``, ``{fault_model_long}``,
``{propagation_gates}``, ``{primary_observation_nets}``,
``{expected_output}``, ``{input_vector}``, ``{detected_faults}``, ...)
that must be resolved before the text is usable for training.

The resolution logic (including fault-string parsing, JSON-dict compaction,
SCOAP-style signal grouping, and step-wise reasoning template rendering) lives
in :class:`atpgllm.training.conversation.ConversationExample`. A legacy
``tests_dir`` override remains only for old external checkouts.
"""

import copy
import hashlib
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import IterableDataset
from torch_geometric.data import Batch, Data

from .fault_context import attach_atpg_context
from .gate_features import GateAttributeVocab
from .netlist_parser import (
    parse_verilog_to_graph,
    parse_verilog_to_pyg,
    parsed_to_pyg,
)


# ---------------------------------------------------------------------
# ConversationExample import (lazy, cached)
# ---------------------------------------------------------------------


_CONVERSATION_EXAMPLE = None
# atpgllm/graph/dataset.py → parents[2] == libatpgllm package root
_DEFAULT_TESTS_DIR = Path(__file__).resolve().parents[2] / "tests"


def _resolve_tests_dir(tests_dir: Optional[Path] = None) -> Path:
    """Locate ``libatpgllm/tests`` for ConversationExample imports."""
    if tests_dir is not None:
        return Path(tests_dir).resolve()
    for env_key in ("ATPGLLM_TESTS_DIR", "GRAPH_TEXT_TESTS_DIR"):
        env_val = os.environ.get(env_key)
        if env_val and str(env_val).strip():
            return Path(env_val).expanduser().resolve()
    return _DEFAULT_TESTS_DIR.resolve()


def _get_conversation_example(tests_dir: Optional[Path] = None):
    """Return the cached ``ConversationExample`` class.

    On first call, ensures the tests directory is on ``sys.path`` and
    imports ``ConversationExample`` from ``conversation.py``. Subsequent
    calls reuse the cached class.
    """
    global _CONVERSATION_EXAMPLE
    if _CONVERSATION_EXAMPLE is not None:
        return _CONVERSATION_EXAMPLE

    if tests_dir is None:
        from atpgllm.training.conversation import ConversationExample

        _CONVERSATION_EXAMPLE = ConversationExample
        return _CONVERSATION_EXAMPLE

    tests_path = _resolve_tests_dir(tests_dir)
    if not (tests_path / "conversation.py").exists():
        raise FileNotFoundError(
            f"conversation.py not found under {tests_path}. "
            "Set ATPGLLM_TESTS_DIR or pass tests_dir= to locate "
            "libatpgllm/tests."
        )
    if str(tests_path) not in sys.path:
        sys.path.insert(0, str(tests_path))

    from conversation import ConversationExample  # type: ignore  # noqa: E402

    _CONVERSATION_EXAMPLE = ConversationExample
    return _CONVERSATION_EXAMPLE


# ---------------------------------------------------------------------
# Rendered messages helper
# ---------------------------------------------------------------------


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


@dataclass
class RenderedRecord:
    """Result of rendering one dataset record.

    All ``{placeholder}`` tokens are resolved. The netlist field is
    preserved as the original Verilog *string* (``from_record`` rewrites
    it in-place into a ``{doc_id, netlist}`` dict, so we snapshot it
    first).
    """
    messages: List[Dict[str, str]]
    system: str
    user: str
    reasoning: str   # inside <think>...</think>, without the wrapper
    answer: str      # final assistant answer (without the lead-in sentence)
    netlist: str
    module_name: str
    fault: str


def render_record(
    record: Dict[str, Any],
    tests_dir: Optional[Path] = None,
) -> RenderedRecord:
    """Render placeholders in a record via ``ConversationExample.from_record``.

    Works on a deep copy so the caller's record is not mutated.
    """
    ConversationExample = _get_conversation_example(tests_dir)

    original_netlist = record.get("netlist", "") or ""
    rec_copy = copy.deepcopy(record)

    example = ConversationExample.from_record(rec_copy)
    messages = list(example.messages)

    system = next(
        (m["content"] for m in messages if m["role"] == "system"), ""
    )
    user = next(
        (m["content"] for m in messages if m["role"] == "user"), ""
    )

    assistant_msgs = [m for m in messages if m["role"] == "assistant"]
    reasoning = ""
    answer = ""
    if assistant_msgs:
        first_asst = assistant_msgs[0]["content"]
        m = _THINK_RE.search(first_asst)
        if m:
            reasoning = m.group(1).strip()
    if len(assistant_msgs) > 1:
        answer = assistant_msgs[-1]["content"]

    return RenderedRecord(
        messages=messages,
        system=system,
        user=user,
        reasoning=reasoning,
        answer=answer,
        netlist=original_netlist,
        module_name=str(record.get("module_name", "") or ""),
        fault=str(record.get("fault", "") or ""),
    )


def render_prompt_and_answer(
    record: Dict[str, Any],
    tokenizer,
    tests_dir: Optional[Path] = None,
    *,
    compact_netlist: bool = False,
) -> tuple[str, str]:
    """Render the exact system/user prompt and assistant target used by SFT."""
    rendered = render_record(record, tests_dir=tests_dir)
    prompt_messages = [
        dict(message)
        for message in rendered.messages
        if message["role"] in ("system", "user")
    ]
    if compact_netlist and rendered.netlist:
        doc_id = hashlib.sha256(rendered.netlist.encode("utf-8")).hexdigest()[:16]
        full_payload = {"doc_id": doc_id, "netlist": rendered.netlist}
        compact_payload = {"doc_id": doc_id, "netlist": "<GRAPH_CONTEXT>"}
        for message in prompt_messages:
            message["content"] = message["content"].replace(
                str(full_payload),
                str(compact_payload),
            )
    assistant_messages = [
        message
        for message in rendered.messages
        if message["role"] == "assistant"
    ]
    answer = "\n".join(m["content"] for m in assistant_messages).strip()

    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = "\n".join(
            [f"<|{m['role']}|>\n{m['content']}" for m in prompt_messages]
            + ["<|assistant|>\n"]
        )
    return prompt, answer


# ---------------------------------------------------------------------
# Caption / text synthesis
# ---------------------------------------------------------------------


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " ..."


def make_text_caption(
    record: Dict[str, Any],
    max_words: int = 160,
    tests_dir: Optional[Path] = None,
) -> str:
    """Natural-language caption for contrastive Stage-1 pairing.

    Builds a short paragraph of the form::

        Module <module>. Target fault: <fault>. Reasoning: <rendered CoT>.

    The caption is derived from :func:`render_record` so every
    ``{placeholder}`` in ``reasoning_content`` is properly substituted
    (``{fault_net}``, ``{fault_model_long}``, ``{propagation_gates}``,
    ``{primary_observation_nets}``, …). The raw netlist is *never*
    embedded here — that goes to the graph encoder.
    """
    rendered = render_record(record, tests_dir=tests_dir)
    header = f"Module {rendered.module_name}. Target fault: {rendered.fault}."
    body = rendered.reasoning or rendered.answer
    body = _truncate_words(body, max_words)
    return f"{header} {body}".strip()


# ---------------------------------------------------------------------
# Record → PyG graph
# ---------------------------------------------------------------------


def record_to_pyg(
    record: Dict[str, Any],
    gate_funcs: Dict[str, Dict[str, str]],
    vocab: GateAttributeVocab,
    min_nodes: int = 1,
) -> Optional[Data]:
    """Parse a dataset record's ``netlist`` to a PyG ``Data``.

    Returns ``None`` if the netlist field is missing, empty, or parses
    to fewer than ``min_nodes`` gates (degenerate).
    """
    netlist = record.get("netlist")
    # ``ConversationExample.from_record`` rewrites the netlist field
    # in-place into ``{"doc_id": ..., "netlist": "..."}`` after rendering.
    # Handle both forms so this helper is order-independent.
    if isinstance(netlist, dict):
        netlist = netlist.get("netlist")
    if not netlist:
        return None
    try:
        parsed = parse_verilog_to_graph(netlist, gate_funcs)
        data = parsed_to_pyg(parsed, vocab=vocab)
        attach_atpg_context(data, parsed, gate_funcs, record)
    except Exception:
        return None
    if data.num_nodes < min_nodes:
        return None
    return data


# ---------------------------------------------------------------------
# Target-fault graph pretraining dataset
# ---------------------------------------------------------------------


class ASAP7GraphPretrainDataset(IterableDataset):
    """Stream target-fault-conditioned graphs with ATPG node labels."""

    def __init__(
        self,
        hf_stream: Iterable[Dict[str, Any]],
        gate_funcs: Dict[str, Dict[str, str]],
        vocab: GateAttributeVocab,
        *,
        skip_invalid: bool = True,
    ) -> None:
        super().__init__()
        self.hf_stream = hf_stream
        self.gate_funcs = gate_funcs
        self.vocab = vocab
        self.skip_invalid = skip_invalid

    def __iter__(self) -> Iterator[Data]:
        for record in self.hf_stream:
            graph = record_to_pyg(record, self.gate_funcs, self.vocab)
            if graph is None:
                if self.skip_invalid:
                    continue
                raise ValueError(
                    f"Failed to parse record: module={record.get('module_name')}"
                )
            if not (
                bool(graph.has_propagation_labels.item())
                or bool(graph.has_backtrack_labels.item())
                or bool(graph.has_discrepancy_labels.item())
            ):
                if self.skip_invalid:
                    continue
                raise ValueError("Record has no graph-pretraining labels.")
            yield graph


def collate_graph_pretrain_batch(items: List[Data]) -> Batch:
    if not items:
        raise ValueError("Cannot collate an empty graph-pretraining batch.")
    return Batch.from_data_list(items)


# ---------------------------------------------------------------------
# Stage-1 streaming dataset
# ---------------------------------------------------------------------


class ASAP7GraphTextDataset(IterableDataset):
    """Stream (graph, caption, answer) triples from the HF dataset.

    Parameters
    ----------
    hf_stream : iterable of dict
        A HuggingFace streaming dataset (or any iterable yielding
        record dicts with the ASAP7 fields).
    gate_funcs : dict
        Gate function library (``sim_config.json["gate_funcs"]``).
    vocab : GateAttributeVocab
        Attribute vocab used by the graph encoder.
    tokenizer : transformers.PreTrainedTokenizerBase
        Tokenizer used for the text encoder path *and* the GTG decoder
        target (so the vocab is shared).
    max_text_len : int
        Max caption length after tokenisation.
    max_answer_len : int
        Max length used for the Stage-1 GTG decoder target.
    caption_words : int
        Word budget for the synthesised caption (see
        :func:`make_text_caption`).
    skip_invalid : bool
        If True, silently drop records whose netlist is missing or
        fails to parse. If False, raise.
    tests_dir : Path, optional
        Override for the ``libatpgllm/tests`` directory containing
        ``conversation.py`` (used for placeholder rendering).
    """

    def __init__(
        self,
        hf_stream: Iterable[Dict[str, Any]],
        gate_funcs: Dict[str, Dict[str, str]],
        vocab: GateAttributeVocab,
        tokenizer,
        max_text_len: int = 256,
        max_answer_len: int = 256,
        caption_words: int = 4096,
        skip_invalid: bool = True,
        tests_dir: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self.hf_stream = hf_stream
        self.gate_funcs = gate_funcs
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.max_answer_len = max_answer_len
        self.caption_words = caption_words
        self.skip_invalid = skip_invalid
        self.tests_dir = tests_dir

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for record in self.hf_stream:
            data = record_to_pyg(record, self.gate_funcs, self.vocab)
            if data is None:
                if self.skip_invalid:
                    continue
                raise ValueError(
                    f"Failed to parse record: module={record.get('module_name')}"
                )

            try:
                rendered = render_record(record, tests_dir=self.tests_dir)
            except Exception:
                if self.skip_invalid:
                    continue
                raise

            caption = _truncate_words(
                f"Module {rendered.module_name}. Target fault: {rendered.fault}. "
                f"{rendered.reasoning or rendered.answer}",
                self.caption_words,
            )

            yield {
                "graph": data,
                "caption": caption,
                "answer": rendered.answer,
                # Rendered conversation pieces (all placeholders resolved)
                "rendered_messages": rendered.messages,
                "system_content": rendered.system,
                "user_content": rendered.user,
                "reasoning_content": rendered.reasoning,
                "answer_content": rendered.answer,
                "module_name": rendered.module_name,
                "fault": rendered.fault,
            }


# ---------------------------------------------------------------------
# Per-design dataset (Stage 1, BRIDGES-faithful)
#
# Reads a precomputed ``design_descriptions.json`` produced by
# ``atpgllm.graph.scripts.precompute_design_descriptions`` and yields one
# ``(graph, description)`` pair per *unique netlist* (de-duplicated by
# sha1). This is the "one graph → one description" discipline the paper
# requires: GTC, GTM, GTG are well-posed because the same netlist never
# appears with two different texts in a batch.
# ---------------------------------------------------------------------


class ASAP7DesignDataset(IterableDataset):
    """Iterable dataset of ``(graph, description)`` pairs for Stage 1.

    Loads the precomputed JSON (one entry per unique netlist hash) once
    at construction time, parses every netlist into a PyG ``Data``
    eagerly (so each iteration step is a dict lookup, not a regex
    parse), and then iterates the design list indefinitely with
    per-epoch shuffling.

    Yields the same dict schema as :class:`ASAP7GraphTextDataset` so the
    existing :func:`collate_graph_text_batch` works unchanged: ``caption``
    drives the text encoder (GTC / GTM input), ``answer`` drives the GTG
    decoder target. For Stage 1, both are the rendered design
    description.

    Parameters
    ----------
    descriptions_path : Path
        Path to ``design_descriptions.json``.
    gate_funcs : dict
        Gate function library (``sim_config.json["gate_funcs"]``).
    vocab : GateAttributeVocab
        Attribute vocab used by the graph encoder.
    tokenizer : transformers.PreTrainedTokenizerBase
        Kept for API parity (the collator does the tokenisation).
    shuffle : bool
        Re-shuffle design order each epoch (default True).
    seed : int
        Seed for the per-worker RNG.
    repeat : bool
        If True (default), iterate forever; if False, stop after one
        full pass over the design list.
    skip_invalid : bool
        Drop designs whose netlist re-parses to fewer than ``min_nodes``
        gates (degenerate). Should never fire if the precompute script
        already filtered them.
    min_nodes : int
        Minimum gate count for a design to be kept.
    """

    def __init__(
        self,
        descriptions_path: Path,
        gate_funcs: Dict[str, Dict[str, str]],
        vocab: GateAttributeVocab,
        tokenizer,
        shuffle: bool = True,
        seed: int = 42,
        repeat: bool = True,
        skip_invalid: bool = True,
        min_nodes: int = 1,
    ) -> None:
        super().__init__()
        self.descriptions_path = Path(descriptions_path)
        self.gate_funcs = gate_funcs
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.shuffle = shuffle
        self.seed = int(seed)
        self.repeat = repeat

        if not self.descriptions_path.is_file():
            raise FileNotFoundError(
                f"design_descriptions.json not found at {self.descriptions_path}. "
                f"Run: python -m atpgllm.graph.scripts.precompute_design_descriptions"
            )
        with self.descriptions_path.open("r", encoding="utf-8") as fh:
            raw: Dict[str, Dict[str, Any]] = json.load(fh)

        # Eager-parse every netlist once. The parsed graphs are small
        # (≤ a few thousand nodes); 374 of them comfortably fit in RAM.
        self._items: List[Tuple[str, Data, str, str]] = []
        n_dropped = 0
        for h, item in raw.items():
            netlist = item.get("netlist") or ""
            description = item.get("description") or ""
            module_name = str(item.get("module_name") or "")
            if not netlist or not description:
                n_dropped += 1
                continue
            try:
                data = parse_verilog_to_pyg(netlist, gate_funcs, vocab=vocab)
            except Exception:
                if not skip_invalid:
                    raise
                n_dropped += 1
                continue
            if data.num_nodes < min_nodes:
                n_dropped += 1
                continue
            self._items.append((h, data, description, module_name))

        if not self._items:
            raise RuntimeError(
                f"ASAP7DesignDataset is empty: 0 valid designs in "
                f"{self.descriptions_path} (dropped {n_dropped})."
            )
        if n_dropped:
            print(
                f"ASAP7DesignDataset: loaded {len(self._items)} designs, "
                f"dropped {n_dropped}.",
                flush=True,
            )

    def __len__(self) -> int:
        # Defined for diagnostics; `IterableDataset` doesn't *require* it.
        return len(self._items)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        # Per-worker seed so multiple DataLoader workers don't collide
        # on the same shuffle order.
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        rng = random.Random(self.seed + 7919 * worker_id)
        order = list(range(len(self._items)))

        while True:
            if self.shuffle:
                rng.shuffle(order)
            for idx in order:
                h, graph, description, module_name = self._items[idx]
                yield {
                    "graph": graph,
                    "caption": description,
                    "answer": description,
                    # Kept for shape parity with ``ASAP7GraphTextDataset``;
                    # the collator does not consume these fields.
                    "rendered_messages": [],
                    "system_content": "",
                    "user_content": "",
                    "reasoning_content": "",
                    "answer_content": description,
                    "module_name": module_name,
                    "fault": "",
                    "netlist_hash": h,
                }
            if not self.repeat:
                return


# ---------------------------------------------------------------------
# Collation: PyG graphs + tokenised caption + decoder inputs
# ---------------------------------------------------------------------


def collate_graph_text_batch(
    items: List[Dict[str, Any]],
    tokenizer,
    max_text_len: int = 256,
    max_answer_len: int = 256,
) -> Dict[str, Any]:
    """Collate Stage-1 examples into trainer-compatible tensors.

    The returned dict matches the keys expected by :meth:`Stage1Trainer.train_step`:

    - ``g``: batched PyG ``Data`` object (via ``torch_geometric.data.Batch``)
    - ``input_ids`` / ``attention_mask``: caption tokens [B, T]
    - ``decoder_input_ids`` / ``decoder_attention_mask`` / ``decoder_labels``:
      teacher-forced tokens for GTG [B, T_dec]
    """
    graphs = [ex["graph"] for ex in items]
    captions = [ex["caption"] for ex in items]
    answers = [ex["answer"] or ex["caption"] for ex in items]

    g_batch = Batch.from_data_list(graphs)

    enc = tokenizer(
        captions,
        max_length=max_text_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    dec = tokenizer(
        answers,
        max_length=max_answer_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    dec_labels = dec["input_ids"].clone()
    # Mask padding positions so they don't contribute to the GTG loss.
    if tokenizer.pad_token_id is not None:
        dec_labels[dec_labels == tokenizer.pad_token_id] = -100

    return {
        "g": g_batch,
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "decoder_input_ids": dec["input_ids"],
        "decoder_attention_mask": dec["attention_mask"],
        "decoder_labels": dec_labels,
    }
