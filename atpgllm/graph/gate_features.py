"""
Gate attribute extraction and vocabulary for ASAP7-style cell libraries.

Decomposes each cell type into ~8 categorical/binary attributes instead of
learning one monolithic embedding per cell name.  This gives:

- **Parameter sharing**: ``AND2x2`` and ``AND2x4`` share logic_family,
  input_count, complemented, sequential embeddings — only drive_strength
  differs.
- **Compositional generalisation**: unseen gate combinations still get
  meaningful embeddings from their constituent attributes.
- **Library portability**: attributes transfer across PDK variants
  (ASAP7 RVT→LVT, or entirely different technology nodes).

Usage::

    vocab = GateAttributeVocab.from_sim_config("sim_config.json")
    indices = vocab.encode("AND2x2_ASAP7_75t_R")  # → (idx0, idx1, ..., idx7)
    vocab.vocab_sizes  # dict of vocabulary sizes per attribute

No PyTorch dependency — pure Python / regex parsing.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# =====================================================================
# Attribute vocabularies
# =====================================================================

LOGIC_FAMILIES: List[str] = [
    "UNKNOWN",
    "AND", "OR", "NAND", "NOR", "XOR", "XNOR",
    "BUF", "INV",
    "AO", "AOI", "OA", "OAI",
    "FA", "HA", "MAJ",
    "DFF", "LATCH", "ICG", "SCAN_FF",
    "TIE",
    "COMPOUND_AOI", "COMPOUND_OAI",
]

INPUT_COUNT_CLASSES: List[str] = [
    "UNKNOWN", "1", "2", "3", "4", "5", "6_PLUS",
]

DRIVE_STRENGTH_CLASSES: List[str] = [
    "UNKNOWN",      # fallback
    "TINY",         # < 0.5
    "SMALL",        # [0.5, 1.0)
    "X1",           # [1.0, 2.0)
    "X2",           # [2.0, 3.0)
    "X3",           # [3.0, 4.0)
    "X4",           # [4.0, 6.0)
    "X6",           # [6.0, 8.0)
    "X8",           # [8.0, 12.0)
    "X12",          # [12.0, 16.0)
    "X16_PLUS",     # >= 16.0
]

NUM_OUTPUT_CLASSES: List[str] = ["1", "2", "SPECIAL"]

# ---------------------------------------------------------------------
# SINGLE SOURCE OF TRUTH for which gate attributes are used by the model.
#
# To disable an attribute for debugging / ablation, simply comment out
# its name below.  Everything else (vocab sizes, encode(), node encoder
# embedding list) derives from this list — no other edits required.
#
# All attributes remain populated on ``GateAttributes`` dataclass
# instances so ``vocab.get_attrs(cell).<field>`` lookups keep working
# regardless of which ones are active.
# ---------------------------------------------------------------------
ATTRIBUTE_NAMES: List[str] = [
    "logic_family",
    "input_count",
    # "output_complemented",
    "is_sequential",
    # "drive_strength",
    "num_outputs",
    # "tristate",
    # "is_clock_related",
]
NUM_ATTRIBUTES = len(ATTRIBUTE_NAMES)

# =====================================================================
# Cell name → family prefix mapping
# =====================================================================

# Maps the *prefix* portion of a cell basename (before the ``x<drive>``
# suffix) to a canonical logic family.  Exact-match entries are tried
# first; if no match, trailing digits are stripped and retried.
_FAMILY_PREFIX_MAP: Dict[str, str] = {
    # Compound AO/OA-Invert
    "A2O1A1O1I": "COMPOUND_AOI",
    "A2O1A1I":   "COMPOUND_AOI",
    "O2A1O1I":   "COMPOUND_OAI",
    # AO / OA families (prefix match after digit stripping)
    "AOI": "AOI", "AO": "AO",
    "OAI": "OAI", "OA": "OA",
    # Sequential elements
    "DFFASRHQN": "DFF", "DFFHQN": "DFF", "DFFLQN": "DFF",
    "DFFHQ": "DFF", "DFFLQ": "DFF",
    "DHL": "LATCH", "DLL": "LATCH",
    "ICG": "ICG",
    "SDFH": "SCAN_FF", "SDFL": "SCAN_FF",
    # Clock / buffer
    "CKINVDC": "INV",
    "HB": "BUF", "BUF": "BUF", "INV": "INV",
    # Basic logic (match after digit stripping: NAND2→NAND)
    "NAND": "NAND", "NOR": "NOR", "AND": "AND", "OR": "OR",
    "XNOR": "XNOR", "XOR": "XOR",
    # Arithmetic
    "MAJI": "MAJ", "MAJ": "MAJ", "FA": "FA", "HA": "HA",
    # Tie cells
    "TIEHI": "TIE", "TIELO": "TIE",
}

# Families whose primary output is complemented (inverted)
_COMPLEMENTED_FAMILIES = frozenset({
    "AOI", "OAI", "INV", "NAND", "NOR", "XNOR",
    "COMPOUND_AOI", "COMPOUND_OAI",
})

# Sequential families
_SEQUENTIAL_FAMILIES = frozenset({"DFF", "LATCH", "ICG", "SCAN_FF"})

# Clock-related prefixes (before family classification)
_CLOCK_PREFIXES = frozenset({"CKINVDC", "ICG"})

# =====================================================================
# Internal non-input identifiers in boolean functions
# =====================================================================

_NON_INPUT_VARS = frozenset({"IQ", "IQN", "VDD", "VSS"})

# =====================================================================
# Parsing helpers
# =====================================================================

# Splits a cell basename into (prefix, drive_string, variant_suffix).
# Non-greedy prefix finds the first valid ``x<digits>`` split point.
_BASENAME_RE = re.compile(r"^(.+?)(x\d*(?:p\d+)?)(.*?)$")

# Extracts variable names from a boolean function string.
# Matches identifiers like A, B1, CI, C2 but not operators/keywords.
_VAR_RE = re.compile(r"\b([A-Z][A-Z0-9]*)\b")


def _strip_tech_suffix(cell_name: str) -> str:
    """Remove technology suffix like ``_ASAP7_75t_R``."""
    idx = cell_name.find("_ASAP7")
    if idx >= 0:
        return cell_name[:idx]
    # Fallback: strip from the last three underscores that look like a tech ID
    parts = cell_name.rsplit("_", 3)
    return parts[0] if len(parts) >= 4 else cell_name


def _parse_drive_value(drive_str: str) -> float:
    """Parse ASAP7 drive strength string to float.

    ``x2`` → 2.0, ``xp33`` → 0.33, ``x1p5`` → 1.5, ``x2p67DC`` → 2.67
    """
    cleaned = re.sub(r"[a-zA-Z]+$", "", drive_str)
    m = re.match(r"x(\d*)(?:p(\d+))?", cleaned)
    if not m:
        return 1.0
    int_part = m.group(1) or "0"
    frac_part = m.group(2)
    if frac_part:
        return float(f"{int_part}.{frac_part}")
    return float(int_part) if int_part != "0" else 1.0


def _classify_family(prefix: str) -> str:
    """Map cell prefix to canonical logic family name."""
    if prefix in _FAMILY_PREFIX_MAP:
        return _FAMILY_PREFIX_MAP[prefix]
    base = re.sub(r"\d+$", "", prefix)
    if base in _FAMILY_PREFIX_MAP:
        return _FAMILY_PREFIX_MAP[base]
    return "UNKNOWN"


def _classify_drive(val: float) -> str:
    if val < 0.5:
        return "TINY"
    if val < 1.0:
        return "SMALL"
    if val < 2.0:
        return "X1"
    if val < 3.0:
        return "X2"
    if val < 4.0:
        return "X3"
    if val < 6.0:
        return "X4"
    if val < 8.0:
        return "X6"
    if val < 12.0:
        return "X8"
    if val < 16.0:
        return "X12"
    return "X16_PLUS"


def _count_inputs_from_function(func_dict: Dict[str, str]) -> int:
    """Count unique input variable names across all output functions."""
    all_vars: set[str] = set()
    for _pin, expr in func_dict.items():
        for m in _VAR_RE.finditer(expr):
            var = m.group(1)
            if var not in _NON_INPUT_VARS:
                all_vars.add(var)
    return len(all_vars)


def _heuristic_input_count(family: str, prefix: str) -> int:
    """Heuristic input count for sequential / special cells whose
    boolean function only references internal state (IQ/IQN)."""
    if family == "TIE":
        return 0
    if family == "SCAN_FF":
        return 4  # D, SI, SE, CLK
    if family == "DFF":
        if "ASRHQN" in prefix.upper():
            return 4  # D, CLK, SET, RESET
        return 2  # D, CLK
    if family == "LATCH":
        return 2  # D, E
    if family == "ICG":
        return 2  # CLK, EN
    return 1


def _input_count_to_class(n: int) -> str:
    if n <= 0:
        return "UNKNOWN"
    if n >= 6:
        return "6_PLUS"
    return str(n)


# =====================================================================
# Attribute registry: vocab_size + per-attribute index computation.
# Add a new attribute here (and in GateAttributes / _extract below) to
# make it togglable via ATTRIBUTE_NAMES.
# =====================================================================


def _idx_logic_family(a: "GateAttributes") -> int:
    return LOGIC_FAMILIES.index(a.logic_family) if a.logic_family in LOGIC_FAMILIES else 0


def _idx_input_count(a: "GateAttributes") -> int:
    cls = _input_count_to_class(a.input_count)
    return INPUT_COUNT_CLASSES.index(cls) if cls in INPUT_COUNT_CLASSES else 0


def _idx_drive_strength(a: "GateAttributes") -> int:
    cls = _classify_drive(a.drive_strength)
    return DRIVE_STRENGTH_CLASSES.index(cls) if cls in DRIVE_STRENGTH_CLASSES else 0


def _idx_num_outputs(a: "GateAttributes") -> int:
    cls = "SPECIAL" if a.num_outputs == 0 else str(min(a.num_outputs, 2))
    return NUM_OUTPUT_CLASSES.index(cls) if cls in NUM_OUTPUT_CLASSES else 0


# name -> (vocab_size, GateAttributes -> int index)
_ATTRIBUTE_REGISTRY: Dict[str, Tuple[int, "callable"]] = {
    "logic_family":        (len(LOGIC_FAMILIES),          _idx_logic_family),
    "input_count":         (len(INPUT_COUNT_CLASSES),     _idx_input_count),
    "output_complemented": (2,                            lambda a: int(a.output_complemented)),
    "is_sequential":       (2,                            lambda a: int(a.is_sequential)),
    "drive_strength":      (len(DRIVE_STRENGTH_CLASSES),  _idx_drive_strength),
    "num_outputs":         (len(NUM_OUTPUT_CLASSES),      _idx_num_outputs),
    "tristate":            (2,                            lambda a: int(a.tristate)),
    "is_clock_related":    (2,                            lambda a: int(a.is_clock_related)),
}

# Parallel registry: index -> human-readable label per attribute.
# Used by ``GateAttributeVocab.describe`` for debugging / inspection.
_ATTRIBUTE_LABELS: Dict[str, List[str]] = {
    "logic_family":        LOGIC_FAMILIES,
    "input_count":         INPUT_COUNT_CLASSES,
    "output_complemented": ["False", "True"],
    "is_sequential":       ["False", "True"],
    "drive_strength":      DRIVE_STRENGTH_CLASSES,
    "num_outputs":         NUM_OUTPUT_CLASSES,
    "tristate":            ["False", "True"],
    "is_clock_related":    ["False", "True"],
}

# Sanity check: every name listed in ATTRIBUTE_NAMES must be registered.
_unknown = set(ATTRIBUTE_NAMES) - set(_ATTRIBUTE_REGISTRY)
if _unknown:
    raise ValueError(
        f"ATTRIBUTE_NAMES contains unknown attribute(s): {sorted(_unknown)}. "
        f"Valid options: {sorted(_ATTRIBUTE_REGISTRY)}"
    )


@dataclass(frozen=True)
class GateAttributes:
    """Decomposed attributes for a single cell type.

    All fields are always populated regardless of which attributes are
    currently enabled in ``ATTRIBUTE_NAMES``; that list only controls
    which ones are fed to the embedding layer.
    """

    logic_family: str
    input_count: int
    output_complemented: bool
    is_sequential: bool
    drive_strength: float
    num_outputs: int
    tristate: bool
    is_clock_related: bool


# =====================================================================
# GateAttributeVocab — the public interface
# =====================================================================


class GateAttributeVocab:
    """Builds a lookup table that maps cell names to attribute index tuples.

    Parameters
    ----------
    gate_funcs : dict
        ``{cell_name: {output_pin: boolean_expression, ...}, ...}``
        as read from ``sim_config.json["gate_funcs"]``.
    """

    def __init__(self, gate_funcs: Dict[str, Dict[str, str]]) -> None:
        self._attrs: Dict[str, GateAttributes] = {}
        self._indices: Dict[str, Tuple[int, ...]] = {}

        for cell_name, func_dict in gate_funcs.items():
            attrs = self._extract(cell_name, func_dict)
            self._attrs[cell_name] = attrs
            self._indices[cell_name] = self._to_indices(attrs, cell_name)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_sim_config(cls, path: str | Path) -> "GateAttributeVocab":
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        return cls(cfg["gate_funcs"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def vocab_sizes(self) -> Dict[str, int]:
        """Number of classes for each *enabled* attribute (embedding layers)."""
        return {name: _ATTRIBUTE_REGISTRY[name][0] for name in ATTRIBUTE_NAMES}

    @property
    def num_attributes(self) -> int:
        return NUM_ATTRIBUTES

    def encode(self, cell_name: str) -> Tuple[int, ...]:
        """Return a tuple of integer indices for the enabled attributes."""
        if cell_name in self._indices:
            return self._indices[cell_name]
        return self._default_indices()

    def get_attrs(self, cell_name: str) -> GateAttributes:
        if cell_name in self._attrs:
            return self._attrs[cell_name]
        return GateAttributes(
            logic_family="UNKNOWN", input_count=0,
            output_complemented=False, is_sequential=False,
            drive_strength=1.0,
            num_outputs=1,
            tristate=False, is_clock_related=False,
        )

    def __len__(self) -> int:
        return len(self._attrs)

    def __contains__(self, cell_name: str) -> bool:
        return cell_name in self._attrs

    # ------------------------------------------------------------------
    # Debug / exploration helpers
    # ------------------------------------------------------------------

    def unique_combinations(self) -> Dict[Tuple[int, ...], List[str]]:
        """Group cells by their encoded attribute tuple under the currently
        enabled ``ATTRIBUTE_NAMES``.

        Returns
        -------
        dict
            ``{index_tuple: [cell_name, ...], ...}``.
            ``len(result)`` is the number of distinct combinations the
            model will see; compare with ``len(vocab)`` (number of cells)
            to see the collapse factor as attributes are toggled.
        """
        groups: Dict[Tuple[int, ...], List[str]] = {}
        for cell, idx in self._indices.items():
            groups.setdefault(idx, []).append(cell)
        return groups

    def report_collapse(self) -> None:
        """Print a summary of how cells collapse into distinct attribute
        combinations under the currently enabled ``ATTRIBUTE_NAMES``."""
        n_cells = len(self)
        n_combos = len(self.unique_combinations())
        ratio = n_cells / n_combos if n_combos else 0.0
        print(f"{NUM_ATTRIBUTES} attrs enabled: {ATTRIBUTE_NAMES}")
        print(
            f"{n_cells} cells -> {n_combos} unique combinations "
            f"({ratio:.2f}x collapse)"
        )

    @staticmethod
    def describe(indices: Tuple[int, ...]) -> Dict[str, str]:
        """Convert an encoded tuple back to ``{attribute_name: label}``
        using the currently enabled ``ATTRIBUTE_NAMES``."""
        if len(indices) != NUM_ATTRIBUTES:
            raise ValueError(
                f"indices has length {len(indices)} but {NUM_ATTRIBUTES} "
                f"attribute(s) are enabled: {ATTRIBUTE_NAMES}"
            )
        return {
            name: _ATTRIBUTE_LABELS[name][idx]
            for name, idx in zip(ATTRIBUTE_NAMES, indices)
        }

    def report_bins(self, show_cells: bool = True) -> None:
        """Print every bin (unique attribute combination) with its members.

        Bins are sorted by size (descending).

        Parameters
        ----------
        show_cells : bool
            If ``True``, list every cell in each bin.  If ``False``,
            print the count and a single example cell per bin.
        """
        combos = self.unique_combinations()
        bins = sorted(combos.items(), key=lambda kv: -len(kv[1]))
        header = f"{len(self)} cells -> {len(bins)} bins (enabled: {ATTRIBUTE_NAMES})"
        print(header)
        print("-" * len(header))
        for i, (idx_tuple, cells) in enumerate(bins, start=1):
            labels = self.describe(idx_tuple)
            label_str = ", ".join(f"{k}={v}" for k, v in labels.items())
            print(f"[{i:>3}/{len(bins)}] {len(cells):>4} cells  {label_str}")
            if show_cells:
                for c in cells:
                    print(f"        - {c}")
            else:
                print(f"        e.g. {cells[0]}")

    def plot_bin_sizes(self, save_path: str | Path) -> None:
        """Save a bar chart of bin sizes (one bar per unique combination,
        sorted descending) to ``save_path``."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        combos = self.unique_combinations()
        bins = sorted(combos.items(), key=lambda kv: -len(kv[1]))
        sizes = [len(cells) for _, cells in bins]
        # Short label per bin: dominant logic_family (if enabled) + bin #
        if "logic_family" in ATTRIBUTE_NAMES:
            lf_pos = ATTRIBUTE_NAMES.index("logic_family")
            labels = [
                f"{i+1}. {LOGIC_FAMILIES[idx[lf_pos]]}"
                for i, (idx, _) in enumerate(bins)
            ]
        else:
            labels = [f"bin {i+1}" for i in range(len(bins))]

        fig, ax = plt.subplots(figsize=(max(6, 0.25 * len(bins)), 4))
        ax.bar(range(len(sizes)), sizes, color="steelblue")
        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_ylabel("# cells in bin")
        ax.set_xlabel("Bin (sorted by size)")
        ax.set_title(
            f"{len(self)} cells -> {len(bins)} bins "
            f"({len(ATTRIBUTE_NAMES)} attrs: {', '.join(ATTRIBUTE_NAMES)})",
            fontsize=9,
        )
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Saved bin-size plot to {save_path}")

    def plot_attribute_distributions(self, save_path: str | Path) -> None:
        """Save a grid of histograms — one per enabled attribute — showing
        how many cells fall into each attribute value."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if NUM_ATTRIBUTES == 0:
            raise ValueError("No attributes enabled — nothing to plot.")

        # Count cells per (attribute, value) across the vocab
        counts: Dict[str, Dict[str, int]] = {
            name: {label: 0 for label in _ATTRIBUTE_LABELS[name]}
            for name in ATTRIBUTE_NAMES
        }
        for attrs in self._attrs.values():
            for name in ATTRIBUTE_NAMES:
                idx = _ATTRIBUTE_REGISTRY[name][1](attrs)
                counts[name][_ATTRIBUTE_LABELS[name][idx]] += 1

        ncols = min(3, NUM_ATTRIBUTES)
        nrows = (NUM_ATTRIBUTES + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5 * ncols, 3 * nrows), squeeze=False
        )
        for ax, name in zip(axes.flat, ATTRIBUTE_NAMES):
            labels = list(counts[name].keys())
            values = [counts[name][lbl] for lbl in labels]
            ax.bar(range(len(labels)), values, color="steelblue")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_title(name, fontsize=10)
            ax.set_ylabel("# cells")
        # Hide any unused axes
        for ax in axes.flat[NUM_ATTRIBUTES:]:
            ax.set_visible(False)
        fig.suptitle(f"Per-attribute cell distribution ({len(self)} cells)")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Saved per-attribute histograms to {save_path}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _extract(cell_name: str, func_dict: Dict[str, str]) -> GateAttributes:
        basename = _strip_tech_suffix(cell_name)

        m = _BASENAME_RE.match(basename)
        if m:
            prefix, drive_str, _suffix = m.groups()
            drive_val = _parse_drive_value(drive_str)
        else:
            prefix = basename
            drive_val = 1.0

        family = _classify_family(prefix)

        # Input count: prefer counting variables from the function; fall
        # back to heuristic for sequential / special cells.
        n_inputs = _count_inputs_from_function(func_dict)
        if n_inputs == 0:
            n_inputs = _heuristic_input_count(family, prefix)

        # Output complementation
        complemented = family in _COMPLEMENTED_FAMILIES
        # For MAJ with inverted variant (MAJI)
        if prefix.upper().startswith("MAJI"):
            complemented = True

        return GateAttributes(
            logic_family=family,
            input_count=n_inputs,
            output_complemented=complemented,
            is_sequential=family in _SEQUENTIAL_FAMILIES,
            drive_strength=drive_val,
            num_outputs=min(len(func_dict), 2),
            tristate=False,  # No tristate in ASAP7; placeholder for future PDKs
            is_clock_related=any(
                prefix.upper().startswith(cp) for cp in _CLOCK_PREFIXES
            ),
        )

    @staticmethod
    def _to_indices(attrs: GateAttributes, cell_name: str = "") -> Tuple[int, ...]:
        return tuple(
            _ATTRIBUTE_REGISTRY[name][1](attrs) for name in ATTRIBUTE_NAMES
        )

    @staticmethod
    def _default_indices() -> Tuple[int, ...]:
        """All-unknown / all-false fallback (one entry per enabled attribute)."""
        return tuple(0 for _ in ATTRIBUTE_NAMES)
