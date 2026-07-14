"""
Public API for ``atpgllm.graph``.

Graph modality stack for circuit (structural netlist) → text / LLM models.

Graph modality pipeline:
    netlist (Verilog)
      → netlist_parser + GateAttributeVocab → PyG Data
      → AttributeDecompositionEncoder → node features
      → DAGGINEncoder (bidirectional GIN, JumpingKnowledge) → node/graph embs
      → GraphQFormer (cross-modal projector, ModernBERT-compatible) → query embs

Text modality pipeline:
    text → ModernBERT-base (or any HF encoder) → [B, H] sentence embedding

Two-stage training:
    Stage 1: contrastive (GTC), matching (GTM) and generative (GTG) losses
             on paired (graph, text) examples (see losses_stage1).
    Stage 2: connect the frozen graph stack to a causal LLM via a soft-prompt
             projection; finetune the LLM with (optional) LoRA on SFT
             targets (see stage2_model).

A future ``atpgllm.multimodal`` package will integrate this encoder with
``atpgllm.llm`` (shared collate, SFT/GRPO, reward wiring).
"""

from .models_stage1 import (  # noqa: F401
    GraphQFormer,
    NetlistGraphEncoder,
    Stage1GraphTextModel,
    Stage1Outputs,
    TextEncoder,
    DEFAULT_TEXT_MODEL,
)
from .losses_stage1 import (  # noqa: F401
    GraphTextContrastiveLoss,
    GraphTextMatchingHead,
    GraphTextMatchingLoss,
    GraphGroundedTextGenerator,
    GraphGroundedTextGenLoss,
)
from .train_stage1 import (  # noqa: F401
    Stage1Losses,
    Stage1Trainer,
)
from .netlist_parser import (  # noqa: F401
    ParsedGate,
    ParsedNetlistGraph,
    netlist_to_pyg,
    parse_verilog_to_pyg,
    parse_verilog_to_graph,
    parsed_to_pyg,
    compute_structural_features,
)
from .gate_features import (  # noqa: F401
    GateAttributeVocab,
    GateAttributes,
)
from .node_encoder import AttributeDecompositionEncoder  # noqa: F401
from .dag_gin import DAGGINEncoder, DAGGINLayer  # noqa: F401
from .dataset import (  # noqa: F401
    ASAP7DesignDataset,
    ASAP7GraphTextDataset,
    collate_graph_text_batch,
    make_text_caption,
)
from .stage2_model import (  # noqa: F401
    Stage2GraphTextLM,
    build_stage2_inputs,
)

__all__ = [
    # Models / encoders
    "Stage1GraphTextModel",
    "Stage1Outputs",
    "NetlistGraphEncoder",
    "GraphQFormer",
    "TextEncoder",
    "DEFAULT_TEXT_MODEL",
    # Training orchestrator
    "Stage1Trainer",
    "Stage1Losses",
    # Node-level encoders
    "GateAttributeVocab",
    "GateAttributes",
    "AttributeDecompositionEncoder",
    "DAGGINEncoder",
    "DAGGINLayer",
    # Losses and heads
    "GraphTextContrastiveLoss",
    "GraphTextMatchingHead",
    "GraphTextMatchingLoss",
    "GraphGroundedTextGenerator",
    "GraphGroundedTextGenLoss",
    # Netlist parsing
    "ParsedGate",
    "ParsedNetlistGraph",
    "netlist_to_pyg",
    "parse_verilog_to_pyg",
    "parse_verilog_to_graph",
    "parsed_to_pyg",
    "compute_structural_features",
    # Dataset utilities
    "ASAP7DesignDataset",
    "ASAP7GraphTextDataset",
    "collate_graph_text_batch",
    "make_text_caption",
    # Stage 2 LLM bridge
    "Stage2GraphTextLM",
    "build_stage2_inputs",
]
