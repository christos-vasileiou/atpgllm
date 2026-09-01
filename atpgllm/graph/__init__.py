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

Staged training:
    Stage A: target-fault-conditioned DAG encoder pretraining.
    Stage B: Q-Former graph/text alignment with GTC, GTM, and GTG.
    Stage C/D: ``atpgllm.multimodal`` soft-prefix QLoRA SFT and GRPO.
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
from .fault_context import (  # noqa: F401
    FAULT_FEATURE_NAMES,
    NUM_FAULT_FEATURES,
    FaultSpec,
    attach_atpg_context,
    parse_fault,
)
from .node_encoder import AttributeDecompositionEncoder  # noqa: F401
from .dag_gin import DAGGINEncoder, DAGGINLayer  # noqa: F401
from .pretrain import (  # noqa: F401
    GraphPretrainingModel,
    GraphPretrainLosses,
    GraphPretrainOutputs,
)
from .dataset import (  # noqa: F401
    ASAP7DesignDataset,
    ASAP7GraphTextDataset,
    ASAP7GraphPretrainDataset,
    collate_graph_pretrain_batch,
    collate_graph_text_batch,
    make_text_caption,
    render_prompt_and_answer,
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
    "FaultSpec",
    "FAULT_FEATURE_NAMES",
    "NUM_FAULT_FEATURES",
    "parse_fault",
    "attach_atpg_context",
    "AttributeDecompositionEncoder",
    "DAGGINEncoder",
    "DAGGINLayer",
    "GraphPretrainingModel",
    "GraphPretrainLosses",
    "GraphPretrainOutputs",
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
    "ASAP7GraphPretrainDataset",
    "collate_graph_pretrain_batch",
    "collate_graph_text_batch",
    "make_text_caption",
    "render_prompt_and_answer",
    # Stage 2 LLM bridge
    "Stage2GraphTextLM",
    "build_stage2_inputs",
]
