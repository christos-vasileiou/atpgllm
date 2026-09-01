"""Staged graph-pipeline hyperparameter optimization utilities."""

from .config import HPO_CONFIG_VERSION, PipelineHPOConfig, load_hpo_config
from .manifests import canonical_hash, design_hash, sha256_file
from .metrics import binary_average_precision, binary_iou, binary_roc_auc
from .promotion import select_top_k
from .splits import DesignSplitManifest, build_design_split_manifest

__all__ = [
    "HPO_CONFIG_VERSION",
    "PipelineHPOConfig",
    "DesignSplitManifest",
    "binary_average_precision",
    "binary_iou",
    "binary_roc_auc",
    "build_design_split_manifest",
    "canonical_hash",
    "design_hash",
    "load_hpo_config",
    "select_top_k",
    "sha256_file",
]
