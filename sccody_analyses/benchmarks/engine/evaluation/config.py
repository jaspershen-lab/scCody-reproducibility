"""Evaluation settings for population reconstruction and age localization."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class EvaluationConfig:
    de_method: str = 'edger_pseudobulk'
    metric_profile: str = 'full'
    min_de_ref: int = 50
    rscript: str = 'Rscript'
    edger_dir: str | None = None
    min_cells: int = 20
    max_sinkhorn_cells: int = 1000
    projection_sinkhorn_blur: float = 0.05
    projection_sinkhorn_scaling: float = 0.5
    mmd_block_size: int = 256
