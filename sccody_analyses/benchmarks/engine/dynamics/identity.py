from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class BenchmarkMethodIdentity:
    method_id: str
    display_name: str
    metric_space: str
