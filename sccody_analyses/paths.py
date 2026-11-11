"""Resolve public study paths relative to the reproducibility checkout."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def study_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path
