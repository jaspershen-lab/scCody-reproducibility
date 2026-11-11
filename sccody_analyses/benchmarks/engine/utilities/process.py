from __future__ import annotations
from pathlib import Path

def repository_root(from_path: str | Path) -> Path:
    """Find the closest ancestor containing this project's pyproject.toml."""
    current = Path(from_path).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / 'requirements.txt').is_file() and (candidate / 'configs').is_dir():
            return candidate
    raise FileNotFoundError(f'Could not locate repository root from {from_path}')
