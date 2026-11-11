"""Safe writers for local experiment artifacts."""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any

def atomic_write_json(path: str | Path, payload: Any, *, indent: int=2, sort_keys: bool=False, trailing_newline: bool=False) -> None:
    """Replace the final JSON result atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    try:
        with temporary.open('w') as handle:
            json.dump(payload, handle, indent=indent, sort_keys=sort_keys)
            if trailing_newline:
                handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
