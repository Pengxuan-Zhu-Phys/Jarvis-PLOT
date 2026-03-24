from __future__ import annotations

from pathlib import Path
from typing import Optional, Union


PathLike = Union[str, Path]


def repo_root() -> Path:
    """Return the Jarvis-PLOT repository root."""
    return Path(__file__).resolve().parents[2]


def resolve_project_path(path: PathLike, base_dir: Optional[PathLike] = None) -> Path:
    """Resolve Jarvis-PLOT-relative paths with support for the internal ``&JP/`` prefix."""
    text = str(path).strip()
    if not text:
        return Path(text).expanduser().resolve()

    if text.startswith("&JP/"):
        return (repo_root() / text[4:]).resolve()

    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    if base_dir is not None:
        return (Path(base_dir).expanduser() / candidate).resolve()

    return candidate.resolve()
