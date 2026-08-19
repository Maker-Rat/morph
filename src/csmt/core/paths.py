from __future__ import annotations

import os
from pathlib import Path
from typing import Any


XMORPH_ROOT_ENV = "XMORPH_ROOT"
XMORPH_ROOT_TOKEN = "${XMORPH_ROOT}"


def xmorph_root() -> Path:
    """Return the configured repository root, or infer it from this package."""
    configured = os.environ.get(XMORPH_ROOT_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


def expand_xmorph_paths(value: Any) -> Any:
    """Recursively expand ``${XMORPH_ROOT}`` in loaded run metadata."""
    if isinstance(value, str):
        return value.replace(XMORPH_ROOT_TOKEN, str(xmorph_root()))
    if isinstance(value, dict):
        return {key: expand_xmorph_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_xmorph_paths(item) for item in value]
    if isinstance(value, tuple):
        return tuple(expand_xmorph_paths(item) for item in value)
    return value


def make_xmorph_paths_portable(value: Any, root: str | Path | None = None) -> Any:
    """Recursively replace paths under the repository root with the root token."""
    root_path = Path(root).expanduser().resolve() if root is not None else xmorph_root()
    root_text = str(root_path)

    if isinstance(value, str):
        return value.replace(root_text, XMORPH_ROOT_TOKEN)
    if isinstance(value, dict):
        return {key: make_xmorph_paths_portable(item, root_path) for key, item in value.items()}
    if isinstance(value, list):
        return [make_xmorph_paths_portable(item, root_path) for item in value]
    if isinstance(value, tuple):
        return tuple(make_xmorph_paths_portable(item, root_path) for item in value)
    return value
