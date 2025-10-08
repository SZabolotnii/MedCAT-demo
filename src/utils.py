"""Utilities for preparing and loading MedCAT models."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - used for type hints only
    from medcat.cat import CAT


def load_model_pack(model_pack_path: str | Path) -> "CAT":
    """Load a MedCAT model pack from disk.

    Args:
        model_pack_path: Path to a ``.zip`` or directory created by MedCAT.

    Returns:
        A fully configured ``CAT`` instance ready for inference.
    """
    from medcat.cat import CAT  # Imported lazily to keep import times low

    resolved_path = Path(model_pack_path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Model pack not found: {resolved_path}")

    return CAT.load_model_pack(resolved_path)
