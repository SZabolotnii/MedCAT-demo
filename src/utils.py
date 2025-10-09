"""Utilities for preparing and loading MedCAT models."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - used for type hints only
    from medcat.cat import CAT


def load_model_pack(
    model_pack_path: str | Path,
    *,
    use_cache: bool = True,
    force_reload: bool = False,
) -> "CAT":
    """Load a MedCAT model pack from disk, optionally caching the result.

    Re-using a cached ``CAT`` instance avoids the expensive deserialisation step
    when the same model file is loaded repeatedly (e.g. in a web service).

    Args:
        model_pack_path: Path to a ``.zip`` or directory created by MedCAT.
        use_cache: Return a shared cached instance when available.
        force_reload: Clear the cache for this process before loading, ensuring
            a fresh instance is returned.

    Returns:
        A fully configured ``CAT`` instance ready for inference.
    """
    resolved_path = Path(model_pack_path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Model pack not found: {resolved_path}")

    model_key = str(resolved_path)
    if force_reload:
        _load_model_pack_cached.cache_clear()

    if use_cache:
        return _load_model_pack_cached(model_key)

    return _load_model_pack_uncached(model_key)


@lru_cache(maxsize=None)
def _load_model_pack_cached(model_key: str) -> "CAT":
    """Shared loader used behind ``load_model_pack`` when caching is enabled."""
    return _load_model_pack_uncached(model_key)


def _load_model_pack_uncached(model_key: str) -> "CAT":
    from medcat.cat import CAT  # Imported lazily to keep import times low

    return CAT.load_model_pack(model_key)


def load_model_pack_auto(
    model_pack_path: str | Path,
    *,
    use_cache: bool = True,
    force_reload: bool = False,
) -> "CAT":
    """Load either a standard MedCAT pack or the custom Phase 1A pack."""

    resolved = Path(model_pack_path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Model pack not found: {resolved}")

    model_key = str(resolved)
    if force_reload:
        _load_model_pack_auto_cached.cache_clear()

    if use_cache:
        return _load_model_pack_auto_cached(model_key)

    return _load_model_pack_auto_uncached(model_key)


@lru_cache(maxsize=None)
def _load_model_pack_auto_cached(model_key: str) -> "CAT":
    return _load_model_pack_auto_uncached(model_key)


def _load_model_pack_auto_uncached(model_key: str) -> "CAT":
    path = Path(model_key)
    if path.is_dir() and (path / "custom_cdb_v2").exists():
        return _load_custom_phase1a_cat(path)

    if path.suffix == ".zip":
        import zipfile

        with zipfile.ZipFile(path, "r") as archive:
            if any(name.startswith("custom_cdb_v2/") for name in archive.namelist()):
                return _load_custom_phase1a_cat(path)

    return _load_model_pack_uncached(model_key)


def _load_custom_phase1a_cat(path: Path) -> "CAT":
    try:
        from .custom_cat_v2 import CustomCAT
    except ImportError:
        from importlib import import_module

        try:
            CustomCAT = import_module("custom_cat_v2").CustomCAT  # type: ignore[attr-defined]
        except (ImportError, AttributeError) as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                "CustomCAT is unavailable. Ensure src.custom_cat_v2 is accessible."
            ) from exc

    if path.is_dir():
        combined = path / "internal_combined_hints.json"
        wrapper = CustomCAT(
            path,
            combined_hints_path=combined if combined.exists() else None,
        )
        return wrapper.cat

    if path.suffix == ".zip":
        import tempfile
        import zipfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            extracted = Path(tmp_dir)
            with zipfile.ZipFile(path, "r") as archive:
                archive.extractall(extracted)
            combined = extracted / "internal_combined_hints.json"
            wrapper = CustomCAT(
                extracted,
                combined_hints_path=combined if combined.exists() else None,
            )
            return wrapper.cat

    raise ValueError(f"Unsupported custom pack location: {path}")
