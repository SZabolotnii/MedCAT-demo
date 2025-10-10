"""Helpers for running MedCAT extractions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

try:
    from .spacy_pipeline import extract_hint_entities
except ImportError:
    extract_hint_entities = None  # type: ignore[assignment]

try:
    # Package import (e.g. python -m src.extractor)
    from .utils import load_model_pack_auto
except ImportError:  # pragma: no cover - fallback for running as a script
    from utils import load_model_pack_auto

if TYPE_CHECKING:  # pragma: no cover - hints only
    from medcat.cat import CAT


def extract_entities(
    cat: "CAT",
    text: str,
    *,
    include_hint_metadata: bool = False,
    hint_config: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    Здійснити витяг медичних сутностей з тексту.
    Повертає словник результатів.
    """
    # Use extract_entities for CustomCAT, get_entities for regular CAT
    if hasattr(cat, "extract_entities"):
        result: dict = cat.extract_entities(text)
    else:
        result = cat.get_entities(text, only_cui=False)

    if include_hint_metadata and extract_hint_entities is not None:
        hint_kwargs: Dict[str, Any] = dict(hint_config or {})
        hint_list = extract_hint_entities(text, **hint_kwargs)
        result["hint_entities"] = hint_list
    elif include_hint_metadata:
        raise RuntimeError("spaCy pipeline is unavailable. Ensure spaCy is installed before requesting hint metadata.")

    return result


if __name__ == "__main__":
    # Для тесту
    # model_path = "models/umls_sm_pt2ch_533bab5115c6c2d6.zip"
    # model_path = "models/v2_Snomed2025_MIMIC_IV_bbe806e192df009f.zip"
    model_path = Path("models/IEE_MedCAT_v1")
    cat = load_model_pack_auto(model_path)
    sample_text = "My heart rate is high (120). Aspirin 100 mg"
    ents = extract_entities(cat, sample_text)
    print("Сутності:", ents)
