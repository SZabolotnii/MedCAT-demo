"""Helpers for running MedCAT extractions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

try:
    # Package import (e.g. python -m src.extractor)
    from .utils import load_model_pack_auto
except ImportError:  # pragma: no cover - fallback for running as a script
    from utils import load_model_pack_auto

if TYPE_CHECKING:  # pragma: no cover - hints only
    from medcat.cat import CAT


def extract_entities(cat: "CAT", text: str) -> dict:
    """
    Здійснити витяг медичних сутностей з тексту.
    Повертає словник результатів.
    """
    return cat.get_entities(text, only_cui=False)


if __name__ == "__main__":
    # Для тесту
    # model_path = "models/umls_sm_pt2ch_533bab5115c6c2d6.zip"
    # model_path = "models/v2_Snomed2025_MIMIC_IV_bbe806e192df009f.zip"
    model_path = Path("models/IEE_MedCAT_v1")
    cat = load_model_pack_auto(model_path)
    sample_text = "My heart rate is high (120)"
    ents = extract_entities(cat, sample_text)
    print("Сутності:", ents)
