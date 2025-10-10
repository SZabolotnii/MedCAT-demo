"""Utility helpers for loading the SpaCy pipeline with HintNER support."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    import spacy
    from spacy.language import Language
    from spacy.tokens import Doc, Span
except ImportError as exc:  # pragma: no cover - informative error if spaCy missing
    raise RuntimeError(
        "spaCy is required for the semantic hint pipeline. "
        "Install it via `pip install spacy` and the `en_core_web_md` model."
    ) from exc

# Ensure the HintNER factory is registered before the pipeline is loaded.
from .pipelines import hint_ner as _hint_ner_module  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_NAME = "en_core_web_md"
DEFAULT_LEXICON_PATH = PROJECT_ROOT / "data" / "hints" / "hint_lexicon.json"


def _resolve_lexicon_path(path: str | Path | None) -> Path:
    if path is None:
        return DEFAULT_LEXICON_PATH
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


@lru_cache(maxsize=8)
def load_spacy_with_hints(
    model_name: str = DEFAULT_MODEL_NAME,
    *,
    lexicon_path: str | Path | None = None,
    similarity_threshold: float = 0.78,
    top_k: int = 4,
    max_ngram: int = 5,
    disable: Iterable[str] | None = None,
) -> Language:
    """Load a spaCy pipeline and inject the HintNER component."""

    disable_list = list(disable or [])
    nlp = spacy.load(model_name, disable=disable_list)

    lexicon = _resolve_lexicon_path(lexicon_path)
    if "hint_ner" in nlp.pipe_names:
        nlp.remove_pipe("hint_ner")

    config = {
        "lexicon_path": str(lexicon),
        "similarity_threshold": similarity_threshold,
        "top_k": top_k,
        "max_ngram": max_ngram,
    }

    insertion_point = "ner" if "ner" in nlp.pipe_names else None
    nlp.add_pipe("hint_ner", config=config, before=insertion_point)
    return nlp


def iter_hint_spans(doc: Doc) -> Iterable[Span]:
    """Yield spans that carry hint metadata."""
    for span in doc.ents:
        if getattr(span._, "hint_id", None):
            yield span


def extract_hint_entities(
    text: str,
    *,
    nlp: Optional[Language] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    lexicon_path: str | Path | None = None,
    similarity_threshold: float = 0.78,
    top_k: int = 4,
    max_ngram: int = 5,
    disable: Iterable[str] | None = None,
) -> List[Dict[str, object]]:
    """Process text and return hint-based entities with metadata."""
    pipeline = nlp or load_spacy_with_hints(
        model_name,
        lexicon_path=lexicon_path,
        similarity_threshold=similarity_threshold,
        top_k=top_k,
        max_ngram=max_ngram,
        disable=disable,
    )

    doc = pipeline(text)
    entities: List[Dict[str, object]] = []
    for span in iter_hint_spans(doc):
        entities.append(
            {
                "text": span.text,
                "label": span.label_,
                "start": span.start_char,
                "end": span.end_char,
                "hint_id": span._.hint_id,
                "hint_score": float(getattr(span._, "hint_score", 0.0) or 0.0),
                "hint_source": getattr(span._, "hint_source", None),
                "hint_term": getattr(span._, "hint_term", None),
            }
        )
    return entities


__all__ = [
    "DEFAULT_LEXICON_PATH",
    "DEFAULT_MODEL_NAME",
    "extract_hint_entities",
    "iter_hint_spans",
    "load_spacy_with_hints",
]
