"""spaCy component that injects ontology keyword matches as entities."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span
from spacy.util import filter_spans


@dataclass(frozen=True)
class HintConcept:
    """Canonical concept enriched with keyword patterns."""

    uid: str
    label: str
    canonical_keyword: str
    cluster_id: str
    cluster_title: str
    patterns: Tuple[str, ...]
    sources: Tuple[str, ...]


def _normalize_cluster_label(cluster_title: str, cluster_id: str) -> str:
    base = (cluster_title or "").strip()
    if base:
        label = base.replace("/", " ").replace("-", " ").upper()
        return "_".join(part for part in label.split() if part)
    cluster = (cluster_id or "").strip() or "UNKNOWN"
    return f"CLUSTER_{cluster}".upper()


def _derive_label(canonical_keyword: str, uid: str, cluster_label: str) -> str:
    keyword = canonical_keyword.strip()
    if keyword:
        return keyword
    if cluster_label:
        return cluster_label
    return uid

def _normalize_hint_term(term: str) -> str:
    """Remove service markers and collapse whitespace in hint terms."""
    cleaned = re.sub(r"\s*\[combined_hint\]\s*", " ", term)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def load_hint_lexicon(path: str | Path) -> List[HintConcept]:
    """Load structured hint lexicon entries from JSON."""
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Hint lexicon not found: {resolved}")

    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    concepts: List[HintConcept] = []
    for item in payload:
        uid = str(item.get("uid") or "").strip()
        if not uid:
            continue
        cluster_id = str(item.get("cluster_id") or "").strip()
        cluster_title = str(item.get("cluster_title") or "").strip()
        cluster_label = _normalize_cluster_label(cluster_title, cluster_id)
        canonical_keyword_raw = str(item.get("canonical_keyword") or "").strip()
        canonical_keyword = _normalize_hint_term(canonical_keyword_raw)
        if not canonical_keyword:
            # Skip entries without canonical keywords to align with ontology keywords only.
            continue
        pattern_terms = {canonical_keyword}
        for term in item.get("keyword_terms") or []:
            if not isinstance(term, str):
                continue
            normalized = _normalize_hint_term(term)
            if normalized:
                pattern_terms.add(normalized)
        sources = tuple(
            source.strip() for source in item.get("sources") or [] if isinstance(source, str) and source.strip()
        )
        concepts.append(
            HintConcept(
                uid=uid,
                label=_derive_label(canonical_keyword, uid, cluster_label),
                canonical_keyword=canonical_keyword,
                cluster_id=cluster_id,
                cluster_title=cluster_title or cluster_label,
                patterns=tuple(sorted({term for term in pattern_terms if term})),
                sources=sources,
            )
        )
    return concepts


class HintNER:
    """spaCy pipeline component that injects hint-driven entity spans."""

    def __init__(
        self,
        nlp: Language,
        name: str,
        *,
        lexicon_path: str,
    ) -> None:
        self.nlp = nlp
        self.name = name

        self.concepts = load_hint_lexicon(lexicon_path)
        self._concept_by_uid: Dict[str, HintConcept] = {concept.uid: concept for concept in self.concepts}
        self._phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        self._build_phrase_patterns()

        self._ensure_extensions()

    @staticmethod
    def _ensure_extensions() -> None:
        if not Span.has_extension("hint_id"):
            Span.set_extension("hint_id", default=None)
        if not Span.has_extension("hint_score"):
            Span.set_extension("hint_score", default=0.0)
        if not Span.has_extension("hint_source"):
            Span.set_extension("hint_source", default=None)
        if not Span.has_extension("hint_term"):
            Span.set_extension("hint_term", default=None)
        if not Span.has_extension("hint_cluster_title"):
            Span.set_extension("hint_cluster_title", default=None)
        if not Span.has_extension("hint_cluster_id"):
            Span.set_extension("hint_cluster_id", default=None)
        if not Span.has_extension("hint_canonical_keyword"):
            Span.set_extension("hint_canonical_keyword", default=None)
        if not Span.has_extension("hint_matched_text"):
            Span.set_extension("hint_matched_text", default=None)

    def _build_phrase_patterns(self) -> None:
        for concept in self.concepts:
            patterns = [self.nlp.make_doc(term) for term in concept.patterns if term]
            if patterns:
                self._phrase_matcher.add(concept.uid, patterns)

    def __call__(self, doc: Doc) -> Doc:
        if not self.concepts:
            return doc

        scored_spans: Dict[Tuple[int, int, str], Tuple[float, Span]] = {}

        for match_id, start, end in self._phrase_matcher(doc):
            uid = self.nlp.vocab.strings[match_id]
            concept = self._concept_by_uid.get(uid)
            if concept is None:
                continue
            span = Span(doc, start, end, label=concept.label)
            self._assign_metadata(span, concept, score=1.0, source="phrase", term_text=concept.canonical_keyword)
            self._save_span(scored_spans, span, score=1.0)

        if not scored_spans:
            return doc

        sorted_spans = sorted(
            scored_spans.values(),
            key=lambda pair: (-pair[0], pair[1].start, pair[1].end),
        )
        new_spans = [span for _score, span in sorted_spans]
        doc.set_ents(filter_spans(tuple(doc.ents) + tuple(new_spans)))
        return doc

    def _assign_metadata(self, span: Span, concept: HintConcept, *, score: float, source: str, term_text: str) -> None:
        span._.hint_id = concept.uid
        span._.hint_score = score
        span._.hint_source = source
        span._.hint_term = term_text
        span._.hint_cluster_title = concept.cluster_title
        span._.hint_cluster_id = concept.cluster_id
        span._.hint_canonical_keyword = concept.canonical_keyword or concept.label
        span._.hint_matched_text = span.text

    @staticmethod
    def _save_span(
        scored_spans: Dict[Tuple[int, int, str], Tuple[float, Span]],
        span: Span,
        *,
        score: float,
    ) -> None:
        key = (span.start, span.end, span.label_)
        existing = scored_spans.get(key)
        if existing is None or score > existing[0]:
            scored_spans[key] = (score, span)

@Language.factory(
    "hint_ner",
    default_config={"lexicon_path": "data/hints/hint_lexicon.json"},
    assigns=["doc.ents"],
)
def create_hint_ner(  # type: ignore[override]
    nlp,
    name: str,
    lexicon_path: str,
) -> HintNER:
    """Factory used by spaCy to construct the HintNER component."""
    return HintNER(
        nlp,
        name,
        lexicon_path=lexicon_path,
    )


__all__ = ["HintNER", "create_hint_ner", "load_hint_lexicon"]
