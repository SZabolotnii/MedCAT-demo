"""spaCy component that projects keyword/value hints into vector space for NER."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span
from spacy.util import filter_spans


@dataclass(frozen=True)
class HintConcept:
    """Canonical concept enriched with keyword and value representations."""

    uid: str
    label: str
    canonical_keyword: str
    cluster_id: str
    cluster_title: str
    keyword_terms: Tuple[str, ...]
    value_terms: Tuple[str, ...]
    sources: Tuple[str, ...]


@dataclass(frozen=True)
class HintVectorMatch:
    """Result of a vector similarity lookup against the hint index."""

    score: float
    concept: HintConcept
    term_text: str
    term_type: str


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
        canonical_keyword = str(item.get("canonical_keyword") or "").strip()
        keyword_terms = tuple(
            term.strip() for term in item.get("keyword_terms") or [] if isinstance(term, str) and term.strip()
        )
        value_terms = tuple(
            term.strip() for term in item.get("value_terms") or [] if isinstance(term, str) and term.strip()
        )
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
                keyword_terms=keyword_terms,
                value_terms=value_terms,
                sources=sources,
            )
        )
    return concepts


class HintVectorIndex:
    """Lightweight cosine-similarity index for hint embeddings."""

    def __init__(self, nlp: Language, concepts: Sequence[HintConcept]) -> None:
        self._nlp = nlp
        vectors: List[np.ndarray] = []
        metadata: List[Tuple[str, str, str, HintConcept]] = []

        for concept in concepts:
            for term in concept.keyword_terms:
                vec = self._term_vector(term)
                if vec is not None:
                    vectors.append(vec)
                    metadata.append(("keyword", term, concept.uid, concept))
            for term in concept.value_terms:
                vec = self._term_vector(term)
                if vec is not None:
                    vectors.append(vec)
                    metadata.append(("value", term, concept.uid, concept))

        if vectors:
            self._matrix = np.stack(vectors).astype("float32")
        else:
            self._matrix = np.zeros((0, nlp.vocab.vectors_length or 0), dtype="float32")
        self._metadata = metadata

    def _term_vector(self, text: str) -> np.ndarray | None:
        doc = self._nlp.make_doc(text)
        if not doc:
            return None
        vector = doc.vector
        if vector is None or vector.shape[0] == 0:
            return None
        norm = float(np.linalg.norm(vector))
        if not norm:
            return None
        return (vector / norm).astype("float32")

    def query(self, span: Span, top_k: int, min_score: float) -> List[HintVectorMatch]:
        """Return top-k vector matches above the similarity threshold."""
        if self._matrix.size == 0:
            return []

        vector = span.vector
        if vector is None or vector.shape[0] == 0:
            return []
        norm = float(np.linalg.norm(vector))
        if not norm:
            return []
        normalized = (vector / norm).astype("float32")
        scores = self._matrix @ normalized

        if scores.size <= top_k:
            indices = np.argsort(-scores)
        else:
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            indices = top_indices[np.argsort(-scores[top_indices])]

        matches: List[HintVectorMatch] = []
        for idx in indices:
            score = float(scores[idx])
            if score < min_score:
                continue
            term_type, term_text, _uid, concept = self._metadata[idx]
            matches.append(HintVectorMatch(score=score, concept=concept, term_text=term_text, term_type=term_type))
        return matches


class HintNER:
    """spaCy pipeline component that injects hint-driven entity spans."""

    _DEFAULT_THRESHOLD = 0.78
    _DEFAULT_TOP_K = 4
    _DEFAULT_MAX_NGRAM = 5

    def __init__(
        self,
        nlp: Language,
        name: str,
        *,
        lexicon_path: str,
        similarity_threshold: float | None = None,
        top_k: int | None = None,
        max_ngram: int | None = None,
    ) -> None:
        self.nlp = nlp
        self.name = name
        self.similarity_threshold = similarity_threshold or self._DEFAULT_THRESHOLD
        self.top_k = top_k or self._DEFAULT_TOP_K
        self.max_ngram = max_ngram or self._DEFAULT_MAX_NGRAM

        self.concepts = load_hint_lexicon(lexicon_path)
        self._concept_by_uid: Dict[str, HintConcept] = {concept.uid: concept for concept in self.concepts}
        self._phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        self._build_phrase_patterns()
        self._vector_index = HintVectorIndex(nlp, self.concepts)

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

    def _build_phrase_patterns(self) -> None:
        for concept in self.concepts:
            patterns = [self.nlp.make_doc(term) for term in concept.keyword_terms if term]
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
            self._assign_metadata(span, concept, score=1.0, source="phrase", term_text=span.text)
            self._save_span(scored_spans, span, score=1.0)

        for span in self._generate_candidate_spans(doc):
            matches = self._vector_index.query(span, top_k=self.top_k, min_score=self.similarity_threshold)
            if not matches:
                continue
            for match in matches:
                concept = match.concept
                new_span = Span(doc, span.start, span.end, label=concept.label)
                self._assign_metadata(
                    new_span,
                    concept,
                    score=match.score,
                    source=f"vector:{match.term_type}",
                    term_text=match.term_text,
                )
                self._save_span(scored_spans, new_span, score=match.score)

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

    def _generate_candidate_spans(self, doc: Doc) -> Iterator[Span]:
        seen: set[Tuple[int, int]] = set()
        # Nominal chunks provide strong candidate spans when the parser is available.
        try:
            for chunk in doc.noun_chunks:
                if chunk.text.strip():
                    seen.add((chunk.start, chunk.end))
        except ValueError:
            # Parser not available; noun_chunks unsupported.
            pass

        sentences: Iterable[Span]
        if doc.has_annotation("SENT_START"):
            sentences = doc.sents
        else:
            sentences = (doc,)

        for sent in sentences:
            for start in range(sent.start, sent.end):
                token = doc[start]
                if token.is_space or token.is_punct:
                    continue
                end_limit = min(sent.end, start + self.max_ngram)
                for end in range(start + 1, end_limit + 1):
                    span = doc[start:end]
                    if not span.text.strip():
                        continue
                    if not any(ch.isalnum() for ch in span.text):
                        continue
                    if span[-1].is_punct:
                        continue
                    seen.add((span.start, span.end))

        for start, end in sorted(seen):
            span = doc[start:end]
            if not span.text.strip():
                continue
            if not any(ch.isalnum() for ch in span.text):
                continue
            yield span


@Language.factory(
    "hint_ner",
    default_config={
        "lexicon_path": "data/hints/hint_lexicon.json",
        "similarity_threshold": HintNER._DEFAULT_THRESHOLD,
        "top_k": HintNER._DEFAULT_TOP_K,
        "max_ngram": HintNER._DEFAULT_MAX_NGRAM,
    },
    assigns=["doc.ents"],
)
def create_hint_ner(  # type: ignore[override]
    nlp,
    name: str,
    lexicon_path: str,
    similarity_threshold: float,
    top_k: int,
    max_ngram: int,
) -> HintNER:
    """Factory used by spaCy to construct the HintNER component."""
    return HintNER(
        nlp,
        name,
        lexicon_path=lexicon_path,
        similarity_threshold=similarity_threshold,
        top_k=top_k,
        max_ngram=max_ngram,
    )


__all__ = ["HintNER", "create_hint_ner", "load_hint_lexicon"]
