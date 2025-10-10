import json
from pathlib import Path

import pytest

try:
    import spacy
except ImportError:  # pragma: no cover - spaCy should be available in dev env
    spacy = None  # type: ignore

from src.pipelines.hint_ner import load_hint_lexicon


@pytest.mark.skipif(spacy is None, reason="spaCy is required for hint_ner tests")
def test_load_hint_lexicon(tmp_path: Path) -> None:
    lexicon_path = tmp_path / "lexicon.json"
    lexicon_path.write_text(
        json.dumps(
            [
                {
                    "uid": "uid-1",
                    "cluster_id": "cluster-1",
                    "cluster_title": "Finding",
                    "canonical_keyword": "Primary Term",
                    "keyword_terms": ["primary term", "main term"],
                    "value_terms": ["positive", "negative"],
                    "sources": ["unit-test"],
                }
            ]
        ),
        encoding="utf-8",
    )

    concepts = load_hint_lexicon(lexicon_path)
    assert len(concepts) == 1
    concept = concepts[0]
    assert concept.uid == "uid-1"
    assert concept.label == "Primary Term"
    assert concept.cluster_title == "Finding"
    assert "Primary Term" in concept.patterns
    assert "primary term" in concept.patterns


@pytest.mark.skipif(spacy is None, reason="spaCy is required for hint_ner tests")
def test_hint_ner_phrase_match(tmp_path: Path) -> None:
    try:
        nlp = spacy.load("en_core_web_md")
    except OSError:
        pytest.skip("en_core_web_md model is not installed")
    lexicon_path = tmp_path / "lexicon.json"
    lexicon_path.write_text(
        json.dumps(
            [
                {
                    "uid": "pain-uid",
                    "cluster_id": "cluster-pain",
                    "cluster_title": "Symptom",
                    "canonical_keyword": "Chest Pain",
                    "keyword_terms": ["chest pain", "thoracic pain"],
                    "value_terms": [],
                    "sources": ["unit-test"],
                }
            ]
        ),
        encoding="utf-8",
    )

    if "hint_ner" in nlp.pipe_names:
        nlp.remove_pipe("hint_ner")
    nlp.add_pipe("hint_ner", config={"lexicon_path": str(lexicon_path)})

    doc = nlp("The patient reports chest pain worsening at night.")
    hint_entities = [ent for ent in doc.ents if getattr(ent._, "hint_id", None) == "pain-uid"]
    assert hint_entities, "Expected HintNER to add an entity span"
    hint = hint_entities[0]
    assert hint.label_ == "Chest Pain"
    assert hint._.hint_source == "phrase"
    assert hint._.hint_cluster_title == "Symptom"
    assert hint._.hint_canonical_keyword == "Chest Pain"
    assert "chest pain" in hint.text.lower()


@pytest.mark.skipif(spacy is None, reason="spaCy is required for hint_ner tests")
def test_hint_ner_canonical_only(tmp_path: Path) -> None:
    try:
        nlp = spacy.load("en_core_web_md")
    except OSError:
        pytest.skip("en_core_web_md model is not installed")
    lexicon_path = tmp_path / "lexicon.json"
    lexicon_path.write_text(
        json.dumps(
            [
                {
                    "uid": "bp-uid",
                    "cluster_id": "cluster-bp",
                    "cluster_title": "Finding",
                    "canonical_keyword": "Blood Pressure",
                    "keyword_terms": [],
                    "sources": ["unit-test"],
                }
            ]
        ),
        encoding="utf-8",
    )

    if "hint_ner" in nlp.pipe_names:
        nlp.remove_pipe("hint_ner")
    nlp.add_pipe("hint_ner", config={"lexicon_path": str(lexicon_path)})

    doc = nlp("Vitals today show the blood pressure remains stable and within range.")
    hint_entities = [ent for ent in doc.ents if getattr(ent._, "hint_id", None) == "bp-uid"]
    assert hint_entities, "Expected HintNER to add an entity span from canonical keyword"
    hint = hint_entities[0]
    assert hint.label_ == "Blood Pressure"
    assert hint._.hint_source == "phrase"
    assert hint._.hint_cluster_title == "Finding"
    assert hint._.hint_term.lower() == "blood pressure"
    assert hint._.hint_canonical_keyword == "Blood Pressure"
