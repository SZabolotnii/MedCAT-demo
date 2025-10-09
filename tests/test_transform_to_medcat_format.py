"""Unit tests for the MedCAT dictionary transformer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.transform_to_medcat_format import (  # noqa: E402
    collect_combined_hint_records,
    expand_keywords,
    parse_hint,
)


def test_parse_hint_combined():
    parsed = parse_hint("aerosol [combined_hint] intranasally")
    assert parsed.is_combined is True
    assert parsed.cleaned == "aerosol intranasally"
    assert parsed.components == ["aerosol", "intranasally"]


def test_parse_hint_regular():
    parsed = parse_hint("simple synonym")
    assert parsed.is_combined is False
    assert parsed.cleaned == "simple synonym"
    assert parsed.components == ["simple synonym"]


def build_row(**overrides: str) -> dict[str, str]:
    base = {
        "uid": "UID123",
        "keyword": "Test Keyword",
        "cluster": "cluster_1",
        "cluster_title": "Findings/General",
        "source": "values_as_hints",
        "keyword_hints": "",
    }
    base.update(overrides)
    return base


def test_expand_keywords_creates_primary_and_synonyms():
    row = build_row(
        keyword_hints="Synonym One|multi word [combined_hint] phrase",
    )
    expanded = list(expand_keywords([row], ontology="INTERNAL", max_gap=3))

    assert len(expanded) == 3

    primary = expanded[0]
    assert primary["name_status"] == "P"
    assert primary["name"] == "Test Keyword"
    assert primary["metadata_json"] == ""

    synonyms = expanded[1:]
    names = {entry["name"] for entry in synonyms}
    assert names == {"Synonym One", "multi word phrase"}

    combined_entry = next(entry for entry in synonyms if entry["name"] == "multi word phrase")
    metadata = json.loads(combined_entry["metadata_json"])
    combined = metadata["combined_hint"]
    assert combined["components"] == ["multi word", "phrase"]
    assert combined["max_gap"] == 3


def test_expand_keywords_skips_duplicate_hints():
    row = build_row(keyword_hints="Test Keyword|test keyword|NEW")
    expanded = list(expand_keywords([row]))

    names = [entry["name"] for entry in expanded]
    assert names.count("Test Keyword") == 1
    assert "NEW" in names


def test_collect_combined_hint_records():
    row = build_row(keyword_hints="multi word [combined_hint] phrase|simple")
    records = collect_combined_hint_records([row], max_gap=2)
    assert len(records) == 1
    record = records[0]
    assert record.cui == "UID123"
    assert record.components == ["multi word", "phrase"]
    assert record.max_gap == 2
