"""Gap-tolerant combined hint tests for the custom ontology."""

from __future__ import annotations

TARGET_CUI = "60758574BE555105F0BC5B6B"


def _has_target_combined_match(result: dict) -> bool:
    matches = result.get("combined_hint_matches") or []
    return any(str(match.get("cui")).upper() == TARGET_CUI for match in matches)


def test_combined_hint_gap_tolerance(custom_cat) -> None:
    cases = [
        ("Care team will check sugar before discharge.", True),
        ("Care team will check her blood sugar before discharge.", True),
        ("Care team will check her morning fasting sugar logs daily.", True),
        ("Care team will check her early morning fasting sugar routines carefully.", False),
    ]

    for text, expected in cases:
        result = custom_cat.extract_entities(text)
        assert _has_target_combined_match(result) is expected, f"Unexpected combined hint result for: {text}"
