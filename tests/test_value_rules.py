from __future__ import annotations

from typing import Dict, Any

from src.custom_cat_v2 import CustomCAT


def _make_entity(cui: str, text: str, phrase: str) -> Dict[str, Any]:
    start = text.lower().index(phrase.lower())
    end = start + len(phrase)
    return {
        "cui": cui,
        "start": start,
        "end": end,
        "detected_name": phrase.replace(" ", "~"),
        "source_value": phrase,
        "acc": 1.0,
    }


def test_numeric_value_required(custom_cat: CustomCAT) -> None:
    text = "Heart rate 120 bpm recorded during check."
    entities: Dict[Any, Dict[str, Any]] = {
        0: _make_entity("5b51b989ada20c282c2487da", text, "Heart rate")
    }

    custom_cat._apply_value_rules(text, entities)

    assert entities, "Numeric entity should be kept when value is present and within range."


def test_numeric_out_of_range_removed(custom_cat: CustomCAT) -> None:
    text = "BMI documented as 100 during admission."
    entities: Dict[Any, Dict[str, Any]] = {
        0: _make_entity("5E12E779A13C2347A094922C", text, "BMI")
    }

    custom_cat._apply_value_rules(text, entities)

    assert not entities, "Numeric entity outside allowable range must be dropped."


def test_numeric_without_value_removed(custom_cat: CustomCAT) -> None:
    text = "Heart rate remains elevated after exercise."
    entities: Dict[Any, Dict[str, Any]] = {
        0: _make_entity("5B51B989ADA20C282C2487DA", text, "Heart rate")
    }

    custom_cat._apply_value_rules(text, entities)

    assert not entities, "Numeric entity without an accompanying value must be removed."


def test_string_cluster_requires_value(custom_cat: CustomCAT) -> None:
    text = "Antibody COVID-19 IgM test returned positive result."
    entities: Dict[Any, Dict[str, Any]] = {
        0: _make_entity("5F2038635637E90374245A14", text, "Antibody COVID-19 IgM test")
    }

    custom_cat._apply_value_rules(text, entities)

    assert entities, "String cluster entity should be kept when textual value hint is present."


def test_string_cluster_without_value_removed(custom_cat: CustomCAT) -> None:
    text = "Antibody COVID-19 IgM test was ordered yesterday."
    entities: Dict[Any, Dict[str, Any]] = {
        0: _make_entity("5f2038635637e90374245a14", text, "Antibody COVID-19 IgM test")
    }

    custom_cat._apply_value_rules(text, entities)

    assert not entities, "String cluster entity without a value hint must be removed."


def test_extract_entities_adds_numeric_with_value(custom_cat: CustomCAT) -> None:
    text = "My heart rate is high (120)."
    result = custom_cat.extract_entities(text)

    detected_cuis = {ent["cui"] for ent in result.get("entities", {}).values()}
    assert "5B51B989ADA20C282C2487DA" in detected_cuis, "Heart rate with numeric hint should be restored."
