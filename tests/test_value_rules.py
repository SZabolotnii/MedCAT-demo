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
    hints = entities[0].get("value_hints")
    assert hints and hints[0]["type"] == "numeric"


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

    hr_entity = next(ent for ent in result["entities"].values() if ent["cui"] == "5B51B989ADA20C282C2487DA")
    hints = hr_entity.get("value_hints")
    assert hints and hints[0]["type"] == "numeric"

    level_entity = next(ent for ent in result["entities"].values() if ent["cui"] == "5F59DFE25786951388090907")
    level_hint = level_entity.get("value_hints")
    assert level_hint and level_hint[0]["type"] == "text"


def test_combination_requires_all_components(custom_cat: CustomCAT) -> None:
    text = "Aspirin 100 mg taken daily."
    result = custom_cat.extract_entities(text)

    detected = {ent["cui"] for ent in result.get("entities", {}).values()}
    assert "5E5D196B91AC1162F7F7B549" in detected, "Base Aspirin dosage CUI expected."

    combo_cuis = {
        "616582D1DA2D098889C2DD19",  # Aspirin / Dipyridamole
        "6539F7B3063E915B2098DB0B",  # Acetaminophen / Aspirin / Caffeine
        "634CEC8BD4649E5E733DE2E7",  # Aspirin / butalbital / caffeine
    }
    assert detected.isdisjoint(combo_cuis), "Combined drug CUIs must not be emitted without all components."


def test_textual_value_keeps_string_cluster(custom_cat: CustomCAT) -> None:
    text = "My heart rate is high (120)."
    result = custom_cat.extract_entities(text)

    detected_cuis = {ent["cui"] for ent in result.get("entities", {}).values()}
    assert "5F59DFE25786951388090907" in detected_cuis, "Heart Rate Level expected with textual value hint."

    entity = next(ent for ent in result["entities"].values() if ent["cui"] == "5F59DFE25786951388090907")
    hints = entity.get("value_hints")
    assert hints and hints[0]["type"] == "text" and "high" in hints[0]["value"].lower()
