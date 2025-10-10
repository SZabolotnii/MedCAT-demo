"""Entity detection accuracy tests using the custom ontology."""

from __future__ import annotations

from typing import Dict, Iterable, List

from medcat.cdb import CDB

from src.testing_framework import EntityDetectionValidator


def _normalize_predicted_entities(result: Dict[str, object], cdb: CDB) -> List[Dict[str, object]]:
    entities: List[Dict[str, object]] = []
    raw_entities = result.get("entities", {}) if isinstance(result, dict) else {}

    for entity in raw_entities.values():
        if not isinstance(entity, dict):
            continue

        cui = entity.get("cui")
        start = entity.get("start")
        end = entity.get("end")
        if cui is None or start is None or end is None:
            continue

        cui_str = str(cui).upper()
        type_ids: Iterable[str] = entity.get("type_ids") or entity.get("types") or ()
        normalized_types = {str(type_id).upper() for type_id in type_ids if type_id}

        if not normalized_types and cui_str in cdb.cui2info:
            normalized_types = {str(type_id).upper() for type_id in cdb.cui2info[cui_str].get("type_ids", [])}

        entities.append(
            {
                "cui": cui_str,
                "start": int(start),
                "end": int(end),
                "type_ids": sorted(normalized_types),
            }
        )

    return entities


def _normalize_gold_entities(raw_entities: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    normalized: List[Dict[str, object]] = []
    for entity in raw_entities:
        cui = str(entity["cui"]).upper()
        normalized.append(
            {
                "cui": cui,
                "start": int(entity["start"]),
                "end": int(entity["end"]),
                "type_ids": [str(type_id).upper() for type_id in entity.get("type_ids", [])],
            }
        )
    return normalized


def test_entity_detection_metrics(custom_cat, cdb: CDB, annotated_entity_dataset) -> None:
    validator = EntityDetectionValidator()
    predicted: List[Dict[str, object]] = []
    gold: List[Dict[str, object]] = []

    for document in annotated_entity_dataset:
        result = custom_cat.extract_entities(document["text"])
        predicted.extend(_normalize_predicted_entities(result, cdb))
        gold.extend(_normalize_gold_entities(document["entities"]))

    metrics = validator.calculate_metrics(predicted, gold)

    assert metrics["exact_match"]["f1"] >= 0.95
    assert metrics["partial_match"]["f1"] >= 0.95
    assert metrics["type_accuracy"]["accuracy"] >= 0.95
