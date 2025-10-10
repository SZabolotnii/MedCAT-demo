"""End-to-end validation runner for the MedCAT custom ontology."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from medcat.cdb import CDB

from src.custom_cat_v2 import CustomCAT
from src.testing_framework import EntityDetectionValidator, PerformanceBenchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/IEE_MedCAT_v1"),
        help="Path to the MedCAT model pack directory or zip.",
    )
    parser.add_argument(
        "--combined-hints",
        type=Path,
        default=Path("models/IEE_MedCAT_v1/internal_combined_hints.json"),
        help="Combined hints metadata file.",
    )
    parser.add_argument(
        "--dictionary-stats",
        type=Path,
        default=Path("models/IEE_MedCAT_v1/cdb_stats.json"),
        help="Path to cdb_stats.json with expected coverage values.",
    )
    parser.add_argument(
        "--cluster-mapping",
        type=Path,
        default=Path("data/valid_clusters.json"),
        help="Cluster mapping JSON exported from the ontology.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/phase1a_annotated_entities.json"),
        help="Annotated dataset for entity detection evaluation.",
    )
    parser.add_argument(
        "--performance-docs",
        type=Path,
        default=Path("data/test_docs"),
        help="Directory or JSON file with documents for performance benchmarking.",
    )
    parser.add_argument(
        "--performance-batch-sizes",
        type=int,
        nargs="+",
        default=(1, 10, 50),
        help="Batch sizes for performance benchmarking.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/validation_suite.json"),
        help="Where to write the JSON validation report.",
    )
    return parser.parse_args()


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_cluster_mapping(path: Path) -> Dict[str, str]:
    raw = load_json(path)
    return {str(entry["id"]).upper(): entry["title"] for entry in raw}


def load_annotations(path: Path) -> List[dict]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Annotation dataset must be a list, got {type(data)}")
    return data


def load_performance_documents(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Performance documents not found: {path}")
    if path.is_dir():
        documents: List[str] = []
        for file in sorted(path.glob("*.txt")):
            text = file.read_text(encoding="utf-8").strip()
            if text:
                documents.append(text)
        if not documents:
            raise ValueError(f"No .txt files found in performance directory: {path}")
        return documents

    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        if suffix == ".json":
            payload = load_json(path)
            data = payload if isinstance(payload, list) else [payload]
        else:
            data = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        documents = [str(item["text"]) for item in data if item.get("text")]
        if not documents:
            raise ValueError(f"No documents found in {path}")
        return documents

    raise ValueError(f"Unsupported performance documents source: {path}")


def run_dictionary_validation(cdb: CDB, stats: dict, cluster_mapping: Dict[str, str]) -> dict:
    total_cuis = len(cdb.cui2info)
    missing_names = [cui for cui, info in cdb.cui2info.items() if not info.get("names")]
    missing_preferred = [cui for cui, info in cdb.cui2info.items() if not info.get("preferred_name")]
    coverage = 1 - len(missing_preferred) / total_cuis if total_cuis else 0.0
    distinct_type_ids = {
        str(type_id).upper()
        for info in cdb.cui2info.values()
        for type_id in (info.get("type_ids") or [])
    }
    unmapped_types = [type_id for type_id in distinct_type_ids if type_id not in cluster_mapping]

    success = (
        total_cuis == int(stats["total_cuis"])
        and not missing_names
        and coverage >= float(stats["preferred_names_coverage"]) / 100.0 - 0.001
        and len(distinct_type_ids) == int(stats["type_ids_count"])
        and not unmapped_types
    )

    return {
        "success": success,
        "total_cuis": total_cuis,
        "expected_cuis": int(stats["total_cuis"]),
        "missing_names": len(missing_names),
        "missing_preferred": len(missing_preferred),
        "preferred_name_coverage": coverage,
        "expected_preferred_name_coverage": float(stats["preferred_names_coverage"]) / 100.0,
        "distinct_type_ids": len(distinct_type_ids),
        "expected_type_ids": int(stats["type_ids_count"]),
        "unmapped_type_ids": unmapped_types[:10],
    }


def normalize_entity(entity: dict) -> dict:
    return {
        "cui": str(entity["cui"]).upper(),
        "start": int(entity["start"]),
        "end": int(entity["end"]),
        "type_ids": [str(type_id).upper() for type_id in entity.get("type_ids", [])],
    }


def extract_predicted_entities(result: dict, cdb: CDB) -> List[dict]:
    predicted: List[dict] = []
    for entity in (result.get("entities") or {}).values():
        if not isinstance(entity, dict):
            continue
        cui = entity.get("cui")
        start = entity.get("start")
        end = entity.get("end")
        if cui is None or start is None or end is None:
            continue
        cui_str = str(cui).upper()
        type_ids = entity.get("type_ids") or entity.get("types") or []
        types = {str(type_id).upper() for type_id in type_ids if type_id}
        if not types and cui_str in cdb.cui2info:
            types = {str(type_id).upper() for type_id in cdb.cui2info[cui_str].get("type_ids", [])}
        predicted.append(
            {
                "cui": cui_str,
                "start": int(start),
                "end": int(end),
                "type_ids": sorted(types),
            }
        )
    return predicted


def run_entity_validation(cat: CustomCAT, cdb: CDB, annotations: Sequence[dict]) -> dict:
    validator = EntityDetectionValidator()
    predicted_entities: List[dict] = []
    gold_entities: List[dict] = []

    for document in annotations:
        result = cat.extract_entities(document["text"])
        predicted_entities.extend(extract_predicted_entities(result, cdb))
        gold_entities.extend(normalize_entity(entity) for entity in document["entities"])

    metrics = validator.calculate_metrics(predicted_entities, gold_entities)
    exact_f1 = metrics["exact_match"]["f1"]
    partial_f1 = metrics["partial_match"]["f1"]
    type_accuracy = metrics["type_accuracy"]["accuracy"]

    success = exact_f1 >= 0.75 and partial_f1 >= 0.80 and type_accuracy >= 0.85

    return {
        "success": success,
        "metrics": metrics,
        "thresholds": {
            "exact_f1": 0.75,
            "partial_f1": 0.80,
            "type_accuracy": 0.85,
        },
    }


def run_combined_hint_validation(cat: CustomCAT) -> dict:
    target_cui = "60758574BE555105F0BC5B6B"
    cases = [
        ("Care team will check sugar before discharge.", True),
        ("Care team will check her blood sugar before discharge.", True),
        ("Care team will check her morning fasting sugar logs daily.", True),
        ("Care team will check her early morning fasting sugar routines carefully.", False),
    ]

    results = []
    all_ok = True
    for text, expected in cases:
        outcome = cat.extract_entities(text)
        matches = outcome.get("combined_hint_matches") or []
        matched = any(str(match.get("cui")).upper() == target_cui for match in matches)
        results.append({"text": text, "expected": expected, "matched": matched})
        if matched != expected:
            all_ok = False

    return {"success": all_ok, "cases": results, "target_cui": target_cui}


def run_performance_benchmark(cat: CustomCAT, documents: List[str], batch_sizes: Iterable[int]) -> dict:
    benchmark = PerformanceBenchmark(cat)
    results = benchmark.benchmark_processing_speed(documents, batch_sizes=batch_sizes)

    success = True
    for metrics in results.values():
        if metrics["docs_per_second"] < 10 or metrics["memory_delta_mb"] > 2048:
            success = False
            break

    return {"success": success, "results": results, "thresholds": {"docs_per_second": 10, "memory_delta_mb": 2048}}


def main() -> None:
    args = parse_args()

    stats = load_json(args.dictionary_stats)
    cluster_mapping = load_cluster_mapping(args.cluster_mapping)
    annotations = load_annotations(args.annotations)
    performance_documents = load_performance_documents(args.performance_docs)

    cat = CustomCAT(args.model, combined_hints_path=args.combined_hints)
    cdb = cat.cdb

    dictionary_summary = run_dictionary_validation(cdb, stats, cluster_mapping)
    entity_summary = run_entity_validation(cat, cdb, annotations)
    combined_hint_summary = run_combined_hint_validation(cat)
    performance_summary = run_performance_benchmark(cat, performance_documents, args.performance_batch_sizes)

    payload = {
        "dictionary": dictionary_summary,
        "entity_detection": entity_summary,
        "combined_hints": combined_hint_summary,
        "performance": performance_summary,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("✅ Validation suite complete")
    print(f"   Dictionary success: {dictionary_summary['success']}")
    print(f"   Entity detection success: {entity_summary['success']}")
    print(f"   Combined hints success: {combined_hint_summary['success']}")
    print(f"   Performance success: {performance_summary['success']}")


if __name__ == "__main__":
    main()
