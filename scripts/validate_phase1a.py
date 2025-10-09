"""Lightweight validation runner for the Phase 1A custom ontology."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from src.custom_cat_v2 import CustomCAT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/IEE_MedCAT_v1"),
        help="Path to the custom MedCAT pack directory or zip.",
    )
    parser.add_argument(
        "--combined-hints",
        type=Path,
        default=Path("data/internal_combined_hints.json"),
        help="Combined hints metadata file (JSON).",
    )
    parser.add_argument(
        "--test-set",
        type=Path,
        default=Path("data/test_clinical_notes.json"),
        help="JSON file with test documents and expected CUIs (optional).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/phase1a_validation.md"),
        help="Path where the Markdown report will be written.",
    )
    return parser.parse_args()


def load_test_documents(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Test set not found: {path}")
    with path.open("r", encoding="utf-8") as src:
        data = json.load(src)
    if not isinstance(data, list):
        raise ValueError("Test set must be a list of documents")
    return data


def run_validation(cat: CustomCAT, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {
        "total_docs": len(documents),
        "total_entities": 0,
        "avg_entities_per_doc": 0.0,
        "documents": [],
        "expected_cui_hits": 0,
        "expected_cui_total": 0,
    }

    for doc in documents:
        text = doc.get("text", "")
        expected_cuis = [str(c) for c in doc.get("expected_cuis", [])]
        result = cat.extract_entities(text)
        entities = result.get("entities", {})
        cuis_in_doc = {str(entity.get("cui")) for entity in entities.values() if entity.get("cui")}

        summary["total_entities"] += len(entities)
        summary["documents"].append(
            {
                "text_preview": text[:160] + ("..." if len(text) > 160 else ""),
                "entity_count": len(entities),
                "cuis": sorted(cuis_in_doc),
                "combined_matches": result.get("combined_hint_matches", []),
            }
        )

        summary["expected_cui_total"] += len(expected_cuis)
        summary["expected_cui_hits"] += sum(1 for cui in expected_cuis if cui in cuis_in_doc)

    if documents:
        summary["avg_entities_per_doc"] = summary["total_entities"] / len(documents)

    return summary


def write_report(summary: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Phase 1A Validation Report",
        "",
        f"*Total documents:* {summary['total_docs']}",
        f"*Total entities:* {summary['total_entities']}",
        f"*Avg entities per document:* {summary['avg_entities_per_doc']:.2f}",
        "",
    ]

    if summary["expected_cui_total"]:
        precision = (
            summary["expected_cui_hits"] / summary["expected_cui_total"] * 100.0
            if summary["expected_cui_total"]
            else 0.0
        )
        lines.append(
            f"*Expected CUI coverage:* {summary['expected_cui_hits']} / {summary['expected_cui_total']} "
            f"({precision:.1f}% hits)"
        )
        lines.append("")

    lines.append("## Sample Documents")
    lines.append("")

    for idx, doc in enumerate(summary["documents"], start=1):
        lines.append(f"### Document {idx}")
        lines.append(doc["text_preview"])
        lines.append(f"\n- Entities detected: {doc['entity_count']}")
        if doc["cuis"]:
            lines.append(f"- CUIs: {', '.join(doc['cuis'])}")
        if doc["combined_matches"]:
            lines.append("- Combined hint matches:")
            for match in doc["combined_matches"]:
                lines.append(
                    f"  - {match['matched_text']} → {match['cui']} (source: {match['source_hint']})"
                )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    documents = load_test_documents(args.test_set)
    cat = CustomCAT(args.model, combined_hints_path=args.combined_hints)
    summary = run_validation(cat, documents)
    write_report(summary, args.report)

    print("✅ Validation complete")
    print(f"   Documents: {summary['total_docs']}")
    print(f"   Total entities: {summary['total_entities']}")
    if summary["expected_cui_total"]:
        coverage = summary["expected_cui_hits"] / summary["expected_cui_total"] * 100.0
        print(f"   Expected CUI coverage: {coverage:.1f}%")


if __name__ == "__main__":
    main()
