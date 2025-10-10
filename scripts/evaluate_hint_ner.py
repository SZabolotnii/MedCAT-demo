#!/usr/bin/env python
"""Evaluate the HintNER component against a labelled dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spacy_pipeline import iter_hint_spans, load_spacy_with_hints


def _load_dataset(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Dataset must be a list of records with 'text' and 'expected_cuis'.")
    return data


def _normalize_ids(ids: Iterable[str]) -> List[str]:
    return sorted({str(item).strip().lower() for item in ids if item})


def evaluate_dataset(
    samples: Sequence[Dict[str, Any]],
    *,
    model_name: str,
    lexicon_path: Path,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    try:
        nlp = load_spacy_with_hints(
            model_name,
            lexicon_path=lexicon_path,
        )
    except OSError as exc:
        raise SystemExit(
            f"Unable to load spaCy model '{model_name}'. "
            "Install it via `python -m spacy download en_core_web_md` or specify --model."
        ) from exc

    totals = {"tp": 0, "fp": 0, "fn": 0}
    per_sample: List[Dict[str, Any]] = []

    for sample in samples:
        text = sample.get("text") or ""
        expected_ids = _normalize_ids(sample.get("expected_cuis") or [])

        doc = nlp(text)
        predicted_spans = [
            {
                "text": span.text,
                "label": span.label_,
                "start": span.start_char,
                "end": span.end_char,
                "hint_id": span._.hint_id,
                "hint_source": span._.hint_source,
                "hint_score": float(span._.hint_score or 0.0),
                "hint_term": span._.hint_term,
                "hint_cluster_title": span._.hint_cluster_title,
                "hint_cluster_id": span._.hint_cluster_id,
                "hint_canonical_keyword": span._.hint_canonical_keyword,
            }
            for span in iter_hint_spans(doc)
        ]
        predicted_ids = _normalize_ids(span["hint_id"] for span in predicted_spans)

        expected_set = set(expected_ids)
        predicted_set = set(predicted_ids)

        tp = len(expected_set & predicted_set)
        fp = len(predicted_set - expected_set)
        fn = len(expected_set - predicted_set)

        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn

        per_sample.append(
            {
                "text": text,
                "expected_ids": expected_ids,
                "predicted_ids": predicted_ids,
                "true_positives": sorted(expected_set & predicted_set),
                "false_positives": sorted(predicted_set - expected_set),
                "false_negatives": sorted(expected_set - predicted_set),
                "predicted_spans": predicted_spans,
            }
        )

    precision = totals["tp"] / (totals["tp"] + totals["fp"]) if (totals["tp"] + totals["fp"]) else 0.0
    recall = totals["tp"] / (totals["tp"] + totals["fn"]) if (totals["tp"] + totals["fn"]) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    overview = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": totals["tp"],
        "false_positive": totals["fp"],
        "false_negative": totals["fn"],
    }

    return overview, per_sample


def _print_metrics(metrics: Dict[str, float]) -> None:
    print("HintNER evaluation metrics")
    print("--------------------------")
    print(f"Precision : {metrics['precision']:.3f}")
    print(f"Recall    : {metrics['recall']:.3f}")
    print(f"F1-score  : {metrics['f1']:.3f}")
    print(f"TP={int(metrics['true_positive'])} FP={int(metrics['false_positive'])} FN={int(metrics['false_negative'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HintNER against a labelled dataset.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/test_clinical_notes.json"),
        help="Path to evaluation dataset with 'text' and 'expected_cuis' fields.",
    )
    parser.add_argument("--model", default="en_core_web_md", help="spaCy model to use for evaluation.")
    parser.add_argument(
        "--lexicon",
        type=Path,
        default=Path("data/hints/hint_lexicon.json"),
        help="Path to hint lexicon JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write full evaluation report (JSON).",
    )

    args = parser.parse_args()
    dataset = _load_dataset(args.dataset)

    metrics, sample_details = evaluate_dataset(
        dataset,
        model_name=args.model,
        lexicon_path=args.lexicon,
    )

    _print_metrics(metrics)

    if args.output:
        report = {"metrics": metrics, "samples": sample_details}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print(f"Wrote detailed report to {args.output}")


if __name__ == "__main__":
    main()
