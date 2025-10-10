"""Benchmark MedCAT extraction performance for the custom ontology."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from src.custom_cat_v2 import CustomCAT
from src.testing_framework import PerformanceBenchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/IEE_MedCAT_v1"),
        help="Path to the MedCAT model pack directory or zip file.",
    )
    parser.add_argument(
        "--combined-hints",
        type=Path,
        default=Path("models/IEE_MedCAT_v1/internal_combined_hints.json"),
        help="Path to the combined hints metadata JSON.",
    )
    parser.add_argument(
        "--documents",
        type=Path,
        default=Path("data/test_docs"),
        help="Directory of .txt files or JSON(.l) file containing benchmark texts.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=(1, 10, 50),
        help="Batch sizes to evaluate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/performance_benchmark.json"),
        help="Where to write the benchmark JSON report.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Optional baseline JSON file for comparison.",
    )
    return parser.parse_args()


def load_documents(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark source not found: {path}")

    if path.is_dir():
        documents = []
        for file in sorted(path.glob("*.txt")):
            text = file.read_text(encoding="utf-8").strip()
            if text:
                documents.append(text)
        if not documents:
            raise ValueError(f"No .txt documents found in directory: {path}")
        return documents

    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        content = path.read_text(encoding="utf-8").splitlines() if suffix == ".jsonl" else [path.read_text(encoding="utf-8")]
        documents: List[str] = []
        for chunk in content:
            data = json.loads(chunk)
            if isinstance(data, list):
                documents.extend(_extract_texts(data))
            elif isinstance(data, dict):
                documents.extend(_extract_texts([data]))
            else:
                raise ValueError(f"Unsupported JSON structure in {path}")
        if not documents:
            raise ValueError(f"No document texts found in {path}")
        return documents

    raise ValueError(f"Unsupported documents source: {path}")


def _extract_texts(items: Sequence[dict]) -> List[str]:
    texts: List[str] = []
    for item in items:
        text = item.get("text")
        if text:
            texts.append(str(text))
    return texts


def load_baseline(path: Path | None) -> Dict[str, dict] | None:
    if not path:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Baseline file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def compare_results(
    current: Dict[int, Dict[str, float]], baseline: Dict[str, dict] | None
) -> Dict[str, Dict[str, float]]:
    if baseline is None:
        return {}

    deltas: Dict[str, Dict[str, float]] = {}
    for batch_size, metrics in current.items():
        baseline_metrics = baseline.get(str(batch_size))
        if not baseline_metrics:
            continue
        deltas[str(batch_size)] = {
            key: metrics[key] - baseline_metrics.get(key, 0.0)
            for key in ("docs_per_second", "total_time", "memory_delta_mb", "peak_memory_mb")
            if key in metrics
        }
    return deltas


def main() -> None:
    args = parse_args()
    documents = load_documents(args.documents)
    cat = CustomCAT(args.model, combined_hints_path=args.combined_hints)
    benchmark = PerformanceBenchmark(cat)
    results = benchmark.benchmark_processing_speed(documents, batch_sizes=args.batch_sizes)

    baseline = load_baseline(args.baseline)
    deltas = compare_results(results, baseline)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialisable_results = {str(batch): metrics for batch, metrics in results.items()}
    payload = {"results": serialisable_results, "deltas": deltas, "document_count": len(documents)}
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("✅ Performance benchmarking complete")
    print(f"   Documents: {len(documents)}")
    for batch_size, metrics in results.items():
        print(
            f"   Batch {batch_size}: {metrics['docs_per_second']:.2f} docs/s | "
            f"ΔMemory {metrics['memory_delta_mb']:.2f} MB"
        )
    if deltas:
        print("   Compared against baseline:")
        for batch_size, metrics in deltas.items():
            change = metrics.get("docs_per_second", 0.0)
            print(f"     Batch {batch_size}: Δdocs/s {change:+.2f}")


if __name__ == "__main__":
    main()
