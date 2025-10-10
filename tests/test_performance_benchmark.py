"""Performance benchmarking regression tests."""

from __future__ import annotations

from src.testing_framework import PerformanceBenchmark


def test_performance_benchmark_returns_metrics(custom_cat, annotated_entity_dataset) -> None:
    documents = [doc["text"] for doc in annotated_entity_dataset]
    benchmark = PerformanceBenchmark(custom_cat)
    results = benchmark.benchmark_processing_speed(documents, batch_sizes=(1, 2))

    assert set(results.keys()) == {1, 2}
    for metrics in results.values():
        assert metrics["documents_processed"] == len(documents)
        assert metrics["docs_per_second"] > 0
        assert "total_time" in metrics and metrics["total_time"] >= 0
        assert "memory_delta_mb" in metrics
