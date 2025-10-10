"""Performance benchmarking utilities for MedCAT pipelines."""

from __future__ import annotations

import time
from typing import Iterable, Sequence

import psutil


class PerformanceBenchmark:
    """Measure processing speed and memory usage for MedCAT extractions."""

    def __init__(self, cat) -> None:
        self.cat = cat
        self._process = psutil.Process()

    def benchmark_processing_speed(
        self,
        documents: Sequence[str],
        batch_sizes: Iterable[int] = (1, 10, 50),
    ) -> dict[int, dict[str, float]]:
        """Evaluate throughput and memory metrics across batch sizes."""

        if not documents:
            raise ValueError("At least one document is required for benchmarking.")

        results: dict[int, dict[str, float]] = {}
        for batch_size in batch_sizes:
            if batch_size <= 0:
                raise ValueError(f"Batch size must be positive, received {batch_size}.")
            metrics = self._benchmark_single_run(documents, batch_size)
            results[batch_size] = metrics
        return results

    def _benchmark_single_run(self, documents: Sequence[str], batch_size: int) -> dict[str, float]:
        total_documents = len(documents)
        start_time = time.perf_counter()
        start_memory = self._memory_mb()
        peak_memory = start_memory

        for index in range(0, total_documents, batch_size):
            batch = documents[index : index + batch_size]
            self.cat.batch_process(batch)
            peak_memory = max(peak_memory, self._memory_mb())

        total_time = time.perf_counter() - start_time
        end_memory = self._memory_mb()
        docs_per_second = total_documents / total_time if total_time else float("inf")

        return {
            "batch_size": float(batch_size),
            "total_time": total_time,
            "docs_per_second": docs_per_second,
            "start_memory_mb": start_memory,
            "end_memory_mb": end_memory,
            "memory_delta_mb": end_memory - start_memory,
            "peak_memory_mb": peak_memory,
            "documents_processed": float(total_documents),
        }

    def _memory_mb(self) -> float:
        return self._process.memory_info().rss / (1024 * 1024)
