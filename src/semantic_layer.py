"""Semantic similarity scaffolding for the optional Phase 1B layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class SemanticConfig:
    model_name: str = "emilyalsentzer/Bio_ClinicalBERT"
    device: str = "cpu"
    similarity_threshold: float = 0.8


class SemanticMatcher:
    """Skeleton implementation for the future semantic similarity module."""

    def __init__(self, config: SemanticConfig | None = None) -> None:
        self.config = config or SemanticConfig()

    def load(self) -> None:  # pragma: no cover - placeholder
        raise NotImplementedError("Semantic layer will be implemented in Phase 1B")

    def build_concept_index(self, names: List[str]) -> None:  # pragma: no cover - placeholder
        raise NotImplementedError("Semantic layer will be implemented in Phase 1B")

    def match(self, text: str):  # pragma: no cover - placeholder
        raise NotImplementedError("Semantic layer will be implemented in Phase 1B")
