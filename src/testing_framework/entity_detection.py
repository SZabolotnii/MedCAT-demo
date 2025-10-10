"""Entity detection validation helpers for the MedCAT testing framework."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence


@dataclass(frozen=True)
class _Entity:
    """Normalized representation of an annotated entity span."""

    cui: str
    start: int
    end: int
    type_ids: frozenset[str]

    @property
    def span(self) -> tuple[int, int]:
        return self.start, self.end


def _to_entity(record: Mapping[str, object]) -> _Entity:
    """Convert an arbitrary mapping into an ``_Entity`` instance."""

    if "cui" not in record or "start" not in record or "end" not in record:
        missing = {key for key in ("cui", "start", "end") if key not in record}
        raise ValueError(f"Entity record is missing required keys: {sorted(missing)}")

    cui = str(record["cui"]).upper()
    start = int(record["start"])
    end = int(record["end"])

    raw_types = record.get("type_ids") or record.get("types") or ()
    type_ids = frozenset(str(type_id).upper() for type_id in raw_types if type_id)

    if start >= end:
        raise ValueError(f"Invalid entity span for CUI {cui}: start={start}, end={end}")

    return _Entity(cui=cui, start=start, end=end, type_ids=type_ids)


def _precision_recall_f1(true_positive: int, false_positive: int, false_negative: int) -> dict[str, float]:
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    if precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(true_positive),
        "fp": float(false_positive),
        "fn": float(false_negative),
    }


class EntityDetectionValidator:
    """Comprehensive entity detection validation framework."""

    def calculate_metrics(
        self,
        predicted_entities: Sequence[Mapping[str, object]],
        gold_entities: Sequence[Mapping[str, object]],
    ) -> dict[str, object]:
        """Calculate precision, recall, and F1 metrics for MedCAT entity detection.

        Args:
            predicted_entities: Iterable of detected entities. Each mapping must contain
                ``cui``, ``start`` (inclusive index), ``end`` (exclusive index), and optional
                ``type_ids``/``types`` collections.
            gold_entities: Iterable of gold-standard annotations with the same schema.

        Returns:
            A dictionary containing summary metrics for exact matching, partial matching,
            and type-level accuracy.
        """

        predicted = [_to_entity(entity) for entity in predicted_entities]
        gold = [_to_entity(entity) for entity in gold_entities]

        exact = self._calculate_exact_matches(predicted, gold)
        partial = self._calculate_partial_matches(predicted, gold)
        type_accuracy = self._calculate_type_accuracy(predicted, gold)

        return {
            "exact_match": exact,
            "partial_match": partial,
            "type_accuracy": type_accuracy,
            "entity_count": len(predicted),
            "gold_count": len(gold),
        }

    @staticmethod
    def _calculate_exact_matches(predicted: List[_Entity], gold: List[_Entity]) -> dict[str, float]:
        gold_lookup = {(entity.start, entity.end, entity.cui): idx for idx, entity in enumerate(gold)}

        true_positive = 0
        matched_gold: set[int] = set()

        for entity in predicted:
            key = (entity.start, entity.end, entity.cui)
            gold_idx = gold_lookup.get(key)
            if gold_idx is None or gold_idx in matched_gold:
                continue
            true_positive += 1
            matched_gold.add(gold_idx)

        false_positive = len(predicted) - true_positive
        false_negative = len(gold) - true_positive

        metrics = _precision_recall_f1(true_positive, false_positive, false_negative)
        metrics.update(
            {
                "matched": float(true_positive),
                "total_predicted": float(len(predicted)),
                "total_gold": float(len(gold)),
            }
        )
        return metrics

    def _calculate_partial_matches(self, predicted: Sequence[_Entity], gold: Sequence[_Entity]) -> dict[str, float]:
        used_gold: set[int] = set()
        true_positive = 0

        for entity in predicted:
            gold_idx = self._find_partial_match(entity, gold, used_gold)
            if gold_idx is None:
                continue
            used_gold.add(gold_idx)
            true_positive += 1

        false_positive = len(predicted) - true_positive
        false_negative = len(gold) - true_positive

        metrics = _precision_recall_f1(true_positive, false_positive, false_negative)
        metrics.update(
            {
                "matched": float(true_positive),
                "total_predicted": float(len(predicted)),
                "total_gold": float(len(gold)),
            }
        )
        return metrics

    @staticmethod
    def _find_partial_match(entity: _Entity, gold: Sequence[_Entity], used_gold: set[int]) -> int | None:
        """Locate the first overlapping gold entity with a matching CUI."""

        for idx, candidate in enumerate(gold):
            if idx in used_gold or candidate.cui != entity.cui:
                continue
            if _spans_overlap(entity.span, candidate.span):
                return idx
        return None

    @staticmethod
    def _calculate_type_accuracy(predicted: Iterable[_Entity], gold: Iterable[_Entity]) -> dict[str, float]:
        gold_by_cui: dict[str, List[_Entity]] = {}
        for entity in gold:
            gold_by_cui.setdefault(entity.cui, []).append(entity)

        matched = 0
        correct = 0

        for entity in predicted:
            candidates = gold_by_cui.get(entity.cui, [])
            for candidate in candidates:
                if not _spans_overlap(entity.span, candidate.span):
                    continue
                matched += 1
                if not candidate.type_ids or candidate.type_ids.issubset(entity.type_ids):
                    correct += 1
                break

        accuracy = correct / matched if matched else 0.0
        return {
            "accuracy": accuracy,
            "matched": float(matched),
            "correct": float(correct),
        }


def _spans_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]
