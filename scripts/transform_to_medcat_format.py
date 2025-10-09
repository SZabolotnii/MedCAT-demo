"""Convert ``data/internal_short.csv`` into a MedCAT-friendly dictionary CSV.

This utility expands each keyword into a primary entry (preferred name) and
additional synonym rows based on the ``keyword_hints`` column. Hints marked with
``[combined_hint]`` are treated as multi-part phrases; the simplest handling is
to join the parts into a single synonym while also capturing metadata that the
downstream pipeline can use for gap-tolerant matching (default: up to three
intervening words).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence

DEFAULT_INPUT = Path("data/internal_short.csv")
DEFAULT_OUTPUT = Path("data/internal_medcat_v2.csv")
DEFAULT_COMBINED_HINTS = Path("data/internal_combined_hints.json")
DEFAULT_ONTOLOGY = "CUSTOM_INTERNAL"
DEFAULT_MAX_GAP = 3

OUTPUT_COLUMNS = [
    "cui",
    "name",
    "ontologies",
    "name_status",
    "type_ids",
    "source",
    "description",
    "metadata_json",
]


@dataclass(frozen=True)
class CombinedHintRecord:
    """Serializable representation of a combined hint mapping."""

    cui: str
    name: str
    components: List[str]
    max_gap: int
    source_hint: str


@dataclass(frozen=True)
class ParsedHint:
    """Container describing a processed hint string."""

    cleaned: str
    is_combined: bool
    components: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the `internal_short.csv` file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write the MedCAT-ready CSV.",
    )
    parser.add_argument(
        "--combined-hints-output",
        type=Path,
        default=DEFAULT_COMBINED_HINTS,
        help=(
            "Optional JSON file to store combined hint metadata. "
            "Pass an empty string to skip writing."
        ),
    )
    parser.add_argument(
        "--ontology",
        default=DEFAULT_ONTOLOGY,
        help="Ontology identifier to populate the `ontologies` column.",
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=DEFAULT_MAX_GAP,
        help=(
            "Maximum number of words allowed between combined hint components. "
            "Captured in metadata for downstream use."
        ),
    )
    return parser.parse_args()


def read_internal_short(path: Path) -> List[dict[str, str]]:
    with path.open("r", encoding="utf-8") as src:
        reader = csv.DictReader(src)
        return [dict(row) for row in reader]


def parse_hint(raw_hint: str) -> ParsedHint:
    """Normalise a raw hint string and detect combined markers."""
    stripped = raw_hint.strip()
    if not stripped:
        return ParsedHint(cleaned="", is_combined=False, components=[])

    parts = [part.strip() for part in re.split(r"\s*\[combined_hint\]\s*", stripped) if part.strip()]
    is_combined = len(parts) > 1
    cleaned = re.sub(r"\s*\[combined_hint\]\s*", " ", stripped).strip()
    return ParsedHint(cleaned=cleaned, is_combined=is_combined, components=parts or [cleaned])


def collect_combined_hint_records(
    rows: Iterable[dict[str, str]],
    *,
    max_gap: int = DEFAULT_MAX_GAP,
) -> List[CombinedHintRecord]:
    """Collect combined hint definitions for downstream processing."""

    records: List[CombinedHintRecord] = []
    seen: set[tuple[str, tuple[str, ...], str]] = set()

    for row in rows:
        cui = row.get("uid", "").strip()
        hints_field = row.get("keyword_hints", "") or ""

        hints = [hint.strip() for hint in hints_field.split("|") if hint.strip()]
        for raw_hint in hints:
            parsed = parse_hint(raw_hint)
            if not parsed.is_combined or not parsed.components:
                continue

            key = (cui, tuple(parsed.components), parsed.cleaned)
            if key in seen:
                continue
            seen.add(key)

            records.append(
                CombinedHintRecord(
                    cui=cui,
                    name=parsed.cleaned,
                    components=parsed.components,
                    max_gap=max_gap,
                    source_hint=raw_hint,
                )
            )

    return records


def expand_keywords(
    rows: Iterable[dict[str, str]],
    *,
    ontology: str = DEFAULT_ONTOLOGY,
    max_gap: int = DEFAULT_MAX_GAP,
) -> Iterator[dict[str, str]]:
    """Yield MedCAT dictionary rows from the compact CSV entries."""
    for row in rows:
        cui = row.get("uid", "").strip()
        keyword = row.get("keyword", "").strip()
        cluster_id = row.get("cluster", "").strip()
        cluster_title = row.get("cluster_title", "").strip()
        source = row.get("source", "").strip()
        hints_field = row.get("keyword_hints", "") or ""

        base_description = f"{cluster_title} | Primary concept name" if cluster_title else "Primary concept name"

        primary_entry = {
            "cui": cui,
            "name": keyword,
            "ontologies": ontology,
            "name_status": "P",
            "type_ids": cluster_id,
            "source": source,
            "description": base_description,
            "metadata_json": "",
        }
        yield primary_entry

        seen_names = {keyword.casefold()} if keyword else set()

        hints = [hint.strip() for hint in hints_field.split("|") if hint.strip()]
        for raw_hint in hints:
            parsed = parse_hint(raw_hint)
            if not parsed.cleaned:
                continue

            canonical = parsed.cleaned.casefold()
            if canonical in seen_names:
                continue
            seen_names.add(canonical)

            metadata: dict[str, object] = {
                "source_hint": raw_hint,
            }
            if parsed.is_combined:
                metadata["combined_hint"] = {
                    "components": parsed.components,
                    "max_gap": max_gap,
                }

            synonym_entry = {
                "cui": cui,
                "name": parsed.cleaned,
                "ontologies": ontology,
                "name_status": "A",
                "type_ids": cluster_id,
                "source": source,
                "description": (
                    f"{cluster_title} | Hint synonym" if cluster_title else "Hint synonym"
                ),
                "metadata_json": json.dumps(metadata, ensure_ascii=False),
            }
            yield synonym_entry


def write_medcat_csv(rows: Sequence[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(
            {column: entry.get(column, "") for column in OUTPUT_COLUMNS}
            for entry in rows
        )


def main() -> None:
    args = parse_args()
    input_rows = read_internal_short(args.input)
    expanded = list(expand_keywords(input_rows, ontology=args.ontology, max_gap=args.max_gap))
    write_medcat_csv(expanded, args.output)

    combined_output = args.combined_hints_output
    if combined_output:
        records = collect_combined_hint_records(input_rows, max_gap=args.max_gap)
        combined_output = Path(combined_output)
        combined_output.parent.mkdir(parents=True, exist_ok=True)
        with combined_output.open("w", encoding="utf-8") as dst:
            json.dump([record.__dict__ for record in records], dst, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
