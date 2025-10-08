"""Convert the custom ``data/internal.json`` structure into a flat CSV file.

The JSON file is organised as a list of sections, each containing a ``source``
identifier and a ``keywords`` list. Every keyword describes a medical concept,
its cluster id, and optional ``data`` entries. ``data`` may be a simple list of
values (strings/numbers) or a list of objects with their own ``value`` and
``hints`` fields.

This script normalises the structure into a tabular CSV where each row
represents a single keyword/value combination. Use it like so:

```
python -m scripts.convert_internal_json_to_csv \
    --input data/internal.json \
    --output data/internal.csv \
    --short-output data/internal_short.csv  # optional compact CSV
```
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence, Set

DEFAULT_INPUT = Path("data/internal.json")
DEFAULT_OUTPUT = Path("data/internal.csv")
DEFAULT_CLUSTERS = Path("data/valid_clusters.json")
DEFAULT_SHORT_OUTPUT = Path("data/internal_short.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the source JSON file (default: data/internal.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write the resulting CSV (default: data/internal.csv).",
    )
    parser.add_argument(
        "--clusters",
        type=Path,
        default=DEFAULT_CLUSTERS,
        help=(
            "Optional path to clusters JSON used to enrich rows with titles "
            "(default: data/valid_clusters.json)."
        ),
    )
    parser.add_argument(
        "--short-output",
        type=Path,
        default=None,
        help=(
            "Optional path to write a compact CSV without data_value/data_hints. "
            f"If omitted, short CSV is not generated. (Suggested default: {DEFAULT_SHORT_OUTPUT})"
        ),
    )
    return parser.parse_args()


def load_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as src:
        return json.load(src)


def load_cluster_titles(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as src:
        data = json.load(src)

    clusters_by_id: Dict[str, Dict[str, Any]] = {}
    parent_by_child: Dict[str, str] = {}

    for item in data:
        cluster_id = str(item.get("id", ""))
        if not cluster_id:
            continue
        clusters_by_id[cluster_id] = item
        for child in item.get("childrenClusterIds") or []:
            child_id = str(child)
            if child_id:
                parent_by_child[child_id] = cluster_id

    def build_full_title(cluster_id: str) -> str:
        parts: List[str] = []
        current = cluster_id
        visited: Set[str] = set()

        while current and current not in visited:
            visited.add(current)
            cluster = clusters_by_id.get(current)
            if not cluster:
                break

            title = cluster.get("title", "")
            if title:
                parts.append(str(title))

            current = parent_by_child.get(current)

        parts.reverse()
        return "/".join(parts)

    titles: Dict[str, str] = {}
    for cluster_id in clusters_by_id:
        full_title = build_full_title(cluster_id)
        if not full_title:
            full_title = clusters_by_id[cluster_id].get("title", "")
        titles[cluster_id] = str(full_title)

    return titles


def normalise_rows(
    records: Iterable[Dict[str, Any]],
    cluster_titles: Mapping[str, str],
) -> Iterator[Dict[str, str]]:
    """Yield flattened rows ready for CSV output."""
    for section in records:
        source = section.get("source", "")
        keywords = section.get("keywords") or []
        for keyword in keywords:
            base = {
                "source": source,
                "keyword": keyword.get("keyword", ""),
                "uid": keyword.get("uid", ""),
                "cluster": keyword.get("cluster", ""),
                "keyword_hints": " | ".join(keyword.get("hints", []) or []),
            }
            base["cluster_title"] = cluster_titles.get(base["cluster"], "")
            data_entries = keyword.get("data")

            if data_entries in (None, []):
                yield {**base, "data_value": "", "data_hints": ""}
                continue

            if isinstance(data_entries, list):
                for entry in data_entries:
                    if isinstance(entry, dict):
                        value = entry.get("value", "")
                        data_hints = " | ".join(entry.get("hints", []) or [])
                        yield {
                            **base,
                            "data_value": str(value),
                            "data_hints": data_hints,
                        }
                    else:
                        yield {
                            **base,
                            "data_value": str(entry),
                            "data_hints": "",
                        }
            else:
                # Safety net: treat any other structure as a single scalar string.
                yield {
                    **base,
                    "data_value": str(data_entries),
                    "data_hints": "",
                }


def write_csv(rows: Sequence[Dict[str, str]], path: Path) -> None:
    fieldnames = [
        "source",
        "keyword",
        "uid",
        "cluster",
        "cluster_title",
        "keyword_hints",
        "data_value",
        "data_hints",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_short_csv(rows: Sequence[Dict[str, str]], path: Path) -> None:
    fieldnames = [
        "source",
        "keyword",
        "uid",
        "cluster",
        "cluster_title",
        "keyword_hints",
    ]

    # Deduplicate rows that would become identical after dropping value columns.
    seen: Set[tuple[str, ...]] = set()
    filtered_rows: List[Dict[str, str]] = []

    for row in rows:
        key = tuple(row.get(field, "") for field in fieldnames)
        if key in seen:
            continue
        seen.add(key)
        filtered_rows.append({field: row.get(field, "") for field in fieldnames})

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)


def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    cluster_titles = load_cluster_titles(args.clusters)
    rows = list(normalise_rows(records, cluster_titles))
    write_csv(rows, args.output)

    if args.short_output:
        write_short_csv(rows, args.short_output)


if __name__ == "__main__":
    main()
