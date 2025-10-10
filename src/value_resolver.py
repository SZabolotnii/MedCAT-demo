"""Value-aware validation and matching module."""

import csv
import json
import re
import yaml
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ValueMatch:
    """Represents a resolved value match for a keyword."""
    numeric: Optional[float] = None
    text: Optional[str] = None
    matched_text: Optional[str] = None
    start: Optional[int] = None
    end: Optional[int] = None
    pattern: Optional[str] = None


@dataclass(frozen=True)
class KeywordRule:
    """Metadata describing value requirements for a given keyword/CUI."""
    cui: str
    keyword: str
    cluster_id: str
    cluster_title: str
    sources: Tuple[str, ...]
    requires_value: bool
    is_numeric: bool
    numeric_ranges: Tuple[Tuple[float, float], ...] = ()
    value_patterns: Tuple[re.Pattern[str], ...] = ()
    required_components: Tuple[str, ...] = ()
    normalized_keyword: str = ""

    def is_value_in_range(self, value: float) -> bool:
        """Check if numeric value is within allowed ranges."""
        if not self.numeric_ranges:
            return True
        return any(lower <= value <= upper for lower, upper in self.numeric_ranges)


class ValueResolver:
    """Handles value-aware validation and matching logic."""

    _VALUE_WINDOW_CHARS = 80
    _NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")
    _KEYWORDS_WITHOUT_HINT: set[str] = set()
    _CLUSTERS_WITHOUT_HINT: set[str] = set()
    _HC_OVERRIDES_LOADED = False

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self._ensure_hint_overrides(project_root)
        self._keyword_rules = self._load_keyword_rules(project_root)

    def apply_value_rules(self, text: str, entities: Dict[Any, Dict[str, Any]]) -> None:
        """Apply rule-based validation requiring value hints or numeric ranges."""
        if not self._keyword_rules:
            return

        initial_entity_count = len(entities)
        keys_to_remove: List[Any] = []
        missing_value_cuis: set[str] = set()

        for key, entity in list(entities.items()):
            if isinstance(key, str) and key.startswith("combined_"):
                # Combined hints already encode explicit value matches.
                continue

            cui = entity.get("cui")
            if cui is None:
                continue
            rule = self._keyword_rules.get(str(cui).upper())
            if not rule or not rule.requires_value:
                continue

            if not self._components_present(rule, text, entity):
                keys_to_remove.append(key)
                missing_value_cuis.add(str(cui).upper())
                continue

            match = self._find_value_match(rule, text, entity)
            if match is None:
                keys_to_remove.append(key)
                missing_value_cuis.add(str(cui).upper())
                continue

            if rule.is_numeric:
                if match.numeric is None or not rule.is_value_in_range(match.numeric):
                    keys_to_remove.append(key)
                    missing_value_cuis.add(str(cui).upper())
                    continue

            self._record_value_hint(entity, match, rule)

        for key in keys_to_remove:
            entities.pop(key, None)

    def components_present(self, rule: KeywordRule, text: str, entity: Dict[str, Any]) -> bool:
        """Check if required components are present in the text window."""
        if not rule.required_components:
            return True

        start = int(entity.get("start", 0))
        end = int(entity.get("end", start))
        window_start = max(0, start - self._VALUE_WINDOW_CHARS)
        window_end = min(len(text), end + self._VALUE_WINDOW_CHARS)
        window = text[window_start:window_end].lower()

        return all(component in window for component in rule.required_components)

    def find_value_match(self, rule: KeywordRule, text: str, entity: Dict[str, Any]) -> Optional[ValueMatch]:
        """Locate textual or numeric value hints near an entity span."""
        entity_start = int(entity.get("start", 0))
        entity_end = int(entity.get("end", entity_start))
        window_start = max(0, entity_start - self._VALUE_WINDOW_CHARS)
        window_end = min(len(text), entity_end + self._VALUE_WINDOW_CHARS)
        window = text[window_start:window_end]

        if rule.value_patterns:
            for pattern in rule.value_patterns:
                match_obj = pattern.search(window)
                if match_obj:
                    match_start = window_start + match_obj.start()
                    match_end = window_start + match_obj.end()
                    return ValueMatch(
                        text=match_obj.group(0),
                        matched_text=match_obj.group(0),
                        start=match_start,
                        end=match_end,
                        pattern=pattern.pattern,
                    )

        if rule.is_numeric:
            matches: List[ValueMatch] = []
            for match in self._NUMBER_PATTERN.finditer(window):
                try:
                    numeric_value = float(match.group())
                except ValueError:
                    continue
                match_start = window_start + match.start()
                match_end = window_start + match.end()
                matches.append(
                    ValueMatch(
                        numeric=numeric_value,
                        matched_text=match.group(0),
                        start=match_start,
                        end=match_end,
                        pattern=None,
                    )
                )

            if matches:
                after = [m for m in matches if m.start is not None and m.start >= entity_end]
                if after:
                    after.sort(key=lambda m: (m.start - entity_end, m.start))
                    return after[0]
                before = [m for m in matches if m.end is not None and m.end <= entity_start]
                if before:
                    before.sort(key=lambda m: (entity_start - m.end, m.start))
                    return before[0]
                entity_mid = (entity_start + entity_end) / 2
                matches.sort(
                    key=lambda m: (
                        abs(((m.start or 0) + (m.end or 0)) / 2 - entity_mid),
                        m.start,
                    )
                )
                return matches[0]

        return None

    def get_keyword_rules(self) -> Dict[str, KeywordRule]:
        """Get loaded keyword rules."""
        return self._keyword_rules

    def surface_matches_keyword(self, rule: KeywordRule, entity: Dict[str, Any]) -> bool:
        """Check if entity surface matches the normalized keyword."""
        surface = (entity.get("source_value") or "").strip()
        if not surface:
            surface = (entity.get("detected_name") or "").replace("~", " ")
        normalized_surface = self._normalize_keyword(surface)
        return normalized_surface == rule.normalized_keyword

    def should_enforce_surface(self, rule: KeywordRule) -> bool:
        """Check if surface matching should be enforced for this rule."""
        return (
            not rule.required_components
            and not rule.requires_value
            and not rule.is_numeric
            and not rule.value_patterns
        )

    def _components_present(self, rule: KeywordRule, text: str, entity: Dict[str, Any]) -> bool:
        """Internal method for component presence checking."""
        return self.components_present(rule, text, entity)

    def _find_value_match(self, rule: KeywordRule, text: str, entity: Dict[str, Any]) -> Optional[ValueMatch]:
        """Internal method for value matching."""
        return self.find_value_match(rule, text, entity)

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_keyword_rules(project_root: Path) -> Dict[str, KeywordRule]:
        """Load keyword rules from CSV and configuration files."""
        data_dir = project_root / "data"
        internal_csv = data_dir / "internal.csv"
        if not internal_csv.exists():
            return {}

        numeric_model_path = data_dir / "numerical_model.json"
        numeric_by_keyword, numeric_by_cluster = ValueResolver._load_numeric_ranges(numeric_model_path)

        builders: Dict[str, Dict[str, Any]] = {}

        with internal_csv.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                cui = (row.get("uid") or "").strip().upper()
                if not cui:
                    continue
                builder = builders.setdefault(
                    cui,
                    {
                        "keyword": (row.get("keyword") or "").strip(),
                        "cluster_id": (row.get("cluster") or "").strip(),
                        "cluster_title": (row.get("cluster_title") or "").strip(),
                        "sources": set(),
                        "raw_values": set(),
                    },
                )

                if not builder["keyword"] and row.get("keyword"):
                    builder["keyword"] = row["keyword"].strip()
                if not builder["cluster_id"] and row.get("cluster"):
                    builder["cluster_id"] = row["cluster"].strip()
                if not builder["cluster_title"] and row.get("cluster_title"):
                    builder["cluster_title"] = row["cluster_title"].strip()

                source = (row.get("source") or "").strip()
                if source:
                    builder["sources"].add(source)

                for value in ValueResolver._split_values(row.get("data_value", "")):
                    builder["raw_values"].add(value)
                for hint in ValueResolver._split_values(row.get("data_hints", "")):
                    builder["raw_values"].add(hint)

        rules: Dict[str, KeywordRule] = {}
        for cui, data in builders.items():
            sources = tuple(sorted(source for source in data["sources"] if source))
            cluster_title = data.get("cluster_title", "")
            keyword = data.get("keyword", "")
            is_numeric = any(source == "numerical" for source in sources)

            cluster_id = data.get("cluster_id", "")
            requires_value = True
            if (
                cui in ValueResolver._KEYWORDS_WITHOUT_HINT
                or str(cluster_id).upper() in ValueResolver._CLUSTERS_WITHOUT_HINT
            ):
                requires_value = False
            elif is_numeric:
                requires_value = True

            numeric_ranges: Tuple[Tuple[float, float], ...] = ()
            if is_numeric:
                numeric_ranges = tuple(
                    numeric_by_keyword.get(keyword.lower(), [])
                ) or tuple(numeric_by_cluster.get(cluster_title.lower(), []))

            value_patterns: List[re.Pattern[str]] = []
            for raw_value in data["raw_values"]:
                pattern = ValueResolver._compile_value_pattern(raw_value)
                if pattern is not None:
                    value_patterns.append(pattern)

            components = ValueResolver._derive_required_components(keyword, data["raw_values"])

            rules[cui] = KeywordRule(
                cui=cui,
                keyword=keyword,
                cluster_id=str(cluster_id),
                cluster_title=cluster_title,
                sources=sources,
                requires_value=requires_value,
                is_numeric=is_numeric,
                numeric_ranges=tuple(numeric_ranges),
                value_patterns=tuple(value_patterns),
                required_components=components,
                normalized_keyword=ValueResolver._normalize_keyword(keyword),
            )

        return rules

    @staticmethod
    def _compile_value_pattern(value: str) -> Optional[re.Pattern[str]]:
        """Compile a value pattern from string."""
        value = value.strip()
        if not value:
            return None

        parts = [part.strip() for part in value.split("[combined_hint]")]
        escaped_parts = [re.escape(part) for part in parts if part]
        if not escaped_parts:
            return None

        pattern = r".*?".join(escaped_parts)
        return re.compile(pattern, flags=re.IGNORECASE)

    @staticmethod
    def _split_values(raw: str) -> Iterable[str]:
        """Split comma-separated values."""
        if not raw:
            return ()
        return (part.strip() for part in raw.split("|") if part.strip())

    @staticmethod
    def _load_numeric_ranges(
        path: Path,
    ) -> Tuple[Dict[str, List[Tuple[float, float]]], Dict[str, List[Tuple[float, float]]]]:
        """Load numeric ranges from JSON file."""
        if not path.exists():
            return {}, {}

        data = json.loads(path.read_text(encoding="utf-8"))
        entries = data.get("ranges_kws", [])
        by_keyword: Dict[str, List[Tuple[float, float]]] = {}
        by_cluster: Dict[str, List[Tuple[float, float]]] = {}

        for entry in entries:
            ranges = [
                (float(r[0]), float(r[1]))
                for r in entry.get("ranges", [])
                if isinstance(r, Sequence) and len(r) == 2
            ]
            if not ranges:
                continue
            keyword = (entry.get("keyword") or "").strip().lower()
            cluster = (entry.get("cluster") or "").strip().lower()
            if keyword:
                by_keyword.setdefault(keyword, []).extend(ranges)
            elif cluster:
                by_cluster.setdefault(cluster, []).extend(ranges)

        return by_keyword, by_cluster

    @staticmethod
    def _ensure_hint_overrides(project_root: Path) -> None:
        """Load hint override configurations."""
        if ValueResolver._HC_OVERRIDES_LOADED:
            return

        config_path = project_root / "data" / "hc.yaml"
        if not config_path.exists():
            ValueResolver._HC_OVERRIDES_LOADED = True
            return

        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except Exception:
            ValueResolver._HC_OVERRIDES_LOADED = True
            return

        keyword_entries = config.get("ft_value_without_hint_by_keyword") or []
        cluster_entries = config.get("ft_value_without_hint_by_cluster") or []

        ValueResolver._KEYWORDS_WITHOUT_HINT = {
            str(entry.get("id")).upper()
            for entry in keyword_entries
            if entry and entry.get("id")
        }
        ValueResolver._CLUSTERS_WITHOUT_HINT = {
            str(entry.get("id")).upper()
            for entry in cluster_entries
            if entry and entry.get("id")
        }
        ValueResolver._HC_OVERRIDES_LOADED = True

    @staticmethod
    def _derive_required_components(keyword: str, raw_values: Iterable[str]) -> Tuple[str, ...]:
        """Derive required components from keyword."""
        base = re.sub(r"\[.*?\]", "", keyword).lower()
        parts = [part.strip() for part in base.split("/") if part.strip()]
        if len(parts) > 1:
            return tuple(parts)
        return ()

    @staticmethod
    def _normalize_keyword(text: str) -> str:
        """Normalize keyword text."""
        cleaned = (text or "")
        cleaned = re.sub(r"\[(.*?)\]", r" \1", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned.strip().lower())
        return cleaned

    @staticmethod
    def _record_value_hint(entity: Dict[str, Any], match: ValueMatch, rule: KeywordRule) -> None:
        """Record value hint in entity metadata."""
        if match is None:
            return

        hints = entity.setdefault("value_hints", [])
        hint_payload: Dict[str, Any] = {
            "rule_keyword": rule.keyword,
        }

        if match.numeric is not None:
            hint_payload.update(
                {
                    "type": "numeric",
                    "value": match.numeric,
                    "matched_text": match.matched_text,
                }
            )
        elif match.matched_text:
            hint_payload.update(
                {
                    "type": "text",
                    "value": match.matched_text,
                    "pattern": match.pattern,
                }
            )
        else:
            hint_payload.update(
                {
                    "type": "unknown",
                    "value": match.text,
                    "pattern": match.pattern,
                }
            )

        if match.start is not None and match.end is not None:
            hint_payload["start"] = match.start
            hint_payload["end"] = match.end

        hints.append(hint_payload)
