"""Custom MedCAT wrapper with gap-tolerant combined hint matching."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from medcat.cdb import CDB
from medcat.cat import CAT
from medcat.config import Config
from medcat.vocab import Vocab
from medcat.components.types import CoreComponentType
try:
    from .utils import load_model_pack
except ImportError:  # pragma: no cover - allows running as a script
    from utils import load_model_pack  # type: ignore


@dataclass(frozen=True)
class CombinedHintDefinition:
    cui: str
    name: str
    components: List[str]
    max_gap: int
    source_hint: str


@dataclass(frozen=True)
class ValueMatch:
    """Represents a resolved value match for a keyword."""

    numeric: Optional[float] = None
    text: Optional[str] = None


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
        if not self.numeric_ranges:
            return True
        return any(lower <= value <= upper for lower, upper in self.numeric_ranges)


class CombinedHintMatcher:
    """Simple regex-based matcher for combined hints."""

    def __init__(self, definitions: List[CombinedHintDefinition]):
        self.definitions = definitions
        self._compiled = [self._compile_definition(defn) for defn in definitions]

    @staticmethod
    def _compile_definition(defn: CombinedHintDefinition) -> re.Pattern[str]:
        pattern_parts: List[str] = []
        for idx, component in enumerate(defn.components):
            component_regex = r"\b" + re.escape(component) + r"\b"
            pattern_parts.append(component_regex)
            if idx < len(defn.components) - 1:
                gap_pattern = rf"(?:\W+\w+){{0,{defn.max_gap}}}\W+"
                pattern_parts.append(gap_pattern)

        regex = "".join(pattern_parts)
        return re.compile(regex, flags=re.IGNORECASE | re.MULTILINE)

    def find_matches(self, text: str) -> List[Dict[str, Any]]:
        matches: List[Dict[str, Any]] = []
        for definition, pattern in zip(self.definitions, self._compiled):
            for match in pattern.finditer(text):
                matches.append(
                    {
                        "cui": definition.cui,
                        "name": definition.name,
                        "source_hint": definition.source_hint,
                        "start": match.start(),
                        "end": match.end(),
                        "matched_text": match.group(0),
                    }
                )
        return matches


def _load_combined_hints(path: Optional[Path]) -> List[CombinedHintDefinition]:
    if not path or not path.exists():
        return []

    with path.open("r", encoding="utf-8") as src:
        raw = json.load(src)

    definitions: List[CombinedHintDefinition] = []
    for item in raw:
        try:
            definitions.append(
                CombinedHintDefinition(
                    cui=str(item["cui"]),
                    name=str(item["name"]),
                    components=[str(part) for part in item.get("components", [])],
                    max_gap=int(item.get("max_gap", 0)),
                    source_hint=str(item.get("source_hint", "")),
                )
            )
        except (KeyError, TypeError, ValueError):  # pragma: no cover - defensive
            continue
    return definitions


class CustomCAT:
    """Composite wrapper around MedCAT CAT adding combined hint support."""

    _VALUE_WINDOW_CHARS = 80
    _NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")

    def __init__(
        self,
        model_pack_path: str | Path,
        *,
        combined_hints_path: str | Path | None = None,
    ) -> None:
        self.model_path = Path(model_pack_path)

        if self.model_path.is_dir() and (self.model_path / "custom_cdb_v2").exists():
            cdb_dir = self.model_path / "custom_cdb_v2"
            self.cdb = CDB.load(str(cdb_dir))

            config_path = self.model_path / "config.json"
            if config_path.exists():
                config_data = json.loads(config_path.read_text(encoding="utf-8"))
                config = Config.model_validate(config_data)
            else:
                config = Config()
            self.cat = CAT(cdb=self.cdb, config=config, vocab=None)
        else:
            self.cat = load_model_pack(self.model_path)
            self.cdb = self.cat.cdb

        if getattr(self.cat, "vocab", None) is None:
            self.cat.vocab = Vocab()
            pipeline = getattr(self.cat, "_pipeline", None)
            if pipeline is not None:
                for component in getattr(pipeline, "_components", []):
                    context_model = getattr(component, "context_model", None)
                    if context_model is not None:
                        context_model.vocab = self.cat.vocab

        combined_path = Path(combined_hints_path).expanduser() if combined_hints_path else None
        if combined_path and combined_path.is_dir():
            combined_path = combined_path / "internal_combined_hints.json"
        self.combined_matcher = CombinedHintMatcher(_load_combined_hints(combined_path))

        project_root = Path(__file__).resolve().parents[1]
        self._keyword_rules = self._load_keyword_rules(project_root)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying CAT instance."""

        return getattr(self.cat, name)

    def extract_entities(self, text: str, *, min_confidence: float = 0.0) -> Dict[str, Any]:
        """Run MedCAT extraction and augment with combined hint matches."""

        result = self.cat.get_entities(text, only_cui=False)
        entities: Dict[str, Any] = result.setdefault("entities", {})

        matches = self.combined_matcher.find_matches(text)
        if matches:
            result.setdefault("combined_hint_matches", matches)
            offset = 0
            while f"combined_{offset}" in entities:
                offset += 1
            for idx, match in enumerate(matches, start=offset):
                key = f"combined_{idx}"
                if key in entities:
                    continue
                entities[key] = {
                    "cui": match["cui"],
                    "detected_name": match["name"],
                    "source_value": match["matched_text"],
                    "acc": 1.0,
                    "start": match["start"],
                    "end": match["end"],
                    "pretty_name": match["name"],
                    "types": [],
                }

        if min_confidence > 0:
            filtered_entities = {
                key: ent
                for key, ent in entities.items()
                if float(ent.get("acc", 0.0)) >= min_confidence
            }
            result["entities"] = filtered_entities
            entities = filtered_entities

        self._apply_value_rules(text, entities)
        result["entities"] = entities

        return result

    def get_entities(self, text: str, *, only_cui: bool = False) -> Dict[str, Any]:
        """Maintain CAT API compatibility while enforcing value rules."""

        result = self.cat.get_entities(text, only_cui=only_cui)
        if not only_cui:
            entities = result.setdefault("entities", {})
            self._apply_value_rules(text, entities)
            result["entities"] = entities
        return result

    def batch_process(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Process a list of texts sequentially."""

        return [self.extract_entities(text) for text in texts]

    def _apply_value_rules(self, text: str, entities: Dict[Any, Dict[str, Any]]) -> None:
        """Apply rule-based validation requiring value hints or numeric ranges."""

        if not self._keyword_rules:
            return

        initial_entity_count = len(entities)
        keys_to_remove: List[Any] = []
        missing_value_cuis: set[str] = set()
        fallback_candidates: Optional[Dict[str, List[Any]]] = None
        added_cuis: set[str] = set()

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

        for key in keys_to_remove:
            entities.pop(key, None)

        existing_cuis = {str(ent.get("cui")).upper() for ent in entities.values()}

        if fallback_candidates is None:
            fallback_candidates = self._collect_candidate_entities(text)

        candidate_cuis_to_attempt: set[str] = set(missing_value_cuis)

        for cui, candidates in fallback_candidates.items():
            if cui in existing_cuis or cui in candidate_cuis_to_attempt:
                continue

            rule = self._keyword_rules.get(cui)
            if not rule:
                continue

            if rule.required_components or rule.requires_value or rule.is_numeric:
                candidate_cuis_to_attempt.add(cui)
            elif self._should_enforce_surface(rule):
                sample_entity = self._candidate_to_entity(candidates[0], cui)
                if self._surface_matches_keyword(rule, sample_entity):
                    candidate_cuis_to_attempt.add(cui)
            elif initial_entity_count == 0 and rule.value_patterns:
                candidate_cuis_to_attempt.add(cui)

        added_cuis: set[str] = set()
        for cui in candidate_cuis_to_attempt:
            if cui in existing_cuis or cui in added_cuis:
                continue

            rule = self._keyword_rules.get(cui)
            if not rule:
                continue

            candidates = fallback_candidates.get(cui, [])
            for candidate in candidates:
                entity_dict = self._candidate_to_entity(candidate, cui)

                if self._should_enforce_surface(rule) and not self._surface_matches_keyword(rule, entity_dict):
                    continue

                if not self._components_present(rule, text, entity_dict):
                    continue

                if rule.requires_value:
                    match = self._find_value_match(rule, text, entity_dict)
                    if match is None:
                        continue
                    if rule.is_numeric and (match.numeric is None or not rule.is_value_in_range(match.numeric)):
                        continue

                new_key = self._next_entity_key(entities)
                entity_dict["id"] = new_key
                entities[new_key] = entity_dict
                existing_cuis.add(cui)
                added_cuis.add(cui)
                break

    def _components_present(self, rule: KeywordRule, text: str, entity: Dict[str, Any]) -> bool:
        if not rule.required_components:
            return True

        start = int(entity.get("start", 0))
        end = int(entity.get("end", start))
        window_start = max(0, start - self._VALUE_WINDOW_CHARS)
        window_end = min(len(text), end + self._VALUE_WINDOW_CHARS)
        window = text[window_start:window_end].lower()

        return all(component in window for component in rule.required_components)

    def _find_value_match(self, rule: KeywordRule, text: str, entity: Dict[str, Any]) -> Optional[ValueMatch]:
        """Locate textual or numeric value hints near an entity span."""

        start = int(entity.get("start", 0))
        end = int(entity.get("end", start))
        window_start = max(0, start - self._VALUE_WINDOW_CHARS)
        window_end = min(len(text), end + self._VALUE_WINDOW_CHARS)
        window = text[window_start:window_end]

        if rule.value_patterns:
            for pattern in rule.value_patterns:
                if pattern.search(window):
                    return ValueMatch(text=pattern.pattern)

        if rule.is_numeric:
            for match in self._NUMBER_PATTERN.finditer(window):
                try:
                    numeric_value = float(match.group())
                except ValueError:
                    continue
                return ValueMatch(numeric=numeric_value)

        return None

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_keyword_rules(project_root: Path) -> Dict[str, KeywordRule]:
        data_dir = project_root / "data"
        internal_csv = data_dir / "internal.csv"
        if not internal_csv.exists():
            return {}

        numeric_model_path = data_dir / "numerical_model.json"
        numeric_by_keyword, numeric_by_cluster = CustomCAT._load_numeric_ranges(numeric_model_path)

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

                for value in CustomCAT._split_values(row.get("data_value", "")):
                    builder["raw_values"].add(value)
                for hint in CustomCAT._split_values(row.get("data_hints", "")):
                    builder["raw_values"].add(hint)

        rules: Dict[str, KeywordRule] = {}
        for cui, data in builders.items():
            sources = tuple(sorted(source for source in data["sources"] if source))
            cluster_title = data.get("cluster_title", "")
            keyword = data.get("keyword", "")
            is_numeric = any(source == "numerical" for source in sources)

            requires_value = is_numeric or ("string" in cluster_title.lower())

            numeric_ranges: Tuple[Tuple[float, float], ...] = ()
            if is_numeric:
                numeric_ranges = tuple(
                    numeric_by_keyword.get(keyword.lower(), [])
                ) or tuple(numeric_by_cluster.get(cluster_title.lower(), []))

            value_patterns: List[re.Pattern[str]] = []
            for raw_value in data["raw_values"]:
                pattern = CustomCAT._compile_value_pattern(raw_value)
                if pattern is not None:
                    value_patterns.append(pattern)

            components = CustomCAT._derive_required_components(keyword, data["raw_values"])

            rules[cui] = KeywordRule(
                cui=cui,
                keyword=keyword,
                cluster_id=data.get("cluster_id", ""),
                cluster_title=cluster_title,
                sources=sources,
                requires_value=requires_value,
                is_numeric=is_numeric,
                numeric_ranges=tuple(numeric_ranges),
                value_patterns=tuple(value_patterns),
                required_components=components,
                normalized_keyword=CustomCAT._normalize_keyword(keyword),
            )

        return rules

    @staticmethod
    def _compile_value_pattern(value: str) -> Optional[re.Pattern[str]]:
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
        if not raw:
            return ()
        return (part.strip() for part in raw.split("|") if part.strip())

    @staticmethod
    def _load_numeric_ranges(
        path: Path,
    ) -> Tuple[Dict[str, List[Tuple[float, float]]], Dict[str, List[Tuple[float, float]]]]:
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
    def _derive_required_components(keyword: str, raw_values: Iterable[str]) -> Tuple[str, ...]:
        base = re.sub(r"\[.*?\]", "", keyword).lower()
        parts = [part.strip() for part in base.split("/") if part.strip()]
        if len(parts) > 1:
            return tuple(parts)

        for value in raw_values:
            lowered = value.lower()
            if "/" in lowered:
                value_parts = [part.strip() for part in lowered.split("/") if part.strip()]
                if len(value_parts) > 1:
                    return tuple(value_parts)

        return ()

    @staticmethod
    def _normalize_keyword(text: str) -> str:
        cleaned = (text or "")
        cleaned = re.sub(r"\[(.*?)\]", r" \1", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned.strip().lower())
        return cleaned

    def _surface_matches_keyword(self, rule: KeywordRule, entity: Dict[str, Any]) -> bool:
        surface = (entity.get("source_value") or "").strip()
        if not surface:
            surface = (entity.get("detected_name") or "").replace("~", " ")
        normalized_surface = self._normalize_keyword(surface)
        return normalized_surface == rule.normalized_keyword

    @staticmethod
    def _should_enforce_surface(rule: KeywordRule) -> bool:
        return (
            not rule.required_components
            and not rule.requires_value
            and not rule.is_numeric
            and not rule.value_patterns
        )

    def _collect_candidate_entities(self, text: str) -> Dict[str, List[Any]]:
        """Run the pipeline up to the linker to collect NER candidates."""

        pipeline = self.cat._pipeline
        doc = pipeline.get_doc(text)

        for component in pipeline.iter_all_components():
            comp_type = getattr(component, "get_type", None)
            comp_kind = comp_type() if callable(comp_type) else None
            if comp_kind == CoreComponentType.linking:
                break
            doc = component(doc)

        candidates: Dict[str, List[Any]] = {}
        for entity in getattr(doc, "ner_ents", []):
            for candidate_cui in getattr(entity, "link_candidates", []) or []:
                if not candidate_cui:
                    continue
                candidates.setdefault(str(candidate_cui).upper(), []).append(entity)
        return candidates

    def _candidate_to_entity(self, candidate: Any, cui: str) -> Dict[str, Any]:
        """Convert a MedCAT candidate entity to the output dictionary format."""

        start_char = int(getattr(candidate, "start_char_index", getattr(candidate, "_start_char_index", 0)))
        end_char = int(getattr(candidate, "end_char_index", getattr(candidate, "_end_char_index", start_char)))
        detected_name = getattr(candidate, "detected_name", "")
        text_value = getattr(candidate, "text", "") or detected_name.replace("~", " ")

        cui_info = self.cdb.cui2info.get(cui, {})
        type_ids = sorted({str(tid).upper() for tid in cui_info.get("type_ids", []) if tid})
        pretty_name = cui_info.get("preferred_name") or text_value

        return {
            "cui": cui,
            "start": start_char,
            "end": end_char,
            "detected_name": detected_name or text_value.replace(" ", "~"),
            "source_value": text_value,
            "pretty_name": pretty_name,
            "acc": 1.0,
            "context_similarity": 1.0,
            "type_ids": type_ids,
            "meta_anns": {},
            "context_left": [],
            "context_center": [],
            "context_right": [],
        }

    @staticmethod
    def _next_entity_key(entities: Dict[Any, Dict[str, Any]]) -> int:
        """Generate the next numeric entity key."""

        max_key = -1
        for key in entities.keys():
            if isinstance(key, int):
                max_key = max(max_key, key)
            elif isinstance(key, str) and key.isdigit():
                max_key = max(max_key, int(key))

        return max_key + 1
