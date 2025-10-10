"""Candidate restoration module for fallback entity detection."""

from typing import Any, Dict, List, Set

from medcat.cdb import CDB
from medcat.components.types import CoreComponentType

try:
    from .value_resolver import KeywordRule, ValueMatch, ValueResolver
except ImportError:  # pragma: no cover - allows running as a script
    from value_resolver import KeywordRule, ValueMatch, ValueResolver  # type: ignore


class CandidateRestoration:
    """Handles fallback candidate restoration and validation."""

    def __init__(self, cat, cdb: CDB, value_resolver: ValueResolver):
        self.cat = cat
        self.cdb = cdb
        self.value_resolver = value_resolver

    def restore_missing_candidates(
        self,
        text: str,
        entities: Dict[Any, Dict[str, Any]],
        missing_value_cuis: Set[str],
        initial_entity_count: int,
    ) -> None:
        """Restore missing candidates using fallback logic."""
        existing_cuis = {str(ent.get("cui")).upper() for ent in entities.values()}
        fallback_candidates = self._collect_candidate_entities(text)
        keyword_rules = self.value_resolver.get_keyword_rules()

        candidate_cuis_to_attempt: Set[str] = set(missing_value_cuis)

        if initial_entity_count == 0:
            for cui, candidates in fallback_candidates.items():
                if cui in existing_cuis or cui in candidate_cuis_to_attempt:
                    continue

                rule = keyword_rules.get(cui)
                if not rule:
                    continue

                if rule.required_components or rule.requires_value or rule.is_numeric:
                    candidate_cuis_to_attempt.add(cui)
                elif self.value_resolver.should_enforce_surface(rule):
                    sample_entity = self._candidate_to_entity(candidates[0], cui)
                    if self.value_resolver.surface_matches_keyword(rule, sample_entity):
                        candidate_cuis_to_attempt.add(cui)
                elif rule.value_patterns:
                    candidate_cuis_to_attempt.add(cui)

        added_cuis: Set[str] = set()
        for cui in candidate_cuis_to_attempt:
            if cui in existing_cuis or cui in added_cuis:
                continue

            rule = keyword_rules.get(cui)
            if not rule:
                continue

            candidates = fallback_candidates.get(cui, [])
            for candidate in candidates:
                entity_dict = self._candidate_to_entity(candidate, cui)

                if self.value_resolver.should_enforce_surface(rule) and not self.value_resolver.surface_matches_keyword(rule, entity_dict):
                    continue

                if not self.value_resolver.components_present(rule, text, entity_dict):
                    continue

                if rule.requires_value:
                    match = self.value_resolver.find_value_match(rule, text, entity_dict)
                    if match is None:
                        continue
                    if rule.is_numeric and (match.numeric is None or not rule.is_value_in_range(match.numeric)):
                        continue
                else:
                    match = self.value_resolver.find_value_match(rule, text, entity_dict)

                new_key = self._next_entity_key(entities)
                entity_dict["id"] = new_key
                if match is not None:
                    self._record_value_hint(entity_dict, match, rule)
                entities[new_key] = entity_dict
                existing_cuis.add(cui)
                added_cuis.add(cui)
                break

        self._deduplicate_overlaps(entities)

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

    @staticmethod
    def _record_value_hint(entity: Dict[str, Any], match: ValueMatch, rule: KeywordRule) -> None:
        """Record value hint in entity metadata."""
        ValueResolver._record_value_hint(entity, match, rule)

    @staticmethod
    def _deduplicate_overlaps(entities: Dict[Any, Dict[str, Any]]) -> None:
        """Remove overlapping entities with identical CUIs keeping the longest span."""
        by_cui: Dict[str, List[tuple[Any, int, int]]] = {}
        for key, entity in entities.items():
            cui = str(entity.get("cui") or "").upper()
            start = entity.get("start")
            end = entity.get("end")
            if not cui or not isinstance(start, int) or not isinstance(end, int):
                continue
            by_cui.setdefault(cui, []).append((key, start, end))

        to_remove: Set[Any] = set()
        for items in by_cui.values():
            items.sort(key=lambda item: (item[1], -(item[2] - item[1])))
            selected: List[tuple[Any, int, int]] = []
            for key, start, end in items:
                span = (start, end)
                length = end - start
                if length <= 0:
                    to_remove.add(key)
                    continue
                keep = True
                for _, sel_start, sel_end in selected:
                    if not (end <= sel_start or start >= sel_end):
                        keep = False
                        break
                if keep:
                    selected.append((key, start, end))
                else:
                    to_remove.add(key)

        for key in to_remove:
            entities.pop(key, None)
