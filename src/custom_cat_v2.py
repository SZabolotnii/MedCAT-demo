"""Custom MedCAT wrapper with gap-tolerant combined hint matching."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from medcat.cdb import CDB
from medcat.cat import CAT
from medcat.config import Config
from medcat.vocab import Vocab

try:
    from .utils import load_model_pack
    from .combined_hints import CombinedHintMatcher, load_combined_hints
    from .value_resolver import ValueResolver
    from .candidate_restoration import CandidateRestoration
except ImportError:  # pragma: no cover - allows running as a script
    from utils import load_model_pack  # type: ignore
    from combined_hints import CombinedHintMatcher, load_combined_hints  # type: ignore
    from value_resolver import ValueResolver  # type: ignore
    from candidate_restoration import CandidateRestoration  # type: ignore


class CustomCAT:
    """Composite wrapper around MedCAT CAT adding combined hint support."""

    def __init__(
        self,
        model_pack_path: str | Path,
        *,
        combined_hints_path: str | Path | None = None,
    ) -> None:
        self.model_path = Path(model_pack_path)

        # Initialize MedCAT model
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

        # Ensure vocab is properly initialized
        if getattr(self.cat, "vocab", None) is None:
            self.cat.vocab = Vocab()
            pipeline = getattr(self.cat, "_pipeline", None)
            if pipeline is not None:
                for component in getattr(pipeline, "_components", []):
                    context_model = getattr(component, "context_model", None)
                    if context_model is not None:
                        context_model.vocab = self.cat.vocab

        # Initialize combined hints matcher
        combined_path = Path(combined_hints_path).expanduser() if combined_hints_path else None
        if combined_path and combined_path.is_dir():
            combined_path = combined_path / "internal_combined_hints.json"
        self.combined_matcher = CombinedHintMatcher(load_combined_hints(combined_path))

        # Initialize component modules lazily
        project_root = Path(__file__).resolve().parents[1]
        self._project_root = project_root
        self._value_resolver = None
        self._candidate_restoration = None

    @property
    def value_resolver(self):
        """Lazy initialization of value resolver."""
        if self._value_resolver is None:
            self._value_resolver = ValueResolver(self._project_root)
        return self._value_resolver
    
    @property
    def candidate_restoration(self):
        """Lazy initialization of candidate restoration."""
        if self._candidate_restoration is None:
            self._candidate_restoration = CandidateRestoration(self.cat, self.cdb, self.value_resolver)
        return self._candidate_restoration

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying CAT instance."""

        return getattr(self.cat, name)

    def extract_entities(self, text: str, *, min_confidence: float = 0.0) -> Dict[str, Any]:
        """Run MedCAT extraction and augment with combined hint matches."""
        result = self.cat.get_entities(text, only_cui=False)
        entities: Dict[str, Any] = result.setdefault("entities", {})

        # Add combined hint matches
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

        # Apply confidence filtering
        if min_confidence > 0:
            filtered_entities = {
                key: ent
                for key, ent in entities.items()
                if float(ent.get("acc", 0.0)) >= min_confidence
            }
            result["entities"] = filtered_entities
            entities = filtered_entities

        # Apply value-aware validation and candidate restoration
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
        initial_entity_count = len(entities)
        missing_value_cuis = set()

        # Apply value validation rules
        self.value_resolver.apply_value_rules(text, entities)
        
        # Collect missing CUIs that were removed
        keyword_rules = self.value_resolver.get_keyword_rules()
        for key, entity in list(entities.items()):
            if isinstance(key, str) and key.startswith("combined_"):
                continue
            cui = entity.get("cui")
            if cui is None:
                continue
            rule = keyword_rules.get(str(cui).upper())
            if not rule or not rule.requires_value:
                continue
            if not self.value_resolver.components_present(rule, text, entity):
                missing_value_cuis.add(str(cui).upper())
                continue
            match = self.value_resolver.find_value_match(rule, text, entity)
            if match is None:
                missing_value_cuis.add(str(cui).upper())
                continue
            if rule.is_numeric and (match.numeric is None or not rule.is_value_in_range(match.numeric)):
                missing_value_cuis.add(str(cui).upper())

        # Restore missing candidates using fallback logic
        self.candidate_restoration.restore_missing_candidates(
            text, entities, missing_value_cuis, initial_entity_count
        )

