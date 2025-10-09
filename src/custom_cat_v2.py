"""Custom MedCAT wrapper with gap-tolerant combined hint matching."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from medcat.cdb import CDB
from medcat.cat import CAT
from medcat.config import Config
from medcat.vocab import Vocab
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
            result["entities"] = {
                key: ent
                for key, ent in entities.items()
                if float(ent.get("acc", 0.0)) >= min_confidence
            }

        return result

    def batch_process(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Process a list of texts sequentially."""

        return [self.extract_entities(text) for text in texts]
