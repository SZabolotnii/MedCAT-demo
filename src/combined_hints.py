"""Combined hints matching module for gap-tolerant entity detection."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class CombinedHintDefinition:
    """Definition of a combined hint pattern."""
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
        """Compile a combined hint definition into a regex pattern."""
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
        """Find all combined hint matches in the given text."""
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


def load_combined_hints(path: Optional[Path]) -> List[CombinedHintDefinition]:
    """Load combined hint definitions from JSON file."""
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
