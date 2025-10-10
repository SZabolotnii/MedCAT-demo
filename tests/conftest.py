"""Shared pytest fixtures for the MedCAT validation test suite."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.custom_cat_v2 import CustomCAT


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def model_path(project_root: Path) -> Path:
    return project_root / "models" / "IEE_MedCAT_v1"


@pytest.fixture(scope="session")
def combined_hints_path(model_path: Path, project_root: Path) -> Path:
    candidate = model_path / "internal_combined_hints.json"
    if candidate.exists():
        return candidate
    fallback = project_root / "data" / "internal_combined_hints.json"
    if fallback.exists():
        return fallback
    raise FileNotFoundError("Combined hints metadata was not found in the expected locations.")


@pytest.fixture(scope="session")
def custom_cat(model_path: Path, combined_hints_path: Path) -> CustomCAT:
    return CustomCAT(model_path, combined_hints_path=combined_hints_path)


@pytest.fixture(scope="session")
def cdb(custom_cat: CustomCAT):
    return custom_cat.cdb


@pytest.fixture(scope="session")
def cluster_mapping(project_root: Path) -> Dict[str, str]:
    path = project_root / "data" / "valid_clusters.json"
    if not path.exists():
        raise FileNotFoundError(f"Cluster mapping file is missing: {path}")
    entries = json.loads(path.read_text(encoding="utf-8"))
    return {str(entry["id"]).upper(): entry["title"] for entry in entries}


@pytest.fixture(scope="session")
def cdb_stats(model_path: Path) -> Dict[str, float]:
    path = model_path / "cdb_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"CDB statistics file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def annotated_entity_dataset(project_root: Path) -> List[dict]:
    path = project_root / "data" / "phase1a_annotated_entities.json"
    if not path.exists():
        raise FileNotFoundError(f"Annotated dataset missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Annotated dataset must be a list of documents.")
    return data
