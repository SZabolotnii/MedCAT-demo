"""Dictionary coverage validation aligned with the custom Phase 1A ontology."""

from __future__ import annotations

from typing import Dict

import pytest

from medcat.cdb import CDB


def test_custom_dictionary_coverage(cdb: CDB, cdb_stats: Dict[str, float], cluster_mapping: Dict[str, str]) -> None:
    total_cuis = len(cdb.cui2info)
    assert total_cuis == int(cdb_stats["total_cuis"])

    missing_names = [cui for cui, info in cdb.cui2info.items() if not info.get("names")]
    assert not missing_names, f"CUIs without synonyms: {missing_names[:5]}"

    missing_preferred = [cui for cui, info in cdb.cui2info.items() if not info.get("preferred_name")]
    coverage = 1 - (len(missing_preferred) / total_cuis)
    expected_coverage = float(cdb_stats["preferred_names_coverage"]) / 100.0
    assert coverage >= expected_coverage - 0.001

    total_names = sum(len(info.get("names", [])) for info in cdb.cui2info.values())
    avg_names = total_names / total_cuis
    assert avg_names == pytest.approx(cdb_stats["avg_names_per_cui"], rel=0.01)

    distinct_type_ids = {
        str(type_id).upper()
        for info in cdb.cui2info.values()
        for type_id in (info.get("type_ids") or [])
    }
    assert len(distinct_type_ids) == int(cdb_stats["type_ids_count"])

    unmapped_types = [type_id for type_id in distinct_type_ids if type_id not in cluster_mapping]
    assert not unmapped_types, f"Type IDs missing from cluster mapping: {unmapped_types[:5]}"
