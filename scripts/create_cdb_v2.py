"""Build a MedCAT Concept Database from the custom ontology CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any, Dict


def _lazy_import_medcat():
    """Import MedCAT modules lazily to keep CLI responsive."""

    try:
        from medcat.cdb import CDB  # noqa: F401  # for type hints
        from medcat.config import Config
        from medcat.model_creation.cdb_maker import CDBMaker
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise SystemExit(f"Failed to import MedCAT dependency: {exc}") from exc

    return CDBMaker, Config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/internal_medcat_v2.csv"),
        help="Path to the MedCAT-ready CSV produced by transform_to_medcat_format.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/IEE_MedCAT_v1"),
        help="Directory where the generated CDB and metadata will be stored.",
    )
    parser.add_argument(
        "--combined-hints",
        type=Path,
        default=Path("data/internal_combined_hints.json"),
        help=(
            "Optional path to combined hints JSON produced during CSV transformation. "
            "If present it will be copied next to the generated CDB."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing artefacts in the output directory.",
    )
    return parser.parse_args()


def configure_for_dictionary_mode(config: "Config") -> "Config":
    """Tune MedCAT config for dictionary-only inference."""

    # General NLP settings
    config.general.nlp.modelname = "en_core_web_md"
    config.general.spell_check = True
    config.general.spell_check_len_limit = 3

    # NER component tweaks
    ner_cfg = config.components.ner
    ner_cfg.min_name_len = 2
    ner_cfg.upper_case_limit_len = 6
    ner_cfg.check_upper_case_names = True
    ner_cfg.try_reverse_word_order = True

    # Linking component tweaks
    linking_cfg = config.components.linking
    linking_cfg.train = False
    linking_cfg.always_calculate_similarity = False
    linking_cfg.calculate_dynamic_threshold = False
    linking_cfg.similarity_threshold = 1.0
    linking_cfg.prefer_primary_name = 0.6
    linking_cfg.prefer_frequent_concepts = 0.3
    linking_cfg.disamb_length_limit = 6

    return config


def create_cdb(csv_path: Path) -> "CDB":
    CDBMaker, Config = _lazy_import_medcat()

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {csv_path}. Run scripts.transform_to_medcat_format first."
        )

    config = configure_for_dictionary_mode(Config())
    maker = CDBMaker(config=config)

    print(f"🔧 Building CDB from {csv_path}")
    cdb = maker.prepare_csvs(
        csv_paths=[str(csv_path)],
        sep=",",
        encoding="utf-8",
        full_build=False,
    )
    print(f"✅ CDB created with {len(cdb.cui2info)} CUIs")
    return cdb


def compute_stats(cdb: "CDB") -> Dict[str, Any]:
    total_cuis = len(cdb.cui2info)
    total_names = 0
    all_type_ids = set()
    preferred_names = 0

    for info in cdb.cui2info.values():
        names = info.get("names") or []
        total_names += len(names)
        type_ids = info.get("type_ids") or set()
        all_type_ids.update(type_ids)
        if info.get("preferred_name"):
            preferred_names += 1

    stats = {
        "total_cuis": total_cuis,
        "total_names": total_names,
        "avg_names_per_cui": (total_names / total_cuis) if total_cuis else 0,
        "type_ids_count": len(all_type_ids),
        "preferred_names_coverage": (
            preferred_names / total_cuis * 100 if total_cuis else 0
        ),
    }
    return stats


def save_cdb(cdb: "CDB", output_dir: Path, overwrite: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cdb_path = output_dir / "custom_cdb_v2"
    if cdb_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{cdb_path} already exists. Re-run with --overwrite to replace it."
            )
        if cdb_path.is_dir():
            shutil.rmtree(cdb_path)
        else:
            cdb_path.unlink()
    cdb_path.mkdir(parents=True, exist_ok=True)

    cdb.save(str(cdb_path), overwrite=overwrite)
    config_path = output_dir / "config.json"
    config_path.write_text(cdb.config.model_dump_json(indent=2), encoding="utf-8")

    stats = compute_stats(cdb)
    stats_path = output_dir / "cdb_stats.json"
    with stats_path.open("w", encoding="utf-8") as dst:
        json.dump(stats, dst, indent=2)

    print("\n💾 Artefacts written:")
    print(f"   CDB: {cdb_path}")
    print(f"   Config: {config_path}")
    print(f"   Stats: {stats_path}")

    return cdb_path


def copy_combined_hints(source: Path, destination_dir: Path, overwrite: bool) -> None:
    if not source or not source.exists():
        return

    destination = destination_dir / source.name
    if destination.exists() and not overwrite:
        return

    data = json.loads(source.read_text(encoding="utf-8"))
    destination.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"   Combined hints metadata: {destination}")


def main() -> None:
    args = parse_args()

    cdb = create_cdb(args.csv)
    save_cdb(cdb, args.output_dir, args.overwrite)

    if args.combined_hints:
        copy_combined_hints(args.combined_hints, args.output_dir, args.overwrite)

    print("\n✅ Phase 1A CDB is ready for use!")


if __name__ == "__main__":
    main()
