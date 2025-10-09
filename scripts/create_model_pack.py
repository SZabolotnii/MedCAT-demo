"""Package the generated MedCAT artefacts into a distributable archive."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("models/IEE_MedCAT_v1"),
        help="Directory with the generated CDB, config, stats and optional metadata.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/IEE_MedCAT_v1.zip"),
        help="Where to write the zipped model pack.",
    )
    return parser.parse_args()


def create_model_pack(source_dir: Path, output_path: Path) -> Path:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for item in source_dir.iterdir():
            target = tmp_path / item.name
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)

        if output_path.suffix == ".zip":
            base_name = str(output_path.with_suffix(""))
        else:
            base_name = str(output_path)
            output_path = output_path.with_suffix(".zip")

        archive_path = shutil.make_archive(base_name, "zip", tmp_path)
        final_path = Path(archive_path)
        if final_path != output_path:
            shutil.move(final_path, output_path)
    return output_path


def main() -> None:
    args = parse_args()
    pack_path = create_model_pack(args.source_dir, args.output)
    print(f"✅ Model pack created at {pack_path}")


if __name__ == "__main__":
    main()
