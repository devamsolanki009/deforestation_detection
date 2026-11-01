"""Cut downloaded GeoTIFFs into patches, split them, and report the result.

Put the Drive files in the raw data folder, then run from the repo root:
    python -m scripts.make_patches --area subset
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.config import DEFAULT_CONFIG, load_config
from src.data.patches import assign_splits, cut_file, split_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--area", choices=["probe", "subset", "full"], required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    prefix = f"stack_{config.region.name}_{args.area}_{config.years.before}_{config.years.after}"
    manifest = json.loads((config.paths.raw_dir / f"{prefix}.manifest.json").read_text(encoding="utf-8"))

    # Skip the small direct-download sample, which shares this name prefix.
    files = sorted(p for p in config.paths.raw_dir.glob(f"{prefix}*.tif") if not p.name.endswith(".local.tif"))
    if not files:
        print(f"No GeoTIFFs named {prefix}*.tif in {config.paths.raw_dir}")
        return 1

    output_dir = config.paths.patches_dir / args.area
    output_dir.mkdir(parents=True, exist_ok=True)

    records, skipped = [], 0
    for path in files:
        file_records, file_skipped = cut_file(path, manifest, config, output_dir)
        records += file_records
        skipped += file_skipped
        print(f"{path.name}: {len(file_records)} patches, {file_skipped} skipped for missing imagery", flush=True)

    if not records:
        print("No patches were produced.")
        return 1

    assign_splits(records, config)
    with (output_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    summary = {
        "source_files": len(files),
        "skipped_for_missing_imagery": skipped,
        "splits": split_summary(records),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
