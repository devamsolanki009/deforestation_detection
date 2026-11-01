"""Cut downloaded GeoTIFFs into patches, split them, and report the result.

Put the Drive files in the raw data folder, then run from the repo root:
    python -m scripts.make_patches --area subset
    python -m scripts.make_patches --area full
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.config import DEFAULT_CONFIG, load_config
from src.data.patches import assign_splits, cut_mosaic, source_prefixes, split_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--area", choices=["probe", "subset", "full"], required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    prefixes = source_prefixes(config, args.area)
    manifest = json.loads((config.paths.raw_dir / f"{prefixes[0]}.manifest.json").read_text(encoding="utf-8"))

    files = []
    for prefix in prefixes:
        # Skip the small direct-download sample, which shares this name prefix.
        found = sorted(p for p in config.paths.raw_dir.glob(f"{prefix}*.tif") if not p.name.endswith(".local.tif"))
        if not found:
            print(f"No GeoTIFFs named {prefix}*.tif in {config.paths.raw_dir}")
            return 1
        print(f"{prefix}: {len(found)} files", flush=True)
        files += found

    output_dir = config.paths.patches_dir / args.area
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_patch in output_dir.glob("*.npz"):
        old_patch.unlink()  # patches are derived data; stale ones would mix with the new grid

    records, skipped = cut_mosaic(files, manifest, config, output_dir)
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
        "patches_written": len(records),
        "cells_skipped_for_missing_imagery_or_outer_edge": skipped,
        "splits": split_summary(records),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
