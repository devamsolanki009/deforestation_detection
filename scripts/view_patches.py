"""Save a figure of sample patches: before, after, radar change, and label.

Picks the patches with the most loss, plus a few without any, from each split.

Run from the repo root:
    python -m scripts.view_patches --area subset
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from src.config import DEFAULT_CONFIG, load_config  # noqa: E402
from src.data.labels import IGNORE, LOSS  # noqa: E402
from src.data.patches import source_prefixes  # noqa: E402

LOSS_SAMPLES = 4
PLAIN_SAMPLES = 2
LABEL_COLORS = ListedColormap(["#1b1b1b", "#e0433a", "#8a8a8a"])  # no loss, loss, ignored
FALSE_COLOR_MAX = 4000  # reflectance shown at full brightness


def false_color(image: np.ndarray, band_names: list[str], year: int) -> np.ndarray:
    """SWIR, NIR, red composite: clear-cut land reads pink or brown, forest reads green."""
    channels = [image[band_names.index(f"{band}_{year}")] for band in ("B11", "B8", "B4")]
    return np.clip(np.dstack(channels) / FALSE_COLOR_MAX, 0, 1)


def choose(records: list[dict]) -> list[dict]:
    with_loss = sorted((r for r in records if int(r["loss_pixels"]) > 0), key=lambda r: -int(r["loss_pixels"]))
    without = [r for r in records if int(r["loss_pixels"]) == 0]
    # Plain patches come from evenly spaced positions, so they are not all neighbours.
    step = max(1, len(without) // max(1, PLAIN_SAMPLES))
    return with_loss[:LOSS_SAMPLES] + without[::step][:PLAIN_SAMPLES]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--area", choices=["probe", "subset", "full"], required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    before, after = config.years.before, config.years.after
    patch_dir = config.paths.patches_dir / args.area

    with (patch_dir / "metadata.csv").open(encoding="utf-8") as handle:
        records = [r for r in csv.DictReader(handle) if r["split"] != "buffer"]

    prefix = source_prefixes(config, args.area)[0]
    manifest = json.loads((config.paths.raw_dir / f"{prefix}.manifest.json").read_text(encoding="utf-8"))
    names = [b for b in manifest["bands"] if not b.startswith(("s2_clear_count", "lossyear", "treecover2000"))]

    samples = choose(records)
    figure, axes = plt.subplots(len(samples), 4, figsize=(12, 3 * len(samples)), squeeze=False)
    titles = [f"{before} false color", f"{after} false color", f"VH change {after}-{before} (dB)", "label"]

    for row, record in enumerate(samples):
        patch = np.load(patch_dir / f"{record['patch_id']}.npz")
        image, label = patch["image"], patch["label"]
        vh_change = (image[names.index(f"VH_{after}")].astype(float)
                     - image[names.index(f"VH_{before}")].astype(float)) / manifest["s1_db_scale"]
        display_label = np.select([label == LOSS, label == IGNORE], [1, 2], default=0)

        axes[row, 0].imshow(false_color(image, names, before))
        axes[row, 1].imshow(false_color(image, names, after))
        axes[row, 2].imshow(vh_change, cmap="RdBu", vmin=-6, vmax=6)
        axes[row, 3].imshow(display_label, cmap=LABEL_COLORS, vmin=0, vmax=2, interpolation="nearest")
        axes[row, 0].set_ylabel(f"{record['split']}\n{record['loss_pixels']} loss px", fontsize=9)
        for column, axis in enumerate(axes[row]):
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(titles[column], fontsize=10)

    figure.suptitle("Label: red = loss, gray = ignored, black = no loss", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.99))
    output = config.paths.outputs_dir / "phase1" / f"samples_{args.area}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=130)
    print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
