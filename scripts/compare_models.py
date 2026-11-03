"""Build a results table comparing models across seeds.

Each model is a group of evaluated runs. Run from the repo root:
    python -m scripts.compare_models \
        --model "U-Net (early fusion)=unet_resnet34_baseline,unet_resnet34_baseline_seed1" \
        --model "Siamese U-Net=siamese_unet_resnet34_seed42,siamese_unet_resnet34_seed1"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.config import DEFAULT_CONFIG, load_config

METRICS = ("f1", "iou", "precision", "recall")


def load_run(runs_dir: Path, run: str) -> dict:
    metrics = json.loads((runs_dir / run / "evaluation" / "metrics.json").read_text(encoding="utf-8"))
    summary = json.loads((runs_dir / run / "summary.json").read_text(encoding="utf-8"))
    pipeline = metrics["pipelines"][metrics["primary_pipeline"]]
    test = pipeline["splits"]["test"]
    return {
        "run": run,
        "pipeline": metrics["primary_pipeline"],
        "threshold": pipeline["threshold"],
        "val_f1": pipeline["splits"]["val"]["metrics"]["f1"],
        **{f"test_{name}": test["metrics"][name] for name in METRICS},
        "test_predicted_ha": test["hectares"]["predicted_loss_ha_on_labeled_pixels"],
        "test_labeled_ha": test["hectares"]["labeled_loss_ha"],
        "parameters": summary.get("parameters"),
        "epoch_seconds": summary.get("mean_epoch_seconds"),
    }


def spread(values: list[float]) -> str:
    """Mean with min to max range; a single run shows just its value."""
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{np.mean(values):.3f} ({min(values):.3f}-{max(values):.3f})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", action="append", required=True, help='"Label=run1,run2,..."')
    args = parser.parse_args()

    config = load_config(args.config)
    runs_dir = config.paths.outputs_dir / "runs"
    groups = {}
    for spec in args.model:
        label, runs = spec.split("=", 1)
        groups[label] = [load_run(runs_dir, run.strip()) for run in runs.split(",")]

    header = ["Model", "Runs", "Val F1", "Test F1", "Test IoU", "Test precision", "Test recall",
              "Predicted ha / labeled ha", "Parameters", "s / epoch"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for label, runs in groups.items():
        hectares = f"{np.mean([r['test_predicted_ha'] for r in runs]):.0f} / {runs[0]['test_labeled_ha']:.0f}"
        params = runs[0]["parameters"]
        seconds = [r["epoch_seconds"] for r in runs if r["epoch_seconds"]]
        lines.append("| " + " | ".join([
            label, str(len(runs)), spread([r["val_f1"] for r in runs]),
            *[spread([r[f"test_{name}"] for r in runs]) for name in METRICS],
            hectares, f"{params / 1e6:.1f}M" if params else "n/a",
            f"{np.mean(seconds):.1f}" if seconds else "n/a",
        ]) + " |")

    table = "\n".join(lines)
    output_dir = config.paths.outputs_dir / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results_table.md").write_text(table + "\n", encoding="utf-8")
    (output_dir / "runs.json").write_text(json.dumps(groups, indent=2), encoding="utf-8")
    print(table)
    print(f"\nSaved {output_dir / 'results_table.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
