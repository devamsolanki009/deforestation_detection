# Deforestation change detection

Detects forest loss between a "before" and an "after" year from Sentinel-2 (optical) and
Sentinel-1 (radar) imagery, and outputs a loss mask plus estimated hectares lost.

Status: Phases 0 to 3 are complete. A baseline U-Net is trained and evaluated on a held-out
test region; see `outputs/runs/unet_resnet34_baseline/evaluation/metrics.json`.

## Project layout

```
configs/    YAML configs (region, years, patch size, paths)
src/        Python package: config.py, data/, models/, training/, inference/
scripts/    Command-line entry points
tests/      Unit tests
notebooks/  Exploration
app/        Demo app (Phase 5)
data/       Downloaded imagery and patches (not in git)
outputs/    Checkpoints, metrics, figures (not in git)
```

## Setup

Requires Python 3.11 and an NVIDIA GPU. Run these from the repo root.

With [uv](https://docs.astral.sh/uv/):

```powershell
uv venv --python 3.11 .venv
.venv\Scripts\activate
uv pip install -r requirements-torch.txt
uv pip install -r requirements.txt
```

Or with the standard library (use `source .venv/bin/activate` on Linux):

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-torch.txt
pip install -r requirements.txt
```

Install `requirements-torch.txt` first. Otherwise `segmentation-models-pytorch` pulls in a
CPU-only PyTorch on Windows.

## Check the environment

```powershell
python -m scripts.check_env
pytest
```

`check_env` validates the config, imports every required package, confirms the GPU is visible,
and runs one mixed-precision training step on it.

## Data pipeline

Requires an Earth Engine account and a Cloud project set in `configs/default.yaml`. Exports use
real quota, so `subset` and `full` need `--confirm`, and a matching task blocks a repeat run.

```powershell
python -m scripts.check_s1_coverage              # Sentinel-1 images per pixel
python -m scripts.report_labels                  # loss pixels kept or dropped by each label rule
python -m scripts.export_stack --area probe      # small export that measures EECU usage
python -m scripts.export_stack --area subset --confirm --wait   # NE quarter
python -m scripts.export_stack --area nw --confirm --wait       # likewise sw and se
```

The `full` area is built from the four quarter exports (`subset` is the NE quarter), so
`make_patches --area full` and `view_patches --area full` need all four downloaded.

Download the exported `.tif` files from Google Drive into `data/raw/`, then run:

```powershell
python -m scripts.make_patches --area subset     # patches, labels, splits, summary
python -m scripts.view_patches --area subset     # sample figure in outputs/phase1/
```

Labels are built locally from the raw Hansen bands, so changing a label rule needs no new export.

## Training and evaluation

```powershell
python -m scripts.train --train-config configs\train\unet_resnet34.yaml
python -m scripts.evaluate --run unet_resnet34_baseline         # tunes on val, scores test
python -m scripts.plot_evaluation --run unet_resnet34_baseline  # overlays and failure gallery
python -m scripts.diagnose_errors --run unet_resnet34_baseline --split val
```

Evaluation compares two post-processing pipelines, each tuned on val: with and without a
forest mask that keeps detections only where Hansen shows forest when the before year began.
The pipeline with the better val F1 becomes the primary one. Results go to
`outputs/runs/<run>/evaluation/`.

## Configuration

`configs/default.yaml` sets the region, years, patch size and paths. To try another region, copy
the file and pass the copy with `--config`. Invalid values stop with an error that names the
problem.
