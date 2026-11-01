# Deforestation change detection

Detects forest loss between a "before" and an "after" year from Sentinel-2 (optical) and
Sentinel-1 (radar) imagery, and outputs a loss mask plus estimated hectares lost.

Status: Phase 1 (data). The pipeline has run on a test quarter of the region. No models have
been trained yet.

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
python -m scripts.export_stack --area subset --confirm --wait
```

Download the exported `.tif` files from Google Drive into `data/raw/`, then run:

```powershell
python -m scripts.make_patches --area subset     # patches, labels, splits, summary
python -m scripts.view_patches --area subset     # sample figure in outputs/phase1/
```

Labels are built locally from the raw Hansen bands, so changing a label rule needs no new export.

## Configuration

`configs/default.yaml` sets the region, years, patch size and paths. To try another region, copy
the file and pass the copy with `--config`. Invalid values stop with an error that names the
problem.
