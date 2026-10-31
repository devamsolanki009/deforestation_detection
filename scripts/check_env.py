"""Check that the environment can run this project.

Run from the repo root:
    python -m scripts.check_env
    python -m scripts.check_env --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from src.config import DEFAULT_CONFIG, ConfigError, load_config

# Import name -> pip package name.
REQUIRED_PACKAGES = {
    "yaml": "pyyaml",
    "numpy": "numpy",
    "rasterio": "rasterio",
    "ee": "earthengine-api",
    "geemap": "geemap",
    "segmentation_models_pytorch": "segmentation-models-pytorch",
    "matplotlib": "matplotlib",
    "torch": "torch",
}

# Two dates of 10 Sentinel-2 bands plus 2 Sentinel-1 bands.
SMOKE_TEST_CHANNELS = 24
SMOKE_TEST_BATCH = 2


def check_config(path: Path) -> bool:
    print(f"\n[config] {path}")
    try:
        config = load_config(path)
    except ConfigError as exc:
        print(f"  FAIL  {exc}")
        return False
    print(f"  OK    region={config.region.name}, years={config.years.before}->{config.years.after}, "
          f"patch={config.patch.size}px")
    return True


def check_packages() -> bool:
    print("\n[packages]")
    all_found = True
    for module_name, pip_name in REQUIRED_PACKAGES.items():
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            print(f"  FAIL  {pip_name}: {exc}")
            all_found = False
        else:
            print(f"  OK    {pip_name} {getattr(module, '__version__', 'unknown version')}")
    return all_found


def check_cuda(patch_size: int) -> bool:
    """Confirm the GPU is visible, then run one AMP training step on it."""
    print("\n[cuda]")
    try:
        import torch
    except ImportError:
        print("  FAIL  torch is not installed, so CUDA cannot be tested")
        return False

    print(f"  torch build CUDA version: {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("  FAIL  torch.cuda.is_available() is False. "
              "A CPU-only torch build or an outdated driver are the usual causes.")
        return False

    properties = torch.cuda.get_device_properties(0)
    print(f"  OK    {properties.name}, {properties.total_memory / 1024**3:.1f} GiB VRAM")

    device = torch.device("cuda")
    model = torch.nn.Conv2d(SMOKE_TEST_CHANNELS, 1, kernel_size=3, padding=1).to(device)
    inputs = torch.randn(SMOKE_TEST_BATCH, SMOKE_TEST_CHANNELS, patch_size, patch_size, device=device)
    scaler = torch.amp.GradScaler("cuda")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    with torch.amp.autocast("cuda"):
        loss = model(inputs).mean()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    print(f"  OK    AMP training step ran, peak VRAM {torch.cuda.max_memory_allocated() / 1024**2:.0f} MiB")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    print(f"Python {sys.version.split()[0]} at {sys.executable}")

    config_ok = check_config(args.config)
    packages_ok = check_packages()
    patch_size = load_config(args.config).patch.size if config_ok else 256
    cuda_ok = check_cuda(patch_size)

    passed = config_ok and packages_ok and cuda_ok
    print(f"\nResult: {'all checks passed' if passed else 'some checks FAILED'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
