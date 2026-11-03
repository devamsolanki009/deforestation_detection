import json

import numpy as np
import pytest
import torch

from src.config import ConfigError, REPO_ROOT
from src.data.dataset import (
    PatchDataset, augment, compute_band_stats, load_or_compute_band_stats, loss_patch_sampler,
)
from src.data.labels import IGNORE
from src.models.factory import build_model
from src.training.config import LossConfig, ModelConfig, load_train_config
from src.training.losses import DiceFocalLoss
from src.training.metrics import LossClassCounts

NODATA = -32768
TRAIN_CONFIG = REPO_ROOT / "configs" / "train" / "unet_resnet34.yaml"


def make_patches(tmp_path, count=3, bands=4, size=32):
    rows = []
    for index in range(count):
        image = np.full((bands, size, size), 100 * (index + 1), dtype=np.int16)
        label = np.zeros((size, size), dtype=np.uint8)
        label[:4, :4] = 1
        label[-2:, :] = IGNORE
        np.savez(tmp_path / f"p{index}.npz", image=image, label=label,
                 clear_count=np.zeros((2, size, size), dtype=np.int16))
        rows.append(f"p{index},train,16,{2 * size},{size * size}")
    (tmp_path / "metadata.csv").write_text(
        "patch_id,split,loss_pixels,ignored_pixels,total_pixels\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return tmp_path


def test_band_stats_skip_nodata(tmp_path):
    image = np.array([[[10, 20], [30, NODATA]]], dtype=np.int16)
    np.savez(tmp_path / "a.npz", image=image)
    stats = compute_band_stats(tmp_path, [{"patch_id": "a"}], NODATA)
    assert stats["mean"] == [pytest.approx(20.0)]
    assert stats["std"] == [pytest.approx(np.std([10, 20, 30]))]


def test_band_stats_are_cached(tmp_path):
    make_patches(tmp_path)
    first = load_or_compute_band_stats(tmp_path, NODATA)
    assert json.loads((tmp_path / "band_stats.json").read_text())["mean"] == first["mean"]
    assert first["patches"] == 3


def test_dataset_normalizes_and_fills_nodata(tmp_path):
    make_patches(tmp_path, count=1)
    image = np.load(tmp_path / "p0.npz")["image"]
    image[0, 0, 0] = NODATA
    np.savez(tmp_path / "p0.npz", image=image, label=np.zeros((32, 32), dtype=np.uint8))

    stats = {"mean": [100.0] * 4, "std": [10.0] * 4}
    x, y = PatchDataset(tmp_path, "train", stats, NODATA)[0]
    assert x.shape == (4, 32, 32) and x.dtype == torch.float32
    assert y.shape == (1, 32, 32)
    assert x[0, 0, 0] == 0 and x[1, 0, 0] == 0  # nodata filled; 100 normalizes to 0


def test_augment_moves_image_and_label_together():
    image = np.zeros((2, 8, 8), dtype=np.float32)
    label = np.zeros((8, 8), dtype=np.uint8)
    image[:, 0, 7], label[0, 7] = 5, 1
    rng = np.random.default_rng(0)
    for _ in range(20):
        new_image, new_label = augment(image, label, rng)
        assert ((new_image[0] == 5) == (new_label == 1)).all()
        assert new_label.sum() == 1


def test_sampler_favours_loss_patches():
    records = [{"loss_pixels": "50", "ignored_pixels": "0", "total_pixels": "100"},
               {"loss_pixels": "0", "ignored_pixels": "0", "total_pixels": "100"}]
    sampler = loss_patch_sampler(records, min_fraction=0.01, weight=3.0, seed=0)
    assert sampler.weights.tolist() == [3.0, 1.0]
    assert len(list(sampler)) == 2


def test_metrics_skip_ignored_pixels():
    logits = torch.tensor([[[[5.0, 5.0, -5.0, 5.0]]]])
    labels = torch.tensor([[[[1.0, 0.0, 1.0, IGNORE]]]])
    counts = LossClassCounts()
    counts.update(logits, labels, threshold=0.5)
    assert (counts.tp, counts.fp, counts.fn) == (1, 1, 1)
    metrics = counts.compute()
    assert metrics["iou"] == pytest.approx(1 / 3)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["precision"] == metrics["recall"] == pytest.approx(0.5)


def test_metrics_with_no_positives_are_zero():
    assert LossClassCounts().compute() == {"iou": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}


def test_loss_ignores_255():
    loss_fn = DiceFocalLoss(LossConfig(dice_weight=1.0, focal_weight=1.0))
    logits = torch.zeros(1, 1, 4, 4)
    labels = torch.full((1, 1, 4, 4), float(IGNORE))
    labels[..., 0, 0] = 1
    changed = logits.clone()
    changed[..., 1:, :] = 50.0  # only ignored pixels change
    assert loss_fn(logits, labels).item() == pytest.approx(loss_fn(changed, labels).item())


def test_model_output_matches_input_size():
    model = build_model(ModelConfig("unet", "resnet34", None), in_channels=24)
    with torch.no_grad():
        assert model(torch.zeros(1, 24, 64, 64)).shape == (1, 1, 64, 64)


def test_unknown_arch_is_rejected():
    with pytest.raises(ValueError, match="arch"):
        build_model(ModelConfig("deeplab", "resnet34", None), in_channels=24)


def test_train_config_loads():
    config = load_train_config(TRAIN_CONFIG)
    assert config.model.encoder == "resnet34"
    assert config.data.oversample_weight >= 1


def test_bad_train_config_is_rejected(tmp_path):
    text = TRAIN_CONFIG.read_text(encoding="utf-8").replace("threshold: 0.5", "threshold: 1.5")
    path = tmp_path / "bad.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="threshold"):
        load_train_config(path)
