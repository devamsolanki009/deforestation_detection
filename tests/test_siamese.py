import torch

from src.data.gee import stack_band_names
from src.config import load_config
from src.data.patches import image_band_indices
from src.models.factory import build_model
from src.models.siamese import SiameseUnet
from src.training.config import ModelConfig

import pytest


def small_model():
    torch.manual_seed(0)
    return SiameseUnet("resnet34", None, channels_per_date=12).eval()


def test_output_matches_input_size():
    with torch.no_grad():
        assert small_model()(torch.zeros(2, 24, 64, 64)).shape == (2, 1, 64, 64)


def test_both_dates_share_one_encoder():
    model = small_model()
    assert sum(1 for name, _ in model.named_modules() if name == "encoder") == 1
    # The first conv sees 12 channels: one date, not both.
    assert model.encoder.conv1.in_channels == 12


def test_swapping_dates_changes_the_output():
    # Fusion keeps direction, so loss (forest then bare) differs from regrowth (bare then forest).
    model = small_model()
    x = torch.randn(1, 24, 64, 64)
    swapped = torch.cat([x[:, 12:], x[:, :12]], dim=1)
    with torch.no_grad():
        assert not torch.allclose(model(x), model(swapped))


def test_identical_dates_do_not_depend_on_batch_partner():
    # Before and after are split from one encoder batch, so a pairing bug would mix samples.
    # Convolution kernels differ slightly by batch size, so allow floating-point noise.
    model = small_model()
    x = torch.randn(2, 24, 64, 64)
    with torch.no_grad():
        assert torch.allclose(model(x)[:1], model(x[:1]), rtol=1e-4, atol=1e-3)
        assert torch.allclose(model(x)[1:], model(x[1:]), rtol=1e-4, atol=1e-3)


def test_first_half_of_stack_is_the_before_year():
    config = load_config()
    bands = stack_band_names(config)
    names = [bands[i] for i in image_band_indices(bands)]
    assert all(name.endswith(str(config.years.before)) for name in names[:12])
    assert all(name.endswith(str(config.years.after)) for name in names[12:])


def test_factory_builds_siamese_and_rejects_odd_channels():
    assert isinstance(build_model(ModelConfig("siamese_unet", "resnet34", None), 24), SiameseUnet)
    with pytest.raises(ValueError, match="even"):
        build_model(ModelConfig("siamese_unet", "resnet34", None), 23)
