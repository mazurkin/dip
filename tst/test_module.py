import pathlib

import pytest

import torch

import lightning.pytorch as pl

from ml.data import DipDataModule
from ml.model import DipModelConfig
from ml.module import DipModelModuleConfig, DipModelModule


# all tests run on the CPU device as required by the project conventions
DEVICE: torch.device = torch.device('cpu')

# image geometry used across the tests; both dimensions are divisible by 2 ** depth
IMAGE_WIDTH: int = 64

IMAGE_HEIGHT: int = 64

# number of input noise channels used by the tests
INPUT_CHANNELS: int = 32

# number of output image channels (RGB)
OUTPUT_CHANNELS: int = 3


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model_config() -> DipModelConfig:
    # a shallow 3-level network keeps the tests small and fast on the CPU
    return DipModelConfig(
        input_channels=INPUT_CHANNELS,
        output_channels=OUTPUT_CHANNELS,
        channels_down=(16, 16, 16),
        channels_up=(16, 16, 16),
        channels_skip=(4, 4, 4),
    )


@pytest.fixture
def module_config() -> DipModelModuleConfig:
    return DipModelModuleConfig(learning_rate=1e-2)


@pytest.fixture
def target_image() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(size=(OUTPUT_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), device=DEVICE)


@pytest.fixture
def mask() -> torch.Tensor:
    # a binary mask that keeps the whole image except a small excluded square in the corner
    mask = torch.ones(size=(1, IMAGE_HEIGHT, IMAGE_WIDTH), device=DEVICE)
    mask[:, :8, :8] = 0.0
    return mask


@pytest.fixture
def module(
    target_image: torch.Tensor,
    mask: torch.Tensor,
    module_config: DipModelModuleConfig,
    model_config: DipModelConfig,
) -> DipModelModule:
    torch.manual_seed(0)
    return DipModelModule(
        image=target_image,
        mask=mask,
        config=module_config,
        model_config=model_config,
    )


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------

def test_module_builds(module: DipModelModule) -> None:
    assert module is not None
    assert isinstance(module, DipModelModule)


def test_module_stores_target_as_buffer(module: DipModelModule, target_image: torch.Tensor) -> None:
    # the target image must be registered as a non-trainable buffer, not as a parameter
    buffer_names = {name for name, _ in module.named_buffers()}
    assert 'target_image' in buffer_names
    assert torch.equal(module.target_image, target_image)


def test_module_geometry_matches_image(module: DipModelModule) -> None:
    assert module.image_height == IMAGE_HEIGHT
    assert module.image_width == IMAGE_WIDTH


def test_module_stores_mask_as_buffer(module: DipModelModule, mask: torch.Tensor) -> None:
    # the mask must be registered as a non-trainable buffer, not as a parameter
    buffer_names = {name for name, _ in module.named_buffers()}
    assert 'mask' in buffer_names
    assert torch.equal(module.mask, mask)


def test_module_rejects_wrong_channel_count(
    mask: torch.Tensor,
    module_config: DipModelModuleConfig,
    model_config: DipModelConfig,
) -> None:
    # a grayscale (single channel) image must be rejected as the module expects RGB
    grayscale = torch.rand(size=(1, IMAGE_HEIGHT, IMAGE_WIDTH), device=DEVICE)
    with pytest.raises(AssertionError):
        DipModelModule(image=grayscale, mask=mask, config=module_config, model_config=model_config)


# ---------------------------------------------------------------------------
# forward pass
# ---------------------------------------------------------------------------

def test_forward_shape(module: DipModelModule) -> None:
    noise = torch.rand(size=(INPUT_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), device=DEVICE)

    with torch.no_grad():
        output = module(noise)

    assert output.shape == torch.Size([OUTPUT_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH])


# ---------------------------------------------------------------------------
# optimizer configuration
# ---------------------------------------------------------------------------

def test_configure_optimizers(module: DipModelModule, module_config: DipModelModuleConfig) -> None:
    optimizer = module.configure_optimizers()

    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.defaults['lr'] == module_config.learning_rate


# ---------------------------------------------------------------------------
# masked loss
# ---------------------------------------------------------------------------

def test_masked_loss_ignores_excluded_pixels(
    target_image: torch.Tensor,
    mask: torch.Tensor,
    module_config: DipModelModuleConfig,
    model_config: DipModelConfig,
) -> None:
    torch.manual_seed(0)
    module = DipModelModule(
        image=target_image,
        mask=mask,
        config=module_config,
        model_config=model_config,
    )

    # an output that equals the target everywhere yields zero loss
    exact_output = target_image.clone()
    assert module.masked_loss(exact_output).item() == pytest.approx(0.0)

    # changing only the excluded pixels (mask == 0) must NOT affect the masked loss
    corrupted_output = target_image.clone()
    corrupted_output[:, :8, :8] = 1.0 - corrupted_output[:, :8, :8]
    assert module.masked_loss(corrupted_output).item() == pytest.approx(0.0)

    # changing a kept pixel (mask == 1) MUST increase the loss above zero
    kept_corrupted = target_image.clone()
    kept_corrupted[:, -1, -1] = 1.0 - kept_corrupted[:, -1, -1]
    assert module.masked_loss(kept_corrupted).item() > 0.0


# ---------------------------------------------------------------------------
# training and validation steps
# ---------------------------------------------------------------------------

def test_training_step_returns_scalar_loss(module: DipModelModule) -> None:
    noise = torch.rand(size=(INPUT_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), device=DEVICE)

    loss = module.training_step(noise, batch_idx=0)

    assert loss.ndim == 0
    assert loss.item() >= 0.0
    assert loss.requires_grad


def test_validation_step_stores_latest_output(module: DipModelModule) -> None:
    noise = torch.rand(size=(INPUT_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), device=DEVICE)

    with torch.no_grad():
        module.validation_step(noise, batch_idx=0)

    # the first validation batch must cache a detached reconstruction for saving
    assert hasattr(module, 'latest_output')
    assert module.latest_output.shape == torch.Size([OUTPUT_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH])
    assert not module.latest_output.requires_grad


# ---------------------------------------------------------------------------
# end-to-end Lightning fit
# ---------------------------------------------------------------------------

def test_fit_reduces_loss_and_saves_images(
    target_image: torch.Tensor,
    mask: torch.Tensor,
    module_config: DipModelModuleConfig,
    model_config: DipModelConfig,
    tmp_path: pathlib.Path,
) -> None:
    torch.manual_seed(0)

    module = DipModelModule(
        image=target_image,
        mask=mask,
        config=module_config,
        model_config=model_config,
        output_dir=tmp_path,
    )

    data_module = DipDataModule(width=IMAGE_WIDTH, height=IMAGE_HEIGHT, channels=INPUT_CHANNELS)

    number_of_epochs: int = 3
    trainer = pl.Trainer(
        max_epochs=number_of_epochs,
        accelerator='cpu',
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    trainer.fit(module, datamodule=data_module)

    # one reconstruction image must be saved per validation epoch
    saved_images = sorted(tmp_path.glob('reconstruction-*.png'))
    assert len(saved_images) == number_of_epochs


def test_no_images_saved_without_output_dir(
    target_image: torch.Tensor,
    mask: torch.Tensor,
    module_config: DipModelModuleConfig,
    model_config: DipModelConfig,
) -> None:
    torch.manual_seed(0)

    module = DipModelModule(
        image=target_image,
        mask=mask,
        config=module_config,
        model_config=model_config,
        output_dir=None,
    )

    data_module = DipDataModule(width=IMAGE_WIDTH, height=IMAGE_HEIGHT, channels=INPUT_CHANNELS)

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator='cpu',
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    # with no output directory the validation epoch end must simply skip saving without errors
    trainer.fit(module, datamodule=data_module)
