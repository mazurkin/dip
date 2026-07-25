import pathlib

import numpy

import pytest

import PIL.Image

import torch

from ml.model import DipModelConfig
from ml.module import DipModelModuleConfig
from ml.trainer import DipTrainer


# image geometry used across the tests; both dimensions are divisible by 2 ** depth
IMAGE_WIDTH: int = 64

IMAGE_HEIGHT: int = 64


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model_config() -> DipModelConfig:
    # a shallow 3-level network keeps the tests small and fast on the CPU
    return DipModelConfig(
        input_channels=32,
        output_channels=3,
        channels_down=(16, 16, 16),
        channels_up=(16, 16, 16),
        channels_skip=(4, 4, 4),
    )


@pytest.fixture
def module_config() -> DipModelModuleConfig:
    return DipModelModuleConfig(learning_rate=1e-2)


@pytest.fixture
def image() -> PIL.Image.Image:
    # a deterministic synthetic RGB image as the reconstruction target
    generator = numpy.random.default_rng(seed=0)
    array = (generator.random(size=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)) * 255).astype('uint8')
    return PIL.Image.fromarray(array, mode='RGB')


@pytest.fixture
def trainer(
    image: PIL.Image.Image,
    module_config: DipModelModuleConfig,
    model_config: DipModelConfig,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> DipTrainer:
    torch.manual_seed(0)

    # keep the optimization tiny and fast for the tests
    monkeypatch.setattr(DipTrainer, 'MAX_EPOCHS', 2)

    # redirect all artifacts into the pytest temporary folder (auto-cleaned by pytest)
    return DipTrainer(
        image=image,
        module_config=module_config,
        model_config=model_config,
        work_folder_path=tmp_path,
    )
    assert trainer is not None
    assert isinstance(trainer, DipTrainer)


def test_trainer_converts_image_to_tensor(trainer: DipTrainer) -> None:
    # the Pillow image must be converted to a normalized RGB tensor
    assert trainer.image_tensor.shape == torch.Size([3, IMAGE_HEIGHT, IMAGE_WIDTH])
    assert trainer.image_tensor.min().item() >= 0.0
    assert trainer.image_tensor.max().item() <= 1.0


def test_trainer_geometry_matches_image(trainer: DipTrainer) -> None:
    assert trainer.image_width == IMAGE_WIDTH
    assert trainer.image_height == IMAGE_HEIGHT


def test_trainer_builds_data_and_model_modules(trainer: DipTrainer) -> None:
    assert trainer.data_module.width == IMAGE_WIDTH
    assert trainer.data_module.height == IMAGE_HEIGHT
    assert trainer.model_module.image_width == IMAGE_WIDTH
    assert trainer.model_module.image_height == IMAGE_HEIGHT


def test_trainer_converts_non_rgb_image(
    module_config: DipModelModuleConfig,
    model_config: DipModelConfig,
    tmp_path: pathlib.Path,
) -> None:
    # a grayscale image must be converted to RGB internally without errors
    array = (numpy.zeros(shape=(IMAGE_HEIGHT, IMAGE_WIDTH), dtype='uint8'))
    grayscale = PIL.Image.fromarray(array, mode='L')

    built = DipTrainer(
        image=grayscale,
        module_config=module_config,
        model_config=model_config,
        work_folder_path=tmp_path,
    )

    assert built.image_tensor.shape == torch.Size([3, IMAGE_HEIGHT, IMAGE_WIDTH])


# ---------------------------------------------------------------------------
# image processing
# ---------------------------------------------------------------------------

def test_process_image_crops_to_divisible_size(
    module_config: DipModelModuleConfig,
    model_config: DipModelConfig,
    tmp_path: pathlib.Path,
) -> None:
    # the test network has 3 levels, so the size must become divisible by 2 ** 3 == 8
    raw_width, raw_height = 100, 70
    generator = numpy.random.default_rng(seed=0)
    array = (generator.random(size=(raw_height, raw_width, 3)) * 255).astype('uint8')
    raw_image = PIL.Image.fromarray(array, mode='RGB')

    built = DipTrainer(
        image=raw_image,
        module_config=module_config,
        model_config=model_config,
        work_folder_path=tmp_path,
    )

    # 100 -> 96 and 70 -> 64 (largest multiples of 8 not exceeding the original)
    assert built.image_width == 96
    assert built.image_height == 64
    assert built.image_width % 8 == 0
    assert built.image_height % 8 == 0


def test_process_image_downscales_large_image(
    module_config: DipModelModuleConfig,
    model_config: DipModelConfig,
    tmp_path: pathlib.Path,
) -> None:
    # an oversized image must be downscaled to fit inside the MAX_WIDTH x MAX_HEIGHT box
    raw_width, raw_height = 1920, 1080
    generator = numpy.random.default_rng(seed=0)
    array = (generator.random(size=(raw_height, raw_width, 3)) * 255).astype('uint8')
    raw_image = PIL.Image.fromarray(array, mode='RGB')

    built = DipTrainer(
        image=raw_image,
        module_config=module_config,
        model_config=model_config,
        work_folder_path=tmp_path,
    )

    # the result must fit within the limits and stay divisible by the resolution divisor
    assert built.image_width <= DipTrainer.MAX_WIDTH
    assert built.image_height <= DipTrainer.MAX_HEIGHT
    assert built.image_width % 8 == 0
    assert built.image_height % 8 == 0


def test_process_image_keeps_small_image_untouched(
    module_config: DipModelModuleConfig,
    model_config: DipModelConfig,
    tmp_path: pathlib.Path,
) -> None:
    # an image already within the limits and divisible by 8 must not be resized
    raw_width, raw_height = 128, 96
    generator = numpy.random.default_rng(seed=0)
    array = (generator.random(size=(raw_height, raw_width, 3)) * 255).astype('uint8')
    raw_image = PIL.Image.fromarray(array, mode='RGB')

    built = DipTrainer(
        image=raw_image,
        module_config=module_config,
        model_config=model_config,
        work_folder_path=tmp_path,
    )

    assert built.image_width == raw_width
    assert built.image_height == raw_height


def test_process_image_saves_processed_original(
    module_config: DipModelModuleConfig,
    model_config: DipModelConfig,
    tmp_path: pathlib.Path,
) -> None:
    generator = numpy.random.default_rng(seed=0)
    array = (generator.random(size=(70, 100, 3)) * 255).astype('uint8')
    raw_image = PIL.Image.fromarray(array, mode='RGB')

    built = DipTrainer(
        image=raw_image,
        module_config=module_config,
        model_config=model_config,
        work_folder_path=tmp_path,
    )

    # the processed original must be saved with a network-compatible size
    original_path = built.output_folder_path / 'original.png'
    assert original_path.is_file()

    with PIL.Image.open(original_path) as saved:
        assert saved.size == (built.image_width, built.image_height)


# ---------------------------------------------------------------------------
# end-to-end optimization
# ---------------------------------------------------------------------------

def test_train_saves_reconstructions(trainer: DipTrainer) -> None:
    output_folder = trainer.train()

    # one reconstruction image is saved per validation epoch
    saved_images = sorted(output_folder.glob('reconstruction-*.png'))
    assert len(saved_images) == trainer.MAX_EPOCHS


def test_train_saves_checkpoints(trainer: DipTrainer) -> None:
    trainer.train()

    # the checkpoint callback must persist at least the "last" checkpoint
    checkpoints = sorted(trainer.snapshot_folder_path.glob('*.ckpt'))
    assert len(checkpoints) > 0


def test_train_returns_output_folder(trainer: DipTrainer) -> None:
    output_folder = trainer.train()

    assert output_folder == trainer.output_folder_path
    assert output_folder.is_dir()
