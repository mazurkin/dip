import pytest

import torch

import lightning.pytorch as pl

from ml.data import DipDataset, DipDataModule


# all tests run on the CPU device as required by the project conventions
DEVICE: torch.device = torch.device('cpu')

# geometry used across the tests
IMAGE_WIDTH: int = 64

IMAGE_HEIGHT: int = 48

# number of channels of the noise seed tensor
CHANNELS: int = 8

# pseudo-size of the dataset
DATASET_SIZE: int = 5


# ---------------------------------------------------------------------------
# DipDataset
# ---------------------------------------------------------------------------

@pytest.fixture
def dataset() -> DipDataset:
    torch.manual_seed(0)
    return DipDataset(
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        channels=CHANNELS,
        n=DATASET_SIZE,
        device=DEVICE,
    )


def test_dataset_length(dataset: DipDataset) -> None:
    assert len(dataset) == DATASET_SIZE


def test_dataset_item_shape(dataset: DipDataset) -> None:
    item = dataset[0]
    assert item.shape == torch.Size([CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH])


def test_dataset_item_device(dataset: DipDataset) -> None:
    assert dataset[0].device.type == DEVICE.type


def test_dataset_returns_same_tensor(dataset: DipDataset) -> None:
    # every index must return the very same fixed noise tensor object
    first = dataset[0]
    for index in range(len(dataset)):
        assert dataset[index] is first


def test_dataset_same_seed_is_reproducible() -> None:
    # two datasets built with the same seed must hold the identical noise tensor
    first = DipDataset(width=IMAGE_WIDTH, height=IMAGE_HEIGHT, channels=CHANNELS, n=DATASET_SIZE, seed=123)
    second = DipDataset(width=IMAGE_WIDTH, height=IMAGE_HEIGHT, channels=CHANNELS, n=DATASET_SIZE, seed=123)
    assert torch.equal(first.noise, second.noise)


def test_dataset_different_seed_differs() -> None:
    # different seeds must produce different noise tensors
    first = DipDataset(width=IMAGE_WIDTH, height=IMAGE_HEIGHT, channels=CHANNELS, n=DATASET_SIZE, seed=1)
    second = DipDataset(width=IMAGE_WIDTH, height=IMAGE_HEIGHT, channels=CHANNELS, n=DATASET_SIZE, seed=2)
    assert not torch.equal(first.noise, second.noise)


def test_dataset_does_not_disturb_global_state() -> None:
    # the dedicated generator must leave the global random state untouched
    torch.manual_seed(0)
    expected = torch.rand(4)

    torch.manual_seed(0)
    DipDataset(width=IMAGE_WIDTH, height=IMAGE_HEIGHT, channels=CHANNELS, n=DATASET_SIZE, seed=7)
    actual = torch.rand(4)

    assert torch.equal(expected, actual)


def test_dataset_default_parameters() -> None:
    # the default channels and pseudo-size must match the documented defaults
    default_dataset = DipDataset(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
    assert len(default_dataset) == 32
    assert default_dataset[0].shape == torch.Size([32, IMAGE_HEIGHT, IMAGE_WIDTH])


# ---------------------------------------------------------------------------
# DipDataModule
# ---------------------------------------------------------------------------

@pytest.fixture
def data_module() -> DipDataModule:
    return DipDataModule(width=IMAGE_WIDTH, height=IMAGE_HEIGHT, channels=CHANNELS)


@pytest.fixture
def setup_data_module(data_module: DipDataModule) -> DipDataModule:
    # the data module reads the device from the attached trainer during setup(), so a lightweight
    # cpu trainer is attached to provide the required `trainer.strategy.root_device`
    trainer = pl.Trainer(accelerator='cpu', devices=1, logger=False)
    data_module.trainer = trainer
    data_module.setup(stage='fit')
    return data_module


def test_data_module_builds(data_module: DipDataModule) -> None:
    assert data_module is not None
    # the dataset is not created until setup() is called
    assert data_module.dataset is None


def test_data_module_geometry(data_module: DipDataModule) -> None:
    assert data_module.width == IMAGE_WIDTH
    assert data_module.height == IMAGE_HEIGHT
    assert data_module.channels == CHANNELS


def test_data_module_dataloader_before_setup_fails(data_module: DipDataModule) -> None:
    # requesting a data loader before setup() must fail clearly
    with pytest.raises(AssertionError):
        data_module.build_dataloader()


def test_data_module_setup_creates_dataset(setup_data_module: DipDataModule) -> None:
    assert setup_data_module.dataset is not None
    assert isinstance(setup_data_module.dataset, DipDataset)
    assert len(setup_data_module.dataset) == DipDataModule.BATCH_SIZE
    assert setup_data_module.dataset[0].shape == torch.Size([
        CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH,
    ])


def test_data_module_setup_is_idempotent(setup_data_module: DipDataModule) -> None:
    # calling setup() a second time must keep the same dataset instance
    first_dataset = setup_data_module.dataset
    setup_data_module.setup(stage='validate')
    assert setup_data_module.dataset is first_dataset


def test_data_module_collate_unwraps_single_item() -> None:
    tensor = torch.rand(size=(CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), device=DEVICE)
    # the collate function must return the single tensor without the extra leading batch dimension
    assert DipDataModule.collate([tensor]) is tensor


def test_train_dataloader_yields_unbatched_noise(setup_data_module: DipDataModule) -> None:
    loader = setup_data_module.train_dataloader()

    items = list(loader)
    assert len(items) == DipDataModule.BATCH_SIZE

    # each item is the fixed noise tensor without a leading batch dimension
    first = items[0]
    assert first.shape == torch.Size([CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH])


def test_val_dataloader_yields_unbatched_noise(setup_data_module: DipDataModule) -> None:
    loader = setup_data_module.val_dataloader()

    items = list(loader)
    assert len(items) == DipDataModule.BATCH_SIZE

    first = items[0]
    assert first.shape == torch.Size([CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH])
