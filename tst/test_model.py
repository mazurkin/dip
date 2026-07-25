import pytest

import torch

from ml.model import DipModelConfig, DipModel


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
def config() -> DipModelConfig:
    # a shallow 3-level network keeps the tests small and fast on the CPU
    return DipModelConfig(
        input_channels=INPUT_CHANNELS,
        output_channels=OUTPUT_CHANNELS,
        channels_down=(16, 16, 16),
        channels_up=(16, 16, 16),
        channels_skip=(4, 4, 4),
    )


@pytest.fixture
def model(config: DipModelConfig) -> DipModel:
    torch.manual_seed(0)
    model = DipModel(width=IMAGE_WIDTH, height=IMAGE_HEIGHT, config=config)
    model.eval()
    return model


@pytest.fixture
def noise() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(size=(INPUT_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), device=DEVICE)


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------

def test_model_builds(model: DipModel) -> None:
    assert model is not None
    assert isinstance(model, DipModel)


def test_model_depth_matches_config(model: DipModel, config: DipModelConfig) -> None:
    assert model.depth == len(config.channels_down)
    assert len(model.down_blocks) == model.depth
    assert len(model.skip_blocks) == model.depth
    assert len(model.up_blocks) == model.depth


def test_model_rejects_indivisible_resolution(config: DipModelConfig) -> None:
    # a 3-level network requires both dimensions to be divisible by 2 ** 3 == 8
    with pytest.raises(AssertionError):
        DipModel(width=70, height=64, config=config)


def test_model_has_trainable_parameters(model: DipModel) -> None:
    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
    assert trainable > 0


# ---------------------------------------------------------------------------
# forward pass
# ---------------------------------------------------------------------------

def test_forward_unbatched_shape(model: DipModel, noise: torch.Tensor) -> None:
    with torch.no_grad():
        output = model(noise)

    assert output.shape == torch.Size([OUTPUT_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH])


def test_forward_batched_shape(model: DipModel, noise: torch.Tensor) -> None:
    batch_size = 2
    batched = noise.unsqueeze(0).expand(batch_size, INPUT_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)

    with torch.no_grad():
        output = model(batched)

    assert output.shape == torch.Size([batch_size, OUTPUT_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH])


def test_forward_output_range(model: DipModel, noise: torch.Tensor) -> None:
    # the final sigmoid must constrain every pixel to the [0, 1] range
    with torch.no_grad():
        output = model(noise)

    assert output.min().item() >= 0.0
    assert output.max().item() <= 1.0


def test_forward_no_nan(model: DipModel, noise: torch.Tensor) -> None:
    with torch.no_grad():
        output = model(noise)

    assert not torch.isnan(output).any()


def test_forward_deterministic_in_eval(model: DipModel, noise: torch.Tensor) -> None:
    # with the same weights and the same input the network must be perfectly reproducible
    with torch.no_grad():
        first = model(noise)
        second = model(noise)

    assert torch.equal(first, second)


# ---------------------------------------------------------------------------
# optimization behaviour
# ---------------------------------------------------------------------------

def test_single_step_reduces_loss(config: DipModelConfig, noise: torch.Tensor) -> None:
    # a few optimization steps must drive the output towards an arbitrary target image,
    # which is the core mechanism of the Deep Image Prior approach
    torch.manual_seed(0)
    model = DipModel(width=IMAGE_WIDTH, height=IMAGE_HEIGHT, config=config)
    model.train()

    target = torch.rand(size=(OUTPUT_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), device=DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss()

    initial_loss: float = loss_fn(model(noise), target).item()

    number_of_steps: int = 5
    final_loss: float = initial_loss
    for _ in range(number_of_steps):
        optimizer.zero_grad()
        loss = loss_fn(model(noise), target)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    assert final_loss < initial_loss


def test_backward_populates_gradients(config: DipModelConfig, noise: torch.Tensor) -> None:
    torch.manual_seed(0)
    model = DipModel(width=IMAGE_WIDTH, height=IMAGE_HEIGHT, config=config)
    model.train()

    target = torch.rand(size=(OUTPUT_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), device=DEVICE)

    loss = torch.nn.MSELoss()(model(noise), target)
    loss.backward()

    # every trainable parameter must receive a gradient
    for parameter in model.parameters():
        assert parameter.grad is not None
