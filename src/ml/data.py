import logging
import typing as t

import torch
import torch.utils
import torch.utils.data

from lightning import pytorch as pl


# noinspection PyOverrides
class DipDataset(torch.utils.data.Dataset):
    """
    pseudo-dataset for the Deep Image Prior approach.

    Deep Image Prior does not train on a collection of samples: it optimizes a network to reproduce
    a single target image starting from a fixed random noise tensor. This dataset therefore allocates
    the noise seed tensor once in the constructor and simply returns the very same tensor `n` times,
    which lets a standard PyTorch Lightning training loop iterate `n` steps per pseudo-epoch.
    """

    # default seed used to make the noise tensor reproducible across runs
    DEFAULT_SEED: t.Final[int] = 42

    # amplitude of the input noise; the Deep Image Prior paper feeds low-amplitude noise (the raw uniform
    # noise is scaled by this factor) for a gentler and more stable start of the optimization
    NOISE_SCALE: t.Final[float] = 0.1

    def __init__(
        self,
        width: int,
        height: int,
        channels: int = 32,
        n: int = 32,
        device: torch.device = torch.device('cpu'),
        seed: int = DEFAULT_SEED,
    ):
        """
        allocate the fixed noise seed tensor once and remember how many times it should be served.

        :param width: width of the noise image in pixels
        :param height: height of the noise image in pixels
        :param channels: number of channels of the noise tensor (default is 32)
        :param n: pseudo-size of the dataset, the number of iterations per pseudo-epoch (default is 32)
        :param device: device on which the noise tensor is allocated (default is `torch.device('cpu')`)
        :param seed: random seed making the noise tensor reproducible across runs (default is 42)
        """
        super().__init__()

        # geometry of the noise seed tensor, all immutable for the whole dataset lifetime
        self.width: t.Final[int] = width
        self.height: t.Final[int] = height
        self.channels: t.Final[int] = channels

        # pseudo-size of the dataset: how many times the same noise tensor is returned
        self.n: t.Final[int] = n

        # device on which the noise tensor lives
        self.device: t.Final[torch.device] = device

        # seed making the generated noise tensor reproducible
        self.seed: t.Final[int] = seed

        # dedicated generator seeded on the target device, so the noise does not depend on and does not
        # disturb the global random state and stays identical between runs with the same seed
        generator: torch.Generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed)

        # the fixed random noise seed, allocated once and reused for every item of the dataset;
        # the tensor is generated directly on the target device (see the `device` argument below) so it
        # never has to be copied from the host to the accelerator later during training;
        # the raw uniform noise is scaled down by `NOISE_SCALE` following the Deep Image Prior paper
        self.noise: t.Final[torch.Tensor] = self.NOISE_SCALE * torch.rand(
            size=(self.channels, self.height, self.width),
            generator=generator,
            device=self.device,
        )
        assert torch.Size([self.channels, self.height, self.width]) == self.noise.shape

    @t.override
    def __len__(self) -> int:
        """
        return the pseudo-size of the dataset.

        :return: the number of items (iterations) in one pseudo-epoch
        """
        return self.n

    @t.override
    def __getitem__(self, index: int) -> torch.Tensor:
        """
        return the fixed noise seed tensor regardless of the requested index.

        :param index: position of the requested item, ignored because every item is identical
        :return: the fixed random noise seed tensor of shape (channels, height, width)
        """
        return self.noise


class DipDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for the Deep Image Prior approach.

    The module wraps a single `DipDataset` that always serves the very same fixed noise seed tensor.
    Because the dataset item is just a pre-allocated tensor, no worker processes, shuffling or batching
    are required: every data loader yields the noise tensor one item at a time.
    """

    # pseudo-dataset size, the number of iterations per pseudo-epoch
    BATCH_SIZE: t.Final[int] = 32

    def __init__(
        self,
        width: int,
        height: int,
        channels: int,
    ):
        """
        remember the geometry of the noise seed tensor.

        the pseudo-dataset itself is not created here: the noise tensor is allocated later in `setup()`,
        once PyTorch Lightning has chosen the accelerator and the target device is known.

        :param width: width of the noise image in pixels
        :param height: height of the noise image in pixels
        :param channels: number of channels of the noise tensor, must match the model input channels
        """
        super().__init__()

        self.logging: t.Final[logging.Logger] = logging.getLogger(self.__class__.__name__)

        # geometry of the noise seed tensor
        self.width: t.Final[int] = width
        self.height: t.Final[int] = height

        # number of channels of the noise seed tensor
        self.channels: t.Final[int] = channels

        # the pseudo-dataset is created in `setup()` when the accelerator device is available
        self.dataset: DipDataset | None = None

    @t.override
    def setup(self, stage: str | None = None) -> None:
        """
        allocate the pseudo-dataset directly on the accelerator device chosen by Lightning.

        Lightning calls `setup()` after the accelerator is initialized and after `self.trainer` has been
        attached, so the real device is known here. Allocating the fixed noise tensor straight on that
        device avoids a per-step host-to-device copy of the very same tensor.

        :param stage: the current stage ('fit', 'validate', 'test', ...), ignored as the dataset is shared
        """
        # the pseudo-dataset is shared across all stages, so build it only once
        if self.dataset is not None:
            return

        # the device Lightning has chosen for this run (for example cpu or cuda:0)
        device: torch.device = self.trainer.strategy.root_device
        self.logging.info('allocating the noise seed dataset on device: %s', device)

        self.dataset = DipDataset(
            width=self.width,
            height=self.height,
            channels=self.channels,
            n=self.BATCH_SIZE,
            device=device,
        )

    @staticmethod
    def collate(batch: list[torch.Tensor]) -> torch.Tensor:
        """
        collapse a one-item batch into the single tensor to avoid a useless leading batch dimension.

        :param batch: list with exactly one noise tensor produced by the data loader
        :return: the single noise tensor from the batch
        """
        # the data loader is always built with batch_size=1, so exactly one item must arrive here
        assert len(batch) == 1, f'expected a single-item batch, got {len(batch)} items'

        return batch[0]

    def build_dataloader(self) -> torch.utils.data.DataLoader:
        """
        build a data loader over the pseudo-dataset with all parameters fixed for the DIP approach.

        no worker processes are used (the dataset just returns a pre-allocated tensor), there is no
        shuffling and the batch size is 1, while the `collate_fn` unwraps the single-item batch.

        :return: the configured data loader over the fixed noise seed tensor
        """
        assert self.dataset is not None, 'call setup() before requesting a data loader'

        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=self.collate,
        )

    @t.override
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        return the training data loader over the fixed noise seed tensor.

        :return: the training data loader
        """
        return self.build_dataloader()

    @t.override
    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        return the validation data loader over the fixed noise seed tensor.

        :return: the validation data loader
        """
        return self.build_dataloader()
