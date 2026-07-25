import dataclasses
import json
import logging
import pathlib
import typing as t

import torch
import torch.nn
import torch.optim
import torch.utils.data

import torchvision.utils

import lightning.pytorch as pl

from ml.model import DipModelConfig, DipModel


@dataclasses.dataclass(frozen=True)
class DipModelModuleConfig:
    """Configuration for the Deep Image Prior Lightning module."""

    # learning rate for the Adam optimizer
    learning_rate: float = dataclasses.field(
        default=1e-2,
        metadata={'help': 'Learning rate for the Adam optimizer'},
    )

    def save(self, file_path: pathlib.Path):
        """
        serialize the configuration to a JSON file.

        :param file_path: destination file
        """
        with file_path.open(mode='wt') as file:
            data_dict = dataclasses.asdict(self)
            json.dump(data_dict, file, sort_keys=True, indent=4, default=str)

    @classmethod
    def load(cls, file_path: pathlib.Path) -> t.Self:
        """
        load the configuration from a JSON file.

        :param file_path: source file
        :return: the loaded configuration
        """
        with file_path.open(mode='rt') as file:
            data_dict = json.load(file)
            return cls(**data_dict)


class DipModelModule(pl.LightningModule):
    """
    PyTorch Lightning module implementing the Deep Image Prior optimization loop.

    Deep Image Prior does not learn from a dataset: it optimizes the weights of a randomly-initialized
    network so that, when fed a fixed noise tensor, the network reproduces one single target image. The
    target image is therefore supplied to the module and is used to compute the reconstruction loss. As
    the convolutional structure fits clean image content faster than noise, the intermediate outputs
    (saved after every validation epoch) gradually become a restored version of the target image.

    There is no mask in this architecture: the whole image is reconstructed and the original and the
    processed images are meant to be blended manually afterwards (for example as layers in GIMP).
    """

    # number of channels of an RGB image
    IMAGE_CHANNELS: t.Final[int] = 3

    def __init__(
        self,
        image: torch.Tensor,
        config: DipModelModuleConfig,
        model_config: DipModelConfig = DipModelConfig(),
        output_dir: pathlib.Path | None = None,
    ):
        """
        build the module and the underlying `DipModel` for the target image resolution.

        :param image: target image tensor of shape (channels, height, width) with pixel values in [0, 1]
        :param config: module configuration (optimization hyper-parameters)
        :param model_config: architecture configuration of the hourglass network
        :param output_dir: directory where intermediate reconstructions are saved (no saving if None)
        """
        super().__init__()

        self.logging: t.Final[logging.Logger] = logging.getLogger(self.__class__.__name__)

        # the target image drives the whole optimization, so it is stored as a non-trainable buffer
        # (a buffer is moved to the right device together with the module but is not optimized)
        channels, height, width = image.shape
        assert torch.Size([self.IMAGE_CHANNELS, height, width]) == image.shape

        self.register_buffer('target_image', image)

        # remember the geometry of the target image
        self.image_height: t.Final[int] = height
        self.image_width: t.Final[int] = width

        # configuration
        self.config: t.Final[DipModelModuleConfig] = config

        # directory to save the intermediate reconstructions after every validation epoch
        self.output_dir: t.Final[pathlib.Path | None] = output_dir

        # the hourglass network is created here, its resolution matches the target image
        self.model: t.Final[DipModel] = DipModel(
            width=width,
            height=height,
            config=model_config,
        )

        # pixel-wise mean squared error between the produced image and the target image
        self.loss_fn: t.Final[torch.nn.MSELoss] = torch.nn.MSELoss()

        # last validation image
        self.latest_output: torch.Tensor | None = None

    @t.override
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        configure the Adam optimizer over the network weights.

        :return: the configured optimizer
        """
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

    @t.override
    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        map the fixed noise tensor to an image using the hourglass network.

        :param noise: random noise input tensor of shape (input_channels, height, width)
        :return: produced image of shape (channels, height, width) with pixel values in [0, 1]
        """

        # noinspection PyCallingNonCallable
        return self.model(noise)

    @t.override
    def training_step(self, noise: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        perform one optimization step towards reproducing the target image.

        :param noise: fixed noise tensor served by the data module of shape (input_channels, height, width)
        :param batch_idx: index of the batch inside the current pseudo-epoch (ignored)
        :return: the reconstruction loss for this step
        """
        output: torch.Tensor = self.forward(noise)
        assert torch.Size([self.IMAGE_CHANNELS, self.image_height, self.image_width]) == output.shape

        loss: torch.Tensor = self.loss_fn(output, self.target_image)

        self.log('trn/loss', loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    @t.override
    def validation_step(self, noise: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        evaluate the current reconstruction against the target image.

        only the first item of the pseudo-epoch is used to log the loss and to keep the output image for
        saving, as every item of the dataset is the very same noise tensor.

        :param noise: fixed noise tensor served by the data module of shape (input_channels, height, width)
        :param batch_idx: index of the batch inside the current pseudo-epoch
        :return: the reconstruction loss for this step
        """
        output: torch.Tensor = self.forward(noise)
        assert torch.Size([self.IMAGE_CHANNELS, self.image_height, self.image_width]) == output.shape

        loss: torch.Tensor = self.loss_fn(output, self.target_image)

        self.log('val/loss', loss, prog_bar=True, on_step=False, on_epoch=True)

        # remember the latest reconstruction so it can be saved at the end of the validation epoch;
        # detached and copied to keep it independent from the autograd graph and the next steps
        if batch_idx == 0:
            self.latest_output = output.detach().clone()

        return loss

    @t.override
    def on_validation_epoch_end(self) -> None:
        """
        save the latest reconstruction to the output directory so it can be inspected later.
        """
        if self.output_dir is None:
            return

        # make sure the output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # name the file after the current epoch to keep the whole optimization history
        file_path: pathlib.Path = self.output_dir / f'reconstruction-{self.current_epoch:06d}.png'

        # save the reconstruction as a regular PNG image
        torchvision.utils.save_image(self.latest_output, file_path)

        self.logging.info('saved reconstruction: %s', file_path)
