import datetime
import logging
import pathlib
import typing as t

import PIL.Image

import torch

import torchvision.transforms.functional

import lightning.pytorch as pl
import lightning.pytorch.loggers as pl_log
import lightning.pytorch.callbacks as pl_callbacks

from ml.data import DipDataModule
from ml.model import DipModelConfig
from ml.module import DipModelModuleConfig, DipModelModule


class DipTrainer:
    """
    orchestrator for the Deep Image Prior optimization.

    The trainer converts a single target image into a tensor, builds the pseudo-dataset data module and
    the Lightning module, wires the TensorBoard logger and the callbacks, and drives the optimization via
    a PyTorch Lightning `Trainer`. Because Deep Image Prior fits one single image, "training" here means
    optimizing the network weights to reproduce that image; intermediate reconstructions are saved to disk
    after every validation epoch.
    """

    # number of channels of an RGB image
    IMAGE_CHANNELS: t.Final[int] = 3

    # metric monitored by the checkpoint callback (the reconstruction loss on the target image)
    MONITOR_METRIC: t.Final[str] = 'val/loss'

    # lower reconstruction loss is better
    MONITOR_MODE: t.Final[str] = 'min'

    # number of best checkpoints kept on disk
    CHECKPOINT_TOP_K: t.Final[int] = 3

    # number of optimization epochs (pseudo-epochs over the fixed noise tensor)
    MAX_EPOCHS: t.Final[int] = 1000

    # maximum wall-clock duration of the optimization
    MAX_DURATION: t.Final[datetime.timedelta] = datetime.timedelta(hours=1)

    # maximum width the target image is downscaled to, keeping the aspect ratio, to avoid GPU OOM
    MAX_WIDTH: t.Final[int] = 800

    # maximum height the target image is downscaled to, keeping the aspect ratio, to avoid GPU OOM
    MAX_HEIGHT: t.Final[int] = 600

    def __init__(
        self,
        image: PIL.Image.Image,
        mask: PIL.Image.Image,
        work_folder_path: pathlib.Path,
        module_config: DipModelModuleConfig = DipModelModuleConfig(),
        model_config: DipModelConfig = DipModelConfig(),
    ):
        """
        build the whole training stack for a single target image.

        the device placement is fully delegated to PyTorch Lightning: the target image becomes a buffer
        on the module and the noise seed dataset is allocated in the data module `setup()` on the device
        chosen by the trainer, so no device is threaded through here.

        :param image: target image to reconstruct (a Pillow image, converted to a tensor internally)
        :param mask: target mask to reconstruct (a Pillow image, converted to a tensor internally)
        :param work_folder_path: base folder for all artifacts of the run
        :param module_config: optimization hyper-parameters of the Lightning module
        :param model_config: architecture configuration of the hourglass network
        """
        self.logging: t.Final[logging.Logger] = logging.getLogger(self.__class__.__name__)

        self.module_config: t.Final[DipModelModuleConfig] = module_config
        self.model_config: t.Final[DipModelConfig] = model_config

        # ----------------------------------------------------------------------
        # folders
        # ----------------------------------------------------------------------

        # timestamp used to keep every run in its own sub-folder
        self.timestamp: t.Final[datetime.datetime] = datetime.datetime.now()

        # base folder for all the artifacts produced by this run
        self.work_folder_path: t.Final[pathlib.Path] = work_folder_path

        # folder where the intermediate reconstructions are saved
        self.output_folder_path: t.Final[pathlib.Path] = self.work_folder_path / 'reconstruction'
        self.output_folder_path.mkdir(parents=True, exist_ok=True)

        # folder for the model checkpoints
        self.snapshot_folder_path: t.Final[pathlib.Path] = self.work_folder_path / 'snapshot'
        self.snapshot_folder_path.mkdir(parents=True, exist_ok=True)

        # folder for the TensorBoard event files
        self.tensorboard_folder_path: t.Final[pathlib.Path] = self.work_folder_path / 'tensorboard'
        self.tensorboard_folder_path.mkdir(parents=True, exist_ok=True)

        # ----------------------------------------------------------------------
        # target image
        # ----------------------------------------------------------------------

        # crop the image so its size fits the network, then keep the processed original for reference
        self.image: t.Final[PIL.Image.Image] = self.process_image(image)

        # save the processed original next to the reconstructions for later comparison / blending
        processed_image_path: pathlib.Path = self.output_folder_path / 'original.png'
        self.image.save(processed_image_path)
        self.logging.info('saved processed original: %s', processed_image_path)

        # convert the processed Pillow image to an RGB tensor of shape (channels, height, width) in the
        # range [0, 1]; the tensor stays on the cpu and is moved to the accelerator by Lightning as a buffer
        self.image_tensor: t.Final[torch.Tensor] = torchvision.transforms.functional.to_tensor(self.image)

        channels, self.image_height, self.image_width = self.image_tensor.shape  # type: int, int, int
        assert torch.Size([self.IMAGE_CHANNELS, self.image_height, self.image_width]) == self.image_tensor.shape

        self.logging.info('target image: %d x %d (RGB)', self.image_width, self.image_height)

        # ----------------------------------------------------------------------
        # data module
        # ----------------------------------------------------------------------

        # the number of noise channels is taken from the model configuration so the data module and the
        # model always agree on the input channel count (the model config is the single source of truth)
        self.data_module: t.Final[DipDataModule] = DipDataModule(
            width=self.image_width,
            height=self.image_height,
            channels=self.model_config.input_channels,
        )

        # ----------------------------------------------------------------------
        # model module
        # ----------------------------------------------------------------------

        self.model_module: t.Final[DipModelModule] = DipModelModule(
            image=self.image_tensor,
            config=self.module_config,
            model_config=self.model_config,
            output_dir=self.output_folder_path,
        )

        # ----------------------------------------------------------------------
        # tensorboard
        # ----------------------------------------------------------------------

        self.tensorboard_logger: t.Final[pl_log.Logger] = pl_log.TensorBoardLogger(
            save_dir=self.tensorboard_folder_path.resolve(),
            name='',
            version=f'{self.timestamp:%Y%m%d-%H%M%S}',
            default_hp_metric=False,
        )

        # ----------------------------------------------------------------------
        # callbacks
        # ----------------------------------------------------------------------

        # track the learning rate over the optimization
        self.learning_rate_callback: t.Final[pl_callbacks.LearningRateMonitor] = \
            pl_callbacks.LearningRateMonitor(
                logging_interval='epoch',
            )

        # progress bar over the pseudo-epochs
        self.progress_callback: t.Final[pl_callbacks.TQDMProgressBar] = \
            pl_callbacks.TQDMProgressBar(
                leave=True,
            )

        # hard wall-clock limit for the optimization
        self.timer_callback: t.Final[pl_callbacks.Timer] = \
            pl_callbacks.Timer(
                duration=self.MAX_DURATION,
            )

        # keep the checkpoints with the lowest reconstruction loss
        self.checkpoint_callback: t.Final[pl_callbacks.ModelCheckpoint] = \
            pl_callbacks.ModelCheckpoint(
                dirpath=self.snapshot_folder_path,
                filename='best-{' + self.MONITOR_METRIC + ':.6f}-{epoch:03d}',
                monitor=self.MONITOR_METRIC,
                mode=self.MONITOR_MODE,
                save_top_k=self.CHECKPOINT_TOP_K,
                save_last=True,
                auto_insert_metric_name=False,
                verbose=True,
            )

        # ----------------------------------------------------------------------
        # trainer
        # ----------------------------------------------------------------------

        self.trainer: t.Final[pl.Trainer] = pl.Trainer(
            default_root_dir=self.work_folder_path,

            min_epochs=1,
            max_epochs=self.MAX_EPOCHS,

            num_sanity_val_steps=0,

            enable_model_summary=True,
            enable_progress_bar=True,
            enable_checkpointing=True,

            log_every_n_steps=1,

            accelerator='auto',
            devices='auto',
            precision='32-true',

            logger=[
                self.tensorboard_logger,
            ],

            callbacks=[
                self.learning_rate_callback,
                self.progress_callback,
                self.timer_callback,
                self.checkpoint_callback,
            ],
        )

    def process_image(self, image: PIL.Image.Image) -> PIL.Image.Image:
        """
        prepare the target image for the network and save the processed original for later reference.

        The method converts the image to RGB, then applies two geometric steps:
          1. downscaling (keeping the aspect ratio) so it fits inside `MAX_WIDTH` x `MAX_HEIGHT`, which
             bounds the memory footprint and avoids GPU out-of-memory errors on large inputs;
          2. center-cropping to the largest size whose width and height are both divisible by 2 ** depth,
             because the hourglass network halves the spatial resolution at every level.
        It is a pure function with no side effects.

        :param image: the raw target image
        :return: the processed RGB image with a network-compatible size
        """
        # the network halves the resolution once per level, so the size must be a multiple of 2 ** depth
        depth: int = len(self.model_config.channels_down)
        resolution_divisor: int = 2 ** depth

        # snap the configured limits down to a multiple of the resolution divisor, so the whole allowed
        # box is actually usable (the raw MAX_WIDTH / MAX_HEIGHT are not necessarily divisible by it)
        max_width: int = (self.MAX_WIDTH // resolution_divisor) * resolution_divisor
        max_height: int = (self.MAX_HEIGHT // resolution_divisor) * resolution_divisor
        assert max_width > 0, f'MAX_WIDTH {self.MAX_WIDTH} is smaller than {resolution_divisor}'
        assert max_height > 0, f'MAX_HEIGHT {self.MAX_HEIGHT} is smaller than {resolution_divisor}'

        # make sure the image is RGB before any geometric processing
        rgb_image: PIL.Image.Image = image.convert('RGB')

        # downscale (never upscale) so the image fits inside the maximum box while keeping the aspect ratio;
        # `scale` is the single factor that brings both dimensions within their limits at once
        scale: float = min(
            1.0,
            max_width / rgb_image.width,
            max_height / rgb_image.height,
        )

        scaled_image: PIL.Image.Image = rgb_image
        if scale < 1.0:
            scaled_width: int = int(rgb_image.width * scale)
            scaled_height: int = int(rgb_image.height * scale)
            scaled_image = rgb_image.resize(
                size=(scaled_width, scaled_height),
                resample=PIL.Image.Resampling.LANCZOS,
            )
            self.logging.info(
                'processed image: downscaled from %d x %d to %d x %d (max %d x %d)',
                rgb_image.width, rgb_image.height, scaled_width, scaled_height,
                self.MAX_WIDTH, self.MAX_HEIGHT,
            )

        # largest width and height that are still divisible by the resolution divisor
        cropped_width: int = (scaled_image.width // resolution_divisor) * resolution_divisor
        cropped_height: int = (scaled_image.height // resolution_divisor) * resolution_divisor
        assert cropped_width > 0, f'image width {scaled_image.width} is smaller than {resolution_divisor}'
        assert cropped_height > 0, f'image height {scaled_image.height} is smaller than {resolution_divisor}'

        # center-crop offsets so the kept region stays in the middle of the scaled image
        left: int = (scaled_image.width - cropped_width) // 2
        top: int = (scaled_image.height - cropped_height) // 2

        processed_image: PIL.Image.Image = scaled_image.crop((
            left,
            top,
            left + cropped_width,
            top + cropped_height,
        ))
        assert (cropped_width, cropped_height) == processed_image.size

        self.logging.info(
            'processed image: cropped from %d x %d to %d x %d (divisor %d)',
            scaled_image.width, scaled_image.height, cropped_width, cropped_height, resolution_divisor,
        )

        return processed_image

    def train(self) -> pathlib.Path:
        """
        run the Deep Image Prior optimization over the target image.

        :return: path to the folder holding the intermediate reconstructions of this run
        """
        self.logging.info('starting optimization, run folder: %s', self.work_folder_path)

        self.trainer.fit(
            model=self.model_module,
            datamodule=self.data_module,
        )

        self.logging.info('optimization finished, reconstructions: %s', self.output_folder_path)

        return self.output_folder_path
