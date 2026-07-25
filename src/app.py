import argh
import pathlib
import logging
import logging.config
import typing as t
import yaml
import sys
import json
import warnings

import PIL.Image

import torch
import torch.cuda
import torch.nn
import torch.profiler
import torch.multiprocessing
import torch.distributed

import ml.trainer


# noinspection DuplicatedCode,PyMethodMayBeStatic
class Application:
    """
    model builder application
    """

    PATH_APPLICATION: t.Final[pathlib.Path] = pathlib.Path(__file__)

    PATH_DIR_SOURCES: t.Final[pathlib.Path] = PATH_APPLICATION.parent.resolve()

    PATH_DIR_PACKAGE: t.Final[pathlib.Path] = PATH_DIR_SOURCES.parent.resolve()

    PATH_DIR_WORK: t.Final[pathlib.Path] = PATH_DIR_PACKAGE / 'work'

    def __init__(self):
        # initialize logging
        logging_config_path: pathlib.Path = self.PATH_DIR_SOURCES / 'app.yaml'
        logging_config = self.load_yaml(logging_config_path, yaml.SafeLoader)
        logging.config.dictConfig(logging_config)
        logging.info('using logging configuration [%s]', logging_config_path)

        # local logger
        self.logger = logging.getLogger('application')
        self.logger.info('command line       :\n%s', json.dumps(sys.argv[1:], default=str, indent=2, sort_keys=False))
        self.logger.info('torch start        : %s', torch.multiprocessing.get_start_method())

        # avoid the warning:
        # TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled.
        # Consider setting `torch.set_float32_matmul_precision('high')` for better performance.
        torch.set_float32_matmul_precision('high')

        # default device is CPU
        torch.set_default_device('cpu')

        # default dtype
        torch.set_default_dtype(torch.float32)

        # torch multi-threading setup
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        # determinism
        torch.use_deterministic_algorithms(mode=False)

        # avoid the CheckPoint warning
        warnings.filterwarnings(
            action='ignore',
            message=r'Checkpoint directory .* exists and is not empty\.',
            category=UserWarning,
        )
        warnings.filterwarnings(
            action='ignore',
            message=r'.* is set, but there is no last checkpoint available\. No checkpoint will be loaded\. .*',
            category=UserWarning,
        )

        # avoid the LitLogger warning
        warnings.filterwarnings(
            action='ignore',
            message=r'LitLogger does not support `log_graph`',
            category=UserWarning,
        )

    @argh.arg('image_path', help='path to the image file to reconstruct')
    @argh.arg('mask_path', help='path to the mask file to be used for reconstruction')
    def train(
        self,
        image_path: str,
        mask_path: str,
    ):
        # resolve and validate the image file path
        image_file_path: pathlib.Path = pathlib.Path(image_path).expanduser().resolve()
        assert image_file_path.is_file(), f'the image file {image_file_path} does not exist'
        self.logger.info('loading image      : %s', image_file_path)

        # resolve and validate the image file path
        mask_file_path: pathlib.Path = pathlib.Path(mask_path).expanduser().resolve()
        assert mask_file_path.is_file(), f'the image file {mask_file_path} does not exist'
        self.logger.info('loading mask       : %s', mask_file_path)

        # load the target image to reconstruct; the trainer converts it to an RGB tensor internally
        with PIL.Image.open(image_file_path) as image_file:
            image: PIL.Image.Image = image_file.convert('RGB')

        # load the mask
        with PIL.Image.open(mask_file_path) as image_file:
            mask: PIL.Image.Image = image_file.convert('1')

        trainer: ml.trainer.DipTrainer = ml.trainer.DipTrainer(
            image=image,
            mask=mask,
            work_folder_path=self.PATH_DIR_WORK,
        )

        trainer.train()

    @staticmethod
    def load_yaml(path: pathlib.Path, yaml_loader_class: t.Type) -> t.Dict:
        with path.open('rt') as file:
            yaml_text = file.read()

        # noinspection PyTypeChecker
        yaml_dict = yaml.load(yaml_text, yaml_loader_class)

        return yaml_dict


if __name__ == '__main__':
    application = Application()

    parser = argh.ArghParser()
    argh.add_commands(parser, [application.train])

    try:
        argh.dispatch(parser)
    finally:
        logging.info('the work is finished')
        logging.shutdown()
