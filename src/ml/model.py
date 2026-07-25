import dataclasses
import logging
import typing as t

import torch
import torch.nn


@dataclasses.dataclass(frozen=True)
class DipModelConfig:
    """
    configuration of the Deep Image Prior hourglass ("skip") network.

    The architecture is an encoder-decoder (a U-Net style "hourglass") with optional skip connections.
    The paper (Ulyanov et al., "Deep Image Prior") shows that such a convolutional structure, when fed a
    fixed random noise tensor, naturally reconstructs clean image content much earlier than it fits the
    noise. The depth of the network equals the number of entries in the channel tuples below: every level
    halves the spatial resolution on the way down and doubles it on the way up.
    """

    # number of channels of the random noise input tensor (must match the data module `CHANNEL_COUNT`)
    input_channels: int = dataclasses.field(default=32)

    # number of channels of the produced image (3 for an RGB image)
    output_channels: int = dataclasses.field(default=3)

    # number of feature channels produced by each downsampling (encoder) level
    channels_down: tuple[int, ...] = dataclasses.field(default=(128, 128, 128, 128, 128))

    # number of feature channels produced by each upsampling (decoder) level
    channels_up: tuple[int, ...] = dataclasses.field(default=(128, 128, 128, 128, 128))

    # number of channels carried by the skip connection at each level (0 disables the skip at that level)
    channels_skip: tuple[int, ...] = dataclasses.field(default=(4, 4, 4, 4, 4))


class DipDownBlock(torch.nn.Module):
    """
    single encoder (downsampling) level of the hourglass network.

    The first convolution uses `stride=2` to halve the spatial resolution, the second convolution keeps
    the resolution. Both convolutions are followed by batch normalization and a LeakyReLU non-linearity,
    which is the exact block layout used in the reference Deep Image Prior implementation. Reflection
    padding is used to reduce boundary artifacts in the reconstructed image.
    """

    # kernel size of the convolutions inside the block
    KERNEL_SIZE: t.Final[int] = 3

    # padding required to keep the spatial size for a kernel of size `KERNEL_SIZE`
    PADDING: t.Final[int] = 1

    # stride that halves the spatial resolution
    STRIDE_DOWN: t.Final[int] = 2

    # stride that keeps the spatial resolution
    STRIDE_KEEP: t.Final[int] = 1

    # negative slope of the LeakyReLU activation
    LEAKY_RELU_SLOPE: t.Final[float] = 0.1

    def __init__(self, in_channels: int, out_channels: int):
        """
        build one encoder level.

        :param in_channels: number of channels of the input feature map
        :param out_channels: number of channels produced by this level
        """
        super().__init__()

        self.in_channels: t.Final[int] = in_channels
        self.out_channels: t.Final[int] = out_channels

        # the whole block is a plain sequential stack of convolutions, normalizations and activations
        self.block: t.Final[torch.nn.Module] = torch.nn.Sequential(
            # downsampling convolution: halves height and width
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.KERNEL_SIZE,
                stride=self.STRIDE_DOWN,
                padding=self.PADDING,
                padding_mode='reflect',
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(self.LEAKY_RELU_SLOPE),
            # refinement convolution: keeps height and width
            torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=self.KERNEL_SIZE,
                stride=self.STRIDE_KEEP,
                padding=self.PADDING,
                padding_mode='reflect',
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(self.LEAKY_RELU_SLOPE),
        )

    @t.override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        downsample and encode the input feature map.

        :param x: input feature map of shape (batch, in_channels, height, width)
        :return: encoded feature map of shape (batch, out_channels, height / 2, width / 2)
        """
        batch, _, height, width = x.shape
        assert torch.Size([batch, self.in_channels, height, width]) == x.shape

        result: torch.Tensor = self.block(x)
        assert torch.Size([batch, self.out_channels, height // 2, width // 2]) == result.shape

        return result


class DipSkipBlock(torch.nn.Module):
    """
    skip connection block of the hourglass network.

    A skip connection copies fine-grained detail from an encoder level directly to the matching decoder
    level, bypassing the bottleneck. Deep Image Prior uses only a very thin (few channels) 1x1 convolution
    here on purpose: a narrow skip lets some structure pass through while still forcing most information
    through the low-dimensional bottleneck, which is what regularizes the reconstruction.
    """

    # 1x1 convolution keeps the spatial size and just mixes the channels
    KERNEL_SIZE: t.Final[int] = 1

    # negative slope of the LeakyReLU activation
    LEAKY_RELU_SLOPE: t.Final[float] = 0.1

    def __init__(self, in_channels: int, skip_channels: int):
        """
        build one skip connection.

        :param in_channels: number of channels of the encoder feature map feeding the skip
        :param skip_channels: number of channels carried across the skip connection
        """
        super().__init__()

        self.in_channels: t.Final[int] = in_channels
        self.skip_channels: t.Final[int] = skip_channels

        # thin 1x1 convolution followed by normalization and activation
        self.block: t.Final[torch.nn.Module] = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=skip_channels,
                kernel_size=self.KERNEL_SIZE,
            ),
            torch.nn.BatchNorm2d(skip_channels),
            torch.nn.LeakyReLU(self.LEAKY_RELU_SLOPE),
        )

    @t.override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        project the encoder feature map to the thin skip representation.

        :param x: encoder feature map of shape (batch, in_channels, height, width)
        :return: skip feature map of shape (batch, skip_channels, height, width)
        """
        batch, _, height, width = x.shape
        assert torch.Size([batch, self.in_channels, height, width]) == x.shape

        result: torch.Tensor = self.block(x)
        assert torch.Size([batch, self.skip_channels, height, width]) == result.shape

        return result


class DipUpBlock(torch.nn.Module):
    """
    single decoder (upsampling) level of the hourglass network.

    The deeper (lower-resolution) feature map is first upsampled to the current resolution, then it is
    concatenated with the incoming skip connection. A refinement 3x3 convolution and a 1x1 convolution
    (each followed by batch normalization and LeakyReLU) mix the skip detail with the upsampled content.
    Bilinear upsampling is used instead of transposed convolutions because it avoids the checkerboard
    artifacts that transposed convolutions tend to introduce.
    """

    # kernel size of the refinement 3x3 convolution
    KERNEL_SIZE_MAIN: t.Final[int] = 3

    # padding to keep the spatial size for the 3x3 convolution
    PADDING_MAIN: t.Final[int] = 1

    # kernel size of the channel-mixing 1x1 convolution
    KERNEL_SIZE_MIX: t.Final[int] = 1

    # spatial magnification factor of the upsampling step
    UPSAMPLE_SCALE: t.Final[int] = 2

    # negative slope of the LeakyReLU activation
    LEAKY_RELU_SLOPE: t.Final[float] = 0.1

    def __init__(self, deeper_channels: int, skip_channels: int, out_channels: int):
        """
        build one decoder level.

        :param deeper_channels: number of channels of the (lower-resolution) feature map coming from below
        :param skip_channels: number of channels of the skip connection at this level
        :param out_channels: number of channels produced by this level
        """
        super().__init__()

        self.deeper_channels: t.Final[int] = deeper_channels
        self.skip_channels: t.Final[int] = skip_channels
        self.out_channels: t.Final[int] = out_channels

        # number of channels after concatenating the upsampled deeper features with the skip features
        self.concat_channels: t.Final[int] = deeper_channels + skip_channels

        # bilinear upsampling that doubles the spatial resolution of the deeper feature map
        self.upsample: t.Final[torch.nn.Module] = torch.nn.Upsample(
            scale_factor=self.UPSAMPLE_SCALE,
            mode='bilinear',
            align_corners=False,
        )

        # normalization applied right after concatenation to balance skip and deeper statistics
        self.norm_concat: t.Final[torch.nn.Module] = torch.nn.BatchNorm2d(self.concat_channels)

        # refinement stack: 3x3 convolution to fuse the features, then a 1x1 convolution to mix channels
        self.block: t.Final[torch.nn.Module] = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=self.concat_channels,
                out_channels=out_channels,
                kernel_size=self.KERNEL_SIZE_MAIN,
                padding=self.PADDING_MAIN,
                padding_mode='reflect',
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(self.LEAKY_RELU_SLOPE),
            torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=self.KERNEL_SIZE_MIX,
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(self.LEAKY_RELU_SLOPE),
        )

    @t.override
    def forward(self, deeper: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        upsample the deeper features, fuse them with the skip features and refine the result.

        :param deeper: lower-resolution feature map of shape (batch, deeper_channels, height, width)
        :param skip: skip feature map of shape (batch, skip_channels, 2 * height, 2 * width)
        :return: refined feature map of shape (batch, out_channels, 2 * height, 2 * width)
        """
        batch, _, height, width = deeper.shape
        assert torch.Size([batch, self.deeper_channels, height, width]) == deeper.shape

        target_height: int = height * self.UPSAMPLE_SCALE
        target_width: int = width * self.UPSAMPLE_SCALE
        assert torch.Size([batch, self.skip_channels, target_height, target_width]) == skip.shape

        # bring the deeper feature map up to the current resolution
        upsampled: torch.Tensor = self.upsample(deeper)
        assert torch.Size([batch, self.deeper_channels, target_height, target_width]) == upsampled.shape

        # concatenate skip detail with the upsampled content along the channel dimension
        merged: torch.Tensor = torch.cat([upsampled, skip], dim=1)
        assert torch.Size([batch, self.concat_channels, target_height, target_width]) == merged.shape

        # normalize the concatenation, then refine it
        merged = self.norm_concat(merged)
        assert torch.Size([batch, self.concat_channels, target_height, target_width]) == merged.shape

        result: torch.Tensor = self.block(merged)
        assert torch.Size([batch, self.out_channels, target_height, target_width]) == result.shape

        return result


class DipModel(torch.nn.Module):
    """
    Deep Image Prior hourglass network.

    The network maps a fixed random noise tensor to an image. It is never trained on a dataset: instead its
    weights are optimized so that its single output matches one target (degraded) image. Because the
    convolutional structure fits smooth, natural image content faster than it fits noise, stopping the
    optimization early yields a restored image - the network architecture itself acts as the "prior".

    Structure (for a network of `depth` levels):
      * `depth` encoder levels, each halving the spatial resolution;
      * `depth` skip connections, one from every encoder level to the matching decoder level;
      * `depth` decoder levels, each doubling the spatial resolution;
      * a final 1x1 convolution plus a sigmoid that produces an image with values in the range [0, 1].

    Because every level halves the resolution, the input width and height must both be divisible by
    2 ** depth; this is validated in the constructor.
    """

    # kernel size of the final projection convolution to the output image
    OUTPUT_KERNEL_SIZE: t.Final[int] = 1

    # number of spatial dimensions of a plain (unbatched) image tensor: channels, height, width
    UNBATCHED_DIMENSIONS: t.Final[int] = 3

    def __init__(self, width: int, height: int, config: DipModelConfig = DipModelConfig()):
        """
        build the hourglass network for a fixed image resolution.

        :param width: width of the produced image in pixels
        :param height: height of the produced image in pixels
        :param config: architecture configuration (channel widths and depth)
        """
        super().__init__()

        self.logging: t.Final[logging.Logger] = logging.getLogger(self.__class__.__name__)

        self.width: t.Final[int] = width
        self.height: t.Final[int] = height
        self.config: t.Final[DipModelConfig] = config

        # the depth of the network is the number of encoder / decoder levels
        self.depth: t.Final[int] = len(config.channels_down)
        assert self.depth == len(config.channels_up)
        assert self.depth == len(config.channels_skip)

        # every level halves the resolution, so both dimensions must be divisible by 2 ** depth
        resolution_divisor: int = 2 ** self.depth
        assert height % resolution_divisor == 0, f'height {height} must be divisible by {resolution_divisor}'
        assert width % resolution_divisor == 0, f'width {width} must be divisible by {resolution_divisor}'

        # build the encoder, skip and decoder levels as parallel module lists indexed by level
        self.down_blocks: t.Final[torch.nn.ModuleList] = torch.nn.ModuleList()
        self.skip_blocks: t.Final[torch.nn.ModuleList] = torch.nn.ModuleList()
        self.up_blocks: t.Final[torch.nn.ModuleList] = torch.nn.ModuleList()

        # number of channels entering the current encoder level (starts with the noise input channels)
        level_in_channels: int = config.input_channels

        for level in range(self.depth):
            # encoder level: reduces resolution and expands to `channels_down[level]`
            self.down_blocks.append(
                DipDownBlock(
                    in_channels=level_in_channels,
                    out_channels=config.channels_down[level],
                )
            )

            # skip connection: taken from the feature map that enters this encoder level
            self.skip_blocks.append(
                DipSkipBlock(
                    in_channels=level_in_channels,
                    skip_channels=config.channels_skip[level],
                )
            )

            # the next encoder level consumes the channels produced by this one
            level_in_channels = config.channels_down[level]

        # decoder levels, built from the bottleneck upwards (index `level` matches the encoder level)
        for level in range(self.depth):
            # channels of the deeper feature map feeding this decoder level:
            #   - the deepest decoder level consumes the bottleneck (last encoder output)
            #   - every shallower decoder level consumes the output of the decoder level below it
            deeper_channels: int = (
                config.channels_down[self.depth - 1]
                if level == self.depth - 1
                else config.channels_up[level + 1]
            )

            self.up_blocks.append(
                DipUpBlock(
                    deeper_channels=deeper_channels,
                    skip_channels=config.channels_skip[level],
                    out_channels=config.channels_up[level],
                )
            )

        # final projection to the output image followed by a sigmoid to constrain pixels to [0, 1]
        self.output_block: t.Final[torch.nn.Module] = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=config.channels_up[0],
                out_channels=config.output_channels,
                kernel_size=self.OUTPUT_KERNEL_SIZE,
            ),
            torch.nn.Sigmoid(),
        )

    @t.override
    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        map the fixed random noise tensor to an image.

        The method accepts both a batched tensor of shape (batch, input_channels, height, width) and a
        single unbatched tensor of shape (input_channels, height, width) - the data module serves the
        latter, so a batch dimension is added transparently and removed again from the result.

        :param noise: random noise input tensor, batched or unbatched
        :return: produced image, matching the batching of the input, with pixel values in [0, 1]
        """
        # remember whether we need to strip the batch dimension from the result
        unbatched: bool = noise.dim() == self.UNBATCHED_DIMENSIONS
        if unbatched:
            noise = noise.unsqueeze(0)

        batch: int = noise.shape[0]
        assert torch.Size([batch, self.config.input_channels, self.height, self.width]) == noise.shape

        # encoder pass: remember the feature map that enters every level for the skip connections
        skip_inputs: list[torch.Tensor] = []
        activation: torch.Tensor = noise

        for level in range(self.depth):
            skip_inputs.append(activation)
            activation = self.down_blocks[level](activation)

        # decoder pass: from the bottleneck upwards, fusing each skip connection back in
        for level in reversed(range(self.depth)):
            skip: torch.Tensor = self.skip_blocks[level](skip_inputs[level])
            activation = self.up_blocks[level](activation, skip)

        # project the final feature map to an image
        image: torch.Tensor = self.output_block(activation)
        assert torch.Size([batch, self.config.output_channels, self.height, self.width]) == image.shape

        # restore the original batching of the input
        if unbatched:
            image = image.squeeze(0)

        return image
