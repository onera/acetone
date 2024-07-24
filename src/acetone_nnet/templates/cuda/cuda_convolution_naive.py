from typing import Self

from acetone_nnet import Conv2D
from pystache import TemplateSpec


class CudaConvolutionParameters:
    """Common parameters for CUDA-based convolution."""

    def __init__(self: Self, layer: Conv2D) -> None:
        """Initialise Convolution template from layer."""
        self.layer = layer

    def kernel_height(self) -> int:
        """Return kernel height."""
        return self.layer.kernel_h

    def kernel_width(self) -> int:
        """Return kernel width."""
        return self.layer.kernel_w

    def input_height(self) -> int:
        """Return input height."""
        return self.layer.input_height

    def input_width(self) -> int:
        """Return input width."""
        return self.layer.input_width

    def output_height(self) -> int:
        """Return output height."""
        return self.layer.output_height

    def output_width(self) -> int:
        """Return output width."""
        return self.layer.output_width

    def output_size(self) -> int:
        """Return number of output elements."""
        return self.layer.output_channels * self.layer.output_width * self.layer.output_height

    def channel_count(self) -> int:
        """Return number of input channels."""
        return self.layer.input_channels

    def kernel_count(self) -> int:
        """Return number of kernels."""
        return self.layer.nb_filters

    def stride(self) -> int:
        """Return convolution input stride."""
        return self.layer.strides

    def dilation(self) -> int:
        """Return convolution input dilation."""
        return self.layer.dilation_rate

    def road(self) -> int:
        """Return output variable index."""
        if self.layer.path is None:
            return 0
        return self.layer.path

    def padding_top(self) -> int:
        """Return padding value at tensor top."""
        return self.layer.pad_top

    def padding_left(self) -> int:
        """Return padding value on tensor left."""
        return self.layer.pad_left

    def kernel_var_name(self) -> str:
        """Return C variable name for kernel tensor."""
        return f"weights_{self.layer.name}_{self.layer.idx:02d}"

    def biases_var_name(self) -> str:
        """Return C variable name for biases array."""
        return f"biases_{self.layer.name}_{self.layer.idx:02d}"

    def input_var_name(self) -> str:
        """Return C variable name for input tensor."""
        return self.layer.previous_layer[0].output_str


class CudaConvolutionNaiveDefinition(CudaConvolutionParameters, TemplateSpec):
    """Naive CUDA-based convolution function declaration template."""

    # TODO Pass template to pystache.render
    template_extension = "hpp"


class CudaConvolutionNaiveCall(CudaConvolutionParameters, TemplateSpec):
    """Naive CUDA-based convolution function call template."""

    template_extension = "cpp"
