from pystache import TemplateSpec, render

from acetone_nnet import Conv2D
from acetone_nnet.versioning.version_implementation.conv_implementation import (
    conv2d_factory,
)


class CudaConvolutionParameters:
    """Common parameters for CUDA-based convolution."""

    def __init__(self, layer: Conv2D) -> None:
        """Initialise Convolution template from layer."""
        self.layer = layer

    def layer_id(self) -> int:
        """Return layer unique identifier."""
        return self.layer.idx

    def layer_name(self) -> str:
        """Return layer name."""
        return self.layer.name

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

    def tensor_type(self) -> str:
        """Return C input and output tensor type."""
        # FIXME This needs to come from the code generator
        return "float"

    def activation_function_src(self) -> str:
        """Return C code for activation function."""
        # FIXME The C temporary variable name should be a template parameter
        return self.layer.activation_function.write_activation_str("output[k]")


class CudaConvolutionNaiveCall(CudaConvolutionParameters, TemplateSpec):
    """Naive CUDA-based convolution function call template."""

    template_extension = "cpp"


class CudaConvolutionNaive(Conv2D):
    """Code generator for CUDA naive convolution implementation."""

    def __init__(self, layer: Conv2D):
        self.layer = layer

    # FIXME Oh, god why. Let's define a cleaner proxy
    def __getattr__(self, item):
        return self.layer.__getattribute__(item)

    def generate_inference_code_layer(self) -> str:
        return render(CudaConvolutionNaiveCall(self.layer))


# FIXME Find a more module-level way, doing it on import causes issues with unused imports
def register_cuda_convolution_naive() -> None:
    conv2d_factory.register_implementation(
        "cuda/naive",
        lambda i, _: CudaConvolutionNaive(i),
    )
