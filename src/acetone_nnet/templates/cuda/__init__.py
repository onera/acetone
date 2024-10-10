from pystache import TemplateSpec

from .cuda_convolution_naive import (
    CudaConvolutionNaiveCall,
)

# Map operation types to known variants
# Each variant is defined by its unique name and a list of templates
operation_variants: dict[str, dict[str, list[type[TemplateSpec]]]] = {
    "convolution": {
        "naive": [CudaConvolutionNaiveCall],
    },
}
