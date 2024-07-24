from cuda_convolution_naive import (
    CudaConvolutionNaiveCall,
    CudaConvolutionNaiveDefinition,
)

operation_variants = {
    "convolution": {
        "naive": [CudaConvolutionNaiveDefinition, CudaConvolutionNaiveCall],
    },
}
