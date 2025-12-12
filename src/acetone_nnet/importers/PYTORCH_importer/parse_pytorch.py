from pathlib import Path

import numpy as np

from acetone_nnet.generator import (
    ActivationFunctions,
    Add,
    Average,
    AveragePooling2D,
    BatchNormalization,
    Concatenate,
    ConstantPad,
    Conv2D,
    Dense,
    Flatten,
    InputLayer,
    LeakyReLu,
    Linear,
    Maximum,
    MaxPooling2D,
    Minimum,
    Multiply,
    ReLu,
    ResizeCubic,
    ResizeLinear,
    ResizeNearest,
    Sigmoid,
    Softmax,
    Subtract,
    TanH,
)
from acetone_nnet.graph import graph_interpretor
from acetone_nnet.ir import Layer

from torch.export import export, ExportedProgram
def load_pytorch(program : ExportedProgram):
    layers: list[Layer] = []
    layers.append(InputLayer(
        original_name="",
        idx=0,
        size=3,
        input_shape=[1,1,1,3],
        data_format="channels_first",
    ))
    return layers, "f4", "f4", "channels_first", 10, {}