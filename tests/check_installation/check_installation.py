"""*******************************************************************************
* ACETONE: Predictable programming framework for ML applications in safety-critical systems
* Copyright (c) 2022. ONERA
* This file is part of ACETONE
*
* ACETONE is free software ;
* you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation ;
* either version 3 of  the License, or (at your option) any later version.
*
* ACETONE is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY ;
* without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
* See the GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License along with this program ;
* if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307  USA
******************************************************************************
"""

"""
This script offers a quick check up of the main fonctions of the framework.
It does not chack the accuracy of the generated code nor realize the inference in C.
Please refer to the tests to check the results, or the exemples to realize the C inference
"""

import tempfile

import numpy as np

### Checking the importation of the package ###
print("Testing the importation...")
import acetone_nnet

print("Importation done\n")

### Creating a few layers ###
print("Testing the creation of a few layers...")

# Input Layer
print("Input Layer")
l1 = acetone_nnet.InputLayer(
    idx=0,
    size=300,
    input_shape=[1, 3, 10, 10],
    data_format="channels_first",
)

# Dense Layer
print("Dense Layer")
l2 = acetone_nnet.Dense(
    idx=1,
    size=300,
    weights=np.random.random((10, 300)),
    biases=np.random.random(300),
    activation_function=acetone_nnet.Linear(),
)

# Convolution layer
print("Convolution Layer")
l3 = acetone_nnet.Conv2D6loops(
    idx=2,
    conv_algorithm="6loops",
    size=300,
    padding=[1, 1, 1, 1],
    strides=1,
    kernel_h=3,
    kernel_w=3,
    dilation_rate=1,
    nb_filters=3,
    input_shape=[1, 3, 10, 10],
    output_shape=[1, 3, 10, 10],
    weights=np.random.random((3, 3, 3, 3)),
    biases=np.random.random(3),
    activation_function=acetone_nnet.Linear(),
)

# MaxPooling
print("MaxPooling Layer")
l4 = acetone_nnet.MaxPooling2D(
    idx=3,
    size=300,
    padding="same",
    strides=1,
    pool_size=3,
    input_shape=[1, 3, 10, 10],
    output_shape=[1, 3, 10, 10],
    activation_function=acetone_nnet.Sigmoid(),
)

# Add
print("Add Layer")
l5 = acetone_nnet.Add(
    idx=4,
    size=140,
    input_shapes=np.array([[1, 4, 7, 5], [1, 4, 7, 5]]),
    output_shape=[1, 4, 7, 5],
    activation_function=acetone_nnet.TanH(),
)

# WrapPad
try:
    print("WrapPadding Layer")
    l6 = acetone_nnet.Wrap_pad(
        idx=5,
        size=501305,
        pads=[0, 0, 5, 7, 0, 0, 6, 3],
        constant_value=56,
        axes=[2, 3],
        input_shape=[1, 52, 68, 7],
        activation_function=acetone_nnet.ReLu(),
    )
except:
    pass
# Concatenate
print("Concatenate Layer")
l7 = acetone_nnet.Concatenate(
    idx=6,
    size=300,
    axis=3,
    input_shapes=[[1, 3, 10, 5], [1, 3, 10, 5]],
    output_shape=[1, 3, 10, 10],
    activation_function=acetone_nnet.Sigmoid(),
)

print("Layer creation done\n")

### Importing a few models ###
print("Testingt the importation of a few models...")
model_path = "/".join(__file__.split("/")[:-1]) + "/models/"

# Model JSON
print("Model JSON")
decr128 = acetone_nnet.CodeGenerator(
    model_path + "acas_decr128.json",
    None,
    "inference",
    1,
    "std_gemm_nn",
    False,
)

# Model H5
print("Model H5")
lenet5 = acetone_nnet.CodeGenerator(
    model_path + "lenet5_trained.h5",
    None,
    "inference",
    1,
    "std_gemm_nn",
    False,
)

# Model NNet
print("Model NNet")
acas_fully_connected = acetone_nnet.CodeGenerator(
    model_path + "acas_fully_connected.nnet",
    None,
    "inference",
    1,
    "std_gemm_nn",
    True,
)

# Model ONNX
print("Model ONNX")
squeezenet = acetone_nnet.CodeGenerator(
    model_path + "squeezenet1.onnx",
    None,
    "inference",
    1,
    "std_gemm_nn",
    False,
)

print("Models importation done\n")

### Testing Python's inference ###
print("Testing Python's inference...")

# Model JSON
print("Model JSON")
tmp_dir = tempfile.TemporaryDirectory()
decr128.compute_inference(tmp_dir.name)
tmp_dir.cleanup()

# Model H5
print("Model H5")
tmp_dir = tempfile.TemporaryDirectory()
lenet5.compute_inference(tmp_dir.name)
tmp_dir.cleanup()

# Model NNet
print("Model NNet")
tmp_dir = tempfile.TemporaryDirectory()
acas_fully_connected.compute_inference(tmp_dir.name)
tmp_dir.cleanup()

# Model ONNX
print("Model ONNX")
tmp_dir = tempfile.TemporaryDirectory()
squeezenet.compute_inference(tmp_dir.name)
tmp_dir.cleanup()

print("Python's inference done")

### Testing Python's inference ###
print("Testing C code generation...")

# Model JSON
print("Model JSON")
tmp_dir = tempfile.TemporaryDirectory()
decr128.generate_c_files(tmp_dir.name)
tmp_dir.cleanup()

# Model H5
print("Model H5")
tmp_dir = tempfile.TemporaryDirectory()
lenet5.generate_c_files(tmp_dir.name)
tmp_dir.cleanup()

# Model NNet
print("Model NNet")
tmp_dir = tempfile.TemporaryDirectory()
acas_fully_connected.generate_c_files(tmp_dir.name)
tmp_dir.cleanup()

# Model ONNX
print("Model ONNX")
tmp_dir = tempfile.TemporaryDirectory()
squeezenet.generate_c_files(tmp_dir.name)
tmp_dir.cleanup()

print("C code generation done")
