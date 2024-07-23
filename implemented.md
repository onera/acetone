# Date

This document was last updated on the 30/05/2024.

# Aspect of the DNN

The DNN can be:
* Sequential
* Non-Sequential

The DNN must be:
* Feedforward (without back-propagation)

The DNN can have multiple inputs or outputs, but only the first input and the output associated with the outmost layer will be taken into account.

# Data format

Only 1D and 2D operations are implemented.

The tensor must be in one of the following format:
* 'channels_first' (B,C,H,W)
* 'channels_last' (B,H,W,C)

with B the batch dimension (None or 1), C the number of channels, H the height of the tensor, W the width of the tensor.

# Formats

The DNN is describeb in one of the following formats:
* JSON (specifique decription made for the framework. Confer to the file [JSON_from_keras_model.py](./src/format_importer/H5_importer/JSON_from_keras_model.py))
* NNET 
* ONNX
* H5

# Layers by formats

For each format, the following layers cans be used in the DNN:

## JSON and H5

### Main layers

* Add
* Average
* AveragePooling2D
* BatchNormalization
* Concatenate
* Conv2D
* Dense
* Dropout
* Flatten
* Input layer
* Maximum
* MaxPooling2D
* Minimum
* Multiply
* Reshape
* Subtract
* ZeroPadding2D

#### Activation layers implemented as a main layer

* Softmax

### Activation layers

* LeakyRelu
* Linear
* ReLu
* Sigmoid
* TanH

## NNet

### Main layers

* Dense
* Input Layer

### Activation layers

* Linear
* ReLu

## ONNX

### Main layers

* Add
* AveragePool
* BatchNormalization
* Concat
* Constant
* Conv
* Div
* Dropout
* Flatten
* Gather
* Gemm
* GlobalAveragePool
* Input layer
* LRN
* MatMul (only 1D, with one of the inputs being an initializer)
* Max
* MaxPool
* Mean
* Min
* Mul
* Pad (4 modes)
* ReduceMax
* ReduceMean
* ReduceMin
* ReduceProd
* ReduceSum
* Reshape
* Resize (3 modes)
* Shape
* Sub
* Transpose
* Unsqueeze

#### Activation layers implemented as a main layer

* Softmax

### Activation layers

* LeakyRelu
* Linear
* ReLu
* Sigmoid
* TanH

#### Main layers implemented as a activation layer

* Clip
* Exp
* Log