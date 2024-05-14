# Date

This document was last updated on the 14/05/2024.

# Aspect of the DNN

The DNN can be:
* Sequential
* Non-Sequential

The DNN must be:
* Feedforward (without back-propagation)

# Formats

The DNN is describeb in one of the following formats:
* JSON (specifique decription made for the framework. Confer to the file [JSON_from_keras_model.py](./src/format_importer/H5_importer/JSON_from_keras_model.py))
* NNET 
* ONNX
* H5

# Layers implemented by formats

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

### Activation layers implemented as a main layer

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
* Reshape
* Resize (3 modes)
* Shape
* Sub
* Unsqueeze

### Activation layers implemented as a main layer

* Softmax

### Activation layers

* LeakyRelu
* Linear
* ReLu
* Sigmoid
* TanH

### Main layers implemented as a activation layer

* Clip
* Exp
* Log
