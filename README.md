# ACETONE
Predictable programming framework for ML applications in safety-critical systems.

This repo contains the code of the framework presented in the ECRTS'22 paper  ["ACETONE: Predictable programming framework for ML applications in safety-critical systems"](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ECRTS.2022.3).

This framework generate a C code corresponding to a neural network given as input.


## Code architecture

You'll find in the home directory the files regarding the licencing and copyright of the framework:

* [AUTHORS.md](./AUTHORS.md)
* [LICENSE](./LICENSE)

This directory also contains the [requirements.txt](./requirements.txt) which list the package versionning used in the framework.

The [test](./test/) directory includes several tests for the framework and the data to run them.

The [src](./src/) folder contains the backend code of ACETONE.

## Installation
Clone the GitHub repo on your computer

```
git clone https://github.com/onera/acetone.git
```

Then install the packages listed in [`requirements.txt`](./requirements.txt)

```
pip install -r acetone/requirements.txt
```


## Code Generation

The following commands generate a test neural network before generating the corresponding C code using ACETONE.

### Generating the neural network

In the *acetone* directory

* Run the *initial_setup.py* code
```
python3 tests/models/lenet5/lenet5_example/initial_setup.py
```

This script defines a neural network with a Lenet-5 architecture using the framework Keras. It then save the model in *.h5* and *.json* files. The later one is created using a specific function, developped by us, to write the keras model in ACETONE's format. The scripts also creates a random input to test the neural network. Finally, the scripts saves and prints, as a reference, the output of the inference done by the Keras framework.

### Generating the C code with ACETONE

Then, generate the C code with ACETONE.

* Call ACETONE with the following arguments:
  * The file describing the model
  * The name of the function to generate (here 'lenet5')
  * The number of test to run (here 1)
  * The algorithm used for the convolution layer ('6loops','indirect_gemm_'+TYPE, 'std_gemm_'+TYPE, with TYPE being amongst 'nn','nt',    'tn','tt')
  * The directory in which the code will be generated
  * The input file with the test data

```
python3 src/cli_acetone.py tests/models/lenet5/lenet5_example/lenet5.h5  lenet5 1 std_gemm_nn tests/models/lenet5/lenet5_example/lenet5_generated .tests/models/lenet5/lenet5_example/test_input_lenet5.txt
```

* Compile the code
```
make -C tests/models/lenet5/lenet5_example all
```

* Execute the file with the path to the directory of the output file as argument
```
./tests/models/lenet5/lenet5_example/lenet5 ./tests/models/lenet5/lenet5_example/output_acetone.txt
```

* Compare the output given by Keras and ACETONE
```
python3 src/cli_compare.py ./tests/models/lenet5/lenet5_example/output_keras.txt ./tests/models/lenet5/lenet5_example/output_acetone.txt 1
```

## Tests

Tests are implemented in the folder *tests*.

To run all of them, use the following command:
```
python3 -m unittest discover tests/test_inference/test_layer tests/tests_network tests/tests_importer
```

To only run a test, use the command
```
python3 -m unittest PATH_TO_TEST
```
where PATH_TO_TEST is the path to your test.

## Reproduce the paper's experiments

To reproduce the result of semantic experiment with ACETONE as described in the paper, use the following commands:

* For the acas_decr128 model
```
pyhton3 src/cli_acteone.py tests/models/acas_decr128/acas_decr128.json acas_dcre128 1000 std_gemm_nn tests/models/acas_decr128/output_acetone tests/models/acas_decr128/test_input_acas_decr128.txt
make -C tests/models/acas_decr128/output_acetone all
./tests/models/acas_decr128/output_acetone/acas_decr128 tests/models/acas_decr128/output_acetone/output_acetone.txt
python3 src/cli_compare.py tests/models/acas_decr128/output_keras.txt tests/models/acas_decr128/output_acetone/output_acetone.txt
```

* For the lent5 model

```
pyhton3 src/cli_acteone.py tests/models/lenet5/lenet5_trained/lenet5_trained.json lenet5_trained 1000 std_gemm_nn tests/models/lenet5/lenet5_trained/output_acetone tests/models/lenet5_trained/test_input_lenet5.txt
make -C tests/models/lenet5/lenet5_trained/output_acetone all
./tests/models/lenet5/lenet5_trained/output_acetone/lenet5_trained tests/models/lenet5/lenet5_trained/output_acetone/output_acetone.txt
python3 src/cli_compare.py tests/models/lenet5/lenet5_trained/output_keras.txt tests/models/lenet5/lenet5_trained/output_acetone/output_acetone.txt
```

## Capability

As of the 26/03/2024, the framework can generate code for neural network meeting the following condition:

* The neural network is Sequential and Feedforward

* The neural-network is describeb in one of the following formats:
  * JSON (specifique decription mad efor the framework. Confer to the file [JSON_from_keras_model.py](./src/format_importer/H5_importer/JSON_from_keras_model.py))
  * NNET 
  * ONNX
  * H5 (by transforming into a JSON file)

* Its layers are amongst the following (both Keras and Onnx layers):
  * Dense
  * Convolutional (Conv2D) (3 option for the algorithm)
  * Pooling (AveragePooling2D, MaxPooling2D)
  * Flatten
  * Normalization (Normalization layers of Keras)
  * Dropout (Dropout layers of Keras)
  * Reshape
  * Permute
  * MatMul
  * Gemm
  * Gather
  * Dot
  * Concatenate
  * Clip
  * Add (where one of the inputs is an ONNX initializer)
  * Resize (Cubic interpolation, Linear interpolation, Nearest)
  * Pad (Wrap padding, Edge padding, Reflect padding, Constant padding)
  * Broadcast (Add, Average, Divide, Maximum, Minimum, Multiply, Subtract)
  * Unsqueeze (ONNX layer)
  * LRN (ONNX layer)
  * Shape (ONNX layer)
  * BatchNormalization (ONNX layer)

* Its activation layers are amongst the following:
  * Linear
  * Tanh
  * ReLu
  * Sigmoid
  * Softmax
  * Exp
  * Log
  

## License

The project is under the GNU Lesser General Public License as published by the Free Software Foundation ; either version 3 of  the License or (at your option) any later version.

See LICENSE for details.