# ACETONE
Predictable programming framework for ML applications in safety-critical systems.

This repo contains the code of the framework presented in the ECRTS'22 paper  "ACETONE: Predictable programming framework for ML applications in safety-critical systems".

This framework generate a C code corresponding to a neural network given as input.


## Code architecture

You'll find in the home directory the files regarding the licencing and copyright of the framework:

* [AUTHORS.md](./AUTHORS.md)
* [LICENSE](./LICENSE)

This directory also contains the [requirements.txt](./requirements.txt) which list the package versionning used in the framework.

The [test](./test/) directory includes several tests for the framework and the data to run them. It also contains a [example](./test/example/) used for the example describeb below.

The [src](./src/) folder contains the backend code of ACETONE.

## Istallation
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

* Go to the *example* directory
```
cd acetone/test/example/
```

* Run the *initial_setup.py* code
```
python3 initial_setup.py
```

This script defines a neural network with a Lenet-5 architecture using the framework Keras. It then save the model in *.h5* and *.json* files. The later one is created using a specific function, developped by us, to write the keras model in ACETONE's format. The scripts also creates a random input to test the neural network. Finally, the scripts saves and prints, as a reference, the output of the inference done by the Keras framework.

### Generating the C code with ACETONE

Then, generate the C code with ACETONE.

* Go to the framework's directory
```
cd ../../src
```

* Call ACETONE with the following arguments:
  * The JSON file describing the model
  * The name of the function to generate (here 'lenet5')
  * The number of test to run (here 1)
  * The algorithm used for the convolution layer ('6loops','indirect_gemm_'+TYPE, 'std_gemm_'+TYPE, with TYPE being amongst 'nn','nt',    'tn','tt')
  * The directory in which the code will be generated
  * The input file with the test data

```
python3 main.py ../test/example/lenet5.h5  lenet5 1 std_gemm_nn ../test/example/lenet5_generated ../test/example/test_input_lenet5.txt
```

* Compile the code
```
cd ../test/example/lenet5_generated
```
```
make all
```

* Execute the file with the path to the directory of the output file as argument
```
./lenet5 output_acetone.txt
```

* Compare the output given by Keras and ACETONE
```
cd ../../../src
python3 eval_semantic_preservation.py ../test/example/output_keras.txt ../test/example/lenet5_generated/output_acetone.txt 1
```


## Tests

No tests are implemented yet.
This service will be provided soon.

## Reproduce the paper's experiments

To reproduce the result of semantic experiment with ACETONE as described in the paper, use the following commands:

* For the acas_decr128 model
```
cd acetone/src
pyhton3 main.py ../test/data/acas_decr128/acas_decr128.json ../test/data/acas_decr128/test_input_acas_decr128.txt acas_decr128 1000 ../test/data/output/acas_decr128
cd ../test/data/output/acas_decr128
make all
./acas_decr128 output_acetone.txt
cd ../../../../src
python3 eval_semantic_preservation ../test/data/acas_decr128/output_keras.txt ../test/data/output/acas_decr128/output_acetone.txt
```

* For the lent5 model

```
cd acetone/src
pyhton3 main.py ../test/data/lenet5_trained/lenet5_trained.json ../test/data/lenet5_trained/test_input_lenet5.txt lenet5_trained 1000  ../test/data/output/lenet5_trained
cd ../test/data/output/lenet5_trained
make all
./lenet5_trained output_acetone.txt
cd ../../../../src
python3 eval_semantic_preservation ../test/data/lenet5_trained/output_keras.txt ../test/data/output/lenet5_trained/output_acetone.txt
```

## Capability

As of the 07/03/2024, the framework can generate code for neural network meeting the following condition:

* The neural network is Sequential and Feedforward

* The neural-network is describeb in one of the following formats:
  * JSON (specifique decription mad efor the framework. Confer to the file [JSON_from_keras_model.py](./src/format_importer/JSON_importer/JSON_from_keras_model.py))
  * NNET (by transforming the format into a JSON description)

* Its layers are amongst the following:
  * Dense
  * Convolutional (Conv2D)
  * Pooling (AveragePooling2D, MaxPooling2D)
  * Flatten

* Its activation layers are amongst the following:
  * Linear
  * Tanh
  * ReLu
  * Sigmoid
  * Softmax

## License

The project is under the GNU Lesser General Public License as published by the Free Software Foundation ; either version 3 of  the License or (at your option) any later version.

See LICENSE for details.