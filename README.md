# ACETONE
Predictable programming framework for ML applications in safety-critical systems.

This repo contains the code of the framework presented in the ECRTS'22 paper  "ACETONE: Predictable programming framework for ML applications in safety-critical systems".

This framework aims at, for a given neural network, generate a C code corresponding to this neural network to later imbed it on a critical system.

## Code architecture

You'll find in the home directory the files regarding the licencing and copyright of the framework:

* [AUTHORS.md](./AUTHORS.md)
* [LICENSE](./LICENSE)

This directory also contains the [requirements.txt](./requirements.txt) which list the package versionning used in the framework.

The [data](./data/) folder contains the generated test files.

The folder [init](./init/) includes the source files creating the test files.

The [test](./test/) directory includes several tests for the framework (currently empty).

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

* Go to the *init* directory
```
cd acetone/init
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
cd ../src
```

* Call ACETONE with the following arguments:
  * The JSON file describing the model
  * The input file with the test data
  * The name of the function to generate (here 'lenet5')
  * The number of test to run (here 1)
  * The version to use ('v1', 'v2' or 'v3')
  * The directory in which the code will be generated

For the first version
```
python3 main.py ../data/example/lenet5.json ../data/example/test_input_lenet5.txt lenet5 1 v1 ../data/example/v1
```

* Compile the code
```
cd ../data/example/v1
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
python3 eval_semantic_preservation.py ../data/example/output_keras.txt ../data/example/v1/output_acetone.txt 1
```


## Tests

No tests are implemented yet. 
This service will be provided soon.

## Reproduce the paper's experiments

To reproduce the result of semantic experiment with ACETONE as described in the paper, use the following commands:

* For the acas_decr128 model
```
cd acetone/src
pyhton3 main.py ../data/acas_decr128/acas_decr128.json ../data/acas_decr128/test_input_acas_decr128.txt acas_decr128 1000 v1 ../output/acas_decr128/v1
cd ../output/acas_decr128/v1
make all
./acas_decr128 output_acetone.txt
cd ../../../src
python3 eval_semantic_preservation ../data/acas_decr128/output_keras.txt ../output/acas_decr128/v1/output_acetone.txt
```

* For the lent5 model

```
cd acetone/src
pyhton3 main.py ../data/lenet5_trained/lenet5_trained.json ../data/lenet5_trained/test_input_lenet5.txt lenet5_trained 1000 v1 ../output/lenet5_trained/v1
cd ../output/lenet5_trained/v1
make all
./lenet5_trained output_acetone.txt
cd ../../../src
python3 eval_semantic_preservation ../data/lenet5_trained/output_keras.txt ../output/lenet5_trained/v1/output_acetone.txt
```

## Capability

As of the 07/03/2024, the framework can generate code for neural network meeting the following condition:

* The neural network is Sequential and Feedforward
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