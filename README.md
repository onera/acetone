# acetone
Predictable programming framework for ML applications in safety-critical systems.

## Code architecture

You'll find in the home directory the files regarding the licencing and copyright of the framework:

* [AUTHORS.md](./AUTHORS.md)
* [LICENSE](./LICENSE)

This directory also contains the [requirements.txt](./requirements.txt) which list the package versionning used in the framework.

The [data](./data/) folder contains the generated test files.

The folder [init](./init/) includes the source files creating the test files.

The [src](./src/) folder contains the backend code of ACETONE.

## Istallation

Firstly, install the packages listed in [`requirements.txt`](./requirements.txt)

```
pip install -r requirements.txt
```


## How to use it

The following commands generate a test neural network before generating the corresponding C code using ACETONE.

### Generating the neural network

* Go to the *init* directory
    `cd acetone/init`

* Run the *initial_setup.py* code
    `python3 initial_setup.py`

This script defines a neural network with a Lenet-5 architecture using the framework Keras. It then save the model in *.h5* and *.json* files. The later one is created using a specific function, developped by us, to write the keras model in ACETONE's format. The scripts also creates a random input to test the neural network. Finally, the scripts saves and prints, as a reference, the output of the inference done by the Keras framework.

### Generating the C code with ACETONE



## Tests

## Reproduce the paper's experiments

describe how to reproduce the paper's experiments

## Capability

which type of CNN can be deal with

