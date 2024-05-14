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

### User mode

Install le package using `pip`
```
pip install acetone-nnet
```


### Development Mode

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

### Generating the C code with ACETONE package

Then, generate the C code with ACETONE.

* Call ACETONE with the following arguments:
  * The file describing the model
  * The name of the function to generate (here 'lenet5')
  * The number of test to run (here 1)
  * The algorithm used for the convolution layer ('6loops','indirect_gemm_'+TYPE, 'std_gemm_'+TYPE, with TYPE being amongst 'nn','nt',    'tn','tt')
  * The directory in which the code will be generated
  * The input file with the test data

```
acetone_generate tests/models/lenet5/lenet5_example/lenet5.h5  lenet5 1 std_gemm_nn tests/models/lenet5/lenet5_example/lenet5_generated tests/models/lenet5/lenet5_example/test_input_lenet5.txt
```

* Compile the code
```
make -C tests/models/lenet5/lenet5_example/lenet5_generated all
```

* Execute the file with the path to the directory of the output file as argument
```
./tests/models/lenet5/lenet5_example/lenet5_generated/lenet5 ./tests/models/lenet5/lenet5_example/lenet5_generated/output_acetone.txt
```

* Compare the output given by Keras and ACETONE
```
acetone_compare ./tests/models/lenet5/lenet5_example/output_keras.txt ./tests/models/lenet5/lenet5_example/lenet5_generated/output_acetone.txt 1
```

## Tests

Tests are implemented in the folder *tests*.

To run them, use the `run_tests.py` script from the `tests/` folder.
```
python3 run_tests.py all
```

You can replace the `all` argument by the name of a subfolder to only run the tests in it.
```
python3 run_tests.py FOLDER_NAME
```
where FOLDER_NAME is the name of your subfolder.

You can run one test by using the command
```
python3 -m unittest PATH_TO_TEST
```
where PATH_TO_TEST is the path tot your test.

## Reproduce the paper's experiments

To reproduce the result of semantic experiment with ACETONE as described in the paper, use the following commands:

* For the acas_decr128 model
```
acetone_generate tests/models/acas/acas_decr128/acas_decr128.json acas_dcre128 1000 std_gemm_nn tests/models/acas/acas_decr128/output_acetone tests/models/acas/acas_decr128/test_input_acas_decr128.txt
make -C tests/models/acas/acas_decr128/output_acetone all
./tests/models/acas/acas_decr128/output_acetone/acas_decr128 tests/models/acas/acas_decr128/output_acetone/output_acetone.txt
acetone_compare tests/models/acas/acas_decr128/output_keras.txt tests/models/acas/acas_decr128/output_acetone/output_acetone.txt
```

* For the lent5 model

```
acetone_generate tests/models/lenet5/lenet5_trained/lenet5_trained.json lenet5_trained 1000 std_gemm_nn tests/models/lenet5/lenet5_trained/output_acetone tests/models/lenet5_trained/test_input_lenet5.txt
make -C tests/models/lenet5/lenet5_trained/output_acetone all
./tests/models/lenet5/lenet5_trained/output_acetone/lenet5_trained tests/models/lenet5/lenet5_trained/output_acetone/output_acetone.txt
acetone_compare tests/models/lenet5/lenet5_trained/output_keras.txt tests/models/lenet5/lenet5_trained/output_acetone/output_acetone.txt
```

## Capability

Please refer to the [`implemented.md`](./implemented.md) file to see the current capabilities of the framework.

## License

The project is under the GNU Lesser General Public License as published by the Free Software Foundation ; either version 3 of  the License or (at your option) any later version.

See LICENSE for details.