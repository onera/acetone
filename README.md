# ACETONE
Predictable programming framework for ML applications in safety-critical systems.

This repo contains the code of the framework presented in the ECRTS'22 paper  ["ACETONE: Predictable programming framework for ML applications in safety-critical systems"](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ECRTS.2022.3).

This framework generate a C code corresponding to a neural network given as input.

## Package

The corresponding package is named [acetone-nnet](https://pypi.org/project/acetone-nnet/). 

The folder [examples/](./examples/) provides examples on how to use the main functionalities of the package (code 
generation and debug mode).

The folder [tests/check_installation](./tests/check_installation/) provides a sample script with a few quick checks and 
samples for the main functionalities of the package. It provides a quick verification of the installation.


## Code architecture

You'll find in the home directory the files regarding the licencing and copyright of the framework:

* [AUTHORS.md](./AUTHORS.md)
* [LICENSE](./LICENSE)

This directory also contains the [requirements.txt](./requirements.txt) which list the package versionning used in the framework.

The [test](./test/) directory includes several tests for the framework and the data to run them.

The [src](./src/) folder contains the backend code of ACETONE.

## Installation

### User mode

Install the package using `pip`
```
pip install acetone-nnet
```


### Development Mode

The following commands are general guidelines, your mileage, workflow or
preferences might vary. 

Clone the GitHub repo on your computer:
```commandline
git clone https://github.com/onera/acetone.git
```

Create a virtual environment for the project:
```commandline
python3 -m venv venv
```

Activate the virtual environment, this will be required for all work using the project:
```commandline
source venv/bin/activate
```

Then install the packages listed in [`requirements.txt`](./requirements.txt):
```commandline
pip install -r acetone/requirements.txt
```

Install the package in edit mode, ensuring all changes to the source are available in the installed package:
```commandline
pip install --editable .[dev]
```



## Code Generation

The following commands generate a test neural network before generating the corresponding C code using ACETONE.

### Generating the neural network

In the *acetone* directory

* Run the *initial_setup.py* code
```
python3 tests/models/lenet5/lenet5_example/initial_setup.py
```

This script defines a neural network with a Lenet-5 architecture using the Keras framework. It then save the model in 
`.h5` and `.json` files. The latter one is created using a specific function, developed by us, to write the keras model 
in ACETONE's format. The scripts also creates a random input to test the neural network. Finally, the scripts saves and 
prints, as a reference, the output of the inference done by the Keras framework.

### Generating the C code with ACETONE package

Then, generate the C code with ACETONE.

* Call ACETONE with the following arguments:
  * The file describing the model
  * The name of the inference function to generate (here 'lenet5')
  * The directory in which the code will be generated
  * The number of test to run (here 1)
  * [Optional] The input file with the test data
  * [Optional] The algorithm used for the convolution layer ('6loops','indirect_gemm_TYPE', 'std_gemm_TYPE', with TYPE being amongst 'nn','nt','tn','tt')

```
acetone_generate \
  --model tests/models/lenet5/lenet5_example/lenet5.h5 \
  --function lenet5 \
  --output tests/models/lenet5/lenet5_example/lenet5_generated \
  --dataset-size 1 \
  --dataset tests/models/lenet5/lenet5_example/test_input_lenet5.txt\
  --conv std_gemm_nn
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

### Tutorials

Some tutorials are present in the [`tutorials/`](./tutorials/README.md) folder to show more detailled examples and usages of the framework.

They can be run oon your local installation of the framework (by cloning the GitHub repository), or online using the above Binder link. 

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
acetone_generate \
  --model tests/models/acas/acas_decr128/acas_decr128.json \
  --function acas_decr128 \
  --output tests/models/acas/acas_decr128/output_acetone \
  --dataset-size 1000 \
  --dataset tests/models/acas/acas_decr128/test_input_acas_decr128.txt\
  --conv std_gemm_nn
  
make -C tests/models/acas/acas_decr128/output_acetone all

./tests/models/acas/acas_decr128/output_acetone/acas_decr128 tests/models/acas/acas_decr128/output_acetone/output_acetone.txt

acetone_compare tests/models/acas/acas_decr128/output_keras.txt tests/models/acas/acas_decr128/output_acetone/output_acetone.txt 1000
```

* For the lenet-5 model

```
acetone_generate \
  --model tests/models/lenet5/lenet5_trained/lenet5_trained.json \
  --function lenet5_trained \
  --output tests/models/lenet5/lenet5_trained/output_acetone \
  --dataset-size 1000 \
  --dataset tests/models/lenet5/lenet5_trained/test_input_lenet5.txt \
  --conv std_gemm_nn

make -C tests/models/lenet5/lenet5_trained/output_acetone all

./tests/models/lenet5/lenet5_trained/output_acetone/lenet5_trained tests/models/lenet5/lenet5_trained/output_acetone/output_acetone.txt

acetone_compare tests/models/lenet5/lenet5_trained/output_keras.txt tests/models/lenet5/lenet5_trained/output_acetone/output_acetone.txt 1000
```

## Capability

Please refer to the [`implemented.md`](./implemented.md) file to see the current capabilities of the framework.

## License

The project is under the GNU Lesser General Public License as published by the Free Software Foundation ; either version 3 of  the License or (at your option) any later version.

See LICENSE for details.

<img src="logos/Logo_IRIT.png" width=25% height=25% alt="Logo Institut de Recherche en Informatique de Toulouse">