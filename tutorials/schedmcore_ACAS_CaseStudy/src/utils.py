'''
This file is part of a program that allows simulating the ACAS Xu and VCAS neural network controlled systems
Copyright (c) 2023 Arthur Claviere ONERA/ISAE-SUPAERO

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
'''

from math import *
import numpy as np
import subprocess
import tempfile
import acetone_nnet as an
from os import remove, listdir

# local imports
from .mlmodels.ffnn import *

def clean_environment(directory):
    list_files = [
        "output_c.txt",
        "Makefile", "main.o",
        "inference.h",
        "global_vars.c",
        "global_vars.o",
        "inference.c",
        "main.c",
        "test_dataset.h",
        "inference.o",
        "inference",
    ]
    if directory.exists():
        for file in listdir(directory):
            if file in list_files:
                remove(directory / file)

def parse_nnet_format(path_nnet, act_func):
    """Parse a neural network encoded in the .nnet format, and the associated normalization parameters
    
    Parameters
    ----------
    path_nnet : string
        The path to the .nnet file to be parsed
    act_func : string
        The activation function to be computed by the hidden neurons of the network
        
    Return
    ------
    tuple
        A 2-tuple containing:
            (i) a FFNN object
            (ii) a list of 1D numpy arrays corresponding to normalization parameters:
                1st array <- minimal values of the inputs, 
                2nd array <- maximal values of the inputs,
                3rd array <- mean values of the inputs,
                last array <- ranges of the inputs

    """
    tempdir = tempfile.TemporaryDirectory()
    tempdir_name = tempdir.name
    generator = an.CodeGenerator(file=path_nnet, nb_tests=0, verbose=False)
    generator.generate_c_files(tempdir_name)

    cmd = ["make", "-C", tempdir_name, "all"]
    subprocess.run(cmd, check=True)

    topology = []
    weights = []
    biases = []
    with open(path_nnet, 'r') as file:
        # Pass header
        line = file.readline()
        line = file.readline()
        line = file.readline()
        # Read the number of layers
        line = file.readline()
        values = line.split(',')
        n_layers = int(values[0]) + 1 # the layers comprise the input layer, the hidden layers and the output layer
        # Read the topology (number of neurons per layer)
        line = file.readline()
        values = line.split(',')
        topology = [int(values[k]) for k in range(n_layers)]
        # Pass the following line (useless)
        line = file.readline()
        # Read the minimal values of inputs
        line = file.readline()
        values = line.split(',')
        x_min = np.array([float(values[k]) for k in range(topology[0])])
        # Read the maximal values of inputs
        line = file.readline()
        values = line.split(',')
        x_max = np.array([float(values[k]) for k in range(topology[0])])
        # Read the mean values of inputs
        line = file.readline()
        values = line.split(',')
        x_mean = np.array([float(values[k]) for k in range(topology[0])])
        # Read the range of inputs
        line = file.readline()
        values = line.split(',')
        x_range = np.array([float(values[k]) for k in range(topology[0])])
        # Read the weights and biases
        for i in range(n_layers - 1):
            # Read weights connecting layer i to i+1
            W = []
            for j in range(topology[i+1]):
                line = file.readline()
                values = line.split(',')
                W_row = [float(values[k]) for k in range(topology[i])]
                W.append(W_row)
            weights.append(np.array(W))
            # Read biases of layer i+1
            B = []
            for j in range(topology[i+1]):
                line = file.readline()
                values = line.split(',')
                B_row = float(values[0])
                B.append(B_row)
            biases.append(np.array(B))
    return (FFNN(n_layers, topology, weights, biases, act_func, tempdir), [x_min, x_max, x_mean, x_range])


def clean_network(network):
    for (nnet_name, nnet, path_nnet) in network:
        nnet.tempdir.cleanup()
