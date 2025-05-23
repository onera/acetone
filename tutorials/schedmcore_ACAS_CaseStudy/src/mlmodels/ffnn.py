'''
This file is part of a program that allows simulating the ACAS Xu and VCAS neural network controlled systems
Copyright (c) 2023 Arthur Claviere ONERA/ISAE-SUPAERO

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
'''

import numpy as np
from pathlib import Path
import subprocess

def write_input(x, tempdir_name):
    with open(tempdir_name / "input.txt" , "w+") as fi:
        out_string = " ".join(
            [float(n).hex().replace("0000000p", "p") for n in x],
        )
        print(f"{out_string}", end=" ", file=fi, flush=True)
        print(" ", file=fi)


def extract_outputs_c(
        path_to_output: str | Path
) -> np.ndarray:
    """Get the outputs values from the txt file."""
    with Path.open(Path(path_to_output)) as f:
        for i, line in enumerate(f):

            line = line[:-2].split(" ")
            line = list(map(float.fromhex, line))
            line = np.array(line)
    f.close()
    return line

class FFNN:
    """A class for representing feed forward neural networks

    Attributes
    ----------
    n_layers : int
        The number of layers in the network
    topology : list of int
        The number of neurons per layer (including input and output layers)
    weights : list of 2D numpy arrays
        The weights conecting the layers
    biases : list of 1D numpy arrays
        The biases of each layer
    act_func : str
        The activation function computed by the hidden layers
    tempdir : str
        The temporary directory storing the C code

    Methods
    -------
    get_n_inputs
        Get the number of inputs of the network
    get_n_outputs
        Get the number of outputs of the network
    get_n_layers
        Get the number of layers in the network
    get_topology
        Get the topology of the network
    get_weights
        Get the weights of the network
    get_biases
        Get the biases of the network
    get_activation_function
        Get the activation function computed by the hidden layers.
    compute_output(x)
        Compute the output of the network, for a given input vector x.
    compute_gradient(x)
        Compute the gradient of the network w.r.t. its inputs, at a given point x.
    """

    def __init__(self, n_layers, topology, weights, biases, act_func, tempdir):
        """Constructor"""
        self.n_layers = n_layers
        self.topology = topology
        self.weights = weights
        self.biases = biases
        self.act_func = act_func
        self.tempdir = tempdir
        self.tempdir_name = Path(tempdir.name)
        assert (act_func in ['relu', 'tanh']), "Non supported activation function"

    def __str__(self):
        return "Number of layers: {}\nNeurons per layer: {}\nWeights:\n{}\nBiases:\n{}\nActivation function: {}\n".format(
                self.n_layers, self.topology, self.weights, self.biases, self.act_func)

    def get_n_inputs(self):
        """Get the number of inputs of the network

        Parameters
        ----------
        None

        Return
        ------
        int
            The number of inputs of the neural network

        """
        return self.topology[0] # number of neurons in the input layer

    def get_n_outputs(self):
        """Get the number of outputs of the network

        Parameters
        ----------
        None

        Return
        ------
        int
            The number of outputs of the neural network

        """
        return self.topology[-1] # number of neurons in the output layer

    def get_n_layers(self):
        """Get the number of layers of the network

        Parameters
        ----------
        None

        Return
        ------
        int
            The number of layers of the neural network

        """
        return self.n_layers

    def get_topology(self):
        """Get the topology of the network

        Parameters
        ----------
        None

        Return
        ------
        1D numpy array
            The topology of the neural network (number of neurons per layer)

        """
        return self.topology

    def get_weights(self):
        """Get the weights of the network

        Parameters
        ----------
        None

        Return
        ------
        list
            A list of 2D numpy arrays where the i^th array corresponds to the weights connecting layer i to layer i+1

        """
        return self.weights

    def get_biases(self):
        """Get the biases of the network

        Parameters
        ----------
        None

        Return
        ------
        list
            A list of 1D numpy arrays where the i^th array corresponds to the biases connecting layer i to layer i+1

        """
        return self.biases
        
    def get_activation_function(self):
        """Get the activation function computed by the hidden layers

        Parameters
        ----------
        None

        Return
        ------
        str
            The activation function computed by the hidden layers

        """
        return self.act_func

    def compute_output(self, x, mode):
        """Map an input vector x to the corresponding neural network output

        Parameters
        ----------
        x : 1D numpy array
            The input vector to be considered
        mode : str
            The mode to compute the inference of the networks (schedmcore or acetone)

        Return
        ------
        1D numpy array
            The output of the neural network

        """
        if mode == "acetone":
            write_input(x, self.tempdir_name)
            cmd = [
                self.tempdir_name / "inference",
                self.tempdir_name / "output_c.txt",
                self.tempdir_name / "input.txt",
            ]
            subprocess.run(cmd, check=True)
            temp_out = extract_outputs_c(self.tempdir_name / "output_c.txt")
        elif mode == "schedmcore":
            temp_out = x
            for i in range(self.n_layers - 1):
                temp_in = np.dot(self.weights[i],temp_out) + self.biases[i]
                if i == self.n_layers - 2:
                    temp_out = temp_in
                else:
                    if self.act_func=='relu':
                        temp_out = np.maximum(temp_in, np.zeros(self.topology[i+1]))
                    elif self.act_func=='tanh':
                        temp_out = np.tanh(temp_in)
                    elif self.act_func=='sigmoid':
                        temp_out = 1 / (1 + np.exp(-temp_in))
        else:
            raise NotImplementedError(f"Mode {mode} not recognized")
        return temp_out

    def compute_gradient(self, x):
        """Compute the gradient the gradient of the neural network for a given input x

        Parameters
        ----------
        x : 1D numpy array
            The input vector to be considered

        Return
        ------
        tuple
            A tuple containing (1) a 2D numpy array representing the 
            gradient (Jacobian matrix) of the neural network and (2) the
            output of the neural network

        """
        if self.act_func=='relu':
            # find active weights of neural network
            active_weights = []
            active_weights.append(self.weights[0])
            temp_out = x
            for i in range(self.n_layers - 2):
                temp_in = np.dot(self.weights[i],temp_out) + self.biases[i]
                list_activated_neurons = []
                for k in range(temp_in.size):
                    if temp_in[k] >= 0:
                        list_activated_neurons.append(k)
                active_weights[i] = active_weights[i][list_activated_neurons, :]
                active_weights.append(self.weights[i+1][:,list_activated_neurons])
                temp_out = np.maximum(temp_in, np.zeros(temp_in.size))
                
            temp_out = np.dot(self.weights[self.n_layers - 2],temp_out) + self.biases[self.n_layers - 2]
                        
            # compute the gradient
            J = active_weights[self.n_layers - 2]
            for i in range(self.n_layers - 3, -1, -1):
                J = np.dot(J,active_weights[i])
                
            return (J, temp_out)
            
        elif self.act_func=='tanh':
            temp_out = x
            J = np.eye(self.topology[0])
            for i in range(self.n_layers - 1):
                temp_in = np.dot(self.weights[i],temp_out) + self.biases[i]
                J = np.dot(self.weights[i], J)
                if i == self.n_layers - 2:
                    temp_out = temp_in
                else:
                    temp_out = np.tanh(temp_in)
                    dtanh_dx = np.eye(self.topology[i+1])
                    for j in range(self.topology[i+1]):
                        dtanh_dx[j,j] = 1 - np.tanh(temp_in[j])**2
                    J = np.dot(dtanh_dx, J)
                    
            return (J, temp_out)
            
            
