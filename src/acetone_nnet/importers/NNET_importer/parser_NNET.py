"""Parser to NNet files.

*******************************************************************************
* ACETONE: Predictable programming framework for ML applications in safety-critical systems
* Copyright (c) 2022. ONERA
* This file is part of ACETONE
*
* ACETONE is free software ;
* you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation ;
* either version 3 of  the License, or (at your option) any later version.
*
* ACETONE is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY ;
* without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
* See the GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License along with this program ;
* if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307  USA
******************************************************************************
"""

from pathlib import Path

import numpy as np
from acetone_nnet.generator import Dense, InputLayer, Layer
from acetone_nnet.generator.activation_functions import Linear, ReLu

from . import nnet_normalize



def generate_layers(
        num_layers: int,
        weights: list[np.ndarray],
        biases: list[np.ndarray],
        layer_sizes: list[int],
) -> list[Layer]:
    """Generate the list of layers."""
    weights_shape = 4
    layers = []
    dict_cst = {}

    layers.append(
        InputLayer(
            idx=0,
            size=layer_sizes[0],
            input_shape=[layer_sizes[0]],
            data_format="channels_first",
        ),
    )
    layers[0].path = 0
    layers[0].find_output_str(dict_cst)

    for i in range(num_layers):
        weight = weights[i]
        if len(weight.shape) < weights_shape:
            for _j in range(4 - len(weight.shape)):
                weight = np.expand_dims(weight, axis=0)

        layer = Dense(idx=i + 1,
                      size=layer_sizes[i + 1],
                      weights=weight,
                      biases=biases[i],
                      activation_function=ReLu())

        layer.path = 0
        layer.find_output_str(dict_cst)
        layer.previous_layer.append(layers[-1])
        layers[-1].next_layer.append(layer)

        layers.append(layer)

    layers[-1].activation_function = Linear()

    return layers


def load_nnet(
        file_to_parse: str | Path,
        normalize: bool,
) -> (list[Layer], str, type, str, int, dict[int, int]):
    """Load an NNet model and return the corresponding ACETONE representation."""
    """Inspired from : # https://github.com/NeuralNetworkVerification/Marabou/blob/master/maraboupy/MarabouNetworkNNet.py."""
    # Recall : Example of nnet file head:
    # 7,5,5,50,  num_layers, input_size, output_size, max_layersize
    # 5,50,50,50,50,50,50,5,    Layersizes
    # 0
    # Inputs Values Min
    # Inputs Values Max
    # Moyennes de normalisation
    # ranges de normalisation
    # weights
    # biases
    file_to_parse = Path(file_to_parse)
    max_road = 1
    dict_cst = {}
    data_type = "float"
    data_type_py = np.float32

    with Path.open(file_to_parse) as f:

        line = f.readline()

        while line[0:2] == "//":  # ignore header lines with credits
            line = f.readline()

            # num_layers doesn't include the input layer!
        num_layers, input_size, output_size, max_layersize = (
            int(x) for x in line.strip().split(",")[:-1]
        )
        line = f.readline()

        # input layer size, layer1size, layer2size...
        layer_sizes = [int(x) for x in line.strip().split(",")[:-1]]

        line = f.readline()
        line = f.readline()
        input_minimums = [float(x) for x in line.strip().split(",")[:-1]]

        line = f.readline()
        input_maximums = [float(x) for x in line.strip().split(",")[:-1]]

        line = f.readline()
        means = [float(x) for x in line.strip().split(",")[:-1]]

        line = f.readline()
        ranges = [float(x) for x in line.strip().split(",")[:-1]]

        weights = []
        biases = []

        for layernum in range(num_layers):
            previous_layer_size = layer_sizes[layernum]
            current_layer_size = layer_sizes[layernum + 1]
            weights.append([])
            biases.append([])

            # weights for non-conventional nnet
            for i in range(current_layer_size):
                line = f.readline()
                aux = [float(x) for x in line.strip().split(",")[:-1]]
                weights[layernum].append([])
                for j in range(previous_layer_size):
                    weights[layernum][i].append(aux[j])

            # biases
            for _i in range(current_layer_size):
                line = f.readline()
                x = float(line.strip().split(",")[0])
                biases[layernum].append(x)

    f.close()

    # list of lists -> list of arrays
    weights = [np.transpose(np.array(weight)) for weight in weights]
    biases = [np.transpose(np.array(bias)) for bias in biases]

    layers = generate_layers(num_layers, weights, biases, layer_sizes)

    data_format = "channels_first"

    print("Finished model initialization.")

    if normalize:
        normalizer = nnet_normalize.Normalizer(
            input_size=layer_sizes[0],
            output_size=layer_sizes[-1],
            mins=input_minimums,
            maxes=input_maximums,
            means=means,
            ranges=ranges,
        )
        return (layers, data_type, data_type_py, data_format, max_road, dict_cst,
                normalizer)
    else:
        return layers, data_type, data_type_py, data_format, max_road, dict_cst
