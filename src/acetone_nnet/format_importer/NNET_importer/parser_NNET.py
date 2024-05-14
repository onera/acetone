"""
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


import keras.backend
import numpy as np

from . import nnet_normalize

from ...code_generator.layers import Dense, InputLayer
from ...code_generator.activation_functions import Linear, ReLu

keras.backend.set_floatx('float32')


def load_nnet(file_to_parse:str, normalize:bool):
    """
    Inspired from : # https://github.com/NeuralNetworkVerification/Marabou/blob/master/maraboupy/MarabouNetworkNNet.py
    """

    #Recall : Example of nnet file head:
    #7,5,5,50,  numLayers, inputSize, outputSize, maxLayersize
    #5,50,50,50,50,50,50,5,    Layersizes
    #0
    #Inputs Values Min
    #Inputs Values Max
    #Moyennes de normalisation
    #ranges de normalisation 
    #weights
    #biaises

    layers = []
    maxRoad = 1
    dict_cst = {}
    data_type = float
    data_type_py = np.float32

    with open(file_to_parse) as f:
     
        line = f.readline()

        while line[0:2] == "//": # ignore header lines with credits
            line = f.readline() 
                       
        # numLayers does't include the input layer!
        numLayers, inputSize, outputSize, maxLayersize = [int(x) for x in line.strip().split(",")[:-1]]
        line = f.readline()
        
        # input layer size, layer1size, layer2size...
        layerSizes = [int(x) for x in line.strip().split(",")[:-1]]

        line = f.readline()
        # symmetric = int(line.strip().split(",")[0])

        line = f.readline()
        inputMinimums = [float(x) for x in line.strip().split(",")[:-1]]

        line = f.readline()
        inputMaximums = [float(x) for x in line.strip().split(",")[:-1]]

        line = f.readline()
        means = [float(x) for x in line.strip().split(",")[:-1]]

        line = f.readline()
        ranges = [float(x) for x in line.strip().split(",")[:-1]]

        weights = []
        biases = []

        for layernum in range(numLayers):
            previousLayerSize = layerSizes[layernum]
            currentLayerSize = layerSizes[layernum + 1]
            weights.append([])
            biases.append([])

            # weights for non conventional nnet
            for i in range(currentLayerSize):
                line = f.readline()
                aux = [float(x) for x in line.strip().split(",")[:-1]]
                weights[layernum].append([])
                for j in range(previousLayerSize):
                    weights[layernum][i].append(aux[j])
                    
            # biases
            for i in range(currentLayerSize):
                line = f.readline()
                x = float(line.strip().split(",")[0])
                biases[layernum].append(x)

    f.close()

    layers.append(InputLayer(idx=0, size=layerSizes[0], input_shape=[layerSizes[0]], data_format='channels_first'))
    layers[0].path = 0
    layers[0].find_output_str(dict_cst)
    
    # list of lists -> list of arrays
    weights = [np.transpose(np.array(weight)) for weight in weights]
    biases = [np.transpose(np.array(bias)) for bias in biases]

    for i in range(0,numLayers):
        weight = weights[i]
        if(len(weight.shape) < 4):
            for j in range(4-len(weight.shape)): 
                weight = np.expand_dims(weight, axis=0)

        if(i == numLayers-1):
            layer = Dense(idx=i+1,
                        size=layerSizes[i+1],
                        weights=weight,
                        biases=biases[i],
                        activation_function=Linear())
        else:
            layer = Dense(idx=i+1,
                        size=layerSizes[i+1],
                        weights=weight,
                        biases=biases[i],
                        activation_function=ReLu())
        
        layer.path = 0
        layer.find_output_str(dict_cst)
        layer.previous_layer.append(layers[-1])
        layers[-1].next_layer.append(layer)
        layers.append(layer)

    data_format = 'channels_first'

    print("Finished model initialization.")    

    if(normalize):
        Normalizer = nnet_normalize.Normalizer(input_size = layerSizes[0], output_size = layerSizes[-1], mins = inputMinimums, maxes = inputMaximums, means = means, ranges = ranges)
        return layers, data_type, data_type_py, data_format, maxRoad, dict_cst, Normalizer
    else:
        return layers, data_type, data_type_py, data_format, maxRoad, dict_cst