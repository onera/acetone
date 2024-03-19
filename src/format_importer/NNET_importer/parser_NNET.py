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


import tensorflow as tf
from tensorflow import keras
import numpy as np
from src.code_generator.layers.Dense import Dense
from src.code_generator.layers.Input import InputLayer
from src.code_generator.activation_functions import Linear, ReLu

tf.keras.backend.set_floatx('float32')


def load_nnet(file_to_parse, normalize):
    """
    Reads nnet networks from ACAS project and to pull it in .h5 format
    Inspired from : # https://github.com/NeuralNetworkVerification/Marabou/blob/master/maraboupy/MarabouNetworkNNet.py  read function
    Args:
        model_file_net (.nnet): file describing the neural network
    Returns: 
        model (.h5): Keras model
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
    listRoad = [0,1]
    maxRoad = 1
    dict_cst = {}
    data_type = 'float'
    data_type_py = 'float32'

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


        for i in range(5): # skip the next five lines 
            next(f)
        
        weights = []
        biases = []

        for layernum in range(numLayers):
            previousLayerSize = layerSizes[layernum]
            currentLayerSize = layerSizes[layernum + 1]
            weights.append([])
            biases.append([])
            
            # weights
            # for i in range(previousLayerSize):
            #     line = f.readline()
            #     aux = [float(x) for x in line.strip().split(",")[:-1]]
            #     weights[layernum].append([])
            #     for j in range(currentLayerSize):
            #         weights[layernum][i].append(aux[j])

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

    layers.append(InputLayer(idx=0, size=layerSizes[0]))
    layers[0].find_output_str(dict_cst)
    layers[0].road = 0

    # list of lists -> list of arrays
    weights = [np.transpose(np.array(weight)) for weight in weights]
    biases = [np.transpose(np.array(bias)) for bias in biases]

    for i in range(1,numLayers+1):

        weight = weights[i]
        if(len(weight.shape) < 3):
            for i in range(3-len(weight.shape)): 
                weight = np.expand_dims(weight, axis=-1)
        weight = np.moveaxis(weight, 2, 0)

        if(i == numLayers):
            layer = Dense(idx=i,
                        size=layerSizes[i],
                        weights=weight,
                        biases=biases[i],
                        activation_function=Linear)
        else:
            layer = Dense(idx=i,
                        size=layerSizes[i],
                        weights=weight,
                        biases=biases[i],
                        activation_function=ReLu)
        layer.find_output_str(dict_cst)
        layer.road = 0

        layers.append(layer)
    
    print("Finished model initialization.")

    if(normalize):
        return layers, data_type, data_type_py, listRoad, maxRoad, dict_cst, inputMaximums, inputMinimums, means, ranges
    else:
        return layers, data_type, data_type_py, listRoad, maxRoad, dict_cst