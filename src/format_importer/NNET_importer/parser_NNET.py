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

tf.keras.backend.set_floatx('float32')


def load_nnet(model_file_nnet):
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
    #biais 

    print("\n Reading the file : "+model_file_nnet+"\n")

    with open(model_file_nnet) as f:
     
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

    # list of lists -> list of arrays
    weights = [np.array(weight) for weight in weights]
    biases = [np.array(bias) for bias in biases]

    keras_weights = []
    # in here we append the weights of each layer followed by the biases of the same layer until numLayers, to be keras compliant 
    for i in range(len(weights)):
        keras_weights.append(weights[i])
        keras_weights.append(biases[i]) 

    
    # weights and biases are both lists of lists, we need a list of arrays
    keras_weights = [np.transpose(np.array(weight)) for weight in keras_weights] # for non conventional nnet
    # keras_weights = [(np.array(weight)) for weight in keras_weights]
    
   
    # create keras model 
    model = keras.Sequential()

    model.add(keras.Input(shape=(layerSizes[0],)))    
    # define every layer size and actv function (relu for hidden layers, linear for output layer)
    for i in range(1,numLayers+1):
        if i == numLayers:
            model.add(keras.layers.Dense(layerSizes[i], activation='linear'))
        else:
            model.add(keras.layers.Dense(layerSizes[i], activation='relu'))

    model.set_weights(keras_weights)
    print(" Model translated from .nnet to .h5 ")

    return model