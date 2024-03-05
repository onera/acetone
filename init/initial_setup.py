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

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
from JSON_from_keras_model import JSON_from_keras_model
import numpy as np
import tensorflow as tf
from tensorflow import keras

tf.get_logger().setLevel('ERROR')
tf.keras.backend.set_floatx('float32')

### Define a sequential model recreating the LeNet-5 architecture ###

bias_initializer  = tf.keras.initializers.RandomUniform(minval=-0.2, maxval=0.2, seed=None)

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(filters=6, kernel_size=(5,5), strides=(1,1), padding='valid', dilation_rate=(1, 1), activation='tanh', 
                                kernel_initializer='glorot_uniform', bias_initializer=bias_initializer, input_shape=(28,28,1)))
model.add(keras.layers.AvgPool2D(pool_size=(2,2)))
model.add(keras.layers.Conv2D(filters=16, kernel_size=(5,5), strides=(1,1), padding='valid', dilation_rate=(1, 1), activation='tanh',
                                kernel_initializer='glorot_uniform', bias_initializer=bias_initializer))
model.add(keras.layers.AvgPool2D(pool_size=(2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=120, activation='tanh', kernel_initializer='glorot_uniform', bias_initializer=bias_initializer))
model.add(keras.layers.Dense(units=84, activation='tanh', kernel_initializer='glorot_uniform', bias_initializer=bias_initializer))
model.add(keras.layers.Dense(units=10, activation='softmax', kernel_initializer='glorot_uniform', bias_initializer=bias_initializer))

print("LeNet-5 architecture created in Keras' framework. Random weights and biases are set.")

### Save neural network model in h5 format to be used in Keras2C framework ###
model.save('../data/example/lenet5.h5')
print("Neural network model exported to ../data/example/lenet5.h5")

### Save neural network model in JSON format to be used in ACETONE framework ###
JSON_from_keras_model(model, '../data/example/lenet5.json')
print("Neural network model exported to ../data/example/lenet5.json")

### Define a random input of shape (1,1,28*28*1) ###
random_input = np.random.default_rng().random((1,28, 28, 1))
with open('../data/example/test_input_lenet5.txt', 'w') as filehandle:
    for i in range(random_input.shape[0]):
        row = (random_input[i]).flatten(order='C')
        json.dump(row.tolist(), filehandle)
        filehandle.write('\n')
filehandle.close()
print("Sample input data exported to ../data/example/test_input_lenet5.txt")

### Perform inference with Keras ###
keras_prediction = model.predict(random_input)

### Print Keras' output ###
print("Keras' inference output: \n", keras_prediction)

### Export Keras' output ###
with open('../data/example/output_keras.txt', 'w+') as fi:
    for i in range(len(keras_prediction)):
        keras_prediction[i] = np.reshape(keras_prediction[i], -1)
        for j in range(len(keras_prediction[i])):
            print('{:.9g}'.format(keras_prediction[i][j]), end=' ', file=fi, flush=True)           
        print(" ",file=fi)     
fi.close()

print("Keras' inference output exported to ../data/example/output_keras.txt")



