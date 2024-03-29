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
import onnx
from format_importer.ONNX_importer.create_layer import *
from graph.graph_interpretor import tri_topo

def load_onnx(file_to_parse, conv_algorithm):
    #Loading the model and adding value_info if it's not already in it 
    model = onnx.load(file_to_parse)
    if (not model.graph.value_info):
        model = onnx.shape_inference.infer_shapes(model)
    
    #The list of layers
    layers = []
    #Dictionnary of the outputs of each layers: {output_name:layer_idx_from_which_the_output_come_from}
    dict_output = {}
    #Dictionnary of the inputs of each layers: {layer_idx_to_which_the_inputs_go:[inputs_name]}
    dict_input = {}
    #Indice of the layer
    idx = 0

    #Creating and adding all the input layers to the list
    for input_layer in model.graph.input:
        layers.append(create_Input_Layer(input_layer,idx,dict_output))
        idx+=1
    
    #Going through all the nodes to creat the layers and add them to the list
    for node in model.graph.node:
        if(node.op_type in layer_type): #If the node is a useful layer, we add it to the list
            if(node.op_type == "Conv"):    
                layers.append(layer_type[node.op_type](node,idx,dict_input,dict_output,model,conv_algorithm))
                idx+=1
            else:
                layers.append(layer_type[node.op_type](node,idx,dict_input,dict_output,model))
                idx+=1
        elif(node.op_type in unused_layers): #If the node is a layer only used in trainning, we do layer_input=layer_output
            unused_layers[node.op_type](node,dict_output,model)
        elif(node.op_type in activation_layers):
            activation_layers[node.op_type](node,dict_output,model,layers)
        else:
            raise TypeError("Error: layer "+node.op_type+" not supported\n")
    
    for layer_idx in dict_input: #Going through all the indices in the list
        layer = layers[layer_idx] #The indice of a layer is the same a it's position in the list
        for input_name in dict_input[layer_idx]: #Going through all the inputs to that layer
            parent = layers[dict_output[input_name]] #Localising that input in the output dictionnary to find the indice of the parent layer
            layer.previous_layer.append(parent)
            parent.next_layer.append(layer)
            
    
    data_type_py = onnx.helper.tensor_dtype_to_np_dtype(model.graph.input[0].type.tensor_type.elem_type)
    
    if data_type_py == 'float64':
        data_type = 'double'

    elif data_type_py == 'float32':
        data_type = 'float'

    elif data_type_py == 'int':
        data_type = 'long int'
    
    #Sorting the graph and adding the road to the layers
    layers, listRoad, maxRoad, dict_cst = tri_topo(layers)
    layers = list(map(lambda x:x.find_output_str(dict_cst), layers))
    
    print("Finished model initialization.")    
    return layers, data_type, data_type_py, listRoad, maxRoad, dict_cst
