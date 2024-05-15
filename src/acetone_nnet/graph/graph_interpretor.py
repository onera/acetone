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

from ..code_generator.Layer import Layer

#Sort the graph (list of nodes) based on the topological sort
def tri_topo(dnn:list):
    list_path = []
    #The sorted list to be returned
    list_layers = []
    #the dict stating which layers need to go in a constant
    dict_cst = {}
    for layer in dnn:
        #If the node isn't sorted, we sort it
        if layer.sorted == None:
            parcours_prof_topo(list_layers, layer)
    for layer in list_layers:
        updatepath(layer,list_path)
        to_save(layer,dict_cst)
    max_path = list_path[-2] + 1
    return list_layers, list_path, max_path, dict_cst

#Compute the sorting of a node
def parcours_prof_topo(list_layers:list, layer:Layer):
    #The node is currently being sorted
    layer.sorted = 1
    for nxt_layer in layer.next_layer:
        #The next_layer need to be found and sorted before the current node
        if nxt_layer.sorted == None:
            parcours_prof_topo(list_layers, nxt_layer)
    #The node is sorted
    layer.sorted = 0
    #And is added to the list of sorted nodes
    list_layers.insert(0,layer)


#The function to open a new path
def setNewpath(layer:Layer, listCurrentpaths:list):
    #if the list isn't in the right format, it return -1
    if(len(listCurrentpaths)%2!=0):
        print("Error: listCurrentpaths must be in format:[name_path, is_path_closed, ...]")
        return -1
    else:
        N = len(listCurrentpaths)//2
        #check if there is a previously closed path avialable 
        for i in range(N):
            #is the path is closed, we open it 
            if (listCurrentpaths[2*i+1]==1):
                layer.path = listCurrentpaths[2*i]
                listCurrentpaths[2*i+1] = 0
                break
        #if there is no path at all, we create the first one
        if(N == 0):
            listCurrentpaths.append(0)
            listCurrentpaths.append(0)
            layer.path = 0
        #if no previously closed path can fit are avialable, we create a new one
        elif(layer.path == None):
            listCurrentpaths.append(listCurrentpaths[-2]+1)
            listCurrentpaths.append(0)
            layer.path = listCurrentpaths[-2]
        
#give a layer the right path    
def updatepath(layer:Layer, listCurrentpaths:list):
    if(layer.previous_layer == []):
        #if the layer has no previous one, we creat a new path
        setNewpath(layer,listCurrentpaths)

    given = False
    #every next_layer need to have a path 
    for nxt_layer in layer.next_layer:
        if ((len(nxt_layer.previous_layer)==1) and (not given)):
            #if the next layer only have one prev_layer, and the path hadn't already be given, it receive the same path
            nxt_layer.path = layer.path
            given = True
        elif(nxt_layer.path != None):
            #if the layer already has a path, we do nothing
            pass
        elif(nxt_layer == layer.next_layer[0]):
            #by convention, if it's the first child amongst other, it receive the same path as the father
            nxt_layer.path = layer.path
            given = True
        else:
            #in any other case, the next layer receive a new path
            setNewpath(nxt_layer,listCurrentpaths)
    
    #if the path isn't given, it is closed
    if (given == False):
        listCurrentpaths[layer.path*2 + 1] = 1

#Function creating the dict {idx_layer:idx_cst} saying if a layer must be stored
def to_save(layer:Layer, dict_cst:dict):
    for parent in layer.previous_layer:
        if(parent in dict_cst):
            #if the previous_layer are in the dict, we add one to the number of next_layer already "taken care of"
            parent.sorted+=1

    if((len(layer.next_layer)>1)):
        #if the layer has more than one child, it must be stored.
        if(len(dict_cst) == 0):
            dict_cst[layer] = 1 #if the dict is empty, we create the first cst

        else:
            given = False
            #Going through the dict, starting from the end (the opened cst are at the end of the dict)
            for i in range(len(dict_cst)-1,-1,-1): 
                #extracting the layer at the i-th position 
                past_layer = list(dict_cst.keys())[i] 
                #if the layer is complete, we can re-use the same constant
                if (past_layer.sorted == len(past_layer.next_layer)):  
                    past_layer.sorted = 0
                    dict_cst[layer] = dict_cst[past_layer]
                    given = True
                    break
            if not given:# if no constant have been attributed, we create a new one
                dict_cst[layer] = list(dict_cst.values())[-1] + 1