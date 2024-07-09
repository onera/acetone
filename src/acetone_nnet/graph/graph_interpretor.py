"""Topological sort for Graph.

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

from acetone_nnet.code_generator.Layer import Layer


def tri_topo(dnn: list):
    """Sort the graph (list of nodes) topologically."""
    paths: dict[int, bool] = {}  # Available paths per id
    # The sorted layer list
    sorted_layers: list[Layer] = []
    # the dict stating which layers need to go in a constant
    dict_cst: dict[int, int] = {}
    for layer in dnn:
        # If the node isn't sorted, we sort it
        if layer.sorted is None:
            parcours_prof_topo(sorted_layers, layer)
    for layer in sorted_layers:
        update_path(layer, paths)
        # FIXME Assess what to_save is supposed to do
        #   to_save(layer, dict_cst)
    # Convert paths
    max_path = max(paths.keys()) + 1
    list_path: list[int] = []
    for path, available in paths.items():
        list_path.append(path)
        list_path.append(1 if available else 0)
    return sorted_layers, list_path, max_path, dict_cst


def parcours_prof_topo(sorted_layers: list[Layer], layer: Layer) -> None:
    """Sort layer in list of sorted layers."""
    # The node is currently being sorted
    layer.sorted = 1
    for nxt_layer in layer.next_layer:
        # The next_layer need to be found and sorted before the current node
        if nxt_layer.sorted is None:
            parcours_prof_topo(sorted_layers, nxt_layer)
    # The node is sorted
    layer.sorted = 0
    # And is added to the list of sorted nodes
    sorted_layers.insert(0, layer)


def allocate_path_to_layer(layer: Layer, paths: dict[int, bool]) -> None:
    """Allocate a new or closed path to layer."""
    # Check if there is a previously closed path available
    for path, available in paths.items():
        if available:
            layer.path = path
            paths[path] = False
            break
    else:
        # If there is no available path, we create a new one
        path = len(paths)
        paths[path] = False
        layer.path = path


def update_path(layer: Layer, paths: dict[int, bool]) -> None:
    """Assign path to layer."""
    if len(layer.previous_layer) == 0:
        # if the layer has no previous one, we create a new path
        allocate_path_to_layer(layer, paths)

    # Every subsequent layer needs to have a path
    given = False
    if len(layer.next_layer) > 0:
        # Allocate the path to the first child, if pathless
        if layer.next_layer[0].path is None:
            layer.next_layer[0].path = layer.path
            given = True
        # Allocate paths to other children
        if len(layer.next_layer) > 1:
            for nxt_layer in layer.next_layer[1:]:
                if len(nxt_layer.previous_layer) == 1 and not given:
                    # Allocate path to child with a single parent, if available
                    nxt_layer.path = layer.path
                    given = True
                elif nxt_layer.path is not None:
                    # if the layer already has a path, we do nothing
                    pass
                else:
                    # in any other case, the next layer receive a new path
                    allocate_path_to_layer(nxt_layer, paths)

    # Close path if not allocated to child
    if not given:
        if layer.path is None:
            msg = "Layer has no path"
            raise AssertionError(msg)
        paths[layer.path] = True


def to_save(layer: Layer, dict_cst: dict[int, int]) -> None:
    """Create the dict {idx_layer:idx_cst} saying if a layer must be stored."""
    for parent in layer.previous_layer:
        if parent in dict_cst:
            # if the previous_layer are in the dict, we add one to the number of next_layer already "taken care of"
            parent.sorted += 1

    if len(layer.next_layer) > 1:
        # if the layer has more than one child, it must be stored.
        if len(dict_cst) == 0:
            dict_cst[layer.idx] = 1  # if the dict is empty, we create the first cst

        else:
            given = False
            # Going through the dict, starting from the end (the opened cst are at the end of the dict)
            for i in range(len(dict_cst) - 1, -1, -1):
                # extracting the layer at the i-th position
                past_layer = list(dict_cst.keys())[i]
                # if the layer is complete, we can re-use the same constant
                if past_layer.sorted == len(past_layer.next_layer):
                    past_layer.sorted = 0
                    dict_cst[layer.idx] = dict_cst[past_layer]
                    given = True
                    break
            if not given:  # if no constant have been attributed, we create a new one
                dict_cst[layer.idx] = list(dict_cst.values())[-1] + 1
