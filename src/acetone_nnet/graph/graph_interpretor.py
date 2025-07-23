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

from collections import deque

from acetone_nnet.ir import Layer


def tri_topo(dnn: list) -> tuple[list[Layer], int, dict[int, int]]:
    """Sort the graph (list of nodes) topologically."""
    paths: dict[int, bool] = {}  # Available paths per id
    # The sorted layer list
    sorted_layers: list[Layer] = []
    # the dict stating which layers need to go in a constant
    dict_cst: dict[int, int] = {}
    # Tag layers as they are visited
    visited_layers: set[int] = set()
    # Sort layer topologically
    for layer in dnn:
        # If the node isn't sorted, we sort it
        if layer.idx not in visited_layers:
            parcours_prof_topo(sorted_layers, layer, visited_layers)
    # Allocate temp variable to layers
    assign_liveliness_index(sorted_layers)
    dict_cst = assign_cst(sorted_layers)
    # Convert paths
    max_path = max(i.path for i in sorted_layers) + 1
    return sorted_layers, max_path, dict_cst


def parcours_prof_topo(
    sorted_layers: list[Layer],
    layer: Layer,
    visited_layers: set[int],
) -> None:
    """Sort layer in list of sorted layers."""
    # The node is currently being sorted
    visited_layers.add(layer.idx)
    for nxt_layer in layer.next_layer:
        # The next_layer need to be found and sorted before the current node
        if nxt_layer.idx not in visited_layers:
            parcours_prof_topo(sorted_layers, nxt_layer, visited_layers)
    # And is added to the list of sorted nodes
    sorted_layers.insert(0, layer)


def assign_liveliness_index(execution_order: list[Layer]) -> dict[int, int]:
    """Assign unique liveliness index to nodes during execution.

    A node is assigned a unique index while it is lively, that is while its
    output may be required for the execution of a future node. At any point
    during execution, all lively nodes will be uniquely identified. The
    method minimises the overall number of allocated indexes.

    :param execution_order: The order in which nodes will be executed. Nodes
                            must be topologically sorted.
    :return: A map of the allocated index to each node.
    """
    index_count: int = 0
    available_indexes: deque[int] = deque()
    indexes: dict[int, int] = {}
    # Allocate index to nodes in execution order
    for n in execution_order:
        assert n.idx not in indexes
        # Free the indices of all predecessors' outputs after use
        for p in n.previous_layer:
            assert p.idx in indexes
            # Check all consumers of a tensor have been executed (except the current one)
            if all(s.idx in indexes for s in p.next_layer if s.idx != n.idx):
                available_indexes.appendleft(indexes[p.idx])
        # TODO Assign index before liberating predecessors' outputs
        # Reuse any available index for the node, or create a new one if none
        if len(available_indexes) == 0:
            available_indexes.append(index_count)
            index_count += 1
        indexes[n.idx] = available_indexes.popleft()
        # FIXME Perform allocation outside method
        n.path = indexes[n.idx]
    return indexes


def assign_cst(
    execution_order: list[Layer],
) -> dict[int, int]:
    dict_cst: dict[int, int] = {}
    visited_layers: set[int] = set()
    index_count: int = 0
    available_indexes: deque[int] = deque()
    for layer in execution_order:
        visited_layers.add(layer.idx)
        # Free cst if all successors have been visited
        for parent in layer.previous_layer:
            if parent.idx in dict_cst and all(
                s.idx in visited_layers for s in parent.next_layer
            ):
                available_indexes.appendleft(dict_cst[parent.idx])
        # Allocate cst if layer has more than one child
        if len(layer.next_layer) > 1:
            if len(available_indexes) == 0:
                available_indexes.append(index_count)
                index_count += 1
            dict_cst[layer.idx] = available_indexes.popleft()
    return dict_cst
