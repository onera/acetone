"""Base Pattern type definition.

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

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import Self

from acetone_nnet.ir import Layer


@dataclass
class Pattern(ABC):
    """Base class for patterns."""

    def __init__(self: Self, name: str, pattern: str, shift: int) -> None:
        """Build a non-specific pattern."""
        super().__init__()
        self.name = name
        self.pattern = pattern
        self.shift = shift

    @abstractmethod
    def is_pattern(self: Self, layer: Layer) -> bool:
        """Check if the layer is the root of the pattern."""

    @abstractmethod
    def apply_pattern(
        self: Self,
        index: int,
        layers: list[Layer],
        dict_cst: dict[int, int],
    ) -> tuple[str, int]:
        """Apply the pattern to the layer."""

def update_indices(
        index: int,
        layers: list[Layer],
        shift: int,
        dict_cst: dict[int,int],
) -> None:
    """Update the indices of the layers after index by shifting them of shift."""
    nb_layers = len(layers)
    ref = layers[index].idx

    if index >= nb_layers:
        m = f"Index out of bound ({index} > {nb_layers})"
        raise ValueError(m)
    if shift > index:
        m = f"Shift can't superior to the index ({shift} > {index})"
        raise ValueError(m)

    for i in range(nb_layers):
        if layers[i].idx > ref:
            cst_index = dict_cst.get(layers[i].idx, None)
            if cst_index is not None:
                dict_cst.pop(layers[i].idx)
                dict_cst[layers[i].idx - shift] = cst_index
            layers[i].idx = layers[i].idx - shift

def update_next_layers(removed_layer: Layer, replacement_layer:Layer) -> None:
    """Link the layers in removed_layer.next_layer to the replacement_layer as input."""
    for next_layer in removed_layer.next_layer:
        if next_layer not in replacement_layer.next_layer:
            replacement_layer.next_layer.append(next_layer)
        update = True
        while update:
            try:
                pos = next_layer.previous_layer.index(removed_layer)
                next_layer.previous_layer[pos]= replacement_layer
            except ValueError:
                update = False

def update_previous_layers(removed_layer: Layer, replacement_layer:Layer) -> None:
    """Link the layers in removed_layer.previous_layer to take the replacement_layer as output."""
    for prev_layer in removed_layer.previous_layer:
        if prev_layer not in replacement_layer.previous_layer:
            replacement_layer.previous_layer.append(prev_layer)
        update = True
        while update:
            try:
                pos = prev_layer.next_layer.index(removed_layer)
                prev_layer.next_layer[pos]= replacement_layer
            except ValueError:
                update = False


def update_dict_cst(removed_layer: Layer, replacement_layer:Layer, dict_cst:dict[int,int]) -> None:
    """Update dict_cst."""
    replacement_layer.output_str = removed_layer.output_str
    cst_index = dict_cst.get(removed_layer.idx, None)
    if cst_index is not None:
        dict_cst[replacement_layer.idx] = cst_index
        dict_cst.pop(removed_layer.idx)
