"""Generic layer versioning manager.

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
from acetone_nnet.generator import Layer
from acetone_nnet.versioning.layer_factories import LayerFactory, implemented


def register_factory(name: str, factory: LayerFactory) -> None:
    """Register a new Layer factory."""
    if name in implemented:
        msg = f"Factory for layer {name} already exists."
        raise KeyError(msg)
    implemented[name] = factory

def list_all_implementations() -> dict[str, list[str]]:
    implem = {}
    for layer_name in implemented:
        implem[layer_name] = implemented[layer_name].list_implementations
        
    return implem
    



def versioning(
        layers: list[Layer],
        version: dict[int, str],
) -> list[Layer]:
    """Check layers and change the layer version if needed."""
    keys = list(version.keys())
    for idx in keys:
        for j in range(len(layers)):
            if layers[j].idx == idx:
                layer = layers[j]

                layer = implemented[layer.name](layer, version[idx])
                layer.path = layers[j].path
                layer.next_layer = layers[j].next_layer
                layer.previous_layer = layers[j].previous_layer
                layer.sorted = layers[j].sorted
                layer.output_str = layers[j].output_str
                layer.fused_layer = layers[j].fused_layer

                layers[j] = layer

    return layers
