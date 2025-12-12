from pathlib import Path

import numpy as np

from acetone_nnet.generator import (
    ActivationFunctions,
    Add,
    Average,
    AveragePooling2D,
    BatchNormalization,
    Concatenate,
    ConstantPad,
    Conv2D,
    Dense,
    Flatten,
    InputLayer,
    LeakyReLu,
    Linear,
    Maximum,
    MaxPooling2D,
    Minimum,
    Multiply,
    ReLu,
    ResizeCubic,
    ResizeLinear,
    ResizeNearest,
    Sigmoid,
    Softmax,
    Subtract,
    TanH,
)
from acetone_nnet.graph import graph_interpretor
from acetone_nnet.ir import Layer

from torch.export import export, ExportedProgram
import logging


import torch
import torch.export
import torch.nn as nn
from typing import Any, Union, Tuple, List

# 1. Base Visitor Class (Standard FX Traversal)
class FXGraphVisitor:
    def visit(self, graph_module: torch.fx.GraphModule):
        self.module = graph_module
        for node in graph_module.graph.nodes:
            self.visit_node(node)

    def visit_node(self, node: torch.fx.Node):
        visit_method_name = f'visit_{node.op}'
        visit_method = getattr(self, visit_method_name, self.generic_visit)
        return visit_method(node)

    def generic_visit(self, node: torch.fx.Node): pass
    def visit_placeholder(self, node: torch.fx.Node): pass
    def visit_call_method(self, node: torch.fx.Node): pass
    def visit_call_module(self, node: torch.fx.Node): pass
    def visit_call_function(self, node: torch.fx.Node): pass
    def visit_get_attr(self, node: torch.fx.Node): pass
    def visit_output(self, node: torch.fx.Node): pass



# 2. Concrete Visitor: ShapeAwareVisitor
class ShapeAwareVisitor(FXGraphVisitor):
    """
    A visitor that prints the execution trace alongside
    inferred tensor shapes and data types derived from node.meta['val'].
    """
    def __init__(self):
        self.indent = "  "
        self.idx=0
        self.layerdic: dict[str:Layer] = {}
    def _format_shape(self, val: Any) -> str:
        """
        Recursively extracts shape and dtype from FakeTensors stored in meta['val'].
        """
        # Case 1: It's a Tensor (or FakeTensor)
        if isinstance(val, torch.Tensor):
            dtype_str = str(val.dtype).replace('torch.', '')
            return f"Shape: {tuple(val.shape)} | {dtype_str}"
        
        # Case 2: It's a sequence (e.g., multiple outputs)
        elif isinstance(val, (list, tuple)):
            inner = [self._format_shape(v) for v in val]
            return f"Tuple({', '.join(inner)})"
        
        # Case 3: Primitives (int, float, etc.)
        elif isinstance(val, (int, float, bool)):
            return f"Scalar({val})"
            
        return "No Shape Info"

    def _get_node_info(self, node: torch.fx.Node) -> str:
        """Helper to get the shape info string from a node."""
        if 'val' in node.meta:
            return self._format_shape(node.meta['val'])
        return "Unknown"

    def _print_node(self, kind: str, node: torch.fx.Node, extra: str = ""):
        shape_info = self._get_node_info(node)
        
        # Formatting for alignment
        node_name = f"%{node.name}"
        left_col = f"{self.indent}[{kind}] {node_name:<15} {extra}"
        logging.info(f"{left_col:<60} -> {shape_info}")

    # --- Visitor Implementation ---

    def visit(self, graph_module: torch.fx.GraphModule):
        logging.info(f"{'OPERATION':<60}    {'INFERRED SHAPE / DTYPE'}")
        logging.info("-" * 100)
        super().visit(graph_module)
        logging.info("-" * 100)

    def visit_placeholder(self, node: torch.fx.Node):
        target_name = node.target.__name__ if hasattr(node.target, '__name__') else str(node.target)
        self._print_node("INPUT", node, f"call: {target_name}(args={node.args})")
        this_layer = InputLayer(
            original_name=node.name,
            idx=self.idx,
            size=node.meta['val'].shape[0],
            input_shape=[1,1,1,node.meta['val'].shape[0]],
            data_format="channels_first",
        )
        self.layerdic[node.name] = this_layer
        self.idx+=1


    def visit_call_function(self, node: torch.fx.Node):
        target_name = node.target.__name__ if hasattr(node.target, '__name__') else str(node.target)
        self._print_node("FUNC", node, f"call: {target_name}(args={node.args})")
        if "linear" in target_name:
            this_layer = Dense(
                original_name=node.name,
                idx=self.idx,
                size=node.meta['val'].shape[0],
                weights=self.module.state_dict()[node.args[1].target].numpy().T,
                biases=self.module.state_dict()[node.args[2].target].numpy(),
                activation_function=Linear()
            )
            self.layerdic[node.args[0].name].next_layer.append(this_layer)
            this_layer.previous_layer.append(self.layerdic[node.args[0].name])
            self.layerdic[node.name] = this_layer
        elif "relu" in target_name:
            self.layerdic[node.args[0].name].activation_function=ReLu()
            self.layerdic[node.name] = self.layerdic[node.args[0].name]
        self.idx+=1

    def visit_call_module(self, node: torch.fx.Node):
        self._print_node("MODULE", node, f"mod: {node.target}")

    def visit_call_method(self, node: torch.fx.Node):
        self._print_node("METHOD", node, f"obj.{node.target}(args={node.args})")

    def visit_get_attr(self, node: torch.fx.Node):
        # Attributes usually identify parameters/buffers, which also have shapes in ExportedProgram
        self._print_node("ATTR", node, f"get: {node.target}")

    def visit_output(self, node: torch.fx.Node):
        # The output node usually contains the return value in args[0]
        # We look at the meta of the output node itself if available, or the inputs
        logging.info(f"{self.indent}[OUTPUT] return {node.args[0]}")



def load_pytorch(program : ExportedProgram):
    visitor = ShapeAwareVisitor()
    visitor.visit(program.module())
    layers, max_road, dict_cst = graph_interpretor.tri_topo(visitor.layerdic.values())
    layers = [x.find_output_str(dict_cst) for x in layers]
    print("Finished model initialization.")    
    return layers, "float", np.float32, "channels_first",max_road, dict_cst