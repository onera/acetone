from pathlib import Path

import numpy as np

from acetone_nnet.generator import (
    ActivationFunctions,
    Add,
    Clip,
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
            size=node.meta['val'].numel(),
            input_shape=node.meta['val'].shape,
            data_format="channels_last" if node.meta['val'].is_contiguous(memory_format=torch.channels_last) else "channels_first",
        )
        self.data_format = this_layer.data_format
        logging.info(f"[LAYOUT] : {self.data_format}")
        self.layerdic[node.name] = (this_layer, node.meta['val'].shape)
        self.idx+=1


    def visit_call_function(self, node: torch.fx.Node):
        target_name = node.target.__name__ if hasattr(node.target, '__name__') else str(node.target)
        self._print_node("FUNC", node, f"call: {target_name}(args={node.args})")
        if "linear" in target_name:
            this_layer = Dense(
                original_name=node.name,
                idx=self.idx,
                size=node.meta['val'].numel(),
                weights=self.module.state_dict()[node.args[1].target].numpy().T,
                biases=self.module.state_dict()[node.args[2].target].numpy(),
                activation_function=Linear()
            )
            self.layerdic[node.args[0].name][0].next_layer.append(this_layer)
            this_layer.previous_layer.append(self.layerdic[node.args[0].name][0])
            self.layerdic[node.name] = (this_layer, node.meta['val'].shape)
            self.idx+=1
        elif "relu" in target_name:
            self.layerdic[node.args[0].name][0].activation_function=ReLu() # set previous layer activation 
            self.layerdic[node.name] = self.layerdic[node.args[0].name] # register 
        elif "clamp" in target_name:
            self.layerdic[node.args[0].name][0].activation_function=Clip( # set previous layer activation 
                max_value=node.args[2],
                min_value=node.args[1]
            )
            self.layerdic[node.name] = self.layerdic[node.args[0].name] # register 
        elif "conv2d" in target_name:
            #first call to conv2d is expected to decide memory layout format based on its input tensor
            if not hasattr(self,"data_format"):
                self.data_format = "channels_last" if self.module.state_dict()[node.args[1].target].is_contiguous(memory_format=torch.channels_last) else "channels_first"
                logging.info(f"[LAYOUT] : {self.data_format}")
            stride = node.args[3][0] if len(node.args) > 3 else 1
            padding = node.args[4] + node.args[4] if len(node.args) > 4 else [0,0,0,0]
            dilation = node.args[5][0] if len(node.args) > 5 else 1
            this_layer = Conv2D(
                    conv_algorithm="specs",
                    original_name=node.name,
                    idx=self.idx,
                    size=node.meta['val'].numel(),
                    padding=padding,
                    strides=stride,
                    kernel_h=self.module.state_dict()[node.args[1].target].shape[-1],
                    kernel_w=self.module.state_dict()[node.args[1].target].shape[-1],
                    dilation_rate=dilation,
                    nb_filters=node.meta['val'].shape[1],
                    input_shape=self.layerdic[node.args[0].name][1],
                    output_shape=node.meta['val'].shape,
                    weights=self.module.state_dict()[node.args[1].target].numpy(),
                    biases=self.module.state_dict()[node.args[2].target].numpy() if node.args[2] is not None else np.zeros(node.meta['val'].shape[1],dtype=np.float32),
                    activation_function=Linear()
                )
            self.layerdic[node.args[0].name][0].next_layer.append(this_layer)
            this_layer.previous_layer.append(self.layerdic[node.args[0].name][0])
            self.layerdic[node.name] = (this_layer, node.meta['val'].shape)
            self.idx+=1
        elif "batch_norm" in target_name:
            # Fusing batch normalization to previous conv2d https://pytorch-cn.com/tutorials/intermediate/fx_conv_bn_fuser.html
            self.layerdic[node.name] = self.layerdic[node.args[0].name]
            bn_w = self.module.state_dict()[node.args[1].target]
            bn_b = self.module.state_dict()[node.args[2].target]
            bn_rm = self.module.state_dict()[node.args[3].target]
            bn_rv = self.module.state_dict()[node.args[4].target]
            bn_eps = node.args[7]
            bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)
            conv_w = torch.from_numpy(self.layerdic[node.args[0].name][0].weights)
            conv_b = torch.from_numpy(self.layerdic[node.args[0].name][0].biases)
            conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 1))
            conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b
            self.layerdic[node.args[0].name][0].weights = conv_w.numpy().astype(np.float32)
            self.layerdic[node.args[0].name][0].biases = conv_b.numpy().astype(np.float32)
        elif "add" in target_name:
            if node.args[0].name in self.layerdic.keys(): # bypass bn2_num_batches_tracked + 1
                this_layer =  Add(
                    original_name=node.name,
                    idx=self.idx,
                    size=node.meta['val'].numel(),
                    input_shapes=[self.layerdic[node.args[i].name][1] for i in range(len(node.args))],
                    output_shape=node.meta['val'].shape,
                    activation_function=Linear(),
                )
                for i in range(len(node.args)):
                    self.layerdic[node.args[i].name][0].next_layer.append(this_layer)
                    this_layer.previous_layer.append(self.layerdic[node.args[i].name][0])
                self.layerdic[node.name] = (this_layer, node.meta['val'].shape)
                self.idx+=1
            else:
                logging.info(f"Add layer bypassed: {node.args[0].name}")
        elif "cat" in target_name:
            this_layer = Concatenate(
                    original_name=node.name,
                    idx=self.idx,
                    size=node.meta['val'].numel(),
                    axis=node.args[1],
                    input_shapes=[self.layerdic[node.args[0][i].name][1] for i in range(len(node.args[0]))],
                    output_shape=node.meta['val'].shape,
                    activation_function=Linear(),
                )
            for i in range(len(node.args[0])):
                self.layerdic[node.args[0][i].name][0].next_layer.append(this_layer)
                this_layer.previous_layer.append(self.layerdic[node.args[0][i].name][0])
            self.layerdic[node.name] = (this_layer, node.meta['val'].shape)
            self.idx+=1
        elif "dropout" in target_name:
            self.layerdic[node.name] = self.layerdic[node.args[0].name]
        elif "flatten" in target_name:
            self.layerdic[node.name] = self.layerdic[node.args[0].name]
        elif "max_pool2d" in target_name:
            kernel = node.args[1][0] if len(node.args) > 1 else 1
            stride = node.args[2][0] if len(node.args) > 2 else 1
            padding = node.args[3] + node.args[3] if len(node.args) > 3 else [0,0,0,0]
            dilation = node.args[4][0] if len(node.args) > 4 else 1
            this_layer = MaxPooling2D(
                    original_name=node.name,
                    idx=self.idx,
                    size=node.meta['val'].numel(),
                    padding=padding,
                    strides=stride,
                    pool_size=kernel,
                    input_shape=self.layerdic[node.args[0].name][1],
                    output_shape=node.meta['val'].shape,
                    activation_function=Linear(),
                    data_format=self.data_format
                )            
            self.layerdic[node.args[0].name][0].next_layer.append(this_layer)
            this_layer.previous_layer.append(self.layerdic[node.args[0].name][0])
            self.layerdic[node.name] = (this_layer, node.meta['val'].shape)
            self.idx+=1
        elif "adaptive_avg_pool2d" in target_name:
            this_layer = AveragePooling2D(
                original_name=node.name,
                idx=self.idx,
                size=node.meta['val'].numel(),
                padding=[0, 0, 0, 0],
                strides=1,
                pool_size=self.layerdic[node.args[0].name][1][-1],
                input_shape=self.layerdic[node.args[0].name][1],
                output_shape=node.meta['val'].shape,
                activation_function=Linear(),
                data_format=self.data_format
            )
            self.layerdic[node.args[0].name][0].next_layer.append(this_layer)
            this_layer.previous_layer.append(self.layerdic[node.args[0].name][0])
            self.layerdic[node.name] = (this_layer, node.meta['val'].shape)
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
    layers, max_road, dict_cst = graph_interpretor.tri_topo([l[0] for l in visitor.layerdic.values()])
    layers = [x.find_output_str(dict_cst) for x in layers]
    print("Finished model initialization.")    
    return layers, "float", np.float32, visitor.data_format,max_road, dict_cst