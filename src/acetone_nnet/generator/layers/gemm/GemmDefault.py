"""Gemm Layer with default implementation type definition.

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

import pystache
from typing_extensions import Any, Self

from acetone_nnet.versioning.layer_factories import gemm_factory

from .Gemm import Gemm


class GemmDefault(Gemm):
    """Gemm layer with default implementation class."""

    def __init__(self: Self, version: str, **kwargs: Any) -> None:
        """Build a Gemm Layer with default implementation."""
        super().__init__(**kwargs)
        self.version = version

        self.algo_gemm_mapping = {(0, 0): self.write_gemm_nn,
                                  (0, 1): self.write_gemm_nt,
                                  (1, 1): self.write_gemm_tt,
                                  (1, 0): self.write_gemm_tn}

    # The various ways to compute the operation:

    # None of the tensor ar transposed
    def write_gemm_nn(
            self: Self,
            m: int,
            n: int,
            k: int,
            a: str,
            b: str,
    ) -> str:
        """Generate computation code for nn gemm."""
        mustach_hash = {}

        mustach_hash["name"] = self.name
        mustach_hash["idx"] = f"{self.idx:02d}"
        mustach_hash["m"] = m
        mustach_hash["n"] = n
        mustach_hash["k"] = k
        mustach_hash["A"] = a
        mustach_hash["B"] = b
        mustach_hash["activation_function"] = self.activation_function.write_activation_str("output")
        mustach_hash["alpha"] = self.alpha
        mustach_hash["beta"] = self.beta
        if self.fused_layer:
            mustach_hash["fused_layer"] = self.fused_layer.write_activation_str(
                "output",
                self.idx,
                f"i*{self.ldC} + j")

            if self.activation_function.name == "linear":
                mustach_hash["linear"] = True

        with open(self.template_path / "layers" / "Gemm" / "template_gemm_nn.c.tpl") as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)

    def write_gemm_nt(
            self: Self,
            m: int,
            n: int,
            k: int,
            a: str,
            b: str,
    ) -> str:
        """Generate computation code for nt gemm."""
        mustach_hash = {}

        mustach_hash["name"] = self.name
        mustach_hash["idx"] = f"{self.idx:02d}"
        mustach_hash["m"] = m
        mustach_hash["n"] = n
        mustach_hash["k"] = k
        mustach_hash["A"] = a
        mustach_hash["B"] = b
        mustach_hash["activation_function"] = self.activation_function.write_activation_str("output")
        mustach_hash["alpha"] = self.alpha
        mustach_hash["beta"] = self.beta
        if self.fused_layer:
            mustach_hash["fused_layer"] = self.fused_layer.write_activation_str(
                "output",
                self.idx,
                f"i*{self.ldC} + j")

            if self.activation_function.name == "linear":
                mustach_hash["linear"] = True

        with open(self.template_path / "layers" / "Gemm" / "template_gemm_nt.c.tpl") as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)

    def write_gemm_tn(
            self: Self,
            m: int,
            n: int,
            k: int,
            a: str,
            b: str,
    ) -> str:
        """Generate computation code for tn gemm."""
        mustach_hash = {}

        mustach_hash["name"] = self.name
        mustach_hash["idx"] = f"{self.idx:02d}"
        mustach_hash["m"] = m
        mustach_hash["n"] = n
        mustach_hash["k"] = k
        mustach_hash["A"] = a
        mustach_hash["B"] = b
        mustach_hash["activation_function"] = self.activation_function.write_activation_str("output")
        mustach_hash["alpha"] = self.alpha
        mustach_hash["beta"] = self.beta
        if self.fused_layer:
            mustach_hash["fused_layer"] = self.fused_layer.write_activation_str(
                "output",
                self.idx,
                f"i*{self.ldC} + j")

            if self.activation_function.name == "linear":
                mustach_hash["linear"] = True

        with open(self.template_path / "layers" / "Gemm" / "template_gemm_tn.c.tpl") as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)

    def write_gemm_tt(
            self: Self,
            m: int,
            n: int,
            k: int,
            a: str,
            b: str,
    ) -> str:
        """Generate computation code for tt gemm."""
        mustach_hash = {}

        mustach_hash["name"] = self.name
        mustach_hash["idx"] = f"{self.idx:02d}"
        mustach_hash["m"] = m
        mustach_hash["n"] = n
        mustach_hash["k"] = k
        mustach_hash["A"] = a
        mustach_hash["B"] = b
        mustach_hash["activation_function"] = self.activation_function.write_activation_str("sum")
        mustach_hash["alpha"] = self.alpha
        mustach_hash["beta"] = self.beta
        if self.fused_layer:
            mustach_hash["fused_layer"] = self.fused_layer.write_activation_str(
                f"output_{self.path}",
                self.idx,
                f"i*{self.ldC} + j")

            if self.activation_function.name == "linear":
                mustach_hash["linear"] = True

        with open(self.template_path / "layers" / "Gemm" / "template_gemm_tt.c.tpl") as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)

    def generate_inference_code_layer(self: Self) -> str:
        """Generate computation code for layer."""
        mustach_hash = {}

        mustach_hash["name"] = self.name
        mustach_hash["idx"] = f"{self.idx:02d}"
        mustach_hash["comment"] = self.activation_function.comment
        mustach_hash["size"] = self.size
        mustach_hash["road"] = self.path

        mustach_hash["patches_size"] = self.output_width * self.output_height
        if self.transpo[0]:
            mustach_hash["gemm_code"] = self.algo_gemm_mapping[self.transpo](
                self.output_height,
                self.output_width,
                self.input_height,
                f"weights_{self.name}_{self.idx:02d}",
                self.previous_layer[0].output_str)
        else:
            mustach_hash["gemm_code"] = self.algo_gemm_mapping[self.transpo](
                self.output_height,
                self.output_width,
                self.input_width,
                f"weights_{self.name}_{self.idx:02d}",
                self.previous_layer[0].output_str)

        with open(self.template_path / "layers" / "Gemm" / "template_Gemm.c.tpl") as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)


def gemm_default_implementation(
        old_layer: Gemm,
        version: str,
) -> GemmDefault:
    """Create a Gemm_Default layer using the attributes of old_layer."""
    return GemmDefault(
        version=version,
        idx=old_layer.idx,
        size=old_layer.size,
        alpha=1 if not old_layer.alpha else old_layer.alpha[0],
        beta=1 if not old_layer.beta else old_layer.beta[0],
        transA=old_layer.transpo[0],
        transB=old_layer.transpo[1],
        weights=old_layer.weights,
        bias=old_layer.biases,
        input_shape=[1, 1, old_layer.input_height, old_layer.input_width],
        output_shape=[1, 1, old_layer.output_height, old_layer.output_width],
        activation_function=old_layer.activation_function,
    )


gemm_factory.register_implementation(
    None,
    gemm_default_implementation,
)
gemm_factory.register_implementation(
    "default",
    gemm_default_implementation,
)
