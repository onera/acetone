"""Convolution Gemm layer base type definiton.

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
from typing_extensions import Self

from .Conv2D import Conv2D


class Conv2DGemm(Conv2D):
    """Convolution base class for gem implementation."""

    def __init__(self: Self, **kwargs: int) -> None:
        """Instantiate a Conv2D with a gemm implementation."""
        super().__init__(**kwargs)
        self.patches_height = self.input_channels * self.kernel_h * self.kernel_w
        self.patches_width = self.output_height * self.output_width
        self.patches_size = self.patches_height * self.patches_width

        self.conv_algorithm = self.conv_algorithm[-7:]
        self.algo_gemm_mapping = {"gemm_nn": self.write_gemm_nn,
                                  "gemm_nt": self.write_gemm_nt,
                                  "gemm_tn": self.write_gemm_tn,
                                  "gemm_tt": self.write_gemm_tt}

    def write_gemm_nn(
            self: Self,
            m: int,
            n: int,
            k: int,
            a: str,
            b: str,
            c: str,
            direct: bool,
    ) -> None:
        """Generate computation code for nn gemm algorithm."""
        mustach_hash = {}

        mustach_hash["direct"] = direct
        mustach_hash["strides"] = self.strides
        mustach_hash["name"] = self.name
        mustach_hash["idx"] = f"{self.idx:02d}"
        mustach_hash["m"] = m
        mustach_hash["n"] = n
        mustach_hash["k"] = k
        mustach_hash["A"] = a
        mustach_hash["ldA"] = k
        mustach_hash["B"] = b
        mustach_hash["ldB"] = n
        mustach_hash["C"] = c
        mustach_hash["ldC"] = n
        mustach_hash["activation_function"] = self.activation_function.write_activation_str("output")
        if self.fused_layer:
            mustach_hash["fused_layer"] = self.fused_layer.write_activation_str(
                "output",
                self.idx,
                f"i*{self.ldC} + j")

            if self.activation_function.name == "linear":
                mustach_hash["linear"] = True

        with open(self.template_path / "layers" / "Conv" / "template_Conv_gemm_nn.c.tpl") as template_file:
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
            c: str,
            direct: bool,
    ) -> None:
        """Generate computation code for nt gemm algorithm."""
        mustach_hash = {}

        mustach_hash["direct"] = direct
        mustach_hash["name"] = self.name
        mustach_hash["idx"] = f"{self.idx:02d}"
        mustach_hash["m"] = m
        mustach_hash["n"] = n
        mustach_hash["k"] = k
        mustach_hash["A"] = a
        mustach_hash["ldA"] = k
        mustach_hash["B"] = b
        mustach_hash["ldB"] = n
        mustach_hash["C"] = c
        mustach_hash["ldC"] = n
        mustach_hash["activation_function"] = self.activation_function.write_activation_str("output")
        if self.fused_layer:
            mustach_hash["fused_layer"] = self.fused_layer.write_activation_str(
                "output",
                self.idx,
                f"i*{self.ldC} + j")

            if self.activation_function.name == "linear":
                mustach_hash["linear"] = True

        with open(self.template_path / "layers" / "Conv" / "template_Conv_gemm_nt.c.tpl") as template_file:
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
            c: str,
            direct: bool,
    ) -> None:
        """Generate computation code for tn gemm algorithm."""
        mustach_hash = {}

        mustach_hash["direct"] = direct
        mustach_hash["name"] = self.name
        mustach_hash["idx"] = f"{self.idx:02d}"
        mustach_hash["m"] = m
        mustach_hash["n"] = n
        mustach_hash["k"] = k
        mustach_hash["A"] = a
        mustach_hash["ldA"] = k
        mustach_hash["B"] = b
        mustach_hash["ldB"] = n
        mustach_hash["C"] = c
        mustach_hash["ldC"] = n
        mustach_hash["activation_function"] = self.activation_function.write_activation_str("output")
        if self.fused_layer:
            mustach_hash["fused_layer"] = self.fused_layer.write_activation_str(
                "output",
                self.idx,
                f"i*{self.ldC} + j")

            if self.activation_function.name == "linear":
                mustach_hash["linear"] = True

        with open(self.template_path / "layers" / "Conv" / "template_Conv_gemm_tn.c.tpl") as template_file:
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
            c: str,
            direct: bool,
    ) -> None:
        """Generate computation code for t gemm algorithm."""
        mustach_hash = {}

        mustach_hash["direct"] = direct
        mustach_hash["name"] = self.name
        mustach_hash["idx"] = f"{self.idx:02d}"
        mustach_hash["m"] = m
        mustach_hash["n"] = n
        mustach_hash["k"] = k
        mustach_hash["A"] = a
        mustach_hash["ldA"] = k
        mustach_hash["B"] = b
        mustach_hash["ldB"] = n
        mustach_hash["C"] = c
        mustach_hash["ldC"] = n
        mustach_hash["activation_function"] = self.activation_function.write_activation_str("sum")
        if self.fused_layer:
            mustach_hash["fused_layer"] = self.fused_layer.write_activation_str(
                "output",
                self.idx,
                f"i*{self.ldC} + j")

            if self.activation_function.name == "linear":
                mustach_hash["linear"] = True

        with open(self.template_path / "layers" / "Conv" / "template_Conv_gemm_tt.c.tpl") as template_file:
            template = template_file.read()
        template_file.close()

        return pystache.render(template, mustach_hash)