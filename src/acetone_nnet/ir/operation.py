from inspect import Parameter, signature
from typing import Generic, ParamSpec, Self, TypeVar

import numpy as np
from traits.api import Dict, Interface, List, Str, Supports, Trait, provides
from traits.has_traits import ABCHasTraits

from acetone_nnet.ir.layer import Layer
from acetone_nnet.ir.tensor import Tensor, TensorSpec

P = ParamSpec("P", bound=Tensor)  # type: ignore[misc] # bound undefined for ParamSpec

R = TypeVar("R", Tensor, dict[str, Tensor], tuple[Tensor, ...])


class _OperationBase(Interface, Generic[P, R]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        raise NotImplementedError


class Operation(Interface):
    """Basic tensor operation."""

    #: Names of input tensors
    input_names = List(Str)

    #: Names of output tensors
    output_names = List(Str)

    def validate_inputs(
        self,
        *args: TensorSpec,
        **kwargs: TensorSpec,
    ) -> bool:
        """Validate the provided inputs against operation requirements."""

    def __call__(
        self,
        *args: Tensor,
        **kwargs: Tensor,
    ) -> dict[str, Tensor]:
        """Apply the operation on the provided inputs."""

    def infer_output_shapes(
        self,
        *args: TensorSpec,
        **kwargs: TensorSpec,
    ) -> dict[str, TensorSpec]:
        """Infer operation output shape from input tensor shapes."""


class _OperationLayer(Operation, Layer, _OperationBase, Interface):
    pass


def layer(cls) -> type[_OperationLayer]:
    """Transform callable class into an ACETONE layer."""
    # TODO Support additional parameters, T.B.D.
    # TODO Support no input/output names (just operation(MyClass))
    # TODO Support decorating functions (operation()(my_func)
    # TODO Check for default parameters *args and **kwargs
    # TODO Assert all operations are of the right type
    # TODO Check typing of all parameters as being `Tensor`
    # TODO Check return type to select appropriate action
    # TODO Validate the args and kwargs contain the required inputs

    bound_parameters = []
    operation_signature = signature(cls.__call__)
    for name, parameter in operation_signature.parameters.items():
        if parameter.kind in [
            Parameter.POSITIONAL_ONLY,
            Parameter.POSITIONAL_OR_KEYWORD,
            Parameter.KEYWORD_ONLY,
        ]:
            bound_parameters.append(name)

    # Create a new class that inherits from both Operation and the decorated class
    @provides(Layer)
    @provides(Operation)
    class OperationLayer(cls, Layer, ABCHasTraits):

        input_names: list[str] = List(Str(), value=bound_parameters)

        output_names: list[str] = List(Str(), value=["output"])

        def validate_inputs(
            self: Self,
            *args: TensorSpec,
            **kwargs: TensorSpec,
        ) -> bool:
            """Validate the provided inputs against operation requirements."""
            # TODO Validate using the registered Spec where appropriate
            is_valid = True
            if callable(getattr(cls, "validate_inputs", None)):
                is_valid = is_valid and self.validate_inputs(*args, **kwargs)
            for i in self.input_names:
                if callable(v := getattr(self, f"_validate_{i}", None)):
                    is_valid = is_valid and v(**{i: kwargs[i]})
            return is_valid

        def __call__(
            self: Self,
            *args: Tensor,
            **kwargs: Tensor,
        ) -> dict[str, Tensor]:
            """Apply the operation on the provided inputs."""
            # TODO Match inputs with the operation expected ones
            if not self.validate_inputs(*args, **kwargs):
                raise Exception("Invalid inputs")
            outputs = super().__call__(*args, **kwargs)
            match outputs:
                case tuple():
                    return dict(zip(self.output_names, outputs, strict=True))
                case dict():
                    return outputs
                case Tensor():
                    return {self.output_names[0]: outputs}
                case _:
                    msg = f"Unsupported operation output type: {type(outputs)}"
                    raise ValueError(msg)

        def forward_path_layer(
            self: Self,
            inputs: np.ndarray | list[np.ndarray],
        ) -> np.ndarray:
            """Call the underlying operation.

            Defaults to picking the first output.
            Defaults to mapping provided inputs to the underlying operation ones in-order.
            """
            # Convert input arrays into tensors
            input_tensors: list[Tensor] = []
            if isinstance(inputs, list):
                input_tensors.extend(map(Tensor, inputs))
            else:
                input_tensors.append(Tensor(inputs))
            # Call operation and collect output
            return next(iter(self.__call__(*input_tensors).values())).data

        def infer_output_shapes(
            self,
            *args: TensorSpec,
            **kwargs: TensorSpec,
        ) -> dict[str, TensorSpec]:
            """Infer operation output shape from input tensor shapes."""
            raise NotImplementedError

    # Add traits to capture tensor information for each input
    for i in bound_parameters:
        OperationLayer.add_class_trait(
            i,
            Trait(Supports(TensorSpec, is_input=True)),
        )
    # Add trait to capture tensor information for args-based inputs
    supports_args = any(
        p.kind == Parameter.VAR_POSITIONAL
        for p in signature(cls.__call__).parameters.values()
    )
    if supports_args:
        a = next(
            p
            for p in signature(cls.__call__).parameters.values()
            if p.kind == Parameter.VAR_POSITIONAL
        )
        OperationLayer.add_class_trait(
            a.name,
            Trait(
                List(Supports(TensorSpec, is_input=True)),
            ),
        )
    # Add trait to capture tensor information for kwargs-based inputs
    supports_kwargs = any(
        p.kind == Parameter.VAR_KEYWORD
        for p in signature(cls.__call__).parameters.values()
    )
    if supports_kwargs:
        a = next(
            p
            for p in signature(cls.__call__).parameters.values()
            if p.kind == Parameter.VAR_KEYWORD
        )
        OperationLayer.add_class_trait(
            a.name,
            Trait(Dict(Str(), Supports(TensorSpec), is_input=True)),
        )

    return OperationLayer


if __name__ == "__main__":

    @layer
    class Add:
        def __call__(self, *inputs: Tensor) -> Tensor:
            # Extend rank for all inputs
            extended_inputs = []
            extended_rank = max(i.rank for i in inputs)
            for i in inputs:
                extended_inputs.append(
                    i.data.reshape(
                        tuple(1 for _ in range(extended_rank - i.rank)) + i.shape,
                    ),
                )
            # Add all inputs together
            output = np.zeros(extended_inputs[0].shape, dtype=extended_inputs[0].dtype)
            for t in extended_inputs:
                output = output + t
            return Tensor(output)

    class AddDefault(Add):
        def generate_inference_code_layer(self):
            pass

    a = AddDefault()
    a.name = "add"
    a.idx = 42

    x = np.array([1])
    y = np.array([2, 3, 4])
    z = np.array([[10, 20, 30], [40, 50, 60]])

    print(a.forward_path_layer([x, y, z]))
    print(a(Tensor(x), Tensor(y), Tensor(z))["output"].data)
