from functools import wraps
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

    def infer_shape(self, *args: P.args, **kwargs: P.kwargs) -> TensorSpec:
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

    def infer_shape(
        self,
        *args: TensorSpec,
        **kwargs: TensorSpec,
    ) -> TensorSpec:
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
        if name == "self":
            continue
        if parameter.kind in [
            Parameter.POSITIONAL_ONLY,
            Parameter.POSITIONAL_OR_KEYWORD,
            Parameter.KEYWORD_ONLY,
        ]:
            bound_parameters.append(name)

    # Create a new class that inherits from both Operation and the decorated class
    @wraps(cls, updated=())
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
            # TODO Check inputs have the correct shape, if specified
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

        def infer_shape(
            self,
        ) -> TensorSpec:
            """Infer operation output shape from input tensor shapes."""
            # TODO Tag unbound inputs
            # Collect input shapes from tagged traits for inputs
            args: list[TensorSpec] = []
            kwargs: dict[str, TensorSpec] = {}
            for n, v in self.trait_get(is_input=True).items():
                t = self.trait(n)
                if getattr(t, "is_args", False):
                    args.extend(v)
                elif getattr(t, "is_kwargs", False):
                    kwargs.update(v)
                else:
                    kwargs[n] = v
            # Defer computation to wrapped class
            return super().infer_shape(*args, **kwargs)

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
                List(Supports(TensorSpec), is_input=True, is_args=True),
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
            Trait(Dict(Str(), Supports(TensorSpec), is_input=True, is_kwargs=True)),
        )

    return OperationLayer


if __name__ == "__main__":
    from traits.api import Instance

    class _N:
        def generate_inference_code_layer(self) -> str:
            raise NotImplementedError

    @layer
    class Add(_N):
        def _extend_rank(self, *inputs: TensorSpec):
            return max(i.rank for i in inputs)

        def _extend_inputs_ranks(self, *inputs: TensorSpec) -> list[TensorSpec]:
            shapes = []
            extended_rank = self._extend_rank(*inputs)
            for i in inputs:
                shapes.append(
                    TensorSpec(
                        shape=tuple(1 for _ in range(extended_rank - i.rank)) + i.shape,
                        dtype=str(i.dtype),
                    ),
                )
            return shapes

        def __call__(self, *inputs: Tensor) -> Tensor:
            # Extend rank for all inputs
            input_shapes = self._extend_inputs_ranks(*inputs)
            extended_inputs = []
            for i, s in zip(inputs, input_shapes, strict=True):
                extended_inputs.append(i.data.reshape(s))
            # Add all inputs together
            output = np.zeros(extended_inputs[0].shape, dtype=extended_inputs[0].dtype)
            for t in extended_inputs:
                output = output + t
            return Tensor(output)

        def infer_shape(self, *inputs: TensorSpec) -> TensorSpec:
            input_shapes = self._extend_inputs_ranks(*inputs)
            output_shape = tuple(
                max(input_shapes[i].shape[d] for i in range(len(input_shapes)))
                for d in range(self._extend_rank(*inputs))
            )
            return TensorSpec(shape=output_shape, dtype=str(inputs[0].dtype))

    @layer
    class Matmul(_N):
        def __call__(self, a: Tensor, b: Tensor) -> Tensor:
            return Tensor(np.dot(a, b))

        def infer_shape(self, a: TensorSpec, b: TensorSpec) -> TensorSpec:
            return TensorSpec(shape=(a.shape[-2], b.shape[-1]), dtype=str(a.dtype))

    @layer
    class Input(_N):
        tensor = Instance(Tensor)

        def __call__(self) -> Tensor:
            return self.tensor

        def infer_shape(self) -> TensorSpec:
            return self.tensor

    from traits.adaptation.api import register_factory

    register_factory(lambda d: d.infer_shape(), Operation, TensorSpec)

    x = np.array([1])
    y = np.array([2, 3, 4])
    z = np.array([[10, 20, 30], [40, 50, 60]])

    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = np.array([[10], [20], [30]])

    ia = Input(tensor=Tensor(a))
    ib = Input(tensor=Tensor(b))

    m = Matmul(a=ia, b=ib)
    print(m.infer_shape().shape)
    print(m.forward_path_layer([a, b]))

    e = Add()
    e.inputs.append(x)
    e.inputs.append(y)
    e.inputs.append(z)
    print(e.infer_shape().shape)
