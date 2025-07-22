from functools import wraps
from inspect import Parameter, signature

import numpy as np
from traits.api import (
    ABCHasTraits,
    Dict,
    HasTraits,
    Interface,
    List,
    Property,
    Str,
    Supports,
    Trait,
    cached_property,
    provides,
)
from typing_extensions import Self

from acetone_nnet.ir.layer import Layer
from acetone_nnet.ir.tensor import Tensor, TensorSpec


class Operation(Interface):
    """Interface a tensor-based operations.

    The interface defines the expected method from an operation on tensors.
    Additional methods may be provided to perform input validation, or provide
    information on the computation itself, e.g. the shape of output tensor.

    An operation may have additional parameters, fixed for each instance of the
    operation itself. As an example, the operation may use weights and bias tensors
    or a scalar padding value.
    """

    def validate_inputs(
        self,
        *args: TensorSpec,
        **kwargs: TensorSpec,
    ) -> bool:
        """Validate the applicability of the operation with the provided inputs."""

    def __call__(
        self,
        *args: Tensor,
        **kwargs: Tensor,
    ) -> Tensor:
        """Perform the operation on the provided input tensors."""

    def infer_shape(
        self,
        *args: TensorSpec,
        **kwargs: TensorSpec,
    ) -> TensorSpec:
        """Infer operation output shape from the provided input tensors."""


class _OperationBaseLayer(Operation, Layer, Interface):

    #: Names of input tensors
    input_names = Property(List(Str))

    def _get_input_names(self) -> list[str]: ...


class ParameterError(Exception):
    """Invalid parameter kind or value."""


class UnboundInputError(Exception):
    """Raise when input value or specification not set."""

    def __init__(self, message: str, name: str):
        super().__init__(message)
        self.name = name


def _add_traits_for_parameters(
    cls: type[HasTraits],
    *parameters: Parameter,
    ptype: type,
) -> None:
    """Add traits supporting information for a set of parameter.

    Add a trait supporting the specified `ptype` for each of the specified `parameters`.
    args-like and kwargs-like parameters will respectively support a list and dict of
    `ptype`.

    Parameters
    ----------
    cls: a trait-based class to which parameters will be added
    parameters : list of parameters to consider
    ptype : type of trait instance

    """
    for p in parameters:
        # Ignore parameters with no usable name for tag
        if p.kind == Parameter.POSITIONAL_ONLY:
            continue
        # Ignore not-to-be bound self parameter on methods
        if p.name == "self":
            continue
        match p.kind:
            case Parameter.POSITIONAL_OR_KEYWORD | Parameter.KEYWORD_ONLY:
                cls.add_class_trait(
                    p.name,
                    Trait(
                        Supports(
                            ptype,
                            is_input=True,
                        ),
                    ),
                )
            case Parameter.VAR_POSITIONAL:
                cls.add_class_trait(
                    p.name,
                    Trait(
                        List(
                            Supports(ptype),
                            is_input=True,
                            is_args=True,
                        ),
                    ),
                )
            case Parameter.VAR_KEYWORD:
                cls.add_class_trait(
                    p.name,
                    Trait(
                        Dict(
                            Str(),
                            Supports(ptype),
                            is_input=True,
                            is_kwargs=True,
                        ),
                    ),
                )


def layer(cls: type[Operation]) -> type[_OperationBaseLayer]:
    """Transform a callable Operation into an ACETONE layer."""
    # TODO Support additional parameters, T.B.D.
    # TODO Support no input/output names (just operation(MyClass))
    # TODO Support decorating functions (operation()(my_func)
    # TODO Check for default parameters *args and **kwargs
    # TODO Assert all operations are of the right type
    # TODO Check typing of all parameters as being `Tensor`
    # TODO Check return type to select appropriate action
    # TODO Validate the args and kwargs contain the required inputs

    # Create a new class that inherits from both Operation and the decorated class
    @wraps(cls, updated=())
    @provides(Layer)
    @provides(Operation)
    @provides(_OperationBaseLayer)
    class OperationLayer(cls, Layer, ABCHasTraits):

        input_names: list[str] = Property(List(Str()))

        @cached_property
        def _get_input_names(self) -> list[str]:
            return list(self.trait_get(is_input=True).keys())

        def _collect_inputs_specifications(
            self: Self,
            *,
            strict: bool = True,
        ) -> tuple[list[TensorSpec], dict[str, TensorSpec]]:
            """Collect tensor specification for all layer inputs.

            Parameters
            ----------
            strict: bool (default: True) Ensure a specification is available for all
                    known inputs.

            Returns
            -------
            [list[TensorSpec], dict[str, TensorSpec]]
            A list of positional and keyword specifications for each input

            """
            args: list[TensorSpec] = []
            kwargs: dict[str, TensorSpec] = {}
            for t, v in self.trait_get(is_input=True).items():
                d = self.trait(t)
                if getattr(d, "is_args", False):
                    args.extend(v)
                elif getattr(d, "is_kwargs", False):
                    kwargs.update(v)
                else:
                    kwargs[t] = v
            # Ensure all inputs are bound in strict mode
            if strict:
                checklist = {f"_{i}": v for i, v in enumerate(args)}
                checklist.update(kwargs)
                for n, v in checklist.items():
                    if v is None:
                        raise UnboundInputError(n, f"Unbound input {n}")
            return args, kwargs

        def validate_inputs(
            self: Self,
        ) -> bool:
            """Validate the provided inputs against operation requirements."""
            args, kwargs = self._collect_inputs_specifications()
            is_valid = True
            if callable(getattr(cls, "validate_inputs", None)):
                is_valid = is_valid and cls.validate_inputs(*args, **kwargs)
            for i in self.input_names:
                if callable(v := getattr(self, f"_validate_{i}", None)):
                    is_valid = is_valid and v(**{i: kwargs[i]})
            return is_valid

        def __call__(
            self: Self,
            *args: Tensor,
            **kwargs: Tensor,
        ) -> Tensor:
            """Apply the operation on the provided inputs."""
            # TODO Check inputs have the correct shape, if specified
            # TODO Match inputs with the operation expected ones
            if not self.validate_inputs():
                raise Exception("Invalid inputs")
            return super().__call__(*args, **kwargs)

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
            return self.__call__(*input_tensors).data

        def infer_shape(
            self,
        ) -> TensorSpec:
            """Infer operation output shape from input tensor shapes."""
            # TODO Tag unbound inputs
            # Collect input shapes from tagged traits for inputs
            args, kwargs = self._collect_inputs_specifications(strict=True)
            # Defer computation to wrapped class
            return super().infer_shape(*args, **kwargs)

    _add_traits_for_parameters(
        OperationLayer,
        *signature(cls.__call__).parameters.values(),
        ptype=TensorSpec,
    )

    return OperationLayer


if __name__ == "__main__":
    from traits.api import Instance

    class _N:
        def generate_inference_code_layer(self) -> str:
            raise NotImplementedError

    @layer
    class Add(_N, Operation):
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
