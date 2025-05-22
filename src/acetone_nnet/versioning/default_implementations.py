from typing_extensions import Self

from .layer_factories import implemented


class DefaultImplementationManager:
    """Class managing the default layers implementation."""

    def __init__(self: Self) -> None:
        """Build the manager."""
        self.default_implementations: dict[str, str] = {}

    def set_as_default(self: Self, layer_name: str, version: str | None) -> None:
        """Set version as the default implementation of layer_name."""
        factory = implemented.get(layer_name, None)
        if factory is None:
            self.default_implementations[layer_name] = version
        elif factory.implementations.get(version, None) is None:
                msg = f"{version} not implemented."
                raise KeyError(msg)
        else:
            self.default_implementations[layer_name] = version
        self.default_implementations = dict(sorted(self.default_implementations.items()))

    def list_default_implementations(self) -> list[tuple[str, str]]:
        """Return known Layer implementations."""
        return list(self.default_implementations.items())

default_implementations_manager: DefaultImplementationManager = DefaultImplementationManager()