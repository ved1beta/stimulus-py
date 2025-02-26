"""Central registry system for Stimulus components.

The registry system provides a way to register and manage different implementations
of components like encoders, decoders, processors etc. Components can be registered
either through decorators or through entry points for plugin support.

Example:
    Here's how to create and use a registry for encoders:

    ```python
    from abc import ABC, abstractmethod
    from stimulus.core.registry import BaseRegistry

    # Define your component interface
    class BaseEncoder(ABC):
        @abstractmethod
        def encode(self, data: str) -> bytes:
            pass

        @abstractmethod
        def decode(self, data: bytes) -> str:
            pass

    # Create a registry
    encoder_registry = BaseRegistry("stimulus.encoders", BaseEncoder)

    # Register a component
    @encoder_registry.register("base64")
    class Base64Encoder(BaseEncoder):
        def encode(self, data: str) -> bytes:
            import base64
            return base64.b64encode(data.encode())
            
        def decode(self, data: bytes) -> str:
            import base64
            return base64.b64decode(data).decode()

    # Use the registered component
    encoder = encoder_registry.get("base64")
    encoded = encoder.encode("Hello World")
    decoded = encoder.decode(encoded)
    ```

For plugin support, external packages can register components via entry points
in their setup.py/pyproject.toml:

```toml
[project.entry-points."stimulus.encoders"]
my_encoder = "mypackage.encoders:CustomEncoder"
```
"""

from importlib.metadata import entry_points
from typing import Dict, Generic, Type, TypeVar

T = TypeVar("T")


class BaseRegistry(Generic[T]):
    """Base registry class for component registration.

    This class provides a foundation for registering and managing components in Stimulus.
    Components must inherit from a specified base class and can be registered either
    through decorators or entry points.

    Args:
        entry_point_group (str): The entry point group name for plugin discovery
        base_class (Type[T]): The base class that all components must inherit from

    Example:
        ```python
        class MyComponent(ABC):
            pass


        registry = BaseRegistry[MyComponent]("stimulus.components", MyComponent)


        @registry.register("my_component")
        class CustomComponent(MyComponent):
            pass
        ```
    """

    def __init__(self, entry_point_group: str, base_class: Type[T]):
        self._components: Dict[str, Type[T]] = {}
        self.entry_point_group = entry_point_group
        self.base_class = base_class
        self.load_builtins()
        self.load_plugins()

    def register(self, name: str) -> callable:
        """Decorator factory for component registration.

        Args:
            name (str): The name to register the component under

        Returns:
            callable: A decorator that registers the component

        Raises:
            TypeError: If the decorated class doesn't inherit from the base class
        """

        def decorator(cls: Type[T]):
            if not issubclass(cls, self.base_class):
                raise TypeError(f"{cls.__name__} must subclass {self.base_class.__name__}")
            self._components[name] = cls
            return cls

        return decorator

    def load_builtins(self):
        """Override in child classes to register built-in components."""

    def load_plugins(self):
        """Load external components from entry points."""
        try:
            eps = entry_points()
            if hasattr(eps, "select"):  # Python 3.10+ API
                plugins = eps.select(group=self.entry_point_group)
            else:  # Legacy API
                plugins = eps.get(self.entry_point_group, [])

            for ep in plugins:
                self._components[ep.name] = ep.load()
        except Exception as e:
            # Log warning but don't fail if plugin loading fails
            import warnings

            warnings.warn(f"Failed to load plugins: {e!s}")

    def get(self, name: str, **params) -> T:
        """Instantiate a component with parameters.

        Args:
            name (str): The registered name of the component
            **params: Parameters to pass to the component constructor

        Returns:
            T: An instance of the requested component

        Raises:
            ValueError: If the component name is not registered
        """
        cls = self._components.get(name)
        if not cls:
            available = ", ".join(sorted(self._components.keys()))
            raise ValueError(
                f"Unknown {self.base_class.__name__} '{name}'. Available: {available}",
            )
        return cls(**params)

    @property
    def component_names(self) -> list[str]:
        """Get a list of all registered component names.

        Returns:
            list[str]: Sorted list of component names
        """
        return sorted(self._components.keys())
