"""Tests for the BaseRegistry class."""

from abc import ABC, abstractmethod

import pytest

from stimulus.core.registry import BaseRegistry


# Test fixtures
class MockComponent(ABC):
    """Mock abstract base class for testing."""

    @abstractmethod
    def do_something(self):
        pass


class ValidComponent(MockComponent):
    """Valid component implementation."""

    def __init__(self, param=None):
        self.param = param

    def do_something(self):
        return "done"


class InvalidComponent:
    """Component that doesn't inherit from base class."""


def test_registry_initialization():
    """Test basic registry initialization."""
    registry = BaseRegistry("test.group", MockComponent)
    assert registry.entry_point_group == "test.group"
    assert registry.base_class == MockComponent
    assert len(registry.component_names) == 0


def test_component_registration():
    """Test component registration via decorator."""
    registry = BaseRegistry("test.group", MockComponent)

    @registry.register("valid")
    class TestComponent(MockComponent):
        def do_something(self):
            return "test"

    assert "valid" in registry.component_names
    instance = registry.get("valid")
    assert isinstance(instance, MockComponent)
    assert instance.do_something() == "test"


def test_invalid_component_registration():
    """Test registration of invalid component."""
    registry = BaseRegistry("test.group", MockComponent)

    with pytest.raises(TypeError):

        @registry.register("invalid")
        class TestInvalid(InvalidComponent):
            pass


def test_component_instantiation_with_params():
    """Test component instantiation with parameters."""
    registry = BaseRegistry("test.group", MockComponent)

    @registry.register("parameterized")
    class TestComponent(ValidComponent):
        pass

    instance = registry.get("parameterized", param="test_value")
    assert instance.param == "test_value"


def test_unknown_component():
    """Test error handling for unknown components."""
    registry = BaseRegistry("test.group", MockComponent)

    with pytest.raises(ValueError) as exc:
        registry.get("nonexistent")

    assert "Unknown MockComponent 'nonexistent'" in str(exc.value)
    assert "Available:" in str(exc.value)


def test_component_names_sorted():
    """Test that component_names returns sorted list."""
    registry = BaseRegistry("test.group", MockComponent)

    @registry.register("zebra")
    class ZebraComponent(ValidComponent):
        pass

    @registry.register("alpha")
    class AlphaComponent(ValidComponent):
        pass

    assert registry.component_names == ["alpha", "zebra"]


def test_practical_encoder_registry_example():
    """
    Demonstrates practical usage of the registry system for encoders.
    
    This test shows how to:
    1. Create a base encoder class
    2. Create a registry for encoders
    3. Register custom encoders
    4. Use the registered encoders
    """
    # 1. Define a base encoder interface
    class BaseEncoder(ABC):
        @abstractmethod
        def encode(self, data: str) -> bytes:
            """Encode string data into bytes."""
            pass
        
        @abstractmethod
        def decode(self, data: bytes) -> str:
            """Decode bytes back into string."""
            pass

    # 2. Create an encoder registry
    encoder_registry = BaseRegistry("stimulus.encoders", BaseEncoder)

    # 3. Register a custom encoder
    @encoder_registry.register("base64")
    class Base64Encoder(BaseEncoder):
        def encode(self, data: str) -> bytes:
            import base64
            return base64.b64encode(data.encode())
        
        def decode(self, data: bytes) -> str:
            import base64
            return base64.b64decode(data).decode()

    # 4. Register another encoder
    @encoder_registry.register("rot13")
    class Rot13Encoder(BaseEncoder):
        def encode(self, data: str) -> bytes:
            return data.encode('rot13').encode()
        
        def decode(self, data: bytes) -> str:
            return data.decode().encode('rot13').decode()

    # Test that encoders are registered
    assert set(encoder_registry.component_names) == {"base64", "rot13"}

    # Test using a registered encoder
    base64_encoder = encoder_registry.get("base64")
    test_data = "Hello, World!"
    encoded = base64_encoder.encode(test_data)
    decoded = base64_encoder.decode(encoded)
    assert decoded == test_data

    # Test getting an encoder with parameters
    @encoder_registry.register("configurable")
    class ConfigurableEncoder(BaseEncoder):
        def __init__(self, encoding: str = "utf-8"):
            self.encoding = encoding
            
        def encode(self, data: str) -> bytes:
            return data.encode(self.encoding)
            
        def decode(self, data: bytes) -> str:
            return data.decode(self.encoding)

    # Get encoder with custom parameter
    utf16_encoder = encoder_registry.get("configurable", encoding="utf-16")
    assert utf16_encoder.encoding == "utf-16"
