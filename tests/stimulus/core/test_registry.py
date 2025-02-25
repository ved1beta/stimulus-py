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
