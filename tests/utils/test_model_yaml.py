"""Tests for YAML model handling functionalities."""

import pytest
import yaml

from src.stimulus.utils import yaml_model_schema


@pytest.fixture
def titanic_model_yaml_path() -> str:
    """Get path to Titanic model YAML file."""
    return "tests/test_model/titanic_model_cpu.yaml"


def test_model_yaml(titanic_model_yaml_path: str) -> None:
    """Test model YAML file."""
    with open(titanic_model_yaml_path) as file:
        model_config = yaml.safe_load(file)
    model = yaml_model_schema.Model.model_validate(model_config)
    assert model is not None
