"""Tests for YAML model handling functionalities."""

import pytest
import yaml
from ray import tune

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


def test_raytune_space_selector(titanic_model_yaml_path: str) -> None:
    """Test raytune_space_selector method."""
    with open(titanic_model_yaml_path) as file:
        model_config = yaml.safe_load(file)
    model = yaml_model_schema.Model.model_validate(model_config)
    loader = yaml_model_schema.YamlRayConfigLoader(model)

    result = loader.raytune_space_selector(tune.choice, [1, 2, 3])
    assert str(type(result).__name__) == "Categorical"

    result = loader.raytune_space_selector(tune.randint, [1, 10])
    assert str(type(result).__name__) == "Integer"


def test_convert_raytune(titanic_model_yaml_path: str) -> None:
    """Test convert_raytune method."""
    with open(titanic_model_yaml_path) as file:
        model_config = yaml.safe_load(file)
    model = yaml_model_schema.Model.model_validate(model_config)
    loader = yaml_model_schema.YamlRayConfigLoader(model)

    param = yaml_model_schema.TunableParameter(space=[16, 32, 64], mode="choice")
    result = loader.convert_raytune(param)
    assert str(type(result).__name__) == "Categorical"

    param = yaml_model_schema.TunableParameter(space=[1, 5], mode="randint")
    result = loader.convert_raytune(param)
    assert str(type(result).__name__) == "Integer"


def test_convert_config_to_ray(titanic_model_yaml_path: str) -> None:
    """Test convert_config_to_ray method."""
    with open(titanic_model_yaml_path) as file:
        model_config = yaml.safe_load(file)
    model = yaml_model_schema.Model.model_validate(model_config)
    loader = yaml_model_schema.YamlRayConfigLoader(model)

    ray_model = loader.convert_config_to_ray(model)
    assert isinstance(ray_model, yaml_model_schema.RayTuneModel)
    assert "network_params" in ray_model.model_dump()
    assert "optimizer_params" in ray_model.model_dump()
    assert "loss_params" in ray_model.model_dump()
    assert "data_params" in ray_model.model_dump()


def test_get_config(titanic_model_yaml_path: str) -> None:
    """Test get_config method."""
    with open(titanic_model_yaml_path) as file:
        model_config = yaml.safe_load(file)
    model = yaml_model_schema.Model.model_validate(model_config)
    loader = yaml_model_schema.YamlRayConfigLoader(model)

    config = loader.get_config()
    assert isinstance(config, yaml_model_schema.RayTuneModel)
    assert hasattr(config, "network_params")
    assert hasattr(config, "optimizer_params")
    assert hasattr(config, "loss_params")
    assert hasattr(config, "data_params")


def test_sampint() -> None:
    """Test sampint static method."""
    sample_space = [1, 5]
    n_space = [2, 4]

    result = yaml_model_schema.YamlRayConfigLoader.sampint(sample_space, n_space)

    assert isinstance(result, list)
    assert all(isinstance(x, int) for x in result)

    assert len(result) >= n_space[0]
    assert len(result) <= n_space[1]
    assert all(sample_space[0] <= x <= sample_space[1] for x in result)
