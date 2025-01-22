"""Tests for YAML data handling functionality."""

import pytest
import yaml

from src.stimulus.utils import yaml_data
from src.stimulus.utils.yaml_data import (
    YamlConfigDict,
    YamlSubConfigDict,
    generate_data_configs,
)


@pytest.fixture
def titanic_csv_path() -> str:
    """Get path to Titanic CSV file."""
    return "tests/test_data/titanic/titanic_stimulus.csv"


@pytest.fixture
def load_titanic_yaml_from_file() -> YamlConfigDict:
    """Fixture that loads a test YAML configuration file."""
    with open("tests/test_data/titanic/titanic.yaml") as f:
        yaml_dict = yaml.safe_load(f)
        return YamlConfigDict(**yaml_dict)


@pytest.fixture
def load_yaml_from_file() -> YamlConfigDict:
    """Fixture that loads a test YAML configuration file."""
    with open("tests/test_data/dna_experiment/dna_experiment_config_template.yaml") as f:
        yaml_dict = yaml.safe_load(f)
        return YamlConfigDict(**yaml_dict)


@pytest.fixture
def load_wrong_type_yaml() -> dict:
    """Fixture that loads a YAML configuration file with wrong typing."""
    with open("tests/test_data/yaml_files/wrong_field_type.yaml") as f:
        return yaml.safe_load(f)


def test_sub_config_validation(
    load_titanic_yaml_from_file: YamlConfigDict,
) -> None:
    """Test sub-config validation."""
    sub_config = generate_data_configs(load_titanic_yaml_from_file)[0]
    YamlSubConfigDict.model_validate(sub_config)


def test_extract_transform_parameters_at_index(
    load_yaml_from_file: YamlConfigDict,
) -> None:
    """Tests extracting parameters at specific indices from transforms."""
    # Test transform with parameter lists
    transform = load_yaml_from_file.transforms[0]
    params = yaml_data.extract_transform_parameters_at_index(transform, 0)
    assert params == {"param1": 1, "param2": "a"}


def test_expand_transform_parameter_combinations(
    load_yaml_from_file: YamlConfigDict,
) -> None:
    """Tests expanding transforms with parameter lists into individual transforms."""
    # Test transform with multiple parameter lists
    transform = load_yaml_from_file.transforms[0]
    results = yaml_data.expand_transform_parameter_combinations(transform)
    assert len(results) == 4  # 2x2 combinations
    assert all(isinstance(r, dict) for r in results)


def test_expand_transform_list_combinations(
    load_yaml_from_file: YamlConfigDict,
) -> None:
    """Tests expanding a list of transforms into all parameter combinations."""
    results = yaml_data.expand_transform_list_combinations(load_yaml_from_file.transforms)
    assert len(results) == 8  # 4 combinations from first transform x 2 from second
    assert all(isinstance(r, list) for r in results)


def test_generate_data_configs(
    load_yaml_from_file: YamlConfigDict,
) -> None:
    """Tests generating all possible data configurations."""
    configs = yaml_data.generate_data_configs(load_yaml_from_file)
    assert len(configs) == 16  # 8 transform combinations x 2 splits
    assert all(isinstance(c, YamlConfigDict) for c in configs)


@pytest.mark.parametrize(
    "test_input",
    [("load_yaml_from_file", False), ("load_wrong_type_yaml", True)],
)
def test_check_yaml_schema(
    request: pytest.FixtureRequest,
    test_input: tuple[str, bool],
) -> None:
    """Tests the Pydantic schema validation."""
    data = request.getfixturevalue(test_input[0])
    if test_input[1]:
        with pytest.raises(ValueError, match="Invalid YAML schema"):
            yaml_data.check_yaml_schema(data)
    else:
        yaml_data.check_yaml_schema(data)
