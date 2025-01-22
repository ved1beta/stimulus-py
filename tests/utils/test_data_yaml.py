import pytest
import yaml

from src.stimulus.utils import yaml_data
from src.stimulus.utils.yaml_data import (
    YamlConfigDict,
    YamlSubConfigDict,
    generate_data_configs,
)


@pytest.fixture
def titanic_csv_path():
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


def test_sub_config_validation(load_titanic_yaml_from_file):
    sub_config = generate_data_configs(load_titanic_yaml_from_file)[0]
    YamlSubConfigDict.model_validate(sub_config)


def test_extract_transform_parameters_at_index(load_yaml_from_file):
    """Tests extracting parameters at specific indices from transforms."""
    # Test transform with parameter lists
    transform = load_yaml_from_file.transforms[1]  # Transform 'B' with probability list

    # Extract first parameter set
    result = yaml_data.extract_transform_parameters_at_index(transform, 0)
    assert result.columns[0].transformations[0].params["probability"] == 0.1

    # Extract second parameter set
    result = yaml_data.extract_transform_parameters_at_index(transform, 1)
    assert result.columns[0].transformations[0].params["probability"] == 0.2


def test_expand_transform_parameter_combinations(load_yaml_from_file):
    """Tests expanding transforms with parameter lists into individual transforms."""
    # Test transform with multiple parameter lists
    transform = load_yaml_from_file.transforms[2]  # Transform 'C' with multiple lists

    results = yaml_data.expand_transform_parameter_combinations(transform)
    assert len(results) == 4  # Should create 4 transforms (longest parameter list length)

    # Check first and last transforms
    assert results[0].columns[0].transformations[1].params["probability"] == 0.1
    assert results[3].columns[0].transformations[1].params["probability"] == 0.4


def test_expand_transform_list_combinations(load_yaml_from_file):
    """Tests expanding a list of transforms into all parameter combinations."""
    results = yaml_data.expand_transform_list_combinations(load_yaml_from_file.transforms)

    # Count expected transforms:
    # Transform A: 1 (no parameters)
    # Transform B: 3 (probability list length 3)
    # Transform C: 4 (probability and std lists length 4)
    assert len(results) == 8


def test_generate_data_configs(load_yaml_from_file):
    """Tests generating all possible data configurations."""
    configs = yaml_data.generate_data_configs(load_yaml_from_file)

    # Expected configs = (transforms combinations) × (number of splits)
    # 8 transform combinations × 2 splits = 16 configs
    assert len(configs) == 16

    # Check that each config is a valid YamlSubConfigDict
    for config in configs:
        assert isinstance(config, YamlSubConfigDict)
        assert config.global_params == load_yaml_from_file.global_params
        assert config.columns == load_yaml_from_file.columns


@pytest.mark.parametrize("test_input", [("load_yaml_from_file", False), ("load_wrong_type_yaml", True)])
def test_check_yaml_schema(request, test_input):
    """Tests the Pydantic schema validation."""
    data = request.getfixturevalue(test_input[0])
    expect_value_error = test_input[1]

    if not expect_value_error:
        yaml_data.check_yaml_schema(data)
        assert True
    else:
        with pytest.raises(ValueError, match="Wrong type on a field, see the pydantic report above"):
            yaml_data.check_yaml_schema(data)
