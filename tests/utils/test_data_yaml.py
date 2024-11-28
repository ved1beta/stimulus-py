import pytest
import yaml

from stimulus.utils import yaml_experiment_utils

@pytest.fixture
def load_yaml_from_file():
    """Fixture that loads a test YAML configuration file.

    This fixture loads a YAML configuration file containing DNA experiment parameters,
    including column definitions and transformation specifications. The file is used
    to test YAML parsing and manipulation utilities.

    Returns:
        dict: The parsed YAML configuration as a dictionary.
    """
    with open("tests/test_data/dna_experiment/dna_experiment_config_template.yaml") as f:
        return yaml.safe_load(f)

def test_get_length_of_params_dict(load_yaml_from_file):
    """Tests the get_length_of_params_dict function.

    This test verifies that get_length_of_params_dict correctly determines the length
    of parameter lists in transform dictionaries. It checks three cases:
    1. A transform with no parameters (should return 1)
    2. A transform with a parameter list of length 3
    3. A transform with multiple parameter lists where the longest is length 4

    Args:
        load_yaml_from_file: Pytest fixture that loads the test YAML config file

    Returns:
        None
    """

    first_dict = load_yaml_from_file['transforms'][0]
    assert yaml_experiment_utils.get_length_of_params_dict(first_dict) == 1

    second_dict = load_yaml_from_file['transforms'][1]
    assert yaml_experiment_utils.get_length_of_params_dict(second_dict) == 3

    third_dict = load_yaml_from_file['transforms'][2]
    assert yaml_experiment_utils.get_length_of_params_dict(third_dict) == 4

