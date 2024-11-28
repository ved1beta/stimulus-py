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

def test_get_transform_base_dict_structure(load_yaml_from_file):
    """Tests the get_transform_base_dict function structure.

    This test verifies that get_transform_base_dict correctly preserves the structure
    of transform dictionaries while emptying parameter values. It checks three cases:
    1. A transform with no parameters (structure should remain identical)
    2. A transform with a single parameter list (structure should remain identical)
    3. A transform with multiple parameter lists (structure should remain identical)

    Args:
        load_yaml_from_file: Pytest fixture that loads the test YAML config file

    Returns:
        None
    """
    first_dict = load_yaml_from_file['transforms'][0]
    base_dict = yaml_experiment_utils.get_transform_base_dict(first_dict)
    assert base_dict.keys() == first_dict.keys()
    assert base_dict[yaml_experiment_utils.TransformKeys.COLUMN_KEY][0].keys() == first_dict[yaml_experiment_utils.TransformKeys.COLUMN_KEY][0].keys()
    assert base_dict[yaml_experiment_utils.TransformKeys.COLUMN_KEY][0][yaml_experiment_utils.TransformKeys.TRANSFORMATIONS_KEY][0].keys() == first_dict[yaml_experiment_utils.TransformKeys.COLUMN_KEY][0][yaml_experiment_utils.TransformKeys.TRANSFORMATIONS_KEY][0].keys()
    # Check that params fields are empty
    for column in base_dict[yaml_experiment_utils.TransformKeys.COLUMN_KEY]:
        for transformation in column[yaml_experiment_utils.TransformKeys.TRANSFORMATIONS_KEY]:
            assert transformation[yaml_experiment_utils.TransformKeys.PARAMS_KEY] == {} or transformation[yaml_experiment_utils.TransformKeys.PARAMS_KEY] == None

    second_dict = load_yaml_from_file['transforms'][1]
    base_dict = yaml_experiment_utils.get_transform_base_dict(second_dict)
    assert base_dict.keys() == second_dict.keys()
    assert base_dict[yaml_experiment_utils.TransformKeys.COLUMN_KEY][0].keys() == second_dict[yaml_experiment_utils.TransformKeys.COLUMN_KEY][0].keys()
    assert base_dict[yaml_experiment_utils.TransformKeys.COLUMN_KEY][0][yaml_experiment_utils.TransformKeys.TRANSFORMATIONS_KEY][0].keys() == second_dict[yaml_experiment_utils.TransformKeys.COLUMN_KEY][0][yaml_experiment_utils.TransformKeys.TRANSFORMATIONS_KEY][0].keys()
    # Check that params fields are empty
    for column in base_dict[yaml_experiment_utils.TransformKeys.COLUMN_KEY]:
        for transformation in column[yaml_experiment_utils.TransformKeys.TRANSFORMATIONS_KEY]:
            assert transformation[yaml_experiment_utils.TransformKeys.PARAMS_KEY] == {} or transformation[yaml_experiment_utils.TransformKeys.PARAMS_KEY] == None

    third_dict = load_yaml_from_file['transforms'][2]
    base_dict = yaml_experiment_utils.get_transform_base_dict(third_dict)
    assert base_dict.keys() == third_dict.keys()
    assert base_dict[yaml_experiment_utils.TransformKeys.COLUMN_KEY][0].keys() == third_dict[yaml_experiment_utils.TransformKeys.COLUMN_KEY][0].keys()
    assert base_dict[yaml_experiment_utils.TransformKeys.COLUMN_KEY][0][yaml_experiment_utils.TransformKeys.TRANSFORMATIONS_KEY][0].keys() == third_dict[yaml_experiment_utils.TransformKeys.COLUMN_KEY][0][yaml_experiment_utils.TransformKeys.TRANSFORMATIONS_KEY][0].keys()
    # Check that params fields are empty
    for column in base_dict[yaml_experiment_utils.TransformKeys.COLUMN_KEY]:
        for transformation in column[yaml_experiment_utils.TransformKeys.TRANSFORMATIONS_KEY]:
            assert transformation[yaml_experiment_utils.TransformKeys.PARAMS_KEY] == {} or transformation[yaml_experiment_utils.TransformKeys.PARAMS_KEY] == None

def test_split_transform_dict(load_yaml_from_file):
    """Tests the split_transform_dict function.

    This test verifies that split_transform_dict correctly splits transform dictionaries
    containing parameter lists into separate dictionaries with single parameter values.
    It checks:
    1. A transform with no parameters (should remain unchanged)
    2. A transform with a single parameter list (should extract single values)
    3. A transform with multiple parameter lists (should extract corresponding values)
    4. Error handling when split index is out of bounds

    Args:
        load_yaml_from_file: Pytest fixture that loads the test YAML config file

    Returns:
        None

    Raises:
        IndexError: When attempting to split with an index larger than parameter list length
    """

    first_dict = load_yaml_from_file['transforms'][0]
    base_dict = yaml_experiment_utils.get_transform_base_dict(first_dict)
    split_dict = yaml_experiment_utils.split_transform_dict(first_dict, base_dict, 0)
    assert yaml_experiment_utils.get_length_of_params_dict(split_dict) == 1

    second_dict = load_yaml_from_file['transforms'][1]
    base_dict = yaml_experiment_utils.get_transform_base_dict(second_dict)
    split_dict = yaml_experiment_utils.split_transform_dict(second_dict, base_dict, 0)
    assert yaml_experiment_utils.get_length_of_params_dict(split_dict) == 1

    third_dict = load_yaml_from_file['transforms'][2]
    base_dict = yaml_experiment_utils.get_transform_base_dict(third_dict)
    split_dict = yaml_experiment_utils.split_transform_dict(third_dict, base_dict, 0)
    assert yaml_experiment_utils.get_length_of_params_dict(split_dict) == 1
    
    # Test that split_transform_dict raises IndexError when index is too large
    with pytest.raises(IndexError):
        third_dict = load_yaml_from_file['transforms'][2]
        base_dict = yaml_experiment_utils.get_transform_base_dict(third_dict)
        yaml_experiment_utils.split_transform_dict(third_dict, base_dict, 10)
    

