import pytest
import yaml

from stimulus.utils import yaml_data

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

@pytest.fixture
def load_wrong_type_yaml():
    """
    Fixture that loads a YAML configuration file with wrong typing

    This fixture loads a YAML configuration file containing DNA experiment parameters,
    BUT the wrong value type have been assigned to some fiels

    Returns:
        dict: The parsed YAML configuration as a dictionnary.
    """
    with open("tests/test_data/yaml_files/wrong_field_type.yaml") as f:
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
    assert yaml_data.get_length_of_params_dict(first_dict) == 1

    second_dict = load_yaml_from_file['transforms'][1]
    assert yaml_data.get_length_of_params_dict(second_dict) == 3

    third_dict = load_yaml_from_file['transforms'][2]
    assert yaml_data.get_length_of_params_dict(third_dict) == 4

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
    base_dict = yaml_data.get_transform_base_dict(first_dict)
    assert base_dict.keys() == first_dict.keys()
    assert base_dict[yaml_data.TransformKeys.COLUMN_KEY][0].keys() == first_dict[yaml_data.TransformKeys.COLUMN_KEY][0].keys()
    assert base_dict[yaml_data.TransformKeys.COLUMN_KEY][0][yaml_data.TransformKeys.TRANSFORMATIONS_KEY][0].keys() == first_dict[yaml_data.TransformKeys.COLUMN_KEY][0][yaml_data.TransformKeys.TRANSFORMATIONS_KEY][0].keys()
    # Check that params fields are empty
    for column in base_dict[yaml_data.TransformKeys.COLUMN_KEY]:
        for transformation in column[yaml_data.TransformKeys.TRANSFORMATIONS_KEY]:
            assert transformation[yaml_data.TransformKeys.PARAMS_KEY] == {} or transformation[yaml_data.TransformKeys.PARAMS_KEY] == None

    second_dict = load_yaml_from_file['transforms'][1]
    base_dict = yaml_data.get_transform_base_dict(second_dict)
    assert base_dict.keys() == second_dict.keys()
    assert base_dict[yaml_data.TransformKeys.COLUMN_KEY][0].keys() == second_dict[yaml_data.TransformKeys.COLUMN_KEY][0].keys()
    assert base_dict[yaml_data.TransformKeys.COLUMN_KEY][0][yaml_data.TransformKeys.TRANSFORMATIONS_KEY][0].keys() == second_dict[yaml_data.TransformKeys.COLUMN_KEY][0][yaml_data.TransformKeys.TRANSFORMATIONS_KEY][0].keys()
    # Check that params fields are empty
    for column in base_dict[yaml_data.TransformKeys.COLUMN_KEY]:
        for transformation in column[yaml_data.TransformKeys.TRANSFORMATIONS_KEY]:
            assert transformation[yaml_data.TransformKeys.PARAMS_KEY] == {} or transformation[yaml_data.TransformKeys.PARAMS_KEY] == None

    third_dict = load_yaml_from_file['transforms'][2]
    base_dict = yaml_data.get_transform_base_dict(third_dict)
    assert base_dict.keys() == third_dict.keys()
    assert base_dict[yaml_data.TransformKeys.COLUMN_KEY][0].keys() == third_dict[yaml_data.TransformKeys.COLUMN_KEY][0].keys()
    assert base_dict[yaml_data.TransformKeys.COLUMN_KEY][0][yaml_data.TransformKeys.TRANSFORMATIONS_KEY][0].keys() == third_dict[yaml_data.TransformKeys.COLUMN_KEY][0][yaml_data.TransformKeys.TRANSFORMATIONS_KEY][0].keys()
    # Check that params fields are empty
    for column in base_dict[yaml_data.TransformKeys.COLUMN_KEY]:
        for transformation in column[yaml_data.TransformKeys.TRANSFORMATIONS_KEY]:
            assert transformation[yaml_data.TransformKeys.PARAMS_KEY] == {} or transformation[yaml_data.TransformKeys.PARAMS_KEY] == None

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
    base_dict = yaml_data.get_transform_base_dict(first_dict)
    split_dict = yaml_data.split_transform_dict(first_dict, base_dict, 0)
    assert yaml_data.get_length_of_params_dict(split_dict) == 1

    second_dict = load_yaml_from_file['transforms'][1]
    base_dict = yaml_data.get_transform_base_dict(second_dict)
    split_dict = yaml_data.split_transform_dict(second_dict, base_dict, 0)
    assert yaml_data.get_length_of_params_dict(split_dict) == 1

    third_dict = load_yaml_from_file['transforms'][2]
    base_dict = yaml_data.get_transform_base_dict(third_dict)
    split_dict = yaml_data.split_transform_dict(third_dict, base_dict, 0)
    assert yaml_data.get_length_of_params_dict(split_dict) == 1
    
    # Test that split_transform_dict raises IndexError when index is too large
    with pytest.raises(IndexError):
        third_dict = load_yaml_from_file['transforms'][2]
        base_dict = yaml_data.get_transform_base_dict(third_dict)
        yaml_data.split_transform_dict(third_dict, base_dict, 10)
    
def test_get_all_transform_dicts(load_yaml_from_file):
    """Tests the get_all_transform_dicts function.

    This test verifies that get_all_transform_dicts correctly generates a list of dictionaries,
    each containing a single parameter value from a transform dictionary with multiple parameter lists.
    It checks:
    1. A transform with no parameters (should return a list with one unchanged dictionary)
    2. A transform with a single parameter list (should return a list with one dictionary containing the single value)
    3. A transform with multiple parameter lists (should return a list with dictionaries containing corresponding values)
    """
    first_dict = load_yaml_from_file['transforms'][0]
    split_dicts = yaml_data.get_all_transform_dicts(first_dict)
    assert len(split_dicts) == 1
    assert split_dicts[0] == first_dict
    second_dict = load_yaml_from_file['transforms'][1]
    split_dicts = yaml_data.get_all_transform_dicts(second_dict)
    assert len(split_dicts) == 3

    third_dict = load_yaml_from_file['transforms'][2]
    split_dicts = yaml_data.get_all_transform_dicts(third_dict)
    assert len(split_dicts) == 4

@pytest.mark.parametrize("test_input", [("load_yaml_from_file", False), ("load_wrong_type_yaml", True)])
def test_check_yaml_schema(request, test_input):
    """
    Tests the pydantix field type assertion made for the input yaml holding the description of the pipeline

    This test uses the @pytest.mark.parametrize to test multiple cases: one where the fields are correct and one where it isn't
    The first value uses a fixture to load a file, the second indicates if the function will call value_error
    
    Args:
        test_input (tuple): 1) the fixture for the file, 2) Boolean value about expecting an value_error call
    
    Returns:
        None
    """
    data = request.getfixturevalue(test_input[0])
    expect_value_error = test_input[1] # defines if we expect an value_error call

    if not expect_value_error:
        yaml_data.check_yaml_schema(data)
        assert True # In case the value_error is not called returns a True value else it stops the test and returns an error
    else: # if we expect an value_error we check that the wrong fields raise an value_error call
        with pytest.raises(ValueError, match="Wrong type on a field, see the pydantic report above") as e: # Catches the ValueError call
            yaml_data.check_yaml_schema(data)
            assert True
