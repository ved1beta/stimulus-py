import yaml
from copy import deepcopy
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class TransformKeys:
    COLUMN_NAME_KEY: str = "column_name"
    COLUMN_KEY: str = "columns"
    NAME_KEY: str = "name"
    PARAMS_KEY: str = "params"
    TRANSFORMATIONS_KEY: str = "transformations"

def get_length_of_params_dict(input_dict: dict) -> int:
    """
    This function takes as input a dictionary and returns the length of a params keys in the nested dictionaries
     
    We assume nested dictionaries are of the same length, if there are no parameters, we return 1.

    Args:
        input_dict (dict): dictionary to split

    Returns:
        int: length of params keys in the nested dictionaries
    """
    for column in input_dict[TransformKeys.COLUMN_KEY]:
        for transformation in column[TransformKeys.TRANSFORMATIONS_KEY]:
            try:
                if isinstance(transformation[TransformKeys.PARAMS_KEY], dict):
                    # check for lists within the params dict
                    for key, value in transformation[TransformKeys.PARAMS_KEY].items():
                        if isinstance(value, list):
                            # check that the list has more than one element
                            if len(value) > 1:
                                return len(value)
            except TypeError:
                print(f"Error: {column[TransformKeys.TRANSFORMATIONS_KEY]} is not parsed as a list, yaml file is not properly formatted, make sure that transformations names are preceded by a dash")
    return 1

def get_transform_base_dict(dict_to_split: dict) -> dict:
    """Gets a base dictionary with empty parameter values.

    Takes a dictionary containing transform configurations and creates a base dictionary
    with the same structure but with empty parameter values. This is used as a template
    for creating split dictionaries.

    Args:
        dict_to_split: A dictionary containing transform configurations with parameters
            that need to be split.

    Returns:
        A dictionary with the same structure as the input but with empty parameter values.
        The dictionary uses defaultdict to return empty strings for missing keys.

    Raises:
        TypeError: If the transformations are not properly formatted as a list in the YAML.
    """
    # Create a defaultdict that will return empty string for missing keys
    base_dict  = defaultdict(str, deepcopy(dict_to_split))
    # Reset all the params keys in the nested dicts
    for column in base_dict[TransformKeys.COLUMN_KEY]:
        for transformation in column[TransformKeys.TRANSFORMATIONS_KEY]:
            # type check that transformation[PARAM_KEY] is a dictionary
            if isinstance(transformation[TransformKeys.PARAMS_KEY], dict):
                transformation[TransformKeys.PARAMS_KEY] = {}
    return dict(base_dict)

def split_transform_dict(dict_to_split: dict, base_dict: dict, split_index: int) -> dict: 
    """Splits a transform dictionary to extract a single parameter value.

    Takes a transform dictionary containing parameter lists and creates a new dictionary
    with only the parameter values at the specified split index. Uses a base dictionary
    as a template for the structure.

    Args:
        dict_to_split (dict): The transform dictionary containing parameter lists to split
        base_dict (dict): Template dictionary with empty parameter values
        split_index (int): Index to extract from parameter lists

    Returns:
        dict: A new dictionary with the same structure but containing only single parameter
            values at the specified index
    """

    split_dict = deepcopy(base_dict)

    for column_index, column in enumerate(dict_to_split[TransformKeys.COLUMN_KEY]):
        for transformation_index, transformation in enumerate(column[TransformKeys.TRANSFORMATIONS_KEY]):
            if isinstance(transformation[TransformKeys.PARAMS_KEY], dict):
                # create a new empty dictionary that has the same keys as the transformation[PARAMS_KEY] dict 
                temp_dict = dict.fromkeys(transformation[TransformKeys.PARAMS_KEY].keys())
                for key, value in transformation[TransformKeys.PARAMS_KEY].items():
                    if isinstance(value, list):
                        # check that the list has more than one element
                        if len(value) > 1:
                            try:
                                temp_dict[key] = value[split_index]
                            except IndexError:
                                print(f"Error: {value} is not long enough to be split at index {split_index}")
                                raise
                        else:
                            temp_dict[key] = value[0]
                    else:
                        temp_dict[key] = value
                split_dict[TransformKeys.COLUMN_KEY][column_index][TransformKeys.TRANSFORMATIONS_KEY][transformation_index][TransformKeys.PARAMS_KEY] = temp_dict

    return split_dict

def get_all_transform_dicts(dict_to_split: dict) -> list[dict]:
    """
    This function takes as input a dictionary to split and returns a list of dictionaries, each with a single param value.
    """
    length_of_params_dict = get_length_of_params_dict(dict_to_split)
    base_dict = get_transform_base_dict(dict_to_split)
    transform_dicts = []
    for i in range(length_of_params_dict):
        transform_dicts.append(split_transform_dict(dict_to_split, base_dict, i))
    return transform_dicts

def dump_yaml_list_into_files(yaml_list: list[dict], directory_path: str, base_name: str) -> None:
    for i, yaml_dict in enumerate(yaml_list):
        with open(f"{directory_path}/{base_name}_{i}.yaml", "w") as f:
            yaml.dump(yaml_dict, f)
