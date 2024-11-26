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

def get_length_of_params_dict(dict_to_split: dict) -> int:
    """
    This function takes as input a dictionary and returns the length of a params keys in the nested dictionaries (assumes all lengths are equal)
    """
    for column in dict_to_split[TransformKeys.COLUMN_KEY]:
        for transformation in column[TransformKeys.TRANSFORMATIONS_KEY]:
            if isinstance(transformation[TransformKeys.PARAMS_KEY], dict):
                # check for lists within the params dict
                for key, value in transformation[TransformKeys.PARAMS_KEY].items():
                    if isinstance(value, list):
                        # check that the list has more than one element
                        if len(value) > 1:
                            return len(value)
    return 1

def get_transform_base_dict(dict_to_split: dict) -> dict:
    """
    This function takes as input a dictionary to expand and returns a dictionary with the params keys reset to empty dictionaries.
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
    """
    This function takes as input a dictionary to split and returns a dictionary with a single param value.
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
                            temp_dict[key] = value[split_index]
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
