import yaml
from copy import deepcopy
from collections import defaultdict
from dataclasses import dataclass
from pydantic import BaseModel, ValidationError, field_validator
from typing import List, Optional, Dict


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
                print(
                    f"Error: {column[TransformKeys.TRANSFORMATIONS_KEY]} is not parsed as a list, yaml file is not properly formatted, make sure that transformations names are preceded by a dash"
                )
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
    base_dict = defaultdict(str, deepcopy(dict_to_split))
    # Reset all the params keys in the nested dicts
    for column in base_dict[TransformKeys.COLUMN_KEY]:
        for transformation in column[TransformKeys.TRANSFORMATIONS_KEY]:
            # type check that transformation[PARAM_KEY] is a dictionary
            if isinstance(transformation[TransformKeys.PARAMS_KEY], dict):
                transformation[TransformKeys.PARAMS_KEY] = {}
    return dict(base_dict)


def split_transform_dict(
    dict_to_split: dict, base_dict: dict, split_index: int
) -> dict:
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
        for transformation_index, transformation in enumerate(
            column[TransformKeys.TRANSFORMATIONS_KEY]
        ):
            if isinstance(transformation[TransformKeys.PARAMS_KEY], dict):
                # create a new empty dictionary that has the same keys as the transformation[PARAMS_KEY] dict
                temp_dict = dict.fromkeys(
                    transformation[TransformKeys.PARAMS_KEY].keys()
                )
                for key, value in transformation[TransformKeys.PARAMS_KEY].items():
                    if isinstance(value, list):
                        # check that the list has more than one element
                        if len(value) > 1:
                            try:
                                temp_dict[key] = value[split_index]
                            except IndexError:
                                print(
                                    f"Error: {value} is not long enough to be split at index {split_index}"
                                )
                                raise
                        else:
                            temp_dict[key] = value[0]
                    else:
                        temp_dict[key] = value
                split_dict[TransformKeys.COLUMN_KEY][column_index][
                    TransformKeys.TRANSFORMATIONS_KEY
                ][transformation_index][TransformKeys.PARAMS_KEY] = temp_dict

    return split_dict


def get_all_transform_dicts(dict_to_split: dict) -> list[dict]:
    """Splits a transform dictionary into multiple dictionaries with single parameter values.

    Takes a transform dictionary containing parameter lists and creates multiple new dictionaries,
    each containing single parameter values from the corresponding indices of the parameter lists.

    Args:
        dict_to_split (dict): The transform dictionary containing parameter lists to split

    Returns:
        list[dict]: A list of dictionaries, each with the same structure but containing only
            single parameter values from sequential indices
    """
    length_of_params_dict = get_length_of_params_dict(dict_to_split)
    base_dict = get_transform_base_dict(dict_to_split)
    transform_dicts = []
    for i in range(length_of_params_dict):
        transform_dicts.append(split_transform_dict(dict_to_split, base_dict, i))
    return transform_dicts


def dump_yaml_list_into_files(
    yaml_list: list[dict], directory_path: str, base_name: str
) -> None:
    for i, yaml_dict in enumerate(yaml_list):
        with open(f"{directory_path}/{base_name}_{i}.yaml", "w") as f:
            yaml.dump(yaml_dict, f)


class YamlGlobalParams(BaseModel):
    seed: int


class YamlColumnsEncoder(BaseModel):
    name: str
    params: Optional[Dict[str, str]]  # The dict can contain or not data


class YamlColumns(BaseModel):
    column_name: str
    column_type: str
    data_type: str
    encoder: List[YamlColumnsEncoder]


class YamlTransformColumnsTransformation(BaseModel):
    name: str
    params: Optional[Dict[str, list]]


class YamlTransformColumns(BaseModel):
    column_name: str
    transformations: List[YamlTransformColumnsTransformation]


class YamlTransform(BaseModel):
    transformation_name: str
    columns: List[YamlTransformColumns]

    @field_validator('columns')
    @classmethod
    def validate_param_lists_across_columns(cls, columns) -> List[YamlTransformColumns]:
        # Get all parameter list lengths across all columns and transformations
        all_list_lengths = set()
        
        for column in columns:
            for transformation in column.transformations:
                if transformation.params:
                    for param_value in transformation.params.values():
                        if isinstance(param_value, list):
                            if len(param_value) > 0:  # Non-empty list
                                all_list_lengths.add(len(param_value))
        
        # Skip validation if no lists found
        if not all_list_lengths:
            return columns
            
        # Check if all lists either have length 1, or all have the same length
        all_list_lengths.discard(1)  # Remove length 1 as it's always valid
        if len(all_list_lengths) > 1:  # Multiple different lengths found
            raise ValueError("All parameter lists across columns must either contain one element or have the same length")
        
        return columns


class YamlSplit(BaseModel):
    split_method: str
    params: Optional[Dict[str, list]]


class YamlConfigDict(BaseModel):
    global_params: YamlGlobalParams
    columns: List[YamlColumns]
    transforms: List[YamlTransform]
    split: List[YamlSplit]


class YamlSchema(BaseModel):
    yaml_conf: YamlConfigDict


def check_yaml_schema(config_yaml: str) -> str:
    """
    Using pydantic this function confirms that the fields have the correct input type
    If the children field is specific to a parent, the children fields class is hosted in the parent fields class

    If any field in not the right type, the function prints an error message explaining the problem and exits the python code

    Args:
        config_yaml (dict): The dict containing the fields of the yaml configuration file

    Returns:
        None
    """
    try:
        YamlSchema(yaml_conf=config_yaml)
    except ValidationError as e:
        print(e)
        raise ValueError("Wrong type on a field, see the pydantic report above")  # Crashes in case of an error raised
