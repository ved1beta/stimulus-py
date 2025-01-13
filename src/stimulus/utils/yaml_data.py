import yaml
from copy import deepcopy
from collections import defaultdict
from dataclasses import dataclass
from pydantic import BaseModel, ValidationError, field_validator
from typing import List, Optional, Dict

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
    split_input_columns: Optional[List[str]]

class YamlConfigDict(BaseModel):
    global_params: YamlGlobalParams
    columns: List[YamlColumns]
    transforms: List[YamlTransform]
    split: List[YamlSplit]

class YamlSubConfigDict(BaseModel):
    global_params: YamlGlobalParams
    columns: List[YamlColumns]
    transforms: YamlTransform
    split: YamlSplit

class YamlSchema(BaseModel):
    yaml_conf: YamlConfigDict

def extract_transform_parameters_at_index(transform: YamlTransform, index: int = 0) -> YamlTransform:
    """Get a transform with parameters at the specified index.
    
    Args:
        transform: The original transform containing parameter lists
        index: Index to extract parameters from (default 0)
        
    Returns:
        A new transform with single parameter values at the specified index
    """
    # Create a copy of the transform
    new_transform = YamlTransform(**transform.model_dump())

    # Process each column and transformation
    for column in new_transform.columns:
        for transformation in column.transformations:
            if transformation.params:
                # Convert each parameter list to single value at index
                new_params = {}
                for param_name, param_value in transformation.params.items():
                    if isinstance(param_value, list):
                        new_params[param_name] = param_value[index]
                    else:
                        new_params[param_name] = param_value
                transformation.params = new_params
                
    return new_transform

def expand_transform_parameter_combinations(transform: YamlTransform) -> list[YamlTransform]:
    """Get all possible transforms by extracting parameters at each valid index.
    
    For a transform with parameter lists, creates multiple new transforms, each containing
    single parameter values from the corresponding indices of the parameter lists.
    
    Args:
        transform: The original transform containing parameter lists
        
    Returns:
        A list of transforms, each with single parameter values from sequential indices
    """
    # Find the length of parameter lists - we only need to check the first list we find
    # since all lists must have the same length (enforced by pydantic validator)
    max_length = 1
    for column in transform.columns:
        for transformation in column.transformations:
            if transformation.params:
                list_lengths = [len(v) for v in transformation.params.values() 
                              if isinstance(v, list) and len(v) > 1]
                if list_lengths:
                    max_length = list_lengths[0]  # All lists have same length due to validator
                    break
    
    # Generate a transform for each index
    transforms = []
    for i in range(max_length):
        transforms.append(extract_transform_parameters_at_index(transform, i))
        
    return transforms


def dump_yaml_list_into_files(
    yaml_list: list[dict], directory_path: str, base_name: str
) -> None:
    for i, yaml_dict in enumerate(yaml_list):
        with open(f"{directory_path}/{base_name}_{i}.yaml", "w") as f:
            yaml.dump(yaml_dict, f)

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
