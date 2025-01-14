"""Loaders serve as interfaces between the CSV master class and custom methods.

Mainly, three types of custom methods are supported:
- Encoders: methods for encoding data before it is fed into the model
- Data transformers: methods for transforming data (i.e. augmenting, noising...)
- Splitters: methods for splitting data into train, validation and test sets

Loaders are built from an input config YAML file which format is described in the documentation, you can find an example here: tests/test_data/dna_experiment/dna_experiment_config_template.yaml
"""

from abc import ABC
from typing import Any
from collections import defaultdict


import inspect
import yaml

from stimulus.data.encoding import encoders as encoders
from stimulus.data.splitters import splitters as splitters
from stimulus.data.transform import data_transformation_generators as data_transformation_generators

class AbstractLoader(ABC):
    """Abstract base class for defining loaders."""

    def get_config_from_yaml(self, yaml_path: str) -> dict:
        """Loads experiment configuration from a YAML file.
        
        Args:
            yaml_path (str): Path to the YAML config file

        Returns:
            dict: The loaded configuration dictionary
        """
        with open(yaml_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    
class EncoderLoader(AbstractLoader):
    """Class for loading encoders from a config file."""

    def __init__(self, seed: float = None) -> None:
        self.seed = seed

    def initialize_column_encoders_from_config(self, config: dict) -> None:
        """Build the loader from a config dictionary.
        
        Args:
            config (dict): Configuration dictionary containing field names (column_name) and their encoder specifications.
        """
        for field in config:
            encoder = self.get_encoder(field["encoder"][0]["name"], field["encoder"][0]["params"])
            self.set_encoder_as_attribute(field["column_name"], encoder)

    def get_function_encode_all(self, field_name: str) -> Any:
        """Gets the encoding function for a specific field.
        
        Args:
            field_name (str): The field name to get the encoder for

        Returns:
            Any: The encode_all function for the specified field
        """
        return getattr(self, field_name)["encoder"].encode_all

    def get_encoder(self, encoder_name: str, encoder_params: dict = None) -> Any:
        """Gets an encoder object from the encoders module and initializes it with the given parametersß.
        
        Args:
            encoder_name (str): The name of the encoder to get
            encoder_params (dict): The parameters for the encoder

        Returns:
            Any: The encoder function for the specified field and parameters
        """

        try:
            return getattr(encoders, encoder_name)(**encoder_params)
        except AttributeError:
            print(f"Encoder '{encoder_name}' not found in the encoders module.")
            print(f"Available encoders: {[name for name, obj in encoders.__dict__.items() if isinstance(obj, type) and name not in ('ABC', 'Any')]}")
            raise

        except TypeError:
            if encoder_params is None:
                return getattr(encoders, encoder_name)()
            else:
                print(f"Encoder '{encoder_name}' has incorrect parameters: {encoder_params}")
                print(f"Expected parameters for '{encoder_name}': {inspect.signature(getattr(encoders, encoder_name))}")
                raise

    def set_encoder_as_attribute(self, field_name: str, encoder: encoders.AbstractEncoder) -> None:
        """Sets the encoder as an attribute of the loader.
        
        Args:
            field_name (str): The name of the field to set the encoder for
            encoder (encoders.AbstractEncoder): The encoder to set
        """
        setattr(self, field_name, {"encoder": encoder})

class TransformLoader(AbstractLoader):
    """Class for loading transformations from a config file."""

    def __init__(self, seed: float = None) -> None:
        self.seed = seed

    def get_data_transformer(self, transformation_name: str) -> Any:
        """Gets a transformer object from the transformers module.
        
        Args:
            transformation_name (str): The name of the transformer to get

        Returns:
            Any: The transformer function for the specified transformation
        """
        try:
            return getattr(data_transformation_generators, transformation_name)()
        except AttributeError:
            print(f"Transformer '{transformation_name}' not found in the transformers module.")
            print(f"Available transformers: {[name for name, obj in data_transformation_generators.__dict__.items() if isinstance(obj, type) and name not in ('ABC', 'Any')]}")
            raise
    
    def set_data_transformer_as_attribute(self, field_name: str, data_transformer: Any) -> None:
        """Sets the data transformer as an attribute of the loader.
        
        Args:
            field_name (str): The name of the field to set the data transformer for
            data_transformer (Any): The data transformer to set
        """
        setattr(self, field_name, {"data_transformation_generators": data_transformer})

    def initialize_column_data_transformers_from_config(self, config: dict) -> None:
        """Build the loader from a config dictionary.
        
        Args:
            config (dict): Configuration dictionary containing transforms configurations.
                Each transform can specify multiple columns and their transformations.
                The method will organize transformers by column, ensuring each column
                has all its required transformations.
        """
        
        
        # Use defaultdict to automatically initialize empty lists
        column_transformers = defaultdict(list)
        
        # First pass: collect all transformations by column
        for transform_group in config:
            for column in transform_group["columns"]:
                col_name = column["column_name"]
                
                # Process each transformation for this column
                for transform_spec in column["transformations"]:
                    # Create transformer instance
                    transformer = self.get_data_transformer(transform_spec["name"])
                    
                    # Get transformer class for comparison
                    transformer_type = type(transformer)
                    
                    # Add transformer if its type isn't already present
                    if not any(isinstance(existing, transformer_type) 
                             for existing in column_transformers[col_name]):
                        column_transformers[col_name].append(transformer)
        
        # Second pass: set all collected transformers as attributes
        for col_name, transformers in column_transformers.items():
            self.set_data_transformer_as_attribute(col_name, transformers)
    
class SplitLoader(AbstractLoader):
    """Class for loading splitters from a config file."""

    def __init__(self, seed: float = None) -> None:
        self.seed = seed

    def get_function_split(self) -> Any:
        """Gets the function for splitting the data.
        
        Args:
            split_method (str): Name of the split method to use

        Returns:
            Any: The split function for the specified method
        """
        return self.split.get_split_indexes
    
    def get_splitter(self, splitter_name: str) -> Any:
        """Gets a splitter object from the splitters module.
        
        Args:
            splitter_name (str): The name of the splitter to get

        Returns:
            Any: The splitter function for the specified splitter
        """
        return getattr(splitters, splitter_name)()
    
    def set_splitter_as_attribute(self, field_name: str, splitter: Any) -> None:
        """Sets the splitter as an attribute of the loader.
        
        Args:
            field_name (str): The name of the field to set the splitter for
            splitter (Any): The splitter to set
        """
        setattr(self, field_name, {"splitter": splitter})
