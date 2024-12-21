"""Experiments are classes parsed by CSV master classes to run experiments.
Conceptually, experiment classes contain data types, transformations etc and are used to duplicate the input data into many datasets.
Here we provide standard experiments as well as an absctract class for users to implement their own.
"""

from abc import ABC
from typing import Any


import inspect
import yaml

from stimulus.data.encoding import encoders as encoders
from stimulus.data.splitters import splitters as splitters
from stimulus.data.transform import data_transformation_generators as data_transformation_generators


class AbstractExperiment(ABC):
    """Abstract base class for defining experiments.
    
    This class provides the base functionality for experiment classes that handle data encoding,
    transformations and splitting. All data type argument names must be lowercase.

    Attributes:
        seed (float): Optional random seed for reproducibility
        split (dict): Dictionary mapping split method names to splitter objects. Defaults to
            RandomSplitter.

    Note:
        Check the data_types module for implemented data types.
    """

    def __init__(self, seed: float = None) -> None:
        # allow ability to add a seed for reproducibility
        self.seed = seed
        # added because if the user does not define this it does not crach the get_function_split, random split works for every class afteralll
        self.split = {"RandomSplitter": splitters.RandomSplitter()}

    def get_function_encode_all(self, data_type: str) -> Any:
        """Gets the encoding function for a specific data type.
        
        Args:
            data_type (str): The data type to get the encoder for

        Returns:
            Any: The encode_all function for the specified data type
        """
        return getattr(self, data_type)["encoder"].encode_all

    def get_data_transformer(self, data_type: str, transformation_generator: str) -> Any:
        """Gets the transformation function for data augmentation.
        
        Args:
            data_type (str): The data type to transform
            transformation_generator (str): Name of the transformation to apply

        Returns:
            Any: The transformation function for the specified data type and transformation
        """
        return getattr(self, data_type)["data_transformation_generators"][transformation_generator]

    def get_function_split(self, split_method: str) -> Any:
        """Gets the function for splitting the data.
        
        Args:
            split_method (str): Name of the split method to use

        Returns:
            Any: The split function for the specified method
        """
        return self.split[split_method].get_split_indexes
    
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

    def get_encoder(self, encoder_name: str, encoder_params: dict) -> Any:
        """Gets the encoder function for a specific data type.
        
        Args:
            encoder_name (str): The name of the encoder to get
            encoder_params (dict): The parameters for the encoder

        Returns:
            Any: The encoder function for the specified data type and parameters
        """

        if encoder_params is not None:
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

    def set_encoder_as_attribute(self, column_name: str, encoder: encoders.AbstractEncoder) -> None:
        """Sets the encoder as an attribute of the experiment class.
        
        Args:
            column_name (str): The name of the column to set the encoder for
            encoder (encoders.AbstractEncoder): The encoder to set
        """
        setattr(self, column_name, {"encoder": encoder})
    
    def build_experiment_class_encoder_from_config(self, columns: list) -> None:
        """Build the experiment class from a config dictionary.
        
        Args:
            config (dict): Configuration dictionary containing column names and their encoder specifications.
        """
        
        for column in columns:
            encoder = self.get_encoder(column["encoder"][0]["name"], column["encoder"][0]["params"])
            self.set_encoder_as_attribute(column["column_name"], encoder)

class DnaToFloatExperiment(AbstractExperiment):
    """Class for dealing with DNA to float predictions (for instance regression from DNA sequence to CAGE value)"""

    def __init__(self) -> None:
        super().__init__()
        self.dna = {
            "encoder": encoders.TextOneHotEncoder(alphabet="acgt"),
            "data_transformation_generators": {
                "UniformTextMasker": data_transformation_generators.UniformTextMasker(mask="N"),
                "ReverseComplement": data_transformation_generators.ReverseComplement(),
                "GaussianChunk": data_transformation_generators.GaussianChunk(),
            },
        }
        self.float = {
            "encoder": encoders.FloatEncoder(),
            "data_transformation_generators": {"GaussianNoise": data_transformation_generators.GaussianNoise()},
        }
        self.split = {"RandomSplitter": splitters.RandomSplitter()}


class ProtDnaToFloatExperiment(DnaToFloatExperiment):
    """Class for dealing with Protein and DNA to float predictions (for instance regression from Protein sequence + DNA sequence to binding score)"""

    def __init__(self) -> None:
        super().__init__()
        self.prot = {
            "encoder": encoders.TextOneHotEncoder(alphabet="acdefghiklmnpqrstvwy"),
            "data_transformation_generators": {
                "UniformTextMasker": data_transformation_generators.UniformTextMasker(mask="X"),
            },
        }


class TitanicExperiment(AbstractExperiment):
    """Class for dealing with the Titanic dataset as a test format."""

    def __init__(self) -> None:
        super().__init__()
        self.int_class = {"encoder": encoders.IntEncoder(), "data_transformation_generators": {}}
        self.str_class = {"encoder": encoders.StrClassificationIntEncoder(), "data_transformation_generators": {}}
        self.int_reg = {"encoder": encoders.IntRankEncoder(), "data_transformation_generators": {}}
        self.float_rank = {"encoder": encoders.FloatRankEncoder(), "data_transformation_generators": {}}

