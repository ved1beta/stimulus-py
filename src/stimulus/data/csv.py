"""This file contains the parser class for parsing an input CSV file which is the STIMULUS data format.

The file contains a header column row where column names are formated as is :
name:category:type

name is straightforward, it is the name of the column
category corresponds to any of those three values : input, meta, or label. Input is the input of the deep learning model, label is the output (what needs to be predicted) and meta corresponds to metadata not used during training (could be used for splitting).
type corresponds to the data type of the columns, as specified in the types module.

The parser is a class that takes as input a CSV file and a experiment class that defines data types to be used, noising procedures, splitting etc.
"""

from functools import partial
from typing import Any, Tuple, Union
from abc import ABC

import numpy as np
import polars as pl
import yaml
import stimulus.data.experiments as experiments
import torch

class DatasetManager:
    """Class for managing the dataset.
    
    This class handles loading and organizing dataset configuration from YAML files.
    It manages column categorization into input, label and meta types based on the config.

    Attributes:
        config (dict): The loaded configuration dictionary from YAML
        column_categories (dict): Dictionary mapping column types to lists of column names

    Methods:
        _load_config(config_path: str) -> dict: Loads the config from a YAML file.
        categorize_columns_by_type() -> dict: Organizes the columns into input, label, meta based on the config.
    """

    def __init__(self, 
                config_path: str,
                ) -> None:
        self.config = self._load_config(config_path)
        self.column_categories = self.categorize_columns_by_type()

    def categorize_columns_by_type(self) -> dict:
        """Organizes columns from config into input, label, and meta categories.

        Reads the column definitions from the config and sorts them into categories
        based on their column_type field.

        Returns:
            dict: Dictionary containing lists of column names for each category:
                {
                    "input": ["col1", "col2"],  # Input columns
                    "label": ["target"],        # Label/output columns  
                    "meta": ["id"]     # Metadata columns
                }

        Example:
            >>> manager = DatasetManager("config.yaml")
            >>> categories = manager.categorize_columns_by_type()
            >>> print(categories)
            {
                'input': ['hello', 'bonjour'],
                'label': ['ciao'],
                'meta': ["id"]
            }
        """
        input_columns = []
        label_columns = []
        meta_columns = []
        for column in self.config["columns"]:
            if column["column_type"] == "input":
                input_columns.append(column["column_name"])
            elif column["column_type"] == "label":
                label_columns.append(column["column_name"])
            elif column["column_type"] == "meta":
                meta_columns.append(column["column_name"])

        return {"input": input_columns, "label": label_columns, "meta": meta_columns}

    def _load_config(self, config_path: str) -> dict:
        """Loads and parses a YAML configuration file.

        Args:
            config_path (str): Path to the YAML config file

        Returns:
            dict: Parsed configuration dictionary

        Example:
            >>> manager = DatasetManager()
            >>> config = manager._load_config("config.yaml")
            >>> print(config["columns"][0]["column_name"])
            'hello'
        """
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def get_split_columns(self) -> str:
        """Get the columns that are used for splitting."""
        return self.config["split"]["split_input_columns"]
  

class EncodeManager:
    """Manages the encoding of data columns using configured encoders.

    This class handles encoding of data columns based on the encoders specified in the
    configuration. It uses an EncoderLoader to get the appropriate encoder for each column
    and applies the encoding.

    Attributes:
        encoder_loader (experiments.EncoderLoader): Loader that provides encoders based on config.

    Example:
        >>> encoder_loader = EncoderLoader(config)
        >>> encode_manager = EncodeManager(encoder_loader)
        >>> data = ["ACGT", "TGCA", "GCTA"] 
        >>> encoded = encode_manager.encode_column("dna_seq", data)
        >>> print(encoded.shape)
        torch.Size([3, 4, 4])  # 3 sequences, length 4, one-hot encoded
    """

    def __init__(self, 
                encoder_loader: experiments.EncoderLoader,
                ) -> None:
        """Initializes the EncodeManager.

        Args:
            encoder_loader: Loader that provides encoders based on configuration.
        """
        self.encoder_loader = encoder_loader

    def encode_column(self, column_name: str, column_data: list) -> torch.Tensor:
        """Encodes a column of data using the configured encoder.

        Gets the appropriate encoder for the column from the encoder_loader and uses it
        to encode all the data in the column.

        Args:
            column_name: Name of the column to encode.
            column_data: List of data values from the column to encode.

        Returns:
            Encoded data as a torch.Tensor. The exact shape depends on the encoder used.

        Example:
            >>> data = ["ACGT", "TGCA"]
            >>> encoded = encode_manager.encode_column("dna_seq", data)
            >>> print(encoded.shape)
            torch.Size([2, 4, 4])  # 2 sequences, length 4, one-hot encoded
        """
        encode_all_function = self.encoder_loader.get_function_encode_all(column_name)
        return encode_all_function(column_data)

    def encode_columns(self, column_data: dict) -> dict:
        """Encodes multiple columns of data using the configured encoders.

        Gets the appropriate encoder for each column from the encoder_loader and encodes
        all data values in those columns.

        Args:
            column_data: Dict mapping column names to lists of data values to encode.

        Returns:
            Dict mapping column names to their encoded tensors. The exact shape of each
            tensor depends on the encoder used for that column.

        Example:
            >>> data = {
            ...     "dna_seq": ["ACGT", "TGCA"],
            ...     "labels": ["1", "2"]
            ... }
            >>> encoded = encode_manager.encode_columns(data)
            >>> print(encoded["dna_seq"].shape)
            torch.Size([2, 4, 4])  # 2 sequences, length 4, one-hot encoded
        """
        return {col: self.encode_column(col, values) for col, values in column_data.items()}

class TransformManager:
    """Class for managing the transformations."""

    def __init__(self, 
                transform_loader: experiments.TransformLoader,
                ) -> None:
        self.transform_loader = transform_loader

class SplitManager:
    """Class for managing the splitting."""

    def __init__(self, 
                split_loader: experiments.SplitLoader,
                ) -> None:
        self.split_loader = split_loader

    def get_split_indices(self, data: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the indices for train, validation, and test splits."""
        return self.split_loader.get_function_split()(data)

class DatasetHandler:
    """Main class for handling dataset loading, encoding, transformation and splitting.
    
    This class coordinates the interaction between different managers to process
    CSV datasets according to the provided configuration.

    Attributes:
        encoder_manager (EncodeManager): Manager for handling data encoding operations.
        transform_manager (TransformManager): Manager for handling data transformations.
        split_manager (SplitManager): Manager for handling dataset splitting.
        dataset_manager (DatasetManager): Manager for organizing dataset columns and config.
    """

    def __init__(self,
                 encoder_loader: experiments.EncoderLoader,
                 transform_loader: experiments.TransformLoader,
                 split_loader: experiments.SplitLoader,
                 config_path: str,
                 csv_path: str,
                 ) -> None:
        """Initialize the DatasetHandler with required loaders and config.

        Args:
            encoder_loader (experiments.EncoderLoader): Loader for getting column encoders.
            transform_loader (experiments.TransformLoader): Loader for getting data transformations.
            split_loader (experiments.SplitLoader): Loader for getting dataset split configurations.
            config_path (str): Path to the dataset configuration file.
            csv_path (str): Path to the CSV data file.
        """
        self.encoder_manager = EncodeManager(encoder_loader)
        self.transform_manager = TransformManager(transform_loader)
        self.split_manager = SplitManager(split_loader)
        self.dataset_manager = DatasetManager(config_path)
        self.data = self.load_csv(csv_path)
        self.columns = self.read_csv_header(csv_path)

    def read_csv_header(self, csv_path: str) -> list:
        """Get the column names from the header of the CSV file.
        
        Args:
            csv_path (str): Path to the CSV file to read headers from.

        Returns:
            list: List of column names from the CSV header.
        """
        with open(csv_path) as f:
            header = f.readline().strip().split(",")
        return header
    
    def load_csv(self, csv_path: str) -> pl.DataFrame:
        """Load the CSV file into a polars DataFrame.
        
        Args:
            csv_path (str): Path to the CSV file to load.

        Returns:
            pl.DataFrame: Polars DataFrame containing the loaded CSV data.
        """
        return pl.read_csv(csv_path)
    
    def select_columns(self, columns: list) -> dict:
        """Select specific columns from the DataFrame and return as a dictionary.
        
        Args:
            columns (list): List of column names to select.
            
        Returns:
            dict: A dictionary where keys are column names and values are lists containing the column data.
            
        Example:
            >>> handler = DatasetHandler(...)
            >>> data_dict = handler.select_columns(["col1", "col2"])
            >>> # Returns {'col1': [1, 2, 3], 'col2': [4, 5, 6]}
        """
        df = self.data.select(columns)
        return {col: df[col].to_list() for col in columns}
    
    def add_split(self, force=False) -> None:
        """Add a column specifying the train, validation, test splits of the data.
        An error exception is raised if the split column is already present in the csv file. This behaviour can be overriden by setting force=True.

        Args:
            config (dict) : the dictionary containing  the following keys:
                            "name" (str)        : the split_function name, as defined in the splitters class and experiment.
                            "parameters" (dict) : the split_function specific optional parameters, passed here as a dict with keys named as in the split function definition.
            force (bool) : If True, the split column will be added even if it is already present in the csv file.
        """
        if ("split" in self.columns) and (not force):
            raise ValueError(
                "The category split is already present in the csv file. If you want to still use this function, set force=True",
            )
        # get relevant split columns from the dataset_manager
        split_columns = self.dataset_manager.get_split_columns()
        
        # if split_columns is none, build an empty dictionary
        if split_columns is None:
            split_input_data = {}
        else:
            split_input_data = self.select_columns(split_columns)

        # get the split indices
        train, validation, test = self.split_manager.get_split_indices(split_input_data)

        # add the split column to the data
        split_column = np.full(len(self.data), -1).astype(int)
        split_column[train] = 0
        split_column[validation] = 1
        split_column[test] = 2
        self.data = self.data.with_columns(pl.Series("split", split_column))

        if "split" not in self.columns:
            self.columns.append("split")

    def get_all_items(self) -> tuple[dict, dict, dict]:
        """Get the full dataset as three separate dictionaries for inputs, labels and metadata.

        Returns:
            tuple[dict, dict, dict]: Three dictionaries containing:
                - Input dictionary mapping input column names to encoded input data
                - Label dictionary mapping label column names to encoded label data
                - Meta dictionary mapping meta column names to meta data

        Example:
            >>> handler = DatasetHandler(...)
            >>> input_dict, label_dict, meta_dict = handler.get_dataset()
            >>> print(input_dict.keys())
            dict_keys(['age', 'fare'])
            >>> print(label_dict.keys()) 
            dict_keys(['survived'])
            >>> print(meta_dict.keys())
            dict_keys(['passenger_id'])
        """
        # Get columns for each category from dataset manager
        input_cols = self.dataset_manager.column_categories["input"]
        label_cols = self.dataset_manager.column_categories["label"] 
        meta_cols = self.dataset_manager.column_categories["meta"]

        # Select and organize data by category
        input_data = self.select_columns(input_cols) if input_cols else {}
        label_data = self.select_columns(label_cols) if label_cols else {}
        meta_data = self.select_columns(meta_cols) if meta_cols else {}

        # Encode input and label data
        encoded_input = self.encoder_manager.encode_columns(input_data) if input_data else {}
        encoded_label = self.encoder_manager.encode_columns(label_data) if label_data else {}

        return encoded_input, encoded_label, meta_data

class CsvHandler:
    """Meta class for handling CSV files."""

    def __init__(self, experiment: Any, csv_path: str) -> None:
        self.experiment = experiment
        self.csv_path = csv_path

class CsvProcessing(CsvHandler):
    """Class to load the input csv data and add noise accordingly."""

    def __init__(self, experiment: Any, csv_path: str) -> None:
        super().__init__(experiment, csv_path)
        self.data = self.load_csv()

    def transform(self, transformations: list) -> None:
        """Transforms the data using the specified configuration."""
        for dictionary in transformations:
            key = dictionary["column_name"]
            data_type = key.split(":")[2]
            data_transformer = dictionary["name"]
            transformer = self.experiment.get_data_transformer(data_type, data_transformer)

            # transform the data
            new_data = transformer.transform_all(list(self.data[key]), **dictionary["params"])

            # if the transformation creates new rows (eg. data augmentation), then add the new rows to the original data
            # otherwise just get the transformation of the data
            if transformer.add_row:
                new_rows = self.data.with_columns(pl.Series(key, new_data))
                self.data = self.data.vstack(new_rows)
            else:
                self.data = self.data.with_columns(pl.Series(key, new_data))

    def shuffle_labels(self, seed: float = None) -> None:
        """Shuffles the labels in the data."""
        # set the np seed
        np.random.seed(seed)

        label_keys = self.get_keys_based_on_name_category_dtype(category="label")
        for key in label_keys:
            self.data = self.data.with_columns(pl.Series(key, np.random.permutation(list(self.data[key]))))

    def save(self, path: str) -> None:
        """Saves the data to a csv file."""
        self.data.write_csv(path)


class CsvLoader(CsvHandler):
    """Class for loading the csv data, and then encode the information.

    It will parse the CSV file into four dictionaries, one for each category [input, label, meta].
    So each dictionary will have the keys in the form name:type, and the values will be the column values.
    Afterwards, one can get one or many items from the data, encoded.
    """

    def __init__(self, experiment: Any, csv_path: str, split: Union[int, None] = None) -> None:
        """Initialize the class by parsing and splitting the csv data into the corresponding categories.

        Args:
            experiment (class) : The experiment class to perform
            csv_path (str) : The path to the csv file
            split (int) : The split to load, 0 is train, 1 is validation, 2 is test.
        """
        super().__init__(experiment, csv_path)

        # we need a different parsing function in case we have the split argument or not
        # NOTE using partial we can define the default split value, without the need to pass it as an argument all the time through the class
        if split is not None:
            prefered_load_method = partial(self.load_csv_per_split, split=split)
        else:
            prefered_load_method = self.load_csv

        # parse csv and split into categories
        self.input, self.label, self.meta = self.parse_csv_to_input_label_meta(prefered_load_method)

    def load_csv_per_split(self, split: int) -> pl.DataFrame:
        """Load the part of csv file that has the specified split value.
        Split is a number that for 0 is train, 1 is validation, 2 is test.
        This is accessed through the column with category `split`. Example column name could be `split:split:int`.

        NOTE that the aim of having this function is that depending on the training, validation and test scenarios,
        we are gonna load only the relevant data for it.
        """
        if "split" not in self.categories:
            raise ValueError("The category split is not present in the csv file")
        if split not in [0, 1, 2]:
            raise ValueError(f"The split value should be 0, 1 or 2. The specified split value is {split}")
        colname = self.get_keys_based_on_name_category_dtype("split")
        if len(colname) > 1:
            raise ValueError(
                f"The split category should have only one column, the specified csv file has {len(colname)} columns",
            )
        colname = colname[0]
        return pl.scan_csv(self.csv_path).filter(pl.col(colname) == split).collect()

    def parse_csv_to_input_label_meta(self, load_method: Any) -> Tuple[dict, dict, dict]:
        """This function reads the csv file into a dictionary,
        and then parses each key with the form name:category:type
        into three dictionaries, one for each category [input, label, meta].
        The keys of each new dictionary are in this form name:type.
        """
        # read csv file into a dictionary of lists
        # the keys of the dictionary are the column names and the values are the column values
        data = load_method().to_dict(as_series=False)

        # parse the dictionary into three dictionaries, one for each category [input, label, meta]
        input_data, label_data, split_data, meta_data = {}, {}, {}, {}
        for key in data:
            name, category, data_type = key.split(":")
            if category.lower() == "input":
                input_data[f"{name}:{data_type}"] = data[key]
            elif category.lower() == "label":
                label_data[f"{name}:{data_type}"] = data[key]
            elif category.lower() == "meta":
                meta_data[f"{name}"] = data[key]
        return input_data, label_data, meta_data

    def get_and_encode(self, dictionary: dict, idx: Any = None) -> dict:
        """It gets the data at a given index, and encodes it according to the data_type.

        `dictionary`:
            The keys of the dictionaries are always in the form `name:type`.
            `type` should always match the name of the initialized data_types in the Experiment class. So if there is a `dna` data_type in the Experiment class, then the input key should be `name:dna`
        `idx`:
            The index of the data to be returned, it can be a single index, a list of indexes or a slice
            If None, then it encodes for all the data, not only the given index or indexes.

        The return value is a dictionary containing numpy array of the encoded data at the given index.
        """
        output = {}
        for key in dictionary:  # processing each column
            # get the name and data_type
            name = key.split(":")[0]
            data_type = key.split(":")[1]

            # get the data at the given index
            # if the data is not a list, it is converted to a list
            # otherwise it breaks Float().encode_all(data) because it expects a list
            data = dictionary[key] if idx is None else dictionary[key][idx]

            if not isinstance(data, list):
                data = [data]

            # check if 'data_type' is in the experiment class attributes
            if not hasattr(self.experiment, data_type.lower()):
                raise ValueError(
                    "The data type",
                    data_type,
                    "is not in the experiment class attributes. the column name is",
                    key,
                    "the available attributes are",
                    self.experiment.__dict__,
                )

            # encode the data at given index
            # For that, it first retrieves the data object and then calls the encode_all method to encode the data
            output[name] = self.experiment.get_function_encode_all(data_type)(data)

        return output

    def get_all_items(self) -> Tuple[dict, dict, dict]:
        """Returns all the items in the csv file, encoded.
        TODO in the future we can optimize this for big datasets (ie. using batches, etc).
        """
        return self.get_and_encode(self.input), self.get_and_encode(self.label), self.meta

    def get_all_items_and_length(self) -> Tuple[dict, dict, dict, int]:
        """Returns all the items in the csv file, encoded, and the length of the data."""
        return self.get_and_encode(self.input), self.get_and_encode(self.label), self.meta, len(self)

    def __len__(self) -> int:
        """Returns the length of the first list in input, assumes that all are the same length"""
        return len(list(self.input.values())[0])

    def __getitem__(self, idx: Any) -> dict:
        """It gets the data at a given index, and encodes the input and label, leaving meta as it is.

        `idx`:
            The index of the data to be returned, it can be a single index, a list of indexes or a slice
        """
        # encode input and labels for given index
        x = self.get_and_encode(self.input, idx)
        y = self.get_and_encode(self.label, idx)

        # get the meta data at the given index for each key
        meta = {}
        for key in self.meta:
            data = self.meta[key][idx]
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            meta[key] = data

        return x, y, meta
