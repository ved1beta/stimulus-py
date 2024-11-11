"""Tests for the PyTorch data handling functionality.

This module contains comprehensive test suites for verifying the proper functioning
of PyTorch dataset handling, particularly focusing on DNA and Protein-DNA experiments.
It includes tests for data loading, tensor shape verification, and content validation.

The test suite is organized into several components:
- TorchTestData: A class that handles test data loading and preprocessing
- Fixtures for different types of experimental data
- Test suites for verifying dataset properties and tensor handling

Key test areas include:
- Dataset length verification
- Dictionary and tensor structure validation
- Tensor shape verification
- Tensor content validation
- Dataset indexing and slicing functionality

Test Fixtures:
    dna_test_data: Basic DNA experiment test data
    dna_test_data_with_float: DNA experiment test data with float values
    prot_dna_test_data: Protein-DNA experiment test data

Test Classes:
    TestDictOfTensors: Verifies proper parsing of datasets into tensor dictionaries
    TestGetItem: Validates the dataset indexing functionality

Dependencies:
    - pytest
    - torch
    - polars
    - os
    - typing
"""

import os
from typing import Any, Dict, Type, Union

import polars as pl
import pytest
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from src.stimulus.data.experiments import DnaToFloatExperiment, ProtDnaToFloatExperiment
from src.stimulus.data.handlertorch import TorchDataset


class TorchTestData:
    """It declares the data for the tests, and the expected data content and shapes.

    This class handles the loading and preprocessing of test data for PyTorch-based experiments.
    It also provides the expected data content and shapes, by loading the data in alternative ways:
    it reads data from a CSV file, encodes and pads the input/label data according to the
    experiment specifications.

    Args:
        filename (str): Path to the CSV file containing the test data.
        experiment (Any): The experiment class that defines data processing methods.

    Attributes:
        experiment: An instance of the experiment class that defines data processing methods.
        csv_path (str): Absolute path to the CSV file containing the test data.
        torch_dataset (TorchDataset): The PyTorch dataset created from the CSV file.
        expected_input (dict): Dictionary containing encoded and padded input data.
        expected_label (dict): Dictionary containing encoded label data.
        expected_len (int): Number of rows in the CSV data.
        expected_input_shape (dict): Dictionary containing shapes of input tensors.
        expected_label_shape (dict): Dictionary containing shapes of label tensors.
        hardcoded_expected_len (int): Expected number of rows in the CSV data.
        hardcoded_expected_input_shape (dict): Expected shapes of input tensors.
        hardcoded_expected_label_shape (dict): Expected shapes of label tensors.
    """

    def __init__(self, filename: str, experiment: Type[Any]):
        # load test data
        self.experiment = experiment()
        self.csv_path = os.path.abspath(filename)
        self.torch_dataset = TorchDataset(self.csv_path, self.experiment)

        # get expected data
        self.expected_input = self.get_encoded_padded_category("input")
        self.expected_label = self.get_encoded_padded_category("label")
        self.expected_len = self.get_data_length()
        self.expected_input_shape = {k: v.shape for k, v in self.expected_input.items()}
        self.expected_label_shape = {k: v.shape for k, v in self.expected_label.items()}

        # provide options for hardcoded expected values
        # they must be the same as the expected values above, otherwise the tests will fail
        # this is for extra verification
        self.hardcoded_expected_values = {
            "length": None,
            "input_shape": None,
            "label_shape": None
        }

    def get_encoded_padded_category(self, category: str):
        """Retrieves encoded data for a specific category from a CSV file.

        This method reads a CSV file and processes columns that match the specified category.
        Each column in the CSV is expected to follow the format 'name:category:datatype'.
        The data from matching columns is encoded using the appropriate encoding function
        based on the datatype. The encoded data is then padded to the same length.

        Args:
            category (str): The category to filter columns by.

        Returns:
            dict: A dictionary where keys are column names (without category and datatype)
                  and values are the encoded data for that column.

        Example:
            If CSV contains a column "stimulus:visual:str", and category="visual",
            the returned dict will have "stimulus" as a key with its encoded values.
        """
        # read data
        data = pl.read_csv(self.csv_path)

        # filter columns by category
        columns = {}
        for colname in data.columns:
            current_name = colname.split(":")[0]
            current_category = colname.split(":")[1]
            current_datatype = colname.split(":")[2]
            if current_category == category:
                # get and encode data into list of tensors
                tmp = self.experiment.get_function_encode_all(current_datatype)(data[colname].to_list())

                # pad sequences to the same length
                # NOTE that this is hardcoded to pad with 0
                # so it will only work for tests where padding with 0 is expected
                if category == "input":
                    tmp = [torch.tensor(item) for item in tmp]
                    tmp = pad_sequence(tmp, batch_first=True, padding_value=0)

                # convert list into tensor
                elif category == "label":
                    tmp = torch.tensor(tmp)

                columns[current_name] = tmp
        return columns

    def get_data_length(self):
        """Returns the length of the CSV data.

        This method reads a CSV file from the specified path and returns the number of rows in the data.

        Returns:
            int: The number of rows in the CSV file.

        Raises:
            FileNotFoundError: If the CSV file does not exist at the specified path.
            pl.exceptions.NoDataError: If the CSV file is empty.
        """
        data = pl.read_csv(self.csv_path)
        return len(data)


# Replace individual fixtures with a parametrized fixture
@pytest.fixture(params=[
    ("tests/test_data/dna_experiment/test.csv", DnaToFloatExperiment, {
        "length": 2,
        "input_shape": {"hello": [2, 16, 4]},
        "label_shape": {"hola": [2]}
    }),
    ("tests/test_data/dna_experiment/test_unequal_dna_float.csv", DnaToFloatExperiment, {
        "length": 4,
        "input_shape": {"hello": [4, 31, 4]},
        "label_shape": {"hola": [4]}
    }),
    ("tests/test_data/prot_dna_experiment/test.csv", ProtDnaToFloatExperiment, {
        "length": 2,
        "input_shape": {"hello": [2, 16, 4], "bonjour": [2, 15, 20]},
        "label_shape": {"hola": [2]}
    })
])
def test_data(request) -> TorchTestData:
    """Parametrized fixture providing test data for all experiment types."""
    filename, experiment_class, expected_values = request.param
    data = TorchTestData(filename, experiment_class)
    data.expected_values = expected_values
    return data


def validate_expected_values(test_data) -> None:
    """Validate the expected values are properly defined.
    
    Since we defined the expected values by computing them from the test data with
    alternative ways, we need to ensure they are correct. This function validates
    the expected values match the hardcoded values in the test data, if provided. 
    Once verified, the rest of the tests will use the expected values for 
    validation.
    
    Args:
        test_data (TorchTestData): Test data fixture.
    """
    if test_data.hardcoded_expected_values["length"]:
        assert test_data.expected_len == test_data.hardcoded_expected_values["length"]
    if test_data.hardcoded_expected_values["input_shape"]:
        for key, shape in test_data.hardcoded_expected_values["input_shape"].items():
            assert test_data.expected_input_shape[key] == torch.Size(shape)
    if test_data.hardcoded_expected_values["label_shape"]:
        for key, shape in test_data.hardcoded_expected_values["label_shape"].items():
            assert test_data.expected_label_shape[key] == torch.Size(shape)


class TestTorchDataset:
    """Test suite for TorchDataset functionality.

    This class contains tests for verifying the behavior and functionality 
    of the TorchDataset class implementation. It tests dataset length, data structure,
    and indexing operations.

    Test Cases:
        test_dataset_length: Verifies that the dataset length matches expected value.
        test_data_structure: Tests dataset structure and content for input and label data.
        test_getitem: Tests indexing functionality for both single items and slices.
    """

    def test_dataset_length(self, test_data: TorchTestData) -> None:
        """Test dataset length matches expected value.

        The test verifies that the length of the torch dataset object matches
        the expected length stored in the test data fixture.

        Args:
            test_data (TorchTestData): Fixture containing torch dataset and expected length

        Raises:
            AssertionError: If dataset length does not match expected length
        """
        assert len(test_data.torch_dataset) == test_data.expected_len

    @pytest.mark.parametrize("category", ["input", "label"])
    def test_data_structure(self, test_data: TorchTestData, category: str):
        """Test dataset structure and content.

        This test verifies the structure and content of the dataset by checking that:
        1. The data dictionary exists and is of type Dict
        2. All values in the dictionary are PyTorch Tensors
        3. Each tensor has the expected shape based on category (input or label)
        4. Each tensor matches its expected value

        Args:
            test_data (TorchTestData): Test dataset object containing input/label data and expected values
            category (str): Category to test, either "input" or "label"
        
        Raises:
            AssertionError: If any of the structure or content checks fail
        """
        # is dictionary of tensors
        data_dict = getattr(test_data.torch_dataset, category)
        assert isinstance(data_dict, Dict)
        for tensor in data_dict.values():
            assert isinstance(tensor, Tensor)
            
        for key, tensor in data_dict.items():

            # verify tensor shapes
            expected_shape = (
                test_data.expected_input_shape[key] if category == "input" 
                else test_data.expected_label_shape[key]
            )
            assert tensor.shape == expected_shape

            # verify tensor content
            expected_tensor = (
                test_data.expected_input[key] if category == "input"
                else test_data.expected_label[key]
            )
            assert torch.equal(tensor, expected_tensor)

    @pytest.mark.parametrize("idx", [0, slice(0, 2)])
    def test_getitem(self, test_data: TorchTestData, idx: Union[int, slice]):
        """Test __getitem__ functionality.
        
        This test verifies that the dataset's __getitem__ method works correctly by checking:
        1. The returned items (x, y, meta) are all dictionaries
        2. The shapes of returned tensors match expected shapes
        3. The content of returned tensors matches expected values
        
        Args:
            test_data (TorchTestData): Test dataset object containing the data to verify
            idx (Union[int, slice]): Index or slice to access the dataset

        Raises:
            AssertionError: If returned items are not dictionaries or shapes don't match expected values
                or content does not match expected values.
        """
        # is dictionary
        x, y, meta = test_data.torch_dataset[idx]
        assert all(isinstance(d, dict) for d in [x, y, meta])
        
        dict_items = {**x, **y, **meta}
        for key in test_data.expected_input.keys():

            # verify shapes
            expected_shape = test_data.expected_input_shape[key]
            base_shape = list(expected_shape)[1:]  # remove batch dimension
            expected_size = [idx.stop - idx.start] + base_shape if isinstance(idx, slice) else base_shape
            assert dict_items[key].shape == torch.Size(expected_size)

            # verify content
            expected_tensor = test_data.expected_input[key][idx]
            assert torch.equal(dict_items[key], expected_tensor)
        

# TODO add tests for titanic dataset
