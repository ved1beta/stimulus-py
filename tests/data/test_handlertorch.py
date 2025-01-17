"""Tests for the PyTorch data handling functionality.

This module contains comprehensive test suites for verifying the proper functioning
of the class handlertorch.TorchDataset. The tests cover the dataset structure, content,
and indexing operations.

The test suite is organized into several components:

TorchTestData:
    This class defines the test data and expected values for the tests.
    The expected values are computed by reading the test data from a CSV file,
    encoding and padding the data according to the experiment specifications.
    So, they rely on the correctness of the upstream functions.
    When available, hardcoded expected values are provided for extra verification,
    to ensure the computation of expected values are correct.
    Once verified, the expected values are used for the rest of the tests.

Fixtures:
    test_data: Parametrized fixture providing different dataset configurations
        - DNA sequence data
        - DNA with float values
        - Protein-DNA combined data

Test Organization:
    TestExpectations
        - Validates test data integrity
        - Verifies expected values match hardcoded values, when provided

    TestTorchDataset
        - TestTorchDatasetStructure: Basic dataset properties
        - TestTorchDatasetContent: Data content validation
        - TestTorchDatasetGetItem: Indexing operations

Usage:
    pytest test_handlertorch.py
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
        experiment: The experiment class.

    Attributes:
        experiment: An instance of the experiment class that defines data processing methods.
        csv_path (str): Absolute path to the CSV file containing the test data.
        torch_dataset (TorchDataset): The PyTorch dataset created from the CSV file.
        expected_input (dict): Dictionary containing encoded and padded input data.
        expected_label (dict): Dictionary containing encoded label data.
        expected_len (int): Number of rows in the CSV data.
        expected_input_shape (dict): Dictionary containing shapes of input tensors.
        expected_label_shape (dict): Dictionary containing shapes of label tensors.
        hardcoded_expected_values (dict): Dictionary containing hardcoded expected values.
    """

    def __init__(self, filename: str, experiment: Type[Any]) -> None:
        # load test data
        self.experiment = experiment()
        self.csv_path = os.path.abspath(filename)
        self.torch_dataset = TorchDataset(self.csv_path, self.experiment)

        # get expected data
        data = pl.read_csv(self.csv_path)
        self.expected_len = len(data)
        self.expected_input = self.get_encoded_padded_category(data, "input")
        self.expected_label = self.get_encoded_padded_category(data, "label")
        self.expected_input_shape = {k: v.shape for k, v in self.expected_input.items()}
        self.expected_label_shape = {k: v.shape for k, v in self.expected_label.items()}

        # provide options for hardcoded expected values
        # they must be the same as the expected values above, otherwise the tests will fail
        # this is for extra verification
        self.hardcoded_expected_values = {
            "length": None,
            "input_shape": None,
            "label_shape": None,
        }

    def get_encoded_padded_category(self, data: pl.DataFrame, category: str) -> Dict[str, Union[Tensor, pl.Series]]:
        """Retrieves encoded data for a specific category from a CSV file.

        This method processes columns that match the specified category.
        Each column in the data is expected to follow the format 'name:category:datatype'.
        The data from matching columns is encoded using the appropriate encoding function
        based on the datatype. The encoded data is then padded to the same length.

        Args:
            data (pl.DataFrame): The CSV data to process.
            category (str): The category to filter columns by.

        Returns:
            dict: A dictionary where keys are column names (without category and datatype)
                  and values are the encoded data for that column.

        Example:
            If CSV contains a column "stimulus:visual:str", and category="visual",
            the returned dict will have "stimulus" as a key with its encoded values.
        """
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


@pytest.fixture(
    params=[
        (
            "tests/test_data/dna_experiment/test.csv",
            DnaToFloatExperiment,
            {
                "length": 2,
                "input_shape": {"hello": [2, 16, 4]},
                "label_shape": {"hola": [2]},
            },
        ),
        (
            "tests/test_data/dna_experiment/test_unequal_dna_float.csv",
            DnaToFloatExperiment,
            {
                "length": 4,
                "input_shape": {"hello": [4, 31, 4]},
                "label_shape": {"hola": [4]},
            },
        ),
        (
            "tests/test_data/prot_dna_experiment/test.csv",
            ProtDnaToFloatExperiment,
            {
                "length": 2,
                "input_shape": {"hello": [2, 16, 4], "bonjour": [2, 15, 20]},
                "label_shape": {"hola": [2]},
            },
        ),
    ],
)
def test_data(request) -> TorchTestData:
    """Parametrized fixture providing test data for all experiment types.

    This parametrized fixture contain tuples of (filename, experiment_class, expected_values)
    for each test data file. It loads the test data and initializes the TorchTestData object.
    By parametrizing the fixture, we can run the same tests on different datasets, without
    the need for individual fixtures or duplicate the code.

    Args:
        request: Pytest request object containing the test data parameters.

    Returns:
        TorchTestData: A test data object containing the initialized torch dataset
            and the expected values for the dataset.
    """
    filename, experiment_class, expected_values = request.param
    data = TorchTestData(filename, experiment_class)
    data.expected_values = expected_values
    return data


class TestExpectations:
    """Test class for validating expectations in test data.

    This class contains test methods to verify that expected values in test data
    are properly defined and match any provided hardcoded values. It helps ensure
    test data integrity before running the real tests.

    Test methods:
        test_expected_values_are_defined: Verifies essential expected values are defined
        test_expected_values_match_hardcoded: Validates expected values against hardcoded values
    """

    def test_expected_values_are_defined(self, test_data) -> None:
        """Test that expected values are defined.

        Verifies that the essential expected values in the test_data fixture are properly defined
        and not None.

        Args:
            test_data: A fixture containing test data with expected value attributes.

        Raises:
            AssertionError: If any of the expected values (expected_len, expected_input_shape,
                or expected_label_shape) is None.
        """
        assert test_data.expected_len is not None, "Expected length is not defined"
        assert test_data.expected_input_shape is not None, "Expected input shape is not defined"
        assert test_data.expected_label_shape is not None, "Expected label shape is not defined"

    def test_expected_values_match_hardcoded(self, test_data) -> None:
        """Validate the expected values match the hardcoded values, when provided.

        Since we defined the expected values by computing them from the test data with
        alternative ways, we need to ensure they are correct. This function validates
        the expected values match the hardcoded values in the test data, if provided.
        Once verified, the rest of the tests will use the expected values for
        validation.

        Args:
            test_data (TorchTestData): Test data fixture.

        Raises:
            AssertionError: If the expected values do not match the hardcoded values.
        """
        if test_data.hardcoded_expected_values["length"]:
            assert test_data.expected_len == test_data.hardcoded_expected_values["length"], (
                f"Length mismatch: "
                f"got {test_data.expected_len}, "
                f"expected {test_data.hardcoded_expected_values['length']}"
            )

        if test_data.hardcoded_expected_values["input_shape"]:
            for key, shape in test_data.hardcoded_expected_values["input_shape"].items():
                assert test_data.expected_input_shape[key] == torch.Size(shape), (
                    f"Input shape mismatch for {key}: "
                    f"got {test_data.expected_input_shape[key]}, "
                    f"expected {torch.Size(shape)}"
                )

        if test_data.hardcoded_expected_values["label_shape"]:
            for key, shape in test_data.hardcoded_expected_values["label_shape"].items():
                assert test_data.expected_label_shape[key] == torch.Size(shape), (
                    f"Label shape mismatch for {key}: "
                    f"got {test_data.expected_label_shape[key]}, "
                    f"expected {torch.Size(shape)}"
                )


class TestTorchDataset:
    """Test suite for TorchDataset functionality.

    This class contains tests for verifying the behavior and functionality
    of the TorchDataset class implementation. It tests dataset length, data structure,
    and indexing operations.

    Test classes:
        TestTorchDatasetStructure: Tests basic dataset properties
        TestTorchDatasetContent: Tests data content validation
        TestTorchDatasetGetItem: Tests indexing operations
    """

    class TestTorchDatasetStructure:
        """Tests for the PyTorch Dataset Structure.

        This class contains unit tests to verify the proper structure and functionality
        of the TorchDataset class. It checks for the presence of required attributes,
        correct dataset length, and proper data types of the dataset components.

        Test methods:
            test_dataset_has_required_attributes: Validates the presence of 'input' and 'label' attributes
            test_dataset_length: Verifies the dataset length
            test_is_dictionary_of_tensors: Checks if input and label are dictionaries of tensors
        """

        def test_dataset_has_required_attributes(self, test_data) -> None:
            """Test if the TorchDataset has the required input and label attributes.

            This test verifies that the torch_dataset object contained within test_data
            has both 'input' and 'label' attributes, which are essential for proper
            dataset functionality.

            Args:
                test_data: A fixture providing test data containing a torch_dataset object.

            Raises:
                AssertionError: If either 'input' or 'label' attributes are missing from
                               the torch_dataset.
            """
            assert hasattr(test_data.torch_dataset, "input"), "TorchDataset does not have 'input' attribute"
            assert hasattr(test_data.torch_dataset, "label"), "TorchDataset does not have 'label' attribute"

        def test_dataset_length(self, test_data) -> None:
            """Test dataset length.

            Verifies that the length of the torch dataset matches the expected length.

            Args:
                test_data: Fixture containing torch dataset and expected length for validation.

            Raises:
                AssertionError: If the torch dataset length does not match expected_len.
            """
            assert (
                len(test_data.torch_dataset) == test_data.expected_len
            ), f"Dataset length mismatch: got {len(test_data.torch_dataset)}, expected {test_data.expected_len}"

        @pytest.mark.parametrize("category", ["input", "label"])
        def test_is_dictionary_of_tensors(self, test_data, category):
            """Test if a dataset category is a dictionary of PyTorch tensors.

            This test verifies that:
            1. The specified category attribute of the torch_dataset is a dictionary
            2. All values in the dictionary are PyTorch Tensor objects

            Args:
                test_data : Test data fixture containing the torch_dataset to test
                category (str): Name of the category/attribute to test (e.g., 'input', 'label')

            Raises:
                AssertionError:
                    - If the category is not a dictionary
                    - If any value in the dictionary is not a PyTorch Tensor
            """
            data_dict = getattr(test_data.torch_dataset, category)
            assert isinstance(data_dict, dict), f"{category} is not a dictionary: got {type(data_dict)}"
            for key, value in data_dict.items():
                assert isinstance(value, Tensor), f"{category}[{key}] is not a Tensor, got {type(value)}"

    class TestTorchDatasetContent:
        """A test class for verifying the content of PyTorch datasets.

        This class contains tests to verify that PyTorch datasets are properly
        constructed and contain the expected data. It checks three main aspects:
        the presence of correct keys, the shapes of tensors, and the actual
        content of tensors.

        Test methods:
            test_tensor_keys: Verifies that the input and label dictionaries contain
                the expected keys.
            test_tensor_shapes: Ensures that each tensor in the dataset has the
                expected shape.
            test_tensor_content: Validates that the actual content of each tensor
                matches the expected values.
        """

        @pytest.mark.parametrize("category", ["input", "label"])
        def test_tensor_keys(self, test_data, category: str) -> None:
            """Test if the tensor keys in the dataset match expected keys.

            Args:
                test_data: TestData object containing the dataset and expected values
                category (str): String indicating which category to check ('input' or 'label')

            Raises:
                AssertionError: If the keys in data_dict don't match the expected keys
            """
            data_dict = getattr(test_data.torch_dataset, category)
            expected_keys = test_data.expected_input.keys() if category == "input" else test_data.expected_label.keys()
            assert set(data_dict.keys()) == set(
                expected_keys
            ), f"Keys mismatch for {category}: got {set(data_dict.keys())}, expected {set(expected_keys)}"

        @pytest.mark.parametrize("category", ["input", "label"])
        def test_tensor_shapes(self, test_data, category: str) -> None:
            """Test tensor shapes in the input or label data.

            This test function verifies that all tensors in either input or label data
            have the expected shapes as defined in test_data.

            Args:
                test_data: A test data object containing torch_dataset and expected shape information
                category (str): Either "input" or "label" to specify which data category to test

            Raises:
                AssertionError: If any tensor's shape doesn't match the expected shape
            """
            data_dict = getattr(test_data.torch_dataset, category)
            for key, tensor in data_dict.items():
                expected_shape = (
                    test_data.expected_input_shape[key] if category == "input" else test_data.expected_label_shape[key]
                )
                assert (
                    tensor.shape == expected_shape
                ), f"Shape mismatch for {category}[{key}]: got {tensor.shape}, expected {expected_shape}"

        @pytest.mark.parametrize("category", ["input", "label"])
        def test_tensor_content(self, test_data, category: str) -> None:
            """Tests if tensor content matches expected values.

            This test verifies that the tensor content in both input and label dictionaries
            matches their expected values from the test data.

            Args:
                test_data: A test data fixture containing torch_dataset and expected values
                category (str): String indicating which category to test ('input' or 'label')

            Raises:
                AssertionError: If tensor content does not match expected values
            """
            data_dict = getattr(test_data.torch_dataset, category)
            for key, tensor in data_dict.items():
                expected_tensor = (
                    test_data.expected_input[key] if category == "input" else test_data.expected_label[key]
                )
                assert torch.equal(
                    tensor, expected_tensor
                ), f"Content mismatch for {category}[{key}]: got {tensor}, expected {expected_tensor}"

    class TestTorchDatasetGetItem:
        """Test suite for dataset's __getitem__ functionality.

        This class tests the behavior of the __getitem__ method in the torch dataset,
        ensuring proper data retrieval, structure, and error handling.

        Tests include:
            - Verification of returned data structure (dictionaries containing tensors)
            - Validation of dictionary keys against expected keys
            - Confirmation of tensor shapes for both single items and slices
            - Verification of tensor contents against expected values
            - Handling of invalid indices

        The test suite uses parametrization to test both single index (int) and slice
        access patterns, as well as to test both input and label components of the
        dataset items.

        Test Methods:
            test_get_item_returns_expected_data_structure: Verifies basic structure of returned data
            test_get_item_keys_are_correct: Ensures dictionary keys match expected keys
            test_get_item_shapes: Validates tensor shapes in returned data
            test_get_item_content: Verifies actual content of tensors
            test_getitem_invalid_index: Tests error handling for invalid indices
        """

        @pytest.mark.parametrize("idx", [0, slice(0, 2)])
        def test_get_item_returns_expected_data_structure(self, test_data, idx: Union[int, slice]) -> None:
            """Test if __getitem__ returns correct data structure.

            This test ensures that the __getitem__ method of the torch_dataset returns data
            in the expected format, specifically checking that:
            1. The method returns three dictionaries (x, y, meta)
            2. All values in x and y dictionaries are PyTorch Tensors

            Args:
                test_data: The test dataset fixture
                idx (Union[int, slice]): Index or slice to access the dataset

            Raises:
                AssertionError: If any of the returned structures don't match expected types
            """
            x, y, meta = test_data.torch_dataset[idx]

            # Test items are dictionaries
            assert isinstance(x, dict), f"Expected input to be dict, got {type(x)}"
            assert isinstance(y, dict), f"Expected label to be dict, got {type(y)}"
            assert isinstance(meta, dict), f"Expected meta to be dict, got {type(meta)}"

            # Test item contents are tensors
            for key, value in x.items():
                assert isinstance(value, Tensor), f"Input tensor {key} is not a Tensor"
            for key, value in y.items():
                assert isinstance(value, Tensor), f"Label tensor {key} is not a Tensor"

        @pytest.mark.parametrize("idx", [0, slice(0, 2)])
        @pytest.mark.parametrize(
            "category_info",
            [
                ("input", "x", "expected_input"),
                ("label", "y", "expected_label"),
            ],
        )
        def test_get_item_keys_are_correct(self, test_data, idx: Union[int, slice], category_info: tuple) -> None:
            """Test if the keys in retrieved dataset items match expected keys.

            This test verifies that the keys in the retrieved dataset items (either input 'x' or label 'y')
            match the expected keys stored in the dataset attributes.

            Args:
                test_data: The test dataset object containing the torch_dataset
                idx (int): Index of the item to retrieve from the dataset
                category_info (tuple): Contains (category, data_attr, expected_attr) where:
                    - category (str): Either "input" or "label" indicating which part to check
                    - data_attr (str): Attribute name for the data being checked
                    - expected_attr (str): Attribute name containing the expected keys

            Raises:
                AssertionError: If the keys in the retrieved item don't match the expected keys
            """
            category, data_attr, expected_attr = category_info

            # get dataset item
            x, y, _ = test_data.torch_dataset[idx]
            keys = set(x.keys()) if category == "input" else set(y.keys())
            expected_keys = set(getattr(test_data, expected_attr).keys())

            # verify keys
            assert keys == expected_keys, f"Keys mismatch for {category}: got {keys}, expected {expected_keys}"

        @pytest.mark.parametrize("idx", [0, slice(0, 2)])
        @pytest.mark.parametrize(
            "category_info",
            [
                ("input", "x", "expected_input_shape"),
                ("label", "y", "expected_label_shape"),
            ],
        )
        def test_get_item_shapes(self, test_data, idx: Union[int, slice], category_info: tuple) -> None:
            """Test if dataset items have the correct shapes for both input and target tensors..

            This test verifies that tensor shapes match expected shapes for either input or label
            data. For slice indices, it accounts for the batch dimension in the expected shape.
            The test compares each tensor's shape against the expected shape stored in the
            data handler's attributes.

            Args:
                test_data: The test dataset object containing the torch_dataset
                idx (int): Index of the item to retrieve from the dataset
                category_info (tuple): Contains (category, data_attr, expected_attr) where:
                    - category (str): Either "input" or "label" indicating which part to check
                    - data_attr (str): Attribute name for the data being checked
                    - expected_attr (str): Attribute name containing the expected keys

            Raises:
                AssertionError: If any tensor's shape doesn't match the expected
            """
            category, data_attr, expected_attr = category_info

            # get dataset item
            x, y, _ = test_data.torch_dataset[idx]
            data = x if category == "input" else y
            expected_shapes = getattr(test_data, expected_attr)

            # test each tensor has the proper shape
            for key, tensor in data.items():
                # get expected shape
                expected_shape = expected_shapes[key]
                base_shape = list(expected_shape)[1:] if len(expected_shape) > 1 else []  # remove batch dimension
                if isinstance(idx, slice):
                    expected_shape = [idx.stop - idx.start] + base_shape
                else:
                    expected_shape = base_shape
                expected_shape = torch.Size(expected_shape)

                # verify shape
                assert (
                    tensor.shape == expected_shape
                ), f"Wrong shape for {category}[{key}]: got {tensor.shape}, expected {expected_shape}"

        @pytest.mark.parametrize("idx", [0, slice(0, 2)])
        @pytest.mark.parametrize(
            "category_info",
            [
                ("input", "x", "expected_input"),
                ("label", "y", "expected_label"),
            ],
        )
        def test_get_item_content(self, test_data, idx: Union[int, slice], category_info: tuple) -> None:
            """Test if the content of items retrieved from torch_dataset is correct.

            The test verifies that for each key in the data dictionary, the tensor matches
            the corresponding expected tensor from the original data at the given index.

            Args:
                test_data: The test dataset object containing the torch_dataset
                idx (int): Index of the item to retrieve from the dataset
                category_info (tuple): Contains (category, data_attr, expected_attr) where:
                    - category (str): Either "input" or "label" indicating which part to check
                    - data_attr (str): Attribute name for the data being checked
                    - expected_attr (str): Attribute name containing the expected keys

            Raises:
                AssertionError: If any tensor content does not match the expected values
            """
            category, data_attr, expected_attr = category_info

            # get dataset item
            x, y, _ = test_data.torch_dataset[idx]
            data = x if category == "input" else y
            expected_data = getattr(test_data, expected_attr)

            # test each tensor has the proper content
            for key, tensor in data.items():
                expected_tensor = expected_data[key][idx]

                assert torch.equal(
                    tensor, expected_tensor
                ), f"Content mismatch for {category}[{key}]: got {tensor}, expected {expected_tensor}"

        @pytest.mark.parametrize("invalid_idx", [5000])
        def test_getitem_invalid_index(self, test_data, invalid_idx: Union[int, slice]) -> None:
            """Test whether invalid indexing raises appropriate exceptions.

            Tests if accessing test_data.torch_dataset with an invalid index raises
            an IndexError exception.

            Args:
                test_data: Fixture providing test dataset
                invalid_idx (Union[int,slice]): Invalid index value to test with

            Raises:
                AssertionError: If IndexError is not raised when accessing invalid index
            """
            with pytest.raises(IndexError):
                _ = test_data.torch_dataset[invalid_idx]


# TODO add tests for titanic dataset
