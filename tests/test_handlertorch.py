import os
from typing import Any, Dict

import polars as pl
import pytest
import torch
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

    def __init__(self, filename: str, experiment: Any):
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
        self.hardcoded_expected_len = None
        self.hardcoded_expected_input_shape = None
        self.hardcoded_expected_label_shape = None

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


@pytest.fixture
def dna_test_data():
    """Fixture providing test data for DNA experiment.

    Returns:
        TorchTestData: Configured test data for DNA experiments.
    """
    data = TorchTestData("tests/test_data/dna_experiment/test.csv", DnaToFloatExperiment)
    data.hardcoded_expected_len = 2
    data.hardcoded_expected_input_shape = {"hello": [2, 16, 4]}
    data.hardcoded_expected_label_shape = {"hola": [2]}
    return data


@pytest.fixture
def dna_test_data_with_float():
    """Fixture providing test data for DNA experiment with float values.

    Returns:
        TorchTestData: Configured test data for DNA experiments with float values.
    """
    data = TorchTestData("tests/test_data/dna_experiment/test_unequal_dna_float.csv", DnaToFloatExperiment)
    data.hardcoded_expected_len = 4
    data.hardcoded_expected_input_shape = {"hello": [4, 31, 4]}
    data.hardcoded_expected_label_shape = {"hola": [4]}
    return data


@pytest.fixture
def prot_dna_test_data():
    """Fixture providing test data for Protein-DNA experiment.

    Returns:
        TorchTestData: Configured test data for Protein-DNA experiments.
    """
    data = TorchTestData("tests/test_data/prot_dna_experiment/test.csv", ProtDnaToFloatExperiment)
    data.hardcoded_expected_len = 2
    data.hardcoded_expected_input_shape = {"hello": [2, 16, 4], "bonjour": [2, 15, 20]}
    data.hardcoded_expected_label_shape = {"hola": [2]}
    return data


@pytest.mark.parametrize(
    "fixture_name",
    [
        ("dna_test_data"),
        ("dna_test_data_with_float"),
        ("prot_dna_test_data"),
    ],
)
def test_data_length(request, fixture_name: str):
    """Test if dataset length matches expected length.

    Args:
        request: Pytest fixture request.
        fixture_name: Name of the fixture to test.
    """
    data = request.getfixturevalue(fixture_name)
    assert len(data.torch_dataset) == data.expected_len
    assert len(data.torch_dataset) == data.hardcoded_expected_len


@pytest.mark.parametrize(
    "fixture_name",
    [
        ("dna_test_data"),
        ("dna_test_data_with_float"),
        ("prot_dna_test_data"),
    ],
)
class TestDictOfTensors:
    """Test suite for verifying proper parsing of PyTorch datasets into dictionaries of tensors.

    This test class ensures that:
    1. Input and label data are properly structured as dictionaries
    2. The dictionaries contain PyTorch tensors
    3. Tensor shapes match expected dimensions
    4. Tensor contents match expected values
    """

    def test_input_is_dict(self, request, fixture_name: str):
        """Test if input is a dictionary.

        Args:
            request: Pytest fixture request.
            fixture_name: Name of the fixture to test.
        """
        data = request.getfixturevalue(fixture_name)
        assert isinstance(data.torch_dataset.input, Dict)

    def test_label_is_dict(self, request, fixture_name: str):
        """Test if label is a dictionary.

        Args:
            request: Pytest fixture request.
            fixture_name: Name of the fixture to test.
        """
        data = request.getfixturevalue(fixture_name)
        assert isinstance(data.torch_dataset.label, Dict)

    def test_input_is_dict_of_tensors(self, request, fixture_name: str):
        """Test if input values in torch dataset are PyTorch tensors.

        This test verifies that all values in the torch dataset's input dictionary
        are instances of torch.Tensor.

        Args:
            request: Pytest fixture request object
            fixture_name (str): Name of the fixture to load test data from
        """
        data = request.getfixturevalue(fixture_name)
        for tensor in data.torch_dataset.input.values():
            assert isinstance(tensor, torch.Tensor)

    def test_label_is_dict_of_tensors(self, request, fixture_name: str):
        """Test if dataset labels are PyTorch tensors.

        This test verifies that all values in the label dictionary of a torch dataset
        are PyTorch tensor objects.

        Args:
            request: Pytest fixture request object
            fixture_name (str): Name of the fixture containing test data
        """
        data = request.getfixturevalue(fixture_name)
        for tensor in data.torch_dataset.label.values():
            assert isinstance(tensor, torch.Tensor)

    def test_input_tensor_shapes(self, request, fixture_name: str):
        """Test if input tensor shapes match expected shapes.

        This test verifies that the shapes of input tensors in the torch dataset match both:
        1. The shapes of expected input tensors
        2. The hardcoded expected input shapes (if provided)

        Args:
            request: Pytest fixture request object
            fixture_name (str): Name of the fixture containing test data

        The fixture data should contain:
            - torch_dataset.input: Dict of input tensors
            - expected_input: Dict of expected tensor shapes
            - hardcoded_expected_input_shape: Dict of hardcoded expected shapes (optional)
        """
        data = request.getfixturevalue(fixture_name)
        for key, tensor in data.torch_dataset.input.items():
            assert tensor.shape == data.expected_input[key].shape
            if data.hardcoded_expected_input_shape is not None:
                assert tensor.shape == torch.Size(data.hardcoded_expected_input_shape[key])

    def test_label_tensor_shapes(self, request, fixture_name: str):
        """Test if label tensor shapes match expected shapes.

        This test verifies that the shapes of label tensors in the torch dataset match both:
        1. The shapes of expected label tensors
        2. The hardcoded expected label shapes (if provided)

        Args:
            request: Pytest fixture request object
            fixture_name (str): Name of the fixture containing test data

        The fixture data should contain:
            - torch_dataset.label: Dict of label tensors
            - expected_label: Dict of expected tensor shapes
            - hardcoded_expected_label_shape: Dict of hardcoded expected shapes (optional)
        """
        data = request.getfixturevalue(fixture_name)
        for key, tensor in data.torch_dataset.label.items():
            assert tensor.shape == data.expected_label[key].shape
            if data.hardcoded_expected_label_shape is not None:
                assert tensor.shape == torch.Size(data.hardcoded_expected_label_shape[key])

    def test_input_tensor_content(self, request, fixture_name: str):
        """Tests if input tensors in a PyTorch dataset match their expected values.

        Args:
            request: Pytest fixture request object for accessing test fixtures
            fixture_name (str): Name of the fixture containing test data

        The test verifies that for each key-tensor pair in the torch_dataset.input,
        the tensor exactly matches the corresponding tensor in expected_input.
        """
        data = request.getfixturevalue(fixture_name)
        for key, tensor in data.torch_dataset.input.items():
            assert torch.equal(tensor, data.expected_input[key])

    def test_label_tensor_content(self, request, fixture_name: str):
        """Tests if label tensors in a PyTorch dataset match their expected values.

        Args:
            request: Pytest fixture request object for accessing test fixtures
            fixture_name (str): Name of the fixture containing test data

        The test verifies that for each key-tensor pair in the torch_dataset.label,
        the tensor exactly matches the corresponding tensor in expected_label.
        """
        data = request.getfixturevalue(fixture_name)
        for key, tensor in data.torch_dataset.label.items():
            assert torch.equal(tensor, data.expected_label[key])


@pytest.mark.parametrize(
    "fixture_name,idx",
    [
        ("dna_test_data", 0),
        ("dna_test_data", slice(0, 2)),
        ("dna_test_data_with_float", 0),
        ("dna_test_data_with_float", slice(0, 2)),
        ("prot_dna_test_data", 0),
        ("prot_dna_test_data", slice(0, 2)),
    ],
)
class TestGetItem:
    """Test suite for verifying the __getitem__ method of the TorchDataset class.

    This class contains tests to verify the behavior of the __getitem__ method,
    ensuring that it returns properly structured dictionaries with correctly shaped tensors.
    """

    def test_is_dict(self, request, fixture_name: str, idx: Any):
        """Test if retrieved items are dictionaries.

        Args:
            request: Pytest fixture request.
            fixture_name: Name of the fixture to test.
            idx: Index or slice to retrieve items.
        """
        data = request.getfixturevalue(fixture_name)
        x, y, meta = data.torch_dataset[idx]
        assert isinstance(x, dict)
        assert isinstance(y, dict)
        assert isinstance(meta, dict)

    def test_get_correct_items(self, request, fixture_name: str, idx: Any):
        """Test if retrieved items have correct shapes.

        Args:
            request: Pytest fixture request.
            fixture_name: Name of the fixture to test.
            idx: Index or slice to retrieve items.
        """
        data = request.getfixturevalue(fixture_name)
        x, y, meta = data.torch_dataset[idx]
        dict_items = {**x, **y, **meta}
        for key, expected_shape in data.expected_input_shape.items():
            expected_shape = list(expected_shape)[1:]  # the first dimension is the batch size
            if isinstance(idx, slice):
                slice_len = idx.stop - idx.start
                expected_shape = [slice_len] + expected_shape
            assert dict_items[key].shape == torch.Size(expected_shape)


# TODO add test for titanic dataset
