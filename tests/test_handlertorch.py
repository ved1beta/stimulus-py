import os
from typing import Any, Dict

import pytest
import torch

from src.stimulus.data.experiments import DnaToFloatExperiment, ProtDnaToFloatExperiment
from src.stimulus.data.handlertorch import TorchDataset


class TorchTestData:
    """Test data container for TorchDataset tests.

    This class handles the initialization and storage of test data for various
    experiments using TorchDataset.

    Args:
        filename: Path to the CSV file containing test data.
        experiment: Experiment class to be instantiated.

    Attributes:
        experiment: Instantiated experiment object.
        csv_path: Absolute path to the test CSV file.
        torch_dataset: Instantiated TorchDataset object.
        expected_len: Expected length of the dataset.
        expected_input_shape: Expected shape of input tensors.
        expected_label_shape: Expected shape of label tensors.
        expected_item_shape: Expected shape of individual items.
    """

    def __init__(self, filename: str, experiment: Any):
        self.experiment = experiment()
        self.csv_path = os.path.abspath(filename)
        self.torch_dataset = TorchDataset(self.csv_path, self.experiment)
        self.expected_len = None
        self.expected_input_shape = None
        self.expected_label_shape = None
        self.expected_item_shape = None


@pytest.fixture
def dna_test_data():
    """Fixture providing test data for DNA experiment.

    Returns:
        TorchTestData: Configured test data for DNA experiments.
    """
    data = TorchTestData("tests/test_data/dna_experiment/test.csv", DnaToFloatExperiment)
    data.expected_len = 2
    data.expected_input_shape = {"hello": [2, 16, 4]}
    data.expected_label_shape = {"hola": [2]}
    data.expected_item_shape = {"hello": [16, 4]}
    return data


@pytest.fixture
def dna_test_data_with_float():
    """Fixture providing test data for DNA experiment with float values.

    Returns:
        TorchTestData: Configured test data for DNA experiments with float values.
    """
    data = TorchTestData("tests/test_data/dna_experiment/test_unequal_dna_float.csv", DnaToFloatExperiment)
    data.expected_len = 4
    data.expected_input_shape = {"hello": [4, 31, 4]}
    data.expected_label_shape = {"hola": [4]}
    data.expected_item_shape = {"hello": [31, 4]}
    return data


@pytest.fixture
def prot_dna_test_data():
    """Fixture providing test data for Protein-DNA experiment.

    Returns:
        TorchTestData: Configured test data for Protein-DNA experiments.
    """
    data = TorchTestData("tests/test_data/prot_dna_experiment/test.csv", ProtDnaToFloatExperiment)
    data.expected_len = 2
    data.expected_input_shape = {"hello": [2, 16, 4], "bonjour": [2, 15, 20]}
    data.expected_label_shape = {"hola": [2]}
    data.expected_item_shape = {"hello": [16, 4], "bonjour": [15, 20]}
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


@pytest.mark.parametrize(
    "fixture_name",
    [
        ("dna_test_data"),
        ("dna_test_data_with_float"),
        ("prot_dna_test_data"),
    ],
)
class TestDictOfTensors:
    """Test that torch dataset properly parses dictionaries of tensors."""

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

    def test_input_tensor_shape(self, request, fixture_name: str):
        """Test if input tensor shapes match expected shapes.

        Args:
            request: Pytest fixture request.
            fixture_name: Name of the fixture to test.
        """
        data = request.getfixturevalue(fixture_name)
        for key, tensor in data.torch_dataset.input.items():
            assert isinstance(tensor, torch.Tensor)
            assert tensor.shape == torch.Size(data.expected_input_shape[key])

    def test_label_tensor_shape(self, request, fixture_name: str):
        """Test if label tensor shapes match expected shapes.

        Args:
            request: Pytest fixture request.
            fixture_name: Name of the fixture to test.
        """
        data = request.getfixturevalue(fixture_name)
        for key, tensor in data.torch_dataset.label.items():
            assert isinstance(tensor, torch.Tensor)
            assert tensor.shape == torch.Size(data.expected_label_shape[key])


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
    """Test suite for dataset item retrieval."""

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
        for key, expected_item_shape in data.expected_item_shape.items():
            if isinstance(idx, slice):
                slice_len = idx.stop - idx.start
                expected_item_shape = [slice_len] + expected_item_shape
            assert dict_items[key].shape == torch.Size(expected_item_shape)


# TODO add test for titanic dataset
