import os
from typing import Any

import numpy as np
import pytest

from src.stimulus.data.csv import CsvLoader
from src.stimulus.data.experiments import DnaToFloatExperiment, ProtDnaToFloatExperiment


class DataCsvLoader:
    """Helper class to store CsvLoader objects and expected values for testing.

    This class initializes CsvLoader objects with given csv data and stores expected
    values for testing purposes.

    Attributes:
        experiment: An experiment instance to process the data.
        csv_path (str): Absolute path to the CSV file.
        csv_loader (CsvLoader): Initialized CsvLoader object.
        data_length (int, optional): Expected length of the data.
        shape_splits (dict, optional): Expected split indices and their lengths.
    """

    def __init__(self, filename: str, experiment: Any):
        """Initialize DataCsvLoader with a CSV file and experiment type.

        Args:
            filename (str): Path to the CSV file.
            experiment (Any): Experiment class to be instantiated.
        """
        self.experiment = experiment()
        self.csv_path = os.path.abspath(filename)
        self.csv_loader = CsvLoader(self.experiment, self.csv_path)
        self.data_length = None
        self.shape_splits = None


@pytest.fixture
def dna_test_data():
    """This stores the basic dna test csv"""
    data = DataCsvLoader("tests/test_data/dna_experiment/test.csv", DnaToFloatExperiment)
    data.data_length = 2
    return data


@pytest.fixture
def dna_test_data_with_split():
    """This stores the basic dna test csv with split"""
    data = DataCsvLoader("tests/test_data/dna_experiment/test_with_split.csv", DnaToFloatExperiment)
    data.data_length = 48
    data.shape_splits = {0: 16, 1: 16, 2: 16}
    return data


@pytest.fixture
def prot_dna_test_data():
    """This stores the basic prot-dna test csv"""
    data = DataCsvLoader("tests/test_data/prot_dna_experiment/test.csv", ProtDnaToFloatExperiment)
    data.data_length = 2
    return data


@pytest.fixture
def prot_dna_test_data_with_split():
    """This stores the basic prot-dna test csv with split"""
    data = DataCsvLoader("tests/test_data/prot_dna_experiment/test_with_split.csv", ProtDnaToFloatExperiment)
    data.data_length = 3
    data.shape_splits = {0: 1, 1: 1, 2: 1}
    return data


@pytest.mark.parametrize(
    "fixture_name",
    [
        ("dna_test_data"),
        ("dna_test_data_with_split"),
        ("prot_dna_test_data"),
        ("prot_dna_test_data_with_split"),
    ],
)
def test_data_length(request, fixture_name: str):
    """Verify data is loaded with correct length.

    Args:
        request: Pytest fixture request object.
        fixture_name (str): Name of the fixture to test.
    """
    data = request.getfixturevalue(fixture_name)
    assert len(data.csv_loader) == data.data_length


@pytest.mark.parametrize(
    "fixture_name",
    [
        ("dna_test_data"),
        ("prot_dna_test_data"),
    ],
)
def test_parse_csv_to_input_label_meta(request, fixture_name: str):
    """Test parsing of CSV to input, label, and meta.

    Args:
        request: Pytest fixture request object.
        fixture_name (str): Name of the fixture to test.

    Verifies:
        - Input data is a dictionary
        - Label data is a dictionary
        - Meta data is a dictionary
    """
    data = request.getfixturevalue(fixture_name)
    assert isinstance(data.csv_loader.input, dict)
    assert isinstance(data.csv_loader.label, dict)
    assert isinstance(data.csv_loader.meta, dict)


@pytest.mark.parametrize(
    "fixture_name",
    [
        ("dna_test_data"),
        ("prot_dna_test_data"),
    ],
)
def test_get_all_items(request, fixture_name: str):
    """Test retrieval of all items from the CSV loader.

    Args:
        request: Pytest fixture request object.
        fixture_name (str): Name of the fixture to test.

    Verifies:
        - All returned data (input, label, meta) are dictionaries
    """
    data = request.getfixturevalue(fixture_name)
    input_data, label_data, meta_data = data.csv_loader.get_all_items()
    assert isinstance(input_data, dict)
    assert isinstance(label_data, dict)
    assert isinstance(meta_data, dict)


@pytest.mark.parametrize(
    "fixture_name,slice,expected_length",
    [
        ("dna_test_data", 0, 1),
        ("dna_test_data", slice(0, 2), 2),
        ("prot_dna_test_data", 0, 1),
        ("prot_dna_test_data", slice(0, 2), 2),
    ],
)
def test_get_encoded_item(request, fixture_name: str, slice: Any, expected_length: int):
    """Test retrieval of encoded items through slicing.

    Args:
        request: Pytest fixture request object.
        fixture_name (str): Name of the fixture to test.
        slice (int or slice): Index or slice object for data access.
        expected_length (int): Expected length of the retrieved data.

    Verifies:
        - Returns 3 dictionaries (input, label, meta)
        - All items are encoded as numpy arrays
        - Arrays have the expected length
    """
    data = request.getfixturevalue(fixture_name)
    encoded_items = data.csv_loader[slice]

    assert len(encoded_items) == 3
    for i in range(3):
        assert isinstance(encoded_items[i], dict)
        for item in encoded_items[i].values():
            assert isinstance(item, np.ndarray)
            if expected_length > 1:
                assert len(item) == expected_length


@pytest.mark.parametrize(
    "fixture_name",
    [
        ("dna_test_data_with_split"),
        ("prot_dna_test_data_with_split"),
    ],
)
def test_splitting(request, fixture_name):
    """Test data splitting functionality.

    Args:
        request: Pytest fixture request object.
        fixture_name (str): Name of the fixture to test.

    Verifies:
        - Data can be loaded with different split indices
        - Splits have correct lengths
        - Invalid split index raises ValueError
    """
    data = request.getfixturevalue(fixture_name)
    for i in [0, 1, 2]:
        data_i = CsvLoader(data.experiment, data.csv_path, split=i)
        assert len(data_i) == data.shape_splits[i]
    with pytest.raises(ValueError):
        CsvLoader(data.experiment, data.csv_path, split=3)
