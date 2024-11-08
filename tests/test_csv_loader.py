import json
import os
from typing import Any

import numpy as np
import numpy.testing as npt
import pytest

from src.stimulus.data.csv import CsvLoader
from src.stimulus.data.experiments import DnaToFloatExperiment, ProtDnaToFloatExperiment


class DataCsvLoader:
    """It stores the CsvLoader objects initialized on a given csv data and the expected values.

    One can use this class to create the data fixtures.
    """

    def __init__(self, filename: str, experiment: Any):
        self.experiment = experiment()
        self.csv_path = os.path.abspath(filename)
        self.csv_loader = CsvLoader(self.experiment, self.csv_path)
        self.data_length = None


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

@pytest.mark.parametrize("fixture_name", [
        ("dna_test_data"),
        ("prot_dna_test_data")
    ])
def test_data_length(request, fixture_name):
    """Verify data is loaded with correct length"""
    data = request.getfixturevalue(fixture_name)
    assert len(data.csv_loader) == data.data_length

@pytest.mark.parametrize("fixture_name", [
        ("dna_test_data"),
        ("prot_dna_test_data")
    ])
def test_parse_csv_to_input_label_meta(request, fixture_name):
    """Test parsing of CSV to input, label, and meta."""
    data = request.getfixturevalue(fixture_name)
    assert isinstance(data.csv_loader.input, dict)
    assert isinstance(data.csv_loader.label, dict)
    assert isinstance(data.csv_loader.meta, dict)

@pytest.mark.parametrize("fixture_name", [
        ("dna_test_data"),
        ("prot_dna_test_data")
    ])
def test_get_all_items(request, fixture_name):
    """Test getting all items."""
    data = request.getfixturevalue(fixture_name)
    input_data, label_data, meta_data = data.get_all_items()
    assert isinstance(input_data, dict)
    assert isinstance(label_data, dict)
    assert isinstance(meta_data, dict)

@pytest.mark.parametrize("fixture_name,slice,expected_length", [
        ("dna_test_data", 0, 1),
        ("dna_test_data", slice(0,2), 2),
        ("prot_dna_test_data", 0, 1),
        ("prot_dna_test_data", slice(0,2), 2)
    ])
def test_get_encoded_item(request, fixture_name, slice, expected_length):
    """Check that one can get the items properly through slicing."""
    # get encoded item
    data = request.getfixturevalue(fixture_name)
    encoded_items = data.csv_loader[slice]

    # it should have 3 dictionaries for input, label, and meta
    assert len(encoded_items) == 3
    for i in range(3):
        assert isinstance(encoded_items[i], dict)

        # for each dictionary, check that the sliced items are encoded as numpy arrays, and that match the expected length
        for item in encoded_items[i].values():
            assert isinstance(item, np.ndarray)
            if (expected_length > 1): # If the expected length is 0, this will fail as we are trying to find the length of an object size 0.
                assert len(item) == expected_length

@pytest.mark.parametrize("fixture_name", [
        ("dna_test_data_with_split"),
        ("prot_dna_test_data_with_split")
    ])
def test_load_with_split(request, fixture_name):
    """Test loading with split."""
    data = request.getfixturevalue(fixture_name)
    assert len(data.csv_loader) == data.data_length

    for i in [0, 1, 2]:
        data_i = CsvLoader(data.experiment, data.csv_path, split=i)
        assert len(data_i) == data.shape_splits[i]

    # with self.assertRaises(ValueError):
    #     CsvLoader(self.experiment, self.csv_path_split, split=3)
