import pytest
import json
import os
from abc import ABC

import numpy as np
import numpy.testing as npt
import polars as pl

from src.stimulus.data.csv import CsvLoader, CsvProcessing
from src.stimulus.data.experiments import DnaToFloatExperiment, ProtDnaToFloatExperiment


# set seed for reproducibility
# SEED = 123
# np.random.seed(SEED)
# pl.set_random_seed(SEED)

class csv_data():
    """It stores the CsvProcessing objects initialized on a given csv data and the expected values.

    One can use this class to create the data fixtures.

    Attributes:
        experiment (Experiment): An instance of the experiment class.
        csv_path (str): The absolute path to the CSV file.
        csv_processing (CsvProcessing): An instance of the CsvProcessing class for handling CSV data.
        data_length (int or None): The length of the data, initialized to None.
        expected_split (List[int] or None): The split values from the CSV, initialized to None.
        expected_values (Any or None): The values from the CSV, initialized to None.

    Args:
        filename (str): The path to the CSV file.
        experiment (type): The class type of the experiment to be instantiated.
    """
    
    def __init__(self, filename, experiment):
        self.experiment = experiment()
        self.csv_path = os.path.abspath(filename)
        self.csv_processing = CsvProcessing(self.experiment, self.csv_path)
        self.data_length = None
        self.expected_split = None
        self.expected_values = None

class TestDnaToFloatCsvProcessing():
    """It validates the functionality of CsvProcessing on DnaToFloatExperiment."""

    @pytest.fixture
    def dna_test_data(self):
        """ This is the csv_data class storing the basic dna test csv """
        data = csv_data("tests/test_data/dna_experiment/test.csv", DnaToFloatExperiment)
        data.data_length = 2
        data.expected_split = [1,0]
        data.expected_values = {
            "pet:meta:str" : ["cat", "dog", "cat", "dog"],
            "hola:label:float" : [12.676405, 12.540016, 12.676405, 12.540016],
            "hello:input:dna" : ["ACTGACTGATCGATNN", "ACTGACTGATCGATNN", "NNATCGATCAGTCAGT", "NNATCGATCAGTCAGT"],
            "split:split:int" : [1, 0, 1, 0]
        }
        return data

    @pytest.fixture
    def dna_test_data_long(self):
        """ This is the csv_data class storing the long dna test csv """
        data = csv_data("tests/test_data/dna_experiment/test_shuffling_long.csv", DnaToFloatExperiment)
        data.data_length = 1000
        return data

    @pytest.fixture
    def dna_test_data_long_shuffled(self):
        """ This is the csv_data class storing the shuffled long dna test csv """
        data = csv_data("tests/test_data/dna_experiment/test_shuffling_long_shuffled.csv", ProtDnaToFloatExperiment)
        data.data_length = 1000
        return data

    @pytest.fixture
    def dna_config(self):
        """ This is the config file for the dna experiment """
        with open("tests/test_data/dna_experiment/test_config.json") as f:
            return json.load(f)
    
    def test_len(self, dna_test_data, dna_test_data_long, dna_test_data_long_shuffled):
        """ this tests the data is properly loaded with the correct length """
        assert len(dna_test_data.csv_processing.data) == dna_test_data.data_length
        assert len(dna_test_data_long.csv_processing.data) == dna_test_data_long.data_length
        assert len(dna_test_data_long_shuffled.csv_processing.data) == dna_test_data_long_shuffled.data_length

    def test_add_split(self, dna_test_data, dna_config):
        """ this tests the add_split function properly adds the split column """
        dna_test_data.csv_processing.add_split(dna_config["split"])
        assert dna_test_data.csv_processing.data["split:split:int"].to_list() == dna_test_data.expected_split
    
    def test_transform_data(self, dna_test_data, dna_config):
        """ this tests the transform_data function properly transforms the data """
        dna_test_data.csv_processing.add_split(dna_config["split"])
        dna_test_data.csv_processing.transform(dna_config["transform"])
        for key, expected_values in dna_test_data.expected_values.items():
            observed_values = list(dna_test_data.csv_processing.data[key])
            observed_values = [round(v, 6) if isinstance(v, float) else v for v in observed_values]
            assert observed_values == expected_values
    
    def test_shuffle_labels(self, dna_test_data_long, dna_test_data_long_shuffled):
        """Test shuffling of labels."""
        dna_test_data_long.csv_processing.shuffle_labels(seed=42)
        npt.assert_array_equal(
            dna_test_data_long.csv_processing.data["hola:label:float"],
            dna_test_data_long_shuffled.csv_processing.data["hola:label:float"]
        )
