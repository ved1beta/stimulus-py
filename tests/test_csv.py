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
SEED = 123
# np.random.seed(SEED)
# pl.set_random_seed(SEED)

# ----- set up objects ----- #

class csv_data():
    """ This is a class to initialize the corresponding classes for a given csv data. """
    def __init__(self, filename, experiment):
        self.experiment = experiment()
        self.csv_path = os.path.abspath(filename)
        self.csv_processing = CsvProcessing(self.experiment, self.csv_path)
        self.data_length = None
        self.values = None

@pytest.fixture
def dna_test_data():
    """ This is the csv_data class storing the basic dna test csv """
    data = csv_data("tests/test_data/dna_experiment/test.csv", DnaToFloatExperiment)
    data.data_length = 2
    data.values = {
        "pet:meta:str" : ["cat", "dog", "cat", "dog"],
        "hola:label:float" : [12.676405, 12.540016, 12.676405, 12.540016],
        "hello:input:dna" : ["ACTGACTGATCGATNN", "ACTGACTGATCGATNN", "NNATCGATCAGTCAGT", "NNATCGATCAGTCAGT"]
    }
    return data

@pytest.fixture
def dna_test_data_long():
    """ This is the csv_data class storing the long dna test csv """
    data = csv_data("tests/test_data/dna_experiment/test_shuffling_long.csv", DnaToFloatExperiment)
    data.data_length = 1000
    return data

@pytest.fixture
def dna_test_data_long_shuffled():
    """ This is the csv_data class storing the shuffled long dna test csv """
    data = csv_data("tests/test_data/dna_experiment/test_shuffling_long_shuffled.csv", ProtDnaToFloatExperiment)
    data.data_length = 1000
    return data

@pytest.fixture
def prot_dna_test_data():
    """ This is the csv_data class storing the basic prot-dna test csv """
    data = csv_data("tests/test_data/prot_dna_experiment/test.csv", ProtDnaToFloatExperiment)
    data.data_length = 2
    return data

@pytest.fixture
def prot_dna_test_data_with_split():
    """ This is the csv_data class storing the basic prot-dna test csv with split """
    data = csv_data("tests/test_data/prot_dna_experiment/test_with_split.csv", ProtDnaToFloatExperiment)
    data.data_length = 3
    return data

@pytest.fixture
def dna_config():
    """ This is the config file for the dna experiment """
    with open("tests/test_data/dna_experiment/test_config.json") as f:
        return json.load(f)
    
@pytest.fixture
def prot_dna_config():
    """ This is the config file for the prot-dna experiment """
    with open("tests/test_data/prot_dna_experiment/test_config.json") as f:
        return json.load(f)

# ----- run tests ----- #

# TODO remove repetitive tests by using parametrize option

def test_len(dna_test_data, dna_test_data_long, dna_test_data_long_shuffled, prot_dna_test_data, prot_dna_test_data_with_split):
    """ this tests the len() function properly returns the length of the csv data """
    assert len(dna_test_data.csv_processing.data) == dna_test_data.data_length
    assert len(dna_test_data_long.csv_processing.data) == dna_test_data_long.data_length
    assert len(dna_test_data_long_shuffled.csv_processing.data) == dna_test_data_long_shuffled.data_length
    assert len(prot_dna_test_data.csv_processing.data) == prot_dna_test_data.data_length
    assert len(prot_dna_test_data_with_split.csv_processing.data) == prot_dna_test_data_with_split.data_length

def test_add_split(dna_test_data, prot_dna_test_data, dna_config, prot_dna_config):
    """ this tests the add_split function properly adds the split column """
    dna_test_data.csv_processing.add_split(dna_config["split"])
    prot_dna_test_data.csv_processing.add_split(prot_dna_config["split"])
    print(dna_test_data.csv_processing.data)
    assert dna_test_data.csv_processing.data["split:split:int"].to_list() == [1, 0]
    assert prot_dna_test_data.csv_processing.data["split:split:int"].to_list() == [1, 0]

