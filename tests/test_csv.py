import json
import os
from abc import ABC

import numpy as np
import numpy.testing as npt
import polars as pl
import pytest

from src.stimulus.data.csv import CsvLoader, CsvProcessing
from src.stimulus.data.experiments import DnaToFloatExperiment, ProtDnaToFloatExperiment


@pytest.fixture
def dna_float_setup():
    np.random.seed(123)
    pl.set_random_seed(123)
    experiment = DnaToFloatExperiment()
    csv_path = os.path.abspath("tests/test_data/dna_experiment/test.csv")
    csv_processing = CsvProcessing(experiment, csv_path)
    
    csv_shuffle_long_path = os.path.abspath("tests/test_data/dna_experiment/test_shuffling_long.csv")
    csv_shuffle_long = CsvProcessing(experiment, csv_shuffle_long_path)
    
    csv_shuffle_long_shuffled_path = os.path.abspath("tests/test_data/dna_experiment/test_shuffling_long_shuffled.csv")
    csv_shuffle_long_shuffled = CsvProcessing(experiment, csv_shuffle_long_shuffled_path)
    
    with open("tests/test_data/dna_experiment/test_config.json") as f:
        configs = json.load(f)
    
    return {
        "csv_processing": csv_processing,
        "configs": configs,
        "data_length": 2,
        "expected_splits": [1, 0],
        "csv_shuffle_long": csv_shuffle_long,
        "csv_shuffle_long_shuffled": csv_shuffle_long_shuffled
    }

# @pytest.fixture
# def prot_dna_float_setup():
#     experiment = ProtDnaToFloatExperiment()
#     csv_path = os.path.abspath("tests/test_data/prot_dna_experiment/test.csv")
#     csv_processing = CsvProcessing(experiment, csv_path)
    
#     with open("tests/test_data/prot_dna_experiment/test_config.json") as f:
#         configs = json.load(f)
    
#     return {
#         "csv_processing": csv_processing,
#         "configs": configs,
#         "data_length": 2,
#         "expected_splits": [1, 0]
#     }

# def test_dna_float_len(dna_float_setup):
#     """Test if data is loaded with correct shape."""
#     assert len(dna_float_setup["csv_processing"].data) == dna_float_setup["data_length"]

# def test_dna_float_add_split(dna_float_setup):
#     """Test adding split to the data."""
#     csv_processing = dna_float_setup["csv_processing"]
#     csv_processing.add_split(dna_float_setup["configs"]["split"])
    
#     for i in range(dna_float_setup["data_length"]):
#         assert csv_processing.data["split:split:int"][i] == dna_float_setup["expected_splits"][i]

# def test_dna_float_transform(dna_float_setup):
#     """Test data transformation."""
#     csv_processing = dna_float_setup["csv_processing"]
#     csv_processing.transform(dna_float_setup["configs"]["transform"])
    
#     # Test transformed data
#     expected_pets = ["cat", "dog", "cat", "dog"]
#     expected_hola = [12.676405, 12.540016, 12.676405, 12.540016]
#     expected_hello = ["ACTGACTGATCGATNN", "ACTGACTGATCGATNN", "NNATCGATCAGTCAGT", "NNATCGATCAGTCAGT"]
#     expected_splits = [1, 0, 1, 0]
    
#     assert list(csv_processing.data["pet:meta:str"]) == expected_pets
#     assert [round(v, 6) for v in csv_processing.data["hola:label:float"]] == expected_hola
#     assert list(csv_processing.data["hello:input:dna"]) == expected_hello
#     assert list(csv_processing.data["split:split:int"]) == expected_splits

# # ... similar transformations for other test methods ...

# @pytest.fixture
# def dna_loader_setup():
#     csv_path = os.path.abspath("tests/test_data/dna_experiment/test.csv")
#     csv_path_split = os.path.abspath("tests/test_data/dna_experiment/test_with_split.csv")
#     experiment = DnaToFloatExperiment()
#     csv_loader = CsvLoader(experiment, csv_path)
    
#     return {
#         "csv_loader": csv_loader,
#         "csv_path_split": csv_path_split,
#         "experiment": experiment,
#         "data_shape": [2, 3],
#         "data_shape_split": [48, 4],
#         "shape_splits": {0: 16, 1: 16, 2: 16}
#     }

# def test_loader_len(dna_loader_setup):
#     """Test the length of the dataset."""
#     assert len(dna_loader_setup["csv_loader"]) == dna_loader_setup["data_shape"][0]

# # ... continue with other test methods ...