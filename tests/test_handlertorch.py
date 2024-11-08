import os
from typing import Any, Dict

import pytest
import torch

from src.stimulus.data.experiments import DnaToFloatExperiment, ProtDnaToFloatExperiment
from src.stimulus.data.handlertorch import TorchDataset


class TorchTestData:
    def __init__(self, filename: str, experiment: Any):
        self.experiment = experiment()
        self.csv_path = os.path.abspath(filename)
        self.torch_dataset = TorchDataset(self.csv_path, self.experiment)
        print(self.torch_dataset.input)
        self.expected_len = None
        self.expected_input_shape = None
        self.expected_label_shape = None
        self.expected_item_shape = None


@pytest.fixture
def dna_test_data():
    data = TorchTestData("tests/test_data/dna_experiment/test.csv", DnaToFloatExperiment)
    data.expected_len = 2
    data.expected_input_shape = {"hello": [2, 16, 4]}
    data.expected_label_shape = {"hola": [2]}
    data.expected_item_shape = {"hello": [16, 4]}
    return data


@pytest.fixture
def dna_test_data_with_float():
    data = TorchTestData("tests/test_data/dna_experiment/test_unequal_dna_float.csv", DnaToFloatExperiment)
    data.expected_len = 4
    data.expected_input_shape = {"hello": [4, 31, 4]}
    data.expected_label_shape = {"hola": [4]}
    data.expected_item_shape = {"hello": [31, 4]}
    return data


@pytest.fixture
def prot_dna_test_data():
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
    def test_input_is_dict(self, request, fixture_name: str):
        data = request.getfixturevalue(fixture_name)
        assert isinstance(data.torch_dataset.input, Dict)

    def test_label_is_dict(self, request, fixture_name: str):
        data = request.getfixturevalue(fixture_name)
        assert isinstance(data.torch_dataset.label, Dict)

    def test_input_tensor_shape(self, request, fixture_name: str):
        data = request.getfixturevalue(fixture_name)
        for key, tensor in data.torch_dataset.input.items():
            assert isinstance(tensor, torch.Tensor)
            assert tensor.shape == torch.Size(data.expected_input_shape[key])

    def test_label_tensor_shape(self, request, fixture_name: str):
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
    def test_is_dict(self, request, fixture_name: str, idx: Any):
        data = request.getfixturevalue(fixture_name)
        x, y, meta = data.torch_dataset[idx]
        assert isinstance(x, dict)
        assert isinstance(y, dict)
        assert isinstance(meta, dict)

    def test_get_correct_items(self, request, fixture_name: str, idx: Any):
        data = request.getfixturevalue(fixture_name)
        x, y, meta = data.torch_dataset[idx]
        dict_items = {**x, **y, **meta}
        for key, expected_item_shape in data.expected_item_shape.items():
            if isinstance(idx, slice):
                slice_len = idx.stop - idx.start
                expected_item_shape = [slice_len] + expected_item_shape
            assert dict_items[key].shape == torch.Size(expected_item_shape)


# TODO add test for titanic dataset
