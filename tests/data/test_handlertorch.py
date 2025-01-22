import os

import pytest
import yaml

from src.stimulus.data import experiments, handlertorch
from src.stimulus.utils import yaml_data


@pytest.fixture
def titanic_config_path():
    return os.path.abspath("tests/test_data/titanic/titanic_sub_config.yaml")


@pytest.fixture
def titanic_csv_path():
    return os.path.abspath("tests/test_data/titanic/titanic_stimulus.csv")


@pytest.fixture
def titanic_yaml_config(titanic_config_path):
    # Load the yaml config
    with open(titanic_config_path) as file:
        config = yaml.safe_load(file)
    return yaml_data.YamlSubConfigDict(**config)


@pytest.fixture
def titanic_encoder_loader(titanic_yaml_config):
    loader = experiments.EncoderLoader()
    loader.initialize_column_encoders_from_config(titanic_yaml_config.columns)
    return loader


def test_init_handlertorch(titanic_config_path, titanic_csv_path, titanic_encoder_loader):
    handlertorch.TorchDataset(
        config_path=titanic_config_path,
        csv_path=titanic_csv_path,
        encoder_loader=titanic_encoder_loader,
    )


def test_len_handlertorch(titanic_config_path, titanic_csv_path, titanic_encoder_loader):
    dataset = handlertorch.TorchDataset(
        config_path=titanic_config_path,
        csv_path=titanic_csv_path,
        encoder_loader=titanic_encoder_loader,
    )
    assert len(dataset) == 712


def test_getitem_handlertorch_slice(titanic_config_path, titanic_csv_path, titanic_encoder_loader):
    dataset = handlertorch.TorchDataset(
        config_path=titanic_config_path,
        csv_path=titanic_csv_path,
        encoder_loader=titanic_encoder_loader,
    )
    assert len(dataset[0:5]) == 3
    assert len(dataset[0:5][0]["pclass"]) == 5


def test_getitem_handlertorch_int(titanic_config_path, titanic_csv_path, titanic_encoder_loader):
    dataset = handlertorch.TorchDataset(
        config_path=titanic_config_path,
        csv_path=titanic_csv_path,
        encoder_loader=titanic_encoder_loader,
    )
    assert len(dataset[0]) == 3
