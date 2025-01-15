import pytest
import polars as pl
import numpy as np
from pathlib import Path
import yaml

from stimulus.data.csv import DatasetHandler, DatasetManager, EncodeManager, TransformManager, SplitManager
from stimulus.utils.yaml_data import generate_data_configs, YamlConfigDict
from stimulus.data import experiments

# Fixtures
@pytest.fixture
def titanic_csv_path():
    return "tests/test_data/titanic/titanic_stimulus.csv"

@pytest.fixture
def config_path():
    return "tests/test_data/titanic/titanic.yaml"

@pytest.fixture
def base_config(config_path):
    with open(config_path, 'r') as f:
        return YamlConfigDict(**yaml.safe_load(f))

@pytest.fixture
def split_configs(base_config):
    """Generate all possible configurations from base config"""
    return generate_data_configs(base_config)


# Test DatasetHandler Integration
@pytest.fixture
def encoder_loader(base_config):
    loader = experiments.EncoderLoader()
    loader.initialize_column_encoders_from_config(base_config.columns)
    return loader

@pytest.fixture
def transform_loader(base_config):
    loader = experiments.TransformLoader()
    if "transforms" in base_config:
        loader.initialize_column_data_transformers_from_config(base_config["transforms"])
    return loader

@pytest.fixture
def split_loader(base_config):
    loader = experiments.SplitLoader()
    if "split" in base_config:
        # Get first split configuration
        split_config = base_config["split"][0]
        splitter = loader.get_splitter(split_config["split_method"])
        loader.set_splitter_as_attribute("split", splitter)
    return loader

# Test DatasetManager
def test_dataset_manager_init(config_path):
    manager = DatasetManager(config_path)
    assert hasattr(manager, "config")
    assert hasattr(manager, "column_categories")

def test_dataset_manager_organize_columns(config_path):
    manager = DatasetManager(config_path)
    categories = manager.categorize_columns_by_type()
    
    assert "pclass" in categories["input"]
    assert "sex" in categories["input"] 
    assert "age" in categories["input"]
    assert "survived" in categories["label"]
    assert "passenger_id" in categories["meta"]

def test_dataset_manager_organize_transforms(config_path):
    manager = DatasetManager(config_path)
    categories = manager.categorize_columns_by_type()
    
    assert len(categories) == 3
    assert all(key in categories for key in ["input", "label", "meta"])

# Test EncodeManager
def test_encode_manager_init():
    encoder_loader = experiments.EncoderLoader()
    manager = EncodeManager(encoder_loader)
    assert hasattr(manager, "encoder_loader")

def test_encode_manager_initialize_encoders():
    encoder_loader = experiments.EncoderLoader()
    manager = EncodeManager(encoder_loader)
    assert hasattr(manager, "encoder_loader")

def test_encode_manager_encode_numeric():
    encoder_loader = experiments.EncoderLoader()
    intencoder = encoder_loader.get_encoder("IntEncoder")
    encoder_loader.set_encoder_as_attribute("test_col", intencoder)
    manager = EncodeManager(encoder_loader)
    data = [1, 2, 3]
    encoded = manager.encode_column("test_col", data)
    assert encoded is not None

# Test TransformManager
def test_transform_manager_init():
    transform_loader = experiments.TransformLoader()
    manager = TransformManager(transform_loader)
    assert hasattr(manager, "transform_loader")

def test_transform_manager_initialize_transforms():
    transform_loader = experiments.TransformLoader()
    manager = TransformManager(transform_loader)
    assert hasattr(manager, "transform_loader")

def test_transform_manager_apply_transforms():
    transform_loader = experiments.TransformLoader()
    manager = TransformManager(transform_loader)
    assert hasattr(manager, "transform_loader")

# Test SplitManager
def test_split_manager_init():
    split_loader = experiments.SplitLoader()
    manager = SplitManager(split_loader)
    assert hasattr(manager, "split_loader")

def test_split_manager_initialize_splits():
    split_loader = experiments.SplitLoader()
    manager = SplitManager(split_loader)
    assert hasattr(manager, "split_loader")

def test_split_manager_apply_split():
    split_loader = experiments.SplitLoader(seed=42)
    manager = SplitManager(split_loader)
    data = pl.DataFrame({"col": range(100)})
    assert hasattr(manager, "split_loader")



def test_dataset_handler_init(config_path, titanic_csv_path, encoder_loader, transform_loader, split_loader):
    handler = DatasetHandler(
        config_path=config_path,
        encoder_loader=encoder_loader,
        transform_loader=transform_loader, 
        split_loader=split_loader,
        csv_path=titanic_csv_path
    )
    
    assert isinstance(handler.encoder_manager, EncodeManager)
    assert isinstance(handler.transform_manager, TransformManager)
    assert isinstance(handler.split_manager, SplitManager)

def test_dataset_handler_get_dataset(config_path, titanic_csv_path, encoder_loader):
    transform_loader = experiments.TransformLoader()
    split_loader = experiments.SplitLoader()
    
    handler = DatasetHandler(
        config_path=config_path,
        encoder_loader=encoder_loader,
        transform_loader=transform_loader, 
        split_loader=split_loader,
        csv_path=titanic_csv_path
    )
    
    dataset = handler.get_all_items()
    assert isinstance(dataset, tuple)

def test_dataset_handler_print_dataset_info(config_path, titanic_csv_path, encoder_loader):
    transform_loader = experiments.TransformLoader()
    split_loader = experiments.SplitLoader()
    
    handler = DatasetHandler(
        config_path=config_path,
        encoder_loader=encoder_loader,
        transform_loader=transform_loader, 
        split_loader=split_loader,
        csv_path=titanic_csv_path
    )
    
    input_dict, label_dict, meta_dict = handler.get_all_items()
    # Print input dict keys and first 5 elements of each value
    print("\nInput dictionary contents:")
    for key, value in input_dict.items():
        print(f"\n{key}:")
        if isinstance(value, np.ndarray):
            print(value[:5])  # Print first 5 elements if numpy array
        else:
            print(value[:5])  # Print first 5 elements if list

@pytest.mark.parametrize("config_idx", [0, 1])  # Test both split configs
def test_dataset_handler_different_configs(config_path, titanic_csv_path, config_idx, encoder_loader):
    transform_loader = experiments.TransformLoader()
    split_loader = experiments.SplitLoader()
    
    handler = DatasetHandler(
        config_path=config_path,
        encoder_loader=encoder_loader,
        transform_loader=transform_loader, 
        split_loader=split_loader,
        csv_path=titanic_csv_path
    )
    
    dataset = handler.get_all_items()
    assert isinstance(dataset, tuple)
