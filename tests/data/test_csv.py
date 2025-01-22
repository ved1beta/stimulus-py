from pathlib import Path

import pytest
import yaml

from stimulus.data import experiments
from stimulus.data.csv import (
    DatasetLoader,
    DatasetManager,
    DatasetProcessor,
    EncodeManager,
    SplitManager,
    TransformManager,
)
from stimulus.utils.yaml_data import (
    YamlConfigDict,
    YamlTransform,
    YamlTransformColumns,
    YamlTransformColumnsTransformation,
    dump_yaml_list_into_files,
    generate_data_configs,
)


# Fixtures
## Data fixtures
@pytest.fixture
def titanic_csv_path():
    return "tests/test_data/titanic/titanic_stimulus.csv"


@pytest.fixture
def config_path():
    return "tests/test_data/titanic/titanic.yaml"


@pytest.fixture
def base_config(config_path):
    with open(config_path) as f:
        return YamlConfigDict(**yaml.safe_load(f))


@pytest.fixture
def generate_sub_configs(base_config):
    """Generate all possible configurations from base config"""
    return generate_data_configs(base_config)


@pytest.fixture
def dump_single_split_config_to_disk():
    return "tests/test_data/titanic/titanic_sub_config.yaml"


## Loader fixtures
@pytest.fixture
def encoder_loader(generate_sub_configs):
    loader = experiments.EncoderLoader()
    loader.initialize_column_encoders_from_config(generate_sub_configs[0].columns)
    return loader


@pytest.fixture
def transform_loader(generate_sub_configs):
    loader = experiments.TransformLoader()
    loader.initialize_column_data_transformers_from_config(generate_sub_configs[0].transforms)
    return loader


@pytest.fixture
def split_loader(generate_sub_configs):
    loader = experiments.SplitLoader()
    loader.initialize_splitter_from_config(generate_sub_configs[0].split)
    return loader


# Test DatasetManager
def test_dataset_manager_init(dump_single_split_config_to_disk):
    manager = DatasetManager(dump_single_split_config_to_disk)
    assert hasattr(manager, "config")
    assert hasattr(manager, "column_categories")


def test_dataset_manager_organize_columns(dump_single_split_config_to_disk):
    manager = DatasetManager(dump_single_split_config_to_disk)
    categories = manager.categorize_columns_by_type()

    assert "pclass" in categories["input"]
    assert "sex" in categories["input"]
    assert "age" in categories["input"]
    assert "survived" in categories["label"]
    assert "passenger_id" in categories["meta"]


def test_dataset_manager_organize_transforms(dump_single_split_config_to_disk):
    manager = DatasetManager(dump_single_split_config_to_disk)
    categories = manager.categorize_columns_by_type()

    assert len(categories) == 3
    assert all(key in categories for key in ["input", "label", "meta"])


def test_dataset_manager_get_transform_logic(dump_single_split_config_to_disk):
    manager = DatasetManager(dump_single_split_config_to_disk)
    transform_logic = manager.get_transform_logic()
    assert transform_logic["transformation_name"] == "noise"
    assert len(transform_logic["transformations"]) == 2


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
    intencoder = encoder_loader.get_encoder("NumericEncoder")
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


def test_transform_manager_transform_column():
    transform_loader = experiments.TransformLoader()
    dummy_config = YamlTransform(
        transformation_name="GaussianNoise",
        columns=[
            YamlTransformColumns(
                column_name="test_col",
                transformations=[YamlTransformColumnsTransformation(name="GaussianNoise", params={"std": 0.1})],
            )
        ],
    )
    transform_loader.initialize_column_data_transformers_from_config(dummy_config)
    manager = TransformManager(transform_loader)
    data = [1, 2, 3]
    transformed, added_row = manager.transform_column("test_col", "GaussianNoise", data)
    assert len(transformed) == len(data)
    assert added_row is False


# Test SplitManager
def test_split_manager_init(split_loader):
    manager = SplitManager(split_loader)
    assert hasattr(manager, "split_loader")


def test_split_manager_initialize_splits(split_loader):
    manager = SplitManager(split_loader)
    assert hasattr(manager, "split_loader")


def test_split_manager_apply_split(split_loader):
    manager = SplitManager(split_loader)
    data = {"col": range(100)}
    split_indices = manager.get_split_indices(data)
    assert len(split_indices) == 3
    assert len(split_indices[0]) == 70
    assert len(split_indices[1]) == 15
    assert len(split_indices[2]) == 15


# Test DatasetProcessor
def test_dataset_processor_init(
    dump_single_split_config_to_disk,
    titanic_csv_path,
):
    processor = DatasetProcessor(
        config_path=dump_single_split_config_to_disk,
        csv_path=titanic_csv_path,
    )

    assert isinstance(processor.dataset_manager, DatasetManager)
    assert processor.columns is not None


def test_dataset_processor_apply_split(
    dump_single_split_config_to_disk,
    titanic_csv_path,
    split_loader,
):
    processor = DatasetProcessor(
        config_path=dump_single_split_config_to_disk,
        csv_path=titanic_csv_path,
    )
    processor.data = processor.load_csv(titanic_csv_path)
    processor.add_split(split_manager=SplitManager(split_loader))
    assert "split" in processor.columns
    assert "split" in processor.data.columns
    assert len(processor.data["split"]) == 712


def test_dataset_processor_apply_transformation_group(
    dump_single_split_config_to_disk,
    titanic_csv_path,
    transform_loader,
):
    processor = DatasetProcessor(
        config_path=dump_single_split_config_to_disk,
        csv_path=titanic_csv_path,
    )
    processor.data = processor.load_csv(titanic_csv_path)

    processor_control = DatasetProcessor(
        config_path=dump_single_split_config_to_disk,
        csv_path=titanic_csv_path,
    )
    processor_control.data = processor_control.load_csv(titanic_csv_path)

    processor.apply_transformation_group(transform_manager=TransformManager(transform_loader))

    assert processor.data["age"].to_list() != processor_control.data["age"].to_list()
    assert processor.data["fare"].to_list() != processor_control.data["fare"].to_list()
    assert processor.data["parch"].to_list() == processor_control.data["parch"].to_list()
    assert processor.data["sibsp"].to_list() == processor_control.data["sibsp"].to_list()
    assert processor.data["pclass"].to_list() == processor_control.data["pclass"].to_list()
    assert processor.data["embarked"].to_list() == processor_control.data["embarked"].to_list()
    assert processor.data["sex"].to_list() == processor_control.data["sex"].to_list()


# Test DatasetLoader
def test_dataset_loader_init(dump_single_split_config_to_disk, titanic_csv_path, encoder_loader):
    loader = DatasetLoader(
        config_path=dump_single_split_config_to_disk,
        csv_path=titanic_csv_path,
        encoder_loader=encoder_loader,
    )

    assert isinstance(loader.dataset_manager, DatasetManager)
    assert loader.data is not None
    assert loader.columns is not None
    assert hasattr(loader, "encoder_manager")


def test_dataset_loader_get_dataset(dump_single_split_config_to_disk, titanic_csv_path, encoder_loader):
    loader = DatasetLoader(
        config_path=dump_single_split_config_to_disk,
        csv_path=titanic_csv_path,
        encoder_loader=encoder_loader,
    )

    dataset = loader.get_all_items()
    assert isinstance(dataset, tuple)
    assert len(dataset) == 3  # input_data, label_data, meta_data
