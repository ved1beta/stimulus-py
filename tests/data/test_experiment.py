import pytest
import yaml

from stimulus.data import experiments
from stimulus.data.encoding.encoders import AbstractEncoder
from stimulus.data.splitters import splitters
from stimulus.data.transform import data_transformation_generators
from stimulus.utils import yaml_data


@pytest.fixture
def dna_experiment_config_path():
    """Fixture that provides the path to the DNA experiment config template YAML file.

    This fixture returns the path to a YAML configuration file containing DNA experiment
    parameters, including column definitions and transformation specifications.

    Returns:
        str: Path to the DNA experiment config template YAML file
    """
    return "tests/test_data/dna_experiment/dna_experiment_config_template.yaml"


@pytest.fixture
def dna_experiment_sub_yaml(dna_experiment_config_path):
    # safe load the yaml file
    with open(dna_experiment_config_path) as f:
        yaml_dict = yaml.safe_load(f)
        yaml_config = yaml_data.YamlConfigDict(**yaml_dict)

    yaml_configs = yaml_data.generate_data_configs(yaml_config)
    return yaml_configs[0]


@pytest.fixture
def titanic_yaml_path():
    return "tests/test_data/titanic/titanic.yaml"


@pytest.fixture
def titanic_sub_yaml_path():
    return "tests/test_data/titanic/titanic_sub_config_0.yaml"


@pytest.fixture
def TextOneHotEncoder_name_and_params():
    return "TextOneHotEncoder", {"alphabet": "acgt"}


def test_get_encoder(TextOneHotEncoder_name_and_params):
    """Test the get_encoder method of the AbstractExperiment class.

    This test checks if the get_encoder method correctly returns the encoder function.
    """
    experiment = experiments.EncoderLoader()
    encoder_name, encoder_params = TextOneHotEncoder_name_and_params
    encoder = experiment.get_encoder(encoder_name, encoder_params)
    assert isinstance(encoder, AbstractEncoder)


def test_set_encoder_as_attribute(TextOneHotEncoder_name_and_params):
    """Test the set_encoder_as_attribute method of the AbstractExperiment class.

    This test checks if the set_encoder_as_attribute method correctly sets the encoder as an attribute of the experiment class.
    """
    experiment = experiments.EncoderLoader()
    encoder_name, encoder_params = TextOneHotEncoder_name_and_params
    encoder = experiment.get_encoder(encoder_name, encoder_params)
    experiment.set_encoder_as_attribute("ciao", encoder)
    assert hasattr(experiment, "ciao")
    assert experiment.ciao == encoder
    assert experiment.get_function_encode_all("ciao") == encoder.encode_all


def test_build_experiment_class_encoder_dict(dna_experiment_sub_yaml):
    """Test the build_experiment_class_encoder_dict method of the AbstractExperiment class.

    This test checks if the build_experiment_class_encoder_dict method correctly builds the experiment class from a config dictionary.
    """
    experiment = experiments.EncoderLoader()
    config = dna_experiment_sub_yaml.columns
    experiment.initialize_column_encoders_from_config(config)
    assert hasattr(experiment, "hello")
    assert hasattr(experiment, "bonjour")
    assert hasattr(experiment, "ciao")

    # call encoder from "hello", check that it completes successfully
    assert experiment.hello.encode_all(["a", "c", "g", "t"]) is not None


def test_get_data_transformer():
    """Test the get_data_transformer method of the TransformLoader class.

    This test checks if the get_data_transformer method correctly returns the transformer function.
    """
    experiment = experiments.TransformLoader()
    transformer = experiment.get_data_transformer("ReverseComplement")
    assert isinstance(transformer, data_transformation_generators.ReverseComplement)


def test_set_data_transformer_as_attribute():
    """Test the set_data_transformer_as_attribute method of the TransformLoader class.

    This test checks if the set_data_transformer_as_attribute method correctly sets the transformer
    as an attribute of the experiment class.
    """
    experiment = experiments.TransformLoader()
    transformer = experiment.get_data_transformer("ReverseComplement")
    experiment.set_data_transformer_as_attribute("col1", transformer)
    assert hasattr(experiment, "col1")
    assert experiment.col1["ReverseComplement"] == transformer


def test_initialize_column_data_transformers_from_config(dna_experiment_sub_yaml):
    """Test the initialize_column_data_transformers_from_config method of the TransformLoader class."""
    experiment = experiments.TransformLoader()
    config = dna_experiment_sub_yaml.transforms
    experiment.initialize_column_data_transformers_from_config(config)

    # Check that the column from the config exists
    assert hasattr(experiment, "col1")

    # Get transformers for the column
    column_transformers = experiment.col1

    # Verify the column has the expected transformers
    assert any(isinstance(t, data_transformation_generators.ReverseComplement) for t in column_transformers.values())


def test_initialize_splitter_from_config(dna_experiment_sub_yaml):
    experiment = experiments.SplitLoader()
    config = dna_experiment_sub_yaml.split
    experiment.initialize_splitter_from_config(config)
    assert hasattr(experiment, "split")
    assert isinstance(experiment.split, splitters.RandomSplit)
