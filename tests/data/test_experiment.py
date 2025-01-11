import pytest
import stimulus.data.experiments as experiments
import numpy as np

from stimulus.data.transform import data_transformation_generators
from stimulus.data.encoding.encoders import AbstractEncoder

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
def TextOneHotEncoder_name_and_params():
    return "TextOneHotEncoder", {"alphabet": "acgt"}


def test_get_config_from_yaml(dna_experiment_config_path):
    """Test the get_config_from_yaml method of the AbstractExperiment class.

    This test checks if the get_config_from_yaml method correctly parses the YAML configuration file.
    """
    experiment = experiments.AbstractLoader()
    config = experiment.get_config_from_yaml(dna_experiment_config_path)
    assert config is not None

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
    assert experiment.ciao["encoder"] == encoder
    assert experiment.get_function_encode_all("ciao") == encoder.encode_all

def test_build_experiment_class_encoder_dict(dna_experiment_config_path):
    """Test the build_experiment_class_encoder_dict method of the AbstractExperiment class.

    This test checks if the build_experiment_class_encoder_dict method correctly builds the experiment class from a config dictionary.
    """
    experiment = experiments.EncoderLoader()
    config = experiment.get_config_from_yaml(dna_experiment_config_path)["columns"]
    experiment.initialize_column_encoders_from_config(config)
    assert hasattr(experiment, "hello")
    assert hasattr(experiment, "bonjour")
    assert hasattr(experiment, "ciao")

    # call encoder from "hello", check that it completes successfully
    assert experiment.hello["encoder"].encode_all(["a", "c", "g", "t"]) is not None
    
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
    assert experiment.col1["data_transformation_generators"] == transformer

def test_initialize_column_data_transformers_from_config(dna_experiment_config_path):
    """Test the initialize_column_data_transformers_from_config method of the TransformLoader class.

    This test checks if the initialize_column_data_transformers_from_config method correctly builds 
    the experiment class from a config dictionary.
    """
    experiment = experiments.TransformLoader()
    config = experiment.get_config_from_yaml(dna_experiment_config_path)["transforms"]
    experiment.initialize_column_data_transformers_from_config(config)
    
    # Check columns have transformers set
    assert hasattr(experiment, "col1")
    assert hasattr(experiment, "col2")

    # Check transformers were properly initialized
    col1_transformers = experiment.col1["data_transformation_generators"]
    col2_transformers = experiment.col2["data_transformation_generators"]

    # Verify col1 has the expected transformers
    assert any(isinstance(t, data_transformation_generators.ReverseComplement) for t in col1_transformers)
    assert any(isinstance(t, data_transformation_generators.UniformTextMasker) for t in col1_transformers)
    assert any(isinstance(t, data_transformation_generators.GaussianNoise) for t in col1_transformers)

    # Verify col2 has the expected transformer
    assert any(isinstance(t, data_transformation_generators.GaussianNoise) for t in col2_transformers)
