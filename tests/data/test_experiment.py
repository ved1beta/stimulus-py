import pytest
import stimulus.data.experiments as experiments
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
    experiment = experiments.AbstractExperiment()
    config = experiment.get_config_from_yaml(dna_experiment_config_path)
    assert config is not None

def test_get_encoder(TextOneHotEncoder_name_and_params):
    """Test the get_encoder method of the AbstractExperiment class.

    This test checks if the get_encoder method correctly returns the encoder function.
    """
    experiment = experiments.AbstractExperiment()
    encoder_name, encoder_params = TextOneHotEncoder_name_and_params
    encoder = experiment.get_encoder(encoder_name, encoder_params)
    assert isinstance(encoder, AbstractEncoder)

def test_set_encoder_as_attribute(TextOneHotEncoder_name_and_params):
    """Test the set_encoder_as_attribute method of the AbstractExperiment class.

    This test checks if the set_encoder_as_attribute method correctly sets the encoder as an attribute of the experiment class.
    """
    experiment = experiments.AbstractExperiment()
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
    experiment = experiments.AbstractExperiment()
    config = experiment.get_config_from_yaml(dna_experiment_config_path)["columns"]
    experiment.build_experiment_class_encoder_from_config(config)
    assert hasattr(experiment, "hello")
    assert hasattr(experiment, "bonjour")
    assert hasattr(experiment, "ciao")




