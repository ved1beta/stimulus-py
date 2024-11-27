import pytest
import yaml

from stimulus.utils import yaml_experiment_utils

@pytest.fixture
def load_yaml_from_file():
    """
    This fixture loads the test yaml file and returns the dictionary.
    """
    with open("tests/test_data/dna_experiment/dna_experiment_config_template.yaml") as f:
        return yaml.safe_load(f)



if __name__ == "__main__":

    yaml_config = {}
    with open("tests/test_data/dna_experiment/dna_experiment_config_template.yaml") as conf_file:
        yaml_config = yaml.safe_load(conf_file)
    print(yaml_config)