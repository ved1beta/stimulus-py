import pytest
import ray
import yaml
import os

from stimulus.data.loaders import EncoderLoader
from stimulus.learner.raytune_learner import TuneWrapper
from stimulus.utils.yaml_data import YamlSubConfigDict
from stimulus.utils.yaml_model_schema import Model, RayTuneModel, YamlRayConfigLoader
from tests.test_model import titanic_model


@pytest.fixture
def ray_config_loader() -> RayTuneModel:
    with open("tests/test_model/titanic_model_cpu.yaml") as file:
        model_config = yaml.safe_load(file)
    return YamlRayConfigLoader(Model(**model_config)).get_config()


@pytest.fixture
def encoder_loader() -> EncoderLoader:
    with open("tests/test_data/titanic/titanic_sub_config.yaml") as file:
        data_config = yaml.safe_load(file)
    return EncoderLoader(YamlSubConfigDict(**data_config).columns)


def test_TuneWrapper_init(ray_config_loader: RayTuneModel, encoder_loader: EncoderLoader):
    ray.init()
    tune_wrapper = TuneWrapper(
        model_config=ray_config_loader,
        model_class=titanic_model,
        data_path="tests/test_data/titanic/titanic_stimulus_split.csv",
        data_config_path="tests/test_data/titanic/titanic_sub_config.yaml",
        encoder_loader=encoder_loader,
        seed=42,
        ray_results_dir=os.path.abspath("tests/test_data/titanic/ray_results"),
        tune_run_name="test_run",
        debug=False,
        autoscaler=False,
    )

    assert isinstance(tune_wrapper, TuneWrapper)
    ray.shutdown()
