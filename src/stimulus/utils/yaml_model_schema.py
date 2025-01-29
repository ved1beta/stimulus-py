"""Module for handling YAML configuration files and converting them to Ray Tune format."""

import random
from collections.abc import Callable
from typing import Any, Optional

import pydantic
from ray import tune
from ray.tune.search.sample import Domain


class CustomTunableParameter(pydantic.BaseModel):
    """Custom tunable parameter."""

    function: str
    sample_space: list[Any]
    n_space: list[Any]
    mode: str


class TunableParameter(pydantic.BaseModel):
    """Tunable parameter."""

    space: list[Any]
    mode: str

    @pydantic.model_validator(mode="after")
    def validate_mode(self) -> "TunableParameter":
        """Validate that mode is supported by Ray Tune."""
        if not hasattr(tune, self.mode):
            raise AttributeError(
                f"Mode {self.mode} not recognized, check the ray.tune documentation at https://docs.ray.io/en/master/tune/api_docs/suggestion.html",
            )

        mode = getattr(tune, self.mode)
        if mode.__name__ not in [
            "choice",
            "uniform",
            "loguniform",
            "quniform",
            "qloguniform",
            "qnormal",
            "randint",
            "sample_from",
        ]:
            raise NotImplementedError(f"Mode {mode.__name__} not implemented yet")

        return self


class Loss(pydantic.BaseModel):
    """Loss parameters."""

    loss_fn: TunableParameter


class Data(pydantic.BaseModel):
    """Data parameters."""

    batch_size: TunableParameter


class TuneParams(pydantic.BaseModel):
    """Tune parameters."""

    metric: str
    mode: str
    num_samples: int


class Scheduler(pydantic.BaseModel):
    """Scheduler parameters."""

    name: str
    params: dict[str, Any]


class RunParams(pydantic.BaseModel):
    """Run parameters."""

    stop: Optional[dict[str, Any]] = None


class Tune(pydantic.BaseModel):
    """Tune parameters."""

    config_name: str
    tune_params: TuneParams
    scheduler: Scheduler
    run_params: RunParams
    step_size: Optional[int] = 1
    gpu_per_trial: int
    cpu_per_trial: int


class Model(pydantic.BaseModel):
    """Model configuration."""

    network_params: dict[str, TunableParameter]
    optimizer_params: dict[str, TunableParameter]
    loss_params: Loss
    data_params: Data
    tune: Tune


class RayTuneModel(pydantic.BaseModel):
    """Ray Tune compatible model configuration."""

    model_config = {
        "arbitrary_types_allowed": True,  # Add this line to allow Domain type
    }

    network_params: dict[str, Domain]
    optimizer_params: dict[str, Domain]
    loss_params: dict[str, Domain]
    data_params: dict[str, Domain]
    tune: Tune


class YamlRayConfigLoader:
    """Load and convert YAML configurations to Ray Tune format.

    This class handles loading model configurations and converting them into
    formats compatible with Ray Tune's hyperparameter search spaces.
    """

    def __init__(self, model: Model) -> None:
        """Initialize the config loader with a Model instance.

        Args:
            model: Pydantic Model instance containing configuration
        """
        self.model = model
        self.ray_model = self.convert_config_to_ray(model)

    def raytune_space_selector(self, mode: Callable, space: list) -> Any:
        """Convert space parameters to Ray Tune format based on the mode.

        Args:
            mode: Ray Tune search space function (e.g., tune.choice, tune.uniform)
            space: List of parameters defining the search space

        Returns:
            Configured Ray Tune search space
        """
        if mode.__name__ == "choice":
            return mode(space)

        return mode(*tuple(space))

    def raytune_sample_from(self, mode: Callable, param: CustomTunableParameter) -> Any:
        """Apply tune.sample_from to a given custom sampling function.

        Args:
            mode: Ray Tune sampling function
            param: TunableParameter containing sampling parameters

        Returns:
            Configured sampling function

        Raises:
            NotImplementedError: If the sampling function is not supported
        """
        if param.function == "sampint":
            return mode(lambda _: self.sampint(param.sample_space, param.n_space))

        raise NotImplementedError(f"Function {param.function} not implemented yet")

    def convert_raytune(self, param: TunableParameter | CustomTunableParameter) -> Any:
        """Convert parameter configuration to Ray Tune format.

        Args:
            param: Parameter configuration

        Returns:
            Ray Tune compatible parameter configuration
        """
        mode = getattr(tune, param.mode)

        if isinstance(param, TunableParameter):
            return self.raytune_space_selector(mode, param.space)
        return self.raytune_sample_from(mode, param)

    def convert_config_to_ray(self, model: Model) -> RayTuneModel:
        """Convert Model configuration to Ray Tune format.

        Converts parameters in network_params and optimizer_params to Ray Tune search spaces.

        Args:
            model: Model configuration

        Returns:
            Ray Tune compatible model configuration
        """
        return RayTuneModel(
            network_params={k: self.convert_raytune(v) for k, v in model.network_params.items()},
            optimizer_params={k: self.convert_raytune(v) for k, v in model.optimizer_params.items()},
            loss_params={k: self.convert_raytune(v) for k, v in model.loss_params},
            data_params={k: self.convert_raytune(v) for k, v in model.data_params},
            tune=model.tune,
        )

    def get_config(self) -> RayTuneModel:
        """Return the current configuration.

        Returns:
            Current configuration dictionary
        """
        return self.ray_model

    @staticmethod
    def sampint(sample_space: list, n_space: list) -> list[int]:
        """Return a list of n random samples from the sample_space.

        This function is useful for sampling different numbers of layers,
        each with different numbers of neurons.

        Args:
            sample_space: List [min, max] defining range of values to sample from
            n_space: List [min, max] defining range for number of samples

        Returns:
            List of randomly sampled integers

        Note:
            Uses Python's random module which is not cryptographically secure.
            This is acceptable for hyperparameter sampling but should not be
            used for security-critical purposes (S311 fails when linting).
        """
        sample_space_list = list(range(sample_space[0], sample_space[1] + 1))
        n_space_list = list(range(n_space[0], n_space[1] + 1))
        n = random.choice(n_space_list)  # noqa: S311
        return random.sample(sample_space_list, n)
