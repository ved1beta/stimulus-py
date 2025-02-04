"""Ray Tune results parser for extracting and saving best model configurations and weights."""

import os
from typing import Any, TypedDict

import pandas as pd
import torch
import yaml
from ray.train import Result
from ray.tune import ResultGrid
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file


class RayTuneResult(TypedDict):
    """TypedDict for storing Ray Tune optimization results."""

    config: dict[str, Any]
    checkpoint: str
    metrics_dataframe: pd.DataFrame


class RayTuneMetrics(TypedDict):
    """TypedDict for storing Ray Tune metrics results."""

    checkpoint: str
    metrics_dataframe: pd.DataFrame


class RayTuneOptimizer(TypedDict):
    """TypedDict for storing Ray Tune optimizer state."""

    checkpoint: str


class TuneParser:
    """Parser class for Ray Tune results to extract best configurations and model weights."""

    def __init__(self, result: ResultGrid) -> None:
        """Initialize with the given Ray Tune result grid."""
        self.result: ResultGrid = result
        self.best_result: Result = self._validate_best_result()

    def _validate_best_result(self) -> Result:
        """Safely retrieve and validate best result.

        Returns:
            The best result.

        Raises:
            ValueError: If no best result or checkpoint is found.
        """
        best_result: Result | None = self.result.get_best_result()
        if best_result is None:
            raise ValueError("No best result found in the result grid.")
        if best_result.checkpoint is None:
            raise ValueError("Best result does not contain a checkpoint.")
        return best_result

    def get_best_config(self) -> dict[str, Any]:
        """Get the best config from the results.

        Returns:
            The configuration dictionary of the best result.

        Raises:
            ValueError: If the config is missing.
        """
        config: dict[str, Any] | None = self.best_result.config
        if config is None:
            raise ValueError("Best result does not contain a configuration.")
        return config

    def save_best_config(self, output: str) -> None:
        """Save the best config to a file.

        TODO: maybe only save the relevant config values.

        Args:
            output: File path to save the configuration.
        """
        config: dict[str, Any] = self.get_best_config()
        config = self.fix_config_values(config)
        with open(output, "w") as f:
            yaml.safe_dump(config, f)

    def fix_config_values(self, config: dict[str, Any]) -> dict[str, Any]:
        """Correct config values.

        This method modifies the configuration dictionary to remove or convert
        non-serializable objects (such as Ray ObjectRefs) so that the entire dictionary
        can be safely dumped to a YAML file.

        Args:
            config: Configuration dictionary to fix.

        Returns:
            Fixed configuration dictionary.
        """
        # Replace the model class with its name for serialization purposes
        config["model"] = config["model"].__name__

        # Remove keys that contain non-serializable objects
        keys_to_remove = [
            "_debug",
            "tune_run_path",
            "_training_ref",
            "_validation_ref",
            "encoder_loader",  # if this key holds a non-serializable object
        ]
        for key in keys_to_remove:
            config.pop(key, None)

        return config

    def save_best_metrics_dataframe(self, output: str) -> None:
        """Save the dataframe with the metrics at each iteration of the best sample to a file.

        Args:
            output: CSV file path to save the metrics.
        """
        metrics_df: pd.DataFrame = pd.DataFrame([self.best_result.metrics])
        metrics_df.to_csv(output, index=False)

    def get_best_model(self) -> dict[str, torch.Tensor]:
        """Get the best model weights from the results.

        Returns:
            Dictionary of model weights.

        Raises:
            ValueError: If the checkpoint is missing.
        """
        if self.best_result.checkpoint is None:
            raise ValueError("Best result does not contain a checkpoint for the model.")
        checkpoint_dir: str = self.best_result.checkpoint.to_directory()
        checkpoint: str = os.path.join(checkpoint_dir, "model.safetensors")
        return safe_load_file(checkpoint)

    def save_best_model(self, output: str) -> None:
        """Save the best model weights to a file.

        This method retrieves the best model weights using the get_best_model helper
        which loads the model data from the checkpoint's directory, then re-saves
        it using safe_save_file.

        Args:
            output: Path where the best model weights will be saved.
        """
        model: dict[str, torch.Tensor] = self.get_best_model()
        safe_save_file(model, output)

    def get_best_optimizer(self) -> dict[str, Any]:
        """Get the best optimizer state from the results.

        Returns:
            Optimizer state dictionary.

        Raises:
            ValueError: If the checkpoint is missing.
        """
        if self.best_result.checkpoint is None:
            raise ValueError("Best result does not contain a checkpoint for the optimizer.")
        checkpoint_dir: str = self.best_result.checkpoint.to_directory()
        checkpoint: str = os.path.join(checkpoint_dir, "optimizer.pt")
        return torch.load(checkpoint)

    def save_best_optimizer(self, output: str) -> None:
        """Save the best optimizer state to a file.

        Args:
            output: Path where the best optimizer state will be saved.
        """
        optimizer_state: dict[str, Any] = self.get_best_optimizer()
        torch.save(optimizer_state, output)
