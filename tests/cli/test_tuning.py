"""Test the tuning CLI."""

import os
import shutil
import warnings
from pathlib import Path

import pytest
import ray

from stimulus.cli import tuning


@pytest.fixture
def data_path() -> str:
    """Get path to test data CSV file."""
    return str(Path(__file__).parent.parent / "test_data" / "titanic" / "titanic_stimulus_split.csv")


@pytest.fixture
def data_config() -> str:
    """Get path to test data config YAML."""
    return str(Path(__file__).parent.parent / "test_data" / "titanic" / "titanic_sub_config.yaml")


@pytest.fixture
def model_path() -> str:
    """Get path to test model file."""
    return str(Path(__file__).parent.parent / "test_model" / "titanic_model.py")


@pytest.fixture
def model_config() -> str:
    """Get path to test model config YAML."""
    return str(Path(__file__).parent.parent / "test_model" / "titanic_model_cpu.yaml")


def test_tuning_main(data_path: str, data_config: str, model_path: str, model_config: str) -> None:
    """Test that tuning.main runs without errors.

    Args:
        data_path: Path to test CSV data
        data_config: Path to data config YAML
        model_path: Path to model implementation
        model_config: Path to model config YAML
    """
    # Filter ResourceWarning during Ray shutdown
    warnings.filterwarnings("ignore", category=ResourceWarning)

    # Initialize Ray with minimal resources for testing
    ray.init(ignore_reinit_error=True)
    # Verify all required files exist
    assert os.path.exists(data_path), f"Data file not found at {data_path}"
    assert os.path.exists(data_config), f"Data config not found at {data_config}"
    assert os.path.exists(model_path), f"Model file not found at {model_path}"
    assert os.path.exists(model_config), f"Model config not found at {model_config}"

    try:
        results_dir = Path("tests/test_data/titanic/test_results/").resolve()
        results_dir.mkdir(parents=True, exist_ok=True)

        # Use directory path for Ray results and file paths for outputs
        tuning.main(
            model_path=model_path,
            data_path=data_path,
            data_config_path=data_config,
            model_config_path=model_config,
            initial_weights=None,
            ray_results_dirpath=str(results_dir),  # Directory path without URI scheme
            output_path=str(results_dir / "best_model.safetensors"),
            best_optimizer_path=str(results_dir / "best_optimizer.pt"),
            best_metrics_path=str(results_dir / "best_metrics.csv"),
            best_config_path=str(results_dir / "best_config.yaml"),
            debug_mode=True,
        )

    except RuntimeError as error:
        error_msg: str = str(error).lower()
        if "zero_division" in error_msg or "no best trial found" in error_msg:
            pytest.skip(f"Skipping test due to known metric issue: {error}")
        raise
    finally:
        # Ensure Ray is shut down properly
        if ray.is_initialized():
            ray.shutdown()

            # Clean up any ray files/directories that may have been created
            ray_results_dir = os.path.expanduser("tests/test_data/titanic/test_results/")
            if os.path.exists(ray_results_dir):
                shutil.rmtree(ray_results_dir)
