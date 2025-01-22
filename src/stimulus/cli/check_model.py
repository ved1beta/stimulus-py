#!/usr/bin/env python3
"""CLI module for checking model configuration and running initial tests."""

import argparse
import json
import logging
import os

import yaml

from stimulus.data.csv import CsvProcessing
from stimulus.learner.raytune_learner import TuneWrapper as StimulusTuneWrapper
from stimulus.utils.json_schema import JsonSchema
from stimulus.utils.launch_utils import get_experiment, import_class_from_file, memory_split_for_ray_init


def get_args() -> argparse.Namespace:
    """Get the arguments when using from the commandline.

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Launch check_model.")
    parser.add_argument("-d", "--data", type=str, required=True, metavar="FILE", help="Path to input csv file.")
    parser.add_argument("-m", "--model", type=str, required=True, metavar="FILE", help="Path to model file.")
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        required=True,
        metavar="FILE",
        help="Experiment config file. From this the experiment class name is extracted.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        metavar="FILE",
        help="Path to yaml config training file.",
    )
    parser.add_argument(
        "-w",
        "--initial_weights",
        type=str,
        required=False,
        nargs="?",
        const=None,
        default=None,
        metavar="FILE",
        help="The path to the initial weights. These can be used by the model instead of the random initialization.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        required=False,
        nargs="?",
        const=None,
        default=None,
        metavar="NUM_OF_MAX_GPU",
        help="Use to limit the number of GPUs ray can use. This might be useful on many occasions, especially in a cluster system.",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        required=False,
        nargs="?",
        const=None,
        default=None,
        metavar="NUM_OF_MAX_CPU",
        help="Use to limit the number of CPUs ray can use. This might be useful on many occasions, especially in a cluster system.",
    )
    parser.add_argument(
        "--memory",
        type=str,
        required=False,
        nargs="?",
        const=None,
        default=None,
        metavar="MAX_MEMORY",
        help="Ray can have a limiter on the total memory it can use. This might be useful on many occasions, especially in a cluster system.",
    )
    parser.add_argument(
        "-n",
        "--num_samples",
        type=int,
        required=False,
        nargs="?",
        const=3,
        default=3,
        metavar="NUM_SAMPLES",
        help="Number of samples for tuning. Overwrites tune.tune_params.num_samples in config.",
    )
    parser.add_argument(
        "--ray_results_dirpath",
        type=str,
        required=False,
        nargs="?",
        const=None,
        default=None,
        metavar="DIR_PATH",
        help="Location where ray_results output dir should be written. If None, uses ~/ray_results.",
    )
    parser.add_argument(
        "--debug_mode",
        action="store_true",
        help="Activate debug mode for tuning. Default false, no debug.",
    )

    return parser.parse_args()


def main(
    data_path: str,
    model_path: str,
    experiment_config: str,
    config_path: str,
    initial_weights_path: str | None = None,
    gpus: int | None = None,
    cpus: int | None = None,
    memory: str | None = None,
    num_samples: int = 3,
    ray_results_dirpath: str | None = None,
    *,
    debug_mode: bool = False,
) -> None:
    """Run the main model checking pipeline.

    Args:
        data_path: Path to input data file.
        model_path: Path to model file.
        experiment_config: Path to experiment config.
        config_path: Path to training config.
        initial_weights_path: Optional path to initial weights.
        gpus: Maximum number of GPUs to use.
        cpus: Maximum number of CPUs to use.
        memory: Maximum memory to use.
        num_samples: Number of samples for tuning.
        ray_results_dirpath: Directory for ray results.
        debug_mode: Whether to run in debug mode.
    """
    # Load experiment config
    with open(experiment_config) as in_json:
        exp_config = json.load(in_json)

    # Initialize json schema and experiment class
    schema = JsonSchema(exp_config)
    initialized_experiment_class = get_experiment(schema.experiment)
    model_class = import_class_from_file(model_path)

    # Update tune config
    updated_tune_conf = "check_model_modified_tune_config.yaml"
    with open(config_path) as conf_file, open(updated_tune_conf, "w") as new_conf:
        user_tune_config = yaml.safe_load(conf_file)
        user_tune_config["tune"]["tune_params"]["num_samples"] = num_samples

        if user_tune_config["tune"]["scheduler"]["name"] == "ASHAScheduler":
            user_tune_config["tune"]["scheduler"]["params"]["max_t"] = 1
            user_tune_config["tune"]["scheduler"]["params"]["grace_period"] = 1
            user_tune_config["tune"]["step_size"] = 1
        elif user_tune_config["tune"]["scheduler"]["name"] == "FIFOScheduler":
            user_tune_config["tune"]["run_params"]["stop"]["training_iteration"] = 1

        if initial_weights_path is not None:
            user_tune_config["model_params"]["initial_weights"] = os.path.abspath(initial_weights_path)

        yaml.dump(user_tune_config, new_conf)

    # Process CSV data
    csv_obj = CsvProcessing(initialized_experiment_class, data_path)
    downsampled_csv = "downsampled.csv"

    if "split" not in csv_obj.check_and_get_categories():
        config_default = {"name": "RandomSplitter", "params": {"split": [0.5, 0.5, 0.0]}}
        csv_obj.add_split(config_default)

    csv_obj.save(downsampled_csv)

    # Initialize ray
    object_store_mem, mem = memory_split_for_ray_init(memory)
    ray_results_dirpath = None if ray_results_dirpath is None else os.path.abspath(ray_results_dirpath)

    # Create and run learner
    learner = StimulusTuneWrapper(
        updated_tune_conf,
        model_class,
        downsampled_csv,
        initialized_experiment_class,
        max_gpus=gpus,
        max_cpus=cpus,
        max_object_store_mem=object_store_mem,
        max_mem=mem,
        ray_results_dir=ray_results_dirpath,
        _debug=debug_mode,
    )

    grid_results = learner.tune()

    # Check results
    logger = logging.getLogger(__name__)
    for i, result in enumerate(grid_results):
        if not result.error:
            logger.info("Trial %d finished successfully with metrics %s.", i, result.metrics)
        else:
            raise TypeError(f"Trial {i} failed with error {result.error}.")


def run() -> None:
    """Run the model checking script."""
    args = get_args()
    main(
        args.data,
        args.model,
        args.experiment,
        args.config,
        args.initial_weights,
        args.gpus,
        args.cpus,
        args.memory,
        args.num_samples,
        args.ray_results_dirpath,
        debug_mode=args.debug_mode,
    )


if __name__ == "__main__":
    run()
