#!/usr/bin/env python3
"""CLI module for running raytune tuning experiment."""

import argparse
import logging

import yaml
from torch.utils.data import DataLoader

from stimulus.data import handlertorch, loaders
from stimulus.learner import raytune_learner, raytune_parser
from stimulus.utils import launch_utils, yaml_data, yaml_model_schema

logger = logging.getLogger(__name__)


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
        "--data_config",
        type=str,
        required=True,
        metavar="FILE",
        help="Path to data config file.",
    )
    parser.add_argument(
        "-c",
        "--model_config",
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
        help="The path to the initial weights (optional).",
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
        "-o",
        "--output",
        type=str,
        required=False,
        nargs="?",
        const="best_model.pt",
        default="best_model.pt",
        metavar="FILE",
        help="The output file path to write the trained model to",
    )

    parser.add_argument(
        "-bm",
        "--best_metrics",
        type=str,
        required=False,
        nargs="?",
        const="best_metrics.csv",
        default="best_metrics.csv",
        metavar="FILE",
        help="The path to write the best metrics to",
    )

    parser.add_argument(
        "-bc",
        "--best_config",
        type=str,
        required=False,
        nargs="?",
        const="best_config.yaml",
        default="best_config.yaml",
        metavar="FILE",
        help="The path to write the best config to",
    )

    parser.add_argument(
        "-bo",
        "--best_optimizer",
        type=str,
        required=False,
        nargs="?",
        const="best_optimizer.pt",
        default="best_optimizer.pt",
        metavar="FILE",
        help="The path to write the best optimizer to",
    )

    parser.add_argument(
        "--tune_run_name",
        type=str,
        required=False,
        nargs="?",
        const=None,
        default=None,
        metavar="CUSTOM_RUN_NAME",
        help="tells ray tune what that the 'experiment_name' aka the given tune_run name should be. This is controlled be the variable name in the RunConfig class of tune. This has two behaviuors: 1 if set the subdir of ray_results is going to be named with this value, 2 the subdir of the above mentioned will also have this value as prefix for the single train dir name. Default None, meaning ray will generate such a name on its own.",
    )

    parser.add_argument(
        "--debug_mode",
        action="store_true",
        help="Activate debug mode for tuning. Default false, no debug.",
    )

    return parser.parse_args()


def main(
    model_path: str,
    data_path: str,
    data_config_path: str,
    model_config_path: str,
    initial_weights: str | None = None,  # noqa: ARG001
    ray_results_dirpath: str | None = None,
    output_path: str | None = None,
    best_optimizer_path: str | None = None,
    best_metrics_path: str | None = None,
    best_config_path: str | None = None,
    *,
    debug_mode: bool = False,
) -> None:
    """Run the main model checking pipeline.

    Args:
        data_path: Path to input data file.
        model_path: Path to model file.
        data_config_path: Path to data config file.
        model_config_path: Path to model config file.
        initial_weights: Optional path to initial weights.
        ray_results_dirpath: Directory for ray results.
        debug_mode: Whether to run in debug mode.
        output_path: Path to write the best model to.
        best_optimizer_path: Path to write the best optimizer to.
        best_metrics_path: Path to write the best metrics to.
        best_config_path: Path to write the best config to.
    """
    with open(data_config_path) as file:
        data_config = yaml.safe_load(file)
        data_config = yaml_data.YamlSubConfigDict(**data_config)

    with open(model_config_path) as file:
        model_config = yaml.safe_load(file)
        model_config = yaml_model_schema.Model(**model_config)

    encoder_loader = loaders.EncoderLoader()
    encoder_loader.initialize_column_encoders_from_config(column_config=data_config.columns)

    model_class = launch_utils.import_class_from_file(model_path)

    ray_config_loader = yaml_model_schema.YamlRayConfigLoader(model=model_config)
    ray_config_model = ray_config_loader.get_config()

    tuner = raytune_learner.TuneWrapper(
        model_config=ray_config_model,
        data_config_path=data_config_path,
        model_class=model_class,
        data_path=data_path,
        encoder_loader=encoder_loader,
        seed=42,
        ray_results_dir=ray_results_dirpath,
        debug=debug_mode,
    )

    grid_results = tuner.tune()

    parser = raytune_parser.TuneParser(grid_results=grid_results)
    parser.save_best_model(output_path=output_path)
    parser.save_best_optimizer(output_path=best_optimizer_path)
    parser.save_best_metrics_dataframe(output_path=best_metrics_path)
    parser.save_best_config(output_path=best_config_path)

def run() -> None:
    """Run the model checking script."""
    args = get_args()
    main(
        data_path=args.data,
        model_path=args.model,
        data_config_path=args.data_config,
        model_config_path=args.model_config,
        initial_weights=args.initial_weights,
        num_samples=args.num_samples,
        ray_results_dirpath=args.ray_results_dirpath,
        debug_mode=args.debug_mode,
    )


if __name__ == "__main__":
    run()
