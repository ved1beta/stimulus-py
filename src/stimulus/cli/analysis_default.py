#!/usr/bin/env python3
"""Analysis default module for running model analysis and performance evaluation."""

import argparse
import json
import os
from typing import Any

from safetensors.torch import load_model as safe_load

from stimulus.analysis.analysis_default import AnalysisPerformanceTune, AnalysisRobustness
from stimulus.utils.launch_utils import get_experiment, import_class_from_file


def get_args() -> argparse.Namespace:
    """Get the arguments when using from the commandline.

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-m", "--model", type=str, required=True, metavar="FILE", help="The model .py file")
    parser.add_argument(
        "-w",
        "--weight",
        type=str,
        required=True,
        nargs="+",
        metavar="FILE",
        help="Model weights .pt file",
    )
    parser.add_argument(
        "-me",
        "--metrics",
        type=str,
        required=True,
        nargs="+",
        metavar="FILE",
        help="The file path for the metrics file obtained during tuning",
    )
    parser.add_argument(
        "-ec",
        "--experiment_config",
        type=str,
        required=True,
        nargs="+",
        metavar="FILE",
        help="The experiment config used to modify the data.",
    )
    parser.add_argument(
        "-mc",
        "--model_config",
        type=str,
        required=True,
        nargs="+",
        metavar="FILE",
        help="The tune config file.",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        required=True,
        nargs="+",
        metavar="FILE",
        help="List of data files to be used for the analysis.",
    )
    parser.add_argument("-o", "--outdir", type=str, required=True, help="output directory")

    return parser.parse_args()


def main(
    model_path: str,
    weight_list: list[str],
    mconfig_list: list[str],
    metrics_list: list[str],
    econfig_list: list[str],
    data_list: list[str],
    outdir: str,
) -> None:
    """Run the main analysis pipeline.

    Args:
        model_path: Path to model file
        weight_list: List of model weight paths
        mconfig_list: List of model config paths
        metrics_list: List of metric file paths
        econfig_list: List of experiment config paths
        data_list: List of data file paths
        outdir: Output directory path
    """
    metrics = ["rocauc", "prauc", "mcc", "f1score", "precision", "recall"]

    # Plot the performance during tuning/training
    run_analysis_performance_tune(
        metrics_list,
        [*metrics, "loss"],  # Use list unpacking instead of concatenation
        os.path.join(outdir, "performance_tune_train"),
    )

    # Run robustness analysis
    run_analysis_performance_model(
        metrics,
        model_path,
        weight_list,
        mconfig_list,
        econfig_list,
        data_list,
        os.path.join(outdir, "performance_robustness"),
    )


def run_analysis_performance_tune(metrics_list: list[str], metrics: list[str], outdir: str) -> None:
    """Run performance analysis during tuning/training.

    Each model has a metrics file obtained during tuning/training,
    check the performance there and plot it.
    This is to track the model performance per training iteration.

    Args:
        metrics_list: List of metric file paths
        metrics: List of metrics to analyze
        outdir: Output directory path
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for metrics_path in metrics_list:
        AnalysisPerformanceTune(metrics_path).plot_metric_vs_iteration(
            metrics=metrics,
            output=os.path.join(outdir, metrics_path.replace("-metrics.csv", "") + "-metric_vs_iteration.png"),
        )


def run_analysis_performance_model(
    metrics: list[str],
    model_path: str,
    weight_list: list[str],
    mconfig_list: list[str],
    econfig_list: list[str],
    data_list: list[str],
    outdir: str,
) -> None:
    """Run analysis to report model robustness.

    This block will compute the predictions of each model for each dataset.
    This information will be parsed and plots will be generated to report the model robustness.

    Args:
        metrics: List of metrics to analyze
        model_path: Path to model file
        weight_list: List of model weight paths
        mconfig_list: List of model config paths
        econfig_list: List of experiment config paths
        data_list: List of data file paths
        outdir: Output directory path
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Load all the models weights into a list
    model_names = []
    model_list = []
    model_class = import_class_from_file(model_path)
    for weight_path, mconfig_path in zip(weight_list, mconfig_list):
        model = load_model(model_class, weight_path, mconfig_path)
        model_names.append(mconfig_path.split("/")[-1].replace("-config.json", ""))
        model_list.append(model)

    # Read experiment config and initialize experiment class
    with open(econfig_list[0]) as in_json:
        experiment_name = json.load(in_json)["experiment"]
    initialized_experiment_class = get_experiment(experiment_name)

    # Initialize analysis
    analysis = AnalysisRobustness(metrics, initialized_experiment_class, batch_size=256)

    # Compute performance metrics
    df = analysis.get_performance_table(model_names, model_list, data_list)
    df.to_csv(os.path.join(outdir, "performance_table.csv"), index=False)

    # Get average performance
    tmp = analysis.get_average_performance_table(df)
    tmp.to_csv(os.path.join(outdir, "average_performance_table.csv"), index=False)

    # Plot heatmap
    analysis.plot_performance_heatmap(df, output=os.path.join(outdir, "performance_heatmap.png"))

    # Plot delta performance
    outdir2 = os.path.join(outdir, "delta_performance_vs_data")
    if not os.path.exists(outdir2):
        os.makedirs(outdir2)
    for metric in metrics:
        analysis.plot_delta_performance(
            metric,
            df,
            output=os.path.join(outdir2, f"delta_performance_{metric}.png"),
        )


def load_model(model_class: Any, weight_path: str, mconfig_path: str) -> Any:
    """Load the model with its config and weights.

    Args:
        model_class: Model class to instantiate
        weight_path: Path to model weights
        mconfig_path: Path to model config

    Returns:
        Loaded model instance
    """
    with open(mconfig_path) as in_json:
        mconfig = json.load(in_json)["model_params"]

    model = model_class(**mconfig)
    return safe_load(model, weight_path, strict=True)


def run() -> None:
    """Run the analysis script."""
    args = get_args()
    main(args.model, args.weight, args.model_config, args.metrics, args.experiment_config, args.data, args.outdir)


if __name__ == "__main__":
    run()
