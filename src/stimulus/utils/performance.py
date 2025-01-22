"""Utility module for computing various performance metrics for machine learning models."""

from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Constants for threshold and number of classes
BINARY_THRESHOLD = 0.5
BINARY_CLASS_COUNT = 2


class Performance:
    """Returns the value of a given metric.

    Parameters
    ----------
    labels (np.array) : labels
    predictions (np.array) : predictions
    metric (str) : the metric to compute

    Returns:
    -------
    value (float) : the value of the metric

    TODO we can add more metrics here

    TODO currently for classification  metrics like precision, recall, f1score and mcc,
    we are using a threshold of 0.5 to convert the probabilities to binary predictions.
    However for models with imbalanced predictions, where the meaningful threshold is not
    located at 0.5, one can end up with full of 0s or 1s, and thus meaningless performance
    metrics.
    """

    def __init__(self, labels: Any, predictions: Any, metric: str = "rocauc") -> float:
        """Initialize Performance class with labels, predictions and metric type.

        Args:
            labels: Ground truth labels
            predictions: Model predictions
            metric: Type of metric to compute (default: "rocauc")
        """
        labels = self.data2array(labels)
        predictions = self.data2array(predictions)
        labels, predictions = self.handle_multiclass(labels, predictions)
        if labels.shape != predictions.shape:
            raise ValueError(
                f"The labels have shape {labels.shape} whereas predictions have shape {predictions.shape}.",
            )
        function = getattr(self, metric)
        self.val = function(labels, predictions)

    def data2array(self, data: Any) -> np.array:
        """Convert input data to numpy array.

        Args:
            data: Input data in various formats

        Returns:
            np.array: Converted numpy array

        Raises:
            ValueError: If input data type is not supported
        """
        if isinstance(data, list):
            return np.array(data)
        if isinstance(data, np.ndarray):
            return data
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        if isinstance(data, (int, float)):
            return np.array([data])
        raise ValueError(f"The data must be a list, np.array, torch.Tensor, int or float. Instead it is {type(data)}")

    def handle_multiclass(self, labels: np.array, predictions: np.array) -> tuple[np.array, np.array]:
        """Handle the case of multiclass classification.

        TODO currently only two class predictions are handled. Needs to handle the other scenarios.
        """
        # if only one columns for labels and predictions
        if (len(labels.shape) == 1) and (len(predictions.shape) == 1):
            return labels, predictions

        # if one columns for labels, but two columns for predictions
        if (len(labels.shape) == 1) and (predictions.shape[1] == BINARY_CLASS_COUNT):
            predictions = predictions[:, 1]  # assumes the second column is the positive class
            return labels, predictions

        # other scenarios not implemented yet
        raise ValueError(f"Labels have shape {labels.shape} and predictions have shape {predictions.shape}.")

    def rocauc(self, labels: np.array, predictions: np.array) -> float:
        """Compute ROC AUC score."""
        return roc_auc_score(labels, predictions)

    def prauc(self, labels: np.array, predictions: np.array) -> float:
        """Compute PR AUC score."""
        return average_precision_score(labels, predictions)

    def mcc(self, labels: np.array, predictions: np.array) -> float:
        """Compute Matthews Correlation Coefficient."""
        predictions = np.array([1 if p > BINARY_THRESHOLD else 0 for p in predictions])
        return matthews_corrcoef(labels, predictions)

    def f1score(self, labels: np.array, predictions: np.array) -> float:
        """Compute F1 score."""
        predictions = np.array([1 if p > BINARY_THRESHOLD else 0 for p in predictions])
        return f1_score(labels, predictions)

    def precision(self, labels: np.array, predictions: np.array) -> float:
        """Compute precision score."""
        predictions = np.array([1 if p > BINARY_THRESHOLD else 0 for p in predictions])
        return precision_score(labels, predictions)

    def recall(self, labels: np.array, predictions: np.array) -> float:
        """Compute recall score."""
        predictions = np.array([1 if p > BINARY_THRESHOLD else 0 for p in predictions])
        return recall_score(labels, predictions)

    def spearmanr(self, labels: np.array, predictions: np.array) -> float:
        """Compute Spearman correlation coefficient."""
        return spearmanr(labels, predictions)[0]
