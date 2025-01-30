"""This package provides splitter classes for splitting data into train, validation, and test sets."""

from stimulus.data.splitters.splitters import AbstractSplitter, RandomSplit

__all__ = ["AbstractSplitter", "RandomSplit"]
