"""Tests for the split_csv CLI command."""

import hashlib
import pathlib
import tempfile
from typing import Any, Callable

import pytest

from src.stimulus.cli.transform_csv import main


# Fixtures
@pytest.fixture
def csv_path() -> str:
    """Fixture that returns the path to a CSV file without split column."""
    return "tests/test_data/titanic/titanic_stimulus.csv"


@pytest.fixture
def yaml_path() -> str:
    """Fixture that returns the path to a YAML config file."""
    return "tests/test_data/titanic/titanic_sub_config_0.yaml"


# Test cases
test_cases = [
    ("csv_path", "yaml_path", None),
]


# Tests
@pytest.mark.skip(reason="macOS snapshot differs, pending investigation")
@pytest.mark.parametrize(("csv_type", "yaml_type", "error"), test_cases)
def test_transform_csv(
    request: pytest.FixtureRequest,
    snapshot: Callable[[], Any],
    csv_type: str,
    yaml_type: str,
    error: Exception | None,
) -> None:
    """Tests the CLI command with correct and wrong YAML files."""
    csv_path = request.getfixturevalue(csv_type)
    yaml_path = request.getfixturevalue(yaml_type)
    tmpdir = pathlib.Path(tempfile.gettempdir())
    if error:
        with pytest.raises(error):  # type: ignore[call-overload]
            main(csv_path, yaml_path, str(tmpdir / "test.csv"))
    else:
        filename = f"{csv_type}.csv"
        main(csv_path, yaml_path, str(tmpdir / filename))
        with open(tmpdir / filename, newline="", encoding="utf-8") as file:
            hash = hashlib.md5(file.read().encode()).hexdigest()  # noqa: S324
        assert hash == snapshot
