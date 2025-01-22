"""Tests for the split_csv CLI command."""

import hashlib
import pathlib
import tempfile

import pytest

from src.stimulus.cli.split_csv import main


# Fixtures
@pytest.fixture
def csv_path_no_split() -> str:
    """Fixture that returns the path to a CSV file without split column."""
    return "tests/test_data/dna_experiment/test.csv"


@pytest.fixture
def csv_path_with_split() -> str:
    """Fixture that returns the path to a CSV file with split column."""
    return "tests/test_data/dna_experiment/test_with_split.csv"


@pytest.fixture
def yaml_path() -> str:
    """Fixture that returns the path to a YAML config file."""
    return "tests/test_data/dna_experiment/dna_experiment_config_template_0.yaml"


# Test cases
test_cases = [
    ("csv_path_no_split", "yaml_path", False, None),
    ("csv_path_with_split", "yaml_path", False, ValueError),
    ("csv_path_no_split", "yaml_path", True, None),
    ("csv_path_with_split", "yaml_path", True, None),
]


# Tests
@pytest.mark.skip(reason="There is an issue with non-deterministic output")
@pytest.mark.parametrize(("csv_type", "yaml_type", "force", "error"), test_cases)
def test_split_csv(
    request: pytest.FixtureRequest,
    snapshot: pytest.fixture,
    csv_type: str,
    yaml_type: str,
    force: bool,
    error: Exception | None,
) -> None:
    """Tests the CLI command with correct and wrong YAML files."""
    csv_path = request.getfixturevalue(csv_type)
    yaml_path = request.getfixturevalue(yaml_type)
    tmpdir = pathlib.Path(tempfile.gettempdir())
    if error:
        with pytest.raises(error):
            main(csv_path, yaml_path, tmpdir / "test.csv", force=force, seed=42)
    else:
        filename = f"{csv_type}_{force}.csv"
        assert main(csv_path, yaml_path, tmpdir / filename, force=force, seed=42) is None
        with open(tmpdir / filename) as file:
            hash = hashlib.md5(file.read().encode()).hexdigest()  # noqa: S324
        assert hash == snapshot
