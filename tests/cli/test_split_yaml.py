import tempfile

import pytest

from src.stimulus.cli.split_yaml import main


# Fixtures
@pytest.fixture
def correct_yaml_path() -> str:
    """Fixture that returns the path to a correct YAML file."""
    return "tests/test_data/titanic/titanic.yaml"


@pytest.fixture
def wrong_yaml_path() -> str:
    """Fixture that returns the path to a wrong YAML file."""
    return "tests/test_data/yaml_files/wrong_field_type.yaml"


# Test cases
test_cases = [
    ("correct_yaml_path", None),
    ("wrong_yaml_path", ValueError),
]


# Tests
@pytest.mark.parametrize("yaml_type, error", test_cases)
def test_split_yaml(request: pytest.FixtureRequest, yaml_type: str, error: Exception | None) -> None:
    """Tests the CLI command with correct and wrong YAML files."""
    yaml_path = request.getfixturevalue(yaml_type)
    tmpdir = tempfile.gettempdir()
    if error:
        with pytest.raises(error):
            main(yaml_path, tmpdir)
    else:
        assert main(yaml_path, tmpdir) is None
