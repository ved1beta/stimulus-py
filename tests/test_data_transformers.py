"""Test suite for the data transformation generators."""

import numpy as np
import pytest

from src.stimulus.data.transform.data_transformation_generators import (
    GaussianChunk,
    GaussianNoise,
    ReverseComplement,
    UniformTextMasker,
)


class TestUniformTestMasker:
    """Test suite for the UniformTextMasker class."""
    def test_transform_single(self) -> None:
        """Test masking a single string."""
        transformer = UniformTextMasker(mask="N")
        params = {"seed": 42, "probability": 0.1}
        single_input = "ACGTACGT"
        expected_output = "ACGTACNT"
        transformed_data = transformer.transform(single_input, **params)
        assert isinstance(transformed_data, str)
        assert transformed_data == expected_output

    def test_transform_multiple(self) -> None:
        """Test masking multiple strings."""
        transformer = UniformTextMasker(mask="N")
        params = {"seed": 42, "probability": 0.1}
        multiple_inputs = ["ATCGATCGATCG", "ATCG"]
        expected_outputs = ["ATCGATNGATNG", "ATCG"]
        transformed_data = transformer.transform_all(multiple_inputs, **params)
        assert isinstance(transformed_data, list)
        for item in transformed_data:
            assert isinstance(item, str)
        assert transformed_data == expected_outputs


class TestGaussianNoise:
    """Test suite for the GaussianNoise class."""
    def test_transform_single(self) -> None:
        """Test transforming a single float."""
        transformer = GaussianNoise()
        params = {"seed": 42, "mean": 0, "std": 1}
        single_input = 5.0
        expected_output = 5.4967141530112327
        transformed_data = transformer.transform(single_input, **params)
        assert isinstance(transformed_data, float)
        assert round(transformed_data, 7) == round(expected_output, 7)

    def test_transform_multiple(self) -> None:
        """Test transforming multiple floats."""
        transformer = GaussianNoise()
        params = {"seed": 42, "mean": 0, "std": 1}
        multiple_inputs = [1.0, 2.0, 3.0]
        expected_outputs = [1.764052345967664, 0.4001572083672233, 0.9787379841057392]
        transformed_data = transformer.transform_all(multiple_inputs, **params)
        assert isinstance(transformed_data, np.ndarray)
        for item in transformed_data:
            assert isinstance(item, float)
        assert len(transformed_data) == len(expected_outputs)
        for item, expected in zip(transformed_data, expected_outputs):
            assert round(item, 7) == round(expected, 7)


class TestGaussianChunk:
    """Test suite for the GaussianChunk class."""
    def test_transform_single(self) -> None:
        """Test transforming a single string."""
        transformer = GaussianChunk()
        params = {"seed": 42, "chunk_size": 10, "std": 1}
        single_input = "AGCATGCTAGCTAGATCAAAATCGATGCATGCTAGCGGCGCGCATGCATGAGGAGACTGAC"
        expected_output = "TGCATGCTAG"
        transformed_data = transformer.transform(single_input, **params)
        assert isinstance(transformed_data, str)
        assert len(transformed_data) == 10
        assert transformed_data == expected_output

    def test_transform_multiple(self) -> None:
        """Test transforming multiple strings."""
        transformer = GaussianChunk()
        params = {"seed": 42, "chunk_size": 10, "std": 1}
        multiple_inputs = [
            "AGCATGCTAGCTAGATCAAAATCGATGCATGCTAGCGGCGCGCATGCATGAGGAGACTGAC",
            "AGCATGCTAGCTAGATCAAAATCGATGCATGCTAGCGGCGCGCATGCATGAGGAGACTGAC",
        ]
        expected_outputs = ["TGCATGCTAG", "GCATGCTAGC"]
        transformed_data = transformer.transform_all(multiple_inputs, **params)
        assert isinstance(transformed_data, list)
        for item in transformed_data:
            assert isinstance(item, str)
            assert len(item) == 10
        assert transformed_data == expected_outputs


class TestReverseComplement:
    """Test suite for the ReverseComplement class."""
    def test_transform_single(self) -> None:
        """Test transforming a single string."""
        transformer = ReverseComplement()
        single_input = "ACCCCTACGTNN"
        expected_output = "NNACGTAGGGGT"
        transformed_data = transformer.transform(single_input)
        assert isinstance(transformed_data, str)
        assert transformed_data == expected_output

    def test_transform_multiple(self) -> None:
        """Test transforming multiple strings."""
        transformer = ReverseComplement()
        multiple_inputs = ["ACCCCTACGTNN", "ACTGA"]
        expected_outputs = ["NNACGTAGGGGT", "TCAGT"]
        transformed_data = transformer.transform_all(multiple_inputs)
        assert isinstance(transformed_data, list)
        for item in transformed_data:
            assert isinstance(item, str)
        assert transformed_data == expected_outputs
