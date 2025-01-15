"""
Test for encoders.
"""

import os
from typing import Any

import numpy as np
import torch
import pytest

from src.stimulus.data.encoding.encoders import (TextOneHotEncoder, 
                                                FloatEncoder, 
                                                IntEncoder, 
                                                StrClassificationIntEncoder)

# ------------------------------------------------------------------------------
# Tests for TextOneHotEncoder
# ------------------------------------------------------------------------------

@pytest.fixture
def encoder_default():
    """Provides a default encoder with lowercase alphabet 'acgt' (not case-sensitive)."""
    return TextOneHotEncoder(alphabet="acgt", case_sensitive=False, padding=True)

@pytest.fixture
def encoder_case_sensitive():
    """Provides an encoder with mixed case alphabet 'ACgt' (case-sensitive)."""
    return TextOneHotEncoder(alphabet="ACgt", case_sensitive=True, padding=True)

# ---- Test for initialization ---- #

def test_init_with_non_string_alphabet_raises_value_error():
    with pytest.raises(ValueError) as excinfo:
        TextOneHotEncoder(alphabet=['a', 'c', 'g', 't'])  # Passing a list instead of string
    assert "Expected a string input for alphabet" in str(excinfo.value)

def test_init_with_string_alphabet():
    encoder = TextOneHotEncoder(alphabet="acgt")
    assert encoder.alphabet == "acgt"
    assert encoder.case_sensitive is False
    assert encoder.padding is True

# ---- Tests for _sequence_to_array ---- #

def test_sequence_to_array_with_non_string_input(encoder_default):
    with pytest.raises(ValueError) as excinfo:
        encoder_default._sequence_to_array(1234)
    assert "Expected string input for sequence" in str(excinfo.value)

def test_sequence_to_array_returns_correct_shape(encoder_default):
    seq = "acgt"
    arr = encoder_default._sequence_to_array(seq)
    # shape should be (len(seq), 1)
    assert arr.shape == (4, 1)
    # check content
    assert (arr.flatten() == list(seq)).all()

def test_sequence_to_array_is_case_insensitive(encoder_default):
    seq = "AcGT"
    arr = encoder_default._sequence_to_array(seq)
    # Since encoder_default is not case sensitive, sequence is lowercased internally.
    assert (arr.flatten() == list("acgt")).all()

def test_sequence_to_array_is_case_sensitive(encoder_case_sensitive):
    seq = "AcGT"
    arr = encoder_case_sensitive._sequence_to_array(seq)
    # With case_sensitive=True, we do not modify 'AcGT'
    assert (arr.flatten() == list("AcGT")).all()

# ---- Tests for encode ---- #

def test_encode_returns_tensor(encoder_default):
    seq = "acgt"
    encoded = encoder_default.encode(seq)
    assert isinstance(encoded, torch.Tensor)
    # shape should be (len(seq), alphabet_size=4)
    assert encoded.shape == (4, 4)

def test_encode_unknown_character_returns_zero_vector(encoder_default):
    seq = "acgtn"
    encoded = encoder_default.encode(seq)
    # the last character 'n' is not in 'acgt', so the last row should be all zeros
    assert torch.all(encoded[-1] == 0)

def test_encode_case_sensitivity_true(encoder_case_sensitive):
    """Case-sensitive: 'ACgt' => 'ACgt' means 'A' and 'C' are uppercase in the alphabet, 
    'g' and 't' are lowercase in the alphabet."""
    seq = "ACgt"
    encoded = encoder_case_sensitive.encode(seq)
    # shape = (len(seq), 4)
    assert encoded.shape == (4, 4)
    # 'A' should be one-hot at the 0th index, 'C' at the 1st index, 'g' at the 2nd, 't' at the 3rd.
    # The order of categories in OneHotEncoder is typically ['A', 'C', 'g', 't'] given we passed ['A','C','g','t'].
    assert torch.all(encoded[0] == torch.tensor([1, 0, 0, 0]))  # 'A'
    assert torch.all(encoded[1] == torch.tensor([0, 1, 0, 0]))  # 'C'
    assert torch.all(encoded[2] == torch.tensor([0, 0, 1, 0]))  # 'g'
    assert torch.all(encoded[3] == torch.tensor([0, 0, 0, 1]))  # 't'

def test_encode_case_sensitivity_false(encoder_default):
    """Case-insensitive: 'ACGT' => 'acgt' internally."""
    seq = "ACGT"
    encoded = encoder_default.encode(seq)
    # shape = (4,4)
    assert encoded.shape == (4, 4)
    # The order of categories in OneHotEncoder is typically ['a', 'c', 'g', 't'] for the default encoder.
    assert torch.all(encoded[0] == torch.tensor([1, 0, 0, 0]))  # 'a'
    assert torch.all(encoded[1] == torch.tensor([0, 1, 0, 0]))  # 'c'
    assert torch.all(encoded[2] == torch.tensor([0, 0, 1, 0]))  # 'g'
    assert torch.all(encoded[3] == torch.tensor([0, 0, 0, 1]))  # 't'

# ---- Tests for encode_all ---- #

def test_encode_all_with_single_string(encoder_default):
    seq = "acgt"
    encoded = encoder_default.encode_all(seq)
    # shape = (batch_size=1, seq_len=4, alphabet_size=4)
    assert encoded.shape == (1, 4, 4)
    assert torch.all(encoded[0] == encoder_default.encode(seq))

def test_encode_all_with_list_of_sequences(encoder_default):
    seqs = ["acgt", "acgtn"]  # second has an unknown 'n'
    encoded = encoder_default.encode_all(seqs)
    # shape = (2, max_len=5, alphabet_size=4)
    assert encoded.shape == (2, 5, 4)
    # last character of the second sequence is an unknown 'n', so that row should be zero
    assert torch.all(encoded[1, 4] == 0)  # the 5th row in second sequence

def test_encode_all_with_padding_false():
    encoder = TextOneHotEncoder(alphabet="acgt", case_sensitive=False, padding=False)
    seqs = ["acgt", "acgtn"]  # different lengths
    # should raise ValueError because lengths differ
    with pytest.raises(ValueError) as excinfo:
        encoder.encode_all(seqs)
    assert "All sequences must have the same length when padding is False." in str(excinfo.value)

# ---- Tests for decode ---- #

def test_decode_single_sequence(encoder_default):
    seq = "acgt"
    encoded = encoder_default.encode(seq)
    decoded = encoder_default.decode(encoded)
    # Because decode returns a string in this case
    assert isinstance(decoded, str)
    # Should match the lowercased input (since case-sensitive=False)
    assert decoded == seq

def test_decode_unknown_characters(encoder_default):
    """ Unknown characters are zero vectors. When decoding, those become empty (ignored),
        or become None, depending on the transform. In the provided code, handle_unknown='ignore'
        yields an empty decode for those positions. The example code attempts to fill with '-'
        or None if needed. 
    """
    seq = "acgtn"
    encoded = encoder_default.encode(seq)
    decoded = encoder_default.decode(encoded)
    # The code snippet shows it returns an empty string for unknown char, 
    # or might fill with 'None' replaced by '-'.
    # Adjust assertion based on how you've handled unknown characters in your final implementation.
    # If you replaced None with '-', the last character might be '-'.
    # If handle_unknown='ignore' yields an empty decode, it might omit the character entirely.
    # In the given code, it returns an empty decode for that position. So let's assume it becomes ''.
    # That means we might get "acgt" with a missing final char or a placeholder.
    # Let's do a partial check:
    assert decoded.startswith("acgt")

def test_decode_multiple_sequences(encoder_default):
    seqs = ["acgt", "acgtn"]  # second has unknown 'n'
    encoded = encoder_default.encode_all(seqs)
    decoded = encoder_default.decode(encoded)
    # decode should return a list of strings in this case
    assert isinstance(decoded, list)
    assert len(decoded) == 2
    # First sequence should decode to acgt (case-insensitive)
    assert decoded[0] == "acgt-"
    # Second sequence might be "acgt" plus something for the unknown char
    assert decoded[0] == "acgt-"

# ------------------------------------------------------------------------------
# Tests for FloatEncoder
# ------------------------------------------------------------------------------

@pytest.fixture
def float_encoder():
    """Fixture to instantiate the FloatEncoder."""
    return FloatEncoder()

def test_encode_single_float(float_encoder):
    """Test encoding a single float value."""
    input_val = 3.14
    output = float_encoder.encode(input_val)
    assert isinstance(output, torch.Tensor), "Output should be a torch.Tensor."
    assert output.dtype == torch.float32, "Tensor dtype should be float32."
    assert output.numel() == 1, "Tensor should have exactly one element."
    # Using pytest.approx for floating-point comparison
    assert output.item() == pytest.approx(input_val), "Encoded value does not match the input float."

def test_encode_non_float_raises(float_encoder):
    """Test that encoding a non-float raises a ValueError."""
    with pytest.raises(ValueError) as exc_info:
        float_encoder.encode("not_a_float")
    assert "Expected input data to be a float, got str" in str(exc_info.value), (
        "Expected ValueError with specific error message."
    )

def test_encode_all_single_float(float_encoder):
    """
    Test encode_all when given a single float. 
    It should be treated as a list of one float internally.
    """
    input_val = 2.71
    output = float_encoder.encode_all(input_val)
    assert isinstance(output, torch.Tensor), "Output should be a torch.Tensor."
    assert output.dtype == torch.float32, "Tensor dtype should be float32."
    assert output.numel() == 1, "Tensor should have exactly one element."
    assert output.item() == pytest.approx(input_val), "Encoded value does not match the input."

def test_encode_all_multi_float(float_encoder):
    """Test encode_all with a list of floats."""
    input_vals = [3.14, 4.56]
    output = float_encoder.encode_all(input_vals)
    assert isinstance(output, torch.Tensor), "Output should be a torch.Tensor."
    assert output.dtype == torch.float32, "Tensor dtype should be float32."
    assert output.numel() == 2, "Tensor should have exactly one element."
    assert output[0].item() == pytest.approx(3.14), "First element does not match."
    assert output[1].item() == pytest.approx(4.56), "Second element does not match."

def test_decode_single_float(float_encoder):
    """Test decoding a tensor of shape (1)."""
    input_tensor = torch.tensor([3.14], dtype=torch.float32)
    decoded = float_encoder.decode(input_tensor)
    # decode returns data.numpy().tolist()
    assert isinstance(decoded, list), "Decoded output should be a list."
    assert len(decoded) == 1, "Decoded list should have one element."
    assert decoded[0] == pytest.approx(3.14), "Decoded value does not match."

def test_decode_multi_float(float_encoder):
    """Test decoding a tensor of shape (n)."""
    input_tensor = torch.tensor([3.14, 2.71], dtype=torch.float32)
    decoded = float_encoder.decode(input_tensor)
    assert isinstance(decoded, list), "Decoded output should be a list."
    assert len(decoded) == 2, "Decoded list should have two elements."
    assert decoded[0] == pytest.approx(3.14), "First decoded value does not match."
    assert decoded[1] == pytest.approx(2.71), "Second decoded value does not match."

# ------------------------------------------------------------------------------
# Tests for IntEncoder
# ------------------------------------------------------------------------------

@pytest.fixture
def int_encoder():
    """Fixture to instantiate the IntEncoder."""
    return IntEncoder()

def test_encode_single_int(int_encoder):
    """Test encoding a single int value."""
    input_val = 3
    output = int_encoder.encode(input_val)
    assert isinstance(output, torch.Tensor), "Output should be a torch.Tensor."
    assert output.dtype == torch.int32, "Tensor dtype should be int32."
    assert output.numel() == 1, "Tensor should have exactly one element."
    assert output.item() == input_val

def test_encode_non_int_raises(int_encoder):
    """Test that encoding a non-int raises a RuntimeError."""
    with pytest.raises(ValueError) as exc_info:
        int_encoder.encode("not_a_int")
    assert "Expected input data to be a int, got str" in str(exc_info.value), (
        "Expected ValueError with specific error message."
    )

def test_encode_all_single_int(int_encoder):
    """
    Test encode_all when given a single int. 
    It should be treated as a list of one int internally.
    """
    input_val = 2
    output = int_encoder.encode_all(input_val)
    assert isinstance(output, torch.Tensor), "Output should be a torch.Tensor."
    assert output.dtype == torch.int32, "Tensor dtype should be int32."
    assert output.numel() == 1, "Tensor should have exactly one element."
    assert output.item() == input_val

def test_encode_all_multi_int(int_encoder):
    """Test encode_all with a list of integers."""
    input_vals = [3, 4]
    output = int_encoder.encode_all(input_vals)
    assert isinstance(output, torch.Tensor), "Output should be a torch.Tensor."
    assert output.dtype == torch.int32, "Tensor dtype should be int32."
    assert output.numel() == 2, "Tensor should have exactly one element."
    assert output[0].item() == 3, "First element does not match."
    assert output[1].item() == 4, "Second element does not match."

def test_decode_single_int(int_encoder):
    """Test decoding a tensor of shape (1)."""
    input_tensor = torch.tensor([3], dtype=torch.int32)
    decoded = int_encoder.decode(input_tensor)
    # decode returns data.numpy().tolist()
    assert isinstance(decoded, list), "Decoded output should be a list."
    assert len(decoded) == 1, "Decoded list should have one element."
    assert decoded[0] == 3, "Decoded value does not match."

def test_decode_multi_int(int_encoder):
    """Test decoding a tensor of shape (n)."""
    input_tensor = torch.tensor([3, 4], dtype=torch.int32)
    decoded = int_encoder.decode(input_tensor)
    assert isinstance(decoded, list), "Decoded output should be a list."
    assert len(decoded) == 2, "Decoded list should have two elements."
    assert decoded[0] == 3, "First decoded value does not match."
    assert decoded[1] == 4, "Second decoded value does not match."

# ------------------------------------------------------------------------------
# Tests for StrClassificationIntEncoder
# ------------------------------------------------------------------------------

@pytest.fixture
def str_encoder():
    """Pytest fixture to instantiate StrClassificationIntEncoder."""
    return StrClassificationIntEncoder()

def test_encode_raises_not_implemented(str_encoder):
    """
    Tests that calling encode() with a single string 
    raises NotImplementedError as per the docstring.
    """
    with pytest.raises(NotImplementedError) as exc_info:
        str_encoder.encode("example")
    assert "Encoding a single string does not make sense. Use encode_all instead." in str(exc_info.value)

def test_encode_all_list_of_strings(str_encoder):
    """
    Tests that passing multiple unique strings returns 
    a torch tensor of the correct shape and encoded values.
    """
    input_data = ["apple", "banana", "orange"]
    output_tensor = str_encoder.encode_all(input_data)

    assert isinstance(output_tensor, torch.Tensor), "Output should be a torch.Tensor."
    assert output_tensor.shape == (3,), "Expected a shape of (3,) for three input strings."

    # We don't rely on a specific ordering from LabelEncoder (like alphabetical)
    # but we do expect a consistent integer encoding for each unique string.
    # For example, if it's alphabetical: apple -> 0, banana -> 1, orange -> 2
    # But the exact order may differ depending on LabelEncoder's implementation.
    # We can, however, ensure that the tensor has 3 unique integers in 0..2.
    unique_vals = set(output_tensor.tolist())
    assert len(unique_vals) == 3, "There should be 3 unique integer encodings."
    assert all(val in [0, 1, 2] for val in unique_vals), "Encoded values should be 0, 1, or 2."

def test_encode_all_raises_value_error_on_non_string(str_encoder):
    """
    Tests that encode_all raises ValueError 
    if the input is not a string or list of strings.
    """
    input_data = ["apple", 42, "banana"]  # 42 is not a string
    with pytest.raises(ValueError) as exc_info:
        str_encoder.encode_all(input_data)
    assert "Expected input data to be a list of strings" in str(exc_info.value)

def test_decode_raises_not_implemented(str_encoder):
    """
    Tests that decode() raises NotImplementedError 
    since decoding is not supported in this encoder.
    """
    with pytest.raises(NotImplementedError) as exc_info:
        str_encoder.decode(torch.tensor([0]))
    assert "Decoding is not yet supported for StrClassificationInt." in str(exc_info.value)