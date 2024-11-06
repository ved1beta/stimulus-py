"""This file contains encoders classes for encoding various types of data."""

import logging
import multiprocessing as mp
from abc import ABC, abstractmethod
from typing import Any, Union

import numpy as np
from sklearn import preprocessing

logger = logging.getLogger(__name__)


class AbstractEncoder(ABC):
    """Abstract class for encoders.

    Encoders are classes that encode string data into numerical representations, said numerical representations should be the exact input of the model.

    Methods:
        encode: encodes a single data point
        encode_all: encodes a list of data points into a numpy array
        encode_multiprocess: encodes a list of data points using multiprocessing
        decode: decodes a single data point

    """

    @abstractmethod
    def encode(self, data: Any) -> Any:
        """Encode a single data point.

        This is an abstract method, child classes should overwrite it.

        Args:
            data (any): a single data point

        Returns:
            encoded_data_point (any): Òthe encoded data point
        """
        raise NotImplementedError

    @abstractmethod
    def encode_all(self, data: list) -> np.array:
        """Encode a list of data points.

        This is an abstract method, child classes should overwrite it.

        Args:
            data (list): a list of data points

        Returns:
            encoded_data (np.array): encoded data points
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, data: Any) -> Any:
        """Decode a single data point.

        This is an abstract method, child classes should overwrite it.

        Args:
            data (any): a single encoded data point

        Returns:
            decoded_data_point (any): the decoded data point
        """
        raise NotImplementedError

    def encode_multiprocess(self, data: list) -> list:
        """Helper function for encoding the data using multiprocessing.

        Args:
            data (list): a list of data points

        Returns:
            encoded_data (list): encoded data points
        """
        with mp.Pool(mp.cpu_count()) as pool:
            return pool.map(self.encode, data)


class TextOneHotEncoder(AbstractEncoder):
    """One hot encoder for text data

    NOTE encodes based on the given alphabet
    If a character c is not in the alphabet, c will be represented by a vector of zeros.

    Attributes:
        alphabet (str): the alphabet to one hot encode the data with
        encoder (OneHotEncoder): preprocessing.OneHotEncoder object initialized with self.alphabet

    Methods:
        encode: encodes a single data point
        encode_all: encodes a list of data points into a numpy array
        encode_multiprocess: encodes a list of data points using multiprocessing
        decode: decodes a single data point
        _sequence_to_array: transforms a sequence into a numpy array
    """

    def __init__(self, alphabet: str = "acgt") -> None:
        """Initialize the TextOneHotEncoder class.

        Args:
            alphabet (str): the alphabet to one hot encode the data with

        Raises:
            ValueError: If the input alphabet is not a string.
        """
        if not isinstance(alphabet, str):
            error_msg = f"Expected string input, got {type(alphabet).__name__}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        self.alphabet = alphabet
        self.encoder = preprocessing.OneHotEncoder(
            categories=[list(alphabet)],
            handle_unknown="ignore",
        )  # handle_unknown='ignore' unsures that a vector of zeros is returned for unknown characters, such as 'Ns' in DNA sequences

    def _sequence_to_array(self, sequence: str) -> np.array:
        """This function transforms the given sequence to an array.

        Args:
            sequence (str): a sequence of characters

        Returns:
            sequence_array (np.array): the sequence as a numpy array

        Raises:
            ValueError: If the input data is not a string.

        Examples:
            >>> encoder = TextOneHotEncoder(alphabet="acgt")
            >>> encoder._sequence_to_array("acgt")
            array(['a'],['b'],['c'],['d'])
        """
        if not isinstance(sequence, str):
            error_msg = f"Expected string input, got {type(sequence).__name__}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        sequence_lower_case = sequence.lower()
        sequence_array = np.array(list(sequence_lower_case))
        return sequence_array.reshape(-1, 1)

    def encode(self, data: str) -> np.array:
        """One hot encodes a single sequence.

        Takes a single string sequence and returns a numpy array of shape (sequence_length, alphabet_length).
        The returned numpy array corresponds to the one hot encoding of the sequence.
        Unknown characters are represented by a vector of zeros.

        Args:
            data (str): single sequence

        Returns:
            encoded_data_point (np.array): one hot encoded sequence

        Raises:
            ValueError: If the input data is not a string.

        Examples:
            >>> encoder = TextOneHotEncoder(alphabet="acgt")
            >>> encoder.encode("acgt")
            array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

            >>> encoder.encode("acgtn")
            array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1],
                   [0, 0, 0, 0]])
        """
        sequence_array = self._sequence_to_array(data)
        transformed = self.encoder.fit_transform(sequence_array)
        return np.squeeze(np.stack(transformed.toarray()))

    def encode_all(self, data: Union[list, str]) -> Union[np.array, list]:
        """Encodes a list of data points

        TODO instead maybe we can run encode_multiprocess when data size is larger than a certain threshold.
        """
        if not isinstance(data, list):
            encoded_data = self.encode(data)
            return np.array(
                [encoded_data],
            )  # reshape the array in a batch of 1 configuration as a np.ndarray (so shape is (1, sequence_length, alphabet_length))

        encoded_data = self.encode_multiprocess(data)
        # try to transform the list of arrays to a single array and return it
        # if it fails (when the list of arrays is not of the same length), return the list of arrays
        try:
            return np.array(encoded_data)
        except ValueError:
            return encoded_data

    def decode(self, data: np.array) -> str:
        """Decodes the data."""
        return self.encoder.inverse_transform(data)


class FloatEncoder(AbstractEncoder):
    """Encoder for float data."""

    def encode(self, data: float) -> float:
        """Encodes the data.
        This method takes as input a single data point, should be mappable to a single output.
        """
        return float(data)

    def encode_all(self, data: list) -> np.array:
        """Encodes the data.
        This method takes as input a list of data points, should be mappable to a single output.
        """
        if not isinstance(data, list):
            data = [data]
        return np.array([self.encode(d) for d in data])

    def decode(self, data: float) -> float:
        """Decodes the data."""
        return data


class IntEncoder(FloatEncoder):
    """Encoder for integer data."""

    def encode(self, data: int) -> int:
        """Encodes the data.
        This method takes as input a single data point, should be mappable to a single output.
        """
        return int(data)


class StrClassificationIntEncoder(AbstractEncoder):
    """Considering a ensemble of strings, this encoder encodes them into integers from 0 to (n-1) where n is the number of unique strings."""

    def encode(self, data: str) -> int:
        """Returns an error since encoding a single string does not make sense."""
        raise NotImplementedError("Encoding a single string does not make sense. Use encode_all instead.")

    def encode_all(self, data: list) -> np.array:
        """Encodes the data.
        This method takes as input a list of data points, should be mappable to a single output, using LabelEncoder from scikit learn and returning a numpy array.
        For more info visit : https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
        """
        if not isinstance(data, list):
            data = [data]
        encoder = preprocessing.LabelEncoder()
        return encoder.fit_transform(data)

    def decode(self, data: Any) -> Any:
        """Returns an error since decoding does not make sense without encoder information, which is not yet supported."""
        raise NotImplementedError("Decoding is not yet supported for StrClassificationInt.")


class StrClassificationScaledEncoder(StrClassificationIntEncoder):
    """Considering a ensemble of strings, this encoder encodes them into floats from 0 to 1 (essentially scaling the integer encoding)."""

    def encode_all(self, data: list) -> np.array:
        """Encodes the data.
        This method takes as input a list of data points, should be mappable to a single output, using LabelEncoder from scikit learn and returning a numpy array.
        For more info visit : https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
        """
        encoded_data = super().encode_all(data)
        return encoded_data / (len(np.unique(encoded_data)) - 1)

    def decode(self, data: Any) -> Any:
        """Returns an error since decoding does not make sense without encoder information, which is not yet supported."""
        raise NotImplementedError("Decoding is not yet supported for StrClassificationScaled.")


class FloatRankEncoder(AbstractEncoder):
    """Considering an ensemble of float values, this encoder encodes them into floats from 0 to 1, where 1 is the maximum value and 0 is the minimum value."""

    def encode(self, data: float) -> float:
        """Returns an error since encoding a single float does not make sense."""
        raise NotImplementedError("Encoding a single float does not make sense. Use encode_all instead.")

    def encode_all(self, data: list) -> np.array:
        """Encodes the data.
        This method takes as input a list of data points, should be mappable to a single output
        """
        # Convert to array if needed
        if not isinstance(data, list):
            data = [data]
        data = np.array(data)

        # Get ranks (0 is lowest, n-1 is highest)
        ranks = np.argsort(np.argsort(data))

        # normalize ranks to be between 0 and 1
        return ranks / (len(ranks) - 1)

    def decode(self, data: Any) -> Any:
        """Returns an error since decoding does not make sense without encoder information, which is not yet supported."""
        raise NotImplementedError("Decoding is not yet supported for FloatRank.")


class IntRankEncoder(FloatRankEncoder):
    """Considering an ensemble of integer values, this encoder encodes them into floats from 0 to 1, where 1 is the maximum value and 0 is the minimum value."""

    def encode(self, data: int) -> int:
        """Returns an error since encoding a single integer does not make sense."""
        raise NotImplementedError("Encoding a single integer does not make sense. Use encode_all instead.")

    def encode_all(self, data: list) -> np.array:
        """Encodes the data.
        This method takes as input a list of data points, should be mappable to a single output, using min-max scaling.
        """
        # Convert to array if needed
        if not isinstance(data, list):
            data = [data]
        data = np.array(data)

        # Get ranks (0 is lowest, n-1 is highest)
        ranks = np.argsort(np.argsort(data))

        return ranks

    def decode(self, data: Any) -> Any:
        """Returns an error since decoding does not make sense without encoder information, which is not yet supported."""
        raise NotImplementedError("Decoding is not yet supported for IntRank.")
