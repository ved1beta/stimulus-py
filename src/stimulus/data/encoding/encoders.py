"""This file contains encoders classes for encoding various types of data."""

import logging
import multiprocessing as mp
from abc import ABC, abstractmethod
from typing import Any, Union, List

import numpy as np
import torch
from sklearn import preprocessing

logger = logging.getLogger(__name__)


class AbstractEncoder(ABC):
    """Abstract class for encoders.

    Encoders are classes that encode the raw data into torch.tensors in meaningful ways.

    Methods:
        encode: encodes a single data point
        encode_all: encodes a list of data points into a torch.tensor
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
        alphabet (str): the alphabet to one hot encode the data with.
        case_sensitive (bool): whether the encoder is case sensitive or not. Default = False
        padding (bool): whether to pad the sequences with zero or not. Default = True
        encoder (OneHotEncoder): preprocessing.OneHotEncoder object initialized with self.alphabet

    Methods:
        encode: encodes a single data point
        encode_all: encodes a list of data points into a numpy array
        encode_multiprocess: encodes a list of data points using multiprocessing
        decode: decodes a single data point
        _sequence_to_array: transforms a sequence into a numpy array
    """

    def __init__(self, alphabet: str = "acgt", case_sensitive: bool = False, padding = True) -> None:
        """Initialize the TextOneHotEncoder class.

        Args:
            alphabet (str): the alphabet to one hot encode the data with.

        Raises:
            ValueError: If the input alphabet is not a string.
        """
        if not isinstance(alphabet, str):
            error_msg = f"Expected a string input for alphabet, got {type(alphabet).__name__}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if not case_sensitive:
            alphabet = alphabet.lower()
        
        self.alphabet = alphabet
        self.case_sensitive = case_sensitive
        self.padding = padding

        self.encoder = preprocessing.OneHotEncoder(
            categories=[list(alphabet)],
            handle_unknown="ignore",
        )  # handle_unknown='ignore' unsures that a vector of zeros is returned for unknown characters, such as 'Ns' in DNA sequences

        # Fit the encoder with all possible characters
        self.encoder.fit(np.array(list(alphabet)).reshape(-1, 1))

    def _sequence_to_array(self, sequence: str) -> np.array:
        """This function transforms the given sequence to an array.

        Args:
            sequence (str): a sequence of characters.

        Returns:
            sequence_array (np.array): the sequence as a numpy array

        Raises:
            ValueError: If the input data is not a string.

        Examples:
            >>> encoder = TextOneHotEncoder(alphabet="acgt")
            >>> encoder._sequence_to_array("acctg")
            array(['a'],['c'],['c'],['t'],['g'])
        """
        if not isinstance(sequence, str):
            error_msg = f"Expected string input for sequence, got {type(sequence).__name__}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if not self.case_sensitive:
            sequence = sequence.lower()

        sequence_array = np.array(list(sequence))
        return sequence_array.reshape(-1, 1)

    def encode(self, data: str) -> torch.Tensor:
        """One hot encodes a single sequence.

        Takes a single string sequence and returns a torch tensor of shape (sequence_length, alphabet_length).
        The returned tensor corresponds to the one hot encoding of the sequence.
        Unknown characters are represented by a vector of zeros.

        Args:
            data (str): single sequence

        Returns:
            encoded_data_point (torch.Tensor): one hot encoded sequence

        Raises:
            ValueError: If the input data is not a string.

        Examples:
            >>> encoder = TextOneHotEncoder(alphabet="acgt")
            >>> encoder.encode("acgt")
            tensor([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
            >>> encoder.encode("acgtn")
            tensor([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0]])

            >>> encoder = TextOneHotEncoder(alphabet="ACgt", case_sensitive=True)
            >>> encoder.encode('acgt')
            tensor([[0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
            >>> encoder.encode('ACgt')
            tensor([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
        """
        sequence_array = self._sequence_to_array(data)
        transformed = self.encoder.transform(sequence_array)
        numpy_array = np.squeeze(np.stack(transformed.toarray()))
        return torch.from_numpy(numpy_array)

    def encode_all(self, data: Union[str, List[str]]) -> torch.Tensor:
        """Encodes a list of sequences.

        Takes a list of string sequences and returns a torch tensor of shape (number_of_sequences, sequence_length, alphabet_length).
        The returned tensor corresponds to the one hot encoding of the sequences.
        Unknown characters are represented by a vector of zeros.

        Args:
            data (Union[list, str]): list of sequences or a single sequence

        Returns:
            encoded_data (torch.Tensor): one hot encoded sequences

        Raises:
            ValueError: If the input data is not a list or a string.
            ValueError: If all sequences do not have the same length when padding is False.

        Examples:
            >>> encoder = TextOneHotEncoder(alphabet="acgt")
            >>> encoder.encode_all(["acgt","acgtn"])
            tensor([[[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1],
                     [0, 0, 0, 0]], // this is padded with zeros

                    [[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1],
                     [0, 0, 0, 0]]])
        """
        # encode data
        if isinstance(data, str):
            encoded_data = self.encode(data)
            return torch.stack([encoded_data])
        elif isinstance(data, list):
            # TODO instead maybe we can run encode_multiprocess when data size is larger than a certain threshold.
            encoded_data = self.encode_multiprocess(data)
        else:
            error_msg = f"Expected list or string input for data, got {type(data).__name__}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # handle padding
        if self.padding:
            max_length = max([len(d) for d in encoded_data])
            encoded_data = [np.pad(d, ((0, max_length - len(d)), (0, 0))) for d in encoded_data]
        else:
            lengths = set([len(d) for d in encoded_data])
            if len(lengths) > 1:
                error_msg = f"All sequences must have the same length when padding is False."
                logger.error(error_msg)
                raise ValueError(error_msg)
            
        # convert to torch tensor
        try:
            return torch.from_numpy(np.array(encoded_data))
        except Exception as e:
            logger.error(f"Failed to convert sequences to tensor: {str(e)}")
            raise RuntimeError(f"Failed to convert sequences to tensor: {str(e)}")

    def decode(self, data: torch.Tensor) -> Union[str, List[str]]:
        """Decodes one-hot encoded tensor back to sequences.
        
        Args:
            data (torch.Tensor): 2D or 3D tensor of one-hot encoded sequences
                - 2D shape: (sequence_length, alphabet_size)
                - 3D shape: (batch_size, sequence_length, alphabet_size)

        NOTE that when decoding 3D shape tensor, it assumes all sequences have the same length.
        
        Returns:
            Union[str, List[str]]: Single sequence string or list of sequence strings

        Raises:
            ValueError: If the input data is not a 2D or 3D tensor

        Examples:
        
        """
        if data.dim() == 2:
            # Single sequence
            data_np = data.numpy().reshape(-1, len(self.alphabet))
            decoded = self.encoder.inverse_transform(data_np).flatten()
            return ''.join([i for i in decoded if i != None])
        
        elif data.dim() == 3:
            # Multiple sequences
            batch_size, seq_len, _ = data.shape
            data_np = data.reshape(-1, len(self.alphabet)).numpy()
            decoded = self.encoder.inverse_transform(data_np)
            sequences = decoded.reshape(batch_size, seq_len)
            sequences = np.where(sequences == None, '-', sequences)
            return [''.join(seq) for seq in sequences]
        
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {data.dim()}D")


class FloatEncoder(AbstractEncoder):
    """Encoder for float data."""

    def __init__(self, dtype: torch.dtype = torch.float) -> None:
        """Initialize the FloatEncoder class.

        Args:
            dtype (torch.dtype): the data type of the encoded data. Default = torch.float (32-bit floating point)
        """
        self.dtype = dtype

    def encode(self, data: float) -> torch.Tensor:
        """Encodes the data.
        This method takes as input a single data point, should be mappable to a single output.
        """
        return self.encode_all(data) # there is no difference in this case

    def encode_all(self, data: Union[float, List[float]]) -> torch.Tensor:
        """Encodes the data.
        This method takes as input a list of data points, should be mappable to a single output.
        """
        if not isinstance(data, list):
            data = [data]
        try:
            return torch.tensor(data, dtype=self.dtype)
        except (TypeError, ValueError) as e:
            err_msg = (
                f"Failed to convert {data} to tensor of type {self.dtype}. "
                "Make sure all elements are valid floats."
            )
            logger.error(err_msg)
            raise RuntimeError(err_msg) from e

    def decode(self, data: torch.Tensor) -> List[float]:
        """Decodes the data."""
        return data.numpy().tolist()

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
