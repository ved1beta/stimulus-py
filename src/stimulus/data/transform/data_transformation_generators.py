"""
This file contains noise generators classes for generating various types of noise.
"""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import multiprocessing as mp


class AbstractDataTransformer(ABC):
    """
    Abstract class for data transformers.

    All data transformers should have the seed in it. This is because the multiprocessing of them could unset the seed.

    Attributes:
        add_row (bool): whether the transformer adds rows to the data

    Methods:
        transform: transforms a data point
        transform_all: transforms a list of data points
    """

    def __init__(self):
        self.add_row = None

    @abstractmethod
    def transform(self, data: Any, seed: float = None) -> Any:
        """
        Transforms a single data point.

        This is an abstract method that should be implemented by the child class.

        Args:
            data (Any): the data to be transformed
            seed (float): the seed for reproducibility

        Returns:
            transformed_data (Any): the transformed data
        """
        #  np.random.seed(seed)
        raise NotImplementedError
    
    @abstractmethod
    def transform_all(self, data: list, seed: float = None) -> list:
        """
        Transforms a list of data points.

        This is an abstract method that should be implemented by the child class.

        Args:
            data (list): the data to be transformed
            seed (float): the seed for reproducibility

        Returns:
            transformed_data (list): the transformed data
        """
        #  np.random.seed(seed)
        raise NotImplementedError


class AbstractNoiseGenerator(AbstractDataTransformer):
    """
    Abstract class for noise generators. 

    All noise function should have the seed in it. This is because the multiprocessing of them could unset the seed.
    """

    def __init__(self):
        super().__init__()
        self.add_row = False 

class AbstractAugmentationGenerator(AbstractDataTransformer):
    """
    Abstract class for augmentation generators.

    All augmentation function should have the seed in it. This is because the multiprocessing of them could unset the seed.
    """

    def __init__(self):
        super().__init__()
        self.add_row = True

class UniformTextMasker(AbstractNoiseGenerator):
    """
    Mask characters in text.

    This noise generators replace characters with a masking character with a given probability.

    Attributes:
        mask (str): the character to use for masking

    Methods:
        transform: adds character masking to a single data point
        transform_all: adds character masking to a list of data points
    """
    def __init__(self, mask: str) -> None:
        super().__init__()
        self.mask = mask

    def transform(self, data: str, probability: float = 0.1, seed: float = None) -> str:
        """
        Adds character masking to the data.

        Args:
            data (str): the data to be transformed
            probability (float): the probability of adding noise
            seed (float): the seed for reproducibility

        Returns:
            transformed_data (str): the transformed data point
        """
        np.random.seed(seed)
        return ''.join([c if np.random.rand() > probability else self.mask for c in data])

    def transform_all(self, data: list, probability: float = 0.1, seed: float = None) -> list:
        """
        Adds character masking to multiple data points using multiprocessing.

        Args:
            data (list): the data to be transformed
            probability (float): the probability of adding noise
            seed (float): the seed for reproducibility

        Returns:
            transformed_data (list): the transformed data points
        """
        with mp.Pool(mp.cpu_count()) as pool:
            function_specific_input = [(item, probability, seed) for item in data]
            return pool.starmap(self.transform, function_specific_input)
        

class GaussianNoise(AbstractNoiseGenerator):
    """
    Add Gaussian noise to data

    This noise generator adds Gaussian noise to float values.

    Methods:
        transform: adds noise to a single data point
        transform_all: adds noise to a list of data points
    """

    def transform(self, data: float, mean: float = 0, std: float = 1, seed: float = None) -> float:
        """
        Adds Gaussian noise to a single point of data.

        Args:
            data (float): the data to be transformed
            mean (float): the mean of the Gaussian distribution
            std (float): the standard deviation of the Gaussian distribution
            seed (float): the seed for reproducibility

        Returns:
            transformed_data (float): the transformed data point
        """
        np.random.seed(seed)
        return data + np.random.normal(mean, std)
    
    def transform_all(self, data: list, mean: float = 0, std: float = 0, seed: float = None) -> np.array:
        """
        Adds Gaussian noise to a list of data points
        
        Args:
            data (list): the data to be transformed
            mean (float): the mean of the Gaussian distribution
            std (float): the standard deviation of the Gaussian distribution
            seed (float): the seed for reproducibility

        Returns:
            transformed_data (np.array): the transformed data points
        """
        np.random.seed(seed)
        return np.array(np.array(data) + 
                        np.random.normal(mean, std, len(data)))
    


class ReverseComplement(AbstractAugmentationGenerator):
    """
    Reverse complement biological sequences

    This augmentation strategy reverse complements the input nucleotide sequences.

    Methods:
        transform: reverse complements a single data point
        transform_all: reverse complements a list of data points

    Raises:
        ValueError: if the type of the sequence is not DNA or RNA
    """
    def __init__(self, type:str = "DNA") -> None:
        super().__init__()
        if (type not in ("DNA", "RNA")):
            raise ValueError("Currently only DNA and RNA sequences are supported. Update the class ReverseComplement to support other types.")
        if type == "DNA":
            self.complement_mapping = str.maketrans('ATCG', 'TAGC')
        elif type == "RNA":
            self.complement_mapping = str.maketrans('AUCG', 'UAGC')


    def transform(self, data: str) -> str:
        """
        Returns the reverse complement of a list of string data using the complement_mapping.

        Args:
            data (str): the sequence to be transformed

        Returns:
            transformed_data (str): the reverse complement of the sequence
        """
        return data.translate(self.complement_mapping)[::-1]

    def transform_all(self, data: list) -> list:
        """
        Reverse complement multiple data points using multiprocessing.

        Args:
            data (list): the sequences to be transformed
        
        Returns:
            transformed_data (list): the reverse complement of the sequences
        """
        with mp.Pool(mp.cpu_count()) as pool:
            function_specific_input = [(item) for item in data]
            return pool.map(self.transform, function_specific_input)
        
class GaussianChunk(AbstractAugmentationGenerator):
    """
    Subset data around a random midpoint

    This augmentation strategy chunks the input sequences, for which the middle positions are obtained through a gaussian distribution.
    
    In concrete, it changes the middle position (ie. peak summit) to another position. This position is chosen based on a gaussian distribution, so the region close to the middle point are more likely to be chosen than the rest.
    Then a chunk with size `chunk_size` around the new middle point is returned.
    This process will be repeated for each sequence with `transform_all`.

    Methods:
        transform: chunk a single list
        transform_all: chunks multiple lists
    """

    def transform(self, data: str, chunk_size: int, seed: float = None, std: float = 1) -> str:
        """
        Chunks a sequence of size chunk_size from the middle position +/- a value obtained through a gaussian distribution.

        Args:
            data (str): the sequence to be transformed
            chunk_size (int): the size of the chunk
            seed (float): the seed for reproducibility
            std (float): the standard deviation of the gaussian distribution

        Returns:
            transformed_data (str): the chunk of the sequence

        Raises:
            AssertionError: if the input data is shorter than the chunk size
        """
        np.random.seed(seed)

        # make sure that the data is longer than chunk_size otherwise raise an error
        assert len(data) > chunk_size, "The input data is shorter than the chunk size"

        # Get the middle position of the input sequence
        middle_position = len(data) // 2

        # Change the middle position by a value obtained through a gaussian distribution
        new_middle_position = int(middle_position + np.random.normal(0, std))

        # Get the start and end position of the chunk
        start_position = new_middle_position - chunk_size // 2
        end_position = new_middle_position + chunk_size // 2

        # if the start position is negative, set it to 0
        if start_position < 0:
            start_position = 0

        # Get the chunk of size chunk_size from the start position if the end position is smaller than the length of the data
        if end_position < len(data):
            return data[start_position: start_position + chunk_size]
        # Otherwise return the chunk of the sequence from the end of the sequence of size chunk_size
        else:
            return data[-chunk_size:]
        

    def transform_all(self, data: list, chunk_size: int, seed: float = None, std: float = 1) -> list:
        """
        Adds chunks to multiple lists using multiprocessing.

        Args:
            data (list): the sequences to be transformed
            chunk_size (int): the size of the chunk
            seed (float): the seed for reproducibility
            std (float): the standard deviation of the gaussian distribution

        Returns:
            transformed_data (list): the transformed sequences
        """
        with mp.Pool(mp.cpu_count()) as pool:
            function_specific_input = [(item, chunk_size, seed, std) for item in data]
            return pool.starmap(self.transform, function_specific_input)


        


    
