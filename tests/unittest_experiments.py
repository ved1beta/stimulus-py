import unittest

from src.stimulus.data.experiments import DnaToFloatExperiment


class TestDnaToFloatExperiment(unittest.TestCase):
    def setUp(self):
        self.dna_to_float_experiment = DnaToFloatExperiment()
