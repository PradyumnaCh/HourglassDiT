import pathlib
import sys
import unittest

import torch
from torch import nn

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from src.configuration_hdit import HDiTConfig
from src.modeling_hdit_transformer import HDiT


class TestHDiT(unittest.TestCase):
    def test_initialization(self):
        # Setup operations
        self.HDiTConfig = HDiTConfig()
        self.model = HDiT(self.HDiTConfig)

    def test_forward(self):
        # Setup operations
        self.HDiTConfig = HDiTConfig()
        self.model = HDiT(self.HDiTConfig)

        input_tensor = torch.randn(1, 10, 10, 384)
        cond_tensor = torch.randn(1, 768)
        # Forward pass
        output = self.model(input_tensor, cond_tensor)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (1, 3, 224, 224))


# This is the typical way of running the tests if this script is run directly
if __name__ == "__main__":
    unittest.main()
