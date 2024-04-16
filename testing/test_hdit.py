import pathlib
import sys
import unittest

import torch
from torch import nn

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from src.configuration_hdit import HDiTConfig
from src.modeling_hdit_transformer import HDiT


class TestHDiT(unittest.TestCase):
    def setUp(self):
        # Setup operations
        self.HDiTConfig = HDiTConfig()
        self.model = HDiT(self.HDiTConfig)

    def test_forward(self):
        input_tensor = torch.randn(
            1,
            3,
            256,
            256,
        )
        timesteps = torch.randn(1, 768)
        cond_tensor = torch.randint(0, 1000, (1,))
        # Forward pass
        output = self.model(input_tensor, timesteps, cond_tensor)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (1, 3, 256, 256))


# This is the typical way of running the tests if this script is run directly
if __name__ == "__main__":
    unittest.main()
