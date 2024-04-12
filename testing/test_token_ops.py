import pathlib
import sys
import unittest

import torch
from torch import nn

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from src.configuration_hdit import HDiTConfig
from src.modeling_hdit_basic import TokenMerge, TokenSplit
from src.modeling_hdit_transformer import HDiT


class TestTokenOps(unittest.TestCase):
    def setUp(self):
        # Setup operations
        self.merge = TokenMerge(384, 768, 2)
        self.split = TokenSplit(768, 384, 2)

    def test_merging(self):
        input_tensor = torch.randn(1, 10, 10, 384)
        out = self.merge(input_tensor)
        assert out.shape == (1, 5, 5, 768)

    def test_splitting(self):
        input_tensor = torch.randn(1, 5, 5, 768)
        out = self.split(input_tensor)
        assert out.shape == (1, 10, 10, 384)


# This is the typical way of running the tests if this script is run directly
if __name__ == "__main__":
    unittest.main()
