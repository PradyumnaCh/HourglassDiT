import pathlib
import sys
import unittest

import torch
from torch import nn

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from diffusers.schedulers import DPMSolverSinglestepScheduler

from src.configuration_hdit import HDiTConfig
from src.modeling_hdit_transformer import HDiT
from src.pipeline_hdit import HDiTPipeline


class TestPipeline(unittest.TestCase):
    def setUp(self):
        # Setup operations
        config = HDiTConfig()
        model = HDiT(config)
        scheduler = DPMSolverSinglestepScheduler()
        self.pipe = HDiTPipeline(
            backbone=model,
            scheduler=scheduler,
            id2label={1: "white shark", 2: "umbrella"},
        )

    def test_inference(self):
        # pick words from Imagenet class labels
        self.pipe.labels  # to print all available words

        # pick words that exist in ImageNet
        words = ["white shark", "umbrella"]

        class_ids = self.pipe.get_label_ids(words)

        generator = torch.manual_seed(33)
        output = self.pipe(
            class_labels=class_ids, num_inference_steps=25, generator=generator
        )

        image = output.images[0]  # label 'white shark'
        self.assertEqual(image.shape, (1, 3, 256, 256))


# This is the typical way of running the tests if this script is run directly
if __name__ == "__main__":
    unittest.main()
