import pathlib
import sys
import unittest

import torch
from torch import nn

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from diffusers.pipelines import DiTPipeline
from diffusers.schedulers import (
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
)

from src.configuration_hdit import HDiTConfig
from src.modeling_hdit_transformer import HDiT
from src.pipeline_hdit import HDiTPipeline


class TestDiTPipeline(unittest.TestCase):
    def setUp(self):
        # Setup operations
        self.pipe = DiTPipeline.from_pretrained(
            "facebook/DiT-XL-2-256", torch_dtype=torch.float16
        )
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe = self.pipe.to("cuda")

    def test_inference(self):
        # pick words from Imagenet class labels
        self.pipe.labels  # to print all available words
        print(self.pipe._execution_device)
        # pick words that exist in ImageNet
        words = ["white shark", "umbrella"]

        class_ids = self.pipe.get_label_ids(words)

        generator = torch.manual_seed(33)
        print(hasattr(self.pipe, "_execution_device"))
        output = self.pipe(
            class_labels=class_ids, num_inference_steps=25, generator=generator
        )

        image = output.images[0]  # label 'white shark'
        self.assertEqual(image.shape, (1, 3, 256, 256))


# This is the typical way of running the tests if this script is run directly
if __name__ == "__main__":
    unittest.main()
