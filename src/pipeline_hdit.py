# Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
# William Peebles and Saining Xie
#
# Copyright (c) 2021 OpenAI
# MIT License
#
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from diffusers.pipelines.pipeline_utils import (
    DiffusionPipeline,
    ImagePipelineOutput,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils.torch_utils import randn_tensor

from src.modeling_hdit_transformer import HDiT


class HDiTPipeline(DiffusionPipeline):
    r"""
    Pipeline for image generation based on a Transformer backbone instead of a UNet.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        hdit_model ([`HDiTModel`]):
            A class conditioned `HDiTModel` to denoise the encoded image latents.
        scheduler ([`DDIMScheduler`]):
            A scheduler to be used in combination with `hdit_model` to denoise the encoded image latents.
    """

    model_cpu_offload_seq = "backbone"

    def __init__(
        self,
        backbone: HDiT,
        scheduler: KarrasDiffusionSchedulers,
        id2label: Optional[Dict[int, str]] = None,
    ):
        super().__init__()
        print("Execution device is", self._execution_device)
        self.register_modules(backbone=backbone, scheduler=scheduler)

        self.num_channels = self.backbone.config.num_channels
        self.input_size = self.backbone.config.input_size
        # create a imagenet -> id dictionary for easier use
        self.labels = {}
        if id2label is not None:
            for key, value in id2label.items():
                for label in value.split(","):
                    self.labels[label.lstrip().rstrip()] = int(key)
            self.labels = dict(sorted(self.labels.items()))

    def get_label_ids(self, label: Union[str, List[str]]) -> List[int]:
        r"""

        Map label strings from ImageNet to corresponding class ids.

        Parameters:
            label (`str` or `dict` of `str`):
                Label strings to be mapped to class ids.

        Returns:
            `list` of `int`:
                Class ids to be processed by pipeline.
        """

        if not isinstance(label, list):
            label = list(label)

        for l in label:
            if l not in self.labels:
                raise ValueError(
                    f"{l} does not exist. Please make sure to select one of the following labels: \n {self.labels}."
                )

        return [self.labels[l] for l in label]

    @torch.no_grad()
    def __call__(
        self,
        class_labels: List[int],
        guidance_scale: float = 4.0,
        generator: Optional[
            Union[torch.Generator, List[torch.Generator]]
        ] = None,
        num_inference_steps: int = 50,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            class_labels (List[int]):
                List of ImageNet class labels for the images to be generated.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 250):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`ImagePipelineOutput`] instead of a plain tuple.

        Examples:

        ```py
        >>> from diffusers import DiTPipeline, DPMSolverMultistepScheduler
        >>> import torch

        >>> pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float16)
        >>> pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        >>> pipe = pipe.to("cuda")

        >>> # pick words from Imagenet class labels
        >>> pipe.labels  # to print all available words

        >>> # pick words that exist in ImageNet
        >>> words = ["white shark", "umbrella"]

        >>> class_ids = pipe.get_label_ids(words)

        >>> generator = torch.manual_seed(33)
        >>> output = pipe(class_labels=class_ids, num_inference_steps=25, generator=generator)

        >>> image = output.images[0]  # label 'white shark'
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """

        batch_size = len(class_labels)
        noisy_sample = randn_tensor(
            shape=(
                batch_size,
                self.num_channels,
                self.input_size,
                self.input_size,
            ),
            generator=generator,
            device=self._execution_device,
            dtype=self.hdit_model.dtype,
        )
        model_input = (
            torch.cat([noisy_sample] * 2)
            if guidance_scale > 1
            else noisy_sample
        )

        class_labels = torch.tensor(
            class_labels, device=self._execution_device
        ).reshape(-1)
        class_null = torch.tensor(
            [1000] * batch_size, device=self._execution_device
        )
        class_labels_input = (
            torch.cat([class_labels, class_null], 0)
            if guidance_scale > 1
            else class_labels
        )

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        for t in self.progress_bar(self.scheduler.timesteps):
            if guidance_scale > 1:
                half = model_input[: len(model_input) // 2]
                model_input = torch.cat([half, half], dim=0)
            model_input = self.scheduler.scale_model_input(model_input, t)

            timesteps = t
            if not torch.is_tensor(timesteps):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                # This would be a good case for the `match` statement (Python 3.10+)
                is_mps = model_input.device.type == "mps"
                if isinstance(timesteps, float):
                    dtype = torch.float32 if is_mps else torch.float64
                else:
                    dtype = torch.int32 if is_mps else torch.int64
                timesteps = torch.tensor(
                    [timesteps], dtype=dtype, device=model_input.device
                )
            elif len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(model_input.device)
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timesteps = timesteps.expand(model_input.shape[0])
            # predict noise model_output
            noise_pred = self.backbone(
                model_input,
                timestep=timesteps,
                class_labels=class_labels_input,
            ).sample

            # perform guidance
            if guidance_scale > 1:
                eps, rest = (
                    noise_pred[:, : self.num_channels],
                    noise_pred[:, self.num_channels :],
                )
                cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

                half_eps = uncond_eps + guidance_scale * (
                    cond_eps - uncond_eps
                )
                eps = torch.cat([half_eps, half_eps], dim=0)

                noise_pred = torch.cat([eps, rest], dim=1)

            # learned sigma
            if self.transformer.config.out_channels // 2 == self.num_channels:
                model_output, _ = torch.split(
                    noise_pred, self.num_channels, dim=1
                )
            else:
                model_output = noise_pred

            # compute previous image: x_t -> x_t-1
            model_input = self.scheduler.step(
                model_output, t, model_input
            ).prev_sample

        if guidance_scale > 1:
            samples, _ = model_input.chunk(2, dim=0)
        else:
            samples = model_input

        # Normalization adjustment
        samples = (samples / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            samples = self.numpy_to_pil(samples)

        if not return_dict:
            return (samples,)

        return ImagePipelineOutput(images=samples)

    def __getattr__(self, name: str) -> Any:
        is_in_config = "_internal_dict" in self.__dict__ and hasattr(
            self.__dict__["_internal_dict"], name
        )
        is_attribute = name in self.__dict__

        if is_in_config and not is_attribute:
            deprecation_message = f"Accessing config attribute `{name}` directly via '{type(self).__name__}' object attribute is deprecated. Please access '{name}' over '{type(self).__name__}'s config object instead, e.g. 'scheduler.config.{name}'."
            deprecate(
                "direct config name access",
                "1.0.0",
                deprecation_message,
                standard_warn=False,
            )
            return self._internal_dict[name]

        if is_in_config:
            return self._internal_dict[name]

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )
