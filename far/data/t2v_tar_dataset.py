# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
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
#
# SPDX-License-Identifier: Apache-2.0

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torchvision.transforms import Compose, Normalize, Resize

from far.utils.data_util import retry_load_error
from far.utils.registry import DATASET_REGISTRY
from far.utils.video_util import VideoLoader_Imageio_Backend

from .web_dataset.wids import WebDataset


@DATASET_REGISTRY.register()
class T2VTarDataset(WebDataset, ConfigMixin):

    config_name = 't2v_tar_dataset_config.json'

    @register_to_config
    def __init__(
        self,
        meta_path,
        num_frames=None,
        data_dtype='.pth'
    ):
        super().__init__(data_dir=None, meta_path=meta_path)
        self.num_frames = num_frames

        if self.config.data_dtype == '.mp4':
            self.transform = Compose([
                Resize((480, 832)),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ])

    @retry_load_error(max_attempts=3)
    def __getitem__(self, index: int):

        sample = self.dataset[index]
        if self.config.data_dtype == '.mp4':
            video, annotation = sample['.mp4'], sample['.json']

            video = VideoLoader_Imageio_Backend(video).get_frames(self.num_frames, target_fps=16)
            assert video.shape[0] == self.num_frames, (f'expected {self.num_frames} frames, got {video.shape[0]}')

            video = torch.from_numpy(video / 255.0).float().permute(0, 3, 1, 2).contiguous()
            pixel_values = self.transform(video)

            return {
                'pixel_values': pixel_values,
                'prompts': annotation['captions'][0],
            }
        elif self.config.data_dtype == '.pth':
            latents, annotation = sample['.pth'], sample['.json']
            if self.num_frames is not None:
                latents = latents[:, :self.num_frames]

            return {
                'latents': latents,
                'prompts': annotation['captions'][0],
            }