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

import copy
import json
from typing import Any, Optional

from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.utils.data import Dataset

from far.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class VBenchT2VDataset(Dataset, ConfigMixin):

    config_name = 'vbench_t2v_dataset_config.json'

    @register_to_config
    def __init__(
        self,
        json_path: str = 'assets/data/meta/vbench/VBench_aug_full_info.json',
        num_samples_per_prompt: Optional[int] = 5,
    ):
        super().__init__()

        with open(json_path, 'r') as json_file:
            meta_data = json.load(json_file)

        self.samples = []

        for prompt_item in meta_data:
            for idx in range(num_samples_per_prompt):
                prompt_item_ = copy.deepcopy(prompt_item)
                prompt_item_['video_path'] = f"{prompt_item_['prompt_en']}-{idx}.mp4"
                self.samples.append(prompt_item_)

        # append sample seed to each samples for metric reimplementation
        for idx, _ in enumerate(self.samples):
            self.samples[idx]['seed'] = idx

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]
