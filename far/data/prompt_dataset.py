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

import json

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from far.utils.registry import DATASET_REGISTRY
from far.utils.video_util import select_frame_indices


@DATASET_REGISTRY.register()
class PromptDataset(Dataset):

    def __init__(
        self,
        prompt_path,
        preprocess_cfg=None
    ):
        with open(prompt_path, 'r') as fr:
            self.prompt_list = json.load(fr)

        self.preprocess_cfg = preprocess_cfg

        if self.preprocess_cfg:
            self.transform = transforms.Compose([
                transforms.Resize([preprocess_cfg['height'], preprocess_cfg['width']]),
            ])

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, index):

        entry = self.prompt_list[index]
        if 'prompt_file' in entry:
            with open(entry['prompt_file'], 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
        else:
            prompt = entry['prompt']

        if 'image' in self.prompt_list[index]:
            # I2V
            image = Image.open(self.prompt_list[index]['image']).convert('RGB')
            image = transforms.ToTensor()(self.transform(image)).unsqueeze(0)
            return {'index': index, 'prompt': prompt, 'image': image}
        elif 'video' in self.prompt_list[index]:
            # video continue
            import decord
            decord.bridge.set_bridge('torch')

            video_path = self.prompt_list[index]['video']

            video_reader = decord.VideoReader(video_path)
            total_frames = len(video_reader)

            original_fps = video_reader.get_avg_fps()

            frame_idxs = select_frame_indices(total_frames, original_fps, target_fps=self.preprocess_cfg['target_fps'])[:self.preprocess_cfg['num_cond_frames']]
            frames = video_reader.get_batch(frame_idxs)

            frames = (frames / 255.0).float().permute(0, 3, 1, 2).contiguous()
            frames = self.transform(frames)
            return {'index': index, 'prompt': prompt, 'video': frames}
        else:
            return {'index': index, 'prompt': prompt}
