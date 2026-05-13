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

import os
from dataclasses import dataclass

import decord
import torch
from diffusers.utils import export_to_video
from omegaconf import MISSING, OmegaConf
from PIL import Image
from torchvision import transforms

from far.pipelines.pipeline_far_wan_anyflow import FARWanAnyFlowPipeline
from far.pipelines.pipeline_wan_anyflow import WanAnyFlowPipeline
from far.utils.video_util import select_frame_indices
from far.utils.vis_util import draw_rectangle

decord.bridge.set_bridge('torch')


@dataclass
class DemoConfig:
    """CLI keys for `python demo.py key=value ...`"""

    # Directory from `save_pretrained` (e.g. AnyFlow-FAR-Wan2.1-1.3B-Diffusers export).
    model_path: str = MISSING
    # t2v | ti2v | tv2v
    task_type: str = 't2v'
    # Where to write demo_*.mp4.
    save_dir: str = MISSING


def inference_causal_demo(model_path, task_type, save_dir):
    pipeline = FARWanAnyFlowPipeline.from_pretrained(model_path).to('cuda', dtype=torch.bfloat16)
    os.makedirs(save_dir, exist_ok=True)

    if task_type == 't2v':
        # t2v inference:
        prompt = 'CG game concept digital art, a majestic elephant with a vibrant tusk and sleek fur running swiftly towards a herd of its kind. The elephant has a calm yet determined expression, with its ears flapping slightly as it moves at high speed. The herd consists of several other elephants of various ages and sizes, all moving in unison. The landscape is vast savanna with rolling hills, tall grasses, and scattered acacia trees. The sun sets behind the horizon, casting a warm golden glow over the scene. Low-angle view, focus on the elephant as it accelerates towards the herd.'  # noqa: E501
        video = pipeline(
            prompt=prompt,
            height=480,
            width=832,
            num_frames=81,
            num_inference_steps=4,
            generator=torch.Generator('cuda').manual_seed(0)
        ).frames[0]
        export_to_video(video, output_video_path=f'{save_dir}/demo_t2v.mp4', fps=16)
    elif task_type == 'ti2v':
        # ti2v inference:
        image_path = 'assets/evaluation/example/images/1.jpg'
        prompt = 'A towering, battle-scarred humanoid robot, reminiscent of a Transformer with powerful, segmented armor and glowing red optics, walking through the skeletal remains of a city ruin. Twisted metal and shattered concrete crunch under its heavy steps, as the robot scans the desolate, dust-choked skyline under an dark sky.'  # noqa: E501
        image = Image.open(image_path).convert('RGB')
        image = transforms.ToTensor()(transforms.Resize([480, 832])(image)).unsqueeze(0).unsqueeze(0)

        context_sequence, context_length = {'raw': image}, 1
        video = pipeline(
            prompt=prompt,
            context_sequence=context_sequence,
            height=480,
            width=832,
            num_frames=81,
            num_inference_steps=4,
            generator=torch.Generator('cuda').manual_seed(0)
        ).frames[0]
        video = draw_rectangle(video, context_length=context_length)
        export_to_video(video, output_video_path=f'{save_dir}/demo_ti2v.mp4', fps=16)
    elif task_type == 'tv2v':
        # tv2v inference:
        video_path = 'assets/evaluation/example/videos/2.mp4'
        prompt = "A focused trail runner's powerful strides through a dense, sun-dappled forest. The camera tracks alongside, highlighting muscular exertion, sweat, and determined facial expression. Golden light filters through the canopy, illuminating the immediate path and kicking up dust from their precise footfalls. The vibrant greens and browns of nature blur slightly as the runner accelerates."  # noqa: E501
        num_cond_frames = 25  # [1, 3, 3]: first is one frame, next is 3*4 frame = 3*4*2+1 = 25

        video_reader = decord.VideoReader(video_path)
        frame_idxs = select_frame_indices(len(video_reader), video_reader.get_avg_fps(), target_fps=16)[:num_cond_frames]
        frames = video_reader.get_batch(frame_idxs)
        frames = (frames / 255.0).float().permute(0, 3, 1, 2).contiguous()
        frames = transforms.Resize([480, 832])(frames).unsqueeze(0)

        context_sequence, context_length = {'raw': frames}, frames.shape[1]
        video = pipeline(
            prompt=prompt,
            context_sequence=context_sequence,
            height=480,
            width=832,
            num_frames=81,
            num_inference_steps=4,
            generator=torch.Generator('cuda').manual_seed(0)
        ).frames[0]
        video = draw_rectangle(video, context_length=context_length)
        export_to_video(video, output_video_path=f'{save_dir}/demo_tv2v.mp4', fps=16)
    else:
        raise NotImplementedError


def inference_bidirectional_demo(model_path, task_type, save_dir):
    pipeline = WanAnyFlowPipeline.from_pretrained(model_path).to('cuda', dtype=torch.bfloat16)
    os.makedirs(save_dir, exist_ok=True)

    if task_type == 't2v':
        # t2v inference:
        prompt = 'CG game concept digital art, a majestic elephant with a vibrant tusk and sleek fur running swiftly towards a herd of its kind. The elephant has a calm yet determined expression, with its ears flapping slightly as it moves at high speed. The herd consists of several other elephants of various ages and sizes, all moving in unison. The landscape is vast savanna with rolling hills, tall grasses, and scattered acacia trees. The sun sets behind the horizon, casting a warm golden glow over the scene. Low-angle view, focus on the elephant as it accelerates towards the herd.'  # noqa: E501
        video = pipeline(
            prompt=prompt,
            height=480,
            width=832,
            num_frames=81,
            num_inference_steps=4,
            generator=torch.Generator('cuda').manual_seed(0)
        ).frames[0]
        export_to_video(video, output_video_path=f'{save_dir}/demo_t2v.mp4', fps=16)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    cfg: DemoConfig = OmegaConf.merge(
        OmegaConf.structured(DemoConfig),
        OmegaConf.from_cli(),
    )
    if 'AnyFlow-FAR' in cfg.model_path:
        inference_causal_demo(cfg.model_path, task_type=cfg.task_type, save_dir=cfg.save_dir)
    elif 'AnyFlow-Wan' in cfg.model_path:
        inference_bidirectional_demo(cfg.model_path, task_type=cfg.task_type, save_dir=cfg.save_dir)
    else:
        raise NotImplementedError

"""
Causal FAR AnyFlow demo (loads `FARWanAnyFlowPipeline.from_pretrained`).

CLI variables:
  model_path   — Diffusers folder path.
  task_type    — t2v | ti2v | tv2v (default: t2v).
  save_dir     — Output directory for demo_*.mp4.

Example (from repository root):
python demo.py \
    model_path=experiments/pretrained_models/AnyFlow-Wan2.1-T2V-1.3B-Diffusers \
    task_type=t2v \
    save_dir=results/demo_v0.9/AnyFlow-Wan2.1-T2V-1.3B-Diffusers
"""
