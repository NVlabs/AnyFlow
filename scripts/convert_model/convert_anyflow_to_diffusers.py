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

from omegaconf import MISSING, OmegaConf

import decord
import torch
from diffusers.utils import export_to_video
from PIL import Image
from torchvision import transforms

from far.models.transformer_far_wan_model import FAR_Wan_Transformer3DModel
from far.pipelines.pipeline_far_wan_anyflow import FARWanAnyFlowPipeline
from far.pipelines.pipeline_wan_anyflow import WanAnyFlowPipeline
from far.schedulers.scheduling_flowmap_euler_discrete import FlowMapDiscreteScheduler
from far.utils.video_util import select_frame_indices
from far.utils.vis_util import draw_rectangle

decord.bridge.set_bridge('torch')


@dataclass
class ConvertAnyflowToDiffusersConfig:
    """CLI keys for `python -m scripts.convert_model.convert_anyflow_to_diffusers key=value ...`."""

    # Causal: AnyFlow-FAR-Wan2.1-1.3B-Diffusers | AnyFlow-FAR-Wan2.1-14B-Diffusers
    # Bidirectional: AnyFlow-Wan2.1-T2V-1.3B-Diffusers | AnyFlow-Wan2.1-T2V-14B-Diffusers
    model_type: str = 'AnyFlow-FAR-Wan2.1-1.3B-Diffusers'
    # Path to the AnyFlow .pt checkpoint (expects an `ema` entry in the dict).
    model_path: str = MISSING
    # Output directory for `pipeline.save_pretrained`.
    model_save_dir: str = MISSING


def build_causal_pipeline(model_type, model_path):

    far_config = {
        'full_chunk_limit': 3,
        'chunk_partition': [1, 3, 3, 3, 3, 3, 3, 2],
        'compressed_patch_size': [1, 4, 4]
    }

    if model_type == 'AnyFlow-FAR-Wan2.1-1.3B-Diffusers':
        base_model_name = 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers'
    elif model_type == 'AnyFlow-FAR-Wan2.1-14B-Diffusers':
        base_model_name = 'Wan-AI/Wan2.1-T2V-14B-Diffusers'
    else:
        raise NotImplementedError

    transformer = FAR_Wan_Transformer3DModel.from_pretrained(
        base_model_name,
        chunk_partition=far_config['chunk_partition'],
        full_chunk_limit=far_config['full_chunk_limit'],
        compressed_patch_size=far_config['compressed_patch_size'],
        subfolder='transformer'
    )
    transformer.setup_far_model()
    transformer.setup_flowmap_model(gate_value=0.25, deltatime_type="r")
    transformer.register_to_config(init_far_model=True, init_flowmap_model=True, deltatime_type='r', gate_value=0.25)

    # load model
    state_dict = torch.load(model_path)['ema']
    transformer.load_state_dict(state_dict)
    transformer = transformer.to('cuda', dtype=torch.bfloat16)

    scheduler = FlowMapDiscreteScheduler(shift=5, num_train_timesteps=1000)

    pipeline = FARWanAnyFlowPipeline.from_pretrained(base_model_name, transformer=transformer, scheduler=scheduler)
    pipeline.to('cuda')
    
    return pipeline


def build_bidirectional_pipeline(model_type='AnyFlow-Wan2.1-T2V-1.3B-Diffusers', model_path=None):

    if model_type == 'AnyFlow-Wan2.1-T2V-1.3B-Diffusers':
        base_model_name = 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers'
    elif model_type == 'AnyFlow-Wan2.1-T2V-14B-Diffusers':
        base_model_name = 'Wan-AI/Wan2.1-T2V-14B-Diffusers'
    else:
        raise NotImplementedError

    transformer = FAR_Wan_Transformer3DModel.from_pretrained(base_model_name, subfolder='transformer')
    transformer.setup_flowmap_model(gate_value=0.25, deltatime_type='r')
    transformer.register_to_config(init_flowmap_model=True, deltatime_type='r', gate_value=0.25)

    # load model
    state_dict = torch.load(model_path)['ema']
    transformer.load_state_dict(state_dict)
    transformer = transformer.to('cuda', dtype=torch.bfloat16)

    scheduler = FlowMapDiscreteScheduler(shift=5, num_train_timesteps=1000)

    pipeline = WanAnyFlowPipeline.from_pretrained(base_model_name, transformer=transformer, scheduler=scheduler)
    pipeline.to('cuda')
    
    return pipeline


if __name__ == '__main__':
    cfg: ConvertAnyflowToDiffusersConfig = OmegaConf.merge(
        OmegaConf.structured(ConvertAnyflowToDiffusersConfig),
        OmegaConf.from_cli(),
    )

    cfg.model_save_dir = os.path.join(cfg.model_save_dir, cfg.model_type)

    os.makedirs(cfg.model_save_dir, exist_ok=True)

    if 'FAR' in cfg.model_type:
        pipeline = build_causal_pipeline(cfg.model_type, cfg.model_path)
    else:
        pipeline = build_bidirectional_pipeline(cfg.model_type, model_path=cfg.model_path)

    pipeline.save_pretrained(cfg.model_save_dir)

"""
Convert AnyFlow checkpoint weights into a Diffusers pipeline on disk.

CLI variables:
  model_type       — Causal: AnyFlow-FAR-Wan2.1-1.3B-Diffusers, AnyFlow-FAR-Wan2.1-14B-Diffusers;
                     Bidirectional: AnyFlow-Wan2.1-T2V-1.3B-Diffusers, AnyFlow-Wan2.1-T2V-14B-Diffusers.
  model_path       — Input .pt checkpoint (lora-merged state_dict and contains `ema`).
  model_save_dir   — Output directory for `pipeline.save_pretrained` (required).

Example:
python -m scripts.convert_model.convert_anyflow_to_diffusers \
    model_type=AnyFlow-Wan2.1-T2V-14B-Diffusers \
    model_path=experiments/pretrained_models/AnyFlow_Demo/anyflow_v1.0/anyflow-wan-14b.pt \
    model_save_dir=experiments/pretrained_models/AnyFlow-Diffusers-V1.0/
"""