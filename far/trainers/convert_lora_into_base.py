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
from collections import OrderedDict
from dataclasses import dataclass

import torch
from omegaconf import MISSING, OmegaConf


@dataclass
class ConvertLoRAIntoBaseConfig:
    load_path: str = MISSING
    save_path: str = MISSING
    lora_type: str = 'LoRA'
    weight_type: str = 'ema'


def merge_lora_weights(lora_state_dict, rank=256, alpha=256, lora_type='LoRA'):
    lora_pairs = OrderedDict()
    merged_state_dict = {}

    for key in list(lora_state_dict.keys()):
        if 'base_layer' in key:
            lora_A_key = key.replace('base_layer', 'lora_A.real')
            lora_B_key = key.replace('base_layer', 'lora_B.real')

            if lora_A_key in lora_state_dict:
                assert lora_B_key in lora_state_dict, 'Incomplete triple (BaseLayer, LoRA_A, LoRA_B)'
                lora_pairs[key] = (lora_A_key, lora_B_key)
            else:
                merged_state_dict[key.replace('.base_layer', '')] = lora_state_dict[key]
        elif 'lora_A' in key or 'lora_B' in key or 'lora_magnitude_vector' in key:
            continue
        else:
            merged_state_dict[key] = lora_state_dict[key]

    for base_key, (lora_A_key, lora_B_key) in lora_pairs.items():
        base_weight = lora_state_dict[base_key]
        lora_A = lora_state_dict[lora_A_key]
        lora_B = lora_state_dict[lora_B_key]

        if len(base_weight.shape) == 1:  # Deal with Bias
            merged_weight = base_weight + float(alpha) / rank * (lora_A @ lora_B).squeeze()
        else:  # Deal with Weight
            delta_W = lora_B @ lora_A

            if lora_type == 'LoRA':
                merged_weight = base_weight + float(alpha) / rank * delta_W
            else:
                raise NotImplementedError(f'LoRA type {lora_type} is not supported.')

        merged_state_dict[base_key.replace('.base_layer', '')] = merged_weight

    merged_state_dict = {k.replace('base_model.model.', ''): v for k, v in merged_state_dict.items()}

    return merged_state_dict


if __name__ == '__main__':
    cfg: ConvertLoRAIntoBaseConfig = OmegaConf.merge(OmegaConf.structured(ConvertLoRAIntoBaseConfig), OmegaConf.from_cli())

    checkpoint = torch.load(cfg.load_path)[cfg.weight_type]

    nonlora_checkpoint = merge_lora_weights(checkpoint, rank=256, alpha=256.0, lora_type='LoRA')
    os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)
    torch.save({cfg.weight_type: nonlora_checkpoint}, cfg.save_path)

"""
python -m far.trainers.convert_lora_into_base \
    load_path=experiments/pretrained_models/final_model_v1.0/3011_wan1b_teacher_fusionx_shift5_81f_480p_lr5e-5_8k_b32/step_6000.pt \
    save_path=experiments/pretrained_models/final_model_v1.0/3011_wan1b_teacher_fusionx_shift5_81f_480p_lr5e-5_8k_b32/step_6000_lora_merged_ema.pt \
    weight_type='ema'
"""
