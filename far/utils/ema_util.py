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
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

from far.utils.dist_util import is_main_process


class ShardEMA(nn.Module):
    def __init__(self, fsdp_model, decay=0.999, warmup_steps=0):
        super().__init__()
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.shadow = {}

        for name, param in fsdp_model.named_parameters():
            if param.requires_grad:
                self.register_buffer(name.replace('.', '_'), param.detach().clone().float().cpu())

    def single_state_dict(self, fsdp_model):
        self.store(fsdp_model)
        self.copy_to(fsdp_model)

        state_dict = get_model_state_dict(
            fsdp_model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )

        self.restore(fsdp_model)

        if is_main_process():
            return state_dict
        else:
            return {}

    def state_dict(self):
        return super().state_dict()

    def load_state_dict(self, state_dict: dict):
        super().load_state_dict(state_dict)

    @torch.no_grad()
    def store(self, fsdp_model):
        self.original = {name: param.detach().clone().cpu() for name, param in fsdp_model.named_parameters()}

    @torch.no_grad()
    def restore(self, fsdp_model):
        if self.original is None:
            raise RuntimeError('Must call `store()` before `restore()`.')

        for name, param in fsdp_model.named_parameters():
            param.copy_(self.original[name].to(device=param.device, dtype=param.dtype))

        self.original = None

    def get_decay(self, step: int) -> float:
        if step >= self.warmup_steps:
            return self.decay
        else:
            return 0

    @torch.no_grad()
    def step(self, fsdp_model, step):
        decay = self.get_decay(step)

        for name, param in fsdp_model.named_parameters():
            if name.replace('.', '_') in self._buffers:
                self.get_buffer(name.replace('.', '_')).mul_(decay).add_(param.detach().float().cpu(), alpha=1. - decay)

    @torch.no_grad()
    def copy_to(self, fsdp_model):
        for name, param in fsdp_model.named_parameters():
            if name.replace('.', '_') in self._buffers:
                param.copy_(self.get_buffer(name.replace('.', '_')).to(device=param.device, dtype=param.dtype))
