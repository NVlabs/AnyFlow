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

import importlib
from os import path as osp

from far.utils.misc import scandir
from far.utils.registry import TRAINER_REGISTRY

__all__ = ['build_trainer']

# automatically scan and import model modules for registry
# scan all the files under the 'models' folder and collect files ending with
# '_model.py'
trainer_folder = osp.dirname(osp.abspath(__file__))
trainer_filenames = [
    osp.splitext(osp.basename(v))[0] for v in scandir(trainer_folder)
    if v.startswith('trainer_')
]
# import all the model modules
_trainer_modules = [
    importlib.import_module(f'far.trainers.{file_name}')
    for file_name in trainer_filenames
]


def build_trainer(trainer_type):
    """Build model from options.

    Args:
        opt (dict): Configuration. It must contain:
            model_type (str): Model type.
    """
    trainer = TRAINER_REGISTRY.get(trainer_type)
    return trainer
