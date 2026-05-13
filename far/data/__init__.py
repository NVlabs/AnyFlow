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
from copy import deepcopy
from os import path as osp

from far.data.dataloader import BaseDataLoader
from far.utils.misc import scandir
from far.utils.registry import DATASET_REGISTRY

__all__ = ['build_dataset', 'IterativeDataLoader']

# automatically scan and import dataset modules for registry
# scan all the files under the data folder with '_dataset' in file names
data_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(data_folder) if v.endswith('_dataset.py')]
# import all the dataset modules
_dataset_modules = [importlib.import_module(f'far.data.{file_name}') for file_name in dataset_filenames]


def build_dataset(dataset_opt):
    """Build dataset from options.

    Args:
        dataset_opt (dict): Configuration for dataset. It must contain:
            name (str): Dataset name.
            type (str): Dataset type.
    """
    dataset_opt = deepcopy(dataset_opt)
    dataset_type = dataset_opt.pop('type')
    dataset = DATASET_REGISTRY.get(dataset_type)(**dataset_opt)
    return dataset


def build_dataloader(dataset_opt, shuffle=False, drop_last=False, num_workers=8):
    dataloader_cfg = dataset_opt.pop('dataloader_cfg')
    dataset = build_dataset(dataset_opt)
    return BaseDataLoader(dataset, dataloader_cfg)
