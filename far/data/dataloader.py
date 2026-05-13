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

from dataclasses import dataclass, replace
from typing import Optional

import torch
from torch.utils.data import DataLoader, default_collate

from far.data.sampler import DistributedRangedSampler
from far.utils.dist_util import get_dist_rank, get_world_size


@dataclass
class BaseDataLoaderConfig:
    batch_size_per_gpu: int = 32
    num_workers: int = 0
    drop_last: bool = False
    seed: int = 0
    shuffle: bool = False
    prefetch_factor: Optional[int] = None
    persistent_workers: bool = False
    shuffle_chunk_size: Optional[int] = None


class BaseDataLoader:
    def __init__(self, dataset, dataloader_cfg):
        self.cfg = replace(BaseDataLoaderConfig(), **dataloader_cfg)

        self.world_size = get_world_size()
        self.rank = get_dist_rank()

        self.dataset = dataset
        self.sampler = self.build_sampler()
        self.data_loader = self.build_data_loader()

        self.data_loader_iter = iter(self.data_loader)

    def build_sampler(self):
        sampler = DistributedRangedSampler(
            self.dataset,
            self.world_size,
            self.rank,
            shuffle=self.cfg.shuffle,
            seed=self.cfg.seed,
            drop_last=self.cfg.drop_last,
            shuffle_chunk_size=self.cfg.shuffle_chunk_size,
        )
        return sampler

    def collate_fn(self, batch):
        return default_collate(batch)

    def build_data_loader(self):
        generator = torch.Generator()
        generator.manual_seed(self.cfg.seed)
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.cfg.batch_size_per_gpu,
            sampler=self.sampler,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=self.cfg.drop_last,
            collate_fn=self.collate_fn,
            generator=generator,
            prefetch_factor=self.cfg.prefetch_factor,
            persistent_workers=self.cfg.persistent_workers,
        )

    def set_state(self, epoch: int, batch_index: int) -> None:
        self.sampler.set_epoch(epoch)
        self.sampler.set_iter_index(batch_index * self.cfg.batch_size_per_gpu)
        self.data_loader_iter = iter(self.data_loader)

    def get_next_batch(self):
        try:
            batch = next(self.data_loader_iter)
        except StopIteration:
            self.sampler.set_epoch(self.sampler.epoch + 1)
            self.data_loader_iter = iter(self.data_loader)
            batch = next(self.data_loader_iter)
        return batch
