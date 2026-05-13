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

from typing import Optional

import torch
from torch.utils.data import Dataset, Sampler

__all__ = ['DistributedRangedSampler']


class DistributedRangedSampler(Sampler):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        num_samples: Optional[int] = None,
        shuffle_chunk_size: Optional[int] = None,
    ):
        assert rank >= 0 and rank < num_replicas
        self.num_samples = num_samples if num_samples is not None else len(dataset)
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        if drop_last:
            self.num_samples_per_rank = self.num_samples // num_replicas
        else:
            self.num_samples_per_rank = (self.num_samples - 1) // num_replicas + 1
        self.shuffle_chunk_size = shuffle_chunk_size

        if shuffle_chunk_size is not None:
            start = self.rank * self.num_samples_per_rank
            end = (self.rank + 1) * self.num_samples_per_rank
            self.ranges = [(i, min(i + shuffle_chunk_size, end)) for i in range(start, end, shuffle_chunk_size)]

        self.epoch = 0
        self.iter_index = 0

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.iter_index = 0

    def set_iter_index(self, iter_index):
        self.iter_index = iter_index

    def __len__(self):
        return self.num_samples_per_rank

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.num_samples, generator=g).tolist()
            if not self.drop_last:
                total_size = self.num_replicas * self.num_samples_per_rank
                padding_size = total_size - len(indices)
                indices += (indices * ((padding_size - 1) // len(indices) + 1))[:padding_size]
            indices = indices[self.rank * self.num_samples_per_rank: (self.rank + 1) * self.num_samples_per_rank]
            assert len(indices) == self.num_samples_per_rank
            yield from indices[self.iter_index:]
        elif self.shuffle_chunk_size is not None:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            shard_indices = torch.randperm(len(self.ranges), generator=g).tolist()
            cnt = 0
            for shard_index in shard_indices:
                shard_start, shard_end = self.ranges[shard_index]
                num_shard_samples = shard_end - shard_start
                sample_indices = (
                    (torch.randperm(num_shard_samples, generator=g) + shard_start) % self.num_samples
                ).tolist()
                if cnt + num_shard_samples <= self.iter_index:
                    cnt += num_shard_samples
                    continue
                yield from sample_indices[max(self.iter_index - cnt, 0):]
                cnt += num_shard_samples
        else:
            start = self.rank * self.num_samples_per_rank + self.iter_index
            end = (self.rank + 1) * self.num_samples_per_rank
            indices = (torch.arange(self.num_replicas * self.num_samples_per_rank) % self.num_samples).tolist()
            yield from indices[start:end]
