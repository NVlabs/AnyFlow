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
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, OffloadPolicy, fully_shard


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def dist_init() -> None:

    if is_dist_initialized():
        return
    try:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        host = os.environ['MASTER_ADDR']
        port = int(os.environ['MASTER_PORT'])

        if ':' in host:  # IPv6
            init_method = f'tcp://[{host}]:{port}'
        else:  # IPv4
            init_method = f'tcp://{host}:{port}'

        dist.init_process_group(rank=rank, world_size=world_size, backend='nccl', init_method=init_method, timeout=timedelta(minutes=30))

        torch.cuda.set_device(local_rank)
        assert torch.distributed.is_initialized()
    except Exception:
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['LOCAL_RANK'] = '0'
        print('warning: dist not init')


def fsdp2_wrap(
    module: nn.Module,
    sharding_strategy: str = 'hybrid_full',
    mixed_precision: bool = True,
    cpu_offload: bool = False,
    transformer_block_clsname: str = None,
    sync_module_state: bool = True
):
    if sync_module_state:
        full_state_dict = module.state_dict() if dist.get_rank() == 0 else {}

    # build mixed precision policy
    if mixed_precision:
        mixed_precision_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            cast_forward_inputs=False,
        )
    else:
        mixed_precision_policy = MixedPrecisionPolicy()

    # build offload policy
    offload_policy = CPUOffloadPolicy() if cpu_offload else OffloadPolicy()

    # build device mesh
    world_size = int(os.environ['WORLD_SIZE'])
    gpus_per_node = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
    num_nodes = world_size // gpus_per_node

    if sharding_strategy == 'hybrid_full' or sharding_strategy == 'hybrid_zero2':
        device_mesh = init_device_mesh(
            'cuda',
            (num_nodes, gpus_per_node),
            mesh_dim_names=('replication', 'sharding')
        )
    else:
        device_mesh = init_device_mesh('cuda', (world_size,))

    if transformer_block_clsname is not None:
        for m in module.modules():
            if type(m).__name__ == transformer_block_clsname:
                fully_shard(m, mesh=device_mesh, reshard_after_forward=True, mp_policy=mixed_precision_policy, offload_policy=offload_policy)
    fully_shard(module, mesh=device_mesh, reshard_after_forward=True, mp_policy=mixed_precision_policy, offload_policy=offload_policy)

    if sync_module_state:
        options = StateDictOptions(full_state_dict=True, cpu_offload=True, broadcast_from_rank0=True)
        set_model_state_dict(module, full_state_dict, options=options)

    return module


def dist_barrier():
    if dist.is_initialized():
        dist.barrier()


def get_dist_rank() -> int:
    return int(os.environ.get('RANK', 0))


def get_world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    else:
        return 1


def is_dist_initialized() -> bool:
    return torch.distributed.is_initialized()


def destroy_process_group():
    if is_dist_initialized():
        dist_barrier()
        torch.distributed.destroy_process_group()


def check_video(fp):
    import decord
    try:
        return os.path.isfile(fp) and len(decord.VideoReader(fp)) > 0 and decord.VideoReader(fp)[0] is not None
    except:
        return False


def all_ranks_path_exists(save_path):
    local_exists = check_video(save_path)
    res_tensor = torch.tensor(1.0 if local_exists else 0.0).cuda()
    dist.all_reduce(res_tensor, op=dist.ReduceOp.SUM)

    world_size = dist.get_world_size()
    return res_tensor.item() == world_size


def reduce_loss(loss):
    rt = loss.detach().clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt.item()
