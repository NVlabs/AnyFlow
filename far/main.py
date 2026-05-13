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

import gc
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.distributed
import torch.utils.checkpoint
from omegaconf import OmegaConf
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, get_state_dict, set_state_dict
from transformers import get_constant_schedule_with_warmup

from far.data import build_dataloader
from far.trainers import build_trainer
from far.utils.dist_util import destroy_process_group, dist_barrier, dist_init, get_dist_rank, get_world_size, is_main_process
from far.utils.logger_util import MessageLogger, dict2str, get_logger, set_path_logger, setup_wandb


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


class BaseTrainer:

    def __init__(self, cfg):
        self.cfg = cfg

        self.setup_dist_env()
        self.setup_seed()

        # Set experiment directory and initialize logger
        set_path_logger(self.cfg)
        get_logger().info(dict2str(self.cfg))

        # Setup trainer pipeline based on mode
        if self.cfg['mode'] in ['train', 'eval']:
            self.setup_model()
            self.build_eval_dataloader()

            self.wandb_logger = None
            self.global_step = 'final'

        if self.cfg['mode'] == 'train':
            self.setup_wandb()

            self.setup_ema_model()
            self.build_train_dataloader()

            self.setup_optimizer()
            self.setup_lr_scheduler()

            self.global_step = 0
            self.resume_checkpoint()

            self.msg_logger = MessageLogger(self.cfg, self.global_step)

    def evaluate(self, compute_metric=True):
        """Evaluate the model on validation and sample datasets."""
        if self.sample_dataloader is not None:
            self.train_pipeline.sample(self.sample_dataloader, self.cfg, global_step=self.global_step)

        if self.i2v_sample_dataloader is not None:
            self.train_pipeline.sample(self.i2v_sample_dataloader, self.cfg, global_step=self.global_step)

        if self.v2v_sample_dataloader is not None:
            self.train_pipeline.sample(self.v2v_sample_dataloader, self.cfg, global_step=self.global_step)

        dist_barrier()
        torch.cuda.empty_cache()
        gc.collect()
        dist_barrier()

        if compute_metric and self.val_dataloader is not None:
            get_logger().info(f'Begin generating {len(self.val_dataloader.dataset)} evaluation videos for vbench')
            self.train_pipeline.validate(self.val_dataloader, self.cfg, global_step=self.global_step)
            dist_barrier()

            if 'eval_cfg' in self.cfg['val']:
                eval_info_dict = self.train_pipeline.eval_performance(self.cfg, global_step=self.global_step)
                get_logger().info(f'Step-{self.global_step} evaluation results: {eval_info_dict}')

                if self.wandb_logger:
                    wandb_log_dict = {f'eval/{k}': v for k, v in eval_info_dict.items()}
                    self.wandb_logger.log(wandb_log_dict, step=self.global_step, commit=True)

        dist_barrier()
        torch.cuda.empty_cache()
        gc.collect()
        dist_barrier()

        if compute_metric and self.i2v_val_dataloader is not None:
            get_logger().info(f'Begin generating {len(self.i2v_val_dataloader.dataset)} evaluation videos for vbench i2v')
            self.train_pipeline.validate_i2v(self.i2v_val_dataloader, self.cfg, global_step=self.global_step)
            dist_barrier()

            if 'eval_cfg' in self.cfg['val']:
                eval_info_dict = self.train_pipeline.eval_i2v_performance(self.cfg, global_step=self.global_step)
                get_logger().info(f'Step-{self.global_step} evaluation results: {eval_info_dict}')

                if self.wandb_logger:
                    wandb_log_dict = {f'eval/{k}': v for k, v in eval_info_dict.items()}
                    self.wandb_logger.log(wandb_log_dict, step=self.global_step, commit=True)

        dist_barrier()
        torch.cuda.empty_cache()
        gc.collect()
        dist_barrier()

    def train(self):
        """Train the model."""
        get_logger().info('***** Running training *****')
        get_logger().info(f'  Num examples = {len(self.train_dataloader.dataset)}')
        get_logger().info(f'  Instantaneous batch size per device = {self.train_dataloader.cfg.batch_size_per_gpu}')
        get_logger().info(f'  Total train batch size (with parallel, distributed & accumulation) = {self.train_dataloader.cfg.batch_size_per_gpu * get_world_size()}')  # noqa: E501
        get_logger().info(f"  Total optimization steps = {self.cfg['train']['total_iter']}")

        self.init_train_dataloader()

        while self.global_step <= self.cfg['train']['total_iter']:

            # Evaluation
            if self.global_step != 0 and self.global_step % self.cfg['logger']['save_eval_checkpoint_freq'] == 0:
                self.save_checkpoint(f'step_{self.global_step}', only_model_state_dict=True)

            if self.global_step % self.cfg['val']['sample_freq'] == 0 and not (self.global_step == 0 and self.cfg['val'].get('skip_first_eval', False)):
                if self.global_step % self.cfg['val']['eval_freq'] == 0 and self.global_step != 0:
                    compute_metric = True
                else:
                    compute_metric = False
                self.evaluate(compute_metric)

            # Training step
            log_dict = {}

            step_begin_time = time.time()

            """Start of an iteration"""
            train_generator = self.global_step % self.cfg['train'].get('discriminator_update_ratio', 1) == 0

            # train generator
            if train_generator:
                batch = self.train_dataloader.get_next_batch()
                self.optimizer_g.zero_grad(set_to_none=True)

                g_loss_dict = self.train_pipeline.train_step(batch, iters=self.global_step, train_generator=True)
                if 'max_grad_norm' in self.cfg['train']:
                    generator_grad_norm = torch.nn.utils.clip_grad_norm_(self.train_pipeline.transformer.parameters(), max_norm=self.cfg['train']['max_grad_norm']).item()  # noqa: E501
                    g_loss_dict['generator_grad_norm'] = generator_grad_norm

                self.optimizer_g.step()
                self.lr_scheduler_g.step()

                if self.cfg['train'].get('ema_decay'):
                    self.train_pipeline.ema.step(self.train_pipeline.transformer, step=self.global_step)
            else:
                g_loss_dict = {}

            if hasattr(self, 'optimizer_d'):
                # train discrminator
                batch = self.train_dataloader.get_next_batch()
                self.optimizer_d.zero_grad(set_to_none=True)

                d_loss_dict = self.train_pipeline.train_step(batch, iters=self.global_step, train_generator=False)
                if 'max_grad_norm' in self.cfg['train']:
                    discriminator_grad_norm = torch.nn.utils.clip_grad_norm_(self.train_pipeline.discriminator.parameters(), max_norm=self.cfg['train']['max_grad_norm']).item()  # noqa: E501
                    d_loss_dict['discriminator_grad_norm'] = discriminator_grad_norm

                self.optimizer_d.step()
                self.lr_scheduler_d.step()

            """End of an iteration"""

            step_end_time = time.time()
            log_dict['step_time'] = step_end_time - step_begin_time

            # Increment global step
            self.global_step += 1

            # Log periodically
            if self.global_step % self.cfg['logger']['print_freq'] == 0:
                log_dict.update({'iter': self.global_step})
                log_dict.update({'lrs_g': self.lr_scheduler_g.get_last_lr()[0]})
                log_dict.update(g_loss_dict)

                if hasattr(self, 'optimizer_d'):
                    log_dict.update({'lrs_d': self.lr_scheduler_d.get_last_lr()[0]})
                    log_dict.update(d_loss_dict)

                if self.cfg['train'].get('use_same_optimizer'):
                    log_dict.update(d_loss_dict)

                self.msg_logger(log_dict)

                if is_main_process() and self.wandb_logger:
                    wandb_log_dict = {f'train/{k}': v for k, v in log_dict.items()}
                    self.wandb_logger.log(wandb_log_dict, step=self.global_step)

            # Save checkpoint periodically
            if self.global_step % self.cfg['logger']['save_checkpoint_freq'] == 0:
                self.save_checkpoint('checkpoint')

            if self.global_step % self.cfg['logger'].get('clean_cache_freq', 10) == 0:
                dist_barrier()
                torch.cuda.empty_cache()
                gc.collect()
                dist_barrier()

    def setup_optimizer(self):
        """Setup the optimizer."""
        optim_g_type = self.cfg['train']['optim_g'].pop('type')

        if optim_g_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(
                [param for param in self.train_pipeline.transformer.parameters() if param.requires_grad],
                **self.cfg['train']['optim_g']
            )
        else:
            raise ValueError(f'Unsupported optimizer type: {optim_g_type}')

        if 'optim_d' in self.cfg['train']:
            optim_d_type = self.cfg['train']['optim_d'].pop('type')

            if optim_d_type == 'AdamW':
                self.optimizer_d = torch.optim.AdamW(
                    [param for param in self.train_pipeline.discriminator.parameters() if param.requires_grad],
                    **self.cfg['train']['optim_d']
                )
            else:
                raise ValueError(f'Unsupported optimizer type: {optim_d_type}')

    def setup_lr_scheduler(self):
        """Setup the learning rate scheduler."""
        num_warmup_steps = self.cfg['train']['warmup_iter']

        if self.cfg['train']['lr_scheduler'] == 'constant_with_warmup':
            self.lr_scheduler_g = get_constant_schedule_with_warmup(
                optimizer=self.optimizer_g,
                num_warmup_steps=num_warmup_steps,
            )
            if hasattr(self, 'optimizer_d'):
                self.lr_scheduler_d = get_constant_schedule_with_warmup(
                    optimizer=self.optimizer_d,
                    num_warmup_steps=num_warmup_steps,
                )
        else:
            raise NotImplementedError

    def build_train_dataloader(self):
        """Build the training dataloader."""
        trainset_cfg = self.cfg['datasets']['train']
        self.train_dataloader = build_dataloader(trainset_cfg)

    def build_eval_dataloader(self):
        """Build the evaluation and sample dataloaders."""
        if self.cfg['datasets'].get('val'):
            valset_cfg = self.cfg['datasets']['val']
            self.val_dataloader = build_dataloader(valset_cfg).data_loader
        else:
            self.val_dataloader = None

        if self.cfg['datasets'].get('val_i2v'):
            i2v_valset_cfg = self.cfg['datasets']['val_i2v']
            self.i2v_val_dataloader = build_dataloader(i2v_valset_cfg).data_loader
        else:
            self.i2v_val_dataloader = None

        if self.cfg['datasets'].get('sample'):
            sampleset_cfg = self.cfg['datasets']['sample']
            self.sample_dataloader = build_dataloader(sampleset_cfg).data_loader
        else:
            self.sample_dataloader = None

        if self.cfg['datasets'].get('sample_i2v'):
            i2v_sampleset_cfg = self.cfg['datasets']['sample_i2v']
            self.i2v_sample_dataloader = build_dataloader(i2v_sampleset_cfg).data_loader
        else:
            self.i2v_sample_dataloader = None

        if self.cfg['datasets'].get('sample_v2v'):
            v2v_sampleset_cfg = self.cfg['datasets']['sample_v2v']
            self.v2v_sample_dataloader = build_dataloader(v2v_sampleset_cfg).data_loader
        else:
            self.v2v_sample_dataloader = None

    def init_train_dataloader(self):
        data_loader_length = len(self.train_dataloader.data_loader)
        get_logger().info(f'Data_loader_length: {data_loader_length}')
        epoch = self.global_step // data_loader_length
        batch_iter = self.global_step - data_loader_length * epoch

        self.train_dataloader.set_state(epoch, batch_iter)
        get_logger().info(f'Init dataloader at epoch: {epoch}, batch_iter {batch_iter}')

    def setup_wandb(self):
        """Setup Weights & Biases logging."""
        if is_main_process() and self.cfg['logger'].get('use_wandb', False):
            self.wandb_logger = setup_wandb(name=self.cfg['name'], save_dir=self.cfg['path']['log'])
        else:
            self.wandb_logger = None

    def setup_model(self):
        """Setup the model."""
        self.train_pipeline = build_trainer(self.cfg['train']['train_pipeline'])(**self.cfg['models'], device=self.device, dtype=self.dtype)

    def setup_ema_model(self):
        """Setup the EMA model (if applicable)."""
        # Setup EMA (Exponential Moving Average) if specified
        if self.cfg['train'].get('ema_decay'):
            self.train_pipeline.set_ema_model(
                ema_decay=self.cfg['train']['ema_decay'],
                ema_warmup_step=self.cfg['train'].get('ema_warmup_step', 0)
            )

    def setup_dist_env(self):
        """Initialize the distributed environment."""
        dist_init()

        # Set deterministic and benchmark flags for CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Set dtype and device based on configuration
        self.dtype = torch.bfloat16 if self.cfg['mixed_precision'] else torch.float32
        self.device = torch.device('cuda', torch.cuda.current_device())

        # Get distributed rank and world size
        self.rank, self.world_size = get_dist_rank(), get_world_size()

    def setup_seed(self) -> None:
        """Set the random seeds for reproducibility."""
        seed = self.rank + self.cfg['manual_seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def get_random_states(self):
        """Return the random states for reproducibility."""
        random_state = random.getstate()
        numpy_random_state = np.random.get_state()
        numpy_random_state = (
            numpy_random_state[0],
            numpy_random_state[1].tolist(),
            numpy_random_state[2],
            numpy_random_state[3],
            numpy_random_state[4],
        )
        torch_rng_state = torch.get_rng_state()
        torch_cuda_rng_state = torch.cuda.get_rng_state()

        random_states_single_rank = {
            'random_state': random_state,
            'numpy_random_state': numpy_random_state,
            'torch_rng_state': torch_rng_state,
            'torch_cuda_rng_state': torch_cuda_rng_state
        }

        random_states = {f'rank_{self.rank}': random_states_single_rank}
        return random_states

    def set_random_states(self, random_states):
        """Restore the random states from a previous checkpoint."""
        random_states_single_rank = random_states[f'rank_{self.rank}']
        random.setstate(random_states_single_rank['random_state'])
        np.random.set_state(random_states_single_rank['numpy_random_state'])
        torch.set_rng_state(random_states_single_rank['torch_rng_state'])
        torch.cuda.set_rng_state(random_states_single_rank['torch_cuda_rng_state'])

    def get_train_state(self, only_model_state_dict: bool = False):
        """Return the current training state, optionally including only the model state."""
        train_states = {}

        train_states['global_step'] = self.global_step
        train_states['world_size'] = self.world_size

        if self.cfg['train'].get('ema_decay'):
            if only_model_state_dict:
                train_states['ema'] = self.train_pipeline.ema.single_state_dict(self.train_pipeline.transformer)
            else:
                train_states['ema'] = self.train_pipeline.ema.state_dict()

        if only_model_state_dict:
            train_states['model_state_dict_g'] = get_model_state_dict(self.train_pipeline.transformer, options=StateDictOptions(cpu_offload=True))
            return train_states

        train_states['model_state_dict_g'], train_states['optimizer_state_dict_g'] = get_state_dict(
            self.train_pipeline.transformer, self.optimizer_g, options=StateDictOptions(cpu_offload=True)
        )
        train_states['lr_scheduler_g'] = self.lr_scheduler_g.state_dict()

        if hasattr(self, 'optimizer_d'):
            train_states['model_state_dict_d'], train_states['optimizer_state_dict_d'] = get_state_dict(
                self.train_pipeline.discriminator, self.optimizer_d, options=StateDictOptions(cpu_offload=True)
            )
            train_states['lr_scheduler_d'] = self.lr_scheduler_d.state_dict()

        train_states['random_states'] = self.get_random_states()

        return train_states

    def resume_checkpoint(self):
        """Resume training from the last checkpoint, if available."""
        if self.cfg['path'].get('pretrain_network', None):
            load_path = self.cfg['path'].get('pretrain_network')
        else:
            load_path = self.cfg['path']['models']

        checkpoint_path = os.path.join(load_path, 'checkpoint')
        if os.path.exists(checkpoint_path):
            get_logger().info(f'Resuming from checkpoint {checkpoint_path}')

            # ----------------modify this part for different trainer---------------
            train_states = self.get_train_state()
            torch.distributed.checkpoint.load(train_states, checkpoint_id=checkpoint_path)

            self.global_step = train_states['global_step']

            set_state_dict(
                self.train_pipeline.transformer,
                self.optimizer_g,
                model_state_dict=train_states['model_state_dict_g'],
                optim_state_dict=train_states['optimizer_state_dict_g'],
                options=StateDictOptions(strict=True),
            )

            if hasattr(self, 'optimizer_d'):
                set_state_dict(
                    self.train_pipeline.discriminator,
                    self.optimizer_d,
                    model_state_dict=train_states['model_state_dict_d'],
                    optim_state_dict=train_states['optimizer_state_dict_d'],
                    options=StateDictOptions(strict=True),
                )

            get_logger().info('Loaded model and optimizer.')

            if self.cfg['train'].get('ema_decay'):
                self.train_pipeline.ema.load_state_dict(train_states['ema'])
                get_logger().info('Loaded EMA states.')

            self.lr_scheduler_g.load_state_dict(train_states['lr_scheduler_g'])
            if hasattr(self, 'optimizer_d'):
                self.lr_scheduler_d.load_state_dict(train_states['lr_scheduler_d'])
            get_logger().info('Loaded LR scheduler states.')

            if self.world_size == train_states['world_size']:
                self.set_random_states(train_states['random_states'])
                get_logger().info('Loaded random states.')
            else:
                get_logger().info(
                    f"Warning: failed to load random states. Current dist_size: {self.world_size}. Loaded dist_size: {train_states['world_size']}"
                )
            dist_barrier()
        else:
            get_logger().info('Checkpoint does not exist. Starting a new training run.')

    def save_checkpoint(self, model_name, only_model_state_dict=False):
        
        """Save the current training checkpoint."""
        train_states = self.get_train_state(only_model_state_dict=only_model_state_dict)
        output_dir = self.cfg['path']['models']

        dist_barrier()
        torch.cuda.empty_cache()
        gc.collect()
        dist_barrier()

        if model_name == 'checkpoint':
            assert not only_model_state_dict, 'training ckpt should save with only_model_state_dict=False'
            save_path_ = os.path.join(output_dir, 'checkpoint_')
            save_path = os.path.join(output_dir, 'checkpoint')

            if is_main_process():
                shutil.rmtree(save_path_, ignore_errors=True)
            dist_barrier()

            torch.distributed.checkpoint.save(train_states, checkpoint_id=save_path_)
            dist_barrier()

            if is_main_process():
                shutil.rmtree(save_path, ignore_errors=True)
                shutil.move(save_path_, save_path)

            dist_barrier()
            get_logger().info(f'Saved state to {save_path}')
        else:
            assert only_model_state_dict, 'evaluation ckpt should save with only_model_state_dict=True'
            dist_barrier()
            save_path = os.path.join(output_dir, f'{model_name}')
            torch.distributed.checkpoint.save(train_states, checkpoint_id=save_path)
            dist_barrier()
            if is_main_process():
                dcp_to_torch_save(save_path, save_path + '.pt')
                shutil.rmtree(save_path, ignore_errors=True)
            dist_barrier()

    def __del__(self):
        destroy_process_group()


if __name__ == '__main__':
    """Load the config and run the trainer."""
    cfg = OmegaConf.merge(
        OmegaConf.load(OmegaConf.from_cli().config_path),
        OmegaConf.from_cli()
    )
    cfg = OmegaConf.to_container(cfg, resolve=True)

    if cfg['mode'] == 'train':
        BaseTrainer(cfg).train()
    else:
        BaseTrainer(cfg).evaluate()
