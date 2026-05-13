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

import copy
import os
import random

import torch
import torch.distributed as dist
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import AutoencoderKLWan
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.utils import export_to_video
from einops import rearrange, repeat
from peft import LoraConfig, get_peft_model
from PIL import Image
from torchvision.transforms import Resize, ToTensor
from tqdm import tqdm
from transformers import AutoTokenizer, UMT5EncoderModel

from far.metrics.vbench import VBenchEvaluator
from far.metrics.vbench_i2v import VBenchI2VEvaluator
from far.models import build_model
from far.pipelines import build_pipeline
from far.schedulers.scheduling_flowmap_euler_discrete import FlowMapDiscreteScheduler
from far.utils.dist_util import all_ranks_path_exists, check_video, fsdp2_wrap, get_dist_rank, get_world_size, reduce_loss
from far.utils.ema_util import ShardEMA
from far.utils.logger_util import get_logger
from far.utils.lora_util import filter_learnable_module
from far.utils.registry import TRAINER_REGISTRY
from far.utils.vis_util import draw_rectangle


@TRAINER_REGISTRY.register()
class Wan_AnyFlow_Pretrain_Trainer(nn.Module, ConfigMixin):

    config_name = 'wan_anyflow_pretrain_trainer.json'

    @register_to_config
    def __init__(
        self,
        transformer_cfg=None,
        text_encoder_cfg=None,
        vae_cfg=None,
        scheduler_cfg=None,
        flowmap_cfg=None,
        dtype=torch.float,
        device='cpu',
    ):
        super(Wan_AnyFlow_Pretrain_Trainer, self).__init__()

        if transformer_cfg is not None:
            self.transformer = build_model(transformer_cfg['model_type']).from_pretrained(
                transformer_cfg['model_name'], subfolder='transformer'
            )

            if not self.transformer.config.get('init_flowmap_model'):
                self.transformer.setup_flowmap_model(
                    gate_value=flowmap_cfg.get('gate_value', 0.0),
                    deltatime_type=flowmap_cfg.get('deltatime_type', 'r')
                )

            if 'lora_config' in transformer_cfg:
                lora_config = LoraConfig(
                    r=transformer_cfg['lora_config']['lora_rank'],
                    lora_alpha=transformer_cfg['lora_config']['lora_alpha'],
                    target_modules=filter_learnable_module(self.transformer, transformer_cfg['lora_config']['target_module_name']),
                    lora_dropout=0.0,
                    bias='none',
                )
                self.transformer = get_peft_model(self.transformer, lora_config, adapter_name='real')

            if 'pretrained_path' in transformer_cfg:
                state_dict = torch.load(transformer_cfg['pretrained_path'], map_location='cpu', weights_only=True)[transformer_cfg['pretrained_weight']]
                missing_keys, unexpected_keys = self.transformer.load_state_dict(state_dict, strict=False)
                get_logger().info(f"Loaded {transformer_cfg['pretrained_weight']} weight from {transformer_cfg['pretrained_path']}, missing_keys: {missing_keys}, unexpected_keys: {unexpected_keys}")  # noqa: E501

            if 'param_names_to_optimize' in transformer_cfg:
                self.set_params_to_optimize(self.transformer, transformer_cfg['param_names_to_optimize'])

            if transformer_cfg['enable_gradient_checkpoint']:
                self.transformer.enable_gradient_checkpointing()
            self.transformer = fsdp2_wrap(self.transformer, transformer_block_clsname='WanTransformerBlock')

        if text_encoder_cfg is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(transformer_cfg['model_name'], subfolder='tokenizer', use_fast=False)
            self.text_encoder = UMT5EncoderModel.from_pretrained(transformer_cfg['model_name'], subfolder='text_encoder')
            self.text_encoder.requires_grad_(False)
            self.text_encoder.eval()
            self.text_encoder = fsdp2_wrap(self.text_encoder, transformer_block_clsname='UMT5Block', sync_module_state=False)

            if 'negative_embedding_path' in text_encoder_cfg:
                negative_embedding_dict = torch.load(text_encoder_cfg['negative_embedding_path'], map_location='cpu', weights_only=False)
                self.negative_embedding = negative_embedding_dict['negative_prompt_embeds'].to(device=device, dtype=dtype)
                self.drop_text_ratio = text_encoder_cfg['drop_text_ratio']

        if vae_cfg is not None:
            self.vae = AutoencoderKLWan.from_pretrained(transformer_cfg['model_name'], subfolder='vae')
            self.vae.requires_grad_(False)
            self.vae.eval()
            self.vae = self.vae.to(device=device, dtype=dtype)

        if scheduler_cfg is not None:
            self.scheduler = FlowMapDiscreteScheduler.from_pretrained(transformer_cfg['model_name'], subfolder='scheduler', **scheduler_cfg)

    def set_params_to_optimize(self, model, param_names_to_optimize):

        def is_keyword_in_param_name(name, keyword_list):
            for keyword in keyword_list:
                if keyword in name:
                    return True
            return False

        params_to_optimize = []
        params_to_fix = []

        for name, param in model.named_parameters():
            if is_keyword_in_param_name(name, param_names_to_optimize):
                param.requires_grad = True
                params_to_optimize.append(param)
                get_logger().info(f'optimizer params: {name}')
            else:
                param.requires_grad = False
                params_to_fix.append(param)
                get_logger().info(f'fix params: {name}')

        get_logger().info(f'#Trained Parameters: {sum([p.numel() for p in params_to_optimize]) / 1e6} M')
        get_logger().info(f'#Fixed Parameters: {sum([p.numel() for p in params_to_fix]) / 1e6} M')

    def set_ema_model(self, ema_decay, ema_warmup_step=0):
        if ema_decay is not None:
            self.ema = ShardEMA(self.transformer, decay=ema_decay, warmup_steps=ema_warmup_step)
            get_logger().info(f'enable EMA training with decay {ema_decay}, warmup_steps: {ema_warmup_step}')

    def _normalize_latents(self, latents, latents_mean, latents_std):
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(device=latents.device)
        latents = ((latents.float() - latents_mean) * latents_std).to(latents)
        return latents

    @torch.no_grad()
    def encode_latents(self, videos, sample=True):
        videos = rearrange(videos, 'b t c h w -> b c t h w')
        moments = self.vae._encode(videos)

        latents_mean = torch.tensor(self.vae.config.latents_mean)
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std)

        mu, logvar = torch.chunk(moments, 2, dim=1)
        mu = self._normalize_latents(mu, latents_mean, latents_std)

        if sample:
            logvar = self._normalize_latents(logvar, latents_mean, latents_std)

            latents = torch.cat([mu, logvar], dim=1)
            posterior = DiagonalGaussianDistribution(latents)
            latents = posterior.sample(generator=None)
            del posterior
        else:
            latents = mu
        return latents

    @torch.no_grad()
    def encode_text_embedding(self, prompts, device):
        max_sequence_length = 512

        # prepare condition embedding
        text_inputs = self.tokenizer(
            prompts,
            padding='max_length',
            max_length=max_sequence_length,  # Wan 2.1: max sequence length 512
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors='pt')

        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        return prompt_embeds

    @torch.no_grad()
    def extract_latents(self, batch):
        latents = self.encode_latents(batch['pixel_values'].to(dtype=self.config.dtype), sample=False)

        return_dict = {
            'latents': latents.contiguous().clone(),
        }
        return return_dict

    def sample_timestep(self, batch_size, dtype, device):

        t_1 = torch.rand(batch_size, dtype=dtype, device=device)
        t_2 = torch.rand(batch_size, dtype=dtype, device=device)

        # t is the larger one, and r is the smaller one
        t = torch.maximum(t_1, t_2)
        r = torch.minimum(t_1, t_2)

        global_start_idx = get_dist_rank() * batch_size
        total_global_bsz = get_world_size() * batch_size

        n_diffusion = round(self.config.flowmap_cfg['diffusion_ratio'] * total_global_bsz)
        n_consistency = round(self.config.flowmap_cfg['consistency_ratio'] * total_global_bsz)
        is_diffusion = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for batch_idx in range(batch_size):
            if global_start_idx + batch_idx < n_diffusion:
                r[batch_idx] = t[batch_idx]
                is_diffusion[batch_idx] = True
            elif global_start_idx + batch_idx < n_diffusion + n_consistency:
                r[batch_idx] = 0
            else:
                pass

        return t, r, is_diffusion

    @torch.no_grad()
    def compute_central_difference(self, noisy_latents, latents, noise, t, r, prompt_embeds, guidance):
        def u_func(noisy_latents, t, r):
            return self.transformer(noisy_latents, timestep=t, r_timestep=r, encoder_hidden_states=prompt_embeds, return_dict=False, is_causal=False)[0]  # noqa: E501

        v_pred = noise - latents

        # t_plus_epsilon
        t_plus = t + self.config.flowmap_cfg['epsilon']
        noisy_latents_plus = noisy_latents + v_pred * (self.config.flowmap_cfg['epsilon'] / self.scheduler.config.num_train_timesteps)
        noise_pred_plus = u_func(noisy_latents_plus, t_plus, r)

        # t_minus_epsilon
        t_minus = t - self.config.flowmap_cfg['epsilon']
        noisy_latents_minus = noisy_latents - v_pred * (self.config.flowmap_cfg['epsilon'] / self.scheduler.config.num_train_timesteps)
        noise_pred_minus = u_func(noisy_latents_minus, t_minus, r)

        return (noise_pred_plus - noise_pred_minus) / (2 * self.config.flowmap_cfg['epsilon'] * guidance)

    def pixel_loss(self, latents, prompt_embeds, iters):
        self.transformer.train()

        if self.drop_text_ratio > 0:
            mask = torch.rand(latents.shape[0], device=latents.device) < self.drop_text_ratio
            prompt_embeds[mask] = self.negative_embedding

        bidirection_loss = self.train_bidirection(latents, prompt_embeds, iters)
        bidirection_loss.backward()

        return {'bidirection_loss': float(reduce_loss(bidirection_loss))}

    def train_bidirection(self, latents, prompt_embeds, iters):

        batch_size, num_channels, num_frames, _, _ = latents.shape
        latents = rearrange(latents, 'b c t h w -> b t c h w')

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)

        t, r, is_diffusion = self.sample_timestep(batch_size, dtype=self.config.dtype, device=latents.device)
        t, r = repeat(t, 'b -> b f', b=batch_size, f=num_frames).contiguous(), repeat(r, 'b -> b f', b=batch_size, f=num_frames).contiguous()

        t = (self.scheduler.apply_shift(t) * self.scheduler.config.num_train_timesteps).to(device=latents.device)
        r = (self.scheduler.apply_shift(r) * self.scheduler.config.num_train_timesteps).to(device=latents.device)

        train_i2v = torch.tensor([random.random()], device=self.config.device)
        torch.distributed.broadcast(train_i2v, src=0)
        if train_i2v < self.config.transformer_cfg['i2v_prob']:
            t[:, 0] = 0

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.scheduler.scale_noise(latents, t, noise)

        # Predict the noise residual
        noise_pred = self.transformer(noisy_latents, timestep=t, r_timestep=r, encoder_hidden_states=prompt_embeds, return_dict=False, is_causal=False)[0]  # noqa: E501
        target = noise - latents

        # guidance distillation
        if self.config.text_encoder_cfg.get('fuse_guidance_scale'):
            guidance = self.config.text_encoder_cfg['fuse_guidance_scale']

            with torch.no_grad():
                noise_pred_uncond = self.transformer(noisy_latents, timestep=t, r_timestep=r, encoder_hidden_states=self.negative_embedding, is_causal=False)[0]  # noqa: E501
            noise_pred = (noise_pred - (1 - guidance) * noise_pred_uncond) / guidance
        else:
            guidance = 1.0

        dF_dt = self.compute_central_difference(noisy_latents, latents, noise, t, r, prompt_embeds, guidance)

        target = (noise - latents) - (t - r).view(batch_size, num_frames, 1, 1, 1) * dF_dt

        loss = torch.mean(((noise_pred.float() - target.float()) ** 2).reshape(batch_size, -1), dim=-1)
        weight = self.scheduler.get_train_weight(t).to(latents.device)
        loss = (loss.unsqueeze(-1) * weight).mean(-1)

        with torch.no_grad():
            global_loss = torch.cat(dist.nn.all_gather(loss), dim=0)
            global_diffusion_mask = torch.cat(dist.nn.all_gather(is_diffusion), dim=0)
            scale_weight = global_loss[global_diffusion_mask].mean() / (loss[~is_diffusion] + 1e-5)

        loss[~is_diffusion] = loss[~is_diffusion] * scale_weight
        return loss.mean()

    def train_step(self, batch, iters=-1, train_generator=True):
        # Convert video to latent space
        if 'latents' in batch:
            latents = batch['latents'].to(device=self.config.device, dtype=self.config.dtype)
        else:
            latents = self.encode_latents(batch['pixel_values'].to(device=self.config.device, dtype=self.config.dtype), sample=False)

        if 'prompt_embeds' in batch:
            prompt_embeds = batch['prompt_embeds'].to(device=self.config.device)
        else:
            prompt_embeds = self.encode_text_embedding(batch['prompts'], device=self.config.device)

        return self.pixel_loss(latents, prompt_embeds, iters)

    def sample(self, val_dataloader, cfg, global_step=0):
        if hasattr(self, 'ema'):
            self._sample_bidirectional(val_dataloader, cfg, global_step, use_ema=True)
        self._sample_bidirectional(val_dataloader, cfg, global_step, use_ema=False)

    def _sample_bidirectional(self, val_dataloader, cfg, global_step=0, use_ema=True):

        if hasattr(self, 'ema') and use_ema:
            self.ema.store(self.transformer)
            self.ema.copy_to(self.transformer)

        self.transformer.eval()

        val_pipeline = build_pipeline(cfg['val']['val_pipeline'])(
            vae=self.vae,
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            transformer=self.transformer,
            scheduler=copy.deepcopy(self.scheduler),
        )

        if use_ema:
            save_root_dir = os.path.join(cfg['path']['visualization'], 'sample_bidirectional', f'iter_{global_step}_ema')
        else:
            save_root_dir = os.path.join(cfg['path']['visualization'], 'sample_bidirectional', f'iter_{global_step}')

        for item in val_dataloader:
            for num_inference_steps in cfg['val']['sample_cfg']['num_inference_steps']:
                for seed in cfg['val']['sample_cfg']['seed']:
                    index = int(item['index'])
                    prompt = item['prompt'][0]

                    if 'image' in item:
                        task_type = 'i2v'
                        context_sequence = item['image']
                        context_length = 1
                    elif 'video' in item:
                        task_type = 'v2v'
                        context_sequence = item['video']
                        context_length = item['video'].shape[1]
                    else:
                        task_type = 't2v'
                        context_sequence = None
                        context_length = 0

                    save_dir = os.path.join(save_root_dir, f'task_{task_type}', f'sample_step_{num_inference_steps}', f'seed_{seed}')
                    save_path = f"{save_dir}/{index:02d}_{prompt[:40].replace(' ', '_')}.mp4"
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    if all_ranks_path_exists(save_path):
                        continue

                    input_params = {
                        'prompt': prompt,
                        'context_sequence': context_sequence,
                        'negative_prompt': cfg['val']['sample_cfg']['negative_prompt'],
                        'guidance_scale': cfg['val']['sample_cfg']['guidance_scale'],
                        'height': cfg['val']['sample_cfg']['height'],
                        'width': cfg['val']['sample_cfg']['width'],
                        'num_frames': cfg['val']['sample_cfg']['num_frames'],
                        'num_inference_steps': num_inference_steps,
                        'generator': torch.Generator('cuda').manual_seed(seed),
                    }

                    video = val_pipeline(**input_params).frames[0]
                    video = draw_rectangle(video, context_length=context_length)

                    export_to_video(video, output_video_path=save_path, fps=16)

        if hasattr(self, 'ema') and use_ema:
            self.ema.restore(self.transformer)

    @torch.no_grad()
    def validate(self, val_dataloader, cfg, global_step=0):

        if hasattr(self, 'ema'):
            self.ema.store(self.transformer)
            self.ema.copy_to(self.transformer)

        self.transformer.eval()

        val_pipeline = build_pipeline(cfg['val']['val_pipeline'])(
            vae=self.vae,
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            transformer=self.transformer,
            scheduler=copy.deepcopy(self.scheduler),
        )

        save_root_dir = os.path.join(cfg['path']['visualization'], 'validation', f'iter_{global_step}')

        for item in tqdm(val_dataloader):

            save_path = os.path.join(save_root_dir, 'samples', f"{item['video_path'][0]}")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            if all_ranks_path_exists(save_path):
                continue

            prompt = item['aug_prompt_en'][0]
            manual_seed = int(item['seed'])

            input_params = {
                'prompt': prompt,
                'negative_prompt': cfg['val']['sample_cfg']['negative_prompt'],
                'guidance_scale': cfg['val']['sample_cfg']['guidance_scale'],
                'height': cfg['val']['sample_cfg']['height'],
                'width': cfg['val']['sample_cfg']['width'],
                'num_frames': cfg['val']['sample_cfg']['num_frames'],
                'num_inference_steps': cfg['val']['sample_cfg']['num_vbench_inference_steps'],
                'generator': torch.Generator(device='cuda').manual_seed(manual_seed)
            }

            video = val_pipeline(**input_params).frames[0]
            export_to_video(video, output_video_path=save_path, fps=16)

            # remove the video if it does not correctly export
            if not check_video(save_path):
                os.remove(save_path)

        if hasattr(self, 'ema'):
            self.ema.restore(self.transformer)

    @torch.no_grad()
    def validate_i2v(self, val_dataloader, cfg, global_step=0):

        if hasattr(self, 'ema'):
            self.ema.store(self.transformer)
            self.ema.copy_to(self.transformer)

        self.transformer.eval()

        val_pipeline = build_pipeline(cfg['val']['val_pipeline'])(
            vae=self.vae,
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            transformer=self.transformer,
            scheduler=copy.deepcopy(self.scheduler),
        )
        val_pipeline.set_progress_bar_config(disable=True)

        save_root_dir = os.path.join(cfg['path']['visualization'], 'validation_i2v', f'iter_{global_step}')

        for item in tqdm(val_dataloader):
            save_path = os.path.join(save_root_dir, 'samples', f"{item['video_path'][0]}")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            if all_ranks_path_exists(save_path):
                continue

            context_sequence = {'raw': ToTensor()(
                Resize([cfg['val']['sample_cfg']['height'], cfg['val']['sample_cfg']['width']])(
                    Image.open(item['image_name'][0]).convert('RGB')
                )
            ).unsqueeze(0).unsqueeze(0)}

            prompt = item['aug_prompt_en'][0]
            manual_seed = int(item['seed'])

            input_params = {
                'prompt': prompt,
                'negative_prompt': cfg['val']['sample_cfg']['negative_prompt'],
                'context_sequence': context_sequence,
                'guidance_scale': cfg['val']['sample_cfg']['guidance_scale'],
                'height': cfg['val']['sample_cfg']['height'],
                'width': cfg['val']['sample_cfg']['width'],
                'num_frames': cfg['val']['sample_cfg']['num_frames'],
                'num_inference_steps': cfg['val']['sample_cfg']['num_vbench_inference_steps'],
                'generator': torch.Generator(device='cuda').manual_seed(manual_seed)
            }

            video = val_pipeline(**input_params).frames[0]
            export_to_video(video, output_video_path=save_path, fps=16)

            # remove the video if it does not correctly export
            if not check_video(save_path):
                os.remove(save_path)

        if hasattr(self, 'ema'):
            self.ema.restore(self.transformer)

    @torch.no_grad()
    def eval_performance(self, cfg, global_step):
        save_root_dir = os.path.join(cfg['path']['visualization'], 'validation', f'iter_{global_step}')
        evaluator = VBenchEvaluator(save_root_dir, device=torch.device(self.config.device))
        eval_info_dict = evaluator.evaluate(save_root_dir)
        return eval_info_dict

    @torch.no_grad()
    def eval_i2v_performance(self, cfg, global_step):
        save_root_dir = os.path.join(cfg['path']['visualization'], 'validation_i2v', f'iter_{global_step}')
        evaluator = VBenchI2VEvaluator(save_root_dir, resolution=cfg['val']['eval_cfg']['resolution'], device=torch.device(self.config.device))
        eval_info_dict = evaluator.evaluate(save_root_dir)
        return eval_info_dict
