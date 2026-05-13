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
import random

import torch
import torch.nn.functional as F
from diffusers.configuration_utils import register_to_config
from diffusers.training_utils import compute_density_for_timestep_sampling
from einops import rearrange, repeat
from peft import LoraConfig, get_peft_model

from far.models import build_model
from far.pipelines import build_pipeline
from far.schedulers.scheduling_flowmap_euler_discrete import FlowMapDiscreteScheduler
from far.trainers.trainer_far_wan_anyflow_pretrain import FAR_Wan_AnyFlow_Pretrain_Trainer
from far.utils.dist_util import all_ranks_path_exists, dist_barrier, fsdp2_wrap, is_main_process  # noqa: F401
from far.utils.logger_util import get_logger
from far.utils.lora_util import filter_learnable_module
from far.utils.registry import TRAINER_REGISTRY


@TRAINER_REGISTRY.register()
class FAR_Wan_AnyFlow_OnPolicy_Trainer(FAR_Wan_AnyFlow_Pretrain_Trainer):

    config_name = 'far_wan_anyflow_onpolicy_trainer.json'

    @register_to_config
    def __init__(
        self,
        transformer_cfg=None,
        real_cfg=None,
        discriminator_cfg=None,
        text_encoder_cfg=None,
        vae_cfg=None,
        scheduler_cfg=None,
        dmd_scheduler_cfg=None,
        rollout_cfg=None,
        dmd_cfg=None,
        flowmap_cfg=None,
        dtype=torch.float,
        device='cpu',
    ):
        super(FAR_Wan_AnyFlow_OnPolicy_Trainer, self).__init__(
            transformer_cfg=transformer_cfg,
            text_encoder_cfg=text_encoder_cfg,
            vae_cfg=vae_cfg,
            scheduler_cfg=scheduler_cfg,
            flowmap_cfg=flowmap_cfg, dtype=dtype, device=device
        )

        if real_cfg is not None:
            self.real_score = build_model(real_cfg['model_type']).from_pretrained(real_cfg['model_name'], subfolder='transformer')

            if 'lora_config' in real_cfg:
                lora_config = LoraConfig(
                    r=real_cfg['lora_config']['lora_rank'],
                    lora_alpha=real_cfg['lora_config']['lora_alpha'],
                    target_modules=filter_learnable_module(self.real_score, real_cfg['lora_config']['target_module_name']),
                    lora_dropout=0.0,
                    bias='none',
                )
                self.real_score = get_peft_model(self.real_score, lora_config, adapter_name='real')

            if 'pretrained_path' in real_cfg:
                state_dict = torch.load(real_cfg['pretrained_path'], map_location='cpu', weights_only=True)[real_cfg['pretrained_weight']]
                missing_keys, unexpected_keys = self.real_score.load_state_dict(state_dict, strict=False)
                get_logger().info(f"Loaded real_score weight from {real_cfg['pretrained_path']}, missing_keys: {missing_keys}, unexpected_keys: {unexpected_keys}")  # noqa: E501

            self.real_score = fsdp2_wrap(self.real_score, transformer_block_clsname='WanTransformerBlock')

        if discriminator_cfg is not None:
            self.discriminator = build_model(discriminator_cfg['model_type']).from_pretrained(discriminator_cfg['model_name'], subfolder='transformer')

            if 'lora_config' in discriminator_cfg:
                lora_config = LoraConfig(
                    r=discriminator_cfg['lora_config']['lora_rank'],
                    lora_alpha=discriminator_cfg['lora_config']['lora_alpha'],
                    target_modules=filter_learnable_module(self.discriminator, discriminator_cfg['lora_config']['target_module_name']),
                    lora_dropout=0.0,
                    bias='none',
                )
                self.discriminator = get_peft_model(self.discriminator, lora_config, adapter_name='real')

            if 'pretrained_path' in discriminator_cfg:
                state_dict = torch.load(discriminator_cfg['pretrained_path'], map_location='cpu', weights_only=True)[discriminator_cfg['pretrained_weight']]
                missing_keys, unexpected_keys = self.discriminator.load_state_dict(state_dict, strict=False)
                get_logger().info(f"Loaded discriminator weight from {discriminator_cfg['pretrained_path']}, missing_keys: {missing_keys}, unexpected_keys: {unexpected_keys}")  # noqa: E501

            self.set_params_to_optimize(self.discriminator, discriminator_cfg['param_names_to_optimize'])

            if discriminator_cfg['enable_gradient_checkpoint']:
                self.discriminator.enable_gradient_checkpointing()
            self.discriminator = fsdp2_wrap(self.discriminator, transformer_block_clsname='WanTransformerBlock')

        if dmd_scheduler_cfg is not None:
            self.dmd_scheduler = FlowMapDiscreteScheduler.from_pretrained(transformer_cfg['model_name'], subfolder='scheduler', **dmd_scheduler_cfg)

    # Modified from: https://github.com/guandeh17/Self-Forcing/blob/main/model/dmd.py#L54
    def _compute_kl_grad(self, noisy_latent, pred_video, timesteps, prompt_embeds, normalization=True):
        batch_size, num_frames = noisy_latent.shape[0], noisy_latent.shape[2]
        noisy_latent = rearrange(noisy_latent, 'b c f h w -> b f c h w')
        timesteps = repeat(timesteps, 'b -> b f', b=batch_size, f=num_frames).contiguous()

        # Step 1: Compute the fake score
        noise_pred_fake_cond = self.discriminator(noisy_latent, timestep=timesteps, r_timestep=timesteps, encoder_hidden_states=prompt_embeds, is_causal=False)[0]  # noqa: E501
        pred_fake_video = self.dmd_scheduler.step(noise_pred_fake_cond, noisy_latent, timesteps, torch.zeros_like(timesteps))
        pred_fake_video = rearrange(pred_fake_video, 'b f c h w -> b c f h w')

        # Step 2: Compute the real score
        noise_pred_real_cond = self.real_score(noisy_latent, timestep=timesteps, r_timestep=timesteps, encoder_hidden_states=prompt_embeds, is_causal=False)[0]  # noqa: E501
        pred_real_video_cond = self.dmd_scheduler.step(noise_pred_real_cond, noisy_latent, timesteps, torch.zeros_like(timesteps))
        pred_real_video_cond = rearrange(pred_real_video_cond, 'b f c h w -> b c f h w')

        if self.config['dmd_cfg'].get('real_guidance_scale'):
            noise_pred_real_uncond = self.real_score(noisy_latent, timestep=timesteps, r_timestep=timesteps, encoder_hidden_states=self.negative_embedding, is_causal=False)[0]  # noqa: E501
            pred_real_video_uncond = self.dmd_scheduler.step(noise_pred_real_uncond, noisy_latent, timesteps, torch.zeros_like(timesteps))
            pred_real_video_uncond = rearrange(pred_real_video_uncond, 'b f c h w -> b c f h w')

            pred_real_video = pred_real_video_cond + (
                pred_real_video_cond - pred_real_video_uncond
            ) * self.config['dmd_cfg']['real_guidance_scale']
        else:
            pred_real_video = pred_real_video_cond

        # Step 3: Compute the DMD gradient.
        grad = (pred_fake_video - pred_real_video)

        if normalization:
            # Step 4: Gradient normalization.
            p_real = (pred_video - pred_real_video)
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad = grad / normalizer
        grad = torch.nan_to_num(grad)

        return grad

    def generator_loss(self, latents, prompt_embeds):
        self.transformer.train()
        self.real_score.eval()
        self.discriminator.eval()

        # step 1: rollout to get prediction (requires_grad)
        val_pipeline = build_pipeline(self.config['rollout_cfg']['rollout_pipeline'])(
            vae=self.vae,
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            transformer=self.transformer,
            scheduler=copy.deepcopy(self.scheduler),
        )

        sample_step = torch.tensor([random.choice(self.config['rollout_cfg']['num_inference_steps_list'])], device=self.config.device)
        torch.distributed.broadcast(sample_step, src=0)
        sample_step = int(sample_step[0])

        grad_timestep = torch.tensor([random.randrange(0, sample_step)], device=self.config.device)
        torch.distributed.broadcast(grad_timestep, src=0)
        grad_timestep = int(grad_timestep[0])

        pred_video = val_pipeline.training_rollout(
            latents=torch.randn_like(latents),
            prompt_embeds=prompt_embeds,
            num_inference_steps=sample_step,
            grad_timestep=grad_timestep
        )

        # step 2: get distribution matching loss
        with torch.no_grad():
            t = torch.rand(pred_video.shape[0])
            timesteps = (self.dmd_scheduler.apply_shift(t) * self.dmd_scheduler.config.num_train_timesteps).to(device=latents.device)
            timesteps = timesteps.clamp(self.config.dmd_cfg.get('min_timestep', 0), self.config.dmd_cfg.get('max_timestep', 1000))
            noisy_latent = self.dmd_scheduler.scale_noise(pred_video, timesteps, torch.randn_like(pred_video)).detach()

            grad = self._compute_kl_grad(
                noisy_latent=noisy_latent,
                pred_video=pred_video,
                timesteps=timesteps,
                prompt_embeds=prompt_embeds
            )

        dmd_loss = self.config.dmd_cfg['dmd_weight'] * F.mse_loss(pred_video.double(), (pred_video.double() - grad.double()).detach(), reduction='mean')
        return dmd_loss

    def discriminator_loss(self, latents, prompt_embeds):

        self.transformer.eval()
        self.real_score.eval()
        self.discriminator.train()

        # step 1: rollout to get prediction
        val_pipeline = build_pipeline(self.config['rollout_cfg']['rollout_pipeline'])(
            vae=self.vae,
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            transformer=self.transformer,
            scheduler=copy.deepcopy(self.scheduler),
        )

        sample_step = torch.tensor([random.choice(self.config['rollout_cfg']['num_inference_steps_list'])], device=self.config.device)
        torch.distributed.broadcast(sample_step, src=0)
        sample_step = int(sample_step[0])

        grad_timestep = torch.tensor([random.randrange(0, sample_step)], device=self.config.device)
        torch.distributed.broadcast(grad_timestep, src=0)
        grad_timestep = int(grad_timestep[0])

        with torch.no_grad():
            pred_video = val_pipeline.training_rollout(
                latents=torch.randn_like(latents),
                prompt_embeds=prompt_embeds,
                num_inference_steps=sample_step,
                grad_timestep=grad_timestep
            )
            latents = pred_video

        batch_size, num_channels, num_frames, _, _ = latents.shape
        latents = rearrange(latents, 'b c t h w -> b t c h w')

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)

        # Sample a random timestep for each image
        t = compute_density_for_timestep_sampling(weighting_scheme='logit_normal', batch_size=batch_size, logit_mean=0, logit_std=1.0)
        t = repeat(t, 'b -> b f', b=batch_size, f=num_frames).contiguous()
        timesteps = (self.dmd_scheduler.apply_shift(t) * self.dmd_scheduler.config.num_train_timesteps).to(device=latents.device)
        timesteps = timesteps.clamp(self.config.dmd_cfg.get('min_timestep', 0), self.config.dmd_cfg.get('max_timestep', 1000))

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.dmd_scheduler.scale_noise(latents, timesteps, noise)

        # Predict the noise residual
        noise_pred = self.discriminator(noisy_latents, timestep=timesteps, r_timestep=timesteps, encoder_hidden_states=prompt_embeds, return_dict=False, is_causal=False)[0]  # noqa: E501
        target = noise - latents

        loss = torch.mean(((noise_pred.float() - target.float()) ** 2).reshape(batch_size, num_frames, -1), dim=-1)
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

        if train_generator:
            loss_dict = {}

            generator_loss = self.generator_loss(latents[:self.config.dmd_cfg['dmd_batch_size']], prompt_embeds[:self.config.dmd_cfg['dmd_batch_size']])  # noqa: E501
            generator_loss.backward()
            loss_dict.update({'generator_loss': float(generator_loss.detach())})

            if self.config.dmd_cfg.get('cotrain_forward_kl', True):
                loss_dict.update(self.pixel_loss(latents, prompt_embeds, iters))
            return loss_dict
        else:
            latents, prompt_embeds = latents[:self.config.dmd_cfg['dmd_batch_size']], prompt_embeds[:self.config.dmd_cfg['dmd_batch_size']]
            discrminator_loss = self.discriminator_loss(latents, prompt_embeds)
            discrminator_loss.backward()
            return {'discriminator_loss': float(discrminator_loss.detach())}
