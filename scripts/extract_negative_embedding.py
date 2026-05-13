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
from diffusers import WanPipeline, LTX2Pipeline

if __name__ == '__main__':
    dtype = torch.bfloat16
    device = torch.device('cuda')

    model_type = "ltx2"

    if model_type == "wan":
        negative_prompt_cn = "色调艳丽, 过曝, 静态, 细节模糊不清, 字幕, 风格, 作品, 画作, 画面, 静止, 整体发灰, 最差质量, 低质量, JPEG压缩残留, 丑陋的, 残缺的, 多余的手指, 画得不好的手部, 画得不好的脸部, 畸形的, 毁容的, 形态畸形的肢体, 手指融合, 静止不动的画面, 杂乱的背景, 三条腿, 背景人很多, 倒着走"
        negative_prompt_en = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        pipe = WanPipeline.from_pretrained('experiments/pretrained_models/Wan2.2-TI2V-5B-Diffusers', torch_dtype=torch.bfloat16)
        pipe = pipe.to(device)
        with torch.no_grad():
            _, negative_prompt_embeds = pipe.encode_prompt(
                '',
                negative_prompt=negative_prompt_en,
                max_sequence_length=512,
                device=device,
                dtype=dtype,
            )
        torch.save({'negative_prompt_embeds': negative_prompt_embeds.cpu()}, 'wan_negative_embeddings_en.pth')
    elif model_type == "ltx2":
        negative_prompt = "shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly, transition, static."
        pipe = LTX2Pipeline.from_pretrained('experiments/pretrained_models/LTX-2', torch_dtype=torch.bfloat16)
        pipe = pipe.to(device)
        with torch.no_grad():
            _, _, negative_prompt_embeds, negative_prompt_attention_mask = pipe.encode_prompt(
                '',
                negative_prompt=negative_prompt,
                max_sequence_length=1024,
                scale_factor=8,
                device=device,
                dtype=dtype,
            )
            tokenizer_padding_side = getattr(pipe.tokenizer, 'padding_side', 'left')
            connector_negative_prompt_embeds, _, connector_negative_attention_mask = pipe.connectors(
                negative_prompt_embeds,
                negative_prompt_attention_mask,
                padding_side=tokenizer_padding_side,
            )
        torch.save(
            {
                'connector_negative_prompt_embeds': connector_negative_prompt_embeds.cpu(),
                'connector_negative_attention_mask': connector_negative_attention_mask.cpu(),
            },
            'ltx2_negative_embeddings.pth',
        )