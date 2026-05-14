#!/usr/bin/env python

from __future__ import annotations

import json
import os

import pathlib
import subprocess
import sys
import uuid
from dataclasses import dataclass

# step 2: import the code
import decord
import gradio as gr
import spaces
import torch
from diffusers.utils import export_to_video
from huggingface_hub import snapshot_download
from torchvision import transforms

decord.bridge.set_bridge('torch')

INTRODUCTION = """
# AnyFlow: Any-Step Video Diffusion Model with On-Policy Flow Map Distillation

<div align="left" style="display:flex; flex-wrap:wrap; gap:6px; align-items:center;"><a href="https://arxiv.org/abs/2605.13724" style="display:inline-block;"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b.svg?logo=arxiv&logoColor=white" alt="arXiv" style="display:inline-block; vertical-align:middle;"></a><a href="https://nvlabs.github.io/AnyFlow/" style="display:inline-block;"><img src="https://img.shields.io/badge/Project-Page-1e90ff.svg?logo=githubpages&logoColor=white" alt="Project Page" style="display:inline-block; vertical-align:middle;"></a><a href="https://github.com/nvlabs/AnyFlow" style="display:inline-block;"><img src="https://img.shields.io/badge/GitHub-Code-181717.svg?logo=github&logoColor=white" alt="GitHub" style="display:inline-block; vertical-align:middle;"></a><a href="https://huggingface.co/collections/nvidia/anyflow" style="display:inline-block;"><img src="https://img.shields.io/badge/🤗%20HuggingFace-Models-ffcd00.svg" alt="HuggingFace" style="display:inline-block; vertical-align:middle;"></a></div>

In this demo, we showcase **AnyFlow**, the first any-step video diffusion distillation framework built on flow maps. **AnyFlow** offers these key features:

- ⚡ **Any-Step Generation**: Unlike traditional distilled models tied to fixed step budgets, **AnyFlow** enables a single model to adapt to arbitrary inference budgets. It achieves high-quality few-step generation while providing stable improvements as more sampling steps are added.
- 🔀 **Multiple Architectures**: **AnyFlow** supports any-step distillation for both **causal** and **bidirectional** video diffusion models.
- 🎬 **Multiple Tasks**: **AnyFlow** supports Text-to-Video, Image-to-Video, and Video-to-Video generation within one causal video diffusion model.
- 📈 **Scalable Performance**: **AnyFlow** is validated from **1.3B** up to **14B** parameters.
---
> **Note:** This demo is served on a single H100 GPU without carefully designed concurrency. Expected inference time is **1.3B: 5-10s (4 steps)** and **14B: 20-30s (4 steps)** per generation, including pipeline CPU↔GPU load/offload cost. If you have your own GPU, you can run the demo locally:

```bash
git clone -b anyflow-gradio-demo https://github.com/NVlabs/AnyFlow.git && cd AnyFlow
conda create -n far python=3.10 && conda activate far
pip install -r requirements.txt
python app.py
```

---
"""

# step 1: clone the repo
repo_url = f'https://github.com/nvlabs/AnyFlow.git'
subprocess.run(['git', 'clone', repo_url])

repo_root = pathlib.Path(__file__).resolve().parent
anyflow_src = repo_root / 'AnyFlow'
sys.path.insert(0, str(anyflow_src.resolve()))

from far.pipelines.pipeline_far_wan_anyflow import FARWanAnyFlowPipeline  # noqa: E402
from far.pipelines.pipeline_wan_anyflow import WanAnyFlowPipeline  # noqa: E402
from far.utils.video_util import select_frame_indices  # noqa: E402


@dataclass(frozen=True)
class AppConfig:
    hub_repo: tuple[str, ...]
    hub_repo_bidirectional: tuple[str, ...]
    device: str
    dtype: torch.dtype
    height: int
    width: int
    num_frames: int
    fps: int
    default_steps: int
    min_steps: int
    max_steps: int
    valid_cond_frames: tuple[int, ...]
    negative_prompt: str


def app_config() -> AppConfig:
    return AppConfig(
        hub_repo=(
            'nvidia/AnyFlow-FAR-Wan2.1-14B-Diffusers',
            'nvidia/AnyFlow-FAR-Wan2.1-1.3B-Diffusers',
        ),
        hub_repo_bidirectional=(
            'nvidia/AnyFlow-Wan2.1-T2V-14B-Diffusers',
            'nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers',
        ),
        device='cuda',
        dtype=torch.bfloat16,
        height=480,
        width=832,
        num_frames=81,
        fps=16,
        default_steps=4,
        min_steps=2,
        max_steps=32,
        valid_cond_frames=(1, 13, 25),
        negative_prompt='',
    )


def load_examples(json_path: pathlib.Path, asset_key: str | None = None):
    if not json_path.is_file():
        return []
    rows = []
    for e in json.loads(json_path.read_text()):
        p = e.get('prompt')
        if not p:
            continue
        if asset_key is None:
            rows.append(p)
            continue
        rel = e.get(asset_key)
        if rel:
            rows.append([p, str((anyflow_src / rel).resolve())])
    return rows


def load_causal_pipeline(repo_id):    
    snapshot_download(repo_id=repo_id, local_dir=f'pretrained_models/{repo_id}')
    pipe = FARWanAnyFlowPipeline.from_pretrained(f'pretrained_models/{repo_id}').to(dtype=CFG.dtype)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_bidirectional_pipeline(repo_id):
    snapshot_download(repo_id=repo_id, local_dir=f'pretrained_models/{repo_id}')
    pipe = WanAnyFlowPipeline.from_pretrained(f'pretrained_models/{repo_id}').to(dtype=CFG.dtype)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def save_video(frames, label: str) -> str:
    out = output_dir / label / f'{uuid.uuid4().hex[:8]}.mp4'
    out.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, output_video_path=str(out), fps=CFG.fps)
    return str(out)


def prepare_image_context(image):
    img = image.convert('RGB')
    t = transforms.ToTensor()(transforms.Resize([CFG.height, CFG.width])(img))
    return {'raw': t.unsqueeze(0).unsqueeze(0)}, 1


def prepare_video_context(video_path: str):
    vr = decord.VideoReader(video_path)
    idxs = select_frame_indices(len(vr), vr.get_avg_fps() or float(CFG.fps), target_fps=CFG.fps)
    ok = [c for c in CFG.valid_cond_frames if c <= len(idxs)]
    num_cond = max(ok or [1])
    idxs = idxs[:num_cond]
    frames = vr.get_batch(idxs)
    if not isinstance(frames, torch.Tensor):
        frames = torch.from_numpy(frames.asnumpy()) if hasattr(frames, 'asnumpy') else torch.as_tensor(frames)
    frames = transforms.Resize([CFG.height, CFG.width])((frames.float() / 255.0).permute(0, 3, 1, 2).contiguous()).unsqueeze(0)
    return {'raw': frames}, frames.shape[1]


def _seed(s):
    return int(s) if s is not None else 0


@spaces.GPU(duration=120)
def causal_inference(prompt: str, num_steps: int, seed, task_type: str, image_in, video_in, model_repo: str):
    causal_pipelines[model_repo] = causal_pipelines[model_repo].to(device=CFG.device)
    pipe = causal_pipelines[model_repo]

    prompt = prompt.strip()
    num_steps = int(num_steps) if num_steps else CFG.default_steps
    context_sequence = None
    if task_type == 'ti2v':
        context_sequence = prepare_image_context(image_in)[0]
    elif task_type == 'tv2v':
        context_sequence = prepare_video_context(video_in)[0]

    kwargs = dict(
        prompt=prompt,
        negative_prompt=None,
        height=CFG.height,
        width=CFG.width,
        num_frames=CFG.num_frames,
        num_inference_steps=num_steps,
        generator=torch.Generator(device=CFG.device).manual_seed(_seed(seed)),
    )
    if context_sequence is not None:
        kwargs['context_sequence'] = context_sequence

    with torch.no_grad():
        out = pipe(**kwargs)
    causal_pipelines[model_repo] = causal_pipelines[model_repo].to(device='cpu')
    torch.cuda.empty_cache()
    return save_video(out.frames[0], 'causal')


@spaces.GPU(duration=120)
def bidirectional_inference(prompt: str, num_steps: int, seed, model_repo: str):
    bidirectional_pipelines[model_repo] = bidirectional_pipelines[model_repo].to(device=CFG.device)
    pipe = bidirectional_pipelines[model_repo]

    prompt = prompt.strip()
    num_steps = int(num_steps) if num_steps else CFG.default_steps
    kwargs = dict(
        prompt=prompt,
        negative_prompt=None,
        height=CFG.height,
        width=CFG.width,
        num_frames=CFG.num_frames,
        num_inference_steps=num_steps,
        generator=torch.Generator(device=CFG.device).manual_seed(_seed(seed)),
    )
    with torch.no_grad():
        out = pipe(**kwargs)
    bidirectional_pipelines[model_repo] = bidirectional_pipelines[model_repo].to(device='cpu')
    torch.cuda.empty_cache()
    return save_video(out.frames[0], 'bidirectional')


def on_run(prompt, num_steps, seed, task_type, image_in, video_in, model_repo):
    return gr.update(value=causal_inference(prompt, num_steps, seed, task_type, image_in, video_in, model_repo))


def on_run_bidirectional(prompt, num_steps, seed, model_repo):
    return gr.update(value=bidirectional_inference(prompt, num_steps, seed, model_repo))


def example_row(prompt_text: str, target_prompt, asset_path=None, asset_kind=None, target_asset=None):
    thumb_w = 140
    with gr.Row(equal_height=True, elem_classes=['example-chip']):
        if asset_kind == 'image' and asset_path:
            with gr.Column(scale=0, min_width=thumb_w, elem_classes=['example-thumb']):
                gr.Image(value=asset_path, interactive=False, show_label=False, height=100, width=thumb_w, container=False)
        elif asset_kind == 'video' and asset_path:
            with gr.Column(scale=0, min_width=thumb_w, elem_classes=['example-thumb']):
                gr.Video(value=asset_path, interactive=False, show_label=False, height=100, width=thumb_w)
        with gr.Column(scale=1, min_width=0):
            gr.Textbox(value=prompt_text, show_label=False, lines=3, max_lines=3, interactive=False, elem_classes=['caption'])
        use = gr.Button('Use', scale=0, min_width=68, variant='secondary')
    if target_asset is not None and asset_path:
        use.click(lambda p=prompt_text, ap=str(asset_path): (p, ap), outputs=[target_prompt, target_asset])
    else:
        use.click(lambda p=prompt_text: p, outputs=[target_prompt])


CFG = app_config()

output_dir = repo_root / 'results' / 'gradio'
output_dir.mkdir(parents=True, exist_ok=True)

i2v_captions = anyflow_src / 'assets' / 'evaluation' / 'eval_caption_i2v.json'
v2v_captions = anyflow_src / 'assets' / 'evaluation' / 'eval_caption_v2v.json'
t2v_captions = anyflow_src / 'assets' / 'evaluation' / 'eval_caption_t2v.json'
i2v_examples = load_examples(i2v_captions, 'image')
v2v_examples = load_examples(v2v_captions, 'video')
t2v_examples = load_examples(t2v_captions)

causal_pipelines = {repo: load_causal_pipeline(repo) for repo in CFG.hub_repo}
bidirectional_pipelines = {repo: load_bidirectional_pipeline(repo) for repo in CFG.hub_repo_bidirectional}

with gr.Blocks(
    title='AnyFlow',
    css="""
.example-chip { gap: 12px !important; align-items: stretch !important; margin-bottom: 10px !important; padding: 10px 12px !important;
  border: 1px solid var(--border-color-primary); border-radius: 12px !important; background: var(--background-fill-secondary); }
.example-chip .example-thumb { flex: 0 0 auto !important; width: fit-content !important; max-width: 160px !important; }
.example-chip .caption textarea { font-size: 13px !important; line-height: 1.45 !important; }
.example-chip .wrap { min-height: 0 !important; }
""",
) as demo:
    gr.Markdown(INTRODUCTION, elem_classes='markdown-text')
    with gr.Tabs():
        with gr.Tab('AnyFlow-Causal'):
            gr.Markdown('### AnyFlow-FAR-Wan2.1 — T2V / I2V / V2V')
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(label='Prompt', lines=4)
                    model_repo = gr.Dropdown(list(CFG.hub_repo), value=CFG.hub_repo[0], label='Model')
                    task_type = gr.Radio(
                        choices=[('Text → Video', 't2v'), ('Image → Video', 'ti2v'), ('Video → Video', 'tv2v')],
                        value='t2v',
                        label='Mode',
                    )
                    with gr.Column(visible=False) as cond_image_col:
                        image_in = gr.Image(label='Conditioning image (I2V)', type='pil', height=180)
                    with gr.Column(visible=False) as cond_video_col:
                        video_in = gr.Video(label='Conditioning video (V2V)', height=180)
                    with gr.Row():
                        num_steps = gr.Slider(CFG.min_steps, CFG.max_steps, value=CFG.default_steps, step=1, label='Inference steps')
                        seed = gr.Number(0, label='Seed', precision=0)
                with gr.Column():
                    out_video = gr.Video(label='Output', autoplay=True)
                    run_btn = gr.Button('Generate')

            gr.Markdown('### Examples')
            with gr.Column(visible=True) as examples_t2v:
                for ex in t2v_examples:
                    example_row(ex, target_prompt=prompt)
            with gr.Column(visible=False) as examples_i2v:
                for p_text, img_path in i2v_examples:
                    example_row(p_text, target_prompt=prompt, asset_path=img_path, asset_kind='image', target_asset=image_in)
            with gr.Column(visible=False) as examples_v2v:
                for p_text, vid_path in v2v_examples:
                    example_row(p_text, target_prompt=prompt, asset_path=vid_path, asset_kind='video', target_asset=video_in)

            def toggle_mode_ui(mode: str):
                return tuple(gr.update(visible=v) for v in (mode == 'ti2v', mode == 'tv2v', mode == 't2v', mode == 'ti2v', mode == 'tv2v'))

            outs_toggle = [cond_image_col, cond_video_col, examples_t2v, examples_i2v, examples_v2v]
            task_type.change(toggle_mode_ui, inputs=[task_type], outputs=outs_toggle, queue=False)
            run_btn.click(on_run, inputs=[prompt, num_steps, seed, task_type, image_in, video_in, model_repo], outputs=[out_video])

        with gr.Tab('AnyFlow-Bidirectional'):
            gr.Markdown('### AnyFlow-Wan2.1-T2V — T2V')
            with gr.Row():
                with gr.Column():
                    prompt_bi = gr.Textbox(label='Prompt', lines=4)
                    model_repo_bi = gr.Dropdown(list(CFG.hub_repo_bidirectional), value=CFG.hub_repo_bidirectional[0], label='Model')
                    with gr.Row():
                        num_steps_bi = gr.Slider(CFG.min_steps, CFG.max_steps, value=CFG.default_steps, step=1, label='Inference steps')
                        seed_bi = gr.Number(0, label='Seed', precision=0)
                with gr.Column():
                    out_video_bi = gr.Video(label='Output', autoplay=True)
                    run_btn_bi = gr.Button('Generate')

            gr.Markdown('### Examples')
            for ex in t2v_examples:
                example_row(ex, target_prompt=prompt_bi)

            run_btn_bi.click(on_run_bidirectional, inputs=[prompt_bi, num_steps_bi, seed_bi, model_repo_bi], outputs=[out_video_bi])

demo.queue(default_concurrency_limit=1)

if __name__ == '__main__':
    demo.launch(show_error=True, share=True, server_name='0.0.0.0', server_port=7860)
