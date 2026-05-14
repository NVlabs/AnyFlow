# 🚀 AnyFlow

[![arXiv](https://img.shields.io/badge/arXiv-2605.13724-b31b1b.svg)](https://arxiv.org/abs/2605.13724)
[![Project Page](https://img.shields.io/badge/Project-Website-orange)](https://nvlabs.github.io/AnyFlow/)
[![Gradio Demo](https://img.shields.io/badge/Gradio-Demo-blue?logo=gradio)](https://nvlabs.github.io/AnyFlow/demo/)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Models-yellow)](https://huggingface.co/collections/nvidia/anyflow)

## 📖 Overview

https://github.com/user-attachments/assets/5698cdea-847f-4732-9910-c78f07aa0404

We introduce **AnyFlow**, the first any-step video diffusion framework built on flow maps. **AnyFlow** offers these key features:

- ⚡ **Any-Step Generation**: Unlike traditional distilled models tied to fixed step budgets, **AnyFlow** enables a single model to adapt to arbitrary inference budgets. It achieves high-quality few-step generation while providing stable improvements as more sampling steps are added.
- 🔀 **Multiple Architectures**: **AnyFlow** supports any-step distillation for both **causal** and **bidirectional** video diffusion models.
- 🎬 **Multiple Tasks**: **AnyFlow** supports Text-to-Video, Image-to-Video, and Video-to-Video generation within one causal video diffusion model.
- 📈 **Scalable Performance**: **AnyFlow** is validated from **1.3B** up to **14B** parameters.

## 🛠️ Setup Environment

#### 1️⃣ Create conda environment

```bash
conda create -n far python=3.10
conda activate far
```

#### 2️⃣ Install PyTorch & dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt --no-build-isolation
```

#### 3️⃣ Install pre-commit (optional)

```bash
pre-commit install
```

#### 🔧 On NVIDIA cluster

If you are on an NVIDIA internal cluster, install the logger utils:

```bash
pip install --index-url=https://sc-hw-artf.nvidia.com/artifactory/api/pypi/hwinf-mlwfo-pypi/simple --upgrade one-logger-utils
```

## 🎮 Run Demo

#### 📥 Download pretrained model

```bash
hf download nvidia/AnyFlow-FAR-Wan2.1-1.3B-Diffusers --repo-type model --local-dir experiments/pretrained_models/AnyFlow-FAR-Wan2.1-1.3B-Diffusers
hf download nvidia/AnyFlow-FAR-Wan2.1-14B-Diffusers --repo-type model --local-dir experiments/pretrained_models/AnyFlow-FAR-Wan2.1-14B-Diffusers
hf download nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers --repo-type model --local-dir experiments/pretrained_models/AnyFlow-Wan2.1-T2V-1.3B-Diffusers
hf download nvidia/AnyFlow-Wan2.1-T2V-14B-Diffusers --repo-type model --local-dir experiments/pretrained_models/AnyFlow-Wan2.1-T2V-14B-Diffusers
```

#### ▶️ Start demo

```bash
python demo.py \
    model_path=experiments/pretrained_models/AnyFlow-Wan2.1-T2V-1.3B-Diffusers \
    task_type=t2v \
    save_dir=results/demo/AnyFlow-Wan2.1-T2V-1.3B-Diffusers
```


## 🏋️ Training

Training uses **`mode: train`** configs under `options/train/anyflow/`.

#### Data preparation

See **[docs/DATA.md](docs/DATA.md)** for how to construction training dataset. We provide an exmaple dummy dataset for quick start:
```bash
hf download dc-ai/vidprom_dummy --repo-type dataset --local-dir datasets/vidprom_dummy
```

#### Launch

```bash
torchrun --nnodes 1 --nproc_per_node=8 --master_port 17154 \
    -m far.main \
    config_path=options/train/anyflow/farwan_causal/pretrain/train_farwan1b_student_shift5_81f_480p_lr5e-5_6k_b32.yml
```

Set `--nproc_per_node` to the number of GPUs you use. Logs and checkpoints go under `experiments/<run_name>/` (the `name` field in the YAML).


## 🤗 Using with Hugging Face `diffusers`

The `nvidia/AnyFlow-*-Diffusers` checkpoints can be loaded through the standard `diffusers` API:

```python
import torch
from diffusers import AnyFlowPipeline
from diffusers.utils import export_to_video

pipe = AnyFlowPipeline.from_pretrained(
    "nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers",
    torch_dtype=torch.bfloat16,
).to("cuda")

video = pipe(
    prompt="A red panda eating bamboo in a forest, cinematic lighting",
    num_inference_steps=4,
    num_frames=33,
).frames[0]
export_to_video(video, "anyflow_t2v.mp4", fps=16)
```

For the FAR variant (T2V / I2V / V2V via `context_sequence`):

```python
import torch
from diffusers import AnyFlowFARPipeline
from diffusers.utils import export_to_video

pipe = AnyFlowFARPipeline.from_pretrained(
    "nvidia/AnyFlow-FAR-Wan2.1-1.3B-Diffusers",
    torch_dtype=torch.bfloat16,
).to("cuda")

video = pipe(
    prompt="A red panda eating bamboo in a forest, cinematic lighting",
    num_inference_steps=4,
    num_frames=81,
).frames[0]
export_to_video(video, "anyflow_far_t2v.mp4", fps=16)
```

The same checkpoints also work with the `demo.py` and training entry points in this repository. See the [diffusers AnyFlow docs](https://huggingface.co/docs/diffusers/api/pipelines/anyflow) for the full reference.


## 📊 Evaluation

Evaluation uses **`mode: eval`** configs under `options/test/anyflow/`.

#### VBench preparation

The evaluators set `VBENCH_CACHE_DIR` to **`experiments/pretrained_models/vbench`**. Download the VBench model bundle there:

```bash
hf download dc-ai/vbench_pretrained_models --repo-type model --local-dir experiments/pretrained_models/vbench
```

If you run configs that evaluate VBench I2V, download reference images for evaluation:

```bash
hf download dc-ai/vbench_i2v --repo-type dataset --local-dir datasets/vbench_i2v
```

#### Launch

```bash
torchrun --nnodes 1 --nproc_per_node=8 --master_port 17154 \
    -m far.main \
    config_path=options/test/anyflow/test_AnyFlow-FAR-Wan2.1-1.3B-Diffusers.yml
```

Outputs and logs are written under **`results/<run_name>/`**.


## 📜 License

This project is released under the **Apache License 2.0**. See [LICENSE](LICENSE) for full text.


## 📬 Contact & Discussion

Feel free to open an issue or email [Yuchao Gu](yuchaogu9710@gmail.com) for questions about the codebase.

## ⭐ Acknowledgement

This codebase is built on [Diffusers](https://github.com/huggingface/diffusers). We also refer to implementations from [FAR](https://github.com/showlab/FAR), [Self-Forcing](https://github.com/guandeh17/Self-Forcing), and [TiM](https://github.com/WZDTHU/TiM). We thank the authors for open-sourcing their work.

## 📚 Citation

If you find AnyFlow useful in your research, please cite our work:

```bibtex
@article{gu2026anyflow,
    title={AnyFlow: Any-Step Video Diffusion Model with On-Policy Flow Map Distillation},
    author={Gu, Yuchao and Fang, Guian and Jiang, Yuxin and Mao, Weijia and Han, Song and Cai, Han and Shou, Mike Zheng},
    journal={arXiv preprint arXiv:2605.13724},
    year={2026}
}

@article{gu2025long,
    title={Long-Context Autoregressive Video Modeling with Next-Frame Prediction},
    author={Gu, Yuchao and Mao, weijia and Shou, Mike Zheng},
    journal={arXiv preprint arXiv:2503.19325},
    year={2025}
}
```
