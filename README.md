# 🚀 AnyFlow Gradio Demo

[![arXiv](https://img.shields.io/badge/arXiv-Coming%20Soon-b31b1b.svg)]()
[![Project Page](https://img.shields.io/badge/Project-Website-orange)](https://nvlabs.github.io/AnyFlow/)
[![Gradio Demo](https://img.shields.io/badge/Gradio-Demo-blue?logo=gradio)](https://nvlabs.github.io/AnyFlow/demo/)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Models-yellow)](https://huggingface.co/collections/nvidia/anyflow)

## 📖 Overview

We introduce **AnyFlow**, the first any-step video diffusion framework built on flow maps. **AnyFlow** offers these key features:

- ⚡ **Any-Step Generation**: Unlike traditional distilled models tied to fixed step budgets, **AnyFlow** enables a single model to adapt to arbitrary inference budgets. It achieves high-quality few-step generation while providing stable improvements as more sampling steps are added.
- 🔀 **Multiple Architectures**: **AnyFlow** supports any-step distillation for both **causal** and **bidirectional** video diffusion models.
- 🎬 **Multiple Tasks**: **AnyFlow** supports Text-to-Video, Image-to-Video, and Video-to-Video generation within one causal video diffusion model.
- 📈 **Scalable Performance**: **AnyFlow** is validated from **1.3B** up to **14B** parameters.

## 🛠️ Quick Start

```bash
git clone -b anyflow-gradio-demo https://github.com/NVlabs/AnyFlow.git && cd AnyFlow
conda create -n far python=3.10 && conda activate far
pip install -r requirements.txt
python app.py
```