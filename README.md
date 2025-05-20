# GMAR: GRADIENT-DRIVEN MULTI-HEAD ATTENTION ROLLOUT FOR VISION TRANSFORMER INTERPRETABILITY

This repository is the official implementation of [GMAR: GRADIENT-DRIVEN MULTI-HEAD ATTENTION ROLLOUT FOR VISION TRANSFORMER INTERPRETABILITY](https://arxiv.org/abs/2504.19414)

## Accepted IEEE International Conference on Image Processing(ICIP), 2025

[Sehyeong Jo](https://sehyeongjo.github.io/), [Gangjae Jang](https://github.com/sehyeongjo/GMAR), [Haesol Park](https://scholar.google.com/citations?user=UG-9gMYAAAAJ&hl=en)

[![arXiv](https://img.shields.io/badge/arXiv-2504.19414-b31b1b.svg)](https://arxiv.org/abs/2504.19414)

## Abstract

The Vision Transformer (ViT) has made significant advancements in computer vision, utilizing self-attention mechanisms to achieve state-of-the-art performance across various tasks, including image classification, object detection, and segmentation. Its architectural flexibility and capabilities have made it a preferred choice among researchers and practitioners. However, the intricate multi-head attention mechanism of ViT presents significant challenges to interpretability, as the underlying prediction process remains opaque. A critical limitation arises from an observation commonly noted in transformer architectures: ”Not all attention heads are equally meaningful.” Overlooking the relative importance of specific heads highlights the limitations of existing interpretability methods. To address these challenges, we introduce Gradient-Driven Multi-Head Attention Rollout (GMAR), a novel method that quantifies the importance of each attention head using gradient-based scores. These scores are normalized to derive a weighted aggregate attention score, effectively capturing the relative contributions of individual heads. GMAR clarifies the role of each head in the prediction process, enabling more precise interpretability at the head level. Experimental results demonstrate that GMAR consistently outperforms traditional attention rollout techniques. This work provides a practical contribution to transformer-based architectures, establishing a robust framework for enhancing the interpretability of Vision Transformer models.

## Overview

<img width="1083" alt="Image" src="https://github.com/user-attachments/assets/1b027c9c-4639-4759-8a50-86b13f94dc44" />

<br>

GMAR leverages class-specific gradient information to quantify the contribution of individual attention heads to the model’s predictions.

## Result

![Image](https://github.com/user-attachments/assets/94f758b7-3108-40ab-90b6-2c7306bc44fc)

## Getting Started

Clone the repo:

```bash
  git clone https://github.com/sehyeongjo/GMAR.git
  cd GMAR
```

### Requirements

- Python 3.9
- PyTorch >= 2.1.0 (CUDA 11.8)
  Create your environment. We recommend using the following comments.
  ```bash
  conda env create -f environment.yaml
  ```

### Usage

Run Demo

```bash
python test.py --pretrained {PRETRAINED_MODEL}
```

## Citation

If you find our work interesting, please cite our paper.

```bibtex
@article{jo2025gmar,
  title={GMAR: Gradient-Driven Multi-Head Attention Rollout for Vision Transformer Interpretability},
  author={Jo, Sehyeong and Jang, Gangjae and Park, Haesol},
  journal={arXiv preprint arXiv:2504.19414},
  year={2025}
}
```
