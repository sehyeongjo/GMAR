# GMAR: GRADIENT-DRIVEN MULTI-HEAD ATTENTION ROLLOUT FOR VISION TRANSFORMER INTERPRETABILITY

This repository is the official implementation of [GMAR: GRADIENT-DRIVEN MULTI-HEAD ATTENTION ROLLOUT FOR VISION TRANSFORMER INTERPRETABILITY](https://github.com/sehyeongjo/GMAR)

[Sehyeong Jo](https://sehyeongjo.github.io/), [Gangjae Jang](https://github.com/sehyeongjo/GMAR), [Haesol Park](https://scholar.google.com/citations?user=UG-9gMYAAAAJ&hl=en)

[![arXiv](https://img.shields.io/badge/arXiv-2311.18608-b31b1b.svg)](https://arxiv.org/abs/2406.08070)

## Abstract

The Vision Transformer (ViT) has made significant advance- ments in computer vision, utilizing self-attention mechanisms to achieve state-of-the-art performance across various tasks, including image classification, object detection, and seg- mentation. Its architectural flexibility and capabilities have made it a preferred choice among researchers and practition- ers. However, the intricate multi-head attention mechanism of ViT presents significant challenges to interpretability, as the underlying prediction process remains opaque. A criti- cal limitation arises from an observation commonly noted in transformer architectures: ”Not all attention heads are equally meaningful.” Overlooking the relative importance of specific heads highlights the limitations of existing interpretabil- ity methods. To address these challenges, we introduce Gradient-Driven Multi-Head Attention Rollout (GMAR), a novel method that quantifies the importance of each attention head using gradient-based scores. These scores are normal- ized to derive a weighted aggregate attention score, effec- tively capturing the relative contributions of individual heads. GMAR clarifies the role of each head in the prediction pro- cess, enabling more precise interpretability at the head level. Experimental results demonstrate that GMAR consistently outperforms traditional attention rollout techniques. This work provides a practical contribution to transformer-based architectures, establishing a robust framework for enhancing the interpretability of Vision Transformer models.

## Overview

![Image](https://github.com/user-attachments/assets/65d4788c-5443-4416-bec3-d1d5b8c96c5f)

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
@article{jo2024proxyllm,
  title={ProxyLLM: LLM-Driven Framework for Customer Support Through Text-Style Transfer},
  author={Jo, Sehyeong and Seo, Jungwon},
  journal={arXiv preprint arXiv:2412.09916},
  year={2024}
}
```
