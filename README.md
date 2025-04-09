# MI2P：Multimodal Information Injection Plug-in

This repository contains an implementation of the **CVPR 2022 paper**:
**"Expanding Large Pre-trained Unimodal Models with Multimodal Information Injection for Image-Text Multimodal Classification"**

> 📄 [Paper PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Liang_Expanding_Large_Pre-Trained_Unimodal_Models_With_Multimodal_Information_Injection_for_CVPR_2022_paper.pdf)

---

## Overview
This project implements the MI2P (Multimodal Information Injection Plug-in) module for injecting textual or visual information into pre-trained unimodal models like DenseNet (image) and BERT (text). This enables better multimodal classification by modeling fine-grained inter-modal interactions while keeping the unimodal backbone structure unchanged.

---
## Implementation

The original CVPR 2022 paper introduces a bidirectional multimodal injection framework:

Injecting textual features into visual models (e.g. BERT → DenseNet)

Injecting visual features into language models (e.g. DenseNet → BERT)

<img src="https://github.com/user-attachments/assets/fa52d2e8-7b10-4f2a-bd09-d18d8b088bf9" width="350"/>
<img src="https://github.com/user-attachments/assets/7cae5992-02f1-45fe-86b2-f8e2555e6948" width="350"/>


>The following diagram is taken from the CVPR 2022 paper  
> *“Expanding Large Pre-trained Unimodal Models with Multimodal Information Injection for Image-Text Multimodal Classification”*  



In this repo, we implement the first direction only (as shown in the figure above):

✅ Injecting BERT-derived textual features into DenseNet at multiple layers.

Specifically, MI2P modules are inserted after each DenseNet block (denseblock1 to denseblock4) to fuse word-wise BERT outputs with intermediate CNN feature maps.

## Included Models

- MI2P: a plug-in module that injects word-level features into visual features via attention.

- MultimodalDenseNet: a full image-text classification model combining DenseNet and BERT, with MI2P modules injected at four DenseNet stages.

## Usage Example



```python
# If the filename is model.py
from model import MultimodalDenseNet
import torch

# Create dummy input data
images = torch.randn(8, 3, 224, 224)         # A batch of images with size 224x224
input_ids = torch.randint(0, 30522, (8, 128))  # Simulated BERT token IDs
attention_mask = torch.ones_like(input_ids)   # Attention mask (typically 0 or 1)

# Initialize the model and run a forward pass
model = MultimodalDenseNet()
outputs = model(images, input_ids, attention_mask)

print(outputs.shape)  # Output classification results; default is 101 classes (e.g., Food101)
```
### Input Format

images: Tensor of shape (B, 3, 224, 224)

input_ids: Tensor of shape (B, L), BERT token ids

attention_mask: Tensor of shape (B, L)

### Output Format

outputs: Tensor of shape (B, 101) — classification logits for 101 classes (e.g., Food101)

## Dependencies
To install all required packages, use the following command:
```bash
pip install -r requirements.txt
```

## Citation

```bibtex
@inproceedings{liang2022mi2p,
  title     = {Expanding Large Pre-Trained Unimodal Models With Multimodal Information Injection},
  author    = {Liang, Tao and Lin, Guosheng and Wan, Mingyang and Li, Tianrui and Ma, Guojun and Lv, Fengmao},
  booktitle = {CVPR},
  year      = {2022}
}
