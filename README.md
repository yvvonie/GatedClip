# GatedCLIP: Gated Multimodal Fusion for Hateful Memes Detection

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)

This repository contains the official implementation for the paper **"GatedCLIP: Gated Multimodal Fusion for Hateful Memes Detection"**. 

Detecting hateful content in multimodal memes is challenging because harmful messages often emerge from the complex interplay between individually benign images and tex. GatedCLIP is a vision-language model that enhances CLIP's multimodal capabilities with specialized architectural improvements specifically designed for hateful memes detection.

## 🌟 Key Features

* **Parameter-Efficient Fine-Tuning:** We freeze the CLIP encoders and train only lightweight projection heads, a fusion gate, and classification layers. This adds only ~350K trainable parameters (0.2% of CLIP's 151M parameters).
* **Dynamic Gated Fusion:** A learnable gate dynamically weights the contributions of visual and textual features for each specific example. The model learns to favor image features for visually explicit memes and text features for linguistically offensive content.
* **High Performance:** Achieves an AUROC of 0.66 on the Hateful Memes validation set, substantially outperforming the simple CLIP averaging baseline.
* **Fast & Efficient:** Training completes in approximately 40 minutes on a single NVIDIA GPU using mixed precision (FP16). Inference processes over 100 examples per second.

## 🏗️ Architecture

GatedCLIP introduces three key architectural enhancements over the standard CLIP model.
1.  **Projection Heads:** Maps CLIP's 512-dimensional embeddings into a 128-dimensional task-optimized semantic space.
2.  **Gated Fusion Mechanism:** Adaptively weights visual and textual features based on their relevance to each meme.
3.  **Contrastive Alignment:** A contrastive learning objective that maintains cross-modal semantic alignment between the projected image and text representations.
## 📊 Results

Performance comparison on the Hateful Memes validation set:

| Method | AUROC | Accuracy |
| :--- | :---: | :---: |
| CLIP Baseline (avg) | 0.49 | 0.50 |
| **GatedCLIP (ours)** | **0.66** | **0.59** |
| *Improvement* | *+0.17* | *+0.09* |

The codebase is built using PyTorch and the Hugging Face Transformers library.

