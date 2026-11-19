---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags: [glyd, lossless]
---

# Qwen3.8-27B, Glyd

[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) by the Qwen team (Alibaba Cloud), unchanged: every weight in this repository is bit for bit the original's. Stored with [Glyd](https://getglyd.com), it takes 31% less space on disk and loads packed into GPU memory, with no packing step on the first run. It holds the language model, for text: the vision tower and the multi-token prediction head are not in this repository (for images, use the original).

Lossless AI compression: 33% less GPU memory, bit for bit.

| Measured | bf16 (the original's tensors) | Glyd |
| :-- | --: | --: |
| Weights on disk | 53.8 GB | **37.0 GB** (31.2% less) |
| Weights in GPU memory, loaded | 53.8 GB | **36.5 GB** |
| Tensors checked bit for bit against the original's | | 852 |

Measured on an NVIDIA A100 80 GB (Runpod), 2026-10-06, Glyd 0.29.0.

## Run it

```bash
pip install "glyd[gpu]"
```
```python
import glyd
model = glyd.from_pretrained("glyd/Qwen3.8-27B")
```

## License and attribution

The model is Qwen/Qwen3.8-27B by the Qwen team (Alibaba Cloud), under the Apache-2.0 license (LICENSE in this repository); its weights are the original's, stored in Glyd's format (the safetensors files and glyd.json differ from the original's files; NOTICE says so). Loading it takes Glyd, under the Business Source License 1.1: free for personal and other non-commercial use; commercial use needs a license (suryakoritala@getglyd.com).
