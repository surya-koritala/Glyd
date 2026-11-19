---
license: apache-2.0
base_model: google/gemma-4-12B-it
tags: [glyd, lossless]
---

# gemma-4-12B-it, Glyd

[google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) by Google, unchanged: every weight in this repository is bit for bit the original's. Stored with [Glyd](https://getglyd.com), it takes 30% less space on disk and loads packed into GPU memory, with no packing step on the first run.

Lossless AI compression: 33% less GPU memory, bit for bit.

| Measured | bf16 (the original's tensors) | Glyd |
| :-- | --: | --: |
| Weights on disk | 23.9 GB | **16.8 GB** (29.8% less) |
| Weights in GPU memory, loaded | 23.9 GB | **17.8 GB** |
| Tensors checked bit for bit against the original's | | 679 |

Measured on an NVIDIA A100 80 GB (Runpod), 2026-10-06, Glyd 0.29.0.

## Run it

```bash
pip install "glyd[gpu]"
```
```python
import glyd
model = glyd.from_pretrained("glyd/gemma-4-12B-it")
```

## License and attribution

The model is google/gemma-4-12B-it by Google, under the Apache-2.0 license (LICENSE in this repository); its weights are the original's, stored in Glyd's format (the safetensors files and glyd.json differ from the original's files; NOTICE says so). Loading it takes Glyd, open source under the Apache License 2.0 (https://github.com/surya-koritala/Glyd).
