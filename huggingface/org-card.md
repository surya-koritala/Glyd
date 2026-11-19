---
title: README
emoji: 🧊
colorFrom: gray
colorTo: green
sdk: static
pinned: false
---

# Glyd

**Lossless AI compression: 33% less GPU memory, bit for bit.**

Glyd keeps a bf16 model's weights and KV cache in fewer bits in GPU memory and rebuilds the exact values as the model runs. It is the same model in about a third less memory, so it fits a smaller GPU or serves more requests on the same one.

## Levels

Each model here comes at a level, the tag after its name in `glyd run`, and its repo is `glyd/<model>-<level>`:

| Level | What it keeps |
| :-- | :-- |
| penguin | lossless: every weight decodes to its bf16 value (the default) |
| kestrel | Glyd's own quantization, about 6.5 bits per weight |
| swift | Glyd's own quantization, about 5.5 bits per weight |

```bash
curl -LsSf https://getglyd.com/install.sh | sh
glyd run Qwen/Qwen3.8-27B:swift
```

```bash
pip install "glyd[vllm]"
vllm serve Qwen/Qwen3.5-9B --quantization glyd
```

```python
import glyd
model = glyd.from_pretrained("Qwen/Qwen3.5-9B")
```

Linux (x86_64 or aarch64) with an NVIDIA GPU: Ampere, Ada, Hopper or Blackwell, driver 580 or newer.

| Measured | bf16 | Glyd |
| :--- | ---: | ---: |
| Qwen3.5-9B under vLLM on an RTX 4090: weights | 17.66 GiB | **12.68 GiB** (−28.2%) |
| Qwen3.5-9B on an RTX 4090 (24 GB): context | out of memory at start | **189,440 tokens** |
| Qwen3.8-27B under vLLM on an H100: weights | 50.22 GiB | **38.77 GiB** (−22.8%) |
| Qwen3.8-27B: its layers' matrices, read back bit for bit | 49.52 GB | **33.28 GB** (−32.8%) |

Every result and its logs: [getglyd.com/benchmarks](https://getglyd.com/benchmarks/).

License: BUSL-1.1. Free for personal and non-commercial use on your own computers; commercial use needs a license; each version becomes Apache-2.0 four years after release. Contact: suryakoritala@getglyd.com.

[getglyd.com](https://getglyd.com) · [GitHub](https://github.com/surya-koritala/Glyd)
