---
license: other
license_name: nvidia-nemotron-open-model-license
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/
base_model: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
base_model_relation: quantized
library_name: glyd
pipeline_tag: text-generation
tags:
- glyd
- hummingbird
- lossy
- nemotron
---

# Nemotron 3 Super (120B-A12B) · Glyd hummingbird

**48.7 GB, 80% smaller than bf16 (241.4 GB): a 120B model on one 16–24 GB consumer GPU.** Glyd's hummingbird level of
[nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16) (commit
[`2dc98e2a`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16/tree/2dc98e2a)): the routed experts (93% of the weights) at 2.99 bits
per weight, every other matrix at 6 bits. Not lossless.

```sh
curl -LsSf https://getglyd.com/install.sh | sh
glyd run nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16:hummingbird
```

## How it fits

Each expert neuron is a row of 1,024 values with its own small codebook (8 entries at 3 bits, 4 at 2), in one rotated basis per layer; each layer's
40% most-used experts are at 3 bits and the rest at 2. The codes come from a one-time calibration against the bf16 model. When the model is larger than
the GPU, Glyd's engine keeps the most-asked-for experts on the GPU and the rest in the computer's memory, and the CPU computes the experts that are not
on the GPU from where they are held, while the GPU runs the others.

## Speed

Decode, one user, 255 greedy tokens of a chat prompt (`glyd-engine generate`; the experts not on the GPU computed by the CPU):

| GPU | GPU memory in use (4K context) | experts a layer on the GPU | tokens a second |
|---|--:|--:|--:|
| RTX 4090 (24 GB) | 22.2 GiB | 190 of 512 | 45.5 |
| RTX 3090 (24 GB) | 22.2 GiB | 192 of 512 | 37.0 |
| 16 GB card's budget (an RTX 4090 held to 15 GiB free, as an RTX 4080 has) | 14.0 GiB | 84 of 512 | 38.1 |

## Benchmarks

| benchmark | this checkpoint | questions |
|---|--:|--:|
| GSM8K (10-shot, non-thinking) | 97.0% | 300 |
| MMLU-Pro (0-shot chain of thought, non-thinking) | 76.6% | 222 |

Inspect AI 0.3.278 with inspect_evals 0.24.0, as shipped, through `glyd serve`, temperature 0. The bf16 model was not run on the same harness.

Closeness to bf16 (the bf16 model run layer by layer as the reference):

| measure | result |
|---|--:|
| top-1 token the same as bf16's, whole model, 97 teacher-forced positions | 96.9% |
| KL divergence from bf16 there | 0.014 |
| top-1 token the same as bf16's, routed experts only, 49,128 held-out positions over six domains | 93.4% |
| perplexity there (bf16: 3.021) | 3.334 (+10.3%) |

## Good to know

- Runs on Glyd's engine (`glyd run`, `glyd serve`): Linux with an NVIDIA GPU of 16 GB or more (Ampere or later), driver 580 or newer. Not for vLLM or transformers.
- Computer's memory: 64 GB for full speed (the experts take 39.2 GiB, held in memory). A 32 GB PC runs it too: the experts that do not fit are read from the SSD as they are needed (measured with a 7 GB/s NVMe: 17.7 tokens a second on a 24 GB card, 13.1 on a 16 GB card). `--host-memory GIB` sets the share; the default is what the computer has free.
- The Hub's "Model size" and precision labels count packed bytes.

## License

Weights: the [NVIDIA Nemotron Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/),
from [nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16); see NOTICE. Licensed by NVIDIA
Corporation under the NVIDIA Nemotron Model License. Glyd's engine, which runs them: BUSL-1.1, free for personal and non-commercial use on your own
computers; commercial use needs a license. [getglyd.com](https://getglyd.com)
