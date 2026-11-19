---
license: apache-2.0
base_model:
- google/gemma-4-12B-it-qat-q4_0-unquantized
base_model_relation: quantized
pipeline_tag: text-generation
library_name: glyd
tags: [glyd, dense, compressed, consumer-gpu]
---

# Gemma 4 12B (QAT) · Glyd Wren

Google's 4-bit-trained Gemma 4 12B, kept bit for bit, running with Glyd's own engine. It takes 6.3 GiB of GPU memory and comes
with Google's draft model, which makes it write 2-3x faster without changing the output.

**Run it:** `glyd run glyd/gemma-4-12B-wren` (Glyd 0.31 or newer, Linux + NVIDIA)

> [!Note]
> These files are in Glyd's own format and run only with the Glyd engine. For Transformers, vLLM or llama.cpp, use
> [Google's weights](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized).

## What this is

Google's QAT checkpoint is already 4-bit codes times a scale per 32 weights. Wren on a QAT checkpoint stores exactly that: 4-bit
codes and a 16-bit scale per 32 weights, 4.5 bits per weight. Every weight decodes to the same bf16 value as Google's QAT
checkpoint. That is lossless relative to the QAT checkpoint, not to the original bf16 Gemma 4 12B, which Google's QAT model itself
differs from.

You can check it: `glyd verify glyd/gemma-4-12B-wren` decodes every tensor and compares its sha256 with the one recorded
from Google's files when it was packed (7 s on a 16-core CPU).

```
weights                     6.3 GiB    (bf16: 23.8 GB)     on disk: 6.8 GB (+0.85 GB draft model in draft/)
GPU memory at 4K context    7.9 GiB, draft model included
load                        11 s from disk
```

## Numbers

RTX 4080 SUPER 16 GB, Ryzen 9 7950X3D, Ubuntu, greedy, thinking off.

Engine test, 5 prompts, 256 tokens:

```
                       plain        with the draft model (8 a round)
writing                82.5 tok/s   232 tok/s on these prompts (3.7 tokens a round), output token for token the same
```

Served (`glyd serve`, OpenAI API), measured with bench/compare.py:

```
                       prompt   first token   prompt read     writing (with the draft model)
chat (short)             34       0.03 s        1257 tok/s     108.7 tok/s
~4K prompt             3839       1.11 s        3463            96.1
~30K prompt           28374      12.9 s         2199            77.0
GSM8K     72 / 75     (first 75, greedy)
MMLU-Pro  66 / 75     (first 75, greedy; 1 answer hit the 3,072-token cap)
```

How much the draft model helps depends on the text: about 1.7-2x on chat, more on predictable text.

## Known limits

- Linux + NVIDIA RTX 30 series or newer. No Windows, Mac or AMD yet.
- Text only, no image or audio input yet.
- One request at a time; others wait.
- When sampling, a seed gives different (equally likely) text with the draft model on than off. Greedy output is the same.

## Base model

[Gemma 4 12B-it QAT](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized) by Google DeepMind, Apache 2.0.
Draft model (in draft/, unchanged): [gemma-4-12B-it-qat-q4_0-unquantized-assistant](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized-assistant).
