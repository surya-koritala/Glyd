---
license: apache-2.0
base_model:
- google/gemma-4-31B-it-qat-q4_0-unquantized
base_model_relation: quantized
pipeline_tag: text-generation
library_name: glyd
tags: [glyd, dense, compressed, consumer-gpu]
---

# Gemma 4 31B (QAT) · Glyd Wren

Google's 4-bit-trained Gemma 4 31B, kept bit for bit, running on one 24 GB card with Glyd's own engine. With Google's draft model
it writes about 3-4x faster than without, and the tokens are the same ones the big model picks on its own.

**Run it:** `glyd run glyd/gemma-4-31B-wren` (Glyd 0.31 or newer, Linux + NVIDIA)

> [!Note]
> These files are in Glyd's own format and run only with the Glyd engine. For Transformers, vLLM or llama.cpp, use
> [Google's weights](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-unquantized).

## What this is

Google's QAT checkpoint is already 4-bit codes times a scale per 32 weights. Wren on a QAT checkpoint stores exactly that: 4-bit
codes and a 16-bit scale per 32 weights, 4.5 bits per weight. Every weight decodes to the same bf16 value as Google's QAT
checkpoint. That is lossless relative to the QAT checkpoint, not to the original bf16 Gemma 4 31B, which Google's QAT model itself
differs from.

You can check it: `glyd verify glyd/gemma-4-31B-wren` decodes every tensor and compares its sha256 with the one recorded
from Google's files when it was packed.

```
weights                     16.2 GiB   (bf16: 58 GiB)       on disk: 17.4 GB (+1.0 GB draft model in draft/)
memory at 8K context        19.3 GiB of 24 GiB, draft model included
load                        24 s from disk (RTX 4090)
```

## Numbers

RTX 4090 24 GB (Runpod), 5 prompts, greedy, 256 tokens:

```
                              plain        with draft (8 a round)
chat, greedy                  46 tok/s     78-176 tok/s   the same tokens, checked token for token (depends on the text)
chat, sampled (T 1.0)         46 tok/s     87 tok/s       the same distribution as without the draft
read a 4K-token prompt        1.7 s
read a 30K-token prompt       20 s
RTX 3090 (plain)              26 tok/s
```

Quality (H100, transformers 5.19 bf16 as the reference, wikitext-2 test 32 x 512 as llama.cpp measures it):

```
vs Google's QAT weights (bf16)    mean KLD 0.018, median 0.003, top-1 96%    (chat prompts: KL 0.0005, 124/124 top-1)
4K-token prompt                   24/24 top-1, KL 0.0006
```

GSM8K 72/75, MMLU-Pro 70/75 (greedy, thinking off, first 75 of each, RTX 4090 with the draft model).

## Known limits

- Linux + NVIDIA RTX 30 series or newer with 24 GB. No Windows, Mac or AMD yet.
- Text only, no image input yet.
- One request at a time; others wait.
- A seed gives different (equally likely) text with the draft model on than off when sampling; greedy output is the same.

## Base model

[Gemma 4 31B-it QAT](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-unquantized) by Google DeepMind, Apache 2.0.
Draft model (in draft/, unchanged): [gemma-4-31B-it-qat-q4_0-unquantized-assistant](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-unquantized-assistant).
