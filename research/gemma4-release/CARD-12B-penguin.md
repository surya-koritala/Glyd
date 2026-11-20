---
license: apache-2.0
base_model:
- google/gemma-4-12B-it
base_model_relation: quantized
pipeline_tag: text-generation
library_name: glyd
tags: [glyd, penguin, lossless, dense, compressed, consumer-gpu]
---

# Gemma 4 12B · Glyd Penguin

Google's Gemma 4 12B with every weight kept bit for bit, in 15.0 GiB of GPU memory in place of 22.2 GiB, so it runs on one 24 GB
card with Glyd's own engine. With Google's draft model it writes about twice as fast as without, and the tokens are the same ones
the model picks on its own.

**Run it:** `glyd run glyd/gemma-4-12B-penguin` (Glyd 0.32 or newer, Linux + NVIDIA, 24 GB). `glyd run google/gemma-4-12B-it`
downloads this repo too.

> [!Note]
> These files are in Glyd's own format and run only with the Glyd engine. For Transformers, vLLM or llama.cpp, use
> [Google's weights](https://huggingface.co/google/gemma-4-12B-it).

## What this is

Penguin is Glyd's lossless level. The bf16 weights are stored in about 10.8 bits each and decoded on the GPU as the model runs.
Every weight decodes to exactly Google's bf16 value: this is not a quantization, nothing is rounded. Any difference in the output
comes from the order of the arithmetic, as between any two bf16 engines (the quality numbers below show how big that is).

You can check it: `glyd verify glyd/gemma-4-12B-penguin` decodes every tensor and compares its sha256 with the one recorded
from Google's files when it was packed.

```
weights                     15.0 GiB   (bf16: 22.2 GiB)     on disk: 16.1 GB (+0.88 GB draft model in draft/)
memory at 32K context       17.1 GiB of 24 GiB, draft model included
load                        12 s from disk (RTX 4090)
```

## Numbers

RTX 4090 24 GB (Runpod), `glyd serve` through the OpenAI API, one user, greedy, thinking off:

```
                              prompt tokens    read       writing, with the draft model
chat (short prompt)           34               0.02 s     89.8 tok/s
4K-token prompt               3,839            0.71 s     97.5 tok/s
28K-token prompt              28,374           7.63 s     64.3 tok/s

without the draft model       chat 50.8 tok/s, 4K-token prompt 48.3 tok/s
```

The original bf16 weights (22.2 GiB) do not load on a 24 GB card with room for context. If they did, their best case without a
draft model would be about 42 tok/s (the card's 1008 GB/s over 23.8 GB a token); penguin without the draft model writes 50.8.

Quality:

```
GSM8K       72 / 75     (every second question, the first 75; greedy, thinking off; RTX 4090 with the draft model)
MMLU-Pro    67 / 75     (the same way)

mean KLD against transformers in float32, wikitext-2 test, 32 x 512:
  this repo on Glyd's engine (bf16)       0.170    same top-1 88.8%
  transformers bf16 (eager attention)     0.132    same top-1 89.9%
  transformers bf16 (sdpa attention)      0.143    same top-1 90.4%
```

The instruct model on raw text moves with any change in rounding: transformers' own two bf16 paths are 0.132 apart from each other.
On 64 chunks the same engine (with Google's 12B QAT weights) measured 0.109 against transformers bf16's 0.102, with the batches of
16 chunks ±0.03 apart, so the gap in the 32-chunk numbers is within that spread.

## Known limits

- Linux + NVIDIA RTX 30 series or newer with 24 GB. No Windows, Mac or AMD yet.
- Text only, no image or audio input yet.
- One request at a time; others wait.
- A seed gives different (equally likely) text with the draft model on than off when sampling; greedy output is the same.

## Base model

[Gemma 4 12B-it](https://huggingface.co/google/gemma-4-12B-it) by Google DeepMind, Apache 2.0.
Draft model (in draft/, unchanged): [gemma-4-12B-it-assistant](https://huggingface.co/google/gemma-4-12B-it-assistant).

The weights here are Google's, under Google's license. Glyd's engine is open source under Apache-2.0:
[github.com/surya-koritala/Glyd](https://github.com/surya-koritala/Glyd).
