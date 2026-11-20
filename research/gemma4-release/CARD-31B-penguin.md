---
license: apache-2.0
base_model:
- google/gemma-4-31B-it
base_model_relation: quantized
pipeline_tag: text-generation
library_name: glyd
tags: [glyd, penguin, lossless, dense, compressed]
---

# Gemma 4 31B · Glyd Penguin

Google's Gemma 4 31B with every weight kept bit for bit, in 38.4 GiB of GPU memory in place of 57.2 GiB, so it runs on one 48 GB
card with Glyd's own engine. With Google's draft model it writes about twice as fast as without, and the tokens are the same ones
the model picks on its own.

**Run it:** `glyd run glyd/gemma-4-31B-penguin` (Glyd 0.32 or newer, Linux + NVIDIA, 48 GB). `glyd run google/gemma-4-31B-it`
downloads this repo too.

> [!Note]
> These files are in Glyd's own format and run only with the Glyd engine. For Transformers, vLLM or llama.cpp, use
> [Google's weights](https://huggingface.co/google/gemma-4-31B-it).

## What this is

Penguin is Glyd's lossless level. The bf16 weights are stored in about 10.8 bits each and decoded on the GPU as the model runs.
Every weight decodes to exactly Google's bf16 value: this is not a quantization, nothing is rounded. Any difference in the output
comes from the order of the arithmetic, as between any two bf16 engines (the quality numbers below show how big that is).

You can check it: `glyd verify glyd/gemma-4-31B-penguin` decodes every tensor and compares its sha256 with the one recorded
from Google's files when it was packed.

```
weights                     38.4 GiB   (bf16: 57.2 GiB)     on disk: 41.3 GB (+0.97 GB draft model in draft/)
memory in use               43.3 GiB of 48 GB at 32K context (RTX A6000, 49140 MiB), draft model included
                            on the 46068 MiB A6000: 42.8 GiB at 25K context (the context is sized to fit)
load                        24-26 s from disk
```

## Numbers

`glyd serve` through the OpenAI API, one user, greedy, thinking off, 512 tokens for chat and 256 after each long prompt.

RTX A6000 48 GB (Runpod):

```
                              prompt tokens    read       writing, with the draft model    without
chat (short prompt)           33               0.09 s     34.3 tok/s                       13.3 tok/s
4K-token prompt               3,217            3.25 s     21.2 tok/s                       12.7 tok/s
24K-token prompt              23,922           35.5 s     15.0 tok/s                       11.8 tok/s
```

H100 PCIe 80 GB (Lambda):

```
                              prompt tokens    read       writing, with the draft model    without
chat (short prompt)           33               0.07 s     61.8 tok/s                       30.0 tok/s
4K-token prompt               3,217            1.93 s     44.8 tok/s                       28.5 tok/s
24K-token prompt              23,922           23.0 s     34.1 tok/s                       24.1 tok/s
```

Without the draft model the A6000 writes at about 71% of what its memory bandwidth allows for these weights (768 GB/s over 41.2 GB
a token is 18.6 tok/s). The original bf16 weights (57.2 GiB) do not fit a 48 GB card.

Quality:

```
GSM8K       72 / 75     (every second question, the first 75; greedy, thinking off; H100 with the draft model)
MMLU-Pro    67 / 75     (the same way)

mean KLD, wikitext-2 test, 32 x 512:
  transformers bf16 (eager) against transformers float32       0.162
  transformers bf16 (sdpa) against transformers float32        0.179
  transformers bf16, eager against sdpa                        0.164
  this repo on Glyd's engine against transformers bf16 (eager) 0.200
```

The instruct model on raw text moves with any change in rounding: transformers' own two bf16 paths are 0.164 apart from each other,
and this repo is 0.200 from one of them. The weights themselves are identical.

## Known limits

- Linux + NVIDIA RTX 30 series or newer with 48 GB (RTX A6000, RTX 6000 Ada, L40S) or more. No Windows, Mac or AMD yet.
- Text only, no image or audio input yet.
- One request at a time; others wait.
- On H100 the engine is not tuned yet: reading a prompt runs at a fraction of the card's peak, and without the draft model it writes
  at about 62% of what the card's memory bandwidth allows.
- A seed gives different (equally likely) text with the draft model on than off when sampling; greedy output is the same.

## Base model

[Gemma 4 31B-it](https://huggingface.co/google/gemma-4-31B-it) by Google DeepMind, Apache 2.0.
Draft model (in draft/, unchanged): [gemma-4-31B-it-assistant](https://huggingface.co/google/gemma-4-31B-it-assistant).

The weights here are Google's, under Google's license. Glyd's engine is open source under Apache-2.0:
[github.com/surya-koritala/Glyd](https://github.com/surya-koritala/Glyd).
