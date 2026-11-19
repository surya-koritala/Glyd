# How it works

Notes on what the engine and the compressed weights do, for anyone curious why a 180B MoE runs on a 16 GB card. This is the Hummingbird
level, and it is lossy: the answers are close to the original model's, not identical.

## The model

Qwen3.8-Flash-Next has 48 layers. Each layer has 512 routed experts plus one shared expert, and for each token the router picks 10 of
the 512 (the shared one always runs). The experts are 120.8B of the parameters, so they are what has to shrink.

## Bits by how often an expert gets used

I ran a 64K-token sample (fiction, code, technical text and chat) through the model and counted how often the router picks each
expert. Use is very uneven:

```
share of experts (busiest first)     share of routed weight they carry
top 10%                              ~41%
top 25%                              ~65%
top 50%                              ~87%
```

So the bits go where the use is. In precise mode, the busiest experts in each layer keep the most bits (4, 3 or 2 bits, from most used
to least). Fast mode keeps every expert at 2 bits a weight (since v0.2.2 those bits are packed to stay closer to the original). On average:

```
                                     precise         fast
experts, avg bits/weight             ~2.7            ~2.0
attention + rest                     4-5 bit         4-5 bit
```

"Attention + rest" is everything outside the routed experts: attention, embeddings, the shared expert and the router. Those run for
every token, so they stay at higher precision.

The bit widths are fixed when the pack is made. Nothing is quantized on the fly while it runs; the engine reads the pack as it is.

## Where the experts live

The whole model doesn't fit in 16 GB, so it's split. The GPU keeps as many of the busiest experts as fit, and the rest stay in RAM. On
a 16 GB card roughly 70-76% of expert reads hit an expert that is already on the GPU.

For the experts that miss, the CPU computes them from RAM while the GPU works on the rest of the layer, and the results are added
together. A bigger context leaves less room for experts on the GPU, which is why 32K is the fastest setting.

One shortcut: an expert that is not on the GPU and that gets only a tiny routing weight from every token in the step is skipped.
Those experts add little to the output, and skipping them saves CPU time. This is an approximation and it does change
the output a little. I measured it: GSM8K and the needle tests came out the same with and without it.

## Drafting a token ahead

Qwen3.8-Flash-Next ships its own multi-token prediction (MTP) layer. The engine uses it to draft 1 token ahead, and the big model checks
the draft in the same pass as the current token. About 80-90% of drafts are kept, so most steps write two tokens instead
of one. A draft the model disagrees with is thrown away.

## Long prompts

A long prompt goes through in chunks, so a 32K or 261K prompt doesn't blow up memory. While a prompt is being read, the GPU's expert
slots are lent to it, and they're given back for writing the answer. Reading times are in the README.

## Lossy vs lossless

Everything above is the Hummingbird level, the most compact one, and it is lossy. Glyd also has a lossless level (Penguin) for other
models, but not for this one.

## Known limits

- Text only. The base model's vision part isn't supported yet.
- Linux + NVIDIA only, 16 GB minimum.
- One request at a time, and others queue.
- Fast mode is noticeably weaker on long code.
- If you don't have enough free RAM, it gets very slow and the disk gets busy. Close the browser tabs eating your RAM.
- The expert skipping above is a small approximation on top of the compression.
