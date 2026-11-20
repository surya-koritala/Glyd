# About this guide

## What Glyd does

Glyd compresses an LLM and runs it. It packs the weights and the KV cache losslessly and decodes them inside the GPU kernels,
so a model that does not fit your GPU in bf16 often fits and answers exactly as before. When lossless is not enough there are
lossy levels, experts kept in RAM, and more GPUs. It is one program written in Rust, from the compressed file to the
tokens, with an OpenAI and Anthropic compatible server and a chat page.

## Features

- Lossless weights and KV cache, bit for bit, decoded in the kernels.
- Four lossy levels (kestrel, swift, wren, hummingbird) for when a model still does not fit.
- Mixture-of-experts models with their experts in RAM or on the SSD.
- One model across the GPUs of a machine, and across machines with Starlings (beta).
- A server for the OpenAI and Anthropic APIs, with tools and constrained JSON, and a chat page.
- `fit`: the plan for your GPU before anything is downloaded.

## How it works

1. You name a model: `google/gemma-4-12B-it`, or a path.
2. `fit` reads the model's config and plans a level, where the weights go and how much context fits.
3. On the first run Glyd downloads the model, packs it for your GPU and builds the KV tables.
4. The kernels read the packed weights and decode them as they compute. Nothing is unpacked into memory first.
5. `run` gives a chat in the terminal and at `http://localhost:8000`. `serve` gives the same server for programs.

The design is in [docs/architecture.md](https://github.com/surya-koritala/Glyd/blob/main/docs/architecture.md).

## How to use it

[Install](install.md), then [first run](first-run.md): three commands. [Choosing a level](levels.md) when the model does not
fit. [Serving](serve.md) and [connecting tools](tools.md) to use it from other programs.

## What runs today

- NVIDIA Ampere or newer (RTX 30 series and up, A10, A100, L4, L40S, H100), Linux x86_64, driver 580 or newer.
- The families are on the [Models](models.md) page, generated from the code.
- A 180B mixture of experts on one 12-16 GB GPU: [Condor](condor.md).

Not yet: Windows, macOS, aarch64 and CPU-only engines. AMD is a development build.

## Pages

1. [Install](install.md)
2. [First run](first-run.md)
3. [Choosing a level](levels.md)
4. [Serving and the APIs](serve.md)
5. [Connecting tools](tools.md)
6. [Several GPUs and Starlings](multi-gpu.md)
7. [Condor](condor.md)
8. [Troubleshooting](troubleshooting.md), by what `glyd-engine doctor` prints
9. [Models](models.md)
10. [The model catalog](catalog.md)

## The command is `glyd-engine`

The engine's program is `glyd-engine`. Its messages say `glyd`, because the front-door program that will hold these
commands under that name is not on main yet. The commands on these pages work with `glyd-engine`. Where `glyd` (the file
compressor) is installed beside `glyd-engine` and the Python tool is not, `glyd run`, `serve`, `doctor`, `login`, `fetch`,
`preflight`, `prepare`, `supports`, `costs` and `rates` go to the engine. `fit` is `glyd-engine fit` only.

## Benchmarks

This guide has no benchmark tables. Every measurement, with its machine, build and log, is in
[benchmarks/](https://github.com/surya-koritala/Glyd/tree/main/benchmarks).

## Versions

This guide at `/guide/` follows main. Each release tag that has the guide also gets its own copy at `/guide/vX.Y.Z/`, built from that tag's text.
