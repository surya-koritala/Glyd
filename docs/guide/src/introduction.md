# About this guide

## What Glyd does

Glyd compresses an LLM and runs it. It packs the weights losslessly and decodes them inside the GPU kernels (the KV cache
too, in 12-bit lossless pages, where `GLYD_KV=lossless` asks for them and the plan finds them no bigger than bf16 pages), so a model that does not fit your GPU in bf16 often fits and answers exactly as before. When lossless is not enough there are
lossy levels, experts kept in RAM, and more GPUs. It is one program written in Rust, from the compressed file to the
tokens, with an OpenAI and Anthropic compatible server and a chat page.

## Features

- Lossless weights, bit for bit, decoded in the kernels; the KV cache in bf16 pages, or in 12-bit lossless pages where `GLYD_KV=lossless` asks for them (Gemma 4 with `--no-draft`; the plain dense kind under `serve`) and the plan finds that they reserve no more than bf16: `fit` and `run` say which.
- Four lossy levels (kestrel, swift, wren, hummingbird) for when a model still does not fit.
- Mixture-of-experts models with their experts in RAM or on the SSD.
- One model across the GPUs of a machine, and across machines with Starlings (beta).
- A server for the OpenAI and Anthropic APIs, with tools and constrained JSON, and a chat page.
- `fit`: the plan for your GPU before anything is downloaded.

## How it works

1. You name a model: `gemma4:12b`, `google/gemma-4-12B-it`, or a path.
2. `fit` reads the model's config and plans a level, where the weights go and how much context fits.
3. On the first run Glyd downloads the model and packs it for your GPU (and, where its plan has the 12-bit KV pages, builds their tables once).
4. The kernels read the packed weights and decode them as they compute. Nothing is unpacked into memory first.
5. `run` gives a chat in the terminal and at `http://localhost:8000`. `serve` gives the same server for programs.

The design is in [docs/architecture.md](https://github.com/surya-koritala/Glyd/blob/main/docs/architecture.md).

## How to use it

[Install](install.md), then [first run](first-run.md): a few commands. [Choosing a level](levels.md) when the model does not
fit. [Serving](serve.md) and [connecting tools](tools.md) to use it from other programs.

## What runs today

- NVIDIA Ampere or newer (RTX 30 series and up, A10, A100, L4, L40S, H100), Linux x86_64, driver 580 or newer.
- The families are on the [Models](models.md) page, generated from the code.
- A 180B mixture of experts on one 12-16 GB GPU: [Condor](condor.md).

Not yet: macOS and CPU-only engines. Windows is a preview and aarch64 is built but not yet measured. AMD is a development build.

## Pages

1. [Install](install.md)
2. [First run](first-run.md)
3. [Choosing a level](levels.md)
4. [Serving and the APIs](serve.md)
5. [Connecting tools](tools.md)
6. [Several GPUs and Starlings](multi-gpu.md)
7. [Condor](condor.md)
8. [Troubleshooting](troubleshooting.md), by what `glyd doctor` prints
9. [API reference](api.md) and [the background service](service.md), generated from `docs/serve-api.md` and `docs/service.md`
10. [Models](models.md), generated from `glyd-engine/src/support.rs`
11. [The model catalog](catalog.md)
12. [Checking a machine](gate.md): the M1 gate, step by step

## The command is `glyd`

Since v0.33.0 `glyd` is one program with the engine linked in: `run`, `serve`, `pull`, `ls`, `rm`, `service`, `ps`, `stop`, `fit`,
`doctor`, `login`, `connect`, `supports`, `preflight` and the rest. `glyd-engine` is a link to it that keeps the engine's older
command line. File compression is the same program: `glyd FILE -o FILE.glyd`.

## Benchmarks

This guide has no benchmark tables. Every measurement, with its machine, build and log, is in
[benchmarks/](https://github.com/surya-koritala/Glyd/tree/main/benchmarks).

## Versions

This guide at `/guide/` follows main. Each release tag that has the guide also gets its own copy at `/guide/vX.Y.Z/`, built from that tag's text.

## Supporting Glyd

Glyd is open source under Apache-2.0. [GitHub Sponsors](https://github.com/sponsors/surya-koritala) pays for the GPUs it is measured and
tested on.

<iframe src="https://github.com/sponsors/surya-koritala/button" title="Sponsor surya-koritala" height="32" width="114" style="border: 0; border-radius: 6px;"></iframe>
