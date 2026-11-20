<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/img/glyd-mark-white.svg">
    <img src="docs/img/glyd-mark.svg" alt="Glyd" width="48">
  </picture>
</p>

<h1 align="center">Glyd</h1>

<p align="center"><strong>Compress and run open LLMs on your GPUs, end to end.</strong></p>

<p align="center">
Glyd packs a model's weights and KV cache losslessly, fits the model to the GPUs you have, and runs and serves it on its
own Rust engine. It works from one consumer card to a multi-GPU server. Apache-2.0.
</p>

<p align="center">
  <a href="https://surya-koritala.github.io/Glyd/">Docs</a> ·
  <a href="ROADMAP.md">Roadmap</a> ·
  <a href="CONTRIBUTING.md">Contributing</a> ·
  <a href="SECURITY.md">Security</a>
</p>

## Features

- **Lossless compression.** Weights and KV cache take less GPU memory and are decoded bit for bit inside the GPU
  kernels. `glyd-engine verify` checks any packed model against the original.
- **Lossy levels.** When lossless is not enough: kestrel, swift and wren (about 6.5, 5.5 and 4.5 bits per weight), and
  hummingbird for mixture-of-experts layers.
- **Fit to your machine.** `glyd-engine fit` plans the level, context length and placement before anything downloads.
- **Many GPUs.** Split a model's layers across GPUs, or share every layer across them. Starlings (beta) extends this
  across machines.
- **Serving.** OpenAI- and Anthropic-compatible APIs, JSON-schema and tool-call constraints, and a chat page.
- **Model families.** Qwen, Gemma, GLM, Kimi, DeepSeek, Muse and Nemotron. `glyd-engine supports MODEL` answers for a
  specific model.

## How it works

1. **Plan.** Glyd reads the model's config and file headers, then your GPUs, RAM and disk, and picks a level, a context
   length and where each part of the model lives.
2. **Pack.** Each weight matrix is packed once for your GPU and cached. Lossless packs are verified equal to the
   original tensors.
3. **Run.** The engine's CUDA kernels decode packed weights and KV cache as they compute, so the full-size weights never
   sit in GPU memory. For mixture-of-experts models, busy experts stay on the GPU, the rest move between RAM and NVMe,
   and the CPU computes the misses.
4. **Serve.** Responses stream through the API or the chat page.

Design notes: [docs/architecture.md](docs/architecture.md).

## Quick start

Linux x86_64 with an NVIDIA GPU (Ampere or newer, driver 580 or newer). Building needs the CUDA toolkit; running needs
only the driver.

```bash
git clone https://github.com/surya-koritala/Glyd && cd Glyd
cargo build --release -p glyd-engine
target/release/glyd-engine fit google/gemma-4-12B-it:kestrel    # the plan for this machine
target/release/glyd-engine run google/gemma-4-12B-it:kestrel    # chat in the terminal and at http://localhost:8000
```

Or install a release, with no Python and no toolkit (about 100 MB):

```bash
curl -LsSf https://getglyd.com/install.sh | sh
```

On the engine alone, `glyd run MODEL` is the engine's. With the Python tool installed (`GLYD_WITH_VLLM=1`), `glyd run` goes to
the engine only on the GPU it has been measured on, the RTX 4080 SUPER ([#214](https://github.com/surya-koritala/Glyd/issues/214)),
and to the frozen vLLM path on the others, or for a model the engine does not run; `GLYD_ENGINE=1` forces the engine,
`GLYD_ENGINE=0` keeps vLLM.

## Status

Runs today on NVIDIA GPUs under Linux; AMD RDNA 3/4 is a development build. Windows, macOS, Apple Silicon and CPU-only
are on the [roadmap](ROADMAP.md). Large mixture-of-experts models on one consumer GPU: [condor/](condor/).

## Contributing

Start with [CONTRIBUTING.md](CONTRIBUTING.md) and the
[good first issues](https://github.com/surya-koritala/Glyd/labels/good%20first%20issue). Commits carry a DCO sign-off.

## License

Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). Releases v0.29.0 to v0.32.1 were published under BUSL-1.1.
