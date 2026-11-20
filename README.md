# Glyd

Lossless compression for LLM weights and KV cache, and the inference engine that runs on it. Rust, Apache-2.0.

[![CI](https://github.com/surya-koritala/Glyd/actions/workflows/ci.yml/badge.svg)](https://github.com/surya-koritala/Glyd/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/surya-koritala/Glyd?label=release)](https://github.com/surya-koritala/Glyd/releases)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

[Docs](https://surya-koritala.github.io/Glyd/) · [Models](#models-and-hardware) · [Benchmarks](benchmarks/) ·
[Roadmap](ROADMAP.md) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md)

You name a model and Glyd fits it to your GPU. Weights and KV cache are packed losslessly (about 10.8 and 12 bits instead
of bf16's 16) and decoded inside the kernels, so the model computes on its original weights, bit for bit. When that is not
enough: lossy levels (about 6.5, 5.5 and 4.5 bits), experts in RAM or on NVMe, then more GPUs.

## Install

Linux x86_64, an NVIDIA GPU of the Ampere generation or newer, driver 580 or newer. Building needs the CUDA toolkit;
running needs only the driver.

```bash
git clone https://github.com/surya-koritala/Glyd && cd Glyd
cargo build --release -p glyd-engine            # builds the CUDA kernels too: CONTRIBUTING.md
```

The one-line installer (`curl -LsSf https://getglyd.com/install.sh | sh`) installs the latest release, v0.32.1, which
still runs models through vLLM and Python. v0.33.0, the first release built only from this repository, installs the
engine alone with no Python ([#129](https://github.com/surya-koritala/Glyd/issues/129)).

## Quick start

```bash
target/release/glyd-engine fit Qwen/Qwen3-8B     # the plan for this GPU, before any download
target/release/glyd-engine serve Qwen/Qwen3-8B   # download, pack once, then the API and a chat page on localhost:8000
```

On an RTX 4080 SUPER (16 GB):

```
Qwen3-8B: 10.3 GiB on the GPU with Glyd, instead of 15.3 GiB
Packed in 12s (12.4 GB).
Ready in 45s: Qwen3-8B on the NVIDIA GeForce RTX 4080 SUPER, 10.3 GiB of weights, 8192 tokens of context, 4 users at once.
Qwen/Qwen3-8B is served at http://127.0.0.1:8000: the chat page at /, the API at /v1 (OpenAI-compatible).
```

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen3-8B", "messages": [{"role": "user", "content": "In one sentence: what does lossless compression of model weights buy you on a 16 GB GPU? /no_think"}]}'
```

```
(the reply's content)
"Lossless compression of model weights on a 16 GB GPU allows you to fit larger models or run multiple models
simultaneously without sacrificing precision or performance."
```

`glyd-engine run MODEL` chats in the terminal instead; `MODEL:kestrel` or `:swift` picks a lossy level; `glyd-engine
verify MODEL` decodes every pack on the CPU and checks each tensor against the original's sha256. The OpenAI API takes
`response_format` with a JSON schema and forced tool calls; the Anthropic Messages API is served too.

## Models and hardware

| Family | Levels on the engine | Across GPUs |
| :--- | :--- | :--- |
| Qwen3 dense | penguin (lossless) | no |
| Qwen3.5, 3.6, 3.8 dense | kestrel, swift | no |
| Gemma 4 12B, 31B | penguin, kestrel, swift, wren | no |
| Nemotron-H (Nemotron 3 Super) | penguin, kestrel, swift, wren, hummingbird | penguin |
| Kimi-K3, GLM-5.3-Flash, DeepSeek-V4, Muse Spark | penguin | penguin |
| Qwen3.8-Flash-Next 180B MoE | about 2.9 bits, experts in RAM | one 12-16 GB GPU: [condor/](condor/) |

`glyd-engine supports MODEL` answers for one model; GGUF files load where their format is supported. Hardware: NVIDIA
Ampere to Blackwell on Linux x86_64. AMD RDNA 3/4 is a development build (`GLYD_HIP=1`). One model across machines is
[starlings/](starlings/), a beta.

## Measured

Each row is one run on the machine named, with its log. Speeds are against the hardware's own limit: weight bytes per
token over memory bandwidth.

| What | Machine | bf16 | Glyd | Log |
| :--- | :--- | ---: | ---: | :--- |
| Qwen3-8B, lossless | RTX 4080 SUPER 16 GB | 15.3 GiB, does not fit | 10.3 GiB; 59.9 tokens/s for one user (90% of the card's limit), 57.1 each with 8 | [bar](benchmarks/gpu/rtx4080s-bar-2026-11-20) |
| Qwen3.8-27B layer matrices, read back bit for bit | A10 | 49.52 GB | 33.28 GB | [sizes](https://github.com/surya-koritala/Glyd/blob/7d9e9be6f33395f39421304b33c9f8481afa5962/benchmarks/gpu/open-models-a10-2026-09-27/sizes.txt) |
| Qwen3.8-Flash-Next 180B MoE, decode | RTX 4080 SUPER, 64 GB RAM | does not fit | 91 tokens/s | [condor](condor/results/v0.2.4/pc) |
| Kimi-K3, every layer across 8 GPUs | 8x A100 80 GB | | 87 ms a token | [run](research/kimi-k3-a100-2026-10-22b) |

The GPU CI runs the kernel checks against CPU references and a model end to end on every merge to main.

## How it works

- **Formats.** A lossless pack per weight matrix and a 12-bit lossless KV cache, decoded in the GPU kernels, plus lossy
  levels. `glyd-engine fit` plans the level, context and placement before any download.
- **MoE.** The engine knows which experts the router picks: the busy ones stay in VRAM, the rest in RAM or on NVMe, and
  the CPU computes the misses while the GPU works.
- **Many GPUs.** Layers split across GPUs, or every layer across them with direct GPU-to-GPU gathers; across machines with
  Starlings.
- **Serving.** Batched prompts and decode (Qwen3 path), OpenAI and Anthropic APIs, constrained decoding, a chat page.

Design: [docs/architecture.md](docs/architecture.md). Engine settings: [glyd-engine/REFERENCE.md](glyd-engine/REFERENCE.md).

## Not yet

- `glyd run` uses the engine by default only on GPUs where it has been measured at least as fast as the vLLM path:
  today the RTX 4080 SUPER ([docs/adding-a-gpu.md](docs/adding-a-gpu.md)). Elsewhere `glyd-engine run`, or
  `GLYD_ENGINE=1`, uses it.
- No Windows, macOS, aarch64 or CPU-only engine release yet. Batching is on the Qwen3 path only (up to 8 requests); the
  other families serve one request at a time.
- Lossy levels change outputs; their measured quality is in [DETAILS.md](DETAILS.md).

## Contributing

Issues and pull requests are welcome; start with [CONTRIBUTING.md](CONTRIBUTING.md) and the
[good first issues](https://github.com/surya-koritala/Glyd/labels/good%20first%20issue). Commits carry a DCO sign-off.
Security reports: [SECURITY.md](SECURITY.md). How decisions are made: [GOVERNANCE.md](GOVERNANCE.md).

## License

Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)); code in [third_party/](third_party/) keeps its own license. Releases
v0.29.0 to v0.32.1 were published under BUSL-1.1.
