<h1 align="center">Glyd</h1>
<p align="center">Open-source LLM inference in Rust. Name a model; Glyd fits it to your GPU.</p>

<p align="center">
<a href="https://github.com/surya-koritala/Glyd/actions"><img alt="CI" src="https://github.com/surya-koritala/Glyd/actions/workflows/ci.yml/badge.svg"></a>
<a href="https://github.com/surya-koritala/Glyd/releases"><img alt="Release" src="https://img.shields.io/github/v/release/surya-koritala/Glyd?label=release"></a>
<a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-blue.svg"></a>
</p>

- **Lossless compression in GPU memory**: weights and KV cache in ~10.8 and 12 bits, decoded in the kernels, bit for bit
  (about a third less memory than bf16; `glyd verify` proves a pack equals the original).
- **Lossy levels when that is not enough**: 6.5, 5.5, 4.5 and ~2-3 bits, chosen for the machine.
- **MoE larger than the GPU**: busy experts in VRAM, the rest in RAM or on NVMe, misses computed on the CPU.
- **Many GPUs and machines**: layer split across GPUs; one model across machines ([starlings/](starlings/), beta).
- **One binary, driver only** (the goal; see Status): no CUDA toolkit, no Python. OpenAI and Anthropic APIs, a chat page.

## Status

The Rust engine runs Qwen3 dense (lossless), Qwen3.5/3.8 and Nemotron-H (lossy levels), Gemma 4 12B/31B and Condor's
180B MoE on NVIDIA Ampere or newer, Linux x86_64, driver 580+. The current release (v0.32.1) still sends other models, and
`glyd run` by default, through a frozen vLLM path that needs Python. v0.33.0 is the first release built only from this
repository, with an engine install that needs no Python; Windows, aarch64 and CUDA 12 drivers follow in v0.34.0. Plan and
dates: [ROADMAP.md](ROADMAP.md).

## Run the engine

```bash
cargo build --release -p glyd-engine                    # needs the CUDA toolkit to build kernels: CONTRIBUTING.md
target/release/glyd-engine fit Qwen/Qwen3-32B             # the plan and predicted speed for this GPU, before any download
target/release/glyd-engine run Qwen/Qwen3-8B             # download, pack, chat in the terminal and at http://localhost:8000
target/release/glyd-engine run Qwen/Qwen3.8-27B:swift    # a lossy level: :kestrel 6.5 bits, :swift 5.5
target/release/glyd-engine serve MODEL                   # OpenAI and Anthropic APIs
target/release/glyd-engine doctor                        # what this machine can run, and why not
```

A 180B MoE on one 12-16 GB GPU, on Linux or Windows: [condor/](condor/).

## Results

| Measured | bf16 | Glyd | Path |
| :--- | ---: | ---: | :--- |
| Qwen3-32B: 48 GB GPUs needed | 2 | **1** | vLLM plugin |
| Qwen3.8-27B layer matrices, bit for bit | 49.52 GB | **33.28 GB** | pack |
| Qwen3.5-9B on an RTX 4090: max context | out of memory | **189,440 tokens** | vLLM plugin |
| Qwen3.8-Flash-Next 180B MoE on an RTX 4080 SUPER 16 GB | does not fit | **91 tokens/s** | engine (2.9 bits, MTP on) |

Logs: [benchmarks/](benchmarks/), [condor/results/](condor/results/). All results: [DETAILS.md](DETAILS.md).

## Repository

| Path | What |
| :--- | :--- |
| [`glyd-engine/`](glyd-engine/) | Inference engine (Rust) and its CUDA/HIP kernels |
| [`gpu/`](gpu/) | GPU library behind a C API: compressed weights and KV cache |
| [`glyd-gpu/`](glyd-gpu/) | Packer and verifier (CPU) |
| [`starlings/`](starlings/) | One model across GPUs and machines |
| [`condor/`](condor/) | Big MoE on one consumer GPU: setup, chat app, benchmarks |
| [`src/`](src/), [`glyd-store/`](glyd-store/) | Lossless compression codec and store |
| [`bindings/`](bindings/), [`glyd-gpu-py/`](glyd-gpu-py/) | Python/Go bindings; Python GPU package and vLLM plugin (frozen) |

## Develop

```bash
cargo build --release --workspace
cargo test --release --workspace
```

Design: [docs/architecture.md](docs/architecture.md). Work list: [ROADMAP.md](ROADMAP.md). Building with CUDA or ROCm,
GPU tests, adding a model: [CONTRIBUTING.md](CONTRIBUTING.md). Changes: [CHANGELOG.md](CHANGELOG.md).

## License

Apache-2.0, all of it ([LICENSE](LICENSE), [NOTICE](NOTICE)). Code in [third_party/](third_party/) keeps its own license.
