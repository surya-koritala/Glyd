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
- **One binary, driver only**: no CUDA toolkit, no Python. OpenAI and Anthropic APIs, a chat page.

## Run

```bash
curl -LsSf https://getglyd.com/install.sh | sh
glyd run Qwen/Qwen3.5-9B               # plan for this GPU, download, chat in the terminal and at http://localhost:8000
glyd run Qwen/Qwen3.8-27B:swift        # pin a level: :kestrel 6.5 bits, :swift 5.5
glyd serve MODEL                       # API server
```

Today: Linux x86_64/aarch64, NVIDIA Ampere or newer, driver 580+. Windows, AMD, Apple and older drivers:
[ROADMAP.md](ROADMAP.md). A 180B MoE on one 12-16 GB GPU: [condor/](condor/).

## Results

| Measured | bf16 | Glyd |
| :--- | ---: | ---: |
| Qwen3-32B: 48 GB GPUs needed | 2 | **1** |
| Qwen3.8-27B layer matrices, bit for bit | 49.52 GB | **33.28 GB** |
| Qwen3.5-9B on an RTX 4090: max context | out of memory | **189,440 tokens** |
| Qwen3.8-Flash-Next 180B MoE on an RTX 4080 SUPER 16 GB | does not fit | **91 tokens/s** |

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
