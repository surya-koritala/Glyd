<h1 align="center">Glyd</h1>
<h3 align="center">Open-source inference for open models: the same model in a third less memory, bit for bit, on one GPU or many.</h3>

<p align="center">
<a href="https://github.com/surya-koritala/Glyd/actions"><img alt="CI" src="https://github.com/surya-koritala/Glyd/actions/workflows/ci.yml/badge.svg"></a>
<a href="https://github.com/surya-koritala/Glyd/releases"><img alt="Release" src="https://img.shields.io/github/v/release/surya-koritala/Glyd?label=release"></a>
<a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-blue.svg"></a>
</p>

Glyd runs open LLMs on the GPUs you have. It keeps a model's weights and KV cache in fewer bits in GPU memory and rebuilds
the exact values as the model runs, so a model fits a smaller GPU or serves more requests on the same one. When even that
does not fit, it steps down to lossy levels, keeps a mixture of experts' busy experts on the GPU and the rest in RAM, or
splits the model across GPUs and machines.

All of it is in this repository, in Rust, under the Apache License 2.0: the GPU kernels, the engine, the packer, the codec,
and the tools around them. Nothing ships compiled-only.

## Run it

```bash
curl -LsSf https://getglyd.com/install.sh | sh
glyd run Qwen/Qwen3.5-9B                       # settings for your GPU, then a chat in the terminal and at http://localhost:8000
glyd run Qwen/Qwen3.8-27B:swift                # fewer bits, not lossless: :kestrel 6.5 bits per weight, :swift 5.5
glyd serve MODEL                               # OpenAI- and Anthropic-compatible APIs
```

Linux (x86_64 or aarch64) with an NVIDIA GPU (Ampere, Ada, Hopper or Blackwell), driver 580 or newer. On a Mac or without
a GPU, the same command installs the compression program. AMD and Apple Silicon are in progress ([the plan](docs/PLAN.md)).

- **A 180B MoE on a 16 GB card:** [Condor](condor/) runs Qwen3.8-Flash-Next on one 12-16 GB GPU with 32-64 GB of RAM
  (`cd condor && ./setup.sh`): 79-131 tokens/s on an RTX 4080 SUPER, Linux and Windows.
- **One model across two GPUs or two machines:** [Starlings](starlings/) (beta): `starlings up` on one machine,
  `starlings join` on the other.
- **With vLLM:** `pip install "glyd[vllm]"`, then `vllm serve Qwen/Qwen3.5-9B --quantization glyd` (the Python path; frozen,
  see [the plan](docs/PLAN.md#4-rust-only-what-happens-to-the-python)).

## Results

| Measured | bf16 | Glyd |
| :--- | ---: | ---: |
| Qwen3-32B: 48 GB GPUs it takes | 2 | **1** |
| Qwen2.5-72B: 48 GB GPUs it takes | 4 | **3** |
| Qwen3.5-9B on an RTX 4090 (24 GB): context | out of memory at start | **189,440 tokens** |
| Qwen3.8-27B: its layers' matrices, read back bit for bit | 49.52 GB | **33.28 GB** (−32.8%) |
| Qwen3.8-Flash-Next (180B MoE) on an RTX 4080 SUPER 16 GB, precise mode | does not fit | **91 tokens/s** |

Logs for every row are in [benchmarks/](benchmarks/) and [condor/results/](condor/results/); the full page with every
result is [DETAILS.md](DETAILS.md).

**What lossless means here:** every weight, and every key and value in the KV cache, reads back as the exact bf16 value.
Outputs can still differ from bf16's in the last bits, because Glyd's kernels add the same products in a different order,
as any two GPU kernels do. The lossy levels (`:kestrel`, `:swift`, `:wren`, `:hummingbird`) say what they give up.

## What is in this repository

| Folder | What |
| :--- | :--- |
| [`glyd-engine/`](glyd-engine/) | The inference engine (Rust; CUDA kernels in `kernels/`, AMD through HIP in development): OpenAI and Anthropic APIs, the chat page, expert offload, multi-GPU |
| [`gpu/`](gpu/) | The GPU library behind a C API: weights and KV cache compressed in GPU memory |
| [`glyd-gpu/`](glyd-gpu/) | The packer and checker (`glyd pack`, `glyd verify`), on the CPU |
| [`condor/`](condor/) | One big MoE on one consumer GPU: setup scripts, chat app, benchmarks |
| [`starlings/`](starlings/) | One model across GPUs and machines: enrollment, TLS, layer-range pipeline |
| [`src/`](src/), [`glyd-store/`](glyd-store/) | The lossless compression codec and CLI (a drop-in alternative to zstd, LZ4 and Snappy), the store |
| [`bindings/`](bindings/), [`glyd-gpu-py/`](glyd-gpu-py/) | Python and Go bindings; the Python GPU package and vLLM plugin (frozen) |

## Where it is going

One Rust program from a gaming card to a datacenter, on NVIDIA, AMD and Apple: the levels chosen for the hardware, one
placement planner across GPU memory, RAM, other GPUs and other machines, and serving from one user to thousands.
What exists today, what is missing, how it compares with Ollama, llama.cpp, vLLM, SGLang and exo, and the order of the
work: **[docs/PLAN.md](docs/PLAN.md)**.

## Build from source

```bash
cargo build --release --workspace              # the codec, the store, the packer and the engine (without nvcc: no CUDA kernels)
NVCC=/usr/local/cuda/bin/nvcc cargo build --release -p glyd-engine     # the engine with its kernels (CUDA 12.8 or newer)
bash gpu/build_lib.sh target/gpu               # the GPU library
GLYD_HIP=1 cargo build --release -p glyd-engine                        # AMD RDNA 3/4 through ROCm's hipcc (in development)
(cd starlings && cargo build --release)        # Starlings (its own workspace and pinned toolchain)
```

## Docs

- [DETAILS.md](DETAILS.md): everything on one page, with the results and their logs.
- [docs/PLAN.md](docs/PLAN.md): the product plan. [CHANGELOG.md](CHANGELOG.md), [ROADMAP.md](ROADMAP.md),
  [SECURITY.md](SECURITY.md), [CONTRIBUTING.md](CONTRIBUTING.md).

## License

All of Glyd is under the [Apache License 2.0](LICENSE): the engine, the GPU library, the packer, the codec, the store, the
bindings, Condor and Starlings. Releases v0.29.0 to v0.32.1 shipped under the Business Source License 1.1, and earlier
releases under the licenses in their tags ([NOTICE](NOTICE)). Code in [third_party/](third_party/) keeps its own license.
