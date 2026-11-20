# Glyd

Glyd is an LLM inference engine written in Rust. You name a model and it works out how to fit it on your GPU: lossless
compression in GPU memory first, then lossy levels, then experts in RAM or on NVMe, then more GPUs. It is also the
codec and store the project started as (`glyd FILE -o OUT`), and the packer and GPU library the engine is built on.

This README says what runs today, what was measured and where the log is, and what does not work yet.
[ROADMAP.md](ROADMAP.md) is the work list.

## Status

- Main is Apache-2.0 and holds everything: the engine, the GPU library, the packer, the codec, Condor and Starlings.
  Releases v0.29.0 to v0.32.1 were under BUSL-1.1; their code is Apache-2.0 here.
- The latest release is v0.32.1: the Python tool with vLLM and PyTorch (about 8 GB), and no engine tarball.
  `scripts/install.sh` on main installs the engine alone (about 100 MB, no Python) and takes `GLYD_WITH_VLLM=1` to add
  vLLM, but it can fetch the engine only once v0.33.0, the first release built only from this repository, is out
  ([#129](https://github.com/surya-koritala/Glyd/issues/129)).
- `glyd run` hands a model to Glyd's engine only on GPUs listed in `GPUS` in
  [`glyd-engine/src/machine.rs`](glyd-engine/src/machine.rs), the GPUs where the engine has been measured at least as
  fast as vLLM. That list is empty, so `glyd run` uses the vLLM path (Python) today. `GLYD_ENGINE=1` forces the engine on
  any GPU it runs on; `glyd-engine run` always uses it.
- The engine runs on NVIDIA GPUs of compute capability 8.0 or newer (Ampere and later), Linux x86_64, driver 580 or
  newer. It refuses aarch64 and older drivers with a message.

## Models on the engine

What `glyd-engine run` takes, from [`glyd-engine/src/support.rs`](glyd-engine/src/support.rs) (`glyd-engine supports MODEL`
asks it for one model):

| Family | Levels | Across GPUs |
| :--- | :--- | :--- |
| Qwen3 dense | penguin | no |
| Qwen3.5, 3.6, 3.8 dense | kestrel, swift (wren in a development build) | no |
| Gemma 4 12B and 31B | penguin, kestrel, swift, wren (wren: a QAT checkpoint, or a development build) | no |
| Nemotron-H (Nemotron 3 Super) | penguin, kestrel, swift, wren, hummingbird (hummingbird: pre-packed only) | penguin |
| Kimi-K3, GLM-5.3-Flash, DeepSeek-V4, Muse Spark | penguin | penguin |

penguin is lossless (weights about 10.8 bits, KV cache 12, bf16 is 16). kestrel, swift and wren are lossy, at about 6.5,
5.5 and 4.5 bits per weight; hummingbird is about 2-3 bits on a mixture of experts' experts. Qwen3.5 has no lossless
level in the engine yet, so `glyd-engine run Qwen/Qwen3.5-9B` is refused and `Qwen/Qwen3.5-9B:kestrel` runs. Gemma 4 26B-A4B
and the E models are refused. Qwen3.8-Flash-Next is not taken by `glyd run` yet; [condor/](condor/) runs it.

## Run the engine

```bash
cargo build --release -p glyd-engine              # the kernels need nvcc to build: CONTRIBUTING.md
target/release/glyd-engine fit Qwen/Qwen3-32B          # the plan and predicted speed for this GPU, before any download
target/release/glyd-engine run Qwen/Qwen3-8B           # download, pack, then chat in the terminal and at http://localhost:8000
target/release/glyd-engine run Qwen/Qwen3.8-27B:swift  # a lossy level, ":kestrel" or ":swift"
target/release/glyd-engine serve Qwen/Qwen3-8B         # the OpenAI and Anthropic message APIs and the chat page
target/release/glyd-engine doctor                      # what this machine can run, and why not
```

`glyd-engine --help` lists the rest. `fit` plans without loading: `run` does not yet load by the plan, and `fit` can
plan a run that `run` refuses. Building needs the CUDA toolkit, and running the engine needs only the driver. `glyd run`
and `glyd serve` from the release go through the Python tool and need Python.

Condor, a 180B mixture of experts on one 12-16 GB GPU, on Linux or Windows: [condor/](condor/).

## Results

Each row is one run on the machine named, with the path it was measured on. "vLLM plugin" is the Python package under
vLLM, which the engine does not use.

| What | Machine | bf16 | Glyd | Path | Log |
| :--- | :--- | ---: | ---: | :--- | :--- |
| Qwen3.5-9B, context that starts | RTX 4090 24 GB, vLLM 0.30.0 | out of memory at 4,096 | 189,440 tokens | vLLM plugin, v0.29 | [log](benchmarks/gpu/rtx4090-5090-qwen35-9b-v029-2026-10-06) |
| Qwen3.8-27B, weights | H100 80GB HBM3, vLLM 0.30.0 | 50.22 GiB | 38.77 GiB | vLLM plugin, v0.28 | [log](benchmarks/gpu/h100-qwen38-v028-2026-10-03) |
| Qwen3.8-27B, layer matrices read back bit for bit | A10 23 GB, driver 580.126 | 49.52 GB | 33.28 GB | pack, no inference | [log](https://github.com/surya-koritala/Glyd/blob/7d9e9be6f33395f39421304b33c9f8481afa5962/benchmarks/gpu/open-models-a10-2026-09-27/sizes.txt) |
| Qwen3-32B, 48 GB GPUs needed | 4x RTX A6000 (Lambda), 2026-09-26 | 2 | 1 | vLLM plugin, v0.25 to v0.28 | [log](https://github.com/surya-koritala/Glyd/tree/7d9e9be6f33395f39421304b33c9f8481afa5962/benchmarks/gpu/lambda-gpu_4x_a6000-20260926-084757) |
| Qwen3.8-Flash-Next 180B MoE, decode, short prompt | RTX 4080 SUPER 16 GB, 64 GB RAM | does not fit | 91 tokens/s | Condor's engine, experts about 2.9 bits | [log](condor/results/v0.2.4/pc) |

None of these is `glyd-engine run`'s speed on a GPU against vLLM; that comparison decides `GPUS` and is not made yet
([#214](https://github.com/surya-koritala/Glyd/issues/214)). More numbers, with the machine and log for each, are in
[DETAILS.md](DETAILS.md) and [benchmarks/](benchmarks/).

## Known limits

- `glyd run` uses vLLM and Python by default (see Status). The release installer is about 8 GB.
- No Windows or macOS engine release, no aarch64, no CPU-only run. AMD RDNA 3/4 builds only as a development build
  (`GLYD_HIP=1`).
- Lossless levels cover Qwen3 dense, Gemma 4, Nemotron-H and the four large MoE families above; Qwen3.5 and 3.8 dense run
  lossy only.
- Batching is on the Qwen3 path only (up to 8 sequences); the other families serve one request at a time.
- Splitting one model across the GPUs of a machine works for the families marked in the table. Across machines is
  [starlings/](starlings/), a beta.
- The GPU tests in CI run on one self-hosted machine, built for sm_89, with a small model (Qwen3-0.6B). See CONTRIBUTING.md.

## Repository

| Path | What |
| :--- | :--- |
| [`glyd-engine/`](glyd-engine/) | Inference engine (Rust) and its CUDA/HIP kernels |
| [`gpu/`](gpu/) | GPU library behind a C API: compressed weights and KV cache |
| [`glyd-gpu/`](glyd-gpu/) | Packer and verifier (CPU) |
| [`starlings/`](starlings/) | One model across GPUs and machines |
| [`condor/`](condor/) | Big MoE on one consumer GPU: setup, chat app, benchmarks |
| [`src/`](src/), [`glyd-store/`](glyd-store/) | Lossless file codec and store |
| [`bindings/`](bindings/), [`glyd-gpu-py/`](glyd-gpu-py/) | Python and Go bindings; Python GPU package and vLLM plugin (frozen) |

## Develop

```bash
cargo build --release --workspace
cargo test --release --workspace
```

Design: [docs/architecture.md](docs/architecture.md). Work list: [ROADMAP.md](ROADMAP.md). Building with CUDA or ROCm, GPU
tests, adding a model: [CONTRIBUTING.md](CONTRIBUTING.md). Changes: [CHANGELOG.md](CHANGELOG.md).

## License

Apache-2.0, all of it ([LICENSE](LICENSE), [NOTICE](NOTICE)). Code in [third_party/](third_party/) keeps its own license.
