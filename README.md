# Glyd

Lossless compression for LLM weights and KV cache, and the inference engine that runs on it. Rust, Apache-2.0.

You name a model and Glyd fits it to your GPU. Weights and KV cache are packed losslessly (about 10.8 and 12 bits
instead of bf16's 16) and decoded inside the kernels, bit for bit. When that is not enough: lossy levels (about 6.5,
5.5 and 4.5 bits), experts in RAM or on NVMe, then more GPUs.

```bash
cargo build --release -p glyd-engine            # building the kernels needs the CUDA toolkit: CONTRIBUTING.md
target/release/glyd-engine fit Qwen/Qwen3-8B     # the plan for this GPU, before any download
target/release/glyd-engine run Qwen/Qwen3-8B     # download, pack, chat in the terminal and at http://localhost:8000
target/release/glyd-engine serve Qwen/Qwen3-8B   # OpenAI and Anthropic APIs, and the chat page
```

## Measured

| Run | Machine | bf16 | Glyd |
| :--- | :--- | ---: | ---: |
| Qwen3-8B on the engine, lossless | RTX 4080 SUPER 16 GB | 15.3 GiB, over the 14.7 GiB free | 10.3 GiB, 59.7 tokens/s ([log](benchmarks/gpu/rtx4080s-qwen3-8b-engine-2026-11-20)) |
| Qwen3.8-27B layer matrices, read back bit for bit | A10 | 49.52 GB | 33.28 GB ([log](https://github.com/surya-koritala/Glyd/blob/7d9e9be6f33395f39421304b33c9f8481afa5962/benchmarks/gpu/open-models-a10-2026-09-27/sizes.txt)) |
| Qwen3.8-Flash-Next 180B MoE, decode | RTX 4080 SUPER, 64 GB RAM | does not fit | 91 tokens/s ([log](condor/results/v0.2.4/pc)) |

More, each with its machine and log: [DETAILS.md](DETAILS.md), [benchmarks/](benchmarks/).

## Runs today

- NVIDIA Ampere or newer, Linux x86_64, driver 580 or newer.
- Lossless: Qwen3 dense, Gemma 4 12B and 31B, Nemotron-H, Kimi-K3, GLM-5.3-Flash, DeepSeek-V4, Muse Spark.
  Lossy only: Qwen3.5, 3.6 and 3.8 dense. `glyd-engine supports MODEL` answers for one model.
- A 180B mixture of experts on one 12-16 GB GPU: [condor/](condor/). One model across machines: [starlings/](starlings/) (beta).

## Not yet

- `glyd run` from the current release (v0.32.1) still goes through vLLM and Python. The engine becomes its default on a
  GPU once it measures at least as fast there ([#214](https://github.com/surya-koritala/Glyd/issues/214)); v0.33.0, the
  first release built only from this repository, is next ([#129](https://github.com/surya-koritala/Glyd/issues/129)).
- No Windows, macOS, aarch64 or CPU-only engine. AMD is a development build (`GLYD_HIP=1`).
- Batching on Qwen3 only (up to 8 requests); the other families serve one request at a time.

## Repository

`glyd-engine/` the engine and its CUDA/HIP kernels · `gpu/` the GPU library (C API) · `glyd-gpu/` the packer and
verifier · `src/`, `glyd-store/` the file codec and store · `condor/`, `starlings/` · `bindings/`, `glyd-gpu-py/`
Python and Go bindings and the frozen vLLM plugin.

Design: [docs/architecture.md](docs/architecture.md). Work list: [ROADMAP.md](ROADMAP.md). Contributing:
[CONTRIBUTING.md](CONTRIBUTING.md). Changes: [CHANGELOG.md](CHANGELOG.md).

## License

Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)); code in [third_party/](third_party/) keeps its own license.
Releases v0.29.0 to v0.32.1 were published under BUSL-1.1.
