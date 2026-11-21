# glyd-engine

Glyd's inference engine: loads Glyd packs and released checkpoints, runs them on NVIDIA GPUs (AMD RDNA in development),
serves OpenAI- and Anthropic-compatible APIs and a chat page. Rust; CUDA kernels in `kernels/`, compiled by `build.rs`
into a fatbin the binary embeds. At run time it needs only the GPU driver and `libglyd_gpu` (`../gpu/`).

## Commands

```
glyd-engine run MODEL [--prompt TEXT]     download, pack for this GPU, chat (terminal + http://127.0.0.1:8000)
glyd-engine serve MODEL --port N          API server: /v1/chat/completions, /v1/completions, /v1/messages, /v1/models
glyd-engine preflight MODEL               exit 0 if it will run here, 3 with the reason if not (nothing downloaded)
glyd-engine costs MODEL[:level] [--predict]   bytes per layer, per token, per KV position; predicted tokens/s
glyd-engine rates                         measure this machine's memory tiers
glyd-engine fit MODEL[:level] [--machine NAME|FILE]   the plan before any download: level, placement, context, predicted tokens/s
glyd-engine doctor | login | fetch | prepare | verify | supports | rangeworker
```

`MODEL` is a Hub ID (`ORG/NAME[:level]`), a Hugging Face cache directory or a packed directory. Common flags:
`--context N`, `--users N`, `--layout auto|mma|mma12`, `--split CUTS --devices GPUS`, `--expert-cache GIB`.
Every flag and variable: [REFERENCE.md](REFERENCE.md).

## Build

```bash
cargo build --release -p glyd-engine                       # release features: --no-default-features
NVCC=/usr/local/cuda/bin/nvcc GLYD_ENGINE_ARCHS="89" cargo build --release -p glyd-engine
GLYD_HIP=1 cargo build --release -p glyd-engine           # AMD RDNA 3/4
cargo test --release -p glyd-engine                        # host-side tests, no GPU needed
```

The default `dev` feature adds the development commands (`check`, `kcheck`, `kvtest`, `stress`, `bench`, `pack`, ...)
and `GLYD_ENGINE_*` tuning variables.

## Layout

| Path | What |
| :--- | :--- |
| `src/ffi.rs`, `dev.rs`, `kern.rs` | driver bindings (CUDA, HIP), device, kernel launch |
| `src/server.rs`, `sched.rs`, `chat.rs` | HTTP APIs, scheduler (Qwen3 path), chat templates |
| `src/model.rs`, `model_nh.rs`, `model_glm.rs`, `model_qwen4x.rs` | model pipelines (Qwen3.5's and Gemma 4's run on `src/dynamic`; `kern35.rs` and `gemma4.rs` hold their kernel tables, configs and packed weights) |
| `src/dynamic/` | shared executor for the large MoE families (Kimi, DeepSeek-V4, GLM, Muse, Nemotron), split and TP |
| `src/offload.rs`, `disk.rs`, `cpu_*.rs` | expert cache across VRAM, RAM, NVMe; CPU expert kernels |
| `src/costs.rs`, `machine.rs`, `fit.rs` | cost profile, machine probe, fit |
| `src/kv.rs`, `kvcal.rs`, `weights.rs`, `pack.rs`, `f8.rs` | lossless KV cache, packed weights, FP8 |
| `src/*_ref.rs` | CPU references for tests |
| `kernels/` | CUDA/HIP kernels |
| `docs/design/` | design notes |

License: Apache-2.0 (`LICENSE`).
