# Roadmap

Milestones in dependency order. Tracks within a milestone can run in parallel. Each milestone ends on a gate measured on
real hardware, with the result committed under `benchmarks/`. Design: [docs/architecture.md](docs/architecture.md).

## Where it stands (v0.32.1, 2026-10-17)

- NVIDIA Ampere-Blackwell on Linux x86_64: engine in production for Qwen3, Qwen3.5/3.8 (lossy), Gemma 4, Nemotron-H, GLM-5.3,
  Kimi-K3, DeepSeek-V4, Muse, Qwen3.8-Flash-Next. Other architectures run through the frozen vLLM plugin.
- Multi-user serving on the Qwen3 path only (1-8 users); other families serve one request at a time.
- MoE experts on GPU/RAM/NVMe with CPU compute for misses; layer split over GPUs; two-machine pipeline (Starlings beta).
- AMD RDNA 3/4 builds, unmeasured. No Apple, no CPU-only, no Windows release build.

## M1 Runtime: one path for every model

- [x] R0 One levels module (`src/levels.rs`).
- [ ] R1 Backend trait (`docs/architecture.md#backends`) over `dev.rs`/`ffi.rs`: memory, streams, events, graphs, module and
      launch, peer access; the four direct-driver call sites first; CUDA behind it with no `glyd bench` regression.
- [ ] R2 Layer-kind library: attention (MHA, GQA, MLA, sliding, linear/delta), Mamba, dense MLP, MoE, norms, rotary
      variants; a CPU f32 reference for each.
- [ ] R3 Model descriptions: config.json → layer graph + weight map. Port Qwen3, Qwen3.5/3.8, Gemma 4, Nemotron-H, GLM,
      Kimi, DeepSeek-V4, Muse, Qwen3.8-Flash-Next onto it; delete the per-family pipelines and `qwen*serve` paths.
- [ ] R4 One scheduler for all models (`sched.rs` made model-agnostic): continuous batching, paged KV with content-addressed
      blocks (vLLM-compatible chained hashes, a radix index), chunked prefill, prefix cache, preemption; no user cap.
- [ ] R5 Sampling: constrained decoding through llguidance (JSON schema, regex, grammar); tool-call and reasoning parsers per
      family (vendored from the Shepherd Model Gateway; openai-harmony for gpt-oss).
- [ ] R6 Speculative decoding for any model: draft model, MTP heads, n-gram/prompt lookup.
- [ ] R7 Logits tests per model against the CPU reference; GPU CI on every merge to `main`.

Gate: every listed family serves 64 concurrent requests through the same scheduler on an RTX 4080 SUPER and an L4,
outputs matching the reference.

## M2 Product: install and it works

- [ ] P1 One binary `glyd` (engine linked in): `run`, `serve`, `pull`, `ls`, `rm`, `ps`, `stop`, `fit`, `doctor`, `bench`,
      `pack`, `verify`, `login`, `cluster`.
- [ ] P2 Planner v1 (`docs/architecture.md#adaptation`): backend, level, placement, context, users from costs × rates;
      `glyd fit` prints the plan and predicted tokens/s before any download. Started: `glyd-engine fit` (src/plan.rs); next:
      `run` and `serve` load by the plan, exact sizes from safetensors headers, measured rates instead of the bandwidth table.
- [ ] P3 Background service: user-level systemd / launchd / Windows service; load on request, unload when idle, several
      models; re-probe before each load, re-measure after warm-up, shrink KV and expert caches when other programs take VRAM.
- [ ] P4 Model names: Hugging Face IDs, local paths, GGUF files, and short names (`qwen3.5`, `qwen3.5:27b`) that pick the
      size and level for the machine.
- [ ] P5 APIs: OpenAI chat/completions/responses/embeddings/models, Anthropic messages (incl. prompt caching), Ollama API,
      MCP server for any model.
- [ ] P6 Drivers: a CUDA 12.8 fatbin (machine code for every arch; driver ≥ 525) and a CUDA 13 fatbin (driver ≥ 580), chosen
      with `cuDriverGetVersion`; `sm_120a` for consumer Blackwell; APIs newer than 12.0 behind a version check.
- [ ] P7 Platforms: Linux x86_64/aarch64, Windows x86_64, macOS arm64; a small core that downloads the backend it needs
      (CUDA 12, CUDA 13, ROCm, Metal; signed, content-addressed); install.sh, winget, Homebrew (CI checks every backend file
      ships), Docker per vendor; a backend that fails says why and the slowdown, never falls back silently.
- [ ] P8 Condor folded in: its model runs with `glyd run`; `condor/` keeps scripts that call `glyd`.
- [ ] P9 No Python on any install path; the vLLM plugin stays frozen at 0.32.x.

Gate: on a clean Ubuntu and a clean Windows machine with only the NVIDIA driver, install → first chat token in under
2 minutes (excluding the model download), for the 10 most-run models.

## M3 Models and formats

- [ ] F1 Families: Llama 3.x/4, Mistral/Mixtral, Gemma 2/3, Qwen 2/2.5/3-MoE, Phi-4, GPT-OSS, DeepSeek V3/R1, Granite,
      OLMo, SmolLM, GLM-4.x.
- [ ] F2 Vision for every VLM family; embeddings and rerankers.
- [ ] F3 GGUF reader and quant kernels ported from ggml; FP8, NVFP4, MXFP4, AWQ, GPTQ, compressed-tensors checkpoints as
      released.
- [ ] F4 LoRA adapters; many adapters batched together.
- [ ] F5 Pack pipeline: new releases packed, verified (`glyd verify`) and published to `glyd/*` with a quality card
      (KL, one eval) per level.
- [ ] F6 Long context past 40,960 positions (KV tables, chunked prefill at 128K+).

Gate: the top 50 text-generation models on Hugging Face run; a new release in a supported family runs within 24 hours.

## M4 Every vendor

- [ ] V0 Kernel portability: a warp-size and lane-mask abstraction (64-bit masks); a tensor-core fragment layer with four
      paths: `mma.sync`, RDNA3 WMMA, RDNA4 WMMA, CDNA MFMA.
- [ ] V1 AMD RDNA 3/4: measured, in CI, in release builds (ROCm on Linux; HIP on Windows, one GPU).
- [ ] V2 AMD CDNA 3/4 (MI300X, MI355X): wave64 kernels, MFMA products, hipBLASLt, RCCL.
- [ ] V3 Apple Silicon: a thin `-sys` crate over `mlx-c`; Glyd formats as MLX custom Metal kernels; the GPU's working-set
      limit (about 75% of RAM) as its memory budget.
- [ ] V4 CPU backend for whole models (AVX2, AVX-512/AMX, NEON/SVE).
- [ ] V5 NVIDIA Turing (T4, RTX 20): bf16 weights as fp16 at load (range-checked), register double-buffering in place of
      cp.async; Volta and older stay out (CUDA 13 dropped them).
- [ ] V6 A Vulkan or CPU fallback for other GPUs (evaluate CubeCL there only); vendor-native kernels stay the fast path.

Gate: the M3 model set passes on NVIDIA (T4, L4, RTX 4080 SUPER, A100, H100), AMD (RX 7900 XTX, MI300X), Apple (M2+) and
CPU (x86_64, arm64).

## M5 Many GPUs, many machines

- [ ] C1 Planner v2: one placement across VRAM, peer GPUs, host RAM, NVMe and other machines; Condor's expert policy for
      every MoE.
- [ ] C2 Tensor and expert parallelism in one node: NCCL/RCCL through cudarc, a custom all-reduce for small decode messages,
      pinned-host staging where GeForce peer access is off; overlapped pipeline for layer splits.
- [ ] C3 `glyd cluster`: Starlings' enrollment and transport for N machines and every model; failure of a node recovers
      without a full restart.
- [ ] C4 A `KvTransport` trait: TLS over TCP by default, NIXL (UCX, EFA, GDS) for RDMA; expert parallel via NCCL send/recv,
      then NCCL EP.

Gate: two consumer machines run a model neither holds alone at the speed `glyd fit` predicted (±10%); a 600B+ MoE runs
across two 8-GPU nodes.

## M6 Datacenter serving

- [ ] D1 Prefill/decode disaggregation with compressed KV transfer; KV offload to RAM/NVMe; a shared KV store.
- [ ] D2 KV-aware routing by integration, not a router of our own: vLLM-named metrics and KV events so llm-d, the Gateway API
      inference extension and Dynamo schedule Glyd.
- [ ] D3 Operations: Prometheus `/metrics`, OpenTelemetry traces, health/readiness, drain; Helm chart, LeaderWorkerSet,
      Gateway API inference extension.
- [ ] D4 Wide expert parallelism across nodes; multi-node NVLink (NVL72).
- [ ] D5 Weight-update API for RL rollouts; batch-invariant deterministic mode.
- [ ] D6 Signed releases (Sigstore), SBOM.

Gate: goodput at or above vLLM and SGLang on the top 10 models on H100 and MI300X with less memory, published with logs.

## Always

- `glyd bench` (Rust) replaces the Python benchmarks; results in `benchmarks/`.
- Python is ported or removed as each milestone replaces it (`glyd-gpu-py/`, `gpu/*.py`, `glyd-engine/tools/*.py`,
  `condor/bench`, `scripts/*.py` → `cargo xtask`).
