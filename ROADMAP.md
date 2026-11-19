# Roadmap

Milestones in dependency order; each ships as one release. Tracks within a milestone can run in parallel. Each milestone
ends on a gate measured on real hardware, with the result committed under `benchmarks/`. Design:
[docs/architecture.md](docs/architecture.md). Every item below is a GitHub issue with tasks, acceptance criteria, test
hardware and dependencies, under its milestone's epic; the epics are tracked in
[#118](https://github.com/surya-koritala/Glyd/issues/118).

| Release | Milestone | Epic |
| :--- | :--- | :--- |
| v0.33.0 | [M0 Open-source baseline](https://github.com/surya-koritala/Glyd/milestone/1) | [#119](https://github.com/surya-koritala/Glyd/issues/119) |
| v0.34.0 | [M1 Runtime: one path for every model](https://github.com/surya-koritala/Glyd/milestone/2) | [#120](https://github.com/surya-koritala/Glyd/issues/120) |
| v0.35.0 | [M2 Install and it works](https://github.com/surya-koritala/Glyd/milestone/3) | [#121](https://github.com/surya-koritala/Glyd/issues/121) |
| v0.36.0 | [M3 Models and formats](https://github.com/surya-koritala/Glyd/milestone/4) | [#122](https://github.com/surya-koritala/Glyd/issues/122) |
| v0.37.0 | [M4 Every vendor](https://github.com/surya-koritala/Glyd/milestone/5) | [#123](https://github.com/surya-koritala/Glyd/issues/123) |
| v0.38.0 | [M5 Many GPUs, many machines](https://github.com/surya-koritala/Glyd/milestone/6) | [#124](https://github.com/surya-koritala/Glyd/issues/124) |
| v1.0.0 | [M6 Datacenter serving](https://github.com/surya-koritala/Glyd/milestone/7) | [#125](https://github.com/surya-koritala/Glyd/issues/125) |

## Where it stands (v0.32.1, 2026-10-17)

- NVIDIA Ampere-Blackwell on Linux x86_64: engine in production for Qwen3, Qwen3.5/3.8 (lossy), Gemma 4, Nemotron-H, GLM-5.3,
  Kimi-K3, DeepSeek-V4, Muse, Qwen3.8-Flash-Next. Other architectures run through the frozen vLLM plugin.
- Multi-user serving on the Qwen3 path only (1-8 users); other families serve one request at a time.
- MoE experts on GPU/RAM/NVMe with CPU compute for misses; layer split over GPUs; two-machine pipeline (Starlings beta).
- AMD RDNA 3/4 builds, unmeasured. No Apple, no CPU-only, no Windows release build.

## M0 Open-source baseline · v0.33.0

[Milestone](https://github.com/surya-koritala/Glyd/milestone/1) · [Epic #119](https://github.com/surya-koritala/Glyd/issues/119)

- [ ] S1 Merge the monorepo; archive the predecessor repositories with pointers here.
- [ ] S2 Self-hosted GPU runner for `gpu.yml` (fork-guarded, labels per GPU).
- [ ] S3 Ephemeral cloud GPU runners for `gpu-cloud.yml` (OIDC, one instance per job, terminated on exit).
- [ ] S4 v0.33.0 built entirely from this repository: CLI tarballs, wheels, engine bundles.
- [ ] S5 macOS runner for the Metal backend.
- [ ] S6 Required checks on `main`; path-filtered workflows report when skipped.

Gate: a tag on `main` produces every release artifact from this repository; GPU CI green on a self-hosted and a cloud GPU.

## M1 Runtime: one path for every model · v0.34.0

[Milestone](https://github.com/surya-koritala/Glyd/milestone/2) · [Epic #120](https://github.com/surya-koritala/Glyd/issues/120)

- [x] R0 One levels module (`src/levels.rs`).
- [ ] R1 Backend trait (`docs/architecture.md#backends`) over `dev.rs`/`ffi.rs`: memory, streams, events, graphs, module and
      launch, peer access; the four direct-driver call sites first; CUDA behind it with no `glyd bench` regression.
- [ ] R2 Layer-kind library: attention (MHA, GQA, MLA, sliding, linear/delta), Mamba, dense MLP, MoE, norms, rotary
      variants; a CPU f32 reference for each.
- [ ] R3 Model descriptions: config.json → layer graph + weight map (R3a). Port Qwen3.5, Gemma 4, Nemotron-H (R3b), Qwen3
      (R3c), Condor's model (R3d); delete the per-family pipelines and `qwen*serve` paths.
- [ ] R4 One scheduler for all models (R4a): paged KV for every attention kind (R4b), content-addressed blocks with
      vLLM-compatible chained hashes and a radix index (R4c); chunked prefill, prefix cache, preemption; no user cap.
- [ ] R5 Constrained decoding through llguidance (R5a); tool-call and reasoning parsers per family (R5b; openai-harmony for
      gpt-oss).
- [ ] R6 Speculative decoding for any model: draft model, MTP heads, n-gram/prompt lookup.
- [ ] R7 Logits tests per model against the CPU reference; GPU CI on every merge to `main`.
- [ ] R8 `glyd bench` (Rust): one harness, results in `benchmarks/`, regression gate in CI.

Gate: every listed family serves 64 concurrent requests through the same scheduler on an RTX 4080 SUPER and an L4,
outputs matching the reference.

## M2 Install and it works · v0.35.0

[Milestone](https://github.com/surya-koritala/Glyd/milestone/3) · [Epic #121](https://github.com/surya-koritala/Glyd/issues/121)

- [ ] P1 One binary `glyd` (engine linked in): `run`, `serve`, `pull`, `ls`, `rm`, `ps`, `stop`, `fit`, `doctor`, `bench`,
      `pack`, `verify`, `login`, `cluster`.
- [ ] P2 Planner v1 (`docs/architecture.md#adaptation`): `run` and `serve` load by the plan (P2a); exact sizes from file
      headers before download (P2b); measured rates and calibrated predictions (P2c); a visible degrade ladder (P2d).
      Started: `glyd-engine fit` (`src/plan.rs`).
- [ ] P3 Background service: user-level systemd / launchd / Windows service; load on request, unload when idle, several
      models; re-probe before each load, shrink KV and expert caches when other programs take VRAM.
- [ ] P4 Model names: Hugging Face IDs, local paths, GGUF files, and short names (`qwen3.5`, `qwen3.5:27b`) that pick the
      size and level for the machine.
- [ ] P5 APIs: OpenAI chat/completions/responses/embeddings/models (P5a), Anthropic messages with prompt caching (P5b),
      Ollama API (P5c), MCP server for any model (P5d).
- [ ] P6 Drivers: a CUDA 12.8 fatbin (driver ≥ 525) and a CUDA 13 fatbin (driver ≥ 580), chosen with `cuDriverGetVersion`;
      `sm_120a` for consumer Blackwell; APIs newer than 12.0 behind a version check.
- [ ] P7 Platforms: Windows x86_64 (P7a); a small core that downloads the backend it needs, signed and content-addressed
      (P7b); install.sh, winget, Homebrew, Docker per vendor (P7c); macOS arm64 (P7d). A backend that fails says why.
- [ ] P8 Condor folded in: its model runs with `glyd run`; `condor/` keeps scripts that call `glyd`.
- [ ] P9 No Python on any install path; the vLLM plugin stays frozen at 0.32.x.
- [ ] P10 `glyd doctor` and a support bundle.

Gate: on a clean Ubuntu and a clean Windows machine with only the NVIDIA driver, install → first chat token in under
2 minutes (excluding the model download), for the 10 most-run models.

## M3 Models and formats · v0.36.0

[Milestone](https://github.com/surya-koritala/Glyd/milestone/4) · [Epic #122](https://github.com/surya-koritala/Glyd/issues/122)

- [ ] F1 Families: Llama 3.x/4 (F1a), Mistral/Mixtral (F1b), Gemma 2/3/4 (F1c), Qwen 2/2.5/3-MoE (F1d), Phi-4 (F1e),
      gpt-oss (F1f), DeepSeek V3/R1 (F1g), Granite/OLMo/SmolLM (F1h).
- [ ] F2 Vision for every VLM family (F2a); embeddings and rerankers (F2b).
- [ ] F3 GGUF reader (F3a) and quant kernels ported from ggml (F3b); FP8, NVFP4, MXFP4, AWQ, GPTQ, compressed-tensors
      checkpoints as released (F3c).
- [ ] F4 LoRA adapters; many adapters batched together.
- [ ] F5 Pack pipeline: new releases packed, verified (`glyd verify`) and published to `glyd/*` with a quality card
      (KL, one eval) per level.
- [ ] F6 Long context past 40,960 positions (rope scaling, KV tables, chunked prefill at 128K+).
- [ ] F7 Retire the vLLM plugin path.

Gate: the top 50 text-generation models on Hugging Face run; a new release in a supported family runs within 24 hours.

## M4 Every vendor · v0.37.0

[Milestone](https://github.com/surya-koritala/Glyd/milestone/5) · [Epic #123](https://github.com/surya-koritala/Glyd/issues/123)

- [ ] V0 Kernel portability: a warp-size and lane-mask abstraction (64-bit masks); a tensor-core fragment layer with four
      paths: `mma.sync`, RDNA3 WMMA, RDNA4 WMMA, CDNA MFMA.
- [ ] V1 AMD RDNA 3/4: measured, in CI, in release builds (ROCm on Linux; HIP on Windows, one GPU).
- [ ] V2 AMD CDNA 3/4 (MI300X, MI355X): wave64 kernels, MFMA products, hipBLASLt, RCCL.
- [ ] V3 Apple Silicon: a thin `-sys` crate over `mlx-c` (V3a); Glyd formats as MLX custom Metal kernels (V3b); the GPU's
      working-set limit (about 75% of RAM) as its memory budget (V3c).
- [ ] V4 CPU backend for whole models (AVX2, AVX-512/AMX, NEON/SVE).
- [ ] V5 NVIDIA Turing (T4, RTX 20): bf16 weights as fp16 at load (range-checked), register double-buffering in place of
      cp.async; Volta and older stay out (CUDA 13 dropped them).
- [ ] V6 A Vulkan or CPU fallback for other GPUs (evaluate CubeCL there only); vendor-native kernels stay the fast path.
- [ ] V7 Blackwell tensor paths: `tcgen05` on sm_100, block-scaled FP4/FP8 `mma` on sm_120a.

Gate: the M3 model set passes on NVIDIA (T4, L4, RTX 4080 SUPER, A100, H100), AMD (RX 7900 XTX, MI300X), Apple (M1+, M4)
and CPU (x86_64, arm64).

## M5 Many GPUs, many machines · v0.38.0

[Milestone](https://github.com/surya-koritala/Glyd/milestone/6) · [Epic #124](https://github.com/surya-koritala/Glyd/issues/124)

- [ ] C1 Planner v2: one placement across VRAM, peer GPUs, host RAM, NVMe and other machines; Condor's expert policy for
      every MoE.
- [ ] C2 In one node: NCCL/RCCL through cudarc and a custom all-reduce for decode (C2a); tensor parallel for every layer
      kind (C2b); expert parallel (C2c); overlapped pipeline for layer splits (C2d); GeForce without peer access (C2e).
- [ ] C3 `glyd cluster`: Starlings' enrollment and transport for N machines, mixed vendors, every model; a failed node
      recovers without a full restart.
- [ ] C4 A `KvTransport` trait: TLS over TCP by default, NIXL (UCX, EFA, GDS) for RDMA; expert parallel via NCCL send/recv,
      then NCCL EP.
- [ ] C5 Starlings crates in the main workspace; `glyd cluster` replaces the `starlings` binary.

Gate: two consumer machines run a model neither holds alone at the speed `glyd fit` predicted (±10%); a 600B+ MoE runs
across two 8-GPU nodes.

## M6 Datacenter serving · v1.0.0

[Milestone](https://github.com/surya-koritala/Glyd/milestone/7) · [Epic #125](https://github.com/surya-koritala/Glyd/issues/125)

- [ ] D1 Prefill/decode disaggregation with compressed KV transfer; KV offload to RAM/NVMe; a shared KV store.
- [ ] D2 KV-aware routing by integration, not a router of our own: vLLM-named metrics and KV events so llm-d, the Gateway API
      inference extension and Dynamo schedule Glyd.
- [ ] D3 Operations: Prometheus `/metrics`, OpenTelemetry traces, health/readiness, drain; Helm chart, LeaderWorkerSet,
      Gateway API inference extension.
- [ ] D4 Wide expert parallelism across nodes; multi-node NVLink (NVL72).
- [ ] D5 Weight-update API for RL rollouts; batch-invariant deterministic mode.
- [ ] D6 Signed releases (Sigstore), SBOM.
- [ ] D7 Published benchmark comparison against vLLM and SGLang, rerun every release.
- [ ] D8 1.0 stability: CLI, APIs, C ABI and pack formats versioned with a compatibility policy.

Gate: goodput at or above vLLM and SGLang on the top 10 models on H100 and MI300X with less memory, published with logs.

## Always

- `glyd bench` (Rust) replaces the Python benchmarks; results in `benchmarks/`.
- Python is ported or removed as each milestone replaces it (`glyd-gpu-py/`, `gpu/*.py`, `glyd-engine/tools/*.py`,
  `condor/bench`, `scripts/*.py` → `cargo xtask`).
