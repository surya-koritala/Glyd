# Roadmap

Milestones in dependency order; each ships as one release. Tracks within a milestone run in parallel. Each milestone ends
on a gate measured on real hardware, with the result committed under `benchmarks/`. Design:
[docs/architecture.md](docs/architecture.md). Every item is a GitHub issue with tasks, acceptance criteria, test hardware and
dependencies, under its milestone's epic; the epics are tracked in [#118](https://github.com/surya-koritala/Glyd/issues/118). The checkboxes follow the issues' state; there is no generator in this repository, so they are updated by hand.

| Release | Milestone | Epic | Issues |
| :--- | :--- | :--- | ---: |
| v0.33.0 | [M0 Open-source baseline](https://github.com/surya-koritala/Glyd/milestone/1) | [#119](https://github.com/surya-koritala/Glyd/issues/119) | 14 |
| v0.34.0 | [M1 Front door](https://github.com/surya-koritala/Glyd/milestone/2) | [#120](https://github.com/surya-koritala/Glyd/issues/120) | 27 |
| v0.35.0 | [M2 One runtime](https://github.com/surya-koritala/Glyd/milestone/3) | [#121](https://github.com/surya-koritala/Glyd/issues/121) | 37 |
| v0.36.0 | [M3 Core models, formats and desktop](https://github.com/surya-koritala/Glyd/milestone/4) | [#122](https://github.com/surya-koritala/Glyd/issues/122) | 27 |
| v0.37.0 | [M4 Many GPUs, many machines](https://github.com/surya-koritala/Glyd/milestone/6) | [#124](https://github.com/surya-koritala/Glyd/issues/124) | 13 |
| v0.38.0 | [M5a Apple Silicon, CPU and desktop extras](https://github.com/surya-koritala/Glyd/milestone/5) | [#123](https://github.com/surya-koritala/Glyd/issues/123) | 11 |
| v0.39.0 | [M5b AMD ROCm and older NVIDIA](https://github.com/surya-koritala/Glyd/milestone/9) | [#445](https://github.com/surya-koritala/Glyd/issues/445) | 7 |
| v1.0.0 | [M6 Stable and operable](https://github.com/surya-koritala/Glyd/milestone/7) | [#125](https://github.com/surya-koritala/Glyd/issues/125) | 8 |
| after 1.0 | [Later: datacenter scale](https://github.com/surya-koritala/Glyd/milestone/8) | [#210](https://github.com/surya-koritala/Glyd/issues/210) | 7 |

## Where it stands (v0.33.1, 2026-11-20)

Released, on main and planned are listed apart.

**Released: v0.33.0** (2026-11-20, Apache-2.0; the first release built only from this repository)
- `install.sh` on Linux with an NVIDIA GPU installs the Rust `glyd` and the engine with its GPU library from the
  release's engine bundle (about 100 MB, no Python); the engine has run on x86_64 only (the aarch64 bundle is built, not yet run).
  `GLYD_WITH_VLLM=1` installs the Python tool with vLLM and PyTorch.
- `glyd run` goes to the engine by default where there is no Python tool, and, with the tool, on the RTX 4080 SUPER (the GPU it
  has been measured on); other GPUs go through the frozen vLLM path unless `GLYD_ENGINE=1`.
- Compression of files, the store, and the Python GPU package and vLLM plugin.

**Before v0.33.0:** v0.32.1 (2026-10-17, BUSL-1.1) went through vLLM and Python and had no engine tarball.

**In v0.33.0** (and on main after it)
- One repository under Apache-2.0 with the engine, the GPU library, Condor and Starlings (#263); raw data out of the tree
  (#265); rustfmt, clippy and cargo-deny in CI (#266, part of S8); one table of what the engine runs, shared by `fit`,
  `run` and `serve` (#267, P16); security updates (S7, #211).
- The engine runs Qwen3 dense (lossless); Qwen3.5/3.6/3.8 dense (lossy levels only); Gemma 4 12B and 31B and Nemotron-H
  (lossless and lossy levels); and, lossless, Kimi-K3, GLM-5.3-Flash, DeepSeek-V4 and Muse Spark, each through its own
  pipeline. Qwen3.8-Flash-Next (Condor's model) runs through Condor's `qwenserve`, not `glyd run`. The list is
  `glyd-engine/src/support.rs`.
- With the Python tool, `glyd run` hands a model to the engine only on GPUs listed in `GPUS` (`glyd-engine/src/machine.rs`):
  the RTX 4080 SUPER is the one so far (S10), the others use vLLM unless `GLYD_ENGINE=1`.
- Batching on the Qwen3 path only (up to 8 sequences); other families serve one request at a time.
- MoE experts on GPU/RAM/NVMe with CPU compute for misses; layer split over the GPUs of one machine for Nemotron-H,
  Kimi-K3, GLM-5.3-Flash, DeepSeek-V4 and Muse Spark; two-machine pipeline (Starlings, beta). `glyd-engine fit` plans before
  download but `run` does not yet load by the plan (P2a).
- NVIDIA Ampere to Blackwell, Linux x86_64, driver 580 or newer. AMD RDNA 3/4 as development builds (`GLYD_HIP=1`). The GPU
  workflow runs on one self-hosted machine (sm_89) with a small model; Windows and macOS CI compile and unit-test the engine,
  the codec, the store and the CLI (the engine itself runs on Linux with an NVIDIA GPU only).

**Not there**: Windows engine release, aarch64, Apple, CPU-only runs, CUDA 12 drivers (M1 and later).

## M0 Open-source baseline · v0.33.0

[Milestone](https://github.com/surya-koritala/Glyd/milestone/1) · [Epic #119](https://github.com/surya-koritala/Glyd/issues/119)

One public Apache-2.0 repository for all of Glyd, releasing and running the Rust engine without Python, with green Linux/Windows/macOS compile CI, real-GPU CI, security updates, and docs matching the code.

- [ ] S7 Security updates: rustls and Dependabot for every manifest ([#211](https://github.com/surya-koritala/Glyd/issues/211))
- [x] S1 Merge the monorepo branch and retire the predecessor repositories ([#126](https://github.com/surya-koritala/Glyd/issues/126))
- [ ] S2 Self-hosted GPU runner for gpu.yml ([#127](https://github.com/surya-koritala/Glyd/issues/127))
- [ ] S4 Release v0.33.0 from this repository ([#129](https://github.com/surya-koritala/Glyd/issues/129))
- [x] S8 Rust baseline: toolchain, clippy, rustfmt, cargo-deny, all crates tested in CI ([#212](https://github.com/surya-koritala/Glyd/issues/212))
- [ ] S6 Required checks and branch protection on main ([#131](https://github.com/surya-koritala/Glyd/issues/131))
- [x] S9 Windows and macOS compile and unit-test CI ([#213](https://github.com/surya-koritala/Glyd/issues/213))
- [ ] S10 The engine on by default where it has been measured ([#214](https://github.com/surya-koritala/Glyd/issues/214))
- [ ] S11 Docs and repository identity match the code ([#215](https://github.com/surya-koritala/Glyd/issues/215))
- [x] S12 Security policy for the inference product ([#216](https://github.com/surya-koritala/Glyd/issues/216))
- [ ] S13 Community: discussions, issue templates, triage, first issues ([#217](https://github.com/surya-koritala/Glyd/issues/217))
- [x] S14 Governance: maintainers, decisions and trademark ([#218](https://github.com/surya-koritala/Glyd/issues/218))
- [x] S15 Hardware access plan for CI and release gates ([#219](https://github.com/surya-koritala/Glyd/issues/219))
- [x] S16 One kernel ABI check and no closed-source leftovers ([#220](https://github.com/surya-koritala/Glyd/issues/220))

Gate: A tag on main produces every release artifact from this repository, including engine tarballs; clean Linux + NVIDIA driver reaches first token without Python; zero open security advisories; Linux/Windows/macOS compile CI and GPU CI are green.

## M1 Front door · v0.34.0

[Milestone](https://github.com/surya-koritala/Glyd/milestone/2) · [Epic #120](https://github.com/surya-koritala/Glyd/issues/120)

Make today's engine the lowest-friction way to run a model on an NVIDIA PC: one zero-Python `glyd` binary, CUDA 12 & 13 builds, Windows & aarch64, `fit == run` (`P2a`/`P16`), background service, OpenAI/Anthropic/Ollama APIs, secure defaults, signed releases, and the `R7` logits + `R8` bench harnesses that make M2's refactor safe.

- [x] R0 One levels module ([#132](https://github.com/surya-koritala/Glyd/issues/132))
- [ ] R8 glyd bench: one benchmark harness with a regression gate ([#133](https://github.com/surya-koritala/Glyd/issues/133))
- [x] R5a Constrained decoding with llguidance ([#143](https://github.com/surya-koritala/Glyd/issues/143))
- [x] R5b Tool-call and reasoning parsers for every family ([#144](https://github.com/surya-koritala/Glyd/issues/144))
- [x] R7 Per-model logits tests on every merge ([#146](https://github.com/surya-koritala/Glyd/issues/146))
- [x] P1 One binary: glyd with the engine linked ([#147](https://github.com/surya-koritala/Glyd/issues/147))
- [ ] P16 One capability matrix for fit, run and refusals ([#221](https://github.com/surya-koritala/Glyd/issues/221))
- [ ] P2a The plan is what loads, on today's paths ([#148](https://github.com/surya-koritala/Glyd/issues/148))
- [ ] P11 Secure defaults for every listener ([#222](https://github.com/surya-koritala/Glyd/issues/222))
- [ ] P3 Background service: load on request, unload when idle ([#152](https://github.com/surya-koritala/Glyd/issues/152))
- [ ] P4 Model names, manifest and pull/ls/rm ([#153](https://github.com/surya-koritala/Glyd/issues/153))
- [x] P5a OpenAI API completeness: Responses, errors, streaming options ([#154](https://github.com/surya-koritala/Glyd/issues/154))
- [x] P5b Anthropic Messages API: full streaming events and tool use ([#155](https://github.com/surya-koritala/Glyd/issues/155))
- [x] P5c Ollama-compatible API ([#156](https://github.com/surya-koritala/Glyd/issues/156))
- [ ] P6 CUDA 12 and CUDA 13 builds chosen at run time ([#158](https://github.com/surya-koritala/Glyd/issues/158))
- [ ] P7a Windows x86_64 release of the engine ([#159](https://github.com/surya-koritala/Glyd/issues/159))
- [x] P10 glyd doctor and a support bundle ([#165](https://github.com/surya-koritala/Glyd/issues/165))
- [x] F3a GGUF reader ([#176](https://github.com/surya-koritala/Glyd/issues/176))
- [x] C5 Starlings crates into the main workspace ([#199](https://github.com/surya-koritala/Glyd/issues/199))
- [ ] D3 Metrics, health, readiness and drain ([#204](https://github.com/surya-koritala/Glyd/issues/204))
- [ ] D6 Signed releases and attestations ([#207](https://github.com/surya-koritala/Glyd/issues/207))
- [ ] P12 Fuzz the engine's parsers ([#224](https://github.com/surya-koritala/Glyd/issues/224))
- [ ] P14 User documentation site ([#225](https://github.com/surya-koritala/Glyd/issues/225))
- [ ] P15 Model catalog: ownership and review ([#226](https://github.com/surya-koritala/Glyd/issues/226))
- [ ] P17 Tools connect with default settings ([#227](https://github.com/surya-koritala/Glyd/issues/227))
- [ ] P18 aarch64 Linux: GH200, GB200, DGX Spark and Jetson ([#228](https://github.com/surya-koritala/Glyd/issues/228))
- [x] P19 The codec's place: crate, CLI and docs ([#229](https://github.com/surya-koritala/Glyd/issues/229))

Gate: Clean Ubuntu 24.04 & Windows 11 with only the NVIDIA driver (8 GB, 16 GB, oldest supported driver): install -> first token in <2 min for `benchmarks/gates/m1-models.txt`; printed plan is what loads; Open WebUI, Continue, and OpenAI SDK connect with defaults; valid JSON-schema output on all 10; one-line refusal on Apple/AMD; no Python anywhere.

## M2 One runtime · v0.35.0

[Milestone](https://github.com/surya-koritala/Glyd/milestone/3) · [Epic #121](https://github.com/surya-koritala/Glyd/issues/121)

Replace the 6 per-family pipelines and separate GPU library build with one config-driven runtime (`ModelStep`, `Backend`, `LayerImpl`), async overlap scheduler, hybrid paged KV + 12-bit KV (hd 256/512/MLA), bucketed/piecewise CUDA graphs, universal MoE tiering, CUDA device seam (`R1a`), NVIDIA kernel library delegation (`V8a`: CUTLASS/FlashInfer/cuBLASLt), and standard families (Llama 3, Qwen 2/2.5/3 MoE, Mistral, Phi-4, Granite, OLMo, SmolLM).

- [ ] R9 Runtime design: model step, layer-kind interface, backend seam ([#230](https://github.com/surya-koritala/Glyd/issues/230))
- [ ] R15 A sound device API and typed kernel launches ([#236](https://github.com/surya-koritala/Glyd/issues/236))
- [ ] R16 A runtime context and typed errors ([#237](https://github.com/surya-koritala/Glyd/issues/237))
- [ ] R1a CUDA device layer and backend seam at the layer-kind level ([#134](https://github.com/surya-koritala/Glyd/issues/134))
- [ ] R17 One kernel build: fold the GPU library into the engine ([#238](https://github.com/surya-koritala/Glyd/issues/238))
- [ ] V8a NVIDIA kernel libraries where they are faster (CUTLASS, FlashInfer, cuBLASLt) ([#255](https://github.com/surya-koritala/Glyd/issues/255))
- [ ] R2 Layer-kind library with a CPU f32 reference for each kind ([#135](https://github.com/surya-koritala/Glyd/issues/135))
- [ ] R3a Config-driven model descriptions ([#136](https://github.com/surya-koritala/Glyd/issues/136))
- [ ] R10 Pack-format compatibility suite ([#231](https://github.com/surya-koritala/Glyd/issues/231))
- [ ] R4a Model-agnostic scheduler ([#138](https://github.com/surya-koritala/Glyd/issues/138))
- [ ] R4b Paged KV for every attention kind ([#141](https://github.com/surya-koritala/Glyd/issues/141))
- [ ] R4f Lossless KV for head dims 256 and 512 and MLA latents ([#332](https://github.com/surya-koritala/Glyd/issues/332))
- [ ] R4c Content-addressed KV blocks and a radix prefix index ([#142](https://github.com/surya-koritala/Glyd/issues/142))
- [ ] R4d Preemption by swapping compressed KV pages and state slots to host ([#328](https://github.com/surya-koritala/Glyd/issues/328))
- [ ] R4e Graphs for batched and mixed steps on the shared executor ([#329](https://github.com/surya-koritala/Glyd/issues/329))
- [ ] R11 Async HTTP front end and a request-level engine interface ([#233](https://github.com/surya-koritala/Glyd/issues/233))
- [ ] P5e Anthropic prompt caching on content-addressed blocks ([#241](https://github.com/surya-koritala/Glyd/issues/241))
- [ ] R12 Expert tiering for every MoE, and dense layers in RAM ([#232](https://github.com/surya-koritala/Glyd/issues/232))
- [ ] R12b Expert store at many rows: one read per missed expert a step, per-layer statistics ([#327](https://github.com/surya-koritala/Glyd/issues/327))
- [ ] P2b Exact model sizes from file headers before download ([#149](https://github.com/surya-koritala/Glyd/issues/149))
- [ ] P2c Measured rates and calibrated speed predictions ([#150](https://github.com/surya-koritala/Glyd/issues/150))
- [ ] P2d Visible degrade ladder with confirmation ([#151](https://github.com/surya-koritala/Glyd/issues/151))
- [ ] P2f Every path loads by the plan; per-path planners deleted ([#239](https://github.com/surya-koritala/Glyd/issues/239))
- [ ] P4b Short names pick the size and level for this machine ([#240](https://github.com/surya-koritala/Glyd/issues/240))
- [ ] F1l Qwen3.8 dense at the lossless level on the engine, batched ([#331](https://github.com/surya-koritala/Glyd/issues/331))
- [ ] F1m Recent releases not yet loaded: GLM-5.3, Qwen3.8-2.4T-A95B, Kimi-K3.1-Pro, DeepSeek-V4.1, DSpark drafters ([#333](https://github.com/surya-koritala/Glyd/issues/333))
- [ ] F1a Llama 3.x ([#166](https://github.com/surya-koritala/Glyd/issues/166))
- [ ] F1b Mistral and Ministral (dense) ([#167](https://github.com/surya-koritala/Glyd/issues/167))
- [ ] F1d Qwen 2/2.5 and Qwen3 MoE ([#169](https://github.com/surya-koritala/Glyd/issues/169))
- [ ] F1e Phi-4 family ([#170](https://github.com/surya-koritala/Glyd/issues/170))
- [ ] F1h Granite, OLMo, SmolLM and other standard dense models ([#173](https://github.com/surya-koritala/Glyd/issues/173))
- [ ] R3d Condor's model (Qwen3.8-Flash-Next) on the shared executor ([#140](https://github.com/surya-koritala/Glyd/issues/140))
- [ ] R3b Retire the hand-written Qwen3.5, Gemma 4 and Nemotron-H engines ([#137](https://github.com/surya-koritala/Glyd/issues/137))
- [ ] R3c Qwen3 dense on the shared executor ([#139](https://github.com/surya-koritala/Glyd/issues/139))
- [ ] R13 Retire or freeze the GLM development engine ([#234](https://github.com/surya-koritala/Glyd/issues/234))
- [ ] R14 A quality bar for lossy levels ([#235](https://github.com/surya-koritala/Glyd/issues/235))
- [ ] P20 Consumer benchmarks published every release ([#242](https://github.com/surya-koritala/Glyd/issues/242))

Gate: Pinned per-GPU list serves 64 concurrent requests through one scheduler on RTX 4080 SUPER and L4 with logits matching `Ref`; Kimi-K3, DeepSeek-V4 and GLM-5.3 keep single-user parity on reference hardware; no `glyd bench` regression beyond noise band; every pack since v0.29 still loads.

## M3 Core models, formats and desktop · v0.36.0

[Milestone](https://github.com/surya-koritala/Glyd/milestone/4) · [Epic #122](https://github.com/surya-koritala/Glyd/issues/122)

Focused consumer adoption launch on NVIDIA: Top-50 HF text models, canonical GGUF quant kernels, quantized safetensors (`FP8`/`FP4`), consumer Blackwell FP4/FP8 (`V7a`), long context >40K (`F6`), Condor folded into `glyd run` (`P8`), universal Hummingbird (`#429`), live VRAM pressure (`P3b`), pre-packed top-15 hub & coding-agent launch kit (`P25`), interactive `plan.rs` web fit calculator (`P26`), installers on every channel, retire vLLM (`F7`), and published consumer benchmarks vs. Ollama, llama.cpp and LM Studio.

- [ ] F3b GGUF quantization kernels ([#177](https://github.com/surya-koritala/Glyd/issues/177))
- [ ] F3c Quantized safetensors releases as released ([#178](https://github.com/surya-koritala/Glyd/issues/178))
- [ ] V7a Consumer Blackwell FP4 and FP8 ([#254](https://github.com/surya-koritala/Glyd/issues/254))
- [ ] Hummingbird for the recent MoE families, calibrated in Rust ([#429](https://github.com/surya-koritala/Glyd/issues/429))
- [ ] F6 Long context past 40,960 tokens ([#181](https://github.com/surya-koritala/Glyd/issues/181))
- [ ] F1c Gemma 2, Gemma 3 and the remaining Gemma 4 models ([#168](https://github.com/surya-koritala/Glyd/issues/168))
- [ ] F1f gpt-oss ([#171](https://github.com/surya-koritala/Glyd/issues/171))
- [ ] F1g DeepSeek R1 distills ([#172](https://github.com/surya-koritala/Glyd/issues/172))
- [ ] F1i Llama 4 ([#244](https://github.com/surya-koritala/Glyd/issues/244))
- [ ] F1j Mixtral ([#245](https://github.com/surya-koritala/Glyd/issues/245))
- [ ] F1k DeepSeek V3 and R1 (full) ([#246](https://github.com/surya-koritala/Glyd/issues/246))
- [ ] P8 Fold Condor into glyd run ([#163](https://github.com/surya-koritala/Glyd/issues/163))
- [ ] P3b Adapt to VRAM pressure at run time ([#247](https://github.com/surya-koritala/Glyd/issues/247))
- [ ] P9 No Python on any install path ([#164](https://github.com/surya-koritala/Glyd/issues/164))
- [ ] F7 Retire the vLLM plugin path ([#182](https://github.com/surya-koritala/Glyd/issues/182))
- [ ] P22 Scripts to cargo xtask; Python references to Rust ([#251](https://github.com/surya-koritala/Glyd/issues/251))
- [ ] F8 Model license and gating policy ([#243](https://github.com/surya-koritala/Glyd/issues/243))
- [ ] F5 Automated pack pipeline for new releases ([#180](https://github.com/surya-koritala/Glyd/issues/180))
- [ ] F9 Day-0 model support process ([#248](https://github.com/surya-koritala/Glyd/issues/248))
- [ ] F10 Rust library API and C ABI for embedding ([#249](https://github.com/surya-koritala/Glyd/issues/249))
- [ ] P13 Code-signing identities for Windows and macOS ([#223](https://github.com/surya-koritala/Glyd/issues/223))
- [ ] P7c Installers: install.sh, winget, Homebrew, Docker ([#161](https://github.com/surya-koritala/Glyd/issues/161))
- [ ] P21 glyd update ([#250](https://github.com/surya-koritala/Glyd/issues/250))
- [ ] P23 Corporate networks: proxies, custom CAs, offline installs ([#252](https://github.com/surya-koritala/Glyd/issues/252))
- [ ] P24 Opt-in crash and compatibility reports ([#253](https://github.com/surya-koritala/Glyd/issues/253))
- [ ] P25 Pre-packed top-15 model hub and local coding-agent launch kit ([#448](https://github.com/surya-koritala/Glyd/issues/448))
- [ ] P26 Interactive GPU fit calculator on the docs site (plan.rs to WASM) ([#449](https://github.com/surya-koritala/Glyd/issues/449))

Gate: Dated snapshot of top-50 HF text-generation models runs; 10 most-pulled local models run (including GGUF); top-15 pre-packed repos hit first token with zero local pack wait; Claude Code / Continue / Open WebUI pass tool-call checks; every installer channel passes; consumer benchmarks published per VRAM tier.

## M4 Many GPUs, many machines · v0.37.0

[Milestone](https://github.com/surya-koritala/Glyd/milestone/6) · [Epic #124](https://github.com/surya-koritala/Glyd/issues/124)

One SLO-aware planner and runtime from 1 GPU to 8-GPU datacenter nodes and N-machine clusters (NVIDIA first): NCCL + custom decode all-reduce (`C2a`), SLO-aware Planner v2 (`C1`), TP (`C2b`), intra-node EP (`C2c`), overlapped PP (`C2d`), GeForce non-P2P (`C2e`), `glyd cluster` replacing Starlings (`C3`/`C5b`), `KvTransport` TLS/TCP (`C4`), 8-GPU OpenAI server (`C6`), compressed streaming/P2P weight distribution (`C7`), and sleep-to-host-RAM multi-model tenancy (`C8`).

- [ ] C2a Collectives: NCCL/RCCL via cudarc and a custom all-reduce for decode ([#193](https://github.com/surya-koritala/Glyd/issues/193))
- [ ] C1 Planner v2: one placement across GPUs, RAM, NVMe and machines ([#194](https://github.com/surya-koritala/Glyd/issues/194))
- [ ] C2b Tensor parallelism for every layer kind ([#195](https://github.com/surya-koritala/Glyd/issues/195))
- [ ] C2c Expert parallelism inside a node ([#196](https://github.com/surya-koritala/Glyd/issues/196))
- [ ] C2d Overlapped pipeline for layer splits ([#197](https://github.com/surya-koritala/Glyd/issues/197))
- [ ] C2e GeForce multi-GPU without peer access ([#198](https://github.com/surya-koritala/Glyd/issues/198))
- [ ] C5b glyd cluster replaces the starlings binary ([#256](https://github.com/surya-koritala/Glyd/issues/256))
- [ ] C3 glyd cluster: N machines, any GPU family, any model ([#200](https://github.com/surya-koritala/Glyd/issues/200))
- [ ] C4 KvTransport trait with TLS over TCP ([#201](https://github.com/surya-koritala/Glyd/issues/201))
- [ ] C6 One 8-GPU node as a drop-in OpenAI server ([#257](https://github.com/surya-koritala/Glyd/issues/257))
- [ ] C7 Compressed streaming and peer-to-peer weight distribution for fast cold start ([#450](https://github.com/surya-koritala/Glyd/issues/450))
- [ ] C8 Multi-model tenancy and sleep-to-host-RAM on GPU servers ([#451](https://github.com/surya-koritala/Glyd/issues/451))
- [ ] S3 Ephemeral cloud GPU runners for gpu-cloud.yml ([#128](https://github.com/surya-koritala/Glyd/issues/128))

Gate: Two consumer machines run a model neither holds alone within +-10% of `glyd fit`; 600B-class MoE runs on one 8x H100 node with host-RAM expert tiering (EP decode within 15% of SGLang); 2-GPU TP >= 1.7x 1 GPU; one 8-GPU node serves OpenAI API with DP replicas, standard metrics, compressed streaming load, and <5 s sleep-to-RAM wake.

## M5a Apple Silicon, CPU and desktop extras · v0.38.0

[Milestone](https://github.com/surya-koritala/Glyd/milestone/5) · [Epic #123](https://github.com/surya-koritala/Glyd/issues/123)

Bring the unified runtime to Apple Silicon Macs (`mlx-c` + custom Glyd Metal kernels + unified-memory planning) and CPU-only machines (x86_64/arm64), alongside macOS arm64 packaging (`P7d`) and the deferred desktop modalities/features: Vision (`F2a`), Embeddings/Rerankers (`F2b`), Batched LoRA (`F4`), and Speculative Decoding for any model (`R6`).

- [ ] P7d macOS arm64 build of glyd ([#162](https://github.com/surya-koritala/Glyd/issues/162))
- [ ] S5 macOS GPU CI runner for the Metal backend ([#130](https://github.com/surya-koritala/Glyd/issues/130))
- [ ] V3a Metal backend through mlx-c ([#186](https://github.com/surya-koritala/Glyd/issues/186))
- [ ] V3b Glyd formats as Metal kernels ([#187](https://github.com/surya-koritala/Glyd/issues/187))
- [ ] V3c Unified-memory planning on every vendor ([#188](https://github.com/surya-koritala/Glyd/issues/188))
- [ ] P18b aarch64 measured on GH200 and DGX Spark ([#459](https://github.com/surya-koritala/Glyd/issues/459))
- [ ] V4 CPU backend for whole models ([#189](https://github.com/surya-koritala/Glyd/issues/189))
- [ ] F2a Vision for every VLM family ([#174](https://github.com/surya-koritala/Glyd/issues/174))
- [ ] F2b Embedding and reranker models ([#175](https://github.com/surya-koritala/Glyd/issues/175))
- [ ] F4 LoRA adapters, batched ([#179](https://github.com/surya-koritala/Glyd/issues/179))
- [ ] R6 Speculative decoding for any model ([#145](https://github.com/surya-koritala/Glyd/issues/145))

Gate: Apple M1/M4 and CPU x86_64/arm64 pass KL thresholds against CUDA on the fitting M3 model subset; Vision, Embeddings, Batched LoRA and Speculative Decoding pass logits and `glyd bench` gates on reference hardware.

## M5b AMD ROCm and older NVIDIA · v0.39.0

[Milestone](https://github.com/surya-koritala/Glyd/milestone/9) · [Epic #445](https://github.com/surya-koritala/Glyd/issues/445)

Extend Glyd's single binary to AMD GPUs (RDNA 3/4, Strix Halo, CDNA 3/4 MI300X/MI325X/MI355X) and older NVIDIA Turing (`sm_75`), with a HIP device layer (`R1b`), warp/fragment portability (`V0`), AMD kernel libraries (`V8b`: hipBLASLt, aiter, composable_kernel), and dynamic backend bundles (`P7b`).

- [ ] R1b HIP device layer over the backend seam ([#446](https://github.com/surya-koritala/Glyd/issues/446))
- [ ] P7b Small core that downloads its backend ([#160](https://github.com/surya-koritala/Glyd/issues/160))
- [ ] V0 Kernel portability layer: warp size, lane masks, tensor-core fragments ([#183](https://github.com/surya-koritala/Glyd/issues/183))
- [ ] V1 AMD RDNA 3/4 and Strix Halo: a beta first, then speed ([#184](https://github.com/surya-koritala/Glyd/issues/184))
- [ ] V2 AMD CDNA 3/4 (MI300X, MI325X, MI355X) ([#185](https://github.com/surya-koritala/Glyd/issues/185))
- [ ] V8b AMD kernel libraries where they are faster (hipBLASLt, aiter, composable_kernel) ([#447](https://github.com/surya-koritala/Glyd/issues/447))
- [ ] V5 NVIDIA Turing (sm_75): fp16 paths ([#190](https://github.com/surya-koritala/Glyd/issues/190))

Gate: Per device (NVIDIA T4; AMD RX 7900 XTX, RX 9070 XT, Strix Halo, MI300X), the fitting M3 subset passes KL thresholds against CUDA with committed `benchmarks/` records; `P7b` downloads signed vendor backend bundles cleanly.

## M6 Stable and operable · v1.0.0

[Milestone](https://github.com/surya-koritala/Glyd/milestone/7) · [Epic #125](https://github.com/surya-koritala/Glyd/issues/125)

Commit to stable CLI, HTTP/token APIs (`D8`/`D9`), library API and pack formats, with Kubernetes operations (`D3b`: Helm, LeaderWorkerSet, dashboards), KV-aware router integration (`D2`: llm-d, Gateway API Inference Extension, Dynamo, SMG), deterministic batch-invariant mode (`D5`), supply-chain security (`D6b`: SBOM, provenance, reproducible builds), and a published comparison with vLLM and SGLang (`D7`).

- [ ] D2 Integrate with KV-aware routers: llm-d, Gateway API, SMG ([#203](https://github.com/surya-koritala/Glyd/issues/203))
- [ ] D5 Deterministic, batch-invariant mode ([#206](https://github.com/surya-koritala/Glyd/issues/206))
- [ ] D7 Published comparison with vLLM and SGLang, per quality tier ([#208](https://github.com/surya-koritala/Glyd/issues/208))
- [ ] D8 1.0 stability: CLI, API and pack formats ([#209](https://github.com/surya-koritala/Glyd/issues/209))
- [ ] D3b Helm, LeaderWorkerSet, images and dashboards ([#258](https://github.com/surya-koritala/Glyd/issues/258))
- [ ] D6b SBOM, provenance level and reproducible builds ([#259](https://github.com/surya-koritala/Glyd/issues/259))
- [ ] D9 Token-level engine API for Dynamo and SMG ([#260](https://github.com/surya-koritala/Glyd/issues/260))
- [ ] S14b Maintainers besides the founder ([#458](https://github.com/surya-koritala/Glyd/issues/458))

Gate: Documented stable surface with CI compatibility checks; `llm-d` routes to Glyd pods by prefix-cache hits; Helm + LWS install passes readiness on a GPU cluster; SBOM & provenance on every artifact; published comparison vs. vLLM & SGLang per quality tier (lossless vs. bf16, lossy vs. FP8/NVFP4) on H100 (and B200 via `V8a` `sm_100` kernels).

## Later: datacenter scale (after 1.0)

[Milestone](https://github.com/surya-koritala/Glyd/milestone/8) · [Epic #210](https://github.com/surya-koritala/Glyd/issues/210)

Hyperscale and specialty work intentionally deferred until single-node and small-cluster adoption is proven: prefill/decode disaggregation with compressed KV transfer (`D1`), NIXL RDMA transport (`C4b`), wide expert parallelism & multi-node NVLink/NVL72 (`D4`), custom `tcgen05` kernels for Glyd's compressed formats (`V7`), RL rollout weight updates (`D5b`), Vulkan fallback (`V6`), and MCP server wrapper (`P5d`).

- [ ] P5d MCP server for any model ([#157](https://github.com/surya-koritala/Glyd/issues/157))
- [ ] V6 A fallback for other GPUs ([#191](https://github.com/surya-koritala/Glyd/issues/191))
- [ ] V7 Datacenter Blackwell: tcgen05 ([#192](https://github.com/surya-koritala/Glyd/issues/192))
- [ ] D1 Prefill/decode disaggregation with compressed KV transfer ([#202](https://github.com/surya-koritala/Glyd/issues/202))
- [ ] C4b NIXL transport for RDMA ([#261](https://github.com/surya-koritala/Glyd/issues/261))
- [ ] D4 Wide expert parallelism across nodes and multi-node NVLink ([#205](https://github.com/surya-koritala/Glyd/issues/205))
- [ ] D5b Weight updates for RL rollouts ([#262](https://github.com/surya-koritala/Glyd/issues/262))

Gate: Revisited after v1.0.0, driven by user demand.

## Always

- `glyd bench` (Rust) replaces the Python benchmarks; results in `benchmarks/`.
- Python is ported or removed as each milestone replaces it (`glyd-gpu-py/`, `gpu/*.py`, `glyd-engine/tools/*.py`,
  `condor/bench`, `scripts/*.py` → `cargo xtask`).
