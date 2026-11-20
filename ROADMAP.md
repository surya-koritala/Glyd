# Roadmap

Milestones in dependency order; each ships as one release. Tracks within a milestone run in parallel. Each milestone ends
on a gate measured on real hardware, with the result committed under `benchmarks/`. Design:
[docs/architecture.md](docs/architecture.md). Every item is a GitHub issue with tasks, acceptance criteria, test hardware and
dependencies, under its milestone's epic; the epics are tracked in [#118](https://github.com/surya-koritala/Glyd/issues/118). The checkboxes follow the issues' state; there is no generator in this repository, so they are updated by hand.

| Release | Milestone | Epic | Issues |
| :--- | :--- | :--- | ---: |
| v0.33.0 | [M0 Open-source baseline](https://github.com/surya-koritala/Glyd/milestone/1) | [#119](https://github.com/surya-koritala/Glyd/issues/119) | 15 |
| v0.34.0 | [M1 Front door](https://github.com/surya-koritala/Glyd/milestone/2) | [#120](https://github.com/surya-koritala/Glyd/issues/120) | 28 |
| v0.35.0 | [M2 One runtime](https://github.com/surya-koritala/Glyd/milestone/3) | [#121](https://github.com/surya-koritala/Glyd/issues/121) | 29 |
| v0.36.0 | [M3 Models, formats and desktop](https://github.com/surya-koritala/Glyd/milestone/4) | [#122](https://github.com/surya-koritala/Glyd/issues/122) | 28 |
| v0.37.0 | [M4 Every vendor](https://github.com/surya-koritala/Glyd/milestone/5) | [#123](https://github.com/surya-koritala/Glyd/issues/123) | 12 |
| v0.38.0 | [M5 Many GPUs, many machines](https://github.com/surya-koritala/Glyd/milestone/6) | [#124](https://github.com/surya-koritala/Glyd/issues/124) | 10 |
| v1.0.0 | [M6 Stable and operable](https://github.com/surya-koritala/Glyd/milestone/7) | [#125](https://github.com/surya-koritala/Glyd/issues/125) | 7 |
| after 1.0 | [Later: datacenter scale](https://github.com/surya-koritala/Glyd/milestone/8) | [#210](https://github.com/surya-koritala/Glyd/issues/210) | 7 |

## Where it stands (v0.33.0, 2026-11-20)

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

One public repository for all of Glyd under Apache-2.0; a release built entirely from it that installs and runs without Python; docs that match the code; and the engineering baseline: security updates, lints, CI on every OS, GPU CI on real hardware.

- [x] S7 Security updates: rustls and Dependabot for every manifest ([#211](https://github.com/surya-koritala/Glyd/issues/211))
- [ ] S1 Merge the monorepo branch and retire the predecessor repositories ([#126](https://github.com/surya-koritala/Glyd/issues/126))
- [ ] S2 Self-hosted GPU runner for gpu.yml ([#127](https://github.com/surya-koritala/Glyd/issues/127))
- [ ] S3 Ephemeral cloud GPU runners for gpu-cloud.yml ([#128](https://github.com/surya-koritala/Glyd/issues/128))
- [ ] S4 Release v0.33.0 from this repository ([#129](https://github.com/surya-koritala/Glyd/issues/129))
- [ ] S8 Rust baseline: toolchain, clippy, rustfmt, cargo-deny, all crates tested in CI ([#212](https://github.com/surya-koritala/Glyd/issues/212))
- [ ] S6 Required checks and branch protection on main ([#131](https://github.com/surya-koritala/Glyd/issues/131))
- [ ] S9 Windows and macOS compile and unit-test CI ([#213](https://github.com/surya-koritala/Glyd/issues/213))
- [ ] S10 The engine on by default where it has been measured ([#214](https://github.com/surya-koritala/Glyd/issues/214))
- [ ] S11 Docs and repository identity match the code ([#215](https://github.com/surya-koritala/Glyd/issues/215))
- [ ] S12 Security policy for the inference product ([#216](https://github.com/surya-koritala/Glyd/issues/216))
- [ ] S13 Community: discussions, issue templates, triage, first issues ([#217](https://github.com/surya-koritala/Glyd/issues/217))
- [ ] S14 Governance: maintainers, decisions and trademark ([#218](https://github.com/surya-koritala/Glyd/issues/218))
- [ ] S15 Hardware access plan for CI and release gates ([#219](https://github.com/surya-koritala/Glyd/issues/219))
- [ ] S16 One kernel ABI check and no closed-source leftovers ([#220](https://github.com/surya-koritala/Glyd/issues/220))

Gate: A tag on main produces every release artifact from this repository, including the engine tarballs the installer downloads; on a clean Linux machine with an NVIDIA driver, install.sh gives a first token from the engine without Python; no open security advisory in any lockfile; Linux, Windows and macOS compile CI and GPU CI are green.

## M1 Front door · v0.34.0

[Milestone](https://github.com/surya-koritala/Glyd/milestone/2) · [Epic #120](https://github.com/surya-koritala/Glyd/issues/120)

Make today's engine the least-friction way to run a model on an NVIDIA PC: one `glyd` binary, CUDA 12 and 13 builds, Windows and aarch64, a plan that is exactly what loads, a background service, Ollama, OpenAI and Anthropic APIs that common tools reach with default settings, secure defaults, signed releases, and the logits and benchmark harnesses that make the runtime refactor safe.

- [x] R0 One levels module ([#132](https://github.com/surya-koritala/Glyd/issues/132))
- [ ] R8 glyd bench: one benchmark harness with a regression gate ([#133](https://github.com/surya-koritala/Glyd/issues/133))
- [ ] R5a Constrained decoding with llguidance ([#143](https://github.com/surya-koritala/Glyd/issues/143))
- [ ] R5b Tool-call and reasoning parsers for every family ([#144](https://github.com/surya-koritala/Glyd/issues/144))
- [ ] R7 Per-model logits tests on every merge ([#146](https://github.com/surya-koritala/Glyd/issues/146))
- [ ] P1 One binary: glyd with the engine linked ([#147](https://github.com/surya-koritala/Glyd/issues/147))
- [ ] P16 One capability matrix for fit, run and refusals ([#221](https://github.com/surya-koritala/Glyd/issues/221))
- [ ] P2a The plan is what loads, on today's paths ([#148](https://github.com/surya-koritala/Glyd/issues/148))
- [ ] P11 Secure defaults for every listener ([#222](https://github.com/surya-koritala/Glyd/issues/222))
- [ ] P3 Background service: load on request, unload when idle ([#152](https://github.com/surya-koritala/Glyd/issues/152))
- [ ] P4 Model names, manifest and pull/ls/rm ([#153](https://github.com/surya-koritala/Glyd/issues/153))
- [ ] P5a OpenAI API completeness: Responses, errors, streaming options ([#154](https://github.com/surya-koritala/Glyd/issues/154))
- [ ] P5b Anthropic Messages API: full streaming events and tool use ([#155](https://github.com/surya-koritala/Glyd/issues/155))
- [ ] P5c Ollama-compatible API ([#156](https://github.com/surya-koritala/Glyd/issues/156))
- [ ] P6 CUDA 12 and CUDA 13 builds chosen at run time ([#158](https://github.com/surya-koritala/Glyd/issues/158))
- [ ] P13 Code-signing identities for Windows and macOS ([#223](https://github.com/surya-koritala/Glyd/issues/223))
- [ ] P7a Windows x86_64 release of the engine ([#159](https://github.com/surya-koritala/Glyd/issues/159))
- [ ] P10 glyd doctor and a support bundle ([#165](https://github.com/surya-koritala/Glyd/issues/165))
- [ ] F3a GGUF reader ([#176](https://github.com/surya-koritala/Glyd/issues/176))
- [ ] C5 Starlings crates into the main workspace ([#199](https://github.com/surya-koritala/Glyd/issues/199))
- [ ] D3 Metrics, health, readiness and drain ([#204](https://github.com/surya-koritala/Glyd/issues/204))
- [ ] D6 Signed releases and attestations ([#207](https://github.com/surya-koritala/Glyd/issues/207))
- [ ] P12 Fuzz the engine's parsers ([#224](https://github.com/surya-koritala/Glyd/issues/224))
- [ ] P14 User documentation site ([#225](https://github.com/surya-koritala/Glyd/issues/225))
- [ ] P15 Model catalog: ownership and review ([#226](https://github.com/surya-koritala/Glyd/issues/226))
- [ ] P17 Tools connect with default settings ([#227](https://github.com/surya-koritala/Glyd/issues/227))
- [ ] P18 aarch64 Linux: GH200, GB200, DGX Spark and Jetson ([#228](https://github.com/surya-koritala/Glyd/issues/228))
- [ ] P19 The codec's place: crate, CLI and docs ([#229](https://github.com/surya-koritala/Glyd/issues/229))

Gate: On clean Ubuntu 24.04 and Windows 11 machines with only the NVIDIA driver (an 8 GB laptop GPU, a 16 GB desktop GPU, and one machine on the oldest supported driver): install → first token in under 2 minutes for a pinned list of 10 models the engine runs (`benchmarks/gates/m1-models.txt`, pre-packed repos present); the printed plan is what loads; Open WebUI, Continue and an OpenAI SDK client connect with default settings; JSON-schema output is valid on all 10; an Apple or AMD machine gets a one-line answer saying what to do; no Python anywhere.

## M2 One runtime · v0.35.0

[Milestone](https://github.com/surya-koritala/Glyd/milestone/3) · [Epic #121](https://github.com/surya-koritala/Glyd/issues/121)

Replace the per-family pipelines with one runtime (layer kinds, config-driven descriptions, one scheduler with paged, content-addressed KV, expert tiering for every MoE) and prove it on the standard families most people run: Llama 3, Qwen 2/2.5, Mistral, Phi-4, Granite, OLMo, SmolLM.

- [ ] R9 Runtime design: model step, layer-kind interface, backend seam ([#230](https://github.com/surya-koritala/Glyd/issues/230))
- [ ] R2 Layer-kind library with a CPU f32 reference for each kind ([#135](https://github.com/surya-koritala/Glyd/issues/135))
- [ ] R3a Config-driven model descriptions ([#136](https://github.com/surya-koritala/Glyd/issues/136))
- [ ] R10 Pack-format compatibility suite ([#231](https://github.com/surya-koritala/Glyd/issues/231))
- [ ] R3b Retire the hand-written Qwen3.5, Gemma 4 and Nemotron-H engines ([#137](https://github.com/surya-koritala/Glyd/issues/137))
- [ ] R4a Model-agnostic scheduler ([#138](https://github.com/surya-koritala/Glyd/issues/138))
- [ ] R3c Qwen3 dense on the shared executor ([#139](https://github.com/surya-koritala/Glyd/issues/139))
- [ ] R3d Condor's model (Qwen3.8-Flash-Next) on the shared executor ([#140](https://github.com/surya-koritala/Glyd/issues/140))
- [ ] R4b Paged KV for every attention kind ([#141](https://github.com/surya-koritala/Glyd/issues/141))
- [ ] R4c Content-addressed KV blocks and a radix prefix index ([#142](https://github.com/surya-koritala/Glyd/issues/142))
- [ ] P2b Exact model sizes from file headers before download ([#149](https://github.com/surya-koritala/Glyd/issues/149))
- [ ] P2c Measured rates and calibrated speed predictions ([#150](https://github.com/surya-koritala/Glyd/issues/150))
- [ ] R12 Expert tiering for every MoE, and dense layers in RAM ([#232](https://github.com/surya-koritala/Glyd/issues/232))
- [ ] P2d Visible degrade ladder with confirmation ([#151](https://github.com/surya-koritala/Glyd/issues/151))
- [ ] F1a Llama 3.x ([#166](https://github.com/surya-koritala/Glyd/issues/166))
- [ ] F1b Mistral and Ministral (dense) ([#167](https://github.com/surya-koritala/Glyd/issues/167))
- [ ] F1d Qwen 2/2.5 and Qwen3 MoE ([#169](https://github.com/surya-koritala/Glyd/issues/169))
- [ ] F1e Phi-4 family ([#170](https://github.com/surya-koritala/Glyd/issues/170))
- [ ] F1h Granite, OLMo, SmolLM and other standard dense models ([#173](https://github.com/surya-koritala/Glyd/issues/173))
- [ ] R11 Async HTTP front end and a request-level engine interface ([#233](https://github.com/surya-koritala/Glyd/issues/233))
- [ ] R13 Retire or freeze the GLM development engine ([#234](https://github.com/surya-koritala/Glyd/issues/234))
- [ ] R14 A quality bar for lossy levels ([#235](https://github.com/surya-koritala/Glyd/issues/235))
- [ ] R15 A sound device API and typed kernel launches ([#236](https://github.com/surya-koritala/Glyd/issues/236))
- [ ] R16 A runtime context and typed errors ([#237](https://github.com/surya-koritala/Glyd/issues/237))
- [ ] R17 One kernel build: fold the GPU library into the engine ([#238](https://github.com/surya-koritala/Glyd/issues/238))
- [ ] P2f Every path loads by the plan; per-path planners deleted ([#239](https://github.com/surya-koritala/Glyd/issues/239))
- [ ] P4b Short names pick the size and level for this machine ([#240](https://github.com/surya-koritala/Glyd/issues/240))
- [ ] P5e Anthropic prompt caching on content-addressed blocks ([#241](https://github.com/surya-koritala/Glyd/issues/241))
- [ ] P20 Consumer benchmarks published every release ([#242](https://github.com/surya-koritala/Glyd/issues/242))

Gate: A pinned per-GPU list (the smallest checkpoint of each family that the plan fits) serves 64 concurrent requests through one scheduler on an RTX 4080 SUPER and an L4 with logits matching the CPU reference; the large families (Kimi-K3, DeepSeek-V4, GLM-5.3) keep single-user parity on their reference hardware; no `glyd bench` regression beyond the measured noise band; every published pack since v0.29 still loads.

## M3 Models, formats and desktop · v0.36.0

[Milestone](https://github.com/surya-koritala/Glyd/milestone/4) · [Epic #122](https://github.com/surya-koritala/Glyd/issues/122)

Run what people already have and make Glyd a desktop citizen: the remaining model families, vision, embeddings, GGUF quant kernels, quantized safetensors releases, consumer-Blackwell FP4, LoRA, long context, speculative decoding, a pack pipeline with license and quality rules, installers on every channel, macOS, self-update, and consumer benchmarks published against Ollama, llama.cpp and LM Studio.

- [ ] R6 Speculative decoding for any model ([#145](https://github.com/surya-koritala/Glyd/issues/145))
- [ ] P7c Installers: install.sh, winget, Homebrew, Docker ([#161](https://github.com/surya-koritala/Glyd/issues/161))
- [ ] P7d macOS arm64 build of glyd ([#162](https://github.com/surya-koritala/Glyd/issues/162))
- [ ] P8 Fold Condor into glyd run ([#163](https://github.com/surya-koritala/Glyd/issues/163))
- [ ] P9 No Python on any install path ([#164](https://github.com/surya-koritala/Glyd/issues/164))
- [ ] F1c Gemma 2, Gemma 3 and the remaining Gemma 4 models ([#168](https://github.com/surya-koritala/Glyd/issues/168))
- [ ] F3c Quantized safetensors releases as released ([#178](https://github.com/surya-koritala/Glyd/issues/178))
- [ ] F1f gpt-oss ([#171](https://github.com/surya-koritala/Glyd/issues/171))
- [ ] F1g DeepSeek R1 distills ([#172](https://github.com/surya-koritala/Glyd/issues/172))
- [ ] F2a Vision for every VLM family ([#174](https://github.com/surya-koritala/Glyd/issues/174))
- [ ] F2b Embedding and reranker models ([#175](https://github.com/surya-koritala/Glyd/issues/175))
- [ ] F3b GGUF quantization kernels ([#177](https://github.com/surya-koritala/Glyd/issues/177))
- [ ] F4 LoRA adapters, batched ([#179](https://github.com/surya-koritala/Glyd/issues/179))
- [ ] F8 Model license and gating policy ([#243](https://github.com/surya-koritala/Glyd/issues/243))
- [ ] F5 Automated pack pipeline for new releases ([#180](https://github.com/surya-koritala/Glyd/issues/180))
- [ ] F6 Long context past 40,960 tokens ([#181](https://github.com/surya-koritala/Glyd/issues/181))
- [ ] F7 Retire the vLLM plugin path ([#182](https://github.com/surya-koritala/Glyd/issues/182))
- [ ] F1i Llama 4 ([#244](https://github.com/surya-koritala/Glyd/issues/244))
- [ ] F1j Mixtral ([#245](https://github.com/surya-koritala/Glyd/issues/245))
- [ ] F1k DeepSeek V3 and R1 (full) ([#246](https://github.com/surya-koritala/Glyd/issues/246))
- [ ] P3b Adapt to VRAM pressure at run time ([#247](https://github.com/surya-koritala/Glyd/issues/247))
- [ ] F9 Day-0 model support process ([#248](https://github.com/surya-koritala/Glyd/issues/248))
- [ ] F10 Rust library API and C ABI for embedding ([#249](https://github.com/surya-koritala/Glyd/issues/249))
- [ ] P21 glyd update ([#250](https://github.com/surya-koritala/Glyd/issues/250))
- [ ] P22 Scripts to cargo xtask; Python references to Rust ([#251](https://github.com/surya-koritala/Glyd/issues/251))
- [ ] P23 Corporate networks: proxies, custom CAs, offline installs ([#252](https://github.com/surya-koritala/Glyd/issues/252))
- [ ] P24 Opt-in crash and compatibility reports ([#253](https://github.com/surya-koritala/Glyd/issues/253))
- [ ] V7a Consumer Blackwell FP4 and FP8 ([#254](https://github.com/surya-koritala/Glyd/issues/254))

Gate: A dated snapshot of the top 50 text-generation models on the Hugging Face Hub (exclusions listed) runs; the 10 most-pulled local models run, including from GGUF; every installer channel passes its install test; consumer benchmarks are published per VRAM tier.

## M4 Every vendor · v0.37.0

[Milestone](https://github.com/surya-koritala/Glyd/milestone/5) · [Epic #123](https://github.com/surya-koritala/Glyd/issues/123)

The same `glyd run` on AMD (RDNA 3/4, Strix Halo, MI300X), Apple Silicon, CPU-only machines and older NVIDIA cards, through a backend seam designed against Metal and CPU, vendor-native kernels, and proven kernel libraries where they are faster.

- [ ] S5 macOS GPU CI runner for the Metal backend ([#130](https://github.com/surya-koritala/Glyd/issues/130))
- [ ] R1 Device layer for CUDA and HIP, and the backend seam at the layer-kind level ([#134](https://github.com/surya-koritala/Glyd/issues/134))
- [ ] P7b Small core that downloads its backend ([#160](https://github.com/surya-koritala/Glyd/issues/160))
- [ ] V0 Kernel portability layer: warp size, lane masks, tensor-core fragments ([#183](https://github.com/surya-koritala/Glyd/issues/183))
- [ ] V1 AMD RDNA 3/4 and Strix Halo: a beta first, then speed ([#184](https://github.com/surya-koritala/Glyd/issues/184))
- [ ] V2 AMD CDNA 3/4 (MI300X, MI325X, MI355X) ([#185](https://github.com/surya-koritala/Glyd/issues/185))
- [ ] V3a Metal backend through mlx-c ([#186](https://github.com/surya-koritala/Glyd/issues/186))
- [ ] V3b Glyd formats as Metal kernels ([#187](https://github.com/surya-koritala/Glyd/issues/187))
- [ ] V3c Unified-memory planning on every vendor ([#188](https://github.com/surya-koritala/Glyd/issues/188))
- [ ] V4 CPU backend for whole models ([#189](https://github.com/surya-koritala/Glyd/issues/189))
- [ ] V5 NVIDIA Turing (sm_75): fp16 paths ([#190](https://github.com/surya-koritala/Glyd/issues/190))
- [ ] V8 Kernel libraries where they are faster ([#255](https://github.com/surya-koritala/Glyd/issues/255))

Gate: Per device, the subset of the M3 model set that the plan fits passes with KL within the per-level threshold against the CUDA run: NVIDIA T4, L4, RTX 4080 SUPER, A100, H100, DGX Spark; AMD RX 7900 XTX, RX 9070 XT, Strix Halo, MI300X; Apple M1 and M4; CPU x86_64 and arm64.

## M5 Many GPUs, many machines · v0.38.0

[Milestone](https://github.com/surya-koritala/Glyd/milestone/6) · [Epic #124](https://github.com/surya-koritala/Glyd/issues/124)

One planner from one GPU to several machines: collectives, tensor and expert parallelism in a node, an overlapped pipeline, GeForce boxes without peer access, `glyd cluster` for N machines, and a drop-in OpenAI server for one 8-GPU node.

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

Gate: Two consumer machines run a model neither holds alone at the speed `glyd fit` predicted (±10%); a 600B-class MoE runs on one 8× H100 node with experts in host RAM; tensor parallel on 2 GPUs is at least 1.7× one GPU on a model that fits one GPU; one 8-GPU node serves through the OpenAI API with data-parallel replicas and standard metrics.

## M6 Stable and operable · v1.0.0

[Milestone](https://github.com/surya-koritala/Glyd/milestone/7) · [Epic #125](https://github.com/surya-koritala/Glyd/issues/125)

1.0 commits to stable CLI, HTTP APIs, library API and pack formats, with operations (Helm, LeaderWorkerSet, dashboards), router integration (llm-d, Gateway API inference extension, Dynamo, SMG), a deterministic mode, supply-chain security, and a published comparison with vLLM and SGLang.

- [ ] D2 Integrate with KV-aware routers: llm-d, Gateway API, SMG ([#203](https://github.com/surya-koritala/Glyd/issues/203))
- [ ] D5 Deterministic, batch-invariant mode ([#206](https://github.com/surya-koritala/Glyd/issues/206))
- [ ] D7 Published comparison with vLLM and SGLang, per quality tier ([#208](https://github.com/surya-koritala/Glyd/issues/208))
- [ ] D8 1.0 stability: CLI, API and pack formats ([#209](https://github.com/surya-koritala/Glyd/issues/209))
- [ ] D3b Helm, LeaderWorkerSet, images and dashboards ([#258](https://github.com/surya-koritala/Glyd/issues/258))
- [ ] D6b SBOM, provenance level and reproducible builds ([#259](https://github.com/surya-koritala/Glyd/issues/259))
- [ ] D9 Token-level engine API for Dynamo and SMG ([#260](https://github.com/surya-koritala/Glyd/issues/260))

Gate: A documented stable surface with CI compatibility checks; llm-d routes to Glyd pods by prefix-cache hits; a Helm install passes readiness on a GPU cluster; SBOM and provenance on every artifact; a comparison with vLLM and SGLang per quality tier (lossless against bf16, lossy levels against FP8 and NVFP4) on H100 and B200, published with configs and logs whatever the result.

## Later: datacenter scale (after 1.0)

[Milestone](https://github.com/surya-koritala/Glyd/milestone/8) · [Epic #210](https://github.com/surya-koritala/Glyd/issues/210)

Work deferred until the single-node server is adopted: prefill/decode disaggregation, RDMA KV transport, wide expert parallelism and NVL72, RL weight updates, Blackwell tcgen05 kernels, a Vulkan fallback, an MCP wrapper.

- [ ] P5d MCP server for any model ([#157](https://github.com/surya-koritala/Glyd/issues/157))
- [ ] V6 A fallback for other GPUs ([#191](https://github.com/surya-koritala/Glyd/issues/191))
- [ ] V7 Datacenter Blackwell: tcgen05 ([#192](https://github.com/surya-koritala/Glyd/issues/192))
- [ ] D1 Prefill/decode disaggregation with compressed KV transfer ([#202](https://github.com/surya-koritala/Glyd/issues/202))
- [ ] C4b NIXL transport for RDMA ([#261](https://github.com/surya-koritala/Glyd/issues/261))
- [ ] D4 Wide expert parallelism across nodes and multi-node NVLink ([#205](https://github.com/surya-koritala/Glyd/issues/205))
- [ ] D5b Weight updates for RL rollouts ([#262](https://github.com/surya-koritala/Glyd/issues/262))

Revisited after 1.0, by demand.

## Always

- `glyd bench` (Rust) replaces the Python benchmarks; results in `benchmarks/`.
- Python is ported or removed as each milestone replaces it (`glyd-gpu-py/`, `gpu/*.py`, `glyd-engine/tools/*.py`,
  `condor/bench`, `scripts/*.py` → `cargo xtask`).
