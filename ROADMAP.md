# Roadmap

Milestones in execution order; each ships as one release. Tracks within a milestone run in parallel. Each milestone ends on a gate measured on real hardware, with the result committed under `benchmarks/`. Design: [docs/architecture.md](docs/architecture.md) · Runtime RFC: [docs/design/runtime.md](docs/design/runtime.md) · Hardware budget: [docs/hardware-plan.md](docs/hardware-plan.md). Every item is a GitHub issue with tasks, acceptance criteria, test hardware and dependencies, under its milestone's epic; the epics are tracked in [#118](https://github.com/surya-koritala/Glyd/issues/118). Checkboxes match the GitHub issues' live state.

| Release | Milestone | Epic | Issues | Focus |
| :--- | :--- | :--- | ---: | :--- |
| v0.33.0 | [M0 Open-source baseline](https://github.com/surya-koritala/Glyd/milestone/1) | [#119](https://github.com/surya-koritala/Glyd/issues/119) | 14 | Apache-2.0 monorepo, no-Python release, CI & GPU CI |
| v0.34.0 | [M1 Front door](https://github.com/surya-koritala/Glyd/milestone/2) | [#120](https://github.com/surya-koritala/Glyd/issues/120) | 27 | Single binary, Linux+Windows, OpenAI/Anthropic/Ollama APIs, `fit == run`, `R7`/`R8` gates |
| v0.35.0 | [M2 One runtime](https://github.com/surya-koritala/Glyd/milestone/3) | [#121](https://github.com/surya-koritala/Glyd/issues/121) | 37 | Config-driven runtime, 64-user batching, hybrid paged/12-bit KV, CUDA seam (`R1a`), NVIDIA kernel libs (`V8a`) |
| v0.36.0 | [M3 Core models, formats and desktop](https://github.com/surya-koritala/Glyd/milestone/4) | [#122](https://github.com/surya-koritala/Glyd/issues/122) | 52 | **Consumer launch:** Top-50 text models, canonical GGUF, FP8/FP4, long context, Condor in `glyd run`, pre-packed top-15 hub & coding-agent kit |
| v0.37.0 | [M4 Many GPUs, many machines](https://github.com/surya-koritala/Glyd/milestone/6) | [#124](https://github.com/surya-koritala/Glyd/issues/124) | 19 | **Datacenter proof (NVIDIA):** NCCL/TP/EP/PP, SLO-aware Planner v2, `glyd cluster`, 8× H100 server, compressed streaming load, sleep-to-RAM |
| v0.38.0 | [M5a Apple Silicon, CPU and desktop extras](https://github.com/surya-koritala/Glyd/milestone/5) | [#123](https://github.com/surya-koritala/Glyd/issues/123) | 13 | macOS/`mlx-c`/Metal kernels, whole-model CPU, Vision, Embeddings, Batched LoRA, general Spec Decode |
| v0.39.0 | [M5b AMD ROCm and older NVIDIA](https://github.com/surya-koritala/Glyd/milestone/9) | [#445](https://github.com/surya-koritala/Glyd/issues/445) | 8 | HIP device layer (`R1b`), warp/fragment portability (`V0`), RDNA 3/4, Strix Halo, MI300X, AMD libs (`V8b`), Turing (`V5`), dynamic backend bundles |
| v1.0.0 | [M6 Stable and operable](https://github.com/surya-koritala/Glyd/milestone/7) | [#125](https://github.com/surya-koritala/Glyd/issues/125) | 18 | Stable CLI/API/packs, Helm/LWS, `llm-d`/GAIE/Dynamo/SMG, deterministic mode, SBOM, published vLLM/SGLang comparison |
| after 1.0 | [Later: datacenter scale](https://github.com/surya-koritala/Glyd/milestone/8) | [#210](https://github.com/surya-koritala/Glyd/issues/210) | 13 | P/D disaggregation, NIXL RDMA, wide EP/NVL72, custom compressed `tcgen05`, RL weight updates, Vulkan, MCP |

## Why this order (guidance for agents and contributors)

Glyd's goal is one unified, zero-Python Rust inference engine from an 8 GB laptop GPU to an 8-GPU datacenter node and multi-node clusters. The milestones are ordered to compound adoption and prove the architecture without hitting solo-maintainer hardware bottlenecks:

1. **Close M0 + M1 before expanding scope.** Every open M0/M1 item directly protects first-run trust (`fit == run`, Windows + Linux single binary, `R7` logits and `R8` bench regression gates). Do not open new M3/M4 feature tracks while M0/M1 blockers remain open.
2. **Pull CUDA `R1a` (#134) and NVIDIA `V8a` (#255) into M2.** A unified 8 GB → datacenter engine depends on a **split execution bottom**: own custom fused kernels for compressed weights (`mma`/`mma12`, lossy tiers, Hummingbird), 12-bit lossless KV, and async MoE streaming; delegate compute-bound dense prefill, long-context paged attention, and native FP8/NVFP4/`sm_100` GEMMs to proven Apache-2.0/MIT/BSD libraries (CUTLASS, FlashInfer, cuBLASLt, Marlin). Pulling `R1a` and `V8a` into M2 prevents writing throwaway kernels during M2/M3 and ensures M6's B200 comparison does not pit `mma.sync` against `tcgen05`. Meanwhile, `#230`'s MLX spike validates the cross-vendor `Backend` seam before any CUDA assumptions harden.
3. **Keep M3 focused on the consumer adoption launch.** Core text models, canonical GGUF quants, quantized safetensors, RTX 50 FP4, long context (>40K for coding agents), Condor folded into `glyd run`, installers, curated pre-packed top-15 repos (`P25`), and the `plan.rs` web calculator (`P26`) ship together as `v0.36.0`. Secondary modalities/features (Vision, Embeddings, Batched LoRA, general Spec Decode, macOS builds) live in M5a so M3 ships cleanly and fast.
4. **Ship M4 (NVIDIA multi-GPU/cluster) ahead of M5a (Apple/CPU) and M5b (AMD/Turing).** 95%+ of rentable CI and datacenter GPUs are NVIDIA, whereas 4 target AMD GPUs have zero rentals on Lambda, AWS or Vast.ai ([docs/hardware-plan.md](docs/hardware-plan.md)). Shipping M4 right after M3 proves the full 8 GB → 8× H100 vision months earlier instead of stalling on unrentable hardware.
5. **Split Apple/CPU (`M5a`) from AMD/Turing (`M5b`).** Most developers test on Apple Silicon laptops and deploy to Linux/NVIDIA servers, so `M4 + M5a` covers ~90%+ of real users. `M5b` can advance in parallel as AMD hardware access or contributors arrive, without blocking `v1.0.0`.

### Rules for agents working on this repository

- **Respect milestone order.** Pick open blockers in the active milestone before starting later milestones unless explicitly instructed.
- **Never add a pipeline of a family's own.** The hand-written ones (`model.rs`, `model35.rs`, `model_gemma4.rs`, `model_nh.rs`, `model_glm.rs`, `model_qwen4x.rs`) are deleted: every family runs on the executor (`glyd-engine/src/dynamic/`) from a description of layer kinds; all new runtime capabilities belong in `glyd-engine/src/runtime/` and the executor (`docs/design/runtime.md`).
- **Delete legacy code immediately after parity.** Once a family passes `R7` logits + `R8` bench gates on the shared runtime, delete its hand-written pipeline in the very next PR.
- **Prevent `Format × Kind × Backend` explosion.** Weight formats plug *only* into linear / grouped-MoE product ops; KV codecs plug *only* into page read/write/decode. Never duplicate attention/SSM/MLA mixers per format.
- **Real hardware gates only.** Never check off a `needs-gpu` acceptance criterion from a compile-only or mocked test; commit measured results under `benchmarks/`.
- **Keep docs and GitHub in sync.** When closing an issue, update the epic checklist and this file in the same PR.

## Where it stands (v0.34.2, 2026-11-22)

Released, on main and planned are listed apart.

**Released: v0.33.0 / v0.33.1** (2026-11-20, Apache-2.0; the first releases built only from this repository)
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
- The engine runs the plain dense kind, the Qwen3.5 layout (3.5, 3.6, 3.8) and Gemma 4 at penguin, kestrel, swift and wren,
  Nemotron-H at those and hummingbird, and Kimi-K3, GLM-5.3-Flash, DeepSeek-V4 and Muse Spark at penguin and hummingbird, all on one executor. Qwen3.8-Flash-Next (Condor's
  model) runs there too: `glyd run` takes it at penguin and its codes packs at hummingbird; Condor's scripts serve the packs
  with `qwenserve`. The list is `glyd-engine/src/support.rs`.
- With the Python tool, `glyd run` hands a plain dense model at penguin to the engine only on GPUs listed in `GPUS` (`glyd-engine/src/machine.rs`):
  the RTX 4080 SUPER is the one so far (S10), the others use vLLM unless `GLYD_ENGINE=1`; every other family and level is the engine's.
- One scheduler batches every family (up to 64 users); a model split across GPUs or with a draft model runs one sequence at a time.
- MoE experts on GPU/RAM/NVMe with CPU compute for misses; layer split over the GPUs of one machine for the families
  `split_levels` in `glyd-engine/src/support.rs` names; two-machine pipeline (Starlings, beta). `glyd-engine fit` plans before
  download, and `run` and `serve` load by the plan it prints (`P2a`, #148).
- NVIDIA Ampere to Blackwell, Linux x86_64, driver 535+ (CUDA 12/13 dual kernel builds landed in #407). AMD RDNA 3/4 as
  development builds (`GLYD_HIP=1`). GPU CI runs on one self-hosted machine (`sm_89`) with a small model; Windows and macOS
  CI compile and unit-test the engine, codec, store and CLI.

**Not there yet**: completed M1 gate verification (Windows logon, oldest driver, code-signing certs), unified M2 runtime across all families, Apple Silicon, AMD releases, CPU-only whole models, multi-GPU collectives.

---

## M0 Open-source baseline · v0.33.0

[Milestone](https://github.com/surya-koritala/Glyd/milestone/1) · [Epic #119](https://github.com/surya-koritala/Glyd/issues/119)

One public repository for all of Glyd under Apache-2.0; a release built entirely from it that installs and runs without Python; docs that match the code; and the engineering baseline: security updates, lints, CI on every OS, GPU CI on real hardware.

**Why here / Agent rules:** Eliminates the closed-binary and licensing adoption blockers and establishes CI/security hygiene. Finish and close the remaining open M0 items rather than leaving them drifting while later milestones proceed.

- [x] S7 Security updates: rustls and Dependabot for every manifest ([#211](https://github.com/surya-koritala/Glyd/issues/211))
- [x] S1 Merge the monorepo branch and retire the predecessor repositories ([#126](https://github.com/surya-koritala/Glyd/issues/126))
- [x] S2 Self-hosted GPU runner for gpu.yml ([#127](https://github.com/surya-koritala/Glyd/issues/127))
- [x] S4 Release v0.33.0 from this repository ([#129](https://github.com/surya-koritala/Glyd/issues/129))
- [x] S8 Rust baseline: toolchain, clippy, rustfmt, cargo-deny, all crates tested in CI ([#212](https://github.com/surya-koritala/Glyd/issues/212))
- [x] S6 Required checks and branch protection on main ([#131](https://github.com/surya-koritala/Glyd/issues/131))
- [x] S9 Windows and macOS compile and unit-test CI ([#213](https://github.com/surya-koritala/Glyd/issues/213))
- [x] S10 The engine on by default where it has been measured ([#214](https://github.com/surya-koritala/Glyd/issues/214))
- [x] S11 Docs and repository identity match the code ([#215](https://github.com/surya-koritala/Glyd/issues/215))
- [x] S12 Security policy for the inference product ([#216](https://github.com/surya-koritala/Glyd/issues/216))
- [x] S13 Community: discussions, issue templates, triage, first issues ([#217](https://github.com/surya-koritala/Glyd/issues/217))
- [x] S14 Governance: maintainers, decisions and trademark ([#218](https://github.com/surya-koritala/Glyd/issues/218))
- [x] S15 Hardware access plan for CI and release gates ([#219](https://github.com/surya-koritala/Glyd/issues/219))
- [x] S16 One kernel ABI check and no closed-source leftovers ([#220](https://github.com/surya-koritala/Glyd/issues/220))

Re-scoped on 2026-11-21 (owner's decision): S3 (#128) moved to M4; recruiting maintainers and a required review of `.github/` changes moved to M6 (#458); S4's PyPI criterion is met by `glyd-gpu` 0.33.1 and crates.io moved to P7c (#161); S6 drops the merge queue (it needs an organization account); S9 is met by Windows and macOS CI on main, nightly and system-specific pull requests, with an issue opened when main fails.

Gate: A tag on main produces every release artifact from this repository, including the engine tarballs the installer downloads; on a clean Linux machine with an NVIDIA driver, `install.sh` gives a first token from the engine without Python; no open security advisory in any lockfile; Linux, Windows and macOS compile CI and GPU CI are green.

---

## M1 Front door · v0.34.0

[Milestone](https://github.com/surya-koritala/Glyd/milestone/2) · [Epic #120](https://github.com/surya-koritala/Glyd/issues/120)

Make today's engine the least-friction way to run a model on an NVIDIA PC: one `glyd` binary, CUDA 12 and 13 builds, Windows and aarch64, a plan that is exactly what loads, a background service, Ollama, OpenAI and Anthropic APIs that common tools reach with default settings, secure defaults, signed releases, and the logits and benchmark harnesses that make the runtime refactor safe.

**Why here / Agent rules:** Adoption wedge #1 and the numerical/speed safety net (`R7`, `R8`) required before M2 deletes legacy pipelines. Prioritize closing remaining M1 gate blockers and deferred hardware checks before pulling M3/M4 features.

- [x] R0 One levels module ([#132](https://github.com/surya-koritala/Glyd/issues/132))
- [x] R8 glyd bench: one benchmark harness with a regression gate ([#133](https://github.com/surya-koritala/Glyd/issues/133))
- [x] R5a Constrained decoding with llguidance ([#143](https://github.com/surya-koritala/Glyd/issues/143))
- [x] R5b Tool-call and reasoning parsers for every family ([#144](https://github.com/surya-koritala/Glyd/issues/144))
- [x] R7 Per-model logits tests on every merge ([#146](https://github.com/surya-koritala/Glyd/issues/146))
- [x] P1 One binary: glyd with the engine linked ([#147](https://github.com/surya-koritala/Glyd/issues/147))
- [x] P16 One capability matrix for fit, run and refusals ([#221](https://github.com/surya-koritala/Glyd/issues/221))
- [x] P2a The plan is what loads, on today's paths ([#148](https://github.com/surya-koritala/Glyd/issues/148))
- [x] P11 Secure defaults for every listener ([#222](https://github.com/surya-koritala/Glyd/issues/222))
- [x] P3 Background service: load on request, unload when idle ([#152](https://github.com/surya-koritala/Glyd/issues/152))
- [x] P4 Model names, manifest and pull/ls/rm ([#153](https://github.com/surya-koritala/Glyd/issues/153))
- [x] P5a OpenAI API completeness: Responses, errors, streaming options ([#154](https://github.com/surya-koritala/Glyd/issues/154))
- [x] P5b Anthropic Messages API: full streaming events and tool use ([#155](https://github.com/surya-koritala/Glyd/issues/155))
- [x] P5c Ollama-compatible API ([#156](https://github.com/surya-koritala/Glyd/issues/156))
- [x] P6 CUDA 12 and CUDA 13 builds chosen at run time ([#158](https://github.com/surya-koritala/Glyd/issues/158))
- [x] P7a Windows x86_64 release of the engine ([#159](https://github.com/surya-koritala/Glyd/issues/159))
- [x] P10 glyd doctor and a support bundle ([#165](https://github.com/surya-koritala/Glyd/issues/165))
- [x] F3a GGUF reader ([#176](https://github.com/surya-koritala/Glyd/issues/176))
- [x] C5 Starlings crates into the main workspace ([#199](https://github.com/surya-koritala/Glyd/issues/199))
- [x] D3 Metrics, health, readiness and drain ([#204](https://github.com/surya-koritala/Glyd/issues/204))
- [x] D6 Signed releases and attestations ([#207](https://github.com/surya-koritala/Glyd/issues/207))
- [x] P12 Fuzz the engine's parsers ([#224](https://github.com/surya-koritala/Glyd/issues/224))
- [x] P14 User documentation site ([#225](https://github.com/surya-koritala/Glyd/issues/225))
- [x] P15 Model catalog: ownership and review ([#226](https://github.com/surya-koritala/Glyd/issues/226))
- [x] P17 Tools connect with default settings ([#227](https://github.com/surya-koritala/Glyd/issues/227))
- [x] P18 aarch64 Linux: GH200, GB200, DGX Spark and Jetson ([#228](https://github.com/surya-koritala/Glyd/issues/228))
- [x] P19 The codec's place: crate, CLI and docs ([#229](https://github.com/surya-koritala/Glyd/issues/229))

Re-scoped on 2026-11-21 (owner's decision): P13 (#223) moved to M3, so v0.34.0 ships unsigned with a note on the release page; P18 keeps the end of the blanket aarch64 refusal and the Arm CPU expert kernels, and its GH200 and DGX Spark runs moved to M5a (#459).

Gate: On clean Ubuntu 24.04 and Windows 11 machines with only the NVIDIA driver (an 8 GB laptop GPU, a 16 GB desktop GPU, and one machine on the oldest supported driver): install → first token in under 2 minutes for the pinned models the engine runs (`benchmarks/gates/m1-models.txt`, pre-packed repos present); the printed plan is what loads; Open WebUI, Continue and an OpenAI SDK client connect with default settings; JSON-schema output is valid on every pinned model that runs; an Apple or AMD machine gets a one-line answer saying what to do; no Python anywhere.

---

## M2 One runtime · v0.35.0

[Milestone](https://github.com/surya-koritala/Glyd/milestone/3) · [Epic #121](https://github.com/surya-koritala/Glyd/issues/121)

Replace the per-family pipelines with one config-driven runtime (`docs/design/runtime.md`: layer kinds, config-driven descriptions, one async overlap scheduler with paged, content-addressed KV and 12-bit KV for recent head dims/MLA, bucketed/piecewise CUDA graphs, universal MoE tiering, CUDA device seam `R1a`, and NVIDIA kernel library delegation `V8a`), and prove it on both recent architectures and the standard families most people run: Llama 3, Qwen 2/2.5/3 MoE, Mistral, Phi-4, Granite, OLMo, SmolLM.

**Why here / Agent rules:**
- **CUDA `R1a` (#134) and NVIDIA `V8a` (#255) live here:** `docs/design/runtime.md` Step 2 needs the CUDA `Device` layer immediately, and `V8a` establishes the split execution bottom (custom kernels for compressed weights/KV/MoE streaming; CUTLASS/FlashInfer/cuBLASLt for compute-bound dense prefill, long-context paged attention, and native FP8/NVFP4/`sm_100` GEMMs). `#230`'s MLX spike validates the cross-vendor seam early.
- **Delete legacy pipelines in the PR immediately after parity** (follow `docs/design/runtime.md` §7: recent families first, legacy last for deletion).
- **Track `1× H100` single-token occupancy in `P20` (#242)** so the 57% A100/H100 occupancy gap (`docs/ceiling.md`) is solved before M4.

- [x] R9 Runtime design: model step, layer-kind interface, backend seam — incl. CUDA/Ref/MLX spikes ([#230](https://github.com/surya-koritala/Glyd/issues/230))
- [x] R15 A sound device API and typed kernel launches ([#236](https://github.com/surya-koritala/Glyd/issues/236))
- [x] R16 A runtime context and typed errors ([#237](https://github.com/surya-koritala/Glyd/issues/237))
- [ ] R1a CUDA device layer and backend seam at the layer-kind level ([#134](https://github.com/surya-koritala/Glyd/issues/134))
- [ ] R17 One kernel build: fold the GPU library into the engine ([#238](https://github.com/surya-koritala/Glyd/issues/238))
- [ ] V8a NVIDIA kernel libraries where they are faster: CUTLASS, FlashInfer, cuBLASLt ([#255](https://github.com/surya-koritala/Glyd/issues/255))
- [ ] R2 Layer-kind library with a CPU f32 reference for each kind ([#135](https://github.com/surya-koritala/Glyd/issues/135))
- [ ] R3a Config-driven model descriptions ([#136](https://github.com/surya-koritala/Glyd/issues/136))
- [ ] R10 Pack-format compatibility suite ([#231](https://github.com/surya-koritala/Glyd/issues/231))
- [ ] R4a Model-agnostic scheduler ([#138](https://github.com/surya-koritala/Glyd/issues/138))
- [ ] R4b Paged KV for every attention kind ([#141](https://github.com/surya-koritala/Glyd/issues/141))
- [ ] R4f Lossless KV for head dims 256 and 512 and MLA latents ([#332](https://github.com/surya-koritala/Glyd/issues/332))
- [x] R4c Content-addressed KV blocks and a radix prefix index ([#142](https://github.com/surya-koritala/Glyd/issues/142))
- [ ] R4d Preemption by swapping compressed KV pages and state slots to host ([#328](https://github.com/surya-koritala/Glyd/issues/328))
- [ ] R4e Graphs for batched and mixed steps on the shared executor ([#329](https://github.com/surya-koritala/Glyd/issues/329))
- [x] R11 Async HTTP front end and a request-level engine interface ([#233](https://github.com/surya-koritala/Glyd/issues/233))
- [x] P5e Anthropic prompt caching on content-addressed blocks ([#241](https://github.com/surya-koritala/Glyd/issues/241))
- [ ] R12 Expert tiering for every MoE, and dense layers in RAM ([#232](https://github.com/surya-koritala/Glyd/issues/232))
- [ ] R12b Expert store at many rows: one read per missed expert a step, per-layer statistics ([#327](https://github.com/surya-koritala/Glyd/issues/327))
- [x] P2b Exact model sizes from file headers before download ([#149](https://github.com/surya-koritala/Glyd/issues/149))
- [ ] P2c Measured rates and calibrated speed predictions ([#150](https://github.com/surya-koritala/Glyd/issues/150))
- [x] P2d Visible degrade ladder with confirmation ([#151](https://github.com/surya-koritala/Glyd/issues/151))
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
- [x] R3b Retire the hand-written Qwen3.5, Gemma 4 and Nemotron-H engines ([#137](https://github.com/surya-koritala/Glyd/issues/137))
- [ ] R3c Qwen3 dense on the shared executor ([#139](https://github.com/surya-koritala/Glyd/issues/139))
- [ ] R13 Retire or freeze the GLM development engine ([#234](https://github.com/surya-koritala/Glyd/issues/234))
- [ ] R14 A quality bar for lossy levels ([#235](https://github.com/surya-koritala/Glyd/issues/235))
- [ ] P20 Consumer benchmarks published every release — incl. 1× H100 occupancy canary ([#242](https://github.com/surya-koritala/Glyd/issues/242))

Gate: A pinned per-GPU list (the smallest checkpoint of each family that the plan fits) serves 64 concurrent requests through one scheduler on an RTX 4080 SUPER and an L4 with logits matching the CPU reference; the large families (Kimi-K3, DeepSeek-V4, GLM-5.3) keep single-user parity on their reference hardware; no `glyd bench` regression beyond the measured noise band; every published pack since v0.29 still loads.

---

## M3 Core models, formats and desktop · v0.36.0

[Milestone](https://github.com/surya-koritala/Glyd/milestone/4) · [Epic #122](https://github.com/surya-koritala/Glyd/issues/122)

Flagship consumer/prosumer NVIDIA launch: remaining core text model families, canonical GGUF quant kernels, quantized safetensors releases, consumer-Blackwell FP4/FP8, universal Hummingbird, long context (>40K), Condor folded into `glyd run`, live VRAM pressure adaptation, pack pipeline with license/quality rules, native installers on every channel, retire vLLM, curated pre-packed top-15 hub & local coding-agent launch kit (`P25`), interactive `plan.rs` web GPU fit calculator (`P26`), and consumer benchmarks published against Ollama, llama.cpp and LM Studio.

**Why here / Agent rules:** Narrowed from the original 30-issue M3 (Vision, Embeddings, Batched LoRA, general Spec Decode, and macOS builds moved to M5a) so Glyd ships a killer consumer + local coding-agent release fast. For GGUF (`F3b`, #177), prioritize native fast kernels for canonical quants (`Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0`, `Q4_0`, `MXFP4`) — do not block M3 on ~25 exotic IQ/K variants. Reserve a 1-week real-machine QA soak across 8/12/16/24/32 GB cards before launch.

- [ ] F3b GGUF quantization kernels — canonical quants first ([#177](https://github.com/surya-koritala/Glyd/issues/177))
- [ ] F3c Quantized safetensors releases as released ([#178](https://github.com/surya-koritala/Glyd/issues/178))
- [ ] V7a Consumer Blackwell FP4 and FP8 ([#254](https://github.com/surya-koritala/Glyd/issues/254))
- [ ] Hummingbird for the recent MoE families, calibrated in Rust ([#429](https://github.com/surya-koritala/Glyd/issues/429))
- [ ] F6 Long context past 40,960 tokens ([#181](https://github.com/surya-koritala/Glyd/issues/181))
- [ ] F1c The remaining Gemma 4 sizes ([#168](https://github.com/surya-koritala/Glyd/issues/168))
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
- [ ] K1 Config maps for recent families whose layer kinds all exist ([#549](https://github.com/surya-koritala/Glyd/issues/549))
- [ ] K2 MLA options: window, sinks, position-scaled softmax, a KV source layer ([#550](https://github.com/surya-koritala/Glyd/issues/550))
- [ ] K3 Short-convolution kind ([#551](https://github.com/surya-koritala/Glyd/issues/551))
- [ ] K4 Looped depth ([#552](https://github.com/surya-koritala/Glyd/issues/552))
- [ ] K5 Attention score options: relative position, L2-clipped q/k, log-scaled ([#553](https://github.com/surya-koritala/Glyd/issues/553))
- [ ] K6 Residual forms: streams with no comb matrix, normalise-and-interpolate, value residual ([#554](https://github.com/surya-koritala/Glyd/issues/554))
- [ ] K7 Small kind options, one batch ([#555](https://github.com/surya-koritala/Glyd/issues/555))
- [ ] K8 Parallel block ([#556](https://github.com/surya-koritala/Glyd/issues/556))
- [ ] GGUF architectures: metadata to a model description for every family the runtime serves ([#557](https://github.com/surya-koritala/Glyd/issues/557))
- [ ] One table-driven tool-call and reasoning parser ([#558](https://github.com/surya-koritala/Glyd/issues/558))
- [ ] Derive tool-call and reasoning markers from the chat template ([#559](https://github.com/surya-koritala/Glyd/issues/559))
- [ ] Grammar-held tool calls for every family and a server strict level ([#560](https://github.com/surya-koritala/Glyd/issues/560))
- [ ] Tokenize and detokenize endpoints with token ids ([#561](https://github.com/surya-koritala/Glyd/issues/561))
- [ ] Context overflow policy: truncate old turns, optional shift ([#562](https://github.com/surya-koritala/Glyd/issues/562))
- [ ] Model management over HTTP: pull with progress, delete, load, unload, events ([#563](https://github.com/surya-koritala/Glyd/issues/563))
- [ ] Honour keep_alive per request; preload, unload and pin in the service ([#564](https://github.com/surya-koritala/Glyd/issues/564))
- [ ] Model variants: a named preset over a base model ([#565](https://github.com/surya-koritala/Glyd/issues/565))
- [ ] Config file for serve and service ([#566](https://github.com/surya-koritala/Glyd/issues/566))
- [ ] Server defaults for the chat template ([#567](https://github.com/surya-koritala/Glyd/issues/567))
- [ ] Grammar starts after thinking closes ([#568](https://github.com/surya-koritala/Glyd/issues/568))
- [ ] Host-RAM tier for finished conversations' KV pages ([#569](https://github.com/surya-koritala/Glyd/issues/569))
- [ ] Thinking-token budget ([#570](https://github.com/surya-koritala/Glyd/issues/570))
- [ ] Extended sampler set and a sampler order ([#571](https://github.com/surya-koritala/Glyd/issues/571))
- [ ] Accept GBNF grammars ([#572](https://github.com/surya-koritala/Glyd/issues/572))
- [ ] Prompt progress events in streams ([#573](https://github.com/surya-koritala/Glyd/issues/573))
- [ ] Server properties in /v1/glyd/info ([#574](https://github.com/surya-koritala/Glyd/issues/574))
- [ ] system_fingerprint and /version ([#575](https://github.com/surya-koritala/Glyd/issues/575))
- [ ] prompt_logprobs and token-id rendering ([#576](https://github.com/surya-koritala/Glyd/issues/576))
- [ ] Load modes: lock weights in RAM, drop the page-cache copy ([#577](https://github.com/surya-koritala/Glyd/issues/577))
- [ ] Agent frameworks and tool calls through the API ([#282](https://github.com/surya-koritala/Glyd/issues/282))

Re-scoped on 2026-11-25 (owner's decision): F1f (#171), F1g (#172), F1i (#244), F1j (#245) and F1k (#246) closed, and F1c (#168) narrowed to the remaining Gemma 4 sizes. Families are taken by layer kind and by family, with a release in the last six months; the kinds the closed issues share with recent families stay covered by the kind issues K1 to K8 (#549 to #556), #178, #232 and #333.

Gate: A dated snapshot of the top 50 text-generation models on the Hugging Face Hub (exclusions listed) runs; the 10 most-pulled local models run (including canonical GGUF quants); top-15 curated pre-packed repos hit first token with zero local packing wait; Claude Code, Continue and Open WebUI pass tool-call checks; an Ollama client's long chat is truncated, not refused; every installer channel passes its install test; consumer benchmarks are published per VRAM tier.

---

## M4 Many GPUs, many machines · v0.37.0

[Milestone](https://github.com/surya-koritala/Glyd/milestone/6) · [Epic #124](https://github.com/surya-koritala/Glyd/issues/124)

One SLO-aware planner and runtime from 1 GPU to 8-GPU datacenter nodes and N-machine clusters (NVIDIA first): collectives (NCCL + custom decode all-reduce), SLO-aware Planner v2, tensor and expert parallelism in a node, overlapped pipeline, GeForce boxes without peer access, `glyd cluster` replacing Starlings, `KvTransport` TLS/TCP, a drop-in OpenAI server for one 8-GPU node, compressed streaming/P2P weight distribution (`C7`), and sleep-to-host-RAM multi-model tenancy (`C8`).

**Why here (moved ahead of multi-vendor) / Agent rules:** 95%+ of rentable CI and datacenter GPUs are NVIDIA, whereas 4 target AMD GPUs in M5b have zero cloud rentals ([docs/hardware-plan.md](docs/hardware-plan.md)). Proving the unified 8 GB → 8× H100 cluster architecture here avoids stalling datacenter delivery on unavailable hardware. Keep collectives and placement strictly behind `Backend` and `Collective` so Metal/ROCm plug in during M5a/M5b; make `C1` (#194) optimize for explicit workload/SLO intent (`--goal latency|throughput|max-context`, TTFT/TPOT).

- [ ] C2a Collectives: NCCL via cudarc and a custom all-reduce for decode ([#193](https://github.com/surya-koritala/Glyd/issues/193))
- [ ] C1 Planner v2: SLO-aware placement across GPUs, RAM, NVMe and machines ([#194](https://github.com/surya-koritala/Glyd/issues/194))
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
- [ ] Admission limits: queued requests and queued prompt tokens, wait and run timeouts ([#578](https://github.com/surya-koritala/Glyd/issues/578))
- [ ] Request priorities, preemption by priority, a wait cap and per-key fairness ([#579](https://github.com/surya-koritala/Glyd/issues/579))
- [ ] Cache-aware and shortest-prefill-first admission order ([#580](https://github.com/surya-koritala/Glyd/issues/580))
- [ ] Fair prefill share among long prompts and a stream interval ([#581](https://github.com/surya-koritala/Glyd/issues/581))
- [ ] Coordinator failover for glyd cluster ([#582](https://github.com/surya-koritala/Glyd/issues/582))
- [ ] Continue a stream on another replica ([#583](https://github.com/surya-koritala/Glyd/issues/583))

Gate: Two consumer machines run a model neither holds alone at the speed `glyd fit` predicted (±10%); a 600B-class MoE runs on one 8× H100 node with experts in host RAM (EP decode within 15% of SGLang); tensor parallel on 2 GPUs is at least 1.7× one GPU on a model that fits one GPU; one 8-GPU node serves through the OpenAI API with data-parallel replicas, standard metrics, compressed streaming load (`C7`), and <5 s sleep-to-RAM wake (`C8`).

---

## M5a Apple Silicon, CPU and desktop extras · v0.38.0

[Milestone](https://github.com/surya-koritala/Glyd/milestone/5) · [Epic #123](https://github.com/surya-koritala/Glyd/issues/123)

The same `glyd run` on Apple Silicon Macs (`mlx-c` + custom Glyd Metal kernels + unified-memory planning + macOS arm64 builds) and CPU-only machines (x86_64 and arm64), paired with the deferred desktop modalities and power features: Vision for every VLM family, Embeddings & Rerankers, Batched LoRA adapters, and Speculative Decoding for any model.

**Why here / Agent rules:** Most developers evaluate local tools on Apple Silicon laptops and deploy to Linux/NVIDIA servers, so `M4 + M5a` covers ~90%+ of real adoption without waiting on unrentable AMD cards. Keep MLX lazy evaluation behind `ModelStep::launch` / `finish`; count unified memory once (`V3c`); build Vision, Embeddings, LoRA and Spec Decode against `ModelDesc` across backends.

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
- [ ] Read MLX affine-quantized checkpoints as released ([#584](https://github.com/surya-koritala/Glyd/issues/584))
- [ ] Desktop tray app over glyd service ([#585](https://github.com/surya-koritala/Glyd/issues/585))

Gate: Apple M1 and M4 (unified memory) and CPU x86_64/arm64 pass KL thresholds against CUDA on the fitting M3 model subset; macOS arm64 builds pass CI and install tests; Vision, Embeddings, Batched LoRA and Speculative Decoding pass logits and `glyd bench` gates on reference hardware.

---

## M5b AMD ROCm and older NVIDIA · v0.39.0

[Milestone](https://github.com/surya-koritala/Glyd/milestone/9) · [Epic #445](https://github.com/surya-koritala/Glyd/issues/445)

Extend the unified runtime to AMD GPUs (RDNA 3/4, Strix Halo, CDNA 3/4 MI300X/MI325X/MI355X) and older NVIDIA Turing (`sm_75`), through the HIP device layer (`R1b`), warp/fragment kernel portability (`V0`), AMD kernel library delegation (`V8b`: hipBLASLt, aiter, composable_kernel), and dynamic vendor backend downloads (`P7b`).

**Why split into its own track / Agent rules:** [docs/hardware-plan.md](docs/hardware-plan.md) records zero rentals on Lambda, AWS or Vast.ai for RX 7900 XTX, RX 9070 XT, Strix Halo or MI300X. Splitting M5b lets AMD progress in parallel as hardware access or contributors join without blocking M4, M5a or NVIDIA 1.0. Never mark AMD/Turing acceptance criteria complete from compile-only checks — real hardware KL and `glyd bench` records are mandatory. Keep `V0` zero-overhead on CUDA.

- [ ] R1b HIP device layer over the backend seam ([#446](https://github.com/surya-koritala/Glyd/issues/446))
- [ ] P7b Small core that downloads its backend ([#160](https://github.com/surya-koritala/Glyd/issues/160))
- [ ] V0 Kernel portability layer: warp size, lane masks, tensor-core fragments ([#183](https://github.com/surya-koritala/Glyd/issues/183))
- [ ] V1 AMD RDNA 3/4 and Strix Halo: a beta first, then speed ([#184](https://github.com/surya-koritala/Glyd/issues/184))
- [ ] V2 AMD CDNA 3/4 (MI300X, MI325X, MI355X) ([#185](https://github.com/surya-koritala/Glyd/issues/185))
- [ ] V8b AMD kernel libraries where they are faster: hipBLASLt, aiter, composable_kernel ([#447](https://github.com/surya-koritala/Glyd/issues/447))
- [ ] V5 NVIDIA Turing (sm_75): fp16 paths ([#190](https://github.com/surya-koritala/Glyd/issues/190))
- [ ] Older GPUs: V100 (sm_70) and RTX 20 series ([#283](https://github.com/surya-koritala/Glyd/issues/283))

Gate: Per device (NVIDIA T4; AMD RX 7900 XTX, RX 9070 XT, Strix Halo, MI300X), the fitting M3 subset passes KL within the per-level threshold against the CUDA run with committed `benchmarks/` records; `P7b` downloads signed backend bundles cleanly.

---

## M6 Stable and operable · v1.0.0

[Milestone](https://github.com/surya-koritala/Glyd/milestone/7) · [Epic #125](https://github.com/surya-koritala/Glyd/issues/125)

1.0 commits to stable CLI, HTTP APIs, token-level engine API, library API and pack formats, with operations (Helm, LeaderWorkerSet, dashboards), router integration (`llm-d`, Gateway API inference extension, Dynamo, SMG), a deterministic batch-invariant mode, supply-chain security (SBOM, provenance, reproducible builds), and a published comparison with vLLM and SGLang.

**Why here / Agent rules:**
- **Blackwell B200 rule (`#208` vs `#192`):** Never publish a B200 (`sm_100`) comparison against vLLM/SGLang running `mma.sync` against their native `tcgen05` kernels. Standard `sm_100` dense prefill and FP8/NVFP4 GEMMs MUST run via CUTLASS/FlashInfer (`V8a`, #255) before benchmarking B200; if `sm_100` library support is not ready at 1.0 freeze, gate M6 on H100/H200 + RTX 5090 (`sm_120`) and publish B200 once `V8a`/`V7` lands.
- **AMD non-blocking rule:** MI300X comparisons publish when M5b (`V2`, #185) lands and do not block NVIDIA 1.0.

- [ ] D2 Integrate with KV-aware routers: llm-d, Gateway API, SMG ([#203](https://github.com/surya-koritala/Glyd/issues/203))
- [ ] D5 Deterministic, batch-invariant mode ([#206](https://github.com/surya-koritala/Glyd/issues/206))
- [ ] D7 Published comparison with vLLM and SGLang, per quality tier ([#208](https://github.com/surya-koritala/Glyd/issues/208))
- [ ] D8 1.0 stability: CLI, API and pack formats ([#209](https://github.com/surya-koritala/Glyd/issues/209))
- [ ] D3b Helm, LeaderWorkerSet, images and dashboards ([#258](https://github.com/surya-koritala/Glyd/issues/258))
- [ ] D6b SBOM, provenance level and reproducible builds ([#259](https://github.com/surya-koritala/Glyd/issues/259))
- [ ] D9 Token-level engine API for Dynamo and SMG ([#260](https://github.com/surya-koritala/Glyd/issues/260))
- [ ] S14b Maintainers besides the founder ([#458](https://github.com/surya-koritala/Glyd/issues/458))
- [ ] TLS and mTLS on the listener with certificate reload ([#586](https://github.com/surya-koritala/Glyd/issues/586))
- [ ] Trace-context propagation, request logging controls and request-id header ([#587](https://github.com/surya-koritala/Glyd/issues/587))
- [ ] Deep health probe (one token) and a step watchdog ([#588](https://github.com/surya-koritala/Glyd/issues/588))
- [ ] Per-step metrics and a capacity answer for planners ([#589](https://github.com/surya-koritala/Glyd/issues/589))
- [ ] Autoscaling recipes on Glyd metrics, with scale to zero ([#590](https://github.com/surya-koritala/Glyd/issues/590))
- [ ] Rolling upgrades: version skew and a drain-and-replace test ([#591](https://github.com/surya-koritala/Glyd/issues/591))
- [ ] Offline batch runner over JSONL ([#592](https://github.com/surya-koritala/Glyd/issues/592))
- [ ] SageMaker, Vertex and KServe v2 routes ([#593](https://github.com/surya-koritala/Glyd/issues/593))
- [ ] Save and restore a conversation's KV pages to a file ([#594](https://github.com/surya-koritala/Glyd/issues/594))
- [ ] Stored Responses: retrieve, cancel and previous_response_id ([#595](https://github.com/surya-koritala/Glyd/issues/595))

Gate: A documented stable surface with CI compatibility checks; `llm-d` routes to Glyd pods by prefix-cache hits; a Helm + LeaderWorkerSet install passes readiness on a GPU cluster; SBOM and provenance on every artifact; a comparison with vLLM and SGLang per quality tier (lossless against bf16, lossy levels against FP8 and NVFP4) on H100 (and B200 via `V8a` `sm_100` kernels), published with configs and logs whatever the result.

---

## Later: datacenter scale (after 1.0)

[Milestone](https://github.com/surya-koritala/Glyd/milestone/8) · [Epic #210](https://github.com/surya-koritala/Glyd/issues/210)

Work deferred until the single-node server and small clusters are adopted: prefill/decode disaggregation with compressed KV transfer, NIXL RDMA KV transport, wide expert parallelism and NVL72, RL weight updates, custom `tcgen05` kernels for Glyd's compressed weight formats (standard `sm_100` dense/FP8/NVFP4 GEMMs are delegated via `V8a` #255), a Vulkan fallback, and an MCP wrapper.

**Agent rules:** Do not start these issues before v1.0.0 unless explicitly prioritized; preserve clean extension hooks in M2–M6 (128-bit content-addressed blocks, `KvTransport`, `Collective`, deterministic mode).

- [ ] P5d MCP server for any model ([#157](https://github.com/surya-koritala/Glyd/issues/157))
- [ ] V6 A fallback for other GPUs ([#191](https://github.com/surya-koritala/Glyd/issues/191))
- [ ] V7 Datacenter Blackwell: custom tcgen05 compressed kernels ([#192](https://github.com/surya-koritala/Glyd/issues/192))
- [ ] D1 Prefill/decode disaggregation with compressed KV transfer ([#202](https://github.com/surya-koritala/Glyd/issues/202))
- [ ] C4b NIXL transport for RDMA ([#261](https://github.com/surya-koritala/Glyd/issues/261))
- [ ] D4 Wide expert parallelism across nodes and multi-node NVLink ([#205](https://github.com/surya-koritala/Glyd/issues/205))
- [ ] D5b Weight updates for RL rollouts ([#262](https://github.com/surya-koritala/Glyd/issues/262))
- [ ] Audio endpoints ([#596](https://github.com/surya-koritala/Glyd/issues/596))
- [ ] Render-only front end ([#597](https://github.com/surya-koritala/Glyd/issues/597))
- [ ] Server-side tool execution ([#598](https://github.com/surya-koritala/Glyd/issues/598))
- [ ] Fleet capacity planner ([#599](https://github.com/surya-koritala/Glyd/issues/599))
- [ ] Tensor parallel across machines over RDMA ([#600](https://github.com/surya-koritala/Glyd/issues/600))
- [ ] Snapshot and restore of a loaded server ([#601](https://github.com/surya-koritala/Glyd/issues/601))

Revisited after 1.0, by demand.

## Always

- `glyd bench` (Rust) replaces the Python benchmarks; results in `benchmarks/`.
- Python is ported or removed as each milestone replaces it (`glyd-gpu-py/`, `gpu/*.py`, `glyd-engine/tools/*.py`,
  `condor/bench`, `scripts/*.py` → `cargo xtask`).
