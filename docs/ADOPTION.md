# How Glyd becomes the inference layer everyone uses

The question: what does it take for Glyd to be the software people reach for by default, from one gaming GPU to the
largest models on multi-cluster datacenters, and have it just work? What are we not doing?

This is the deliberation behind [PLAN.md](PLAN.md). PLAN.md says what to build in what order; this file says why, what
wins adoption, where Glyd is strong and weak, and what is missing. Statements about Glyd are about this repository as
of November 2026, with file references; statements about other projects are from their public documentation and reporting.

---

## 1. The answer in one paragraph

Inference software gets adopted for five reasons: **the model I want runs on day 0, on the hardware I have, through the API
my tools already speak, it wins a number I care about, and I trust it to keep working and stay open.** llama.cpp/Ollama won
the first three for consumers, vLLM/SGLang won all five for datacenters. Glyd has two things none of them have (lossless
compression that is verifiably the original model, and MoE placement that runs models far larger than the GPU) and one
architecture choice that is right (one Rust binary that needs only the GPU driver). What it lacks is breadth: models,
hardware, APIs, integrations, scale-out and trust. The single biggest change is architectural: today the engine is a set of
excellent hand-built pipelines, one per model family, each with its own subset of features. To be adopted everywhere it
must become **one runtime where every model gets every feature on every backend by construction**. With that in place,
the go-to-market is: win consumers with "the biggest model your GPU can run, at the best verified quality, with Ollama's
ease and Ollama's API", win homelabs with "pool your GPUs, securely, any vendor", and enter datacenters through the places
compression is worth most (cold start, KV movement, prefix capacity, more replicas per GPU) before competing head-on on
throughput.

---

## 2. What adoption runs on: lessons from the projects that won

| Project | Why it won | What it means for Glyd |
| :--- | :--- | :--- |
| **llama.cpp** | Runs everywhere (CPU, Metal, CUDA, ROCm, Vulkan, SYCL, phones). GGUF became the local format. A C library others embed: LM Studio, Jan, GPT4All, KoboldCpp, mobile apps. Community quantizers publish GGUFs within hours of a release. | Be embeddable, not only a server. Read GGUF. Have packs on the Hub the day a model ships. |
| **Ollama** | `ollama run llama3`: one install, short model names, a background service, works on Mac, Windows, Linux, any GPU or none. Its API became a de facto standard: Open WebUI, Continue, n8n, LangChain, Home Assistant all speak it. | Match the UX and **speak Ollama's API** so every integration works unchanged. |
| **vLLM** | Throughput (PagedAttention, continuous batching). Day-0 support: model labs upstream their architectures before release. Hardware vendors (AMD, Intel, Google TPU, AWS Neuron, Gaudi) build backends for it. Kubernetes ecosystems (KServe, llm-d, production-stack) build on it. Foundation governance and many corporate maintainers. | Day-0 needs lab relationships and a model system cheap enough to extend. Datacenter adoption runs through Kubernetes projects and hardware vendors, not individual users. |
| **SGLang** | Fastest on large MoE (DeepSeek-class), prefix sharing (RadixAttention), mature prefill/decode disaggregation. Day-0 for the models it targets. | Leadership on the largest models at datacenter scale is a reputation in itself. |
| **exo** | A viral demo (a cluster of Macs running a model none could alone), zero configuration, Apple-first. | One striking demo per segment, reproducible by anyone. |
| **TensorRT-LLM / Dynamo** | Peak NVIDIA performance; NVIDIA's backing; orchestration for disaggregated serving. | The datacenter bar for kernels on NVIDIA. |

Five forces, in the order users feel them:

1. **Day-0 models.** If the new model does not run today, nothing else matters.
2. **Runs on what I have.** Every GPU vendor, old GPUs, any OS, no toolkit, any driver from the last two years.
3. **Drop-in.** The formats I already downloaded, the APIs my tools speak, the platforms I deploy on.
4. **Wins a number I care about.** Consumers: the biggest model at a usable speed, quality, first-token time. Datacenters:
   goodput per dollar at a latency target, cold start, reliability.
5. **Trust.** It keeps working, it is safe to expose, it stays open, someone will fix it.

---

## 3. Where Glyd is strong, and where it is not

### Unfair advantages (things others cannot copy quickly)

- **Lossless and verifiable.** A Glyd pack is the original model: `glyd verify` checks every tensor against the source's
  sha256. No other compressed format can say that. For consumers it removes "is this quant any good?"; for regulated
  industries it is provenance: the model you serve is provably the model the lab released. ([gpu/README.md](../gpu/README.md),
  [glyd-gpu/](../glyd-gpu/))
- **A third less memory for bf16 weights and KV cache** (weights ~10.75 bits, KV 12 bits; the measured lossless floor is
  about 10.5-10.6). Qwen3-32B on one 48 GB GPU instead of two. ([DETAILS.md](../DETAILS.md))
- **MoE placement by use.** Condor runs a 180B MoE on a 16 GB card at 79-131 tokens/s by keeping busy experts on the GPU,
  the rest in RAM or on disk, computing misses on the CPU, and spending bits where use is high
  ([condor/README.md](../condor/README.md), `glyd-engine/src/offload.rs`). Open models are moving to large sparse MoE; this
  is the capability that lets consumers follow them.
- **Driver-only, one Rust binary.** The engine opens the CUDA driver at run time and links no CUDA runtime; no toolkit, no
  Python, no PyTorch (`glyd-engine/src/ffi.rs`). The install is ~100 MB, not ~8 GB.
- **A secured cluster.** Starlings has enrollment with fingerprints, mutual TLS, revocation and scoped tokens
  ([starlings/docs/daemon.md](../starlings/docs/daemon.md)); exo and llama.cpp RPC trust the network.
- **A codec for data in motion.** Glyd started as a compression codec and store. Datacenter inference is increasingly
  limited by moving bytes: weights at cold start, KV between prefill and decode, KV offloaded to CPU and SSD, prefixes
  shared across nodes. Lossless compression of those bytes is a direct win no engine has.

### Where the advantage is small or negative (be honest about it)

- **Datacenters serve FP8 and FP4.** Lossless coding takes ~18% off an FP8 checkpoint and ~7% off NVFP4 (DETAILS.md). The
  headline "-33%" is a bf16 number. In the datacenter, Glyd competes on kernels and scheduling first.
- **High batch is compute-bound.** At 64 sequences the packed model ran at 0.82-0.86x bf16's tokens/s on A6000s; on an H100
  the `mma` layout costs more GPU time a token than bf16 (Qwen3-32B: 40.6 vs 28.2 ms) and only `mma12` is faster (26.4 ms)
  (DETAILS.md). Compression pays where memory binds: small batches, long contexts, big models on small GPUs.
- **Breadth.** vLLM and llama.cpp each run hundreds of architectures on many vendors with hundreds of contributors. Glyd
  runs about a dozen families on NVIDIA, written largely by one person with AI agents.

### What this implies

Lead where the advantages are decisive (consumers and prosumers running models bigger than their GPU; memory-bound and
cost-bound serving; data movement in datacenters), and close the breadth gap with architecture rather than headcount.

---

## 4. What we are not doing: the audit

State is the code in this repository. Priority: **P0** blocks "just works" for the segment that adopts first; **P1** blocks
the next segment; **P2** widens the lead.

### Models and formats

| Gap | Why it matters | State today | Priority |
| :--- | :--- | :--- | :--- |
| A general model system: a new architecture as a description plus a weight map, not a hand-written pipeline | Day-0 support at the pace labs release (several notable models a week); every feature applies to every model | Per-family files (`qwen35.rs`, `model_gemma4.rs`, `kimi.rs`, `glm.rs`, `muse.rs`, `nemotron.rs`, `deepseek_v4.rs`) and a separate serve path for Qwen3.8-Flash-Next (`qwen4x_serve.rs`); a shared `dynamic/` executor exists for the large MoE families | **P0** |
| Coverage of the long tail: Llama 3.x/4, Mistral/Mixtral, Phi, Gemma 2/3, Qwen 2/2.5, GPT-OSS, DeepSeek V3/R1, SmolLM, Granite, OLMo | These are most of what people run | vLLM only, through Python | **P0** |
| GGUF loading (llama.cpp quant types) | The Ollama/llama.cpp library people already downloaded; community quantizers' output | None in the engine (the codec reads GGUF tensors: `src/weights/`) | **P0** |
| Native quantized checkpoints: FP8, NVFP4, MXFP4, AWQ, GPTQ, compressed-tensors, MLX format | Labs and the Hub ship these; datacenters serve them | FP8/FP4/MXFP4 for some families; others through vLLM | **P1** |
| Pre-packed repos for every popular model within hours of release (a bot, verified by `glyd verify`) | Packing a 70B on a CPU is minutes of friction; trust needs consistency | Hand-made `glyd/*` repos per release | **P0** |
| Embeddings and rerankers | RAG is the most common production workload after chat | No `/v1/embeddings` | **P1** |
| Vision for every VLM family; audio (ASR, TTS) | Multimodal is now default for new releases | Vision for Qwen3.8-Flash-Next only (`vision.rs`) | **P1** |
| LoRA adapters, many at once (multi-LoRA batching) | Enterprises serve hundreds of fine-tunes on one base | None | **P1** |

### Hardware

| Gap | Why it matters | State today | Priority |
| :--- | :--- | :--- | :--- |
| Driver floor: CUDA 12 builds for drivers 525-570 | Cloud images and enterprises lag; "update your driver" is the most common install failure | The engine needs driver 580+ (CUDA 13) (`machine.rs`); the GPU library already builds CUDA 12 and 13 | **P0** |
| Turing and Volta (T4, V100, RTX 20xx) | T4 is the most common cloud GPU (Colab, inference instances); V100s fill universities | Ampere or newer only | **P1** |
| AMD: RDNA 3/4 measured, Instinct (CDNA, wave64) | AMD is the second datacenter vendor and wants alternatives; consumers with Radeon cards have few good options | RDNA builds in development mode, never run; no CDNA | **P0** (RDNA), **P1** (CDNA) |
| Apple Silicon (MLX/Metal) | The largest pool of high-memory consumer machines (64-512 GB unified) | None | **P0** |
| CPU-only, and a Vulkan fallback for every other GPU (Intel Arc, older AMD, iGPUs) | "Runs on anything" means a slow path is better than none | Expert kernels only (AVX2/AVX-512/NEON) | **P1** |
| Windows as a first-class release target | Most consumer GPUs are in Windows machines | Condor's engine runs on Windows; the release workflow builds Linux only; `glyd run` refuses off Linux | **P0** |
| aarch64 servers (Grace, Graviton + GPUs) | GH200/GB200 hosts are aarch64 | Builds, never run (`ARCHS = ["x86_64"]` in `machine.rs`) | **P1** |

### APIs and interop

| Gap | Why it matters | State today | Priority |
| :--- | :--- | :--- | :--- |
| Ollama's API (`/api/chat`, `/api/generate`, `/api/tags`, `/api/pull`, ...) | Every Ollama integration works unchanged: switching costs nothing | None | **P0** |
| Constrained decoding: JSON schema, regex, grammars | Agents and tool use break without it; vLLM, SGLang, llama.cpp all have it | JSON is asked for in the prompt (`server.rs:327`); `guided_*` and schema refused | **P0** |
| Tool-call and reasoning parsers for every family | Wrong tool parsing is the most common "the model is dumb" report | Hermes and XML formats | **P0** |
| OpenAI Responses API, batch API; Anthropic prompt caching (`cache_control`) | What new agent frameworks and coding agents call | Chat/completions and Anthropic messages only | **P1** |
| An embeddable library: a stable C ABI and Rust crate for the engine, thin generated bindings (Swift, Kotlin, JS, Python) | The llama.cpp path: be the engine inside other apps | The GPU library has a C API; the engine is a program only | **P1** |

### Serving and scale

| Gap | Why it matters | State today | Priority |
| :--- | :--- | :--- | :--- |
| One scheduler for every family: continuous batching, paged KV, chunked prefill, prefix cache, preemption | Without it, only one model family serves more than one user | Qwen3 path only (1-8 users, `sched.rs`); every other family one request at a time | **P0** |
| Hundreds of concurrent requests per GPU | The datacenter's first question | Capped at 8 | **P0** |
| Long context: KV past 40,960 positions, context parallelism for 128K-1M prompts | Agents, code bases, documents | KV tables calibrated up to 40,960 (`kvcal.rs`) | **P1** |
| Speculative decoding for every model: draft models, MTP, EAGLE-style heads, n-gram/prompt lookup | 2-3x single-stream speed; code editing benefits most from prompt lookup | Gemma 4 drafts, MTP for GLM (dev) and Qwen3.8-Flash-Next | **P1** |
| Tensor/expert parallelism with NCCL/RCCL; RDMA transport (InfiniBand, RoCE, EFA, GPUDirect) | Every model above one node; every datacenter interconnect | TP for four families in development builds with its own gather over peer access; Starlings over TLS/TCP; no NCCL, no RDMA | **P1** |
| Prefill/decode disaggregation with KV transfer; KV offload to CPU/SSD and a shared KV store | Cost at scale for long prompts and agent workloads | None (the codec and `glyd-store` are the natural base) | **P1** |
| Wide expert parallelism for 600B-3T MoE across nodes; NVL72 multi-node NVLink | The largest models in the largest deployments | Kimi-K3 on one 8x A100 node at ~4 tokens/s with host experts | **P2** |
| Weight update API for RL rollouts (trainer pushes new weights in place) | Labs pick inference engines for RL as much as for serving | None | **P2** |
| Batch-invariant deterministic mode | RL numerics, evals, audits; Glyd's bit-exact story extends naturally | `exact=True` gives bf16's logits on the Python path | **P2** |

### Production, distribution, trust

| Gap | Why it matters | State today | Priority |
| :--- | :--- | :--- | :--- |
| Prometheus metrics, OpenTelemetry traces, structured logs | No operator deploys what they cannot see | `/stats` (JSON), no `/metrics`, no tracing | **P0** for datacenter |
| Kubernetes: Helm chart, multi-node pods (LeaderWorkerSet), Gateway API Inference Extension endpoint picker, KServe runtime, llm-d integration, autoscaling signals | How datacenters adopt engines | None | **P1** |
| A background service: models loaded on demand, unloaded when idle, several at once, adapting when other apps take VRAM | Ollama's core UX; consumers run games and browsers on the same GPU | `run`/`serve` start one model in the foreground | **P0** |
| Installers: winget, apt/yum repo, Nix, Docker images per vendor, cloud marketplace images; a desktop app | Each missing channel is a segment that never tries it | install.sh, Homebrew, PyPI | **P0** (winget, Docker), **P1** (rest) |
| Inclusion where users already are: Hugging Face "Use this model" local apps, LM Studio/Jan runtimes, Docker Model Runner backends, LiteLLM/OpenRouter/LangChain/LlamaIndex providers | Distribution through other products reaches millions | None | **P0** (Hub, Ollama API), **P1** (rest) |
| Signed releases (Sigstore), SBOM, reproducible builds, a security advisory process that has been exercised | Enterprise procurement asks for each | SHA256 sums; SECURITY.md outline | **P1** |
| A hardware × model × OS test matrix on real GPUs, run on every release; opt-in crash reports | "Just works" is a measured rate, not a promise | GPU checks run by hand on rented machines; CI is CPU-only plus one self-hosted PC | **P0** |
| Governance after the license history | v0.29-v0.32 were BUSL. Companies will ask whether it can happen again; the CLA lets the licensor relicense contributions | Apache-2.0 now; CLA with relicensing rights; one maintainer | **P0** |
| Contributor on-ramp: architecture docs, "add a model / a kernel / a backend" guides, CPU-runnable tests for every model, good first issues | Breadth comes from contributors, not one author | Terse per-file comments; no architecture guide | **P0** |

---

## 5. The architecture this requires

Every row above that says "for this family only" or "in development builds" has the same cause. Features are implemented
per model pipeline. Adding the 13th family means re-implementing batching, TP, offload, drafting and parsing again; adding
a backend means porting every pipeline. That does not scale to day-0 support for every release on every vendor.

The target is one runtime with four orthogonal axes, so that **models × features × backends × topologies** are filled by
construction:

```
                        ┌──────────────────────────────────────────────────────────────┐
  APIs                  │ OpenAI (chat, completions, responses, embeddings) · Anthropic │
                        │ Ollama · MCP · gRPC/C ABI · Rust crate                        │
                        └──────────────────────────────┬───────────────────────────────┘
                        ┌──────────────────────────────┴───────────────────────────────┐
  Serving core          │ one scheduler: continuous batching, paged KV (lossless pages),│
                        │ prefix cache, chunked prefill, preemption, spec decode,       │
                        │ constrained decoding, multi-LoRA, sampling                   │
                        └──────────────────────────────┬───────────────────────────────┘
                        ┌──────────────────────────────┴───────────────────────────────┐
  Model system          │ model = description (layers, attention kind, MoE, norms,      │
                        │ rope) + weight map; a library of layer kinds: MHA/GQA/MLA,    │
                        │ sliding, linear/delta attention, Mamba, MoE, vision encoders; │
                        │ a CPU f32 reference for every layer kind (the *_ref.rs files) │
                        └──────────────────────────────┬───────────────────────────────┘
                        ┌──────────────────────────────┴───────────────────────────────┐
  Placement planner     │ costs × rates × topology → which layers/experts/KV live where,│
                        │ at which level: VRAM → peer GPUs → host RAM → NVMe → other    │
                        │ machines (Condor's policy and Starlings' split, generalized)  │
                        └───────────────┬──────────────────────────────┬───────────────┘
                        ┌───────────────┴──────────────┐ ┌─────────────┴────────────────┐
  Backends              │ CUDA · ROCm/HIP · Metal(MLX) │ │ Transports: shared memory,   │
                        │ · CPU · (Vulkan)             │ │ NVLink/P2P, NCCL/RCCL, RDMA  │
                        │ hot kernels hand-tuned per   │ │ (IB/RoCE/EFA), TLS/TCP for   │
                        │ backend; long tail portable  │ │ homes and WANs               │
                        └──────────────────────────────┘ └──────────────────────────────┘
  Formats               glyd packs (lossless, lossy levels) · safetensors · GGUF · FP8/FP4/AWQ/GPTQ · MLX
```

What already exists maps onto this: `dynamic/` is the start of the model system (families as descriptions on shared
kernels); `sched.rs` is the scheduler to generalize; `offload.rs`, `costs.rs` and `split.rs` are the planner's pieces;
`ffi.rs` already takes HIP's driver table, the seam for the backend trait; `range.rs` and Starlings are the transport's
first implementation. The work is to make these the only path, not one path among several.

Rules that keep it that way:

1. A model lands only as a description on the shared runtime, with its CPU reference and a logits test.
2. A feature lands only if it works for every model the runtime runs (or is refused for a stated reason in one line).
3. A backend lands with the full model test suite passing on it.
4. Speed claims land with a `glyd bench` file; CI refuses a regression on the reference GPUs.

---

## 6. Two earlier decisions, reconsidered

### "Everything Rust, no Python"

Right for the runtime: one binary, no environment, no dependency drift, memory safety, the install that just works. Wrong
if it means "no Python users". Most ML engineers, notebooks, evaluation harnesses and RL trainers are Python. Recommended
reading of the rule: **no Python in the engine, the installer or the user's runtime path; thin, generated bindings for
other languages (Python included) over the C ABI, so Glyd can be called from wherever people work.** A binding of a few
hundred generated lines carries none of the costs that made Python a problem (environments, CUDA wheels, version pins).

The same applies to vLLM and SGLang: datacenter teams will not switch engines on day one, but they will adopt a component.
Instead of deleting the vLLM plugin outright, replace its 10,700 lines of Python logic with a thin shim over the Rust/C
library (weights decode, KV pages), so Glyd's compression reaches vLLM and SGLang users while the full engine catches up.
This is a decision for the owner (section 10); PLAN.md currently recommends freezing and deleting the plugin.

### What compression is for in the datacenter

Not "a third off the weights" (FP8/FP4 checkpoints leave 7-18%). It is for bytes in motion and memory-bound capacity:

- **Cold start and autoscaling:** a smaller checkpoint loads faster from object storage; scale-from-zero is a top
  serving cost.
- **KV transfer** between prefill and decode workers and across nodes: fewer bytes on the network for the same cache.
- **KV offload and a shared KV store** (CPU RAM, SSD, a cluster-wide store with deduplication of shared prefixes): the
  codec and `glyd-store` were built for exactly this kind of data.
- **More replicas and longer contexts per GPU** where serving is memory-bound (smaller models, long prompts, many tenants).
- **Provenance:** lossless packs verifiably equal the released weights, for regulated deployments.

---

## 7. Segments: who adopts first, and what wins each

| Segment | Who | What they need | What wins them | Gate to enter |
| :--- | :--- | :--- | :--- | :--- |
| 1. Consumers | One GPU, 8-24 GB, Windows or Linux; Mac users | One command, a model library, a chat app, their tools working | **The biggest model their machine can run, at verified quality** ("a 180B MoE on your 16 GB card"), with Ollama's ease and Ollama's API | Windows + Linux + macOS binaries, driver floor lowered, Ollama API, background service, GGUF import, top-50 models packed |
| 2. Prosumers and homelabs | 2-8 GPUs across 1-4 machines, mixed vendors | Pool everything, no networking expertise | **One secure cluster across all their GPUs and Macs**, the planner placing the model | Starlings for N machines, any family, AMD + Apple in the cluster |
| 3. Startups and single-node production | 1-8 datacenter GPUs, Kubernetes | Throughput at a latency target, reliability, metrics, standard deployment | **Same model on fewer GPUs, one static binary**, Helm chart, Prometheus | Scheduler parity with vLLM at 1-256 concurrency on the top-10 models, structured output, multi-LoRA, metrics |
| 4. Datacenter and labs | Hundreds to thousands of GPUs, multiple clusters | The largest models, disaggregation, EP across nodes, RL rollouts | **Lower cost per token through data movement** (KV store, transfer, cold start) plus parity on kernels | NCCL/RCCL + RDMA, PD disaggregation, wide EP, K8s inference gateway, weight-update API, InferenceMAX/MLPerf results |

Order matters: segments 1 and 2 generate users, contributors and credibility quickly and need breadth more than peak
speed; segments 3 and 4 need sustained engineering depth and partnerships. Run segment 3-4 foundations (scheduler,
backends, transports) in parallel from the start so the architecture does not need to be redone.

---

## 8. Distribution: where users actually come from

In order of reach per unit of effort:

1. **Ollama API compatibility.** Every app that supports Ollama supports Glyd: Open WebUI, Continue, n8n, Home Assistant,
   LangChain, LlamaIndex, Raycast, Obsidian plugins.
2. **Hugging Face.** The "Use this model" local apps list; `glyd/` packs for every popular model, published by a bot
   within hours of release with verification and a quality card (KL, one eval) per level.
3. **Be a runtime inside other apps.** LM Studio and Jan host multiple engines; Docker Model Runner hosts backends; mobile
   and desktop apps embed llama.cpp today through its C API. A stable C ABI makes Glyd embeddable the same way.
4. **Package managers and images.** winget, Homebrew, apt/yum, Nix, Docker per vendor, cloud marketplace images, RunPod/Vast/
   Modal templates, SkyPilot recipes.
5. **Kubernetes projects.** Helm chart, KServe runtime, llm-d and Gateway API Inference Extension integration, Dynamo
   compatibility where it applies.
6. **Hardware vendors.** AMD (developer cloud, ROCm partnerships: AMD funds alternatives that run well on Instinct),
   Apple's MLX team, NVIDIA Inception: hardware access for CI in exchange for first-class support.
7. **Model labs.** Pre-release access for day-0 support. Labs give it to engines with users; segments 1-2 earn it.
8. **Proof.** Reproducible benchmarks anyone can rerun (`glyd bench`), submission to independent continuous benchmarks
   (SemiAnalysis InferenceMAX, MLPerf Inference), and one demo per segment that spreads on its own.

---

## 9. What "just works" means, measured

"Just works" is a number. Track it publicly:

| Metric | Target |
| :--- | :--- |
| Install to first chat token on a clean machine (excluding the model download) | Under 2 minutes on Windows, Linux and macOS |
| Test-matrix pass rate: GPUs (NVIDIA Turing→Blackwell, AMD RDNA 3/4 and CDNA 3, Apple M1→M4, CPU) × OS × top-50 models × levels | 99% runs correctly; every failure says why in one line |
| Day-0 rate: of the 20 most-downloaded new open models each quarter, share runnable within 24 hours of release | 90% |
| Never fail at start for lack of memory | 100%: preflight refuses with the numbers before downloading |
| Consumer: largest model at ≥20 tokens/s per tier (8, 12, 16, 24, 32 GB VRAM; 16-128 GB unified memory) | Published table, ahead of Ollama and llama.cpp on the same machine |
| Single node: goodput (requests/s at p99 TTFT and ITL targets) on the top-10 models | At least vLLM and SGLang on H100, B200 and MI300X, with less memory |
| Multi-node: cost per million tokens for a 600B+ MoE on 2-4 nodes | Published, independently reproduced |
| Quality: KL and one eval per lossy level per packed model | On every model card |
| Time to fix a reported "doesn't work on my machine" | Median under a week |

---

## 10. The sequence

Stages with gates, tracks in parallel. PLAN.md's phases map onto these.

**Stage 1: one runtime, one front door (foundation).**
The model system (descriptions on shared kernels), one scheduler for every family, the backend trait with CUDA first
(no speed lost), `glyd` as one program with a background service, Windows/Linux/macOS builds, CUDA 12 driver floor,
Ollama + OpenAI + Anthropic APIs complete, constrained decoding, tool/reasoning parsers per family, GGUF import, the
pack bot for the top 50 models, metrics. Contributor docs. Governance decided.
*Gate:* the top-50 models run through one code path with batching on NVIDIA, on all three OSes.

**Stage 2: every consumer machine (segments 1-2).**
Apple via MLX, AMD RDNA measured, CPU-only, Turing; Condor's placement for every MoE; Starlings for N machines and any
family; the desktop app; HF local-apps listing; LM Studio/Jan/Docker Model Runner runtimes.
*Gate:* the matrix pass rate above 95% on consumer hardware; the per-tier "largest model" table published.

**Stage 3: single-node production (segment 3).**
256+ concurrency, long context, speculative decoding everywhere, multi-LoRA, embeddings/rerankers, vision for every VLM,
Helm/KServe/gateway integration, signed releases and SBOM, AMD Instinct.
*Gate:* goodput at or above vLLM/SGLang on the top-10 models on H100 and MI300X with less memory; three external
production users.

**Stage 4: multi-node and multi-cluster (segment 4).**
NCCL/RCCL and RDMA transports, tensor/expert/pipeline parallelism across nodes, prefill/decode disaggregation with
compressed KV transfer, the cluster KV store, KV-aware routing across replicas and clusters, elastic recovery when a node
fails, NVL72-class multi-node NVLink, weight-update API for RL, deterministic mode.
*Gate:* a 600B+ MoE served across nodes at published cost per token; an independent benchmark result.

Rough horizon with a funded team of 8-12 engineers: stage 1 in one to two quarters, stage 2 overlapping into the third,
stage 3 by the end of the first year, stage 4 into the second year. With one maintainer plus AI agents, stage 1 alone is
the realistic year; that is the strongest argument for section 11's first item.

---

## 11. Risks, and what would sink it

1. **Bus factor.** The engine is ~84K lines written largely by one person with AI agents. No company standardizes on
   that. Mitigation: maintainers per area (runtime, kernels per vendor, models, serving, cluster, distribution), a
   foundation or a company that funds them, architecture docs that let others contribute.
2. **Trust after the license change.** Mitigation: DCO instead of the relicensing CLA, a public governance document,
   a trademark policy, and ideally a neutral foundation (LF AI & Data, PyTorch Foundation, CNCF) so "Apache-2.0 forever"
   is structural, not a promise.
3. **Breadth versus depth.** Hand-tuning every model on every backend loses to projects with 50x the contributors.
   Mitigation: section 5's architecture; a portable kernel path for the long tail; hand-tuning only the hot paths.
4. **Moving targets.** Model architectures change monthly (linear attention, MLA, sparse attention, new MoE routers);
   NVIDIA, vLLM and SGLang ship weekly. Mitigation: lab partnerships for day 0; layer kinds as composable pieces.
5. **Correctness across backends.** A silently wrong kernel on one vendor destroys trust. Mitigation: CPU f32 references
   for every layer kind, logits tests per model per backend in CI, KL thresholds per level.
6. **Support load.** "Works everywhere" invites every machine's problems. Mitigation: `glyd doctor --report`, opt-in crash
   reports, a public compatibility table, a triage rotation.
7. **Over-promising.** Plain claims with logs have been the project's style; keep it. One wrong headline costs more than
   ten right ones earn.

---

## 12. Decisions for the owner

1. **Python, precisely:** none in the runtime (agreed), but thin generated bindings and a vLLM/SGLang shim over the C ABI
   as adoption channels? Recommended: yes.
2. **Governance:** move from the CLA to DCO, publish governance and a trademark policy, and pursue a foundation?
   Recommended: yes; the BUSL history makes this the cheapest trust repair available.
3. **First segment:** consumers and homelabs first (breadth, UX, MoE placement), with datacenter foundations built in
   parallel? Recommended: yes.
4. **Funding and team:** stage targets above assume a team; how will it be staffed (company, sponsors, foundation)?
5. **Level names:** keep the birds as brand, but show bits and lossless/lossy everywhere, and accept aliases (`:lossless`,
   `:q6`, `:q5`, `:q4`, `:q3`) so users of other tools understand them? Recommended: yes.
6. **Hardware partnerships:** approach AMD, Apple's MLX team and NVIDIA Inception for CI hardware now, before the
   backends ship? Recommended: yes.
