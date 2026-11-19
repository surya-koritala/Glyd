# Glyd: the plan

One open-source program that runs open models on whatever hardware someone has, from a 12 GB gaming card to a rack of
datacenter GPUs, NVIDIA, AMD or Apple, at the best quality that hardware can hold, with one command. Rust from top to
bottom, Apache-2.0, no Python on the user's path.

This file says what exists today (measured, with where it is in this repository), what the product looks like when it is
done, how it compares with Ollama, llama.cpp, vLLM, SGLang, exo, mistral.rs and TensorRT-LLM, and the order of the work.
Why, what wins adoption in each segment, and the full audit of what is missing: [ADOPTION.md](ADOPTION.md).

---

## 1. What we have (November 2026)

Four repositories became this one. What each brought, and its state:

| Part | Where | What it does today | State |
| :--- | :--- | :--- | :--- |
| Codec and store | `src/`, `glyd-store/` | Lossless compression library and CLI (Rust, C ABI); safetensors and GGUF tensors coded by element type | Shipping since v0.1 |
| GPU library | `gpu/` (`glyd_gpu.cu`, C API) | bf16 weights and KV cache held in ~10.8 bits a value on the GPU and decoded in the kernels, bit for bit (`mma`, `mma12` layouts) | Shipping (was compiled-only from v0.27) |
| Packer | `glyd-gpu/` (Rust) | Packs a Hugging Face checkpoint for the GPU layouts on the CPU; `glyd verify` checks it | Shipping |
| Engine | `glyd-engine/` (Rust, ~84K lines; CUDA kernels in `kernels/`, ~10K lines) | Its own inference engine: loads Glyd packs; OpenAI and Anthropic APIs and a chat page; CUDA graphs; drafting and MTP; MoE experts cached on the GPU, held in RAM or on disk, misses computed on the CPU; layers split over GPUs; `costs`/`rates` cost models | Shipping (was compiled-only) |
| Python GPU package | `glyd-gpu-py/`, `gpu/vllm/` | `glyd run/serve/doctor`, the vLLM plugin (`--quantization glyd`, lossless KV in vLLM), `glyd.from_pretrained` for transformers | Shipping; to be retired (section 4) |
| Condor | `condor/` | Qwen3.8-Flash-Next (180B MoE) on one 12-16 GB GPU with 32-64 GB RAM, lossy levels, OpenAI + Anthropic APIs, MCP, chat app | Shipping (v0.2.4) |
| Starlings | `starlings/` (Rust, ~63K lines) | One model across GPUs and machines: enrollment, mutual TLS, layer-range pipeline through the engine's `rangeworker` | Beta: 2 GPUs, 2 machines, Qwen3.5 family |

### Hardware, today

| Backend | State | Evidence |
| :--- | :--- | :--- |
| NVIDIA CUDA, Ampere to Blackwell (sm_80, 86, 89, 90a, 100, 120) | Production. The engine opens the CUDA driver at run time, no CUDA runtime linked; cuBLAS used for long prompts where present | `glyd-engine/build.rs`, `gpu/build_lib.sh`; benchmarks/gpu (L4, A10, A100, H100, GH200, RTX 3090/4080 SUPER/4090/5090, A6000) |
| AMD ROCm/HIP, RDNA 3 and 4 (gfx1100, 1101, 1200, 1201) | Builds (development builds only): the engine's device layer, kernels and the GPU library compile with hipcc in wave32 (`GLYD_HIP=1`); Hopper-only kernels trap on AMD. No measured run in this repository yet | `glyd-engine/build.rs` (`hip()`), `gpu/hip.h`, `gpu/build_lib_hip.sh`, `tools/runpod/jobs/amd_job.sh` (RX 7900 XTX bring-up job, not yet run) |
| AMD Instinct, CDNA 3/4 (MI300X, MI325X, MI355X) | None. CDNA runs wave64 only; the kernels assume 32-lane warps | — |
| Apple Silicon (Metal / MLX) | None in the engine. A source-only review of MLX and mlx-lm; one starlings route over an M1 Max **CPU** | `starlings/docs/apple-backend-feasibility.md` |
| CPU | For MoE experts that miss the GPU: AVX2, AVX-512 and NEON kernels; no CPU-only model run | `glyd-engine/src/cpu_simd.rs`, `cpu_experts.rs`, `cpu_fp4.rs`, `cpu_hb.rs` |
| Windows | Condor's engine runs on Windows (L4, Windows Server 2022); the release workflow builds Linux only | `condor/setup.ps1`, `.cargo/config.toml` |

### Quality levels, today

| Level | What it is | Bits a weight | Where |
| :--- | :--- | ---: | :--- |
| penguin | Lossless: every bf16 weight back bit for bit (`mma` ~10.75 bits, `mma12` ~12.04, the smaller or faster per GPU); an FP8/FP4/MXFP4 release kept as its own codes (FP8 also in a lossless ~7-bit coding) | ~10.75-12 (bf16 sources) | engine (Qwen3, Gemma 4, the large MoE families), vLLM plugin |
| kestrel | Lossy, Int6 | 6.5 | engine (Qwen3.5/3.6/3.8 family) |
| swift | Lossy, Int5 | 5.5 | engine (Qwen3.5 family), Starlings beta |
| wren | Int4; Google's QAT 4-bit codes kept exactly for Gemma 4 | 4.5 | engine (QAT checkpoints in release builds; others development only) |
| hummingbird | Experts at 2-4 bits by how often they are used, the rest 4-6 | ~2.0-3.4 (experts) | engine: Nemotron 3 Super; Condor's Qwen3.8-Flash-Next (`qwenserve`) |

Lossless KV cache: 12 bits a value where bf16 has 16, bit for bit, in 16-token pages with per-model tables (Qwen3 path:
head_dim 128, full attention, up to 40,960 positions). Qwen3.5/3.8 and Gemma 4 keep a bf16 cache today.

### Measured, the headline numbers

- Lossless weights: 32-33% smaller on every popular open model measured (README, DETAILS.md); Qwen3-32B on one 48 GB GPU
  where bf16 needs two, 1.05-1.42x bf16's tokens/s at 1-32 sequences on those GPUs.
- Condor, RTX 4080 SUPER 16 GB + 64 GB RAM: a 180B MoE at 79-131 tokens/s, GSM8K 147-149/150, KL 0.12-0.39 by mode
  (`condor/README.md`).
- Starlings, Qwen3.8-27B swift on two RTX 4090 machines (0.4-0.6 ms apart): 43.5 tokens/s split against 46.0 on one,
  51.9-53.8 with drafting; about 1.2 ms a token for the link and daemons (`starlings/benchmarks/`).

### Gaps that matter most

1. **Model coverage.** The engine runs a list of families (Qwen3 dense; Qwen3.5/3.6/3.8 at lossy levels; Gemma 4 12B/31B;
   Nemotron-H; GLM-5.3-Flash; Kimi-K3; DeepSeek-V4-Flash; Muse Spark; Qwen3.8-Flash-Next with vision). Llama, Mistral, Phi,
   GPT-OSS, older Gemma and Qwen2.x, and Qwen3.5 at penguin run only through vLLM (Python).
2. **Concurrency.** The Qwen3 path has a real scheduler (paged KV, prefix caching, chunked prefill, batched decode,
   preemption) for 1-8 users; every other family serves one request at a time. vLLM and SGLang serve thousands.
3. **Hardware.** NVIDIA x86_64 only in practice (aarch64 builds, never run). AMD builds in development mode and is
   unmeasured; Apple and CPU-only do not exist.
4. **Python behind the fallback.** install.sh's default is already Rust-only (the engine runs directly), but every model
   the engine does not run needs the Python tool and vLLM (`GLYD_WITH_VLLM=1`), and so does the speed-bar choice.
5. **Three front doors.** `glyd`, `condor/setup.sh` and `starlings up` each install, download and serve differently;
   Condor's model runs through its own commands (`qwenserve`, `qwenget`, `qwenbench`, `qwenmcp`).
6. **Multi-GPU is sequential.** The layer split passes activations through host memory with one GPU busy at a time; tensor
   parallel exists for four MoE families in development builds only, with its own gather kernel and no NCCL.
7. **Formats.** No GGUF loading, so the Ollama/llama.cpp model library is not reachable; no constrained decoding (JSON
   schema, grammars are refused with a 400).

### Starlings and Condor, specifically

- **Starlings today** (`starlings/docs/beta.md`): `starlings up` / `join` with one-way join and approval codes, mutual
  TLS over TCP (rustls), credit-based flow, opt-in mDNS hints that never enroll anything; each stage runs
  `glyd-engine rangeworker` over a framed stdio protocol (`docs/range-worker.md`), bit-exact against the whole model.
  Limits: 2 GPUs, 2 machines, Qwen3.5 family, 4,096 context by default, 1-2 sessions, private addresses only.
- **Condor today**: `qwenserve` with GPU expert slots, RAM and disk tiers, CPU misses, MTP drafting, 1-8 users,
  2-GPU split, OpenAI + Anthropic, `qwenmcp`, the chat app; Linux and Windows.
- **Together:** the `rangeworker` contract generalizes from Qwen3.5 to every family the engine runs; Condor's expert cache
  (`offload.rs`: LRU with promotion, copy/CPU/split per miss by measured cost) becomes the planner's MoE policy on every
  tier, including a remote machine's RAM; Starlings' enrollment and transport carry it between machines.

---

## 2. What the product looks like

### One program

```
glyd run Qwen/Qwen3.5-27B            # picks the level, context and placement for this machine; chat here and at :8000
glyd run Qwen/Qwen3.8-Flash-Next     # a 180B MoE on a 16 GB card: hummingbird, experts in RAM (what Condor does now)
glyd serve MODEL [--users N]         # OpenAI + Anthropic APIs, MCP, the chat page, /metrics
glyd pull | ls | rm | ps             # the local model library (Hugging Face cache layout; glyd/* pre-packed repos)
glyd doctor                          # this machine: GPUs of every vendor, RAM, disks, links, what fits
glyd fit MODEL                       # every level against this machine: memory, expected speed, quality (KL)
glyd cluster up | join | status      # Starlings: one model over this machine's GPUs and other machines'
glyd bench [--against URL]           # speed and quality here, or against any OpenAI-compatible server
glyd pack | verify                   # the packer and the check, on the CPU
```

One binary per platform: Linux (x86_64, aarch64), macOS (Apple Silicon), Windows (x86_64). Installed by `curl | sh`,
Homebrew, winget/scoop, `cargo install`, a Docker image per vendor, or `pip install glyd` (a wheel that holds the binary and
no Python, as ruff and uv ship).

### Levels chosen, not configured

`glyd run MODEL` reads the model's cost profile (`glyd-engine costs`: bytes per layer and per token) and the machine's
measured rates (`glyd-engine rates`: each memory tier's bandwidth), and picks:

1. **penguin** (lossless) if it fits the GPUs with the asked context;
2. else the highest lossy level that fits (kestrel → swift → wren-class → hummingbird), saying what it gives up (KL,
   one eval) in one line;
3. else placement across tiers (below), saying the expected tokens/s before downloading.

`MODEL:level` pins it. The user never sets a layer count, a quantization type or a memory fraction.

### One placement planner: Condor and Starlings are the same idea

Condor puts the busiest experts on the GPU and the rest in RAM, and computes misses on the CPU. Starlings puts layer ranges on
other GPUs and other machines. Both answer the same question: **where does each piece of the model live, and where is it
computed?** They become one planner over a list of tiers, each with a measured bandwidth, latency and capacity:

```
this GPU's VRAM  →  this machine's other GPUs (P2P / NVLink / PCIe)  →  host RAM (CPU computes there)
                 →  other machines' GPUs (Starlings, TLS over TCP; RDMA where present)  →  NVMe
```

- Dense layers are placed by contiguous ranges (pipeline), or split across GPUs in a box (tensor parallel) where the link
  is fast enough.
- MoE experts are placed by use: hot experts near compute, cold experts further out, each at the bits its use earns
  (hummingbird's rule, generalized).
- KV cache and recurrent state are placed with the layers that use them; lossless KV compression applies on every tier.
- The plan is printed before anything downloads: "27B swift: layers 0-31 on this 4090, 32-63 on 192.168.1.20's 4090,
  ~44 tokens/s".

### Every vendor, one engine

The engine's model code and scheduler are backend-neutral Rust. A backend trait owns memory, streams, graphs, kernels and
collectives:

| Backend | Kernels | Libraries | Notes |
| :--- | :--- | :--- | :--- |
| CUDA (NVIDIA) | `kernels/*.cu`, hand-tuned (mma.sync, cp.async, wgmma on Hopper) | cuBLAS(Lt) where present; NCCL optional | The reference backend; what exists now |
| ROCm/HIP (AMD) | The same `.cu` sources through `gpu/hip.h`; wave-size-generic rewrite for CDNA (wave64) and MFMA/WMMA paths for the tensor products | hipBLASLt, RCCL | RDNA 3/4 builds today; Instinct needs the wave64 work |
| Metal (Apple) | MLX through its C API (`mlx-c`) from Rust for matmul, quantized matmul, attention and norms; Glyd's own decode kernels as custom Metal kernels | MLX | Unified memory: the "host RAM" tier is the GPU's own memory, so placement is simpler and large MoE models fit |
| CPU | AVX2, AVX-512, NEON (exist for experts); extended to whole models | — | The fallback, and the tier for cold experts |

A kernel exists once per backend for the hot paths (packed-weight products, attention, MoE dispatch, sampling). For the long
tail (norms, rotary, elementwise), a portable path written once in Rust (evaluate CubeCL, which targets CUDA, ROCm, Metal and
Vulkan) so a new model does not need four hand ports.

### Serving, from one user to thousands

What a datacenter needs, in the same binary:

- continuous batching with a paged KV cache (lossless-compressed pages), chunked prefill, prefix caching;
- speculative decoding (draft models, MTP heads: both exist for some models today);
- structured output (JSON schema, grammars), tools, logprobs (tools and logprobs exist);
- tensor, pipeline and expert parallelism; prefill/decode disaggregation for large deployments;
- Prometheus metrics, health and readiness, a Helm chart and Docker images per vendor.

And what a person at home needs: one command, a chat page, OpenAI and Anthropic APIs on localhost, MCP for coding agents,
nothing to configure.

---

## 3. Where this sits against the others

What each is, as of November 2026. Competitor columns are from their documentation and public reporting (sources at the
end); "today" for Glyd is this repository.

| | Ollama | llama.cpp | vLLM | SGLang | exo | mistral.rs | TensorRT-LLM | **Glyd today** | **Glyd target** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Built for | Local, one user | Local, everyone | Datacenter | Datacenter | Clusters of Macs | Local/server, Rust | NVIDIA datacenter | Local and single-node serving | Local to datacenter |
| Language | Go + ggml/llama.cpp, MLX on Mac | C/C++ | Python + CUDA/Triton | Python + CUDA | Python + MLX | Rust | C++/Python | Rust (+ Python tool) | Rust only |
| Install | One command | Build or binaries | pip / Docker | pip / Docker | App | cargo / binaries | Containers | One command | One command, every OS |
| NVIDIA | Yes | Yes | Yes | Yes | Community fork | Yes | Yes (only) | Yes | Yes |
| AMD | ROCm, Vulkan | ROCm, Vulkan | ROCm | ROCm | No | Partial | No | Builds (RDNA) | RDNA + Instinct |
| Apple | Metal, MLX | Metal | No | No | MLX (primary) | Metal | No | No | MLX/Metal |
| Big MoE on a small GPU | Layer offload | `--cpu-moe`, `-ot` | Limited | Limited | Spread over Macs | Partial | No | **Experts placed by use, CPU computes misses, bits by use (Condor)** | Every MoE |
| Multi-GPU in a box | Layer split | Layer/row split, TP experimental | TP/PP/EP | TP/PP/EP/DP | — | Yes | TP/PP/EP | Layer split (sequential); TP in development builds | TP/PP/EP |
| Multi-machine | No | RPC over TCP | Yes (Ray) | Yes | Yes (TB5 RDMA on Mac) | No | Yes | Starlings beta: 2 machines, pipeline | Any number, consumer and datacenter |
| Weights | GGUF quants (lossy) | GGUF quants (lossy) | FP8/INT4/AWQ/GPTQ (lossy) | Same | MLX quants | ISQ/GGUF | FP8/FP4/INT4 | **Lossless (-33%)** + lossy levels | Lossless + lossy + GGUF import |
| KV cache | Quantized option | Quantized option | FP8 option | FP8 option | — | — | FP8/INT8 | **Lossless, ~12 bits** | Same, on every backend |
| Many users | Few | Slots | Thousands | Thousands | Few | Some | Thousands | 1-8 (Qwen3: paged KV, prefix cache); 1 for other families | Thousands |
| Structured output | JSON schema | Grammars | Yes | Yes | — | Yes | Yes | No (refused) | Yes |
| APIs | Own + OpenAI | OpenAI | OpenAI | OpenAI | OpenAI | OpenAI | OpenAI (via Triton/Dynamo) | OpenAI, Anthropic; MCP for Condor | Same, MCP for every model |

**What only Glyd does:** the same model in a third less memory with nothing changed (lossless weights and KV cache); a model
larger than the GPU by placing experts by use instead of by layer; one placement planner from one card to many machines.

**Where Glyd is behind and must catch up:** model coverage (vLLM and llama.cpp run hundreds of architectures), concurrency
(8 users against thousands), Apple and AMD, the GGUF library Ollama users already have, and the operational surface
(metrics, Kubernetes) datacenters expect.

**How to win each group:**

- *Ollama/LM Studio users*: same one-command experience, plus models that do not fit Ollama on the same card (lossless
  saves a third; Condor-style placement runs a 180B MoE on 16 GB). GGUF import so their library comes along.
- *vLLM/SGLang operators*: the same model on fewer GPUs (Qwen3-32B: 1 GPU instead of 2; Qwen2.5-72B: 3 instead of 4),
  bit-exact outputs, one static binary instead of a Python environment. Needs batching parity first (phase 3).
- *exo users and home-lab clusters*: NVIDIA and AMD as well as Macs, a secured cluster (enrollment, TLS), and the planner
  using every tier.

---

## 4. Rust only: what happens to the Python

About 53,500 lines of Python outside `benchmarks/` and `research/` (another ~28,000 there are analysis records of past
runs and stay as records). By area:

| Python | Lines | What it does | Rust replacement | Action |
| :--- | ---: | :--- | :--- | :--- |
| `glyd-gpu-py/glyd_gpu`: `run.py`, `preflight.py`, `engine.py`, `_cli.py`, `chat.py`, `page.py`, `fit.py` | ~3,100 | `glyd run/serve/doctor/login`, choosing the engine or vLLM, terminal chat, memory fit | The engine already has `preflight`, `doctor`, `login`, `fetch`, `serve`, the chat page. Port the terminal chat and `fit`; the Rust `glyd` calls the engine directly | **Port, then delete** |
| `vllm_*.py` (plugin, KV backend, scheduler, FP8, quant) + `gpu/vllm/` | ~10,700 | Glyd inside vLLM | None possible: vLLM's plugin API is Python | **Freeze at v0.32, delete when the engine covers what it serves** (phase 3 exit) |
| `hf.py`, `model.py`, `moe.py`, `kernels.py`, `_lib.py`, `_trace.py`, `format.py` | ~4,600 | `glyd.from_pretrained` for PyTorch/transformers | The C ABI of `libglyd_gpu` stays for anyone who binds it; the engine replaces the use case | **Delete** (a community package can live outside) |
| `kvcal.py`, `f8codec.py`, `fp8ref.py` | ~1,000 | KV calibration, FP8 codec reference | `glyd-engine/src/kvcal.rs` exists; FP8 codec into `glyd-gpu` | **Port the codec, delete** |
| `gpu/*.py`, `gpu/fp8/`, `gpu/experimental/`, `glyd-engine/tools/` (incl. `ref_vllm.py`) | ~15,300 | Kernel checks, end-to-end checks, references against vLLM | `cargo test` integration tests; references from the engine's CPU reference modules (`*_ref.rs` exist for Qwen3.5, Kimi, DeepSeek-V4, Muse, MoE) and stored reference logits | **Port as tests** |
| `glyd-gpu-py/tests/` | ~9,200 | The package's tests | Goes with the package | **Delete with it** |
| `bindings/python/` | ~530 | `pip install glyd` | maturin `bindings = "bin"`: the wheel holds the Rust binary only | **Replace** |
| `condor/bench/`, `condor/tests/` | ~750 | Speed/quality comparison, server tests | `glyd bench --against URL`; Rust integration tests. (GSM8K/MMLU-Pro via inspect-ai stay an external tool) | **Port** |
| `scripts/*.py`, `huggingface/`, `tools/hf-prepacked/`, `tools/cards/` | ~2,100 | Version bumps, model cards, Hub uploads | `cargo xtask` | **Port** |
| `starlings/docs/` scripts | ~2,500 | Evidence generation | `cargo xtask` or drop | **Port or drop** |

The CI that checks Python goes with it. When phase 1 ends, no command starts Python unless the user installed the frozen
vLLM fallback (`GLYD_WITH_VLLM=1`); when phase 3 ends, the fallback is gone.

---

## 5. The order of the work

Each phase ends on measurable exit criteria; nothing is "done" without a benchmark file in `benchmarks/`.

### Phase 0: one open repository (this change)

Done in this branch: the four repositories merged, Apache-2.0 everywhere, no compiled-only parts, release builds from this
source, engine bundles attached to releases by CI.

### Phase 1: one program, no Python on the user's path

- `glyd` (Rust) runs `run/serve/doctor/login/pull/ls/ps/fit` itself through the engine; terminal chat in Rust.
- Condor folds into `glyd run`: `qwenserve/qwenget/qwenbench/qwenmcp` become `glyd run|pull|bench|mcp` for any model with
  a hummingbird pack; `condor/` keeps its docs and results.
- Starlings' CLI becomes `glyd cluster`; its crates move into the main workspace (edition 2024 per crate).
- Distribution: install.sh, Homebrew, winget, Docker (CUDA), PyPI binary wheel; Windows built by the release workflow.
- The vLLM plugin stays installable as `glyd-gpu` v0.32.x, unchanged, for models the engine does not run.

**Exit:** on a clean Linux, Windows and macOS machine, `curl | sh` then `glyd run` reaches a chat with no Python installed;
`glyd run` of every model in the README works or says why in one line.

### Phase 2: the backend layer, then every vendor

First, because every model added after it runs on every vendor, where a model added before it is written for CUDA and
ported three times. Tracks can run in parallel (one owner each), the backend layer first.

- **Backend layer:** the engine's device code (`dev.rs`, `ffi.rs`, `kern.rs`; HIP already enters through the same driver
  table in `ffi.rs`) becomes a trait: memory, streams, events, graphs, kernel launch, collectives. CUDA is the first
  implementation and must not lose a token/s (checked by `glyd bench` before and after).

- **AMD RDNA 3/4:** run `amd_job.sh`, fix what it finds, add to CI (a rented runner), publish numbers.
- **AMD Instinct (CDNA 3/4):** wave64-generic kernels, MFMA products, hipBLASLt for long prompts, RCCL for multi-GPU.
- **Apple Silicon:** the Metal backend through `mlx-c`; Glyd's lossless and lossy formats as custom Metal kernels; the
  placement planner treats unified memory as one tier.
- **CPU:** whole models on CPU (AVX-512, AVX2, NEON paths extended from the expert kernels).
- **Portable kernels:** evaluate CubeCL for the long tail so a new model is written once.

**Exit:** the same `glyd run` on an RX 7900 XTX, an MI300X, an M-series Mac and a CPU-only machine, each with a
benchmark file; quality checked against the CUDA run (KL) for each.

### Phase 3: the engine covers what vLLM covered, and serves many users

Model definitions written against the backend layer, so each lands on NVIDIA, AMD and Apple together.


- Model coverage: Llama 3.x/4, Mistral/Mixtral, Gemma 2/3/4, Qwen 2/2.5/3/3.5/3.8, Phi-4, GPT-OSS, DeepSeek V3/R1/V4,
  GLM, Kimi, Nemotron, SmolLM; vision for the families that have it. A model definition is a Rust file plus a weight map,
  checked against a CPU reference.
- Scheduler: the Qwen3 path's scheduler (paged lossless KV, prefix cache, chunked prefill, preemption) made the one
  scheduler for every family, with the 8-user cap removed; 256+ concurrent requests on one GPU; KV context past 40,960.
- Structured output (JSON schema, grammar), `n`, `best_of`, everything the engine refuses today with a 400.
- GGUF import: load a GGUF model (llama.cpp's quant types decoded to the engine's formats, or run as is).
- Retire the vLLM plugin.

**Exit:** on an H100 and an RTX 4090, the engine's throughput at 1, 8, 64 and 256 concurrent requests is at least vLLM's
on the same model with less memory, for the 10 most-downloaded open models; no model `glyd run` handled in v0.32 is lost.

### Phase 4: one model, any number of machines

- The placement planner (section 2) replaces Condor's fixed policy and Starlings' fixed split.
- Starlings: N machines, any vendor mix that shares a format, tensor parallel inside a box, pipeline across boxes, failure
  of a stage retried on the rest; RDMA (RoCE, Thunderbolt) where present.
- Datacenter: expert parallelism, prefill/decode disaggregation, Kubernetes (Helm chart, operator), Prometheus metrics.

**Exit:** a 2-machine consumer setup (two 16 GB cards) runs a model neither runs alone at a speed it prints before it
starts; an 8-GPU node serves a 1T+ MoE (Kimi-K3 class) at datacenter batch sizes.

### Phase 5, always on: performance and proof

- `glyd bench` is the one harness; CI runs it on real GPUs of each vendor and refuses a regression.
- Public results against Ollama/llama.cpp on consumer cards and vLLM/SGLang on datacenter GPUs, same model, same
  hardware, logs in `benchmarks/`.

---

## 6. Decisions for the owner

1. **When the vLLM plugin stops.** Recommended here: frozen now (fixes only), deleted at the end of phase 3.
   [ADOPTION.md](ADOPTION.md#6-two-earlier-decisions-reconsidered) argues for the alternative: replace its Python logic with
   a thin shim over the Rust/C library, as a channel into vLLM and SGLang deployments while the engine catches up.
2. **pip.** Recommended: keep `pip install glyd` as a binary-only wheel; drop `glyd.from_pretrained`.
3. **Apple backend route.** Recommended: MLX through `mlx-c` (fast path to good speed), custom Metal kernels for Glyd's
   formats; revisit direct Metal if MLX limits the formats.
4. **Contributor terms.** The CLA (relicense rights to the licensor) stays as it is; a DCO sign-off alone is the more
   common choice for Apache-2.0 projects that want outside contributors.
5. **What the first public release of the open repository is called** (v0.33.0, or v1.0 at the end of phase 1).

---

Sources for the comparison: [Ollama: new model scheduling](https://ollama.com/blog/new-model-scheduling),
[Ollama 0.17 coverage](https://www.webpronews.com/ollama-0-17-arrives-with-massive-performance-gains-and-a-new-architecture-that-could-reshape-local-ai-deployment/),
[Ollama 0.30](https://webscraft.org/blog/ollama-030-scho-novogo-gguf-vulkan-llamacpp-i-tool-calling?lang=en),
[Ollama on MLX](https://registry.ollama.ai/blog/mlx-performance),
[exo](https://docsearch.algolia.com/mcp/docs/repo/exo-explore/exo),
[exo in 2026](https://dev.to/jovan_chan_9500711396d4e6/exo-framework-in-2026-can-you-pool-rtx-3090s-to-beat-a-dgx-spark-the-honest-distributed-inference-5ccm),
[vLLM disaggregated serving](https://vllm.ai/blog/2026-09-29-disaggregated-serving-guide),
[SGLang vs vLLM (2026)](https://www.morphllm.com/comparisons/sglang-vs-vllm),
[llama.cpp RPC](https://sources.debian.org/src/llama.cpp/8064%2Bdfsg-1/tools/rpc/README.md),
[llama.cpp MoE offload](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide),
[TensorRT-LLM distributed inference](https://www.mintlify.com/NVIDIA/TensorRT-LLM/deployment/distributed-inference).
mistral.rs and TensorRT-LLM columns are from their projects' documentation as of mid-2026 and should be rechecked before
publishing a comparison.
