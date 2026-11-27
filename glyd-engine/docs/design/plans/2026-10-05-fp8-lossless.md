# Lossless FP8 weights on the GPU: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development. Steps use checkboxes (`- [ ]`).

**Goal:** `--quantization glyd` on 2026 FP8 checkpoints. Their FP8 weights are held about 16–18% smaller with every byte exact, and they are served at least as fast as vLLM's FP8.

**Architecture:** a fixed-rate lossless code for e4m3 (Task 1). Our own pack, unpack and decode kernels, with a fused decode-and-FP8-matmul for decode steps (Tasks 2 and 5). A wrapper around vLLM 0.30's FP8 Linear and MoE methods that packs at load and decodes per layer for prompts (Tasks 3 and 4).

**Spec:** `glyd-engine/docs/design/specs/2026-10-05-fp8-lossless-design.md`.

`$S` is the scratch directory of the run (not kept).

## Global constraints

- **Lossless means bit for bit.** No check may be loosened.
- **Our own kernels only.** vLLM's FP8 GEMM and fused-MoE are called as installed, never copied. No third-party kernel code.
- **Compute:**
  - The Mac compiles CUDA in Docker: `nvidia/cuda:13.0.2-devel-ubuntu24.04`, `-Xptxas -v`.
  - GPU runs are Runpod pods through `$S/rp_run.sh`, with `$S/rp_guard.sh` running. No pod outlives its job.
  - Dev GPUs are an RTX 4090 or L40S (Ada) and an H100 (Hopper).
- **BF16 must not change:** the existing checks keep passing (`python gpu/vllm/check_vllm.py --kernels-only`, `glyd-gpu-py/tests/*`).
- **Public material** says what was done and the measured results, never how. Designs stay here.

---

### Task 1: The FP8 format, chosen by measurement, with its Python reference

**Model:** Sonnet. CPU only.

**Files:**
- Create: `gpu/fp8/fetch_samples.py`, `gpu/fp8/fmt_study.py`, `glyd-gpu-py/glyd_gpu/fp8ref.py`, `glyd-gpu-py/tests/test_fp8ref.py`, `gpu/fp8/README.md` (the results and the chosen layout).

**Interfaces it produces:**
- `fp8ref.pack(w: np.ndarray[uint8, (O, K)]) -> Packed`, where `Packed` holds the arrays and the `nbytes()` the GPU format will take.
- `fp8ref.unpack(p: Packed) -> np.ndarray[uint8, (O, K)]`.
- `fp8ref.TILE`, the tile shape the GPU kernels decode as a unit.
- The byte layout, written in `fp8ref.py`'s docstring field by field, which Task 2's CUDA follows.

**Steps:**
- [ ] Fetch about 10 FP8 weight tensors per repo with `gpu/fp8/fetch_samples.py`: attention, MLP, and experts where there are any. Use a slice of 8–16 MB each, from layers chosen at random with seed 0.
  - **Repos:** zai-org/GLM-5.3-Flash, Qwen/Qwen3.8-27B-FP8, MiniMaxAI/MiniMax-M3-MXFP8, mistralai/Mistral-Small-4-119B-2603, deepseek-ai/DeepSeek-V4.1-Flash.
  - Reuse `$S/fmt/fmt_measure.py`'s `tensors`, `classify` and `get` (HTTP range reads; the repos are public).
  - Save `.npy` files plus a `manifest.json` (repo, tensor, shape, slice rows) under `$S/fp8data/`.
- [ ] In `fmt_study.py`, measure at least these candidates on every sample. Report the total bits per value, including tile headers, escapes and raw tiles:
  - **(a)** sign + k-bit exponent offset from a per-tile window + 3 mantissa bits, with escapes, for k = 2 and 3.
  - **(b)** a two-tier code in the style of the BF16 tiered layout (read `pack_mma` and its tier code in `glyd-gpu-py/glyd_gpu/kernels.py`), adapted to e4m3's 4-bit exponent.
  - **(c)** the per-tile window as against a per-tensor or per-128-row window.
  - **(d)** the order-0 floor, for reference.

  Tile sizes to try: 16×128, 64×128 and 128×128 values. Escapes per tile are bounded; past the bound the tile is stored raw.
- [ ] Choose the format with the fewest bits whose decode is a fixed-rate read of each tile, at most about 12 integer operations per value, with no data-dependent loop longer than the escape list. Record the decision and the table in `gpu/fp8/README.md`.
- [ ] Write `fp8ref.py` (numpy) for the chosen format: `pack`, `unpack` and the docstring layout.
- [ ] Write `test_fp8ref.py`:
  - all 256 e4m3 byte patterns, NaN 0x7F and 0xFF and both zeros included, through pack and unpack;
  - random tensors of shapes 1×1, 17×33, 128×128 and 4096×7168;
  - a tile that is all escapes, which falls back to raw;
  - an all-zero tile;
  - every fetched real sample, bit for bit.

  Run `python glyd-gpu-py/tests/test_fp8ref.py`. Expected: all pass.
- [ ] Commit: `fp8: the lossless FP8 format, chosen on 2026 checkpoints, with its reference`.
- [ ] **Report:** bits per value for each candidate per repo, the choice, and the savings on the samples against the floor.

### Task 2: CUDA pack, unpack and decode kernels for the format, with C API and Python bindings

**Model:** Sonnet, with Opus for review.

**Files:**
- Modify:
  - `gpu/glyd_gpu.cu`: a new namespace `f8c` with `glyd_gpu_f8_pack`, `glyd_gpu_f8_unpack` and `glyd_gpu_f8_decode_tiles` (decode a list of tiles into an e4m3 scratch);
  - `gpu/glyd_gpu.h`: the C API version goes up by one, with its history line;
  - `glyd-gpu-py/include/glyd_gpu.h`;
  - `glyd-gpu-py/glyd_gpu/_lib.py`: the `_ARGS` rows and the API version;
  - `glyd-gpu-py/glyd_gpu/kernels.py`: `f8_pack(w)`, `f8_unpack(p)` and `f8_decode(p, out)` on torch tensors.
- Tests:
  - `glyd-gpu-py/tests/test_gpu.py` `test_c_header` must pass;
  - new `gpu/fp8/check_f8.py`: every case of `test_fp8ref.py` on the GPU, with the GPU's bytes equal to `fp8ref`'s.

**Steps:**
- [ ] Implement it, matching `fp8ref`'s byte layout exactly.
- [ ] Compile with `-Xptxas -v` for sm_89 and sm_90a in Docker on the Mac: 0 spills.
- [ ] Run on an Ada pod: `python gpu/fp8/check_f8.py`. Expected: all bit for bit.
- [ ] Measure the unpack bandwidth on an H100 pod (GB/s of e4m3 written) and record it.
- [ ] Commit.

### Task 3: vLLM FP8 Linear layers through Glyd (prompt path, exact)

**Model:** Sonnet.

**Files:**
- Modify: `glyd-gpu-py/glyd_gpu/vllm_plugin.py` (GlydConfig's handling of a checkpoint's FP8 `quantization_config`).
- Create: `glyd-gpu-py/glyd_gpu/vllm_fp8.py` (`GlydFp8LinearMethod` wraps vLLM 0.30's `Fp8LinearMethod`, block and per-tensor, and the compressed-tensors and modelopt FP8 schemes).
- Tests: `glyd-gpu-py/tests/test_vllm.py`, for the config detection, the refusals and the wrapping, with no GPU.

**Behaviour:**
- After the inner method's `process_weights_after_loading`, the inner method's e4m3 `layer.weight` (which vLLM may have repacked: record which layout each scheme leaves) is packed with `kernels.f8_pack` and freed.
- `apply()` decodes it into a per-device scratch, sized for the largest layer, and calls the inner `apply()` with the scratch as `layer.weight`.
- Scales are untouched.
- `exact` on a per-tensor or per-channel checkpoint is refused under torch.compile (inductor) unless vLLM's quantization op is on (`custom_ops +quant_fp8`) or the run is eager: the reason is stated in the refusal (the quantization is torch ops that inductor fuses into vLLM's graph).
- MLA's `kv_b_proj` stays vLLM's own method: vLLM reads its weight directly after the load.

**Gates (pods):**
- **Exact:** logits bit for bit vLLM FP8's, on the real first 4 layers of Qwen3.8-27B-FP8 and of GLM-5.3-Flash's dense layers (Ada and H100).
- **Size:** weights as vLLM logs them at least 15% under FP8's on those Linears.

**Steps:**
- [ ] Write `gpu/fp8/make_slice.py`. It builds a few-layer model from a real checkpoint (config `num_hidden_layers` = N), fetching only the shards that hold layers 0..N-1 and the embedding and head, with their real FP8 weights and scales.
- [ ] Commit.

### Task 4: FP8 mixtures of experts

**Model:** Sonnet, with Opus for review.

**Files:**
- Modify: `vllm_fp8.py`, adding `GlydFp8MoEMethod`, which wraps vLLM 0.30's FP8 FusedMoE method (and MXFP8's).

**Behaviour:**
- At load, every expert's e4m3 weights are packed.
- In `apply()`, the step's routed expert ids (from vLLM's router, before the experts run) are deduplicated.
- Only those experts are decoded, into a compact scratch for the step's at most `num_experts_per_tok * tokens` experts. The ids are remapped, and vLLM's own FP8 fused-MoE is called on the compact set.

**Gates:**
- **Exact:** bit for bit vLLM FP8's logits on the real first layers of MiniMax-M3-MXFP8, GLM-5.3-Flash and DeepSeek-V4.1-Flash's FP8 experts.
- **Expert parallel / tensor parallel 2:** load and run.

**Steps:** commit.

### Task 5: Fused decode and FP8 matmul for 1–64 tokens (the speed gate)

**Model:** Opus. This is the kernel design.

**Files:** `gpu/glyd_gpu.cu` (`f8c::gemm_step` for Ada `mma.sync.m16n8k32.e4m3`, and a Hopper path), the bindings, and `gpu/fp8/bench_f8.py`. The bench compares against vLLM's FP8 GEMM, which on Hopper is CUTLASS or DeepGEMM, at the shapes of the 5 models for 1, 8, 32 and 64 tokens.

**Gates:**
- Outputs within the FP8 GEMM's own accumulation tolerance, with the weights exact.
- Decode-step time at most vLLM FP8's at every shape on an RTX 4090 or L40S and an H100.
- The vLLM step path uses it for 64 tokens or fewer, and for routed experts at decode.

**Steps:** commit.

### Task 6: Multi-GPU and whole-model runs

**Model:** Sonnet.

**Runs:**
- Qwen3.8-27B-FP8, whole, on one H100: `vllm bench serve` against vLLM FP8 at 1k and 8k tokens, the KV and weights lines, and a quality probe.
- Mistral Small 4 at tensor parallel 2 on two GPUs.

**Records** go to `benchmarks/gpu/` (private first).

**Gates:** the spec's speed and size gates.

**Steps:** commit.

### Task 7: `auto`, the docs, and the release preparation

- `GlydConfig` turns on for FP8 checkpoints where every gate passed; elsewhere it refuses with why.
- `exact` on per-tensor / per-channel FP8 is refused under inductor without `custom_ops +quant_fp8`, as Task 3 does; the docs state the same rule.
- Public CHANGELOG and README lines (what was done and the results only) for the next release.
- The site goes through the usual release flow.

---

## Owner rulings, 2026-10-05: whole models and data centers

These amend Tasks 4, 6 and 7. The owner said: "support for everything, not just a few layers", "the datacenter levels as well", and approved about $120 of whole-model runs, made only after Task 5's speed gate passes.

**Whole models, every tensor.**
- Each model gets an inventory (`gpu/fp8/inventory.py`, from the checkpoint's headers): every tensor by kind and dtype, with its bytes and status.
- Status is either *packed* or *kept as vLLM's*. A kept tensor needs a stated reason and no silent skips.
- Packed covers:
  - every FP8 tensor: attention projections, MLA projections other than `kv_b_proj` (see Task 3's review M10), DeepSeek's indexer, dense MLPs, routed and shared experts;
  - the BF16 embedding and LM head, through Glyd's existing BF16 path, as in v0.28.
- Slices are for development only. The release gate is the whole model served.

**Data-center coverage.**
- GPUs:
  - Ada and Hopper (H100, H200), where the fused kernel runs;
  - B200 (sm_100: compile, check, bench);
  - A100, where FP8 has no tensor cores and vLLM serves FP8 through Marlin weight-only. Our path decodes to bf16 inside our own kernel, so FP8 checkpoints get smaller on A100 too.
- Tensor parallel 2, 4 and 8, and expert parallel for MoE.
- Prefix caching, chunked prefill, vLLM's default CUDA graphs, and `--kv-cache-dtype fp8` beside our weights.

**Whole-model runs, Task 6.** Each run uses `RP_GPUS` = N in `rp_run.sh`. Every run reports weights GiB, KV tokens, and `vllm bench serve` at 1k and 8k tokens against vLLM FP8, saturated and at 1 request a second, plus a quality probe.

| Model | Pod |
| :--- | :--- |
| Qwen3.8-27B-FP8 | 1×H100 |
| Mistral Small 4 | 2×H100, TP 2 |
| GLM-5.3-Flash, 328 GB | 4×H200, TP 4 |
| MiniMax-M3-MXFP8, 444 GB | 4×B200 or 8×H100 |
| DeepSeek-V4.1-Flash, 510 GB | 4×B200 or 8×H100 |

Then:
- one B200 check (1 GPU) of the kernels and of the vLLM path;
- one A100 check (1 GPU).

**Engines.** SGLang follows vLLM, as the next adapter over the same C API.
