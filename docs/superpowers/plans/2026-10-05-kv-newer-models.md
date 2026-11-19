# The lossless KV cache for newer models: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to carry out this plan task by task. Steps use checkboxes (`- [ ]`) for tracking.

**Goal:** `kv: auto` holds the KV cache of Qwen3.8-style models, which have head size 256 and are hybrid (linear-attention layers beside full-attention ones), up to 131,072 tokens, with every value exact.

**Architecture:**
- The `kvc` CUDA namespace gets a template head size (128 or 256), reached through C API 16.
- The vLLM plugin serves only the full-attention layers of a hybrid model. It maps vLLM's manager blocks to the kernels' 16-token pages and clears a freed block before any other cache group can take it.
- The calibration's long tiers leave the dev gate.

**Tech stack:**
- CUDA C++20 in `gpu/glyd_gpu.cu`, built by `gpu/build_lib.sh` with nvcc 13.
- Python 3.12 and vLLM 0.30.0.
- ctypes bindings in `glyd-gpu-py/glyd_gpu/_lib.py`.

**Spec:** `docs/superpowers/specs/2026-10-05-kv-newer-models-design.md`. The code map, with every file:line this plan cites, is `$S/kvnew-map.md`.

`$S` = `/private/tmp/claude-501/-Users-suryakoritala-alatirok/a9feedc0-e067-4d76-bca6-e4b6b9538345/scratchpad`.

## Global constraints

- **Private work.** This is private work in glyd-gpu-private, branch `kv-newer`. Push to the `private` remote only, never to `origin`.
- **Commits.** Identity `git -c user.name="Surya Koritala" -c user.email="suryakoritala1324@gmail.com" commit -s -m "<one line>"`. No AI attribution.
- **Exact means exact.** Every value read back is the value written. No check may be loosened to pass.
- **Head size 128 must not change.** Its outputs must stay the same bit for bit: the 81 kernel checks (`check_vllm.py --kv --kernels-only`) pass unchanged at every task, and the decode plan's bits for D = 128 do not move.
- **Our own kernels only.** No third-party kernel code or dependencies; FlashAttention stays vLLM's installed one, called, not copied.
- **Compute.** The Mac compiles in Docker for free. GPU runs are short Runpod pods through `$S/rp_run.sh`, under `$S/rp_guard.sh`; no pod outlives its job. The owner approves the GPU budget before the first pod (see Budget). There is no AWS dev box any more, and tyca-prod in ca-central-1 is never touched.
- **Public material.** The public docs and changelog say what was done and the measured results, never how. Kernel and format details stay in this repo.

## Budget, asked of the owner before Task 3's first pod

- **Dev pods** for Tasks 3–9, each a session of 30–90 minutes:
  - an L4 or RTX 4090 for Ada at about $0.4–0.7/h;
  - an H100 at $2.69/h, for its own paths only.
- **Estimate:** about 8 h of Ada and 2 h of H100 for development, plus the Task 10 run (about 40 min of H100). That is **about $12–15 in all**.

## How to compile and run

- **Compile on the Mac in Docker,** no GPU needed:
  ```bash
  docker run --rm -v /Users/suryakoritala/glyd-kvnew:/src -w /src/gpu nvidia/cuda:13.0.2-devel-ubuntu24.04 \
    bash -c 'nvcc -O3 -std=c++20 --expt-relaxed-constexpr -Xptxas -v -gencode arch=compute_89,code=sm_89 -c glyd_gpu.cu -o /tmp/k.o 2>&1 | grep -E "kvc|error" | head -80'
  ```
  - Use `compute_90a,code=sm_90a` for the H100.
  - The `grep` shows the `kvc` kernels' registers, spill stores and loads, and shared memory.
  - If that image tag is missing, use the newest `13.0.*-devel-ubuntu24.04` with an arm64 manifest.
- **A GPU job:** a job script that untars the tree, runs `bash gpu/build_lib.sh`, and runs the checks into `~/results`, launched with:
  ```bash
  GPU="NVIDIA L4" MAX_MIN=60 NAME=<name> UPLOAD="$S/kvnew_src.tar" bash $S/rp_try.sh <job.sh>
  ```
  Results land in `$S/ceiling/results-<name>`. The template job is `$S/ceiling/qwen38_job.sh`; reuse its environment section, which installs vLLM 0.30.0 and torch with CUDA 13.
- **The tree tarball:**
  ```bash
  cd /Users/suryakoritala/glyd-kvnew && git archive -o $S/kvnew_src.tar HEAD
  ```

---

### Task 1: Head size as a template parameter (no change at D = 128)

**Files:**
- Modify: `gpu/glyd_gpu.cu`, namespace `kvc` (cu:4398-5860) and its host entries (cu:7556-7738).

**Interfaces:**
- Produces: every `kvc` kernel and helper takes `template <int D>` (128 or 256); `kvc::Dims<D>` holds `TILE = 24*D`, `HEADPAGE = 48*D`, the tile's low-byte and code offsets, and the K-step and V-group counts (`D/16`, `D/8`).
- The host keeps refusing every D but 128 (`kv_shape_ok` cu:7616, cu:7686, cu:7700) until Task 3.

- [ ] **Step 1: Run the baseline register report.** Compile with `-Xptxas -v` for sm_89 and for sm_90a, and save the `kvc` lines to `$S/kvnew/regs-before-89.txt` and `-90a.txt`.
- [ ] **Step 2: Turn the constants into a template.**
  - Replace `constexpr int D = 128, PT = 16, TILE = 3072, HEADPAGE = 2*TILE;` (cu:4431) with `template <int D> struct Dims { static constexpr int TILE = 24*D, HEADPAGE = 48*D, KSTEPS = D/16, VGROUPS = D/8, CODE_OFF = 16*D, PLANE = 8*D; };` and keep `PT = 16`.
  - `static_assert(NW*32 == D)` (cu:4433) moves into the decode kernel as `static_assert(NW*32 == 128)`. Task 4 replaces the merge that needs it.
- [ ] **Step 3: Make every user of `D`, `TILE` or `HEADPAGE` a template on `int D`.** These are listed in `$S/kvnew-map.md` §1.7 items 1–6: `page_bytes`, `plane_bytes`, `raw_bytes`, `raw_of`, `coord`, `load_bases`, `Raw`, `raw_load`, `raw_load_plain`, `decode_tile`, `decode_raw`, `F8`, `patch`, `apply_entries`, `KPre`, `VPre`, `KOne`, `VOne`, the k_* and v_* functions, `raw_tile`, `make_raw`, `raw_run`, `write_run`, `write_kernel`, `unpack_kernel`, `gather_kernel`, `stage_put_kernel`, `attn_kernel`, `merge_head` and `combine_kernel`.
  - Array sizes become expressions of `Dims<D>`: `H[16]` becomes `H[D/8]`, `f[8]` becomes `f[D/16]`, `Raw.lo[4]` becomes `lo[D/32]`, `Raw.cd[2]` becomes `cd[D/64]`, and the `raw_load` offsets become `512*j` for `j < 3*D/64`.
  - Every loop bound becomes a `Dims<D>` constant.
  - The host entries instantiate `<128>` only.
- [ ] **Step 4: Compile and compare registers.** Compile for sm_89 and sm_90a; the register and spill report for every `<128>` instance must equal Step 1's line for line. A difference means the refactor changed code generation: find why before going on.
- [ ] **Step 5: Run the kernel checks on an Ada pod.**
  ```bash
  python gpu/vllm/check_vllm.py --kv --kernels-only --out ~/results/t1
  ```
  Expected: 81 of 81 pass, with output bits identical to the previous library. Also run, with `GLYD_GPU_LIB` set to the main build:
  ```bash
  python gpu/vllm/kv_decode_bench.py --calls 40 --repeat 3 --out ~/results/t1/bench-main.json
  ```
  and again with `GLYD_GPU_LIB` set to this build, then compare the two JSON files' `bits` fields.
- [ ] **Step 6: Commit.** Message: `gpu: the KV kernels' head size a template parameter (128 only, unchanged)`.

### Task 2: Five-bit group field in a page's exception entries

**Files:**
- Modify: `gpu/glyd_gpu.cu` cu:4418-4419 (the layout comment), cu:4435 (`LIVE`), cu:4473-4478 (`entry`, `e_tok`, `e_kind`, `e_head`), cu:4647 (`patch` switch), cu:4663 (`apply_entries`).

**Interfaces:**
- Produces: the entry is `lane 0-4 | group 5-9 | byte 10-11 | XOR byte 12-19 | token 20-23 | kind 24 | head 25-29 | LIVE 30`. `patch` switches on `(e>>5)&31` over `D/8` cases.
- Pages live only in memory, so no migration is needed.

- [ ] **Step 1: Change the field positions and shifts.** Shift every field above the group up by one, set `LIVE = 1u << 30`, and keep bit 31 free. `KV_PATCH(i)` gets cases 16–31, compiled only when `D == 256` (`if constexpr`).
- [ ] **Step 2: Add a self-check of the layout in the kernel checks.**
  - In `check_vllm.py` `kv_kernel_checks`, add one case that writes, per head, a token whose every channel is an exception: all 128 channels, and at D = 256 all 256.
  - Read it back with `kv_unpack` and require it bit for bit, with no void page in the ring.
  - The case is named `exceptions in every channel (D)`.
- [ ] **Step 3: Compile, then run the D = 128 kernel checks on an Ada pod.** Expected: 82 of 82 pass, the new case included.
- [ ] **Step 4: Commit.** Message: `gpu: a KV page's exception entries take 32 channel groups`.

### Task 3: D = 256 in the write, unpack, gather and staging paths, through C API 16

**Files:**
- Modify:
  - `gpu/glyd_gpu.cu`: the host entries, `kv_shape_ok`, and `kv_second_bytes` with a D argument.
  - `gpu/glyd_gpu.h`: h:34-55 (the version history and `#define GLYD_GPU_API_VERSION 16`), the prototypes, and the page-format comment at h:392-415, which gains the D = 256 tile sizes.
  - `glyd-gpu-py/include/glyd_gpu.h`: line 42, version 16.
  - `glyd-gpu-py/glyd_gpu/_lib.py`: `API_VERSION = 16` (:76), the `_PLAIN` `kv_second_bytes` row (`[_I64]*3+[_P]`), the `kv_release` D argument (:838-846), `Extra` (:775-805) with plane bytes `heads*16*D`, and `kv_decode_workspace_bytes(..., D)` with D required, passed by `kv_decode_workspace` (:915) and `kv_decode` (:931).
  - `glyd-gpu-py/glyd_gpu/vllm_kv.py`: `leaf_bytes(H, bay, D)` uses `H*48*D` (:110-113); `free_bay(H, D)` (:116-118); `second_percent` and `make_extra` use `H*16*D` per slab (:143-148, :327).
  - `gpu/vllm/check_vllm.py`: the helpers `kv_pool` (:493), `kv_natural` (:503) and `kv_bases` (:519) take D, and `kv_kernel_checks(D)` runs every case for D in (128, 256).
  - `glyd-gpu-py/tests/test_vllm.py`: `test_kv_page` (:609-657) gets D = 256 expectations.

**Interfaces:**
- Consumes: Task 1's `<D>` kernels and Task 2's entry layout.
- Produces:
  - `glyd_gpu_kv_second_bytes(int64_t nslabs, int64_t H, int64_t D, int64_t* bytes)`.
  - `kv_shape_ok` accepts D in {128, 256}.
  - The Python `_lib.kv_second_bytes(nslabs, H, D)`, `_lib.kv_release(..., D)` and `vllm_kv.leaf_bytes(H, bay, D)`.
  - `leaf_bytes(4, free_bay(4, 256), 256) == 49408`, and `leaf_bytes(8, 15, 128) == 49408` as before.

- [ ] **Step 1: Write the failing no-GPU tests.**
  - In `test_kv_page`, add:
    ```python
    assert vllm_kv.leaf_bytes(4, vllm_kv.free_bay(4, 256), 256) == 49408
    assert vllm_kv.free_bay(4, 256) == 31 and vllm_kv.free_bay(8, 128) == 15
    assert vllm_kv.cell_bytes(4, 31, 256) == 772
    ```
    `cell_bytes` gains its D argument here as well.
  - Run `python glyd-gpu-py/tests/test_gpu.py` and `python glyd-gpu-py/tests/test_vllm.py`. Expected: they fail on the new asserts and on `test_c_header` once the header changes.
- [ ] **Step 2: Implement.**
  - Host dispatch: `if (D == 128) launch<128>(...) else launch<256>(...)` in `kv_write`, `kv_unpack`, `kv_gather` and `kv_work_put`.
  - `kv_release` checks `page_stride` with D.
  - `kv_second_bytes` takes D.
  - Bump the API version in the three places. In h:34-54 add the sentence `16 from head size 256 (glyd_gpu_kv_*'s D: 128 or 256; glyd_gpu_kv_second_bytes's argument D)`.
  - The decode entries still refuse 256 until Task 4.
- [ ] **Step 3: Pass the no-GPU tests.** Run `python glyd-gpu-py/tests/test_gpu.py` (including `test_c_header`) and `python glyd-gpu-py/tests/test_vllm.py`. Expected: both pass.
- [ ] **Step 4: Compile and check registers.** Compile with `-Xptxas -v` for sm_89 and sm_90a. Hard requirement: 0 spill bytes for `write_kernel<256>`, `unpack_kernel<256>` and `gather_kernel<256>`. If `write_kernel<256>` spills, change its shape to two warps per head: 512 threads, 8 heads a round in two halves, each warp owning one half of the channels. Its `__launch_bounds__(512)` is recorded here.
- [ ] **Step 5: Run the kernel checks on an Ada pod** with `python gpu/vllm/check_vllm.py --kv --kernels-only`. Expected: every case passes at D = 128 and at D = 256, decode cases excepted (they are D = 128 only until Task 4). That is 82 + 60-odd checks; the job writes the exact count.
- [ ] **Step 6: Commit.** Message: `gpu: KV pages of head size 256 written, unpacked and gathered exactly (C API 16)`.

### Task 4: The decode kernel at D = 256

This is the performance core of the work; use the most capable model.

**Files:**
- Modify:
  - `gpu/glyd_gpu.cu`: `attn_kernel` (cu:5692-5848), `merge_head` (cu:5644), `combine_kernel` (cu:5851), `kv_ctas_per_sm` (cu:7588; its `known` cache gains a D index), `kv_splits` (cu:7608), `kv_decode_workspace` (cu:7699; `(2 + D)`) and `kv_decode` (cu:7705).
  - `gpu/vllm/kv_decode_bench.py`: the `--head 256 --kv-heads 4 --group 6` options, beside the current `H, G, D, PT = 8, 4, 128, 16` (:31).

**Interfaces:**
- Produces: `glyd_gpu_kv_decode` and its workspace and shares functions accept D = 256 with G from 1 to 8. A sequence's output bits depend only on its own length and `splits`, as at D = 128.

- [ ] **Step 1: Choose the CTA shape from measurements.** Compile two candidates for sm_89 and sm_90a with `-Xptxas -v` and pick the one with 0 spills and the highest occupancy:
  - **(a) Halves in sequence:** NW = 4 warps, a warp per page. K is decoded as two 128-channel halves accumulating scores, and V as two halves into `o[16][4]`. Shared memory is `s_o[NW][8][D]` (32 KiB, static).
  - **(b) Two warps per page:** NW = 8, each warp holding 128 of the channels. Scores are summed across the pair through shared memory, each warp keeps its own V channels, and `s_o` uses dynamic shared memory. `per_sm` already takes `shared`, and `kv_ctas_per_sm` passes it.

  Record the registers, spills and occupancy of both in `$S/kvnew/decode256-shape.md`.
- [ ] **Step 2: Implement the chosen shape.**
  - The merge stops assuming one thread a channel: each thread takes `D/(NW*32)` channels.
  - `merge_head` takes 8 channels a lane at D = 256 (two float4).
  - `combine_kernel` loops over the channels.
  - D = 128 keeps today's code path, selected by `if constexpr`, so its bits do not move.
- [ ] **Step 3: Extend the decode checks to D = 256.** In `kv_kernel_checks`, every decode case now also runs at D = 256 with H = 4 and G in {1, 2, 4, 6, 8}:
  - against the fp32 reference;
  - forced splits 1 to 64;
  - the same bits alone, batched and padded;
  - PIPE and not, FUSE and not;
  - CUDA graph replay.
  Pages with exceptions and raw pages are included.
- [ ] **Step 4: Run the checks and the benchmark** on an Ada pod and then an H100 pod:
  ```bash
  python gpu/vllm/check_vllm.py --kv --kernels-only
  python gpu/vllm/kv_decode_bench.py --head 256 --kv-heads 4 --group 6 --shapes 1x1024,8x4096,32x1024,4x32768,1x131072 --out ~/results/t4/bench256.json
  ```
  Expected:
  - every check passes;
  - the bench's FA2 comparison is within the fp32 reference's error;
  - decode time at each shape is reported against FA2's.

  The gate: graph-mode decode time at most FA2's at every shape on the H100 and the Ada GPU. Below the gate, iterate on the shape (Step 1's other candidate, PPC_MIN, the floor) within this task. If the gate cannot be met, record it: `auto` stays off for D = 256 on that GPU (Task 9).
- [ ] **Step 5: Commit.** Message: `gpu: the KV decode kernel at head size 256`, with the bench JSON summarized in the commit body's one line.

### Task 5: Head size 256 in the plugin (full-attention models)

**Files:**
- Modify:
  - `glyd-gpu-py/glyd_gpu/vllm_kv.py`. The module constant `D` becomes a per-layer value taken from `layer.head_size` in these places:
    - `create_weights` (:509);
    - the tables' shape check (:522);
    - `supports_head_size` (:636), which returns `head_size in (128, 256)`;
    - the FA3 schedule (:667, :674);
    - `GlydImpl.__init__` (:754);
    - the verify twin (:801, :811-815, :855, :867-869);
    - `work()` (:358-373), sized for the largest D served.

    `_STATE` gains `D`, and one D for every Glyd layer of a model is required, raising if they differ.
  - `glyd-gpu-py/glyd_gpu/vllm_plugin.py`: `_kv_refusals` (:969-970) accepts head size 128 or 256.
  - `glyd-gpu-py/tests/test_vllm.py`: `test_kv_refusals` (:715-746) and `test_kv_backend`.
  - `gpu/vllm/kv_stress.py`: `H, D, LAYERS` come from the tables' meta instead of the constants (:31-35).

**Interfaces:**
- Consumes: Task 3's `leaf_bytes(H, bay, D)` and `_lib` D arguments, and Task 4's decode at D = 256.
- Produces: `vllm_kv._STATE.D`, the head size of the model being served.

- [ ] **Step 1: Write the failing tests.**
  - In `test_kv_refusals`, a config with `head_dim = 256` gets no head-size refusal; one with `head_dim = 64` gets `"head size 64: 128 or 256 only"`.
  - In `test_kv_backend`, `GlydBackend.supports_head_size(256)` is True and `supports_head_size(192)` is False.
  - Run them; expected: FAIL.
- [ ] **Step 2: Implement** as listed under Files.
- [ ] **Step 3: Pass the no-GPU tests** with `python glyd-gpu-py/tests/test_vllm.py`.
- [ ] **Step 4: Build the dev model.** Write `gpu/vllm/make_skeleton.py`:
  - `python make_skeleton.py CONFIG_REPO OUT --layers N --hidden H [--keep-heads]` builds a random-weight model from a real model's `config.json`, with its tokenizer and vocabulary.
  - It shrinks only `num_hidden_layers`, `hidden_size` and `intermediate_size` (and any `layer_types` list, cut to N while keeping the full/linear pattern), uses `AutoModelForCausalLM.from_config`, seeds 0, saves bf16 safetensors, and checks that the tensor names match the real checkpoint's patterns.
  - Make `hd256-tiny` from `Qwen/WebWorld-8B` (2026, Qwen3's architecture, every layer full attention) with `head_dim` 256 and 4 KV heads: a full-attention model of head size 256 for this task.
- [ ] **Step 5: Run on an Ada pod:**
  - `python -m glyd_gpu kvcal <skeleton> --positions 4096`;
  - `python gpu/vllm/check_vllm.py --kv <skeleton>` (the engine checks: `kv_real` 0 differ, verify, prompt logprobs bit for bit, default mode);
  - `python kv_stress.py engine <tables> --verify --ext --quick`.

  Expected: every check passes, with 0 values that differ.
- [ ] **Step 6: Commit.** Message: `vllm: the lossless KV cache serves head size 256`.

### Task 6: Hybrid models: manager blocks, pages, and clearing before reuse

**Files:**
- Modify:
  - `glyd-gpu-py/glyd_gpu/vllm_sched.py`: `_glyd_exhausted` (:224-272) and `_glyd_release` (:274-285).
  - `glyd-gpu-py/glyd_gpu/vllm_kv.py`: `release`, `release_pending` (:288-316), `RELEASE_MAX` (:94), `Monitor.lost` and `second_new` (:182-205), `gap_blocks` (:272), `make_extra` (:317-340).
  - `glyd-gpu-py/glyd_gpu/vllm_admit.py`: `room()` (:76-79).
  - `glyd-gpu-py/glyd_gpu/vllm_plugin.py`: `_kv_refusals` (:965-967).
  - Tests in `glyd-gpu-py/tests/test_vllm.py`.

**Interfaces:**
- Produces:
  - `vllm_kv.pages_of(manager_block_ids) -> list[int]` and `vllm_kv.manager_of(page) -> int`, with `ratio = cache_config.block_size // 16` stored as `_STATE.ratio`.
  - `vllm_kv.glyd_group_ids(kv_cache_config) -> set[int]`, the KV cache groups whose layers are Glyd's.
  - `vllm_sched` holds a freed block out of vLLM's free queue until it is cleared.

- [ ] **Step 1: Write the failing no-GPU tests.**
  - `test_kv_pages_of`: with ratio 65, `pages_of([0, 2]) == list(range(0, 65)) + list(range(130, 195))` and `manager_of(131) == 2`.
  - `test_kv_hybrid_refusals`: a config with `layer_types` of 48 `linear_attention` and 16 `full_attention` and head_dim 256 gets no refusal; a `sliding_attention` layer is still refused with `"sliding window layers are not held"`.
  - `test_kv_release_before_reuse`: a fake scheduler and block pool. A block freed by a Glyd request is not in the free queue until `release_pending()` has run its clear, after which it is in the queue. A block freed by a linear-attention group is cleared too: its pages may hold Glyd's headers from an earlier owner.
  - Run them; expected: FAIL.
- [ ] **Step 2: Implement the units.**
  - The lost and raw rings name pages, and the scheduler maps them to manager blocks with `manager_of`.
  - Release hands `pages_of(blocks)` to the kernel.
  - `make_extra`, `room()` and `second_percent` size the reserves from the attention groups' share: `num_gpu_blocks * ratio * L * leaf_bytes`, where the pool's blocks are counted only for the share of the attention group (`1 / groups`).
- [ ] **Step 3: Implement clear-before-reuse.** In `_glyd_release`, vLLM's `block_pool` free path is intercepted:
  - Blocks that reach ref_cnt 0 are kept in `_HELD`, not in the free queue, and handed to `vllm_kv.release`.
  - `GlydMetadataBuilder.build` runs `release_pending()` with no cap. RELEASE_MAX is a per-call chunk size, looped until empty.
  - After the clear is enqueued on the stream, the blocks return to vLLM's free queue at the next scheduler step.
  - The early return during graph capture stays, but held blocks then stay held, never freed uncleared.
  - The exact vLLM 0.30 functions intercepted are named in the code with their file:line, read from the installed vLLM on the pod; record them in the module docstring's list of internals.
- [ ] **Step 4: Lift the refusal.** `_kv_refusals` keeps refusing sliding window but not `linear_attention` or `mamba`. The hybrid model's layers that are not `Attention` keep vLLM's backends.
- [ ] **Step 5: Pass the no-GPU tests** with `python glyd-gpu-py/tests/test_vllm.py`.
- [ ] **Step 6: Commit.** Message: `vllm: hybrid models: Glyd's pages in vLLM's shared blocks, cleared before reuse`.

### Task 7: Calibration and engine checks on a hybrid model

**Files:**
- Modify:
  - `glyd-gpu-py/glyd_gpu/kvcal.py`: the sweep batch for hybrid models, at most the pool's state blocks; `meta["hybrid"]`; `VERSION` 9.
  - `gpu/vllm/check_vllm.py`: `cap_bar` (:1864) takes D and the groups.
  - `gpu/vllm/kv_stress.py`: a `--churn` mode for engine runs, cycling requests of mixed lengths so blocks pass between the attention and linear-attention groups, with verify on.
- Create: the skeletons `qwen38-tiny` (8 layers, hidden 512, the real attention and linear-attention sizes) and `qwen38-real4` (4 layers of the real sizes), both through `make_skeleton.py` from `Qwen/Qwen3.8-27B`.

**Interfaces:**
- Consumes: Tasks 5 and 6.
- Produces: tables for hybrid models whose `meta["layers"]` counts the full-attention layers only.

- [ ] **Step 1: Run on an Ada pod:**
  - `python -m glyd_gpu kvcal qwen38-tiny --positions 8192` and the same for `qwen38-real4`;
  - `python gpu/vllm/check_vllm.py --kv qwen38-real4`;
  - `python gpu/vllm/check_vllm.py --kv --ext-only qwen38-real4` (prefix hits and chunked prefill);
  - `python kv_stress.py engine <tables> --verify --ext --churn --quick`.

  Expected:
  - the tables are made, with held-out rate at most 0.15%;
  - `kv_real` 0 differ;
  - verify finds no difference over the churn run;
  - prompt logprobs bit for bit vLLM's with prefix hits;
  - vLLM's log shows "Setting attention block size to ~1,040 tokens" for the real-shape model.
- [ ] **Step 2: If a check fails,** use superpowers:systematic-debugging. The likely spots are block-id units and the release order. Fix it here, with a test.
- [ ] **Step 3: Commit.** Message: `vllm: the lossless KV cache serves hybrid models (checked on Qwen3.8 skeletons)`.

### Task 8: Long context: 65,536 and 131,072 tokens

**Files:**
- Modify:
  - `glyd-gpu-py/glyd_gpu/kvcal.py`: `BANDS` gains 65536 and 131072; remove `LONG_BANDS` and its `# dev:` gate (:31, :320-326); update the docstring and `minutes()` (:529-531).
  - `glyd-gpu-py/glyd_gpu/vllm_plugin.py`: the window text (:58, :931-936, :1012).
  - Tests: `tests/test_vllm.py` :742, :791, :1521.
  - `gpu/vllm/README.md`: :366, :383, :465, :487.

- [ ] **Step 1: Update the tests first.** `kvcal.bands()` ends at 131072 with no environment variable, and a `--max-model-len 131072` model gets no window refusal. Run them; expected: FAIL. Then implement and pass them.
- [ ] **Step 2: Time the long tiers' calibration** on an H100 pod: Qwen/Qwen3.5-9B at 65,536 and 131,072 positions, then Qwen3.8-27B at 131,072, with `kv_firststart.py`. Record the seconds.
  - If Qwen3.8-27B at 131,072 takes over 600 s, add on-demand tiers:
    - `_kv_prepare` calibrates the tier that covers 40,960, or `max_model_len` if smaller.
    - The engine serves requests up to that tier with the lossless cache.
    - A request past the tier gets the bigger tier made in a child process while the server keeps serving.
    - Until that tier exists, a longer request is refused with a 400 that says why. This is a separate commit, with a test.
- [ ] **Step 3: Check the long tiers.** Run `kv_long.py stock|kv` at 120,000 tokens on Qwen/Qwen3.5-9B with the 131,072 tables. Expected: every prompt logprob bit for bit, and `kv_long.py compare` reports N of N.
- [ ] **Step 4: Commit.** Message: `vllm: the lossless KV cache's tables up to 131,072 positions`.

### Task 9: `auto` for these models

**Files:**
- Modify: `glyd-gpu-py/glyd_gpu/vllm_plugin.py`, `_kv_decide` (:985-1014) and `KV_AUTO` (:180), plus tests in `test_kv_auto` (:748).

- [ ] **Step 1: Write the failing test.** `auto` on an H100 for a head-256 hybrid config holds the cache, or stays off with `"head size 256 is not measured faster on this GPU yet"`, following a per-(GPU, D) table `AUTO_D256 = {"H100": bool, "L4": bool, "A100": bool, "GH200": bool}` set from Task 4's and Task 10's gates.
- [ ] **Step 2: Implement it, pass the test, commit.** Message: `vllm: kv auto for head size 256 where it is measured faster`.

### Task 10: The H100 run on Qwen3.8-27B (the release gate)

**Files:**
- Create:
  - `$S/ceiling/kvnew_h100_job.sh`, built from `$S/ceiling/qwen38_job.sh`.
  - `benchmarks/gpu/h100-qwen38-kv-<date>/`, the record, made by `$S/q38job/make_record.py`.

- [ ] **Step 1: Run the job.** Three servers, with `--language-model-only --max-model-len 32768 --max-num-seqs 160`:
  - bf16;
  - Glyd weights with `kv` off;
  - Glyd weights with `kv` auto (lossless).

  `vllm bench serve` runs random prompts, 1,024 / 8,192 / 32,768 tokens in and 256 out: saturated and at 1 request a second. Add the quality probe, the clock guard, and vLLM's KV lines.
- [ ] **Step 2: Ask the owner before launch** with the exact estimate: about 40 minutes of H100 at $2.69/h, about $2.
- [ ] **Step 3: Check against the gate.**
  - Every value exact: verify in a side run of the same tables, 0 differ.
  - At least 1.2x the requests at once against Glyd without the KV cache, at 8,192 and at 32,768 tokens.
  - Decode not slower than vLLM's own cache.

  Write the record, then set `AUTO_D256["H100"]` by the result (Task 9's table).
- [ ] **Step 4: Commit the record and the auto table.** Then open the private PR `kv-newer -> main`, run CI, and merge only once every check-run has completed.

---

## Self-review

- **Spec coverage:**
  - D = 256 kernels: Tasks 1–4.
  - Hybrid models, with block units, clear-before-reuse and reserves from the attention share: Task 6. The hybrid calibration is Task 7.
  - Long context: Task 8.
  - `auto`: Task 9.
  - The checks named in the spec: kv_stress (Tasks 5 and 7), verify (Tasks 5 and 7), prompt logprobs with chunked prefill and prefix hits (Task 7), decode speed (Task 4's bench and Task 10), no D = 128 regression (every task's 81 checks), the H100 run (Task 10).
  - Costs: the Budget section.
- **Placeholders:** none. The only open measurements are the gates themselves: Task 4's shape choice, Task 8's 600 s threshold and Task 10's results. Each has a decided rule.
- **Names used across tasks:**
  - `leaf_bytes(H, bay, D)`, `free_bay(H, D)` and `cell_bytes(H, bay, D)`: Tasks 3, 5 and 6.
  - `_lib.kv_second_bytes(nslabs, H, D)`: Tasks 3 and 6.
  - `_STATE.D`: Task 5.
  - `_STATE.ratio`, `pages_of` and `manager_of`: Tasks 6 and 7.
  - `AUTO_D256`: Tasks 9 and 10.
