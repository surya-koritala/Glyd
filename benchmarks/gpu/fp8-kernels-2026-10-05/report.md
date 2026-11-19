# Task 2 report: CUDA pack, unpack and list unpack of the lossless FP8 format, C API 16, bindings

Status: DONE_WITH_CONCERNS (concerns at the end). Worktree `/Users/suryakoritala/glyd-fp8`, branch `fp8-lossless`, pushed to `private` only.
Commits (in order, on top of Task 1's `f2ec33ea`):

- `b7bb5da0` fp8: CUDA pack, unpack and list unpack of the lossless FP8 format, C API 16, bindings, check_f8 and a CPU emulation
- `e368397e` fp8: the GPU kernels, their ptxas lines, the checks and the unpack's bandwidth in gpu/fp8/README.md

(A third commit, a "level-3 values in one loop a thread" variant, was measured slower on the H100 and dropped before the push: see Speed.)
No subagent. f8codec.py not created. Nothing pushed to origin.

## The C API added (gpu/glyd_gpu.h, version 15 -> 16, history line added; include/glyd_gpu.h and glyd-gpu/src/lib.rs at 16)

```
int glyd_gpu_f8_size_workspace(int64_t O, int64_t K, size_t* bytes);             // 16 bytes (the scan's total), 8-byte aligned, device memory
int glyd_gpu_f8_size(const uint8_t* w, int64_t ld, int64_t O, int64_t K, uint32_t* heads, void* workspace, size_t workspace_bytes, int64_t* var_words, cudaStream_t cs);
int glyd_gpu_f8_pack(const uint8_t* w, int64_t ld, int64_t O, int64_t K, const uint32_t* heads, uint8_t* core, uint32_t* var, int64_t var_words, cudaStream_t cs);
int glyd_gpu_f8_unpack(const uint8_t* core, const uint32_t* heads, const uint32_t* var, int64_t O, int64_t K, uint8_t* out, int64_t ld, cudaStream_t cs);
int glyd_gpu_f8_unpack_experts(const uint8_t* core, const uint32_t* heads, const uint32_t* var, int64_t E, int64_t O, int64_t K, const int32_t* ids, int64_t n, uint8_t* out, int64_t ld, cudaStream_t cs);
```

- The size is data dependent (var's words), so the "size query" is a call that reads W once, writes `heads` (each patch's words, then bases by an in-place scan) and waits for the stream: `*var_words` (guard words included) is what the caller allocates. `glyd_gpu_f8_pack` then fills core and var. Bytes are `fp8ref.pack`'s for every shape and for rows `ld` apart (ld >= K).
- `glyd_gpu_f8_unpack`: one launch, no allocation, no wait, no host sync: CUDA-graph capturable (fixed pointers; verified by capture and replay).
- `glyd_gpu_f8_unpack_experts`: a stack of E matrices packed as one [E O, K] (O a multiple of 64, so each expert is `O/64 x ceil(K/128)` whole patches); decodes the matrices `ids[i]` (int32 on the device, read by the kernel; an id outside [0, E) is skipped and its place in `out` is left as it was) into `out [n, O, K]`, rows `ld` apart. Grid y = the item; n <= 65535. One launch for any ids, graph capturable.
- Alignment: core and heads 16-byte aligned (torch allocations are); w and out need none (16-byte accesses are used only where the pointer and ld allow, else byte accesses). Free-standing patch-range lists were not added: the expert list covers the MoE case and the patches of one expert are contiguous in the pack.
- All five are GLYD_GPU_INTERNAL (reached through the table by FNV id): table entries and the unique-id list updated. The Rust crate (`glyd-gpu/src/lib.rs`) declares them and has API_VERSION 16 (its `declarations_are_the_headers` test and the others pass: 21 passed).
- Python: `_lib.py` rows in `_ARGS` (`f8_size`, `f8_pack`, `f8_unpack`, `f8_unpack_experts`), `_SIZES["f8_size"] = 2`, `API_VERSION = 16`, and the four wrappers; `kernels.py`: `F8Packed` (`nbytes()`, `bits_per_value()`, `plain()`), `f8_pack(w)` (CUDA uint8 or float8_e4m3fn [O, K], rows of any distance), `f8_unpack(p, out=None)` (float8_e4m3fn out when allocated), `f8_unpack_into(p, out)` (graph safe, light checks: no allocation, no sync), `f8_unpack_experts(p, E, ids, out)`. No name or message holds a method word (`test_strings_say_no_method` passes). The dev pybind module (`TORCH_EXTENSION_NAME`, not what vLLM uses) has no f8 functions; `_ext` must be the prebuilt library.

## Kernels and launch shapes (namespace `f8c` in gpu/glyd_gpu.cu, between `// f8c begin` and `// f8c end`)

All our own code. One thread = two lanes = 32 values (stream bytes 32 tid..32 tid+31), tid = 16 unit + m, so a patch is 8 warps.

| Kernel | Launch | Work |
| :--- | :--- | :--- |
| `f8_head_kernel` | grid patches, 256 threads | rows loaded 32 bytes a thread (zero past the edges), exponent counts in shared memory a warp, ranks (count desc, exponent desc: the reference's tie rule), head + words, plain if words >= 512 |
| `f8_scan_kernel` | 1 CTA, 1,024 threads | in-place exclusive sum of the words (the bases), the total (64-bit, checked against 2^32 - 2) to the workspace |
| `f8_pack_kernel` | grid patches, 256 threads | patch to shared memory (swizzled), re-read in stream order as 8 words a thread; d1 digits, nibbles, ordinals of levels 2, 3 and the literals by two CTA scans (packed 16-bit fields), digits OR'd (shared atomics) into a shared copy of the patch's part of var, then core (8- and 16-byte stores) and var written; plain patches copy their stream bytes (core 6,144, 512 words of var) |
| `f8_unpack_kernel` | grid (patches, items), 256 threads | head (broadcast), 8 B of digits + 16 B of nibbles a thread, the patch's part of var staged in shared memory, ordinals by three CTA scans (c1, then the marked count, then the literals), levels 1 and 2 for all 32 values with `prmt` selectors, the values the level-2 digit leaves to level 3 marked in the table's byte and fixed by short loops, patch to rows through an 8 KiB swizzled shared tile (no bank conflicts), rows written 16 bytes a thread, clipped at O and K |

The scan helper uses `__shfl_up_sync` and `__syncthreads` only; no sm-specific instruction, so the same code serves sm_80 to sm_120.

## ptxas (nvidia/cuda:13.0.2-devel-ubuntu24.04, `-O3 -std=c++20 --expt-relaxed-constexpr -Xptxas -v`; logs in `scratchpad/fp8sdd/t2/ptxas_*.log`)

| | sm_89 | sm_90a |
| :--- | ---: | ---: |
| f8_unpack_kernel | 48 regs, 10,368 B smem, 0 spills | 48, 10,368, 0 |
| f8_pack_kernel | 55, 10,368, 0 | 63, 10,368, 0 |
| f8_head_kernel | 40, 640, 0 | 31, 640, 0 |
| f8_scan_kernel | 44, 128, 0 | 30, 128, 0 |

All four: 0 bytes stack frame, 0 spill stores, 0 spill loads. Base (HEAD before the change) against new, per function (regs, smem, stack, spills): 247 functions on each arch, 0 differences, none missing (`t2/cmp.py`). A release-style compile (`gen_names.py` header, `-DGLYD_RELEASE`, sm_89) also builds: the kernels become gk_5..gk_8 in namespace gt_39; gen_names renames all f8c kernels, device functions and types.

## GPU checks (scripts: `gpu/fp8/check_f8.py`; job `scratchpad/f8k/f8k_job.sh`; results `scratchpad/ceiling/results-f8k-*`)

`check_f8.py` runs every test of `test_fp8ref.py` with the backend's pack and unpack replacing the reference's (the GPU's core, heads and var must equal `fp8ref.pack`'s array by array, else the first differing patch is reported; unpack of the reference's bytes must equal the matrix bit for bit), then: strided unpack and strided pack (ld 128/144/200/203/160/300/33/32; K and O not multiples of 16 or of a patch; the bytes between rows must not change), the experts' unpack (ids repeated, in any order, -1, E, INT_MAX, INT_MIN; skipped places stay 0xA5; ld != K), CUDA graphs (capture `f8_unpack_experts` + `f8_unpack_into`, no allocation during capture, replay with other ids and with other bytes of the same size written in place), and the bandwidth line.

- Pod 1, RTX 4090 (sm_89, driver 580.178.04, torch 2.14.1+cu130, nvcc 13.0, library built by `gpu/build_lib.sh`, all six architectures, 92 s): `results-f8k-ada1/check_f8.log`: **ALL BIT FOR BIT**: every test_fp8ref case (all 256 byte patterns, NaNs and zeros, 1x1 .. 4096x7168, plain patches beside coded ones, all-zero patches, 80 fuzz matrices), the 50 real samples of 5 repos (fetched on the pod by `fetch_samples.py`, 10 s a repo; bits a value 6.665, 6.685, 6.687, 6.667, 6.720 as on the Mac), strides, experts, graphs. compute-sanitizer racecheck and memcheck on the strides and the experts cases (`san-*.log`): 0 hazards, 0 errors.
- Pod 2, H100 80GB HBM3 (sm_90a, driver 580.126.09): `results-f8k-hop1/check_f8.log`: **ALL BIT FOR BIT**, same cases, samples included.
- The same logic on the CPU without a GPU (`check_f8.py --emu`; `emu.py` + `emu_f8.cpp` compile the text between the f8c markers, one std::thread a CUDA thread, barriers for `__syncthreads` and each warp's shuffles): every test_fp8ref case (fuzz included), strides, experts, and 5 real samples (256 rows each, with level-3 values and literals present: n3 3,852 .. 16,971 literals a sample): bit for bit. That is how the layout was got right before the first GPU run (the lane order from the PTX documentation, which Task 1 had not run, is confirmed by the Ada and Hopper runs: PERM needed no change).

## Speed (7168 x 18432 matrix, block-scaled Gaussian weights, 6.647 bits a value, 0 plain patches; 200 launches back to back, best of 5 runs)

| GPU | unpack | e4m3 written | with packed bytes read | pack (size query + its wait + allocations) |
| :--- | ---: | ---: | ---: | ---: |
| RTX 4090 | 265.3 us | **498 GB/s** | 912 GB/s of 1,008 (memory-bound) | 0.5 ms |
| H100 80GB HBM3 | 178.0 us | **742 GB/s** | 1,359 GB/s of 3,350 | 0.4 ms |

The 4090 is at the memory's limit. The H100 is instruction-limited: about 39 instructions a value (the unpack kernel's coded path is some 1,100 static SASS instructions a thread for 32 values; that count at the full issue rate on 132 SMs at ~1.75 GHz is 175 us, the measured time). Where it goes (SASS): the level-1/2 selector chain about 7 instructions a value, three CTA scans about 40 each, the marked values' short loops about a quarter. A variant that moved those to one loop a thread (`pm` mask, smem reads, an 8-way select) and unrolled the scan's sum was bit-exact (emu, not on a GPU) but ran 189 us (699 GB/s) on an H100 and was dropped, so the commit is the first version. To go faster: fewer instructions a value (no scan for the second and third level: a prefix table over var's words built beside the first scan; a cheaper chain), or avoid the decode (Task 5).

## Pods used (Runpod via `scratchpad/rp_run.sh`, rp_guard running, `MAX_MIN=40`, each deleted at its end; names `glyd-bench-f8k-*`)

3 pods, 13 pod minutes in all: `f8k-ada1` RTX 4090 4 min (first GPU tried, no capacity problem), `f8k-hop1` H100 4 min (the committed code: the bandwidth line above), `f8k-hop2` H100 5 min (the dropped variant: 189 us, also bit for bit). One pod of the budget of four unused. (`glyd-bench-f8a1` / `f8a2` in the pod lists are presumably the other engineer's (vLLM worktree); not touched.)

## Existing checks (no GPU here)

`glyd-gpu-py/tests/test_gpu.py` passes (exit 0; `test_c_header`, `test_public_header`, `test_gen_names`, `test_strings_say_no_method` among them), `test_vllm.py` passes (vLLM not installed: 26 of 27 skip, as before), `test_fp8ref.py` passes (all tests; real samples skipped locally, bit for bit in `check_f8.py` on the pod), `cargo test -p glyd-gpu` 21 passed. `python gpu/vllm/check_vllm.py --kernels-only` needs a GPU and was not run: no BF16 file or function was changed, and every existing kernel has identical ptxas lines.

## Open issues and concerns

1. **Unpack speed on Hopper**: 742 GB/s of e4m3 written, 41% of the H100's memory bandwidth in total traffic. A 132 MB layer matrix takes 0.18 ms to decode before vLLM's FP8 GEMM; for a prompt of 2k tokens that is a visible share of a layer's GEMM time. Task 3 can use it as is (correctness first); the instruction cut above is the lever, or Task 5's fused kernel for few tokens.
2. **Pack's size call syncs** the stream (it must: var's size is data dependent); it is a load-time call. Pack time 0.4-0.5 ms for 132 MB.
3. **Expert list** needs O % 64 == 0 (the stack is the pack of [E O, K]); for other O pack each expert apart (separate core/heads/var). Not an issue for the models of the plan (moe_intermediate x 2 and hidden sizes are multiples of 64 and 128). TP shards with K a multiple of 64 but not of 128 are handled (padding rule), also in the list.
4. `kernels.py` needs the prebuilt library as `_ext` (the dev JIT pybind module has no f8 entries).
5. The 4090 run used the all-architecture library of `gpu/build_lib.sh`; the release name gate (`check_wheel.py --lib`) was not run on a built release library here (no linux build of it), only gen_names + a release-style compile.
6. rust crate: `glyd-gpu/src/lib.rs` declares the five functions and API_VERSION 16 (not in the task text, but its header-equality test would fail otherwise and a v16 library is refused by a crate at 15).
