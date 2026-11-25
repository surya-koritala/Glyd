# R17: one kernel build

Status: step 0 done; steps 1 to 6 begun (see "Progress" at the end): the library's operations run from the engine's fatbin (Hopper's wgmma and TMA products are written and have run on no H100 yet), and the engine still loads the library for what its own kernels do not take (step 4). Decided 2026-11-21: CUDA 12 stays (step 5); HIP stays a compile-only job until M5b, and the CUDA fold keeps the HIP path building (step 6); step 0 is R1a, #134. Each step carries its own bit-for-bit check in the GPU job (`gpu-tests.yml`, on the project's GPU runner): every step is compared bit for bit with the library on a GPU.

## What there was when this was written (2026-11-21)

The counts in this section are the library's before step 1 moved the layouts' kernels out of `gpu/glyd_gpu.cu` into `gpu/glyd_gpu_kernels.h`
("Progress" below). The engine reaches the GPU through two paths:

1. its own fatbin: `glyd-engine/kernels/*.cu`, compiled by `build.rs` into one file the binary embeds, loaded with the driver API (`cuModuleLoadData`, launches by kernel name); no CUDA runtime is linked;
2. `libglyd_gpu_cuda13.so` or `_cuda12.so` (`gpu/glyd_gpu.cu`, built by `gpu/build_lib.sh`), opened at run time by `glyd-engine/src/ffi.rs` (47 entry points: the packed matrix products, the grouped-MoE products, the unpackers, the paged KV cache's write, gather and decode, the FP8 and FP8-hybrid ops, and three that say the library's versions and errors). The Python package `glyd-gpu` loads the same library through its C API (`gpu/glyd_gpu.h`).

The library is one translation unit of 12,217 lines: 55 `__global__` kernels, 112 templates (a kernel is instantiated per weight layout: tiered, compact, 12-bit, int6, int5, int4, int4h, fp4, fp4r, int3), 76 `<<<...>>>` launches and 104 CUDA runtime calls (`cudaMalloc`, `cudaStream*`, `cudaMemcpy*`: the library takes the caller's stream and workspace where it can). The 47 entry points themselves are thin: 452 lines of bodies in all (the largest, `glyd_gpu_f8_linear`, is 35), which pick the layout from the pack's tier words (`int6_words`, `tiered_words`, ...) and call a templated `*_run`. The weight is in the kernels and their launch plans, not in the C API.

## Why fold it

The hot operations go through a C ABI of a library that ships in lockstep with the engine (the engine bundle holds both), so the ABI adds a second place for a layout or a signature to drift (S16 checks them, #220), a second compiler invocation (CUDA 12 and 13 builds of the library; the engine's kernels are CUDA 13 only), a second set of architectures, and a runtime lookup that can fail at start (`Glyd's GPU library not found`).

## The steps

Each step lands alone, keeps `libglyd_gpu` loadable (the Python package needs it), and is checked on the GPU runner with the library as the oracle: the same pack and activations through the library and through the new path, outputs equal bit for bit (`glyd-engine check`/`kcheck` have the shape of that comparison; the logits tests in `glyd-engine/tests/logits` cover it end to end).

0. **A seam.** One Rust trait for the 47 operations (the device layer of R1, #134), implemented first by the library's `ffi.rs` table as it is, so the engine calls the trait and not the table. Nothing changes on the GPU; this is the part that can be reviewed and tested without one.
1. **Kernels in the fatbin, one family at a time.** (Each family's step adds its check to the GPU job in `gpu-tests.yml`: the library and the new path on the same pack and activations, bit for bit.) For a family (say the mma linear on the layouts the engine runs: tiered, int6, int5, int4), move the templated kernels it launches into `glyd-engine/kernels/` as explicit instantiations with `extern "C" __global__` names (`mma_linear_int4_m16`, ...), so the driver API finds them by name; the launch (grid, block, shared memory, the way-selection of `glyd_gpu_mma_way`) becomes Rust next to the kernel table. Keep the CUDA source of the kernels in one file that both builds include (as built: `gpu/glyd_gpu_kernels.h`), so the library and the engine cannot differ.
2. **Workspace and stream in Rust.** The `*_workspace` entry points (size queries) become Rust functions of the same plan; the engine already owns its streams and buffers.
3. **Order of families**, by value and risk: the mma linear and its unpackers (the decode path of every model), the grouped-MoE products, the paged KV cache ops, then FP8 and FP8-hybrid. The legacy per-family pipelines are not touched: new capability goes on the shared runtime (`glyd-engine/src/runtime/`). The first PRs must not collide with R3b's (`ffi.rs` and the engine call sites): they wait for them.
4. **Remove the library from the engine.** When the trait's library implementation has no caller: delete `ffi.rs`'s library table, `GLYD_GPU_LIB`, the engine bundle's `libglyd_gpu_cuda13.so` (and the CUDA 12 one, if step 5 is not done), and `install.sh`'s copy of it. The library stays in `gpu/` and in the `glyd-gpu` wheels for Python.
5. **CUDA 12: kept (decided).** The library is built with CUDA 12.8 for drivers older than CUDA 13 (down to R535), which P6 shipped and M1's gate wants. The engine's fatbin is built by nvcc 13 only, which a driver that old cannot load, so the folded kernels get a second fatbin built by nvcc 12.8 and the engine picks one by the driver, as `ffi.rs` picks the library now. The GPU job checks the CUDA 12 path on the runner's driver if it can run it (the runner has one driver: a CUDA 12 fatbin is checked by loading it, on a driver that takes it).
6. **HIP: compile-only until M5b (decided).** `gpu/build_lib_hip.sh` builds the library for RDNA 3 and 4; the engine's HIP build (`GLYD_HIP=1`) builds the fatbin with `hipcc`. The CUDA fold comes first and must keep the HIP path building: the folded kernels compile there too, checked by `HIP kernels compile (AMD)`; nothing is run on an AMD GPU before M5b.

## Limits to keep

- The release fatbin must stay small: the engine bundle and the `glyd-gpu` wheels (PyPI's 100 MB). The library's release build is about 24 MB per CUDA version (`-DGLYD_RELEASE`: the default plans' kernels alone, no PTX, size-compressed device code); the folded kernels compile the same default plans only.
- Compile time: the library alone takes minutes of nvcc per architecture; the engine's kernels are already a long build. The `kernels.yml` compile check (cached by kernel hash, `Kernels compile`) and `build.rs`'s fatbin gain the folded kernels; budget for it.
- A Python consumer must see no change: the library's C API and `gpu/glyd_gpu.h` are frozen, not edited.

## Acceptance (from the issue)

"The engine has no runtime dependency on `libglyd_gpu`": true when step 4 lands and `ldd`/`dlopen` of the engine binary names no `libglyd_gpu*`, checked in the engine bundle test of `release-gpu.yml`.

## Progress

- **Step 0, done.** `glyd-engine/src/gpu_ops.rs`: `GpuOps`, the 32 operations the engine runs of the library, and `Gemm`, the dense product
  (cuBLAS). `ffi::Lib` and `ffi::Blas` implement them over the C API's table; every caller goes through the traits. The device layer under
  them is `device::Device` (R1a, #134).
- **Step 1, begun.** The one source both builds compile is `gpu/glyd_gpu_kernels.h` (a `.h`, so the library's cache keys, `gpu/*.h`, take
  it): the mma layouts' section of `glyd_gpu.cu` moved there unchanged (Tiers through Nib, pf_l2, mma16816), with the bodies of
  `mma_unpack_kernel`, `moe_route_kernel` and `moe_route_block_kernel` as `__device__` functions the library's kernels call.
  `glyd-engine/kernels/fold.cu` includes it in a namespace of its own and names an extern "C" kernel per instance the engine runs:
  `fold_unpack_{tiered, compact, int6, int5, int4, int4h, fp4, fp4r, int3, twelve}` and `fold_moe_route`, `fold_moe_route_block`.
  Their launches are Rust (`src/fold.rs`: the words read as the library reads them, the same grids, blocks and shared memory).
  `GpuOps::unpack` and `GpuOps::moe_way` run them; the library's stay as `Lib::library_unpack` and `Lib::library_moe_way` (the oracle,
  and `GLYD_ENGINE_FOLD=lib` in a development build). `glyd-engine foldcheck` compares both on every layout and plan shape.
- **Step 2, begun: the host side in Rust.** `glyd-engine/src/fold.rs` holds what the library's host code decided for the engine and needs no
  kernel of its: the packed products' way for M tokens on a GPU (`glyd_gpu_mma_way`, `_mma12_way`, with the route table's measured
  thresholds and its tuning variables, read as the library's development and release builds read them), the GPU's code (`glyd_gpu_gpu`:
  compute capability plus its class by name), and the paged KV cache's sizes and decode workspace (`glyd_gpu_kv_leaf_bytes`,
  `_extra_bytes`, `_second_bytes`, `_kv_decode_workspace`). `GpuOps for Lib` answers these from Rust; the library's own answers stay as
  `Lib::oracle_*` (development builds) for `glyd-engine foldcheck`, in the GPU job, which compares the two for every GPU code the library
  knows, both layouts, matrices of every kind of K, token counts on each side of every threshold, and valid and refused KV shapes. The
  workspaces left (`linear_workspace`, `moe_workspace`, the FP8 ones, `kv_decode_splits`) depend on their kernels' occupancy and launch
  plans: each folds with its family.
- **Step 3, order.** Folded (each compared bit for bit with the library by `foldcheck` in the GPU job): the unpackers; the experts' plan;
  the step product to 64 tokens (route GEMM, and the step product of the routes MID and WG for the layouts that take it); the prompt
  kernel (routes BIG, DECODE, AHEAD and SPLIT: mma_gemm_big in its three default shapes and, on GeForce Ada, its stream-K form, with
  split-K's finish); the 12-bit layout's route MID (mma12_mid, and the A100's mma12_ws); the grouped experts' step products, which
  every call takes (C API 26: their prompt kernel and rows kernel, which a development build's variables alone ask for now, are the
  library's and left the engine's fatbin), and the expert cache's places; the 12-bit KV pages' write, unpack, gather, decode and prompt attention (and over a
  bf16 cache); the lossless FP8 format's size, pack and unpack and its products (both layouts, every kind of scales, the experts').
  Their grids are the library's: occupancy of the engine's own kernels (`cuOccupancyMaxActiveBlocksPerMultiprocessor` in the device
  layer) and the same split rules, so the sums' order and the bits are the library's. A development build's variables that ask for
  another plan keep the library's. Hopper's kernels (the 12-bit layout's TMA and wgmma products, the tiered layout's TMA product, the
  experts' wgmma product) are in the fatbin, built for sm_90a, with their launches (tensor maps from the driver's
  `cuTensorMapEncodeTiled`, clusters); they have run on no H100 yet.
- **Step 5.** The folded kernels are in `kernels/engine.cu`'s fatbin, so the CUDA 12 fatbin (`NVCC12`) carries them as it carries the rest.
- **Step 6.** The HIP build compiles `engine.cu`, so `fold.cu` and the header with it: the header is the library's own HIP-compiling source.
  Not compiled by hipcc yet (the AMD job runs it).
- **Step 4, begun.** The entries `Lib` still called outside `GpuOps` run from the engine's fatbin, each compared with the library's by
  `foldcheck`: rows by id (`glyd_gpu_mma_rows`, `_mma12_rows`: `fold_rows_*`, a kernel a layout as the unpackers'), and the 12-bit KV
  pages' release, save into an image and load from one (`fold_kv_release`, `_save`, `_load`), with an image's size (a host formula).
  Layout H's plain or coded choice (`f8h_plain`) is the engine's own state; the library is told it before each FP8 call of its own.
  `foldcheck` ends with the kernels of `kernels/fold.cu` it launched on the GPU it ran on and the names of those it did not (the routes
  of another GPU's class: stream-K on GeForce Ada, an A100's tiles of the 12-bit route MID, Hopper's TMA and wgmma products). Its case
  of the 12-bit route MID given (the library's product takes a route) runs the route's tiles that no GPU's own route reaches with the
  default plans: the 16-token one (a development build's `GLYD_MID_MIN` under 17 alone asks it) on any GPU of the route, and the four
  warpgroups' where a block's shared memory holds 153 KB (an H100). `GLYD_GPU_LIB=none` loads no library: the engine's own kernels
  alone, and a call they do not take fails naming the library's function it needed (`foldcheck`, which needs the library, refuses).
  What is left before the engine stops loading the library: every `GpuOps` method of `Lib` still ends in a `library_*` call for
  what its `fold` function does not take. Those are, by kind: arguments the library refuses (the engine has to refuse them itself, in
  the same words); an AMD build (its fold functions say false: nothing of them has run on an AMD GPU); a development build's tuning
  variable that asks for a plan not folded (a release build reads none); the layout of v0.27 and before in `unpack`; and Hopper's
  routes, which are written and have run on no H100 yet. Then `Lib::find()`'s
  callers, `GLYD_GPU_LIB`, the bundle's libraries and `install.sh`'s copy go, and `foldcheck` keeps the library as a development
  build's oracle.
- **The builds' time** (measured 2026-11-25 on a rented A10 machine of 30 processors, nvcc 12.8, a development build for sm_80 and
  sm_86; a `ps` sample a second). Both compiles are one translation unit each, and an architecture of a unit is one `cicc` followed by
  one `ptxas`, each a single thread. `build.rs` ran the engine's two architectures one after the other: 501 s (`cicc` 149 and 148 s,
  `ptxas` 101 and 102 s). `gpu/build_lib.sh` ran the library's six at once: 914 s (`cicc` 465 to 540 s, `ptxas` 269 to 326 s, then
  44 s of host compile), 24 of the 30 processors idle throughout; the library of the tree before the fold took 915 s, so the fold added
  the engine's time and not the library's. `scripts/ci_gpu_build.sh` now builds the library beside cargo's build, the engine's
  architectures at once (`GLYD_ENGINE_NVCC_THREADS=0`), and the library for the engine's architectures alone (`GLYD_GPU_ARCHS`): the
  library for sm_80 and sm_86 alone carries the same machine code for both, byte for byte, as the six built at once, and the whole
  build took 841 s (the library's nvcc 840 s beside cargo's build of 462 s, the engine's kernels 395 s in it; again with the library
  kept, 392 s). The engine's kernels compiled with their architectures at once (`--threads 0`) are the same bytes as compiled one
  after the other: four compiles of one source, two of each kind and cargo's among them, every architecture's machine code and the
  PTX alike. Six architectures at once are
  already six processes side by side, so a compile an architecture would end no sooner: the longest, sm_90a, is 540 s of `cicc` and
  326 s of `ptxas`. nvcc's `--split-compile` does not help: with 15 threads an architecture's `cicc` took 576 s against 523 (15
  threads for about 50 s, then one thread for the code generation), in twice the memory, and the machine code it made was not the
  same bytes. What would: several translation units an architecture, each its own `cicc` and `ptxas`. The engine's kernels are found
  by name, so its fatbin can be several modules (`fold.cu` by family beside `engine.cu`); the library's kernels are templates its
  host code launches, so a unit has to hold explicit instances that the host unit declares `extern template` (nvcc 12.8 compiles and
  runs that, and warns that it will want `-static-global-template-stub=false`).
