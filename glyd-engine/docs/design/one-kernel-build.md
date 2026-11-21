# R17: one kernel build

Status: a plan, not started in code. Decisions (the coordinator, Nov 21): CUDA 12 stays (step 5); HIP stays a compile-only job until M5b, and the CUDA fold keeps the HIP path building (step 6); step 0 is R1a, #134 ("Part of #134"), started once the R3b pull requests that touch `ffi.rs` have merged and the other sessions working on `ffi.rs` have been asked; each step carries its own bit-for-bit check in the GPU job on glyd-pc (a CI job, no rental). It needs a GPU to validate every step (each step is compared bit for bit with the library on the GPU runner), which this repository's CI has only on glyd-pc.

## What there is today

The engine reaches the GPU through two paths:

1. its own fatbin: `glyd-engine/kernels/*.cu`, compiled by `build.rs` into one file the binary embeds, loaded with the driver API (`cuModuleLoadData`, launches by kernel name); no CUDA runtime is linked;
2. `libglyd_gpu_cuda13.so` or `_cuda12.so` (`gpu/glyd_gpu.cu`, built by `gpu/build_lib.sh`), opened at run time by `glyd-engine/src/ffi.rs` (47 entry points: the packed matrix products, the grouped-MoE products, the unpackers, the paged KV cache's write, gather and decode, the FP8 and FP8-hybrid ops, and three that say the library's versions and errors). The Python package `glyd-gpu` loads the same library through its C API (`gpu/glyd_gpu.h`).

The library is one translation unit of 12,217 lines: 55 `__global__` kernels, 112 templates (a kernel is instantiated per weight layout: tiered, compact, 12-bit, int6, int5, int4, int4h, fp4, fp4r, int3), 76 `<<<...>>>` launches and 104 CUDA runtime calls (`cudaMalloc`, `cudaStream*`, `cudaMemcpy*`: the library takes the caller's stream and workspace where it can). The 47 entry points themselves are thin: 452 lines of bodies in all (the largest, `glyd_gpu_f8_linear`, is 35), which pick the layout from the pack's tier words (`int6_words`, `tiered_words`, ...) and call a templated `*_run`. The weight is in the kernels and their launch plans, not in the C API.

## Why fold it

The hot operations go through a C ABI of a library that ships in lockstep with the engine (the engine bundle holds both), so the ABI adds a second place for a layout or a signature to drift (S16 checks them, #220), a second compiler invocation (CUDA 12 and 13 builds of the library; the engine's kernels are CUDA 13 only), a second set of architectures, and a runtime lookup that can fail at start (`Glyd's GPU library not found`).

## The steps

Each step lands alone, keeps `libglyd_gpu` loadable (the Python package needs it), and is checked on the GPU runner with the library as the oracle: the same pack and activations through the library and through the new path, outputs equal bit for bit (`glyd-engine check`/`kcheck` have the shape of that comparison; the logits tests in `glyd-engine/tests/logits` cover it end to end).

0. **A seam.** One Rust trait for the 47 operations (the device layer of R1, #134), implemented first by the library's `ffi.rs` table as it is, so the engine calls the trait and not the table. Nothing changes on the GPU; this is the part that can be reviewed and tested without one.
1. **Kernels in the fatbin, one family at a time.** (Each family's step adds its check to the GPU job in `gpu-tests.yml`: the library and the new path on the same pack and activations, bit for bit.) For a family (say the mma linear on the layouts the engine runs: tiered, int6, int5, int4), move the templated kernels it launches into `glyd-engine/kernels/` as explicit instantiations with `extern "C" __global__` names (`mma_linear_int4_m16`, ...), so the driver API finds them by name; the launch (grid, block, shared memory, the way-selection of `glyd_gpu_mma_way`) becomes Rust next to the kernel table. Keep the CUDA source of the kernels in one file that both builds include (`gpu/glyd_gpu_kernels.cuh`), so the library and the engine cannot differ.
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
