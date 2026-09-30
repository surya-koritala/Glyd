# glyd-gpu

Glyd's GPU weights from Rust: a bf16 model's matrices held compressed in GPU
memory (the tiered layout, about 10.8 bits a weight, or the 12-bit one) and
decoded on the GPU bit for bit, whole or inside their products.

`Library` is the CUDA library every release ships
(`libglyd_gpu_cudaN.so`, built by `gpu/build_lib.sh`) behind its C API
(`gpu/glyd_gpu.h`): loaded at run time and refused where its API version is
not this crate's, each function typed, a product's workspace query first,
its status a `Result`. Device pointers and streams are the caller's, as in
the C API; `cuda` has the few driver calls a caller without a CUDA runtime
of its own needs (a device's context, memory, copies, streams). Nothing is
linked at build time: the crate builds anywhere, and its calls return
`Error::Load` where the library or a GPU is not there (the library is
built for Linux). The library also routes: `route`
gives the kernel glyd takes for a matrix and a token count on a GPU (as
glyd.gpu's Linears take it), `linear` runs it. `Route::Split` (an A100
SXM's and a GH200's long 12-bit prompts: each matrix decoded ahead on SMs
set apart, the caller's cuBLAS on the rest) `linear` takes by the prompt
kernel; the crate declares the ring's functions (C API 6) but does not wrap
them yet: ask the route of the GPU's code plus `NO_SPLIT` for the routes
without it.

`pack` packs a matrix on the CPU in either layout, byte for byte as
glyd.gpu's `pack_mma` and `pack_mma12` do on the GPU (the 12-bit layout in
split byte: a weight's low byte as it is, its high byte a sign and an offset
from the matrix's base `hb`); `save` packs a bf16
checkpoint and saves it as glyd-v1 (glyd-v2 with a mixture of experts'
packs, glyd-v3 in the 12-bit layout, each pack's base as `"hb"` in
glyd.json), byte for byte as `python -m glyd.gpu
pack` saves it (Qwen3, Qwen2, Llama, Mistral, Granite and GraniteMoe; other
families: Python), and `verify` checks a saved one as `python -m glyd.gpu
verify` does: each file's tensors back to back, each a pack's buffer or of a
sha256 in glyd.json, every pack decoded and each of its tensors' sha256
checked, and each tensor saved as it is. The `glyd-gpu` command runs them
(`glyd pack`, `glyd verify` run it):

    glyd-gpu pack Qwen/Qwen3-8B qwen3-8b-glyd       # a directory, or a repo in the Hugging Face cache
    glyd-gpu pack Qwen/Qwen3-8B qwen3-8b-glyd12 --layout mma12   # the 12-bit layout (glyd-v3)
    glyd-gpu verify qwen3-8b-glyd [--device cuda:0]  # every pack decoded, every tensor's sha256 checked

Memory: a save holds the shard its writer writes and the one it fills
(about 5 GB each) and at most a shard's worth of bf16 weights being packed
past the one it waits for (about 15 GB at most for a model of several
shards; Qwen3-8B measured 11.4-13.6 GB peak RSS on 16 threads); a verify on the CPU, a pack and its matrix a
thread and one more (Qwen3-8B: 2.5-2.7 GB). On the AWS dev machine (a
g6.4xlarge: 16 vCPUs of an AMD EPYC 7R13), 16 threads:
Qwen3-4B-Instruct-2507 packed at 1.58-1.60 GB/s of bf16 tiered and
2.01-2.03 in the 12-bit layout (Qwen3-8B at 0.67-0.88, at the pace of the
disk its shards were written to), and Qwen3-8B verified in 8.8-11.8 s; the
same bytes as Python's for five models and eight tiny ones in both layouts
([benchmarks/gpu/l4-rust-2026-09-29](../benchmarks/gpu/l4-rust-2026-09-29)).

Its one dependency is sha2 (the store's).

```rust
let lib = glyd_gpu::Library::find()?;            // $GLYD_GPU_LIB, else libglyd_gpu_cuda13.so / cuda12
let ctx = glyd_gpu::cuda::Context::new(0)?;       // GPU 0's primary context, current on this thread (one a thread)
let w = glyd_gpu::Matrix { pack, rows, cols };    // a pack's arrays in device memory
unsafe { lib.unpack(&w, 0, rows, out.ptr(), 0, glyd_gpu::Stream::DEFAULT)? };
```

`examples/unpack.rs` decodes a matrix of a model saved by
`glyd.save_pretrained` on the GPU and checks it against the bf16 checkpoint
bit for bit (`gpu/examples/unpack.c` in Rust):

    cargo run --release -p glyd-gpu --example unpack -- qwen3-0.6b-glyd ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/

License: the Business Source License 1.1 (LICENSE), as `gpu/`; the Glyd
codec (the `glyd` crate) is BSD-3-Clause OR GPL-2.0.
