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
`Error::Load` where there is no GPU. No dependency.

```rust
let lib = glyd_gpu::Library::find()?;            // $GLYD_GPU_LIB, else libglyd_gpu_cuda13.so / cuda12
let ctx = glyd_gpu::cuda::Context::new(0)?;       // GPU 0's primary context, current on this thread
let w = glyd_gpu::Matrix { pack, rows, cols };    // a pack's arrays in device memory
unsafe { lib.unpack(&w, 0, rows, out.ptr(), 0, glyd_gpu::Stream::DEFAULT)? };
```

`examples/unpack.rs` decodes a matrix of a model saved by
`glyd.save_pretrained` on the GPU and checks it against the bf16 checkpoint
bit for bit (`gpu/examples/unpack.c` in Rust):

    cargo run --release -p glyd-gpu --example unpack -- qwen3-0.6b-glyd ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/

License: the Business Source License 1.1 (LICENSE), as `gpu/`; the Glyd
codec (the `glyd` crate) is BSD-3-Clause OR GPL-2.0.
