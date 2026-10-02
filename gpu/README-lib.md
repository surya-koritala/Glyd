# Glyd GPU: the library

A bf16 model's weights held compressed in GPU memory and rebuilt on the GPU
bit for bit: Glyd's CUDA kernels behind a C API, for engines in C, C++, Rust
or any language with a C FFI. No Python, no PyTorch.

- `libglyd_gpu_cudaN.so`: the kernels. It carries its own CUDA runtime,
  linked in statically, so it needs only the NVIDIA driver and the system's
  C and C++ libraries (Linux, glibc 2.28 or later). Code for Ampere (sm_80,
  sm_86), Ada (sm_89), Hopper (sm_90a) and Blackwell (sm_100, sm_120), PTX
  for the GPUs after them. Blackwell's native code is built, not yet run
  on a GPU (an RTX PRO 6000 Blackwell Server Edition ran v0.21.0's
  compute_80 PTX).
- `glyd_gpu.h`: its C API. Every function; the arrays of a packed matrix;
  a product's workspace query, then its call; the stream; the return codes.
  Check `glyd_gpu_api_version()` against `GLYD_GPU_API_VERSION`: the
  library's functions are the header's where the two agree.
- `unpack.c`: an example in C alone. A matrix of a model saved by
  `glyd.save_pretrained` read back, unpacked on the GPU with this library and
  checked against the bf16 checkpoint it was packed from, bit for bit.
- `LICENSE`: the Business Source License 1.1: source available, free for
  personal, educational, research and other non-commercial use; any
  commercial production use needs a license (suryakoritala1324@gmail.com);
  each version converts to Apache-2.0 four years after its release.

Take the download whose CUDA major version is your toolkit's and runtime's:
cuda12 for CUDA 12, cuda13 for CUDA 13. Your program links its own CUDA
runtime (for its memory and streams), which runs beside the one inside the
library, on the same driver.

The example, built in this directory with the CUDA toolkit's headers (for
`cudaStream_t`) and runtime (for its own memory; `/usr/local/cuda`, or
yours), is how any C program builds against the library:

    gcc -O2 -I . -I /usr/local/cuda/include unpack.c -o unpack -L . -lglyd_gpu_cudaN -L /usr/local/cuda/lib64 -lcudart -Wl,-rpath,"$PWD:/usr/local/cuda/lib64"

A saved model comes from the glyd package (`pip install "glyd[gpu]"`); the
bf16 checkpoint it was packed from is in the Hugging Face cache:

    python -m glyd.gpu pack Qwen/Qwen3-0.6B qwen3-0.6b-glyd
    ./unpack qwen3-0.6b-glyd ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/ [PACK]

`glyd_gpu_mma_linear` and `glyd_gpu_mma12_linear` multiply by a packed
matrix with the kernel glyd takes for that many tokens on this GPU
(`glyd_gpu_mma_route` says which, as measured on an RTX 4080 SUPER, an
L4, an L40S, an A10, an A100 SXM4 40 GB, an H100 and a GH200, and the same
for a GPU of the same code, not measured there: an A100 SXM4 80 GB, an A800,
an H200, an H100 NVL; where glyd.gpu takes its long-prompt path, `linear` runs
the prompt kernel, on every GPU). Where K is not a
multiple of 64 no kernel takes those prompts: past 64 tokens (in the mma12
layout also from `GLYD_DEC_MIN` where that is set lower) they return
`cudaErrorNotSupported`, nothing launched: unpack the matrix
(`glyd_gpu_mma_unpack`) for a GEMM of your own.

The long-prompt path (the route SPLIT) is opt-in: `glyd_gpu_mma12_route` gives
it only for a GPU's code with `GLYD_GPU_WITH_SPLIT`, and
`glyd_gpu_mma12_linear`'s own route (-1) never is, so without the flag the
routes are v0.25.1's. Asked for, a prompt of the mma12 layout on an A100
SXM4 40 GB (769-4096 tokens, and to 8192 for a matrix whose O and K are both at
least 5120), a GH200 or an H100 SXM (2048-8192, such a matrix alone; an H200,
an H100 NVL and the PCIe cards not yet, until measured) takes it
(`glyd_gpu_mma12_route`, `glyd_gpu_mma12_split_sms`): its matrices are decoded
ahead into a ring of slots in your device memory while your cuBLAS multiplies
from the ring. The library does not link cuBLAS: give
`glyd_gpu_mma12_ring_linear` your handle and its functions
(`glyd_gpu_blas`); queue a prompt's matrices in their order
(`glyd_gpu_mma12_ring_queue`), then call each product, on the device the
ring was made on. Where it cannot run (a driver before CUDA 12.5 or one that
refuses it; a stream being captured) it returns `cudaErrorNotSupported`: take
the route the GPU's code without the flag gives. Measured on an A100-SXM4-40GB,
a GH200 480GB and an H100 80GB HBM3 (the SXM5, with Qwen3-14B's matrices;
Qwen3-32B's have O and K at least 5120 too: not run there). The library's rule
goes by the GPU's code alone, so any GPU of an A100 SXM4's code (compute
capability 8.0, no PCIe in its name: an A100 SXM4 80 GB, an A800, an A30), a
GH200's or an H100 SXM's (`GLYD_GPU_H100`: "H100" as a word in its name, not an
H100 NVL) gets the route where it is asked for, not measured there; glyd.gpu's
Linears ask for it only on an A100 SXM4, a GH200 or an H100 SXM, never on a MIG
slice. `glyd_gpu_mma12_linear` given SPLIT takes it by the prompt kernel.

Rust calls them through the `glyd-gpu` crate in the Glyd repository (the
library loaded at run time and held to its API version, each function
typed, device pointers and streams the caller's); other languages through
their C FFI: device pointers as raw pointers, a stream as the CUDA
runtime's or driver's handle.
The source: https://github.com/surya-koritala/Glyd/tree/main/gpu#the-library
