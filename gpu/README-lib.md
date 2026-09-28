# Glyd GPU: the library

A bf16 model's weights held compressed in GPU memory and decoded on the GPU
bit for bit, whole or inside the matrix products: Glyd's CUDA kernels behind
a C API, for engines in C, C++, Rust or any language with a C FFI. No Python,
no PyTorch.

- `libglyd_gpu_cudaN.so`: the kernels, CUDA N's runtime linked in: it needs
  the NVIDIA driver and the system's C and C++ libraries alone (Linux, glibc
  2.28 or later). Code for Ampere (sm_80, sm_86), Ada (sm_89), Hopper
  (sm_90a) and Blackwell (sm_100, sm_120), PTX for the GPUs after them.
- `glyd_gpu.h`: its C API. Every function; the arrays of a packed matrix;
  a product's workspace query, then its call; the stream; the return codes.
  Check `glyd_gpu_api_version()` against `GLYD_GPU_API_VERSION`: the
  library's functions are the header's where the two agree.
- `unpack.c`: an example in C alone. A matrix of a model saved by
  `glyd.save_pretrained` read back, decoded on the GPU with this library and
  checked against the bf16 checkpoint it was packed from, bit for bit.
- `LICENSE`: the Business Source License 1.1: source available, free for
  personal, educational, research and other non-commercial use; any
  commercial production use needs a license (suryakoritala1324@gmail.com);
  each version converts to Apache-2.0 four years after its release.

The example, built in this directory with the CUDA toolkit's headers (for
`cudaStream_t`) and runtime (for its own memory; `/usr/local/cuda`, or
yours), is how any C program builds against the library:

    gcc -O2 -I . -I /usr/local/cuda/include unpack.c -o unpack -L . -lglyd_gpu_cudaN -L /usr/local/cuda/lib64 -lcudart -Wl,-rpath,"$PWD:/usr/local/cuda/lib64"

A saved model comes from the glyd package (`pip install "glyd[gpu]"`); the
bf16 checkpoint it was packed from is in the Hugging Face cache:

    python -m glyd.gpu pack Qwen/Qwen3-0.6B qwen3-0.6b-glyd
    ./unpack qwen3-0.6b-glyd ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/ [PACK]

Rust and other languages call the same functions through their C FFI
(bindgen over `glyd_gpu.h`, or its declarations written out): device
pointers as raw pointers, a stream as the CUDA runtime's or driver's handle.
The source: https://github.com/surya-koritala/Glyd/tree/main/gpu#the-library
