/*
 * Glyd GPU: the C API of libglyd_gpu_cuda12.so and libglyd_gpu_cuda13.so, which the glyd-gpu wheel holds in its glyd_gpu
 * directory (the CUDA runtime linked in: they need only the NVIDIA driver; code for Ampere and later). A bf16 matrix that
 * glyd saved compressed is multiplied from where it is held in GPU memory, or unpacked back to bf16 bit for bit. The
 * glyd_gpu Python package calls these through ctypes; C, C++, Rust or any language with a C FFI calls the same.
 *
 * License: Apache-2.0 (LICENSE).
 *
 * A packed matrix W [O, K] (O rows, K columns: the pack's "shape" in glyd.json) is passed as a glyd save holds it, as it
 * is: its arrays, the safetensors tensors <module>.glyd_<array>, copied to device memory, and its parameters, the
 * entries glyd.json holds for the pack, as uint32 values in host memory. glyd.json's "layout" says which functions
 * take it:
 *   "mma"    glyd_gpu_mma_*:    data, blocks, block_base; tiers
 *   "mma12"  glyd_gpu_mma12_*:  data, exc, exc_base; hb
 * The packs of a glyd-v3, glyd-v4 or glyd-v5 save are in the layout their entries in glyd.json name; the "mma" packs of glyd-v1 and glyd-v2
 * saves (glyd 0.27 and before) are unpacked by glyd_gpu_mma_unpack as before, and refused (cudaErrorInvalidValue) by
 * glyd_gpu_mma_linear: unpack them and pack the matrix again with the glyd_gpu package.
 * Nothing is changed, padded or expanded: the arrays are taken whole (a pack of merged Linears is one matrix, its rows
 * in the order glyd.json lists them).
 *
 * Every call: arrays in device memory, 16-byte aligned (cudaMalloc's are; bf16 as its bits, uint16_t) but for the
 * words and a workspace query's bytes, in host memory; sizes as values. Its kernels are launched on stream cs (0: the
 * default stream) of the current device and run after it returns, as any launch; every output, workspace and counter
 * is the caller's. It returns 0, or a cudaError_t: cudaErrorInvalidValue for an argument out of range,
 * cudaErrorNotSupported where the library has no kernel for the call on the current GPU (nothing launched), else the
 * launch's (glyd_gpu_error_string gives its text).
 *
 * A product has a workspace and counters: glyd_gpu_*_linear_workspace(its sizes, &bytes) first, on the device it will
 * run on, then glyd_gpu_*_linear with a buffer of at least those bytes (NULL where 0) and GLYD_GPU_DONE_COUNTERS(O, M)
 * int32 counters in device memory, zero before its first call (cudaMemset) and left zero by each. A workspace and a set
 * of counters serve one stream at a time.
 */
#ifndef GLYD_GPU_H
#define GLYD_GPU_H

#include <stddef.h>
#include <stdint.h>
#include <cuda_runtime_api.h> /* cudaStream_t */

/* The C API's version: the library's glyd_gpu_api_version() is this where its functions are this header's (check it: a
 * C FFI does not see a call's arguments). */
#define GLYD_GPU_API_VERSION 26

/* The counters of a product of W [O, K] and M tokens, at most. */
#define GLYD_GPU_DONE_COUNTERS(O, M) (1024 + (int64_t)(O) * ((int64_t)(M) / 1024 + 1))

/* A product's mode: the library runs it the way it measured fastest on the current GPU for M tokens. */
#define GLYD_GPU_AUTO (-1)

#ifdef __cplusplus
extern "C" {
#endif

/* The library's GLYD_GPU_API_VERSION. */
int glyd_gpu_api_version(void);
/* A status's text. */
const char* glyd_gpu_error_string(int status);

/* Y [M, O] = X W^T (+ bias [O]; NULL: none) for X [M, K] and Y [M, O] row-major bf16.
 * mode: GLYD_GPU_AUTO (any other value is reserved). cudaErrorNotSupported where the library has no kernel for this W
 * and M: unpack W (glyd_gpu_*_unpack) and multiply by a bf16 GEMM of your own. */
int glyd_gpu_mma_linear_workspace(int64_t O, int64_t K, int64_t M, int64_t mode, size_t* bytes);
int glyd_gpu_mma_linear(const void* data, const void* blocks, const void* block_base, const uint32_t* tiers, int64_t O,
                        int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y, int64_t mode,
                        void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs);
int glyd_gpu_mma12_linear_workspace(int64_t O, int64_t K, int64_t M, int64_t mode, size_t* bytes);
int glyd_gpu_mma12_linear(const void* data, const void* exc, const void* exc_base, const uint32_t* hb, int64_t O,
                          int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y, int64_t mode,
                          void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs);

/* W back to bf16 bit for bit, into out [O, K]: row0 0 and rows O (any other range is reserved). reserved: 0. */
int glyd_gpu_mma_unpack(const void* data, const void* blocks, const void* block_base, const uint32_t* tiers, int64_t K,
                        int64_t row0, int64_t rows, uint16_t* out, int64_t reserved, cudaStream_t cs);
int glyd_gpu_mma12_unpack(const void* data, const void* exc, const void* exc_base, const uint32_t* hb, int64_t K,
                          int64_t row0, int64_t rows, uint16_t* out, int64_t reserved, cudaStream_t cs);

#ifdef __cplusplus
}
#endif

#endif /* GLYD_GPU_H */
