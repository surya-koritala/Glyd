/*
 * Glyd GPU: the C API of gpu/glyd_gpu.cu's kernels, which gpu/build_lib.sh
 * builds into libglyd_gpu_cuda12.so and libglyd_gpu_cuda13.so (the CUDA
 * runtime linked in: they need only the driver; code for Ampere and later).
 * A bf16 matrix held compressed in GPU memory: unpacked back to bf16 bit for
 * bit, or multiplied from where it is held. The glyd Python package calls these
 * through ctypes (bindings/python/glyd/gpu/_lib.py); C, C++, Rust or any
 * language with a C FFI calls the same. gpu/examples/unpack.c decodes a
 * saved model's matrix with them.
 *
 * License: Apache-2.0 (gpu/LICENSE).
 *
 * Every call: arrays in device memory (bf16 as its bits, uint16_t) but for a
 * layout's words (tiers[3], kt[3], vt[3], sym[4]) and a workspace query's
 * bytes, in host memory; sizes as values. Its kernels are launched on stream
 * cs (0: the default stream) of the current device and run after it returns,
 * as any launch; every output, workspace and counter is the caller's. It
 * returns 0, or a cudaError_t: cudaErrorInvalidValue for an argument out of
 * range, cudaErrorNotSupported where the kernel is not for this GPU, else the
 * launch's (glyd_gpu_error_string gives its text).
 *
 * A product with a workspace: glyd_gpu_NAME_workspace(its sizes, &bytes)
 * first, on the device it will run on, then glyd_gpu_NAME with a buffer of at
 * least those bytes (NULL where 0). Its done counters: int32, as many as it
 * says, zero before its first call (cudaMemset) and left zero by each. A
 * workspace and a set of counters serve one stream at a time.
 */
#ifndef GLYD_GPU_H
#define GLYD_GPU_H

#include <stddef.h>
#include <stdint.h>
#if defined(__HIPCC__) /* (an AMD build, through HIP: build_lib_hip.sh) */
#include <hip/hip_runtime_api.h>
typedef hipStream_t cudaStream_t;
#else
#include <cuda_runtime_api.h> /* cudaStream_t */
#endif

/* The C API's version, one more whenever a function's arguments, or what
 * they must hold, change: 2 from the prompt products' done counters,
 * glyd_gpu_hold and the decode's warps (0.21.0's library has no
 * glyd_gpu_api_version: 1); 3 from glyd_gpu_mma12_gemm_x3's counters, at
 * least 1024 (0.24.0); 4 from the routes, glyd_gpu_*_linear and a GPU's
 * class in its code (never released: builds of main alone); 5 from a change
 * of the mma12 layout (its data, exc and sym[4] as below): the mma12 packs of
 * 0.24 and before have other bytes and words, which the library refuses: pack
 * them again; 6 from the route SPLIT (glyd_gpu_ring_*, glyd_gpu_mma12_ring_*,
 * glyd_gpu_mma12_unpack_part, glyd_gpu_mma12_part_cus), a GPU's PCIe class
 * and GLYD_GPU_NO_SPLIT in its code (never released: builds of the route's
 * branch alone); 7 from GLYD_GPU_WITH_SPLIT in its place, the route opt-in,
 * and a GPU's PCIe class 5000 (0.26.0); 8 from glyd_gpu_kv_*, the lossless 12-bit KV cache; 9 from its escape pool
 * (glyd_gpu_kv_extra_bytes and the pool's arguments of glyd_gpu_kv_write, _unpack and _decode:
 * a page's arena is what its header leaves, and a page that needs more takes chunks of the pool); 10 from
 * glyd_gpu_kv_decode's plan from each sequence's own length (its argument `splits` in place of max_pages, and
 * glyd_gpu_kv_decode_shares), which lets a CUDA graph capture it; 11 from the raw pool (glyd_gpu_kv_second_bytes and the raw pool's
 * arguments of glyd_gpu_kv_write, _unpack, _gather and _decode); 14 from the places of the status's rings (glyd_gpu_kv_write's argument `ring`);
 * 15 from a change of the mma layout (its data, blocks, block_base and tiers[3] as below): the mma packs of 0.27 and before, whose tiers[0] has 0xFF in its
 * top byte, are unpacked (glyd_gpu_mma_unpack) as before and refused by every other function: pack them again;
 * 16 from glyd_gpu_f8_*, the lossless FP8 (e4m3) weights: pack, unpack, unpack of a list of a stack's matrices, and the product of 1 to 64
 *    tokens with the weights as they are held; 17 from glyd_gpu_mma_moe_placed, the experts' product over an expert cache's places;
 * 18 from glyd_gpu_kv_decode_from, the decode attention over the tokens from a sequence's first one on (a sliding window's);
 * 19 from glyd_gpu_kv_prefill, a prompt chunk's attention over its sequence's pages and its own bf16 rows;
 * 20 from glyd_gpu_kv_prefill_bf16 and glyd_gpu_kv_decode_bf16, the same attentions over a bf16 cache; 21 from glyd_gpu_kv_unpack's
 * skipping a negative page number (a list as long as the most a sequence can need, filled on the device);
 * 22 from glyd_gpu_kv_image_bytes, glyd_gpu_kv_save and glyd_gpu_kv_load, pages moved whole to another place and back (their escape chunks and raw slabs lent anew);
 * 23 from glyd_gpu_kv_decode_workspace's partial outputs on a 16-byte line (the workspace up to 8 bytes larger: an odd count of partials faulted before);
 * 24 from glyd_gpu_mma_moe_as, glyd_gpu_mma_moe_placed_as and glyd_gpu_mma12_moe_as (and their workspaces): the experts' product of a part of a layer
 *    (a GPU's slice of each expert's rows, an expert cache's places) cut as the whole layer's, so each row's sum is the whole layer's, bit for bit;
 * 25 from the experts' products splitting K by the layer's shape alone (an expert's whole rows and the layer's experts), never by the pairs a call holds:
 *    a pair's sum is the same whatever shares its call (other bits than 24's where a call of few pairs split K; the workspaces no larger). */
#define GLYD_GPU_API_VERSION 25

#ifdef __cplusplus
extern "C" {
#endif

/* The library's GLYD_GPU_API_VERSION: its functions are this header's where
 * the two agree (check it: a C FFI does not see a call's arguments). */
int glyd_gpu_api_version(void);
/* The CUDA runtime built in, e.g. 13000. */
int glyd_gpu_cuda_version(void);
/* A status's text. */
const char* glyd_gpu_error_string(int status);
/* Every function below is exported by its name, in every build of the library. (glyd_gpu_table, which the compiled
 * libraries of v0.27 to v0.32.1 had to reach the functions whose names they did not export, is gone.) */

/* ------------------------------------------------------------------------
 * The mma layouts: W [O, K] (O a multiple of 64, K of 16) held as O K / 1024
 * steps (64 rows by 16 columns each), in these arrays. glyd.gpu's pack_mma and
 * pack_mma12 and the glyd-gpu crate's pack make them, and glyd.save_pretrained
 * saves them; the kernels read them as they are, so a caller passes on the
 * arrays it was given.
 *
 * mma (about 10.8 bits a weight, the most memory off):
 *   data        uint8 [steps][1280]
 *   blocks      uint8: a step's block runs from block_base[step] to
 *               block_base[step + 1]; padded 128 bytes before the first and
 *               256 after the last
 *   block_base  int32 [steps + 1]; bit 31 of an entry is a step's flag,
 *               which every reader masks
 *   tiers[3]    host words, the matrix's own; the top byte of tiers[0] is the
 *               layout's number, 2 (0xFF: the layout of 0.27 and before)
 * mma12 (about 12 bits a weight, the faster one on an A10, an A100 and an H100):
 *   data        uint8 [steps][1536]
 *   exc         int32: a step's entries run from exc_base[step] to
 *               exc_base[step + 1]; zeros after, to a multiple of 4 (1 at
 *               least)
 *   exc_base    int32 [steps + 1]
 *   sym[4]      host words, the matrix's own (sym[1-3] zero, else
 *               cudaErrorInvalidValue)
 * The mma12 bytes and words changed from 0.24's: the library refuses its
 * packs, so pack such a matrix again.
 * ---------------------------------------------------------------------- */

/* Y [M, O] = X W^T (+ bias [O]; NULL: none) for 0 to 64 tokens (X [M, K], Y
 * row-major): a generation step's product, multiplied from the packed W.
 * done: O / 64 counters. */
int glyd_gpu_mma_gemm_workspace(int64_t O, int64_t K, int64_t M, size_t* bytes);
int glyd_gpu_mma_gemm(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base, const uint32_t tiers[3],
                      int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y,
                      void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs);
int glyd_gpu_mma12_gemm_workspace(int64_t O, int64_t K, int64_t M, size_t* bytes);
int glyd_gpu_mma12_gemm(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4],
                        int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y,
                        void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs);

/* The same for many tokens (a prompt; K a multiple of 64, x 16-byte aligned).
 * variant 0: chosen by M and the GPU; 1 to 3 (3: mma12 only): a fixed choice,
 * for measurement. done: (M + 127) / 128 x O / 64 counters. */
int glyd_gpu_mma_gemm_x1_workspace(int64_t O, int64_t K, int64_t M, int64_t variant, size_t* bytes);
int glyd_gpu_mma_gemm_x1(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base,
                          const uint32_t tiers[3], int64_t O, int64_t K, const uint16_t* x, int64_t M,
                          const uint16_t* bias, uint16_t* y, int64_t variant, void* workspace, size_t workspace_bytes,
                          int* done, cudaStream_t cs);
int glyd_gpu_mma12_gemm_x1_workspace(int64_t O, int64_t K, int64_t M, int64_t variant, size_t* bytes);
int glyd_gpu_mma12_gemm_x1(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4],
                            int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y,
                            int64_t variant, void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs);

/* Many tokens in the mma12 layout (K a multiple of 64; x, data and exc
 * 16-byte aligned): mid on Ampere and later (cudaErrorNotSupported before);
 * wg on Hopper (compute capability 9.0) alone. done: O / 64 counters (wg: at
 * least 1024, or O / 64 where that is more). */
int glyd_gpu_mma12_gemm_x2_workspace(int64_t O, int64_t K, int64_t M, size_t* bytes);
int glyd_gpu_mma12_gemm_x2(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4],
                            int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y,
                            void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs);
int glyd_gpu_mma12_gemm_x3_workspace(int64_t O, int64_t K, int64_t M, size_t* bytes);
int glyd_gpu_mma12_gemm_x3(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4],
                           int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y,
                           void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs);

/* Rows [row0, row0 + rows) of W (multiples of 64) back to bf16, into out
 * [rows, K]. warps 0: the default launch; else the launch is limited to that
 * many warps in all, so it can run beside a product on another stream.
 * glyd_gpu_mma12_unpack_part: the same for the route SPLIT, sized by sms (the
 * split size of glyd_gpu_mma12_part_cus; K a multiple of 64, out 16-byte
 * aligned). */
int glyd_gpu_mma_unpack(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base, const uint32_t tiers[3],
                        int64_t K, int64_t row0, int64_t rows, uint16_t* out, int64_t warps, cudaStream_t cs);
int glyd_gpu_mma12_unpack(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4],
                          int64_t K, int64_t row0, int64_t rows, uint16_t* out, int64_t warps, cudaStream_t cs);
int glyd_gpu_mma12_unpack_part(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4],
                                int64_t K, int64_t row0, int64_t rows, uint16_t* out, int64_t sms, cudaStream_t cs);

/* Rows of W [rows, K] by id (an embedding's lookup): out [n, K] is row ids[i] of W as bf16 bits, bit for bit, ids int64 in device memory
 * (K a multiple of 16, rows of 64, n >= 0; out and ids need no alignment beyond their types'). An id outside [0, rows) is a row of
 * zeros and reads nothing. One launch for any ids, so a CUDA graph can hold it (the table's functions mma_rows and mma12_rows: added to
 * version 14's table without changing any function's arguments; as every function's but glyd_gpu_mma_unpack's, glyd_gpu_mma_rows refuses a
 * pack of the layout of 0.27 and before: cudaErrorInvalidValue). */
int glyd_gpu_mma_rows(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base, const uint32_t tiers[3], int64_t K,
                      int64_t rows, const int64_t* ids, int64_t n, uint16_t* out, cudaStream_t cs);
int glyd_gpu_mma12_rows(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4], int64_t K,
                        int64_t rows, const int64_t* ids, int64_t n, uint16_t* out, cudaStream_t cs);

/* Holds stream cs for ns nanoseconds: work queued on cs after it starts that
 * much later. */
int glyd_gpu_hold(int64_t ns, cudaStream_t cs);

/* ------------------------------------------------------------------------
 * Routes: how glyd.gpu multiplies by a packed W [O, K] for M tokens on a
 * GPU, as measured there (gpu/README.md): the kernel it takes, or W unpacked
 * for a bf16 GEMM of the caller's own (cuBLAS) where that is the faster.
 * glyd.gpu's Linears take their routes from here. A GPU is a code: its
 * compute capability, major * 10 + minor, plus its class by name where the
 * compute capability does not tell GPUs apart: GLYD_GPU_GEFORCE with
 * "GeForce" in its name, GLYD_GPU_A10 with "A10" in it as a word (between
 * characters that are not ASCII letters, digits or '_': an A10, not an A10G,
 * A100 or A40), GLYD_GPU_L4 with "L4" in it as a word (an L4, not an L40S or
 * L40), GLYD_GPU_L40S with "L40S" in it as a word (not an L40), GLYD_GPU_PCIE
 * with "PCIe" in it in any case, GLYD_GPU_GH200 with "GH200" in it as a word,
 * GLYD_GPU_H100 with "H100" in it as a word, but not an H100 NVL (an H100 SXM
 * is "NVIDIA H100 80GB HBM3"; an H100 PCIe is PCIE's), else none. 1089: an RTX
 * 40; 3089: an L4; 4089: an L40S; 89: an L40 or RTX 6000 Ada; 2086: an A10; 86:
 * an A10G, A40 or RTX A6000; 80: an A100 SXM4; 5080: an A100 PCIe; 90: an H200
 * or H100 NVL; 7090: an H100 SXM; 6090: a GH200; 5090: an H100 PCIe.
 * GLYD_GPU_WITH_SPLIT added to a code asks for the route SPLIT, the faster
 * route for a long mma12 prompt on an A100 SXM, a GH200 and an H100 SXM, which
 * a caller runs through glyd_gpu_mma12_aux_linear (glyd.gpu's Linears do,
 * where it can run); without the flag no route is SPLIT (v0.25.1's routes).
 * ---------------------------------------------------------------------- */
#define GLYD_GPU_ROUTE_DECODE 0 /* W unpacked (glyd_gpu_*_unpack), then the caller's GEMM */
#define GLYD_GPU_ROUTE_GEMM 1   /* glyd_gpu_mma_gemm, glyd_gpu_mma12_gemm */
#define GLYD_GPU_ROUTE_MID 2    /* glyd_gpu_mma12_gemm_x2 */
#define GLYD_GPU_ROUTE_WG 3     /* glyd_gpu_mma12_gemm_x3; given, in the mma layout on Hopper to 64 tokens, glyd_gpu_mma_linear's */
#define GLYD_GPU_ROUTE_BIG 4    /* glyd_gpu_mma_gemm_x1, glyd_gpu_mma12_gemm_x1: variant 0 */
#define GLYD_GPU_ROUTE_AHEAD 5  /* DECODE for a caller; the faster route for a long prompt on GeForce Ada, an A10 and an L40S in glyd.gpu */
#define GLYD_GPU_ROUTE_SPLIT 6  /* mma12, opt-in: the faster route for a long prompt on an A100 SXM, a GH200 and an H100 SXM (glyd_gpu_mma12_aux_linear) */
#define GLYD_GPU_ROUTE_GEMM_CC12 7 /* opt-in, never a GPU's own: GEMM, and on compute capability 12 (RTX 50) the mma layout's products of 9-64 tokens as measured faster there */
#define GLYD_GPU_GEFORCE 1000   /* a GPU's class: "GeForce" in its name */
#define GLYD_GPU_A10 2000       /* a GPU's class: "A10" in its name as a word */
#define GLYD_GPU_L4 3000        /* a GPU's class: "L4" in its name as a word */
#define GLYD_GPU_L40S 4000      /* a GPU's class: "L40S" in its name as a word */
#define GLYD_GPU_PCIE 5000      /* a GPU's class: "PCIe" in its name */
#define GLYD_GPU_GH200 6000     /* a GPU's class: "GH200" in its name as a word */
#define GLYD_GPU_H100 7000      /* a GPU's class: "H100" in its name as a word, not an H100 NVL (an H100 PCIe is PCIE's) */
#define GLYD_GPU_WITH_SPLIT 1048576 /* a flag in a GPU's code (1 << 20): its routes with SPLIT, which the caller runs (opt-in) */

/* The current device's GPU as the routes take it: its code. */
int glyd_gpu_gpu(int* gpu);
/* The route of W [O, K] for M tokens on gpu, in the mma layout or the mma12
 * one; last (NULL: not asked): the last token count from M on that takes it
 * (INT64_MAX: every one past M). These environment variables move the token
 * counts where a route starts or stops; each is read once a process, at the
 * first route (a later change has no effect), as a whole number in base 10
 * (spaces around it, a sign), else taken as unset (the glyd package refuses
 * such a value at import):
 *   GLYD_WG_MIN, GLYD_WG_MAX  raise or lower the token counts where Hopper's
 *                             route for many tokens a step (WG) starts and stops
 *   GLYD_MID_MIN              raises or lowers the token count where the route
 *                             for many tokens a step (MID) starts, on Ampere and
 *                             Ada
 *   GLYD_DEC_MIN              raises or lowers the token count where an mma12
 *                             prompt takes the DECODE route, on any GPU (0 or
 *                             unset: the GPU's own) */
int glyd_gpu_mma_way(int64_t gpu, int64_t O, int64_t K, int64_t M, int* route, int64_t* last);
int glyd_gpu_mma12_way(int64_t gpu, int64_t O, int64_t K, int64_t M, int* route, int64_t* last);
/* The split size of the route SPLIT for W [O, K] and M tokens on gpu (a code
 * with GLYD_GPU_WITH_SPLIT): pass it as sms to glyd_gpu_aux_part,
 * glyd_gpu_mma12_aux_queue and glyd_gpu_mma12_aux_linear. 0 where the route is
 * not SPLIT (and without the flag). The route runs a long mma12 prompt's
 * products faster than the others where a forward pass took at least 2% less
 * time by it (gpu/README.md): on an A100 SXM from 769 to 4096 tokens, and up
 * to 8192 tokens for 14B and larger models; on a GH200 and an H100 SXM from
 * 2048 to 8192 tokens, for 14B and larger models (an H200, an H100 NVL and the
 * PCIe cards: never, until measured). It is opt-in: glyd_gpu_*_route give it
 * only for a code with GLYD_GPU_WITH_SPLIT, and glyd_gpu_*_linear's own route
 * (-1) never is. These environment variables work on any GPU from Ampere and
 * any model, and are read as the routes' others:
 *   GLYD_SPLIT_MIN, GLYD_SPLIT_MAX  raise or lower the token counts where the
 *                                   route starts and stops (0 or unset: the
 *                                   GPU's own; GLYD_SPLIT_MIN -1 turns it off)
 *   GLYD_SPLIT_SMS                  sets the split size (0 or unset: the GPU's
 *                                   own)
 * K a multiple of 64. */
int glyd_gpu_mma12_part_cus(int64_t gpu, int64_t O, int64_t K, int64_t M, int64_t* sms);

/* Y [M, O] = X W^T (+ bias) by a route (negative: the current GPU's for M):
 * its kernel, its arguments as that kernel's; DECODE and AHEAD by the prompt
 * kernel (BIG, variant 0) on every GPU, Hopper's too (not measured there). No
 * kernel here takes DECODE or AHEAD where K is not a multiple of 64, and a
 * matrix of such a K takes one of them for M past 64 on every GPU and in
 * either layout (in the mma12 layout also from GLYD_DEC_MIN tokens where that
 * is set lower): there these functions and their workspace queries return
 * cudaErrorNotSupported, nothing launched, and the caller unpacks W
 * (glyd_gpu_*_unpack, [O, K] bf16) for a GEMM of its own, Y = X W^T + bias, as
 * glyd.gpu does. done: (M + 127) / 128 x O / 64 counters, and at least 1024
 * (the WG route's, as glyd_gpu_mma12_gemm_x3's). */
int glyd_gpu_mma_linear_workspace(int64_t O, int64_t K, int64_t M, int64_t route, size_t* bytes);
int glyd_gpu_mma_linear(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base, const uint32_t tiers[3],
                        int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y,
                        int64_t route, void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs);
int glyd_gpu_mma12_linear_workspace(int64_t O, int64_t K, int64_t M, int64_t route, size_t* bytes);
int glyd_gpu_mma12_linear(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4],
                          int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y,
                          int64_t route, void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs);

/* ------------------------------------------------------------------------
 * The route SPLIT: the faster route for a long mma12 prompt on an A100 SXM, a
 * GH200 and an H100 SXM, opt-in (glyd_gpu_*_route give it only for a code with
 * GLYD_GPU_WITH_SPLIT). You give the library a scratch buffer (a ring) and your
 * cuBLAS handle. Queue a prompt's matrices in the order their products will be
 * called, as its first product starts (or a few ahead of its products as they
 * go), then call each product with glyd_gpu_mma12_aux_linear; a product whose
 * matrix is not next in the queue drops it and runs on its own. None of these
 * calls waits on the host. Its products are cuBLAS's own on the unpacked bf16:
 * not bit for bit a whole-matrix product. A ring serves one host thread and one
 * device (the current one when made: cudaErrorInvalidDevice with another
 * current); cudaErrorNotSupported where the route cannot run (a driver before
 * CUDA 12.5 or one that refuses it; stream cs being captured into a CUDA
 * graph): take the route the code without GLYD_GPU_WITH_SPLIT gives.
 * ---------------------------------------------------------------------- */
typedef struct glyd_gpu_ring glyd_gpu_ring;

/* The caller's cuBLAS, which the library does not link: a handle and its
 * functions (the calls' own types, int for their enums and status; get_* and
 * set_workspace may be NULL). A call changes the handle's stream, SM-count
 * target and, where one is given here, workspace (used by these calls alone)
 * while it runs, and puts back the stream and target it had where get_stream and
 * get_sm_count_target read them (the target is set only where it can be read
 * back). A cuBLAS status s is returned as GLYD_GPU_BLAS_ERROR + s. */
typedef struct glyd_gpu_blas {
    void* handle; /* cublasHandle_t */
    int (*gemm_ex)(void* handle, int transa, int transb, int m, int n, int k, const void* alpha, const void* A, int Atype, int lda,
                   const void* B, int Btype, int ldb, const void* beta, void* C, int Ctype, int ldc, int computeType,
                   int algo); /* cublasGemmEx */
    int (*set_stream)(void* handle, cudaStream_t stream);           /* cublasSetStream */
    int (*get_stream)(void* handle, cudaStream_t* stream);          /* cublasGetStream */
    int (*set_workspace)(void* handle, void* workspace, size_t bytes); /* cublasSetWorkspace */
    int (*set_sm_count_target)(void* handle, int sms);              /* cublasSetSmCountTarget */
    int (*get_sm_count_target)(void* handle, int* sms);             /* cublasGetSmCountTarget */
    void* workspace;
    size_t workspace_bytes;
} glyd_gpu_blas;
#define GLYD_GPU_BLAS_ERROR 10000

/* A ring over the scratch buffer you give it (bytes long, 16-byte aligned), used
 * in parts of slot_bytes (a multiple of 256; 3 to 16 of them), on the current
 * device. */
int glyd_gpu_aux_create(void* buffer, size_t bytes, size_t slot_bytes, glyd_gpu_ring** ring);
/* Destroys the ring: its streams are waited for and let go with its events (on
 * its device, whichever is current); the first failure's status (then its buffer
 * may still be written: keep it). */
int glyd_gpu_aux_destroy(glyd_gpu_ring* ring);
/* Sets the ring up for a split of size sms (glyd_gpu_mma12_part_cus's), where it
 * is not yet, and keeps it; the sizes of its two sides. */
int glyd_gpu_aux_part(glyd_gpu_ring* ring, int64_t sms, int64_t* decode_sms, int64_t* product_sms);
/* The queue dropped: stream cs waits for everything the ring has queued. */
int glyd_gpu_aux_reset(glyd_gpu_ring* ring, cudaStream_t cs);
/* W [O, K] queued after the matrices before it (sms: the queue's split size; a
 * queue's matrices share one). */
int glyd_gpu_mma12_aux_queue(glyd_gpu_ring* ring, int64_t sms, const uint8_t* data, const uint32_t* exc,
                              const int32_t* exc_base, const uint32_t sym[4], int64_t O, int64_t K);
/* Y [M, O] = X W^T (+ bias) for a long prompt: W is the next matrix in the queue
 * (else it is queued now), and your cuBLAS runs the products (bf16, fp32 sums; a
 * bias written into Y first, then added by cuBLAS). X is taken once the work
 * queued on cs before is done, and cs waits for Y. x and y 16-byte aligned, rows
 * of K and O. */
int glyd_gpu_mma12_aux_linear(glyd_gpu_ring* ring, int64_t sms, const uint8_t* data, const uint32_t* exc,
                               const int32_t* exc_base, const uint32_t sym[4], int64_t O, int64_t K, const uint16_t* x,
                               int64_t M, const uint16_t* bias, uint16_t* y, const glyd_gpu_blas* blas, cudaStream_t cs);

/* ------------------------------------------------------------------------
 * Mixtures of experts: a layer's E experts' matrices [O, K] as one pack
 * [E O, K] (in the tiered layout, or the six-, five- and four-bit block layouts,
 * which tiers[0]'s top byte names: E packs of [O, K] end to end are that pack). A
 * token's k choices are its pairs (P = T k for T tokens; pair j:
 * token j / k, choice j mod k).
 * ---------------------------------------------------------------------- */

/* The plan (int32 [2 + 2E + P]): the pairs, whose experts are ids (int64
 * [P]), sorted by expert; E up to 12288. */
int glyd_gpu_moe_way(const int64_t* ids, int64_t P, int64_t E, int32_t* plan, cudaStream_t cs);

/* The experts' product by the plan: X [T, K] (gather: a pair takes its
 * token's row) or [P, K] (the pairs in the plan's order). act 0: Y [P, O] in
 * the plan's order (+ bias [E, O]; NULL: none); act 1 (SiLU) or 2 (GELU,
 * tanh): Y [P, O / 2] = act(gate) up, the gate an expert's first O / 2 rows
 * (O a multiple of 128), each + its bias. w (act 0; bf16, or fp32 where
 * wf32), the pairs' weights, with ids (int64 [T, k]): Y [T, O], a token's k
 * rows times their weights, added (weighted: w given, for the workspace
 * query). done: O / 64 (O / 128 with act) x min(E, P) counters. */
int glyd_gpu_mma_moe_workspace(int64_t E, int64_t O, int64_t K, int64_t T, int64_t k, int64_t act, int64_t weighted,
                               size_t* bytes);
int glyd_gpu_mma_moe(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base, const uint32_t tiers[3],
                     int64_t E, int64_t O, int64_t K, const uint16_t* x, int64_t T, int64_t k, int64_t gather,
                     const int32_t* plan, int64_t act, const uint16_t* bias, const void* w, int64_t wf32,
                     const int64_t* ids, uint16_t* y, void* workspace, size_t workspace_bytes, int* done,
                     cudaStream_t cs);
/* glyd_gpu_mma_moe over E places of an expert cache (the tiered layout alone; an engine copies an expert of a stack's pack into a place):
 * place e's O K / 1024 steps of data as the stack's ([E][O K / 1024][1280]), its block entries its own O K / 1024 + 1 (block_base
 * [E][O K / 1024 + 1], bit 31 the flag as a pack's), offsets from its blocks at blocks + e stride (stride a multiple of 4; a place's
 * blocks with 128 bytes before its first and 256 after its last, and each block's start as a multiple of 4 where the pack had it so).
 * The rest, and the workspace, as glyd_gpu_mma_moe's. */
int glyd_gpu_mma_moe_placed(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base, const uint32_t tiers[3], int64_t stride,
                            int64_t E, int64_t O, int64_t K, const uint16_t* x, int64_t T, int64_t k, int64_t gather, const int32_t* plan,
                            int64_t act, const uint16_t* bias, const void* w, int64_t wf32, const int64_t* ids, uint16_t* y, void* workspace,
                            size_t workspace_bytes, int* done, cudaStream_t cs);
/* The same three (and their workspaces) for a call that holds a part of a layer: wO the rows an expert has in the whole layer (this call's O a GPU's
 * slice of them), wE the layer's experts (this call's E an expert cache's places); 0: the call's own. The work is cut as the whole layer's call would
 * cut it (its split of K over blocks, its kernel), so each row of the output is the whole layer's sum, bit for bit. */
int glyd_gpu_mma_moe_as_workspace(int64_t E, int64_t O, int64_t K, int64_t T, int64_t k, int64_t act, int64_t weighted, int64_t wO, int64_t wE,
                                  size_t* bytes);
int glyd_gpu_mma_moe_as(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base, const uint32_t tiers[3], int64_t E, int64_t O,
                        int64_t K, const uint16_t* x, int64_t T, int64_t k, int64_t gather, const int32_t* plan, int64_t act, const uint16_t* bias,
                        const void* w, int64_t wf32, const int64_t* ids, uint16_t* y, void* workspace, size_t workspace_bytes, int* done,
                        cudaStream_t cs, int64_t wO, int64_t wE);
int glyd_gpu_mma_moe_placed_as(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base, const uint32_t tiers[3], int64_t stride,
                               int64_t E, int64_t O, int64_t K, const uint16_t* x, int64_t T, int64_t k, int64_t gather, const int32_t* plan,
                               int64_t act, const uint16_t* bias, const void* w, int64_t wf32, const int64_t* ids, uint16_t* y, void* workspace,
                               size_t workspace_bytes, int* done, cudaStream_t cs, int64_t wO, int64_t wE);
int glyd_gpu_mma12_moe_as_workspace(int64_t E, int64_t O, int64_t K, int64_t T, int64_t k, int64_t act, int64_t weighted, int64_t wO, int64_t wE,
                                    size_t* bytes);
int glyd_gpu_mma12_moe_as(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4], int64_t E, int64_t O,
                          int64_t K, const uint16_t* x, int64_t T, int64_t k, int64_t gather, const int32_t* plan, int64_t act, const uint16_t* bias,
                          const void* w, int64_t wf32, const int64_t* ids, uint16_t* y, void* workspace, size_t workspace_bytes, int* done,
                          cudaStream_t cs, int64_t wO, int64_t wE);
int glyd_gpu_mma12_moe_workspace(int64_t E, int64_t O, int64_t K, int64_t T, int64_t k, int64_t act, int64_t weighted,
                                 size_t* bytes);
int glyd_gpu_mma12_moe(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4],
                       int64_t E, int64_t O, int64_t K, const uint16_t* x, int64_t T, int64_t k, int64_t gather,
                       const int32_t* plan, int64_t act, const uint16_t* bias, const void* w, int64_t wf32,
                       const int64_t* ids, uint16_t* y, void* workspace, size_t workspace_bytes, int* done,
                       cudaStream_t cs);

/* Exact: the experts the plan's P pairs hit back to bf16, into their rows of
 * out [E O, K]; the rest of out left as it is. */
int glyd_gpu_mma_moe_unpack(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base,
                            const uint32_t tiers[3], int64_t E, int64_t O, int64_t K, int64_t P, const int32_t* plan,
                            uint16_t* out, cudaStream_t cs);
int glyd_gpu_mma12_moe_unpack(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4],
                              int64_t E, int64_t O, int64_t K, int64_t P, const int32_t* plan, uint16_t* out,
                              cudaStream_t cs);

/* ------------------------------------------------------------------------
 * Attention for one new token a sequence over a KV cache in the mma layout
 * (gpu/kv.py): q and out [pairs x G, D] (a pair: a sequence's KV head; G, 1 to
 * 16, the queries it serves; D 64 or 128). The keys a pack of P pages of
 * [pairs x 64 tokens, D] (kd, kb, kbb, kt), the values one of [pairs x D, 64
 * tokens] (vd, vb, vbb, vt), then a tail of tlen (0 to 63) tokens as they
 * are, tk and tv [pairs, tlen, D]. scale: the scores'. done: pairs counters.
 * ---------------------------------------------------------------------- */
int glyd_gpu_attn_decode_workspace(int64_t D, int64_t tlen, int64_t pairs, int64_t P, size_t* bytes);
int glyd_gpu_attn_decode(const uint16_t* q, int64_t D, const uint8_t* kd, const uint8_t* kb, const int32_t* kbb,
                         const uint32_t kt[3], const uint8_t* vd, const uint8_t* vb, const int32_t* vbb,
                         const uint32_t vt[3], const uint16_t* tk, const uint16_t* tv, int64_t tlen, int64_t pairs,
                         int64_t G, int64_t P, double scale, uint16_t* out, void* workspace, size_t workspace_bytes,
                         int* done, cudaStream_t cs);

/* ------------------------------------------------------------------------
 * The lossless 12-bit KV cache (glyd_gpu's vLLM plugin, kv: lossless): a pool of pages, a page the K and V of 16
 * consecutive tokens of H kv heads (1 to 32) of D = 128 channels, every bf16 value held in 12 bits and read back bit
 * for bit. D = 256 (H 1 to 16) is held as 2 H heads of 128, a head's channels 0-127 the page's head 2 h and 128-255 its
 * head 2 h + 1 (kbase, vbase [H][256] are those heads' [2 H][128]); D = 512 (H 1 to 8) as 4 H heads of 128, the same way (decode: a page
 * is four warps', their scores summed in the order of the halves): every function below takes the model's H and D, and
 * the page's heads are 2 H (D = 256) or 4 H (D = 512) where it says H (kv_second_bytes takes the page's heads). A value is its bf16 low byte as it is and a 4-bit code for the high byte: the sign and an offset 0-7 from
 * a base, the window of 8 values of exponent >> 1 that holds the most of the (layer, head, channel)'s values in other
 * text (kbase, vbase: uint8 [H][D] a layer, each 0 to 120); a value outside its window is an exception: an entry in its
 * head's header (three inline), else in its page's list of entries, whose first A are in the page (its arena) and the rest
 * in chunks of an escape pool a page takes as it needs them. Every 16-bit pattern (NaN, infinity, denormals, both zeros)
 * comes back as it went in. A page is, at page_stride bytes from the one before (a multiple of 16, at least
 * glyd_gpu_kv_leaf_bytes; the pool 16-byte aligned):
 *   16 H bytes     H head headers: u32 count (the head-page's entries, K and V), u32 e0, e1, e2
 *   4 (1 + 16 + A) u32 used (the page's entries past the inline ones), u32 dir[16] (the chunks of the escape pool the page
 *                  holds, a chunk's number + 1), then the arena of A u32 entries (up to a multiple of 128 bytes with the
 *                  headers: A = 15 costs nothing for 8 heads)
 *   H x 6,144      a head's K tile of 3,072 bytes (2,048 low bytes, 1,024 of codes, in the order a lane of a warp takes
 *                  them as operands of mma.sync.m16n8k16), then its V tile
 * bytes (H = 8, A = 15: 49,408, against 65,536 for bf16). A page holds nothing a caller may assume: a recycled
 * page's bytes are a previous owner's, and glyd_gpu_kv_write clears a page when its first token is written (giving the
 * chunks it holds back to the escape pool). An escape pool is nchunks chunks of 64 entries, in one buffer of
 * glyd_gpu_kv_extra_bytes bytes (16-byte aligned, zero before its first use) that every layer's calls of a process
 * share (a layer's number tells its pages apart): a page takes up to 16 chunks, 1,024 entries past its arena.
 * One stream: where D has GLYD_GPU_KV_ONE_STREAM (1 << 16) set the tensor is held in the K tile's layout alone (a latent, or keys that are the values): glyd_gpu_kv_leaf_bytes, _write, _release and _unpack
 * take it (write reads `key` and no value; unpack writes out_k only, the same [n][16][H][D] bf16; both ignore the value arguments, which may be null), a page has H head-pages of one 3,072-byte tile in place
 * of two (D = 128, H = 5, A = 15: 16,384 bytes, against 32,768 for two streams), and _gather, _work_put, _decode and _decode_workspace refuse it. The tables are kbase's alone.
 * A page that has no place for an entry is stored raw, in a slab of the raw pool (nslabs slabs, in one buffer of
 * glyd_gpu_kv_second_bytes bytes, 16-byte aligned, zero before its first use, shared by every layer's calls of a process); the
 * readers take either kind. A page that finds no slab is void: its block is named in the status's ring, and what reads it is not to
 * be used; a block with a raw page is named in the raw ring.
 * ---------------------------------------------------------------------- */

#define GLYD_GPU_KV_ONE_STREAM (1 << 16)

/* A page's bytes for H heads, D and an arena of A entries (0 to 1023); cudaErrorInvalidValue where they are out of range. */
int glyd_gpu_kv_leaf_bytes(int64_t H, int64_t D, int64_t A, int64_t* bytes);

/* The bytes of an escape pool of nchunks chunks (0 to 2^30: none to many). */
int glyd_gpu_kv_extra_bytes(int64_t nchunks, int64_t* bytes);

/* The bytes of a raw pool of nslabs slabs for pages of H heads (a slab is H 2,048 bytes: 0 to 2^30 slabs). */
int glyd_gpu_kv_second_bytes(int64_t nslabs, int64_t H, int64_t* bytes);

/* One step's tokens into the pool: key and value [N, H, D] bf16 (token n's row at key + n * key_stride elements, a
 * head's D channels contiguous; strides even, at least H D, the pointers 4-byte aligned), slots [N] int64: token n's
 * slot, page * 16 + offset (page p at pool + p * page_stride), a negative slot skipped (padding). A step's tokens of
 * one page must follow one another, their slots consecutive (the tokens of a request do; a token that is the only one of its page in the step, as a decode step's are, takes a path of its own, and GLYD_KV_LEAN=0 sends every token through
 * the one of a run of any length). status: uint32 [16 + 2 ring] in
 * device memory (zero them before the first call; a call adds to them): [0] the exceptions that found no place in a
 * page (its arena and its 16 chunks full, or the escape pool out of chunks: the page was then made raw, or void), [1] the most entries
 * any page has been given, [2] the pages whose arena was used, [3] the entries appended, [4] the escape pool's chunks
 * taken now, [5] the most taken at once, [6] the pages that took a chunk, [7] directory words of a recycled page that the
 * pools did not list as the page's (none where the memory was zero or the pool the page's own; a stale word names no
 * chunk or slab that is freed), [8] the raw pool's slabs taken now, [9] the most taken at once, [10] the pages made raw, [11] the
 * pages made void (no slab for them: the pages written hold a value not as it was; the caller fails what holds them), [12] the
 * ring's head: the number of blocks lost so far, the block of the n-th is [16 + n % ring] - 1, [13] the raw ring's head: the blocks
 * that have a raw page, the block of the n-th is [16 + ring + n % ring] - 1 (ring: 1 to 2^24 places each, status has 16 + 2 ring words). lost, nlost: a bitmap of nlost bits (uint32
 * words, zero before the first call; null, 0: none), two bits a block (2 b: it has a void page, 2 b + 1: a raw page), that makes each ring name a
 * block once while it stays so; the bits are cleared when the block's first token of layer 0 is written again. escape,
 * nchunks: the escape pool (nchunks 0: none: a page's entries are its arena's), raw, nslabs: the raw pool (nslabs 0: none:
 * a page that overflows is void), layer: this layer's number (0 to 254: its pages' owner tag in the pools). A token written again (a slot written before) replaces the one before, its old entries killed; a
 * page's first token written again clears the page, so a rewrite from there on writes every later token of the page
 * again too (a sequence's rewrite is from a position to its end). */
int glyd_gpu_kv_write(const uint16_t* key, int64_t key_stride, const uint16_t* value, int64_t value_stride, const int64_t* slots,
                      int64_t N, uint8_t* pool, int64_t page_stride, const uint8_t* kbase, const uint8_t* vbase, int64_t H,
                      int64_t D, int64_t A, uint8_t* escape, int64_t nchunks, uint8_t* raw, int64_t nslabs, int64_t layer,
                      uint32_t* status, uint32_t* lost, int64_t nlost, int64_t ring, cudaStream_t cs);

/* A page moved whole, its 12 bits as they are (a sequence swapped to the host and back): its image is the page's bytes (glyd_gpu_kv_leaf_bytes), then the
 * entries of the escape chunks its directory names, 256 bytes a chunk at the directory's place (16 places), then its raw slab's bytes where it is raw (kv_second_bytes of one slab):
 * glyd_gpu_kv_image_bytes apart, 16-byte aligned. Only what the page holds is written: the image's places for chunks and a slab it does not have are left as they were. */
int glyd_gpu_kv_image_bytes(int64_t H, int64_t D, int64_t A, int64_t* bytes);

/* The pages of the blocks pages[0 .. n) (int32, below nblocks) into images 0 .. n at out (memory the device writes: a mapped host buffer, or the device's), as they are; moved[j]
 * (uint32, null: none) the bytes written for page j. A page whose chunks or slab the pools do not list as its own (one never written, or void) is saved void (its image's count
 * word has the void flag): it loads void. The pages are read, not changed; pool, H, D, A, escape and raw as glyd_gpu_kv_write's, layer the pages' (their owner tag). */
int glyd_gpu_kv_save(const uint8_t* pool, int64_t page_stride, int64_t nblocks, const int32_t* pages, int64_t n, int64_t H, int64_t D, int64_t A, const uint8_t* escape,
                     int64_t nchunks, const uint8_t* raw, int64_t nslabs, int64_t layer, uint8_t* out, uint32_t* moved, cudaStream_t cs);

/* The images 0 .. n at img (glyd_gpu_kv_save's, of pages of this H, D and A) into the pages of the blocks dst[0 .. n) (int32, below nblocks; any blocks, apart): each page's chunks or
 * slab first given back to their pools as a page's first token gives them, then the image's bytes, with chunks of the escape pool and a slab of the raw pool lent to the block for
 * what the image holds (owner tag: layer and block), the page's directory naming them. Read back, a loaded page holds the values the saved one did, bit for bit. A page that
 * finds no chunk or slab, or whose image is void, is void: its block is named in the status's ring as glyd_gpu_kv_write names one (status[11] counts it); a raw page's block
 * in the raw ring. status and ring as glyd_gpu_kv_write's. */
int glyd_gpu_kv_load(const uint8_t* img, const int32_t* dst, int64_t n, uint8_t* pool, int64_t page_stride, int64_t nblocks, int64_t H, int64_t D, int64_t A, uint8_t* escape,
                     int64_t nchunks, uint8_t* raw, int64_t nslabs, int64_t layer, uint32_t* status, int64_t ring, cudaStream_t cs);

/* The pages of the blocks listed (int32 [n] block numbers, below nblocks: page p at pool + p * page_stride) cleared as a page's first token clears them: a page's chunks or its slab back to their
 * pools (by owner), its count and directory zero, and the block's bits in `lost` (nlost bits, as kv_write's). For blocks that no request holds and whose content is not to be read again;
 * layer, status, escape and raw as kv_write's. */
int glyd_gpu_kv_release(uint8_t* pool, int64_t page_stride, int64_t nblocks, const int32_t* blocks, int64_t n, int64_t H, int64_t D, int64_t A, uint8_t* escape, int64_t nchunks, uint8_t* raw,
                        int64_t nslabs, int64_t layer, uint32_t* status, uint32_t* lost, int64_t nlost, cudaStream_t cs);

/* The pages pages[0 .. n) (int32) back to bf16: out_k and out_v [n, 16, H, D]; a negative pages[j] is a place no page is in: nothing is written for it. */
int glyd_gpu_kv_unpack(const uint8_t* pool, int64_t page_stride, const int32_t* pages, int64_t n, const uint8_t* kbase,
                       const uint8_t* vbase, int64_t H, int64_t D, int64_t A, const uint8_t* escape, int64_t nchunks,
                       const uint8_t* raw, int64_t nslabs, uint16_t* out_k, uint16_t* out_v, cudaStream_t cs);

/* The pages pages[0 .. n) (int32) decoded into a paged bf16 cache, as vLLM's FlashAttention reads one: out [blocks, 16, H, 2 D] bf16 (4-byte aligned), page j into the block
 * dst[j] (int32; the caller keeps them in range and apart): token t of head h of block b has its keys at ((b 16 + t) H + h) 2 D elements and its values D after (D = 256: the
 * model's heads, each its 256 channels whole). A negative pages[j] is a place no page is in: skipped. */
int glyd_gpu_kv_gather(const uint8_t* pool, int64_t page_stride, const int32_t* pages, const int32_t* dst, int64_t n, const uint8_t* kbase,
                       const uint8_t* vbase, int64_t H, int64_t D, int64_t A, const uint8_t* escape, int64_t nchunks,
                       const uint8_t* raw, int64_t nslabs, uint16_t* out, cudaStream_t cs);

/* The keys and values of tokens 0 .. n of a step (key and value [n, H, D] bf16: token t's row at key + t * key_stride elements, a head's D channels contiguous, strides even, at least H D, the pointers
 * 4-byte aligned) into the bf16 staging cache out [blocks, 16, H, 2 D]: token t to slot slots[t] (int64: block * 16 + offset), its keys at the head's first D channels and its values at the next D;
 * a negative slot, or one at or past blocks * 16, is skipped (a group of rows of a step takes its tokens and leaves the others'). */
int glyd_gpu_kv_work_put(const uint16_t* key, int64_t key_stride, const uint16_t* value, int64_t value_stride, const int64_t* slots,
                          int64_t n, int64_t H, int64_t D, uint16_t* out, int64_t blocks, cudaStream_t cs);

/* Attention for one new token a sequence over its pages (D 128, 256 or 512): B sequences, each its query q [H G, D] (sequence b's at q +
 * b * q_stride elements; G, 1 to 8, the query heads a kv head serves), its seq_lens[b] tokens (int32; 0: a zero output)
 * in the pages block_table[b] names (int32, rows bt_stride apart; the last page may be partly written, its
 * bytes past the length anything), softmax scale; out [H G, D] at out + b * out_stride. fp32 softmax and sums, the
 * probabilities rounded to bf16 for the values, as FlashAttention does. The work is split over each sequence's pages by a plan
 * made on the device from the sequence's own length alone (seq_lens is read there: nothing is read, copied or allocated on the host,
 * and the grid is B x H x splits, fixed): at least 16 pages a CTA (GLYD_KV_PPC_MIN, 4 to 64), at most `splits` CTAs, and on a GPU of 100 SMs or more, where B has so few
 * (sequence, kv head) pairs that 4 to an SM are fewer than the SMs, at least 16 CTAs for a sequence of 64 pages or more (4 pages a CTA at the least; GLYD_KV_FLOOR, 1 to 64,
 * sets that number for every B and GPU: 1 is none). A sequence's output's bits depend on its own length, on `splits`, on those two numbers (the second from B rounded up to a
 * power of two: a batch padded for a CUDA graph has the one) and not on the other sequences. `splits` (1 to 65535) is the same number for every call whose outputs are to be compared: glyd_gpu_kv_decode_shares gives one for a batch of B sequences (B rounded up to a power of two) of a model
 * whose longest has max_pages pages (the waves of the machine's CTAs it aims for: GLYD_KV_WAVES, 6; GLYD_KV_PIPE=1: loading each
 * warp's next page ahead, which the plan's bits do not depend on). workspace: glyd_gpu_kv_decode_workspace's bytes for B, H, G and splits, in device memory, zero-filled when it is made
 * (a call leaves it so; B x H at most 262,144; GLYD_KV_FUSE=0: the sums merged by a kernel of their own). */
int glyd_gpu_kv_decode_shares(int64_t B, int64_t H, int64_t max_pages, int64_t* splits);
int glyd_gpu_kv_decode_workspace(int64_t B, int64_t H, int64_t G, int64_t D, int64_t splits, size_t* bytes);
/* The plan's CTAs a sequence has at least, 1 to 64 (0: the GPU's own): what GLYD_KV_FLOOR sets in a development build. */
int glyd_gpu_kv_floor_set(int floor);
int glyd_gpu_kv_decode(const uint16_t* q, int64_t q_stride, const uint8_t* pool, int64_t page_stride, const uint8_t* kbase,
                       const uint8_t* vbase, const int32_t* block_table, int64_t bt_stride, const int32_t* seq_lens, int64_t B,
                       int64_t splits, int64_t H, int64_t G, int64_t D, int64_t A, const uint8_t* escape,
                       int64_t nchunks, const uint8_t* raw, int64_t nslabs, double scale, uint16_t* out, int64_t out_stride,
                       void* workspace, size_t workspace_bytes, cudaStream_t cs);
/* glyd_gpu_kv_decode over the tokens from first[b] (int32: a sliding window's start, 0 the whole sequence) to seq_lens[b] alone: the pages before
 * the one that holds token first[b] are not read, and that page's tokens before it count for nothing (their scores are -inf); a first at or past
 * the length gives a zero output. The plan is from the tokens attended to, so a sequence's output's bits depend on its first and its length (with
 * `splits` and the floor, as above). round_scores 1: each score is the bf16 of its dot product before the scale, as a bf16 matmul gives it (transformers'
 * attention, Gemma 4's); 0: fp32. first null and round 0: glyd_gpu_kv_decode. */
int glyd_gpu_kv_decode_from(const uint16_t* q, int64_t q_stride, const uint8_t* pool, int64_t page_stride, const uint8_t* kbase,
                            const uint8_t* vbase, const int32_t* block_table, int64_t bt_stride, const int32_t* seq_lens,
                            const int32_t* first, int64_t round_scores, int64_t B, int64_t splits, int64_t H, int64_t G, int64_t D, int64_t A,
                            const uint8_t* escape, int64_t nchunks, const uint8_t* raw, int64_t nslabs, double scale, uint16_t* out,
                            int64_t out_stride, void* workspace, size_t workspace_bytes, cudaStream_t cs);
/* The attention of a prompt chunk: n rows (0 to 2^31 - 1 - start) of one sequence at positions start .. start + n - 1, each causal over the keys from
 * its window's first (position - window + 1; window 0: none, the sequence's first) to itself. The keys before `start` are read from the sequence's
 * pages (block_table: int32, page j holding its tokens 16 j .. 16 j + 15, the pages from the one of the first key on read; D 128, 256 or 512, not
 * GLYD_GPU_KV_ONE_STREAM), never unpacked; the chunk's own from its bf16 rows key and value [n][H][D] (row t at key + t * key_stride elements; strides multiples of 8,
 * at least H D; the pointers 16-byte aligned), which are what glyd_gpu_kv_write was given for them (the caller's staging rows: whether the call is
 * before or after that write does not matter). key null: the chunk's keys and values are read from the pages too (written before the call;
 * value is then not read). q and out [n][H G][D] bf16 (row t at q + t * q_stride, out + t * out_stride elements; strides even, at least H G D, the
 * pointers 4-byte aligned; query head j of kv head h is h G + j; G 1 to 65535 / H: more query heads than the decode takes need no parts here). The
 * query is used as it is (bf16), never rounded further. A score is the dot product of a query and a key in fp32 (tensor cores), rounded to bf16 where
 * round_scores is 1 (before the scale, as a bf16 matmul gives it: transformers' attention, Gemma 4's; 0: fp32, the Qwen3.5 layout's), times scale,
 * then, where softcap > 0, softcap tanh(score / softcap) (transformers' attention soft-capping); sinks: null, or float [H G], each query head's logit
 * joining the softmax's sum and taking no value (gpt-oss's). fp32 softmax and sums, the probabilities rounded to bf16 for the values, the output
 * the bf16 of the sum over the sum of the probabilities (as glyd_gpu_kv_decode and FlashAttention do). One launch, no workspace, nothing allocated
 * or read on the host: a CUDA graph can hold it (the arguments are fixed in it, start too). A void page (glyd_gpu_kv_write's status) reads as
 * what it holds: the caller fails what was computed from it, as for the decode. cudaErrorNotSupported where the GPU has not the shared memory a
 * CTA takes (82 KB for D = 512, 50 KB for D = 256, 17 KB for 128). The executor's prompt step: kv_write of the chunk's rows from its staging rows,
 * then this call with those rows as key and value, in place of unpacking the pages to a bf16 ring for a family's kernel. */
int glyd_gpu_kv_prefill(const uint16_t* q, int64_t q_stride, const uint16_t* key, int64_t key_stride, const uint16_t* value,
                        int64_t value_stride, int64_t n, int64_t start, int64_t window, const uint8_t* pool, int64_t page_stride,
                        const uint8_t* kbase, const uint8_t* vbase, const int32_t* block_table, int64_t H, int64_t G, int64_t D,
                        int64_t A, const uint8_t* escape, int64_t nchunks, const uint8_t* raw, int64_t nslabs, double scale,
                        int64_t round_scores, double softcap, const float* sinks, uint16_t* out, int64_t out_stride, cudaStream_t cs);
/* glyd_gpu_kv_decode_from's attention with the keys and values in a bf16 cache in place of the pages: kc and vc [places][H][D] (4-byte aligned), token p of
 * sequence b at block_table[b bt_stride + p / 16] * 16 + p % 16, or p modulo cap where block_table is null. D 128, 256 or 512. The same kernel, plan and merge
 * (with the same splits, ppc and floor): where the cache holds the values the pages hold, the outputs are glyd_gpu_kv_decode_from's bit for bit. */
int glyd_gpu_kv_decode_bf16(const uint16_t* q, int64_t q_stride, const uint16_t* kc, const uint16_t* vc, const int32_t* block_table, int64_t bt_stride, int64_t cap,
                         const int32_t* seq_lens, const int32_t* first, int64_t round_scores, int64_t B, int64_t splits, int64_t H, int64_t G, int64_t D, double scale,
                         uint16_t* out, int64_t out_stride, void* workspace, size_t workspace_bytes, cudaStream_t cs);
/* glyd_gpu_kv_prefill's attention with the keys and values in a bf16 cache in place of the pages: kc and vc [places][H][D] (16-byte aligned), position p's
 * place block_table[p / 16] * 16 + p % 16 (int32, the sequence's row of the table), or p modulo cap where block_table is null; the chunk's own keys and values
 * written there before the call. D 128, 256 or 512. The same kernel, its tiles and its sums in the same order: where the cache holds the values the pages hold
 * (the pages are lossless), the outputs are glyd_gpu_kv_prefill's bit for bit. One launch, no workspace: a CUDA graph can hold it. */
int glyd_gpu_kv_prefill_bf16(const uint16_t* q, int64_t q_stride, const uint16_t* kc, const uint16_t* vc, const int32_t* block_table, int64_t cap, int64_t n, int64_t start,
                          int64_t window, int64_t H, int64_t G, int64_t D, double scale, int64_t round_scores, double softcap, const float* sinks, uint16_t* out,
                          int64_t out_stride, cudaStream_t cs);

/* ------------------------------------------------------------------------
 * The lossless FP8 weights: W [O, K] of e4m3 bytes (any byte: NaNs, both zeros) held in about 6.7 bits a value, as
 *   core   uint8 [patches x 6144]
 *   heads  uint32 [patches x 8]
 *   var    uint32 [var_words]
 * for patches = ceil(O / 64) ceil(K / 128), numbered in rows of patches. glyd_gpu/fp8ref.py (numpy) makes the same bytes and its docstring
 * is the layout; every function here reads or writes them as that says. W's rows are ld bytes apart (ld >= K); core and heads are
 * 16-byte aligned (heads: also written by glyd_gpu_f8_size).
 * ---------------------------------------------------------------------- */

/* The pack's size: glyd_gpu_f8_size_workspace's bytes of workspace (8-byte aligned, device memory), then glyd_gpu_f8_size, which reads W,
 * writes heads [patches x 8] (the words of each patch, its place in var in the first) and waits for the stream: *var_words, what var
 * holds (its eight guard words included), is the size to allocate. */
int glyd_gpu_f8_size_workspace(int64_t O, int64_t K, size_t* bytes);
int glyd_gpu_f8_size(const uint8_t* w, int64_t ld, int64_t O, int64_t K, uint32_t* heads, void* workspace, size_t workspace_bytes,
                     int64_t* var_words, cudaStream_t cs);
/* Then the pack: W's patches into core [patches x 6144] and var [var_words], from the heads that glyd_gpu_f8_size made for the same W
 * (its bytes are the same as fp8ref.pack's, whatever the shape). */
int glyd_gpu_f8_pack(const uint8_t* w, int64_t ld, int64_t O, int64_t K, const uint32_t* heads, uint8_t* core, uint32_t* var,
                     int64_t var_words, cudaStream_t cs);
/* W back, bit for bit, into out [O, K] (rows ld bytes apart). One launch: nothing is allocated or waited for, and a CUDA graph can hold it. */
int glyd_gpu_f8_unpack(const uint8_t* core, const uint32_t* heads, const uint32_t* var, int64_t O, int64_t K, uint8_t* out, int64_t ld,
                       cudaStream_t cs);
/* The matrices ids[0 .. n) of a stack of E matrices [O, K] (O a multiple of 64: the pack of the one matrix [E O, K]) into out [n, O, K]
 * (rows ld bytes apart, a matrix O ld bytes after the one before). ids int32 in device memory; an id outside [0, E) is skipped, its
 * place in out left as it is. One launch for any ids, so a CUDA graph can hold it (n at most 65535). */
int glyd_gpu_f8_unpack_experts(const uint8_t* core, const uint32_t* heads, const uint32_t* var, int64_t E, int64_t O, int64_t K,
                               const int32_t* ids, int64_t n, uint8_t* out, int64_t ld, cudaStream_t cs);
/* Y [M, N] = X [M, K] W^T for M of 1 to 64 (a decode step's tokens), W [N, K] held as above and never written out, with its scales (kind 0: fp32 a
 * 128 x 128 block, ceil(N / 128) rows lds floats apart; 1: one fp32; 2: e8m0 a 1 x 32 block, N rows lds bytes apart; 3: e8m0 a 32 x 32 block,
 * ceil(N / 32) rows lds bytes apart; 4: fp32 a row, N of them). X and Y bf16 (as uint16_t), rows ldx and ldy elements apart; X multiplied as it is
 * (bf16), the sums in fp32. glyd_gpu_f8_linear_workspace
 * gives the bytes of workspace it needs for W and M on the current GPU; done: ceil(N / 128) counters (int), zero before the first product (each product
 * leaves them zero). Nothing allocated or waited for: a CUDA graph can hold it. */
int glyd_gpu_f8_linear_workspace(int64_t N, int64_t K, int64_t M, size_t* bytes);
int glyd_gpu_f8_linear(const uint8_t* core, const uint32_t* heads, const uint32_t* var, int64_t N, int64_t K, int64_t kind, const void* scale,
                       int64_t lds, const uint16_t* x, int64_t ldx, int64_t M, uint16_t* y, int64_t ldy, void* workspace, size_t workspace_bytes,
                       int* done, cudaStream_t cs);
/* The experts' product: E matrices [N, K] (N a multiple of 64) packed as the one matrix [E N, K], their block scales (fp32 a 128 x 128 block, [E][ceil(N / 128)]
 * rows lds floats apart), and a plan of glyd_gpu_moe_way's (P pairs, k a token): each pair's row of Y [P, N] (the plan's order, rows ldy apart) = its row of X times
 * its expert's matrix^T, X's row its token's (gather: X [tokens, K]) or its own in the plan's order (X [P, K]), rows ldx apart. M: the most pairs an expert has (1 to
 * 64), slots: the most experts the plan hits (at most 65535). glyd_gpu_f8_moe_workspace gives the workspace; done: slots ceil(N / 128) counters, zero at first and
 * left zero. One launch, read on the device: a CUDA graph can hold it. */
int glyd_gpu_f8_moe_workspace(int64_t N, int64_t K, int64_t M, int64_t slots, size_t* bytes);
int glyd_gpu_f8_moe(const uint8_t* core, const uint32_t* heads, const uint32_t* var, int64_t E, int64_t N, int64_t K, const float* scale, int64_t lds,
                    const uint16_t* x, int64_t ldx, const int32_t* plan, int64_t k, int64_t gather, int64_t M, int64_t slots, uint16_t* y, int64_t ldy,
                    void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs);
/* Layout H (glyd_gpu.cu's f8c, for HBM GPUs: about 7.1 bits a value, a third of the decode's instructions): the calls above, a patch's core 7168 bytes
 * (core [patches x 7168]; heads and var as above, from glyd_gpu_f8h_size). The products: block scales alone (kind 0), the workspaces of
 * glyd_gpu_f8_linear_workspace and glyd_gpu_f8_moe_workspace. */
/* Layout H's packs plain (1: every value its e4m3 byte) or coded (0), or the GPU's own (-1: plain on an HBM GPU), for the process: set before any pack. */
int glyd_gpu_f8h_plain(int plain);
int glyd_gpu_f8h_size(const uint8_t* w, int64_t ld, int64_t O, int64_t K, uint32_t* heads, void* workspace, size_t workspace_bytes,
                      int64_t* var_words, cudaStream_t cs);
int glyd_gpu_f8h_pack(const uint8_t* w, int64_t ld, int64_t O, int64_t K, const uint32_t* heads, uint8_t* core, uint32_t* var,
                      int64_t var_words, cudaStream_t cs);
int glyd_gpu_f8h_unpack(const uint8_t* core, const uint32_t* heads, const uint32_t* var, int64_t O, int64_t K, uint8_t* out, int64_t ld,
                        cudaStream_t cs);
int glyd_gpu_f8h_linear(const uint8_t* core, const uint32_t* heads, const uint32_t* var, int64_t N, int64_t K, int64_t kind, const void* scale,
                        int64_t lds, const uint16_t* x, int64_t ldx, int64_t M, uint16_t* y, int64_t ldy, void* workspace, size_t workspace_bytes,
                        int* done, cudaStream_t cs);
int glyd_gpu_f8h_moe(const uint8_t* core, const uint32_t* heads, const uint32_t* var, int64_t E, int64_t N, int64_t K, const float* scale, int64_t lds,
                     const uint16_t* x, int64_t ldx, const int32_t* plan, int64_t k, int64_t gather, int64_t M, int64_t slots, uint16_t* y, int64_t ldy,
                     void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs);

/* ------------------------------------------------------------------------
 * The fast format (embeddings): W [O, K] (K a multiple of 128) as
 *   sm          uint8 [O K]
 *   planes      uint32 [O K / 32 x 3]
 *   exc         uint8
 *   exc_base    int32 [O x ceil(K / 1024)]
 *   top         uint64, the matrix's own
 * ---------------------------------------------------------------------- */

/* y [O] = W x (+ bias; NULL: none) for one token. */
int glyd_gpu_fast_gemv(const uint8_t* sm, const uint32_t* planes, const uint8_t* exc, const int32_t* exc_base,
                       uint64_t top, int64_t O, int64_t K, const uint16_t* x, const uint16_t* bias, uint16_t* y,
                       cudaStream_t cs);
/* Rows [row0, row0 + rows), or (n_ids > 0) the n_ids rows row_ids (int64),
 * into out [rows, K]: an embedding's lookup. */
int glyd_gpu_fast_decode(const uint8_t* sm, const uint32_t* planes, const uint8_t* exc, const int32_t* exc_base,
                         uint64_t top, int64_t row0, int64_t rows, const int64_t* row_ids, int64_t n_ids, int64_t K,
                         uint16_t* out, cudaStream_t cs);
/* Y [M, O] = X W^T (+ bias) for X [M, K] (K a multiple of 64, O of 16). */
int glyd_gpu_fast_gemm_workspace(int64_t O, int64_t K, int64_t M, size_t* bytes);
int glyd_gpu_fast_gemm(const uint8_t* sm, const uint32_t* planes, const uint8_t* exc, const int32_t* exc_base,
                       uint64_t top, int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias,
                       uint16_t* y, void* workspace, size_t workspace_bytes, cudaStream_t cs);
/* The same for M = 2, 4, 8 or 16 tokens (K a multiple of 512). */
int glyd_gpu_fast_bgemv_workspace(int64_t O, int64_t K, int64_t M, size_t* bytes);
int glyd_gpu_fast_bgemv(const uint8_t* sm, const uint32_t* planes, const uint8_t* exc, const int32_t* exc_base,
                        uint64_t top, int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias,
                        uint16_t* y, void* workspace, size_t workspace_bytes, cudaStream_t cs);

/* ------------------------------------------------------------------------
 * The dense format: n weights in tiles of tw (a matrix's: whole rows, or
 * pieces of long ones), V (4 or 16) weights a lane a step:
 *   sm          uint8 [n]
 *   stream      uint32 [stream_words]
 *   offs        uint32 [tiles x 32]
 *   tables      uint32 [64]
 *   tile_words  the words staged for a tile (0: none)
 * ---------------------------------------------------------------------- */

/* The packer's first pass: bits [tiles x 32], from w and len [256], the code
 * tables the packer built. */
int glyd_gpu_pass_a(const uint16_t* w, int64_t n, const uint8_t* len, int64_t tw, int64_t V, uint32_t* bits,
                       cudaStream_t cs);
/* Its second: the streams into out (zero before; the lengths' total / 32 + 4
 * words) from offs (the lengths' sums before each) and code [256]. */
int glyd_gpu_pass_b(const uint16_t* w, int64_t n, const uint8_t* len, const uint32_t* code, const uint32_t* offs,
                         uint32_t* out, int64_t tw, int64_t V, cudaStream_t cs);
/* Every tile into out [n], or (n_ids > 0) tiles tile_ids (int64) into out
 * [n_ids x tw]. */
int glyd_gpu_decode(const uint8_t* sm, const uint32_t* stream, int64_t stream_words, const uint32_t* offs,
                    const uint32_t* tables, int64_t n, int64_t tw, int64_t V, int64_t tile_words,
                    const int64_t* tile_ids, int64_t n_ids, uint16_t* out, cudaStream_t cs);
/* y [O] = W x (+ bias; NULL: none) for W [O, K] (K and tw multiples of 32 V);
 * where tiles split rows (tw % K), sum (float) and count (int32) [O], zero
 * before and left zero. */
int glyd_gpu_gemv(const uint8_t* sm, const uint32_t* stream, int64_t stream_words, const uint32_t* offs,
                  const uint32_t* tables, int64_t O, int64_t K, int64_t tw, int64_t V, int64_t tile_words,
                  const uint16_t* x, const uint16_t* bias, uint16_t* y, float* sum, int* count, cudaStream_t cs);

#ifdef __cplusplus
}
#endif

#endif /* GLYD_GPU_H */
