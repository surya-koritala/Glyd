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
 * License: BUSL-1.1 (gpu/LICENSE).
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
#include <cuda_runtime_api.h> /* cudaStream_t */

/* The C API's version, one more whenever a function's arguments, or what
 * they must hold, change: 2 from the prompt products' done counters,
 * glyd_gpu_hold and the decode's warps (0.21.0's library has no
 * glyd_gpu_api_version: 1); 3 from glyd_gpu_mma12_gemm_wg's counters, at
 * least 1024 (0.24.0); 4 from the routes, glyd_gpu_*_linear and a GPU's
 * class in its code (never released: builds of main alone); 5 from a change
 * of the mma12 layout (its data, exc and sym[4] as below): the mma12 packs of
 * 0.24 and before have other bytes and words, which the library refuses: pack
 * them again; 6 from the route SPLIT (glyd_gpu_ring_*, glyd_gpu_mma12_ring_*,
 * glyd_gpu_mma12_unpack_split, glyd_gpu_mma12_split_sms), a GPU's PCIe class
 * and GLYD_GPU_NO_SPLIT in its code (never released: builds of the route's
 * branch alone); 7 from GLYD_GPU_WITH_SPLIT in its place, the route opt-in,
 * and a GPU's PCIe class 5000 (0.26.0). */
#define GLYD_GPU_API_VERSION 7

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
 *   block_base  int32 [steps + 1]
 *   tiers[3]    host words, the matrix's own
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
int glyd_gpu_mma_gemm_big_workspace(int64_t O, int64_t K, int64_t M, int64_t variant, size_t* bytes);
int glyd_gpu_mma_gemm_big(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base,
                          const uint32_t tiers[3], int64_t O, int64_t K, const uint16_t* x, int64_t M,
                          const uint16_t* bias, uint16_t* y, int64_t variant, void* workspace, size_t workspace_bytes,
                          int* done, cudaStream_t cs);
int glyd_gpu_mma12_gemm_big_workspace(int64_t O, int64_t K, int64_t M, int64_t variant, size_t* bytes);
int glyd_gpu_mma12_gemm_big(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4],
                            int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y,
                            int64_t variant, void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs);

/* Many tokens in the mma12 layout (K a multiple of 64; x, data and exc
 * 16-byte aligned): mid on Ampere and later (cudaErrorNotSupported before);
 * wg on Hopper (compute capability 9.0) alone. done: O / 64 counters (wg: at
 * least 1024, or O / 64 where that is more). */
int glyd_gpu_mma12_gemm_mid_workspace(int64_t O, int64_t K, int64_t M, size_t* bytes);
int glyd_gpu_mma12_gemm_mid(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4],
                            int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y,
                            void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs);
int glyd_gpu_mma12_gemm_wg_workspace(int64_t O, int64_t K, int64_t M, size_t* bytes);
int glyd_gpu_mma12_gemm_wg(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4],
                           int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y,
                           void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs);

/* Rows [row0, row0 + rows) of W (multiples of 64) back to bf16, into out
 * [rows, K]. warps 0: the default launch; else the launch is limited to that
 * many warps in all, so it can run beside a product on another stream.
 * glyd_gpu_mma12_unpack_split: the same for the route SPLIT, sized by sms (the
 * split size of glyd_gpu_mma12_split_sms; K a multiple of 64, out 16-byte
 * aligned). */
int glyd_gpu_mma_unpack(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base, const uint32_t tiers[3],
                        int64_t K, int64_t row0, int64_t rows, uint16_t* out, int64_t warps, cudaStream_t cs);
int glyd_gpu_mma12_unpack(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4],
                          int64_t K, int64_t row0, int64_t rows, uint16_t* out, int64_t warps, cudaStream_t cs);
int glyd_gpu_mma12_unpack_split(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4],
                                int64_t K, int64_t row0, int64_t rows, uint16_t* out, int64_t sms, cudaStream_t cs);

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
 * a caller runs through glyd_gpu_mma12_ring_linear (glyd.gpu's Linears do,
 * where it can run); without the flag no route is SPLIT (v0.25.1's routes).
 * ---------------------------------------------------------------------- */
#define GLYD_GPU_ROUTE_DECODE 0 /* W unpacked (glyd_gpu_*_unpack), then the caller's GEMM */
#define GLYD_GPU_ROUTE_GEMM 1   /* glyd_gpu_mma_gemm, glyd_gpu_mma12_gemm */
#define GLYD_GPU_ROUTE_MID 2    /* glyd_gpu_mma12_gemm_mid */
#define GLYD_GPU_ROUTE_WG 3     /* glyd_gpu_mma12_gemm_wg */
#define GLYD_GPU_ROUTE_BIG 4    /* glyd_gpu_mma_gemm_big, glyd_gpu_mma12_gemm_big: variant 0 */
#define GLYD_GPU_ROUTE_AHEAD 5  /* DECODE for a caller; the faster route for a long prompt on GeForce Ada, an A10 and an L40S in glyd.gpu */
#define GLYD_GPU_ROUTE_SPLIT 6  /* mma12, opt-in: the faster route for a long prompt on an A100 SXM, a GH200 and an H100 SXM (glyd_gpu_mma12_ring_linear) */
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
int glyd_gpu_mma_route(int64_t gpu, int64_t O, int64_t K, int64_t M, int* route, int64_t* last);
int glyd_gpu_mma12_route(int64_t gpu, int64_t O, int64_t K, int64_t M, int* route, int64_t* last);
/* The split size of the route SPLIT for W [O, K] and M tokens on gpu (a code
 * with GLYD_GPU_WITH_SPLIT): pass it as sms to glyd_gpu_ring_split,
 * glyd_gpu_mma12_ring_queue and glyd_gpu_mma12_ring_linear. 0 where the route is
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
int glyd_gpu_mma12_split_sms(int64_t gpu, int64_t O, int64_t K, int64_t M, int64_t* sms);

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
 * (the WG route's, as glyd_gpu_mma12_gemm_wg's). */
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
 * go), then call each product with glyd_gpu_mma12_ring_linear; a product whose
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
int glyd_gpu_ring_create(void* buffer, size_t bytes, size_t slot_bytes, glyd_gpu_ring** ring);
/* Destroys the ring: its streams are waited for and let go with its events (on
 * its device, whichever is current); the first failure's status (then its buffer
 * may still be written: keep it). */
int glyd_gpu_ring_destroy(glyd_gpu_ring* ring);
/* Sets the ring up for a split of size sms (glyd_gpu_mma12_split_sms's), where it
 * is not yet, and keeps it; the sizes of its two sides. */
int glyd_gpu_ring_split(glyd_gpu_ring* ring, int64_t sms, int64_t* decode_sms, int64_t* product_sms);
/* The queue dropped: stream cs waits for everything the ring has queued. */
int glyd_gpu_ring_reset(glyd_gpu_ring* ring, cudaStream_t cs);
/* W [O, K] queued after the matrices before it (sms: the queue's split size; a
 * queue's matrices share one). */
int glyd_gpu_mma12_ring_queue(glyd_gpu_ring* ring, int64_t sms, const uint8_t* data, const uint32_t* exc,
                              const int32_t* exc_base, const uint32_t sym[4], int64_t O, int64_t K);
/* Y [M, O] = X W^T (+ bias) for a long prompt: W is the next matrix in the queue
 * (else it is queued now), and your cuBLAS runs the products (bf16, fp32 sums; a
 * bias written into Y first, then added by cuBLAS). X is taken once the work
 * queued on cs before is done, and cs waits for Y. x and y 16-byte aligned, rows
 * of K and O. */
int glyd_gpu_mma12_ring_linear(glyd_gpu_ring* ring, int64_t sms, const uint8_t* data, const uint32_t* exc,
                               const int32_t* exc_base, const uint32_t sym[4], int64_t O, int64_t K, const uint16_t* x,
                               int64_t M, const uint16_t* bias, uint16_t* y, const glyd_gpu_blas* blas, cudaStream_t cs);

/* ------------------------------------------------------------------------
 * Mixtures of experts: a layer's E experts' matrices [O, K] as one pack
 * [E O, K]. A token's k choices are its pairs (P = T k for T tokens; pair j:
 * token j / k, choice j mod k).
 * ---------------------------------------------------------------------- */

/* The plan (int32 [2 + 2E + P]): the pairs, whose experts are ids (int64
 * [P]), sorted by expert; E up to 12288. */
int glyd_gpu_moe_route(const int64_t* ids, int64_t P, int64_t E, int32_t* plan, cudaStream_t cs);

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
int glyd_gpu_lane_bits(const uint16_t* w, int64_t n, const uint8_t* len, int64_t tw, int64_t V, uint32_t* bits,
                       cudaStream_t cs);
/* Its second: the streams into out (zero before; the lengths' total / 32 + 4
 * words) from offs (the lengths' sums before each) and code [256]. */
int glyd_gpu_write_codes(const uint16_t* w, int64_t n, const uint8_t* len, const uint32_t* code, const uint32_t* offs,
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
