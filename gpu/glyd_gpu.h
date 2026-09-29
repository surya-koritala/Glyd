/*
 * Glyd GPU: the C API of gpu/glyd_gpu.cu's kernels, which gpu/build_lib.sh
 * builds into libglyd_gpu_cuda12.so and libglyd_gpu_cuda13.so (the CUDA
 * runtime linked in: they need only the driver; code for Ampere and later).
 * A bf16 matrix held compressed in GPU memory and decoded on the GPU bit for
 * bit, whole or inside its products. The glyd Python package calls these
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
 * least 1024 (0.24.0); 4 from the routes, glyd_gpu_*_linear, a GPU's class
 * in its code, and the 12-bit layout in split byte (its data, exc and sym[4]
 * as below; the 12-bit layout before it, never released, is refused); 5 from
 * the route SPLIT (a prompt decoded ahead on SMs set apart, cuBLAS on the
 * rest: glyd_gpu_mma12_ring_*), a GPU's PCIe class and GLYD_GPU_NO_SPLIT in
 * its code. */
#define GLYD_GPU_API_VERSION 5

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
 * The mma layouts: W [O, K] (O a multiple of 64, K of 16) as O K / 1024 warp
 * steps of 64 rows by 16 columns, row block by row block, in the order the
 * tensor cores take their operand: lane l = 4g + t of step (rb, ks) holds
 * weights i = 4n + j (n 0-7, j 0-3), W[64 rb + 8n + g][16 ks + 8 (j >> 1) +
 * 2t + (j & 1)]; a byte a weight as it is, the rest coded. glyd.gpu's
 * pack_mma and pack_mma12 make them, and glyd.save_pretrained saves them (a
 * Linear's NAME.glyd_data, NAME.glyd_blocks, NAME.glyd_block_base and its
 * tiers in glyd.json; 12-bit, glyd-v3: NAME.glyd_data, NAME.glyd_exc,
 * NAME.glyd_exc_base and its hb in glyd.json).
 *
 * Tiered (about 10.8 bits a weight): the byte a weight's sign and mantissa,
 * its exponent in 2-bit digits over three tiers of the matrix's commonest (3
 * a tier, digit 3 on to the next tier; past the third, the exponent's byte).
 *   data        uint8 [steps][1280]: a step's tier-1 digits and its 1024 bytes
 *   blocks      uint8: a step's escapes (its tier-2 and tier-3 digits and
 *               exponent bytes) from block_base[step] to block_base[step + 1];
 *               128 bytes before the first, 256 after the last
 *   block_base  int32 [steps + 1]
 *   tiers[3]    host words: tier k's three exponents in bytes 0-2 of word k
 * 12-bit, split byte (about 12 bits a weight, a lighter decode): the byte a
 * weight's bf16 low byte (its exponent's lowest bit and its mantissa); its
 * high byte (the sign, the exponent >> 1) a 4-bit code, the sign and an
 * offset 0-7 from the matrix's base hb (0-120), hb + offset the exponent >> 1;
 * any other weight an exception, coded with offset 0.
 *   data        uint8 [steps][1536]: a step's codes, 16 bytes a lane (lane l's
 *               at 16 l, words q 0-3 holding weights 8q to 8q + 7: weight
 *               8q + j's sign in bit 8j + 7 of word q, its offset in bits 8j
 *               to 8j + 2; weight 8q + 4 + j's sign in bit 8j + 3, its offset
 *               in bits 8 ((j + 3) % 4) + 4 to + 6), then its 1024 low bytes
 *               (weight i of lane l at 512 + 16 l + i for i < 16, at 1024 +
 *               16 l + i - 16 for the rest)
 *   exc         int32: the exceptions, a weight's place in its step, 32 l + i
 *               (bits 0-9), and the byte XORed into its high byte, hb ^ (the
 *               exponent >> 1) (bits 16-23), a step's in that order from
 *               exc_base[step] to exc_base[step + 1]; zeros after, to a
 *               multiple of 4 (1 at least)
 *   exc_base    int32 [steps + 1]
 *   sym[4]      host words: hb in each byte of sym[0], sym[1-3] zero (else
 *               cudaErrorInvalidValue)
 * glyd_gpu.cu has both to the bit.
 * ---------------------------------------------------------------------- */

/* Y [M, O] = X W^T (+ bias [O]; NULL: none) for 0 to 64 tokens (X [M, K], Y
 * row-major): a generation step's product, the weights decoded in registers
 * into the tensor cores' operands. done: O / 64 counters. */
int glyd_gpu_mma_gemm_workspace(int64_t O, int64_t K, int64_t M, size_t* bytes);
int glyd_gpu_mma_gemm(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base, const uint32_t tiers[3],
                      int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y,
                      void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs);
int glyd_gpu_mma12_gemm_workspace(int64_t O, int64_t K, int64_t M, size_t* bytes);
int glyd_gpu_mma12_gemm(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4],
                        int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y,
                        void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs);

/* The same for many tokens (a prompt; K a multiple of 64, x 16-byte aligned):
 * a tiled product, each weight decoded once for a block's tokens. variant 0:
 * chosen by M and the GPU; 1: blocks of 128 tokens by 128 rows; 2: 256 by 64;
 * 3 (12-bit): 256 by 128. done: (M + 127) / 128 x O / 64 counters. */
int glyd_gpu_mma_gemm_big_workspace(int64_t O, int64_t K, int64_t M, int64_t variant, size_t* bytes);
int glyd_gpu_mma_gemm_big(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base,
                          const uint32_t tiers[3], int64_t O, int64_t K, const uint16_t* x, int64_t M,
                          const uint16_t* bias, uint16_t* y, int64_t variant, void* workspace, size_t workspace_bytes,
                          int* done, cudaStream_t cs);
int glyd_gpu_mma12_gemm_big_workspace(int64_t O, int64_t K, int64_t M, int64_t variant, size_t* bytes);
int glyd_gpu_mma12_gemm_big(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4],
                            int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y,
                            int64_t variant, void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs);

/* Many tokens in the 12-bit layout (K a multiple of 64; x, data and exc
 * 16-byte aligned), the compressed weights copied into shared memory a stage
 * at a time: mid on Ampere and later (cudaErrorNotSupported before); wg on
 * Hopper (compute capability 9.0) alone, by TMA and wgmma. done: O / 64
 * counters (wg: at least 1024, or O / 64 where that is more). */
int glyd_gpu_mma12_gemm_mid_workspace(int64_t O, int64_t K, int64_t M, size_t* bytes);
int glyd_gpu_mma12_gemm_mid(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4],
                            int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y,
                            void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs);
int glyd_gpu_mma12_gemm_wg_workspace(int64_t O, int64_t K, int64_t M, size_t* bytes);
int glyd_gpu_mma12_gemm_wg(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4],
                           int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y,
                           void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs);

/* Rows [row0, row0 + rows) of W (multiples of 64) back to bf16, into out
 * [rows, K]. warps 0: a warp a step; else that many warps in all, each taking
 * every so many steps (a decode beside a product on another stream). Also,
 * 12-bit: the route SPLIT's decode (K a multiple of 64, out 16-byte
 * aligned), built for the few SMs a green context sets apart for it: a grid
 * for sms SMs (glyd_gpu_ring_* below run it on theirs). */
int glyd_gpu_mma_unpack(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base, const uint32_t tiers[3],
                        int64_t K, int64_t row0, int64_t rows, uint16_t* out, int64_t warps, cudaStream_t cs);
int glyd_gpu_mma12_unpack(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4],
                          int64_t K, int64_t row0, int64_t rows, uint16_t* out, int64_t warps, cudaStream_t cs);
int glyd_gpu_mma12_unpack_split(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4],
                                int64_t K, int64_t row0, int64_t rows, uint16_t* out, int64_t sms, cudaStream_t cs);

/* Stream cs held ns nanoseconds by one thread: a decode launched after it
 * there starts once a product launched with it on another stream has placed
 * its blocks. */
int glyd_gpu_hold(int64_t ns, cudaStream_t cs);

/* ------------------------------------------------------------------------
 * Routes: how glyd.gpu multiplies by a packed W [O, K] for M tokens on a
 * GPU, as measured there (gpu/README.md): the kernel it takes, or W decoded
 * for a bf16 GEMM of the caller's own (cuBLAS) where that is the faster.
 * glyd.gpu's Linears take their routes from here. A GPU is a code: its
 * compute capability, major * 10 + minor, plus its class by name where the
 * compute capability does not tell GPUs apart: GLYD_GPU_GEFORCE with
 * "GeForce" in its name, GLYD_GPU_A10 with "A10" in it as a word (between
 * characters that are not ASCII letters, digits or '_': an A10, not an A10G,
 * A100 or A40), GLYD_GPU_PCIE with "PCIe" in it in any case, else none.
 * 1089: an RTX 40; 2086: an A10; 86: an A10G, A40 or RTX A6000; 80: an A100
 * SXM4; 3080: an A100 PCIe; 90: an H100 SXM, H200 or GH200; 3090: an H100
 * PCIe. GLYD_GPU_NO_SPLIT added to a code: its routes where the route SPLIT
 * cannot run (no green contexts, a CUDA graph being captured).
 * ---------------------------------------------------------------------- */
#define GLYD_GPU_ROUTE_DECODE 0 /* W decoded (glyd_gpu_*_unpack), then the caller's GEMM */
#define GLYD_GPU_ROUTE_GEMM 1   /* glyd_gpu_mma_gemm, glyd_gpu_mma12_gemm */
#define GLYD_GPU_ROUTE_MID 2    /* glyd_gpu_mma12_gemm_mid */
#define GLYD_GPU_ROUTE_WG 3     /* glyd_gpu_mma12_gemm_wg */
#define GLYD_GPU_ROUTE_BIG 4    /* glyd_gpu_mma_gemm_big, glyd_gpu_mma12_gemm_big: variant 0 */
#define GLYD_GPU_ROUTE_AHEAD 5  /* DECODE, W decoded ahead beside the products before it (GeForce Ada's, an A10's prompts) */
#define GLYD_GPU_ROUTE_SPLIT 6  /* 12-bit: W decoded ahead on SMs set apart, the caller's cuBLAS on the rest (glyd_gpu_mma12_ring_linear) */
#define GLYD_GPU_GEFORCE 1000   /* a GPU's class: "GeForce" in its name */
#define GLYD_GPU_A10 2000       /* a GPU's class: "A10" in its name as a word */
#define GLYD_GPU_PCIE 3000      /* a GPU's class: "PCIe" in its name */
#define GLYD_GPU_NO_SPLIT 1048576 /* a flag in a GPU's code (1 << 20): the routes without SPLIT */

/* The current device's GPU as the routes take it: its code. */
int glyd_gpu_gpu(int* gpu);
/* The route of W [O, K] for M tokens on gpu, in the tiered layout or the
 * 12-bit one; last (NULL: not asked): the last token count from M on that
 * takes it (INT64_MAX: every one past M). GLYD_WG_MIN, GLYD_WG_MAX,
 * GLYD_MID_MIN and GLYD_DEC_MIN in the environment move its thresholds:
 * read once a process, at the first route (a later change has no effect),
 * each a whole number in base 10 (spaces around it, a sign), else taken as
 * unset (the glyd package refuses such a value at import); GLYD_DEC_MIN: a
 * 12-bit prompt decoded from that many tokens on any GPU, where unset or 0
 * an A100's from 769. */
int glyd_gpu_mma_route(int64_t gpu, int64_t O, int64_t K, int64_t M, int* route, int64_t* last);
int glyd_gpu_mma12_route(int64_t gpu, int64_t O, int64_t K, int64_t M, int* route, int64_t* last);
/* The route SPLIT's SMs for the decode, where the route is SPLIT (else 0): a
 * 12-bit prompt decoded ahead on SMs set apart (green contexts), cuBLAS on
 * the rest, where that beat the routes before it (gpu/README.md): an A100's
 * from 769 tokens (a matrix over 2 x 50 M weights to 4096), Hopper's from
 * 1024 (an H100 PCIe's at 1024 alone); the decode's SMs 4 to 20 by the GPU
 * and M. GLYD_SPLIT_MIN, GLYD_SPLIT_MAX (0 or unset: the GPU's; a negative
 * GLYD_SPLIT_MIN: never) and GLYD_SPLIT_SMS move them, on any GPU from
 * Ampere, read as the routes' others. K a multiple of 64. */
int glyd_gpu_mma12_split_sms(int64_t gpu, int64_t O, int64_t K, int64_t M, int64_t* sms);

/* Y [M, O] = X W^T (+ bias) by a route (negative: the current GPU's for M):
 * its kernel, its arguments as that kernel's; DECODE and AHEAD by the prompt
 * kernel (BIG, variant 0) on every GPU, Hopper's too (not measured there:
 * glyd.gpu's own prompts there are decoded for cuBLAS). No kernel here takes
 * DECODE or AHEAD where K is not a multiple of 64 (the prompt kernel's
 * blocks), and a matrix of such a K takes one of them for M past 64 on every
 * GPU and in either layout (in the 12-bit layout also from GLYD_DEC_MIN
 * tokens where that is set lower): there these functions and their
 * workspace queries return cudaErrorNotSupported, nothing launched, and the
 * caller decodes W (glyd_gpu_*_unpack, [O, K] bf16) for a GEMM of its own,
 * Y = X W^T + bias, as glyd.gpu does. done: (M + 127) / 128 x O / 64
 * counters, and at least 1024 (the WG route's, as glyd_gpu_mma12_gemm_wg's). */
int glyd_gpu_mma_linear_workspace(int64_t O, int64_t K, int64_t M, int64_t route, size_t* bytes);
int glyd_gpu_mma_linear(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base, const uint32_t tiers[3],
                        int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y,
                        int64_t route, void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs);
int glyd_gpu_mma12_linear_workspace(int64_t O, int64_t K, int64_t M, int64_t route, size_t* bytes);
int glyd_gpu_mma12_linear(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4],
                          int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y,
                          int64_t route, void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs);

/* ------------------------------------------------------------------------
 * The route SPLIT: a prompt's matrices decoded ahead, into a ring of slots in
 * the caller's device memory, on SMs set apart for the decode by the
 * driver's green contexts, while the caller's cuBLAS multiplies from the ring
 * on the rest (told its SMs), the two sides ordered by events (the host never
 * waits). A matrix is decoded in row chunks of at most a slot, a slot a chunk
 * as slots come free, the matrices in the order queued: queue a prompt's
 * matrices as its first product starts, then call each product; a product
 * whose matrix is not next in the queue drops it and decodes its own, chunk
 * by chunk, beside its products. Its products are cuBLAS's own on the
 * decoded bf16 (a row chunk a call): not bit for bit a whole-matrix product.
 * A ring serves one host thread and one device (the current one when made);
 * cudaErrorNotSupported where the split cannot run (a driver before CUDA
 * 12.4, one that refuses the split: MIG, MPS; stream cs being captured into
 * a CUDA graph): take the route the code with GLYD_GPU_NO_SPLIT gives.
 * ---------------------------------------------------------------------- */
typedef struct glyd_gpu_ring glyd_gpu_ring;

/* The caller's cuBLAS, which the library does not link: a handle and its
 * functions (the calls' own types, int for their enums and status; get_* and
 * set_workspace may be NULL). A call sets the handle's stream to the ring's
 * product stream, its workspace to workspace (where given: the ring's stream
 * alone uses it) and its SM count target to the products' SMs, and puts back
 * the stream and target it had where get_stream and get_sm_count_target are
 * given. A cuBLAS status s is returned as GLYD_GPU_BLAS_ERROR + s. */
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

/* A ring over buffer (bytes long, 16-byte aligned) in slots of slot_bytes (a
 * multiple of 256; 3 to 16 of them), on the current device. */
int glyd_gpu_ring_create(void* buffer, size_t bytes, size_t slot_bytes, glyd_gpu_ring** ring);
/* Its streams waited for, its green contexts let go. */
int glyd_gpu_ring_destroy(glyd_gpu_ring* ring);
/* The split for a decode of sms SMs (glyd_gpu_mma12_split_sms's): made where
 * it is not yet (a green context of the remaining SMs, a co-scheduled group,
 * for the products; the decode's of the SMs left) and kept; the SMs each side
 * has. */
int glyd_gpu_ring_split(glyd_gpu_ring* ring, int64_t sms, int64_t* decode_sms, int64_t* product_sms);
/* The queue dropped: stream cs waits for everything the ring has queued. */
int glyd_gpu_ring_reset(glyd_gpu_ring* ring, cudaStream_t cs);
/* W [O, K] queued after the matrices before it, decoded on the split of sms
 * SMs as slots come free (the queue's split: a queue's matrices share one). */
int glyd_gpu_mma12_ring_queue(glyd_gpu_ring* ring, int64_t sms, const uint8_t* data, const uint32_t* exc,
                              const int32_t* exc_base, const uint32_t sym[4], int64_t O, int64_t K);
/* Y [M, O] = X W^T (+ bias) by W's chunks from the ring (decoded ahead where
 * W is next in the queue, else queued alone now), each multiplied by cuBLAS
 * on the split's product stream (bf16, fp32 sums; a bias written into Y
 * first, then added by cuBLAS); X taken once the work queued on cs before is
 * done, and cs waits for Y. x and y 16-byte aligned, rows of K and O. */
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
 * Attention for one new token a sequence over a KV cache in the tiered layout
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
 * The fast format (embeddings): W [O, K] (K a multiple of 128), an exponent a
 * 3-bit code into the matrix's 7 commonest, code 7 an escape.
 *   sm          uint8 [O K]: each weight's sign (bit 7) and mantissa
 *   planes      uint32 [O K / 32 x 3]: code bit b of weight 32g + i at bit i
 *               of planes[3g + b]
 *   exc         uint8: the escapes' exponents, in weight order
 *   exc_base    int32 [O x ceil(K / 1024)]: the first escape of each 1024
 *               weights of a row
 *   top         the 7 exponents, code c in byte c
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
/* Y [M, O] = X W^T (+ bias) for X [M, K] (K a multiple of 64, O of 16), on
 * the tensor cores. */
int glyd_gpu_fast_gemm_workspace(int64_t O, int64_t K, int64_t M, size_t* bytes);
int glyd_gpu_fast_gemm(const uint8_t* sm, const uint32_t* planes, const uint8_t* exc, const int32_t* exc_base,
                       uint64_t top, int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias,
                       uint16_t* y, void* workspace, size_t workspace_bytes, cudaStream_t cs);
/* The same for M = 2, 4, 8 or 16 tokens (K a multiple of 512), on the CUDA
 * cores. */
int glyd_gpu_fast_bgemv_workspace(int64_t O, int64_t K, int64_t M, size_t* bytes);
int glyd_gpu_fast_bgemv(const uint8_t* sm, const uint32_t* planes, const uint8_t* exc, const int32_t* exc_base,
                        uint64_t top, int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias,
                        uint16_t* y, void* workspace, size_t workspace_bytes, cudaStream_t cs);

/* ------------------------------------------------------------------------
 * The dense format: n weights in tiles of tw (a matrix's: whole rows, or
 * pieces of long ones), a tile 32 lanes' bit streams of its exponents in a
 * prefix code, V (4 or 16) weights a lane a step.
 *   sm          uint8 [n]: each weight's sign (bit 7) and mantissa
 *   stream      uint32 [stream_words]: the streams, back to back
 *   offs        uint32 [tiles x 32]: where each lane's stream starts, in bits
 *   tables      uint32 [64]: the code's 32 classes, then its 32 ranks
 *   tile_words  the words a warp stages in shared memory (0: none)
 * ---------------------------------------------------------------------- */

/* The packer's first pass: bits [tiles x 32], each lane's stream's length,
 * for len [256], each exponent's code length. */
int glyd_gpu_lane_bits(const uint16_t* w, int64_t n, const uint8_t* len, int64_t tw, int64_t V, uint32_t* bits,
                       cudaStream_t cs);
/* Its second: the streams into out (zero before; the lengths' total / 32 + 4
 * words, as the decode reads ahead) from offs (the lengths' sums before
 * each), code [256] each exponent's code. */
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
