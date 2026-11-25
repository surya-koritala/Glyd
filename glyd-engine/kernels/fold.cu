// R17 (#238, docs/design/one-kernel-build.md): the GPU library's kernels in the engine's own fatbin, compiled from the library's own source
// (gpu/glyd_gpu_kernels.h, in a namespace of its own here), as extern "C" kernels the driver finds by name. Their launches are Rust
// (src/fold.rs), as the library's host code makes them, and `glyd-engine foldcheck` holds each to the library's bit for bit.
#include <type_traits>
#if !defined(__HIPCC__)
#include <cuda.h>  // (CUtensorMap: the Hopper kernels' tensor maps, a kernel parameter; hip.h has its own)
#include <cstddef>
// (the numbers src/device.rs passes the driver for Hopper's launches, and the layouts it builds, held to cuda.h's)
static_assert(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 == 9 && CU_TENSOR_MAP_INTERLEAVE_NONE == 0 && CU_TENSOR_MAP_SWIZZLE_128B == 3 && CU_TENSOR_MAP_L2_PROMOTION_L2_256B == 3 &&
                  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE == 0,
              "device.rs's tensor map numbers");
static_assert(sizeof(CUtensorMap) == 128 && alignof(CUtensorMap) == 64, "fold::TensorMap");
static_assert(CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION == 4 && sizeof(CUlaunchConfig) == 56 && sizeof(CUlaunchAttribute) == 72 && offsetof(CUlaunchConfig, hStream) == 32 &&
                  offsetof(CUlaunchConfig, numAttrs) == 48,
              "device.rs's LaunchConfig and LaunchAttribute");
#endif

namespace glib {
#include "../../gpu/glyd_gpu_kernels.h"
}

// glyd_gpu_mma_unpack and glyd_gpu_mma12_unpack as the engine calls them (no warps given): rows [row0, row0 + rows) of W [., K] back to bf16
// into out [rows, K], a warp a 64 x 16 step, blocks of 256 threads (mma_unpack_body). A layout's kernel each; its words as the library reads
// them (glyd_gpu.cu's *_words, fp4r_fmt and nib12) are read in Rust and passed here.
extern "C" __global__ void fold_unpack_tiered(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base, uint32_t s1, uint32_t s2, uint32_t s3, int64_t K,
                                              int64_t row0, int64_t rows, uint16_t* out) {
    glib::mma_unpack_body<glib::Tiered>(glib::Tiered{data, blocks, block_base, glib::Tiers{s1, s2, s3}}, K, row0, rows, out, nullptr);
}

extern "C" __global__ void fold_unpack_compact(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base, uint32_t s1, uint32_t s2, uint32_t s3, int64_t K,
                                               int64_t row0, int64_t rows, uint16_t* out) {
    glib::mma_unpack_body<glib::Compact>(glib::Compact{data, blocks, block_base, glib::Tiers{s1, s2, s3}}, K, row0, rows, out, nullptr);
}

extern "C" __global__ void fold_unpack_int6(const uint8_t* data, const uint8_t* scales, int64_t K, int64_t row0, int64_t rows, uint16_t* out) {
    glib::mma_unpack_body<glib::Int6>(glib::Int6{data, scales}, K, row0, rows, out, nullptr);
}

extern "C" __global__ void fold_unpack_int5(const uint8_t* data, const uint8_t* scales, int64_t K, int64_t row0, int64_t rows, uint16_t* out) {
    glib::mma_unpack_body<glib::Int5>(glib::Int5{data, scales}, K, row0, rows, out, nullptr);
}

extern "C" __global__ void fold_unpack_int4(const uint8_t* data, const uint8_t* scales, int64_t K, int64_t row0, int64_t rows, uint16_t* out) {
    glib::mma_unpack_body<glib::Int4>(glib::Int4{data, scales}, K, row0, rows, out, nullptr);
}

extern "C" __global__ void fold_unpack_int4h(const uint8_t* data, const uint8_t* scales, uint32_t base, int64_t K, int64_t row0, int64_t rows, uint16_t* out) {
    glib::mma_unpack_body<glib::Int4H>(glib::Int4H{{data, scales}, base}, K, row0, rows, out, nullptr);
}

extern "C" __global__ void fold_unpack_fp4(const uint8_t* data, const uint8_t* scales, int64_t K, int64_t row0, int64_t rows, uint16_t* out) {
    glib::mma_unpack_body<glib::Fp4>(glib::Fp4{data, scales}, K, row0, rows, out, nullptr);
}

extern "C" __global__ void fold_unpack_fp4r(const uint8_t* data, const uint8_t* scales, uint32_t ks, uint32_t mg, uint32_t sh, int64_t K, int64_t row0, int64_t rows,
                                            uint16_t* out) {
    glib::mma_unpack_body<glib::Fp4R>(glib::Fp4R{data, scales, ks, mg, sh}, K, row0, rows, out, nullptr);
}

extern "C" __global__ void fold_unpack_int3(const uint8_t* data, const uint8_t* scales, int64_t K, int64_t row0, int64_t rows, uint16_t* out) {
    glib::mma_unpack_body<glib::Int3>(glib::Int3{data, scales}, K, row0, rows, out, nullptr);
}

extern "C" __global__ void fold_unpack_twelve(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, uint32_t hb4, int xmax, int64_t K, int64_t row0,
                                              int64_t rows, uint16_t* out) {
    glib::mma_unpack_body<glib::Nib>(glib::Nib{data, exc, exc_base, hb4, xmax}, K, row0, rows, out, nullptr);
}

// glyd_gpu_mma_rows and glyd_gpu_mma12_rows: rows of W [rows, K] by id (ids int64 [n] on the device; one outside W's rows: a row of zeros) into out [n, K], a
// warp a (row, step), blocks of 256 threads (mma_rows_body). A layout's kernel each, its words read as the unpackers' are.
extern "C" __global__ void fold_rows_tiered(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base, uint32_t s1, uint32_t s2, uint32_t s3, int64_t K, int64_t rows, const int64_t* ids,
                                            int64_t n, uint16_t* out) {
    glib::mma_rows_body<glib::Tiered>(glib::Tiered{data, blocks, block_base, glib::Tiers{s1, s2, s3}}, K, rows, ids, n, out);
}

extern "C" __global__ void fold_rows_compact(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base, uint32_t s1, uint32_t s2, uint32_t s3, int64_t K, int64_t rows, const int64_t* ids,
                                             int64_t n, uint16_t* out) {
    glib::mma_rows_body<glib::Compact>(glib::Compact{data, blocks, block_base, glib::Tiers{s1, s2, s3}}, K, rows, ids, n, out);
}

extern "C" __global__ void fold_rows_int6(const uint8_t* data, const uint8_t* scales, int64_t K, int64_t rows, const int64_t* ids,
                                          int64_t n, uint16_t* out) {
    glib::mma_rows_body<glib::Int6>(glib::Int6{data, scales}, K, rows, ids, n, out);
}

extern "C" __global__ void fold_rows_int5(const uint8_t* data, const uint8_t* scales, int64_t K, int64_t rows, const int64_t* ids,
                                          int64_t n, uint16_t* out) {
    glib::mma_rows_body<glib::Int5>(glib::Int5{data, scales}, K, rows, ids, n, out);
}

extern "C" __global__ void fold_rows_int4(const uint8_t* data, const uint8_t* scales, int64_t K, int64_t rows, const int64_t* ids,
                                          int64_t n, uint16_t* out) {
    glib::mma_rows_body<glib::Int4>(glib::Int4{data, scales}, K, rows, ids, n, out);
}

extern "C" __global__ void fold_rows_int4h(const uint8_t* data, const uint8_t* scales, uint32_t base, int64_t K, int64_t rows, const int64_t* ids,
                                           int64_t n, uint16_t* out) {
    glib::mma_rows_body<glib::Int4H>(glib::Int4H{{data, scales}, base}, K, rows, ids, n, out);
}

extern "C" __global__ void fold_rows_fp4(const uint8_t* data, const uint8_t* scales, int64_t K, int64_t rows, const int64_t* ids,
                                         int64_t n, uint16_t* out) {
    glib::mma_rows_body<glib::Fp4>(glib::Fp4{data, scales}, K, rows, ids, n, out);
}

extern "C" __global__ void fold_rows_fp4r(const uint8_t* data, const uint8_t* scales, uint32_t ks, uint32_t mg, uint32_t sh, int64_t K, int64_t rows, const int64_t* ids,
                                          int64_t n, uint16_t* out) {
    glib::mma_rows_body<glib::Fp4R>(glib::Fp4R{data, scales, ks, mg, sh}, K, rows, ids, n, out);
}

extern "C" __global__ void fold_rows_int3(const uint8_t* data, const uint8_t* scales, int64_t K, int64_t rows, const int64_t* ids,
                                          int64_t n, uint16_t* out) {
    glib::mma_rows_body<glib::Int3>(glib::Int3{data, scales}, K, rows, ids, n, out);
}

extern "C" __global__ void fold_rows_twelve(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, uint32_t hb4, int xmax, int64_t K, int64_t rows, const int64_t* ids,
                                            int64_t n, uint16_t* out) {
    glib::mma_rows_body<glib::Nib>(glib::Nib{data, exc, exc_base, hb4, xmax}, K, rows, ids, n, out);
}

// glyd_gpu_moe_way: the plan of a mixture of experts' pairs sorted by expert (moe_route_body: one warp; moe_route_block_body: one block of up to 32
// warps for a prompt's thousands of pairs, the same plan bit for bit). Which one, and its shared memory, as the library chooses (src/fold.rs).
extern "C" __global__ void fold_moe_route(const int64_t* ids, int64_t P, int E, int* plan) {
    glib::moe_route_body(ids, P, E, plan);
}

extern "C" __global__ void __launch_bounds__(1024) fold_moe_route_block(const int64_t* ids, int64_t P, int E, int* plan) {
    glib::moe_route_block_body(ids, P, E, plan);
}

// glyd_gpu_mma_linear's and _mma12_linear's route GEMM (1 to 64 tokens, mma_gemm_run): mma_gemm_body of a layout for MT tiles of 16 tokens (1, 2 or 4;
// MT 1 with GEMM1_BLOCKS blocks an SM, as the library's kernel), and the tiered layout's step loop for 1 to 16 tokens (step_body; _w: W as the A operand,
// to 8 tokens). The default plans alone (no GLYD_GPU_PATCH_B or GLYD_GPU_PFD variant). Their grids and workspace as the library's (src/fold.rs).
#define FOLD_GEMM_T(NAME, FMT, MT, MINB) \
    extern "C" __global__ void __launch_bounds__(256, MINB) fold_gemm_##NAME##_##MT(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base, uint32_t s1, \
        uint32_t s2, uint32_t s3, int64_t O, int64_t K, int64_t M, const __nv_bfloat16* X, const __nv_bfloat16* bias, __nv_bfloat16* Y, float* parts, int* done, int pf) { \
        glib::mma_gemm_body<glib::FMT, MT>(glib::FMT{data, blocks, block_base, glib::Tiers{s1, s2, s3}}, O, K, M, X, bias, Y, parts, done, pf); \
    }
#define FOLD_GEMM_B(NAME, FMT, MT, MINB) \
    extern "C" __global__ void __launch_bounds__(256, MINB) fold_gemm_##NAME##_##MT(const uint8_t* data, const uint8_t* scales, int64_t O, int64_t K, int64_t M, \
        const __nv_bfloat16* X, const __nv_bfloat16* bias, __nv_bfloat16* Y, float* parts, int* done, int pf) { \
        glib::mma_gemm_body<glib::FMT, MT>(glib::FMT{data, scales}, O, K, M, X, bias, Y, parts, done, pf); \
    }
#define FOLD_GEMM_4H(MT, MINB) \
    extern "C" __global__ void __launch_bounds__(256, MINB) fold_gemm_int4h_##MT(const uint8_t* data, const uint8_t* scales, uint32_t base, int64_t O, int64_t K, int64_t M, \
        const __nv_bfloat16* X, const __nv_bfloat16* bias, __nv_bfloat16* Y, float* parts, int* done, int pf) { \
        glib::mma_gemm_body<glib::Int4H, MT>(glib::Int4H{{data, scales}, base}, O, K, M, X, bias, Y, parts, done, pf); \
    }
#define FOLD_GEMM_4R(MT, MINB) \
    extern "C" __global__ void __launch_bounds__(256, MINB) fold_gemm_fp4r_##MT(const uint8_t* data, const uint8_t* scales, uint32_t ks, uint32_t mg, uint32_t sh, int64_t O, \
        int64_t K, int64_t M, const __nv_bfloat16* X, const __nv_bfloat16* bias, __nv_bfloat16* Y, float* parts, int* done, int pf) { \
        glib::mma_gemm_body<glib::Fp4R, MT>(glib::Fp4R{data, scales, ks, mg, sh}, O, K, M, X, bias, Y, parts, done, pf); \
    }
#define FOLD_GEMM_12(MT, MINB) \
    extern "C" __global__ void __launch_bounds__(256, MINB) fold_gemm_twelve_##MT(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, uint32_t hb4, int xmax, \
        int64_t O, int64_t K, int64_t M, const __nv_bfloat16* X, const __nv_bfloat16* bias, __nv_bfloat16* Y, float* parts, int* done, int pf) { \
        glib::mma_gemm_body<glib::Nib, MT>(glib::Nib{data, exc, exc_base, hb4, xmax}, O, K, M, X, bias, Y, parts, done, pf); \
    }
FOLD_GEMM_T(tiered, Tiered, 1, GEMM1_BLOCKS)
FOLD_GEMM_T(tiered, Tiered, 2, 1)
FOLD_GEMM_T(tiered, Tiered, 4, 1)
FOLD_GEMM_T(compact, Compact, 1, GEMM1_BLOCKS)
FOLD_GEMM_T(compact, Compact, 2, 1)
FOLD_GEMM_T(compact, Compact, 4, 1)
FOLD_GEMM_B(int6, Int6, 1, GEMM1_BLOCKS)
FOLD_GEMM_B(int6, Int6, 2, 1)
FOLD_GEMM_B(int6, Int6, 4, 1)
FOLD_GEMM_B(int5, Int5, 1, GEMM1_BLOCKS)
FOLD_GEMM_B(int5, Int5, 2, 1)
FOLD_GEMM_B(int5, Int5, 4, 1)
FOLD_GEMM_B(int4, Int4, 1, GEMM1_BLOCKS)
FOLD_GEMM_B(int4, Int4, 2, 1)
FOLD_GEMM_B(int4, Int4, 4, 1)
FOLD_GEMM_B(fp4, Fp4, 1, GEMM1_BLOCKS)
FOLD_GEMM_B(fp4, Fp4, 2, 1)
FOLD_GEMM_B(fp4, Fp4, 4, 1)
FOLD_GEMM_B(int3, Int3, 1, GEMM1_BLOCKS)
FOLD_GEMM_B(int3, Int3, 2, 1)
FOLD_GEMM_B(int3, Int3, 4, 1)
FOLD_GEMM_4H(1, GEMM1_BLOCKS)
FOLD_GEMM_4H(2, 1)
FOLD_GEMM_4H(4, 1)
FOLD_GEMM_4R(1, GEMM1_BLOCKS)
FOLD_GEMM_4R(2, 1)
FOLD_GEMM_4R(4, 1)
FOLD_GEMM_12(1, GEMM1_BLOCKS)
FOLD_GEMM_12(2, 1)
FOLD_GEMM_12(4, 1)

#define FOLD_STEP(NAME, SW) \
    extern "C" __global__ void __launch_bounds__(32 * glib::STEP_WARPS, glib::STEP_BLOCKS) fold_step_tiered_##NAME(const uint8_t* data, const uint8_t* blocks, \
        const int32_t* block_base, uint32_t s1, uint32_t s2, uint32_t s3, int64_t O, int64_t K, int64_t M, const __nv_bfloat16* X, const __nv_bfloat16* bias, \
        __nv_bfloat16* Y, float* parts, int* done) { \
        glib::step_body<SW>(glib::Tiered{data, blocks, block_base, glib::Tiers{s1, s2, s3}}, O, K, M, X, bias, Y, parts, done); \
    }
FOLD_STEP(w, true)
FOLD_STEP(x, false)

// glyd_gpu_kv_decode, _decode_from and _decode_bf16 (kv_attn_launch with the merge in the attention kernel, no page loaded ahead: the defaults) and the occupancy
// glyd_gpu_kv_decode_shares plans by (the kernel that loads a page ahead, heads of 128): kvc::attn_body, its AttnParams built in Rust (fold::KvAttnParams).
#define FOLD_KV_ATTN(NAME, PIPE, HV, BF16) \
    extern "C" __global__ void __launch_bounds__(glib::kvc::NW * 32, 1) fold_kv_attn_##NAME(const glib::kvc::AttnParams P) { \
        glib::kvc::attn_body<PIPE, true, HV, BF16>(P); \
    }
FOLD_KV_ATTN(1, false, 1, false)
FOLD_KV_ATTN(2, false, 2, false)
FOLD_KV_ATTN(4, false, 4, false)
FOLD_KV_ATTN(bf16_1, false, 1, true)
FOLD_KV_ATTN(bf16_2, false, 2, true)
FOLD_KV_ATTN(bf16_4, false, 4, true)
FOLD_KV_ATTN(pipe_1, true, 1, false)

// glyd_gpu_kv_prefill and _prefill_bf16: kvc::prefill_body for heads of 128 and 256 (4 groups of 16 rows a CTA) and 512 (2), as the library launches it.
#define FOLD_KV_PREFILL(HV, RG) \
    extern "C" __global__ void __launch_bounds__(32 * RG * HV, 1) fold_kv_prefill_##HV(const glib::kvc::PrefillParams P) { \
        glib::kvc::prefill_body<HV, RG>(P); \
    }
FOLD_KV_PREFILL(1, 4)
FOLD_KV_PREFILL(2, 4)
FOLD_KV_PREFILL(4, 2)

// glyd_gpu_kv_write (a CTA a token: its keys and values into its page, 12 bits a value), _kv_unpack (pages to bf16 keys and values, NHD) and _kv_gather
// (pages into the bf16 staging cache [blocks][16][H][2 D]): kvc's write_body, unpack_body and gather_body, launched as the library launches them.
extern "C" __global__ void __launch_bounds__(256) fold_kv_write(const __grid_constant__ glib::kvc::WriteParams P) {
    glib::kvc::write_body(P);
}

extern "C" __global__ void __launch_bounds__(128) fold_kv_unpack(const uint8_t* pool, int64_t page_stride, const int32_t* pages, int n, const uint8_t* kbase, const uint8_t* vbase,
                                                                  int H, int A, const uint32_t* ent, int nc, const glib::kvc::RawPool R, uint16_t* out_k, uint16_t* out_v, int one) {
    glib::kvc::unpack_body(pool, page_stride, pages, n, kbase, vbase, H, A, ent, nc, R, out_k, out_v, one);
}

#define FOLD_KV_GATHER(HV) \
    extern "C" __global__ void __launch_bounds__(128) fold_kv_gather_##HV(const uint8_t* pool, int64_t page_stride, const int32_t* pages, const int32_t* dst, int n, \
        const uint8_t* kbase, const uint8_t* vbase, int H, int A, const uint32_t* ent, int nc, const glib::kvc::RawPool R, uint16_t* out) { \
        glib::kvc::gather_body<HV>(pool, page_stride, pages, dst, n, kbase, vbase, H, A, ent, nc, R, out); \
    }
FOLD_KV_GATHER(1)
FOLD_KV_GATHER(2)
FOLD_KV_GATHER(4)

// glyd_gpu_kv_release (a warp a block: its page cleared, its chunks or slab back to their pools), _kv_save (a page into its image) and _kv_load (an image into a
// page, chunks or a slab lent anew), a CTA a page of 128 threads: kvc's release_body, save_body and load_body, launched as the library launches them.
extern "C" __global__ void __launch_bounds__(32) fold_kv_release(uint8_t* pool, int64_t page_stride, int64_t nblocks, const int32_t* blocks, int H, const glib::kvc::EscapePool esc,
                                                                  const glib::kvc::RawPool raw, uint32_t layer_tag, uint32_t* status, uint32_t* lost, int64_t nlost) {
    glib::kvc::release_body(pool, page_stride, nblocks, blocks, H, esc, raw, layer_tag, status, lost, nlost);
}

extern "C" __global__ void __launch_bounds__(128) fold_kv_save(const uint8_t* pool, int64_t page_stride, int64_t nblocks, const int32_t* pages, int64_t pb, int64_t ib, int HP,
                                                                const glib::kvc::EscapePool esc, const glib::kvc::RawPool raw, uint32_t layer_tag, uint8_t* img, uint32_t* moved) {
    glib::kvc::save_body(pool, page_stride, nblocks, pages, pb, ib, HP, esc, raw, layer_tag, img, moved);
}

extern "C" __global__ void __launch_bounds__(128) fold_kv_load(const uint8_t* img, const int32_t* dst, int64_t pb, int64_t ib, uint8_t* pool, int64_t page_stride, int64_t nblocks, int HP,
                                                                const glib::kvc::EscapePool esc, const glib::kvc::RawPool raw, uint32_t layer_tag, uint32_t* status, uint32_t ring) {
    glib::kvc::load_body(img, dst, pb, ib, pool, page_stride, nblocks, HP, esc, raw, layer_tag, status, ring);
}

// glyd_gpu_mma_moe, _mma12_moe and _mma_moe_placed (mma_moe_run) at a decode step's and a few rows' sizes: mma_moe_body for MT tiles of 16 pairs (1, 2, 4),
// the activation (0 none, 1 SiLU, 2 GELU) and the step loop's variant the library takes by default on each GPU class (0; 2: the m-tiles with pairs alone, an H100's
// and an A100's; 8 and 10: two steps loaded ahead, an HBM GPU's one-token products), and mma_moe_rows_body (no gate, a warp a row block over the whole K: MT 2
// and 4) and moe_sum_body (the weighted sum of a token's choices). The prompt kernel (many pairs an expert: fold_big_*_moe) and Hopper's wgmma route
// (fold_moe_wg_*) are further down; the variants a development build's tuning variable asks for are the library's still.
#define FOLD_MOE_TIERED(MT, ACT, V, MINB) \
    extern "C" __global__ void __launch_bounds__(256, MINB) fold_moe_tiered_##MT##_##ACT##_##V(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base, uint32_t s1, uint32_t s2, uint32_t s3, int64_t O, int64_t K, const int* plan, int E, int64_t k, int gather, const __nv_bfloat16* X, const __nv_bfloat16* bias, const void* w, int wf32, __nv_bfloat16* Y, float* Y32, int64_t P, float* parts, int* done, int pf) { \
        glib::mma_moe_body<glib::Tiered, MT, ACT, V>(glib::Tiered{data, blocks, block_base, glib::Tiers{s1, s2, s3}}, O, K, plan, E, k, gather, X, bias, w, wf32, Y, Y32, P, parts, done, pf); \
    }
#define FOLD_MOE_ROWS_TIERED(MT, V) \
    extern "C" __global__ void __launch_bounds__(256, 1) fold_moe_rows_tiered_##MT##_##V(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base, uint32_t s1, uint32_t s2, uint32_t s3, int64_t O, int64_t K, const int* plan, int E, int64_t k, int gather, const __nv_bfloat16* X, const __nv_bfloat16* bias, const void* w, int wf32, __nv_bfloat16* Y, float* Y32, int pf) { \
        glib::mma_moe_rows_body<glib::Tiered, MT, V>(glib::Tiered{data, blocks, block_base, glib::Tiers{s1, s2, s3}}, O, K, plan, E, k, gather, X, bias, w, wf32, Y, Y32, pf); \
    }
FOLD_MOE_TIERED(1, 0, 0, 2)
FOLD_MOE_TIERED(1, 0, 2, 2)
FOLD_MOE_TIERED(1, 0, 8, 2)
FOLD_MOE_TIERED(1, 0, 10, 2)
FOLD_MOE_TIERED(1, 1, 0, 2)
FOLD_MOE_TIERED(1, 1, 2, 2)
FOLD_MOE_TIERED(1, 1, 8, 2)
FOLD_MOE_TIERED(1, 1, 10, 2)
FOLD_MOE_TIERED(1, 2, 0, 2)
FOLD_MOE_TIERED(1, 2, 2, 2)
FOLD_MOE_TIERED(1, 2, 8, 2)
FOLD_MOE_TIERED(1, 2, 10, 2)
FOLD_MOE_TIERED(2, 0, 0, 1)
FOLD_MOE_TIERED(2, 0, 2, 1)
FOLD_MOE_TIERED(2, 1, 0, 1)
FOLD_MOE_TIERED(2, 1, 2, 1)
FOLD_MOE_TIERED(2, 2, 0, 1)
FOLD_MOE_TIERED(2, 2, 2, 1)
FOLD_MOE_TIERED(4, 0, 0, 1)
FOLD_MOE_TIERED(4, 0, 2, 1)
FOLD_MOE_TIERED(4, 1, 0, 1)
FOLD_MOE_TIERED(4, 1, 2, 1)
FOLD_MOE_TIERED(4, 2, 0, 1)
FOLD_MOE_TIERED(4, 2, 2, 1)
FOLD_MOE_ROWS_TIERED(2, 0)
FOLD_MOE_ROWS_TIERED(2, 2)
FOLD_MOE_ROWS_TIERED(4, 0)
FOLD_MOE_ROWS_TIERED(4, 2)
#define FOLD_MOE_PLACED(MT, ACT, V, MINB) \
    extern "C" __global__ void __launch_bounds__(256, MINB) fold_moe_placed_##MT##_##ACT##_##V(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base, uint32_t s1, uint32_t s2, uint32_t s3, int64_t per, int64_t stride, int64_t O, int64_t K, const int* plan, int E, int64_t k, int gather, const __nv_bfloat16* X, const __nv_bfloat16* bias, const void* w, int wf32, __nv_bfloat16* Y, float* Y32, int64_t P, float* parts, int* done, int pf) { \
        glib::mma_moe_body<glib::TieredPlaced, MT, ACT, V>(glib::TieredPlaced{data, blocks, block_base, glib::Tiers{s1, s2, s3}, per, stride}, O, K, plan, E, k, gather, X, bias, w, wf32, Y, Y32, P, parts, done, pf); \
    }
#define FOLD_MOE_ROWS_PLACED(MT, V) \
    extern "C" __global__ void __launch_bounds__(256, 1) fold_moe_rows_placed_##MT##_##V(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base, uint32_t s1, uint32_t s2, uint32_t s3, int64_t per, int64_t stride, int64_t O, int64_t K, const int* plan, int E, int64_t k, int gather, const __nv_bfloat16* X, const __nv_bfloat16* bias, const void* w, int wf32, __nv_bfloat16* Y, float* Y32, int pf) { \
        glib::mma_moe_rows_body<glib::TieredPlaced, MT, V>(glib::TieredPlaced{data, blocks, block_base, glib::Tiers{s1, s2, s3}, per, stride}, O, K, plan, E, k, gather, X, bias, w, wf32, Y, Y32, pf); \
    }
FOLD_MOE_PLACED(1, 0, 0, 2)
FOLD_MOE_PLACED(1, 0, 2, 2)
FOLD_MOE_PLACED(1, 0, 8, 2)
FOLD_MOE_PLACED(1, 0, 10, 2)
FOLD_MOE_PLACED(1, 1, 0, 2)
FOLD_MOE_PLACED(1, 1, 2, 2)
FOLD_MOE_PLACED(1, 1, 8, 2)
FOLD_MOE_PLACED(1, 1, 10, 2)
FOLD_MOE_PLACED(1, 2, 0, 2)
FOLD_MOE_PLACED(1, 2, 2, 2)
FOLD_MOE_PLACED(1, 2, 8, 2)
FOLD_MOE_PLACED(1, 2, 10, 2)
FOLD_MOE_PLACED(2, 0, 0, 1)
FOLD_MOE_PLACED(2, 0, 2, 1)
FOLD_MOE_PLACED(2, 1, 0, 1)
FOLD_MOE_PLACED(2, 1, 2, 1)
FOLD_MOE_PLACED(2, 2, 0, 1)
FOLD_MOE_PLACED(2, 2, 2, 1)
FOLD_MOE_PLACED(4, 0, 0, 1)
FOLD_MOE_PLACED(4, 0, 2, 1)
FOLD_MOE_PLACED(4, 1, 0, 1)
FOLD_MOE_PLACED(4, 1, 2, 1)
FOLD_MOE_PLACED(4, 2, 0, 1)
FOLD_MOE_PLACED(4, 2, 2, 1)
FOLD_MOE_ROWS_PLACED(2, 0)
FOLD_MOE_ROWS_PLACED(2, 2)
FOLD_MOE_ROWS_PLACED(4, 0)
FOLD_MOE_ROWS_PLACED(4, 2)
#define FOLD_MOE_INT6(MT, ACT, V, MINB) \
    extern "C" __global__ void __launch_bounds__(256, MINB) fold_moe_int6_##MT##_##ACT##_##V(const uint8_t* data, const uint8_t* scales, int64_t O, int64_t K, const int* plan, int E, int64_t k, int gather, const __nv_bfloat16* X, const __nv_bfloat16* bias, const void* w, int wf32, __nv_bfloat16* Y, float* Y32, int64_t P, float* parts, int* done, int pf) { \
        glib::mma_moe_body<glib::Int6, MT, ACT, V>(glib::Int6{data, scales}, O, K, plan, E, k, gather, X, bias, w, wf32, Y, Y32, P, parts, done, pf); \
    }
#define FOLD_MOE_ROWS_INT6(MT, V) \
    extern "C" __global__ void __launch_bounds__(256, 1) fold_moe_rows_int6_##MT##_##V(const uint8_t* data, const uint8_t* scales, int64_t O, int64_t K, const int* plan, int E, int64_t k, int gather, const __nv_bfloat16* X, const __nv_bfloat16* bias, const void* w, int wf32, __nv_bfloat16* Y, float* Y32, int pf) { \
        glib::mma_moe_rows_body<glib::Int6, MT, V>(glib::Int6{data, scales}, O, K, plan, E, k, gather, X, bias, w, wf32, Y, Y32, pf); \
    }
FOLD_MOE_INT6(1, 0, 0, 2)
FOLD_MOE_INT6(1, 0, 2, 2)
FOLD_MOE_INT6(1, 0, 8, 2)
FOLD_MOE_INT6(1, 0, 10, 2)
FOLD_MOE_INT6(1, 1, 0, 2)
FOLD_MOE_INT6(1, 1, 2, 2)
FOLD_MOE_INT6(1, 1, 8, 2)
FOLD_MOE_INT6(1, 1, 10, 2)
FOLD_MOE_INT6(1, 2, 0, 2)
FOLD_MOE_INT6(1, 2, 2, 2)
FOLD_MOE_INT6(1, 2, 8, 2)
FOLD_MOE_INT6(1, 2, 10, 2)
FOLD_MOE_INT6(2, 0, 0, 1)
FOLD_MOE_INT6(2, 0, 2, 1)
FOLD_MOE_INT6(2, 1, 0, 1)
FOLD_MOE_INT6(2, 1, 2, 1)
FOLD_MOE_INT6(2, 2, 0, 1)
FOLD_MOE_INT6(2, 2, 2, 1)
FOLD_MOE_INT6(4, 0, 0, 1)
FOLD_MOE_INT6(4, 0, 2, 1)
FOLD_MOE_INT6(4, 1, 0, 1)
FOLD_MOE_INT6(4, 1, 2, 1)
FOLD_MOE_INT6(4, 2, 0, 1)
FOLD_MOE_INT6(4, 2, 2, 1)
FOLD_MOE_ROWS_INT6(2, 0)
FOLD_MOE_ROWS_INT6(2, 2)
FOLD_MOE_ROWS_INT6(4, 0)
FOLD_MOE_ROWS_INT6(4, 2)
#define FOLD_MOE_INT5(MT, ACT, V, MINB) \
    extern "C" __global__ void __launch_bounds__(256, MINB) fold_moe_int5_##MT##_##ACT##_##V(const uint8_t* data, const uint8_t* scales, int64_t O, int64_t K, const int* plan, int E, int64_t k, int gather, const __nv_bfloat16* X, const __nv_bfloat16* bias, const void* w, int wf32, __nv_bfloat16* Y, float* Y32, int64_t P, float* parts, int* done, int pf) { \
        glib::mma_moe_body<glib::Int5, MT, ACT, V>(glib::Int5{data, scales}, O, K, plan, E, k, gather, X, bias, w, wf32, Y, Y32, P, parts, done, pf); \
    }
#define FOLD_MOE_ROWS_INT5(MT, V) \
    extern "C" __global__ void __launch_bounds__(256, 1) fold_moe_rows_int5_##MT##_##V(const uint8_t* data, const uint8_t* scales, int64_t O, int64_t K, const int* plan, int E, int64_t k, int gather, const __nv_bfloat16* X, const __nv_bfloat16* bias, const void* w, int wf32, __nv_bfloat16* Y, float* Y32, int pf) { \
        glib::mma_moe_rows_body<glib::Int5, MT, V>(glib::Int5{data, scales}, O, K, plan, E, k, gather, X, bias, w, wf32, Y, Y32, pf); \
    }
FOLD_MOE_INT5(1, 0, 0, 2)
FOLD_MOE_INT5(1, 0, 2, 2)
FOLD_MOE_INT5(1, 0, 8, 2)
FOLD_MOE_INT5(1, 0, 10, 2)
FOLD_MOE_INT5(1, 1, 0, 2)
FOLD_MOE_INT5(1, 1, 2, 2)
FOLD_MOE_INT5(1, 1, 8, 2)
FOLD_MOE_INT5(1, 1, 10, 2)
FOLD_MOE_INT5(1, 2, 0, 2)
FOLD_MOE_INT5(1, 2, 2, 2)
FOLD_MOE_INT5(1, 2, 8, 2)
FOLD_MOE_INT5(1, 2, 10, 2)
FOLD_MOE_INT5(2, 0, 0, 1)
FOLD_MOE_INT5(2, 0, 2, 1)
FOLD_MOE_INT5(2, 1, 0, 1)
FOLD_MOE_INT5(2, 1, 2, 1)
FOLD_MOE_INT5(2, 2, 0, 1)
FOLD_MOE_INT5(2, 2, 2, 1)
FOLD_MOE_INT5(4, 0, 0, 1)
FOLD_MOE_INT5(4, 0, 2, 1)
FOLD_MOE_INT5(4, 1, 0, 1)
FOLD_MOE_INT5(4, 1, 2, 1)
FOLD_MOE_INT5(4, 2, 0, 1)
FOLD_MOE_INT5(4, 2, 2, 1)
FOLD_MOE_ROWS_INT5(2, 0)
FOLD_MOE_ROWS_INT5(2, 2)
FOLD_MOE_ROWS_INT5(4, 0)
FOLD_MOE_ROWS_INT5(4, 2)
#define FOLD_MOE_INT4(MT, ACT, V, MINB) \
    extern "C" __global__ void __launch_bounds__(256, MINB) fold_moe_int4_##MT##_##ACT##_##V(const uint8_t* data, const uint8_t* scales, int64_t O, int64_t K, const int* plan, int E, int64_t k, int gather, const __nv_bfloat16* X, const __nv_bfloat16* bias, const void* w, int wf32, __nv_bfloat16* Y, float* Y32, int64_t P, float* parts, int* done, int pf) { \
        glib::mma_moe_body<glib::Int4, MT, ACT, V>(glib::Int4{data, scales}, O, K, plan, E, k, gather, X, bias, w, wf32, Y, Y32, P, parts, done, pf); \
    }
#define FOLD_MOE_ROWS_INT4(MT, V) \
    extern "C" __global__ void __launch_bounds__(256, 1) fold_moe_rows_int4_##MT##_##V(const uint8_t* data, const uint8_t* scales, int64_t O, int64_t K, const int* plan, int E, int64_t k, int gather, const __nv_bfloat16* X, const __nv_bfloat16* bias, const void* w, int wf32, __nv_bfloat16* Y, float* Y32, int pf) { \
        glib::mma_moe_rows_body<glib::Int4, MT, V>(glib::Int4{data, scales}, O, K, plan, E, k, gather, X, bias, w, wf32, Y, Y32, pf); \
    }
FOLD_MOE_INT4(1, 0, 0, 2)
FOLD_MOE_INT4(1, 0, 2, 2)
FOLD_MOE_INT4(1, 0, 8, 2)
FOLD_MOE_INT4(1, 0, 10, 2)
FOLD_MOE_INT4(1, 1, 0, 2)
FOLD_MOE_INT4(1, 1, 2, 2)
FOLD_MOE_INT4(1, 1, 8, 2)
FOLD_MOE_INT4(1, 1, 10, 2)
FOLD_MOE_INT4(1, 2, 0, 2)
FOLD_MOE_INT4(1, 2, 2, 2)
FOLD_MOE_INT4(1, 2, 8, 2)
FOLD_MOE_INT4(1, 2, 10, 2)
FOLD_MOE_INT4(2, 0, 0, 1)
FOLD_MOE_INT4(2, 0, 2, 1)
FOLD_MOE_INT4(2, 1, 0, 1)
FOLD_MOE_INT4(2, 1, 2, 1)
FOLD_MOE_INT4(2, 2, 0, 1)
FOLD_MOE_INT4(2, 2, 2, 1)
FOLD_MOE_INT4(4, 0, 0, 1)
FOLD_MOE_INT4(4, 0, 2, 1)
FOLD_MOE_INT4(4, 1, 0, 1)
FOLD_MOE_INT4(4, 1, 2, 1)
FOLD_MOE_INT4(4, 2, 0, 1)
FOLD_MOE_INT4(4, 2, 2, 1)
FOLD_MOE_ROWS_INT4(2, 0)
FOLD_MOE_ROWS_INT4(2, 2)
FOLD_MOE_ROWS_INT4(4, 0)
FOLD_MOE_ROWS_INT4(4, 2)
#define FOLD_MOE_INT4H(MT, ACT, V, MINB) \
    extern "C" __global__ void __launch_bounds__(256, MINB) fold_moe_int4h_##MT##_##ACT##_##V(const uint8_t* data, const uint8_t* scales, uint32_t base, int64_t O, int64_t K, const int* plan, int E, int64_t k, int gather, const __nv_bfloat16* X, const __nv_bfloat16* bias, const void* w, int wf32, __nv_bfloat16* Y, float* Y32, int64_t P, float* parts, int* done, int pf) { \
        glib::mma_moe_body<glib::Int4H, MT, ACT, V>(glib::Int4H{{data, scales}, base}, O, K, plan, E, k, gather, X, bias, w, wf32, Y, Y32, P, parts, done, pf); \
    }
#define FOLD_MOE_ROWS_INT4H(MT, V) \
    extern "C" __global__ void __launch_bounds__(256, 1) fold_moe_rows_int4h_##MT##_##V(const uint8_t* data, const uint8_t* scales, uint32_t base, int64_t O, int64_t K, const int* plan, int E, int64_t k, int gather, const __nv_bfloat16* X, const __nv_bfloat16* bias, const void* w, int wf32, __nv_bfloat16* Y, float* Y32, int pf) { \
        glib::mma_moe_rows_body<glib::Int4H, MT, V>(glib::Int4H{{data, scales}, base}, O, K, plan, E, k, gather, X, bias, w, wf32, Y, Y32, pf); \
    }
FOLD_MOE_INT4H(1, 0, 0, 2)
FOLD_MOE_INT4H(1, 0, 2, 2)
FOLD_MOE_INT4H(1, 0, 8, 2)
FOLD_MOE_INT4H(1, 0, 10, 2)
FOLD_MOE_INT4H(1, 1, 0, 2)
FOLD_MOE_INT4H(1, 1, 2, 2)
FOLD_MOE_INT4H(1, 1, 8, 2)
FOLD_MOE_INT4H(1, 1, 10, 2)
FOLD_MOE_INT4H(1, 2, 0, 2)
FOLD_MOE_INT4H(1, 2, 2, 2)
FOLD_MOE_INT4H(1, 2, 8, 2)
FOLD_MOE_INT4H(1, 2, 10, 2)
FOLD_MOE_INT4H(2, 0, 0, 1)
FOLD_MOE_INT4H(2, 0, 2, 1)
FOLD_MOE_INT4H(2, 1, 0, 1)
FOLD_MOE_INT4H(2, 1, 2, 1)
FOLD_MOE_INT4H(2, 2, 0, 1)
FOLD_MOE_INT4H(2, 2, 2, 1)
FOLD_MOE_INT4H(4, 0, 0, 1)
FOLD_MOE_INT4H(4, 0, 2, 1)
FOLD_MOE_INT4H(4, 1, 0, 1)
FOLD_MOE_INT4H(4, 1, 2, 1)
FOLD_MOE_INT4H(4, 2, 0, 1)
FOLD_MOE_INT4H(4, 2, 2, 1)
FOLD_MOE_ROWS_INT4H(2, 0)
FOLD_MOE_ROWS_INT4H(2, 2)
FOLD_MOE_ROWS_INT4H(4, 0)
FOLD_MOE_ROWS_INT4H(4, 2)
#define FOLD_MOE_FP4(MT, ACT, V, MINB) \
    extern "C" __global__ void __launch_bounds__(256, MINB) fold_moe_fp4_##MT##_##ACT##_##V(const uint8_t* data, const uint8_t* scales, int64_t O, int64_t K, const int* plan, int E, int64_t k, int gather, const __nv_bfloat16* X, const __nv_bfloat16* bias, const void* w, int wf32, __nv_bfloat16* Y, float* Y32, int64_t P, float* parts, int* done, int pf) { \
        glib::mma_moe_body<glib::Fp4, MT, ACT, V>(glib::Fp4{data, scales}, O, K, plan, E, k, gather, X, bias, w, wf32, Y, Y32, P, parts, done, pf); \
    }
#define FOLD_MOE_ROWS_FP4(MT, V) \
    extern "C" __global__ void __launch_bounds__(256, 1) fold_moe_rows_fp4_##MT##_##V(const uint8_t* data, const uint8_t* scales, int64_t O, int64_t K, const int* plan, int E, int64_t k, int gather, const __nv_bfloat16* X, const __nv_bfloat16* bias, const void* w, int wf32, __nv_bfloat16* Y, float* Y32, int pf) { \
        glib::mma_moe_rows_body<glib::Fp4, MT, V>(glib::Fp4{data, scales}, O, K, plan, E, k, gather, X, bias, w, wf32, Y, Y32, pf); \
    }
FOLD_MOE_FP4(1, 0, 0, 2)
FOLD_MOE_FP4(1, 0, 2, 2)
FOLD_MOE_FP4(1, 0, 8, 2)
FOLD_MOE_FP4(1, 0, 10, 2)
FOLD_MOE_FP4(1, 1, 0, 2)
FOLD_MOE_FP4(1, 1, 2, 2)
FOLD_MOE_FP4(1, 1, 8, 2)
FOLD_MOE_FP4(1, 1, 10, 2)
FOLD_MOE_FP4(1, 2, 0, 2)
FOLD_MOE_FP4(1, 2, 2, 2)
FOLD_MOE_FP4(1, 2, 8, 2)
FOLD_MOE_FP4(1, 2, 10, 2)
FOLD_MOE_FP4(2, 0, 0, 1)
FOLD_MOE_FP4(2, 0, 2, 1)
FOLD_MOE_FP4(2, 1, 0, 1)
FOLD_MOE_FP4(2, 1, 2, 1)
FOLD_MOE_FP4(2, 2, 0, 1)
FOLD_MOE_FP4(2, 2, 2, 1)
FOLD_MOE_FP4(4, 0, 0, 1)
FOLD_MOE_FP4(4, 0, 2, 1)
FOLD_MOE_FP4(4, 1, 0, 1)
FOLD_MOE_FP4(4, 1, 2, 1)
FOLD_MOE_FP4(4, 2, 0, 1)
FOLD_MOE_FP4(4, 2, 2, 1)
FOLD_MOE_ROWS_FP4(2, 0)
FOLD_MOE_ROWS_FP4(2, 2)
FOLD_MOE_ROWS_FP4(4, 0)
FOLD_MOE_ROWS_FP4(4, 2)
#define FOLD_MOE_FP4R(MT, ACT, V, MINB) \
    extern "C" __global__ void __launch_bounds__(256, MINB) fold_moe_fp4r_##MT##_##ACT##_##V(const uint8_t* data, const uint8_t* scales, uint32_t ks, uint32_t mg, uint32_t sh, int64_t O, int64_t K, const int* plan, int E, int64_t k, int gather, const __nv_bfloat16* X, const __nv_bfloat16* bias, const void* w, int wf32, __nv_bfloat16* Y, float* Y32, int64_t P, float* parts, int* done, int pf) { \
        glib::mma_moe_body<glib::Fp4R, MT, ACT, V>(glib::Fp4R{data, scales, ks, mg, sh}, O, K, plan, E, k, gather, X, bias, w, wf32, Y, Y32, P, parts, done, pf); \
    }
#define FOLD_MOE_ROWS_FP4R(MT, V) \
    extern "C" __global__ void __launch_bounds__(256, 1) fold_moe_rows_fp4r_##MT##_##V(const uint8_t* data, const uint8_t* scales, uint32_t ks, uint32_t mg, uint32_t sh, int64_t O, int64_t K, const int* plan, int E, int64_t k, int gather, const __nv_bfloat16* X, const __nv_bfloat16* bias, const void* w, int wf32, __nv_bfloat16* Y, float* Y32, int pf) { \
        glib::mma_moe_rows_body<glib::Fp4R, MT, V>(glib::Fp4R{data, scales, ks, mg, sh}, O, K, plan, E, k, gather, X, bias, w, wf32, Y, Y32, pf); \
    }
FOLD_MOE_FP4R(1, 0, 0, 2)
FOLD_MOE_FP4R(1, 0, 2, 2)
FOLD_MOE_FP4R(1, 0, 8, 2)
FOLD_MOE_FP4R(1, 0, 10, 2)
FOLD_MOE_FP4R(1, 1, 0, 2)
FOLD_MOE_FP4R(1, 1, 2, 2)
FOLD_MOE_FP4R(1, 1, 8, 2)
FOLD_MOE_FP4R(1, 1, 10, 2)
FOLD_MOE_FP4R(1, 2, 0, 2)
FOLD_MOE_FP4R(1, 2, 2, 2)
FOLD_MOE_FP4R(1, 2, 8, 2)
FOLD_MOE_FP4R(1, 2, 10, 2)
FOLD_MOE_FP4R(2, 0, 0, 1)
FOLD_MOE_FP4R(2, 0, 2, 1)
FOLD_MOE_FP4R(2, 1, 0, 1)
FOLD_MOE_FP4R(2, 1, 2, 1)
FOLD_MOE_FP4R(2, 2, 0, 1)
FOLD_MOE_FP4R(2, 2, 2, 1)
FOLD_MOE_FP4R(4, 0, 0, 1)
FOLD_MOE_FP4R(4, 0, 2, 1)
FOLD_MOE_FP4R(4, 1, 0, 1)
FOLD_MOE_FP4R(4, 1, 2, 1)
FOLD_MOE_FP4R(4, 2, 0, 1)
FOLD_MOE_FP4R(4, 2, 2, 1)
FOLD_MOE_ROWS_FP4R(2, 0)
FOLD_MOE_ROWS_FP4R(2, 2)
FOLD_MOE_ROWS_FP4R(4, 0)
FOLD_MOE_ROWS_FP4R(4, 2)
#define FOLD_MOE_TWELVE(MT, ACT, V, MINB) \
    extern "C" __global__ void __launch_bounds__(256, MINB) fold_moe_twelve_##MT##_##ACT##_##V(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, uint32_t hb4, int xmax, int64_t O, int64_t K, const int* plan, int E, int64_t k, int gather, const __nv_bfloat16* X, const __nv_bfloat16* bias, const void* w, int wf32, __nv_bfloat16* Y, float* Y32, int64_t P, float* parts, int* done, int pf) { \
        glib::mma_moe_body<glib::Nib, MT, ACT, V>(glib::Nib{data, exc, exc_base, hb4, xmax}, O, K, plan, E, k, gather, X, bias, w, wf32, Y, Y32, P, parts, done, pf); \
    }
#define FOLD_MOE_ROWS_TWELVE(MT, V) \
    extern "C" __global__ void __launch_bounds__(256, 1) fold_moe_rows_twelve_##MT##_##V(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, uint32_t hb4, int xmax, int64_t O, int64_t K, const int* plan, int E, int64_t k, int gather, const __nv_bfloat16* X, const __nv_bfloat16* bias, const void* w, int wf32, __nv_bfloat16* Y, float* Y32, int pf) { \
        glib::mma_moe_rows_body<glib::Nib, MT, V>(glib::Nib{data, exc, exc_base, hb4, xmax}, O, K, plan, E, k, gather, X, bias, w, wf32, Y, Y32, pf); \
    }
FOLD_MOE_TWELVE(1, 0, 0, 2)
FOLD_MOE_TWELVE(1, 0, 2, 2)
FOLD_MOE_TWELVE(1, 1, 0, 2)
FOLD_MOE_TWELVE(1, 1, 2, 2)
FOLD_MOE_TWELVE(1, 2, 0, 2)
FOLD_MOE_TWELVE(1, 2, 2, 2)
FOLD_MOE_TWELVE(2, 0, 0, 1)
FOLD_MOE_TWELVE(2, 0, 2, 1)
FOLD_MOE_TWELVE(2, 1, 0, 1)
FOLD_MOE_TWELVE(2, 1, 2, 1)
FOLD_MOE_TWELVE(2, 2, 0, 1)
FOLD_MOE_TWELVE(2, 2, 2, 1)
FOLD_MOE_TWELVE(4, 0, 0, 1)
FOLD_MOE_TWELVE(4, 0, 2, 1)
FOLD_MOE_TWELVE(4, 1, 0, 1)
FOLD_MOE_TWELVE(4, 1, 2, 1)
FOLD_MOE_TWELVE(4, 2, 0, 1)
FOLD_MOE_TWELVE(4, 2, 2, 1)
FOLD_MOE_ROWS_TWELVE(2, 0)
FOLD_MOE_ROWS_TWELVE(2, 2)
FOLD_MOE_ROWS_TWELVE(4, 0)
FOLD_MOE_ROWS_TWELVE(4, 2)
extern "C" __global__ void fold_moe_sum(const float* y32, const int64_t* ids, int E, int64_t T, int64_t k, int64_t O, __nv_bfloat16* y) {
    glib::moe_sum_body(y32, ids, E, T, k, O, y);
}

// glyd_gpu_f8_size, _f8_pack, _f8_unpack and layout H's (f8h_*): the lossless FP8 format's heads, the scan of their var words, the pack and the unpack, a CTA a
// patch, as the library launches them (f8c's *_body).
extern "C" __global__ void __launch_bounds__(glib::f8c::NT) fold_f8_head(const glib::f8c::F8Pk P, uint32_t* heads) {
    glib::f8c::f8_head_body(P, heads);
}
extern "C" __global__ void __launch_bounds__(glib::f8c::NT) fold_f8h_head(const glib::f8c::F8Pk P, uint32_t* heads) {
    glib::f8c::f8h_head_body(P, heads);
}
extern "C" __global__ void __launch_bounds__(glib::f8c::SCAN_NT) fold_f8_scan(uint32_t* heads, int64_t patches, uint64_t* total) {
    glib::f8c::f8_scan_body(heads, patches, total);
}
extern "C" __global__ void __launch_bounds__(glib::f8c::NT) fold_f8_pack(const glib::f8c::F8Pk P, const uint32_t* heads, uint8_t* core_out, uint32_t* var, int64_t var_words) {
    glib::f8c::f8_pack_body(P, heads, core_out, var, var_words);
}
extern "C" __global__ void __launch_bounds__(glib::f8c::NT) fold_f8h_pack(const glib::f8c::F8Pk P, const uint32_t* heads, uint8_t* core_out, uint32_t* var, int64_t var_words) {
    glib::f8c::f8h_pack_body(P, heads, core_out, var, var_words);
}
extern "C" __global__ void __launch_bounds__(glib::f8c::NT) fold_f8_unpack(const glib::f8c::F8Un P) {
    glib::f8c::f8_unpack_body(P);
}
extern "C" __global__ void __launch_bounds__(glib::f8c::HUNT) fold_f8h_unpack(const glib::f8c::F8Un P) {
    glib::f8c::f8h_unpack_body(P);
}

// glyd_gpu_f8_linear, _f8_moe, _f8h_linear and _f8h_moe (the FP8 products, Y = X W^T for 1 to 64 tokens, W in the lossless format) at the library's default plan:
// f8_linear_body with no decode ahead, no spills' pass, bf16 X and two slots, for 8-token tiles a warp of 1, 2, 4 and 8 (NTL), each kind of scales
// (block, tensor, mx, ds32, channel), the experts' product (block scales, a plan's slot a blockIdx.y) and layout H's (block scales, G 0 and 1); and layout H's
// one-token kernels (f8h_lin1_body, and f8b_lin1_body for plain packs) as the library launches them by default (3 CTAs an SM). The plans a development build's
// variable asks for (GLYD_F8_LIN_A8, _AHEAD, _DENSE, GLYD_F8_SLOTS=3, GLYD_F8H_ONE=2 or 4, GLYD_F8H_CTAS=2) are the library's still.
#define FOLD_F8_LIN(NAME, NTL, KIND, G) \
    extern "C" __global__ void __launch_bounds__(glib::f8c::LNT, 1) fold_f8_##NAME##_##NTL(const glib::f8c::F8Lin P) { \
        glib::f8c::f8_linear_body<NTL, KIND, 0, 0, G, 2>(P); \
    }
FOLD_F8_LIN(block, 1, glib::f8c::F8_BLOCK, 0)
FOLD_F8_LIN(block, 2, glib::f8c::F8_BLOCK, 0)
FOLD_F8_LIN(block, 4, glib::f8c::F8_BLOCK, 0)
FOLD_F8_LIN(block, 8, glib::f8c::F8_BLOCK, 0)
FOLD_F8_LIN(tensor, 1, glib::f8c::F8_TENSOR, 0)
FOLD_F8_LIN(tensor, 2, glib::f8c::F8_TENSOR, 0)
FOLD_F8_LIN(tensor, 4, glib::f8c::F8_TENSOR, 0)
FOLD_F8_LIN(tensor, 8, glib::f8c::F8_TENSOR, 0)
FOLD_F8_LIN(mx, 1, glib::f8c::F8_MX, 0)
FOLD_F8_LIN(mx, 2, glib::f8c::F8_MX, 0)
FOLD_F8_LIN(mx, 4, glib::f8c::F8_MX, 0)
FOLD_F8_LIN(mx, 8, glib::f8c::F8_MX, 0)
FOLD_F8_LIN(ds32, 1, glib::f8c::F8_DS32, 0)
FOLD_F8_LIN(ds32, 2, glib::f8c::F8_DS32, 0)
FOLD_F8_LIN(ds32, 4, glib::f8c::F8_DS32, 0)
FOLD_F8_LIN(ds32, 8, glib::f8c::F8_DS32, 0)
FOLD_F8_LIN(channel, 1, glib::f8c::F8_CHANNEL, 0)
FOLD_F8_LIN(channel, 2, glib::f8c::F8_CHANNEL, 0)
FOLD_F8_LIN(channel, 4, glib::f8c::F8_CHANNEL, 0)
FOLD_F8_LIN(channel, 8, glib::f8c::F8_CHANNEL, 0)
FOLD_F8_LIN(moe, 1, glib::f8c::F8_BLOCK, 1)
FOLD_F8_LIN(moe, 2, glib::f8c::F8_BLOCK, 1)
FOLD_F8_LIN(moe, 4, glib::f8c::F8_BLOCK, 1)
FOLD_F8_LIN(moe, 8, glib::f8c::F8_BLOCK, 1)
FOLD_F8_LIN(h, 1, glib::f8c::F8_BLOCK | glib::f8c::F8_H, 0)
FOLD_F8_LIN(h, 2, glib::f8c::F8_BLOCK | glib::f8c::F8_H, 0)
FOLD_F8_LIN(h, 4, glib::f8c::F8_BLOCK | glib::f8c::F8_H, 0)
FOLD_F8_LIN(h, 8, glib::f8c::F8_BLOCK | glib::f8c::F8_H, 0)
FOLD_F8_LIN(hmoe, 1, glib::f8c::F8_BLOCK | glib::f8c::F8_H, 1)
FOLD_F8_LIN(hmoe, 2, glib::f8c::F8_BLOCK | glib::f8c::F8_H, 1)
FOLD_F8_LIN(hmoe, 4, glib::f8c::F8_BLOCK | glib::f8c::F8_H, 1)
FOLD_F8_LIN(hmoe, 8, glib::f8c::F8_BLOCK | glib::f8c::F8_H, 1)
#define FOLD_F8H_ONE(NAME, BODY, G) \
    extern "C" __global__ void __launch_bounds__(glib::f8c::LNT, 3) fold_f8h_one_##NAME##_##G(const glib::f8c::F8Lin P) { \
        glib::f8c::BODY<G, 3>(P); \
    }
FOLD_F8H_ONE(coded, f8h_lin1_body, 0)
FOLD_F8H_ONE(coded, f8h_lin1_body, 1)
FOLD_F8H_ONE(plain, f8b_lin1_body, 0)
FOLD_F8H_ONE(plain, f8b_lin1_body, 1)

// glyd_gpu_mma_linear's and _mma12_linear's prompt kernel (mma_gemm_big_any: the routes BIG, DECODE, AHEAD and SPLIT, K a multiple of 64) at the library's
// default plans: mma_gemm_big_body in blocks of 128 tokens by two row blocks (v1), of 256 by one (v2) and, the 12-bit layout's on an A100, of 256 by two
// with eight consumer warps (v3), K split over blocks as mma_gemm_grid_run splits it (finish_body adds the splits); on GeForce Ada mma_gemm_sk_body
// (stream-K) in its two shapes; the experts' prompt kernel (mma_moe_big: v1 with the plan, MOE); and the 12-bit layout's route MID (mma12_mid_any:
// mma12_mid_body in tiles of 16, 32 and 64 tokens and two or four warpgroups, mma12_ws_body in an A100's four tiles of 32 to 128 tokens).
// Their grids, splits and workspaces are Rust (src/fold.rs), as the library's host code makes them.
extern "C" __global__ void fold_finish(const float* y32, int64_t split, const __nv_bfloat16* bias, int64_t M, int64_t O, __nv_bfloat16* y) {
    glib::finish_body(y32, split, bias, M, O, y);
}
#define FOLD_BIG_T(NAME, FMT, CW, PW, NB, RBB, MOE, TAG) \
    extern "C" __global__ void __launch_bounds__(32 * (CW + PW), 1) fold_big_##NAME##_##TAG(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base, uint32_t s1, uint32_t s2, uint32_t s3, int64_t O, int64_t K, int64_t M, int64_t stages_per_split, const __nv_bfloat16* X, const __nv_bfloat16* bias, __nv_bfloat16* Y, float* Y32, const glib::MoePairs moe) { \
        glib::mma_gemm_big_body<glib::FMT, CW, PW, NB, RBB, MOE>(glib::FMT{data, blocks, block_base, glib::Tiers{s1, s2, s3}}, O, K, M, stages_per_split, X, bias, Y, Y32, moe); \
    }
#define FOLD_SK_T(NAME, FMT, CW, PW, NB, RBB, CR, TAG) \
    extern "C" __global__ void __launch_bounds__(32 * (CW + PW), 1) fold_sk_##NAME##_##TAG(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base, uint32_t s1, uint32_t s2, uint32_t s3, int64_t O, int64_t K, int64_t M, const __nv_bfloat16* X, const __nv_bfloat16* bias, __nv_bfloat16* Y, float* parts, int* done) { \
        glib::mma_gemm_sk_body<glib::FMT, CW, PW, NB, RBB, CR, false>(glib::FMT{data, blocks, block_base, glib::Tiers{s1, s2, s3}}, O, K, M, X, bias, Y, parts, done); \
    }
#define FOLD_BIG_P(NAME, FMT, CW, PW, NB, RBB, MOE, TAG) \
    extern "C" __global__ void __launch_bounds__(32 * (CW + PW), 1) fold_big_##NAME##_##TAG(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base, uint32_t s1, uint32_t s2, uint32_t s3, int64_t per, int64_t stride, int64_t O, int64_t K, int64_t M, int64_t stages_per_split, const __nv_bfloat16* X, const __nv_bfloat16* bias, __nv_bfloat16* Y, float* Y32, const glib::MoePairs moe) { \
        glib::mma_gemm_big_body<glib::FMT, CW, PW, NB, RBB, MOE>(glib::FMT{data, blocks, block_base, glib::Tiers{s1, s2, s3}, per, stride}, O, K, M, stages_per_split, X, bias, Y, Y32, moe); \
    }
#define FOLD_BIG_B(NAME, FMT, CW, PW, NB, RBB, MOE, TAG) \
    extern "C" __global__ void __launch_bounds__(32 * (CW + PW), 1) fold_big_##NAME##_##TAG(const uint8_t* data, const uint8_t* scales, int64_t O, int64_t K, int64_t M, int64_t stages_per_split, const __nv_bfloat16* X, const __nv_bfloat16* bias, __nv_bfloat16* Y, float* Y32, const glib::MoePairs moe) { \
        glib::mma_gemm_big_body<glib::FMT, CW, PW, NB, RBB, MOE>(glib::FMT{data, scales}, O, K, M, stages_per_split, X, bias, Y, Y32, moe); \
    }
#define FOLD_SK_B(NAME, FMT, CW, PW, NB, RBB, CR, TAG) \
    extern "C" __global__ void __launch_bounds__(32 * (CW + PW), 1) fold_sk_##NAME##_##TAG(const uint8_t* data, const uint8_t* scales, int64_t O, int64_t K, int64_t M, const __nv_bfloat16* X, const __nv_bfloat16* bias, __nv_bfloat16* Y, float* parts, int* done) { \
        glib::mma_gemm_sk_body<glib::FMT, CW, PW, NB, RBB, CR, false>(glib::FMT{data, scales}, O, K, M, X, bias, Y, parts, done); \
    }
#define FOLD_BIG_4H(NAME, FMT, CW, PW, NB, RBB, MOE, TAG) \
    extern "C" __global__ void __launch_bounds__(32 * (CW + PW), 1) fold_big_##NAME##_##TAG(const uint8_t* data, const uint8_t* scales, uint32_t base, int64_t O, int64_t K, int64_t M, int64_t stages_per_split, const __nv_bfloat16* X, const __nv_bfloat16* bias, __nv_bfloat16* Y, float* Y32, const glib::MoePairs moe) { \
        glib::mma_gemm_big_body<glib::FMT, CW, PW, NB, RBB, MOE>(glib::FMT{{data, scales}, base}, O, K, M, stages_per_split, X, bias, Y, Y32, moe); \
    }
#define FOLD_SK_4H(NAME, FMT, CW, PW, NB, RBB, CR, TAG) \
    extern "C" __global__ void __launch_bounds__(32 * (CW + PW), 1) fold_sk_##NAME##_##TAG(const uint8_t* data, const uint8_t* scales, uint32_t base, int64_t O, int64_t K, int64_t M, const __nv_bfloat16* X, const __nv_bfloat16* bias, __nv_bfloat16* Y, float* parts, int* done) { \
        glib::mma_gemm_sk_body<glib::FMT, CW, PW, NB, RBB, CR, false>(glib::FMT{{data, scales}, base}, O, K, M, X, bias, Y, parts, done); \
    }
#define FOLD_BIG_4R(NAME, FMT, CW, PW, NB, RBB, MOE, TAG) \
    extern "C" __global__ void __launch_bounds__(32 * (CW + PW), 1) fold_big_##NAME##_##TAG(const uint8_t* data, const uint8_t* scales, uint32_t ks, uint32_t mg, uint32_t sh, int64_t O, int64_t K, int64_t M, int64_t stages_per_split, const __nv_bfloat16* X, const __nv_bfloat16* bias, __nv_bfloat16* Y, float* Y32, const glib::MoePairs moe) { \
        glib::mma_gemm_big_body<glib::FMT, CW, PW, NB, RBB, MOE>(glib::FMT{data, scales, ks, mg, sh}, O, K, M, stages_per_split, X, bias, Y, Y32, moe); \
    }
#define FOLD_SK_4R(NAME, FMT, CW, PW, NB, RBB, CR, TAG) \
    extern "C" __global__ void __launch_bounds__(32 * (CW + PW), 1) fold_sk_##NAME##_##TAG(const uint8_t* data, const uint8_t* scales, uint32_t ks, uint32_t mg, uint32_t sh, int64_t O, int64_t K, int64_t M, const __nv_bfloat16* X, const __nv_bfloat16* bias, __nv_bfloat16* Y, float* parts, int* done) { \
        glib::mma_gemm_sk_body<glib::FMT, CW, PW, NB, RBB, CR, false>(glib::FMT{data, scales, ks, mg, sh}, O, K, M, X, bias, Y, parts, done); \
    }
#define FOLD_BIG_12(NAME, FMT, CW, PW, NB, RBB, MOE, TAG) \
    extern "C" __global__ void __launch_bounds__(32 * (CW + PW), 1) fold_big_##NAME##_##TAG(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, uint32_t hb4, int xmax, int64_t O, int64_t K, int64_t M, int64_t stages_per_split, const __nv_bfloat16* X, const __nv_bfloat16* bias, __nv_bfloat16* Y, float* Y32, const glib::MoePairs moe) { \
        glib::mma_gemm_big_body<glib::FMT, CW, PW, NB, RBB, MOE>(glib::FMT{data, exc, exc_base, hb4, xmax}, O, K, M, stages_per_split, X, bias, Y, Y32, moe); \
    }
#define FOLD_SK_12(NAME, FMT, CW, PW, NB, RBB, CR, TAG) \
    extern "C" __global__ void __launch_bounds__(32 * (CW + PW), 1) fold_sk_##NAME##_##TAG(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, uint32_t hb4, int xmax, int64_t O, int64_t K, int64_t M, const __nv_bfloat16* X, const __nv_bfloat16* bias, __nv_bfloat16* Y, float* parts, int* done) { \
        glib::mma_gemm_sk_body<glib::FMT, CW, PW, NB, RBB, CR, false>(glib::FMT{data, exc, exc_base, hb4, xmax}, O, K, M, X, bias, Y, parts, done); \
    }
FOLD_BIG_T(tiered, Tiered, 4, 4, 3, 2, false, v1)
FOLD_BIG_T(tiered, Tiered, 4, 4, 2, 1, false, v2)
FOLD_BIG_T(tiered, Tiered, 4, 4, 3, 2, true, moe)
FOLD_SK_T(tiered, Tiered, 4, 4, 3, 2, 64, v1)
FOLD_SK_T(tiered, Tiered, 8, 4, 2, 1, 32, v2)
FOLD_BIG_T(compact, Compact, 4, 4, 3, 2, false, v1)
FOLD_BIG_T(compact, Compact, 4, 4, 2, 1, false, v2)
FOLD_SK_T(compact, Compact, 4, 4, 3, 2, 64, v1)
FOLD_SK_T(compact, Compact, 8, 4, 2, 1, 32, v2)
FOLD_BIG_B(int6, Int6, 4, 4, 3, 2, false, v1)
FOLD_BIG_B(int6, Int6, 4, 4, 2, 1, false, v2)
FOLD_BIG_B(int6, Int6, 4, 4, 3, 2, true, moe)
FOLD_SK_B(int6, Int6, 4, 4, 3, 2, 64, v1)
FOLD_SK_B(int6, Int6, 8, 4, 2, 1, 32, v2)
FOLD_BIG_B(int5, Int5, 4, 4, 3, 2, false, v1)
FOLD_BIG_B(int5, Int5, 4, 4, 2, 1, false, v2)
FOLD_BIG_B(int5, Int5, 4, 4, 3, 2, true, moe)
FOLD_SK_B(int5, Int5, 4, 4, 3, 2, 64, v1)
FOLD_SK_B(int5, Int5, 8, 4, 2, 1, 32, v2)
FOLD_BIG_B(int4, Int4, 4, 4, 3, 2, false, v1)
FOLD_BIG_B(int4, Int4, 4, 4, 2, 1, false, v2)
FOLD_BIG_B(int4, Int4, 4, 4, 3, 2, true, moe)
FOLD_SK_B(int4, Int4, 4, 4, 3, 2, 64, v1)
FOLD_SK_B(int4, Int4, 8, 4, 2, 1, 32, v2)
FOLD_BIG_4H(int4h, Int4H, 4, 4, 3, 2, false, v1)
FOLD_BIG_4H(int4h, Int4H, 4, 4, 2, 1, false, v2)
FOLD_BIG_4H(int4h, Int4H, 4, 4, 3, 2, true, moe)
FOLD_SK_4H(int4h, Int4H, 4, 4, 3, 2, 64, v1)
FOLD_SK_4H(int4h, Int4H, 8, 4, 2, 1, 32, v2)
FOLD_BIG_B(fp4, Fp4, 4, 4, 3, 2, false, v1)
FOLD_BIG_B(fp4, Fp4, 4, 4, 2, 1, false, v2)
FOLD_BIG_B(fp4, Fp4, 4, 4, 3, 2, true, moe)
FOLD_SK_B(fp4, Fp4, 4, 4, 3, 2, 64, v1)
FOLD_SK_B(fp4, Fp4, 8, 4, 2, 1, 32, v2)
FOLD_BIG_4R(fp4r, Fp4R, 4, 4, 3, 2, false, v1)
FOLD_BIG_4R(fp4r, Fp4R, 4, 4, 2, 1, false, v2)
FOLD_BIG_4R(fp4r, Fp4R, 4, 4, 3, 2, true, moe)
FOLD_SK_4R(fp4r, Fp4R, 4, 4, 3, 2, 64, v1)
FOLD_SK_4R(fp4r, Fp4R, 8, 4, 2, 1, 32, v2)
FOLD_BIG_B(int3, Int3, 4, 4, 3, 2, false, v1)
FOLD_BIG_B(int3, Int3, 4, 4, 2, 1, false, v2)
FOLD_SK_B(int3, Int3, 4, 4, 3, 2, 64, v1)
FOLD_SK_B(int3, Int3, 8, 4, 2, 1, 32, v2)
FOLD_BIG_12(twelve, Nib, 4, 4, 3, 2, false, v1)
FOLD_BIG_12(twelve, Nib, 4, 4, 2, 1, false, v2)
FOLD_BIG_12(twelve, Nib, 8, 4, 2, 2, false, v3)
FOLD_BIG_12(twelve, Nib, 4, 4, 3, 2, true, moe)
FOLD_SK_12(twelve, Nib, 8, 4, 2, 2, 32, v1)
FOLD_SK_12(twelve, Nib, 8, 4, 2, 1, 32, v2)
FOLD_BIG_P(placed, TieredPlaced, 4, 4, 3, 2, true, moe)
#define FOLD_MID12(NT, WG) \
    extern "C" __global__ void __launch_bounds__(128 * WG + 32, 1) fold_mid12_##NT##_##WG(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, uint32_t hb4, int xmax, int64_t O, int64_t K, int64_t M, const __nv_bfloat16* X, const __nv_bfloat16* bias, __nv_bfloat16* Y, float* parts, int* done) { \
        glib::mma12_mid_body<NT, WG>(glib::Nib{data, exc, exc_base, hb4, xmax}, O, K, M, X, bias, Y, parts, done); \
    }
FOLD_MID12(16, 2)
FOLD_MID12(16, 4)
FOLD_MID12(32, 2)
FOLD_MID12(32, 4)
FOLD_MID12(64, 2)
FOLD_MID12(64, 4)
#define FOLD_WS12(CW, NB, NW, RBB, MT, TM) \
    extern "C" __global__ void __launch_bounds__(32 * (CW + 4 * RBB), 1) fold_ws12_##TM(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, uint32_t hb4, int xmax, int64_t O, int64_t K, int64_t M, const __nv_bfloat16* X, const __nv_bfloat16* bias, __nv_bfloat16* Y, float* parts, int* done) { \
        glib::mma12_ws_body<CW, NB, NW, RBB, MT>(glib::Nib{data, exc, exc_base, hb4, xmax}, O, K, M, X, bias, Y, parts, done); \
    }
FOLD_WS12(4, 4, 5, 2, 1, 32)
FOLD_WS12(4, 4, 4, 2, 2, 64)
FOLD_WS12(4, 3, 4, 2, 3, 96)
FOLD_WS12(8, 3, 4, 2, 2, 128)

// Hopper's (sm_90a) products as the library launches them by default: the 12-bit layout's route WG to 128 tokens (mma12_tma_body in tiles of 16 to
// 128 tokens, two warpgroups) and past 128 (mma12_wgp_body, clusters of two, tiles of 192 and 256), the tiered layout's route WG (tiered_tma_body,
// tiles of 16, 32 and 64 tokens, two warpgroups, a block an SM), and the experts' wgmma product (mma12_moe_wg_body, tiles of 16 to 128 pairs, with
// the gate's SiLU and without; moe_wg_tiles_body, moe_wg_gather_body). X reaches them through a tensor map (a __grid_constant__ parameter, made by
// the host with the driver's cuTensorMapEncodeTiled). Compiled for sm_90a; elsewhere their bodies trap and the host never launches them.
#define FOLD_TMA12(NT) \
    extern "C" __global__ void __launch_bounds__((glib::Tma12<NT, 2>::THREADS), 1) fold_tma12_##NT(const __grid_constant__ CUtensorMap xmap, const uint8_t* data, \
        const uint32_t* exc, const int32_t* exc_base, uint32_t hb4, int xmax, int64_t O, int64_t K, int64_t M, const __nv_bfloat16* bias, __nv_bfloat16* Y, float* parts, int* done) { \
        glib::mma12_tma_body<NT, 2>(xmap, glib::Nib{data, exc, exc_base, hb4, xmax}, O, K, M, bias, Y, parts, done); \
    }
FOLD_TMA12(16)
FOLD_TMA12(32)
FOLD_TMA12(64)
FOLD_TMA12(96)
FOLD_TMA12(112)
FOLD_TMA12(128)
#define FOLD_WGP12(NT) \
    extern "C" __global__ void __launch_bounds__((glib::Wgp12<NT, 2>::THREADS), 1) fold_wgp12_##NT(const __grid_constant__ CUtensorMap xmap, const uint8_t* data, \
        const uint32_t* exc, const int32_t* exc_base, uint32_t hb4, int xmax, int64_t O, int64_t K, int64_t M, int64_t D, int ncs, const __nv_bfloat16* bias, \
        __nv_bfloat16* Y, float* parts, int* done) { \
        glib::mma12_wgp_body<NT, 2>(xmap, glib::Nib{data, exc, exc_base, hb4, xmax}, O, K, M, D, ncs, bias, Y, parts, done); \
    }
FOLD_WGP12(192)
FOLD_WGP12(256)
#define FOLD_TMAT(NT) \
    extern "C" __global__ void __launch_bounds__((glib::TmaT<NT, 2, 1>::THREADS), 1) fold_tmat_##NT(const __grid_constant__ CUtensorMap xmap, const uint8_t* data, \
        const uint8_t* blocks, const int32_t* block_base, uint32_t s1, uint32_t s2, uint32_t s3, int64_t O, int64_t K, int64_t M, const __nv_bfloat16* bias, \
        __nv_bfloat16* Y, float* parts, int* done, int bb) { \
        glib::tiered_tma_body<NT, 2, 1, true>(xmap, glib::Tiered{data, blocks, block_base, glib::Tiers{s1, s2, s3}}, O, K, M, bias, Y, parts, done, bb); \
    }
FOLD_TMAT(16)
FOLD_TMAT(32)
FOLD_TMAT(64)
#define FOLD_MOE_WG(NT, ACT) \
    extern "C" __global__ void __launch_bounds__((glib::Wm12<NT, ACT>::THREADS), 1) fold_moe_wg_##NT##_##ACT(const __grid_constant__ CUtensorMap xmap, const uint8_t* data, \
        const uint32_t* exc, const int32_t* exc_base, uint32_t hb4, int xmax, int64_t O, int64_t K, int E, const int* plan, const int4* tiles, int RU, \
        const __nv_bfloat16* bias, const void* wts, int wf32, __nv_bfloat16* Y, float* Y32) { \
        glib::mma12_moe_wg_body<NT, ACT>(xmap, glib::Nib{data, exc, exc_base, hb4, xmax}, O, K, E, plan, tiles, RU, bias, wts, wf32, Y, Y32); \
    }
FOLD_MOE_WG(16, 0)
FOLD_MOE_WG(16, 1)
FOLD_MOE_WG(32, 0)
FOLD_MOE_WG(32, 1)
FOLD_MOE_WG(64, 0)
FOLD_MOE_WG(64, 1)
FOLD_MOE_WG(128, 0)
FOLD_MOE_WG(128, 1)
extern "C" __global__ void __launch_bounds__(1024) fold_moe_wg_tiles(const int* plan, int E, int NT, int4* tiles) {
    glib::moe_wg_tiles_body(plan, E, NT, tiles);
}
extern "C" __global__ void fold_moe_wg_gather(const uint4* x, const int* plan, int E, int64_t k, int64_t K8, uint4* out) {
    glib::moe_wg_gather_body(x, plan, E, k, K8, out);
}
