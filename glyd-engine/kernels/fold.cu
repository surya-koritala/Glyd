// R17 (#238, docs/design/one-kernel-build.md): the GPU library's kernels in the engine's own fatbin, compiled from the library's own source
// (gpu/glyd_gpu_kernels.h, in a namespace of its own here), as extern "C" kernels the driver finds by name. Their launches are Rust
// (src/fold.rs), as the library's host code makes them, and `glyd-engine foldcheck` holds each to the library's bit for bit.
#include <type_traits>

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

// glyd_gpu_mma_moe, _mma12_moe and _mma_moe_placed (mma_moe_run) at a decode step's and a few rows' sizes: mma_moe_body for MT tiles of 16 pairs (1, 2, 4),
// the activation (0 none, 1 SiLU, 2 GELU) and the step loop's variant the library takes by default on each GPU class (0; 2: the m-tiles with pairs alone, an H100's
// and an A100's; 8 and 10: two steps loaded ahead, an HBM GPU's one-token products), and mma_moe_rows_body (no gate, a warp a row block over the whole K: MT 2
// and 4) and moe_sum_body (the weighted sum of a token's choices). The prompt kernel (many pairs an expert), the wgmma route and the variants a tuning variable asks
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
