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
