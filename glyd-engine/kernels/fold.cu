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
