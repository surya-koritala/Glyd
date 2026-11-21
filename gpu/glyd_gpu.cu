// Glyd weights on the GPU: a bf16 tensor held as its sign-and-mantissa
// bytes (8 bits a weight, as they are: noise) and its exponents coded by
// a per-tensor Huffman code (about 2.6 bits a weight where 8 are spent).
//
// Layout. The weights, flat, are cut into tiles of `tw` weights (a
// multiple of 128; for a matrix, whole rows: a tile is T rows of K). A
// tile is 32 lanes; in it, lane l holds the quads (4 neighbouring
// weights) l, l + 32, l + 64, ... so that at every step the 32 threads of
// a warp take 128 neighbouring weights: 128 sign-and-mantissa bytes read,
// 256 bytes of bf16 written, or 128 products for a row. Each lane's
// exponents are one bit stream (LSB first), the streams back to back; a
// lane's stream starts at its bit offset (`offs`, 32 bits).
//
// The host side is a C API (glyd_gpu_*, at the end; glyd_gpu.h declares it);
// PyTorch's JIT build (glyd_gpu.py) adds a pybind module over it,
// build_lib.sh builds it alone into a library for the glyd package's
// glyd/gpu/_lib.py and for C, C++, Rust or any language with a C FFI.
#ifdef TORCH_EXTENSION_NAME
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <map>
#endif
#if defined(__HIPCC__)
#include "hip.h"  // AMD GPUs through HIP (build_lib_hip.sh): the CUDA names below as HIP's
#else
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#endif
#include <stdint.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <deque>
#include <map>
#include <type_traits>
#include "glyd_gpu.h"  // every definition of the C API below held to its declaration there
// (a lambda forced inline: MSVC's dialect, nvcc's on Windows, takes no attribute there, so the inliner decides)
#ifdef _MSC_VER
#define GLYD_LAMBDA_INLINE
#else
#define GLYD_LAMBDA_INLINE __attribute__((always_inline))
#endif

__device__ __forceinline__ uint32_t exponent_of(uint16_t v) { return (v >> 7) & 0xff; }
__device__ __forceinline__ uint32_t bf16_bits(uint32_t s, uint32_t e) { return ((s & 0x80) << 8) | (e << 7) | (s & 0x7f); }

// A line of 128 bytes asked of L2 ahead (RDNA has no such prefetch: nothing there).
__device__ __forceinline__ void pf_l2(const void* p) {
#if defined(__HIP_PLATFORM_AMD__)
    (void)p;
#else
    asm volatile("prefetch.global.L2 [%0];\n" ::"l"(p));
#endif
}

// The dense format: each exponent in a prefix code read by counting
// leading zeros. Class c is c zeros, a one, then s_c bits: 2^s_c ranks
// from base_c (ranks by frequency, the rank's exponent from `sym`); the
// escape class's s_c = 8 bits are the exponent itself. Streams are MSB
// first. Lane l of a tile holds the groups of V neighbouring weights
// l, l + 32, ...: a warp takes 32 V neighbouring weights a step. A lane's
// stream starts at its bit offset (`offs`, 32 bits); a warp stages its
// tile's streams in shared memory before it decodes.

// Pass 1: the bits each lane's stream takes.
template <int V>
__global__ void lane_bits_kernel(const uint16_t* __restrict__ w, int64_t n, int64_t tw, const uint8_t* __restrict__ len, uint32_t* __restrict__ bits, int64_t lanes) {
    int64_t g = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (g >= lanes) return;
    int64_t base = (g >> 5) * tw + (g & 31) * V, end = min((g >> 5) * tw + tw, n);
    uint32_t b = 0;
    for (int64_t q = base; q < end; q += 32 * V)
        for (int k = 0; k < V && q + k < end; k++) b += len[exponent_of(w[q + k])];
    bits[g] = b;
}

// Pass 2: every lane writes its stream, MSB first, from its offset. A
// lane's first and last words may be shared with its neighbours (OR'd
// in); the words between are its own.
template <int V>
__global__ void write_kernel(const uint16_t* __restrict__ w, int64_t n, int64_t tw, const uint8_t* __restrict__ len, const uint32_t* __restrict__ code, const uint32_t* __restrict__ offs, uint32_t* __restrict__ out, int64_t lanes) {
    int64_t g = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (g >= lanes) return;
    int64_t base = (g >> 5) * tw + (g & 31) * V, end = min((g >> 5) * tw + tw, n);
    uint32_t pos = offs[g];
    uint32_t wi = pos >> 5;
    int nb = pos & 31;  // the first word's leading bits are the lane before's
    uint64_t acc = 0;
    bool first = true;
    for (int64_t q = base; q < end; q += 32 * V) {
        for (int k = 0; k < V && q + k < end; k++) {
            uint32_t e = exponent_of(w[q + k]);
            acc = (acc << len[e]) | code[e];
            nb += len[e];
            if (nb >= 32) {
                uint32_t word = (uint32_t)(acc >> (nb - 32));
                if (first) atomicOr(&out[wi], word); else out[wi] = word;
                first = false;
                nb -= 32;
                acc &= (1ull << nb) - 1;
                wi++;
            }
        }
    }
    if (nb > 0) atomicOr(&out[wi], (uint32_t)(acc << (32 - nb)));
}

// A lane's reader over its stream (MSB first): two words and a bit
// position; the next 32 bits are one funnel shift. The class table (lane c
// holds class c: code length | (base - 2^s) mod 32 << 8 | escape << 16)
// and the rank table (lane r holds rank r's exponent << 23, where a float
// keeps it) are read by shuffles. A code's value, read as the l bits it
// takes, is 2^s plus its offset in the class: its rank is that value
// plus the class's stored base, taken mod 32 by the shuffle.
struct Reader {
    const uint32_t* words;
    uint32_t hi, lo, nx, wi;  // nx: the word after lo, asked for a word ahead
    int bp;
    __device__ __forceinline__ Reader(const uint32_t* s, uint32_t pos) : words(s) {
        wi = pos >> 5;
        bp = pos & 31;
        hi = s[wi];
        lo = s[wi + 1];
        nx = s[wi + 2];
        wi += 3;
    }
    // The exponent in a float's place (<< 23).
    __device__ __forceinline__ uint32_t next(uint32_t cls, uint32_t sym) {
        uint32_t top = __funnelshift_l(lo, hi, bp);
        uint32_t info = __shfl_sync(0xffffffff, cls, __clz(top));
        int l = info & 31;
        uint32_t code = top >> (32 - l);
        uint32_t e = __shfl_sync(0xffffffff, sym, code + (info >> 8));
        bp += l;
        if (bp >= 32) {
            hi = lo;
            lo = nx;
            nx = words[wi++];  // used a word (a dozen codes) later
            bp -= 32;
        }
        return (info & 0x10000) ? (code & 0xff) << 23 : e;
    }
};

// A weight's float from its sign-and-mantissa byte k of s4 (sign bit 7,
// mantissa bits 0-6) and its exponent in place (<< 23): the sign's byte
// and the mantissa's byte moved by byte permutes.
__device__ __forceinline__ float weight_of(uint32_t sgn4, uint32_t man4, int k, uint32_t e23) {
    uint32_t sgn = __byte_perm(sgn4, 0, 0x0444 | (k << 12));  // byte k to byte 3
    uint32_t man = __byte_perm(man4, 0, 0x4044 | (k << 8));   // byte k to byte 2
    return __uint_as_float(sgn | man | e23);
}

// A warp's tile streams into shared memory, and its reader there.
__device__ __forceinline__ const uint32_t* stage(uint32_t* words, const uint32_t* __restrict__ stream, int64_t stream_words, const uint32_t* __restrict__ offs, int64_t tile, int tile_words, int lane) {
    int64_t w0 = offs[tile * 32] >> 5;
    int n = (int)min((int64_t)tile_words, stream_words - w0);
    // Eight loads in flight a lane before their stores: one at a time,
    // each store waited out its load's whole latency.
    for (int i = lane; i < n; i += 32 * 8) {
        uint32_t v[8];
#pragma unroll
        for (int k = 0; k < 8; k++) v[k] = i + 32 * k < n ? __ldg(stream + w0 + i + 32 * k) : 0;
#pragma unroll
        for (int k = 0; k < 8; k++)
            if (i + 32 * k < n) words[i + 32 * k] = v[k];
    }
    __syncwarp();
    return words - w0;  // indexed by absolute word
}

// Decode every tile into `out`, or the tiles listed in `tile_ids`, one
// after another from out[0]. One warp a tile.
template <int V>
__global__ void decode_kernel(const uint8_t* __restrict__ sm, const uint32_t* __restrict__ stream, int64_t stream_words, const uint32_t* __restrict__ offs, const uint32_t* __restrict__ tables, int64_t n, int64_t tw, int tile_words, const int64_t* __restrict__ tile_ids, int64_t tiles, uint16_t* __restrict__ out) {
    extern __shared__ uint32_t shared[];
    int64_t t = (blockIdx.x * (int64_t)blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (t >= tiles) return;
    int64_t tile = tile_ids ? tile_ids[t] : t;
    uint32_t cls = tables[lane], sym = tables[32 + lane];
    const uint32_t* words = tile_words ? stage(shared + (threadIdx.x >> 5) * tile_words, stream, stream_words, offs, tile, tile_words, lane) : stream;
    Reader r(words, offs[tile * 32 + lane]);
    int64_t start = tile * tw, end = min(start + tw, n);
    int64_t dst = tile_ids ? t * tw - start : 0;
    for (int64_t b = start; b < end; b += 32 * V) {
        int64_t q = b + lane * V;
        uint32_t e[V];
#pragma unroll
        for (int k = 0; k < V; k++) e[k] = r.next(cls, sym);  // every lane in step: the shuffles
        if (q + V <= end) {
#pragma unroll
            for (int k = 0; k < V; k += 4) {
                uint32_t s4 = *(const uint32_t*)(sm + q + k), sgn4 = s4 & 0x80808080, man4 = s4 & 0x7f7f7f7f;
                uint32_t o0 = __float_as_uint(weight_of(sgn4, man4, 0, e[k])) >> 16, o1 = __float_as_uint(weight_of(sgn4, man4, 1, e[k + 1])) >> 16;
                uint32_t o2 = __float_as_uint(weight_of(sgn4, man4, 2, e[k + 2])) >> 16, o3 = __float_as_uint(weight_of(sgn4, man4, 3, e[k + 3])) >> 16;
                *(uint2*)(out + q + k + dst) = make_uint2(o0 | (o1 << 16), o2 | (o3 << 16));
            }
        } else {
            for (int k = 0; k < V && q + k < end; k++) out[q + k + dst] = (uint16_t)bf16_bits(sm[q + k], e[k] >> 23);
        }
    }
}

// A row's dot product (warp-reduced) is done: written, or, where tiles
// split rows (SPLIT: rows longer than a tile), added to the row's fp32 sum;
// the last of the row's tiles to add writes it out and clears the sum.
template <bool SPLIT>
__device__ __forceinline__ void row_done(float acc, int64_t row, int64_t K, int64_t tw, int lane, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ y, float* sum, int* count) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, o);
    if (lane != 0) return;
    float b = bias ? __bfloat162float(bias[row]) : 0.f;
    if (!SPLIT) {
        y[row] = __float2bfloat16(acc + b);
        return;
    }
    int parts = (int)((row * K + K - 1) / tw - (row * K) / tw + 1);
    atomicAdd(sum + row, acc);
    __threadfence();
    if (atomicAdd(count + row, 1) == parts - 1) {
        y[row] = __float2bfloat16(atomicExch(sum + row, 0.f) + b);
        count[row] = 0;
    }
}

// y = W x (+ bias) for a matrix [O, K] (K a multiple of 32 V): one warp a
// tile, its rows' dot products in fp32, reduced across the warp as each row
// ends. Tiles are whole rows, or (SPLIT) flat pieces of long rows. The
// weights are read packed and never written out.
template <int V, bool SPLIT>
__global__ void gemv_kernel(const uint8_t* __restrict__ sm, const uint32_t* __restrict__ stream, int64_t stream_words, const uint32_t* __restrict__ offs, const uint32_t* __restrict__ tables, int64_t O, int64_t K, int64_t tw, int tile_words, const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ y, int64_t tiles, float* sum, int* count) {
    extern __shared__ uint32_t shared[];
    int64_t tile = (blockIdx.x * (int64_t)blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (tile >= tiles) return;
    uint32_t cls = tables[lane], sym = tables[32 + lane];
    // Staged in shared memory where the tile's streams fit the budget
    // (tile_words > 0); read from global memory otherwise.
    const uint32_t* words = tile_words ? stage(shared + (threadIdx.x >> 5) * tile_words, stream, stream_words, offs, tile, tile_words, lane) : stream;
    Reader r(words, offs[tile * 32 + lane]);
    int64_t start = tile * tw, end = min(start + tw, O * K);
    float acc = 0.f;
    int64_t row = start / K, row_end = (row + 1) * K;
    for (int64_t b = start; b < end; b += 32 * V) {
        if (b >= row_end) {
            row_done<SPLIT>(acc, row, K, tw, lane, bias, y, sum, count);
            acc = 0.f;
            row++;
            row_end += K;
        }
        int64_t q = b + lane * V, k = q - row * K;
        // The step's sign-and-mantissa bytes and inputs are asked for
        // before its codes are decoded, so they arrive meanwhile.
        uint32_t s4v[V / 4];
        uint2 xvv[V / 4];
#pragma unroll
        for (int i = 0; i < V; i += 4) {
            s4v[i / 4] = __ldg((const unsigned int*)(sm + q + i));
            xvv[i / 4] = *(const uint2*)(x + k + i);
        }
        uint32_t e[V];
#pragma unroll
        for (int i = 0; i < V; i++) e[i] = r.next(cls, sym);
#pragma unroll
        for (int i = 0; i < V; i += 4) {
            uint32_t s4 = s4v[i / 4], sgn4 = s4 & 0x80808080, man4 = s4 & 0x7f7f7f7f;
            uint2 xv = xvv[i / 4];
            acc = fmaf(weight_of(sgn4, man4, 0, e[i]), __uint_as_float(xv.x << 16), acc);
            acc = fmaf(weight_of(sgn4, man4, 1, e[i + 1]), __uint_as_float(xv.x & 0xffff0000), acc);
            acc = fmaf(weight_of(sgn4, man4, 2, e[i + 2]), __uint_as_float(xv.y << 16), acc);
            acc = fmaf(weight_of(sgn4, man4, 3, e[i + 3]), __uint_as_float(xv.y & 0xffff0000), acc);
        }
    }
    row_done<SPLIT>(acc, row, K, tw, lane, bias, y, sum, count);
}

// The fast format: each weight's exponent as a 3-bit code into the
// tensor's 7 most common exponents (`top`, one byte each, in a 64-bit
// argument), code 7 an escape to the exponent itself (`exc`, in weight
// order; `exc_base[r * segs + s]` the first escape of segment s of row r,
// a segment SEG weights of a row). Codes in three bit planes a 32 weights
// (`planes[3g + b]`, bit i the code bit b of weight 32g + i). A warp takes
// 128 weights a step, 4 a lane.
constexpr int SEG = 1024;

struct Quad {
    uint32_t e[4];
};

__device__ __forceinline__ Quad fast_quad(const uint32_t* __restrict__ planes, const uint8_t* __restrict__ exc, int64_t q, uint64_t top, int64_t& esc, int lane) {
    int64_t gword = (q >> 5) * 3;
    int sh = q & 31;
    uint32_t p0 = planes[gword] >> sh, p1 = planes[gword + 1] >> sh, p2 = planes[gword + 2] >> sh;
    uint32_t c[4];
    uint32_t m = 0;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        c[k] = ((p0 >> k) & 1) | (((p1 >> k) & 1) << 1) | (((p2 >> k) & 1) << 2);
        m |= (uint32_t)(c[k] == 7) << k;
    }
    Quad r;
#pragma unroll
    for (int k = 0; k < 4; k++) r.e[k] = (uint32_t)(top >> (8 * c[k])) & 0xff;
    // Escapes: the warp counts them in lane order.
    uint32_t any = __ballot_sync(0xffffffff, m != 0);
    if (any) {
        uint32_t n = __popc(m), before = n;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            uint32_t t = __shfl_up_sync(0xffffffff, before, o);
            if (lane >= o) before += t;
        }
        uint32_t total = __shfl_sync(0xffffffff, before, 31);
        int64_t at = esc + before - n;
#pragma unroll
        for (int k = 0; k < 4; k++)
            if (m >> k & 1) r.e[k] = exc[at++];
        esc += total;
    }
    return r;
}

// A block a row group: rows of `wpr` warps each, a row's warps taking its
// segments in turn, their sums added in shared memory (wpr 1: every warp
// a row of its own, no barrier).
template <bool SPLIT>
__global__ void fast_gemv_kernel(const uint8_t* __restrict__ sm, const uint32_t* __restrict__ planes, const uint8_t* __restrict__ exc, const int32_t* __restrict__ exc_base, uint64_t top, int64_t O, int64_t K, int wpr, const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ y) {
    __shared__ float part[32];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int64_t row = (int64_t)blockIdx.x * (blockDim.x / 32 / wpr) + warp / wpr;
    int w = warp % wpr;
    if (!SPLIT && row >= O) return;
    int64_t segs = (K + SEG - 1) / SEG;
    float acc = 0.f;
    if (row < O) for (int64_t s = w; s < segs; s += wpr) {
      int64_t esc = exc_base[row * segs + s];
      int64_t kend = min((s + 1) * SEG, K);
      for (int64_t k = s * SEG + lane * 4; k < kend; k += 128) {
        int64_t q = row * K + k;
        uint32_t s4 = *(const uint32_t*)(sm + q);
        uint2 xv = *(const uint2*)(x + k);
        Quad d = fast_quad(planes, exc, q, top, esc, lane);
        __nv_bfloat162 x01 = *(__nv_bfloat162*)&xv.x, x23 = *(__nv_bfloat162*)&xv.y;
        float w0 = __uint_as_float(bf16_bits(s4 & 0xff, d.e[0]) << 16), w1 = __uint_as_float(bf16_bits((s4 >> 8) & 0xff, d.e[1]) << 16);
        float w2 = __uint_as_float(bf16_bits((s4 >> 16) & 0xff, d.e[2]) << 16), w3 = __uint_as_float(bf16_bits(s4 >> 24, d.e[3]) << 16);
        acc += w0 * __low2float(x01) + w1 * __high2float(x01) + w2 * __low2float(x23) + w3 * __high2float(x23);
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, o);
    if (!SPLIT) {
        if (lane == 0) y[row] = __float2bfloat16(acc + (bias ? __bfloat162float(bias[row]) : 0.f));
        return;
    }
    if (lane == 0) part[warp] = acc;
    __syncthreads();
    if (w == 0 && lane == 0 && row < O) {
        float t = 0.f;
        for (int i = 0; i < wpr; i++) t += part[warp + i];
        y[row] = __float2bfloat16(t + (bias ? __bfloat162float(bias[row]) : 0.f));
    }
}

// fast_gemv_kernel with V weights a lane a step (V = 8 or 16; K a
// multiple of 32 V): 16-byte loads of sign-and-mantissa bytes and
// inputs, V code bits of each plane from one word.
template <int V, bool SPLIT>
__global__ void fast_gemv_wide_kernel(const uint8_t* __restrict__ sm, const uint32_t* __restrict__ planes, const uint8_t* __restrict__ exc, const int32_t* __restrict__ exc_base, uint64_t top, int64_t O, int64_t K, int wpr, const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ y) {
    __shared__ float part[32];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int64_t row = (int64_t)blockIdx.x * (blockDim.x / 32 / wpr) + warp / wpr;
    int w = warp % wpr;
    if (!SPLIT && row >= O) return;
    int64_t segs = (K + SEG - 1) / SEG;
    float acc = 0.f;
    if (row < O) for (int64_t s = w; s < segs; s += wpr) {
        int64_t esc = exc_base[row * segs + s];
        int64_t kend = min((s + 1) * SEG, K);
#pragma unroll 2
        for (int64_t k = s * SEG + lane * V; k < kend; k += 32 * V) {
            int64_t q = row * K + k;
            uint32_t smw[V / 4];
            uint32_t xw[V / 2];
            if (V == 16) {
                uint4 a = *(const uint4*)(sm + q);
                smw[0] = a.x; smw[1] = a.y; smw[2] = a.z; smw[3] = a.w;
                uint4 b = *(const uint4*)(x + k), c = *(const uint4*)(x + k + 8);
                xw[0] = b.x; xw[1] = b.y; xw[2] = b.z; xw[3] = b.w; xw[4] = c.x; xw[5] = c.y; xw[6] = c.z; xw[7] = c.w;
            } else {
                uint2 a = *(const uint2*)(sm + q);
                smw[0] = a.x; smw[1] = a.y;
                uint4 b = *(const uint4*)(x + k);
                xw[0] = b.x; xw[1] = b.y; xw[2] = b.z; xw[3] = b.w;
            }
            int64_t gword = (q >> 5) * 3;
            int sh = q & 31;
            uint32_t p0 = planes[gword] >> sh, p1 = planes[gword + 1] >> sh, p2 = planes[gword + 2] >> sh;
            uint32_t m = p0 & p1 & p2 & ((1u << V) - 1);  // code 7: every plane bit set
            uint32_t e[V];
#pragma unroll
            for (int i = 0; i < V; i++) {
                uint32_t c = ((p0 >> i) & 1) | (((p1 >> i) & 1) << 1) | (((p2 >> i) & 1) << 2);
                e[i] = (uint32_t)(top >> (8 * c)) & 0xff;
            }
            if (__ballot_sync(0xffffffff, m != 0)) {
                uint32_t n = __popc(m), before = n;
#pragma unroll
                for (int o = 1; o < 32; o <<= 1) {
                    uint32_t t = __shfl_up_sync(0xffffffff, before, o);
                    if (lane >= o) before += t;
                }
                uint32_t total = __shfl_sync(0xffffffff, before, 31);
                int64_t at = esc + before - n;
#pragma unroll
                for (int i = 0; i < V; i++)
                    if (m >> i & 1) e[i] = exc[at++];
                esc += total;
            }
#pragma unroll
            for (int i = 0; i < V; i += 2) {
                uint32_t s0 = (smw[i / 4] >> (8 * (i % 4))) & 0xff, s1 = (smw[i / 4] >> (8 * (i % 4) + 8)) & 0xff;
                __nv_bfloat162 xx = *(__nv_bfloat162*)&xw[i / 2];
                acc += __uint_as_float(bf16_bits(s0, e[i]) << 16) * __low2float(xx) + __uint_as_float(bf16_bits(s1, e[i + 1]) << 16) * __high2float(xx);
            }
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, o);
    if (!SPLIT) {
        if (lane == 0) y[row] = __float2bfloat16(acc + (bias ? __bfloat162float(bias[row]) : 0.f));
        return;
    }
    if (lane == 0) part[warp] = acc;
    __syncthreads();
    if (w == 0 && lane == 0 && row < O) {
        float t = 0.f;
        for (int i = 0; i < wpr; i++) t += part[warp + i];
        y[row] = __float2bfloat16(t + (bias ? __bfloat162float(bias[row]) : 0.f));
    }
}

// Y = X W^T (+ bias) for several tokens (X [M, K], Y [M, O]) from the fast
// format, on the tensor cores. A block takes 64 rows of W, 64 tokens and a
// range of K (split across blocks where W has few rows; the parts are
// added in fp32). Each 64-weight step of its W rows is decoded into shared
// memory as bf16, once for all its tokens, while the next step's packed
// weights and inputs are already on their way; the weights never reach
// global memory as bf16.
constexpr int GM_BM = 64, GM_BO = 64, GM_BK = 64, GM_LD = GM_BK + 8;

__global__ void __launch_bounds__(128) fast_gemm_kernel(const uint8_t* __restrict__ sm, const uint32_t* __restrict__ planes, const uint8_t* __restrict__ exc, const int32_t* __restrict__ exc_base, uint64_t top, int64_t O, int64_t K, int64_t M, int64_t kchunk, const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ Y, float* __restrict__ Y32) {
#if defined(__HIP_PLATFORM_AMD__)
    __builtin_trap();  // (the fast format's product of many tokens is CUDA's alone: glyd_gpu_fast_gemm answers not supported on AMD)
#else
    using namespace nvcuda;
    __shared__ __align__(16) __nv_bfloat16 xs[GM_BM][GM_LD];
    __shared__ __align__(16) __nv_bfloat16 ws[GM_BO][GM_LD];
    __shared__ float stage[4][16 * 16];
    int64_t m0 = (int64_t)blockIdx.y * GM_BM, o0 = (int64_t)blockIdx.x * GM_BO;
    int64_t kb = (int64_t)blockIdx.z * kchunk, ke = min(K, kb + kchunk);
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int wm = warp >> 1, wo = warp & 1;  // a warp: 32 tokens x 32 outputs
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 2; j++) wmma::fill_fragment(acc[i][j], 0.f);
    // This thread's W row, and its 32-weight group of each step (h: which
    // half of the step); x: its token row and 32 columns of each step.
    int r = tid >> 1, h = tid & 1;
    int64_t row = o0 + r, gm = m0 + r;
    bool wrow = row < O, xrow = gm < M;
    int64_t segs = (K + SEG - 1) / SEG;
    int64_t esc = 0;
    // The next step's loads, in registers.
    uint32_t p0, p1, p2;
    uint4 sa, sb, xa, xb, xc, xd;
    auto fetch = [&](int64_t k0) {
        if (wrow) {
            int64_t q = row * K + k0 + h * 32;
            int64_t g = (q >> 5) * 3;
            p0 = planes[g]; p1 = planes[g + 1]; p2 = planes[g + 2];
            sa = *(const uint4*)(sm + q); sb = *(const uint4*)(sm + q + 16);
        }
        uint4 z = make_uint4(0, 0, 0, 0);
        const __nv_bfloat16* xp = X + gm * K + k0 + h * 32;
        xa = xrow ? *(const uint4*)xp : z; xb = xrow ? *(const uint4*)(xp + 8) : z;
        xc = xrow ? *(const uint4*)(xp + 16) : z; xd = xrow ? *(const uint4*)(xp + 24) : z;
    };
    fetch(kb);
    for (int64_t k0 = kb; k0 < ke; k0 += GM_BK) {
        // This step into shared memory: the inputs as they are, the weights decoded.
        *(uint4*)&xs[r][h * 32] = xa; *(uint4*)&xs[r][h * 32 + 8] = xb;
        *(uint4*)&xs[r][h * 32 + 16] = xc; *(uint4*)&xs[r][h * 32 + 24] = xd;
        if (wrow) {
            if (k0 % SEG == 0 || k0 == kb) {
                // The segment's first escape, then those of its steps before k0.
                int64_t s0 = k0 / SEG * SEG;
                esc = exc_base[row * segs + k0 / SEG];
                for (int64_t k = s0; k < k0; k += 32) {
                    int64_t g = ((row * K + k) >> 5) * 3;
                    esc += __popc(planes[g] & planes[g + 1] & planes[g + 2]);
                }
            }
            uint32_t escm = p0 & p1 & p2;
            uint32_t mine = __popc(escm), other = __shfl_xor_sync(0xffffffff, mine, 1);
            int64_t at = esc + (h ? other : 0);
            uint32_t sw[8] = {sa.x, sa.y, sa.z, sa.w, sb.x, sb.y, sb.z, sb.w};
            uint32_t out[16];
#pragma unroll
            for (int i = 0; i < 32; i++) {
                uint32_t c = ((p0 >> i) & 1) | (((p1 >> i) & 1) << 1) | (((p2 >> i) & 1) << 2);
                uint32_t e = c == 7 ? exc[at++] : (uint32_t)(top >> (8 * c)) & 0xff;
                uint32_t v = bf16_bits((sw[i >> 2] >> (8 * (i & 3))) & 0xff, e);
                if (i & 1) out[i >> 1] |= v << 16; else out[i >> 1] = v;
            }
#pragma unroll
            for (int q = 0; q < 4; q++) *(uint4*)&ws[r][h * 32 + q * 8] = make_uint4(out[4 * q], out[4 * q + 1], out[4 * q + 2], out[4 * q + 3]);
            esc += mine + other;
        } else {
            uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll
            for (int q = 0; q < 4; q++) *(uint4*)&ws[r][h * 32 + q * 8] = z;
        }
        __syncthreads();
        if (k0 + GM_BK < ke) fetch(k0 + GM_BK);
#pragma unroll
        for (int kk = 0; kk < GM_BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b[2];
#pragma unroll
            for (int i = 0; i < 2; i++) wmma::load_matrix_sync(a[i], &xs[wm * 32 + i * 16][kk], GM_LD);
#pragma unroll
            for (int j = 0; j < 2; j++) wmma::load_matrix_sync(b[j], &ws[wo * 32 + j * 16][kk], GM_LD);
#pragma unroll
            for (int i = 0; i < 2; i++)
#pragma unroll
                for (int j = 0; j < 2; j++) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::store_matrix_sync(stage[warp], acc[i][j], 16, wmma::mem_row_major);
            __syncwarp();
            for (int e = lane; e < 256; e += 32) {
                int64_t tm = m0 + wm * 32 + i * 16 + e / 16, to = o0 + wo * 32 + j * 16 + e % 16;
                if (tm < M && to < O) {
                    if (Y32) Y32[(blockIdx.z * M + tm) * O + to] = stage[warp][e];
                    else Y[tm * O + to] = __float2bfloat16(stage[warp][e] + (bias ? __bfloat162float(bias[to]) : 0.f));
                }
            }
            __syncwarp();
        }
#endif
}

// Split-K's parts (one [M, O] slice each) added in a fixed order, so the
// result is the same every run, to bf16 with the bias.
__global__ void finish_kernel(const float* __restrict__ y32, int64_t split, const __nv_bfloat16* __restrict__ bias, int64_t M, int64_t O, __nv_bfloat16* __restrict__ y) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= M * O) return;
    float t = 0.f;
    for (int64_t z = 0; z < split; z++) t += y32[z * M * O + i];
    y[i] = __float2bfloat16(t + (bias ? __bfloat162float(bias[i % O]) : 0.f));
}

// Y = X W^T for a few tokens (MT of them, X [MT, K]) from the fast format:
// the one-token product's layout (a warp a row, 16 weights a lane a step)
// with each decoded weight multiplied into MT sums on the CUDA cores. A
// block stages its K segment of X in shared memory once, then its warps
// stream W rows through it; K is split into segments where X would not fit
// (their parts added in a fixed order).
template <int MT>
__global__ void __launch_bounds__(256) fast_bgemv_kernel(const uint8_t* __restrict__ sm, const uint32_t* __restrict__ planes, const uint8_t* __restrict__ exc, const int32_t* __restrict__ exc_base, uint64_t top, int64_t O, int64_t K, int64_t kseg, int64_t rows_per_block, const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ Y, float* __restrict__ Y32) {
    extern __shared__ __align__(16) __nv_bfloat16 xs[];  // [MT][kseg]
    int64_t kb = (int64_t)blockIdx.y * kseg, ke = min(K, kb + kseg), kn = ke - kb;
    for (int64_t i = threadIdx.x * 8; i < MT * kn; i += blockDim.x * 8) {
        int64_t m = i / kn, k = i % kn;
        *(uint4*)&xs[m * kseg + k] = *(const uint4*)&X[m * K + kb + k];
    }
    __syncthreads();
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int64_t segs = (K + SEG - 1) / SEG;
    int64_t r0 = (int64_t)blockIdx.x * rows_per_block, r1 = min(O, r0 + rows_per_block);
    for (int64_t row = r0 + warp; row < r1; row += blockDim.x / 32) {
        float acc[MT];
#pragma unroll
        for (int m = 0; m < MT; m++) acc[m] = 0.f;
        for (int64_t s = kb / SEG; s * SEG < ke; s++) {
            int64_t esc = exc_base[row * segs + s];
            int64_t kend = min((s + 1) * SEG, ke);
            for (int64_t k = s * SEG + lane * 16; k < kend; k += 512) {
                int64_t q = row * K + k;
                uint4 a = *(const uint4*)(sm + q);
                uint32_t smw[4] = {a.x, a.y, a.z, a.w};
                int64_t gword = (q >> 5) * 3;
                int sh = q & 31;
                uint32_t p0 = planes[gword] >> sh, p1 = planes[gword + 1] >> sh, p2 = planes[gword + 2] >> sh;
                uint32_t mk = p0 & p1 & p2 & 0xffff;
                uint32_t e[16];
#pragma unroll
                for (int i = 0; i < 16; i++) {
                    uint32_t c = ((p0 >> i) & 1) | (((p1 >> i) & 1) << 1) | (((p2 >> i) & 1) << 2);
                    e[i] = (uint32_t)(top >> (8 * c)) & 0xff;
                }
                if (__ballot_sync(0xffffffff, mk != 0)) {
                    uint32_t n = __popc(mk), before = n;
#pragma unroll
                    for (int o = 1; o < 32; o <<= 1) {
                        uint32_t t = __shfl_up_sync(0xffffffff, before, o);
                        if (lane >= o) before += t;
                    }
                    uint32_t total = __shfl_sync(0xffffffff, before, 31);
                    int64_t at = esc + before - n;
#pragma unroll
                    for (int i = 0; i < 16; i++)
                        if (mk >> i & 1) e[i] = exc[at++];
                    esc += total;
                }
                const __nv_bfloat16* xk = xs + (k - kb);
#pragma unroll
                for (int i = 0; i < 16; i += 8) {
                    float w[8];
#pragma unroll
                    for (int t = 0; t < 8; t++) w[t] = __uint_as_float(bf16_bits((smw[(i + t) >> 2] >> (8 * ((i + t) & 3))) & 0xff, e[i + t]) << 16);
#pragma unroll
                    for (int m = 0; m < MT; m++) {
                        uint4 xv = *(const uint4*)(xk + m * kseg + i);
                        acc[m] = fmaf(w[0], __uint_as_float(xv.x << 16), acc[m]);
                        acc[m] = fmaf(w[1], __uint_as_float(xv.x & 0xffff0000), acc[m]);
                        acc[m] = fmaf(w[2], __uint_as_float(xv.y << 16), acc[m]);
                        acc[m] = fmaf(w[3], __uint_as_float(xv.y & 0xffff0000), acc[m]);
                        acc[m] = fmaf(w[4], __uint_as_float(xv.z << 16), acc[m]);
                        acc[m] = fmaf(w[5], __uint_as_float(xv.z & 0xffff0000), acc[m]);
                        acc[m] = fmaf(w[6], __uint_as_float(xv.w << 16), acc[m]);
                        acc[m] = fmaf(w[7], __uint_as_float(xv.w & 0xffff0000), acc[m]);
                    }
                }
            }
        }
#pragma unroll
        for (int m = 0; m < MT; m++) {
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) acc[m] += __shfl_xor_sync(0xffffffff, acc[m], o);
        }
        if (lane == 0) {
#pragma unroll
            for (int m = 0; m < MT; m++) {
                if (Y32) Y32[((int64_t)blockIdx.y * MT + m) * O + row] = acc[m];
                else Y[m * O + row] = __float2bfloat16(acc[m] + (bias ? __bfloat162float(bias[row]) : 0.f));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The mma layout: a tensor's weights in the order the tensor cores'
// mma.sync.m16n8k16 takes its B operand, so a thread's 32 weights of a step
// arrive as a few loads and are decoded in registers straight into its B
// fragments (Marlin's arrangement, for this code). For W [O, K] (O a
// multiple of 64, K of 16): warp step (rb, ks) covers rows rb*64.. and
// columns ks*16..; its lane l = 4g + t holds, as its group's weight 4n + j
// (n-tile n 0-7, j 0-3),
//   W[rb*64 + 8n + g][ks*16 + 8(j >> 1) + 2t + (j & 1)].
// A weight's exponent is coded in tiers of 2-bit digits: tier 1's digits
// 0-2 are the tensor's 3 commonest exponents, digit 3 goes on to tier 2
// (the next 3), whose digit 3 goes on to tier 3 (the next 4: all four digits
// are exponents); an exponent past the tenth is a literal. A warp step's
// fixed 1280 bytes: its lanes' tier-1 digits (weight i at bits
// 2(i mod 16) of word i / 16; [2 words][32 lanes]), then their 32 bytes each
// as two halves (every lane's first 16, then every lane's last 16): weight
// i's 7 mantissa bits above the sign of its pair's other weight (i ^ 1), so
// a pair's two bytes and two exponents, permuted into one word and rotated
// by a bit, are its two bf16s. Its block (bytes block_base[step] to
// block_base[step + 1], the entries' bit 31 aside): from its start, the
// tier-3 digits of its weights of ranks 6-9 (2 bits each, in the order of their
// tier-1 escapes), then from the next byte the list of its literals, the
// weights of ranks 10 on: an entry of 18 bits, low bits first, one after the
// other from the list's first byte, holds the place of its weight among the
// step's tier-1 escapes (10 bits) and its exponent (8 bits), the entries in
// the order of their places; at its end, the tier-2 digits of its tier-1
// escapes in lane order (a literal's: 0), as words of 16 (word c: escapes
// 16c to 16c + 15, the c-th word back from the end; unused digits 0). All
// in order: lane by lane, a lane's weights in order. Bit 31 of block_base[step]
// is the packer's flag: set where the step's block is more than a decode of
// a short form takes (more than 512 tier-1 escapes or 32 literals, its
// tier-3 digits or its last literal's entry not within the first 124 bytes
// of the 128 from the word its block starts in); the decode here takes every
// step, and masks it. The tiers' words: tiers[0]'s top byte is the layout's
// number, 2 (the layout of v0.27 and before has 0xFF, three exponents in
// tier 3 and its digit 3 an escape to the exponent's byte, which follows in
// the block: TieredV1, unpacked as before and given to no product).

constexpr int64_t STEP_BYTES = 1280;  // a warp step's fixed part
constexpr int S2_BYTES = 1040;        // a warp's shared scratch for a step's tier-2 exponents
constexpr uint32_t FULL = 0xffffffffu;

// The tiers' symbols: tier k's three exponents in bytes 0-2; tier 3's fourth in byte 3 of s3, the layout's number in byte 3 of s1.
constexpr uint32_t TIERED_TAG = 2, TIERED_V1_TAG = 0xFF;
struct Tiers {
    uint32_t s1, s2, s3;
};

// Eight 2-bit digits (bits 0-15) as the nibbles of two __byte_perm
// selectors: digits 0-3 in bits 0-15, 4-7 in bits 16-31.
__device__ __forceinline__ uint32_t nib8(uint32_t x) {
    uint32_t t = ((x & 0xFFFFu) | (x << 8)) & 0x00FF00FFu;
    t = (t | (t << 4)) & 0x0F0F0F0Fu;
    return (t | (t << 2)) & 0x33333333u;
}

// A group's selector (bits 0-15; z: its digits' nibbles, fn: their escapes,
// 1 at bit 0 of each 3; bits past 15 do not reach them): a digit 0-2 picks
// its symbol, an escape byte 4 + the escapes before it (the stream's bytes,
// in order).
__device__ __forceinline__ uint32_t selector(uint32_t z, uint32_t fn) { return z ^ ((z ^ (0x4444u + fn * 0x1110u)) & (fn * 0xFu)); }

// A stream of 16 bytes on by `bytes` (0-4).
__device__ __forceinline__ void advance(uint32_t s[4], uint32_t bytes) {
    uint32_t sh = 8 * bytes;
    s[0] = __funnelshift_rc(s[0], s[1], sh);
    s[1] = __funnelshift_rc(s[1], s[2], sh);
    s[2] = __funnelshift_rc(s[2], s[3], sh);
    s[3] = __funnelshift_rc(s[3], 0u, sh);
}

// The groups' table (256 words in shared memory): for four digits (an
// 8-bit field), their selector (bits 0-15) and escapes (bits 16-18).
__device__ __forceinline__ void fill_groups(uint32_t* tab) {
    for (int f = threadIdx.x; f < 256; f += blockDim.x) {
        uint32_t z = nib8((uint32_t)f), fn = z & (z >> 1) & 0x1111u;
        tab[f] = selector(z, fn) | (uint32_t)__popc(fn) << 16;
    }
}

// Four digits (bits 0-7 of field) as a word of bytes: digits 0-2 the
// symbols' bytes, escapes the stream's next bytes (taken from it).
__device__ __forceinline__ uint32_t group(uint32_t field, uint32_t sym, uint32_t s[4], const uint32_t* tab) {
    uint32_t e = tab[field & 0xFF], o = __byte_perm(sym, s[0], e);
    advance(s, e >> 16);
    return o;
}

__device__ __forceinline__ uint32_t escapes(uint32_t d) { return d & (d >> 1) & 0x55555555u; }
__device__ __forceinline__ uint32_t first_digits(uint32_t n) { return n >= 16 ? FULL : (1u << (2 * n)) - 1; }

// 4 bytes from any address (blocks are padded past their ends).
__device__ __forceinline__ uint32_t load4(const uint8_t* __restrict__ p) {
    uintptr_t a = (uintptr_t)p;
    const uint32_t* q = (const uint32_t*)(a & ~(uintptr_t)3);
    return __funnelshift_r(__ldg(q), __ldg(q + 1), (uint32_t)(a & 3) * 8);
}

// 32 bits of a step's block from bit `pos` of its first byte's word on:
// from the window (each lane's word of the 128 bytes from there; pos < 992,
// every lane calls it), or from memory.
__device__ __forceinline__ uint32_t win_bits(uint32_t win, uint32_t pos) {
    uint32_t k = pos >> 5;
    return __funnelshift_r(__shfl_sync(FULL, win, k), __shfl_sync(FULL, win, k + 1), pos);
}
__device__ __forceinline__ uint32_t mem_bits(const uint8_t* __restrict__ blocks, int64_t blk, uint32_t pos) {
    const uint32_t* q = (const uint32_t*)(blocks + (blk & ~(int64_t)3)) + (pos >> 5);
    return __funnelshift_r(__ldg(q), __ldg(q + 1), pos);
}

// A count across the warp: this lane's start (the counts of the lanes
// before it) and the whole.
__device__ __forceinline__ uint32_t scan(uint32_t n, int lane, uint32_t& total) {
    uint32_t before = n;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        uint32_t v = __shfl_up_sync(FULL, before, o);
        before += lane >= o ? v : 0;
    }
    total = __shfl_sync(FULL, before, 31);
    return before - n;
}

// A step's loads for a lane: tier-1 digits, bytes, and its block: where it
// starts and ends, its word from the start (win) and back from the end
// (tail; tail1: the end's word).
struct Step {
    uint32_t pd[2], sw[8], win, tail, tail1;
    int64_t blk, end;
};

__device__ __forceinline__ void load_step(Step& st, const uint8_t* __restrict__ data, const uint8_t* __restrict__ blocks, const int32_t* __restrict__ block_base, int64_t step, int lane) {
    const uint32_t* cp = (const uint32_t*)(data + step * STEP_BYTES) + lane;
    st.pd[0] = __ldg(cp);
    st.pd[1] = __ldg(cp + 32);
    const uint4* sp = (const uint4*)(data + step * STEP_BYTES + 256) + lane;
    uint4 x0 = __ldg(sp), x1 = __ldg(sp + 32);
    st.sw[0] = x0.x; st.sw[1] = x0.y; st.sw[2] = x0.z; st.sw[3] = x0.w;
    st.sw[4] = x1.x; st.sw[5] = x1.y; st.sw[6] = x1.z; st.sw[7] = x1.w;
    st.blk = (uint32_t)__ldg(block_base + step) & 0x7FFFFFFFu;  // (bit 31: the step's flag)
    st.end = (uint32_t)__ldg(block_base + step + 1) & 0x7FFFFFFFu;
    const uint32_t* bw = (const uint32_t*)blocks;
    st.win = __ldg(bw + (st.blk >> 2) + lane);
    st.tail = __ldg(bw + (st.end >> 2) - 1 - lane);
    st.tail1 = __ldg(bw + (st.end >> 2));
}

// A word of 16 tier-2 digits (d) as 16 bytes into the warp's scratch at
// word 4c: their escapes' from its tier-3 digits (d3), theirs from the bytes
// (x). g3: the warp's most tier-3 digits a word, in fours.
__device__ __forceinline__ void tier23_v1(uint32_t d, uint32_t d3, uint32_t x[4], Tiers ts, uint32_t g3, uint32_t* s2, int c, const uint32_t* tab) {
    uint32_t y[4] = {0u, 0u, 0u, 0u}, o[4];
#pragma unroll
    for (int q = 0; q < 4; q++)
        if (q < (int)g3) y[q] = group(d3 >> (8 * q), ts.s3, x, tab);
#pragma unroll
    for (int q = 0; q < 4; q++) o[q] = group(d >> (8 * q), ts.s2, y, tab);
    ((uint4*)s2)[c] = make_uint4(o[0], o[1], o[2], o[3]);
}

// A group's 32 exponents, 4 a word (ew). Every lane of the warp calls it
// for the same step: lane c decodes the tier-2 digits of escapes 16c to
// 16c + 15 (and 16(c + 32) on, past 512) with what they escape to, into
// the warp's scratch (s2, S2_BYTES), where each lane reads its escapes'.
__device__ __forceinline__ void exponents_v1(const Step& st, Tiers ts, const uint8_t* __restrict__ blocks, int lane, uint32_t* s2, const uint32_t* tab, uint32_t ew[8]) {
    uint32_t n1 = __popc(escapes(st.pd[0])) + __popc(escapes(st.pd[1]));
    uint32_t T1 = __reduce_add_sync(FULL, n1), words = (T1 + 15) >> 4;
    // Word c of the tier-2 digits: bytes [end - 4(c + 1), end - 4c).
    uint32_t up = __shfl_up_sync(FULL, st.tail, 1);
    uint32_t da = (uint32_t)lane < words ? __funnelshift_r(st.tail, lane ? up : st.tail1, 8 * (uint32_t)(st.end & 3)) : 0u, db = 0;
    if (T1 > 512 && (uint32_t)lane + 32 < words) db = load4(blocks + st.end - 4 * (33 + lane));
    uint32_t na = __popc(escapes(da)), nb = __popc(escapes(db)), tot;
    uint32_t pre = scan(n1 | na << 11 | nb << 21, lane, tot);
    uint32_t b1 = pre & 0x7FF, ta = (tot >> 11) & 0x3FF, T2 = ta + (tot >> 21);
    // Their escapes' tier-3 digits, and those's escapes (bytes), all from bit o of the window.
    uint32_t o = 8 * (uint32_t)(st.blk & 3), pa = o + 2 * ((pre >> 11) & 0x3FF), pb = o + 2 * (ta + (pre >> 21));
    bool near = o + 2 * T2 + 32 < 992;
    uint32_t d3a = near ? win_bits(st.win, pa) : mem_bits(blocks, st.blk, pa), d3b = 0;
    if (T1 > 512) d3b = near ? win_bits(st.win, pb) : mem_bits(blocks, st.blk, pb);
    uint32_t n3a = __popc(escapes(d3a) & first_digits(na)), n3b = __popc(escapes(d3b) & first_digits(nb)), t3;
    uint32_t pre3 = scan(n3a | n3b << 16, lane, t3);
    uint32_t r0 = o + 8 * ((2 * T2 + 7) >> 3), T3 = (t3 & 0xFFFF) + (t3 >> 16);
    uint32_t ba = r0 + 8 * (pre3 & 0xFFFF), bb = r0 + 8 * ((t3 & 0xFFFF) + (pre3 >> 16));
    uint32_t g3 = (__reduce_max_sync(FULL, max(na, nb)) + 3) >> 2;
    bool many = __any_sync(FULL, max(n3a, n3b) > 8);
    near = r0 + 8 * T3 + 128 < 992;
    uint32_t x[4];
    auto bytes = [&](uint32_t b) {
#pragma unroll
        for (int k = 0; k < 4; k++) x[k] = k < 2 || many ? (near ? win_bits(st.win, b + 32 * k) : mem_bits(blocks, st.blk, b + 32 * k)) : 0u;
    };
    bytes(ba);
    tier23_v1(da, d3a, x, ts, g3, s2, lane, tab);
    if (T1 > 512) {
        bytes(bb);
        tier23_v1(db, d3b, x, ts, g3, s2, lane + 32, tab);
    }
    __syncwarp();
    // Tier 1, its escapes' from the scratch (this lane's: bytes b1 on).
    uint32_t c = b1;
#pragma unroll
    for (int q = 0; q < 8; q++) {
        uint32_t e = tab[(st.pd[q >> 2] >> (8 * (q & 3))) & 0xFF];
        ew[q] = __byte_perm(ts.s1, __funnelshift_r(s2[c >> 2], s2[(c >> 2) + 1], c << 3), e);
        c += e >> 16;
    }
    __syncwarp();  // the scratch is read
}

// Pairs: [byte, exponent, byte, exponent] rotated right by one bit.
__device__ __forceinline__ void pairs(const uint32_t sw[8], const uint32_t ew[8], uint32_t R[16]) {
#pragma unroll
    for (int p = 0; p < 16; p++) {
        uint32_t y = __byte_perm(sw[p >> 1], ew[p >> 1], (p & 1) ? 0x7362 : 0x5140);
        R[p] = __funnelshift_r(y, y, 1);
    }
}

// A step's 32 weights a lane as the B fragments of its 8 n-tiles (R[2n], R[2n + 1]), in the layout of v0.27 and before.
__device__ __forceinline__ void decode_step_v1(const Step& st, Tiers ts, const uint8_t* __restrict__ blocks, int lane, uint32_t* s2, const uint32_t* tab, uint32_t R[16]) {
    uint32_t ew[8];
    exponents_v1(st, ts, blocks, lane, s2, tab, ew);
    pairs(st.sw, ew, R);
}

// The entries of a list of `bytes` bytes of 18 bits each: 8 bytes / 18, exact for the list a packer makes (bytes = 18 n / 8 up; at most 1024:
// a step has no more weights, and a block cut short does not run a lane's loop for ever).
__device__ __forceinline__ uint32_t list_entries(uint32_t bytes) { return min(__umulhi(4u * bytes, 0x1C71C71Du), 1024u); }

// A step's 32 exponents a lane, 4 a word (ew), by the general decode (any step, whatever its flag). Every lane of the warp
// calls it for the same step: lane c decodes the tier-2 digits of escapes 16c to 16c + 15 (and 16(c + 32) on, past 512) with
// what they go on to (tier 3's four exponents by a digit: the group table's escape, which takes a byte of a stream, takes
// here a byte of S3, every one tier 3's fourth), into the warp's scratch (s2, S2_BYTES); then each literal of the step's
// list sets its escape's byte there; then each lane reads its escapes' from it.
__device__ __forceinline__ void exponents(const Step& st, Tiers ts, const uint8_t* __restrict__ blocks, int lane, uint32_t* s2, const uint32_t* tab, uint32_t ew[8]) {
    uint32_t n1 = __popc(escapes(st.pd[0])) + __popc(escapes(st.pd[1]));
    uint32_t T1 = __reduce_add_sync(FULL, n1), words = (T1 + 15) >> 4;
    uint32_t e1[8];  // tier 1's group lookups, issued ahead of the scans (they need the digits alone)
#pragma unroll
    for (int q = 0; q < 8; q++) e1[q] = tab[(st.pd[q >> 2] >> (8 * (q & 3))) & 0xFF];
    uint32_t b1, T2;
    const uint32_t S3 = 0x01010101u * (ts.s3 >> 24);
    const uint32_t o = 8 * (uint32_t)(st.blk & 3);  // the bit the block starts at in its first word
    if (T1 <= 512) {
        // Word c of the tier-2 digits: bytes [end - 4(c + 1), end - 4c).
        uint32_t up = __shfl_up_sync(FULL, st.tail, 1);
        uint32_t da = (uint32_t)lane < words ? __funnelshift_r(st.tail, lane ? up : st.tail1, 8 * (uint32_t)(st.end & 3)) : 0u;
        uint32_t na = __popc(escapes(da)), tot;
        uint32_t pre = scan(n1 | na << 11, lane, tot);
        b1 = pre & 0x7FF;
        T2 = tot >> 11;
        // Their tier-3 digits, from bit pa of the window (the 128 bytes from the block's first word) or from memory.
        uint32_t pa = o + 2 * (pre >> 11);
        bool near = o + 2 * T2 + 32 < 992;
        uint32_t d3 = near ? win_bits(st.win, pa) : mem_bits(blocks, st.blk, pa);
        uint32_t g3 = (__reduce_max_sync(FULL, na) + 3) >> 2;
        uint32_t y[4] = {0u, 0u, 0u, 0u}, q4[4];
#pragma unroll
        for (int q = 0; q < 4; q++)
            if (q < (int)g3) y[q] = __byte_perm(ts.s3, S3, tab[(d3 >> (8 * q)) & 0xFF]);
#pragma unroll
        for (int q = 0; q < 4; q++) q4[q] = group(da >> (8 * q), ts.s2, y, tab);
        ((uint4*)s2)[lane] = make_uint4(q4[0], q4[1], q4[2], q4[3]);
    } else {
        // More than 32 words: chunks of 32, each its words' tier-3 digits after the chunks before it.
        uint32_t tot1;
        b1 = scan(n1, lane, tot1);
        uint32_t carry = 0;
        for (uint32_t ch = 0; ch * 32 < words; ch++) {
            uint32_t c = ch * 32 + lane;
            uint32_t da = c < words ? load4(blocks + st.end - 4 * (c + 1)) : 0u;
            uint32_t na = __popc(escapes(da)), tot;
            uint32_t pre = scan(na, lane, tot);
            uint32_t d3 = mem_bits(blocks, st.blk, o + 2 * (carry + pre));  // (the window holds 128 bytes: tier-3 digits can run past it)
            uint32_t g3 = (__reduce_max_sync(FULL, na) + 3) >> 2;
            uint32_t y[4] = {0u, 0u, 0u, 0u}, q4[4];
#pragma unroll
            for (int q = 0; q < 4; q++)
                if (q < (int)g3) y[q] = __byte_perm(ts.s3, S3, tab[(d3 >> (8 * q)) & 0xFF]);
#pragma unroll
            for (int q = 0; q < 4; q++) q4[q] = group(da >> (8 * q), ts.s2, y, tab);
            ((uint4*)s2)[c] = make_uint4(q4[0], q4[1], q4[2], q4[3]);
            carry += tot;
        }
        T2 = carry;
    }
    // The list: entry i (lane i, i + 32, ...) of 18 bits from the list's first byte sets the byte of the escape it names.
    uint32_t r0 = (2 * T2 + 7) >> 3, nl = list_entries((uint32_t)(st.end - st.blk) - r0 - 4 * words);
    __syncwarp();
    if (nl != 0 && nl <= 32 && o + 8 * r0 + 18 * (nl - 1) < 992) {
        uint32_t x = win_bits(st.win, o + 8 * r0 + 18 * (uint32_t)lane);  // (every lane's: the shuffles are the warp's; a lane past the list's end has no entry)
        if ((uint32_t)lane < nl) ((uint8_t*)s2)[x & 0x3FF] = (uint8_t)(x >> 10);
    } else {
        for (uint32_t i = lane; i < nl; i += 32) {
            uint32_t x = mem_bits(blocks, st.blk, o + 8 * r0 + 18 * i);
            ((uint8_t*)s2)[x & 0x3FF] = (uint8_t)(x >> 10);
        }
    }
    __syncwarp();
    // Tier 1, its escapes' from the scratch (this lane's: bytes b1 on).
    uint32_t c = b1;
#pragma unroll
    for (int q = 0; q < 8; q++) {
        uint32_t e = e1[q];
        ew[q] = __byte_perm(ts.s1, __funnelshift_r(s2[c >> 2], s2[(c >> 2) + 1], c << 3), e);
        c += e >> 16;
    }
    __syncwarp();  // the scratch is read
}

// A step's 32 weights a lane as the B fragments of its 8 n-tiles (R[2n], R[2n + 1]).
__device__ __forceinline__ void decode_step(const Step& st, Tiers ts, const uint8_t* __restrict__ blocks, int lane, uint32_t* s2, const uint32_t* tab, uint32_t R[16]) {
    uint32_t ew[8];
    exponents(st, ts, blocks, lane, s2, tab, ew);
    pairs(st.sw, ew, R);
}

// The layout of v0.27 and before as a policy (its decode only: unpacking it, to pack the matrix again).
struct TieredV1 {
    const uint8_t* data;
    const uint8_t* blocks;
    const int32_t* block_base;
    Tiers ts;
    typedef Step St;
    static constexpr bool kTable = true;  // decode needs the groups' table and a warp's scratch
    __device__ __forceinline__ void load(St& st, int64_t step, int lane) const { load_step(st, data, blocks, block_base, step, lane); }
    __device__ __forceinline__ void decode(const St& st, int lane, uint32_t* s2, const uint32_t* tab, uint32_t R[16]) const { decode_step_v1(st, ts, blocks, lane, s2, tab, R); }
    // Ahead: a step's fixed part (1280 bytes, 10 lines of 128: lane i takes line i) to L2; its block (a few bytes) is read when it is decoded.
    __device__ __forceinline__ void prefetch(int64_t step, int lane) const {
        if (lane < (int)(STEP_BYTES / 128)) pf_l2(data + step * STEP_BYTES + lane * 128);
    }
    __device__ __forceinline__ void prefetch_exc(int64_t, int64_t, int) const {}
};

// The kernels below are written once for both layouts, each a policy:
// its step's loads (St, load) and their decode into a lane's B fragments.
struct Tiered {
    const uint8_t* data;
    const uint8_t* blocks;
    const int32_t* block_base;
    Tiers ts;
    typedef Step St;
    static constexpr bool kTable = true;  // decode needs the groups' table and a warp's scratch
    __device__ __forceinline__ void load(St& st, int64_t step, int lane) const { load_step(st, data, blocks, block_base, step, lane); }
    __device__ __forceinline__ void decode(const St& st, int lane, uint32_t* s2, const uint32_t* tab, uint32_t R[16]) const { decode_step(st, ts, blocks, lane, s2, tab, R); }
    // Ahead: a step's fixed part (1280 bytes, 10 lines of 128: lane i takes line i) to L2; its block (a few bytes) is read when it is decoded.
    __device__ __forceinline__ void prefetch(int64_t step, int lane) const {
        if (lane < (int)(STEP_BYTES / 128)) pf_l2(data + step * STEP_BYTES + lane * 128);
    }
    __device__ __forceinline__ void prefetch_exc(int64_t, int64_t, int) const {}
};

// The compact tiered layout (COMPACT_TAG in tiers[0]'s top byte): a matrix whose finite weights are bf16 with three
// fraction bits, the four below zero (Candidate A's values), in the tiered layout with each sign-and-mantissa byte
// kept as a nibble (its bits 5-7 and 0; the omitted bits zero): a step's 256 bytes of digits as they are, then its
// 512 nibble bytes, two bytes each in lane and half order (768 bytes a step). Exponents, blocks and words are the
// tiered layout's; a step decodes as Tiered's once its bytes are restored in registers.
constexpr uint32_t COMPACT_TAG = 3;
constexpr int64_t COMPACT_STEP_BYTES = 768;
// Four nibbles restore four sign-and-mantissa bytes, the omitted bits zero.
__host__ __device__ __forceinline__ uint32_t compact_expand4(uint32_t value) {
    uint32_t x = value & 0xffffu;
    x = (x | (x << 8)) & 0x00ff00ffu;
    x = (x | (x << 4)) & 0x0f0f0f0fu;
    return (x & 0x01010101u) | ((x & 0x0e0e0e0eu) << 4);
}
struct Compact {
    const uint8_t* data;
    const uint8_t* blocks;
    const int32_t* block_base;
    Tiers ts;
    typedef Step St;
    static constexpr bool kTable = true;
    __device__ __forceinline__ void load(St& st, int64_t step, int lane) const {
        const uint8_t* p = data + step * COMPACT_STEP_BYTES;
        const uint32_t* digits = (const uint32_t*)p + lane;
        st.pd[0] = __ldg(digits);
        st.pd[1] = __ldg(digits + 32);
        const uint2* sig = (const uint2*)(p + 256) + lane;
        uint2 lo = __ldg(sig), hi = __ldg(sig + 32);
        st.sw[0] = compact_expand4(lo.x); st.sw[1] = compact_expand4(lo.x >> 16);
        st.sw[2] = compact_expand4(lo.y); st.sw[3] = compact_expand4(lo.y >> 16);
        st.sw[4] = compact_expand4(hi.x); st.sw[5] = compact_expand4(hi.x >> 16);
        st.sw[6] = compact_expand4(hi.y); st.sw[7] = compact_expand4(hi.y >> 16);
        st.blk = (uint32_t)__ldg(block_base + step) & 0x7fffffffu;
        st.end = (uint32_t)__ldg(block_base + step + 1) & 0x7fffffffu;
        const uint32_t* bw = (const uint32_t*)blocks;
        st.win = __ldg(bw + (st.blk >> 2) + lane);
        st.tail = __ldg(bw + (st.end >> 2) - 1 - lane);
        st.tail1 = __ldg(bw + (st.end >> 2));
    }
    __device__ __forceinline__ void decode(const St& st, int lane, uint32_t* s2, const uint32_t* tab, uint32_t R[16]) const { decode_step(st, ts, blocks, lane, s2, tab, R); }
    __device__ __forceinline__ void prefetch(int64_t step, int lane) const {
        if (lane < (int)(COMPACT_STEP_BYTES / 128)) pf_l2(data + step * COMPACT_STEP_BYTES + lane * 128);
    }
    __device__ __forceinline__ void prefetch_exc(int64_t, int64_t, int) const {}
};

// The six-bit block layout (INT6_TAG in tiers[0]'s top byte): each run of 32 values of a row of W (columns 32 kb on) as six-bit codes u,
// 1 to 63, and one bf16 scale s for the run; the weight is round_bf16((u - 32) s) (the product of a 6-bit integer and an 8-bit significand,
// rounded once: the packer's own reference). A warp step's 768 bytes have the compact layout's shape: 256 bytes of the codes' high two bits
// (lane l's two words at d * 32 + l; its pair r, weights 2r and 2r + 1 of the step's 32 a lane, at bits 2(r mod 8) and 16 + 2(r mod 8) of
// word r / 8), then 512 bytes of their low four bits (lane l's four words, two halves of 8 bytes at l and l + 32; pair r at bits 4(r mod 4)
// and 16 + 4(r mod 4) of word r / 4). The scales (`blocks`): 128 bytes a pair of steps, those of step >> 1 (row block rb, columns 32 kb on:
// a row block's steps are rb * KS + ks with KS even), its 64 rows' scales as row 8n + g at place 8g + n, so that a lane's eight rows (g its
// group) are one 16-byte load. `block_base` is not read. A pair's two bf16s are an OR (0x4300 | u: 128 + u), a subtraction of 160 (exact)
// and a product by the scale, two bf16x2 instructions a pair.
constexpr uint32_t INT6_TAG = 4;
constexpr int64_t INT6_STEP_BYTES = 768;
struct Int6 {
    const uint8_t* data;
    const uint8_t* scales;
    struct St {
        uint32_t hi[2], lo[4], sc[4];
    };
    static constexpr bool kTable = true;  // (the tiered kernels' shape: their table and a warp's scratch are not read)
    __device__ __forceinline__ void load(St& st, int64_t step, int lane) const {
        const uint8_t* p = data + step * INT6_STEP_BYTES;
        const uint32_t* h = (const uint32_t*)p + lane;
        st.hi[0] = __ldg(h);
        st.hi[1] = __ldg(h + 32);
        const uint2* l = (const uint2*)(p + 256) + lane;
        uint2 a = __ldg(l), b = __ldg(l + 32);
        st.lo[0] = a.x; st.lo[1] = a.y; st.lo[2] = b.x; st.lo[3] = b.y;
        uint4 c = __ldg((const uint4*)(scales + (step >> 1) * 128) + (lane >> 2));
        st.sc[0] = c.x; st.sc[1] = c.y; st.sc[2] = c.z; st.sc[3] = c.w;
    }
    // (a step through shared memory, mma_steps_staged: its bytes and its pair's scales, in 16-byte chunks; read back with the offsets load uses)
    static constexpr int kStage = INT6_STEP_BYTES + 128;
    __device__ __forceinline__ const uint8_t* chunk(int64_t step, int c) const { return c < 48 ? data + step * INT6_STEP_BYTES + 16 * c : scales + (step >> 1) * 128 + 16 * (c - 48); }
    __device__ __forceinline__ void load_s(St& st, const uint8_t* p, int lane) const {
        const uint32_t* h = (const uint32_t*)p + lane;
        st.hi[0] = h[0];
        st.hi[1] = h[32];
        const uint2* l = (const uint2*)(p + 256) + lane;
        uint2 a = l[0], b = l[32];
        st.lo[0] = a.x; st.lo[1] = a.y; st.lo[2] = b.x; st.lo[3] = b.y;
        uint4 c = ((const uint4*)(p + INT6_STEP_BYTES))[lane >> 2];
        st.sc[0] = c.x; st.sc[1] = c.y; st.sc[2] = c.z; st.sc[3] = c.w;
    }
    __device__ __forceinline__ void decode(const St& st, int, uint32_t*, const uint32_t*, uint32_t R[16]) const {
        const uint32_t off = 0x43204320u;  // (160, 160)
#pragma unroll
        for (int r = 0; r < 16; r++) {
            uint32_t x = 0x43004300u | ((st.lo[r >> 2] >> (4 * (r & 3))) & 0x000F000Fu) | (((st.hi[r >> 3] >> (2 * (r & 7))) & 0x00030003u) << 4);
            uint32_t sw = __byte_perm(st.sc[r >> 2], 0, ((r >> 1) & 1) ? 0x3232 : 0x1010);
            __nv_bfloat162 v = __hmul2(__hsub2(*reinterpret_cast<const __nv_bfloat162*>(&x), *reinterpret_cast<const __nv_bfloat162*>(&off)), *reinterpret_cast<const __nv_bfloat162*>(&sw));
            R[r] = *reinterpret_cast<uint32_t*>(&v);
        }
    }
    __device__ __forceinline__ void prefetch(int64_t step, int lane) const {
        if (lane < (int)(INT6_STEP_BYTES / 128)) pf_l2(data + step * INT6_STEP_BYTES + lane * 128);
        else if (lane == (int)(INT6_STEP_BYTES / 128) && !(step & 1)) pf_l2(scales + (step >> 1) * 128);
    }
    __device__ __forceinline__ void prefetch_exc(int64_t, int64_t, int) const {}
};

// The five-bit block layout (INT5_TAG in tiers[0]'s top byte): each run of 32 values of a row of W (columns 32 kb on) as five-bit codes u,
// 1 to 31, and one bf16 scale s for the run; the weight is round_bf16((u - 16) s) (the product of a 5-bit integer and an 8-bit significand,
// rounded once: the packer's own reference). A warp step's 640 bytes have the six-bit layout's shape: 128 bytes of the codes' high bit
// (lane l's word at l; its pair r, weights 2r and 2r + 1 of the step's 32 a lane, at bits r and 16 + r), then 512 bytes of their low four
// bits (lane l's four words, two halves of 8 bytes at l and l + 32; pair r at bits 4(r mod 4) and 16 + 4(r mod 4) of word r / 4). The scales
// (`blocks`): 128 bytes a pair of steps, those of step >> 1 (row block rb, columns 32 kb on: a row block's steps are rb * KS + ks with KS
// even), its 64 rows' scales as row 8n + g at place 8g + n, so that a lane's eight rows (g its group) are one 16-byte load. `block_base` is
// not read. A pair's two bf16s are an OR (0x4300 | u: 128 + u), a subtraction of 144 (exact) and a product by the scale, two bf16x2
// instructions a pair.
constexpr uint32_t INT5_TAG = 5;
constexpr int64_t INT5_STEP_BYTES = 640;
struct Int5 {
    const uint8_t* data;
    const uint8_t* scales;
    struct St {
        uint32_t hi, lo[4], sc[4];
    };
    static constexpr bool kTable = true;  // (the tiered kernels' shape: their table and a warp's scratch are not read)
    __device__ __forceinline__ void load(St& st, int64_t step, int lane) const {
        const uint8_t* p = data + step * INT5_STEP_BYTES;
        const uint32_t* h = (const uint32_t*)p + lane;
        st.hi = __ldg(h);
        const uint2* l = (const uint2*)(p + 128) + lane;
        uint2 a = __ldg(l), b = __ldg(l + 32);
        st.lo[0] = a.x; st.lo[1] = a.y; st.lo[2] = b.x; st.lo[3] = b.y;
        uint4 c = __ldg((const uint4*)(scales + (step >> 1) * 128) + (lane >> 2));
        st.sc[0] = c.x; st.sc[1] = c.y; st.sc[2] = c.z; st.sc[3] = c.w;
    }
    static constexpr int kStage = INT5_STEP_BYTES + 128;  // (as Int6's)
    __device__ __forceinline__ const uint8_t* chunk(int64_t step, int c) const { return c < 40 ? data + step * INT5_STEP_BYTES + 16 * c : scales + (step >> 1) * 128 + 16 * (c - 40); }
    __device__ __forceinline__ void load_s(St& st, const uint8_t* p, int lane) const {
        st.hi = ((const uint32_t*)p)[lane];
        const uint2* l = (const uint2*)(p + 128) + lane;
        uint2 a = l[0], b = l[32];
        st.lo[0] = a.x; st.lo[1] = a.y; st.lo[2] = b.x; st.lo[3] = b.y;
        uint4 c = ((const uint4*)(p + INT5_STEP_BYTES))[lane >> 2];
        st.sc[0] = c.x; st.sc[1] = c.y; st.sc[2] = c.z; st.sc[3] = c.w;
    }
    __device__ __forceinline__ void decode(const St& st, int, uint32_t*, const uint32_t*, uint32_t R[16]) const {
        const uint32_t off = 0x43104310u;  // (144, 144)
#pragma unroll
        for (int r = 0; r < 16; r++) {
            uint32_t x = 0x43004300u | ((st.lo[r >> 2] >> (4 * (r & 3))) & 0x000F000Fu) | (((st.hi >> r) & 0x00010001u) << 4);
            uint32_t sw = __byte_perm(st.sc[r >> 2], 0, ((r >> 1) & 1) ? 0x3232 : 0x1010);
            __nv_bfloat162 v = __hmul2(__hsub2(*reinterpret_cast<const __nv_bfloat162*>(&x), *reinterpret_cast<const __nv_bfloat162*>(&off)), *reinterpret_cast<const __nv_bfloat162*>(&sw));
            R[r] = *reinterpret_cast<uint32_t*>(&v);
        }
    }
    __device__ __forceinline__ void prefetch(int64_t step, int lane) const {
        if (lane < (int)(INT5_STEP_BYTES / 128)) pf_l2(data + step * INT5_STEP_BYTES + lane * 128);
        else if (lane == (int)(INT5_STEP_BYTES / 128) && !(step & 1)) pf_l2(scales + (step >> 1) * 128);
    }
    __device__ __forceinline__ void prefetch_exc(int64_t, int64_t, int) const {}
};

// The four-bit block layout (INT4_TAG in tiers[0]'s top byte): each run of 32 values of a row of W (columns 32 kb on) as four-bit codes u,
// 1 to 15, and one bf16 scale s for the run; the weight is round_bf16((u - 8) s) (the product of a 4-bit integer and an 8-bit significand,
// rounded once: the packer's own reference). A warp step's 512 bytes are the five- and six-bit layouts' low four bits alone (lane l's four
// words, two halves of 8 bytes at l and l + 32; pair r, weights 2r and 2r + 1 of the step's 32 a lane, at bits 4(r mod 4) and 16 + 4(r mod 4)
// of word r / 4). The scales (`blocks`): 128 bytes a pair of steps, those of step >> 1 (row block rb, columns 32 kb on: a row block's steps are
// rb * KS + ks with KS even), its 64 rows' scales as row 8n + g at place 8g + n, so that a lane's eight rows (g its group) are one 16-byte
// load. `block_base` is not read. A pair's two bf16s are an OR (0x4300 | u: 128 + u), a subtraction of 136 (exact) and a product by the
// scale, two bf16x2 instructions a pair.
constexpr uint32_t INT4_TAG = 6;
constexpr int64_t INT4_STEP_BYTES = 512;
struct Int4 {
    const uint8_t* data;
    const uint8_t* scales;
    struct St {
        uint32_t lo[4], sc[4];
    };
    static constexpr bool kTable = true;  // (the tiered kernels' shape: their table and a warp's scratch are not read)
    __device__ __forceinline__ void load(St& st, int64_t step, int lane) const {
        const uint8_t* p = data + step * INT4_STEP_BYTES;
        const uint2* l = (const uint2*)p + lane;
        uint2 a = __ldg(l), b = __ldg(l + 32);
        st.lo[0] = a.x; st.lo[1] = a.y; st.lo[2] = b.x; st.lo[3] = b.y;
        uint4 c = __ldg((const uint4*)(scales + (step >> 1) * 128) + (lane >> 2));
        st.sc[0] = c.x; st.sc[1] = c.y; st.sc[2] = c.z; st.sc[3] = c.w;
    }
    static constexpr int kStage = INT4_STEP_BYTES + 128;  // (as Int6's)
    __device__ __forceinline__ const uint8_t* chunk(int64_t step, int c) const { return c < 32 ? data + step * INT4_STEP_BYTES + 16 * c : scales + (step >> 1) * 128 + 16 * (c - 32); }
    __device__ __forceinline__ void load_s(St& st, const uint8_t* p, int lane) const {
        const uint2* l = (const uint2*)p + lane;
        uint2 a = l[0], b = l[32];
        st.lo[0] = a.x; st.lo[1] = a.y; st.lo[2] = b.x; st.lo[3] = b.y;
        uint4 c = ((const uint4*)(p + INT4_STEP_BYTES))[lane >> 2];
        st.sc[0] = c.x; st.sc[1] = c.y; st.sc[2] = c.z; st.sc[3] = c.w;
    }
    __device__ __forceinline__ void decode(const St& st, int, uint32_t*, const uint32_t*, uint32_t R[16]) const {
        const uint32_t off = 0x43084308u;  // (136, 136)
#pragma unroll
        for (int r = 0; r < 16; r++) {
            uint32_t x = 0x43004300u | ((st.lo[r >> 2] >> (4 * (r & 3))) & 0x000F000Fu);
            uint32_t sw = __byte_perm(st.sc[r >> 2], 0, ((r >> 1) & 1) ? 0x3232 : 0x1010);
            __nv_bfloat162 v = __hmul2(__hsub2(*reinterpret_cast<const __nv_bfloat162*>(&x), *reinterpret_cast<const __nv_bfloat162*>(&off)), *reinterpret_cast<const __nv_bfloat162*>(&sw));
            R[r] = *reinterpret_cast<uint32_t*>(&v);
        }
    }
    __device__ __forceinline__ void prefetch(int64_t step, int lane) const {
        if (lane < (int)(INT4_STEP_BYTES / 128)) pf_l2(data + step * INT4_STEP_BYTES + lane * 128);
        else if (lane == (int)(INT4_STEP_BYTES / 128) && !(step & 1)) pf_l2(scales + (step >> 1) * 128);
    }
    __device__ __forceinline__ void prefetch_exc(int64_t, int64_t, int) const {}
};

// The exact four-bit block layout (INT4H_TAG in tiers[0]'s top byte): Int4's steps, codes and places, for checkpoints trained to four bits
// (quantization-aware), whose bf16 weights it holds bit for bit. A run's weight is round_bf16(k s), k = u - 8 (-8 to 7), with a scale of its own
// form: bit 15 the sign, bits 11-14 the exponent over the pack's base (tiers[1]: the least biased f32 exponent of its scales), bits 0-10 the top
// of the fraction (a 12-bit significand). The product is exact in fp32 (4 bits by 12) and rounded to bf16 once, as the packer's reference
// (glyd-gpu/src/pack.rs int4h_value). A pair: the codes as floats (an OR into 2^23 and a subtraction), two products, one rounding to bf16x2.
constexpr uint32_t INT4H_TAG = 13;
struct Int4H : Int4 {
    uint32_t base;
    __device__ __forceinline__ void decode(const St& st, int, uint32_t*, const uint32_t*, uint32_t R[16]) const {
#pragma unroll
        for (int r = 0; r < 16; r++) {
            const uint32_t h = (st.sc[r >> 2] >> (16 * ((r >> 1) & 1))) & 0xFFFFu;
            const float sc = __uint_as_float(((h & 0x8000u) << 16) | ((((h >> 11) & 0xFu) + base) << 23) | ((h & 0x7FFu) << 12));
            const uint32_t w = st.lo[r >> 2] >> (4 * (r & 3));
            const float a = __uint_as_float(0x4B000000u | (w & 0xFu)) - 8388616.f, b = __uint_as_float(0x4B000000u | ((w >> 16) & 0xFu)) - 8388616.f;
            __nv_bfloat162 v = __floats2bfloat162_rn(__fmul_rn(a, sc), __fmul_rn(b, sc));
            R[r] = *reinterpret_cast<uint32_t*>(&v);
        }
    }
};

// The exact FP4 block layout (FP4_TAG in tiers[0]'s top byte): an MX FP4 checkpoint's matrices (E2M1 values, an E8M0 scale a run of 32) held as released, 4.25 bits
// a weight. Int4's steps and places, each code an E2M1 value (bit 3 its sign, bits 0-2 its magnitude: 0, 0.5, 1, 1.5, 2, 3, 4, 6), and a run's scale one E8M0 byte e
// (2^(e - 127); the packer takes e of 2 to 128), 64 bytes for two steps, a row's at 8 g + n as Int4's: the weight is bf16(value 2^(e - 127)), exact (glyd-gpu/src/pack.rs
// fp4_value).
constexpr uint32_t FP4_TAG = 14;
// A lane's 16 pairs of a step from its four code words (pair r: bits 4 (r mod 4) and 16 + 4 (r mod 4) of lo[r / 4]) and its eight scale bytes (scale n = r / 2: byte n mod 4 of e[n / 4]), as two bf16
// each, exactly: a code's bits placed as a bf16's (its sign at 15, its two exponent bits at 7-8, its mantissa bit at 6) are its E2M1 value times 2^-126, the two smallest (0 and 0.5) as bf16 subnormals,
// so one product by 2^(e - 1) (e + 126 in the exponent: e at most 128) is the weight. Two masks, two shifts and the product a pair; the tables, the -64 s and the signs' XOR it replaced had been 6 integer
// instructions a pair, and the integer pipe issues at half the rate of the others (an A100 and an H100: the FP4 experts bound by it).
__device__ __forceinline__ void fp4_step(const uint32_t (&lo)[4], uint32_t e0, uint32_t e1, uint32_t R[16]) {
    uint32_t s[8];
    e0 += 0x7E7E7E7Eu;  // (e + 126 a byte: no carries, e at most 128)
    e1 += 0x7E7E7E7Eu;
#pragma unroll
    for (int n = 0; n < 8; n++) s[n] = __byte_perm(n < 4 ? e0 : e1, 0, 0x0404u | ((n & 3) << 4) | ((n & 3) << 12)) >> 1;  // (2^(e - 1) twice)
#pragma unroll
    for (int q = 0; q < 4; q++)
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int r = 4 * q + j;
            const uint32_t t = lo[q] >> (4 * j);
            const uint32_t x = ((t & 0x00070007u) << 6) | ((t & 0x00080008u) << 12);
            __nv_bfloat162 v = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&x), *reinterpret_cast<const __nv_bfloat162*>(&s[r >> 1]));
            R[r] = *reinterpret_cast<uint32_t*>(&v);
        }
}
struct Fp4 {
    const uint8_t* data;
    const uint8_t* scales;
    struct St {
        uint32_t lo[4], sc[2];
    };
    static constexpr bool kTable = true;  // (the tiered kernels' shape: their table and a warp's scratch are not read)
    __device__ __forceinline__ void load(St& st, int64_t step, int lane) const {
        const uint8_t* p = data + step * INT4_STEP_BYTES;
        const uint2* l = (const uint2*)p + lane;
        uint2 a = __ldg(l), b = __ldg(l + 32);
        st.lo[0] = a.x; st.lo[1] = a.y; st.lo[2] = b.x; st.lo[3] = b.y;
        uint2 c = __ldg((const uint2*)(scales + (step >> 1) * 64) + (lane >> 2));
        st.sc[0] = c.x; st.sc[1] = c.y;
    }
    static constexpr int kStage = INT4_STEP_BYTES + 64;  // (as Int6's: the codes, then the pair's 64 bytes of scales)
    __device__ __forceinline__ const uint8_t* chunk(int64_t step, int c) const { return c < 32 ? data + step * INT4_STEP_BYTES + 16 * c : scales + (step >> 1) * 64 + 16 * (c - 32); }
    __device__ __forceinline__ void load_s(St& st, const uint8_t* p, int lane) const {
        const uint2* l = (const uint2*)p + lane;
        uint2 a = l[0], b = l[32];
        st.lo[0] = a.x; st.lo[1] = a.y; st.lo[2] = b.x; st.lo[3] = b.y;
        uint2 c = ((const uint2*)(p + INT4_STEP_BYTES))[lane >> 2];
        st.sc[0] = c.x; st.sc[1] = c.y;
    }
    __device__ __forceinline__ void decode(const St& st, int, uint32_t*, const uint32_t*, uint32_t R[16]) const { fp4_step(st.lo, st.sc[0], st.sc[1], R); }
    __device__ __forceinline__ void prefetch(int64_t step, int lane) const {
        if (lane < (int)(INT4_STEP_BYTES / 128)) pf_l2(data + step * INT4_STEP_BYTES + lane * 128);
        else if (lane == (int)(INT4_STEP_BYTES / 128) && !(step & 1)) pf_l2(scales + (step >> 1) * 64);
    }
    __device__ __forceinline__ void prefetch_exc(int64_t, int64_t, int) const {}
};

// The exact FP4 layout with packed scales (FP4R_TAG): Fp4's codes and decode, each run's E8M0 scale its row's base plus two bits (an MX checkpoint's scales differ little
// along a row: the packer takes a matrix whose rows each span at most 4 values, else Fp4). 4.07 bits a weight at 4,096 columns (Fp4's 4.25). A row block's scales: its
// 64 rows' bases (a row's at 8 g + n, as Int4's scales), then a pair of steps' 64 offsets in 16 bytes (a row's 2 bits at bit 2 n of the u16 at 2 g).
constexpr uint32_t FP4R_TAG = 15;
struct Fp4R {
    const uint8_t* data;
    const uint8_t* scales;
    uint32_t ks, mg, sh;  // steps a row block (K / 16), and step / ks as a multiply-high (fp4r_fmt's mg, sh: five instructions; the division and its
                          // power-of-two branch had been 24 a step, issued whether taken or not)
    struct St {
        uint32_t lo[4], base[2], off;
    };
    static constexpr bool kTable = true;  // (the tiered kernels' shape: their table and a warp's scratch are not read)
    __device__ __forceinline__ const uint8_t* block(int64_t step, uint32_t& k) const {
        const uint32_t n = (uint32_t)step, t = __umulhi(n, mg), rb = (t + ((n - t) >> 1)) >> sh;
        k = (uint32_t)step - rb * ks;
        return scales + (size_t)rb * (64 + ks * 8);
    }
    __device__ __forceinline__ void load(St& st, int64_t step, int lane) const {
        const uint8_t* p = data + step * INT4_STEP_BYTES;
        const uint2* l = (const uint2*)p + lane;
        uint2 a = __ldg(l), b = __ldg(l + 32);
        st.lo[0] = a.x; st.lo[1] = a.y; st.lo[2] = b.x; st.lo[3] = b.y;
        uint32_t k;
        const uint8_t* r = block(step, k);
        const uint2 c = __ldg((const uint2*)r + (lane >> 2));
        st.base[0] = c.x; st.base[1] = c.y;
        st.off = __ldg((const uint16_t*)(r + 64 + (k >> 1) * 16) + (lane >> 2));
    }
    static constexpr int kStage = INT4_STEP_BYTES + 64 + 16;  // (as Int6's: the codes, the row block's bases, the pair's offsets)
    __device__ __forceinline__ const uint8_t* chunk(int64_t step, int c) const {
        if (c < 32) return data + step * INT4_STEP_BYTES + 16 * c;
        uint32_t k;
        const uint8_t* r = block(step, k);
        return c < 36 ? r + 16 * (c - 32) : r + 64 + (k >> 1) * 16;
    }
    __device__ __forceinline__ void load_s(St& st, const uint8_t* p, int lane) const {
        const uint2* l = (const uint2*)p + lane;
        uint2 a = l[0], b = l[32];
        st.lo[0] = a.x; st.lo[1] = a.y; st.lo[2] = b.x; st.lo[3] = b.y;
        const uint2 c = ((const uint2*)(p + INT4_STEP_BYTES))[lane >> 2];
        st.base[0] = c.x; st.base[1] = c.y;
        st.off = ((const uint16_t*)(p + INT4_STEP_BYTES + 64))[lane >> 2];
    }
    __device__ __forceinline__ void decode(const St& st, int, uint32_t*, const uint32_t*, uint32_t R[16]) const {
        // (the offsets' 2-bit fields spread to bytes and added to the bases: a scale is at most 254, so no byte carries)
        const auto bytes = [](uint32_t x) { return (x | (x << 6) | (x << 12) | (x << 18)) & 0x03030303u; };
        fp4_step(st.lo, st.base[0] + bytes(st.off & 0xFFu), st.base[1] + bytes(st.off >> 8), R);
    }
    __device__ __forceinline__ void prefetch(int64_t step, int lane) const {
        if (lane < (int)(INT4_STEP_BYTES / 128)) pf_l2(data + step * INT4_STEP_BYTES + lane * 128);
        else if (lane == (int)(INT4_STEP_BYTES / 128) && !(step & 1)) {
            uint32_t k;
            const uint8_t* r = block(step, k);
            pf_l2(r + 64 + (k >> 1) * 16);
        }
    }
    __device__ __forceinline__ void prefetch_exc(int64_t, int64_t, int) const {}
};

// The three-bit block layout (INT3_TAG in tiers[0]'s top byte): each run of 32 values of a row of W (columns 32 kb on) as three-bit codes u,
// 1 to 7 (the packer's codes run -3 to 3, as the six- and five-bit layouts' -31 to 31 and -15 to 15: the eighth code, 0, decodes as -4 s), and
// one bf16 scale s for the run; the weight is round_bf16((u - 4) s) (the product of a 3-bit integer and an 8-bit significand, rounded once: the
// packer's own reference). A warp step's 384 bytes: 128 bytes of the codes' high bit as the five-bit layout's (lane l's word at l; its pair r,
// weights 2r and 2r + 1 of the step's 32 a lane, at bits r and 16 + r), then 256 bytes of their low two bits as the six-bit layout's high two
// (lane l's two words at 32 d + l; pair r at bits 2(r mod 8) and 16 + 2(r mod 8) of word r / 8). The scales (`blocks`) as the four-bit layout's.
// A pair's two bf16s are an OR (0x4300 | u: 128 + u), a subtraction of 132 (exact) and a product by the scale, two bf16x2 instructions a pair.
constexpr uint32_t INT3_TAG = 7;
constexpr int64_t INT3_STEP_BYTES = 384;
struct Int3 {
    const uint8_t* data;
    const uint8_t* scales;
    struct St {
        uint32_t hi, lo[2], sc[4];
    };
    static constexpr bool kTable = true;  // (the tiered kernels' shape: their table and a warp's scratch are not read)
    __device__ __forceinline__ void load(St& st, int64_t step, int lane) const {
        const uint8_t* p = data + step * INT3_STEP_BYTES;
        const uint32_t* h = (const uint32_t*)p + lane;
        st.hi = __ldg(h);
        st.lo[0] = __ldg(h + 32);
        st.lo[1] = __ldg(h + 64);
        uint4 c = __ldg((const uint4*)(scales + (step >> 1) * 128) + (lane >> 2));
        st.sc[0] = c.x; st.sc[1] = c.y; st.sc[2] = c.z; st.sc[3] = c.w;
    }
    static constexpr int kStage = INT3_STEP_BYTES + 128;  // (as Int6's)
    __device__ __forceinline__ const uint8_t* chunk(int64_t step, int c) const { return c < 24 ? data + step * INT3_STEP_BYTES + 16 * c : scales + (step >> 1) * 128 + 16 * (c - 24); }
    __device__ __forceinline__ void load_s(St& st, const uint8_t* p, int lane) const {
        const uint32_t* h = (const uint32_t*)p + lane;
        st.hi = h[0];
        st.lo[0] = h[32];
        st.lo[1] = h[64];
        uint4 c = ((const uint4*)(p + INT3_STEP_BYTES))[lane >> 2];
        st.sc[0] = c.x; st.sc[1] = c.y; st.sc[2] = c.z; st.sc[3] = c.w;
    }
    __device__ __forceinline__ void decode(const St& st, int, uint32_t*, const uint32_t*, uint32_t R[16]) const {
        const uint32_t off = 0x43044304u;  // (132, 132)
#pragma unroll
        for (int r = 0; r < 16; r++) {
            uint32_t x = 0x43004300u | ((st.lo[r >> 3] >> (2 * (r & 7))) & 0x00030003u) | (((st.hi >> r) & 0x00010001u) << 2);
            uint32_t sw = __byte_perm(st.sc[r >> 2], 0, ((r >> 1) & 1) ? 0x3232 : 0x1010);
            __nv_bfloat162 v = __hmul2(__hsub2(*reinterpret_cast<const __nv_bfloat162*>(&x), *reinterpret_cast<const __nv_bfloat162*>(&off)), *reinterpret_cast<const __nv_bfloat162*>(&sw));
            R[r] = *reinterpret_cast<uint32_t*>(&v);
        }
    }
    __device__ __forceinline__ void prefetch(int64_t step, int lane) const {
        if (lane < (int)(INT3_STEP_BYTES / 128)) pf_l2(data + step * INT3_STEP_BYTES + lane * 128);
        else if (lane == (int)(INT3_STEP_BYTES / 128) && !(step & 1)) pf_l2(scales + (step >> 1) * 128);
    }
    __device__ __forceinline__ void prefetch_exc(int64_t, int64_t, int) const {}
};

// The 12-bit layout (mma12), split byte: a weight's bf16 as its low byte (the exponent's lowest bit and the 7
// mantissa bits), stored as it is, and its high byte (the sign and the exponent's other 7 bits), coded in 4 bits: the
// sign and an offset 0-7 from the matrix's base hb (the 8 values of the high byte's 7 bits from hb, 16 exponents,
// holding the most weights; hb 0-120, in each byte of hb4). A warp step: its lanes' codes, 16 bytes a lane, then its
// 1024 low bytes as two halves (every lane's first 16, then every lane's last 16). Code word q of a lane holds n-tiles
// 2q and 2q + 1 (weights 8q to 8q + 7): 2q's codes in bits {7, 2, 1, 0} of byte j (sign, offset), 2q + 1's rotated
// by 4 (its sign in bit 3 of byte j, its offset in bits 4-6 of byte j - 1 mod 4). A lane's 4 high bytes are then an
// AND and an add (2q + 1's a funnel shift first), a pair of bf16s one byte permute of [high, low, high, low]. A weight
// outside the 8 (whatever its exponent: zeros, subnormals, infinities alike) is an exception, coded with offset 0: its
// entry, its weight (lane * 32 + i) and the byte to XOR into its high byte (hb ^ the exponent >> 1) << 16; a step's run
// from exc_base[step] to exc_base[step + 1].
constexpr int64_t STEP12 = 1536;

struct Nib {
    const uint8_t* data;
    const uint32_t* exc;
    const int32_t* exc_base;
    uint32_t hb4;  // the high bytes' base, hb in each byte
    int xmax = 4;  // decode_x: a step with more than xmax exceptions takes them through the scratch (GLYD_GPU_XMAX, read by nib12)
    struct St {
        uint32_t nb[4], sw[8];
        int e0, e1;
    };
    static constexpr bool kTable = false;
    __device__ __forceinline__ void load(St& st, int64_t step, int lane) const {
        const uint8_t* p = data + step * STEP12;
        uint4 c = __ldg((const uint4*)p + lane);
        st.nb[0] = c.x; st.nb[1] = c.y; st.nb[2] = c.z; st.nb[3] = c.w;
        const uint4* sp = (const uint4*)(p + 512) + lane;
        uint4 x0 = __ldg(sp), x1 = __ldg(sp + 32);
        st.sw[0] = x0.x; st.sw[1] = x0.y; st.sw[2] = x0.z; st.sw[3] = x0.w;
        st.sw[4] = x1.x; st.sw[5] = x1.y; st.sw[6] = x1.z; st.sw[7] = x1.w;
        st.e0 = __ldg(exc_base + step);
        st.e1 = __ldg(exc_base + step + 1);
    }
    // The whole-matrix decode's step (mma_unpack_kernel, a warp a step: for cuBLAS, exact mode and the experts'), its
    // loads in two batches: the low bytes and the step's exception bounds, then the codes once those are in (load()
    // issues all at once). The codes' address waits on the first batch through after(its sum), 0: the sum shuffled
    // from its own lane, less the sum (a shuffle the compiler does not see through), so the same bytes, bit for bit.
    // then(): the kernel's own work (its row block and step, a division), once the codes' load is issued. Hopper's
    // alone: on a GH200 (layer 10 of Qwen3-8B, 14B and 32B, 2026-09-29) the layer's decode takes 0.964-0.975 of the
    // time of the 12-bit layout's before split byte, load()'s all at once 1.035-1.060; on an L4 (Qwen3-8B and 4B)
    // 1.020-1.022 against load()'s 1.001-1.003. The decode ahead (a few warps an SM) keeps load() everywhere: on the
    // GH200 0.703-0.725, this order 1.058-1.068; on the L4 0.994-0.995 against 1.106-1.134.
    template <class Then>
    __device__ __forceinline__ void load_decode(int64_t step, int lane, uint32_t R[16], Then then) const {
        const uint8_t* p = data + step * STEP12;
        auto after = [lane](uint32_t v) { return __shfl_sync(0xffffffffu, v, lane) - v; };
        int e0 = __ldg(exc_base + step), e1 = __ldg(exc_base + step + 1);
        const uint4* sp = (const uint4*)(p + 512) + lane;
        uint4 x0 = __ldg(sp), x1 = __ldg(sp + 32);
        uint4 c = __ldg((const uint4*)(p + after(x0.x + x1.x + e0 + e1)) + lane);
        then();
        uint32_t nb[4] = {c.x, c.y, c.z, c.w}, H[8];
        high(nb, H);
        const uint32_t* e = exc;
        patch([e](int k) { return __ldg(e + k); }, e0, e1, lane, H);
        uint32_t L[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
        pairs(L, H, R);
    }
    // Code word q's high bytes: n-tile 2q's (h = 0), 2q + 1's (h = 1).
    static __device__ __forceinline__ uint32_t high1(uint32_t nb, uint32_t hb4, int h) { return ((h ? __funnelshift_l(nb, nb, 4) : nb) & 0x87878787u) + hb4; }
    // A lane's codes (nb, 4 words) as its high bytes, 4 a word (H).
    __device__ __forceinline__ void high(const uint32_t nb[4], uint32_t H[8]) const {
#pragma unroll
        for (int q = 0; q < 8; q++) H[q] = high1(nb[q >> 1], hb4, q & 1);
    }
    // The step's exceptions, entries e0 to e1 (the same run for every lane of the warp; entry k is at(k)): each XORs
    // its byte into this lane's word where the weight is this lane's. U: the loop unrolled U times, 0 as nvcc chooses
    // (4 a pass in these kernels, as it did the 12-bit layout's before split byte). An entry a pass (1) where unrolled it
    // spilled (mma_moe_kernel's 64-token products with an activation, sm_80: main's 4 spill instructions, 12
    // unrolled, none so), and 2 in mma12_ws_kernel, as nvcc unrolled it there before split byte: kept a pass an entry,
    // the L4's prompts took 1-2% longer (2026-09-29).
    template <int U = 0, class At>
    static __device__ __forceinline__ void patch(At at, int e0, int e1, int lane, uint32_t H[8]) {
        auto one = [&](int k) {
            uint32_t x = at(k), i = x & 31, v = (x >> 16 & 0xFFu) << (8 * (i & 3));
            uint32_t w = (int)((x >> 5) & 31) == lane ? i >> 2 : 8u;
#pragma unroll
            for (int q = 0; q < 8; q++) H[q] ^= w == (uint32_t)q ? v : 0u;
        };
        if constexpr (U == 0) {
            for (int k = e0; k < e1; k++) one(k);
        } else {
#pragma unroll U
            for (int k = e0; k < e1; k++) one(k);
        }
    }
    // Pairs: [high, low, high, low], one byte permute each (a lane's low bytes L, 4 a word).
    static __device__ __forceinline__ void pairs(const uint32_t L[8], const uint32_t H[8], uint32_t R[16]) {
#pragma unroll
        for (int p = 0; p < 16; p++) R[p] = __byte_perm(L[p >> 1], H[p >> 1], (p & 1) ? 0x7362 : 0x5140);
    }
    // A step's decode where its block's steps have many exceptions (mma_gemm_kernel's heavy blocks): past xmax (4) in the
    // step, through the warp's scratch (s2: 1 KB, a lane's 8 words, zero between steps) the lanes take the run's
    // entries 32 at a time, each setting its byte there, which every lane then XORs into its words and clears, as
    // decode12_rows does; else as decode.
    __device__ __forceinline__ void decode_x(const St& st, int lane, uint32_t* s2, uint32_t R[16]) const {
        if (st.e1 - st.e0 <= xmax) return decode(st, lane, nullptr, nullptr, R);
        uint32_t H[8];
        high(st.nb, H);
        for (int k = st.e0 + lane; k < st.e1; k += 32) {
            uint32_t x = __ldg(exc + k), i = x & 31;
            ((uint8_t*)s2)[(((x >> 5) & 31) * 8 + (i >> 2)) * 4 + (i & 3)] = (uint8_t)(x >> 16);
        }
        __syncwarp();
        uint4* xl = (uint4*)(s2 + 8 * lane);
        uint4 a = xl[0], b = xl[1];
        H[0] ^= a.x, H[1] ^= a.y, H[2] ^= a.z, H[3] ^= a.w, H[4] ^= b.x, H[5] ^= b.y, H[6] ^= b.z, H[7] ^= b.w;
        xl[0] = xl[1] = make_uint4(0u, 0u, 0u, 0u);
        __syncwarp();
        pairs(st.sw, H, R);
    }
    // decode_x on scratch that is not zero (a decoded step's slot of shared memory, the producers' of mma_gemm_big_kernel): zeroed
    // here where the step has the many exceptions that go through it.
    __device__ __forceinline__ void decode_xz(const St& st, int lane, uint32_t* s2, uint32_t R[16]) const {
        if (st.e1 - st.e0 <= xmax) return decode(st, lane, nullptr, nullptr, R);
        ((uint4*)s2)[2 * lane] = ((uint4*)s2)[2 * lane + 1] = make_uint4(0u, 0u, 0u, 0u);
        __syncwarp();
        decode_x(st, lane, s2, R);
    }
    template <int U = 0, bool B = false>  // patch's unrolling; B: the step's first 4 exceptions loaded together (patch_b)
    __device__ __forceinline__ void decode(const St& st, int lane, uint32_t*, const uint32_t*, uint32_t R[16]) const {
        uint32_t H[8];
        high(st.nb, H);
        const uint32_t* e = exc;
        if constexpr (B) patch_b([e](int k) { return __ldg(e + k); }, st.e0, st.e1, lane, H);
        else patch<U>([e](int k) { return __ldg(e + k); }, st.e0, st.e1, lane, H);
        pairs(st.sw, H, R);
    }
    // patch with the step's first 4 entries loaded together, before any is applied: an entry's load is a dependent trip to
    // L2 or memory in the middle of the decode, and patch's loop waits for each before it loads the next.
    template <class At>
    static __device__ __forceinline__ void patch_b(At at, int e0, int e1, int lane, uint32_t H[8]) {
        auto one = [&](uint32_t x) {
            uint32_t i = x & 31, v = (x >> 16 & 0xFFu) << (8 * (i & 3));
            uint32_t w = (int)((x >> 5) & 31) == lane ? i >> 2 : 8u;
#pragma unroll
            for (int q = 0; q < 8; q++) H[q] ^= w == (uint32_t)q ? v : 0u;
        };
        int n = e1 - e0;
        uint32_t x0 = n > 0 ? at(e0) : 0u, x1 = n > 1 ? at(e0 + 1) : 0u, x2 = n > 2 ? at(e0 + 2) : 0u, x3 = n > 3 ? at(e0 + 3) : 0u;
        if (n > 0) one(x0);
        if (n > 1) one(x1);
        if (n > 2) one(x2);
        if (n > 3) one(x3);
        for (int k = e0 + 4; k < e1; k++) one(at(k));
    }
    // Ahead: step's 12 lines of 128 bytes (data, a step 1536 bytes: lane i takes line i) to L2, where the loads
    // that follow find them.
    __device__ __forceinline__ void prefetch(int64_t step, int lane) const {
        if (lane < (int)(STEP12 / 128)) pf_l2(data + step * STEP12 + lane * 128);
    }
    // The lines of the exceptions of steps [a, b) (one run: entries exc_base[a] to exc_base[b]) to L2.
    __device__ __forceinline__ void prefetch_exc(int64_t a, int64_t b, int lane) const {
        int64_t lo = (int64_t)__ldg(exc_base + a) * 4, hi = (int64_t)__ldg(exc_base + b) * 4;
        for (int64_t p = (lo & ~127ll) + 128 * lane; p < hi; p += 128 * 32) pf_l2((const uint8_t*)exc + p);
    }
};

__device__ __forceinline__ void mma16816(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
#if defined(__HIP_PLATFORM_AMD__)
    glyd_mma16816(c, a, b0, b1);  // (warp_emu.h: shuffles and FMAs)
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

// Y = X W^T for up to 16 MT tokens from the mma layout. The W steps
// (64 rows by 16 columns), in row-block order, are split evenly over the
// blocks (stream-K: no block is left for a second wave); a block's run, cut
// where a row block ends, is split over its 8 warps, whose sums are added in
// shared memory. A row block covered by several blocks: each writes its
// part to a slot, and the last to finish adds the parts in block order (the
// same result every run). A warp's step: 1024 weights decoded into B
// fragments, 8 MT tensor-core products. The next step's loads are issued
// before this step's decode.
__device__ __forceinline__ int64_t block_of_step(int64_t x, int64_t nb, int64_t total) {
    return ((x + 1) * nb - 1) / total;  // block b runs steps [b total / nb, (b + 1) total / nb)
}

// A warp's steps s0 to s1 of a row block (W's steps base + s): acc += X's rows by the step's 64 rows, 16 MT tokens
// (xr: this thread's rows g and g + 8 of each m-tile, at X + row K + 2t; null past the tokens). The next step's
// loads, W's and then the inputs, are issued before this step's decode. pf: the lines of the steps pf ahead of the next
// one are asked of L2 meanwhile (and the warp's exceptions at the start), so that the loads find them there and not in
// memory; PF: only where compiled in (a register and a branch a step otherwise), 0: not. SK: na, the m-tiles with tokens in them, the first na of MT (the others' loads and products are not made).
// mma_gemm_kernel's and mma_moe_kernel's.
template <class Fmt, int MT, bool X = false, int U = 0, bool B = false, bool SK = false, bool PF = false, int D = 1>  // X: decode_x (the 12-bit layout's heavy blocks); U: Nib::patch's unrolling; B: Nib::patch_b; SK: only the m-tiles below na; PF: the L2 prefetch (pf ahead); D: the steps loaded ahead (2: two steps' loads in flight a warp, for HBM GPUs' one-token products, which waited on one at a time)
__device__ __forceinline__ void mma_steps(const Fmt& f, int64_t base, int64_t s0, int64_t s1, const __nv_bfloat16* const (&xr)[MT][2], int lane, uint32_t* s2, const uint32_t* tab, float (&acc)[MT][8][4], int pf = 0, int na = MT) {
    typename Fmt::St st[D];
    uint32_t a[D][MT][4];
    auto load = [&](int d, int64_t s) {
        f.load(st[d], base + s, lane);
#pragma unroll
        for (int mt = 0; mt < MT; mt++) {
            if (!SK || MT == 1 || mt < na) {
                a[d][mt][0] = xr[mt][0] ? *(const uint32_t*)(xr[mt][0] + s * 16) : 0u;
                a[d][mt][2] = xr[mt][0] ? *(const uint32_t*)(xr[mt][0] + s * 16 + 8) : 0u;
                a[d][mt][1] = xr[mt][1] ? *(const uint32_t*)(xr[mt][1] + s * 16) : 0u;
                a[d][mt][3] = xr[mt][1] ? *(const uint32_t*)(xr[mt][1] + s * 16 + 8) : 0u;
            }
        }
    };
    if constexpr (PF)
        for (int j = 1; j <= pf && s0 + j < s1; j++) f.prefetch(base + s0 + j, lane);
#pragma unroll
    for (int d = 0; d < D; d++)
        if (s0 + d < s1) load(d, s0 + d);
    if constexpr (PF)
        if (pf && s0 < s1) f.prefetch_exc(base + s0, base + s1, lane);
    for (int64_t s = s0; s < s1; s += D) {
#pragma unroll
        for (int d = 0; d < D; d++) {
            const int64_t sd = s + d;
            if (sd >= s1) break;
            typename Fmt::St cur = st[d];
            uint32_t ca[MT][4];
#pragma unroll
            for (int mt = 0; mt < MT; mt++)
#pragma unroll
                for (int q = 0; q < 4; q++) ca[mt][q] = a[d][mt][q];
            if constexpr (PF)
                if (pf && sd + 1 + pf < s1) f.prefetch(base + sd + 1 + pf, lane);
            if (sd + D < s1) load(d, sd + D);  // (the slot's next: D steps ahead)
            uint32_t R[16];
            if constexpr (X) f.decode_x(cur, lane, s2, R);
            else if constexpr (std::is_same_v<Fmt, Nib>) f.template decode<U, B>(cur, lane, s2, tab, R);
            else f.decode(cur, lane, s2, tab, R);
#pragma unroll
            for (int mt = 0; mt < MT; mt++)
                if (!SK || MT == 1 || mt < na)
#pragma unroll
                    for (int nn = 0; nn < 8; nn++) mma16816(acc[mt][nn], ca[mt], R[2 * nn], R[2 * nn + 1]);
        }
    }
}

// The block layouts' one-token steps through shared memory (mma_moe_kernel's V & 16, an HBM GPU): a warp's next P - 1 steps (their bytes as Fmt::chunk gives them)
// copied by cp.async into its ring of P slots, so that the steps in flight take no registers; a slot read back with Fmt::load_s and decoded as mma_steps does. The
// activations' words two steps ahead in registers (from L2; a slot of them had made the rings too big for 3 blocks an SM on an A100).
template <class F, class = void>
struct has_stage : std::false_type {};
template <class F>
struct has_stage<F, std::void_t<decltype(F::kStage)>> : std::true_type {};
template <class Fmt>
constexpr int ring_slot() {
    if constexpr (has_stage<Fmt>::value) return Fmt::kStage;
    else return 0;
}
#if defined(__HIP_PLATFORM_AMD__)  // (AMD: cp_async16 copies when asked, so there is nothing to wait for)
template <int N>
__device__ __forceinline__ void cp_wait() {}
__device__ __forceinline__ void cp_commit() {}
#else
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
#endif
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int bytes);
template <class Fmt, int P>
__device__ __forceinline__ void mma_steps_staged(const Fmt& f, int64_t base, int64_t s0, int64_t s1, const __nv_bfloat16* const (&xr)[1][2], int lane, uint8_t* ring, const uint32_t* tab, float (&acc)[1][8][4]) {
    constexpr int SLOT = ring_slot<Fmt>(), CH = Fmt::kStage / 16;
    const uint32_t r0 = (uint32_t)__cvta_generic_to_shared(ring);
    const __nv_bfloat16* x0 = xr[0][0];
    const __nv_bfloat16* x1 = xr[0][1];
    auto stage = [&](int j, int64_t s) {
        const uint32_t d = r0 + j * SLOT;
#pragma unroll
        for (int i = 0; i < (CH + 31) / 32; i++) {
            const int c = lane + 32 * i;
            if (c < CH) cp_async16(d + c * 16, f.chunk(base + s, c), 16);
        }
    };
    // (mma_steps' a: row g's words at columns 2t and 8 + 2t, then row g + 8's)
    auto loadx = [&](int64_t s, uint32_t (&a)[4]) {
        a[0] = x0 ? *(const uint32_t*)(x0 + s * 16) : 0u;
        a[2] = x0 ? *(const uint32_t*)(x0 + s * 16 + 8) : 0u;
        a[1] = x1 ? *(const uint32_t*)(x1 + s * 16) : 0u;
        a[3] = x1 ? *(const uint32_t*)(x1 + s * 16 + 8) : 0u;
    };
    uint32_t a1[4] = {0u, 0u, 0u, 0u}, a2[4] = {0u, 0u, 0u, 0u};
    if (s0 < s1) loadx(s0, a1);
    if (s0 + 1 < s1) loadx(s0 + 1, a2);
#pragma unroll
    for (int j = 0; j < P - 1; j++) {
        if (s0 + j < s1) stage(j, s0 + j);
        cp_commit();
    }
    for (int64_t s = s0; s < s1; s++) {
        const int j = (int)((s - s0) & (P - 1));
        uint32_t ca[4] = {a1[0], a1[1], a1[2], a1[3]};
#pragma unroll
        for (int q = 0; q < 4; q++) a1[q] = a2[q];
        if (s + 2 < s1) loadx(s + 2, a2);
        __syncwarp();  // (every lane done with step s - 1's slot, which the next copy fills)
        if (s + P - 1 < s1) stage((j + P - 1) & (P - 1), s + P - 1);
        cp_commit();
        cp_wait<P - 1>();
        __syncwarp();  // (step s's slot in: every lane's copies)
        typename Fmt::St cur;
        f.load_s(cur, ring + j * SLOT, lane);
        uint32_t R[16];
        f.decode(cur, lane, nullptr, tab, R);
#pragma unroll
        for (int nn = 0; nn < 8; nn++) mma16816(acc[0][nn], ca, R[2 * nn], R[2 * nn + 1]);
    }
}

// The block layouts' one-token steps through shared memory, S steps a stage (research, 2026-10-20): a warp's ring is two groups of S slots; a stage copies the
// group after next's S steps in one go (one commit, one wait and two warp syncs a group instead of a step), and the group's S decodes follow one another.
// The sums as mma_steps_staged's, step by step in the same order.
template <class Fmt, int S>
__device__ __forceinline__ void mma_steps_staged_s(const Fmt& f, int64_t base, int64_t s0, int64_t s1, const __nv_bfloat16* const (&xr)[1][2], int lane, uint8_t* ring, const uint32_t* tab, float (&acc)[1][8][4]) {
    constexpr int SLOT = ring_slot<Fmt>(), CH = Fmt::kStage / 16, GROUP = S * SLOT;
    const uint32_t r0 = (uint32_t)__cvta_generic_to_shared(ring);
    const __nv_bfloat16* x0 = xr[0][0];
    const __nv_bfloat16* x1 = xr[0][1];
    auto stage = [&](int grp, int64_t s) {  // steps s to s + S - 1 (those before s1) into group grp
        const uint32_t d = r0 + grp * GROUP;
#pragma unroll
        for (int i = 0; i < (S * CH + 31) / 32; i++) {
            const int c = lane + 32 * i;
            if (c < S * CH) {
                const int st = c / CH, cc = c - st * CH;
                if (s + st < s1) cp_async16(d + st * SLOT + cc * 16, f.chunk(base + s + st, cc), 16);
            }
        }
    };
    auto loadx = [&](int64_t s, uint32_t (&a)[4]) {
        a[0] = x0 ? *(const uint32_t*)(x0 + s * 16) : 0u;
        a[2] = x0 ? *(const uint32_t*)(x0 + s * 16 + 8) : 0u;
        a[1] = x1 ? *(const uint32_t*)(x1 + s * 16) : 0u;
        a[3] = x1 ? *(const uint32_t*)(x1 + s * 16 + 8) : 0u;
    };
    uint32_t a1[4] = {0u, 0u, 0u, 0u}, a2[4] = {0u, 0u, 0u, 0u};
    if (s0 < s1) loadx(s0, a1);
    if (s0 + 1 < s1) loadx(s0 + 1, a2);
    stage(0, s0);
    cp_commit();
    if (s0 + S < s1) stage(1, s0 + S);
    cp_commit();
    int grp = 0;
    for (int64_t s = s0; s < s1; s += S) {
        cp_wait<1>();  // this group's steps in (the next group's may still be coming)
        __syncwarp();
#pragma unroll
        for (int st = 0; st < S; st++) {
            if (s + st >= s1) break;
            uint32_t ca[4] = {a1[0], a1[1], a1[2], a1[3]};
#pragma unroll
            for (int q = 0; q < 4; q++) a1[q] = a2[q];
            if (s + st + 2 < s1) loadx(s + st + 2, a2);
            typename Fmt::St cur;
            f.load_s(cur, ring + grp * GROUP + st * SLOT, lane);
            uint32_t R[16];
            f.decode(cur, lane, nullptr, tab, R);
#pragma unroll
            for (int nn = 0; nn < 8; nn++) mma16816(acc[0][nn], ca, R[2 * nn], R[2 * nn + 1]);
        }
        __syncwarp();  // every lane done with the group: the group after next takes it
        if (s + 2 * S < s1) stage(grp, s + 2 * S);
        cp_commit();
        grp ^= 1;
    }
}

// The 8 warps' sums (C fragment rows g and g + 8, columns 2t and 2t + 1 of each n-tile) into shared memory, red
// [4][16 MT][65]: warps 4-7 put theirs, 0-3 add them to theirs and put those; the caller adds the four.
template <int MT>
__device__ __forceinline__ void warp_sums(float (&acc)[MT][8][4], float* red, int warp, int g, int t) {
    float* mine = red + (int64_t)(warp & 3) * MT * 16 * 65;
    for (int half = 1; half >= 0; half--) {
        if ((warp >> 2) == half)
#pragma unroll
            for (int mt = 0; mt < MT; mt++)
#pragma unroll
                for (int nn = 0; nn < 8; nn++) {
                    float* r0 = mine + (mt * 16 + g) * 65 + nn * 8 + t * 2;
                    float* r1 = r0 + 8 * 65;
                    if (!half) {
                        acc[mt][nn][0] += r0[0];
                        acc[mt][nn][1] += r0[1];
                        acc[mt][nn][2] += r1[0];
                        acc[mt][nn][3] += r1[1];
                    }
                    r0[0] = acc[mt][nn][0];
                    r0[1] = acc[mt][nn][1];
                    r1[0] = acc[mt][nn][2];
                    r1[1] = acc[mt][nn][3];
                }
        __syncthreads();
    }
}

// A block has written its part of a unit: whether it is the last of the unit's n blocks to finish (done: the unit's
// counter, zero before, reset by the last for the next product), every other part visible to it then (last: the
// block's shared flag).
__device__ __forceinline__ bool last_of(int* done, int64_t n, int& last) {
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        last = atomicAdd(done, 1) == n - 1;
        if (last) *done = 0;  // ready for the next product
    }
    __syncthreads();
    if (last) __threadfence();
    return last;
}

// A one-token product's blocks an SM (mma_gemm_kernel, MT 1): 3 on Ada (sm_89), measured on an RTX 4080 SUPER (Gemma 4 12B wren, a step 12.12 -> 11.96 ms;
// 4 is slower); 2 elsewhere until measured there (GLY-312). mma_gemm_mt sizes its grid from the occupancy the build gives.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 890
#define GEMM1_BLOCKS 3
#else
#define GEMM1_BLOCKS 2
#endif
template <class Fmt, int MT, bool B = false, bool PF = false>
__global__ void __launch_bounds__(256, MT == 1 ? GEMM1_BLOCKS : 1) mma_gemm_kernel(Fmt f, int64_t O, int64_t K, int64_t M, const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ Y, float* __restrict__ parts, int* __restrict__ done, int pf) {
    extern __shared__ __align__(128) float red[];  // [4 warps][16 MT rows][65], then (tiered) the warps' scratch [8][S2_BYTES]
    __shared__ int last;
    __shared__ uint32_t tab[Fmt::kTable ? 256 : 1];
    if constexpr (Fmt::kTable) fill_groups(tab);
    __syncthreads();
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    uint32_t* s2 = (uint32_t*)(red + 4 * MT * 16 * 65) + warp * (S2_BYTES / 4);
    int64_t KS = K / 16, total = O / 64 * KS, nb = gridDim.x;
    int64_t B1 = (blockIdx.x + 1) * total / nb;
    // 12-bit: where the block's steps have more than an exception each on average (a few layers' matrices: 4-5),
    // they go through the warp's scratch, 1 KB of red (zero between steps; the warps' sums take red only after
    // every warp's steps), with decode_x; elsewhere decode as before (its code with decode_x's in it, or more
    // shared memory a block, had cost the other layers 1%).
    bool heavy = false;
    if constexpr (!Fmt::kTable) {
        int64_t B0 = blockIdx.x * total / nb;
        heavy = __ldg(f.exc_base + B1) - __ldg(f.exc_base + B0) > B1 - B0;
        s2 = heavy ? (uint32_t*)red + 256 * warp : nullptr;
    }
    for (int64_t seg = blockIdx.x * total / nb; seg < B1;) {
        int64_t rb = seg / KS, sb = seg - rb * KS, se = min(B1, (rb + 1) * KS) - rb * KS;
        seg = rb * KS + se;
        int64_t per = (se - sb + 7) / 8;
        int64_t s0 = min(se, sb + warp * per), s1 = min(se, s0 + per);
        float acc[MT][8][4];
#pragma unroll
        for (int mt = 0; mt < MT; mt++)
#pragma unroll
            for (int nn = 0; nn < 8; nn++)
#pragma unroll
                for (int q = 0; q < 4; q++) acc[mt][nn][q] = 0.f;
        const __nv_bfloat16* xr[MT][2];  // this thread's rows of X (none past M)
#pragma unroll
        for (int mt = 0; mt < MT; mt++)
#pragma unroll
            for (int hh = 0; hh < 2; hh++) {
                int64_t r = mt * 16 + g + 8 * hh;
                xr[mt][hh] = r < M ? X + r * K + t * 2 : nullptr;
            }
        if (!Fmt::kTable && heavy) {
            ((uint4*)s2)[2 * lane] = ((uint4*)s2)[2 * lane + 1] = make_uint4(0u, 0u, 0u, 0u);
            __syncwarp();
            mma_steps<Fmt, MT, !Fmt::kTable, 0, B, false, PF>(f, rb * KS, s0, s1, xr, lane, s2, tab, acc, pf);
            __syncthreads();  // every warp's scratch done with
        } else {
            mma_steps<Fmt, MT, false, 0, B, false, PF>(f, rb * KS, s0, s1, xr, lane, s2, tab, acc, pf);
        }
        warp_sums<MT>(acc, red, warp, g, t);
        int64_t first = block_of_step(rb * KS, nb, total), fin = block_of_step(rb * KS + KS - 1, nb, total);
        for (int i = threadIdx.x; i < MT * 16 * 64; i += 256) {
            int r = i / 64, c = i % 64;
            if (r >= M) continue;
            float v = 0.f;
#pragma unroll
            for (int ww = 0; ww < 4; ww++) v += red[(ww * MT * 16 + r) * 65 + c];
            if (first == fin) Y[r * O + rb * 64 + c] = __float2bfloat16(v + (bias ? __bfloat162float(bias[rb * 64 + c]) : 0.f));
            else parts[((blockIdx.x + rb) * M + r) * 64 + c] = v;  // slot b + rb: distinct for every (block, row block) pair
        }
        if (first != fin && last_of(done + rb, fin - first + 1, last))
            for (int i = threadIdx.x; i < M * 64; i += 256) {
                int r = i / 64, c = i % 64;
                float v = 0.f;
                for (int64_t b = first; b <= fin; b++) v += __ldcg(parts + ((b + rb) * M + r) * 64 + c);
                Y[r * O + rb * 64 + c] = __float2bfloat16(v + (bias ? __bfloat162float(bias[rb * 64 + c]) : 0.f));
            }
        __syncthreads();  // red and last are reused
    }
}

// The tiered layout's product for 1 to 16 tokens (a generation step). mma_gemm_kernel's split of the steps over blocks and its
// last-to-arrive sum of a row block's parts, in block order, with blocks of 4 warps (their barriers, and the sums at their ends, are fewer
// and cheaper than 8 warps'); a warp's step decoded by the short form of `exponents` where the packer's flag (bit 31 of block_base[step])
// is clear, which says every check of it holds, and else by `exponents` itself, out of line; and up to 8 tokens with W's registers as the
// A operand of the tensor cores' product and X's as the B operand (16 accumulators a lane for 32, two loads of X a step for four).
//
// A step's loads: its fixed part's words, the block's first word a lane (win) and its last words (tail, tail1), and block_base's two
// entries as they are (rb's bit 31: the flag).
struct StepH {
    uint32_t pd[2], sw[8], win, tail, tail1, rb, re;
};

__device__ __forceinline__ void load_step_h(StepH& st, const uint8_t* __restrict__ data, const uint8_t* __restrict__ blocks, const int32_t* __restrict__ block_base, int64_t step, int lane) {
    const uint32_t* cp = (const uint32_t*)(data + step * STEP_BYTES) + lane;
    st.pd[0] = __ldg(cp);
    st.pd[1] = __ldg(cp + 32);
    const uint4* sp = (const uint4*)(data + step * STEP_BYTES + 256) + lane;
    uint4 x0 = __ldg(sp), x1 = __ldg(sp + 32);
    st.sw[0] = x0.x; st.sw[1] = x0.y; st.sw[2] = x0.z; st.sw[3] = x0.w;
    st.sw[4] = x1.x; st.sw[5] = x1.y; st.sw[6] = x1.z; st.sw[7] = x1.w;
    st.rb = (uint32_t)__ldg(block_base + step);
    st.re = (uint32_t)__ldg(block_base + step + 1);
    const uint32_t* bw = (const uint32_t*)blocks;
    uint32_t b = (st.rb & 0x7FFFFFFFu) >> 2, e = (st.re & 0x7FFFFFFFu) >> 2;
    st.win = __ldg(bw + b + lane);
    st.tail = __ldg(bw + e - 1 - lane);
    st.tail1 = __ldg(bw + e);
}

// The short form of `exponents` (the flag clear: at most 512 tier-1 escapes, the block's tier-3 digits and its list, of at most 32
// literals, within the 124 bytes of its window): the same exponents, no checks and no other paths.
__device__ __forceinline__ void exponents_h(const StepH& st, Tiers ts, int lane, uint32_t* s2, const uint32_t* tab, uint32_t ew[8]) {
    uint32_t e1[8];
#pragma unroll
    for (int q = 0; q < 8; q++) e1[q] = tab[(st.pd[q >> 2] >> (8 * (q & 3))) & 0xFF];
    uint32_t n1 = __popc(escapes(st.pd[0])) + __popc(escapes(st.pd[1]));
    uint32_t T1 = __reduce_add_sync(FULL, n1), words = (T1 + 15) >> 4;
    const uint32_t S3 = 0x01010101u * (ts.s3 >> 24);
    const uint32_t o = 8 * (st.rb & 3u);
    uint32_t up = __shfl_up_sync(FULL, st.tail, 1);
    uint32_t da = (uint32_t)lane < words ? __funnelshift_r(st.tail, lane ? up : st.tail1, 8 * (st.re & 3u)) : 0u;
    uint32_t na = __popc(escapes(da)), tot;
    uint32_t pre = scan(n1 | na << 11, lane, tot);
    uint32_t b1 = pre & 0x7FF, T2 = tot >> 11;
    uint32_t d3 = win_bits(st.win, o + 2 * (pre >> 11));
    uint32_t g3 = (__reduce_max_sync(FULL, na) + 3) >> 2;
    uint32_t y[4] = {0u, 0u, 0u, 0u}, q4[4];
#pragma unroll
    for (int q = 0; q < 4; q++)
        if (q < (int)g3) y[q] = __byte_perm(ts.s3, S3, tab[(d3 >> (8 * q)) & 0xFF]);
#pragma unroll
    for (int q = 0; q < 4; q++) q4[q] = group(da >> (8 * q), ts.s2, y, tab);
    ((uint4*)s2)[lane] = make_uint4(q4[0], q4[1], q4[2], q4[3]);
    uint32_t r0 = (2 * T2 + 7) >> 3, nl = list_entries(((st.re - st.rb) & 0x7FFFFFFFu) - r0 - 4 * words);
    __syncwarp();
    uint32_t x = win_bits(st.win, o + 8 * r0 + 18 * (uint32_t)lane);  // (every lane's: the shuffles are the warp's; a lane past the list's end has no entry)
    if ((uint32_t)lane < nl) ((uint8_t*)s2)[x & 0x3FF] = (uint8_t)(x >> 10);
    __syncwarp();
    uint32_t c = b1;
#pragma unroll
    for (int q = 0; q < 8; q++) {
        uint32_t e = e1[q];
        ew[q] = __byte_perm(ts.s1, __funnelshift_r(s2[c >> 2], s2[(c >> 2) + 1], c << 3), e);
        c += e >> 16;
    }
    __syncwarp();  // the scratch is read
}

// A step the short form cannot take: `exponents` on the step by value (the loop's own steps stay in registers: it saves them around
// the call, on this path alone), its 8 words a lane left in eout [8][32].
__device__ __noinline__ void exponents_cold(Step st, Tiers ts, const uint8_t* __restrict__ blocks, int lane, uint32_t* s2, const uint32_t* tab, uint32_t* eout) {
    uint32_t ew[8];
    exponents(st, ts, blocks, lane, s2, tab, ew);
#pragma unroll
    for (int q = 0; q < 8; q++) eout[q * 32 + lane] = ew[q];  // (the scratch is read, exponents' last barrier: eout may be it)
    __syncwarp();
}

__device__ __forceinline__ Step step_of(const StepH& h) {
    Step s;
    s.pd[0] = h.pd[0]; s.pd[1] = h.pd[1];
#pragma unroll
    for (int i = 0; i < 8; i++) s.sw[i] = h.sw[i];
    s.win = h.win; s.tail = h.tail; s.tail1 = h.tail1;
    s.blk = h.rb & 0x7FFFFFFFu;
    s.end = h.re & 0x7FFFFFFFu;
    return s;
}

constexpr int STEP_WARPS = 4, STEP_BLOCKS = 4;  // warps a block, blocks an SM

// A warp's steps s0 to s1 of a row block (W's steps base + s) into acc. SW (up to 8 tokens): 4 products of 16 rows by 16 columns by 8
// tokens, W the A operand (acc[i]: rows 16 i to 16 i + 15 of the row block); else 8 of 16 tokens by 16 columns by 8 rows (acc[n]: rows 8 n
// to 8 n + 7). xr: this lane's row of X, g, and g + 8 (not SW), at X + row K + 2t, null past the tokens.
// A step's loads (and the 4 words of X it takes) are made a step ahead, into one of two sets of registers (the step s is in the first
// where s - s0 is even, else in the second): the step in one set is decoded and multiplied while the next one's loads, issued before
// its decode, fill the other. The flags of the next 32 steps are read at once, by one ballot: the run of steps that have none goes by the
// short form with no test, and the step that has one by `exponents`.
template <bool SW>
__device__ __forceinline__ void step_steps(const Tiered& f, int64_t base, int s0, int s1, const __nv_bfloat16* const (&xr)[2], int lane, uint32_t* s2, const uint32_t* tab, float (&acc)[SW ? 4 : 8][4]) {
    StepH sa, sb;
    uint32_t xa[4] = {0u, 0u, 0u, 0u}, xb[4] = {0u, 0u, 0u, 0u};
    auto load = [&](StepH& st, uint32_t (&x)[4], int s) {
        load_step_h(st, f.data, f.blocks, f.block_base, base + s, lane);
        x[0] = xr[0] ? *(const uint32_t*)(xr[0] + s * 16) : 0u;
        x[2] = xr[0] ? *(const uint32_t*)(xr[0] + s * 16 + 8) : 0u;
        if constexpr (!SW) {
            x[1] = xr[1] ? *(const uint32_t*)(xr[1] + s * 16) : 0u;
            x[3] = xr[1] ? *(const uint32_t*)(xr[1] + s * 16 + 8) : 0u;
        }
    };
    // the step s, in (cur, xc), its successor's loads first into (nxt, xn); cold: the flag is set
    auto step = [&](StepH& cur, const uint32_t (&xc)[4], StepH& nxt, uint32_t (&xn)[4], int s, auto cold) {
        if (s + 1 < s1) load(nxt, xn, s + 1);
        uint32_t ew[8], R[16];
        if constexpr (decltype(cold)::value) {
            exponents_cold(step_of(cur), f.ts, f.blocks, lane, s2, tab, s2);
#pragma unroll
            for (int q = 0; q < 8; q++) ew[q] = s2[q * 32 + lane];
            __syncwarp();
        } else {
            exponents_h(cur, f.ts, lane, s2, tab, ew);
        }
        pairs(cur.sw, ew, R);
        if constexpr (SW) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const uint32_t A4[4] = {R[4 * i], R[4 * i + 2], R[4 * i + 1], R[4 * i + 3]};
                mma16816(acc[i], A4, xc[0], xc[2]);
            }
        } else {
#pragma unroll
            for (int nn = 0; nn < 8; nn++) mma16816(acc[nn], xc, R[2 * nn], R[2 * nn + 1]);
        }
    };
    if (s0 < s1) load(sa, xa, s0);
    for (int s = s0; s < s1;) {
        int n = s1 - s < 32 ? s1 - s : 32;
        uint32_t flags = __ballot_sync(FULL, lane < n && __ldg(f.block_base + base + s + lane) < 0);
        int e = s + (flags ? __ffs(flags) - 1 : n);  // the steps s to e - 1 have none
        if (((s - s0) & 1) && s < e) {
            step(sb, xb, sa, xa, s, std::false_type{});
            s++;
        }
        for (; s < e; s += 2) {
            step(sa, xa, sb, xb, s, std::false_type{});
            if (s + 1 < e) step(sb, xb, sa, xa, s + 1, std::false_type{});
        }
        s = e;
        if (flags) {
            if ((s - s0) & 1) step(sb, xb, sa, xa, s, std::true_type{});
            else step(sa, xa, sb, xb, s, std::true_type{});
            s++;
        }
    }
}

template <bool SW>
__global__ void __launch_bounds__(32 * STEP_WARPS, STEP_BLOCKS) step_kernel(Tiered f, int64_t O, int64_t K, int64_t M, const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ Y, float* __restrict__ parts, int* __restrict__ done) {
    constexpr int NW = STEP_WARPS, ROWS = SW ? 8 : 16;  // warps a block, tokens a pass
    extern __shared__ __align__(128) float red[];  // [NW warps][ROWS][65] sums, then the warps' scratch [NW][S2_BYTES]
    __shared__ int last;
    __shared__ uint32_t tab[256];
    fill_groups(tab);
    __syncthreads();
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    uint32_t* s2 = (uint32_t*)(red + NW * ROWS * 65) + warp * (S2_BYTES / 4);
    int64_t KS = K / 16, total = O / 64 * KS, nb = gridDim.x;
    int64_t B1 = (blockIdx.x + 1) * total / nb;
    for (int64_t seg = blockIdx.x * total / nb; seg < B1;) {
        int64_t rb = seg / KS, sb = seg - rb * KS, se = min(B1, (rb + 1) * KS) - rb * KS;
        seg = rb * KS + se;
        int64_t per = (se - sb + NW - 1) / NW;
        int64_t s0 = min(se, sb + warp * per), s1 = min(se, s0 + per);
        const __nv_bfloat16* xr[2];
        xr[0] = g < M ? X + g * K + t * 2 : nullptr;
        xr[1] = !SW && g + 8 < M ? X + (g + 8) * K + t * 2 : nullptr;
        float acc[SW ? 4 : 8][4];
#pragma unroll
        for (int i = 0; i < (SW ? 4 : 8); i++)
#pragma unroll
            for (int q = 0; q < 4; q++) acc[i][q] = 0.f;
        step_steps<SW>(f, rb * KS, (int)s0, (int)s1, xr, lane, s2, tab, acc);  // (a row block's steps: K / 16 of them, an int)
        float* mine = red + warp * ROWS * 65;  // this warp's sums: C fragment (rows g, g + 8; columns 2t, 2t + 1) of each product
        if constexpr (SW) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
                mine[(2 * t) * 65 + 16 * i + g] = acc[i][0];
                mine[(2 * t + 1) * 65 + 16 * i + g] = acc[i][1];
                mine[(2 * t) * 65 + 16 * i + g + 8] = acc[i][2];
                mine[(2 * t + 1) * 65 + 16 * i + g + 8] = acc[i][3];
            }
        } else {
#pragma unroll
            for (int nn = 0; nn < 8; nn++) {
                float* r0 = mine + g * 65 + nn * 8 + t * 2;
                float* r1 = r0 + 8 * 65;
                r0[0] = acc[nn][0];
                r0[1] = acc[nn][1];
                r1[0] = acc[nn][2];
                r1[1] = acc[nn][3];
            }
        }
        __syncthreads();
        int64_t first = block_of_step(rb * KS, nb, total), fin = block_of_step(rb * KS + KS - 1, nb, total);
        for (int i = threadIdx.x; i < ROWS * 64; i += 32 * NW) {
            int r = i / 64, c = i % 64;
            if (r >= M) continue;
            float v = 0.f;
#pragma unroll
            for (int ww = 0; ww < NW; ww++) v += red[(ww * ROWS + r) * 65 + c];
            if (first == fin) Y[r * O + rb * 64 + c] = __float2bfloat16(v + (bias ? __bfloat162float(bias[rb * 64 + c]) : 0.f));
            else parts[((blockIdx.x + rb) * M + r) * 64 + c] = v;  // slot b + rb: distinct for every (block, row block) pair
        }
        if (first != fin && last_of(done + rb, fin - first + 1, last))
            for (int i = threadIdx.x; i < M * 64; i += 32 * NW) {
                int r = i / 64, c = i % 64;
                float v = 0.f;
                for (int64_t b = first; b <= fin; b++) v += __ldcg(parts + ((b + rb) * M + r) * 64 + c);
                Y[r * O + rb * 64 + c] = __float2bfloat16(v + (bias ? __bfloat162float(bias[rb * 64 + c]) : 0.f));
            }
        __syncthreads();  // red and last are reused
    }
}

// Y = X W^T for many tokens from the mma layout (a prompt), as a tiled
// GEMM with its work split between warps: a block is TM tokens by RBB row
// blocks of W (64 rows each). K goes in stages of 4 steps (64 columns), NB
// of them in flight in shared memory. PW warps produce: X's tile of a
// stage by cp.async (swizzled for ldmatrix), and W's 4 RBB steps of it
// decoded into B fragments; CW warps consume, each 64 tokens by 64 rows
// on the tensor cores, never waiting on a decode. Named barriers pass a stage's
// buffers from producers to consumers (full) and back (empty). A weight
// is decoded once for TM tokens. Blocks may split K (Y32: their parts,
// added in a fixed order by finish_kernel). A mixture of experts' prompt runs
// it (MOE), and a dense one but on GeForce Ada: there mma_gemm_sk_kernel, its
// blocks by stream-K.
constexpr int BIG_KK = 4;  // steps a stage
template <int CW, int PW, int NB, int RBB, int CR = 64> struct Big {
    static constexpr int THREADS = 32 * (CW + PW);
    static constexpr int TM = CR * CW / RBB;             // tokens a block: its consumers of 64 tokens by CR rows, 64 / CR a row block, by RBB row blocks
    static constexpr int A_BYTES = TM * 64 * 2;          // X's tile a stage
    static constexpr int B_UINT4 = RBB * BIG_KK * 4 * 32;  // W's fragments a stage: [row block][step][4][lane]
    static constexpr int SHARED = NB * (A_BYTES + B_UINT4 * 16);
    static constexpr int STEPS = RBB * BIG_KK / PW;      // W's steps a producer warp decodes a stage
    static constexpr int CHUNKS = TM * 8 / (32 * PW);    // X's 16-byte chunks a producer thread copies a stage
};

#if defined(__HIP_PLATFORM_AMD__)
// AMD: the copy made when asked (zeros past `bytes`, 0 or 16), so a commit and a wait have nothing to wait for; ldmatrix by shuffles (warp_emu.h). RDNA has
// no named barriers: the kernels that pass stages between producer and consumer warps by them (the prompt products mma_gemm_big and mma_gemm_sk, the
// A100's and Hopper's) trap there, and the host takes AMD's prompts through the step product instead (mma_linear_run).
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int bytes) { glyd_lds_put16(dst, bytes ? *(const uint4*)src : make_uint4(0, 0, 0, 0)); }
__device__ __forceinline__ void ldmatrix_x4(uint32_t r[4], uint32_t addr) { glyd_ldsm4(r, addr); }
template <int THREADS> __device__ __forceinline__ void bar_sync(int) { __builtin_trap(); }
template <int THREADS> __device__ __forceinline__ void bar_arrive(int) { __builtin_trap(); }
#else
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t r[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

// (the "memory" clobbers: shared and global memory written before a barrier is not kept in registers past it, nor read
// ahead of it)
template <int THREADS> __device__ __forceinline__ void bar_sync(int id) { asm volatile("bar.sync %0, %1;\n" ::"r"(id), "n"(THREADS) : "memory"); }
template <int THREADS> __device__ __forceinline__ void bar_arrive(int id) { asm volatile("bar.arrive %0, %1;\n" ::"r"(id), "n"(THREADS) : "memory"); }
#endif

// *p + 1 with release and acquire at the GPU's scope (a block's part made visible to the unit's last block, the others' to it): the value before.
__device__ __forceinline__ int atom_inc_acq_rel(int* p) {
#if defined(__HIP_PLATFORM_AMD__)
    return __hip_atomic_fetch_add(p, 1, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
#else
    int before;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;\n" : "=r"(before) : "l"(p) : "memory");
    return before;
#endif
}

__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }
// The same with the approximate reciprocal (2 instructions where the IEEE division is a dozen and a call): 2 ulp, past the bf16 the product is rounded to
// (the tiled kernel's epilogue makes 128 of them a thread, and its code was 17,000 instructions long: an instruction cache that missed 54% of the stalls there).
__device__ __forceinline__ float silu_fast(float x) { return __fdividef(x, 1.f + __expf(-x)); }
__device__ __forceinline__ float gelu_tanh(float x) { return 0.5f * x * (1.f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x))); }

// A mixture of experts' layer for mma_gemm_big_kernel (MOE; mma_moe_kernel has the plan and the rest): blockIdx.z a
// hit expert, M its pairs (a block TM of them); X's rows the tokens' (gather) or the pairs'; the output's as
// mma_moe_kernel's: with the gate's activation (act; RBB 2: blockIdx.y a row block of the gate's half, the
// consumers of row block 1 taking the up's, their sums met in shared memory) or times the weights (w, into Y32).
struct MoePairs {
    const int* plan;
    int E;
    int64_t k;
    int gather, act;
    const void* w;
    int wf32;
    int xahead, pf;  // the producers: X's tile of stage j + 1 copied before stage j is decoded (0: of stage j); the W lines of the stages pf ahead of the next asked of L2 (0: not)
    int skip = 1;    // the consumers multiply the m-tiles of 16 tokens that have pairs alone (an expert with 72 pairs has 8 in its tile's second 64: one m-tile of four)
};

template <class Fmt, int CW, int PW, int NB, int RBB, bool MOE = false>
__global__ void __launch_bounds__((Big<CW, PW, NB, RBB>::THREADS), 1) mma_gemm_big_kernel(Fmt f, int64_t O, int64_t K, int64_t M, int64_t stages_per_split, const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ Y, float* __restrict__ Y32, MoePairs moe) {
    using C = Big<CW, PW, NB, RBB>;
    int64_t e = 0, p0 = 0;  // MOE: the expert (its rows e O on of W's), its first pair in the plan's order
    const int* order = nullptr;
    if constexpr (MOE) {
        if ((int)blockIdx.z >= __ldg(moe.plan)) return;  // the whole block: no expert this far down the hits
        e = __ldg(moe.plan + 1 + blockIdx.z);
        p0 = __ldg(moe.plan + 1 + moe.E + blockIdx.z);
        M = __ldg(moe.plan + 2 + moe.E + blockIdx.z) - p0;
        if ((int64_t)blockIdx.x * C::TM >= M) return;
        order = moe.plan + 2 + 2 * moe.E;
    }
    extern __shared__ uint4 smem[];  // X's tiles [NB][TM rows][8 16-byte chunks, swizzled], then W's [NB][B_UINT4]
    __shared__ uint32_t tab[Fmt::kTable ? 256 : 1];
    if constexpr (Fmt::kTable) fill_groups(tab);
    __syncthreads();
    const uint32_t a_base = (uint32_t)__cvta_generic_to_shared(smem);
    uint4* Bs = smem + NB * C::A_BYTES / 16;
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int64_t KS = K / 16, RB = O / 64, pair = blockIdx.y;
    int64_t st0 = MOE ? 0 : (int64_t)blockIdx.z * stages_per_split, nst = MOE ? KS / BIG_KK : min(KS / BIG_KK, st0 + stages_per_split) - st0;
    // A block's tile of TM tokens: blockIdx.x; MOE, and every gridDim.x on while the expert has pairs (gridDim.x is
    // sized for T an expert, the most where no token lists an expert twice), the stages' buffers passed on between
    // tiles once every warp is done with them.
    for (int64_t tile = blockIdx.x;; tile += gridDim.x) {
        do {  // a tile's work (a break: this warp is done with it)
            if (warp >= CW) {
                // Producer p: X's chunks id = pt + 32 PW i of a stage's; W's steps STEPS p on of its 4 RBB (row block, step).
                int pt = tid - 32 * CW, p = warp - CW;
                typename Fmt::St st[C::STEPS];
                auto d_load = [&](int64_t j) {
#pragma unroll
                    for (int i = 0; i < C::STEPS; i++) {
                        int u = p * C::STEPS + i;  // of the stage's 4 RBB: row block u / 4, step u % 4
                        int64_t rb = min(pair * RBB + u / BIG_KK, RB - 1), s = min((st0 + j) * BIG_KK + u % BIG_KK, KS - 1);
                        if constexpr (MOE) {
                            if (moe.act) rb = pair + u / BIG_KK * (RB / 2);  // the gate's row block, then the up's
                            f.load(st[i], (e * RB + rb) * KS + s, lane);
                        } else {
                            f.load(st[i], rb * KS + s, lane);
                        }
                    }
                };
                int xr[MOE ? C::CHUNKS : 1];  // MOE: X's row of each of this thread's chunks (-1: past the pairs)
                if constexpr (MOE)
#pragma unroll
                    for (int i = 0; i < C::CHUNKS; i++) {
                        int64_t m = tile * C::TM + ((pt + 32 * PW * i) >> 3);
                        xr[i] = m < M ? (int)(moe.gather ? __ldg(order + p0 + m) / moe.k : p0 + m) : -1;
                    }
                auto copy_x = [&](int64_t j, int b) {  // X's tile of stage j into buffer b
                    uint32_t slot = a_base + (uint32_t)b * C::A_BYTES;
                    int64_t col = (st0 + j) * BIG_KK * 16;
#pragma unroll
                    for (int i = 0; i < C::CHUNKS; i++) {
                        int id = pt + 32 * PW * i, r = id >> 3, c = id & 7;
                        if constexpr (MOE) {
                            cp_async16(slot + r * 128 + ((c ^ (r & 7)) << 4), X + (xr[i] < 0 ? 0 : (int64_t)xr[i]) * K + col + c * 8, xr[i] < 0 ? 0 : 16);
                        } else {
                            int64_t m = tile * C::TM + r;
                            cp_async16(slot + r * 128 + ((c ^ (r & 7)) << 4), X + (m < M ? m : 0) * K + col + c * 8, m < M ? 16 : 0);
                        }
                    }
                };
                auto d_prefetch = [&](int64_t j) {  // this warp's steps of stage j, their lines to L2
#pragma unroll
                    for (int i = 0; i < C::STEPS; i++) {
                        int u = p * C::STEPS + i;
                        int64_t rb = min(pair * RBB + u / BIG_KK, RB - 1), s = min((st0 + j) * BIG_KK + u % BIG_KK, KS - 1);
                        if constexpr (MOE) {
                            if (moe.act) rb = pair + u / BIG_KK * (RB / 2);
                            f.prefetch((e * RB + rb) * KS + s, lane);
                        } else {
                            f.prefetch(rb * KS + s, lane);
                        }
                    }
                };
                const bool ahead = MOE && moe.xahead != 0;  // X's tile a stage ahead (NB >= 3: the buffer the consumers free after stage j + 1 - NB)
                if (MOE && moe.pf)
                    for (int jj = 1; jj <= moe.pf && jj < nst; jj++) d_prefetch(jj);
                d_load(0);
                if (ahead) {
                    copy_x(0, 0);
                    cp_commit();
                }
                for (int64_t j = 0; j < nst; j++) {
                    int b = (int)(j % NB);
                    if (ahead) {
                        if (j + 1 < nst) {
                            int b1 = (int)((j + 1) % NB);
                            if (j + 1 >= NB) bar_sync<C::THREADS>(1 + NB + b1);  // consumers are done with stage j + 1 - NB
                            copy_x(j + 1, b1);
                        }
                    } else {
                        if (j >= NB) bar_sync<C::THREADS>(1 + NB + b);  // consumers are done with stage j - NB
                        copy_x(j, b);
                    }
                    cp_commit();
                    typename Fmt::St cur[C::STEPS];
#pragma unroll
                    for (int i = 0; i < C::STEPS; i++) cur[i] = st[i];
                    if (j + 1 < nst) d_load(j + 1);
                    if (MOE && moe.pf && j + 1 + moe.pf < nst) d_prefetch(j + 1 + moe.pf);
#pragma unroll
                    for (int i = 0; i < C::STEPS; i++) {
                        uint32_t R[16];
                        uint4* slot = Bs + b * C::B_UINT4 + ((p * C::STEPS + i) * 4) * 32;
                        if constexpr (!Fmt::kTable) f.decode_xz(cur[i], lane, (uint32_t*)slot, R);  // (12-bit) many exceptions: through the step's slot, its scratch till then
                        else f.decode(cur[i], lane, (uint32_t*)slot, tab, R);  // (tiered) the step's slot its scratch till then
                        uint4* d = slot + lane;
#pragma unroll
                        for (int q = 0; q < 4; q++) d[q * 32] = make_uint4(R[4 * q], R[4 * q + 1], R[4 * q + 2], R[4 * q + 3]);
                    }
                    if (ahead) cp_wait<1>();  // stage j's tile (stage j + 1's may still come)
                    else cp_wait<0>();
                    bar_arrive<C::THREADS>(1 + b);  // stage j is ready
                }
                break;
            }
            // Consumer: tokens 64 (warp % (CW / RBB)) on of the block's TM, row block warp / (CW / RBB) of its RBB.
            int g = lane >> 2, t = lane & 3, wm = warp % (CW / RBB), rbl = warp / (CW / RBB);
            float acc[4][8][4];
#pragma unroll
            for (int mt = 0; mt < 4; mt++)
#pragma unroll
                for (int nn = 0; nn < 8; nn++)
#pragma unroll
                    for (int q = 0; q < 4; q++) acc[mt][nn][q] = 0.f;
            // ldmatrix rows: this lane's row of the 16 (0-7, then 8-15) and its 16-byte half of k.
            uint32_t a_row = (uint32_t)(wm * 64 + (lane & 7) + ((lane >> 3) & 1) * 8) * 128, a_half = lane >> 4, a_sw = lane & 7;
            // MOE: the m-tiles (of 16 tokens) of this warp's 64 that hold pairs: the others, past the expert's last pair, are not multiplied (the same for every lane of the warp)
            int nmt = 4;
            if constexpr (MOE) {
                if (moe.skip) {
                    const int64_t left = M - (tile * C::TM + wm * 64);
                    nmt = left >= 64 ? 4 : left <= 0 ? 0 : (int)((left + 15) >> 4);
                }
            }
            if constexpr (CW >= 8) {
                // Eight consumer warps (a thread's 128 sums in 168 registers): a step's fragments loaded as it is
                // multiplied, the B ones a pair of n-tiles at a time (each sum's products in the same order).
                for (int64_t j = 0; j < nst; j++) {
                    int bi = (int)(j % NB);
                    bar_sync<C::THREADS>(1 + bi);  // stage j is ready
                    uint32_t slot = a_base + (uint32_t)bi * C::A_BYTES;
                    const uint4* bsrc = Bs + bi * C::B_UINT4 + rbl * BIG_KK * 4 * 32 + lane;
#pragma unroll
                    for (int kk = 0; kk < BIG_KK; kk++) {
                        uint32_t a[4][4];
#pragma unroll
                        for (int mt = 0; mt < 4; mt++)
                            if (mt < nmt) ldmatrix_x4(a[mt], slot + a_row + mt * 16 * 128 + (((2 * kk + a_half) ^ a_sw) << 4));
#pragma unroll
                        for (int q = 0; q < 4; q++) {
                            uint4 b = bsrc[(kk * 4 + q) * 32];
#pragma unroll
                            for (int mt = 0; mt < 4; mt++) {
                                if (mt >= nmt) continue;
                                mma16816(acc[mt][2 * q], a[mt], b.x, b.y);
                                mma16816(acc[mt][2 * q + 1], a[mt], b.z, b.w);
                            }
                        }
                    }
                    if (j + NB < nst) bar_arrive<C::THREADS>(1 + NB + bi);  // done with stage j's buffers
                }
            } else {
                // Fragments double-buffered: a step's are asked for while the step
                // before multiplies; a stage's first, during the last of the one before.
                uint32_t fa[2][4][4];
                uint4 fb[2][4];
                auto frags = [&](int64_t j, int kk, int f) {
                    int bi = (int)(j % NB);
                    uint32_t slot = a_base + (uint32_t)bi * C::A_BYTES;
                    const uint4* bsrc = Bs + bi * C::B_UINT4 + rbl * BIG_KK * 4 * 32 + lane;
#pragma unroll
                    for (int mt = 0; mt < 4; mt++)
                        if (mt < nmt) ldmatrix_x4(fa[f][mt], slot + a_row + mt * 16 * 128 + (((2 * kk + a_half) ^ a_sw) << 4));
#pragma unroll
                    for (int q = 0; q < 4; q++) fb[f][q] = bsrc[(kk * 4 + q) * 32];
                };
                if (nst > 0) {
                    bar_sync<C::THREADS>(1);  // stage 0 is ready
                    frags(0, 0, 0);
                }
                for (int64_t j = 0; j < nst; j++) {
#pragma unroll
                    for (int kk = 0; kk < BIG_KK; kk++) {
                        int f = kk & 1;
                        if (kk + 1 < BIG_KK) frags(j, kk + 1, f ^ 1);
                        else if (j + 1 < nst) {
                            bar_sync<C::THREADS>(1 + (int)((j + 1) % NB));  // stage j + 1 is ready
                            frags(j + 1, 0, f ^ 1);
                        }
#pragma unroll
                        for (int mt = 0; mt < 4; mt++) {
                            if (mt >= nmt) continue;
                            mma16816(acc[mt][0], fa[f][mt], fb[f][0].x, fb[f][0].y);
                            mma16816(acc[mt][1], fa[f][mt], fb[f][0].z, fb[f][0].w);
                            mma16816(acc[mt][2], fa[f][mt], fb[f][1].x, fb[f][1].y);
                            mma16816(acc[mt][3], fa[f][mt], fb[f][1].z, fb[f][1].w);
                            mma16816(acc[mt][4], fa[f][mt], fb[f][2].x, fb[f][2].y);
                            mma16816(acc[mt][5], fa[f][mt], fb[f][2].z, fb[f][2].w);
                            mma16816(acc[mt][6], fa[f][mt], fb[f][3].x, fb[f][3].y);
                            mma16816(acc[mt][7], fa[f][mt], fb[f][3].z, fb[f][3].w);
                        }
                    }
                    if (j + NB < nst) bar_arrive<C::THREADS>(1 + NB + (int)(j % NB));  // done with stage j's buffers
                }
            }
            // C fragments: token rows g and g + 8 of each m-tile, columns 2t and 2t + 1 of each n-tile.
            if constexpr (MOE) {
                if (moe.act) {
                    // The up's sums (row block 1) into shared memory, the stages' buffers once every consumer is done with
                    // them; the gate's warps of the same tokens take theirs, in the same fragment order.
                    float* up = (float*)smem + wm * 128 * 32 + lane;
                    bar_sync<32 * CW>(1 + 2 * NB);
                    if (rbl == 1)
#pragma unroll
                        for (int mt = 0; mt < 4; mt++)
#pragma unroll
                            for (int nn = 0; nn < 8; nn++)
#pragma unroll
                                for (int q = 0; q < 4; q++) up[((mt * 8 + nn) * 4 + q) * 32] = acc[mt][nn][q];
                    bar_sync<32 * CW>(1 + 2 * NB);
                    if (rbl == 1) break;
                    // (the activation chosen once, not at each of the 128 products: SiLU's code and GELU's, one among the other at each, filled the instruction cache)
                    auto epilogue = [&](auto kind) GLYD_LAMBDA_INLINE {
#pragma unroll
                        for (int mt = 0; mt < 4; mt++)
#pragma unroll
                            for (int nn = 0; nn < 8; nn++) {
                                int64_t o = pair * 64 + nn * 8 + t * 2, at = e * O + o;  // the gate's row; the up's O / 2 on
#pragma unroll
                                for (int h = 0; h < 2; h++) {
                                    int64_t m = tile * C::TM + wm * 64 + mt * 16 + g + h * 8;
                                    if (m >= M) continue;
                                    float v[2];
#pragma unroll
                                    for (int c = 0; c < 2; c++) {
                                        float gate = acc[mt][nn][2 * h + c] + (bias ? __bfloat162float(bias[at + c]) : 0.f);
                                        float u = up[((mt * 8 + nn) * 4 + 2 * h + c) * 32] + (bias ? __bfloat162float(bias[at + O / 2 + c]) : 0.f);
                                        v[c] = (decltype(kind)::value == 1 ? silu_fast(gate) : gelu_tanh(gate)) * u;
                                    }
                                    *(__nv_bfloat162*)(Y + (p0 + m) * (O / 2) + o) = __floats2bfloat162_rn(v[0], v[1]);
                                }
                            }
                    };
                    if (moe.act == 1) epilogue(std::integral_constant<int, 1>());
                    else epilogue(std::integral_constant<int, 2>());
                    break;
                }
            }
            int64_t my_rb = pair * RBB + rbl;
            if (my_rb >= RB) break;
            int jr[MOE ? 4 : 1][2];  // MOE with Y32: this thread's 8 pairs' rows of Y32 and their weights, asked for together before the first is stored
            float wr[MOE ? 4 : 1][2];
            if constexpr (MOE) {
                if (Y32) {
#pragma unroll
                    for (int mt = 0; mt < 4; mt++)
#pragma unroll
                        for (int h = 0; h < 2; h++) {
                            int64_t m = tile * C::TM + wm * 64 + mt * 16 + g + h * 8;
                            jr[mt][h] = 0;
                            wr[mt][h] = 0.f;
                            if (m < M) {
                                jr[mt][h] = (int)__ldg(order + p0 + m);
                                wr[mt][h] = moe.wf32 ? __ldg((const float*)moe.w + jr[mt][h]) : __bfloat162float(__ldg((const __nv_bfloat16*)moe.w + jr[mt][h]));
                            }
                        }
                }
            }
#pragma unroll
            for (int mt = 0; mt < 4; mt++)
#pragma unroll
                for (int nn = 0; nn < 8; nn++) {
                    int64_t o = my_rb * 64 + nn * 8 + t * 2;
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        int64_t m = tile * C::TM + wm * 64 + mt * 16 + g + h * 8;
                        if (m >= M) continue;
                        float v0 = acc[mt][nn][2 * h], v1 = acc[mt][nn][2 * h + 1];
                        if constexpr (MOE) {
                            if (bias) {
                                v0 += __bfloat162float(bias[e * O + o]);
                                v1 += __bfloat162float(bias[e * O + o + 1]);
                            }
                            if (Y32) {
                                const int64_t j = jr[mt][h];
                                const float wj = wr[mt][h];
                                *(float2*)(Y32 + j * O + o) = make_float2(v0 * wj, v1 * wj);
                            } else {
                                *(__nv_bfloat162*)(Y + (p0 + m) * O + o) = __floats2bfloat162_rn(v0, v1);
                            }
                        } else if (Y32) {
                            *(float2*)(Y32 + ((int64_t)blockIdx.z * M + m) * O + o) = make_float2(v0, v1);
                        } else {
                            if (bias) {
                                v0 += __bfloat162float(bias[o]);
                                v1 += __bfloat162float(bias[o + 1]);
                            }
                            *(__nv_bfloat162*)(Y + m * O + o) = __floats2bfloat162_rn(v0, v1);
                        }
                    }
                }
        } while (0);
        if (!MOE || (tile + gridDim.x) * C::TM >= M) break;
        __syncthreads();
    }
}

// A prompt's product on GeForce Ada (not a mixture of experts'): mma_gemm_big_kernel's blocks, TM tokens by RBB row blocks, and its
// producer and consumer warps, as many blocks as the GPU holds at once, each taking an equal share of the units'
// stages in turn (a unit: a tile of tokens by a pair of row blocks; units by pair, then tile), its stages kept in
// flight from one unit to the next (stream-K: no wave part empty, no split of K summed by another kernel). A unit
// that blocks share: each writes its sums to a slot (parts: two a block, for the unit it starts in and the one it
// ends in, [2 gridDim.x][TM 64 RBB] floats in fragment order), and the last of them to finish adds the slots in
// block order (the same sum every run) with the bias (done: a counter a unit, zero before, reset by the last).
// A consumer multiplies 64 tokens by CR rows: 64 (4 consumers, 128 sums a thread) or 32, half a row block (8
// consumers, 64 sums a thread, which with 4 producers fit the 168 registers a thread of 12 warps: two consumers a
// scheduler, not one, each as lean: 2.2 instructions a product, and a weight still decoded once for TM tokens). The
// sums' order is a unit's stages', whatever CR: the same bits for the same TM and RBB. LIVE: the consumers multiply only
// the m-tiles that hold tokens (tiered_mid_run's shape, where it was measured; GeForce Ada's prompts as before, until measured there).
template <class Fmt, int CW, int PW, int NB, int RBB, int CR = 64, bool LIVE = false>
__global__ void __launch_bounds__((Big<CW, PW, NB, RBB, CR>::THREADS), 1) mma_gemm_sk_kernel(Fmt f, int64_t O, int64_t K, int64_t M, const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ Y, float* __restrict__ parts, int* __restrict__ done) {
    using C = Big<CW, PW, NB, RBB, CR>;
    constexpr int NN = CR / 8, TW = C::TM / 64, HR = 64 / CR;  // a consumer's n-tiles; consumers along the tokens; a row block's
    constexpr int64_t SLOT = C::TM * 64 * RBB;
    extern __shared__ uint4 smem[];  // X's tiles [NB][TM rows][8 16-byte chunks, swizzled], then W's [NB][B_UINT4]
    __shared__ uint32_t tab[Fmt::kTable ? 256 : 1];
    __shared__ int last;
    if constexpr (Fmt::kTable) fill_groups(tab);
    __syncthreads();
    const uint32_t a_base = (uint32_t)__cvta_generic_to_shared(smem);
    uint4* Bs = smem + NB * C::A_BYTES / 16;
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int64_t KS = K / 16, RB = O / 64, tiles = (M + C::TM - 1) / C::TM, nb = gridDim.x;
    const int S = (int)(KS / BIG_KK);  // stages a unit
    const int64_t total = tiles * ((RB + RBB - 1) / RBB) * S, it0 = blockIdx.x * total / nb, n = (blockIdx.x + 1) * total / nb - it0;
    const int64_t u0 = it0 / S;  // the unit this block starts in, at its stage s0 (then each stage's place kept in turn)
    const int s0 = (int)(it0 - u0 * S);
    if (warp >= CW) {
        // Producer p: X's chunks id = pt + 32 PW i of a stage's; W's steps STEPS p on of its 4 RBB (row block, step).
        int pt = tid - 32 * CW, p = warp - CW, s = s0;
        int64_t tile = u0 % tiles, pair = u0 / tiles;
        typename Fmt::St st[C::STEPS];
        auto d_load = [&](int64_t pair, int s) {
#pragma unroll
            for (int i = 0; i < C::STEPS; i++) {
                int u = p * C::STEPS + i;  // of the stage's 4 RBB: row block u / 4, step u % 4
                f.load(st[i], min(pair * RBB + u / BIG_KK, RB - 1) * KS + s * BIG_KK + u % BIG_KK, lane);
            }
        };
        if (n > 0) d_load(pair, s);
        for (int64_t j = 0; j < n; j++) {
            int ns = s + 1;  // the next stage's place
            int64_t ntile = tile, npair = pair;
            if (ns == S) {
                ns = 0;
                if (++ntile == tiles) ntile = 0, npair++;
            }
            int b = (int)(j % NB);
            if (j >= NB) bar_sync<C::THREADS>(1 + NB + b);  // consumers are done with stage j - NB
            uint32_t slot = a_base + (uint32_t)b * C::A_BYTES;
            int64_t col = (int64_t)s * BIG_KK * 16;
#pragma unroll
            for (int i = 0; i < C::CHUNKS; i++) {
                int id = pt + 32 * PW * i, r = id >> 3, c = id & 7;
                int64_t m = tile * C::TM + r;
                cp_async16(slot + r * 128 + ((c ^ (r & 7)) << 4), X + (m < M ? m : 0) * K + col + c * 8, m < M ? 16 : 0);
            }
            cp_commit();
            typename Fmt::St cur[C::STEPS];
#pragma unroll
            for (int i = 0; i < C::STEPS; i++) cur[i] = st[i];
            if (j + 1 < n) d_load(npair, ns);
#pragma unroll
            for (int i = 0; i < C::STEPS; i++) {
                uint32_t R[16];
                uint4* sl = Bs + b * C::B_UINT4 + ((p * C::STEPS + i) * 4) * 32;
                f.decode(cur[i], lane, (uint32_t*)sl, tab, R);  // (tiered) the step's slot its scratch till then
                uint4* d = sl + lane;
#pragma unroll
                for (int q = 0; q < 4; q++) d[q * 32] = make_uint4(R[4 * q], R[4 * q + 1], R[4 * q + 2], R[4 * q + 3]);
            }
            cp_wait<0>();
            bar_arrive<C::THREADS>(1 + b);  // stage j is ready
            s = ns, tile = ntile, pair = npair;
        }
        return;
    }
    // Consumer: tokens 64 wm on of a unit's TM, rows CR hf on of row block rbl of its RBB (consumers by token slice,
    // then half a row block, then row block).
    int g = lane >> 2, t = lane & 3, wm = warp % TW, hf = warp / TW % HR, rbl = warp / (TW * HR);
    float acc[4][NN][4];
#pragma unroll
    for (int mt = 0; mt < 4; mt++)
#pragma unroll
        for (int nn = 0; nn < NN; nn++)
#pragma unroll
            for (int q = 0; q < 4; q++) acc[mt][nn][q] = 0.f;
    uint32_t a_row = (uint32_t)(wm * 64 + (lane & 7) + ((lane >> 3) & 1) * 8) * 128, a_half = lane >> 4, a_sw = lane & 7;
    uint32_t fa[2][4][4];
    uint4 fb[2][NN / 2];
    auto frags = [&](int64_t j, int kk, int fi) {
        int bi = (int)(j % NB);
        uint32_t slot = a_base + (uint32_t)bi * C::A_BYTES;
        const uint4* bsrc = Bs + bi * C::B_UINT4 + rbl * BIG_KK * 4 * 32 + lane;
#pragma unroll
        for (int mt = 0; mt < 4; mt++) ldmatrix_x4(fa[fi][mt], slot + a_row + mt * 16 * 128 + (((2 * kk + a_half) ^ a_sw) << 4));
#pragma unroll
        for (int q = 0; q < NN / 2; q++) fb[fi][q] = bsrc[(kk * 4 + NN / 2 * hf + q) * 32];
    };
    auto next = [&](int64_t j, int fi) {  // stage j's first fragments, once it is ready
        bar_sync<C::THREADS>(1 + (int)(j % NB));
        frags(j, 0, fi);
    };
    if (n > 0) next(0, 0);
    int s = s0;
    for (int64_t j = 0, u = u0; j < n; j++) {
        bool end = s == S - 1 || j + 1 == n;  // the unit's last stage here: its sums out after it
        // LIVE: the m-tiles (16 tokens) of this warp's 64 that hold tokens of the unit's tile: the others, past M, are not multiplied
        // (their sums, zero, are never stored; the rows that are keep their products and their order)
        const int64_t left = M - u % tiles * C::TM - wm * 64;
        const int nmt = !LIVE || left >= 64 ? 4 : left <= 0 ? 0 : (int)((left + 15) >> 4);
#pragma unroll
        for (int kk = 0; kk < BIG_KK; kk++) {
            int fi = kk & 1;
            if (kk + 1 < BIG_KK) frags(j, kk + 1, fi ^ 1);
            else if (j + 1 < n) next(j + 1, fi ^ 1);
#pragma unroll
            for (int mt = 0; mt < 4; mt++)
                if (mt < nmt)
#pragma unroll
                    for (int q = 0; q < NN / 2; q++) {
                        mma16816(acc[mt][2 * q], fa[fi][mt], fb[fi][q].x, fb[fi][q].y);
                        mma16816(acc[mt][2 * q + 1], fa[fi][mt], fb[fi][q].z, fb[fi][q].w);
                    }
        }
        if (j + NB < n) bar_arrive<C::THREADS>(1 + NB + (int)(j % NB));  // done with stage j's buffers
        if (++s == S) s = 0;
        if (!end) continue;
        // The unit's stages here are done: its blocks, from the one its first stage is in to its last's; a block's
        // slot for it: 2b + 1 for the one it started before it (first, where it did), else 2b.
        int64_t first = block_of_step(u * S, nb, total), fin = block_of_step(u * S + S - 1, nb, total);
        bool out = first == fin;
        if (!out) {
            auto slot_of = [&](int64_t b) { return (float4*)(parts + (2 * b + (b == first && first * total / nb < u * S)) * SLOT) + tid; };
            float4* mine = slot_of(blockIdx.x);
#pragma unroll
            for (int mt = 0; mt < 4; mt++)
#pragma unroll
                for (int nn = 0; nn < NN; nn++) mine[(mt * NN + nn) * 32 * CW] = make_float4(acc[mt][nn][0], acc[mt][nn][1], acc[mt][nn][2], acc[mt][nn][3]);
            __threadfence();
            bar_sync<32 * CW>(1 + 2 * NB);
            if (tid == 0) {
                last = atomicAdd(done + u, 1) == fin - first;
                if (last) done[u] = 0;  // ready for the next product
            }
            bar_sync<32 * CW>(1 + 2 * NB);
            out = last;
            if (out) {  // every block's slot in block order (this one's read back too), NN loads in flight at a time
                __threadfence();
#pragma unroll
                for (int mt = 0; mt < 4; mt++)
#pragma unroll
                    for (int nn = 0; nn < NN; nn++)
#pragma unroll
                        for (int q = 0; q < 4; q++) acc[mt][nn][q] = 0.f;
#pragma unroll
                for (int mt = 0; mt < 4; mt++)  // (more at once: more registers than the kernel has)
                    for (int64_t b = first; b <= fin; b++) {
                        const float4* sl = slot_of(b) + mt * NN * 32 * CW;
#pragma unroll
                        for (int nn = 0; nn < NN; nn++) {
                            float4 w = __ldcg(sl + nn * 32 * CW);
                            acc[mt][nn][0] += w.x;
                            acc[mt][nn][1] += w.y;
                            acc[mt][nn][2] += w.z;
                            acc[mt][nn][3] += w.w;
                        }
                    }
            }
        }
        if (out) {
            int64_t tile = u % tiles, my_rb = u / tiles * RBB + rbl;
#pragma unroll
            for (int mt = 0; mt < 4; mt++)
#pragma unroll
                for (int nn = 0; nn < NN; nn++) {
                    int64_t o = my_rb * 64 + CR * hf + nn * 8 + t * 2;
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        int64_t m = tile * C::TM + wm * 64 + mt * 16 + g + h * 8;
                        if (my_rb >= RB || m >= M) continue;
                        float v0 = acc[mt][nn][2 * h], v1 = acc[mt][nn][2 * h + 1];
                        if (bias) {
                            v0 += __bfloat162float(bias[o]);
                            v1 += __bfloat162float(bias[o + 1]);
                        }
                        *(__nv_bfloat162*)(Y + m * O + o) = __floats2bfloat162_rn(v0, v1);
                    }
                }
        }
#pragma unroll
        for (int mt = 0; mt < 4; mt++)
#pragma unroll
            for (int nn = 0; nn < NN; nn++)
#pragma unroll
                for (int q = 0; q < 4; q++) acc[mt][nn][q] = 0.f;
        u++;
    }
}

// Many tokens on Hopper (sm_90a), the 12-bit layout: wgmma, Hopper's
// warpgroup tensor-core instruction, with W as its A operand, decoded
// straight into registers, and the tokens as B, read from shared memory
// (as Machete, a mixed-input GEMM for Hopper, does with 4-bit weights).
// The copies are the TMA's: one lane of a warp of its own hands the copy
// engine a stage at a time, 64 columns of a unit of WG row blocks (64 of
// W's rows each): each row block's 4 compressed steps (a run of 6144
// bytes), their exceptions, and X's tile through a tensor map, laid out in
// wgmma's 128-byte swizzle (a row's 16-byte chunk c at chunk c ^ (row mod
// 8)); each stage lands on an mbarrier, NS of them in flight. WG consumer
// warpgroups, a row block each: a warp decodes its 16 rows of the stage's
// 4 steps (its quarter of each: a word of codes and 8 low bytes a lane)
// into A fragments, then 4 wgmma of 64 rows by NT tokens by
// 16 columns. The work (units by stages) is split evenly over the blocks
// (stream-K, as mma_gemm_kernel's): a unit covered by several blocks is
// summed by the last to finish, in block order.
__device__ __forceinline__ uint64_t sw128_desc(uint32_t addr) {
    // Start address >> 4, stride between 8-row groups 1024 bytes, 128-byte swizzle (K-major).
    return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}

// wgmma is sm_90a's alone (sm_90's PTX has none, nor Blackwell's sm_100 and sm_120): its code and the TMA
// kernel's body compile for sm_90a only (__CUDA_ARCH_FEAT_SM90_ALL), a trap for the other targets, where the
// host never launches that kernel (compute capability 9.0 alone): run anyway (an H100 on a build for sm_90
// without the a, or on compute_80 PTX), it fails loudly rather than leave Y as it was.
__device__ __forceinline__ void wg_fence() {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#endif
}
template <int N> __device__ __forceinline__ void wg_wait() {  // until at most N groups of products are running
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
#endif
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ void mbar_init(uint32_t bar, int count) { asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(bar), "r"(count)); }
__device__ __forceinline__ void mbar_arrive(uint32_t bar) { asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.shared.b64 st, [%0];\n}\n" ::"r"(bar) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
    uint32_t ok;
    do {
#if __CUDA_ARCH__ >= 900
        asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n" : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
#else
        asm volatile("{\n.reg .pred p;\nmbarrier.test_wait.parity.shared.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n" : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
#endif
    } while (!ok);
}
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar), "r"(bytes) : "memory"); }
// bytes (a multiple of 16) from global memory to shared, counted on the mbarrier;
// once: read to be evicted from L2 first (data read once, in a stream).
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar, bool once = false) {
    if (once) {
        uint64_t pol;
        asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n" ::"r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(pol) : "memory");
    } else {
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
    }
}
// A 2-D tile through a tensor map (coordinates innermost first).
__device__ __forceinline__ void tma_2d(uint32_t dst, const CUtensorMap* map, int c0, int c1, uint32_t bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" ::"r"(dst), "l"((uint64_t)map), "r"(c0), "r"(c1), "r"(bar) : "memory");
}

#endif
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
// A stage's 4 wgmma m64nNk16 (64 columns), A (W's 64 rows by 16 columns each) from registers, B (N tokens) from
// shared memory, then their commit: D = A B + (acc ? D : 0) for the first, D += A B for the rest.
__device__ __forceinline__ void wgmma4_rs(float (&d)[8], const uint32_t (&a)[4][4], const uint64_t (&db)[4], int acc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %28, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %24, p, 1, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%12, %13, %14, %15}, %25, 1, 1, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%16, %17, %18, %19}, %26, 1, 1, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%20, %21, %22, %23}, %27, 1, 1, 1, 0;\n"
                 "wgmma.commit_group.sync.aligned;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                 : "r"(a[0][0]), "r"(a[0][1]), "r"(a[0][2]), "r"(a[0][3]), "r"(a[1][0]), "r"(a[1][1]), "r"(a[1][2]), "r"(a[1][3]), "r"(a[2][0]), "r"(a[2][1]), "r"(a[2][2]), "r"(a[2][3]), "r"(a[3][0]), "r"(a[3][1]), "r"(a[3][2]), "r"(a[3][3]), "l"(db[0]), "l"(db[1]), "l"(db[2]), "l"(db[3]), "r"(acc));
}
__device__ __forceinline__ void wgmma4_rs(float (&d)[16], const uint32_t (&a)[4][4], const uint64_t (&db)[4], int acc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %36, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %32, p, 1, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%20, %21, %22, %23}, %33, 1, 1, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%24, %25, %26, %27}, %34, 1, 1, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%28, %29, %30, %31}, %35, 1, 1, 1, 0;\n"
                 "wgmma.commit_group.sync.aligned;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
                 : "r"(a[0][0]), "r"(a[0][1]), "r"(a[0][2]), "r"(a[0][3]), "r"(a[1][0]), "r"(a[1][1]), "r"(a[1][2]), "r"(a[1][3]), "r"(a[2][0]), "r"(a[2][1]), "r"(a[2][2]), "r"(a[2][3]), "r"(a[3][0]), "r"(a[3][1]), "r"(a[3][2]), "r"(a[3][3]), "l"(db[0]), "l"(db[1]), "l"(db[2]), "l"(db[3]), "r"(acc));
}
__device__ __forceinline__ void wgmma4_rs(float (&d)[32], const uint32_t (&a)[4][4], const uint64_t (&db)[4], int acc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %52, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %48, p, 1, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%36, %37, %38, %39}, %49, 1, 1, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%40, %41, %42, %43}, %50, 1, 1, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%44, %45, %46, %47}, %51, 1, 1, 1, 0;\n"
                 "wgmma.commit_group.sync.aligned;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
                 : "r"(a[0][0]), "r"(a[0][1]), "r"(a[0][2]), "r"(a[0][3]), "r"(a[1][0]), "r"(a[1][1]), "r"(a[1][2]), "r"(a[1][3]), "r"(a[2][0]), "r"(a[2][1]), "r"(a[2][2]), "r"(a[2][3]), "r"(a[3][0]), "r"(a[3][1]), "r"(a[3][2]), "r"(a[3][3]), "l"(db[0]), "l"(db[1]), "l"(db[2]), "l"(db[3]), "r"(acc));
}
__device__ __forceinline__ void wgmma4_rs(float (&d)[48], const uint32_t (&a)[4][4], const uint64_t (&db)[4], int acc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %68, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47}, {%48, %49, %50, %51}, %64, p, 1, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47}, {%52, %53, %54, %55}, %65, 1, 1, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47}, {%56, %57, %58, %59}, %66, 1, 1, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47}, {%60, %61, %62, %63}, %67, 1, 1, 1, 0;\n"
                 "wgmma.commit_group.sync.aligned;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47])
                 : "r"(a[0][0]), "r"(a[0][1]), "r"(a[0][2]), "r"(a[0][3]), "r"(a[1][0]), "r"(a[1][1]), "r"(a[1][2]), "r"(a[1][3]), "r"(a[2][0]), "r"(a[2][1]), "r"(a[2][2]), "r"(a[2][3]), "r"(a[3][0]), "r"(a[3][1]), "r"(a[3][2]), "r"(a[3][3]), "l"(db[0]), "l"(db[1]), "l"(db[2]), "l"(db[3]), "r"(acc));
}
__device__ __forceinline__ void wgmma4_rs(float (&d)[56], const uint32_t (&a)[4][4], const uint64_t (&db)[4], int acc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %76, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n112k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55}, {%56, %57, %58, %59}, %72, p, 1, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n112k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55}, {%60, %61, %62, %63}, %73, 1, 1, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n112k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55}, {%64, %65, %66, %67}, %74, 1, 1, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n112k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55}, {%68, %69, %70, %71}, %75, 1, 1, 1, 0;\n"
                 "wgmma.commit_group.sync.aligned;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55])
                 : "r"(a[0][0]), "r"(a[0][1]), "r"(a[0][2]), "r"(a[0][3]), "r"(a[1][0]), "r"(a[1][1]), "r"(a[1][2]), "r"(a[1][3]), "r"(a[2][0]), "r"(a[2][1]), "r"(a[2][2]), "r"(a[2][3]), "r"(a[3][0]), "r"(a[3][1]), "r"(a[3][2]), "r"(a[3][3]), "l"(db[0]), "l"(db[1]), "l"(db[2]), "l"(db[3]), "r"(acc));
}
__device__ __forceinline__ void wgmma4_rs(float (&d)[64], const uint32_t (&a)[4][4], const uint64_t (&db)[4], int acc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %84, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %80, p, 1, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%68, %69, %70, %71}, %81, 1, 1, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%72, %73, %74, %75}, %82, 1, 1, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%76, %77, %78, %79}, %83, 1, 1, 1, 0;\n"
                 "wgmma.commit_group.sync.aligned;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
                 : "r"(a[0][0]), "r"(a[0][1]), "r"(a[0][2]), "r"(a[0][3]), "r"(a[1][0]), "r"(a[1][1]), "r"(a[1][2]), "r"(a[1][3]), "r"(a[2][0]), "r"(a[2][1]), "r"(a[2][2]), "r"(a[2][3]), "r"(a[3][0]), "r"(a[3][1]), "r"(a[3][2]), "r"(a[3][3]), "l"(db[0]), "l"(db[1]), "l"(db[2]), "l"(db[3]), "r"(acc));
}
// The same 4 products with A (W's 64 rows by 16 columns each) from shared memory too, by descriptor (da).
__device__ __forceinline__ void wgmma4_ss(float (&d)[8], const uint64_t (&da)[4], const uint64_t (&db)[4], int acc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %16, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %12, p, 1, 1, 0, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %9, %13, 1, 1, 1, 0, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %10, %14, 1, 1, 1, 0, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %11, %15, 1, 1, 1, 0, 0;\n"
                 "wgmma.commit_group.sync.aligned;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                 : "l"(da[0]), "l"(da[1]), "l"(da[2]), "l"(da[3]), "l"(db[0]), "l"(db[1]), "l"(db[2]), "l"(db[3]), "r"(acc));
}
__device__ __forceinline__ void wgmma4_ss(float (&d)[16], const uint64_t (&da)[4], const uint64_t (&db)[4], int acc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %24, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %20, p, 1, 1, 0, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %17, %21, 1, 1, 1, 0, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %18, %22, 1, 1, 1, 0, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %19, %23, 1, 1, 1, 0, 0;\n"
                 "wgmma.commit_group.sync.aligned;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
                 : "l"(da[0]), "l"(da[1]), "l"(da[2]), "l"(da[3]), "l"(db[0]), "l"(db[1]), "l"(db[2]), "l"(db[3]), "r"(acc));
}
__device__ __forceinline__ void wgmma4_ss(float (&d)[32], const uint64_t (&da)[4], const uint64_t (&db)[4], int acc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %40, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %36, p, 1, 1, 0, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %33, %37, 1, 1, 1, 0, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %34, %38, 1, 1, 1, 0, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %35, %39, 1, 1, 1, 0, 0;\n"
                 "wgmma.commit_group.sync.aligned;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
                 : "l"(da[0]), "l"(da[1]), "l"(da[2]), "l"(da[3]), "l"(db[0]), "l"(db[1]), "l"(db[2]), "l"(db[3]), "r"(acc));
}
#endif

// Shared by the TMA kernel and mma12_mid_kernel. A warp's 16 rows (16w to 16w + 15) of a row block's 4
// steps (a stage), from shared memory into mma's A fragments, a register set a step: of each lane, the
// codes' word w and the low bytes 8 (w mod 2) to 8 (w mod 2) + 7 of half w / 2 (its rows g and g + 8);
// then the stage's run of exceptions, one run for the 4 steps (se: its copy, whose first entry is ea; ea <
// 0: read from global memory; eb: exc_base at the 4 steps and the next). The lanes take the run's entries
// 32 at a time, each in this warp's rows setting its byte in the warp's scratch (xw: a lane's 8 words, 256
// in all, zero between stages), which every lane then XORs into its high bytes and clears: a stage's cost
// grows by a pass per 32 of its exceptions, not by an entry per lane (a few layers' matrices have dozens a
// stage; past the stage's copy, 256 entries in the TMA kernel and 128 in mma12_mid_kernel, they are read
// from global memory).
__device__ __forceinline__ void decode12_rows(const Nib& f, const uint8_t* sp, const uint32_t* se, const int (&eb)[5], int ea, int lane, int w, uint32_t* xw, uint32_t (&A)[4][4]) {
    uint32_t nw[4];
    uint2 sw[4];
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {  // the stage's loads first, 8 in flight at once
        const uint8_t* q = sp + kk * STEP12;
        nw[kk] = *(const uint32_t*)(q + 16 * lane + 4 * w);
        sw[kk] = *(const uint2*)(q + 512 + 512 * (w >> 1) + 16 * lane + 8 * (w & 1));
    }
    uint32_t ew[8];  // step kk's high-byte words 2w and 2w + 1 of each lane: ew[2kk], ew[2kk + 1]
#pragma unroll
    for (int q = 0; q < 8; q++) ew[q] = Nib::high1(nw[q >> 1], f.hb4, q & 1);
    // Entry k is step kk's while eb[kk] <= k < eb[kk + 1]; weight i of lane (x >> 5) mod 32, byte i mod 4 of its
    // word 2 kk + (i / 4 mod 2) here where i / 8 is this warp's.
    if (eb[4] > eb[0]) {
        bool mine = false;
        for (int k = eb[0] + lane; k < eb[4]; k += 32) {
            uint32_t x = ea >= 0 ? se[k - ea] : __ldg(f.exc + k), i = x & 31;
            if ((int)(i >> 3) == w) {
                uint32_t kk = (k >= eb[1]) + (k >= eb[2]) + (k >= eb[3]);
                ((uint8_t*)xw)[(((x >> 5) & 31) * 8 + 2 * kk + ((i >> 2) & 1)) * 4 + (i & 3)] = (uint8_t)(x >> 16);
                mine = true;
            }
        }
        if (__any_sync(FULL, mine)) {
            __syncwarp();
            uint4* xl = (uint4*)(xw + 8 * lane);
            uint4 a = xl[0], b = xl[1];
            ew[0] ^= a.x, ew[1] ^= a.y, ew[2] ^= a.z, ew[3] ^= a.w, ew[4] ^= b.x, ew[5] ^= b.y, ew[6] ^= b.z, ew[7] ^= b.w;
            xl[0] = xl[1] = make_uint4(0u, 0u, 0u, 0u);
            __syncwarp();
        }
    }
    // A fragments: rows g and g + 8 (words 2w and 2w + 1), columns 2t and 8 + 2t (bytes 0-1 and 2-3).
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
        A[kk][0] = __byte_perm(sw[kk].x, ew[2 * kk], 0x5140);
        A[kk][1] = __byte_perm(sw[kk].y, ew[2 * kk + 1], 0x5140);
        A[kk][2] = __byte_perm(sw[kk].x, ew[2 * kk], 0x7362);
        A[kk][3] = __byte_perm(sw[kk].y, ew[2 * kk + 1], 0x7362);
    }
}

// exc_base at a stage's 4 steps and the next, for each of a unit's WG row blocks.
template <int WG>
__device__ __forceinline__ void stage_bounds(const Nib& f, int64_t RB, int64_t KS, int p, int s, int (&e)[5 * WG]) {
#pragma unroll
    for (int r = 0; r < WG; r++)
#pragma unroll
        for (int i = 0; i < 5; i++) e[5 * r + i] = __ldg(f.exc_base + min((int64_t)WG * p + r, RB - 1) * KS + 4 * s + i);
}

// A row block's exceptions for a stage as a copy: na entries from entry a (a multiple of 4), the run
// widened to 16 bytes each way (pack_mma12 pads exc); na = -1 past cap bytes (read from global memory then).
__device__ __forceinline__ void exc_copy(int lo, int hi, int cap, int& a, int& na) {
    a = lo & ~3;
    na = hi > lo ? ((hi + 3) & ~3) - a : 0;
    if (4 * na > cap) na = -1;
}

// A unit's sum out (its WG row blocks, row unit pr, NT tokens m0 on), r in each thread's fragment order (rows
// 16w + g (+ 8) of row block wg, tokens 8jn + 2t (+ 1): r[4jn + 2h + c]): to Y when the block covers the unit,
// else to its slot, and the last of the unit's blocks to finish adds the slots in block order (the same result
// every run). A block's slots: 2b for its first unit, 2b + 1 for its last (only those two may be shared). The WG
// warpgroups' threads (ct), named barrier 1.
template <int NT, int WG>
__device__ __forceinline__ void sum_out12(float (&r)[NT / 2], int64_t p, int64_t pr, int64_t m0, int S, int64_t nb, int64_t U, float* parts, int* done, int& last, int ct, int64_t O, int64_t M, const __nv_bfloat16* bias, __nv_bfloat16* Y) {
    constexpr int R = 64 * WG;
    int wg = ct >> 7, w = (ct >> 5) & 3, g = (ct & 31) >> 2, t = ct & 3;
    int64_t first = block_of_step(p * S, nb, U), fin = block_of_step(p * S + S - 1, nb, U);
    // Block b's slot for the unit: 2b where the unit is b's first (b starts in it: every block of it but the first,
    // which starts in it only at its start), else 2b + 1.
    auto slot = [&](int64_t b) { return parts + (2 * b + (b == first && b * U / nb != p * S)) * (NT * R); };
    if (first != fin) {
        // Slots [NT / 8 float4s][threads]: each thread's r in order, coalesced.
        float4* pp = (float4*)slot(blockIdx.x);
#pragma unroll
        for (int q = 0; q < NT / 8; q++) pp[q * 128 * WG + ct] = make_float4(r[4 * q], r[4 * q + 1], r[4 * q + 2], r[4 * q + 3]);
        // The warpgroups' slot written (the barrier), one thread's count with release and acquire: the others' parts
        // seen by the last, whose threads read them after the second barrier (fences are cumulative; a fence a thread
        // had held each up till its own writes landed).
        bar_sync<128 * WG>(1);
        if (ct == 0) {
            int before = atom_inc_acq_rel(done + p);
            last = before == fin - first;
            if (last) done[p] = 0;  // ready for the next product
        }
        bar_sync<128 * WG>(1);
        if (last) {
#pragma unroll
            for (int i = 0; i < NT / 2; i++) r[i] = 0.f;
            for (int64_t b = first; b <= fin; b++) {
                const float4* bp = (const float4*)slot(b);
#pragma unroll
                for (int q = 0; q < NT / 8; q++) {
                    float4 v = __ldcg(bp + q * 128 * WG + ct);
                    r[4 * q] += v.x;
                    r[4 * q + 1] += v.y;
                    r[4 * q + 2] += v.z;
                    r[4 * q + 3] += v.w;
                }
            }
        }
    }
    if (first == fin || last) {
#pragma unroll
        for (int jn = 0; jn < NT / 8; jn++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                int64_t o = ((int64_t)WG * pr + wg) * 64 + 16 * w + g + 8 * h;
                float bo = bias && o < O ? __bfloat162float(bias[o]) : 0.f;
#pragma unroll
                for (int c = 0; c < 2; c++) {
                    int64_t m = m0 + 8 * jn + 2 * t + c;
                    if (o < O && m < M) Y[m * O + o] = __float2bfloat16(r[4 * jn + 2 * h + c] + bo);
                }
            }
    }
}

template <int NT, int WG> struct Tma12 {
    static constexpr int THREADS = 128 * WG + 32;  // WG consumer warpgroups (a warpgroup's first warp a multiple of 4), the TMA warp
    static constexpr int R = 64 * WG;              // W's rows a unit (a row block a warpgroup)
    static constexpr int XB = NT * 128;            // X's tile a stage: NT tokens by 64 columns
    static constexpr int CB = 4 * (int)STEP12, EB = 1024, BB = 32;  // a row block's steps a stage, its exceptions (up to 256), their bounds
    static constexpr int RBB = CB + EB + BB;
    static constexpr int SLOT = (XB + WG * RBB + 1023) / 1024 * 1024;
    static constexpr int NS = 200 * 1024 / SLOT < 8 ? 200 * 1024 / SLOT : 8;  // stages in flight
    static constexpr int SHARED = NS * SLOT + 1024 + 16 * NS + 4096 * WG;   // alignment, the full and empty barriers, the warps' exceptions' scratch
};

template <int NT, int WG>
__global__ void __launch_bounds__((Tma12<NT, WG>::THREADS), 1) mma12_tma_kernel(const __grid_constant__ CUtensorMap xmap, Nib f, int64_t O, int64_t K, int64_t M, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ Y, float* __restrict__ parts, int* __restrict__ done) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    using C = Tma12<NT, WG>;
    extern __shared__ __align__(1024) uint8_t smem_raw[];  // NS slots [X tile | a row block's steps, exceptions, bounds | the other's], the barriers
    __shared__ int last;
    uint32_t raw = (uint32_t)__cvta_generic_to_shared(smem_raw), base = (raw + 1023) & ~1023u;
    uint8_t* gbase = smem_raw + (base - raw);
    uint32_t full = base + C::NS * C::SLOT, empty = full + 8 * C::NS;
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    // Units: RU row units (WG row blocks) by TC chunks of NT tokens, unit p chunk p / RU and row unit p mod RU; the
    // work, units by stages, split evenly over the blocks. Chunk by chunk: the blocks at work at once share few row
    // units, a chunk's blocks each of theirs at about the same time (read from memory about once, not once a chunk).
    int TC = (int)((M + NT - 1) / NT), RU = (int)((O / 64 + WG - 1) / WG);
    int64_t KS = K / 16, RB = O / 64, U = (int64_t)RU * TC * (K / 64), nb = gridDim.x;
    int S = (int)(K / 64);
    int64_t u0 = blockIdx.x * U / nb;
    int n = (int)((blockIdx.x + 1) * U / nb - u0);  // the block's stages, u0 on: unit u / S, stage u mod S
    int p0 = (int)(u0 / S), s0 = (int)(u0 - (int64_t)p0 * S);
    if (tid == 0) {
        for (int i = 0; i < C::NS; i++) {
            mbar_init(full + 8 * i, 1);
            mbar_init(empty + 8 * i, 4 * WG);
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();
    if (warp >= 4 * WG) {
        // The TMA warp: lane j mod 8 issues stage j. Stages go in batches of 8,
        // lane l's bounds for stage l of the batch loaded while the batch before
        // was issued (a load pends for the whole warp: one register set a lane,
        // reloaded by one lane a stage, would wait out every load).
        int e[5 * WG], en[5 * WG];  // exc_base at the stage's 4 steps and the next, each row block: this batch's, the next's
        auto ahead = [&](int p, int s, int k) {  // the stage k after (p, s)
            s += k;
            stage_bounds<WG>(f, RB, KS, (p + s / S) % RU, s % S, en);
        };
        if (lane < 8 && lane < n) ahead(p0, s0, lane);
        // (unit p's row unit pr and chunk pc kept as p goes: no division a stage, where a stage is short)
        for (int j = 0, p = p0, s = s0, pr = p0 % RU, pc = p0 / RU; j < n; j++) {
            if ((j & 7) == 0) {
#pragma unroll
                for (int i = 0; i < 5 * WG; i++) e[i] = en[i];
                if (lane < 8 && j + 8 + lane < n) ahead(p, s, 8 + lane);
            }
            if (lane == (j & 7)) {
                int sl = j % C::NS;
                uint32_t fb = full + 8 * sl, xs = base + sl * C::SLOT;
                if (j >= C::NS) mbar_wait(empty + 8 * sl, (j / C::NS - 1) & 1);  // the consumers are done with stage j - NS
                uint32_t bytes = C::XB + WG * C::CB;
                int a[WG], na[WG];
#pragma unroll
                for (int r = 0; r < WG; r++) {
                    exc_copy(e[5 * r], e[5 * r + 4], C::EB, a[r], na[r]);
                    bytes += na[r] > 0 ? 4 * na[r] : 0;
                    int* bs = (int*)(gbase + sl * C::SLOT + C::XB + r * C::RBB + C::CB + C::EB);
#pragma unroll
                    for (int i = 0; i < 5; i++) bs[i] = e[5 * r + i];
                    bs[5] = na[r] < 0 ? -1 : a[r];
                }
                mbar_expect_tx(fb, bytes);
                tma_2d(xs, &xmap, (int)(s * 64), pc * NT, fb);
#pragma unroll
                for (int r = 0; r < WG; r++) {
                    uint32_t cs = xs + C::XB + r * C::RBB;
                    bulk_g2s(cs, f.data + (min((int64_t)WG * pr + r, RB - 1) * KS + 4 * s) * STEP12, C::CB, fb, TC == 1);  // (read once but where the chunks share it)
                    if (na[r] > 0) bulk_g2s(cs + C::CB, f.exc + a[r], 4 * na[r], fb);
                }
            }
            __syncwarp();
            if (++s == S) {
                s = 0, p++;
                if (++pr == RU) pr = 0, pc++;
            }
        }
        return;
    }
    // Consumer warpgroup wg: row block WG (p mod RU) + wg of each unit p; warp w its rows 16w to 16w + 15.
    // Two sets of A registers, used in turn: a stage's products may still run while the next decodes.
    int ct = tid, wg = ct >> 7, w = (ct >> 5) & 3;
    uint32_t* xw = (uint32_t*)(gbase + C::NS * C::SLOT + 16 * C::NS) + 256 * warp;  // this warp's exceptions' scratch
    ((uint4*)xw)[2 * lane] = ((uint4*)xw)[2 * lane + 1] = make_uint4(0u, 0u, 0u, 0u);
    __syncwarp();
    float d[NT / 2] = {};  // rows 16w + g (+ 8), tokens 8jn + 2t (+ 1): d[4jn + 2h + c]; set by a unit's first product
    // (zeroed all the same: left unset, CUDA 12.8's ptxas saw it defined inside the loop and serialized every wgmma,
    // its C7515, where CUDA 13's did not; on an H100 SXM the CUDA 12 library's products took a median 4% longer so)
    uint32_t A0[4][4], A1[4][4];
    int held = -1;  // the slot whose products may still be running
    auto release = [&](int sl) {  // the slot is free for stage j + NS (lane 0 arrives, by predicate: no branch among the products)
        __syncwarp();
        asm volatile("{\n.reg .pred p;\nsetp.eq.u32 p, %1, 0;\n@p mbarrier.arrive.shared::cta.b64 _, [%0];\n}\n" ::"r"(empty + 8 * sl), "r"(lane) : "memory");
    };
    auto keep_d = [&]() {  // d stays in its registers while the products run
#pragma unroll
        for (int i = 0; i < NT / 2; i++) asm volatile("" : "+f"(d[i])::"memory");
    };
    auto keep = [&](uint32_t(&A)[4][4]) {  // A is held until its products are done
#pragma unroll
        for (int kk = 0; kk < 4; kk++)
#pragma unroll
            for (int i = 0; i < 4; i++) asm volatile("" : "+r"(A[kk][i])::"memory");
    };
    auto stage = [&](int j, bool first, uint32_t(&A)[4][4], uint32_t(&Aprev)[4][4]) {
        int sl = j % C::NS;
        mbar_wait(full + 8 * sl, (j / C::NS) & 1);
        const uint8_t* sp = gbase + sl * C::SLOT + C::XB + wg * C::RBB;
        const int* bs = (const int*)(sp + C::CB + C::EB);
        int eb[5];
#pragma unroll
        for (int i = 0; i < 5; i++) eb[i] = bs[i];
        decode12_rows(f, sp, (const uint32_t*)(sp + C::CB), eb, bs[5], lane, w, xw, A);
        uint32_t xs = base + sl * C::SLOT;
        uint64_t desc[4];
#pragma unroll
        for (int kk = 0; kk < 4; kk++) desc[kk] = sw128_desc(xs + kk * 32);
        keep_d();
        wg_fence();
        wgmma4_rs(d, A, desc, !first);
        wg_wait<1>();  // the stage before's products are done: its A set and its slot are free
        keep(Aprev);
        keep_d();
        if (held >= 0) release(held);
        held = sl;
    };
    // A unit's stages, then its sum out: no branch touches d while products run. The stages in pairs, an odd
    // one after them: every path gives each set of A registers to the products before it is decoded into
    // again (a pair's second stage skipped inside the loop had ptxas serialize every product, its C7513).
    for (int j = 0, p = p0, s = s0; j < n; p++, s = 0) {
        int len = min(S - s, n - j), k = 0;
        for (; k + 1 < len; k += 2) {
            stage(j + k, k == 0, A0, A1);
            stage(j + k + 1, false, A1, A0);
        }
        if (k < len) stage(j + k, k == 0, A0, A1);
        wg_wait<0>();
        keep(A0);
        keep(A1);
        keep_d();
        if (held >= 0) release(held);
        held = -1;
        sum_out12<NT, WG>(d, p, p % RU, (int64_t)(p / RU) * NT, S, nb, U, parts, done, last, ct, O, M, bias, Y);  // (d, summed in place: the next unit's first product sets it)
        j += len;
    }
#elif defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    __trap();  // sm_90a code, launched from a build without it (or on AMD)
#endif
}

// The tiered layout's products of 1 to 64 tokens on Hopper (sm_90a), by the TMA kernel's plan (mma12_tma_kernel). A
// TMA warp copies each stage (64 columns) into a ring of NS slots, landing on an mbarrier: X's tile of NT tokens by its
// tensor map (128-byte swizzle) and, for each of the WG row blocks, its 4 steps' fixed parts (5,120 bytes), the bytes of
// their blocks from 128 before the first one's end to 128 past the last one's start (all that the short form of a
// step's decode reads: its window and its tail), up to BB of them (more: read from global memory instead), and
// block_base at the 4 steps and the next. Consumer warpgroups, a row block each: warp kk decodes step kk (64 rows by 16
// columns) from shared memory (a step with the flag by the general decode, its block from global memory), the 4 warps
// trade the decoded fragments through shared memory (warp w then holds rows 16w to 16w + 15 of all 4 steps: wgmma's A
// registers), and wgmma multiplies them by X's tile. A weight is decoded once for NT tokens, and no global load of the
// decode waits in a consumer: the TMA warp makes them stages ahead. BPS: blocks an SM (the registers' bound). SS: the
// warps put their steps' fragments as one tile of the stage's 64 rows by 64 columns in the 128-byte swizzle of X's
// tile, and wgmma reads A from it as it reads X (no A registers); else the trade (A in registers).
constexpr int TT_BB = 2048;  // a row block's blocks' bytes a stage, at most, in shared memory
template <int NT, int WG, int BPS> struct TmaT {
    static constexpr int THREADS = 128 * WG + 32;  // WG consumer warpgroups, the TMA warp
    static constexpr int R = 64 * WG;              // W's rows a unit (a row block a warpgroup)
    static constexpr int XB = NT * 128;            // X's tile a stage: NT tokens by 64 columns
    static constexpr int CB = 4 * (int)STEP_BYTES, BB = TT_BB, HB = 32;  // a row block's 4 steps' fixed parts, their blocks' bytes, the bounds
    static constexpr int RBB = CB + BB + HB;
    static constexpr int SLOT = (XB + WG * RBB + 1023) / 1024 * 1024;
    static constexpr int XCH = 2 * 16 * 32 * 16;                       // a warpgroup's trade buffers: two (a stage's parity) of [4 steps][4 row groups][32 lanes] uint4, or of the tile [64 rows][128 bytes]
    static constexpr int FIXED = 1024 + WG * XCH + 4 * WG * S2_BYTES;  // the alignment, the trade buffers, the warps' scratch
    static constexpr int ROOM = 228 * 1024 / BPS - 2304;  // (an SM's 228 KB, less a block's 1 KB and its static tab)
    static constexpr int NS0 = (ROOM - FIXED) / (SLOT + 16);
    static constexpr int NS = NS0 < 8 ? NS0 : 8;  // stages in flight
    static constexpr int SHARED = NS * SLOT + 16 * NS + FIXED;
    static_assert(NS >= 2, "a ring of at least two stages");
};

// block_base at a stage's 4 steps and the next, for each of a unit's WG row blocks (bit 31: a step's flag).
template <int WG>
__device__ __forceinline__ void tiered_bounds(const Tiered& f, int64_t RB, int64_t KS, int p, int s, int (&e)[5 * WG]) {
#pragma unroll
    for (int r = 0; r < WG; r++)
#pragma unroll
        for (int i = 0; i < 5; i++) e[5 * r + i] = __ldg(f.block_base + min((int64_t)WG * p + r, RB - 1) * KS + 4 * s + i);
}

// Step w of a stage's row block from its copy in shared memory (sp: the 4 steps' fixed parts, their blocks' bytes from
// bs[5] on, the bounds bs) into a lane's B fragments of its 8 n-tiles, as the step kernel decodes it: the short form, a
// step with the flag by the general decode. bs[5] < 0: the blocks' bytes were not copied (read from global memory).
__device__ __forceinline__ void tiered_step_smem(const Tiered& f, const uint8_t* sp, int w, int lane, uint32_t* s2, const uint32_t* tab, uint32_t (&R)[16]) {
    constexpr int CB = 4 * (int)STEP_BYTES;
    const int* bs = (const int*)(sp + CB + TT_BB);
    const uint8_t* q = sp + w * STEP_BYTES;
    StepH st;
    st.pd[0] = ((const uint32_t*)q)[lane];
    st.pd[1] = ((const uint32_t*)q)[32 + lane];
    uint4 x0 = ((const uint4*)(q + 256))[lane], x1 = ((const uint4*)(q + 256))[32 + lane];
    st.sw[0] = x0.x; st.sw[1] = x0.y; st.sw[2] = x0.z; st.sw[3] = x0.w;
    st.sw[4] = x1.x; st.sw[5] = x1.y; st.sw[6] = x1.z; st.sw[7] = x1.w;
    st.rb = (uint32_t)bs[w];
    st.re = (uint32_t)bs[w + 1];
    const int lo = bs[5];
    const uint32_t b = (st.rb & 0x7FFFFFFFu) >> 2, e = (st.re & 0x7FFFFFFFu) >> 2;
    if (lo >= 0) {  // (blocks' byte i at sp + CB + i - lo)
        const uint8_t* wb = sp + CB;
        st.win = *(const uint32_t*)(wb + (int)(4u * (b + lane) - (uint32_t)lo));
        st.tail = *(const uint32_t*)(wb + (int)(4u * (e - 1 - lane) - (uint32_t)lo));
        st.tail1 = *(const uint32_t*)(wb + (int)(4u * e - (uint32_t)lo));
    } else {
        const uint32_t* bw = (const uint32_t*)f.blocks;
        st.win = __ldg(bw + b + lane);
        st.tail = __ldg(bw + e - 1 - lane);
        st.tail1 = __ldg(bw + e);
    }
    uint32_t ew[8];
    if (st.rb >> 31) exponents(step_of(st), f.ts, f.blocks, lane, s2, tab, ew);  // (inline: a call would serialize every wgmma)
    else exponents_h(st, f.ts, lane, s2, tab, ew);
    pairs(st.sw, ew, R);
}

template <int NT, int WG, int BPS, bool SS>
__global__ void __launch_bounds__((TmaT<NT, WG, BPS>::THREADS), BPS) tiered_tma_kernel(const __grid_constant__ CUtensorMap xmap, Tiered f, int64_t O, int64_t K, int64_t M, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ Y, float* __restrict__ parts, int* __restrict__ done, int bb) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    using C = TmaT<NT, WG, BPS>;
    extern __shared__ __align__(1024) uint8_t smem_raw[];  // NS slots [X tile | a row block's steps, blocks' bytes, bounds | the others'], the trade buffers, the barriers, the scratch
    __shared__ int last;
    __shared__ uint32_t tab[256];
    uint32_t raw = (uint32_t)__cvta_generic_to_shared(smem_raw), base = (raw + 1023) & ~1023u;
    uint8_t* gbase = smem_raw + (base - raw);
    uint8_t* more = gbase + C::NS * C::SLOT;  // (1024-byte aligned: the tiles' swizzle)
    uint32_t full = base + C::NS * C::SLOT + WG * C::XCH, empty = full + 8 * C::NS;
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    // Units, stages and their split over the blocks as mma12_tma_kernel's.
    int TC = (int)((M + NT - 1) / NT), RU = (int)((O / 64 + WG - 1) / WG);
    int64_t KS = K / 16, RB = O / 64, U = (int64_t)RU * TC * (K / 64), nb = gridDim.x;
    int S = (int)(K / 64);
    int64_t u0 = blockIdx.x * U / nb;
    int n = (int)((blockIdx.x + 1) * U / nb - u0);
    int p0 = (int)(u0 / S), s0 = (int)(u0 - (int64_t)p0 * S);
    fill_groups(tab);
    if (tid == 0) {
        for (int i = 0; i < C::NS; i++) {
            mbar_init(full + 8 * i, 1);
            mbar_init(empty + 8 * i, 4 * WG);
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();
    if (warp >= 4 * WG) {
        // The TMA warp, as mma12_tma_kernel's: lane j mod 8 issues stage j, the bounds of a batch of 8 loaded while the
        // batch before is issued.
        int e[5 * WG], en[5 * WG];
        auto ahead = [&](int p, int s, int k) {
            s += k;
            tiered_bounds<WG>(f, RB, KS, (p + s / S) % RU, s % S, en);
        };
        if (lane < 8 && lane < n) ahead(p0, s0, lane);
        for (int j = 0, p = p0, s = s0, pr = p0 % RU, pc = p0 / RU; j < n; j++) {
            if ((j & 7) == 0) {
#pragma unroll
                for (int i = 0; i < 5 * WG; i++) e[i] = en[i];
                if (lane < 8 && j + 8 + lane < n) ahead(p, s, 8 + lane);
            }
            if (lane == (j & 7)) {
                int sl = j % C::NS;
                uint32_t fb = full + 8 * sl, xs = base + sl * C::SLOT;
                if (j >= C::NS) mbar_wait(empty + 8 * sl, (j / C::NS - 1) & 1);  // the consumers are done with stage j - NS
                uint32_t bytes = C::XB + WG * C::CB;
                int lo[WG], nw[WG];
#pragma unroll
                for (int r = 0; r < WG; r++) {
                    // The 4 steps' blocks' bytes as their decode reads them: tails from 128 before the first one's end,
                    // windows to 128 past the last one's start (blocks hold 128 bytes before the first and 256 after the last).
                    uint32_t b0 = (uint32_t)e[5 * r] & 0x7FFFFFFFu, e0 = (uint32_t)e[5 * r + 1] & 0x7FFFFFFFu;
                    uint32_t b3 = (uint32_t)e[5 * r + 3] & 0x7FFFFFFFu, e3 = (uint32_t)e[5 * r + 4] & 0x7FFFFFFFu;
                    uint32_t l = min(b0 & ~3u, (e0 & ~3u) - 128u) & ~15u, h = (max((b3 & ~3u) + 128u, (e3 & ~3u) + 4u) + 15u) & ~15u;
                    lo[r] = (int)l;
                    nw[r] = h - l <= (uint32_t)bb ? (int)(h - l) : -1;  // (bb: BB, or less in the checks)
                    bytes += nw[r] > 0 ? (uint32_t)nw[r] : 0u;
                    int* bs = (int*)(gbase + sl * C::SLOT + C::XB + r * C::RBB + C::CB + C::BB);
#pragma unroll
                    for (int i = 0; i < 5; i++) bs[i] = e[5 * r + i];
                    bs[5] = nw[r] < 0 ? -1 : lo[r];
                }
                mbar_expect_tx(fb, bytes);
                tma_2d(xs, &xmap, (int)(s * 64), pc * NT, fb);
#pragma unroll
                for (int r = 0; r < WG; r++) {
                    uint32_t cs = xs + C::XB + r * C::RBB;
                    int64_t rbk = min((int64_t)WG * pr + r, RB - 1);
                    bulk_g2s(cs, f.data + (rbk * KS + 4 * s) * STEP_BYTES, C::CB, fb, TC == 1);  // (read once but where the chunks share it)
                    if (nw[r] > 0) bulk_g2s(cs + C::CB, f.blocks + lo[r], (uint32_t)nw[r], fb, TC == 1);
                }
            }
            __syncwarp();
            if (++s == S) {
                s = 0, p++;
                if (++pr == RU) pr = 0, pc++;
            }
        }
        return;
    }
    // Consumer warpgroup wg: row block WG (p mod RU) + wg of each unit p; warp w decodes step w of each stage and then
    // holds rows 16w to 16w + 15 of the 4. Two sets of A registers, used in turn, as mma12_tma_kernel's.
    int ct = tid, wg = ct >> 7, w = (ct >> 5) & 3;
    uint4* trade = (uint4*)(more + wg * C::XCH);
    uint32_t* s2 = (uint32_t*)(more + WG * C::XCH + 16 * C::NS) + warp * (S2_BYTES / 4);
    float d[NT / 2];  // rows 16w + g (+ 8), tokens 8jn + 2t (+ 1): d[4jn + 2h + c]; set by a unit's first product
#pragma unroll
    for (int i = 0; i < NT / 2; i++) d[i] = 0.f;  // (zeroed all the same, as mma12_tma_kernel's)
    uint32_t A0[4][4], A1[4][4];
    int held = -1;  // the slot whose products may still be running
    auto release = [&](int sl) {
        __syncwarp();
        asm volatile("{\n.reg .pred p;\nsetp.eq.u32 p, %1, 0;\n@p mbarrier.arrive.shared::cta.b64 _, [%0];\n}\n" ::"r"(empty + 8 * sl), "r"(lane) : "memory");
    };
    auto keep_d = [&]() {
#pragma unroll
        for (int i = 0; i < NT / 2; i++) asm volatile("" : "+f"(d[i])::"memory");
    };
    auto keep = [&](uint32_t(&A)[4][4]) {
#pragma unroll
        for (int kk = 0; kk < 4; kk++)
#pragma unroll
            for (int i = 0; i < 4; i++) asm volatile("" : "+r"(A[kk][i])::"memory");
    };
    auto stage = [&](int j, bool first, uint32_t(&A)[4][4], uint32_t(&Aprev)[4][4]) {
        int sl = j % C::NS;
        mbar_wait(full + 8 * sl, (j / C::NS) & 1);
        uint32_t R[16];
        tiered_step_smem(f, gbase + sl * C::SLOT + C::XB + wg * C::RBB, w, lane, s2, tab, R);
        // The trade: this warp's step, rows 16q to 16q + 15 as A fragments (rows g, g + 8 by columns 2t, 2t + 8: n-tiles
        // 2q and 2q + 1) to [w][q]; then [kk][w] of each step kk, once the warpgroup's 4 have put theirs. (Two buffers: a
        // warp puts stage j + 2's after the barrier of j + 1, which every warp passes after taking stage j's.)
        uint4* tb = trade + (j & 1) * 512;
        uint32_t xs = base + sl * C::SLOT;
        uint64_t desc[4];
#pragma unroll
        for (int kk = 0; kk < 4; kk++) desc[kk] = sw128_desc(xs + kk * 32);
        if constexpr (SS) {
            // Row 8n + g's columns 16w + 2t (n-tile n's R[2n]) and 16w + 8 + 2t (R[2n + 1]): 16-byte chunks 2w and 2w + 1 of
            // its 128 bytes, at chunk c ^ (row mod 8) (the 128-byte swizzle; a store of the warp's: 8 rows' 16 bytes, one
            // bank a lane). Then the tile to the tensor cores' proxy, and the warpgroup's 4 steps in it.
            uint8_t* at = (uint8_t*)tb + 4 * (lane & 3);
            const uint32_t g = (uint32_t)lane >> 2, c0 = ((2u * w) ^ g) << 4, c1 = ((2u * w + 1) ^ g) << 4;
#pragma unroll
            for (int nn = 0; nn < 8; nn++) {
                uint8_t* row = at + (8 * nn + g) * 128;
                *(uint32_t*)(row + c0) = R[2 * nn];
                *(uint32_t*)(row + c1) = R[2 * nn + 1];
            }
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            bar_sync<128>(2 + wg);
            uint32_t ab = (uint32_t)__cvta_generic_to_shared(tb);
            uint64_t da[4];
#pragma unroll
            for (int kk = 0; kk < 4; kk++) da[kk] = sw128_desc(ab + kk * 32);
            keep_d();
            wg_fence();
            wgmma4_ss(d, da, desc, !first);
            wg_wait<1>();  // the stage before's products are done: its tile and its slot are free
            keep_d();
        } else {
#pragma unroll
            for (int q = 0; q < 4; q++) tb[(w * 4 + q) * 32 + lane] = make_uint4(R[4 * q], R[4 * q + 2], R[4 * q + 1], R[4 * q + 3]);
            bar_sync<128>(2 + wg);
#pragma unroll
            for (int kk = 0; kk < 4; kk++) {
                uint4 v = tb[(kk * 4 + w) * 32 + lane];
                A[kk][0] = v.x;
                A[kk][1] = v.y;
                A[kk][2] = v.z;
                A[kk][3] = v.w;
            }
            keep_d();
            wg_fence();
            wgmma4_rs(d, A, desc, !first);
            wg_wait<1>();  // the stage before's products are done: its A set and its slot are free
            keep(Aprev);
            keep_d();
        }
        if (held >= 0) release(held);
        held = sl;
    };
    for (int j = 0, p = p0, s = s0; j < n; p++, s = 0) {
        int len = min(S - s, n - j), k = 0;
        for (; k + 1 < len; k += 2) {
            stage(j + k, k == 0, A0, A1);
            stage(j + k + 1, false, A1, A0);
        }
        if (k < len) stage(j + k, k == 0, A0, A1);
        wg_wait<0>();
        if constexpr (!SS) {
            keep(A0);
            keep(A1);
        }
        keep_d();
        if (held >= 0) release(held);
        held = -1;
        sum_out12<NT, WG>(d, p, p % RU, (int64_t)(p / RU) * NT, S, nb, U, parts, done, last, ct, O, M, bias, Y);
        j += len;
    }
#elif defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    __trap();  // sm_90a code, launched from a build without it (or on AMD)
#endif
}

// Prompts on Hopper (sm_90a) past 128 tokens, the 12-bit layout: blocks that stay (one an SM, taking tile after tile),
// warp-specialized as the mixed-input GEMMs for Hopper are (CUTLASS 3.x's, Machete's). Lane 0 of a warp of its own
// copies each stage (64 columns) by TMA into a ring of NS stages, each landing on an mbarrier: X's tile of NT tokens
// through a tensor map in wgmma's 128-byte swizzle, W's two row blocks' 4 steps still compact, and their exceptions.
// Two consumer warpgroups, a row block each (a tile 128 of W's rows by NT tokens), decode a step (a k-block of 16
// columns) at a time into wgmma's A registers, a register set a k-block, and hand it to wgmma (64 rows by NT tokens by
// 16 columns) while the two k-blocks before it still multiply: a weight decoded once for NT tokens. The blocks go in
// clusters of CL, a block a row unit (128 rows) of the cluster's tile (CL row units by a chunk of NT tokens), X's tile
// copied once for the cluster (each block copies its CL-th into every block's shared memory: TMA's multicast). Tiles,
// chunks fastest (the clusters at work at once read each row unit's weights from memory once), in whole waves are each
// a cluster's own, summed in its registers and written out; the R left (fewer than a wave) are split by stages over up
// to 3 R clusters (all of them where the tiles are fewer than the clusters): stream-K, a tile covered by several summed
// by the last of them to finish, in cluster order. What bounds it (an H100 SXM, builds for timing alone): with nothing
// decoded (A a constant) Qwen3-8B's gate_up and o took 1.09 / 1.22x cuBLAS's time at 4096 tokens (1.19 / 1.38x with
// each block copying X, in 4 stages); the decode's integer instructions then add about their issue time (gate_up, with
// the stage's loads and exceptions, 0 / 11 / 21 of them a k-block: 1228 / 1310 / 1508 us), no less with some of them
// on the multiply pipe instead or decoded while the other warpgroup's products run: a weight decoded once for 256
// tokens (the most sums a warpgroup's registers hold) costs the tensor cores a quarter to a third more time. So the
// fewest instructions a stage: the exceptions in one pass (4 stages of 256 tokens then fit, not 5, which with a pass
// a step were the slower).
template <int NT, int CL> struct Wgp12 {
    static constexpr int THREADS = 384;  // two consumer warpgroups, then the producer warpgroup (its first warp the TMA's)
    static constexpr int CR = 232;       // a consumer's registers after setmaxnreg (the producers' 40): 64 NT / 128 its sums
    static constexpr int XB = NT * 128;  // X's tile a stage: NT tokens by 64 columns
    static constexpr int CB = 4 * (int)STEP12, EB = 512, BB = 32;  // a row block's steps a stage, its exceptions (up to 128), their bounds
    static constexpr int RBB = CB + EB + BB, WB = 2 * RBB;
    static constexpr int SCRATCH = 8 * 1024;  // the consumer warps' exceptions' scratch, 1 KB each
    // Stages in flight: the X tiles' ring (1024-byte aligned, as the swizzle wants), then W's, the barriers, the scratch.
    static constexpr int NS = (227 * 1024 - SCRATCH - 256) / (XB + WB) < 8 ? (227 * 1024 - SCRATCH - 256) / (XB + WB) : 8;
    static constexpr int SHARED = NS * (XB + WB) + 16 * NS + 16 + SCRATCH;
    static constexpr int SPLIT = 3;  // the split tiles' clusters, at most, for each (past whole waves)
};

// A consumer warp's stage (its 16 rows of a row block's 4 steps at sp, as decode12_rows reads them): the codes' word w
// and 8 low bytes of each step.
__device__ __forceinline__ void stage12(const uint8_t* sp, int lane, int w, uint32_t (&nw)[4], uint2 (&sw)[4]) {
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
        const uint8_t* q = sp + kk * STEP12;
        nw[kk] = *(const uint32_t*)(q + 16 * lane + 4 * w);
        sw[kk] = *(const uint2*)(q + 512 + 512 * (w >> 1) + 16 * lane + 8 * (w & 1));
    }
}

// A stage's exceptions in a warp's rows (bs: the run's bounds at the 4 steps and the next, then ea: the copy's first
// entry, -1 where read from global memory) as bytes to XOR into its high-byte words (xe[2kk], xe[2kk + 1]: step kk's),
// through the warp's scratch (xw: 8 words a lane, zero between stages), decode12_rows's plan: a pass per 32 entries.
__device__ __forceinline__ void stage12_exc(const Nib& f, const uint32_t* se, const int* bs, int lane, int w, uint32_t* xw, uint32_t (&xe)[8]) {
    int eb[5];
#pragma unroll
    for (int i = 0; i < 5; i++) eb[i] = bs[i];
#pragma unroll
    for (int i = 0; i < 8; i++) xe[i] = 0u;
    if (eb[4] > eb[0]) {
        int ea = bs[5];
        bool mine = false;
        for (int k = eb[0] + lane; k < eb[4]; k += 32) {
            uint32_t x = ea >= 0 ? se[k - ea] : __ldg(f.exc + k), i = x & 31;
            if ((int)(i >> 3) == w) {
                uint32_t kk = (k >= eb[1]) + (k >= eb[2]) + (k >= eb[3]);
                ((uint8_t*)xw)[(((x >> 5) & 31) * 8 + 2 * kk + ((i >> 2) & 1)) * 4 + (i & 3)] = (uint8_t)(x >> 16);
                mine = true;
            }
        }
        if (__any_sync(FULL, mine)) {
            __syncwarp();
            uint4* xl = (uint4*)(xw + 8 * lane);
            uint4 a = xl[0], b = xl[1];
            xe[0] = a.x, xe[1] = a.y, xe[2] = a.z, xe[3] = a.w, xe[4] = b.x, xe[5] = b.y, xe[6] = b.z, xe[7] = b.w;
            xl[0] = xl[1] = make_uint4(0u, 0u, 0u, 0u);
            __syncwarp();
        }
    }
}

// Step kk of a stage as a warp's A fragments (rows g and g + 8 of its 16, columns 2t and 8 + 2t), from stage12's.
__device__ __forceinline__ void step12(const Nib& f, uint32_t nw, uint2 sw, uint32_t x0, uint32_t x1, uint32_t (&A)[4]) {
    uint32_t e0 = Nib::high1(nw, f.hb4, 0) ^ x0, e1 = Nib::high1(nw, f.hb4, 1) ^ x1;
    A[0] = __byte_perm(sw.x, e0, 0x5140);
    A[1] = __byte_perm(sw.y, e1, 0x5140);
    A[2] = __byte_perm(sw.x, e0, 0x7362);
    A[3] = __byte_perm(sw.y, e1, 0x7362);
}

#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
// One wgmma m64nNk16 (a k-block), A (W's 64 rows by 16 columns) from registers, B (N tokens) from shared memory, and
// its commit: D = A B + (acc ? D : 0).
__device__ __forceinline__ void wgmma1_rs(float (&d)[96], const uint32_t (&a)[4], uint64_t db, int acc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %101, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n192k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95}, {%96, %97, %98, %99}, %100, p, 1, 1, 0;\n"
                 "wgmma.commit_group.sync.aligned;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(acc));
}
__device__ __forceinline__ void wgmma1_rs(float (&d)[128], const uint32_t (&a)[4], uint64_t db, int acc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %133, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, {%128, %129, %130, %131}, %132, p, 1, 1, 0;\n"
                 "wgmma.commit_group.sync.aligned;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(acc));
}
#endif

// A block's sum out (sum_out12's for the kernel below): its tile's rows (row unit pr: rows 16w + g (+ 8) of row
// block 2 pr + wg), tokens m0 on; r in each thread's fragment order. A whole tile (q < 0), or one the cluster covers
// alone: to Y. Else split tile q of the cluster tiles past whole waves, covered by clusters first to fin (U: the split
// tiles' stages, over nc clusters): to the block's slot (2 (cluster CL + rank) for the cluster's first split tile, + 1
// for its last), and the last of the tile's blocks to finish adds the slots in cluster order (the same every run);
// done: a counter a split tile a rank. nc (mma12_wgp_run's ncs) <= U, as capped there, so every cluster from first to
// fin has a stage of the tile and arrives here: the last finds fin - first before it.
template <int NT, int CL>
__device__ __forceinline__ void sum_out_wgp(float (&r)[NT / 2], int64_t q, int rank, int64_t pr, int64_t m0, int S, int64_t nc, int64_t U, float* parts, int* done, int& last, int ct, int64_t O, int64_t M, const __nv_bfloat16* bias, __nv_bfloat16* Y) {
    constexpr int R = 128;
    int wg = ct >> 7, w = (ct >> 5) & 3, g = (ct & 31) >> 2, t = ct & 3;
    int64_t first = q < 0 ? 0 : block_of_step(q * S, nc, U), fin = q < 0 ? 0 : block_of_step(q * S + S - 1, nc, U);
    auto slot = [&](int64_t c) { return parts + (2 * (c * CL + rank) + (c == first && c * U / nc != q * S)) * (NT * R); };
    if (first != fin) {
        float4* pp = (float4*)slot(blockIdx.x / CL);
#pragma unroll
        for (int i = 0; i < NT / 8; i++) pp[i * 256 + ct] = make_float4(r[4 * i], r[4 * i + 1], r[4 * i + 2], r[4 * i + 3]);
        bar_sync<256>(1);
        if (ct == 0) {
            int before = atom_inc_acq_rel(done + q * CL + rank);
            last = before == fin - first;
            if (last) done[q * CL + rank] = 0;  // ready for the next product
        }
        bar_sync<256>(1);
        if (last) {
#pragma unroll
            for (int i = 0; i < NT / 2; i++) r[i] = 0.f;
            for (int64_t c = first; c <= fin; c++) {
                const float4* bp = (const float4*)slot(c);
#pragma unroll
                for (int i = 0; i < NT / 8; i++) {
                    float4 v = __ldcg(bp + i * 256 + ct);
                    r[4 * i] += v.x;
                    r[4 * i + 1] += v.y;
                    r[4 * i + 2] += v.z;
                    r[4 * i + 3] += v.w;
                }
            }
        }
    }
    if (first == fin || last) {
#pragma unroll
        for (int jn = 0; jn < NT / 8; jn++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                int64_t o = (2 * pr + wg) * 64 + 16 * w + g + 8 * h;
                float bo = bias && o < O ? __bfloat162float(bias[o]) : 0.f;
#pragma unroll
                for (int c = 0; c < 2; c++) {
                    int64_t m = m0 + 8 * jn + 2 * t + c;
                    if (o < O && m < M) Y[m * O + o] = __float2bfloat16(r[4 * jn + 2 * h + c] + bo);
                }
            }
    }
}

#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
// A 2-D tile through a tensor map into the shared memory of each block of the cluster in mask (at dst there, its
// mbarrier at bar there).
__device__ __forceinline__ void tma_2d_mc(uint32_t dst, const CUtensorMap* map, int c0, int c1, uint32_t bar, uint16_t mask) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3}], [%4], %5;\n" ::"r"(dst), "l"((uint64_t)map), "r"(c0), "r"(c1), "r"(bar), "h"(mask) : "memory");
}
#endif

// A cluster's stages in order: its whole tiles c, c + nc, ... below D (cluster tiles: CL row units by a chunk), then its
// share of the split tiles' stages ([z0, z1) of the (T - D) S from tile D on: tile D + z / S, stage z mod S); a block
// its row unit of each.
template <int NT, int CL>
__global__ void __launch_bounds__((Wgp12<NT, CL>::THREADS), 1) mma12_wgp_kernel(const __grid_constant__ CUtensorMap xmap, Nib f, int64_t O, int64_t K, int64_t M, int64_t D, int ncs, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ Y, float* __restrict__ parts, int* __restrict__ done) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    using C = Wgp12<NT, CL>;
    // NS X tiles, NS W slots [a row block's steps, exceptions, bounds | the other's], the full and empty barriers, the
    // sum out's flag, the scratch. The X tiles 1024-byte aligned: the block's shared memory is its dynamic memory alone.
    extern __shared__ __align__(1024) uint8_t wsmem[];
    uint32_t base = (uint32_t)__cvta_generic_to_shared(wsmem);
    if (base & 1023) __trap();
    uint8_t *smem = wsmem, *gw = smem + C::NS * C::XB;
    uint32_t full = base + C::NS * (C::XB + C::WB), empty = full + 8 * C::NS;
    int& last = *(int*)(smem + C::NS * (C::XB + C::WB) + 16 * C::NS);
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, rank = 0;
    if constexpr (CL > 1) asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rank));
    int TC = (int)((M + NT - 1) / NT), S = (int)(K / 64);
    int64_t KS = K / 16, RB = O / 64, T = ((RB + 1) / 2 + CL - 1) / CL * TC, nc = gridDim.x / CL, c = blockIdx.x / CL;
    int64_t U = (T - D) * S, z0 = c < ncs ? c * U / ncs : 0, z1 = c < ncs ? (c + 1) * U / ncs : 0;
    int64_t whole = c < D ? (D - 1 - c) / nc + 1 : 0;  // the cluster's whole tiles
    int n = (int)(whole * S + z1 - z0);                // and its stages
    if (tid == 0) {
        for (int i = 0; i < C::NS; i++) {
            mbar_init(full + 8 * i, 1);
            mbar_init(empty + 8 * i, 2 * CL);  // each consumer warpgroup of the cluster's blocks: X's tile is theirs too
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    if constexpr (CL > 1) asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
    else __syncthreads();
    if (warp >= 8) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n" ::: "memory");
        if (warp != 8) return;
        // The TMA warp: lane 0 issues each stage; lanes 0-9 meanwhile load the next stage's bounds (row block lane / 5,
        // exc_base at its step lane mod 5 of the 4 and the next), gathered by lane 0 a stage later. A tile's row unit
        // and chunk are worked out once a tile.
        int64_t t = whole ? c : D + z0 / S, rb = 0;
        int s = whole ? 0 : (int)(z0 % S), pu, pc;
        bool w0 = whole > 0;
        auto at = [&]() {
            pu = (int)(t / TC), pc = (int)(t - (int64_t)pu * TC);
            rb = 2 * ((int64_t)pu * CL + rank);  // (row blocks past O: the last one's, their products not written)
        };
        auto bound = [&]() { return lane < 10 ? __ldg(f.exc_base + min(rb + lane / 5, RB - 1) * KS + 4 * s + lane % 5) : 0; };
        at();
        int ev = n > 0 ? bound() : 0;
        for (int j = 0; j < n; j++) {
            int e[10];
#pragma unroll
            for (int i = 0; i < 10; i++) e[i] = __shfl_sync(FULL, ev, i);
            int sl = j % C::NS, s0 = s, pc0 = pc;
            int64_t rb0 = rb;
            if (++s == S) {
                s = 0;
                if (!w0) t++;
                else if ((t += nc) >= D) w0 = false, t = D + z0 / S, s = (int)(z0 % S);
                at();
            }
            if (j + 1 < n) ev = bound();
            if (lane == 0) {
                uint32_t fb = full + 8 * sl, xs = base + sl * C::XB, ws = base + C::NS * C::XB + sl * C::WB;
                if (j >= C::NS) mbar_wait(empty + 8 * sl, (j / C::NS - 1) & 1);  // the cluster's consumers are done with stage j - NS
                uint32_t bytes = C::XB + 2 * C::CB;
                int a[2], na[2];
#pragma unroll
                for (int r = 0; r < 2; r++) {
                    exc_copy(e[5 * r], e[5 * r + 4], C::EB, a[r], na[r]);
                    bytes += na[r] > 0 ? 4 * na[r] : 0;
                    int* bs = (int*)(gw + sl * C::WB + r * C::RBB + C::CB + C::EB);
#pragma unroll
                    for (int i = 0; i < 5; i++) bs[i] = e[5 * r + i];
                    bs[5] = na[r] < 0 ? -1 : a[r];
                }
                mbar_expect_tx(fb, bytes);
                if constexpr (CL > 1) tma_2d_mc(xs + rank * (C::XB / CL), &xmap, s0 * 64, pc0 * NT + rank * (NT / CL), fb, (uint16_t)((1 << CL) - 1));
                else tma_2d(xs, &xmap, s0 * 64, pc0 * NT, fb);
#pragma unroll
                for (int r = 0; r < 2; r++) {
                    uint32_t cs = ws + r * C::RBB;
                    bulk_g2s(cs, f.data + (min(rb0 + r, RB - 1) * KS + 4 * s0) * STEP12, C::CB, fb, TC == 1);  // (read once but where chunks share it)
                    if (na[r] > 0) bulk_g2s(cs + C::CB, f.exc + a[r], 4 * na[r], fb);
                }
            }
            __syncwarp();
        }
        if (lane) return;
        // Every stage's slot released by the cluster's consumers before the block ends (their arrivals and the other
        // blocks' copies land in its shared memory).
        for (int j = max(0, n - C::NS); j < n; j++) mbar_wait(empty + 8 * (j % C::NS), (j / C::NS) & 1);
        return;
    }
    // Consumer warpgroup wg: row block 2 pr + wg of each tile; warp w its rows 16w to 16w + 15.
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(C::CR) : "memory");
    int ct = tid, wg = ct >> 7, w = (ct >> 5) & 3;
    uint32_t* xw = (uint32_t*)(smem + C::NS * (C::XB + C::WB) + 16 * C::NS + 16) + 256 * warp;  // this warp's exceptions' scratch
    ((uint4*)xw)[2 * lane] = ((uint4*)xw)[2 * lane + 1] = make_uint4(0u, 0u, 0u, 0u);
    __syncwarp();
    float d[NT / 2] = {};  // rows 16w + g (+ 8), tokens 8jn + 2t (+ 1): d[4jn + 2h + c]; set by a tile's first product
    // (zeroed all the same, as mma12_tma_kernel's: left unset, CUDA 12.8's ptxas serialized every wgmma, its C7515)
    uint32_t A[4][4];
    int held = -1;  // the slot of the stage before, whose products may still be running
    auto release = [&](int sl) {  // a warpgroup's first warp's lane i arrives on block i's barrier of the cluster, by predicate
        __syncwarp();          // (no branch among the products): its products are the warpgroup's, their reads done
        uint32_t go = w == 0 && lane < CL && sl >= 0, bar = empty + 8 * max(sl, 0);
        if constexpr (CL > 1) asm volatile("{\n.reg .pred p;\n.reg .b32 ra;\nsetp.ne.u32 p, %1, 0;\nmapa.shared::cluster.u32 ra, %0, %2;\n@p mbarrier.arrive.shared::cluster.b64 _, [ra];\n}\n" ::"r"(bar), "r"(go), "r"(lane) : "memory");
        else asm volatile("{\n.reg .pred p;\nsetp.ne.u32 p, %1, 0;\n@p mbarrier.arrive.shared::cta.b64 _, [%0];\n}\n" ::"r"(bar), "r"(go) : "memory");
    };
    auto keep_d = [&]() {  // d in its registers, untouched while the products run
#pragma unroll
        for (int i = 0; i < NT / 2; i++) asm volatile("" : "+f"(d[i])::"memory");
    };
    int64_t t = whole ? c : D + z0 / S;
    int s = whole ? 0 : (int)(z0 % S);
    bool w0 = whole > 0;
    for (int j = 0; j < n;) {
        int s0 = s, e = w0 ? S : min(S, s + (n - j));
        for (; s < e; s++, j++) {
            int sl = j % C::NS;
            mbar_wait(full + 8 * sl, (j / C::NS) & 1);
            const uint8_t* sp = gw + sl * C::WB + wg * C::RBB;
            const int* bs = (const int*)(sp + C::CB + C::EB);
            uint32_t nw[4];
            uint2 sw[4];
            stage12(sp, lane, w, nw, sw);
            uint32_t xe[8];  // the stage's exceptions, before its first product (ptxas keeps branches out from among them)
            stage12_exc(f, (const uint32_t*)(sp + C::CB), bs, lane, w, xw, xe);
            uint64_t xd = sw128_desc(base + sl * C::XB);  // X's tile: k-block kk's 32 bytes on (+2 a k-block)
#pragma unroll
            for (int kk = 0; kk < 4; kk++) {
                wg_wait<2>();  // at most two k-blocks' products running: the A set 4 back is free
                if (kk == 2) release(held), held = sl;  // (and the stage before's products all done: its slot)
                step12(f, nw[kk], sw[kk], xe[2 * kk], xe[2 * kk + 1], A[kk]);
                wg_fence();
                wgmma1_rs(d, A[kk], xd + 2 * kk, s > s0 || kk > 0);
            }
        }
        wg_wait<0>();
        keep_d();
        release(held);
        held = -1;
        int pu = (int)(t / TC);
        int64_t m0 = (t - (int64_t)pu * TC) * NT;
        sum_out_wgp<NT, CL>(d, w0 ? -1 : t - D, rank, (int64_t)pu * CL + rank, m0, S, ncs, U, parts, done, last, ct, O, M, bias, Y);
        s = 0;
        if (!w0) t++;
        else if ((t += nc) >= D) w0 = false, t = D + z0 / S, s = (int)(z0 % S);
    }
#elif defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    __trap();  // sm_90a code, launched from a build without it (or on AMD)
#endif
}

// Many tokens on Ampere and Ada (sm_86 to sm_89; the A100 its own kernel), the 12-bit layout: mma12_tma_kernel's plan with this
// generation's instructions. A producer warp copies each stage with cp.async, 16 bytes a thread, and the
// stage's mbarrier completes once every lane's copies have landed (cp.async.mbarrier.arrive) and lane 0
// has written the bounds; X's tile is laid out with a row's 16-byte chunk c at chunk c ^ (row mod 8), read
// as B fragments by ldmatrix. The consumer warps decode their 16 rows into A fragments as the TMA
// kernel's do and multiply with mma.sync, 16 rows by 8 tokens by 16 columns. Stream-K and the sum out as
// there. NS stages in flight; WG 4 where a block may hold 153 KB of shared memory (sm_87, sm_90, sm_100; an
// A100 takes a kernel of its own), else 2 (sm_86, sm_89, sm_120).
template <int NT, int WG> struct Mid12 {
    static constexpr int THREADS = 128 * WG + 32;  // WG consumer warpgroups, the producer warp
    static constexpr int NS = 4;
    static constexpr int XB = NT * 128;
    static constexpr int CB = 4 * (int)STEP12, EB = 512, BB = 32;  // a row block's steps a stage, its exceptions (up to 128), their bounds
    static constexpr int RBB = CB + EB + BB;
    static constexpr int SLOT = (XB + WG * RBB + 127) / 128 * 128;
    static constexpr int SHARED = NS * SLOT + 128 + 16 * NS + 4096 * WG;  // alignment, the full and empty barriers, the warps' exceptions' scratch
};

template <int NT, int WG>
__global__ void __launch_bounds__((Mid12<NT, WG>::THREADS), 1) mma12_mid_kernel(Nib f, int64_t O, int64_t K, int64_t M, const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ Y, float* __restrict__ parts, int* __restrict__ done) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using C = Mid12<NT, WG>;
    extern __shared__ __align__(128) uint8_t smem_raw[];  // NS slots [X tile | a row block's steps, exceptions, bounds | ...], the barriers
    __shared__ int last;
    uint32_t raw = (uint32_t)__cvta_generic_to_shared(smem_raw), base = (raw + 127) & ~127u;
    uint8_t* gbase = smem_raw + (base - raw);
    uint32_t full = base + C::NS * C::SLOT, empty = full + 8 * C::NS;
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int64_t KS = K / 16, RB = O / 64, U = (RB + WG - 1) / WG * (K / 64), nb = gridDim.x;
    int S = (int)(K / 64);
    int64_t u0 = blockIdx.x * U / nb;
    int n = (int)((blockIdx.x + 1) * U / nb - u0);
    int p0 = (int)(u0 / S), s0 = (int)(u0 - (int64_t)p0 * S);
    if (tid == 0)
        for (int i = 0; i < C::NS; i++) {
            mbar_init(full + 8 * i, 33);  // the 32 lanes' copies, lane 0's bounds
            mbar_init(empty + 8 * i, 4 * WG);
        }
    __syncthreads();
    if (warp == 4 * WG) {
        // The producer warp: every lane copies a share of stage j; lane j mod 8 holds its bounds, loaded
        // with its batch of 8 stages while the batch before was copied (as the TMA warp's).
        int e[5 * WG], en[5 * WG];
        auto ahead = [&](int p, int s, int k) {
            s += k;
            stage_bounds<WG>(f, RB, KS, p + s / S, s % S, en);
        };
        if (lane < 8 && lane < n) ahead(p0, s0, lane);
        for (int j = 0, p = p0, s = s0; j < n; j++, s = s + 1 == S ? 0 : s + 1, p += s == 0) {
            if ((j & 7) == 0) {
#pragma unroll
                for (int i = 0; i < 5 * WG; i++) e[i] = en[i];
                if (lane < 8 && j + 8 + lane < n) ahead(p, s, 8 + lane);
            }
            int eb[5 * WG];
#pragma unroll
            for (int i = 0; i < 5 * WG; i++) eb[i] = __shfl_sync(FULL, e[i], j & 7);
            int sl = j % C::NS;
            uint32_t xs = base + sl * C::SLOT;
            if (j >= C::NS) mbar_wait(empty + 8 * sl, (j / C::NS - 1) & 1);  // the consumers are done with stage j - NS
            for (int i = lane; i < NT * 8; i += 32) {  // X's tile; rows past M read as zeros
                int rr = i >> 3, c = i & 7;
                cp_async16(xs + rr * 128 + ((c ^ (rr & 7)) << 4), X + (rr < M ? (int64_t)rr * K : 0) + (int64_t)s * 64 + c * 8, rr < M ? 16 : 0);
            }
#pragma unroll
            for (int r = 0; r < WG; r++) {
                uint32_t cs = xs + C::XB + r * C::RBB;
                const uint8_t* src = f.data + (min((int64_t)WG * p + r, RB - 1) * KS + 4 * s) * STEP12;
                for (int i = lane; i < C::CB / 16; i += 32) cp_async16(cs + 16 * i, src + 16 * i, 16);
                int a, na;
                exc_copy(eb[5 * r], eb[5 * r + 4], C::EB, a, na);
                for (int i = lane; 4 * i < na; i += 32) cp_async16(cs + C::CB + 16 * i, f.exc + a + 4 * i, 16);
                if (lane == 0) {
                    int* bs = (int*)(gbase + sl * C::SLOT + C::XB + r * C::RBB + C::CB + C::EB);
#pragma unroll
                    for (int i = 0; i < 5; i++) bs[i] = eb[5 * r + i];
                    bs[5] = na < 0 ? -1 : a;
                }
            }
            asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" ::"r"(full + 8 * sl) : "memory");
            if (lane == 0) mbar_arrive(full + 8 * sl);
        }
        return;
    }
    // Consumer warpgroup wg: row block WG p + wg of each unit p; warp w its rows 16w to 16w + 15.
    int ct = tid, wg = ct >> 7, w = (ct >> 5) & 3;
    uint32_t* xw = (uint32_t*)(gbase + C::NS * C::SLOT + 16 * C::NS) + 256 * warp;  // this warp's exceptions' scratch
    ((uint4*)xw)[2 * lane] = ((uint4*)xw)[2 * lane + 1] = make_uint4(0u, 0u, 0u, 0u);
    __syncwarp();
    for (int j = 0, p = p0, s = s0; j < n; p++, s = 0) {
        int len = min(S - s, n - j);
        float d[NT / 2];  // n-tile jn's C fragment at d[4jn]: rows g, g + 8 by tokens 8jn + 2t, + 1
#pragma unroll
        for (int i = 0; i < NT / 2; i++) d[i] = 0.f;
        for (int k = 0; k < len; k++) {
            int jj = j + k, sl = jj % C::NS;
            mbar_wait(full + 8 * sl, (jj / C::NS) & 1);
            const uint8_t* sp = gbase + sl * C::SLOT + C::XB + wg * C::RBB;
            const int* bs = (const int*)(sp + C::CB + C::EB);
            int eb[5];
#pragma unroll
            for (int i = 0; i < 5; i++) eb[i] = bs[i];
            uint32_t A[4][4];
            decode12_rows(f, sp, (const uint32_t*)(sp + C::CB), eb, bs[5], lane, w, xw, A);
            uint32_t xs = base + sl * C::SLOT;
#pragma unroll
            for (int kk = 0; kk < 4; kk++)
#pragma unroll
                for (int jp = 0; jp < NT / 16; jp++) {
                    // Tokens 16jp to 16jp + 15, columns 16kk to 16kk + 15: B fragments of n-tiles 2jp and 2jp + 1.
                    int rr = 16 * jp + (lane & 7) + ((lane >> 4) << 3), c = 2 * kk + ((lane >> 3) & 1);
                    uint32_t b[4];
                    ldmatrix_x4(b, xs + rr * 128 + ((c ^ (rr & 7)) << 4));
                    mma16816(&d[8 * jp], A[kk], b[0], b[1]);
                    mma16816(&d[8 * jp + 4], A[kk], b[2], b[3]);
                }
            __syncwarp();
            if (lane == 0) mbar_arrive(empty + 8 * sl);  // the slot is free for stage jj + NS
        }
        sum_out12<NT, WG>(d, p, p, 0, S, nb, U, parts, done, last, ct, O, M, bias, Y);
        j += len;
    }
#elif defined(__HIP_DEVICE_COMPILE__)
    __trap();  // (Ampere's and later NVIDIA GPUs' alone: never launched on AMD)
#endif
}

// A 16-byte copy of data read once (W's steps): evicted from L2 first.
__device__ __forceinline__ void cp_async16_once(uint32_t dst, const void* src, uint64_t pol) {
#if defined(__HIP_PLATFORM_AMD__)
    (void)pol;
    cp_async16(dst, src, 16);
#else
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "l"(pol));
#endif
}
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// Many tokens on the A100 (sm_80), the 12-bit layout: mma_gemm_big_kernel's split of the warps, with the
// producers' reads staged. A producer warp a step of W's 4 RBB a stage (64 columns of RBB row blocks): it copies
// its step and the step's exceptions by cp.async NW - 1 stages ahead into its share of a ring (the exceptions' bounds
// loaded 32 stages ahead, a stage a lane), and decodes it from there into B fragments in shared memory; the
// producers copy X's tile a stage ahead of its decode. CW consumer warps multiply, each 16 MT tokens by a row
// block, on the tensor cores; named barriers pass a stage's buffers (NB of them) between the two. Stream-K over
// units (RBB row blocks by TM tokens) and their stages, as mma_gemm_kernel's: a unit covered by several blocks is
// summed by the last to finish, in block order (the same result every run).
template <int CW, int NB, int NW, int RBB, int MT> struct Ws12 {
    static constexpr int PW = 4 * RBB, THREADS = 32 * (CW + PW), PT = 32 * PW, CT = 32 * CW;
    static constexpr int TM = 16 * MT * CW / RBB;     // tokens a unit
    static constexpr int A_BYTES = TM * 128;          // X's tile a stage
    static constexpr int B_UINT4 = RBB * 4 * 4 * 32;  // W's fragments a stage: [row block][step][4][lane]
    static constexpr int DSLOT = A_BYTES + B_UINT4 * 16;
    static constexpr int EB = 128, SB = (int)STEP12 + EB + 16;  // a producer warp's share of a ring slot: its step, the step's exceptions (past EB bytes read from global memory), their bounds
    static constexpr int WSLOT = PW * SB;
    static constexpr int SHARED = NB * DSLOT + NW * WSLOT + 128;
    static constexpr int XQ = TM * 8 / PT;            // X's chunks a producer thread
    // NB >= 3: the producers copy stage j + 1's X tile before they decode stage j, into the slot the consumers
    // free after stage j + 1 - NB, and the consumers ask for stage j + 1 before they free stage j.
    static_assert(NB >= 3 && NW >= 2 && 2 * NB + 2 <= 15 && XQ * PT == TM * 8, "shapes");
};

template <int CW, int NB, int NW, int RBB, int MT>
__global__ void __launch_bounds__((Ws12<CW, NB, NW, RBB, MT>::THREADS), 1) mma12_ws_kernel(Nib f, int64_t O, int64_t K, int64_t M, const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ Y, float* __restrict__ parts, int* __restrict__ done) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using C = Ws12<CW, NB, NW, RBB, MT>;
    constexpr int FULL0 = 1, EMPTY0 = 1 + NB, CBAR = 1 + 2 * NB;  // named barriers
    extern __shared__ __align__(128) uint8_t smem_raw[];  // NB slots [X tile | B fragments], then the ring: NW slots [a producer warp's step, exceptions, bounds | ...]
    __shared__ int last;
    uint32_t raw = (uint32_t)__cvta_generic_to_shared(smem_raw), base = (raw + 127) & ~127u;
    uint8_t* gbase = smem_raw + (base - raw);
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int64_t KS = K / 16, RB = O / 64;
    int tiles = (int)((M + C::TM - 1) / C::TM), S = (int)(K / 64);
    int64_t T = (RB + RBB - 1) / RBB * tiles * S, nb = gridDim.x, u0 = blockIdx.x * T / nb;
    int n = (int)((blockIdx.x + 1) * T / nb - u0);  // the block's stages, u0 on: unit (tile of a pair of row blocks) u / S, stage u mod S
    int p0 = (int)(u0 / S), s0 = (int)(u0 - (int64_t)p0 * S);
    if (warp >= CW) {
        // Producer warp pw: step kk of row block pw / 4 of each stage. Commit groups a stage j: X's tile of stage
        // j + 1, then the warp's step of stage j + NW - 1.
        int pt = tid - C::CT, pw = warp - CW, kk = pw & 3, xr = pt >> 3, xc = pt & 7;
        uint64_t pol;
        asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
        const uint32_t wbase = base + NB * C::DSLOT + pw * C::SB;
        const uint8_t* gw = gbase + NB * C::DSLOT + pw * C::SB;
        // The step's exceptions' bounds (exc_base at it and the next): lane i holds them for stage i of the batch of
        // 32 (e0, e1), the next batch's (en0, en1) loaded meanwhile.
        int e0 = 0, e1 = 0, en0 = 0, en1 = 0;
        auto ahead = [&](int p, int s, int k) {  // the stage k after (p, s)
            s += k;
            p += s / S;
            s %= S;
            const int32_t* q = f.exc_base + min((int64_t)(p / tiles) * RBB + (pw >> 2), RB - 1) * KS + 4 * s + kk;
            en0 = __ldg(q);
            en1 = __ldg(q + 1);
        };
        if (lane < n) ahead(p0, s0, lane);
        int cj = 0, cp = p0, cs = s0, csl = 0;  // the next step to copy: stage cj (unit cp, stage cs), slot csl
        const uint8_t* wsrc = f.data + (min((int64_t)(p0 / tiles) * RBB + (pw >> 2), RB - 1) * KS + 4 * s0 + kk) * STEP12 + 16 * lane;
        auto wcopy = [&]() {
            if (cj < n) {
                uint32_t d = wbase + csl * C::WSLOT + 16 * lane;
#pragma unroll
                for (int q = 0; q < (int)STEP12 / 512; q++) cp_async16_once(d + 512 * q, wsrc + 512 * q, pol);
                if ((cj & 31) == 0) {
                    e0 = en0, e1 = en1;
                    if (cj + 32 + lane < n) ahead(cp, cs, 32 + lane);
                }
                int lo = __shfl_sync(FULL, e0, cj & 31), hi = __shfl_sync(FULL, e1, cj & 31), a, na;
                exc_copy(lo, hi, C::EB, a, na);
                if (4 * lane < na) cp_async16(wbase + csl * C::WSLOT + STEP12 + 16 * lane, f.exc + a + 4 * lane, 16);
                if (lane == 0) *(int4*)(gw + csl * C::WSLOT + STEP12 + C::EB) = make_int4(lo, hi, na < 0 ? -1 : a, 0);
                cj++;
                csl = csl + 1 == NW ? 0 : csl + 1;
                if (++cs == S) {
                    cs = 0, cp++;
                    wsrc = f.data + (min((int64_t)(cp / tiles) * RBB + (pw >> 2), RB - 1) * KS + kk) * STEP12 + 16 * lane;
                } else {
                    wsrc += 4 * STEP12;
                }
            }
            asm volatile("cp.async.commit_group;\n" ::);
        };
        int xj = 0, xp = p0, xs = s0, xsl = 0;  // the next X tile to copy: stage xj (unit xp, stage xs), slot xsl; this thread's first row xm (none past M)
        int64_t xm = (int64_t)(p0 % tiles) * C::TM + xr;
        auto xcopy = [&]() {
            if (xj < n) {
                uint32_t d = base + xsl * C::DSLOT + xr * 128 + ((xc ^ (xr & 7)) << 4);
#pragma unroll
                for (int q = 0; q < C::XQ; q++)
                    if (xm + C::PT / 8 * q < M) cp_async16(d + C::PT / 8 * 128 * q, X + (xm + C::PT / 8 * q) * K + (int64_t)xs * 64 + xc * 8, 16);
                xj++;
                xsl = xsl + 1 == NB ? 0 : xsl + 1;
                if (++xs == S) {
                    xs = 0, xp++;
                    xm = (int64_t)(xp % tiles) * C::TM + xr;
                }
            }
            asm volatile("cp.async.commit_group;\n" ::);
        };
#pragma unroll 1
        for (int i = 1 - NW; i < 0; i++) {
            if (i == -1) xcopy();
            else asm volatile("cp.async.commit_group;\n" ::);
            wcopy();
        }
        int wsl = 0, bsl = 0;  // stage j's ring slot, and its slot of fragments
        uint4* bdst = (uint4*)(gbase + C::A_BYTES) + pw * 4 * 32 + lane;
#pragma unroll 1
        for (int j = 0; j < n; j++) {
            cp_async_wait<2 * (NW - 2)>();
            __syncwarp();  // stage j's step has landed
            if (j + 1 < n && j + 1 >= NB) bar_sync<C::THREADS>(EMPTY0 + xsl);  // the consumers are done with stage j + 1 - NB (xsl: stage j + 1's slot)
            xcopy();
            wcopy();
            // The step into B fragments, as Nib::decode (its loads and exceptions from the ring).
            const uint8_t* q = gw + wsl * C::WSLOT;
            uint4 c4 = *(const uint4*)(q + 16 * lane), x0 = *(const uint4*)(q + 512 + 16 * lane), x1 = *(const uint4*)(q + 1024 + 16 * lane);
            uint32_t nbw[4] = {c4.x, c4.y, c4.z, c4.w}, sw[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w}, ew[8];
            f.high(nbw, ew);
            int4 bd = *(const int4*)(q + STEP12 + C::EB);
            const uint32_t *se = (const uint32_t*)(q + STEP12), *ge = f.exc;
            Nib::patch<2>([&](int k) { return bd.z >= 0 ? se[k - bd.z] : __ldg(ge + k); }, bd.x, bd.y, lane, ew);
            uint32_t R[16];
            Nib::pairs(sw, ew, R);
            uint4* d = bdst + bsl * (C::DSLOT / 16);
#pragma unroll
            for (int qq = 0; qq < 4; qq++) d[qq * 32] = make_uint4(R[4 * qq], R[4 * qq + 1], R[4 * qq + 2], R[4 * qq + 3]);
            cp_async_wait<3>();                   // stage j's X tile has landed
            bar_arrive<C::THREADS>(FULL0 + bsl);  // stage j is ready
            wsl = wsl + 1 == NW ? 0 : wsl + 1;
            bsl = bsl + 1 == NB ? 0 : bsl + 1;
        }
        cp_async_wait<0>();
        return;
    }
    // Consumer: tokens 16 MT (warp mod (CW / RBB)) on of the unit's TM, row block warp / (CW / RBB) of its RBB.
    int g = lane >> 2, t = lane & 3, wm = warp % (CW / RBB), rbl = warp / (CW / RBB);
    float acc[MT][8][4];
#pragma unroll
    for (int mt = 0; mt < MT; mt++)
#pragma unroll
        for (int nn = 0; nn < 8; nn++)
#pragma unroll
            for (int q = 0; q < 4; q++) acc[mt][nn][q] = 0.f;
    uint32_t a_row = (uint32_t)(wm * 16 * MT + (lane & 7) + ((lane >> 3) & 1) * 8) * 128, a_half = lane >> 4, a_sw = lane & 7;
    uint32_t fa[2][MT][4];
    uint4 fb[2][4];
    auto frags = [&](int bi, int kk, int fi) {
        uint32_t slot = base + (uint32_t)bi * C::DSLOT;
        const uint4* bsrc = (const uint4*)(gbase + bi * C::DSLOT + C::A_BYTES) + rbl * 4 * 4 * 32 + lane;
#pragma unroll
        for (int mt = 0; mt < MT; mt++) ldmatrix_x4(fa[fi][mt], slot + a_row + mt * 16 * 128 + (((2 * kk + a_half) ^ a_sw) << 4));
#pragma unroll
        for (int q = 0; q < 4; q++) fb[fi][q] = bsrc[(kk * 4 + q) * 32];
    };
    if (n > 0) {
        bar_sync<C::THREADS>(FULL0);
        frags(0, 0, 0);
    }
    int p = p0, s = s0, bsl = 0;
#pragma unroll 1
    for (int j = 0; j < n; j++) {
        int nsl = bsl + 1 == NB ? 0 : bsl + 1;
        bool out = s == S - 1 || j == n - 1;  // the unit's last stage here: its sum out next
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            int fi = kk & 1;
            if (kk + 1 < 4) frags(bsl, kk + 1, fi ^ 1);
            else if (!out) {  // (at a unit's end, after its sum out: the next stage's fragments not held through it)
                bar_sync<C::THREADS>(FULL0 + nsl);  // stage j + 1 is ready
                frags(nsl, 0, fi ^ 1);
            }
#pragma unroll
            for (int mt = 0; mt < MT; mt++) {
                mma16816(acc[mt][0], fa[fi][mt], fb[fi][0].x, fb[fi][0].y);
                mma16816(acc[mt][1], fa[fi][mt], fb[fi][0].z, fb[fi][0].w);
                mma16816(acc[mt][2], fa[fi][mt], fb[fi][1].x, fb[fi][1].y);
                mma16816(acc[mt][3], fa[fi][mt], fb[fi][1].z, fb[fi][1].w);
                mma16816(acc[mt][4], fa[fi][mt], fb[fi][2].x, fb[fi][2].y);
                mma16816(acc[mt][5], fa[fi][mt], fb[fi][2].z, fb[fi][2].w);
                mma16816(acc[mt][6], fa[fi][mt], fb[fi][3].x, fb[fi][3].y);
                mma16816(acc[mt][7], fa[fi][mt], fb[fi][3].z, fb[fi][3].w);
            }
        }
        if (j + NB < n) bar_arrive<C::THREADS>(EMPTY0 + bsl);  // done with stage j's buffers
        bsl = nsl;
        if (out) {
            // Unit p's sum out: to Y where this block covers it, else to its slot (the consumers' accumulators in
            // thread order), the last of its blocks to finish adding the slots in block order.
            int64_t first = block_of_step((int64_t)p * S, nb, T), fin = block_of_step((int64_t)p * S + S - 1, nb, T);
            if (first != fin) {
                float4* pp = (float4*)(parts + ((int64_t)blockIdx.x + p) * (C::TM * 64 * RBB));
#pragma unroll
                for (int mt = 0; mt < MT; mt++)
#pragma unroll
                    for (int nn = 0; nn < 8; nn++) pp[(mt * 8 + nn) * C::CT + tid] = make_float4(acc[mt][nn][0], acc[mt][nn][1], acc[mt][nn][2], acc[mt][nn][3]);
                __threadfence();
                bar_sync<C::CT>(CBAR);
                if (tid == 0) {
                    last = atomicAdd(done + p, 1) == fin - first;
                    if (last) done[p] = 0;  // ready for the next product
                }
                bar_sync<C::CT>(CBAR);
                if (last) {
                    __threadfence();
#pragma unroll
                    for (int mt = 0; mt < MT; mt++)
#pragma unroll
                        for (int nn = 0; nn < 8; nn++)
#pragma unroll
                            for (int q = 0; q < 4; q++) acc[mt][nn][q] = 0.f;
                    for (int64_t b = first; b <= fin; b++) {
                        const float4* bp = (const float4*)(parts + (b + p) * (C::TM * 64 * RBB));
#pragma unroll
                        for (int mt = 0; mt < MT; mt++)
#pragma unroll
                            for (int nn = 0; nn < 8; nn++) {
                                float4 v = __ldcg(bp + (mt * 8 + nn) * C::CT + tid);
                                acc[mt][nn][0] += v.x;
                                acc[mt][nn][1] += v.y;
                                acc[mt][nn][2] += v.z;
                                acc[mt][nn][3] += v.w;
                            }
                    }
                }
            }
            if (first == fin || last) {
                int64_t rb = (int64_t)(p / tiles) * RBB + rbl, m0 = (int64_t)(p % tiles) * C::TM + wm * 16 * MT + g;
                if (rb < RB)
#pragma unroll
                    for (int nn = 0; nn < 8; nn++) {
                        int64_t o = rb * 64 + nn * 8 + t * 2;
                        float b0 = bias ? __bfloat162float(bias[o]) : 0.f, b1 = bias ? __bfloat162float(bias[o + 1]) : 0.f;
#pragma unroll
                        for (int mt = 0; mt < MT; mt++)
#pragma unroll
                            for (int h = 0; h < 2; h++) {
                                int64_t m = m0 + mt * 16 + h * 8;
                                if (m < M) *(__nv_bfloat162*)(Y + m * O + o) = __floats2bfloat162_rn(acc[mt][nn][2 * h] + b0, acc[mt][nn][2 * h + 1] + b1);
                            }
                    }
            }
#pragma unroll
            for (int mt = 0; mt < MT; mt++)
#pragma unroll
                for (int nn = 0; nn < 8; nn++)
#pragma unroll
                    for (int q = 0; q < 4; q++) acc[mt][nn][q] = 0.f;
            if (j + 1 < n) {
                bar_sync<C::THREADS>(FULL0 + bsl);  // stage j + 1 is ready
                frags(bsl, 0, 0);
            }
        }
        if (++s == S) s = 0, p++;
    }
#elif defined(__HIP_DEVICE_COMPILE__)
    __trap();  // (Ampere's and later NVIDIA GPUs' alone: never launched on AMD)
#endif
}

// A mixture-of-experts layer: its E experts' matrices [O, K] stacked as one matrix [E O, K] in either mma
// layout (expert e's rows e O to e O + O - 1; O a multiple of 64), each multiplied by the tokens routed to it.
// A token's k choices are its pairs (pair j: token j / k, choice j mod k; ids[j] its expert). moe_route_kernel
// sorts the pairs by expert into a plan (int32): [0] the experts hit, then E entries: those experts in order,
// then E + 1: where each one's pairs start in the order (the last: the pairs routed), then P: the pairs by
// expert, in order within each (a pair whose id is outside 0 to E - 1 is routed nowhere). One warp: the counts
// and the order a chunk of 32 pairs at a time, the lanes with the same expert counted by __match_any_sync.
__global__ void moe_route_kernel(const int64_t* __restrict__ ids, int64_t P, int E, int* __restrict__ plan) {
    extern __shared__ int at[];  // an expert's pairs, then where its next one goes in the order
    int lane = threadIdx.x;
    uint32_t below = (1u << lane) - 1;
    int *hit = plan + 1, *start = plan + 1 + E, *order = plan + 2 + 2 * E;
    for (int e = lane; e < E; e += 32) at[e] = 0;
    __syncwarp();
    auto expert = [&](int64_t j) {
        int64_t e = j < P ? ids[j] : -1;
        return e >= 0 && e < E ? (int)e : -1;
    };
    for (int64_t b = 0; b < P; b += 32) {
        int e = expert(b + lane);
        uint32_t same = __match_any_sync(FULL, e);
        if (e >= 0 && !(same & below)) at[e] += __popc(same);  // the group's first lane: distinct experts, no race
        __syncwarp();
    }
    int n = 0, h = 0;
    for (int e0 = 0; e0 < E; e0 += 32) {
        int e = e0 + lane, c = e < E ? at[e] : 0, s = c;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            int v = __shfl_up_sync(FULL, s, o);
            s += lane >= o ? v : 0;
        }
        uint32_t hits = __ballot_sync(FULL, c > 0);
        if (c > 0) {
            hit[h + __popc(hits & below)] = e;
            start[h + __popc(hits & below)] = n + s - c;
        }
        if (e < E) at[e] = n + s - c;
        n += __shfl_sync(FULL, s, 31);
        h += __popc(hits);
    }
    if (lane == 0) {
        plan[0] = h;
        start[h] = n;
    }
    __syncwarp();
    for (int64_t b = 0; b < P; b += 32) {
        int e = expert(b + lane);
        uint32_t same = __match_any_sync(FULL, e);
        if (e >= 0) order[at[e] + __popc(same & below)] = (int)(b + lane);
        __syncwarp();
        if (e >= 0 && !(same & below)) at[e] += __popc(same);
        __syncwarp();
    }
}

// The same plan, bit for bit, from one block of NW warps (a prompt's thousands of pairs: the one warp above takes a global
// load's latency a group of 32, twice, 24 us for 1,856 pairs and 0.85 ms for 65,536). Warp w takes a run of whole groups
// of 32 pairs, in order; counts a group at a time (cnt[w][e], ROUTE_R groups' ids loaded at once), then each expert's counts
// are added across the warps in order (cnt becomes the pairs of the warps before), one warp scans the experts as above, and
// each warp writes its pairs at start[e] + the warps' before + the group's rank: the stable order of the one-warp kernel.
constexpr int ROUTE_R = 8;
__global__ void __launch_bounds__(1024) moe_route_block_kernel(const int64_t* __restrict__ ids, int64_t P, int E, int* __restrict__ plan) {
    extern __shared__ int rs[];  // cnt [NW][E] (a warp's pairs of an expert, then where its next one goes), tot [E] (the expert's, then its start)
    const int nw = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int* cnt = rs;
    int* tot = rs + nw * E;
    int* mine = cnt + warp * E;
    const uint32_t below = (1u << lane) - 1;
    int *hit = plan + 1, *start = plan + 1 + E, *order = plan + 2 + 2 * E;
    for (int i = threadIdx.x; i < nw * E; i += blockDim.x) cnt[i] = 0;
    const int64_t groups = (P + 31) / 32, per = (groups + nw - 1) / nw;
    const int64_t g0 = min(groups, (int64_t)warp * per), g1 = min(groups, g0 + per);
    __syncthreads();
    auto expert = [&](int64_t j) {
        int64_t e = j < P ? __ldg(ids + j) : -1;
        return e >= 0 && e < E ? (int)e : -1;
    };
    for (int64_t g = g0; g < g1; g += ROUTE_R) {
        int e[ROUTE_R];
#pragma unroll
        for (int r = 0; r < ROUTE_R; r++) e[r] = g + r < g1 ? expert((g + r) * 32 + lane) : -1;
#pragma unroll
        for (int r = 0; r < ROUTE_R; r++) {
            uint32_t same = __match_any_sync(FULL, e[r]);
            if (e[r] >= 0 && !(same & below)) mine[e[r]] += __popc(same);  // the group's first lane: distinct experts, no race
            __syncwarp();
        }
    }
    __syncthreads();
    for (int x = threadIdx.x; x < E; x += blockDim.x) {
        int a = 0;
        for (int w = 0; w < nw; w++) {
            int c = cnt[w * E + x];
            cnt[w * E + x] = a;
            a += c;
        }
        tot[x] = a;
    }
    __syncthreads();
    if (warp == 0) {
        int n = 0, h = 0;
        for (int e0 = 0; e0 < E; e0 += 32) {
            int e = e0 + lane, c = e < E ? tot[e] : 0, s = c;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                int v = __shfl_up_sync(FULL, s, o);
                s += lane >= o ? v : 0;
            }
            uint32_t hits = __ballot_sync(FULL, c > 0);
            if (c > 0) {
                hit[h + __popc(hits & below)] = e;
                start[h + __popc(hits & below)] = n + s - c;
            }
            if (e < E) tot[e] = n + s - c;
            n += __shfl_sync(FULL, s, 31);
            h += __popc(hits);
        }
        if (lane == 0) {
            plan[0] = h;
            start[h] = n;
        }
    }
    __syncthreads();
    for (int64_t g = g0; g < g1; g += ROUTE_R) {
        int e[ROUTE_R];
#pragma unroll
        for (int r = 0; r < ROUTE_R; r++) e[r] = g + r < g1 ? expert((g + r) * 32 + lane) : -1;
#pragma unroll
        for (int r = 0; r < ROUTE_R; r++) {
            uint32_t same = __match_any_sync(FULL, e[r]);
            if (e[r] >= 0) order[tot[e[r]] + mine[e[r]] + __popc(same & below)] = (int)((g + r) * 32 + lane);
            __syncwarp();
            if (e[r] >= 0 && !(same & below)) mine[e[r]] += __popc(same);
            __syncwarp();
        }
    }
}

// The experts' product for the pairs of a plan, 16 MT at a time: a block a unit of a hit expert's rows, blockIdx.y
// the hit (past those hit: nothing to do), blockIdx.x the unit; X's row for the i-th pair of the order: token
// order[i] / k (gather) or i. ACT 0: a unit a row block; its rows' sums (+ bias [E, O]) to Y [pairs, O], the
// pairs in the plan's order, or (Y32) times the pair's weight w (bf16, or fp32: wf32) to row j of Y32 [P, O] for
// pair j. ACT 1 (SiLU) and 2 (GELU, tanh): a unit a row block of the gate's half (the first O / 2 rows of the
// expert's) and the same of the up's (the rest): Y [pairs, O / 2] = act(gate) up, each + its bias. The 8 warps
// split K (with the gate: even warps the gate's rows, odd the up's) and add their sums through mma_gemm_kernel's
// mma_steps and warp_sums. Few units (few experts hit) split K over gridDim.z blocks (blockIdx.z's share of the
// steps): each writes its sums to parts ([splits][P][units][64, or with the gate 128] floats, P the pairs), and
// the last of a unit's to finish (last_of; done: a counter a unit) adds them in order and writes the unit out: the
// same result every run.
template <class Fmt, int MT, int ACT, int V = 0>  // V: the step loop's variants, bit 0 Nib::patch_b, bit 1 the m-tiles with pairs alone, bit 2 the L2 prefetch
__global__ void __launch_bounds__(256, MT == 1 ? ((V & 32) ? 3 : 2) : 1) mma_moe_kernel(Fmt f, int64_t O, int64_t K, const int* __restrict__ plan, int E, int64_t k, int gather, const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ bias, const void* __restrict__ w, int wf32, __nv_bfloat16* __restrict__ Y, float* __restrict__ Y32, int64_t P, float* __restrict__ parts, int* __restrict__ done, int pf) {
    extern __shared__ __align__(128) float red[];  // [4 warps][16 MT rows][65], then the warps' scratch [8][S2_BYTES]
    __shared__ uint32_t tab[Fmt::kTable ? 256 : 1];
    __shared__ int last;
    if ((int)blockIdx.y >= __ldg(plan)) return;  // the whole block: no expert this far down the hits
    if constexpr (Fmt::kTable) fill_groups(tab);
    __syncthreads();
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    uint32_t* s2 = (uint32_t*)(red + 4 * MT * 16 * 65) + warp * (S2_BYTES / 4);
    if constexpr (!Fmt::kTable) {  // 12-bit: the warp's scratch for a step's many exceptions (decode_x: zero between steps)
        ((uint4*)s2)[2 * lane] = ((uint4*)s2)[2 * lane + 1] = make_uint4(0u, 0u, 0u, 0u);
        __syncwarp();
    }
    const int* order = plan + 2 + 2 * E;
    int e = __ldg(plan + 1 + blockIdx.y), p0 = __ldg(plan + 1 + E + blockIdx.y), p1 = __ldg(plan + 2 + E + blockIdx.y);
    int64_t RB = O / 64, KS = K / 16, chunks = ACT ? 4 : 8, chunk = ACT ? warp >> 1 : warp;
    int64_t rb = blockIdx.x + (ACT && (warp & 1) ? RB / 2 : 0);  // this warp's row block of the expert's
    int64_t k0 = blockIdx.z * KS / gridDim.z, k1 = (blockIdx.z + 1) * KS / gridDim.z;  // this block's steps
    int64_t per = (k1 - k0 + chunks - 1) / chunks, s0 = min(k1, k0 + chunk * per), s1 = min(k1, s0 + per);
    int64_t base = ((int64_t)e * RB + rb) * KS;
    constexpr int W = ACT ? 128 : 64;  // a unit's columns in parts
    // A pair's sums (the i-th of the order; column c of the unit's; ACT: gate a, up b) written out.
    auto finish = [&](int64_t i, int c, float a, float b) {
        if (ACT) {
            int64_t o = blockIdx.x * 64 + c, at = (int64_t)e * O + o;  // the gate's row; the up's O / 2 on
            float gate = a + (bias ? __bfloat162float(bias[at]) : 0.f), up = b + (bias ? __bfloat162float(bias[at + O / 2]) : 0.f);
            Y[i * (O / 2) + o] = __float2bfloat16((ACT == 1 ? silu(gate) : gelu_tanh(gate)) * up);
        } else {
            int64_t o = blockIdx.x * 64 + c;
            float y = a + (bias ? __bfloat162float(bias[(int64_t)e * O + o]) : 0.f);
            if (Y32) {
                int j = __ldg(order + i);
                Y32[(int64_t)j * O + o] = y * (wf32 ? ((const float*)w)[j] : __bfloat162float(((const __nv_bfloat16*)w)[j]));
            } else {
                Y[i * O + o] = __float2bfloat16(y);
            }
        }
    };
    for (int m0 = p0; m0 < p1; m0 += 16 * MT) {
        int n = min(16 * MT, p1 - m0);
        // This thread's rows of X: rows g and g + 8 of each m-tile (none past the pairs).
        const __nv_bfloat16* xr[MT][2];
#pragma unroll
        for (int mt = 0; mt < MT; mt++)
#pragma unroll
            for (int hh = 0; hh < 2; hh++) {
                int r = mt * 16 + g + 8 * hh;
                xr[mt][hh] = r < n ? X + (gather ? (int64_t)(__ldg(order + m0 + r) / k) : (int64_t)(m0 + r)) * K + t * 2 : nullptr;
            }
        float acc[MT][8][4];
#pragma unroll
        for (int mt = 0; mt < MT; mt++)
#pragma unroll
            for (int nn = 0; nn < 8; nn++)
#pragma unroll
                for (int q = 0; q < 4; q++) acc[mt][nn][q] = 0.f;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 800
        constexpr int U = std::is_same_v<Fmt, Nib> && MT == 4 && ACT ? 1 : 0;  // an exception a pass: see Nib::patch
#else
        constexpr int U = 0;
#endif
        if constexpr ((V & 16) && MT == 1 && has_stage<Fmt>::value) {
            constexpr int P = (V & 64) ? 8 : 4;
            if constexpr ((V & 128) != 0)
                mma_steps_staged_s<Fmt, 4>(f, base, s0, s1, xr, lane, (uint8_t*)(red + 4 * MT * 16 * 65) + 8 * S2_BYTES + warp * 8 * ring_slot<Fmt>(), tab, acc);
            else
                mma_steps_staged<Fmt, P>(f, base, s0, s1, xr, lane, (uint8_t*)(red + 4 * MT * 16 * 65) + 8 * S2_BYTES + warp * P * ring_slot<Fmt>(), tab, acc);
        } else {
            mma_steps<Fmt, MT, !Fmt::kTable, U, (V & 1) != 0, (V & 2) != 0, (V & 4) != 0, (V & 8) && MT == 1 ? 2 : 1>(f, base, s0, s1, xr, lane, s2, tab, acc, pf, (n + 15) / 16);
        }
        warp_sums<MT>(acc, red, warp, g, t);
        for (int i = threadIdx.x; i < n * 64; i += 256) {
            int r = i / 64, c = i % 64;
            float v[4];
#pragma unroll
            for (int ww = 0; ww < 4; ww++) v[ww] = red[(ww * MT * 16 + r) * 65 + c];
            float a = ACT ? v[0] + v[2] : v[0] + v[1] + v[2] + v[3], b = ACT ? v[1] + v[3] : 0.f;
            if (gridDim.z == 1) {
                finish(m0 + r, c, a, b);
            } else {
                float* pp = parts + (((int64_t)blockIdx.z * P + m0 + r) * gridDim.x + blockIdx.x) * W + c;
                pp[0] = a;
                if (ACT) pp[64] = b;
            }
        }
        __syncthreads();  // red is reused
    }
    if (gridDim.z > 1 && last_of(done + (int64_t)blockIdx.y * gridDim.x + blockIdx.x, gridDim.z, last))
        for (int i = threadIdx.x; i < (p1 - p0) * 64; i += 256) {
            int r = i / 64, c = i % 64;
            float a = 0.f, b = 0.f;
            for (int z = 0; z < (int)gridDim.z; z++) {
                const float* pp = parts + (((int64_t)z * P + p0 + r) * gridDim.x + blockIdx.x) * W + c;
                a += __ldcg(pp);
                if (ACT) b += __ldcg(pp + 64);
            }
            finish(p0 + r, c, a, b);
        }
}

// The experts' product with no gate (the down), a warp a row block over the whole of K: a block 8 row blocks of a hit
// expert (blockIdx.x, blockIdx.y the hit), the warps' sums not added across warps (none in shared memory, no barrier
// but the tiered layout's table), each thread writing its own, 16 MT pairs at a time as mma_moe_kernel does. K = 768 is
// 48 steps a warp where mma_moe_kernel's 8 warps split it 6 each, and a block's start and its sums in shared memory cost
// as much as its steps: 512 blocks of a long loop instead of 4,096 of a short one (Qwen3-30B-A3B's down, 32 row blocks).
// Y [pairs, O] in the plan's order (+ bias [E, O]), or (Y32) times the pair's weight w to row j of Y32 [P, O] for pair j.
template <class Fmt, int MT, int V>
__global__ void __launch_bounds__(256, MT == 1 ? 2 : 1) mma_moe_rows_kernel(Fmt f, int64_t O, int64_t K, const int* __restrict__ plan, int E, int64_t k, int gather, const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ bias, const void* __restrict__ w, int wf32, __nv_bfloat16* __restrict__ Y, float* __restrict__ Y32, int pf) {
    extern __shared__ __align__(128) uint32_t rows_scratch[];  // the warps' scratch [8][S2_BYTES]
    __shared__ uint32_t tab[Fmt::kTable ? 256 : 1];
    if ((int)blockIdx.y >= __ldg(plan)) return;  // the whole block: no expert this far down the hits
    if constexpr (Fmt::kTable) fill_groups(tab);
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    uint32_t* s2 = rows_scratch + warp * (S2_BYTES / 4);
    if constexpr (!Fmt::kTable) {  // 12-bit: the warp's scratch for a step's many exceptions (decode_x: zero between steps)
        ((uint4*)s2)[2 * lane] = ((uint4*)s2)[2 * lane + 1] = make_uint4(0u, 0u, 0u, 0u);
        __syncwarp();
    }
    if constexpr (Fmt::kTable) __syncthreads();
    const int* order = plan + 2 + 2 * E;
    int e = __ldg(plan + 1 + blockIdx.y), p0 = __ldg(plan + 1 + E + blockIdx.y), p1 = __ldg(plan + 2 + E + blockIdx.y);
    int64_t RB = O / 64, KS = K / 16, rb = (int64_t)blockIdx.x * 8 + warp;
    if (rb >= RB) return;
    int64_t base = ((int64_t)e * RB + rb) * KS;
    for (int m0 = p0; m0 < p1; m0 += 16 * MT) {
        int n = min(16 * MT, p1 - m0);
        const __nv_bfloat16* xr[MT][2];
#pragma unroll
        for (int mt = 0; mt < MT; mt++)
#pragma unroll
            for (int hh = 0; hh < 2; hh++) {
                int r = mt * 16 + g + 8 * hh;
                xr[mt][hh] = r < n ? X + (gather ? (int64_t)(__ldg(order + m0 + r) / k) : (int64_t)(m0 + r)) * K + t * 2 : nullptr;
            }
        float acc[MT][8][4];
#pragma unroll
        for (int mt = 0; mt < MT; mt++)
#pragma unroll
            for (int nn = 0; nn < 8; nn++)
#pragma unroll
                for (int q = 0; q < 4; q++) acc[mt][nn][q] = 0.f;
        mma_steps<Fmt, MT, !Fmt::kTable, 0, (V & 1) != 0, (V & 2) != 0, (V & 4) != 0>(f, base, 0, KS, xr, lane, s2, tab, acc, pf, (n + 15) / 16);
#pragma unroll
        for (int mt = 0; mt < MT; mt++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                int r = mt * 16 + g + 8 * h;
                if (r >= n) continue;
                float wj = 0.f;
                int64_t j = 0;
                if (Y32) {
                    j = __ldg(order + m0 + r);
                    wj = wf32 ? ((const float*)w)[j] : __bfloat162float(((const __nv_bfloat16*)w)[j]);
                }
#pragma unroll
                for (int nn = 0; nn < 8; nn++) {
                    int64_t o = rb * 64 + nn * 8 + t * 2;
                    float y0 = acc[mt][nn][2 * h] + (bias ? __bfloat162float(bias[(int64_t)e * O + o]) : 0.f), y1 = acc[mt][nn][2 * h + 1] + (bias ? __bfloat162float(bias[(int64_t)e * O + o + 1]) : 0.f);
                    if (Y32) *(float2*)(Y32 + j * O + o) = make_float2(y0 * wj, y1 * wj);
                    else *(__nv_bfloat162*)(Y + (int64_t)(m0 + r) * O + o) = __floats2bfloat162_rn(y0, y1);
                }
            }
    }
}

// A weighted product's sum: y [T, O] = each token's k rows of y32 [T k, O] added in order, in fp32 (a pair
// routed nowhere adds nothing); 4 columns a thread.
__global__ void moe_sum_kernel(const float* __restrict__ y32, const int64_t* __restrict__ ids, int E, int64_t T, int64_t k, int64_t O, __nv_bfloat16* __restrict__ y) {
    int64_t i = (blockIdx.x * (int64_t)blockDim.x + threadIdx.x) * 4;
    if (i >= T * O) return;
    int64_t tok = i / O, c = i - tok * O;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int64_t j = tok * k; j < tok * k + k; j++) {
        int64_t e = __ldg(ids + j);
        if (e < 0 || e >= E) continue;
        float4 u = __ldg((const float4*)(y32 + j * O + c));
        v.x += u.x;
        v.y += u.y;
        v.z += u.z;
        v.w += u.w;
    }
    __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y), hi = __floats2bfloat162_rn(v.z, v.w);
    *(uint2*)(y + i) = make_uint2(*(uint32_t*)&lo, *(uint32_t*)&hi);
}

// The experts' products on Hopper (sm_90a), the 12-bit layout, by wgmma: mma12_tma_kernel's plan (W decoded into wgmma's A registers
// a stage of 4 steps, the pairs as B through a tensor map, a TMA warp feeding a ring of stages) over the tiles of a plan. A tile is a
// hit expert's chunk of NT pairs (moe_wg_tiles_kernel's record: the expert, the chunk's first row of the order, its pairs); a unit a
// tile by a row unit (two row blocks, one a warpgroup: 128 rows; with the gate, a gate row block and its up's, so that
// SiLU(gate) up is made from both warpgroups' sums), the blocks (one an SM) taking units in turn, each unit's K in order and its
// sums written out when it is done (no K split, no parts). X's rows are the order's: the gate and up product's through
// moe_wg_gather_kernel (a TMA tile is rows in a row), the down's the gate and up's output as it was written (the order). With the
// gate (ACT 1): y [pairs, O / 2] = SiLU(gate) up, + biases [E, O], in the order; without: the order's rows of y [., O] (+ bias),
// or (Y32) times the pair's weight to row j of Y32 [P, O] for pair j, as mma_moe_kernel does.
//
// moe_wg_tiles_kernel: one block, the tiles' records after a header {the records, 0, 0, 0}: a record is {expert, first row of
// the order, pairs (at most NT), 1 where the expert's pairs are one chunk}. The hits are scanned 1,024 at a time.
__global__ void __launch_bounds__(1024) moe_wg_tiles_kernel(const int* __restrict__ plan, int E, int NT, int4* __restrict__ tiles) {
    __shared__ int wsum[32];
    const int h = __ldg(plan), lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int base = 0;
    for (int g0 = 0; g0 < h; g0 += 1024) {
        int i = g0 + threadIdx.x, p0 = 0, p1 = 0, v = 0;
        if (i < h) {
            p0 = __ldg(plan + 1 + E + i), p1 = __ldg(plan + 2 + E + i);
            v = (p1 - p0 + NT - 1) / NT;
        }
        int x = v;  // the inclusive scan over the warp, then over the warps' sums
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            int y = __shfl_up_sync(FULL, x, d);
            if (lane >= d) x += y;
        }
        if (lane == 31) wsum[warp] = x;
        __syncthreads();
        if (warp == 0) {
            int z = wsum[lane];
#pragma unroll
            for (int d = 1; d < 32; d <<= 1) {
                int y = __shfl_up_sync(FULL, z, d);
                if (lane >= d) z += y;
            }
            wsum[lane] = z;
        }
        __syncthreads();
        int at = base + (x - v) + (warp ? wsum[warp - 1] : 0), total = wsum[31];
        if (i < h) {
            const int e = __ldg(plan + 1 + i), whole = p1 - p0 <= NT;
            for (int c = 0; c < v; c++) tiles[1 + at + c] = make_int4(e, p0 + c * NT, min(NT, p1 - p0 - c * NT), whole);
        }
        base += total;
        __syncthreads();  // wsum is reused
    }
    if (threadIdx.x == 0) tiles[0] = make_int4(base, 0, 0, 0);
}

// X's rows in the order's: out [pairs, K] row i = x row order[i] / k (the pairs the plan routes: start[h] of them), 16 bytes a thread.
__global__ void moe_wg_gather_kernel(const uint4* __restrict__ x, const int* __restrict__ plan, int E, int64_t k, int64_t K8, uint4* __restrict__ out) {
    const int* order = plan + 2 + 2 * E;
    const int64_t pairs = __ldg(plan + 1 + E + __ldg(plan));
    for (int64_t i = blockIdx.x; i < pairs; i += gridDim.x) {
        const uint4* src = x + (int64_t)(__ldg(order + i) / k) * K8;
        for (int64_t c = threadIdx.x; c < K8; c += blockDim.x) out[i * K8 + c] = __ldg(src + c);
    }
}

template <int NT, int ACT> struct Wm12 {
    static constexpr int THREADS = 288;  // two consumer warpgroups (a row block each), then the TMA warp
    static constexpr int XB = NT * 128;  // X's tile a stage: NT pairs by 64 columns
    static constexpr int CB = 4 * (int)STEP12, EB = 512, BB = 64;  // a row block's steps a stage, its exceptions (up to 128), their bounds and the unit's record
    static constexpr int RBB = CB + EB + BB;
    static constexpr int SLOT = (XB + 2 * RBB + 1023) / 1024 * 1024;
    static constexpr int XCH = ACT ? NT * 256 : 0;  // the gate's and the up's sums exchanged: 2 halves of NT / 16 float4s a thread of a warpgroup
    static constexpr int SCR = 8 * 1024;            // the consumer warps' exceptions' scratch, 1 KB each
    static constexpr int NSF = (227 * 1024 - 1024 - SCR - XCH - 256) / SLOT;
    static constexpr int NS = NSF < 8 ? NSF : 8;    // stages in flight
    static constexpr int SHARED = NS * SLOT + 16 * NS + SCR + XCH + 1024;  // the barriers, the scratch, the exchange, the alignment
};

// A unit's sums out of the registers (the kernel's last step; a test's alone, on any GPU: moe_wg_epilogue_test_kernel): d as wgmma leaves
// it, thread ct of the two warpgroups (256 threads, named barrier 1) holding rows 16w + g (+ 8) of its row block and pairs 8jn + 2t (+ 1)
// of the tile as d[4jn + 2h + c]; info: the tile's first row of the order, its pairs, its expert, the unit's row unit.
template <int NT, int ACT>
__device__ __forceinline__ void wm_epilogue(float (&d)[NT / 2], const int4 info, int ct, int64_t O, int E, const int* __restrict__ plan, float4* xch, const __nv_bfloat16* __restrict__ bias, const void* __restrict__ wts, int wf32, __nv_bfloat16* __restrict__ Y, float* __restrict__ Y32) {
    const int wg = ct >> 7, w = (ct >> 5) & 3, g = (ct & 31) >> 2, t = ct & 3;
    const int64_t RB = O / 64;
    const int* order = plan + 2 + 2 * E;
    const int row0 = info.x, np = info.y, ex = info.z, ru = info.w;
    if constexpr (ACT) {
        // The gate (warpgroup 0) and the up (1) of the same rows and pairs sit in the same thread of each: the first half of the
        // pairs is finished by warpgroup 0 (it takes the up's from 1), the second by 1 (the gate's from 0), through xch: [the
        // first half's | the second's][NT / 16 float4s][128 threads]. (d is indexed by constants alone: both halves are
        // read and one chosen.)
        constexpr int HALF = NT / 16;
        const int cl = ct & 127;
#pragma unroll
        for (int jh = 0; jh < HALF; jh++) {
            const float4 lo = make_float4(d[4 * jh], d[4 * jh + 1], d[4 * jh + 2], d[4 * jh + 3]);  // 1 gives the up's of the first half,
            const float4 hi = make_float4(d[4 * (HALF + jh)], d[4 * (HALF + jh) + 1], d[4 * (HALF + jh) + 2], d[4 * (HALF + jh) + 3]);  // 0 the gate's of the second
            xch[((wg ? 0 : HALF) + jh) * 128 + cl] = wg ? lo : hi;
        }
        bar_sync<256>(1);
#pragma unroll
        for (int jh = 0; jh < HALF; jh++) {
            const int jn = wg ? HALF + jh : jh;
            const float mine[4] = {wg ? d[4 * (HALF + jh)] : d[4 * jh], wg ? d[4 * (HALF + jh) + 1] : d[4 * jh + 1], wg ? d[4 * (HALF + jh) + 2] : d[4 * jh + 2], wg ? d[4 * (HALF + jh) + 3] : d[4 * jh + 3]};
            const float4 o4 = xch[((wg ? HALF : 0) + jh) * 128 + cl];  // the other's, for the same rows and pairs
            const float other[4] = {o4.x, o4.y, o4.z, o4.w};
#pragma unroll
            for (int h = 0; h < 2; h++)
#pragma unroll
                for (int c = 0; c < 2; c++) {
                    const int m = 8 * jn + 2 * t + c;
                    if (m < np) {
                        const int64_t o = (int64_t)ru * 64 + 16 * w + g + 8 * h, at = (int64_t)ex * O + o;
                        const float gate = (wg ? other[2 * h + c] : mine[2 * h + c]) + (bias ? __bfloat162float(bias[at]) : 0.f);
                        const float up = (wg ? mine[2 * h + c] : other[2 * h + c]) + (bias ? __bfloat162float(bias[at + O / 2]) : 0.f);
                        Y[(int64_t)(row0 + m) * (O / 2) + o] = __float2bfloat16(silu(gate) * up);
                    }
                }
        }
        bar_sync<256>(1);  // the exchange is free for the next unit
    } else {
        const int64_t rb = 2 * (int64_t)ru + wg;
        if (rb < RB) {
#pragma unroll
            for (int jn = 0; jn < NT / 8; jn++)
#pragma unroll
                for (int c = 0; c < 2; c++) {
                    const int m = 8 * jn + 2 * t + c;
                    if (m < np) {
                        const int64_t i = row0 + m, jp = Y32 ? __ldg(order + i) : 0;
                        const float wj = Y32 ? (wf32 ? ((const float*)wts)[jp] : __bfloat162float(((const __nv_bfloat16*)wts)[jp])) : 1.f;
#pragma unroll
                        for (int h = 0; h < 2; h++) {
                            const int64_t o = rb * 64 + 16 * w + g + 8 * h;
                            const float y = d[4 * jn + 2 * h + c] + (bias ? __bfloat162float(bias[(int64_t)ex * O + o]) : 0.f);
                            if (Y32) Y32[jp * O + o] = y * wj;
                            else Y[i * O + o] = __float2bfloat16(y);
                        }
                    }
                }
        }
    }
}

// The kernel's two loops, as functions of what each stage does (the kernel's: copies and products; moe_wg_walk_test_kernel's: signatures written and
// checked, on any GPU).
// A block's units: block b takes units b, b + nb, ... of RU row units by the tiles; `my` of them.
__device__ __forceinline__ int wm_units(const int4* __restrict__ tiles, int RU, int nb) {
    const int64_t total = (int64_t)RU * __ldg(&tiles[0].x);
    return total > (int64_t)blockIdx.x ? (int)((total - blockIdx.x + nb - 1) / nb) : 0;
}

// The first steps (in the pack's order) of the unit's two row blocks (past O: the last one's, its products not written; with the gate: the
// gate's block ru and the up's RB / 2 + ru), the tile's record (expert, first row of the order, pairs, 1 where the expert's pairs are one chunk).
template <int ACT>
__device__ __forceinline__ int64_t wm_row_block(int64_t RB, int ru, int r) { return ACT ? (r ? RB / 2 + ru : ru) : min((int64_t)2 * ru + r, RB - 1); }

// The TMA warp's loop over a block's stages: lanes 0-9 load the next stage's bounds (row block lane / 5, exc_base at its step lane mod 5 of
// the 4 and the next) while lane 0 issues this one; the next unit's record is loaded a unit ahead. wait_empty(slot, parity): until the
// consumers are done with the stage before in the slot; issue(j, slot, s0, tile record, row unit, the row blocks' first steps, bounds).
template <int ACT, int NS, class Wait, class Issue>
__device__ __forceinline__ void wm_produce(const int4* __restrict__ tiles, const int* __restrict__ exc_base, int64_t RB, int64_t KS, int RU, int S, int nb, int my, int lane, Wait wait_empty, Issue issue) {
    const int n = my * S;
    int uk = 0, s = 0, ru = 0;
    int4 rec = make_int4(0, 0, 0, 0), recn = rec;
    int64_t b0 = 0, b1 = 0;  // the unit's two row blocks' first steps
    auto tile_of = [&](int k, int& row_unit) {  // the tile's index of this block's unit k, its row unit in row_unit
        const int64_t u = blockIdx.x + (int64_t)k * nb, q = u / RU;
        row_unit = (int)(u - q * RU);
        return (int)q;
    };
    if (n > 0) {
        int dummy;
        rec = __ldg(tiles + 1 + tile_of(0, ru));
        if (my > 1) recn = __ldg(tiles + 1 + tile_of(1, dummy));
        b0 = ((int64_t)rec.x * RB + wm_row_block<ACT>(RB, ru, 0)) * KS, b1 = ((int64_t)rec.x * RB + wm_row_block<ACT>(RB, ru, 1)) * KS;
    }
    auto bound = [&]() { return lane < 10 ? __ldg(exc_base + (lane < 5 ? b0 : b1) + 4 * s + lane % 5) : 0; };
    int ev = n > 0 ? bound() : 0;
    for (int j = 0; j < n; j++) {
        int e[10];
#pragma unroll
        for (int i = 0; i < 10; i++) e[i] = __shfl_sync(FULL, ev, i);
        const int sl = j % NS, s0 = s, cru = ru;
        const int4 cur = rec;
        const int64_t cb0 = b0, cb1 = b1;
        if (++s == S) {
            s = 0;
            if (++uk < my) {
                rec = recn;
                tile_of(uk, ru);
                if (uk + 1 < my) {
                    int dummy;
                    recn = __ldg(tiles + 1 + tile_of(uk + 1, dummy));
                }
                b0 = ((int64_t)rec.x * RB + wm_row_block<ACT>(RB, ru, 0)) * KS, b1 = ((int64_t)rec.x * RB + wm_row_block<ACT>(RB, ru, 1)) * KS;
            }
        }
        if (j + 1 < n) ev = bound();
        if (lane == 0) {
            if (j >= NS) wait_empty(sl, (j / NS - 1) & 1);  // the consumers are done with stage j - NS
            issue(j, sl, s0, cur, cru, cb0, cb1, e);
        }
        __syncwarp();
    }
}

// The consumers' loop: a unit's S stages in pairs (stage_a: even ones, with `first` set for a unit's first; stage_b: odd ones), an odd one
// after them, then finish() (every path gives each set of A registers to the products before it is decoded into again: a pair's second
// stage skipped inside the loop had ptxas serialize every product, its C7513).
template <class StageA, class StageB, class Finish>
__device__ __forceinline__ void wm_consume(int my, int S, StageA stage_a, StageB stage_b, Finish finish) {
    for (int j = 0, u = 0; u < my; u++, j += S) {
        int k = 0;
        for (; k + 1 < S; k += 2) {
            stage_a(j + k, k == 0);
            stage_b(j + k + 1);
        }
        if (k < S) stage_a(j + k, k == 0);
        finish();
    }
}

template <int NT, int ACT>
__global__ void __launch_bounds__((Wm12<NT, ACT>::THREADS), 1) mma12_moe_wg_kernel(const __grid_constant__ CUtensorMap xmap, Nib f, int64_t O, int64_t K, int E, const int* __restrict__ plan, const int4* __restrict__ tiles, int RU, const __nv_bfloat16* __restrict__ bias, const void* __restrict__ wts, int wf32, __nv_bfloat16* __restrict__ Y, float* __restrict__ Y32) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    using C = Wm12<NT, ACT>;
    extern __shared__ __align__(1024) uint8_t smem_raw[];  // NS slots [X tile | a row block's steps, exceptions, bounds and record | the other's], the barriers, the scratch, the exchange
    uint32_t raw = (uint32_t)__cvta_generic_to_shared(smem_raw), base = (raw + 1023) & ~1023u;
    uint8_t* gbase = smem_raw + (base - raw);
    uint32_t full = base + C::NS * C::SLOT, empty = full + 8 * C::NS;
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int64_t KS = K / 16, RB = O / 64;
    const int S = (int)(K / 64), nb = gridDim.x;
    // The units: row units by the tiles (a tile's row units together), block b taking units b, b + nb, ...
    const int my = wm_units(tiles, RU, nb);  // this block's units
    if (tid == 0) {
        for (int i = 0; i < C::NS; i++) {
            mbar_init(full + 8 * i, 1);
            mbar_init(empty + 8 * i, 8);  // each consumer warp
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();
    if (warp == 8) {
        wm_produce<ACT, C::NS>(
            tiles, f.exc_base, RB, KS, RU, S, nb, my, lane, [&](int sl, uint32_t parity) { mbar_wait(empty + 8 * sl, parity); },
            [&](int, int sl, int s0, const int4& cur, int cru, int64_t cb0, int64_t cb1, const int (&e)[10]) {
                uint32_t fb = full + 8 * sl, xs = base + sl * C::SLOT, ws = xs + C::XB;
                uint32_t bytes = C::XB + 2 * C::CB;
                int a[2], na[2];
#pragma unroll
                for (int r = 0; r < 2; r++) {
                    exc_copy(e[5 * r], e[5 * r + 4], C::EB, a[r], na[r]);
                    bytes += na[r] > 0 ? 4 * na[r] : 0;
                    int* bs = (int*)(gbase + sl * C::SLOT + C::XB + r * C::RBB + C::CB + C::EB);
#pragma unroll
                    for (int i = 0; i < 5; i++) bs[i] = e[5 * r + i];
                    bs[5] = na[r] < 0 ? -1 : a[r];
                    bs[6] = cur.y, bs[7] = cur.z, bs[8] = cur.x, bs[9] = cru;  // the tile's first row, its pairs, the expert, the row unit
                }
                mbar_expect_tx(fb, bytes);
                tma_2d(xs, &xmap, s0 * 64, cur.y, fb);
#pragma unroll
                for (int r = 0; r < 2; r++) {
                    uint32_t cs = ws + r * C::RBB;
                    bulk_g2s(cs, f.data + ((r ? cb1 : cb0) + 4 * s0) * STEP12, C::CB, fb, cur.w != 0);  // (read once but where the expert's chunks share it)
                    if (na[r] > 0) bulk_g2s(cs + C::CB, f.exc + a[r], 4 * na[r], fb);
                }
            });
        return;
    }
    // Consumer warpgroup wg: row block 2 ru + wg (with the gate: ru, and RB / 2 + ru for the up) of each unit; warp w its rows 16w to 16w + 15.
    int ct = tid, wg = ct >> 7, w = (ct >> 5) & 3;
    uint32_t* xw = (uint32_t*)(gbase + C::NS * C::SLOT + 16 * C::NS) + 256 * warp;  // this warp's exceptions' scratch
    ((uint4*)xw)[2 * lane] = ((uint4*)xw)[2 * lane + 1] = make_uint4(0u, 0u, 0u, 0u);
    __syncwarp();
    float4* xch = (float4*)(gbase + C::NS * C::SLOT + 16 * C::NS + C::SCR);  // the exchange (ACT)
    float d[NT / 2] = {};  // rows 16w + g (+ 8), pairs 8jn + 2t (+ 1): d[4jn + 2h + c]; set by a unit's first product
    // (zeroed all the same: left unset, CUDA 12.8's ptxas saw it defined inside the loop and serialized every wgmma, its C7515)
    uint32_t A0[4][4], A1[4][4];
    int held = -1;  // the slot whose products may still be running
    int4 info = make_int4(0, 0, 0, 0);  // the unit's tile as its last stage had it: first row, pairs, expert, row unit
    auto release = [&](int sl) {  // the slot is free for stage j + NS (lane 0 arrives, by predicate: no branch among the products)
        __syncwarp();
        asm volatile("{\n.reg .pred p;\nsetp.eq.u32 p, %1, 0;\n@p mbarrier.arrive.shared::cta.b64 _, [%0];\n}\n" ::"r"(empty + 8 * sl), "r"(lane) : "memory");
    };
    auto keep_d = [&]() {  // d stays in its registers while the products run
#pragma unroll
        for (int i = 0; i < NT / 2; i++) asm volatile("" : "+f"(d[i])::"memory");
    };
    auto keep = [&](uint32_t(&A)[4][4]) {  // A is held until its products are done
#pragma unroll
        for (int kk = 0; kk < 4; kk++)
#pragma unroll
            for (int i = 0; i < 4; i++) asm volatile("" : "+r"(A[kk][i])::"memory");
    };
    auto stage = [&](int j, bool first, uint32_t(&A)[4][4], uint32_t(&Aprev)[4][4]) {
        int sl = j % C::NS;
        mbar_wait(full + 8 * sl, (j / C::NS) & 1);
        const uint8_t* sp = gbase + sl * C::SLOT + C::XB + wg * C::RBB;
        const int* bs = (const int*)(sp + C::CB + C::EB);
        int eb[5];
#pragma unroll
        for (int i = 0; i < 5; i++) eb[i] = bs[i];
        info = make_int4(bs[6], bs[7], bs[8], bs[9]);
        decode12_rows(f, sp, (const uint32_t*)(sp + C::CB), eb, bs[5], lane, w, xw, A);
        uint32_t xs = base + sl * C::SLOT;
        uint64_t desc[4];
#pragma unroll
        for (int kk = 0; kk < 4; kk++) desc[kk] = sw128_desc(xs + kk * 32);
        keep_d();
        wg_fence();
        wgmma4_rs(d, A, desc, !first);
        wg_wait<1>();  // the stage before's products are done: its A set and its slot are free
        keep(Aprev);
        keep_d();
        if (held >= 0) release(held);
        held = sl;
    };
    wm_consume(
        my, S, [&](int j, bool first) { stage(j, first, A0, A1); }, [&](int j) { stage(j, false, A1, A0); },
        [&]() {
            wg_wait<0>();
            keep(A0);
            keep(A1);
            keep_d();
            if (held >= 0) release(held);
            held = -1;
            wm_epilogue<NT, ACT>(d, info, ct, O, E, plan, xch, bias, wts, wf32, Y, Y32);
        });
#elif defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    __trap();  // sm_90a code, launched from a build without it (or on AMD)
#endif
}

// wm_produce and wm_consume with signatures in place of the copies and the products (tests, any GPU): the producer writes each stage's tile
// record, row unit, stage, the row blocks' first steps and the bounds it loaded into a slot of a ring of NS (a few: the slots and the
// barriers' phases wrap often; two at least: a slot is released a stage after its stage, as the kernel does) and arrives; the consumers check them against what the tile table and exc_base give for that stage of that
// block, and a unit's last record goes to result [8 + 8 (block * max_units + unit)]: row0, pairs, expert, row unit, the stages so far; result[0]
// the checks' failures (ORed) or -1 where a wait did not end (the waits here are bounded: a broken protocol ends the kernel).
template <int ACT, int NS>
__global__ void __launch_bounds__(288, 1) moe_wg_walk_test_kernel(const int4* __restrict__ tiles, const int* __restrict__ exc_base, int64_t RB, int64_t KS, int RU, int S, int* __restrict__ result, int max_units) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    extern __shared__ __align__(16) uint8_t wsm[];
    const uint32_t full = (uint32_t)__cvta_generic_to_shared(wsm), empty = full + 8 * NS;
    int* slots = (int*)(wsm + 16 * NS);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, nb = gridDim.x;
    if (tid == 0) {
        for (int i = 0; i < NS; i++) {
            mbar_init(full + 8 * i, 1);
            mbar_init(empty + 8 * i, 8);
        }
    }
    __syncthreads();
    int timed_out = 0;
    auto wait = [&](uint32_t bar, uint32_t parity) {
        for (int i = 0; i < (1 << 22); i++) {
            uint32_t ok;
            asm volatile("{\n.reg .pred p;\nmbarrier.test_wait.parity.shared.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n" : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
            if (ok) return;
        }
        timed_out = 1;
    };
    const int my = wm_units(tiles, RU, nb);
    if (warp == 8) {
        wm_produce<ACT, NS>(
            tiles, exc_base, RB, KS, RU, S, nb, my, lane, [&](int sl, uint32_t parity) { wait(empty + 8 * sl, parity); },
            [&](int j, int sl, int s0, const int4& cur, int cru, int64_t cb0, int64_t cb1, const int (&e)[10]) {
                int* sg = slots + 32 * sl;
                sg[0] = cur.x, sg[1] = cur.y, sg[2] = cur.z, sg[3] = cur.w, sg[4] = cru, sg[5] = s0, sg[6] = (int)cb0, sg[7] = (int)cb1, sg[18] = j;
                for (int i = 0; i < 10; i++) sg[8 + i] = e[i];
                mbar_arrive(full + 8 * sl);
            });
        if (timed_out) atomicOr(result, -1);
        return;
    }
    int bad = 0, held = -1, units = 0, stages = 0;
    int4 info = make_int4(0, 0, 0, 0);
    auto release = [&](int sl) {
        __syncwarp();
        asm volatile("{\n.reg .pred p;\nsetp.eq.u32 p, %1, 0;\n@p mbarrier.arrive.shared::cta.b64 _, [%0];\n}\n" ::"r"(empty + 8 * sl), "r"(lane) : "memory");
    };
    auto stage = [&](int j) {
        const int sl = j % NS;
        wait(full + 8 * sl, (j / NS) & 1);
        const int* sg = slots + 32 * sl;
        const int uk = j / S, s = j % S;
        const int64_t u = blockIdx.x + (int64_t)uk * nb, q = u / RU;
        const int ru = (int)(u - q * RU);
        const int4 rec = tiles[1 + q];
        const int64_t b0 = ((int64_t)rec.x * RB + wm_row_block<ACT>(RB, ru, 0)) * KS, b1 = ((int64_t)rec.x * RB + wm_row_block<ACT>(RB, ru, 1)) * KS;
        bad |= (sg[0] != rec.x) | (sg[1] != rec.y) | (sg[2] != rec.z) | (sg[3] != rec.w) | (sg[4] != ru) | (sg[5] != s) | (sg[6] != (int)b0) | (sg[7] != (int)b1) | (sg[18] != j);
        for (int i = 0; i < 10; i++) bad |= sg[8 + i] != exc_base[(i < 5 ? b0 : b1) + 4 * s + i % 5];
        info = make_int4(sg[1], sg[2], sg[0], sg[4]);
        stages++;
        if (held >= 0) release(held);
        held = sl;
    };
    wm_consume(
        my, S, [&](int j, bool) { stage(j); }, [&](int j) { stage(j); },
        [&]() {
            if (held >= 0) release(held);
            held = -1;
            if (tid == 0 && units < max_units) {
                int* r = result + 8 + 8 * ((int64_t)blockIdx.x * max_units + units);
                r[0] = info.x, r[1] = info.y, r[2] = info.z, r[3] = info.w, r[4] = stages;
            }
            units++;
        });
    if (bad | timed_out) atomicOr(result, timed_out ? -1 : bad);
#elif defined(__HIP_DEVICE_COMPILE__)
    __trap();  // (Ampere's and later NVIDIA GPUs' alone: never launched on AMD)
#endif
}

// wm_epilogue alone (tests, any GPU): mat [2][64][NT] floats, warpgroup wg's row block's value at row r of the block and pair m, laid
// out in each thread's d as wgmma's D is.
template <int NT, int ACT>
__global__ void __launch_bounds__(256) moe_wg_epilogue_test_kernel(const float* __restrict__ mat, int4 info, int64_t O, int E, const int* __restrict__ plan, const __nv_bfloat16* __restrict__ bias, const void* __restrict__ wts, int wf32, __nv_bfloat16* __restrict__ Y, float* __restrict__ Y32) {
    extern __shared__ float4 xch_test[];
    const int ct = threadIdx.x, wg = ct >> 7, w = (ct >> 5) & 3, g = (ct & 31) >> 2, t = ct & 3;
    float d[NT / 2];
#pragma unroll
    for (int jn = 0; jn < NT / 8; jn++)
#pragma unroll
        for (int h = 0; h < 2; h++)
#pragma unroll
            for (int c = 0; c < 2; c++) d[4 * jn + 2 * h + c] = mat[((int64_t)wg * 64 + 16 * w + g + 8 * h) * NT + 8 * jn + 2 * t + c];
    wm_epilogue<NT, ACT>(d, info, ct, O, E, plan, xch_test, bias, wts, wf32, Y, Y32);
}

// The mma layout back to bf16, rows [row0, row0 + rows) of W (multiples of
// 64) into out [rows, K] (checks, and the many-token path that multiplies
// with PyTorch): a warp a step; FEW (a decode ahead, a few warps an SM),
// each warp its step and on by the grid's warps. MOE (exact, a mixture of
// experts' layer, W its experts' matrices of `rows` rows stacked):
// blockIdx.y a hit expert of plan (moe_route's; past those hit: nothing to
// do), its rows into the same rows of out [E rows, K], the rest of out left
// as it is. The 12-bit layout's steps: a warp a step on Hopper by
// Nib::load_decode (the low bytes first), elsewhere and FEW by load() (all at
// once), each as it measured fastest (Hopper measured on a GH200 alone; any
// other GPU not measured, as it was).
template <class Fmt, bool MOE = false, bool FEW = false>
__global__ void mma_unpack_kernel(Fmt f, int64_t K, int64_t row0, int64_t rows, uint16_t* __restrict__ out, const int* __restrict__ plan = nullptr) {
    if constexpr (MOE) {
        if ((int)blockIdx.y >= __ldg(plan)) return;  // the whole block: no expert this far down the hits
        row0 = (int64_t)__ldg(plan + 1 + blockIdx.y) * rows;
        out += row0 * K;
    }
    __shared__ uint32_t tab[Fmt::kTable ? 256 : 1];
    if constexpr (Fmt::kTable) fill_groups(tab);
    __syncthreads();
    int64_t KS = K / 16, total = rows / 64 * KS;
    int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    __shared__ uint4 scratch[8][Fmt::kTable ? S2_BYTES / 16 : 1];
    uint32_t* out32 = (uint32_t*)out;
    for (int64_t local = (blockIdx.x * (int64_t)blockDim.x + threadIdx.x) >> 5; local < total; local += (int64_t)gridDim.x * blockDim.x >> 5) {
        int64_t step = row0 / 64 * KS + local, rb, s;
        uint32_t R[16];
#if __CUDA_ARCH__ == 900
        constexpr bool low_first = std::is_same_v<Fmt, Nib> && !FEW;
#else
        constexpr bool low_first = false;
#endif
        if constexpr (low_first) {
            f.load_decode(step, lane, R, [&] { rb = local / KS, s = local % KS; });
        } else {
            rb = local / KS, s = local % KS;
            typename Fmt::St st;
            f.load(st, step, lane);
            f.decode(st, lane, (uint32_t*)scratch[threadIdx.x >> 5], tab, R);
        }
        // R[2n]: row 8n + g, columns 2t and 2t + 1; R[2n + 1]: columns 8 + 2t, 9 + 2t.
#pragma unroll
        for (int n = 0; n < 8; n++) {
            int64_t at2 = ((rb * 64 + 8 * n + g) * K + s * 16 + 2 * t) / 2;
            out32[at2] = R[2 * n];
            out32[at2 + 4] = R[2 * n + 1];
        }
        if constexpr (!FEW) break;
    }
}

// Rows of W [., K] by id (an embedding's lookup; the ids on the device, so a batch's lookup is one launch whatever they
// are, and a CUDA graph can hold it) into out [n, K]: a warp a (row, step), the step decoded as mma_unpack_kernel decodes
// it, and only the picked row's 16 weights written. Row r of its block of 64 is row 8 m + g of the step (r = 8 m + g), in
// the lanes 4 g to 4 g + 3: R[2 m] and R[2 m + 1] of each, its columns 2 t, 2 t + 1 and 8 + 2 t, 9 + 2 t. An id outside
// [0, rows) has no row: zeros (nothing is read).
template <class Fmt>
__global__ void mma_rows_kernel(Fmt f, int64_t K, int64_t rows, const int64_t* __restrict__ ids, int64_t n, uint16_t* __restrict__ out) {
    __shared__ uint32_t tab[Fmt::kTable ? 256 : 1];
    if constexpr (Fmt::kTable) fill_groups(tab);
    __syncthreads();
    __shared__ uint4 scratch[8][Fmt::kTable ? S2_BYTES / 16 : 1];
    int64_t KS = K / 16, unit = (blockIdx.x * (int64_t)blockDim.x + threadIdx.x) >> 5;
    if (unit >= n * KS) return;  // (a whole warp: a unit is a warp's)
    int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    int64_t row = unit / KS, s = unit % KS, id = (int64_t)__ldg((const long long*)ids + row);
    uint32_t* o = (uint32_t*)out + (row * K + s * 16) / 2;
    if ((uint64_t)id >= (uint64_t)rows) {
        if (lane < 8) o[lane] = 0;
        return;
    }
    int r = (int)(id % 64), pick = r >> 3;
    uint32_t R[16];
    typename Fmt::St st;
    f.load(st, id / 64 * KS + s, lane);
    f.decode(st, lane, (uint32_t*)scratch[threadIdx.x >> 5], tab, R);
    if (g == (r & 7)) {
        uint32_t a = 0, b = 0;
#pragma unroll
        for (int m = 0; m < 8; m++) {
            if (m == pick) {
                a = R[2 * m];
                b = R[2 * m + 1];
            }
        }
        o[t] = a;
        o[4 + t] = b;
    }
}

// Option 2's decode (the route SPLIT: glyd_gpu_mma12_unpack_part, glyd_gpu_mma12_ring_*): the 12-bit layout's rows
// [row0, row0 + rows) into out [rows, K] on the few SMs a green context sets apart for it, the products on the rest. A
// warp takes units of 4 steps of a row block (64 rows x 64 columns), grid-stride, the next unit's steps loaded into
// registers while this one's are decoded; each step's B-fragment words go into the warp's tile in shared memory
// (16-byte chunk c of row r at c ^ (r & 7): its 32-bit stores and 16-byte loads without bank conflicts), and out in
// whole 128-byte lines. mma_unpack_kernel's step a warp with 4-byte stores is bound by its latency on few SMs (2.7
// weights a clock an SM on an RTX 4080 SUPER, research-2026-09-29/box/dec2.txt); this runs 9.6 on an A100 and 10.2-10.4
// on an H100 SXM and PCIe on 16 SMs, its memory path the bound (benchmarks/gpu/research-2026-09-29/round1).
constexpr int SPLIT_U = 4, SPLIT_WARPS = 4;

__global__ void __launch_bounds__(32 * SPLIT_WARPS) mma12_split_kernel(Nib f, int64_t K, int64_t row0, int64_t rows, uint16_t* __restrict__ out) {
    __shared__ __align__(16) uint32_t tiles[SPLIT_WARPS][64 * 32];  // a warp's unit: 64 rows of 32 words
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    uint32_t* tile = tiles[threadIdx.x >> 5];
    const int KS = (int)(K / 16), per_rb = KS / SPLIT_U, units = (int)(rows / 64) * per_rb;  // (units < 2^31: split_decode)
    const int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, nw = (gridDim.x * blockDim.x) >> 5;
    const int64_t s0 = row0 / 64 * KS;
    auto first = [&](int u) { return s0 + (int64_t)(u / per_rb) * KS + (u % per_rb) * SPLIT_U; };  // a unit's first step
    Nib::St next[SPLIT_U];
    if (gw < units) {
#pragma unroll
        for (int j = 0; j < SPLIT_U; j++) f.load(next[j], first(gw) + j, lane);
    }
    for (int u = gw; u < units; u += nw) {
        Nib::St cur[SPLIT_U];
#pragma unroll
        for (int j = 0; j < SPLIT_U; j++) cur[j] = next[j];
        if (u + nw < units) {
#pragma unroll
            for (int j = 0; j < SPLIT_U; j++) f.load(next[j], first(u + nw) + j, lane);
        }
#pragma unroll
        for (int j = 0; j < SPLIT_U; j++) {
            uint32_t R[16];
            f.decode(cur[j], lane, nullptr, nullptr, R);
#pragma unroll
            for (int n = 0; n < 8; n++) {  // R[2n]: row 8n + g, word t of step j; R[2n + 1]: word 4 + t (and (8n + g) & 7 == g)
                tile[(8 * n + g) * 32 + (((2 * j) ^ g) << 2) + t] = R[2 * n];
                tile[(8 * n + g) * 32 + (((2 * j + 1) ^ g) << 2) + t] = R[2 * n + 1];
            }
        }
        __syncwarp();
        const int64_t rb = u / per_rb, col = (int64_t)(u % per_rb) * SPLIT_U * 16;
#pragma unroll
        for (int i = 0; i < 16; i++) {  // 8 lanes a 128-byte line, 4 rows an instruction
            int r = 4 * i + (lane >> 3), c = lane & 7;
            *(uint4*)(out + (rb * 64 + r) * K + col + 8 * c) = *(const uint4*)(tile + r * 32 + ((c ^ (r & 7)) << 2));
        }
        __syncwarp();
    }
}

// A product's bias as its output's every row (the route SPLIT: cuBLAS then adds X W^T, one rounding).
__global__ void bias_rows_kernel(uint16_t* __restrict__ y, const uint16_t* __restrict__ bias, int64_t M, int64_t O) {
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < M * O; i += (int64_t)gridDim.x * blockDim.x) y[i] = bias[i % O];
}

// Attention for one new token a sequence (decoding) over a KV cache held in
// the mma layout (gpu/kv.py): a layer's keys as [pages x pairs x 64 tokens,
// D] (a pair: one sequence's KV head), its values as [pages x pairs x D, 64
// tokens], then a tail of tlen (< 64) tokens in bf16, [pairs, tlen, D]. The
// G queries a KV head serves are one m16 tile. A block takes one pair and a
// run of its pages (the last run also the tail), a page a warp at a time: its
// keys decoded into B fragments, the queries' scores on the tensor cores, an
// online softmax (in base 2), its values decoded, P V accumulated. The warps
// are merged in shared memory, the pair's blocks by the last to finish, in
// order (the same result every run).
constexpr int ATT_WARPS = 4;

__device__ __forceinline__ uint32_t bf16x2(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *(uint32_t*)&v;
}

template <int D>
__global__ void __launch_bounds__(32 * ATT_WARPS) attn_decode_kernel(const __nv_bfloat16* __restrict__ q, const uint8_t* __restrict__ kd, const uint8_t* __restrict__ kb, const int32_t* __restrict__ kbb, Tiers kt, const uint8_t* __restrict__ vd, const uint8_t* __restrict__ vb, const int32_t* __restrict__ vbb, Tiers vt, const __nv_bfloat16* __restrict__ tk, const __nv_bfloat16* __restrict__ tv, int tlen, int pairs, int G, int P, int per, float sl2, float* __restrict__ part, int* __restrict__ done, __nv_bfloat16* __restrict__ out) {
    constexpr int KS = D / 16, NT = D / 8, RV = D / 64;  // key steps a page, n-tiles of D, value row blocks a page
    __shared__ uint32_t tab[256];
    __shared__ __align__(16) uint32_t scratch[ATT_WARPS][S2_BYTES / 4];
    __shared__ float red[ATT_WARPS][16][D + 2];  // a warp's rows: O, then its max and sum
    __shared__ int last;
    fill_groups(tab);
    __syncthreads();
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    int pair = blockIdx.x, split = blockIdx.y, splits = gridDim.y;
    uint32_t* s2 = scratch[warp];
    // The queries as A fragments: rows g and g + 8 (zero past G).
    uint32_t aq[KS][4];
    const __nv_bfloat16* qp = q + (int64_t)pair * G * D;
#pragma unroll
    for (int s = 0; s < KS; s++) {
        int c = 16 * s + 2 * t;
        aq[s][0] = g < G ? *(const uint32_t*)(qp + g * D + c) : 0u;
        aq[s][1] = g + 8 < G ? *(const uint32_t*)(qp + (g + 8) * D + c) : 0u;
        aq[s][2] = g < G ? *(const uint32_t*)(qp + g * D + c + 8) : 0u;
        aq[s][3] = g + 8 < G ? *(const uint32_t*)(qp + (g + 8) * D + c + 8) : 0u;
    }
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f}, o[NT][4];
#pragma unroll
    for (int n = 0; n < NT; n++) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    int p0 = split * per, p1 = min(P + (tlen > 0 ? 1 : 0), p0 + per);  // page P: the tail
    for (int p = p0 + warp; p < p1; p += ATT_WARPS) {
        float c[8][4];
#pragma unroll
        for (int n = 0; n < 8; n++) c[n][0] = c[n][1] = c[n][2] = c[n][3] = 0.f;
        if (p < P) {
            int64_t rb = (int64_t)p * pairs + pair;
#pragma unroll 1
            for (int s = 0; s < KS; s++) {
                Step st;
                uint32_t R[16];
                load_step(st, kd, kb, kbb, rb * KS + s, lane);
                decode_step(st, kt, kb, lane, s2, tab, R);
#pragma unroll
                for (int n = 0; n < 8; n++) mma16816(c[n], aq[s], R[2 * n], R[2 * n + 1]);
            }
        } else {
            const __nv_bfloat16* kp = tk + (int64_t)pair * tlen * D;
#pragma unroll
            for (int s = 0; s < KS; s++)
#pragma unroll
                for (int n = 0; n < 8; n++) {
                    int tok = 8 * n + g;
                    uint32_t b0 = tok < tlen ? *(const uint32_t*)(kp + tok * D + 16 * s + 2 * t) : 0u;
                    uint32_t b1 = tok < tlen ? *(const uint32_t*)(kp + tok * D + 16 * s + 8 + 2 * t) : 0u;
                    mma16816(c[n], aq[s], b0, b1);
                }
        }
        // The page's scores: rows g (c[.][0..1]) and g + 8 (c[.][2..3]), columns 8n + 2t, 8n + 2t + 1.
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int n = 0; n < 8; n++) {
            int tok = 8 * n + 2 * t;
            if (p >= P && tok >= tlen) c[n][0] = c[n][2] = -INFINITY;
            if (p >= P && tok + 1 >= tlen) c[n][1] = c[n][3] = -INFINITY;
            mx0 = fmaxf(mx0, fmaxf(c[n][0], c[n][1]));
            mx1 = fmaxf(mx1, fmaxf(c[n][2], c[n][3]));
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(FULL, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(FULL, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(FULL, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(FULL, mx1, 2));
        float n0 = fmaxf(m[0], mx0 * sl2), n1 = fmaxf(m[1], mx1 * sl2);
        float a0 = exp2f(m[0] - n0), a1 = exp2f(m[1] - n1);
        m[0] = n0;
        m[1] = n1;
        l[0] *= a0;
        l[1] *= a1;
#pragma unroll
        for (int n = 0; n < NT; n++) {
            o[n][0] *= a0;
            o[n][1] *= a0;
            o[n][2] *= a1;
            o[n][3] *= a1;
        }
        // P as A fragments of the values' product: tokens 16s.. of n-tiles 2s, 2s + 1.
        uint32_t ap[4][4];
#pragma unroll
        for (int s = 0; s < 4; s++) {
            float e[2][4];
#pragma unroll
            for (int h = 0; h < 2; h++) {
                int n = 2 * s + h;
                e[h][0] = exp2f(c[n][0] * sl2 - n0);
                e[h][1] = exp2f(c[n][1] * sl2 - n0);
                e[h][2] = exp2f(c[n][2] * sl2 - n1);
                e[h][3] = exp2f(c[n][3] * sl2 - n1);
                l[0] += e[h][0] + e[h][1];
                l[1] += e[h][2] + e[h][3];
            }
            ap[s][0] = bf16x2(e[0][0], e[0][1]);
            ap[s][1] = bf16x2(e[0][2], e[0][3]);
            ap[s][2] = bf16x2(e[1][0], e[1][1]);
            ap[s][3] = bf16x2(e[1][2], e[1][3]);
        }
        if (p < P) {
            int64_t rb = ((int64_t)p * pairs + pair) * RV;
#pragma unroll
            for (int j = 0; j < RV; j++)
#pragma unroll 1
                for (int s = 0; s < 4; s++) {
                    Step st;
                    uint32_t R[16];
                    load_step(st, vd, vb, vbb, (rb + j) * 4 + s, lane);
                    decode_step(st, vt, vb, lane, s2, tab, R);
#pragma unroll
                    for (int n = 0; n < 8; n++) mma16816(o[8 * j + n], ap[s], R[2 * n], R[2 * n + 1]);
                }
        } else {
            const uint16_t* vp = (const uint16_t*)(tv + (int64_t)pair * tlen * D);
#pragma unroll
            for (int j = 0; j < RV; j++)
#pragma unroll
                for (int s = 0; s < 4; s++)
#pragma unroll
                    for (int n = 0; n < 8; n++) {
                        int d = 64 * j + 8 * n + g, k0 = 16 * s + 2 * t;
                        auto v = [&](int tok) -> uint32_t { return tok < tlen ? (uint32_t)vp[tok * D + d] : 0u; };
                        mma16816(o[8 * j + n], ap[s], v(k0) | v(k0 + 1) << 16, v(k0 + 8) | v(k0 + 9) << 16);
                    }
        }
    }
    // The warps' rows into shared memory (sums over the quad first), merged.
#pragma unroll
    for (int r = 0; r < 2; r++) {
        l[r] += __shfl_xor_sync(FULL, l[r], 1);
        l[r] += __shfl_xor_sync(FULL, l[r], 2);
    }
#pragma unroll
    for (int n = 0; n < NT; n++) {
        red[warp][g][8 * n + 2 * t] = o[n][0];
        red[warp][g][8 * n + 2 * t + 1] = o[n][1];
        red[warp][g + 8][8 * n + 2 * t] = o[n][2];
        red[warp][g + 8][8 * n + 2 * t + 1] = o[n][3];
    }
    if (t == 0) {
        red[warp][g][D] = m[0];
        red[warp][g][D + 1] = l[0];
        red[warp][g + 8][D] = m[1];
        red[warp][g + 8][D + 1] = l[1];
    }
    __syncthreads();
    float* mine = part + ((int64_t)pair * splits + split) * 16 * (D + 2);
    for (int i = threadIdx.x; i < 16 * (D + 2); i += 32 * ATT_WARPS) {
        int r = i / (D + 2), d = i % (D + 2);
        float M = -INFINITY;
#pragma unroll
        for (int w = 0; w < ATT_WARPS; w++) M = fmaxf(M, red[w][r][D]);
        float v = 0.f;
        if (d == D) v = M;
        else if (M != -INFINITY)
#pragma unroll
            for (int w = 0; w < ATT_WARPS; w++) v += red[w][r][d] * exp2f(red[w][r][D] - M);
        mine[i] = v;
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        last = atomicAdd(done + pair, 1) == splits - 1;
        if (last) done[pair] = 0;  // ready for the next call
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    const float* all = part + (int64_t)pair * splits * 16 * (D + 2);
    for (int i = threadIdx.x; i < G * D; i += 32 * ATT_WARPS) {
        int r = i / D, d = i % D;
        float M = -INFINITY;
        for (int sp = 0; sp < splits; sp++) M = fmaxf(M, __ldcg(all + ((int64_t)sp * 16 + r) * (D + 2) + D));
        float L = 0.f, O = 0.f;
        for (int sp = 0; sp < splits; sp++) {
            const float* row = all + ((int64_t)sp * 16 + r) * (D + 2);
            float ms = __ldcg(row + D);
            if (ms == -INFINITY) continue;
            float w = exp2f(ms - M);
            L += __ldcg(row + D + 1) * w;
            O += __ldcg(row + d) * w;
        }
        out[((int64_t)pair * G + r) * D + d] = __float2bfloat16(O / L);
    }
}

// The fast format decoded to bf16: rows [row0, row0 + rows), or the rows
// listed in row_ids, one after another into out.
__global__ void fast_decode_kernel(const uint8_t* __restrict__ sm, const uint32_t* __restrict__ planes, const uint8_t* __restrict__ exc, const int32_t* __restrict__ exc_base, uint64_t top, int64_t row0, const int64_t* __restrict__ row_ids, int64_t rows, int64_t K, uint16_t* __restrict__ out) {
    int64_t gw = (blockIdx.x * (int64_t)blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int64_t segs = (K + SEG - 1) / SEG;
    int64_t r = gw / segs, s = gw % segs;
    if (r >= rows) return;
    int64_t row = row_ids ? row_ids[r] : row0 + r;
    int64_t esc = exc_base[row * segs + s];
    for (int64_t k = s * SEG + lane * 4; k < min((s + 1) * SEG, K); k += 128) {
        int64_t q = row * K + k;
        uint32_t s4 = *(const uint32_t*)(sm + q);
        Quad d = fast_quad(planes, exc, q, top, esc, lane);
        uint32_t o0 = bf16_bits(s4 & 0xff, d.e[0]), o1 = bf16_bits((s4 >> 8) & 0xff, d.e[1]);
        uint32_t o2 = bf16_bits((s4 >> 16) & 0xff, d.e[2]), o3 = bf16_bits(s4 >> 24, d.e[3]);
        *(uint2*)(out + r * K + k) = make_uint2(o0 | (o1 << 16), o2 | (o3 << 16));
    }
}

// ---------------------------------------------------------------------------
// The lossless 12-bit KV cache (glyd_gpu's vLLM plugin, kv: lossless): pages of 16 tokens of H kv heads of 128
// channels, bf16 values held in 12 bits each and read back bit for bit.
//
// A value is its bf16 low byte as it is (the exponent's lowest bit and the 7 mantissa bits) and a 4-bit code for the high
// byte: the sign and an offset 0-7 from a base, the window of 8 values of exponent >> 1 that holds the most of the
// (layer, head, channel)'s values in other text (kbase, vbase: uint8 [H][D] a layer, calibrated once a model). A value
// outside its window is an exception: its code is offset 0, and an entry in its head's header, its page's arena or its page's chunks XORs
// the byte base ^ (exponent >> 1) into the decoded high byte, so that every 16-bit pattern (NaN, infinity, denormals,
// both zeros) comes back as it went in.
//
// A page, for H heads and an arena of A entries (hdr_bytes(H, A) = 16 H + 4 (1 + ND + A) up to a multiple of 128):
//   0                  H head headers of 16 bytes: u32 count (the head-page's entries, K and V, dead ones too), u32 e0, e1, e2
//   16 H               u32 used (the entries past the heads' inline ones that the page has been given, dead ones and dropped ones too),
//                      u32 dir[ND] (the escape pool's chunks the page holds: the chunk's number + 1, 0 none), then A u32 entries (the arena)
//   hdr_bytes(H, A)    H head-pages of 6,144 bytes: the K tile of 3,072 bytes, then the V tile
// A page's entries past the three inline in its head's header are its entries number 0, 1, 2 ...: number i < A is arena[i], and number i >= A is
// entry (i - A) % CH of the chunk dir[(i - A) / CH] of the escape pool (EscapePool: chunks of CH entries a process keeps apart from its pages, taken
// as an entry needs one and given back when the page is cleared). The arena is what the header's last bytes up to a multiple of 128 leave: 15
// entries for 8 heads, and a page that needs more takes a chunk (a page at most A + ND CH entries).
// An entry (u32): lane 0-4, group 5-8, byte 9-10, the XOR byte 11-18, token 19-22, kind (0 K, 1 V) 23, head 24-28,
// live 29 (a rewritten token's old entries are killed, not moved).
//
// Raw pages: a page that has no place for an entry is stored without exceptions, its extra bits in a slab of a raw pool; a page that finds no slab is void and its block is named in the status ring.
//
// Writing is owned: one CTA a run of the step's tokens that fall in one page (runs found from the slots alone), a warp a
// head; the warp alone writes its head's tile bytes and header, so no word is written by two threads; the page's entry
// counter, and the escape pool's chunks it takes, are the atomics. A page whose first token is written is cleared first (its
// headers, its counter, its directory, the chunks it held given back to the pool): a recycled block's bytes are never assumed
// zero. A token written again kills its old entries first (in its head's header, the arena and the chunks); a page's first
// token written again clears the page, so a rewrite from there on writes every token after it too.
namespace kvc {

constexpr int D = 128, PT = 16, TILE = 3072, HEADPAGE = 2 * TILE;
constexpr int NW = 4;  // warps a CTA of the attention kernel
static_assert(NW * 32 == D, "the attention kernel's CTA merges its warps' outputs a thread a channel");
constexpr unsigned FULLM = 0xffffffffu;
constexpr uint32_t LIVE = 1u << 29;
constexpr int ND = 16;  // chunks of the escape pool a page's header can name
constexpr int CH = 64;  // entries in a chunk of the escape pool
constexpr uint32_t PG_RAW = 1u << 31, PG_VOID = 1u << 30, PG_COUNT = (1u << 24) - 1u;  // flags and count of a page's `used` word
constexpr uint32_t HEAD_RAW = 0xFFFFFFFFu;  // a raw head-page's count word
// the write kernel's status words (glyd_gpu.h); the two rings follow them: the blocks with a void page, the blocks with a raw page, `ring` places each
enum { ST_OVERFLOW = 0, ST_PAGE_HW = 1, ST_ARENA_PAGES = 2, ST_ENTRIES = 3, ST_POOL_USED = 4, ST_POOL_HW = 5, ST_POOL_PAGES = 6, ST_STALE = 7, ST_RAW_USED = 8, ST_RAW_HW = 9, ST_RAW_PAGES = 10, ST_VOID = 11,
       ST_RING_HEAD = 12, ST_RAW_RING_HEAD = 13, ST_RING = 16 };

__host__ __device__ constexpr int hdr_bytes(int H, int A) { return (16 * H + 4 * (1 + ND + A) + 127) & ~127; }
// A head-page's bytes: its K tile and its V tile, or (one stream: a tensor held in the K tile's layout alone) its one tile.
__host__ __device__ constexpr int64_t head_page(bool one) { return one ? TILE : HEADPAGE; }
__host__ __device__ constexpr int64_t page_bytes(int H, int A, bool one = false) { return hdr_bytes(H, A) + (int64_t)H * head_page(one); }

template <int OFF>
__device__ __forceinline__ uint4 ldg_stream(const uint8_t* p) {  // a streaming read: through the read-only path, no L1 line kept
#if defined(__HIP_PLATFORM_AMD__)
    return *(const uint4*)(p + OFF);
#else
    uint4 r;
    asm("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4+%5];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p), "n"(OFF));
    return r;
#endif
}
__device__ __forceinline__ float ex2(float x) {
#if defined(__HIP_PLATFORM_AMD__)
    return __builtin_amdgcn_exp2f(x);
#else
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
#endif
}
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&v);
}
__device__ __forceinline__ uint32_t transpose8x8(uint32_t a) {  // the 8 x 8 matrix of b16 a warp holds (a lane: row lane / 4, columns 2 (lane % 4) and the next), transposed
#if defined(__HIP_PLATFORM_AMD__)
    return glyd_transpose8x8(a);
#else
    uint32_t d;
    asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(d) : "r"(a));
    return d;
#endif
}
__device__ __forceinline__ void mma(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
#if defined(__HIP_PLATFORM_AMD__)
    const uint32_t a[4] = {a0, a1, a2, a3};
    glyd_mma16816(c, a, b0, b1);
#else
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ uint32_t entry(int lane, int group, int byte, uint32_t x, int tok, int kind, int head) {
    return (uint32_t)lane | ((uint32_t)group << 5) | ((uint32_t)byte << 9) | (x << 11) | ((uint32_t)tok << 19) | ((uint32_t)kind << 23) | ((uint32_t)head << 24) | LIVE;
}
__device__ __forceinline__ int e_tok(uint32_t e) { return (e >> 19) & 15; }
__device__ __forceinline__ int e_kind(uint32_t e) { return (e >> 23) & 1; }
__device__ __forceinline__ int e_head(uint32_t e) { return (e >> 24) & 31; }

// ---- the escape pool ----------------------------------------------------------------------------------------------------------------
// Chunks of CH entries kept apart from the pages, for the entries past a page's arena (one pool a process: every layer's pages draw on it). A pool of nc chunks is, in one
// buffer: u32 bits[nc / 32 up to a multiple of 4] (a chunk's bit set: taken), u32 owner[nc up to a multiple of 4] (the page a taken chunk is on loan to: its layer's tag
// (layer + 1) << 24 and its number, 0 free), u32 ent[nc][CH]. A page names the chunks it holds in its header (dir); they go back when the page is cleared, and only
// those whose owner says they are the page's (a block whose bytes were never written by this pool names no chunk it can free).
struct EscapePool {
    uint32_t* bits;
    uint32_t* owner;
    uint32_t* ent;
    int nc;  // chunks (0: none)
};
__host__ __device__ constexpr int64_t esc_bits_bytes(int64_t nc) { return (((nc + 31) / 32 + 3) / 4) * 16; }
__host__ __device__ constexpr int64_t esc_owner_bytes(int64_t nc) { return ((nc + 3) / 4) * 16; }
__host__ __device__ constexpr int64_t esc_bytes(int64_t nc) { return esc_bits_bytes(nc) + esc_owner_bytes(nc) + nc * CH * 4; }
static EscapePool esc_of(uint8_t* base, int64_t nc) {
    if (!nc || !base) return EscapePool{nullptr, nullptr, nullptr, 0};
    return EscapePool{(uint32_t*)base, (uint32_t*)(base + esc_bits_bytes(nc)), (uint32_t*)(base + esc_bits_bytes(nc) + esc_owner_bytes(nc)), (int)nc};
}

// A chunk taken (its number, or -1 where none is free): the bitmap searched from `hint`'s word, a bit claimed by atomicOr.
__device__ __forceinline__ int esc_alloc(const EscapePool& E, uint32_t tag, uint32_t hint, uint32_t* status) {
    const int nw = (E.nc + 31) >> 5;
    for (int probe = 0; probe < nw; probe++) {
        const int w = (int)((hint + (uint32_t)probe) % (uint32_t)nw);
        const uint32_t valid = (w == nw - 1 && (E.nc & 31)) ? (1u << (E.nc & 31)) - 1u : 0xFFFFFFFFu;
        uint32_t x = *(volatile uint32_t*)(E.bits + w);
        while ((~x) & valid) {
            const int b = __ffs((~x) & valid) - 1;
            const uint32_t old = atomicOr(E.bits + w, 1u << b);
            if (!(old & (1u << b))) {
                E.owner[w * 32 + b] = tag;
                const uint32_t use = atomicAdd(status + ST_POOL_USED, 1u) + 1u;
                atomicMax(status + ST_POOL_HW, use);
                return w * 32 + b;
            }
            x = old | (1u << b);
        }
    }
    return -1;
}
__device__ __forceinline__ void esc_free(const EscapePool& E, int c, uint32_t* status) {
    E.owner[c] = 0;
    __threadfence();
    atomicAnd(E.bits + (c >> 5), ~(1u << (c & 31)));
    atomicSub(status + ST_POOL_USED, 1u);
}
// The page's chunk j (dirw = &dir[j]: the chunk + 1, 0 none): taken where the page has none yet (two warps of a page may both try: the loser gives its chunk back and uses the
// winner's); 0 where the pool is out of chunks.
__device__ __forceinline__ uint32_t esc_chunk(const EscapePool& E, uint32_t* dirw, uint32_t tag, uint32_t hint, uint32_t* status) {
    const uint32_t d = *(volatile uint32_t*)dirw;
    if (d) return d - 1 < (uint32_t)E.nc ? d : 0u;
    if (!E.nc) return 0u;
    const int c = esc_alloc(E, tag, hint, status);
    if (c < 0) return 0;
    const uint32_t old = atomicCAS(dirw, 0u, (uint32_t)c + 1u);
    if (old != 0u) {
        esc_free(E, c, status);
        return old;
    }
    return (uint32_t)c + 1u;
}
// ---- the raw pool ---------------------------------------------------------------------------------------------------------------
// Bitmap, owners and ns slabs of plane_bytes(H), in one buffer, as the escape pool's.
struct RawPool {
    uint32_t* bits;
    uint32_t* owner;
    uint8_t* data;
    int ns;
    int64_t plane;  // a slab's bytes
};
__host__ __device__ constexpr int64_t raw_bits_bytes(int64_t ns) { return (((ns + 31) / 32 + 3) / 4) * 16; }
__host__ __device__ constexpr int64_t raw_owner_bytes(int64_t ns) { return ((ns + 3) / 4) * 16; }
__host__ __device__ constexpr int64_t plane_bytes(int H) { return (int64_t)H * 2048; }
__host__ __device__ constexpr int64_t raw_bytes(int64_t ns, int H) { return raw_bits_bytes(ns) + raw_owner_bytes(ns) + ns * plane_bytes(H); }
static RawPool raw_of(uint8_t* base, int64_t ns, int H) {
    if (!ns || !base) return RawPool{nullptr, nullptr, nullptr, 0, 0};
    return RawPool{(uint32_t*)base, (uint32_t*)(base + raw_bits_bytes(ns)), base + raw_bits_bytes(ns) + raw_owner_bytes(ns), (int)ns, plane_bytes(H)};
}
__device__ __forceinline__ int raw_alloc(const RawPool& R, uint32_t tag, uint32_t hint, uint32_t* status) {
    const int nw = (R.ns + 31) >> 5;
    for (int probe = 0; probe < nw; probe++) {
        const int w = (int)((hint + (uint32_t)probe) % (uint32_t)nw);
        const uint32_t valid = (w == nw - 1 && (R.ns & 31)) ? (1u << (R.ns & 31)) - 1u : 0xFFFFFFFFu;
        uint32_t x = *(volatile uint32_t*)(R.bits + w);
        while ((~x) & valid) {
            const int b = __ffs((~x) & valid) - 1;
            const uint32_t old = atomicOr(R.bits + w, 1u << b);
            if (!(old & (1u << b))) {
                R.owner[w * 32 + b] = tag;
                const uint32_t use = atomicAdd(status + ST_RAW_USED, 1u) + 1u;
                atomicMax(status + ST_RAW_HW, use);
                return w * 32 + b;
            }
            x = old | (1u << b);
        }
    }
    return -1;
}
__device__ __forceinline__ void raw_free(const RawPool& R, int c, uint32_t* status) {
    R.owner[c] = 0;
    __threadfence();
    atomicAnd(R.bits + (c >> 5), ~(1u << (c & 31)));
    atomicSub(status + ST_RAW_USED, 1u);
}

// The word of a page's entry number idx (pg: the page's `used` word, then its dir and arena), in its arena or in the chunk its directory names; null where that chunk is missing, or
// the directory's word is not a chunk of this pool (a page that was never written holds a previous owner's bytes, and a decode of one must not go where they point).
__device__ __forceinline__ const uint32_t* entry_at(const uint32_t* pg, const uint32_t* ent, int nc, int A, uint32_t idx) {
    if (idx < (uint32_t)A) return pg + 1 + ND + idx;
    const uint32_t r = idx - (uint32_t)A;
    const uint32_t d = __ldg(pg + 1 + r / CH);
    return d - 1 < (uint32_t)nc ? ent + (size_t)(d - 1) * CH + r % CH : nullptr;
}

// The (token, channel) of value m of group i of lane (g, q).
template <int KIND>  // 0: K, 1: V
__device__ __forceinline__ void coord(int g, int q, int i, int m, int& tok, int& ch) {
    if (KIND == 0) {
        tok = 8 * (i & 1) + g;
        ch = 16 * (i >> 1) + 2 * q + (m & 1) + 8 * (m >> 1);
    } else {
        ch = g + 8 * i;
        tok = 2 * q + (m & 1) + 8 * (m >> 1);
    }
}

// The bases as bytes of the lane's channels: K: a word a code word i (the channels of its group's step, 16 s + 2 q + {0, 1, 8, 9});
// V: a word a group, the channel g + 8 i's base in every byte.
template <int KIND>
__device__ __forceinline__ void load_bases(const uint8_t* base, int lane, uint32_t (&hb)[KIND == 0 ? 8 : 16]) {
    const int g = lane >> 2, q = lane & 3;
    if (KIND == 0) {
#pragma unroll
        for (int s = 0; s < 8; s++) hb[s] = (uint32_t)base[16 * s + 2 * q] | ((uint32_t)base[16 * s + 2 * q + 1] << 8) | ((uint32_t)base[16 * s + 2 * q + 8] << 16) | ((uint32_t)base[16 * s + 2 * q + 9] << 24);
    } else {
#pragma unroll
        for (int i = 0; i < 16; i++) hb[i] = (uint32_t)base[g + 8 * i] * 0x01010101u;
    }
}

// ---- reading a tile ----------------------------------------------------------------------------------------------------
struct Raw {  // a lane's bytes of a tile: 4 chunks of low bytes, 2 of codes
    uint4 lo[4], cd[2];
};
__device__ __forceinline__ void raw_load(Raw& r, const uint8_t* tile, int lane) {
    const uint8_t* p = tile + lane * 16;
    r.lo[0] = ldg_stream<0>(p);
    r.lo[1] = ldg_stream<512>(p);
    r.lo[2] = ldg_stream<1024>(p);
    r.lo[3] = ldg_stream<1536>(p);
    r.cd[0] = ldg_stream<2048>(p);
    r.cd[1] = ldg_stream<2560>(p);
}

// raw_load by the ordinary path (for a tile written in this kernel).
__device__ __forceinline__ void raw_load_plain(Raw& r, const uint8_t* tile, int lane) {
    const uint4* p = reinterpret_cast<const uint4*>(tile + lane * 16);
    r.lo[0] = p[0];
    r.lo[1] = p[32];
    r.lo[2] = p[64];
    r.lo[3] = p[96];
    r.cd[0] = p[128];
    r.cd[1] = p[160];
}

// One entry: the byte XORed into the high bytes word of its group where this lane owns it (the entry is the same for the warp,
// so which word is patched is a warp-uniform branch, not a select on each of the 16).
__device__ __forceinline__ void patch(uint32_t (&H)[16], uint32_t e, int lane) {
    const uint32_t v = ((e & 31u) == (uint32_t)lane) ? (((e >> 11) & 0xFFu) << (((e >> 9) & 3u) * 8u)) : 0u;
    switch ((e >> 5) & 15u) {
#define KV_PATCH(i) \
    case i:         \
        H[i] ^= v;  \
        break;
        KV_PATCH(0) KV_PATCH(1) KV_PATCH(2) KV_PATCH(3) KV_PATCH(4) KV_PATCH(5) KV_PATCH(6) KV_PATCH(7)
        KV_PATCH(8) KV_PATCH(9) KV_PATCH(10) KV_PATCH(11) KV_PATCH(12) KV_PATCH(13) KV_PATCH(14) KV_PATCH(15)
#undef KV_PATCH
    }
}

// The entries of this head (kind KIND) applied to H: the three inline ones, and where the head holds more the page's others (its arena, then the chunks of
// the escape pool its directory names), read a line at a time (a lane an entry) and the matches taken in turn. pg: the page's `used` word.
template <int KIND>
__device__ __forceinline__ void apply_entries(uint32_t (&H)[16], const uint4 hd, const uint32_t* pg, const uint32_t* ent, int nc, int A, int head, int lane, int limit) {
    const uint32_t cnt = hd.x;
    if (cnt == 0) return;
    const uint32_t in[3] = {hd.y, hd.z, hd.w};
#pragma unroll
    for (int j = 0; j < 3; j++)
        if (j < (int)cnt && (in[j] & LIVE) && e_kind(in[j]) == KIND) patch(H, in[j], lane);
    if (cnt > 3) {
        const uint32_t used = min(limit >= 0 ? (uint32_t)limit : __ldg(pg) & PG_COUNT, (uint32_t)(A + ND * CH));  // limit >= 0: only the entries before the run
        for (uint32_t base = 0; base < used; base += 32) {
            const uint32_t idx = base + lane;
            uint32_t e = 0;
            if (idx < used) {
                const uint32_t* p = entry_at(pg, ent, nc, A, idx);
                if (p) e = __ldg(p);
            }
            uint32_t bal = __ballot_sync(FULLM, (e & LIVE) && e_kind(e) == KIND && e_head(e) == head);
            while (bal) {
                const int src = __ffs(bal) - 1;
                bal &= bal - 1;
                patch(H, __shfl_sync(FULLM, e, src), lane);
            }
        }
    }
}

struct F8 {
    uint4 v[8];
};

// A raw head-page's tile, decoded (not inlined: rare).
template <int KIND>
__device__ __noinline__ F8 decode_raw(const Raw r, const uint32_t* pg, const RawPool R, int head, int lane) {
    const uint32_t d = __ldg(pg + 1);
    uint32_t pw[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
    if (d - 1 < (uint32_t)R.ns) {
        const uint8_t* pt = R.data + (size_t)(d - 1) * (size_t)R.plane + (size_t)(2 * head + KIND) * 1024 + lane * 16;
        const uint4 a = __ldg(reinterpret_cast<const uint4*>(pt)), b = __ldg(reinterpret_cast<const uint4*>(pt + 512));
        pw[0] = a.x, pw[1] = a.y, pw[2] = a.z, pw[3] = a.w, pw[4] = b.x, pw[5] = b.y, pw[6] = b.z, pw[7] = b.w;
    }
    const uint32_t cw[8] = {r.cd[0].x, r.cd[0].y, r.cd[0].z, r.cd[0].w, r.cd[1].x, r.cd[1].y, r.cd[1].z, r.cd[1].w};
    const uint32_t L[16] = {r.lo[0].x, r.lo[0].y, r.lo[0].z, r.lo[0].w, r.lo[1].x, r.lo[1].y, r.lo[1].z, r.lo[1].w,
                            r.lo[2].x, r.lo[2].y, r.lo[2].z, r.lo[2].w, r.lo[3].x, r.lo[3].y, r.lo[3].z, r.lo[3].w};
    F8 out;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const uint32_t cb = __funnelshift_l(cw[i], cw[i], 4);
        const uint32_t H0 = (cw[i] & 0x80808080u) | ((cw[i] & 0x07070707u) << 4) | (pw[i] & 0x0F0F0F0Fu);
        const uint32_t H1 = (cb & 0x80808080u) | ((cb & 0x07070707u) << 4) | ((pw[i] >> 4) & 0x0F0F0F0Fu);
        out.v[i] = make_uint4(__byte_perm(L[2 * i], H0, 0x5140), __byte_perm(L[2 * i + 1], H1, 0x5140), __byte_perm(L[2 * i], H0, 0x7362), __byte_perm(L[2 * i + 1], H1, 0x7362));
    }
    return out;
}

// A tile's 64 values a lane as the 8 chunks of bf16 pairs the tensor cores take as A operand: f[i] = (pair 0 of group 2i, pair 0 of
// group 2i + 1, pair 1 of group 2i, pair 1 of group 2i + 1) = a0, a1, a2, a3 of step i.
template <int KIND>
__device__ __forceinline__ void decode_tile(const Raw& r, const uint32_t* hb, const uint4 hd, const uint32_t* pg, const uint32_t* ent, int nc, int A, const RawPool& R, int head, int lane, uint4 (&f)[8], int limit = -1) {
    if (__builtin_expect(hd.x == HEAD_RAW, 0)) {
        const F8 fr = decode_raw<KIND>(r, pg, R, head, lane);
#pragma unroll
        for (int i = 0; i < 8; i++) f[i] = fr.v[i];
        return;
    }
    const uint32_t cw[8] = {r.cd[0].x, r.cd[0].y, r.cd[0].z, r.cd[0].w, r.cd[1].x, r.cd[1].y, r.cd[1].z, r.cd[1].w};
    const uint32_t L[16] = {r.lo[0].x, r.lo[0].y, r.lo[0].z, r.lo[0].w, r.lo[1].x, r.lo[1].y, r.lo[1].z, r.lo[1].w,
                            r.lo[2].x, r.lo[2].y, r.lo[2].z, r.lo[2].w, r.lo[3].x, r.lo[3].y, r.lo[3].z, r.lo[3].w};
    uint32_t H[16];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        H[2 * i] = (cw[i] & 0x87878787u) + hb[KIND == 0 ? i : 2 * i];
        H[2 * i + 1] = (__funnelshift_l(cw[i], cw[i], 4) & 0x87878787u) + hb[KIND == 0 ? i : 2 * i + 1];
    }
    apply_entries<KIND>(H, hd, pg, ent, nc, A, head, lane, limit);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const uint32_t a0 = __byte_perm(L[2 * i], H[2 * i], 0x5140), a1 = __byte_perm(L[2 * i], H[2 * i], 0x7362);
        const uint32_t b0 = __byte_perm(L[2 * i + 1], H[2 * i + 1], 0x5140), b1 = __byte_perm(L[2 * i + 1], H[2 * i + 1], 0x7362);
        f[i] = make_uint4(a0, b0, a1, b1);
    }
}

// ---- writing ------------------------------------------------------------------------------------------------------------
struct WriteParams {
    const uint16_t* key;
    const uint16_t* value;
    int64_t key_stride, value_stride;
    const int64_t* slots;
    uint8_t* pool;
    int64_t page_stride;
    const uint8_t* kbase;
    const uint8_t* vbase;
    uint32_t* status;
    EscapePool esc;
    RawPool raw;
    uint32_t* lost;  // two bits a block, 2 b: it has a void page, 2 b + 1: a raw page (null: none); nlost bits
    int64_t nlost;
    uint32_t ring;  // places of each of the status's two rings
    uint32_t layer_tag;  // (layer + 1) << 24: the owner tag of this layer's pages in the pools
    int N, H, A;
    int lean;  // 0: every run through write_run (the path of a run of any length); else a run of one token through the decode step's path
    int one;   // 1: one stream (the keys alone: no value is read, a page holds no V tile)
};

__shared__ uint32_t g_need;  // an entry of the CTA's page found no place: the page is made raw (make_raw)

// The entries past a head's three inline ones, appended by a warp (all 32 lanes call it, together: obal is the lanes that have one, not 0): the lanes' places in the page's list
// of entries by one atomic on its counter (their numbers are consecutive), the chunks of the escape pool those places fall in taken first (by the first lane, before any lane
// stores), then each lane stores its entry in the arena below A, else in its chunk. An entry that has no place (the pool is out of chunks, or the page's directory is full)
// is dropped and counted (the status's overflow) and the CTA marked (g_need): it makes the page raw before it ends.
// Not inlined: it is rare (a page past its heads' three inline entries) and each call of an inlined one would be a copy of it in
// the write kernel, which has no registers to spare.
__device__ __noinline__ void append_over(const WriteParams& P, uint32_t* pg, uint32_t tag, uint32_t e, bool over, uint32_t obal, int lane) {
    const int A = P.A;
    const EscapePool esc = P.esc;
    uint32_t* status = P.status;
    const int leader = __ffs(obal) - 1;
    const uint32_t n = __popc(obal);
    uint32_t base = 0;
    if (lane == leader) {
        base = atomicAdd(pg, n) & PG_COUNT;
        if (!*(volatile uint32_t*)&g_need && base + n > (uint32_t)A) {  // the chunks these entries fall in, taken before any lane stores
            const uint32_t lo = base > (uint32_t)A ? base - A : 0u, hi = base + n - A;
            for (uint32_t j = lo / CH; j <= (hi - 1) / CH && j < (uint32_t)ND; j++) esc_chunk(esc, pg + 1 + j, tag, tag * 2654435761u + j * 40503u, status);
        }
    }
    __syncwarp();
    base = __shfl_sync(FULLM, base, leader);
    bool lost = false;
    if (over) {
        const uint32_t a = base + __popc(obal & ((1u << lane) - 1u));
        if (a < (uint32_t)A) {
            pg[1 + ND + a] = e;
        } else {
            const uint32_t r = a - A, j = r / CH;
            const uint32_t d = j < (uint32_t)ND ? *(volatile uint32_t*)(pg + 1 + j) : 0u;
            if (d - 1 < (uint32_t)esc.nc) esc.ent[(size_t)(d - 1) * CH + r % CH] = e;
            else lost = true;
        }
    }
    const uint32_t lbal = __ballot_sync(FULLM, lost);
    if (lane == leader) {
        if (lbal) {
            atomicOr(&g_need, 1u);
            atomicAdd(status + ST_OVERFLOW, (uint32_t)__popc(lbal));
        }
        atomicMax(status + ST_PAGE_HW, base + n);
        if (base == 0 && A > 0) atomicAdd(status + ST_ARENA_PAGES, 1u);
        if (base <= (uint32_t)A && base + n > (uint32_t)A) atomicAdd(status + ST_POOL_PAGES, 1u);
    }
}

struct Appender {  // a head-page's entries being appended by a warp
    uint32_t cnt;        // the entries the head-page holds (warp-uniform)
    uint32_t* hdr;       // its header: the count, then three inline entries
    uint32_t* pg;        // the page's `used` word, its directory and its arena
    uint32_t tag;        // this page's owner tag in the pool
    const WriteParams* P;  // (the kernel's parameters: read where an entry needs them, held in no register)
    // Every lane calls it (the votes are the warp's): the lanes with `mine` append e, inline while the head holds fewer than three, else to the page's list of entries (append_over).
    __device__ __forceinline__ void add(uint32_t e, bool mine, int lane) {
        const uint32_t bal = __ballot_sync(FULLM, mine);
        if (!bal) return;
        const uint32_t idx = cnt + (mine ? __popc(bal & ((1u << lane) - 1u)) : 0u);
        const bool over = mine && idx >= 3;
        if (mine && idx < 3) hdr[1 + idx] = e;
        const uint32_t obal = __ballot_sync(FULLM, over);
        if (obal) append_over(*P, pg, tag, e, over, obal, lane);
        if (lane == __ffs(bal) - 1) atomicAdd(P->status + ST_ENTRIES, (uint32_t)__popc(bal));
        cnt += __popc(bal);
    }
};

// A token's 4 values of a group, as the code bits and the exception: value v (bf16 bits), its channel's base byte b.
// Returns the 4-bit code (sign << 3 | offset, 0 for an exception); esc, x: the exception and the byte to XOR in.
__device__ __forceinline__ uint32_t code_of(uint32_t v, uint32_t b, bool& esc, uint32_t& x) {
    const uint32_t e7 = (v >> 8) & 127u;
    const int off = (int)e7 - (int)b;
    esc = off < 0 || off > 7;
    x = esc ? (b ^ e7) : 0u;
    return ((v >> 15) << 3) | (esc ? 0u : (uint32_t)off);
}

// ---- writing a run of tokens into a head-page's two tiles -------------------------------------------------------------------------
// A tile is laid out for the readers' lanes (a lane (g, q) of the decode warp owns words of 16 bytes at lane * 16 of each 512-byte chunk), but the writer's
// lanes divide the work by value, not by owner: a decode step writes one token, and a warp that gave the owners' lanes their values would have 4 to 8
// of its 32 lanes at work through 16 passes. Here a lane takes 4 values of every token of the run (K: a step s and a q, 16 s + 2 q + {0, 1, 8, 9}; V: 4 of
// the 16 channels g + 8 i of a g) and writes the words of the owner lanes they belong to, so a token is one pass over all 32 lanes, a run of 16 sixteen.
// Everything a lane reads (its values, the bases, the words it will write part of) is read first (k_load, v_load: before the kills' barrier), then
// computed and written (k_write, v_write: after it).
struct KPre {
    uint32_t kb;         // the bases of the lane's 4 channels, a byte each
    uint32_t p[2][8][2]; // [nt][g][half]: the lane's keys of token 8 nt + g (2 channels a word), where the run holds it
    uint32_t ow[8];      // [g]: the code word it writes of owner 4 g + q, where the run holds one of its two tokens
};
__device__ __forceinline__ bool k_tok(int nt, int g, int o0, int len) { return 8 * nt + g >= o0 && 8 * nt + g < o0 + len; }
__device__ __forceinline__ void k_load(KPre& r, const uint16_t* __restrict__ key, int64_t stride, const uint8_t* __restrict__ base, const uint8_t* tile, int o0, int len, int lane) {
    const int s = lane >> 2, q = lane & 3;
    r.kb = (uint32_t)base[16 * s + 2 * q] | ((uint32_t)base[16 * s + 2 * q + 1] << 8) | ((uint32_t)base[16 * s + 2 * q + 8] << 16) | ((uint32_t)base[16 * s + 2 * q + 9] << 24);
#pragma unroll
    for (int g = 0; g < 8; g++) {
#pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            const bool act = k_tok(nt, g, o0, len);
            const uint16_t* src = key + (int64_t)(8 * nt + g - o0) * stride + 16 * s + 2 * q;  // (read only where act)
            r.p[nt][g][0] = act ? __ldg(reinterpret_cast<const uint32_t*>(src)) : 0u;
            r.p[nt][g][1] = act ? __ldg(reinterpret_cast<const uint32_t*>(src + 8)) : 0u;
        }
        const bool a0 = k_tok(0, g, o0, len), a1 = k_tok(1, g, o0, len);
        r.ow[g] = (a0 != a1) ? *reinterpret_cast<const uint32_t*>(tile + 2048 + (s >> 2) * 512 + (4 * g + q) * 16 + (s & 3) * 4) : 0u;  // (a half of the word is the other token's)
    }
}
__device__ __forceinline__ void k_write(const KPre& r, uint8_t* tile, int o0, int len, int lane, int head, Appender& ap) {
    const int s = lane >> 2, q = lane & 3;
    const uint32_t cmask[2] = {0x87878787u, 0x78787878u};
#pragma unroll
    for (int g = 0; g < 8; g++) {
        const bool act[2] = {k_tok(0, g, o0, len), k_tok(1, g, o0, len)};
        if (!(act[0] || act[1])) continue;  // (g is the loop's: every lane skips alike)
        const int owner = 4 * g + q;
        uint32_t w = r.ow[g];
        uint32_t ent[2][4];
        bool esc[2][4], any = false;
#pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            const uint32_t v[4] = {r.p[nt][g][0] & 0xFFFFu, r.p[nt][g][0] >> 16, r.p[nt][g][1] & 0xFFFFu, r.p[nt][g][1] >> 16};
            const int i = 2 * s + nt;
            uint32_t lw = 0, cb = 0;
#pragma unroll
            for (int m = 0; m < 4; m++) {
                const uint32_t b = (r.kb >> (8 * m)) & 0xFFu;
                uint32_t x;
                const uint32_t code = code_of(v[m], b, esc[nt][m], x);
                esc[nt][m] &= act[nt];
                any |= esc[nt][m];
                ent[nt][m] = esc[nt][m] ? entry(owner, i, m, x, 8 * nt + g, 0, head) : 0u;
                lw |= (v[m] & 255u) << (8 * m);
                if (nt == 0) cb |= (((code >> 3) << 7) | (code & 7u)) << (8 * m);
                else cb |= (((code >> 3) << 3) << (8 * m)) | ((code & 7u) << (8 * ((m + 3) & 3) + 4));
            }
            if (act[nt]) {
                *reinterpret_cast<uint32_t*>(tile + (i >> 2) * 512 + owner * 16 + (i & 3) * 4) = lw;
                w = (w & ~cmask[nt]) | cb;
            }
        }
        *reinterpret_cast<uint32_t*>(tile + 2048 + (s >> 2) * 512 + owner * 16 + (s & 3) * 4) = w;
        if (__any_sync(FULLM, any)) {
#pragma unroll
            for (int nt = 0; nt < 2; nt++) {
#pragma unroll
                for (int m = 0; m < 4; m++) ap.add(ent[nt][m], esc[nt][m], lane);
            }
        }
    }
}

struct VPre {
    uint32_t bs[4];        // the bases of the lane's 4 channels g + 8 (4 ig + jj)
    uint16_t vv[4][4][4];  // [token group][token of it][jj]: the lane's values (a token group q' holds tokens 2 q' + {0, 1, 8, 9})
    uint4 lw[4];           // [token group]: the 4 low words (4 ig .. 4 ig + 3) it writes bytes of, of owner 4 g + q'
    uint2 cw[4];           // [token group]: the 2 code words (2 ig, 2 ig + 1)
};
__device__ __forceinline__ void v_masks(int qg, int o0, int len, uint32_t& lmask, uint32_t& cmask) {  // the bytes of a low word and the bits of a code word the run holds of token group qg
    lmask = cmask = 0;
#pragma unroll
    for (int m = 0; m < 4; m++) {
        const int t = 2 * qg + (m & 1) + 8 * (m >> 1);
        if (t >= o0 && t < o0 + len) {
            lmask |= 0xFFu << (8 * m);
            cmask |= (0x87u << (8 * m)) | (0x08u << (8 * m)) | (0x70u << (8 * ((m + 3) & 3)));
        }
    }
}
__device__ __forceinline__ void v_load(VPre& r, const uint16_t* __restrict__ value, int64_t stride, const uint8_t* __restrict__ base, const uint8_t* tile, int o0, int len, int lane) {
    const int g = lane & 7, ig = lane >> 3;
#pragma unroll
    for (int jj = 0; jj < 4; jj++) r.bs[jj] = base[g + 8 * (4 * ig + jj)];
#pragma unroll
    for (int qg = 0; qg < 4; qg++) {
        uint32_t lmask, cmask;
        v_masks(qg, o0, len, lmask, cmask);
#pragma unroll
        for (int m = 0; m < 4; m++) {
            const int t = 2 * qg + (m & 1) + 8 * (m >> 1);
            const bool act = t >= o0 && t < o0 + len;
            const uint16_t* row = value + (int64_t)(t - o0) * stride + g + 8 * (4 * ig);  // (read only where act)
#pragma unroll
            for (int jj = 0; jj < 4; jj++) r.vv[qg][m][jj] = act ? __ldg(row + 8 * jj) : (uint16_t)0;
        }
        r.lw[qg] = make_uint4(0, 0, 0, 0);
        r.cw[qg] = make_uint2(0, 0);
        if (lmask && lmask != 0xFFFFFFFFu) r.lw[qg] = *reinterpret_cast<const uint4*>(tile + ig * 512 + (4 * g + qg) * 16);
        if (cmask && cmask != 0xFFFFFFFFu) r.cw[qg] = *reinterpret_cast<const uint2*>(tile + 2048 + (ig >> 1) * 512 + (4 * g + qg) * 16 + 2 * (ig & 1) * 4);
    }
}
__device__ __forceinline__ void v_write(const VPre& r, uint8_t* tile, int o0, int len, int lane, int head, Appender& ap) {
    const int g = lane & 7, ig = lane >> 3;
#pragma unroll
    for (int qg = 0; qg < 4; qg++) {
        uint32_t lmask, cmask;
        v_masks(qg, o0, len, lmask, cmask);
        if (!lmask) continue;  // (qg is the loop's: every lane skips alike)
        const int owner = 4 * g + qg;
        uint32_t lbits[4] = {0, 0, 0, 0}, cb[2] = {0, 0}, ent[4][4];
        bool esc[4][4], any = false;
#pragma unroll
        for (int m = 0; m < 4; m++) {
            const int t = 2 * qg + (m & 1) + 8 * (m >> 1);
            const bool act = t >= o0 && t < o0 + len;
#pragma unroll
            for (int jj = 0; jj < 4; jj++) {
                const uint32_t v = r.vv[qg][m][jj];
                uint32_t x;
                const uint32_t code = code_of(v, r.bs[jj], esc[m][jj], x);
                esc[m][jj] &= act;
                any |= esc[m][jj];
                ent[m][jj] = esc[m][jj] ? entry(owner, 4 * ig + jj, m, x, t, 1, head) : 0u;
                if (act) {
                    lbits[jj] |= (v & 255u) << (8 * m);
                    if ((jj & 1) == 0) cb[jj >> 1] |= (((code >> 3) << 7) | (code & 7u)) << (8 * m);
                    else cb[jj >> 1] |= (((code >> 3) << 3) << (8 * m)) | ((code & 7u) << (8 * ((m + 3) & 3) + 4));
                }
            }
        }
        const uint32_t olw[4] = {r.lw[qg].x, r.lw[qg].y, r.lw[qg].z, r.lw[qg].w}, ocw[2] = {r.cw[qg].x, r.cw[qg].y};
        uint32_t nlw[4], ncw[2];
#pragma unroll
        for (int jj = 0; jj < 4; jj++) nlw[jj] = ((lmask == 0xFFFFFFFFu ? 0u : olw[jj]) & ~lmask) | lbits[jj];
#pragma unroll
        for (int e = 0; e < 2; e++) ncw[e] = ((cmask == 0xFFFFFFFFu ? 0u : ocw[e]) & ~cmask) | cb[e];
        *reinterpret_cast<uint4*>(tile + ig * 512 + owner * 16) = make_uint4(nlw[0], nlw[1], nlw[2], nlw[3]);
        *reinterpret_cast<uint2*>(tile + 2048 + (ig >> 1) * 512 + owner * 16 + 2 * (ig & 1) * 4) = make_uint2(ncw[0], ncw[1]);
        if (__any_sync(FULLM, any)) {
#pragma unroll
            for (int m = 0; m < 4; m++) {
#pragma unroll
                for (int jj = 0; jj < 4; jj++) ap.add(ent[m][jj], esc[m][jj], lane);
            }
        }
    }
}


// ---- raw pages -----------------------------------------------------------------------------------------------------------------------------
// A head-page's K or V tile written raw. CONV: the page was not raw (old tokens decoded, the run's from the inputs); else the run's tokens only.
template <int KIND, bool CONV>
__device__ __forceinline__ void raw_tile(const WriteParams& P, uint8_t* tile, uint8_t* ptile, const uint4 hd, const uint32_t* pg32, int head, int lane, int o0, int len, int i0, int used0) {
    const int g = lane >> 2, q = lane & 3;
    uint4 f[8];
    if (CONV) {
        Raw r;
        raw_load_plain(r, tile, lane);
        uint32_t hb[KIND == 0 ? 8 : 16];
        load_bases<KIND>((KIND == 0 ? P.kbase : P.vbase) + head * D, lane, hb);
        decode_tile<KIND>(r, hb, hd, pg32, P.esc.ent, P.esc.nc, P.A, P.raw, head, lane, f, used0);
    }
    const int64_t stride = KIND == 0 ? P.key_stride : P.value_stride;
    const uint16_t* in = reinterpret_cast<const uint16_t*>(KIND == 0 ? P.key : P.value) + (int64_t)i0 * stride + head * D;
    uint32_t cn[8], cm[8], pn[8], pm[8], ln[16], lm[16];
#pragma unroll
    for (int s = 0; s < 8; s++) cn[s] = cm[s] = pn[s] = pm[s] = 0u;
#pragma unroll
    for (int i = 0; i < 16; i++) ln[i] = lm[i] = 0u;
#pragma unroll
    for (int i = 0; i < 16; i++) {
#pragma unroll
        for (int m = 0; m < 4; m++) {
            int tok, ch;
            coord<KIND>(g, q, i, m, tok, ch);
            const bool run = tok >= o0 && tok < o0 + len;
            if (!CONV && !run) continue;
            uint32_t v;
            if (run) {
                v = in[(int64_t)(tok - o0) * stride + ch];
            } else {
                const uint4& c = f[i >> 1];
                const uint32_t w0 = (i & 1) ? c.y : c.x, w1 = (i & 1) ? c.w : c.z;
                const uint32_t w = m < 2 ? w0 : w1;
                v = (m & 1) ? (w >> 16) : (w & 0xFFFFu);
            }
            const uint32_t hi = v >> 8, sign = hi >> 7, e7 = hi & 127u, off = e7 >> 4, pl = e7 & 15u;
            const int s = i >> 1;
            if ((i & 1) == 0) {
                cn[s] |= ((sign << 7) | off) << (8 * m);
                cm[s] |= 0x87u << (8 * m);
                pn[s] |= pl << (8 * m);
                pm[s] |= 0x0Fu << (8 * m);
            } else {
                cn[s] |= ((sign << 3) << (8 * m)) | (off << (8 * ((m + 3) & 3) + 4));
                cm[s] |= (0x08u << (8 * m)) | (0x70u << (8 * ((m + 3) & 3)));
                pn[s] |= pl << (8 * m + 4);
                pm[s] |= 0xF0u << (8 * m);
            }
            ln[i] |= (v & 0xFFu) << (8 * m);
            lm[i] |= 0xFFu << (8 * m);
        }
    }
#pragma unroll
    for (int s = 0; s < 8; s++) {
        uint32_t* cp = reinterpret_cast<uint32_t*>(tile + 2048 + (s >> 2) * 512 + lane * 16 + (s & 3) * 4);
        uint32_t* pp = reinterpret_cast<uint32_t*>(ptile + (s >> 2) * 512 + lane * 16 + (s & 3) * 4);
        if (cm[s]) *cp = (cm[s] == 0xFFFFFFFFu ? 0u : (*cp & ~cm[s])) | cn[s];
        if (pm[s]) *pp = (pm[s] == 0xFFFFFFFFu ? 0u : (*pp & ~pm[s])) | pn[s];
    }
    if (!CONV) {
#pragma unroll
        for (int i = 0; i < 16; i++) {
            uint32_t* lp = reinterpret_cast<uint32_t*>(tile + (i >> 2) * 512 + lane * 16 + (i & 3) * 4);
            if (lm[i]) *lp = (lm[i] == 0xFFFFFFFFu ? 0u : (*lp & ~lm[i])) | ln[i];
        }
    }
}

// A page that found no slab (or whose slab is not its own) is void; its block goes in the status ring (once while it stays lost).
__device__ __forceinline__ void fail_page(const WriteParams& P, uint32_t* pg32, uint32_t block) {
    pg32[0] = PG_VOID;
    atomicAdd(P.status + ST_VOID, 1u);
    const uint64_t b = 2ull * block;
    if (P.lost && b + 1 < (uint64_t)P.nlost && (atomicOr(P.lost + (b >> 5), 1u << (b & 31)) & (1u << (b & 31)))) return;
    const uint32_t idx = atomicAdd(P.status + ST_RING_HEAD, 1u);
    P.status[ST_RING + idx % P.ring] = block + 1u;
}

// A block that has a raw page goes in the status's raw ring, once while it keeps one (the host keeps such blocks out of the prefix cache, so that they are reused first).
__device__ __forceinline__ void note_raw(const WriteParams& P, uint32_t block) {
    const uint64_t b = 2ull * block + 1;
    if (P.lost && b < (uint64_t)P.nlost && (atomicOr(P.lost + (b >> 5), 1u << (b & 31)) & (1u << (b & 31)))) return;
    const uint32_t idx = atomicAdd(P.status + ST_RAW_RING_HEAD, 1u);
    P.status[ST_RING + P.ring + idx % P.ring] = block + 1u;
}

// The CTA's page made raw (every thread calls it): void where no slab is free.
__device__ __noinline__ void make_raw(const WriteParams& P, uint8_t* pg, uint32_t* pg32, int hb, uint32_t tag, int o0, int len, int i0, uint32_t block, uint32_t used0, const uint32_t* cnt0) {
    __shared__ int s_slab;
    if (threadIdx.x == 0) s_slab = P.raw.ns ? raw_alloc(P.raw, tag, tag * 2654435761u, P.status) : -1;
    __syncthreads();
    const int slab = s_slab;
    if (slab < 0) {
        if (threadIdx.x == 0) fail_page(P, pg32, block);
        __syncthreads();
        return;
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    uint8_t* plane = P.raw.data + (size_t)slab * (size_t)P.raw.plane;
    for (int pr = warp; pr < 2 * P.H; pr += (int)(blockDim.x >> 5)) {  // pr = 2 head + kind
        const int h = pr >> 1;
        if (P.one && (pr & 1)) continue;  // (one stream: no V tile)
        uint8_t* tile = pg + hb + (int64_t)h * head_page(P.one) + (pr & 1) * TILE;
        uint4 hd = *reinterpret_cast<const uint4*>(pg + 16 * h);
        hd.x = cnt0[h];  // the entries before the run
        if (pr & 1) raw_tile<1, true>(P, tile, plane + (size_t)pr * 1024, hd, pg32, h, lane, o0, len, i0, (int)used0);
        else raw_tile<0, true>(P, tile, plane + (size_t)pr * 1024, hd, pg32, h, lane, o0, len, i0, (int)used0);
    }
    __syncthreads();
    for (int j = threadIdx.x; j < ND; j += blockDim.x) {
        const uint32_t d = pg32[1 + j];
        if (d) {
            if (d - 1 < (uint32_t)P.esc.nc && P.esc.owner[d - 1] == tag) esc_free(P.esc, (int)(d - 1), P.status);
            else atomicAdd(P.status + ST_STALE, 1u);
        }
        pg32[1 + j] = 0;
    }
    for (int h = threadIdx.x; h < P.H; h += blockDim.x) *reinterpret_cast<uint4*>(pg + 16 * h) = make_uint4(HEAD_RAW, 0u, 0u, 0u);
    __syncthreads();
    if (threadIdx.x == 0) {
        pg32[1] = (uint32_t)slab + 1u;
        pg32[0] = PG_RAW;
        atomicAdd(P.status + ST_RAW_PAGES, 1u);
        note_raw(P, block);
    }
    __syncthreads();
}

// A run into a raw page (every thread calls it); void if its slab is not its own.
__device__ __noinline__ void raw_run(const WriteParams& P, uint8_t* pg, uint32_t* pg32, int hb, uint32_t tag, int o0, int len, int i0, uint32_t block) {
    const uint32_t d = pg32[1];
    if (d - 1 >= (uint32_t)P.raw.ns || P.raw.owner[d - 1] != tag) {
        if (threadIdx.x == 0) fail_page(P, pg32, block);
        return;
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    uint8_t* plane = P.raw.data + (size_t)(d - 1) * (size_t)P.raw.plane;
    for (int pr = warp; pr < 2 * P.H; pr += (int)(blockDim.x >> 5)) {
        const int h = pr >> 1;
        if (P.one && (pr & 1)) continue;  // (one stream: no V tile)
        uint8_t* tile = pg + hb + (int64_t)h * head_page(P.one) + (pr & 1) * TILE;
        const uint4 hd = make_uint4(HEAD_RAW, 0u, 0u, 0u);
        if (pr & 1) raw_tile<1, false>(P, tile, plane + (size_t)pr * 1024, hd, pg32, h, lane, o0, len, i0, 0);
        else raw_tile<0, false>(P, tile, plane + (size_t)pr * 1024, hd, pg32, h, lane, o0, len, i0, 0);
    }
}

// A block's page cleared as a page's first token clears it, for a block no request holds (a warp a block): its chunks or its slab (by owner tag) back to their pools, its headers, its
// directory and its count zero, and the block's bits in the lost bitmap.
__global__ void __launch_bounds__(32) release_kernel(uint8_t* pool, int64_t page_stride, int64_t nblocks, const int32_t* blocks, int H, EscapePool esc, RawPool raw, uint32_t layer_tag, uint32_t* status,
                                                      uint32_t* lost, int64_t nlost) {
    const int64_t block = blocks[blockIdx.x];
    if (block < 0 || block >= nblocks) return;
    uint8_t* pg = pool + block * page_stride;
    uint32_t* pg32 = reinterpret_cast<uint32_t*>(pg + 16 * H);
    const uint32_t tag = layer_tag | (uint32_t)(block & 0xFFFFFF);
    const bool wasraw = (*(volatile uint32_t*)pg32 & PG_RAW) != 0;
    __syncwarp();
    for (int j = threadIdx.x; j < ND; j += 32) {
        const uint32_t d = pg32[1 + j];
        if (d) {  // only what the pools list as this page's
            if (wasraw) {
                if (j == 0 && d - 1 < (uint32_t)raw.ns && raw.owner[d - 1] == tag) raw_free(raw, (int)(d - 1), status);
                else atomicAdd(status + ST_STALE, 1u);
            } else if (d - 1 < (uint32_t)esc.nc && esc.owner[d - 1] == tag) {
                esc_free(esc, (int)(d - 1), status);
            } else {
                atomicAdd(status + ST_STALE, 1u);
            }
        }
        pg32[1 + j] = 0;
    }
    for (int h = threadIdx.x; h < H; h += 32) *reinterpret_cast<uint4*>(pg + 16 * h) = make_uint4(0, 0, 0, 0);
    __syncwarp();
    if (threadIdx.x == 0) {
        pg32[0] = 0;
        if (lost && 2ull * block + 1 < (uint64_t)nlost) atomicAnd(lost + ((2ull * block) >> 5), ~(3u << ((2ull * block) & 31)));
    }
}

// A page's first token clears its headers and its arena and gives its chunks back to the pool before any entry is appended (every thread of the CTA calls it, and it ends in a barrier: a warp
// appends to every head's).
__device__ __forceinline__ void first_token_clear(const WriteParams& P, uint8_t* pg, uint32_t* pg32, uint32_t pg0, uint32_t tag, uint32_t block) {
    const bool wasraw = (pg0 & PG_RAW) != 0;
    for (int j = threadIdx.x; j < ND; j += blockDim.x) {
        const uint32_t d = pg32[1 + j];
        if (d) {  // only what the pool lists as this page's
            if (wasraw) {
                if (j == 0 && d - 1 < (uint32_t)P.raw.ns && P.raw.owner[d - 1] == tag) raw_free(P.raw, (int)(d - 1), P.status);
                else atomicAdd(P.status + ST_STALE, 1u);
            } else if (d - 1 < (uint32_t)P.esc.nc && P.esc.owner[d - 1] == tag) {
                esc_free(P.esc, (int)(d - 1), P.status);
            } else {
                atomicAdd(P.status + ST_STALE, 1u);
            }
            pg32[1 + j] = 0;
        }
    }
    if (threadIdx.x == 0) {
        pg32[0] = 0;
        if (P.lost && (P.layer_tag >> 24) == 1u && 2ull * block + 1 < (uint64_t)P.nlost) atomicAnd(P.lost + ((2ull * block) >> 5), ~(3u << ((2ull * block) & 31)));  // a new life of the block
    }
    for (int h = threadIdx.x; h < P.H; h += blockDim.x) *reinterpret_cast<uint4*>(pg + 16 * h) = make_uint4(0, 0, 0, 0);
    __syncthreads();
}

// A later run kills the entries of the tokens it writes over (a head's own, in its header, the arena and the chunks; a warp calls it for its head, where the head holds entries: cnt != 0).
__device__ __forceinline__ void kill_old(const WriteParams& P, uint32_t* pg32, uint32_t* hdr, const uint4 hv, uint32_t cnt, uint32_t used, int h, int o0, int len, int lane) {
    if (lane == 0) {
        uint32_t e[3] = {hv.y, hv.z, hv.w};
        bool changed = false;
        for (int j = 0; j < 3 && j < (int)cnt; j++) {
            if ((e[j] & LIVE) && e_tok(e[j]) >= o0 && e_tok(e[j]) < o0 + len) {
                e[j] &= ~LIVE;
                changed = true;
            }
        }
        if (changed) *reinterpret_cast<uint4*>(hdr) = make_uint4(cnt, e[0], e[1], e[2]);
    }
    if (cnt > 3) {
        for (uint32_t base = 0; base < used; base += 128) {
            uint32_t* ep[4];
            uint32_t e[4];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const uint32_t idx = base + lane + 32 * j;
                ep[j] = idx < used ? const_cast<uint32_t*>(entry_at(pg32, P.esc.ent, P.esc.nc, P.A, idx)) : nullptr;
                e[j] = ep[j] ? *ep[j] : 0u;
            }
#pragma unroll
            for (int j = 0; j < 4; j++)
                if (ep[j] && (e[j] & LIVE) && e_head(e[j]) == h && e_tok(e[j]) >= o0 && e_tok(e[j]) < o0 + len) *ep[j] = e[j] & ~LIVE;
        }
    }
}

// A run of any length, a CTA (the run's first token's), found from the slots alone: the others return at once. A warp a head (8 at a time). The kernel is a short
// chain of reads of memory no cache holds (the slots, then the page's header, counter and tile words, then the page's other entries (arena, then chunks) of a head that has
// many), so each is made one for all that can be: the slots of the run in one read (a lane a slot), and everything a head's warp needs of its
// page in one go before it kills, computes or writes.
__device__ __noinline__ void write_run(const WriteParams& P) {
    const int i = blockIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t mine = P.slots[i];
    if (mine < 0) return;
    uint8_t* pg = P.pool + (mine >> 4) * P.page_stride;
    uint32_t* pg32 = reinterpret_cast<uint32_t*>(pg + 16 * P.H);
    const uint32_t pg0 = *(volatile uint32_t*)pg32;  // flags and count before this run
    if (threadIdx.x == 0) g_need = 0;
    // lanes 0 .. 15 the slots i .. i + 15 (a run is at most a page), lane 16 the one before
    const int64_t at = lane < 16 ? (i + lane < P.N ? P.slots[i + lane] : -1) : lane == 16 && i > 0 ? P.slots[i - 1] : -1;
    const int64_t before = __shfl_sync(FULLM, at, 16);
    if (i > 0 && before >= 0 && before + 1 == mine && (mine & 15) != 0) return;
    const int o0 = (int)(mine & 15);
    const uint32_t follows = __ballot_sync(FULLM, lane < 16 && at == mine + lane) & 0xFFFFu;  // lane j: slot i + j is slot + j
    const int len = min(__ffs(~follows) - 1, 16 - o0);  // (the run's tokens: the first slot that does not follow)
    const uint32_t tag = P.layer_tag | (uint32_t)((mine >> 4) & 0xFFFFFF);
    const uint32_t block = (uint32_t)(mine >> 4);
    const int hb = hdr_bytes(P.H, P.A);
    __shared__ uint32_t s_used0, s_cnt0[32];  // entries before this run: the page's, each head-page's
    if (o0 == 0) {
        first_token_clear(P, pg, pg32, pg0, tag, block);
    } else {
        if (pg0 & PG_VOID) return;
        if (pg0 & PG_RAW) {
            raw_run(P, pg, pg32, hb, tag, o0, len, i, block);
            return;
        }
    }
    // 8 heads a round, a warp a head: its kills, then (a barrier: an entry a kill reads is not one another warp is still appending) its writes
    for (int h0 = 0; h0 < P.H; h0 += 8) {
        const int h = h0 + warp;
        const bool have = h < P.H;
        uint32_t* hdr = reinterpret_cast<uint32_t*>(pg + 16 * (have ? h : 0));
        uint8_t* tile = pg + hb + (int64_t)(have ? h : 0) * head_page(P.one);
        uint32_t cnt = 0;
        KPre kp;
        VPre vp;
        if (have) {
            // all of it at once: the header (count, three entries), the page's entry counter, the lane's values, bases and words of the K and V tiles
            const uint4 hv = o0 == 0 ? make_uint4(0, 0, 0, 0) : *reinterpret_cast<const uint4*>(hdr);
            const uint32_t used = o0 == 0 ? 0u : min(pg0 & PG_COUNT, (uint32_t)(P.A + ND * CH));
            k_load(kp, P.key + (int64_t)i * P.key_stride + h * D, P.key_stride, P.kbase + h * D, tile, o0, len, lane);
            if (!P.one) v_load(vp, P.value + (int64_t)i * P.value_stride + h * D, P.value_stride, P.vbase + h * D, tile + TILE, o0, len, lane);
            cnt = hv.x;
            if (lane == 0) s_cnt0[h] = cnt;
            if (h0 == 0 && threadIdx.x == 0) s_used0 = used;
            if (cnt != 0) kill_old(P, pg32, hdr, hv, cnt, used, h, o0, len, lane);
        }
        __syncthreads();
        if (have) {
            Appender ap{cnt, hdr, pg32, tag, &P};
            k_write(kp, tile, o0, len, lane, h, ap);
            if (!P.one) v_write(vp, tile + TILE, o0, len, lane, h, ap);
            __syncwarp();
            if (lane == 0) hdr[0] = ap.cnt;
        }
        if (h0 + 8 < P.H) __syncthreads();  // (the next round's kills read the entries this one appended)
    }
    __syncthreads();
    if (*(volatile uint32_t*)&g_need) make_raw(P, pg, pg32, hb, tag, o0, len, i, block, s_used0, s_cnt0);
}

// ---- a decode step's token ---------------------------------------------------------------------------------------------------------
// A token that is the only one of its run (a decode step writes one token a sequence: the slot before it is another page's and the one after it is not the next) goes through this path, which
// does for one token what write_run's loops do for 16 of which one is the run's: what does not wait for its slot (its keys and values, the bases) is read as the slot is, what does (the
// page's counter, its head's header, the words of the tiles it writes part of) together after it, and its codes, low bytes and entries are computed and written. The same bytes as write_run's.
struct KOne {
    uint32_t kb;      // the bases of the lane's 4 channels, a byte each
    uint32_t p0, p1;  // the lane's keys: 2 channels of a word, 16 s + 2 q + {0, 1} and + {8, 9}
    uint32_t ow;      // the code word of the pair of tokens (8 nt + g, g the token's) it writes a half of
};
__device__ __forceinline__ void k_one_in(KOne& r, const uint16_t* __restrict__ key, const uint8_t* __restrict__ base, int lane) {
    const int s = lane >> 2, q = lane & 3;
    r.kb = (uint32_t)base[16 * s + 2 * q] | ((uint32_t)base[16 * s + 2 * q + 1] << 8) | ((uint32_t)base[16 * s + 2 * q + 8] << 16) | ((uint32_t)base[16 * s + 2 * q + 9] << 24);
    r.p0 = __ldg(reinterpret_cast<const uint32_t*>(key + 16 * s + 2 * q));
    r.p1 = __ldg(reinterpret_cast<const uint32_t*>(key + 16 * s + 2 * q + 8));
}
__device__ __forceinline__ void k_one_tile(KOne& r, const uint8_t* tile, int t, int lane) {
    const int s = lane >> 2, q = lane & 3;
    r.ow = *reinterpret_cast<const uint32_t*>(tile + 2048 + (s >> 2) * 512 + (4 * (t & 7) + q) * 16 + (s & 3) * 4);
}
struct KCalc {
    uint32_t lw, cb, ent[4];  // the 4 low bytes (a word), the code bits of the token in the pair's code word, the entries
    bool esc[4], any;         // (any: some lane of the warp has one)
};
__device__ __forceinline__ void k_one_calc(KCalc& c, const KOne& r, int t, int lane, int head) {  // (what needs no word of the page: the token's codes, low bytes and entries)
    const int s = lane >> 2, q = lane & 3, nt = t >> 3, g = t & 7;
    const int owner = 4 * g + q, i = 2 * s + nt;
    const uint32_t v[4] = {r.p0 & 0xFFFFu, r.p0 >> 16, r.p1 & 0xFFFFu, r.p1 >> 16};
    c.lw = 0;
    c.cb = 0;
    bool any = false;
#pragma unroll
    for (int m = 0; m < 4; m++) {
        const uint32_t b = (r.kb >> (8 * m)) & 0xFFu;
        uint32_t x;
        const uint32_t code = code_of(v[m], b, c.esc[m], x);
        any |= c.esc[m];
        c.ent[m] = c.esc[m] ? entry(owner, i, m, x, t, 0, head) : 0u;
        c.lw |= (v[m] & 255u) << (8 * m);
        if (nt == 0) c.cb |= (((code >> 3) << 7) | (code & 7u)) << (8 * m);
        else c.cb |= (((code >> 3) << 3) << (8 * m)) | ((code & 7u) << (8 * ((m + 3) & 3) + 4));
    }
    c.any = any;
}
__device__ __forceinline__ void k_one_store(const KCalc& c, const KOne& r, uint8_t* tile, int t, int lane, Appender& ap) {
    const int s = lane >> 2, q = lane & 3, nt = t >> 3, g = t & 7;
    const int owner = 4 * g + q, i = 2 * s + nt;
    *reinterpret_cast<uint32_t*>(tile + (i >> 2) * 512 + owner * 16 + (i & 3) * 4) = c.lw;
    *reinterpret_cast<uint32_t*>(tile + 2048 + (s >> 2) * 512 + owner * 16 + (s & 3) * 4) = (r.ow & ~(nt ? 0x78787878u : 0x87878787u)) | c.cb;
    if (__any_sync(FULLM, c.any)) {
#pragma unroll
        for (int m = 0; m < 4; m++) ap.add(c.ent[m], c.esc[m], lane);
    }
}

struct VOne {
    uint32_t bs[4];  // the bases of the lane's 4 channels g + 8 (4 ig + jj)
    uint32_t vv[4];  // its values
    uint4 lw;        // the 4 low words (4 ig .. 4 ig + 3) it writes a byte of, of owner 4 g + (the token's group)
    uint2 cw;        // the 2 code words (2 ig, 2 ig + 1)
};
__device__ __forceinline__ void v_one_in(VOne& r, const uint16_t* __restrict__ value, const uint8_t* __restrict__ base, int lane) {
    const int g = lane & 7, ig = lane >> 3;
#pragma unroll
    for (int jj = 0; jj < 4; jj++) {
        r.bs[jj] = base[g + 8 * (4 * ig + jj)];
        r.vv[jj] = __ldg(value + g + 8 * (4 * ig + jj));
    }
}
__device__ __forceinline__ void v_one_tile(VOne& r, const uint8_t* tile, int t, int lane) {  // (tile: the V tile)
    const int g = lane & 7, ig = lane >> 3, owner = 4 * g + ((t & 7) >> 1);
    r.lw = *reinterpret_cast<const uint4*>(tile + ig * 512 + owner * 16);
    r.cw = *reinterpret_cast<const uint2*>(tile + 2048 + (ig >> 1) * 512 + owner * 16 + 2 * (ig & 1) * 4);
}
struct VCalc {
    uint32_t lbits[4], cb[2], ent[4];  // the token's byte of each of the 4 low words, its bits of the 2 code words, the entries
    bool esc[4], any;
};
__device__ __forceinline__ void v_one_calc(VCalc& c, const VOne& r, int t, int lane, int head) {
    const int g = lane & 7, ig = lane >> 3;
    const int owner = 4 * g + ((t & 7) >> 1), m = (t & 1) | ((t >> 3) << 1);  // (token t is slot m of its group of 4: 2 qg + (m & 1) + 8 (m >> 1))
    c.cb[0] = c.cb[1] = 0;
    bool any = false;
#pragma unroll
    for (int jj = 0; jj < 4; jj++) {
        uint32_t x;
        const uint32_t code = code_of(r.vv[jj], r.bs[jj], c.esc[jj], x);
        any |= c.esc[jj];
        c.ent[jj] = c.esc[jj] ? entry(owner, 4 * ig + jj, m, x, t, 1, head) : 0u;
        c.lbits[jj] = (r.vv[jj] & 255u) << (8 * m);
        if ((jj & 1) == 0) c.cb[jj >> 1] |= (((code >> 3) << 7) | (code & 7u)) << (8 * m);
        else c.cb[jj >> 1] |= (((code >> 3) << 3) << (8 * m)) | ((code & 7u) << (8 * ((m + 3) & 3) + 4));
    }
    c.any = any;
}
__device__ __forceinline__ void v_one_store(const VCalc& c, const VOne& r, uint8_t* tile, int t, int lane, Appender& ap) {
    const int g = lane & 7, ig = lane >> 3;
    const int owner = 4 * g + ((t & 7) >> 1), m = (t & 1) | ((t >> 3) << 1);
    const uint32_t lmask = 0xFFu << (8 * m), cmask = (0x87u << (8 * m)) | (0x08u << (8 * m)) | (0x70u << (8 * ((m + 3) & 3)));
    *reinterpret_cast<uint4*>(tile + ig * 512 + owner * 16) = make_uint4((r.lw.x & ~lmask) | c.lbits[0], (r.lw.y & ~lmask) | c.lbits[1], (r.lw.z & ~lmask) | c.lbits[2], (r.lw.w & ~lmask) | c.lbits[3]);
    *reinterpret_cast<uint2*>(tile + 2048 + (ig >> 1) * 512 + owner * 16 + 2 * (ig & 1) * 4) = make_uint2((r.cw.x & ~cmask) | c.cb[0], (r.cw.y & ~cmask) | c.cb[1]);
    if (__any_sync(FULLM, c.any)) {
#pragma unroll
        for (int jj = 0; jj < 4; jj++) ap.add(c.ent[jj], c.esc[jj], lane);
    }
}

// The write kernel: a CTA a token. A token that is the only one of its run goes through the decode step's path (the rest of this kernel); a longer run's first token runs write_run; the
// run's other tokens return at once. (P.lean 0: write_run for every run.)
__global__ void __launch_bounds__(256) write_kernel(const __grid_constant__ WriteParams P) {
    const int i = blockIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t mine = P.slots[i];
    const int64_t prv = i > 0 ? P.slots[i - 1] : -1, nxt = i + 1 < P.N ? P.slots[i + 1] : -1;
    KOne ko;
    VOne vo;
    KCalc kc;
    VCalc vc;
    if (warp < P.H) {  // (the first round's head: its inputs do not wait for the slot)
        k_one_in(ko, P.key + (int64_t)i * P.key_stride + warp * D, P.kbase + warp * D, lane);
        if (!P.one) v_one_in(vo, P.value + (int64_t)i * P.value_stride + warp * D, P.vbase + warp * D, lane);
    }
    if (mine < 0) return;
    if (i > 0 && prv >= 0 && prv + 1 == mine && (mine & 15) != 0) return;  // (not the first token of its run)
    if (!P.lean || (nxt == mine + 1 && (mine & 15) != 15)) {  // (a run of more than one token)
        write_run(P);
        return;
    }
    const int o0 = (int)(mine & 15);
    const uint32_t block = (uint32_t)(mine >> 4), tag = P.layer_tag | (block & 0xFFFFFFu);
    uint8_t* pg = P.pool + (int64_t)block * P.page_stride;
    uint32_t* pg32 = reinterpret_cast<uint32_t*>(pg + 16 * P.H);
    const uint32_t pg0 = *(volatile uint32_t*)pg32;  // flags and count before this token
    if (threadIdx.x == 0) g_need = 0;
    const int hb = hdr_bytes(P.H, P.A);
    __shared__ uint32_t s_used0, s_cnt0[32];  // entries before this token: the page's, each head-page's
    for (int h0 = 0; h0 < P.H; h0 += 8) {
        const int h = h0 + warp;
        const bool have = h < P.H;
        uint32_t* hdr = reinterpret_cast<uint32_t*>(pg + 16 * (have ? h : 0));
        uint8_t* tile = pg + hb + (int64_t)(have ? h : 0) * head_page(P.one);
        uint32_t cnt = 0;
        uint4 hv = make_uint4(0, 0, 0, 0);
        const uint32_t used = o0 == 0 ? 0u : min(pg0 & PG_COUNT, (uint32_t)(P.A + ND * CH));
        if (have) {
            if (h0 > 0) {
                k_one_in(ko, P.key + (int64_t)i * P.key_stride + h * D, P.kbase + h * D, lane);
                if (!P.one) v_one_in(vo, P.value + (int64_t)i * P.value_stride + h * D, P.vbase + h * D, lane);
            }
            if (o0 != 0) hv = *reinterpret_cast<const uint4*>(hdr);
            k_one_tile(ko, tile, o0, lane);
            if (!P.one) v_one_tile(vo, tile + TILE, o0, lane);
            k_one_calc(kc, ko, o0, lane, h);  // (while the page's words are on their way)
            if (!P.one) v_one_calc(vc, vo, o0, lane, h);
        }
        if (h0 == 0) {  // (the page: a first token clears it, any other finds it void or raw, or goes on)
            if (o0 == 0) {
                first_token_clear(P, pg, pg32, pg0, tag, block);
            } else if (pg0 & (PG_VOID | PG_RAW)) {
                if (pg0 & PG_VOID) return;
                raw_run(P, pg, pg32, hb, tag, o0, 1, i, block);
                return;
            }
        }
        if (have) {
            cnt = hv.x;
            if (lane == 0) s_cnt0[h] = cnt;
            if (h0 == 0 && threadIdx.x == 0) s_used0 = used;
            if (cnt != 0) kill_old(P, pg32, hdr, hv, cnt, used, h, o0, 1, lane);
        }
        __syncthreads();
        if (have) {
            Appender ap{cnt, hdr, pg32, tag, &P};
            k_one_store(kc, ko, tile, o0, lane, ap);
            if (!P.one) v_one_store(vc, vo, tile + TILE, o0, lane, ap);
            __syncwarp();
            if (lane == 0 && ap.cnt != cnt) hdr[0] = ap.cnt;
        }
        if (h0 + 8 < P.H) __syncthreads();
    }
    __syncthreads();
    if (*(volatile uint32_t*)&g_need) make_raw(P, pg, pg32, hb, tag, o0, 1, i, block, s_used0, s_cnt0);
}

// ---- reading pages back to bf16: out [page][token][head][channel] for the pages listed (one warp a head-page) -----------------
__global__ void __launch_bounds__(128) unpack_kernel(const uint8_t* __restrict__ pool, int64_t page_stride, const int32_t* __restrict__ pages, int n, const uint8_t* __restrict__ kbase,
                                                     const uint8_t* __restrict__ vbase, int H, int A, const uint32_t* __restrict__ ent, int nc, const RawPool R, uint16_t* __restrict__ out_k, uint16_t* __restrict__ out_v, int one) {
    const int wl = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (wl >= n * H) return;
    const int lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
    const int j = wl / H, h = wl % H;
    const uint8_t* pg = pool + (int64_t)pages[j] * page_stride;
    const uint4 hd = *reinterpret_cast<const uint4*>(pg + 16 * h);
    const uint32_t* pg32 = reinterpret_cast<const uint32_t*>(pg + 16 * H);
    const uint8_t* tile = pg + hdr_bytes(H, A) + (int64_t)h * head_page(one != 0);
#pragma unroll
    for (int kind = 0; kind < 2; kind++) {
        if (kind == 1 && one) break;  // (one stream: the keys' layout alone)
        uint4 f[8];
        Raw r;
        raw_load(r, tile + kind * TILE, lane);
        if (kind == 0) {
            uint32_t hb[8];
            load_bases<0>(kbase + h * D, lane, hb);
            decode_tile<0>(r, hb, hd, pg32, ent, nc, A, R, h, lane, f);
        } else {
            uint32_t hb[16];
            load_bases<1>(vbase + h * D, lane, hb);
            decode_tile<1>(r, hb, hd, pg32, ent, nc, A, R, h, lane, f);
        }
        uint16_t* out = (kind == 0 ? out_k : out_v) + (int64_t)j * PT * H * D;
#pragma unroll
        for (int i = 0; i < 16; i++) {
            const uint4& c = f[i >> 1];
            const uint32_t w0 = (i & 1) ? c.y : c.x, w1 = (i & 1) ? c.w : c.z;  // group i's two pairs
#pragma unroll
            for (int m = 0; m < 4; m++) {
                int tok, ch;
                if (kind == 0) coord<0>(g, q, i, m, tok, ch);
                else coord<1>(g, q, i, m, tok, ch);
                const uint32_t w = m < 2 ? w0 : w1;
                out[((int64_t)tok * H + h) * D + ch] = (uint16_t)((m & 1) ? (w >> 16) : (w & 0xFFFFu));
            }
        }
    }
}

// ---- decoding pages into a paged bf16 cache, as vLLM's FlashAttention reads one ----------------------------------------------------
// The pages listed (pages[j]) decoded into the block dst[j] of `out` [blocks][16 tokens][H][2 D] bf16: token t of head h of block b has its keys at ((b 16 + t) H + h) 2 D and its
// values D after (the cache the prefill rows' FlashAttention runs on: a context's pages made bf16 before the step's own tokens are put after it). One warp a head-page, as unpack.
// HALVES 2: a head of 256 channels, held as the page's heads 2 h and 2 h + 1 (its channels 0-127 and 128-255): the head-page 2 h + f's keys at channels 128 f of head h's 256, its values 256 after.
template <int HALVES>
__global__ void __launch_bounds__(128) gather_kernel(const uint8_t* __restrict__ pool, int64_t page_stride, const int32_t* __restrict__ pages, const int32_t* __restrict__ dst, int n,
                                                     const uint8_t* __restrict__ kbase, const uint8_t* __restrict__ vbase, int H, int A, const uint32_t* __restrict__ ent, int nc, const RawPool R,
                                                     uint16_t* __restrict__ out) {
    const int wl = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (wl >= n * H) return;
    const int lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
    const int j = wl / H, h = wl % H;
    if (pages[j] < 0) return;  // (a place of the list no page is in: the plan's lists are as long as the most the rows can need)
    const uint8_t* pg = pool + (int64_t)pages[j] * page_stride;
    const uint4 hd = *reinterpret_cast<const uint4*>(pg + 16 * h);
    const uint32_t* pg32 = reinterpret_cast<const uint32_t*>(pg + 16 * H);
    const uint8_t* tile = pg + hdr_bytes(H, A) + (int64_t)h * HEADPAGE;
    uint16_t* row0 = out + ((int64_t)dst[j] * PT * H + (h - h % HALVES)) * 2 * D + (h % HALVES) * D;  // (token t's row is t H 2 D further)
    {
        uint4 f[8];
        Raw r;
        raw_load(r, tile, lane);
        uint32_t hb[8];
        load_bases<0>(kbase + h * D, lane, hb);
        decode_tile<0>(r, hb, hd, pg32, ent, nc, A, R, h, lane, f);
#pragma unroll
        for (int i = 0; i < 16; i++) {  // keys: group i is token 8 (i & 1) + g, channels 16 (i >> 1) + 2 q + {0, 1} and + {8, 9}: two words
            const uint4& c = f[i >> 1];
            const uint32_t w0 = (i & 1) ? c.y : c.x, w1 = (i & 1) ? c.w : c.z;
            uint16_t* p = row0 + (int64_t)(8 * (i & 1) + g) * H * 2 * D + 16 * (i >> 1) + 2 * q;
            *reinterpret_cast<uint32_t*>(p) = w0;
            *reinterpret_cast<uint32_t*>(p + 8) = w1;
        }
    }
    {
        uint4 f[8];
        Raw r;
        raw_load(r, tile + TILE, lane);
        uint32_t hb[16];
        load_bases<1>(vbase + h * D, lane, hb);
        decode_tile<1>(r, hb, hd, pg32, ent, nc, A, R, h, lane, f);
#pragma unroll
        for (int i = 0; i < 16; i++) {  // values: group i is channel g + 8 i, tokens 2 q + {0, 1} and 8 + 2 q + {0, 1}
            const uint4& c = f[i >> 1];
            const uint32_t w0 = (i & 1) ? c.y : c.x, w1 = (i & 1) ? c.w : c.z;
            uint16_t* p = row0 + HALVES * D + g + 8 * i;
            p[(int64_t)(2 * q) * H * 2 * D] = (uint16_t)(w0 & 0xFFFFu);
            p[(int64_t)(2 * q + 1) * H * 2 * D] = (uint16_t)(w0 >> 16);
            p[(int64_t)(2 * q + 8) * H * 2 * D] = (uint16_t)(w1 & 0xFFFFu);
            p[(int64_t)(2 * q + 9) * H * 2 * D] = (uint16_t)(w1 >> 16);
        }
    }
}

// ---- a step's own keys and values into the bf16 staging cache ------------------------------------------------------------------------
// Token t's keys and values (rows of [H][DD] bf16 at key + t key_stride, value + t value_stride) into the slot slots[t] of `out` [blocks][16][H][2 DD]: its keys at (slot H + h) 2 DD and its values DD
// after; a negative slot (or one past the cache) is skipped, so a plan's group takes the tokens of its rows and leaves the others' (one CTA a token, 4-byte words).
template <int DD>
__global__ void __launch_bounds__(128) stage_put_kernel(const uint16_t* __restrict__ key, int64_t key_stride, const uint16_t* __restrict__ value, int64_t value_stride, const int64_t* __restrict__ slots,
                                                        int64_t nslots, int H, uint16_t* __restrict__ out) {
    constexpr int D = DD;
    const int64_t t = blockIdx.x, slot = slots[t];
    if (slot < 0 || slot >= nslots) return;
    const uint32_t* kp = reinterpret_cast<const uint32_t*>(key + t * key_stride);
    const uint32_t* vp = reinterpret_cast<const uint32_t*>(value + t * value_stride);
    uint32_t* row = reinterpret_cast<uint32_t*>(out + slot * H * 2 * D);
    for (int i = threadIdx.x; i < H * D / 2; i += blockDim.x) {
        const int h = i / (D / 2), c = i % (D / 2);  // (head, word of its D channels)
        row[h * D + c] = kp[i];
        row[h * D + D / 2 + c] = vp[i];
    }
}

// ---- decode attention ---------------------------------------------------------------------------------------------------
// One query token a sequence over its pages, G query heads (up to 8) to a kv head, fp32 softmax and accumulation, split over CTAs
// (flash decoding). A CTA is (split, kv head, sequence): NW warps, each the pages split * ppc + w, + NW, ..., one online softmax a
// warp; the CTA's warps are merged through shared memory into one partial (max, sum, output), which goes to a workspace that
// combine_kernel merges over the splits. The scores are S^T = K Q^T (K's tile the A operand,
// the G heads the columns of the 8-column B operand) and the output O^T = V^T P^T (V^T the A operand, P^T the scores' fragments by
// movmatrix): both straight from registers, with P rounded to bf16 for the second product as FlashAttention does.
// A head of 256 channels (HALVES 2) is two of the page's heads of 128 (2 h: channels 0-127, 2 h + 1: 128-255): a page is two warps', each its half's tiles, their
// scores summed through shared memory before the softmax (the pair's two warps hold the same sums, maxima and weights), each warp the output of its half's channels.
struct AttnParams {
    const __nv_bfloat16* q;  // sequence b's queries at q + b * q_stride: [H * G][D HALVES]
    int64_t q_stride;
    const uint8_t* pool;
    int64_t page_stride;
    const uint8_t* kbase;  // [H][D]
    const uint8_t* vbase;
    const int32_t* bt;  // [B][bt_stride]: page index of a sequence's page
    int64_t bt_stride;
    const int32_t* slen;  // [B]
    const uint32_t* ent;  // the escape pool's entries (chunks of CH), where a page's directory names them, and its chunks
    int nc;
    RawPool raw;
    float* ws_ml;         // [B * H][splits][G][2]
    float* ws_o;          // [B * H][splits][G][D HALVES]
    int H, G, A, hdrb, splits;  // splits: the grid's x, the most CTAs a (sequence, kv head) is split over (the partials' stride)
    int HP;                     // the page's heads of D channels: H HALVES
    float sl2;  // scale * log2(e)
    int ppc_min;  // the plan's least pages a CTA
    int pfloor;   // the plan's CTAs a sequence has at least (as far as it has 4 pages for each)
    uint32_t* cnt;  // [B * H] the CTAs of a (sequence, head) that have written their partials (zero between calls): the last merges them (FUSE)
    __nv_bfloat16* out;
    int64_t out_stride;
};

// The plan, from a sequence's own length alone (so a sequence's partition, and its output's bits, do not depend on the batch it is in, nor on a batch padded for a CUDA graph): at
// least ppc_min pages a CTA (16 by default: 4 a warp), at most `splits` CTAs, each the same share of the pages (ceil); a sequence of no tokens has one CTA, whose partial is empty.
// pfloor (1: none): a sequence has at least that many CTAs where its pages are 4 for each (a page a warp), however few ppc_min would give it: a batch of few sequences leaves most of a
// big GPU's SMs without a CTA otherwise. It is the same number for every call whose outputs are compared (the host's, from the batch's size and the GPU's: kv_floor).
constexpr int PPC_MIN = NW * 4;
__device__ __forceinline__ int plan_splits(int np, int splits, int ppc_min, int pfloor) {
    return min(splits, max(max(1, (np + ppc_min - 1) / ppc_min), min(pfloor, (np + NW - 1) / NW)));
}

// The partials of a (sequence, kv head) merged for query head `head`, 128 of its DO channels at a time: a lane 4 channels (a sequence of no tokens gives zeros). The partials' reads are L2's (another
// CTA's, or another kernel's), so they are made in as few waits as there are: the partials' maxima and sums a lane a partial (32 at a time) and the max by shuffles (a max does not depend on its order),
// then the weights and the sums in the partials' order, with the vectors of 16 partials in flight.
template <int DO>
__device__ __forceinline__ void merge_head(const float* __restrict__ mlp, const float* __restrict__ wop, __nv_bfloat16* __restrict__ op, int G, int head, int nparts, int lane) {
#pragma unroll 1
    for (int part = 0; part < DO / D; part++) {
        float M = -INFINITY;
        for (int c = 0; c < nparts; c += 32)
            if (c + lane < nparts) M = fmaxf(M, __ldcg(mlp + ((c + lane) * G + head) * 2));
#pragma unroll
        for (int o = 16; o; o >>= 1) M = fmaxf(M, __shfl_xor_sync(FULLM, M, o));
        float L = 0.f, a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        if (M != -INFINITY) {
            for (int c = 0; c < nparts; c += 32) {
                const int n = min(32, nparts - c);
                float2 ml = make_float2(-INFINITY, 0.f);
                if (lane < n) ml = __ldcg(reinterpret_cast<const float2*>(mlp + ((c + lane) * G + head) * 2));
                const float wl = ex2(ml.x - M), ll = ml.y;  // (lane j: partial c + j's weight and sum)
#pragma unroll 16
                for (int j = 0; j < n; j++) {
                    const float w = __shfl_sync(FULLM, wl, j);
                    L += w * __shfl_sync(FULLM, ll, j);
                    const float4 v = __ldcg(reinterpret_cast<const float4*>(wop + ((size_t)(c + j) * G + head) * DO + part * D + lane * 4));
                    a0 += w * v.x;
                    a1 += w * v.y;
                    a2 += w * v.z;
                    a3 += w * v.w;
                }
            }
        }
        const float inv = L > 0.f ? 1.f / L : 0.f;
        __nv_bfloat16* p = op + (int64_t)head * DO + part * D + lane * 4;
        *reinterpret_cast<__nv_bfloat162*>(p) = __floats2bfloat162_rn(a0 * inv, a1 * inv);
        *reinterpret_cast<__nv_bfloat162*>(p + 2) = __floats2bfloat162_rn(a2 * inv, a3 * inv);
    }
}

struct Buf {  // a page's raw reads
    Raw kr, vr;
    uint4 hd;
    const uint32_t* pg;  // the page's `used` word, directory and arena
};
__device__ __forceinline__ void load_buf(Buf& c, const AttnParams& P, int b, int h, int pg, int lane) {
    const uint8_t* page = P.pool + (int64_t)__ldg(P.bt + (int64_t)b * P.bt_stride + pg) * P.page_stride;
    const uint8_t* kt = page + P.hdrb + (int64_t)h * HEADPAGE;
    raw_load(c.kr, kt, lane);
    raw_load(c.vr, kt + TILE, lane);
    c.hd = ldg_stream<0>(page + 16 * h);
    c.pg = reinterpret_cast<const uint32_t*>(page + 16 * P.HP);
}

// PIPE: the next page's loads are issued before the page in hand is computed (two pages of raw data a warp in registers). FUSE: the last CTA of a (sequence, kv head) to write its partial merges
// them all (the arithmetic of combine_kernel, which a launch of its own does otherwise). HALVES 2 and 4: a head of 256 or 512 channels, a page two or four warps' (AttnParams).
template <bool PIPE, bool FUSE, int HALVES>
__global__ void __launch_bounds__(NW * 32, 1) attn_kernel(const AttnParams P) {
    constexpr int DO = D * HALVES;   // a query head's channels
    constexpr int NP = NW / HALVES;  // pages a CTA has in hand at once
    const int split = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
    const int hf = warp % HALVES, pw = warp / HALVES;  // the warp's half of the channels, and its page's place among the CTA's
    const int hp = h * HALVES + hf;                    // the page's head the warp reads
    const int len = P.slen[b];
    const int np = (len + 15) >> 4;
    const int sb = plan_splits(np, P.splits, P.ppc_min, P.pfloor);  // the CTAs this sequence has: past them, no partial (the combine reads sb of them)
    if (split >= sb) return;
    const int ppc = (np + sb - 1) / sb;
    const int p0 = min(np, split * ppc), p1 = min(np, p0 + ppc);
    const int G = P.G;

    uint32_t qb0[8], qb1[8];  // the B operand of the scores: column g = query head g (zero past G), rows the step's channels 2q, 2q + 1 and 2q + 8, 2q + 9
    {
        const uint32_t* qp = reinterpret_cast<const uint32_t*>(P.q + (int64_t)b * P.q_stride + (int64_t)(h * G + (g < G ? g : 0)) * DO + hf * D);
#pragma unroll
        for (int s = 0; s < 8; s++) {
            qb0[s] = g < G ? qp[8 * s + q] : 0u;
            qb1[s] = g < G ? qp[8 * s + q + 4] : 0u;
        }
    }
    uint32_t kb[8], vb[16];
    load_bases<0>(P.kbase + hp * D, lane, kb);
    load_bases<1>(P.vbase + hp * D, lane, vb);

    // a lane's softmax state is for heads 2q (index 0) and 2q + 1 (index 1); the lanes of the other g hold the same heads' other tokens
    float m_run0 = -INFINITY, m_run1 = -INFINITY, l_run0 = 0.f, l_run1 = 0.f;
    float o[8][4];  // O^T: channels 16i + g (o[i][0], o[i][1]) and 16i + 8 + g (o[i][2], o[i][3]) of heads 2q, 2q + 1 (of the warp's half)
#pragma unroll
    for (int i = 0; i < 8; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;

    Buf cur, nxt;  // the page in hand, and (PIPE) the next one
    int pg = p0 + pw;
    [[maybe_unused]] int par = 0;  // (HALVES 2: the scores' buffer of this page, one of two in turn)
    if constexpr (PIPE) {
        if (pg < p1) load_buf(cur, P, b, hp, pg, lane);
    }
    for (; pg < p1; pg += NP) {
        if constexpr (PIPE) {
            if (pg + NP < p1) load_buf(nxt, P, b, hp, pg + NP, lane);
        } else {
            load_buf(cur, P, b, hp, pg, lane);
        }
        uint4 f[8];
        decode_tile<0>(cur.kr, kb, cur.hd, cur.pg, P.ent, P.nc, P.A, P.raw, hp, lane, f);

        float s[4] = {0.f, 0.f, 0.f, 0.f};  // tokens g (s[0], s[1]) and g + 8 (s[2], s[3]) of heads 2q, 2q + 1
#pragma unroll
        for (int i = 0; i < 8; i++) mma(s, f[i].x, f[i].y, f[i].z, f[i].w, qb0[i], qb1[i]);
        if constexpr (HALVES >= 2) {  // the halves' scores summed: each of a page's warps adds the others' to its own, in the order of the halves, so that all of them hold the same sums bit for bit
            __shared__ float4 s_x[2][NW][32];
            s_x[par][warp][lane] = make_float4(s[0], s[1], s[2], s[3]);
#if defined(__HIP_PLATFORM_AMD__)
            __trap();  // (no named barriers on RDNA: heads of 256 channels and more are CUDA's alone)
#else
            asm volatile("bar.sync %0, %1;" ::"r"(1 + pw), "r"(HALVES * 32) : "memory");  // (the page's warps alone; a buffer is written again two pages on, after the warps have passed the next barrier)
#endif
            if constexpr (HALVES == 2) {
                const float4 x = s_x[par][warp ^ 1][lane];  // (a + b is b + a, bit for bit)
                s[0] += x.x;
                s[1] += x.y;
                s[2] += x.z;
                s[3] += x.w;
            } else {  // 4 halves: (h0 + h1) + (h2 + h3), the same expression in every warp
                const float4 x0 = s_x[par][pw * HALVES][lane], x1 = s_x[par][pw * HALVES + 1][lane], x2 = s_x[par][pw * HALVES + 2][lane], x3 = s_x[par][pw * HALVES + 3][lane];
                s[0] = (x0.x + x1.x) + (x2.x + x3.x);
                s[1] = (x0.y + x1.y) + (x2.y + x3.y);
                s[2] = (x0.z + x1.z) + (x2.z + x3.z);
                s[3] = (x0.w + x1.w) + (x2.w + x3.w);
            }
            par ^= 1;
        }
        float x0 = s[0] * P.sl2, x1 = s[1] * P.sl2, x2 = s[2] * P.sl2, x3 = s[3] * P.sl2;
        const int valid = len - 16 * pg;
        if (valid < 16) {
            x0 = g < valid ? x0 : -INFINITY;
            x1 = g < valid ? x1 : -INFINITY;
            x2 = g + 8 < valid ? x2 : -INFINITY;
            x3 = g + 8 < valid ? x3 : -INFINITY;
        }
        float a0 = fmaxf(x0, x2), a1 = fmaxf(x1, x3);
        a0 = fmaxf(a0, __shfl_xor_sync(FULLM, a0, 4));
        a1 = fmaxf(a1, __shfl_xor_sync(FULLM, a1, 4));
        a0 = fmaxf(a0, __shfl_xor_sync(FULLM, a0, 8));
        a1 = fmaxf(a1, __shfl_xor_sync(FULLM, a1, 8));
        a0 = fmaxf(a0, __shfl_xor_sync(FULLM, a0, 16));
        a1 = fmaxf(a1, __shfl_xor_sync(FULLM, a1, 16));
        const float mn0 = fmaxf(m_run0, a0), mn1 = fmaxf(m_run1, a1);
        const float al0 = ex2(m_run0 - mn0), al1 = ex2(m_run1 - mn1);
        const float p0f = ex2(x0 - mn0), p1f = ex2(x1 - mn1), p2f = ex2(x2 - mn0), p3f = ex2(x3 - mn1);
        l_run0 = l_run0 * al0 + (p0f + p2f);
        l_run1 = l_run1 * al1 + (p1f + p3f);
        if (__any_sync(FULLM, (mn0 > m_run0) | (mn1 > m_run1))) {  // the output's scale, per head
#pragma unroll
            for (int i = 0; i < 8; i++) {
                o[i][0] *= al0;
                o[i][1] *= al1;
                o[i][2] *= al0;
                o[i][3] *= al1;
            }
        }
        m_run0 = mn0;
        m_run1 = mn1;
        const uint32_t pb0 = transpose8x8(pack_bf16x2(p0f, p1f)), pb1 = transpose8x8(pack_bf16x2(p2f, p3f));  // P^T: tokens 2q, 2q + 1 and 8 + 2q, 9 + 2q of head g

        decode_tile<1>(cur.vr, vb, cur.hd, cur.pg, P.ent, P.nc, P.A, P.raw, hp, lane, f);
        if (valid < 16) {  // the tokens past the sequence's end zeroed (a recycled page's bytes there can decode to infinity, which times a probability of 0 is NaN)
            const uint32_t k0 = (2 * q < valid ? 0x0000FFFFu : 0u) | (2 * q + 1 < valid ? 0xFFFF0000u : 0u);
            const uint32_t k1 = (2 * q + 8 < valid ? 0x0000FFFFu : 0u) | (2 * q + 9 < valid ? 0xFFFF0000u : 0u);
#pragma unroll
            for (int i = 0; i < 8; i++) {
                f[i].x &= k0;
                f[i].y &= k0;
                f[i].z &= k1;
                f[i].w &= k1;
            }
        }
#pragma unroll
        for (int i = 0; i < 8; i++) mma(o[i], f[i].x, f[i].y, f[i].z, f[i].w, pb0, pb1);
        if constexpr (PIPE) cur = nxt;
    }

    // the sums over the lanes of the same q (the other tokens)
    float l0 = l_run0, l1 = l_run1;
    l0 += __shfl_xor_sync(FULLM, l0, 4);
    l1 += __shfl_xor_sync(FULLM, l1, 4);
    l0 += __shfl_xor_sync(FULLM, l0, 8);
    l1 += __shfl_xor_sync(FULLM, l1, 8);
    l0 += __shfl_xor_sync(FULLM, l0, 16);
    l1 += __shfl_xor_sync(FULLM, l1, 16);
    // the warps' states into shared memory (a warp's lanes of the same q hold the same max and sum: heads 2q and 2q + 1; the output's channels 16 i + g, 16 i + 8 + g of those heads, of its half)
    __shared__ float s_ml[NW][8][2];
    __shared__ float s_o[NW][8][D];
    if (g == 0) {
        s_ml[warp][2 * q][0] = m_run0;
        s_ml[warp][2 * q][1] = l0;
        s_ml[warp][2 * q + 1][0] = m_run1;
        s_ml[warp][2 * q + 1][1] = l1;
    }
#pragma unroll
    for (int i = 0; i < 8; i++) {
        s_o[warp][2 * q][16 * i + g] = o[i][0];
        s_o[warp][2 * q + 1][16 * i + g] = o[i][1];
        s_o[warp][2 * q][16 * i + 8 + g] = o[i][2];
        s_o[warp][2 * q + 1][16 * i + 8 + g] = o[i][3];
    }
    __syncthreads();
    // the CTA's partial: a thread a channel (DO / D of them, D the CTA's threads), a head at a time, over its pages' places in order (a pair's two warps hold the same max and sum); a CTA of no pages
    // has the max -infinity, the sum 0 and the output 0
    const size_t pidx = ((size_t)(b * P.H + h) * P.splits + split) * G;
    for (int head = 0; head < G; head++) {
        float M = s_ml[0][head][0];
#pragma unroll
        for (int w = 1; w < NP; w++) M = fmaxf(M, s_ml[w * HALVES][head][0]);
#pragma unroll
        for (int c = threadIdx.x; c < DO; c += NW * 32) {
            const int half = c / D, ch = c % D;
            float L = 0.f, O = 0.f;
            if (M != -INFINITY) {
#pragma unroll
                for (int w = 0; w < NP; w++) {
                    const float sc = ex2(s_ml[w * HALVES][head][0] - M);
                    L += sc * s_ml[w * HALVES][head][1];
                    O += sc * s_o[w * HALVES + half][head][ch];
                }
            }
            P.ws_o[(pidx + head) * DO + c] = O;
            if (c == 0) {
                P.ws_ml[(pidx + head) * 2] = M;
                P.ws_ml[(pidx + head) * 2 + 1] = L;
            }
        }
    }
    if constexpr (FUSE) {
        __shared__ unsigned s_last;
        __threadfence();  // (the partial is in L2 before the count says so)
        __syncthreads();
        if (threadIdx.x == 0) s_last = atomicAdd(P.cnt + (size_t)b * P.H + h, 1u) == (unsigned)(sb - 1);
        __syncthreads();
        if (!s_last) return;
        __threadfence();
        const size_t bh = (size_t)b * P.H + h;
        for (int head = warp; head < G; head += NW) merge_head<DO>(P.ws_ml + bh * P.splits * G * 2, P.ws_o + bh * P.splits * G * DO, P.out + (int64_t)b * P.out_stride + (int64_t)h * G * DO, G, head, sb, lane);
        if (threadIdx.x == 0) P.cnt[bh] = 0;  // (zero for the next call)
    }
}

// the partials of a (sequence, kv head) merged: one warp a query head, a lane DO / 32 channels; a sequence of no tokens gives zeros
template <int DO>
__global__ void __launch_bounds__(256) combine_kernel(const float* __restrict__ ml, const float* __restrict__ wo, __nv_bfloat16* __restrict__ out, int64_t out_stride, int H, int G, int splits, int ppc_min,
                                                      int pfloor, const int32_t* __restrict__ slen) {
    const int bh = blockIdx.x, head = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (head >= G) return;
    const int b = bh / H, h = bh % H;
    const int nparts = plan_splits((slen[b] + 15) >> 4, splits, ppc_min, pfloor);  // (the partials the sequence's CTAs wrote: the attention kernel's plan)
    merge_head<DO>(ml + (size_t)bh * splits * G * 2, wo + (size_t)bh * splits * G * DO, out + (int64_t)b * out_stride + (int64_t)h * G * DO, G, head, nparts, lane);
}

}  // namespace kvc

// ---------------------------------------------------------------------------
// The lossless FP8 (e4m3) weights: glyd_gpu/fp8ref.py is the format (its docstring the layout, byte for byte; gpu/fp8/README.md the choice).
// A matrix [O, K] of bytes (any byte is a value: NaNs, both zeros) is cut into patches of 64 rows by 128 columns (zero bytes past its edges),
// a patch 16 units of 16 x 32, a unit 32 lanes of 16 values, in the order a lane holds the A operand of mma.m16n8k32 with e4m3. A patch is
// packed and unpacked on its own (and each of its row blocks of 16 rows, by the offsets in its head): its core (6 bits a value: sign and mantissa, a
// 2-bit digit), its head (8 words) and its part of var (the values the digit does not name). The pack and the unpack: one CTA of 256 threads a patch,
// a thread a lane of two units (32 values); the 8 KiB patch is staged in shared memory, so that the matrix is read and written in rows.
// Everything between `f8c begin` and `f8c end` is compiled on the CPU, thread by thread, by gpu/fp8/emu_f8.cpp (gpu/fp8/emu_check.py): the
// layout is checked against fp8ref.py there with no GPU.
// f8c begin
#ifndef F8_ALIGN
#define F8_ALIGN(p, n) ((void)0)
#endif
namespace f8c {

constexpr int PR = 64, PC = 128, VALUES = PR * PC;  // a patch: rows, columns, values
constexpr int CORE = VALUES * 6 / 8;                // its core's bytes
constexpr int TAIL = (VALUES - CORE) / 4;           // the words of var a plain patch holds, and the most a coded one's variable part has (less one)
constexpr int HEAD = 8;                             // a head's words
constexpr int MAX_LIT = 1024;                       // a coded patch has fewer literals than this
constexpr int GUARD = 8;                            // zero words after the last patch's part of var
constexpr int NT = 256;                             // threads a CTA: a thread's 32 values are stream bytes 32 tid on (a lane of two units)
constexpr int SCAN_NT = 1024;
static_assert(CORE == 6144 && TAIL == 512 && NT * 32 == VALUES, "the layout of fp8ref.py");

__host__ __device__ constexpr int64_t f8_cdiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

// The pack's arguments: the matrix w [O, K] by rows ld bytes apart, KT patches a row of patches, vec: 16-byte loads allowed.
struct F8Pk {
    const uint8_t* w;
    int64_t ld;
    int O, K, KT, vec;
    int plain;  // layout H: every patch plain (its e4m3 bytes as they are; GLYD_F8H_PLAIN=1, a measurement)
};
// The unpack's: out by rows ld bytes apart; ids (null: the one matrix) the items to take (item i of E: the matrix ids[i] of a stack of E, `per` patches
// each, its rows `item` bytes apart in out); vec: 16-byte stores allowed.
struct F8Un {
    const uint8_t* core;
    const uint32_t* heads;
    const uint32_t* var;
    uint8_t* out;
    const int32_t* ids;
    int64_t ld, item;
    int O, K, KT, per, E, vec;
};

__device__ __forceinline__ uint4 f8_ld16(const void* p) {
    F8_ALIGN(p, 16);
    return __ldcs((const uint4*)p);
}
__device__ __forceinline__ uint2 f8_ld8(const void* p) {
    F8_ALIGN(p, 8);
    return __ldcs((const uint2*)p);
}
__device__ __forceinline__ uint32_t f8_threes(uint32_t d) { return d & (d >> 1) & 0x55555555u; }  // the 2-bit digits of d that are 3: a bit in the low place of each
// The low 2 c bits (c: 0 to 16).
__device__ __forceinline__ uint32_t f8_low2(uint32_t c) { return c >= 16 ? 0xFFFFFFFFu : (1u << (2 * c)) - 1u; }
// The low c nibbles (c: 0 to 8; a funnel shift that clamps), the low c bytes (c: any).
__device__ __forceinline__ uint32_t f8_nibs(uint32_t c) { return __funnelshift_rc(0xFFFFFFFFu, 0u, 32 - 4 * c); }
__device__ __forceinline__ uint32_t f8_bytes(int c) { return c <= 0 ? 0u : c >= 4 ? 0xFFFFFFFFu : (1u << (8 * c)) - 1u; }
// prmt as the PTX instruction is (a selector nibble's top bit: the byte's sign spread), on the selector's low 16 bits as they are (__byte_perm masks them;
// the bits above them are never read, so a selector here may have anything there).
__device__ __forceinline__ uint32_t f8_prmt(uint32_t a, uint32_t b, uint32_t s) {
#if defined(__CUDA_ARCH__)
    uint32_t d;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));
    return d;
#elif defined(__HIP_DEVICE_COMPILE__)
    // (prmt byte by byte: a selector nibble's low 3 bits pick a byte of b:a, its top bit spreads that byte's sign; HIP's __byte_perm has no sign mode)
    const uint64_t x = (uint64_t)b << 32 | a;
    uint32_t d = 0;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const uint32_t n = s >> (4 * i) & 0xFu, v = (uint32_t)(x >> (8 * (n & 7))) & 0xFFu;
        d |= (n & 8 ? (v & 0x80u ? 0xFFu : 0u) : v) << (8 * i);
    }
    return d;
#else
    return __byte_perm(a, b, s);  // (the emulation's: the sign mode of a selector's top bit as prmt's)
#endif
}
// 32 bits of the digits (2 bits each) at s from digit k on; 32 bits of the bit string s from bit p on.
__device__ __forceinline__ uint32_t f8_digits(const uint32_t* s, uint32_t k) { return __funnelshift_r(s[k >> 4], s[(k >> 4) + 1], (k & 15) << 1); }
__device__ __forceinline__ uint32_t f8_bits(const uint32_t* s, uint32_t p) { return __funnelshift_r(s[p >> 5], s[(p >> 5) + 1], p & 31); }
// The 3-bit digits that are 7 among the first n (n: 0 to 16) of the 64 bits lo, hi.
__device__ __forceinline__ uint32_t f8_sevens(uint32_t lo, uint32_t hi, uint32_t n) {
    if (n <= 10) return __popc(lo & (lo >> 1) & (lo >> 2) & 0x09249249u & __funnelshift_rc(0xFFFFFFFFu, 0u, 32 - 3 * n));
    const uint64_t w = (uint64_t)lo | ((uint64_t)hi << 32);
    return (uint32_t)__popcll(w & (w >> 1) & (w >> 2) & 0x1249249249249249ull & ((1ull << (3 * n)) - 1ull));
}
// Where word wc (0 to 31) of row r of a patch is in its 8 KiB of shared memory, as a word index: a row is 32 words, its 8 groups of 4 words in the order
// (group ^ r % 8), so that the 8 rows a half-warp stores or reads a group of do not meet in a bank.
__device__ __forceinline__ int f8_word(int r, int wc) { return r * 32 + ((((wc >> 2) ^ r) & 7) << 2) + (wc & 3); }

// The sum over the CTA's threads before this one, in thread order. tot: 8 words that no other call uses. Every thread calls it; it holds a __syncthreads().
__device__ __forceinline__ uint32_t f8_excl(uint32_t v, uint32_t* tot) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    uint32_t x = v;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const uint32_t y = __shfl_up_sync(0xffffffffu, x, d);
        if (lane >= d) x += y;
    }
    if (lane == 31) tot[w] = x;
    __syncthreads();
    uint32_t b = 0;
    for (int i = 0; i < w; i++) b += tot[i];
    return b + x - v;
}

// A thread's 32 bytes of the patch `patch` as 8 words: row tid / 4, columns 32 (tid % 4) on (0 where they are past the matrix's edge).
__device__ __forceinline__ void f8_load(const F8Pk& P, int64_t patch, int tid, uint32_t (&x)[8]) {
    const int rp = (int)(patch / P.KT), cp = (int)(patch % P.KT);
    const int row = rp * PR + (tid >> 2), col = cp * PC + (tid & 3) * 32;
#pragma unroll
    for (int i = 0; i < 8; i++) x[i] = 0;
    if (row >= P.O) return;
    const uint8_t* s = P.w + (int64_t)row * P.ld + col;
    if (P.vec && col + 32 <= P.K) {
        F8_ALIGN(s, 16);
        const uint4 a = __ldg((const uint4*)s), b = __ldg((const uint4*)(s + 16));
        x[0] = a.x, x[1] = a.y, x[2] = a.z, x[3] = a.w, x[4] = b.x, x[5] = b.y, x[6] = b.z, x[7] = b.w;
    } else {
#pragma unroll
        for (int i = 0; i < 32; i++)
            if (col + i < P.K) x[i >> 2] |= (uint32_t)s[i] << (8 * (i & 3));
    }
}

// Pass 1 of the pack: each patch's head (heads[8 p], its words of var in the first, its base to come from the scan) from the counts of its values'
// exponents, a row block's apart (warps 2 mb and 2 mb + 1 hold row block mb's rows).
__global__ void __launch_bounds__(NT) f8_head_kernel(const F8Pk P, uint32_t* heads) {
    __shared__ uint32_t s_cnt[NT / 32][16];
    __shared__ uint32_t s_sum[16];
    __shared__ uint8_t s_sym[16], s_rank[16];
    const int tid = threadIdx.x;
    if (tid < NT / 2) ((uint32_t*)s_cnt)[tid] = 0;
    __syncthreads();
    uint32_t x[8];
    f8_load(P, blockIdx.x, tid, x);
#pragma unroll
    for (int i = 0; i < 8; i++)
#pragma unroll
        for (int k = 0; k < 4; k++) atomicAdd(&s_cnt[tid >> 5][(x[i] >> (8 * k + 3)) & 15], 1u);
    __syncthreads();
    if (tid < 16) {
        uint32_t c = 0;
        for (int w = 0; w < NT / 32; w++) c += s_cnt[w][tid];
        s_sum[tid] = c;
    }
    __syncthreads();
    if (tid < 16) {  // the rank of exponent tid: the exponents before it
        uint32_t r = 0;
        for (int f = 0; f < 16; f++) r += (s_sum[f] > s_sum[tid]) | ((s_sum[f] == s_sum[tid]) & (f > tid));
        s_sym[r] = (uint8_t)tid;
        s_rank[tid] = (uint8_t)r;
    }
    __syncthreads();
    if (tid == 0) {
        uint32_t n1 = 0, n2 = 0, n3 = 0, o2[4], o3[4], o4[4];
        for (int mb = 0; mb < 4; mb++) {
            o2[mb] = n1, o3[mb] = n2, o4[mb] = n3;
            for (int e = 0; e < 16; e++) {
                const uint32_t c = s_cnt[2 * mb][e] + s_cnt[2 * mb + 1][e], r = s_rank[e];
                n1 += r >= 3 ? c : 0u;
                n2 += r >= 6 ? c : 0u;
                n3 += r >= 13 ? c : 0u;
            }
        }
        const uint32_t words = ((n1 + 15) >> 4) + ((3 * n2 + 31) >> 5) + ((n3 + 7) >> 3);
        uint4* h = (uint4*)(heads + (int64_t)blockIdx.x * HEAD);
        if (words >= (uint32_t)TAIL || n3 >= (uint32_t)MAX_LIT) {
            h[0] = make_uint4(TAIL, 0, 0, 1u << 31);
            h[1] = make_uint4(0, 0, 0, 0);
        } else {
            uint32_t s1 = 0, s2 = 0;
            for (int r = 0; r < 8; r++) s1 |= (uint32_t)s_sym[r] << (4 * r);
            for (int r = 0; r < 5; r++) s2 |= (uint32_t)s_sym[8 + r] << (4 * r);
            h[0] = make_uint4(words, s1, s2 | (n3 << 20), n1 | (n2 << 14));
            h[1] = make_uint4(o2[1] | (o2[2] << 16), o2[3] | (o3[1] << 16), o3[2] | (o3[3] << 16), o4[1] | (o4[2] << 10) | (o4[3] << 20));
        }
    }
}

// Pass 2: heads[8 p] the words of each patch's part of var, to the sum of those before it; the total to *total. One CTA.
__global__ void __launch_bounds__(SCAN_NT) f8_scan_kernel(uint32_t* heads, int64_t patches, uint64_t* total) {
    __shared__ uint32_t s_tot[SCAN_NT / 32];
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    uint64_t carry = 0;
    for (int64_t t0 = 0; t0 < patches; t0 += SCAN_NT) {
        const int64_t i = t0 + tid;
        const uint32_t v = i < patches ? heads[i * HEAD] : 0;
        uint32_t x = v;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const uint32_t y = __shfl_up_sync(0xffffffffu, x, d);
            if (lane >= d) x += y;
        }
        if (lane == 31) s_tot[w] = x;
        __syncthreads();
        uint32_t b = 0, all = 0;
        for (int k = 0; k < SCAN_NT / 32; k++) {
            const uint32_t t = s_tot[k];
            b += k < w ? t : 0u;
            all += t;
        }
        if (i < patches) heads[i * HEAD] = (uint32_t)(carry + b + x - v);
        carry += all;
        __syncthreads();
    }
    if (tid == 0) *total = carry;
}

// A value's place in its lane's core words: its d1 digit's bit (j: 0 to 15), its sign's and its mantissa's in the word of values 8 (j >> 3) on.
__device__ __forceinline__ int f8_d1_bit(int j) { return 4 * (j & 7) + 2 * (j >> 3); }
__device__ __forceinline__ int f8_s_bit(int j) { return (j & 4) ? 8 * (j & 3) + 3 : 8 * (j & 3) + 7; }
__device__ __forceinline__ int f8_m_bit(int j) { return (j & 4) ? 8 * ((j + 3) & 3) + 4 : 8 * (j & 3); }

// Pass 3: each patch's core and part of var (heads from passes 1 and 2). Thread tid: row block mb = tid / 64, lane l = (tid / 2) % 32, units 4 mb + 2 (tid % 2)
// and the next (its 32 stream bytes, 32 tid on).
__global__ void __launch_bounds__(NT) f8_pack_kernel(const F8Pk P, const uint32_t* heads, uint8_t* core_out, uint32_t* var, int64_t var_words) {
    __shared__ __align__(16) uint32_t s_tile[VALUES / 4];
    __shared__ uint32_t s_var[TAIL + 8];
    __shared__ uint32_t s_tot[2][NT / 32];
    __shared__ uint8_t s_rank[16];
    const int tid = threadIdx.x, mb = tid >> 6, l = (tid >> 1) & 31, hh = tid & 1, g = l >> 2, t = l & 3;
    const int64_t pat = blockIdx.x;
    const uint4 hd = __ldg((const uint4*)(heads + pat * HEAD));
    const uint32_t base = hd.x, h1 = hd.y, h2 = hd.z, h3 = hd.w;
    const bool plain = (h3 >> 31) != 0;
    const uint32_t n1 = h3 & 0x3FFF, n2 = (h3 >> 14) & 0x3FFF, n3 = h2 >> 20;
    const uint32_t w2 = (n1 + 15) >> 4, w3 = (3 * n2 + 31) >> 5, words = plain ? (uint32_t)TAIL : w2 + w3 + ((n3 + 7) >> 3);
    if (blockIdx.x == 0 && tid < GUARD) var[var_words - GUARD + tid] = 0;
    if ((int64_t)base + words > var_words - GUARD) return;  // (a heads array that is not this matrix's: nothing is written)
    if (tid < 16) {  // the rank of an exponent: its place in the head's list, 13 for any other
        uint32_t r = 13;
        for (int i = 0; i < 13; i++)
            if (((i < 8 ? h1 >> (4 * i) : h2 >> (4 * (i - 8))) & 15) == (uint32_t)tid) r = i;
        s_rank[tid] = (uint8_t)r;
    }
    for (int i = tid; i < TAIL; i += NT) s_var[i] = 0;
    {  // the patch in rows, into shared memory
        uint32_t x[8];
        f8_load(P, pat, tid, x);
        const int row = tid >> 2, wc = (tid & 3) * 8;
        *(uint4*)&s_tile[f8_word(row, wc)] = make_uint4(x[0], x[1], x[2], x[3]);
        *(uint4*)&s_tile[f8_word(row, wc + 4)] = make_uint4(x[4], x[5], x[6], x[7]);
    }
    __syncthreads();
    uint32_t y[2][4];  // unit h (ks = 2 hh + h), value j = 4 q + k: byte k of y[h][q] (the stream's bytes 32 tid on)
#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
        for (int q = 0; q < 4; q++) y[h][q] = s_tile[f8_word(16 * mb + g + 8 * (q & 1), 8 * (2 * hh + h) + t + 4 * (q >> 1))];
    uint8_t* core = core_out + pat * CORE;
    if (plain) {
        if (tid < CORE / 32) {
            *(uint4*)(core + tid * 32) = make_uint4(y[0][0], y[0][1], y[0][2], y[0][3]);
            *(uint4*)(core + tid * 32 + 16) = make_uint4(y[1][0], y[1][1], y[1][2], y[1][3]);
        } else {
            uint32_t* v = var + base + (tid - CORE / 32) * 8;
#pragma unroll
            for (int h = 0; h < 2; h++)
#pragma unroll
                for (int q = 0; q < 4; q++) v[4 * h + q] = y[h][q];
        }
        return;
    }
    uint32_t d1[2] = {0, 0}, sw[2][2] = {{0, 0}, {0, 0}};
    uint32_t c1 = 0, c2 = 0, c3 = 0;
#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
        for (int q = 0; q < 4; q++)
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const int j = 4 * q + k;
                const uint32_t b = (y[h][q] >> (8 * k)) & 0xFF, r = s_rank[(b >> 3) & 15];
                d1[h] |= (r < 3 ? r : 3u) << f8_d1_bit(j);
                sw[h][j >> 3] |= ((b >> 7) << f8_s_bit(j)) | ((b & 7) << f8_m_bit(j));
                c1 += r >= 3, c2 += r >= 6, c3 += r >= 13;
            }
    const uint32_t A = f8_excl(c1 + (c2 << 16), s_tot[0]);  // (the ordinals of the values this thread's are among those that reach each level)
    const uint32_t B = f8_excl(c3, s_tot[1]);
    uint32_t k1 = A & 0xFFFF, k2 = A >> 16, k3 = B;
#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
        for (int q = 0; q < 4; q++)
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const uint32_t e = (y[h][q] >> (8 * k + 3)) & 15, r = s_rank[e];
                if (r >= 3) {
                    atomicOr(&s_var[k1 >> 4], (r < 6 ? r - 3 : 3u) << (2 * (k1 & 15)));
                    k1++;
                    if (r >= 6) {
                        const uint32_t d3 = r < 13 ? r - 6 : 7u, p = 32 * w2 + 3 * k2;
                        atomicOr(&s_var[p >> 5], d3 << (p & 31));
                        if ((p & 31) > 29) atomicOr(&s_var[(p >> 5) + 1], d3 >> (32 - (p & 31)));
                        k2++;
                        if (r >= 13) {
                            atomicOr(&s_var[w2 + w3 + (k3 >> 3)], e << (4 * (k3 & 7)));
                            k3++;
                        }
                    }
                }
            }
    // the core: per unit 32 words of digits, then each lane's 2 words of signs and mantissas
#pragma unroll
    for (int h = 0; h < 2; h++) {
        const int u = 4 * mb + 2 * hh + h;
        *(uint32_t*)(core + u * 384 + l * 4) = d1[h];
        *(uint2*)(core + u * 384 + 128 + l * 8) = make_uint2(sw[h][0], sw[h][1]);
    }
    __syncthreads();
    for (uint32_t i = tid; i < words; i += NT) var[base + i] = s_var[i];
}

// ---- the decode of a lane of a unit (the unpack's and the fused product's) ----

// The patch's exponents as bytes (exponent e: the byte e << 3): ranks 0-2 and a 0 (lo), 3-5 and the mark 0x80 (l2), 6-9 (l3a), 10-12 and the mark (l3b).
struct F8Tab {
    uint32_t lo, l2, l3a, l3b;
};
__device__ __forceinline__ uint32_t f8_nib4(uint32_t v) {  // v's low 4 nibbles as 4 bytes, each nibble << 3
    uint32_t t = f8_prmt(v, 0, 0x4140);
    t = (t | (t << 4)) & 0x0F0F0F0Fu;
    return t << 3;
}
__device__ __forceinline__ F8Tab f8_tab(uint32_t h1, uint32_t h2) {
    const uint32_t a = f8_nib4(h1), b = f8_nib4(h1 >> 16), c = f8_nib4(h2), d = f8_nib4(h2 >> 16);
    return F8Tab{a & 0x00FFFFFFu, (f8_prmt(a, b, 0x0543) & 0x00FFFFFFu) | 0x80000000u, f8_prmt(b, c, 0x5432), (f8_prmt(c, d, 0x0432) & 0x00FFFFFFu) | 0x80000000u};
}
// 8 digits of 2 bits (v's low 16 bits) as 8 nibbles; 8 digits of 3 bits (v's low 24 bits) as 8 nibbles.
__device__ __forceinline__ uint32_t f8_spread2(uint32_t v) {
    uint32_t y = f8_prmt(v, 0, 0x4140);
    y = (y | (y << 4)) & 0x0F0F0F0Fu;
    return (y | (y << 2)) & 0x33333333u;
}
__device__ __forceinline__ uint32_t f8_spread3(uint32_t v) {
    uint32_t y = (v & 0xFFFu) | ((v << 4) & 0x0FFF0000u);
    y = (y & 0x003F003Fu) | ((y << 2) & 0x3F003F00u);
    return (y & 0x07070707u) | ((y << 1) & 0x70707070u);
}

// The 16 values of a lane of a unit as 4 registers (value 4 q + k in byte k of o[q]: the A operand of mma.m16n8k32 with e4m3) from its d1 word dw, its sign
// and mantissa words s0 and s1, its level-2 digits from its first (b2: 32 bits), its level-3 digits from its first (b3lo, b3hi: 64 bits), and its literals,
// lit(k4) its first (lit(k): the patch's literal k, read only where the lane has one). c1, c2, c3: the digits of levels 2 and 3 and the literals it took.
// Halves a and b: values 0-7 and 8-15. Every step is a few byte permutes and masks a register, whatever the digits (no loop but the literals').
template <class Lit>
__device__ __forceinline__ void f8_lane_unit(const F8Tab& T, uint32_t dw, uint32_t s0, uint32_t s1, uint32_t b2, uint32_t b3lo, uint32_t b3hi, Lit lit, uint32_t k4,
                                             uint32_t (&o)[4], uint32_t& c1, uint32_t& c2, uint32_t& c3) {
    const uint32_t Wa = dw & 0x33333333u, Wb = (dw >> 2) & 0x33333333u;            // the d1 digits, a nibble a value
    const uint32_t Ta = Wa & (Wa >> 1) & 0x11111111u, Tb = Wb & (Wb >> 1) & 0x11111111u;  // the values that spill (digit 3)
    const uint32_t ca = (Ta * 0x11111111u) >> 28, cb = (Tb * 0x11111111u) >> 28;      // how many a half
    const uint32_t xa = f8_spread2(b2), xb = f8_spread2(b2 >> (2 * ca));             // a half's level-2 digits, in the order of its spills
    const uint32_t Ma = xa & (xa >> 1) & 0x11111111u & f8_nibs(ca), Mb = xb & (xb >> 1) & 0x11111111u & f8_nibs(cb);  // the spills that reach level 3
    const uint32_t ma = (Ma * 0x11111111u) >> 28, mb = (Mb * 0x11111111u) >> 28;
    uint32_t sp[4] = {f8_prmt(T.l2, 0, xa), f8_prmt(T.l2, 0, xa >> 16), f8_prmt(T.l2, 0, xb), f8_prmt(T.l2, 0, xb >> 16)};  // the spills' exponents
    c1 = ca + cb;
    c2 = ma + mb;
    c3 = 0;
    if (c2) {  // level 3: the marked spills' exponents (marks 0-7 of the unit from one window, half b's from ma on; past 8, half b's from its own), then each into its spill's place
        const uint32_t y = f8_spread3(b3lo);
        uint32_t L[4], ob = ma;
        L[0] = L[2] = f8_prmt(T.l3a, T.l3b, y), L[1] = L[3] = f8_prmt(T.l3a, T.l3b, y >> 16);
        uint32_t esc = y & (y >> 1) & (y >> 2) & 0x11111111u & f8_nibs(c2 < 8 ? c2 : 8);  // (the marks whose digit is 7, the literals' places)
        if (c2 > 8) {  // (rare)
            const uint32_t yb = f8_spread3(__funnelshift_r(b3lo, b3hi, 3 * ma));
            L[2] = f8_prmt(T.l3a, T.l3b, yb), L[3] = f8_prmt(T.l3a, T.l3b, yb >> 16), ob = 0;
            esc = (y & (y >> 1) & (y >> 2) & 0x11111111u & f8_nibs(ma)) | (yb & (yb >> 1) & (yb >> 2) & 0x11111111u & f8_nibs(mb));
        }
        if (esc) {  // the literals (rare): in order, into the marks' bytes
            const int n[4] = {(int)(ob ? c2 : ma), (int)(ob ? c2 : ma) - 4, ob ? 0 : (int)mb, ob ? 0 : (int)mb - 4};
#pragma unroll
            for (int i = 0; i < 4; i++) {
                uint32_t e = L[i] & f8_bytes(n[i]) & 0x80808080u;
                while (e) {
                    const int bit = __ffs((int)e) - 1;  // 8 byte + 7
                    e &= e - 1;
                    L[i] = (L[i] & ~(0xFFu << (bit - 7))) | (lit(k4 + c3) << (bit - 4));
                    c3++;
                }
            }
            if (ob) L[2] = L[0], L[3] = L[1];
        }
        const uint32_t Ra = Ma * 0x11111110u, Rb = Mb * 0x11111110u + ob * 0x11111111u;  // a spill's marked spills before it (half b: after half a's, in one window)
        sp[0] = f8_prmt(sp[0], f8_prmt(L[0], L[1], Ra), 0x3210 + (Ma << 2));
        sp[1] = f8_prmt(sp[1], f8_prmt(L[0], L[1], Ra >> 16), 0x3210 + (Ma >> 14));
        sp[2] = f8_prmt(sp[2], f8_prmt(L[2], L[3], Rb), 0x3210 + (Mb << 2));
        sp[3] = f8_prmt(sp[3], f8_prmt(L[2], L[3], Rb >> 16), 0x3210 + (Mb >> 14));
    }
    const uint32_t Ra = Ta * 0x11111110u, Rb = Tb * 0x11111110u;  // a value's spills before it in its half
    o[0] = (s0 & 0x87878787u) | f8_prmt(f8_prmt(T.lo, 0, Wa), f8_prmt(sp[0], sp[1], Ra), 0x3210 + (Ta << 2));
    o[1] = (__funnelshift_l(s0, s0, 4) & 0x87878787u) | f8_prmt(f8_prmt(T.lo, 0, Wa >> 16), f8_prmt(sp[0], sp[1], Ra >> 16), 0x3210 + (Ta >> 14));
    o[2] = (s1 & 0x87878787u) | f8_prmt(f8_prmt(T.lo, 0, Wb), f8_prmt(sp[2], sp[3], Rb), 0x3210 + (Tb << 2));
    o[3] = (__funnelshift_l(s1, s1, 4) & 0x87878787u) | f8_prmt(f8_prmt(T.lo, 0, Wb >> 16), f8_prmt(sp[2], sp[3], Rb >> 16), 0x3210 + (Tb >> 14));
}

// The unpack: each patch, or (ids) each patch of the stacked matrices ids[blockIdx.y], into out. Thread tid as the pack's: two lanes of units, its 32 stream bytes.
__global__ void __launch_bounds__(NT) f8_unpack_kernel(const F8Un P) {
    __shared__ __align__(16) uint32_t s_tile[VALUES / 4];
    __shared__ uint32_t s_var[TAIL + 8];
    __shared__ uint32_t s_tot[3][NT / 32];
    const int tid = threadIdx.x, mb = tid >> 6, l = (tid >> 1) & 31, hh = tid & 1, g = l >> 2, t = l & 3;
    const int pl = blockIdx.x;
    int64_t pat = pl, item = 0;
    if (P.ids) {
        item = blockIdx.y;
        const int id = P.ids[item];
        if ((unsigned)id >= (unsigned)P.E) return;  // (no such matrix: its place in out is left as it is)
        pat += (int64_t)id * P.per;
    }
    const uint4 hd = __ldg((const uint4*)(P.heads + pat * HEAD));
    const uint32_t base = hd.x, h1 = hd.y, h2 = hd.z, h3 = hd.w;
    const uint8_t* core = P.core + pat * CORE;
    uint32_t o[2][4];  // unit h (ks = 2 hh + h): value 4 q + k at byte k of o[h][q]
    if (h3 >> 31) {  // a plain patch: its stream's bytes
        uint32_t x[8];
        if (tid < CORE / 32) {
            const uint4 a = f8_ld16(core + tid * 32), b = f8_ld16(core + tid * 32 + 16);
            x[0] = a.x, x[1] = a.y, x[2] = a.z, x[3] = a.w, x[4] = b.x, x[5] = b.y, x[6] = b.z, x[7] = b.w;
        } else {
            const uint32_t* v = P.var + base + (tid - CORE / 32) * 8;
#pragma unroll
            for (int i = 0; i < 8; i++) x[i] = __ldg(v + i);
        }
#pragma unroll
        for (int i = 0; i < 8; i++) o[i >> 2][i & 3] = x[i];
    } else {
        const uint32_t n1 = h3 & 0x3FFF, n2 = (h3 >> 14) & 0x3FFF, n3 = h2 >> 20;
        const uint32_t w2 = (n1 + 15) >> 4, w3 = (3 * n2 + 31) >> 5, words = w2 + w3 + ((n3 + 7) >> 3);
        uint32_t dw[2];
        uint2 sw[2];
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int u = 4 * mb + 2 * hh + h;
            dw[h] = __ldg((const uint32_t*)(core + u * 384 + l * 4));
            sw[h] = f8_ld8(core + u * 384 + 128 + l * 8);
        }
        for (uint32_t i = tid; i < words; i += NT) s_var[i] = __ldg(P.var + base + i);
        if (tid < 8) s_var[words + tid] = 0;  // (a read of the digits runs a word or two past them)
        const F8Tab T = f8_tab(h1, h2);
        const uint32_t c1a = __popc(f8_threes(dw[0])), c1b = __popc(f8_threes(dw[1]));
        const uint32_t K1 = f8_excl(c1a + c1b, s_tot[0]);  // (its __syncthreads has s_var in)
        const uint32_t b2[2] = {f8_digits(s_var, K1), f8_digits(s_var, K1 + c1a)};
        const uint32_t c2a = __popc(f8_threes(b2[0]) & f8_low2(c1a)), c2b = __popc(f8_threes(b2[1]) & f8_low2(c1b));
        const uint32_t K2 = f8_excl(c2a + c2b, s_tot[1]);
        const uint32_t p3[2] = {32 * w2 + 3 * K2, 32 * w2 + 3 * (K2 + c2a)};
        uint32_t b3[2][2];
#pragma unroll
        for (int h = 0; h < 2; h++) b3[h][0] = f8_bits(s_var, p3[h]), b3[h][1] = f8_bits(s_var, p3[h] + 32);
        const uint32_t c3a = f8_sevens(b3[0][0], b3[0][1], c2a), c3b = f8_sevens(b3[1][0], b3[1][1], c2b);
        const uint32_t K3 = f8_excl(c3a + c3b, s_tot[2]);
        const uint32_t* sl = s_var + w2 + w3;
        auto lit = [sl](uint32_t k) { return (sl[k >> 3] >> (4 * (k & 7))) & 15u; };
        uint32_t c1, c2, c3;
        f8_lane_unit(T, dw[0], sw[0].x, sw[0].y, b2[0], b3[0][0], b3[0][1], lit, K3, o[0], c1, c2, c3);
        f8_lane_unit(T, dw[1], sw[1].x, sw[1].y, b2[1], b3[1][0], b3[1][1], lit, K3 + c3a, o[1], c1, c2, c3);
    }
    // the patch in rows, through shared memory, to out
#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
        for (int q = 0; q < 4; q++) s_tile[f8_word(16 * mb + g + 8 * (q & 1), 8 * (2 * hh + h) + t + 4 * (q >> 1))] = o[h][q];
    __syncthreads();
    const int row0 = (pl / P.KT) * PR, col0 = (pl % P.KT) * PC;
    uint8_t* dst = P.out + item * P.item + (int64_t)row0 * P.ld + col0;
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int c = tid + i * NT, row = c >> 3, g8 = c & 7;
        if (row0 + row >= P.O) continue;
        const uint4 v = *(const uint4*)&s_tile[row * 32 + (((g8 ^ row) & 7) << 2)];
        uint8_t* d = dst + (int64_t)row * P.ld + g8 * 16;
        if (P.vec && col0 + g8 * 16 + 16 <= P.K) {
            F8_ALIGN(d, 16);
            *(uint4*)d = v;
        } else {
            const uint32_t b[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int k = 0; k < 16; k++)
                if (col0 + g8 * 16 + k < P.K) d[k] = (uint8_t)(b[k >> 2] >> (8 * (k & 3)));
        }
    }
}

// ---- the fused product: Y [M, N] = X [M, K] W^T for 1 to 64 tokens, W [N, K] in this layout and its scales, W never written out ----
// The weights' bytes go into the tensor cores' A operand as bf16 (W8A16, as vLLM's Marlin FP8: an e4m3 byte's sign, exponent and mantissa in a bf16's
// places, 2^-120 times its value exactly; X bf16, times 2^64 (exact); products and sums in fp32; the scales and 2^56 applied to the sums). A warp takes a row block
// (16 rows) of W, a CTA 8 of them (a row group of 128 rows); a row block of a patch is a "granule": its head, its 4 units' core words and its variable
// part, decoded by f8_lane_unit into 16 registers a lane, then 2 m16n8k16 products a unit and 8 tokens. The work is the row groups' patches in order
// (row group by row group, each across K), cut into as many equal parts as CTAs are resident at once: one wave, no CTA idle at its end. A CTA steps
// through its part a patch at a time; at each step it copies (cp.async, never waited for at once) what the next step reads: X's 128 columns of the
// next patch, each warp's core and variable part of its next granule, and the heads two patches on (which say where the variable parts are). Where a
// row group's K is cut between CTAs, each leaves its sums in parts and the last to finish adds them in the CTAs' order (the same result every run).

#ifndef F8_EXTERN_SMEM
#define F8_EXTERN_SMEM(T, name) extern __shared__ __align__(16) T name[]
#endif

constexpr int F8_BLOCK = 0, F8_TENSOR = 1, F8_MX = 2, F8_DS32 = 3, F8_CHANNEL = 4;  // the scales: fp32 per 128 x 128, one fp32, e8m0 per 1 x 32, e8m0 per 32 x 32,
                                                                       // fp32 a row
constexpr int F8_A8 = 8;  // KIND | F8_A8 (block, tensor, channel): W8A8, X as e4m3 a token and 128 columns (f8_xq_kernel's) into mma.m16n8k32 e4m3
constexpr int LW = 8, LNT = 32 * LW;  // the product's warps a CTA (a row block each), threads
constexpr int CSL = 384;              // words of a row block's core (4 units of 384 bytes): a slot of the warp's two
constexpr int VB = 80;                // words of a slot of the warp's two for a row block's variable part; a longer part is read where it is in var
constexpr int HSL = 3, XSL = 2;       // the CTA's slots of heads (its two rows of patches at a column) and of X (a patch's 128 columns of the tokens)
constexpr int XROW = PC * 2;          // bytes a token's row of an X slot (its 16-byte pieces swizzled by the row's parity: rows g and g + 1 in other banks)
constexpr int LSMEM = 4 * (HSL * 16 + LW * (2 * CSL + 2 * VB));
// (S slots of core, variable parts and X a warp or CTA, heads 2 S - 3 steps ahead: S = 2 as above; 3, one token on an HBM GPU, GLYD_F8_SLOTS=3: two steps' copies in flight, not one)
template <int S> constexpr int f8_hs() { return S == 2 ? HSL : 2 * S - 1; }
template <int S> constexpr int f8_xs() { return S == 2 ? XSL : S; }
template <int S, bool H = false> constexpr int f8_csl() { return H ? 32 * 14 : CSL; }  // (H: layout H's row block, HCSL)
template <int S, bool H = false> constexpr int f8_lsmem() { return 4 * (f8_hs<S>() * 16 + LW * (S * f8_csl<S, H>() + S * VB)); }  // bytes of shared memory before the X slots (XSL slots of M rows of XROW bytes); DENSE: before
                                                                  // the warps' buffers of exponents (LW of EBS bytes), the X slots after them
// Bytes of an X slot of M tokens: bf16 rows of XROW bytes; A8: e4m3 rows of 128 bytes (a row's groups of 32 columns at group ^ (row & 3)), then the M scales.
__host__ __device__ constexpr int f8_xslot(int M, bool a8) { return a8 ? (M * (PC + 4) + 15) & ~15 : M * XROW; }

struct F8Lin {
    const uint8_t* core;
    const uint32_t* heads;
    const uint32_t* var;
    const void* scale;  // F8_BLOCK: float [ceil(N / 128)] rows lds apart; F8_TENSOR: float [1]; F8_MX: uint8 [N] rows lds apart; F8_DS32: uint8 [ceil(N / 32)] rows lds apart;
                        // F8_CHANNEL: float [N]
    const uint16_t* x;  // bf16 [M, K], rows ldx apart
    uint16_t* y;        // bf16 [M, N], rows ldy apart
    const uint8_t* xq;  // F8_A8: X as e4m3 [M][KT 128] in the B fragments' order, and its scales sx [M][KT] (f8_xq_kernel's)
    const float* sx;
    float* parts;       // ctas > 1: [2 ctas][M][128], the sums of a CTA's first and last row group where it holds part of its K
    int* done;          // ctas > 1: a counter a row group of 128 rows, zero (the last CTA leaves it zero)
    int64_t lds, ldx, ldy;
    int M, N, K, KT, RG, ctas, xvec;  // KT: patches across K; RG: row groups (of 128 rows); ctas: the grid; xvec: X's rows 16-byte aligned
    // The experts' product (glyd_gpu_f8_moe; null plan: one matrix): the stack of E matrices [N, K] as one [E N, K] (block scales [E][ceil(N / 128)] rows lds apart), blockIdx.y the
    // plan's slot (moe_route_kernel's: its expert, its pairs' rows of X and Y, M of them at most); gather: X's row of a pair is its token's (order[m] / gk), else the pair's own in the
    // plan's order. f8_slot sets order, and each pointer at the slot's expert and rows.
    const int* plan;
    const int* order;
    int E, gk, gather;
};

// cp.async of 4, 8 and 16 bytes (global to shared, not waited for: the 16 bytes past L1), a commit of the thread's copies so far, a wait for all of them
// (the emulation: copies at once).
__device__ __forceinline__ void f8_cp4(uint32_t* dst, const uint32_t* src) {
#if defined(__CUDA_ARCH__)
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src));
#else
    *dst = *src;
#endif
}
__device__ __forceinline__ void f8_cp8(uint32_t* dst, const uint32_t* src) {
#if defined(__CUDA_ARCH__)
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src));
#else
    dst[0] = src[0], dst[1] = src[1];
#endif
}
__device__ __forceinline__ void f8_cp16(void* dst, const void* src) {
#if defined(__CUDA_ARCH__)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src));
#else
    F8_ALIGN(dst, 16);
    F8_ALIGN(src, 16);
    *(uint4*)dst = *(const uint4*)src;
#endif
}
__device__ __forceinline__ void f8_cp_commit() {
#if defined(__CUDA_ARCH__)
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
__device__ __forceinline__ void f8_cp_wait0() {
#if defined(__CUDA_ARCH__)
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void f8_cp_wait1() {  // all but the newest group in
#if defined(__CUDA_ARCH__)
    asm volatile("cp.async.wait_group 1;\n" ::: "memory");
#endif
}
template <int N>
__device__ __forceinline__ void f8_cp_wait() {
#if defined(__CUDA_ARCH__)
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
#endif
}

// e4m3 bytes 0 and 1 (lo) or 2 and 3 (hi) of o as two bf16 (low half: the first), each 2^-120 times the byte's value: its sign, exponent and mantissa
// in a bf16's places (the byte and its sign spread to 16 bits by prmt, a shift and a mask).
__device__ __forceinline__ uint32_t f8_bf_lo(uint32_t o) { return (f8_prmt(o, 0, 0x9180) << 4) & 0x87F087F0u; }
__device__ __forceinline__ uint32_t f8_bf_hi(uint32_t o) { return (f8_prmt(o, 0, 0xB3A2) << 4) & 0x87F087F0u; }
// Two bf16 of X times 2^64, exactly (for |x| below 2^64): with the weights' 2^-120 the products are 2^-56 times the true ones, in fp32's normal range for
// |x| down to 2^-64 (at 2^-120 alone, activations of about 2^-16 and below gave products in fp32's subnormal range, where the sums lose bits).
__device__ __forceinline__ uint32_t f8_x64(uint32_t v) {
#if defined(__CUDA_ARCH__)
    uint32_t r;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(r) : "r"(v), "r"(0x5F805F80u), "r"(0x80008000u));  // (v 2^64 + -0)
    return r;
#else
    const auto one = [](uint32_t h) { return __float_as_uint(__int_as_float((h & 0xFFFFu) << 16) * 0x1p64f) >> 16; };
    return one(v) | (one(v >> 16) << 16);
#endif
}

// (mma.sync is warp-collective and these asm statements are not volatile, so the compiler may move or merge them: every guard around a call must stay
// uniform across the warp, as each is.)
__device__ __forceinline__ void f8_mma(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__)
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#elif defined(F8_EMU)
    f8_emu_mma(c, a0, a1, a2, a3, b0, b1);
#elif defined(__HIP_DEVICE_COMPILE__)
    const uint32_t a[4] = {a0, a1, a2, a3};
    glyd_mma16816(c, a, b0, b1);
#endif
}

// mma.sync.m16n8k32 with e4m3 A and B (Ada, Hopper and Blackwell): A the decoded registers as they are.
__device__ __forceinline__ void f8_mma8(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#elif defined(F8_EMU)
    f8_emu_mma8(c, a0, a1, a2, a3, b0, b1);
#elif defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    __trap();  // (no e4m3 MMA before compute capability 8.9, nor on AMD: the host never asks for it there, nor where the code loaded is older)
#endif
}
// Two floats as e4m3 bytes (lo: the first), rounded to nearest even, past 448 saturated to it.
__device__ __forceinline__ uint32_t f8_e4m3x2(float lo, float hi) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    uint16_t r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
    return r;
#elif defined(F8_EMU)
    return f8_emu_e4m3(lo) | (f8_emu_e4m3(hi) << 8);
#else
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    __trap();
#endif
    return 0;
#endif
}
// W8A8's X: each token's 128 columns of K (zeros past K, to KT 128) as e4m3 of 448 over their largest magnitude, the bytes of a group of 32 columns in the
// order the B fragments want (4-column pieces i at 2 (i & 3) + (i >> 2)), into xq [M][KT 128]; the scale back (the largest over 448) into sx [M][KT]. A warp a
// token's 128 columns (a lane 4).
__global__ void __launch_bounds__(128) f8_xq_kernel(const uint16_t* x, int64_t ldx, int M, int K, int KT, uint8_t* xq, float* sx) {
    const int w = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (w >= M * KT) return;
    const int m = w / KT, kb = w - m * KT, k = kb * PC + 4 * lane;
    float f[4], mx = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) f[i] = k + i < K ? __int_as_float((uint32_t)x[(int64_t)m * ldx + k + i] << 16) : 0.f, mx = fmaxf(mx, fabsf(f[i]));
#pragma unroll
    for (int d = 16; d >= 1; d >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, d));
    const float inv = mx > 0.f ? 448.f / mx : 0.f;
    const int pc = lane & 7;
    *(uint32_t*)(xq + ((int64_t)m * KT + kb) * PC + (lane >> 3) * 32 + 4 * (2 * (pc & 3) + (pc >> 2))) = f8_e4m3x2(f[0] * inv, f[1] * inv) | (f8_e4m3x2(f[2] * inv, f[3] * inv) << 16);
    if (lane == 0) sx[(int64_t)m * KT + kb] = mx / 448.f;
}

// A float to bf16 bits, rounded to the nearest even (a NaN stays a NaN).
__device__ __forceinline__ uint16_t f8_to_bf16(float v) {
    uint32_t u = __float_as_uint(v);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) return (uint16_t)((u >> 16) | 0x40u);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

// The sum of v over the warp's lanes before this one.
__device__ __forceinline__ uint32_t f8_wexcl(uint32_t v, int lane) {
    uint32_t x = v;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const uint32_t y = __shfl_up_sync(0xffffffffu, x, d);
        if (lane >= d) x += y;
    }
    return x - v;
}

// An e8m0 scale byte as the float 2^(E - 127).
__device__ __forceinline__ float f8_e8m0(uint32_t e) { return __int_as_float(e ? e << 23 : 0x00400000u); }

// Row block mb's place in its patch's variable part (head hA, hB): its first ordinals at levels 2, 3 and the literals (o2, o3, o4), whether it has
// literals (lits), the words of each run it starts at (a2, a3, a4: word w2 + a3 of the patch's part is its level 3's first, w2 + w3 + a4 its literals'),
// and where they are in a copy of it (level 2 at 0, level 3 at v3, the literals at v4; all: its words, with 2 after the level-2 and level-3 runs: a lane's
// reads may touch 2 words past a run, and 3 past level 3's (its second window) where the lane is past the row block's last mark, a value then unused).
struct F8Geo {
    uint32_t w2, w3, o2, o3, o4, lits, a2, a3, a4, v3, v4, all, n1b;  // (n1b: the row block's values that reach level 2)
};
__device__ __forceinline__ F8Geo f8_geo(const uint4& hA, const uint4& hB, int mb) {
    F8Geo G;
    const uint32_t n1 = hA.w & 0x3FFF, n2 = (hA.w >> 14) & 0x3FFF, n3 = hA.z >> 20;
    uint32_t e2, e3, e4;
    G.w2 = (n1 + 15) >> 4, G.w3 = (3 * n2 + 31) >> 5, G.o2 = G.o3 = G.o4 = 0;
    if (mb == 0) e2 = hB.x & 0xFFFF, e3 = hB.y >> 16, e4 = hB.w & 0x3FF;
    else if (mb == 1) G.o2 = hB.x & 0xFFFF, G.o3 = hB.y >> 16, G.o4 = hB.w & 0x3FF, e2 = hB.x >> 16, e3 = hB.z & 0xFFFF, e4 = (hB.w >> 10) & 0x3FF;
    else if (mb == 2) G.o2 = hB.x >> 16, G.o3 = hB.z & 0xFFFF, G.o4 = (hB.w >> 10) & 0x3FF, e2 = hB.y & 0xFFFF, e3 = hB.z >> 16, e4 = hB.w >> 20;
    else G.o2 = hB.y & 0xFFFF, G.o3 = hB.z >> 16, G.o4 = hB.w >> 20, e2 = n1, e3 = n2, e4 = n3;
    const uint32_t z2 = (e2 + 15) >> 4, z3 = (3 * e3 + 31) >> 5, z4 = (e4 + 7) >> 3;
    G.lits = e4 != G.o4, G.n1b = e2 - G.o2;
    G.a2 = G.o2 >> 4, G.a3 = (3 * G.o3) >> 5, G.a4 = G.o4 >> 3;
    G.v3 = z2 - G.a2 + 2, G.v4 = G.v3 + z3 - G.a3 + 2, G.all = G.v4 + z4 - G.a4;
    return G;
}

// A warp's copies of row block mb of patch p for its decode a step or two later: its core (1536 bytes, 3 cp.async of 16 bytes a lane; whatever the
// patch: a plain one's are not read) into cw, and (its head hA, hB in) its variable part, where it fits, into vw (a plain patch's: read at the decode).
__device__ __forceinline__ void f8_core_in(const F8Lin& P, int64_t p, int mb, int lane, uint32_t* cw) {
    const uint4* cs = (const uint4*)(P.core + p * CORE + mb * 4 * 384);
#pragma unroll
    for (int i = 0; i < 3; i++) f8_cp16(cw + 4 * (lane + 32 * i), cs + lane + 32 * i);
}
__device__ __forceinline__ void f8_var_in(const F8Lin& P, int mb, int lane, const uint4& hA, const uint4& hB, uint32_t* vw) {
    if (hA.w >> 31) return;
    const F8Geo G = f8_geo(hA, hB, mb);
    if (G.all > (uint32_t)VB) return;
    const uint32_t* src = P.var + hA.x;
    for (uint32_t i = lane; i < G.all; i += 32) f8_cp4(vw + i, src + (i < G.v3 ? G.a2 + i : i < G.v4 ? G.w2 + G.a3 + (i - G.v3) : G.w2 + G.w3 + G.a4 + (i - G.v4)));
}

// Row block mb of patch p (16 rows x 128 columns: its units 4 mb + ks) into a lane's 16 registers o[ks][q], from its head (hA, hB) and its copies
// (f8_core_in's and f8_var_in's, in: cw its core, vw its variable part where it fit, else var itself is read).
__device__ __forceinline__ void f8_granule(const F8Lin& P, int64_t p, int mb, int lane, const uint4& hA, const uint4& hB, const uint32_t* cw, const uint32_t* vw,
                                           uint32_t (&o)[4][4]) {
    if (hA.w >> 31) {  // plain: the lane's 64 stream bytes (row blocks 0-2 in core, 16-byte aligned; 3 in var, at any word: read by words)
        if (mb < 3) {
            const uint4* s = (const uint4*)(P.core + p * CORE + 2048 * mb + 64 * lane);
#pragma unroll
            for (int ks = 0; ks < 4; ks++) {
                const uint4 v = __ldg(s + ks);
                o[ks][0] = v.x, o[ks][1] = v.y, o[ks][2] = v.z, o[ks][3] = v.w;
            }
        } else {
            const uint32_t* s = P.var + hA.x + 16 * lane;
#pragma unroll
            for (int i = 0; i < 16; i++) o[i >> 2][i & 3] = __ldg(s + i);
        }
        return;
    }
    const F8Geo G = f8_geo(hA, hB, mb);
    const bool in = G.all <= (uint32_t)VB;
    const uint32_t* src = P.var + hA.x;
    const uint32_t* p2 = in ? vw : src + G.a2;                // level 2's digits, from the row block's first word of them
    const uint32_t* p3 = in ? vw + G.v3 : src + G.w2 + G.a3;  // level 3's
    const uint32_t* p4 = in ? vw + G.v4 : src + G.w2 + G.w3 + G.a4;  // the literals'
    uint32_t dw[4], c1[4], t1 = 0;
    uint2 sw[4];
#pragma unroll
    for (int ks = 0; ks < 4; ks++) {
        dw[ks] = cw[ks * 96 + lane];
        sw[ks] = ((const uint2*)(cw + ks * 96 + 32))[lane];
        c1[ks] = __popc(f8_threes(dw[ks]));
        t1 += c1[ks];
    }
    const F8Tab T = f8_tab(hA.y, hA.z);
    uint32_t k1 = G.o2 - 16 * G.a2 + f8_wexcl(t1, lane);  // the lane's first level-2 digit from p2
    uint32_t b2[4], m[4], t2 = 0;
#pragma unroll
    for (int ks = 0; ks < 4; ks++) {
        b2[ks] = f8_digits(p2, k1);
        k1 += c1[ks];
        m[ks] = __popc(f8_threes(b2[ks]) & f8_low2(c1[ks]));
        t2 += m[ks];
    }
    uint32_t q3 = 3 * (G.o3 + f8_wexcl(t2, lane)) - 32 * G.a3;  // the lane's first level-3 digit's bit from p3
    uint32_t b3[4][2], c3[4] = {0, 0, 0, 0};
#pragma unroll
    for (int ks = 0; ks < 4; ks++) {
        b3[ks][0] = f8_bits(p3, q3), b3[ks][1] = f8_bits(p3, q3 + 32);
        q3 += 3 * m[ks];
    }
    uint32_t k4 = 0;
    if (G.lits) {  // (literals are rare: their counts and scan only where the row block has one)
        uint32_t t3 = 0;
#pragma unroll
        for (int ks = 0; ks < 4; ks++) t3 += c3[ks] = f8_sevens(b3[ks][0], b3[ks][1], m[ks]);
        k4 = G.o4 - 8 * G.a4 + f8_wexcl(t3, lane);
    }
    auto lit = [p4](uint32_t k) { return (p4[k >> 3] >> (4 * (k & 7))) & 15u; };
#pragma unroll
    for (int ks = 0; ks < 4; ks++) {
        uint32_t x1, x2, x3;
        f8_lane_unit(T, dw[ks], sw[ks].x, sw[ks].y, b2[ks], b3[ks][0], b3[ks][1], lit, k4, o[ks], x1, x2, x3);
        k4 += c3[ks];
    }
}

// The same granule decoded in two passes (DENSE): first the row block's spills' exponents in stream order, a lane 24 of them (their level-2 digits by a
// byte permute, the level-3 and literal ones of its marks after a warp scan of the marks), into the warp's buffer eb; then each lane's units from them
// (its spills' bytes read from eb at its first spill, then put in place by byte permutes, as f8_lane_unit does). A row block with more spills than eb
// holds, or a lane with more than 8 marks (both rare), and a plain patch: f8_granule.
// between(ks), once unit ks is in o (AHEAD: the products of the granule before, one unit at a time, among the decode's instructions).
constexpr int ES = 24, EB = 32 * ES, EBS = EB + 16;  // spills a lane of the first pass, bytes of a warp's buffer, its stride (the reads past its end land in it)
template <class Between>
__device__ __forceinline__ void f8_granule_dense(const F8Lin& P, int64_t p, int mb, int lane, const uint4& hA, const uint4& hB, const uint32_t* cw,
                                                 const uint32_t* vw, uint32_t* eb, uint32_t (&o)[4][4], Between&& between) {
    auto whole = [&]() {  // (f8_granule's decode)
        f8_granule(P, p, mb, lane, hA, hB, cw, vw, o);
#pragma unroll
        for (int ks = 0; ks < 4; ks++) between(ks);
    };
    if (hA.w >> 31) return whole();
    const F8Geo G = f8_geo(hA, hB, mb);
    if (G.n1b > (uint32_t)EB) return whole();
    const bool in = G.all <= (uint32_t)VB;
    const uint32_t* src = P.var + hA.x;
    const uint32_t* p2 = in ? vw : src + G.a2;
    const uint32_t* p3 = in ? vw + G.v3 : src + G.w2 + G.a3;
    const uint32_t* p4 = in ? vw + G.v4 : src + G.w2 + G.w3 + G.a4;
    const F8Tab T = f8_tab(hA.y, hA.z);
    // pass 1: spills ES lane .. ES lane + v - 1 of the row block
    const uint32_t k0 = ES * lane, v = k0 < G.n1b ? min(G.n1b - k0, (uint32_t)ES) : 0u;
    const uint32_t kd = G.o2 - 16 * G.a2 + (v ? k0 : 0u);  // (its first level-2 digit from p2; a lane with none reads the row block's first)
    const uint32_t* q = p2 + (kd >> 4);
    const uint32_t q0 = q[0], q1 = q[1], q2 = q[2], sh = 2 * (kd & 15);
    const uint32_t lo = __funnelshift_r(q0, q1, sh), hi = __funnelshift_r(q1, q2, sh);
    const uint32_t x0 = f8_spread2(lo), x1 = f8_spread2(lo >> 16), x2 = f8_spread2(hi);  // its 24 digits, a nibble each
    const uint32_t m0 = x0 & (x0 >> 1) & 0x11111111u & f8_nibs(min(v, 8u)), m1 = x1 & (x1 >> 1) & 0x11111111u & f8_nibs(v > 8 ? min(v - 8, 8u) : 0u),
                   m2 = x2 & (x2 >> 1) & 0x11111111u & f8_nibs(v > 16 ? v - 16 : 0u);  // its marks (digit 3)
    const uint32_t cm = __popc(m0) + __popc(m1) + __popc(m2);
    if (__any_sync(0xffffffffu, cm > 8)) return whole();
    const uint32_t c1r = __popc(m0 & 0xFFFFu), c1 = __popc(m0), c2r = c1 + __popc(m1 & 0xFFFFu), c2 = c1 + __popc(m1), c3r = c2 + __popc(m2 & 0xFFFFu);
    const uint32_t mo = f8_wexcl(cm, lane);  // its first mark of the row block's
    const uint32_t y = f8_spread3(f8_bits(p3, 3 * (G.o3 + mo) - 32 * G.a3));  // its marks' level-3 digits (8), a nibble each
    uint32_t L0 = f8_prmt(T.l3a, T.l3b, y), L1 = f8_prmt(T.l3a, T.l3b, y >> 16);  // their exponents (0x80: a literal)
    if (G.lits) {  // (the literals: in order, into the marks' bytes; rare)
        uint32_t e0 = L0 & f8_bytes((int)min(cm, 4u)) & 0x80808080u, e1 = L1 & f8_bytes(cm > 4 ? (int)cm - 4 : 0) & 0x80808080u;
        uint32_t k4 = G.o4 - 8 * G.a4 + f8_wexcl(__popc(e0) + __popc(e1), lane);
        while (e0) {
            const int bit = __ffs((int)e0) - 1;
            e0 &= e0 - 1;
            L0 = (L0 & ~(0xFFu << (bit - 7))) | ((((p4[k4 >> 3] >> (4 * (k4 & 7))) & 15u) << 3) << (bit - 7));
            k4++;
        }
        while (e1) {
            const int bit = __ffs((int)e1) - 1;
            e1 &= e1 - 1;
            L1 = (L1 & ~(0xFFu << (bit - 7))) | ((((p4[k4 >> 3] >> (4 * (k4 & 7))) & 15u) << 3) << (bit - 7));
            k4++;
        }
    }
    // the exponents of its 24 spills: level 2's, a mark's from L by its rank among the lane's marks
    uint32_t e[6] = {f8_prmt(T.l2, 0, x0), f8_prmt(T.l2, 0, x0 >> 16), f8_prmt(T.l2, 0, x1), f8_prmt(T.l2, 0, x1 >> 16), f8_prmt(T.l2, 0, x2), f8_prmt(T.l2, 0, x2 >> 16)};
    const uint32_t mw[6] = {m0 & 0xFFFFu, m0 >> 16, m1 & 0xFFFFu, m1 >> 16, m2 & 0xFFFFu, m2 >> 16}, pre[6] = {0, c1r, c1, c2r, c2, c3r};
#pragma unroll
    for (int i = 0; i < 6; i++) e[i] = f8_prmt(e[i], f8_prmt(L0, L1, mw[i] * 0x1110u + pre[i] * 0x1111u), 0x3210 + (mw[i] << 2));
    __syncwarp();  // (eb: every lane past the granule before)
    uint2* ed = (uint2*)(eb + ES / 4 * lane);
    ed[0] = make_uint2(e[0], e[1]), ed[1] = make_uint2(e[2], e[3]), ed[2] = make_uint2(e[4], e[5]);
    // pass 2: each unit's values from its core words and its spills' bytes in eb
    uint32_t dw[4], c1u[4], t1 = 0;
    uint2 sw[4];
#pragma unroll
    for (int ks = 0; ks < 4; ks++) {
        dw[ks] = cw[ks * 96 + lane];
        sw[ks] = ((const uint2*)(cw + ks * 96 + 32))[lane];
        c1u[ks] = __popc(f8_threes(dw[ks]));
        t1 += c1u[ks];
    }
    uint32_t s = f8_wexcl(t1, lane);  // the lane's first spill of the row block's
    __syncwarp();  // (eb written by every lane)
    const uint8_t* e8 = (const uint8_t*)eb;
#pragma unroll
    for (int ks = 0; ks < 4; ks++) {
        const uint32_t Wa = dw[ks] & 0x33333333u, Wb = (dw[ks] >> 2) & 0x33333333u;
        const uint32_t Ta = Wa & (Wa >> 1) & 0x11111111u, Tb = Wb & (Wb >> 1) & 0x11111111u;
        const uint32_t ca = (Ta * 0x11111111u) >> 28;
        uint32_t sp[4];  // half a's spills' bytes (from s), half b's (from s + ca)
#pragma unroll
        for (int hb = 0; hb < 2; hb++) {
            const uint32_t at = s + (hb ? ca : 0u), r = (at & 3) * 0x1111u + 0x3210u;
            const uint32_t* ew = (const uint32_t*)(e8 + (at & ~3u));
            const uint32_t u0 = ew[0], u1 = ew[1], u2 = ew[2];
            sp[2 * hb] = f8_prmt(u0, u1, r), sp[2 * hb + 1] = f8_prmt(u1, u2, r);
        }
        s += c1u[ks];
        const uint32_t Ra = Ta * 0x11111110u, Rb = Tb * 0x11111110u;  // a value's spills before it in its half
        const uint32_t s0 = sw[ks].x, s1 = sw[ks].y;
        o[ks][0] = (s0 & 0x87878787u) | f8_prmt(f8_prmt(T.lo, 0, Wa), f8_prmt(sp[0], sp[1], Ra), 0x3210 + (Ta << 2));
        o[ks][1] = (__funnelshift_l(s0, s0, 4) & 0x87878787u) | f8_prmt(f8_prmt(T.lo, 0, Wa >> 16), f8_prmt(sp[0], sp[1], Ra >> 16), 0x3210 + (Ta >> 14));
        o[ks][2] = (s1 & 0x87878787u) | f8_prmt(f8_prmt(T.lo, 0, Wb), f8_prmt(sp[2], sp[3], Rb), 0x3210 + (Tb << 2));
        o[ks][3] = (__funnelshift_l(s1, s1, 4) & 0x87878787u) | f8_prmt(f8_prmt(T.lo, 0, Wb >> 16), f8_prmt(sp[2], sp[3], Rb >> 16), 0x3210 + (Tb >> 14));
        between(ks);
    }
}

// ---- layout H, for HBM GPUs: fewer instructions a value, a little more memory ----
// An HBM GPU issues 9 to 12 instructions a byte it reads (an H100 SXM 9-10, an A100 12.5; a 4080 SUPER 35) and the layout above takes about 25 to decode
// (research/dsv4/a100-stalls-2026-10-19): there the product waits on its decode, not on memory. H keeps the patches, row blocks, lanes, the order of values and
// the sign and mantissa words; a value's exponent is a 3-bit code, its rank among the patch's exponents (0-6: the 7 most common; 7: an escape, the exponent a
// 4-bit literal of the patch's literals in var, in the order row block, lane, value). About 7.1 bits a value on FP8 checkpoints (DeepSeek-V4-Flash's dense
// matrices: 2.9% escapes; 8 as released, 6.75 above), decoded by byte permutes: a code selects a byte of the patch's table.
// A row block's core: 14 words a lane, word w of lane l at w * 32 + l: words 2 ks and 2 ks + 1 unit ks's signs and mantissas (as above), words 8-13 its 64 codes
// as 8 groups of 24 bits (group 2 ks + h: values 8 h to 8 h + 7 of unit ks, code i at bit 3 i). A patch's head: x its literals' first word of var (a plain
// patch: its part's), y the table (rank r's exponent at bit 4 r), w bit 31 plain; then x row blocks 1 and 2's first literal (16 bits each, counted in the
// patch), y row block 3's and the patch's count. A plain patch (more than HMAXE escapes, or more than 16 in a lane's 64 values) as above: its stream's bytes, row blocks
// 0-2 in core, 3 in var.
constexpr int F8_H = 16;           // KIND | F8_H: a product of a matrix in layout H (block scales, the default plan)
constexpr int HLW = 14;            // a lane's words of a row block
constexpr int HGB = 32 * HLW * 4;  // a row block's core bytes
constexpr int HCORE = 4 * HGB;     // a patch's
constexpr int HCSL = 32 * HLW;     // the words of a row block's core: a slot of the product's
constexpr int HMAXE = 1024;        // a patch with more escapes is plain
static_assert(HCORE == 7168 && 3 * 2048 <= HCORE && TAIL * 4 == 2048, "a plain patch: row blocks 0-2 in core, 3 in var");

// H's pass 1: a patch's head (its words of var first, for f8_scan_kernel) from the counts of its exponents (warps 2 mb and 2 mb + 1: row block mb's rows).
__global__ void __launch_bounds__(NT) f8h_head_kernel(const F8Pk P, uint32_t* heads) {
    __shared__ __align__(16) uint32_t s_tile[VALUES / 4];
    __shared__ uint32_t s_cnt[NT / 32][16];
    __shared__ uint32_t s_sum[16];
    __shared__ uint8_t s_sym[16], s_rank[16];
    const int tid = threadIdx.x, mb = tid >> 6, l = (tid >> 1) & 31, hh = tid & 1, g = l >> 2, t = l & 3;
    if (tid < NT / 2) ((uint32_t*)s_cnt)[tid] = 0;
    __syncthreads();
    uint32_t x[8];
    f8_load(P, blockIdx.x, tid, x);
#pragma unroll
    for (int i = 0; i < 8; i++)
#pragma unroll
        for (int k = 0; k < 4; k++) atomicAdd(&s_cnt[tid >> 5][(x[i] >> (8 * k + 3)) & 15], 1u);
    {
        const int row = tid >> 2, wc = (tid & 3) * 8;
        *(uint4*)&s_tile[f8_word(row, wc)] = make_uint4(x[0], x[1], x[2], x[3]);
        *(uint4*)&s_tile[f8_word(row, wc + 4)] = make_uint4(x[4], x[5], x[6], x[7]);
    }
    __syncthreads();
    if (tid < 16) {
        uint32_t c = 0;
        for (int w = 0; w < NT / 32; w++) c += s_cnt[w][tid];
        s_sum[tid] = c;
    }
    __syncthreads();
    if (tid < 16) {  // (the layout above's ranks)
        uint32_t r = 0;
        for (int f = 0; f < 16; f++) r += (s_sum[f] > s_sum[tid]) | ((s_sum[f] == s_sum[tid]) & (f > tid));
        s_sym[r] = (uint8_t)tid;
        s_rank[tid] = (uint8_t)r;
    }
    __syncthreads();
    uint32_t ne = 0;  // the escapes of this thread's 32 values (the pack's: half its lane's row block), then of its lane's 64
#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
        for (int q = 0; q < 4; q++) {
            const uint32_t v = s_tile[f8_word(16 * mb + g + 8 * (q & 1), 8 * (2 * hh + h) + t + 4 * (q >> 1))];
#pragma unroll
            for (int k = 0; k < 4; k++) ne += s_rank[(v >> (8 * k + 3)) & 15] >= 7;
        }
    ne += __shfl_xor_sync(0xffffffffu, ne, 1);
    const int many = __syncthreads_or(ne > 16);  // (a lane of a row block with more than 16: the decode's window holds 16, so plain)
    if (tid == 0) {
        uint32_t n = 0, o[4];
        for (int mb = 0; mb < 4; mb++) {
            o[mb] = n;
            for (int e = 0; e < 16; e++) n += s_rank[e] >= 7 ? s_cnt[2 * mb][e] + s_cnt[2 * mb + 1][e] : 0u;
        }
        uint4* h = (uint4*)(heads + (int64_t)blockIdx.x * HEAD);
        if (n > (uint32_t)HMAXE || many || P.plain) {
            h[0] = make_uint4(TAIL, 0, 0, 1u << 31);
            h[1] = make_uint4(0, 0, 0, 0);
        } else {
            uint32_t tb = 0;
            for (int r = 0; r < 7; r++) tb |= (uint32_t)s_sym[r] << (4 * r);
            h[0] = make_uint4((n + 7) >> 3, tb, 0, 0);
            h[1] = make_uint4(o[1] | (o[2] << 16), o[3] | (n << 16), 0, 0);
        }
    }
}

// H's pass 3: a patch's core and literals (heads from passes 1 and 2). Thread tid as the pack above: row block mb = tid / 64, lane l = (tid / 2) % 32, units
// 2 (tid % 2) and the next: the lane's values 32 (tid % 2) to 32 (tid % 2) + 31, its 3 words of codes 3 (tid % 2) on.
__global__ void __launch_bounds__(NT) f8h_pack_kernel(const F8Pk P, const uint32_t* heads, uint8_t* core_out, uint32_t* var, int64_t var_words) {
    __shared__ __align__(16) uint32_t s_tile[VALUES / 4];
    __shared__ uint32_t s_var[HMAXE / 8 + 8];
    __shared__ uint32_t s_tot[NT / 32];
    __shared__ uint8_t s_rank[16];
    const int tid = threadIdx.x, mb = tid >> 6, l = (tid >> 1) & 31, hh = tid & 1, g = l >> 2, t = l & 3;
    const int64_t pat = blockIdx.x;
    const uint4 hd = __ldg((const uint4*)(heads + pat * HEAD)), hb = __ldg((const uint4*)(heads + pat * HEAD) + 1);
    const bool plain = (hd.w >> 31) != 0;
    const uint32_t base = hd.x, words = plain ? (uint32_t)TAIL : ((hb.y >> 16) + 7) >> 3;
    if (blockIdx.x == 0 && tid < GUARD) var[var_words - GUARD + tid] = 0;
    if ((int64_t)base + words > var_words - GUARD) return;  // (a heads array that is not this matrix's: nothing is written)
    if (tid < 16) {  // the code of an exponent: its rank where it is in the table, else 7
        uint32_t r = 7;
        for (int i = 0; i < 7; i++)
            if (((hd.y >> (4 * i)) & 15) == (uint32_t)tid) r = i;
        s_rank[tid] = (uint8_t)r;
    }
    for (int i = tid; i < HMAXE / 8 + 8; i += NT) s_var[i] = 0;
    {
        uint32_t x[8];
        f8_load(P, pat, tid, x);
        const int row = tid >> 2, wc = (tid & 3) * 8;
        *(uint4*)&s_tile[f8_word(row, wc)] = make_uint4(x[0], x[1], x[2], x[3]);
        *(uint4*)&s_tile[f8_word(row, wc + 4)] = make_uint4(x[4], x[5], x[6], x[7]);
    }
    __syncthreads();
    uint32_t y[2][4];  // unit h (ks = 2 hh + h), value j = 4 q + k: byte k of y[h][q]
#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
        for (int q = 0; q < 4; q++) y[h][q] = s_tile[f8_word(16 * mb + g + 8 * (q & 1), 8 * (2 * hh + h) + t + 4 * (q >> 1))];
    uint8_t* core = core_out + pat * HCORE;
    if (plain) {
        if (tid < CORE / 32) {
            *(uint4*)(core + tid * 32) = make_uint4(y[0][0], y[0][1], y[0][2], y[0][3]);
            *(uint4*)(core + tid * 32 + 16) = make_uint4(y[1][0], y[1][1], y[1][2], y[1][3]);
        } else {
            uint32_t* v = var + base + (tid - CORE / 32) * 8;
#pragma unroll
            for (int h = 0; h < 2; h++)
#pragma unroll
                for (int q = 0; q < 4; q++) v[4 * h + q] = y[h][q];
        }
        return;
    }
    uint32_t sw[2][2] = {{0, 0}, {0, 0}}, cd[3] = {0, 0, 0}, ne = 0;
#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
        for (int q = 0; q < 4; q++)
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const int j = 4 * q + k, bit = 24 * (2 * h + (j >> 3)) + 3 * (j & 7);
                const uint32_t b = (y[h][q] >> (8 * k)) & 0xFF, r = s_rank[(b >> 3) & 15];
                sw[h][j >> 3] |= ((b >> 7) << f8_s_bit(j)) | ((b & 7) << f8_m_bit(j));
                cd[bit >> 5] |= r << (bit & 31);
                if ((bit & 31) > 29) cd[(bit >> 5) + 1] |= r >> (32 - (bit & 31));
                ne += r == 7;
            }
    uint32_t kl = f8_excl(ne, s_tot);  // (the patch's literals before this thread's; its __syncthreads has s_var zeroed)
#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
        for (int q = 0; q < 4; q++)
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const uint32_t e = (y[h][q] >> (8 * k + 3)) & 15;
                if (s_rank[e] == 7) {
                    atomicOr(&s_var[kl >> 3], e << (4 * (kl & 7)));
                    kl++;
                }
            }
    uint32_t* cw = (uint32_t*)(core + mb * HGB);
#pragma unroll
    for (int h = 0; h < 2; h++) {
        const int ks = 2 * hh + h;
        cw[(2 * ks) * 32 + l] = sw[h][0];
        cw[(2 * ks + 1) * 32 + l] = sw[h][1];
    }
#pragma unroll
    for (int i = 0; i < 3; i++) cw[(8 + 3 * hh + i) * 32 + l] = cd[i];
    __syncthreads();
    for (uint32_t i = tid; i < words; i += NT) var[base + i] = s_var[i];
}

// Row block mb's literals in its patch's (head hB): its first and the next row block's first.
__device__ __forceinline__ void f8h_lits(const uint4& hB, int mb, uint32_t& s, uint32_t& e) {
    const uint32_t a = hB.x & 0xFFFF, b = hB.x >> 16, c = hB.y & 0xFFFF, n = hB.y >> 16;
    s = mb == 0 ? 0u : mb == 1 ? a : mb == 2 ? b : c;
    e = mb == 0 ? a : mb == 1 ? b : mb == 2 ? c : n;
}

// A lane's row block in layout H into o as f8_granule leaves it (value 4 q + k of unit ks at byte k of o[ks][q]): cw its core (word w at w * 32 + lane), lp its literals
// from the row block's first word of them (kb: its first literal's place in that word; read where a lane of the warp has an escape, two words past its last too), tab the
// head's table. Each value: its code's byte of the table (an exponent e as e << 3; an escape 0) by a byte permute; then where the warp has escapes, each lane's
// literals (16 at most, from one window) put in its escapes' bytes in order.
__device__ __forceinline__ void f8h_core_words(const uint32_t* cw, int lane, uint32_t (&sm)[8], uint32_t (&cd)[7]) {  // the lane's 14 words of a core slot
#pragma unroll
    for (int w = 0; w < 8; w++) sm[w] = cw[w * 32 + lane];
#pragma unroll
    for (int w = 0; w < 6; w++) cd[w] = cw[(8 + w) * 32 + lane];
    cd[6] = 0;
}
__device__ __forceinline__ void f8h_granule_regs(const uint32_t (&sm)[8], const uint32_t (&cd)[7], const uint32_t* lp, uint32_t kb, uint32_t tab, int lane,
                                                 uint32_t (&o)[4][4]);
__device__ __forceinline__ void f8h_granule(const uint32_t* cw, const uint32_t* lp, uint32_t kb, uint32_t tab, int lane, uint32_t (&o)[4][4]) {
    uint32_t sm[8], cd[7];
    f8h_core_words(cw, lane, sm, cd);
    f8h_granule_regs(sm, cd, lp, kb, tab, lane, o);
}
__device__ __forceinline__ void f8h_granule_regs(const uint32_t (&sm)[8], const uint32_t (&cd)[7], const uint32_t* lp, uint32_t kb, uint32_t tab, int lane,
                                                 uint32_t (&o)[4][4]) {
    const uint32_t T0 = f8_nib4(tab), T1 = f8_nib4(tab >> 16) & 0x00FFFFFFu;  // ranks 0-3; 4-6 and an escape's 0 (its literal ORed in below)
    uint32_t y[8], f[8], c = 0;
#pragma unroll
    for (int gi = 0; gi < 8; gi++) {  // group gi's 8 codes a nibble each; the escapes' (code 7) bit 3 of their nibble (7 + 1 alone reaches it)
        const int b = 24 * gi;
        y[gi] = f8_spread3(__funnelshift_r(cd[b >> 5], cd[(b >> 5) + 1], b & 31));
        f[gi] = (y[gi] + 0x11111111u) & 0x88888888u;
        c += __popc(f[gi]);
    }
    uint32_t E[4][4];
#pragma unroll
    for (int ks = 0; ks < 4; ks++)
#pragma unroll
        for (int q = 0; q < 4; q++) E[ks][q] = f8_prmt(T0, T1, y[2 * ks + (q >> 1)] >> (16 * (q & 1)));
    if (__any_sync(0xffffffffu, c != 0)) {
        const uint32_t k = kb + f8_wexcl(c, lane);  // the lane's first literal, from lp's first
        // Its literals (16 at most: f8h_head_kernel's) as bytes e << 3, 4 a register, from one window. Byte p of a register: literal pre + the marks below p, pre the
        // lane's marks before the register (kept as pre 0x1111, a nibble a byte; prmt reads a selector's low 16 bits alone). Where no lane has more than 7 (nearly every
        // row block): one permute of LB0, LB1 a register ORed in, an unmarked byte's selector 8 more (prmt's sign mode on a literal: 0), so f * 0x221 + 0x8888 + pre
        // 0x1111 (f's marks 8 a nibble; 8 + 7 at most, nibble for nibble). Else also of LB2, LB3 by the selector less 8, the two OR'd and put in the marked bytes.
        const uint32_t w0 = lp[k >> 3], w1 = lp[(k >> 3) + 1], w2 = lp[(k >> 3) + 2], sh = 4 * (k & 7);
        const uint32_t L0 = __funnelshift_r(w0, w1, sh), L1 = __funnelshift_r(w1, w2, sh);
        const uint32_t LB0 = f8_nib4(L0), LB1 = f8_nib4(L0 >> 16);
        if (!__any_sync(0xffffffffu, c > 7)) {
            uint32_t pre = 0x8888u;
#pragma unroll
            for (int gi = 0; gi < 8; gi++) {
                const uint32_t lo = f[gi] & 0xFFFFu, hi = f[gi] >> 16, p1 = pre + __popc(lo) * 0x1111u;
                E[gi >> 1][2 * (gi & 1)] |= f8_prmt(LB0, LB1, lo * 0x221u + pre);
                E[gi >> 1][2 * (gi & 1) + 1] |= f8_prmt(LB0, LB1, hi * 0x221u + p1);
                pre = p1 + __popc(hi) * 0x1111u;
            }
        } else {
            const uint32_t LB2 = f8_nib4(L1), LB3 = f8_nib4(L1 >> 16);
            auto put = [](uint32_t& e, uint32_t m, uint32_t lit) {  // (lit into e's bytes marked in m, 8 a nibble: M 0xFF there, by prmt's sign mode on 0x80)
                const uint32_t M = f8_prmt(0x80u, 0, 0x4444u + (m >> 1));
                e = (e & ~M) | (lit & M);
            };
            uint32_t pre = 0;
#pragma unroll
            for (int gi = 0; gi < 8; gi++) {
                const uint32_t lo = f[gi] & 0xFFFFu, hi = f[gi] >> 16, p1 = pre + __popc(lo) * 0x1111u, s0 = lo * 0x222u + pre, s1 = hi * 0x222u + p1;
                put(E[gi >> 1][2 * (gi & 1)], lo, f8_prmt(LB0, LB1, s0) | f8_prmt(LB2, LB3, s0 ^ 0x8888u));
                put(E[gi >> 1][2 * (gi & 1) + 1], hi, f8_prmt(LB0, LB1, s1) | f8_prmt(LB2, LB3, s1 ^ 0x8888u));
                pre = p1 + __popc(hi) * 0x1111u;
            }
        }
    }
#pragma unroll
    for (int ks = 0; ks < 4; ks++) {
        const uint32_t s0 = sm[2 * ks], s1 = sm[2 * ks + 1];
        o[ks][0] = (s0 & 0x87878787u) | E[ks][0];
        o[ks][1] = (__funnelshift_l(s0, s0, 4) & 0x87878787u) | E[ks][1];
        o[ks][2] = (s1 & 0x87878787u) | E[ks][2];
        o[ks][3] = (__funnelshift_l(s1, s1, 4) & 0x87878787u) | E[ks][3];
    }
}

// The product's copies of row block mb of patch p for its decode a step or two later: its core (1,792 bytes) into cw; its literals (f8h_words: from its first's word,
// two words past its last: a window reads 3 words) into vw where they fit (else read where they are).
__device__ __forceinline__ void f8h_core_in(const uint8_t* core, int64_t p, int mb, int lane, uint32_t* cw) {
    const uint4* cs = (const uint4*)(core + p * HCORE + mb * HGB);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int c = lane + 32 * i;
        if (c < HGB / 16) f8_cp16(cw + 4 * c, cs + c);
    }
}
__device__ __forceinline__ uint32_t f8h_words(const uint4& hB, int mb, uint32_t& w0) {
    uint32_t s, e;
    f8h_lits(hB, mb, s, e);
    w0 = s >> 3;
    return ((e + 7) >> 3) - w0 + 2;
}
__device__ __forceinline__ void f8h_var_in(const uint32_t* var, int mb, int lane, const uint4& hA, const uint4& hB, uint32_t* vw) {
    if (hA.w >> 31) return;
    uint32_t w0;
    const uint32_t n = f8h_words(hB, mb, w0);
    if (n > (uint32_t)VB) return;
    const uint32_t* src = var + hA.x + w0;
    for (uint32_t i = lane; i < n; i += 32) f8_cp4(vw + i, src + i);
}
__device__ __forceinline__ void f8h_decode(const uint8_t* core, const uint32_t* var, int64_t p, int mb, int lane, const uint4& hA, const uint4& hB, const uint32_t* cw,
                                           const uint32_t* vw, uint32_t (&o)[4][4]) {
    if (hA.w >> 31) {  // plain: the lane's 64 stream bytes (row blocks 0-2 in core, 3 in var)
        const uint32_t* s = mb < 3 ? (const uint32_t*)(core + p * HCORE + 2048 * mb) + 16 * lane : var + hA.x + 16 * lane;
#pragma unroll
        for (int i = 0; i < 16; i++) o[i >> 2][i & 3] = __ldg(s + i);
        return;
    }
    uint32_t s, e, w0;
    f8h_lits(hB, mb, s, e);
    const uint32_t n = f8h_words(hB, mb, w0);
    f8h_granule(cw, n <= (uint32_t)VB ? vw : var + hA.x + w0, s & 7, hA.y, lane, o);
}
static_assert(HCSL == f8_csl<2, true>(), "the product's slots of layout H's cores");

// H's unpack: a patch (or, ids, a patch of the stacked matrices ids[blockIdx.y]) into out; warp mb row block mb, decoded as the product does from core and var.
constexpr int HUNT = 128;
__global__ void __launch_bounds__(HUNT) f8h_unpack_kernel(const F8Un P) {
    __shared__ __align__(16) uint32_t s_tile[VALUES / 4];
    const int tid = threadIdx.x, mb = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int pl = blockIdx.x;
    int64_t pat = pl, item = 0;
    if (P.ids) {
        item = blockIdx.y;
        const int id = P.ids[item];
        if ((unsigned)id >= (unsigned)P.E) return;  // (no such matrix: its place in out is left as it is)
        pat += (int64_t)id * P.per;
    }
    const uint4 hA = __ldg((const uint4*)(P.heads + pat * HEAD)), hB = __ldg((const uint4*)(P.heads + pat * HEAD) + 1);
    const uint8_t* core = P.core + pat * HCORE;
    uint32_t o[4][4];
    if (hA.w >> 31) {  // plain: the lane's 64 stream bytes
        const uint32_t* s = mb < 3 ? (const uint32_t*)(core + 2048 * mb) + 16 * lane : P.var + hA.x + 16 * lane;
#pragma unroll
        for (int i = 0; i < 16; i++) o[i >> 2][i & 3] = __ldg(s + i);
    } else {
        uint32_t s, e;
        f8h_lits(hB, mb, s, e);
        f8h_granule((const uint32_t*)(core + mb * HGB), P.var + hA.x + (s >> 3), s & 7, hA.y, lane, o);
    }
#pragma unroll
    for (int ks = 0; ks < 4; ks++)
#pragma unroll
        for (int q = 0; q < 4; q++) s_tile[f8_word(16 * mb + g + 8 * (q & 1), 8 * ks + t + 4 * (q >> 1))] = o[ks][q];
    __syncthreads();
    const int row0 = (pl / P.KT) * PR, col0 = (pl % P.KT) * PC;
    uint8_t* dst = P.out + item * P.item + (int64_t)row0 * P.ld + col0;
    for (int c = tid; c < PR * 8; c += HUNT) {
        const int row = c >> 3, g8 = c & 7;
        if (row0 + row >= P.O) continue;
        const uint4 v = *(const uint4*)&s_tile[row * 32 + (((g8 ^ row) & 7) << 2)];
        uint8_t* d = dst + (int64_t)row * P.ld + g8 * 16;
        if (P.vec && col0 + g8 * 16 + 16 <= P.K) {
            F8_ALIGN(d, 16);
            *(uint4*)d = v;
        } else {
            const uint32_t b[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int k = 0; k < 16; k++)
                if (col0 + g8 * 16 + k < P.K) d[k] = (uint8_t)(b[k >> 2] >> (8 * (k & 3)));
        }
    }
}

// The experts' product's slot blockIdx.y as one matrix's product (P0's pointers at its expert and its rows; done and parts: the slot's own), or false where the plan has no such slot.
template <int SK, bool H = false>
__device__ __forceinline__ bool f8_slot(const F8Lin& P0, F8Lin& P) {
    const int s = blockIdx.y, E = P0.E, *plan = P0.plan;
    if (s >= plan[0]) return false;
    const int e = plan[1 + s], a = plan[1 + E + s], b = plan[2 + E + s];
    if (b <= a) return false;
    const int64_t per = f8_cdiv(P0.N, PR) * P0.KT;  // (an expert's patches)
    P = P0;
    P.core += e * per * (H ? HCORE : CORE);  // (layout H's cores are HCORE bytes a patch)
    P.heads += e * per * HEAD;
    if constexpr (SK == F8_BLOCK) P.scale = (const float*)P0.scale + e * f8_cdiv(P0.N, 128) * P0.lds;
    P.order = P0.gather ? plan + 2 + 2 * E + a : nullptr;
    if (!P0.gather) P.x += (int64_t)a * P0.ldx;
    P.y += (int64_t)a * P0.ldy;
    P.parts += (int64_t)s * 2 * P0.ctas * P0.M * 128;
    P.done += (int64_t)s * P0.RG;
    P.M = b - a;
    return true;
}

// AHEAD (32-64 tokens): a warp decodes the granule of the step after its products' (copies two steps ahead), and of the two warps of an SM's
// quarter (w and w + 4) one does its products first and the other its decode, so that the tensor cores and the rest of the SM work at once.
// KIND | F8_A8: the W8A8 product (block, tensor and channel scales), never AHEAD or DENSE.
// G: the experts' product (glyd_gpu_f8_moe: a slot of the plan a blockIdx.y, f8_slot's).
// KIND | F8_H: W in layout H (block scales, the default plan alone): its row blocks' cores and literals copied and decoded as H has them, the rest as above.
template <int NTL, int KIND, int AHEAD, int DENSE, int G = 0, int S = 2>
__global__ void __launch_bounds__(LNT, 1) f8_linear_kernel(const F8Lin P0) {
    constexpr bool H = (KIND & F8_H) != 0;
    constexpr int D = AHEAD ? 1 : 0, SK = KIND & 7, HS = f8_hs<S>(), XS = f8_xs<S>(), LSM = f8_lsmem<S, H>(), CS = f8_csl<S, H>();
    static_assert(S == 2 || (!AHEAD && !DENSE), "three slots: the one-token product's path alone");
    static_assert(!H || (SK == F8_BLOCK && !(KIND & F8_A8) && !AHEAD && !DENSE && S == 2), "layout H: block scales, the default plan");
    constexpr bool A8 = (KIND & F8_A8) != 0, DIRECT = !A8 && (SK == F8_TENSOR || SK == F8_CHANNEL);  // (DIRECT: the sums straight into acc, the scale at the end)
    F8Lin Pg;
    if constexpr (G != 0)
        if (!f8_slot<SK, H>(P0, Pg)) return;  // (the CTA's slot: no expert, or none of its tokens)
    const F8Lin& P = G != 0 ? Pg : P0;
    F8_EXTERN_SMEM(uint32_t, f8_smem);
    __shared__ int s_last;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3, mb = warp & 3, half = warp >> 2;
    const int64_t W = (int64_t)P.RG * P.KT, cta = blockIdx.x;
    const int64_t a0 = cta * W / P.ctas;           // the CTA's part of the work: patches a0 to a0 + n - 1 (patch a: row group a / KT, column a % KT)
    const int n = (int)((cta + 1) * W / P.ctas - a0);
    uint4* hs = (uint4*)f8_smem;  // HSL slots of 4: patch rows 2 rg and 2 rg + 1, hA and hB each
    uint32_t* cw = f8_smem + HS * 16 + warp * (S * CS + S * VB);  // the warp's S slots of core, then its S of variable parts (H: of literals)
    uint32_t* vw = cw + S * CS;
    uint32_t* eb = (uint32_t*)((uint8_t*)f8_smem + LSM + warp * EBS);  // (DENSE: the warp's buffer of exponents)
    uint8_t* xs = (uint8_t*)f8_smem + LSM + (DENSE ? LW * EBS : 0);  // XS slots of M token rows of XROW bytes
    const int XSLOT = f8_xslot(P.M, A8), RP = (int)f8_cdiv(P.N, PR), RB = RP * 4;
    const int nts = (P.M + 7) >> 3;  // the 8-token tiles with tokens in them
    const int kun = (int)f8_cdiv(P.K, 32);
    auto heads_in = [&](int j, int64_t rg, int kp) {  // the heads of step j (row group rg, column kp: the patches of its two rows there) into slot j % HSL
        if (tid < 4) {
            const int64_t rp = 2 * rg + (tid >> 1);
            uint4* d = hs + (j % HS) * 4 + tid;
            if (rp < RP) f8_cp16(d, (const uint4*)(P.heads + (rp * P.KT + kp) * HEAD) + (tid & 1));
            else *d = make_uint4(0, 0, 0, 0);  // (past the matrix: no warp's)
        }
    };
    auto x_in = [&](int j, int kp) {  // step j's 128 columns of X (column kp; tokens < M) into slot j % XSL: in a group of 32 columns, its 8-byte pieces i
                                      // (columns 4i..4i+3) at 2 (i & 3) + (i >> 2); an odd row's groups in pairs swapped
        uint8_t* base = xs + (j % XS) * XSLOT;
        if (A8) {  // the token rows of xq (16 bytes a copy), then their scales
            for (int idx = tid; idx < P.M * 8; idx += LNT) {
                const int mm = idx >> 3, c = idx & 7;
                f8_cp16(base + mm * PC + 32 * ((c >> 1) ^ (mm & 3)) + 16 * (c & 1), P.xq + ((int64_t)mm * P.KT + kp) * PC + 16 * c);
            }
            if (tid < P.M) f8_cp4((uint32_t*)(base + P.M * PC) + tid, (const uint32_t*)(P.sx + (int64_t)tid * P.KT + kp));
        }
        if (!A8)
        for (int idx = tid; idx < P.M * 16; idx += LNT) {
            const int mm = idx >> 4, c8 = idx & 15, k = kp * PC + 8 * c8, i0 = (c8 & 3) * 2, i1 = i0 + 1, sw = (mm & 1) << 6;
            const uint16_t* s = P.x + (int64_t)(G != 0 && P.order ? P.order[mm] / P.gk : mm) * P.ldx + k;
            uint8_t* row = base + mm * XROW;
            uint32_t* d0 = (uint32_t*)(row + (((c8 >> 2) * 64 + 8 * (2 * (i0 & 3) + (i0 >> 2))) ^ sw));
            uint32_t* d1 = (uint32_t*)(row + (((c8 >> 2) * 64 + 8 * (2 * (i1 & 3) + (i1 >> 2))) ^ sw));
            if (P.xvec && k + 8 <= P.K) {
                f8_cp8(d0, (const uint32_t*)s);
                f8_cp8(d1, (const uint32_t*)(s + 4));
            } else {
                uint32_t v[4];
#pragma unroll
                for (int i = 0; i < 4; i++) v[i] = (k + 2 * i < P.K ? (uint32_t)s[2 * i] : 0u) | (k + 2 * i + 1 < P.K ? (uint32_t)s[2 * i + 1] << 16 : 0u);
                d0[0] = v[0], d0[1] = v[1], d1[0] = v[2], d1[1] = v[3];
            }
        }
    };
    auto mine = [&](int64_t rg) { return rg * LW + warp < RB; };  // (the warp's row block of row group rg in the matrix)
    auto core_in = [&](int j, int64_t rg, int kp) {
        if (!mine(rg)) return;
        if constexpr (H) f8h_core_in(P.core, (2 * rg + half) * P.KT + kp, mb, lane, cw + (j % S) * CS);
        else f8_core_in(P, (2 * rg + half) * P.KT + kp, mb, lane, cw + (j % S) * CS);
    };
    auto var_in = [&](int j, int64_t rg) {  // (step j's heads in)
        if (mine(rg)) {
            const uint4* h = hs + (j % HS) * 4 + 2 * half;
            if constexpr (H) f8h_var_in(P.var, mb, lane, h[0], h[1], vw + (j % S) * VB);
            else f8_var_in(P, mb, lane, h[0], h[1], vw + (j % S) * VB);
        }
    };
    auto decode = [&](int j, int64_t rg, int kp, uint32_t (&o)[4][4]) {
        const uint4* h = hs + (j % HS) * 4 + 2 * half;
        if constexpr (H) f8h_decode(P.core, P.var, (2 * rg + half) * P.KT + kp, mb, lane, h[0], h[1], cw + (j % S) * CS, vw + (j % S) * VB, o);
        else if (DENSE) f8_granule_dense(P, (2 * rg + half) * P.KT + kp, mb, lane, h[0], h[1], cw + (j % S) * CS, vw + (j % S) * VB, eb, o, [](int) {});
        else f8_granule(P, (2 * rg + half) * P.KT + kp, mb, lane, h[0], h[1], cw + (j % S) * CS, vw + (j % S) * VB, o);
    };
    float acc[NTL][4];
#pragma unroll
    for (int nt = 0; nt < NTL; nt++) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
    const int g1 = g & 1;
    float tmp[NTL][4];  // (a granule's sums (block scales) or a unit's (mx, ds32), before its scale)
    auto prod_begin = [&]() {
        if (SK == F8_BLOCK || A8)
#pragma unroll
            for (int nt = 0; nt < NTL; nt++) tmp[nt][0] = tmp[nt][1] = tmp[nt][2] = tmp[nt][3] = 0.f;
    };
    auto prod_unit = [&](int j, int64_t rg, int kp, int ks, const uint32_t (&ok)[4]) {  // unit ks's products with step j's X
        if constexpr (A8) {  // the decoded bytes as they are, X's e4m3 8 bytes a lane: one product a unit and 8 tokens
            const uint8_t* xsl = xs + (j % XS) * XSLOT + 8 * t;
            uint2 xb[NTL];
#pragma unroll
            for (int nt = 0; nt < NTL; nt++)
                if (nt < nts) {
                    const int m = min(8 * nt + g, P.M - 1);
                    xb[nt] = *(const uint2*)(xsl + m * PC + 32 * (ks ^ (m & 3)));
                }
#pragma unroll
            for (int nt = 0; nt < NTL; nt++)
                if (nt < nts) f8_mma8(tmp[nt], ok[0], ok[1], ok[2], ok[3], xb[nt].x, xb[nt].y);
            return;
        }
        const uint8_t* xsl = xs + (j % XS) * XSLOT + 16 * t;
        const uint32_t a0_ = f8_bf_lo(ok[0]), a1_ = f8_bf_lo(ok[1]), a2_ = f8_bf_lo(ok[2]), a3_ = f8_bf_lo(ok[3]);
        const uint32_t b0_ = f8_bf_hi(ok[0]), b1_ = f8_bf_hi(ok[1]), b2_ = f8_bf_hi(ok[2]), b3_ = f8_bf_hi(ok[3]);
        if (SK == F8_MX || SK == F8_DS32)
#pragma unroll
            for (int nt = 0; nt < NTL; nt++) tmp[nt][0] = tmp[nt][1] = tmp[nt][2] = tmp[nt][3] = 0.f;
        uint4 xb[NTL];  // (a token past M: any token's row (a row of the slot: its parity need not be g's), its sums not kept)
#pragma unroll
        for (int nt = 0; nt < NTL; nt++)
            if (nt < nts) {
                const uint4 v = *(const uint4*)(xsl + min(8 * nt + g, P.M - 1) * XROW + 64 * (ks ^ g1));
                xb[nt] = make_uint4(f8_x64(v.x), f8_x64(v.y), f8_x64(v.z), f8_x64(v.w));
            }
#pragma unroll
        for (int nt = 0; nt < NTL; nt++)  // (every tile's first product, then every tile's second: a tile's two are NTL products apart)
            if (nt < nts) f8_mma(DIRECT ? acc[nt] : tmp[nt], a0_, a1_, a2_, a3_, xb[nt].x, xb[nt].z);
#pragma unroll
        for (int nt = 0; nt < NTL; nt++)
            if (nt < nts) f8_mma(DIRECT ? acc[nt] : tmp[nt], b0_, b1_, b2_, b3_, xb[nt].y, xb[nt].w);
        if (SK == F8_MX || SK == F8_DS32) {
            const int rb = (int)(rg * LW) + warp;
            const int r0 = min(16 * rb + g, P.N - 1), r1 = min(16 * rb + g + 8, P.N - 1);  // (scale rows: past N, any row's)
            const int ku = min(4 * kp + ks, kun - 1);
            const uint8_t* sc = (const uint8_t*)P.scale;
            const float s0 = f8_e8m0(SK == F8_MX ? sc[(int64_t)r0 * P.lds + ku] : sc[(int64_t)(r0 >> 5) * P.lds + ku]);
            const float s1 = SK == F8_MX ? f8_e8m0(sc[(int64_t)r1 * P.lds + ku]) : s0;
#pragma unroll
            for (int nt = 0; nt < NTL; nt++) {
                acc[nt][0] += tmp[nt][0] * s0, acc[nt][1] += tmp[nt][1] * s0;
                acc[nt][2] += tmp[nt][2] * s1, acc[nt][3] += tmp[nt][3] * s1;
            }
        }
    };
    auto prod_end = [&](int j, int64_t rg, int kp) {
        if constexpr (A8) {  // the granule's sums times each token's scale of step j (and the block's)
            const float* sxs = (const float*)(xs + (j % XS) * XSLOT + P.M * PC);
            const float w = SK == F8_BLOCK ? ((const float*)P.scale)[rg * P.lds + kp] : 1.f;
#pragma unroll
            for (int nt = 0; nt < NTL; nt++) {
                const float x0 = sxs[min(8 * nt + 2 * t, P.M - 1)] * w, x1 = sxs[min(8 * nt + 2 * t + 1, P.M - 1)] * w;
                acc[nt][0] += tmp[nt][0] * x0, acc[nt][1] += tmp[nt][1] * x1, acc[nt][2] += tmp[nt][2] * x0, acc[nt][3] += tmp[nt][3] * x1;
            }
        } else if (SK == F8_BLOCK) {
            const float s = ((const float*)P.scale)[rg * P.lds + kp];
#pragma unroll
            for (int nt = 0; nt < NTL; nt++)
#pragma unroll
                for (int i = 0; i < 4; i++) acc[nt][i] += tmp[nt][i] * s;
        }
    };
    auto product = [&](int j, int64_t rg, int kp, const uint32_t (&o)[4][4]) {  // the granule's products with step j's X into acc, its scales applied
        prod_begin();
#pragma unroll
        for (int ks = 0; ks < 4; ks++) prod_unit(j, rg, kp, ks, o[ks]);
        prod_end(j, rg, kp);
    };
    auto step = [&](int64_t& rg, int& kp) {  // the next patch of the order
        if (++kp == P.KT) kp = 0, rg++;
    };
    const int64_t rg0 = a0 / P.KT;
    const int kp0 = (int)(a0 % P.KT);
    // the steps' places: A = j (products), B = j + 1, X = j + S - 1 (X's copies), C = j + S - 1 + D (core's and variable parts' copies), H = j + 2 S - 2 + D (heads)
    int64_t rgA = rg0, rgB = rg0, rgC, rgH, rgX;
    int kpA = kp0, kpB = kp0, kpC, kpH, kpX;
    step(rgB, kpB);
    rgX = rgB, kpX = kpB;
    for (int i = 2; i < S; i++) step(rgX, kpX);
    rgC = rgX, kpC = kpX;
    if (D) step(rgC, kpC);
    rgH = rgC, kpH = kpC;
    for (int i = 1; i < S; i++) step(rgH, kpH);
    // the first steps' core and heads and X, waited for; then their variable parts (their heads in)
    {
        int64_t r = rg0;
        int k = kp0;
        for (int i = 0; i <= 2 * S - 3 + D && i < n; i++) {
            heads_in(i, r, k);
            if (i <= S - 2 + D) core_in(i, r, k);
            if (i <= S - 2) x_in(i, k);
            step(r, k);
        }
    }
    f8_cp_commit();
    f8_cp_wait0();
    __syncthreads();
    if constexpr (S == 2) {
        var_in(0, rg0);
        if (D && n > 1) var_in(1, rgB);
        f8_cp_commit();
    } else {  // (all the first steps' in before the loop, whose waits leave S - 2 steps' copies in flight)
        int64_t r = rg0;
        int k = kp0;
        for (int i = 0; i <= S - 2 && i < n; i++) {
            var_in(i, r);
            step(r, k);
        }
        f8_cp_commit();
        f8_cp_wait0();
        __syncthreads();
    }
    uint32_t oc[4][4];  // (AHEAD: the granule decoded a step ahead)
    if (D) {
        f8_cp_wait0();
        __syncwarp();  // (a warp's copies are its own)
        if (mine(rg0)) decode(0, rg0, kp0, oc);
    }
    for (int j = 0; j < n; j++) {
        f8_cp_wait<S - 2>();
        __syncthreads();  // step j's copies in for every thread (and step j + S - 1's heads), every warp past step j - 1 (its slots free)
        if (j + S - 1 + D < n) {
            core_in(j + S - 1 + D, rgC, kpC);
            var_in(j + S - 1 + D, rgC);
        }
        if (j + S - 1 < n) x_in(j + S - 1, kpX);
        if (j + 2 * S - 2 + D < n) heads_in(j + 2 * S - 2 + D, rgH, kpH);
        f8_cp_commit();
        if (D) {
            uint32_t on[4][4];
            const bool nx = j + 1 < n && mine(rgB), cur = mine(rgA);
            if (DENSE && nx && cur) {  // the products of unit ks among the decode of the next granule's units
                const uint4* h = hs + ((j + 1) % HS) * 4 + 2 * half;
                prod_begin();
                f8_granule_dense(P, (2 * rgB + half) * P.KT + kpB, mb, lane, h[0], h[1], cw + ((j + 1) % S) * CS, vw + ((j + 1) % S) * VB, eb, on,
                                 [&](int ks) { prod_unit(j, rgA, kpA, ks, oc[ks]); });
                prod_end(j, rgA, kpA);
            } else if (half == 0) {
                if (cur) product(j, rgA, kpA, oc);
                if (nx) decode(j + 1, rgB, kpB, on);
            } else {
                if (nx) decode(j + 1, rgB, kpB, on);
                if (cur) product(j, rgA, kpA, oc);
            }
#pragma unroll
            for (int ks = 0; ks < 4; ks++)
#pragma unroll
                for (int q = 0; q < 4; q++) oc[ks][q] = on[ks][q];
        } else if (mine(rgA)) {
            decode(j, rgA, kpA, oc);
            product(j, rgA, kpA, oc);
        }
        if (kpA == P.KT - 1 || j == n - 1) {  // the end of the CTA's part of row group rgA: its sums out
            const int64_t rg = rgA;
            const int rb = (int)(rg * LW) + warp;
            const float f = (SK == F8_TENSOR ? ((const float*)P.scale)[0] : 1.f) * (A8 ? 1.f : 0x1p56f);  // (W8A16: the bf16 weights' 2^-120 and X's 2^64)
            const int64_t i0 = rg * P.KT;
            const int cf = (int)(((i0 + 1) * P.ctas - 1) / W), cl = (int)(((i0 + P.KT) * P.ctas - 1) / W);  // the CTAs that hold its patches
            const int row0 = 16 * rb + g;
            float rs[2] = {f, f};  // (channel: each of the lane's two rows' scale too)
            if (SK == F8_CHANNEL) rs[0] *= ((const float*)P.scale)[min(row0, P.N - 1)], rs[1] *= ((const float*)P.scale)[min(row0 + 8, P.N - 1)];
            if (cf == cl) {  // all of its K here: Y
                if (rb < RB)
#pragma unroll
                    for (int nt = 0; nt < NTL; nt++)
#pragma unroll
                        for (int i = 0; i < 4; i++) {
                            const int mm = 8 * nt + 2 * t + (i & 1), nn = row0 + 8 * (i >> 1);
                            if (mm < P.M && nn < P.N) P.y[(int64_t)mm * P.ldy + nn] = f8_to_bf16(acc[nt][i] * rs[i >> 1]);
                        }
            } else {  // part of it: the sums into this CTA's first (0) or last (1) slot of parts; the last CTA of the row group adds them
                float* pt = P.parts + (int64_t)(2 * (int)cta + (rg == rg0 ? 0 : 1)) * P.M * 128;
                if (rb < RB)
#pragma unroll
                    for (int nt = 0; nt < NTL; nt++)
#pragma unroll
                        for (int i = 0; i < 4; i++) {
                            const int mm = 8 * nt + 2 * t + (i & 1), r = 16 * warp + g + 8 * (i >> 1);
                            if (mm < P.M) pt[mm * 128 + r] = acc[nt][i] * rs[i >> 1];
                        }
                __threadfence();
                __syncthreads();
                if (tid == 0) {
                    s_last = atomicAdd(P.done + rg, 1) == cl - cf;
                    if (s_last) P.done[rg] = 0;  // ready for the next product
                }
                __syncthreads();
                if (s_last) {
                    __threadfence();
                    const int n0 = (int)rg * 128, wf = (int64_t)cf * W / P.ctas >= i0 ? 0 : 1;  // (CTA cf's slot for it: 0 where its part starts in it; every later CTA's: 0)
                    for (int idx = tid; idx < P.M * 128; idx += LNT) {
                        const int mm = idx >> 7, r = idx & 127;
                        if (n0 + r >= P.N) continue;
                        float s = 0.f;
                        for (int c = cf; c <= cl; c++) s += __ldcg(P.parts + ((int64_t)(2 * c + (c == cf ? wf : 0)) * P.M + mm) * 128 + r);
                        P.y[(int64_t)mm * P.ldy + n0 + r] = f8_to_bf16(s);
                    }
                }
            }
#pragma unroll
            for (int nt = 0; nt < NTL; nt++) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
        }
        step(rgA, kpA);
        step(rgB, kpB);
        step(rgC, kpC);
        step(rgH, kpH);
        step(rgX, kpX);
    }
}

// Layout H, one token (block scales; G: the experts' product, a slot's one pair), on f8_linear_kernel's plan: the CTA's patches, the sums of its first and last row
// groups in parts and the last CTA's adds, so its results to the bit. Each warp copies its own patch's head (two steps ahead, to registers) and its core, literals
// and X (a step ahead, to its own slots), so no barrier a step; the steps' places in 32 bits.
constexpr int XW1 = PC * 2 / 4;                    // words of a warp's X slot: one token's 128 columns
constexpr int L1W = 2 * HCSL + 2 * VB + 2 * XW1;   // words of a warp's slots
// C: the CTAs an SM the compiler fits (3: 80 registers a thread; 2: as many as it likes). Being measured on HBM GPUs (GLYD_F8H_CTAS).
template <int G, int C>
__global__ void __launch_bounds__(LNT, C) f8h_lin1_kernel(const F8Lin P) {
    // (the slot's pointers alone as registers; the rest read from the parameters, which f8_slot leaves as they are)
    const uint8_t* core = P.core;
    const uint32_t* heads = P.heads;
    const float* scale = (const float*)P.scale;
    const uint16_t* x = P.x;
    uint16_t* y = P.y;
    float* parts = P.parts;
    int* done = P.done;
    if constexpr (G != 0) {
        F8Lin Pg;
        if (!f8_slot<F8_BLOCK, true>(P, Pg)) return;
        core = Pg.core, heads = Pg.heads, scale = (const float*)Pg.scale, y = Pg.y, parts = Pg.parts, done = Pg.done;
        x = Pg.order ? P.x + (int64_t)(Pg.order[0] / P.gk) * P.ldx : Pg.x;  // (the pair's token's row)
    }
    const uint32_t* var = P.var;
    F8_EXTERN_SMEM(uint32_t, f8_smem);
    __shared__ int s_last;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3, mb = warp & 3, half = warp >> 2, g1 = g & 1;
    uint32_t* cw = f8_smem + warp * L1W;  // two slots of core, then two of literals, then two of X
    uint32_t* vw = cw + 2 * HCSL;
    uint32_t* xw = vw + 2 * VB;
    const int64_t W = (int64_t)P.RG * P.KT, cta = blockIdx.x;
    const int64_t a0 = cta * W / P.ctas;
    const int n = (int)((cta + 1) * W / P.ctas - a0), KT = P.KT, RP = (int)f8_cdiv(P.N, PR), RB = RP * 4;
    const int rg0 = (int)(a0 / KT), kp0 = (int)(a0 % KT);
    auto step = [&](int& r, int& k) {
        if (++k == KT) k = 0, r++;
    };
    auto patch = [&](int r, int k) { return (2 * r + half) * KT + k; };  // the warp's patch at row group r, column k
    auto mine = [&](int r) { return 2 * r + half < RP; };
    auto head = [&](int r, int k, uint4 (&h)[2]) {
        if (mine(r)) {
            const uint4* s = (const uint4*)(heads + (int64_t)patch(r, k) * HEAD);
            h[0] = __ldg(s), h[1] = __ldg(s + 1);
        } else {
            h[0] = h[1] = make_uint4(0, 0, 0, 0);
        }
    };
    auto x_in = [&](int sl, int k) {  // column k's X into slot sl as f8_linear_kernel's x_in has a token's row (lanes 0-15: 8 columns each)
        if (lane >= 16) return;
        const int kk = k * PC + 8 * lane, i0 = (lane & 3) * 2, i1 = i0 + 1;
        uint8_t* row = (uint8_t*)(xw + sl * XW1);
        uint32_t* d0 = (uint32_t*)(row + (lane >> 2) * 64 + 8 * (2 * (i0 & 3) + (i0 >> 2)));
        uint32_t* d1 = (uint32_t*)(row + (lane >> 2) * 64 + 8 * (2 * (i1 & 3) + (i1 >> 2)));
        const uint16_t* s = x + kk;
        if (P.xvec && kk + 8 <= P.K) {
            f8_cp8(d0, (const uint32_t*)s);
            f8_cp8(d1, (const uint32_t*)(s + 4));
        } else {
            uint32_t v[4];
#pragma unroll
            for (int i = 0; i < 4; i++) v[i] = (kk + 2 * i < P.K ? (uint32_t)s[2 * i] : 0u) | (kk + 2 * i + 1 < P.K ? (uint32_t)s[2 * i + 1] << 16 : 0u);
            d0[0] = v[0], d0[1] = v[1], d1[0] = v[2], d1[1] = v[3];
        }
    };
    auto copies = [&](int sl, int r, int k, const uint4 (&h)[2]) {
        if (mine(r)) {
            f8h_core_in(core, patch(r, k), mb, lane, cw + sl * HCSL);
            f8h_var_in(var, mb, lane, h[0], h[1], vw + sl * VB);
        }
        x_in(sl, k);
    };
    // the cursors: A the step decoded (j), C the step copied (j + 1), H the step whose head comes in (j + 2)
    int rgA = rg0, kpA = kp0, rgC = rg0, kpC = kp0, rgH = rg0, kpH = kp0;
    uint4 hc[2], hn[2] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
    head(rgA, kpA, hc);
    step(rgH, kpH);
    if (n > 1) head(rgH, kpH, hn);
    step(rgH, kpH);
    copies(0, rgC, kpC, hc);
    f8_cp_commit();
    step(rgC, kpC);
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int j = 0; j < n; j++) {
        const int sl = j & 1;
        f8_cp_wait0();
        __syncwarp();  // step j's copies in for the warp, and the warp past step j - 1 (slot sl ^ 1 free)
        uint4 h2[2] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
        if (j + 2 < n) head(rgH, kpH, h2);
        if (mine(rgA)) {
            uint32_t o[4][4];
            f8h_decode(core, var, patch(rgA, kpA), mb, lane, hc[0], hc[1], cw + sl * HCSL, vw + sl * VB, o);
            const uint8_t* xsl = (const uint8_t*)(xw + sl * XW1) + 16 * t;
            float tmp[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int ks = 0; ks < 4; ks++) {
                const uint32_t a0_ = f8_bf_lo(o[ks][0]), a1_ = f8_bf_lo(o[ks][1]), a2_ = f8_bf_lo(o[ks][2]), a3_ = f8_bf_lo(o[ks][3]);
                const uint32_t b0_ = f8_bf_hi(o[ks][0]), b1_ = f8_bf_hi(o[ks][1]), b2_ = f8_bf_hi(o[ks][2]), b3_ = f8_bf_hi(o[ks][3]);
                const uint4 v = *(const uint4*)(xsl + 64 * (ks ^ g1));
                const uint4 xb = make_uint4(f8_x64(v.x), f8_x64(v.y), f8_x64(v.z), f8_x64(v.w));
                f8_mma(tmp, a0_, a1_, a2_, a3_, xb.x, xb.z);
                f8_mma(tmp, b0_, b1_, b2_, b3_, xb.y, xb.w);
            }
            const float s = scale[(int64_t)rgA * P.lds + kpA];
#pragma unroll
            for (int i = 0; i < 4; i++) acc[i] += tmp[i] * s;
        }
        if (j + 1 < n) copies(sl ^ 1, rgC, kpC, hn);  // (after the products: see f8h_lin2_kernel)
        f8_cp_commit();
        if (kpA == KT - 1 || j == n - 1) {  // the end of the CTA's part of row group rgA: its sums out, as f8_linear_kernel's
            const int rb = rgA * LW + warp, row0 = 16 * rb + g;
            const float f = 0x1p56f;  // (the bf16 weights' 2^-120 and X's 2^64)
            const int64_t i0 = (int64_t)rgA * KT;
            const int cf = (int)(((i0 + 1) * P.ctas - 1) / W), cl = (int)(((i0 + KT) * P.ctas - 1) / W);
            if (cf == cl) {
                if (rb < RB && t == 0) {
                    if (row0 < P.N) y[row0] = f8_to_bf16(acc[0] * f);
                    if (row0 + 8 < P.N) y[row0 + 8] = f8_to_bf16(acc[2] * f);
                }
            } else {
                float* pt = parts + (int64_t)(2 * (int)cta + (rgA == rg0 ? 0 : 1)) * 128;
                if (rb < RB && t == 0) pt[16 * warp + g] = acc[0] * f, pt[16 * warp + g + 8] = acc[2] * f;
                __threadfence();
                __syncthreads();
                if (tid == 0) {
                    s_last = atomicAdd(done + rgA, 1) == cl - cf;
                    if (s_last) done[rgA] = 0;  // ready for the next product
                }
                __syncthreads();
                if (s_last) {
                    __threadfence();
                    const int n0 = rgA * 128, wf = (int64_t)cf * W / P.ctas >= i0 ? 0 : 1;
                    for (int r = tid; r < 128; r += LNT) {
                        if (n0 + r >= P.N) continue;
                        float sum = 0.f;
                        for (int c = cf; c <= cl; c++) sum += __ldcg(parts + (int64_t)(2 * c + (c == cf ? wf : 0)) * 128 + r);
                        y[n0 + r] = f8_to_bf16(sum);
                    }
                }
            }
            acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
        }
        hc[0] = hn[0], hc[1] = hn[1], hn[0] = h2[0], hn[1] = h2[1];
        step(rgA, kpA);
        step(rgC, kpC);
        step(rgH, kpH);
    }
}

// Layout H, one token, the copies two steps ahead in the same two core slots: a step's core words go to registers first (14 a lane), then its slot takes the step
// after next's core; a step's literals, X and head have one of three small slots. f8h_lin1_kernel's plan and sums (the same digits); its heads come through shared
// memory, so two heads a warp in registers instead of three.
constexpr int SW2 = VB + XW1 + HEAD;      // words of a small slot: a step's literals, X and head
constexpr int CSL2 = 512;                 // words of a core slot: a coded row block's core (HCSL) or a plain one's 2,048 bytes
constexpr int L2W = 2 * CSL2 + 3 * SW2;   // words of a warp's slots
static_assert(CSL2 >= HCSL && CSL2 * 4 == 2048, "a core slot of layout H's one-token kernel");
template <int G, int C>
__global__ void __launch_bounds__(LNT, C) f8h_lin2_kernel(const F8Lin P) {
    const uint8_t* core = P.core;
    const uint32_t* heads = P.heads;
    const float* scale = (const float*)P.scale;
    const uint16_t* x = P.x;
    uint16_t* y = P.y;
    float* parts = P.parts;
    int* done = P.done;
    if constexpr (G != 0) {
        F8Lin Pg;
        if (!f8_slot<F8_BLOCK, true>(P, Pg)) return;
        core = Pg.core, heads = Pg.heads, scale = (const float*)Pg.scale, y = Pg.y, parts = Pg.parts, done = Pg.done;
        x = Pg.order ? P.x + (int64_t)(Pg.order[0] / P.gk) * P.ldx : Pg.x;  // (the pair's token's row)
    }
    const uint32_t* var = P.var;
    F8_EXTERN_SMEM(uint32_t, f8_smem);
    __shared__ int s_last;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3, mb = warp & 3, half = warp >> 2, g1 = g & 1;
    uint32_t* cw = f8_smem + warp * L2W;  // two core slots (step j's at j & 1), then three small slots (step j's at j % 3)
    uint32_t* sw = cw + 2 * CSL2;
    const int64_t W = (int64_t)P.RG * P.KT, cta = blockIdx.x;
    const int64_t a0 = cta * W / P.ctas;
    const int n = (int)((cta + 1) * W / P.ctas - a0), KT = P.KT, RP = (int)f8_cdiv(P.N, PR), RB = RP * 4;
    const int rg0 = (int)(a0 / KT), kp0 = (int)(a0 % KT);
    auto step = [&](int& r, int& k) {
        if (++k == KT) k = 0, r++;
    };
    auto patch = [&](int r, int k) { return (2 * r + half) * KT + k; };  // the warp's patch at row group r, column k
    auto mine = [&](int r) { return 2 * r + half < RP; };
    auto head = [&](int r, int k, uint4 (&h)[2]) {
        if (mine(r)) {
            const uint4* s = (const uint4*)(heads + (int64_t)patch(r, k) * HEAD);
            h[0] = __ldg(s), h[1] = __ldg(s + 1);
        } else {
            h[0] = h[1] = make_uint4(0, 0, 0, 0);
        }
    };
    auto x_in = [&](uint32_t* xw, int k) {  // column k's X into a small slot's X words as f8_linear_kernel's x_in has a token's row (lanes 0-15: 8 columns each)
        if (lane >= 16) return;
        const int kk = k * PC + 8 * lane, i0 = (lane & 3) * 2, i1 = i0 + 1;
        uint8_t* row = (uint8_t*)xw;
        uint32_t* d0 = (uint32_t*)(row + (lane >> 2) * 64 + 8 * (2 * (i0 & 3) + (i0 >> 2)));
        uint32_t* d1 = (uint32_t*)(row + (lane >> 2) * 64 + 8 * (2 * (i1 & 3) + (i1 >> 2)));
        const uint16_t* s = x + kk;
        if (P.xvec && kk + 8 <= P.K) {
            f8_cp8(d0, (const uint32_t*)s);
            f8_cp8(d1, (const uint32_t*)(s + 4));
        } else {
            uint32_t v[4];
#pragma unroll
            for (int i = 0; i < 4; i++) v[i] = (kk + 2 * i < P.K ? (uint32_t)s[2 * i] : 0u) | (kk + 2 * i + 1 < P.K ? (uint32_t)s[2 * i + 1] << 16 : 0u);
            d0[0] = v[0], d0[1] = v[1], d1[0] = v[2], d1[1] = v[3];
        }
    };
    auto copies = [&](int j2, int r, int k, const uint4 (&h)[2]) {  // step j2's: its core to core slot j2 & 1; its literals, X and head to small slot j2 % 3
        uint32_t* s = sw + (j2 % 3) * SW2;
        if (mine(r)) {
            uint32_t* cd = cw + (j2 & 1) * CSL2;
            if (h[0].w >> 31) {  // plain: the lane's 64 stream bytes (row blocks 0-2 in core, 3 in var), 16 bytes q of the lane at (32 q + lane) * 16 of the slot
                const uint32_t* src = (mb < 3 ? (const uint32_t*)(core + patch(r, k) * HCORE + 2048 * mb) : var + h[0].x) + 16 * lane;
                if (mb < 3 || (h[0].x & 3) == 0) {
#pragma unroll
                    for (int q = 0; q < 4; q++) f8_cp16(cd + (32 * q + lane) * 4, src + 4 * q);
                } else {
#pragma unroll
                    for (int i = 0; i < 16; i++) f8_cp4(cd + (32 * (i >> 2) + lane) * 4 + (i & 3), src + i);
                }
            } else {
                f8h_core_in(core, patch(r, k), mb, lane, cd);
                f8h_var_in(var, mb, lane, h[0], h[1], s);
            }
        }
        x_in(s + VB, k);
        if (lane < 2) ((uint4*)(s + VB + XW1))[lane] = lane ? h[1] : h[0];
    };
    // the cursors: A the step decoded (j), C the step copied (j + 2), H the step whose head comes in (j + 3); hcp the head of the step copied next
    int rgA = rg0, kpA = kp0, rgC = rg0, kpC = kp0, rgH = rg0, kpH = kp0;
    uint4 h0[2], hcp[2] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
    head(rgA, kpA, h0);
    copies(0, rgC, kpC, h0);
    f8_cp_commit();
    step(rgC, kpC);
    step(rgH, kpH);
    if (n > 1) {
        head(rgH, kpH, h0);
        copies(1, rgC, kpC, h0);
    }
    f8_cp_commit();  // (a group a step, empty or not: wait_group 1 below counts groups)
    step(rgC, kpC);
    step(rgH, kpH);
    if (n > 2) head(rgH, kpH, hcp);
    step(rgH, kpH);
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int j = 0; j < n; j++) {
        const uint32_t* cs = cw + (j & 1) * CSL2;
        const uint32_t* ss = sw + (j % 3) * SW2;
        f8_cp_wait1();  // step j's copies in (step j + 1's may still be coming)
        __syncwarp();   // for the whole warp
        const uint4 hA = ((const uint4*)(ss + VB + XW1))[0], hB = ((const uint4*)(ss + VB + XW1))[1];
        const bool mn = mine(rgA), plain = (hA.w >> 31) != 0;
        uint32_t sm[8], cd[7];
        uint4 pv[4];
        if (mn) {
            if (plain) {
#pragma unroll
                for (int q = 0; q < 4; q++) pv[q] = ((const uint4*)cs)[32 * q + lane];
            } else {
                f8h_core_words(cs, lane, sm, cd);
            }
        }
        __syncwarp();  // every lane past its reads of the core slot: step j + 2's core may come in
        uint4 hld[2] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
        if (j + 3 < n) head(rgH, kpH, hld);
        const float s = mn ? scale[(int64_t)rgA * P.lds + kpA] : 0.f;
        if (mn) {
            uint32_t o[4][4];
            if (plain) {  // the lane's 64 stream bytes, from the slot
#pragma unroll
                for (int q = 0; q < 4; q++) o[q][0] = pv[q].x, o[q][1] = pv[q].y, o[q][2] = pv[q].z, o[q][3] = pv[q].w;
            } else {
                uint32_t ls, le, w0;
                f8h_lits(hB, mb, ls, le);
                const uint32_t nw = f8h_words(hB, mb, w0);
                f8h_granule_regs(sm, cd, nw <= (uint32_t)VB ? ss : var + hA.x + w0, ls & 7, hA.y, lane, o);
            }
            const uint8_t* xsl = (const uint8_t*)(ss + VB) + 16 * t;
            float tmp[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int ks = 0; ks < 4; ks++) {
                const uint32_t a0_ = f8_bf_lo(o[ks][0]), a1_ = f8_bf_lo(o[ks][1]), a2_ = f8_bf_lo(o[ks][2]), a3_ = f8_bf_lo(o[ks][3]);
                const uint32_t b0_ = f8_bf_hi(o[ks][0]), b1_ = f8_bf_hi(o[ks][1]), b2_ = f8_bf_hi(o[ks][2]), b3_ = f8_bf_hi(o[ks][3]);
                const uint4 v = *(const uint4*)(xsl + 64 * (ks ^ g1));
                const uint4 xb = make_uint4(f8_x64(v.x), f8_x64(v.y), f8_x64(v.z), f8_x64(v.w));
                f8_mma(tmp, a0_, a1_, a2_, a3_, xb.x, xb.z);
                f8_mma(tmp, b0_, b1_, b2_, b3_, xb.y, xb.w);
            }
#pragma unroll
            for (int i = 0; i < 4; i++) acc[i] += tmp[i] * s;
        }
        // step j + 2's copies after the step's products: a full copy queue then holds up nothing but the next wait (session 8: the order and the
        // mechanism alone read at the card's streaming rate; with the copies before the products the warps queued on them instead of computing)
        if (j + 2 < n) copies(j + 2, rgC, kpC, hcp);
        f8_cp_commit();
        if (kpA == KT - 1 || j == n - 1) {  // the end of the CTA's part of row group rgA: its sums out, as f8_linear_kernel's
            const int rb = rgA * LW + warp, row0 = 16 * rb + g;
            const float f = 0x1p56f;  // (the bf16 weights' 2^-120 and X's 2^64)
            const int64_t i0 = (int64_t)rgA * KT;
            const int cf = (int)(((i0 + 1) * P.ctas - 1) / W), cl = (int)(((i0 + KT) * P.ctas - 1) / W);
            if (cf == cl) {
                if (rb < RB && t == 0) {
                    if (row0 < P.N) y[row0] = f8_to_bf16(acc[0] * f);
                    if (row0 + 8 < P.N) y[row0 + 8] = f8_to_bf16(acc[2] * f);
                }
            } else {
                float* pt = parts + (int64_t)(2 * (int)cta + (rgA == rg0 ? 0 : 1)) * 128;
                if (rb < RB && t == 0) pt[16 * warp + g] = acc[0] * f, pt[16 * warp + g + 8] = acc[2] * f;
                __threadfence();
                __syncthreads();
                if (tid == 0) {
                    s_last = atomicAdd(done + rgA, 1) == cl - cf;
                    if (s_last) done[rgA] = 0;  // ready for the next product
                }
                __syncthreads();
                if (s_last) {
                    __threadfence();
                    const int n0 = rgA * 128, wf = (int64_t)cf * W / P.ctas >= i0 ? 0 : 1;
                    for (int r = tid; r < 128; r += LNT) {
                        if (n0 + r >= P.N) continue;
                        float sum = 0.f;
                        for (int c = cf; c <= cl; c++) sum += __ldcg(parts + (int64_t)(2 * c + (c == cf ? wf : 0)) * 128 + r);
                        y[n0 + r] = f8_to_bf16(sum);
                    }
                }
            }
            acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
        }
        hcp[0] = hld[0], hcp[1] = hld[1];
        step(rgA, kpA);
        step(rgC, kpC);
        step(rgH, kpH);
    }
}

// Plain patches alone, lean (research, 2026-10-20): the one-token product on f8h_lin1_kernel's plan for a pack whose every patch is plain
// (GLYD_F8H_PLAIN=1: row blocks 0-2 of patch p at core + p HCORE, row block 3 at var + the head's x), with nothing a step but its copies, its bf16
// conversion and its 8 MMAs: what the step's structure costs. Two core slots and two X slots a warp; a step's words go to registers, then its slots
// take step j + 2's. The sums as f8h_lin1_kernel's, to the bit. GLYD_F8H_ONE=3 (only with plain packs: a coded patch read this way is wrong).
constexpr int BW = 2 * 512 + 2 * XW1;   // words of a warp's slots
template <int G, int C>
__global__ void __launch_bounds__(LNT, C) f8b_lin1_kernel(const F8Lin P) {
    const uint8_t* core = P.core;
    const uint32_t* heads = P.heads;
    const float* scale = (const float*)P.scale;
    const uint16_t* x = P.x;
    uint16_t* y = P.y;
    float* parts = P.parts;
    int* done = P.done;
    if constexpr (G != 0) {
        F8Lin Pg;
        if (!f8_slot<F8_BLOCK, true>(P, Pg)) return;
        core = Pg.core, heads = Pg.heads, scale = (const float*)Pg.scale, y = Pg.y, parts = Pg.parts, done = Pg.done;
        x = Pg.order ? P.x + (int64_t)(Pg.order[0] / P.gk) * P.ldx : Pg.x;
    }
    const uint32_t* var = P.var;
    F8_EXTERN_SMEM(uint32_t, f8_smem);
    __shared__ int s_last;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3, mb = warp & 3, half = warp >> 2, g1 = g & 1;
    uint32_t* cw = f8_smem + warp * BW;  // two core slots (step j's at j & 1), then two X slots
    uint32_t* xw = cw + 2 * 512;
    const int64_t W = (int64_t)P.RG * P.KT, cta = blockIdx.x;
    const int64_t a0 = cta * W / P.ctas;
    const int n = (int)((cta + 1) * W / P.ctas - a0), KT = P.KT, RP = (int)f8_cdiv(P.N, PR), RB = RP * 4;
    const int rg0 = (int)(a0 / KT), kp0 = (int)(a0 % KT);
    auto step = [&](int& r, int& k) {
        if (++k == KT) k = 0, r++;
    };
    auto copies = [&](int sl, int r, int k) {  // step (r, k)'s row block into core slot sl (16 bytes q of the lane at (32 q + lane) * 16), its X into X slot sl
        if (2 * r + half < RP) {
            const int64_t p = (int64_t)(2 * r + half) * KT + k;
            const uint8_t* src = mb < 3 ? core + p * HCORE + mb * 2048 : (const uint8_t*)(var + __ldg(heads + p * HEAD));
            uint32_t* d = cw + sl * 512;
            // copy q: the warp's 512 contiguous bytes (chunk c = 32 q + lane of 16 bytes; lane c / 4's chunk c % 4), put where the decode's read of
            // lane l's chunk q' (position 4 l + ((q' + l / 2) & 3)) has 8 lanes in 8 bank groups
#pragma unroll
            for (int q = 0; q < 4; q++) {
                const int c = 32 * q + lane, l = c >> 2, qq = c & 3;
                f8_cp16(d + (4 * l + ((qq + (l >> 1)) & 3)) * 4, src + c * 16);
            }
        }
        if (lane < 16) {  // column k's X as f8_linear_kernel's x_in has a token's row (lanes 0-15: 8 columns each)
            const int kk = k * PC + 8 * lane, i0 = (lane & 3) * 2, i1 = i0 + 1;
            uint8_t* row = (uint8_t*)(xw + sl * XW1);
            uint32_t* d0 = (uint32_t*)(row + (lane >> 2) * 64 + 8 * (2 * (i0 & 3) + (i0 >> 2)));
            uint32_t* d1 = (uint32_t*)(row + (lane >> 2) * 64 + 8 * (2 * (i1 & 3) + (i1 >> 2)));
            const uint16_t* sx = x + kk;
            if (P.xvec && kk + 8 <= P.K) {
                f8_cp8(d0, (const uint32_t*)sx);
                f8_cp8(d1, (const uint32_t*)(sx + 4));
            } else {
                uint32_t v[4];
#pragma unroll
                for (int i = 0; i < 4; i++) v[i] = (kk + 2 * i < P.K ? (uint32_t)sx[2 * i] : 0u) | (kk + 2 * i + 1 < P.K ? (uint32_t)sx[2 * i + 1] << 16 : 0u);
                d0[0] = v[0], d0[1] = v[1], d1[0] = v[2], d1[1] = v[3];
            }
        }
    };
    int rgA = rg0, kpA = kp0, rgC = rg0, kpC = kp0;  // A the step decoded (j), C the step copied (j + 2)
    copies(0, rgC, kpC);
    f8_cp_commit();
    step(rgC, kpC);
    if (n > 1) copies(1, rgC, kpC);
    f8_cp_commit();
    step(rgC, kpC);
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int j = 0; j < n; j++) {
        const int sl = j & 1;
        f8_cp_wait1();  // step j's copies in (step j + 1's may still be coming)
        __syncwarp();
        const bool mn = 2 * rgA + half < RP;
        uint4 pv[4], xv[4];
        if (mn) {  // the lane's 64 bytes: chunk q' at position 4 lane + ((q' + lane / 2) & 3) (the copies above)
#pragma unroll
            for (int q = 0; q < 4; q++) pv[q] = ((const uint4*)(cw + sl * 512))[4 * lane + ((q + (lane >> 1)) & 3)];
        }
        {
            const uint8_t* xsl = (const uint8_t*)(xw + sl * XW1) + 16 * t;
#pragma unroll
            for (int ks = 0; ks < 4; ks++) xv[ks] = *(const uint4*)(xsl + 64 * (ks ^ g1));
        }
        __syncwarp();  // every lane past its reads: the slots take step j + 2's
        if (j + 2 < n) copies(sl, rgC, kpC);
        f8_cp_commit();
        if (mn) {
            const float s = scale[(int64_t)rgA * P.lds + kpA];
            float tmp[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int ks = 0; ks < 4; ks++) {
                const uint32_t a0_ = f8_bf_lo(pv[ks].x), a1_ = f8_bf_lo(pv[ks].y), a2_ = f8_bf_lo(pv[ks].z), a3_ = f8_bf_lo(pv[ks].w);
                const uint32_t b0_ = f8_bf_hi(pv[ks].x), b1_ = f8_bf_hi(pv[ks].y), b2_ = f8_bf_hi(pv[ks].z), b3_ = f8_bf_hi(pv[ks].w);
                const uint4 xb = make_uint4(f8_x64(xv[ks].x), f8_x64(xv[ks].y), f8_x64(xv[ks].z), f8_x64(xv[ks].w));
                f8_mma(tmp, a0_, a1_, a2_, a3_, xb.x, xb.z);
                f8_mma(tmp, b0_, b1_, b2_, b3_, xb.y, xb.w);
            }
#pragma unroll
            for (int i = 0; i < 4; i++) acc[i] += tmp[i] * s;
        }
        if (kpA == KT - 1 || j == n - 1) {  // the end of the CTA's part of row group rgA: its sums out, as f8_linear_kernel's
            const int rb = rgA * LW + warp, row0 = 16 * rb + g;
            const float f = 0x1p56f;  // (the bf16 weights' 2^-120 and X's 2^64)
            const int64_t i0 = (int64_t)rgA * KT;
            const int cf = (int)(((i0 + 1) * P.ctas - 1) / W), cl = (int)(((i0 + KT) * P.ctas - 1) / W);
            if (cf == cl) {
                if (rb < RB && t == 0) {
                    if (row0 < P.N) y[row0] = f8_to_bf16(acc[0] * f);
                    if (row0 + 8 < P.N) y[row0 + 8] = f8_to_bf16(acc[2] * f);
                }
            } else {
                float* pt = parts + (int64_t)(2 * (int)cta + (rgA == rg0 ? 0 : 1)) * 128;
                if (rb < RB && t == 0) pt[16 * warp + g] = acc[0] * f, pt[16 * warp + g + 8] = acc[2] * f;
                __threadfence();
                __syncthreads();
                if (tid == 0) {
                    s_last = atomicAdd(done + rgA, 1) == cl - cf;
                    if (s_last) done[rgA] = 0;  // ready for the next product
                }
                __syncthreads();
                if (s_last) {
                    __threadfence();
                    const int n0 = rgA * 128, wf = (int64_t)cf * W / P.ctas >= i0 ? 0 : 1;
                    for (int r = tid; r < 128; r += LNT) {
                        if (n0 + r >= P.N) continue;
                        float sum = 0.f;
                        for (int c = cf; c <= cl; c++) sum += __ldcg(parts + (int64_t)(2 * c + (c == cf ? wf : 0)) * 128 + r);
                        y[n0 + r] = f8_to_bf16(sum);
                    }
                }
            }
            acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
        }
        step(rgA, kpA);
        step(rgC, kpC);
    }
}

// Layout H, one token, lean (research, 2026-10-20): f8h_lin2_kernel's work with f8b_lin1_kernel's step. Two core slots (a step's words to registers, then
// the slot takes step j + 2's core) and two small slots (step j + 1's literals, X and head, filled a step ahead); three heads a warp in registers
// (steps j + 1, j + 2, j + 3: the small copies and the core copy need theirs before their step); X copied by all 32 lanes; no slot of three.
// The same plan and sums as f8h_lin1_kernel. GLYD_F8H_ONE=4.
constexpr int S3W = VB + XW1 + HEAD;      // words of a small slot
constexpr int L3W = 2 * CSL2 + 2 * S3W;   // words of a warp's slots
template <int G, int C>
__global__ void __launch_bounds__(LNT, C) f8h_lin3_kernel(const F8Lin P) {
    const uint8_t* core = P.core;
    const uint32_t* heads = P.heads;
    const float* scale = (const float*)P.scale;
    const uint16_t* x = P.x;
    uint16_t* y = P.y;
    float* parts = P.parts;
    int* done = P.done;
    if constexpr (G != 0) {
        F8Lin Pg;
        if (!f8_slot<F8_BLOCK, true>(P, Pg)) return;
        core = Pg.core, heads = Pg.heads, scale = (const float*)Pg.scale, y = Pg.y, parts = Pg.parts, done = Pg.done;
        x = Pg.order ? P.x + (int64_t)(Pg.order[0] / P.gk) * P.ldx : Pg.x;
    }
    const uint32_t* var = P.var;
    F8_EXTERN_SMEM(uint32_t, f8_smem);
    __shared__ int s_last;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3, mb = warp & 3, half = warp >> 2, g1 = g & 1;
    uint32_t* cw = f8_smem + warp * L3W;  // two core slots (step j's at j & 1), then two small slots (step j's at j & 1)
    uint32_t* sw = cw + 2 * CSL2;
    const int64_t W = (int64_t)P.RG * P.KT, cta = blockIdx.x;
    const int64_t a0 = cta * W / P.ctas;
    const int n = (int)((cta + 1) * W / P.ctas - a0), KT = P.KT, RP = (int)f8_cdiv(P.N, PR), RB = RP * 4;
    const int rg0 = (int)(a0 / KT), kp0 = (int)(a0 % KT);
    auto step = [&](int& r, int& k) {
        if (++k == KT) k = 0, r++;
    };
    auto patch = [&](int r, int k) { return (int64_t)(2 * r + half) * KT + k; };
    auto mine = [&](int r) { return 2 * r + half < RP; };
    auto head = [&](int r, int k, uint4 (&h)[2]) {
        if (mine(r)) {
            const uint4* s = (const uint4*)(heads + patch(r, k) * HEAD);
            h[0] = __ldg(s), h[1] = __ldg(s + 1);
        } else {
            h[0] = h[1] = make_uint4(0, 0, 0, 0);
        }
    };
    auto small = [&](int sl, int r, int k, const uint4 (&h)[2]) {  // step (r, k)'s literals, X and head into small slot sl
        uint32_t* s = sw + sl * S3W;
        if (mine(r)) f8h_var_in(var, mb, lane, h[0], h[1], s);
        {   // column k's X as f8_linear_kernel's x_in has a token's row, 8 bytes a lane (lane 2 l + hf: 16-lane l's bytes hf)
            const int l16 = lane >> 1, hf = lane & 1, kk = k * PC + 8 * l16, i = (l16 & 3) * 2 + hf;
            uint32_t* d = (uint32_t*)((uint8_t*)(s + VB) + (l16 >> 2) * 64 + 8 * (2 * (i & 3) + (i >> 2)));
            const uint16_t* sx = x + kk + 4 * hf;
            if (P.xvec && kk + 8 <= P.K) {
                f8_cp8(d, (const uint32_t*)sx);
            } else {
                const int k0 = kk + 4 * hf;
                d[0] = (k0 < P.K ? (uint32_t)sx[0] : 0u) | (k0 + 1 < P.K ? (uint32_t)sx[1] << 16 : 0u);
                d[1] = (k0 + 2 < P.K ? (uint32_t)sx[2] : 0u) | (k0 + 3 < P.K ? (uint32_t)sx[3] << 16 : 0u);
            }
        }
        if (lane < 2) ((uint4*)(s + VB + XW1))[lane] = lane ? h[1] : h[0];
    };
    auto core_in = [&](int sl, int r, int k, const uint4 (&h)[2]) {  // step (r, k)'s core, or a plain patch's bytes, into core slot sl
        if (!mine(r)) return;
        const int64_t p = patch(r, k);
        uint32_t* d = cw + sl * CSL2;
        if (h[0].w >> 31) {
            const uint8_t* src = mb < 3 ? core + p * HCORE + mb * 2048 : (const uint8_t*)(var + h[0].x);
#pragma unroll
            for (int q = 0; q < 4; q++) {
                const int c = 32 * q + lane, l = c >> 2, qq = c & 3;
                f8_cp16(d + (4 * l + ((qq + (l >> 1)) & 3)) * 4, src + c * 16);
            }
        } else {
            f8h_core_in(core, p, mb, lane, d);
        }
    };
    // cursors: A the step decoded (j), B the step whose small parts are copied (j + 1), D the step whose core is copied (j + 2), H the head read (j + 3)
    int rgA = rg0, kpA = kp0, rgB = rg0, kpB = kp0, rgD = rg0, kpD = kp0, rgH = rg0, kpH = kp0;
    uint4 h1[2], h2[2], h3[2] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
    {
        uint4 h0[2];
        head(rgA, kpA, h0);
        small(0, rgB, kpB, h0);
        core_in(0, rgD, kpD, h0);
        f8_cp_commit();
        step(rgB, kpB);
        step(rgD, kpD);
        step(rgH, kpH);
        if (n > 1) head(rgH, kpH, h1); else h1[0] = h1[1] = make_uint4(0, 0, 0, 0);
        step(rgH, kpH);
        if (n > 1) core_in(1, rgD, kpD, h1);
        f8_cp_commit();
        step(rgD, kpD);
        if (n > 2) head(rgH, kpH, h2); else h2[0] = h2[1] = make_uint4(0, 0, 0, 0);
        step(rgH, kpH);
    }
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int j = 0; j < n; j++) {
        const int sl = j & 1;
        const uint32_t* cs = cw + sl * CSL2;
        const uint32_t* ss = sw + sl * S3W;
        f8_cp_wait1();  // step j's small parts and core in (the newest group, step j + 1's core, may still be coming)
        __syncwarp();
        const uint4 hA = ((const uint4*)(ss + VB + XW1))[0], hB = ((const uint4*)(ss + VB + XW1))[1];
        const bool mn = mine(rgA), plain = (hA.w >> 31) != 0;
        uint32_t sm[8], cd[7];
        uint4 pv[4];
        if (mn) {
            if (plain) {
#pragma unroll
                for (int q = 0; q < 4; q++) pv[q] = ((const uint4*)cs)[4 * lane + ((q + (lane >> 1)) & 3)];
            } else {
                f8h_core_words(cs, lane, sm, cd);
            }
        }
        __syncwarp();  // every lane past its reads of the core slot
        if (j + 3 < n) head(rgH, kpH, h3); else h3[0] = h3[1] = make_uint4(0, 0, 0, 0);
        if (j + 1 < n) small(sl ^ 1, rgB, kpB, h1);
        f8_cp_commit();
        if (j + 2 < n) core_in(sl, rgD, kpD, h2);
        f8_cp_commit();
        const float s = mn ? scale[(int64_t)rgA * P.lds + kpA] : 0.f;
        if (mn) {
            uint32_t o[4][4];
            if (plain) {
#pragma unroll
                for (int q = 0; q < 4; q++) o[q][0] = pv[q].x, o[q][1] = pv[q].y, o[q][2] = pv[q].z, o[q][3] = pv[q].w;
            } else {
                uint32_t ls, le, w0;
                f8h_lits(hB, mb, ls, le);
                const uint32_t nw = f8h_words(hB, mb, w0);
                f8h_granule_regs(sm, cd, nw <= (uint32_t)VB ? ss : var + hA.x + w0, ls & 7, hA.y, lane, o);
            }
            const uint8_t* xsl = (const uint8_t*)(ss + VB) + 16 * t;
            float tmp[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int ks = 0; ks < 4; ks++) {
                const uint32_t a0_ = f8_bf_lo(o[ks][0]), a1_ = f8_bf_lo(o[ks][1]), a2_ = f8_bf_lo(o[ks][2]), a3_ = f8_bf_lo(o[ks][3]);
                const uint32_t b0_ = f8_bf_hi(o[ks][0]), b1_ = f8_bf_hi(o[ks][1]), b2_ = f8_bf_hi(o[ks][2]), b3_ = f8_bf_hi(o[ks][3]);
                const uint4 v = *(const uint4*)(xsl + 64 * (ks ^ g1));
                const uint4 xb = make_uint4(f8_x64(v.x), f8_x64(v.y), f8_x64(v.z), f8_x64(v.w));
                f8_mma(tmp, a0_, a1_, a2_, a3_, xb.x, xb.z);
                f8_mma(tmp, b0_, b1_, b2_, b3_, xb.y, xb.w);
            }
#pragma unroll
            for (int i = 0; i < 4; i++) acc[i] += tmp[i] * s;
        }
        if (kpA == KT - 1 || j == n - 1) {  // the end of the CTA's part of row group rgA: its sums out, as f8_linear_kernel's
            const int rb = rgA * LW + warp, row0 = 16 * rb + g;
            const float f = 0x1p56f;  // (the bf16 weights' 2^-120 and X's 2^64)
            const int64_t i0 = (int64_t)rgA * KT;
            const int cf = (int)(((i0 + 1) * P.ctas - 1) / W), cl = (int)(((i0 + KT) * P.ctas - 1) / W);
            if (cf == cl) {
                if (rb < RB && t == 0) {
                    if (row0 < P.N) y[row0] = f8_to_bf16(acc[0] * f);
                    if (row0 + 8 < P.N) y[row0 + 8] = f8_to_bf16(acc[2] * f);
                }
            } else {
                float* pt = parts + (int64_t)(2 * (int)cta + (rgA == rg0 ? 0 : 1)) * 128;
                if (rb < RB && t == 0) pt[16 * warp + g] = acc[0] * f, pt[16 * warp + g + 8] = acc[2] * f;
                __threadfence();
                __syncthreads();
                if (tid == 0) {
                    s_last = atomicAdd(done + rgA, 1) == cl - cf;
                    if (s_last) done[rgA] = 0;  // ready for the next product
                }
                __syncthreads();
                if (s_last) {
                    __threadfence();
                    const int n0 = rgA * 128, wf = (int64_t)cf * W / P.ctas >= i0 ? 0 : 1;
                    for (int r = tid; r < 128; r += LNT) {
                        if (n0 + r >= P.N) continue;
                        float sum = 0.f;
                        for (int c = cf; c <= cl; c++) sum += __ldcg(parts + (int64_t)(2 * c + (c == cf ? wf : 0)) * 128 + r);
                        y[n0 + r] = f8_to_bf16(sum);
                    }
                }
            }
            acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
        }
        h1[0] = h2[0], h1[1] = h2[1], h2[0] = h3[0], h2[1] = h3[1];
        step(rgA, kpA);
        step(rgB, kpB);
        step(rgD, kpD);
        step(rgH, kpH);
    }
}

}  // namespace f8c
// f8c end

// ---------------------------------------------------------------------------
// The C API: device pointers and sizes in (bf16 as its bits, uint16_t), the
// kernels launched on stream cs of the current device, every output and
// scratch buffer the caller's; 0 back, or a cudaError_t (cudaErrorInvalidValue:
// an argument out of range; cudaErrorNotSupported: not on this GPU). A
// product's workspace: at least the bytes its glyd_gpu_*_workspace gives for
// the same sizes on the same device (none where that is 0). Its `done`
// counters (int32, as many as said): zero before its first call and left zero
// by each, for one stream at a time. The pybind module (after it) and the
// library (build_lib.sh) launch the same kernels with the same arguments.
// glyd_gpu.h declares it for its callers, each function's arguments said.
// GLYD_GPU_API: a function the library exports by its name (every function of glyd_gpu.h, in every build; the loaders find them
// by name). GLYD_GPU_INTERNAL: the same for a function the public header (glyd-gpu-py/include/glyd_gpu.h) does not have.
#ifdef TORCH_EXTENSION_NAME
#define GLYD_GPU_API extern "C" __attribute__((visibility("hidden")))  // the pybind module's own
#define GLYD_GPU_INTERNAL GLYD_GPU_API
#elif defined(_WIN32)
// (exported by the module-definition file build_lib.ps1 writes from this file's own definitions: the header declares each function plainly, and nvcc refuses a definition that adds
// __declspec(dllexport) to a declaration without it)
#define GLYD_GPU_API extern "C"
#define GLYD_GPU_INTERNAL GLYD_GPU_API
#else
#define GLYD_GPU_API extern "C" __attribute__((visibility("default")))
#define GLYD_GPU_INTERNAL GLYD_GPU_API
#endif

static int64_t tiles_for(int64_t n, int64_t tw) { return (n + tw - 1) / tw; }

// Tuning variables (the library's thresholds and sizes, for measuring) are read by a development build and by no release
// build: its library holds no name of one, and takes the values the measurements chose. GLYD_SPLIT_MIN alone is read by both
// (the changelog tells users to set it to turn the route off).
#ifdef GLYD_RELEASE
#define TUNING_ENV(name) ((const char*)nullptr)
#define TUNING_INT(name, fallback) ((int64_t)(fallback))
#else
#define TUNING_ENV(name) std::getenv(name)
static int64_t tuning_int(const char* name, int64_t fallback) {
    const char* v = std::getenv(name);
    return v ? atoll(v) : fallback;
}
#define TUNING_INT(name, fallback) tuning_int(name, fallback)
#endif

#define BY_V(V, CALL) do { if ((V) == 16) { constexpr int VV = 16; CALL; } else { constexpr int VV = 4; CALL; } } while (0)

static int current_device() {
    int dev = 0;
    cudaGetDevice(&dev);
    return dev;
}

static int attribute(cudaDeviceAttr a, int dev) {
    int v = 0;
    cudaDeviceGetAttribute(&v, a, dev);
    return v;
}

static int64_t sm_count(int dev) { return std::max(1, attribute(cudaDevAttrMultiProcessorCount, dev)); }

// A kernel's blocks an SM on device dev with `shared` bytes of dynamic shared
// memory (allowed first), found once a device (known: the kernel's own).
constexpr int MAX_DEVICES = 64;
static int64_t per_sm(const void* kernel, int threads, int shared, std::atomic<int>* known, int dev) {
    int n = dev < MAX_DEVICES ? known[dev].load() : 0;
    if (!n) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kernel, threads, shared);
        n = std::max(n, 1);
        if (dev < MAX_DEVICES) known[dev] = n;
    }
    return n;
}

// Room for need bytes at ws, which has `have`?
static bool fits(const void* ws, size_t have, size_t need) { return !need || (ws && have >= need); }

// Device dev a GeForce Ada (an RTX 40), where the prompts' choices below were measured (an RTX 4080 SUPER); asked
// once a device.
static bool geforce_ada(int dev) {
    static std::atomic<int> known[MAX_DEVICES];  // 1 yes, 2 no
    int k = dev < MAX_DEVICES ? known[dev].load() : 0;
    if (!k) {
        cudaDeviceProp p;
        k = cudaGetDeviceProperties(&p, dev) == cudaSuccess && p.major == 8 && p.minor == 9 && strstr(p.name, "GeForce") ? 1 : 2;
        if (dev < MAX_DEVICES) known[dev] = k;
    }
    return k == 1;
}

static const __nv_bfloat16* bf(const uint16_t* p) { return (const __nv_bfloat16*)p; }
static __nv_bfloat16* bf(uint16_t* p) { return (__nv_bfloat16*)p; }

// The CUDA runtime built in (e.g. 13000), and a status's text.
GLYD_GPU_INTERNAL int glyd_gpu_cuda_version() { return CUDART_VERSION; }
// The C API's version (glyd_gpu.h: one more whenever a function's arguments, or what they must hold, change; the
// caller checks it, as ctypes does not check arguments).
GLYD_GPU_API int glyd_gpu_api_version() { return GLYD_GPU_API_VERSION; }
GLYD_GPU_API const char* glyd_gpu_error_string(int status) {
#ifdef GLYD_RELEASE  // (a release library's text for a status that is not the CUDA runtime's says nothing of the call: the status is the caller's)
    if (status >= GLYD_GPU_BLAS_ERROR) return "a call the library makes failed";
#else
    static const char* blas[] = {"cuBLAS: success", "cuBLAS: not initialized", "cuBLAS status 2", "cuBLAS: allocation failed", "cuBLAS status 4", "cuBLAS status 5", "cuBLAS status 6",
                                 "cuBLAS: invalid value", "cuBLAS: architecture mismatch", "cuBLAS status 9", "cuBLAS status 10", "cuBLAS: mapping error", "cuBLAS status 12",
                                 "cuBLAS: execution failed", "cuBLAS: internal error", "cuBLAS: not supported", "cuBLAS: license error"};
    if (status >= GLYD_GPU_BLAS_ERROR) return status - GLYD_GPU_BLAS_ERROR < 17 ? blas[status - GLYD_GPU_BLAS_ERROR] : "cuBLAS: an error";
#endif
    return cudaGetErrorString((cudaError_t)status);
}

// The dense format. lane_bits: bits [tiles * 32], the tiles of tw weights of w's n.
GLYD_GPU_INTERNAL int glyd_gpu_pass_a(const uint16_t* w, int64_t n, const uint8_t* len, int64_t tw, int64_t V, uint32_t* bits, cudaStream_t cs) {
    if (tw < 1) return cudaErrorInvalidValue;
    int64_t lanes = tiles_for(n, tw) * 32;
    BY_V(V, (lane_bits_kernel<VV><<<(lanes + 255) / 256, 256, 0, cs>>>(w, n, tw, len, bits, lanes)));
    return cudaGetLastError();
}

GLYD_GPU_INTERNAL int glyd_gpu_pass_b(const uint16_t* w, int64_t n, const uint8_t* len, const uint32_t* code, const uint32_t* offs, uint32_t* out, int64_t tw, int64_t V, cudaStream_t cs) {
    if (tw < 1) return cudaErrorInvalidValue;
    int64_t lanes = tiles_for(n, tw) * 32;
    BY_V(V, (write_kernel<VV><<<(lanes + 255) / 256, 256, 0, cs>>>(w, n, tw, len, code, offs, out, lanes)));
    return cudaGetLastError();
}

static void allow_shared(const void* kernel) { cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024); }

// Every tile of the n weights into out [n], or (n_ids > 0) tiles tile_ids into out [n_ids * tw].
GLYD_GPU_INTERNAL int glyd_gpu_decode(const uint8_t* sm, const uint32_t* stream, int64_t stream_words, const uint32_t* offs, const uint32_t* tables, int64_t n, int64_t tw, int64_t V, int64_t tile_words, const int64_t* tile_ids, int64_t n_ids, uint16_t* out, cudaStream_t cs) {
    if (tw < 1) return cudaErrorInvalidValue;
    bool all = n_ids == 0;
    int64_t tiles = all ? tiles_for(n, tw) : n_ids;
    int threads = 128;
    size_t shared = (threads / 32) * tile_words * sizeof(uint32_t);
    BY_V(V, (allow_shared((const void*)decode_kernel<VV>), decode_kernel<VV><<<(tiles * 32 + threads - 1) / threads, threads, shared, cs>>>(sm, stream, stream_words, offs, tables, n, tw, (int)tile_words, all ? nullptr : tile_ids, tiles, out)));
    return cudaGetLastError();
}

// y [O] = W x (+ bias; NULL: none) for W [O, K]; where tiles split rows (tw % K != 0), sum [O] and count [O]
// (zero, and left zero).
GLYD_GPU_INTERNAL int glyd_gpu_gemv(const uint8_t* sm, const uint32_t* stream, int64_t stream_words, const uint32_t* offs, const uint32_t* tables, int64_t O, int64_t K, int64_t tw, int64_t V, int64_t tile_words, const uint16_t* x, const uint16_t* bias, uint16_t* y, float* sum, int* count, cudaStream_t cs) {
    if (K < 1 || V < 1 || tw < 1) return cudaErrorInvalidValue;
    bool split = tw % K != 0;
    if (K % (32 * V) || tw % (32 * V) || (split && !(sum && count))) return cudaErrorInvalidValue;
    int64_t tiles = tiles_for(O * K, tw);
    int threads = 128;
    size_t shared = (threads / 32) * tile_words * sizeof(uint32_t);
    auto launch = [&](auto kernel) {
        allow_shared((const void*)kernel);
        kernel<<<(tiles * 32 + threads - 1) / threads, threads, shared, cs>>>(sm, stream, stream_words, offs, tables, O, K, tw, (int)tile_words, bf(x), bf(bias), bf(y), tiles, split ? sum : nullptr, split ? count : nullptr);
    };
    if (V == 16) { if (split) launch(gemv_kernel<16, true>); else launch(gemv_kernel<16, false>); }
    else { if (split) launch(gemv_kernel<4, true>); else launch(gemv_kernel<4, false>); }
    return cudaGetLastError();
}

// The fast format. y [O] = W x (+ bias) for W [O, K], K a multiple of 128.
GLYD_GPU_INTERNAL int glyd_gpu_fast_gemv(const uint8_t* sm, const uint32_t* planes, const uint8_t* exc, const int32_t* exc_base, uint64_t top, int64_t O, int64_t K, const uint16_t* x, const uint16_t* bias, uint16_t* y, cudaStream_t cs) {
    if (K < 1 || K % 128) return cudaErrorInvalidValue;
    // Warps a row: enough for the GPU to hold ~16k warps of work, at most
    // one a segment and 8 a row.
    int64_t segs = (K + SEG - 1) / SEG;
    int wpr = O >= 16384 ? 1 : (int)std::min<int64_t>(segs, 8);
    int rows_per_block = std::max(1, 8 / wpr), threads = rows_per_block * wpr * 32;
    auto kernel = wpr == 1 ? fast_gemv_kernel<false> : fast_gemv_kernel<true>;
    if (K % 512 == 0) kernel = wpr == 1 ? fast_gemv_wide_kernel<16, false> : fast_gemv_wide_kernel<16, true>;
    else if (K % 256 == 0) kernel = wpr == 1 ? fast_gemv_wide_kernel<8, false> : fast_gemv_wide_kernel<8, true>;
    kernel<<<(O + rows_per_block - 1) / rows_per_block, threads, 0, cs>>>(sm, planes, exc, exc_base, top, O, K, wpr, bf(x), bf(bias), bf(y));
    return cudaGetLastError();
}

// Rows [row0, row0 + rows) of W [., K], or (n_ids > 0) the n_ids rows row_ids, into out [rows, K].
GLYD_GPU_INTERNAL int glyd_gpu_fast_decode(const uint8_t* sm, const uint32_t* planes, const uint8_t* exc, const int32_t* exc_base, uint64_t top, int64_t row0, int64_t rows, const int64_t* row_ids, int64_t n_ids, int64_t K, uint16_t* out, cudaStream_t cs) {
    if (K % 128) return cudaErrorInvalidValue;
    if (n_ids) rows = n_ids;
    int threads = 256;
    int64_t warps = rows * ((K + SEG - 1) / SEG);
    fast_decode_kernel<<<(warps * 32 + threads - 1) / threads, threads, 0, cs>>>(sm, planes, exc, exc_base, top, row0, n_ids ? row_ids : nullptr, rows, K, out);
    return cudaGetLastError();
}

// Y [M, O] = X W^T (+ bias) for X [M, K], K a multiple of 64, O of 16 (a warp's 16 rows all in W or all
// past it: past O a warp would split at its shuffle and hang). K split until the GPU has some `target`
// blocks, in whole steps (kchunk columns a part); the workspace: the parts, [split][M][O] floats, past one.
static int64_t fast_gemm_split(int64_t O, int64_t K, int64_t M, int64_t& kchunk) {
    static int64_t target = TUNING_INT("GLYD_GPU_GEMM_BLOCKS", 1280);
    int64_t bo = (O + GM_BO - 1) / GM_BO, bm = (M + GM_BM - 1) / GM_BM;
    int64_t split = std::max<int64_t>(1, std::min<int64_t>(K / (4 * GM_BK), target / (bo * bm)));
    kchunk = (K / GM_BK + split - 1) / split * GM_BK;
    return (K + kchunk - 1) / kchunk;
}

GLYD_GPU_INTERNAL int glyd_gpu_fast_gemm_workspace(int64_t O, int64_t K, int64_t M, size_t* bytes) {
    if (!bytes || O < 1 || O % 16 || M < 1 || K < 1 || K % GM_BK) return cudaErrorInvalidValue;
    int64_t kchunk, split = fast_gemm_split(O, K, M, kchunk);
    *bytes = split > 1 ? (size_t)(split * M * O) * sizeof(float) : 0;
    return 0;
}

GLYD_GPU_INTERNAL int glyd_gpu_fast_gemm(const uint8_t* sm, const uint32_t* planes, const uint8_t* exc, const int32_t* exc_base, uint64_t top, int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y, void* workspace, size_t workspace_bytes, cudaStream_t cs) {
#if defined(__HIP_PLATFORM_AMD__)
    return cudaErrorNotSupported;  // (fast_gemm_kernel's WMMA: CUDA's alone)
#endif
    size_t need;
    if (glyd_gpu_fast_gemm_workspace(O, K, M, &need) || !fits(workspace, workspace_bytes, need)) return cudaErrorInvalidValue;
    int64_t kchunk, split = fast_gemm_split(O, K, M, kchunk);
    int64_t bo = (O + GM_BO - 1) / GM_BO, bm = (M + GM_BM - 1) / GM_BM;
    float* y32 = split > 1 ? (float*)workspace : nullptr;
    fast_gemm_kernel<<<dim3(bo, bm, split), 128, 0, cs>>>(sm, planes, exc, exc_base, top, O, K, M, kchunk, bf(x), bf(bias), bf(y), y32);
    if (split > 1) finish_kernel<<<(M * O + 255) / 256, 256, 0, cs>>>(y32, split, bf(bias), M, O, bf(y));
    return cudaGetLastError();
}

// Y [M, O] = X W^T (+ bias) for M = 2, 4, 8 or 16 tokens, K a multiple of 512. X's K segment in shared
// memory: up to 48 KB, whole segments of the escapes' index; the workspace: the segments' parts,
// [segments][M][O] floats, past one.
static int64_t bgemv_kseg(int64_t K, int64_t M) { return std::min<int64_t>(K, std::max<int64_t>(SEG, (48 * 1024 / (2 * M)) / SEG * SEG)); }

GLYD_GPU_INTERNAL int glyd_gpu_fast_bgemv_workspace(int64_t O, int64_t K, int64_t M, size_t* bytes) {
    if (!bytes || O < 1 || K < 1 || K % 512 || !(M == 2 || M == 4 || M == 8 || M == 16)) return cudaErrorInvalidValue;
    int64_t kseg = bgemv_kseg(K, M), nseg = (K + kseg - 1) / kseg;
    *bytes = nseg > 1 ? (size_t)(nseg * M * O) * sizeof(float) : 0;
    return 0;
}

GLYD_GPU_INTERNAL int glyd_gpu_fast_bgemv(const uint8_t* sm, const uint32_t* planes, const uint8_t* exc, const int32_t* exc_base, uint64_t top, int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y, void* workspace, size_t workspace_bytes, cudaStream_t cs) {
    size_t need;
    if (glyd_gpu_fast_bgemv_workspace(O, K, M, &need) || !fits(workspace, workspace_bytes, need)) return cudaErrorInvalidValue;
    int64_t kseg = bgemv_kseg(K, M), nseg = (K + kseg - 1) / kseg;
    // Rows a block: enough blocks for every SM (4 an SM), at least 64 rows each.
    int64_t blocks = std::max<int64_t>(1, 320 / nseg);
    int64_t rows = std::max<int64_t>(64, (O + blocks - 1) / blocks);
    dim3 grid((O + rows - 1) / rows, nseg);
    size_t shared = M * kseg * 2;
    float* p32 = nseg > 1 ? (float*)workspace : nullptr;
    auto launch = [&](auto kernel) {
        cudaFuncSetAttribute((const void*)kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
        kernel<<<grid, 256, shared, cs>>>(sm, planes, exc, exc_base, top, O, K, kseg, rows, bf(x), bf(bias), bf(y), p32);
    };
    if (M == 2) launch(fast_bgemv_kernel<2>);
    else if (M == 4) launch(fast_bgemv_kernel<4>);
    else if (M == 8) launch(fast_bgemv_kernel<8>);
    else launch(fast_bgemv_kernel<16>);
    if (nseg > 1) finish_kernel<<<(M * O + 255) / 256, 256, 0, cs>>>(p32, nseg, bf(bias), M, O, bf(y));
    return cudaGetLastError();
}

// A 12-bit pack's words (sym, host): its base hb (0-120) alone, as a save stores it (sym[0] up to 120: no other word is
// read), or as the package passes it: sym[0] the high bytes' base hb in each of its bytes, sym[1-3] zero; false otherwise
// (cudaErrorInvalidValue). A pack of the 12-bit layout before split byte (0.24's, v0.19-v0.24) held its 15 commonest
// exponents there, 4 distinct ones in sym[0]: refused, never decoded as this layout.
static bool nib12(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t* sym, Nib& f) {
    if (!sym) return false;
    const int xmax = (int)std::max<int64_t>(0, std::min<int64_t>(64, TUNING_INT("GLYD_GPU_XMAX", 4)));
    if (sym[0] <= 120) {
        f = Nib{data, exc, exc_base, sym[0] * 0x01010101u, xmax};
        return true;
    }
    uint32_t hb = sym[0] & 0xFFu;
    if (sym[0] != hb * 0x01010101u || hb > 120 || sym[1] || sym[2] || sym[3]) return false;
    f = Nib{data, exc, exc_base, sym[0], xmax};
    return true;
}

// An H100 (an SXM, a PCIe or an NVL card: "H100" in the name as a word): the GPU the experts' products' settings below were measured on
// (Qwen3-30B-A3B's layer and its serve, 2026-10-03). A GH200 and an H200 (the same architecture) take the other GPUs' settings until
// they are measured, as the lossless KV cache's auto does. Asked once a device.
static bool h100_gpu(int dev);  // (defined after gpu_class: it asks for has_word, which stays with the class's functions)
// The A100's architecture (compute capability 8.0: an A100, an A800, an A30): the settings that name it were measured on an A100 SXM4 (Qwen3-30B-A3B's layer, 2026-10-03).
static bool ga100_gpu(int dev) { return attribute(cudaDevAttrComputeCapabilityMajor, dev) == 8 && attribute(cudaDevAttrComputeCapabilityMinor, dev) == 0; }

// The step loop's variants of mma_steps (the small-M product and the experts' products), each by environment (read at each
// call): GLYD_GPU_PF, the steps asked of L2 ahead (0-16; default 0), GLYD_GPU_PATCH_B, the exceptions of a step loaded together
// (0 or 1; default 0); GLYD_GPU_PFD, the same for the dense small-M product (the 12-bit layout's); GLYD_GPU_MOE_ROWS, the experts' product with no gate a warp a row block over the whole K
// (mma_moe_rows_kernel): 0 never, 1 (the default) from 17 tokens on an H100 and an A100 (an A100, Qwen3-30B-A3B at 24-512 tokens: 11-24% off the layer) and from 33 on a GDDR GPU (an L4's 17-32 took 3% longer, 64-232 3-10% less), never on another HBM part until it is measured there, N from N tokens; GLYD_GPU_NA, the experts' products making the m-tiles with pairs in them
// alone (mma_steps' SK; 0 or 1; default 1 on an H100, where it took 3-11% off the layer at 32-512 tokens, and on an A100, 3-9% at 16-512 (Qwen3-30B-A3B's layer, none slower), else 0: an L4's took 3-4% longer at 1 and 232).
struct StepOpts {
    int pf;   // the experts' products' prefetch (GLYD_GPU_PF)
    int pfd;  // the dense small-M product's (GLYD_GPU_PFD)
    bool b;
    int64_t rows;  // GLYD_GPU_MOE_ROWS as set
    bool skip;
};
static StepOpts step_opts() {
    const auto clamp = [](int64_t v) { return (int)std::max<int64_t>(0, std::min<int64_t>(16, v)); };
    return {clamp(TUNING_INT("GLYD_GPU_PF", 0)), clamp(TUNING_INT("GLYD_GPU_PFD", 0)), TUNING_INT("GLYD_GPU_PATCH_B", 0) != 0, TUNING_INT("GLYD_GPU_MOE_ROWS", 1), TUNING_INT("GLYD_GPU_NA", h100_gpu(current_device()) || ga100_gpu(current_device()) ? 1 : 0) != 0};
}

// The mma layouts (tiered: tiers[3]; 12-bit: sym[4], its base). Y [M, O] = X W^T (+ bias) for up to 64 tokens: as
// many blocks as fit at once, but at least 32 steps (4 a warp) a block; the workspace: their parts,
// [blocks + O / 64][M][64] floats; done: O / 64. need: set to the workspace's bytes, nothing launched.
template <class Fmt, int MT, bool B, bool PF>
static int mma_gemm_mt(Fmt f, int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y, void* ws, size_t ws_bytes, int* done, cudaStream_t cs, size_t* need) {
    auto kernel = mma_gemm_kernel<Fmt, MT, B, PF>;
    static std::atomic<int> known[MAX_DEVICES];
    size_t shared = (size_t)4 * MT * 16 * 65 * sizeof(float) + (Fmt::kTable ? 8 * S2_BYTES : 0);
    int dev = current_device();
    int64_t RB = O / 64, total = RB * (K / 16);
    int64_t nb = std::max<int64_t>(1, std::min<int64_t>(per_sm((const void*)kernel, 256, (int)shared, known, dev) * sm_count(dev), total / 32));
    size_t bytes = (size_t)((nb + RB) * M * 64) * sizeof(float);
    if (need) {
        *need = bytes;
        return cudaGetLastError();
    }
    if (!fits(ws, ws_bytes, bytes) || !done) return cudaErrorInvalidValue;
    kernel<<<nb, 256, shared, cs>>>(f, O, K, M, bf(x), bf(bias), bf(y), (float*)ws, done, step_opts().pfd);
    return cudaGetLastError();
}

// The tiered layout's product for 1 to 16 tokens (step_kernel): up to 8 tokens W as the A operand, else X; as many blocks as fit at once,
// at least 32 steps each; the workspace and the counters as mma_gemm_mt's (their parts [blocks + O / 64][M][64] floats; done: O / 64).
// GLYD_GPU_STEP 0: the 8-warp kernel (mma_gemm_kernel) again.
template <bool SW>
static int step_launch(const Tiered& f, int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y, void* ws, size_t ws_bytes, int* done, cudaStream_t cs, size_t* need) {
    auto kernel = step_kernel<SW>;
    static std::atomic<int> known[MAX_DEVICES];
    size_t shared = (size_t)STEP_WARPS * (SW ? 8 : 16) * 65 * sizeof(float) + (size_t)STEP_WARPS * S2_BYTES;
    int dev = current_device();
    int64_t RB = O / 64, total = RB * (K / 16);
    int64_t nb = std::max<int64_t>(1, std::min<int64_t>(per_sm((const void*)kernel, 32 * STEP_WARPS, (int)shared, known, dev) * sm_count(dev), total / 32));
    size_t bytes = (size_t)((nb + RB) * M * 64) * sizeof(float);
    if (need) {
        *need = bytes;
        return cudaGetLastError();
    }
    if (!fits(ws, ws_bytes, bytes) || !done) return cudaErrorInvalidValue;
    kernel<<<nb, 32 * STEP_WARPS, shared, cs>>>(f, O, K, M, bf(x), bf(bias), bf(y), (float*)ws, done);
    return cudaGetLastError();
}

static int tiered_mid_run(const Tiered& f, int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y, void* ws, size_t ws_bytes, int* done, cudaStream_t cs, size_t* need);

// A GPU of compute capability 12 (GeForce and workstation Blackwell: the RTX 50 series, the RTX PRO Blackwell cards, GB10), asked once a
// device: there the route GEMM_CC12 takes the tiered layout's products of a generation step as measured on an RTX 5090 (mma_gemm_run).
static bool cc12_gpu(int dev) {
    static std::atomic<int> known[MAX_DEVICES];  // 1 yes, 2 no
    int k = dev < MAX_DEVICES ? known[dev].load() : 0;
    if (!k) {
        k = attribute(cudaDevAttrComputeCapabilityMajor, dev) == 12 ? 1 : 2;
        if (dev < MAX_DEVICES) known[dev] = k;
    }
    return k == 1;
}

template <class Fmt>
static int mma_gemm_run(Fmt f, int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y, void* ws, size_t ws_bytes, int* done, cudaStream_t cs, size_t* need, [[maybe_unused]] bool cc12_asked = false) {
    if (O % 64 || K % 16 || M < 0 || M > 64) return cudaErrorInvalidValue;
    if constexpr (std::is_same_v<Fmt, Tiered>) {
        // Asked for (the route GEMM_CC12, opt-in until measured end to end), on compute capability 12 the tiered layout's products
        // of 9-32 tokens by the 2-m-tile product, and of 33-64, K a multiple of 64, by tiered_mid_run (a weight decoded by producer
        // warps alone, multiplied by consumer warps), where both were measured faster than the 12-bit layout's products, and the
        // step kernel at 9-16 and the 4-m-tile product at 33-64 slower (an RTX 5090, the matrices of Qwen3.5-9B, Qwen3.5-27B,
        // Gemma-4-12B and Gemma-4-31B: benchmarks/gpu/tiered-speed-2026-10-06). GLYD_GPU_TMT2 and GLYD_GPU_TMID move where they
        // start on any GPU and route (dev builds; 0: never).
        const bool cc12 = cc12_asked && cc12_gpu(current_device());
        const int64_t mid = TUNING_INT("GLYD_GPU_TMID", cc12 ? 33 : 0), mt2 = TUNING_INT("GLYD_GPU_TMT2", cc12 ? 9 : 0);
        if (mid > 0 && M >= mid && K % 64 == 0) {
            if (need) {  // (its workspace, or the 4-m-tile product's where x is not 16-byte aligned: the larger)
                size_t a = 0, b = 0;
                if (int r = tiered_mid_run(f, O, K, x, M, bias, y, ws, ws_bytes, done, cs, &a)) return r;
                if (int r = mma_gemm_mt<Fmt, 4, false, false>(f, O, K, x, M, bias, y, ws, ws_bytes, done, cs, &b)) return r;
                *need = std::max(a, b);
                return cudaGetLastError();
            }
            if ((uintptr_t)x % 16 == 0) return tiered_mid_run(f, O, K, x, M, bias, y, ws, ws_bytes, done, cs, nullptr);
        }
        if (mt2 > 0 && M >= mt2 && M <= 32) return mma_gemm_mt<Fmt, 2, false, false>(f, O, K, x, M, bias, y, ws, ws_bytes, done, cs, need);
        if (M <= 16 && TUNING_INT("GLYD_GPU_STEP", 1)) return (M <= 8 ? step_launch<true> : step_launch<false>)(f, O, K, x, M, bias, y, ws, ws_bytes, done, cs, need);
    }
    // (the tiered layout's loop has neither the batched exception loads nor, on the GPUs it serves, a use for the prefetch: one kernel a size)
    const StepOpts so = step_opts();
    const bool b = !Fmt::kTable && so.b, pf = !Fmt::kTable && so.pfd > 0;
    if (b && pf) {
        auto run = M <= 16 ? mma_gemm_mt<Fmt, 1, true, true> : M <= 32 ? mma_gemm_mt<Fmt, 2, true, true> : mma_gemm_mt<Fmt, 4, true, true>;
        return run(f, O, K, x, M, bias, y, ws, ws_bytes, done, cs, need);
    }
    if (b) {
        auto run = M <= 16 ? mma_gemm_mt<Fmt, 1, true, false> : M <= 32 ? mma_gemm_mt<Fmt, 2, true, false> : mma_gemm_mt<Fmt, 4, true, false>;
        return run(f, O, K, x, M, bias, y, ws, ws_bytes, done, cs, need);
    }
    if (pf) {
        auto run = M <= 16 ? mma_gemm_mt<Fmt, 1, false, true> : M <= 32 ? mma_gemm_mt<Fmt, 2, false, true> : mma_gemm_mt<Fmt, 4, false, true>;
        return run(f, O, K, x, M, bias, y, ws, ws_bytes, done, cs, need);
    }
    auto run = M <= 16 ? mma_gemm_mt<Fmt, 1, false, false> : M <= 32 ? mma_gemm_mt<Fmt, 2, false, false> : mma_gemm_mt<Fmt, 4, false, false>;
    return run(f, O, K, x, M, bias, y, ws, ws_bytes, done, cs, need);
}

// A tiered pack's words (host, tiers[3]): the layout these kernels read (its number, TIERED_TAG, in the top byte of tiers[0]); false
// otherwise (cudaErrorInvalidValue: the layout of v0.27 and before, TieredV1, is unpacked and packed again, never multiplied from).
static bool compact_words(const uint32_t* t, Tiers& ts) {
    if (!t || t[0] >> 24 != COMPACT_TAG) return false;
    ts = Tiers{t[0], t[1], t[2]};
    return true;
}

// A six-bit block pack's words (Int6): its number in the top byte of tiers[0]; K a multiple of 32 (a run of a row a scale) and the codes
// and scales 16-byte aligned, else the pack is not taken (`bad`).
static bool int6_words(const uint32_t* t, int64_t K, const uint8_t* data, const uint8_t* scales, bool& bad) {
    if (!t || t[0] >> 24 != INT6_TAG) return false;
    bad = K % 32 || (uintptr_t)data % 16 || (uintptr_t)scales % 16;
    return true;
}

// A five-bit block pack's words (Int5): its number in the top byte of tiers[0]; K a multiple of 32 (a run of a row a scale) and the codes
// and scales 16-byte aligned, else the pack is not taken (`bad`).
static bool int5_words(const uint32_t* t, int64_t K, const uint8_t* data, const uint8_t* scales, bool& bad) {
    if (!t || t[0] >> 24 != INT5_TAG) return false;
    bad = K % 32 || (uintptr_t)data % 16 || (uintptr_t)scales % 16;
    return true;
}

// A four-bit block pack's words (Int4): as int5_words.
static bool int4_words(const uint32_t* t, int64_t K, const uint8_t* data, const uint8_t* scales, bool& bad) {
    if (!t || t[0] >> 24 != INT4_TAG) return false;
    bad = K % 32 || (uintptr_t)data % 16 || (uintptr_t)scales % 16;
    return true;
}

// An exact FP4 block pack's words (Fp4): as int4_words, its scales 8-byte aligned.
static bool fp4_words(const uint32_t* t, int64_t K, const uint8_t* data, const uint8_t* scales, bool& bad) {
    if (!t || t[0] >> 24 != FP4_TAG) return false;
    bad = K % 32 || (uintptr_t)data % 16 || (uintptr_t)scales % 8;
    return true;
}

// Fp4R's shift: log2(ks) where ks is a power of two, else 0.
// Fp4R's format: n / ks for any 32-bit n as (t + (n - t) / 2) >> sh, t = umulhi(n, mg), with l = ceil(log2 ks), mg = 2^32 (2^l - ks) / ks + 1, sh = l - 1
// (ks is even: K a multiple of 32).
static Fp4R fp4r_fmt(const uint8_t* data, const uint8_t* blocks, int64_t K) {
    const uint64_t ks = (uint64_t)(K / 16);
    uint32_t l = 1;
    while ((1ull << l) < ks) l++;
    return Fp4R{data, blocks, (uint32_t)ks, (uint32_t)((((1ull << l) - ks) << 32) / ks + 1), l - 1};
}

// An exact FP4 pack's words with packed scales (Fp4R): as int4_words.
static bool fp4r_words(const uint32_t* t, int64_t K, const uint8_t* data, const uint8_t* scales, bool& bad) {
    if (!t || t[0] >> 24 != FP4R_TAG) return false;
    bad = K % 32 || K / 16 > 0x7FFFFFFF || (uintptr_t)data % 16 || (uintptr_t)scales % 16;
    return true;
}

// An exact four-bit block pack's words (Int4H): as int4_words, and its scales' base.
static bool int4h_words(const uint32_t* t, int64_t K, const uint8_t* data, const uint8_t* scales, bool& bad, uint32_t& base) {
    if (!t || t[0] >> 24 != INT4H_TAG) return false;
    base = t[1];
    bad = K % 32 || (uintptr_t)data % 16 || (uintptr_t)scales % 16 || base == 0 || base > 239;
    return true;
}

// A three-bit block pack's words (Int3): as int5_words.
static bool int3_words(const uint32_t* t, int64_t K, const uint8_t* data, const uint8_t* scales, bool& bad) {
    if (!t || t[0] >> 24 != INT3_TAG) return false;
    bad = K % 32 || (uintptr_t)data % 16 || (uintptr_t)scales % 16;
    return true;
}

static bool tiered_words(const uint32_t* t, Tiers& ts) {
    if (!t || t[0] >> 24 != TIERED_TAG) return false;
    ts = Tiers{t[0], t[1], t[2]};
    return true;
}

GLYD_GPU_INTERNAL int glyd_gpu_mma_gemm_workspace(int64_t O, int64_t K, int64_t M, size_t* bytes) {
    if (!bytes) return cudaErrorInvalidValue;
    size_t a = 0, b = 0;  // (either tiered layout's pack may come: the larger)
    if (int r = mma_gemm_run(Tiered{}, O, K, nullptr, M, nullptr, nullptr, nullptr, 0, nullptr, 0, &a)) return r;
    if (int r = mma_gemm_run(Compact{}, O, K, nullptr, M, nullptr, nullptr, nullptr, 0, nullptr, 0, &b)) return r;
    size_t c = 0;
    if (int r = mma_gemm_run(Int6{}, O, K, nullptr, M, nullptr, nullptr, nullptr, 0, nullptr, 0, &c)) return r;
    size_t d = 0;
    if (int r = mma_gemm_run(Int5{}, O, K, nullptr, M, nullptr, nullptr, nullptr, 0, nullptr, 0, &d)) return r;
    size_t e = 0;
    if (int r = mma_gemm_run(Int4{}, O, K, nullptr, M, nullptr, nullptr, nullptr, 0, nullptr, 0, &e)) return r;
    size_t f = 0;
    if (int r = mma_gemm_run(Int3{}, O, K, nullptr, M, nullptr, nullptr, nullptr, 0, nullptr, 0, &f)) return r;
    *bytes = std::max({a, b, c, d, e, f});
    return 0;
}

GLYD_GPU_INTERNAL int glyd_gpu_mma_gemm(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base, const uint32_t tiers[3], int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y, void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs) {
    Tiers ts;
    bool bad = false;
    uint32_t base4h = 0;
    if (int6_words(tiers, K, data, blocks, bad)) return bad ? cudaErrorInvalidValue : mma_gemm_run(Int6{data, blocks}, O, K, x, M, bias, y, workspace, workspace_bytes, done, cs, nullptr);
    if (int5_words(tiers, K, data, blocks, bad)) return bad ? cudaErrorInvalidValue : mma_gemm_run(Int5{data, blocks}, O, K, x, M, bias, y, workspace, workspace_bytes, done, cs, nullptr);
    if (int4_words(tiers, K, data, blocks, bad)) return bad ? cudaErrorInvalidValue : mma_gemm_run(Int4{data, blocks}, O, K, x, M, bias, y, workspace, workspace_bytes, done, cs, nullptr);
    if (int4h_words(tiers, K, data, blocks, bad, base4h)) return bad ? cudaErrorInvalidValue : mma_gemm_run(Int4H{{data, blocks}, base4h}, O, K, x, M, bias, y, workspace, workspace_bytes, done, cs, nullptr);
    if (fp4_words(tiers, K, data, blocks, bad)) return bad ? cudaErrorInvalidValue : mma_gemm_run(Fp4{data, blocks}, O, K, x, M, bias, y, workspace, workspace_bytes, done, cs, nullptr);
    if (fp4r_words(tiers, K, data, blocks, bad)) return bad ? cudaErrorInvalidValue : mma_gemm_run(fp4r_fmt(data, blocks, K), O, K, x, M, bias, y, workspace, workspace_bytes, done, cs, nullptr);
    if (int3_words(tiers, K, data, blocks, bad)) return bad ? cudaErrorInvalidValue : mma_gemm_run(Int3{data, blocks}, O, K, x, M, bias, y, workspace, workspace_bytes, done, cs, nullptr);
    if (compact_words(tiers, ts)) return mma_gemm_run(Compact{data, blocks, block_base, ts}, O, K, x, M, bias, y, workspace, workspace_bytes, done, cs, nullptr);
    if (!tiered_words(tiers, ts)) return cudaErrorInvalidValue;
    return mma_gemm_run(Tiered{data, blocks, block_base, ts}, O, K, x, M, bias, y, workspace, workspace_bytes, done, cs, nullptr);
}

GLYD_GPU_INTERNAL int glyd_gpu_mma12_gemm_workspace(int64_t O, int64_t K, int64_t M, size_t* bytes) {
    return bytes ? mma_gemm_run(Nib{}, O, K, nullptr, M, nullptr, nullptr, nullptr, 0, nullptr, 0, bytes) : cudaErrorInvalidValue;
}

GLYD_GPU_INTERNAL int glyd_gpu_mma12_gemm(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4], int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y, void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs) {
    Nib f;
    if (!nib12(data, exc, exc_base, sym, f)) return cudaErrorInvalidValue;
    return mma_gemm_run(f, O, K, x, M, bias, y, workspace, workspace_bytes, done, cs, nullptr);
}

// Many tokens (a prompt), K a multiple of 64, x 16-byte aligned, but on GeForce Ada: mma_gemm_big_kernel, K split
// so the blocks fill their last wave: the fewest splits that leave under 15% of it idle (25% with eight consumer
// warps, whose blocks' parts cost more to add), else the fullest; at least 4 stages a split. The workspace: the
// splits' parts, [splits][M][O] floats, past one.
template <class Fmt, int CW, int PW, int NB, int RBB>
static int mma_gemm_grid_run(Fmt f, int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y, void* ws, size_t ws_bytes, cudaStream_t cs, size_t* need) {
    using C = Big<CW, PW, NB, RBB>;
    auto kernel = mma_gemm_big_kernel<Fmt, CW, PW, NB, RBB>;
    static std::atomic<int> known[MAX_DEVICES];  // a device's blocks an SM for this kernel (its shared memory allowed once)
    int dev = current_device();
    int64_t stages = K / 16 / BIG_KK, pairs = (O / 64 + RBB - 1) / RBB, tiles = (M + C::TM - 1) / C::TM;
    int64_t cap = per_sm((const void*)kernel, C::THREADS, C::SHARED, known, dev) * sm_count(dev), most = std::max<int64_t>(1, stages / 4);
    int64_t splits = 1;
    double best = 0;
    for (int64_t sp = 1; sp <= most; sp++) {
        int64_t nblocks = pairs * tiles * sp;
        double fill = (double)nblocks / (double)(((nblocks + cap - 1) / cap) * cap);
        if (fill > best + 1e-9) best = fill, splits = sp;
        if (fill >= (CW >= 8 ? 0.75 : 0.85)) break;
    }
    int64_t per = (stages + splits - 1) / splits;
    splits = (stages + per - 1) / per;
    size_t bytes = splits > 1 ? (size_t)(splits * M * O) * sizeof(float) : 0;
    if (need) {
        *need = bytes;
        return cudaGetLastError();
    }
    if (!fits(ws, ws_bytes, bytes)) return cudaErrorInvalidValue;
    float* y32 = splits > 1 ? (float*)ws : nullptr;
    kernel<<<dim3(tiles, pairs, splits), C::THREADS, C::SHARED, cs>>>(f, O, K, M, per, bf(x), bf(bias), bf(y), y32, MoePairs{});
    if (splits > 1) finish_kernel<<<(M * O + 255) / 256, 256, 0, cs>>>(y32, splits, bf(bias), M, O, bf(y));
    return cudaGetLastError();
}

// The same on GeForce Ada: mma_gemm_sk_kernel, as many blocks as the GPU holds (at most 8 a unit, at least 4
// stages a block). The workspace: the blocks' slots, where they share a unit; done: a counter a unit (units: tiles
// by pairs), zero.
template <class Fmt, int CW, int PW, int NB, int RBB, int CR = 64, bool LIVE = false>
static int mma_gemm_big_run(Fmt f, int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y, void* ws, size_t ws_bytes, int* done, cudaStream_t cs, size_t* need) {
    using C = Big<CW, PW, NB, RBB, CR>;
    auto kernel = mma_gemm_sk_kernel<Fmt, CW, PW, NB, RBB, CR, LIVE>;
    static std::atomic<int> known[MAX_DEVICES];  // a device's blocks an SM for this kernel (its shared memory allowed once)
    int dev = current_device();
    int64_t S = K / 16 / BIG_KK, units = (M + C::TM - 1) / C::TM * ((O / 64 + RBB - 1) / RBB), total = units * S;
    int64_t nb = std::min(per_sm((const void*)kernel, C::THREADS, C::SHARED, known, dev) * sm_count(dev), std::min(8 * units, total / 4));
    nb = total ? std::max<int64_t>(nb, 1) : 0;
    bool shared = nb && (total % nb || total / nb % S);  // a unit's stages in several blocks
    size_t bytes = shared ? (size_t)(2 * nb * C::TM * 64 * RBB) * sizeof(float) : 0;
    if (need) {
        *need = bytes;
        return cudaGetLastError();
    }
    if (!fits(ws, ws_bytes, bytes) || (shared && !done)) return cudaErrorInvalidValue;
    if (nb) kernel<<<nb, C::THREADS, C::SHARED, cs>>>(f, O, K, M, bf(x), bf(bias), bf(y), (float*)ws, done);
    return cudaGetLastError();
}

// The tiered layout's products of up to 64 tokens by the stream-K kernel (mma_gemm_sk_kernel), x 16-byte aligned: blocks of 64 tokens
// by two row blocks, 8 producer warps (a step each a stage) and 4 consumers (64 tokens by half a row block each), 3 stages; its
// consumers multiply only the m-tiles that hold tokens. (Measured beside it on an RTX 5090: 4 producers, 4 by 4 row blocks, 2 by 1, 8
// by 4 row blocks; and for prompts 8 producers by 8 consumers, 8 consumers of 256 by one or by two row blocks, stream-K or not: slower
// there than this, or than the prompt kernel's own shapes.)
static int tiered_mid_run(const Tiered& f, int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y, void* ws, size_t ws_bytes, int* done, cudaStream_t cs, size_t* need) {
    return mma_gemm_big_run<Tiered, 4, 8, 3, 2, 32, true>(f, O, K, x, M, bias, y, ws, ws_bytes, done, cs, need);
}

template <class Fmt>
static int mma_gemm_big_any(Fmt f, int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y, int64_t variant, void* ws, size_t ws_bytes, int* done, cudaStream_t cs, size_t* need) {
    if (O % 64 || K < 1 || K % 64 || M < 0 || (uintptr_t)x % 16) return cudaErrorInvalidValue;
    // 4 producer warps and 4 consumers (on GeForce Ada 8 consumers of half a row block, but in the tiered layout's
    // blocks of 128: below): blocks of 128 tokens by two row blocks (3 stages; 2 with 8 consumers), or, past 128
    // tokens, of 256 by one (2 stages; a weight decoded once for twice the tokens); on GeForce Ada only where
    // the last block of 256 would be more than half full (else its empty half costs more than the second
    // decode: a Qwen3-4B layer's products take 0.79-0.89x the time in blocks of 128 at 300 tokens, 1.02-1.14x
    // at 448, RTX 4080 SUPER; elsewhere not measured), to 1024 tokens tiered and 4224 12-bit, as measured (past
    // 1024 blocks of 128 cost Qwen3-1.7B's and Qwen3-4B's tiered layers -0.2-1.3% more at 1025-1152 tokens and
    // 1.3-6.9% from 1600, and save their 12-bit layers 0.8-4.5% to 4224). Stream-K likewise on GeForce Ada only,
    // where it was measured. An A100's 12-bit layout the same way to 640 tokens (its Qwen3-8B and Qwen3-14B
    // layers 13-14% faster at 300 and 384 tokens in blocks of 128, 5-15% at 640; GLinear decodes its prompts for
    // cuBLAS from 769), its blocks of 256 tokens by two row blocks and eight consumer warps (variant 3: a weight
    // decoded once for 256 tokens and X's tile read once for 128 rows, less of L2's traffic a product than 256 by
    // one; the 12-bit layout's on any GPU asked for it).
    int dev = current_device();
    bool ada = geforce_ada(dev), a100 = std::is_same_v<Fmt, Nib> && attribute(cudaDevAttrComputeCapabilityMajor, dev) == 8 && attribute(cudaDevAttrComputeCapabilityMinor, dev) == 0;
    int64_t most = a100 ? 640 : std::is_same_v<Fmt, Nib> ? 4224 : 1024;
    if (variant == 0) variant = M > 128 && (M % 256 == 0 || M % 256 > 128 || M > most || !(ada || a100)) ? (a100 ? 3 : 2) : 1;
    if constexpr (std::is_same_v<Fmt, Nib>)
        if (variant == 3) return mma_gemm_grid_run<Fmt, 8, 4, 2, 2>(f, O, K, x, M, bias, y, ws, ws_bytes, cs, need);
    if (!ada) {
        auto grid = variant == 2 ? mma_gemm_grid_run<Fmt, 4, 4, 2, 1> : mma_gemm_grid_run<Fmt, 4, 4, 3, 2>;
        return grid(f, O, K, x, M, bias, y, ws, ws_bytes, cs, need);
    }
    // Consumers of half a row block (two a scheduler) where they were measured the faster, RTX 4080 SUPER: blocks of
    // 256 tokens in both layouts (a Qwen3-1.7B, 4B or 8B layer 2-5% faster 12-bit, 1-2% tiered, at 512-4096 tokens),
    // and of 128 by two row blocks in the 12-bit layout (two stages, as prototyped: 2-4% at 300-896); there the
    // tiered layout's producers, a weight decoded for 128 tokens, do not keep up with them (11-14% slower than with
    // consumers of a row block).
    if (variant == 2) return mma_gemm_big_run<Fmt, 8, 4, 2, 1, 32>(f, O, K, x, M, bias, y, ws, ws_bytes, done, cs, need);
    if constexpr (std::is_same_v<Fmt, Nib>) return mma_gemm_big_run<Fmt, 8, 4, 2, 2, 32>(f, O, K, x, M, bias, y, ws, ws_bytes, done, cs, need);
    else return mma_gemm_big_run<Fmt, 4, 4, 3, 2>(f, O, K, x, M, bias, y, ws, ws_bytes, done, cs, need);
}

GLYD_GPU_INTERNAL int glyd_gpu_mma_gemm_x1_workspace(int64_t O, int64_t K, int64_t M, int64_t variant, size_t* bytes) {
    if (!bytes) return cudaErrorInvalidValue;
    size_t a = 0, b = 0;  // (either tiered layout's pack may come: the larger)
    if (int r = mma_gemm_big_any(Tiered{}, O, K, nullptr, M, nullptr, nullptr, variant, nullptr, 0, nullptr, 0, &a)) return r;
    if (int r = mma_gemm_big_any(Compact{}, O, K, nullptr, M, nullptr, nullptr, variant, nullptr, 0, nullptr, 0, &b)) return r;
    size_t c = 0;
    if (int r = mma_gemm_big_any(Int6{}, O, K, nullptr, M, nullptr, nullptr, variant, nullptr, 0, nullptr, 0, &c)) return r;
    size_t d = 0;
    if (int r = mma_gemm_big_any(Int5{}, O, K, nullptr, M, nullptr, nullptr, variant, nullptr, 0, nullptr, 0, &d)) return r;
    size_t e = 0;
    if (int r = mma_gemm_big_any(Int4{}, O, K, nullptr, M, nullptr, nullptr, variant, nullptr, 0, nullptr, 0, &e)) return r;
    size_t f = 0;
    if (int r = mma_gemm_big_any(Int3{}, O, K, nullptr, M, nullptr, nullptr, variant, nullptr, 0, nullptr, 0, &f)) return r;
    *bytes = std::max({a, b, c, d, e, f});
    return 0;
}

// done: (M + 127) / 128 times O / 64 counters at least, zero (the last block of a unit resets its own; read on
// GeForce Ada, where the product runs by stream-K).
GLYD_GPU_INTERNAL int glyd_gpu_mma_gemm_x1(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base, const uint32_t tiers[3], int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y, int64_t variant, void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs) {
    Tiers ts;
    bool bad = false;
    uint32_t base4h = 0;
    if (int6_words(tiers, K, data, blocks, bad)) return bad ? cudaErrorInvalidValue : mma_gemm_big_any(Int6{data, blocks}, O, K, x, M, bias, y, variant, workspace, workspace_bytes, done, cs, nullptr);
    if (int5_words(tiers, K, data, blocks, bad)) return bad ? cudaErrorInvalidValue : mma_gemm_big_any(Int5{data, blocks}, O, K, x, M, bias, y, variant, workspace, workspace_bytes, done, cs, nullptr);
    if (int4_words(tiers, K, data, blocks, bad)) return bad ? cudaErrorInvalidValue : mma_gemm_big_any(Int4{data, blocks}, O, K, x, M, bias, y, variant, workspace, workspace_bytes, done, cs, nullptr);
    if (int4h_words(tiers, K, data, blocks, bad, base4h)) return bad ? cudaErrorInvalidValue : mma_gemm_big_any(Int4H{{data, blocks}, base4h}, O, K, x, M, bias, y, variant, workspace, workspace_bytes, done, cs, nullptr);
    if (fp4_words(tiers, K, data, blocks, bad)) return bad ? cudaErrorInvalidValue : mma_gemm_big_any(Fp4{data, blocks}, O, K, x, M, bias, y, variant, workspace, workspace_bytes, done, cs, nullptr);
    if (fp4r_words(tiers, K, data, blocks, bad)) return bad ? cudaErrorInvalidValue : mma_gemm_big_any(fp4r_fmt(data, blocks, K), O, K, x, M, bias, y, variant, workspace, workspace_bytes, done, cs, nullptr);
    if (int3_words(tiers, K, data, blocks, bad)) return bad ? cudaErrorInvalidValue : mma_gemm_big_any(Int3{data, blocks}, O, K, x, M, bias, y, variant, workspace, workspace_bytes, done, cs, nullptr);
    if (compact_words(tiers, ts)) return mma_gemm_big_any(Compact{data, blocks, block_base, ts}, O, K, x, M, bias, y, variant, workspace, workspace_bytes, done, cs, nullptr);
    if (!tiered_words(tiers, ts)) return cudaErrorInvalidValue;
    return mma_gemm_big_any(Tiered{data, blocks, block_base, ts}, O, K, x, M, bias, y, variant, workspace, workspace_bytes, done, cs, nullptr);
}

GLYD_GPU_INTERNAL int glyd_gpu_mma12_gemm_x1_workspace(int64_t O, int64_t K, int64_t M, int64_t variant, size_t* bytes) {
    return bytes ? mma_gemm_big_any(Nib{}, O, K, nullptr, M, nullptr, nullptr, variant, nullptr, 0, nullptr, 0, bytes) : cudaErrorInvalidValue;
}

GLYD_GPU_INTERNAL int glyd_gpu_mma12_gemm_x1(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4], int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y, int64_t variant, void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs) {
    Nib f;
    if (!nib12(data, exc, exc_base, sym, f)) return cudaErrorInvalidValue;
    return mma_gemm_big_any(f, O, K, x, M, bias, y, variant, workspace, workspace_bytes, done, cs, nullptr);
}

// Many tokens on Ampere and Ada, the 12-bit layout: 64 tokens a launch (x + 64c on), in the smallest
// tile that holds them. The workspace: the largest launch's slots, two a block, [2 blocks][NT 64 WG] floats
// (one buffer for all: the launches are in stream order); done: O / 64. need: the largest's bytes, nothing launched.
template <int NT, int WG>
static int mma12_mid_run(Nib f, int64_t O, int64_t K, const uint16_t* x, int64_t m0, int64_t M, const uint16_t* bias, uint16_t* y, float* parts, int* done, cudaStream_t cs, size_t* need) {
    using C = Mid12<NT, WG>;
    auto kernel = mma12_mid_kernel<NT, WG>;
    static std::atomic<int> known[MAX_DEVICES];
    int dev = current_device();
    int64_t P = (O / 64 + WG - 1) / WG, U = P * (K / 64);
    int64_t nb = std::max<int64_t>(1, std::min<int64_t>(per_sm((const void*)kernel, C::THREADS, C::SHARED, known, dev) * sm_count(dev), U / 8));
    if (need) *need = std::max(*need, (size_t)(2 * nb * NT * 64 * WG) * sizeof(float));
    else kernel<<<nb, C::THREADS, C::SHARED, cs>>>(f, O, K, M, bf(x) + m0 * K, bf(bias), bf(y) + m0 * O, parts, done);
    return 0;
}

template <int CW, int NB, int NW, int RBB, int MT>
static int mma12_ws_run(Nib f, int64_t O, int64_t K, const uint16_t* x, int64_t m0, int64_t M, const uint16_t* bias, uint16_t* y, float* parts, int* done, cudaStream_t cs, size_t* need) {
    using C = Ws12<CW, NB, NW, RBB, MT>;
    auto kernel = mma12_ws_kernel<CW, NB, NW, RBB, MT>;
    static std::atomic<int> known[MAX_DEVICES];
    int dev = current_device();
    int64_t units = (O / 64 + RBB - 1) / RBB * ((M + C::TM - 1) / C::TM), T = units * (K / 64);
    int64_t nb = std::max<int64_t>(1, std::min<int64_t>(per_sm((const void*)kernel, C::THREADS, C::SHARED, known, dev) * sm_count(dev), T / 8));
    if (need) *need = std::max(*need, (size_t)((nb + units) * C::TM * 64 * RBB) * sizeof(float));
    else kernel<<<nb, C::THREADS, C::SHARED, cs>>>(f, O, K, M, bf(x) + m0 * K, bf(bias), bf(y) + m0 * O, parts, done);
    return 0;
}

static int mma12_mid_any(Nib f, int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y, float* parts, int* done, cudaStream_t cs, size_t* need) {
    int dev = current_device();
    if (attribute(cudaDevAttrComputeCapabilityMajor, dev) < 8) return cudaErrorNotSupported;
    if (O % 64 || K < 1 || K % 64 || M < 0 || (uintptr_t)x % 16 || (uintptr_t)f.data % 16 || (uintptr_t)f.exc % 16) return cudaErrorInvalidValue;
    if (attribute(cudaDevAttrComputeCapabilityMajor, dev) == 8 && attribute(cudaDevAttrComputeCapabilityMinor, dev) == 0) {
        // The A100: producer and consumer warps, units of two row blocks by 32, 64, 96 or 128 tokens, the
        // fewest that hold the step's (65-96: four consumer warps of 48 tokens; 97-128: eight of 32, as four of 64
        // spill their sums); past 128 a launch a 128 (each reading W again: prompts go to mma_gemm_big or are
        // decoded for cuBLAS), so the units and their done counters stay O / 128.
        for (int64_t m0 = 0; m0 < M; m0 += 128) {
            int64_t mc = std::min<int64_t>(128, M - m0);
            auto run = mc <= 32 ? mma12_ws_run<4, 4, 5, 2, 1> : mc <= 64 ? mma12_ws_run<4, 4, 4, 2, 2> : mc <= 96 ? mma12_ws_run<4, 3, 4, 2, 3> : mma12_ws_run<8, 3, 4, 2, 2>;
            run(f, O, K, x, m0, mc, bias, y, parts, done, cs, need);
        }
        return cudaGetLastError();
    }
    bool big = attribute(cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) >= Mid12<64, 4>::SHARED;  // four warpgroups where 153 KB fit
    for (int64_t m0 = 0; m0 < M; m0 += 64) {
        int64_t mc = std::min<int64_t>(64, M - m0);
        auto run = mc <= 16 ? (big ? mma12_mid_run<16, 4> : mma12_mid_run<16, 2>) : mc <= 32 ? (big ? mma12_mid_run<32, 4> : mma12_mid_run<32, 2>) : (big ? mma12_mid_run<64, 4> : mma12_mid_run<64, 2>);
        run(f, O, K, x, m0, mc, bias, y, parts, done, cs, need);
    }
    return cudaGetLastError();
}

GLYD_GPU_INTERNAL int glyd_gpu_mma12_gemm_x2_workspace(int64_t O, int64_t K, int64_t M, size_t* bytes) {
    if (!bytes) return cudaErrorInvalidValue;
    *bytes = 0;
    return mma12_mid_any(Nib{}, O, K, nullptr, M, nullptr, nullptr, nullptr, nullptr, 0, bytes);
}

GLYD_GPU_INTERNAL int glyd_gpu_mma12_gemm_x2(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4], int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y, void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs) {
    Nib f;
    if (!nib12(data, exc, exc_base, sym, f)) return cudaErrorInvalidValue;
    size_t need = 0;
    if (int r = mma12_mid_any(f, O, K, x, M, bias, y, nullptr, nullptr, cs, &need)) return r;
    if (!fits(workspace, workspace_bytes, need) || !done) return cudaErrorInvalidValue;
    return mma12_mid_any(f, O, K, x, M, bias, y, (float*)workspace, done, cs, nullptr);
}

// cuTensorMapEncodeTiled from the driver, found at run time (no link to libcuda); null where it has none. (Hopper's tensor maps, clusters and the
// kernels that take them are CUDA's alone: on AMD their runs below answer not supported.)
#if !defined(__HIP_PLATFORM_AMD__)
static PFN_cuTensorMapEncodeTiled_v12000 tensor_map_encoder() {
    static PFN_cuTensorMapEncodeTiled_v12000 fn = [] {
        void* p = nullptr;
        cudaDriverEntryPointQueryResult q = cudaDriverEntryPointSymbolNotFound;
#if CUDART_VERSION >= 12050
        cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q);
#else
        cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q);
#endif
        return p && q == cudaDriverEntryPointSuccess ? (PFN_cuTensorMapEncodeTiled_v12000)p : nullptr;
    }();
    return fn;
}
#endif

// Steps of 17 to 128 tokens on Hopper, the 12-bit layout, as mma12_mid_run's (the workspace: two slots a block, [2
// blocks][NT 64 WG] floats; done: a counter a unit, O / 64).
template <int NT, int WG>
static int mma12_tma_run(Nib f, int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y, float* parts, int* done, cudaStream_t cs, size_t* need) {
    using C = Tma12<NT, WG>;
    auto kernel = mma12_tma_kernel<NT, WG>;
    static std::atomic<int> known[MAX_DEVICES];  // a device's blocks an SM for this kernel
    int dev = current_device();
    int64_t P = (O / 64 + WG - 1) / WG * ((M + NT - 1) / NT), U = P * (K / 64);  // units: row units by chunks of NT tokens
    // As many blocks as fit at once, but at least 8 stages a block; where that leaves a block under 32 stages
    // and there are fewer units than blocks, a whole number of blocks a unit (each unit's sum over as few parts,
    // its blocks done together) if that idles at most a sixth of the blocks and gives a unit 3 or more. On an H100
    // PCIe Qwen3-8B's o (3 blocks a unit, 96 of 114) is 3-10% the faster at 17-256 tokens, its q, k, v (2) 1-3% the
    // slower at 17-128, Gemma-2-9B's q, k, v (1, 64 of 114) 47% the slower (benchmarks/gpu/h100-prompts-2026-09-28).
    int64_t nb = std::max<int64_t>(1, std::min<int64_t>(per_sm((const void*)kernel, C::THREADS, C::SHARED, known, dev) * sm_count(dev), U / 8));
    int64_t w = nb / P * P;
    if (P < nb && U < 32 * nb && 6 * w >= 5 * nb && w / P >= 3) nb = w;
    if (need) {
        *need = std::max(*need, (size_t)(2 * nb * NT * C::R) * sizeof(float));
        return 0;
    }
#if defined(__HIP_PLATFORM_AMD__)
    return cudaErrorNotSupported;
#else
    // X [M, K] as tiles of NT tokens by 64 columns, 128-byte swizzle; rows past M read as zeros.
    auto encode = tensor_map_encoder();
    if (!encode) return cudaErrorNotSupported;
    CUtensorMap map;
    cuuint64_t dims[2] = {(cuuint64_t)K, (cuuint64_t)M}, strides[1] = {(cuuint64_t)K * 2};
    cuuint32_t box[2] = {64, NT}, unit[2] = {1, 1};
    CUresult r = encode(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)x, dims, strides, box, unit, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) return cudaErrorInvalidValue;
    kernel<<<nb, C::THREADS, C::SHARED, cs>>>(map, f, O, K, M, bf(bias), bf(y), parts, done);
    return 0;
#endif
}

// The tiered layout's products of 1 to 64 tokens on Hopper (tiered_tma_kernel), as mma12_tma_run's: the workspace two
// slots a block, [2 blocks][NT 64 WG] floats; done: a counter a unit (O / 64 at most).
template <int NT, int WG, int BPS, bool SS>
static int tiered_tma_run(const Tiered& f, int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y, float* parts, int* done, cudaStream_t cs, size_t* need) {
    using C = TmaT<NT, WG, BPS>;
    auto kernel = tiered_tma_kernel<NT, WG, BPS, SS>;
    static std::atomic<int> known[MAX_DEVICES];
    int dev = current_device();
    int64_t P = (O / 64 + WG - 1) / WG * ((M + NT - 1) / NT), U = P * (K / 64);
    int64_t nb = std::max<int64_t>(1, std::min<int64_t>(per_sm((const void*)kernel, C::THREADS, C::SHARED, known, dev) * sm_count(dev), U / 8));
    int64_t w = nb / P * P;
    if (P < nb && U < 32 * nb && 6 * w >= 5 * nb && w / P >= 3) nb = w;
    if (need) {
        *need = std::max(*need, (size_t)(2 * nb * NT * C::R) * sizeof(float));
        return 0;
    }
#if defined(__HIP_PLATFORM_AMD__)
    return cudaErrorNotSupported;
#else
    auto encode = tensor_map_encoder();
    if (!encode) return cudaErrorNotSupported;
    CUtensorMap map;
    cuuint64_t dims[2] = {(cuuint64_t)K, (cuuint64_t)M}, strides[1] = {(cuuint64_t)K * 2};
    cuuint32_t box[2] = {64, NT}, unit[2] = {1, 1};
    CUresult r = encode(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)x, dims, strides, box, unit, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) return cudaErrorInvalidValue;
    int bb = (int)std::min<int64_t>(C::BB, TUNING_INT("GLYD_GPU_HT_BB", C::BB));  // (dev builds: less, to check the decode from global memory)
    kernel<<<nb, C::THREADS, C::SHARED, cs>>>(map, f, O, K, M, bf(bias), bf(y), parts, done, bb);
    return 0;
#endif
}

// The tiered layout's route WG (Hopper, 1 to 64 tokens): tiered_tma_kernel, the smallest tile of NT tokens that holds
// M. GLYD_GPU_HT_V (dev builds, measurements): its shape, consumer warpgroups a block by blocks an SM.
static int tiered_wg_any(const Tiered& f, int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y, float* parts, int* done, cudaStream_t cs, size_t* need) {
    int dev = current_device();
    if (attribute(cudaDevAttrComputeCapabilityMajor, dev) != 9 || attribute(cudaDevAttrComputeCapabilityMinor, dev) != 0) return cudaErrorNotSupported;
    if (O < 64 || O % 64 || K < 1 || K % 64 || M < 0 || M > 64 || (uintptr_t)x % 16 || (uintptr_t)f.data % 16 || (uintptr_t)f.blocks % 16) return cudaErrorInvalidValue;
    if (M == 0) return 0;
#define TT_BY_NT(WG, BPS, SS) (M <= 16 ? tiered_tma_run<16, WG, BPS, SS> : M <= 32 ? tiered_tma_run<32, WG, BPS, SS> : tiered_tma_run<64, WG, BPS, SS>)
    int (*run)(const Tiered&, int64_t, int64_t, const uint16_t*, int64_t, const uint16_t*, uint16_t*, float*, int*, cudaStream_t, size_t*) = nullptr;
    switch (TUNING_INT("GLYD_GPU_HT_V", 0)) {
    case 1: run = TT_BY_NT(3, 1, true); break;
    case 2: run = TT_BY_NT(1, 3, true); break;
    case 3: run = TT_BY_NT(1, 4, true); break;
    case 4: run = TT_BY_NT(2, 1, false); break;
    case 5: run = TT_BY_NT(2, 2, true); break;
    default: run = TT_BY_NT(2, 1, true); break;
    }
#undef TT_BY_NT
    if (int r = run(f, O, K, x, M, bias, y, parts, done, cs, need)) return r;
    return need ? 0 : cudaGetLastError();
}

// The route WG's product in the tiered layout as glyd_gpu_mma_linear takes it (staged_run's for it): need, or the call
// with its workspace checked; x not 16-byte aligned (the tensor map's), by the route GEMM (the workspace: the larger).
static int tiered_wg_linear(const Tiered& f, int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y, void* ws, size_t ws_bytes, int* done, cudaStream_t cs, size_t* need) {
    size_t n = 0, n2 = 0;
    if (int r = tiered_wg_any(f, O, K, nullptr, M, bias, y, nullptr, nullptr, cs, &n)) return r;
    if (int r = mma_gemm_run(f, O, K, nullptr, M, nullptr, nullptr, nullptr, 0, nullptr, 0, &n2)) return r;
    n = std::max(n, n2);
    if (need) {
        *need = n;
        return 0;
    }
    if (!fits(ws, ws_bytes, n) || !done) return cudaErrorInvalidValue;
    if ((uintptr_t)x % 16) return mma_gemm_run(f, O, K, x, M, bias, y, ws, ws_bytes, done, cs, nullptr);
    return tiered_wg_any(f, O, K, x, M, bias, y, (float*)ws, done, cs, nullptr);
}

// Prompts on Hopper past 128 tokens: one launch of mma12_wgp_kernel, a block an SM, in clusters of CL (as many as
// the GPU holds at once, at least 8 stages each; or a whole number a tile where the tiles are fewer, below). The
// workspace, where tiles are split: two slots a block that takes part, [2 ncs CL][NT 128] floats; done: a counter a
// split tile a block of its cluster (fewer than the blocks).
template <int NT, int CL>
static int mma12_wgp_run(Nib f, int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y, float* parts, int* done, cudaStream_t cs, size_t* need) {
#if defined(__HIP_PLATFORM_AMD__)
    return cudaErrorNotSupported;
#else
    using C = Wgp12<NT, CL>;
    auto kernel = mma12_wgp_kernel<NT, CL>;
    static std::atomic<int> known[MAX_DEVICES];  // a device's clusters at once for this kernel
    int dev = current_device();
    int64_t T = ((O / 64 + 1) / 2 + CL - 1) / CL * ((M + NT - 1) / NT), S = K / 64;  // cluster tiles: CL row units of 128 rows by chunks of NT tokens
    cudaLaunchConfig_t cfg = {};
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = CL, at[0].val.clusterDim.y = 1, at[0].val.clusterDim.z = 1;
    cfg.blockDim = dim3(C::THREADS), cfg.dynamicSmemBytes = C::SHARED, cfg.stream = cs, cfg.attrs = at, cfg.numAttrs = 1;
    int most = dev < MAX_DEVICES ? known[dev].load() : 0;
    if (!most) {
        cudaFuncSetAttribute((const void*)kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SHARED);
        cfg.gridDim = dim3(CL * sm_count(dev));
        cudaOccupancyMaxActiveClusters(&most, (const void*)kernel, &cfg);
        most = std::max(most, 1);
        if (dev < MAX_DEVICES) known[dev] = most;
    }
    int64_t nc = std::max<int64_t>(1, std::min<int64_t>(most, T * S / 8));
    // Fewer tiles than clusters: a whole number of clusters a tile, where that idles at most a sixth of them (a tile's
    // stages split at the same places, its parts summed over as few), as the TMA kernel's blocks a unit. The tiles left
    // past whole waves, split over ncs clusters: all of them where those are all the tiles, else a whole number a tile
    // (up to SPLIT) where that too idles at most a sixth of them, else up to SPLIT a tile over all. On an H100 SXM (66
    // clusters) whole tiles took 10-13% off Qwen3-8B's o at 129-256 tokens (16 tiles, 4 clusters each), 8-9% at 384-512
    // (2 each), 19% at 1024 (1), 6-10% off 14B's o and 4-7% off 32B's at 129-256 (20 tiles, 3 each), 3-11% off 14B's
    // q, k, v at 129-512 (28 or 56 tiles), and 2% off 8B's q, k, v at 1024 tokens (30 left past the wave: 2 each, 60
    // clusters); 14B's there (46 left: 1 each, 46 clusters) took 3% more, and keeps its split over all 66.
    if (T < nc && 6 * (nc / T * T) >= 5 * nc) nc = nc / T * T;
    int64_t R = T % nc, D = T - R, w = R ? R * std::min<int64_t>(C::SPLIT, nc / R) : 0;
    // ncs <= U, the split stages (R S): so every cluster from a split tile's first to its last has a stage of the tile
    // and arrives on its counter (sum_out_wgp). Past U (only at K = 128) a cluster with none left a tile unwritten.
    int64_t ncs = std::min<int64_t>(R * S, !D ? nc : 6 * w >= 5 * nc ? w : std::min<int64_t>(nc, C::SPLIT * R));
    if (need) {
        *need = std::max(*need, (size_t)(R ? 2 * ncs * CL * NT * 128 : 0) * sizeof(float));
        return 0;
    }
    // X [M, K] as tiles of NT / CL tokens by 64 columns, 128-byte swizzle; rows past M read as zeros.
    auto encode = tensor_map_encoder();
    if (!encode) return cudaErrorNotSupported;
    CUtensorMap map;
    cuuint64_t dims[2] = {(cuuint64_t)K, (cuuint64_t)M}, strides[1] = {(cuuint64_t)K * 2};
    cuuint32_t box[2] = {64, NT / CL}, unit[2] = {1, 1};
    CUresult r = encode(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)x, dims, strides, box, unit, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) return cudaErrorInvalidValue;
    cfg.gridDim = dim3((unsigned)(CL * nc));
    return cudaLaunchKernelEx(&cfg, kernel, map, f, O, K, M, D, (int)ncs, bf(bias), bf(y), parts, done);
#endif
}

static int mma12_wg_any(Nib f, int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y, float* parts, int* done, cudaStream_t cs, size_t* need) {
    // The TMA kernel is sm_90a code alone (the library's compute_80 PTX, compiled for GPUs after Hopper, has none of it).
    int dev = current_device();
    if (attribute(cudaDevAttrComputeCapabilityMajor, dev) != 9 || attribute(cudaDevAttrComputeCapabilityMinor, dev) != 0) return cudaErrorNotSupported;
    // (O at least 64: the partition below divides by its units)
    if (O < 64 || O % 64 || K < 1 || K % 64 || M < 0 || (uintptr_t)x % 16 || (uintptr_t)f.data % 16 || (uintptr_t)f.exc % 16) return cudaErrorInvalidValue;
    // Past 128 tokens mma12_wgp_kernel, its blocks staying, in clusters of 2 (X's tiles copied once for both), in one
    // launch: tiles of 192 tokens where they take no more chunks than 256. Qwen3-8B's, 14B's and 32B's layers on an H100
    // SXM at 256 / 512 / 1024 tokens took 1.30 / 1.51 / 1.46x, 1.28 / 1.44 / 1.33x and 1.19 / 1.39 / 1.35x cuBLAS's
    // time against the TMA kernel's 1.33 / 1.55 / 1.69x, 1.30 / 1.45 / 1.51x and 1.24 / 1.42 / 1.51x. Every layer
    // was faster but 14B's at 129-160 tokens (level); 8B's and 14B's o alone were 6-16% slower here than in the TMA
    // kernel at 129-512 tokens (8B's at all six lengths measured, 14B's at 129-256), a few others 4% at most. Whole
    // tiles (mma12_wgp_run) have since taken 8-13% off 8B's o there and 6-10% off 14B's, as measured in another run.
    if (M > 128) {
        auto run = (M + 191) / 192 == (M + 255) / 256 ? mma12_wgp_run<192, 2> : mma12_wgp_run<256, 2>;
        if (int r = run(f, O, K, x, M, bias, y, parts, done, cs, need)) return r;
        return need ? 0 : cudaGetLastError();
    }
    // To 128 tokens (a step's) the smallest tile that holds them. Two warpgroups (128 rows a stage): with four and the
    // TMA warp (17 warps, 5 on one scheduler) a thread had 96 registers, and ptxas spilled and serialized the products
    // (its C7512).
    auto run = M <= 16 ? mma12_tma_run<16, 2> : M <= 32 ? mma12_tma_run<32, 2> : M <= 64 ? mma12_tma_run<64, 2> : M <= 96 ? mma12_tma_run<96, 2> : M <= 112 ? mma12_tma_run<112, 2> : mma12_tma_run<128, 2>;
    if (M > 0)
        if (int r = run(f, O, K, x, M, bias, y, parts, done, cs, need)) return r;
    return cudaGetLastError();
}

GLYD_GPU_INTERNAL int glyd_gpu_mma12_gemm_x3_workspace(int64_t O, int64_t K, int64_t M, size_t* bytes) {
    if (!bytes) return cudaErrorInvalidValue;
    *bytes = 0;
    return mma12_wg_any(Nib{}, O, K, nullptr, M, nullptr, nullptr, nullptr, nullptr, 0, bytes);
}

GLYD_GPU_INTERNAL int glyd_gpu_mma12_gemm_x3(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4], int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y, void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs) {
    Nib f;
    if (!nib12(data, exc, exc_base, sym, f)) return cudaErrorInvalidValue;
    size_t need = 0;
    if (int r = mma12_wg_any(f, O, K, x, M, bias, y, nullptr, nullptr, cs, &need)) return r;
    if (!fits(workspace, workspace_bytes, need) || !done) return cudaErrorInvalidValue;
    return mma12_wg_any(f, O, K, x, M, bias, y, (float*)workspace, done, cs, nullptr);
}

// The routes: how glyd.gpu multiplies by W [O, K] for M tokens on a GPU, as measured there (gpu/README.md), for
// model.py's GLinear and glyd_gpu_*_linear alike: a step's kernel to 64 tokens (in the 12-bit layout mma_gemm_mid
// from 17 on Ampere and Ada, an A100's to 128, and mma_gemm_wg from 17 to 1024 on Hopper, mma12_wgp_kernel past 128;
// in the tiered layout on Hopper tiered_tma_kernel, the route WG, from 1 to tiered_wg_max(), K a multiple of 64),
// past that the prompt kernel (mma_gemm_big), but where W decoded for a bf16 GEMM (cuBLAS) is the faster (DECODE): a
// 12-bit prompt from GLYD_DEC_MIN tokens where it is set (any GPU), else an A100's from 769 and an L4's from 2560, an
// L4's tiered one from 896 (dec_from), Hopper's past its wgmma kernel's (none on Hopper takes a prompt), and every
// prompt of a matrix whose K is not a multiple of 64 (the prompt kernel's blocks); and with W decoded ahead, beside the
// products before it (AHEAD), whatever K: GeForce Ada's from 513 tokens tiered and 1793 12-bit, an A10's from 512 and
// 640, an L40S's from 1024 and 2048 (ahead_min). GLYD_WG_MIN, GLYD_WG_MAX, GLYD_MID_MIN and GLYD_DEC_MIN move those
// thresholds (read at the first call; GLYD_DEC_MIN 0 or unset: the A100's and the L4's 12-bit ones above). A GPU is
// given as glyd_gpu_gpu gives it: its compute capability, major * 10 + minor, plus its class by name (gpu_class).
struct RouteMins {
    int64_t wg_min, wg_max, mid_min, dec_min;  // (dec_min 0: unset)
    int64_t split_min, split_max, split_sms;   // (0: unset, the GPU's; split_min negative: never)
};

static const RouteMins& route_mins() {
    // a whole number (strtoll's, base 10: spaces, a sign, digits, then spaces alone) within int64, else unset (the
    // package refuses such a value at import, as glyd 0.24 did: model.py's route_env)
    auto get = [](const char* name, int64_t fallback) {
        const char* v = getenv(name);
        if (!v) return fallback;
        char* end;
        errno = 0;
        long long x = strtoll(v, &end, 10);
        bool number = end != v;  // (no digits: end is v)
        while (isspace((unsigned char)*end)) end++;
        return number && !*end && errno != ERANGE ? (int64_t)x : fallback;
    };
#ifdef GLYD_RELEASE
    static const RouteMins t{17, 1024, 17, 0, get("GLYD_SPLIT_MIN", 0), 0, 0};
#else
    static const RouteMins t{get("GLYD_WG_MIN", 17), get("GLYD_WG_MAX", 1024), get("GLYD_MID_MIN", 17), get("GLYD_DEC_MIN", 0),
                             get("GLYD_SPLIT_MIN", 0), get("GLYD_SPLIT_MAX", 0), get("GLYD_SPLIT_SMS", 0)};
#endif
    return t;
}

// A prompt decoded ahead from this many tokens (the fused, not exact, product's): GeForce Ada's (measured on an RTX
// 4080 SUPER) past 512 tiered and 1792 12-bit, an A10's (150 W, full-rate tensor cores: the fused kernel's decode costs
// it clocks at the power cap) from 512 tiered and 640 12-bit, an L40S's (350 W, full-rate tensor cores, the L4's
// bandwidth a FLOP) from 1024 tiered and 2048 12-bit (model.py's GLinear for the measurements); else never.
static int64_t ahead_min(bool twelve, int64_t gpu) {
    if (gpu == GLYD_GPU_L40S + 89) return twelve ? 2048 : 1024;
    return gpu == GLYD_GPU_GEFORCE + 89 ? (twelve ? 1793 : 513) : gpu == GLYD_GPU_A10 + 86 ? (twelve ? 640 : 512) : INT64_MAX;
}

// A prompt decoded for cuBLAS, never fused, from this many tokens: in the 12-bit layout from GLYD_DEC_MIN tokens where
// it is set (any GPU), else an A100's from 769; an L4's from 896 tiered and 2560 12-bit (72 W, full-rate tensor cores,
// half an A10's bandwidth a FLOP: at its power cap the fused kernel, decoding each weight again for every 256 tokens,
// loses to the decode, by more the longer the prompt, and a decode ahead beside cuBLAS gains nothing there; Qwen3-8B's
// and Qwen3-4B-Instruct-2507's prompt passes, fused against decoded: tiered 4-5% slower decoded at 768 tokens, 9-11%
// faster at 896 (the 4B's even at 1024) and 17-47% at 3072-8192; 12-bit 0.4-21% slower to 2304, 1-3% faster at 2560
// and 7-49% at 4096-8192;
// decoded ahead within 1% of decoded first at 4096-8192 tokens, 1-7% slower at 896-2048: benchmarks/gpu/l4-routes-
// 2026-09-29); else never (INT64_MAX, here and in ahead_min: route_for takes no prompt there, whatever its M).
constexpr int64_t L4_TIERED = 896, L4_TWELVE = 2560;
static int64_t dec_from(bool twelve, int64_t gpu, int64_t dec_min) {
    if (twelve && dec_min) return dec_min;
    if (gpu == GLYD_GPU_L4 + 89) return twelve ? L4_TWELVE : L4_TIERED;
    return twelve && gpu % 1000 == 80 ? 769 : INT64_MAX;
}

// Option 2 (the route SPLIT): a 12-bit prompt's matrices decoded ahead on SMs set apart (green contexts), cuBLAS on the
// rest (glyd_gpu_mma12_aux_linear), where a model's forward pass took at least 2% less time than by the routes without
// it, measured end to end against v0.25.1's routes (e2e.py --prefill, the Linears merged, the median of 3 rounds each
// way in turn; benchmarks/gpu/option2-2026-09-29): an A100 SXM's prompts from 769 to 4096 tokens, and to 8192 for a
// matrix whose O and K are both at least 5120 (on an A100-SXM4-40GB: Qwen3-8B's pass 0.876-0.971x at 769-4096, 0.994x
// at 8192, not taken there; 14B's, every matrix O and K at least 5120, 0.845-0.947x, 0.968x at 8192); a GH200's (the
// Hopper measured; its class by name) from 2048 to 8192 tokens for a matrix whose O and K are both at least 5120,
// every matrix of a layer of hidden size 5120 or more with q, k, v and gate, up merged (Qwen3-32B's pass
// 0.909-0.952x; 8B's, hidden size 4096, 0.978x at 2048 and 0.994x at 4096 and
// 8192, not taken: 2.2% at 2048 alone, for a ring of 600 MiB; at 1024 tokens both slower in the first session, 32B's
// 1.012x and 8B's 1.106x); an H100 SXM's by the same rule (its class by name, 132 SMs; measured with Qwen3-14B, every
// matrix of which has O and K at least 5120, SPLIT forced on: its pass 0.893x, 0.937x and 0.938x at 2048, 4096 and 8192
// tokens, benchmarks/gpu/option2-2026-09-29/h100-sxm-measure; Qwen3-32B's matrices are such too and take the route by
// that shape, not run on an H100 SXM). Never on a PCIe card (an A100 PCIe, an H100 PCIe), an H200 or an H100 NVL (not
// measured end to end), nor past 8192 tokens (not measured). Its SMs for the decode: an A100's 12 to 1535 tokens, 8 to
// 3071, then 4; Hopper's 12 to 6143,
// then 4 (the split's granularity there: 8, cuBLAS a co-scheduled group). GLYD_SPLIT_MIN, GLYD_SPLIT_MAX (0 or unset:
// the GPU's; GLYD_SPLIT_MIN negative: never) and GLYD_SPLIT_SMS move them, on any GPU from Ampere and any matrix. The
// route is opt-in: a GPU's code with GLYD_GPU_WITH_SPLIT asks for it (a caller that runs the ring: glyd.gpu's GLinear,
// where the split can run); without the flag no route is SPLIT and glyd_gpu_*_linear's own route never is (v0.25.1's
// routes, for the C API's other callers). The decode is the 12-bit layout's (K a multiple of 64).
static int64_t split_sms(bool twelve, int64_t gpu, int64_t O, int64_t K, int64_t M) {
    const RouteMins& t = route_mins();
    int64_t code = gpu & ~(int64_t)GLYD_GPU_WITH_SPLIT, cc = code % 1000;
    bool a100 = code == 80, hopper = code == GLYD_GPU_GH200 + 90 || code == GLYD_GPU_H100 + 90, large = O >= 5120 && K >= 5120;
    if (!twelve || !(gpu & GLYD_GPU_WITH_SPLIT) || K % 64 || t.split_min < 0 || cc < 80) return 0;
    int64_t lo = t.split_min ? t.split_min : a100 ? 769 : hopper && large ? 2048 : INT64_MAX;
    int64_t hi = t.split_max ? t.split_max : a100 && !large ? 4096 : 8192;
    if (M < lo || M > hi) return 0;
    if (t.split_sms) return t.split_sms;
    if (a100) return M < 1536 ? 12 : M < 3072 ? 8 : 4;
    if (cc == 90) return M < 6144 ? 12 : 4;  // (a GH200's and an H100 SXM's; any Hopper's where GLYD_SPLIT_MIN is set)
    return 12;  // (GLYD_SPLIT_MIN set on another GPU: not measured)
}

// The tiered layout's steps on Hopper by its TMA kernel (the route WG) to this many tokens: none until it is measured (an
// H100 job is ready: benchmarks/gpu/hopper-tiered-2026-10-06); GLYD_GPU_HT_MAX, dev builds, up to 64 (its measurement).
// The route WG given (glyd_gpu_mma_linear's route 3) takes it whatever this says.
static int64_t tiered_wg_max() { return std::min<int64_t>(64, TUNING_INT("GLYD_GPU_HT_MAX", 0)); }

static int route_for(bool twelve, int64_t gpu, int64_t O, int64_t K, int64_t M) {
    const RouteMins& t = route_mins();
    if (split_sms(twelve, gpu, O, K, M)) return GLYD_GPU_ROUTE_SPLIT;
    gpu &= ~(int64_t)GLYD_GPU_WITH_SPLIT;
    int64_t cc = gpu % 1000;
    bool a100 = cc == 80, hopper = cc == 90, mid = cc == 80 || cc == 86 || cc == 87 || cc == 89, k64 = K % 64 == 0;
    if (hopper && twelve && k64 && M >= t.wg_min && M <= t.wg_max) return GLYD_GPU_ROUTE_WG;  // TMA and wgmma
    if (hopper && !twelve && k64 && M >= 1 && M <= tiered_wg_max()) return GLYD_GPU_ROUTE_WG;  // tiered_tma_kernel
    if (mid && twelve && k64 && M >= t.mid_min && M <= (a100 ? 128 : 64)) return GLYD_GPU_ROUTE_MID;  // cp.async, mma.sync
    int64_t dec = dec_from(twelve, gpu, t.dec_min), ahead = ahead_min(twelve, gpu);  // (INT64_MAX: never)
    if (dec != INT64_MAX && M >= dec) return GLYD_GPU_ROUTE_DECODE;
    if (M <= 64) return GLYD_GPU_ROUTE_GEMM;
    if (ahead != INT64_MAX && M >= ahead) return GLYD_GPU_ROUTE_AHEAD;  // (any K)
    return k64 && !hopper ? GLYD_GPU_ROUTE_BIG : GLYD_GPU_ROUTE_DECODE;
}

// A GPU's class by its name, where its compute capability does not tell it apart (glyd_gpu.h): GLYD_GPU_GEFORCE with
// "GeForce" in the name; GLYD_GPU_A10 with "A10" as a word, between characters that are not letters, digits or '_'
// (an A10, not an A10G, A100 or A40: Python's re.search(r"\bA10\b", name, re.ASCII), as model.py's GLinear asks);
// GLYD_GPU_L4 with "L4" as one (an L4, not an L40S or L40); GLYD_GPU_L40S with "L40S" as one (not an L40);
// GLYD_GPU_PCIE with "PCIe" in any case (an A100 PCIe, an H100 PCIe); GLYD_GPU_GH200 with "GH200" as a word;
// GLYD_GPU_H100 with "H100" as a word, unless "NVL" is one (an H100 SXM, "NVIDIA H100 80GB HBM3": not an H100 NVL, a PCIe
// card by its form, nor an H100 PCIe, PCIE's above); else 0.
static bool has_word(const char* name, const char* word) {
    size_t n = strlen(word);
    auto part = [](char c) { return isalnum((unsigned char)c) || c == '_'; };
    for (const char* p = strstr(name, word); p; p = strstr(p + 1, word))
        if ((p == name || !part(p[-1])) && !part(p[n])) return true;
    return false;
}

static bool has_pcie(const char* name) {  // "PCIe" in any case (an A100-PCIE-40GB, an H100 PCIe)
    for (const char* p = name; p[0] && p[1] && p[2] && p[3]; p++)
        if (tolower((unsigned char)p[0]) == 'p' && tolower((unsigned char)p[1]) == 'c' && tolower((unsigned char)p[2]) == 'i' && tolower((unsigned char)p[3]) == 'e') return true;
    return false;
}

static int gpu_class(const char* name) { return strstr(name, "GeForce") ? GLYD_GPU_GEFORCE : has_word(name, "A10") ? GLYD_GPU_A10 : has_word(name, "L4") ? GLYD_GPU_L4 : has_word(name, "L40S") ? GLYD_GPU_L40S : has_pcie(name) ? GLYD_GPU_PCIE : has_word(name, "GH200") ? GLYD_GPU_GH200 : has_word(name, "H100") && !has_word(name, "NVL") ? GLYD_GPU_H100 : 0; }

// The current device as the routes take it (asked once a device).
GLYD_GPU_INTERNAL int glyd_gpu_gpu(int* gpu) {
    static std::atomic<int> known[MAX_DEVICES];  // the code + 1
    if (!gpu) return cudaErrorInvalidValue;
    int dev = current_device(), k = dev < MAX_DEVICES ? known[dev].load() : 0;
    if (!k) {
        cudaDeviceProp p;
        if (cudaError_t e = cudaGetDeviceProperties(&p, dev)) return e;
        k = p.major * 10 + p.minor + gpu_class(p.name) + 1;
        if (dev < MAX_DEVICES) known[dev] = k;
    }
    *gpu = k - 1;
    return 0;
}

static bool h100_gpu(int dev) {
    static std::atomic<int> known[MAX_DEVICES];  // 1 yes, 2 no
    int k = dev < MAX_DEVICES ? known[dev].load() : 0;
    if (!k) {
        cudaDeviceProp p;
        k = cudaGetDeviceProperties(&p, dev) == cudaSuccess && p.major == 9 && p.minor == 0 && has_word(p.name, "H100") ? 1 : 2;
        if (dev < MAX_DEVICES) known[dev] = k;
    }
    return k == 1;
}

// The route for M tokens, and (last) the last token count from M on that takes it: before the first threshold past
// M where another starts.
static int route_run(bool twelve, int64_t gpu, int64_t O, int64_t K, int64_t M, int* route, int64_t* last) {
    if (!route || O < 1 || K < 1 || M < 0) return cudaErrorInvalidValue;
    const RouteMins& t = route_mins();
    int here = *route = route_for(twelve, gpu, O, K, M);
    if (last) {
        int64_t plain = gpu & ~(int64_t)GLYD_GPU_WITH_SPLIT;  // (dec_from and ahead_min take the code without the flag)
        int64_t cuts[] = {t.wg_min, t.wg_max + 1, t.mid_min, 65, 129, dec_from(twelve, plain, t.dec_min), 769, ahead_min(twelve, plain), 512, 513, 640, 1793,
                          1024, 1025, 2048, 4097, 8193, t.split_min, t.split_max + 1, tiered_wg_max() + 1}, next = INT64_MAX;
        for (int64_t c : cuts)
            if (c > M && c < next && route_for(twelve, gpu, O, K, c) != here) next = c;
        *last = next == INT64_MAX ? INT64_MAX : next - 1;
    }
    return 0;
}

GLYD_GPU_INTERNAL int glyd_gpu_mma_way(int64_t gpu, int64_t O, int64_t K, int64_t M, int* route, int64_t* last) {
    return route_run(false, gpu, O, K, M, route, last);
}

GLYD_GPU_INTERNAL int glyd_gpu_mma12_way(int64_t gpu, int64_t O, int64_t K, int64_t M, int* route, int64_t* last) {
    return route_run(true, gpu, O, K, M, route, last);
}

// The 12-bit layout's staged products (mma12_mid_any, mma12_wg_any): need, or the call with its workspace checked.
static int staged_run(int (*any)(Nib, int64_t, int64_t, const uint16_t*, int64_t, const uint16_t*, uint16_t*, float*, int*, cudaStream_t, size_t*), Nib f, int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y, void* ws, size_t ws_bytes, int* done, cudaStream_t cs, size_t* need) {
    size_t n = 0;
    if (int r = any(f, O, K, x, M, bias, y, nullptr, nullptr, cs, &n)) return r;
    if (need) {
        *need = n;
        return 0;
    }
    if (!fits(ws, ws_bytes, n) || !done) return cudaErrorInvalidValue;
    return any(f, O, K, x, M, bias, y, (float*)ws, done, cs, nullptr);
}

// Y [M, O] = X W^T (+ bias) by a route (route_for's; negative: the current GPU's for M): its kernel; DECODE and
// AHEAD, where glyd.gpu decodes W for a GEMM of its own (cuBLAS, which this library does not call), the prompt kernel
// on every GPU (Hopper's too, where glyd.gpu's own prompts take cuBLAS: not measured there); but where K is not a
// multiple of 64 no kernel here takes them (the prompt kernel's blocks): cudaErrorNotSupported, the caller decodes W
// for a GEMM of its own (glyd_gpu.h). need: set to the workspace's bytes, nothing launched.
template <class Fmt>
static int mma_linear_run(Fmt f, int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y, int64_t route, void* ws, size_t ws_bytes, int* done, cudaStream_t cs, size_t* need) {
#if defined(__HIP_PLATFORM_AMD__)
    // AMD, whatever the route: the step product (mma_gemm_run) 32 tokens at a time, one slice's workspace and counters reused in turn on the stream (its
    // 33-64 token tile takes 66 KiB of shared memory, past RDNA's 64 KiB a block, and the prompt kernels' named barriers have no AMD form).
    // ponytail: W is read once a slice of 32 tokens; a prompt kernel on RDNA's WMMA reads it once a prompt.
    (void)route;
    if (need) return mma_gemm_run(f, O, K, x, std::min<int64_t>(M, 32), bias, y, ws, ws_bytes, done, cs, need);
    for (int64_t m0 = 0; m0 < M; m0 += 32)
        if (int r = mma_gemm_run(f, O, K, x + m0 * K, std::min<int64_t>(M - m0, 32), bias, y + m0 * O, ws, ws_bytes, done, cs, nullptr)) return r;
    return 0;
#endif
    constexpr bool twelve = std::is_same_v<Fmt, Nib>;
    int gpu = 0;
    if (int r = glyd_gpu_gpu(&gpu)) return r;
    if (route < 0) route = route_for(twelve, gpu, O, K, M);
    switch (route) {
    case GLYD_GPU_ROUTE_GEMM:
    case GLYD_GPU_ROUTE_GEMM_CC12:
        return mma_gemm_run(f, O, K, x, M, bias, y, ws, ws_bytes, done, cs, need, route == GLYD_GPU_ROUTE_GEMM_CC12);
    case GLYD_GPU_ROUTE_MID:
    case GLYD_GPU_ROUTE_WG:
        if constexpr (twelve) return staged_run(route == GLYD_GPU_ROUTE_MID ? mma12_mid_any : mma12_wg_any, f, O, K, x, M, bias, y, ws, ws_bytes, done, cs, need);
        else if constexpr (std::is_same_v<Fmt, Tiered>) {
            if (route == GLYD_GPU_ROUTE_WG) return tiered_wg_linear(f, O, K, x, M, bias, y, ws, ws_bytes, done, cs, need);  // (to 64 tokens)
        } else if (M <= 64) {
            return mma_gemm_run(f, O, K, x, M, bias, y, ws, ws_bytes, done, cs, need);  // (the compact layout: the step product)
        }
        return cudaErrorInvalidValue;  // MID: the 12-bit layout's alone
    case GLYD_GPU_ROUTE_DECODE:
    case GLYD_GPU_ROUTE_AHEAD:
    case GLYD_GPU_ROUTE_SPLIT:  // (given: the prompt kernel; never its own route. The pipelined product is glyd_gpu_mma12_aux_linear)
        if (K % 64) return cudaErrorNotSupported;
        [[fallthrough]];
    case GLYD_GPU_ROUTE_BIG:
        return mma_gemm_big_any(f, O, K, x, M, bias, y, 0, ws, ws_bytes, done, cs, need);
    }
    return cudaErrorInvalidValue;
}

GLYD_GPU_API int glyd_gpu_mma_linear_workspace(int64_t O, int64_t K, int64_t M, int64_t route, size_t* bytes) {
    if (!bytes) return cudaErrorInvalidValue;
    size_t a = 0, b = 0;  // (either tiered layout's pack may come: the larger)
    if (int r = mma_linear_run(Tiered{}, O, K, nullptr, M, nullptr, nullptr, route, nullptr, 0, nullptr, 0, &a)) return r;
    if (int r = mma_linear_run(Compact{}, O, K, nullptr, M, nullptr, nullptr, route, nullptr, 0, nullptr, 0, &b)) return r;
    size_t c = 0;
    if (int r = mma_linear_run(Int6{}, O, K, nullptr, M, nullptr, nullptr, route, nullptr, 0, nullptr, 0, &c)) return r;
    size_t d = 0;
    if (int r = mma_linear_run(Int5{}, O, K, nullptr, M, nullptr, nullptr, route, nullptr, 0, nullptr, 0, &d)) return r;
    size_t e = 0;
    if (int r = mma_linear_run(Int4{}, O, K, nullptr, M, nullptr, nullptr, route, nullptr, 0, nullptr, 0, &e)) return r;
    size_t f = 0;
    if (int r = mma_linear_run(Int3{}, O, K, nullptr, M, nullptr, nullptr, route, nullptr, 0, nullptr, 0, &f)) return r;
    *bytes = std::max({a, b, c, d, e, f});
    return 0;
}

GLYD_GPU_API int glyd_gpu_mma_linear(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base, const uint32_t tiers[3], int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y, int64_t route, void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs) {
    Tiers ts;
    bool bad = false;
    uint32_t base4h = 0;
    if (int6_words(tiers, K, data, blocks, bad)) return bad ? cudaErrorInvalidValue : mma_linear_run(Int6{data, blocks}, O, K, x, M, bias, y, route, workspace, workspace_bytes, done, cs, nullptr);
    if (int5_words(tiers, K, data, blocks, bad)) return bad ? cudaErrorInvalidValue : mma_linear_run(Int5{data, blocks}, O, K, x, M, bias, y, route, workspace, workspace_bytes, done, cs, nullptr);
    if (int4_words(tiers, K, data, blocks, bad)) return bad ? cudaErrorInvalidValue : mma_linear_run(Int4{data, blocks}, O, K, x, M, bias, y, route, workspace, workspace_bytes, done, cs, nullptr);
    if (int4h_words(tiers, K, data, blocks, bad, base4h)) return bad ? cudaErrorInvalidValue : mma_linear_run(Int4H{{data, blocks}, base4h}, O, K, x, M, bias, y, route, workspace, workspace_bytes, done, cs, nullptr);
    if (fp4_words(tiers, K, data, blocks, bad)) return bad ? cudaErrorInvalidValue : mma_linear_run(Fp4{data, blocks}, O, K, x, M, bias, y, route, workspace, workspace_bytes, done, cs, nullptr);
    if (fp4r_words(tiers, K, data, blocks, bad)) return bad ? cudaErrorInvalidValue : mma_linear_run(fp4r_fmt(data, blocks, K), O, K, x, M, bias, y, route, workspace, workspace_bytes, done, cs, nullptr);
    if (int3_words(tiers, K, data, blocks, bad)) return bad ? cudaErrorInvalidValue : mma_linear_run(Int3{data, blocks}, O, K, x, M, bias, y, route, workspace, workspace_bytes, done, cs, nullptr);
    if (compact_words(tiers, ts)) return mma_linear_run(Compact{data, blocks, block_base, ts}, O, K, x, M, bias, y, route, workspace, workspace_bytes, done, cs, nullptr);
    if (!tiered_words(tiers, ts)) return cudaErrorInvalidValue;
    return mma_linear_run(Tiered{data, blocks, block_base, ts}, O, K, x, M, bias, y, route, workspace, workspace_bytes, done, cs, nullptr);
}

GLYD_GPU_API int glyd_gpu_mma12_linear_workspace(int64_t O, int64_t K, int64_t M, int64_t route, size_t* bytes) {
    return bytes ? mma_linear_run(Nib{}, O, K, nullptr, M, nullptr, nullptr, route, nullptr, 0, nullptr, 0, bytes) : cudaErrorInvalidValue;
}

GLYD_GPU_API int glyd_gpu_mma12_linear(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4], int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y, int64_t route, void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs) {
    Nib f;
    if (!nib12(data, exc, exc_base, sym, f)) return cudaErrorInvalidValue;
    return mma_linear_run(f, O, K, x, M, bias, y, route, workspace, workspace_bytes, done, cs, nullptr);
}

// A stream held for ns nanoseconds, by one thread: a decode ahead launched after it, beside a product that starts
// as it does, starts once the product has placed its blocks (a decode placed first on an SM, in a carveout of its
// own choosing, may leave the product's blocks no room there).
__global__ void hold_kernel(int64_t ns) {
    uint64_t t0, t;
#if defined(__HIP_PLATFORM_AMD__)
    // ponytail: RDNA's wall clock taken as 100 MHz (10 ns a tick, hipDeviceAttributeWallClockRate on RDNA 3); read the rate if a card says otherwise
    t0 = (uint64_t)wall_clock64() * 10;
    do {
        __builtin_amdgcn_s_sleep(8);
        t = (uint64_t)wall_clock64() * 10;
    } while (t - t0 < (uint64_t)ns);
#else
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do {
        __nanosleep(500);
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    } while (t - t0 < (uint64_t)ns);
#endif
}

GLYD_GPU_INTERNAL int glyd_gpu_hold(int64_t ns, cudaStream_t cs) {
    if (ns < 0) return cudaErrorInvalidValue;
    hold_kernel<<<1, 1, 0, cs>>>(ns);
    return cudaGetLastError();
}

// Back to bf16: rows [row0, row0 + rows) of W [., K] (multiples of 64) into out [rows, K]: a warp a step, or (warps
// > 0) that many warps in a block an SM (up to 8 a block), each taking every so many steps: a decode that runs
// beside a product on another stream (a prompt's next matrix), its blocks small enough for an SM to hold beside a
// cuBLAS block. Its SMs keep the most shared memory (the carveout a block leaves them in is theirs until it ends:
// with less, a product's blocks would not fit beside it), a kernel of its own (FEW: a warp a step keeps its code
// and carveout).
template <class Fmt>
static int mma_unpack_run(Fmt f, int64_t K, int64_t row0, int64_t rows, uint16_t* out, int64_t warps, cudaStream_t cs) {
    if (row0 % 64 || rows % 64 || warps < 0) return cudaErrorInvalidValue;
    int64_t steps = rows / 64 * (K / 16);
    if (warps) {
        static std::atomic<int> most[MAX_DEVICES];
        int dev = current_device();
        if (dev >= MAX_DEVICES || !most[dev].exchange(1))
            cudaFuncSetAttribute((const void*)mma_unpack_kernel<Fmt, false, true>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        int64_t blocks = std::min(warps, sm_count(dev)), per = std::min<int64_t>(8, (warps + blocks - 1) / blocks);
        mma_unpack_kernel<Fmt, false, true><<<(unsigned)blocks, (unsigned)(32 * per), 0, cs>>>(f, K, row0, rows, out);
    } else {
        mma_unpack_kernel<Fmt><<<(steps * 32 + 255) / 256, 256, 0, cs>>>(f, K, row0, rows, out);
    }
    return cudaGetLastError();
}

GLYD_GPU_API int glyd_gpu_mma_unpack(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base, const uint32_t tiers[3], int64_t K, int64_t row0, int64_t rows, uint16_t* out, int64_t warps, cudaStream_t cs) {
    if (tiers && tiers[0] >> 24 == TIERED_V1_TAG) return mma_unpack_run(TieredV1{data, blocks, block_base, {tiers[0], tiers[1], tiers[2]}}, K, row0, rows, out, warps, cs);
    Tiers ts;
    bool bad = false;
    uint32_t base4h = 0;
    if (int6_words(tiers, K, data, blocks, bad)) return bad ? cudaErrorInvalidValue : mma_unpack_run(Int6{data, blocks}, K, row0, rows, out, warps, cs);
    if (int5_words(tiers, K, data, blocks, bad)) return bad ? cudaErrorInvalidValue : mma_unpack_run(Int5{data, blocks}, K, row0, rows, out, warps, cs);
    if (int4_words(tiers, K, data, blocks, bad)) return bad ? cudaErrorInvalidValue : mma_unpack_run(Int4{data, blocks}, K, row0, rows, out, warps, cs);
    if (int4h_words(tiers, K, data, blocks, bad, base4h)) return bad ? cudaErrorInvalidValue : mma_unpack_run(Int4H{{data, blocks}, base4h}, K, row0, rows, out, warps, cs);
    if (fp4_words(tiers, K, data, blocks, bad)) return bad ? cudaErrorInvalidValue : mma_unpack_run(Fp4{data, blocks}, K, row0, rows, out, warps, cs);
    if (fp4r_words(tiers, K, data, blocks, bad)) return bad ? cudaErrorInvalidValue : mma_unpack_run(fp4r_fmt(data, blocks, K), K, row0, rows, out, warps, cs);
    if (int3_words(tiers, K, data, blocks, bad)) return bad ? cudaErrorInvalidValue : mma_unpack_run(Int3{data, blocks}, K, row0, rows, out, warps, cs);
    if (compact_words(tiers, ts)) return mma_unpack_run(Compact{data, blocks, block_base, ts}, K, row0, rows, out, warps, cs);
    if (!tiered_words(tiers, ts)) return cudaErrorInvalidValue;
    return mma_unpack_run(Tiered{data, blocks, block_base, ts}, K, row0, rows, out, warps, cs);
}

GLYD_GPU_API int glyd_gpu_mma12_unpack(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4], int64_t K, int64_t row0, int64_t rows, uint16_t* out, int64_t warps, cudaStream_t cs) {
    Nib f;
    if (!nib12(data, exc, exc_base, sym, f)) return cudaErrorInvalidValue;
    return mma_unpack_run(f, K, row0, rows, out, warps, cs);
}

// Rows ids[0, n) (int64, on the device) of W [rows, K] into out [n, K], by mma_rows_kernel.
template <class Fmt>
static int mma_rows_run(Fmt f, int64_t K, int64_t rows, const int64_t* ids, int64_t n, uint16_t* out, cudaStream_t cs) {
    if (K < 16 || K % 16 || rows < 0 || rows % 64 || n < 0 || ((!ids || !out) && n)) return cudaErrorInvalidValue;
    int64_t warps = n * (K / 16);
    if (!warps) return 0;
    if ((warps + 7) / 8 >= (int64_t)1 << 31) return cudaErrorInvalidValue;
    mma_rows_kernel<Fmt><<<(unsigned)((warps + 7) / 8), 256, 0, cs>>>(f, K, rows, ids, n, out);
    return cudaGetLastError();
}

GLYD_GPU_INTERNAL int glyd_gpu_mma_rows(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base, const uint32_t tiers[3], int64_t K, int64_t rows, const int64_t* ids, int64_t n, uint16_t* out, cudaStream_t cs) {
    Tiers ts;
    bool bad = false;
    uint32_t base4h = 0;
    if (int6_words(tiers, K, data, blocks, bad)) return bad ? cudaErrorInvalidValue : mma_rows_run(Int6{data, blocks}, K, rows, ids, n, out, cs);
    if (int5_words(tiers, K, data, blocks, bad)) return bad ? cudaErrorInvalidValue : mma_rows_run(Int5{data, blocks}, K, rows, ids, n, out, cs);
    if (int4_words(tiers, K, data, blocks, bad)) return bad ? cudaErrorInvalidValue : mma_rows_run(Int4{data, blocks}, K, rows, ids, n, out, cs);
    if (int4h_words(tiers, K, data, blocks, bad, base4h)) return bad ? cudaErrorInvalidValue : mma_rows_run(Int4H{{data, blocks}, base4h}, K, rows, ids, n, out, cs);
    if (fp4_words(tiers, K, data, blocks, bad)) return bad ? cudaErrorInvalidValue : mma_rows_run(Fp4{data, blocks}, K, rows, ids, n, out, cs);
    if (fp4r_words(tiers, K, data, blocks, bad)) return bad ? cudaErrorInvalidValue : mma_rows_run(fp4r_fmt(data, blocks, K), K, rows, ids, n, out, cs);
    if (int3_words(tiers, K, data, blocks, bad)) return bad ? cudaErrorInvalidValue : mma_rows_run(Int3{data, blocks}, K, rows, ids, n, out, cs);
    if (compact_words(tiers, ts)) return mma_rows_run(Compact{data, blocks, block_base, ts}, K, rows, ids, n, out, cs);
    if (!tiered_words(tiers, ts)) return cudaErrorInvalidValue;  // (the layout of v0.27 and before: unpacked and packed again, never read by a product)
    return mma_rows_run(Tiered{data, blocks, block_base, ts}, K, rows, ids, n, out, cs);
}

GLYD_GPU_INTERNAL int glyd_gpu_mma12_rows(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4], int64_t K, int64_t rows, const int64_t* ids, int64_t n, uint16_t* out, cudaStream_t cs) {
    Nib f;
    if (!nib12(data, exc, exc_base, sym, f)) return cudaErrorInvalidValue;
    return mma_rows_run(f, K, rows, ids, n, out, cs);
}

// ---------------------------------------------------------------------------
// The route SPLIT (option 2): route_for's rule above; here its decode on a few SMs (mma12_split_kernel), the SMs set
// apart for it by the driver's green contexts, and the ring its matrices are decoded ahead into while the caller's
// cuBLAS multiplies from it on the other SMs (glyd_gpu.h).

// The decode for sms SMs: as many blocks of 4 warps an SM as fit (3 on an A100, its registers), on stream cs.
static int split_decode(Nib f, int64_t K, int64_t row0, int64_t rows, uint16_t* out, int64_t sms, cudaStream_t cs) {
    if (K < 64 || K % 64 || row0 < 0 || row0 % 64 || rows < 64 || rows % 64 || sms < 1 || sms > 1024 || (uintptr_t)out % 16) return cudaErrorInvalidValue;
    if (rows / 64 * (K / 64) + sms * 64 * SPLIT_WARPS >= (int64_t)1 << 31) return cudaErrorInvalidValue;  // (the kernel's units in 32 bits)
    static std::atomic<int> known[MAX_DEVICES];
    int64_t per = per_sm((const void*)mma12_split_kernel, 32 * SPLIT_WARPS, 0, known, current_device());
    mma12_split_kernel<<<(unsigned)(sms * per), 32 * SPLIT_WARPS, 0, cs>>>(f, K, row0, rows, out);
    return cudaGetLastError();
}

GLYD_GPU_INTERNAL int glyd_gpu_mma12_unpack_part(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4], int64_t K, int64_t row0, int64_t rows, uint16_t* out, int64_t sms, cudaStream_t cs) {
    Nib f;
    if (!nib12(data, exc, exc_base, sym, f)) return cudaErrorInvalidValue;
    return split_decode(f, K, row0, rows, out, sms, cs);
}

// A split of the device's SMs: a green context (CUgreenCtx) and a stream each for the decode and the products.
struct SplitPart {
    int64_t dec = 0, gemm = 0;  // the SMs each has
    void *gd = nullptr, *gg = nullptr;
    cudaStream_t sd = nullptr, sg = nullptr;
};

#if CUDA_VERSION >= 12050
// The driver's green contexts, found at run time as the tensor-map encoder is (no link to libcuda): null where the
// driver lacks any of these (a green context's stream: CUDA 12.5 on).
struct Green {
    decltype(&cuDeviceGet) device;
    decltype(&cuDeviceGetDevResource) resource;
    decltype(&cuDevSmResourceSplitByCount) split;
    decltype(&cuDevResourceGenerateDesc) desc;
    decltype(&cuGreenCtxCreate) create;
    decltype(&cuGreenCtxStreamCreate) stream;
    decltype(&cuGreenCtxRecordEvent) record;
    decltype(&cuGreenCtxDestroy) destroy;
    decltype(&cuStreamDestroy) stream_destroy;
};

template <class F>
static bool driver_fn(const char* name, int version, F& fn) {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q = cudaDriverEntryPointSymbolNotFound;
    cudaGetDriverEntryPointByVersion(name, &p, version, cudaEnableDefault, &q);
    fn = p && q == cudaDriverEntryPointSuccess ? (F)p : nullptr;
    return fn != nullptr;
}

static const Green* green() {
    static const Green* g = []() -> const Green* {
        static Green x;
        bool ok = driver_fn("cuDeviceGet", 2000, x.device) & driver_fn("cuDeviceGetDevResource", 12040, x.resource) &
                  driver_fn("cuDevSmResourceSplitByCount", 12040, x.split) & driver_fn("cuDevResourceGenerateDesc", 12040, x.desc) &
                  driver_fn("cuGreenCtxCreate", 12040, x.create) & driver_fn("cuGreenCtxStreamCreate", 12050, x.stream) &
                  driver_fn("cuGreenCtxRecordEvent", 12040, x.record) & driver_fn("cuGreenCtxDestroy", 12040, x.destroy) &
                  driver_fn("cuStreamDestroy", 4000, x.stream_destroy);  // (cuda.h's cuStreamDestroy_v2)
        return ok ? &x : nullptr;
    }();
    return g;
}

// Its streams' work waited for, then its streams and contexts let go: the first failure's status.
static int part_free(SplitPart& p) {
    const Green* G = green();
    int r = 0;
    for (cudaStream_t s : {p.sd, p.sg})
        if (s) {
            int a = (int)cudaStreamSynchronize(s), b = (int)G->stream_destroy((CUstream)s);
            r = r ? r : a ? a : b;
        }
    for (void* c : {p.gd, p.gg})
        if (c) {
            int a = (int)G->destroy((CUgreenCtx)c);
            r = r ? r : a;
        }
    p = SplitPart{};
    return r;
}

// The products' SMs a co-scheduled group of at least S - sms (their clusters launch there, on Hopper in groups of
// 8), the decode's the rest: sms or a few fewer.
static int part_make(int64_t sms, SplitPart& p) {
    const Green* G = green();
    if (!G) return cudaErrorNotSupported;
    CUdevice d;
    CUdevResource all, group, rest;
    unsigned n = 1;
    if (G->device(&d, current_device()) || G->resource(d, &all, CU_DEV_RESOURCE_TYPE_SM)) return cudaErrorNotSupported;
    if (sms < 1 || sms >= (int64_t)all.sm.smCount) return cudaErrorInvalidValue;
    if (G->split(&group, &n, &all, &rest, 0, (unsigned)(all.sm.smCount - sms)) || n != 1 || rest.sm.smCount < 1) return cudaErrorNotSupported;
    CUdevResourceDesc dg, dd;
    CUgreenCtx cg = nullptr, cd = nullptr;
    CUstream a = nullptr, b = nullptr;
    bool bad = G->desc(&dg, &group, 1) || G->desc(&dd, &rest, 1) || G->create(&cg, dg, d, CU_GREEN_CTX_DEFAULT_STREAM) ||
               G->create(&cd, dd, d, CU_GREEN_CTX_DEFAULT_STREAM) || G->stream(&a, cd, CU_STREAM_NON_BLOCKING, 0) ||
               G->stream(&b, cg, CU_STREAM_NON_BLOCKING, 0);
    p.gg = cg, p.gd = cd, p.sd = (cudaStream_t)a, p.sg = (cudaStream_t)b, p.dec = rest.sm.smCount, p.gemm = group.sm.smCount;
    if (bad) part_free(p);
    return bad ? cudaErrorNotSupported : 0;
}

// An event recorded on a stream of a green context: cudaEventRecord (drivers take an event of the primary context
// there), else cuGreenCtxRecordEvent (documented for one of its primary context; a stream a green context, the same
// point).
static int record_on(cudaEvent_t e, cudaStream_t s, void* c) {
    cudaError_t r = cudaEventRecord(e, s);
    if (r == cudaSuccess) return 0;
    cudaGetLastError();
    return green()->record((CUgreenCtx)c, (CUevent)e) == CUDA_SUCCESS ? 0 : (int)r;
}
#else  // (a CUDA before 12.5, as a JIT build may be: no green context's streams in its headers, the split refused)
static int part_free(SplitPart& p) { p = SplitPart{}; return 0; }
static int part_make(int64_t, SplitPart&) { return cudaErrorNotSupported; }
static int record_on(cudaEvent_t e, cudaStream_t s, void*) { return (int)cudaEventRecord(e, s); }
#endif

constexpr int RING_MAX = 16, RING_STARTS = 64;

// A chunk's decode waits for the product of the last chunk of its shape queued before it to start (its gate: in a
// model's order, the same matrix of the layer before), so that the decode runs beside the products, whose tensor-core
// work leaves memory bandwidth to spare, and not beside the norms, activations and attention between them, which it
// slowed by taking a third of it (an A100: Qwen3-8B's prompt of 1024 tokens lost 15 of the layers' 18.5 ms so,
// benchmarks/gpu/option2-2026-09-29); the ring holds a layer's chunks ahead where it has the slots.
struct glyd_gpu_ring {
    int dev;
    uint8_t* buf;
    size_t slot_bytes;
    int slots;
    std::map<int64_t, SplitPart> parts;  // by the SMs asked for the decode
    SplitPart* cur = nullptr;            // the queue's split
    struct Chunk {
        Nib f;
        int64_t O, K, row0, rows;
        int slot;
        int64_t seq, gate;  // its number, and its gate's (-1: none)
    };
    std::deque<Chunk> q;  // queued, not yet multiplied; the first `issued` decoding or decoded
    size_t issued = 0;
    int next = 0;                                   // the slot the next decode takes
    int64_t seq = 0, started = -1;                  // the next chunk's number; the last whose product has started
    std::map<std::array<int64_t, 4>, int64_t> last; // the last chunk queued of each shape (O, K, row0, rows)
    bool busy[RING_MAX] = {}, read[RING_MAX] = {};  // a decode there not yet multiplied; a product has read it
    cudaEvent_t ready[RING_MAX], free_[RING_MAX], start[RING_STARTS], mark;
};

// The same pack: its data, exceptions, their bases and its base (a pack's addresses alone may be another's, freed and
// taken again).
static bool same_pack(const Nib& a, const Nib& b) { return a.data == b.data && a.exc == b.exc && a.exc_base == b.exc_base && a.hb4 == b.hb4; }

// The split for sms: made once and kept (on the ring's device alone).
static int ring_part(glyd_gpu_ring* g, int64_t sms, SplitPart** p) {
    if (current_device() != g->dev) return cudaErrorInvalidDevice;
    auto it = g->parts.find(sms);
    if (it == g->parts.end()) {
        SplitPart made;
        if (int r = part_make(sms, made)) return r;
        it = g->parts.emplace(sms, made).first;
    }
    *p = &it->second;
    return 0;
}

// Decodes queued into the slots free, in order, each once its gate's product has been issued (it waits for it to start).
static int ring_pump(glyd_gpu_ring* g) {
    SplitPart* p = g->cur;
    while (g->issued < g->q.size() && !g->busy[g->next] && g->q[g->issued].gate <= g->started) {
        glyd_gpu_ring::Chunk& c = g->q[g->issued];
        int s = g->next;
        if (g->read[s])  // the product that read it done
            if (cudaError_t r = cudaStreamWaitEvent(p->sd, g->free_[s], 0)) return r;
        if (c.gate >= 0 && g->started - c.gate < RING_STARTS)
            if (cudaError_t r = cudaStreamWaitEvent(p->sd, g->start[c.gate % RING_STARTS], 0)) return r;
        if (int r = split_decode(c.f, c.K, c.row0, c.rows, (uint16_t*)(g->buf + s * g->slot_bytes), p->dec, p->sd)) return r;
        if (int r = record_on(g->ready[s], p->sd, p->gd)) return r;
        c.slot = s, g->busy[s] = true, g->next = (s + 1) % g->slots, g->issued++;
    }
    return 0;
}

// Stream cs waits for everything the ring has queued on split p's streams.
static int ring_join(glyd_gpu_ring* g, SplitPart* p, cudaStream_t cs) {
    for (int k = 0; k < 2; k++) {
        if (int r = record_on(g->mark, k ? p->sg : p->sd, k ? p->gg : p->gd)) return r;
        if (cudaError_t r = cudaStreamWaitEvent(cs, g->mark, 0)) return r;
    }
    return 0;
}

// The queue dropped (cs waits for its work), its split p from here (whose streams wait for the last one's).
static int ring_restart(glyd_gpu_ring* g, SplitPart* p, cudaStream_t cs) {
    if (g->cur) {
        if (int r = ring_join(g, g->cur, cs)) return r;
        if (g->cur != p)
            for (cudaStream_t s : {p->sd, p->sg})
                if (int r = ring_join(g, g->cur, s)) return r;
    }
    g->q.clear();
    g->last.clear();
    g->issued = 0;
    std::fill(g->busy, g->busy + RING_MAX, false);  // (their decodes and products in stream order before the next)
    g->cur = p;
    return 0;
}

// W's rows in even chunks of at most a slot, multiples of 64, queued (each gated by the last chunk of its shape queued
// since the queue started) and their decodes started as slots are free.
static int ring_queue(glyd_gpu_ring* g, Nib f, int64_t O, int64_t K) {
    if (O < 64 || O % 64 || K < 64 || K % 64) return cudaErrorInvalidValue;
    int64_t per = std::min<int64_t>(O, (int64_t)(g->slot_bytes / (2 * K)) / 64 * 64);
    if (per < 64) return cudaErrorInvalidValue;  // not 64 rows to a slot
    int64_t n = (O + per - 1) / per, rows = ((O + n - 1) / n + 63) / 64 * 64;
    for (int64_t r0 = 0; r0 < O; r0 += rows) {
        int64_t k = std::min(rows, O - r0);
        auto it = g->last.try_emplace({O, K, r0, k}, -1).first;
        g->q.push_back({f, O, K, r0, k, -1, g->seq, it->second});
        it->second = g->seq++;
    }
    return ring_pump(g);
}

GLYD_GPU_INTERNAL int glyd_gpu_aux_create(void* buffer, size_t bytes, size_t slot_bytes, glyd_gpu_ring** ring) {
    if (!ring || !buffer || (uintptr_t)buffer % 16 || !slot_bytes || slot_bytes % 256 || bytes / slot_bytes < 3) return cudaErrorInvalidValue;
    glyd_gpu_ring* g = new glyd_gpu_ring;
    g->dev = current_device(), g->buf = (uint8_t*)buffer, g->slot_bytes = slot_bytes, g->slots = (int)std::min<size_t>(RING_MAX, bytes / slot_bytes);
    cudaError_t r = cudaEventCreateWithFlags(&g->mark, cudaEventDisableTiming);
    for (int s = 0; s < g->slots && !r; s++)
        if (!(r = cudaEventCreateWithFlags(&g->ready[s], cudaEventDisableTiming))) r = cudaEventCreateWithFlags(&g->free_[s], cudaEventDisableTiming);
    for (int s = 0; s < RING_STARTS && !r; s++) r = cudaEventCreateWithFlags(&g->start[s], cudaEventDisableTiming);
    if (r) {
        delete g;  // (its events let go with the process: a failed start)
        return r;
    }
    *ring = g;
    return 0;
}

GLYD_GPU_INTERNAL int glyd_gpu_aux_destroy(glyd_gpu_ring* ring) {
    if (!ring) return cudaErrorInvalidValue;
    int was = current_device(), r = was == ring->dev ? 0 : (int)cudaSetDevice(ring->dev);  // (its streams, contexts and events the ring's device's)
    auto keep = [&r](int e) { r = r ? r : e; };
    for (auto& kv : ring->parts) keep(part_free(kv.second));
    keep((int)cudaEventDestroy(ring->mark));
    for (int s = 0; s < ring->slots; s++) keep((int)cudaEventDestroy(ring->ready[s])), keep((int)cudaEventDestroy(ring->free_[s]));
    for (int s = 0; s < RING_STARTS; s++) keep((int)cudaEventDestroy(ring->start[s]));
    if (was != ring->dev) cudaSetDevice(was);
    delete ring;
    return r;
}

GLYD_GPU_INTERNAL int glyd_gpu_aux_part(glyd_gpu_ring* ring, int64_t sms, int64_t* decode_sms, int64_t* product_sms) {
    SplitPart* p;
    if (!ring) return cudaErrorInvalidValue;
    if (int r = ring_part(ring, sms, &p)) return r;
    if (decode_sms) *decode_sms = p->dec;
    if (product_sms) *product_sms = p->gemm;
    return 0;
}

GLYD_GPU_INTERNAL int glyd_gpu_aux_reset(glyd_gpu_ring* ring, cudaStream_t cs) {
    if (!ring) return cudaErrorInvalidValue;
    return current_device() != ring->dev ? cudaErrorInvalidDevice : ring_restart(ring, ring->cur, cs);
}

GLYD_GPU_INTERNAL int glyd_gpu_mma12_aux_queue(glyd_gpu_ring* ring, int64_t sms, const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4], int64_t O, int64_t K) {
    Nib f;
    SplitPart* p;
    if (!ring || !nib12(data, exc, exc_base, sym, f)) return cudaErrorInvalidValue;
    if (int r = ring_part(ring, sms, &p)) return r;
    if (p != ring->cur) {
        if (!ring->q.empty()) return cudaErrorInvalidValue;  // a queue's matrices on one split
        if (int r = ring_restart(ring, p, ring->cur ? ring->cur->sd : p->sd)) return r;
    }
    return ring_queue(ring, f, O, K);
}

GLYD_GPU_INTERNAL int glyd_gpu_mma12_aux_linear(glyd_gpu_ring* ring, int64_t sms, const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4], int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y, const glyd_gpu_blas* blas, cudaStream_t cs) {
    Nib f;
    SplitPart* p;
    if (!ring || !blas || !blas->handle || !blas->gemm_ex || !blas->set_stream || !nib12(data, exc, exc_base, sym, f)) return cudaErrorInvalidValue;
    if (M < 1 || M > INT32_MAX || O > INT32_MAX || K > INT32_MAX || (uintptr_t)x % 16 || (uintptr_t)y % 16) return cudaErrorInvalidValue;
    cudaStreamCaptureStatus cap;
    if (cudaStreamIsCapturing(cs, &cap) || cap != cudaStreamCaptureStatusNone) return cudaErrorNotSupported;  // (a capture takes another route)
    if (int r = ring_part(ring, sms, &p)) return r;
    bool next = ring->cur == p && !ring->q.empty() && same_pack(ring->q.front().f, f) && ring->q.front().row0 == 0 && ring->q.front().O == O && ring->q.front().K == K;
    if (!next) {  // off the queue: W alone, now
        if (int r = ring_restart(ring, p, cs)) return r;
        if (int r = ring_queue(ring, f, O, K)) return r;
    }
    // the products after cs's work (X); a bias into Y first, cuBLAS adding to it
    if (int r = cudaEventRecord(ring->mark, cs)) return r;
    if (int r = cudaStreamWaitEvent(p->sg, ring->mark, 0)) return r;
    if (bias) {
        bias_rows_kernel<<<(unsigned)std::max<int64_t>(1, std::min<int64_t>(p->gemm * 4, (M * O + 255) / 256)), 256, 0, p->sg>>>(y, bias, M, O);
        if (cudaError_t r = cudaGetLastError()) return r;
    }
    const float one = 1.f, beta = bias ? 1.f : 0.f;
    cudaStream_t was = nullptr;
    int target = 0, r = 0, b = 0;
    // the handle's stream and SM count target put back after, each where it was read (the target set only so)
    bool stream_read = blas->get_stream && blas->get_stream(blas->handle, &was) == 0;
    bool target_read = blas->set_sm_count_target && blas->get_sm_count_target && blas->get_sm_count_target(blas->handle, &target) == 0;
    if ((b = blas->set_stream(blas->handle, p->sg)) == 0 && blas->set_workspace && blas->workspace)
        b = blas->set_workspace(blas->handle, blas->workspace, blas->workspace_bytes);
    if (!b && target_read) b = blas->set_sm_count_target(blas->handle, (int)p->gemm);
    for (bool first = true; !b && !r && !ring->q.empty() && same_pack(ring->q.front().f, f) && (first || ring->q.front().row0); first = false) {
        if ((r = ring_pump(ring))) break;  // (the front's decode issued: its gate's product came before it, its slot is free)
        glyd_gpu_ring::Chunk c = ring->q.front();
        if (c.slot < 0) {  // (never: the front's gate is a chunk multiplied before it)
            r = cudaErrorIllegalState;
            break;
        }
        if ((r = cudaStreamWaitEvent(p->sg, ring->ready[c.slot], 0))) break;
        if ((r = record_on(ring->start[c.seq % RING_STARTS], p->sg, p->gg))) break;  // its product's start: the gate of the decodes waiting for it
        ring->started = c.seq;
        if ((r = ring_pump(ring))) break;
        b = blas->gemm_ex(blas->handle, 1, 0, (int)c.rows, (int)M, (int)K, &one, ring->buf + c.slot * ring->slot_bytes, 14, (int)K, x, 14, (int)K, &beta, y + c.row0, 14, (int)O, 68, -1);  // op T, op N; CUDA_R_16BF; CUBLAS_COMPUTE_32F; CUBLAS_GEMM_DEFAULT
        if (b || (r = record_on(ring->free_[c.slot], p->sg, p->gg))) break;
        ring->read[c.slot] = true, ring->busy[c.slot] = false;
        ring->q.pop_front();
        ring->issued--;
        r = ring_pump(ring);  // the next decodes, into the slot freed
    }
    if (target_read) blas->set_sm_count_target(blas->handle, target);
    if (stream_read) blas->set_stream(blas->handle, was);
    if (b) return GLYD_GPU_BLAS_ERROR + b;
    if (!r) r = record_on(ring->mark, p->sg, p->gg);
    if (!r) r = cudaStreamWaitEvent(cs, ring->mark, 0);  // Y
    return r ? r : (int)cudaGetLastError();
}

GLYD_GPU_INTERNAL int glyd_gpu_mma12_part_cus(int64_t gpu, int64_t O, int64_t K, int64_t M, int64_t* sms) {
    if (!sms || O < 1 || K < 1 || M < 0) return cudaErrorInvalidValue;
    *sms = split_sms(true, gpu, O, K, M);
    return 0;
}

// Exact, a mixture of experts' layer (its E matrices [O, K] stacked): the experts the plan of P pairs hits back to
// bf16, into their rows of out [E O, K]; the rest of out left as it is.
template <class Fmt>
static int mma_moe_unpack_run(Fmt f, int64_t E, int64_t O, int64_t K, int64_t P, const int32_t* plan, uint16_t* out, cudaStream_t cs) {
    if (E < 1 || O < 64 || O % 64 || K < 16 || K % 16 || P < 0) return cudaErrorInvalidValue;
    if (P == 0) return 0;
    int64_t steps = O / 64 * (K / 16);
    mma_unpack_kernel<Fmt, true><<<dim3((unsigned)((steps * 32 + 255) / 256), (unsigned)std::min(E, P)), 256, 0, cs>>>(f, K, 0, O, out, plan);
    return cudaGetLastError();
}

GLYD_GPU_INTERNAL int glyd_gpu_mma_moe_unpack(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base, const uint32_t tiers[3], int64_t E, int64_t O, int64_t K, int64_t P, const int32_t* plan, uint16_t* out, cudaStream_t cs) {
    Tiers ts;
    bool bad = false;  // (Int6, Int5 and Int4 too: E experts' packs end to end, each of whole row blocks of 64 rows, are one pack of [E O, K])
    uint32_t base4h = 0;
    if (int6_words(tiers, K, data, blocks, bad)) return bad ? cudaErrorInvalidValue : mma_moe_unpack_run(Int6{data, blocks}, E, O, K, P, plan, out, cs);
    if (int5_words(tiers, K, data, blocks, bad)) return bad ? cudaErrorInvalidValue : mma_moe_unpack_run(Int5{data, blocks}, E, O, K, P, plan, out, cs);
    if (int4_words(tiers, K, data, blocks, bad)) return bad ? cudaErrorInvalidValue : mma_moe_unpack_run(Int4{data, blocks}, E, O, K, P, plan, out, cs);
    if (int4h_words(tiers, K, data, blocks, bad, base4h)) return bad ? cudaErrorInvalidValue : mma_moe_unpack_run(Int4H{{data, blocks}, base4h}, E, O, K, P, plan, out, cs);
    if (fp4_words(tiers, K, data, blocks, bad)) return bad ? cudaErrorInvalidValue : mma_moe_unpack_run(Fp4{data, blocks}, E, O, K, P, plan, out, cs);
    if (fp4r_words(tiers, K, data, blocks, bad)) return bad ? cudaErrorInvalidValue : mma_moe_unpack_run(fp4r_fmt(data, blocks, K), E, O, K, P, plan, out, cs);
    if (!tiered_words(tiers, ts)) return cudaErrorInvalidValue;
    return mma_moe_unpack_run(Tiered{data, blocks, block_base, ts}, E, O, K, P, plan, out, cs);
}

GLYD_GPU_INTERNAL int glyd_gpu_mma12_moe_unpack(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4], int64_t E, int64_t O, int64_t K, int64_t P, const int32_t* plan, uint16_t* out, cudaStream_t cs) {
    Nib f;
    if (!nib12(data, exc, exc_base, sym, f)) return cudaErrorInvalidValue;
    return mma_moe_unpack_run(f, E, O, K, P, plan, out, cs);
}

// A mixture-of-experts layer (the E experts' [O, K] matrices stacked, [E O, K]). moe_route: the plan [2 + 2E + P]
// (int32) of the P pairs whose experts are ids [P] (int64; pair j: token j / k, choice j mod k).
GLYD_GPU_INTERNAL int glyd_gpu_moe_way(const int64_t* ids, int64_t P, int64_t E, int32_t* plan, cudaStream_t cs) {
    if (P < 0 || P >= (1ll << 31) - 2 * E || E < 1 || E > 12288) return cudaErrorInvalidValue;
    // From GLYD_GPU_ROUTE_MIN pairs (default 512; 0 or less: never) one block of up to 32 warps (its counts a warp an expert in 48 KB
    // of shared memory: 8 warps at most 1,364 experts); fewer pairs or more experts: the one warp.
    const int64_t min_pairs = TUNING_INT("GLYD_GPU_ROUTE_MIN", 512);
    int64_t nw = std::min<int64_t>(32, (48 * 1024 / 4) / E - 1);
    nw = std::min<int64_t>(nw, (P + 31) / 32);
    if (min_pairs > 0 && P >= min_pairs && nw >= 8)
        moe_route_block_kernel<<<1, (unsigned)(32 * nw), (size_t)(nw * E + E) * sizeof(int), cs>>>(ids, P, (int)E, plan);
    else
        moe_route_kernel<<<1, 32, E * sizeof(int), cs>>>(ids, P, (int)E, plan);
    return cudaGetLastError();
}

// The experts' product for T tokens' k choices each (P = T k pairs) by the plan: X [T, K] (gather: pair j takes
// token j / k's row) or [P, K] (the pairs in the plan's order); act 0: y [P, O] in the plan's order, + bias [E, O]
// (none: NULL); act 1 (SiLU) or 2 (GELU, tanh): y [P, O / 2] = act(gate) up, the gate an expert's first O / 2
// rows, the up the rest, each + its bias (O / 2 a multiple of 64); w (act 0; bf16, or fp32: wf32): y [T, O] =
// the sum of a token's k rows times their weights, added in fp32 in the workspace ([P, O] floats; none
// otherwise), a pair routed nowhere adding nothing. 16, 32 or 64 pairs of an expert at a time (as T: a token
// is routed to an expert once), K split over blocks where the units are few (their sums in the workspace after
// Y32's, [splits][P][O] floats; done: a counter a unit, O / 64 (O / 128 with the gate) by min(E, P), zero before,
// left zero). need: set to the workspace's bytes, nothing launched.
template <class Fmt, int MT, int ACT, int V>
static void mma_moe_launch(Fmt f, int64_t E, int64_t O, int64_t K, const uint16_t* x, int64_t k, int64_t gather, const int32_t* plan, const uint16_t* bias, const void* w, int64_t wf32, uint16_t* y, float* y32, int64_t P, float* parts, int* done, dim3 grid, cudaStream_t cs) {
    auto kernel = mma_moe_kernel<Fmt, MT, ACT, V>;
    size_t shared = (size_t)4 * MT * 16 * 65 * sizeof(float) + 8 * S2_BYTES;  // red, the warps' scratch
    if constexpr ((V & 16) && MT == 1) shared += (size_t)8 * ((V & (64 | 128)) ? 8 : 4) * ring_slot<Fmt>();  // (the warps' rings; V & 128: two groups of four steps)
    static std::atomic<int> allowed[MAX_DEVICES];  // the shared memory allowed on a device, once
    int dev = current_device();
    if (dev >= MAX_DEVICES || !allowed[dev].load()) {
        cudaFuncSetAttribute((const void*)kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared);
        if (dev < MAX_DEVICES) allowed[dev] = 1;
    }
    kernel<<<grid, 256, shared, cs>>>(f, O, K, plan, (int)E, k, (int)gather, bf(x), bf(bias), w, (int)wf32, bf(y), y32, P, parts, done, step_opts().pf);
}

template <class Fmt, int MT, int V>
static void mma_moe_rows_launch(Fmt f, int64_t E, int64_t O, int64_t K, const uint16_t* x, int64_t k, int64_t gather, const int32_t* plan, const uint16_t* bias, const void* w, int64_t wf32, uint16_t* y, float* y32, int64_t hits, cudaStream_t cs) {
    mma_moe_rows_kernel<Fmt, MT, V><<<dim3((unsigned)((O / 64 + 7) / 8), (unsigned)hits), 256, 8 * S2_BYTES, cs>>>(f, O, K, plan, (int)E, k, (int)gather, bf(x), bf(bias), w, (int)wf32, bf(y), y32, step_opts().pf);
}

// Blocks splitting K for the experts' product (units: its units by the experts that can be hit): up to the blocks
// the GPU holds at once (a wave more costs a block's whole time), at least 16 steps (two a warp) each. On a GDDR GPU
// a block an SM keeps its memory busy (an A10: split, the product was slower, measured); an HBM part (A100, H100,
// B200: two to six times the bandwidth an SM) takes two. GLYD_GPU_MOE_SLOTS: the blocks held at once (tests).
static bool hbm_gpu(int dev) {
    const int major = attribute(cudaDevAttrComputeCapabilityMajor, dev);
    return attribute(cudaDevAttrComputeCapabilityMinor, dev) == 0 && major >= 8 && major <= 10;
}
// Layout H's packs on an HBM GPU are plain by default (every value its e4m3 byte: the release's size in VRAM, the products at the card's streaming rate; sessions
// 10-12, research/dsv4/f8h-one-2026-10-19) and the one-token product is then f8b_lin1_kernel; GLYD_F8H_PLAIN=0 keeps the coded layout in VRAM where the raw bytes
// do not fit (its one-token product f8h_lin1_kernel). A GDDR GPU keeps the coded layout: memory-bound either way, and the coded form is 11% fewer bytes.
// glyd_gpu_f8h_plain sets it for the process (the engine's choice by fit, a release build's too): 1 plain, 0 coded, -1 the GPU's own; before any pack, as every
// pack and product of the process reads it.
static int g_f8h_plain = -1;
static bool f8h_plain() { return TUNING_INT("GLYD_F8H_PLAIN", g_f8h_plain >= 0 ? g_f8h_plain : hbm_gpu(current_device()) ? 1 : 0) != 0; }
GLYD_GPU_INTERNAL int glyd_gpu_f8h_plain(int plain) {
    if (plain < -1 || plain > 1) return cudaErrorInvalidValue;
    g_f8h_plain = plain;
    return 0;
}
static int64_t moe_splits(int64_t units, int64_t K) {
    const int64_t forced = TUNING_INT("GLYD_GPU_MOE_SLOTS", 0);
    int dev = current_device();
    bool hbm = hbm_gpu(dev);
    int64_t slots = forced ? forced : sm_count(dev) * (hbm ? 2 : 1);
    return std::max<int64_t>(1, std::min<int64_t>(slots / units, K / 256));
}

// Many pairs an expert (a prompt): mma_gemm_big_kernel's variant 1 (128 of them by two row blocks a block, a weight
// decoded once for the 128), blockIdx.z the experts hit.
template <class Fmt, int CW, int NB>
static void mma_moe_big_launch(Fmt f, int64_t E, int64_t O, int64_t K, const uint16_t* x, int64_t T, int64_t k, int64_t gather, const int32_t* plan, int64_t act, const uint16_t* bias, const void* w, int64_t wf32, uint16_t* y, float* y32, cudaStream_t cs) {
    using C = Big<CW, 4, NB, 2>;
    auto kernel = mma_gemm_big_kernel<Fmt, CW, 4, NB, 2, true>;
    static std::atomic<int> known[MAX_DEVICES];
    per_sm((const void*)kernel, C::THREADS, C::SHARED, known, current_device());  // its shared memory allowed, once
    dim3 grid((unsigned)((T + C::TM - 1) / C::TM), (unsigned)(act ? O / 128 : (O / 64 + 1) / 2), (unsigned)std::min(E, T * k));
    const int xahead = (int)TUNING_INT("GLYD_GPU_BIG_XAHEAD", 0), pf = (int)std::max<int64_t>(0, std::min<int64_t>(8, TUNING_INT("GLYD_GPU_BIG_PF", 0)));  // the producers' variants (see MoePairs)
    const int skip = TUNING_INT("GLYD_GPU_BIG_SKIP", 1) != 0;  // the consumers' (see MoePairs)
    kernel<<<grid, C::THREADS, C::SHARED, cs>>>(f, O, K, 0, 0, bf(x), bf(bias), bf(y), y32, MoePairs{plan, (int)E, k, (int)gather, (int)act, w, (int)wf32, xahead, pf, skip});
}

// The tile of 128 tokens (four consumer warps of 64, three stages), or of 256 (eight, two stages; GLYD_GPU_BIG_NB 3 where the block's shared
// memory holds them: an A100's, not GeForce Ada's 99 KB): a weight is decoded once for 256 pairs where an expert has that many (an expert with
// 129 pairs has a second tile of one, which costs the decode again), the consumers' m-tiles with no pairs left out. GLYD_GPU_BIG_TM: 128 (the
// default), 256. On an L4 (6 experts, Qwen3-30B-A3B's shapes, the weights in the L2 cache, the epilogues as they are now) 256 took 1-6% less at 1,024 pairs an
// expert, the same at 512, 10% more at 256 and 60% more at 128: its eight consumer warps do not hide their fragment loads where few have pairs (the four-warp
// loop's are asked for a step ahead); before the epilogues were shrunk it took 9-14% less at 512-1,024, the epilogue's cost being what a larger tile shared.
template <class Fmt>
static void mma_moe_big(Fmt f, int64_t E, int64_t O, int64_t K, const uint16_t* x, int64_t T, int64_t k, int64_t gather, const int32_t* plan, int64_t act, const uint16_t* bias, const void* w, int64_t wf32, uint16_t* y, float* y32, cudaStream_t cs) {
#ifndef GLYD_RELEASE  // (a variant for measuring: a release library has no copy of its kernels)
    if (TUNING_INT("GLYD_GPU_BIG_TM", 128) == 256) {
        if (TUNING_INT("GLYD_GPU_BIG_NB", 2) == 3 && attribute(cudaDevAttrMaxSharedMemoryPerBlockOptin, current_device()) >= (int)Big<8, 4, 3, 2>::SHARED) return mma_moe_big_launch<Fmt, 8, 3>(f, E, O, K, x, T, k, gather, plan, act, bias, w, wf32, y, y32, cs);
        return mma_moe_big_launch<Fmt, 8, 2>(f, E, O, K, x, T, k, gather, plan, act, bias, w, wf32, y, y32, cs);
    }
#endif
    mma_moe_big_launch<Fmt, 4, 3>(f, E, O, K, x, T, k, gather, plan, act, bias, w, wf32, y, y32, cs);
}

// The experts' products by wgmma (mma12_moe_wg_kernel) on Hopper, from GLYD_GPU_MOE_WG tokens a step (the default: 17 on an H100, where
// it took 11-26% off the layer at 32-232 tokens and 45-60% at 768-8,192, else 0: not; the 12-bit layout, K a multiple of 64, O of 128 with
// the gate (SiLU) or 64): the tile of pairs (16, 32, 64 or 128; GLYD_GPU_MOE_WG_NT forces one) the smallest that holds the pairs an expert
// has on average (all of them hit) and 1.5 standard deviations more, as many as one has by chance: a tile that most experts overflow
// reads their weights again for the rest (an H100, Qwen3-30B-A3B at 232 tokens, 14.5 an expert: a tile of 16 took 537 us a layer, 32 438). The workspace after Y32 (where weighted):
// the tiles' records (a header and at most one for each hit and each 16 pairs), then the pairs' rows of X in the order's (where X is
// the tokens': the product with the gate, or gather); a workspace without them (the variable set after the size was asked) takes
// the other kernels.
static bool wg_device(int dev) { return attribute(cudaDevAttrComputeCapabilityMajor, dev) == 9 && attribute(cudaDevAttrComputeCapabilityMinor, dev) == 0; }
static size_t wg_align(size_t n) { return (n + 255) & ~(size_t)255; }
static size_t wg_records(int64_t E, int64_t P) { return wg_align((size_t)(1 + std::min(E, P) + (P + 15) / 16) * sizeof(int4)); }
static size_t wg_tail(int64_t E, int64_t K, int64_t P, bool rows) { return wg_records(E, P) + (rows ? wg_align((size_t)P * K * sizeof(uint16_t)) : 0); }
static int wg_tile(int64_t P, int64_t E) {
    const int64_t forced = TUNING_INT("GLYD_GPU_MOE_WG_NT", 0);
    if (forced == 16 || forced == 32 || forced == 64 || forced == 128) return (int)forced;
    const double hit = (double)std::max<int64_t>(1, std::min(E, P)), mean = (double)P / hit, need = mean + 1.5 * std::sqrt(mean);
    return need <= 16 ? 16 : need <= 32 ? 32 : need <= 64 ? 64 : 128;
}

template <int NT, int ACT>
static int mma12_moe_wg_launch(const Nib& f, int64_t E, int64_t O, int64_t K, const uint16_t* xs, int64_t P, const int32_t* plan, const void* tiles, const uint16_t* bias, const void* wts, int64_t wf32, uint16_t* y, float* y32, cudaStream_t cs) {
#if defined(__HIP_PLATFORM_AMD__)
    return cudaErrorNotSupported;
#else
    using C = Wm12<NT, ACT>;
    auto kernel = mma12_moe_wg_kernel<NT, ACT>;
    static std::atomic<int> known[MAX_DEVICES];
    const int dev = current_device();
    auto encode = tensor_map_encoder();
    if (!encode) return cudaErrorNotSupported;
    // X [P, K] as tiles of NT pairs by 64 columns, 128-byte swizzle; rows past P read as zeros.
    CUtensorMap map;
    cuuint64_t dims[2] = {(cuuint64_t)K, (cuuint64_t)P}, strides[1] = {(cuuint64_t)K * 2};
    cuuint32_t box[2] = {64, NT}, unit[2] = {1, 1};
    CUresult r = encode(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)xs, dims, strides, box, unit, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) return cudaErrorInvalidValue;
    const int RU = ACT ? (int)(O / 128) : (int)((O / 64 + 1) / 2);
    const int64_t records = std::min(E, P) + (P + NT - 1) / NT;  // (at most)
    const int64_t nb = std::max<int64_t>(1, std::min<int64_t>(per_sm((const void*)kernel, C::THREADS, C::SHARED, known, dev) * sm_count(dev), (int64_t)RU * records));
    kernel<<<(unsigned)nb, C::THREADS, C::SHARED, cs>>>(map, f, O, K, (int)E, plan, (const int4*)tiles, RU, bf(bias), wts, (int)wf32, bf(y), y32);
    return cudaGetLastError();
#endif
}

static int mma_moe_wg(const Nib& f, int64_t E, int64_t O, int64_t K, const uint16_t* x, int64_t T, int64_t k, int64_t gather, const int32_t* plan, int64_t act, const uint16_t* bias, const void* w, int64_t wf32, uint16_t* y, float* y32, void* ws, size_t ws_bytes, cudaStream_t cs) {
    const int64_t P = T * k;
    const bool rows = act || gather;  // X in the order's rows through the workspace
    const size_t head = w ? (size_t)P * O * sizeof(float) : 0;
    if (!fits(ws, ws_bytes, head + wg_tail(E, K, P, rows))) return cudaErrorNotSupported;
    if (act > 1 || K % 64 || O % (act ? 128 : 64) || (uintptr_t)x % 16 || (uintptr_t)f.data % 16 || (uintptr_t)f.exc % 16) return cudaErrorNotSupported;
    char* tail = (char*)ws + head;
    const int nt = wg_tile(P, E);
    moe_wg_tiles_kernel<<<1, 1024, 0, cs>>>(plan, (int)E, nt, (int4*)tail);
    const uint16_t* xs = x;
    if (rows) {
        if (gather) {
            uint16_t* g = (uint16_t*)(tail + wg_records(E, P));
            moe_wg_gather_kernel<<<(unsigned)std::min<int64_t>(P, 4096), 128, 0, cs>>>((const uint4*)x, plan, (int)E, k, K / 8, (uint4*)g);
            xs = g;
        }
    }
    int r;
    auto go = [&](auto nt_c) {
        constexpr int NT = decltype(nt_c)::value;
        r = act ? mma12_moe_wg_launch<NT, 1>(f, E, O, K, xs, P, plan, tail, bias, w, wf32, y, y32, cs) : mma12_moe_wg_launch<NT, 0>(f, E, O, K, xs, P, plan, tail, bias, w, wf32, y, y32, cs);
    };
    if (nt == 16) go(std::integral_constant<int, 16>());
    else if (nt == 32) go(std::integral_constant<int, 32>());
    else if (nt == 64) go(std::integral_constant<int, 64>());
    else go(std::integral_constant<int, 128>());
    return r;
}

template <class Fmt>
static int mma_moe_run(Fmt f, int64_t E, int64_t O, int64_t K, const uint16_t* x, int64_t T, int64_t k, int64_t gather, const int32_t* plan, int64_t act, const uint16_t* bias, const void* w, int64_t wf32, const int64_t* ids, uint16_t* y, void* ws, size_t ws_bytes, int* done, cudaStream_t cs, size_t* need) {
    if (E < 1 || E > 12288 || O < 64 || O % 64 || K < 16 || K % 16 || T < 0 || k < 1 || act < 0 || act > 2 || (act && (O % 128 || w))) return cudaErrorInvalidValue;
    int64_t P = T * k, MT = T <= 16 ? 1 : T <= 32 ? 2 : 4, units = act ? O / 128 : O / 64, hits = std::min(E, P);
    int64_t splits = P ? moe_splits(units * hits, K) : 1;
    size_t bytes = ((w ? 1 : 0) + (splits > 1 ? splits : 0)) * (size_t)(P * O) * sizeof(float);  // [Y32][parts]
    bool wg = false;  // the wgmma route (Hopper, the 12-bit layout, GLYD_GPU_MOE_WG tokens a step and more)
    if constexpr (std::is_same_v<Fmt, Nib>) {
        const int64_t from = TUNING_INT("GLYD_GPU_MOE_WG", h100_gpu(current_device()) ? 17 : 0);
        wg = from > 0 && T >= from && P > 0 && K % 64 == 0 && O % (act ? 128 : 64) == 0 && act <= 1 && wg_device(current_device());
        if (need && from > 0 && P > 0 && wg_device(current_device())) bytes = std::max(bytes, (w ? (size_t)P * O * sizeof(float) : 0) + wg_tail(E, K, P, act || gather));
    }
    if (need) {
        *need = bytes;
        return 0;
    }
    if (!fits(ws, ws_bytes, bytes) || (w && !ids) || (splits > 1 && !done)) return cudaErrorInvalidValue;
    if (P == 0) return 0;
    float* y32 = w ? (float*)ws : nullptr;
    if constexpr (std::is_same_v<Fmt, Nib>) {
        if (wg) {
            int r = mma_moe_wg(f, E, O, K, x, T, k, gather, plan, act, bias, w, wf32, y, y32, ws, ws_bytes, cs);
            if (r == 0) {
                if (w) moe_sum_kernel<<<(unsigned)((T * O / 4 + 255) / 256), 256, 0, cs>>>(y32, ids, (int)E, T, k, O, bf(y));
                return cudaGetLastError();
            }
            if (r != cudaErrorNotSupported || TUNING_INT("GLYD_GPU_MOE_WG_STRICT", 0)) return r;  // (STRICT: a refusal is an error, for the checks that would otherwise measure the other kernels)
        }
    }
    float* parts = splits > 1 ? (float*)ws + (w ? P * O : 0) : nullptr;
    dim3 grid((unsigned)units, (unsigned)hits, (unsigned)splits);
    const StepOpts so = step_opts();
    // The step loop's variants (mma_moe_kernel's V): batched exception loads (1), the m-tiles with pairs alone (2), the L2 prefetch (4). The tiered
    // layout's has the second alone (no batched loads there; its GPUs gain nothing from the prefetch, and a register and a branch a step cost its
    // 4-tile kernel spills).
    // (the integer block layouts' products of up to 16 tokens two steps ahead on an HBM GPU, V & 8: on an H100 SXM 7-10% faster at 1 and 4 tokens (int4, int5,
    // int6; research/dsv4/h100-profile-2026-10-18/depth2), the same on a 4080 SUPER; FP4 too since its cheaper decode (A100 166 -> 158 us a layer, research/dsv4/
    // a100-stalls-2026-10-19); FP4R not (the same either way, 10 more registers); GLYD_GPU_DEPTH=1 or 2 sets it)
    constexpr bool fp4r = std::is_same_v<Fmt, Fp4R>;
    const int depth = (int)TUNING_INT("GLYD_GPU_DEPTH", hbm_gpu(current_device()) && !fp4r ? 2 : 1);
    const int deep = Fmt::kTable && MT == 1 && depth == 2 ? 8 : 0;
    // (GLYD_GPU_PF: the integer block layouts' one-token products also prefetch into L2, V & 4: steps in flight that take no registers; being measured on an H100)
    // (GLYD_GPU_STAGES=4 or 8: the block layouts' one-token products through shared memory, V & 16 (8 slots: V & 64); GLYD_GPU_STAGE_BLOCKS=3: 3 blocks an SM, V & 32;
    // being measured on an H100. The packs' bytes 16-aligned, for cp.async.)
    const int64_t stages = TUNING_INT("GLYD_GPU_STAGES", 0);
    bool aligned16 = false;
    if constexpr (has_stage<Fmt>::value) aligned16 = (((uintptr_t)f.data | (uintptr_t)f.scales) & 15) == 0;
    const int stage_steps = (int)TUNING_INT("GLYD_GPU_STAGE_STEPS", 1);  // (4: four steps a stage, V & 128, the 8-slot ring as two groups; research)
    const int staged = has_stage<Fmt>::value && MT == 1 && aligned16 && (stages == 4 || stages == 8) ? 16 | (stages == 8 || stage_steps == 4 ? 64 : 0) | (TUNING_INT("GLYD_GPU_STAGE_BLOCKS", 2) == 3 ? 32 : 0) | (stage_steps == 4 ? 128 : 0) : 0;
    const int variant = staged ? staged : Fmt::kTable ? ((so.skip ? 2 : 0) | deep | (MT == 1 && so.pf > 0 ? 4 : 0)) : ((so.b ? 1 : 0) | (so.skip ? 2 : 0) | (so.pf > 0 ? 4 : 0));
    auto with_variant = [&](auto&& body) {
        if constexpr (Fmt::kTable) {
            if (variant == 2) body(std::integral_constant<int, 2>());
            else if (variant == 8) body(std::integral_constant<int, 8>());
            else if (variant == 10) body(std::integral_constant<int, 10>());
            else if (variant == 4) body(std::integral_constant<int, 4>());
            else if (variant == 6) body(std::integral_constant<int, 6>());
            else if (variant == 12) body(std::integral_constant<int, 12>());
            else if (variant == 14) body(std::integral_constant<int, 14>());
            else if constexpr (has_stage<Fmt>::value) {
                if (variant == 16) body(std::integral_constant<int, 16>());
                else if (variant == 48) body(std::integral_constant<int, 48>());
                else if (variant == 80) body(std::integral_constant<int, 80>());
                else if (variant == 112) body(std::integral_constant<int, 112>());
                else if (variant == 208) body(std::integral_constant<int, 208>());
                else if (variant == 240) body(std::integral_constant<int, 240>());
                else body(std::integral_constant<int, 0>());
            }
            else body(std::integral_constant<int, 0>());
        } else {
            switch (variant) {
                case 0: body(std::integral_constant<int, 0>()); break;
                case 1: body(std::integral_constant<int, 1>()); break;
                case 2: body(std::integral_constant<int, 2>()); break;
                case 3: body(std::integral_constant<int, 3>()); break;
                case 4: body(std::integral_constant<int, 4>()); break;
                case 5: body(std::integral_constant<int, 5>()); break;
                case 6: body(std::integral_constant<int, 6>()); break;
                default: body(std::integral_constant<int, 7>()); break;
            }
        }
    };
    auto run = [&](auto mt) {
        constexpr int M = decltype(mt)::value;
        with_variant([&](auto vv) {
            constexpr int V = decltype(vv)::value;
            if (act == 1) mma_moe_launch<Fmt, M, 1, V>(f, E, O, K, x, k, gather, plan, bias, w, wf32, y, y32, P, parts, done, grid, cs);
            else if (act == 2) mma_moe_launch<Fmt, M, 2, V>(f, E, O, K, x, k, gather, plan, bias, w, wf32, y, y32, P, parts, done, grid, cs);
            else mma_moe_launch<Fmt, M, 0, V>(f, E, O, K, x, k, gather, plan, bias, w, wf32, y, y32, P, parts, done, grid, cs);
        });
    };
    // From 48 pairs an expert, on average (a prompt): the tiled GEMM (measured on an A10, OLMoE-1B-7B).
    // (an H100: 96, where the decode products beat the tiled GEMM at 768 tokens, 714 us a layer against 1,379; an A100, both from 64: Qwen3-30B-A3B's layer, its second A100 run, at 768 tokens the down 448 us on the decode
    // products against 591 on the tiled GEMM, at 1,152 the gate and up 981 on the tiled GEMM against 1,309 and the down 661 against 754; the layer 11% and 16% faster than from 84 and 48.
    // GLYD_GPU_MOE_BIG_GU and GLYD_GPU_MOE_BIG_DN: the gate and up's, the down's alone; GLYD_GPU_MOE_BIG: both)
    const bool h100 = h100_gpu(current_device()), a100 = ga100_gpu(current_device());
    const int64_t big_set = TUNING_INT("GLYD_GPU_MOE_BIG", -1);  // (negative: not set)
    const int64_t big = act ? TUNING_INT("GLYD_GPU_MOE_BIG_GU", big_set >= 0 ? big_set : a100 ? 64 : h100 ? 96 : 48) : TUNING_INT("GLYD_GPU_MOE_BIG_DN", big_set >= 0 ? big_set : a100 ? 64 : h100 ? 96 : 48);
    // No gate, rows a warp over the whole K (mma_moe_rows_kernel; MT 2 and 4): from 17 tokens on an H100 and an A100 and 33 on a GDDR GPU, the GPUs it was measured on (an L4: 17-32 took 3% longer); on another HBM part
    // the other kernels, until it is measured there (GLYD_GPU_MOE_ROWS=N: from N tokens).
    const int64_t rows_from = so.rows == 0 ? INT64_MAX : so.rows >= 2 ? so.rows : (h100 || a100) ? 17 : hbm_gpu(current_device()) ? INT64_MAX : 33;
    bool rows = act == 0 && MT >= 2 && T >= rows_from;
    auto go_rows = [&](auto mt) {
        constexpr int M = decltype(mt)::value;
        with_variant([&](auto vv) { mma_moe_rows_launch<Fmt, M, decltype(vv)::value>(f, E, O, K, x, k, gather, plan, bias, w, wf32, y, y32, hits, cs); });
    };
    if (T * k >= big * E && K % 64 == 0 && (uintptr_t)x % 16 == 0) mma_moe_big(f, E, O, K, x, T, k, gather, plan, act, bias, w, wf32, y, y32, cs);
    else if (rows && MT == 1) go_rows(std::integral_constant<int, 1>());
    else if (rows && MT == 2) go_rows(std::integral_constant<int, 2>());
    else if (rows) go_rows(std::integral_constant<int, 4>());
    else if (MT == 1) run(std::integral_constant<int, 1>());
    else if (MT == 2) run(std::integral_constant<int, 2>());
    else run(std::integral_constant<int, 4>());
    if (w) moe_sum_kernel<<<(unsigned)((T * O / 4 + 255) / 256), 256, 0, cs>>>(y32, ids, (int)E, T, k, O, bf(y));
    return cudaGetLastError();
}

GLYD_GPU_INTERNAL int glyd_gpu_mma_moe_workspace(int64_t E, int64_t O, int64_t K, int64_t T, int64_t k, int64_t act, int64_t weighted, size_t* bytes) {
    int64_t one = 1;  // a stand-in for the weights: only whether there are any counts
    return bytes ? mma_moe_run(Tiered{}, E, O, K, nullptr, T, k, 0, nullptr, act, nullptr, weighted ? &one : nullptr, 0, nullptr, nullptr, nullptr, 0, nullptr, 0, bytes) : cudaErrorInvalidValue;
}

GLYD_GPU_INTERNAL int glyd_gpu_mma_moe(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base, const uint32_t tiers[3], int64_t E, int64_t O, int64_t K, const uint16_t* x, int64_t T, int64_t k, int64_t gather, const int32_t* plan, int64_t act, const uint16_t* bias, const void* w, int64_t wf32, const int64_t* ids, uint16_t* y, void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs) {
    Tiers ts;
    bool bad = false;  // (Int6, Int5 and Int4 too: E experts' packs end to end are one pack of [E O, K]; K a multiple of 32; the workspace is the same)
    uint32_t base4h = 0;
    if (int6_words(tiers, K, data, blocks, bad)) return bad ? cudaErrorInvalidValue : mma_moe_run(Int6{data, blocks}, E, O, K, x, T, k, gather, plan, act, bias, w, wf32, ids, y, workspace, workspace_bytes, done, cs, nullptr);
    if (int5_words(tiers, K, data, blocks, bad)) return bad ? cudaErrorInvalidValue : mma_moe_run(Int5{data, blocks}, E, O, K, x, T, k, gather, plan, act, bias, w, wf32, ids, y, workspace, workspace_bytes, done, cs, nullptr);
    if (int4_words(tiers, K, data, blocks, bad)) return bad ? cudaErrorInvalidValue : mma_moe_run(Int4{data, blocks}, E, O, K, x, T, k, gather, plan, act, bias, w, wf32, ids, y, workspace, workspace_bytes, done, cs, nullptr);
    if (int4h_words(tiers, K, data, blocks, bad, base4h)) return bad ? cudaErrorInvalidValue : mma_moe_run(Int4H{{data, blocks}, base4h}, E, O, K, x, T, k, gather, plan, act, bias, w, wf32, ids, y, workspace, workspace_bytes, done, cs, nullptr);
    if (fp4_words(tiers, K, data, blocks, bad)) return bad ? cudaErrorInvalidValue : mma_moe_run(Fp4{data, blocks}, E, O, K, x, T, k, gather, plan, act, bias, w, wf32, ids, y, workspace, workspace_bytes, done, cs, nullptr);
    if (fp4r_words(tiers, K, data, blocks, bad)) return bad ? cudaErrorInvalidValue : mma_moe_run(fp4r_fmt(data, blocks, K), E, O, K, x, T, k, gather, plan, act, bias, w, wf32, ids, y, workspace, workspace_bytes, done, cs, nullptr);
    if (!tiered_words(tiers, ts)) return cudaErrorInvalidValue;
    return mma_moe_run(Tiered{data, blocks, block_base, ts}, E, O, K, x, T, k, gather, plan, act, bias, w, wf32, ids, y, workspace, workspace_bytes, done, cs, nullptr);
}

GLYD_GPU_INTERNAL int glyd_gpu_mma12_moe_workspace(int64_t E, int64_t O, int64_t K, int64_t T, int64_t k, int64_t act, int64_t weighted, size_t* bytes) {
    int64_t one = 1;  // a stand-in for the weights: only whether there are any counts
    return bytes ? mma_moe_run(Nib{}, E, O, K, nullptr, T, k, 0, nullptr, act, nullptr, weighted ? &one : nullptr, 0, nullptr, nullptr, nullptr, 0, nullptr, 0, bytes) : cudaErrorInvalidValue;
}

GLYD_GPU_INTERNAL int glyd_gpu_mma12_moe(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4], int64_t E, int64_t O, int64_t K, const uint16_t* x, int64_t T, int64_t k, int64_t gather, const int32_t* plan, int64_t act, const uint16_t* bias, const void* w, int64_t wf32, const int64_t* ids, uint16_t* y, void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs) {
    Nib f;
    if (!nib12(data, exc, exc_base, sym, f)) return cudaErrorInvalidValue;
    return mma_moe_run(f, E, O, K, x, T, k, gather, plan, act, bias, w, wf32, ids, y, workspace, workspace_bytes, done, cs, nullptr);
}

// Attention for one new token a sequence over the KV cache in the mma layout (gpu/kv.py): q [pairs x G, D]
// (D 64 or 128, G up to 16), out the same. Pages a block: enough blocks for four an SM, at least a page a
// warp; the workspace: the blocks' rows, [pairs][splits][16][D + 2] floats; done: pairs.
static int64_t attn_splits(int64_t tlen, int64_t pairs, int64_t P, int64_t& per) {
    int64_t units = P + (tlen > 0), sms = sm_count(current_device());
    per = std::max<int64_t>(ATT_WARPS, (pairs * units + 4 * sms - 1) / (4 * sms));
    per = (per + ATT_WARPS - 1) / ATT_WARPS * ATT_WARPS;
    return (units + per - 1) / per;
}

GLYD_GPU_INTERNAL int glyd_gpu_attn_decode_workspace(int64_t D, int64_t tlen, int64_t pairs, int64_t P, size_t* bytes) {
    if (!bytes || !(D == 64 || D == 128) || tlen >= 64 || P + (tlen > 0) <= 0) return cudaErrorInvalidValue;
    int64_t per, splits = attn_splits(tlen, pairs, P, per);
    *bytes = (size_t)(pairs * splits * 16 * (D + 2)) * sizeof(float);
    return 0;
}

GLYD_GPU_INTERNAL int glyd_gpu_attn_decode(const uint16_t* q, int64_t D, const uint8_t* kd, const uint8_t* kb, const int32_t* kbb, const uint32_t kt[3], const uint8_t* vd, const uint8_t* vb, const int32_t* vbb, const uint32_t vt[3], const uint16_t* tk, const uint16_t* tv, int64_t tlen, int64_t pairs, int64_t G, int64_t P, double scale, uint16_t* out, void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs) {
    size_t need;
    if (G < 1 || G > 16 || glyd_gpu_attn_decode_workspace(D, tlen, pairs, P, &need) || !fits(workspace, workspace_bytes, need) || !done) return cudaErrorInvalidValue;
    int64_t per, splits = attn_splits(tlen, pairs, P, per);
    float sl2 = (float)(scale * 1.4426950408889634);
    Tiers kts, vts;
    if (!tiered_words(kt, kts) || !tiered_words(vt, vts)) return cudaErrorInvalidValue;
    auto launch = [&](auto kernel) {
        kernel<<<dim3(pairs, splits), 32 * ATT_WARPS, 0, cs>>>(bf(q), kd, kb, kbb, kts, vd, vb, vbb, vts, tlen ? bf(tk) : nullptr, tlen ? bf(tv) : nullptr, (int)tlen, (int)pairs, (int)G, (int)P, (int)per, sl2, (float*)workspace, done, bf(out));
    };
    if (D == 128) launch(attn_decode_kernel<128>);
    else launch(attn_decode_kernel<64>);
    return cudaGetLastError();
}

// The lossless 12-bit KV cache (kvc, above): its C API (glyd_gpu.h says what each takes).
static constexpr size_t KV_CNT_BYTES = 1 << 20;  // the decode workspace's counters (a u32 a (sequence, kv head)), first in it
static bool kv_pipe() {  // the attention kernel loading the next page ahead (GLYD_KV_PIPE=1; not by default: it takes 219 registers of the 168 and an SM holds 2 of its CTAs where it holds 3, which measured as fast on an L4 and
    // faster on an A100), read at each call: the plan's bits do not depend on it
    const char* e = TUNING_ENV("GLYD_KV_PIPE");
    return e && e[0] == '1';
}

static bool kv_fuse() {  // the last CTA of a (sequence, head) merges the partials in the attention kernel (GLYD_KV_FUSE=0: a kernel of its own does), read at each launch
    const char* e = TUNING_ENV("GLYD_KV_FUSE");
    return !(e && e[0] == '0');
}

static int kv_ppc_min() {  // the plan's least pages a CTA (GLYD_KV_PPC_MIN: 4 to 64; 16), read at each call: a sequence's partition, and so its output's bits, depend on it
    const char* e = TUNING_ENV("GLYD_KV_PPC_MIN");
    const int v = e ? std::atoi(e) : 0;
    return v >= kvc::NW && v <= 64 ? v : kvc::PPC_MIN;
}

static std::atomic<int> g_kv_floor{0};  // glyd_gpu_kv_floor_set: the floor the package asks for (0: the GPU's own)
static int kv_floor(int64_t B, int64_t H, int dev) {  // the plan's CTAs a sequence has at least (GLYD_KV_FLOOR, 1 to 64: that number whatever the batch and the GPU; unset: on a GPU of 100 SMs or more 16 for a batch of so few
    // sequences that its (sequence, kv head) pairs, 4 to an SM, are fewer than the SMs, else 1: none; an A100 (108 SMs) decodes a lone sequence of 1,024 to 4,096 tokens 5 to 17% faster so, an L4 (58 SMs, 300 GB/s) no faster), read
    // at each call: a sequence's partition, and so its output's bits, depend on it; the batch's size is B rounded up to a power of two (a batch padded for a CUDA graph has the one)
    const char* e = TUNING_ENV("GLYD_KV_FLOOR");
    const int v = e ? std::atoi(e) : g_kv_floor.load(std::memory_order_relaxed);
    if (v >= 1 && v <= 64) return v;
    int64_t bb = 1;
    while (bb < B) bb <<= 1;
    const int64_t sms = sm_count(dev);
    return sms >= 100 && bb * H * 4 <= sms ? 16 : 1;
}

static int kv_ctas_per_sm(bool pipe, bool fuse, int dev) {
    static std::atomic<int> known[2][2][MAX_DEVICES];
    const void* kernel = pipe ? (fuse ? (const void*)kvc::attn_kernel<true, true, 1> : (const void*)kvc::attn_kernel<true, false, 1>) : (fuse ? (const void*)kvc::attn_kernel<false, true, 1> : (const void*)kvc::attn_kernel<false, false, 1>);
    return (int)per_sm(kernel, kvc::NW * 32, 0, known[pipe ? 1 : 0][fuse ? 1 : 0], dev);
}

static bool kv_lean() {  // the write kernel's decode-step path for the runs of one token (GLYD_KV_LEAN=0: every run through the general path), read at each launch (a test sets it between launches)
    const char* e = TUNING_ENV("GLYD_KV_LEAN");
    return !(e && e[0] == '0');
}
static int kv_waves() {  // waves of the machine the attention kernel's CTAs make (GLYD_KV_WAVES; 6), read at each call
    const char* e = TUNING_ENV("GLYD_KV_WAVES");
    const int w = e ? std::atoi(e) : 0;
    return w >= 1 && w <= 64 ? w : 6;
}

// The most CTAs a (sequence, kv head) is split over (the grid's x), for a batch of B sequences of a model whose longest has max_pages pages: enough for kv_waves() waves of the machine (of the
// kernel with the next page loaded ahead and the merge in it, whichever the calls run: the plan's bits do not depend on GLYD_KV_PIPE) at the batch's size rounded up to a power of two (so a batch
// and the batch padded for a CUDA graph have the one), and at least 4 pages a warp. A sequence has fewer of them where its own pages are fewer (kvc::plan_splits): its output's bits depend on its
// length and this number.
static int64_t kv_splits(int64_t B, int64_t H, int64_t max_pages, int dev) {
    int64_t bb = 1;
    while (bb < B) bb <<= 1;
    const int64_t want = std::max<int64_t>(1, (sm_count(dev) * kv_ctas_per_sm(true, true, dev) * kv_waves() + bb * H - 1) / (bb * H));
    const int64_t cap = std::max<int64_t>(1, std::max<int64_t>((max_pages + kv_ppc_min() - 1) / kv_ppc_min(), std::min<int64_t>(kv_floor(B, H, dev), (max_pages + kvc::NW - 1) / kvc::NW)));
    return std::min(want, cap);
}

// A head of 256 or 512 channels is held as two or four of the page's heads of 128 (channels 0-127, then 128-255 ...): kv_heads(H, D) of them a page, at most 32.
// One stream: D with GLYD_GPU_KV_ONE_STREAM set (glyd_gpu.h) is the head dim of a tensor held in the K tile's layout alone (no V tile: a page is smaller by half its tiles), written from `key`, read into `out_k`.
// Only write, release, unpack and leaf_bytes take it; the others refuse it.
static const int64_t KV_ONE = 1 << 16;
static bool kv_one(int64_t D) { return (D & KV_ONE) != 0; }
static int64_t kv_dim(int64_t D) { return D & (KV_ONE - 1); }
static bool kv_shape_ok(int64_t H, int64_t D, int64_t A) {
    D = kv_dim(D);
    return H >= 1 && (D == kvc::D || D == 2 * kvc::D || D == 4 * kvc::D) && H * (D / kvc::D) <= 32 && A >= 0 && A <= 1023;
}
static int kv_heads(int64_t H, int64_t D) { return (int)(H * (kv_dim(D) / kvc::D)); }
static bool kv_escape_ok(const void* escape, int64_t nchunks) { return nchunks >= 0 && nchunks <= (1 << 30) && (nchunks == 0 || (escape && (uintptr_t)escape % 16 == 0)); }
static bool kv_raw_ok(const void* raw, int64_t nslabs) { return nslabs >= 0 && nslabs <= (1 << 30) && (nslabs == 0 || (raw && (uintptr_t)raw % 16 == 0)); }

GLYD_GPU_INTERNAL int glyd_gpu_kv_leaf_bytes(int64_t H, int64_t D, int64_t A, int64_t* bytes) {
    if (!bytes || !kv_shape_ok(H, D, A)) return cudaErrorInvalidValue;
    *bytes = kvc::page_bytes(kv_heads(H, D), (int)A, kv_one(D));
    return 0;
}

GLYD_GPU_INTERNAL int glyd_gpu_kv_extra_bytes(int64_t nchunks, int64_t* bytes) {
    if (!bytes || nchunks < 0 || nchunks > (1 << 30)) return cudaErrorInvalidValue;
    *bytes = kvc::esc_bytes(nchunks);
    return 0;
}

GLYD_GPU_INTERNAL int glyd_gpu_kv_second_bytes(int64_t nslabs, int64_t H, int64_t* bytes) {
    if (!bytes || nslabs < 0 || nslabs > (1 << 30) || H < 1 || H > 32) return cudaErrorInvalidValue;
    *bytes = kvc::raw_bytes(nslabs, (int)H);
    return 0;
}

GLYD_GPU_INTERNAL int glyd_gpu_kv_write(const uint16_t* key, int64_t key_stride, const uint16_t* value, int64_t value_stride, const int64_t* slots, int64_t N, uint8_t* pool, int64_t page_stride,
                                   const uint8_t* kbase, const uint8_t* vbase, int64_t H, int64_t D, int64_t A, uint8_t* escape, int64_t nchunks, uint8_t* raw, int64_t nslabs, int64_t layer, uint32_t* status,
                                   uint32_t* lost, int64_t nlost, int64_t ring, cudaStream_t cs) {
    int64_t need;
    if (glyd_gpu_kv_leaf_bytes(H, D, A, &need) || N < 0 || N > INT32_MAX || !status || page_stride < need || page_stride % 16 || (uintptr_t)pool % 16) return cudaErrorInvalidValue;
    if (!kv_escape_ok(escape, nchunks) || !kv_raw_ok(raw, nslabs) || layer < 0 || layer > 254 || nlost < 0 || (nlost && (!lost || (uintptr_t)lost % 4)) || ring < 1 || ring > (1 << 24)) return cudaErrorInvalidValue;
    if (!N) return 0;
    const int64_t Dm = kv_dim(D);
    if (!key || !slots || !kbase || key_stride < H * Dm || key_stride % 2 || (uintptr_t)key % 4) return cudaErrorInvalidValue;
    if (!kv_one(D) && (!value || !vbase || value_stride < H * Dm || value_stride % 2 || (uintptr_t)value % 4)) return cudaErrorInvalidValue;
    const int HP = kv_heads(H, D);  // (the page's heads of 128 channels: a head's 256 are two of them, its channels' halves in turn, as a row's keys are)
    kvc::WriteParams P{key, value, key_stride, value_stride, slots, pool, page_stride, kbase, vbase, status, kvc::esc_of(escape, nchunks), kvc::raw_of(raw, nslabs, HP), nlost ? lost : nullptr, nlost, (uint32_t)ring, (uint32_t)(layer + 1) << 24, (int)N, HP, (int)A, kv_lean() ? 1 : 0, kv_one(D) ? 1 : 0};
    kvc::write_kernel<<<(unsigned)N, 256, 0, cs>>>(P);
    return cudaGetLastError();
}

GLYD_GPU_INTERNAL int glyd_gpu_kv_release(uint8_t* pool, int64_t page_stride, int64_t nblocks, const int32_t* blocks, int64_t n, int64_t H, int64_t D, int64_t A, uint8_t* escape, int64_t nchunks, uint8_t* raw,
                                     int64_t nslabs, int64_t layer, uint32_t* status, uint32_t* lost, int64_t nlost, cudaStream_t cs) {
    int64_t need;
    if (glyd_gpu_kv_leaf_bytes(H, D, A, &need) || n < 0 || n > INT32_MAX || nblocks < 0 || !status || page_stride < need || page_stride % 16 || (uintptr_t)pool % 16) return cudaErrorInvalidValue;
    if (!kv_escape_ok(escape, nchunks) || !kv_raw_ok(raw, nslabs) || layer < 0 || layer > 254 || nlost < 0 || (nlost && (!lost || (uintptr_t)lost % 4))) return cudaErrorInvalidValue;
    if (!n) return 0;
    if (!blocks) return cudaErrorInvalidValue;
    const int HP = kv_heads(H, D);
    kvc::release_kernel<<<(unsigned)n, 32, 0, cs>>>(pool, page_stride, nblocks, blocks, HP, kvc::esc_of(escape, nchunks), kvc::raw_of(raw, nslabs, HP), (uint32_t)(layer + 1) << 24, status, nlost ? lost : nullptr, nlost);
    return cudaGetLastError();
}

GLYD_GPU_INTERNAL int glyd_gpu_kv_unpack(const uint8_t* pool, int64_t page_stride, const int32_t* pages, int64_t n, const uint8_t* kbase, const uint8_t* vbase, int64_t H, int64_t D, int64_t A,
                                    const uint8_t* escape, int64_t nchunks, const uint8_t* raw, int64_t nslabs, uint16_t* out_k, uint16_t* out_v, cudaStream_t cs) {
    int64_t need;
    const int HP = kv_heads(H, D);  // (out [n][16][H][D] is [n][16][HP][128]: a head's two halves in turn)
    if (glyd_gpu_kv_leaf_bytes(H, D, A, &need) || n < 0 || n * HP > INT32_MAX || page_stride < need || page_stride % 16 || (uintptr_t)pool % 16 || !kv_escape_ok(escape, nchunks) || !kv_raw_ok(raw, nslabs)) return cudaErrorInvalidValue;
    if (!n) return 0;
    if (!pages || !kbase || !out_k || (!kv_one(D) && (!vbase || !out_v))) return cudaErrorInvalidValue;
    const kvc::EscapePool E = kvc::esc_of((uint8_t*)escape, nchunks);
    kvc::unpack_kernel<<<(unsigned)((n * HP + 3) / 4), 128, 0, cs>>>(pool, page_stride, pages, (int)n, kbase, kv_one(D) ? kbase : vbase, HP, (int)A, E.ent, E.nc, kvc::raw_of((uint8_t*)raw, nslabs, HP), out_k, kv_one(D) ? out_k : out_v, kv_one(D) ? 1 : 0);
    return cudaGetLastError();
}

GLYD_GPU_INTERNAL int glyd_gpu_kv_gather(const uint8_t* pool, int64_t page_stride, const int32_t* pages, const int32_t* dst, int64_t n, const uint8_t* kbase, const uint8_t* vbase, int64_t H, int64_t D,
                                    int64_t A, const uint8_t* escape, int64_t nchunks, const uint8_t* raw, int64_t nslabs, uint16_t* out, cudaStream_t cs) {
    int64_t need;
    const int HP = kv_heads(H, D);
    if (kv_one(D) || glyd_gpu_kv_leaf_bytes(H, D, A, &need) || n < 0 || n * HP > INT32_MAX || page_stride < need || page_stride % 16 || (uintptr_t)pool % 16 || !kv_escape_ok(escape, nchunks) || !kv_raw_ok(raw, nslabs)) return cudaErrorInvalidValue;
    if (!n) return 0;
    if (!pages || !dst || !kbase || !vbase || !out || (uintptr_t)out % 4) return cudaErrorInvalidValue;
    const kvc::EscapePool E = kvc::esc_of((uint8_t*)escape, nchunks);
    const unsigned grid = (unsigned)((n * HP + 3) / 4);
    if (D == kvc::D) kvc::gather_kernel<1><<<grid, 128, 0, cs>>>(pool, page_stride, pages, dst, (int)n, kbase, vbase, HP, (int)A, E.ent, E.nc, kvc::raw_of((uint8_t*)raw, nslabs, HP), out);
    else if (D == 2 * kvc::D) kvc::gather_kernel<2><<<grid, 128, 0, cs>>>(pool, page_stride, pages, dst, (int)n, kbase, vbase, HP, (int)A, E.ent, E.nc, kvc::raw_of((uint8_t*)raw, nslabs, HP), out);
    else kvc::gather_kernel<4><<<grid, 128, 0, cs>>>(pool, page_stride, pages, dst, (int)n, kbase, vbase, HP, (int)A, E.ent, E.nc, kvc::raw_of((uint8_t*)raw, nslabs, HP), out);
    return cudaGetLastError();
}

GLYD_GPU_INTERNAL int glyd_gpu_kv_work_put(const uint16_t* key, int64_t key_stride, const uint16_t* value, int64_t value_stride, const int64_t* slots, int64_t n, int64_t H, int64_t D, uint16_t* out, int64_t blocks,
                                       cudaStream_t cs) {
    if (kv_one(D) || n < 0 || n > INT32_MAX || !kv_shape_ok(H, D, 0) || blocks < 0) return cudaErrorInvalidValue;
    if (!n) return 0;
    if (!key || !value || !slots || !out || key_stride < H * D || value_stride < H * D || key_stride % 2 || value_stride % 2 || (uintptr_t)key % 4 || (uintptr_t)value % 4 || (uintptr_t)out % 4) return cudaErrorInvalidValue;
    if (D == kvc::D) kvc::stage_put_kernel<kvc::D><<<(unsigned)n, 128, 0, cs>>>(key, key_stride, value, value_stride, slots, blocks * kvc::PT, (int)H, out);
    else if (D == 2 * kvc::D) kvc::stage_put_kernel<2 * kvc::D><<<(unsigned)n, 128, 0, cs>>>(key, key_stride, value, value_stride, slots, blocks * kvc::PT, (int)H, out);
    else kvc::stage_put_kernel<4 * kvc::D><<<(unsigned)n, 128, 0, cs>>>(key, key_stride, value, value_stride, slots, blocks * kvc::PT, (int)H, out);
    return cudaGetLastError();
}

GLYD_GPU_INTERNAL int glyd_gpu_kv_decode_shares(int64_t B, int64_t H, int64_t max_pages, int64_t* splits) {
    if (!splits || B < 1 || H < 1 || H > 32 || max_pages < 1) return cudaErrorInvalidValue;
    *splits = kv_splits(B, H, max_pages, current_device());
    return 0;
}

GLYD_GPU_INTERNAL int glyd_gpu_kv_decode_workspace(int64_t B, int64_t H, int64_t G, int64_t D, int64_t splits, size_t* bytes) {
    if (!bytes || B < 1 || kv_one(D) || !kv_shape_ok(H, D, 0) || G < 1 || G > 8 || splits < 1 || splits > 65535 || B * H > (int64_t)(KV_CNT_BYTES / sizeof(uint32_t))) return cudaErrorInvalidValue;
    *bytes = KV_CNT_BYTES + (size_t)(B * H * splits * G * (2 + D)) * sizeof(float);
    return 0;
}

GLYD_GPU_INTERNAL int glyd_gpu_kv_decode(const uint16_t* q, int64_t q_stride, const uint8_t* pool, int64_t page_stride, const uint8_t* kbase, const uint8_t* vbase, const int32_t* block_table, int64_t bt_stride,
                                    const int32_t* seq_lens, int64_t B, int64_t splits, int64_t H, int64_t G, int64_t D, int64_t A, const uint8_t* escape, int64_t nchunks, const uint8_t* raw, int64_t nslabs,
                                    double scale, uint16_t* out, int64_t out_stride, void* workspace, size_t workspace_bytes, cudaStream_t cs) {
    int64_t page;
    size_t need;
    if (kv_one(D) || glyd_gpu_kv_leaf_bytes(H, D, A, &page) || B < 0 || B > 65535 || page_stride < page || page_stride % 16 || (uintptr_t)pool % 16 || !kv_escape_ok(escape, nchunks) || !kv_raw_ok(raw, nslabs)) return cudaErrorInvalidValue;
    if (!B) return 0;
    if (glyd_gpu_kv_decode_workspace(B, H, G, D, splits, &need) || !fits(workspace, workspace_bytes, need)) return cudaErrorInvalidValue;
    if (!q || !kbase || !vbase || !block_table || !seq_lens || !out || q_stride < H * G * D || out_stride < H * G * D || q_stride % 2 || ((uintptr_t)q) % 4) return cudaErrorInvalidValue;
    const bool pipe = kv_pipe(), fuse = kv_fuse();
    uint32_t* ws_cnt = (uint32_t*)workspace;  // (the workspace's first KV_CNT_BYTES: the counters, zero when a call starts and when it ends)
    float* ws_ml = (float*)((uint8_t*)workspace + KV_CNT_BYTES);
    const kvc::EscapePool E = kvc::esc_of((uint8_t*)escape, nchunks);
    const int HP = kv_heads(H, D);
    kvc::AttnParams P{bf(q), q_stride, pool, page_stride, kbase, vbase, block_table, bt_stride, seq_lens, E.ent, E.nc, kvc::raw_of((uint8_t*)raw, nslabs, HP), ws_ml, ws_ml + B * H * splits * G * 2, (int)H, (int)G, (int)A, kvc::hdr_bytes(HP, (int)A), (int)splits,
                      HP, (float)(scale * 1.4426950408889634), kv_ppc_min(), kv_floor(B, H, current_device()), ws_cnt, bf(out), out_stride};
    const dim3 grid((unsigned)splits, (unsigned)H, (unsigned)B);
    const auto launch = [&](auto halves) {
        constexpr int HV = decltype(halves)::value;
        if (fuse) {
            if (pipe) kvc::attn_kernel<true, true, HV><<<grid, kvc::NW * 32, 0, cs>>>(P);
            else kvc::attn_kernel<false, true, HV><<<grid, kvc::NW * 32, 0, cs>>>(P);
        } else {
            if (pipe) kvc::attn_kernel<true, false, HV><<<grid, kvc::NW * 32, 0, cs>>>(P);
            else kvc::attn_kernel<false, false, HV><<<grid, kvc::NW * 32, 0, cs>>>(P);
            kvc::combine_kernel<HV * kvc::D><<<(unsigned)(B * H), 256, 0, cs>>>(P.ws_ml, P.ws_o, bf(out), out_stride, (int)H, (int)G, (int)splits, P.ppc_min, P.pfloor, seq_lens);
        }
    };
    if (D == kvc::D) launch(std::integral_constant<int, 1>{});
    else if (D == 2 * kvc::D) launch(std::integral_constant<int, 2>{});
    else launch(std::integral_constant<int, 4>{});
    return cudaGetLastError();
}

// The kernels' partition a sequence has at least (glyd_gpu_kv_decode's plan: its CTAs from the sequence's own length, at least
// this many, 1 to 64; 0: the GPU's own, as without the call): what a batch-invariant server asks for, in place of GLYD_KV_FLOOR.
GLYD_GPU_INTERNAL int glyd_gpu_kv_floor_set(int floor) {
    if (floor < 0 || floor > 64) return cudaErrorInvalidValue;
    g_kv_floor.store(floor, std::memory_order_relaxed);
    return 0;
}

// The lossless FP8 weights (glyd_gpu.h): pack in two calls (the size, then the arrays), unpack, and unpack of the matrices of a stack that a list names.
// A matrix's sizes: its patches (INT32_MAX at most: a grid's x).
static bool f8_shape(int64_t O, int64_t K, int64_t* patches) {
    if (O < 1 || K < 1 || O > INT32_MAX - 64 || K > INT32_MAX - 128) return false;
    const int64_t n = f8c::f8_cdiv(O, f8c::PR) * f8c::f8_cdiv(K, f8c::PC);
    if (n > INT32_MAX) return false;
    if (patches) *patches = n;
    return true;
}

GLYD_GPU_INTERNAL int glyd_gpu_f8_size_workspace(int64_t O, int64_t K, size_t* bytes) {
    if (!bytes || !f8_shape(O, K, nullptr)) return cudaErrorInvalidValue;
    *bytes = 16;
    return 0;
}

GLYD_GPU_INTERNAL int glyd_gpu_f8_size(const uint8_t* w, int64_t ld, int64_t O, int64_t K, uint32_t* heads, void* workspace, size_t workspace_bytes, int64_t* var_words, cudaStream_t cs) {
    int64_t patches;
    if (!w || !heads || !var_words || !f8_shape(O, K, &patches) || ld < K || (uintptr_t)heads % 16 || !workspace || (uintptr_t)workspace % 8 || !fits(workspace, workspace_bytes, 16)) return cudaErrorInvalidValue;
    const f8c::F8Pk P{w, ld, (int)O, (int)K, (int)f8c::f8_cdiv(K, f8c::PC), ((uintptr_t)w % 16 == 0 && ld % 16 == 0) ? 1 : 0};
    f8c::f8_head_kernel<<<(unsigned)patches, f8c::NT, 0, cs>>>(P, heads);
    f8c::f8_scan_kernel<<<1, f8c::SCAN_NT, 0, cs>>>(heads, patches, (uint64_t*)workspace);
    cudaError_t e = cudaGetLastError();
    uint64_t total = 0;
    if (e == cudaSuccess) e = cudaMemcpyAsync(&total, workspace, sizeof total, cudaMemcpyDeviceToHost, cs);
    if (e == cudaSuccess) e = cudaStreamSynchronize(cs);
    if (e != cudaSuccess) return e;
    if (total + f8c::GUARD >= (1ull << 32)) return cudaErrorInvalidValue;  // (a base is 32 bits)
    *var_words = (int64_t)total + f8c::GUARD;
    return 0;
}

GLYD_GPU_INTERNAL int glyd_gpu_f8_pack(const uint8_t* w, int64_t ld, int64_t O, int64_t K, const uint32_t* heads, uint8_t* core, uint32_t* var, int64_t var_words, cudaStream_t cs) {
    int64_t patches;
    if (!w || !heads || !core || !var || !f8_shape(O, K, &patches) || ld < K || var_words < f8c::GUARD || var_words >= (1ll << 32) || (uintptr_t)heads % 16 || (uintptr_t)core % 16) return cudaErrorInvalidValue;
    const f8c::F8Pk P{w, ld, (int)O, (int)K, (int)f8c::f8_cdiv(K, f8c::PC), ((uintptr_t)w % 16 == 0 && ld % 16 == 0) ? 1 : 0};
    f8c::f8_pack_kernel<<<(unsigned)patches, f8c::NT, 0, cs>>>(P, heads, core, var, var_words);
    return cudaGetLastError();
}

static int f8_unpack_launch(const uint8_t* core, const uint32_t* heads, const uint32_t* var, int64_t E, int64_t O, int64_t K, const int32_t* ids, int64_t n, uint8_t* out, int64_t ld, cudaStream_t cs) {
    int64_t patches;
    if (!core || !heads || !var || !out || !f8_shape(O, K, &patches) || ld < K || (uintptr_t)core % 16 || (uintptr_t)heads % 16) return cudaErrorInvalidValue;
    const f8c::F8Un P{core, heads, var, out, ids, ld, O * ld, (int)O, (int)K, (int)f8c::f8_cdiv(K, f8c::PC), (int)patches, (int)E, ((uintptr_t)out % 16 == 0 && ld % 16 == 0) ? 1 : 0};
    f8c::f8_unpack_kernel<<<dim3((unsigned)patches, (unsigned)n), f8c::NT, 0, cs>>>(P);
    return cudaGetLastError();
}

// Everything of W [O, K] into out [O, K], rows ld bytes apart. One launch, nothing allocated or waited for: a CUDA graph can hold it.
GLYD_GPU_INTERNAL int glyd_gpu_f8_unpack(const uint8_t* core, const uint32_t* heads, const uint32_t* var, int64_t O, int64_t K, uint8_t* out, int64_t ld, cudaStream_t cs) {
    return f8_unpack_launch(core, heads, var, 1, O, K, nullptr, 1, out, ld, cs);
}

// A stack of E matrices [O, K] (O a multiple of 64: the pack of the matrix [E O, K]): the matrices ids[i], i < n, into out [n, O, K] (rows ld bytes apart). An id
// outside [0, E) is skipped, its place in out left as it is. One launch for any ids, read on the device.
GLYD_GPU_INTERNAL int glyd_gpu_f8_unpack_experts(const uint8_t* core, const uint32_t* heads, const uint32_t* var, int64_t E, int64_t O, int64_t K, const int32_t* ids, int64_t n, uint8_t* out, int64_t ld, cudaStream_t cs) {
    if (E < 1 || E > INT32_MAX || n < 0 || n > 65535 || O % f8c::PR || (n && !ids)) return cudaErrorInvalidValue;
    if (!n) return 0;
    return f8_unpack_launch(core, heads, var, E, O, K, ids, n, out, ld, cs);
}

// Layout H (f8c's, for HBM GPUs): the calls above with its kernels, a patch's core f8c::HCORE bytes (heads and var as above: the size's words, the workspace's 16 bytes).
GLYD_GPU_INTERNAL int glyd_gpu_f8h_size(const uint8_t* w, int64_t ld, int64_t O, int64_t K, uint32_t* heads, void* workspace, size_t workspace_bytes, int64_t* var_words, cudaStream_t cs) {
    int64_t patches;
    if (!w || !heads || !var_words || !f8_shape(O, K, &patches) || ld < K || (uintptr_t)heads % 16 || !workspace || (uintptr_t)workspace % 8 || !fits(workspace, workspace_bytes, 16)) return cudaErrorInvalidValue;
    const f8c::F8Pk P{w, ld, (int)O, (int)K, (int)f8c::f8_cdiv(K, f8c::PC), ((uintptr_t)w % 16 == 0 && ld % 16 == 0) ? 1 : 0, f8h_plain() ? 1 : 0};
    f8c::f8h_head_kernel<<<(unsigned)patches, f8c::NT, 0, cs>>>(P, heads);
    f8c::f8_scan_kernel<<<1, f8c::SCAN_NT, 0, cs>>>(heads, patches, (uint64_t*)workspace);
    cudaError_t e = cudaGetLastError();
    uint64_t total = 0;
    if (e == cudaSuccess) e = cudaMemcpyAsync(&total, workspace, sizeof total, cudaMemcpyDeviceToHost, cs);
    if (e == cudaSuccess) e = cudaStreamSynchronize(cs);
    if (e != cudaSuccess) return e;
    if (total + f8c::GUARD >= (1ull << 32)) return cudaErrorInvalidValue;
    *var_words = (int64_t)total + f8c::GUARD;
    return 0;
}

GLYD_GPU_INTERNAL int glyd_gpu_f8h_pack(const uint8_t* w, int64_t ld, int64_t O, int64_t K, const uint32_t* heads, uint8_t* core, uint32_t* var, int64_t var_words, cudaStream_t cs) {
    int64_t patches;
    if (!w || !heads || !core || !var || !f8_shape(O, K, &patches) || ld < K || var_words < f8c::GUARD || var_words >= (1ll << 32) || (uintptr_t)heads % 16 || (uintptr_t)core % 16) return cudaErrorInvalidValue;
    const f8c::F8Pk P{w, ld, (int)O, (int)K, (int)f8c::f8_cdiv(K, f8c::PC), ((uintptr_t)w % 16 == 0 && ld % 16 == 0) ? 1 : 0, f8h_plain() ? 1 : 0};
    f8c::f8h_pack_kernel<<<(unsigned)patches, f8c::NT, 0, cs>>>(P, heads, core, var, var_words);
    return cudaGetLastError();
}

GLYD_GPU_INTERNAL int glyd_gpu_f8h_unpack(const uint8_t* core, const uint32_t* heads, const uint32_t* var, int64_t O, int64_t K, uint8_t* out, int64_t ld, cudaStream_t cs) {
    int64_t patches;
    if (!core || !heads || !var || !out || !f8_shape(O, K, &patches) || ld < K || (uintptr_t)core % 16 || (uintptr_t)heads % 16) return cudaErrorInvalidValue;
    const f8c::F8Un P{core, heads, var, out, nullptr, ld, O * ld, (int)O, (int)K, (int)f8c::f8_cdiv(K, f8c::PC), (int)patches, 1, ((uintptr_t)out % 16 == 0 && ld % 16 == 0) ? 1 : 0};
    f8c::f8h_unpack_kernel<<<(unsigned)patches, f8c::HUNT, 0, cs>>>(P);
    return cudaGetLastError();
}

// The fused product (glyd_gpu.h): its plan for W [N, K] and M tokens: 8-token tiles a warp (ntl), whether a warp decodes a step ahead (ahead: 17-64 tokens),
// whether it decodes a granule's spills in a pass of their own first (dense), whether X goes in as e4m3 (a8: W8A8, below), and the grid (ctas): as many CTAs
// as the GPU holds at once of the kernel for the kind (its SMs, and the kernel's registers and shared memory), at most one a row group's patch. Its
// workspace is that of the kind of the most CTAs, and X's e4m3 copy where a8.
// The default everywhere: bf16 X (W8A16), the grid of the most CTAs, no decode ahead, no spills' pass. Measured (benchmarks/gpu/fp8-speed-consumer-2026-10-06)
// slower than vLLM's own FP8 GEMMs on every shape at 32-64 tokens (at 1-8 tokens at or under them on some shapes and models), so not at every load: no
// served path takes it by default.
// a8 (development builds only, GLYD_F8_LIN_A8=1): block, tensor and channel scales, X as e4m3 a token and 128 columns (f8_xq_kernel), one e4m3 MMA a unit and
// 8 tokens, on compute capability 12.x where the code loaded was compiled for it; mx and ds32 never. Its error is about 15x bf16 X's, and the X scales are
// this kernel's own (vLLM quantizes X a token and 128 columns for block scales, a token for channel scales, and with the checkpoint's input scale for tensor).
// Not on Ada: an RTX 4090's e4m3 MMA sums fell outside check_f8.py's bound (one output in 1,690, 1% off where the sum of |x w| was 100 times it; 851cf2e8).
// Development builds, for the checks and for measuring: GLYD_F8_LIN_GRID = c > 0, c CTAs; GLYD_F8_LIN_MIN = m > 1, at least m patches a CTA;
// GLYD_F8_LIN_AHEAD = 1 (17-64 tokens); GLYD_F8_LIN_DENSE = 1; GLYD_F8_LIN_A8 = 1.
static int f8_lin_ntl(int64_t M) { return M <= 8 ? 1 : M <= 16 ? 2 : M <= 32 ? 4 : 8; }
static int f8_lin_cc(int dev) { return attribute(cudaDevAttrComputeCapabilityMajor, dev) * 10 + attribute(cudaDevAttrComputeCapabilityMinor, dev); }
static int f8_lin_a8(int64_t kind, int dev) {
    if ((kind != 0 && kind != 1 && kind != 4) || TUNING_INT("GLYD_F8_LIN_A8", 0) != 1 || f8_lin_cc(dev) / 10 != 12) return 0;
    cudaFuncAttributes fa;  // (the e4m3 code loaded must be compiled for 8.9 on: compute_80 PTX run on a newer GPU has none)
    return cudaFuncGetAttributes(&fa, (const void*)f8c::f8_xq_kernel) == cudaSuccess && fa.ptxVersion >= 89 ? 1 : 0;
}
static int f8_lin_ahead(int64_t M, int a8) {
    return !a8 && f8_lin_ntl(M) >= 4 && TUNING_INT("GLYD_F8_LIN_AHEAD", 0) == 1 ? 1 : 0;  // (asked for, 17-64 tokens; W8A8 never)
}
static int f8_lin_dense(int a8) { return !a8 && TUNING_INT("GLYD_F8_LIN_DENSE", 0) == 1 ? 1 : 0; }
template <int S = 2, bool H = false>
static size_t f8_lin_shared(int64_t M, int dense, int a8) { return (size_t)f8c::f8_lsmem<S, H>() + (dense ? (size_t)f8c::LW * f8c::EBS : 0) + (size_t)f8c::f8_xs<S>() * f8c::f8_xslot((int)M, a8); }
// (GLYD_F8_SLOTS=3: the one-token product's three slots, f8_linear_kernel's S; being measured on HBM GPUs)
static int f8_lin_slots(int64_t M, int a8, int ahead, int dense) { return !a8 && !ahead && !dense && M <= 8 && TUNING_INT("GLYD_F8_SLOTS", 2) == 3 ? 3 : 2; }
static size_t f8_lin_xq(int64_t K, int64_t M) { return (size_t)M * f8c::f8_cdiv(K, f8c::PC) * (f8c::PC + 4); }  // (X's e4m3 copy and its scales)
#ifdef GLYD_RELEASE
constexpr bool F8_DEV_PLANS = false;  // (a release build compiles the default plan's kernels alone: ahead, dense and a8 are development builds')
#else
constexpr bool F8_DEV_PLANS = true;
#endif
// f.template operator()<NTL, KIND, AHEAD, DENSE>() for the plan's kernel
template <int KIND, class F>
static int f8_lin_with(int ntl, int ahead, int dense, F&& f) {
#define F8_W(N_, A_, D_) return f.template operator()<N_, KIND, A_, D_>()
    if constexpr (F8_DEV_PLANS && (KIND & (f8c::F8_A8 | f8c::F8_H)) == 0) {
        switch (ntl) {
            case 1:
                if (dense) F8_W(1, 0, 1);
                break;
            case 2:
                if (dense) F8_W(2, 0, 1);
                break;
            case 4:
                if (ahead && dense) F8_W(4, 1, 1);
                if (ahead) F8_W(4, 1, 0);
                if (dense) F8_W(4, 0, 1);
                break;
            default:
                if (ahead && dense) F8_W(8, 1, 1);
                if (ahead) F8_W(8, 1, 0);
                if (dense) F8_W(8, 0, 1);
                break;
        }
    }
    switch (ntl) {
        case 1: F8_W(1, 0, 0);
        case 2: F8_W(2, 0, 0);
        case 4: F8_W(4, 0, 0);
        default: F8_W(8, 0, 0);
    }
#undef F8_W
    return 0;
}
template <int KIND, class F>
static int f8_lin_k(int a8, int ntl, int ahead, int dense, F&& f) {
    if constexpr (F8_DEV_PLANS && KIND != f8c::F8_MX && KIND != f8c::F8_DS32)
        if (a8) return f8_lin_with<KIND | f8c::F8_A8>(ntl, 0, 0, f);
    return f8_lin_with<KIND>(ntl, ahead, dense, f);
}
template <class F>
static int f8_lin_kind(int64_t kind, int a8, int ntl, int ahead, int dense, F&& f) {
    switch (kind) {
        case 0: return f8_lin_k<f8c::F8_BLOCK>(a8, ntl, ahead, dense, f);
        case 1: return f8_lin_k<f8c::F8_TENSOR>(a8, ntl, ahead, dense, f);
        case 2: return f8_lin_k<f8c::F8_MX>(a8, ntl, ahead, dense, f);
        case 3: return f8_lin_k<f8c::F8_DS32>(a8, ntl, ahead, dense, f);
        default: return f8_lin_k<f8c::F8_CHANNEL>(a8, ntl, ahead, dense, f);
    }
}
template <int NTL, int KIND, int AHEAD, int DENSE>
static void f8_lin_allow() {  // the kernel's limit of shared memory raised to its most (M = 64), once a device
    static std::atomic<uint64_t> allowed{0};
    const int dev = current_device();
    if (dev < 64 && (allowed.load() >> dev & 1)) return;
    cudaFuncSetAttribute((const void*)f8c::f8_linear_kernel<NTL, KIND, AHEAD, DENSE>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)f8_lin_shared(64, DENSE, (KIND & f8c::F8_A8) != 0));
    if (dev < 64) allowed.fetch_or(1ull << dev);
}
// The grid for kind (0-4; -1: the most of the five's).
static int64_t f8_lin_ctas(int64_t N, int64_t K, int64_t M, int64_t kind, int dev) {
    static std::atomic<int> resident[64][2][2][10][65];  // device, ahead, dense, kind (+ 5: a8), M: the CTAs an SM (0: not asked yet)
    const int64_t W = f8c::f8_cdiv(f8c::f8_cdiv(N, f8c::PR) * 4, f8c::LW) * f8c::f8_cdiv(K, f8c::PC);
    const int64_t c = TUNING_INT("GLYD_F8_LIN_GRID", 0);
    if (c > 0) return std::min<int64_t>(c, W);
    if (kind < 0) {
        int64_t most = 1;
        for (int k = 0; k < 5; k++) most = std::max(most, f8_lin_ctas(N, K, M, k, dev));
        return most;
    }
    const int a8 = f8_lin_a8(kind, dev), ahead = f8_lin_ahead(M, a8), dense = f8_lin_dense(a8);
    int r = dev < 64 ? resident[dev][ahead][dense][kind + 5 * a8][M].load(std::memory_order_relaxed) : 0;
    if (!r) {
        const int sl = f8_lin_slots(M, a8, ahead, dense);
        r = f8_lin_kind(kind, a8, f8_lin_ntl(M), ahead, dense, [&]<int N_, int K_, int A_, int D_>() {
            int nb = 0;
            if constexpr (N_ == 1 && A_ == 0 && D_ == 0 && !(K_ & f8c::F8_A8))
                if (sl == 3) {
                    cudaFuncSetAttribute((const void*)f8c::f8_linear_kernel<N_, K_, A_, D_, 0, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)f8_lin_shared<3>(64, 0, 0));
                    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, (const void*)f8c::f8_linear_kernel<N_, K_, A_, D_, 0, 3>, f8c::LNT, f8_lin_shared<3>(M, 0, 0)) != cudaSuccess) nb = 1;
                    return std::max(nb, 1);
                }
            f8_lin_allow<N_, K_, A_, D_>();
            if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, (const void*)f8c::f8_linear_kernel<N_, K_, A_, D_>, f8c::LNT, f8_lin_shared(M, D_, a8)) != cudaSuccess) nb = 1;
            return std::max(nb, 1);
        });
        if (dev < 64) resident[dev][ahead][dense][kind + 5 * a8][M].store(r, std::memory_order_relaxed);
    }
    const int64_t least = std::max<int64_t>(1, TUNING_INT("GLYD_F8_LIN_MIN", 1));
    return std::max<int64_t>(1, std::min<int64_t>(sm_count(dev) * r, W / least));
}
static size_t f8_lin_parts(int64_t ctas, int64_t M) { return ctas > 1 ? (size_t)ctas * 2 * M * 128 * 4 : 0; }  // (a CTA's first and last row group's sums)
// Layout H's one-token kernel (f8h_lin1_kernel; development builds: GLYD_F8H_ONE=0 runs f8_linear_kernel for one token as before).
// GLYD_F8H_ONE: 1 f8h_lin1_kernel (a step's copies a step ahead), 2 f8h_lin2_kernel (two steps ahead), 0 f8_linear_kernel as before. Being measured on HBM GPUs.
static int f8h_one() {
    const int m = TUNING_INT("GLYD_F8H_ONE", f8h_plain() ? 3 : 1);
    return m < 0 || m > 4 ? 1 : m;  // 1: f8h_lin1_kernel; 2: f8h_lin2_kernel; 3: f8b_lin1_kernel, plain packs alone; 4: f8h_lin3_kernel (research)
}
static size_t f8h_one_shared() {
    const int m = f8h_one();
    return (size_t)f8c::LW * (m == 4 ? f8c::L3W : m == 3 ? f8c::BW : m == 2 ? f8c::L2W : f8c::L1W) * 4;
}
static int f8h_one_c() { return TUNING_INT("GLYD_F8H_CTAS", 3) == 2 ? 2 : 3; }
static const void* f8h_one_kernel(int g) {
    const int c = f8h_one_c();
    if (f8h_one() == 4)
        return g ? (c == 2 ? (const void*)f8c::f8h_lin3_kernel<1, 2> : (const void*)f8c::f8h_lin3_kernel<1, 3>) : (c == 2 ? (const void*)f8c::f8h_lin3_kernel<0, 2> : (const void*)f8c::f8h_lin3_kernel<0, 3>);
    if (f8h_one() == 3)
        return g ? (c == 2 ? (const void*)f8c::f8b_lin1_kernel<1, 2> : (const void*)f8c::f8b_lin1_kernel<1, 3>) : (c == 2 ? (const void*)f8c::f8b_lin1_kernel<0, 2> : (const void*)f8c::f8b_lin1_kernel<0, 3>);
    if (f8h_one() == 2)
        return g ? (c == 2 ? (const void*)f8c::f8h_lin2_kernel<1, 2> : (const void*)f8c::f8h_lin2_kernel<1, 3>) : (c == 2 ? (const void*)f8c::f8h_lin2_kernel<0, 2> : (const void*)f8c::f8h_lin2_kernel<0, 3>);
    return g ? (c == 2 ? (const void*)f8c::f8h_lin1_kernel<1, 2> : (const void*)f8c::f8h_lin1_kernel<1, 3>) : (c == 2 ? (const void*)f8c::f8h_lin1_kernel<0, 2> : (const void*)f8c::f8h_lin1_kernel<0, 3>);
}
template <int G>
static void f8h_one_launch(const f8c::F8Lin& P1, dim3 grid, cudaStream_t cs) {
    const int c = f8h_one_c();
    if (f8h_one() == 4) {
        if (c == 2) f8c::f8h_lin3_kernel<G, 2><<<grid, f8c::LNT, f8h_one_shared(), cs>>>(P1);
        else f8c::f8h_lin3_kernel<G, 3><<<grid, f8c::LNT, f8h_one_shared(), cs>>>(P1);
    } else if (f8h_one() == 3) {
        if (c == 2) f8c::f8b_lin1_kernel<G, 2><<<grid, f8c::LNT, f8h_one_shared(), cs>>>(P1);
        else f8c::f8b_lin1_kernel<G, 3><<<grid, f8c::LNT, f8h_one_shared(), cs>>>(P1);
    } else if (f8h_one() == 2) {
        if (c == 2) f8c::f8h_lin2_kernel<G, 2><<<grid, f8c::LNT, f8h_one_shared(), cs>>>(P1);
        else f8c::f8h_lin2_kernel<G, 3><<<grid, f8c::LNT, f8h_one_shared(), cs>>>(P1);
    } else {
        if (c == 2) f8c::f8h_lin1_kernel<G, 2><<<grid, f8c::LNT, f8h_one_shared(), cs>>>(P1);
        else f8c::f8h_lin1_kernel<G, 3><<<grid, f8c::LNT, f8h_one_shared(), cs>>>(P1);
    }
}
static int64_t f8h_one_ctas(int64_t N, int64_t K, int64_t slots, int dev) {  // its grid a slot: the CTAs the GPU holds at once, at most a patch each
    static std::atomic<int> resident[64][2][2][4];
    const int g = slots > 0 ? 1 : 0, c = f8h_one_c() - 2, m = f8h_one() == 4 ? 3 : f8h_one() == 3 ? 2 : f8h_one() == 2 ? 1 : 0;
    int r = dev < 64 ? resident[dev][g][c][m].load(std::memory_order_relaxed) : 0;
    if (!r) {
        const void* k = f8h_one_kernel(g);
        cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)f8h_one_shared());
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&r, k, f8c::LNT, f8h_one_shared()) != cudaSuccess) r = 1;
        r = std::max(r, 1);
        if (dev < 64) resident[dev][g][c][m].store(r, std::memory_order_relaxed);
    }
    const int64_t W = f8c::f8_cdiv(f8c::f8_cdiv(N, f8c::PR) * 4, f8c::LW) * f8c::f8_cdiv(K, f8c::PC);
    return std::max<int64_t>(1, std::min<int64_t>(sm_count(dev) * r / std::max<int64_t>(slots, 1), W));
}

GLYD_GPU_INTERNAL int glyd_gpu_f8_linear_workspace(int64_t N, int64_t K, int64_t M, size_t* bytes) {
    if (!bytes || M < 1 || M > 64 || !f8_shape(N, K, nullptr)) return cudaErrorInvalidValue;
    const int dev = current_device();
    *bytes = (f8_lin_parts(std::max(f8_lin_ctas(N, K, M, -1, dev), M == 1 ? f8h_one_ctas(N, K, 0, dev) : 1), M) + 15) / 16 * 16 + (f8_lin_a8(0, dev) ? f8_lin_xq(K, M) : 0);
    return 0;
}

// Y [M, N] = X [M, K] W^T for M of 1 to 64, W [N, K] in the lossless layout and its scales (kind 0: fp32 a 128 x 128 block, [ceil(N / 128)] rows lds floats
// apart; 1: one fp32; 2: e8m0 a 1 x 32 block, [N] rows lds bytes apart; 3: e8m0 a 32 x 32 block, [ceil(N / 32)] rows lds bytes apart; 4: fp32 a row, [N]).
// X and Y bf16, rows ldx and ldy elements apart. workspace: glyd_gpu_f8_linear_workspace's bytes; done: ceil(N / 128) counters, zero at first and left zero.
// One launch (two where X goes in as e4m3: its quantization first), nothing waited for.
GLYD_GPU_INTERNAL int glyd_gpu_f8_linear(const uint8_t* core, const uint32_t* heads, const uint32_t* var, int64_t N, int64_t K, int64_t kind, const void* scale, int64_t lds,
                                         const uint16_t* x, int64_t ldx, int64_t M, uint16_t* y, int64_t ldy, void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs) {
    int64_t patches;
    if (!core || !heads || !var || !scale || !x || !y || !f8_shape(N, K, &patches) || M < 1 || M > 64 || kind < 0 || kind > 4 || ldx < K || ldy < N ||
        (uintptr_t)core % 16 || (uintptr_t)heads % 16 || (uintptr_t)var % 4 || (uintptr_t)x % 2 || (uintptr_t)y % 2)
        return cudaErrorInvalidValue;
    if ((kind == 0 && (lds < f8c::f8_cdiv(K, 128) || (uintptr_t)scale % 4)) || ((kind == 2 || kind == 3) && lds < f8c::f8_cdiv(K, 32)) || ((kind == 1 || kind == 4) && (uintptr_t)scale % 4))
        return cudaErrorInvalidValue;
    const int dev = current_device();
    const int64_t ctas = f8_lin_ctas(N, K, M, kind, dev), KT = f8c::f8_cdiv(K, f8c::PC);
    const int a8 = f8_lin_a8(kind, dev), ahead = f8_lin_ahead(M, a8), dense = f8_lin_dense(a8);
    const size_t parts = (f8_lin_parts(ctas, M) + 15) / 16 * 16, need = parts + (a8 ? f8_lin_xq(K, M) : 0);
    if ((ctas > 1 && !done) || !fits(workspace, workspace_bytes, need) || (need && (uintptr_t)workspace % 16)) return cudaErrorInvalidValue;
    uint8_t* xq = a8 ? (uint8_t*)workspace + parts : nullptr;
    float* sx = a8 ? (float*)(xq + (size_t)M * KT * f8c::PC) : nullptr;
    const f8c::F8Lin P{core, heads, var, scale, x, y, xq, sx, (float*)workspace, done, lds, ldx, ldy, (int)M, (int)N, (int)K, (int)KT,
                       (int)f8c::f8_cdiv(f8c::f8_cdiv(N, f8c::PR) * 4, f8c::LW), (int)ctas, ((uintptr_t)x % 16 == 0 && ldx % 8 == 0) ? 1 : 0};
    if (a8) {
        f8c::f8_xq_kernel<<<(unsigned)f8c::f8_cdiv(M * KT, 4), 128, 0, cs>>>(x, ldx, (int)M, (int)K, (int)KT, xq, sx);
        if (cudaGetLastError() != cudaSuccess) return cudaErrorLaunchFailure;
    }
    const size_t shared = f8_lin_shared(M, dense, a8);
    const int sl = f8_lin_slots(M, a8, ahead, dense);
    return f8_lin_kind(kind, a8, f8_lin_ntl(M), ahead, dense, [&]<int N_, int K_, int A_, int D_>() {
        if constexpr (N_ == 1 && A_ == 0 && D_ == 0 && !(K_ & f8c::F8_A8))
            if (sl == 3) {
                cudaFuncSetAttribute((const void*)f8c::f8_linear_kernel<N_, K_, A_, D_, 0, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)f8_lin_shared<3>(64, 0, 0));
                f8c::f8_linear_kernel<N_, K_, A_, D_, 0, 3><<<(unsigned)P.ctas, f8c::LNT, f8_lin_shared<3>(M, 0, 0), cs>>>(P);
                return (int)cudaGetLastError();
            }
        f8_lin_allow<N_, K_, A_, D_>();
        f8c::f8_linear_kernel<N_, K_, A_, D_><<<(unsigned)P.ctas, f8c::LNT, shared, cs>>>(P);
        return (int)cudaGetLastError();
    });
}

// The experts' product's grid: a slot (blockIdx.y) of the plan for each of `slots`, and as many CTAs a slot (blockIdx.x) as the GPU holds at once of the kernel shared among them,
// at most one a row group's patch of a matrix.
static int64_t f8_moe_ctas(int64_t N, int64_t K, int64_t M, int64_t slots, int dev) {
    const int64_t W = f8c::f8_cdiv(f8c::f8_cdiv(N, f8c::PR) * 4, f8c::LW) * f8c::f8_cdiv(K, f8c::PC);
    const int64_t c = TUNING_INT("GLYD_F8_MOE_GRID", 0);
    if (c > 0) return std::min<int64_t>(c, W);
    static std::atomic<int> resident[64][65];
    int r = dev < 64 ? resident[dev][M].load(std::memory_order_relaxed) : 0;
    if (!r) {
        const int sl = f8_lin_slots(M, 0, 0, 0);
        r = f8_lin_with<f8c::F8_BLOCK>(f8_lin_ntl(M), 0, 0, [&]<int N_, int K_, int A_, int D_>() {
            int nb = 0;
            if constexpr (N_ == 1 && A_ == 0 && D_ == 0)
                if (sl == 3) {
                    cudaFuncSetAttribute((const void*)f8c::f8_linear_kernel<N_, K_, A_, D_, 1, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)f8_lin_shared<3>(64, 0, 0));
                    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, (const void*)f8c::f8_linear_kernel<N_, K_, A_, D_, 1, 3>, f8c::LNT, f8_lin_shared<3>(M, 0, 0)) != cudaSuccess) nb = 1;
                    return std::max(nb, 1);
                }
            cudaFuncSetAttribute((const void*)f8c::f8_linear_kernel<N_, K_, A_, D_, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)f8_lin_shared(64, D_, 0));
            if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, (const void*)f8c::f8_linear_kernel<N_, K_, A_, D_, 1>, f8c::LNT, f8_lin_shared(M, D_, 0)) != cudaSuccess) nb = 1;
            return std::max(nb, 1);
        });
        if (dev < 64) resident[dev][M].store(r, std::memory_order_relaxed);
    }
    return std::max<int64_t>(1, std::min<int64_t>(sm_count(dev) * r / std::max<int64_t>(slots, 1), W));
}

GLYD_GPU_INTERNAL int glyd_gpu_f8_moe_workspace(int64_t N, int64_t K, int64_t M, int64_t slots, size_t* bytes) {
    if (!bytes || M < 1 || M > 64 || slots < 1 || slots > 65535 || !f8_shape(N, K, nullptr)) return cudaErrorInvalidValue;
    const int dev = current_device();
    *bytes = (size_t)slots * f8_lin_parts(std::max(f8_moe_ctas(N, K, M, slots, dev), M == 1 ? f8h_one_ctas(N, K, slots, dev) : 1), M);
    return 0;
}

// The experts' product: E matrices [N, K] (N a multiple of 64) packed as one [E N, K] in the lossless layout, their block scales (fp32 a 128 x 128 block, [E][ceil(N / 128)] rows lds floats
// apart), and the plan of moe_route_kernel (glyd_gpu_moe_way) of P pairs (k a token): each pair's row of Y [P, N] (in the plan's order, rows ldy apart) = its X row times its expert's matrix^T,
// X's row the pair's token's (gather: X [tokens, K]) or the pair's own in the plan's order (X [P, K]); rows ldx apart. M: the most pairs an expert has (1 to 64: a token's k choices are
// distinct, so the tokens), slots: the most experts the plan hits (at most 65535; slots past the plan's do nothing). workspace: glyd_gpu_f8_moe_workspace's bytes; done: slots ceil(N / 128)
// counters, zero at first and left zero. One launch, read on the device (a CUDA graph holds it).
GLYD_GPU_INTERNAL int glyd_gpu_f8_moe(const uint8_t* core, const uint32_t* heads, const uint32_t* var, int64_t E, int64_t N, int64_t K, const float* scale, int64_t lds, const uint16_t* x,
                                      int64_t ldx, const int32_t* plan, int64_t k, int64_t gather, int64_t M, int64_t slots, uint16_t* y, int64_t ldy, void* workspace, size_t workspace_bytes,
                                      int* done, cudaStream_t cs) {
    if (!core || !heads || !var || !scale || !x || !y || !plan || E < 1 || E > INT32_MAX || N % f8c::PR || !f8_shape(N, K, nullptr) || M < 1 || M > 64 || slots < 1 || slots > 65535 ||
        k < 1 || k > INT32_MAX || ldx < K || ldy < N || lds < f8c::f8_cdiv(K, 128) || (uintptr_t)core % 16 || (uintptr_t)heads % 16 || (uintptr_t)var % 4 || (uintptr_t)scale % 4 ||
        (uintptr_t)x % 2 || (uintptr_t)y % 2)
        return cudaErrorInvalidValue;
    const int dev = current_device();
    const int64_t ctas = f8_moe_ctas(N, K, M, slots, dev), KT = f8c::f8_cdiv(K, f8c::PC);
    const size_t need = (size_t)slots * f8_lin_parts(ctas, M);
    if ((ctas > 1 && !done) || !fits(workspace, workspace_bytes, need) || (need && (uintptr_t)workspace % 16)) return cudaErrorInvalidValue;
    const f8c::F8Lin P{core, heads, var, scale, x, y, nullptr, nullptr, (float*)workspace, done, lds, ldx, ldy, (int)M, (int)N, (int)K, (int)KT,
                       (int)f8c::f8_cdiv(f8c::f8_cdiv(N, f8c::PR) * 4, f8c::LW), (int)ctas, ((uintptr_t)x % 16 == 0 && ldx % 8 == 0) ? 1 : 0, plan, nullptr, (int)E, (int)k, gather ? 1 : 0};
    const size_t shared = f8_lin_shared(M, 0, 0);
    const int sl = f8_lin_slots(M, 0, 0, 0);
    return f8_lin_with<f8c::F8_BLOCK>(f8_lin_ntl(M), 0, 0, [&]<int N_, int K_, int A_, int D_>() {
        if constexpr (N_ == 1 && A_ == 0 && D_ == 0)
            if (sl == 3) {
                cudaFuncSetAttribute((const void*)f8c::f8_linear_kernel<N_, K_, A_, D_, 1, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)f8_lin_shared<3>(64, 0, 0));
                f8c::f8_linear_kernel<N_, K_, A_, D_, 1, 3><<<dim3((unsigned)ctas, (unsigned)slots), f8c::LNT, f8_lin_shared<3>(M, 0, 0), cs>>>(P);
                return (int)cudaGetLastError();
            }
        cudaFuncSetAttribute((const void*)f8c::f8_linear_kernel<N_, K_, A_, D_, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)f8_lin_shared(64, D_, 0));
        f8c::f8_linear_kernel<N_, K_, A_, D_, 1><<<dim3((unsigned)ctas, (unsigned)slots), f8c::LNT, shared, cs>>>(P);
        return (int)cudaGetLastError();
    });
}

// Layout H's products (block scales alone: kind 0), as glyd_gpu_f8_linear's and glyd_gpu_f8_moe's: the same workspaces (their grids at most those above's); one token
// goes to f8h_lin1_kernel (the same plan and sums, a warp's own copies).
template <int G>
static int f8h_resident(int64_t M, int dev) {  // the CTAs an SM of the kernel for M tokens
    static std::atomic<int> resident[64][65];
    int r = dev < 64 ? resident[dev][M].load(std::memory_order_relaxed) : 0;
    if (!r) {
        r = f8_lin_with<f8c::F8_BLOCK | f8c::F8_H>(f8_lin_ntl(M), 0, 0, [&]<int N_, int K_, int A_, int D_>() {
            int nb = 0;
            cudaFuncSetAttribute((const void*)f8c::f8_linear_kernel<N_, K_, 0, 0, G>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)f8_lin_shared<2, true>(64, 0, 0));
            if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, (const void*)f8c::f8_linear_kernel<N_, K_, 0, 0, G>, f8c::LNT, f8_lin_shared<2, true>(M, 0, 0)) != cudaSuccess) nb = 1;
            return std::max(nb, 1);
        });
        if (dev < 64) resident[dev][M].store(r, std::memory_order_relaxed);
    }
    return r;
}

GLYD_GPU_INTERNAL int glyd_gpu_f8h_linear(const uint8_t* core, const uint32_t* heads, const uint32_t* var, int64_t N, int64_t K, int64_t kind, const void* scale, int64_t lds,
                                          const uint16_t* x, int64_t ldx, int64_t M, uint16_t* y, int64_t ldy, void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs) {
    int64_t patches;
    if (!core || !heads || !var || !scale || !x || !y || !f8_shape(N, K, &patches) || M < 1 || M > 64 || kind != 0 || ldx < K || ldy < N || lds < f8c::f8_cdiv(K, 128) ||
        (uintptr_t)core % 16 || (uintptr_t)heads % 16 || (uintptr_t)var % 4 || (uintptr_t)scale % 4 || (uintptr_t)x % 2 || (uintptr_t)y % 2)
        return cudaErrorInvalidValue;
    const int dev = current_device();
    const int64_t W = f8c::f8_cdiv(f8c::f8_cdiv(N, f8c::PR) * 4, f8c::LW) * f8c::f8_cdiv(K, f8c::PC);
    const int64_t ctas = std::min<int64_t>(f8_lin_ctas(N, K, M, 0, dev), std::max<int64_t>(1, std::min<int64_t>(sm_count(dev) * f8h_resident<0>(M, dev), W)));
    const size_t need = (f8_lin_parts(ctas, M) + 15) / 16 * 16;
    if ((ctas > 1 && !done) || !fits(workspace, workspace_bytes, need) || (need && (uintptr_t)workspace % 16)) return cudaErrorInvalidValue;
    const f8c::F8Lin P{core, heads, var, scale, x, y, nullptr, nullptr, (float*)workspace, done, lds, ldx, ldy, (int)M, (int)N, (int)K, (int)f8c::f8_cdiv(K, f8c::PC),
                       (int)f8c::f8_cdiv(f8c::f8_cdiv(N, f8c::PR) * 4, f8c::LW), (int)ctas, ((uintptr_t)x % 16 == 0 && ldx % 8 == 0) ? 1 : 0};
    if (M == 1 && f8h_one()) {
        f8c::F8Lin P1 = P;
        P1.ctas = (int)f8h_one_ctas(N, K, 0, dev);
        if ((P1.ctas > 1 && !done) || !fits(workspace, workspace_bytes, (f8_lin_parts(P1.ctas, 1) + 15) / 16 * 16)) return cudaErrorInvalidValue;
        f8h_one_launch<0>(P1, dim3((unsigned)P1.ctas), cs);
        return (int)cudaGetLastError();
    }
    const size_t shared = f8_lin_shared<2, true>(M, 0, 0);
    return f8_lin_with<f8c::F8_BLOCK | f8c::F8_H>(f8_lin_ntl(M), 0, 0, [&]<int N_, int K_, int A_, int D_>() {
        cudaFuncSetAttribute((const void*)f8c::f8_linear_kernel<N_, K_, 0, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)f8_lin_shared<2, true>(64, 0, 0));
        f8c::f8_linear_kernel<N_, K_, 0, 0><<<(unsigned)P.ctas, f8c::LNT, shared, cs>>>(P);
        return (int)cudaGetLastError();
    });
}

GLYD_GPU_INTERNAL int glyd_gpu_f8h_moe(const uint8_t* core, const uint32_t* heads, const uint32_t* var, int64_t E, int64_t N, int64_t K, const float* scale, int64_t lds, const uint16_t* x,
                                       int64_t ldx, const int32_t* plan, int64_t k, int64_t gather, int64_t M, int64_t slots, uint16_t* y, int64_t ldy, void* workspace, size_t workspace_bytes,
                                       int* done, cudaStream_t cs) {
    if (!core || !heads || !var || !scale || !x || !y || !plan || E < 1 || E > INT32_MAX || N % f8c::PR || !f8_shape(N, K, nullptr) || M < 1 || M > 64 || slots < 1 || slots > 65535 ||
        k < 1 || k > INT32_MAX || ldx < K || ldy < N || lds < f8c::f8_cdiv(K, 128) || (uintptr_t)core % 16 || (uintptr_t)heads % 16 || (uintptr_t)var % 4 || (uintptr_t)scale % 4 ||
        (uintptr_t)x % 2 || (uintptr_t)y % 2)
        return cudaErrorInvalidValue;
    const int dev = current_device();
    const int64_t W = f8c::f8_cdiv(f8c::f8_cdiv(N, f8c::PR) * 4, f8c::LW) * f8c::f8_cdiv(K, f8c::PC);
    const int64_t ctas = std::min<int64_t>(f8_moe_ctas(N, K, M, slots, dev), std::max<int64_t>(1, std::min<int64_t>(sm_count(dev) * f8h_resident<1>(M, dev) / slots, W)));
    const size_t need = (size_t)slots * f8_lin_parts(ctas, M);
    if ((ctas > 1 && !done) || !fits(workspace, workspace_bytes, need) || (need && (uintptr_t)workspace % 16)) return cudaErrorInvalidValue;
    const f8c::F8Lin P{core, heads, var, scale, x, y, nullptr, nullptr, (float*)workspace, done, lds, ldx, ldy, (int)M, (int)N, (int)K, (int)f8c::f8_cdiv(K, f8c::PC),
                       (int)f8c::f8_cdiv(f8c::f8_cdiv(N, f8c::PR) * 4, f8c::LW), (int)ctas, ((uintptr_t)x % 16 == 0 && ldx % 8 == 0) ? 1 : 0, plan, nullptr, (int)E, (int)k, gather ? 1 : 0};
    if (M == 1 && f8h_one()) {
        f8c::F8Lin P1 = P;
        P1.ctas = (int)f8h_one_ctas(N, K, slots, dev);
        if ((P1.ctas > 1 && !done) || !fits(workspace, workspace_bytes, (size_t)slots * f8_lin_parts(P1.ctas, 1))) return cudaErrorInvalidValue;
        f8h_one_launch<1>(P1, dim3((unsigned)P1.ctas, (unsigned)slots), cs);
        return (int)cudaGetLastError();
    }
    const size_t shared = f8_lin_shared<2, true>(M, 0, 0);
    return f8_lin_with<f8c::F8_BLOCK | f8c::F8_H>(f8_lin_ntl(M), 0, 0, [&]<int N_, int K_, int A_, int D_>() {
        cudaFuncSetAttribute((const void*)f8c::f8_linear_kernel<N_, K_, 0, 0, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)f8_lin_shared<2, true>(64, 0, 0));
        f8c::f8_linear_kernel<N_, K_, 0, 0, 1><<<dim3((unsigned)ctas, (unsigned)slots), f8c::LNT, shared, cs>>>(P);
        return (int)cudaGetLastError();
    });
}

#ifdef TORCH_EXTENSION_NAME
// The pybind module: the C API on tensors, on the current stream of their
// device; outputs, workspaces and counters allocated as PyTorch tensors.
static cudaStream_t current_stream() { return at::cuda::getCurrentCUDAStream(); }
static void ok(int r, const char* fn) { TORCH_CHECK(r == 0, fn, ": ", cudaGetErrorString((cudaError_t)r)); }
template <class T> static T* ptr(const torch::Tensor& t) { return (T*)t.data_ptr(); }
static const uint16_t* opt(const torch::Tensor& t) { return t.numel() ? ptr<const uint16_t>(t) : nullptr; }  // a bias: none when empty
static void* addr(const torch::Tensor& t) { return t.defined() ? t.data_ptr() : nullptr; }

// A product's workspace (none for 0 bytes).
static torch::Tensor scratch(size_t bytes, const torch::Tensor& like) { return bytes ? torch::empty({(int64_t)bytes}, like.options().dtype(torch::kUInt8)) : torch::Tensor(); }

// A product's done counters on like's device for the current stream (at least n; least when first made): zero
// between products, a set a stream (one stream's products at a time on a set).
using Counters = std::map<std::pair<int, cudaStream_t>, torch::Tensor>;
static int* counters(Counters& of, const torch::Tensor& like, int64_t n, int64_t least) {
    torch::Tensor& t = of[{like.get_device(), current_stream()}];
    if (!t.defined() || t.numel() < n) t = torch::zeros({std::max<int64_t>(n, least)}, like.options().dtype(torch::kInt32));
    return ptr<int>(t);
}

static void words(const std::vector<int64_t>& v, size_t n, uint32_t* w, const char* what) {
    TORCH_CHECK(v.size() == n, what);
    for (size_t i = 0; i < n; i++) w[i] = (uint32_t)v[i];
}

torch::Tensor pass_a(torch::Tensor w, torch::Tensor len, int64_t tw, int64_t V) {
    const c10::cuda::CUDAGuard guard(w.device());
    auto bits = torch::empty({tiles_for(w.numel(), tw) * 32}, w.options().dtype(torch::kInt32));
    ok(glyd_gpu_pass_a(ptr<uint16_t>(w), w.numel(), ptr<uint8_t>(len), tw, V, ptr<uint32_t>(bits), current_stream()), "pass_a");
    return bits;
}

void pass_b(torch::Tensor w, torch::Tensor len, torch::Tensor code, torch::Tensor offs, torch::Tensor out, int64_t tw, int64_t V) {
    const c10::cuda::CUDAGuard guard(w.device());
    ok(glyd_gpu_pass_b(ptr<uint16_t>(w), w.numel(), ptr<uint8_t>(len), ptr<uint32_t>(code), ptr<uint32_t>(offs), ptr<uint32_t>(out), tw, V, current_stream()), "pass_b");
}

void decode(torch::Tensor sm, torch::Tensor stream, torch::Tensor offs, torch::Tensor tables, int64_t n, int64_t tw, int64_t V, int64_t tile_words, torch::Tensor tile_ids, torch::Tensor out) {
    const c10::cuda::CUDAGuard guard(sm.device());
    bool all = tile_ids.numel() == 0;
    TORCH_CHECK(out.numel() >= (all ? n : tile_ids.numel() * tw), "the output is too small");
    ok(glyd_gpu_decode(ptr<uint8_t>(sm), ptr<uint32_t>(stream), stream.numel(), ptr<uint32_t>(offs), ptr<uint32_t>(tables), n, tw, V, tile_words, ptr<int64_t>(tile_ids), tile_ids.numel(), ptr<uint16_t>(out), current_stream()), "decode");
}

void gemv(torch::Tensor sm, torch::Tensor stream, torch::Tensor offs, torch::Tensor tables, int64_t O, int64_t K, int64_t tw, int64_t V, int64_t tile_words, torch::Tensor x, torch::Tensor bias, torch::Tensor y, torch::Tensor sum, torch::Tensor count) {
    const c10::cuda::CUDAGuard guard(sm.device());
    bool split = tw % K != 0;
    TORCH_CHECK(K % (32 * V) == 0 && tw % (32 * V) == 0 && (!split || sum.numel() >= O), "gemv needs K and tiles multiples of 32 V, and row sums for split rows");
    ok(glyd_gpu_gemv(ptr<uint8_t>(sm), ptr<uint32_t>(stream), stream.numel(), ptr<uint32_t>(offs), ptr<uint32_t>(tables), O, K, tw, V, tile_words, ptr<uint16_t>(x), opt(bias), ptr<uint16_t>(y), ptr<float>(sum), ptr<int>(count), current_stream()), "gemv");
}

void fast_gemv(torch::Tensor sm, torch::Tensor planes, torch::Tensor exc, torch::Tensor exc_base, int64_t top, int64_t O, int64_t K, torch::Tensor x, torch::Tensor bias, torch::Tensor y) {
    const c10::cuda::CUDAGuard guard(sm.device());
    TORCH_CHECK(K % 128 == 0, "rows a multiple of 128 long");
    ok(glyd_gpu_fast_gemv(ptr<uint8_t>(sm), ptr<uint32_t>(planes), ptr<uint8_t>(exc), ptr<int32_t>(exc_base), (uint64_t)top, O, K, ptr<uint16_t>(x), opt(bias), ptr<uint16_t>(y), current_stream()), "fast_gemv");
}

void fast_decode(torch::Tensor sm, torch::Tensor planes, torch::Tensor exc, torch::Tensor exc_base, int64_t top, int64_t row0, int64_t rows, torch::Tensor row_ids, int64_t K, torch::Tensor out) {
    const c10::cuda::CUDAGuard guard(sm.device());
    int64_t ids = row_ids.numel();
    TORCH_CHECK(K % 128 == 0 && out.numel() >= (ids ? ids : rows) * K, "rows a multiple of 128 long, room for them");
    ok(glyd_gpu_fast_decode(ptr<uint8_t>(sm), ptr<uint32_t>(planes), ptr<uint8_t>(exc), ptr<int32_t>(exc_base), (uint64_t)top, row0, rows, ptr<int64_t>(row_ids), ids, K, ptr<uint16_t>(out), current_stream()), "fast_decode");
}

void fast_gemm(torch::Tensor sm, torch::Tensor planes, torch::Tensor exc, torch::Tensor exc_base, int64_t top, int64_t O, int64_t K, torch::Tensor x, torch::Tensor bias, torch::Tensor y) {
    const c10::cuda::CUDAGuard guard(sm.device());
    TORCH_CHECK(K % GM_BK == 0 && x.is_contiguous() && x.size(1) == K, "K a multiple of 64, X contiguous [M, K]");
    int64_t M = x.size(0);
    size_t bytes = 0;
    ok(glyd_gpu_fast_gemm_workspace(O, K, M, &bytes), "fast_gemm");
    auto ws = scratch(bytes, x);
    ok(glyd_gpu_fast_gemm(ptr<uint8_t>(sm), ptr<uint32_t>(planes), ptr<uint8_t>(exc), ptr<int32_t>(exc_base), (uint64_t)top, O, K, ptr<uint16_t>(x), M, opt(bias), ptr<uint16_t>(y), addr(ws), bytes, current_stream()), "fast_gemm");
}

void fast_bgemv(torch::Tensor sm, torch::Tensor planes, torch::Tensor exc, torch::Tensor exc_base, int64_t top, int64_t O, int64_t K, torch::Tensor x, torch::Tensor bias, torch::Tensor y) {
    const c10::cuda::CUDAGuard guard(sm.device());
    int64_t M = x.size(0);
    TORCH_CHECK(K % 512 == 0 && x.is_contiguous() && x.size(1) == K && (M == 2 || M == 4 || M == 8 || M == 16), "K a multiple of 512, X contiguous [M, K], M 2, 4, 8 or 16");
    size_t bytes = 0;
    ok(glyd_gpu_fast_bgemv_workspace(O, K, M, &bytes), "fast_bgemv");
    auto ws = scratch(bytes, x);
    ok(glyd_gpu_fast_bgemv(ptr<uint8_t>(sm), ptr<uint32_t>(planes), ptr<uint8_t>(exc), ptr<int32_t>(exc_base), (uint64_t)top, O, K, ptr<uint16_t>(x), M, opt(bias), ptr<uint16_t>(y), addr(ws), bytes, current_stream()), "fast_bgemv");
}

void mma_gemm(torch::Tensor data, torch::Tensor blocks, torch::Tensor block_base, std::vector<int64_t> tiers, int64_t O, int64_t K, torch::Tensor x, torch::Tensor bias, torch::Tensor y) {
    uint32_t t[3];
    words(tiers, 3, t, "three tiers");
    const c10::cuda::CUDAGuard guard(data.device());
    int64_t M = x.size(0);
    TORCH_CHECK(O % 64 == 0 && K % 16 == 0 && M <= 64 && x.is_contiguous() && x.size(1) == K, "O a multiple of 64, K of 16, up to 64 tokens, X contiguous [M, K]");
    size_t bytes = 0;
    ok(glyd_gpu_mma_gemm_workspace(O, K, M, &bytes), "mma_gemm");
    auto ws = scratch(bytes, x);
    static auto* done_of = new Counters;  // kept to the end (no teardown after CUDA's)
    ok(glyd_gpu_mma_gemm(ptr<uint8_t>(data), ptr<uint8_t>(blocks), ptr<int32_t>(block_base), t, O, K, ptr<uint16_t>(x), M, opt(bias), ptr<uint16_t>(y), addr(ws), bytes, counters(*done_of, data, O / 64, 1 << 16), current_stream()), "mma_gemm");
}

void mma12_gemm(torch::Tensor data, torch::Tensor exc, torch::Tensor exc_base, std::vector<int64_t> sym, int64_t O, int64_t K, torch::Tensor x, torch::Tensor bias, torch::Tensor y) {
    uint32_t s[4];
    words(sym, 4, s, "the 12-bit layout's four words (its base)");
    const c10::cuda::CUDAGuard guard(data.device());
    int64_t M = x.size(0);
    TORCH_CHECK(O % 64 == 0 && K % 16 == 0 && M <= 64 && x.is_contiguous() && x.size(1) == K, "O a multiple of 64, K of 16, up to 64 tokens, X contiguous [M, K]");
    size_t bytes = 0;
    ok(glyd_gpu_mma12_gemm_workspace(O, K, M, &bytes), "mma12_gemm");
    auto ws = scratch(bytes, x);
    static auto* done_of = new Counters;
    ok(glyd_gpu_mma12_gemm(ptr<uint8_t>(data), ptr<uint32_t>(exc), ptr<int32_t>(exc_base), s, O, K, ptr<uint16_t>(x), M, opt(bias), ptr<uint16_t>(y), addr(ws), bytes, counters(*done_of, data, O / 64, 1 << 16), current_stream()), "mma12_gemm");
}

void mma_gemm_x1(torch::Tensor data, torch::Tensor blocks, torch::Tensor block_base, std::vector<int64_t> tiers, int64_t O, int64_t K, torch::Tensor x, torch::Tensor bias, torch::Tensor y, int64_t variant) {
    uint32_t t[3];
    words(tiers, 3, t, "three tiers");
    const c10::cuda::CUDAGuard guard(data.device());
    TORCH_CHECK(O % 64 == 0 && K % 64 == 0 && x.is_contiguous() && x.size(1) == K && (uintptr_t)x.data_ptr() % 16 == 0, "O a multiple of 64, K of 64, X contiguous [M, K]");
    int64_t M = x.size(0);
    size_t bytes = 0;
    ok(glyd_gpu_mma_gemm_x1_workspace(O, K, M, variant, &bytes), "mma_gemm_x1");
    auto ws = scratch(bytes, x);
    static auto* done_of = new Counters;
    ok(glyd_gpu_mma_gemm_x1(ptr<uint8_t>(data), ptr<uint8_t>(blocks), ptr<int32_t>(block_base), t, O, K, ptr<uint16_t>(x), M, opt(bias), ptr<uint16_t>(y), variant, addr(ws), bytes, counters(*done_of, data, (M + 127) / 128 * (O / 64), 1 << 18), current_stream()), "mma_gemm_x1");
}

void mma12_gemm_x1(torch::Tensor data, torch::Tensor exc, torch::Tensor exc_base, std::vector<int64_t> sym, int64_t O, int64_t K, torch::Tensor x, torch::Tensor bias, torch::Tensor y, int64_t variant) {
    uint32_t s[4];
    words(sym, 4, s, "the 12-bit layout's four words (its base)");
    const c10::cuda::CUDAGuard guard(data.device());
    TORCH_CHECK(O % 64 == 0 && K % 64 == 0 && x.is_contiguous() && x.size(1) == K && (uintptr_t)x.data_ptr() % 16 == 0, "O a multiple of 64, K of 64, X contiguous [M, K]");
    int64_t M = x.size(0);
    size_t bytes = 0;
    ok(glyd_gpu_mma12_gemm_x1_workspace(O, K, M, variant, &bytes), "mma12_gemm_x1");
    auto ws = scratch(bytes, x);
    static auto* done_of = new Counters;
    ok(glyd_gpu_mma12_gemm_x1(ptr<uint8_t>(data), ptr<uint32_t>(exc), ptr<int32_t>(exc_base), s, O, K, ptr<uint16_t>(x), M, opt(bias), ptr<uint16_t>(y), variant, addr(ws), bytes, counters(*done_of, data, (M + 127) / 128 * (O / 64), 1 << 18), current_stream()), "mma12_gemm_x1");
}

// Many tokens on Ampere and Ada (to 64 a launch, in chunks past that).
void mma12_gemm_x2(torch::Tensor data, torch::Tensor exc, torch::Tensor exc_base, std::vector<int64_t> sym, int64_t O, int64_t K, torch::Tensor x, torch::Tensor bias, torch::Tensor y) {
    uint32_t s[4];
    words(sym, 4, s, "the 12-bit layout's four words (its base)");
    const c10::cuda::CUDAGuard guard(data.device());
    TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major >= 8, "mma12_gemm_x2: Ampere or later");
    TORCH_CHECK(O % 64 == 0 && K % 64 == 0 && x.is_contiguous() && x.size(1) == K && (uintptr_t)x.data_ptr() % 16 == 0, "O a multiple of 64, K of 64, X contiguous [M, K]");
    TORCH_CHECK((uintptr_t)data.data_ptr() % 16 == 0 && (uintptr_t)exc.data_ptr() % 16 == 0 && exc.numel() % 4 == 0, "the pack 16-byte aligned, exc padded to 4 (pack_mma12)");
    int64_t M = x.size(0);
    size_t bytes = 0;
    ok(glyd_gpu_mma12_gemm_x2_workspace(O, K, M, &bytes), "mma12_gemm_x2");
    auto ws = scratch(bytes, data);
    static auto* done_of = new Counters;
    ok(glyd_gpu_mma12_gemm_x2(ptr<uint8_t>(data), ptr<uint32_t>(exc), ptr<int32_t>(exc_base), s, O, K, ptr<uint16_t>(x), M, opt(bias), ptr<uint16_t>(y), addr(ws), bytes, counters(*done_of, data, O / 64, 1 << 16), current_stream()), "mma12_gemm_x2");
}

void mma12_gemm_x3(torch::Tensor data, torch::Tensor exc, torch::Tensor exc_base, std::vector<int64_t> sym, int64_t O, int64_t K, torch::Tensor x, torch::Tensor bias, torch::Tensor y) {
    uint32_t s[4];
    words(sym, 4, s, "the 12-bit layout's four words (its base)");
    const c10::cuda::CUDAGuard guard(data.device());
    TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major == 9 && at::cuda::getCurrentDeviceProperties()->minor == 0, "wgmma: Hopper (compute capability 9.0) alone");
    TORCH_CHECK(O % 64 == 0 && K % 64 == 0 && x.is_contiguous() && x.size(1) == K && (uintptr_t)x.data_ptr() % 16 == 0, "O a multiple of 64, K of 64, X contiguous [M, K]");
    TORCH_CHECK((uintptr_t)data.data_ptr() % 16 == 0 && (uintptr_t)exc.data_ptr() % 16 == 0 && exc.numel() % 4 == 0, "the pack 16-byte aligned, exc padded to 4 (pack_mma12)");
    int64_t M = x.size(0);
    size_t bytes = 0;
    ok(glyd_gpu_mma12_gemm_x3_workspace(O, K, M, &bytes), "mma12_gemm_x3");
    auto ws = scratch(bytes, data);
    static auto* done_of = new Counters;
    ok(glyd_gpu_mma12_gemm_x3(ptr<uint8_t>(data), ptr<uint32_t>(exc), ptr<int32_t>(exc_base), s, O, K, ptr<uint16_t>(x), M, opt(bias), ptr<uint16_t>(y), addr(ws), bytes, counters(*done_of, data, O / 64, 1 << 16), current_stream()), "mma12_gemm_x3");
}

// The routes: (route, last) for M tokens on gpu; the current device's gpu.
std::tuple<int64_t, int64_t> mma_way(int64_t gpu, int64_t O, int64_t K, int64_t M) {
    int r;
    int64_t last;
    ok(glyd_gpu_mma_way(gpu, O, K, M, &r, &last), "mma_way");
    return {r, last};
}

std::tuple<int64_t, int64_t> mma12_way(int64_t gpu, int64_t O, int64_t K, int64_t M) {
    int r;
    int64_t last;
    ok(glyd_gpu_mma12_way(gpu, O, K, M, &r, &last), "mma12_way");
    return {r, last};
}

int64_t mma12_part_cus(int64_t gpu, int64_t O, int64_t K, int64_t M) {
    int64_t sms;
    ok(glyd_gpu_mma12_part_cus(gpu, O, K, M, &sms), "mma12_part_cus");
    return sms;
}

int64_t gpu() {
    int g;
    ok(glyd_gpu_gpu(&g), "gpu");
    return g;
}

// A product by a route (negative: this GPU's for M).
void mma_linear(torch::Tensor data, torch::Tensor blocks, torch::Tensor block_base, std::vector<int64_t> tiers, int64_t O, int64_t K, torch::Tensor x, torch::Tensor bias, torch::Tensor y, int64_t route) {
    uint32_t t[3];
    words(tiers, 3, t, "three tiers");
    const c10::cuda::CUDAGuard guard(data.device());
    TORCH_CHECK(O % 64 == 0 && K % 16 == 0 && x.is_contiguous() && x.size(1) == K, "O a multiple of 64, K of 16, X contiguous [M, K]");
    int64_t M = x.size(0);
    size_t bytes = 0;
    ok(glyd_gpu_mma_linear_workspace(O, K, M, route, &bytes), "mma_linear");
    auto ws = scratch(bytes, x);
    static auto* done_of = new Counters;
    ok(glyd_gpu_mma_linear(ptr<uint8_t>(data), ptr<uint8_t>(blocks), ptr<int32_t>(block_base), t, O, K, ptr<uint16_t>(x), M, opt(bias), ptr<uint16_t>(y), route, addr(ws), bytes, counters(*done_of, data, (M + 127) / 128 * (O / 64), 1 << 18), current_stream()), "mma_linear");
}

void mma12_linear(torch::Tensor data, torch::Tensor exc, torch::Tensor exc_base, std::vector<int64_t> sym, int64_t O, int64_t K, torch::Tensor x, torch::Tensor bias, torch::Tensor y, int64_t route) {
    uint32_t s[4];
    words(sym, 4, s, "the 12-bit layout's four words (its base)");
    const c10::cuda::CUDAGuard guard(data.device());
    TORCH_CHECK(O % 64 == 0 && K % 16 == 0 && x.is_contiguous() && x.size(1) == K, "O a multiple of 64, K of 16, X contiguous [M, K]");
    int64_t M = x.size(0);
    size_t bytes = 0;
    ok(glyd_gpu_mma12_linear_workspace(O, K, M, route, &bytes), "mma12_linear");
    auto ws = scratch(bytes, x);
    static auto* done_of = new Counters;
    ok(glyd_gpu_mma12_linear(ptr<uint8_t>(data), ptr<uint32_t>(exc), ptr<int32_t>(exc_base), s, O, K, ptr<uint16_t>(x), M, opt(bias), ptr<uint16_t>(y), route, addr(ws), bytes, counters(*done_of, data, (M + 127) / 128 * (O / 64), 1 << 18), current_stream()), "mma12_linear");
}

void mma_unpack(torch::Tensor data, torch::Tensor blocks, torch::Tensor block_base, std::vector<int64_t> tiers, int64_t K, int64_t row0, int64_t rows, torch::Tensor out, int64_t warps) {
    uint32_t t[3];
    words(tiers, 3, t, "three tiers");
    const c10::cuda::CUDAGuard guard(data.device());
    TORCH_CHECK(row0 % 64 == 0 && rows % 64 == 0 && out.numel() >= rows * K, "rows a multiple of 64");
    ok(glyd_gpu_mma_unpack(ptr<uint8_t>(data), ptr<uint8_t>(blocks), ptr<int32_t>(block_base), t, K, row0, rows, ptr<uint16_t>(out), warps, current_stream()), "mma_unpack");
}

void hold(int64_t ns) { ok(glyd_gpu_hold(ns, current_stream()), "hold"); }

void mma12_unpack(torch::Tensor data, torch::Tensor exc, torch::Tensor exc_base, std::vector<int64_t> sym, int64_t K, int64_t row0, int64_t rows, torch::Tensor out, int64_t warps) {
    uint32_t s[4];
    words(sym, 4, s, "the 12-bit layout's four words (its base)");
    const c10::cuda::CUDAGuard guard(data.device());
    TORCH_CHECK(row0 % 64 == 0 && rows % 64 == 0 && out.numel() >= rows * K, "rows a multiple of 64");
    ok(glyd_gpu_mma12_unpack(ptr<uint8_t>(data), ptr<uint32_t>(exc), ptr<int32_t>(exc_base), s, K, row0, rows, ptr<uint16_t>(out), warps, current_stream()), "mma12_unpack");
}

void mma_rows(torch::Tensor data, torch::Tensor blocks, torch::Tensor block_base, std::vector<int64_t> tiers, int64_t K, int64_t rows, torch::Tensor ids, torch::Tensor out) {
    uint32_t t[3];
    words(tiers, 3, t, "three tiers");
    const c10::cuda::CUDAGuard guard(data.device());
    TORCH_CHECK(ids.scalar_type() == torch::kInt64 && ids.is_contiguous() && ids.device() == data.device() && out.is_contiguous() && out.device() == data.device() && out.numel() >= ids.numel() * K, "ids int64, out [n, K], on the pack's GPU");
    ok(glyd_gpu_mma_rows(ptr<uint8_t>(data), ptr<uint8_t>(blocks), ptr<int32_t>(block_base), t, K, rows, ptr<int64_t>(ids), ids.numel(), ptr<uint16_t>(out), current_stream()), "mma_rows");
}

void mma12_rows(torch::Tensor data, torch::Tensor exc, torch::Tensor exc_base, std::vector<int64_t> sym, int64_t K, int64_t rows, torch::Tensor ids, torch::Tensor out) {
    uint32_t s[4];
    words(sym, 4, s, "the 12-bit layout's four words (its base)");
    const c10::cuda::CUDAGuard guard(data.device());
    TORCH_CHECK(ids.scalar_type() == torch::kInt64 && ids.is_contiguous() && ids.device() == data.device() && out.is_contiguous() && out.device() == data.device() && out.numel() >= ids.numel() * K, "ids int64, out [n, K], on the pack's GPU");
    ok(glyd_gpu_mma12_rows(ptr<uint8_t>(data), ptr<uint32_t>(exc), ptr<int32_t>(exc_base), s, K, rows, ptr<int64_t>(ids), ids.numel(), ptr<uint16_t>(out), current_stream()), "mma12_rows");
}

void mma12_unpack_part(torch::Tensor data, torch::Tensor exc, torch::Tensor exc_base, std::vector<int64_t> sym, int64_t K, int64_t row0, int64_t rows, torch::Tensor out, int64_t sms) {
    uint32_t s[4];
    words(sym, 4, s, "the 12-bit layout's four words (its base)");
    const c10::cuda::CUDAGuard guard(data.device());
    TORCH_CHECK(row0 % 64 == 0 && rows % 64 == 0 && K % 64 == 0 && out.numel() >= rows * K, "rows a multiple of 64, K of 64");
    ok(glyd_gpu_mma12_unpack_part(ptr<uint8_t>(data), ptr<uint32_t>(exc), ptr<int32_t>(exc_base), s, K, row0, rows, ptr<uint16_t>(out), sms, current_stream()), "mma12_unpack_part");
}

void mma_moe_unpack(torch::Tensor data, torch::Tensor blocks, torch::Tensor block_base, std::vector<int64_t> tiers, int64_t E, int64_t O, int64_t K, int64_t P, torch::Tensor plan, torch::Tensor out) {
    uint32_t t[3];
    words(tiers, 3, t, "three tiers");
    const c10::cuda::CUDAGuard guard(data.device());
    TORCH_CHECK(plan.scalar_type() == torch::kInt32 && out.numel() >= E * O * K && plan.device() == data.device() && out.device() == data.device(), "plan int32, out [E O, K], on the pack's GPU");
    ok(glyd_gpu_mma_moe_unpack(ptr<uint8_t>(data), ptr<uint8_t>(blocks), ptr<int32_t>(block_base), t, E, O, K, P, ptr<int32_t>(plan), ptr<uint16_t>(out), current_stream()), "mma_moe_unpack");
}

void mma12_moe_unpack(torch::Tensor data, torch::Tensor exc, torch::Tensor exc_base, std::vector<int64_t> sym, int64_t E, int64_t O, int64_t K, int64_t P, torch::Tensor plan, torch::Tensor out) {
    uint32_t s[4];
    words(sym, 4, s, "the 12-bit layout's four words (its base)");
    const c10::cuda::CUDAGuard guard(data.device());
    TORCH_CHECK(plan.scalar_type() == torch::kInt32 && out.numel() >= E * O * K && plan.device() == data.device() && out.device() == data.device(), "plan int32, out [E O, K], on the pack's GPU");
    ok(glyd_gpu_mma12_moe_unpack(ptr<uint8_t>(data), ptr<uint32_t>(exc), ptr<int32_t>(exc_base), s, E, O, K, P, ptr<int32_t>(plan), ptr<uint16_t>(out), current_stream()), "mma12_moe_unpack");
}

void moe_way(torch::Tensor ids, int64_t E, torch::Tensor plan) {
    const c10::cuda::CUDAGuard guard(ids.device());
    TORCH_CHECK(ids.scalar_type() == torch::kInt64 && ids.is_contiguous() && plan.scalar_type() == torch::kInt32 && plan.numel() >= 2 + 2 * E + ids.numel() && plan.device() == ids.device(), "ids int64, plan int32 [2 + 2E + P] on the same GPU");
    ok(glyd_gpu_moe_way(ptr<int64_t>(ids), ids.numel(), E, ptr<int32_t>(plan), current_stream()), "moe_way");
}

// The experts' product (w, ids: empty for none).
template <class F>
static void moe_product(const char* name, F run, torch::Tensor data, int64_t E, int64_t O, int64_t K, torch::Tensor x, int64_t k, int64_t gather, torch::Tensor plan, int64_t act, torch::Tensor bias, torch::Tensor w, torch::Tensor ids, torch::Tensor y) {
    const c10::cuda::CUDAGuard guard(data.device());
    TORCH_CHECK(x.is_contiguous() && x.size(1) == K && plan.scalar_type() == torch::kInt32, "X contiguous [., K], plan int32");
    TORCH_CHECK(!w.numel() || ((w.scalar_type() == torch::kFloat32 || w.scalar_type() == torch::kBFloat16) && ids.scalar_type() == torch::kInt64), "weights bf16 or fp32, ids int64");
    TORCH_CHECK(!bias.numel() || (bias.scalar_type() == torch::kBFloat16 && bias.is_contiguous() && bias.numel() >= E * O), "bias bf16, contiguous [E, O]");
    bool here = x.device() == data.device() && plan.device() == data.device() && y.device() == data.device() && (!bias.numel() || bias.device() == data.device());
    TORCH_CHECK(here && (!w.numel() || (w.device() == data.device() && ids.device() == data.device())), "every tensor on the pack's GPU");
    int64_t T = gather ? x.size(0) : x.size(0) / k;
    size_t bytes = 0;
    ok(glyd_gpu_mma_moe_workspace(E, O, K, T, k, act, w.numel() > 0, &bytes), name);
    auto ws = scratch(bytes, x);
    static auto* done_of = new Counters;
    int* done = counters(*done_of, data, (act ? O / 128 : O / 64) * std::min(E, T * k), 1 << 16);
    run(T, w.numel() ? w.data_ptr() : nullptr, w.scalar_type() == torch::kFloat32, addr(ws), bytes, done);
}

void mma_moe(torch::Tensor data, torch::Tensor blocks, torch::Tensor block_base, std::vector<int64_t> tiers, int64_t E, int64_t O, int64_t K, torch::Tensor x, int64_t k, int64_t gather, torch::Tensor plan, int64_t act, torch::Tensor bias, torch::Tensor w, torch::Tensor ids, torch::Tensor y) {
    uint32_t t[3];
    words(tiers, 3, t, "three tiers");
    moe_product("mma_moe", [&](int64_t T, const void* wp, bool wf32, void* ws, size_t bytes, int* done) {
        ok(glyd_gpu_mma_moe(ptr<uint8_t>(data), ptr<uint8_t>(blocks), ptr<int32_t>(block_base), t, E, O, K, ptr<uint16_t>(x), T, k, gather, ptr<int32_t>(plan), act, opt(bias), wp, wf32, ids.numel() ? ptr<int64_t>(ids) : nullptr, ptr<uint16_t>(y), ws, bytes, done, current_stream()), "mma_moe");
    }, data, E, O, K, x, k, gather, plan, act, bias, w, ids, y);
}

void mma12_moe(torch::Tensor data, torch::Tensor exc, torch::Tensor exc_base, std::vector<int64_t> sym, int64_t E, int64_t O, int64_t K, torch::Tensor x, int64_t k, int64_t gather, torch::Tensor plan, int64_t act, torch::Tensor bias, torch::Tensor w, torch::Tensor ids, torch::Tensor y) {
    uint32_t s[4];
    words(sym, 4, s, "the 12-bit layout's four words (its base)");
    moe_product("mma12_moe", [&](int64_t T, const void* wp, bool wf32, void* ws, size_t bytes, int* done) {
        ok(glyd_gpu_mma12_moe(ptr<uint8_t>(data), ptr<uint32_t>(exc), ptr<int32_t>(exc_base), s, E, O, K, ptr<uint16_t>(x), T, k, gather, ptr<int32_t>(plan), act, opt(bias), wp, wf32, ids.numel() ? ptr<int64_t>(ids) : nullptr, ptr<uint16_t>(y), ws, bytes, done, current_stream()), "mma12_moe");
    }, data, E, O, K, x, k, gather, plan, act, bias, w, ids, y);
}

// (tests, any GPU) the wgmma route's pieces on their own: the tiles of a plan [1 + records, 4] int32, the pairs' rows of X in the order's
// [P, K] (those the plan routes; the rest zero), and a unit's epilogue from a given matrix of sums.
torch::Tensor moe_wg_tiles(torch::Tensor plan, int64_t E, int64_t NT, int64_t P) {
    const c10::cuda::CUDAGuard guard(plan.device());
    TORCH_CHECK(plan.scalar_type() == torch::kInt32 && (NT == 16 || NT == 32 || NT == 64 || NT == 128), "plan int32, NT 16, 32, 64 or 128");
    auto out = torch::zeros({(int64_t)(wg_records(E, P) / sizeof(int4)), 4}, plan.options());
    moe_wg_tiles_kernel<<<1, 1024, 0, current_stream()>>>(ptr<int>(plan), (int)E, (int)NT, (int4*)out.data_ptr());
    ok(cudaGetLastError(), "moe_wg_tiles");
    return out;
}

torch::Tensor moe_wg_gather(torch::Tensor x, int64_t k, torch::Tensor plan, int64_t E) {
    const c10::cuda::CUDAGuard guard(x.device());
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && x.is_contiguous() && x.size(1) % 8 == 0 && plan.scalar_type() == torch::kInt32, "x bf16 contiguous [T, K], K a multiple of 8, plan int32");
    auto out = torch::zeros({x.size(0) * k, x.size(1)}, x.options());
    moe_wg_gather_kernel<<<(unsigned)std::min<int64_t>(x.size(0) * k, 4096), 128, 0, current_stream()>>>((const uint4*)x.data_ptr(), ptr<int>(plan), (int)E, k, x.size(1) / 8, (uint4*)out.data_ptr());
    ok(cudaGetLastError(), "moe_wg_gather");
    return out;
}

torch::Tensor moe_wg_walk_test(torch::Tensor tiles, torch::Tensor exc_base, int64_t RB, int64_t KS, int64_t RU, int64_t S, int64_t nb, int64_t ns, int64_t act, int64_t max_units) {
    const c10::cuda::CUDAGuard guard(tiles.device());
    TORCH_CHECK(tiles.scalar_type() == torch::kInt32 && exc_base.scalar_type() == torch::kInt32 && (act == 0 || act == 1) && KS == 4 * S, "tiles and exc_base int32, KS = 4 S");
    auto result = torch::zeros({8 + 8 * nb * max_units}, tiles.options());
    auto go = [&](auto act_c, auto ns_c) {
        constexpr int A = decltype(act_c)::value, N = decltype(ns_c)::value;
        moe_wg_walk_test_kernel<A, N><<<(unsigned)nb, 288, 16 * N + 128 * N, current_stream()>>>((const int4*)tiles.data_ptr(), ptr<int>(exc_base), RB, KS, (int)RU, (int)S, ptr<int>(result), (int)max_units);
    };
    auto by_ns = [&](auto act_c) {
        TORCH_CHECK(ns >= 2, "a ring of 2 slots or more (a slot is released a stage late)");
        if (ns == 2) go(act_c, std::integral_constant<int, 2>());
        else if (ns == 3) go(act_c, std::integral_constant<int, 3>());
        else if (ns == 5) go(act_c, std::integral_constant<int, 5>());
        else go(act_c, std::integral_constant<int, 8>());
    };
    if (act) by_ns(std::integral_constant<int, 1>());
    else by_ns(std::integral_constant<int, 0>());
    ok(cudaGetLastError(), "moe_wg_walk_test");
    return result;
}

void moe_wg_epilogue_test(torch::Tensor mat, std::vector<int64_t> info, int64_t NT, int64_t act, int64_t O, int64_t E, torch::Tensor plan, torch::Tensor bias, torch::Tensor w, torch::Tensor Y, torch::Tensor Y32) {
    const c10::cuda::CUDAGuard guard(mat.device());
    TORCH_CHECK(info.size() == 4 && mat.scalar_type() == torch::kFloat32 && mat.numel() == 2 * 64 * NT && (NT == 16 || NT == 32 || NT == 64 || NT == 128) && (act == 0 || act == 1), "mat float [2, 64, NT]; info the tile's first row, its pairs, its expert, the row unit");
    const int4 inf = make_int4((int)info[0], (int)info[1], (int)info[2], (int)info[3]);
    const int wf32 = w.numel() && w.scalar_type() == torch::kFloat32;
    const void* wp = w.numel() ? w.data_ptr() : nullptr;
    float* y32 = Y32.numel() ? ptr<float>(Y32) : nullptr;
    auto go = [&](auto nt_c, auto act_c) {
        constexpr int N = decltype(nt_c)::value, A = decltype(act_c)::value;
        const int shared = Wm12<N, A>::XCH;
        moe_wg_epilogue_test_kernel<N, A><<<1, 256, shared, current_stream()>>>(ptr<float>(mat), inf, O, (int)E, ptr<int>(plan), (const __nv_bfloat16*)opt(bias), wp, wf32, (__nv_bfloat16*)ptr<uint16_t>(Y), y32);
    };
    auto by_act = [&](auto nt_c) { if (act) go(nt_c, std::integral_constant<int, 1>()); else go(nt_c, std::integral_constant<int, 0>()); };
    if (NT == 16) by_act(std::integral_constant<int, 16>());
    else if (NT == 32) by_act(std::integral_constant<int, 32>());
    else if (NT == 64) by_act(std::integral_constant<int, 64>());
    else by_act(std::integral_constant<int, 128>());
    ok(cudaGetLastError(), "moe_wg_epilogue_test");
}

void attn_decode(torch::Tensor q, torch::Tensor kd, torch::Tensor kb, torch::Tensor kbb, std::vector<int64_t> kt, torch::Tensor vd, torch::Tensor vb, torch::Tensor vbb, std::vector<int64_t> vt, torch::Tensor tk, torch::Tensor tv, int64_t tlen, int64_t pairs, int64_t G, int64_t P, double scale, torch::Tensor out) {
    uint32_t k3[3], v3[3];
    words(kt, 3, k3, "three tiers");
    words(vt, 3, v3, "three tiers");
    const c10::cuda::CUDAGuard guard(q.device());
    int64_t D = q.size(-1);
    TORCH_CHECK((D == 64 || D == 128) && G >= 1 && G <= 16 && q.is_contiguous() && tlen < 64 && P + (tlen > 0) > 0, "head_dim 64 or 128, up to 16 queries a KV head");
    size_t bytes = 0;
    ok(glyd_gpu_attn_decode_workspace(D, tlen, pairs, P, &bytes), "attn_decode");
    auto ws = scratch(bytes, q);
    static auto* done_of = new Counters;  // a pair's finished blocks: zero between calls
    ok(glyd_gpu_attn_decode(ptr<uint16_t>(q), D, ptr<uint8_t>(kd), ptr<uint8_t>(kb), ptr<int32_t>(kbb), k3, ptr<uint8_t>(vd), ptr<uint8_t>(vb), ptr<int32_t>(vbb), v3, ptr<uint16_t>(tk), ptr<uint16_t>(tv), tlen, pairs, G, P, scale, ptr<uint16_t>(out), addr(ws), bytes, counters(*done_of, q, pairs, 1 << 12), current_stream()), "attn_decode");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mma_gemm", &mma_gemm);
    m.def("mma_unpack", &mma_unpack);
    m.def("hold", &hold);
    m.def("mma12_gemm", &mma12_gemm);
    m.def("mma12_gemm_x1", &mma12_gemm_x1);
    m.def("mma12_unpack", &mma12_unpack);
    m.def("mma12_unpack_part", &mma12_unpack_part);
    m.def("mma_rows", &mma_rows);
    m.def("mma12_rows", &mma12_rows);
    m.def("mma12_part_cus", &mma12_part_cus);
    m.def("mma12_gemm_x3", &mma12_gemm_x3);
    m.def("mma12_gemm_x2", &mma12_gemm_x2);
    m.def("attn_decode", &attn_decode);
    m.def("moe_way", &moe_way);
    m.def("mma_moe", &mma_moe);
    m.def("mma12_moe", &mma12_moe);
    m.def("mma_moe_unpack", &mma_moe_unpack);
    m.def("mma12_moe_unpack", &mma12_moe_unpack);
    m.def("moe_wg_tiles", &moe_wg_tiles);
    m.def("moe_wg_gather", &moe_wg_gather);
    m.def("moe_wg_epilogue_test", &moe_wg_epilogue_test);
    m.def("moe_wg_walk_test", &moe_wg_walk_test);
    m.def("mma_gemm_x1", &mma_gemm_x1);
    m.def("mma_way", &mma_way);
    m.def("mma12_way", &mma12_way);
    m.def("gpu", &gpu);
    m.def("mma_linear", &mma_linear);
    m.def("mma12_linear", &mma12_linear);
    m.def("fast_bgemv", &fast_bgemv);
    m.def("fast_gemm", &fast_gemm);
    m.def("pass_a", &pass_a);
    m.def("pass_b", &pass_b);
    m.def("decode", &decode);
    m.def("gemv", &gemv);
    m.def("fast_gemv", &fast_gemv);
    m.def("fast_decode", &fast_decode);
}
#endif
