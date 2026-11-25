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

__device__ __forceinline__ uint32_t exponent_of(uint16_t v) { return (v >> 7) & 0xff; }
__device__ __forceinline__ uint32_t bf16_bits(uint32_t s, uint32_t e) { return ((s & 0x80) << 8) | (e << 7) | (s & 0x7f); }

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

#include "glyd_gpu_kernels.h"  // pf_l2 and the mma layouts (the engine compiles the same: glyd-engine/kernels/fold.cu)

// moe_route_body and moe_route_block_body (glyd_gpu_kernels.h: the plan of a mixture of experts' pairs by expert) as the library launches them.
__global__ void moe_route_kernel(const int64_t* __restrict__ ids, int64_t P, int E, int* __restrict__ plan) {
    moe_route_body(ids, P, E, plan);
}

__global__ void __launch_bounds__(1024) moe_route_block_kernel(const int64_t* __restrict__ ids, int64_t P, int E, int* __restrict__ plan) {
    moe_route_block_body(ids, P, E, plan);
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

// mma_unpack_body (glyd_gpu_kernels.h) as the library launches it.
template <class Fmt, bool MOE = false, bool FEW = false>
__global__ void mma_unpack_kernel(Fmt f, int64_t K, int64_t row0, int64_t rows, uint16_t* __restrict__ out, const int* __restrict__ plan = nullptr) {
    mma_unpack_body<Fmt, MOE, FEW>(f, K, row0, rows, out, plan);
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
// wO, wE: the whole layer's rows an expert and experts where this call holds a part of them (a GPU's slice of each expert's rows, an expert cache's places;
// 0: the call's own). K is split over blocks by the layer's shape alone (an expert's whole rows and the layer's experts), never by the pairs the call holds:
// a pair's sum is the same whatever shares its call (a row alone or beside others, a GPU's slice, a wave of the cache's places), bit for bit.
static int mma_moe_run(Fmt f, int64_t E, int64_t O, int64_t K, const uint16_t* x, int64_t T, int64_t k, int64_t gather, const int32_t* plan, int64_t act, const uint16_t* bias, const void* w, int64_t wf32, const int64_t* ids, uint16_t* y, void* ws, size_t ws_bytes, int* done, cudaStream_t cs, size_t* need, int64_t wO = 0, int64_t wE = 0) {
    if (E < 1 || E > 12288 || O < 64 || O % 64 || K < 16 || K % 16 || T < 0 || k < 1 || act < 0 || act > 2 || (act && (O % 128 || w))) return cudaErrorInvalidValue;
    if (wO < 0 || (wO && (wO < O || wO % 64)) || wE < 0 || (wE && wE < E)) return cudaErrorInvalidValue;
    const int64_t sO = wO ? wO : O, sE = wE ? wE : E;
    int64_t P = T * k, MT = T <= 16 ? 1 : T <= 32 ? 2 : 4, units = act ? O / 128 : O / 64, hits = std::min(E, P);
    int64_t splits = P ? moe_splits((act ? sO / 128 : sO / 64) * sE, K) : 1;
    size_t bytes = ((w ? 1 : 0) + (splits > 1 ? splits : 0)) * (size_t)(P * O) * sizeof(float);  // [Y32][parts]
    bool wg = false;  // the wgmma route (Hopper, the 12-bit layout, GLYD_GPU_MOE_WG tokens a step and more)
    if constexpr (std::is_same_v<Fmt, Nib>) {
        // (never by default: its sums are added in another order than the step products', so a pair's bits would turn on how many tokens share its call)
        const int64_t from = TUNING_INT("GLYD_GPU_MOE_WG", 0);
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
    // Every call takes the step products (mma_moe_kernel), whatever its tokens: the kernel follows from nothing that changes with the pairs a call holds, so a
    // pair's sum is the same for a row alone and beside others, a GPU's slice and the whole layer, an expert cache's wave and the resident layer. The tiled
    // GEMM (mma_moe_big) and the rows kernel (mma_moe_rows_kernel) add a row's terms in another order; a development build's GLYD_GPU_MOE_BIG (_GU, _DN: N
    // pairs an expert) and GLYD_GPU_MOE_ROWS (from N tokens) still take them, as measured before for prompts (an A10 from 48 pairs an expert on OLMoE-1B-7B;
    // an H100 from 96 and an A100 from 64 on Qwen3-30B-A3B; the rows kernel from 17 tokens on an H100 and an A100, 33 on a GDDR GPU), until they add in
    // the step products' order.
    const int64_t big_set = TUNING_INT("GLYD_GPU_MOE_BIG", -1);  // (negative: not set, never)
    const int64_t big = act ? TUNING_INT("GLYD_GPU_MOE_BIG_GU", big_set) : TUNING_INT("GLYD_GPU_MOE_BIG_DN", big_set);
    const int64_t rows_from = so.rows >= 2 ? so.rows : INT64_MAX;
    bool rows = act == 0 && MT >= 2 && T >= rows_from;
    auto go_rows = [&](auto mt) {
        constexpr int M = decltype(mt)::value;
        with_variant([&](auto vv) { mma_moe_rows_launch<Fmt, M, decltype(vv)::value>(f, E, O, K, x, k, gather, plan, bias, w, wf32, y, y32, hits, cs); });
    };
    const bool big_call = big >= 0 && T * k >= big * sE && K % 64 == 0 && (uintptr_t)x % 16 == 0;
    // (GLYD_GPU_MOE_TRACE=1: each call's kernel and shape on stderr, the checks' record of which kernel a pair's sum took)
    if (TUNING_INT("GLYD_GPU_MOE_TRACE", 0))
        fprintf(stderr, "moe-kernel lib %s T=%lld k=%lld E=%lld O=%lld K=%lld act=%lld splits=%lld\n", big_call ? "big" : rows ? "rows" : "step", (long long)T, (long long)k, (long long)E,
                (long long)O, (long long)K, (long long)act, (long long)splits);
    if (big_call) mma_moe_big(f, E, O, K, x, T, k, gather, plan, act, bias, w, wf32, y, y32, cs);
    else if (rows && MT == 1) go_rows(std::integral_constant<int, 1>());
    else if (rows && MT == 2) go_rows(std::integral_constant<int, 2>());
    else if (rows) go_rows(std::integral_constant<int, 4>());
    else if (MT == 1) run(std::integral_constant<int, 1>());
    else if (MT == 2) run(std::integral_constant<int, 2>());
    else run(std::integral_constant<int, 4>());
    if (w) moe_sum_kernel<<<(unsigned)((T * O / 4 + 255) / 256), 256, 0, cs>>>(y32, ids, (int)E, T, k, O, bf(y));
    return cudaGetLastError();
}

GLYD_GPU_INTERNAL int glyd_gpu_mma_moe_as_workspace(int64_t E, int64_t O, int64_t K, int64_t T, int64_t k, int64_t act, int64_t weighted, int64_t wO, int64_t wE, size_t* bytes) {
    int64_t one = 1;  // a stand-in for the weights: only whether there are any counts
    return bytes ? mma_moe_run(Tiered{}, E, O, K, nullptr, T, k, 0, nullptr, act, nullptr, weighted ? &one : nullptr, 0, nullptr, nullptr, nullptr, 0, nullptr, 0, bytes, wO, wE) : cudaErrorInvalidValue;
}

GLYD_GPU_INTERNAL int glyd_gpu_mma_moe_workspace(int64_t E, int64_t O, int64_t K, int64_t T, int64_t k, int64_t act, int64_t weighted, size_t* bytes) {
    return glyd_gpu_mma_moe_as_workspace(E, O, K, T, k, act, weighted, 0, 0, bytes);
}

GLYD_GPU_INTERNAL int glyd_gpu_mma_moe_as(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base, const uint32_t tiers[3], int64_t E, int64_t O, int64_t K, const uint16_t* x, int64_t T, int64_t k, int64_t gather, const int32_t* plan, int64_t act, const uint16_t* bias, const void* w, int64_t wf32, const int64_t* ids, uint16_t* y, void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs, int64_t wO, int64_t wE) {
    Tiers ts;
    bool bad = false;  // (Int6, Int5 and Int4 too: E experts' packs end to end are one pack of [E O, K]; K a multiple of 32; the workspace is the same)
    uint32_t base4h = 0;
    if (int6_words(tiers, K, data, blocks, bad)) return bad ? cudaErrorInvalidValue : mma_moe_run(Int6{data, blocks}, E, O, K, x, T, k, gather, plan, act, bias, w, wf32, ids, y, workspace, workspace_bytes, done, cs, nullptr, wO, wE);
    if (int5_words(tiers, K, data, blocks, bad)) return bad ? cudaErrorInvalidValue : mma_moe_run(Int5{data, blocks}, E, O, K, x, T, k, gather, plan, act, bias, w, wf32, ids, y, workspace, workspace_bytes, done, cs, nullptr, wO, wE);
    if (int4_words(tiers, K, data, blocks, bad)) return bad ? cudaErrorInvalidValue : mma_moe_run(Int4{data, blocks}, E, O, K, x, T, k, gather, plan, act, bias, w, wf32, ids, y, workspace, workspace_bytes, done, cs, nullptr, wO, wE);
    if (int4h_words(tiers, K, data, blocks, bad, base4h)) return bad ? cudaErrorInvalidValue : mma_moe_run(Int4H{{data, blocks}, base4h}, E, O, K, x, T, k, gather, plan, act, bias, w, wf32, ids, y, workspace, workspace_bytes, done, cs, nullptr, wO, wE);
    if (fp4_words(tiers, K, data, blocks, bad)) return bad ? cudaErrorInvalidValue : mma_moe_run(Fp4{data, blocks}, E, O, K, x, T, k, gather, plan, act, bias, w, wf32, ids, y, workspace, workspace_bytes, done, cs, nullptr, wO, wE);
    if (fp4r_words(tiers, K, data, blocks, bad)) return bad ? cudaErrorInvalidValue : mma_moe_run(fp4r_fmt(data, blocks, K), E, O, K, x, T, k, gather, plan, act, bias, w, wf32, ids, y, workspace, workspace_bytes, done, cs, nullptr, wO, wE);
    if (!tiered_words(tiers, ts)) return cudaErrorInvalidValue;
    return mma_moe_run(Tiered{data, blocks, block_base, ts}, E, O, K, x, T, k, gather, plan, act, bias, w, wf32, ids, y, workspace, workspace_bytes, done, cs, nullptr, wO, wE);
}

GLYD_GPU_INTERNAL int glyd_gpu_mma_moe(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base, const uint32_t tiers[3], int64_t E, int64_t O, int64_t K, const uint16_t* x, int64_t T, int64_t k, int64_t gather, const int32_t* plan, int64_t act, const uint16_t* bias, const void* w, int64_t wf32, const int64_t* ids, uint16_t* y, void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs) {
    return glyd_gpu_mma_moe_as(data, blocks, block_base, tiers, E, O, K, x, T, k, gather, plan, act, bias, w, wf32, ids, y, workspace, workspace_bytes, done, cs, 0, 0);
}

GLYD_GPU_INTERNAL int glyd_gpu_mma_moe_placed_as(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base, const uint32_t tiers[3], int64_t stride, int64_t E, int64_t O, int64_t K, const uint16_t* x, int64_t T, int64_t k, int64_t gather, const int32_t* plan, int64_t act, const uint16_t* bias, const void* w, int64_t wf32, const int64_t* ids, uint16_t* y, void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs, int64_t wO, int64_t wE) {
    Tiers ts;
    if (!tiered_words(tiers, ts) || stride < 0 || stride % 4 || O < 64 || O % 64 || K < 16 || K % 16) return cudaErrorInvalidValue;
    return mma_moe_run(TieredPlaced{data, blocks, block_base, ts, O * K / 1024, stride}, E, O, K, x, T, k, gather, plan, act, bias, w, wf32, ids, y, workspace, workspace_bytes, done, cs, nullptr, wO, wE);
}

GLYD_GPU_INTERNAL int glyd_gpu_mma_moe_placed(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base, const uint32_t tiers[3], int64_t stride, int64_t E, int64_t O, int64_t K, const uint16_t* x, int64_t T, int64_t k, int64_t gather, const int32_t* plan, int64_t act, const uint16_t* bias, const void* w, int64_t wf32, const int64_t* ids, uint16_t* y, void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs) {
    return glyd_gpu_mma_moe_placed_as(data, blocks, block_base, tiers, stride, E, O, K, x, T, k, gather, plan, act, bias, w, wf32, ids, y, workspace, workspace_bytes, done, cs, 0, 0);
}

GLYD_GPU_INTERNAL int glyd_gpu_mma12_moe_as_workspace(int64_t E, int64_t O, int64_t K, int64_t T, int64_t k, int64_t act, int64_t weighted, int64_t wO, int64_t wE, size_t* bytes) {
    int64_t one = 1;  // a stand-in for the weights: only whether there are any counts
    return bytes ? mma_moe_run(Nib{}, E, O, K, nullptr, T, k, 0, nullptr, act, nullptr, weighted ? &one : nullptr, 0, nullptr, nullptr, nullptr, 0, nullptr, 0, bytes, wO, wE) : cudaErrorInvalidValue;
}

GLYD_GPU_INTERNAL int glyd_gpu_mma12_moe_workspace(int64_t E, int64_t O, int64_t K, int64_t T, int64_t k, int64_t act, int64_t weighted, size_t* bytes) {
    return glyd_gpu_mma12_moe_as_workspace(E, O, K, T, k, act, weighted, 0, 0, bytes);
}

GLYD_GPU_INTERNAL int glyd_gpu_mma12_moe_as(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4], int64_t E, int64_t O, int64_t K, const uint16_t* x, int64_t T, int64_t k, int64_t gather, const int32_t* plan, int64_t act, const uint16_t* bias, const void* w, int64_t wf32, const int64_t* ids, uint16_t* y, void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs, int64_t wO, int64_t wE) {
    Nib f;
    if (!nib12(data, exc, exc_base, sym, f)) return cudaErrorInvalidValue;
    return mma_moe_run(f, E, O, K, x, T, k, gather, plan, act, bias, w, wf32, ids, y, workspace, workspace_bytes, done, cs, nullptr, wO, wE);
}

GLYD_GPU_INTERNAL int glyd_gpu_mma12_moe(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4], int64_t E, int64_t O, int64_t K, const uint16_t* x, int64_t T, int64_t k, int64_t gather, const int32_t* plan, int64_t act, const uint16_t* bias, const void* w, int64_t wf32, const int64_t* ids, uint16_t* y, void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs) {
    return glyd_gpu_mma12_moe_as(data, exc, exc_base, sym, E, O, K, x, T, k, gather, plan, act, bias, w, wf32, ids, y, workspace, workspace_bytes, done, cs, 0, 0);
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
// The partials' maxima and sums (two floats each) after the counters, rounded up to 16 bytes: the partial outputs after them are read as float4 (merge_head), so an odd
// count of partials would leave them 8 bytes off.
static constexpr int64_t kv_ml_floats(int64_t partials) { return (partials * 2 + 3) / 4 * 4; }
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

GLYD_GPU_INTERNAL int glyd_gpu_kv_image_bytes(int64_t H, int64_t D, int64_t A, int64_t* bytes) {
    int64_t pb;
    if (!bytes || glyd_gpu_kv_leaf_bytes(H, D, A, &pb)) return cudaErrorInvalidValue;
    *bytes = kvc::image_bytes(pb, kv_heads(H, D));
    return 0;
}

GLYD_GPU_INTERNAL int glyd_gpu_kv_save(const uint8_t* pool, int64_t page_stride, int64_t nblocks, const int32_t* pages, int64_t n, int64_t H, int64_t D, int64_t A, const uint8_t* escape, int64_t nchunks,
                                  const uint8_t* raw, int64_t nslabs, int64_t layer, uint8_t* out, uint32_t* moved, cudaStream_t cs) {
    int64_t pb;
    if (glyd_gpu_kv_leaf_bytes(H, D, A, &pb) || n < 0 || n > INT32_MAX || nblocks < 0 || page_stride < pb || page_stride % 16 || (uintptr_t)pool % 16) return cudaErrorInvalidValue;
    if (!kv_escape_ok(escape, nchunks) || !kv_raw_ok(raw, nslabs) || layer < 0 || layer > 254) return cudaErrorInvalidValue;
    if (!n) return 0;
    if (!pages || !out || (uintptr_t)out % 16 || (moved && (uintptr_t)moved % 4)) return cudaErrorInvalidValue;
    const int HP = kv_heads(H, D);
    kvc::save_kernel<<<(unsigned)n, 128, 0, cs>>>(pool, page_stride, nblocks, pages, pb, kvc::image_bytes(pb, HP), HP, kvc::esc_of((uint8_t*)escape, nchunks), kvc::raw_of((uint8_t*)raw, nslabs, HP),
                                                (uint32_t)(layer + 1) << 24, out, moved);
    return cudaGetLastError();
}

GLYD_GPU_INTERNAL int glyd_gpu_kv_load(const uint8_t* img, const int32_t* dst, int64_t n, uint8_t* pool, int64_t page_stride, int64_t nblocks, int64_t H, int64_t D, int64_t A, uint8_t* escape, int64_t nchunks,
                                  uint8_t* raw, int64_t nslabs, int64_t layer, uint32_t* status, int64_t ring, cudaStream_t cs) {
    int64_t pb;
    if (glyd_gpu_kv_leaf_bytes(H, D, A, &pb) || n < 0 || n > INT32_MAX || nblocks < 0 || !status || page_stride < pb || page_stride % 16 || (uintptr_t)pool % 16) return cudaErrorInvalidValue;
    if (!kv_escape_ok(escape, nchunks) || !kv_raw_ok(raw, nslabs) || layer < 0 || layer > 254 || ring < 1 || ring > (1 << 24)) return cudaErrorInvalidValue;
    if (!n) return 0;
    if (!img || !dst || (uintptr_t)img % 16) return cudaErrorInvalidValue;
    const int HP = kv_heads(H, D);
    kvc::load_kernel<<<(unsigned)n, 128, 0, cs>>>(img, dst, pb, kvc::image_bytes(pb, HP), pool, page_stride, nblocks, HP, kvc::esc_of(escape, nchunks), kvc::raw_of(raw, nslabs, HP), (uint32_t)(layer + 1) << 24,
                                                status, (uint32_t)ring);
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
    *bytes = KV_CNT_BYTES + (size_t)(kv_ml_floats(B * H * splits * G) + B * H * splits * G * D) * sizeof(float);
    return 0;
}

// The decode kernels of a call (glyd_gpu_kv_decode_from's, glyd_gpu_kv_decode_bf16's): the attention, and the merge where it is not fused. A bf16 cache takes the kernel's
// instance of its own (no pipeline: it reads a page's words at once).
static int kv_attn_launch(const kvc::AttnParams& P, int64_t B, int64_t H, int64_t D, int64_t splits, const int32_t* seq_lens, const int32_t* first, uint16_t* out, int64_t out_stride, cudaStream_t cs) {
    const bool pipe = kv_pipe(), fuse = kv_fuse();
    const dim3 grid((unsigned)H, (unsigned)B, (unsigned)splits);  // (attn_kernel's: the splits last)
    const auto launch = [&](auto halves) {
        constexpr int HV = decltype(halves)::value;
        if (P.kc) {
            if (fuse) kvc::attn_kernel<false, true, HV, true><<<grid, kvc::NW * 32, 0, cs>>>(P);
            else kvc::attn_kernel<false, false, HV, true><<<grid, kvc::NW * 32, 0, cs>>>(P);
        } else if (fuse) {
            if (pipe) kvc::attn_kernel<true, true, HV><<<grid, kvc::NW * 32, 0, cs>>>(P);
            else kvc::attn_kernel<false, true, HV><<<grid, kvc::NW * 32, 0, cs>>>(P);
        } else {
            if (pipe) kvc::attn_kernel<true, false, HV><<<grid, kvc::NW * 32, 0, cs>>>(P);
            else kvc::attn_kernel<false, false, HV><<<grid, kvc::NW * 32, 0, cs>>>(P);
        }
        if (!fuse) kvc::combine_kernel<HV * kvc::D><<<(unsigned)(B * H), 256, 0, cs>>>(P.ws_ml, P.ws_o, bf(out), out_stride, (int)H, P.G, (int)splits, P.ppc_min, P.pfloor, seq_lens, first);
    };
    if (D == kvc::D) launch(std::integral_constant<int, 1>{});
    else if (D == 2 * kvc::D) launch(std::integral_constant<int, 2>{});
    else launch(std::integral_constant<int, 4>{});
    return cudaGetLastError();
}

GLYD_GPU_INTERNAL int glyd_gpu_kv_decode_from(const uint16_t* q, int64_t q_stride, const uint8_t* pool, int64_t page_stride, const uint8_t* kbase, const uint8_t* vbase, const int32_t* block_table,
                                         int64_t bt_stride, const int32_t* seq_lens, const int32_t* first, int64_t round_scores, int64_t B, int64_t splits, int64_t H, int64_t G, int64_t D, int64_t A,
                                         const uint8_t* escape, int64_t nchunks, const uint8_t* raw, int64_t nslabs, double scale, uint16_t* out, int64_t out_stride, void* workspace,
                                         size_t workspace_bytes, cudaStream_t cs) {
    int64_t page;
    size_t need;
    if (kv_one(D) || glyd_gpu_kv_leaf_bytes(H, D, A, &page) || B < 0 || B > 65535 || page_stride < page || page_stride % 16 || (uintptr_t)pool % 16 || !kv_escape_ok(escape, nchunks) || !kv_raw_ok(raw, nslabs)) return cudaErrorInvalidValue;
    if (!B) return 0;
    if (glyd_gpu_kv_decode_workspace(B, H, G, D, splits, &need) || !fits(workspace, workspace_bytes, need)) return cudaErrorInvalidValue;
    if (!q || !kbase || !vbase || !block_table || !seq_lens || !out || q_stride < H * G * D || out_stride < H * G * D || q_stride % 2 || ((uintptr_t)q) % 4) return cudaErrorInvalidValue;
    uint32_t* ws_cnt = (uint32_t*)workspace;  // (the workspace's first KV_CNT_BYTES: the counters, zero when a call starts and when it ends)
    float* ws_ml = (float*)((uint8_t*)workspace + KV_CNT_BYTES);
    const kvc::EscapePool E = kvc::esc_of((uint8_t*)escape, nchunks);
    const int HP = kv_heads(H, D);
    kvc::AttnParams P{bf(q), q_stride, pool, page_stride, kbase, vbase, block_table, bt_stride, seq_lens, E.ent, E.nc, kvc::raw_of((uint8_t*)raw, nslabs, HP), ws_ml, ws_ml + kv_ml_floats(B * H * splits * G), (int)H, (int)G, (int)A, kvc::hdr_bytes(HP, (int)A), (int)splits,
                      HP, (float)(scale * 1.4426950408889634), kv_ppc_min(), kv_floor(B, H, current_device()), ws_cnt, bf(out), out_stride, first, (int)(round_scores != 0), nullptr, nullptr, 0};
    return kv_attn_launch(P, B, H, D, splits, seq_lens, first, out, out_stride, cs);
}

// glyd_gpu_kv_decode_from's attention over a bf16 cache (keys kc, values vc [places][H][D]; token p of sequence b at block_table[b bt_stride + p / 16] * 16 + p % 16, or p
// modulo cap where block_table is null): the same kernel, plan and merge, so the same bits where the cache holds what the pages hold.
GLYD_GPU_INTERNAL int glyd_gpu_kv_decode_bf16(const uint16_t* q, int64_t q_stride, const uint16_t* kc, const uint16_t* vc, const int32_t* block_table, int64_t bt_stride, int64_t cap,
                                      const int32_t* seq_lens, const int32_t* first, int64_t round_scores, int64_t B, int64_t splits, int64_t H, int64_t G, int64_t D, double scale,
                                      uint16_t* out, int64_t out_stride, void* workspace, size_t workspace_bytes, cudaStream_t cs) {
    size_t need;
    if (D % kvc::D || D < kvc::D || D > 4 * kvc::D || D == 3 * kvc::D || B < 0 || B > 65535) return cudaErrorInvalidValue;
    if (!B) return 0;
    if (glyd_gpu_kv_decode_workspace(B, H, G, D, splits, &need) || !fits(workspace, workspace_bytes, need)) return cudaErrorInvalidValue;
    if (!q || !kc || !vc || !seq_lens || !out || (!block_table && (cap < 1 || cap > INT32_MAX)) || q_stride < H * G * D || out_stride < H * G * D || q_stride % 2 || (uintptr_t)q % 4 ||
        (uintptr_t)kc % 4 || (uintptr_t)vc % 4)
        return cudaErrorInvalidValue;
    uint32_t* ws_cnt = (uint32_t*)workspace;
    float* ws_ml = (float*)((uint8_t*)workspace + KV_CNT_BYTES);
    kvc::AttnParams P{bf(q), q_stride, nullptr, 0, nullptr, nullptr, block_table, bt_stride, seq_lens, nullptr, 0, kvc::RawPool{}, ws_ml, ws_ml + kv_ml_floats(B * H * splits * G), (int)H, (int)G, 0, 0,
                      (int)splits, (int)(H * (D / kvc::D)), (float)(scale * 1.4426950408889634), kv_ppc_min(), kv_floor(B, H, current_device()), ws_cnt, bf(out), out_stride, first,
                      (int)(round_scores != 0), bf(kc), bf(vc), (int)std::min<int64_t>(cap, INT32_MAX)};
    return kv_attn_launch(P, B, H, D, splits, seq_lens, first, out, out_stride, cs);
}

GLYD_GPU_INTERNAL int glyd_gpu_kv_decode(const uint16_t* q, int64_t q_stride, const uint8_t* pool, int64_t page_stride, const uint8_t* kbase, const uint8_t* vbase, const int32_t* block_table, int64_t bt_stride,
                                    const int32_t* seq_lens, int64_t B, int64_t splits, int64_t H, int64_t G, int64_t D, int64_t A, const uint8_t* escape, int64_t nchunks, const uint8_t* raw, int64_t nslabs,
                                    double scale, uint16_t* out, int64_t out_stride, void* workspace, size_t workspace_bytes, cudaStream_t cs) {
    return glyd_gpu_kv_decode_from(q, q_stride, pool, page_stride, kbase, vbase, block_table, bt_stride, seq_lens, nullptr, 0, B, splits, H, G, D, A, escape, nchunks, raw, nslabs, scale, out,
                                   out_stride, workspace, workspace_bytes, cs);
}

GLYD_GPU_INTERNAL int glyd_gpu_kv_prefill(const uint16_t* q, int64_t q_stride, const uint16_t* key, int64_t key_stride, const uint16_t* value, int64_t value_stride, int64_t n, int64_t start,
                                     int64_t window, const uint8_t* pool, int64_t page_stride, const uint8_t* kbase, const uint8_t* vbase, const int32_t* block_table, int64_t H, int64_t G,
                                     int64_t D, int64_t A, const uint8_t* escape, int64_t nchunks, const uint8_t* raw, int64_t nslabs, double scale, int64_t round_scores, double softcap,
                                     const float* sinks, uint16_t* out, int64_t out_stride, cudaStream_t cs) {
    int64_t page;
    if (kv_one(D) || glyd_gpu_kv_leaf_bytes(H, D, A, &page) || page_stride < page || page_stride % 16 || (uintptr_t)pool % 16 || !kv_escape_ok(escape, nchunks) || !kv_raw_ok(raw, nslabs)) return cudaErrorInvalidValue;
    if (n < 0 || start < 0 || window < 0 || start + n > INT32_MAX || G < 1 || H * G > 65535 || !(softcap >= 0.0) || !std::isfinite(scale) || !std::isfinite(softcap)) return cudaErrorInvalidValue;
    if (!n) return 0;
    const int64_t W = H * G * D;  // a row's query heads (and its output's)
    if (!q || !out || !kbase || !vbase || (start > 0 && !block_table) || (!key && !block_table) || q_stride < W || out_stride < W || q_stride % 2 || out_stride % 2 || (uintptr_t)q % 4 || (uintptr_t)out % 4) return cudaErrorInvalidValue;
    if (key && (!value || key_stride < H * D || value_stride < H * D || key_stride % 8 || value_stride % 8 || (uintptr_t)key % 16 || (uintptr_t)value % 16)) return cudaErrorInvalidValue;
    const kvc::EscapePool E = kvc::esc_of((uint8_t*)escape, nchunks);
    const int HP = kv_heads(H, D), halves = (int)(D / kvc::D);
    kvc::PrefillParams P{bf(q), q_stride, bf(key), key_stride, bf(value), value_stride, pool, page_stride, kbase, vbase, block_table, E.ent, E.nc, kvc::raw_of((uint8_t*)raw, nslabs, HP), sinks, bf(out),
                         out_stride, nullptr, nullptr, 0, (int)n, (int)start, (int)std::min<int64_t>(window, INT32_MAX), (int)G, (int)A, kvc::hdr_bytes(HP, (int)A), HP, (int)(round_scores != 0),
                         (float)(scale * 1.4426950408889634), softcap > 0.0 ? (float)(scale / softcap) : 0.f, (float)(softcap * 1.4426950408889634)};
    // (the CTA's row groups of 16, measured on an RTX 4080 SUPER, gemma-4-12B's shapes, a chunk of 256 rows after 1,024 to 8,192 tokens: heads of 256, 4 (one CTA an SM,
    // 64 of them): 0.186 to 0.189 ms, where 2 (two an SM, 128) took 0.224 to 0.229; heads of 512, 2: 4 would hold more registers than the SM has)
    const auto launch = [&](auto kernel, int hv, int groups) {
        const int shared = kvc::pf_shared(hv, groups);
        if (shared > 48 * 1024 && cudaFuncSetAttribute((const void*)kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared) != cudaSuccess) {
            cudaGetLastError();  // (the GPU has not the shared memory a CTA takes: an RDNA card's 64 KB for heads of 512)
            return (int)cudaErrorNotSupported;
        }
        kernel<<<dim3((unsigned)((n + 16 * groups - 1) / (16 * groups)), (unsigned)(H * G)), 32 * groups * hv, shared, cs>>>(P);
        return (int)cudaGetLastError();
    };
    if (halves == 1) return launch(kvc::prefill_kernel<1, 4>, 1, 4);
    if (halves == 2) return launch(kvc::prefill_kernel<2, 4>, 2, 4);
    return launch(kvc::prefill_kernel<4, 2>, 4, 2);
}

// glyd_gpu_kv_prefill's attention over a bf16 cache (keys kc, values vc [places][H][D]; a position's place block_table[p / 16] * 16 + p % 16, or p modulo cap
// where block_table is null), the chunk's own keys and values written there before the call: the same kernel and tiles, so the same bits where the cache holds
// what the pages hold.
GLYD_GPU_INTERNAL int glyd_gpu_kv_prefill_bf16(const uint16_t* q, int64_t q_stride, const uint16_t* kc, const uint16_t* vc, const int32_t* block_table, int64_t cap, int64_t n, int64_t start,
                                       int64_t window, int64_t H, int64_t G, int64_t D, double scale, int64_t round_scores, double softcap, const float* sinks, uint16_t* out,
                                       int64_t out_stride, cudaStream_t cs) {
    if (D % kvc::D || D < kvc::D || D > 4 * kvc::D || D == 3 * kvc::D || H < 1 || n < 0 || start < 0 || window < 0 || start + n > INT32_MAX || G < 1 || H * G > 65535 || !(softcap >= 0.0) ||
        !std::isfinite(scale) || !std::isfinite(softcap))
        return cudaErrorInvalidValue;
    if (!n) return 0;
    const int64_t W = H * G * D;
    if (!q || !out || !kc || !vc || (!block_table && (cap < 1 || cap > INT32_MAX)) || q_stride < W || out_stride < W || q_stride % 2 || out_stride % 2 || (uintptr_t)q % 4 || (uintptr_t)out % 4 ||
        (uintptr_t)kc % 16 || (uintptr_t)vc % 16)
        return cudaErrorInvalidValue;
    const int halves = (int)(D / kvc::D);
    kvc::PrefillParams P{bf(q), q_stride, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, nullptr, block_table, nullptr, 0, kvc::RawPool{}, sinks, bf(out), out_stride, bf(kc), bf(vc),
                         (int)std::min<int64_t>(cap, INT32_MAX), (int)n, (int)start, (int)std::min<int64_t>(window, INT32_MAX), (int)G, 0, 0, (int)(H * halves), (int)(round_scores != 0),
                         (float)(scale * 1.4426950408889634), softcap > 0.0 ? (float)(scale / softcap) : 0.f, (float)(softcap * 1.4426950408889634)};
    const auto launch = [&](auto kernel, int hv, int groups) {
        const int shared = kvc::pf_shared(hv, groups);
        if (shared > 48 * 1024 && cudaFuncSetAttribute((const void*)kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared) != cudaSuccess) {
            cudaGetLastError();
            return (int)cudaErrorNotSupported;
        }
        kernel<<<dim3((unsigned)((n + 16 * groups - 1) / (16 * groups)), (unsigned)(H * G)), 32 * groups * hv, shared, cs>>>(P);
        return (int)cudaGetLastError();
    };
    if (halves == 1) return launch(kvc::prefill_kernel<1, 4>, 1, 4);
    if (halves == 2) return launch(kvc::prefill_kernel<2, 4>, 2, 4);
    return launch(kvc::prefill_kernel<4, 2>, 4, 2);
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
