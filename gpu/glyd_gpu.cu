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
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <mma.h>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cctype>
#include <cerrno>
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
// (the next 3), then tier 3 (the next 3), then the exponent byte itself.
// A warp step's fixed 1280 bytes: its lanes' tier-1 digits (weight i at bits
// 2(i mod 16) of word i / 16; [2 words][32 lanes]), then their 32 bytes each
// as two halves (every lane's first 16, then every lane's last 16): weight
// i's 7 mantissa bits above the sign of its pair's other weight (i ^ 1), so
// a pair's two bytes and two exponents, permuted into one word and rotated
// by a bit, are its two bf16s. Its block (bytes block_base[step] to
// block_base[step + 1]): from its start, the tier-3 digits of its tier-2
// escapes (2 bits each), then from the next byte the exponent bytes of its
// tier-3 escapes; at its end, the tier-2 digits of its tier-1 escapes in
// lane order, as words of 16 (word c: escapes 16c to 16c + 15, the c-th
// word back from the end; unused digits 0). All in order: lane by lane,
// a lane's weights in order.

constexpr int64_t STEP_BYTES = 1280;  // a warp step's fixed part
constexpr int S2_BYTES = 1040;        // a warp's shared scratch for a step's tier-2 exponents
constexpr uint32_t FULL = 0xffffffffu;

// The tiers' symbols: tier k's three exponents in bytes 0-2.
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
    st.blk = __ldg(block_base + step);
    st.end = __ldg(block_base + step + 1);
    const uint32_t* bw = (const uint32_t*)blocks;
    st.win = __ldg(bw + (st.blk >> 2) + lane);
    st.tail = __ldg(bw + (st.end >> 2) - 1 - lane);
    st.tail1 = __ldg(bw + (st.end >> 2));
}

// A word of 16 tier-2 digits (d) as 16 bytes into the warp's scratch at
// word 4c: their escapes' from its tier-3 digits (d3), theirs from the bytes
// (x). g3: the warp's most tier-3 digits a word, in fours.
__device__ __forceinline__ void tier23(uint32_t d, uint32_t d3, uint32_t x[4], Tiers ts, uint32_t g3, uint32_t* s2, int c, const uint32_t* tab) {
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
__device__ __forceinline__ void exponents(const Step& st, Tiers ts, const uint8_t* __restrict__ blocks, int lane, uint32_t* s2, const uint32_t* tab, uint32_t ew[8]) {
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
    tier23(da, d3a, x, ts, g3, s2, lane, tab);
    if (T1 > 512) {
        bytes(bb);
        tier23(db, d3b, x, ts, g3, s2, lane + 32, tab);
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

// A step's 32 weights a lane as the B fragments of its 8 n-tiles (R[2n], R[2n + 1]).
__device__ __forceinline__ void decode_step(const Step& st, Tiers ts, const uint8_t* __restrict__ blocks, int lane, uint32_t* s2, const uint32_t* tab, uint32_t R[16]) {
    uint32_t ew[8];
    exponents(st, ts, blocks, lane, s2, tab, ew);
    pairs(st.sw, ew, R);
}

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
    // A step's decode where its block's steps have many exceptions (mma_gemm_kernel's heavy blocks): past 4 in the
    // step, through the warp's scratch (s2: 1 KB, a lane's 8 words, zero between steps) the lanes take the run's
    // entries 32 at a time, each setting its byte there, which every lane then XORs into its words and clears, as
    // decode12_rows does; else as decode.
    __device__ __forceinline__ void decode_x(const St& st, int lane, uint32_t* s2, uint32_t R[16]) const {
        if (st.e1 - st.e0 <= 4) return decode(st, lane, nullptr, nullptr, R);
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
    template <int U = 0>  // patch's unrolling
    __device__ __forceinline__ void decode(const St& st, int lane, uint32_t*, const uint32_t*, uint32_t R[16]) const {
        uint32_t H[8];
        high(st.nb, H);
        const uint32_t* e = exc;
        patch<U>([e](int k) { return __ldg(e + k); }, st.e0, st.e1, lane, H);
        pairs(st.sw, H, R);
    }
};

__device__ __forceinline__ void mma16816(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
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
// loads, W's and then the inputs, are issued before this step's decode. mma_gemm_kernel's and mma_moe_kernel's.
template <class Fmt, int MT, bool X = false, int U = 0>  // X: decode_x (the 12-bit layout's heavy blocks); U: Nib::patch's unrolling
__device__ __forceinline__ void mma_steps(const Fmt& f, int64_t base, int64_t s0, int64_t s1, const __nv_bfloat16* const (&xr)[MT][2], int lane, uint32_t* s2, const uint32_t* tab, float (&acc)[MT][8][4]) {
    typename Fmt::St st;
    uint32_t a[MT][4];
    auto load = [&](int64_t s) {
        f.load(st, base + s, lane);
#pragma unroll
        for (int mt = 0; mt < MT; mt++) {
            a[mt][0] = xr[mt][0] ? *(const uint32_t*)(xr[mt][0] + s * 16) : 0u;
            a[mt][2] = xr[mt][0] ? *(const uint32_t*)(xr[mt][0] + s * 16 + 8) : 0u;
            a[mt][1] = xr[mt][1] ? *(const uint32_t*)(xr[mt][1] + s * 16) : 0u;
            a[mt][3] = xr[mt][1] ? *(const uint32_t*)(xr[mt][1] + s * 16 + 8) : 0u;
        }
    };
    if (s0 < s1) load(s0);
    for (int64_t s = s0; s < s1; s++) {
        typename Fmt::St cur = st;
        uint32_t ca[MT][4];
#pragma unroll
        for (int mt = 0; mt < MT; mt++)
#pragma unroll
            for (int q = 0; q < 4; q++) ca[mt][q] = a[mt][q];
        if (s + 1 < s1) load(s + 1);
        uint32_t R[16];
        if constexpr (X) f.decode_x(cur, lane, s2, R);
        else if constexpr (U != 0) f.template decode<U>(cur, lane, s2, tab, R);
        else f.decode(cur, lane, s2, tab, R);
#pragma unroll
        for (int mt = 0; mt < MT; mt++)
#pragma unroll
            for (int nn = 0; nn < 8; nn++) mma16816(acc[mt][nn], ca[mt], R[2 * nn], R[2 * nn + 1]);
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

template <class Fmt, int MT>
__global__ void __launch_bounds__(256, MT == 1 ? 2 : 1) mma_gemm_kernel(Fmt f, int64_t O, int64_t K, int64_t M, const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ Y, float* __restrict__ parts, int* __restrict__ done) {
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
            mma_steps<Fmt, MT, !Fmt::kTable>(f, rb * KS, s0, s1, xr, lane, s2, tab, acc);
            __syncthreads();  // every warp's scratch done with
        } else {
            mma_steps<Fmt, MT>(f, rb * KS, s0, s1, xr, lane, s2, tab, acc);
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

__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }
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
};

template <class Fmt, int CW, int PW, int NB, int RBB, bool MOE = false>
__global__ void __launch_bounds__(Big<CW, PW, NB, RBB>::THREADS, 1) mma_gemm_big_kernel(Fmt f, int64_t O, int64_t K, int64_t M, int64_t stages_per_split, const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ Y, float* __restrict__ Y32, MoePairs moe) {
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
                int64_t xr[MOE ? C::CHUNKS : 1];  // MOE: X's row of each of this thread's chunks (-1: past the pairs)
                if constexpr (MOE)
#pragma unroll
                    for (int i = 0; i < C::CHUNKS; i++) {
                        int64_t m = tile * C::TM + ((pt + 32 * PW * i) >> 3);
                        xr[i] = m < M ? (moe.gather ? __ldg(order + p0 + m) / moe.k : p0 + m) : -1;
                    }
                d_load(0);
                for (int64_t j = 0; j < nst; j++) {
                    int b = (int)(j % NB);
                    if (j >= NB) bar_sync<C::THREADS>(1 + NB + b);  // consumers are done with stage j - NB
                    uint32_t slot = a_base + (uint32_t)b * C::A_BYTES;
                    int64_t col = (st0 + j) * BIG_KK * 16;
#pragma unroll
                    for (int i = 0; i < C::CHUNKS; i++) {
                        int id = pt + 32 * PW * i, r = id >> 3, c = id & 7;
                        if constexpr (MOE) {
                            cp_async16(slot + r * 128 + ((c ^ (r & 7)) << 4), X + (xr[i] < 0 ? 0 : xr[i]) * K + col + c * 8, xr[i] < 0 ? 0 : 16);
                        } else {
                            int64_t m = tile * C::TM + r;
                            cp_async16(slot + r * 128 + ((c ^ (r & 7)) << 4), X + (m < M ? m : 0) * K + col + c * 8, m < M ? 16 : 0);
                        }
                    }
                    asm volatile("cp.async.commit_group;\n" ::);
                    typename Fmt::St cur[C::STEPS];
#pragma unroll
                    for (int i = 0; i < C::STEPS; i++) cur[i] = st[i];
                    if (j + 1 < nst) d_load(j + 1);
#pragma unroll
                    for (int i = 0; i < C::STEPS; i++) {
                        uint32_t R[16];
                        uint4* slot = Bs + b * C::B_UINT4 + ((p * C::STEPS + i) * 4) * 32;
                        f.decode(cur[i], lane, (uint32_t*)slot, tab, R);  // (tiered) the step's slot its scratch till then
                        uint4* d = slot + lane;
#pragma unroll
                        for (int q = 0; q < 4; q++) d[q * 32] = make_uint4(R[4 * q], R[4 * q + 1], R[4 * q + 2], R[4 * q + 3]);
                    }
                    asm volatile("cp.async.wait_group 0;\n" ::);
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
                        for (int mt = 0; mt < 4; mt++) ldmatrix_x4(a[mt], slot + a_row + mt * 16 * 128 + (((2 * kk + a_half) ^ a_sw) << 4));
#pragma unroll
                        for (int q = 0; q < 4; q++) {
                            uint4 b = bsrc[(kk * 4 + q) * 32];
#pragma unroll
                            for (int mt = 0; mt < 4; mt++) {
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
                    for (int mt = 0; mt < 4; mt++) ldmatrix_x4(fa[f][mt], slot + a_row + mt * 16 * 128 + (((2 * kk + a_half) ^ a_sw) << 4));
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
                                    v[c] = (moe.act == 1 ? silu(gate) : gelu_tanh(gate)) * u;
                                }
                                *(__nv_bfloat162*)(Y + (p0 + m) * (O / 2) + o) = __floats2bfloat162_rn(v[0], v[1]);
                            }
                        }
                    break;
                }
            }
            int64_t my_rb = pair * RBB + rbl;
            if (my_rb >= RB) break;
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
                                int64_t j = __ldg(order + p0 + m);
                                float wj = moe.wf32 ? ((const float*)moe.w)[j] : __bfloat162float(((const __nv_bfloat16*)moe.w)[j]);
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
// sums' order is a unit's stages', whatever CR: the same bits for the same TM and RBB.
template <class Fmt, int CW, int PW, int NB, int RBB, int CR = 64>
__global__ void __launch_bounds__(Big<CW, PW, NB, RBB, CR>::THREADS, 1) mma_gemm_sk_kernel(Fmt f, int64_t O, int64_t K, int64_t M, const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ Y, float* __restrict__ parts, int* __restrict__ done) {
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
            asm volatile("cp.async.commit_group;\n" ::);
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
            asm volatile("cp.async.wait_group 0;\n" ::);
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
#pragma unroll
        for (int kk = 0; kk < BIG_KK; kk++) {
            int fi = kk & 1;
            if (kk + 1 < BIG_KK) frags(j, kk + 1, fi ^ 1);
            else if (j + 1 < n) next(j + 1, fi ^ 1);
#pragma unroll
            for (int mt = 0; mt < 4; mt++)
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
            int before;
            asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;\n" : "=r"(before) : "l"(done + p) : "memory");
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
__global__ void __launch_bounds__(Tma12<NT, WG>::THREADS, 1) mma12_tma_kernel(const __grid_constant__ CUtensorMap xmap, Nib f, int64_t O, int64_t K, int64_t M, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ Y, float* __restrict__ parts, int* __restrict__ done) {
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
#elif defined(__CUDA_ARCH__)
    __trap();  // sm_90a code, launched from a build without it
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
            int before;
            asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;\n" : "=r"(before) : "l"(done + q * CL + rank) : "memory");
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
__global__ void __launch_bounds__(Wgp12<NT, CL>::THREADS, 1) mma12_wgp_kernel(const __grid_constant__ CUtensorMap xmap, Nib f, int64_t O, int64_t K, int64_t M, int64_t D, int ncs, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ Y, float* __restrict__ parts, int* __restrict__ done) {
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
#elif defined(__CUDA_ARCH__)
    __trap();  // sm_90a code, launched from a build without it
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
__global__ void __launch_bounds__(Mid12<NT, WG>::THREADS, 1) mma12_mid_kernel(Nib f, int64_t O, int64_t K, int64_t M, const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ Y, float* __restrict__ parts, int* __restrict__ done) {
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
#endif
}

// A 16-byte copy of data read once (W's steps): evicted from L2 first.
__device__ __forceinline__ void cp_async16_once(uint32_t dst, const void* src, uint64_t pol) {
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "l"(pol));
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
__global__ void __launch_bounds__(Ws12<CW, NB, NW, RBB, MT>::THREADS, 1) mma12_ws_kernel(Nib f, int64_t O, int64_t K, int64_t M, const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ Y, float* __restrict__ parts, int* __restrict__ done) {
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
template <class Fmt, int MT, int ACT>
__global__ void __launch_bounds__(256, MT == 1 ? 2 : 1) mma_moe_kernel(Fmt f, int64_t O, int64_t K, const int* __restrict__ plan, int E, int64_t k, int gather, const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ bias, const void* __restrict__ w, int wf32, __nv_bfloat16* __restrict__ Y, float* __restrict__ Y32, int64_t P, float* __restrict__ parts, int* __restrict__ done) {
    extern __shared__ __align__(128) float red[];  // [4 warps][16 MT rows][65], then (tiered) the warps' scratch [8][S2_BYTES]
    __shared__ uint32_t tab[Fmt::kTable ? 256 : 1];
    __shared__ int last;
    if ((int)blockIdx.y >= __ldg(plan)) return;  // the whole block: no expert this far down the hits
    if constexpr (Fmt::kTable) fill_groups(tab);
    __syncthreads();
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    uint32_t* s2 = (uint32_t*)(red + 4 * MT * 16 * 65) + warp * (S2_BYTES / 4);
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
        mma_steps<Fmt, MT, false, U>(f, base, s0, s1, xr, lane, s2, tab, acc);
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

// The mma layout back to bf16, rows [row0, row0 + rows) of W (multiples of
// 64) into out [rows, K] (checks, and the many-token path that multiplies
// with PyTorch): a warp a step; FEW (a decode ahead, a few warps an SM),
// each warp its step and on by the grid's warps. MOE (exact, a mixture of
// experts' layer, W its experts' matrices of `rows` rows stacked):
// blockIdx.y a hit expert of plan (moe_route's; past those hit: nothing to
// do), its rows into the same rows of out [E rows, K], the rest of out left
// as it is.
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
        int64_t step = row0 / 64 * KS + local, rb = local / KS, s = local % KS;
        typename Fmt::St st;
        f.load(st, step, lane);
        uint32_t R[16];
        f.decode(st, lane, (uint32_t*)scratch[threadIdx.x >> 5], tab, R);
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

// Option 2's decode (the route SPLIT: glyd_gpu_mma12_unpack_split, glyd_gpu_mma12_ring_*): the 12-bit layout's rows
// [row0, row0 + rows) into out [rows, K] on the few SMs a green context sets apart for it, the products on the rest. A
// warp takes units of 4 steps of a row block (64 rows x 64 columns), grid-stride, the next unit's steps loaded into
// registers while this one's are decoded; each step's B-fragment words go into the warp's tile in shared memory
// (16-byte chunk c of row r at c ^ (r & 7): its 32-bit stores and 16-byte loads without bank conflicts), and out in
// whole 128-byte lines. mma_unpack_kernel's step a warp with 4-byte stores is bound by its latency on few SMs (2.7
// weights a clock an SM on an RTX 4080 SUPER); this runs 9.6 on an A100 and 10.2-10.4 on an H100 SXM and PCIe on 16
// SMs, its memory path the bound (benchmarks/gpu/research-2026-09-29/round1).
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
#ifdef TORCH_EXTENSION_NAME
#define GLYD_GPU_API extern "C" __attribute__((visibility("hidden")))  // the pybind module's own
#else
#define GLYD_GPU_API extern "C" __attribute__((visibility("default")))
#endif

static int64_t tiles_for(int64_t n, int64_t tw) { return (n + tw - 1) / tw; }

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
GLYD_GPU_API int glyd_gpu_cuda_version() { return CUDART_VERSION; }
// The C API's version (glyd_gpu.h: one more whenever a function's arguments, or what they must hold, change; the
// caller checks it, as ctypes does not check arguments).
GLYD_GPU_API int glyd_gpu_api_version() { return GLYD_GPU_API_VERSION; }
GLYD_GPU_API const char* glyd_gpu_error_string(int status) {
    static const char* blas[] = {"cuBLAS: success", "cuBLAS: not initialized", "cuBLAS status 2", "cuBLAS: allocation failed", "cuBLAS status 4", "cuBLAS status 5", "cuBLAS status 6",
                                 "cuBLAS: invalid value", "cuBLAS: architecture mismatch", "cuBLAS status 9", "cuBLAS status 10", "cuBLAS: mapping error", "cuBLAS status 12",
                                 "cuBLAS: execution failed", "cuBLAS: internal error", "cuBLAS: not supported", "cuBLAS: license error"};
    if (status >= GLYD_GPU_BLAS_ERROR) return status - GLYD_GPU_BLAS_ERROR < 17 ? blas[status - GLYD_GPU_BLAS_ERROR] : "cuBLAS: an error";
    return cudaGetErrorString((cudaError_t)status);
}

// The dense format. lane_bits: bits [tiles * 32], the tiles of tw weights of w's n.
GLYD_GPU_API int glyd_gpu_lane_bits(const uint16_t* w, int64_t n, const uint8_t* len, int64_t tw, int64_t V, uint32_t* bits, cudaStream_t cs) {
    if (tw < 1) return cudaErrorInvalidValue;
    int64_t lanes = tiles_for(n, tw) * 32;
    BY_V(V, (lane_bits_kernel<VV><<<(lanes + 255) / 256, 256, 0, cs>>>(w, n, tw, len, bits, lanes)));
    return cudaGetLastError();
}

GLYD_GPU_API int glyd_gpu_write_codes(const uint16_t* w, int64_t n, const uint8_t* len, const uint32_t* code, const uint32_t* offs, uint32_t* out, int64_t tw, int64_t V, cudaStream_t cs) {
    if (tw < 1) return cudaErrorInvalidValue;
    int64_t lanes = tiles_for(n, tw) * 32;
    BY_V(V, (write_kernel<VV><<<(lanes + 255) / 256, 256, 0, cs>>>(w, n, tw, len, code, offs, out, lanes)));
    return cudaGetLastError();
}

static void allow_shared(const void* kernel) { cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024); }

// Every tile of the n weights into out [n], or (n_ids > 0) tiles tile_ids into out [n_ids * tw].
GLYD_GPU_API int glyd_gpu_decode(const uint8_t* sm, const uint32_t* stream, int64_t stream_words, const uint32_t* offs, const uint32_t* tables, int64_t n, int64_t tw, int64_t V, int64_t tile_words, const int64_t* tile_ids, int64_t n_ids, uint16_t* out, cudaStream_t cs) {
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
GLYD_GPU_API int glyd_gpu_gemv(const uint8_t* sm, const uint32_t* stream, int64_t stream_words, const uint32_t* offs, const uint32_t* tables, int64_t O, int64_t K, int64_t tw, int64_t V, int64_t tile_words, const uint16_t* x, const uint16_t* bias, uint16_t* y, float* sum, int* count, cudaStream_t cs) {
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
GLYD_GPU_API int glyd_gpu_fast_gemv(const uint8_t* sm, const uint32_t* planes, const uint8_t* exc, const int32_t* exc_base, uint64_t top, int64_t O, int64_t K, const uint16_t* x, const uint16_t* bias, uint16_t* y, cudaStream_t cs) {
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
GLYD_GPU_API int glyd_gpu_fast_decode(const uint8_t* sm, const uint32_t* planes, const uint8_t* exc, const int32_t* exc_base, uint64_t top, int64_t row0, int64_t rows, const int64_t* row_ids, int64_t n_ids, int64_t K, uint16_t* out, cudaStream_t cs) {
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
    static int64_t target = getenv("GLYD_GPU_GEMM_BLOCKS") ? atoll(getenv("GLYD_GPU_GEMM_BLOCKS")) : 1280;
    int64_t bo = (O + GM_BO - 1) / GM_BO, bm = (M + GM_BM - 1) / GM_BM;
    int64_t split = std::max<int64_t>(1, std::min<int64_t>(K / (4 * GM_BK), target / (bo * bm)));
    kchunk = (K / GM_BK + split - 1) / split * GM_BK;
    return (K + kchunk - 1) / kchunk;
}

GLYD_GPU_API int glyd_gpu_fast_gemm_workspace(int64_t O, int64_t K, int64_t M, size_t* bytes) {
    if (!bytes || O < 1 || O % 16 || M < 1 || K < 1 || K % GM_BK) return cudaErrorInvalidValue;
    int64_t kchunk, split = fast_gemm_split(O, K, M, kchunk);
    *bytes = split > 1 ? (size_t)(split * M * O) * sizeof(float) : 0;
    return 0;
}

GLYD_GPU_API int glyd_gpu_fast_gemm(const uint8_t* sm, const uint32_t* planes, const uint8_t* exc, const int32_t* exc_base, uint64_t top, int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y, void* workspace, size_t workspace_bytes, cudaStream_t cs) {
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

GLYD_GPU_API int glyd_gpu_fast_bgemv_workspace(int64_t O, int64_t K, int64_t M, size_t* bytes) {
    if (!bytes || O < 1 || K < 1 || K % 512 || !(M == 2 || M == 4 || M == 8 || M == 16)) return cudaErrorInvalidValue;
    int64_t kseg = bgemv_kseg(K, M), nseg = (K + kseg - 1) / kseg;
    *bytes = nseg > 1 ? (size_t)(nseg * M * O) * sizeof(float) : 0;
    return 0;
}

GLYD_GPU_API int glyd_gpu_fast_bgemv(const uint8_t* sm, const uint32_t* planes, const uint8_t* exc, const int32_t* exc_base, uint64_t top, int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y, void* workspace, size_t workspace_bytes, cudaStream_t cs) {
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

// A 12-bit pack's words (sym, host): sym[0] the high bytes' base hb (0-120) in each of its bytes, sym[1-3] zero; false
// otherwise (cudaErrorInvalidValue). A pack of the 12-bit layout before split byte (unreleased) held its 15 commonest
// exponents there, 4 distinct ones in sym[0]: refused, never decoded as this layout.
static bool nib12(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t* sym, Nib& f) {
    if (!sym) return false;
    uint32_t hb = sym[0] & 0xFFu;
    if (sym[0] != hb * 0x01010101u || hb > 120 || sym[1] || sym[2] || sym[3]) return false;
    f = Nib{data, exc, exc_base, sym[0]};
    return true;
}

// The mma layouts (tiered: tiers[3]; 12-bit: sym[4], its base). Y [M, O] = X W^T (+ bias) for up to 64 tokens: as
// many blocks as fit at once, but at least 32 steps (4 a warp) a block; the workspace: their parts,
// [blocks + O / 64][M][64] floats; done: O / 64. need: set to the workspace's bytes, nothing launched.
template <class Fmt, int MT>
static int mma_gemm_mt(Fmt f, int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y, void* ws, size_t ws_bytes, int* done, cudaStream_t cs, size_t* need) {
    auto kernel = mma_gemm_kernel<Fmt, MT>;
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
    kernel<<<nb, 256, shared, cs>>>(f, O, K, M, bf(x), bf(bias), bf(y), (float*)ws, done);
    return cudaGetLastError();
}

template <class Fmt>
static int mma_gemm_run(Fmt f, int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y, void* ws, size_t ws_bytes, int* done, cudaStream_t cs, size_t* need) {
    if (O % 64 || K % 16 || M < 0 || M > 64) return cudaErrorInvalidValue;
    auto run = M <= 16 ? mma_gemm_mt<Fmt, 1> : M <= 32 ? mma_gemm_mt<Fmt, 2> : mma_gemm_mt<Fmt, 4>;
    return run(f, O, K, x, M, bias, y, ws, ws_bytes, done, cs, need);
}

GLYD_GPU_API int glyd_gpu_mma_gemm_workspace(int64_t O, int64_t K, int64_t M, size_t* bytes) {
    return bytes ? mma_gemm_run(Tiered{}, O, K, nullptr, M, nullptr, nullptr, nullptr, 0, nullptr, 0, bytes) : cudaErrorInvalidValue;
}

GLYD_GPU_API int glyd_gpu_mma_gemm(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base, const uint32_t tiers[3], int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y, void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs) {
    return mma_gemm_run(Tiered{data, blocks, block_base, {tiers[0], tiers[1], tiers[2]}}, O, K, x, M, bias, y, workspace, workspace_bytes, done, cs, nullptr);
}

GLYD_GPU_API int glyd_gpu_mma12_gemm_workspace(int64_t O, int64_t K, int64_t M, size_t* bytes) {
    return bytes ? mma_gemm_run(Nib{}, O, K, nullptr, M, nullptr, nullptr, nullptr, 0, nullptr, 0, bytes) : cudaErrorInvalidValue;
}

GLYD_GPU_API int glyd_gpu_mma12_gemm(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4], int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y, void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs) {
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
template <class Fmt, int CW, int PW, int NB, int RBB, int CR = 64>
static int mma_gemm_big_run(Fmt f, int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y, void* ws, size_t ws_bytes, int* done, cudaStream_t cs, size_t* need) {
    using C = Big<CW, PW, NB, RBB, CR>;
    auto kernel = mma_gemm_sk_kernel<Fmt, CW, PW, NB, RBB, CR>;
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

GLYD_GPU_API int glyd_gpu_mma_gemm_big_workspace(int64_t O, int64_t K, int64_t M, int64_t variant, size_t* bytes) {
    return bytes ? mma_gemm_big_any(Tiered{}, O, K, nullptr, M, nullptr, nullptr, variant, nullptr, 0, nullptr, 0, bytes) : cudaErrorInvalidValue;
}

// done: (M + 127) / 128 times O / 64 counters at least, zero (the last block of a unit resets its own; read on
// GeForce Ada, where the product runs by stream-K).
GLYD_GPU_API int glyd_gpu_mma_gemm_big(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base, const uint32_t tiers[3], int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y, int64_t variant, void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs) {
    return mma_gemm_big_any(Tiered{data, blocks, block_base, {tiers[0], tiers[1], tiers[2]}}, O, K, x, M, bias, y, variant, workspace, workspace_bytes, done, cs, nullptr);
}

GLYD_GPU_API int glyd_gpu_mma12_gemm_big_workspace(int64_t O, int64_t K, int64_t M, int64_t variant, size_t* bytes) {
    return bytes ? mma_gemm_big_any(Nib{}, O, K, nullptr, M, nullptr, nullptr, variant, nullptr, 0, nullptr, 0, bytes) : cudaErrorInvalidValue;
}

GLYD_GPU_API int glyd_gpu_mma12_gemm_big(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4], int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y, int64_t variant, void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs) {
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

GLYD_GPU_API int glyd_gpu_mma12_gemm_mid_workspace(int64_t O, int64_t K, int64_t M, size_t* bytes) {
    if (!bytes) return cudaErrorInvalidValue;
    *bytes = 0;
    return mma12_mid_any(Nib{}, O, K, nullptr, M, nullptr, nullptr, nullptr, nullptr, 0, bytes);
}

GLYD_GPU_API int glyd_gpu_mma12_gemm_mid(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4], int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y, void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs) {
    Nib f;
    if (!nib12(data, exc, exc_base, sym, f)) return cudaErrorInvalidValue;
    size_t need = 0;
    if (int r = mma12_mid_any(f, O, K, x, M, bias, y, nullptr, nullptr, cs, &need)) return r;
    if (!fits(workspace, workspace_bytes, need) || !done) return cudaErrorInvalidValue;
    return mma12_mid_any(f, O, K, x, M, bias, y, (float*)workspace, done, cs, nullptr);
}

// cuTensorMapEncodeTiled from the driver, found at run time (no link to libcuda); null where it has none.
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
}

// Prompts on Hopper past 128 tokens: one launch of mma12_wgp_kernel, a block an SM, in clusters of CL (as many as
// the GPU holds at once, at least 8 stages each; or a whole number a tile where the tiles are fewer, below). The
// workspace, where tiles are split: two slots a block that takes part, [2 ncs CL][NT 128] floats; done: a counter a
// split tile a block of its cluster (fewer than the blocks).
template <int NT, int CL>
static int mma12_wgp_run(Nib f, int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y, float* parts, int* done, cudaStream_t cs, size_t* need) {
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

GLYD_GPU_API int glyd_gpu_mma12_gemm_wg_workspace(int64_t O, int64_t K, int64_t M, size_t* bytes) {
    if (!bytes) return cudaErrorInvalidValue;
    *bytes = 0;
    return mma12_wg_any(Nib{}, O, K, nullptr, M, nullptr, nullptr, nullptr, nullptr, 0, bytes);
}

GLYD_GPU_API int glyd_gpu_mma12_gemm_wg(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4], int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y, void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs) {
    Nib f;
    if (!nib12(data, exc, exc_base, sym, f)) return cudaErrorInvalidValue;
    size_t need = 0;
    if (int r = mma12_wg_any(f, O, K, x, M, bias, y, nullptr, nullptr, cs, &need)) return r;
    if (!fits(workspace, workspace_bytes, need) || !done) return cudaErrorInvalidValue;
    return mma12_wg_any(f, O, K, x, M, bias, y, (float*)workspace, done, cs, nullptr);
}

// The routes: how glyd.gpu multiplies by W [O, K] for M tokens on a GPU, as measured there (gpu/README.md), for
// model.py's GLinear and glyd_gpu_*_linear alike: a step's kernel to 64 tokens (in the 12-bit layout mma_gemm_mid
// from 17 on Ampere and Ada, an A100's to 128, and mma_gemm_wg from 17 to 1024 on Hopper, mma12_wgp_kernel past 128),
// past that the prompt kernel (mma_gemm_big), but where W decoded for a bf16 GEMM (cuBLAS) is the faster (DECODE): a
// 12-bit prompt from GLYD_DEC_MIN tokens where it is set (any GPU), else an A100's from 769, Hopper's past its wgmma
// kernel's (none on Hopper takes a prompt), and every prompt of a matrix whose K is not a multiple of 64 (the prompt
// kernel's blocks); and with W decoded ahead, beside the products before it (AHEAD), whatever K: GeForce Ada's from
// 513 tokens tiered and 1793 12-bit, an A10's from 512 and 640 (ahead_min). GLYD_WG_MIN, GLYD_WG_MAX, GLYD_MID_MIN
// and GLYD_DEC_MIN move those thresholds (read at the first call; GLYD_DEC_MIN 0 or unset: an A100's 769 alone). A
// GPU is given as glyd_gpu_gpu gives it: its compute capability, major * 10 + minor, plus its class by name
// (gpu_class).
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
    static const RouteMins t{get("GLYD_WG_MIN", 17), get("GLYD_WG_MAX", 1024), get("GLYD_MID_MIN", 17), get("GLYD_DEC_MIN", 0),
                             get("GLYD_SPLIT_MIN", 0), get("GLYD_SPLIT_MAX", 0), get("GLYD_SPLIT_SMS", 0)};
    return t;
}

// A prompt decoded ahead from this many tokens (the fused, not exact, product's): GeForce Ada's (measured on an RTX
// 4080 SUPER) past 512 tiered and 1792 12-bit, an A10's (150 W, full-rate tensor cores: the fused kernel's decode costs
// it clocks at the power cap) from 512 tiered and 640 12-bit (model.py's GLinear for the measurements); else never.
static int64_t ahead_min(bool twelve, int64_t gpu) {
    return gpu == GLYD_GPU_GEFORCE + 89 ? (twelve ? 1793 : 513) : gpu == GLYD_GPU_A10 + 86 ? (twelve ? 640 : 512) : INT64_MAX;
}

// Option 2 (the route SPLIT): a 12-bit prompt's matrices decoded ahead on SMs set apart (green contexts), cuBLAS on the
// rest (glyd_gpu_mma12_ring_linear), where it beat v0.25.0's routes (benchmarks/gpu/research-2026-09-29/round1: an
// A100-SXM4-40GB, an H100 SXM and an H100 PCIe, Qwen3-8B's and 32B's layers against bf16 cuBLAS): an A100's prompts
// from 769 tokens (today's decoded path there; Qwen3-8B's layer 21% / 8% / 7% / 2% faster at 1024 / 2048 / 4096 / 8192,
// 32B's 8% / 1.5% at 1024 / 2048, even at 4096, 1.4% slower at 8192), a matrix over two ring slots of 100 MiB (32B's,
// 14B's gate and up) there to 4096, until slots its size are measured; Hopper's from 1024 (32B's 2% faster at 1024,
// 13% / 1% / 6% at 2048 / 4096 / 8192; 8B's 8% / 12% / 4% / 5%), an H100 PCIe's at 1024 alone (its longer prompts not
// yet measured against its own routes). Its SMs for the decode: an A100's 12 to 1535 tokens, 8 to 3071, then 4;
// Hopper's 20 to 1535, 12 to 6143, then 4 (the split's granularity there: 8, cuBLAS a co-scheduled group); an H100
// PCIe's 18 (as measured best). GLYD_SPLIT_MIN, GLYD_SPLIT_MAX (0 or unset: the GPU's; GLYD_SPLIT_MIN negative: never)
// and GLYD_SPLIT_SMS move them, on any GPU from Ampere; a GPU's code with GLYD_GPU_NO_SPLIT never takes it (where the
// split cannot run: no green contexts, a capture). The decode is the 12-bit layout's (K a multiple of 64).
constexpr int64_t SPLIT_SLOT_WEIGHTS = 50 << 20;  // a ring slot of 100 MiB of bf16, as measured

static int64_t split_sms(bool twelve, int64_t gpu, int64_t O, int64_t K, int64_t M) {
    const RouteMins& t = route_mins();
    int64_t code = gpu & ~(int64_t)GLYD_GPU_NO_SPLIT, cc = code % 1000;
    bool a100 = cc == 80, hopper = cc == 90, pcie = code - cc == GLYD_GPU_PCIE;
    if (!twelve || gpu & GLYD_GPU_NO_SPLIT || K % 64 || t.split_min < 0 || cc < 80) return 0;
    int64_t lo = t.split_min ? t.split_min : a100 ? 769 : hopper ? 1024 : INT64_MAX;
    int64_t hi = t.split_max ? t.split_max : hopper && pcie ? 1024 : a100 && O * K > 2 * SPLIT_SLOT_WEIGHTS ? 4096 : INT64_MAX;
    if (M < lo || M > hi) return 0;
    if (t.split_sms) return t.split_sms;
    if (a100) return M < 1536 ? 12 : M < 3072 ? 8 : 4;
    if (hopper) return pcie ? 18 : M < 1536 ? 20 : M < 6144 ? 12 : 4;
    return 12;  // (GLYD_SPLIT_MIN set on another GPU: not measured)
}

static int route_for(bool twelve, int64_t gpu, int64_t O, int64_t K, int64_t M) {
    const RouteMins& t = route_mins();
    if (split_sms(twelve, gpu, O, K, M)) return GLYD_GPU_ROUTE_SPLIT;
    gpu &= ~(int64_t)GLYD_GPU_NO_SPLIT;
    int64_t cc = gpu % 1000;
    bool a100 = cc == 80, hopper = cc == 90, mid = cc == 80 || cc == 86 || cc == 87 || cc == 89, k64 = K % 64 == 0;
    if (hopper && twelve && k64 && M >= t.wg_min && M <= t.wg_max) return GLYD_GPU_ROUTE_WG;  // TMA and wgmma
    if (mid && twelve && k64 && M >= t.mid_min && M <= (a100 ? 128 : 64)) return GLYD_GPU_ROUTE_MID;  // cp.async, mma.sync
    if (twelve && M >= (t.dec_min ? t.dec_min : a100 ? 769 : INT64_MAX)) return GLYD_GPU_ROUTE_DECODE;
    if (M <= 64) return GLYD_GPU_ROUTE_GEMM;
    if (M >= ahead_min(twelve, gpu)) return GLYD_GPU_ROUTE_AHEAD;  // (any K)
    return k64 && !hopper ? GLYD_GPU_ROUTE_BIG : GLYD_GPU_ROUTE_DECODE;
}

// A GPU's class by its name, where its compute capability does not tell it apart (glyd_gpu.h): GLYD_GPU_GEFORCE with
// "GeForce" in the name; GLYD_GPU_A10 with "A10" as a word, between characters that are not letters, digits or '_'
// (an A10, not an A10G, A100 or A40: Python's re.search(r"\bA10\b", name, re.ASCII), as model.py's GLinear asks);
// else 0.
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

static int gpu_class(const char* name) { return strstr(name, "GeForce") ? GLYD_GPU_GEFORCE : has_word(name, "A10") ? GLYD_GPU_A10 : has_pcie(name) ? GLYD_GPU_PCIE : 0; }

// The current device as the routes take it (asked once a device).
GLYD_GPU_API int glyd_gpu_gpu(int* gpu) {
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

// The route for M tokens, and (last) the last token count from M on that takes it: before the first threshold past
// M where another starts.
static int route_run(bool twelve, int64_t gpu, int64_t O, int64_t K, int64_t M, int* route, int64_t* last) {
    if (!route || O < 1 || K < 1 || M < 0) return cudaErrorInvalidValue;
    const RouteMins& t = route_mins();
    int here = *route = route_for(twelve, gpu, O, K, M);
    if (last) {
        int64_t cuts[] = {t.wg_min, t.wg_max + 1, t.mid_min, 65, 129, t.dec_min ? t.dec_min : 769, 512, 513, 640, 1793, 1024, 1025, 4097,
                          t.split_min, t.split_max + 1}, next = INT64_MAX;
        for (int64_t c : cuts)
            if (c > M && c < next && route_for(twelve, gpu, O, K, c) != here) next = c;
        *last = next == INT64_MAX ? INT64_MAX : next - 1;
    }
    return 0;
}

GLYD_GPU_API int glyd_gpu_mma_route(int64_t gpu, int64_t O, int64_t K, int64_t M, int* route, int64_t* last) {
    return route_run(false, gpu, O, K, M, route, last);
}

GLYD_GPU_API int glyd_gpu_mma12_route(int64_t gpu, int64_t O, int64_t K, int64_t M, int* route, int64_t* last) {
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
    constexpr bool twelve = std::is_same_v<Fmt, Nib>;
    int gpu = 0;
    if (int r = glyd_gpu_gpu(&gpu)) return r;
    if (route < 0) route = route_for(twelve, gpu, O, K, M);
    switch (route) {
    case GLYD_GPU_ROUTE_GEMM:
        return mma_gemm_run(f, O, K, x, M, bias, y, ws, ws_bytes, done, cs, need);
    case GLYD_GPU_ROUTE_MID:
    case GLYD_GPU_ROUTE_WG:
        if constexpr (twelve) return staged_run(route == GLYD_GPU_ROUTE_MID ? mma12_mid_any : mma12_wg_any, f, O, K, x, M, bias, y, ws, ws_bytes, done, cs, need);
        return cudaErrorInvalidValue;  // the 12-bit layout's alone
    case GLYD_GPU_ROUTE_DECODE:
    case GLYD_GPU_ROUTE_AHEAD:
    case GLYD_GPU_ROUTE_SPLIT:  // (the pipelined product is glyd_gpu_mma12_ring_linear, with the caller's cuBLAS)
        if (K % 64) return cudaErrorNotSupported;
        [[fallthrough]];
    case GLYD_GPU_ROUTE_BIG:
        return mma_gemm_big_any(f, O, K, x, M, bias, y, 0, ws, ws_bytes, done, cs, need);
    }
    return cudaErrorInvalidValue;
}

GLYD_GPU_API int glyd_gpu_mma_linear_workspace(int64_t O, int64_t K, int64_t M, int64_t route, size_t* bytes) {
    return bytes ? mma_linear_run(Tiered{}, O, K, nullptr, M, nullptr, nullptr, route, nullptr, 0, nullptr, 0, bytes) : cudaErrorInvalidValue;
}

GLYD_GPU_API int glyd_gpu_mma_linear(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base, const uint32_t tiers[3], int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y, int64_t route, void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs) {
    return mma_linear_run(Tiered{data, blocks, block_base, {tiers[0], tiers[1], tiers[2]}}, O, K, x, M, bias, y, route, workspace, workspace_bytes, done, cs, nullptr);
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
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do {
        __nanosleep(500);
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    } while (t - t0 < (uint64_t)ns);
}

GLYD_GPU_API int glyd_gpu_hold(int64_t ns, cudaStream_t cs) {
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
    return mma_unpack_run(Tiered{data, blocks, block_base, {tiers[0], tiers[1], tiers[2]}}, K, row0, rows, out, warps, cs);
}

GLYD_GPU_API int glyd_gpu_mma12_unpack(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4], int64_t K, int64_t row0, int64_t rows, uint16_t* out, int64_t warps, cudaStream_t cs) {
    Nib f;
    if (!nib12(data, exc, exc_base, sym, f)) return cudaErrorInvalidValue;
    return mma_unpack_run(f, K, row0, rows, out, warps, cs);
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

GLYD_GPU_API int glyd_gpu_mma12_unpack_split(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4], int64_t K, int64_t row0, int64_t rows, uint16_t* out, int64_t sms, cudaStream_t cs) {
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
// driver has none (before CUDA 12.4; a stream of one from 12.5).
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
                  driver_fn("cuStreamDestroy", 2000, x.stream_destroy);
        return ok ? &x : nullptr;
    }();
    return g;
}

static void part_free(SplitPart& p) {
    const Green* G = green();
    for (cudaStream_t s : {p.sd, p.sg})
        if (s) cudaStreamSynchronize(s), G->stream_destroy((CUstream)s);
    for (void* c : {p.gd, p.gg})
        if (c) G->destroy((CUgreenCtx)c);
    p = SplitPart{};
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
static void part_free(SplitPart& p) { p = SplitPart{}; }
static int part_make(int64_t, SplitPart&) { return cudaErrorNotSupported; }
static int record_on(cudaEvent_t e, cudaStream_t s, void*) { return (int)cudaEventRecord(e, s); }
#endif

constexpr int RING_MAX = 16;

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
    };
    std::deque<Chunk> q;  // queued, not yet multiplied; the first `issued` decoding or decoded
    size_t issued = 0;
    int next = 0;                                   // the slot the next decode takes
    bool busy[RING_MAX] = {}, read[RING_MAX] = {};  // a decode there not yet multiplied; a product has read it
    cudaEvent_t ready[RING_MAX], free_[RING_MAX], mark;
};

// The split for sms: made once and kept.
static int ring_part(glyd_gpu_ring* g, int64_t sms, SplitPart** p) {
    auto it = g->parts.find(sms);
    if (it == g->parts.end()) {
        SplitPart made;
        if (int r = part_make(sms, made)) return r;
        it = g->parts.emplace(sms, made).first;
    }
    *p = &it->second;
    return 0;
}

// Decodes queued into the slots free, in order.
static int ring_pump(glyd_gpu_ring* g) {
    SplitPart* p = g->cur;
    while (g->issued < g->q.size() && !g->busy[g->next]) {
        glyd_gpu_ring::Chunk& c = g->q[g->issued];
        int s = g->next;
        if (g->read[s]) cudaStreamWaitEvent(p->sd, g->free_[s], 0);  // the product that read it done
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
    g->issued = 0;
    std::fill(g->busy, g->busy + RING_MAX, false);  // (their decodes and products in stream order before the next)
    g->cur = p;
    return 0;
}

// W's rows in even chunks of at most a slot, multiples of 64, queued and their decodes started as slots are free.
static int ring_queue(glyd_gpu_ring* g, Nib f, int64_t O, int64_t K) {
    if (O < 64 || O % 64 || K < 64 || K % 64) return cudaErrorInvalidValue;
    int64_t per = std::min<int64_t>(O, (int64_t)(g->slot_bytes / (2 * K)) / 64 * 64);
    if (per < 64) return cudaErrorInvalidValue;  // not 64 rows to a slot
    int64_t n = (O + per - 1) / per, rows = ((O + n - 1) / n + 63) / 64 * 64;
    for (int64_t r0 = 0; r0 < O; r0 += rows) g->q.push_back({f, O, K, r0, std::min(rows, O - r0), -1});
    return ring_pump(g);
}

GLYD_GPU_API int glyd_gpu_ring_create(void* buffer, size_t bytes, size_t slot_bytes, glyd_gpu_ring** ring) {
    if (!ring || !buffer || (uintptr_t)buffer % 16 || !slot_bytes || slot_bytes % 256 || bytes / slot_bytes < 3) return cudaErrorInvalidValue;
    glyd_gpu_ring* g = new glyd_gpu_ring;
    g->dev = current_device(), g->buf = (uint8_t*)buffer, g->slot_bytes = slot_bytes, g->slots = (int)std::min<size_t>(RING_MAX, bytes / slot_bytes);
    cudaError_t r = cudaEventCreateWithFlags(&g->mark, cudaEventDisableTiming);
    for (int s = 0; s < g->slots && !r; s++)
        if (!(r = cudaEventCreateWithFlags(&g->ready[s], cudaEventDisableTiming))) r = cudaEventCreateWithFlags(&g->free_[s], cudaEventDisableTiming);
    if (r) {
        delete g;  // (its events let go with the process: a failed start)
        return r;
    }
    *ring = g;
    return 0;
}

GLYD_GPU_API int glyd_gpu_ring_destroy(glyd_gpu_ring* ring) {
    if (!ring) return cudaErrorInvalidValue;
    for (auto& kv : ring->parts) part_free(kv.second);
    cudaEventDestroy(ring->mark);
    for (int s = 0; s < ring->slots; s++) cudaEventDestroy(ring->ready[s]), cudaEventDestroy(ring->free_[s]);
    delete ring;
    return 0;
}

GLYD_GPU_API int glyd_gpu_ring_split(glyd_gpu_ring* ring, int64_t sms, int64_t* decode_sms, int64_t* product_sms) {
    SplitPart* p;
    if (!ring) return cudaErrorInvalidValue;
    if (int r = ring_part(ring, sms, &p)) return r;
    if (decode_sms) *decode_sms = p->dec;
    if (product_sms) *product_sms = p->gemm;
    return 0;
}

GLYD_GPU_API int glyd_gpu_ring_reset(glyd_gpu_ring* ring, cudaStream_t cs) {
    return ring ? ring_restart(ring, ring->cur, cs) : cudaErrorInvalidValue;
}

GLYD_GPU_API int glyd_gpu_mma12_ring_queue(glyd_gpu_ring* ring, int64_t sms, const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4], int64_t O, int64_t K) {
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

GLYD_GPU_API int glyd_gpu_mma12_ring_linear(glyd_gpu_ring* ring, int64_t sms, const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4], int64_t O, int64_t K, const uint16_t* x, int64_t M, const uint16_t* bias, uint16_t* y, const glyd_gpu_blas* blas, cudaStream_t cs) {
    Nib f;
    SplitPart* p;
    if (!ring || !blas || !blas->handle || !blas->gemm_ex || !blas->set_stream || !nib12(data, exc, exc_base, sym, f)) return cudaErrorInvalidValue;
    if (M < 1 || M > INT32_MAX || O > INT32_MAX || K > INT32_MAX || (uintptr_t)x % 16 || (uintptr_t)y % 16) return cudaErrorInvalidValue;
    cudaStreamCaptureStatus cap;
    if (cudaStreamIsCapturing(cs, &cap) || cap != cudaStreamCaptureStatusNone) return cudaErrorNotSupported;  // (a capture takes another route)
    if (int r = ring_part(ring, sms, &p)) return r;
    bool next = ring->cur == p && !ring->q.empty() && ring->q.front().f.data == data && ring->q.front().row0 == 0 && ring->q.front().O == O && ring->q.front().K == K;
    if (!next) {  // off the queue: W alone, now
        if (int r = ring_restart(ring, p, cs)) return r;
        if (int r = ring_queue(ring, f, O, K)) return r;
    }
    // the products after cs's work (X); a bias into Y first, cuBLAS adding to it
    if (int r = cudaEventRecord(ring->mark, cs)) return r;
    if (int r = cudaStreamWaitEvent(p->sg, ring->mark, 0)) return r;
    if (bias) bias_rows_kernel<<<(unsigned)std::max<int64_t>(1, std::min<int64_t>(p->gemm * 4, (M * O + 255) / 256)), 256, 0, p->sg>>>(y, bias, M, O);
    const float one = 1.f, beta = bias ? 1.f : 0.f;
    cudaStream_t was = nullptr;
    int target = 0, r = 0, b = 0;
    if (blas->get_stream) blas->get_stream(blas->handle, &was);
    if (blas->get_sm_count_target) blas->get_sm_count_target(blas->handle, &target);
    if ((b = blas->set_stream(blas->handle, p->sg)) == 0 && blas->set_workspace && blas->workspace)
        b = blas->set_workspace(blas->handle, blas->workspace, blas->workspace_bytes);
    if (!b && blas->set_sm_count_target) b = blas->set_sm_count_target(blas->handle, (int)p->gemm);
    for (bool first = true; !b && !r && !ring->q.empty() && ring->q.front().f.data == data && (first || ring->q.front().row0); first = false) {
        if ((r = ring_pump(ring))) break;  // (the front's decode started: its slot is free once the chunks before are read)
        glyd_gpu_ring::Chunk c = ring->q.front();
        if ((r = cudaStreamWaitEvent(p->sg, ring->ready[c.slot], 0))) break;
        b = blas->gemm_ex(blas->handle, 1, 0, (int)c.rows, (int)M, (int)K, &one, ring->buf + c.slot * ring->slot_bytes, 14, (int)K, x, 14, (int)K, &beta, y + c.row0, 14, (int)O, 68, -1);  // op T, op N; CUDA_R_16BF; CUBLAS_COMPUTE_32F; CUBLAS_GEMM_DEFAULT
        if (b || (r = record_on(ring->free_[c.slot], p->sg, p->gg))) break;
        ring->read[c.slot] = true, ring->busy[c.slot] = false;
        ring->q.pop_front();
        ring->issued--;
        r = ring_pump(ring);  // the next decodes, into the slot freed
    }
    if (blas->get_sm_count_target && blas->set_sm_count_target) blas->set_sm_count_target(blas->handle, target);
    if (blas->get_stream) blas->set_stream(blas->handle, was);
    if (b) return GLYD_GPU_BLAS_ERROR + b;
    if (!r) r = record_on(ring->mark, p->sg, p->gg);
    if (!r) r = cudaStreamWaitEvent(cs, ring->mark, 0);  // Y
    return r ? r : (int)cudaGetLastError();
}

GLYD_GPU_API int glyd_gpu_mma12_split_sms(int64_t gpu, int64_t O, int64_t K, int64_t M, int64_t* sms) {
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

GLYD_GPU_API int glyd_gpu_mma_moe_unpack(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base, const uint32_t tiers[3], int64_t E, int64_t O, int64_t K, int64_t P, const int32_t* plan, uint16_t* out, cudaStream_t cs) {
    return mma_moe_unpack_run(Tiered{data, blocks, block_base, {tiers[0], tiers[1], tiers[2]}}, E, O, K, P, plan, out, cs);
}

GLYD_GPU_API int glyd_gpu_mma12_moe_unpack(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4], int64_t E, int64_t O, int64_t K, int64_t P, const int32_t* plan, uint16_t* out, cudaStream_t cs) {
    Nib f;
    if (!nib12(data, exc, exc_base, sym, f)) return cudaErrorInvalidValue;
    return mma_moe_unpack_run(f, E, O, K, P, plan, out, cs);
}

// A mixture-of-experts layer (the E experts' [O, K] matrices stacked, [E O, K]). moe_route: the plan [2 + 2E + P]
// (int32) of the P pairs whose experts are ids [P] (int64; pair j: token j / k, choice j mod k).
GLYD_GPU_API int glyd_gpu_moe_route(const int64_t* ids, int64_t P, int64_t E, int32_t* plan, cudaStream_t cs) {
    if (P < 0 || P >= (1ll << 31) - 2 * E || E < 1 || E > 12288) return cudaErrorInvalidValue;
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
template <class Fmt, int MT, int ACT>
static void mma_moe_launch(Fmt f, int64_t E, int64_t O, int64_t K, const uint16_t* x, int64_t k, int64_t gather, const int32_t* plan, const uint16_t* bias, const void* w, int64_t wf32, uint16_t* y, float* y32, int64_t P, float* parts, int* done, dim3 grid, cudaStream_t cs) {
    auto kernel = mma_moe_kernel<Fmt, MT, ACT>;
    size_t shared = (size_t)4 * MT * 16 * 65 * sizeof(float) + (Fmt::kTable ? 8 * S2_BYTES : 0);
    static std::atomic<int> allowed[MAX_DEVICES];  // the shared memory allowed on a device, once
    int dev = current_device();
    if (dev >= MAX_DEVICES || !allowed[dev].load()) {
        cudaFuncSetAttribute((const void*)kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared);
        if (dev < MAX_DEVICES) allowed[dev] = 1;
    }
    kernel<<<grid, 256, shared, cs>>>(f, O, K, plan, (int)E, k, (int)gather, bf(x), bf(bias), w, (int)wf32, bf(y), y32, P, parts, done);
}

// Blocks splitting K for the experts' product (units: its units by the experts that can be hit): up to the blocks
// the GPU holds at once (a wave more costs a block's whole time), at least 16 steps (two a warp) each. On a GDDR GPU
// a block an SM keeps its memory busy (an A10: split, the product was slower, measured); an HBM part (A100, H100,
// B200: two to six times the bandwidth an SM) takes two. GLYD_GPU_MOE_SLOTS: the blocks held at once (tests).
static int64_t moe_splits(int64_t units, int64_t K) {
    static int64_t forced = getenv("GLYD_GPU_MOE_SLOTS") ? atoll(getenv("GLYD_GPU_MOE_SLOTS")) : 0;
    int dev = current_device(), major = attribute(cudaDevAttrComputeCapabilityMajor, dev);
    bool hbm = attribute(cudaDevAttrComputeCapabilityMinor, dev) == 0 && major >= 8 && major <= 10;
    int64_t slots = forced ? forced : sm_count(dev) * (hbm ? 2 : 1);
    return std::max<int64_t>(1, std::min<int64_t>(slots / units, K / 256));
}

// Many pairs an expert (a prompt): mma_gemm_big_kernel's variant 1 (128 of them by two row blocks a block, a weight
// decoded once for the 128), blockIdx.z the experts hit.
template <class Fmt>
static void mma_moe_big(Fmt f, int64_t E, int64_t O, int64_t K, const uint16_t* x, int64_t T, int64_t k, int64_t gather, const int32_t* plan, int64_t act, const uint16_t* bias, const void* w, int64_t wf32, uint16_t* y, float* y32, cudaStream_t cs) {
    using C = Big<4, 4, 3, 2>;
    auto kernel = mma_gemm_big_kernel<Fmt, 4, 4, 3, 2, true>;
    static std::atomic<int> known[MAX_DEVICES];
    per_sm((const void*)kernel, C::THREADS, C::SHARED, known, current_device());  // its shared memory allowed, once
    dim3 grid((unsigned)((T + C::TM - 1) / C::TM), (unsigned)(act ? O / 128 : (O / 64 + 1) / 2), (unsigned)std::min(E, T * k));
    kernel<<<grid, C::THREADS, C::SHARED, cs>>>(f, O, K, 0, 0, bf(x), bf(bias), bf(y), y32, MoePairs{plan, (int)E, k, (int)gather, (int)act, w, (int)wf32});
}

template <class Fmt>
static int mma_moe_run(Fmt f, int64_t E, int64_t O, int64_t K, const uint16_t* x, int64_t T, int64_t k, int64_t gather, const int32_t* plan, int64_t act, const uint16_t* bias, const void* w, int64_t wf32, const int64_t* ids, uint16_t* y, void* ws, size_t ws_bytes, int* done, cudaStream_t cs, size_t* need) {
    if (E < 1 || E > 12288 || O < 64 || O % 64 || K < 16 || K % 16 || T < 0 || k < 1 || act < 0 || act > 2 || (act && (O % 128 || w))) return cudaErrorInvalidValue;
    int64_t P = T * k, MT = T <= 16 ? 1 : T <= 32 ? 2 : 4, units = act ? O / 128 : O / 64, hits = std::min(E, P);
    int64_t splits = P ? moe_splits(units * hits, K) : 1;
    size_t bytes = ((w ? 1 : 0) + (splits > 1 ? splits : 0)) * (size_t)(P * O) * sizeof(float);  // [Y32][parts]
    if (need) {
        *need = bytes;
        return 0;
    }
    if (!fits(ws, ws_bytes, bytes) || (w && !ids) || (splits > 1 && !done)) return cudaErrorInvalidValue;
    if (P == 0) return 0;
    float* y32 = w ? (float*)ws : nullptr;
    float* parts = splits > 1 ? (float*)ws + (w ? P * O : 0) : nullptr;
    dim3 grid((unsigned)units, (unsigned)hits, (unsigned)splits);
    auto run = [&](auto mt) {
        constexpr int M = decltype(mt)::value;
        if (act == 1) mma_moe_launch<Fmt, M, 1>(f, E, O, K, x, k, gather, plan, bias, w, wf32, y, y32, P, parts, done, grid, cs);
        else if (act == 2) mma_moe_launch<Fmt, M, 2>(f, E, O, K, x, k, gather, plan, bias, w, wf32, y, y32, P, parts, done, grid, cs);
        else mma_moe_launch<Fmt, M, 0>(f, E, O, K, x, k, gather, plan, bias, w, wf32, y, y32, P, parts, done, grid, cs);
    };
    // From 48 pairs an expert, on average (a prompt): the tiled GEMM (measured on an A10, OLMoE-1B-7B).
    static int64_t big = getenv("GLYD_GPU_MOE_BIG") ? atoll(getenv("GLYD_GPU_MOE_BIG")) : 48;
    if (T * k >= big * E && K % 64 == 0 && (uintptr_t)x % 16 == 0) mma_moe_big(f, E, O, K, x, T, k, gather, plan, act, bias, w, wf32, y, y32, cs);
    else if (MT == 1) run(std::integral_constant<int, 1>());
    else if (MT == 2) run(std::integral_constant<int, 2>());
    else run(std::integral_constant<int, 4>());
    if (w) moe_sum_kernel<<<(unsigned)((T * O / 4 + 255) / 256), 256, 0, cs>>>(y32, ids, (int)E, T, k, O, bf(y));
    return cudaGetLastError();
}

GLYD_GPU_API int glyd_gpu_mma_moe_workspace(int64_t E, int64_t O, int64_t K, int64_t T, int64_t k, int64_t act, int64_t weighted, size_t* bytes) {
    int64_t one = 1;  // a stand-in for the weights: only whether there are any counts
    return bytes ? mma_moe_run(Tiered{}, E, O, K, nullptr, T, k, 0, nullptr, act, nullptr, weighted ? &one : nullptr, 0, nullptr, nullptr, nullptr, 0, nullptr, 0, bytes) : cudaErrorInvalidValue;
}

GLYD_GPU_API int glyd_gpu_mma_moe(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base, const uint32_t tiers[3], int64_t E, int64_t O, int64_t K, const uint16_t* x, int64_t T, int64_t k, int64_t gather, const int32_t* plan, int64_t act, const uint16_t* bias, const void* w, int64_t wf32, const int64_t* ids, uint16_t* y, void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs) {
    return mma_moe_run(Tiered{data, blocks, block_base, {tiers[0], tiers[1], tiers[2]}}, E, O, K, x, T, k, gather, plan, act, bias, w, wf32, ids, y, workspace, workspace_bytes, done, cs, nullptr);
}

GLYD_GPU_API int glyd_gpu_mma12_moe_workspace(int64_t E, int64_t O, int64_t K, int64_t T, int64_t k, int64_t act, int64_t weighted, size_t* bytes) {
    return glyd_gpu_mma_moe_workspace(E, O, K, T, k, act, weighted, bytes);  // the same for both layouts
}

GLYD_GPU_API int glyd_gpu_mma12_moe(const uint8_t* data, const uint32_t* exc, const int32_t* exc_base, const uint32_t sym[4], int64_t E, int64_t O, int64_t K, const uint16_t* x, int64_t T, int64_t k, int64_t gather, const int32_t* plan, int64_t act, const uint16_t* bias, const void* w, int64_t wf32, const int64_t* ids, uint16_t* y, void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs) {
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

GLYD_GPU_API int glyd_gpu_attn_decode_workspace(int64_t D, int64_t tlen, int64_t pairs, int64_t P, size_t* bytes) {
    if (!bytes || !(D == 64 || D == 128) || tlen >= 64 || P + (tlen > 0) <= 0) return cudaErrorInvalidValue;
    int64_t per, splits = attn_splits(tlen, pairs, P, per);
    *bytes = (size_t)(pairs * splits * 16 * (D + 2)) * sizeof(float);
    return 0;
}

GLYD_GPU_API int glyd_gpu_attn_decode(const uint16_t* q, int64_t D, const uint8_t* kd, const uint8_t* kb, const int32_t* kbb, const uint32_t kt[3], const uint8_t* vd, const uint8_t* vb, const int32_t* vbb, const uint32_t vt[3], const uint16_t* tk, const uint16_t* tv, int64_t tlen, int64_t pairs, int64_t G, int64_t P, double scale, uint16_t* out, void* workspace, size_t workspace_bytes, int* done, cudaStream_t cs) {
    size_t need;
    if (G < 1 || G > 16 || glyd_gpu_attn_decode_workspace(D, tlen, pairs, P, &need) || !fits(workspace, workspace_bytes, need) || !done) return cudaErrorInvalidValue;
    int64_t per, splits = attn_splits(tlen, pairs, P, per);
    float sl2 = (float)(scale * 1.4426950408889634);
    Tiers kts{kt[0], kt[1], kt[2]}, vts{vt[0], vt[1], vt[2]};
    auto launch = [&](auto kernel) {
        kernel<<<dim3(pairs, splits), 32 * ATT_WARPS, 0, cs>>>(bf(q), kd, kb, kbb, kts, vd, vb, vbb, vts, tlen ? bf(tk) : nullptr, tlen ? bf(tv) : nullptr, (int)tlen, (int)pairs, (int)G, (int)P, (int)per, sl2, (float*)workspace, done, bf(out));
    };
    if (D == 128) launch(attn_decode_kernel<128>);
    else launch(attn_decode_kernel<64>);
    return cudaGetLastError();
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

torch::Tensor lane_bits(torch::Tensor w, torch::Tensor len, int64_t tw, int64_t V) {
    const c10::cuda::CUDAGuard guard(w.device());
    auto bits = torch::empty({tiles_for(w.numel(), tw) * 32}, w.options().dtype(torch::kInt32));
    ok(glyd_gpu_lane_bits(ptr<uint16_t>(w), w.numel(), ptr<uint8_t>(len), tw, V, ptr<uint32_t>(bits), current_stream()), "lane_bits");
    return bits;
}

void write_codes(torch::Tensor w, torch::Tensor len, torch::Tensor code, torch::Tensor offs, torch::Tensor out, int64_t tw, int64_t V) {
    const c10::cuda::CUDAGuard guard(w.device());
    ok(glyd_gpu_write_codes(ptr<uint16_t>(w), w.numel(), ptr<uint8_t>(len), ptr<uint32_t>(code), ptr<uint32_t>(offs), ptr<uint32_t>(out), tw, V, current_stream()), "write_codes");
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

void mma_gemm_big(torch::Tensor data, torch::Tensor blocks, torch::Tensor block_base, std::vector<int64_t> tiers, int64_t O, int64_t K, torch::Tensor x, torch::Tensor bias, torch::Tensor y, int64_t variant) {
    uint32_t t[3];
    words(tiers, 3, t, "three tiers");
    const c10::cuda::CUDAGuard guard(data.device());
    TORCH_CHECK(O % 64 == 0 && K % 64 == 0 && x.is_contiguous() && x.size(1) == K && (uintptr_t)x.data_ptr() % 16 == 0, "O a multiple of 64, K of 64, X contiguous [M, K]");
    int64_t M = x.size(0);
    size_t bytes = 0;
    ok(glyd_gpu_mma_gemm_big_workspace(O, K, M, variant, &bytes), "mma_gemm_big");
    auto ws = scratch(bytes, x);
    static auto* done_of = new Counters;
    ok(glyd_gpu_mma_gemm_big(ptr<uint8_t>(data), ptr<uint8_t>(blocks), ptr<int32_t>(block_base), t, O, K, ptr<uint16_t>(x), M, opt(bias), ptr<uint16_t>(y), variant, addr(ws), bytes, counters(*done_of, data, (M + 127) / 128 * (O / 64), 1 << 18), current_stream()), "mma_gemm_big");
}

void mma12_gemm_big(torch::Tensor data, torch::Tensor exc, torch::Tensor exc_base, std::vector<int64_t> sym, int64_t O, int64_t K, torch::Tensor x, torch::Tensor bias, torch::Tensor y, int64_t variant) {
    uint32_t s[4];
    words(sym, 4, s, "the 12-bit layout's four words (its base)");
    const c10::cuda::CUDAGuard guard(data.device());
    TORCH_CHECK(O % 64 == 0 && K % 64 == 0 && x.is_contiguous() && x.size(1) == K && (uintptr_t)x.data_ptr() % 16 == 0, "O a multiple of 64, K of 64, X contiguous [M, K]");
    int64_t M = x.size(0);
    size_t bytes = 0;
    ok(glyd_gpu_mma12_gemm_big_workspace(O, K, M, variant, &bytes), "mma12_gemm_big");
    auto ws = scratch(bytes, x);
    static auto* done_of = new Counters;
    ok(glyd_gpu_mma12_gemm_big(ptr<uint8_t>(data), ptr<uint32_t>(exc), ptr<int32_t>(exc_base), s, O, K, ptr<uint16_t>(x), M, opt(bias), ptr<uint16_t>(y), variant, addr(ws), bytes, counters(*done_of, data, (M + 127) / 128 * (O / 64), 1 << 18), current_stream()), "mma12_gemm_big");
}

// Many tokens on Ampere and Ada (to 64 a launch, in chunks past that).
void mma12_gemm_mid(torch::Tensor data, torch::Tensor exc, torch::Tensor exc_base, std::vector<int64_t> sym, int64_t O, int64_t K, torch::Tensor x, torch::Tensor bias, torch::Tensor y) {
    uint32_t s[4];
    words(sym, 4, s, "the 12-bit layout's four words (its base)");
    const c10::cuda::CUDAGuard guard(data.device());
    TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major >= 8, "mma12_gemm_mid: Ampere or later");
    TORCH_CHECK(O % 64 == 0 && K % 64 == 0 && x.is_contiguous() && x.size(1) == K && (uintptr_t)x.data_ptr() % 16 == 0, "O a multiple of 64, K of 64, X contiguous [M, K]");
    TORCH_CHECK((uintptr_t)data.data_ptr() % 16 == 0 && (uintptr_t)exc.data_ptr() % 16 == 0 && exc.numel() % 4 == 0, "the pack 16-byte aligned, exc padded to 4 (pack_mma12)");
    int64_t M = x.size(0);
    size_t bytes = 0;
    ok(glyd_gpu_mma12_gemm_mid_workspace(O, K, M, &bytes), "mma12_gemm_mid");
    auto ws = scratch(bytes, data);
    static auto* done_of = new Counters;
    ok(glyd_gpu_mma12_gemm_mid(ptr<uint8_t>(data), ptr<uint32_t>(exc), ptr<int32_t>(exc_base), s, O, K, ptr<uint16_t>(x), M, opt(bias), ptr<uint16_t>(y), addr(ws), bytes, counters(*done_of, data, O / 64, 1 << 16), current_stream()), "mma12_gemm_mid");
}

void mma12_gemm_wg(torch::Tensor data, torch::Tensor exc, torch::Tensor exc_base, std::vector<int64_t> sym, int64_t O, int64_t K, torch::Tensor x, torch::Tensor bias, torch::Tensor y) {
    uint32_t s[4];
    words(sym, 4, s, "the 12-bit layout's four words (its base)");
    const c10::cuda::CUDAGuard guard(data.device());
    TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major == 9 && at::cuda::getCurrentDeviceProperties()->minor == 0, "wgmma: Hopper (compute capability 9.0) alone");
    TORCH_CHECK(O % 64 == 0 && K % 64 == 0 && x.is_contiguous() && x.size(1) == K && (uintptr_t)x.data_ptr() % 16 == 0, "O a multiple of 64, K of 64, X contiguous [M, K]");
    TORCH_CHECK((uintptr_t)data.data_ptr() % 16 == 0 && (uintptr_t)exc.data_ptr() % 16 == 0 && exc.numel() % 4 == 0, "the pack 16-byte aligned, exc padded to 4 (pack_mma12)");
    int64_t M = x.size(0);
    size_t bytes = 0;
    ok(glyd_gpu_mma12_gemm_wg_workspace(O, K, M, &bytes), "mma12_gemm_wg");
    auto ws = scratch(bytes, data);
    static auto* done_of = new Counters;
    ok(glyd_gpu_mma12_gemm_wg(ptr<uint8_t>(data), ptr<uint32_t>(exc), ptr<int32_t>(exc_base), s, O, K, ptr<uint16_t>(x), M, opt(bias), ptr<uint16_t>(y), addr(ws), bytes, counters(*done_of, data, O / 64, 1 << 16), current_stream()), "mma12_gemm_wg");
}

// The routes: (route, last) for M tokens on gpu; the current device's gpu.
std::tuple<int64_t, int64_t> mma_route(int64_t gpu, int64_t O, int64_t K, int64_t M) {
    int r;
    int64_t last;
    ok(glyd_gpu_mma_route(gpu, O, K, M, &r, &last), "mma_route");
    return {r, last};
}

std::tuple<int64_t, int64_t> mma12_route(int64_t gpu, int64_t O, int64_t K, int64_t M) {
    int r;
    int64_t last;
    ok(glyd_gpu_mma12_route(gpu, O, K, M, &r, &last), "mma12_route");
    return {r, last};
}

int64_t mma12_split_sms(int64_t gpu, int64_t O, int64_t K, int64_t M) {
    int64_t sms;
    ok(glyd_gpu_mma12_split_sms(gpu, O, K, M, &sms), "mma12_split_sms");
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

void mma12_unpack_split(torch::Tensor data, torch::Tensor exc, torch::Tensor exc_base, std::vector<int64_t> sym, int64_t K, int64_t row0, int64_t rows, torch::Tensor out, int64_t sms) {
    uint32_t s[4];
    words(sym, 4, s, "the 12-bit layout's four words (its base)");
    const c10::cuda::CUDAGuard guard(data.device());
    TORCH_CHECK(row0 % 64 == 0 && rows % 64 == 0 && K % 64 == 0 && out.numel() >= rows * K, "rows a multiple of 64, K of 64");
    ok(glyd_gpu_mma12_unpack_split(ptr<uint8_t>(data), ptr<uint32_t>(exc), ptr<int32_t>(exc_base), s, K, row0, rows, ptr<uint16_t>(out), sms, current_stream()), "mma12_unpack_split");
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

void moe_route(torch::Tensor ids, int64_t E, torch::Tensor plan) {
    const c10::cuda::CUDAGuard guard(ids.device());
    TORCH_CHECK(ids.scalar_type() == torch::kInt64 && ids.is_contiguous() && plan.scalar_type() == torch::kInt32 && plan.numel() >= 2 + 2 * E + ids.numel() && plan.device() == ids.device(), "ids int64, plan int32 [2 + 2E + P] on the same GPU");
    ok(glyd_gpu_moe_route(ptr<int64_t>(ids), ids.numel(), E, ptr<int32_t>(plan), current_stream()), "moe_route");
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
    m.def("mma12_gemm_big", &mma12_gemm_big);
    m.def("mma12_unpack", &mma12_unpack);
    m.def("mma12_unpack_split", &mma12_unpack_split);
    m.def("mma12_split_sms", &mma12_split_sms);
    m.def("mma12_gemm_wg", &mma12_gemm_wg);
    m.def("mma12_gemm_mid", &mma12_gemm_mid);
    m.def("attn_decode", &attn_decode);
    m.def("moe_route", &moe_route);
    m.def("mma_moe", &mma_moe);
    m.def("mma12_moe", &mma12_moe);
    m.def("mma_moe_unpack", &mma_moe_unpack);
    m.def("mma12_moe_unpack", &mma12_moe_unpack);
    m.def("mma_gemm_big", &mma_gemm_big);
    m.def("mma_route", &mma_route);
    m.def("mma12_route", &mma12_route);
    m.def("gpu", &gpu);
    m.def("mma_linear", &mma_linear);
    m.def("mma12_linear", &mma12_linear);
    m.def("fast_bgemv", &fast_bgemv);
    m.def("fast_gemm", &fast_gemm);
    m.def("lane_bits", &lane_bits);
    m.def("write_codes", &write_codes);
    m.def("decode", &decode);
    m.def("gemv", &gemv);
    m.def("fast_gemv", &fast_gemv);
    m.def("fast_decode", &fast_decode);
}
#endif
