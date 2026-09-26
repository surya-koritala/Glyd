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
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <mma.h>
#include <map>

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

// The 12-bit layout (mma12): a weight's exponent a 4-bit code into the
// tensor's 15 commonest exponents, code 15 its step's exception list. A
// warp step: its lanes' codes, 16 bytes a lane (weight i at bits 4(i mod 8)
// of word i / 8), then its 1024 sign-and-mantissa bytes as the tiered
// layout's (two halves). An exception: its weight (lane * 32 + i) and its
// exponent << 16; a step's run from exc_base[step] to exc_base[step + 1].
// Four codes are three byte permutes: the low eight symbols' and the high
// eight's by the codes' low three bits, then each byte from one or the
// other by the fourth.
constexpr int64_t STEP12 = 1536;

struct Nib {
    const uint8_t* data;
    const uint32_t* exc;
    const int32_t* exc_base;
    uint32_t sym[4];  // the codes' exponents, 4 a word (code 15: 0)
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
    __device__ __forceinline__ void decode(const St& st, int lane, uint32_t*, const uint32_t*, uint32_t R[16]) const {
        uint32_t ew[8];
#pragma unroll
        for (int q = 0; q < 8; q++) {
            uint32_t n = st.nb[q >> 1] >> (16 * (q & 1)), n7 = n & 0x7777u;
            ew[q] = __byte_perm(__byte_perm(sym[0], sym[1], n7), __byte_perm(sym[2], sym[3], n7), ((n >> 1) & 0x4444u) | 0x3210u);
        }
        // The step's exceptions (the same run for every lane of the warp):
        // each word takes its byte where the exception is this lane's and in it.
        for (int k = st.e0; k < st.e1; k++) {
            uint32_t x = __ldg(exc + k), i = x & 31, sel = 0x3210u + ((4u - (i & 3)) << (4 * (i & 3)));
            uint32_t w = (int)((x >> 5) & 31) == lane ? i >> 2 : 8u;
#pragma unroll
            for (int q = 0; q < 8; q++) {
                uint32_t v = __byte_perm(ew[q], x >> 16, sel);
                ew[q] = w == (uint32_t)q ? v : ew[q];
            }
        }
        pairs(st.sw, ew, R);
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
        // The step's loads: W's, then the inputs.
        typename Fmt::St st;
        uint32_t a[MT][4];
        auto load = [&](int64_t s) {
            f.load(st, rb * KS + s, lane);
#pragma unroll
            for (int mt = 0; mt < MT; mt++) {
                int64_t r0 = mt * 16 + g, r1 = r0 + 8;
                const __nv_bfloat16* xr = X + s * 16 + t * 2;
                a[mt][0] = r0 < M ? *(const uint32_t*)(xr + r0 * K) : 0u;
                a[mt][2] = r0 < M ? *(const uint32_t*)(xr + r0 * K + 8) : 0u;
                a[mt][1] = r1 < M ? *(const uint32_t*)(xr + r1 * K) : 0u;
                a[mt][3] = r1 < M ? *(const uint32_t*)(xr + r1 * K + 8) : 0u;
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
            f.decode(cur, lane, s2, tab, R);
#pragma unroll
            for (int mt = 0; mt < MT; mt++)
#pragma unroll
                for (int nn = 0; nn < 8; nn++) mma16816(acc[mt][nn], ca[mt], R[2 * nn], R[2 * nn + 1]);
        }
        // The warps' sums (C fragment rows g and g + 8, columns 2t and 2t + 1
        // of each n-tile): warps 4-7 put theirs in shared memory, 0-3 add
        // them to theirs and put those.
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
        if (first != fin) {
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0) {
                last = atomicAdd(done + rb, 1) == fin - first;
                if (last) done[rb] = 0;  // ready for the next product
            }
            __syncthreads();
            if (last) {
                __threadfence();
                for (int i = threadIdx.x; i < M * 64; i += 256) {
                    int r = i / 64, c = i % 64;
                    float v = 0.f;
                    for (int64_t b = first; b <= fin; b++) v += __ldcg(parts + ((b + rb) * M + r) * 64 + c);
                    Y[r * O + rb * 64 + c] = __float2bfloat16(v + (bias ? __bfloat162float(bias[rb * 64 + c]) : 0.f));
                }
            }
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
// added in a fixed order by finish_kernel).
constexpr int BIG_KK = 4;  // steps a stage
template <int CW, int PW, int NB, int RBB> struct Big {
    static constexpr int THREADS = 32 * (CW + PW);
    static constexpr int TM = 64 * CW / RBB;             // tokens a block: CW / RBB slices of 64, by RBB row blocks
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

template <int THREADS> __device__ __forceinline__ void bar_sync(int id) { asm volatile("bar.sync %0, %1;\n" ::"r"(id), "n"(THREADS)); }
template <int THREADS> __device__ __forceinline__ void bar_arrive(int id) { asm volatile("bar.arrive %0, %1;\n" ::"r"(id), "n"(THREADS)); }

template <class Fmt, int CW, int PW, int NB, int RBB>
__global__ void __launch_bounds__(Big<CW, PW, NB, RBB>::THREADS, 1) mma_gemm_big_kernel(Fmt f, int64_t O, int64_t K, int64_t M, int64_t stages_per_split, const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ Y, float* __restrict__ Y32) {
    using C = Big<CW, PW, NB, RBB>;
    extern __shared__ uint4 smem[];  // X's tiles [NB][TM rows][8 16-byte chunks, swizzled], then W's [NB][B_UINT4]
    __shared__ uint32_t tab[Fmt::kTable ? 256 : 1];
    if constexpr (Fmt::kTable) fill_groups(tab);
    __syncthreads();
    const uint32_t a_base = (uint32_t)__cvta_generic_to_shared(smem);
    uint4* Bs = smem + NB * C::A_BYTES / 16;
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int64_t KS = K / 16, RB = O / 64, tile = blockIdx.x, pair = blockIdx.y;
    int64_t st0 = (int64_t)blockIdx.z * stages_per_split, nst = min(KS / BIG_KK, st0 + stages_per_split) - st0;
    if (warp >= CW) {
        // Producer p: X's chunks id = pt + 32 PW i of a stage's; W's steps STEPS p on of its 4 RBB (row block, step).
        int pt = tid - 32 * CW, p = warp - CW;
        typename Fmt::St st[C::STEPS];
        auto d_load = [&](int64_t j) {
#pragma unroll
            for (int i = 0; i < C::STEPS; i++) {
                int u = p * C::STEPS + i;  // of the stage's 4 RBB: row block u / 4, step u % 4
                int64_t rb = min(pair * RBB + u / BIG_KK, RB - 1), s = min((st0 + j) * BIG_KK + u % BIG_KK, KS - 1);
                f.load(st[i], rb * KS + s, lane);
            }
        };
        d_load(0);
        for (int64_t j = 0; j < nst; j++) {
            int b = (int)(j % NB);
            if (j >= NB) bar_sync<C::THREADS>(1 + NB + b);  // consumers are done with stage j - NB
            uint32_t slot = a_base + (uint32_t)b * C::A_BYTES;
            int64_t col = (st0 + j) * BIG_KK * 16;
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
        return;
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
    // C fragments: token rows g and g + 8 of each m-tile, columns 2t and 2t + 1 of each n-tile.
    int64_t my_rb = pair * RBB + rbl;
    if (my_rb >= RB) return;
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
                if (Y32) {
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
// 4 steps (its quarter of each: a word of codes and 8 sign-and-mantissa
// bytes a lane) into A fragments, then 4 wgmma of 64 rows by NT tokens by
// 16 columns. The work (units by stages) is split evenly over the blocks
// (stream-K, as mma_gemm_kernel's): a unit covered by several blocks is
// summed by the last to finish, in block order.
__device__ __forceinline__ uint64_t sw128_desc(uint32_t addr) {
    // Start address >> 4, stride between 8-row groups 1024 bytes, 128-byte swizzle (K-major).
    return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}

__device__ __forceinline__ void wg_fence() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void wg_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
#endif
}
template <int N> __device__ __forceinline__ void wg_wait() {  // until at most N groups of products are running
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
#endif
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
__device__ __forceinline__ void mbar_init(uint32_t bar, int count) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar), "r"(count)); }
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar), "r"(bytes) : "memory"); }
__device__ __forceinline__ void mbar_arrive(uint32_t bar) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(bar) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
    uint32_t ok;
    do {
        asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n" : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
    } while (!ok);
}
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
__device__ __forceinline__ void wgmma4_rs(float (&d)[128], const uint32_t (&a)[4][4], const uint64_t (&db)[4], int acc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %148, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, {%128, %129, %130, %131}, %144, p, 1, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, {%132, %133, %134, %135}, %145, 1, 1, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, {%136, %137, %138, %139}, %146, 1, 1, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, {%140, %141, %142, %143}, %147, 1, 1, 1, 0;\n"
                 "wgmma.commit_group.sync.aligned;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
                 : "r"(a[0][0]), "r"(a[0][1]), "r"(a[0][2]), "r"(a[0][3]), "r"(a[1][0]), "r"(a[1][1]), "r"(a[1][2]), "r"(a[1][3]), "r"(a[2][0]), "r"(a[2][1]), "r"(a[2][2]), "r"(a[2][3]), "r"(a[3][0]), "r"(a[3][1]), "r"(a[3][2]), "r"(a[3][3]), "l"(db[0]), "l"(db[1]), "l"(db[2]), "l"(db[3]), "r"(acc));
}
#endif

template <int NT, int WG> struct Tma12 {
    static constexpr int THREADS = 128 * WG + 32;  // WG consumer warpgroups (a warpgroup's first warp a multiple of 4), the TMA warp
    static constexpr int R = 64 * WG;              // W's rows a unit (a row block a warpgroup)
    static constexpr int XB = NT * 128;            // X's tile a stage: NT tokens by 64 columns
    static constexpr int CB = 4 * (int)STEP12, EB = 1024, BB = 32;  // a row block's steps a stage, its exceptions (up to 256), their bounds
    static constexpr int RBB = CB + EB + BB;
    static constexpr int SLOT = (XB + WG * RBB + 1023) / 1024 * 1024;
    static constexpr int NS = 200 * 1024 / SLOT < 8 ? 200 * 1024 / SLOT : 8;  // stages in flight
    static constexpr int SHARED = NS * SLOT + 1024 + 16 * NS;               // alignment, the full and empty barriers
};

template <int NT, int WG>
__global__ void __launch_bounds__(Tma12<NT, WG>::THREADS, 1) mma12_tma_kernel(const __grid_constant__ CUtensorMap xmap, Nib f, int64_t O, int64_t K, int64_t M, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ Y, float* __restrict__ parts, int* __restrict__ done) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    using C = Tma12<NT, WG>;
    extern __shared__ __align__(1024) uint8_t smem_raw[];  // NS slots [X tile | a row block's steps, exceptions, bounds | the other's], the barriers
    __shared__ int last;
    uint32_t raw = (uint32_t)__cvta_generic_to_shared(smem_raw), base = (raw + 1023) & ~1023u;
    uint8_t* gbase = smem_raw + (base - raw);
    uint32_t full = base + C::NS * C::SLOT, empty = full + 8 * C::NS;
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int64_t KS = K / 16, RB = O / 64, U = (RB + WG - 1) / WG * (K / 64), nb = gridDim.x;
    int S = (int)(K / 64);
    int64_t u0 = blockIdx.x * U / nb;
    int n = (int)((blockIdx.x + 1) * U / nb - u0);  // the block's stages, u0 on: unit (WG row blocks) u / S, stage u mod S
    int p0 = (int)(u0 / S), s0 = (int)(u0 - (int64_t)p0 * S);
    if (tid == 0) {
        for (int i = 0; i < C::NS; i++) {
            mbar_init(full + 8 * i, 1);
            mbar_init(empty + 8 * i, 4 * WG);
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();
    if (warp == 4 * WG) {
        // The TMA warp: lane j mod 8 issues stage j. Stages go in batches of 8,
        // lane l's bounds for stage l of the batch loaded while the batch before
        // was issued (a load pends for the whole warp: one register set a lane,
        // reloaded by one lane a stage, would wait out every load).
        int e[5 * WG], en[5 * WG];  // exc_base at the stage's 4 steps and the next, each row block: this batch's, the next's
        auto bounds = [&](int p, int s) {
#pragma unroll
            for (int r = 0; r < WG; r++)
#pragma unroll
                for (int i = 0; i < 5; i++) en[5 * r + i] = __ldg(f.exc_base + min((int64_t)WG * p + r, RB - 1) * KS + 4 * s + i);
        };
        auto ahead = [&](int p, int s, int k) {  // the stage k after (p, s)
            s += k;
            p += s / S;
            bounds(p, s % S);
        };
        if (lane < 8 && lane < n) ahead(p0, s0, lane);
        for (int j = 0, p = p0, s = s0; j < n; j++, s = s + 1 == S ? 0 : s + 1, p += s == 0) {
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
                    // The exceptions' run, widened to 16 bytes each way (pack_mma12 pads exc); a longer one is read from global memory.
                    int lo = e[5 * r], hi = e[5 * r + 4];
                    a[r] = lo & ~3;
                    na[r] = hi > lo ? ((hi + 3) & ~3) - a[r] : 0;
                    if (4 * na[r] > C::EB) na[r] = -1;
                    bytes += na[r] > 0 ? 4 * na[r] : 0;
                    int* bs = (int*)(gbase + sl * C::SLOT + C::XB + r * C::RBB + C::CB + C::EB);
#pragma unroll
                    for (int i = 0; i < 5; i++) bs[i] = e[5 * r + i];
                    bs[5] = na[r] < 0 ? -1 : a[r];
                }
                mbar_expect_tx(fb, bytes);
                tma_2d(xs, &xmap, (int)(s * 64), 0, fb);
#pragma unroll
                for (int r = 0; r < WG; r++) {
                    uint32_t cs = xs + C::XB + r * C::RBB;
                    bulk_g2s(cs, f.data + (min((int64_t)WG * p + r, RB - 1) * KS + 4 * s) * STEP12, C::CB, fb, true);
                    if (na[r] > 0) bulk_g2s(cs + C::CB, f.exc + a[r], 4 * na[r], fb);
                }
            }
            __syncwarp();
        }
        return;
    }
    // Consumer warpgroup wg: row block WG p + wg of each unit p; warp w its rows 16w to 16w + 15.
    // Two sets of A registers, used in turn: a stage's products may still run while the next decodes.
    int ct = tid, wg = ct >> 7, w = (ct >> 5) & 3, g = lane >> 2, t = lane & 3;
    float d[NT / 2];  // rows 16w + g (+ 8), tokens 8jn + 2t (+ 1): d[4jn + 2h + c]; set by a row pair's first product
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
        const uint32_t* se = (const uint32_t*)(sp + C::CB);
        int eb[5];
#pragma unroll
        for (int i = 0; i < 5; i++) eb[i] = bs[i];
        int ea = bs[5];
        // The stage's loads first (8 in flight at once), then its 4 steps' codes, then one run of exceptions for all 4.
        uint32_t nw[4];
        uint2 sw[4];
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            const uint8_t* q = sp + kk * STEP12;
            nw[kk] = *(const uint32_t*)(q + 16 * lane + 4 * w);
            sw[kk] = *(const uint2*)(q + 512 + 512 * (w >> 1) + 16 * lane + 8 * (w & 1));
        }
        uint32_t ew[8];  // step kk's exponent words 2w and 2w + 1 of each lane: ew[2kk], ew[2kk + 1]
#pragma unroll
        for (int q = 0; q < 8; q++) {
            uint32_t c = nw[q >> 1] >> (16 * (q & 1)), c7 = c & 0x7777u;
            ew[q] = __byte_perm(__byte_perm(f.sym[0], f.sym[1], c7), __byte_perm(f.sym[2], f.sym[3], c7), ((c >> 1) & 0x4444u) | 0x3210u);
        }
        // The exceptions in this warp's rows: entry k is step kk's while eb[kk] <= k < eb[kk + 1]; weight i of a
        // lane in word i / 4, words 2w and 2w + 1 here.
        for (int k = eb[0]; k < eb[4]; k++) {
            uint32_t x = ea >= 0 ? se[k - ea] : __ldg(f.exc + k), i = x & 31, sel = 0x3210u + ((4u - (i & 3)) << (4 * (i & 3)));
            uint32_t kk = (k >= eb[1]) + (k >= eb[2]) + (k >= eb[3]);
            uint32_t hq = (int)((x >> 5) & 31) == lane && (int)(i >> 3) == w ? 2 * kk + ((i >> 2) & 1) : 8u;
#pragma unroll
            for (int q = 0; q < 8; q++) {
                uint32_t v = __byte_perm(ew[q], x >> 16, sel);
                ew[q] = hq == (uint32_t)q ? v : ew[q];
            }
        }
        // A fragments: rows g and g + 8 (words 2w and 2w + 1), columns 2t and 8 + 2t (bytes 0-1 and 2-3).
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            uint32_t y0 = __byte_perm(sw[kk].x, ew[2 * kk], 0x5140), y1 = __byte_perm(sw[kk].y, ew[2 * kk + 1], 0x5140);
            uint32_t y2 = __byte_perm(sw[kk].x, ew[2 * kk], 0x7362), y3 = __byte_perm(sw[kk].y, ew[2 * kk + 1], 0x7362);
            A[kk][0] = __funnelshift_r(y0, y0, 1);
            A[kk][1] = __funnelshift_r(y1, y1, 1);
            A[kk][2] = __funnelshift_r(y2, y2, 1);
            A[kk][3] = __funnelshift_r(y3, y3, 1);
        }
        // (ptxas moves each product's descriptor into the same uniform registers just before it, so the four
        // run one after another: "serialized", its C7513.)
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
    // A unit's stages, then its sum out: no branch touches d while products run.
    for (int j = 0, p = p0, s = s0; j < n; p++, s = 0) {
        int len = min(S - s, n - j);
        for (int k = 0; k < len; k += 2) {
            stage(j + k, k == 0, A0, A1);
            if (k + 1 < len) stage(j + k + 1, false, A1, A0);
        }
        wg_wait<0>();
        keep(A0);
        keep(A1);
        keep_d();
        if (held >= 0) release(held);
        held = -1;
        float r[NT / 2];
#pragma unroll
        for (int i = 0; i < NT / 2; i++) r[i] = d[i];
        // The block's part of unit p: to Y, or to its slot, the last block to finish adding the slots in block order.
        int64_t first = block_of_step((int64_t)p * S, nb, U), fin = block_of_step((int64_t)p * S + S - 1, nb, U);
        if (first != fin) {
            // Slots [NT / 8 float4s][threads]: each thread's d in order, coalesced; distinct for every (block, unit).
            float4* pp = (float4*)(parts + ((int64_t)blockIdx.x + p) * (NT * C::R));
#pragma unroll
            for (int q = 0; q < NT / 8; q++) pp[q * 128 * WG + ct] = make_float4(r[4 * q], r[4 * q + 1], r[4 * q + 2], r[4 * q + 3]);
            __threadfence();
            bar_sync<128 * WG>(1);
            if (ct == 0) {
                last = atomicAdd(done + p, 1) == fin - first;
                if (last) done[p] = 0;  // ready for the next product
            }
            bar_sync<128 * WG>(1);
            if (last) {
                __threadfence();
#pragma unroll
                for (int i = 0; i < NT / 2; i++) r[i] = 0.f;
                for (int64_t b = first; b <= fin; b++) {
                    const float4* bp = (const float4*)(parts + (b + p) * (NT * C::R));
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
                    int64_t o = ((int64_t)WG * p + wg) * 64 + 16 * w + g + 8 * h;
                    float bo = bias && o < O ? __bfloat162float(bias[o]) : 0.f;
#pragma unroll
                    for (int c = 0; c < 2; c++) {
                        int64_t m = 8 * jn + 2 * t + c;
                        if (o < O && m < M) Y[m * O + o] = __float2bfloat16(r[4 * jn + 2 * h + c] + bo);
                    }
                }
        }
        j += len;
    }
#endif
}

// The mma layout back to bf16, rows [row0, row0 + rows) of W (multiples of
// 64) into out [rows, K] (checks, and the many-token path that multiplies
// with PyTorch): a warp a step.
template <class Fmt>
__global__ void mma_unpack_kernel(Fmt f, int64_t K, int64_t row0, int64_t rows, uint16_t* __restrict__ out) {
    __shared__ uint32_t tab[Fmt::kTable ? 256 : 1];
    if constexpr (Fmt::kTable) fill_groups(tab);
    __syncthreads();
    int64_t KS = K / 16, local = (blockIdx.x * (int64_t)blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    if (local >= rows / 64 * KS) return;
    int64_t step = row0 / 64 * KS + local, rb = local / KS, s = local % KS;
    __shared__ uint4 scratch[8][Fmt::kTable ? S2_BYTES / 16 : 1];
    typename Fmt::St st;
    f.load(st, step, lane);
    uint32_t R[16];
    f.decode(st, lane, (uint32_t*)scratch[threadIdx.x >> 5], tab, R);
    // R[2n]: row 8n + g, columns 2t and 2t + 1; R[2n + 1]: columns 8 + 2t, 9 + 2t.
    uint32_t* out32 = (uint32_t*)out;
#pragma unroll
    for (int n = 0; n < 8; n++) {
        int64_t at2 = ((rb * 64 + 8 * n + g) * K + s * 16 + 2 * t) / 2;
        out32[at2] = R[2 * n];
        out32[at2 + 4] = R[2 * n + 1];
    }
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

static int64_t tiles_for(int64_t n, int64_t tw) { return (n + tw - 1) / tw; }

#define BY_V(V, CALL) do { if ((V) == 16) { constexpr int VV = 16; CALL; } else { constexpr int VV = 4; CALL; } } while (0)

torch::Tensor lane_bits(torch::Tensor w, torch::Tensor len, int64_t tw, int64_t V) {
    const c10::cuda::CUDAGuard guard(w.device());
    cudaStream_t cs = at::cuda::getCurrentCUDAStream();
    int64_t n = w.numel(), lanes = tiles_for(n, tw) * 32;
    auto bits = torch::empty({lanes}, w.options().dtype(torch::kInt32));
    BY_V(V, (lane_bits_kernel<VV><<<(lanes + 255) / 256, 256, 0, cs>>>((const uint16_t*)w.data_ptr(), n, tw, (const uint8_t*)len.data_ptr(), (uint32_t*)bits.data_ptr(), lanes)));
    return bits;
}

void write_codes(torch::Tensor w, torch::Tensor len, torch::Tensor code, torch::Tensor offs, torch::Tensor out, int64_t tw, int64_t V) {
    const c10::cuda::CUDAGuard guard(w.device());
    cudaStream_t cs = at::cuda::getCurrentCUDAStream();
    int64_t n = w.numel(), lanes = tiles_for(n, tw) * 32;
    BY_V(V, (write_kernel<VV><<<(lanes + 255) / 256, 256, 0, cs>>>((const uint16_t*)w.data_ptr(), n, tw, (const uint8_t*)len.data_ptr(), (const uint32_t*)code.data_ptr(), (const uint32_t*)offs.data_ptr(), (uint32_t*)out.data_ptr(), lanes)));
}

static void allow_shared(const void* kernel) { cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024); }

void decode(torch::Tensor sm, torch::Tensor stream, torch::Tensor offs, torch::Tensor tables, int64_t n, int64_t tw, int64_t V, int64_t tile_words, torch::Tensor tile_ids, torch::Tensor out) {
    const c10::cuda::CUDAGuard guard(sm.device());
    cudaStream_t cs = at::cuda::getCurrentCUDAStream();
    bool all = tile_ids.numel() == 0;
    int64_t tiles = all ? tiles_for(n, tw) : tile_ids.numel();
    TORCH_CHECK(out.numel() >= (all ? n : tiles * tw), "the output is too small");
    int threads = 128;
    size_t shared = (threads / 32) * tile_words * sizeof(uint32_t);
    BY_V(V, (allow_shared((const void*)decode_kernel<VV>), decode_kernel<VV><<<(tiles * 32 + threads - 1) / threads, threads, shared, cs>>>((const uint8_t*)sm.data_ptr(), (const uint32_t*)stream.data_ptr(), stream.numel(), (const uint32_t*)offs.data_ptr(), (const uint32_t*)tables.data_ptr(), n, tw, (int)tile_words, all ? nullptr : (const int64_t*)tile_ids.data_ptr(), tiles, (uint16_t*)out.data_ptr())));
}

void gemv(torch::Tensor sm, torch::Tensor stream, torch::Tensor offs, torch::Tensor tables, int64_t O, int64_t K, int64_t tw, int64_t V, int64_t tile_words, torch::Tensor x, torch::Tensor bias, torch::Tensor y, torch::Tensor sum, torch::Tensor count) {
    const c10::cuda::CUDAGuard guard(sm.device());
    cudaStream_t cs = at::cuda::getCurrentCUDAStream();
    bool split = tw % K != 0;
    TORCH_CHECK(K % (32 * V) == 0 && tw % (32 * V) == 0 && (!split || sum.numel() >= O), "gemv needs K and tiles multiples of 32 V, and row sums for split rows");
    int64_t tiles = tiles_for(O * K, tw);
    int threads = 128;
    size_t shared = (threads / 32) * tile_words * sizeof(uint32_t);
    auto launch = [&](auto kernel) {
        allow_shared((const void*)kernel);
        kernel<<<(tiles * 32 + threads - 1) / threads, threads, shared, cs>>>((const uint8_t*)sm.data_ptr(), (const uint32_t*)stream.data_ptr(), stream.numel(), (const uint32_t*)offs.data_ptr(), (const uint32_t*)tables.data_ptr(), O, K, tw, (int)tile_words, (const __nv_bfloat16*)x.data_ptr(), bias.numel() ? (const __nv_bfloat16*)bias.data_ptr() : nullptr, (__nv_bfloat16*)y.data_ptr(), tiles, split ? (float*)sum.data_ptr() : nullptr, split ? (int*)count.data_ptr() : nullptr);
    };
    if (V == 16) { if (split) launch(gemv_kernel<16, true>); else launch(gemv_kernel<16, false>); }
    else { if (split) launch(gemv_kernel<4, true>); else launch(gemv_kernel<4, false>); }
}

void fast_gemv(torch::Tensor sm, torch::Tensor planes, torch::Tensor exc, torch::Tensor exc_base, int64_t top, int64_t O, int64_t K, torch::Tensor x, torch::Tensor bias, torch::Tensor y) {
    const c10::cuda::CUDAGuard guard(sm.device());
    cudaStream_t cs = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(K % 128 == 0, "rows a multiple of 128 long");
    // Warps a row: enough for the GPU to hold ~16k warps of work, at most
    // one a segment and 8 a row.
    int64_t segs = (K + SEG - 1) / SEG;
    int wpr = O >= 16384 ? 1 : (int)std::min<int64_t>(segs, 8);
    int rows_per_block = std::max(1, 8 / wpr), threads = rows_per_block * wpr * 32;
    auto kernel = wpr == 1 ? fast_gemv_kernel<false> : fast_gemv_kernel<true>;
    if (K % 512 == 0) kernel = wpr == 1 ? fast_gemv_wide_kernel<16, false> : fast_gemv_wide_kernel<16, true>;
    else if (K % 256 == 0) kernel = wpr == 1 ? fast_gemv_wide_kernel<8, false> : fast_gemv_wide_kernel<8, true>;
    kernel<<<(O + rows_per_block - 1) / rows_per_block, threads, 0, cs>>>((const uint8_t*)sm.data_ptr(), (const uint32_t*)planes.data_ptr(), (const uint8_t*)exc.data_ptr(), (const int32_t*)exc_base.data_ptr(), (uint64_t)top, O, K, wpr, (const __nv_bfloat16*)x.data_ptr(), bias.numel() ? (const __nv_bfloat16*)bias.data_ptr() : nullptr, (__nv_bfloat16*)y.data_ptr());
}

void fast_decode(torch::Tensor sm, torch::Tensor planes, torch::Tensor exc, torch::Tensor exc_base, int64_t top, int64_t row0, int64_t rows, torch::Tensor row_ids, int64_t K, torch::Tensor out) {
    const c10::cuda::CUDAGuard guard(sm.device());
    cudaStream_t cs = at::cuda::getCurrentCUDAStream();
    if (row_ids.numel()) rows = row_ids.numel();
    TORCH_CHECK(K % 128 == 0 && out.numel() >= rows * K, "rows a multiple of 128 long, room for them");
    int threads = 256;
    int64_t warps = rows * ((K + SEG - 1) / SEG);
    fast_decode_kernel<<<(warps * 32 + threads - 1) / threads, threads, 0, cs>>>((const uint8_t*)sm.data_ptr(), (const uint32_t*)planes.data_ptr(), (const uint8_t*)exc.data_ptr(), (const int32_t*)exc_base.data_ptr(), (uint64_t)top, row0, row_ids.numel() ? (const int64_t*)row_ids.data_ptr() : nullptr, rows, K, (uint16_t*)out.data_ptr());
}

void fast_gemm(torch::Tensor sm, torch::Tensor planes, torch::Tensor exc, torch::Tensor exc_base, int64_t top, int64_t O, int64_t K, torch::Tensor x, torch::Tensor bias, torch::Tensor y) {
    const c10::cuda::CUDAGuard guard(sm.device());
    cudaStream_t cs = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(K % GM_BK == 0 && x.is_contiguous() && x.size(1) == K, "K a multiple of 64, X contiguous [M, K]");
    int64_t M = x.size(0);
    int64_t bo = (O + GM_BO - 1) / GM_BO, bm = (M + GM_BM - 1) / GM_BM;
    // Split K until the GPU has some `target` blocks, in whole steps.
    static int64_t target = getenv("GLYD_GPU_GEMM_BLOCKS") ? atoll(getenv("GLYD_GPU_GEMM_BLOCKS")) : 1280;
    int64_t split = std::max<int64_t>(1, std::min<int64_t>(K / (4 * GM_BK), target / (bo * bm)));
    int64_t kchunk = (K / GM_BK + split - 1) / split * GM_BK;
    split = (K + kchunk - 1) / kchunk;
    const __nv_bfloat16* b = bias.numel() ? (const __nv_bfloat16*)bias.data_ptr() : nullptr;
    torch::Tensor y32;
    if (split > 1) y32 = torch::empty({split, M, O}, x.options().dtype(torch::kFloat32));
    fast_gemm_kernel<<<dim3(bo, bm, split), 128, 0, cs>>>((const uint8_t*)sm.data_ptr(), (const uint32_t*)planes.data_ptr(), (const uint8_t*)exc.data_ptr(), (const int32_t*)exc_base.data_ptr(), (uint64_t)top, O, K, M, kchunk, (const __nv_bfloat16*)x.data_ptr(), b, (__nv_bfloat16*)y.data_ptr(), split > 1 ? (float*)y32.data_ptr() : nullptr);
    if (split > 1) finish_kernel<<<(M * O + 255) / 256, 256, 0, cs>>>((const float*)y32.data_ptr(), split, b, M, O, (__nv_bfloat16*)y.data_ptr());
}

void fast_bgemv(torch::Tensor sm, torch::Tensor planes, torch::Tensor exc, torch::Tensor exc_base, int64_t top, int64_t O, int64_t K, torch::Tensor x, torch::Tensor bias, torch::Tensor y) {
    const c10::cuda::CUDAGuard guard(sm.device());
    cudaStream_t cs = at::cuda::getCurrentCUDAStream();
    int64_t M = x.size(0);
    TORCH_CHECK(K % 512 == 0 && x.is_contiguous() && x.size(1) == K && (M == 2 || M == 4 || M == 8 || M == 16), "K a multiple of 512, X contiguous [M, K], M 2, 4, 8 or 16");
    // X's K segment in shared memory: up to 48 KB, whole segments of the escapes' index.
    int64_t kseg = std::min<int64_t>(K, std::max<int64_t>(SEG, (48 * 1024 / (2 * M)) / SEG * SEG));
    int64_t nseg = (K + kseg - 1) / kseg;
    // Rows a block: enough blocks for every SM (4 an SM), at least 64 rows each.
    int64_t blocks = std::max<int64_t>(1, 320 / nseg);
    int64_t rows = std::max<int64_t>(64, (O + blocks - 1) / blocks);
    dim3 grid((O + rows - 1) / rows, nseg);
    size_t shared = M * kseg * 2;
    const __nv_bfloat16* b = bias.numel() ? (const __nv_bfloat16*)bias.data_ptr() : nullptr;
    torch::Tensor y32;
    if (nseg > 1) y32 = torch::empty({nseg, M, O}, x.options().dtype(torch::kFloat32));
    float* p32 = nseg > 1 ? (float*)y32.data_ptr() : nullptr;
    auto launch = [&](auto kernel) {
        cudaFuncSetAttribute((const void*)kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
        kernel<<<grid, 256, shared, cs>>>((const uint8_t*)sm.data_ptr(), (const uint32_t*)planes.data_ptr(), (const uint8_t*)exc.data_ptr(), (const int32_t*)exc_base.data_ptr(), (uint64_t)top, O, K, kseg, rows, (const __nv_bfloat16*)x.data_ptr(), b, (__nv_bfloat16*)y.data_ptr(), p32);
    };
    if (M == 2) launch(fast_bgemv_kernel<2>);
    else if (M == 4) launch(fast_bgemv_kernel<4>);
    else if (M == 8) launch(fast_bgemv_kernel<8>);
    else launch(fast_bgemv_kernel<16>);
    if (nseg > 1) finish_kernel<<<(M * O + 255) / 256, 256, 0, cs>>>((const float*)y32.data_ptr(), nseg, b, M, O, (__nv_bfloat16*)y.data_ptr());
}

static Tiers tiers_of(const std::vector<int64_t>& t) {
    TORCH_CHECK(t.size() == 3, "three tiers");
    return Tiers{(uint32_t)t[0], (uint32_t)t[1], (uint32_t)t[2]};
}

static Nib nib_of(torch::Tensor data, torch::Tensor exc, torch::Tensor exc_base, const std::vector<int64_t>& sym) {
    TORCH_CHECK(sym.size() == 4, "four words of symbols");
    return Nib{(const uint8_t*)data.data_ptr(), (const uint32_t*)exc.data_ptr(), (const int32_t*)exc_base.data_ptr(), {(uint32_t)sym[0], (uint32_t)sym[1], (uint32_t)sym[2], (uint32_t)sym[3]}};
}

static Tiered tiered_of(torch::Tensor data, torch::Tensor blocks, torch::Tensor block_base, const std::vector<int64_t>& tiers) {
    return Tiered{(const uint8_t*)data.data_ptr(), (const uint8_t*)blocks.data_ptr(), (const int32_t*)block_base.data_ptr(), tiers_of(tiers)};
}

template <class Fmt>
static void mma_gemm_run(Fmt f, torch::Tensor data, int64_t O, int64_t K, torch::Tensor x, torch::Tensor bias, torch::Tensor y) {
    const c10::cuda::CUDAGuard guard(data.device());
    cudaStream_t cs = at::cuda::getCurrentCUDAStream();
    int64_t M = x.size(0);
    TORCH_CHECK(O % 64 == 0 && K % 16 == 0 && M <= 64 && x.is_contiguous() && x.size(1) == K, "O a multiple of 64, K of 16, up to 64 tokens, X contiguous [M, K]");
    int64_t KS = K / 16, RB = O / 64, total = RB * KS;
    // Row blocks' done counts: zero between products (the last block of a row block resets it).
    static auto* done_of = new std::map<int, torch::Tensor>;  // kept to the end (no teardown after CUDA's)
    torch::Tensor& done = (*done_of)[data.get_device()];
    if (!done.defined() || done.numel() < RB) done = torch::zeros({std::max<int64_t>(RB, 1 << 16)}, data.options().dtype(torch::kInt32));
    const __nv_bfloat16* b = bias.numel() ? (const __nv_bfloat16*)bias.data_ptr() : nullptr;
    auto launch = [&](auto kernel, int mt) {
        size_t shared = (size_t)4 * mt * 16 * 65 * sizeof(float) + (Fmt::kTable ? 8 * S2_BYTES : 0);
        cudaFuncSetAttribute((const void*)kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared);
        int per_sm = 1;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, 256, shared);
        // As many blocks as fit at once, but at least 32 steps (4 a warp) a block.
        int64_t nb = std::max<int64_t>(1, std::min<int64_t>((int64_t)std::max(per_sm, 1) * at::cuda::getCurrentDeviceProperties()->multiProcessorCount, total / 32));
        torch::Tensor parts = torch::empty({nb + RB, M, 64}, x.options().dtype(torch::kFloat32));
        kernel<<<nb, 256, shared, cs>>>(f, O, K, M, (const __nv_bfloat16*)x.data_ptr(), b, (__nv_bfloat16*)y.data_ptr(), (float*)parts.data_ptr(), (int*)done.data_ptr());
    };
    if (M <= 16) launch(mma_gemm_kernel<Fmt, 1>, 1);
    else if (M <= 32) launch(mma_gemm_kernel<Fmt, 2>, 2);
    else launch(mma_gemm_kernel<Fmt, 4>, 4);
}

void mma_gemm(torch::Tensor data, torch::Tensor blocks, torch::Tensor block_base, std::vector<int64_t> tiers, int64_t O, int64_t K, torch::Tensor x, torch::Tensor bias, torch::Tensor y) {
    mma_gemm_run(tiered_of(data, blocks, block_base, tiers), data, O, K, x, bias, y);
}

void mma12_gemm(torch::Tensor data, torch::Tensor exc, torch::Tensor exc_base, std::vector<int64_t> sym, int64_t O, int64_t K, torch::Tensor x, torch::Tensor bias, torch::Tensor y) {
    mma_gemm_run(nib_of(data, exc, exc_base, sym), data, O, K, x, bias, y);
}

template <class Fmt, int CW, int PW, int NB, int RBB>
static void mma_gemm_big_run(Fmt f, torch::Tensor data, int64_t O, int64_t K, torch::Tensor x, torch::Tensor bias, torch::Tensor y, cudaStream_t cs) {
    using C = Big<CW, PW, NB, RBB>;
    auto kernel = mma_gemm_big_kernel<Fmt, CW, PW, NB, RBB>;
    int64_t M = x.size(0), stages = K / 16 / BIG_KK, pairs = (O / 64 + RBB - 1) / RBB, tiles = (M + C::TM - 1) / C::TM;
    static auto* per_sm_of = new std::map<int, int>;  // a device's blocks an SM for this kernel (its shared memory allowed once)
    int dev = data.get_device();
    if (!per_sm_of->count(dev)) {
        int n = 1;
        cudaFuncSetAttribute((const void*)kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SHARED);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kernel, C::THREADS, C::SHARED);
        (*per_sm_of)[dev] = std::max(n, 1);
    }
    // K split so the blocks fill their last wave: the fewest splits that
    // leave under 15% of it idle, else the fullest; at least 4 stages a split.
    int64_t cap = (*per_sm_of)[dev] * at::cuda::getCurrentDeviceProperties()->multiProcessorCount, most = std::max<int64_t>(1, stages / 4);
    int64_t splits = 1;
    double best = 0;
    for (int64_t sp = 1; sp <= most; sp++) {
        int64_t nblocks = pairs * tiles * sp;
        double fill = (double)nblocks / (double)(((nblocks + cap - 1) / cap) * cap);
        if (fill > best + 1e-9) best = fill, splits = sp;
        if (fill >= 0.85) break;
    }
    int64_t per = (stages + splits - 1) / splits;
    splits = (stages + per - 1) / per;
    const __nv_bfloat16* b = bias.numel() ? (const __nv_bfloat16*)bias.data_ptr() : nullptr;
    torch::Tensor y32;
    if (splits > 1) y32 = torch::empty({splits, M, O}, x.options().dtype(torch::kFloat32));
    kernel<<<dim3(tiles, pairs, splits), C::THREADS, C::SHARED, cs>>>(f, O, K, M, per, (const __nv_bfloat16*)x.data_ptr(), b, (__nv_bfloat16*)y.data_ptr(), splits > 1 ? (float*)y32.data_ptr() : nullptr);
    if (splits > 1) finish_kernel<<<(M * O + 255) / 256, 256, 0, cs>>>((const float*)y32.data_ptr(), splits, b, M, O, (__nv_bfloat16*)y.data_ptr());
}

template <class Fmt>
static void mma_gemm_big_any(Fmt f, torch::Tensor data, int64_t O, int64_t K, torch::Tensor x, torch::Tensor bias, torch::Tensor y, int64_t variant) {
    const c10::cuda::CUDAGuard guard(data.device());
    cudaStream_t cs = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(O % 64 == 0 && K % 64 == 0 && x.is_contiguous() && x.size(1) == K && (uintptr_t)x.data_ptr() % 16 == 0, "O a multiple of 64, K of 64, X contiguous [M, K]");
    // 4 consumer warps and 4 producers: blocks of 128 tokens by two row blocks (3 stages), or, past 128
    // tokens, of 256 by one (2 stages; a weight decoded once for twice the tokens).
    if (variant == 0) variant = x.size(0) > 128 ? 2 : 1;
    if (variant == 2) mma_gemm_big_run<Fmt, 4, 4, 2, 1>(f, data, O, K, x, bias, y, cs);
    else mma_gemm_big_run<Fmt, 4, 4, 3, 2>(f, data, O, K, x, bias, y, cs);
}

void mma_gemm_big(torch::Tensor data, torch::Tensor blocks, torch::Tensor block_base, std::vector<int64_t> tiers, int64_t O, int64_t K, torch::Tensor x, torch::Tensor bias, torch::Tensor y, int64_t variant) {
    mma_gemm_big_any(tiered_of(data, blocks, block_base, tiers), data, O, K, x, bias, y, variant);
}

void mma12_gemm_big(torch::Tensor data, torch::Tensor exc, torch::Tensor exc_base, std::vector<int64_t> sym, int64_t O, int64_t K, torch::Tensor x, torch::Tensor bias, torch::Tensor y, int64_t variant) {
    mma_gemm_big_any(nib_of(data, exc, exc_base, sym), data, O, K, x, bias, y, variant);
}

// cuTensorMapEncodeTiled from the driver, found at run time (no link to libcuda).
static PFN_cuTensorMapEncodeTiled_v12000 tensor_map_encoder() {
    static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
    if (!fn) {
        void* p = nullptr;
        cudaDriverEntryPointQueryResult q;
#if CUDART_VERSION >= 12050
        cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q);
#else
        cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q);
#endif
        TORCH_CHECK(p && q == cudaDriverEntryPointSuccess, "cuTensorMapEncodeTiled: not in this driver");
        fn = (PFN_cuTensorMapEncodeTiled_v12000)p;
    }
    return fn;
}

template <int NT, int WG>
static void mma12_tma_run(Nib f, torch::Tensor data, int64_t O, int64_t K, const __nv_bfloat16* x, int64_t M, const __nv_bfloat16* b, __nv_bfloat16* y, cudaStream_t cs) {
    using C = Tma12<NT, WG>;
    auto kernel = mma12_tma_kernel<NT, WG>;
    int dev = data.get_device();
    static auto* per_sm_of = new std::map<int, int>;  // a device's blocks an SM for this kernel
    if (!per_sm_of->count(dev)) {
        cudaFuncSetAttribute((const void*)kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SHARED);
        int n = 1;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kernel, C::THREADS, C::SHARED);
        (*per_sm_of)[dev] = std::max(n, 1);
    }
    int64_t P = (O / 64 + WG - 1) / WG, U = P * (K / 64);
    // Units' done counts: zero between products (the last block of a unit resets it).
    static auto* done_of = new std::map<int, torch::Tensor>;
    torch::Tensor& done = (*done_of)[dev];
    if (!done.defined() || done.numel() < P) done = torch::zeros({std::max<int64_t>(P, 1 << 16)}, data.options().dtype(torch::kInt32));
    // As many blocks as fit at once, but at least 8 stages a block.
    int64_t nb = std::max<int64_t>(1, std::min<int64_t>((int64_t)(*per_sm_of)[dev] * at::cuda::getCurrentDeviceProperties()->multiProcessorCount, U / 8));
    torch::Tensor parts = torch::empty({nb + P, NT * C::R}, data.options().dtype(torch::kFloat32));
    // X [M, K] as tiles of NT tokens by 64 columns, 128-byte swizzle; rows past M read as zeros.
    CUtensorMap map;
    cuuint64_t dims[2] = {(cuuint64_t)K, (cuuint64_t)M}, strides[1] = {(cuuint64_t)K * 2};
    cuuint32_t box[2] = {64, NT}, unit[2] = {1, 1};
    CUresult r = tensor_map_encoder()(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)x, dims, strides, box, unit, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "X's tensor map: error ", (int)r);
    kernel<<<nb, C::THREADS, C::SHARED, cs>>>(map, f, O, K, M, b, y, (float*)parts.data_ptr(), (int*)done.data_ptr());
}

void mma12_gemm_wg(torch::Tensor data, torch::Tensor exc, torch::Tensor exc_base, std::vector<int64_t> sym, int64_t O, int64_t K, torch::Tensor x, torch::Tensor bias, torch::Tensor y) {
    const c10::cuda::CUDAGuard guard(data.device());
    cudaStream_t cs = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major >= 9, "wgmma: Hopper or later");
    TORCH_CHECK(O % 64 == 0 && K % 64 == 0 && x.is_contiguous() && x.size(1) == K && (uintptr_t)x.data_ptr() % 16 == 0, "O a multiple of 64, K of 64, X contiguous [M, K]");
    TORCH_CHECK((uintptr_t)data.data_ptr() % 16 == 0 && (uintptr_t)exc.data_ptr() % 16 == 0 && exc.numel() % 4 == 0, "the pack 16-byte aligned, exc padded to 4 (pack_mma12)");
    Nib f = nib_of(data, exc, exc_base, sym);
    const __nv_bfloat16* b = bias.numel() ? (const __nv_bfloat16*)bias.data_ptr() : nullptr;
    const __nv_bfloat16* xp = (const __nv_bfloat16*)x.data_ptr();
    __nv_bfloat16* yp = (__nv_bfloat16*)y.data_ptr();
    // 256 tokens at a time, in the smallest tile that holds them.
    for (int64_t m0 = 0, M = x.size(0); m0 < M; m0 += 256) {
        int64_t mc = std::min<int64_t>(256, M - m0);
        // Four warpgroups (256 rows a stage) to 64 tokens; past that two, for their registers.
        auto run = mc <= 16 ? mma12_tma_run<16, 4> : mc <= 32 ? mma12_tma_run<32, 4> : mc <= 64 ? mma12_tma_run<64, 4> : mc <= 128 ? mma12_tma_run<128, 2> : mma12_tma_run<256, 2>;
        run(f, data, O, K, xp + m0 * K, mc, b, yp + m0 * O, cs);
    }
}

template <class Fmt>
static void mma_unpack_run(Fmt f, torch::Tensor data, int64_t K, int64_t row0, int64_t rows, torch::Tensor out) {
    const c10::cuda::CUDAGuard guard(data.device());
    cudaStream_t cs = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(row0 % 64 == 0 && rows % 64 == 0 && out.numel() >= rows * K, "rows a multiple of 64");
    int64_t steps = rows / 64 * (K / 16);
    mma_unpack_kernel<Fmt><<<(steps * 32 + 255) / 256, 256, 0, cs>>>(f, K, row0, rows, (uint16_t*)out.data_ptr());
}

void mma_unpack(torch::Tensor data, torch::Tensor blocks, torch::Tensor block_base, std::vector<int64_t> tiers, int64_t K, int64_t row0, int64_t rows, torch::Tensor out) {
    mma_unpack_run(tiered_of(data, blocks, block_base, tiers), data, K, row0, rows, out);
}

void mma12_unpack(torch::Tensor data, torch::Tensor exc, torch::Tensor exc_base, std::vector<int64_t> sym, int64_t K, int64_t row0, int64_t rows, torch::Tensor out) {
    mma_unpack_run(nib_of(data, exc, exc_base, sym), data, K, row0, rows, out);
}

void attn_decode(torch::Tensor q, torch::Tensor kd, torch::Tensor kb, torch::Tensor kbb, std::vector<int64_t> kt, torch::Tensor vd, torch::Tensor vb, torch::Tensor vbb, std::vector<int64_t> vt, torch::Tensor tk, torch::Tensor tv, int64_t tlen, int64_t pairs, int64_t G, int64_t P, double scale, torch::Tensor out) {
    const c10::cuda::CUDAGuard guard(q.device());
    cudaStream_t cs = at::cuda::getCurrentCUDAStream();
    int64_t D = q.size(-1);
    TORCH_CHECK((D == 64 || D == 128) && G >= 1 && G <= 16 && q.is_contiguous() && tlen < 64 && P + (tlen > 0) > 0, "head_dim 64 or 128, up to 16 queries a KV head");
    static auto* done_of = new std::map<int, torch::Tensor>;  // a pair's finished blocks: zero between calls
    torch::Tensor& done = (*done_of)[q.get_device()];
    if (!done.defined() || done.numel() < pairs) done = torch::zeros({std::max<int64_t>(pairs, 1 << 12)}, q.options().dtype(torch::kInt32));
    // Pages a block: enough blocks for four an SM, at least a page a warp.
    int64_t units = P + (tlen > 0), sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    int64_t per = std::max<int64_t>(ATT_WARPS, (pairs * units + 4 * sms - 1) / (4 * sms));
    per = (per + ATT_WARPS - 1) / ATT_WARPS * ATT_WARPS;
    int64_t splits = (units + per - 1) / per;
    torch::Tensor part = torch::empty({pairs * splits * 16 * (D + 2)}, q.options().dtype(torch::kFloat32));
    float sl2 = (float)(scale * 1.4426950408889634);
    const __nv_bfloat16* tkp = tlen ? (const __nv_bfloat16*)tk.data_ptr() : nullptr;
    const __nv_bfloat16* tvp = tlen ? (const __nv_bfloat16*)tv.data_ptr() : nullptr;
    auto launch = [&](auto kernel) {
        kernel<<<dim3(pairs, splits), 32 * ATT_WARPS, 0, cs>>>((const __nv_bfloat16*)q.data_ptr(), (const uint8_t*)kd.data_ptr(), (const uint8_t*)kb.data_ptr(), (const int32_t*)kbb.data_ptr(), tiers_of(kt), (const uint8_t*)vd.data_ptr(), (const uint8_t*)vb.data_ptr(), (const int32_t*)vbb.data_ptr(), tiers_of(vt), tkp, tvp, (int)tlen, (int)pairs, (int)G, (int)P, (int)per, sl2, (float*)part.data_ptr(), (int*)done.data_ptr(), (__nv_bfloat16*)out.data_ptr());
    };
    if (D == 128) launch(attn_decode_kernel<128>);
    else launch(attn_decode_kernel<64>);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mma_gemm", &mma_gemm);
    m.def("mma_unpack", &mma_unpack);
    m.def("mma12_gemm", &mma12_gemm);
    m.def("mma12_gemm_big", &mma12_gemm_big);
    m.def("mma12_unpack", &mma12_unpack);
    m.def("mma12_gemm_wg", &mma12_gemm_wg);
    m.def("attn_decode", &attn_decode);
    m.def("mma_gemm_big", &mma_gemm_big);
    m.def("fast_bgemv", &fast_bgemv);
    m.def("fast_gemm", &fast_gemm);
    m.def("lane_bits", &lane_bits);
    m.def("write_codes", &write_codes);
    m.def("decode", &decode);
    m.def("gemv", &gemv);
    m.def("fast_gemv", &fast_gemv);
    m.def("fast_decode", &fast_decode);
}
