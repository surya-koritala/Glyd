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
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

__device__ __forceinline__ uint32_t exponent_of(uint16_t v) { return (v >> 7) & 0xff; }

// Pass 1: the bits each lane's stream takes.
__global__ void lane_bits_kernel(const uint16_t* __restrict__ w, int64_t n, int64_t tw, const uint8_t* __restrict__ len, uint32_t* __restrict__ bits, int64_t lanes) {
    int64_t g = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (g >= lanes) return;
    int64_t base = (g >> 5) * tw + (g & 31) * 4, end = min((g >> 5) * tw + tw, n);
    uint32_t b = 0;
    for (int64_t q = base; q < end; q += 128)
        for (int k = 0; k < 4 && q + k < end; k++) b += len[exponent_of(w[q + k])];
    bits[g] = b;
}

// Pass 2: every lane writes its stream from its offset. A lane's first
// and last words may be shared with its neighbours (OR'd in); the words
// between are its own.
__global__ void write_kernel(const uint16_t* __restrict__ w, int64_t n, int64_t tw, const uint8_t* __restrict__ len, const uint16_t* __restrict__ code, const uint32_t* __restrict__ offs, uint32_t* __restrict__ out, int64_t lanes) {
    int64_t g = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (g >= lanes) return;
    int64_t base = (g >> 5) * tw + (g & 31) * 4, end = min((g >> 5) * tw + tw, n);
    uint32_t pos = offs[g];
    uint32_t wi = pos >> 5;
    int nb = pos & 31;
    uint64_t acc = 0;
    bool first = true;
    for (int64_t q = base; q < end; q += 128) {
        for (int k = 0; k < 4 && q + k < end; k++) {
            uint32_t e = exponent_of(w[q + k]);
            acc |= (uint64_t)code[e] << nb;
            nb += len[e];
            if (nb >= 32) {
                if (first) atomicOr(&out[wi], (uint32_t)acc); else out[wi] = (uint32_t)acc;
                first = false;
                acc >>= 32;
                nb -= 32;
                wi++;
            }
        }
    }
    if (nb > 0) atomicOr(&out[wi], (uint32_t)acc);
}

// A lane's reader: its stream through a 64-bit window, the code table in
// shared memory.
struct Reader {
    const uint32_t* stream;
    uint64_t buf;
    uint32_t wi;
    int nb;
    __device__ __forceinline__ Reader(const uint32_t* s, uint32_t pos) : stream(s) {
        wi = pos >> 5;
        int sh = pos & 31;
        buf = (((uint64_t)s[wi + 1] << 32) | s[wi]) >> sh;
        nb = 64 - sh;
        wi += 2;
    }
    // Four exponents; two codes of at most 12 bits between refills (a
    // refill leaves at least 33 bits in the window).
    __device__ __forceinline__ void quad(const uint16_t* table, uint32_t mask, uint32_t e[4]) {
#pragma unroll
        for (int k = 0; k < 4; k++) {
            if ((k & 1) == 0 && nb <= 32) {
                buf |= (uint64_t)stream[wi++] << nb;
                nb += 32;
            }
            uint16_t t = table[buf & mask];
            e[k] = t & 0xff;
            buf >>= t >> 8;
            nb -= t >> 8;
        }
    }
};

__device__ __forceinline__ uint32_t bf16_bits(uint32_t s, uint32_t e) { return ((s & 0x80) << 8) | (e << 7) | (s & 0x7f); }

__device__ __forceinline__ void load_table(uint16_t* table, const uint16_t* lut, int L) {
    for (int i = threadIdx.x; i < (1 << L); i += blockDim.x) table[i] = lut[i];
    __syncthreads();
}

// Decode every tile into `out`, or the tiles listed in `tile_ids`, one
// after another from out[0].
__global__ void decode_kernel(const uint8_t* __restrict__ sm, const uint32_t* __restrict__ stream, const uint32_t* __restrict__ offs, const uint16_t* __restrict__ lut, int L, int64_t n, int64_t tw, const int64_t* __restrict__ tile_ids, int64_t lanes, uint16_t* __restrict__ out) {
    extern __shared__ uint16_t table[];
    load_table(table, lut, L);
    int64_t g = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (g >= lanes) return;
    int64_t tile = tile_ids ? tile_ids[g >> 5] : (g >> 5);
    int64_t start = tile * tw, end = min(start + tw, n);
    int64_t dst = tile_ids ? (g >> 5) * tw - start : 0;
    Reader r(stream, offs[tile * 32 + (g & 31)]);
    const uint32_t mask = (1u << L) - 1;
    for (int64_t q = start + (g & 31) * 4; q < end; q += 128) {
        uint32_t e[4];
        r.quad(table, mask, e);
        if (q + 4 <= end) {
            uint32_t s4 = *(const uint32_t*)(sm + q);
            uint32_t o0 = bf16_bits(s4 & 0xff, e[0]), o1 = bf16_bits((s4 >> 8) & 0xff, e[1]);
            uint32_t o2 = bf16_bits((s4 >> 16) & 0xff, e[2]), o3 = bf16_bits(s4 >> 24, e[3]);
            *(uint2*)(out + q + dst) = make_uint2(o0 | (o1 << 16), o2 | (o3 << 16));
        } else {
            for (int k = 0; q + k < end; k++) out[q + k + dst] = (uint16_t)bf16_bits(sm[q + k], e[k]);
        }
    }
}

// y = W x (+ bias) for a matrix [O, K] packed in row tiles (tw = T * K,
// K a multiple of 128): one warp a tile, its rows' dot products in fp32,
// reduced across the warp as each row ends. The weights are read packed
// and never written out. The warp first copies its tile's exponent
// streams (contiguous, at most `tile_words` words) into shared memory in
// one coalesced burst: read from there, a lane's refills wait on no
// global load.
__global__ void gemv_kernel(const uint8_t* __restrict__ sm, const uint32_t* __restrict__ stream, int64_t stream_words, const uint32_t* __restrict__ offs, const uint16_t* __restrict__ lut, int L, int64_t O, int64_t K, int64_t tw, int tile_words, const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ y, int64_t tiles) {
    extern __shared__ uint32_t shared[];
    uint16_t* table = (uint16_t*)shared;
    load_table(table, lut, L);
    int64_t warp = (blockIdx.x * (int64_t)blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= tiles) return;
    uint32_t* words = shared + (1 << L) / 2 + (threadIdx.x >> 5) * tile_words;
    int64_t w0 = offs[warp * 32] >> 5;
    for (int i = lane; i < tile_words && w0 + i < stream_words; i += 32) words[i] = stream[w0 + i];
    __syncwarp();
    int64_t start = warp * tw, end = min(start + tw, O * K);
    Reader r(words, offs[warp * 32 + lane] - (uint32_t)(w0 << 5));
    const uint32_t mask = (1u << L) - 1;
    float acc = 0.f;
    int64_t row = start / K;
    int64_t row_end = (row + 1) * K;
    // A step's sign-and-mantissa bytes and inputs are loaded a step ahead,
    // so their latency hides behind the step before.
    int64_t k = lane * 4;
    uint32_t s4n = *(const uint32_t*)(sm + start + lane * 4);
    uint2 xn = *(const uint2*)(x + k);
    // Every lane is at the same step: the warp's 128 weights of a step lie
    // in one row (K is a multiple of 128).
    for (int64_t b = start; b < end; b += 128) {
        if (b >= row_end) {
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, o);
            if (lane == 0) y[row] = __float2bfloat16(acc + (bias ? __bfloat162float(bias[row]) : 0.f));
            acc = 0.f;
            row++;
            row_end += K;
        }
        uint32_t s4 = s4n;
        uint2 xv = xn;
        k += 128;
        if (k >= K) k -= K;
        if (b + 128 < end) {
            s4n = *(const uint32_t*)(sm + b + 128 + lane * 4);
            xn = *(const uint2*)(x + k);
        }
        uint32_t e[4];
        r.quad(table, mask, e);
        __nv_bfloat162 x01 = *(__nv_bfloat162*)&xv.x, x23 = *(__nv_bfloat162*)&xv.y;
        float w0 = __uint_as_float(bf16_bits(s4 & 0xff, e[0]) << 16), w1 = __uint_as_float(bf16_bits((s4 >> 8) & 0xff, e[1]) << 16);
        float w2 = __uint_as_float(bf16_bits((s4 >> 16) & 0xff, e[2]) << 16), w3 = __uint_as_float(bf16_bits(s4 >> 24, e[3]) << 16);
        acc += w0 * __low2float(x01) + w1 * __high2float(x01) + w2 * __low2float(x23) + w3 * __high2float(x23);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, o);
    if (lane == 0) y[row] = __float2bfloat16(acc + (bias ? __bfloat162float(bias[row]) : 0.f));
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

void fast_gemv(torch::Tensor sm, torch::Tensor planes, torch::Tensor exc, torch::Tensor exc_base, int64_t top, int64_t O, int64_t K, torch::Tensor x, torch::Tensor bias, torch::Tensor y) {
    TORCH_CHECK(K % 128 == 0, "rows a multiple of 128 long");
    // Warps a row: enough for the GPU to hold ~16k warps of work, at most
    // one a segment and 8 a row.
    int64_t segs = (K + SEG - 1) / SEG;
    int wpr = O >= 16384 ? 1 : (int)std::min<int64_t>(segs, 8);
    int rows_per_block = std::max(1, 8 / wpr), threads = rows_per_block * wpr * 32;
    auto kernel = wpr == 1 ? fast_gemv_kernel<false> : fast_gemv_kernel<true>;
    if (K % 512 == 0) kernel = wpr == 1 ? fast_gemv_wide_kernel<16, false> : fast_gemv_wide_kernel<16, true>;
    else if (K % 256 == 0) kernel = wpr == 1 ? fast_gemv_wide_kernel<8, false> : fast_gemv_wide_kernel<8, true>;
    kernel<<<(O + rows_per_block - 1) / rows_per_block, threads>>>((const uint8_t*)sm.data_ptr(), (const uint32_t*)planes.data_ptr(), (const uint8_t*)exc.data_ptr(), (const int32_t*)exc_base.data_ptr(), (uint64_t)top, O, K, wpr, (const __nv_bfloat16*)x.data_ptr(), bias.numel() ? (const __nv_bfloat16*)bias.data_ptr() : nullptr, (__nv_bfloat16*)y.data_ptr());
}

void fast_decode(torch::Tensor sm, torch::Tensor planes, torch::Tensor exc, torch::Tensor exc_base, int64_t top, int64_t row0, int64_t rows, torch::Tensor row_ids, int64_t K, torch::Tensor out) {
    if (row_ids.numel()) rows = row_ids.numel();
    TORCH_CHECK(K % 128 == 0 && out.numel() >= rows * K, "rows a multiple of 128 long, room for them");
    int threads = 256;
    int64_t warps = rows * ((K + SEG - 1) / SEG);
    fast_decode_kernel<<<(warps * 32 + threads - 1) / threads, threads>>>((const uint8_t*)sm.data_ptr(), (const uint32_t*)planes.data_ptr(), (const uint8_t*)exc.data_ptr(), (const int32_t*)exc_base.data_ptr(), (uint64_t)top, row0, row_ids.numel() ? (const int64_t*)row_ids.data_ptr() : nullptr, rows, K, (uint16_t*)out.data_ptr());
}

torch::Tensor lane_bits(torch::Tensor w, torch::Tensor len, int64_t tw) {
    int64_t n = w.numel(), lanes = tiles_for(n, tw) * 32;
    auto bits = torch::empty({lanes}, w.options().dtype(torch::kInt32));
    lane_bits_kernel<<<(lanes + 255) / 256, 256>>>((const uint16_t*)w.data_ptr(), n, tw, (const uint8_t*)len.data_ptr(), (uint32_t*)bits.data_ptr(), lanes);
    return bits;
}

void write_codes(torch::Tensor w, torch::Tensor len, torch::Tensor code, torch::Tensor offs, torch::Tensor out, int64_t tw) {
    int64_t n = w.numel(), lanes = tiles_for(n, tw) * 32;
    write_kernel<<<(lanes + 255) / 256, 256>>>((const uint16_t*)w.data_ptr(), n, tw, (const uint8_t*)len.data_ptr(), (const uint16_t*)code.data_ptr(), (const uint32_t*)offs.data_ptr(), (uint32_t*)out.data_ptr(), lanes);
}

void decode(torch::Tensor sm, torch::Tensor stream, torch::Tensor offs, torch::Tensor lut, int64_t L, int64_t n, int64_t tw, torch::Tensor tile_ids, torch::Tensor out) {
    bool all = tile_ids.numel() == 0;
    int64_t tiles = all ? tiles_for(n, tw) : tile_ids.numel();
    TORCH_CHECK(out.numel() >= (all ? n : tiles * tw), "the output is too small");
    int64_t lanes = tiles * 32;
    int threads = 128;
    decode_kernel<<<(lanes + threads - 1) / threads, threads, (1 << L) * sizeof(uint16_t)>>>((const uint8_t*)sm.data_ptr(), (const uint32_t*)stream.data_ptr(), (const uint32_t*)offs.data_ptr(), (const uint16_t*)lut.data_ptr(), (int)L, n, tw, all ? nullptr : (const int64_t*)tile_ids.data_ptr(), lanes, (uint16_t*)out.data_ptr());
}

void gemv(torch::Tensor sm, torch::Tensor stream, torch::Tensor offs, torch::Tensor lut, int64_t L, int64_t O, int64_t K, int64_t tw, int64_t tile_words, torch::Tensor x, torch::Tensor bias, torch::Tensor y) {
    TORCH_CHECK(K % 128 == 0 && tw % K == 0, "gemv needs row tiles of K a multiple of 128");
    int64_t tiles = tiles_for(O * K, tw);
    int threads = 256;
    size_t shared = (1 << L) * sizeof(uint16_t) + (threads / 32) * tile_words * sizeof(uint32_t);
    static bool raised = false;
    if (!raised) {
        cudaFuncSetAttribute(gemv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
        raised = true;
    }
    gemv_kernel<<<(tiles * 32 + threads - 1) / threads, threads, shared>>>((const uint8_t*)sm.data_ptr(), (const uint32_t*)stream.data_ptr(), stream.numel(), (const uint32_t*)offs.data_ptr(), (const uint16_t*)lut.data_ptr(), (int)L, O, K, tw, (int)tile_words, (const __nv_bfloat16*)x.data_ptr(), bias.numel() ? (const __nv_bfloat16*)bias.data_ptr() : nullptr, (__nv_bfloat16*)y.data_ptr(), tiles);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lane_bits", &lane_bits);
    m.def("write_codes", &write_codes);
    m.def("decode", &decode);
    m.def("gemv", &gemv);
    m.def("fast_gemv", &fast_gemv);
    m.def("fast_decode", &fast_decode);
}
