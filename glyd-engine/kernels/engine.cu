// glyd-engine's own small kernels: the pieces of a Qwen3 layer that are not a packed matrix product or the KV
// cache's attention (those are the library's). bf16 in memory, fp32 inside, one rounding where vLLM's native path
// rounds once. Built by build.rs into a fatbin and loaded with the driver.
#if defined(__HIPCC__)
#include "../../gpu/hip.h"  // AMD GPUs through HIP (build.rs, GLYD_HIP=1): the CUDA names the kernels use
#else
#include <cuda_bf16.h>
#endif
#include <math.h>
#include <stdint.h>
#include "gemm_idx.h"

typedef __nv_bfloat16 bf16;

// A broken state's words before a kernel traps (device printf), in every build.
#define DEV_SAY(...) printf(__VA_ARGS__)

__device__ __forceinline__ float b2f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 f2b(float x) { return __float2bfloat16_rn(x); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// The sum over a block (blockDim.x a multiple of 32, at most 1024), the same on every thread.
__device__ float block_sum(float v) {
    __shared__ float sh[32];
    __syncthreads();
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        float t = threadIdx.x < (blockDim.x >> 5) ? sh[threadIdx.x] : 0.f;
        t = warp_sum(t);
        if (threadIdx.x == 0) sh[0] = t;
    }
    __syncthreads();
    return sh[0];
}

// dst [K] = row `row` of src [., K].
extern "C" __global__ void copy_row(const bf16* src, int row, int K, bf16* dst) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < K; i += gridDim.x * blockDim.x) dst[i] = src[(size_t)row * K + i];
}

// dst rows <- src rows named by `pairs` ((dst row, src row) a pair; K a multiple of 8): block i copies pair i's row.
extern "C" __global__ void copy_rows(const bf16* src, const int2* pairs, int K, bf16* dst) {
    const int2 p = pairs[blockIdx.x];
    const uint4* s = reinterpret_cast<const uint4*>(src + (size_t)p.y * K);
    uint4* d = reinterpret_cast<uint4*>(dst + (size_t)p.x * K);
    for (int i = threadIdx.x; i < K / 8; i += blockDim.x) d[i] = s[i];
}

#define RN_PER_THREAD 8  // elements a thread holds: the norms take K up to RN_PER_THREAD * blockDim.x (8192 with 1024 threads)

// out [M, K] = rmsnorm(x [M, K]) * w, in fp32 with one rounding to bf16, as vLLM's default (compiled) mode computes it: x * rsqrt(mean(x^2) + eps) * w.
// One block a row, the row held in registers between the sum and the product.
extern "C" __global__ void rmsnorm(const bf16* x, const bf16* w, bf16* out, int K, float eps) {
    const bf16* xr = x + (size_t)blockIdx.x * K;
    bf16* o = out + (size_t)blockIdx.x * K;
    float v[RN_PER_THREAD];
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < RN_PER_THREAD; j++) {
        int i = threadIdx.x + j * blockDim.x;
        v[j] = i < K ? b2f(xr[i]) : 0.f;
        acc += v[j] * v[j];
    }
    acc = block_sum(acc);
    float s = rsqrtf(acc / K + eps);
#pragma unroll
    for (int j = 0; j < RN_PER_THREAD; j++) {
        int i = threadIdx.x + j * blockDim.x;
        if (i < K) o[i] = f2b(v[j] * s * b2f(w[i]));
    }
}

// x [M, K] += y [M, K] (the sum in fp32, rounded to bf16 as the residual stream is stored), then out = rmsnorm(sum) * w from the sum before its
// rounding, as vLLM's compiled fused add and norm: the variance and the normalized values come from the fp32 sum, one rounding at the end.
extern "C" __global__ void rmsnorm_add(bf16* x, const bf16* y, const bf16* w, bf16* out, int K, float eps) {
    bf16* xr = x + (size_t)blockIdx.x * K;
    const bf16* yr = y + (size_t)blockIdx.x * K;
    bf16* o = out + (size_t)blockIdx.x * K;
    float z[RN_PER_THREAD];
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < RN_PER_THREAD; j++) {
        int i = threadIdx.x + j * blockDim.x;
        z[j] = i < K ? b2f(xr[i]) + b2f(yr[i]) : 0.f;
        acc += z[j] * z[j];
    }
    acc = block_sum(acc);
    float s = rsqrtf(acc / K + eps);
#pragma unroll
    for (int j = 0; j < RN_PER_THREAD; j++) {
        int i = threadIdx.x + j * blockDim.x;
        if (i < K) {
            xr[i] = f2b(z[j]);
            o[i] = f2b(z[j] * s * b2f(w[i]));
        }
    }
}

// The query and key heads of a qkv row [q heads | k heads | v heads] (each 128 channels): per head the RMSNorm with the
// head's weight (qw for the query heads, kw for the key heads), then rotary embedding in the NeoX layout (channel i with
// i + 64) by the position's cos and sin (bf16 tables [max_pos, 64]), all in fp32 and rounded once at the end. Query heads go to
// qo [M, nq 128], key heads to ko [M, nk 128]; the value heads stay where they are. Block (token, head), 64 threads.
extern "C" __global__ void qk_norm_rope(const bf16* qkv, int stride, const bf16* qw, const bf16* kw, const bf16* cs, const bf16* sn, const int* pos, bf16* qo, bf16* ko, int nq, int nk, float eps) {
    int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;  // 64 threads: channels i and i + 64
    const bf16* x = qkv + (size_t)t * stride + (size_t)h * 128;
    const bool isq = h < nq;
    const bf16* w = isq ? qw : kw;
    float a = b2f(x[i]), b = b2f(x[i + 64]);
    float acc = block_sum(a * a + b * b);  // (two warps)
    float s = rsqrtf(acc / 128.f + eps);
    float n1 = a * s * b2f(w[i]);  // (no rounding between the norm and the rotation: vLLM's compiled graph fuses them)
    float n2 = b * s * b2f(w[i + 64]);
    int p = pos[t];
    float c = b2f(cs[(size_t)p * 64 + i]), sv = b2f(sn[(size_t)p * 64 + i]);
    float o1 = n1 * c - n2 * sv, o2 = n2 * c + n1 * sv;
    bf16* o = isq ? qo + (size_t)t * nq * 128 + (size_t)h * 128 : ko + (size_t)t * nk * 128 + (size_t)(h - nq) * 128;
    o[i] = f2b(o1);
    o[i + 64] = f2b(o2);
}

// out [M, I] = silu(gate) * up for gu [M, 2 I] (gate in the first I columns), one rounding. I a multiple of 8; a thread takes 8 columns
// (16-byte loads), block (row, chunk of 256 threads' columns).
extern "C" __global__ void silu_mul(const bf16* gu, bf16* out, int I) {
    int i = (blockIdx.y * blockDim.x + threadIdx.x) * 8;
    if (i >= I) return;
    const bf16* g = gu + (size_t)blockIdx.x * 2 * I;
    uint4 gv = *reinterpret_cast<const uint4*>(g + i);
    uint4 uv = *reinterpret_cast<const uint4*>(g + I + i);
    const bf16* gp = reinterpret_cast<const bf16*>(&gv);
    const bf16* up = reinterpret_cast<const bf16*>(&uv);
    uint4 ov;
    bf16* op = reinterpret_cast<bf16*>(&ov);
#pragma unroll
    for (int k = 0; k < 8; k++) {
        float x = b2f(gp[k]), u = b2f(up[k]);
        op[k] = f2b(x / (1.f + expf(-x)) * u);
    }
    *reinterpret_cast<uint4*>(out + (size_t)blockIdx.x * I + i) = ov;
}

// The first largest of V bf16 logits, V a multiple of 8 (torch.argmax's choice among ties): its index into *token, its value into *value.
// One block of 1024 threads a row of logits [rows, V]: row r's into token[r] and value[r].
extern "C" __global__ void argmax_bf16(const bf16* logits, int V, int* token, float* value) {
    logits += (size_t)blockIdx.x * V;
    float best = -INFINITY;
    int bi = 0x7fffffff;
    for (int c = threadIdx.x; c * 8 < V; c += blockDim.x) {  // (V a multiple of 8: 16-byte loads)
        uint4 q = *reinterpret_cast<const uint4*>(logits + (size_t)c * 8);
        const bf16* p = reinterpret_cast<const bf16*>(&q);
#pragma unroll
        for (int k = 0; k < 8; k++) {
            float v = b2f(p[k]);
            int i = c * 8 + k;
            if (v > best || (v == best && i < bi)) {
                best = v;
                bi = i;
            }
        }
    }
    __shared__ float sv[32];
    __shared__ int si[32];
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, best, o);
        int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > best || (ov == best && oi < bi)) {
            best = ov;
            bi = oi;
        }
    }
    if ((threadIdx.x & 31) == 0) {
        sv[threadIdx.x >> 5] = best;
        si[threadIdx.x >> 5] = bi;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        int nw = blockDim.x >> 5;
        best = threadIdx.x < nw ? sv[threadIdx.x] : -INFINITY;
        bi = threadIdx.x < nw ? si[threadIdx.x] : 0x7fffffff;
        for (int o = 16; o > 0; o >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, best, o);
            int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > best || (ov == best && oi < bi)) {
                best = ov;
                bi = oi;
            }
        }
        if (threadIdx.x == 0) {
            token[blockIdx.x] = bi;
            value[blockIdx.x] = best;
        }
    }
}

// The TOPK largest of V bf16 logits per row, largest first, equal values by the smaller index first (the order the host's sampler takes its candidates in): their indices into tok[row * TOPK + j] and their values into
// val[row * TOPK + j] (as float). V a multiple of 8, at least TOPK. One block of 1024 threads a row (a step's rows side by side).
// A value's 16 bits are made an unsigned number that grows with the value (-0 and +0 the same), and the TOPK-th largest is found by two histograms of 8 bits each (the high byte over the row, the low byte over the high byte's bin); then
// one pass in index order takes every number above it and the first of those equal to it, by block-wide counts, so that which of several equal values is taken does not depend on the order the threads run in; a last step ranks the TOPK.
#define TOPK 64

__device__ __forceinline__ unsigned bf16_key(unsigned short b) {
    if (b == 0x8000u) b = 0;
    return (b & 0x8000u) ? (~b & 0xFFFFu) : (b | 0x8000u);
}

// Eight logits of row at chunk c (16 bytes), those the row's mask `mrow` (V bits, a token's in word token / 32, bit token % 32; null: no mask) does not allow put at 0xFFFF, the bits whose key is the least (a grammar's mask, as the host puts -inf on them): they
// are chosen last, and only where the row allows fewer than TOPK tokens, in the index order; the host drops what its mask does not allow.
__device__ __forceinline__ uint4 load_chunk(const unsigned short* row, int c, const unsigned* mrow) {
    uint4 q = *reinterpret_cast<const uint4*>(row + (size_t)c * 8);
    if (mrow) {
        const unsigned mb = (mrow[c >> 2] >> ((c & 3) * 8)) & 0xFFu;
        if (mb != 0xFFu) {
            unsigned short* p = reinterpret_cast<unsigned short*>(&q);
#pragma unroll
            for (int k = 0; k < 8; k++)
                if (!((mb >> k) & 1u)) p[k] = 0xFFFFu;
        }
    }
    return q;
}

// `mask` (or null): per row `words` words, the first nonzero where the row is held to a mask (else the row is as if there were none), then the mask's bits (V / 32 rounded up words).
__device__ __forceinline__ void topk64_impl(const bf16* logits, int V, int* tok, float* val, const unsigned* mask, int words) {
    __shared__ unsigned hist[256];
    __shared__ unsigned hist32[256 * 32];  // a lane's counters apart from the other lanes' (the values of a row crowd into a few bins: counters shared by a warp's lanes would be updated one lane at a time)
    __shared__ unsigned sel[5];  // the high byte, the numbers above its bin, then the low byte, how many equal ones to take, the numbers above the threshold
    __shared__ int s_idx[TOPK];
    __shared__ unsigned short s_bits[TOPK];
    __shared__ unsigned wg[32], we[32];
    __shared__ unsigned run[2];
    const unsigned short* row = reinterpret_cast<const unsigned short*>(logits) + (size_t)blockIdx.x * V;
    const int nch = V / 8;  // 16-byte loads
    const unsigned* mrow = (mask && mask[(size_t)blockIdx.x * words] != 0u) ? mask + (size_t)blockIdx.x * words + 1 : nullptr;
    // pass 1: the high bytes
    const int lane0 = threadIdx.x & 31;
    for (int i = threadIdx.x; i < 256 * 32; i += blockDim.x) hist32[i] = 0;
    __syncthreads();
    for (int c = threadIdx.x; c < nch; c += blockDim.x) {
        const uint4 q = load_chunk(row, c, mrow);
        const unsigned short* p = reinterpret_cast<const unsigned short*>(&q);
#pragma unroll
        for (int k = 0; k < 8; k++) atomicAdd(&hist32[(bf16_key(p[k]) >> 8) * 32 + lane0], 1u);
    }
    __syncthreads();
    for (int b = threadIdx.x; b < 256; b += blockDim.x) {
        unsigned t = 0;
        for (int l = 0; l < 32; l++) t += hist32[b * 32 + l];
        hist[b] = t;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned acc = 0;
        int b = 255;
        for (; b > 0; b--) {
            if (acc + hist[b] >= TOPK) break;
            acc += hist[b];
        }
        sel[0] = b;
        sel[1] = acc;
    }
    __syncthreads();
    // pass 2: the low bytes of the elements in that bin
    const unsigned hi = sel[0];
    for (int i = threadIdx.x; i < 256 * 32; i += blockDim.x) hist32[i] = 0;
    __syncthreads();
    for (int c = threadIdx.x; c < nch; c += blockDim.x) {
        const uint4 q = load_chunk(row, c, mrow);
        const unsigned short* p = reinterpret_cast<const unsigned short*>(&q);
#pragma unroll
        for (int k = 0; k < 8; k++) {
            const unsigned key = bf16_key(p[k]);
            if ((key >> 8) == hi) atomicAdd(&hist32[(key & 0xFFu) * 32 + lane0], 1u);
        }
    }
    __syncthreads();
    for (int b = threadIdx.x; b < 256; b += blockDim.x) {
        unsigned t = 0;
        for (int l = 0; l < 32; l++) t += hist32[b * 32 + l];
        hist[b] = t;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned want = TOPK - sel[1];
        unsigned acc = 0;
        int b = 255;
        for (; b > 0; b--) {
            if (acc + hist[b] >= want) break;
            acc += hist[b];
        }
        sel[2] = b;
        sel[3] = want - acc;
        sel[4] = sel[1] + acc;
        run[0] = 0;
        run[1] = 0;
    }
    __syncthreads();
    const unsigned thr = (hi << 8) | sel[2], need_eq = sel[3], n_above = sel[4];
    // pass 3: the numbers above the threshold, and the first `need_eq` equal to it, in index order
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
    for (int base = 0; base < nch; base += blockDim.x) {
        const int c = base + threadIdx.x;
        unsigned short bits[8];
        unsigned gm = 0, em = 0;  // bit k: element k is above the threshold, equal to it
        if (c < nch) {
            const uint4 q = load_chunk(row, c, mrow);
            const unsigned short* p = reinterpret_cast<const unsigned short*>(&q);
#pragma unroll
            for (int k = 0; k < 8; k++) {
                bits[k] = p[k];
                const unsigned key = bf16_key(p[k]);
                gm |= (key > thr ? 1u : 0u) << k;
                em |= (key == thr ? 1u : 0u) << k;
            }
        }
        const unsigned g = __popc(gm), e = __popc(em);
        // inclusive scans over the warp, then over the warps' totals
        unsigned ig = g, ie = e;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const unsigned tg = __shfl_up_sync(0xffffffffu, ig, o), te = __shfl_up_sync(0xffffffffu, ie, o);
            if (lane >= o) {
                ig += tg;
                ie += te;
            }
        }
        if (lane == 31) {
            wg[warp] = ig;
            we[warp] = ie;
        }
        __syncthreads();
        unsigned wog = 0, woe = 0;
        for (int w = 0; w < warp; w++) {
            wog += wg[w];
            woe += we[w];
        }
        unsigned tot_g = 0, tot_e = 0;
        for (int w = 0; w < nwarps; w++) {
            tot_g += wg[w];
            tot_e += we[w];
        }
        unsigned og = run[0] + wog + ig - g, oe = run[1] + woe + ie - e;
#pragma unroll
        for (int k = 0; k < 8; k++) {
            if ((gm >> k) & 1u) {
                s_idx[og] = c * 8 + k;
                s_bits[og] = bits[k];
                og++;
            }
            if ((em >> k) & 1u) {
                if (oe < need_eq) {
                    s_idx[n_above + oe] = c * 8 + k;
                    s_bits[n_above + oe] = bits[k];
                }
                oe++;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            run[0] += tot_g;
            run[1] += tot_e;
        }
        __syncthreads();
    }
    // the ranks: by key, largest first, the smaller index first among equal keys
    if (threadIdx.x < TOPK) {
        const int j = threadIdx.x;
        const unsigned kj = bf16_key(s_bits[j]);
        const int ij = s_idx[j];
        int rank = 0;
        for (int m = 0; m < TOPK; m++) {
            const unsigned km = bf16_key(s_bits[m]);
            rank += (km > kj || (km == kj && s_idx[m] < ij)) ? 1 : 0;
        }
        tok[blockIdx.x * TOPK + rank] = ij;
        val[blockIdx.x * TOPK + rank] = __uint_as_float((unsigned)s_bits[j] << 16);
    }
}

extern "C" __global__ void __launch_bounds__(1024) topk64_bf16(const bf16* logits, int V, int* tok, float* val) {
    topk64_impl(logits, V, tok, val, nullptr, 0);
}

// `topk64_bf16` with a mask per row (see `topk64_impl`): `words` is the row's words in `mask`, 1 + (V + 31) / 32.
extern "C" __global__ void __launch_bounds__(1024) topk64_bf16_masked(const bf16* logits, int V, int* tok, float* val, const unsigned* mask, int words) {
    topk64_impl(logits, V, tok, val, mask, words);
}

// The sum over each row of V bf16 logits [rows, V] of exp((x - max) / temp): the normalizer of the host's top-p cut over a row's largest logits (src/sampler.rs). The difference and the quotient in fp32 as the host forms them (the same
// exponents), each exponential by expf (within 2 ulp), eight of them added in fp32 and the eights in fp64: out[row] (topkcheck holds it within 2e-6 of the sum in fp64). maxv[row]: the row's largest (argmax_bf16's value). V a multiple of 8.
// One block of 1024 threads a row.
extern "C" __global__ void expsum_bf16(const bf16* logits, int V, const float* maxv, float temp, double* out) {
    logits += (size_t)blockIdx.x * V;
    const float mx = maxv[blockIdx.x];
    double acc = 0.0;
    for (int c = threadIdx.x; c * 8 < V; c += blockDim.x) {  // (V a multiple of 8: 16-byte loads)
        const uint4 q = *reinterpret_cast<const uint4*>(logits + (size_t)c * 8);
        const bf16* p = reinterpret_cast<const bf16*>(&q);
        float s = 0.f;
#pragma unroll
        for (int k = 0; k < 8; k++) s += expf(__fdiv_rn(b2f(p[k]) - mx, temp));
        acc += (double)s;
    }
    __shared__ double sh[32];
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 32) {
        double t = threadIdx.x < (blockDim.x >> 5) ? sh[threadIdx.x] : 0.0;
        for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
        if (threadIdx.x == 0) out[blockIdx.x] = t;
    }
}

// The 16-bit values of a [n] (a page's [16, H, 2 D]) that are not b's: added to count[0] (all of them) and, by head and by key or value, to count[1 + 2 h + half] (the KV cache's check against a bf16 copy).
extern "C" __global__ void count_diff_u16(const unsigned short* a, const unsigned short* b, size_t n, unsigned long long* count, int H, int D) {
    __shared__ unsigned long long sc[17];
    if (threadIdx.x < 17) sc[threadIdx.x] = 0;
    __syncthreads();
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        if (a[i] != b[i]) {
            int h = (int)((i / (2 * D)) % H), half = (int)((i % (2 * D)) / D);
            atomicAdd(&sc[0], 1ull);
            atomicAdd(&sc[1 + (2 * h + half) % 16], 1ull);
        }
    }
    __syncthreads();
    if (threadIdx.x < 17 && sc[threadIdx.x]) atomicAdd(&count[threadIdx.x], sc[threadIdx.x]);
}

// For the proofs: the log-sum-exp of V bf16 logits (fp32, one block of 1024 threads), so the chosen token's logprob is
// value - lse without moving the logits.
extern "C" __global__ void logsumexp_bf16(const bf16* logits, int V, float maxv, float* lse) {
    float acc = 0.f;
    for (int i = threadIdx.x; i < V; i += blockDim.x) acc += expf(b2f(logits[i]) - maxv);
    acc = block_sum(acc);
    if (threadIdx.x == 0) *lse = maxv + logf(acc);
}

// For the proofs, per row of logits [R, V]: the first largest (torch.argmax's choice), the log-sum-exp (fp32), and the logit of the
// row's target token (target < 0: none, 0 is written). One block of 1024 threads a row.
extern "C" __global__ void row_stats(const bf16* logits, int V, const int* target, int* top1, float* lse, float* tlogit) {
    const bf16* l = logits + (size_t)blockIdx.x * V;
    float best = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        float v = b2f(l[i]);
        if (v > best || (v == best && i < bi)) {
            best = v;
            bi = i;
        }
    }
    __shared__ float sv[32];
    __shared__ int si[32];
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, best, o);
        int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > best || (ov == best && oi < bi)) {
            best = ov;
            bi = oi;
        }
    }
    if ((threadIdx.x & 31) == 0) {
        sv[threadIdx.x >> 5] = best;
        si[threadIdx.x >> 5] = bi;
    }
    __syncthreads();
    __shared__ float mx;
    __shared__ int mi;
    if (threadIdx.x < 32) {
        int nw = blockDim.x >> 5;
        best = threadIdx.x < nw ? sv[threadIdx.x] : -INFINITY;
        bi = threadIdx.x < nw ? si[threadIdx.x] : 0x7fffffff;
        for (int o = 16; o > 0; o >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, best, o);
            int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > best || (ov == best && oi < bi)) {
                best = ov;
                bi = oi;
            }
        }
        if (threadIdx.x == 0) {
            mx = best;
            mi = bi;
        }
    }
    __syncthreads();
    float acc = 0.f;
    for (int i = threadIdx.x; i < V; i += blockDim.x) acc += expf(b2f(l[i]) - mx);
    acc = block_sum(acc);
    if (threadIdx.x == 0) {
        top1[blockIdx.x] = mi;
        lse[blockIdx.x] = mx + logf(acc);
        int t = target[blockIdx.x];
        tlogit[blockIdx.x] = t >= 0 ? b2f(l[t]) : 0.f;
    }
}

// For the proofs: a streaming read of n 16-byte words, the memory bandwidth a kernel that only reads can get (each thread folds what it loads
// so the loads stay).
extern "C" __global__ void read_stream(const uint4* p, size_t n, unsigned* out) {
    unsigned acc = 0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint4 v = __ldcs(p + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x9e3779b9u) out[0] = acc;
}

// l2check (a development command, GLY-312): does a line asked of L2 ahead make the read of it faster on this GPU? mode 0: prefetch.global.L2::evict_last a line of
// 128 bytes; mode 1: a load of each line (the warp waits for it). gap_wait: a block busy for ns nanoseconds (the gap a small kernel makes).
extern "C" __global__ void l2_ask(const unsigned char* p, size_t bytes, int mode, unsigned* out) {
    unsigned acc = 0;
    for (size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 128; i < bytes; i += (size_t)gridDim.x * blockDim.x * 128) {
#if defined(__HIP_PLATFORM_AMD__)
        if (mode == 0) continue;  // (RDNA has no prefetch into L2: mode 0 asks nothing)
#else
        if (mode == 0) asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(p + i));
#endif
        else acc ^= __ldcg(reinterpret_cast<const unsigned*>(p + i));
    }
    if (acc == 0x9e3779b9u) out[0] = acc;
}

extern "C" __global__ void gap_wait(long long ns) {
    long long t0, t;
#if defined(__HIP_PLATFORM_AMD__)
    // ponytail: RDNA's wall clock taken as 100 MHz (10 ns a tick, hipDeviceAttributeWallClockRate on RDNA 3); read the rate if a card says otherwise
    t0 = (long long)wall_clock64() * 10;
    do t = (long long)wall_clock64() * 10; while (t - t0 < ns);
#else
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); while (t - t0 < ns);
#endif
}

// ---- prompt attention --------------------------------------------------------------------------------------------------------------
// Causal attention for the n tokens of a prompt chunk (positions start .. start + n - 1) over the keys and values of tokens 0 .. start + n - 1
// held in bf16 as kv [tokens, kv_heads, 256] (a token's head: its 128 keys, then its 128 values), the query heads q [n, heads, 128] (rows q_stride
// apart), out [n, heads, 128] (rows out_stride apart); heads / kv_heads query heads read a kv head. fp32 softmax and sums, the probabilities rounded to bf16
// for the values, as FlashAttention does. A block takes 16 * WARPS query rows of one head and walks the keys BN at a time, K and V in shared memory
// two tiles deep (cp.async), the scores and the output in registers (mma.sync m16n8k16). scale_log2 = softmax scale * log2(e).
#define HD 128

#if defined(__HIP_PLATFORM_AMD__)
// AMD (RDNA, wave32) has no async copies, ldmatrix or mma.sync: the copies are made when asked (cp_commit and cp_wait have nothing to wait for), and ldmatrix and mma.sync are
// gpu/warp_emu.h's shuffles and FMAs over the same fragments (glyd-engine/tools/warp_emu_test.cu holds them to the instructions on an NVIDIA GPU). The tensor-core kernels (gemm3,
// the prompt attention, the cb2 and qwen4x prompt kernels, skinny's tensor form) run unchanged on these: correct, slower.
__device__ __forceinline__ uint32_t s_addr(const void* p) { return glyd_lds_addr(p); }

// 16 bytes global -> shared, zeros where !ok
__device__ __forceinline__ void cp16(uint32_t dst, const void* src, bool ok) { glyd_lds_put16(dst, ok ? *reinterpret_cast<const uint4*>(src) : make_uint4(0, 0, 0, 0)); }
__device__ __forceinline__ void cp_commit() {}
template <int N>
__device__ __forceinline__ void cp_wait() {}
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], uint32_t a) { glyd_ldsm4(r, a); }
__device__ __forceinline__ void ldsm4t(uint32_t (&r)[4], uint32_t a) { glyd_ldsm4t(r, a); }
// c += a (16x16 bf16, row) * b (16x8 bf16, col), fp32
__device__ __forceinline__ void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) { glyd_mma16816(c, a, b0, b1); }
__device__ __forceinline__ float ex2(float x) { return __builtin_amdgcn_exp2f(x); }  // (v_exp_f32: ex2.approx's)
#else
__device__ __forceinline__ uint32_t s_addr(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

// 16 bytes global -> shared, zeros where !ok
__device__ __forceinline__ void cp16(uint32_t dst, const void* src, bool ok) {
    int n = ok ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(n));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm4t(uint32_t (&r)[4], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
// c += a (16x16 bf16, row) * b (16x8 bf16, col), fp32
__device__ __forceinline__ void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}
#endif
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
}
// the element offset of 16-byte chunk c (0..15) of a 256-byte row r in a tile: the chunks of a row XORed by r % 8 so that ldmatrix's eight rows meet no bank twice
__device__ __forceinline__ int swz(int r, int c) { return r * HD + ((c ^ (r & 7)) << 3); }

template <int WARPS, int BN>
__device__ __forceinline__ void prefill_attn_t(const bf16* __restrict__ q, int q_stride, const bf16* __restrict__ kv, bf16* __restrict__ out, int out_stride, int n, int start,
                                               int heads, int kv_heads, float scale_log2) {
    constexpr int BM = WARPS * 16;
    constexpr int NT = WARPS * 32;
    constexpr int NJ = BN / 8;  // 8-key tiles of the score tile
    extern __shared__ __align__(128) unsigned char smem_raw[];
    bf16* sQ = reinterpret_cast<bf16*>(smem_raw);  // [BM][128]
    bf16* sK = sQ + BM * HD;                       // [2][BN][128]
    bf16* sV = sK + 2 * BN * HD;                   // [2][BN][128]

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int qt = gridDim.x - 1 - blockIdx.x;  // the tiles with the most keys first
    const int h = blockIdx.y;
    const int hk = h / (heads / kv_heads);
    const int q0 = qt * BM;
    const int kv_len = start + n;
    const int kv_end = min(kv_len, start + q0 + BM);
    const int nblk = (kv_end + BN - 1) / BN;

    for (int i = tid; i < BM * 16; i += NT) {
        int r = i >> 4, c = i & 15;
        bool ok = q0 + r < n;
        cp16(s_addr(sQ + swz(r, c)), ok ? q + (size_t)(q0 + r) * q_stride + (size_t)h * HD + c * 8 : q, ok);
    }
    auto load_kv = [&](int stage, int kb) {
        bf16* dk = sK + stage * BN * HD;
        bf16* dv = sV + stage * BN * HD;
        for (int i = tid; i < BN * 16; i += NT) {
            int r = i >> 4, c = i & 15;
            int kt = kb * BN + r;
            bool ok = kt < kv_len;
            const bf16* src = kv + ((size_t)(ok ? kt : 0) * kv_heads + hk) * 256 + c * 8;
            cp16(s_addr(dk + swz(r, c)), src, ok);
            cp16(s_addr(dv + swz(r, c)), src + HD, ok);
        }
    };
    load_kv(0, 0);
    cp_commit();

    const int g = lane >> 2, t4 = lane & 3;   // the row of an accumulator and its column pair
    const int mi = lane >> 3, r8 = lane & 7;  // ldmatrix: the matrix a lane's address is for, and its row
    const int wrow = warp * 16;
    uint32_t qa[HD / 16][4];
    float o[HD / 8][4];
#pragma unroll
    for (int i = 0; i < HD / 8; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};

    for (int jb = 0; jb < nblk; jb++) {
        if (jb + 1 < nblk) load_kv((jb + 1) & 1, jb + 1);
        cp_commit();
        cp_wait<1>();
        __syncthreads();
        if (jb == 0) {
#pragma unroll
            for (int ks = 0; ks < HD / 16; ks++) ldsm4(qa[ks], s_addr(sQ + swz(wrow + (mi & 1) * 8 + r8, ks * 2 + (mi >> 1))));
        }
        const bf16* cK = sK + (jb & 1) * BN * HD;
        const bf16* cV = sV + (jb & 1) * BN * HD;

        float s[NJ][4];
#pragma unroll
        for (int i = 0; i < NJ; i++) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f;
#pragma unroll
        for (int ks = 0; ks < HD / 16; ks++) {
#pragma unroll
            for (int nt2 = 0; nt2 < NJ / 2; nt2++) {
                uint32_t b[4];
                ldsm4(b, s_addr(cK + swz(nt2 * 16 + (mi >> 1) * 8 + r8, ks * 2 + (mi & 1))));
                mma16816(s[nt2 * 2], qa[ks], b[0], b[1]);
                mma16816(s[nt2 * 2 + 1], qa[ks], b[2], b[3]);
            }
        }

        // the scale, the causal mask where this block reaches past the warp's first row or the last key, the new row maxima
        const bool mask = (jb * BN + BN - 1 > start + q0 + wrow) || (jb * BN + BN > kv_len);
        float mx[2] = {m[0], m[1]};
#pragma unroll
        for (int nt = 0; nt < NJ; nt++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                float v = s[nt][e] * scale_log2;
                if (mask) {
                    int key = jb * BN + nt * 8 + 2 * t4 + (e & 1);
                    int qpos = start + q0 + wrow + g + ((e >> 1) << 3);
                    if (key > qpos || key >= kv_len) v = -INFINITY;
                }
                s[nt][e] = v;
                mx[e >> 1] = fmaxf(mx[e >> 1], v);
            }
        }
#pragma unroll
        for (int i = 0; i < 2; i++) {
            mx[i] = fmaxf(mx[i], __shfl_xor_sync(0xffffffffu, mx[i], 1));
            mx[i] = fmaxf(mx[i], __shfl_xor_sync(0xffffffffu, mx[i], 2));
        }
        float mm[2], corr[2];
#pragma unroll
        for (int i = 0; i < 2; i++) {
            mm[i] = mx[i] == -INFINITY ? 0.f : mx[i];
            corr[i] = ex2(m[i] - mm[i]);
            m[i] = mx[i];
            l[i] *= corr[i];
        }
#pragma unroll
        for (int i = 0; i < HD / 8; i++) {
            o[i][0] *= corr[0];
            o[i][1] *= corr[0];
            o[i][2] *= corr[1];
            o[i][3] *= corr[1];
        }
#pragma unroll
        for (int nt = 0; nt < NJ; nt++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                float p = ex2(s[nt][e] - mm[e >> 1]);
                s[nt][e] = p;
                l[e >> 1] += p;
            }
        }
#pragma unroll
        for (int kk = 0; kk < BN / 16; kk++) {
            uint32_t pa[4] = {pack_bf16(s[2 * kk][0], s[2 * kk][1]), pack_bf16(s[2 * kk][2], s[2 * kk][3]), pack_bf16(s[2 * kk + 1][0], s[2 * kk + 1][1]), pack_bf16(s[2 * kk + 1][2], s[2 * kk + 1][3])};
#pragma unroll
            for (int nt2 = 0; nt2 < HD / 16; nt2++) {
                uint32_t b[4];
                ldsm4t(b, s_addr(cV + swz(kk * 16 + (mi & 1) * 8 + r8, nt2 * 2 + (mi >> 1))));
                mma16816(o[nt2 * 2], pa, b[0], b[1]);
                mma16816(o[nt2 * 2 + 1], pa, b[2], b[3]);
            }
        }
        __syncthreads();
    }

    float inv[2];
#pragma unroll
    for (int i = 0; i < 2; i++) {
        float t = l[i];
        t += __shfl_xor_sync(0xffffffffu, t, 1);
        t += __shfl_xor_sync(0xffffffffu, t, 2);
        inv[i] = 1.f / t;
    }
    const int row0 = q0 + wrow + g, row1 = row0 + 8;
#pragma unroll
    for (int nt = 0; nt < HD / 8; nt++) {
        int col = h * HD + nt * 8 + 2 * t4;
        if (row0 < n) *reinterpret_cast<uint32_t*>(out + (size_t)row0 * out_stride + col) = pack_bf16(o[nt][0] * inv[0], o[nt][1] * inv[0]);
        if (row1 < n) *reinterpret_cast<uint32_t*>(out + (size_t)row1 * out_stride + col) = pack_bf16(o[nt][2] * inv[1], o[nt][3] * inv[1]);
    }
}

// dynamic shared memory: (16 WARPS + 4 BN) * 256 bytes
extern "C" __global__ void __launch_bounds__(128, 1) prefill_attn_w4_b64(const bf16* q, int qs, const bf16* kv, bf16* out, int os, int n, int start, int heads, int kvh, float sl2) { prefill_attn_t<4, 64>(q, qs, kv, out, os, n, start, heads, kvh, sl2); }
extern "C" __global__ void __launch_bounds__(256, 1) prefill_attn_w8_b64(const bf16* q, int qs, const bf16* kv, bf16* out, int os, int n, int start, int heads, int kvh, float sl2) { prefill_attn_t<8, 64>(q, qs, kv, out, os, n, start, heads, kvh, sl2); }
extern "C" __global__ void __launch_bounds__(128, 2) prefill_attn_w4_b32(const bf16* q, int qs, const bf16* kv, bf16* out, int os, int n, int start, int heads, int kvh, float sl2) { prefill_attn_t<4, 32>(q, qs, kv, out, os, n, start, heads, kvh, sl2); }
extern "C" __global__ void __launch_bounds__(256, 1) prefill_attn_w8_b32(const bf16* q, int qs, const bf16* kv, bf16* out, int os, int n, int start, int heads, int kvh, float sl2) { prefill_attn_t<8, 32>(q, qs, kv, out, os, n, start, heads, kvh, sl2); }

// ---- KV calibration (src/calib.rs; the plugin's kvcal rule) ---------------------------------------------------------------------------
// The keys or values of tokens are bf16 bits x [T, C] (C = kv heads * 128 channels, rows `stride` elements apart). A value's exponent >> 1 is bits 8 to 14: 128 bins; a channel's window is 8
// consecutive bins from its base; a value is outside it where its bin is below the base or at base + 8 or more.

// counts [C, 128] += the bins of x's values, per channel. A block takes 16 channels; its threads walk the tokens.
extern "C" __global__ void kv_hist(const unsigned short* x, int stride, int T, int C, int* counts) {
    __shared__ int sh[16 * 128];
    for (int i = threadIdx.x; i < 16 * 128; i += blockDim.x) sh[i] = 0;
    __syncthreads();
    const int c0 = blockIdx.x * 16;
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        const unsigned short* r = x + (size_t)t * stride + c0;
#pragma unroll
        for (int j = 0; j < 16; j++) atomicAdd(&sh[j * 128 + ((r[j] >> 8) & 127)], 1);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 16 * 128; i += blockDim.x)
        if (sh[i]) atomicAdd(&counts[(size_t)c0 * 128 + i], sh[i]);
}

__device__ __forceinline__ int outside(unsigned short bits, int base) {
    int e = (bits >> 8) & 127;
    return e < base || e >= base + 8;
}

// *total += the values of x [T, C] outside their channels' windows.
extern "C" __global__ void kv_outside_total(const unsigned short* x, int stride, int T, int C, const unsigned char* base, unsigned long long* total) {
    unsigned long long acc = 0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < (size_t)T * C; i += (size_t)gridDim.x * blockDim.x) {
        int t = (int)(i / C), c = (int)(i % C);
        acc += outside(x[(size_t)t * stride + c], base[c]);
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if ((threadIdx.x & 31) == 0 && acc) atomicAdd(total, acc);
}

// hp [page, head] = the values outside the windows in a page's 16 tokens' keys and values together, for `npages` pages from the token at row 0 of k and v (rows ks and vs apart); the page's
// number in hp is page0 + its index. Block (page, head), 128 threads: a thread a channel.
extern "C" __global__ void kv_leaf_hp(const unsigned short* k, int ks, const unsigned short* v, int vs, const unsigned char* kbase, const unsigned char* vbase, int H, int page0, int* hp) {
    const int p = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const int c = h * 128 + d;
    int n = 0;
    for (int t = 0; t < 16; t++) {
        size_t row = (size_t)p * 16 + t;
        n += outside(k[row * ks + c], kbase[c]) + outside(v[row * vs + c], vbase[c]);
    }
    n = (int)block_sum((float)n);
    if (threadIdx.x == 0) hp[(size_t)(page0 + p) * H + h] = n;
}

// The sweep of one token a row: for row t of k and v [M, C]: its exceptions in each head (keys and values together), the entries past its heads' inline ones that a page of 16 of it needs
// (sum over heads of max(16 e - 3, 0)) into need[(row0 + t) * layers + layer] and its exceptions in all heads into exc[...] (int16). Block (row), a warp a head.
extern "C" __global__ void kv_sweep(const unsigned short* k, int ks, const unsigned short* v, int vs, const unsigned char* kbase, const unsigned char* vbase, int H, int layer, int layers, int row0, int* need, short* exc) {
    const int t = blockIdx.x, h = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int n = 0;
    for (int j = lane; j < 128; j += 32) {
        int c = h * 128 + j;
        n += outside(k[(size_t)t * ks + c], kbase[c]) + outside(v[(size_t)t * vs + c], vbase[c]);
    }
    for (int o = 16; o > 0; o >>= 1) n += __shfl_xor_sync(0xffffffffu, n, o);
    __shared__ int hp[32];
    if (lane == 0) hp[h] = n;
    __syncthreads();
    if (threadIdx.x == 0) {
        int nd = 0, ex = 0;
        for (int i = 0; i < H; i++) {
            nd += max(hp[i] * 16 - 3, 0);
            ex += hp[i];
        }
        need[(size_t)(row0 + t) * layers + layer] = nd;
        exc[(size_t)(row0 + t) * layers + layer] = (short)ex;
    }
}

// A one-token sequence's attention: the softmax of its one score is 1, so each query head's output is its key-value head's value row. out [M, heads, 128] from the values v [M, kv_heads, 128]
// (rows vs apart, out rows os apart). Block (row, head), 128 threads.
extern "C" __global__ void attn_self(const bf16* v, int vs, bf16* out, int os, int heads, int kv_heads) {
    const int t = blockIdx.x, h = blockIdx.y;
    out[(size_t)t * os + h * 128 + threadIdx.x] = v[(size_t)t * vs + (h / (heads / kv_heads)) * 128 + threadIdx.x];
}

// ---- dense bf16 products ----------------------------------------------------------------------------------------------------------------
// Y [M, N] = X [M, K] W^T with W [N, K] bf16 (the weights decoded for a long prompt): row-major, K contiguous in both operands, fp32 sums, one rounding to bf16. A block takes BM rows of
// X and BN rows of W (WARPS_M x WARPS_N warps, each WM x WN of the tile), walking K in steps of BK through STAGES shared-memory stages of cp.async loads, each warp's tile in registers
// (mma.sync m16n8k16, ldmatrix). N a multiple of BN and K of BK; M any (rows past it are zeros in, nothing out). Blocks are numbered in groups of GROUP_M row tiles so that the blocks
// that run together share operands in L2.
// The same product with its address arithmetic done once, before the main loop (gemm2's loads were loops of a runtime trip count, each chunk's row, swizzle and 64-bit address worked out again
// every k-tile: 7 instructions to a tensor-core instruction in the loop, and the power cap of an L4 is spent on them): each thread's chunk offsets and ldmatrix offsets are registers, a k-tile adds one
// constant to a pointer, the loads and fragment loads are unrolled with immediate offsets (kernels/gemm_idx.h, checked on the host by tools/gemm_idx_test.cpp).
// FUSE: Y [M, N/2] = silu(gate) * up where W's rows are [gate rows (N/2) | up rows (N/2)]: a tile's B rows are gathered so that each warp holds a gate column and its up column in registers (its WN rows:
// the first half gate rows, the second half up rows of the same columns), and the product is rounded to bf16 first and then combined, as the separate product and silu_mul do (the same bits).
template <int BM, int BN, int BK, int WARPS_M, int WARPS_N, int STAGES, int GROUP_M, bool FUSE, bool F32 = false>
__device__ __forceinline__ void gemm3_nt_t(const bf16* __restrict__ X, const bf16* __restrict__ W, bf16* __restrict__ Y, int M, int N, int K) {
    constexpr int NTH = WARPS_M * WARPS_N * 32;
    constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N;
    constexpr int MT = WM / 16, NTL = WN / 8;
    constexpr int KS = BK / 16;
    using F = Fill<BK, NTH>;
    constexpr int RPI = F::RPI, PA = BM / RPI, PB = BN / RPI;
    constexpr uint32_t STAGE_A = BM * BK * 2, STAGE_B = BN * BK * 2;
    static_assert(BM % RPI == 0 && BN % RPI == 0 && KS % 2 == 0 && WM % 16 == 0 && WN % 16 == 0, "the tile's shape");
    static_assert(!FUSE || (NTL % 2 == 0), "a fused warp tile holds gate and up halves");
    extern __shared__ __align__(128) unsigned char smem_raw[];
    const uint32_t sbase = s_addr(smem_raw);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp / WARPS_N, wn = warp % WARPS_N;
    const int num_m = (M + BM - 1) / BM, num_n = N / BN;
    const int pid = blockIdx.x;
    const int in_group = GROUP_M * num_n;
    const int first_m = (pid / in_group) * GROUP_M;
    const int gsz = min(num_m - first_m, GROUP_M);
    const int m_tile = first_m + (pid % in_group) % gsz;
    const int n_tile = (pid % in_group) / gsz;
    const int m0 = m_tile * BM, n0 = n_tile * BN;
    const int nk = K / BK;
    const int mi = lane >> 3, r8 = lane & 7;

    // the loads: this thread's chunk of each pass, as byte offsets from the start of the operand's k-tile
    const int lr = F::row(tid), lc = F::chunk(tid);
    const uint32_t soff = (uint32_t)F::soff(tid) * 2;
    uint32_t goffA[PA], goffB[PB];
    bool okA[PA];
#pragma unroll
    for (int p = 0; p < PA; p++) {
        const int row = m0 + lr + p * RPI;
        okA[p] = row < M;
        goffA[p] = (uint32_t)(okA[p] ? row : 0) * (uint32_t)K * 2u + (uint32_t)lc * 16u;
    }
#pragma unroll
    for (int p = 0; p < PB; p++) {
        int wrow;
        if (FUSE) {
            const int r = lr + p * RPI, slice = r / WN, within = r % WN, half = within / (WN / 2), cc = within % (WN / 2);
            wrow = half * (N / 2) + n_tile * (BN / 2) + slice * (WN / 2) + cc;
        } else {
            wrow = n0 + lr + p * RPI;
        }
        goffB[p] = (uint32_t)wrow * (uint32_t)K * 2u + (uint32_t)lc * 16u;
    }
    const char* xb = reinterpret_cast<const char*>(X);
    const char* wb = reinterpret_cast<const char*>(W);
    auto load_stage = [&](int st, int kt) {
        const char* xk = xb + (size_t)kt * (BK * 2);
        const char* wk = wb + (size_t)kt * (BK * 2);
        const uint32_t da = sbase + st * STAGE_A + soff;
        const uint32_t db = sbase + STAGES * STAGE_A + st * STAGE_B + soff;
#pragma unroll
        for (int p = 0; p < PA; p++) cp16(da + p * RPI * BK * 2, xk + goffA[p], okA[p]);
#pragma unroll
        for (int p = 0; p < PB; p++) cp16(db + p * RPI * BK * 2, wk + goffB[p], true);
    };
    // the fragments: one offset a k-step for A and for B, in bytes from the stage's start
    uint32_t aoff[KS], boff[KS];
#pragma unroll
    for (int ks = 0; ks < KS; ks++) {
        aoff[ks] = (uint32_t)frag_a_off<BK>(wm * WM, lane, ks) * 2;
        boff[ks] = (uint32_t)frag_b_off<BK>(wn * WN, lane, ks) * 2;
    }
    auto load_frags = [&](uint32_t (&af)[MT][4], uint32_t (&bfr)[NTL][2], int st, int ks) {
        const uint32_t a = sbase + st * STAGE_A + aoff[ks];
        const uint32_t b = sbase + STAGES * STAGE_A + st * STAGE_B + boff[ks];
#pragma unroll
        for (int mt = 0; mt < MT; mt++) ldsm4(af[mt], a + mt * 16 * BK * 2);
#pragma unroll
        for (int np = 0; np < NTL / 2; np++) {
            uint32_t t[4];
            ldsm4(t, b + np * 16 * BK * 2);
            bfr[2 * np][0] = t[0];
            bfr[2 * np][1] = t[1];
            bfr[2 * np + 1][0] = t[2];
            bfr[2 * np + 1][1] = t[3];
        }
    };
#pragma unroll
    for (int st = 0; st < STAGES - 1; st++) {
        if (st < nk) load_stage(st, st);
        cp_commit();
    }
    float acc[MT][NTL][4];
#pragma unroll
    for (int i = 0; i < MT; i++)
#pragma unroll
        for (int j = 0; j < NTL; j++) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
    cp_wait<STAGES - 2>();
    __syncthreads();
    uint32_t af[2][MT][4], bfr[2][NTL][2];
    load_frags(af[0], bfr[0], 0, 0);
    int rd = 0, wr = STAGES - 1;
    for (int kt = 0; kt < nk; kt++) {
#pragma unroll
        for (int ks = 0; ks < KS; ks++) {
            const int cur = ks & 1, nxt = cur ^ 1;
            if (ks == KS - 1) {
                cp_wait<STAGES - 2>();
                __syncthreads();
                rd = rd + 1 == STAGES ? 0 : rd + 1;
            }
            load_frags(af[nxt], bfr[nxt], rd, (ks + 1) % KS);
            if (ks == 0) {
                if (kt + STAGES - 1 < nk) load_stage(wr, kt + STAGES - 1);
                cp_commit();
                wr = wr + 1 == STAGES ? 0 : wr + 1;
            }
#pragma unroll
            for (int mt = 0; mt < MT; mt++)
#pragma unroll
                for (int nt = 0; nt < NTL; nt++) mma16816(acc[mt][nt], af[cur][mt], bfr[cur][nt][0], bfr[cur][nt][1]);
        }
    }
    cp_wait<0>();
    const int g = lane >> 2, t4 = lane & 3;
    if (FUSE) {
        // silu(gate) * up from the two halves of the warp's columns
        constexpr int HALF = NTL / 2;
#pragma unroll
        for (int mt = 0; mt < MT; mt++) {
            const int r0 = m0 + wm * WM + mt * 16 + g, r1 = r0 + 8;
#pragma unroll
            for (int nt = 0; nt < HALF; nt++) {
                const int col = n_tile * (BN / 2) + wn * (WN / 2) + nt * 8 + 2 * t4;
                float v[4];
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const float x = b2f(f2b(acc[mt][nt][e])), u = b2f(f2b(acc[mt][nt + HALF][e]));
                    v[e] = x / (1.f + expf(-x)) * u;
                }
                if (r0 < M) *reinterpret_cast<uint32_t*>(Y + (size_t)r0 * (N / 2) + col) = pack_bf16(v[0], v[1]);
                if (r1 < M) *reinterpret_cast<uint32_t*>(Y + (size_t)r1 * (N / 2) + col) = pack_bf16(v[2], v[3]);
            }
        }
    } else if (F32) {
        // (the sums as they are, fp32: Y is a float [M, N])
        float* Yf = reinterpret_cast<float*>(Y);
#pragma unroll
        for (int mt = 0; mt < MT; mt++) {
            const int r0 = m0 + wm * WM + mt * 16 + g, r1 = r0 + 8;
#pragma unroll
            for (int nt = 0; nt < NTL; nt++) {
                const int col = n0 + wn * WN + nt * 8 + 2 * t4;
                if (r0 < M) *reinterpret_cast<float2*>(Yf + (size_t)r0 * N + col) = make_float2(acc[mt][nt][0], acc[mt][nt][1]);
                if (r1 < M) *reinterpret_cast<float2*>(Yf + (size_t)r1 * N + col) = make_float2(acc[mt][nt][2], acc[mt][nt][3]);
            }
        }
    } else {
#pragma unroll
        for (int mt = 0; mt < MT; mt++) {
            const int r0 = m0 + wm * WM + mt * 16 + g, r1 = r0 + 8;
#pragma unroll
            for (int nt = 0; nt < NTL; nt++) {
                const int col = n0 + wn * WN + nt * 8 + 2 * t4;
                if (r0 < M) *reinterpret_cast<uint32_t*>(Y + (size_t)r0 * N + col) = pack_bf16(acc[mt][nt][0], acc[mt][nt][1]);
                if (r1 < M) *reinterpret_cast<uint32_t*>(Y + (size_t)r1 * N + col) = pack_bf16(acc[mt][nt][2], acc[mt][nt][3]);
            }
        }
    }
}

// dynamic shared memory: STAGES * (BM + BN) * BK * 2 bytes
// the tiles kern.rs chooses among (and the experiments' in tools/gemm_old.cuh); each is written out, not made by a macro
extern "C" __global__ void __launch_bounds__(256, 1) gemm3_256x128_s3(const bf16* X, const bf16* W, bf16* Y, int M, int N, int K) { gemm3_nt_t<256, 128, 32, 4, 2, 3, 8, false>(X, W, Y, M, N, K); }
extern "C" __global__ void __launch_bounds__(256, 1) gemm3_128x256_b64_s2(const bf16* X, const bf16* W, bf16* Y, int M, int N, int K) { gemm3_nt_t<128, 256, 64, 2, 4, 2, 8, false>(X, W, Y, M, N, K); }
extern "C" __global__ void __launch_bounds__(128, 1) gemm3_128x128_b64_s3(const bf16* X, const bf16* W, bf16* Y, int M, int N, int K) { gemm3_nt_t<128, 128, 64, 2, 2, 3, 8, false>(X, W, Y, M, N, K); }
extern "C" __global__ void __launch_bounds__(256, 1) gemm3_128x256_s3(const bf16* X, const bf16* W, bf16* Y, int M, int N, int K) { gemm3_nt_t<128, 256, 32, 2, 4, 3, 8, false>(X, W, Y, M, N, K); }
extern "C" __global__ void __launch_bounds__(128, 1) gemm3_128x128_s4(const bf16* X, const bf16* W, bf16* Y, int M, int N, int K) { gemm3_nt_t<128, 128, 32, 2, 2, 4, 8, false>(X, W, Y, M, N, K); }
extern "C" __global__ void __launch_bounds__(256, 1) gemm3_256x128_b64_s2(const bf16* X, const bf16* W, bf16* Y, int M, int N, int K) { gemm3_nt_t<256, 128, 64, 4, 2, 2, 8, false>(X, W, Y, M, N, K); }
// a small tile for the products a big tile leaves the GPU half idle with (few columns: Qwen3.8-Flash-Next's Gated Residual bottleneck of 384, its router's 512), two blocks an SM; and the same with fp32 sums out (the router's logits)
extern "C" __global__ void __launch_bounds__(128, 2) gemm3_64x64_b64_s3(const bf16* X, const bf16* W, bf16* Y, int M, int N, int K) { gemm3_nt_t<64, 64, 64, 2, 2, 3, 8, false>(X, W, Y, M, N, K); }
extern "C" __global__ void __launch_bounds__(128, 2) gemm3o_64x64_b64_s3(const bf16* X, const bf16* W, bf16* Y, int M, int N, int K) { gemm3_nt_t<64, 64, 64, 2, 2, 3, 8, false, true>(X, W, Y, M, N, K); }
extern "C" __global__ void __launch_bounds__(256, 1) gemm3f_256x128_s3(const bf16* X, const bf16* W, bf16* Y, int M, int N, int K) { gemm3_nt_t<256, 128, 32, 4, 2, 3, 8, true>(X, W, Y, M, N, K); }
extern "C" __global__ void __launch_bounds__(256, 1) gemm3f_128x256_b64_s2(const bf16* X, const bf16* W, bf16* Y, int M, int N, int K) { gemm3_nt_t<128, 256, 64, 2, 4, 2, 8, true>(X, W, Y, M, N, K); }
extern "C" __global__ void __launch_bounds__(256, 1) gemm3f_128x256_s3(const bf16* X, const bf16* W, bf16* Y, int M, int N, int K) { gemm3_nt_t<128, 256, 32, 2, 4, 3, 8, true>(X, W, Y, M, N, K); }

#include "qwen35.cu"  // Qwen3.5's kernels
#include "moe.cu"     // the mixture-of-experts layer's (the router, the activations between the experts' products)
#include "moe_rows.cu"  // the sparse experts' (the second product for the neurons that are not zero)
#include "moe_hb.cu"    // the Hummingbird experts' (codebook rows: the rotation, the two products, the weighted sum and the rotation back)
#include "nemotron.cu"  // Nemotron 3 Super's kernels (after qwen35.cu: they use its helpers)
#include "gemma4.cu"   // Gemma 4's kernels (after qwen35.cu: they use its helpers)
#include "glm.cu"  // GLM-5.3-Flash's kernels (after qwen35.cu and moe.cu: they use their helpers)
#include "glm_cb.cu"  // GLM-5.3-Flash's routed experts as codebook rows (after glm.cu)
#include "kimi.cu"  // Kimi-K3's kernels (after glm.cu and moe.cu)
#include "dsv4.cu"  // DeepSeek-V4's kernels (after glm.cu and moe.cu: they use their helpers)
#include "muse.cu"  // Muse Spark's kernels
#include "cb2_prefill.cu"  // GLM-5.3-Flash's experts as codebook rows for the tokens of a prompt chunk (after moe_hb.cu: its tile helpers)
#include "qwen4x.cu"  // Qwen3.8-Flash-Next's kernels (after qwen35.cu and glm.cu: they use their helpers)
#include "vision.cu"  // its vision tower's (after the prompt attention's helpers)
#include "skinny.cu"  // the four-, five- and six-bit block layouts' products for a few rows (a decode step's and a verification's)
#include "tp.cu"  // a model's layers across the GPUs of one machine: the gather of the slices of a product's rows
