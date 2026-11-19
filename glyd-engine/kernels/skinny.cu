// The skinny products of the four-, five- and six-bit block layouts (gpu/glyd_gpu.cu's Int4, Int5 and Int6; glyd-gpu/src/pack.rs has their bytes): y [M, N] = x [M, K] W^T for M of 1 to 8 rows, which is what a decode step
// (and the verification of a few drafts) asks of a model's always-on matrices. The library's tensor-core path pads M and leaves the memory half idle at these sizes; these read the steps as they are laid out, make the weights bf16
// pairs in registers and multiply them on the tensor cores, and are bound by the bytes (95 to 101% of the card's 736 GB/s on the matrices of 18 MB and more, whatever the rows).
//
// A block takes one row block (64 rows, blockIdx.x) over the whole of K, its 8 warps taking the pairs of steps (32 columns, the span of one scale) in turn: warp w the pairs w, w + 8, ... so that the 8 of them read adjacent
// kilobytes. A step is 64 rows by 16 columns in the order the tensor cores take their operand (pack.rs `lane_place`): lane l = 4g + t holds, for weight i = 4n + 2jh + jl (n 0 to 7), the weight of row 8n + g and column
// 8jh + 2t + jl of the step: exactly the A fragments of four tiles of 16 rows. A step's bytes: Int4 512 (lane l's weights 0 to 15 as 8 bytes at 8 l, 16 to 31 at 256 + 8 l; a word holds 8 weights of a lane, the even ones (jl 0)
// in its low half's nibbles 0 to 3 and the odd ones in nibbles 4 to 7), Int5 640 (a word of high bits at 4 l, then the same 512 bytes of low nibbles at 128), Int6 768 (two words of high bits, 2 bits a weight, at 4 l and
// 128 + 4 l, then the low nibbles as Int4's at 256). The scales: 128 bytes a pair of steps, the (g, n) row's at 2 (8 g + n). A weight is (code - 8) s (Int4), (code - 16) s (Int5) or (code - 32) s (Int6), s the bf16 scale,
// rounded to bf16 once, as the library does.

#define SK_WARPS 8


// one step's registers: the words of lane l (lo01 the first 8 bytes, lo23 the next 8), and the words of high bits (Int5: hi; Int6: hi, hi2)
struct SkStep {
    uint2 lo01, lo23;
    unsigned hi, hi2;
};

template <int BITS>
__device__ __forceinline__ SkStep sk_load_step(const unsigned char* __restrict__ step, int lane) {
    SkStep s;
    s.hi = s.hi2 = 0;
    if (BITS == 4) {
        s.lo01 = __ldcs(reinterpret_cast<const uint2*>(step + 8 * lane));
        s.lo23 = __ldcs(reinterpret_cast<const uint2*>(step + 256 + 8 * lane));
    } else if (BITS == 5) {
        s.hi = __ldcs(reinterpret_cast<const unsigned*>(step + 4 * lane));
        s.lo01 = __ldcs(reinterpret_cast<const uint2*>(step + 128 + 8 * lane));
        s.lo23 = __ldcs(reinterpret_cast<const uint2*>(step + 384 + 8 * lane));
    } else {
        s.hi = __ldcs(reinterpret_cast<const unsigned*>(step + 4 * lane));
        s.hi2 = __ldcs(reinterpret_cast<const unsigned*>(step + 128 + 4 * lane));
        s.lo01 = __ldcs(reinterpret_cast<const uint2*>(step + 256 + 8 * lane));
        s.lo23 = __ldcs(reinterpret_cast<const uint2*>(step + 512 + 8 * lane));
    }
    return s;
}

// A pair of steps and its scales
struct SkPair {
    SkStep a, b;
    uint4 sc;
};

template <int BITS>
__device__ __forceinline__ SkPair sk_load_pair(const unsigned char* __restrict__ d0, const unsigned char* __restrict__ b0, int p, int lane) {
    constexpr int SB = BITS == 4 ? 512 : BITS == 5 ? 640 : 768;
    SkPair r;
    r.a = sk_load_step<BITS>(d0 + (size_t)(2 * p) * SB, lane);
    r.b = sk_load_step<BITS>(d0 + (size_t)(2 * p + 1) * SB, lane);
    r.sc = __ldg(reinterpret_cast<const uint4*>(b0 + (size_t)p * 128 + 16 * (lane >> 2)));
    return r;
}

// ---- the tensor cores' form -------------------------------------------------------------------------------------------------------------------------
// The same products with the weights made bf16 pairs and multiplied on the tensor cores (mma.sync m16n8k16, the weights as A, the up to 8 rows of x as B's columns): a step's 32 weights of a lane are the A fragments of 4 tiles of
// 16 rows, and the code of two weights (even and odd column) sits in one 32-bit word at 16 bits apart, which is bf16x2's layout: a mask and the exponent bits of 128.0 make 128 + code, one fused multiply-add takes the bias off and
// a second multiplies by the scale (rounded to bf16 once, as the library's dequantization does). The cost of a weight no longer grows with the rows: 2 to 3 instructions a weight at 4 to 6 bits, whatever the number of rows.

__device__ __forceinline__ uint32_t sk_fma2(uint32_t a, uint32_t b, uint32_t c) {
#if defined(__HIP_PLATFORM_AMD__)
    // (fp32 FMAs rounded to bf16: one rounding where the fp32 sum is exact, as in the uses below: 128 + code less its bias, a bf16 times a scale)
    return pack_bf16(fmaf(glyd_bf_lo(a), glyd_bf_lo(b), glyd_bf_lo(c)), fmaf(glyd_bf_hi(a), glyd_bf_hi(b), glyd_bf_hi(c)));
#else
    uint32_t d;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
#endif
}

// the weights 2 PR and 2 PR + 1 of a lane's step (PR = 2 N + JH: rows 8 N + g, columns 8 JH + 2 t and +1) times the scale pair `sc2`, as bf16x2. X (the exact four-bit layout, Int4H: Int4's codes,
// a scale of its own form): `sc2` is the run's scale as an fp32's bits, and each weight round_bf16((code - 8) s), the product exact in fp32 and rounded once, as the library's Int4H.
template <int BITS, int N, int JH, bool X = false>
__device__ __forceinline__ uint32_t sk_pair(const SkStep& s, uint32_t sc2) {
    constexpr int PR = 2 * N + JH;
    constexpr int K4 = 4 * (2 * (N & 1) + JH);
    const uint32_t wq = N < 2 ? s.lo01.x : N < 4 ? s.lo01.y : N < 6 ? s.lo23.x : s.lo23.y;
    uint32_t code = (wq >> K4) & 0x000F000Fu;
    if constexpr (X) {
        const float sc = __uint_as_float(sc2);
        const float a = __uint_as_float(0x4B000000u | (code & 0xFu)) - 8388616.f, b = __uint_as_float(0x4B000000u | (code >> 16)) - 8388616.f;
        __nv_bfloat162 v = __floats2bfloat162_rn(__fmul_rn(a, sc), __fmul_rn(b, sc));
        return *reinterpret_cast<uint32_t*>(&v);
    }
    if (BITS == 5) {
        // (the high bits: weight 2 PR at bit PR and weight 2 PR + 1 at bit PR + 16, moved to bits 4 and 20)
        const uint32_t h = PR >= 4 ? s.hi >> (PR - 4) : s.hi << (4 - PR);
        code |= h & 0x00100010u;
    } else if (BITS == 6) {
        constexpr int SH = 2 * (PR & 7);
        const uint32_t hw = PR < 8 ? s.hi : s.hi2;
        const uint32_t h = SH >= 4 ? hw >> (SH - 4) : hw << (4 - SH);
        code |= h & 0x00300030u;
    }
    code |= 0x43004300u;  // (bf16 128 + code)
    constexpr uint32_t NEG_BIAS = BITS == 4 ? 0xC308C308u : BITS == 5 ? 0xC310C310u : 0xC320C320u;  // -(128 + bias) as bf16: -136, -144, -160
    return sk_fma2(sk_fma2(code, 0x3F803F80u, NEG_BIAS), sc2, 0x80008000u);
}

template <int BITS, int Q, bool X = false, bool WIDE = false>
__device__ __forceinline__ void sk_tile(const SkStep& st, uint32_t s0, uint32_t s1, float (&c)[4], uint32_t b0, uint32_t b1, float (&c2)[4], uint32_t e0, uint32_t e1) {
    uint32_t a[4];
    a[0] = sk_pair<BITS, 2 * Q, 0, X>(st, s0);
    a[1] = sk_pair<BITS, 2 * Q + 1, 0, X>(st, s1);
    a[2] = sk_pair<BITS, 2 * Q, 1, X>(st, s0);
    a[3] = sk_pair<BITS, 2 * Q + 1, 1, X>(st, s1);
    mma16816(c, a, b0, b1);
    if (WIDE) mma16816(c2, a, e0, e1);  // (rows 8 to 15 of x: the same weights, a second tile of columns)
}

// an Int4H scale (bit 15 the sign, bits 11-14 the exponent over the pack's base, bits 0-10 the top of a 12-bit significand) as an fp32's bits
__device__ __forceinline__ uint32_t sk_scale_h(uint32_t h, uint32_t base) {
    return ((h & 0x8000u) << 16) | ((((h >> 11) & 0xFu) + base) << 23) | ((h & 0x7FFu) << 12);
}

template <int BITS, int Q, bool X = false, bool WIDE = false>
__device__ __forceinline__ void sk_tiles(const SkPair& pr, uint32_t sw, float (&c)[4], const uint32_t (&bx)[4], float (&c2)[4], const uint32_t (&ex)[4], uint32_t base) {
    const uint32_t s0 = X ? sk_scale_h(sw & 0xFFFFu, base) : __byte_perm(sw, 0, 0x1010), s1 = X ? sk_scale_h(sw >> 16, base) : __byte_perm(sw, 0, 0x3232);
    sk_tile<BITS, Q, X, WIDE>(pr.a, s0, s1, c, bx[0], bx[1], c2, ex[0], ex[1]);
    sk_tile<BITS, Q, X, WIDE>(pr.b, s0, s1, c, bx[2], bx[3], c2, ex[2], ex[3]);
}

// WIDE: 9 to 16 rows of x (a verification of 8 drafts and more), a second tile of columns on the same decoded weights; a row's sum is the same as at 1 to 8 rows (its own column, the same order).
template <int BITS, bool X, bool WIDE>
__device__ __forceinline__ void skinny_mma_t(const unsigned char* __restrict__ data, const unsigned char* __restrict__ blocks, const unsigned short* __restrict__ x, unsigned short* __restrict__ y, int N, int K, int M, uint32_t base, float (*red)[64][16]) {
    constexpr int SB = BITS == 4 ? 512 : BITS == 5 ? 640 : 768;
    const int ks = K >> 4, pairs = ks >> 1;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const unsigned char* d0 = data + (size_t)blockIdx.x * ks * SB;
    const unsigned char* b0 = blocks + (size_t)blockIdx.x * pairs * 128;
    float acc[4][4], acc2[4][4];
#pragma unroll
    for (int q = 0; q < 4; q++) acc[q][0] = acc[q][1] = acc[q][2] = acc[q][3] = acc2[q][0] = acc2[q][1] = acc2[q][2] = acc2[q][3] = 0.f;
    if (warp < pairs) {
        SkPair cur = sk_load_pair<BITS>(d0, b0, warp, lane);
        for (int p = warp; p < pairs; p += SK_WARPS) {
            SkPair nxt;
            const bool more = p + SK_WARPS < pairs;
            if (more) nxt = sk_load_pair<BITS>(d0, b0, p + SK_WARPS, lane);
            // x as B's columns: column g of the tile is row g of x (zero past the M rows); the lane's two pairs of columns of each step
            uint32_t bx[4] = {0, 0, 0, 0}, ex[4] = {0, 0, 0, 0};
            if (g < M) {
                const unsigned short* xr = x + (size_t)g * K + 32 * p + 2 * t;
#pragma unroll
                for (int i = 0; i < 4; i++) bx[i] = __ldg(reinterpret_cast<const unsigned*>(xr + 8 * i));
            }
            if (WIDE && g + 8 < M) {
                const unsigned short* xr = x + (size_t)(g + 8) * K + 32 * p + 2 * t;
#pragma unroll
                for (int i = 0; i < 4; i++) ex[i] = __ldg(reinterpret_cast<const unsigned*>(xr + 8 * i));
            }
            sk_tiles<BITS, 0, X, WIDE>(cur, cur.sc.x, acc[0], bx, acc2[0], ex, base);
            sk_tiles<BITS, 1, X, WIDE>(cur, cur.sc.y, acc[1], bx, acc2[1], ex, base);
            sk_tiles<BITS, 2, X, WIDE>(cur, cur.sc.z, acc[2], bx, acc2[2], ex, base);
            sk_tiles<BITS, 3, X, WIDE>(cur, cur.sc.w, acc[3], bx, acc2[3], ex, base);
            if (more) cur = nxt;
        }
    }
    // C: tile q, rows 16 q + g and 16 q + 8 + g, columns 2 t and 2 t + 1 (the rows of x): the warps' partial sums in shared memory, added in a fixed order
#pragma unroll
    for (int q = 0; q < 4; q++) {
        red[warp][16 * q + g][2 * t] = acc[q][0];
        red[warp][16 * q + g][2 * t + 1] = acc[q][1];
        red[warp][16 * q + 8 + g][2 * t] = acc[q][2];
        red[warp][16 * q + 8 + g][2 * t + 1] = acc[q][3];
        if (WIDE) {
            red[warp][16 * q + g][8 + 2 * t] = acc2[q][0];
            red[warp][16 * q + g][8 + 2 * t + 1] = acc2[q][1];
            red[warp][16 * q + 8 + g][8 + 2 * t] = acc2[q][2];
            red[warp][16 * q + 8 + g][8 + 2 * t + 1] = acc2[q][3];
        }
    }
    __syncthreads();
    if (threadIdx.x < 64) {
        for (int m = 0; m < M; m++) {
            float v = 0.f;
#pragma unroll
            for (int w = 0; w < SK_WARPS; w++) v += red[w][threadIdx.x][m];
            y[(size_t)m * N + (size_t)blockIdx.x * 64 + threadIdx.x] = __bfloat16_as_ushort(__float2bfloat16_rn(v));
        }
    }
}

// the entry points: the shared partial sums [warps][64 rows of W][16 rows of x], and the wide form where M is over 8
template <int BITS, bool X>
__device__ __forceinline__ void skinny_entry(const unsigned char* data, const unsigned char* blocks, const unsigned short* x, unsigned short* y, int N, int K, int M, uint32_t base) {
    __shared__ float red[SK_WARPS][64][16];
    if (M > 8) skinny_mma_t<BITS, X, true>(data, blocks, x, y, N, K, M, base, red);
    else skinny_mma_t<BITS, X, false>(data, blocks, x, y, N, K, M, base, red);
}
extern "C" __global__ void __launch_bounds__(256, 2) skinnyx4(const unsigned char* data, const unsigned char* blocks, const unsigned short* x, unsigned short* y, int N, int K, int M) { skinny_entry<4, false>(data, blocks, x, y, N, K, M, 0); }
extern "C" __global__ void __launch_bounds__(256, 2) skinnyx5(const unsigned char* data, const unsigned char* blocks, const unsigned short* x, unsigned short* y, int N, int K, int M) { skinny_entry<5, false>(data, blocks, x, y, N, K, M, 0); }
// the exact four-bit layout (Int4H): Int4's steps, its own scales over `base` (the pack's tiers[1])
extern "C" __global__ void __launch_bounds__(256, 2) skinnyx4h(const unsigned char* data, const unsigned char* blocks, const unsigned short* x, unsigned short* y, int N, int K, int M, unsigned base) { skinny_entry<4, true>(data, blocks, x, y, N, K, M, base); }
extern "C" __global__ void __launch_bounds__(256, 2) skinnyx6(const unsigned char* data, const unsigned char* blocks, const unsigned short* x, unsigned short* y, int N, int K, int M) { skinny_entry<6, false>(data, blocks, x, y, N, K, M, 0); }
