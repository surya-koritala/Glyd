// Research-only build: reuse the first-party decoder and product implementations.
// This translation unit is not part of the shipped GPU library or automatic routing.
#include "../../gpu/glyd_gpu.cu"
#include "expand.h"
#include "device.h"

namespace candidate_a {
constexpr uint32_t TAG = 3;
constexpr int64_t BYTES = 768;

struct Compact {
    const uint8_t* data;
    const uint8_t* blocks;
    const int32_t* block_base;
    Tiers ts;
    using St = Step;
    static constexpr bool kTable = true;

    __device__ __forceinline__ void load(St& st, int64_t step, int lane) const {
        const uint8_t* p = data + step * BYTES;
        const uint32_t* digits = (const uint32_t*)p + lane;
        st.pd[0] = __ldg(digits); st.pd[1] = __ldg(digits + 32);
        const uint2* significance = (const uint2*)(p + 256) + lane;
        uint2 lo = __ldg(significance), hi = __ldg(significance + 32);
        st.sw[0] = candidate_a_expand4(lo.x); st.sw[1] = candidate_a_expand4(lo.x >> 16);
        st.sw[2] = candidate_a_expand4(lo.y); st.sw[3] = candidate_a_expand4(lo.y >> 16);
        st.sw[4] = candidate_a_expand4(hi.x); st.sw[5] = candidate_a_expand4(hi.x >> 16);
        st.sw[6] = candidate_a_expand4(hi.y); st.sw[7] = candidate_a_expand4(hi.y >> 16);
        st.blk = (uint32_t)__ldg(block_base + step) & 0x7fffffffu;
        st.end = (uint32_t)__ldg(block_base + step + 1) & 0x7fffffffu;
        const uint32_t* bw = (const uint32_t*)blocks;
        st.win = __ldg(bw + (st.blk >> 2) + lane);
        st.tail = __ldg(bw + (st.end >> 2) - 1 - lane);
        st.tail1 = __ldg(bw + (st.end >> 2));
    }
    __device__ __forceinline__ void decode(const St& st, int lane, uint32_t* scratch,
                                         const uint32_t* table, uint32_t result[16]) const {
        decode_step(st, ts, blocks, lane, scratch, table, result);
    }
    __device__ __forceinline__ void prefetch(int64_t step, int lane) const {
        if (lane < BYTES/128) asm volatile("prefetch.global.L2 [%0];\n" :: "l"(data + step*BYTES + lane*128));
    }
    __device__ __forceinline__ void prefetch_exc(int64_t, int64_t, int) const {}
};

static bool dimensions(int64_t rows, int64_t cols) {
    return rows > 0 && rows <= (1<<20) && cols > 0 && cols <= (1<<20)
        && rows % 64 == 0 && cols % 16 == 0 && rows*cols <= (int64_t(1)<<31);
}
static bool product_dimensions(int64_t rows, int64_t cols, int64_t tokens) {
    return dimensions(rows,cols) && tokens >= 1 && tokens <= 2048 && (tokens <= 64 || cols % 64 == 0);
}
static bool words(const uint32_t* tiers) { return tiers && tiers[0] >> 24 == TAG; }

// Variant 0 selects as the expanded representation does; above 64 tokens 1 forces blocks of 128 tokens by
// two row blocks and 2 blocks of 256 tokens by one (diagnostic comparison only).
static bool variant_ok(int64_t tokens, int64_t variant) { return tokens <= 64 ? variant == 0 : variant >= 0 && variant <= 2; }
static int product(Compact f, int64_t rows, int64_t cols, const uint16_t* x, int64_t tokens, int64_t variant,
                   const uint16_t* bias, uint16_t* y, void* work, size_t bytes, int* counters,
                   cudaStream_t stream, size_t* required) {
    if (tokens <= 64) return mma_gemm_run(f,rows,cols,x,tokens,bias,y,work,bytes,counters,stream,required);
    return mma_gemm_big_any(f,rows,cols,x,tokens,bias,y,variant,work,bytes,counters,stream,required);
}
}

extern "C" __attribute__((visibility("default")))
int glyd_candidate_a_workspace(int64_t rows, int64_t cols, int64_t tokens, int64_t variant, size_t* work, size_t* counters) {
    if (!work || !counters || !candidate_a::product_dimensions(rows,cols,tokens) || !candidate_a::variant_ok(tokens,variant))
        return cudaErrorInvalidValue;
    // One counter per 128 tokens by 64 output rows bounds both large-product block shapes.
    *counters = sizeof(int) * (tokens <= 64 ? rows/64 : ((tokens+127)/128)*(rows/64));
    return candidate_a::product({},rows,cols,nullptr,tokens,variant,nullptr,nullptr,nullptr,0,nullptr,0,work);
}

extern "C" __attribute__((visibility("default")))
int glyd_candidate_a_unpack(const uint8_t* data, const uint8_t* blocks, const int32_t* bases,
                          const uint32_t tiers[3], int64_t rows, int64_t cols, uint16_t* out, cudaStream_t stream) {
    if (!candidate_a::dimensions(rows,cols) || !candidate_a::words(tiers) || !data || !blocks || !bases || !out)
        return cudaErrorInvalidValue;
    return mma_unpack_run(candidate_a::Compact{data,blocks,bases,{tiers[0],tiers[1],tiers[2]}},cols,0,rows,out,0,stream);
}

extern "C" __attribute__((visibility("default")))
int glyd_candidate_a_product(const uint8_t* data, const uint8_t* blocks, const int32_t* bases,
                           const uint32_t tiers[3], int64_t rows, int64_t cols, const uint16_t* x,
                           int64_t tokens, int64_t variant, const uint16_t* bias, uint16_t* y, void* work, size_t bytes,
                           int* counters, cudaStream_t stream) {
    if (!candidate_a::product_dimensions(rows,cols,tokens) || !candidate_a::variant_ok(tokens,variant) || !candidate_a::words(tiers)
        || !data || !blocks || !bases || !x || !y || !counters
        || (uintptr_t)x % 16 || (uintptr_t)y % 16 || (uintptr_t)data % 16) return cudaErrorInvalidValue;
    return candidate_a::product({data,blocks,bases,{tiers[0],tiers[1],tiers[2]}},rows,cols,x,tokens,variant,bias,y,work,bytes,counters,stream,nullptr);
}

// ---- FP8-arithmetic variant (diagnostic research, compute capability 8.9 and later) ----
// Compact Candidate A weights decoded by producer warps and converted in shared memory to e4m3 with one power-of-two
// scale a row (the smallest covering its largest magnitude; values below e4m3's range round to nearest even), X as
// e4m3 with one scale a token, mma.m16n8k32 with fp32 sums, both scales applied to the sums. A separate numerical
// variant: these products are not the exact Candidate A path's above.
// The B fragments keep the decoded order: K positions 4t..4t+3 of each 16 hold columns 2t, 2t+1, 2t+8 and 2t+9, so
// X's bytes are stored in that order (quantize_kernel) and no lane exchanges values.
namespace candidate_a8 {
using candidate_a::Compact;
using f8c::f8_e4m3x2;
using f8c::f8_mma8;
constexpr int PAIRS = BIG_KK / 2;  // k32 groups a stage

template <int CW, int PW, int NB, int RBB, int CR> struct Big8 {
    static constexpr int THREADS = 32 * (CW + PW);
    static constexpr int TM = CR * CW / RBB;               // tokens a unit
    static constexpr int A_BYTES = TM * 64;                // X's e4m3 tile a stage (64 columns a token)
    static constexpr int B_UINT4 = RBB * PAIRS * 4 * 32;   // W's e4m3 fragments a stage: [row block][pair][4][lane]
    static constexpr int SHARED = NB * (A_BYTES + B_UINT4 * 16);
    static constexpr int STEPS = RBB * BIG_KK / PW;        // steps a producer decodes a stage, in consecutive pairs
    static constexpr int CHUNKS = TM * 4 / (32 * PW);      // X's 16-byte chunks a producer thread copies a stage
    static_assert(STEPS % 2 == 0 && CHUNKS * 32 * PW == TM * 4, "producer shapes");
};

// Two bf16 (one register) times inv (a power of two, exact) as e4m3 bytes, rounded to nearest even.
__device__ __forceinline__ uint32_t e4m3_pair(uint32_t r, float inv) {
    return f8_e4m3x2(__uint_as_float(r << 16) * inv, __uint_as_float(r & 0xffff0000u) * inv);
}
// A row's 64-byte X tile: 16-byte chunk c of row r at chunk c ^ ((r >> 1) & 3) (eight rows' ldmatrix reads in distinct banks).
__device__ __forceinline__ uint32_t xchunk(int r, int c) { return (uint32_t)(r * 64 + ((c ^ ((r >> 1) & 3)) << 4)); }

// Seven-bit weights (Candidate A rounded to e3m3 with one power-of-two scale a row and 128 columns). Each weight is
// stored as the e4m3 byte of its e3m3 value times 2^-4, which is its e3m3 code with a zero inserted as the top exponent
// bit (subnormals included), so bit 6 of every byte is zero. A lane's 8 B-fragment registers of a 16-column step
// (rows nn * 8 + g, columns 2t, 2t + 1, 2t + 8, 2t + 9: B8[2nn + h] of the decoded path for step h of a k32 pair) are
// one group: seven words, the eighth's bits 0-5 and 7 in their bytes' bit 6. Words: uint32 [row block][pair][step]
// [7][lane]; exps: int8 [row block][column / 128][g][nn], the scale's power of two including the 2^4.
struct Seven {
    const uint32_t* words;
    const int8_t* exps;
};
__device__ __forceinline__ void unpack7(const uint32_t* w, uint32_t (&r)[8]) {
    uint32_t last = 0;
#pragma unroll
    for (int i = 0; i < 7; i++) {
        r[i] = w[i] & 0xbfbfbfbfu;
        last |= i < 6 ? (w[i] >> (6 - i)) & (0x01010101u << i) : (w[i] << 1) & 0x80808080u;
    }
    r[7] = last;
}
__device__ __forceinline__ float pow2(int e) { return __int_as_float((e + 127) << 23); }  // normal range only

// Seven-bit weights for the engine's small-M BF16 products (mma_gemm_run, X itself as the A operand): a step's group
// unpacked and each byte made its BF16 weight exactly (its bits placed as a float's are its value times 2^-120; times
// 2^(exp + 120), exps of -117 to 7).
struct Seven16 {
    const uint32_t* words;
    const int8_t* exps;
    int64_t K;
    struct St {
        uint32_t w[7];
        uint2 e;
    };
    static constexpr bool kTable = true;  // (the generic kernel's table and scratch go unused)
    __device__ __forceinline__ void load(St& st, int64_t step, int lane) const {
        const int64_t KS = K / 16, rb = step / KS, s = step - rb * KS;
        const uint32_t* p = words + ((rb * (K / 32) + s / 2) * 2 + (s & 1)) * 7 * 32 + lane;
#pragma unroll
        for (int i = 0; i < 7; i++) st.w[i] = __ldg(p + i * 32);
        st.e = __ldg((const uint2*)(exps + ((rb * (K / 128) + s / 8) * 8 + (lane >> 2)) * 8));
    }
    __device__ __forceinline__ void decode(const St& st, int, uint32_t*, const uint32_t*, uint32_t result[16]) const {
        uint32_t r[8];
        unpack7(st.w, r);
#pragma unroll
        for (int nn = 0; nn < 8; nn++) {
            const float sc = pow2((int)(int8_t)(((nn < 4 ? st.e.x : st.e.y) >> (8 * (nn & 3))) & 0xffu) + 120);
            auto w = [&](int k) {
                const uint32_t b = r[nn] >> (8 * k);
                return __uint_as_float(((b & 0x80u) << 24) | ((b & 0x7fu) << 20)) * sc;
            };
            const __nv_bfloat162 lo = __floats2bfloat162_rn(w(0), w(1)), hi = __floats2bfloat162_rn(w(2), w(3));
            result[2 * nn] = *(const uint32_t*)&lo;
            result[2 * nn + 1] = *(const uint32_t*)&hi;
        }
    }
    __device__ __forceinline__ void prefetch(int64_t, int) const {}
    __device__ __forceinline__ void prefetch_exc(int64_t, int64_t, int) const {}
};

// A7: seven-bit weights (sv; f and sw unused). Producers unpack instead of decoding; consumers sum each 128 columns
// apart and add them times that block's scale, so the output applies only the token's scale.
template <int CW, int PW, int NB, int RBB, int CR, bool LIVE, bool A7 = false>
__global__ void __launch_bounds__(Big8<CW, PW, NB, RBB, CR>::THREADS, 1) sk8_kernel(Compact f, Seven sv, const float* __restrict__ sw, int64_t O, int64_t K, int64_t M, const uint8_t* __restrict__ X8, const float* __restrict__ sx, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ Y, float* __restrict__ parts, int* __restrict__ done) {
    using C = Big8<CW, PW, NB, RBB, CR>;
    constexpr int NN = CR / 8, TW = C::TM / 64, HR = 64 / CR;
    constexpr int64_t SLOT = C::TM * 64 * RBB;
    extern __shared__ uint4 smem[];  // X's tiles [NB][TM][4 chunks], then W's fragments [NB][B_UINT4]
    __shared__ uint32_t tab[256];
    __shared__ int last;
    if constexpr (!A7) fill_groups(tab);
    __syncthreads();
    const uint32_t a_base = (uint32_t)__cvta_generic_to_shared(smem);
    uint4* Bs = smem + NB * C::A_BYTES / 16;
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int64_t KS = K / 16, RB = O / 64, tiles = (M + C::TM - 1) / C::TM, nb = gridDim.x;
    const int S = (int)(KS / BIG_KK);
    const int64_t total = tiles * ((RB + RBB - 1) / RBB) * S, it0 = blockIdx.x * total / nb, n = (blockIdx.x + 1) * total / nb - it0;
    const int64_t u0 = it0 / S;
    const int s0 = (int)(it0 - u0 * S);
    if (warp >= CW) {
        int pt = tid - 32 * CW, p = warp - CW, s = s0;
        int64_t tile = u0 % tiles, pair = u0 / tiles;
        auto row_block = [&](int64_t pr, int u) { return min(pr * RBB + u / BIG_KK, RB - 1); };
        Compact::St st[A7 ? 1 : C::STEPS];
        uint32_t pk[A7 ? C::STEPS / 2 : 1][14];  // A7: the next stage's packed words, a k32 pair (two steps' groups) each
        auto d_load = [&](int64_t pr, int ss) {
            if constexpr (A7) {
#pragma unroll
                for (int i = 0; i < C::STEPS / 2; i++) {
                    const int u = p * C::STEPS + 2 * i;
                    const uint32_t* src = sv.words + ((row_block(pr, u) * (K / 32) + ss * PAIRS + u % BIG_KK / 2) * 14) * 32 + lane;
#pragma unroll
                    for (int w = 0; w < 14; w++) pk[i][w] = __ldg(src + w * 32);
                }
            } else {
#pragma unroll
                for (int i = 0; i < C::STEPS; i++) {
                    int u = p * C::STEPS + i;
                    f.load(st[i], row_block(pr, u) * KS + ss * BIG_KK + u % BIG_KK, lane);
                }
            }
        };
        if (n > 0) d_load(pair, s);
        for (int64_t j = 0; j < n; j++) {
            int ns = s + 1;
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
                int id = pt + 32 * PW * i, r = id >> 2, c = id & 3;
                int64_t m = tile * C::TM + r;
                cp_async16(slot + xchunk(r, c), X8 + (m < M ? m : 0) * K + col + c * 16, m < M ? 16 : 0);
            }
            asm volatile("cp.async.commit_group;\n" ::);
            if constexpr (A7) {
                uint32_t cur[C::STEPS / 2][14];
#pragma unroll
                for (int i = 0; i < C::STEPS / 2; i++)
#pragma unroll
                    for (int w = 0; w < 14; w++) cur[i][w] = pk[i][w];
                if (j + 1 < n) d_load(npair, ns);
#pragma unroll
                for (int i = 0; i < C::STEPS / 2; i++) {
                    uint32_t r0[8], r1[8], B8[16];
                    unpack7(cur[i], r0);
                    unpack7(cur[i] + 7, r1);
#pragma unroll
                    for (int nn = 0; nn < 8; nn++) B8[2 * nn] = r0[nn], B8[2 * nn + 1] = r1[nn];
                    uint4* d = Bs + b * C::B_UINT4 + (p * C::STEPS / 2 + i) * 4 * 32 + lane;
#pragma unroll
                    for (int q = 0; q < 4; q++) d[q * 32] = make_uint4(B8[4 * q], B8[4 * q + 1], B8[4 * q + 2], B8[4 * q + 3]);
                }
            } else {
            Compact::St cur[C::STEPS];
#pragma unroll
            for (int i = 0; i < C::STEPS; i++) cur[i] = st[i];
            const int64_t cur_pair = pair;
            if (j + 1 < n) d_load(npair, ns);
#pragma unroll
            for (int i = 0; i < C::STEPS; i += 2) {
                const int u = p * C::STEPS + i;  // an even step of the stage: it and the next form a pair
                uint4* sl = Bs + b * C::B_UINT4 + (u / 2) * 4 * 32;
                uint32_t R0[16], R1[16], B8[16];
                f.decode(cur[i], lane, (uint32_t*)sl, tab, R0);  // the pair's slot is the decode's scratch until written
                __syncwarp();
                f.decode(cur[i + 1], lane, (uint32_t*)sl, tab, R1);
                __syncwarp();
                const float* rs = sw + row_block(cur_pair, u) * 64 + (lane >> 2);
#pragma unroll
                for (int nn = 0; nn < 8; nn++) {
                    const float inv = 1.f / __ldg(rs + nn * 8);
                    B8[2 * nn] = e4m3_pair(R0[2 * nn], inv) | (e4m3_pair(R0[2 * nn + 1], inv) << 16);
                    B8[2 * nn + 1] = e4m3_pair(R1[2 * nn], inv) | (e4m3_pair(R1[2 * nn + 1], inv) << 16);
                }
                uint4* d = sl + lane;
#pragma unroll
                for (int q = 0; q < 4; q++) d[q * 32] = make_uint4(B8[4 * q], B8[4 * q + 1], B8[4 * q + 2], B8[4 * q + 3]);
            }
            }
            asm volatile("cp.async.wait_group 0;\n" ::);
            bar_arrive<C::THREADS>(1 + b);  // stage j is ready
            s = ns, tile = ntile, pair = npair;
        }
        return;
    }
    int g = lane >> 2, t = lane & 3, wm = warp % TW, hf = warp / TW % HR, rbl = warp / (TW * HR);
    float acc[4][NN][4], blk[4][NN][4];  // A7: blk holds the current 128 columns' sums
#pragma unroll
    for (int mt = 0; mt < 4; mt++)
#pragma unroll
        for (int nn = 0; nn < NN; nn++)
#pragma unroll
            for (int q = 0; q < 4; q++) acc[mt][nn][q] = blk[mt][nn][q] = 0.f;
    const int a_r = wm * 64 + (lane & 7) + ((lane >> 3) & 1) * 8, a_half = lane >> 4;
    uint32_t fa[2][4][4];
    uint4 fb[2][NN / 2];
    auto frags = [&](int64_t j, int kp, int fi) {
        int bi = (int)(j % NB);
        uint32_t slot = a_base + (uint32_t)bi * C::A_BYTES;
        const uint4* bsrc = Bs + bi * C::B_UINT4 + rbl * PAIRS * 4 * 32 + lane;
#pragma unroll
        for (int mt = 0; mt < 4; mt++) ldmatrix_x4(fa[fi][mt], slot + xchunk(a_r + mt * 16, 2 * kp + a_half));
#pragma unroll
        for (int q = 0; q < NN / 2; q++) fb[fi][q] = bsrc[(kp * 4 + NN / 2 * hf + q) * 32];
    };
    auto next = [&](int64_t j, int fi) {
        bar_sync<C::THREADS>(1 + (int)(j % NB));
        frags(j, 0, fi);
    };
    if (n > 0) next(0, 0);
    int s = s0;
    for (int64_t j = 0, u = u0; j < n; j++) {
        bool end = s == S - 1 || j + 1 == n;
        const int64_t left = M - u % tiles * C::TM - wm * 64;
        const int nmt = !LIVE || left >= 64 ? 4 : left <= 0 ? 0 : (int)((left + 15) >> 4);
#pragma unroll
        for (int kp = 0; kp < PAIRS; kp++) {
            int fi = kp & 1;
            if (kp + 1 < PAIRS) frags(j, kp + 1, fi ^ 1);
            else if (j + 1 < n) next(j + 1, fi ^ 1);
#pragma unroll
            for (int mt = 0; mt < 4; mt++)
                if (mt < nmt)
#pragma unroll
                    for (int q = 0; q < NN / 2; q++) {
                        f8_mma8(A7 ? blk[mt][2 * q] : acc[mt][2 * q], fa[fi][mt][0], fa[fi][mt][1], fa[fi][mt][2], fa[fi][mt][3], fb[fi][q].x, fb[fi][q].y);
                        f8_mma8(A7 ? blk[mt][2 * q + 1] : acc[mt][2 * q + 1], fa[fi][mt][0], fa[fi][mt][1], fa[fi][mt][2], fa[fi][mt][3], fb[fi][q].z, fb[fi][q].w);
                    }
        }
        if (j + NB < n) bar_arrive<C::THREADS>(1 + NB + (int)(j % NB));
        if constexpr (A7) {
            if ((s & 1) || end) {  // a 128-column block (two stages) ends, or this block's share of it
                const int64_t rb = min(u / tiles * RBB + rbl, RB - 1);
                const int8_t* e = sv.exps + ((rb * (K / 128) + s / 2) * 8 + 2 * t) * 8 + CR * hf / 8;  // rows 2t and 2t + 1 of each n-tile
#pragma unroll
                for (int nn = 0; nn < NN; nn++) {
                    const float s0 = pow2(__ldg(e + nn)), s1 = pow2(__ldg(e + 8 + nn));
#pragma unroll
                    for (int mt = 0; mt < 4; mt++) {
                        acc[mt][nn][0] += blk[mt][nn][0] * s0;
                        acc[mt][nn][1] += blk[mt][nn][1] * s1;
                        acc[mt][nn][2] += blk[mt][nn][2] * s0;
                        acc[mt][nn][3] += blk[mt][nn][3] * s1;
#pragma unroll
                        for (int q = 0; q < 4; q++) blk[mt][nn][q] = 0.f;
                    }
                }
            }
        }
        if (++s == S) s = 0;
        if (!end) continue;
        int64_t first = block_of_step(u * S, nb, total), fin = block_of_step(u * S + S - 1, nb, total);
        bool out = first == fin;
        if (!out) {  // unscaled sums in slots; the last block adds them in block order, as mma_gemm_sk_kernel does
            auto slot_of = [&](int64_t bb) { return (float4*)(parts + (2 * bb + (bb == first && first * total / nb < u * S)) * SLOT) + tid; };
            float4* mine = slot_of(blockIdx.x);
#pragma unroll
            for (int mt = 0; mt < 4; mt++)
#pragma unroll
                for (int nn = 0; nn < NN; nn++) mine[(mt * NN + nn) * 32 * CW] = make_float4(acc[mt][nn][0], acc[mt][nn][1], acc[mt][nn][2], acc[mt][nn][3]);
            __threadfence();
            bar_sync<32 * CW>(1 + 2 * NB);
            if (tid == 0) {
                last = atomicAdd(done + u, 1) == fin - first;
                if (last) done[u] = 0;
            }
            bar_sync<32 * CW>(1 + 2 * NB);
            out = last;
            if (out) {
                __threadfence();
#pragma unroll
                for (int mt = 0; mt < 4; mt++)
#pragma unroll
                    for (int nn = 0; nn < NN; nn++)
#pragma unroll
                        for (int q = 0; q < 4; q++) acc[mt][nn][q] = 0.f;
#pragma unroll
                for (int mt = 0; mt < 4; mt++)
                    for (int64_t bb = first; bb <= fin; bb++) {
                        const float4* sl = slot_of(bb) + mt * NN * 32 * CW;
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
                        const float xs = __ldg(sx + m);
                        float v0 = acc[mt][nn][2 * h] * xs, v1 = acc[mt][nn][2 * h + 1] * xs;
                        if constexpr (!A7) v0 *= __ldg(sw + o), v1 *= __ldg(sw + o + 1);
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

template <int CW, int PW, int NB, int RBB, int CR, bool LIVE, bool A7 = false>
static int sk8_run(const Compact& f, Seven sv, const float* sw, int64_t O, int64_t K, const uint8_t* x8, const float* sx, int64_t M, const uint16_t* bias, uint16_t* y, void* ws, size_t ws_bytes, int* done, cudaStream_t cs, size_t* need) {
    using C = Big8<CW, PW, NB, RBB, CR>;
    auto kernel = sk8_kernel<CW, PW, NB, RBB, CR, LIVE, A7>;
    static std::atomic<int> known[MAX_DEVICES];
    int dev = current_device();
    if constexpr (C::SHARED > 48 * 1024) {  // above the default dynamic limit: opt in once a device
        static std::atomic<bool> big[MAX_DEVICES];
        if (dev < MAX_DEVICES && !big[dev].exchange(true)) cudaFuncSetAttribute((const void*)kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SHARED);
    }
    int64_t S = K / 16 / BIG_KK, units = (M + C::TM - 1) / C::TM * ((O / 64 + RBB - 1) / RBB), total = units * S;
    int64_t nb = std::min(per_sm((const void*)kernel, C::THREADS, C::SHARED, known, dev) * sm_count(dev), std::min(8 * units, total / 4));
    nb = total ? std::max<int64_t>(nb, 1) : 0;
    bool shared = nb && (total % nb || total / nb % S);
    size_t bytes = shared ? (size_t)(2 * nb * C::TM * 64 * RBB) * sizeof(float) : 0;
    if (need) {
        *need = bytes;
        return cudaGetLastError();
    }
    if (!fits(ws, ws_bytes, bytes) || (shared && !done)) return cudaErrorInvalidValue;
    if (nb) kernel<<<nb, C::THREADS, C::SHARED, cs>>>(f, sv, sw, O, K, M, x8, sx, bf(bias), bf(y), (float*)ws, done);
    return cudaGetLastError();
}

// Variants: 1 units of 64 tokens by two row blocks (consumers of 32 rows, empty m-tiles skipped), 2 of 128 tokens by two
// row blocks, 3 of 256 tokens by one (eight consumers, two producers); 0: 1 to 64 tokens, then 2 to 128, then 3.
static int product(const Compact& f, const float* sw, int64_t O, int64_t K, const uint8_t* x8, const float* sx, int64_t M, int64_t variant, const uint16_t* bias, uint16_t* y, void* ws, size_t bytes, int* done, cudaStream_t cs, size_t* need) {
    if (variant == 0) variant = M <= 64 ? 1 : M <= 128 ? 2 : 3;
    if (variant == 1) return sk8_run<4, 4, 3, 2, 32, true>(f, {}, sw, O, K, x8, sx, M, bias, y, ws, bytes, done, cs, need);
    if (variant == 2) return sk8_run<4, 4, 3, 2, 64, false>(f, {}, sw, O, K, x8, sx, M, bias, y, ws, bytes, done, cs, need);
    return sk8_run<8, 2, 2, 1, 32, false>(f, {}, sw, O, K, x8, sx, M, bias, y, ws, bytes, done, cs, need);
}

// X as e4m3 with one scale a token (its largest magnitude over 448), each 16 columns' bytes in the B fragments' order
// (byte 4t + j holds column 2t, 2t + 1, 2t + 8, 2t + 9 for j = 0..3). A block a token.
__global__ void __launch_bounds__(256) quantize_kernel(const __nv_bfloat16* __restrict__ X, int64_t K, uint8_t* __restrict__ X8, float* __restrict__ sx) {
    __shared__ float red[8];
    const int64_t m = blockIdx.x;
    const uint4* row = (const uint4*)(X + m * K);
    float mx = 0.f;
    for (int64_t i = threadIdx.x; i < K / 8; i += blockDim.x) {
        uint4 v = row[i];
        const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int q = 0; q < 4; q++) mx = fmaxf(mx, fmaxf(fabsf(__uint_as_float(w[q] << 16)), fabsf(__uint_as_float(w[q] & 0xffff0000u))));
    }
#pragma unroll
    for (int d = 16; d >= 1; d >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, d));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = mx;
    __syncthreads();
    mx = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) mx = fmaxf(mx, red[i]);
    const float inv = mx > 0.f ? 448.f / mx : 0.f;
    for (int64_t gi = threadIdx.x; gi < K / 16; gi += blockDim.x) {
        uint4 a = row[2 * gi], b = row[2 * gi + 1];  // columns 0-7, 8-15 of the group
        const uint32_t lo[4] = {a.x, a.y, a.z, a.w}, hi[4] = {b.x, b.y, b.z, b.w};
        uint32_t o[4];
#pragma unroll
        for (int q = 0; q < 4; q++) o[q] = e4m3_pair(lo[q], inv) | (e4m3_pair(hi[q], inv) << 16);
        ((uint4*)(X8 + m * K))[gi] = make_uint4(o[0], o[1], o[2], o[3]);
    }
    if (threadIdx.x == 0) sx[m] = mx / 448.f;
}

// Every weight as its e4m3 byte (row-major, natural column order) for comparison with a host conversion. A warp a step.
__global__ void __launch_bounds__(256) unpack8_kernel(Compact f, const float* __restrict__ sw, int64_t O, int64_t K, uint8_t* __restrict__ out) {
    __shared__ uint32_t tab[256];
    __shared__ __align__(16) uint32_t scratch[8][S2_BYTES / 4];
    fill_groups(tab);
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int64_t KS = K / 16, step = (int64_t)blockIdx.x * 8 + warp;
    if (step >= O / 64 * KS) return;
    Compact::St st;
    uint32_t R[16];
    f.load(st, step, lane);
    f.decode(st, lane, scratch[warp], tab, R);
    const int64_t rb = step / KS, s = step - rb * KS;
#pragma unroll
    for (int nn = 0; nn < 8; nn++) {
        const int64_t r = rb * 64 + nn * 8 + g;
        const float inv = 1.f / __ldg(sw + r);
        uint8_t* p = out + r * K + s * 16 + 2 * t;
        *(uint16_t*)p = (uint16_t)e4m3_pair(R[2 * nn], inv);
        *(uint16_t*)(p + 8) = (uint16_t)e4m3_pair(R[2 * nn + 1], inv);
    }
}

static bool fp8_device() {
    int dev = current_device();
    int cc = attribute(cudaDevAttrComputeCapabilityMajor, dev) * 10 + attribute(cudaDevAttrComputeCapabilityMinor, dev);
    return cc >= 89;
}
static bool dims(int64_t rows, int64_t cols, int64_t tokens) {
    return candidate_a::dimensions(rows, cols) && cols % 64 == 0 && tokens >= 1 && tokens <= 2048;
}

// Seven-bit products. FP8 arithmetic: 1 units of 64 tokens by two row blocks, empty m-tiles skipped; 2 of 128 tokens by
// one; 4 of 64 tokens by two row blocks with eight 16-row consumers; 5 of 128 tokens by two row blocks; 6 of 256 tokens by
// one; 7 of 128 tokens by one with eight 16-row consumers (4-7 skip empty m-tiles; 5 and 6 exceed their 168 registers and
// spill). 3: BF16 arithmetic from X itself, up to 64 tokens (the engine's small-M product). 0: 3 up
// to 16 tokens, 1 up to 64, then 2. Each FP8 consumer holds a second set of sums for the block scales.
static int product7(Seven sv, int64_t O, int64_t K, const uint16_t* x, const uint8_t* x8, const float* sx, int64_t M, int64_t variant, const uint16_t* bias, uint16_t* y, void* ws, size_t bytes, int* done, cudaStream_t cs, size_t* need) {
    if (variant == 0) variant = M <= 16 ? 3 : M <= 64 ? 1 : 2;
    switch (variant) {
    case 1: return sk8_run<4, 4, 3, 2, 32, true, true>({}, sv, nullptr, O, K, x8, sx, M, bias, y, ws, bytes, done, cs, need);
    case 2: return sk8_run<4, 2, 2, 1, 32, false, true>({}, sv, nullptr, O, K, x8, sx, M, bias, y, ws, bytes, done, cs, need);
    case 3: return M <= 64 ? mma_gemm_run(Seven16{sv.words, sv.exps, K}, O, K, x, M, bias, y, ws, bytes, done, cs, need) : (int)cudaErrorInvalidValue;
    case 4: return sk8_run<8, 4, 3, 2, 16, true, true>({}, sv, nullptr, O, K, x8, sx, M, bias, y, ws, bytes, done, cs, need);
    case 5: return sk8_run<8, 2, 3, 2, 32, true, true>({}, sv, nullptr, O, K, x8, sx, M, bias, y, ws, bytes, done, cs, need);
    case 6: return sk8_run<8, 2, 2, 1, 32, true, true>({}, sv, nullptr, O, K, x8, sx, M, bias, y, ws, bytes, done, cs, need);
    default: return sk8_run<8, 2, 3, 1, 16, true, true>({}, sv, nullptr, O, K, x8, sx, M, bias, y, ws, bytes, done, cs, need);
    }
}

// Every seven-bit weight as its e4m3 byte (row-major, natural column order). A thread a lane's row block and k32 pair.
__global__ void __launch_bounds__(256) unpack7_kernel(Seven sv, int64_t O, int64_t K, uint8_t* __restrict__ out) {
    const int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x, unit = id >> 5, NP = K / 32;
    const int lane = (int)(id & 31), g = lane >> 2, t = lane & 3;
    if (unit >= O / 64 * NP) return;
    const int64_t rb = unit / NP, gp = unit - rb * NP;
    uint32_t pk[14], r0[8], r1[8], r[16];
#pragma unroll
    for (int w = 0; w < 14; w++) pk[w] = __ldg(sv.words + (unit * 14 + w) * 32 + lane);
    unpack7(pk, r0);
    unpack7(pk + 7, r1);
#pragma unroll
    for (int nn = 0; nn < 8; nn++) r[2 * nn] = r0[nn], r[2 * nn + 1] = r1[nn];
#pragma unroll
    for (int i = 0; i < 16; i++) {  // register 2nn + h: row nn * 8 + g, columns 2t, 2t + 1, 2t + 8, 2t + 9 of step h
        uint8_t* q = out + (rb * 64 + i / 2 * 8 + g) * K + gp * 32 + (i & 1) * 16 + 2 * t;
        *(uint16_t*)q = (uint16_t)r[i];
        *(uint16_t*)(q + 8) = (uint16_t)(r[i] >> 16);
    }
}
static bool dims7(int64_t rows, int64_t cols, int64_t tokens) {
    return candidate_a::dimensions(rows, cols) && cols % 128 == 0 && tokens >= 1 && tokens <= 2048;
}
}  // namespace candidate_a8

extern "C" __attribute__((visibility("default")))
int glyd_candidate_a8_quantize(const uint16_t* x, int64_t tokens, int64_t cols, uint8_t* x8, float* sx, cudaStream_t stream) {
    if (!x || !x8 || !sx || tokens < 1 || tokens > 2048 || cols < 16 || cols % 16 || (uintptr_t)x % 16 || (uintptr_t)x8 % 16) return cudaErrorInvalidValue;
    if (!candidate_a8::fp8_device()) return cudaErrorInvalidDevice;
    candidate_a8::quantize_kernel<<<(unsigned)tokens, 256, 0, stream>>>(bf(x), cols, x8, sx);
    return cudaGetLastError();
}

extern "C" __attribute__((visibility("default")))
int glyd_candidate_a8_unpack(const uint8_t* data, const uint8_t* blocks, const int32_t* bases, const uint32_t tiers[3], int64_t rows, int64_t cols, const float* scale, uint8_t* out, cudaStream_t stream) {
    if (!candidate_a::dimensions(rows, cols) || !candidate_a::words(tiers) || !data || !blocks || !bases || !scale || !out) return cudaErrorInvalidValue;
    if (!candidate_a8::fp8_device()) return cudaErrorInvalidDevice;
    const int64_t steps = rows / 64 * (cols / 16);
    candidate_a8::unpack8_kernel<<<(unsigned)((steps + 7) / 8), 256, 0, stream>>>(candidate_a::Compact{data, blocks, bases, {tiers[0], tiers[1], tiers[2]}}, scale, rows, cols, out);
    return cudaGetLastError();
}

extern "C" __attribute__((visibility("default")))
int glyd_candidate_a8_workspace(int64_t rows, int64_t cols, int64_t tokens, int64_t variant, size_t* work, size_t* counters) {
    if (!work || !counters || !candidate_a8::dims(rows, cols, tokens) || variant < 0 || variant > 3) return cudaErrorInvalidValue;
    if (!candidate_a8::fp8_device()) return cudaErrorInvalidDevice;
    // One counter per 64 tokens by 64 output rows bounds every variant's units.
    *counters = sizeof(int) * ((tokens + 63) / 64) * (rows / 64);
    return candidate_a8::product({}, nullptr, rows, cols, nullptr, nullptr, tokens, variant, nullptr, nullptr, nullptr, 0, nullptr, 0, work);
}

extern "C" __attribute__((visibility("default")))
int glyd_candidate_a8_product(const uint8_t* data, const uint8_t* blocks, const int32_t* bases, const uint32_t tiers[3], int64_t rows, int64_t cols, const float* scale, const uint8_t* x8, const float* sx, int64_t tokens, int64_t variant, const uint16_t* bias, uint16_t* y, void* work, size_t bytes, int* counters, cudaStream_t stream) {
    if (!candidate_a8::dims(rows, cols, tokens) || variant < 0 || variant > 3 || !candidate_a::words(tiers) || !data || !blocks || !bases || !scale || !x8 || !sx || !y || !counters
        || (uintptr_t)x8 % 16 || (uintptr_t)y % 16 || (uintptr_t)data % 16) return cudaErrorInvalidValue;
    if (!candidate_a8::fp8_device()) return cudaErrorInvalidDevice;
    return candidate_a8::product(candidate_a::Compact{data, blocks, bases, {tiers[0], tiers[1], tiers[2]}}, scale, rows, cols, x8, sx, tokens, variant, bias, y, work, bytes, counters, stream, nullptr);
}

extern "C" __attribute__((visibility("default")))
int glyd_candidate_a7_unpack(const void* words, int64_t rows, int64_t cols, uint8_t* out, cudaStream_t stream) {
    if (!candidate_a8::dims7(rows, cols, 1) || !words || !out || (uintptr_t)words % 8) return cudaErrorInvalidValue;
    const int64_t threads = rows / 64 * (cols / 32) * 32;
    candidate_a8::unpack7_kernel<<<(unsigned)((threads + 255) / 256), 256, 0, stream>>>({(const uint32_t*)words, nullptr}, rows, cols, out);
    return cudaGetLastError();
}

extern "C" __attribute__((visibility("default")))
int glyd_candidate_a7_workspace(int64_t rows, int64_t cols, int64_t tokens, int64_t variant, size_t* work, size_t* counters) {
    if (!work || !counters || !candidate_a8::dims7(rows, cols, tokens) || variant < 0 || variant > 7 || (variant == 3 && tokens > 64)) return cudaErrorInvalidValue;
    if (!candidate_a8::fp8_device()) return cudaErrorInvalidDevice;
    *counters = sizeof(int) * ((tokens + 63) / 64) * (rows / 64);
    return candidate_a8::product7({}, rows, cols, nullptr, nullptr, nullptr, tokens, variant, nullptr, nullptr, nullptr, 0, nullptr, 0, work);
}

extern "C" __attribute__((visibility("default")))
// x: the BF16 input (variant 3, and 0 up to 16 tokens); x8, sx: its e4m3 bytes and token scales (the FP8 variants).
int glyd_candidate_a7_product(const void* words, const int8_t* exps, int64_t rows, int64_t cols, const uint16_t* x, const uint8_t* x8, const float* sx, int64_t tokens, int64_t variant, const uint16_t* bias, uint16_t* y, void* work, size_t bytes, int* counters, cudaStream_t stream) {
    const bool bf16 = variant == 3 || (variant == 0 && tokens <= 16);
    if (!candidate_a8::dims7(rows, cols, tokens) || variant < 0 || variant > 7 || (variant == 3 && tokens > 64) || !words || !exps || !y || !counters
        || (bf16 ? !x || (uintptr_t)x % 16 : !x8 || !sx || (uintptr_t)x8 % 16) || (uintptr_t)words % 16 || (uintptr_t)exps % 8 || (uintptr_t)y % 16) return cudaErrorInvalidValue;
    if (!candidate_a8::fp8_device()) return cudaErrorInvalidDevice;
    return candidate_a8::product7({(const uint32_t*)words, exps}, rows, cols, x, x8, sx, tokens, variant, bias, y, work, bytes, counters, stream, nullptr);
}
