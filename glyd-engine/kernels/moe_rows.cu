// The sparse experts' kernels (included by engine.cu after moe.cu; src/moe_rows.rs runs them): the second product of a relu squared expert for the neurons that are not zero. After relu squared 92% of an expert's
// 2,688 activations are exactly 0 for a token (benchmarks/weights/offload-probe-2026-10-10), and a zero activation needs no column of the down matrix: an expert's output is the sum over the neurons n that are
// not zero of a_n times down's column n, which the row layout (glyd-gpu's pack_rows) holds as a row of its own, the weights of neuron n to all the outputs, in runs of 32 with a bf16 scale each. The weights
// are those of the block layouts, decoded as they decode a pair (the library's Int4, Int5 and Int6: (128 + u) as a bf16 pair, less 128 + bias, times the run's scale, two bf16x2 instructions): round_bf16((u - bias) s)
// for a code u and its run's scale s, so that a product a_n w is exact in fp32; the sums are fp32, in an order of their own (not the tensor cores').
//
// moe_down_rows{4,5,6} (a block a pair, or a part of one): the pair's activations (row i of the first product's output, in the plan's order, relu squared applied) are listed in order, the neurons whose bf16 is not zero,
// in shared memory; the block's warps take every 8th of its share of the list (the share: every split-th neuron of the list, from s) and add, for each, the neuron's row times the activation into the 32 outputs
// a lane has (lane l: the run of 32 outputs os 1,024 + 32 l on, the row's codes and scale for that run read by it alone); the warps' sums are added in shared memory in the warps' order and written to
// part [pair j][s][hidden] (fp32). Nothing here depends on the values but through the list: the grid is fixed by the tokens, so a decode step captures into a CUDA graph; and nothing is added in an order
// that depends on timing (no atomics): a launch gives the same bits every time.
// moe_rows_sum (a block 64 outputs of a token): y [token][o] = bf16 of the sum over the token's choices c (ids[token k + c], the router's: an expert, else none), in order within four groups of them, of weight[c]
// times the sum of the choice's parts over s (parts that an earlier launch for the layer's other pairs wrote, where the expert cache computes a layer in waves), and then extra [token][o] (fp32; none where extra is
// null): the sums of the pairs that the host computed for experts that are not in the cache's places, which those pairs' ids (not an expert here: -1) leave out of the groups. One rounding.
//
// The grid of moe_down_rows: (pairs, split, hidden / 1,024 rounded up); 256 threads; the pairs in the plan's order (a block past the plan's pairs does nothing; a pair routed nowhere is not in the plan).

#define MOE_ROWS_WARPS 8         // warps a block
#define MOE_ROWS_OUT 1024        // outputs a block takes: a warp's 32 lanes, each a run of 32
#define MOE_ROWS_MAX_INTER 4096  // neurons a block lists (a word each in shared memory)
#define MOE_ROWS_UNROLL 4        // neurons a warp has loaded before it decodes them

static_assert(MOE_ROWS_MAX_INTER * 4 <= MOE_ROWS_WARPS * (MOE_ROWS_OUT + 32) * 4, "the list of neurons fits the shared memory the warps' sums use");
static_assert(MOE_ROWS_MAX_INTER <= MOE_ROWS_WARPS * 32 * 2 * 8, "a thread holds at most 2 chunks of 8 activations");
static_assert(MOE_ROWS_MAX_INTER <= 65536, "a neuron is the low 16 bits of its entry in the list");

// One block of moe_down_rows for BITS (4, 5 or 6) a code. a: the first product's rows [pairs, inter] bf16 after relu squared; plan: Lib::moe_way's [2 + 2 E + pairs] int32; ids: [tokens, k] int64, the place of pair j's
// expert in data and scales (the expert's own number where they hold every expert, else the expert cache's place for it); data and scales: the experts' (or places') codes and scales, `data_stride` and `scale_stride`
// bytes apart; part: [pairs][split][hidden] fp32. hidden a multiple of 32, inter of 8, at most MOE_ROWS_MAX_INTER.
template <int BITS>
__device__ __forceinline__ void moe_down_rows_t(const bf16* a, const int* plan, const long long* ids, const unsigned char* data, const unsigned char* scales, long long data_stride, long long scale_stride, float* part, int E,
                                                int inter, int hidden, int split) {
    constexpr int BIAS = 1 << (BITS - 1);  // a code u stands for (u - BIAS) s
    constexpr int HI = 4 * (BITS - 4);     // bytes of high bits a run has (none, one word, two)
    // the listed neurons (a word each: the neuron, then the activation's bf16), and, where the list is dead, the warps' sums [warp][lane * 33 + place] (33: a lane's run apart from the next lane's banks)
    __shared__ __align__(16) unsigned char smem[MOE_ROWS_WARPS * (MOE_ROWS_OUT + 32) * 4];
    __shared__ int warp_total[MOE_ROWS_WARPS];
    __shared__ int listed;
    unsigned* list = reinterpret_cast<unsigned*>(smem);
    float* red = reinterpret_cast<float*>(smem);
    const int i = blockIdx.x, s = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int hits = plan[0];
    if (i >= plan[1 + E + hits]) return;  // (the whole block: no pair this far down the plan)
    const int j = plan[2 + 2 * E + i];
    const long long e = ids[j];
    // ---- the pair's neurons that are not zero, in order: a thread takes consecutive chunks of 8 activations, the block counts them in thread order
    const int nchunk = inter >> 3, per = (nchunk + MOE_ROWS_WARPS * 32 - 1) / (MOE_ROWS_WARPS * 32);
    const uint4* arow = reinterpret_cast<const uint4*>(a + (size_t)i * inter);
    uint4 held[2];  // (inter at most MOE_ROWS_MAX_INTER: a thread takes at most 2 chunks)
    int mine = 0;
#pragma unroll
    for (int c = 0; c < 2; c++) {
        const int ci = tid * per + c;
        uint4 v = {0u, 0u, 0u, 0u};
        if (c < per && ci < nchunk) v = arow[ci];
        held[c] = v;
        const unsigned w4[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int q = 0; q < 4; q++) mine += ((w4[q] & 0x7fffu) != 0u) + ((w4[q] & 0x7fff0000u) != 0u);
    }
    int x = mine;  // the inclusive sum over the block's threads: within the warp, then over the warps' totals
    for (int o = 1; o < 32; o <<= 1) {
        const int v = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += v;
    }
    if (lane == 31) warp_total[warp] = x;
    __syncthreads();
    int at = x - mine;
    for (int w = 0; w < warp; w++) at += warp_total[w];
    if (tid == MOE_ROWS_WARPS * 32 - 1) listed = at + mine;
#pragma unroll
    for (int c = 0; c < 2; c++) {
        const unsigned w4[4] = {held[c].x, held[c].y, held[c].z, held[c].w};
#pragma unroll
        for (int q = 0; q < 4; q++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const unsigned bits = h ? w4[q] >> 16 : w4[q] & 0xffffu;
                if ((bits & 0x7fffu) != 0u) list[at++] = (unsigned)((tid * per + c) * 8 + 2 * q + h) | (bits << 16);
            }
    }
    __syncthreads();
    const int count = listed;
    // ---- this lane's run of 32 outputs: the codes and the scale of each listed neuron's row, added in
    const int runs = hidden >> 5, run = blockIdx.z * 32 + lane;
    const bool live = run < runs;
    const size_t rstride = (size_t)((runs * (16 + HI) + 15) & ~15);  // a row's bytes of codes
    const unsigned char* drow = data + e * data_stride;
    const unsigned char* srow = scales + e * scale_stride;
    float acc[32];
#pragma unroll
    for (int q = 0; q < 32; q++) acc[q] = 0.f;
    const unsigned off_bits = 0x43004300u | ((unsigned)BIAS * 0x00010001u);  // (128 + BIAS in both halves)
    const __nv_bfloat162 off2 = *reinterpret_cast<const __nv_bfloat162*>(&off_bits);
    const int step = split * MOE_ROWS_WARPS;
    for (int m0 = s + split * warp; m0 < count; m0 += MOE_ROWS_UNROLL * step) {
        uint4 lo[MOE_ROWS_UNROLL];
        unsigned hi[MOE_ROWS_UNROLL][2];
        unsigned sc[MOE_ROWS_UNROLL], ent[MOE_ROWS_UNROLL];
#pragma unroll
        for (int u = 0; u < MOE_ROWS_UNROLL; u++) {  // (all the loads first)
            const int m = m0 + u * step;
            ent[u] = m < count ? list[m] : 0u;
            lo[u] = {0u, 0u, 0u, 0u};
            hi[u][0] = hi[u][1] = 0u;
            sc[u] = 0u;
            if (m < count && live) {
                const size_t n = ent[u] & 0xffffu;
                const unsigned char* row = drow + n * rstride;
                lo[u] = *reinterpret_cast<const uint4*>(row + 16 * run);
                if (BITS == 5) hi[u][0] = *reinterpret_cast<const unsigned*>(row + 16 * runs + HI * run);
                if (BITS == 6) {
                    hi[u][0] = *reinterpret_cast<const unsigned*>(row + 16 * runs + HI * run);
                    hi[u][1] = *reinterpret_cast<const unsigned*>(row + 16 * runs + HI * run + 4);
                }
                sc[u] = *reinterpret_cast<const unsigned short*>(srow + n * runs * 2 + 2 * run);
            }
        }
#pragma unroll
        for (int u = 0; u < MOE_ROWS_UNROLL; u++) {
            if (m0 + u * step >= count || !live) continue;
            const float av = __uint_as_float(ent[u] & 0xffff0000u);
            const unsigned scale_bits = sc[u] | (sc[u] << 16);  // (the run's scale in both halves)
            const __nv_bfloat162 scale2 = *reinterpret_cast<const __nv_bfloat162*>(&scale_bits);
            const unsigned lw[4] = {lo[u].x, lo[u].y, lo[u].z, lo[u].w};
#pragma unroll
            for (int p = 0; p < 16; p++) {  // pair p: weights 2p and 2p + 1 of the run, as the block layouts decode a pair: (128 + u) as a bf16 pair, less 128 + BIAS (exact), times the scale, rounded once
                unsigned x = 0x43004300u | ((lw[p >> 2] >> (4 * (p & 3))) & 0x000F000Fu);
                if (BITS == 5) x |= ((hi[u][0] >> p) & 0x00010001u) << 4;
                if (BITS == 6) x |= ((hi[u][p >> 3] >> (2 * (p & 7))) & 0x00030003u) << 4;
                const __nv_bfloat162 w2 = __hmul2(__hsub2(*reinterpret_cast<const __nv_bfloat162*>(&x), off2), scale2);
                const unsigned wb = *reinterpret_cast<const unsigned*>(&w2);
                acc[2 * p] = fmaf(av, __uint_as_float(wb << 16), acc[2 * p]);
                acc[2 * p + 1] = fmaf(av, __uint_as_float(wb & 0xffff0000u), acc[2 * p + 1]);
            }
        }
    }
    // ---- the warps' sums added in the warps' order, and the block's written
    __syncthreads();  // (the list is dead)
    if (live) {
#pragma unroll
        for (int q = 0; q < 32; q++) red[warp * (MOE_ROWS_OUT + 32) + lane * 33 + q] = acc[q];
    }
    __syncthreads();
    for (int o = tid; o < MOE_ROWS_OUT; o += MOE_ROWS_WARPS * 32) {
        const int out = blockIdx.z * MOE_ROWS_OUT + o;
        if (out >= hidden) break;
        float sum = 0.f;
#pragma unroll
        for (int w = 0; w < MOE_ROWS_WARPS; w++) sum += red[w * (MOE_ROWS_OUT + 32) + (o >> 5) * 33 + (o & 31)];
        part[((size_t)j * split + s) * hidden + out] = sum;
    }
}

extern "C" __global__ void __launch_bounds__(256) moe_down_rows4(const bf16* a, const int* plan, const long long* ids, const unsigned char* data, const unsigned char* scales, long long data_stride, long long scale_stride, float* part,
                                                                 int E, int inter, int hidden, int split) {
    moe_down_rows_t<4>(a, plan, ids, data, scales, data_stride, scale_stride, part, E, inter, hidden, split);
}
extern "C" __global__ void __launch_bounds__(256) moe_down_rows5(const bf16* a, const int* plan, const long long* ids, const unsigned char* data, const unsigned char* scales, long long data_stride, long long scale_stride, float* part,
                                                                 int E, int inter, int hidden, int split) {
    moe_down_rows_t<5>(a, plan, ids, data, scales, data_stride, scale_stride, part, E, inter, hidden, split);
}
extern "C" __global__ void __launch_bounds__(256) moe_down_rows6(const bf16* a, const int* plan, const long long* ids, const unsigned char* data, const unsigned char* scales, long long data_stride, long long scale_stride, float* part,
                                                                 int E, int inter, int hidden, int split) {
    moe_down_rows_t<6>(a, plan, ids, data, scales, data_stride, scale_stride, part, E, inter, hidden, split);
}

// y [T, hidden] (bf16) = for each token the sum over its k choices c (ids[t k + c] an expert; outside 0 to E - 1: none) of wts[t k + c] times the sum over s of part [t k + c][s][o], plus extra [t][o] where extra is
// not null. Block (token, 64 outputs), 256 threads: four groups of 64, group g taking the choices g, g + 4, ...; the groups' sums are added in order, then extra (one rounding at the end).
extern "C" __global__ void __launch_bounds__(256) moe_rows_sum(const float* part, const float* wts, const long long* ids, bf16* y, int E, int k, int hidden, int split, const float* extra) {
    __shared__ float groups[4][64];
    const int t = blockIdx.x, g = threadIdx.x >> 6, o = blockIdx.y * 64 + (threadIdx.x & 63);
    float acc = 0.f;
    if (o < hidden)
        for (int c = g; c < k; c += 4) {
            const long long e = ids[(size_t)t * k + c];
            if (e < 0 || e >= E) continue;
            const float* p = part + ((size_t)t * k + c) * split * hidden + o;
            float sum = 0.f;
#pragma unroll 4
            for (int s = 0; s < split; s++) sum += p[(size_t)s * hidden];
            acc += sum * wts[(size_t)t * k + c];
        }
    groups[g][threadIdx.x & 63] = acc;
    __syncthreads();
    if (g == 0 && o < hidden) {
        float v = ((groups[0][o & 63] + groups[1][o & 63]) + groups[2][o & 63]) + groups[3][o & 63];
        if (extra) v += extra[(size_t)t * hidden + o];
        y[(size_t)t * hidden + o] = f2b(v);
    }
}
