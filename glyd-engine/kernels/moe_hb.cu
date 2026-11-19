// The Hummingbird experts' kernels (included by engine.cu after moe_rows.cu): the products of a relu squared expert over codebook rows, and the rotation around them. A matrix of the format is `rows` neuron rows of `cols` weights
// in three arrays: `codes` (2 or 3 bits a weight; a row is cols * bits / 8 bytes, a little-endian bitstream, code j at stream bits bits j .. bits j + bits - 1, so that a run of 32 codes is 8 or 12 bytes), `scales` (a bf16 for each
// run of 32, [row][run]) and `cb` (the row's codebook, 2^bits bf16, [row][entry]; the code indexes it); a weight is f32(scale of its run) * f32(codebook[code]), exact in fp32, never rounded to bf16 (the block layouts round theirs).
// The experts' weights are held in a basis rotated by R = H diag(signs) (H the natural-order Walsh-Hadamard matrix, the signs +1 or -1, both shared by the layer's experts): a token is rotated once (z' = R z), every expert multiplies
// in that basis, and the weighted sum of the experts' outputs is rotated back once (y = R^T y' = signs o (H y')). The host passes scale = 1 / sqrt(N) for the H's normalization (exactly 1 / 32 at N = 1,024).
//
// hb_rot_in (a block a token, N / 4 threads): zr [token][N] (fp32) = scale H (signs o z) of the token's row of z (bf16, `zs` elements apart): a thread holds 4 consecutive values, the butterflies (a, b) -> (a + b, a - b) in fp32 go bit 0
// first: bits 0 and 1 in the thread, bits 2 to 6 across the lanes of a warp (xor shuffles), the higher bits across the warps through shared memory. No atomics: the same bits every time.
// hb_group_ids: the router's ids of a layer whose experts are in two stacks (a group each) as the ids of one stack's experts: the expert's number in its group, else -1.
// hb_up2, hb_up3 (a block a pair of the plan and 64 neurons, 8 warps): the first product over the rows of the up matrix, in the rotated basis, with relu squared: a[plan row i][n] = bf16(max(bf16(u_n), 0)^2) for u_n = the row's weights times
// zr [token of the pair] (fp32). A warp takes 8 of the tile's neurons, R = cols / 32 lanes a neuron (lane r: the run r of the row, with its 32 values of zr in registers), 32 / R neurons together, up to 4 such steps with their loads issued
// first; the run's partial sum is the 32 products added in order from 0 (fmaf, the codebook entry times zr), times the run's scale, and the R runs' partial sums are added by an xor-shuffle tree (offsets 1, 2, 4, ...), so that a row's
// sum depends on the row alone: not on the place the expert is in, nor on the timing. The row's codebook is staged in shared memory (a table a neuron of a warp step) and indexed by the codes.
// hb_down2, hb_down3 (a block a pair, or a part of one, as moe_down_rows): the second product for the neurons that are not zero, over the rows of the transposed down matrix (rows = neurons, cols = outputs): the pair's activations (row
// i of a, bf16, the plan's order) are listed in order, the block's warps take every (split * 8)-th listed neuron from s + split * warp, and a lane (a run of 32 outputs) adds the neuron's row times the activation into 32 fp32 accumulators,
// c = activation * scale of the run (one fp32 multiply, exact), acc = fmaf(codebook entry, c, acc); the codebook of a neuron is the same for the 32 lanes (staged in shared memory a warp, 4 neurons at a time). The warps' sums are added
// in the warps' order and written to part [pair j][s][hidden] (fp32), as moe_down_rows does (its list, its shared memory reuse, its sums).
// hb_sum_rot (a block a token, hidden / 4 threads): v = the sum over the token's choices c in order (an id outside 0 to E - 1: skipped) of weight[c] * (the sum over s in order of the choice's parts), fmaf in fp32, then extra [token]
// [o] (fp32, in the rotated basis: the expert cache's host computes some of a token's pairs and sends their weighted sum; none where extra is null) added; then y = bf16(signs o (scale H v)), the butterflies of hb_rot_in.
//
// The grids: hb_rot_in and hb_sum_rot (tokens); hb_group_ids (pairs / 256 rounded up); hb_up{2,3} (pairs, neurons / 64 rounded up) of 256 threads; hb_down{2,3} (pairs, split, hidden / 1,024 rounded up) of 256 threads; the pairs in the plan's
// order as for moe_down_rows (a block past the plan's pairs does nothing; a pair routed nowhere is not in the plan). Nothing here reads the values but through the lists: a decode step captures into a CUDA graph.
// It uses moe.cu's MOE_FULL and moe_rows.cu's MOE_ROWS_* sizes.

#define HB_UP_UNROLL 4  // neurons a warp has loaded before it decodes them (hb_up: steps of 32 / R neurons)

// A bf16 word as a float (exact).
__device__ __forceinline__ float hb_bf(unsigned short w) { return __uint_as_float((unsigned)w << 16); }

// Code q (0 to 31) of a run's BITS words: the run's 32 codes are a little-endian bitstream (code q at bits BITS q to BITS q + BITS - 1), a code of 3 bits can straddle two words.
template <int BITS>
__device__ __forceinline__ unsigned hb_code(const unsigned* w, int q) {
    const int b = BITS * q, wi = b >> 5, sh = b & 31;
    unsigned c = w[wi] >> sh;
    if (sh + BITS > 32) c |= w[wi + 1] << (32 - sh);
    return c & ((1u << BITS) - 1u);
}

// The Walsh-Hadamard butterflies (unnormalized, natural order) of N = 4 blockDim.x values, v[0..3] the thread's values 4 threadIdx.x ... 4 threadIdx.x + 3: (a, b) -> (a + b, a - b) for the index pairs of each bit, bit 0 up. Bits 0 and
// 1 in the thread, bits 2 to 6 across the lanes (xor shuffles), the rest across the warps through sm [4 blockDim.x]. blockDim.x a power of two from 32 to 256; every thread of the block calls it.
__device__ __forceinline__ void hb_wht(float* v, float* sm) {
    const int tid = threadIdx.x, lane = tid & 31;
    float a = v[0], b = v[1], c = v[2], d = v[3];
    v[0] = a + b;
    v[1] = a - b;
    v[2] = c + d;
    v[3] = c - d;
    a = v[0], b = v[1], c = v[2], d = v[3];
    v[0] = a + c;
    v[1] = b + d;
    v[2] = a - c;
    v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {  // (the partner lane is lane ^ m; the one with the bit clear holds a)
        const bool hi = (lane & m) != 0;
#pragma unroll
        for (int q = 0; q < 4; q++) {
            const float o = __shfl_xor_sync(MOE_FULL, v[q], m);
            v[q] = hi ? o - v[q] : v[q] + o;
        }
    }
    for (int m = 32; m < (int)blockDim.x; m <<= 1) {  // (the partner thread is tid ^ m, the same lane of another warp)
#pragma unroll
        for (int q = 0; q < 4; q++) sm[4 * tid + q] = v[q];
        __syncthreads();
#pragma unroll
        for (int q = 0; q < 4; q++) {
            const float o = sm[4 * (tid ^ m) + q];
            v[q] = (tid & m) ? o - v[q] : v[q] + o;
        }
        __syncthreads();
    }
}

// zr [T, N] (fp32) = scale H (signs o z) for each token's row of z [T, zs] (bf16): block (token), N / 4 threads (N a power of two from 128 to 1,024), signs [N] = +1 or -1.
extern "C" __global__ void __launch_bounds__(256) hb_rot_in(const bf16* z, int zs, const signed char* signs, float* zr, int N, float scale) {
    __shared__ float sm[1024];
    const size_t t = blockIdx.x;
    const int o = 4 * threadIdx.x;
    float v[4];
#pragma unroll
    for (int q = 0; q < 4; q++) v[q] = (float)signs[o + q] * b2f(z[t * zs + o + q]);
    hb_wht(v, sm);
    uint4 ov;
    ov.x = __float_as_uint(scale * v[0]);
    ov.y = __float_as_uint(scale * v[1]);
    ov.z = __float_as_uint(scale * v[2]);
    ov.w = __float_as_uint(scale * v[3]);
    *reinterpret_cast<uint4*>(zr + t * N + o) = ov;
}

// out [pairs] = the pair's expert's number in its group where ids[j] is an expert (0 to E - 1) of `group` (group_of [E]; local_of [E] its number there), else -1. Block 256 threads.
extern "C" __global__ void __launch_bounds__(256) hb_group_ids(const long long* ids, const unsigned char* group_of, const unsigned short* local_of, int group, long long* out, int pairs, int E) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= pairs) return;
    const long long e = ids[j];
    out[j] = (e >= 0 && e < E && group_of[e] == group) ? (long long)local_of[e] : -1LL;
}

// One block of hb_up for BITS (2 or 3) a code. zr: [tokens, cols] fp32 (hb_rot_in's); plan: Lib::moe_way's [2 + 2 E + pairs] int32; via: [pairs] int64, the place of pair j's expert in codes, scales and cb (the expert's own number where
// they hold every expert, else the expert cache's place); codes, scales, cb: the up matrices' three arrays, `codes_stride`, `scales_stride`, `cb_stride` bytes between places; a: [pairs, inter] bf16, row i the plan's pair i. cols a
// power of two from 128 to 1,024, every row's bytes 4-byte aligned.
template <int BITS>
__device__ __forceinline__ void hb_up_t(const float* zr, const int* plan, const long long* via, const unsigned char* codes, const unsigned char* scales, const unsigned char* cb, long long codes_stride, long long scales_stride,
                                        long long cb_stride, bf16* a, int E, int inter, int cols, int k) {
    constexpr int NCB = 1 << BITS;  // entries of a row's codebook
    __shared__ float tab[MOE_ROWS_WARPS][HB_UP_UNROLL][8 * NCB];  // a warp step's codebooks: its 32 / R neurons' entries (at most 8 neurons)
    const int i = blockIdx.x, tile = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int hits = plan[0];
    if (i >= plan[1 + E + hits]) return;  // (the whole block: no pair this far down the plan)
    const int j = plan[2 + 2 * E + i];
    const long long place = via[j];
    const int token = j / k;
    const int R = cols >> 5, G = 32 / R, steps = 8 / G;  // runs a row, neurons a warp step, steps a warp (8 neurons)
    const int r = lane % R, g = lane / R;                // this lane's run, and which of the step's neurons
    const size_t cstride = (size_t)R * 4 * BITS;         // bytes of a row's codes
    const unsigned char* crow = codes + place * codes_stride;
    const unsigned char* srow = scales + place * scales_stride;
    const unsigned short* brow = reinterpret_cast<const unsigned short*>(cb + place * cb_stride);
    float zv[32];  // the 32 values of zr for this lane's run
    const uint4* zp = reinterpret_cast<const uint4*>(zr + (size_t)token * cols + 32 * r);
#pragma unroll
    for (int h = 0; h < 8; h++) {
        const uint4 q = zp[h];
        zv[4 * h] = __uint_as_float(q.x);
        zv[4 * h + 1] = __uint_as_float(q.y);
        zv[4 * h + 2] = __uint_as_float(q.z);
        zv[4 * h + 3] = __uint_as_float(q.w);
    }
    const int n0 = tile * 64 + warp * 8;
    for (int s0 = 0; s0 < steps; s0 += HB_UP_UNROLL) {
        unsigned cw[HB_UP_UNROLL][BITS], sc[HB_UP_UNROLL];
#pragma unroll
        for (int u = 0; u < HB_UP_UNROLL; u++) {  // (all the loads first; a step past the warp's 8 neurons is skipped by every lane of the warp)
            const int st = s0 + u, n = n0 + st * G + g;
#pragma unroll
            for (int w = 0; w < BITS; w++) cw[u][w] = 0u;
            sc[u] = 0u;
            if (st < steps) {
                if (n < inter) {
                    const unsigned* cp = reinterpret_cast<const unsigned*>(crow + (size_t)n * cstride + 4 * BITS * r);
#pragma unroll
                    for (int w = 0; w < BITS; w++) cw[u][w] = cp[w];
                    sc[u] = *reinterpret_cast<const unsigned short*>(srow + ((size_t)n * R + r) * 2);
                }
                for (int e = lane; e < G * NCB; e += 32) {  // (lane e: entry e % NCB of the step's neuron e / NCB)
                    const int ne = n0 + st * G + e / NCB;
                    tab[warp][u][e] = ne < inter ? hb_bf(brow[(size_t)ne * NCB + e % NCB]) : 0.f;
                }
            }
        }
        __syncwarp();
        float p[HB_UP_UNROLL];
#pragma unroll
        for (int u = 0; u < HB_UP_UNROLL; u++) {
            p[u] = 0.f;
            if (s0 + u < steps) {
                const float* tb = &tab[warp][u][g * NCB];
                float t = 0.f;
#pragma unroll
                for (int q = 0; q < 32; q++) t = fmaf(tb[hb_code<BITS>(cw[u], q)], zv[q], t);
                p[u] = hb_bf((unsigned short)sc[u]) * t;
            }
        }
        for (int off = 1; off < R; off <<= 1) {  // (the R runs of a neuron: lanes g R .. g R + R - 1)
#pragma unroll
            for (int u = 0; u < HB_UP_UNROLL; u++)
                if (s0 + u < steps) p[u] += __shfl_xor_sync(MOE_FULL, p[u], off);
        }
#pragma unroll
        for (int u = 0; u < HB_UP_UNROLL; u++) {
            const int st = s0 + u, n = n0 + st * G + g;
            if (r == 0 && st < steps && n < inter) {
                const float h = fmaxf(b2f(f2b(p[u])), 0.f);
                a[(size_t)i * inter + n] = f2b(h * h);
            }
        }
        __syncwarp();  // (the tables are read: the next step's may be written)
    }
}

extern "C" __global__ void __launch_bounds__(256) hb_up2(const float* zr, const int* plan, const long long* via, const unsigned char* codes, const unsigned char* scales, const unsigned char* cb, long long codes_stride,
                                                         long long scales_stride, long long cb_stride, bf16* a, int E, int inter, int cols, int k) {
    hb_up_t<2>(zr, plan, via, codes, scales, cb, codes_stride, scales_stride, cb_stride, a, E, inter, cols, k);
}
extern "C" __global__ void __launch_bounds__(256) hb_up3(const float* zr, const int* plan, const long long* via, const unsigned char* codes, const unsigned char* scales, const unsigned char* cb, long long codes_stride,
                                                         long long scales_stride, long long cb_stride, bf16* a, int E, int inter, int cols, int k) {
    hb_up_t<3>(zr, plan, via, codes, scales, cb, codes_stride, scales_stride, cb_stride, a, E, inter, cols, k);
}

// One block of hb_down for BITS (2 or 3) a code: moe_down_rows_t's, with the row decoded through the row's codebook. a: the first product's rows [pairs, inter] bf16 (relu squared applied); plan: Lib::moe_way's; via: [pairs] int64, the
// place of pair j's expert in codes, scales and cb; part: [pairs][split][hidden] fp32. hidden a multiple of 32, inter of 8, at most MOE_ROWS_MAX_INTER; a row's bytes 4-byte aligned.
template <int BITS>
__device__ __forceinline__ void hb_down_t(const bf16* a, const int* plan, const long long* via, const unsigned char* codes, const unsigned char* scales, const unsigned char* cb, long long codes_stride, long long scales_stride,
                                          long long cb_stride, float* part, int E, int inter, int hidden, int split) {
    constexpr int NCB = 1 << BITS;  // entries of a row's codebook
    // the listed neurons (a word each: the neuron, then the activation's bf16), and, where the list is dead, the warps' sums [warp][lane * 33 + place] (33: a lane's run apart from the next lane's banks)
    __shared__ __align__(16) unsigned char smem[MOE_ROWS_WARPS * (MOE_ROWS_OUT + 32) * 4];
    __shared__ int warp_total[MOE_ROWS_WARPS];
    __shared__ int listed;
    __shared__ float tab[MOE_ROWS_WARPS][MOE_ROWS_UNROLL][NCB];  // the codebooks of the neurons a warp has loaded
    unsigned* list = reinterpret_cast<unsigned*>(smem);
    float* red = reinterpret_cast<float*>(smem);
    const int i = blockIdx.x, s = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int hits = plan[0];
    if (i >= plan[1 + E + hits]) return;  // (the whole block: no pair this far down the plan)
    const int j = plan[2 + 2 * E + i];
    const long long place = via[j];
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
    // ---- this lane's run of 32 outputs: the codes and the scale of each listed neuron's row, and the neuron's codebook (the same for the warp's lanes), added in
    const int runs = hidden >> 5, run = blockIdx.z * 32 + lane;
    const bool live = run < runs;
    const size_t rstride = (size_t)runs * 4 * BITS;  // a row's bytes of codes
    const unsigned char* drow = codes + place * codes_stride;
    const unsigned char* srow = scales + place * scales_stride;
    const unsigned short* brow = reinterpret_cast<const unsigned short*>(cb + place * cb_stride);
    float acc[32];
#pragma unroll
    for (int q = 0; q < 32; q++) acc[q] = 0.f;
    const int step = split * MOE_ROWS_WARPS;
    for (int m0 = s + split * warp; m0 < count; m0 += MOE_ROWS_UNROLL * step) {
        unsigned lo[MOE_ROWS_UNROLL][BITS], sc[MOE_ROWS_UNROLL], ent[MOE_ROWS_UNROLL];
#pragma unroll
        for (int u = 0; u < MOE_ROWS_UNROLL; u++) {  // (all the loads first)
            const int m = m0 + u * step;
            ent[u] = m < count ? list[m] : 0u;
#pragma unroll
            for (int w = 0; w < BITS; w++) lo[u][w] = 0u;
            sc[u] = 0u;
            if (m < count && live) {
                const size_t n = ent[u] & 0xffffu;
                const unsigned* cp = reinterpret_cast<const unsigned*>(drow + n * rstride + 4 * BITS * run);
#pragma unroll
                for (int w = 0; w < BITS; w++) lo[u][w] = cp[w];
                sc[u] = *reinterpret_cast<const unsigned short*>(srow + (n * runs + run) * 2);
            }
            if (lane < NCB) tab[warp][u][lane] = m < count ? hb_bf(brow[(size_t)(ent[u] & 0xffffu) * NCB + lane]) : 0.f;
        }
        __syncwarp();
#pragma unroll
        for (int u = 0; u < MOE_ROWS_UNROLL; u++) {
            if (m0 + u * step >= count || !live) continue;
            const float c = __uint_as_float(ent[u] & 0xffff0000u) * hb_bf((unsigned short)sc[u]);  // activation * scale: exact
            const float* tb = tab[warp][u];
#pragma unroll
            for (int q = 0; q < 32; q++) acc[q] = fmaf(tb[hb_code<BITS>(lo[u], q)], c, acc[q]);
        }
        __syncwarp();  // (the tables are read: the next neurons' may be written)
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

extern "C" __global__ void __launch_bounds__(256) hb_down2(const bf16* a, const int* plan, const long long* via, const unsigned char* codes, const unsigned char* scales, const unsigned char* cb, long long codes_stride,
                                                           long long scales_stride, long long cb_stride, float* part, int E, int inter, int hidden, int split) {
    hb_down_t<2>(a, plan, via, codes, scales, cb, codes_stride, scales_stride, cb_stride, part, E, inter, hidden, split);
}
extern "C" __global__ void __launch_bounds__(256) hb_down3(const bf16* a, const int* plan, const long long* via, const unsigned char* codes, const unsigned char* scales, const unsigned char* cb, long long codes_stride,
                                                           long long scales_stride, long long cb_stride, float* part, int E, int inter, int hidden, int split) {
    hb_down_t<3>(a, plan, via, codes, scales, cb, codes_stride, scales_stride, cb_stride, part, E, inter, hidden, split);
}

// y [T, hidden] (bf16) = for each token signs o (scale H v), v = the sum over its k choices c (ids[t k + c] an expert; outside 0 to E - 1: none) in order of wts[t k + c] times the sum over s in order of part [t k + c][s][o] (fp32,
// fmaf), then extra [t][o] where extra is not null (one rounding at the end, after the rotation back). Block (token), hidden / 4 threads (hidden a power of two from 128 to 1,024), signs [hidden] = +1 or -1.
extern "C" __global__ void __launch_bounds__(256) hb_sum_rot(const float* part, const float* wts, const long long* ids, const signed char* signs, bf16* y, const float* extra, int E, int k, int hidden, int split, float scale) {
    __shared__ float sm[1024];
    const size_t t = blockIdx.x;
    const int o = 4 * threadIdx.x;
    float v[4] = {0.f, 0.f, 0.f, 0.f};
    for (int c = 0; c < k; c++) {
        const long long e = ids[t * k + c];
        if (e < 0 || e >= E) continue;
        const float w = wts[t * k + c];
        float sum[4] = {0.f, 0.f, 0.f, 0.f};
        for (int s = 0; s < split; s++) {
            const uint4 pv = *reinterpret_cast<const uint4*>(part + ((t * k + c) * split + s) * hidden + o);
            sum[0] += __uint_as_float(pv.x);
            sum[1] += __uint_as_float(pv.y);
            sum[2] += __uint_as_float(pv.z);
            sum[3] += __uint_as_float(pv.w);
        }
#pragma unroll
        for (int q = 0; q < 4; q++) v[q] = fmaf(w, sum[q], v[q]);
    }
    if (extra) {
        const uint4 xv = *reinterpret_cast<const uint4*>(extra + t * hidden + o);
        v[0] += __uint_as_float(xv.x);
        v[1] += __uint_as_float(xv.y);
        v[2] += __uint_as_float(xv.z);
        v[3] += __uint_as_float(xv.w);
    }
    hb_wht(v, sm);
#pragma unroll
    for (int q = 0; q < 4; q++) y[t * hidden + o + q] = f2b((float)signs[o + q] * (scale * v[q]));
}
