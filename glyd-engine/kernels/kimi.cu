// Kimi-K3's own kernels (included by engine.cu after glm.cu and moe.cu; src/dynamic/kimi.rs describes the model): the depth mix that stands in for the residual stream (each block's input a softmax over the
// embedding and the sums of the blocks of layers before it). bf16 in memory, fp32 inside; the rounding points are research/kimi-k3.md's (src/kimi_ref.rs: the CPU
// reference). The delta attention is glm.cu's, the latent attention mla.cu's (the layer kind's), the experts' situ moe.cu's moe_act.

#define KM_SOURCES 16  // a depth mix's sources at most (Kimi-K3: 9)

// y row i (rows ys apart) = w * bf16(x row i (rows xs apart) * rsqrt(mean(x^2) + eps)): KimiRMSNorm, the normed value rounded, times the weight, rounded again; any width. A block a row, 512 threads,
// thread t's columns t, t + 512, ... in that order; for K up to 512 KR_COLS a thread's columns are read into registers once (a column a round had made a decode step's norm 8-15 us on one SM).
#define KR_COLS 16
extern "C" __global__ void __launch_bounds__(512) kimi_rms2(const bf16* x, int xs, const bf16* w, bf16* y, int ys, int K, float eps) {
    const bf16* xr = x + (size_t)blockIdx.x * xs;
    bf16* yr = y + (size_t)blockIdx.x * ys;
    const int tid = threadIdx.x;
    if (blockDim.x == 512 && K <= 512 * KR_COLS) {
        float v[KR_COLS];
#pragma unroll
        for (int k = 0; k < KR_COLS; k++) {
            const int d = tid + 512 * k;
            v[k] = d < K ? b2f(xr[d]) : 0.f;
        }
        float ss = 0.f;
#pragma unroll
        for (int k = 0; k < KR_COLS; k++)
            if (tid + 512 * k < K) ss = fmaf(v[k], v[k], ss);
        const float r = rsqrtf(block_sum(ss) / K + eps);
#pragma unroll
        for (int k = 0; k < KR_COLS; k++) {
            const int d = tid + 512 * k;
            if (d < K) yr[d] = f2b(b2f(w[d]) * b2f(f2b(v[k] * r)));
        }
        return;
    }
    float ss = 0.f;
    for (int d = tid; d < K; d += blockDim.x) {
        const float v = b2f(xr[d]);
        ss = fmaf(v, v, ss);
    }
    const float r = rsqrtf(block_sum(ss) / K + eps);
    for (int d = tid; d < K; d += blockDim.x) yr[d] = f2b(b2f(w[d]) * b2f(f2b(b2f(xr[d]) * r)));
}

// x = bf16(x + a), e values: the partial sum's add (an empty partial sum is zeros, so the first add is a itself).
extern "C" __global__ void kimi_add(bf16* x, const bf16* a, long long e) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < e) x[i] = f2b(b2f(x[i]) + b2f(a[i]));
}

// A half-layer's start as one kernel: where o is given, x [n, D] = bf16(x + o) (the partial sum's add); out = the depth mix of the nb completed blocks
// (blk: row i's block s at blk + i rs + s D) and x: u = wn wp (fp32), each source's score (u . v) / rms(v) (eps 1e-5), the softmax over the sources
// weighting their sum, rounded once; where push is given (blk's place for the next block), x made that block and then zeros (the next partial sum starts
// empty); y (rows ys apart) = KimiRMSNorm of the mix by w (the normed value rounded, times the weight, rounded again). A block a row of 512 threads, thread
// t's eight-value chunks t and t + 512 (16-byte loads), the blocks read four at a time with every source's sums in registers, each sum a warp's butterfly
// and then the warps' in order. As four kernels (the add, the mix on one SM reading a source at a time, the push, the norm) a half-layer took 9.7 us with
// two sources and 22 us with ten on a 4080 SUPER, in a graph; this one 5.8 and 9.8. D a multiple of 8, up to 8,192.
#define KN_T 512
#define KN_CH 2
__device__ __forceinline__ void kn_load(const bf16* r, int t, int D, float (&v)[KN_CH][8]) {
#pragma unroll
    for (int j = 0; j < KN_CH; j++) {
        const int c = (t + KN_T * j) * 8;
        const uint4 q = c < D ? *(const uint4*)(r + c) : make_uint4(0, 0, 0, 0);
        const bf16* h = (const bf16*)&q;
#pragma unroll
        for (int e = 0; e < 8; e++) v[j][e] = b2f(h[e]);
    }
}
__device__ __forceinline__ void kn_store(bf16* r, int t, int D, int j, const float (&v)[8]) {
    const int c = (t + KN_T * j) * 8;
    if (c >= D) return;
    uint4 q;
    bf16* h = (bf16*)&q;
#pragma unroll
    for (int e = 0; e < 8; e++) h[e] = f2b(v[e]);
    *(uint4*)(r + c) = q;
}
extern "C" __global__ void __launch_bounds__(KN_T) kimi_mix_norm(const bf16* blk, long long rs, int nb, bf16* x, const bf16* o, const bf16* wn, const bf16* wp, bf16* push, const bf16* w, bf16* y, int ys, int D, float eps) {
    const size_t i = blockIdx.x;
    const int tid = threadIdx.x, wi = tid >> 5, lane = tid & 31;
    __shared__ float red[KN_T / 32][2 * KM_SOURCES];
    __shared__ float tot[2 * KM_SOURCES];
    bf16* xr = x + i * D;
    const bf16* br = blk + i * rs;
    float u[KN_CH][8], p[KN_CH][8], ww[KN_CH][8];
    {
        float a[KN_CH][8], b[KN_CH][8];
        kn_load(wn, tid, D, a);
        kn_load(wp, tid, D, b);
        kn_load(xr, tid, D, p);
        kn_load(w, tid, D, ww);
#pragma unroll
        for (int j = 0; j < KN_CH; j++)
#pragma unroll
            for (int e = 0; e < 8; e++) u[j][e] = a[j][e] * b[j][e];
        if (o) {
            kn_load(o + i * D, tid, D, a);
#pragma unroll
            for (int j = 0; j < KN_CH; j++)
#pragma unroll
                for (int e = 0; e < 8; e++) p[j][e] = b2f(f2b(p[j][e] + a[j][e]));
        }
    }
    // (the blocks' sums, four blocks' loads at a time: a source's own address past the last block, its sums left out)
    float dot[KM_SOURCES], sq[KM_SOURCES];
#pragma unroll
    for (int s = 0; s < KM_SOURCES; s++) dot[s] = sq[s] = 0.f;
#pragma unroll
    for (int g = 0; g < KM_SOURCES; g += 4) {
        if (g < nb) {
            float v[4][KN_CH][8];
#pragma unroll
            for (int q = 0; q < 4; q++) kn_load(br + (size_t)min(g + q, nb - 1) * D, tid, D, v[q]);
#pragma unroll
            for (int q = 0; q < 4; q++)
                if (g + q < nb)
#pragma unroll
                    for (int j = 0; j < KN_CH; j++)
#pragma unroll
                        for (int e = 0; e < 8; e++) {
                            dot[g + q] = fmaf(u[j][e], v[q][j][e], dot[g + q]);
                            sq[g + q] = fmaf(v[q][j][e], v[q][j][e], sq[g + q]);
                        }
        }
    }
    // (the partial sum: source nb)
    float pd = 0.f, pq = 0.f;
#pragma unroll
    for (int j = 0; j < KN_CH; j++)
#pragma unroll
        for (int e = 0; e < 8; e++) {
            pd = fmaf(u[j][e], p[j][e], pd);
            pq = fmaf(p[j][e], p[j][e], pq);
        }
#pragma unroll
    for (int s = 0; s < KM_SOURCES; s++)
        if (s < nb) {
            dot[s] = warp_sum(dot[s]);
            sq[s] = warp_sum(sq[s]);
            if (lane == 0) red[wi][2 * s] = dot[s], red[wi][2 * s + 1] = sq[s];
        }
    pd = warp_sum(pd);
    pq = warp_sum(pq);
    if (lane == 0) red[wi][2 * nb] = pd, red[wi][2 * nb + 1] = pq;
    __syncthreads();
    if (tid < 2 * (nb + 1)) {
        float t = 0.f;
        for (int k = 0; k < KN_T / 32; k++) t += red[k][tid];
        tot[tid] = t;
    }
    __syncthreads();
    float a[KM_SOURCES], m = -INFINITY, z = 0.f;
#pragma unroll
    for (int s = 0; s < KM_SOURCES; s++)
        if (s <= nb) m = fmaxf(m, tot[2 * s] * rsqrtf(tot[2 * s + 1] / D + 1e-5f));
#pragma unroll
    for (int s = 0; s < KM_SOURCES; s++) {
        a[s] = s <= nb ? expf(tot[2 * s] * rsqrtf(tot[2 * s + 1] / D + 1e-5f) - m) : 0.f;
        z += a[s];
    }
    float acc[KN_CH][8];
#pragma unroll
    for (int j = 0; j < KN_CH; j++)
#pragma unroll
        for (int e = 0; e < 8; e++) acc[j][e] = 0.f;
#pragma unroll
    for (int g = 0; g < KM_SOURCES; g += 4) {
        if (g < nb) {
            float v[4][KN_CH][8];
#pragma unroll
            for (int q = 0; q < 4; q++) kn_load(br + (size_t)min(g + q, nb - 1) * D, tid, D, v[q]);
#pragma unroll
            for (int q = 0; q < 4; q++)
                if (g + q < nb)
#pragma unroll
                    for (int j = 0; j < KN_CH; j++)
#pragma unroll
                        for (int e = 0; e < 8; e++) acc[j][e] = fmaf(a[g + q], v[q][j][e], acc[j][e]);
        }
    }
    float an = 0.f;  // (the partial sum's weight, a[nb] read without an index into the registers)
#pragma unroll
    for (int s = 0; s < KM_SOURCES; s++)
        if (s == nb) an = a[s];
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < KN_CH; j++) {
#pragma unroll
        for (int e = 0; e < 8; e++) {
            acc[j][e] = b2f(f2b(fmaf(an, p[j][e], acc[j][e]) / z));
            ss = fmaf(acc[j][e], acc[j][e], ss);
        }
        // (a thread writes only the values it read: x and the next block need no barrier)
        if (push) {
            kn_store(push + i * rs, tid, D, j, p[j]);
            const float zero[8] = {};
            kn_store(xr, tid, D, j, zero);
        } else if (o) {
            kn_store(xr, tid, D, j, p[j]);
        }
    }
    const float r = rsqrtf(block_sum(ss) / D + eps);
#pragma unroll
    for (int j = 0; j < KN_CH; j++) {
        float h[8];
#pragma unroll
        for (int e = 0; e < 8; e++) h[e] = ww[j][e] * b2f(f2b(acc[j][e] * r));
        kn_store(y + i * ys, tid, D, j, h);
    }
}

// The expert cache's plan on the GPU (src/dynamic/tp_cache.rs's device path): a decode step's choices looked up in this GPU's table of places, each miss
// given the place a clock hand comes to first that no choice of this step uses and that was not used since the hand last passed it, the pairs' places (way)
// and the copies to make (fills) written. Every GPU of a group decides the same from the same choices, so no host takes part. One block: the tables in
// shared memory while the first thread decides, its changes written through to global memory. tab: place_of [experts] int, expert_of [slots] int, last
// [slots] uint (the step a place was last used in), ref [slots] uchar; meta [8] uint: step, hand, reads, hits, fills.
#define DC_MAX 1024
extern "C" __global__ void __launch_bounds__(256) dc_plan(const long long* ids, int pairs, int* place_of, int* expert_of, unsigned* last, unsigned char* refb, unsigned* meta, int experts, int slots, long long* way, int2* fills, int* nfill) {
    __shared__ int sp[DC_MAX];
    __shared__ int se[DC_MAX];
    __shared__ unsigned sl[DC_MAX];
    __shared__ unsigned char sr[DC_MAX];
    for (int i = threadIdx.x; i < experts; i += blockDim.x) sp[i] = place_of[i];
    for (int i = threadIdx.x; i < slots; i += blockDim.x) {
        se[i] = expert_of[i];
        sl[i] = last[i];
        sr[i] = refb[i];
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    const unsigned step = meta[0] + 1;
    unsigned hand = meta[1], hits = 0;
    int nf = 0;
    for (int p = 0; p < pairs; p++) {
        const long long e = ids[p];
        if (e < 0 || e >= experts) {
            DEV_SAY("dc_plan: choice %lld of %d experts\n", e, experts);
            __trap();
        }
        const int q = sp[e];
        if (q >= 0) {
            hits++;
            sl[q] = step;
            sr[q] = 1;
            last[q] = step;
            refb[q] = 1;
        }
    }
    for (int p = 0; p < pairs; p++) {
        const int e = (int)ids[p];
        if (sp[e] >= 0) continue;
        int q = -1;
        for (int guard = 0; guard < 2 * slots + 2; guard++) {
            const int c = (int)hand;
            hand = hand + 1 == (unsigned)slots ? 0 : hand + 1;
            if (sl[c] == step) continue;
            if (sr[c]) {
                sr[c] = 0;
                refb[c] = 0;
                continue;
            }
            q = c;
            break;
        }
        if (q < 0) {
            DEV_SAY("dc_plan: no place for expert %d (%d places, all this step's)\n", e, slots);
            __trap();
        }
        const int old = se[q];
        if (old >= 0) {
            sp[old] = -1;
            place_of[old] = -1;
        }
        se[q] = e;
        sp[e] = q;
        sl[q] = step;
        sr[q] = 1;
        expert_of[q] = e;
        place_of[e] = q;
        last[q] = step;
        refb[q] = 1;
        fills[nf++] = make_int2(e, q);
    }
    for (int p = 0; p < pairs; p++) way[p] = sp[(int)ids[p]];
    meta[0] = step;
    meta[1] = hand;
    meta[2] += pairs;
    meta[3] += hits;
    meta[4] += nf;
    *nfill = nf;
}

// dc_plan's copies: each fill's expert, this GPU's slice of its four parts (the codes and scales of its two matrices), from the host tier (pinned host
// memory, which the GPU reads through the same addresses) into its place. reg: 4 x (host base, host bytes an expert, this GPU's offset in an expert's,
// device base, bytes a place), u64s, every one a multiple of 16. Grid: (blocks a fill, the most fills); a block past the fills there are does nothing.
extern "C" __global__ void __launch_bounds__(256) dc_fill(const int2* fills, const int* nfill, const unsigned long long* reg) {
    const int f = blockIdx.y;
    if (f >= *nfill) return;
    const int2 x = fills[f];
    const unsigned long long step = (unsigned long long)gridDim.x * blockDim.x;
    for (int r = 0; r < 4; r++) {
        const unsigned long long* g = reg + 5 * r;
        const uint4* s4 = reinterpret_cast<const uint4*>(g[0] + (unsigned long long)x.x * g[1] + g[2]);
        uint4* d4 = reinterpret_cast<uint4*>(g[3] + (unsigned long long)x.y * g[4]);
        const unsigned long long n = g[4] / 16;
        unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        // (four reads in flight a thread before their writes: PCIe's latency is hidden by many)
        for (; i + 3 * step < n; i += 4 * step) {
            const uint4 a = s4[i], b = s4[i + step], c = s4[i + 2 * step], d = s4[i + 3 * step];
            d4[i] = a;
            d4[i + step] = b;
            d4[i + 2 * step] = c;
            d4[i + 3 * step] = d;
        }
        for (; i < n; i += step) d4[i] = s4[i];
    }
}
