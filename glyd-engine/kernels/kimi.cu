// Kimi-K3's own kernels (included by engine.cu after glm.cu and moe.cu; src/dynamic/kimi.rs describes the model): the expert cache's plan on the GPU. bf16 in memory, fp32 inside; the
// rounding points are research/kimi-k3.md's (src/kimi_ref.rs: the CPU reference). The depth mix that stands in for the residual stream is the residual kind's (residual.cu), the delta
// attention glm.cu's, the latent attention and KimiRMSNorm (rms2_rows512) mla.cu's, the experts' situ moe.cu's moe_act.

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
