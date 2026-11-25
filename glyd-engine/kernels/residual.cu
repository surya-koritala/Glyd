// The residual rules' kernels and the norms that are not engine.cu's, every family's (included by engine.cu after moe.cu; src/dynamic/residual.rs launches them): the residual streams (mHC, kind
// `Residual::Streams`: the embedding copied into the streams, a block's mixing weights and its normed collapse, the streams' update from its output, the final collapse), the depth mix (kind
// `Residual::DepthMix`), the gated residual (`Residual::Gated`), the plain add, and the RMS norm whose weight scales its input. bf16 in memory, fp32 inside. The streams' kernels take the kind's
// rounding as a parameter: the release's fp32 arithmetic with each result rounded once (DeepSeek-V4's inference/model.py), or transformers' bf16 tensors between operations (GLM-5.3-Flash's
// modeling_glm5_next.py).

#define HC_N 4         // residual streams
#define HC_NP 24       // rows of the mixing projection: (2 + 4) * 4 (the collapse weights, the spreading weights, the 4 x 4 mixing matrix)
#define HC_PER 16      // elements of a row a thread holds in the norms (D up to HC_PER * (blockDim.x - 32): 7,680 with 512 threads, hc_tail's collapse on warps 1 on)
#define HC_CPB_F32 256   // columns of a row of streams a block of hc_proj_f32 takes (a decode step's row on 64 blocks at 4,096 wide: the 1.5 MB of an fp32 projection read by more SMs than 512 columns a block gave)
#define HC_CPB_BF16 512  // columns a block of hc_proj_bf16 takes

__device__ __forceinline__ float hc_sig(float x) { return 1.f / (1.f + expf(-x)); }

// ---- the streams -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// x [n, ns, D] <- e [n, D] ns times: every stream starts as the token's embedding. Block (row, 256 columns).
extern "C" __global__ void hc_streams(const bf16* e, bf16* x, int D, int ns) {
    const int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (d >= D) return;
    const bf16 v = e[(size_t)blockIdx.x * D + d];
    for (int s = 0; s < ns; s++) x[((size_t)blockIdx.x * ns + s) * D + d] = v;
}

// ---- the mixing weights and the collapse -----------------------------------------------------------------------------------------------------------------------------------------------
// The rest of a mixing kernel, once tot [HC_NP + 1] (the 24 projections' sums and the sum of squares of the row) is in shared memory: the weights, `iters` rounds of Sinkhorn-Knopp, the collapse and the norm. Called by a
// block of 512 threads a row (hc_mix and hc_fin); pp [HC_NP] is shared memory too.
__device__ __forceinline__ void hc_tail(const bf16* xr, const float* base, const float* scale, const bf16* nw, bf16* h, float* hcp, int t, int D, float eps, float hc_eps, int iters, float* tot, float* pp, int single) {
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
    const int K = HC_N * D;
    // The collapse's weights first (pre: all the layer's input needs); then warp 0 makes comb by Sinkhorn-Knopp while warps 1 on collapse the row and norm it, at once (the
    // norm's sum over those warps alone, a barrier of their own): Sinkhorn's serial rounds had kept the block's other 508 threads waiting at a barrier.
    __shared__ float red15[16];
    float r[HC_N];
    if (tid < HC_N) {
        const int i = tid;
        const float rs = rsqrtf(tot[HC_NP] / K + eps);
        pp[i] = hc_sig(tot[i] * rs * scale[0] + base[i]) + hc_eps;
        pp[HC_N + i] = 2.f * hc_sig(tot[HC_N + i] * rs * scale[1] + base[HC_N + i]);
        float z[HC_N], mx = -INFINITY, sum = 0.f;
#pragma unroll
        for (int j = 0; j < HC_N; j++) {
            z[j] = tot[2 * HC_N + HC_N * i + j] * rs * scale[2] + base[2 * HC_N + HC_N * i + j];
            mx = fmaxf(mx, z[j]);
        }
#pragma unroll
        for (int j = 0; j < HC_N; j++) {
            z[j] = expf(z[j] - mx);
            sum += z[j];
        }
#pragma unroll
        for (int j = 0; j < HC_N; j++) r[j] = z[j] / sum + hc_eps;
    }
    __syncthreads();
    if (w == 0) {
        // (four threads, a row of comb each, in the order one thread took: a row's sums and a column's in order of index, every division by sum + hc_eps)
        if (lane < HC_N) {
            const int rounds = iters > 0 ? iters : 1;
            for (int it = 0; it < rounds; it++) {
                if (it > 0) {
                    float s = 0.f;
#pragma unroll
                    for (int j = 0; j < HC_N; j++) s += r[j];
#pragma unroll
                    for (int j = 0; j < HC_N; j++) r[j] /= (s + hc_eps);
                }
                // (every column's sum on every thread, from all four rows by shuffles: in the order of the rows, as one thread added them)
                float all[HC_N][HC_N];
#pragma unroll
                for (int q = 0; q < HC_N; q++)
#pragma unroll
                    for (int j = 0; j < HC_N; j++) all[q][j] = __shfl_sync(0xFu, r[j], q, HC_N);
#pragma unroll
                for (int j = 0; j < HC_N; j++) {
                    float s = 0.f;
#pragma unroll
                    for (int q = 0; q < HC_N; q++) s += all[q][j];
                    r[j] /= (s + hc_eps);
                }
            }
#pragma unroll
            for (int j = 0; j < HC_N; j++) pp[2 * HC_N + HC_N * lane + j] = r[j];
        }
        __syncwarp();
        if (lane < HC_NP) hcp[(size_t)t * HC_NP + lane] = pp[lane];
        return;
    }
    // the collapse and the norm: warps 1 on (NT threads)
    const int u = tid - 32, NT = blockDim.x - 32;
    float cv[HC_PER], ss2 = 0.f;
#pragma unroll
    for (int j = 0; j < HC_PER; j++) {
        const int d = u + j * NT;
        float c = 0.f;
        if (d < D) {
            c = pp[0] * b2f(xr[d]);
            c += pp[1] * b2f(xr[D + d]);
            c += pp[2] * b2f(xr[2 * D + d]);
            c += pp[3] * b2f(xr[3 * D + d]);
            c = b2f(f2b(c));
        }
        cv[j] = c;
        ss2 += c * c;
    }
    ss2 = warp_sum(ss2);
    if (lane == 0) red15[w] = ss2;
#if defined(__HIP_PLATFORM_AMD__)
    __syncthreads();  // (RDNA: warp 0 has ended, and a barrier waits for the waves still running)
#else
    asm volatile("bar.sync 1, %0;" ::"r"(NT) : "memory");
#endif
    float tot2 = 0.f;
    for (int q = 1; q < (int)(blockDim.x >> 5); q++) tot2 += red15[q];
    const float rn = rsqrtf(tot2 / D + eps);
#pragma unroll
    for (int j = 0; j < HC_PER; j++) {
        const int d = u + j * NT;
        // (single: the weight times the normed value rounded once, the release's; else transformers', the normed value rounded first)
        if (d < D) h[(size_t)t * D + d] = f2b(b2f(nw[d]) * (single ? cv[j] * rn : b2f(f2b(cv[j] * rn))));
    }
}

// For each row t of the streams x [n, 4, D] (bf16; transformers' hyper-connection and the norm that follows it): the 4D values normed without a weight (fp32, eps inside the root) are projected by fn [24, 4D] (bf16),
// the 24 results into pre = sigmoid(w scale0 + base) + hc_eps (4), post = 2 sigmoid(w scale1 + base) (4) and comb = softmax(w scale2 + base) + hc_eps (4 x 4: each row of the softmax over its 4 columns), which `iters`
// rounds of Sinkhorn-Knopp make doubly stochastic (the columns' sums divided out first, once even where `iters` is 0, then iters - 1 rounds of the rows' and the columns': every division by sum + hc_eps); hcp [n, 24]
// takes pre, post and comb (fp32). The row collapses to the layer's input, sum over streams of pre * x rounded to bf16, which is normed with the weight nw [D] (`single`: rounded once; else the normed value rounded
// to bf16, times the weight, rounded again) into h [n, D]. A block a row, 512 threads; the projection is a thread's share of the 4D columns for all 24 rows at once. Streams whose 4D the bf16 projection's
// blocks do not divide (src/dynamic/residual.rs `split`: the layer's shape, never the call's rows, so a row's sums are its own pass's); every other width, bf16 or fp32, takes hc_proj_* and hc_fin (another
// order of the sums).
extern "C" __global__ void __launch_bounds__(512) hc_mix(const bf16* x, const bf16* fn, const float* base, const float* scale, const bf16* nw, bf16* h, float* hcp, int D, float eps, float hc_eps, int iters, int single) {
    const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, w = tid >> 5, nwarps = blockDim.x >> 5;
    const int K = HC_N * D;
    const bf16* xr = x + (size_t)t * K;
    __shared__ float red[HC_NP + 1][16];
    __shared__ float tot[HC_NP + 1];
    __shared__ float pp[HC_NP];
    float acc[HC_NP], ss = 0.f;
#pragma unroll
    for (int r = 0; r < HC_NP; r++) acc[r] = 0.f;
    for (int e = tid * 2; e < K; e += blockDim.x * 2) {
        const float2 xv = bf2(*reinterpret_cast<const uint32_t*>(xr + e));
        ss = fmaf(xv.x, xv.x, fmaf(xv.y, xv.y, ss));
#pragma unroll
        for (int r = 0; r < HC_NP; r++) {
            const float2 fv = bf2(*reinterpret_cast<const uint32_t*>(fn + (size_t)r * K + e));
            acc[r] = fmaf(xv.x, fv.x, fmaf(xv.y, fv.y, acc[r]));
        }
    }
#pragma unroll
    for (int r = 0; r < HC_NP; r++) {
        const float v = warp_sum(acc[r]);
        if (lane == 0) red[r][w] = v;
    }
    ss = warp_sum(ss);
    if (lane == 0) red[HC_NP][w] = ss;
    __syncthreads();
    if (tid <= HC_NP) {
        float s = 0.f;
        for (int i = 0; i < nwarps; i++) s += red[tid][i];
        tot[tid] = s;
    }
    __syncthreads();
    hc_tail(xr, base, scale, nw, h, hcp, t, D, eps, hc_eps, iters, tot, pp, single);
}

// A projection's two adjacent weights as floats, loaded as they are stored (fp32 pairs, or bf16 pairs in one word) and converted where they are multiplied.
template <typename W> struct HcW;
template <> struct HcW<float> {
    typedef float2 T;
    static __device__ __forceinline__ T ld(const float* p) { return __ldg(reinterpret_cast<const float2*>(p)); }
    static __device__ __forceinline__ T zero() { return make_float2(0.f, 0.f); }
    static __device__ __forceinline__ float2 f2(T v) { return v; }
};
template <> struct HcW<bf16> {
    typedef uint32_t T;
    static __device__ __forceinline__ T ld(const bf16* p) { return __ldg(reinterpret_cast<const unsigned int*>(p)); }
    static __device__ __forceinline__ T zero() { return 0u; }
    static __device__ __forceinline__ float2 f2(T v) { return bf2(v); }
};

// The mixing projection for rows of streams x [n, 4D] (bf16) by fn [NP, 4D] (fp32 or bf16, as the checkpoint keeps it), split over the row's columns so that many blocks read the matrix at once (one block a row, hc_mix,
// reads it at the speed of one SM): block (row t, split s) of CPB columns writes the NP products' partial sums and the sum of squares of its columns to part [n][4D / CPB][HC_NP + 1] (fp32). NP up to HC_NP (24 for a
// block, 4 for the weighted head). 128 threads (2 columns each); 4D a multiple of CPB.
template <typename W, int CPB>
__device__ __forceinline__ void hc_proj_of(const bf16* x, const W* fn, float* part, int D, int NP) {
    const int t = blockIdx.x, s = blockIdx.y, tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int K = HC_N * D, KS = K / CPB;
    const bf16* xr = x + (size_t)t * K + (size_t)s * CPB;
    const W* fr = fn + (size_t)s * CPB;
    __shared__ float red[HC_NP + 1][4];
    float acc[HC_NP], ss = 0.f;
#pragma unroll
    for (int r = 0; r < HC_NP; r++) acc[r] = 0.f;
    for (int e = tid * 2; e < CPB; e += 256) {
        const float2 xv = bf2(*reinterpret_cast<const uint32_t*>(xr + e));
        ss = fmaf(xv.x, xv.x, fmaf(xv.y, xv.y, ss));
        // (all the rows' loads issued before any product: with the check in the products' loop each load waited for the one before, 24 trips to memory in a row)
        typename HcW<W>::T fv[HC_NP];
#pragma unroll
        for (int r = 0; r < HC_NP; r++) fv[r] = r < NP ? HcW<W>::ld(fr + (size_t)r * K + e) : HcW<W>::zero();
#pragma unroll
        for (int r = 0; r < HC_NP; r++) {
            const float2 f = HcW<W>::f2(fv[r]);
            acc[r] = fmaf(xv.x, f.x, fmaf(xv.y, f.y, acc[r]));
        }
    }
#pragma unroll
    for (int r = 0; r < HC_NP; r++) {
        const float v = warp_sum(acc[r]);
        if (lane == 0) red[r][w] = v;
    }
    ss = warp_sum(ss);
    if (lane == 0) red[HC_NP][w] = ss;
    __syncthreads();
    if (tid <= HC_NP) part[((size_t)t * KS + s) * (HC_NP + 1) + tid] = (red[tid][0] + red[tid][1]) + (red[tid][2] + red[tid][3]);
}
extern "C" __global__ void __launch_bounds__(128) hc_proj_f32(const bf16* x, const float* fn, float* part, int D, int NP) { hc_proj_of<float, HC_CPB_F32>(x, fn, part, D, NP); }
extern "C" __global__ void __launch_bounds__(128) hc_proj_bf16(const bf16* x, const bf16* fn, float* part, int D, int NP) { hc_proj_of<bf16, HC_CPB_BF16>(x, fn, part, D, NP); }

// The rest of hc_mix for rows whose partial sums hc_proj_* wrote (KS of them a row): arguments as hc_mix's. The parts are added by warps (`warps`: a warp a sum, its lanes a part apiece, then the warp's sum; an fp32
// projection's 4D / 256 parts) or in order by one thread a sum. A block a row, 512 threads.
extern "C" __global__ void __launch_bounds__(512) hc_fin(const bf16* x, const float* part, int KS, const float* base, const float* scale, const bf16* nw, bf16* h, float* hcp, int D, float eps, float hc_eps, int iters, int warps, int single) {
    const int t = blockIdx.x, tid = threadIdx.x;
    __shared__ float tot[HC_NP + 1];
    __shared__ float pp[HC_NP];
    if (warps) {
        const int w = tid >> 5, lane = tid & 31;
        for (int r = w; r <= HC_NP; r += blockDim.x >> 5) {
            float v = 0.f;
            for (int k = lane; k < KS; k += 32) v += part[((size_t)t * KS + k) * (HC_NP + 1) + r];
            v = warp_sum(v);
            if (lane == 0) tot[r] = v;
        }
    } else if (tid <= HC_NP) {
        float v = 0.f;
        for (int k = 0; k < KS; k++) v += part[((size_t)t * KS + k) * (HC_NP + 1) + tid];
        tot[tid] = v;
    }
    __syncthreads();
    hc_tail(x + (size_t)t * HC_N * D, base, scale, nw, h, hcp, t, D, eps, hc_eps, iters, tot, pp, single);
}

// x [n, 4, D] in place: stream i of a row from the block's output o [n, D] and the old streams, by post and comb (hcp [n, 24]'s, fp32). `bf16r`: as transformers computes it in bf16
// (post.to(dtype) * out + comb.to(dtype)^T @ residual): bf16(bf16(post_i o) + bf16(sum over j of comb[j][i] x_j)), post and comb rounded to bf16 first; else the release's, bf16(post_i o + sum over j of comb[j][i] x_j)
// in fp32. A thread a column of a row, all four streams.
extern "C" __global__ void hc_post(bf16* __restrict__ x, const bf16* __restrict__ o, const float* __restrict__ hcp, int D, int bf16r) {  // (restrict: the weights loaded once, not again after each stream's store)
    const int t = blockIdx.x, d = blockIdx.y * blockDim.x + threadIdx.x;
    if (d >= D) return;
    const float* p = hcp + (size_t)t * HC_NP;
    bf16* xr = x + (size_t)t * HC_N * D;
    float xv[HC_N];
#pragma unroll
    for (int j = 0; j < HC_N; j++) xv[j] = b2f(xr[(size_t)j * D + d]);
    const float ov = b2f(o[(size_t)t * D + d]);
    if (bf16r) {
#pragma unroll
        for (int i = 0; i < HC_N; i++) {
            const float a = b2f(f2b(b2f(f2b(p[HC_N + i])) * ov));
            float m = 0.f;
#pragma unroll
            for (int j = 0; j < HC_N; j++) m = fmaf(b2f(f2b(p[2 * HC_N + HC_N * j + i])), xv[j], m);
            xr[(size_t)i * D + d] = f2b(a + b2f(f2b(m)));
        }
    } else {
#pragma unroll
        for (int i = 0; i < HC_N; i++) {
            float m = 0.f;
#pragma unroll
            for (int j = 0; j < HC_N; j++) m += p[2 * HC_N + HC_N * j + i] * xv[j];
            xr[(size_t)i * D + d] = f2b(p[HC_N + i] * ov + m);
        }
    }
}

// out [rows, D] = the model's last step: the streams of a row (x [rows, 4, D]) collapsed, then the final norm with the weight nw [D] (`single`: rounded once; else the normed value rounded first). `weighted`: the hyper-head,
// for rows whose partial sums hc_proj_* wrote with the head's 4 rows (KS a row, added in order): pre = sigmoid(m scale + base) + hc_eps, the weighted sum rounded to bf16; else the mean (the sum in fp32 over four, rounded
// to bf16 as torch's mean of a bf16 tensor is; part, base and scale unread). A block a row, 512 threads.
extern "C" __global__ void __launch_bounds__(512) hc_head(const bf16* x, const float* part, int KS, const float* base, const float* scale, const bf16* nw, bf16* out, int D, float eps, float hc_eps, int weighted, int single) {
    const int t = blockIdx.x, tid = threadIdx.x;
    __shared__ float tot[HC_NP + 1];
    __shared__ float pre[HC_N];
    if (weighted) {
        if (tid <= HC_NP) {
            float v = 0.f;
#pragma unroll 8
            for (int k = 0; k < KS; k++) v += part[((size_t)t * KS + k) * (HC_NP + 1) + tid];
            tot[tid] = v;
        }
        __syncthreads();
        if (tid < HC_N) {
            const float rs = rsqrtf(tot[HC_NP] / (HC_N * D) + eps);
            pre[tid] = hc_sig(tot[tid] * rs * scale[0] + base[tid]) + hc_eps;
        }
        __syncthreads();
    }
    const bf16* xr = x + (size_t)t * HC_N * D;
    float cv[HC_PER], ss = 0.f;
#pragma unroll
    for (int j = 0; j < HC_PER; j++) {
        const int d = tid + j * blockDim.x;
        float c = 0.f;
        if (d < D) {
            if (weighted) {
                c = pre[0] * b2f(xr[d]);
                c += pre[1] * b2f(xr[D + d]);
                c += pre[2] * b2f(xr[2 * D + d]);
                c += pre[3] * b2f(xr[3 * D + d]);
                c = b2f(f2b(c));
            } else {
                c = b2f(f2b(((b2f(xr[d]) + b2f(xr[D + d])) + (b2f(xr[2 * D + d]) + b2f(xr[3 * D + d]))) * 0.25f));
            }
        }
        cv[j] = c;
        ss += c * c;
    }
    const float r = rsqrtf(block_sum(ss) / D + eps);
#pragma unroll
    for (int j = 0; j < HC_PER; j++) {
        const int d = tid + j * blockDim.x;
        if (d < D) out[(size_t)t * D + d] = f2b(b2f(nw[d]) * (single ? cv[j] * r : b2f(f2b(cv[j] * r))));
    }
}

// Lagged mHC (kind `StreamsP::lagged`, DeepSeek-V4.1): the block's input from the streams x [n, 4, D] weighed by the previous lagged block's input weights lag [n, 4] (fp32; `first`: one-hot on
// stream 0 before the first block), summed in the order of hc_tail's collapse and rounded to bf16, normed with the weight nw [D] (`single`: rounded once; else the normed value rounded first) into
// h [n, D]; then, where `keep`, lag takes this block's own input weights, the first 4 of hcp [n, 24] (hc_fin's, hc_mix's). With `keep` 0 it is the head's collapse (`Collapse::LastPre`) and
// the final norm. A block a row, 512 threads.
extern "C" __global__ void __launch_bounds__(512) hc_lag(const bf16* x, float* lag, const float* hcp, const bf16* nw, bf16* h, int D, float eps, int first, int keep, int single) {
    const int t = blockIdx.x, tid = threadIdx.x;
    __shared__ float w[HC_N];
    if (tid < HC_N) w[tid] = first ? (tid == 0 ? 1.f : 0.f) : lag[(size_t)t * HC_N + tid];
    __syncthreads();
    const bf16* xr = x + (size_t)t * HC_N * D;
    float cv[HC_PER], ss = 0.f;
#pragma unroll
    for (int j = 0; j < HC_PER; j++) {
        const int d = tid + j * blockDim.x;
        float c = 0.f;
        if (d < D) {
            c = w[0] * b2f(xr[d]);
            c += w[1] * b2f(xr[D + d]);
            c += w[2] * b2f(xr[2 * D + d]);
            c += w[3] * b2f(xr[3 * D + d]);
            c = b2f(f2b(c));
        }
        cv[j] = c;
        ss += c * c;
    }
    const float r = rsqrtf(block_sum(ss) / D + eps);
#pragma unroll
    for (int j = 0; j < HC_PER; j++) {
        const int d = tid + j * blockDim.x;
        if (d < D) h[(size_t)t * D + d] = f2b(b2f(nw[d]) * (single ? cv[j] * r : b2f(f2b(cv[j] * r))));
    }
    // (the weights were read into shared memory before: the row's own replace them for the next lagged block)
    if (keep && tid < HC_N) lag[(size_t)t * HC_N + tid] = hcp[(size_t)t * HC_NP + tid];
}

// ---- the plain add, the gated residual, the input-scaled norm ------------------------------------------------------------------------------------------------------------------------------
// x = bf16(x + a), e values (a depth mix's partial sum: an empty one is zeros, so the first add is a itself).
extern "C" __global__ void res_add(bf16* x, const bf16* a, long long e) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < e) x[i] = f2b(b2f(x[i]) + b2f(a[i]));
}

// The gated residual, x in place: x = bf16(bf16(alpha x) + bf16(beta h)) (alpha, beta: bf16 [K], the gate's factors). A thread a column.
extern "C" __global__ void res_gate(bf16* x, const bf16* h, const bf16* alpha, const bf16* beta, int K) {
    const int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= K) return;
    const size_t at = (size_t)blockIdx.x * K + i;
    x[at] = f2b(b2f(f2b(b2f(alpha[i]) * b2f(x[at]))) + b2f(f2b(b2f(beta[i]) * b2f(h[at]))));
}

// The RMS norm whose weight scales its input: y = RMS(bf16(x w)) without a weight (eps), rounded; rows of K. Block a row, 256 threads (K up to 256 x 32).
extern "C" __global__ void __launch_bounds__(256) rmsnorm_in(const bf16* x, const bf16* w, bf16* y, int K, float eps) {
    const bf16* xr = x + (size_t)blockIdx.x * K;
    float v[32], ss = 0.f;
#pragma unroll
    for (int j = 0; j < 32; j++) {
        const int i = threadIdx.x + j * 256;
        v[j] = i < K ? b2f(f2b(b2f(xr[i]) * b2f(w[i]))) : 0.f;
        ss += v[j] * v[j];
    }
    const float r = rsqrtf(block_sum(ss) / K + eps);
#pragma unroll
    for (int j = 0; j < 32; j++) {
        const int i = threadIdx.x + j * 256;
        if (i < K) y[(size_t)blockIdx.x * K + i] = f2b(v[j] * r);
    }
}

// ---- the depth mix ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// A block's start as one kernel: where o is given, x [n, D] = bf16(x + o) (the partial sum's add); out = the depth mix of the nb completed groups' sums
// (blk: row i's source s at blk + i rs + s D) and x: u = wn wp (fp32), each source's score (u . v) / rms(v) (eps 1e-5), the softmax over the sources
// weighting their sum, rounded once; where push is given (blk's place for the next source), x made that source and then zeros (the next partial sum starts
// empty); y (rows ys apart) = the RMS norm of the mix by w, the normed value rounded, times the weight, rounded again. A block a row of 512 threads, thread
// t's eight-value chunks t and t + 512 (16-byte loads), the sources read four at a time with every source's sums in registers, each sum a warp's butterfly
// and then the warps' in order. One kernel where four ran before (the add, the mix on one SM reading a source at a time, the push, the norm): a launch
// and a pass over the rows in place of four. D a multiple of 8, up to 8,192.
#define DM_SOURCES 16  // a depth mix's sources at most (Kimi-K3: 9)
#define DM_T 512
#define DM_CH 2
__device__ __forceinline__ void dm_load(const bf16* r, int t, int D, float (&v)[DM_CH][8]) {
#pragma unroll
    for (int j = 0; j < DM_CH; j++) {
        const int c = (t + DM_T * j) * 8;
        const uint4 q = c < D ? *(const uint4*)(r + c) : make_uint4(0, 0, 0, 0);
        const bf16* h = (const bf16*)&q;
#pragma unroll
        for (int e = 0; e < 8; e++) v[j][e] = b2f(h[e]);
    }
}
__device__ __forceinline__ void dm_store(bf16* r, int t, int D, int j, const float (&v)[8]) {
    const int c = (t + DM_T * j) * 8;
    if (c >= D) return;
    uint4 q;
    bf16* h = (bf16*)&q;
#pragma unroll
    for (int e = 0; e < 8; e++) h[e] = f2b(v[e]);
    *(uint4*)(r + c) = q;
}
extern "C" __global__ void __launch_bounds__(DM_T) depth_mix(const bf16* blk, long long rs, int nb, bf16* x, const bf16* o, const bf16* wn, const bf16* wp, bf16* push, const bf16* w, bf16* y, int ys, int D, float eps) {
    const size_t i = blockIdx.x;
    const int tid = threadIdx.x, wi = tid >> 5, lane = tid & 31;
    __shared__ float red[DM_T / 32][2 * DM_SOURCES];
    __shared__ float tot[2 * DM_SOURCES];
    bf16* xr = x + i * D;
    const bf16* br = blk + i * rs;
    float u[DM_CH][8], p[DM_CH][8], ww[DM_CH][8];
    {
        float a[DM_CH][8], b[DM_CH][8];
        dm_load(wn, tid, D, a);
        dm_load(wp, tid, D, b);
        dm_load(xr, tid, D, p);
        dm_load(w, tid, D, ww);
#pragma unroll
        for (int j = 0; j < DM_CH; j++)
#pragma unroll
            for (int e = 0; e < 8; e++) u[j][e] = a[j][e] * b[j][e];
        if (o) {
            dm_load(o + i * D, tid, D, a);
#pragma unroll
            for (int j = 0; j < DM_CH; j++)
#pragma unroll
                for (int e = 0; e < 8; e++) p[j][e] = b2f(f2b(p[j][e] + a[j][e]));
        }
    }
    // (the sources' sums, four sources' loads at a time: a source's own address past the last one, its sums left out)
    float dot[DM_SOURCES], sq[DM_SOURCES];
#pragma unroll
    for (int s = 0; s < DM_SOURCES; s++) dot[s] = sq[s] = 0.f;
#pragma unroll
    for (int g = 0; g < DM_SOURCES; g += 4) {
        if (g < nb) {
            float v[4][DM_CH][8];
#pragma unroll
            for (int q = 0; q < 4; q++) dm_load(br + (size_t)min(g + q, nb - 1) * D, tid, D, v[q]);
#pragma unroll
            for (int q = 0; q < 4; q++)
                if (g + q < nb)
#pragma unroll
                    for (int j = 0; j < DM_CH; j++)
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
    for (int j = 0; j < DM_CH; j++)
#pragma unroll
        for (int e = 0; e < 8; e++) {
            pd = fmaf(u[j][e], p[j][e], pd);
            pq = fmaf(p[j][e], p[j][e], pq);
        }
#pragma unroll
    for (int s = 0; s < DM_SOURCES; s++)
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
        for (int k = 0; k < DM_T / 32; k++) t += red[k][tid];
        tot[tid] = t;
    }
    __syncthreads();
    float a[DM_SOURCES], m = -INFINITY, z = 0.f;
#pragma unroll
    for (int s = 0; s < DM_SOURCES; s++)
        if (s <= nb) m = fmaxf(m, tot[2 * s] * rsqrtf(tot[2 * s + 1] / D + 1e-5f));
#pragma unroll
    for (int s = 0; s < DM_SOURCES; s++) {
        a[s] = s <= nb ? expf(tot[2 * s] * rsqrtf(tot[2 * s + 1] / D + 1e-5f) - m) : 0.f;
        z += a[s];
    }
    float acc[DM_CH][8];
#pragma unroll
    for (int j = 0; j < DM_CH; j++)
#pragma unroll
        for (int e = 0; e < 8; e++) acc[j][e] = 0.f;
#pragma unroll
    for (int g = 0; g < DM_SOURCES; g += 4) {
        if (g < nb) {
            float v[4][DM_CH][8];
#pragma unroll
            for (int q = 0; q < 4; q++) dm_load(br + (size_t)min(g + q, nb - 1) * D, tid, D, v[q]);
#pragma unroll
            for (int q = 0; q < 4; q++)
                if (g + q < nb)
#pragma unroll
                    for (int j = 0; j < DM_CH; j++)
#pragma unroll
                        for (int e = 0; e < 8; e++) acc[j][e] = fmaf(a[g + q], v[q][j][e], acc[j][e]);
        }
    }
    float an = 0.f;  // (the partial sum's weight, a[nb] read without an index into the registers)
#pragma unroll
    for (int s = 0; s < DM_SOURCES; s++)
        if (s == nb) an = a[s];
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < DM_CH; j++) {
#pragma unroll
        for (int e = 0; e < 8; e++) {
            acc[j][e] = b2f(f2b(fmaf(an, p[j][e], acc[j][e]) / z));
            ss = fmaf(acc[j][e], acc[j][e], ss);
        }
        // (a thread writes only the values it read: x and the next source need no barrier)
        if (push) {
            dm_store(push + i * rs, tid, D, j, p[j]);
            const float zero[8] = {};
            dm_store(xr, tid, D, j, zero);
        } else if (o) {
            dm_store(xr, tid, D, j, p[j]);
        }
    }
    const float r = rsqrtf(block_sum(ss) / D + eps);
#pragma unroll
    for (int j = 0; j < DM_CH; j++) {
        float h[8];
#pragma unroll
        for (int e = 0; e < 8; e++) h[e] = ww[j][e] * b2f(f2b(acc[j][e] * r));
        dm_store(y + i * ys, tid, D, j, h);
    }
}
