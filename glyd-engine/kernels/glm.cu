// GLM-5.3-Flash's own kernels (included by engine.cu after qwen35.cu and moe.cu; src/model_glm.rs runs them): Kimi delta attention (the delta rule with a decay for each key channel, one token after another and in chunks, and the
// gated norm of its output) and the hyper-connections (mHC: the four residual streams' mixing weights, the collapse into a layer's input, the update of the streams from its output, the final mean). bf16 in memory, fp32
// inside; the rounding points are transformers 5.19.0's (modeling_glm5_next.py). The causal convolution of q, k and v is the engine's `conv_silu` (qwen35.cu: the same operation, three convolutions side by side),
// the products are the library's. Reference: tools/glm_ref.py.
// Shapes: a head has GM_D = 128 channels, the state 128 x 128 (S[i][j]: i over the keys' channels, j over the values'); the layer's q, k and v are H heads each, in rows [q | k | v] of 3 H 128.

#define GM_D 128      // channels of a head
#define GM_HC 4       // residual streams
#define GM_NP 24      // rows of the mixing projection: (2 + 4) * 4 (the collapse weights, the spreading weights, the 4 x 4 mixing matrix)
#define GM_PER 16     // elements of a row a thread holds in the norms (D up to GM_PER * (blockDim.x - 32): 7,680 with 512 threads, gm_hc_tail's collapse on warps 1 on)
#define GM_L 32       // tokens a chunk of the chunked delta rule
#define GM_BV 64      // value columns a block of the chunked delta rule
#define GM_KS 129     // a token's row in shared memory: 128 floats and one more, so that 32 lanes reading 32 rows meet no bank twice
#define GM_LS 33
#define GM_CHUNK_SMEM ((3 * GM_L * GM_KS + GM_D * GM_BV + GM_L * GM_BV + 2 * GM_L * GM_LS + GM_L) * 4)  // bytes of dynamic shared memory (99,072)
#define GM_QSCALE 0.08838834764831845f  // 128^-0.5

// (the emulator, tools/glm_emu.cpp, has a static array where the kernel has dynamic shared memory)
#ifdef GLM_EMU
#define GM_DYN_SMEM static __align__(16) float smf[GM_CHUNK_SMEM / 4];
#else
#define GM_DYN_SMEM extern __shared__ __align__(16) float smf[];
#endif
// a float array in dynamic shared memory (the emulator: a static array of 16,384 floats)
#ifdef GLM_EMU
#define GM_DYN_FLOATS(name) static float name[16384];
#else
#define GM_DYN_FLOATS(name) extern __shared__ float name[];
#endif

__device__ __forceinline__ float gm_sig(float x) { return 1.f / (1.f + expf(-x)); }

// ---- the streams -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// x [n, 4, D] <- e [n, D] four times: every stream starts as the token's embedding. Block (row, 256 columns).
extern "C" __global__ void glm_streams(const bf16* e, bf16* x, int D) {
    const int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (d >= D) return;
    const bf16 v = e[(size_t)blockIdx.x * D + d];
#pragma unroll
    for (int s = 0; s < GM_HC; s++) x[((size_t)blockIdx.x * GM_HC + s) * D + d] = v;
}

// ---- the mixing weights and the collapse -----------------------------------------------------------------------------------------------------------------------------------------------
// The rest of a mixing kernel, once tot [GM_NP + 1] (the 24 projections' sums and the sum of squares of the row) is in shared memory: the weights, `iters` rounds of Sinkhorn-Knopp, the collapse and the norm. Called by a
// block of 512 threads a row (glm_hc_mix and glm_hc_fin); pp [GM_NP] is shared memory too.
__device__ __forceinline__ void gm_hc_tail(const bf16* xr, const float* base, const float* scale, const bf16* nw, bf16* h, float* hcp, int t, int D, float eps, float hc_eps, int iters, float* tot, float* pp, int single) {
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
    const int K = GM_HC * D;
    // The collapse's weights first (pre: all the layer's input needs); then warp 0 makes comb by Sinkhorn-Knopp while warps 1 on collapse the row and norm it, at once (the
    // norm's sum over those warps alone, a barrier of their own): Sinkhorn's serial rounds had kept the block's other 508 threads waiting at a barrier.
    __shared__ float red15[16];
    float r[GM_HC];
    if (tid < GM_HC) {
        const int i = tid;
        const float rs = rsqrtf(tot[GM_NP] / K + eps);
        pp[i] = gm_sig(tot[i] * rs * scale[0] + base[i]) + hc_eps;
        pp[GM_HC + i] = 2.f * gm_sig(tot[GM_HC + i] * rs * scale[1] + base[GM_HC + i]);
        float z[GM_HC], mx = -INFINITY, sum = 0.f;
#pragma unroll
        for (int j = 0; j < GM_HC; j++) {
            z[j] = tot[2 * GM_HC + GM_HC * i + j] * rs * scale[2] + base[2 * GM_HC + GM_HC * i + j];
            mx = fmaxf(mx, z[j]);
        }
#pragma unroll
        for (int j = 0; j < GM_HC; j++) {
            z[j] = expf(z[j] - mx);
            sum += z[j];
        }
#pragma unroll
        for (int j = 0; j < GM_HC; j++) r[j] = z[j] / sum + hc_eps;
    }
    __syncthreads();
    if (w == 0) {
        // (four threads, a row of comb each, in the order one thread took: a row's sums and a column's in order of index, every division by sum + hc_eps)
        if (lane < GM_HC) {
            const int rounds = iters > 0 ? iters : 1;
            for (int it = 0; it < rounds; it++) {
                if (it > 0) {
                    float s = 0.f;
#pragma unroll
                    for (int j = 0; j < GM_HC; j++) s += r[j];
#pragma unroll
                    for (int j = 0; j < GM_HC; j++) r[j] /= (s + hc_eps);
                }
                // (every column's sum on every thread, from all four rows by shuffles: in the order of the rows, as one thread added them)
                float all[GM_HC][GM_HC];
#pragma unroll
                for (int q = 0; q < GM_HC; q++)
#pragma unroll
                    for (int j = 0; j < GM_HC; j++) all[q][j] = __shfl_sync(0xFu, r[j], q, GM_HC);
#pragma unroll
                for (int j = 0; j < GM_HC; j++) {
                    float s = 0.f;
#pragma unroll
                    for (int q = 0; q < GM_HC; q++) s += all[q][j];
                    r[j] /= (s + hc_eps);
                }
            }
#pragma unroll
            for (int j = 0; j < GM_HC; j++) pp[2 * GM_HC + GM_HC * lane + j] = r[j];
        }
        __syncwarp();
        if (lane < GM_NP) hcp[(size_t)t * GM_NP + lane] = pp[lane];
        return;
    }
    // the collapse and the norm: warps 1 on (NT threads)
    const int u = tid - 32, NT = blockDim.x - 32;
    float cv[GM_PER], ss2 = 0.f;
#pragma unroll
    for (int j = 0; j < GM_PER; j++) {
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
    for (int j = 0; j < GM_PER; j++) {
        const int d = u + j * NT;
        // (single: the weight times the normed value rounded once, DeepSeek-V4's; else transformers', the normed value rounded first)
        if (d < D) h[(size_t)t * D + d] = f2b(b2f(nw[d]) * (single ? cv[j] * rn : b2f(f2b(cv[j] * rn))));
    }
}

// For each row t of the streams x [n, 4, D] (bf16; Glm5NextTextHyperConnection and the norm that follows it): the 4D values normed without a weight (fp32, eps inside the root) are projected by fn [24, 4D] (bf16), the 24
// results into pre = sigmoid(w scale0 + base) + hc_eps (4), post = 2 sigmoid(w scale1 + base) (4) and comb = softmax(w scale2 + base) + hc_eps (4 x 4: each row of the softmax over its 4 columns), which `iters` rounds of
// Sinkhorn-Knopp make doubly stochastic (the columns' sums divided out first, once even where `iters` is 0, then iters - 1 rounds of the rows' and the columns': every division by sum + hc_eps); hcp [n, 24] takes pre, post and comb (fp32). The row collapses
// to the layer's input, sum over streams of pre * x rounded to bf16, which is normed with the weight nw [D] (transformers' RMSNorm: the normed value rounded to bf16, times the weight, rounded again) into h [n, D].
// A block a row, 512 threads; the projection is a thread's share of the 4D columns for all 24 rows at once.
extern "C" __global__ void __launch_bounds__(512) glm_hc_mix(const bf16* x, const bf16* fn, const float* base, const float* scale, const bf16* nw, bf16* h, float* hcp, int D, float eps, float hc_eps, int iters) {
    const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, w = tid >> 5, nwarps = blockDim.x >> 5;
    const int K = GM_HC * D;
    const bf16* xr = x + (size_t)t * K;
    __shared__ float red[GM_NP + 1][16];
    __shared__ float tot[GM_NP + 1];
    __shared__ float pp[GM_NP];
    float acc[GM_NP], ss = 0.f;
#pragma unroll
    for (int r = 0; r < GM_NP; r++) acc[r] = 0.f;
    for (int e = tid * 2; e < K; e += blockDim.x * 2) {
        const float2 xv = bf2(*reinterpret_cast<const uint32_t*>(xr + e));
        ss = fmaf(xv.x, xv.x, fmaf(xv.y, xv.y, ss));
#pragma unroll
        for (int r = 0; r < GM_NP; r++) {
            const float2 fv = bf2(*reinterpret_cast<const uint32_t*>(fn + (size_t)r * K + e));
            acc[r] = fmaf(xv.x, fv.x, fmaf(xv.y, fv.y, acc[r]));
        }
    }
#pragma unroll
    for (int r = 0; r < GM_NP; r++) {
        const float v = warp_sum(acc[r]);
        if (lane == 0) red[r][w] = v;
    }
    ss = warp_sum(ss);
    if (lane == 0) red[GM_NP][w] = ss;
    __syncthreads();
    if (tid <= GM_NP) {
        float s = 0.f;
        for (int i = 0; i < nwarps; i++) s += red[tid][i];
        tot[tid] = s;
    }
    __syncthreads();
    gm_hc_tail(xr, base, scale, nw, h, hcp, t, D, eps, hc_eps, iters, tot, pp, 0);
}

#define GM_HC_CPB 512  // columns of a row of streams a block of glm_hc_proj takes

// The mixing projection for few rows (a decode step, the rows of a verified draft), split over the row's columns so that many blocks read the 24 x 4D matrix at once (one block a row, glm_hc_mix, reads it at the speed of
// one SM): block (row t, split s) of GM_HC_CPB columns writes the 24 products' partial sums and the sum of squares of its columns to part [n][4D / GM_HC_CPB][GM_NP + 1] (fp32). 128 threads; 4D a multiple of GM_HC_CPB.
extern "C" __global__ void __launch_bounds__(128) glm_hc_proj(const bf16* x, const bf16* fn, float* part, int D) {
    const int t = blockIdx.x, s = blockIdx.y, tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int K = GM_HC * D, KS = K / GM_HC_CPB;
    const bf16* xr = x + (size_t)t * K + (size_t)s * GM_HC_CPB;
    const bf16* fr = fn + (size_t)s * GM_HC_CPB;
    __shared__ float red[GM_NP + 1][4];
    float acc[GM_NP], ss = 0.f;
#pragma unroll
    for (int r = 0; r < GM_NP; r++) acc[r] = 0.f;
    for (int e = tid * 2; e < GM_HC_CPB; e += 256) {
        const float2 xv = bf2(*reinterpret_cast<const uint32_t*>(xr + e));
        ss = fmaf(xv.x, xv.x, fmaf(xv.y, xv.y, ss));
        uint32_t fv[GM_NP];  // (all the rows' loads before any product, as v4_hc_proj)
#pragma unroll
        for (int r = 0; r < GM_NP; r++) fv[r] = __ldg(reinterpret_cast<const unsigned int*>(fr + (size_t)r * K + e));
#pragma unroll
        for (int r = 0; r < GM_NP; r++) {
            const float2 f = bf2(fv[r]);
            acc[r] = fmaf(xv.x, f.x, fmaf(xv.y, f.y, acc[r]));
        }
    }
#pragma unroll
    for (int r = 0; r < GM_NP; r++) {
        const float v = warp_sum(acc[r]);
        if (lane == 0) red[r][w] = v;
    }
    ss = warp_sum(ss);
    if (lane == 0) red[GM_NP][w] = ss;
    __syncthreads();
    if (tid <= GM_NP) part[((size_t)t * KS + s) * (GM_NP + 1) + tid] = (red[tid][0] + red[tid][1]) + (red[tid][2] + red[tid][3]);
}

// The rest of glm_hc_mix for rows whose partial sums glm_hc_proj wrote (KS of them a row, added in order): arguments as glm_hc_mix's. A block a row, 512 threads.
extern "C" __global__ void __launch_bounds__(512) glm_hc_fin(const bf16* x, const float* part, int KS, const float* base, const float* scale, const bf16* nw, bf16* h, float* hcp, int D, float eps, float hc_eps, int iters) {
    const int t = blockIdx.x, tid = threadIdx.x;
    __shared__ float tot[GM_NP + 1];
    __shared__ float pp[GM_NP];
    if (tid <= GM_NP) {
        float v = 0.f;
        for (int k = 0; k < KS; k++) v += part[((size_t)t * KS + k) * (GM_NP + 1) + tid];
        tot[tid] = v;
    }
    __syncthreads();
    gm_hc_tail(x + (size_t)t * GM_HC * D, base, scale, nw, h, hcp, t, D, eps, hc_eps, iters, tot, pp, 0);
}

// x [n, 4, D] in place: stream i of a row becomes bf16(bf16(post_i * o) + bf16(sum over j of comb[j][i] * x_j)), the layer's output o [n, D] spread over the streams and the old streams mixed, post and comb (hcp [n, 24]'s,
// fp32) rounded to bf16 first, as transformers computes it in bf16 (post.to(dtype) * out + comb.to(dtype)^T @ residual). A thread a column of a row, all four streams.
extern "C" __global__ void glm_hc_post(bf16* __restrict__ x, const bf16* __restrict__ o, const float* __restrict__ hcp, int D) {  // (restrict: the weights loaded once, not again after each stream's store)
    const int t = blockIdx.x, d = blockIdx.y * blockDim.x + threadIdx.x;
    if (d >= D) return;
    const float* p = hcp + (size_t)t * GM_NP;
    bf16* xr = x + (size_t)t * GM_HC * D;
    float xv[GM_HC];
#pragma unroll
    for (int j = 0; j < GM_HC; j++) xv[j] = b2f(xr[(size_t)j * D + d]);
    const float ov = b2f(o[(size_t)t * D + d]);
#pragma unroll
    for (int i = 0; i < GM_HC; i++) {
        const float a = b2f(f2b(b2f(f2b(p[GM_HC + i])) * ov));
        float m = 0.f;
#pragma unroll
        for (int j = 0; j < GM_HC; j++) m = fmaf(b2f(f2b(p[2 * GM_HC + GM_HC * j + i])), xv[j], m);
        xr[(size_t)i * D + d] = f2b(a + b2f(f2b(m)));
    }
}

// out [rows, D] = the model's last step: the mean of a row's four streams (x [rows, 4, D]; the sum in fp32 over four, rounded to bf16 as torch's mean of a bf16 tensor is), then the final norm with the weight w [D] (as
// glm_hc_mix's). A block a row, 512 threads.
extern "C" __global__ void __launch_bounds__(512) glm_hc_head(const bf16* x, const bf16* w, bf16* out, int D, float eps) {
    const bf16* xr = x + (size_t)blockIdx.x * GM_HC * D;
    float cv[GM_PER], ss = 0.f;
#pragma unroll
    for (int j = 0; j < GM_PER; j++) {
        const int d = threadIdx.x + j * blockDim.x;
        float c = 0.f;
        if (d < D) c = b2f(f2b(((b2f(xr[d]) + b2f(xr[D + d])) + (b2f(xr[2 * D + d]) + b2f(xr[3 * D + d]))) * 0.25f));
        cv[j] = c;
        ss += c * c;
    }
    const float r = rsqrtf(block_sum(ss) / D + eps);
#pragma unroll
    for (int j = 0; j < GM_PER; j++) {
        const int d = threadIdx.x + j * blockDim.x;
        if (d < D) out[(size_t)blockIdx.x * D + d] = f2b(b2f(w[d]) * b2f(f2b(cv[j] * r)));
    }
}

// ---- Kimi delta attention --------------------------------------------------------------------------------------------------------------------------------------------------------------
// The delta rule with a decay for each key channel, one token after another, for the n rows of a sequence: xc [n, 3 H 128] the convolution's output (q, k, v: H heads of 128 each), fg [n, H 128] the forget gate's second
// product (bf16), dtb [H 128] and alog [H] fp32, bb [n, H] the input gate's product (bf16), state fp32 [H, 128, 128] (S[i][j]), read and written back; core [n, H 128] the outputs (bf16, before the gated norm). Per token
// and head: q and k l2-normed (x / sqrt(sum x^2 + 1e-6); q times 128^-0.5), beta = sigmoid(b) rounded to bf16, g[i] = bound * sigmoid(exp(A_log) * (fg[i] + dt_bias[i])) (fp32, the bound -5), S = diag(exp(g)) S;
// S += k (beta (v - S^T k))^T; o = S^T q. A block a head, a thread a column of S (in registers); the heads' gates are one a thread, shared through shared memory.
extern "C" __global__ void __launch_bounds__(128, 1) glm_kda_rec(const bf16* xc, const bf16* fg, const float* dtb, const float* alog, const bf16* bb, float* state, bf16* core, int n, int H, float bound) {
    const int h = blockIdx.x, j = threadIdx.x, lane = j & 31, w = j >> 5;
    const int qd = H * GM_D;
    __shared__ __align__(16) float qk[3][GM_D];
    __shared__ float red[2][4];
    float* st = state + (size_t)h * GM_D * GM_D;
    float S[GM_D];
#pragma unroll
    for (int i = 0; i < GM_D; i++) S[i] = st[i * GM_D + j];
    const float Ah = expf(alog[h]), dtj = dtb[h * GM_D + j];
    for (int t = 0; t < n; t++) {
        const bf16* x = xc + (size_t)t * 3 * qd + h * GM_D;
        const float qv = b2f(x[j]), kv = b2f(x[qd + j]), vv = b2f(x[2 * qd + j]);
        float sq = warp_sum(qv * qv), sk = warp_sum(kv * kv);
        if (lane == 0) {
            red[0][w] = sq;
            red[1][w] = sk;
        }
        const float gj = bound * gm_sig(Ah * (b2f(fg[(size_t)t * qd + h * GM_D + j]) + dtj));
        const float beta = b2f(f2b(gm_sig(b2f(bb[(size_t)t * H + h]))));
        __syncthreads();
        sq = (red[0][0] + red[0][1]) + (red[0][2] + red[0][3]);
        sk = (red[1][0] + red[1][1]) + (red[1][2] + red[1][3]);
        qk[0][j] = qv * rsqrtf(sq + 1e-6f) * GM_QSCALE;
        qk[1][j] = kv * rsqrtf(sk + 1e-6f);
        qk[2][j] = expf(gj);
        __syncthreads();
        const float4* q4 = reinterpret_cast<const float4*>(qk[0]);
        const float4* k4 = reinterpret_cast<const float4*>(qk[1]);
        const float4* d4 = reinterpret_cast<const float4*>(qk[2]);
        float m0 = 0.f, m1 = 0.f, m2 = 0.f, m3 = 0.f;
#pragma unroll
        for (int i = 0; i < GM_D / 4; i++) {
            const float4 kk = k4[i], dd = d4[i];
            S[4 * i] *= dd.x;
            S[4 * i + 1] *= dd.y;
            S[4 * i + 2] *= dd.z;
            S[4 * i + 3] *= dd.w;
            m0 = fmaf(S[4 * i], kk.x, m0);
            m1 = fmaf(S[4 * i + 1], kk.y, m1);
            m2 = fmaf(S[4 * i + 2], kk.z, m2);
            m3 = fmaf(S[4 * i + 3], kk.w, m3);
        }
        const float d = (vv - ((m0 + m1) + (m2 + m3))) * beta;
        float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
#pragma unroll
        for (int i = 0; i < GM_D / 4; i++) {
            const float4 kk = k4[i], qq = q4[i];
            S[4 * i] = fmaf(kk.x, d, S[4 * i]);
            S[4 * i + 1] = fmaf(kk.y, d, S[4 * i + 1]);
            S[4 * i + 2] = fmaf(kk.z, d, S[4 * i + 2]);
            S[4 * i + 3] = fmaf(kk.w, d, S[4 * i + 3]);
            o0 = fmaf(S[4 * i], qq.x, o0);
            o1 = fmaf(S[4 * i + 1], qq.y, o1);
            o2 = fmaf(S[4 * i + 2], qq.z, o2);
            o3 = fmaf(S[4 * i + 3], qq.w, o3);
        }
        core[(size_t)t * qd + h * GM_D + j] = f2b((o0 + o1) + (o2 + o3));
    }
#pragma unroll
    for (int i = 0; i < GM_D; i++) st[i * GM_D + j] = S[i];
}

// The same rule over chunks of GM_L tokens (the WY form of the gated delta rule with a decay for each key channel: flash-linear-attention's chunk_kda, transformers' chunk_kimi_delta_attention), for a prompt. With
// G the channels' cumulative log-decays inside the chunk, S0 the state at its start and every exponent a difference of two G (never positive): A[t][s] = beta_t sum_i k_t[i] k_s[i] exp(G_t[i] - G_s[i]) (s < t),
// Y = beta (V - (k exp(G)) S0), (I + A) Vn = Y, P[t][s] = sum_i q_t[i] k_s[i] exp(G_t[i] - G_s[i]) (s <= t), the outputs O = (q exp(G)) S0 + P Vn and the state exp(G_last) S0 + sum_s (k_s exp(G_last - G_s)) Vn_s^T.
// Arguments as glm_kda_rec's; fp32 throughout (q and k normed as the recurrence does). A block a head and GM_BV of its value columns (grid H x 2), 256 threads; dynamic shared memory GM_CHUNK_SMEM.
// ponytail: the exponents are one exp for each pair and each use (about 270,000 a chunk and head), not measured against the recurrence on a GPU; flash-linear-attention's sub-chunks of 16 with a reference point turn the
// off-diagonal blocks into products (on tensor cores). A chunk's pairwise sums are a thread a token (the lane) and the keys s = w, w + 8, ...: the lanes whose key is past the token idle (half of the slots).
extern "C" __global__ void __launch_bounds__(256, 1) glm_kda_chunk(const bf16* xc, const bf16* fg, const float* dtb, const float* alog, const bf16* bb, float* state, bf16* core, int n, int H, float bound) {
    GM_DYN_SMEM
    float* Kn = smf;
    float* Qn = Kn + GM_L * GM_KS;
    float* Gc = Qn + GM_L * GM_KS;
    float* S = Gc + GM_L * GM_KS;
    float* V = S + GM_D * GM_BV;
    float* Am = V + GM_L * GM_BV;
    float* Pm = Am + GM_L * GM_LS;
    float* Bt = Pm + GM_L * GM_LS;
    const int h = blockIdx.x, cb = blockIdx.y, tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int qd = H * GM_D;
    float* st = state + (size_t)h * GM_D * GM_D + cb * GM_BV;
    for (int e = tid; e < GM_D * GM_BV; e += 256) S[e] = st[(e / GM_BV) * GM_D + e % GM_BV];
    const float Ah = expf(alog[h]);
    for (int t0 = 0; t0 < n; t0 += GM_L) {
        const int tc = min(GM_L, n - t0);
        __syncthreads();  // (the chunk before is done with every buffer)
        // 1. the chunk's queries and keys normed, its decays and betas, its values (a token past the end of the sequence: zeros, which change nothing)
        for (int t = w; t < GM_L; t += 8) {
            float qv[4], kv[4], gv[4], sq = 0.f, sk = 0.f;
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int i = lane + 32 * e;
                const bool ok = t < tc;
                qv[e] = ok ? b2f(xc[(size_t)(t0 + t) * 3 * qd + h * GM_D + i]) : 0.f;
                kv[e] = ok ? b2f(xc[(size_t)(t0 + t) * 3 * qd + qd + h * GM_D + i]) : 0.f;
                gv[e] = ok ? bound * gm_sig(Ah * (b2f(fg[(size_t)(t0 + t) * qd + h * GM_D + i]) + dtb[h * GM_D + i])) : 0.f;
                sq += qv[e] * qv[e];
                sk += kv[e] * kv[e];
            }
            const float rq = rsqrtf(warp_sum(sq) + 1e-6f), rk = rsqrtf(warp_sum(sk) + 1e-6f);
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int i = lane + 32 * e;
                Qn[t * GM_KS + i] = qv[e] * rq * GM_QSCALE;
                Kn[t * GM_KS + i] = kv[e] * rk;
                Gc[t * GM_KS + i] = gv[e];
            }
            if (lane == 0) Bt[t] = t < tc ? b2f(f2b(gm_sig(b2f(bb[(size_t)(t0 + t) * H + h])))) : 0.f;
        }
        for (int e = tid; e < GM_L * GM_BV; e += 256) {
            const int t = e / GM_BV;
            V[e] = t < tc ? b2f(xc[(size_t)(t0 + t) * 3 * qd + 2 * qd + h * GM_D + cb * GM_BV + e % GM_BV]) : 0.f;
        }
        __syncthreads();
        // 2. G: the decays summed over the chunk's tokens, a channel a thread
        if (tid < GM_D) {
            float a = 0.f;
            for (int t = 0; t < GM_L; t++) {
                a += Gc[t * GM_KS + tid];
                Gc[t * GM_KS + tid] = a;
            }
        }
        __syncthreads();
        // 3. A and P: a thread the token t = lane and the keys s = w + 8 r
        {
            const int t = lane;
            float a4[4] = {0.f, 0.f, 0.f, 0.f}, p4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int r = 0; r < 4; r++) {
                const int s = w + 8 * r;
                if (s <= t) {
                    float a = 0.f, p = 0.f;
                    for (int i = 0; i < GM_D; i++) {
                        const float ks = Kn[s * GM_KS + i], e = expf(Gc[t * GM_KS + i] - Gc[s * GM_KS + i]);
                        a = fmaf(Kn[t * GM_KS + i] * ks, e, a);
                        p = fmaf(Qn[t * GM_KS + i] * ks, e, p);
                    }
                    a4[r] = a;
                    p4[r] = p;
                }
            }
#pragma unroll
            for (int r = 0; r < 4; r++) {
                const int s = w + 8 * r;
                Am[t * GM_LS + s] = s < t ? Bt[t] * a4[r] : 0.f;
                Pm[t * GM_LS + s] = s <= t ? p4[r] : 0.f;
            }
        }
        __syncthreads();
        // 4. the queries times exp(G) (in place), and Y = beta (V - (k exp(G)) S0) into V: a thread the token t = lane and eight value columns 8 w ...
        for (int e = tid; e < GM_L * GM_D; e += 256) Qn[(e / GM_D) * GM_KS + e % GM_D] *= expf(Gc[(e / GM_D) * GM_KS + e % GM_D]);
        {
            const int t = lane;
            float a8[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            for (int i = 0; i < GM_D; i++) {
                const float kg = Kn[t * GM_KS + i] * expf(Gc[t * GM_KS + i]);
                const float4 s0 = *reinterpret_cast<const float4*>(&S[i * GM_BV + 8 * w]), s1 = *reinterpret_cast<const float4*>(&S[i * GM_BV + 8 * w + 4]);
                a8[0] = fmaf(kg, s0.x, a8[0]);
                a8[1] = fmaf(kg, s0.y, a8[1]);
                a8[2] = fmaf(kg, s0.z, a8[2]);
                a8[3] = fmaf(kg, s0.w, a8[3]);
                a8[4] = fmaf(kg, s1.x, a8[4]);
                a8[5] = fmaf(kg, s1.y, a8[5]);
                a8[6] = fmaf(kg, s1.z, a8[6]);
                a8[7] = fmaf(kg, s1.w, a8[7]);
            }
#pragma unroll
            for (int c = 0; c < 8; c++) V[t * GM_BV + 8 * w + c] = Bt[t] * (V[t * GM_BV + 8 * w + c] - a8[c]);
        }
        __syncthreads();
        // 5. (I + A) Vn = Y, a row at a time (a column a thread)
        if (tid < GM_BV) {
            for (int t = 1; t < GM_L; t++) {
                float a = V[t * GM_BV + tid];
                for (int s = 0; s < t; s++) a = fmaf(-Am[t * GM_LS + s], V[s * GM_BV + tid], a);
                V[t * GM_BV + tid] = a;
            }
        }
        __syncthreads();
        // 6. O = (q exp(G)) S0 + P Vn, a thread the token t = lane and eight value columns
        {
            const int t = lane;
            float a8[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            for (int i = 0; i < GM_D; i++) {
                const float qg = Qn[t * GM_KS + i];
                const float4 s0 = *reinterpret_cast<const float4*>(&S[i * GM_BV + 8 * w]), s1 = *reinterpret_cast<const float4*>(&S[i * GM_BV + 8 * w + 4]);
                a8[0] = fmaf(qg, s0.x, a8[0]);
                a8[1] = fmaf(qg, s0.y, a8[1]);
                a8[2] = fmaf(qg, s0.z, a8[2]);
                a8[3] = fmaf(qg, s0.w, a8[3]);
                a8[4] = fmaf(qg, s1.x, a8[4]);
                a8[5] = fmaf(qg, s1.y, a8[5]);
                a8[6] = fmaf(qg, s1.z, a8[6]);
                a8[7] = fmaf(qg, s1.w, a8[7]);
            }
            for (int s = 0; s <= t; s++) {
                const float p = Pm[t * GM_LS + s];
#pragma unroll
                for (int c = 0; c < 8; c++) a8[c] = fmaf(p, V[s * GM_BV + 8 * w + c], a8[c]);
            }
            if (t < tc) {
#pragma unroll
                for (int c = 0; c < 8; c++) core[(size_t)(t0 + t) * qd + h * GM_D + cb * GM_BV + 8 * w + c] = f2b(a8[c]);
            }
        }
        __syncthreads();
        // 7. S = exp(G_last) S0 + sum_s (k_s exp(G_last - G_s)) Vn_s^T: the keys scaled into Qn (the queries are done with), then a thread 8 rows by 4 columns of S
        for (int e = tid; e < GM_L * GM_D; e += 256) {
            const int t = e / GM_D, i = e % GM_D;
            Qn[t * GM_KS + i] = Kn[t * GM_KS + i] * expf(Gc[(GM_L - 1) * GM_KS + i] - Gc[t * GM_KS + i]);
        }
        __syncthreads();
        {
            const int ty = tid >> 4, tx = tid & 15;
            float a8[8][4];
#pragma unroll
            for (int r = 0; r < 8; r++) {
                const float el = expf(Gc[(GM_L - 1) * GM_KS + ty + 16 * r]);
#pragma unroll
                for (int c = 0; c < 4; c++) a8[r][c] = S[(ty + 16 * r) * GM_BV + tx + 16 * c] * el;
            }
            for (int s = 0; s < GM_L; s++) {
                float kd[8], u[4];
#pragma unroll
                for (int r = 0; r < 8; r++) kd[r] = Qn[s * GM_KS + ty + 16 * r];
#pragma unroll
                for (int c = 0; c < 4; c++) u[c] = V[s * GM_BV + tx + 16 * c];
#pragma unroll
                for (int r = 0; r < 8; r++)
#pragma unroll
                    for (int c = 0; c < 4; c++) a8[r][c] = fmaf(kd[r], u[c], a8[r][c]);
            }
#pragma unroll
            for (int r = 0; r < 8; r++)
#pragma unroll
                for (int c = 0; c < 4; c++) S[(ty + 16 * r) * GM_BV + tx + 16 * c] = a8[r][c];
        }
    }
    __syncthreads();
    for (int e = tid; e < GM_D * GM_BV; e += 256) st[(e / GM_BV) * GM_D + e % GM_BV] = S[e];
}

// att [n, H 128] = the gated norm of the delta rule's outputs core (bf16): per head the rms norm of its 128 channels (fp32, eps inside the root), times the weight w [128], times sigmoid(og) (og [n, H 128], the output
// gate's second product), one rounding (transformers' RMSNormGated with the sigmoid activation: the weight times the normed value, then the gate). A block (row, head), 128 threads.
template <typename W>
__device__ __forceinline__ void kda_out_of(const bf16* core, const bf16* og, const W* w, bf16* att, int H, float eps) {
    const size_t at = ((size_t)blockIdx.x * H + blockIdx.y) * GM_D + threadIdx.x;
    const float x = b2f(core[at]);
    const float r = rsqrtf(block_sum(x * x) / (float)GM_D + eps);
    att[at] = f2b(wval(w[threadIdx.x]) * (x * r) * gm_sig(b2f(og[at])));
}

extern "C" __global__ void glm_kda_out(const bf16* core, const bf16* og, const bf16* w, bf16* att, int H, float eps) { kda_out_of(core, og, w, att, H, eps); }

// glm_kda_out with an fp32 weight (Kimi-K3's o_norm).
extern "C" __global__ void glm_kda_out_w32(const bf16* core, const bf16* og, const float* w, bf16* att, int H, float eps) { kda_out_of(core, og, w, att, H, eps); }

// (the latent attention over the keys an indexer picks: kernels/mla.cu, the latent attention layer kind's)
