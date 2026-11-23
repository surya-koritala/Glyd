// GLM-5.3-Flash's own kernels (included by engine.cu after qwen35.cu and moe.cu; src/model_glm.rs runs them): Kimi delta attention (the delta rule with a decay for each key channel, one token after another and in chunks, and the
// gated norm of its output) and the latent attention over the keys an indexer picks; its residual streams are the streams kind's (residual.cu). bf16 in memory, fp32
// inside; the rounding points are transformers 5.19.0's (modeling_glm5_next.py). The causal convolution of q, k and v is the engine's `conv_silu` (qwen35.cu: the same operation, three convolutions side by side),
// the products are the library's. Reference: tools/glm_ref.py.
// Shapes: a head has GM_D = 128 channels, the state 128 x 128 (S[i][j]: i over the keys' channels, j over the values'); the layer's q, k and v are H heads each, in rows [q | k | v] of 3 H 128.

#define GM_D 128      // channels of a head
#define GM_PER 16     // elements of a row a thread holds in the norms (K up to GM_PER * blockDim.x)
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

// ---- latent attention over the keys an indexer picks ------------------------------------------------------------------------------------------------------------------------------------------
// (Glm5NextTextAttention and Glm5NextTextIndexer; tools/glm_ref.py's mla_* functions are the reference.) No rotary: a token's cache is its latent c (kv_lora_rank wide: kv_a_proj_with_mqa's product, normed), which is its key and
// its value through kv_b_proj, so the attention runs in the latent space (the absorbed form): the query q_h becomes q_abs = Wk_h^T q_h, the scores are q_abs . c_t, the output is (sum_t p_t c_t) multiplied by Wv_h at the end.
// The indexer scores the complete pools of kpool tokens (a pool's key a softmax-weighted mean of its tokens' normed keys) for each query, picks the best ksel = topk / kpool, and the query attends to their tokens and to the
// tokens of the incomplete last pool. A query at position q sees the pools whose last token is q at the latest; where those are no more than ksel, it attends to every token up to q. Every kernel reads the positions of the
// rows from the device (pos), launches over bounds that depend on the engine's context and not on a position, and reads no result on the host: a decode step captures into a CUDA graph.

#define GM_NOPE_MAX 256    // a query head's width (nope) at most
#define GM_KSEL_MAX 1024   // pools picked at most
#define GM_IDX_W 8         // pools a block of glm_idx_score scores (a warp each)
#define GM_RB 8            // rows a block of glm_absorb takes
#define GM_VB 4            // rows a block of glm_expand_v takes

__device__ float block_max(float v) {  // (warp_max is qwen35.cu's)
    __shared__ float shm[32];
    __syncthreads();
    v = warp_max(v);
    if ((threadIdx.x & 31) == 0) shm[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        float t = threadIdx.x < (blockDim.x >> 5) ? shm[threadIdx.x] : -INFINITY;
        t = warp_max(t);
        if (threadIdx.x == 0) shm[0] = t;
    }
    __syncthreads();
    return shm[0];
}

// y row (dst ? dst[i] : i) = w * bf16(x row i * rsqrt(mean(x^2) + eps)): transformers' RMSNorm (the normed value rounded to bf16, times the weight, rounded again). x rows xs apart, y rows ys apart; K up to GM_PER * blockDim.x.
// With dst (the positions of the rows) the result lands in a cache's row for its position. A block a row, 256 threads.
extern "C" __global__ void __launch_bounds__(256) glm_rms2(const bf16* x, int xs, const bf16* w, bf16* y, int ys, const int* dst, int K, float eps) {
    const bf16* xr = x + (size_t)blockIdx.x * xs;
    bf16* yr = y + (dst ? (size_t)dst[blockIdx.x] : (size_t)blockIdx.x) * ys;
    float v[GM_PER], ss = 0.f;
#pragma unroll
    for (int j = 0; j < GM_PER; j++) {
        const int d = threadIdx.x + j * blockDim.x;
        v[j] = d < K ? b2f(xr[d]) : 0.f;
        ss += v[j] * v[j];
    }
    const float r = rsqrtf(block_sum(ss) / K + eps);
#pragma unroll
    for (int j = 0; j < GM_PER; j++) {
        const int d = threadIdx.x + j * blockDim.x;
        if (d < K) yr[d] = f2b(b2f(w[d]) * b2f(f2b(v[j] * r)));
    }
}

// nn.LayerNorm over K (<= 2 blockDim.x) channels: fp32, the mean and the biased variance of the row, (x - mean) * rsqrt(var + eps) * w + b, one rounding; y's row dst[i] where dst. A block a row, 128 threads.
extern "C" __global__ void __launch_bounds__(128) glm_layernorm(const bf16* x, int xs, const bf16* w, const bf16* b, bf16* y, int ys, const int* dst, int K, float eps) {
    const bf16* xr = x + (size_t)blockIdx.x * xs;
    bf16* yr = y + (dst ? (size_t)dst[blockIdx.x] : (size_t)blockIdx.x) * ys;
    float v[2], s = 0.f;
#pragma unroll
    for (int j = 0; j < 2; j++) {
        const int d = threadIdx.x + j * blockDim.x;
        v[j] = d < K ? b2f(xr[d]) : 0.f;
        s += v[j];
    }
    const float mu = block_sum(s) / K;
    float q = 0.f;
#pragma unroll
    for (int j = 0; j < 2; j++) {
        const int d = threadIdx.x + j * blockDim.x;
        const float c = d < K ? v[j] - mu : 0.f;
        q += c * c;
    }
    const float r = rsqrtf(block_sum(q) / K + eps);
#pragma unroll
    for (int j = 0; j < 2; j++) {
        const int d = threadIdx.x + j * blockDim.x;
        if (d < K) yr[d] = f2b((v[j] - mu) * r * b2f(w[d]) + b2f(b[d]));
    }
}

// dst row pos[i] = src row i (K bf16 each, src rows ss apart): the rows of a call into a cache at their positions. A block a row.
extern "C" __global__ void glm_scatter(const bf16* src, int ss, bf16* dst, int ds, const int* pos, int K) {
    const bf16* s = src + (size_t)blockIdx.x * ss;
    bf16* d = dst + (size_t)pos[blockIdx.x] * ds;
    for (int k = threadIdx.x; k < K; k += blockDim.x) d[k] = s[k];
}

// qa [n, NH, KR] fp32 = for each row and head, Wk_h^T q_h: qa[i][h][j] = sum over d < nope of q[i][h nope + d] * wkv[h (nope + vd) + d][j] (wkv = kv_b_proj, bf16 [NH (nope + vd), KR]). A block a head and GM_RB rows (grid NH x
// ceil(n / GM_RB)), 256 threads: KR / 8 threads a group, each 8 columns of a row of the matrix (one 16-byte load), the groups down the head's nope rows by turns, so that a head's 256 rows are read by 4 groups (KR = 512) or
// 32 (KR = 64) at once (a thread a column read them one after another and a decode step's 64 blocks took 78 us); the groups' sums are added in order through shared memory. KR a multiple of 64 up to 512.
// A head's query is qhs wide in q (its nope channels first) and KS wide in qa: past KR its channels nope .. nope + KS - KR copied as they are (Kimi-K3's key part every head shares; GLM: qhs = nope, KS = KR).
extern "C" __global__ void __launch_bounds__(256) glm_absorb(const bf16* q, int qs, int qhs, const bf16* wkv, float* qa, int n, int NH, int nope, int vd, int KR, int KS) {
    const int h = blockIdx.x, r0 = blockIdx.y * GM_RB, tid = threadIdx.x;
    const int CT = KR / 8, G = 256 / CT, g = tid / CT, ct = tid % CT;
    const bool live = g < G;  // (KR / 8 need not divide 256: the threads past the last whole group take no rows)
    __shared__ float qh[GM_RB][GM_NOPE_MAX];
    __shared__ float red[1024 * GM_RB];  // [G][GM_RB / 2 rows at a time][KR] floats: G KR = 2048
    for (int e = tid; e < GM_RB * nope; e += blockDim.x) {
        const int r = e / nope, d = e % nope;
        qh[r][d] = r0 + r < n ? b2f(q[(size_t)(r0 + r) * qs + h * qhs + d]) : 0.f;
    }
    __syncthreads();
    float acc[GM_RB][8];
#pragma unroll
    for (int r = 0; r < GM_RB; r++)
#pragma unroll
        for (int c = 0; c < 8; c++) acc[r][c] = 0.f;
    const bf16* W = wkv + (size_t)h * (nope + vd) * KR + ct * 8;
    for (int d = live ? g : nope; d < nope; d += G) {
        const uint4 u = *reinterpret_cast<const uint4*>(W + (size_t)d * KR);
        const float2 w0 = bf2(u.x), w1 = bf2(u.y), w2 = bf2(u.z), w3 = bf2(u.w);
#pragma unroll
        for (int r = 0; r < GM_RB; r++) {
            const float qv = qh[r][d];
            acc[r][0] = fmaf(qv, w0.x, acc[r][0]);
            acc[r][1] = fmaf(qv, w0.y, acc[r][1]);
            acc[r][2] = fmaf(qv, w1.x, acc[r][2]);
            acc[r][3] = fmaf(qv, w1.y, acc[r][3]);
            acc[r][4] = fmaf(qv, w2.x, acc[r][4]);
            acc[r][5] = fmaf(qv, w2.y, acc[r][5]);
            acc[r][6] = fmaf(qv, w3.x, acc[r][6]);
            acc[r][7] = fmaf(qv, w3.y, acc[r][7]);
        }
    }
    // the groups' sums, GM_RB / 2 rows at a time through shared memory ([G][half][KR] floats: 2048 half floats)
#pragma unroll
    for (int part = 0; part < 2; part++) {
        __syncthreads();
        if (live) {
#pragma unroll
            for (int r = 0; r < GM_RB / 2; r++)
#pragma unroll
                for (int c = 0; c < 8; c++) red[(g * (GM_RB / 2) + r) * KR + ct * 8 + c] = acc[part * (GM_RB / 2) + r][c];
        }
        __syncthreads();
        for (int e = tid; e < (GM_RB / 2) * KR; e += blockDim.x) {
            const int r = e / KR, j = e % KR;
            float v = 0.f;
            for (int gg = 0; gg < G; gg++) v += red[(gg * (GM_RB / 2) + r) * KR + j];
            const int row = r0 + part * (GM_RB / 2) + r;
            if (row < n) qa[((size_t)row * NH + h) * KS + j] = v;
        }
    }
    for (int e = tid; e < GM_RB * (KS - KR); e += blockDim.x) {
        const int r = e / (KS - KR), j = e % (KS - KR), row = r0 + r;
        if (row < n) qa[((size_t)row * NH + h) * KS + KR + j] = b2f(q[(size_t)row * qs + h * qhs + nope + j]);
    }
}

// The keys of the complete pools that this call (rows at pos[0] .. pos[n-1], consecutive) completes, from the cache of the normed keys ik and of the gate scores gs (rows D wide, one a position) and the pooling bias ape
// [kp, D]: pk[p][c] = sum over the pool's kp tokens j of softmax_j(gs[kp p + j][c] + ape[j][c]) * ik[kp p + j][c], rounded to bf16. A block a pool from pos[0] / kp on (grid ceil(n / kp) + 1); a pool whose last token is past
// the call's last row is left (a later call completes it).
extern "C" __global__ void __launch_bounds__(128) glm_kp_keys(const bf16* ik, const bf16* gs, const bf16* ape, bf16* pk, const int* pos, int n, int D, int kp) {
    const int p = pos[0] / kp + blockIdx.x;
    if ((p + 1) * kp - 1 > pos[n - 1]) return;
    for (int c = threadIdx.x; c < D; c += blockDim.x) {
        float l[8], mx = -INFINITY;
        for (int j = 0; j < kp; j++) {
            l[j] = b2f(gs[(size_t)(p * kp + j) * D + c]) + b2f(ape[j * D + c]);
            mx = fmaxf(mx, l[j]);
        }
        float sum = 0.f, acc = 0.f;
        for (int j = 0; j < kp; j++) {
            l[j] = expf(l[j] - mx);
            sum += l[j];
        }
        for (int j = 0; j < kp; j++) acc = fmaf(l[j] / sum, b2f(ik[(size_t)(p * kp + j) * D + c]), acc);
        pk[(size_t)p * D + c] = f2b(acc);
    }
}

// scores [n, Pmax] fp32: for each row i (a query at pos[i]) and each complete pool p < P_i = (pos[i] + 1) / kp: sum over the indexer's heads h of w_h relu(scale iq[i][h] . pk[p]), w_h = wscale wts[i][h] (iq rows iqs apart,
// IH heads of D channels; pk [., D]). Rows with P_i <= ksel take every pool and need no scores. A block of GM_IDX_W pools (a warp each) of a row, grid (ceil(Pmax / GM_IDX_W), n), 256 threads; the lanes are heads (IH > 32: a
// lane takes several), so a lane's sum is over its head's D channels and a warp_sum adds the heads. Dynamic shared memory (IH (D + 1) + IH + GM_IDX_W D) floats.
extern "C" __global__ void __launch_bounds__(256) glm_idx_score(const bf16* iq, int iqs, const bf16* wts, int wsr, const bf16* pk, const int* pos, float* scores, int Pmax, int IH, int D, int kp, int ksel, float scale, float wscale) {
    GM_DYN_FLOATS(smi)
    const int i = blockIdx.y, tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int P = (pos[i] + 1) / kp;
    if (P <= ksel || (int)blockIdx.x * GM_IDX_W >= P) return;
    float* qs = smi;                      // [IH][D + 1]
    float* wh = qs + IH * (D + 1);        // [IH]
    float* pkw = wh + IH;                 // [GM_IDX_W][D]
    for (int e = tid; e < IH * D; e += blockDim.x) qs[(e / D) * (D + 1) + e % D] = b2f(iq[(size_t)i * iqs + e]);
    for (int h = tid; h < IH; h += blockDim.x) wh[h] = b2f(wts[(size_t)i * wsr + h]) * wscale;
    const int p = blockIdx.x * GM_IDX_W + w;
    if (p < P)
        for (int d = lane; d < D; d += 32) pkw[w * D + d] = b2f(pk[(size_t)p * D + d]);
    __syncthreads();
    if (p >= P) return;
    float s = 0.f;
    for (int h = lane; h < IH; h += 32) {
        float dot = 0.f;
        for (int d = 0; d < D; d++) dot = fmaf(qs[h * (D + 1) + d], pkw[w * D + d], dot);
        s = fmaf(wh[h], fmaxf(dot * scale, 0.f), s);
    }
    s = warp_sum(s);
    if (lane == 0) scores[(size_t)i * Pmax + p] = s;
}

__device__ __forceinline__ unsigned gm_key(float f) {  // (the order of the floats as an unsigned number's)
    const unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// The tokens each row attends to: sel [n, Tmax] int32, ascending, the first cnt[i] of a row. A query at q = pos[i] sees P = (q + 1) / kp complete pools; with no more than ksel of them it attends to every token up to q;
// otherwise to the tokens of the ksel best-scoring pools (scores [n, Pmax]; equal scores: the smaller pool), found by a radix select on the scores' bits and sorted by pool, then, where `tail`, the (q + 1) % kp tokens of the
// incomplete pool after them. A block a row, 1024 threads.
extern "C" __global__ void __launch_bounds__(1024) glm_select(const float* scores, int Pmax, const int* pos, int* sel, int* cnt, int Tmax, int kp, int ksel, int tail) {
    const int i = blockIdx.x, tid = threadIdx.x;
    const int q = pos[i], P = (q + 1) / kp, tl = tail ? q + 1 - P * kp : 0;
    int* out = sel + (size_t)i * Tmax;
    if (P <= ksel) {
        const int T = P * kp + tl;
        for (int t = tid; t < T; t += blockDim.x) out[t] = t;
        if (tid == 0) cnt[i] = T;
        return;
    }
    const float* s = scores + (size_t)i * Pmax;
    __shared__ unsigned hist[256];
    __shared__ int chosen, rem, nsel;
    __shared__ int selp[GM_KSEL_MAX];
    __shared__ int scan[1024];
    unsigned prefix = 0, pmask = 0;
    int remaining = ksel;
    for (int pass = 0; pass < 4; pass++) {
        const int shift = 24 - 8 * pass;
        if (tid < 256) hist[tid] = 0;
        __syncthreads();
        for (int p = tid; p < P; p += blockDim.x) {
            const unsigned k = gm_key(s[p]);
            if ((k & pmask) == prefix) atomicAdd(&hist[(k >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (tid < 32) {  // the bucket that holds the remaining-th largest: a lane 8 buckets from the top down, the lanes' totals added in order
            const int base = 255 - 8 * tid;
            unsigned hv[8];
            int sum = 0;
#pragma unroll
            for (int k = 0; k < 8; k++) {
                hv[k] = hist[base - k];
                sum += (int)hv[k];
            }
            int incl = sum;
#pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                const int v = __shfl_up_sync(0xffffffffu, incl, off);
                if (tid >= off) incl += v;
            }
            const int before = incl - sum;
            if (before < remaining && before + sum >= remaining) {
                int acc = before, b = base - 7;
                for (int k = 0; k < 8; k++) {
                    if (acc + (int)hv[k] >= remaining) {
                        b = base - k;
                        break;
                    }
                    acc += (int)hv[k];
                }
                chosen = b;
                rem = remaining - acc;
            }
        }
        __syncthreads();
        prefix |= (unsigned)chosen << shift;
        pmask |= 0xFFu << shift;
        remaining = rem;
        __syncthreads();
    }
    // every pool whose key is above the threshold `prefix`, and the first `remaining` of those equal to it
    const int cp = (P + blockDim.x - 1) / blockDim.x, lo = min(tid * cp, P), hi = min(lo + cp, P);
    int ceq = 0;
    for (int p = lo; p < hi; p++) ceq += gm_key(s[p]) == prefix;
    scan[tid] = ceq;
    __syncthreads();
    for (int off = 1; off < (int)blockDim.x; off <<= 1) {
        const int v = tid >= off ? scan[tid - off] : 0;
        __syncthreads();
        scan[tid] += v;
        __syncthreads();
    }
    int rank = scan[tid] - ceq;
    if (tid == 0) nsel = 0;
    __syncthreads();
    for (int p = lo; p < hi; p++) {
        const unsigned k = gm_key(s[p]);
        const bool take = k > prefix || (k == prefix && rank < remaining);
        if (k == prefix) rank++;
        if (take) selp[atomicAdd(&nsel, 1)] = p;
    }
    __syncthreads();
    // the pools in order (a bitonic sort of ksel padded to a power of two)
    int n2 = 1;
    while (n2 < ksel) n2 <<= 1;
    for (int j = ksel + tid; j < n2; j += blockDim.x) selp[j] = 0x7fffffff;
    __syncthreads();
    for (int k = 2; k <= n2; k <<= 1)
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int e = tid; e < n2; e += blockDim.x) {
                const int x = e ^ j;
                if (x > e) {
                    const int a = selp[e], b = selp[x];
                    if ((a > b) == ((e & k) == 0)) {
                        selp[e] = b;
                        selp[x] = a;
                    }
                }
            }
            __syncthreads();
        }
    for (int j = tid; j < ksel * kp; j += blockDim.x) out[j] = selp[j / kp] * kp + j % kp;
    for (int u = tid; u < tl; u += blockDim.x) out[ksel * kp + u] = P * kp + u;
    if (tid == 0) cnt[i] = ksel * kp + tl;
}

// olat [n, NH, KR] fp32 = for each row i and head h, softmax over its cnt[i] tokens t of (qa[i][h] . lat[sel[i][t]]) scale, times the latents: sum_t p_t lat[sel[i][t]] (lat [., KR] bf16, a row a position). A block a
// (head, row, split) (grid NH x n x splits), 256 threads; the row's tokens are cut into `splits` runs, a block takes one: a warp four tokens at a time for the scores (bf16 pairs, KR a multiple of 64; the four rows' loads
// are in flight together, where one at a time a decode step's 2,051 tokens took 270 us), the softmax in fp32 over the block's tokens, then a thread two columns of the weighted sum, four tokens at a time. With one split
// the block writes olat; with more, its unnormalised sum, its largest score and the sum of its exponentials to part [n][NH][splits][KR + 2] for glm_mla_merge. Dynamic shared memory (KS + 2 tile) 4-byte words: a block's
// tokens `tile` at a time (GLY-357: Kimi-K3's dense attention over the whole context).
// The scores are over KS channels (qa's heads and lat's rows KS wide), the values the first KR of them (Kimi-K3: the latent's 512 and the shared key part's 64; GLM: KS = KR); sel null: every position from 0
// (a dense attention, cnt[i] = pos + 1).
extern "C" __global__ void __launch_bounds__(256) glm_mla_attn(const float* qa, const bf16* lat, const int* sel, const int* cnt, float* olat, float* part, int Tmax, int NH, int KR, int KS, float scale, int splits, int tile) {
    GM_DYN_FLOATS(sma)
    const int h = blockIdx.x, i = blockIdx.y, sp = blockIdx.z, tid = threadIdx.x, lane = tid & 31, w = tid >> 5, nw = blockDim.x >> 5;
    const int T = cnt[i], per = (T + splits - 1) / splits, t0 = min(T, sp * per), t1 = min(T, t0 + per), m_n = t1 - t0;
    float* qh = sma;                                // [KS]
    float* sc = sma + KS;                           // [tile] a tile's scores, then its weights
    int* idm = reinterpret_cast<int*>(sma + KS + tile);  // [tile] the tile's tokens' positions (a copy: the rows' loads then wait for no other load)
    const int* ids = sel ? sel + (size_t)i * Tmax + t0 : nullptr;
    for (int j = tid; j < KS; j += blockDim.x) qh[j] = qa[((size_t)i * NH + h) * KS + j];
    // the block's tokens a tile at a time (the most the shared memory holds), the softmax's largest score, its sum and the weighted sum carried over and
    // rescaled where a tile's largest is larger: one tile, the arithmetic of a single pass
    float m = -INFINITY, z = 0.f, a0 = 0.f, a1 = 0.f;
    const int j2 = 2 * tid;
    for (int c0 = 0; c0 < m_n; c0 += tile) {
        const int cn = min(tile, m_n - c0);
        for (int t = tid; t < cn; t += blockDim.x) idm[t] = ids ? ids[c0 + t] : t0 + c0 + t;
        __syncthreads();
        for (int tb = w * 4; tb < cn; tb += nw * 4) {
            const bf16* rows[4];
#pragma unroll
            for (int u = 0; u < 4; u++) rows[u] = lat + (size_t)idm[min(tb + u, cn - 1)] * KS;
            float a[4] = {0.f, 0.f, 0.f, 0.f};
            for (int j = 2 * lane; j < KS; j += 64) {
                float2 c[4];
#pragma unroll
                for (int u = 0; u < 4; u++) c[u] = bf2(*reinterpret_cast<const uint32_t*>(rows[u] + j));
                const float q0 = qh[j], q1 = qh[j + 1];
#pragma unroll
                for (int u = 0; u < 4; u++) a[u] = fmaf(q0, c[u].x, fmaf(q1, c[u].y, a[u]));
            }
#pragma unroll
            for (int u = 0; u < 4; u++) {
                const float v = warp_sum(a[u]);
                if (lane == 0 && tb + u < cn) sc[tb + u] = v * scale;
            }
        }
        __syncthreads();
        float mt = -INFINITY;
        for (int t = tid; t < cn; t += blockDim.x) mt = fmaxf(mt, sc[t]);
        mt = fmaxf(m, block_max(mt));
        const float r = m == -INFINITY ? 0.f : expf(m - mt);
        float zt = 0.f;
        for (int t = tid; t < cn; t += blockDim.x) {
            const float e = expf(sc[t] - mt);
            sc[t] = e;
            zt += e;
        }
        z = z * r + block_sum(zt);
        a0 *= r;
        a1 *= r;
        m = mt;
        // the weighted sum: a thread a pair of columns (KR <= 512), eight tokens' loads in flight
        for (int tb = 0; tb < cn; tb += 8) {
            float2 v[8];
            float p[8];
#pragma unroll
            for (int u = 0; u < 8; u++) {
                const bool ok = tb + u < cn;
                p[u] = ok ? sc[tb + u] : 0.f;
                v[u] = j2 < KR ? bf2(*reinterpret_cast<const uint32_t*>(lat + (size_t)idm[ok ? tb + u : cn - 1] * KS + j2)) : make_float2(0.f, 0.f);
            }
#pragma unroll
            for (int u = 0; u < 8; u++) {
                a0 = fmaf(p[u], v[u].x, a0);
                a1 = fmaf(p[u], v[u].y, a1);
            }
        }
        __syncthreads();
    }
    if (splits == 1) {
        const float inv = z > 0.f ? 1.f / z : 0.f;  // (no tokens at all: zeros, as the reference's)
        float* o = olat + ((size_t)i * NH + h) * KR;
        if (j2 < KR) {
            o[j2] = a0 * inv;
            o[j2 + 1] = a1 * inv;
        }
    } else {
        float* o = part + (((size_t)i * NH + h) * splits + sp) * (KR + 2);
        if (j2 < KR) {
            o[j2] = a0;
            o[j2 + 1] = a1;
        }
        if (tid == 0) {
            o[KR] = m;
            o[KR + 1] = z;
        }
    }
}

// olat [n, NH, KR] = the splits of glm_mla_attn merged: with M the largest of the splits' largest scores, sum_s exp(m_s - M) o_s / sum_s exp(m_s - M) z_s (a split without tokens has m = -inf and z = 0). A block a (head,
// row), 256 threads.
extern "C" __global__ void __launch_bounds__(256) glm_mla_merge(const float* part, float* olat, int NH, int KR, int splits) {
    const int h = blockIdx.x, i = blockIdx.y;
    const float* p = part + ((size_t)i * NH + h) * splits * (KR + 2);
    float M = -INFINITY;
    for (int sp = 0; sp < splits; sp++) M = fmaxf(M, p[(size_t)sp * (KR + 2) + KR]);
    if (M == -INFINITY) M = 0.f;  // (no tokens in any split: zeros)
    float den = 0.f;
    for (int sp = 0; sp < splits; sp++) den += expf(p[(size_t)sp * (KR + 2) + KR] - M) * p[(size_t)sp * (KR + 2) + KR + 1];
    for (int j = threadIdx.x; j < KR; j += blockDim.x) {
        float v = 0.f;
        for (int sp = 0; sp < splits; sp++) v += expf(p[(size_t)sp * (KR + 2) + KR] - M) * p[(size_t)sp * (KR + 2) + j];
        olat[((size_t)i * NH + h) * KR + j] = den > 0.f ? v / den : 0.f;
    }
}

// ov [n, NH vd] bf16 = for each row and head, olat [n, NH, KR] times the head's value rows of kv_b_proj: ov[i][h vd + e] = sum_j olat[i][h][j] wkv[h (nope + vd) + nope + e][j], fp32 sums, one rounding. A block a head and
// GM_VB rows (grid NH x ceil(n / GM_VB)), 256 threads: a warp a value channel e at a time, its lanes down the row of the matrix (KR a multiple of 64). Dynamic shared memory GM_VB KR floats.
extern "C" __global__ void __launch_bounds__(256) glm_expand_v(const float* olat, const bf16* wkv, bf16* ov, int n, int NH, int nope, int vd, int KR) {
    GM_DYN_FLOATS(smv)
    const int h = blockIdx.x, r0 = blockIdx.y * GM_VB, tid = threadIdx.x, lane = tid & 31, w = tid >> 5, nw = blockDim.x >> 5;
    for (int e = tid; e < GM_VB * KR; e += blockDim.x) {
        const int r = e / KR, j = e % KR;
        smv[e] = r0 + r < n ? olat[((size_t)(r0 + r) * NH + h) * KR + j] : 0.f;
    }
    __syncthreads();
    for (int e = w; e < vd; e += nw) {
        const bf16* row = wkv + ((size_t)h * (nope + vd) + nope + e) * KR;
        float a[GM_VB];
#pragma unroll
        for (int r = 0; r < GM_VB; r++) a[r] = 0.f;
        for (int j = 2 * lane; j < KR; j += 64) {
            const float2 c = bf2(*reinterpret_cast<const uint32_t*>(row + j));
#pragma unroll
            for (int r = 0; r < GM_VB; r++) a[r] = fmaf(smv[r * KR + j], c.x, fmaf(smv[r * KR + j + 1], c.y, a[r]));
        }
#pragma unroll
        for (int r = 0; r < GM_VB; r++) {
            const float v = warp_sum(a[r]);
            if (lane == 0 && r0 + r < n) ov[(size_t)(r0 + r) * NH * vd + h * vd + e] = f2b(v);
        }
    }
}
