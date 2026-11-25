// GLM-5.3-Flash's own kernels (included by engine.cu after qwen35.cu and moe.cu; the executor runs them through src/kern_glm.rs's `KernGlm`): Kimi delta attention (the delta rule with a decay for each key channel, one token after another and in chunks, and the
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

// (the latent attention over the keys an indexer picks: kernels/mla.cu, the latent attention layer kind's)
