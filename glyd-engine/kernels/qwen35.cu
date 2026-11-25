// Qwen3.5's own kernels (included by engine.cu in a development build; the description executor runs them through src/kern35.rs's table): the zero-centered norms, the gated attention at head 256
// with its partial rotary, and the linear-attention layer (the causal convolution, the gated delta rule as a recurrence and in chunks, the gated norm).
// bf16 in memory, fp32 inside; the rounding points are vLLM 0.30's (glyd-engine/docs/design/specs/2026-10-06-engine-qwen35-design.md).

// ---- norms -----------------------------------------------------------------------------------------------------------------------------
// out [M, K] = rmsnorm(x) * (1 + w): the norms of Qwen3.5 store their weight centered on zero (vLLM's GemmaRMSNorm). One block a row, as rmsnorm.
extern "C" __global__ void rmsnorm1p(const bf16* x, const bf16* w, bf16* out, int K, float eps) {
    const bf16* xr = x + (size_t)blockIdx.x * K;
    bf16* o = out + (size_t)blockIdx.x * K;
    float v[RN_PER_THREAD];
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < RN_PER_THREAD; j++) {
        int i = threadIdx.x + j * blockDim.x;
        v[j] = i < K ? b2f(xr[i]) : 0.f;
        acc += v[j] * v[j];
    }
    acc = block_sum(acc);
    float s = rsqrtf(acc / K + eps);
#pragma unroll
    for (int j = 0; j < RN_PER_THREAD; j++) {
        int i = threadIdx.x + j * blockDim.x;
        if (i < K) o[i] = f2b(v[j] * s * (1.f + b2f(w[i])));
    }
}

// x += y, then out = rmsnorm(sum) * (1 + w), the sum's rounding as rmsnorm_add's.
extern "C" __global__ void rmsnorm1p_add(bf16* x, const bf16* y, const bf16* w, bf16* out, int K, float eps) {
    bf16* xr = x + (size_t)blockIdx.x * K;
    const bf16* yr = y + (size_t)blockIdx.x * K;
    bf16* o = out + (size_t)blockIdx.x * K;
    float z[RN_PER_THREAD];
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < RN_PER_THREAD; j++) {
        int i = threadIdx.x + j * blockDim.x;
        z[j] = i < K ? b2f(xr[i]) + b2f(yr[i]) : 0.f;
        acc += z[j] * z[j];
    }
    acc = block_sum(acc);
    float s = rsqrtf(acc / K + eps);
#pragma unroll
    for (int j = 0; j < RN_PER_THREAD; j++) {
        int i = threadIdx.x + j * blockDim.x;
        if (i < K) {
            xr[i] = f2b(z[j]);
            o[i] = f2b(z[j] * s * (1.f + b2f(w[i])));
        }
    }
}

// ---- helpers the other families' kernels share ---------------------------------------------------------------------------------------------
__device__ __forceinline__ float2 bf2(uint32_t u) { return make_float2(__uint_as_float(u << 16), __uint_as_float(u & 0xffff0000u)); }

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// ---- the linear-attention layer (Gated DeltaNet) ---------------------------------------------------------------------------------------
// The causal depthwise convolution of kernel 4 over the first C channels of the rows x [n, xs], then SiLU (fp32, one rounding, as vLLM's): w [C, 4] (the oldest tap first),
// st [3, C] the channel's last three inputs (the oldest first), read and left holding the last three of these rows; y [n, C]. A thread a channel.
__device__ __forceinline__ float wval(bf16 v) { return b2f(v); }
__device__ __forceinline__ float wval(float v) { return v; }

template <typename W>
__device__ __forceinline__ void conv_silu_of(const bf16* x, int xs, const W* w, bf16* st, bf16* y, int n, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const float w0 = wval(w[c * 4]), w1 = wval(w[c * 4 + 1]), w2 = wval(w[c * 4 + 2]), w3 = wval(w[c * 4 + 3]);
    float s0 = b2f(st[c]), s1 = b2f(st[C + c]), s2 = b2f(st[2 * C + c]);
    for (int t = 0; t < n; t++) {
        const float xv = b2f(x[(size_t)t * xs + c]);
        // (the reference's order: the taps oldest first, no fused multiply-add)
        float acc = __fmul_rn(w0, s0);
        acc = __fadd_rn(acc, __fmul_rn(w1, s1));
        acc = __fadd_rn(acc, __fmul_rn(w2, s2));
        acc = __fadd_rn(acc, __fmul_rn(w3, xv));
        y[(size_t)t * C + c] = f2b(acc / (1.f + expf(-acc)));
        s0 = s1;
        s1 = s2;
        s2 = xv;
    }
    st[c] = f2b(s0);
    st[C + c] = f2b(s1);
    st[2 * C + c] = f2b(s2);
}

extern "C" __global__ void conv_silu(const bf16* x, int xs, const bf16* w, bf16* st, bf16* y, int n, int C) { conv_silu_of(x, xs, w, st, y, n, C); }

// conv_silu with fp32 weights (Kimi-K3 releases them so; each tap as it is).
extern "C" __global__ void conv_silu_w32(const bf16* x, int xs, const float* w, bf16* st, bf16* y, int n, int C) { conv_silu_of(x, xs, w, st, y, n, C); }

// conv_silu of a step's rows that decode, a token of a sequence each (grid C / 256 by rows): row r's state at sts[r ss] (a table of addresses, as vLLM's conv_state_indices
// pick a request's slot; 0: a padding row, whose output is zeroed and which reads and writes no state), its input row r of x, its output row r of y. Each row's arithmetic
// is conv_silu's on that sequence's one token.
extern "C" __global__ void conv_silu_rows(const bf16* x, int xs, const bf16* w, const unsigned long long* sts, int ss, bf16* y, int C) {
    const int r = blockIdx.y;
    bf16* st = reinterpret_cast<bf16*>(sts[(size_t)r * ss]);
    if (!st) {
        const int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c < C) y[(size_t)r * C + c] = f2b(0.f);
        return;
    }
    conv_silu_of(x + (size_t)r * xs, xs, w, st, y + (size_t)r * C, 1, C);
}

__device__ __forceinline__ float softplus_f(float x) { return x > 20.f ? x : log1pf(expf(x)); }

// The gated delta rule, one token after another, for the n rows of a sequence: xc [n, C] the convolution's output (queries: nk heads of 128, keys: nk heads of 128, values: nv heads
// of 128; value head h reads key head h / (nv / nk)), ba [n, sba] (b: nv values, then a: nv values), A_log fp32 [nv], dt_bias bf16 [nv]; state fp32 [nv, 128, 128] (S[i][j]: i over
// the key's dims, j over the value's), read and written back; core [n, nv 128] the outputs (bf16, before the gated norm). Per token: q and k l2-normed (q times 128^-0.5),
// beta = sigmoid(b), g = -exp(A_log) softplus(a + dt_bias); S = exp(g) S; S += k (beta (v - S^T k))^T; o = S^T q. A block a value head, a thread a column of S (in registers).
__device__ __forceinline__ void gdn_recurrent_of(int h, const bf16* xc, int C, const bf16* ba, int sba, const float* A_log, const bf16* dt_bias, float* state, bf16* core, int n,
                                                 int nk, int nv) {
    const int j = threadIdx.x, lane = j & 31, w = j >> 5;
    const int kh = h / (nv / nk), kd = nk * 128, vd = nv * 128;
    __shared__ __align__(16) float qk[2][128];
    __shared__ float red[2][4];
    float* st = state + (size_t)h * 128 * 128;
    float S[128];
#pragma unroll
    for (int i = 0; i < 128; i++) S[i] = st[i * 128 + j];
    const float Ah = -expf(A_log[h]), dtb = b2f(dt_bias[h]), scale = rsqrtf(128.f);
    for (int t = 0; t < n; t++) {
        const bf16* x = xc + (size_t)t * C;
        const float qv = b2f(x[kh * 128 + j]), kv = b2f(x[kd + kh * 128 + j]), vv = b2f(x[2 * kd + h * 128 + j]);
        float sq = warp_sum(qv * qv), sk = warp_sum(kv * kv);
        if (lane == 0) {
            red[0][w] = sq;
            red[1][w] = sk;
        }
        __syncthreads();
        sq = (red[0][0] + red[0][1]) + (red[0][2] + red[0][3]);
        sk = (red[1][0] + red[1][1]) + (red[1][2] + red[1][3]);
        qk[0][j] = qv * rsqrtf(sq + 1e-6f) * scale;
        qk[1][j] = kv * rsqrtf(sk + 1e-6f);
        const float bb = b2f(ba[(size_t)t * sba + h]), aa = b2f(ba[(size_t)t * sba + nv + h]);
        const float beta = 1.f / (1.f + expf(-bb)), decay = expf(Ah * softplus_f(aa + dtb));
        __syncthreads();
        const float4* q4 = reinterpret_cast<const float4*>(qk[0]);
        const float4* k4 = reinterpret_cast<const float4*>(qk[1]);
        float m0 = 0.f, m1 = 0.f, m2 = 0.f, m3 = 0.f;
#pragma unroll
        for (int i = 0; i < 32; i++) {
            const float4 kk = k4[i];
            S[4 * i] *= decay;
            S[4 * i + 1] *= decay;
            S[4 * i + 2] *= decay;
            S[4 * i + 3] *= decay;
            m0 = fmaf(S[4 * i], kk.x, m0);
            m1 = fmaf(S[4 * i + 1], kk.y, m1);
            m2 = fmaf(S[4 * i + 2], kk.z, m2);
            m3 = fmaf(S[4 * i + 3], kk.w, m3);
        }
        const float d = (vv - ((m0 + m1) + (m2 + m3))) * beta;
        float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
#pragma unroll
        for (int i = 0; i < 32; i++) {
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
        core[(size_t)t * vd + h * 128 + j] = f2b((o0 + o1) + (o2 + o3));
    }
#pragma unroll
    for (int i = 0; i < 128; i++) st[i * 128 + j] = S[i];
}

extern "C" __global__ void __launch_bounds__(128, 1) gdn_recurrent(const bf16* xc, int C, const bf16* ba, int sba, const float* A_log, const bf16* dt_bias, float* state, bf16* core, int n, int nk,
                                                                   int nv) {
    gdn_recurrent_of(blockIdx.x, xc, C, ba, sba, A_log, dt_bias, state, core, n, nk, nv);
}

// gdn_recurrent of a step's rows that decode, a token of a sequence each (grid nv by rows): row r's state at sts[r ss] (0: a padding row, whose output is zeroed and which
// reads and writes no state), its rows of xc, ba and core at r C, r sba and r nv 128. Each row's arithmetic is gdn_recurrent's on that sequence's one token.
extern "C" __global__ void __launch_bounds__(128, 1) gdn_recurrent_rows(const bf16* xc, int C, const bf16* ba, int sba, const float* A_log, const bf16* dt_bias,
                                                                        const unsigned long long* sts, int ss, bf16* core, int nk, int nv) {
    const int r = blockIdx.y;
    float* st = reinterpret_cast<float*>(sts[(size_t)r * ss]);
    bf16* out = core + (size_t)r * nv * 128;
    if (!st) {
        out[blockIdx.x * 128 + threadIdx.x] = f2b(0.f);
        return;
    }
    gdn_recurrent_of(blockIdx.x, xc + (size_t)r * C, C, ba + (size_t)r * sba, sba, A_log, dt_bias, st, out, 1, nk, nv);
}

#define GC 64   // tokens a chunk
#define GBV 64  // value columns a block
#define GKS 65  // a query's or key's row in shared memory: 128 bf16 and one more word

// The same rule over chunks of GC tokens (the WY form: Yang, Kautz and Hatamizadeh, "Gated Delta Networks", ICLR 2025), for a prompt: within a chunk, with G the decays' cumulative
// sums and S0 the state at its start, U = (I + L)^-1 diag(beta) (V - diag(exp G) K S0) where L[t][s] = beta_t exp(G_t - G_s) k_t . k_s (s < t), the outputs
// O = diag(exp G) Q S0 + (exp(G_t - G_s) q_t . k_s, s <= t) U, and the state exp(G_last) S0 + K^T diag(exp(G_last - G)) U. The queries and keys are l2-normed and rounded to bf16
// (as vLLM's prompt path keeps them), the rest fp32. Arguments as gdn_recurrent's. A block a value head and GBV of its value columns (grid nv x 128 / GBV), 256 threads.
// Dynamic shared memory: (2 GC GKS + GC GC + GC GBV + 128 GBV + 2 GC) * 4 bytes (99,328).
extern "C" __global__ void __launch_bounds__(256, 1) gdn_chunk(const bf16* xc, int C, const bf16* ba, int sba, const float* A_log, const bf16* dt_bias, float* state, bf16* core, int n, int nk,
                                                               int nv) {
    extern __shared__ float smf[];
    uint32_t* Qs = reinterpret_cast<uint32_t*>(smf);
    uint32_t* Ks = Qs + GC * GKS;
    float* LP = reinterpret_cast<float*>(Ks + GC * GKS);
    float* U = LP + GC * GC;
    float* S = U + GC * GBV;
    float* Gc = S + 128 * GBV;
    float* Bt = Gc + GC;
    bf16* Qh = reinterpret_cast<bf16*>(Qs);
    bf16* Kh = reinterpret_cast<bf16*>(Ks);
    const int h = blockIdx.x, cb = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int kh = h / (nv / nk), kd = nk * 128, vd = nv * 128;
    const int ty = tid >> 4, tx = tid & 15;
    const float scale = rsqrtf(128.f);
    float* st = state + (size_t)h * 128 * 128 + cb * GBV;
    for (int e = tid; e < 128 * GBV; e += 256) S[e] = st[(e / GBV) * 128 + e % GBV];
    const float Ah = -expf(A_log[h]), dtb = b2f(dt_bias[h]);
    for (int t0 = 0; t0 < n; t0 += GC) {
        const int tc = min(GC, n - t0);
        __syncthreads();  // (the chunk before is done with every buffer)
        // 1. the chunk's queries and keys normed, its decays and betas, its values into U
        for (int t = warp; t < GC; t += 8) {
            const bf16* x = xc + (size_t)(t0 + t) * C;
            float qv[4], kv[4], sq = 0.f, sk = 0.f;
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int d = lane + 32 * e;
                qv[e] = t < tc ? b2f(x[kh * 128 + d]) : 0.f;
                kv[e] = t < tc ? b2f(x[kd + kh * 128 + d]) : 0.f;
                sq += qv[e] * qv[e];
                sk += kv[e] * kv[e];
            }
            const float rq = rsqrtf(warp_sum(sq) + 1e-6f), rk = rsqrtf(warp_sum(sk) + 1e-6f);
#pragma unroll
            for (int e = 0; e < 4; e++) {
                Qh[t * 2 * GKS + lane + 32 * e] = f2b(qv[e] * rq);
                Kh[t * 2 * GKS + lane + 32 * e] = f2b(kv[e] * rk);
            }
            if (lane == 0) {
                float g = 0.f, beta = 0.f;
                if (t < tc) {
                    g = Ah * softplus_f(b2f(ba[(size_t)(t0 + t) * sba + nv + h]) + dtb);
                    beta = 1.f / (1.f + expf(-b2f(ba[(size_t)(t0 + t) * sba + h])));
                }
                Gc[t] = g;
                Bt[t] = beta;
            }
        }
        for (int e = tid; e < GC * GBV; e += 256) {
            const int t = e / GBV;
            U[e] = t < tc ? b2f(xc[(size_t)(t0 + t) * C + 2 * kd + h * 128 + cb * GBV + e % GBV]) : 0.f;
        }
        __syncthreads();
        if (tid == 0) {
            float a = 0.f;
            for (int t = 0; t < GC; t++) {
                a += Gc[t];
                Gc[t] = a;
            }
        }
        __syncthreads();
        // 2. L (rows ty + 16 r, columns tx + 16 c of each thread)
        {
            float a4[4][4] = {};
            for (int c = 0; c < 64; c++) {
                float2 a[4], b[4];
#pragma unroll
                for (int r = 0; r < 4; r++) {
                    a[r] = bf2(Ks[(ty + 16 * r) * GKS + c]);
                    b[r] = bf2(Ks[(tx + 16 * r) * GKS + c]);
                }
#pragma unroll
                for (int r = 0; r < 4; r++)
#pragma unroll
                    for (int cc = 0; cc < 4; cc++) a4[r][cc] = fmaf(a[r].y, b[cc].y, fmaf(a[r].x, b[cc].x, a4[r][cc]));
            }
#pragma unroll
            for (int r = 0; r < 4; r++)
#pragma unroll
                for (int cc = 0; cc < 4; cc++) {
                    const int t = ty + 16 * r, s = tx + 16 * cc;
                    LP[t * GC + s] = s < t ? Bt[t] * expf(Gc[t] - Gc[s]) * a4[r][cc] : 0.f;
                }
        }
        // 3. Y = diag(beta) (V - diag(exp G) K S0) into U (each thread its own entries of U)
        {
            float a4[4][4] = {};
            for (int c = 0; c < 64; c++) {
                float2 a[4];
#pragma unroll
                for (int r = 0; r < 4; r++) a[r] = bf2(Ks[(ty + 16 * r) * GKS + c]);
#pragma unroll
                for (int cc = 0; cc < 4; cc++) {
                    const float s0 = S[(2 * c) * GBV + tx + 16 * cc], s1 = S[(2 * c + 1) * GBV + tx + 16 * cc];
#pragma unroll
                    for (int r = 0; r < 4; r++) a4[r][cc] = fmaf(a[r].y, s1, fmaf(a[r].x, s0, a4[r][cc]));
                }
            }
#pragma unroll
            for (int r = 0; r < 4; r++)
#pragma unroll
                for (int cc = 0; cc < 4; cc++) {
                    const int t = ty + 16 * r, j = tx + 16 * cc;
                    U[t * GBV + j] = Bt[t] * (U[t * GBV + j] - expf(Gc[t]) * a4[r][cc]);
                }
        }
        __syncthreads();
        // 4. U_t = Y_t - sum over s < t of L[t][s] U_s, a row at a time (two warps, a column a lane)
        if (warp < 2) {
            const int j = warp * 32 + lane;
            for (int t = 1; t < GC; t++) {
                float a = U[t * GBV + j];
                for (int s = 0; s < t; s++) a = fmaf(-LP[t * GC + s], U[s * GBV + j], a);
                U[t * GBV + j] = a;
            }
        }
        __syncthreads();
        // 5. P = (exp(G_t - G_s) q_t . k_s, s <= t) into LP; then O
        {
            float a4[4][4] = {};
            for (int c = 0; c < 64; c++) {
                float2 a[4], b[4];
#pragma unroll
                for (int r = 0; r < 4; r++) {
                    a[r] = bf2(Qs[(ty + 16 * r) * GKS + c]);
                    b[r] = bf2(Ks[(tx + 16 * r) * GKS + c]);
                }
#pragma unroll
                for (int r = 0; r < 4; r++)
#pragma unroll
                    for (int cc = 0; cc < 4; cc++) a4[r][cc] = fmaf(a[r].y, b[cc].y, fmaf(a[r].x, b[cc].x, a4[r][cc]));
            }
#pragma unroll
            for (int r = 0; r < 4; r++)
#pragma unroll
                for (int cc = 0; cc < 4; cc++) {
                    const int t = ty + 16 * r, s = tx + 16 * cc;
                    LP[t * GC + s] = s <= t ? expf(Gc[t] - Gc[s]) * a4[r][cc] : 0.f;
                }
        }
        __syncthreads();
        {
            float a4[4][4] = {};
            for (int c = 0; c < 64; c++) {
                float2 a[4];
#pragma unroll
                for (int r = 0; r < 4; r++) a[r] = bf2(Qs[(ty + 16 * r) * GKS + c]);
#pragma unroll
                for (int cc = 0; cc < 4; cc++) {
                    const float s0 = S[(2 * c) * GBV + tx + 16 * cc], s1 = S[(2 * c + 1) * GBV + tx + 16 * cc];
#pragma unroll
                    for (int r = 0; r < 4; r++) a4[r][cc] = fmaf(a[r].y, s1, fmaf(a[r].x, s0, a4[r][cc]));
                }
            }
#pragma unroll
            for (int r = 0; r < 4; r++) {
                const float e = expf(Gc[ty + 16 * r]);
#pragma unroll
                for (int cc = 0; cc < 4; cc++) a4[r][cc] *= e;
            }
            for (int s = 0; s < GC; s++) {
                float p[4], u[4];
#pragma unroll
                for (int r = 0; r < 4; r++) p[r] = LP[(ty + 16 * r) * GC + s];
#pragma unroll
                for (int cc = 0; cc < 4; cc++) u[cc] = U[s * GBV + tx + 16 * cc];
#pragma unroll
                for (int r = 0; r < 4; r++)
#pragma unroll
                    for (int cc = 0; cc < 4; cc++) a4[r][cc] = fmaf(p[r], u[cc], a4[r][cc]);
            }
#pragma unroll
            for (int r = 0; r < 4; r++) {
                const int t = ty + 16 * r;
                if (t < tc)
#pragma unroll
                    for (int cc = 0; cc < 4; cc++) core[(size_t)(t0 + t) * vd + h * 128 + cb * GBV + tx + 16 * cc] = f2b(a4[r][cc] * scale);
            }
        }
        __syncthreads();
        // 6. S = exp(G_last) S0 + K^T diag(exp(G_last - G)) U
        const float gl = Gc[GC - 1];
        for (int e = tid; e < GC * GBV; e += 256) U[e] *= expf(gl - Gc[e / GBV]);
        __syncthreads();
        {
            const float el = expf(gl);
            float a8[8][4];
#pragma unroll
            for (int r = 0; r < 8; r++)
#pragma unroll
                for (int cc = 0; cc < 4; cc++) a8[r][cc] = S[(ty + 16 * r) * GBV + tx + 16 * cc] * el;
            for (int t = 0; t < GC; t++) {
                float k8[8], u[4];
#pragma unroll
                for (int r = 0; r < 8; r++) k8[r] = b2f(Kh[t * 2 * GKS + ty + 16 * r]);
#pragma unroll
                for (int cc = 0; cc < 4; cc++) u[cc] = U[t * GBV + tx + 16 * cc];
#pragma unroll
                for (int r = 0; r < 8; r++)
#pragma unroll
                    for (int cc = 0; cc < 4; cc++) a8[r][cc] = fmaf(k8[r], u[cc], a8[r][cc]);
            }
#pragma unroll
            for (int r = 0; r < 8; r++)
#pragma unroll
                for (int cc = 0; cc < 4; cc++) S[(ty + 16 * r) * GBV + tx + 16 * cc] = a8[r][cc];
        }
    }
    __syncthreads();
    for (int e = tid; e < 128 * GBV; e += 256) st[(e / GBV) * 128 + e % GBV] = S[e];
}

// The gated norm, in place: core [n, nv 128] (a value head's 128 outputs) = rmsnorm(o) * w * silu(z), w [128] (not centered on zero; bf16 values held as fp32: the loader rounds the
// checkpoint's fp32 as vLLM loads it), z at z[t * zs + h * 128 ..]; fp32, one rounding (vLLM's RMSNormGated, the norm before the gate). Block (row, value head), 128 threads.
extern "C" __global__ void gated_norm(bf16* core, const bf16* z, int zs, const float* w, int nv, float eps) {
    const int t = blockIdx.x, h = blockIdx.y, j = threadIdx.x;
    bf16* o = core + ((size_t)t * nv + h) * 128;
    const float x = b2f(o[j]);
    const float r = rsqrtf(block_sum(x * x) / 128.f + eps);
    const float zz = b2f(z[(size_t)t * zs + h * 128 + j]);
    o[j] = f2b(x * r * w[j] * (zz / (1.f + expf(-zz))));
}

// y [m, N] = x [m, K] W^T for a small bf16 W [N, K] (a matrix the library cannot pack: its rows not a multiple of 64): fp32 sums, one rounding. Block a row of x, a warp an output.
// Row blockIdx.x of x times W's rows: a warp a row of W, the rows spread over gridDim.y blocks (a decode step's one row of x would
// otherwise leave the product to one block); 16-byte loads where K and the rows allow them.
extern "C" __global__ void small_linear(const bf16* x, int xs, const bf16* W, bf16* y, int ys, int N, int K) {
    const bf16* xr = x + (size_t)blockIdx.x * xs;
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    const bool wide = (K & 7) == 0 && ((reinterpret_cast<uintptr_t>(xr) | reinterpret_cast<uintptr_t>(W)) & 15) == 0;
    for (int o = blockIdx.y * nw + w; o < N; o += nw * gridDim.y) {
        const bf16* wr = W + (size_t)o * K;
        float s = 0.f;
        if (wide) {
            const uint4* xv = reinterpret_cast<const uint4*>(xr);
            const uint4* wv = reinterpret_cast<const uint4*>(wr);
            for (int k = lane; k < (K >> 3); k += 32) {
                const uint4 a = xv[k], b = wv[k];
                const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a);
                const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&b);
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const float2 fa = __bfloat1622float2(a2[j]), fb = __bfloat1622float2(b2[j]);
                    s = fmaf(fa.x, fb.x, s);
                    s = fmaf(fa.y, fb.y, s);
                }
            }
        } else {
            for (int k = lane; k < K; k += 32) s = fmaf(b2f(xr[k]), b2f(wr[k]), s);
        }
        s = warp_sum(s);
        if (lane == 0) y[(size_t)blockIdx.x * ys + o] = f2b(s);
    }
}
