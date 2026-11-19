// Qwen3.5's own kernels (included by engine.cu in a development build; src/model35.rs runs them): the zero-centered norms, the gated attention at head 256
// with its partial rotary, and the linear-attention layer (the causal convolution, the gated delta rule as a recurrence and in chunks, the gated norm).
// bf16 in memory, fp32 inside; the rounding points are vLLM 0.30's (docs/superpowers/specs/2026-10-06-engine-qwen35-design.md).

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

// ---- gated attention at head 256 -------------------------------------------------------------------------------------------------------
// A full-attention layer's qkv row [n, stride]: each query head's 256 query values then its 256 gate values (nq heads), then nk key heads, then nk value heads (256 each).
// Query and key heads: the norm with (1 + w) over the head, then the rotary on its first 2 half dims (pairs i, i + half; cos and sin bf16 [max_pos, half]), the rest as normed,
// one rounding; query heads to qo [n, nq 256], key heads to the cache kc [positions, nk, 256] at the row's position; value heads copied to vc. Block (row, head), 128 threads.
extern "C" __global__ void q35_prep(const bf16* qkv, int stride, const bf16* qw, const bf16* kw, const bf16* cs, const bf16* sn, int half, const int* pos, bf16* qo, bf16* kc, bf16* vc, int nq, int nk,
                                    float eps) {
    const int t = blockIdx.x, hh = blockIdx.y, i = threadIdx.x;
    const bf16* row = qkv + (size_t)t * stride;
    const int p = pos[t];
    if (hh >= nq + nk) {
        const int j = hh - nq - nk;
        const bf16* x = row + nq * 512 + nk * 256 + j * 256;
        bf16* o = vc + ((size_t)p * nk + j) * 256;
        o[i] = x[i];
        o[i + 128] = x[i + 128];
        return;
    }
    const bool isq = hh < nq;
    const bf16* x = isq ? row + hh * 512 : row + nq * 512 + (hh - nq) * 256;
    const bf16* w = isq ? qw : kw;
    float a = b2f(x[i]), b = b2f(x[i + 128]);
    const float s = rsqrtf(block_sum(a * a + b * b) / 256.f + eps);
    a = a * s * (1.f + b2f(w[i]));
    b = b * s * (1.f + b2f(w[i + 128]));
    __shared__ float nrm[128];
    nrm[i] = a;
    __syncthreads();
    if (i < 2 * half) {
        const int k = i < half ? i : i - half;
        const float c = b2f(cs[(size_t)p * half + k]), sv = b2f(sn[(size_t)p * half + k]);
        a = i < half ? nrm[i] * c - nrm[i + half] * sv : nrm[i] * c + nrm[i - half] * sv;
    }
    bf16* o = isq ? qo + (size_t)t * nq * 256 + hh * 256 : kc + ((size_t)p * nk + (hh - nq)) * 256;
    o[i] = f2b(a);
    o[i + 128] = f2b(b);
}

__device__ __forceinline__ float2 bf2(uint32_t u) { return make_float2(__uint_as_float(u << 16), __uint_as_float(u & 0xffff0000u)); }

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

#define AT_KB 32   // keys a tile
#define AT_KS 129  // a key's row in shared memory: 128 words and one more, so that 32 lanes reading 32 rows meet no bank twice

// Causal attention of the n rows of q [n, qs] (query head h at h * 256; row t's keys are 0 .. pos[t]) over the bf16 cache kc, vc [positions, nk, 256], nq / nk query heads to a
// key-value head; then out = bf16(bf16(attention) * sigmoid(gate)), the gate at gate[t * gs + h * 512 + 256 ..]. fp32 scores and sums, the probabilities rounded to bf16 for the
// values (as FlashAttention). A block takes R rows and the G = nq / nk query heads of one key-value head (a warp each), the keys AT_KB at a time through shared memory: lane j
// scores key j, then each lane sums 8 of the 256 output dims. sl2 = 256^-0.5 * log2(e). Dynamic shared memory: (AT_KB (AT_KS + 128) + 128 R G) * 4 bytes.
// SPLIT (a decode step, R = 1): block z takes the S-th part z of each row's keys (a whole number of tiles) and writes its maximum, sum and unnormalized outputs to
// part [(t nq + h) S + z][258] (outputs, then the maximum in log2 units, then the sum); attn256_merge combines the parts.
template <bool SPLIT>
__device__ __forceinline__ void attn256_t(const bf16* q, int qs, const bf16* gate, int gs, const bf16* kc, const bf16* vc, const int* pos, bf16* out, int os, int n, int nq, int nk, int R,
                                          float sl2, float* part, int S) {
    extern __shared__ uint32_t sm[];
    uint32_t* Ks = sm;
    uint32_t* Vs = Ks + AT_KB * AT_KS;
    uint32_t* Qs = Vs + AT_KB * 128;
    __shared__ int s_k0, s_k1;
    const int G = nq / nk, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int kvh = blockIdx.y, r = w / G, h = kvh * G + w % G, t = blockIdx.x * R + r;
    const bool live = t < n;
    if (live)
        for (int c = lane; c < 128; c += 32) Qs[w * 128 + c] = reinterpret_cast<const uint32_t*>(q + (size_t)t * qs + h * 256)[c];
    if (threadIdx.x == 0) {
        int m = 0;
        for (int i = 0; i < R; i++) {
            const int tt = blockIdx.x * R + i;
            if (tt < n) m = max(m, pos[tt] + 1);
        }
        if (SPLIT) {
            const int span = ((m + S - 1) / S + AT_KB - 1) / AT_KB * AT_KB;
            s_k0 = min(m, blockIdx.z * span);
            s_k1 = min(m, s_k0 + span);
        } else {
            s_k0 = 0;
            s_k1 = m;
        }
    }
    __syncthreads();
    const int k_lo = s_k0, kmax = s_k1, len = live ? min(pos[t] + 1, kmax) : 0;
    float m_run = -INFINITY, l_run = 0.f, acc[8];
#pragma unroll
    for (int e = 0; e < 8; e++) acc[e] = 0.f;
    for (int k0 = k_lo; k0 < kmax; k0 += AT_KB) {
        __syncthreads();  // (the tile before is read)
        for (int c = threadIdx.x; c < AT_KB * 64; c += blockDim.x) {
            const bool isv = c >= AT_KB * 32;
            const int cc = isv ? c - AT_KB * 32 : c, j = cc >> 5, part_ = cc & 31;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (k0 + j < kmax) v = reinterpret_cast<const uint4*>((isv ? vc : kc) + ((size_t)(k0 + j) * nk + kvh) * 256)[part_];
            if (isv) {
                *reinterpret_cast<uint4*>(Vs + j * 128 + part_ * 4) = v;
            } else {
                uint32_t* d = Ks + j * AT_KS + part_ * 4;
                d[0] = v.x;
                d[1] = v.y;
                d[2] = v.z;
                d[3] = v.w;
            }
        }
        __syncthreads();
        if (!live) continue;
        const uint32_t* kr = Ks + lane * AT_KS;
        const uint32_t* qr = Qs + w * 128;
        float s0 = 0.f, s1 = 0.f;
#pragma unroll 8
        for (int c = 0; c < 128; c++) {
            const float2 a = bf2(qr[c]), b = bf2(kr[c]);
            s0 = fmaf(a.x, b.x, s0);
            s1 = fmaf(a.y, b.y, s1);
        }
        const float s = k0 + lane < len ? (s0 + s1) * sl2 : -INFINITY;
        const float m_new = fmaxf(m_run, warp_max(s));  // (finite: a part's first key is in each row's keys)
        const float p = exp2f(s - m_new), corr = exp2f(m_run - m_new);
        l_run = l_run * corr + warp_sum(p);
        const float pb = b2f(f2b(p));
#pragma unroll
        for (int e = 0; e < 8; e++) acc[e] *= corr;
#pragma unroll 4
        for (int j = 0; j < AT_KB; j++) {
            const float pj = __shfl_sync(0xffffffffu, pb, j);
            const uint4 v = *reinterpret_cast<const uint4*>(Vs + j * 128 + lane * 4);
            const float2 v0 = bf2(v.x), v1 = bf2(v.y), v2 = bf2(v.z), v3 = bf2(v.w);
            acc[0] = fmaf(pj, v0.x, acc[0]);
            acc[1] = fmaf(pj, v0.y, acc[1]);
            acc[2] = fmaf(pj, v1.x, acc[2]);
            acc[3] = fmaf(pj, v1.y, acc[3]);
            acc[4] = fmaf(pj, v2.x, acc[4]);
            acc[5] = fmaf(pj, v2.y, acc[5]);
            acc[6] = fmaf(pj, v3.x, acc[6]);
            acc[7] = fmaf(pj, v3.y, acc[7]);
        }
        m_run = m_new;
    }
    if (!live) return;
    if (SPLIT) {
        float* pp = part + ((size_t)(t * nq + h) * S + blockIdx.z) * 258;
#pragma unroll
        for (int e = 0; e < 8; e++) pp[lane * 8 + e] = acc[e];
        if (lane == 0) {
            pp[256] = m_run;
            pp[257] = l_run;
        }
        return;
    }
    const bf16* gr = gate + (size_t)t * gs + h * 512 + 256 + lane * 8;
    bf16* o = out + (size_t)t * os + h * 256 + lane * 8;
#pragma unroll
    for (int e = 0; e < 8; e++) {
        const float a = b2f(f2b(acc[e] / l_run)), g = b2f(gr[e]);
        o[e] = f2b(a / (1.f + expf(-g)));
    }
}

extern "C" __global__ void __launch_bounds__(512) attn256(const bf16* q, int qs, const bf16* gate, int gs, const bf16* kc, const bf16* vc, const int* pos, bf16* out, int os, int n, int nq, int nk,
                                                         int R, float sl2) {
    attn256_t<false>(q, qs, gate, gs, kc, vc, pos, out, os, n, nq, nk, R, sl2, nullptr, 1);
}

// A decode step's attention in S parts of the keys (grid n, nk, S; R = 1): see attn256_t.
extern "C" __global__ void __launch_bounds__(512) attn256_parts(const bf16* q, int qs, const bf16* kc, const bf16* vc, const int* pos, int n, int nq, int nk, float sl2, float* part, int S) {
    attn256_t<true>(q, qs, nullptr, 0, kc, vc, pos, nullptr, 0, n, nq, nk, 1, sl2, part, S);
}

// The S parts of attn256_split combined: out = bf16(bf16(sum of the parts' outputs, each times 2^(its maximum - the largest) / the sum likewise) * sigmoid(gate)). Block (row, query
// head), 256 threads (a dim each).
extern "C" __global__ void attn256_merge(const float* part, int S, const bf16* gate, int gs, bf16* out, int os, int nq) {
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const float* pp = part + (size_t)(t * nq + h) * S * 258;
    float mx = -INFINITY;
    for (int z = 0; z < S; z++) mx = fmaxf(mx, pp[z * 258 + 256]);
    float l = 0.f, a = 0.f;
    for (int z = 0; z < S; z++) {
        const float m = pp[z * 258 + 256];
        if (m == -INFINITY) continue;  // (a part with no keys)
        const float wz = exp2f(m - mx);
        l = fmaf(pp[z * 258 + 257], wz, l);
        a = fmaf(pp[z * 258 + d], wz, a);
    }
    const float o = b2f(f2b(a / l)), g = b2f(gate[(size_t)t * gs + h * 512 + 256 + d]);
    out[(size_t)t * os + h * 256 + d] = f2b(o / (1.f + expf(-g)));
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

__device__ __forceinline__ float softplus_f(float x) { return x > 20.f ? x : log1pf(expf(x)); }

// The gated delta rule, one token after another, for the n rows of a sequence: xc [n, C] the convolution's output (queries: nk heads of 128, keys: nk heads of 128, values: nv heads
// of 128; value head h reads key head h / (nv / nk)), ba [n, sba] (b: nv values, then a: nv values), A_log fp32 [nv], dt_bias bf16 [nv]; state fp32 [nv, 128, 128] (S[i][j]: i over
// the key's dims, j over the value's), read and written back; core [n, nv 128] the outputs (bf16, before the gated norm). Per token: q and k l2-normed (q times 128^-0.5),
// beta = sigmoid(b), g = -exp(A_log) softplus(a + dt_bias); S = exp(g) S; S += k (beta (v - S^T k))^T; o = S^T q. A block a value head, a thread a column of S (in registers).
extern "C" __global__ void __launch_bounds__(128, 1) gdn_recurrent(const bf16* xc, int C, const bf16* ba, int sba, const float* A_log, const bf16* dt_bias, float* state, bf16* core, int n, int nk,
                                                                   int nv) {
    const int h = blockIdx.x, j = threadIdx.x, lane = j & 31, w = j >> 5;
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
