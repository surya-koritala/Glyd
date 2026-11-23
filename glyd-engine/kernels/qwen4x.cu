// Qwen3.8-Flash-Next's own kernels (included by engine.cu after qwen35.cu and glm.cu; src/model_qwen4x.rs runs them): the Gated Residual of the four streams (the per-stream norm, the mixing weights' activation,
// the mix into a block's input, the update of the streams from its output), the GDN's gated norm with a sigmoid gate, the n-gram embedding's gate and dilated convolution (PLE), the QSA indexer's query and block keys,
// and the attention over the tokens the indexer picked. bf16 in memory, fp32 inside; the rounding points are transformers 5.19.0's (modeling_qwen4_exp.py), which rounds every elementwise op of a bf16 tensor.
// Reference: tools/qwen4x_ref.py (transformers' own modules).

#define QX_HC 4  // residual streams

__device__ __forceinline__ float qx_r(float x) { return b2f(f2b(x)); }  // a bf16 tensor op's rounding
__device__ __forceinline__ float qx_sig(float x) { return 1.f / (1.f + expf(-x)); }

// The sum over the 256 threads of one quarter of a 1024-thread block (warps 8 q .. 8 q + 7), the same on those threads. sh: 32 floats of shared memory.
__device__ __forceinline__ float qx_quarter_sum(float v, float* sh) {
    v = warp_sum(v);
    const int w = threadIdx.x >> 5, q = w >> 3;
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[w] = v;
    __syncthreads();
    float t = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) t += sh[q * 8 + i];
    return t;
}

// ---- the Gated Residual ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
// xn [n, 4 D] = each stream of x [n, 4, D] normed (rms over its D values; `whole`: over the row's 4 D, as the MTP head's pre_fc_norm_hidden) times (1 + w [4 D]), one rounding; a block of 1024 threads a row, a quarter
// of them a stream (D at most 256 * 16).
extern "C" __global__ void __launch_bounds__(1024) qx_hc_norm(const bf16* x, const bf16* w, bf16* xn, int D, float eps, int whole) {
    __shared__ float sh[32];
    __shared__ float tot[QX_HC];
    const int q = threadIdx.x >> 8, j = threadIdx.x & 255;
    const size_t row = (size_t)blockIdx.x * QX_HC * D + (size_t)q * D;
    float v[16], acc = 0.f;
#pragma unroll
    for (int k = 0; k < 16; k++) {
        const int c = j + k * 256;
        v[k] = c < D ? b2f(x[row + c]) : 0.f;
        acc += v[k] * v[k];
    }
    float sum = qx_quarter_sum(acc, sh);
    if (whole) {
        if (j == 0) tot[q] = sum;
        __syncthreads();
        sum = (tot[0] + tot[1]) + (tot[2] + tot[3]);
    }
    const float s = rsqrtf(sum / (whole ? QX_HC * D : D) + eps);
#pragma unroll
    for (int k = 0; k < 16; k++) {
        const int c = j + k * 256;
        if (c < D) xn[row + c] = f2b(v[k] * s * (1.f + b2f(w[q * D + c])));
    }
}

// y = silu(y / 4) in place (the mixing weights' bottleneck; y bf16 [count], a division by 4 that bf16 holds exactly, then the activation rounded).
extern "C" __global__ void qx_silu4(bf16* y, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float v = b2f(y[i]) * 0.25f;
    y[i] = f2b(v * qx_sig(v));
}

// The mix of a row's streams into the block's input: h [n, D] = mean over the streams s of bf16(bf16(sigmoid(gw[s])) * xn[s]) (gw [n, 4 D] the up product, xn [n, 4 D]), the sum in fp32, one rounding;
// where injl (bf16 [n, 4], the inject product) is given, inj [n, 4] = 2 bf16(sigmoid(bf16(injl / 4))) as fp32. Block (row, 256 columns).
extern "C" __global__ void qx_hc_mix(const bf16* gw, const bf16* xn, const bf16* injl, bf16* h, float* inj, int D) {
    const int t = blockIdx.x, c = blockIdx.y * blockDim.x + threadIdx.x;
    if (injl && blockIdx.y == 0 && threadIdx.x < QX_HC) inj[t * QX_HC + threadIdx.x] = 2.f * qx_r(qx_sig(b2f(injl[t * QX_HC + threadIdx.x]) * 0.25f));
    if (c >= D) return;
    const size_t r = (size_t)t * QX_HC * D + c;
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < QX_HC; k++) s += qx_r(qx_r(qx_sig(b2f(gw[r + k * D]))) * b2f(xn[r + k * D]));
    h[(size_t)t * D + c] = f2b(s * 0.25f);
}

// The streams after a block: x[s] = bf16(x[s] + bf16(o * inj[s])) for o [n, D] the block's output, inj [n, 4] fp32. Block (row, 256 columns).
extern "C" __global__ void qx_hc_post(bf16* x, const bf16* o, const float* inj, int D) {
    const int t = blockIdx.x, c = blockIdx.y * blockDim.x + threadIdx.x;
    if (c >= D) return;
    const float ov = b2f(o[(size_t)t * D + c]);
#pragma unroll
    for (int k = 0; k < QX_HC; k++) {
        const size_t i = ((size_t)t * QX_HC + k) * D + c;
        x[i] = f2b(b2f(x[i]) + qx_r(ov * inj[t * QX_HC + k]));
    }
}

// ---- the GDN's gated norm ----------------------------------------------------------------------------------------------------------------------------------------------------------------
// In place: core [n, nv 128] = bf16(bf16(w * bf16(rmsnorm(o))) * sigmoid(z)) (transformers' RMSNormGated with the sigmoid gate: the normed values and their product with the bf16 weight w rounded, then the gate in
// fp32), z at z[t * zs + h * 128 ..]. Block (row, value head), 128 threads.
extern "C" __global__ void qx_gnorm_sig(bf16* core, const bf16* z, int zs, const bf16* w, int nv, float eps) {
    const int t = blockIdx.x, h = blockIdx.y, j = threadIdx.x;
    bf16* o = core + ((size_t)t * nv + h) * 128;
    const float x = b2f(o[j]);
    const float r = rsqrtf(block_sum(x * x) / 128.f + eps);
    const float zz = b2f(z[(size_t)t * zs + h * 128 + j]);
    o[j] = f2b(qx_r(b2f(w[j]) * qx_r(x * r)) * qx_sig(zz));
}

// The probability of a row's top token: p[0] = 1 / sum over the V logits l (bf16; -inf for the ones left out) of exp(l - vmax[0]) (vmax: argmax_bf16's value). One block of 1024 threads (V a multiple of 8).
extern "C" __global__ void __launch_bounds__(1024) qx_top_prob(const bf16* lg, int V, const float* vmax, float* p) {
    const float m = vmax[0];
    float sum = 0.f;
    for (int c = threadIdx.x; c * 8 < V; c += blockDim.x) {
        const uint4 q = *reinterpret_cast<const uint4*>(lg + (size_t)c * 8);
        const bf16* v = reinterpret_cast<const bf16*>(&q);
#pragma unroll
        for (int k = 0; k < 8; k++) sum += __expf(b2f(v[k]) - m);
    }
    sum = block_sum(sum);
    if (threadIdx.x == 0) p[0] = 1.f / sum;
}

// ---- the GDN mixer of a decode step in fewer launches ------------------------------------------------------------------------------------------------------------------------------------------
// conv_silu and small_linear in one launch: blocks 0 .. cb - 1 the causal convolution and SiLU of the first C channels of x [n, xs] (conv_silu's: w [C, 4], st [3, C], y [n, C]); the next n x nby blocks
// yb [n, N] (rows ybs apart) = xb [n, K] (rows xbs apart) Wb^T (small_linear's: a warp an output, 8 a block; Wb bf16 [N, K]). The same sums as the two kernels'. 256 threads.
extern "C" __global__ void __launch_bounds__(256) qx_gdn_conv_ba(const bf16* x, int xs, const bf16* w, bf16* st, bf16* y, int n, int C, int cb, const bf16* xb, int xbs, const bf16* Wb, bf16* yb, int ybs, int N, int K, int nby) {
    if ((int)blockIdx.x < cb) {
        const int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c >= C) return;
        const float w0 = b2f(w[c * 4]), w1 = b2f(w[c * 4 + 1]), w2 = b2f(w[c * 4 + 2]), w3 = b2f(w[c * 4 + 3]);
        float s0 = b2f(st[c]), s1 = b2f(st[C + c]), s2 = b2f(st[2 * C + c]);
        for (int t = 0; t < n; t++) {
            const float xv = b2f(x[(size_t)t * xs + c]);
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
        return;
    }
    const int b = blockIdx.x - cb, row = b / nby, by = b % nby;
    const bf16* xr = xb + (size_t)row * xbs;
    const int wp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const bool wide = (K & 7) == 0 && ((reinterpret_cast<uintptr_t>(xr) | reinterpret_cast<uintptr_t>(Wb)) & 15) == 0;
    for (int o = by * 8 + wp; o < N; o += 8 * nby) {
        const bf16* wr = Wb + (size_t)o * K;
        float sum = 0.f;
        if (wide) {
            const uint4* xv = reinterpret_cast<const uint4*>(xr);
            const uint4* wv = reinterpret_cast<const uint4*>(wr);
            for (int kk = lane; kk < (K >> 3); kk += 32) {
                const uint4 a = xv[kk], bb = wv[kk];
                const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a);
                const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&bb);
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const float2 fa = __bfloat1622float2(a2[j]), fb = __bfloat1622float2(b2[j]);
                    sum = fmaf(fa.x, fb.x, sum);
                    sum = fmaf(fa.y, fb.y, sum);
                }
            }
        } else {
            for (int kk = lane; kk < K; kk += 32) sum = fmaf(b2f(xr[kk]), b2f(wr[kk]), sum);
        }
        sum = warp_sum(sum);
        if (lane == 0) yb[(size_t)row * ybs + o] = f2b(sum);
    }
}

// gdn_recurrent with qx_gnorm_sig after each token's outputs (the block owns its value head's 128): core [n, nv 128] = the gated norm of the rule's outputs (rounded to bf16 first, as they were stored), the gate z at
// z[t * zs + h * 128 ..], weight gw [128]. The same sums as the two kernels'. `st1` (where given, n > 1): the state after the first row too (a verification's of two rows: the state to go back to if the guess is
// wrong, in place of a copy of the state before the pass and the first row run again). A block a value head, 128 threads.
extern "C" __global__ void __launch_bounds__(128, 1) qx_gdn_rec_gn(const bf16* xc, int C, const bf16* ba, int sba, const float* A_log, const bf16* dt_bias, float* state, bf16* core, int n, int nk, int nv, const bf16* z, int zs, const bf16* gw, float eps, float* st1) {
    const int h = blockIdx.x, j = threadIdx.x, lane = j & 31, w = j >> 5;
    const int kh = h / (nv / nk), kd = nk * 128, vd = nv * 128;
    __shared__ __align__(16) float qk[2][128];
    __shared__ float red[2][4];
    float* st = state + (size_t)h * 128 * 128;
    float S[128];
#pragma unroll
    for (int i = 0; i < 128; i++) S[i] = st[i * 128 + j];
    const float Ah = -expf(A_log[h]), dtb = b2f(dt_bias[h]), scale = rsqrtf(128.f), gwj = b2f(gw[j]);
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
        // the gated norm of the head's outputs (qx_gnorm_sig's)
        const float ox = b2f(f2b((o0 + o1) + (o2 + o3)));
        const float r = rsqrtf(block_sum(ox * ox) / 128.f + eps);
        const float zz = b2f(z[(size_t)t * zs + h * 128 + j]);
        core[(size_t)t * vd + h * 128 + j] = f2b(qx_r(gwj * qx_r(ox * r)) * qx_sig(zz));
        if (t == 0 && st1 && n > 1) {
            float* s1 = st1 + (size_t)h * 128 * 128;
#pragma unroll
            for (int i = 0; i < 128; i++) s1[i * 128 + j] = S[i];
        }
    }
#pragma unroll
    for (int i = 0; i < 128; i++) st[i * 128 + j] = S[i];
}

// ---- PLE (the n-gram embedding's injection) -----------------------------------------------------------------------------------------------------------------------------------------------
// For a row: kv [n, 5 D] the products of its n-gram embedding (key: 4 D, value: D), x [n, 4 D] the streams. key_s = bf16(norm(key_s) (1 + wk_s)), q_s = bf16(norm(x_s) (1 + wq_s)), g_s = bf16(bf16(sum bf16(key_s q_s))
// / sqrt(D)), g_s = bf16(sign(g) sqrt(max(|g|, 1e-6))) (each op rounded as the reference's), gv_s = bf16(bf16(sigmoid(g_s)) value) into gv [n, 4 D], and its norm per stream bf16(norm(gv_s) (1 + wc_s)) into gvn
// [n, 4 D]. A block of 1024 threads a row, a quarter a stream (D at most 256 * 16).
extern "C" __global__ void __launch_bounds__(1024) qx_ple_gate(const bf16* kv, const bf16* x, const bf16* wk, const bf16* wq, const bf16* wc, bf16* gv, bf16* gvn, int D, float eps) {
    __shared__ float sh[32];
    const int q = threadIdx.x >> 8, j = threadIdx.x & 255;
    const size_t t = blockIdx.x, krow = t * 5 * D + (size_t)q * D, vrow = t * 5 * D + 4 * (size_t)D, xrow = t * QX_HC * D + (size_t)q * D;
    float kk[16], xx[16], ak = 0.f, ax = 0.f;
#pragma unroll
    for (int k = 0; k < 16; k++) {
        const int c = j + k * 256;
        kk[k] = c < D ? b2f(kv[krow + c]) : 0.f;
        xx[k] = c < D ? b2f(x[xrow + c]) : 0.f;
        ak += kk[k] * kk[k];
        ax += xx[k] * xx[k];
    }
    const float sk = rsqrtf(qx_quarter_sum(ak, sh) / D + eps), sx = rsqrtf(qx_quarter_sum(ax, sh) / D + eps);
    float dot = 0.f;
#pragma unroll
    for (int k = 0; k < 16; k++) {
        const int c = j + k * 256;
        if (c < D) dot += qx_r(qx_r(kk[k] * sk * (1.f + b2f(wk[q * D + c]))) * qx_r(xx[k] * sx * (1.f + b2f(wq[q * D + c]))));
    }
    float g = qx_r(qx_r(qx_quarter_sum(dot, sh)) / sqrtf((float)D));
    g = qx_r(copysignf(qx_r(sqrtf(fmaxf(fabsf(g), 1e-6f))), g));
    if (g == 0.f) g = 0.f;  // (sign(0) is 0 in the reference)
    const float sg = qx_r(qx_sig(g));
    float a2 = 0.f;
#pragma unroll
    for (int k = 0; k < 16; k++) {
        const int c = j + k * 256;
        kk[k] = c < D ? qx_r(sg * b2f(kv[vrow + c])) : 0.f;  // (gv, kept in kk)
        a2 += kk[k] * kk[k];
    }
    const float sc = rsqrtf(qx_quarter_sum(a2, sh) / D + eps);
#pragma unroll
    for (int k = 0; k < 16; k++) {
        const int c = j + k * 256;
        if (c < D) {
            gv[xrow + c] = f2b(kk[k]);
            gvn[xrow + c] = f2b(kk[k] * sc * (1.f + b2f(wc[q * D + c])));
        }
    }
}

// The dilated causal convolution of PLE and the injection: for each channel c of C (= 4 D) and row t, y = bf16(silu(bf16(sum_k w[c][k] gvn[t - 3 (3 - k)]))) (kernel 4, dilation 3; rows before the call's from
// the state st [9, C], oldest first, which is left holding the call's last 9 rows of gvn), then x = bf16(x + bf16(gv + y)). A thread a channel.
extern "C" __global__ void qx_ple_conv(const bf16* gvn, const bf16* gv, const bf16* w, bf16* st, bf16* x, int n, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const float w0 = b2f(w[c * 4]), w1 = b2f(w[c * 4 + 1]), w2 = b2f(w[c * 4 + 2]), w3 = b2f(w[c * 4 + 3]);
    float h[9];
#pragma unroll
    for (int i = 0; i < 9; i++) h[i] = b2f(st[i * C + c]);  // h[i]: the input 9 - i rows back
    for (int t = 0; t < n; t++) {
        const float in = b2f(gvn[(size_t)t * C + c]);
        // (taps oldest first: the inputs 9, 6, 3 and 0 rows back)
        float acc = __fmul_rn(w0, h[0]);
        acc = __fadd_rn(acc, __fmul_rn(w1, h[3]));
        acc = __fadd_rn(acc, __fmul_rn(w2, h[6]));
        acc = __fadd_rn(acc, __fmul_rn(w3, in));
        const float a = qx_r(acc);
        const float y = qx_r(a * qx_sig(a));
        const size_t i = (size_t)t * C + c;
        x[i] = f2b(b2f(x[i]) + qx_r(b2f(gv[i]) + y));
#pragma unroll
        for (int k = 0; k < 8; k++) h[k] = h[k + 1];
        h[8] = in;
    }
#pragma unroll
    for (int i = 0; i < 9; i++) st[i * C + c] = f2b(h[i]);
}

// The n-gram embedding's rows of a call's n rows of one sequence (the description executor's `Op::NgramRows`): row t's token tok[t], the token before it and the one before that (the call's earlier rows,
// else the sequence's state hist [2], oldest first, each token + 1 and 0 before the sequence, which reads as the end token eos; the one before that is eos too where the one before is eos), hashed as
// transformers' Qwen4ExpTextNGramEmbedding: bigram b = t mult[0] ^ c1 mult[1], trigram b ^ c2 mult[2] (unsigned 64-bit: each product is below 2^63, so as the reference's int64), table h of H (the first `per`
// bigrams') the row (b or the trigram) mod sizes[h] + offsets[h]; e [n, H row] = those rows of the table [rows, row] bf16. mult [3], sizes [H], offsets [H]: the checkpoint's int64 words. Block a row.
extern "C" __global__ void qx_ngram_rows(const int* tok, const int* hist, const unsigned long long* mult, const unsigned long long* sizes, const unsigned long long* offsets, const bf16* table, bf16* e, int H,
                                         int per, int row, int eos) {
    const int t = blockIdx.x;
    const int h1 = t >= 1 ? tok[t - 1] : (hist[1] > 0 ? hist[1] - 1 : eos);
    const int h2 = t >= 2 ? tok[t - 2] : (t == 1 ? (hist[1] > 0 ? hist[1] - 1 : eos) : (hist[0] > 0 ? hist[0] - 1 : eos));
    const unsigned long long c1 = (unsigned)h1, c2 = h1 == eos ? (unsigned long long)(unsigned)eos : (unsigned)h2;
    const unsigned long long bi = ((unsigned long long)(unsigned)tok[t] * mult[0]) ^ (c1 * mult[1]), tri = bi ^ (c2 * mult[2]);
    for (int i = threadIdx.x; i < H * row; i += blockDim.x) {
        const int h = i / row;
        const unsigned long long id = (h < per ? bi : tri) % sizes[h] + offsets[h];
        e[(size_t)t * H * row + i] = table[id * row + i % row];
    }
}

// The sequence's state after a call's n rows: the last two tokens (+ 1), oldest first. One thread.
extern "C" __global__ void qx_ngram_hist(const int* tok, int* hist, int n) {
    if (n >= 2) {
        hist[0] = tok[n - 2] + 1;
        hist[1] = tok[n - 1] + 1;
    } else if (n == 1) {
        hist[0] = hist[1];
        hist[1] = tok[0] + 1;
    }
}

// ---- the QSA indexer and attention --------------------------------------------------------------------------------------------------------------------------------------------------------
// The indexer's rows [n, (IH + 1) 128] (IH query heads, then the token's key): each query head normed (1 + wq) and turned (the rotary on its first 2 half dims, cos and sin bf16 [max_pos, half]) into iq [n, IH 128];
// the key, raw, into the key cache ik [positions, 128] at the row's position. The norm rounded, the rotary's products and sum each rounded (the reference's bf16 ops). Block (row, head), 128 threads.
extern "C" __global__ void qx_ix_prep(const bf16* qk, const bf16* wq, const bf16* cs, const bf16* sn, int half, const int* pos, bf16* iq, bf16* ik, int IH, float eps) {
    const int t = blockIdx.x, hh = blockIdx.y, i = threadIdx.x, p = pos[t];
    const bf16* x = qk + (size_t)t * (IH + 1) * 128 + hh * 128;
    if (hh == IH) {
        ik[(size_t)p * 128 + i] = x[i];
        return;
    }
    __shared__ float nrm[128];
    const float a = b2f(x[i]);
    const float s = rsqrtf(block_sum(a * a) / 128.f + eps);
    nrm[i] = qx_r(a * s * (1.f + b2f(wq[i])));
    __syncthreads();
    float o = nrm[i];
    if (i < 2 * half) {
        const int k = i < half ? i : i - half;
        const float c = b2f(cs[(size_t)p * half + k]), sv = b2f(sn[(size_t)p * half + k]);
        o = i < half ? qx_r(qx_r(nrm[i] * c) - qx_r(nrm[i + half] * sv)) : qx_r(qx_r(nrm[i] * c) + qx_r(nrm[i - half] * sv));
    }
    iq[(size_t)t * IH * 128 + hh * 128 + i] = f2b(o);
}

// The keys of the blocks of 4 tokens that the call's rows (at pos[0] .. pos[n-1], consecutive) complete: pk[b] = rotary(bf16(norm(bf16(mean of ik[4b .. 4b+3])) (1 + wk)), position 4b), the mean in fp32. A block a
// key block from pos[0] / 4 on (grid ceil(n / 4) + 1); one whose last token is past the call's last row is left to the call that completes it. 128 threads.
extern "C" __global__ void qx_ix_cmp(const bf16* ik, const bf16* wk, const bf16* cs, const bf16* sn, int half, const int* pos, bf16* pk, int n, float eps) {
    const int b = pos[0] / 4 + blockIdx.x, i = threadIdx.x;
    if (4 * b + 3 > pos[n - 1]) return;
    const float m = qx_r((((b2f(ik[(size_t)(4 * b) * 128 + i]) + b2f(ik[(size_t)(4 * b + 1) * 128 + i])) + b2f(ik[(size_t)(4 * b + 2) * 128 + i])) + b2f(ik[(size_t)(4 * b + 3) * 128 + i])) * 0.25f);
    const float s = rsqrtf(block_sum(m * m) / 128.f + eps);
    __shared__ float nrm[128];
    nrm[i] = qx_r(m * s * (1.f + b2f(wk[i])));
    __syncthreads();
    float o = nrm[i];
    if (i < 2 * half) {
        const int k = i < half ? i : i - half, p = 4 * b;
        const float c = b2f(cs[(size_t)p * half + k]), sv = b2f(sn[(size_t)p * half + k]);
        o = i < half ? qx_r(qx_r(nrm[i] * c) - qx_r(nrm[i + half] * sv)) : qx_r(qx_r(nrm[i] * c) + qx_r(nrm[i - half] * sv));
    }
    pk[(size_t)b * 128 + i] = f2b(o);
}

// conv_silu (qwen35.cu) for a prompt chunk with a thread a channel and QX_CT of the rows (grid (ceil(C / 256), ceil(n / QX_CT))), the same arithmetic for each output: conv_silu's single pass over the rows with a
// thread a channel had 40 blocks for Qwen's 10,240 channels. The state st [3, C] (the inputs before the chunk, the oldest first) is only read here; qx_conv_state moves it on after.
#define QX_CT 16
extern "C" __global__ void qx_conv_tok(const bf16* x, int xs, const bf16* w, const bf16* st, bf16* y, int n, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, t0 = blockIdx.y * QX_CT;
    if (c >= C) return;
    const float w0 = b2f(w[c * 4]), w1 = b2f(w[c * 4 + 1]), w2 = b2f(w[c * 4 + 2]), w3 = b2f(w[c * 4 + 3]);
    auto in = [&](int t) { return t >= 0 ? b2f(x[(size_t)t * xs + c]) : b2f(st[(3 + t) * C + c]); };
    float s0 = in(t0 - 3), s1 = in(t0 - 2), s2 = in(t0 - 1);
    const int t1 = min(n, t0 + QX_CT);
    for (int t = t0; t < t1; t++) {
        const float xv = b2f(x[(size_t)t * xs + c]);
        float acc = __fmul_rn(w0, s0);
        acc = __fadd_rn(acc, __fmul_rn(w1, s1));
        acc = __fadd_rn(acc, __fmul_rn(w2, s2));
        acc = __fadd_rn(acc, __fmul_rn(w3, xv));
        y[(size_t)t * C + c] = f2b(acc / (1.f + expf(-acc)));
        s0 = s1;
        s1 = s2;
        s2 = xv;
    }
}

// The convolution's state after n rows: their last three inputs (the state's own before them where n < 3), a thread a channel.
extern "C" __global__ void qx_conv_state(const bf16* x, int xs, bf16* st, int n, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    bf16 v[3];
#pragma unroll
    for (int i = 0; i < 3; i++) {
        const int t = n - 3 + i;
        v[i] = t >= 0 ? x[(size_t)t * xs + c] : st[(3 + t) * C + c];
    }
#pragma unroll
    for (int i = 0; i < 3; i++) st[i * C + c] = v[i];
}

// ---- the gated delta rule of a prompt chunk in two passes ---------------------------------------------------------------------------------------------------------------------------------------
// gdn_chunk's rule (qwen35.cu: chunks of 64 tokens, the WY form) with what does not depend on the state taken out of the pass over the chunks. Within a chunk, with G the decays' cumulative sums, beta the gates,
// L[t][s] = beta_t exp(G_t - G_s) k_t . k_s (s < t) and P[t][s] = exp(G_t - G_s) q_t . k_s (s <= t), gdn_chunk's U = (I + L)^-1 diag(beta) (V - diag(exp G) K S0) is T' R with T' = (I + L)^-1 diag(beta) and the
// residual R = V - diag(exp G) K S0; its outputs O = diag(exp G) Q S0 + P U are diag(exp G) Q S0 + M R with M = P T'; the state goes on as exp(G_last) S0 + K^T diag(exp(G_last - G)) U.
// qx_gdn_prep makes T', M and G of every chunk and value head at once, and the normed queries and keys of every chunk and key head (a block a chunk and value head; 64 by 64 matrices, a few KB each);
// qx_gdn_scan walks the chunks with the state (a block a value head's 32 value columns) on the tensor cores: K S0, Q S0, R, T' R, M R and the state's update. The prep's sums are fp32. What feeds the state
// keeps about fp32's precision on bf16 tensor cores: S0, R, T' and U' as bf16 pairs (high, low: x - high), their products the three or two of the pairs' parts that matter (the keys are bf16 already); the
// outputs' part takes S0's and R's high parts and M bf16, as the outputs are bf16. Scratch per chunk and value head: T''s two parts and M bf16 [64, 64], G fp32 [64]; per chunk and key head: the normed queries
// and keys bf16 [64, 128].
#define QG_C 64
#define QG_KS 65  // a bf16 row of 128 in shared memory: 64 words and one more
#define QG_LS 68  // a row of L, P or T': 64 floats and four more (16-byte rows: the substitution reads four at once)
__device__ __forceinline__ size_t qg_at(int c, int h, int nv) { return ((size_t)c * nv + h) * QG_C * QG_C; }
__device__ __forceinline__ size_t qg_kat(int c, int kh, int nk) { return ((size_t)c * nk + kh) * QG_C * 128; }

// Dynamic shared memory: 2 * 64 * 128 bf16 + (3 * 64 * QG_LS + 192) floats (85,760 bytes). The chunk's raw q and k rows come by cp.async into L and P's space, all at once; K K^T and Q K^T on the tensor cores
// (bf16 products, fp32 sums).
extern "C" __global__ void __launch_bounds__(256, 1) qx_gdn_prep(const bf16* xc, int C, const bf16* ba, int sba, const float* A_log, const bf16* dt_bias, int n, int nk, int nv, bf16* Thg, bf16* Tlg, bf16* Mg,
                                                                float* Gg, bf16* Kng, bf16* Qng) {
    extern __shared__ __align__(128) float sgp[];
    bf16* Kh = reinterpret_cast<bf16*>(sgp);  // the normed keys and queries [64][128] (swz)
    bf16* Qh = Kh + QG_C * 128;
    float* L = reinterpret_cast<float*>(Qh + QG_C * 128);
    float* P = L + QG_C * QG_LS;
    float* Tp = P + QG_C * QG_LS;  // T'
    float* Gc = Tp + QG_C * QG_LS;
    float* Bt = Gc + QG_C;
    const bf16* Rq = reinterpret_cast<const bf16*>(L);  // the raw rows [64][128], q then k
    const bf16* Rk = Rq + QG_C * 128;
    const int c = blockIdx.x, h = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g3 = nv / nk, kh = h / g3, kd = nk * 128, t0 = c * QG_C, tc = min(QG_C, n - t0);
    for (int e = tid; e < QG_C * 16; e += 256) {
        const int t = e >> 4, ch = e & 15;
        const bool ok = t < tc;
        const bf16* x = xc + (size_t)(t0 + (ok ? t : 0)) * C + ch * 8;
        cp16(s_addr(L + t * 64 + ch * 4), x + kh * 128, ok);
        cp16(s_addr(L + (QG_C + t) * 64 + ch * 4), x + kd + kh * 128, ok);
    }
    cp_commit();
    const float Ah = -expf(A_log[h]), dtb = b2f(dt_bias[h]);
    if (tid < QG_C) {
        const int t = tid;
        float g = 0.f, beta = 0.f;
        if (t < tc) {
            g = Ah * softplus_f(b2f(ba[(size_t)(t0 + t) * sba + nv + h]) + dtb);
            beta = 1.f / (1.f + expf(-b2f(ba[(size_t)(t0 + t) * sba + h])));
        }
        Gc[t] = g;
        Bt[t] = beta;
    }
    cp_wait<0>();
    __syncthreads();
    // the chunk's queries and keys normed and rounded to bf16 (kept for the scan by the key head's first value head)
    const bool first = h % g3 == 0;
    for (int t = warp; t < QG_C; t += 8) {
        float qv[4], kv[4], sq = 0.f, sk = 0.f;
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const int d = lane + 32 * e;
            qv[e] = b2f(Rq[t * 128 + d]);
            kv[e] = b2f(Rk[t * 128 + d]);
            sq += qv[e] * qv[e];
            sk += kv[e] * kv[e];
        }
        const float rq = rsqrtf(warp_sum(sq) + 1e-6f), rk = rsqrtf(warp_sum(sk) + 1e-6f);
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const bf16 kb = f2b(kv[e] * rk), qb = f2b(qv[e] * rq);
            const int d = lane + 32 * e;
            Qh[swz(t, d >> 3) + (d & 7)] = qb;
            Kh[swz(t, d >> 3) + (d & 7)] = kb;
            if (first) {
                Kng[qg_kat(c, kh, nk) + t * 128 + lane + 32 * e] = kb;
                Qng[qg_kat(c, kh, nk) + t * 128 + lane + 32 * e] = qb;
            }
        }
    }
    if (tid == 0) {
        float a = 0.f;
#pragma unroll
        for (int t = 0; t < QG_C; t++) {
            a += Gc[t];
            Gc[t] = a;
        }
    }
    __syncthreads();
    // L (warps 0-3: K K^T) and P (warps 4-7: Q K^T), a warp 16 rows by the 64 columns; the tiles right of the diagonal are zeros
    {
        const int mi = lane >> 3, r8 = lane & 7, g4 = lane >> 2, c4 = lane & 3, r0 = (warp & 3) * 16;
        const bool isL = warp < 4;
        const bf16* A = isL ? Kh : Qh;
        float acc[8][4];
#pragma unroll
        for (int j = 0; j < 8; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
#pragma unroll
        for (int ks = 0; ks < 8; ks++) {
            uint32_t a[4];
            ldsm4(a, s_addr(A + swz(r0 + (mi & 1) * 8 + r8, ks * 2 + (mi >> 1))));
#pragma unroll
            for (int np = 0; np < 4; np++) {
                if (np * 16 > r0 + 15) break;
                uint32_t b[4];
                ldsm4(b, s_addr(Kh + swz(np * 16 + (mi >> 1) * 8 + r8, ks * 2 + (mi & 1))));
                mma16816(acc[2 * np], a, b[0], b[1]);
                mma16816(acc[2 * np + 1], a, b[2], b[3]);
            }
        }
        float* D = isL ? L : P;
#pragma unroll
        for (int hf = 0; hf < 2; hf++) {
            const int t = r0 + g4 + 8 * hf;
            const float gt = Gc[t], bt = isL ? Bt[t] : 1.f;
#pragma unroll
            for (int j = 0; j < 8; j++)
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    const int u = 8 * j + 2 * c4 + e;
                    const bool in = isL ? u < t : u <= t;
                    D[t * QG_LS + u] = in ? bt * expf(gt - Gc[u]) * acc[j][2 * hf + e] : 0.f;
                }
        }
    }
    __syncthreads();
    const int ty = tid >> 4, tx = tid & 15;
    // T = (I + L)^-1 by forward substitution, a thread a column j (threads 0-63): T[t][j] = [t == j] - sum over u < t of L[t][u] T[u][j]; T' = T diag(beta) into shared memory and, as two bf16 parts, the scratch
    if (tid < QG_C) {
        const int j = tid;
        const size_t at = qg_at(c, h, nv) + j;
        float x[QG_C];
#pragma unroll
        for (int t = 0; t < QG_C; t++) {
            float a = t == j ? 1.f : 0.f;
#pragma unroll
            for (int u = 0; u < t; u += 4) {
                const float4 l = *reinterpret_cast<const float4*>(L + t * QG_LS + u);
                a = fmaf(-l.x, x[u], a);
                if (u + 1 < t) a = fmaf(-l.y, x[u + 1], a);
                if (u + 2 < t) a = fmaf(-l.z, x[u + 2], a);
                if (u + 3 < t) a = fmaf(-l.w, x[u + 3], a);
            }
            x[t] = a;
            const float tp = a * Bt[j];
            Tp[t * QG_LS + j] = tp;
            const bf16 hi = f2b(tp);
            Thg[at + t * QG_C] = hi;
            Tlg[at + t * QG_C] = f2b(tp - b2f(hi));
        }
    }
    __syncthreads();
    // M = P T' (lower triangular as both), bf16
    {
        float m[4][4] = {};
        for (int s2 = 0; s2 < QG_C; s2++) {
            float pa[4], tb[4];
#pragma unroll
            for (int r = 0; r < 4; r++) {
                pa[r] = P[(ty + 16 * r) * QG_LS + s2];
                tb[r] = Tp[s2 * QG_LS + tx + 16 * r];
            }
#pragma unroll
            for (int r = 0; r < 4; r++)
#pragma unroll
                for (int cc = 0; cc < 4; cc++) m[r][cc] = fmaf(pa[r], tb[cc], m[r][cc]);
        }
        const size_t at = qg_at(c, h, nv);
#pragma unroll
        for (int r = 0; r < 4; r++)
#pragma unroll
            for (int cc = 0; cc < 4; cc++) Mg[at + (ty + 16 * r) * QG_C + tx + 16 * cc] = f2b(m[r][cc]);
    }
    if (tid < QG_C) Gg[((size_t)c * nv + h) * QG_C + tid] = Gc[tid];
}

// The pass over the chunks of the n rows on the tensor cores: a block a value head's 32 value columns (grid nv x 4), 8 warps, warp w a chunk's rows 16 (w % 4) by columns 16 (w / 4) in (a) and (b). A chunk:
// (a) K S0 and diag(exp G) Q S0, then the residual R = V - diag(exp G) K S0; (b) U = T' R, U' = diag(exp(G_last - G)) U and the outputs diag(exp G) Q S0 + M R (times 128^-0.5, bf16 into core [n, nv 128]);
// (c) S0 = exp(G_last) S0 + K^T U' (a warp 16 of the state's 128 rows by the 32 columns). The state stays in the warps' accumulators, fp32, from the first chunk to the last; its bf16 pair
// in shared memory is the products' B. A chunk's queries and keys share two buffers in turn (the next chunk's keys load into the queries' once (a) is done, its queries into the keys' once (c) is); T' and M load
// during (c), V into registers. Dynamic shared memory GDN_SCAN_SMEM (90,368 bytes).
// the element offset of 16-byte chunk c of a 64-byte row r (four chunks) or a 128-byte row (eight): the chunks XORed so that ldmatrix's eight rows meet no bank twice
__device__ __forceinline__ int sw32(int r, int c) { return r * 32 + ((c ^ ((r >> 1) & 3)) << 3); }
__device__ __forceinline__ int sw64(int r, int c) { return r * 64 + ((c ^ (r & 7)) << 3); }
__device__ __forceinline__ void split_bf16(float a, float b, uint32_t& hi, uint32_t& lo) {
    hi = pack_bf16(a, b);
    const float2 h = bf2(hi);
    lo = pack_bf16(a - h.x, b - h.y);
}
extern "C" __global__ void __launch_bounds__(256, 1) qx_gdn_scan(const bf16* Thg, const bf16* Tlg, const bf16* Mg, const float* Gg, const bf16* Kng, const bf16* Qng, const bf16* xc, int C, float* state,
                                                                bf16* core, int n, int nk, int nv) {
    extern __shared__ __align__(128) bf16 sgs2[];
    bf16* X = sgs2;                  // two buffers of [64][128] (swz): a chunk's queries in X[c & 1], its keys in the other
    bf16* Th = X + 2 * QG_C * 128;   // T''s parts [64][64] each (sw64)
    bf16* Tl = Th + QG_C * QG_C;
    bf16* Mb = Tl + QG_C * QG_C;     // M [64][64] (sw64)
    bf16* Sh = Mb + QG_C * QG_C;     // S0's parts [128][32] each (sw32)
    bf16* Sl = Sh + 128 * 32;
    bf16* Rh = Sl + 128 * 32;        // R's parts [64][32] each (sw32)
    bf16* Rl = Rh + QG_C * 32;
    bf16* Uh = Rl + QG_C * 32;       // U''s parts [64][32] each (sw32)
    bf16* Ul = Uh + QG_C * 32;
    float* Gs = reinterpret_cast<float*>(Ul + QG_C * 32);  // [64]
    const int h = blockIdx.x, cb = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int kh = h / (nv / nk), kd = nk * 128, vd = nv * 128, mi = lane >> 3, r8 = lane & 7, g4 = lane >> 2, c4 = lane & 3;
    const float scale = rsqrtf(128.f);
    const int chunks = (n + QG_C - 1) / QG_C;
    const int r0 = (warp & 3) * 16, i0 = 16 * warp, nh = warp >> 2;
    auto load_q = [&](int c) {
        const bf16* src = Qng + qg_kat(c, kh, nk);
        bf16* dst = X + (c & 1) * QG_C * 128;
        for (int e = tid; e < QG_C * 16; e += 256) cp16(s_addr(dst + swz(e >> 4, e & 15)), src + (e >> 4) * 128 + (e & 15) * 8, true);
        if (tid < 16) cp16(s_addr(Gs + 4 * tid), Gg + ((size_t)c * nv + h) * QG_C + 4 * tid, true);
    };
    auto load_k = [&](int c) {
        const bf16* src = Kng + qg_kat(c, kh, nk);
        bf16* dst = X + ((c + 1) & 1) * QG_C * 128;
        for (int e = tid; e < QG_C * 16; e += 256) cp16(s_addr(dst + swz(e >> 4, e & 15)), src + (e >> 4) * 128 + (e & 15) * 8, true);
    };
    auto load_t = [&](int c) {
        const size_t at = qg_at(c, h, nv);
        for (int e = tid; e < QG_C * 8; e += 256) {
            const int r = e >> 3, ch = e & 7;
            cp16(s_addr(Th + sw64(r, ch)), Thg + at + r * QG_C + ch * 8, true);
            cp16(s_addr(Tl + sw64(r, ch)), Tlg + at + r * QG_C + ch * 8, true);
            cp16(s_addr(Mb + sw64(r, ch)), Mg + at + r * QG_C + ch * 8, true);
        }
    };
    // a warp's region of V: rows r0 + g4 (+ 8), columns 16 nh + 8 j + 2 c4 (+ 1) of the block's 32
    uint32_t vr[2][2];
    auto load_v = [&](int c) {
#pragma unroll
        for (int hf = 0; hf < 2; hf++) {
            const int t = c * QG_C + r0 + g4 + 8 * hf;
            const bf16* src = xc + (size_t)min(t, n - 1) * C + 2 * kd + h * 128 + cb * 32 + 16 * nh + 2 * c4;
#pragma unroll
            for (int j = 0; j < 2; j++) vr[hf][j] = t < n ? *reinterpret_cast<const uint32_t*>(src + 8 * j) : 0u;
        }
    };
    load_q(0);
    load_k(0);
    load_t(0);
    cp_commit();
    load_v(0);
    // the state: warp w's rows 16 w + g4 (+ 8), columns 8 j + 2 c4 (+ 1) of the block's 32
    float* st = state + (size_t)h * 128 * 128 + cb * 32;
    float sa[4][4];
    auto put_s = [&]() {
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int hf = 0; hf < 2; hf++) {
                uint32_t hi, lo;
                split_bf16(sa[j][2 * hf], sa[j][2 * hf + 1], hi, lo);
                const int o = sw32(i0 + g4 + 8 * hf, j) + 2 * c4;
                *reinterpret_cast<uint32_t*>(Sh + o) = hi;
                *reinterpret_cast<uint32_t*>(Sl + o) = lo;
            }
    };
#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
        for (int hf = 0; hf < 2; hf++) {
            const float2 v = *reinterpret_cast<const float2*>(st + (size_t)(i0 + g4 + 8 * hf) * 128 + 8 * j + 2 * c4);
            sa[j][2 * hf] = v.x;
            sa[j][2 * hf + 1] = v.y;
        }
    put_s();
    for (int c = 0; c < chunks; c++) {
        const int t0 = c * QG_C, tc = min(QG_C, n - t0);
        cp_wait<0>();
        __syncthreads();  // (the chunk's loads and the state's pair are in)
        const bf16* Qb = X + (c & 1) * QG_C * 128;
        const bf16* Kb = X + ((c + 1) & 1) * QG_C * 128;
        // (a) a warp's region of K S0 (S0's high and low parts) and of Q S0 (its high part)
        float ks_[2][4], qs[2][4];
#pragma unroll
        for (int j = 0; j < 2; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) ks_[j][e] = qs[j][e] = 0.f;
#pragma unroll
        for (int ks = 0; ks < 8; ks++) {
            uint32_t ak[4], aq[4], bh[4], bl[4];
            ldsm4(ak, s_addr(Kb + swz(r0 + (mi & 1) * 8 + r8, ks * 2 + (mi >> 1))));
            ldsm4(aq, s_addr(Qb + swz(r0 + (mi & 1) * 8 + r8, ks * 2 + (mi >> 1))));
            const int o = sw32(ks * 16 + (mi & 1) * 8 + r8, nh * 2 + (mi >> 1));
            ldsm4t(bh, s_addr(Sh + o));
            ldsm4t(bl, s_addr(Sl + o));
            mma16816(ks_[0], ak, bl[0], bl[1]);
            mma16816(ks_[1], ak, bl[2], bl[3]);
            mma16816(ks_[0], ak, bh[0], bh[1]);
            mma16816(ks_[1], ak, bh[2], bh[3]);
            mma16816(qs[0], aq, bh[0], bh[1]);
            mma16816(qs[1], aq, bh[2], bh[3]);
        }
        const float gl = Gs[QG_C - 1];
#pragma unroll
        for (int hf = 0; hf < 2; hf++) {
            const float e = expf(Gs[r0 + g4 + 8 * hf]);
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const float2 v = bf2(vr[hf][j]);
                uint32_t hi, lo;
                split_bf16(v.x - e * ks_[j][2 * hf], v.y - e * ks_[j][2 * hf + 1], hi, lo);
                const int o = sw32(r0 + g4 + 8 * hf, nh * 2 + j) + 2 * c4;
                *reinterpret_cast<uint32_t*>(Rh + o) = hi;
                *reinterpret_cast<uint32_t*>(Rl + o) = lo;
                qs[j][2 * hf] *= e;
                qs[j][2 * hf + 1] *= e;
            }
        }
        __syncthreads();  // (R is in; every warp is done with the queries)
        if (c + 1 < chunks) {
            load_k(c + 1);
            cp_commit();
            load_v(c + 1);
        }
        // (b) the region's U = T' R (both parts of each), U' into shared memory, and outputs diag(exp G) Q S0 + M R
        float u[2][4];
#pragma unroll
        for (int j = 0; j < 2; j++) u[j][0] = u[j][1] = u[j][2] = u[j][3] = 0.f;
#pragma unroll
        for (int ks = 0; ks < 4; ks++) {
            uint32_t ah[4], al[4], am[4], bh[4], bl[4];
            ldsm4(ah, s_addr(Th + sw64(r0 + (mi & 1) * 8 + r8, ks * 2 + (mi >> 1))));
            ldsm4(al, s_addr(Tl + sw64(r0 + (mi & 1) * 8 + r8, ks * 2 + (mi >> 1))));
            ldsm4(am, s_addr(Mb + sw64(r0 + (mi & 1) * 8 + r8, ks * 2 + (mi >> 1))));
            const int o = sw32(ks * 16 + (mi & 1) * 8 + r8, nh * 2 + (mi >> 1));
            ldsm4t(bh, s_addr(Rh + o));
            ldsm4t(bl, s_addr(Rl + o));
            mma16816(u[0], al, bh[0], bh[1]);
            mma16816(u[1], al, bh[2], bh[3]);
            mma16816(u[0], ah, bl[0], bl[1]);
            mma16816(u[1], ah, bl[2], bl[3]);
            mma16816(u[0], ah, bh[0], bh[1]);
            mma16816(u[1], ah, bh[2], bh[3]);
            mma16816(qs[0], am, bh[0], bh[1]);
            mma16816(qs[1], am, bh[2], bh[3]);
        }
#pragma unroll
        for (int hf = 0; hf < 2; hf++) {
            const int t = r0 + g4 + 8 * hf;
            const float d = expf(gl - Gs[t]);
            if (t < tc) {
                bf16* op = core + (size_t)(t0 + t) * vd + h * 128 + cb * 32 + 16 * nh + 2 * c4;
#pragma unroll
                for (int j = 0; j < 2; j++) *reinterpret_cast<uint32_t*>(op + 8 * j) = pack_bf16(qs[j][2 * hf] * scale, qs[j][2 * hf + 1] * scale);
            }
#pragma unroll
            for (int j = 0; j < 2; j++) {
                uint32_t hi, lo;
                split_bf16(u[j][2 * hf] * d, u[j][2 * hf + 1] * d, hi, lo);
                const int o = sw32(t, nh * 2 + j) + 2 * c4;
                *reinterpret_cast<uint32_t*>(Uh + o) = hi;
                *reinterpret_cast<uint32_t*>(Ul + o) = lo;
            }
        }
        __syncthreads();  // (U' is in; every warp is done with T', M and R)
        if (c + 1 < chunks) {
            load_t(c + 1);
            cp_commit();
        }
        // (c) the state's update
        const float el = expf(gl);
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) sa[j][e] *= el;
#pragma unroll
        for (int ks = 0; ks < 4; ks++) {
            uint32_t a[4];
            ldsm4t(a, s_addr(Kb + swz(ks * 16 + (mi >> 1) * 8 + r8, (i0 >> 3) + (mi & 1))));
#pragma unroll
            for (int np = 0; np < 2; np++) {
                uint32_t bh[4], bl[4];
                const int o = sw32(ks * 16 + (mi & 1) * 8 + r8, np * 2 + (mi >> 1));
                ldsm4t(bh, s_addr(Uh + o));
                ldsm4t(bl, s_addr(Ul + o));
                mma16816(sa[2 * np], a, bl[0], bl[1]);
                mma16816(sa[2 * np], a, bh[0], bh[1]);
                mma16816(sa[2 * np + 1], a, bl[2], bl[3]);
                mma16816(sa[2 * np + 1], a, bh[2], bh[3]);
            }
        }
        put_s();
        if (c + 1 < chunks) {
            __syncthreads();  // (every warp is done with this chunk's keys and G)
            load_q(c + 1);
            cp_commit();
        }
    }
#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
        for (int hf = 0; hf < 2; hf++) *reinterpret_cast<float2*>(st + (size_t)(i0 + g4 + 8 * hf) * 128 + 8 * j + 2 * c4) = make_float2(sa[j][2 * hf], sa[j][2 * hf + 1]);
}

// The indexer's scores of a prompt chunk on the tensor cores: scores [n, pmax] fp32, s[i][p] = the sum over the 4 heads h of relu(scale iq[i][h] . pk[p]) (the heads' weights are 1) for each pool p < P_i = (pos[i] + 1) / 4,
// iq [n, 4, 128] and pk [pools, 128] bf16; nothing for the rows with P_i <= ksel (they attend to every token: glm_select reads no scores of theirs). glm_idx_score's arithmetic, the products summed in another order
// (fp32). A block 16 rows (64 rows of A: a row's 4 heads in turn) by 128 pools, both whole in shared memory (K = 128); 4 warps, each the 16 A-rows of 4 rows by the 128 pools; a row's heads are the accumulators' rows
// 4 r + h, lanes 4 h apart, added by two shuffles. Grid (ceil(pools / 128), ceil(n / 16)), pools the most a row of the call sees.
extern "C" __global__ void __launch_bounds__(128) qx_ix_score(const bf16* __restrict__ iq, const bf16* __restrict__ pk, const int* __restrict__ pos, float* __restrict__ scores, int n, int pmax, int ksel, float scale) {
    __shared__ __align__(128) bf16 sa[64 * HD];
    __shared__ __align__(128) bf16 sb[128 * HD];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int r0 = blockIdx.y * 16, p0 = blockIdx.x * 128;
    const int last = (pos[min(n, r0 + 16) - 1] + 1) / 4;  // (the positions ascend: the block's last row sees the most pools)
    if (last <= ksel || p0 >= last) return;
    for (int c = tid; c < 64 * 16; c += 128) {
        const int r = c >> 4, ch = c & 15;
        const bool ok = r0 * 4 + r < n * 4;
        cp16(s_addr(sa + swz(r, ch)), iq + ((size_t)r0 * 4 + (ok ? r : 0)) * HD + ch * 8, ok);
    }
    for (int c = tid; c < 128 * 16; c += 128) {
        const int r = c >> 4, ch = c & 15;
        const bool ok = p0 + r < last;
        cp16(s_addr(sb + swz(r, ch)), pk + ((size_t)p0 + (ok ? r : 0)) * HD + ch * 8, ok);
    }
    cp_commit();
    cp_wait<0>();
    __syncthreads();
    const int mi = lane >> 3, r8 = lane & 7;
    float acc[16][4];
#pragma unroll
    for (int i = 0; i < 16; i++) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < HD / 16; ks++) {
        uint32_t a[4];
        ldsm4(a, s_addr(sa + swz(warp * 16 + (mi & 1) * 8 + r8, ks * 2 + (mi >> 1))));
#pragma unroll
        for (int np = 0; np < 8; np++) {
            uint32_t b[4];
            ldsm4(b, s_addr(sb + swz(np * 16 + (mi >> 1) * 8 + r8, ks * 2 + (mi & 1))));
            mma16816(acc[2 * np], a, b[0], b[1]);
            mma16816(acc[2 * np + 1], a, b[2], b[3]);
        }
    }
    // the accumulators' rows g and g + 8 are A-rows warp * 16 + g (+ 8): row r0 + warp * 4 + g / 4 (+ 2), head g % 4
    const int g = lane >> 2, t4 = lane & 3;
    const int ra = r0 + warp * 4 + (g >> 2), rb = ra + 2;
    const int pa = ra < n ? (pos[ra] + 1) / 4 : 0, pb = rb < n ? (pos[rb] + 1) / 4 : 0;
#pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        float v[4];
#pragma unroll
        for (int q = 0; q < 4; q++) {
            v[q] = fmaxf(acc[nt][q] * scale, 0.f);
            v[q] += __shfl_xor_sync(0xffffffffu, v[q], 4);
            v[q] += __shfl_xor_sync(0xffffffffu, v[q], 8);
        }
        const int p = p0 + nt * 8 + 2 * t4;
        if ((g & 3) == 0) {
            if (p < pa) scores[(size_t)ra * pmax + p] = v[0];
            if (p + 1 < pa) scores[(size_t)ra * pmax + p + 1] = v[1];
            if (p < pb) scores[(size_t)rb * pmax + p] = v[2];
            if (p + 1 < pb) scores[(size_t)rb * pmax + p + 1] = v[3];
        }
    }
}

// The attention of each row t of a prompt chunk over the tokens the indexer picked (sel[t][0 .. cnt[t])), on the tensor cores: qx_attn_sel's arithmetic (fp32 scores and sums, exp2 of the scores times sl2, the
// probabilities rounded to bf16 for the values, out = bf16(bf16(attention) sigmoid(gate))), 12 query heads to a key-value head. A warp takes a (row, key-value head): its 12 query heads are the rows of an m16 tile (the
// last 4 zero), the picked tokens 16 a step. The keys and values go from the cache straight into the fragments' registers by 16-byte loads, with no shared memory: the first product's k axis (the channels) is
// permuted (a dot product's terms add in any order) so that in k-step ks lane t4 holds channels 64 t4 + 4 ks .. + 3, 64 consecutive ones over the steps; a step's tokens sit in the score tiles' columns so that the
// second product's k axis gives lane t4 the tokens 4 t4 .. 4 t4 + 3 (tile 0's column c is token 4 (c / 2) + c % 2, tile 1's that + 2); the output's channels are permuted too (n-tile j's column c is channel
// 32 c + j) and put back by the stores. The queries in shared memory, a warp's 12 rows of 64 groups of 4 channels, a group's place XOR-ed by its row and lane so that a k-step's loads meet no bank twice. Grid
// ceil(n / 2) blocks of 4 warps (2 rows, 2 key-value heads); nk == 2.
#define QA_G 12
extern "C" __global__ void __launch_bounds__(128) qx_attn_tc(const bf16* __restrict__ q, int qs, const bf16* __restrict__ gate, int gs, const bf16* __restrict__ kc, const bf16* __restrict__ vc,
                                                            const int* __restrict__ sel, const int* __restrict__ cnt, int tmax, bf16* __restrict__ out, int os, int n, float sl2) {
    __shared__ uint2 sq[4][QA_G * 64];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3;
    const int t = blockIdx.x * 2 + (warp >> 1), kvh = warp & 1;
    if (t >= n) return;
    uint2* myq = sq[warp];
    for (int e = lane; e < QA_G * 64; e += 32) {
        const int r = e >> 6, G = e & 63, tq = G >> 4, kq = G & 15;
        myq[r * 64 + tq * 16 + (kq ^ (4 * (r & 3) + tq))] = *reinterpret_cast<const uint2*>(q + (size_t)t * qs + (kvh * QA_G + r) * 256 + G * 4);
    }
    __syncwarp();
    const int total = cnt[t];
    const int* tok = sel + (size_t)t * tmax;
    const bool hi = g < QA_G - 8;  // row g + 8 is a query head
    const uint2* qa = myq + g * 64 + t4 * 16;
    const uint2* qb = myq + (hi ? g + 8 : g) * 64 + t4 * 16;
    const int qx = 4 * (g & 3) + t4;  // (rows g and g + 8 have the same XOR)
    float o[32][4];
#pragma unroll
    for (int j = 0; j < 32; j++) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
    for (int k0 = 0; k0 < total; k0 += 16) {
        // the first product: keys k0 + 4 (g / 2) + g % 2 (tile 0's column g) and + 2 (tile 1's), lane t4's 64 channels of each
        const int ka = k0 + 4 * (g >> 1) + (g & 1), kb = ka + 2;
        const bf16* Ka = kc + ((size_t)(ka < total ? tok[ka] : 0) * 2 + kvh) * 256 + 64 * t4;
        const bf16* Kb = kc + ((size_t)(kb < total ? tok[kb] : 0) * 2 + kvh) * 256 + 64 * t4;
        float sa[4] = {0.f, 0.f, 0.f, 0.f}, sb[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int c = 0; c < 8; c++) {
            const uint4 xa = *reinterpret_cast<const uint4*>(Ka + 8 * c), xb = *reinterpret_cast<const uint4*>(Kb + 8 * c);
#pragma unroll
            for (int h2 = 0; h2 < 2; h2++) {
                const int ks = 2 * c + h2;
                const uint2 ua = qa[ks ^ qx], ub = hi ? qb[ks ^ qx] : make_uint2(0, 0);
                const uint32_t a[4] = {ua.x, ub.x, ua.y, ub.y};
                mma16816(sa, a, h2 ? xa.z : xa.x, h2 ? xa.w : xa.y);
                mma16816(sb, a, h2 ? xb.z : xb.x, h2 ? xb.w : xb.y);
            }
        }
        // lane (g, t4) holds the scores of tokens k0 + 4 t4 .. + 3 (sa[0], sa[1], sb[0], sb[1] for row g; sa[2], sa[3], sb[2], sb[3] for row g + 8)
        float v[2][4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const bool ok = k0 + 4 * t4 + i < total;
            const float x0 = i < 2 ? sa[i] : sb[i - 2], x1 = i < 2 ? sa[i + 2] : sb[i];
            v[0][i] = ok ? x0 * sl2 : -INFINITY;
            v[1][i] = ok ? x1 * sl2 : -INFINITY;
        }
        float corr[2];
#pragma unroll
        for (int r = 0; r < 2; r++) {
            float mx = fmaxf(fmaxf(v[r][0], v[r][1]), fmaxf(v[r][2], v[r][3]));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            const float mn = fmaxf(m[r], mx);
            corr[r] = exp2f(m[r] - mn);
            m[r] = mn;
            float ps = 0.f;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                v[r][i] = exp2f(v[r][i] - mn);
                ps += v[r][i];
            }
            l[r] = l[r] * corr[r] + ps;
        }
#pragma unroll
        for (int j = 0; j < 32; j++) {
            o[j][0] *= corr[0];
            o[j][1] *= corr[0];
            o[j][2] *= corr[1];
            o[j][3] *= corr[1];
        }
        // the second product: P (bf16) as the A fragment (k = 2 t4, 2 t4 + 1: tokens 4 t4, 4 t4 + 1; k = 2 t4 + 8, + 9: tokens 4 t4 + 2, + 3), the values of lane t4's 4 tokens at channels 32 g + j for n-tile j
        const uint32_t pa[4] = {pack_bf16(v[0][0], v[0][1]), pack_bf16(v[1][0], v[1][1]), pack_bf16(v[0][2], v[0][3]), pack_bf16(v[1][2], v[1][3])};
        const bf16* V0 = vc + ((size_t)(k0 + 4 * t4 < total ? tok[k0 + 4 * t4] : 0) * 2 + kvh) * 256 + 32 * g;
        const bf16* V1 = vc + ((size_t)(k0 + 4 * t4 + 1 < total ? tok[k0 + 4 * t4 + 1] : 0) * 2 + kvh) * 256 + 32 * g;
        const bf16* V2 = vc + ((size_t)(k0 + 4 * t4 + 2 < total ? tok[k0 + 4 * t4 + 2] : 0) * 2 + kvh) * 256 + 32 * g;
        const bf16* V3 = vc + ((size_t)(k0 + 4 * t4 + 3 < total ? tok[k0 + 4 * t4 + 3] : 0) * 2 + kvh) * 256 + 32 * g;
#pragma unroll
        for (int pc = 0; pc < 4; pc++) {
            const uint4 y0 = *reinterpret_cast<const uint4*>(V0 + 8 * pc), y1 = *reinterpret_cast<const uint4*>(V1 + 8 * pc);
            const uint4 y2 = *reinterpret_cast<const uint4*>(V2 + 8 * pc), y3 = *reinterpret_cast<const uint4*>(V3 + 8 * pc);
            const uint32_t w0[4] = {y0.x, y0.y, y0.z, y0.w}, w1[4] = {y1.x, y1.y, y1.z, y1.w}, w2[4] = {y2.x, y2.y, y2.z, y2.w}, w3[4] = {y3.x, y3.y, y3.z, y3.w};
#pragma unroll
            for (int e = 0; e < 8; e++) {
                const uint32_t sel2 = (e & 1) ? 0x7632u : 0x5410u;
                mma16816(o[8 * pc + e], pa, __byte_perm(w0[e >> 1], w1[e >> 1], sel2), __byte_perm(w2[e >> 1], w3[e >> 1], sel2));
            }
        }
    }
    float ls[2];
#pragma unroll
    for (int r = 0; r < 2; r++) {
        float x = l[r];
        x += __shfl_xor_sync(0xffffffffu, x, 1);
        x += __shfl_xor_sync(0xffffffffu, x, 2);
        ls[r] = x;
    }
    // row g's channels 64 t4 + j and 64 t4 + 32 + j are o[j][0] and o[j][1] (row g + 8: o[j][2], o[j][3]); 8 channels a 16-byte store, gated
#pragma unroll
    for (int r = 0; r < 2; r++) {
        if (r == 1 && !hi) break;
        const int h = kvh * QA_G + g + 8 * r;
        const bf16* gr = gate + (size_t)t * gs + h * 512 + 256 + 64 * t4;
        bf16* op = out + (size_t)t * os + h * 256 + 64 * t4;
#pragma unroll
        for (int half = 0; half < 2; half++) {
#pragma unroll
            for (int u = 0; u < 4; u++) {
                const uint4 gv = *reinterpret_cast<const uint4*>(gr + 32 * half + 8 * u);
                const uint32_t gw[4] = {gv.x, gv.y, gv.z, gv.w};
                uint32_t ow[4];
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const float a0 = b2f(f2b(o[8 * u + 2 * e][2 * r + half] / ls[r])), a1 = b2f(f2b(o[8 * u + 2 * e + 1][2 * r + half] / ls[r]));
                    const float g0 = __uint_as_float(gw[e] << 16), g1 = __uint_as_float(gw[e] & 0xffff0000u);
                    ow[e] = pack_bf16(a0 / (1.f + expf(-g0)), a1 / (1.f + expf(-g1)));
                }
                *reinterpret_cast<uint4*>(op + 32 * half + 8 * u) = make_uint4(ow[0], ow[1], ow[2], ow[3]);
            }
        }
    }
}

// The attention of a decode step's few rows over their picked tokens, split for the memory's sake: W warps a block, each one 16-token step of qx_attn_tc's warp (a row's 12 query heads of a key-value head on the
// tensor cores, the cache straight into the fragments): block b takes the row's tokens 16 W b .. 16 W (b + 1) - 1, its warps' outputs combined in shared memory (each rescaled to the block's maximum, added in
// pairs), the block's unnormalized outputs, maximum (log2 units) and sum into part [(t nq + h) S + b][258] for attn_merge (kernels/attn.cu), S = gridDim.x. Grid (ceil(tmax / (16 W)), nk, n); nk == 2. Dynamic shared memory
// (W / 2) 16 KB (the pairs' adds) + W 128 floats (the maxima and sums).
template <int W>
__device__ __forceinline__ void qx_attn_dec_t(const bf16* __restrict__ q, int qs, const bf16* __restrict__ kc, const bf16* __restrict__ vc, const int* __restrict__ sel, const int* __restrict__ cnt, int tmax,
                                              float sl2, float* __restrict__ part) {
    __shared__ uint2 myq[QA_G * 64];
    extern __shared__ __align__(16) float dsm[];
    float4* slot = reinterpret_cast<float4*>(dsm);  // [W / 2][32 j][32 lanes] float4
    float* mw = dsm + (W / 2) * 4096;                // [W][16 rows]
    float* lw = mw + W * 16;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3;
    const int t = blockIdx.z, kvh = blockIdx.y, S = gridDim.x;
    for (int e = threadIdx.x; e < QA_G * 64; e += 32 * W) {
        const int r = e >> 6, G = e & 63, tq = G >> 4, kq = G & 15;
        myq[r * 64 + tq * 16 + (kq ^ (4 * (r & 3) + tq))] = *reinterpret_cast<const uint2*>(q + (size_t)t * qs + (kvh * QA_G + r) * 256 + G * 4);
    }
    __syncthreads();
    const int total = cnt[t], k0 = (blockIdx.x * W + warp) * 16, j1 = min(total, k0 + 16);
    const int* tok = sel + (size_t)t * tmax;
    const bool hi = g < QA_G - 8;
    const uint2* qa = myq + g * 64 + t4 * 16;
    const uint2* qb = myq + (hi ? g + 8 : g) * 64 + t4 * 16;
    const int qx = 4 * (g & 3) + t4;
    float o[32][4];
#pragma unroll
    for (int j = 0; j < 32; j++) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
    if (k0 < total) {
        const int ka = k0 + 4 * (g >> 1) + (g & 1), kb = ka + 2;
        const int i0 = k0 + 4 * t4;
        // (every index first: the keys' and values' loads then go out together)
        const int ta = ka < j1 ? tok[ka] : 0, tb = kb < j1 ? tok[kb] : 0;
        const int v0 = i0 < j1 ? tok[i0] : 0, v1 = i0 + 1 < j1 ? tok[i0 + 1] : 0, v2 = i0 + 2 < j1 ? tok[i0 + 2] : 0, v3 = i0 + 3 < j1 ? tok[i0 + 3] : 0;
        const bf16* Ka = kc + ((size_t)ta * 2 + kvh) * 256 + 64 * t4;
        const bf16* Kb = kc + ((size_t)tb * 2 + kvh) * 256 + 64 * t4;
        float sa[4] = {0.f, 0.f, 0.f, 0.f}, sb[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int c = 0; c < 8; c++) {
            const uint4 xa = *reinterpret_cast<const uint4*>(Ka + 8 * c), xb = *reinterpret_cast<const uint4*>(Kb + 8 * c);
#pragma unroll
            for (int h2 = 0; h2 < 2; h2++) {
                const int ks = 2 * c + h2;
                const uint2 ua = qa[ks ^ qx], ub = hi ? qb[ks ^ qx] : make_uint2(0, 0);
                const uint32_t a[4] = {ua.x, ub.x, ua.y, ub.y};
                mma16816(sa, a, h2 ? xa.z : xa.x, h2 ? xa.w : xa.y);
                mma16816(sb, a, h2 ? xb.z : xb.x, h2 ? xb.w : xb.y);
            }
        }
        float v[2][4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const bool ok = i0 + i < j1;
            const float x0 = i < 2 ? sa[i] : sb[i - 2], x1 = i < 2 ? sa[i + 2] : sb[i];
            v[0][i] = ok ? x0 * sl2 : -INFINITY;
            v[1][i] = ok ? x1 * sl2 : -INFINITY;
        }
#pragma unroll
        for (int r = 0; r < 2; r++) {
            float mx = fmaxf(fmaxf(v[r][0], v[r][1]), fmaxf(v[r][2], v[r][3]));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            m[r] = mx;
            float ps = 0.f;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                v[r][i] = exp2f(v[r][i] - mx);
                ps += v[r][i];
            }
            l[r] = ps;
        }
        const uint32_t pa[4] = {pack_bf16(v[0][0], v[0][1]), pack_bf16(v[1][0], v[1][1]), pack_bf16(v[0][2], v[0][3]), pack_bf16(v[1][2], v[1][3])};
        const bf16* V0 = vc + ((size_t)v0 * 2 + kvh) * 256 + 32 * g;
        const bf16* V1 = vc + ((size_t)v1 * 2 + kvh) * 256 + 32 * g;
        const bf16* V2 = vc + ((size_t)v2 * 2 + kvh) * 256 + 32 * g;
        const bf16* V3 = vc + ((size_t)v3 * 2 + kvh) * 256 + 32 * g;
#pragma unroll
        for (int pc = 0; pc < 4; pc++) {
            const uint4 y0 = *reinterpret_cast<const uint4*>(V0 + 8 * pc), y1 = *reinterpret_cast<const uint4*>(V1 + 8 * pc);
            const uint4 y2 = *reinterpret_cast<const uint4*>(V2 + 8 * pc), y3 = *reinterpret_cast<const uint4*>(V3 + 8 * pc);
            const uint32_t w0[4] = {y0.x, y0.y, y0.z, y0.w}, w1[4] = {y1.x, y1.y, y1.z, y1.w}, w2[4] = {y2.x, y2.y, y2.z, y2.w}, w3[4] = {y3.x, y3.y, y3.z, y3.w};
#pragma unroll
            for (int e = 0; e < 8; e++) {
                const uint32_t sel2 = (e & 1) ? 0x7632u : 0x5410u;
                mma16816(o[8 * pc + e], pa, __byte_perm(w0[e >> 1], w1[e >> 1], sel2), __byte_perm(w2[e >> 1], w3[e >> 1], sel2));
            }
        }
#pragma unroll
        for (int r = 0; r < 2; r++) {
            l[r] += __shfl_xor_sync(0xffffffffu, l[r], 1);
            l[r] += __shfl_xor_sync(0xffffffffu, l[r], 2);
        }
    }
    // the block's maximum of each row, every warp's outputs and sum rescaled to it, then added in pairs
    if (t4 == 0) {
        mw[warp * 16 + g] = m[0];
        mw[warp * 16 + g + 8] = m[1];
    }
    __syncthreads();
    float M[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int w = 0; w < W; w++) {
        M[0] = fmaxf(M[0], mw[w * 16 + g]);
        M[1] = fmaxf(M[1], mw[w * 16 + g + 8]);
    }
    float sc[2];
#pragma unroll
    for (int r = 0; r < 2; r++) {
        sc[r] = m[r] == -INFINITY ? 0.f : exp2f(m[r] - M[r]);
        l[r] *= sc[r];
    }
#pragma unroll
    for (int j = 0; j < 32; j++) {
        o[j][0] *= sc[0];
        o[j][1] *= sc[0];
        o[j][2] *= sc[1];
        o[j][3] *= sc[1];
    }
#pragma unroll
    for (int half = W / 2; half >= 1; half >>= 1) {
        if (warp >= half && warp < 2 * half) {
            float4* sl = slot + (warp - half) * 1024;
#pragma unroll
            for (int j = 0; j < 32; j++) sl[j * 32 + lane] = make_float4(o[j][0], o[j][1], o[j][2], o[j][3]);
            if (t4 == 0) {
                lw[(warp - half) * 16 + g] = l[0];
                lw[(warp - half) * 16 + g + 8] = l[1];
            }
        }
        __syncthreads();
        if (warp < half) {
            const float4* sl = slot + warp * 1024;
#pragma unroll
            for (int j = 0; j < 32; j++) {
                const float4 x = sl[j * 32 + lane];
                o[j][0] += x.x;
                o[j][1] += x.y;
                o[j][2] += x.z;
                o[j][3] += x.w;
            }
            l[0] += lw[warp * 16 + g];
            l[1] += lw[warp * 16 + g + 8];
        }
        __syncthreads();
    }
    if (warp != 0) return;
#pragma unroll
    for (int r = 0; r < 2; r++) {
        if (r == 1 && !hi) break;
        // row g's channels 64 t4 + j and 64 t4 + 32 + j are o[j][2 r] and o[j][2 r + 1]
        float* pp = part + ((size_t)(t * 2 * QA_G + kvh * QA_G + g + 8 * r) * S + blockIdx.x) * 258;
#pragma unroll
        for (int j = 0; j < 32; j++) {
            pp[64 * t4 + j] = o[j][2 * r];
            pp[64 * t4 + 32 + j] = o[j][2 * r + 1];
        }
        if (t4 == 0) {
            pp[256] = M[r];
            pp[257] = l[r];
        }
    }
}
extern "C" __global__ void __launch_bounds__(128) qx_attn_dec4(const bf16* q, int qs, const bf16* kc, const bf16* vc, const int* sel, const int* cnt, int tmax, float sl2, float* part) {
    qx_attn_dec_t<4>(q, qs, kc, vc, sel, cnt, tmax, sl2, part);
}
extern "C" __global__ void __launch_bounds__(256) qx_attn_dec8(const bf16* q, int qs, const bf16* kc, const bf16* vc, const int* sel, const int* cnt, int tmax, float sl2, float* part) {
    qx_attn_dec_t<8>(q, qs, kc, vc, sel, cnt, tmax, sl2, part);
}

#define QX_KB 32   // tokens a tile of the attention over picked tokens
#define QX_KS 129  // a key's row in shared memory: 128 words and one more

// The attention of each row t of q [n, qs] (query head h at h * 256) over the tokens sel[t][0 .. cnt[t]) (the indexer's picks, ascending) of the cache kc, vc [positions, nk, 256], nq / nk query heads to a key-value
// head, as the attention kind's warp kernel computes it (fp32 scores and sums, the probabilities rounded to bf16 for the values; sl2 = 256^-0.5 log2(e)); then out = bf16(bf16(attention) sigmoid(gate)), the gate at gate[t gs + h 512 + 256 ..].
// Grid (n, nk, S): with S > 1, block z takes the z-th part of the row's tokens and writes its unnormalized outputs, maximum and sum to part [(t nq + h) S + z][258] for attn_merge. A block the G query heads of a
// key-value head (a warp each). Dynamic shared memory (QX_KB (QX_KS + 128) + 128 G) * 4 bytes.
extern "C" __global__ void __launch_bounds__(512) qx_attn_sel(const bf16* q, int qs, const bf16* gate, int gs, const bf16* kc, const bf16* vc, const int* sel, const int* cnt, int tmax, bf16* out, int os, int nq, int nk,
                                                             float sl2, float* part, int S) {
    extern __shared__ uint32_t sm[];
    uint32_t* Ks = sm;
    uint32_t* Vs = Ks + QX_KB * QX_KS;
    uint32_t* Qs = Vs + QX_KB * 128;
    const int G = nq / nk, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int t = blockIdx.x, kvh = blockIdx.y, h = kvh * G + w;
    const int* tok = sel + (size_t)t * tmax;
    const int total = cnt[t];
    const int span = ((total + S - 1) / S + QX_KB - 1) / QX_KB * QX_KB;
    const int j0 = min(total, (int)blockIdx.z * span), j1 = min(total, j0 + span);
    for (int c = lane; c < 128; c += 32) Qs[w * 128 + c] = reinterpret_cast<const uint32_t*>(q + (size_t)t * qs + h * 256)[c];
    float m_run = -INFINITY, l_run = 0.f, acc[8];
#pragma unroll
    for (int e = 0; e < 8; e++) acc[e] = 0.f;
    for (int k0 = j0; k0 < j1; k0 += QX_KB) {
        __syncthreads();
        for (int c = threadIdx.x; c < QX_KB * 64; c += blockDim.x) {
            const bool isv = c >= QX_KB * 32;
            const int cc = isv ? c - QX_KB * 32 : c, j = cc >> 5, pp = cc & 31;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (k0 + j < j1) v = reinterpret_cast<const uint4*>((isv ? vc : kc) + ((size_t)tok[k0 + j] * nk + kvh) * 256)[pp];
            if (isv) {
                *reinterpret_cast<uint4*>(Vs + j * 128 + pp * 4) = v;
            } else {
                uint32_t* d = Ks + j * QX_KS + pp * 4;
                d[0] = v.x;
                d[1] = v.y;
                d[2] = v.z;
                d[3] = v.w;
            }
        }
        __syncthreads();
        const uint32_t* kr = Ks + lane * QX_KS;
        const uint32_t* qr = Qs + w * 128;
        float s0 = 0.f, s1 = 0.f;
#pragma unroll 8
        for (int c = 0; c < 128; c++) {
            const float2 a = bf2(qr[c]), b = bf2(kr[c]);
            s0 = fmaf(a.x, b.x, s0);
            s1 = fmaf(a.y, b.y, s1);
        }
        const float s = k0 + lane < j1 ? (s0 + s1) * sl2 : -INFINITY;
        const float m_new = fmaxf(m_run, warp_max(s));
        const float p = exp2f(s - m_new), corr = exp2f(m_run - m_new);
        l_run = l_run * corr + warp_sum(p);
        const float pb = b2f(f2b(p));
#pragma unroll
        for (int e = 0; e < 8; e++) acc[e] *= corr;
#pragma unroll 4
        for (int j = 0; j < QX_KB; j++) {
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
    if (S > 1) {
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

// The tokens each of a decode step's rows attends to: glm_select's picks (the same sel and cnt: the ksel best-scoring complete pools, equal scores the smaller pool first, in pool order, then the tail's tokens), found
// faster for a long context: the row's keys read once into shared memory (where P <= cap, else read each pass), and the picks written in pool order by a scan
// of each thread's run of pools (no sort). A block a row, 1024 threads; dynamic shared memory min(Pmax, cap) * 4 bytes.
extern "C" __global__ void __launch_bounds__(1024) qx_select(const float* scores, int Pmax, const int* pos, int* sel, int* cnt, int Tmax, int kp, int ksel, int cap) {
    extern __shared__ unsigned qks[];
    __shared__ unsigned hist[256];
    __shared__ unsigned long long wsum[32];
    __shared__ int chosen, rem;
    const int i = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int q = pos[i], P = (q + 1) / kp, tl = q + 1 - P * kp;
    int* out = sel + (size_t)i * Tmax;
    if (P <= ksel) {
        const int T = P * kp + tl;
        for (int t = tid; t < T; t += blockDim.x) out[t] = t;
        if (tid == 0) cnt[i] = T;
        return;
    }
    const float* s = scores + (size_t)i * Pmax;
    const bool insm = P <= cap;
    if (insm)
        for (int p = tid; p < P; p += blockDim.x) qks[p] = gm_key(s[p]);
    auto key = [&](int p) { return insm ? qks[p] : gm_key(s[p]); };
    unsigned prefix = 0, pmask = 0;
    int remaining = ksel;
    for (int pass = 0; pass < 4; pass++) {
        const int shift = 24 - 8 * pass;
        if (tid < 256) hist[tid] = 0;
        __syncthreads();
        for (int p = tid; p < P; p += blockDim.x) {
            const unsigned k = key(p);
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
    }
    // every pool whose key is above the threshold `prefix`, and the first `remaining` of those equal to it: a thread's run of pools counted (above, equal), the counts scanned, each run's picks written at their place
    const int cp = (P + blockDim.x - 1) / blockDim.x, lo = min(tid * cp, P), hi = min(lo + cp, P);
    unsigned gt = 0, eq = 0;
    for (int p = lo; p < hi; p++) {
        const unsigned k = key(p);
        gt += k > prefix;
        eq += k == prefix;
    }
    const unsigned long long mine = ((unsigned long long)gt << 32) | eq;
    unsigned long long incl = mine;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const unsigned long long v = __shfl_up_sync(0xffffffffu, incl, off);
        if (lane >= off) incl += v;
    }
    if (lane == 31) wsum[warp] = incl;
    __syncthreads();
    if (warp == 0) {
        unsigned long long w = wsum[lane], wi = w;
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const unsigned long long v = __shfl_up_sync(0xffffffffu, wi, off);
            if (lane >= off) wi += v;
        }
        wsum[lane] = wi - w;
    }
    __syncthreads();
    const unsigned long long before = wsum[warp] + incl - mine;
    const int gt0 = (int)(before >> 32), eq0 = (int)(before & 0xffffffffu);
    int o = gt0 + min(remaining, eq0), er = eq0;
    for (int p = lo; p < hi; p++) {
        const unsigned k = key(p);
        const bool take = k > prefix || (k == prefix && er < remaining);
        er += k == prefix;
        if (take) {
            for (int j = 0; j < kp; j++) out[o * kp + j] = p * kp + j;
            o++;
        }
    }
    for (int u = tid; u < tl; u += blockDim.x) out[ksel * kp + u] = P * kp + u;
    if (tid == 0) cnt[i] = ksel * kp + tl;
}

// ---- the routed experts as codebook rows (glyd-hbr-codes/2 at 2560 x 640: the expert cache's places; src/model_qwen4x.rs `QxMoeCb`) -----------------------------------------------------------------------------
// A layer's experts are held in the basis of one rotation of the hidden axis, R x = 512^-0.5 (I_5 (x) H_512) (s x): five natural-order Walsh-Hadamard transforms of 512 values each, s the layer's 2,560 signs. An expert
// is two matrices of rows of 2,560 codes in that basis, each row its codes and its own codebook (bf16 entries, absolute): `downT` (640 rows: down transposed, row n the weights of neuron n over the outputs) then `up`
// (1,280 rows: gate's 640 and up's 640). A row's codes by format f (0 cb2: 4 entries, 2 bits, byte b codes 4b..4b+3 low bits first; 1 cb3: 8 entries, codes 8g..8g+7 the 24-bit little-endian integer of bytes
// 3g..3g+2; 2 cb4: 16 entries, the low nibble first; 3 tern: 3 entries, byte = c0 + 3 c1 + 9 c2 + 27 c3 + 81 c4). A place: downT's codes, then up's; downT's codebooks, then up's; in a layer of mixed formats the
// last byte of the place's codebook bytes holds the formats (up's in the low nibble, down's in the high one). A lane takes 80 codes of a row in the first product (20, 30, 40 or 16 bytes), a thread 40 outputs in the
// second (10, 15, 20 or 8 bytes): every format's bytes split evenly. No atomics: the same bits in whatever place, every time.

#define QC_H 2560
#define QC_I 640
__device__ __forceinline__ int qc_row_bytes(int f) { return f == 0 || f == 4 ? 640 : f == 1 || f == 5 ? 960 : f == 2 || f == 6 ? 1280 : 512; }
__device__ __forceinline__ int qc_book_bytes(int f) { return f == 0 ? 8 : f == 1 ? 16 : f == 2 ? 32 : f >= 4 ? 2 : 6; }

// (the trellis rows' values, tq_val / tq_v / tq_state and the TQ_ constants: kernels/cb2_prefill.cu, included before this file)

// Format q at K = 3 and 4 bits a weight (formats 5 and 6: q3, q4; a row 240 or 320 words, its state of weight t the 14 bits from bit K t, around the end). The decode's pieces,
// whole functions so that any product kernel takes them:
// tq_words<K, N>(row, j0, r, w): N words of a row from word j0 on (around the end), shifted left r bits (0 to 31), so that bit 0 of w[0] is the row's bit 32 j0 + r.
template <int K, int N>
__device__ __forceinline__ void tq_words(const unsigned* row, int j0, int r, unsigned* w) {
    constexpr int W = 2560 * K / 32;
    unsigned v[N + 1];
#pragma unroll
    for (int q = 0; q <= N; q++) v[q] = row[j0 + q < W ? j0 + q : j0 + q - W];
#pragma unroll
    for (int q = 0; q < N; q++) w[q] = __funnelshift_l(v[q + 1], v[q], r);
}

// The first product's lane: the row's 80 weights from weight 80 lane on against z (16-byte aligned, the lane's 80 values), unscaled.
template <int K>
__device__ __forceinline__ float tq_lane_dot80(const unsigned char* rp, const float* z) {
    constexpr int N = (K * 79 >> 5) + 2;
    const int lane = threadIdx.x & 31, bit = 80 * K * lane;
    unsigned w[N];
    tq_words<K, N>(reinterpret_cast<const unsigned*>(rp), bit >> 5, bit & 31, w);
    float acc = 0.f;
#pragma unroll
    for (int q = 0; q < 5; q++) {
        float zv[16];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const float4 u = *reinterpret_cast<const float4*>(z + q * 16 + 4 * i);
            zv[4 * i] = u.x, zv[4 * i + 1] = u.y, zv[4 * i + 2] = u.z, zv[4 * i + 3] = u.w;
        }
#pragma unroll
        for (int j = 0; j < 16; j++) acc = fmaf(tq_vk(w, K * (16 * q + j)), zv[j], acc);
    }
    return acc;
}

// The second product's thread: acc[0..40] += c times the row's weights 40 l .. 40 l + 39.
template <int K>
__device__ __forceinline__ void tq_down40k(const unsigned char* rp, int l, float c, float* acc) {
    constexpr int N = (K * 39 >> 5) + 2;
    const int bit = 40 * K * l;
    unsigned w[N];
    tq_words<K, N>(reinterpret_cast<const unsigned*>(rp), bit >> 5, bit & 31, w);
#pragma unroll
    for (int j = 0; j < 40; j++) acc[j] = fmaf(tq_vk(w, K * j), c, acc[j]);
}


// One 512-point transform of v [512] in shared memory (a block of 512 threads), unscaled.
__device__ __forceinline__ void qc_wht512(float* v) {
    for (int h = 1; h < 512; h <<= 1) {
        if (threadIdx.x < 256) {
            const int r = threadIdx.x, i = (r / h) * 2 * h + (r % h);
            const float a = v[i], c = v[i + h];
            v[i] = a + c;
            v[i + h] = a - c;
        }
        __syncthreads();
    }
}

// x' = R x for the n rows of x (bf16, rows xs apart): xr [n, 2560] fp32 and xrb [n, 2560] bf16 where given. A block a row's 512 values of one transform (grid n x 5), 512 threads.
extern "C" __global__ void __launch_bounds__(512) qx_cb_rot_in(const bf16* x, int xs, const signed char* sg, float* xr, bf16* xrb) {
    __shared__ float v[512];
    const int t = blockIdx.x, i = blockIdx.y * 512 + threadIdx.x;
    v[threadIdx.x] = b2f(x[(size_t)t * xs + i]) * (float)sg[i];
    __syncthreads();
    qc_wht512(v);
    const float o = v[threadIdx.x] * 0.044194173824159216f;
    xr[(size_t)t * QC_H + i] = o;
    if (xrb) xrb[(size_t)t * QC_H + i] = f2b(o);
}



// A decode step's front in one launch (T at most 8 tokens; H = QC_H): blocks 0 .. ceil(E / 8) - 1 the logits (two warps an expert, each half its row, 8 a block of 512), then T x 5 blocks x' = R x (to xr,
// xrb and the host's hxrb), side by side; the block that finishes last picks (moe_select_row, a warp a token), writes the choices' weights and their numbers among the layer's kept experts (gids, and the host's copies)
// and, where the layer's map of places is given (`map`: a place for each kept expert, -1 for none; the host's memory), each pair's place (way, -1 for none: the host's pair), then moves the host's word.
extern "C" __global__ void __launch_bounds__(512) qx_router_rot(const bf16* x, int xs, const bf16* W, float* logits, int T, int E, int H, const float* bias, long long* ids, float* wts, int k, int flags, float scale, unsigned* counter, float* hwts, const signed char* sg, float* xr, bf16* xrb, const int* local, long long* gids, int experts, long long* hgids, bf16* hxrb, unsigned* hflag, const int* map, long long* way) {
    __shared__ float v[512];
    __shared__ float half2s[8][8];
    __shared__ bool last;
    const int rb = (E + 7) / 8, lane = threadIdx.x & 31;
    if ((int)blockIdx.x < rb) {
        // (two warps an expert, each half of its row: the block's 16 warps 8 experts, so that 64 SMs read the router's rows)
        const int wp = threadIdx.x >> 5, hf = wp & 1, e = blockIdx.x * 8 + (wp >> 1);
        float mine = 0.f;
        if (e < E) {
            const uint4* wv = reinterpret_cast<const uint4*>(W + (size_t)e * H) + hf * (H >> 4);
            float acc[8];
#pragma unroll
            for (int j = 0; j < 8; j++) acc[j] = 0.f;
#pragma unroll 5
            for (int c = lane; c < (H >> 4); c += 32) {
                float w[8];
                moe_bf8(wv[c], w);
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    if (j < T) {
                        float u[8];
                        moe_bf8(reinterpret_cast<const uint4*>(x + (size_t)j * xs)[hf * (H >> 4) + c], u);
#pragma unroll
                        for (int i = 0; i < 8; i++) acc[j] = fmaf(u[i], w[i], acc[j]);
                    }
                }
            }
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const float sm = warp_sum(acc[j]);
                if (lane == j) mine = sm;  // (lane j: the half's sum for token j)
            }
            if (hf == 1 && lane < 8) half2s[wp >> 1][lane] = mine;
        }
        __syncthreads();
        if (e < E && hf == 0 && lane < T) logits[(size_t)lane * E + e] = mine + half2s[wp >> 1][lane];
    } else {
        const int b = blockIdx.x - rb, t = b / 5, i = (b % 5) * 512 + threadIdx.x;
        v[threadIdx.x] = b2f(x[(size_t)t * xs + i]) * (float)sg[i];
        __syncthreads();
        qc_wht512(v);
        const float o = v[threadIdx.x] * 0.044194173824159216f;
        xr[(size_t)t * QC_H + i] = o;
        if (xrb) xrb[(size_t)t * QC_H + i] = f2b(o);
        if (hxrb) hxrb[(size_t)t * QC_H + i] = f2b(o);
    }
    // the last block to finish (every block's writes fenced before it counts: the rotation's to the host's buffer for the system, the logits for the device)
    if ((int)blockIdx.x < rb) {
        __threadfence();
    } else {
        __threadfence_system();
    }
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(counter, 1u) == gridDim.x - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    const int t = threadIdx.x >> 5;
    if (t < T) {
        moe_select_row<16>(logits + (size_t)t * E, bias, nullptr, ids, wts, t, E, k, flags, scale);
        __syncwarp();
        if (lane < k) {
            const size_t j = (size_t)t * k + lane;
            if (hwts) hwts[j] = wts[j];
            const long long e = ids[j];
            const long long g = (e >= 0 && e < experts) ? (long long)local[e] : -1ll;
            gids[j] = g;
            if (hgids) hgids[j] = g;
            if (map) way[j] = g >= 0 ? (long long)map[g] : -1ll;
        }
        __threadfence_system();
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        *counter = 0;
        if (hflag) {
            // (the word's count kept beside the block counter, counter[1]: the host's word written, not read across the bus; no fence after it, nothing on the GPU waits for it)
            const unsigned v = counter[1] + 1;
            counter[1] = v;
            __threadfence_system();
            *(volatile unsigned*)hflag = v;
        }
    }
}

// Entry `lane` of a codebook of k entries (0 past them).
__device__ __forceinline__ float qc_entry(const unsigned char* book, int lane, int k) {
    return lane < k ? __uint_as_float(((unsigned)(*reinterpret_cast<const unsigned short*>(book + 2 * lane))) << 16) : 0.f;
}

// The five base-3 digits of each byte, three bits each.
__device__ __forceinline__ void qc_tern_table(unsigned short* tab) {
    for (int b = threadIdx.x; b < 256; b += blockDim.x) {
        unsigned v = b, o = 0;
#pragma unroll
        for (int j = 0; j < 5; j++) {
            o |= (v % 3u) << (3 * j);
            v /= 3u;
        }
        tab[b] = (unsigned short)o;
    }
}

// The 4 floats at z (16-byte aligned) into f.
__device__ __forceinline__ void qc_z4(const float* z, float* f) {
    const float4 u = *reinterpret_cast<const float4*>(z);
    f[0] = u.x, f[1] = u.y, f[2] = u.z, f[3] = u.w;
}

// A lane's 80 codes of row `rp` (format f, entries `ent` one to a lane, the cb2 four in e) times x' (z: the lane's 80 values, 16-byte aligned; read 4 at a time but for the ternary's): its part of the row's
// product.
__device__ __forceinline__ float qc_lane_dot(int f, const unsigned char* rp, const float* z, float ent, float e0, float e1, float e2, float e3, const unsigned short* tab) {
    const int lane = threadIdx.x & 31;
    float acc = 0.f;
    if (f == 0) {
        const unsigned* wp = reinterpret_cast<const unsigned*>(rp + lane * 20);
#pragma unroll
        for (int q = 0; q < 5; q++) {
            const unsigned w = wp[q];
            float zv[16];
#pragma unroll
            for (int i = 0; i < 4; i++) qc_z4(z + q * 16 + 4 * i, zv + 4 * i);
#pragma unroll
            for (int j = 0; j < 16; j++) acc = fmaf(gc_pick((w >> (2 * j)) & 3u, e0, e1, e2, e3), zv[j], acc);
        }
    } else if (f == 1) {
        const unsigned short* sp = reinterpret_cast<const unsigned short*>(rp + lane * 30);
        unsigned char b[30];
#pragma unroll
        for (int q = 0; q < 15; q++) {
            const unsigned short s = sp[q];
            b[2 * q] = s & 255;
            b[2 * q + 1] = s >> 8;
        }
#pragma unroll
        for (int g = 0; g < 10; g++) {
            const unsigned v = b[3 * g] | (b[3 * g + 1] << 8) | (b[3 * g + 2] << 16);
            float zv[8];
            qc_z4(z + g * 8, zv);
            qc_z4(z + g * 8 + 4, zv + 4);
#pragma unroll
            for (int j = 0; j < 8; j++) acc = fmaf(__shfl_sync(0xffffffffu, ent, (v >> (3 * j)) & 7), zv[j], acc);
        }
    } else if (f == 2) {
        const unsigned* wp = reinterpret_cast<const unsigned*>(rp + lane * 40);
#pragma unroll
        for (int q = 0; q < 10; q++) {
            const unsigned w = wp[q];
            float zv[8];
            qc_z4(z + q * 8, zv);
            qc_z4(z + q * 8 + 4, zv + 4);
#pragma unroll
            for (int j = 0; j < 8; j++) acc = fmaf(__shfl_sync(0xffffffffu, ent, (w >> (4 * j)) & 15), zv[j], acc);
        }
    } else if (f == 4) {
        const unsigned* wp = reinterpret_cast<const unsigned*>(rp);
        unsigned w6[6];
#pragma unroll
        for (int q = 0; q < 5; q++) w6[q] = wp[5 * lane + q];
        w6[5] = wp[lane == 31 ? 0 : 5 * lane + 5];
#pragma unroll
        for (int q = 0; q < 5; q++) {
            float zv[16];
#pragma unroll
            for (int i = 0; i < 4; i++) qc_z4(z + q * 16 + 4 * i, zv + 4 * i);
#pragma unroll
            for (int j = 0; j < 16; j++) acc = fmaf(tq_v(__funnelshift_l(w6[q + 1], w6[q], 2 * j) >> (32 - TQ_L)), zv[j], acc);
        }
        acc *= ent;
    } else if (f == 5) {
        acc = tq_lane_dot80<3>(rp, z) * ent;
    } else if (f == 6) {
        acc = tq_lane_dot80<4>(rp, z) * ent;
    } else {
        const uint4 u = *reinterpret_cast<const uint4*>(rp + lane * 16);
        const unsigned w4[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int q = 0; q < 16; q++) {
            const unsigned d = tab[(w4[q >> 2] >> (8 * (q & 3))) & 255];
#pragma unroll
            for (int j = 0; j < 5; j++) acc = fmaf(__shfl_sync(0xffffffffu, ent, (d >> (3 * j)) & 7), z[q * 5 + j], acc);
        }
    }
    return acc;
}

// gu [pairs, 1280] fp32: for each pair p with a place pl = way[p] >= 0, up's 1,280 rows times x' of the pair's token (p / k). A block 32 rows of a pair (grid 40 x pairs), 8 warps of 4 rows; x' in shared memory with 4
// floats of padding after each lane's 80 (float4 reads, the lanes' in different banks). `mixed`: the formats from the place's last codebook byte (else cb2). At most 40 registers, 6 blocks an SM (qxprod: 4 tokens
// 58.9 -> 55.5 us, 1 token 18.3 -> 17.5; tokens sharing an expert in one block, its codes made values once: slower in a decode step, few of its pairs share)
__device__ __forceinline__ void qx_cb_up_b(const unsigned char* codes, unsigned long long cstride, const unsigned char* books, unsigned long long bstride, const long long* way, const float* xr, float* gu, int k, int mixed) {
    const int p = blockIdx.y;
    const long long pl = way[p];
    if (pl < 0) return;
    __shared__ __align__(16) float xs[32 * 84];
    __shared__ unsigned short tab[256];
    const unsigned char* bplace = books + (size_t)pl * bstride;
    const int fb = mixed ? bplace[bstride - 1] : 0, fu = fb & 15, fd = fb >> 4;
    const float* xrow = xr + (size_t)(p / k) * QC_H;
    for (int i = threadIdx.x; i < QC_H; i += 256) xs[(i / 80) * 84 + i % 80] = xrow[i];
    if (fu == 3) qc_tern_table(tab);
    __syncthreads();
    const unsigned char* cb = codes + (size_t)pl * cstride + (size_t)QC_I * qc_row_bytes(fd);
    const unsigned char* bb = bplace + (size_t)QC_I * qc_book_bytes(fd);
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, rb = qc_row_bytes(fu), kb = qc_book_bytes(fu);
    const float* z = xs + lane * 84;
    for (int rr = 0; rr < 4; rr++) {
        const int row = blockIdx.x * 32 + w * 4 + rr;
        const unsigned char* book = bb + (size_t)row * kb;
        float ent = 0.f, e0 = 0.f, e1 = 0.f, e2 = 0.f, e3 = 0.f;
        if (fu == 0) {
            const uint2 bk = *reinterpret_cast<const uint2*>(book);
            const float2 b01 = bf2(bk.x), b23 = bf2(bk.y);
            e0 = b01.x, e1 = b01.y, e2 = b23.x, e3 = b23.y;
        } else if (fu >= 4) {
            ent = b2f(*reinterpret_cast<const bf16*>(book)) * TQ_K;
        } else {
            ent = qc_entry(book, lane, fu == 1 ? 8 : fu == 2 ? 16 : 3);
        }
        float acc = qc_lane_dot(fu, cb + (size_t)row * rb, z, ent, e0, e1, e2, e3, tab);
        acc = warp_sum(acc);
        if (lane == 0) gu[(size_t)p * 2 * QC_I + row] = acc;
    }
}

// acc[0..40] += c times weights OFF .. OFF + 39 of the words w4 (16 a word; the window around into the next).
template <int OFF>
__device__ __forceinline__ void tq_down40(const unsigned* w4, float c, float* acc) {
#pragma unroll
    for (int j = 0; j < 40; j++) {
        const int t = OFF + j, q = t >> 4;
        acc[j] = fmaf(tq_v(__funnelshift_l(w4[q + 1], w4[q], 2 * (t & 15)) >> (32 - TQ_L)), c, acc[j]);
    }
}

// part [pairs, S, 2560] fp32: for each pair p with a place, the sum over the neurons n of run s (640 / S of them) of act[n] times row n of downT, act[n] = bf16(silu(min(gate_n, limit)) clamp(up_n, -limit, limit))
// (Qwen's SwiGLU has no clamp: limit is the largest float). A block (run, pair), 256 threads: four groups of 64 taking a neuron each in turn, a thread 40 outputs of the row; the groups' sums added through shared memory.
__device__ __forceinline__ void qx_cb_down_b(const unsigned char* codes, unsigned long long cstride, const unsigned char* books, unsigned long long bstride, const long long* way, const float* gu, float* part, int S, float limit, int mixed) {
    const int s = blockIdx.x, p = blockIdx.y;
    const long long pl = way[p];
    if (pl < 0) return;
    __shared__ float act[QC_I];
    __shared__ float red[4 * QC_H];
    __shared__ unsigned short tab[256];
    const unsigned char* bplace = books + (size_t)pl * bstride;
    const int fd = mixed ? bplace[bstride - 1] >> 4 : 0;
    const int tid = threadIdx.x, g = tid >> 6, l = tid & 63, lane = tid & 31;
    const int ns = QC_I / S, n0 = s * ns;
    for (int i = tid; i < ns; i += 256) {
        const float gg = fminf(gu[(size_t)p * 2 * QC_I + n0 + i], limit), uu = fminf(fmaxf(gu[(size_t)p * 2 * QC_I + QC_I + n0 + i], -limit), limit);
        act[i] = b2f(f2b(gg / (1.f + expf(-gg)) * uu));
    }
    if (fd == 3) qc_tern_table(tab);
    __syncthreads();
    const int rb = qc_row_bytes(fd), kb = qc_book_bytes(fd);
    const unsigned char* cb = codes + (size_t)pl * cstride + (size_t)n0 * rb;
    const unsigned char* bb = bplace + (size_t)n0 * kb;
    float acc[40];
#pragma unroll
    for (int j = 0; j < 40; j++) acc[j] = 0.f;
    for (int i = g; i < ns; i += 4) {
        const unsigned char* rp = cb + (size_t)i * rb;
        const float a = act[i];
        if (fd == 0) {
            const uint2 bk = *reinterpret_cast<const uint2*>(bb + (size_t)i * 8);
            const float2 b01 = bf2(bk.x), b23 = bf2(bk.y);
            const float e0 = a * b01.x, e1 = a * b01.y, e2 = a * b23.x, e3 = a * b23.y;
            const unsigned short* sp = reinterpret_cast<const unsigned short*>(rp + l * 10);
#pragma unroll
            for (int q = 0; q < 5; q++) {
                const unsigned w = sp[q];
#pragma unroll
                for (int j = 0; j < 8; j++) acc[q * 8 + j] += gc_pick((w >> (2 * j)) & 3u, e0, e1, e2, e3);
            }
        } else if (fd == 4) {
            const float c = a * b2f(*reinterpret_cast<const bf16*>(bb + (size_t)i * kb)) * TQ_K;
            const unsigned* wp = reinterpret_cast<const unsigned*>(rp);
            const int j0 = (5 * l) >> 1;
            unsigned w5[5], w4[4];
#pragma unroll
            for (int q = 0; q < 5; q++) w5[q] = wp[j0 + q < 160 ? j0 + q : j0 + q - 160];
            // (an odd thread's 40 start half a word in: its words shifted by 16 bits, so that every thread runs the same instructions, not each half of a warp in turn)
#pragma unroll
            for (int q = 0; q < 4; q++) w4[q] = (l & 1) ? __funnelshift_l(w5[q + 1], w5[q], 16) : w5[q];
            tq_down40<0>(w4, c, acc);
        } else if (fd >= 5) {
            const float c = a * b2f(*reinterpret_cast<const bf16*>(bb + (size_t)i * kb)) * TQ_K;
            if (fd == 5) tq_down40k<3>(rp, l, c, acc);
            else tq_down40k<4>(rp, l, c, acc);
        } else {
            const float ent = a * qc_entry(bb + (size_t)i * kb, lane, fd == 1 ? 8 : fd == 2 ? 16 : 3);
            if (fd == 1) {
                const unsigned char* bp = rp + l * 15;
#pragma unroll
                for (int gg = 0; gg < 5; gg++) {
                    const unsigned v = bp[3 * gg] | (bp[3 * gg + 1] << 8) | (bp[3 * gg + 2] << 16);
#pragma unroll
                    for (int j = 0; j < 8; j++) acc[gg * 8 + j] += __shfl_sync(0xffffffffu, ent, (v >> (3 * j)) & 7);
                }
            } else if (fd == 2) {
                const unsigned* wp = reinterpret_cast<const unsigned*>(rp + l * 20);
#pragma unroll
                for (int q = 0; q < 5; q++) {
                    const unsigned w = wp[q];
#pragma unroll
                    for (int j = 0; j < 8; j++) acc[q * 8 + j] += __shfl_sync(0xffffffffu, ent, (w >> (4 * j)) & 15);
                }
            } else {
                const uint2 u = *reinterpret_cast<const uint2*>(rp + l * 8);
#pragma unroll
                for (int q = 0; q < 8; q++) {
                    const unsigned d = tab[((q < 4 ? u.x : u.y) >> (8 * (q & 3))) & 255];
#pragma unroll
                    for (int j = 0; j < 5; j++) acc[q * 5 + j] += __shfl_sync(0xffffffffu, ent, (d >> (3 * j)) & 7);
                }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < 40; j++) red[g * QC_H + l * 40 + j] = acc[j];
    __syncthreads();
    for (int o = tid; o < QC_H; o += 256) part[((size_t)p * S + s) * QC_H + o] = (red[o] + red[QC_H + o]) + (red[2 * QC_H + o] + red[3 * QC_H + o]);
}

// ---- the second product of a call of 2 to 8 rows on the tensor cores, a place's pairs together (a verification's) -----------------------------------------------------------------------------------------
// A call's pairs of one place are the rows that chose its expert, one a row at most: at most 8. The block of a place's first pair (qc_group) takes them all and the other pairs' blocks leave; the place's downT
// codes are made fp16 tiles in shared memory once for all of them (format q: the integers, exact, the row's scale on the activations; two-bit: the row's four entries) and multiplied on the tensor cores (mma
// m16n8k16, fp16 in, fp32 sums; the place's pairs its 8 columns, those past R zero) by the pairs' activations made from qx_cb_up's sums as qx_cb_down makes them, rounded to fp16 with the scales: not
// qx_cb_down's sums bit for bit. All 640 neurons a block: one sum a pair. A format's tiles: qt_store_q / c2_store. Two-bit and format q places (QxMoeCb's tc_ok). (The first product's place-major form on the
// tensor cores was measured no faster than qx_cb_up's at a verification's sharing of places: qx_cb_up serves it.)
#define QX_GR 8

// The pairs of the call (`pairs` of them, their places in way) at pair p's place, in order, into rows: their count, where p is the first of them, else 0 (and 0 for a pair with no place). The block's first warp looks.
__device__ __forceinline__ int qc_group(const long long* way, int p, int pairs, int* rows) {
    __shared__ int cnt;
    if (threadIdx.x < 32) {
        const int lane = threadIdx.x;
        const long long pl = way[p];
        bool before = false;
        for (int q = lane; q < p; q += 32) before |= way[q] == pl;
        int c = 0;
        if (!__any_sync(0xffffffffu, before) && pl >= 0) {
            for (int q0 = p; q0 < pairs; q0 += 32) {
                const int q = q0 + lane;
                const bool m = q < pairs && way[q] == pl;
                const unsigned b = __ballot_sync(0xffffffffu, m);
                const int at = c + __popc(b & ((1u << lane) - 1u));
                if (m && at < QX_GR) rows[at] = q;
                c += __popc(b);
            }
        }
        if (lane == 0) cnt = min(c, QX_GR);
    }
    __syncthreads();
    return cnt;
}

#if defined(__HIP_PLATFORM_AMD__)
// (fp16 bits, the low 16, as a float; two floats as fp16 bits; the mma as gpu/warp_emu.h's bf16 one: shuffles and FMAs over the same fragments)
__device__ __forceinline__ float qh_f(uint32_t h) {
    const uint32_t s = (h & 0x8000u) << 16, e = (h >> 10) & 31u, m = h & 0x3FFu;
    if (e == 0) return __uint_as_float(s | __float_as_uint((float)m * 5.9604645e-8f));
    if (e == 31) return __uint_as_float(s | 0x7F800000u | (m << 13));
    return __uint_as_float(s | ((e + 112u) << 23) | (m << 13));
}
__device__ __forceinline__ uint32_t qh2(float lo, float hi) {
    const _Float16 a = (_Float16)lo, b = (_Float16)hi;
    return (uint32_t)__builtin_bit_cast(unsigned short, a) | (uint32_t)__builtin_bit_cast(unsigned short, b) << 16;
}
__device__ __forceinline__ void mma16816h(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    const int lane = GLYD_LANE(), t = lane & 3;
#pragma unroll
    for (int q = 0; q < 4; q++) {
        const int la = (lane & ~3) | q, lb = 8 * t + q;
        const uint32_t r0 = GLYD_SHFL(a[0], la), r8 = GLYD_SHFL(a[1], la), r0k = GLYD_SHFL(a[2], la), r8k = GLYD_SHFL(a[3], la);
        const uint32_t c0 = GLYD_SHFL(b0, lb), c0k = GLYD_SHFL(b1, lb), c1 = GLYD_SHFL(b0, lb + 4), c1k = GLYD_SHFL(b1, lb + 4);
        const uint32_t ra[4] = {r0, r0, r8, r8}, rk[4] = {r0k, r0k, r8k, r8k}, cb_[4] = {c0, c1, c0, c1}, ck[4] = {c0k, c1k, c0k, c1k};
#pragma unroll
        for (int i = 0; i < 4; i++)
            c[i] = fmaf(qh_f(rk[i] >> 16), qh_f(ck[i] >> 16), fmaf(qh_f(rk[i]), qh_f(ck[i]), fmaf(qh_f(ra[i] >> 16), qh_f(cb_[i] >> 16), fmaf(qh_f(ra[i]), qh_f(cb_[i]), c[i]))));
    }
}
#else
// two floats as fp16 bits, the first in the low half
__device__ __forceinline__ uint32_t qh2(float lo, float hi) {
    uint32_t r;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}
// c += a (16x16 f16, row) b (16x8 f16, col), fp32
__device__ __forceinline__ void mma16816h(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
#endif

// A two-bit row's four entries (bf16 pairs) as fp16 pairs, for c2_w2's selection of 16-bit entries.
__device__ __forceinline__ void qh_book(uint2 bk, uint32_t& r01, uint32_t& r23) {
    const float2 a = bf2(bk.x), b = bf2(bk.y);
    r01 = qh2(a.x, a.y);
    r23 = qh2(b.x, b.y);
}

// The 64 weights of a format q row's chunk (c2_load_tq's words) as fp16 integers before the row's scale (exact), into a tile of ROWLEN a row at `row`, chunks c0 .. c0 + 7 (XORed by the row, as c2_store's).
template <int ROWLEN>
__device__ __forceinline__ void qt_store_q(uint32_t base, int row, int c0, const C2Cw<5>& cw) {
#pragma unroll
    for (int c = 0; c < 8; c++) {
        const uint32_t hi = cw.w[c >> 1], lo = cw.w[(c >> 1) + 1];
        uint32_t p[4];
#pragma unroll
        for (int m = 0; m < 4; m++) {
            const int i = 8 * (c & 1) + 2 * m;
            p[m] = qh2(tq_v(__funnelshift_l(lo, hi, 2 * i) >> (32 - TQ_L)), tq_v(__funnelshift_l(lo, hi, 2 * i + 2) >> (32 - TQ_L)));
        }
        c2_sts16(base + (uint32_t)(row * ROWLEN + (((c0 + c) ^ (row & 7)) << 3)) * 2u, make_uint4(p[0], p[1], p[2], p[3]));
    }
}

// The second product of a place's R pairs (format B: 2 or 5; cd and bd its downT rows' codes and books), 256 outputs a block over all 640 neurons (10 steps of 64): first the pairs' activations of all 640
// neurons, made from gu by all the block's threads (bf16(silu(min(gate, limit)) clamp(up, -limit, limit)), qx_cb_down's), times their rows' scales (format q; times 2^12, so that they are fp16's well
// inside), as fp16 in sA [8][640] (chunks of 8 XORed by the row; rows past R zero); then a step's tile [neuron][output] of fp16 weights (format q: the integers; two-bit: the row's entries), a 64-output
// chunk a thread, read as W^T's fragments by ldmatrix.trans; warp w the outputs o0 + 32 w .. (two m-tiles): part [pair][2560], one sum a pair.
template <int B>
__device__ __forceinline__ void qt_down(const unsigned char* __restrict__ cd, const unsigned char* __restrict__ bd, const float* __restrict__ gu, float limit, const int* rows, int R, float* __restrict__ part, uint32_t sW,
                                        unsigned short* sA) {
    constexpr int BK = 64, BN = 256, NK = QC_I / BK, ROWB = QC_H / 4;  // (two bits a weight, either format)
    constexpr float UP = B == 5 ? 4096.f : 1.f;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, o0 = blockIdx.x * BN;
    const int brow = tid & 63, bh = tid >> 6;
    for (int i = tid; i < 8 * QC_I / 2; i += 256) {
        const int r = i / (QC_I / 2), n = 2 * (i % (QC_I / 2));
        float a[2] = {0.f, 0.f};
        if (r < R) {
            const float2 g2 = *reinterpret_cast<const float2*>(gu + (size_t)rows[r] * 2 * QC_I + n), u2 = *reinterpret_cast<const float2*>(gu + (size_t)rows[r] * 2 * QC_I + QC_I + n);
            const float gs[2] = {g2.x, g2.y}, us[2] = {u2.x, u2.y};
            float2 m2 = make_float2(1.f, 1.f);
            if constexpr (B == 5) {
                const float2 s2 = bf2(*reinterpret_cast<const uint32_t*>(bd + (size_t)n * 2));
                m2 = make_float2(s2.x * (TQ_K * UP), s2.y * (TQ_K * UP));
            }
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const float gg = fminf(gs[e], limit), uu = fminf(fmaxf(us[e], -limit), limit);
                a[e] = b2f(f2b(gg / (1.f + expf(-gg)) * uu)) * (e ? m2.y : m2.x);
            }
        }
        *reinterpret_cast<uint32_t*>(sA + r * QC_I + ((((n >> 3) ^ (r & 7)) << 3) | (n & 7))) = qh2(a[0], a[1]);
    }
    auto ldc = [&](int kt) {
        const unsigned char* row = cd + (size_t)(kt * BK + brow) * ROWB;
        if constexpr (B == 5) {
            return c2_load_tq(row, (o0 + 64 * bh) / 16, QC_H / 16);
        } else {
            return c2_load<B>(row + (size_t)(o0 + 64 * bh) / 4);
        }
    };
    // (two-bit: the row's four entries as fp16; format q: none, its scale is on the activations)
    auto ldb = [&](int kt) {
        uint2 r = make_uint2(0u, 0u);
        if constexpr (B == 2) qh_book(*reinterpret_cast<const uint2*>(bd + (size_t)(kt * BK + brow) * 8), r.x, r.y);
        return r;
    };
    float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
    C2Cw<B> cw = ldc(0);
    uint2 bk = ldb(0);
    const int mi = lane >> 3, r8 = lane & 7;
    const uint32_t sa = s_addr(sA);
    for (int kt = 0; kt < NK; kt++) {
        if constexpr (B == 5) {
            qt_store_q<BN>(sW, brow, 8 * bh, cw);
        } else {
            c2_store<BN, B>(sW, brow, 8 * bh, cw, bk.x, bk.y, nullptr);
        }
        if (kt + 1 < NK) {
            cw = ldc(kt + 1);
            bk = ldb(kt + 1);
        }
        __syncthreads();
#pragma unroll
        for (int ks = 0; ks < BK / 16; ks += 2) {
            // (b0, b1 of k-steps ks and ks + 1: the pairs' rows of sA at the step's chunks)
            uint32_t ba[4];
            ldsm4(ba, sa + (uint32_t)(r8 * QC_I + (((kt * 8 + ks * 2 + mi) ^ r8) << 3)) * 2);
#pragma unroll
            for (int s2 = 0; s2 < 2; s2++) {
                const int krow = (ks + s2) * 16 + (mi >> 1) * 8 + r8;
#pragma unroll
                for (int mt = 0; mt < 2; mt++) {
                    uint32_t aw[4];
                    ldsm4t(aw, sW + (uint32_t)(krow * BN + ((((32 * warp + 16 * mt) >> 3) + (mi & 1)) ^ r8) * 8) * 2);
                    mma16816h(acc[mt], aw, ba[2 * s2], ba[2 * s2 + 1]);
                }
            }
        }
        __syncthreads();
    }
    const int g = lane >> 2, t4 = lane & 3;
#pragma unroll
    for (int mt = 0; mt < 2; mt++) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
#pragma unroll
            for (int c = 0; c < 2; c++) {
                const int r = 2 * t4 + c;
                if (r < R) part[(size_t)rows[r] * QC_H + o0 + 32 * warp + 16 * mt + g + 8 * h] = acc[mt][2 * h + c] * (1.f / UP);
            }
        }
    }
}

// part [pairs, 1, 2560] for the pairs of a call of 2 to 8 rows (grid 10 x pairs, 256 threads): their places' second products over all 640 neurons, a place's pairs together (gu: qx_cb_up's; format 0 and 4 places).
__device__ __forceinline__ void qx_cb_downt_b(const unsigned char* codes, unsigned long long cstride, const unsigned char* books, unsigned long long bstride, const long long* way, const float* gu, float* part, float limit, int mixed, int pairs) {
    __shared__ int rows[QX_GR];
    const int R = qc_group(way, blockIdx.y, pairs, rows);
    if (R == 0) return;
    __shared__ __align__(128) unsigned short sW[64 * 256];
    __shared__ __align__(128) unsigned short sA[8 * QC_I];
    const long long pl = way[blockIdx.y];
    const unsigned char* bplace = books + (size_t)pl * bstride;
    const int fd = mixed ? bplace[bstride - 1] >> 4 : 0;
    const unsigned char* cd = codes + (size_t)pl * cstride;
    const uint32_t w = s_addr(sW);
    if (fd == 0) qt_down<2>(cd, bplace, gu, limit, rows, R, part, w, sA);
    else qt_down<5>(cd, bplace, gu, limit, rows, R, part, w, sA);
}

// The experts' products as kernels: the places' two buffers as arguments, or their addresses read from pp (the GPU's memory) by the launches of a graph, which keep them while the places are made again.
extern "C" __global__ void __launch_bounds__(256, 6) qx_cb_up(const unsigned char* codes, unsigned long long cstride, const unsigned char* books, unsigned long long bstride, const long long* way, const float* xr, float* gu, int k, int mixed) {
    qx_cb_up_b(codes, cstride, books, bstride, way, xr, gu, k, mixed);
}
extern "C" __global__ void __launch_bounds__(256, 6) qx_cb_upi(const unsigned long long* pp, unsigned long long cstride, unsigned long long bstride, const long long* way, const float* xr, float* gu, int k, int mixed) {
    qx_cb_up_b(reinterpret_cast<const unsigned char*>(pp[0]), cstride, reinterpret_cast<const unsigned char*>(pp[1]), bstride, way, xr, gu, k, mixed);
}
extern "C" __global__ void __launch_bounds__(256) qx_cb_down(const unsigned char* codes, unsigned long long cstride, const unsigned char* books, unsigned long long bstride, const long long* way, const float* gu, float* part, int S, float limit, int mixed) {
    qx_cb_down_b(codes, cstride, books, bstride, way, gu, part, S, limit, mixed);
}
extern "C" __global__ void __launch_bounds__(256) qx_cb_downi(const unsigned long long* pp, unsigned long long cstride, unsigned long long bstride, const long long* way, const float* gu, float* part, int S, float limit, int mixed) {
    qx_cb_down_b(reinterpret_cast<const unsigned char*>(pp[0]), cstride, reinterpret_cast<const unsigned char*>(pp[1]), bstride, way, gu, part, S, limit, mixed);
}
extern "C" __global__ void __launch_bounds__(256) qx_cb_downt(const unsigned char* codes, unsigned long long cstride, const unsigned char* books, unsigned long long bstride, const long long* way, const float* gu, float* part, float limit, int mixed, int pairs) {
    qx_cb_downt_b(codes, cstride, books, bstride, way, gu, part, limit, mixed, pairs);
}
extern "C" __global__ void __launch_bounds__(256) qx_cb_downti(const unsigned long long* pp, unsigned long long cstride, unsigned long long bstride, const long long* way, const float* gu, float* part, float limit, int mixed, int pairs) {
    qx_cb_downt_b(reinterpret_cast<const unsigned char*>(pp[0]), cstride, reinterpret_cast<const unsigned char*>(pp[1]), bstride, way, gu, part, limit, mixed, pairs);
}

// y [n, 2560] bf16 = R^T (extra[t] + sum over the token's k pairs c of wts[t k + c] times the sum of their S runs), R^T v = s 512^-0.5 (I_5 (x) H_512) v, one rounding. extra: fp32 [n, 2560] (the host's pairs, rotated
// basis) or 0; a pair whose skip entry is negative is left out (skip int64 [pairs] or 0). A block a token's 512 values of one transform (grid n x 5), 512 threads.
extern "C" __global__ void __launch_bounds__(512) qx_cb_sum_rot(const float* part, const float* wts, const float* extra, const long long* skip, const signed char* sg, bf16* y, int k, int S) {
    __shared__ float v[512];
    const int t = blockIdx.x, o = blockIdx.y * 512 + threadIdx.x;
    float acc = extra ? extra[(size_t)t * QC_H + o] : 0.f;
    for (int c = 0; c < k; c++) {
        const size_t p = (size_t)t * k + c;
        if (skip && skip[p] < 0) continue;
        float a = 0.f;
        for (int s = 0; s < S; s++) a += part[(p * S + s) * QC_H + o];
        acc = fmaf(wts[p], a, acc);
    }
    v[threadIdx.x] = acc;
    __syncthreads();
    qc_wht512(v);
    y[(size_t)t * QC_H + o] = f2b(v[threadIdx.x] * (float)sg[o] * 0.044194173824159216f);
}

// A value the host writes, read from its memory anew each time (a volatile load: no cached line, and never taken out of a loop as __ldcv's asm may be)
__device__ __forceinline__ unsigned qx_host_u32(const unsigned* p) {
    return *(const volatile unsigned*)p;
}
__device__ __forceinline__ float qx_host_f32(const float* p) {
    return *(const volatile float*)p;
}

// The decode step's sum of the experts and what follows it, in one launch: qx_cb_sum_rot's output (the pairs' partial sums, the host's, R^T), the shared expert's output added (bf16(y + sh), as qx_add_bf16), and the
// streams' update (qx_hc_post: x += bf16(y inj)). Block (row, 512 outputs), 512 threads. Where `hword` is given, the launch was queued before the host's sums were made: `extra` is then the host's memory, read
// once its word hword[0] reads the front's count (`gen`; hword[1]: whether there are sums, 0: none), looked at for about 2 s at most (a host that failed leaves the GPU going on).
extern "C" __global__ void __launch_bounds__(512) qx_cb_back(const float* part, const float* wts, const float* extra, const long long* skip, const signed char* sg, const bf16* sh, const float* inj, bf16* x, int k, int S, const unsigned* hword, const unsigned* gen) {
    __shared__ float v[512];
    __shared__ unsigned has;
    const int t = blockIdx.x, o = blockIdx.y * 512 + threadIdx.x;
    if (hword) {
        if (threadIdx.x == 0) {
            const unsigned want = *gen;
            for (int i = 0; i < (1 << 21) && qx_host_u32(hword) != want; i++) {
            }
            __threadfence_system();
            has = qx_host_u32(hword + 1);
        }
        __syncthreads();
    }
    float acc = !extra || (hword && !has) ? 0.f : hword ? qx_host_f32(extra + (size_t)t * QC_H + o) : extra[(size_t)t * QC_H + o];
    for (int c = 0; c < k; c++) {
        const size_t p = (size_t)t * k + c;
        if (skip && skip[p] < 0) continue;
        float a = 0.f;
        for (int s = 0; s < S; s++) a += part[(p * S + s) * QC_H + o];
        acc = fmaf(wts[p], a, acc);
    }
    v[threadIdx.x] = acc;
    __syncthreads();
    qc_wht512(v);
    const float y1 = qx_r(v[threadIdx.x] * (float)sg[o] * 0.044194173824159216f);
    const float y = qx_r(y1 + b2f(sh[(size_t)t * QC_H + o]));
#pragma unroll
    for (int q = 0; q < QX_HC; q++) {
        const size_t i = ((size_t)t * QX_HC + q) * QC_H + o;
        x[i] = f2b(b2f(x[i]) + qx_r(y * inj[t * QX_HC + q]));
    }
}

// ---- small products of int8 rows for a few rows of input (a decode step's, a verification's: n at most 8): the Gated Residual's bottleneck and the shared expert ---------------------------------------------
// Each matrix's rows are int8 with an fp32 scale each (w = q * scale), read 16 bytes a lane at a time; the inputs are read beside them (they stay in the caches); a kernel for each count of rows NT (1, 2, 4, 8: the
// smallest at least n, rows past n left out). The reference's roundings are kept (each product's sum rounded to bf16, then the elementwise ops).

// 16 int8 of an int4 as floats
__device__ __forceinline__ void qx_i8x16(const int4 v, float* f) {
    const int w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 4; i++) {
        f[4 * i] = (float)(signed char)(w[i] & 0xff);
        f[4 * i + 1] = (float)(signed char)((w[i] >> 8) & 0xff);
        f[4 * i + 2] = (float)(signed char)((w[i] >> 16) & 0xff);
        f[4 * i + 3] = (float)(signed char)(w[i] >> 24);
    }
}

// 8 bf16 of an int4 as floats
__device__ __forceinline__ void qx_bf8(const int4 v, float* f) {
    const unsigned w[4] = {(unsigned)v.x, (unsigned)v.y, (unsigned)v.z, (unsigned)v.w};
#pragma unroll
    for (int i = 0; i < 4; i++) {
        f[2 * i] = __uint_as_float(w[i] << 16);
        f[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
    }
}

// 16 bf16 at p (16-byte aligned) as floats
__device__ __forceinline__ void qx_bf16v(const bf16* p, float* f) {
    qx_bf8(__ldg(reinterpret_cast<const int4*>(p)), f);
    qx_bf8(__ldg(reinterpret_cast<const int4*>(p) + 1), f + 8);
}

// The sum over the 8 lanes of a group (lanes 8 g .. 8 g + 7 of a warp), on its lanes.
__device__ __forceinline__ float qx_sum8(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v;
}

// 8 bf16 at p (16-byte aligned) as floats
__device__ __forceinline__ void qx_bf8v(const bf16* p, float* f) { qx_bf8(__ldg(reinterpret_cast<const int4*>(p)), f); }

// 8 int8 of an int2 as floats
__device__ __forceinline__ void qx_i8x8(const int2 v, float* f) {
    const int w[2] = {v.x, v.y};
#pragma unroll
    for (int i = 0; i < 2; i++) {
        f[4 * i] = (float)(signed char)(w[i] & 0xff);
        f[4 * i + 1] = (float)(signed char)((w[i] >> 8) & 0xff);
        f[4 * i + 2] = (float)(signed char)((w[i] >> 16) & 0xff);
        f[4 * i + 3] = (float)(signed char)(w[i] >> 24);
    }
}

// qx_gr1: for the n rows of x [n, 4, D] (D a multiple of 8): rs [n, 4] (each stream's 1 / rms; block 0 writes it), lo [n, R] bf16 = bf16(silu(bf16(down . xn) / 4)) for the block's RB rows of down (int8 [R, 4D], scales [R];
// blocks 0 .. R / RB - 1, R a multiple of RB), and in the last 4 / RB blocks, where `injw` (bf16 [4, 4D]) is given, inj [n, 4] = 2 bf16(sigmoid(bf16(bf16(inject . xn) / 4))). xn = x rs (1 + w): one pass over x, the 64 threads of a stream going over its
// columns 8 at a time, each row's product summed stream by stream with x (1 + w) and scaled by the stream's 1 / rms at the end (the squares summed in the same pass), so no thread waits for the norms. 256 threads.
#define QG_RB 2  // rows of down a block of qx_gr1 (the inject's 4 rows: the last 4 / QG_RB blocks; few rows a block, so that a block's bytes are few: an SM reads them at a rate its loads in flight set)
template <int NT, int RB>
__device__ __forceinline__ void qx_gr1_t(const bf16* x, const bf16* w, const signed char* dq, const float* ds, const bf16* injw, float* rs, bf16* lo, float* inj, int n, int D, int R, float eps, const bf16* po, const float* pinj) {
    __shared__ float red[8][RB][NT];
    __shared__ float sq[8][NT];
    const int tid = threadIdx.x, lane = tid & 31, wp = tid >> 5, W = QX_HC * D;
    const bool injb = injw && (int)blockIdx.x >= R / RB;
    const int r0 = injb ? ((int)blockIdx.x - R / RB) * RB : blockIdx.x * RB, nr = injb ? min(RB, QX_HC - r0) : min(RB, R - r0);
    const int st = tid >> 6, j = tid & 63;  // the thread's stream, and its place among the stream's 64
    float acc[RB][NT], ss[NT];
#pragma unroll
    for (int t = 0; t < NT; t++) {
        ss[t] = 0.f;
#pragma unroll
        for (int r = 0; r < RB; r++) acc[r][t] = 0.f;
    }
#pragma unroll 5
    for (int c = j * 8; c < D; c += 512) {
        const int col = st * D + c;
        float g[8];
        qx_bf8v(w + col, g);
        float wv[RB][8];
#pragma unroll
        for (int r = 0; r < RB; r++) {
            if (r < nr) {
                if (injb) {
                    qx_bf8v(injw + (size_t)(r0 + r) * W + col, wv[r]);
                } else {
                    qx_i8x8(__ldg(reinterpret_cast<const int2*>(dq + (size_t)(r0 + r) * W + col)), wv[r]);
                }
            }
        }
#pragma unroll
        for (int t = 0; t < NT; t++) {
            if (t < n) {
                float xv[8];
                qx_bf8v(x + (size_t)t * W + col, xv);
                if (po) {
                    // (the update the mixer's half left pending, as qx_hc_post makes it: x + bf16(o inj))
                    float ov[8];
                    qx_bf8v(po + (size_t)t * D + c, ov);
                    const float pj = pinj[t * QX_HC + st];
#pragma unroll
                    for (int i = 0; i < 8; i++) xv[i] = qx_r(xv[i] + qx_r(ov[i] * pj));
                }
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    ss[t] = fmaf(xv[i], xv[i], ss[t]);
                    xv[i] *= 1.f + g[i];
                }
#pragma unroll
                for (int r = 0; r < RB; r++)
#pragma unroll
                    for (int i = 0; i < 8; i++) acc[r][t] = fmaf(wv[r][i], xv[i], acc[r][t]);
            }
        }
    }
    // a warp's sums by shuffles (each warp is of one stream), then the streams' through shared memory, each scaled by its 1 / rms
#pragma unroll
    for (int t = 0; t < NT; t++) {
        const float q = warp_sum(ss[t]);
        if (lane == 0) sq[wp][t] = q;
#pragma unroll
        for (int r = 0; r < RB; r++) {
            const float v = warp_sum(acc[r][t]);
            if (lane == 0) red[wp][r][t] = v;
        }
    }
    __syncthreads();
    if (blockIdx.x == 0 && tid < n * QX_HC) {
        const int t = tid / QX_HC, q = tid % QX_HC;
        rs[tid] = rsqrtf((sq[2 * q][t] + sq[2 * q + 1][t]) / D + eps);
    }
    if (tid < nr * n) {
        const int r = tid / n, t = tid % n;
        float v = 0.f;
#pragma unroll
        for (int q = 0; q < QX_HC; q++) v = fmaf(rsqrtf((sq[2 * q][t] + sq[2 * q + 1][t]) / D + eps), red[2 * q][r][t] + red[2 * q + 1][r][t], v);
        if (injb) {
            inj[t * QX_HC + r0 + r] = 2.f * qx_r(qx_sig(qx_r(qx_r(v) * 0.25f)));
        } else {
            const float d = qx_r(qx_r(v * ds[r0 + r]) * 0.25f);
            lo[(size_t)t * R + r0 + r] = f2b(d * qx_sig(d));
        }
    }
}

// qx_gr1 for 5 to 8 rows of x, split by columns (a verification's: the row-sliced kernel reads all of x again in each of its blocks, 26 MB of the L2 for 8 rows): a block a slice of 256 columns (of one stream)
// by a group of 32 of the R + 4 rows (down's, then the inject's), grid (4 D / 256) x ceil((R + 4) / 32); a lane 8 columns, a warp 4 rows; x (with the pending update, times 1 + w) made once a thread for its
// columns and all rows of x. Each block's sums of its slice (and the slice's squares of x, a copy a group) into part / sq; the last block of a group to finish adds its rows' slices in order, a stream at a time,
// scaled by the stream's 1 / rms, into lo and inj (and rs, group 0's), and sets the group's counter back. 256 threads.
#define QG_SW 256
template <int NT>
__device__ __forceinline__ void qx_gr1s_t(const bf16* x, const bf16* w, const signed char* dq, const float* ds, const bf16* injw, float* rs, bf16* lo, float* inj, int n, int D, int R, float eps, const bf16* po, const float* pinj, float* part, float* sqp, unsigned* cnt) {
    __shared__ float tr[8][32][33];
    __shared__ float sqs[8][NT];
    __shared__ bool last;
    const int tid = threadIdx.x, lane = tid & 31, wp = tid >> 5, W = QX_HC * D;
    const int sl = blockIdx.x, ns = gridDim.x, g = blockIdx.y, RT = R + QX_HC;
    const int col = sl * QG_SW + lane * 8, st = col / D, c = col - st * D;
    // the lane's 8 columns of each row of x, made as qx_gr1 makes them, and their squares
    float xv[NT][8], ss[NT];
    float gg[8];
    qx_bf8v(w + col, gg);
#pragma unroll
    for (int t = 0; t < NT; t++) {
        ss[t] = 0.f;
        if (t < n) {
            qx_bf8v(x + (size_t)t * W + col, xv[t]);
            if (po) {
                float ov[8];
                qx_bf8v(po + (size_t)t * D + c, ov);
                const float pj = pinj[t * QX_HC + st];
#pragma unroll
                for (int i = 0; i < 8; i++) xv[t][i] = qx_r(xv[t][i] + qx_r(ov[i] * pj));
            }
#pragma unroll
            for (int i = 0; i < 8; i++) {
                ss[t] = fmaf(xv[t][i], xv[t][i], ss[t]);
                xv[t][i] *= 1.f + gg[i];
            }
        } else {
#pragma unroll
            for (int i = 0; i < 8; i++) xv[t][i] = 0.f;
        }
    }
    // the warp's 4 rows: each lane's 8 columns' sums, through shared memory to a lane a (row, row of x)
#pragma unroll
    for (int rr = 0; rr < 4; rr++) {
        const int row = g * 32 + wp * 4 + rr;
        float wv[8];
        if (row < R) {
            qx_i8x8(__ldg(reinterpret_cast<const int2*>(dq + (size_t)row * W + col)), wv);
        } else if (row < RT && injw) {
            qx_bf8v(injw + (size_t)(row - R) * W + col, wv);
        } else {
#pragma unroll
            for (int i = 0; i < 8; i++) wv[i] = 0.f;
        }
#pragma unroll
        for (int t = 0; t < NT; t++) {
            float a = 0.f;
#pragma unroll
            for (int i = 0; i < 8; i++) a = fmaf(wv[i], xv[t][i], a);
            tr[wp][rr * 8 + t][lane] = a;
        }
    }
#pragma unroll
    for (int t = 0; t < NT; t++) {
        const float q = warp_sum(ss[t]);
        if (lane == 0) sqs[wp][t] = q;
    }
    __syncwarp();
    {
        // lane v: (row rr = v / 8, row of x t = v % 8) of the warp's: the 32 lanes' sums in order
        const int rr = lane >> 3, t = lane & 7;
        float a = 0.f;
        for (int l = 0; l < 32; l++) a += tr[wp][lane][l];
        const int row = g * 32 + wp * 4 + rr;
        if (t < NT) part[((size_t)sl * gridDim.y * 32 + (size_t)(g * 32 + wp * 4 + rr)) * NT + t] = a;
        (void)row;
    }
    __syncthreads();
    if (tid < NT) sqp[((size_t)g * ns + sl) * NT + tid] = sqs[0][tid];  // (every warp has the slice's squares: the warps' columns are the same)
    __threadfence();
    __syncthreads();
    if (tid == 0) last = atomicAdd(cnt + g, 1u) == (unsigned)ns - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    const int per = D / QG_SW;  // slices a stream
    if (tid < 32 * n) {
        const int rl = tid / n, t = tid % n, row = g * 32 + rl;
        float v = 0.f;
#pragma unroll
        for (int q = 0; q < QX_HC; q++) {
            float sq = 0.f, a = 0.f;
            for (int k2 = 0; k2 < per; k2++) {
                const int s2 = q * per + k2;
                sq += sqp[((size_t)g * ns + s2) * NT + t];
                a += part[((size_t)s2 * gridDim.y * 32 + row) * NT + t];
            }
            const float r = rsqrtf(sq / D + eps);
            if (g == 0 && rl == 0) rs[t * QX_HC + q] = r;
            v = fmaf(r, a, v);
        }
        if (row < R) {
            const float d = qx_r(qx_r(v * ds[row]) * 0.25f);
            lo[(size_t)t * R + row] = f2b(d * qx_sig(d));
        } else if (row < RT && injw) {
            inj[t * QX_HC + (row - R)] = 2.f * qx_r(qx_sig(qx_r(qx_r(v) * 0.25f)));
        }
    }
    if (tid == 0) cnt[g] = 0;
}

extern "C" __global__ void __launch_bounds__(256) qx_gr1s_8(const bf16* x, const bf16* w, const signed char* dq, const float* ds, const bf16* injw, float* rs, bf16* lo, float* inj, int n, int D, int R, float eps, const bf16* po, const float* pinj, float* part, float* sqp, unsigned* cnt) {
    qx_gr1s_t<8>(x, w, dq, ds, injw, rs, lo, inj, n, D, R, eps, po, pinj, part, sqp, cnt);
}

// qx_gr1 by streams: a block a stream q's D columns by a group of RB of the R + 4 rows (down's, then the inject's), grid 4 x ceil((R + 4) / RB), D / 8 threads (D / 8 a multiple of 32, at most 320): a lane 8
// columns, so each block makes x (with the pending update, times 1 + w) for its stream only (qx_gr1's blocks make all 4 streams of x again each, its time mostly that). Each block's sums into part [4, R + 4, NT]
// and its stream's 1 / rms into sqp [groups, 4, NT] (and rs, group 0's); the last of a group's 4 blocks to finish adds the streams' in order, each scaled by its 1 / rms, into lo and inj, and sets the group's
// counter back.
template <int NT, int RB>
__device__ __forceinline__ void qx_gr1q_t(const bf16* x, const bf16* w, const signed char* dq, const float* ds, const bf16* injw, float* rs, bf16* lo, float* inj, int n, int D, int R, float eps, const bf16* po, const float* pinj, float* part, float* sqp, unsigned* cnt) {
    __shared__ float red[16][RB][NT];
    __shared__ float sq[16][NT];
    __shared__ bool last;
    const int tid = threadIdx.x, lane = tid & 31, wp = tid >> 5, nw = blockDim.x >> 5, W = QX_HC * D;
    const int q = blockIdx.x, g = blockIdx.y, RT = R + QX_HC, r0 = g * RB;
    float acc[RB][NT], ss[NT];
#pragma unroll
    for (int t = 0; t < NT; t++) {
        ss[t] = 0.f;
#pragma unroll
        for (int r = 0; r < RB; r++) acc[r][t] = 0.f;
    }
    for (int c = tid * 8; c < D; c += blockDim.x * 8) {
        const int col = q * D + c;
        // the lane's 8 columns of each row of x first, then the rows of weights one by one (few registers: two blocks an SM, the grid in one wave)
        float gv[8], xv[NT][8];
        qx_bf8v(w + col, gv);
#pragma unroll
        for (int t = 0; t < NT; t++) {
            if (t < n) {
                qx_bf8v(x + (size_t)t * W + col, xv[t]);
                if (po) {
                    float ov[8];
                    qx_bf8v(po + (size_t)t * D + c, ov);
                    const float pj = pinj[t * QX_HC + q];
#pragma unroll
                    for (int i = 0; i < 8; i++) xv[t][i] = qx_r(xv[t][i] + qx_r(ov[i] * pj));
                }
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    ss[t] = fmaf(xv[t][i], xv[t][i], ss[t]);
                    xv[t][i] *= 1.f + gv[i];
                }
            }
        }
#pragma unroll
        for (int r = 0; r < RB; r++) {
            const int row = r0 + r;
            float wv[8];
            if (row < R) {
                qx_i8x8(__ldg(reinterpret_cast<const int2*>(dq + (size_t)row * W + col)), wv);
            } else if (row < RT && injw) {
                qx_bf8v(injw + (size_t)(row - R) * W + col, wv);
            } else {
                continue;
            }
#pragma unroll
            for (int t = 0; t < NT; t++)
                if (t < n)
#pragma unroll
                    for (int i = 0; i < 8; i++) acc[r][t] = fmaf(wv[i], xv[t][i], acc[r][t]);
        }
    }
#pragma unroll
    for (int t = 0; t < NT; t++) {
        const float v = warp_sum(ss[t]);
        if (lane == 0) sq[wp][t] = v;
#pragma unroll
        for (int r = 0; r < RB; r++) {
            const float a = warp_sum(acc[r][t]);
            if (lane == 0) red[wp][r][t] = a;
        }
    }
    __syncthreads();
    if (tid < RB * n) {
        const int r = tid / n, t = tid % n;
        float s2 = 0.f, a = 0.f;
        for (int k = 0; k < nw; k++) {
            s2 += sq[k][t];
            a += red[k][r][t];
        }
        const float rq = rsqrtf(s2 / D + eps);
        if (r == 0) sqp[((size_t)g * QX_HC + q) * NT + t] = rq;
        if (g == 0 && r == 0) rs[t * QX_HC + q] = rq;
        if (r0 + r < RT) part[((size_t)q * RT + r0 + r) * NT + t] = a;
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) last = atomicAdd(cnt + g, 1u) == QX_HC - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    if (tid < RB * n) {
        const int r = tid / n, t = tid % n, row = r0 + r;
        float v = 0.f;
#pragma unroll
        for (int k = 0; k < QX_HC; k++) v = fmaf(sqp[((size_t)g * QX_HC + k) * NT + t], part[((size_t)k * RT + row) * NT + t], v);
        if (row < R) {
            const float d = qx_r(qx_r(v * ds[row]) * 0.25f);
            lo[(size_t)t * R + row] = f2b(d * qx_sig(d));
        } else if (row < RT && injw) {
            inj[t * QX_HC + (row - R)] = 2.f * qx_r(qx_sig(qx_r(qx_r(v) * 0.25f)));
        }
    }
    if (tid == 0) cnt[g] = 0;
}
#define QG_RQ 9  // rows a block of qx_gr1q (the 324 rows of 320 + 4 in 36 groups: 144 blocks, in one wave of two an SM on 80 SMs)

// qx_gr2: h [n, D] = bf16(mean over the streams s of bf16(bf16(sigmoid(bf16(up_{s D + c} . lo))) xn[s][c])) for the block's 8 columns c (D / 8 blocks), up's rows int8 [4D, R] (R a multiple of 16, at most 384) with scales
// us [4D]: 8 lanes a row (the block's 32: 4 streams of 8 columns), lo [n, R] read beside them; xn made again from x, rs and w. 256 threads.
template <int NT>
__device__ __forceinline__ void qx_gr2_t(bf16* x, const bf16* w, const float* rs, const bf16* lo, const signed char* uq, const float* us, bf16* h, int n, int D, int R, const bf16* po, const float* pinj) {
    __shared__ float sg[QX_HC][8][NT];
    const int tid = threadIdx.x, W = QX_HC * D, c0 = blockIdx.x * 8;
    const int g = tid >> 3, gl = tid & 7, s = g >> 3, cc = g & 7;
    const signed char* row = uq + (size_t)(s * D + c0 + cc) * R;
    // (what the end needs, read before the products: its loads in flight beside theirs, not after them)
    const bool ep = tid < n * 8;
    const int et = tid >> 3, ec = c0 + (tid & 7);
    float xe[QX_HC], ge[QX_HC], re[QX_HC], pj[QX_HC], pe = 0.f;
    if (ep) {
#pragma unroll
        for (int q = 0; q < QX_HC; q++) {
            xe[q] = b2f(x[(size_t)et * W + q * D + ec]);
            ge[q] = b2f(w[q * D + ec]);
            re[q] = rs[et * QX_HC + q];
            pj[q] = po ? pinj[et * QX_HC + q] : 0.f;
        }
        if (po) pe = b2f(po[(size_t)et * D + ec]);
    }
    float a[NT];
#pragma unroll
    for (int t = 0; t < NT; t++) a[t] = 0.f;
    for (int k0 = gl * 16; k0 < R; k0 += 128) {
        float wv[16];
        qx_i8x16(__ldg(reinterpret_cast<const int4*>(row + k0)), wv);
#pragma unroll
        for (int t = 0; t < NT; t++) {
            if (t < n) {
                float lv[16];
                qx_bf16v(lo + (size_t)t * R + k0, lv);
#pragma unroll
                for (int i = 0; i < 16; i++) a[t] = fmaf(wv[i], lv[i], a[t]);
            }
        }
    }
    const float sc = us[s * D + c0 + cc];
#pragma unroll
    for (int t = 0; t < NT; t++) {
        const float v = qx_sum8(a[t]);
        if (gl == 0) sg[s][cc][t] = qx_r(qx_sig(qx_r(v * sc)));
    }
    __syncthreads();
    if (ep) {
        const int t = et, c = ec;
        float m = 0.f;
#pragma unroll
        for (int q = 0; q < QX_HC; q++) {
            const int col = q * D + c;
            float xv = xe[q];
            if (po) {
                // (the pending update made here and written back: the block's columns are its own)
                xv = qx_r(xv + qx_r(pe * pj[q]));
                x[(size_t)t * W + col] = f2b(xv);
            }
            const float xn = qx_r(xv * re[q] * (1.f + ge[q]));
            m += qx_r(sg[q][tid & 7][t] * xn);
        }
        h[(size_t)t * D + c] = f2b(m * 0.25f);
    }
}

// qx_sh1: act [n, I] bf16 = bf16(bf16(silu(bf16(gate . x))) bf16(up . x)), a warp a neuron (its gate row j and up row I + j of q [2 I, H] int8, scales sc [2 I]; 8 a block, blocks 0 .. I / 8 - 1); the last block's first warp:
// sg [n] = bf16(sigmoid(bf16(gw . x))) (the shared expert's gate, bf16 [H]). H a multiple of 512. 256 threads.
template <int NT>
__device__ __forceinline__ void qx_sh1_t(const bf16* x, const signed char* q, const float* sc, const bf16* gw, bf16* act, float* sg, int n, int H, int I) {
    const int tid = threadIdx.x, lane = tid & 31, wp = tid >> 5;
    const bool gb = blockIdx.x == gridDim.x - 1;
    const int j = blockIdx.x * 8 + wp;
    if (gb ? wp != 0 : j >= I) return;
    float ag[NT], au[NT];
#pragma unroll
    for (int t = 0; t < NT; t++) ag[t] = au[t] = 0.f;
    for (int c0 = lane * 16; c0 < H; c0 += 512) {
        float gv[16], uv[16];
        if (gb) {
            qx_bf16v(gw + c0, gv);
        } else {
            qx_i8x16(__ldg(reinterpret_cast<const int4*>(q + (size_t)j * H + c0)), gv);
            qx_i8x16(__ldg(reinterpret_cast<const int4*>(q + (size_t)(I + j) * H + c0)), uv);
        }
#pragma unroll
        for (int t = 0; t < NT; t++) {
            if (t < n) {
                float xv[16];
                qx_bf16v(x + (size_t)t * H + c0, xv);
#pragma unroll
                for (int i = 0; i < 16; i++) {
                    ag[t] = fmaf(gv[i], xv[i], ag[t]);
                    if (!gb) au[t] = fmaf(uv[i], xv[i], au[t]);
                }
            }
        }
    }
#pragma unroll
    for (int t = 0; t < NT; t++) {
        const float g = warp_sum(ag[t]);
        if (gb) {
            if (lane == 0 && t < n) sg[t] = qx_r(qx_sig(qx_r(g)));
            continue;
        }
        const float u = warp_sum(au[t]);
        if (lane == 0 && t < n) {
            const float gs = qx_r(g * sc[j]), us = qx_r(u * sc[I + j]);
            act[(size_t)t * I + j] = f2b(qx_r(gs * qx_sig(gs)) * us);
        }
    }
}

// qx_sh2: sh [n, H] bf16 = bf16(sg[t] bf16(down . act)) (down's int8 rows [H, I], I a multiple of 128, scales [H]), 8 lanes a row, 32 rows a block. 256 threads.
template <int NT>
__device__ __forceinline__ void qx_sh2_t(const bf16* act, const signed char* q, const float* sc, const float* sg, bf16* sh, int n, int H, int I) {
    const int tid = threadIdx.x, gl = tid & 7, o = blockIdx.x * 32 + (tid >> 3);
    if (o >= H) return;
    const signed char* row = q + (size_t)o * I;
    float a[NT];
#pragma unroll
    for (int t = 0; t < NT; t++) a[t] = 0.f;
    for (int k0 = gl * 16; k0 < I; k0 += 128) {
        float wv[16];
        qx_i8x16(__ldg(reinterpret_cast<const int4*>(row + k0)), wv);
#pragma unroll
        for (int t = 0; t < NT; t++) {
            if (t < n) {
                float av[16];
                qx_bf16v(act + (size_t)t * I + k0, av);
#pragma unroll
                for (int i = 0; i < 16; i++) a[t] = fmaf(wv[i], av[i], a[t]);
            }
        }
    }
    const float s = sc[o];
#pragma unroll
    for (int t = 0; t < NT; t++) {
        const float v = qx_sum8(a[t]);
        if (gl == 0 && t < n) sh[(size_t)t * H + o] = f2b(sg[t] * qx_r(v * s));
    }
}

#define QX_SMALL(NT)                                                                                                                                                                              \
    extern "C" __global__ void __launch_bounds__(256) qx_gr1_##NT(const bf16* x, const bf16* w, const signed char* dq, const float* ds, const bf16* injw, float* rs, bf16* lo, float* inj, int n, int D, int R, float eps, const bf16* po, const float* pinj) { \
        qx_gr1_t<NT, QG_RB>(x, w, dq, ds, injw, rs, lo, inj, n, D, R, eps, po, pinj);                                                                                                                              \
    }                                                                                                                                                                                             \
    extern "C" __global__ void __launch_bounds__(256) qx_gr2_##NT(bf16* x, const bf16* w, const float* rs, const bf16* lo, const signed char* uq, const float* us, bf16* h, int n, int D, int R, const bf16* po, const float* pinj) { \
        qx_gr2_t<NT>(x, w, rs, lo, uq, us, h, n, D, R, po, pinj);                                                                                                                                           \
    }                                                                                                                                                                                             \
    extern "C" __global__ void __launch_bounds__(256) qx_sh1_##NT(const bf16* x, const signed char* q, const float* sc, const bf16* gw, bf16* act, float* sg, int n, int H, int I) {               \
        qx_sh1_t<NT>(x, q, sc, gw, act, sg, n, H, I);                                                                                                                                             \
    }                                                                                                                                                                                             \
    extern "C" __global__ void __launch_bounds__(256) qx_sh2_##NT(const bf16* act, const signed char* q, const float* sc, const float* sg, bf16* sh, int n, int H, int I) {                        \
        qx_sh2_t<NT>(act, q, sc, sg, sh, n, H, I);                                                                                                                                                \
    }
QX_SMALL(1)
QX_SMALL(2)
QX_SMALL(4)
QX_SMALL(8)
// (qx_gr1q for 1, 2 and 4 rows of x, D / 8 at most 320 threads: 8 rows' sums do not fit two blocks an SM)
#define QX_GR1Q(NT)                                                                                                                                                                               \
    extern "C" __global__ void __launch_bounds__(320, 2) qx_gr1q_##NT(const bf16* x, const bf16* w, const signed char* dq, const float* ds, const bf16* injw, float* rs, bf16* lo, float* inj, int n, int D, int R, float eps, const bf16* po, const float* pinj, float* part, float* sqp, unsigned* cnt) { \
        qx_gr1q_t<NT, QG_RQ>(x, w, dq, ds, injw, rs, lo, inj, n, D, R, eps, po, pinj, part, sqp, cnt);                                                                                                          \
    }
QX_GR1Q(1)
QX_GR1Q(2)
QX_GR1Q(4)

// out [orows, ocols] bf16 = q [rows, cols] int8 times scale [rows], zeros past them (the Gated Residuals' int8 rows made bf16 for a prompt chunk's dense product, padded). A thread four values (cols, ocols multiples of 4).
extern "C" __global__ void qx_i8_bf16(const signed char* q, const float* scale, bf16* out, int rows, int cols, int orows, int ocols) {
    const size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i >= (size_t)orows * ocols) return;
    const int r = (int)(i / ocols), c = (int)(i % ocols);
    if (r >= rows || c >= cols) {
        out[i] = out[i + 1] = out[i + 2] = out[i + 3] = f2b(0.f);
        return;
    }
    const float sc = scale[r];
    const char4 v = *reinterpret_cast<const char4*>(q + (size_t)r * cols + c);
    out[i] = f2b(v.x * sc);
    out[i + 1] = f2b(v.y * sc);
    out[i + 2] = f2b(v.z * sc);
    out[i + 3] = f2b(v.w * sc);
}

// y [n, H] += sh: bf16(y + sh) (the routed experts' sum and the shared expert's, as the reference adds them). A thread a value.
extern "C" __global__ void qx_add_bf16(bf16* y, const bf16* sh, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) y[i] = f2b(b2f(y[i]) + b2f(sh[i]));
}

// out [n, I] = bf16(bf16(silu(g)) up) of g [n, 2 I] (gate then up, bf16): the shared expert's activation for a prompt chunk. A thread a value.
extern "C" __global__ void qx_swiglu(const bf16* g, bf16* out, int n, int I) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * I) return;
    const int t = i / I, j = i % I;
    const float a = b2f(g[(size_t)t * 2 * I + j]), u = b2f(g[(size_t)t * 2 * I + I + j]);
    out[i] = f2b(qx_r(a * qx_sig(a)) * u);
}

// sh [n, H] = bf16(bf16(sigmoid(logit[t])) sh) in place (logit bf16 [n]): the shared expert's gate for a prompt chunk. A thread a value.
extern "C" __global__ void qx_gate_rows(bf16* sh, const bf16* logit, int n, int H) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * H) return;
    sh[i] = f2b(qx_r(qx_sig(b2f(logit[i / H]))) * b2f(sh[i]));
}

// The Viterbi search of one row a block (1024 threads; dynamic shared memory (TQ_S + T) floats): the row (fp32 [n, T] in w) over its RMS
// against gain times the states' values, the cost of every state kept in shared memory and updated in place a step at a time (all reads,
// a sync, all writes). A step's choice for group q (its successors 4q .. 4q + 3; its predecessors q + j TQ_Q) is 2 bits of dec
// [n, T, 1024] (byte q % 1024, bits 2 (q / 1024)). h null: any start, end[row] the best last state; else the start's 12 high bits and the
// last state's 12 low bits are h[row] (the stream closes on itself), end[row] the best such last state.
extern "C" __global__ void __launch_bounds__(1024) qx_tq_viterbi(const float* w, int T, float gain, unsigned a, unsigned b, const unsigned* h, unsigned char* dec, unsigned* end) {
    extern __shared__ float tq_sm[];
    float* cost = tq_sm;
    float* z = tq_sm + TQ_S;
    __shared__ float red[32];
    __shared__ unsigned redi[32];
    const int r = blockIdx.x, tid = threadIdx.x, lane = tid & 31, wp = tid >> 5;
    const float* row = w + (size_t)r * T;
    float ss = 0.f;
    for (int i = tid; i < T; i += 1024) {
        const float v = row[i];
        z[i] = v;
        ss += v * v;
    }
    for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if (lane == 0) red[wp] = ss;
    __syncthreads();
    if (wp == 0) {
        float v = red[lane];
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) red[0] = v;
    }
    __syncthreads();
    const float inv = rsqrtf(fmaxf(red[0] / (float)T, 1e-30f)), gk = gain * TQ_K, gc = gain * TQ_C;
    for (int i = tid; i < T; i += 1024) z[i] *= inv;
    __syncthreads();
    const unsigned hh = h ? h[r] : 0u;
    for (int s = tid; s < TQ_S; s += 1024) {
        const float d = z[0] - fmaf((float)__dp4a(s * a + b, 0x01010101u, 0u), gk, -gc);
        cost[s] = (h && ((unsigned)s >> 2) != hh) ? 1e30f : d * d;
    }
    __syncthreads();
    unsigned char* dr = dec + (size_t)r * T * 1024;
    for (int t = 1; t < T; t++) {
        const float zt = z[t];
        float m[4];
        unsigned pick = 0;
#pragma unroll
        for (int g = 0; g < 4; g++) {
            const int q = tid + 1024 * g;
            const float c0 = cost[q], c1 = cost[q + TQ_Q], c2 = cost[q + 2 * TQ_Q], c3 = cost[q + 3 * TQ_Q];
            float mm = c0;
            unsigned j = 0;
            if (c1 < mm) { mm = c1; j = 1; }
            if (c2 < mm) { mm = c2; j = 2; }
            if (c3 < mm) { mm = c3; j = 3; }
            m[g] = mm;
            pick |= j << (2 * g);
        }
        dr[(size_t)t * 1024 + tid] = (unsigned char)pick;
        __syncthreads();
#pragma unroll
        for (int g = 0; g < 4; g++) {
            const unsigned s = 4u * (unsigned)(tid + 1024 * g);
            float4 o;
            float d = zt - fmaf((float)__dp4a(s * a + b, 0x01010101u, 0u), gk, -gc);
            o.x = m[g] + d * d;
            d = zt - fmaf((float)__dp4a((s + 1) * a + b, 0x01010101u, 0u), gk, -gc);
            o.y = m[g] + d * d;
            d = zt - fmaf((float)__dp4a((s + 2) * a + b, 0x01010101u, 0u), gk, -gc);
            o.z = m[g] + d * d;
            d = zt - fmaf((float)__dp4a((s + 3) * a + b, 0x01010101u, 0u), gk, -gc);
            o.w = m[g] + d * d;
            reinterpret_cast<float4*>(cost)[tid + 1024 * g] = o;
        }
        __syncthreads();
    }
    if (h) {
        if (tid == 0) {
            unsigned best = hh;
            for (unsigned x = 1; x < 4; x++)
                if (cost[(x << 12) | hh] < cost[best]) best = (x << 12) | hh;
            end[r] = best;
        }
        return;
    }
    float bv = 3e38f;
    unsigned bi = 0;
    for (int s = tid; s < TQ_S; s += 1024)
        if (cost[s] < bv) { bv = cost[s]; bi = s; }
    for (int o = 16; o; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const unsigned oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov < bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0) { red[wp] = bv; redi[wp] = bi; }
    __syncthreads();
    if (wp == 0) {
        bv = red[lane];
        bi = redi[lane];
        for (int o = 16; o; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const unsigned oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov < bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) end[r] = bi;
    }
}

// The best path back from end[row] (a thread a row, n rows). codes null: the start's 12 high bits to h[row]. Else the stream to codes
// [n, T / 16] (the start's 14 bits, then each later state's 2 low bits; the last 6 states' wrap onto the start's), the least-squares scale
// of the path's values to the row (w, fp32 [n, T]) to scale[row], and the row's squared error at that scale to err[row].
extern "C" __global__ void qx_tq_back(const unsigned char* dec, const unsigned* end, int T, int n, unsigned* h, const float* w, unsigned a, unsigned b, unsigned* codes, float* scale, float* err) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    const unsigned char* dr = dec + (size_t)r * T * 1024;
    const int W = T / 16;
    unsigned* cr = codes ? codes + (size_t)r * W : nullptr;
    if (cr)
        for (int i = 0; i < W; i++) cr[i] = 0u;
    const float* row = w + (size_t)r * T;
    float wc = 0.f, cc = 0.f, ww = 0.f;
    unsigned s = end[r];
    for (int t = T - 1;; t--) {
        if (cr) {
            const float c = tq_val(s, a, b), x = row[t];
            wc += x * c;
            cc += c * c;
            ww += x * x;
            if (t >= 1 && t <= T - 7) {
                const int p = 2 * t + 12;
                cr[p >> 5] |= (s & 3u) << (30 - (p & 31));
            }
        }
        if (t == 0) break;
        const unsigned q = s >> 2, j = (dr[(size_t)t * 1024 + (q & 1023u)] >> (2 * (q >> 10))) & 3u;
        s = (j << 12) | q;
    }
    if (!cr) {
        h[r] = s >> 2;
        return;
    }
    cr[0] |= s << (32 - TQ_L);
    const float sc = wc / fmaxf(cc, 1e-30f);
    scale[r] = sc;
    err[r] = fmaxf(ww - 2.f * sc * wc + sc * sc * cc, 0.f);
}

// Rows back from their codes (grid n x T / 512, 512 threads): the values in the rotated basis to xr (fp32 [n, T]) where given, and the
// rows R^T y (each 512 values' Walsh-Hadamard transform over sqrt(512), then the signs sg) to x (bf16 [n, T]) where given.
extern "C" __global__ void __launch_bounds__(512) qx_tq_dequant(const unsigned* codes, const float* scale, int T, unsigned a, unsigned b, const signed char* sg, float* xr, bf16* x) {
    __shared__ float v[512];
    const int r = blockIdx.x, i = blockIdx.y * 512 + threadIdx.x, W = T / 16;
    const float y = scale[r] * tq_val(tq_state(codes + (size_t)r * W, W, i), a, b);
    if (xr) xr[(size_t)r * T + i] = y;
    if (!x) return;
    v[threadIdx.x] = y;
    __syncthreads();
    qc_wht512(v);
    x[(size_t)r * T + i] = f2b(v[threadIdx.x] * 0.044194173824159216f * (float)sg[i]);
}

// Format q at K = 3 and 4 bits a weight: a row of T weights as a circular stream of K T bits, weight t's state the TQ_L bits from bit K t (tq_state_k), its value as at
// 2 bits. The search of one row a block (1024 threads): thread i keeps in registers the costs of its NG groups q = i + 1024 m (a group: the 2^K states q + G j sharing
// their TQ_L - K low bits, the predecessors of the states (q << K) | c). A step takes each group's least cost (its choices, K bits a group, to dec [n, T, 1024] u16), then
// the successors' costs go through shared memory (a float of padding every 32: the writes, 2^K apart, in different banks) to the threads that keep them. Dynamic shared
// memory: (TQ_S + TQ_S / 32 + T) floats. h null: any start; else the start's TQ_L - K high bits and the last state's TQ_L - K low bits are h[row]. end[row]: the best
// last state. (States of 15 bits: 1-2% less squared error than 14 for 3-5 times the time; 16 do not fit a block's registers.)
template <int K>
__device__ __forceinline__ void tq_vit_k(const float* w, int T, float gain, unsigned a, unsigned b, const unsigned* h, unsigned short* dec, unsigned* end) {
    constexpr int G = TQ_S >> K, B = 1 << K, NG = G / 1024;
    static_assert(NG >= 1 && NG * K <= 16, "a thread's groups: whole, their choices in 16 bits");
    extern __shared__ float tqk_sm[];
    float* xs = tqk_sm;
    float* z = tqk_sm + TQ_S + TQ_S / 32;
    __shared__ float red[32];
    __shared__ unsigned redi[32];
    const int r = blockIdx.x, tid = threadIdx.x, lane = tid & 31, wp = tid >> 5;
    const float* row = w + (size_t)r * T;
    float ss = 0.f;
    for (int i = tid; i < T; i += 1024) {
        const float v = row[i];
        z[i] = v;
        ss += v * v;
    }
    for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if (lane == 0) red[wp] = ss;
    __syncthreads();
    if (wp == 0) {
        float v = red[lane];
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) red[0] = v;
    }
    __syncthreads();
    const float inv = rsqrtf(fmaxf(red[0] / (float)T, 1e-30f)), gk = gain * TQ_K, gc = gain * TQ_C;
    for (int i = tid; i < T; i += 1024) z[i] *= inv;
    __syncthreads();
    const unsigned hh = h ? h[r] : 0u;
    float c[NG][B];
#pragma unroll
    for (int m = 0; m < NG; m++)
#pragma unroll
        for (int j = 0; j < B; j++) {
            const unsigned s = (unsigned)(tid + 1024 * m + G * j);
            const float d = z[0] - fmaf((float)__dp4a(s * a + b, 0x01010101u, 0u), gk, -gc);
            c[m][j] = (h && (s >> K) != hh) ? 1e30f : d * d;
        }
    unsigned short* dr = dec + (size_t)r * T * 1024;
    for (int t = 1; t < T; t++) {
        const float zt = z[t];
        unsigned pick = 0;
#pragma unroll
        for (int m = 0; m < NG; m++) {
            float v = c[m][0];
            unsigned jb = 0;
#pragma unroll
            for (int j = 1; j < B; j++)
                if (c[m][j] < v) {
                    v = c[m][j];
                    jb = j;
                }
            pick |= jb << (K * m);
            const unsigned q = (unsigned)(tid + 1024 * m);
#pragma unroll
            for (int x = 0; x < B; x++) {
                const unsigned s = (q << K) | (unsigned)x;
                const float d = zt - fmaf((float)__dp4a(s * a + b, 0x01010101u, 0u), gk, -gc);
                xs[s + (s >> 5)] = fmaf(d, d, v);
            }
        }
        dr[(size_t)t * 1024 + tid] = (unsigned short)pick;
        __syncthreads();
#pragma unroll
        for (int m = 0; m < NG; m++)
#pragma unroll
            for (int j = 0; j < B; j++) {
                const unsigned o = (unsigned)(tid + 1024 * m + G * j);
                c[m][j] = xs[o + (o >> 5)];
            }
        __syncthreads();
    }
    float bv = 3e38f;
    unsigned bi = 0;
    if (h) {
        // (the states whose low TQ_L - K bits are h: group h's, one thread's)
        if ((hh & 1023u) == (unsigned)tid) {
#pragma unroll
            for (int m = 0; m < NG; m++)
                if ((unsigned)m == hh >> 10)
#pragma unroll
                    for (int j = 0; j < B; j++)
                        if (c[m][j] < bv) {
                            bv = c[m][j];
                            bi = hh + G * j;
                        }
            end[r] = bi;
        }
        return;
    }
#pragma unroll
    for (int m = 0; m < NG; m++)
#pragma unroll
        for (int j = 0; j < B; j++) {
            const unsigned s = (unsigned)(tid + 1024 * m + G * j);
            if (c[m][j] < bv || (c[m][j] == bv && s < bi)) {
                bv = c[m][j];
                bi = s;
            }
        }
    for (int o = 16; o; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const unsigned oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov < bv || (ov == bv && oi < bi)) {
            bv = ov;
            bi = oi;
        }
    }
    __syncthreads();
    if (lane == 0) {
        red[wp] = bv;
        redi[wp] = bi;
    }
    __syncthreads();
    if (wp == 0) {
        bv = red[lane];
        bi = redi[lane];
        for (int o = 16; o; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const unsigned oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov < bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0) end[r] = bi;
    }
}

// nbits of v (most significant first) at bit p of a circular stream of W words.
__device__ __forceinline__ void tq_put(unsigned* cr, int W, int p, unsigned v, int nbits) {
    for (int i = 0; i < nbits; i++) {
        const int q = (p + i) % (32 * W);
        cr[q >> 5] |= ((v >> (nbits - 1 - i)) & 1u) << (31 - (q & 31));
    }
}

// The best path back from end[row] (a thread a row, n rows), as qx_tq_back: codes null, the start's TQ_L - K high bits to h[row]; else the stream to codes [n, K T / 32]
// (the start's TQ_L bits at bit 0, each later state's K low bits at bit K t + TQ_L - K, around the end), the least-squares scale and the squared error at it.
template <int K>
__device__ __forceinline__ void tq_back_k(const unsigned short* dec, const unsigned* end, int T, int n, unsigned* h, const float* w, unsigned a, unsigned b, unsigned* codes, float* scale, float* err) {
    constexpr int G = TQ_S >> K;
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    const unsigned short* dr = dec + (size_t)r * T * 1024;
    const int W = K * T / 32;
    unsigned* cr = codes ? codes + (size_t)r * W : nullptr;
    if (cr)
        for (int i = 0; i < W; i++) cr[i] = 0u;
    const float* row = w + (size_t)r * T;
    float wc = 0.f, cc = 0.f, ww = 0.f;
    unsigned s = end[r];
    for (int t = T - 1;; t--) {
        if (cr) {
            const float c = tq_val(s, a, b), x = row[t];
            wc += x * c;
            cc += c * c;
            ww += x * x;
            if (t >= 1) tq_put(cr, W, K * t + TQ_L - K, s & ((1u << K) - 1u), K);
        }
        if (t == 0) break;
        const unsigned q = s >> K, j = (dr[(size_t)t * 1024 + (q & 1023u)] >> (K * (q >> 10))) & ((1u << K) - 1u);
        s = q + G * j;
    }
    if (!cr) {
        h[r] = s >> K;
        return;
    }
    tq_put(cr, W, 0, s, TQ_L);
    const float sc = wc / fmaxf(cc, 1e-30f);
    scale[r] = sc;
    err[r] = fmaxf(ww - 2.f * sc * wc + sc * sc * cc, 0.f);
}

// Rows back from their codes (grid n x T / 512, 512 threads), as qx_tq_dequant.
template <int K>
__device__ __forceinline__ void tq_deq_k(const unsigned* codes, const float* scale, int T, unsigned a, unsigned b, const signed char* sg, float* xr, bf16* x) {
    __shared__ float v[512];
    const int r = blockIdx.x, i = blockIdx.y * 512 + threadIdx.x, W = K * T / 32;
    const float y = scale[r] * tq_val(tq_state_k<K>(codes + (size_t)r * W, W, i), a, b);
    if (xr) xr[(size_t)r * T + i] = y;
    if (!x) return;
    v[threadIdx.x] = y;
    __syncthreads();
    qc_wht512(v);
    x[(size_t)r * T + i] = f2b(v[threadIdx.x] * 0.044194173824159216f * (float)sg[i]);
}

#define TQ_ENC(K)                                                                                                                                                                              \
    extern "C" __global__ void __launch_bounds__(1024, 1) qx_tq_vit##K(const float* w, int T, float gain, unsigned a, unsigned b, const unsigned* h, unsigned short* dec, unsigned* end) {       \
        tq_vit_k<K>(w, T, gain, a, b, h, dec, end);                                                                                                                                             \
    }                                                                                                                                                                                           \
    extern "C" __global__ void qx_tq_back##K(const unsigned short* dec, const unsigned* end, int T, int n, unsigned* h, const float* w, unsigned a, unsigned b, unsigned* codes, float* scale, float* err) { \
        tq_back_k<K>(dec, end, T, n, h, w, a, b, codes, scale, err);                                                                                                                            \
    }                                                                                                                                                                                           \
    extern "C" __global__ void __launch_bounds__(512) qx_tq_deq##K(const unsigned* codes, const float* scale, int T, unsigned a, unsigned b, const signed char* sg, float* xr, bf16* x) {         \
        tq_deq_k<K>(codes, scale, T, a, b, sg, xr, x);                                                                                                                                          \
    }
TQ_ENC(3)
TQ_ENC(4)
