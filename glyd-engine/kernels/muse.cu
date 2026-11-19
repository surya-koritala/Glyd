// Muse Spark's own kernels (included by engine.cu after glm.cu and moe.cu; src/dynamic/muse.rs describes the model and src/dynamic/mod.rs and src/moe.rs run them): the queries' and keys' norms and
// rotary, the attention over heads of 64 (a window, or every position), the output's norm and gate, the input-scaled norm, the residual gates, the router of fp32 weights, the experts' activation and the
// norm of each expert's output with the weighted sum, the logits' soft-cap. bf16 in memory, fp32 inside; the rounding points are transformers' modeling_muse_spark.py's (src/muse_ref.rs: the CPU reference).

#define MU_HD 64   // a head's channels

// The RMS factor of the 64 values a warp holds (2 a lane), with eps.
__device__ __forceinline__ float mu_rs64(float a, float b, float eps) {
    return rsqrtf(warp_sum(a * a + b * b) / MU_HD + eps);
}

// The heads of each row of qkvg [n, stride] (q: nq heads, then k: nk, v: nk; channels of 64): each q and k head normed without a weight (eps), q times `scale` (rounded), the rotary where `rope` (rotate_half;
// cos and sin rounded to bf16, each product and the sum rounded, transformers' bf16 arithmetic) at the row's position; the queries into qo [n, nq, 64], the keys and values into the ring kc, vc [cap, nk, 64] at
// the position modulo cap. A warp a head (nq + 2 nk of them), a lane 2 channels (i and i + 32: the rotary's pair).
extern "C" __global__ void muse_qk(const bf16* qkvg, int stride, bf16* qo, bf16* kc, bf16* vc, const float* cs, const float* sn, const int* pos, int cap, int nq, int nk, int rope, float scale, float eps) {
    const int t = blockIdx.x, w = blockIdx.y * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (w >= nq + 2 * nk) return;
    const bf16* x = qkvg + (size_t)t * stride + (size_t)w * MU_HD;
    float a = b2f(x[lane]), b = b2f(x[lane + 32]);
    const int p = pos[t];
    if (w < nq + nk) {
        const float r = mu_rs64(a, b, eps);
        a = b2f(f2b(a * r));
        b = b2f(f2b(b * r));
        if (w < nq) {
            a = b2f(f2b(a * scale));
            b = b2f(f2b(b * scale));
        }
        if (rope) {
            const float c = b2f(f2b(cs[(size_t)p * 32 + lane])), s = b2f(f2b(sn[(size_t)p * 32 + lane]));
            const float na = b2f(f2b(b2f(f2b(a * c)) + b2f(f2b(-b * s))));
            const float nb = b2f(f2b(b2f(f2b(b * c)) + b2f(f2b(a * s))));
            a = na;
            b = nb;
        }
    }
    bf16* o;
    if (w < nq) {
        o = qo + ((size_t)t * nq + w) * MU_HD;
    } else if (w < nq + nk) {
        o = kc + ((size_t)(p % cap) * nk + (w - nq)) * MU_HD;
    } else {
        o = vc + ((size_t)(p % cap) * nk + (w - nq - nk)) * MU_HD;
    }
    o[lane] = f2b(a);
    o[lane + 32] = f2b(b);
}

// The attention of row t's queries (qo [n, nq, 64]) over the keys and values of the ring (kc, vc [cap, nk, 64]) at positions max(0, p - window + 1) .. p (window 0: from 0), scale 64^-1/2, the softmax in fp32,
// the output rounded once into att [n, nq, 64]. Block (row, key-value head), a warp each of its nq / nk query heads, a lane 2 channels; the keys in order (an online softmax).
extern "C" __global__ void muse_attn(const bf16* qo, const bf16* kc, const bf16* vc, const int* pos, bf16* att, int cap, int nq, int nk, int window) {
    const int t = blockIdx.x, kv = blockIdx.y, w = threadIdx.x >> 5, lane = threadIdx.x & 31, g = nq / nk;
    if (w >= g) return;
    const int h = kv * g + w, p = pos[t];
    const bf16* q = qo + ((size_t)t * nq + h) * MU_HD;
    const float qa = b2f(q[lane]), qb = b2f(q[lane + 32]);
    const int first = window > 0 ? max(0, p - window + 1) : 0;
    float m = -INFINITY, l = 0.f, oa = 0.f, ob = 0.f;
    for (int j = first; j <= p; j++) {
        const bf16* k = kc + ((size_t)(j % cap) * nk + kv) * MU_HD;
        const float s = warp_sum(qa * b2f(k[lane]) + qb * b2f(k[lane + 32])) * 0.125f;
        const float mn = fmaxf(m, s), c = expf(m - mn), e = expf(s - mn);
        const bf16* v = vc + ((size_t)(j % cap) * nk + kv) * MU_HD;
        l = l * c + e;
        oa = oa * c + e * b2f(v[lane]);
        ob = ob * c + e * b2f(v[lane + 32]);
        m = mn;
    }
    bf16* o = att + ((size_t)t * nq + h) * MU_HD;
    o[lane] = f2b(oa / l);
    o[lane + 32] = f2b(ob / l);
}

// out [n, nq 64] = each head of att normed without a weight (eps), rounded, times bf16(sigmoid(gate)), rounded (the gate: qkvg's columns from `goff`, rows `stride` apart). A warp a head.
extern "C" __global__ void muse_ogate(const bf16* att, const bf16* qkvg, int stride, int goff, bf16* out, int nq, float eps) {
    const int t = blockIdx.x, h = blockIdx.y * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (h >= nq) return;
    const bf16* a = att + ((size_t)t * nq + h) * MU_HD;
    const bf16* g = qkvg + (size_t)t * stride + goff + (size_t)h * MU_HD;
    float x = b2f(a[lane]), y = b2f(a[lane + 32]);
    const float r = mu_rs64(x, y, eps);
    x = b2f(f2b(x * r));
    y = b2f(f2b(y * r));
    const float sg0 = b2f(f2b(1.f / (1.f + expf(-b2f(g[lane]))))), sg1 = b2f(f2b(1.f / (1.f + expf(-b2f(g[lane + 32])))));
    bf16* o = out + ((size_t)t * nq + h) * MU_HD;
    o[lane] = f2b(x * sg0);
    o[lane + 32] = f2b(y * sg1);
}

// The input-scaled norm: y = RMS(bf16(x w)) without a weight (eps), rounded; rows of K. Block a row, 256 threads (K up to 256 x 32).
extern "C" __global__ void __launch_bounds__(256) muse_rms_in(const bf16* x, const bf16* w, bf16* y, int K, float eps) {
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

// The residual gate, x in place: x = bf16(bf16(alpha x) + bf16(beta h)) (alpha, beta: bf16 [K], the gate's factors). A thread a column.
extern "C" __global__ void muse_rgate(bf16* x, const bf16* h, const bf16* alpha, const bf16* beta, int K) {
    const int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= K) return;
    const size_t at = (size_t)blockIdx.x * K + i;
    x[at] = f2b(b2f(f2b(b2f(alpha[i]) * b2f(x[at]))) + b2f(f2b(b2f(beta[i]) * b2f(h[at]))));
}

// The router's logits [T, E] (fp32) = x [T, H] (bf16) W^T (fp32 [E, H]) times `mult`. A warp an expert of a token: block (token, 8 experts), 256 threads.
extern "C" __global__ void muse_router(const bf16* x, const float* W, float* logits, int E, int H, float mult) {
    const int t = blockIdx.x, e = blockIdx.y * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (e >= E) return;
    const bf16* xr = x + (size_t)t * H;
    const float* wr = W + (size_t)e * H;
    float s = 0.f;
    for (int i = lane; i < H; i += 32) s = fmaf(b2f(xr[i]), wr[i], s);
    s = warp_sum(s);
    if (lane == 0) logits[(size_t)t * E + e] = s * mult;
}

// The experts' activation in the plan's order: out [P, I] = bf16(bf16(silu(gate)) up) of gu [P, 2 I] (transformers' act_fn on a bf16 tensor, then the product). A thread a column of a row.
extern "C" __global__ void muse_act(const bf16* gu, bf16* out, int I) {
    const int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= I) return;
    const size_t r = blockIdx.x;
    const float g = b2f(gu[r * 2 * I + i]), u = b2f(gu[r * 2 * I + I + i]);
    out[r * I + i] = f2b(b2f(f2b(g / (1.f + expf(-g)))) * u);
}

// inv[pair] = its row in the plan's order (plan [2 + 2E + P]: the pairs by expert, glyd_gpu_moe_way's). A thread a row.
extern "C" __global__ void muse_inv(const int* plan, int* inv, int P, int E) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < P) inv[plan[2 + 2 * E + r]] = r;
}

// Each token's weighted sum of its experts' outputs, as transformers adds them: rows [P, L] (each pair's second product, unweighted, in the plan's order), each normed (input-scaled with w, eps), times
// its weight wts[t k + j] rounded to bf16, added in the order of the experts' ids with each sum rounded, into out [T, L]. Block a token, 256 threads (L up to 256 x 16).
extern "C" __global__ void __launch_bounds__(256) muse_moe_sum(const bf16* rows, const int* inv, const long long* ids, const float* wts, const bf16* w, bf16* out, int k, int L, float eps) {
    const int t = blockIdx.x;
    float acc[16];
#pragma unroll
    for (int j = 0; j < 16; j++) acc[j] = 0.f;
    // (the token's choices by expert id: k is small, a selection pass each)
    long long done = -1;
    for (int c = 0; c < k; c++) {
        long long best = 0x7fffffffffffffffll;
        int bj = -1;
        for (int j = 0; j < k; j++) {
            const long long e = ids[(size_t)t * k + j];
            if (e > done && e < best) {
                best = e;
                bj = j;
            }
        }
        if (bj < 0) break;
        done = best;
        const bf16* xr = rows + (size_t)inv[t * k + bj] * L;
        float v[16], ss = 0.f;
#pragma unroll
        for (int j = 0; j < 16; j++) {
            const int i = threadIdx.x + j * 256;
            v[j] = i < L ? b2f(f2b(b2f(xr[i]) * b2f(w[i]))) : 0.f;
            ss += v[j] * v[j];
        }
        const float r = rsqrtf(block_sum(ss) / L + eps), wt = wts[(size_t)t * k + bj];
#pragma unroll
        for (int j = 0; j < 16; j++) acc[j] = b2f(f2b(acc[j] + b2f(f2b(b2f(f2b(v[j] * r)) * wt))));
    }
#pragma unroll
    for (int j = 0; j < 16; j++) {
        const int i = threadIdx.x + j * 256;
        if (i < L) out[(size_t)t * L + i] = f2b(acc[j]);
    }
}

// The logits' soft-cap in place (bf16 [n]): cap tanh(x mult / cap), each step rounded as transformers' bf16 tensors are.
extern "C" __global__ void muse_softcap(bf16* x, int n, float mult, float cap) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float a = b2f(f2b(b2f(x[i]) * mult));
        const float b = b2f(f2b(a / cap));
        x[i] = f2b(b2f(f2b(tanhf(b))) * cap);
    }
}
