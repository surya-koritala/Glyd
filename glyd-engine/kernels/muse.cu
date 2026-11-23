// Muse Spark's own kernels (included by engine.cu after glm.cu and moe.cu; src/dynamic/muse.rs describes the model and src/dynamic/mod.rs and src/moe.rs run them; its attention is the attention
// kind's, kernels/attn.cu): the input-scaled norm, the residual gates, the router of fp32 weights, the experts' activation and the norm of each expert's output with the weighted sum, the logits' soft-cap.
// bf16 in memory, fp32 inside; the rounding points are transformers' modeling_muse_spark.py's (src/muse_ref.rs: the CPU reference).

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
