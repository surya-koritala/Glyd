// Gemma 4's own kernels (included by engine.cu; the description executor runs them through src/kern/gemma4.rs's table): the norms (the weight as it is, x * w, in fp32 with one rounding), the sandwich residual (the
// sublayer's output normed before it is added), the per-layer scalar, gelu(tanh) times up, and the final soft-cap; its attention is the attention kind's (kernels/attn.cu).
// bf16 in memory, fp32 inside; the rounding points are transformers' eager ones (models/gemma4/modeling_gemma4.py): a norm rounds once, gelu rounds before the product by up.

// ---- norms and the residual ------------------------------------------------------------------------------------------------------------
// 1 / sqrt(mean of squares + eps): the norm's scale (transformers: torch.pow(mean + eps, -0.5) in fp32).
__device__ __forceinline__ float g4_scale(float ss, int K, float eps) { return 1.f / sqrtf(ss / (float)K + eps); }

// out [M, K] = bf16(rmsnorm(x) * w); w null: no weight. One block a row.
extern "C" __global__ void g4_rmsnorm(const bf16* x, const bf16* w, bf16* out, int K, float eps) {
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
    const float s = g4_scale(block_sum(acc), K, eps);
#pragma unroll
    for (int j = 0; j < RN_PER_THREAD; j++) {
        int i = threadIdx.x + j * blockDim.x;
        if (i < K) o[i] = f2b(w ? v[j] * s * b2f(w[i]) : v[j] * s);
    }
}

// The embedding's scale, then the first layer's input norm: x = bf16(x * scale) (scale the bf16 of sqrt(hidden), as transformers casts it), out = bf16(rmsnorm(x) * w).
extern "C" __global__ void g4_embed_norm(bf16* x, float scale, const bf16* w, bf16* out, int K, float eps) {
    bf16* xr = x + (size_t)blockIdx.x * K;
    bf16* o = out + (size_t)blockIdx.x * K;
    float v[RN_PER_THREAD];
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < RN_PER_THREAD; j++) {
        int i = threadIdx.x + j * blockDim.x;
        v[j] = i < K ? b2f(f2b(b2f(xr[i]) * scale)) : 0.f;
        acc += v[j] * v[j];
    }
    const float s = g4_scale(block_sum(acc), K, eps);
#pragma unroll
    for (int j = 0; j < RN_PER_THREAD; j++) {
        int i = threadIdx.x + j * blockDim.x;
        if (i < K) {
            xr[i] = f2b(v[j]);
            o[i] = f2b(v[j] * s * b2f(w[i]));
        }
    }
}

// The sandwich residual of a sublayer: x = bf16(x + bf16(rmsnorm(y) * wpost)); where `scalar`, x = bf16(x * scalar[0]) (the layer's scalar, at the end of a layer); where `wnext`,
// out = bf16(rmsnorm(x) * wnext) (the next norm: the feed-forward's pre-norm, the next layer's input norm or the final norm). Two reductions, one block a row.
// blockIdx.y above 0: `pf`'s prefetch.
extern "C" __global__ void __launch_bounds__(1024) g4_add_norm(bf16* x, const bf16* y, const bf16* wpost, const bf16* scalar, const bf16* wnext, bf16* out, int K, float eps, NextPf pf) {
    if (blockIdx.y > 0) return next_pf(pf, (blockIdx.y - 1) * gridDim.x + blockIdx.x, (gridDim.y - 1) * gridDim.x);
    bf16* xr = x + (size_t)blockIdx.x * K;
    const bf16* yr = y + (size_t)blockIdx.x * K;
    // (every load first: one wait on L2 instead of three, the sums after)
    float v[RN_PER_THREAD], xv[RN_PER_THREAD], wp[RN_PER_THREAD], wn[RN_PER_THREAD];
#pragma unroll
    for (int j = 0; j < RN_PER_THREAD; j++) {
        int i = threadIdx.x + j * blockDim.x;
        const bool in = i < K;
        v[j] = in ? b2f(yr[i]) : 0.f;
        xv[j] = in ? b2f(xr[i]) : 0.f;
        wp[j] = in ? b2f(wpost[i]) : 0.f;
        wn[j] = in && wnext ? b2f(wnext[i]) : 0.f;
    }
    const float sc = scalar ? b2f(scalar[0]) : 1.f;
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < RN_PER_THREAD; j++) acc += v[j] * v[j];
    const float s = g4_scale(block_sum(acc), K, eps);
    acc = 0.f;
#pragma unroll
    for (int j = 0; j < RN_PER_THREAD; j++) {
        int i = threadIdx.x + j * blockDim.x;
        if (i < K) {
            float z = b2f(f2b(xv[j] + b2f(f2b(v[j] * s * wp[j]))));
            if (scalar) z = b2f(f2b(z * sc));
            v[j] = z;
        } else {
            v[j] = 0.f;
        }
        acc += v[j] * v[j];
    }
    const float s2 = wnext ? g4_scale(block_sum(acc), K, eps) : 0.f;
    bf16* o = out + (size_t)blockIdx.x * K;
#pragma unroll
    for (int j = 0; j < RN_PER_THREAD; j++) {
        int i = threadIdx.x + j * blockDim.x;
        if (i < K) {
            xr[i] = f2b(v[j]);
            if (wnext) o[i] = f2b(v[j] * s2 * wn[j]);
        }
    }
}

// ---- the feed-forward and the logits ------------------------------------------------------------------------------------------------------
// m [n, inter] = bf16(bf16(gelu_tanh(gate)) * up), gate and up the halves of gu [n, 2 inter] (transformers: the activation's bf16 output times up, rounded).
extern "C" __global__ void g4_gelu_mul(const bf16* gu, bf16* m, int inter, NextPf pf) {
    if (blockIdx.z > 0) return next_pf(pf, (blockIdx.z - 1) * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x, (gridDim.z - 1) * gridDim.x * gridDim.y);
    const int r = blockIdx.y;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < inter; i += gridDim.x * blockDim.x) {
        const float g = b2f(gu[(size_t)r * 2 * inter + i]), u = b2f(gu[(size_t)r * 2 * inter + inter + i]);
        const float a = b2f(f2b(0.5f * g * (1.f + tanhf(0.7978845608028654f * (g + 0.044715f * g * g * g)))));
        m[(size_t)r * inter + i] = f2b(a * u);
    }
}

// The draft model's input (Gemma 4's "-assistant" models): out [2H] = the last token's embedding as the big model scales it, bf16(emb * scale), then the big model's final
// (normed) hidden state for the position before it. One grid-stride loop.
extern "C" __global__ void g4_draft_in(const bf16* emb, float scale, const bf16* hid, bf16* out, int H) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < 2 * H; i += gridDim.x * blockDim.x) out[i] = i < H ? f2b(b2f(emb[i]) * scale) : hid[i - H];
}

// The final soft-cap of a row of logits, in place: bf16(bf16(tanh(bf16(l / cap))) * cap).
extern "C" __global__ void g4_softcap(bf16* l, int V, float cap) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < V; i += gridDim.x * blockDim.x) {
        const float a = b2f(f2b(b2f(l[i]) / cap));
        l[i] = f2b(b2f(f2b(tanhf(a))) * cap);
    }
}
