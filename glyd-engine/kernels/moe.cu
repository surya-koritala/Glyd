// The mixture-of-experts layer's own kernels (included by engine.cu; src/moe.rs runs them): the router and the pieces around the experts' two products. The products themselves are the library's
// (glyd_gpu_mma_moe over E experts' packs stacked as one matrix; glyd_gpu_moe_way sorts the pairs, a token's k choices, by expert). bf16 in memory, fp32 inside, one rounding where vLLM's fused MoE
// rounds once. The router's own arithmetic is fp32 (a DeepSeek-style router is computed in it); the logits of a bf16 gate are rounded to bf16 where moe_select reads them.
// Nothing here reads a result on the host: the experts a token takes stay on the device from the logits to the weighted sum, so a decode step's layers capture into the CUDA graph.

#define MOE_FULL 0xffffffffu
#define MOE_SIGMOID 1  // moe_select's flags: the scores by sigmoid (else softmax over the experts)
#define MOE_RENORM 2   // the chosen scores divided by their sum
#define MOE_BF16 4     // the logits rounded to bf16 first (the gate is a bf16 Linear)
#define MOE_SQRTSP 8   // the scores by sqrt(softplus) (DeepSeek-V4; softplus is the logit itself past 20, as torch's)
#define MOE_SILU 0     // moe_act's modes: silu(gate) up, gelu_tanh(gate) up, relu(x)^2 without a gate
#define MOE_GELU 1
#define MOE_RELU2 2
#define MOE_SITU 3     // situ: lim tanh(g / lim) sigmoid(g) x lin tanh(u / lin) (Kimi-K3: 4 and 25)

// 8 bf16 of a 16-byte load as floats (exact).
__device__ __forceinline__ void moe_bf8(const uint4 q, float* f) {
    f[0] = __uint_as_float(q.x << 16);
    f[1] = __uint_as_float(q.x & 0xffff0000u);
    f[2] = __uint_as_float(q.y << 16);
    f[3] = __uint_as_float(q.y & 0xffff0000u);
    f[4] = __uint_as_float(q.z << 16);
    f[5] = __uint_as_float(q.z & 0xffff0000u);
    f[6] = __uint_as_float(q.w << 16);
    f[7] = __uint_as_float(q.w & 0xffff0000u);
}

__device__ __forceinline__ float moe_warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(MOE_FULL, v, o));
    return v;
}

// ---- the router ------------------------------------------------------------------------------------------------------------------------
// logits [T, E] (fp32) = x [T, H] W^T for the router's bf16 W [E, H] and the rows of x (xs elements apart): fp32 sums (two bf16 multiply exactly in fp32), nothing rounded after. Block (2 experts,
// 8 tokens), 256 threads: 4 warps an expert, each a quarter of its row of W read once for the 8 tokens, the lanes' sums added, then the quarters' in order (more blocks, a shorter chain a warp: a decode
// step's router on a datacenter GPU waited on its row's loads one after another). H a multiple of 8; x, W and xs * 2 bytes 16-byte aligned. (Also the shared expert's gate: E = 1, W the gate's [1, H].)
extern "C" __global__ void __launch_bounds__(256) moe_logits(const bf16* x, int xs, const bf16* W, float* logits, int T, int E, int H) {
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, e = blockIdx.x * 2 + (w >> 2), q = w & 3, t0 = blockIdx.y * 8;
    __shared__ float red[2][8][4];
    const int H8 = H >> 3, c0 = q * H8 / 4, c1 = (q + 1) * H8 / 4;
    float acc[8];
#pragma unroll
    for (int j = 0; j < 8; j++) acc[j] = 0.f;
    if (e < E) {
        const uint4* wv = reinterpret_cast<const uint4*>(W + (size_t)e * H);
#pragma unroll 4
        for (int c = c0 + lane; c < c1; c += 32) {
            float wf[8];
            moe_bf8(wv[c], wf);
#pragma unroll
            for (int j = 0; j < 8; j++) {
                if (t0 + j < T) {
                    float v[8];
                    moe_bf8(reinterpret_cast<const uint4*>(x + (size_t)(t0 + j) * xs)[c], v);
#pragma unroll
                    for (int i = 0; i < 8; i++) acc[j] = fmaf(v[i], wf[i], acc[j]);
                }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const float sj = warp_sum(acc[j]);
        if (lane == 0) red[w >> 2][j][q] = sj;
    }
    __syncthreads();
    if (threadIdx.x < 16) {
        const int ee = threadIdx.x >> 3, j = threadIdx.x & 7, e2 = blockIdx.x * 2 + ee;
        if (e2 < E && t0 + j < T) logits[(size_t)(t0 + j) * E + e2] = ((red[ee][j][0] + red[ee][j][1]) + red[ee][j][2]) + red[ee][j][3];
    }
}

// A token's experts and their weights from its logits [E]: a warp a token, a lane's experts lane + 32 i in registers (N * 32 >= E). The scores: softmax over the experts, or sigmoid; the k largest of the
// scores (+ bias [E] where there is one: the bias picks, the score weighs; with softmax scores it is the expert cache's -1e30 for an expert a pack left out), largest first, equal ones by the smaller expert; their scores divided by their sum + 1e-20 (RENORM),
// times escale[expert] (Gemma 4's per-expert scale) where there is one, times scale (routed_scaling_factor). ids [T, k] are int64, as glyd_gpu_moe_way reads them (-1: no expert, if k > E); wts [T, k] fp32.
// k at most 32.
template <int N>
__device__ __forceinline__ void moe_select_row(const float* row, const float* bias, const float* escale, long long* ids, float* wts, int t, int E, int k, int flags, float scale);

template <int N>
__device__ __forceinline__ void moe_select_t(const float* logits, const float* bias, const float* escale, long long* ids, float* wts, int T, int E, int k, int flags, float scale) {
    const int t = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (t >= T) return;  // (a whole warp)
    moe_select_row<N>(logits + (size_t)t * E, bias, escale, ids, wts, t, E, k, flags, scale);
}

// moe_select_t's work for token t's row of logits, on the calling warp.
template <int N>
__device__ __forceinline__ void moe_select_row(const float* row, const float* bias, const float* escale, long long* ids, float* wts, int t, int E, int k, int flags, float scale) {
    const int lane = threadIdx.x & 31;
    float s[N], c[N];
    float mx = -INFINITY;
#pragma unroll
    for (int i = 0; i < N; i++) {
        float l = lane + 32 * i < E ? row[lane + 32 * i] : -INFINITY;
        if (flags & MOE_BF16) l = b2f(f2b(l));
        s[i] = l;
        mx = fmaxf(mx, l);
    }
    if (flags & MOE_SQRTSP) {
#pragma unroll
        for (int i = 0; i < N; i++) s[i] = lane + 32 * i < E ? sqrtf(s[i] > 20.f ? s[i] : log1pf(expf(s[i]))) : -INFINITY;
    } else if (flags & MOE_SIGMOID) {
#pragma unroll
        for (int i = 0; i < N; i++) s[i] = lane + 32 * i < E ? 1.f / (1.f + expf(-s[i])) : -INFINITY;
    } else {
        mx = moe_warp_max(mx);
        float z = 0.f;
#pragma unroll
        for (int i = 0; i < N; i++) {
            s[i] = lane + 32 * i < E ? expf(s[i] - mx) : 0.f;
            z += s[i];
        }
        z = warp_sum(z);
#pragma unroll
        for (int i = 0; i < N; i++) s[i] = lane + 32 * i < E ? s[i] / z : -INFINITY;
    }
#pragma unroll
    for (int i = 0; i < N; i++) c[i] = lane + 32 * i < E ? (bias ? s[i] + bias[lane + 32 * i] : s[i]) : -INFINITY;
    float sw = 0.f;  // lane r keeps the r-th choice and its score
    int sid = -1;
    for (int r = 0; r < k; r++) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
#pragma unroll
        for (int i = 0; i < N; i++) {
            if (c[i] > bv) {  // (strictly: of a lane's equal ones the smaller i, the smaller expert)
                bv = c[i];
                bi = lane + 32 * i;
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(MOE_FULL, bv, o);
            const int oi = __shfl_xor_sync(MOE_FULL, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
        float mine = 0.f;  // the winner's lane takes its score and leaves the running
#pragma unroll
        for (int i = 0; i < N; i++) {
            if (lane + 32 * i == bi) {
                mine = s[i];
                c[i] = -INFINITY;
            }
        }
        const float sc = __shfl_sync(MOE_FULL, mine, bi & 31);
        if (lane == r) {
            sid = bi < E ? bi : -1;
            sw = bi < E ? sc : 0.f;
        }
    }
    const float tot = warp_sum(lane < k ? sw : 0.f);
    float w = (flags & MOE_RENORM) ? sw / (tot + 1e-20f) : sw;
    if (escale && sid >= 0) w *= escale[sid];
    w *= scale;
    if (lane < k) {
        ids[(size_t)t * k + lane] = sid;
        wts[(size_t)t * k + lane] = w;
    }
}

// (E up to 256, 512 and 1024 experts: 8, 16 and 32 registers a lane for each of the scores and the ranks; 4 tokens a block of 128 threads.)
extern "C" __global__ void __launch_bounds__(128) moe_select8(const float* logits, const float* bias, const float* escale, long long* ids, float* wts, int T, int E, int k, int flags, float scale) {
    moe_select_t<8>(logits, bias, escale, ids, wts, T, E, k, flags, scale);
}
extern "C" __global__ void __launch_bounds__(128) moe_select16(const float* logits, const float* bias, const float* escale, long long* ids, float* wts, int T, int E, int k, int flags, float scale) {
    moe_select_t<16>(logits, bias, escale, ids, wts, T, E, k, flags, scale);
}
extern "C" __global__ void __launch_bounds__(128) moe_select32(const float* logits, const float* bias, const float* escale, long long* ids, float* wts, int T, int E, int k, int flags, float scale) {
    moe_select_t<32>(logits, bias, escale, ids, wts, T, E, k, flags, scale);
}

// ---- between the experts' two products, and after them ------------------------------------------------------------------------------------
// out [rows, I]: MOE_SILU or MOE_GELU (tanh): act(min(g, lim)) * clamp(u, -lim, lim) of gu [rows, 2 I] (the gate in the first I columns, the up in the next I; lim = INFINITY: no clamp, GLM-5.x's is 10);
// MOE_SITU: lim tanh(g / lim) sigmoid(g) x lin tanh(u / lin) of the same halves; MOE_RELU2: relu(x)^2 of gu [rows, I] (in place where out = gu). One rounding. I a multiple of 8; a thread takes
// 8 columns, block (row, chunk of 256 threads' columns).
extern "C" __global__ void moe_act(const bf16* gu, bf16* out, int I, int mode, float lim, float lin) {
    const int i = (blockIdx.y * blockDim.x + threadIdx.x) * 8;
    if (i >= I) return;
    const size_t r = blockIdx.x;
    uint4 ov;
    bf16* op = reinterpret_cast<bf16*>(&ov);
    if (mode == MOE_RELU2) {
        const uint4 xv = *reinterpret_cast<const uint4*>(gu + r * I + i);
        const bf16* xp = reinterpret_cast<const bf16*>(&xv);
#pragma unroll
        for (int k = 0; k < 8; k++) {
            const float x = fmaxf(b2f(xp[k]), 0.f);
            op[k] = f2b(x * x);
        }
    } else {
        const uint4 gv = *reinterpret_cast<const uint4*>(gu + r * 2 * I + i);
        const uint4 uv = *reinterpret_cast<const uint4*>(gu + r * 2 * I + I + i);
        const bf16* gp = reinterpret_cast<const bf16*>(&gv);
        const bf16* up = reinterpret_cast<const bf16*>(&uv);
#pragma unroll
        for (int k = 0; k < 8; k++) {
            if (mode == MOE_SITU) {
                const float g = b2f(gp[k]), u = b2f(up[k]);
                op[k] = f2b(lim * tanhf(g / lim) / (1.f + expf(-g)) * lin * tanhf(u / lin));
                continue;
            }
            const float g = fminf(b2f(gp[k]), lim), u = fminf(fmaxf(b2f(up[k]), -lim), lim);
            const float a = mode == MOE_GELU ? 0.5f * g * (1.f + tanhf(0.7978845608028654f * (g + 0.044715f * g * g * g))) : g / (1.f + expf(-g));
            op[k] = f2b(a * u);
        }
    }
    *reinterpret_cast<uint4*>(out + r * I + i) = ov;
}

// y [n] (bf16) += x [n] (fp32): the sums the host made of the pairs it computed, added to the routed sum of the GPU's (a block layout's, already rounded once).
extern "C" __global__ void moe_add_f32(bf16* y, const float* x, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = f2b(b2f(y[i]) + x[i]);
}

// The heads of an FP8 stack's places in the expert cache (src/f8.rs F8Places): n heads of 8 words, `per` a place, each place's var `vmax` words after the one before. A head the store copied into place s
// holds its base from the start of its expert's own var (less than vmax); it becomes base % vmax + s vmax, which a head already set keeps: every place's heads are its own before the products, whichever were copied.
extern "C" __global__ void moe_f8_rebase(unsigned* heads, long long n, long long per, long long vmax) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) heads[i * 8] = (unsigned)((long long)heads[i * 8] % vmax + i / per * vmax);
}

// y [rows, H] += g * sh [rows, H], g = sigmoid(gate[row]) where gate (fp32 [rows], moe_logits' with E = 1) is given, else 1: the shared expert's output added to the routed experts' sum, one rounding.
// H a multiple of 8; block (row, chunk of 256 threads' columns).
extern "C" __global__ void moe_add(bf16* y, const bf16* sh, const float* gate, int H) {
    const int i = (blockIdx.y * blockDim.x + threadIdx.x) * 8;
    if (i >= H) return;
    const size_t at = (size_t)blockIdx.x * H + i;
    const float g = gate ? 1.f / (1.f + expf(-gate[blockIdx.x])) : 1.f;
    const uint4 av = *reinterpret_cast<const uint4*>(y + at), bv = *reinterpret_cast<const uint4*>(sh + at);
    const bf16* ap = reinterpret_cast<const bf16*>(&av);
    const bf16* bp = reinterpret_cast<const bf16*>(&bv);
    uint4 ov;
    bf16* op = reinterpret_cast<bf16*>(&ov);
#pragma unroll
    for (int k = 0; k < 8; k++) op[k] = f2b(b2f(ap[k]) + g * b2f(bp[k]));
    *reinterpret_cast<uint4*>(y + at) = ov;
}

// The experts' sum where each pair's row is its own (the FP8 experts' second product, src/f8.rs): inv[pair] = its row in the plan's order (plan [2 + 2E + P]: glyd_gpu_moe_way's), then
// y [T, H] = each token's rows [P, H] times its choices' weights wts [T k], summed in fp32 in the order of its choices, rounded once. A thread a pair; a thread 8 columns of a token.
extern "C" __global__ void moe_inv(const int* plan, int* inv, int P, int E) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < P) inv[plan[2 + 2 * E + r]] = r;
}
extern "C" __global__ void moe_wsum(const bf16* rows, const int* inv, const float* wts, bf16* y, int k, int H) {
    const int t = blockIdx.x, c = (blockIdx.y * blockDim.x + threadIdx.x) * 8;
    if (c >= H) return;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int j = 0; j < k; j++) {
        const float w = wts[(size_t)t * k + j];
        const uint4 v = *(const uint4*)(rows + (size_t)inv[t * k + j] * H + c);
        const bf16* b = (const bf16*)&v;
#pragma unroll
        for (int i = 0; i < 8; i++) acc[i] += w * b2f(b[i]);
    }
    uint4 o;
    bf16* ob = (bf16*)&o;
#pragma unroll
    for (int i = 0; i < 8; i++) ob[i] = f2b(acc[i]);
    *(uint4*)(y + (size_t)t * H + c) = o;
}
