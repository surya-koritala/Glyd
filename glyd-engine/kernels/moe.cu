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
#define MOE_SILU_R 4   // silu(gate) rounded to bf16, then times up (transformers' act_fn on a bf16 tensor, then the product: Muse Spark's), no clamp

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
// MOE_SITU: lim tanh(g / lim) sigmoid(g) x lin tanh(u / lin) of the same halves; MOE_RELU2: relu(x)^2 of gu [rows, I] (in place where out = gu). One rounding (MOE_SILU_R: silu(g) rounded before the
// product too). I a multiple of 8; a thread takes
// 8 columns, block (row, chunk of 256 threads' columns). With an experts' plan (`plan` of E experts, not null) the rows are its order's: a row past the pairs it routes is none (no
// product wrote it), as in v4_moe_act.
extern "C" __global__ void moe_act(const bf16* gu, bf16* out, int I, int mode, float lim, float lin, const int* plan, int E) {
    const int i = (blockIdx.y * blockDim.x + threadIdx.x) * 8;
    if (i >= I) return;
    const size_t r = blockIdx.x;
    if (plan && (int)r >= plan[1 + E + plan[0]]) return;
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
            if (mode == MOE_SILU_R) {
                const float g = b2f(gp[k]), u = b2f(up[k]);
                op[k] = f2b(b2f(f2b(g / (1.f + expf(-g)))) * u);
                continue;
            }
            const float g = fminf(b2f(gp[k]), lim), u = fminf(fmaxf(b2f(up[k]), -lim), lim);
            const float a = mode == MOE_GELU ? 0.5f * g * (1.f + tanhf(0.7978845608028654f * (g + 0.044715f * g * g * g))) : g / (1.f + expf(-g));
            op[k] = f2b(a * u);
        }
    }
    *reinterpret_cast<uint4*>(out + r * I + i) = ov;
}

// y [n] (bf16) += x [n] (fp32): the sums the host made of the pairs it computed, added to the routed sum of the GPU's (a block layout's, already rounded once: a GPU's slice of a layer across
// GPUs, whose rows a gather put together).
extern "C" __global__ void moe_add_f32(bf16* y, const float* x, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = f2b(b2f(y[i]) + x[i]);
}

// The routed sum of a block layout's weighted second product and the host's: y [T, O] = each token's pairs that `ids` [T k] names (an id below E) added in the order of its choices in fp32 from
// their weighted rows y32 [T k, O] (the product's, kept in its workspace), as the product's own sum adds them (the library's moe_sum_body), then the host's sum of the token's pairs it computed
// (x [T, O], fp32), rounded once: not the product's sum rounded and the host's added and rounded again. 4 columns a thread; O a multiple of 4.
extern "C" __global__ void moe_sum_host(const float* y32, const long long* ids, int E, long long T, long long k, long long O, const float* x, bf16* y) {
    const long long i = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i >= T * O) return;
    const long long tok = i / O, c = i - tok * O;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    for (long long j = tok * k; j < tok * k + k; j++) {
        const long long e = ids[j];
        if (e < 0 || e >= E) continue;
        const float4 u = *reinterpret_cast<const float4*>(y32 + j * O + c);
        v.x += u.x;
        v.y += u.y;
        v.z += u.z;
        v.w += u.w;
    }
    const float4 h = *reinterpret_cast<const float4*>(x + i);
    y[i] = f2b(v.x + h.x);
    y[i + 1] = f2b(v.y + h.y);
    y[i + 2] = f2b(v.z + h.z);
    y[i + 3] = f2b(v.w + h.w);
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
    if (r < P && r < plan[1 + E + plan[0]]) inv[plan[2 + 2 * E + r]] = r;  // (a row past the routed pairs is none)
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

// ---- the expert cache planned on the GPU (src/devplan.rs, `DevPlan`): a decode call's choices looked up in its layer's table of places, each miss either copied into a place or sent to
// the host's thread, so that no host waits inside a call and a step of such calls is one graph. One block: the tables in shared memory while the first thread decides, then written back.
// A layer's table: place_of [experts] int, expert_of [slots] int, last [slots] uint (the step a place was last used in), hostlast [experts] uint (the step the host last computed an expert), refb [slots]
// uchar; meta [16] uint: 0 step, 1 hand, 2 experts asked for, 3 found, 4 copied, 5 the host's experts, 6 the host's pairs, 7 promoted (counted over the calls), 8 this call's copies, 9 its promotions,
// 10 its host pairs, 11 its request's number (0: none). params [8] words: the times of a copy, of the host's call's fixed part, of a miss's first row and of each further row (float, microseconds: the
// store's `Costs`), the mode (0 copy, 1 cpu, 2 split, 3 auto), the window, the most promotions a call. The misses are copied as the store's `plan` copies them: the ones with the most pairs (the hot ones first
// among equals), as many as make the longer of the copies and the host's pairs shortest (auto; split: the two the same), within the places the call's hits leave; a hot miss (the host computed it within
// `window` calls before) that stays the host's is promoted (a place found for it, filled behind the call's products) while the copies so far leave the host's time room for its copy.
// The request (mapped host memory the host's thread reads, src/devplan.rs `DevMail`): words 0 its number (written last), 1 the layer, 2 tokens, 3 pairs, 4 z's width, 5 choices a token; from word XP_HDR
// the host's pairs (expert, token, weight's bits), `max_pairs` of room; after them the tokens' rows of z (bf16 [tokens, width]).
#define XP_MAX 1024
#define XP_HDR 16
#define XP_CMAX 2048

// A place for expert e: the first the clock hand comes to that no choice of this step uses and that was not used since the hand last passed it (its reference cleared as it passes); -1 where none is.
__device__ __forceinline__ int xp_place(int* sp, int* se, unsigned* sl, unsigned char* sr, int slots, unsigned* hand, unsigned step, int e) {
    for (int guard = 0; guard < 2 * slots + 2; guard++) {
        const int c = (int)*hand;
        *hand = *hand + 1 == (unsigned)slots ? 0 : *hand + 1;
        if (sl[c] == step) continue;
        if (sr[c]) {
            sr[c] = 0;
            continue;
        }
        const int old = se[c];
        if (old >= 0) sp[old] = -1;
        se[c] = e;
        sp[e] = c;
        sl[c] = step;
        sr[c] = 1;
        return c;
    }
    return -1;
}

extern "C" __global__ void __launch_bounds__(256) xp_plan(const long long* ids, const float* wts, const bf16* z, int tokens, int k, int width, int* place_of, int* expert_of, unsigned* last,
                                                          unsigned* hostlast, unsigned char* refb, unsigned* meta, const unsigned* params, int experts, int slots, long long* way, int2* fills,
                                                          int2* promos, unsigned* mail, unsigned* seq, int layer, int max_pairs) {
    __shared__ int sp[XP_MAX];
    __shared__ int se[XP_MAX];
    __shared__ unsigned sl[XP_MAX];
    __shared__ unsigned char sr[XP_MAX];
    __shared__ unsigned short cnt[XP_MAX];
    // (each expert's state this call: 0 not asked for, 1 found, 2 copied, 3 the host's, 4 a miss, 5 a miss counted)
    __shared__ unsigned char st[XP_MAX];
    // (the misses by their pairs, and the hot ones among them)
    __shared__ unsigned short nb[XP_CMAX + 1];
    __shared__ unsigned short nhot[XP_CMAX + 1];
    __shared__ int nh_s;
    const int pairs = tokens * k;
    for (int i = threadIdx.x; i < experts; i += blockDim.x) {
        sp[i] = place_of[i];
        cnt[i] = 0;
        st[i] = 0;
    }
    for (int i = threadIdx.x; i < slots; i += blockDim.x) {
        se[i] = expert_of[i];
        sl[i] = last[i];
        sr[i] = refb[i];
    }
    for (int i = threadIdx.x; i <= XP_CMAX; i += blockDim.x) {
        nb[i] = 0;
        nhot[i] = 0;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned step = meta[0] + 1;
        const float cp = __uint_as_float(params[0]), fx = __uint_as_float(params[1]), pr = __uint_as_float(params[2]), rw = __uint_as_float(params[3]);
        const int mode = (int)params[4], window = (int)params[5], max_promote = (int)params[6];
        unsigned hand = meta[1], reads = 0, hits = 0, nf = 0, np = 0, hx = 0, nh = 0;
        for (int p = 0; p < pairs; p++) {
            const long long e = ids[p];
            if (e >= experts) {
                DEV_SAY("xp_plan: choice %lld of %d experts\n", e, experts);
                __trap();
            }
            if (e >= 0) cnt[e]++;
        }
        // the experts in places, in the order the choices name them
        for (int p = 0; p < pairs; p++) {
            const long long e = ids[p];
            if (e < 0 || st[e]) continue;
            reads++;
            const int q = sp[e];
            if (q >= 0) {
                st[e] = 1;
                hits++;
                sl[q] = step;
                sr[q] = 1;
            } else {
                st[e] = 4;
            }
        }
        // the misses by their pairs, the host's time for all of them
        unsigned nmiss = 0;
        int cmax = 0;
        float left = 0.f;
        for (int p = 0; p < pairs; p++) {
            const long long e = ids[p];
            if (e < 0 || st[e] != 4) continue;
            st[e] = 5;
            const int c = min((int)cnt[e], XP_CMAX);
            nb[c]++;
            if (hostlast[e] && step - hostlast[e] <= (unsigned)window) nhot[c]++;
            nmiss++;
            cmax = max(cmax, c);
            left += pr + rw * (float)(c - 1);
        }
        // how many to copy: the misses with the most pairs first, as many as the mode's score is best for (the store's `plan`); copied: more pairs than `cstar`, or `cstar` and among its first `mstar`
        const unsigned room = (unsigned)slots - hits;
        int cstar = XP_CMAX + 1, mstar = 0;
        float copying = 0.f, computing = nmiss ? fx + left : 0.f;
        if (mode == 0) {
            cstar = 0;
        } else if (mode != 1 && nmiss) {
            float best = mode == 2 ? fabsf(0.f - computing) : computing;
            unsigned taken = 0, kept = nmiss;
            float l = left;
            for (int c = cmax; c >= 1 && taken < room; c--) {
                for (int j = 0; j < (int)nb[c] && taken < room; j++) {
                    taken++;
                    kept--;
                    l -= pr + rw * (float)(c - 1);
                    const float cpy = cp * (float)taken, cmp = kept ? fx + l : 0.f;
                    const float now = mode == 2 ? fabsf(cpy - cmp) : fmaxf(cpy, cmp);
                    if (now < best - 1e-3f) {
                        best = now;
                        cstar = c;
                        mstar = j + 1;
                        copying = cpy;
                        computing = cmp;
                    }
                }
            }
        }
        // (promotions while the host's time leaves room for their copies beside the call's)
        const int np_max = mode == 0 ? 0 : min(max_promote, computing > copying && cp > 0.f ? (int)((computing - copying) / cp) : 0);
        int hot_left = cstar <= XP_CMAX ? min(mstar, (int)nhot[cstar]) : 0;
        int cold_left = cstar <= XP_CMAX ? mstar - hot_left : 0;
        for (int p = 0; p < pairs; p++) {
            const long long e = ids[p];
            if (e < 0 || st[e] != 5) continue;
            const int c = min((int)cnt[e], XP_CMAX);
            const bool hot = hostlast[e] && step - hostlast[e] <= (unsigned)window;
            bool want = mode == 0 || c > cstar;
            if (!want && c == cstar) {
                int& n = hot ? hot_left : cold_left;
                if (n > 0) {
                    n--;
                    want = true;
                }
            }
            const int q = want ? xp_place(sp, se, sl, sr, slots, &hand, step, (int)e) : -1;
            if (q >= 0) {
                st[e] = 2;
                fills[nf++] = make_int2((int)e, q);
                continue;
            }
            st[e] = 3;
            hx++;
            if (hot && (int)np < np_max) {
                const int qp = xp_place(sp, se, sl, sr, slots, &hand, step, (int)e);
                if (qp >= 0) promos[np++] = make_int2((int)e, qp);
            }
            hostlast[e] = step;
        }
        volatile unsigned* m = mail;
        for (int p = 0; p < pairs; p++) {
            const long long e = ids[p];
            if (e < 0 || st[e] != 3) continue;
            if ((int)nh >= max_pairs) {
                DEV_SAY("xp_plan: more than %d pairs for the host\n", max_pairs);
                __trap();
            }
            m[XP_HDR + 3 * nh] = (unsigned)e;
            m[XP_HDR + 3 * nh + 1] = (unsigned)(p / k);
            m[XP_HDR + 3 * nh + 2] = __float_as_uint(wts[p]);
            nh++;
        }
        meta[0] = step;
        meta[1] = hand;
        meta[2] += reads;
        meta[3] += hits;
        meta[4] += nf;
        meta[5] += hx;
        meta[6] += nh;
        meta[7] += np;
        meta[8] = nf;
        meta[9] = np;
        meta[10] = nh;
        nh_s = (int)nh;
    }
    __syncthreads();
    // each pair's place (-1: the host's, or no expert); the tables back to the layer's
    for (int p = threadIdx.x; p < pairs; p += blockDim.x) {
        const long long e = ids[p];
        way[p] = (e >= 0 && (st[e] == 1 || st[e] == 2)) ? (long long)sp[e] : -1;
    }
    for (int i = threadIdx.x; i < experts; i += blockDim.x) place_of[i] = sp[i];
    for (int i = threadIdx.x; i < slots; i += blockDim.x) {
        expert_of[i] = se[i];
        last[i] = sl[i];
        refb[i] = sr[i];
    }
    // the request: the tokens' rows of z beside the pairs (16 bytes at a time where they are aligned), every thread's writes seen before its number
    const int nh = nh_s;
    if (nh > 0) {
        unsigned* zm = mail + XP_HDR + 3 * max_pairs;
        const long long n = (long long)tokens * width;
        if (n % 8 == 0 && (uintptr_t)z % 16 == 0 && (uintptr_t)zm % 16 == 0) {
            volatile uint4* d = (volatile uint4*)zm;
            const uint4* s = (const uint4*)z;
            for (long long i = threadIdx.x; i < n / 8; i += blockDim.x) {
                const uint4 v = s[i];
                d[i].x = v.x;
                d[i].y = v.y;
                d[i].z = v.z;
                d[i].w = v.w;
            }
        } else {
            volatile unsigned short* d = (volatile unsigned short*)zm;
            const unsigned short* s = (const unsigned short*)z;
            for (long long i = threadIdx.x; i < n; i += blockDim.x) d[i] = s[i];
        }
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        if (nh > 0) {
            volatile unsigned* m = mail;
            m[1] = (unsigned)layer;
            m[2] = (unsigned)tokens;
            m[3] = (unsigned)nh;
            m[4] = (unsigned)width;
            m[5] = (unsigned)k;
            __threadfence_system();
            const unsigned s = *seq + 1 ? *seq + 1 : 1;  // (0 is no request)
            *seq = s;
            m[0] = s;
            __threadfence_system();
            meta[11] = s;
        } else {
            meta[11] = 0;
        }
    }
}

// The copies of a plan's list (fills or promotions: (expert, place); `meta[at]` of them): each of the layer's `nreg` regions (u64s: host base, an expert's bytes, place base, a place's bytes; each a multiple of
// 16) read from the host tier through its mapped address into the expert's place. Grid: (blocks a copy, the most copies a list holds); a block past the list's copies does nothing.
extern "C" __global__ void __launch_bounds__(256) xp_fill(const int2* list, const unsigned* meta, int at, const unsigned long long* reg, int nreg) {
    const int f = blockIdx.y;
    if (f >= (int)meta[at]) return;
    const int2 x = list[f];
    const unsigned long long step = (unsigned long long)gridDim.x * blockDim.x;
    for (int r = 0; r < nreg; r++) {
        const unsigned long long* g = reg + 4 * r;
        const uint4* s4 = reinterpret_cast<const uint4*>(g[0] + (unsigned long long)x.x * g[1]);
        uint4* d4 = reinterpret_cast<uint4*>(g[2] + (unsigned long long)x.y * g[3]);
        const unsigned long long n = g[1] / 16;
        unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
        // (four reads in flight a thread before their writes: the link's latency is hidden by many)
        for (; i + 3 * step < n; i += 4 * step) {
            const uint4 a = s4[i], b = s4[i + step], c = s4[i + 2 * step], d = s4[i + 3 * step];
            d4[i] = a;
            d4[i + step] = b;
            d4[i + 2 * step] = c;
            d4[i + 3 * step] = d;
        }
        for (; i < n; i += step) d4[i] = s4[i];
    }
}

// A call's host pairs waited for on the GPU: its request's number (meta[11]; 0: none) read back in the answer's first word (mapped host memory the host's thread writes), the second word 0 (else the
// host's pairs failed: trapped, which the engine reports); at most `limit` of the GPU's clock cycles, then trapped. One thread.
extern "C" __global__ void xp_wait(const unsigned* meta, const unsigned* ans, long long limit) {
    if (threadIdx.x || blockIdx.x) return;
    const unsigned s = meta[11];
    if (!s) return;
    volatile const unsigned* a = ans;
    const long long t0 = clock64();
    while (a[0] != s) {
        if (clock64() - t0 > limit) {
            DEV_SAY("xp_wait: no answer to request %u (the host's last %u)\n", s, a[0]);
            __trap();
        }
    }
    __threadfence_system();
    if (a[1]) {
        DEV_SAY("xp_wait: the host's pairs of request %u failed\n", s);
        __trap();
    }
}

// The host's sums (fp32 [n], the answer's after its 16 words) into `out` on the GPU, zeros where the call sent the host nothing: what the experts' sum adds.
extern "C" __global__ void xp_take(const unsigned* meta, const float* sums, float* out, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = meta[11] ? ((volatile const float*)sums)[i] : 0.f;
}
