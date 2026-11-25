// Nemotron 3 Super's own kernels (included by engine.cu after qwen35.cu; src/model_nh.rs runs them): the Mamba-2 layer (the causal convolution with its bias, the selective state
// update as a recurrence and in chunks by the state-space duality, the gated norm over groups) and the attention layer at head 128 without any positional encoding (the keys and values
// into the cache, a decode step's attention). bf16 in memory, fp32 inside; the rounding points are vLLM's (nemotron_h.py, mamba_mixer2.py). Reference: tools/nemotron_ref.py.
// Shapes: a head has NH_P = 64 channels, the state NH_N = 128 per channel; B and C are shared by the nh / ng heads of a group. The in_proj's output rows are [z | x B C | dt]; the convolution's
// output rows are [x (nh 64) | B (ng 128) | C (ng 128)].

#define NH_P 64    // channels of a head
#define NH_N 128   // the state's size
#define NH_L 128   // tokens a chunk of the chunked scan

// ---- the convolution ---------------------------------------------------------------------------------------------------------------------------------------------------------------
// The causal depthwise convolution of kernel 4 over the first C channels of the rows x [n, xs], plus the channel's bias, then SiLU (fp32, one rounding): w [C, 4] (the oldest tap first),
// st [3, C] the channel's last three inputs (the oldest first), read and left holding the last three of these rows; y [n, C]. A thread a channel.
// ponytail: a thread a channel and the tokens one after another is 40 blocks and a prompt's latency of its length; split the tokens over blocks (each taking the three before its first as its
// state) when a profile shows it.
__device__ __forceinline__ void nh_conv_of(const bf16* x, int xs, const bf16* w, const bf16* bias, bf16* st, bf16* y, int n, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const float w0 = b2f(w[c * 4]), w1 = b2f(w[c * 4 + 1]), w2 = b2f(w[c * 4 + 2]), w3 = b2f(w[c * 4 + 3]), bb = b2f(bias[c]);
    float s0 = b2f(st[c]), s1 = b2f(st[C + c]), s2 = b2f(st[2 * C + c]);
    for (int t = 0; t < n; t++) {
        const float xv = b2f(x[(size_t)t * xs + c]);
        float acc = fmaf(w0, s0, bb);
        acc = fmaf(w1, s1, acc);
        acc = fmaf(w2, s2, acc);
        acc = fmaf(w3, xv, acc);
        y[(size_t)t * C + c] = f2b(acc / (1.f + expf(-acc)));
        s0 = s1;
        s1 = s2;
        s2 = xv;
    }
    st[c] = f2b(s0);
    st[C + c] = f2b(s1);
    st[2 * C + c] = f2b(s2);
}

extern "C" __global__ void nh_conv(const bf16* x, int xs, const bf16* w, const bf16* bias, bf16* st, bf16* y, int n, int C) { nh_conv_of(x, xs, w, bias, st, y, n, C); }

// nh_conv of a step's rows that decode, a token of a sequence each (grid C / 256 by rows): row r's state at sts[r ss] (a table of addresses; 0: a padding row, whose output
// is zeroed and which reads and writes no state), its input row r of x, its output row r of y. Each row's arithmetic is nh_conv's on that sequence's one token.
extern "C" __global__ void nh_conv_rows(const bf16* x, int xs, const bf16* w, const bf16* bias, const unsigned long long* sts, int ss, bf16* y, int C) {
    const int r = blockIdx.y;
    bf16* st = reinterpret_cast<bf16*>(sts[(size_t)r * ss]);
    if (!st) {
        const int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c < C) y[(size_t)r * C + c] = f2b(0.f);
        return;
    }
    nh_conv_of(x + (size_t)r * xs, xs, w, bias, st, y + (size_t)r * C, 1, C);
}

// ---- the selective state update, one token after another ---------------------------------------------------------------------------------------------------------------------------
// For the n rows of a sequence: xc [n, cs] the convolution's output, dtr [n, dts] the dt columns of the in_proj's output (bf16), dt_bias, A_log and D bf16 [nh]; state fp32 [nh, 64, 128] (the
// state S[h][p][i]), read and written back; y [n, nh 64] (bf16, before the gated norm). Per token and head: dt = max(softplus(dt + dt_bias), dt_min), A = -exp(A_log),
// S = S exp(A dt) + dt x B^T, y = S C + D x (B and C of the head's group). A warp a channel of a head (grid nh x 64 / warps), a lane 4 of the state's 128.
__device__ __forceinline__ void nh_ssm_of(const bf16* xc, int cs, const bf16* dtr, int dts, const bf16* dt_bias, const bf16* A_log, const bf16* Dp, float* state, bf16* y, int n, int nh,
                                          int ng, float dt_min) {
    const int h = blockIdx.x, lane = threadIdx.x & 31, p = blockIdx.y * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int di = nh * NH_P, g = h / (nh / ng);
    float* st = state + ((size_t)h * NH_P + p) * NH_N + lane * 4;
    float4 s = *reinterpret_cast<const float4*>(st);
    const float Ah = -expf(b2f(A_log[h])), dtb = b2f(dt_bias[h]), Dh = b2f(Dp[h]);
    for (int t = 0; t < n; t++) {
        const bf16* row = xc + (size_t)t * cs;
        const float xv = b2f(row[h * NH_P + p]);
        const uint2 bq = *reinterpret_cast<const uint2*>(row + di + g * NH_N + lane * 4);
        const uint2 cq = *reinterpret_cast<const uint2*>(row + di + ng * NH_N + g * NH_N + lane * 4);
        const float dt = fmaxf(softplus_f(b2f(dtr[(size_t)t * dts + h]) + dtb), dt_min);
        const float dA = expf(Ah * dt), dx = dt * xv;
        const float2 b01 = bf2(bq.x), b23 = bf2(bq.y), c01 = bf2(cq.x), c23 = bf2(cq.y);
        s.x = fmaf(dx, b01.x, s.x * dA);
        s.y = fmaf(dx, b01.y, s.y * dA);
        s.z = fmaf(dx, b23.x, s.z * dA);
        s.w = fmaf(dx, b23.y, s.w * dA);
        float acc = fmaf(s.x, c01.x, 0.f);
        acc = fmaf(s.y, c01.y, acc);
        acc = fmaf(s.z, c23.x, acc);
        acc = fmaf(s.w, c23.y, acc);
        acc = warp_sum(acc);
        if (lane == 0) y[(size_t)t * di + h * NH_P + p] = f2b(fmaf(Dh, xv, acc));
    }
    *reinterpret_cast<float4*>(st) = s;
}

extern "C" __global__ void __launch_bounds__(256) nh_ssm(const bf16* xc, int cs, const bf16* dtr, int dts, const bf16* dt_bias, const bf16* A_log, const bf16* Dp, float* state, bf16* y, int n, int nh,
                                                         int ng, float dt_min) {
    nh_ssm_of(xc, cs, dtr, dts, dt_bias, A_log, Dp, state, y, n, nh, ng, dt_min);
}

// nh_ssm of a step's rows that decode, a token of a sequence each (grid nh by 64 / warps by rows): row r's state at sts[r ss] (0: a padding row, whose output is zeroed and
// which reads and writes no state), its rows of xc, dtr and y at r cs, r dts and r nh 64. Each row's arithmetic is nh_ssm's on that sequence's one token.
extern "C" __global__ void __launch_bounds__(256) nh_ssm_rows(const bf16* xc, int cs, const bf16* dtr, int dts, const bf16* dt_bias, const bf16* A_log, const bf16* Dp,
                                                              const unsigned long long* sts, int ss, bf16* y, int nh, int ng, float dt_min) {
    const int r = blockIdx.z;
    float* st = reinterpret_cast<float*>(sts[(size_t)r * ss]);
    bf16* out = y + (size_t)r * nh * NH_P;
    if (!st) {
        const int p = blockIdx.y * (blockDim.x >> 5) + (threadIdx.x >> 5);
        if ((threadIdx.x & 31) == 0) out[blockIdx.x * NH_P + p] = f2b(0.f);
        return;
    }
    nh_ssm_of(xc + (size_t)r * cs, cs, dtr + (size_t)r * dts, dts, dt_bias, A_log, Dp, st, out, 1, nh, ng, dt_min);
}

// ---- the same in chunks of NH_L tokens (state-space duality: transformers' mamba2_chunk_scan), for a prompt --------------------------------------------------------------------------------
// With a the running sum of A dt inside a chunk (acum), x' = dt x, a chunk's outputs are  y[t] = sum over s <= t of (C_t . B_s) exp(acum_t - acum_s) x'_s  +  exp(acum_t) C_t . S_start  +  D x_t,
// and its state at the end is  exp(acum_last) S_start + sum over s of exp(acum_last - acum_s) x'_s B_s^T. Five kernels: dt and acum of every token; C B^T of every chunk and group; every chunk's
// own state; the states passed from chunk to chunk (the call's initial state in, the final state out); the outputs.

// dtv [n, nh] = max(softplus(dt + dt_bias), dt_min) and acum [n, nh] its running sum times A, restarting at each chunk. A thread a (chunk, head).
extern "C" __global__ void nh_ssd_cum(const bf16* dtr, int dts, const bf16* dt_bias, const bf16* A_log, float* dtv, float* acum, int n, int nh, float dt_min) {
    const int c = blockIdx.x, h = blockIdx.y * blockDim.x + threadIdx.x;
    if (h >= nh) return;
    const float Ah = -expf(b2f(A_log[h])), dtb = b2f(dt_bias[h]);
    const int t1 = min((c + 1) * NH_L, n);
    float a = 0.f;
    for (int t = c * NH_L; t < t1; t++) {
        const float dt = fmaxf(softplus_f(b2f(dtr[(size_t)t * dts + h]) + dtb), dt_min);
        a = fmaf(Ah, dt, a);
        dtv[(size_t)t * nh + h] = dt;
        acum[(size_t)t * nh + h] = a;
    }
}

// G [chunks, ng, NH_L, NH_L] = C_t . B_s for s <= t (0 above the diagonal and past the chunk's tokens), fp32 sums. Block (chunk, group), 256 threads: a thread 8 x 8 of the 128 x 128.
extern "C" __global__ void __launch_bounds__(256) nh_ssd_g(const bf16* xc, int cs, float* G, int n, int nh, int ng) {
    __shared__ float Cs[NH_L][33], Bs[NH_L][33];
    const int c = blockIdx.x, g = blockIdx.y, tid = threadIdx.x, ty = tid >> 4, tx = tid & 15;
    const int t0 = c * NH_L, tc = min(NH_L, n - t0), di = nh * NH_P;
    float acc[8][8];
#pragma unroll
    for (int r = 0; r < 8; r++)
#pragma unroll
        for (int j = 0; j < 8; j++) acc[r][j] = 0.f;
    for (int k0 = 0; k0 < NH_N; k0 += 32) {
        __syncthreads();
        for (int e = tid; e < NH_L * 32; e += 256) {
            const int r = e >> 5, k = e & 31;
            float bv = 0.f, cv = 0.f;
            if (r < tc) {
                const bf16* row = xc + (size_t)(t0 + r) * cs;
                bv = b2f(row[di + g * NH_N + k0 + k]);
                cv = b2f(row[di + ng * NH_N + g * NH_N + k0 + k]);
            }
            Bs[r][k] = bv;
            Cs[r][k] = cv;
        }
        __syncthreads();
        for (int k = 0; k < 32; k++) {
            float a[8], b[8];
#pragma unroll
            for (int r = 0; r < 8; r++) {
                a[r] = Cs[ty + 16 * r][k];
                b[r] = Bs[tx + 16 * r][k];
            }
#pragma unroll
            for (int r = 0; r < 8; r++)
#pragma unroll
                for (int j = 0; j < 8; j++) acc[r][j] = fmaf(a[r], b[j], acc[r][j]);
        }
    }
    float* out = G + (size_t)(c * ng + g) * NH_L * NH_L;
#pragma unroll
    for (int r = 0; r < 8; r++)
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int t = ty + 16 * r, s = tx + 16 * j;
            out[t * NH_L + s] = s <= t ? acc[r][j] : 0.f;
        }
}

// stb [chunks, nh, 64, 128] = every chunk's own state: sum over its tokens s of exp(acum_last - acum_s) dt_s x_s B_s^T (fp32). Block (chunk, head), 256 threads: a thread 4 channels x 8 of the state.
extern "C" __global__ void __launch_bounds__(256) nh_ssd_states(const bf16* xc, int cs, const float* dtv, const float* acum, float* stb, int n, int nh, int ng) {
    __shared__ __align__(16) float Xw[32][NH_P];
    __shared__ float Bs[32][NH_N];
    const int c = blockIdx.x, h = blockIdx.y, g = h / (nh / ng), tid = threadIdx.x, ty = tid >> 4, tx = tid & 15;
    const int t0 = c * NH_L, tc = min(NH_L, n - t0), di = nh * NH_P;
    const float alast = acum[(size_t)(t0 + tc - 1) * nh + h];
    float acc[4][8];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 8; j++) acc[i][j] = 0.f;
    for (int s0 = 0; s0 < NH_L; s0 += 32) {
        __syncthreads();
        for (int e = tid; e < 32 * NH_P; e += 256) {
            const int s = e >> 6, p = e & 63;
            float v = 0.f;
            if (s0 + s < tc) {
                const size_t t = (size_t)(t0 + s0 + s);
                v = b2f(xc[t * cs + h * NH_P + p]) * dtv[t * nh + h] * expf(alast - acum[t * nh + h]);
            }
            Xw[s][p] = v;
        }
        for (int e = tid; e < 32 * NH_N; e += 256) {
            const int s = e >> 7, i = e & 127;
            Bs[s][i] = (s0 + s < tc) ? b2f(xc[(size_t)(t0 + s0 + s) * cs + di + g * NH_N + i]) : 0.f;
        }
        __syncthreads();
        for (int s = 0; s < 32; s++) {
            float a[4], b[8];
#pragma unroll
            for (int i = 0; i < 4; i++) a[i] = Xw[s][ty * 4 + i];
#pragma unroll
            for (int j = 0; j < 8; j++) b[j] = Bs[s][tx + 16 * j];
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 8; j++) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
    }
    float* out = stb + (size_t)(c * nh + h) * NH_P * NH_N;
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 8; j++) out[(ty * 4 + i) * NH_N + tx + 16 * j] = acc[i][j];
}

// The states passed on: chunk c starts from S_c, with S_0 the call's initial state (state) and S_(c+1) = exp(acum_last of chunk c) S_c + chunk c's own state. stb [chunks, nh 64 128] holds the chunks'
// own states in and S_c out (in place); state ends holding the state after the last token. A thread an element of the state.
extern "C" __global__ void nh_ssd_pass(const float* acum, float* stb, float* state, int n, int nh, int nch) {
    const size_t total = (size_t)nh * NH_P * NH_N, e = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= total) return;
    const int h = (int)(e / (NH_P * NH_N));
    float s = state[e];
    for (int c = 0; c < nch; c++) {
        float* slot = stb + (size_t)c * total + e;
        const float own = *slot;
        *slot = s;
        s = fmaf(expf(acum[(size_t)(min((c + 1) * NH_L, n) - 1) * nh + h]), s, own);
    }
    state[e] = s;
}

// y [n, nh 64] (bf16) = the chunks' outputs: block (chunk, head, half of the head's channels), 256 threads: a thread 4 consecutive channels of one token of each block of 32 tokens. The part from the
// chunk's start state (C . S, 32 channels by slices of 32 of the state) times exp(acum_t), the part inside the chunk (a block of 32 rows of G exp(acum_t - acum_s), times x' = dt x), and D x.
extern "C" __global__ void __launch_bounds__(256) nh_ssd_y(const bf16* xc, int cs, const float* dtv, const float* acum, const float* G, const float* stb, const bf16* Dp, bf16* y, int n, int nh, int ng) {
    __shared__ __align__(16) float Xs[NH_L][32];
    __shared__ float Ms[32][NH_L + 1];
    __shared__ float Cs[32][33], Ss[32][33];
    __shared__ float Ac[NH_L];
    const int c = blockIdx.x, h = blockIdx.y, g = h / (nh / ng), tid = threadIdx.x;
    const int tr = tid >> 3, tq = tid & 7;
    const int t0 = c * NH_L, tc = min(NH_L, n - t0), di = nh * NH_P, p0 = blockIdx.z * 32;
    for (int e = tid; e < NH_L; e += 256) Ac[e] = e < tc ? acum[(size_t)(t0 + e) * nh + h] : 0.f;
    for (int e = tid; e < NH_L * 32; e += 256) {
        const int s = e >> 5, p = e & 31;
        Xs[s][p] = s < tc ? b2f(xc[(size_t)(t0 + s) * cs + h * NH_P + p0 + p]) * dtv[(size_t)(t0 + s) * nh + h] : 0.f;
    }
    const float* Gc = G + (size_t)(c * ng + g) * NH_L * NH_L;
    const float* S = stb + ((size_t)(c * nh + h) * NH_P + p0) * NH_N;
    const float Dh = b2f(Dp[h]);
    for (int rb = 0; rb < NH_L; rb += 32) {
        if (rb >= tc) break;
        float off[4] = {0.f, 0.f, 0.f, 0.f}, acc[4] = {0.f, 0.f, 0.f, 0.f};
        for (int n0 = 0; n0 < NH_N; n0 += 32) {
            __syncthreads();
            for (int e = tid; e < 32 * 32; e += 256) {
                const int r = e >> 5, k = e & 31;
                Cs[r][k] = (rb + r < tc) ? b2f(xc[(size_t)(t0 + rb + r) * cs + di + ng * NH_N + g * NH_N + n0 + k]) : 0.f;
                Ss[r][k] = S[(size_t)r * NH_N + n0 + k];
            }
            __syncthreads();
            for (int k = 0; k < 32; k++) {
                const float cv = Cs[tr][k];
#pragma unroll
                for (int j = 0; j < 4; j++) off[j] = fmaf(cv, Ss[tq * 4 + j][k], off[j]);
            }
        }
        const int width = rb + 32;
        __syncthreads();
        for (int e = tid; e < 32 * width; e += 256) {
            const int r = e / width, s = e % width, t = rb + r;
            Ms[r][s] = (s <= t && t < tc) ? Gc[t * NH_L + s] * expf(Ac[t] - Ac[s]) : 0.f;  // (a row past the chunk's tokens has no acum to take a difference of)
        }
        __syncthreads();
        for (int s = 0; s < width; s++) {
            const float m = Ms[tr][s];
            const float4 x4 = *reinterpret_cast<const float4*>(&Xs[s][tq * 4]);
            acc[0] = fmaf(m, x4.x, acc[0]);
            acc[1] = fmaf(m, x4.y, acc[1]);
            acc[2] = fmaf(m, x4.z, acc[2]);
            acc[3] = fmaf(m, x4.w, acc[3]);
        }
        const int t = rb + tr;
        if (t < tc) {
            const float ea = expf(Ac[t]);
            const bf16* xr = xc + (size_t)(t0 + t) * cs + h * NH_P + p0 + tq * 4;
            bf16* yr = y + (size_t)(t0 + t) * di + h * NH_P + p0 + tq * 4;
#pragma unroll
            for (int j = 0; j < 4; j++) yr[j] = f2b(fmaf(Dh, b2f(xr[j]), fmaf(ea, off[j], acc[j])));
        }
    }
}

// ---- the gated norm ----------------------------------------------------------------------------------------------------------------------------------------------------------------
// In place on y [n, di] (the scan's bf16 output): y = bf16(w * bf16(norm(y silu(z)))) with the norm over each of ng groups of di / ng channels (fp32, eps inside the root), z at z[t * zs ..] (the in_proj's
// gate columns): vLLM's Mixer2RMSNormGated rounds the normed value to bf16 before the weight. Block (row, group), 256 threads, di / ng at most 1024.
extern "C" __global__ void nh_gated_norm(bf16* y, const bf16* z, int zs, const bf16* w, int di, int ng, float eps) {
    const int t = blockIdx.x, g = blockIdx.y, gs = di / ng;
    bf16* yr = y + (size_t)t * di + g * gs;
    const bf16* zr = z + (size_t)t * zs + g * gs;
    float v[4];
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const int i = threadIdx.x + j * 256;
        float a = 0.f;
        if (i < gs) {
            const float zz = b2f(zr[i]);
            a = b2f(yr[i]) * (zz / (1.f + expf(-zz)));
        }
        v[j] = a;
        acc += a * a;
    }
    const float r = rsqrtf(block_sum(acc) / gs + eps);
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const int i = threadIdx.x + j * 256;
        if (i < gs) yr[i] = f2b(b2f(w[g * gs + i]) * b2f(f2b(v[j] * r)));
    }
}
