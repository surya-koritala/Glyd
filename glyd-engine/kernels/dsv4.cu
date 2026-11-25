// DeepSeek-V4's own kernels (included by engine.cu after glm.cu and moe.cu: they use their helpers; src/dynamic/deepseek_v4.rs describes the model and src/dynamic/mod.rs runs them; its residual streams are
// the streams kind's, residual.cu): the query's and the key's preparation, the compressors (the pooling of every `ratio` tokens into an entry of a cache), the indexer (its queries, its scores and
// the entries it picks), the attention over the window and the picked entries with its sink, the grouped low rank's gather, the activations' FP8 that the release's products quantize their inputs to, and the
// experts' activation with each pair's weight in it (the release weighs an expert's activation before its second product). bf16 in memory, fp32 inside; the rounding points are inference/model.py's and the
// quantizations kernel.py's (src/dsv4_ref.rs: the CPU reference). Positions are read from the device (`pos`, a row's), so a decode step is captured in the engine's CUDA graph. `sim` 0
// leaves out the release's simulated quantizations of the caches and the indexer (a development check's: the arithmetic without their roundings, against the reference's).
// The window's ring and the compressed entries are the executor's rings (src/dynamic/pages.rs): row r of one (a position of the window, an entry of the entries) is tab[r / 16] 16 + r % 16
// where the sequence has a block table, else r (the window's: r modulo its rows).

#define V4_HD 512     // a head's channels, the key-value head's too
#define V4_RD 64      // the rotary channels, a head's last
#define V4_IDX 128    // an indexer head's channels (its Hadamard rotation's size)
#define V4_HG 8       // heads a block of the attention's output takes (a warp each)
#define V4_AQ 8       // warps a head's output takes in v4_attn_out (64 channels each, 2 a lane)
#define V4_SG 16      // heads a block of the attention's scores takes
#define V4_KC 32      // keys a block of the attention's scores takes

// kernel.py's power-of-two scale: 2^ceil(log2(amax x inv)), from the f32's bits.
__device__ __forceinline__ float v4_pow2(float amax, float inv) {
    const uint32_t b = __float_as_uint(amax * inv);
    const int e = (int)((b >> 23) & 0xFF) - 127 + ((b & 0x7FFFFF) != 0);
    return __uint_as_float((uint32_t)(e + 127) << 23);
}

// v (at most 448 in size) to E4M3, to nearest and ties to even.
__device__ __forceinline__ float v4_e4m3(float v) {
    const float a = fabsf(v);
    const int e = max((int)((__float_as_uint(a) >> 23) & 0xFF) - 127, -6);
    const float q = __uint_as_float((uint32_t)(e - 3 + 127) << 23);
    return copysignf(rintf(a / q) * q, v);
}

// v (at most 6 in size) to E2M1, to nearest and ties to even.
__device__ __forceinline__ float v4_e2m1(float v) {
    const float a = fabsf(v);
    const float q = a < 2.f ? 0.5f : (a < 4.f ? 1.f : 2.f);
    return copysignf(rintf(a / q) * q, v);
}

// The largest size over the `lanes` lanes of a group (lanes a power of two, groups aligned).
__device__ __forceinline__ float v4_group_max(float v, int lanes) {
    for (int o = lanes >> 1; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// act_quant in place on the m values of each lane of a group of `lanes` lanes (one block of the quantization): E4M3 at the block's power-of-two scale, back as bf16 values.
__device__ __forceinline__ void v4_fp8_group(float* v, int m, int lanes) {
    float a = 0.f;
    for (int i = 0; i < m; i++) a = fmaxf(a, fabsf(v[i]));
    const float s = v4_pow2(fmaxf(v4_group_max(a, lanes), 1e-4f), 1.f / 448.f);
    for (int i = 0; i < m; i++) v[i] = b2f(f2b(v4_e4m3(fminf(fmaxf(v[i] / s, -448.f), 448.f)) * s));
}

// fp4_act_quant in place, the same for E2M1 (blocks of 32).
__device__ __forceinline__ void v4_fp4_group(float* v, int m, int lanes) {
    float a = 0.f;
    for (int i = 0; i < m; i++) a = fmaxf(a, fabsf(v[i]));
    const float s = v4_pow2(fmaxf(v4_group_max(a, lanes), 6.f * 1.17549435e-38f), 1.f / 6.f);
    for (int i = 0; i < m; i++) v[i] = b2f(f2b(v4_e2m1(fminf(fmaxf(v[i] / s, -6.f), 6.f)) * s));
}

// Row r of a ring: through the sequence's block table (16 rows a block) where it has one, else r modulo the ring's `cap` rows (0: r).
__device__ __forceinline__ size_t v4_place(const int* tab, int r, int cap) { return tab ? (size_t)tab[r >> 4] * 16 + (r & 15) : (size_t)(cap ? r % cap : r); }

// The rotary on a pair (a, b) at the angle whose cosine and sine are given (inverse: by the conjugate), rounded to bf16.
__device__ __forceinline__ void v4_rot(float& a, float& b, float c, float s) {
    const float x = a * c - b * s, y = a * s + b * c;
    a = b2f(f2b(x));
    b = b2f(f2b(y));
}

// ---- the activations' FP8 ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
// y [n, W] = x [n, W] as the release's products read it: each block of `block` columns (64 or 128) rounded to E4M3 at its power-of-two scale (act_quant), bf16 values (y = x is fine). A warp a block of columns:
// block (row, 4 blocks of columns), 128 threads; W a multiple of block.
extern "C" __global__ void v4_fp8(const bf16* x, bf16* y, int W, int block) {
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, b = blockIdx.y * 4 + w, m = block >> 5;
    if (b * block >= W) return;  // (a whole warp)
    const size_t at = (size_t)blockIdx.x * W + (size_t)b * block + lane * m;
    float v[4];
    for (int i = 0; i < m; i++) v[i] = b2f(x[at + i]);
    v4_fp8_group(v, m, 32);
    for (int i = 0; i < m; i++) y[at + i] = f2b(v[i]);
}

// ---- the attention's query and key ------------------------------------------------------------------------------------------------------------------------------------------------------------
// q [n, H, 512] in place: each head normed in the release's bf16 arithmetic (the squares, their mean, the root's reciprocal and the product each rounded), then the rotary on its last 64 channels at the row's
// position (cos and sin [positions, 32] fp32). Block (row, head), 128 threads of 4 channels.
// (`norm` 0: no per-head norm, the rotary alone: `Comp::q_head_norm` false, DeepSeek-V4.1's)
extern "C" __global__ void __launch_bounds__(128) v4_q_prep(bf16* q, const float* cs, const float* sn, const int* pos, float eps, int norm) {
    const int t = blockIdx.x, h = blockIdx.y, H = gridDim.y, c = threadIdx.x * 4;
    bf16* qr = q + ((size_t)t * H + h) * V4_HD;
    float v[4], ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        v[i] = b2f(qr[c + i]);
        ss += b2f(f2b(v[i] * v[i]));
    }
    if (norm) {
        const float mean = b2f(f2b(block_sum(ss) / V4_HD));
        const float r = b2f(f2b(1.f / sqrtf(b2f(f2b(mean + eps)))));
#pragma unroll
        for (int i = 0; i < 4; i++) v[i] = b2f(f2b(v[i] * r));
    }
    if (c >= V4_HD - V4_RD) {
        const float* ct = cs + (size_t)pos[t] * (V4_RD / 2);
        const float* st = sn + (size_t)pos[t] * (V4_RD / 2);
        const int p = (c - (V4_HD - V4_RD)) / 2;
        v4_rot(v[0], v[1], ct[p], st[p]);
        v4_rot(v[2], v[3], ct[p + 1], st[p + 1]);
    }
#pragma unroll
    for (int i = 0; i < 4; i++) qr[c + i] = f2b(v[i]);
}

// The key-value row of each row: kv [n, 512] (the product's) normed with w (rounded once), the rotary on its last 64 channels, its first 448 rounded to E4M3 in blocks of 64 (act_quant, the release's simulation),
// written to the window's ring [cap, 512] at the row's position (through tab). Block a row, 128 threads of 4 channels (a block of 64 channels is 16 lanes).
extern "C" __global__ void __launch_bounds__(128) v4_kv_prep(const bf16* kv, const bf16* w, const float* cs, const float* sn, const int* pos, bf16* ring, int cap, const int* tab, float eps, int sim) {
    const int t = blockIdx.x, c = threadIdx.x * 4;
    const bf16* kr = kv + (size_t)t * V4_HD;
    float v[4], ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        v[i] = b2f(kr[c + i]);
        ss += v[i] * v[i];
    }
    const float r = rsqrtf(block_sum(ss) / V4_HD + eps);
#pragma unroll
    for (int i = 0; i < 4; i++) v[i] = b2f(f2b(b2f(w[c + i]) * (v[i] * r)));
    // (every lane takes part in the groups' shuffles; the rotary's keep their own values)
    float g[4] = {v[0], v[1], v[2], v[3]};
    v4_fp8_group(g, 4, 16);
    if (c >= V4_HD - V4_RD) {
        const float* ct = cs + (size_t)pos[t] * (V4_RD / 2);
        const float* st = sn + (size_t)pos[t] * (V4_RD / 2);
        const int p = (c - (V4_HD - V4_RD)) / 2;
        v4_rot(v[0], v[1], ct[p], st[p]);
        v4_rot(v[2], v[3], ct[p + 1], st[p + 1]);
    } else if (sim) {
        for (int i = 0; i < 4; i++) v[i] = g[i];
    }
    bf16* o = ring + v4_place(tab, pos[t], cap) * V4_HD;
#pragma unroll
    for (int i = 0; i < 4; i++) o[c + i] = f2b(v[i]);
}

// ---- the compressors ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// y [n, R] (fp32) = x [n, K] W^T for a bf16 W [R, K]: the compressors' two products, which the release makes in fp32 (their sums not rounded to bf16). Block (8 rows of W, 8 rows of x), 8 warps: a warp a row of
// W, its 8 sums in registers; x's rows through shared memory 1,024 columns at a time. K a multiple of 256.
extern "C" __global__ void __launch_bounds__(256) v4_lin32(const bf16* x, const bf16* W, float* y, int n, int K, int R) {
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, r = blockIdx.x * 8 + w, t0 = blockIdx.y * 8;
    __shared__ __align__(16) bf16 xs[8][1024];
    float acc[8];
#pragma unroll
    for (int j = 0; j < 8; j++) acc[j] = 0.f;
    for (int k0 = 0; k0 < K; k0 += 1024) {
        const int kc = min(1024, K - k0);
        __syncthreads();
        for (int e = threadIdx.x * 8; e < 8 * 1024; e += 256 * 8) {
            const int j = e / 1024, k = e % 1024;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (t0 + j < n && k < kc) v = *reinterpret_cast<const uint4*>(x + (size_t)(t0 + j) * K + k0 + k);
            *reinterpret_cast<uint4*>(&xs[j][k]) = v;
        }
        __syncthreads();
        if (r < R) {
            for (int k = lane * 8; k < kc; k += 256) {
                const uint4 wv = *reinterpret_cast<const uint4*>(W + (size_t)r * K + k0 + k);
                const uint32_t* wp = reinterpret_cast<const uint32_t*>(&wv);
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const uint4 xv = *reinterpret_cast<const uint4*>(&xs[j][k]);
                    const uint32_t* xp = reinterpret_cast<const uint32_t*>(&xv);
#pragma unroll
                    for (int q = 0; q < 4; q++) {
                        const float2 a = bf2(wp[q]), b = bf2(xp[q]);
                        acc[j] = fmaf(a.x, b.x, fmaf(a.y, b.y, acc[j]));
                    }
                }
            }
        }
    }
    if (r >= R) return;
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const float v = warp_sum(acc[j]);
        if (lane == 0 && t0 + j < n) y[(size_t)(t0 + j) * R + r] = v;
    }
}

// v4_lin32 for one row of x (a decode step): x through shared memory once, a warp a row of W, its loads 8 x 16 bytes a lane at a time, no rounds of 1,024 columns; a lane's columns in v4_lin32's order
// (k = 8 lane + 256 m), each sum as it adds them. K a multiple of 256, at most 8,192.
extern "C" __global__ void __launch_bounds__(256) v4_lin32_1(const bf16* x, const bf16* W, float* y, int K, int R) {
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, r = blockIdx.x * 8 + w;
    __shared__ __align__(16) bf16 xs[8192];
    for (int e = threadIdx.x * 8; e < K; e += 256 * 8) *reinterpret_cast<uint4*>(&xs[e]) = *reinterpret_cast<const uint4*>(x + e);
    __syncthreads();
    if (r >= R) return;
    const bf16* wr = W + (size_t)r * K + lane * 8;
    float acc = 0.f;
    for (int m0 = 0; m0 < K / 256; m0 += 8) {
        uint4 wv[8];
#pragma unroll
        for (int u = 0; u < 8; u++)
            if (m0 + u < K / 256) wv[u] = *reinterpret_cast<const uint4*>(wr + (size_t)(m0 + u) * 256);
#pragma unroll
        for (int u = 0; u < 8; u++) {
            if (m0 + u >= K / 256) break;
            const uint4 xv = *reinterpret_cast<const uint4*>(&xs[(m0 + u) * 256 + lane * 8]);
            const uint32_t* wp = reinterpret_cast<const uint32_t*>(&wv[u]);
            const uint32_t* xp = reinterpret_cast<const uint32_t*>(&xv);
#pragma unroll
            for (int q = 0; q < 4; q++) {
                const float2 a = bf2(wp[q]), b = bf2(xp[q]);
                acc = fmaf(a.x, b.x, fmaf(a.y, b.y, acc));
            }
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) y[r] = acc;
}

// A compressor's pooling for the n rows of a chunk (their positions in order): proj [n, 2W] (fp32: the kv product's W columns, then the gate's), ape [ratio, W] (fp32); the state (fp32, read and written
// back): kvs and scs [rows, W] (rows = 2 ratio overlapping: the previous window's then the current's; else ratio). Each row goes to its window's place; where it completes a window, the window's entry
// (softmax over its rows of each channel's scores, the kv weighted by it; overlapping: channel c of the previous window's first half and of the current's second half) into pooled [n / ratio + 1, d] (fp32;
// the k-th window completed in the chunk) and, overlapping, the current window's rows become the previous. A thread a channel c of d (its columns c and, overlapping, d + c: all its state is its own).
extern "C" __global__ void v4_comp_pool(const float* proj, const float* ape, float* kvs, float* scs, float* pooled, const int* pos, int n, int ratio, int d, int overlap) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= d) return;
    const int W = overlap ? 2 * d : d, first = pos[0] / ratio;
    for (int t = 0; t < n; t++) {
        const int p = pos[t], m = p % ratio, row = overlap ? ratio + m : m;
        const float* pr = proj + (size_t)t * 2 * W;
        const float* ap = ape + (size_t)m * W;
        kvs[(size_t)row * W + c] = pr[c];
        scs[(size_t)row * W + c] = pr[W + c] + ap[c];
        if (overlap) {
            kvs[(size_t)row * W + d + c] = pr[d + c];
            scs[(size_t)row * W + d + c] = pr[W + d + c] + ap[d + c];
        }
        if ((p + 1) % ratio) continue;
        // (the first window has no previous one: its rows are left out, where the release's state holds -inf scores)
        const int rows = overlap ? 2 * ratio : ratio, j0 = overlap && p + 1 == ratio ? ratio : 0;
        float mx = -INFINITY;
        for (int j = j0; j < rows; j++) mx = fmaxf(mx, scs[(size_t)j * W + (overlap && j >= ratio ? d : 0) + c]);
        float sum = 0.f;
        for (int j = j0; j < rows; j++) sum += expf(scs[(size_t)j * W + (overlap && j >= ratio ? d : 0) + c] - mx);
        float acc = 0.f;
        for (int j = j0; j < rows; j++) {
            const size_t at = (size_t)j * W + (overlap && j >= ratio ? d : 0) + c;
            acc += kvs[at] * (expf(scs[at] - mx) / sum);
        }
        pooled[(size_t)((p + 1) / ratio - 1 - first) * d + c] = acc;
        if (overlap) {
            for (int j = 0; j < ratio; j++) {
                const size_t a = (size_t)j * W, b = (size_t)(ratio + j) * W;
                kvs[a + c] = kvs[b + c];
                kvs[a + d + c] = kvs[b + d + c];
                scs[a + c] = scs[b + c];
                scs[a + d + c] = scs[b + d + c];
            }
        }
    }
}

// An indexer key's channel c (a thread a channel, 128 threads) after its norm: the rotary on its last 64 channels (cosines and sines ct, st at the key's position), the Hadamard rotation, E2M1 in blocks of
// 32 where `sim`; through shared memory hs [128].
__device__ __forceinline__ float v4_idx_tail(float v, float* hs, int c, const float* ct, const float* st, int sim) {
    hs[c] = v;
    __syncthreads();
    if (c >= V4_IDX - V4_RD) {
        const int p = (c - (V4_IDX - V4_RD)) / 2;
        const float a = hs[c & ~1], b = hs[c | 1], cc = ct[p], sv = st[p];
        v = b2f(f2b((c & 1) ? a * sv + b * cc : a * cc - b * sv));
    }
    __syncthreads();
    hs[c] = v;
    for (int h = 1; h < V4_IDX; h *= 2) {
        __syncthreads();
        const float a = hs[c], b = hs[c ^ h];
        __syncthreads();
        hs[c] = (c & h) ? b - a : a + b;
    }
    __syncthreads();
    v = b2f(f2b(hs[c] * 0.08838834764831845f));
    float q4[1] = {v};
    v4_fp4_group(q4, 1, 32);
    return sim ? q4[0] : v;
}

// Each window the chunk completed (block k: the k-th, window g = pos[0] / ratio + k; nothing where the chunk does not complete it): its pooled entry rounded to bf16, normed with w (rounded once), the rotary on
// its last 64 channels at the window's first position, then (mode 0, the attention's: d = 512) its first d - 64 channels to E4M3 in blocks of 64, or (mode 1, the indexer's: d = 128) the Hadamard rotation and
// E2M1 in blocks of 32; into cache [entries, d] at entry g (through tab). Where `raw` ([entries, d], through tab: a ring of the cache's group), the normed entry before the rotary too, exact in
// bf16 (a shared compressed attention's indexer keys are made of it: v4_idx_keys). 128 threads, d / 128 channels each.
extern "C" __global__ void __launch_bounds__(128) v4_comp_out(const float* pooled, const bf16* w, const float* cs, const float* sn, const int* pos, int n, bf16* cache, const int* tab, int ratio, int d, int mode,
                                                              float eps, int sim, bf16* raw) {
    const int k = blockIdx.x, g = pos[0] / ratio + k, m = d / 128, c = threadIdx.x * m;
    if ((g + 1) * ratio - 1 > pos[n - 1]) return;  // (the whole block)
    __shared__ float hs[V4_IDX];
    float v[4], ss = 0.f;
    for (int i = 0; i < m; i++) {
        v[i] = b2f(f2b(pooled[(size_t)k * d + c + i]));
        ss += v[i] * v[i];
    }
    const float r = rsqrtf(block_sum(ss) / d + eps);
    for (int i = 0; i < m; i++) v[i] = b2f(f2b(b2f(w[c + i]) * (v[i] * r)));
    if (raw)
        for (int i = 0; i < m; i++) raw[v4_place(tab, g, 0) * d + c + i] = f2b(v[i]);
    const float* ct = cs + (size_t)g * ratio * (V4_RD / 2);
    const float* st = sn + (size_t)g * ratio * (V4_RD / 2);
    if (mode == 0) {
        float q[4] = {v[0], v[1], v[2], v[3]};
        v4_fp8_group(q, 4, 16);
        if (c >= d - V4_RD) {
            const int p = (c - (d - V4_RD)) / 2;
            v4_rot(v[0], v[1], ct[p], st[p]);
            v4_rot(v[2], v[3], ct[p + 1], st[p + 1]);
        } else if (sim) {
            for (int i = 0; i < 4; i++) v[i] = q[i];
        }
    } else {
        // (d = 128: a channel a thread; the rotary's pairs and the Hadamard's butterflies through shared memory)
        v[0] = v4_idx_tail(v[0], hs, c, ct, st, sim);
    }
    bf16* out = cache + v4_place(tab, g, 0) * d;
    for (int i = 0; i < m; i++) out[c + i] = f2b(v[i]);
}

// The indexer's keys of a shared compressed attention's entries (`Op::IdxKeys`, DeepSeek-V4.1's CSA2): each window the chunk completed (block k, window g = pos[0] / ratio + k), of raw [entries, 512] (its
// compressor's entry normed, before the rotary: v4_comp_out's), by wk [128, 512] (bf16) in fp32 rounded to bf16 (the release's bf16 product), normed with w (rounded once), then as an indexer entry of
// v4_comp_out: the rotary at the window's first position, the Hadamard rotation, E2M1 where `sim`; into keys [entries, 128] at g. Both rings of the entries' group: rows through its table tab. 128
// threads, a channel each.
extern "C" __global__ void __launch_bounds__(128) v4_idx_keys(const bf16* raw, const int* tab, const bf16* wk, const bf16* w, const float* cs, const float* sn, const int* pos, int n, bf16* keys, int ratio, float eps,
                                                              int sim) {
    const int k = blockIdx.x, g = pos[0] / ratio + k, c = threadIdx.x;
    if ((g + 1) * ratio - 1 > pos[n - 1]) return;  // (the whole block)
    __shared__ float xs[V4_HD];
    __shared__ float hs[V4_IDX];
    const bf16* xr = raw + v4_place(tab, g, 0) * V4_HD;
    for (int i = c; i < V4_HD; i += V4_IDX) xs[i] = b2f(xr[i]);
    __syncthreads();
    const bf16* wr = wk + (size_t)c * V4_HD;
    float acc = 0.f;
    for (int i = 0; i < V4_HD; i++) acc += b2f(wr[i]) * xs[i];
    float v = b2f(f2b(acc));
    const float r = rsqrtf(block_sum(v * v) / V4_IDX + eps);
    v = b2f(f2b(b2f(w[c]) * (v * r)));
    v = v4_idx_tail(v, hs, c, cs + (size_t)g * ratio * (V4_RD / 2), sn + (size_t)g * ratio * (V4_RD / 2), sim);
    keys[v4_place(tab, g, 0) * V4_IDX + c] = f2b(v);
}

// ---- the indexer -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// The indexer's queries q [n, IH, 128] in place: the rotary on each head's last 64 channels at the row's position, the Hadamard rotation, E2M1 in blocks of 32. Block (row, head), 128 threads of a channel.
extern "C" __global__ void __launch_bounds__(128) v4_idx_q(bf16* q, const float* cs, const float* sn, const int* pos, int sim) {
    const int t = blockIdx.x, h = blockIdx.y, c = threadIdx.x;
    bf16* qr = q + ((size_t)t * gridDim.y + h) * V4_IDX;
    __shared__ float hs[V4_IDX];
    float v = b2f(qr[c]);
    hs[c] = v;
    __syncthreads();
    if (c >= V4_IDX - V4_RD) {
        const int p = (c - (V4_IDX - V4_RD)) / 2;
        const float a = hs[c & ~1], b = hs[c | 1], cc = cs[(size_t)pos[t] * (V4_RD / 2) + p], sv = sn[(size_t)pos[t] * (V4_RD / 2) + p];
        v = b2f(f2b((c & 1) ? a * sv + b * cc : a * cc - b * sv));
    }
    __syncthreads();
    hs[c] = v;
    for (int hh = 1; hh < V4_IDX; hh *= 2) {
        __syncthreads();
        const float a = hs[c], b = hs[c ^ hh];
        __syncthreads();
        hs[c] = (c & hh) ? b - a : a + b;
    }
    __syncthreads();
    float o[1] = {b2f(f2b(hs[c] * 0.08838834764831845f))};
    const float r = o[0];
    v4_fp4_group(o, 1, 32);
    qr[c] = f2b(sim ? o[0] : r);
}

// (the indexer's scores and its pick of the entries: kernels/mla.cu's `idx_score_bf16` and `idx_pick`, every indexer's)

// ---- the attention ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// The keys row t reads, in order: the window's (positions max(0, p - window + 1) to p, from the ring [cap, 512] through ktab), then the cache's entries (through ctab): the indexer's picks (sel [n, k]
// and cnt [n]: ratio 4), or every entry complete at p (the others), or none (no cache). Their number, and the j-th's row.
__device__ __forceinline__ int v4_keys(int p, int window, int ratio, const int* cnt, int t) {
    const int nw = min(window, p + 1);
    if (ratio == 0) return nw;
    return nw + (cnt ? cnt[t] : (p + 1) / ratio);
}
__device__ __forceinline__ const bf16* v4_key_row(int j, int p, int window, const bf16* ring, int cap, const int* ktab, const bf16* cache, const int* ctab, const int* sel, int k, int t) {
    const int nw = min(window, p + 1);
    if (j < nw) return ring + v4_place(ktab, p - nw + 1 + j, cap) * V4_HD;
    j -= nw;
    return cache + v4_place(ctab, sel ? sel[(size_t)t * k + j] : j, 0) * V4_HD;
}

// The scores q_h . key_j x scale of row t (q [n, H, 512]) into sc [n, H, lmax] (fp32). Block (row, V4_KC keys, V4_SG heads), 8 warps: a warp a key at a time, a lane 16 of its channels, the heads' queries in
// shared memory.
extern "C" __global__ void __launch_bounds__(256) v4_attn_scores(const bf16* q, const bf16* ring, int cap, const int* ktab, const bf16* cache, const int* ctab, const int* sel, const int* cnt, int k, const int* pos,
                                                                 float* sc, int H, int lmax, int window, int ratio, float scale) {
    const int t = blockIdx.x, h0 = blockIdx.z * V4_SG, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int p = pos[t], L = v4_keys(p, window, ratio, cnt, t);
    __shared__ __align__(16) bf16 qs[V4_SG][V4_HD];
    for (int i = threadIdx.x * 8; i < V4_SG * V4_HD; i += 256 * 8) *reinterpret_cast<uint4*>(&qs[0][0] + i) = h0 * V4_HD + i < H * V4_HD ? *reinterpret_cast<const uint4*>(q + ((size_t)t * H + h0) * V4_HD + i) : make_uint4(0u, 0u, 0u, 0u);
    __syncthreads();
    // (a warp's V4_KC / 8 keys loaded at once, their rows' lookups first: one at a time each had waited for two trips to memory)
    constexpr int KW = V4_KC / 8;
    const int jb = blockIdx.y * V4_KC + w, je = min(L, (int)(blockIdx.y + 1) * V4_KC);
    uint4 ka[KW], kb[KW];
#pragma unroll
    for (int u = 0; u < KW; u++) {
        const int j = jb + 8 * u;
        if (j < je) {
            const bf16* kr = v4_key_row(j, p, window, ring, cap, ktab, cache, ctab, sel, k, t) + lane * 16;
            ka[u] = *reinterpret_cast<const uint4*>(kr);
            kb[u] = *reinterpret_cast<const uint4*>(kr + 8);
        }
    }
#pragma unroll
    for (int u = 0; u < KW; u++) {
        const int j = jb + 8 * u;
        if (j >= je) break;
        float kv[16];
        const uint32_t* ap = reinterpret_cast<const uint32_t*>(&ka[u]);
        const uint32_t* bp = reinterpret_cast<const uint32_t*>(&kb[u]);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const float2 x = bf2(ap[i]), y = bf2(bp[i]);
            kv[2 * i] = x.x, kv[2 * i + 1] = x.y, kv[8 + 2 * i] = y.x, kv[8 + 2 * i + 1] = y.y;
        }
        float v[V4_SG];
#pragma unroll
        for (int h = 0; h < V4_SG; h++) {
            // (the query's 16 channels in two 16-byte loads, not 16 of 2 bytes; the products in the same order)
            const uint4 qa = *reinterpret_cast<const uint4*>(&qs[h][lane * 16]), qb = *reinterpret_cast<const uint4*>(&qs[h][lane * 16 + 8]);
            const uint32_t* qp[2] = {reinterpret_cast<const uint32_t*>(&qa), reinterpret_cast<const uint32_t*>(&qb)};
            float s = 0.f;
#pragma unroll
            for (int i = 0; i < 16; i += 2) {
                const float2 qv = bf2(qp[i >> 3][(i & 7) >> 1]);
                s = fmaf(qv.x, kv[i], s);
                s = fmaf(qv.y, kv[i + 1], s);
            }
            v[h] = s;
        }
        // (the 16 heads' sums across the lanes as warp_sum's butterfly pairs them, each lane keeping half of its heads a stage: lanes 2h and 2h + 1 end with head h's, the same additions in
        // the same order, in 16 shuffles where a warp_sum a head took 80)
#pragma unroll
        for (int m = V4_SG, o = 16; m > 1; m >>= 1, o >>= 1) {
            const bool hi = lane & o;
#pragma unroll
            for (int i = 0; i < m / 2; i++) {
                const float give = hi ? v[i] : v[i + m / 2];
                const float keep = hi ? v[i + m / 2] : v[i];
                v[i] = keep + __shfl_xor_sync(0xffffffffu, give, o);
            }
        }
        v[0] += __shfl_xor_sync(0xffffffffu, v[0], 1);
        const int h = lane >> 1;
        if (!(lane & 1) && h0 + h < H) sc[((size_t)t * H + h0 + h) * lmax + j] = v[0] * scale;
    }
}

// Each (row, head)'s softmax over its keys with the sink: the largest score M (of the keys), each score made exp(s - M) in place, and den [n, H] = their sum + exp(sink_h - M). A warp a (row, head).
extern "C" __global__ void v4_attn_soft(float* sc, const float* sink, float* den, const int* pos, const int* cnt, int H, int lmax, int window, int ratio) {
    const int t = blockIdx.x, h = blockIdx.y * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (h >= H) return;
    const int L = v4_keys(pos[t], window, ratio, cnt, t);
    float* s = sc + ((size_t)t * H + h) * lmax;
    float mx = -INFINITY;
    for (int j = lane; j < L; j += 32) mx = fmaxf(mx, s[j]);
    mx = v4_group_max(mx, 32);
    float sum = 0.f;
    for (int j = lane; j < L; j += 32) {
        const float e = expf(s[j] - mx);
        s[j] = e;
        sum += e;
    }
    sum = warp_sum(sum);
    if (lane == 0) den[(size_t)t * H + h] = sum + expf(sink[h] - mx);
}

// The output of row t's heads: sum over keys of bf16(p_j) key_j / den, rounded, the rotary taken off its last 64 channels at the row's position (the conjugate), into o in the grouped low rank's order: head h of
// row t0 + t at ((h / hpg) n + t0 + t) hpg + h % hpg (a group's heads, then its rows: each group's rows one matrix for its product; t0: the first of the rows given, which are
// taken a group of rows at a time). Block (row, 8 / V4_AQ heads), V4_AQ warps a head; the keys through shared memory, 32 a round.
extern "C" __global__ void __launch_bounds__(256) v4_attn_out(const float* pr, const float* den, const bf16* ring, int cap, const int* ktab, const bf16* cache, const int* ctab, const int* sel, const int* cnt, int k,
                                                              const int* pos, const float* cs, const float* sn, bf16* o, int n, int H, int hpg, int lmax, int window, int ratio, int t0) {
    // (a head's 512 channels on V4_AQ warps, 4 a lane: 8 / V4_AQ heads a block; each channel's sum over the keys in their order, as one warp a head summed them)
    constexpr int C = V4_HD / V4_AQ / 32;  // (a lane's channels)
    const int t = blockIdx.x, w = threadIdx.x >> 5, lane = threadIdx.x & 31, h = blockIdx.y * (8 / V4_AQ) + w / V4_AQ, d0 = (w % V4_AQ) * (V4_HD / V4_AQ) + lane * C;
    const int p = pos[t], L = v4_keys(p, window, ratio, cnt, t);
    // (keys a round through shared memory, with their probabilities; the next round's loads issued before this one's sums, so that a round waits on less than its two trips to memory, the
    // lookup of a picked key's row and the row)
    constexpr int R = 32, PT = R * V4_HD / 8 / 256;
    __shared__ __align__(16) bf16 ks[R][V4_HD];
    __shared__ float ps[R];
    float acc[C];
#pragma unroll
    for (int i = 0; i < C; i++) acc[i] = 0.f;
    const float* ph = pr + ((size_t)t * H + min(h, H - 1)) * lmax;
    uint4 nx[PT];
    float np = 0.f;
    const int c8 = (threadIdx.x & 63) * 8;
    auto fetch = [&](int j0) {
#pragma unroll
        for (int q = 0; q < PT; q++) {
            const int jj = q * 4 + (threadIdx.x >> 6);
            if (j0 + jj < L) nx[q] = *reinterpret_cast<const uint4*>(v4_key_row(j0 + jj, p, window, ring, cap, ktab, cache, ctab, sel, k, t) + c8);
        }
        if (threadIdx.x < R && j0 + (int)threadIdx.x < L) np = ph[j0 + threadIdx.x];
    };
    fetch(0);
    for (int j0 = 0; j0 < L; j0 += R) {
        __syncthreads();
#pragma unroll
        for (int q = 0; q < PT; q++) {
            const int jj = q * 4 + (threadIdx.x >> 6);
            if (j0 + jj < L) *reinterpret_cast<uint4*>(&ks[jj][c8]) = nx[q];
        }
        if (threadIdx.x < R) ps[threadIdx.x] = np;
        __syncthreads();
        if (j0 + R < L) fetch(j0 + R);
        if (h < H) {
            const int n = min(R, L - j0);
            for (int jj = 0; jj < n; jj++) {
                const float pb = b2f(f2b(ps[jj]));
                const bf16* kb = &ks[jj][d0];
#pragma unroll
                for (int i = 0; i < C; i++) acc[i] = fmaf(pb, b2f(kb[i]), acc[i]);
            }
        }
    }
    if (h >= H) return;
    const float dn = den[(size_t)t * H + h];
    float v[C];
#pragma unroll
    for (int i = 0; i < C; i++) v[i] = b2f(f2b(acc[i] / dn));
    if (d0 >= V4_HD - V4_RD) {
        const float* ct = cs + (size_t)p * (V4_RD / 2);
        const float* st = sn + (size_t)p * (V4_RD / 2);
        const int p0 = (d0 - (V4_HD - V4_RD)) / 2;
#pragma unroll
        for (int i = 0; i < C / 2; i++) v4_rot(v[2 * i], v[2 * i + 1], ct[p0 + i], -st[p0 + i]);
    }
    bf16* orow = o + (((size_t)(h / hpg) * n + t0 + t) * hpg + h % hpg) * V4_HD + d0;
#pragma unroll
    for (int i = 0; i < C; i++) orow[i] = f2b(v[i]);
}

// The grouped low rank's products back in rows: mid [G, n, ol] (a group's products of its rows) into out [n, G ol], each block of 128 of a row rounded to E4M3 where `fp8` (the next product's input as the
// release quantizes it). A warp a block of 128: block (row, 4 blocks), 128 threads; ol a multiple of 128.
extern "C" __global__ void v4_ogather(const bf16* mid, bf16* out, int n, int G, int ol, int fp8) {
    const int t = blockIdx.x, w = threadIdx.x >> 5, lane = threadIdx.x & 31, b = blockIdx.y * 4 + w;
    if (b * 128 >= G * ol) return;
    const int g = (b * 128) / ol, c = (b * 128) % ol + lane * 4;
    const bf16* src = mid + ((size_t)g * n + t) * ol + c;
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; i++) v[i] = b2f(src[i]);
    if (fp8) v4_fp8_group(v, 4, 32);
    bf16* dst = out + (size_t)t * G * ol + (size_t)b * 128 + lane * 4;
#pragma unroll
    for (int i = 0; i < 4; i++) dst[i] = f2b(v[i]);
}

// ---- the experts -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// out [rows, I] = the experts' activation with each pair's weight in it, as the release computes it: bf16(w silu(min(g, lim)) clamp(u, -lim, lim)) of gu [rows, 2 I] (gate then up), then where `fp8` each block
// of 128 rounded to E4M3 (the second product's input). Row r is the plan's r-th pair (plan [2 + 2E + P]: the pairs sorted by expert, glyd_gpu_moe_way's) and w its weight wts[pair]; no plan: w = 1 (the shared
// expert). A row past the pairs the plan routes is none (a call in waves routes a wave's pairs alone: the plan's order holds nothing past them) and is left as it is. A warp a block of 128:
// block (row, 4 blocks), 128 threads; I a multiple of 128.
extern "C" __global__ void v4_moe_act(const bf16* gu, bf16* out, const int* plan, const float* wts, int E, int I, float lim, int fp8) {
    const int r = blockIdx.x, w = threadIdx.x >> 5, lane = threadIdx.x & 31, b = blockIdx.y * 4 + w;
    if (b * 128 >= I || (plan && r >= plan[1 + E + plan[0]])) return;
    const float wt = plan ? wts[plan[2 + 2 * E + r]] : 1.f;
    const int c = b * 128 + lane * 4;
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const float g = fminf(b2f(gu[(size_t)r * 2 * I + c + i]), lim), u = fminf(fmaxf(b2f(gu[(size_t)r * 2 * I + I + c + i]), -lim), lim);
        v[i] = b2f(f2b(wt * (g / (1.f + expf(-g)) * u)));
    }
    if (fp8) v4_fp8_group(v, 4, 32);
#pragma unroll
    for (int i = 0; i < 4; i++) out[(size_t)r * I + c + i] = f2b(v[i]);
}

// The first layers' routing by token: token t's k experts are tid2eid[tok[t]] (int64 [vocab, k]), their weights their scores sqrt(softplus(logit)) divided by their sum, times scale. A thread a token.
extern "C" __global__ void v4_hash_select(const float* logits, const long long* tid2eid, const int* tok, long long* ids, float* wts, int T, int E, int k, float scale) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    float s[32], sum = 0.f;
    for (int j = 0; j < k; j++) {
        const long long e = tid2eid[(size_t)tok[t] * k + j];
        const float l = logits[(size_t)t * E + e];
        s[j] = sqrtf(l > 20.f ? l : log1pf(expf(l)));
        sum += s[j];
        ids[(size_t)t * k + j] = e;
    }
    for (int j = 0; j < k; j++) wts[(size_t)t * k + j] = s[j] / sum * scale;
}
