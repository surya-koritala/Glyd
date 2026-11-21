// Gemma 4's own kernels (included by engine.cu; the description executor runs them through src/gemma4.rs's table): the norms (the weight as it is, x * w, in fp32 with one rounding), the sandwich residual (the
// sublayer's output normed before it is added), the per-layer scalar, the q/k/v preparation at head 256 (sliding layers, full rotary, theta 1e4) and 512 (global layers,
// the proportional rotary on the first 64 of each half), the causal attention with an optional sliding window over a ring of positions, gelu(tanh) times up, and the final
// soft-cap. bf16 in memory, fp32 inside; the rounding points are transformers' eager ones (models/gemma4/modeling_gemma4.py): a norm rounds once, the rotary rounds each
// product and their sum, the attention rounds its scores and its probabilities, gelu rounds before the product by up.

// ---- the next product's bytes into L2 (GLY-312) -----------------------------------------------------------------------------------------
// The memory is idle while a small kernel (a norm, the q/k/v preparation and the attention after it, gelu) runs; its spare blocks (blockIdx.y or .z above 0) ask L2 for
// the start of each first-wave block of the skinny product that comes next (kernels/skinny.cu: block r's steps at data + r dstride, its scales at scales + r sstride),
// dhead and shead bytes of them (multiples of 128), a prefetch every 2^shift bytes, so that the product starts on L2's copies. data null: nothing.
struct G4Pf {
    const unsigned char* data;
    const unsigned char* scales;
    long long dstride, sstride;
    int dhead, shead, regions, shift;
};

__device__ __forceinline__ void g4_pf(const G4Pf& p, int b, int nb) {
    if (!p.data) return;
    const int dl = p.dhead >> p.shift, per = dl + (p.shead >> p.shift);
    for (int i = b * blockDim.x + threadIdx.x; i < p.regions * per; i += nb * blockDim.x) {
        const int r = i / per, j = i - r * per;
        const unsigned char* a = j < dl ? p.data + r * p.dstride + ((size_t)j << p.shift) : p.scales + r * p.sstride + ((size_t)(j - dl) << p.shift);
#if defined(__HIP_PLATFORM_AMD__)
        (void)a;  // (RDNA has no prefetch into L2)
#else
        asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(a));
#endif
    }
}

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
extern "C" __global__ void g4_add_norm(bf16* x, const bf16* y, const bf16* wpost, const bf16* scalar, const bf16* wnext, bf16* out, int K, float eps, G4Pf pf) {
    if (blockIdx.y > 0) return g4_pf(pf, (blockIdx.y - 1) * gridDim.x + blockIdx.x, (gridDim.y - 1) * gridDim.x);
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

// ---- q, k, v -----------------------------------------------------------------------------------------------------------------------------
// The projection row of a layer [n, stride]: nq query heads, then nk key heads, then (has_v) nk value heads, D values each. A query or key head: rmsnorm over D times its
// weight (qw or kw [D]), one rounding, then the rotary: pairs (i, i + D/2), the first `rot` of them turned by cos and sin (bf16 tables [positions, rot]; the rest as they are,
// transformers' cos 1 and sin 0), each product rounded and their sum rounded: x_i c - x_{i+D/2} s, x_{i+D/2} c + x_i s. A value: rmsnorm over D with no weight, one rounding,
// from its own head, or (no v) from the key head's raw values (Gemma 4's global layers, K = V). Queries to qo [n, nq D]; keys and values to the ring kc, vc [cap, nk, D] at
// slot pos % cap. Block (row, head of nq + nk), D / 2 threads: thread t takes i = t and i + D/2.
// A key's place in a layer's cache: through the sequence's block table where it has one (16-token blocks, each a page of the pool: src/dynamic/pages.rs), else the ring's
// slot p % cap.
__device__ __forceinline__ size_t g4_place(const int* tab, int p, int cap) { return tab ? (size_t)tab[p >> 4] * 16 + (p & 15) : (size_t)(p % cap); }

template <int D>
__device__ __forceinline__ void g4_prep_t(const bf16* qkv, int stride, const bf16* qw, const bf16* kw, const bf16* cs, const bf16* sn, int rot, const int* pos, int cap, bf16* qo, bf16* kc,
                                          bf16* vc, int nq, int nk, int has_v, float eps, const int* tab) {
    constexpr int H = D / 2;
    const int r = blockIdx.x, hh = blockIdx.y, t = threadIdx.x;
    const bf16* row = qkv + (size_t)r * stride;
    const int p = pos[r];
    const bool isq = hh < nq;
    const int kh = hh - nq;
    const bf16* x = isq ? row + hh * D : row + nq * D + kh * D;
    float a = b2f(x[t]), b = b2f(x[t + H]);
    const float s = g4_scale(block_sum(a * a + b * b), D, eps);
    const bf16* w = isq ? qw : kw;
    const float na = b2f(f2b(a * s * b2f(w[t]))), nb = b2f(f2b(b * s * b2f(w[t + H])));
    float oa = na, ob = nb;
    if (t < rot) {
        const float c = b2f(cs[(size_t)p * rot + t]), sv = b2f(sn[(size_t)p * rot + t]);
        oa = b2f(f2b(b2f(f2b(na * c)) - b2f(f2b(nb * sv))));
        ob = b2f(f2b(b2f(f2b(nb * c)) + b2f(f2b(na * sv))));
    }
    const size_t slot = g4_place(tab, p, cap);
    bf16* o = isq ? qo + (size_t)r * nq * D + hh * D : kc + (slot * nk + kh) * D;
    o[t] = f2b(oa);
    o[t + H] = f2b(ob);
    if (isq) return;
    // the value of this key-value head
    const bf16* xv = has_v ? row + nq * D + nk * D + kh * D : x;
    const float va = b2f(xv[t]), vb = b2f(xv[t + H]);
    const float sv2 = g4_scale(block_sum(va * va + vb * vb), D, eps);
    bf16* ov = vc + (slot * nk + kh) * D;
    ov[t] = f2b(va * sv2);
    ov[t + H] = f2b(vb * sv2);
}

extern "C" __global__ void __launch_bounds__(128) g4_prep256(const bf16* qkv, int stride, const bf16* qw, const bf16* kw, const bf16* cs, const bf16* sn, int rot, const int* pos, int cap, bf16* qo,
                                                            bf16* kc, bf16* vc, int nq, int nk, int has_v, float eps, G4Pf pf, const int* tab) {
    if (blockIdx.z > 0) return g4_pf(pf, (blockIdx.z - 1) * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x, (gridDim.z - 1) * gridDim.x * gridDim.y);
    g4_prep_t<256>(qkv, stride, qw, kw, cs, sn, rot, pos, cap, qo, kc, vc, nq, nk, has_v, eps, tab);
}

extern "C" __global__ void __launch_bounds__(256) g4_prep512(const bf16* qkv, int stride, const bf16* qw, const bf16* kw, const bf16* cs, const bf16* sn, int rot, const int* pos, int cap, bf16* qo,
                                                            bf16* kc, bf16* vc, int nq, int nk, int has_v, float eps, G4Pf pf, const int* tab) {
    if (blockIdx.z > 0) return g4_pf(pf, (blockIdx.z - 1) * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x, (gridDim.z - 1) * gridDim.x * gridDim.y);
    g4_prep_t<512>(qkv, stride, qw, kw, cs, sn, rot, pos, cap, qo, kc, vc, nq, nk, has_v, eps, tab);
}

// ---- attention ---------------------------------------------------------------------------------------------------------------------------
#define G4_KB 32  // keys a tile

// Causal attention of the n rows of q [n, nq D] over the ring kc, vc [cap, nk, D] (position k at slot k % cap): row t sees keys lo_t .. pos[t], lo_t = pos[t] - window + 1 (0 for
// window 0, and at least 0). Scale 1 (Gemma 4's q and k norms carry it): a score is the bf16 of the dot product (transformers' matmul output), the probabilities fp32 and rounded
// to bf16 for the values (as FlashAttention). A block takes R rows and the G = nq / nk query heads of one key-value head (a warp each), the keys G4_KB at a time through shared
// memory: lane j scores key j, then each lane sums D / 32 output dims. Dynamic shared memory: (G4_KB (D/2 + 1) + G4_KB D/2 + R G D/2) * 4 bytes.
// SPLIT (a decode step, R = 1): block z takes the S-th part z of the row's keys and writes its outputs, maximum (log2 units) and sum to part [(t nq + h) S + z][D + 2].
template <int D, bool SPLIT>
__device__ __forceinline__ void g4_attn_t(const bf16* q, const bf16* kc, const bf16* vc, const int* pos, bf16* out, int n, int nq, int nk, int R, int window, int cap, float* part, int S,
                                          const int* tab) {
    constexpr int W2 = D / 2, KS = W2 + 1, PER = D / 32;
    extern __shared__ uint32_t sm[];
    uint32_t* Ks = sm;
    uint32_t* Vs = Ks + G4_KB * KS;
    uint32_t* Qs = Vs + G4_KB * W2;
    __shared__ int s_k0, s_k1;
    const int G = nq / nk, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int kvh = blockIdx.y, r = w / G, h = kvh * G + w % G, t = blockIdx.x * R + r;
    // (SPLIT: the block's first G warps score the row; the others, launched for more loads in flight, only fill the tiles)
    const bool live = t < n && (!SPLIT || r == 0);
    if (live)
        for (int c = lane; c < W2; c += 32) Qs[w * W2 + c] = reinterpret_cast<const uint32_t*>(q + (size_t)t * nq * D + h * D)[c];
    if (threadIdx.x == 0) {
        int lo = 1 << 30, hi = 0;
        for (int i = 0; i < R; i++) {
            const int tt = blockIdx.x * R + i;
            if (tt < n) {
                hi = max(hi, pos[tt] + 1);
                lo = min(lo, window > 0 ? max(0, pos[tt] - window + 1) : 0);
            }
        }
        if (lo > hi) lo = hi;
        if (SPLIT) {
            const int span = ((hi - lo + S - 1) / S + G4_KB - 1) / G4_KB * G4_KB;
            s_k0 = min(hi, lo + blockIdx.z * span);
            s_k1 = min(hi, s_k0 + span);
        } else {
            s_k0 = lo;
            s_k1 = hi;
        }
    }
    __syncthreads();
    const int k_lo = s_k0, kmax = s_k1;
    const int my_hi = live ? pos[t] + 1 : 0, my_lo = live && window > 0 ? max(0, pos[t] - window + 1) : 0;
    float m_run = -INFINITY, l_run = 0.f, acc[PER];
#pragma unroll
    for (int e = 0; e < PER; e++) acc[e] = 0.f;
    for (int k0 = k_lo; k0 < kmax; k0 += G4_KB) {
        __syncthreads();  // (the tile before is read)
        for (int c = threadIdx.x; c < G4_KB * (W2 / 4) * 2; c += blockDim.x) {
            const bool isv = c >= G4_KB * (W2 / 4);
            const int cc = isv ? c - G4_KB * (W2 / 4) : c, j = cc / (W2 / 4), q4 = cc % (W2 / 4);
            uint4 v = make_uint4(0, 0, 0, 0);
            if (k0 + j < kmax) v = reinterpret_cast<const uint4*>((isv ? vc : kc) + (g4_place(tab, k0 + j, cap) * nk + kvh) * D)[q4];
            if (isv) {
                *reinterpret_cast<uint4*>(Vs + j * W2 + q4 * 4) = v;
            } else {
                uint32_t* d = Ks + j * KS + q4 * 4;
                d[0] = v.x;
                d[1] = v.y;
                d[2] = v.z;
                d[3] = v.w;
            }
        }
        __syncthreads();
        if (!live) continue;
        const uint32_t* kr = Ks + lane * KS;
        const uint32_t* qr = Qs + w * W2;
        float s0 = 0.f, s1 = 0.f;
#pragma unroll 8
        for (int c = 0; c < W2; c++) {
            const float2 a = bf2(qr[c]), b = bf2(kr[c]);
            s0 = fmaf(a.x, b.x, s0);
            s1 = fmaf(a.y, b.y, s1);
        }
        const int kp = k0 + lane;
        const float s = kp >= my_lo && kp < my_hi ? b2f(f2b(s0 + s1)) * 1.4426950408889634f : -INFINITY;
        const float m_new = fmaxf(m_run, warp_max(s));
        if (m_new == -INFINITY) continue;  // (no key of this tile, nor before, in the row's window)
        const float p = exp2f(s - m_new), corr = exp2f(m_run - m_new);
        l_run = l_run * corr + warp_sum(p);
        const float pb = b2f(f2b(p));
#pragma unroll
        for (int e = 0; e < PER; e++) acc[e] *= corr;
#pragma unroll 4
        for (int j = 0; j < G4_KB; j++) {
            const float pj = __shfl_sync(0xffffffffu, pb, j);
            const uint32_t* vr = Vs + j * W2 + lane * (PER / 2);
#pragma unroll
            for (int e = 0; e < PER / 2; e++) {
                const float2 v2 = bf2(vr[e]);
                acc[2 * e] = fmaf(pj, v2.x, acc[2 * e]);
                acc[2 * e + 1] = fmaf(pj, v2.y, acc[2 * e + 1]);
            }
        }
        m_run = m_new;
    }
    if (!live) return;
    if (SPLIT) {
        float* pp = part + ((size_t)(t * nq + h) * S + blockIdx.z) * (D + 2);
#pragma unroll
        for (int e = 0; e < PER; e++) pp[lane * PER + e] = acc[e];
        if (lane == 0) {
            pp[D] = m_run;
            pp[D + 1] = l_run;
        }
        return;
    }
    bf16* o = out + (size_t)t * nq * D + h * D + lane * PER;
#pragma unroll
    for (int e = 0; e < PER; e++) o[e] = f2b(acc[e] / l_run);
}

extern "C" __global__ void __launch_bounds__(512) g4_attn256(const bf16* q, const bf16* kc, const bf16* vc, const int* pos, bf16* out, int n, int nq, int nk, int R, int window, int cap,
                                                           const int* tab) {
    g4_attn_t<256, false>(q, kc, vc, pos, out, n, nq, nk, R, window, cap, nullptr, 1, tab);
}
extern "C" __global__ void __launch_bounds__(512) g4_attn512(const bf16* q, const bf16* kc, const bf16* vc, const int* pos, bf16* out, int n, int nq, int nk, int R, int window, int cap,
                                                           const int* tab) {
    g4_attn_t<512, false>(q, kc, vc, pos, out, n, nq, nk, R, window, cap, nullptr, 1, tab);
}
extern "C" __global__ void __launch_bounds__(512) g4_attn256_parts(const bf16* q, const bf16* kc, const bf16* vc, const int* pos, int n, int nq, int nk, int window, int cap, float* part, int S,
                                                                 const int* tab) {
    g4_attn_t<256, true>(q, kc, vc, pos, nullptr, n, nq, nk, 1, window, cap, part, S, tab);
}
extern "C" __global__ void __launch_bounds__(512) g4_attn512_parts(const bf16* q, const bf16* kc, const bf16* vc, const int* pos, int n, int nq, int nk, int window, int cap, float* part, int S,
                                                                 const int* tab) {
    g4_attn_t<512, true>(q, kc, vc, pos, nullptr, n, nq, nk, 1, window, cap, part, S, tab);
}

// ---- a prompt chunk's attention on the tensor cores -------------------------------------------------------------------------------------
// g4_attn_t's arithmetic (scale 1, the score rounded to bf16, exp2 of the score times log2 e, the probabilities rounded to bf16 for the values,
// their fp32 sum dividing at the end) with Q K^T and P V as mma.sync m16n8k16 bf16 products. A block takes BM rows of one query head h and the
// value columns z DV .. z DV + DV - 1 (a 512-wide head's values in two passes, each computing the scores again: the accumulators of 16 rows by 512
// columns would not fit a warp's registers); 16 rows a warp, the keys 32 a tile from the ring (slot k % cap), each row's keys lo_t .. pos[t]. Shared
// memory: Q [BM][D + 8], K [32][D + 8], V [32][DV + 8] bf16 (rows padded by 16 bytes so ldmatrix meets no bank twice). Grid (ceil(n / BM), nq, D / DV).
__device__ __forceinline__ uint32_t g4_pk(float a, float b) {
    __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&v);
}

template <int D, int DV, int BM>
__device__ __forceinline__ void g4_fa_t(const bf16* q, const bf16* kc, const bf16* vc, const int* pos, bf16* out, int n, int nq, int nk, int window, int cap, const int* tab) {
    constexpr int KT = 32, QS = D + 8, VS = DV + 8, NT = DV / 8, W = BM / 16;
    extern __shared__ __align__(16) unsigned char g4_fa_sm[];
    bf16* Qs = reinterpret_cast<bf16*>(g4_fa_sm);
    bf16* Ks = Qs + BM * QS;
    bf16* Vs = Ks + KT * QS;
    __shared__ int s_lo, s_hi;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3;
    const int h = blockIdx.y, kvh = h / (nq / nk), r0 = blockIdx.x * BM, v0 = blockIdx.z * DV;
    // the queries of the block's rows (zeros past n)
    for (int c = threadIdx.x; c < BM * (D / 8); c += blockDim.x) {
        const int r = c / (D / 8), q8 = c % (D / 8);
        uint4 v = make_uint4(0, 0, 0, 0);
        if (r0 + r < n) v = reinterpret_cast<const uint4*>(q + (size_t)(r0 + r) * nq * D + h * D)[q8];
        *reinterpret_cast<uint4*>(Qs + r * QS + q8 * 8) = v;
    }
    if (threadIdx.x == 0) {
        int lo = 1 << 30, hi = 0;
        for (int r = r0; r < min(n, r0 + BM); r++) {
            hi = max(hi, pos[r] + 1);
            lo = min(lo, window > 0 ? max(0, pos[r] - window + 1) : 0);
        }
        s_lo = min(lo, hi);
        s_hi = hi;
    }
    __syncthreads();
    const int ra = r0 + warp * 16 + g, rb = ra + 8;
    const int pa = ra < n ? pos[ra] : -1, pb = rb < n ? pos[rb] : -1;
    const int la = window > 0 ? pa - window + 1 : 0, lb = window > 0 ? pb - window + 1 : 0;
    float acc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    float ma = -INFINITY, mb = -INFINITY, sa = 0.f, sb = 0.f;
    for (int k0 = s_lo; k0 < s_hi; k0 += KT) {
        __syncthreads();  // (the tiles before are read)
        for (int c = threadIdx.x; c < KT * (D / 8); c += blockDim.x) {
            const int j = c / (D / 8), q8 = c % (D / 8);
            uint4 v = make_uint4(0, 0, 0, 0);
            if (k0 + j < s_hi) v = reinterpret_cast<const uint4*>(kc + (g4_place(tab, k0 + j, cap) * nk + kvh) * D)[q8];
            *reinterpret_cast<uint4*>(Ks + j * QS + q8 * 8) = v;
        }
        for (int c = threadIdx.x; c < KT * (DV / 8); c += blockDim.x) {
            const int j = c / (DV / 8), q8 = c % (DV / 8);
            uint4 v = make_uint4(0, 0, 0, 0);
            if (k0 + j < s_hi) v = reinterpret_cast<const uint4*>(vc + (g4_place(tab, k0 + j, cap) * nk + kvh) * D + v0)[q8];
            *reinterpret_cast<uint4*>(Vs + j * VS + q8 * 8) = v;
        }
        __syncthreads();
        // S [16 rows][32 keys] = Q K^T: 4 n-tiles of 8 keys
        float s[4][4];
#pragma unroll
        for (int j = 0; j < 4; j++) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
#pragma unroll 4
        for (int ks = 0; ks < D / 16; ks++) {
            uint32_t a[4];
            ldsm4(a, s_addr(Qs + (warp * 16 + (lane & 15)) * QS + ks * 16 + (lane >> 4) * 8));
#pragma unroll
            for (int np = 0; np < 2; np++) {
                uint32_t b[4];
                const int m = lane >> 3;
                ldsm4(b, s_addr(Ks + (np * 16 + (m >> 1) * 8 + (lane & 7)) * QS + ks * 16 + (m & 1) * 8));
                mma16816(s[2 * np], a, b[0], b[1]);
                mma16816(s[2 * np + 1], a, b[2], b[3]);
            }
        }
        // the scores rounded to bf16, masked, in log2 units; the running maxima and sums (a row's 4 lanes share g)
        float mxa = ma, mxb = mb;
#pragma unroll
        for (int j = 0; j < 4; j++) {
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const int kp = k0 + j * 8 + 2 * t4 + e;
                s[j][e] = kp >= la && kp <= pa ? b2f(f2b(s[j][e])) * 1.4426950408889634f : -INFINITY;
                s[j][2 + e] = kp >= lb && kp <= pb ? b2f(f2b(s[j][2 + e])) * 1.4426950408889634f : -INFINITY;
                mxa = fmaxf(mxa, s[j][e]);
                mxb = fmaxf(mxb, s[j][2 + e]);
            }
        }
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 1));
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 2));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 1));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 2));
        const float ca = mxa == -INFINITY ? 1.f : exp2f(ma - mxa), cb = mxb == -INFINITY ? 1.f : exp2f(mb - mxb);
        float ta = 0.f, tb = 0.f;
#pragma unroll
        for (int j = 0; j < 4; j++) {
#pragma unroll
            for (int e = 0; e < 2; e++) {
                s[j][e] = mxa == -INFINITY ? 0.f : exp2f(s[j][e] - mxa);
                s[j][2 + e] = mxb == -INFINITY ? 0.f : exp2f(s[j][2 + e] - mxb);
                ta += s[j][e];
                tb += s[j][2 + e];
            }
        }
        ta += __shfl_xor_sync(0xffffffffu, ta, 1);
        ta += __shfl_xor_sync(0xffffffffu, ta, 2);
        tb += __shfl_xor_sync(0xffffffffu, tb, 1);
        tb += __shfl_xor_sync(0xffffffffu, tb, 2);
        sa = sa * ca + ta;
        sb = sb * cb + tb;
        ma = mxa;
        mb = mxb;
#pragma unroll
        for (int j = 0; j < NT; j++) {
            acc[j][0] *= ca;
            acc[j][1] *= ca;
            acc[j][2] *= cb;
            acc[j][3] *= cb;
        }
        // O += bf16(P) V: the score tiles as the A operand, 16 keys a k-step
#pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            const uint32_t a[4] = {g4_pk(s[2 * kk][0], s[2 * kk][1]), g4_pk(s[2 * kk][2], s[2 * kk][3]), g4_pk(s[2 * kk + 1][0], s[2 * kk + 1][1]), g4_pk(s[2 * kk + 1][2], s[2 * kk + 1][3])};
#pragma unroll
            for (int np = 0; np < NT / 2; np++) {
                uint32_t b[4];
                const int m = lane >> 3;
                ldsm4t(b, s_addr(Vs + (kk * 16 + (m & 1) * 8 + (lane & 7)) * VS + np * 16 + (m >> 1) * 8));
                mma16816(acc[2 * np], a, b[0], b[1]);
                mma16816(acc[2 * np + 1], a, b[2], b[3]);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < NT; j++) {
        const int c = v0 + j * 8 + 2 * t4;
        if (ra < n) *reinterpret_cast<__nv_bfloat162*>(out + (size_t)ra * nq * D + h * D + c) = __floats2bfloat162_rn(acc[j][0] / sa, acc[j][1] / sa);
        if (rb < n) *reinterpret_cast<__nv_bfloat162*>(out + (size_t)rb * nq * D + h * D + c) = __floats2bfloat162_rn(acc[j][2] / sb, acc[j][3] / sb);
    }
}

// The 512-wide heads (global layers) on the tensor cores, 8 warps a block: 32 rows of one query head, warp w = 4 rg + ds the rows 16 rg .. 16 rg + 15
// and the channels 128 ds .. 128 ds + 127. Its queries' A fragments stay in registers. Q K^T: each warp its 128 channels' share of the 16 x 32 scores,
// the four shares added through shared memory in a fixed order (every warp of a row group the same sums); then g4_fa_t's rounding, mask and online
// softmax (the four warps alike), and P V into the warp's own 128 value columns: the scores are made once. Shared memory: K and V [32][520] bf16
// (the queries pass through V's space first) and the shares [2][4][16][33] floats: 83,456 bytes. Grid (ceil(n / 32), nq).
#define G4B_SP 33
extern "C" __global__ void __launch_bounds__(256, 1) g4_fa512b(const bf16* q, const bf16* kc, const bf16* vc, const int* pos, bf16* out, int n, int nq, int nk, int window, int cap,
                                                                const int* tab) {
    constexpr int D = 512, KT = 32, QS = D + 8, BM = 32;
    extern __shared__ __align__(16) unsigned char g4_fa_sm[];
    bf16* Ks = reinterpret_cast<bf16*>(g4_fa_sm);
    bf16* Vs = Ks + KT * QS;
    float* Sp = reinterpret_cast<float*>(Vs + KT * QS);
    __shared__ int s_lo, s_hi;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3, rg = warp >> 2, ds = warp & 3;
    const int h = blockIdx.y, kvh = h / (nq / nk), r0 = blockIdx.x * BM;
    for (int c = threadIdx.x; c < BM * (D / 8); c += blockDim.x) {
        const int r = c / (D / 8), q8 = c % (D / 8);
        uint4 v = make_uint4(0, 0, 0, 0);
        if (r0 + r < n) v = reinterpret_cast<const uint4*>(q + (size_t)(r0 + r) * nq * D + h * D)[q8];
        *reinterpret_cast<uint4*>(Vs + r * QS + q8 * 8) = v;
    }
    if (threadIdx.x == 0) {
        int lo = 1 << 30, hi = 0;
        for (int r = r0; r < min(n, r0 + BM); r++) {
            hi = max(hi, pos[r] + 1);
            lo = min(lo, window > 0 ? max(0, pos[r] - window + 1) : 0);
        }
        s_lo = min(lo, hi);
        s_hi = hi;
    }
    __syncthreads();
    uint32_t qa[8][4];
#pragma unroll
    for (int ks = 0; ks < 8; ks++) ldsm4(qa[ks], s_addr(Vs + (rg * 16 + (lane & 15)) * QS + ds * 128 + ks * 16 + (lane >> 4) * 8));
    const int ra = r0 + rg * 16 + g, rb = ra + 8;
    const int pa = ra < n ? pos[ra] : -1, pb = rb < n ? pos[rb] : -1;
    const int la = window > 0 ? pa - window + 1 : 0, lb = window > 0 ? pb - window + 1 : 0;
    float acc[16][4];
#pragma unroll
    for (int j = 0; j < 16; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    float ma = -INFINITY, mb = -INFINITY, sa = 0.f, sb = 0.f;
    float* myS = Sp + (rg * 4 + ds) * 16 * G4B_SP;
    for (int k0 = s_lo; k0 < s_hi; k0 += KT) {
        __syncthreads();  // (the queries, or the tiles before, are read)
        for (int c = threadIdx.x; c < 2 * KT * (D / 8); c += blockDim.x) {
            const bool isv = c >= KT * (D / 8);
            const int cc = isv ? c - KT * (D / 8) : c, j = cc / (D / 8), q8 = cc % (D / 8);
            uint4 v = make_uint4(0, 0, 0, 0);
            if (k0 + j < s_hi) v = reinterpret_cast<const uint4*>((isv ? vc : kc) + (g4_place(tab, k0 + j, cap) * nk + kvh) * D)[q8];
            *reinterpret_cast<uint4*>((isv ? Vs : Ks) + j * QS + q8 * 8) = v;
        }
        __syncthreads();
        float s[4][4];
#pragma unroll
        for (int j = 0; j < 4; j++) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
#pragma unroll
        for (int ks = 0; ks < 8; ks++) {
#pragma unroll
            for (int np = 0; np < 2; np++) {
                uint32_t b[4];
                const int m = lane >> 3;
                ldsm4(b, s_addr(Ks + (np * 16 + (m >> 1) * 8 + (lane & 7)) * QS + ds * 128 + ks * 16 + (m & 1) * 8));
                mma16816(s[2 * np], qa[ks], b[0], b[1]);
                mma16816(s[2 * np + 1], qa[ks], b[2], b[3]);
            }
        }
#pragma unroll
        for (int j = 0; j < 4; j++) {
            myS[g * G4B_SP + j * 8 + 2 * t4] = s[j][0];
            myS[g * G4B_SP + j * 8 + 2 * t4 + 1] = s[j][1];
            myS[(g + 8) * G4B_SP + j * 8 + 2 * t4] = s[j][2];
            myS[(g + 8) * G4B_SP + j * 8 + 2 * t4 + 1] = s[j][3];
        }
        __syncthreads();
        const float* gS = Sp + rg * 4 * 16 * G4B_SP;
        float mxa = ma, mxb = mb;
#pragma unroll
        for (int j = 0; j < 4; j++) {
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const int col = j * 8 + 2 * t4 + e, kp = k0 + col;
                float xa = 0.f, xb = 0.f;
#pragma unroll
                for (int d4 = 0; d4 < 4; d4++) {
                    xa += gS[(d4 * 16 + g) * G4B_SP + col];
                    xb += gS[(d4 * 16 + g + 8) * G4B_SP + col];
                }
                s[j][e] = kp >= la && kp <= pa ? b2f(f2b(xa)) * 1.4426950408889634f : -INFINITY;
                s[j][2 + e] = kp >= lb && kp <= pb ? b2f(f2b(xb)) * 1.4426950408889634f : -INFINITY;
                mxa = fmaxf(mxa, s[j][e]);
                mxb = fmaxf(mxb, s[j][2 + e]);
            }
        }
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 1));
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 2));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 1));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 2));
        const float ca = mxa == -INFINITY ? 1.f : exp2f(ma - mxa), cb = mxb == -INFINITY ? 1.f : exp2f(mb - mxb);
        float ta = 0.f, tb = 0.f;
#pragma unroll
        for (int j = 0; j < 4; j++) {
#pragma unroll
            for (int e = 0; e < 2; e++) {
                s[j][e] = mxa == -INFINITY ? 0.f : exp2f(s[j][e] - mxa);
                s[j][2 + e] = mxb == -INFINITY ? 0.f : exp2f(s[j][2 + e] - mxb);
                ta += s[j][e];
                tb += s[j][2 + e];
            }
        }
        ta += __shfl_xor_sync(0xffffffffu, ta, 1);
        ta += __shfl_xor_sync(0xffffffffu, ta, 2);
        tb += __shfl_xor_sync(0xffffffffu, tb, 1);
        tb += __shfl_xor_sync(0xffffffffu, tb, 2);
        sa = sa * ca + ta;
        sb = sb * cb + tb;
        ma = mxa;
        mb = mxb;
#pragma unroll
        for (int j = 0; j < 16; j++) {
            acc[j][0] *= ca;
            acc[j][1] *= ca;
            acc[j][2] *= cb;
            acc[j][3] *= cb;
        }
#pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            const uint32_t a[4] = {g4_pk(s[2 * kk][0], s[2 * kk][1]), g4_pk(s[2 * kk][2], s[2 * kk][3]), g4_pk(s[2 * kk + 1][0], s[2 * kk + 1][1]), g4_pk(s[2 * kk + 1][2], s[2 * kk + 1][3])};
#pragma unroll
            for (int np = 0; np < 8; np++) {
                uint32_t b[4];
                const int m = lane >> 3;
                ldsm4t(b, s_addr(Vs + (kk * 16 + (m & 1) * 8 + (lane & 7)) * QS + ds * 128 + np * 16 + (m >> 1) * 8));
                mma16816(acc[2 * np], a, b[0], b[1]);
                mma16816(acc[2 * np + 1], a, b[2], b[3]);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < 16; j++) {
        const int c = ds * 128 + j * 8 + 2 * t4;
        if (ra < n) *reinterpret_cast<__nv_bfloat162*>(out + (size_t)ra * nq * D + h * D + c) = __floats2bfloat162_rn(acc[j][0] / sa, acc[j][1] / sa);
        if (rb < n) *reinterpret_cast<__nv_bfloat162*>(out + (size_t)rb * nq * D + h * D + c) = __floats2bfloat162_rn(acc[j][2] / sb, acc[j][3] / sb);
    }
}

extern "C" __global__ void __launch_bounds__(128, 1) g4_fa256(const bf16* q, const bf16* kc, const bf16* vc, const int* pos, bf16* out, int n, int nq, int nk, int window, int cap,
                                                              const int* tab) {
    g4_fa_t<256, 256, 64>(q, kc, vc, pos, out, n, nq, nk, window, cap, tab);
}
extern "C" __global__ void __launch_bounds__(64, 1) g4_fa512(const bf16* q, const bf16* kc, const bf16* vc, const int* pos, bf16* out, int n, int nq, int nk, int window, int cap, const int* tab) {
    g4_fa_t<512, 256, 32>(q, kc, vc, pos, out, n, nq, nk, window, cap, tab);
}

// The S parts combined: out = bf16(sum of the parts' outputs, each times 2^(its maximum - the largest), / the sum likewise). Block (row, query head), D threads (a dim each).
extern "C" __global__ void g4_attn_merge(const float* part, int S, bf16* out, int nq, int D) {
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const float* pp = part + (size_t)(t * nq + h) * S * (D + 2);
    float mx = -INFINITY;
    for (int z = 0; z < S; z++) mx = fmaxf(mx, pp[z * (D + 2) + D]);
    float l = 0.f, a = 0.f;
    for (int z = 0; z < S; z++) {
        const float m = pp[z * (D + 2) + D];
        if (m == -INFINITY) continue;  // (a part with no keys)
        const float wz = exp2f(m - mx);
        l = fmaf(pp[z * (D + 2) + D + 1], wz, l);
        a = fmaf(pp[z * (D + 2) + d], wz, a);
    }
    out[(size_t)t * nq * D + h * D + d] = f2b(a / l);
}

// ---- the feed-forward and the logits ------------------------------------------------------------------------------------------------------
// m [n, inter] = bf16(bf16(gelu_tanh(gate)) * up), gate and up the halves of gu [n, 2 inter] (transformers: the activation's bf16 output times up, rounded).
extern "C" __global__ void g4_gelu_mul(const bf16* gu, bf16* m, int inter, G4Pf pf) {
    if (blockIdx.z > 0) return g4_pf(pf, (blockIdx.z - 1) * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x, (gridDim.z - 1) * gridDim.x * gridDim.y);
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
