// The attention layer kind's kernels (src/runtime/kinds.rs `Attn`; the description executor's `Op::Attn`, src/dynamic/attn.rs, runs them for every dense-attention layer
// of every family). One set, its arithmetic and shapes in `AttnArgs`: the preparation (the q and k norms, the rotary, the cache write, the values' norm), the decode
// step's attention (the keys in parts, then combined), a prompt chunk's attention on the tensor cores (and a warp-per-head kernel for any shape), and the output gate.
// Head widths 64, 128, 256 and 512 are template instances. Included by engine.cu after gemma4.cu (its G4Pf prefetch) and qwen35.cu (bf2, warp_max).
//
// Arithmetic: bf16 in memory, fp32 inside. `round`: the rounding points of transformers' eager bf16 arithmetic in the preparation and the gate (a norm rounds once,
// the queries' scale rounds, each rotary product and their sum round, the gate's sigmoid rounds before its product; the norm's factor 1 / sqrt(mean + eps) exact);
// else one rounding at the end of each (the norm's factor rsqrtf). `sbf16`: a score is the bf16 of its dot product. The probabilities are rounded to bf16 for the
// values (as FlashAttention), their fp32 sum divides at the end.

// Every argument of the kind's kernels (src/dynamic/attn.rs `AttnArgs`, field for field).
struct AttnArgs {
    const bf16* src;   // the projection's rows, `stride` apart: query head h at h q_step, key head j at k_off + j d, value head j at v_off + j d (v_off < 0: the key's projection)
    const bf16* qw;    // the q and k norms' weights [d] (norm 1 or 2)
    const bf16* kw;
    const bf16* cs;    // the rotary's cos and sin, bf16 [positions, pairs] (null: no rotary)
    const bf16* sn;
    const int* qpos;   // each row's position (its query's, its key's)
    const int* kpos;   // the last key each row reads (a draft model's queries read another's)
    bf16* qo;          // the prepared queries [n, nq d]
    bf16* kc;          // the ring's keys and values [places, nk, d]
    bf16* vc;
    const int* tab;    // the sequence's block table (16-token blocks) or null: position modulo cap
    bf16* out;         // the attention's output, rows `os` apart, head h at h d
    const bf16* gate;  // the output gate's values, head h of row t at t gs + h g_step (null: no gate)
    float* part;       // a decode step's parts [(t nq + h) S + z][d + 2]
    int d, stride, q_step, k_off, v_off;
    int norm;          // the q and k norms: 0 none, 1 x w, 2 x (1 + w), 3 no weight
    int v_norm;        // the values normed without a weight
    int span, pairs;   // the rotary: pairs (i, i + span / 2) of the first span dims, the first `pairs` of them turned
    int cap;           // the ring's places (cap < 0: row r's own place, a staging buffer of the step's rows)
    int n, nq, nk, window;
    int os, gs, g_step, gnorm;
    int round, sbf16, S;
    float q_scale, eps, scale, sl2, softcap;
};

// A key's place: through the block table where the sequence has one (src/dynamic/pages.rs), else the ring's slot p % cap.
__device__ __forceinline__ size_t at_place(const int* tab, int p, int cap) { return tab ? (size_t)tab[p >> 4] * 16 + (p & 15) : (size_t)(p % cap); }

__device__ __forceinline__ float at_r(float x, int round) { return round ? b2f(f2b(x)) : x; }

// A norm's factor over d values whose squares sum to ss.
__device__ __forceinline__ float at_rs(float ss, int d, float eps, int round) { return round ? 1.f / sqrtf(ss / (float)d + eps) : rsqrtf(ss / (float)d + eps); }

// A score in log2 units: the bf16 of the dot product where the kind's scores are, the soft-cap (cap tanh(s / cap) of the scaled score), the scale.
__device__ __forceinline__ float at_score(const AttnArgs& a, float s) {
    if (a.sbf16) s = b2f(f2b(s));
    if (a.softcap > 0.f) return a.softcap * tanhf(s * a.scale / a.softcap) * 1.4426950408889634f;
    return s * a.sl2;
}

// ---- the preparation --------------------------------------------------------------------------------------------------------------------------------------------------
// Row r's head hh (a query head below nq, else key head hh - nq with its values): the norm (one block sum), the queries' scale, the rotary (value j < span with
// its pair j +- span / 2, angle j mod span / 2 where that is below `pairs`), the query to qo, the key and the value (normed or as it is) to the ring at the
// row's position. Block (row, head), d / 2 threads: thread t holds t and t + d / 2. Grid y: nq + nk heads, or nq where no keys are written (a draft model's).
// blockIdx.z above 0: `pf`'s prefetch of the next product.
__device__ __forceinline__ float at_rot(const AttnArgs& a, const float* xs, int j, float v, int p) {
    const int half = a.span >> 1;
    if (j >= a.span) return v;
    const bool lo = j < half;
    const int k = lo ? j : j - half;
    if (k >= a.pairs) return v;
    const float c = b2f(a.cs[(size_t)p * a.pairs + k]), sv = b2f(a.sn[(size_t)p * a.pairs + k]);
    const float o = xs[lo ? j + half : j - half];
    if (a.round) return lo ? b2f(f2b(b2f(f2b(v * c)) - b2f(f2b(o * sv)))) : b2f(f2b(b2f(f2b(v * c)) + b2f(f2b(o * sv))));
    return lo ? v * c - o * sv : v * c + o * sv;
}

template <int D>
__device__ __forceinline__ void attn_prep_t(const AttnArgs& a) {
    constexpr int H = D / 2;
    __shared__ float xs[D];
    const int r = blockIdx.x, hh = blockIdx.y, t = threadIdx.x;
    const bf16* row = a.src + (size_t)r * a.stride;
    const int p = a.qpos[r];
    const bool isq = hh < a.nq;
    const int kh = hh - a.nq;
    const bf16* x = isq ? row + (size_t)hh * a.q_step : row + a.k_off + kh * D;
    float v0 = b2f(x[t]), v1 = b2f(x[t + H]);
    if (a.norm) {
        const float s = at_rs(block_sum(v0 * v0 + v1 * v1), D, a.eps, a.round);
        const bf16* w = isq ? a.qw : a.kw;
        v0 = v0 * s;
        v1 = v1 * s;
        if (a.norm == 1) {
            v0 = v0 * b2f(w[t]);
            v1 = v1 * b2f(w[t + H]);
        } else if (a.norm == 2) {
            v0 = v0 * (1.f + b2f(w[t]));
            v1 = v1 * (1.f + b2f(w[t + H]));
        }
        v0 = at_r(v0, a.round);
        v1 = at_r(v1, a.round);
    }
    if (isq && a.q_scale != 1.f) {
        v0 = at_r(v0 * a.q_scale, a.round);
        v1 = at_r(v1 * a.q_scale, a.round);
    }
    if (a.cs) {
        xs[t] = v0;
        xs[t + H] = v1;
        __syncthreads();
        v0 = at_rot(a, xs, t, v0, p);
        v1 = at_rot(a, xs, t + H, v1, p);
    }
    const size_t slot = a.cap < 0 ? (size_t)r : at_place(a.tab, p, a.cap);
    bf16* o = isq ? a.qo + (size_t)r * a.nq * D + hh * D : a.kc + (slot * a.nk + kh) * D;
    o[t] = f2b(v0);
    o[t + H] = f2b(v1);
    if (isq) return;
    const bf16* xv = a.v_off >= 0 ? row + a.v_off + kh * D : x;
    float u0 = b2f(xv[t]), u1 = b2f(xv[t + H]);
    if (a.v_norm) {
        const float s = at_rs(block_sum(u0 * u0 + u1 * u1), D, a.eps, a.round);
        u0 *= s;
        u1 *= s;
    }
    bf16* ov = a.vc + (slot * a.nk + kh) * D;
    ov[t] = f2b(u0);
    ov[t + H] = f2b(u1);
}

#define AT_PF if (blockIdx.z > 0) return g4_pf(pf, (blockIdx.z - 1) * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x, (gridDim.z - 1) * gridDim.x * gridDim.y)
extern "C" __global__ void __launch_bounds__(32) attn_prep64(AttnArgs a, G4Pf pf) {
    AT_PF;
    attn_prep_t<64>(a);
}
extern "C" __global__ void __launch_bounds__(64) attn_prep128(AttnArgs a, G4Pf pf) {
    AT_PF;
    attn_prep_t<128>(a);
}
extern "C" __global__ void __launch_bounds__(128) attn_prep256(AttnArgs a, G4Pf pf) {
    AT_PF;
    attn_prep_t<256>(a);
}
extern "C" __global__ void __launch_bounds__(256) attn_prep512(AttnArgs a, G4Pf pf) {
    AT_PF;
    attn_prep_t<512>(a);
}

// ---- the attention, a warp a query head ------------------------------------------------------------------------------------------------------------------------------
#define AT_KB 32  // keys a tile

// Causal attention of the n rows of qo over the ring: row t sees keys lo_t .. kpos[t], lo_t = kpos[t] - window + 1 (0 for window 0, and at least 0). A block takes R rows
// and the G = nq / nk query heads of one key-value head (a warp each), the keys AT_KB at a time through shared memory: lane j scores key j, then each lane sums D / 32 output
// dims. Dynamic shared memory: (AT_KB (D/2 + 1) + AT_KB D/2 + R G D/2) * 4 bytes.
// SPLIT (a decode step, R = 1): block z takes the S-th part z of the row's keys and writes its outputs, maximum (log2 units) and sum to part [(t nq + h) S + z][D + 2]; the
// block's warps past the first G only fill the tiles (more loads in flight).
template <int D, bool SPLIT>
__device__ __forceinline__ void attn_rows_t(const AttnArgs& a, int R) {
    constexpr int W2 = D / 2, KS = W2 + 1, PER = D / 32;
    extern __shared__ uint32_t sm[];
    uint32_t* Ks = sm;
    uint32_t* Vs = Ks + AT_KB * KS;
    uint32_t* Qs = Vs + AT_KB * W2;
    __shared__ int s_k0, s_k1;
    const int n = a.n, nq = a.nq, nk = a.nk, window = a.window, S = a.S;
    const int G = nq / nk, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int kvh = blockIdx.y, r = w / G, h = kvh * G + w % G, t = blockIdx.x * R + r;
    const bool live = t < n && (!SPLIT || r == 0);
    if (live)
        for (int c = lane; c < W2; c += 32) Qs[w * W2 + c] = reinterpret_cast<const uint32_t*>(a.qo + (size_t)t * nq * D + h * D)[c];
    if (threadIdx.x == 0) {
        int lo = 1 << 30, hi = 0;
        for (int i = 0; i < R; i++) {
            const int tt = blockIdx.x * R + i;
            if (tt < n) {
                hi = max(hi, a.kpos[tt] + 1);
                lo = min(lo, window > 0 ? max(0, a.kpos[tt] - window + 1) : 0);
            }
        }
        if (lo > hi) lo = hi;
        if (SPLIT) {
            const int span = ((hi - lo + S - 1) / S + AT_KB - 1) / AT_KB * AT_KB;
            s_k0 = min(hi, lo + blockIdx.z * span);
            s_k1 = min(hi, s_k0 + span);
        } else {
            s_k0 = lo;
            s_k1 = hi;
        }
    }
    __syncthreads();
    const int k_lo = s_k0, kmax = s_k1;
    const int my_hi = live ? a.kpos[t] + 1 : 0, my_lo = live && window > 0 ? max(0, a.kpos[t] - window + 1) : 0;
    float m_run = -INFINITY, l_run = 0.f, acc[PER];
#pragma unroll
    for (int e = 0; e < PER; e++) acc[e] = 0.f;
    for (int k0 = k_lo; k0 < kmax; k0 += AT_KB) {
        __syncthreads();  // (the tile before is read)
        for (int c = threadIdx.x; c < AT_KB * (W2 / 4) * 2; c += blockDim.x) {
            const bool isv = c >= AT_KB * (W2 / 4);
            const int cc = isv ? c - AT_KB * (W2 / 4) : c, j = cc / (W2 / 4), q4 = cc % (W2 / 4);
            uint4 v = make_uint4(0, 0, 0, 0);
            if (k0 + j < kmax) v = reinterpret_cast<const uint4*>((isv ? a.vc : a.kc) + (at_place(a.tab, k0 + j, a.cap) * nk + kvh) * D)[q4];
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
            const float2 x = bf2(qr[c]), y = bf2(kr[c]);
            s0 = fmaf(x.x, y.x, s0);
            s1 = fmaf(x.y, y.y, s1);
        }
        const int kp = k0 + lane;
        const float s = kp >= my_lo && kp < my_hi ? at_score(a, s0 + s1) : -INFINITY;
        const float m_new = fmaxf(m_run, warp_max(s));
        if (m_new == -INFINITY) continue;  // (no key of this tile, nor before, in the row's window)
        const float p = exp2f(s - m_new), corr = exp2f(m_run - m_new);
        l_run = l_run * corr + warp_sum(p);
        const float pb = b2f(f2b(p));
#pragma unroll
        for (int e = 0; e < PER; e++) acc[e] *= corr;
#pragma unroll 4
        for (int j = 0; j < AT_KB; j++) {
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
        float* pp = a.part + ((size_t)(t * nq + h) * S + blockIdx.z) * (D + 2);
#pragma unroll
        for (int e = 0; e < PER; e++) pp[lane * PER + e] = acc[e];
        if (lane == 0) {
            pp[D] = m_run;
            pp[D + 1] = l_run;
        }
        return;
    }
    bf16* o = a.out + (size_t)t * a.os + h * D + lane * PER;
#pragma unroll
    for (int e = 0; e < PER; e++) o[e] = f2b(acc[e] / l_run);
}

extern "C" __global__ void __launch_bounds__(512) attn_rows64(AttnArgs a, int R) { attn_rows_t<64, false>(a, R); }
extern "C" __global__ void __launch_bounds__(512) attn_rows128(AttnArgs a, int R) { attn_rows_t<128, false>(a, R); }
extern "C" __global__ void __launch_bounds__(512) attn_rows256(AttnArgs a, int R) { attn_rows_t<256, false>(a, R); }
extern "C" __global__ void __launch_bounds__(512) attn_rows512(AttnArgs a, int R) { attn_rows_t<512, false>(a, R); }
extern "C" __global__ void __launch_bounds__(512) attn_parts64(AttnArgs a) { attn_rows_t<64, true>(a, 1); }
extern "C" __global__ void __launch_bounds__(512) attn_parts128(AttnArgs a) { attn_rows_t<128, true>(a, 1); }
extern "C" __global__ void __launch_bounds__(512) attn_parts256(AttnArgs a) { attn_rows_t<256, true>(a, 1); }
extern "C" __global__ void __launch_bounds__(512) attn_parts512(AttnArgs a) { attn_rows_t<512, true>(a, 1); }

// ---- the output gate --------------------------------------------------------------------------------------------------------------------------------------------------
// The gated value of dim d of row t's head h (o: the head's output as bf16 holds it; a block of d threads holds the head): its norm without a weight where `gnorm`
// (one rounding where `round`), times sigmoid(gate) (the sigmoid rounded first where `round`).
__device__ __forceinline__ float at_gated(const AttnArgs& a, float o, int t, int h, int d) {
    if (a.gnorm) o = at_r(o * at_rs(block_sum(o * o), a.d, a.eps, a.round), a.round);
    const float g = b2f(a.gate[(size_t)t * a.gs + (size_t)h * a.g_step + d]);
    return a.round ? o * b2f(f2b(1.f / (1.f + expf(-g)))) : o / (1.f + expf(-g));
}

// The S parts of a decode step combined: out = bf16(sum of the parts' outputs, each times 2^(its maximum - the largest), / the sum likewise), then the gate where
// there is one. Block (row, query head), d threads (a dim each).
extern "C" __global__ void attn_merge(AttnArgs a) {
    const int D = a.d, S = a.S, t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const float* pp = a.part + (size_t)(t * a.nq + h) * S * (D + 2);
    float mx = -INFINITY;
    for (int z = 0; z < S; z++) mx = fmaxf(mx, pp[z * (D + 2) + D]);
    float l = 0.f, acc = 0.f;
    for (int z = 0; z < S; z++) {
        const float m = pp[z * (D + 2) + D];
        if (m == -INFINITY) continue;  // (a part with no keys)
        const float wz = exp2f(m - mx);
        l = fmaf(pp[z * (D + 2) + D + 1], wz, l);
        acc = fmaf(pp[z * (D + 2) + d], wz, acc);
    }
    bf16* o = a.out + (size_t)t * a.os + h * D + d;
    *o = a.gate ? f2b(at_gated(a, b2f(f2b(acc / l)), t, h, d)) : f2b(acc / l);
}

// The gate of n rows' attention in place (a prompt chunk's, or a decode over 12-bit pages). Block (row, query head), d threads.
extern "C" __global__ void attn_gate(AttnArgs a) {
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    bf16* o = a.out + (size_t)t * a.os + h * a.d + d;
    *o = f2b(at_gated(a, b2f(*o), t, h, d));
}

// ---- a prompt chunk's attention on the tensor cores -------------------------------------------------------------------------------------------------------------------
// attn_rows_t's arithmetic (at_score, exp2 of the score less the running maximum, the probabilities rounded to bf16 for the values, their fp32 sum dividing at the end)
// with Q K^T and P V as mma.sync m16n8k16 bf16 products. A block takes BM rows of one query head h and the value columns z DV .. z DV + DV - 1; 16 rows a warp, the
// keys 32 a tile from the ring, each row's keys lo_t .. kpos[t]. Shared memory: Q [BM][D + 8], K [32][D + 8], V [32][DV + 8] bf16 (rows padded by 16 bytes so ldmatrix
// meets no bank twice). Grid (ceil(n / BM), nq, D / DV).
__device__ __forceinline__ uint32_t at_pk(float x, float y) {
    __nv_bfloat162 v = __floats2bfloat162_rn(x, y);
    return *reinterpret_cast<uint32_t*>(&v);
}

template <int D, int DV, int BM>
__device__ __forceinline__ void attn_fa_t(const AttnArgs& a) {
    constexpr int KT = 32, QS = D + 8, VS = DV + 8, NT = DV / 8, W = BM / 16;
    (void)W;
    extern __shared__ __align__(16) unsigned char at_fa_sm[];
    bf16* Qs = reinterpret_cast<bf16*>(at_fa_sm);
    bf16* Ks = Qs + BM * QS;
    bf16* Vs = Ks + KT * QS;
    __shared__ int s_lo, s_hi;
    const int n = a.n, nq = a.nq, nk = a.nk, window = a.window;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3;
    const int h = blockIdx.y, kvh = h / (nq / nk), r0 = blockIdx.x * BM, v0 = blockIdx.z * DV;
    for (int c = threadIdx.x; c < BM * (D / 8); c += blockDim.x) {
        const int r = c / (D / 8), q8 = c % (D / 8);
        uint4 v = make_uint4(0, 0, 0, 0);
        if (r0 + r < n) v = reinterpret_cast<const uint4*>(a.qo + (size_t)(r0 + r) * nq * D + h * D)[q8];
        *reinterpret_cast<uint4*>(Qs + r * QS + q8 * 8) = v;
    }
    if (threadIdx.x == 0) {
        int lo = 1 << 30, hi = 0;
        for (int r = r0; r < min(n, r0 + BM); r++) {
            hi = max(hi, a.kpos[r] + 1);
            lo = min(lo, window > 0 ? max(0, a.kpos[r] - window + 1) : 0);
        }
        s_lo = min(lo, hi);
        s_hi = hi;
    }
    __syncthreads();
    const int ra = r0 + warp * 16 + g, rb = ra + 8;
    const int pa = ra < n ? a.kpos[ra] : -1, pb = rb < n ? a.kpos[rb] : -1;
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
            if (k0 + j < s_hi) v = reinterpret_cast<const uint4*>(a.kc + (at_place(a.tab, k0 + j, a.cap) * nk + kvh) * D)[q8];
            *reinterpret_cast<uint4*>(Ks + j * QS + q8 * 8) = v;
        }
        for (int c = threadIdx.x; c < KT * (DV / 8); c += blockDim.x) {
            const int j = c / (DV / 8), q8 = c % (DV / 8);
            uint4 v = make_uint4(0, 0, 0, 0);
            if (k0 + j < s_hi) v = reinterpret_cast<const uint4*>(a.vc + (at_place(a.tab, k0 + j, a.cap) * nk + kvh) * D + v0)[q8];
            *reinterpret_cast<uint4*>(Vs + j * VS + q8 * 8) = v;
        }
        __syncthreads();
        // S [16 rows][32 keys] = Q K^T: 4 n-tiles of 8 keys
        float s[4][4];
#pragma unroll
        for (int j = 0; j < 4; j++) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
#pragma unroll 4
        for (int ks = 0; ks < D / 16; ks++) {
            uint32_t qa[4];
            ldsm4(qa, s_addr(Qs + (warp * 16 + (lane & 15)) * QS + ks * 16 + (lane >> 4) * 8));
#pragma unroll
            for (int np = 0; np < 2; np++) {
                uint32_t b[4];
                const int m = lane >> 3;
                ldsm4(b, s_addr(Ks + (np * 16 + (m >> 1) * 8 + (lane & 7)) * QS + ks * 16 + (m & 1) * 8));
                mma16816(s[2 * np], qa, b[0], b[1]);
                mma16816(s[2 * np + 1], qa, b[2], b[3]);
            }
        }
        // the scores (at_score), masked; the running maxima and sums (a row's 4 lanes share g)
        float mxa = ma, mxb = mb;
#pragma unroll
        for (int j = 0; j < 4; j++) {
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const int kp = k0 + j * 8 + 2 * t4 + e;
                s[j][e] = kp >= la && kp <= pa ? at_score(a, s[j][e]) : -INFINITY;
                s[j][2 + e] = kp >= lb && kp <= pb ? at_score(a, s[j][2 + e]) : -INFINITY;
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
            const uint32_t pa4[4] = {at_pk(s[2 * kk][0], s[2 * kk][1]), at_pk(s[2 * kk][2], s[2 * kk][3]), at_pk(s[2 * kk + 1][0], s[2 * kk + 1][1]), at_pk(s[2 * kk + 1][2], s[2 * kk + 1][3])};
#pragma unroll
            for (int np = 0; np < NT / 2; np++) {
                uint32_t b[4];
                const int m = lane >> 3;
                ldsm4t(b, s_addr(Vs + (kk * 16 + (m & 1) * 8 + (lane & 7)) * VS + np * 16 + (m >> 1) * 8));
                mma16816(acc[2 * np], pa4, b[0], b[1]);
                mma16816(acc[2 * np + 1], pa4, b[2], b[3]);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < NT; j++) {
        const int c = v0 + j * 8 + 2 * t4;
        if (ra < n) *reinterpret_cast<__nv_bfloat162*>(a.out + (size_t)ra * a.os + h * D + c) = __floats2bfloat162_rn(acc[j][0] / sa, acc[j][1] / sa);
        if (rb < n) *reinterpret_cast<__nv_bfloat162*>(a.out + (size_t)rb * a.os + h * D + c) = __floats2bfloat162_rn(acc[j][2] / sb, acc[j][3] / sb);
    }
}

extern "C" __global__ void __launch_bounds__(128, 1) attn_fa64(AttnArgs a) { attn_fa_t<64, 64, 64>(a); }
extern "C" __global__ void __launch_bounds__(128, 1) attn_fa128(AttnArgs a) { attn_fa_t<128, 128, 64>(a); }
extern "C" __global__ void __launch_bounds__(128, 1) attn_fa256(AttnArgs a) { attn_fa_t<256, 256, 64>(a); }

// The 512-wide heads on the tensor cores, 8 warps a block: 32 rows of one query head, warp w = 4 rg + ds the rows 16 rg .. 16 rg + 15 and the channels 128 ds ..
// 128 ds + 127. Its queries' A fragments stay in registers. Q K^T: each warp its 128 channels' share of the 16 x 32 scores, the four shares added through shared memory in
// a fixed order (every warp of a row group the same sums); then attn_fa_t's score, mask and online softmax (the four warps alike), and P V into the warp's own 128 value
// columns: the scores are made once. Shared memory: K and V [32][520] bf16 (the queries pass through V's space first) and the shares [2][4][16][33] floats: 83,456
// bytes. Grid (ceil(n / 32), nq).
#define AT_SP 33
extern "C" __global__ void __launch_bounds__(256, 1) attn_fa512(AttnArgs a) {
    constexpr int D = 512, KT = 32, QS = D + 8, BM = 32;
    extern __shared__ __align__(16) unsigned char at_fa_sm[];
    bf16* Ks = reinterpret_cast<bf16*>(at_fa_sm);
    bf16* Vs = Ks + KT * QS;
    float* Sp = reinterpret_cast<float*>(Vs + KT * QS);
    __shared__ int s_lo, s_hi;
    const int n = a.n, nq = a.nq, nk = a.nk, window = a.window;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3, rg = warp >> 2, ds = warp & 3;
    const int h = blockIdx.y, kvh = h / (nq / nk), r0 = blockIdx.x * BM;
    for (int c = threadIdx.x; c < BM * (D / 8); c += blockDim.x) {
        const int r = c / (D / 8), q8 = c % (D / 8);
        uint4 v = make_uint4(0, 0, 0, 0);
        if (r0 + r < n) v = reinterpret_cast<const uint4*>(a.qo + (size_t)(r0 + r) * nq * D + h * D)[q8];
        *reinterpret_cast<uint4*>(Vs + r * QS + q8 * 8) = v;
    }
    if (threadIdx.x == 0) {
        int lo = 1 << 30, hi = 0;
        for (int r = r0; r < min(n, r0 + BM); r++) {
            hi = max(hi, a.kpos[r] + 1);
            lo = min(lo, window > 0 ? max(0, a.kpos[r] - window + 1) : 0);
        }
        s_lo = min(lo, hi);
        s_hi = hi;
    }
    __syncthreads();
    uint32_t qa[8][4];
#pragma unroll
    for (int ks = 0; ks < 8; ks++) ldsm4(qa[ks], s_addr(Vs + (rg * 16 + (lane & 15)) * QS + ds * 128 + ks * 16 + (lane >> 4) * 8));
    const int ra = r0 + rg * 16 + g, rb = ra + 8;
    const int pa = ra < n ? a.kpos[ra] : -1, pb = rb < n ? a.kpos[rb] : -1;
    const int la = window > 0 ? pa - window + 1 : 0, lb = window > 0 ? pb - window + 1 : 0;
    float acc[16][4];
#pragma unroll
    for (int j = 0; j < 16; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    float ma = -INFINITY, mb = -INFINITY, sa = 0.f, sb = 0.f;
    float* myS = Sp + (rg * 4 + ds) * 16 * AT_SP;
    for (int k0 = s_lo; k0 < s_hi; k0 += KT) {
        __syncthreads();  // (the queries, or the tiles before, are read)
        for (int c = threadIdx.x; c < 2 * KT * (D / 8); c += blockDim.x) {
            const bool isv = c >= KT * (D / 8);
            const int cc = isv ? c - KT * (D / 8) : c, j = cc / (D / 8), q8 = cc % (D / 8);
            uint4 v = make_uint4(0, 0, 0, 0);
            if (k0 + j < s_hi) v = reinterpret_cast<const uint4*>((isv ? a.vc : a.kc) + (at_place(a.tab, k0 + j, a.cap) * nk + kvh) * D)[q8];
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
            myS[g * AT_SP + j * 8 + 2 * t4] = s[j][0];
            myS[g * AT_SP + j * 8 + 2 * t4 + 1] = s[j][1];
            myS[(g + 8) * AT_SP + j * 8 + 2 * t4] = s[j][2];
            myS[(g + 8) * AT_SP + j * 8 + 2 * t4 + 1] = s[j][3];
        }
        __syncthreads();
        const float* gS = Sp + rg * 4 * 16 * AT_SP;
        float mxa = ma, mxb = mb;
#pragma unroll
        for (int j = 0; j < 4; j++) {
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const int col = j * 8 + 2 * t4 + e, kp = k0 + col;
                float xa = 0.f, xb = 0.f;
#pragma unroll
                for (int d4 = 0; d4 < 4; d4++) {
                    xa += gS[(d4 * 16 + g) * AT_SP + col];
                    xb += gS[(d4 * 16 + g + 8) * AT_SP + col];
                }
                s[j][e] = kp >= la && kp <= pa ? at_score(a, xa) : -INFINITY;
                s[j][2 + e] = kp >= lb && kp <= pb ? at_score(a, xb) : -INFINITY;
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
            const uint32_t pa4[4] = {at_pk(s[2 * kk][0], s[2 * kk][1]), at_pk(s[2 * kk][2], s[2 * kk][3]), at_pk(s[2 * kk + 1][0], s[2 * kk + 1][1]), at_pk(s[2 * kk + 1][2], s[2 * kk + 1][3])};
#pragma unroll
            for (int np = 0; np < 8; np++) {
                uint32_t b[4];
                const int m = lane >> 3;
                ldsm4t(b, s_addr(Vs + (kk * 16 + (m & 1) * 8 + (lane & 7)) * QS + ds * 128 + np * 16 + (m >> 1) * 8));
                mma16816(acc[2 * np], pa4, b[0], b[1]);
                mma16816(acc[2 * np + 1], pa4, b[2], b[3]);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < 16; j++) {
        const int c = ds * 128 + j * 8 + 2 * t4;
        if (ra < n) *reinterpret_cast<__nv_bfloat162*>(a.out + (size_t)ra * a.os + h * D + c) = __floats2bfloat162_rn(acc[j][0] / sa, acc[j][1] / sa);
        if (rb < n) *reinterpret_cast<__nv_bfloat162*>(a.out + (size_t)rb * a.os + h * D + c) = __floats2bfloat162_rn(acc[j][2] / sb, acc[j][3] / sb);
    }
}
