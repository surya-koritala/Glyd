// The attention layer kind's kernels (src/runtime/kinds.rs `Attn`; the description executor's `Op::Attn`, src/dynamic/attn.rs, runs them for every dense-attention layer
// of every family). One set, its arithmetic and shapes in `AttnArgs`: the preparation (the q and k norms, the rotary, the cache write, the values' norm), the decode
// step's attention (the keys in parts, then combined), a prompt chunk's attention for heads of 64 and 128 on the tensor cores (heads of 256 and 512: the GPU library's
// glyd_gpu_kv_prefill_bf16, the kernel of its 12-bit pages too), a warp-per-head kernel for any shape, and the output gate. Head widths 64, 128, 256 and 512 are template
// instances. Included by engine.cu after qwen35.cu (bf2, warp_max); the next product's prefetch (NextPf) is engine.cu's.
//
// Arithmetic (the kind's rounding points, src/runtime/kinds.rs `AttnRounding`): bf16 in memory, fp32 inside. `round` (Eager, Sdpa): transformers' bf16 rounding
// points in the preparation and the gate (a norm rounds once, the queries' scale rounds, each rotary product and their sum round, the gate's sigmoid rounds before
// its product; the norm's factor 1 / sqrt(mean + eps) exact); else (Flash) one rounding at the end of each (the norm's factor rsqrtf). `sbf16` (Eager): a score is
// the bf16 of its dot product, then of its scale. The probabilities are rounded to bf16 for the values (as FlashAttention; Sdpa's are f32 in the reference), their
// fp32 sum divides at the end.

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
    const bf16* gate;  // the output gate's values, head h of row t at t gs + h g_step: a value a dim, or one for the head (g_head); null: no gate
    float* part;       // a decode step's parts [(t nq + h) S + z][d + 2]
    const int* rpos;   // each row's M-RoPE position (t, h, w) at 6 r (src/dynamic/images.rs), each angle turning by the one `rax` names; null: qpos alone
    const bf16* sink;  // a learned logit a query head [nq] that joins each row's softmax and attends to nothing (`kinds::Attn::sinks`); null: none
    long long tstride; // row t's table at tab + t tstride (a pass's decode rows of several sequences: src/dynamic/rows.rs `RowTabs`); 0: one table for every row
    unsigned long long rax; // two bits an angle: the position (0 t, 1 h, 2 w) of rpos its M-RoPE section turns by
    int d, stride, q_step, k_off, v_off;
    int norm;          // the q and k norms: 0 none, 1 x w, 2 x (1 + w), 3 no weight
    int v_norm;        // the values normed without a weight
    int span, pairs;   // the rotary: pairs (i, i + span / 2) of the first span dims, the first `pairs` of them turned
    int cap;           // the ring's places (cap < 0: row r's own place, a staging buffer of the step's rows)
    int n, nq, nk, window;
    int os, gs, g_step, gnorm;
    int g_head;        // the gate is one value a head (every dim of head h times the sigmoid of the value at t gs + h g_step), not a value a dim
    int round, sbf16, S;
    float q_scale, eps, scale, sl2, softcap;
};

// A key's place: through the block table where the sequence has one (src/dynamic/pages.rs), else the ring's slot p % cap.
__device__ __forceinline__ size_t at_place(const int* tab, int p, int cap) { return tab ? (size_t)tab[p >> 4] * 16 + (p & 15) : (size_t)(p % cap); }

__device__ __forceinline__ float at_r(float x, int round) { return round ? b2f(f2b(x)) : x; }

// A norm's factor over d values whose squares sum to ss.
__device__ __forceinline__ float at_rs(float ss, int d, float eps, int round) { return round ? 1.f / sqrtf(ss / (float)d + eps) : rsqrtf(ss / (float)d + eps); }

// A score in log2 units: the scaled dot product, soft-capped (cap tanh(s / cap)) where the kind is; with `sbf16` (transformers' eager bf16) the dot product,
// its scale and its cap each rounded to bf16.
// (The prompt kernel attn_pf_t and the GPU library's kernels have no sink: an attention with sinks takes attn_rows_t and the parts kernel with attn_merge.)
__device__ __forceinline__ float at_score(const AttnArgs& a, float s) {
    if (a.sbf16) {
        s = b2f(f2b(b2f(f2b(s)) * a.scale));
        if (a.softcap > 0.f) s = b2f(f2b(a.softcap * tanhf(s / a.softcap)));
        return s * 1.4426950408889634f;
    }
    if (a.softcap > 0.f) return a.softcap * tanhf(s * a.scale / a.softcap) * 1.4426950408889634f;
    return s * a.sl2;
}

// A sink's share of a row's softmax sum, the sum being of 2^(score - m) (m the row's largest score, log2 units): 2^(sink log2(e) - m). The sink is a logit as the
// scores are after their scale, so it is not scaled. A row with no key (m = -inf) is all sink: the sum is then infinite and the output 0.
__device__ __forceinline__ float at_sink(const AttnArgs& a, int h, float m) { return a.sink ? exp2f(b2f(a.sink[h]) * 1.4426950408889634f - m) : 0.f; }

// ---- the preparation --------------------------------------------------------------------------------------------------------------------------------------------------
// Row r's head hh (a query head below nq, else key head hh - nq with its values): the norm (one block sum), the queries' scale, the rotary (value j < span with
// its pair j +- span / 2, angle j mod span / 2 where that is below `pairs`; an M-RoPE row's angle k from the table's row of the position its section turns by),
// the query to qo, the key and the value (normed or as it is) to the ring at the row's position. Block (row, head), d / 2 threads: thread t holds t and t + d / 2.
// Grid y: nq + nk heads, or nq where no keys are written (a draft model's). blockIdx.z above 0: `pf`'s prefetch of the next product. A padding row (a negative
// position: a decode step padded to its graph's rows) writes nothing.
__device__ __forceinline__ float at_rot(const AttnArgs& a, const float* xs, int j, float v, int p, int r) {
    const int half = a.span >> 1;
    if (j >= a.span) return v;
    const bool lo = j < half;
    const int k = lo ? j : j - half;
    if (k >= a.pairs) return v;
    if (a.rpos) p = a.rpos[6 * r + (int)((a.rax >> (2 * k)) & 3)];
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
    if (p < 0) return;
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
        v0 = at_rot(a, xs, t, v0, p, r);
        v1 = at_rot(a, xs, t + H, v1, p, r);
    }
    const size_t slot = a.cap < 0 ? (size_t)r : at_place(a.tab ? a.tab + (size_t)r * a.tstride : nullptr, p, a.cap);
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

#define AT_PF if (blockIdx.z > 0) return next_pf(pf, (blockIdx.z - 1) * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x, (gridDim.z - 1) * gridDim.x * gridDim.y)
extern "C" __global__ void __launch_bounds__(32) attn_prep64(AttnArgs a, NextPf pf) {
    AT_PF;
    attn_prep_t<64>(a);
}
extern "C" __global__ void __launch_bounds__(64) attn_prep128(AttnArgs a, NextPf pf) {
    AT_PF;
    attn_prep_t<128>(a);
}
extern "C" __global__ void __launch_bounds__(128) attn_prep256(AttnArgs a, NextPf pf) {
    AT_PF;
    attn_prep_t<256>(a);
}
extern "C" __global__ void __launch_bounds__(256) attn_prep512(AttnArgs a, NextPf pf) {
    AT_PF;
    attn_prep_t<512>(a);
}

// ---- the attention, a warp a query head ------------------------------------------------------------------------------------------------------------------------------
#define AT_KB 32  // keys a tile

// Causal attention of the n rows of qo over the ring: row t sees keys lo_t .. kpos[t], lo_t = kpos[t] - window + 1 (0 for window 0, and at least 0). A block takes R rows
// and the G = nq / nk query heads of one key-value head (a warp each), the keys AT_KB at a time through shared memory: lane j scores key j, then each lane sums D / 32 output
// dims. Dynamic shared memory: (AT_KB (D/2 + 1) + AT_KB D/2 + R G D/2) * 4 bytes.
// SPLIT (a decode step, R = 1): block z takes the S-th part z of the row's keys and writes its outputs, maximum (log2 units) and sum to part [(t nq + h) S + z][D + 2]; the
// block's warps past the first G only fill the tiles (more loads in flight). Its row's own table (tstride: the rows of several sequences in one launch); a padding row (a negative
// position) has no keys.
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
    const int* tab = SPLIT && a.tab ? a.tab + (size_t)blockIdx.x * a.tstride : a.tab;
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
            if (k0 + j < kmax) v = reinterpret_cast<const uint4*>((isv ? a.vc : a.kc) + (at_place(tab, k0 + j, a.cap) * nk + kvh) * D)[q4];
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
    const float den = l_run + at_sink(a, h, m_run);
#pragma unroll
    for (int e = 0; e < PER; e++) o[e] = f2b(acc[e] / den);
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
// (one rounding where `round`), times sigmoid(gate) (the sigmoid rounded first where `round`); the gate's value the dim's own, or the head's one (`g_head`).
__device__ __forceinline__ float at_gated(const AttnArgs& a, float o, int t, int h, int d) {
    if (a.gnorm) o = at_r(o * at_rs(block_sum(o * o), a.d, a.eps, a.round), a.round);
    const float g = b2f(a.gate[(size_t)t * a.gs + (size_t)h * a.g_step + (a.g_head ? 0 : d)]);
    return a.round ? o * b2f(f2b(1.f / (1.f + expf(-g)))) : o / (1.f + expf(-g));
}

// The S parts of a decode step combined: out = bf16(sum of the parts' outputs, each times 2^(its maximum - the largest), / the sum likewise), then the gate where
// there is one. Block (row, query head), d threads (a dim each). A row with no keys (a padding row) is zeros, as the library's decode gives it.
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
    if (mx > -INFINITY) l += at_sink(a, h, mx);
    const float v = l > 0.f ? acc / l : 0.f;
    *o = a.gate ? f2b(at_gated(a, b2f(f2b(v)), t, h, d)) : f2b(v);
}

// The gate of n rows' attention in place (a prompt chunk's, or a decode over 12-bit pages). Block (row, query head), d threads.
extern "C" __global__ void attn_gate(AttnArgs a) {
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    bf16* o = a.out + (size_t)t * a.os + h * a.d + d;
    *o = f2b(at_gated(a, b2f(*o), t, h, d));
}

// ---- a prompt chunk's attention on the tensor cores -------------------------------------------------------------------------------------------------------------------
// FlashAttention-2's loop with attn_rows_t's arithmetic (at_score, exp2 of the score less the
// running maximum, the probabilities rounded to bf16 for the values, their fp32 sum dividing at the end), Q K^T and P V as mma.sync m16n8k16 bf16 products. A block takes
// one key-value head and PF_ROWS of the chunk's rows of its query heads, packed token by token (row r: token r / G, query head kvh G + r % G), so that a token's G query
// heads share every tile of keys and values the block reads; 4 warps of 16 rows, the queries' A fragments in registers. The keys PF_KT a tile at positions that are multiples
// of PF_KT, from the tile of the block's first key (its first row's window) to its last row's: a tile no key of which is a row's leaves that row's sums as they were (its
// probabilities 0, its rescale 1), so a row's sums are those of its own position and window alone, the same bits in any chunk, at any start, beside any rows. K and V
// each one tile in shared memory, filled by cp.async as FlashAttention-2 orders it: V(j)'s copy runs while S = Q K(j)^T and its softmax are made, K(j + 1)'s while
// P V(j) is. Only a tile that holds a key outside some row's window (the diagonal's, a window's first) is masked; a key outside the block's is read as zeros (a recycled
// place's bytes can be infinities, and zero times infinity is not zero), and such a tile is always masked. Shared memory: K and V [PF_KT][D + 8] bf16 (rows 16 bytes apart
// in the banks, so ldmatrix meets no bank twice). Grid (ceil(n G / PF_ROWS), nk), PF_ROWS / 16 warps.
__device__ __forceinline__ uint32_t at_pk(float x, float y) {
    __nv_bfloat162 v = __floats2bfloat162_rn(x, y);
    return *reinterpret_cast<uint32_t*>(&v);
}

constexpr int PF_KT = 64;    // keys a tile
constexpr int PF_ROWS = 64;  // packed rows a block

template <int D>
__device__ __forceinline__ void attn_pf_t(const AttnArgs& a) {
    constexpr int KS = D + 8, C8 = D / 8, NK = PF_KT / 8, NT = 32 * PF_ROWS / 16;
    extern __shared__ __align__(16) unsigned char at_pf_sm[];
    bf16* Ks = reinterpret_cast<bf16*>(at_pf_sm);
    bf16* Vs = Ks + PF_KT * KS;
    __shared__ int s_lo, s_hi, s_in0, s_in1;
    const int n = a.n, nq = a.nq, nk = a.nk, G = nq / nk, W = a.window;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3, m8 = lane >> 3;
    const int kvh = blockIdx.y, r0 = (gridDim.x - 1 - blockIdx.x) * PF_ROWS;  // (the blocks with the most keys first)
    if (warp == 0) {
        // the block's keys lo (its first window's first) .. hi (past its last row's); in0 .. in1: the keys every one of its rows takes (its tokens a lane each)
        int lo = 1 << 30, hi = 0, in0 = 0, in1 = 1 << 30;
        for (int t = r0 / G + lane; t < n && t * G < r0 + PF_ROWS; t += 32) {
            const int p = a.kpos[t], l = W > 0 ? max(0, p - W + 1) : 0;
            lo = min(lo, l);
            hi = max(hi, p + 1);
            in0 = max(in0, l);
            in1 = min(in1, p + 1);
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            lo = min(lo, __shfl_xor_sync(0xffffffffu, lo, o));
            hi = max(hi, __shfl_xor_sync(0xffffffffu, hi, o));
            in0 = max(in0, __shfl_xor_sync(0xffffffffu, in0, o));
            in1 = min(in1, __shfl_xor_sync(0xffffffffu, in1, o));
        }
        if (lane == 0) {
            s_lo = min(lo, hi);
            s_hi = hi;
            s_in0 = in0;
            s_in1 = in1;
        }
    }
    // the warp's rows ra (g) and rb (g + 8): their tokens, query heads, positions and windows' first keys (a row past the chunk's takes no key)
    const int ra = r0 + warp * 16 + g, rb = ra + 8;
    const bool ia = ra < n * G, ib = rb < n * G;
    const int ta = ia ? ra / G : 0, tb = ib ? rb / G : 0, ha = kvh * G + ra % G, hb = kvh * G + rb % G;
    const int pa = ia ? a.kpos[ta] : -1, pb = ib ? a.kpos[tb] : -1;
    const int la = W > 0 ? pa - W + 1 : 0, lb = W > 0 ? pb - W + 1 : 0;
    // the queries' A fragments: rows ra and rb, 16 channels a step
    uint32_t qa[D / 16][4];
    {
        const uint32_t* qx = reinterpret_cast<const uint32_t*>(a.qo + ((size_t)ta * nq + ha) * D);
        const uint32_t* qy = reinterpret_cast<const uint32_t*>(a.qo + ((size_t)tb * nq + hb) * D);
#pragma unroll
        for (int ks = 0; ks < D / 16; ks++) {
            qa[ks][0] = ia ? qx[8 * ks + t4] : 0u;
            qa[ks][1] = ib ? qy[8 * ks + t4] : 0u;
            qa[ks][2] = ia ? qx[8 * ks + 4 + t4] : 0u;
            qa[ks][3] = ib ? qy[8 * ks + 4 + t4] : 0u;
        }
    }
    __syncthreads();
    const int lo = s_lo, hi = s_hi, in0 = s_in0, in1 = s_in1;
    // a tile of keys (src a.kc) or values (a.vc) from position k0 into dst: thread i the 16-byte chunks i, i + NT, ... (chunk c: key c / C8, its channels 8 (c % C8) on)
    const auto fill = [&](bf16* dst, const bf16* src, int k0) {
#pragma unroll
        for (int i = 0; i < PF_KT * C8 / NT; i++) {
            const int c = threadIdx.x + i * NT, j = c / C8, c8 = c % C8, kp = k0 + j;
            const bool ok = kp >= lo && kp < hi;
            cp16(s_addr(dst + j * KS + c8 * 8), src + (ok ? (at_place(a.tab, kp, a.cap) * nk + kvh) * D + c8 * 8 : 0), ok);
        }
    };
    float acc[D / 8][4];
#pragma unroll
    for (int j = 0; j < D / 8; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    float ma = -INFINITY, mb = -INFINITY, sa = 0.f, sb = 0.f;
    const int j0 = lo / PF_KT, j1 = (hi + PF_KT - 1) / PF_KT;
    if (j0 < j1) fill(Ks, a.kc, j0 * PF_KT);
    cp_commit();
    for (int jt = j0; jt < j1; jt++) {
        const int k0 = jt * PF_KT;
        cp_wait<0>();
        __syncthreads();  // K(jt) is in; every warp is done with V(jt - 1)
        fill(Vs, a.vc, k0);
        cp_commit();
        // S [16 rows][PF_KT keys] = Q K^T: NK n-tiles of 8 keys
        float s[NK][4];
#pragma unroll
        for (int j = 0; j < NK; j++) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
#pragma unroll
        for (int ks = 0; ks < D / 16; ks++) {
#pragma unroll
            for (int np = 0; np < NK / 2; np++) {
                uint32_t b[4];
                ldsm4(b, s_addr(Ks + (np * 16 + (m8 >> 1) * 8 + (lane & 7)) * KS + ks * 16 + (m8 & 1) * 8));
                mma16816(s[2 * np], qa[ks], b[0], b[1]);
                mma16816(s[2 * np + 1], qa[ks], b[2], b[3]);
            }
        }
        // the scores (at_score), masked where the tile is not every row's whole; the running maxima and sums (a row's 4 lanes share g)
        const bool edge = k0 < in0 || k0 + PF_KT > in1;
        float mxa = ma, mxb = mb;
#pragma unroll
        for (int j = 0; j < NK; j++) {
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const int kp = k0 + j * 8 + 2 * t4 + e;
                s[j][e] = !edge || (kp >= la && kp <= pa) ? at_score(a, s[j][e]) : -INFINITY;
                s[j][2 + e] = !edge || (kp >= lb && kp <= pb) ? at_score(a, s[j][2 + e]) : -INFINITY;
                mxa = fmaxf(mxa, s[j][e]);
                mxb = fmaxf(mxb, s[j][2 + e]);
            }
        }
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 1));
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 2));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 1));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 2));
        const float ca = mxa == -INFINITY ? 1.f : exp2f(ma - mxa), cb = mxb == -INFINITY ? 1.f : exp2f(mb - mxb);
        float ta_ = 0.f, tb_ = 0.f;
#pragma unroll
        for (int j = 0; j < NK; j++) {
#pragma unroll
            for (int e = 0; e < 2; e++) {
                s[j][e] = mxa == -INFINITY ? 0.f : exp2f(s[j][e] - mxa);
                s[j][2 + e] = mxb == -INFINITY ? 0.f : exp2f(s[j][2 + e] - mxb);
                ta_ += s[j][e];
                tb_ += s[j][2 + e];
            }
        }
        ta_ += __shfl_xor_sync(0xffffffffu, ta_, 1);
        ta_ += __shfl_xor_sync(0xffffffffu, ta_, 2);
        tb_ += __shfl_xor_sync(0xffffffffu, tb_, 1);
        tb_ += __shfl_xor_sync(0xffffffffu, tb_, 2);
        sa = sa * ca + ta_;
        sb = sb * cb + tb_;
        ma = mxa;
        mb = mxb;
#pragma unroll
        for (int j = 0; j < D / 8; j++) {
            acc[j][0] *= ca;
            acc[j][1] *= ca;
            acc[j][2] *= cb;
            acc[j][3] *= cb;
        }
        cp_wait<0>();
        __syncthreads();  // V(jt) is in; every warp is done with K(jt)
        if (jt + 1 < j1) fill(Ks, a.kc, k0 + PF_KT);
        cp_commit();
        // O += bf16(P) V: the score tiles as the A operand, 16 keys a k-step; V's rows through ldmatrix.trans
#pragma unroll
        for (int kk = 0; kk < PF_KT / 16; kk++) {
            const uint32_t p4[4] = {at_pk(s[2 * kk][0], s[2 * kk][1]), at_pk(s[2 * kk][2], s[2 * kk][3]), at_pk(s[2 * kk + 1][0], s[2 * kk + 1][1]), at_pk(s[2 * kk + 1][2], s[2 * kk + 1][3])};
#pragma unroll
            for (int np = 0; np < D / 16; np++) {
                uint32_t b[4];
                ldsm4t(b, s_addr(Vs + (kk * 16 + (m8 & 1) * 8 + (lane & 7)) * KS + np * 16 + (m8 >> 1) * 8));
                mma16816(acc[2 * np], p4, b[0], b[1]);
                mma16816(acc[2 * np + 1], p4, b[2], b[3]);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < D / 8; j++) {
        const int c = j * 8 + 2 * t4;
        if (ia) *reinterpret_cast<__nv_bfloat162*>(a.out + (size_t)ta * a.os + ha * D + c) = __floats2bfloat162_rn(acc[j][0] / sa, acc[j][1] / sa);
        if (ib) *reinterpret_cast<__nv_bfloat162*>(a.out + (size_t)tb * a.os + hb * D + c) = __floats2bfloat162_rn(acc[j][2] / sb, acc[j][3] / sb);
    }
}

// (heads of 256 and 512 take the GPU library's prompt kernel, glyd_gpu_kv_prefill_bf16, and over 12-bit pages glyd_gpu_kv_prefill: the same tiles, so the same bits; heads
// of 64 and 128 these, over 12-bit pages as over bf16 ones: the pages unpacked to a bf16 ring first, src/dynamic/attn.rs `attn_read`)
extern "C" __global__ void __launch_bounds__(128, 2) attn_pf64(AttnArgs a) { attn_pf_t<64>(a); }
extern "C" __global__ void __launch_bounds__(128, 2) attn_pf128(AttnArgs a) { attn_pf_t<128>(a); }
