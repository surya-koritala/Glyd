// Tiered v2 decode variants and a step kernel with options, for the harness (include after glyd_gpu.cu).
//
// OPT bits (Tiered2o<OPT>):
//   1  the scan by predicated shuffles (shfl.up's own predicate: 2 instructions a step, not 4)
//   2  the tier-1 groups' escape bytes through a register window (4 words from the lane's first byte, advanced a group), where
//      no lane has more than 16 escapes; else as v1
//   4  the groups' table 16 times over in shared memory (a lane pair a copy: its lookups conflict less)
//   128 the fast path only (no tier-1 escapes > 512 branch, tier-3 digits and literals always in the window, <= 32 literals): 1024, 2048, 4096 each
//   of those three alone (1024: T1 <= 512; 2048: tier-3 digits in the window; 4096: literals in the window and <= 32)
//   16384 the groups' table 32 times over (a copy a lane: no bank conflicts, 32 KB a block), 32768 16 times over (two lanes a bank)
// The step kernel's UNR: 1 as mma_gemm_kernel's mma_steps, 2 two steps a pass in two register sets (no copies, the next loads
// after a step's decode).
#pragma once

__device__ __forceinline__ uint32_t scan_p(uint32_t n, int lane, uint32_t& total) {
    uint32_t before = n;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        uint32_t v;
        asm volatile("{\n.reg .pred p;\nshfl.sync.up.b32 %0|p, %1, %2, 0, 0xffffffff;\n@p add.u32 %1, %1, %0;\n}\n" : "=r"(v), "+r"(before) : "r"(o));
    }
    total = __shfl_sync(FULL, before, 31);
    return before - n;
}


// A scan in 3 levels (shuffles by 1-3, 4-12, 16: each level's shuffles independent), by predicated shuffles.
__device__ __forceinline__ uint32_t scan_r4(uint32_t n, int lane, uint32_t& total) {
    uint32_t a = n, p1, p2, p3;
    asm volatile("{\n.reg .pred p;\nshfl.sync.up.b32 %0|p, %4, 1, 0, 0xffffffff;\n@!p mov.u32 %0, 0;\nshfl.sync.up.b32 %1|p, %4, 2, 0, 0xffffffff;\n@!p mov.u32 %1, 0;\nshfl.sync.up.b32 %2|p, %4, 3, 0, 0xffffffff;\n@!p mov.u32 %2, 0;\nadd.u32 %3, %4, %0;\nadd.u32 %3, %3, %1;\nadd.u32 %3, %3, %2;\n}\n" : "=r"(p1), "=r"(p2), "=r"(p3), "=r"(a) : "r"(n));
    uint32_t b;
    asm volatile("{\n.reg .pred p;\nshfl.sync.up.b32 %0|p, %4, 4, 0, 0xffffffff;\n@!p mov.u32 %0, 0;\nshfl.sync.up.b32 %1|p, %4, 8, 0, 0xffffffff;\n@!p mov.u32 %1, 0;\nshfl.sync.up.b32 %2|p, %4, 12, 0, 0xffffffff;\n@!p mov.u32 %2, 0;\nadd.u32 %3, %4, %0;\nadd.u32 %3, %3, %1;\nadd.u32 %3, %3, %2;\n}\n" : "=r"(p1), "=r"(p2), "=r"(p3), "=r"(b) : "r"(a));
    uint32_t c;
    asm volatile("{\n.reg .pred p;\nshfl.sync.up.b32 %0|p, %2, 16, 0, 0xffffffff;\n@!p mov.u32 %0, 0;\nadd.u32 %1, %2, %0;\n}\n" : "=r"(p1), "=r"(c) : "r"(b));
    total = __shfl_sync(FULL, c, 31);
    return c - n;
}

// The table's lookup: tab[field] (OPT & 4: 16 copies, field * 16 + lane / 2).
template <int OPT> __device__ __forceinline__ uint32_t tl(const uint32_t* tab, uint32_t field, int lane) {
    if constexpr (OPT & 16384) return tab[((field & 0xFF) << 5) | lane];  // 32 copies: every lane its own bank (the OR: one LOP3 with the mask)
    else if constexpr (OPT & 32768) return tab[((field & 0xFF) << 4) | (lane & 15)];  // 16 copies: two lanes a bank
    else if constexpr (OPT & 4) return tab[(field & 0xFF) * 16 + (lane >> 1)];
    else return tab[field & 0xFF];
}
constexpr int table_copies(int opt) { return (opt & 16384) ? 32 : (opt & (4 | 32768)) ? 16 : 1; }
template <int COPIES>
__device__ __forceinline__ void fill_groups_n(uint32_t* tab) {  // each entry computed once, stored COPIES times (a thread an entry, 16-byte stores)
    for (int f = threadIdx.x; f < 256; f += blockDim.x) {
        uint32_t z = nib8(f), fn = z & (z >> 1) & 0x1111u;
        uint32_t e = selector(z, fn) | (uint32_t)__popc(fn) << 16;
#pragma unroll
        for (int c = 0; c < COPIES; c += 4) *(uint4*)(tab + f * COPIES + c) = make_uint4(e, e, e, e);
    }
}

template <int OPT>
__device__ __forceinline__ uint32_t group_t(uint32_t field, uint32_t sym, uint32_t s[4], const uint32_t* tab, int lane) {
    uint32_t e = tl<OPT>(tab, field, lane), o = __byte_perm(sym, s[0], e);
    advance(s, e >> 16);
    return o;
}

__device__ __forceinline__ void fill_groups16(uint32_t* tab) {
    for (int i = threadIdx.x; i < 256 * 16; i += blockDim.x) {
        uint32_t f = i >> 4, z = nib8(f), fn = z & (z >> 1) & 0x1111u;
        tab[i] = selector(z, fn) | (uint32_t)__popc(fn) << 16;
    }
}

template <int OPT>
__device__ __forceinline__ bool exponents2o(const Step& st, Tiers ts, const uint8_t* __restrict__ blocks, int lane, uint32_t* s2, const uint32_t* tab, uint32_t ew[8]) {
    uint32_t e1[8];
    if constexpr (OPT & 32) {
#pragma unroll
        for (int q = 0; q < 8; q++) e1[q] = tl<OPT>(tab, st.pd[q >> 2] >> (8 * (q & 3)), lane);
    }
    uint32_t n1 = __popc(escapes(st.pd[0])) + __popc(escapes(st.pd[1]));
    uint32_t T1 = __reduce_add_sync(FULL, n1), words = (T1 + 15) >> 4;
    uint32_t b1, T2;
    uint32_t S3 = 0x01010101u * (ts.s3 >> 24);
    uint32_t o = 8 * (uint32_t)(st.blk & 3);
    // OPT & 256: the fast path with a way out (false, nothing used yet but the scratch) where its assumptions fail:
    // T1 > 512, the block's tier-3 digits and literals not within the window, more than 32 literals
    if constexpr (OPT & 256) {
        if (T1 > 512 || (uint32_t)(st.blk & 3) + (uint32_t)(st.end - st.blk) - 4 * words > 124) return false;
    }
    if (((OPT & (128 | 256 | 1024)) != 0) || T1 <= 512) {
        uint32_t up = __shfl_up_sync(FULL, st.tail, 1);
        uint32_t da = (uint32_t)lane < words ? __funnelshift_r(st.tail, lane ? up : st.tail1, 8 * (uint32_t)(st.end & 3)) : 0u;
        uint32_t na = __popc(escapes(da)), tot;
        uint32_t pre;
        if constexpr (OPT & 64) pre = scan_r4(n1 | na << 11, lane, tot);
        else if constexpr (OPT & 1) pre = scan_p(n1 | na << 11, lane, tot);
        else pre = scan(n1 | na << 11, lane, tot);
        b1 = pre & 0x7FF;
        T2 = tot >> 11;
        if constexpr (OPT & 256) {
            uint32_t r0e = (2 * T2 + 7) >> 3;
            if (o + 2 * T2 + 32 >= 992 || third((uint32_t)(st.end - st.blk) - r0e - 4 * words) > 32) return false;
        }
        uint32_t pa = o + 2 * ((pre >> 11) & 0x3FF);
        bool near = (OPT & (128 | 256 | 2048)) || o + 2 * T2 + 32 < 992;
        uint32_t d3 = near ? win_bits(st.win, pa) : mem_bits(blocks, st.blk, pa);
        uint32_t g3 = (__reduce_max_sync(FULL, na) + 3) >> 2;
        uint32_t y[4] = {0u, 0u, 0u, 0u}, q4[4];
#pragma unroll
        for (int q = 0; q < 4; q++)
            if (q < (int)g3) y[q] = __byte_perm(ts.s3, S3, tl<OPT>(tab, d3 >> (8 * q), lane));
#pragma unroll
        for (int q = 0; q < 4; q++) q4[q] = group_t<OPT>(da >> (8 * q), ts.s2, y, tab, lane);
        ((uint4*)s2)[lane] = make_uint4(q4[0], q4[1], q4[2], q4[3]);
    } else if constexpr (!(OPT & (128 | 256 | 1024))) {
        uint32_t tot1;
        b1 = scan(n1, lane, tot1);
        uint32_t carry = 0;
        for (uint32_t ch = 0; ch * 32 < words; ch++) {
            uint32_t c = ch * 32 + lane;
            uint32_t da = c < words ? load4(blocks + st.end - 4 * (c + 1)) : 0u;
            uint32_t na = __popc(escapes(da)), tot;
            uint32_t pre = scan(na, lane, tot);
            uint32_t pa = o + 2 * (carry + pre);
            uint32_t d3 = mem_bits(blocks, st.blk, pa);
            uint32_t g3 = (__reduce_max_sync(FULL, na) + 3) >> 2;
            uint32_t y[4] = {0u, 0u, 0u, 0u}, q4[4];
#pragma unroll
            for (int q = 0; q < 4; q++)
                if (q < (int)g3) y[q] = __byte_perm(ts.s3, S3, tl<OPT>(tab, d3 >> (8 * q), lane));
#pragma unroll
            for (int q = 0; q < 4; q++) q4[q] = group_t<OPT>(da >> (8 * q), ts.s2, y, tab, lane);
            ((uint4*)s2)[c] = make_uint4(q4[0], q4[1], q4[2], q4[3]);
            carry += tot;
        }
        T2 = carry;
    }
    uint32_t r0 = (2 * T2 + 7) >> 3, nl = third((uint32_t)(st.end - st.blk) - r0 - 4 * words);
    __syncwarp();
    if constexpr (OPT & 8) {
        // one entry a lane, from the window (every entry within it: the shuffles are the whole warp's), a loop only past 32 entries
        if ((OPT & (128 | 256 | 4096)) || (uint32_t)(st.blk & 3) + r0 + 97 <= 124) {
            uint32_t x = win_bits(st.win, o + 8 * (r0 + 3 * (uint32_t)lane));
            if ((uint32_t)lane < nl) ((uint8_t*)s2)[x & 0x3FF] = (uint8_t)(x >> 10);
            if (!(OPT & (128 | 256 | 4096)) && __builtin_expect(nl > 32, 0))
                for (uint32_t i = lane + 32; i < nl; i += 32) {
                    uint32_t y = load4(blocks + st.blk + r0 + 3 * i);
                    ((uint8_t*)s2)[y & 0x3FF] = (uint8_t)(y >> 10);
                }
        } else {
            for (uint32_t i = lane; i < nl; i += 32) {
                uint32_t x = load4(blocks + st.blk + r0 + 3 * i);
                ((uint8_t*)s2)[x & 0x3FF] = (uint8_t)(x >> 10);
            }
        }
    } else {
        for (uint32_t i = lane; i < nl; i += 32) {
            uint32_t x = load4(blocks + st.blk + r0 + 3 * i);
            ((uint8_t*)s2)[x & 0x3FF] = (uint8_t)(x >> 10);
        }
    }
    __syncwarp();
    if constexpr (OPT & 2) {
        bool window = !__any_sync(FULL, n1 > 16);
        if (window) {
            uint32_t w0 = b1 >> 2, sh = 8 * (b1 & 3);
            uint32_t r[5];
#pragma unroll
            for (int i = 0; i < 5; i++) r[i] = s2[w0 + i];
            uint32_t S[4];
#pragma unroll
            for (int i = 0; i < 4; i++) S[i] = __funnelshift_r(r[i], r[i + 1], sh);
#pragma unroll
            for (int q = 0; q < 8; q++) {
                uint32_t e = tl<OPT>(tab, st.pd[q >> 2] >> (8 * (q & 3)), lane);
                ew[q] = __byte_perm(ts.s1, S[0], e);
                if (q < 7) advance(S, e >> 16);
            }
            __syncwarp();
            return true;
        }
    }
    uint32_t c = b1;
#pragma unroll
    for (int q = 0; q < 8; q++) {
        uint32_t e = (OPT & 32) ? e1[q] : tl<OPT>(tab, st.pd[q >> 2] >> (8 * (q & 3)), lane);
        ew[q] = __byte_perm(ts.s1, __funnelshift_r(s2[c >> 2], s2[(c >> 2) + 1], c << 3), e);
        c += e >> 16;
    }
    __syncwarp();
    return true;
}




// A step with a flag: bit 31 of its block_base entry says the hot decode cannot take it (pack_v2.py cold_flags); its own block start and end
// are the entries' other bits.
struct St2 : Step {
    uint32_t cold;
};

__device__ __forceinline__ void load_step_flagged(St2& st, const uint8_t* __restrict__ data, const uint8_t* __restrict__ blocks, const int32_t* __restrict__ block_base, int64_t step, int lane) {
    const uint32_t* cp = (const uint32_t*)(data + step * STEP_BYTES) + lane;
    st.pd[0] = __ldg(cp);
    st.pd[1] = __ldg(cp + 32);
    const uint4* sp = (const uint4*)(data + step * STEP_BYTES + 256) + lane;
    uint4 x0 = __ldg(sp), x1 = __ldg(sp + 32);
    st.sw[0] = x0.x; st.sw[1] = x0.y; st.sw[2] = x0.z; st.sw[3] = x0.w;
    st.sw[4] = x1.x; st.sw[5] = x1.y; st.sw[6] = x1.z; st.sw[7] = x1.w;
    uint32_t rb = (uint32_t)__ldg(block_base + step), re = (uint32_t)__ldg(block_base + step + 1);
    st.cold = rb >> 31;
    st.blk = rb & 0x7FFFFFFFu;
    st.end = re & 0x7FFFFFFFu;
    const uint32_t* bw = (const uint32_t*)blocks;
    st.win = __ldg(bw + (st.blk >> 2) + lane);
    st.tail = __ldg(bw + (st.end >> 2) - 1 - lane);
    st.tail1 = __ldg(bw + (st.end >> 2));
}

// The general decode of one step (every path of exponents2o<0>), out of line, its 8 words a lane left in eout[q * 32 + lane]. Called where the hot
// path (OPT & 256) finds its assumptions broken: the step by value, so the hot path's own step stays in registers.
template <int TB>
__device__ __noinline__ void exponents_cold(Step st, Tiers ts, const uint8_t* __restrict__ blocks, int lane, uint32_t* s2, const uint32_t* tab, uint32_t* eout) {
    uint32_t ew[8];
    exponents2o<TB>(st, ts, blocks, lane, s2, tab, ew);
    __syncwarp();  // (the scratch is read: eout may be it)
#pragma unroll
    for (int q = 0; q < 8; q++) eout[q * 32 + lane] = ew[q];
    __syncwarp();
}

// Two steps at once (the fast path of exponents2o<OPT | 1 | 8>, a phase at a time for both): their chains are independent, so
// a scheduler can run them side by side. False (nothing written but a count) where either step needs the general path.
template <bool FAST, int OPT = 0>
__device__ __forceinline__ bool exponents_pair(const Step& A, const Step& B, Tiers ts, const uint8_t* __restrict__ blocks, int lane, uint32_t* sA, uint32_t* sB, const uint32_t* tab, uint32_t ewA[8], uint32_t ewB[8]) {
    const uint32_t S3 = 0x01010101u * (ts.s3 >> 24);
    uint32_t n1[2], T1[2], words[2];
    n1[0] = __popc(escapes(A.pd[0])) + __popc(escapes(A.pd[1]));
    n1[1] = __popc(escapes(B.pd[0])) + __popc(escapes(B.pd[1]));
    T1[0] = __reduce_add_sync(FULL, n1[0]);
    T1[1] = __reduce_add_sync(FULL, n1[1]);
    if (!FAST && (T1[0] > 512 || T1[1] > 512)) return false;
    words[0] = (T1[0] + 15) >> 4;
    words[1] = (T1[1] + 15) >> 4;
    uint32_t up[2] = {__shfl_up_sync(FULL, A.tail, 1), __shfl_up_sync(FULL, B.tail, 1)};
    uint32_t da[2], na[2], v[2], pre[2], tot[2];
    da[0] = (uint32_t)lane < words[0] ? __funnelshift_r(A.tail, lane ? up[0] : A.tail1, 8 * (uint32_t)(A.end & 3)) : 0u;
    da[1] = (uint32_t)lane < words[1] ? __funnelshift_r(B.tail, lane ? up[1] : B.tail1, 8 * (uint32_t)(B.end & 3)) : 0u;
#pragma unroll
    for (int k = 0; k < 2; k++) {
        na[k] = __popc(escapes(da[k]));
        v[k] = n1[k] | na[k] << 11;
        pre[k] = v[k];
    }
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        uint32_t t0, t1;
        asm volatile("{\n.reg .pred p;\nshfl.sync.up.b32 %0|p, %2, %4, 0, 0xffffffff;\n@p add.u32 %2, %2, %0;\nshfl.sync.up.b32 %1|p, %3, %4, 0, 0xffffffff;\n@p add.u32 %3, %3, %1;\n}\n" : "=r"(t0), "=r"(t1), "+r"(pre[0]), "+r"(pre[1]) : "r"(o));
    }
    tot[0] = __shfl_sync(FULL, pre[0], 31);
    tot[1] = __shfl_sync(FULL, pre[1], 31);
    pre[0] -= v[0];
    pre[1] -= v[1];
    const Step* st[2] = {&A, &B};
    uint32_t b1[2], T2[2], d3[2], g3[2];
    bool ok = true;
#pragma unroll
    for (int k = 0; k < 2; k++) {
        b1[k] = pre[k] & 0x7FF;
        T2[k] = tot[k] >> 11;
        uint32_t o = 8 * (uint32_t)(st[k]->blk & 3), pa = o + 2 * ((pre[k] >> 11) & 0x3FF);
        ok = ok && (o + 2 * T2[k] + 32 < 992);
        d3[k] = win_bits(st[k]->win, pa);
        g3[k] = (__reduce_max_sync(FULL, na[k]) + 3) >> 2;
    }
    // (the entries: all within the window, no more than 32)
    uint32_t r0[2], nl[2];
#pragma unroll
    for (int k = 0; k < 2; k++) {
        r0[k] = (2 * T2[k] + 7) >> 3;
        nl[k] = third((uint32_t)(st[k]->end - st[k]->blk) - r0[k] - 4 * words[k]);
        ok = ok && ((uint32_t)(st[k]->blk & 3) + r0[k] + 97 <= 124) && nl[k] <= 32;
    }
    if (!FAST && !ok) return false;
    uint32_t* s2[2] = {sA, sB};
#pragma unroll
    for (int k = 0; k < 2; k++) {
        uint32_t y[4] = {0u, 0u, 0u, 0u}, q4[4];
#pragma unroll
        for (int q = 0; q < 4; q++)
            if (q < (int)g3[k]) y[q] = __byte_perm(ts.s3, S3, tl<OPT>(tab, d3[k] >> (8 * q), lane));
#pragma unroll
        for (int q = 0; q < 4; q++) q4[q] = group_t<OPT>(da[k] >> (8 * q), ts.s2, y, tab, lane);
        ((uint4*)s2[k])[lane] = make_uint4(q4[0], q4[1], q4[2], q4[3]);
    }
    __syncwarp();
    uint32_t xe[2];
#pragma unroll
    for (int k = 0; k < 2; k++) xe[k] = win_bits(st[k]->win, 8 * (uint32_t)(st[k]->blk & 3) + 8 * (r0[k] + 3 * (uint32_t)lane));
#pragma unroll
    for (int k = 0; k < 2; k++)
        if ((uint32_t)lane < nl[k]) ((uint8_t*)s2[k])[xe[k] & 0x3FF] = (uint8_t)(xe[k] >> 10);
    __syncwarp();
    uint32_t c[2] = {b1[0], b1[1]};
    uint32_t* ew[2] = {ewA, ewB};
#pragma unroll
    for (int q = 0; q < 8; q++) {
#pragma unroll
        for (int k = 0; k < 2; k++) {
            uint32_t e = tl<OPT>(tab, st[k]->pd[q >> 2] >> (8 * (q & 3)), lane);
            ew[k][q] = __byte_perm(ts.s1, __funnelshift_r(s2[k][c[k] >> 2], s2[k][(c[k] >> 2) + 1], c[k] << 3), e);
            c[k] += e >> 16;
        }
    }
    __syncwarp();
    return true;
}

template <int OPT>
struct Tiered2o {
    const uint8_t* data;
    const uint8_t* blocks;
    const int32_t* block_base;
    Tiers ts;
    typedef typename std::conditional<(OPT & 512) != 0, St2, Step>::type St;
    static constexpr bool kTable = true;
    static constexpr int kOpt = OPT;
    static constexpr int kTB = OPT & (4 | 16384 | 32768);  // the table's layout bits
    __device__ __forceinline__ void load(St& st, int64_t step, int lane) const {
        if constexpr (OPT & 512) load_step_flagged(st, data, blocks, block_base, step, lane);
        else load_step(st, data, blocks, block_base, step, lane);
    }
    static constexpr int kScratch = (OPT & 16) ? 2 : 1;  // scratch buffers a warp
    __device__ __forceinline__ void decode(const St& st, int lane, uint32_t* s2, const uint32_t* tab, uint32_t R[16]) const {
        uint32_t ew[8];
        bool bail = false;
        if constexpr (OPT & 512) bail = st.cold != 0;  // the flag (uniform): the general decode
        if constexpr (OPT & 512) {
            if (!bail) exponents2o<OPT | 128>(st, ts, blocks, lane, s2, tab, ew);
        } else {
            bail = !exponents2o<OPT>(st, ts, blocks, lane, s2, tab, ew);  // (only OPT & 256 says no)
        }
        if (bail) {
            exponents_cold<kTB>(st, ts, blocks, lane, s2, tab, s2);  // (its result over the scratch it is done with)
#pragma unroll
            for (int q = 0; q < 8; q++) ew[q] = s2[q * 32 + lane];
            __syncwarp();
        }
        pairs(st.sw, ew, R);
    }
    // two steps (OPT & 16: the scratch holds two buffers)
    __device__ __forceinline__ void decode_pair(const St& a, const St& b, int lane, uint32_t* s2, const uint32_t* tab, uint32_t Ra[16], uint32_t Rb[16]) const {
        uint32_t ea[8], eb[8];
        bool hot = true;
        if constexpr (OPT & 512) hot = (a.cold | b.cold) == 0;
        if (hot && exponents_pair<(OPT & (128 | 512)) != 0, OPT>(a, b, ts, blocks, lane, s2, s2 + S2_BYTES / 4, tab, ea, eb)) {
            pairs(a.sw, ea, Ra);
            pairs(b.sw, eb, Rb);
        } else {
            decode(a, lane, s2, tab, Ra);
            decode(b, lane, s2, tab, Rb);
        }
    }
};

// ---------------------------------------------------------------------------------------------------------------------
// mma_gemm_kernel (MT = 1..4) with the options: the table in dynamic shared memory after the scratch (256 words, or 256 x 16),
// UNR the steps a pass.

template <class Fmt, int MT, int UNR, bool SW, class AccT>
__device__ __forceinline__ void steps2(const Fmt& f, int64_t base, int64_t s0, int64_t s1, const __nv_bfloat16* const (&xr)[MT][2], int lane, uint32_t* s2, const uint32_t* tab, AccT& acc) {
    auto loadx = [&](int64_t s, uint32_t (&a)[MT][4]) {
#pragma unroll
        for (int mt = 0; mt < MT; mt++) {
            a[mt][0] = xr[mt][0] ? *(const uint32_t*)(xr[mt][0] + s * 16) : 0u;
            a[mt][2] = xr[mt][0] ? *(const uint32_t*)(xr[mt][0] + s * 16 + 8) : 0u;
            a[mt][1] = xr[mt][1] ? *(const uint32_t*)(xr[mt][1] + s * 16) : 0u;
            a[mt][3] = xr[mt][1] ? *(const uint32_t*)(xr[mt][1] + s * 16 + 8) : 0u;
        }
    };
    auto mma = [&](const uint32_t (&a)[MT][4], const uint32_t (&R)[16]) {
        if constexpr (SW) {  // M <= 8: W as the A operand (16 rows a MMA, 4 a step), X as B (8 tokens): 16 accumulators, not 32
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const uint32_t A4[4] = {R[4 * i], R[4 * i + 2], R[4 * i + 1], R[4 * i + 3]};
                mma16816(acc[i], A4, a[0][0], a[0][2]);
            }
        } else {
#pragma unroll
            for (int mt = 0; mt < MT; mt++)
#pragma unroll
                for (int nn = 0; nn < 8; nn++) mma16816(acc[mt][nn], a[mt], R[2 * nn], R[2 * nn + 1]);
        }
    };
    if constexpr (UNR == 1) {
        typename Fmt::St st;
        uint32_t a[MT][4];
        if (s0 < s1) { f.load(st, base + s0, lane); loadx(s0, a); }
        for (int64_t s = s0; s < s1; s++) {
            typename Fmt::St cur = st;
            uint32_t ca[MT][4];
#pragma unroll
            for (int mt = 0; mt < MT; mt++)
#pragma unroll
                for (int q = 0; q < 4; q++) ca[mt][q] = a[mt][q];
            if (s + 1 < s1) { f.load(st, base + s + 1, lane); loadx(s + 1, a); }
            uint32_t R[16];
            f.decode(cur, lane, s2, tab, R);
            mma(ca, R);
        }
    } else if constexpr (UNR == 3) {
        typename Fmt::St sa, sb;
        uint32_t aa[MT][4], ab[MT][4];
        if (s0 < s1) { f.load(sa, base + s0, lane); loadx(s0, aa); }
        if (s0 + 1 < s1) { f.load(sb, base + s0 + 1, lane); loadx(s0 + 1, ab); }
        for (int64_t s = s0; s < s1; s += 2) {
            uint32_t Ra[16];
            if (s + 1 < s1) {
                uint32_t Rb[16];
                f.decode_pair(sa, sb, lane, s2, tab, Ra, Rb);
                mma(aa, Ra);
                mma(ab, Rb);
            } else {
                f.decode(sa, lane, s2, tab, Ra);
                mma(aa, Ra);
            }
            if (s + 2 < s1) { f.load(sa, base + s + 2, lane); loadx(s + 2, aa); }
            if (s + 3 < s1) { f.load(sb, base + s + 3, lane); loadx(s + 3, ab); }
        }
    } else {
        typename Fmt::St sa, sb;
        uint32_t aa[MT][4], ab[MT][4];
        if (s0 < s1) { f.load(sa, base + s0, lane); loadx(s0, aa); }
        if (s0 + 1 < s1) { f.load(sb, base + s0 + 1, lane); loadx(s0 + 1, ab); }
        for (int64_t s = s0; s < s1; s += 2) {
            {
                uint32_t R[16];
                f.decode(sa, lane, s2, tab, R);
                mma(aa, R);
                if (s + 2 < s1) { f.load(sa, base + s + 2, lane); loadx(s + 2, aa); }
            }
            if (s + 1 < s1) {
                uint32_t R[16];
                f.decode(sb, lane, s2, tab, R);
                mma(ab, R);
                if (s + 3 < s1) { f.load(sb, base + s + 3, lane); loadx(s + 3, ab); }
            }
        }
    }
}


// 4 warps: each puts its sums in a slab of red, [4][16 MT][65]; the caller adds the four.
template <int MT>
__device__ __forceinline__ void warp_sums4(float (&acc)[MT][8][4], float* red, int warp, int g, int t) {
    float* mine = red + (int64_t)warp * MT * 16 * 65;
#pragma unroll
    for (int mt = 0; mt < MT; mt++)
#pragma unroll
        for (int nn = 0; nn < 8; nn++) {
            float* r0 = mine + (mt * 16 + g) * 65 + nn * 8 + t * 2;
            float* r1 = r0 + 8 * 65;
            r0[0] = acc[mt][nn][0];
            r0[1] = acc[mt][nn][1];
            r1[0] = acc[mt][nn][2];
            r1[1] = acc[mt][nn][3];
        }
    __syncthreads();
}


// SW: each warp puts its sums in a slab of red, [NW][8 tokens][65]; the caller adds the NW slabs.
template <int NW>
__device__ __forceinline__ void warp_sums_sw(float (&acc)[4][4], float* red, int warp, int g, int t) {
    float* mine = red + (int64_t)warp * 8 * 65;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        mine[(2 * t) * 65 + 16 * i + g] = acc[i][0];
        mine[(2 * t + 1) * 65 + 16 * i + g] = acc[i][1];
        mine[(2 * t) * 65 + 16 * i + g + 8] = acc[i][2];
        mine[(2 * t + 1) * 65 + 16 * i + g + 8] = acc[i][3];
    }
    __syncthreads();
}

template <class Fmt, int MT, int UNR, int NW = 8, int MB = 0, bool SW = false>
__global__ void __launch_bounds__(32 * NW, MB ? MB : (NW == 4 ? (MT == 1 ? (UNR == 3 ? 3 : 4) : 2) : ((MT == 1 && UNR != 3) ? 2 : 1))) step_kernel(Fmt f, int64_t O, int64_t K, int64_t M, const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ Y, float* __restrict__ parts, int* __restrict__ done) {
    extern __shared__ __align__(128) float red[];  // [4 warps][16 MT rows][65], then the warps' scratch [8][S2_BYTES], then the table
    __shared__ int last;
    constexpr int RED = SW ? NW * 8 * 65 : 4 * MT * 16 * 65;
    uint32_t* s2all = (uint32_t*)(red + RED);
    uint32_t* tab = s2all + NW * Fmt::kScratch * (S2_BYTES / 4);
    if constexpr (table_copies(Fmt::kOpt) > 1) fill_groups_n<table_copies(Fmt::kOpt)>(tab);
    else fill_groups(tab);
    __syncthreads();
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    uint32_t* s2 = s2all + warp * Fmt::kScratch * (S2_BYTES / 4);
    int64_t KS = K / 16, total = O / 64 * KS, nb = gridDim.x;
    int64_t B1 = (blockIdx.x + 1) * total / nb;
    for (int64_t seg = blockIdx.x * total / nb; seg < B1;) {
        int64_t rb = seg / KS, sb = seg - rb * KS, se = min(B1, (rb + 1) * KS) - rb * KS;
        seg = rb * KS + se;
        int64_t per = (se - sb + NW - 1) / NW;
        int64_t s0 = min(se, sb + warp * per), s1 = min(se, s0 + per);
        constexpr int ROWS = SW ? 8 : MT * 16, SLABS = SW ? NW : (NW == 8 ? 4 : NW);
        const __nv_bfloat16* xr[MT][2];
#pragma unroll
        for (int mt = 0; mt < MT; mt++)
#pragma unroll
            for (int hh = 0; hh < 2; hh++) {
                int64_t r = mt * 16 + g + 8 * hh;
                xr[mt][hh] = r < M ? X + ((Fmt::kOpt & 8192) ? 0 : r * K) + t * 2 : nullptr;  // (8192: every row the same: the loads coalesce; a timing experiment, the products are wrong)
            }
        if constexpr (SW) {
            float acc[4][4];
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int q = 0; q < 4; q++) acc[i][q] = 0.f;
            steps2<Fmt, MT, UNR, true>(f, rb * KS, s0, s1, xr, lane, s2, tab, acc);
            warp_sums_sw<NW>(acc, red, warp, g, t);
        } else {
            float acc[MT][8][4];
#pragma unroll
            for (int mt = 0; mt < MT; mt++)
#pragma unroll
                for (int nn = 0; nn < 8; nn++)
#pragma unroll
                    for (int q = 0; q < 4; q++) acc[mt][nn][q] = 0.f;
            steps2<Fmt, MT, UNR, false>(f, rb * KS, s0, s1, xr, lane, s2, tab, acc);
            if constexpr (NW == 8) warp_sums<MT>(acc, red, warp, g, t);
            else warp_sums4<MT>(acc, red, warp, g, t);
        }
        int64_t first = block_of_step(rb * KS, nb, total), fin = block_of_step(rb * KS + KS - 1, nb, total);
        for (int i = threadIdx.x; i < ROWS * 64; i += 32 * NW) {
            int r = i / 64, c = i % 64;
            if (r >= M) continue;
            float v = 0.f;
#pragma unroll
            for (int ww = 0; ww < SLABS; ww++) v += red[(ww * ROWS + r) * 65 + c];
            if (first == fin) Y[r * O + rb * 64 + c] = __float2bfloat16(v + (bias ? __bfloat162float(bias[rb * 64 + c]) : 0.f));
            else parts[((blockIdx.x + rb) * M + r) * 64 + c] = v;
        }
        if (first != fin && last_of(done + rb, fin - first + 1, last))
            for (int i = threadIdx.x; i < M * 64; i += 32 * NW) {
                int r = i / 64, c = i % 64;
                float v = 0.f;
                for (int64_t b = first; b <= fin; b++) v += __ldcg(parts + ((b + rb) * M + r) * 64 + c);
                Y[r * O + rb * 64 + c] = __float2bfloat16(v + (bias ? __bfloat162float(bias[rb * 64 + c]) : 0.f));
            }
        __syncthreads();
    }
}

template <class Fmt, int MT, int UNR, int NW = 8, int MB = 0, bool SW = false>
static int step_launch(Fmt f, int64_t O, int64_t K, const uint16_t* x, int64_t M, uint16_t* y, void* ws, size_t ws_bytes, int* done, cudaStream_t cs, size_t* need) {
    auto kernel = step_kernel<Fmt, MT, UNR, NW, MB, SW>;
    static std::atomic<int> known[MAX_DEVICES];
    size_t shared = (size_t)(SW ? NW * 8 * 65 : 4 * MT * 16 * 65) * sizeof(float) + NW * Fmt::kScratch * S2_BYTES + sizeof(uint32_t) * 256 * table_copies(Fmt::kOpt);
    if (const char* pad = getenv("PAD")) shared += atoi(pad);  // (an experiment: fewer blocks an SM)
    int dev = current_device();
    int64_t RB = O / 64, total = RB * (K / 16);
    int64_t nb = std::max<int64_t>(1, std::min<int64_t>(per_sm((const void*)kernel, 32 * NW, (int)shared, known, dev) * sm_count(dev), total / 32));
    size_t bytes = (size_t)((nb + RB) * M * 64) * sizeof(float);
    if (need) { *need = bytes; return 0; }
    if (!fits(ws, ws_bytes, bytes) || !done) return cudaErrorInvalidValue;
    kernel<<<nb, 32 * NW, shared, cs>>>(f, O, K, M, bf(x), nullptr, bf(y), (float*)ws, done);
    return cudaGetLastError();
}

template <class Fmt, int UNR, int NW = 8, int MB = 0, bool SW = false>
static int step_run(Fmt f, int64_t O, int64_t K, const uint16_t* x, int64_t M, uint16_t* y, void* ws, size_t ws_bytes, int* done, cudaStream_t cs, size_t* need) {
    if (O % 64 || K % 16 || M < 0 || M > 64) return cudaErrorInvalidValue;
    if constexpr (SW) {
        if (M > 8) return cudaErrorInvalidValue;
        return step_launch<Fmt, 1, UNR, NW, MB, true>(f, O, K, x, M, y, ws, ws_bytes, done, cs, need);
    }
    auto run = M <= 16 ? step_launch<Fmt, 1, UNR, NW, MB> : M <= 32 ? step_launch<Fmt, 2, UNR, NW, MB> : step_launch<Fmt, 4, UNR, NW, MB>;
    return run(f, O, K, x, M, y, ws, ws_bytes, done, cs, need);
}
