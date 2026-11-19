// format-33's decode check: the tiered layout as built (the library's Tiered policy, unchanged), its four-tier variant T4 (tier 3
// with an escape to a dense tier 4 of 2-bit (t4a) or 3-bit (t4b) digits; written here on the library's decode, the tier-3 stage of
// the layout of v0.27 and before (tier23_v1) over a tier-4 stream decoded the way the library decodes its dense tier 3), and a
// static rANS of the exponent (32 interleaved lane states over a strip of 16 steps). Packs from f33_pack.py.
//   nvcc -O3 -std=c++20 --expt-relaxed-constexpr -gencode arch=compute_89,code=sm_89 -I../../gpu f33_dec.cu -o f33dec
//   ./f33dec PACK_DIR      every pack unpacked bit for bit against its weights; decode-only and unpack times (cudaEvent, median)
// Kernels k_dec_* (decode only, R xor-folded) are the ones whose SASS is counted.
#include "glyd_gpu.cu"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

struct Tiers4 {
    uint32_t s1, s2, s3, s4, s5;
};

// Tier 4's symbols for group q of this lane's digits (4 digits a group): W4 = 2, the library's dense-tier trick (digit 3 is
// the table's escape, which here picks the 4th symbol from S4); W4 = 3, the 3-bit digits spread to PRMT nibbles over s4:s5.
template <int W4>
__device__ __forceinline__ uint32_t tier4(uint64_t d4, int q, const Tiers4& ts, uint32_t S4, const uint32_t* tab) {
    if constexpr (W4 == 2) {
        return __byte_perm(ts.s4, S4, tab[(uint32_t)(d4 >> (8 * q)) & 0xFF]);
    } else {
        uint32_t v = (uint32_t)(d4 >> (12 * q)) & 0xFFF;
        uint32_t sel = (v & 7) | ((v & 0x38) << 1) | ((v & 0x1C0) << 2) | ((v & 0xE00) << 3);
        return __byte_perm(ts.s4, ts.s5, sel);
    }
}

// A word of 16 tier-2 digits (da) with na escapes, its escapes' tier-3 digits (d3) and their escapes' tier-4 digits (d4): 16
// exponent bytes. g3, g4: the warp's most tier-3 / tier-4 digits a lane, in fours.
template <int W4>
__device__ __forceinline__ uint4 tier234(uint32_t da, uint32_t d3, uint64_t d4, uint32_t g3, uint32_t g4, const Tiers4& ts, uint32_t S4, const uint32_t* tab) {
    uint32_t x[4] = {0u, 0u, 0u, 0u}, y[4] = {0u, 0u, 0u, 0u}, o[4];
#pragma unroll
    for (int q = 0; q < 4; q++)
        if (q < (int)g4) x[q] = tier4<W4>(d4, q, ts, S4, tab);
#pragma unroll
    for (int q = 0; q < 4; q++)
        if (q < (int)g3) y[q] = group(d3 >> (8 * q), ts.s3, x, tab);
#pragma unroll
    for (int q = 0; q < 4; q++) o[q] = group(da >> (8 * q), ts.s2, y, tab);
    return make_uint4(o[0], o[1], o[2], o[3]);
}

__device__ __forceinline__ uint64_t bits64(bool near, uint32_t win, const uint8_t* __restrict__ blocks, int64_t blk, uint32_t pos, bool two) {
    uint32_t a = near ? win_bits(win, pos) : mem_bits(blocks, blk, pos);
    uint32_t b = two ? (near ? win_bits(win, pos + 32) : mem_bits(blocks, blk, pos + 32)) : 0u;
    return a | (uint64_t)b << 32;
}

// The library's exponents() with tier 3 escaping to tier 4. HOT (for the instruction count only, never run): the hot path alone
// (at most 512 tier-1 escapes, the block's digits and list in its first 128 bytes), as the packer's flag would route a step.
template <int W4, bool HOT = false>
__device__ __forceinline__ void exponents4(const Step& st, const Tiers4& ts, const uint8_t* __restrict__ blocks, int lane, uint32_t* s2, const uint32_t* tab, uint32_t ew[8]) {
    uint32_t n1 = __popc(escapes(st.pd[0])) + __popc(escapes(st.pd[1]));
    uint32_t T1 = __reduce_add_sync(FULL, n1), words = (T1 + 15) >> 4;
    uint32_t e1[8];
#pragma unroll
    for (int q = 0; q < 8; q++) e1[q] = tab[(st.pd[q >> 2] >> (8 * (q & 3))) & 0xFF];
    uint32_t b1, T2, T3;
    const uint32_t S4 = 0x01010101u * (ts.s4 >> 24);
    const uint32_t o = 8 * (uint32_t)(st.blk & 3);
    if (HOT || T1 <= 512) {
        uint32_t up = __shfl_up_sync(FULL, st.tail, 1);
        uint32_t da = (uint32_t)lane < words ? __funnelshift_r(st.tail, lane ? up : st.tail1, 8 * (uint32_t)(st.end & 3)) : 0u;
        uint32_t na = __popc(escapes(da)), tot;
        uint32_t pre = scan(n1 | na << 11, lane, tot);
        b1 = pre & 0x7FF;
        T2 = tot >> 11;
        uint32_t pa = o + 2 * (pre >> 11);
        bool near = HOT || o + 2 * T2 + 32 < 992;
        uint32_t d3 = near ? win_bits(st.win, pa) : mem_bits(blocks, st.blk, pa);
        uint32_t n3 = __popc(escapes(d3) & first_digits(na));
        uint32_t pre3 = scan(n3, lane, T3);
        uint32_t r0 = (2 * T2 + 7) >> 3, p4 = o + 8 * r0 + W4 * pre3;
        bool near4 = HOT || o + 8 * r0 + W4 * T3 + (W4 == 3 ? 32 : 0) < 992;
        uint64_t d4 = bits64(near4, st.win, blocks, st.blk, p4, W4 == 3);
        uint32_t g3 = (__reduce_max_sync(FULL, na) + 3) >> 2, g4 = (__reduce_max_sync(FULL, n3) + 3) >> 2;
        ((uint4*)s2)[lane] = tier234<W4>(da, d3, d4, g3, g4, ts, S4, tab);
    } else {
        uint32_t tot1;
        b1 = scan(n1, lane, tot1);
        // tier 4 starts after all the tier-3 digits: their count first
        uint32_t t2 = 0;
        for (uint32_t ch = 0; ch * 32 < words; ch++) {
            uint32_t c = ch * 32 + lane;
            t2 += __reduce_add_sync(FULL, __popc(escapes(c < words ? load4(blocks + st.end - 4 * (c + 1)) : 0u)));
        }
        T2 = t2;
        uint32_t r0 = (2 * T2 + 7) >> 3, carry = 0, carry3 = 0;
        for (uint32_t ch = 0; ch * 32 < words; ch++) {
            uint32_t c = ch * 32 + lane;
            uint32_t da = c < words ? load4(blocks + st.end - 4 * (c + 1)) : 0u;
            uint32_t na = __popc(escapes(da)), tot, t3;
            uint32_t pre = scan(na, lane, tot);
            uint32_t d3 = mem_bits(blocks, st.blk, o + 2 * (carry + pre));
            uint32_t n3 = __popc(escapes(d3) & first_digits(na));
            uint32_t pre3 = scan(n3, lane, t3);
            uint64_t d4 = bits64(false, 0u, blocks, st.blk, o + 8 * r0 + W4 * (carry3 + pre3), W4 == 3);
            uint32_t g3 = (__reduce_max_sync(FULL, na) + 3) >> 2, g4 = (__reduce_max_sync(FULL, n3) + 3) >> 2;
            ((uint4*)s2)[c] = tier234<W4>(da, d3, d4, g3, g4, ts, S4, tab);
            carry += tot;
            carry3 += t3;
        }
        T3 = carry3;
    }
    // the list (from byte r1 of the block), then tier 1: as the library's
    uint32_t r1 = ((2 * T2 + 7) >> 3) + ((W4 * T3 + 7) >> 3), nl = list_entries((uint32_t)(st.end - st.blk) - r1 - 4 * words);
    __syncwarp();
    if (HOT || (nl != 0 && nl <= 32 && o + 8 * r1 + 18 * (nl - 1) < 992)) {
        uint32_t x = win_bits(st.win, o + 8 * r1 + 18 * (uint32_t)lane);
        if ((uint32_t)lane < nl) ((uint8_t*)s2)[x & 0x3FF] = (uint8_t)(x >> 10);
    } else {
        for (uint32_t i = lane; i < nl; i += 32) {
            uint32_t x = mem_bits(blocks, st.blk, o + 8 * r1 + 18 * i);
            ((uint8_t*)s2)[x & 0x3FF] = (uint8_t)(x >> 10);
        }
    }
    __syncwarp();
    uint32_t c = b1;
#pragma unroll
    for (int q = 0; q < 8; q++) {
        uint32_t e = e1[q];
        ew[q] = __byte_perm(ts.s1, __funnelshift_r(s2[c >> 2], s2[(c >> 2) + 1], c << 3), e);
        c += e >> 16;
    }
    __syncwarp();
}

template <int W4, bool HOT = false>
struct Tiered4 {
    const uint8_t* data;
    const uint8_t* blocks;
    const int32_t* block_base;
    Tiers4 ts;
    typedef Step St;
    static constexpr bool kTable = true;
    __device__ __forceinline__ void load(St& st, int64_t step, int lane) const { load_step(st, data, blocks, block_base, step, lane); }
    __device__ __forceinline__ void decode(const St& st, int lane, uint32_t* s2, const uint32_t* tab, uint32_t R[16]) const {
        uint32_t ew[8];
        exponents4<W4, HOT>(st, ts, blocks, lane, s2, tab, ew);
        pairs(st.sw, ew, R);
    }
};


// The library's exponents() (glyd_gpu.cu, copied verbatim but for HOT): HOT, for the instruction count only, keeps its hot path alone.
template <bool HOT>
__device__ __forceinline__ void exponents_lib(const Step& st, Tiers ts, const uint8_t* __restrict__ blocks, int lane, uint32_t* s2, const uint32_t* tab, uint32_t ew[8]) {
    uint32_t n1 = __popc(escapes(st.pd[0])) + __popc(escapes(st.pd[1]));
    uint32_t T1 = __reduce_add_sync(FULL, n1), words = (T1 + 15) >> 4;
    uint32_t e1[8];  // tier 1's group lookups, issued ahead of the scans (they need the digits alone)
#pragma unroll
    for (int q = 0; q < 8; q++) e1[q] = tab[(st.pd[q >> 2] >> (8 * (q & 3))) & 0xFF];
    uint32_t b1, T2;
    const uint32_t S3 = 0x01010101u * (ts.s3 >> 24);
    const uint32_t o = 8 * (uint32_t)(st.blk & 3);  // the bit the block starts at in its first word
    if (HOT || T1 <= 512) {
        // Word c of the tier-2 digits: bytes [end - 4(c + 1), end - 4c).
        uint32_t up = __shfl_up_sync(FULL, st.tail, 1);
        uint32_t da = (uint32_t)lane < words ? __funnelshift_r(st.tail, lane ? up : st.tail1, 8 * (uint32_t)(st.end & 3)) : 0u;
        uint32_t na = __popc(escapes(da)), tot;
        uint32_t pre = scan(n1 | na << 11, lane, tot);
        b1 = pre & 0x7FF;
        T2 = tot >> 11;
        // Their tier-3 digits, from bit pa of the window (the 128 bytes from the block's first word) or from memory.
        uint32_t pa = o + 2 * (pre >> 11);
        bool near = HOT || o + 2 * T2 + 32 < 992;
        uint32_t d3 = near ? win_bits(st.win, pa) : mem_bits(blocks, st.blk, pa);
        uint32_t g3 = (__reduce_max_sync(FULL, na) + 3) >> 2;
        uint32_t y[4] = {0u, 0u, 0u, 0u}, q4[4];
#pragma unroll
        for (int q = 0; q < 4; q++)
            if (q < (int)g3) y[q] = __byte_perm(ts.s3, S3, tab[(d3 >> (8 * q)) & 0xFF]);
#pragma unroll
        for (int q = 0; q < 4; q++) q4[q] = group(da >> (8 * q), ts.s2, y, tab);
        ((uint4*)s2)[lane] = make_uint4(q4[0], q4[1], q4[2], q4[3]);
    } else {
        // More than 32 words: chunks of 32, each its words' tier-3 digits after the chunks before it.
        uint32_t tot1;
        b1 = scan(n1, lane, tot1);
        uint32_t carry = 0;
        for (uint32_t ch = 0; ch * 32 < words; ch++) {
            uint32_t c = ch * 32 + lane;
            uint32_t da = c < words ? load4(blocks + st.end - 4 * (c + 1)) : 0u;
            uint32_t na = __popc(escapes(da)), tot;
            uint32_t pre = scan(na, lane, tot);
            uint32_t d3 = mem_bits(blocks, st.blk, o + 2 * (carry + pre));  // (the window holds 128 bytes: tier-3 digits can run past it)
            uint32_t g3 = (__reduce_max_sync(FULL, na) + 3) >> 2;
            uint32_t y[4] = {0u, 0u, 0u, 0u}, q4[4];
#pragma unroll
            for (int q = 0; q < 4; q++)
                if (q < (int)g3) y[q] = __byte_perm(ts.s3, S3, tab[(d3 >> (8 * q)) & 0xFF]);
#pragma unroll
            for (int q = 0; q < 4; q++) q4[q] = group(da >> (8 * q), ts.s2, y, tab);
            ((uint4*)s2)[c] = make_uint4(q4[0], q4[1], q4[2], q4[3]);
            carry += tot;
        }
        T2 = carry;
    }
    // The list: entry i (lane i, i + 32, ...) of 18 bits from the list's first byte sets the byte of the escape it names.
    uint32_t r0 = (2 * T2 + 7) >> 3, nl = list_entries((uint32_t)(st.end - st.blk) - r0 - 4 * words);
    __syncwarp();
    if (HOT || (nl != 0 && nl <= 32 && o + 8 * r0 + 18 * (nl - 1) < 992)) {
        uint32_t x = win_bits(st.win, o + 8 * r0 + 18 * (uint32_t)lane);  // (every lane's: the shuffles are the warp's; a lane past the list's end has no entry)
        if ((uint32_t)lane < nl) ((uint8_t*)s2)[x & 0x3FF] = (uint8_t)(x >> 10);
    } else {
        for (uint32_t i = lane; i < nl; i += 32) {
            uint32_t x = mem_bits(blocks, st.blk, o + 8 * r0 + 18 * i);
            ((uint8_t*)s2)[x & 0x3FF] = (uint8_t)(x >> 10);
        }
    }
    __syncwarp();
    // Tier 1, its escapes' from the scratch (this lane's: bytes b1 on).
    uint32_t c = b1;
#pragma unroll
    for (int q = 0; q < 8; q++) {
        uint32_t e = e1[q];
        ew[q] = __byte_perm(ts.s1, __funnelshift_r(s2[c >> 2], s2[(c >> 2) + 1], c << 3), e);
        c += e >> 16;
    }
    __syncwarp();  // the scratch is read
}


struct TieredHot {
    const uint8_t* data;
    const uint8_t* blocks;
    const int32_t* block_base;
    Tiers ts;
    typedef Step St;
    static constexpr bool kTable = true;
    __device__ __forceinline__ void load(St& st, int64_t step, int lane) const { load_step(st, data, blocks, block_base, step, lane); }
    __device__ __forceinline__ void decode(const St& st, int lane, uint32_t* s2, const uint32_t* tab, uint32_t R[16]) const {
        uint32_t ew[8];
        exponents_lib<true>(st, ts, blocks, lane, s2, tab, ew);
        pairs(st.sw, ew, R);
    }
};

// ---------------------------------------------------------------------------------------------------------------------
// rANS: strip j (SS steps from step j * SS): lane states from states[32 j], 16-bit words from stream[off[j]]; per step each lane
// decodes its 32 exponents in order (all lanes at once), renormalizing from the warp's shared stream; the bytes as the library's.
struct Rans {
    const uint8_t* raw;
    const uint32_t* states;
    const uint16_t* stream;
    const int64_t* off;
    const uint32_t* table;
    int ss;
    int64_t steps;
};

template <class Emit>
__device__ __forceinline__ void rans_strip(const Rans& P, int64_t strip, int lane, const uint32_t* tab, Emit emit) {
    uint32_t x = P.states[strip * 32 + lane];
    int64_t pos = P.off[strip];
    const uint32_t lt = (1u << lane) - 1u;
    int64_t s1 = min((strip + 1) * (int64_t)P.ss, P.steps);
    for (int64_t s = strip * P.ss; s < s1; s++) {
        const uint4* sp = (const uint4*)(P.raw + s * 1024) + lane;
        uint4 a = __ldg(sp), b = __ldg(sp + 32);
        uint32_t sw[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w}, ew[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
#pragma unroll
        for (int i = 0; i < 32; i++) {
            uint32_t slot = x & 4095u, t = tab[slot];
            x = ((t >> 8 & 0xFFFu) + 1u) * (x >> 12) + slot - (t >> 20);
            bool need = x < 65536u;
            uint32_t m = __ballot_sync(FULL, need);
            if (need) x = x << 16 | __ldg(P.stream + pos + __popc(m & lt));
            pos += __popc(m);
            ew[i >> 2] |= (t & 0xFFu) << (8 * (i & 3));
        }
        uint32_t R[16];
        pairs(sw, ew, R);
        emit(s, R);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
template <class Fmt>
__device__ __forceinline__ void unpack_body(const Fmt& f, int64_t K, int64_t rows, uint16_t* out) {
    __shared__ uint32_t tab[256];
    fill_groups(tab);
    __syncthreads();
    __shared__ uint4 scratch[4][S2_BYTES / 16];
    int64_t KS = K / 16, total = rows / 64 * KS;
    int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    uint32_t* out32 = (uint32_t*)out;
    for (int64_t step = (blockIdx.x * (int64_t)blockDim.x + threadIdx.x) >> 5; step < total; step += (int64_t)gridDim.x * blockDim.x >> 5) {
        typename Fmt::St st;
        uint32_t R[16];
        f.load(st, step, lane);
        f.decode(st, lane, (uint32_t*)scratch[threadIdx.x >> 5], tab, R);
        int64_t rb = step / KS, s = step % KS;
#pragma unroll
        for (int n = 0; n < 8; n++) {
            int64_t at2 = ((rb * 64 + 8 * n + g) * K + s * 16 + 2 * t) / 2;
            out32[at2] = R[2 * n];
            out32[at2 + 4] = R[2 * n + 1];
        }
    }
}

template <class Fmt>
__device__ __forceinline__ void dec_body(const Fmt& f, int64_t total, uint32_t* acc_out) {
    __shared__ uint32_t tab[256];
    fill_groups(tab);
    __syncthreads();
    __shared__ uint4 scratch[4][S2_BYTES / 16];
    int lane = threadIdx.x & 31;
    uint32_t acc = 0;
    for (int64_t step = (blockIdx.x * (int64_t)blockDim.x + threadIdx.x) >> 5; step < total; step += (int64_t)gridDim.x * blockDim.x >> 5) {
        typename Fmt::St st;
        uint32_t R[16];
        f.load(st, step, lane);
        f.decode(st, lane, (uint32_t*)scratch[threadIdx.x >> 5], tab, R);
#pragma unroll
        for (int k = 0; k < 16; k++) acc ^= R[k];
    }
    acc_out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

extern "C" __global__ void __launch_bounds__(128) k_unp_tiered(Tiered f, int64_t K, int64_t rows, uint16_t* out) { unpack_body(f, K, rows, out); }
extern "C" __global__ void __launch_bounds__(128) k_unp_t4a(Tiered4<2> f, int64_t K, int64_t rows, uint16_t* out) { unpack_body(f, K, rows, out); }
extern "C" __global__ void __launch_bounds__(128) k_unp_t4b(Tiered4<3> f, int64_t K, int64_t rows, uint16_t* out) { unpack_body(f, K, rows, out); }
extern "C" __global__ void __launch_bounds__(128) k_dec_tiered(Tiered f, int64_t total, uint32_t* acc) { dec_body(f, total, acc); }
extern "C" __global__ void __launch_bounds__(128) k_dec_t4a(Tiered4<2> f, int64_t total, uint32_t* acc) { dec_body(f, total, acc); }
extern "C" __global__ void __launch_bounds__(128) k_dec_t4b(Tiered4<3> f, int64_t total, uint32_t* acc) { dec_body(f, total, acc); }
// the hot paths alone (compiled for their SASS, never launched)
extern "C" __global__ void __launch_bounds__(128) k_hot_tiered(TieredHot f, int64_t total, uint32_t* acc) { dec_body(f, total, acc); }
extern "C" __global__ void __launch_bounds__(128) k_hot_t4a(Tiered4<2, true> f, int64_t total, uint32_t* acc) { dec_body(f, total, acc); }
extern "C" __global__ void __launch_bounds__(128) k_hot_t4b(Tiered4<3, true> f, int64_t total, uint32_t* acc) { dec_body(f, total, acc); }

__device__ __forceinline__ void rans_tab(const Rans& P, uint32_t* tab) {
    for (int i = threadIdx.x; i < 4096; i += blockDim.x) tab[i] = P.table[i];
    __syncthreads();
}
extern "C" __global__ void __launch_bounds__(128) k_unp_rans(Rans P, int64_t K, uint16_t* out) {
    __shared__ uint32_t tab[4096];
    rans_tab(P, tab);
    int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    int64_t KS = K / 16, strips = (P.steps + P.ss - 1) / P.ss;
    uint32_t* out32 = (uint32_t*)out;
    for (int64_t j = (blockIdx.x * (int64_t)blockDim.x + threadIdx.x) >> 5; j < strips; j += (int64_t)gridDim.x * blockDim.x >> 5)
        rans_strip(P, j, lane, tab, [&](int64_t step, const uint32_t* R) {
            int64_t rb = step / KS, s = step % KS;
#pragma unroll
            for (int n = 0; n < 8; n++) {
                int64_t at2 = ((rb * 64 + 8 * n + g) * K + s * 16 + 2 * t) / 2;
                out32[at2] = R[2 * n];
                out32[at2 + 4] = R[2 * n + 1];
            }
        });
}
extern "C" __global__ void __launch_bounds__(128) k_dec_rans(Rans P, uint32_t* acc_out) {
    __shared__ uint32_t tab[4096];
    rans_tab(P, tab);
    int lane = threadIdx.x & 31;
    uint32_t acc = 0;
    int64_t strips = (P.steps + P.ss - 1) / P.ss;
    for (int64_t j = (blockIdx.x * (int64_t)blockDim.x + threadIdx.x) >> 5; j < strips; j += (int64_t)gridDim.x * blockDim.x >> 5)
        rans_strip(P, j, lane, tab, [&](int64_t, const uint32_t* R) {
#pragma unroll
            for (int k = 0; k < 16; k++) acc ^= R[k];
        });
    acc_out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

// ---------------------------------------------------------------------------------------------------------------------
static std::vector<char> slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) { fprintf(stderr, "missing %s\n", p.c_str()); exit(1); }
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
template <class T>
static T* up(const std::vector<char>& v, size_t pad = 256) {
    void* d;
    CK(cudaMalloc(&d, v.size() + pad));
    CK(cudaMemset(d, 0, v.size() + pad));
    CK(cudaMemcpy(d, v.data(), v.size(), cudaMemcpyHostToDevice));
    return (T*)d;
}

static float median_ms(std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

template <class F>
static float timeit(F launch, int reps = getenv("REPS") ? atoi(getenv("REPS")) : 31) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    launch();
    CK(cudaDeviceSynchronize());
    std::vector<float> ms;
    for (int i = 0; i < reps; i++) {
        CK(cudaEventRecord(a));
        launch();
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float t;
        CK(cudaEventElapsedTime(&t, a, b));
        ms.push_back(t);
    }
    CK(cudaGetLastError());
    return median_ms(ms);
}

int main(int argc, char** argv) {
    std::string dir = argv[1];
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    int clk_khz = 0;
    CK(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, 0));
    int SMs = prop.multiProcessorCount, grid = SMs * 8, NT = 128;
    printf("%s, %d SMs, max SM clock %d MHz\n", prop.name, SMs, clk_khz / 1000);
    std::ifstream lst(dir + "/list.txt");
    std::string name;
    uint32_t* acc;
    CK(cudaMalloc(&acc, (size_t)grid * NT * 4));
    int bad = 0;
    printf("%-8s %-7s %10s %10s %12s %s\n", "tensor", "layout", "dec us", "unpack us", "dec ns/step*SM", "bit for bit");
    while (lst >> name) {
        std::string p = dir + "/" + name;
        int64_t O, K;
        { std::ifstream d(p + ".dims"); d >> O >> K; }
        auto w = slurp(p + ".w");
        int64_t steps = O * K / 1024;
        uint16_t* out;
        CK(cudaMalloc(&out, O * K * 2));
        auto check = [&](const char* fmt, auto unpack, auto dec) {
            CK(cudaMemset(out, 0xFF, O * K * 2));
            unpack();
            CK(cudaDeviceSynchronize());
            std::vector<char> h(O * K * 2);
            CK(cudaMemcpy(h.data(), out, O * K * 2, cudaMemcpyDeviceToHost));
            int64_t diff = 0;
            for (int64_t i = 0; i < O * K; i++) diff += ((uint16_t*)h.data())[i] != ((uint16_t*)w.data())[i];
            float td = timeit(dec), tu = timeit(unpack);
            printf("%-8s %-7s %10.2f %10.2f %12.2f %s\n", name.c_str(), fmt, 1e3 * td, 1e3 * tu, 1e6 * td * SMs / steps, diff ? "DIFFERS" : "yes");
            if (diff) { printf("   %lld weights differ\n", (long long)diff); bad++; }
            fflush(stdout);
        };
        auto tiers = [&](const std::string& f) { auto v = slurp(p + "." + f + ".tiers"); return std::vector<uint32_t>((uint32_t*)v.data(), (uint32_t*)(v.data() + v.size())); };
        {
            auto t = tiers("v2b");
            Tiered f{up<uint8_t>(slurp(p + ".v2b.data")), up<uint8_t>(slurp(p + ".v2b.blocks")), up<int32_t>(slurp(p + ".v2b.bb")), {t[0], t[1], t[2]}};
            check("tiered", [&] { k_unp_tiered<<<grid, NT>>>(f, K, O, out); }, [&] { k_dec_tiered<<<grid, NT>>>(f, steps, acc); });
        }
        {
            auto t = tiers("t4a");
            Tiered4<2> f{up<uint8_t>(slurp(p + ".t4a.data")), up<uint8_t>(slurp(p + ".t4a.blocks")), up<int32_t>(slurp(p + ".t4a.bb")), {t[0], t[1], t[2], t[3], t[4]}};
            check("t4a", [&] { k_unp_t4a<<<grid, NT>>>(f, K, O, out); }, [&] { k_dec_t4a<<<grid, NT>>>(f, steps, acc); });
        }
        {
            auto t = tiers("t4b");
            Tiered4<3> f{up<uint8_t>(slurp(p + ".t4b.data")), up<uint8_t>(slurp(p + ".t4b.blocks")), up<int32_t>(slurp(p + ".t4b.bb")), {t[0], t[1], t[2], t[3], t[4]}};
            check("t4b", [&] { k_unp_t4b<<<grid, NT>>>(f, K, O, out); }, [&] { k_dec_t4b<<<grid, NT>>>(f, steps, acc); });
        }
        {
            auto ss = slurp(p + ".rans.ss");
            Rans P{up<uint8_t>(slurp(p + ".rans.raw")), up<uint32_t>(slurp(p + ".rans.states")), up<uint16_t>(slurp(p + ".rans.stream")),
                   up<int64_t>(slurp(p + ".rans.off")), up<uint32_t>(slurp(p + ".rans.table")), *(int32_t*)ss.data(), steps};
            check("rans", [&] { k_unp_rans<<<grid, NT>>>(P, K, out); }, [&] { k_dec_rans<<<grid, NT>>>(P, acc); });
        }
        CK(cudaFree(out));
    }
    printf(bad ? "FAILED: %d packs differ\n" : "every pack bit for bit\n", bad);
    return bad != 0;
}
