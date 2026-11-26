// Glyd's GPU kernels' source that the GPU library (glyd_gpu.cu) and the engine (glyd-engine/kernels/fold.cu) both compile (R17, #238;
// glyd-engine/docs/design/one-kernel-build.md): one source, so the two cannot differ. Device code alone, no include: the includer has the CUDA
// headers (and, for an AMD build, hip.h) before it. The engine includes it inside a namespace of its own.
#pragma once

// A line of 128 bytes asked of L2 ahead (RDNA has no such prefetch: nothing there).
__device__ __forceinline__ void pf_l2(const void* p) {
#if defined(__HIP_PLATFORM_AMD__)
    (void)p;
#else
    asm volatile("prefetch.global.L2 [%0];\n" ::"l"(p));
#endif
}

// ---------------------------------------------------------------------------
// The mma layout: a tensor's weights in the order the tensor cores'
// mma.sync.m16n8k16 takes its B operand, so a thread's 32 weights of a step
// arrive as a few loads and are decoded in registers straight into its B
// fragments (Marlin's arrangement, for this code). For W [O, K] (O a
// multiple of 64, K of 16): warp step (rb, ks) covers rows rb*64.. and
// columns ks*16..; its lane l = 4g + t holds, as its group's weight 4n + j
// (n-tile n 0-7, j 0-3),
//   W[rb*64 + 8n + g][ks*16 + 8(j >> 1) + 2t + (j & 1)].
// A weight's exponent is coded in tiers of 2-bit digits: tier 1's digits
// 0-2 are the tensor's 3 commonest exponents, digit 3 goes on to tier 2
// (the next 3), whose digit 3 goes on to tier 3 (the next 4: all four digits
// are exponents); an exponent past the tenth is a literal. A warp step's
// fixed 1280 bytes: its lanes' tier-1 digits (weight i at bits
// 2(i mod 16) of word i / 16; [2 words][32 lanes]), then their 32 bytes each
// as two halves (every lane's first 16, then every lane's last 16): weight
// i's 7 mantissa bits above the sign of its pair's other weight (i ^ 1), so
// a pair's two bytes and two exponents, permuted into one word and rotated
// by a bit, are its two bf16s. Its block (bytes block_base[step] to
// block_base[step + 1], the entries' bit 31 aside): from its start, the
// tier-3 digits of its weights of ranks 6-9 (2 bits each, in the order of their
// tier-1 escapes), then from the next byte the list of its literals, the
// weights of ranks 10 on: an entry of 18 bits, low bits first, one after the
// other from the list's first byte, holds the place of its weight among the
// step's tier-1 escapes (10 bits) and its exponent (8 bits), the entries in
// the order of their places; at its end, the tier-2 digits of its tier-1
// escapes in lane order (a literal's: 0), as words of 16 (word c: escapes
// 16c to 16c + 15, the c-th word back from the end; unused digits 0). All
// in order: lane by lane, a lane's weights in order. Bit 31 of block_base[step]
// is the packer's flag: set where the step's block is more than a decode of
// a short form takes (more than 512 tier-1 escapes or 32 literals, its
// tier-3 digits or its last literal's entry not within the first 124 bytes
// of the 128 from the word its block starts in); the decode here takes every
// step, and masks it. The tiers' words: tiers[0]'s top byte is the layout's
// number, 2 (the layout of v0.27 and before has 0xFF, three exponents in
// tier 3 and its digit 3 an escape to the exponent's byte, which follows in
// the block: TieredV1, unpacked as before and given to no product).

constexpr int64_t STEP_BYTES = 1280;  // a warp step's fixed part
constexpr int S2_BYTES = 1040;        // a warp's shared scratch for a step's tier-2 exponents
constexpr uint32_t FULL = 0xffffffffu;

// The tiers' symbols: tier k's three exponents in bytes 0-2; tier 3's fourth in byte 3 of s3, the layout's number in byte 3 of s1.
constexpr uint32_t TIERED_TAG = 2, TIERED_V1_TAG = 0xFF;
struct Tiers {
    uint32_t s1, s2, s3;
};

// Eight 2-bit digits (bits 0-15) as the nibbles of two __byte_perm
// selectors: digits 0-3 in bits 0-15, 4-7 in bits 16-31.
__device__ __forceinline__ uint32_t nib8(uint32_t x) {
    uint32_t t = ((x & 0xFFFFu) | (x << 8)) & 0x00FF00FFu;
    t = (t | (t << 4)) & 0x0F0F0F0Fu;
    return (t | (t << 2)) & 0x33333333u;
}

// A group's selector (bits 0-15; z: its digits' nibbles, fn: their escapes,
// 1 at bit 0 of each 3; bits past 15 do not reach them): a digit 0-2 picks
// its symbol, an escape byte 4 + the escapes before it (the stream's bytes,
// in order).
__device__ __forceinline__ uint32_t selector(uint32_t z, uint32_t fn) { return z ^ ((z ^ (0x4444u + fn * 0x1110u)) & (fn * 0xFu)); }

// A stream of 16 bytes on by `bytes` (0-4).
__device__ __forceinline__ void advance(uint32_t s[4], uint32_t bytes) {
    uint32_t sh = 8 * bytes;
    s[0] = __funnelshift_rc(s[0], s[1], sh);
    s[1] = __funnelshift_rc(s[1], s[2], sh);
    s[2] = __funnelshift_rc(s[2], s[3], sh);
    s[3] = __funnelshift_rc(s[3], 0u, sh);
}

// The groups' table (256 words in shared memory): for four digits (an
// 8-bit field), their selector (bits 0-15) and escapes (bits 16-18).
__device__ __forceinline__ void fill_groups(uint32_t* tab) {
    for (int f = threadIdx.x; f < 256; f += blockDim.x) {
        uint32_t z = nib8((uint32_t)f), fn = z & (z >> 1) & 0x1111u;
        tab[f] = selector(z, fn) | (uint32_t)__popc(fn) << 16;
    }
}

// Four digits (bits 0-7 of field) as a word of bytes: digits 0-2 the
// symbols' bytes, escapes the stream's next bytes (taken from it).
__device__ __forceinline__ uint32_t group(uint32_t field, uint32_t sym, uint32_t s[4], const uint32_t* tab) {
    uint32_t e = tab[field & 0xFF], o = __byte_perm(sym, s[0], e);
    advance(s, e >> 16);
    return o;
}

__device__ __forceinline__ uint32_t escapes(uint32_t d) { return d & (d >> 1) & 0x55555555u; }
__device__ __forceinline__ uint32_t first_digits(uint32_t n) { return n >= 16 ? FULL : (1u << (2 * n)) - 1; }

// 4 bytes from any address (blocks are padded past their ends).
__device__ __forceinline__ uint32_t load4(const uint8_t* __restrict__ p) {
    uintptr_t a = (uintptr_t)p;
    const uint32_t* q = (const uint32_t*)(a & ~(uintptr_t)3);
    return __funnelshift_r(__ldg(q), __ldg(q + 1), (uint32_t)(a & 3) * 8);
}

// 32 bits of a step's block from bit `pos` of its first byte's word on:
// from the window (each lane's word of the 128 bytes from there; pos < 992,
// every lane calls it), or from memory.
__device__ __forceinline__ uint32_t win_bits(uint32_t win, uint32_t pos) {
    uint32_t k = pos >> 5;
    return __funnelshift_r(__shfl_sync(FULL, win, k), __shfl_sync(FULL, win, k + 1), pos);
}
__device__ __forceinline__ uint32_t mem_bits(const uint8_t* __restrict__ blocks, int64_t blk, uint32_t pos) {
    const uint32_t* q = (const uint32_t*)(blocks + (blk & ~(int64_t)3)) + (pos >> 5);
    return __funnelshift_r(__ldg(q), __ldg(q + 1), pos);
}

// A count across the warp: this lane's start (the counts of the lanes
// before it) and the whole.
__device__ __forceinline__ uint32_t scan(uint32_t n, int lane, uint32_t& total) {
    uint32_t before = n;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        uint32_t v = __shfl_up_sync(FULL, before, o);
        before += lane >= o ? v : 0;
    }
    total = __shfl_sync(FULL, before, 31);
    return before - n;
}

// A step's loads for a lane: tier-1 digits, bytes, and its block: where it
// starts and ends, its word from the start (win) and back from the end
// (tail; tail1: the end's word).
struct Step {
    uint32_t pd[2], sw[8], win, tail, tail1;
    int64_t blk, end;
};

// (off: added to the entries, a multiple of 4: an expert cache's place's blocks, TieredPlaced)
__device__ __forceinline__ void load_step(Step& st, const uint8_t* __restrict__ data, const uint8_t* __restrict__ blocks, const int32_t* __restrict__ block_base, int64_t step, int lane, int64_t off = 0) {
    const uint32_t* cp = (const uint32_t*)(data + step * STEP_BYTES) + lane;
    st.pd[0] = __ldg(cp);
    st.pd[1] = __ldg(cp + 32);
    const uint4* sp = (const uint4*)(data + step * STEP_BYTES + 256) + lane;
    uint4 x0 = __ldg(sp), x1 = __ldg(sp + 32);
    st.sw[0] = x0.x; st.sw[1] = x0.y; st.sw[2] = x0.z; st.sw[3] = x0.w;
    st.sw[4] = x1.x; st.sw[5] = x1.y; st.sw[6] = x1.z; st.sw[7] = x1.w;
    st.blk = ((uint32_t)__ldg(block_base + step) & 0x7FFFFFFFu) + off;  // (bit 31: the step's flag)
    st.end = ((uint32_t)__ldg(block_base + step + 1) & 0x7FFFFFFFu) + off;
    const uint32_t* bw = (const uint32_t*)blocks;
    st.win = __ldg(bw + (st.blk >> 2) + lane);
    st.tail = __ldg(bw + (st.end >> 2) - 1 - lane);
    st.tail1 = __ldg(bw + (st.end >> 2));
}

// A word of 16 tier-2 digits (d) as 16 bytes into the warp's scratch at
// word 4c: their escapes' from its tier-3 digits (d3), theirs from the bytes
// (x). g3: the warp's most tier-3 digits a word, in fours.
__device__ __forceinline__ void tier23_v1(uint32_t d, uint32_t d3, uint32_t x[4], Tiers ts, uint32_t g3, uint32_t* s2, int c, const uint32_t* tab) {
    uint32_t y[4] = {0u, 0u, 0u, 0u}, o[4];
#pragma unroll
    for (int q = 0; q < 4; q++)
        if (q < (int)g3) y[q] = group(d3 >> (8 * q), ts.s3, x, tab);
#pragma unroll
    for (int q = 0; q < 4; q++) o[q] = group(d >> (8 * q), ts.s2, y, tab);
    ((uint4*)s2)[c] = make_uint4(o[0], o[1], o[2], o[3]);
}

// A group's 32 exponents, 4 a word (ew). Every lane of the warp calls it
// for the same step: lane c decodes the tier-2 digits of escapes 16c to
// 16c + 15 (and 16(c + 32) on, past 512) with what they escape to, into
// the warp's scratch (s2, S2_BYTES), where each lane reads its escapes'.
__device__ __forceinline__ void exponents_v1(const Step& st, Tiers ts, const uint8_t* __restrict__ blocks, int lane, uint32_t* s2, const uint32_t* tab, uint32_t ew[8]) {
    uint32_t n1 = __popc(escapes(st.pd[0])) + __popc(escapes(st.pd[1]));
    uint32_t T1 = __reduce_add_sync(FULL, n1), words = (T1 + 15) >> 4;
    // Word c of the tier-2 digits: bytes [end - 4(c + 1), end - 4c).
    uint32_t up = __shfl_up_sync(FULL, st.tail, 1);
    uint32_t da = (uint32_t)lane < words ? __funnelshift_r(st.tail, lane ? up : st.tail1, 8 * (uint32_t)(st.end & 3)) : 0u, db = 0;
    if (T1 > 512 && (uint32_t)lane + 32 < words) db = load4(blocks + st.end - 4 * (33 + lane));
    uint32_t na = __popc(escapes(da)), nb = __popc(escapes(db)), tot;
    uint32_t pre = scan(n1 | na << 11 | nb << 21, lane, tot);
    uint32_t b1 = pre & 0x7FF, ta = (tot >> 11) & 0x3FF, T2 = ta + (tot >> 21);
    // Their escapes' tier-3 digits, and those's escapes (bytes), all from bit o of the window.
    uint32_t o = 8 * (uint32_t)(st.blk & 3), pa = o + 2 * ((pre >> 11) & 0x3FF), pb = o + 2 * (ta + (pre >> 21));
    bool near = o + 2 * T2 + 32 < 992;
    uint32_t d3a = near ? win_bits(st.win, pa) : mem_bits(blocks, st.blk, pa), d3b = 0;
    if (T1 > 512) d3b = near ? win_bits(st.win, pb) : mem_bits(blocks, st.blk, pb);
    uint32_t n3a = __popc(escapes(d3a) & first_digits(na)), n3b = __popc(escapes(d3b) & first_digits(nb)), t3;
    uint32_t pre3 = scan(n3a | n3b << 16, lane, t3);
    uint32_t r0 = o + 8 * ((2 * T2 + 7) >> 3), T3 = (t3 & 0xFFFF) + (t3 >> 16);
    uint32_t ba = r0 + 8 * (pre3 & 0xFFFF), bb = r0 + 8 * ((t3 & 0xFFFF) + (pre3 >> 16));
    uint32_t g3 = (__reduce_max_sync(FULL, max(na, nb)) + 3) >> 2;
    bool many = __any_sync(FULL, max(n3a, n3b) > 8);
    near = r0 + 8 * T3 + 128 < 992;
    uint32_t x[4];
    auto bytes = [&](uint32_t b) {
#pragma unroll
        for (int k = 0; k < 4; k++) x[k] = k < 2 || many ? (near ? win_bits(st.win, b + 32 * k) : mem_bits(blocks, st.blk, b + 32 * k)) : 0u;
    };
    bytes(ba);
    tier23_v1(da, d3a, x, ts, g3, s2, lane, tab);
    if (T1 > 512) {
        bytes(bb);
        tier23_v1(db, d3b, x, ts, g3, s2, lane + 32, tab);
    }
    __syncwarp();
    // Tier 1, its escapes' from the scratch (this lane's: bytes b1 on).
    uint32_t c = b1;
#pragma unroll
    for (int q = 0; q < 8; q++) {
        uint32_t e = tab[(st.pd[q >> 2] >> (8 * (q & 3))) & 0xFF];
        ew[q] = __byte_perm(ts.s1, __funnelshift_r(s2[c >> 2], s2[(c >> 2) + 1], c << 3), e);
        c += e >> 16;
    }
    __syncwarp();  // the scratch is read
}

// Pairs: [byte, exponent, byte, exponent] rotated right by one bit.
__device__ __forceinline__ void pairs(const uint32_t sw[8], const uint32_t ew[8], uint32_t R[16]) {
#pragma unroll
    for (int p = 0; p < 16; p++) {
        uint32_t y = __byte_perm(sw[p >> 1], ew[p >> 1], (p & 1) ? 0x7362 : 0x5140);
        R[p] = __funnelshift_r(y, y, 1);
    }
}

// A step's 32 weights a lane as the B fragments of its 8 n-tiles (R[2n], R[2n + 1]), in the layout of v0.27 and before.
__device__ __forceinline__ void decode_step_v1(const Step& st, Tiers ts, const uint8_t* __restrict__ blocks, int lane, uint32_t* s2, const uint32_t* tab, uint32_t R[16]) {
    uint32_t ew[8];
    exponents_v1(st, ts, blocks, lane, s2, tab, ew);
    pairs(st.sw, ew, R);
}

// The entries of a list of `bytes` bytes of 18 bits each: 8 bytes / 18, exact for the list a packer makes (bytes = 18 n / 8 up; at most 1024:
// a step has no more weights, and a block cut short does not run a lane's loop for ever).
__device__ __forceinline__ uint32_t list_entries(uint32_t bytes) { return min(__umulhi(4u * bytes, 0x1C71C71Du), 1024u); }

// A step's 32 exponents a lane, 4 a word (ew), by the general decode (any step, whatever its flag). Every lane of the warp
// calls it for the same step: lane c decodes the tier-2 digits of escapes 16c to 16c + 15 (and 16(c + 32) on, past 512) with
// what they go on to (tier 3's four exponents by a digit: the group table's escape, which takes a byte of a stream, takes
// here a byte of S3, every one tier 3's fourth), into the warp's scratch (s2, S2_BYTES); then each literal of the step's
// list sets its escape's byte there; then each lane reads its escapes' from it.
__device__ __forceinline__ void exponents(const Step& st, Tiers ts, const uint8_t* __restrict__ blocks, int lane, uint32_t* s2, const uint32_t* tab, uint32_t ew[8]) {
    uint32_t n1 = __popc(escapes(st.pd[0])) + __popc(escapes(st.pd[1]));
    uint32_t T1 = __reduce_add_sync(FULL, n1), words = (T1 + 15) >> 4;
    uint32_t e1[8];  // tier 1's group lookups, issued ahead of the scans (they need the digits alone)
#pragma unroll
    for (int q = 0; q < 8; q++) e1[q] = tab[(st.pd[q >> 2] >> (8 * (q & 3))) & 0xFF];
    uint32_t b1, T2;
    const uint32_t S3 = 0x01010101u * (ts.s3 >> 24);
    const uint32_t o = 8 * (uint32_t)(st.blk & 3);  // the bit the block starts at in its first word
    if (T1 <= 512) {
        // Word c of the tier-2 digits: bytes [end - 4(c + 1), end - 4c).
        uint32_t up = __shfl_up_sync(FULL, st.tail, 1);
        uint32_t da = (uint32_t)lane < words ? __funnelshift_r(st.tail, lane ? up : st.tail1, 8 * (uint32_t)(st.end & 3)) : 0u;
        uint32_t na = __popc(escapes(da)), tot;
        uint32_t pre = scan(n1 | na << 11, lane, tot);
        b1 = pre & 0x7FF;
        T2 = tot >> 11;
        // Their tier-3 digits, from bit pa of the window (the 128 bytes from the block's first word) or from memory.
        uint32_t pa = o + 2 * (pre >> 11);
        bool near = o + 2 * T2 + 32 < 992;
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
    if (nl != 0 && nl <= 32 && o + 8 * r0 + 18 * (nl - 1) < 992) {
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

// A step's 32 weights a lane as the B fragments of its 8 n-tiles (R[2n], R[2n + 1]).
__device__ __forceinline__ void decode_step(const Step& st, Tiers ts, const uint8_t* __restrict__ blocks, int lane, uint32_t* s2, const uint32_t* tab, uint32_t R[16]) {
    uint32_t ew[8];
    exponents(st, ts, blocks, lane, s2, tab, ew);
    pairs(st.sw, ew, R);
}

// The layout of v0.27 and before as a policy (its decode only: unpacking it, to pack the matrix again).
struct TieredV1 {
    const uint8_t* data;
    const uint8_t* blocks;
    const int32_t* block_base;
    Tiers ts;
    typedef Step St;
    static constexpr bool kTable = true;  // decode needs the groups' table and a warp's scratch
    __device__ __forceinline__ void load(St& st, int64_t step, int lane) const { load_step(st, data, blocks, block_base, step, lane); }
    __device__ __forceinline__ void decode(const St& st, int lane, uint32_t* s2, const uint32_t* tab, uint32_t R[16]) const { decode_step_v1(st, ts, blocks, lane, s2, tab, R); }
    // Ahead: a step's fixed part (1280 bytes, 10 lines of 128: lane i takes line i) to L2; its block (a few bytes) is read when it is decoded.
    __device__ __forceinline__ void prefetch(int64_t step, int lane) const {
        if (lane < (int)(STEP_BYTES / 128)) pf_l2(data + step * STEP_BYTES + lane * 128);
    }
    __device__ __forceinline__ void prefetch_exc(int64_t, int64_t, int) const {}
};

// The kernels below are written once for both layouts, each a policy:
// its step's loads (St, load) and their decode into a lane's B fragments.
struct Tiered {
    const uint8_t* data;
    const uint8_t* blocks;
    const int32_t* block_base;
    Tiers ts;
    typedef Step St;
    static constexpr bool kTable = true;  // decode needs the groups' table and a warp's scratch
    __device__ __forceinline__ void load(St& st, int64_t step, int lane) const { load_step(st, data, blocks, block_base, step, lane); }
    __device__ __forceinline__ void decode(const St& st, int lane, uint32_t* s2, const uint32_t* tab, uint32_t R[16]) const { decode_step(st, ts, blocks, lane, s2, tab, R); }
    // Ahead: a step's fixed part (1280 bytes, 10 lines of 128: lane i takes line i) to L2; its block (a few bytes) is read when it is decoded.
    __device__ __forceinline__ void prefetch(int64_t step, int lane) const {
        if (lane < (int)(STEP_BYTES / 128)) pf_l2(data + step * STEP_BYTES + lane * 128);
    }
    __device__ __forceinline__ void prefetch_exc(int64_t, int64_t, int) const {}
};

// The tiered layout in an expert cache's places (glyd_gpu_mma_moe_placed): place e's `per` steps are the stack's (data as Tiered's), its
// block entries its own per + 1 (from block_base + e (per + 1)), offsets from its own blocks, which start `stride` bytes after the place
// before's (a multiple of 4: a block's word alignment, which the packer's flag depends on, is kept). Its decode is Tiered's.
struct TieredPlaced {
    const uint8_t* data;
    const uint8_t* blocks;
    const int32_t* block_base;
    Tiers ts;
    int64_t per, stride;
    typedef Step St;
    static constexpr bool kTable = true;
    __device__ __forceinline__ void load(St& st, int64_t step, int lane) const {
        const int64_t e = step / per;
        load_step(st, data, blocks, block_base + e, step, lane, e * stride);
    }
    __device__ __forceinline__ void decode(const St& st, int lane, uint32_t* s2, const uint32_t* tab, uint32_t R[16]) const { decode_step(st, ts, blocks, lane, s2, tab, R); }
    __device__ __forceinline__ void prefetch(int64_t step, int lane) const {
        if (lane < (int)(STEP_BYTES / 128)) pf_l2(data + step * STEP_BYTES + lane * 128);
    }
    __device__ __forceinline__ void prefetch_exc(int64_t, int64_t, int) const {}
};

// The compact tiered layout (COMPACT_TAG in tiers[0]'s top byte): a matrix whose finite weights are bf16 with three
// fraction bits, the four below zero (Candidate A's values), in the tiered layout with each sign-and-mantissa byte
// kept as a nibble (its bits 5-7 and 0; the omitted bits zero): a step's 256 bytes of digits as they are, then its
// 512 nibble bytes, two bytes each in lane and half order (768 bytes a step). Exponents, blocks and words are the
// tiered layout's; a step decodes as Tiered's once its bytes are restored in registers.
constexpr uint32_t COMPACT_TAG = 3;
constexpr int64_t COMPACT_STEP_BYTES = 768;
// Four nibbles restore four sign-and-mantissa bytes, the omitted bits zero.
__host__ __device__ __forceinline__ uint32_t compact_expand4(uint32_t value) {
    uint32_t x = value & 0xffffu;
    x = (x | (x << 8)) & 0x00ff00ffu;
    x = (x | (x << 4)) & 0x0f0f0f0fu;
    return (x & 0x01010101u) | ((x & 0x0e0e0e0eu) << 4);
}
struct Compact {
    const uint8_t* data;
    const uint8_t* blocks;
    const int32_t* block_base;
    Tiers ts;
    typedef Step St;
    static constexpr bool kTable = true;
    __device__ __forceinline__ void load(St& st, int64_t step, int lane) const {
        const uint8_t* p = data + step * COMPACT_STEP_BYTES;
        const uint32_t* digits = (const uint32_t*)p + lane;
        st.pd[0] = __ldg(digits);
        st.pd[1] = __ldg(digits + 32);
        const uint2* sig = (const uint2*)(p + 256) + lane;
        uint2 lo = __ldg(sig), hi = __ldg(sig + 32);
        st.sw[0] = compact_expand4(lo.x); st.sw[1] = compact_expand4(lo.x >> 16);
        st.sw[2] = compact_expand4(lo.y); st.sw[3] = compact_expand4(lo.y >> 16);
        st.sw[4] = compact_expand4(hi.x); st.sw[5] = compact_expand4(hi.x >> 16);
        st.sw[6] = compact_expand4(hi.y); st.sw[7] = compact_expand4(hi.y >> 16);
        st.blk = (uint32_t)__ldg(block_base + step) & 0x7fffffffu;
        st.end = (uint32_t)__ldg(block_base + step + 1) & 0x7fffffffu;
        const uint32_t* bw = (const uint32_t*)blocks;
        st.win = __ldg(bw + (st.blk >> 2) + lane);
        st.tail = __ldg(bw + (st.end >> 2) - 1 - lane);
        st.tail1 = __ldg(bw + (st.end >> 2));
    }
    __device__ __forceinline__ void decode(const St& st, int lane, uint32_t* s2, const uint32_t* tab, uint32_t R[16]) const { decode_step(st, ts, blocks, lane, s2, tab, R); }
    __device__ __forceinline__ void prefetch(int64_t step, int lane) const {
        if (lane < (int)(COMPACT_STEP_BYTES / 128)) pf_l2(data + step * COMPACT_STEP_BYTES + lane * 128);
    }
    __device__ __forceinline__ void prefetch_exc(int64_t, int64_t, int) const {}
};

// The six-bit block layout (INT6_TAG in tiers[0]'s top byte): each run of 32 values of a row of W (columns 32 kb on) as six-bit codes u,
// 1 to 63, and one bf16 scale s for the run; the weight is round_bf16((u - 32) s) (the product of a 6-bit integer and an 8-bit significand,
// rounded once: the packer's own reference). A warp step's 768 bytes have the compact layout's shape: 256 bytes of the codes' high two bits
// (lane l's two words at d * 32 + l; its pair r, weights 2r and 2r + 1 of the step's 32 a lane, at bits 2(r mod 8) and 16 + 2(r mod 8) of
// word r / 8), then 512 bytes of their low four bits (lane l's four words, two halves of 8 bytes at l and l + 32; pair r at bits 4(r mod 4)
// and 16 + 4(r mod 4) of word r / 4). The scales (`blocks`): 128 bytes a pair of steps, those of step >> 1 (row block rb, columns 32 kb on:
// a row block's steps are rb * KS + ks with KS even), its 64 rows' scales as row 8n + g at place 8g + n, so that a lane's eight rows (g its
// group) are one 16-byte load. `block_base` is not read. A pair's two bf16s are an OR (0x4300 | u: 128 + u), a subtraction of 160 (exact)
// and a product by the scale, two bf16x2 instructions a pair.
constexpr uint32_t INT6_TAG = 4;
constexpr int64_t INT6_STEP_BYTES = 768;
struct Int6 {
    const uint8_t* data;
    const uint8_t* scales;
    struct St {
        uint32_t hi[2], lo[4], sc[4];
    };
    static constexpr bool kTable = true;  // (the tiered kernels' shape: their table and a warp's scratch are not read)
    __device__ __forceinline__ void load(St& st, int64_t step, int lane) const {
        const uint8_t* p = data + step * INT6_STEP_BYTES;
        const uint32_t* h = (const uint32_t*)p + lane;
        st.hi[0] = __ldg(h);
        st.hi[1] = __ldg(h + 32);
        const uint2* l = (const uint2*)(p + 256) + lane;
        uint2 a = __ldg(l), b = __ldg(l + 32);
        st.lo[0] = a.x; st.lo[1] = a.y; st.lo[2] = b.x; st.lo[3] = b.y;
        uint4 c = __ldg((const uint4*)(scales + (step >> 1) * 128) + (lane >> 2));
        st.sc[0] = c.x; st.sc[1] = c.y; st.sc[2] = c.z; st.sc[3] = c.w;
    }
    // (a step through shared memory, mma_steps_staged: its bytes and its pair's scales, in 16-byte chunks; read back with the offsets load uses)
    static constexpr int kStage = INT6_STEP_BYTES + 128;
    __device__ __forceinline__ const uint8_t* chunk(int64_t step, int c) const { return c < 48 ? data + step * INT6_STEP_BYTES + 16 * c : scales + (step >> 1) * 128 + 16 * (c - 48); }
    __device__ __forceinline__ void load_s(St& st, const uint8_t* p, int lane) const {
        const uint32_t* h = (const uint32_t*)p + lane;
        st.hi[0] = h[0];
        st.hi[1] = h[32];
        const uint2* l = (const uint2*)(p + 256) + lane;
        uint2 a = l[0], b = l[32];
        st.lo[0] = a.x; st.lo[1] = a.y; st.lo[2] = b.x; st.lo[3] = b.y;
        uint4 c = ((const uint4*)(p + INT6_STEP_BYTES))[lane >> 2];
        st.sc[0] = c.x; st.sc[1] = c.y; st.sc[2] = c.z; st.sc[3] = c.w;
    }
    __device__ __forceinline__ void decode(const St& st, int, uint32_t*, const uint32_t*, uint32_t R[16]) const {
        const uint32_t off = 0x43204320u;  // (160, 160)
#pragma unroll
        for (int r = 0; r < 16; r++) {
            uint32_t x = 0x43004300u | ((st.lo[r >> 2] >> (4 * (r & 3))) & 0x000F000Fu) | (((st.hi[r >> 3] >> (2 * (r & 7))) & 0x00030003u) << 4);
            uint32_t sw = __byte_perm(st.sc[r >> 2], 0, ((r >> 1) & 1) ? 0x3232 : 0x1010);
            __nv_bfloat162 v = __hmul2(__hsub2(*reinterpret_cast<const __nv_bfloat162*>(&x), *reinterpret_cast<const __nv_bfloat162*>(&off)), *reinterpret_cast<const __nv_bfloat162*>(&sw));
            R[r] = *reinterpret_cast<uint32_t*>(&v);
        }
    }
    __device__ __forceinline__ void prefetch(int64_t step, int lane) const {
        if (lane < (int)(INT6_STEP_BYTES / 128)) pf_l2(data + step * INT6_STEP_BYTES + lane * 128);
        else if (lane == (int)(INT6_STEP_BYTES / 128) && !(step & 1)) pf_l2(scales + (step >> 1) * 128);
    }
    __device__ __forceinline__ void prefetch_exc(int64_t, int64_t, int) const {}
};

// The five-bit block layout (INT5_TAG in tiers[0]'s top byte): each run of 32 values of a row of W (columns 32 kb on) as five-bit codes u,
// 1 to 31, and one bf16 scale s for the run; the weight is round_bf16((u - 16) s) (the product of a 5-bit integer and an 8-bit significand,
// rounded once: the packer's own reference). A warp step's 640 bytes have the six-bit layout's shape: 128 bytes of the codes' high bit
// (lane l's word at l; its pair r, weights 2r and 2r + 1 of the step's 32 a lane, at bits r and 16 + r), then 512 bytes of their low four
// bits (lane l's four words, two halves of 8 bytes at l and l + 32; pair r at bits 4(r mod 4) and 16 + 4(r mod 4) of word r / 4). The scales
// (`blocks`): 128 bytes a pair of steps, those of step >> 1 (row block rb, columns 32 kb on: a row block's steps are rb * KS + ks with KS
// even), its 64 rows' scales as row 8n + g at place 8g + n, so that a lane's eight rows (g its group) are one 16-byte load. `block_base` is
// not read. A pair's two bf16s are an OR (0x4300 | u: 128 + u), a subtraction of 144 (exact) and a product by the scale, two bf16x2
// instructions a pair.
constexpr uint32_t INT5_TAG = 5;
constexpr int64_t INT5_STEP_BYTES = 640;
struct Int5 {
    const uint8_t* data;
    const uint8_t* scales;
    struct St {
        uint32_t hi, lo[4], sc[4];
    };
    static constexpr bool kTable = true;  // (the tiered kernels' shape: their table and a warp's scratch are not read)
    __device__ __forceinline__ void load(St& st, int64_t step, int lane) const {
        const uint8_t* p = data + step * INT5_STEP_BYTES;
        const uint32_t* h = (const uint32_t*)p + lane;
        st.hi = __ldg(h);
        const uint2* l = (const uint2*)(p + 128) + lane;
        uint2 a = __ldg(l), b = __ldg(l + 32);
        st.lo[0] = a.x; st.lo[1] = a.y; st.lo[2] = b.x; st.lo[3] = b.y;
        uint4 c = __ldg((const uint4*)(scales + (step >> 1) * 128) + (lane >> 2));
        st.sc[0] = c.x; st.sc[1] = c.y; st.sc[2] = c.z; st.sc[3] = c.w;
    }
    static constexpr int kStage = INT5_STEP_BYTES + 128;  // (as Int6's)
    __device__ __forceinline__ const uint8_t* chunk(int64_t step, int c) const { return c < 40 ? data + step * INT5_STEP_BYTES + 16 * c : scales + (step >> 1) * 128 + 16 * (c - 40); }
    __device__ __forceinline__ void load_s(St& st, const uint8_t* p, int lane) const {
        st.hi = ((const uint32_t*)p)[lane];
        const uint2* l = (const uint2*)(p + 128) + lane;
        uint2 a = l[0], b = l[32];
        st.lo[0] = a.x; st.lo[1] = a.y; st.lo[2] = b.x; st.lo[3] = b.y;
        uint4 c = ((const uint4*)(p + INT5_STEP_BYTES))[lane >> 2];
        st.sc[0] = c.x; st.sc[1] = c.y; st.sc[2] = c.z; st.sc[3] = c.w;
    }
    __device__ __forceinline__ void decode(const St& st, int, uint32_t*, const uint32_t*, uint32_t R[16]) const {
        const uint32_t off = 0x43104310u;  // (144, 144)
#pragma unroll
        for (int r = 0; r < 16; r++) {
            uint32_t x = 0x43004300u | ((st.lo[r >> 2] >> (4 * (r & 3))) & 0x000F000Fu) | (((st.hi >> r) & 0x00010001u) << 4);
            uint32_t sw = __byte_perm(st.sc[r >> 2], 0, ((r >> 1) & 1) ? 0x3232 : 0x1010);
            __nv_bfloat162 v = __hmul2(__hsub2(*reinterpret_cast<const __nv_bfloat162*>(&x), *reinterpret_cast<const __nv_bfloat162*>(&off)), *reinterpret_cast<const __nv_bfloat162*>(&sw));
            R[r] = *reinterpret_cast<uint32_t*>(&v);
        }
    }
    __device__ __forceinline__ void prefetch(int64_t step, int lane) const {
        if (lane < (int)(INT5_STEP_BYTES / 128)) pf_l2(data + step * INT5_STEP_BYTES + lane * 128);
        else if (lane == (int)(INT5_STEP_BYTES / 128) && !(step & 1)) pf_l2(scales + (step >> 1) * 128);
    }
    __device__ __forceinline__ void prefetch_exc(int64_t, int64_t, int) const {}
};

// The four-bit block layout (INT4_TAG in tiers[0]'s top byte): each run of 32 values of a row of W (columns 32 kb on) as four-bit codes u,
// 1 to 15, and one bf16 scale s for the run; the weight is round_bf16((u - 8) s) (the product of a 4-bit integer and an 8-bit significand,
// rounded once: the packer's own reference). A warp step's 512 bytes are the five- and six-bit layouts' low four bits alone (lane l's four
// words, two halves of 8 bytes at l and l + 32; pair r, weights 2r and 2r + 1 of the step's 32 a lane, at bits 4(r mod 4) and 16 + 4(r mod 4)
// of word r / 4). The scales (`blocks`): 128 bytes a pair of steps, those of step >> 1 (row block rb, columns 32 kb on: a row block's steps are
// rb * KS + ks with KS even), its 64 rows' scales as row 8n + g at place 8g + n, so that a lane's eight rows (g its group) are one 16-byte
// load. `block_base` is not read. A pair's two bf16s are an OR (0x4300 | u: 128 + u), a subtraction of 136 (exact) and a product by the
// scale, two bf16x2 instructions a pair.
constexpr uint32_t INT4_TAG = 6;
constexpr int64_t INT4_STEP_BYTES = 512;
struct Int4 {
    const uint8_t* data;
    const uint8_t* scales;
    struct St {
        uint32_t lo[4], sc[4];
    };
    static constexpr bool kTable = true;  // (the tiered kernels' shape: their table and a warp's scratch are not read)
    __device__ __forceinline__ void load(St& st, int64_t step, int lane) const {
        const uint8_t* p = data + step * INT4_STEP_BYTES;
        const uint2* l = (const uint2*)p + lane;
        uint2 a = __ldg(l), b = __ldg(l + 32);
        st.lo[0] = a.x; st.lo[1] = a.y; st.lo[2] = b.x; st.lo[3] = b.y;
        uint4 c = __ldg((const uint4*)(scales + (step >> 1) * 128) + (lane >> 2));
        st.sc[0] = c.x; st.sc[1] = c.y; st.sc[2] = c.z; st.sc[3] = c.w;
    }
    static constexpr int kStage = INT4_STEP_BYTES + 128;  // (as Int6's)
    __device__ __forceinline__ const uint8_t* chunk(int64_t step, int c) const { return c < 32 ? data + step * INT4_STEP_BYTES + 16 * c : scales + (step >> 1) * 128 + 16 * (c - 32); }
    __device__ __forceinline__ void load_s(St& st, const uint8_t* p, int lane) const {
        const uint2* l = (const uint2*)p + lane;
        uint2 a = l[0], b = l[32];
        st.lo[0] = a.x; st.lo[1] = a.y; st.lo[2] = b.x; st.lo[3] = b.y;
        uint4 c = ((const uint4*)(p + INT4_STEP_BYTES))[lane >> 2];
        st.sc[0] = c.x; st.sc[1] = c.y; st.sc[2] = c.z; st.sc[3] = c.w;
    }
    __device__ __forceinline__ void decode(const St& st, int, uint32_t*, const uint32_t*, uint32_t R[16]) const {
        const uint32_t off = 0x43084308u;  // (136, 136)
#pragma unroll
        for (int r = 0; r < 16; r++) {
            uint32_t x = 0x43004300u | ((st.lo[r >> 2] >> (4 * (r & 3))) & 0x000F000Fu);
            uint32_t sw = __byte_perm(st.sc[r >> 2], 0, ((r >> 1) & 1) ? 0x3232 : 0x1010);
            __nv_bfloat162 v = __hmul2(__hsub2(*reinterpret_cast<const __nv_bfloat162*>(&x), *reinterpret_cast<const __nv_bfloat162*>(&off)), *reinterpret_cast<const __nv_bfloat162*>(&sw));
            R[r] = *reinterpret_cast<uint32_t*>(&v);
        }
    }
    __device__ __forceinline__ void prefetch(int64_t step, int lane) const {
        if (lane < (int)(INT4_STEP_BYTES / 128)) pf_l2(data + step * INT4_STEP_BYTES + lane * 128);
        else if (lane == (int)(INT4_STEP_BYTES / 128) && !(step & 1)) pf_l2(scales + (step >> 1) * 128);
    }
    __device__ __forceinline__ void prefetch_exc(int64_t, int64_t, int) const {}
};

// The exact four-bit block layout (INT4H_TAG in tiers[0]'s top byte): Int4's steps, codes and places, for checkpoints trained to four bits
// (quantization-aware), whose bf16 weights it holds bit for bit. A run's weight is round_bf16(k s), k = u - 8 (-8 to 7), with a scale of its own
// form: bit 15 the sign, bits 11-14 the exponent over the pack's base (tiers[1]: the least biased f32 exponent of its scales), bits 0-10 the top
// of the fraction (a 12-bit significand). The product is exact in fp32 (4 bits by 12) and rounded to bf16 once, as the packer's reference
// (glyd-gpu/src/pack.rs int4h_value). A pair: the codes as floats (an OR into 2^23 and a subtraction), two products, one rounding to bf16x2.
constexpr uint32_t INT4H_TAG = 13;
struct Int4H : Int4 {
    uint32_t base;
    __device__ __forceinline__ void decode(const St& st, int, uint32_t*, const uint32_t*, uint32_t R[16]) const {
#pragma unroll
        for (int r = 0; r < 16; r++) {
            const uint32_t h = (st.sc[r >> 2] >> (16 * ((r >> 1) & 1))) & 0xFFFFu;
            const float sc = __uint_as_float(((h & 0x8000u) << 16) | ((((h >> 11) & 0xFu) + base) << 23) | ((h & 0x7FFu) << 12));
            const uint32_t w = st.lo[r >> 2] >> (4 * (r & 3));
            const float a = __uint_as_float(0x4B000000u | (w & 0xFu)) - 8388616.f, b = __uint_as_float(0x4B000000u | ((w >> 16) & 0xFu)) - 8388616.f;
            __nv_bfloat162 v = __floats2bfloat162_rn(__fmul_rn(a, sc), __fmul_rn(b, sc));
            R[r] = *reinterpret_cast<uint32_t*>(&v);
        }
    }
};

// The exact FP4 block layout (FP4_TAG in tiers[0]'s top byte): an MX FP4 checkpoint's matrices (E2M1 values, an E8M0 scale a run of 32) held as released, 4.25 bits
// a weight. Int4's steps and places, each code an E2M1 value (bit 3 its sign, bits 0-2 its magnitude: 0, 0.5, 1, 1.5, 2, 3, 4, 6), and a run's scale one E8M0 byte e
// (2^(e - 127); the packer takes e of 2 to 128), 64 bytes for two steps, a row's at 8 g + n as Int4's: the weight is bf16(value 2^(e - 127)), exact (glyd-gpu/src/pack.rs
// fp4_value).
constexpr uint32_t FP4_TAG = 14;
// A lane's 16 pairs of a step from its four code words (pair r: bits 4 (r mod 4) and 16 + 4 (r mod 4) of lo[r / 4]) and its eight scale bytes (scale n = r / 2: byte n mod 4 of e[n / 4]), as two bf16
// each, exactly: a code's bits placed as a bf16's (its sign at 15, its two exponent bits at 7-8, its mantissa bit at 6) are its E2M1 value times 2^-126, the two smallest (0 and 0.5) as bf16 subnormals,
// so one product by 2^(e - 1) (e + 126 in the exponent: e at most 128) is the weight. Two masks, two shifts and the product a pair; the tables, the -64 s and the signs' XOR it replaced had been 6 integer
// instructions a pair, and the integer pipe issues at half the rate of the others (an A100 and an H100: the FP4 experts bound by it).
__device__ __forceinline__ void fp4_step(const uint32_t (&lo)[4], uint32_t e0, uint32_t e1, uint32_t R[16]) {
    uint32_t s[8];
    e0 += 0x7E7E7E7Eu;  // (e + 126 a byte: no carries, e at most 128)
    e1 += 0x7E7E7E7Eu;
#pragma unroll
    for (int n = 0; n < 8; n++) s[n] = __byte_perm(n < 4 ? e0 : e1, 0, 0x0404u | ((n & 3) << 4) | ((n & 3) << 12)) >> 1;  // (2^(e - 1) twice)
#pragma unroll
    for (int q = 0; q < 4; q++)
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int r = 4 * q + j;
            const uint32_t t = lo[q] >> (4 * j);
            const uint32_t x = ((t & 0x00070007u) << 6) | ((t & 0x00080008u) << 12);
            __nv_bfloat162 v = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&x), *reinterpret_cast<const __nv_bfloat162*>(&s[r >> 1]));
            R[r] = *reinterpret_cast<uint32_t*>(&v);
        }
}
struct Fp4 {
    const uint8_t* data;
    const uint8_t* scales;
    struct St {
        uint32_t lo[4], sc[2];
    };
    static constexpr bool kTable = true;  // (the tiered kernels' shape: their table and a warp's scratch are not read)
    __device__ __forceinline__ void load(St& st, int64_t step, int lane) const {
        const uint8_t* p = data + step * INT4_STEP_BYTES;
        const uint2* l = (const uint2*)p + lane;
        uint2 a = __ldg(l), b = __ldg(l + 32);
        st.lo[0] = a.x; st.lo[1] = a.y; st.lo[2] = b.x; st.lo[3] = b.y;
        uint2 c = __ldg((const uint2*)(scales + (step >> 1) * 64) + (lane >> 2));
        st.sc[0] = c.x; st.sc[1] = c.y;
    }
    static constexpr int kStage = INT4_STEP_BYTES + 64;  // (as Int6's: the codes, then the pair's 64 bytes of scales)
    __device__ __forceinline__ const uint8_t* chunk(int64_t step, int c) const { return c < 32 ? data + step * INT4_STEP_BYTES + 16 * c : scales + (step >> 1) * 64 + 16 * (c - 32); }
    __device__ __forceinline__ void load_s(St& st, const uint8_t* p, int lane) const {
        const uint2* l = (const uint2*)p + lane;
        uint2 a = l[0], b = l[32];
        st.lo[0] = a.x; st.lo[1] = a.y; st.lo[2] = b.x; st.lo[3] = b.y;
        uint2 c = ((const uint2*)(p + INT4_STEP_BYTES))[lane >> 2];
        st.sc[0] = c.x; st.sc[1] = c.y;
    }
    __device__ __forceinline__ void decode(const St& st, int, uint32_t*, const uint32_t*, uint32_t R[16]) const { fp4_step(st.lo, st.sc[0], st.sc[1], R); }
    __device__ __forceinline__ void prefetch(int64_t step, int lane) const {
        if (lane < (int)(INT4_STEP_BYTES / 128)) pf_l2(data + step * INT4_STEP_BYTES + lane * 128);
        else if (lane == (int)(INT4_STEP_BYTES / 128) && !(step & 1)) pf_l2(scales + (step >> 1) * 64);
    }
    __device__ __forceinline__ void prefetch_exc(int64_t, int64_t, int) const {}
};

// The exact FP4 layout with packed scales (FP4R_TAG): Fp4's codes and decode, each run's E8M0 scale its row's base plus two bits (an MX checkpoint's scales differ little
// along a row: the packer takes a matrix whose rows each span at most 4 values, else Fp4). 4.07 bits a weight at 4,096 columns (Fp4's 4.25). A row block's scales: its
// 64 rows' bases (a row's at 8 g + n, as Int4's scales), then a pair of steps' 64 offsets in 16 bytes (a row's 2 bits at bit 2 n of the u16 at 2 g).
constexpr uint32_t FP4R_TAG = 15;
struct Fp4R {
    const uint8_t* data;
    const uint8_t* scales;
    uint32_t ks, mg, sh;  // steps a row block (K / 16), and step / ks as a multiply-high (fp4r_fmt's mg, sh: five instructions; the division and its
                          // power-of-two branch had been 24 a step, issued whether taken or not)
    struct St {
        uint32_t lo[4], base[2], off;
    };
    static constexpr bool kTable = true;  // (the tiered kernels' shape: their table and a warp's scratch are not read)
    __device__ __forceinline__ const uint8_t* block(int64_t step, uint32_t& k) const {
        const uint32_t n = (uint32_t)step, t = __umulhi(n, mg), rb = (t + ((n - t) >> 1)) >> sh;
        k = (uint32_t)step - rb * ks;
        return scales + (size_t)rb * (64 + ks * 8);
    }
    __device__ __forceinline__ void load(St& st, int64_t step, int lane) const {
        const uint8_t* p = data + step * INT4_STEP_BYTES;
        const uint2* l = (const uint2*)p + lane;
        uint2 a = __ldg(l), b = __ldg(l + 32);
        st.lo[0] = a.x; st.lo[1] = a.y; st.lo[2] = b.x; st.lo[3] = b.y;
        uint32_t k;
        const uint8_t* r = block(step, k);
        const uint2 c = __ldg((const uint2*)r + (lane >> 2));
        st.base[0] = c.x; st.base[1] = c.y;
        st.off = __ldg((const uint16_t*)(r + 64 + (k >> 1) * 16) + (lane >> 2));
    }
    static constexpr int kStage = INT4_STEP_BYTES + 64 + 16;  // (as Int6's: the codes, the row block's bases, the pair's offsets)
    __device__ __forceinline__ const uint8_t* chunk(int64_t step, int c) const {
        if (c < 32) return data + step * INT4_STEP_BYTES + 16 * c;
        uint32_t k;
        const uint8_t* r = block(step, k);
        return c < 36 ? r + 16 * (c - 32) : r + 64 + (k >> 1) * 16;
    }
    __device__ __forceinline__ void load_s(St& st, const uint8_t* p, int lane) const {
        const uint2* l = (const uint2*)p + lane;
        uint2 a = l[0], b = l[32];
        st.lo[0] = a.x; st.lo[1] = a.y; st.lo[2] = b.x; st.lo[3] = b.y;
        const uint2 c = ((const uint2*)(p + INT4_STEP_BYTES))[lane >> 2];
        st.base[0] = c.x; st.base[1] = c.y;
        st.off = ((const uint16_t*)(p + INT4_STEP_BYTES + 64))[lane >> 2];
    }
    __device__ __forceinline__ void decode(const St& st, int, uint32_t*, const uint32_t*, uint32_t R[16]) const {
        // (the offsets' 2-bit fields spread to bytes and added to the bases: a scale is at most 254, so no byte carries)
        const auto bytes = [](uint32_t x) { return (x | (x << 6) | (x << 12) | (x << 18)) & 0x03030303u; };
        fp4_step(st.lo, st.base[0] + bytes(st.off & 0xFFu), st.base[1] + bytes(st.off >> 8), R);
    }
    __device__ __forceinline__ void prefetch(int64_t step, int lane) const {
        if (lane < (int)(INT4_STEP_BYTES / 128)) pf_l2(data + step * INT4_STEP_BYTES + lane * 128);
        else if (lane == (int)(INT4_STEP_BYTES / 128) && !(step & 1)) {
            uint32_t k;
            const uint8_t* r = block(step, k);
            pf_l2(r + 64 + (k >> 1) * 16);
        }
    }
    __device__ __forceinline__ void prefetch_exc(int64_t, int64_t, int) const {}
};

// The three-bit block layout (INT3_TAG in tiers[0]'s top byte): each run of 32 values of a row of W (columns 32 kb on) as three-bit codes u,
// 1 to 7 (the packer's codes run -3 to 3, as the six- and five-bit layouts' -31 to 31 and -15 to 15: the eighth code, 0, decodes as -4 s), and
// one bf16 scale s for the run; the weight is round_bf16((u - 4) s) (the product of a 3-bit integer and an 8-bit significand, rounded once: the
// packer's own reference). A warp step's 384 bytes: 128 bytes of the codes' high bit as the five-bit layout's (lane l's word at l; its pair r,
// weights 2r and 2r + 1 of the step's 32 a lane, at bits r and 16 + r), then 256 bytes of their low two bits as the six-bit layout's high two
// (lane l's two words at 32 d + l; pair r at bits 2(r mod 8) and 16 + 2(r mod 8) of word r / 8). The scales (`blocks`) as the four-bit layout's.
// A pair's two bf16s are an OR (0x4300 | u: 128 + u), a subtraction of 132 (exact) and a product by the scale, two bf16x2 instructions a pair.
constexpr uint32_t INT3_TAG = 7;
constexpr int64_t INT3_STEP_BYTES = 384;
struct Int3 {
    const uint8_t* data;
    const uint8_t* scales;
    struct St {
        uint32_t hi, lo[2], sc[4];
    };
    static constexpr bool kTable = true;  // (the tiered kernels' shape: their table and a warp's scratch are not read)
    __device__ __forceinline__ void load(St& st, int64_t step, int lane) const {
        const uint8_t* p = data + step * INT3_STEP_BYTES;
        const uint32_t* h = (const uint32_t*)p + lane;
        st.hi = __ldg(h);
        st.lo[0] = __ldg(h + 32);
        st.lo[1] = __ldg(h + 64);
        uint4 c = __ldg((const uint4*)(scales + (step >> 1) * 128) + (lane >> 2));
        st.sc[0] = c.x; st.sc[1] = c.y; st.sc[2] = c.z; st.sc[3] = c.w;
    }
    static constexpr int kStage = INT3_STEP_BYTES + 128;  // (as Int6's)
    __device__ __forceinline__ const uint8_t* chunk(int64_t step, int c) const { return c < 24 ? data + step * INT3_STEP_BYTES + 16 * c : scales + (step >> 1) * 128 + 16 * (c - 24); }
    __device__ __forceinline__ void load_s(St& st, const uint8_t* p, int lane) const {
        const uint32_t* h = (const uint32_t*)p + lane;
        st.hi = h[0];
        st.lo[0] = h[32];
        st.lo[1] = h[64];
        uint4 c = ((const uint4*)(p + INT3_STEP_BYTES))[lane >> 2];
        st.sc[0] = c.x; st.sc[1] = c.y; st.sc[2] = c.z; st.sc[3] = c.w;
    }
    __device__ __forceinline__ void decode(const St& st, int, uint32_t*, const uint32_t*, uint32_t R[16]) const {
        const uint32_t off = 0x43044304u;  // (132, 132)
#pragma unroll
        for (int r = 0; r < 16; r++) {
            uint32_t x = 0x43004300u | ((st.lo[r >> 3] >> (2 * (r & 7))) & 0x00030003u) | (((st.hi >> r) & 0x00010001u) << 2);
            uint32_t sw = __byte_perm(st.sc[r >> 2], 0, ((r >> 1) & 1) ? 0x3232 : 0x1010);
            __nv_bfloat162 v = __hmul2(__hsub2(*reinterpret_cast<const __nv_bfloat162*>(&x), *reinterpret_cast<const __nv_bfloat162*>(&off)), *reinterpret_cast<const __nv_bfloat162*>(&sw));
            R[r] = *reinterpret_cast<uint32_t*>(&v);
        }
    }
    __device__ __forceinline__ void prefetch(int64_t step, int lane) const {
        if (lane < (int)(INT3_STEP_BYTES / 128)) pf_l2(data + step * INT3_STEP_BYTES + lane * 128);
        else if (lane == (int)(INT3_STEP_BYTES / 128) && !(step & 1)) pf_l2(scales + (step >> 1) * 128);
    }
    __device__ __forceinline__ void prefetch_exc(int64_t, int64_t, int) const {}
};

// The 12-bit layout (mma12), split byte: a weight's bf16 as its low byte (the exponent's lowest bit and the 7
// mantissa bits), stored as it is, and its high byte (the sign and the exponent's other 7 bits), coded in 4 bits: the
// sign and an offset 0-7 from the matrix's base hb (the 8 values of the high byte's 7 bits from hb, 16 exponents,
// holding the most weights; hb 0-120, in each byte of hb4). A warp step: its lanes' codes, 16 bytes a lane, then its
// 1024 low bytes as two halves (every lane's first 16, then every lane's last 16). Code word q of a lane holds n-tiles
// 2q and 2q + 1 (weights 8q to 8q + 7): 2q's codes in bits {7, 2, 1, 0} of byte j (sign, offset), 2q + 1's rotated
// by 4 (its sign in bit 3 of byte j, its offset in bits 4-6 of byte j - 1 mod 4). A lane's 4 high bytes are then an
// AND and an add (2q + 1's a funnel shift first), a pair of bf16s one byte permute of [high, low, high, low]. A weight
// outside the 8 (whatever its exponent: zeros, subnormals, infinities alike) is an exception, coded with offset 0: its
// entry, its weight (lane * 32 + i) and the byte to XOR into its high byte (hb ^ the exponent >> 1) << 16; a step's run
// from exc_base[step] to exc_base[step + 1].
constexpr int64_t STEP12 = 1536;

struct Nib {
    const uint8_t* data;
    const uint32_t* exc;
    const int32_t* exc_base;
    uint32_t hb4;  // the high bytes' base, hb in each byte
    int xmax = 4;  // decode_x: a step with more than xmax exceptions takes them through the scratch (GLYD_GPU_XMAX, read by nib12)
    struct St {
        uint32_t nb[4], sw[8];
        int e0, e1;
    };
    static constexpr bool kTable = false;
    __device__ __forceinline__ void load(St& st, int64_t step, int lane) const {
        const uint8_t* p = data + step * STEP12;
        uint4 c = __ldg((const uint4*)p + lane);
        st.nb[0] = c.x; st.nb[1] = c.y; st.nb[2] = c.z; st.nb[3] = c.w;
        const uint4* sp = (const uint4*)(p + 512) + lane;
        uint4 x0 = __ldg(sp), x1 = __ldg(sp + 32);
        st.sw[0] = x0.x; st.sw[1] = x0.y; st.sw[2] = x0.z; st.sw[3] = x0.w;
        st.sw[4] = x1.x; st.sw[5] = x1.y; st.sw[6] = x1.z; st.sw[7] = x1.w;
        st.e0 = __ldg(exc_base + step);
        st.e1 = __ldg(exc_base + step + 1);
    }
    // The whole-matrix decode's step (mma_unpack_kernel, a warp a step: for cuBLAS, exact mode and the experts'), its
    // loads in two batches: the low bytes and the step's exception bounds, then the codes once those are in (load()
    // issues all at once). The codes' address waits on the first batch through after(its sum), 0: the sum shuffled
    // from its own lane, less the sum (a shuffle the compiler does not see through), so the same bytes, bit for bit.
    // then(): the kernel's own work (its row block and step, a division), once the codes' load is issued. Hopper's
    // alone: on a GH200 (layer 10 of Qwen3-8B, 14B and 32B, 2026-09-29) the layer's decode takes 0.964-0.975 of the
    // time of the 12-bit layout's before split byte, load()'s all at once 1.035-1.060; on an L4 (Qwen3-8B and 4B)
    // 1.020-1.022 against load()'s 1.001-1.003. The decode ahead (a few warps an SM) keeps load() everywhere: on the
    // GH200 0.703-0.725, this order 1.058-1.068; on the L4 0.994-0.995 against 1.106-1.134.
    template <class Then>
    __device__ __forceinline__ void load_decode(int64_t step, int lane, uint32_t R[16], Then then) const {
        const uint8_t* p = data + step * STEP12;
        auto after = [lane](uint32_t v) { return __shfl_sync(0xffffffffu, v, lane) - v; };
        int e0 = __ldg(exc_base + step), e1 = __ldg(exc_base + step + 1);
        const uint4* sp = (const uint4*)(p + 512) + lane;
        uint4 x0 = __ldg(sp), x1 = __ldg(sp + 32);
        uint4 c = __ldg((const uint4*)(p + after(x0.x + x1.x + e0 + e1)) + lane);
        then();
        uint32_t nb[4] = {c.x, c.y, c.z, c.w}, H[8];
        high(nb, H);
        const uint32_t* e = exc;
        patch([e](int k) { return __ldg(e + k); }, e0, e1, lane, H);
        uint32_t L[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
        pairs(L, H, R);
    }
    // Code word q's high bytes: n-tile 2q's (h = 0), 2q + 1's (h = 1).
    static __device__ __forceinline__ uint32_t high1(uint32_t nb, uint32_t hb4, int h) { return ((h ? __funnelshift_l(nb, nb, 4) : nb) & 0x87878787u) + hb4; }
    // A lane's codes (nb, 4 words) as its high bytes, 4 a word (H).
    __device__ __forceinline__ void high(const uint32_t nb[4], uint32_t H[8]) const {
#pragma unroll
        for (int q = 0; q < 8; q++) H[q] = high1(nb[q >> 1], hb4, q & 1);
    }
    // The step's exceptions, entries e0 to e1 (the same run for every lane of the warp; entry k is at(k)): each XORs
    // its byte into this lane's word where the weight is this lane's. U: the loop unrolled U times, 0 as nvcc chooses
    // (4 a pass in these kernels, as it did the 12-bit layout's before split byte). An entry a pass (1) where unrolled it
    // spilled (mma_moe_kernel's 64-token products with an activation, sm_80: main's 4 spill instructions, 12
    // unrolled, none so), and 2 in mma12_ws_kernel, as nvcc unrolled it there before split byte: kept a pass an entry,
    // the L4's prompts took 1-2% longer (2026-09-29).
    template <int U = 0, class At>
    static __device__ __forceinline__ void patch(At at, int e0, int e1, int lane, uint32_t H[8]) {
        auto one = [&](int k) {
            uint32_t x = at(k), i = x & 31, v = (x >> 16 & 0xFFu) << (8 * (i & 3));
            uint32_t w = (int)((x >> 5) & 31) == lane ? i >> 2 : 8u;
#pragma unroll
            for (int q = 0; q < 8; q++) H[q] ^= w == (uint32_t)q ? v : 0u;
        };
        if constexpr (U == 0) {
            for (int k = e0; k < e1; k++) one(k);
        } else {
#pragma unroll U
            for (int k = e0; k < e1; k++) one(k);
        }
    }
    // Pairs: [high, low, high, low], one byte permute each (a lane's low bytes L, 4 a word).
    static __device__ __forceinline__ void pairs(const uint32_t L[8], const uint32_t H[8], uint32_t R[16]) {
#pragma unroll
        for (int p = 0; p < 16; p++) R[p] = __byte_perm(L[p >> 1], H[p >> 1], (p & 1) ? 0x7362 : 0x5140);
    }
    // A step's decode where its block's steps have many exceptions (mma_gemm_kernel's heavy blocks): past xmax (4) in the
    // step, through the warp's scratch (s2: 1 KB, a lane's 8 words, zero between steps) the lanes take the run's
    // entries 32 at a time, each setting its byte there, which every lane then XORs into its words and clears, as
    // decode12_rows does; else as decode.
    __device__ __forceinline__ void decode_x(const St& st, int lane, uint32_t* s2, uint32_t R[16]) const {
        if (st.e1 - st.e0 <= xmax) return decode(st, lane, nullptr, nullptr, R);
        uint32_t H[8];
        high(st.nb, H);
        for (int k = st.e0 + lane; k < st.e1; k += 32) {
            uint32_t x = __ldg(exc + k), i = x & 31;
            ((uint8_t*)s2)[(((x >> 5) & 31) * 8 + (i >> 2)) * 4 + (i & 3)] = (uint8_t)(x >> 16);
        }
        __syncwarp();
        uint4* xl = (uint4*)(s2 + 8 * lane);
        uint4 a = xl[0], b = xl[1];
        H[0] ^= a.x, H[1] ^= a.y, H[2] ^= a.z, H[3] ^= a.w, H[4] ^= b.x, H[5] ^= b.y, H[6] ^= b.z, H[7] ^= b.w;
        xl[0] = xl[1] = make_uint4(0u, 0u, 0u, 0u);
        __syncwarp();
        pairs(st.sw, H, R);
    }
    // decode_x on scratch that is not zero (a decoded step's slot of shared memory, the producers' of mma_gemm_big_kernel): zeroed
    // here where the step has the many exceptions that go through it.
    __device__ __forceinline__ void decode_xz(const St& st, int lane, uint32_t* s2, uint32_t R[16]) const {
        if (st.e1 - st.e0 <= xmax) return decode(st, lane, nullptr, nullptr, R);
        ((uint4*)s2)[2 * lane] = ((uint4*)s2)[2 * lane + 1] = make_uint4(0u, 0u, 0u, 0u);
        __syncwarp();
        decode_x(st, lane, s2, R);
    }
    template <int U = 0, bool B = false>  // patch's unrolling; B: the step's first 4 exceptions loaded together (patch_b)
    __device__ __forceinline__ void decode(const St& st, int lane, uint32_t*, const uint32_t*, uint32_t R[16]) const {
        uint32_t H[8];
        high(st.nb, H);
        const uint32_t* e = exc;
        if constexpr (B) patch_b([e](int k) { return __ldg(e + k); }, st.e0, st.e1, lane, H);
        else patch<U>([e](int k) { return __ldg(e + k); }, st.e0, st.e1, lane, H);
        pairs(st.sw, H, R);
    }
    // patch with the step's first 4 entries loaded together, before any is applied: an entry's load is a dependent trip to
    // L2 or memory in the middle of the decode, and patch's loop waits for each before it loads the next.
    template <class At>
    static __device__ __forceinline__ void patch_b(At at, int e0, int e1, int lane, uint32_t H[8]) {
        auto one = [&](uint32_t x) {
            uint32_t i = x & 31, v = (x >> 16 & 0xFFu) << (8 * (i & 3));
            uint32_t w = (int)((x >> 5) & 31) == lane ? i >> 2 : 8u;
#pragma unroll
            for (int q = 0; q < 8; q++) H[q] ^= w == (uint32_t)q ? v : 0u;
        };
        int n = e1 - e0;
        uint32_t x0 = n > 0 ? at(e0) : 0u, x1 = n > 1 ? at(e0 + 1) : 0u, x2 = n > 2 ? at(e0 + 2) : 0u, x3 = n > 3 ? at(e0 + 3) : 0u;
        if (n > 0) one(x0);
        if (n > 1) one(x1);
        if (n > 2) one(x2);
        if (n > 3) one(x3);
        for (int k = e0 + 4; k < e1; k++) one(at(k));
    }
    // Ahead: step's 12 lines of 128 bytes (data, a step 1536 bytes: lane i takes line i) to L2, where the loads
    // that follow find them.
    __device__ __forceinline__ void prefetch(int64_t step, int lane) const {
        if (lane < (int)(STEP12 / 128)) pf_l2(data + step * STEP12 + lane * 128);
    }
    // The lines of the exceptions of steps [a, b) (one run: entries exc_base[a] to exc_base[b]) to L2.
    __device__ __forceinline__ void prefetch_exc(int64_t a, int64_t b, int lane) const {
        int64_t lo = (int64_t)__ldg(exc_base + a) * 4, hi = (int64_t)__ldg(exc_base + b) * 4;
        for (int64_t p = (lo & ~127ll) + 128 * lane; p < hi; p += 128 * 32) pf_l2((const uint8_t*)exc + p);
    }
};

__device__ __forceinline__ void mma16816(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
#if defined(__HIP_PLATFORM_AMD__)
    glyd_mma16816(c, a, b0, b1);  // (warp_emu.h: shuffles and FMAs)
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

// The mma layout back to bf16, rows [row0, row0 + rows) of W (multiples of
// 64) into out [rows, K] (checks, and the many-token path that multiplies
// with PyTorch): a warp a step; FEW (a decode ahead, a few warps an SM),
// each warp its step and on by the grid's warps. MOE (exact, a mixture of
// experts' layer, W its experts' matrices of `rows` rows stacked):
// blockIdx.y a hit expert of plan (moe_route's; past those hit: nothing to
// do), its rows into the same rows of out [E rows, K], the rest of out left
// as it is. The 12-bit layout's steps: a warp a step on Hopper by
// Nib::load_decode (the low bytes first), elsewhere and FEW by load() (all at
// once), each as it measured fastest (Hopper measured on a GH200 alone; any
// other GPU not measured, as it was).
template <class Fmt, bool MOE = false, bool FEW = false>
__device__ __forceinline__ void mma_unpack_body(Fmt f, int64_t K, int64_t row0, int64_t rows, uint16_t* __restrict__ out, const int* __restrict__ plan) {
    if constexpr (MOE) {
        if ((int)blockIdx.y >= __ldg(plan)) return;  // the whole block: no expert this far down the hits
        row0 = (int64_t)__ldg(plan + 1 + blockIdx.y) * rows;
        out += row0 * K;
    }
    __shared__ uint32_t tab[Fmt::kTable ? 256 : 1];
    if constexpr (Fmt::kTable) fill_groups(tab);
    __syncthreads();
    int64_t KS = K / 16, total = rows / 64 * KS;
    int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    __shared__ uint4 scratch[8][Fmt::kTable ? S2_BYTES / 16 : 1];
    uint32_t* out32 = (uint32_t*)out;
    for (int64_t local = (blockIdx.x * (int64_t)blockDim.x + threadIdx.x) >> 5; local < total; local += (int64_t)gridDim.x * blockDim.x >> 5) {
        int64_t step = row0 / 64 * KS + local, rb, s;
        uint32_t R[16];
#if __CUDA_ARCH__ == 900
        constexpr bool low_first = std::is_same_v<Fmt, Nib> && !FEW;
#else
        constexpr bool low_first = false;
#endif
        if constexpr (low_first) {
            f.load_decode(step, lane, R, [&] { rb = local / KS, s = local % KS; });
        } else {
            rb = local / KS, s = local % KS;
            typename Fmt::St st;
            f.load(st, step, lane);
            f.decode(st, lane, (uint32_t*)scratch[threadIdx.x >> 5], tab, R);
        }
        // R[2n]: row 8n + g, columns 2t and 2t + 1; R[2n + 1]: columns 8 + 2t, 9 + 2t.
#pragma unroll
        for (int n = 0; n < 8; n++) {
            int64_t at2 = ((rb * 64 + 8 * n + g) * K + s * 16 + 2 * t) / 2;
            out32[at2] = R[2 * n];
            out32[at2 + 4] = R[2 * n + 1];
        }
        if constexpr (!FEW) break;
    }
}

// Rows of W [., K] by id (an embedding's lookup; the ids on the device, so a batch's lookup is one launch whatever they
// are, and a CUDA graph can hold it) into out [n, K]: a warp a (row, step), the step decoded as mma_unpack_kernel decodes
// it, and only the picked row's 16 weights written. Row r of its block of 64 is row 8 m + g of the step (r = 8 m + g), in
// the lanes 4 g to 4 g + 3: R[2 m] and R[2 m + 1] of each, its columns 2 t, 2 t + 1 and 8 + 2 t, 9 + 2 t. An id outside
// [0, rows) has no row: zeros (nothing is read).
template <class Fmt>
__device__ __forceinline__ void mma_rows_body(Fmt f, int64_t K, int64_t rows, const int64_t* __restrict__ ids, int64_t n, uint16_t* __restrict__ out) {
    __shared__ uint32_t tab[Fmt::kTable ? 256 : 1];
    if constexpr (Fmt::kTable) fill_groups(tab);
    __syncthreads();
    __shared__ uint4 scratch[8][Fmt::kTable ? S2_BYTES / 16 : 1];
    int64_t KS = K / 16, unit = (blockIdx.x * (int64_t)blockDim.x + threadIdx.x) >> 5;
    if (unit >= n * KS) return;  // (a whole warp: a unit is a warp's)
    int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    int64_t row = unit / KS, s = unit % KS, id = (int64_t)__ldg((const long long*)ids + row);
    uint32_t* o = (uint32_t*)out + (row * K + s * 16) / 2;
    if ((uint64_t)id >= (uint64_t)rows) {
        if (lane < 8) o[lane] = 0;
        return;
    }
    int r = (int)(id % 64), pick = r >> 3;
    uint32_t R[16];
    typename Fmt::St st;
    f.load(st, id / 64 * KS + s, lane);
    f.decode(st, lane, (uint32_t*)scratch[threadIdx.x >> 5], tab, R);
    if (g == (r & 7)) {
        uint32_t a = 0, b = 0;
#pragma unroll
        for (int m = 0; m < 8; m++) {
            if (m == pick) {
                a = R[2 * m];
                b = R[2 * m + 1];
            }
        }
        o[t] = a;
        o[4 + t] = b;
    }
}

template <class Fmt>
__global__ void mma_rows_kernel(Fmt f, int64_t K, int64_t rows, const int64_t* __restrict__ ids, int64_t n, uint16_t* __restrict__ out) {
    mma_rows_body<Fmt>(f, K, rows, ids, n, out);
}

// A mixture-of-experts layer: its E experts' matrices [O, K] stacked as one matrix [E O, K] in either mma
// layout (expert e's rows e O to e O + O - 1; O a multiple of 64), each multiplied by the tokens routed to it.
// A token's k choices are its pairs (pair j: token j / k, choice j mod k; ids[j] its expert). moe_route_kernel
// sorts the pairs by expert into a plan (int32): [0] the experts hit, then E entries: those experts in order,
// then E + 1: where each one's pairs start in the order (the last: the pairs routed), then P: the pairs by
// expert, in order within each (a pair whose id is outside 0 to E - 1 is routed nowhere). One warp: the counts
// and the order a chunk of 32 pairs at a time, the lanes with the same expert counted by __match_any_sync.
__device__ __forceinline__ void moe_route_body(const int64_t* __restrict__ ids, int64_t P, int E, int* __restrict__ plan) {
    extern __shared__ int at[];  // an expert's pairs, then where its next one goes in the order
    int lane = threadIdx.x;
    uint32_t below = (1u << lane) - 1;
    int *hit = plan + 1, *start = plan + 1 + E, *order = plan + 2 + 2 * E;
    for (int e = lane; e < E; e += 32) at[e] = 0;
    __syncwarp();
    auto expert = [&](int64_t j) {
        int64_t e = j < P ? ids[j] : -1;
        return e >= 0 && e < E ? (int)e : -1;
    };
    for (int64_t b = 0; b < P; b += 32) {
        int e = expert(b + lane);
        uint32_t same = __match_any_sync(FULL, e);
        if (e >= 0 && !(same & below)) at[e] += __popc(same);  // the group's first lane: distinct experts, no race
        __syncwarp();
    }
    int n = 0, h = 0;
    for (int e0 = 0; e0 < E; e0 += 32) {
        int e = e0 + lane, c = e < E ? at[e] : 0, s = c;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            int v = __shfl_up_sync(FULL, s, o);
            s += lane >= o ? v : 0;
        }
        uint32_t hits = __ballot_sync(FULL, c > 0);
        if (c > 0) {
            hit[h + __popc(hits & below)] = e;
            start[h + __popc(hits & below)] = n + s - c;
        }
        if (e < E) at[e] = n + s - c;
        n += __shfl_sync(FULL, s, 31);
        h += __popc(hits);
    }
    if (lane == 0) {
        plan[0] = h;
        start[h] = n;
    }
    __syncwarp();
    for (int64_t b = 0; b < P; b += 32) {
        int e = expert(b + lane);
        uint32_t same = __match_any_sync(FULL, e);
        if (e >= 0) order[at[e] + __popc(same & below)] = (int)(b + lane);
        __syncwarp();
        if (e >= 0 && !(same & below)) at[e] += __popc(same);
        __syncwarp();
    }
}

// The same plan, bit for bit, from one block of NW warps (a prompt's thousands of pairs: the one warp above takes a global
// load's latency a group of 32, twice, 24 us for 1,856 pairs and 0.85 ms for 65,536). Warp w takes a run of whole groups
// of 32 pairs, in order; counts a group at a time (cnt[w][e], ROUTE_R groups' ids loaded at once), then each expert's counts
// are added across the warps in order (cnt becomes the pairs of the warps before), one warp scans the experts as above, and
// each warp writes its pairs at start[e] + the warps' before + the group's rank: the stable order of the one-warp kernel.
constexpr int ROUTE_R = 8;
__device__ __forceinline__ void moe_route_block_body(const int64_t* __restrict__ ids, int64_t P, int E, int* __restrict__ plan) {
    extern __shared__ int rs[];  // cnt [NW][E] (a warp's pairs of an expert, then where its next one goes), tot [E] (the expert's, then its start)
    const int nw = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int* cnt = rs;
    int* tot = rs + nw * E;
    int* mine = cnt + warp * E;
    const uint32_t below = (1u << lane) - 1;
    int *hit = plan + 1, *start = plan + 1 + E, *order = plan + 2 + 2 * E;
    for (int i = threadIdx.x; i < nw * E; i += blockDim.x) cnt[i] = 0;
    const int64_t groups = (P + 31) / 32, per = (groups + nw - 1) / nw;
    const int64_t g0 = min(groups, (int64_t)warp * per), g1 = min(groups, g0 + per);
    __syncthreads();
    auto expert = [&](int64_t j) {
        int64_t e = j < P ? __ldg(ids + j) : -1;
        return e >= 0 && e < E ? (int)e : -1;
    };
    for (int64_t g = g0; g < g1; g += ROUTE_R) {
        int e[ROUTE_R];
#pragma unroll
        for (int r = 0; r < ROUTE_R; r++) e[r] = g + r < g1 ? expert((g + r) * 32 + lane) : -1;
#pragma unroll
        for (int r = 0; r < ROUTE_R; r++) {
            uint32_t same = __match_any_sync(FULL, e[r]);
            if (e[r] >= 0 && !(same & below)) mine[e[r]] += __popc(same);  // the group's first lane: distinct experts, no race
            __syncwarp();
        }
    }
    __syncthreads();
    for (int x = threadIdx.x; x < E; x += blockDim.x) {
        int a = 0;
        for (int w = 0; w < nw; w++) {
            int c = cnt[w * E + x];
            cnt[w * E + x] = a;
            a += c;
        }
        tot[x] = a;
    }
    __syncthreads();
    if (warp == 0) {
        int n = 0, h = 0;
        for (int e0 = 0; e0 < E; e0 += 32) {
            int e = e0 + lane, c = e < E ? tot[e] : 0, s = c;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                int v = __shfl_up_sync(FULL, s, o);
                s += lane >= o ? v : 0;
            }
            uint32_t hits = __ballot_sync(FULL, c > 0);
            if (c > 0) {
                hit[h + __popc(hits & below)] = e;
                start[h + __popc(hits & below)] = n + s - c;
            }
            if (e < E) tot[e] = n + s - c;
            n += __shfl_sync(FULL, s, 31);
            h += __popc(hits);
        }
        if (lane == 0) {
            plan[0] = h;
            start[h] = n;
        }
    }
    __syncthreads();
    for (int64_t g = g0; g < g1; g += ROUTE_R) {
        int e[ROUTE_R];
#pragma unroll
        for (int r = 0; r < ROUTE_R; r++) e[r] = g + r < g1 ? expert((g + r) * 32 + lane) : -1;
#pragma unroll
        for (int r = 0; r < ROUTE_R; r++) {
            uint32_t same = __match_any_sync(FULL, e[r]);
            if (e[r] >= 0) order[tot[e[r]] + mine[e[r]] + __popc(same & below)] = (int)((g + r) * 32 + lane);
            __syncwarp();
            if (e[r] >= 0 && !(same & below)) mine[e[r]] += __popc(same);
            __syncwarp();
        }
    }
}

// Y = X W^T for up to 16 MT tokens from the mma layout. The W steps
// (64 rows by 16 columns), in row-block order, are split evenly over the
// blocks (stream-K: no block is left for a second wave); a block's run, cut
// where a row block ends, is split over its 8 warps, whose sums are added in
// shared memory. A row block covered by several blocks: each writes its
// part to a slot, and the last to finish adds the parts in block order (the
// same result every run). A warp's step: 1024 weights decoded into B
// fragments, 8 MT tensor-core products. The next step's loads are issued
// before this step's decode.
__device__ __forceinline__ int64_t block_of_step(int64_t x, int64_t nb, int64_t total) {
    return ((x + 1) * nb - 1) / total;  // block b runs steps [b total / nb, (b + 1) total / nb)
}

// A warp's steps s0 to s1 of a row block (W's steps base + s): acc += X's rows by the step's 64 rows, 16 MT tokens
// (xr: this thread's rows g and g + 8 of each m-tile, at X + row K + 2t; null past the tokens). The next step's
// loads, W's and then the inputs, are issued before this step's decode. pf: the lines of the steps pf ahead of the next
// one are asked of L2 meanwhile (and the warp's exceptions at the start), so that the loads find them there and not in
// memory; PF: only where compiled in (a register and a branch a step otherwise), 0: not. SK: na, the m-tiles with tokens in them, the first na of MT (the others' loads and products are not made).
// mma_gemm_kernel's and mma_moe_kernel's.
template <class Fmt, int MT, bool X = false, int U = 0, bool B = false, bool SK = false, bool PF = false, int D = 1>  // X: decode_x (the 12-bit layout's heavy blocks); U: Nib::patch's unrolling; B: Nib::patch_b; SK: only the m-tiles below na; PF: the L2 prefetch (pf ahead); D: the steps loaded ahead (2: two steps' loads in flight a warp, for HBM GPUs' one-token products, which waited on one at a time)
__device__ __forceinline__ void mma_steps(const Fmt& f, int64_t base, int64_t s0, int64_t s1, const __nv_bfloat16* const (&xr)[MT][2], int lane, uint32_t* s2, const uint32_t* tab, float (&acc)[MT][8][4], int pf = 0, int na = MT) {
    typename Fmt::St st[D];
    uint32_t a[D][MT][4];
    auto load = [&](int d, int64_t s) {
        f.load(st[d], base + s, lane);
#pragma unroll
        for (int mt = 0; mt < MT; mt++) {
            if (!SK || MT == 1 || mt < na) {
                a[d][mt][0] = xr[mt][0] ? *(const uint32_t*)(xr[mt][0] + s * 16) : 0u;
                a[d][mt][2] = xr[mt][0] ? *(const uint32_t*)(xr[mt][0] + s * 16 + 8) : 0u;
                a[d][mt][1] = xr[mt][1] ? *(const uint32_t*)(xr[mt][1] + s * 16) : 0u;
                a[d][mt][3] = xr[mt][1] ? *(const uint32_t*)(xr[mt][1] + s * 16 + 8) : 0u;
            }
        }
    };
    if constexpr (PF)
        for (int j = 1; j <= pf && s0 + j < s1; j++) f.prefetch(base + s0 + j, lane);
#pragma unroll
    for (int d = 0; d < D; d++)
        if (s0 + d < s1) load(d, s0 + d);
    if constexpr (PF)
        if (pf && s0 < s1) f.prefetch_exc(base + s0, base + s1, lane);
    for (int64_t s = s0; s < s1; s += D) {
#pragma unroll
        for (int d = 0; d < D; d++) {
            const int64_t sd = s + d;
            if (sd >= s1) break;
            typename Fmt::St cur = st[d];
            uint32_t ca[MT][4];
#pragma unroll
            for (int mt = 0; mt < MT; mt++)
#pragma unroll
                for (int q = 0; q < 4; q++) ca[mt][q] = a[d][mt][q];
            if constexpr (PF)
                if (pf && sd + 1 + pf < s1) f.prefetch(base + sd + 1 + pf, lane);
            if (sd + D < s1) load(d, sd + D);  // (the slot's next: D steps ahead)
            uint32_t R[16];
            if constexpr (X) f.decode_x(cur, lane, s2, R);
            else if constexpr (std::is_same_v<Fmt, Nib>) f.template decode<U, B>(cur, lane, s2, tab, R);
            else f.decode(cur, lane, s2, tab, R);
#pragma unroll
            for (int mt = 0; mt < MT; mt++)
                if (!SK || MT == 1 || mt < na)
#pragma unroll
                    for (int nn = 0; nn < 8; nn++) mma16816(acc[mt][nn], ca[mt], R[2 * nn], R[2 * nn + 1]);
        }
    }
}

// The block layouts' one-token steps through shared memory (mma_moe_kernel's V & 16, an HBM GPU): a warp's next P - 1 steps (their bytes as Fmt::chunk gives them)
// copied by cp.async into its ring of P slots, so that the steps in flight take no registers; a slot read back with Fmt::load_s and decoded as mma_steps does. The
// activations' words two steps ahead in registers (from L2; a slot of them had made the rings too big for 3 blocks an SM on an A100).
template <class F, class = void>
struct has_stage : std::false_type {};
template <class F>
struct has_stage<F, std::void_t<decltype(F::kStage)>> : std::true_type {};
template <class Fmt>
constexpr int ring_slot() {
    if constexpr (has_stage<Fmt>::value) return Fmt::kStage;
    else return 0;
}
#if defined(__HIP_PLATFORM_AMD__)  // (AMD: cp_async16 copies when asked, so there is nothing to wait for)
template <int N>
__device__ __forceinline__ void cp_wait() {}
__device__ __forceinline__ void cp_commit() {}
#else
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
#endif
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int bytes);
template <class Fmt, int P>
__device__ __forceinline__ void mma_steps_staged(const Fmt& f, int64_t base, int64_t s0, int64_t s1, const __nv_bfloat16* const (&xr)[1][2], int lane, uint8_t* ring, const uint32_t* tab, float (&acc)[1][8][4]) {
    constexpr int SLOT = ring_slot<Fmt>(), CH = Fmt::kStage / 16;
    const uint32_t r0 = (uint32_t)__cvta_generic_to_shared(ring);
    const __nv_bfloat16* x0 = xr[0][0];
    const __nv_bfloat16* x1 = xr[0][1];
    auto stage = [&](int j, int64_t s) {
        const uint32_t d = r0 + j * SLOT;
#pragma unroll
        for (int i = 0; i < (CH + 31) / 32; i++) {
            const int c = lane + 32 * i;
            if (c < CH) cp_async16(d + c * 16, f.chunk(base + s, c), 16);
        }
    };
    // (mma_steps' a: row g's words at columns 2t and 8 + 2t, then row g + 8's)
    auto loadx = [&](int64_t s, uint32_t (&a)[4]) {
        a[0] = x0 ? *(const uint32_t*)(x0 + s * 16) : 0u;
        a[2] = x0 ? *(const uint32_t*)(x0 + s * 16 + 8) : 0u;
        a[1] = x1 ? *(const uint32_t*)(x1 + s * 16) : 0u;
        a[3] = x1 ? *(const uint32_t*)(x1 + s * 16 + 8) : 0u;
    };
    uint32_t a1[4] = {0u, 0u, 0u, 0u}, a2[4] = {0u, 0u, 0u, 0u};
    if (s0 < s1) loadx(s0, a1);
    if (s0 + 1 < s1) loadx(s0 + 1, a2);
#pragma unroll
    for (int j = 0; j < P - 1; j++) {
        if (s0 + j < s1) stage(j, s0 + j);
        cp_commit();
    }
    for (int64_t s = s0; s < s1; s++) {
        const int j = (int)((s - s0) & (P - 1));
        uint32_t ca[4] = {a1[0], a1[1], a1[2], a1[3]};
#pragma unroll
        for (int q = 0; q < 4; q++) a1[q] = a2[q];
        if (s + 2 < s1) loadx(s + 2, a2);
        __syncwarp();  // (every lane done with step s - 1's slot, which the next copy fills)
        if (s + P - 1 < s1) stage((j + P - 1) & (P - 1), s + P - 1);
        cp_commit();
        cp_wait<P - 1>();
        __syncwarp();  // (step s's slot in: every lane's copies)
        typename Fmt::St cur;
        f.load_s(cur, ring + j * SLOT, lane);
        uint32_t R[16];
        f.decode(cur, lane, nullptr, tab, R);
#pragma unroll
        for (int nn = 0; nn < 8; nn++) mma16816(acc[0][nn], ca, R[2 * nn], R[2 * nn + 1]);
    }
}

// The block layouts' one-token steps through shared memory, S steps a stage (research, 2026-10-20): a warp's ring is two groups of S slots; a stage copies the
// group after next's S steps in one go (one commit, one wait and two warp syncs a group instead of a step), and the group's S decodes follow one another.
// The sums as mma_steps_staged's, step by step in the same order.
template <class Fmt, int S>
__device__ __forceinline__ void mma_steps_staged_s(const Fmt& f, int64_t base, int64_t s0, int64_t s1, const __nv_bfloat16* const (&xr)[1][2], int lane, uint8_t* ring, const uint32_t* tab, float (&acc)[1][8][4]) {
    constexpr int SLOT = ring_slot<Fmt>(), CH = Fmt::kStage / 16, GROUP = S * SLOT;
    const uint32_t r0 = (uint32_t)__cvta_generic_to_shared(ring);
    const __nv_bfloat16* x0 = xr[0][0];
    const __nv_bfloat16* x1 = xr[0][1];
    auto stage = [&](int grp, int64_t s) {  // steps s to s + S - 1 (those before s1) into group grp
        const uint32_t d = r0 + grp * GROUP;
#pragma unroll
        for (int i = 0; i < (S * CH + 31) / 32; i++) {
            const int c = lane + 32 * i;
            if (c < S * CH) {
                const int st = c / CH, cc = c - st * CH;
                if (s + st < s1) cp_async16(d + st * SLOT + cc * 16, f.chunk(base + s + st, cc), 16);
            }
        }
    };
    auto loadx = [&](int64_t s, uint32_t (&a)[4]) {
        a[0] = x0 ? *(const uint32_t*)(x0 + s * 16) : 0u;
        a[2] = x0 ? *(const uint32_t*)(x0 + s * 16 + 8) : 0u;
        a[1] = x1 ? *(const uint32_t*)(x1 + s * 16) : 0u;
        a[3] = x1 ? *(const uint32_t*)(x1 + s * 16 + 8) : 0u;
    };
    uint32_t a1[4] = {0u, 0u, 0u, 0u}, a2[4] = {0u, 0u, 0u, 0u};
    if (s0 < s1) loadx(s0, a1);
    if (s0 + 1 < s1) loadx(s0 + 1, a2);
    stage(0, s0);
    cp_commit();
    if (s0 + S < s1) stage(1, s0 + S);
    cp_commit();
    int grp = 0;
    for (int64_t s = s0; s < s1; s += S) {
        cp_wait<1>();  // this group's steps in (the next group's may still be coming)
        __syncwarp();
#pragma unroll
        for (int st = 0; st < S; st++) {
            if (s + st >= s1) break;
            uint32_t ca[4] = {a1[0], a1[1], a1[2], a1[3]};
#pragma unroll
            for (int q = 0; q < 4; q++) a1[q] = a2[q];
            if (s + st + 2 < s1) loadx(s + st + 2, a2);
            typename Fmt::St cur;
            f.load_s(cur, ring + grp * GROUP + st * SLOT, lane);
            uint32_t R[16];
            f.decode(cur, lane, nullptr, tab, R);
#pragma unroll
            for (int nn = 0; nn < 8; nn++) mma16816(acc[0][nn], ca, R[2 * nn], R[2 * nn + 1]);
        }
        __syncwarp();  // every lane done with the group: the group after next takes it
        if (s + 2 * S < s1) stage(grp, s + 2 * S);
        cp_commit();
        grp ^= 1;
    }
}

// The 8 warps' sums (C fragment rows g and g + 8, columns 2t and 2t + 1 of each n-tile) into shared memory, red
// [4][16 MT][65]: warps 4-7 put theirs, 0-3 add them to theirs and put those; the caller adds the four.
template <int MT>
__device__ __forceinline__ void warp_sums(float (&acc)[MT][8][4], float* red, int warp, int g, int t) {
    float* mine = red + (int64_t)(warp & 3) * MT * 16 * 65;
    for (int half = 1; half >= 0; half--) {
        if ((warp >> 2) == half)
#pragma unroll
            for (int mt = 0; mt < MT; mt++)
#pragma unroll
                for (int nn = 0; nn < 8; nn++) {
                    float* r0 = mine + (mt * 16 + g) * 65 + nn * 8 + t * 2;
                    float* r1 = r0 + 8 * 65;
                    if (!half) {
                        acc[mt][nn][0] += r0[0];
                        acc[mt][nn][1] += r0[1];
                        acc[mt][nn][2] += r1[0];
                        acc[mt][nn][3] += r1[1];
                    }
                    r0[0] = acc[mt][nn][0];
                    r0[1] = acc[mt][nn][1];
                    r1[0] = acc[mt][nn][2];
                    r1[1] = acc[mt][nn][3];
                }
        __syncthreads();
    }
}

// A block has written its part of a unit: whether it is the last of the unit's n blocks to finish (done: the unit's
// counter, zero before, reset by the last for the next product), every other part visible to it then (last: the
// block's shared flag).
__device__ __forceinline__ bool last_of(int* done, int64_t n, int& last) {
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        last = atomicAdd(done, 1) == n - 1;
        if (last) *done = 0;  // ready for the next product
    }
    __syncthreads();
    if (last) __threadfence();
    return last;
}

// A one-token product's blocks an SM (mma_gemm_kernel, MT 1): 3 on Ada (sm_89), measured on an RTX 4080 SUPER (Gemma 4 12B wren, a step 12.12 -> 11.96 ms;
// 4 is slower); 2 elsewhere until measured there (GLY-312). mma_gemm_mt sizes its grid from the occupancy the build gives.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 890
#define GEMM1_BLOCKS 3
#else
#define GEMM1_BLOCKS 2
#endif
template <class Fmt, int MT, bool B = false, bool PF = false>
__device__ __forceinline__ void mma_gemm_body(Fmt f, int64_t O, int64_t K, int64_t M, const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ Y, float* __restrict__ parts, int* __restrict__ done, int pf) {
    extern __shared__ __align__(128) float red[];  // [4 warps][16 MT rows][65], then (tiered) the warps' scratch [8][S2_BYTES]
    __shared__ int last;
    __shared__ uint32_t tab[Fmt::kTable ? 256 : 1];
    if constexpr (Fmt::kTable) fill_groups(tab);
    __syncthreads();
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    uint32_t* s2 = (uint32_t*)(red + 4 * MT * 16 * 65) + warp * (S2_BYTES / 4);
    int64_t KS = K / 16, total = O / 64 * KS, nb = gridDim.x;
    int64_t B1 = (blockIdx.x + 1) * total / nb;
    // 12-bit: where the block's steps have more than an exception each on average (a few layers' matrices: 4-5),
    // they go through the warp's scratch, 1 KB of red (zero between steps; the warps' sums take red only after
    // every warp's steps), with decode_x; elsewhere decode as before (its code with decode_x's in it, or more
    // shared memory a block, had cost the other layers 1%).
    bool heavy = false;
    if constexpr (!Fmt::kTable) {
        int64_t B0 = blockIdx.x * total / nb;
        heavy = __ldg(f.exc_base + B1) - __ldg(f.exc_base + B0) > B1 - B0;
        s2 = heavy ? (uint32_t*)red + 256 * warp : nullptr;
    }
    for (int64_t seg = blockIdx.x * total / nb; seg < B1;) {
        int64_t rb = seg / KS, sb = seg - rb * KS, se = min(B1, (rb + 1) * KS) - rb * KS;
        seg = rb * KS + se;
        int64_t per = (se - sb + 7) / 8;
        int64_t s0 = min(se, sb + warp * per), s1 = min(se, s0 + per);
        float acc[MT][8][4];
#pragma unroll
        for (int mt = 0; mt < MT; mt++)
#pragma unroll
            for (int nn = 0; nn < 8; nn++)
#pragma unroll
                for (int q = 0; q < 4; q++) acc[mt][nn][q] = 0.f;
        const __nv_bfloat16* xr[MT][2];  // this thread's rows of X (none past M)
#pragma unroll
        for (int mt = 0; mt < MT; mt++)
#pragma unroll
            for (int hh = 0; hh < 2; hh++) {
                int64_t r = mt * 16 + g + 8 * hh;
                xr[mt][hh] = r < M ? X + r * K + t * 2 : nullptr;
            }
        if (!Fmt::kTable && heavy) {
            ((uint4*)s2)[2 * lane] = ((uint4*)s2)[2 * lane + 1] = make_uint4(0u, 0u, 0u, 0u);
            __syncwarp();
            mma_steps<Fmt, MT, !Fmt::kTable, 0, B, false, PF>(f, rb * KS, s0, s1, xr, lane, s2, tab, acc, pf);
            __syncthreads();  // every warp's scratch done with
        } else {
            mma_steps<Fmt, MT, false, 0, B, false, PF>(f, rb * KS, s0, s1, xr, lane, s2, tab, acc, pf);
        }
        warp_sums<MT>(acc, red, warp, g, t);
        int64_t first = block_of_step(rb * KS, nb, total), fin = block_of_step(rb * KS + KS - 1, nb, total);
        for (int i = threadIdx.x; i < MT * 16 * 64; i += 256) {
            int r = i / 64, c = i % 64;
            if (r >= M) continue;
            float v = 0.f;
#pragma unroll
            for (int ww = 0; ww < 4; ww++) v += red[(ww * MT * 16 + r) * 65 + c];
            if (first == fin) Y[r * O + rb * 64 + c] = __float2bfloat16(v + (bias ? __bfloat162float(bias[rb * 64 + c]) : 0.f));
            else parts[((blockIdx.x + rb) * M + r) * 64 + c] = v;  // slot b + rb: distinct for every (block, row block) pair
        }
        if (first != fin && last_of(done + rb, fin - first + 1, last))
            for (int i = threadIdx.x; i < M * 64; i += 256) {
                int r = i / 64, c = i % 64;
                float v = 0.f;
                for (int64_t b = first; b <= fin; b++) v += __ldcg(parts + ((b + rb) * M + r) * 64 + c);
                Y[r * O + rb * 64 + c] = __float2bfloat16(v + (bias ? __bfloat162float(bias[rb * 64 + c]) : 0.f));
            }
        __syncthreads();  // red and last are reused
    }
}

template <class Fmt, int MT, bool B = false, bool PF = false>
__global__ void __launch_bounds__(256, MT == 1 ? GEMM1_BLOCKS : 1) mma_gemm_kernel(Fmt f, int64_t O, int64_t K, int64_t M, const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ Y, float* __restrict__ parts, int* __restrict__ done, int pf) {
    mma_gemm_body<Fmt, MT, B, PF>(f, O, K, M, X, bias, Y, parts, done, pf);
}

// The tiered layout's product for 1 to 16 tokens (a generation step). mma_gemm_kernel's split of the steps over blocks and its
// last-to-arrive sum of a row block's parts, in block order, with blocks of 4 warps (their barriers, and the sums at their ends, are fewer
// and cheaper than 8 warps'); a warp's step decoded by the short form of `exponents` where the packer's flag (bit 31 of block_base[step])
// is clear, which says every check of it holds, and else by `exponents` itself, out of line; and up to 8 tokens with W's registers as the
// A operand of the tensor cores' product and X's as the B operand (16 accumulators a lane for 32, two loads of X a step for four).
//
// A step's loads: its fixed part's words, the block's first word a lane (win) and its last words (tail, tail1), and block_base's two
// entries as they are (rb's bit 31: the flag).
struct StepH {
    uint32_t pd[2], sw[8], win, tail, tail1, rb, re;
};

__device__ __forceinline__ void load_step_h(StepH& st, const uint8_t* __restrict__ data, const uint8_t* __restrict__ blocks, const int32_t* __restrict__ block_base, int64_t step, int lane) {
    const uint32_t* cp = (const uint32_t*)(data + step * STEP_BYTES) + lane;
    st.pd[0] = __ldg(cp);
    st.pd[1] = __ldg(cp + 32);
    const uint4* sp = (const uint4*)(data + step * STEP_BYTES + 256) + lane;
    uint4 x0 = __ldg(sp), x1 = __ldg(sp + 32);
    st.sw[0] = x0.x; st.sw[1] = x0.y; st.sw[2] = x0.z; st.sw[3] = x0.w;
    st.sw[4] = x1.x; st.sw[5] = x1.y; st.sw[6] = x1.z; st.sw[7] = x1.w;
    st.rb = (uint32_t)__ldg(block_base + step);
    st.re = (uint32_t)__ldg(block_base + step + 1);
    const uint32_t* bw = (const uint32_t*)blocks;
    uint32_t b = (st.rb & 0x7FFFFFFFu) >> 2, e = (st.re & 0x7FFFFFFFu) >> 2;
    st.win = __ldg(bw + b + lane);
    st.tail = __ldg(bw + e - 1 - lane);
    st.tail1 = __ldg(bw + e);
}

// The short form of `exponents` (the flag clear: at most 512 tier-1 escapes, the block's tier-3 digits and its list, of at most 32
// literals, within the 124 bytes of its window): the same exponents, no checks and no other paths.
__device__ __forceinline__ void exponents_h(const StepH& st, Tiers ts, int lane, uint32_t* s2, const uint32_t* tab, uint32_t ew[8]) {
    uint32_t e1[8];
#pragma unroll
    for (int q = 0; q < 8; q++) e1[q] = tab[(st.pd[q >> 2] >> (8 * (q & 3))) & 0xFF];
    uint32_t n1 = __popc(escapes(st.pd[0])) + __popc(escapes(st.pd[1]));
    uint32_t T1 = __reduce_add_sync(FULL, n1), words = (T1 + 15) >> 4;
    const uint32_t S3 = 0x01010101u * (ts.s3 >> 24);
    const uint32_t o = 8 * (st.rb & 3u);
    uint32_t up = __shfl_up_sync(FULL, st.tail, 1);
    uint32_t da = (uint32_t)lane < words ? __funnelshift_r(st.tail, lane ? up : st.tail1, 8 * (st.re & 3u)) : 0u;
    uint32_t na = __popc(escapes(da)), tot;
    uint32_t pre = scan(n1 | na << 11, lane, tot);
    uint32_t b1 = pre & 0x7FF, T2 = tot >> 11;
    uint32_t d3 = win_bits(st.win, o + 2 * (pre >> 11));
    uint32_t g3 = (__reduce_max_sync(FULL, na) + 3) >> 2;
    uint32_t y[4] = {0u, 0u, 0u, 0u}, q4[4];
#pragma unroll
    for (int q = 0; q < 4; q++)
        if (q < (int)g3) y[q] = __byte_perm(ts.s3, S3, tab[(d3 >> (8 * q)) & 0xFF]);
#pragma unroll
    for (int q = 0; q < 4; q++) q4[q] = group(da >> (8 * q), ts.s2, y, tab);
    ((uint4*)s2)[lane] = make_uint4(q4[0], q4[1], q4[2], q4[3]);
    uint32_t r0 = (2 * T2 + 7) >> 3, nl = list_entries(((st.re - st.rb) & 0x7FFFFFFFu) - r0 - 4 * words);
    __syncwarp();
    uint32_t x = win_bits(st.win, o + 8 * r0 + 18 * (uint32_t)lane);  // (every lane's: the shuffles are the warp's; a lane past the list's end has no entry)
    if ((uint32_t)lane < nl) ((uint8_t*)s2)[x & 0x3FF] = (uint8_t)(x >> 10);
    __syncwarp();
    uint32_t c = b1;
#pragma unroll
    for (int q = 0; q < 8; q++) {
        uint32_t e = e1[q];
        ew[q] = __byte_perm(ts.s1, __funnelshift_r(s2[c >> 2], s2[(c >> 2) + 1], c << 3), e);
        c += e >> 16;
    }
    __syncwarp();  // the scratch is read
}

// A step the short form cannot take: `exponents` on the step by value (the loop's own steps stay in registers: it saves them around
// the call, on this path alone), its 8 words a lane left in eout [8][32].
__device__ __noinline__ void exponents_cold(Step st, Tiers ts, const uint8_t* __restrict__ blocks, int lane, uint32_t* s2, const uint32_t* tab, uint32_t* eout) {
    uint32_t ew[8];
    exponents(st, ts, blocks, lane, s2, tab, ew);
#pragma unroll
    for (int q = 0; q < 8; q++) eout[q * 32 + lane] = ew[q];  // (the scratch is read, exponents' last barrier: eout may be it)
    __syncwarp();
}

__device__ __forceinline__ Step step_of(const StepH& h) {
    Step s;
    s.pd[0] = h.pd[0]; s.pd[1] = h.pd[1];
#pragma unroll
    for (int i = 0; i < 8; i++) s.sw[i] = h.sw[i];
    s.win = h.win; s.tail = h.tail; s.tail1 = h.tail1;
    s.blk = h.rb & 0x7FFFFFFFu;
    s.end = h.re & 0x7FFFFFFFu;
    return s;
}

constexpr int STEP_WARPS = 4, STEP_BLOCKS = 4;  // warps a block, blocks an SM

// A warp's steps s0 to s1 of a row block (W's steps base + s) into acc. SW (up to 8 tokens): 4 products of 16 rows by 16 columns by 8
// tokens, W the A operand (acc[i]: rows 16 i to 16 i + 15 of the row block); else 8 of 16 tokens by 16 columns by 8 rows (acc[n]: rows 8 n
// to 8 n + 7). xr: this lane's row of X, g, and g + 8 (not SW), at X + row K + 2t, null past the tokens.
// A step's loads (and the 4 words of X it takes) are made a step ahead, into one of two sets of registers (the step s is in the first
// where s - s0 is even, else in the second): the step in one set is decoded and multiplied while the next one's loads, issued before
// its decode, fill the other. The flags of the next 32 steps are read at once, by one ballot: the run of steps that have none goes by the
// short form with no test, and the step that has one by `exponents`.
template <bool SW>
__device__ __forceinline__ void step_steps(const Tiered& f, int64_t base, int s0, int s1, const __nv_bfloat16* const (&xr)[2], int lane, uint32_t* s2, const uint32_t* tab, float (&acc)[SW ? 4 : 8][4]) {
    StepH sa, sb;
    uint32_t xa[4] = {0u, 0u, 0u, 0u}, xb[4] = {0u, 0u, 0u, 0u};
    auto load = [&](StepH& st, uint32_t (&x)[4], int s) {
        load_step_h(st, f.data, f.blocks, f.block_base, base + s, lane);
        x[0] = xr[0] ? *(const uint32_t*)(xr[0] + s * 16) : 0u;
        x[2] = xr[0] ? *(const uint32_t*)(xr[0] + s * 16 + 8) : 0u;
        if constexpr (!SW) {
            x[1] = xr[1] ? *(const uint32_t*)(xr[1] + s * 16) : 0u;
            x[3] = xr[1] ? *(const uint32_t*)(xr[1] + s * 16 + 8) : 0u;
        }
    };
    // the step s, in (cur, xc), its successor's loads first into (nxt, xn); cold: the flag is set
    auto step = [&](StepH& cur, const uint32_t (&xc)[4], StepH& nxt, uint32_t (&xn)[4], int s, auto cold) {
        if (s + 1 < s1) load(nxt, xn, s + 1);
        uint32_t ew[8], R[16];
        if constexpr (decltype(cold)::value) {
            exponents_cold(step_of(cur), f.ts, f.blocks, lane, s2, tab, s2);
#pragma unroll
            for (int q = 0; q < 8; q++) ew[q] = s2[q * 32 + lane];
            __syncwarp();
        } else {
            exponents_h(cur, f.ts, lane, s2, tab, ew);
        }
        pairs(cur.sw, ew, R);
        if constexpr (SW) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const uint32_t A4[4] = {R[4 * i], R[4 * i + 2], R[4 * i + 1], R[4 * i + 3]};
                mma16816(acc[i], A4, xc[0], xc[2]);
            }
        } else {
#pragma unroll
            for (int nn = 0; nn < 8; nn++) mma16816(acc[nn], xc, R[2 * nn], R[2 * nn + 1]);
        }
    };
    if (s0 < s1) load(sa, xa, s0);
    for (int s = s0; s < s1;) {
        int n = s1 - s < 32 ? s1 - s : 32;
        uint32_t flags = __ballot_sync(FULL, lane < n && __ldg(f.block_base + base + s + lane) < 0);
        int e = s + (flags ? __ffs(flags) - 1 : n);  // the steps s to e - 1 have none
        if (((s - s0) & 1) && s < e) {
            step(sb, xb, sa, xa, s, std::false_type{});
            s++;
        }
        for (; s < e; s += 2) {
            step(sa, xa, sb, xb, s, std::false_type{});
            if (s + 1 < e) step(sb, xb, sa, xa, s + 1, std::false_type{});
        }
        s = e;
        if (flags) {
            if ((s - s0) & 1) step(sb, xb, sa, xa, s, std::true_type{});
            else step(sa, xa, sb, xb, s, std::true_type{});
            s++;
        }
    }
}

template <bool SW>
__device__ __forceinline__ void step_body(Tiered f, int64_t O, int64_t K, int64_t M, const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ Y, float* __restrict__ parts, int* __restrict__ done) {
    constexpr int NW = STEP_WARPS, ROWS = SW ? 8 : 16;  // warps a block, tokens a pass
    extern __shared__ __align__(128) float red[];  // [NW warps][ROWS][65] sums, then the warps' scratch [NW][S2_BYTES]
    __shared__ int last;
    __shared__ uint32_t tab[256];
    fill_groups(tab);
    __syncthreads();
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    uint32_t* s2 = (uint32_t*)(red + NW * ROWS * 65) + warp * (S2_BYTES / 4);
    int64_t KS = K / 16, total = O / 64 * KS, nb = gridDim.x;
    int64_t B1 = (blockIdx.x + 1) * total / nb;
    for (int64_t seg = blockIdx.x * total / nb; seg < B1;) {
        int64_t rb = seg / KS, sb = seg - rb * KS, se = min(B1, (rb + 1) * KS) - rb * KS;
        seg = rb * KS + se;
        int64_t per = (se - sb + NW - 1) / NW;
        int64_t s0 = min(se, sb + warp * per), s1 = min(se, s0 + per);
        const __nv_bfloat16* xr[2];
        xr[0] = g < M ? X + g * K + t * 2 : nullptr;
        xr[1] = !SW && g + 8 < M ? X + (g + 8) * K + t * 2 : nullptr;
        float acc[SW ? 4 : 8][4];
#pragma unroll
        for (int i = 0; i < (SW ? 4 : 8); i++)
#pragma unroll
            for (int q = 0; q < 4; q++) acc[i][q] = 0.f;
        step_steps<SW>(f, rb * KS, (int)s0, (int)s1, xr, lane, s2, tab, acc);  // (a row block's steps: K / 16 of them, an int)
        float* mine = red + warp * ROWS * 65;  // this warp's sums: C fragment (rows g, g + 8; columns 2t, 2t + 1) of each product
        if constexpr (SW) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
                mine[(2 * t) * 65 + 16 * i + g] = acc[i][0];
                mine[(2 * t + 1) * 65 + 16 * i + g] = acc[i][1];
                mine[(2 * t) * 65 + 16 * i + g + 8] = acc[i][2];
                mine[(2 * t + 1) * 65 + 16 * i + g + 8] = acc[i][3];
            }
        } else {
#pragma unroll
            for (int nn = 0; nn < 8; nn++) {
                float* r0 = mine + g * 65 + nn * 8 + t * 2;
                float* r1 = r0 + 8 * 65;
                r0[0] = acc[nn][0];
                r0[1] = acc[nn][1];
                r1[0] = acc[nn][2];
                r1[1] = acc[nn][3];
            }
        }
        __syncthreads();
        int64_t first = block_of_step(rb * KS, nb, total), fin = block_of_step(rb * KS + KS - 1, nb, total);
        for (int i = threadIdx.x; i < ROWS * 64; i += 32 * NW) {
            int r = i / 64, c = i % 64;
            if (r >= M) continue;
            float v = 0.f;
#pragma unroll
            for (int ww = 0; ww < NW; ww++) v += red[(ww * ROWS + r) * 65 + c];
            if (first == fin) Y[r * O + rb * 64 + c] = __float2bfloat16(v + (bias ? __bfloat162float(bias[rb * 64 + c]) : 0.f));
            else parts[((blockIdx.x + rb) * M + r) * 64 + c] = v;  // slot b + rb: distinct for every (block, row block) pair
        }
        if (first != fin && last_of(done + rb, fin - first + 1, last))
            for (int i = threadIdx.x; i < M * 64; i += 32 * NW) {
                int r = i / 64, c = i % 64;
                float v = 0.f;
                for (int64_t b = first; b <= fin; b++) v += __ldcg(parts + ((b + rb) * M + r) * 64 + c);
                Y[r * O + rb * 64 + c] = __float2bfloat16(v + (bias ? __bfloat162float(bias[rb * 64 + c]) : 0.f));
            }
        __syncthreads();  // red and last are reused
    }
}

template <bool SW>
__global__ void __launch_bounds__(32 * STEP_WARPS, STEP_BLOCKS) step_kernel(Tiered f, int64_t O, int64_t K, int64_t M, const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ Y, float* __restrict__ parts, int* __restrict__ done) {
    step_body<SW>(f, O, K, M, X, bias, Y, parts, done);
}

#if defined(__HIP_PLATFORM_AMD__)
// AMD: the copy made when asked (zeros past `bytes`, 0 or 16), so a commit and a wait have nothing to wait for; ldmatrix by shuffles (warp_emu.h). RDNA has
// no named barriers: the kernels that pass stages between producer and consumer warps by them (the prompt products mma_gemm_big and mma_gemm_sk, the
// A100's and Hopper's) trap there, and the host takes AMD's prompts through the step product instead (mma_linear_run).
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int bytes) { glyd_lds_put16(dst, bytes ? *(const uint4*)src : make_uint4(0, 0, 0, 0)); }
__device__ __forceinline__ void ldmatrix_x4(uint32_t r[4], uint32_t addr) { glyd_ldsm4(r, addr); }
template <int THREADS> __device__ __forceinline__ void bar_sync(int) { __builtin_trap(); }
template <int THREADS> __device__ __forceinline__ void bar_arrive(int) { __builtin_trap(); }
#else
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t r[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

// (the "memory" clobbers: shared and global memory written before a barrier is not kept in registers past it, nor read
// ahead of it)
template <int THREADS> __device__ __forceinline__ void bar_sync(int id) { asm volatile("bar.sync %0, %1;\n" ::"r"(id), "n"(THREADS) : "memory"); }
template <int THREADS> __device__ __forceinline__ void bar_arrive(int id) { asm volatile("bar.arrive %0, %1;\n" ::"r"(id), "n"(THREADS) : "memory"); }
#endif

// ---------------------------------------------------------------------------
// The lossless 12-bit KV cache (glyd_gpu's vLLM plugin, kv: lossless): pages of 16 tokens of H kv heads of 128
// channels, bf16 values held in 12 bits each and read back bit for bit.
//
// A value is its bf16 low byte as it is (the exponent's lowest bit and the 7 mantissa bits) and a 4-bit code for the high
// byte: the sign and an offset 0-7 from a base, the window of 8 values of exponent >> 1 that holds the most of the
// (layer, head, channel)'s values in other text (kbase, vbase: uint8 [H][D] a layer, calibrated once a model). A value
// outside its window is an exception: its code is offset 0, and an entry in its head's header, its page's arena or its page's chunks XORs
// the byte base ^ (exponent >> 1) into the decoded high byte, so that every 16-bit pattern (NaN, infinity, denormals,
// both zeros) comes back as it went in.
//
// A page, for H heads and an arena of A entries (hdr_bytes(H, A) = 16 H + 4 (1 + ND + A) up to a multiple of 128):
//   0                  H head headers of 16 bytes: u32 count (the head-page's entries, K and V, dead ones too), u32 e0, e1, e2
//   16 H               u32 used (the entries past the heads' inline ones that the page has been given, dead ones and dropped ones too),
//                      u32 dir[ND] (the escape pool's chunks the page holds: the chunk's number + 1, 0 none), then A u32 entries (the arena)
//   hdr_bytes(H, A)    H head-pages of 6,144 bytes: the K tile of 3,072 bytes, then the V tile
// A page's entries past the three inline in its head's header are its entries number 0, 1, 2 ...: number i < A is arena[i], and number i >= A is
// entry (i - A) % CH of the chunk dir[(i - A) / CH] of the escape pool (EscapePool: chunks of CH entries a process keeps apart from its pages, taken
// as an entry needs one and given back when the page is cleared). The arena is what the header's last bytes up to a multiple of 128 leave: 15
// entries for 8 heads, and a page that needs more takes a chunk (a page at most A + ND CH entries).
// An entry (u32): lane 0-4, group 5-8, byte 9-10, the XOR byte 11-18, token 19-22, kind (0 K, 1 V) 23, head 24-28,
// live 29 (a rewritten token's old entries are killed, not moved).
//
// Raw pages: a page that has no place for an entry is stored without exceptions, its extra bits in a slab of a raw pool; a page that finds no slab is void and its block is named in the status ring.
//
// Writing is owned: one CTA a run of the step's tokens that fall in one page (runs found from the slots alone), a warp a
// head; the warp alone writes its head's tile bytes and header, so no word is written by two threads; the page's entry
// counter, and the escape pool's chunks it takes, are the atomics. A page whose first token is written is cleared first (its
// headers, its counter, its directory, the chunks it held given back to the pool): a recycled block's bytes are never assumed
// zero. A token written again kills its old entries first (in its head's header, the arena and the chunks); a page's first
// token written again clears the page, so a rewrite from there on writes every token after it too.
namespace kvc {

constexpr int D = 128, PT = 16, TILE = 3072, HEADPAGE = 2 * TILE;
constexpr int NW = 4;  // warps a CTA of the attention kernel
static_assert(NW * 32 == D, "the attention kernel's CTA merges its warps' outputs a thread a channel");
constexpr unsigned FULLM = 0xffffffffu;
constexpr uint32_t LIVE = 1u << 29;
constexpr int ND = 16;  // chunks of the escape pool a page's header can name
constexpr int CH = 64;  // entries in a chunk of the escape pool
constexpr uint32_t PG_RAW = 1u << 31, PG_VOID = 1u << 30, PG_COUNT = (1u << 24) - 1u;  // flags and count of a page's `used` word
constexpr uint32_t HEAD_RAW = 0xFFFFFFFFu;  // a raw head-page's count word
// the write kernel's status words (glyd_gpu.h); the two rings follow them: the blocks with a void page, the blocks with a raw page, `ring` places each
enum { ST_OVERFLOW = 0, ST_PAGE_HW = 1, ST_ARENA_PAGES = 2, ST_ENTRIES = 3, ST_POOL_USED = 4, ST_POOL_HW = 5, ST_POOL_PAGES = 6, ST_STALE = 7, ST_RAW_USED = 8, ST_RAW_HW = 9, ST_RAW_PAGES = 10, ST_VOID = 11,
       ST_RING_HEAD = 12, ST_RAW_RING_HEAD = 13, ST_CLASH = 14, ST_CLASH_PAGE = 15, ST_RING = 16 };

__host__ __device__ constexpr int hdr_bytes(int H, int A) { return (16 * H + 4 * (1 + ND + A) + 127) & ~127; }
// A head-page's bytes: its K tile and its V tile, or (one stream: a tensor held in the K tile's layout alone) its one tile.
__host__ __device__ constexpr int64_t head_page(bool one) { return one ? TILE : HEADPAGE; }
__host__ __device__ constexpr int64_t page_bytes(int H, int A, bool one = false) { return hdr_bytes(H, A) + (int64_t)H * head_page(one); }

template <int OFF>
__device__ __forceinline__ uint4 ldg_stream(const uint8_t* p) {  // a streaming read: through the read-only path, no L1 line kept
#if defined(__HIP_PLATFORM_AMD__)
    return *(const uint4*)(p + OFF);
#else
    uint4 r;
    asm("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4+%5];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p), "n"(OFF));
    return r;
#endif
}
__device__ __forceinline__ float ex2(float x) {
#if defined(__HIP_PLATFORM_AMD__)
    return __builtin_amdgcn_exp2f(x);
#else
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
#endif
}
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&v);
}
__device__ __forceinline__ uint32_t transpose8x8(uint32_t a) {  // the 8 x 8 matrix of b16 a warp holds (a lane: row lane / 4, columns 2 (lane % 4) and the next), transposed
#if defined(__HIP_PLATFORM_AMD__)
    return glyd_transpose8x8(a);
#else
    uint32_t d;
    asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(d) : "r"(a));
    return d;
#endif
}
__device__ __forceinline__ void mma(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
#if defined(__HIP_PLATFORM_AMD__)
    const uint32_t a[4] = {a0, a1, a2, a3};
    glyd_mma16816(c, a, b0, b1);
#else
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ uint32_t entry(int lane, int group, int byte, uint32_t x, int tok, int kind, int head) {
    return (uint32_t)lane | ((uint32_t)group << 5) | ((uint32_t)byte << 9) | (x << 11) | ((uint32_t)tok << 19) | ((uint32_t)kind << 23) | ((uint32_t)head << 24) | LIVE;
}
__device__ __forceinline__ int e_tok(uint32_t e) { return (e >> 19) & 15; }
__device__ __forceinline__ int e_kind(uint32_t e) { return (e >> 23) & 1; }
__device__ __forceinline__ int e_head(uint32_t e) { return (e >> 24) & 31; }

// ---- the escape pool ----------------------------------------------------------------------------------------------------------------
// Chunks of CH entries kept apart from the pages, for the entries past a page's arena (one pool a process: every layer's pages draw on it). A pool of nc chunks is, in one
// buffer: u32 bits[nc / 32 up to a multiple of 4] (a chunk's bit set: taken), u32 owner[nc up to a multiple of 4] (the page a taken chunk is on loan to: its layer's tag
// (layer + 1) << 24 and its number, 0 free), u32 ent[nc][CH]. A page names the chunks it holds in its header (dir); they go back when the page is cleared, and only
// those whose owner says they are the page's (a block whose bytes were never written by this pool names no chunk it can free).
struct EscapePool {
    uint32_t* bits;
    uint32_t* owner;
    uint32_t* ent;
    int nc;  // chunks (0: none)
};
__host__ __device__ constexpr int64_t esc_bits_bytes(int64_t nc) { return (((nc + 31) / 32 + 3) / 4) * 16; }
__host__ __device__ constexpr int64_t esc_owner_bytes(int64_t nc) { return ((nc + 3) / 4) * 16; }
__host__ __device__ constexpr int64_t esc_bytes(int64_t nc) { return esc_bits_bytes(nc) + esc_owner_bytes(nc) + nc * CH * 4; }
static EscapePool esc_of(uint8_t* base, int64_t nc) {
    if (!nc || !base) return EscapePool{nullptr, nullptr, nullptr, 0};
    return EscapePool{(uint32_t*)base, (uint32_t*)(base + esc_bits_bytes(nc)), (uint32_t*)(base + esc_bits_bytes(nc) + esc_owner_bytes(nc)), (int)nc};
}

// A chunk taken (its number, or -1 where none is free): the bitmap searched from `hint`'s word, a bit claimed by atomicOr.
__device__ __forceinline__ int esc_alloc(const EscapePool& E, uint32_t tag, uint32_t hint, uint32_t* status) {
    const int nw = (E.nc + 31) >> 5;
    for (int probe = 0; probe < nw; probe++) {
        const int w = (int)((hint + (uint32_t)probe) % (uint32_t)nw);
        const uint32_t valid = (w == nw - 1 && (E.nc & 31)) ? (1u << (E.nc & 31)) - 1u : 0xFFFFFFFFu;
        uint32_t x = *(volatile uint32_t*)(E.bits + w);
        while ((~x) & valid) {
            const int b = __ffs((~x) & valid) - 1;
            const uint32_t old = atomicOr(E.bits + w, 1u << b);
            if (!(old & (1u << b))) {
                E.owner[w * 32 + b] = tag;
                const uint32_t use = atomicAdd(status + ST_POOL_USED, 1u) + 1u;
                atomicMax(status + ST_POOL_HW, use);
                return w * 32 + b;
            }
            x = old | (1u << b);
        }
    }
    return -1;
}
__device__ __forceinline__ void esc_free(const EscapePool& E, int c, uint32_t* status) {
    E.owner[c] = 0;
    __threadfence();
    atomicAnd(E.bits + (c >> 5), ~(1u << (c & 31)));
    atomicSub(status + ST_POOL_USED, 1u);
}
// The page's chunk j (dirw = &dir[j]: the chunk + 1, 0 none): taken where the page has none yet (two warps of a page may both try: the loser gives its chunk back and uses the
// winner's); 0 where the pool is out of chunks.
__device__ __forceinline__ uint32_t esc_chunk(const EscapePool& E, uint32_t* dirw, uint32_t tag, uint32_t hint, uint32_t* status) {
    const uint32_t d = *(volatile uint32_t*)dirw;
    if (d) return d - 1 < (uint32_t)E.nc ? d : 0u;
    if (!E.nc) return 0u;
    const int c = esc_alloc(E, tag, hint, status);
    if (c < 0) return 0;
    const uint32_t old = atomicCAS(dirw, 0u, (uint32_t)c + 1u);
    if (old != 0u) {
        esc_free(E, c, status);
        return old;
    }
    return (uint32_t)c + 1u;
}
// ---- the raw pool ---------------------------------------------------------------------------------------------------------------
// Bitmap, owners and ns slabs of plane_bytes(H), in one buffer, as the escape pool's.
struct RawPool {
    uint32_t* bits;
    uint32_t* owner;
    uint8_t* data;
    int ns;
    int64_t plane;  // a slab's bytes
};
__host__ __device__ constexpr int64_t raw_bits_bytes(int64_t ns) { return (((ns + 31) / 32 + 3) / 4) * 16; }
__host__ __device__ constexpr int64_t raw_owner_bytes(int64_t ns) { return ((ns + 3) / 4) * 16; }
__host__ __device__ constexpr int64_t plane_bytes(int H) { return (int64_t)H * 2048; }
__host__ __device__ constexpr int64_t raw_bytes(int64_t ns, int H) { return raw_bits_bytes(ns) + raw_owner_bytes(ns) + ns * plane_bytes(H); }
static RawPool raw_of(uint8_t* base, int64_t ns, int H) {
    if (!ns || !base) return RawPool{nullptr, nullptr, nullptr, 0, 0};
    return RawPool{(uint32_t*)base, (uint32_t*)(base + raw_bits_bytes(ns)), base + raw_bits_bytes(ns) + raw_owner_bytes(ns), (int)ns, plane_bytes(H)};
}
__device__ __forceinline__ int raw_alloc(const RawPool& R, uint32_t tag, uint32_t hint, uint32_t* status) {
    const int nw = (R.ns + 31) >> 5;
    for (int probe = 0; probe < nw; probe++) {
        const int w = (int)((hint + (uint32_t)probe) % (uint32_t)nw);
        const uint32_t valid = (w == nw - 1 && (R.ns & 31)) ? (1u << (R.ns & 31)) - 1u : 0xFFFFFFFFu;
        uint32_t x = *(volatile uint32_t*)(R.bits + w);
        while ((~x) & valid) {
            const int b = __ffs((~x) & valid) - 1;
            const uint32_t old = atomicOr(R.bits + w, 1u << b);
            if (!(old & (1u << b))) {
                R.owner[w * 32 + b] = tag;
                const uint32_t use = atomicAdd(status + ST_RAW_USED, 1u) + 1u;
                atomicMax(status + ST_RAW_HW, use);
                return w * 32 + b;
            }
            x = old | (1u << b);
        }
    }
    return -1;
}
__device__ __forceinline__ void raw_free(const RawPool& R, int c, uint32_t* status) {
    R.owner[c] = 0;
    __threadfence();
    atomicAnd(R.bits + (c >> 5), ~(1u << (c & 31)));
    atomicSub(status + ST_RAW_USED, 1u);
}

// The word of a page's entry number idx (pg: the page's `used` word, then its dir and arena), in its arena or in the chunk its directory names; null where that chunk is missing, or
// the directory's word is not a chunk of this pool (a page that was never written holds a previous owner's bytes, and a decode of one must not go where they point).
__device__ __forceinline__ const uint32_t* entry_at(const uint32_t* pg, const uint32_t* ent, int nc, int A, uint32_t idx) {
    if (idx < (uint32_t)A) return pg + 1 + ND + idx;
    const uint32_t r = idx - (uint32_t)A;
    const uint32_t d = __ldg(pg + 1 + r / CH);
    return d - 1 < (uint32_t)nc ? ent + (size_t)(d - 1) * CH + r % CH : nullptr;
}

// The (token, channel) of value m of group i of lane (g, q).
template <int KIND>  // 0: K, 1: V
__device__ __forceinline__ void coord(int g, int q, int i, int m, int& tok, int& ch) {
    if (KIND == 0) {
        tok = 8 * (i & 1) + g;
        ch = 16 * (i >> 1) + 2 * q + (m & 1) + 8 * (m >> 1);
    } else {
        ch = g + 8 * i;
        tok = 2 * q + (m & 1) + 8 * (m >> 1);
    }
}

// The bases as bytes of the lane's channels: K: a word a code word i (the channels of its group's step, 16 s + 2 q + {0, 1, 8, 9});
// V: a word a group, the channel g + 8 i's base in every byte.
template <int KIND>
__device__ __forceinline__ void load_bases(const uint8_t* base, int lane, uint32_t (&hb)[KIND == 0 ? 8 : 16]) {
    const int g = lane >> 2, q = lane & 3;
    if (KIND == 0) {
#pragma unroll
        for (int s = 0; s < 8; s++) hb[s] = (uint32_t)base[16 * s + 2 * q] | ((uint32_t)base[16 * s + 2 * q + 1] << 8) | ((uint32_t)base[16 * s + 2 * q + 8] << 16) | ((uint32_t)base[16 * s + 2 * q + 9] << 24);
    } else {
#pragma unroll
        for (int i = 0; i < 16; i++) hb[i] = (uint32_t)base[g + 8 * i] * 0x01010101u;
    }
}

// ---- reading a tile ----------------------------------------------------------------------------------------------------
struct Raw {  // a lane's bytes of a tile: 4 chunks of low bytes, 2 of codes
    uint4 lo[4], cd[2];
};
__device__ __forceinline__ void raw_load(Raw& r, const uint8_t* tile, int lane) {
    const uint8_t* p = tile + lane * 16;
    r.lo[0] = ldg_stream<0>(p);
    r.lo[1] = ldg_stream<512>(p);
    r.lo[2] = ldg_stream<1024>(p);
    r.lo[3] = ldg_stream<1536>(p);
    r.cd[0] = ldg_stream<2048>(p);
    r.cd[1] = ldg_stream<2560>(p);
}

// raw_load by the ordinary path (for a tile written in this kernel).
__device__ __forceinline__ void raw_load_plain(Raw& r, const uint8_t* tile, int lane) {
    const uint4* p = reinterpret_cast<const uint4*>(tile + lane * 16);
    r.lo[0] = p[0];
    r.lo[1] = p[32];
    r.lo[2] = p[64];
    r.lo[3] = p[96];
    r.cd[0] = p[128];
    r.cd[1] = p[160];
}

// One entry: the byte XORed into the high bytes word of its group where this lane owns it (the entry is the same for the warp,
// so which word is patched is a warp-uniform branch, not a select on each of the 16).
__device__ __forceinline__ void patch(uint32_t (&H)[16], uint32_t e, int lane) {
    const uint32_t v = ((e & 31u) == (uint32_t)lane) ? (((e >> 11) & 0xFFu) << (((e >> 9) & 3u) * 8u)) : 0u;
    switch ((e >> 5) & 15u) {
#define KV_PATCH(i) \
    case i:         \
        H[i] ^= v;  \
        break;
        KV_PATCH(0) KV_PATCH(1) KV_PATCH(2) KV_PATCH(3) KV_PATCH(4) KV_PATCH(5) KV_PATCH(6) KV_PATCH(7)
        KV_PATCH(8) KV_PATCH(9) KV_PATCH(10) KV_PATCH(11) KV_PATCH(12) KV_PATCH(13) KV_PATCH(14) KV_PATCH(15)
#undef KV_PATCH
    }
}

// The entries of this head (kind KIND) applied to H: the three inline ones, and where the head holds more the page's others (its arena, then the chunks of
// the escape pool its directory names), read a line at a time (a lane an entry) and the matches taken in turn. pg: the page's `used` word.
template <int KIND>
__device__ __forceinline__ void apply_entries(uint32_t (&H)[16], const uint4 hd, const uint32_t* pg, const uint32_t* ent, int nc, int A, int head, int lane, int limit) {
    const uint32_t cnt = hd.x;
    if (cnt == 0) return;
    const uint32_t in[3] = {hd.y, hd.z, hd.w};
#pragma unroll
    for (int j = 0; j < 3; j++)
        if (j < (int)cnt && (in[j] & LIVE) && e_kind(in[j]) == KIND) patch(H, in[j], lane);
    if (cnt > 3) {
        const uint32_t used = min(limit >= 0 ? (uint32_t)limit : __ldg(pg) & PG_COUNT, (uint32_t)(A + ND * CH));  // limit >= 0: only the entries before the run
        for (uint32_t base = 0; base < used; base += 32) {
            const uint32_t idx = base + lane;
            uint32_t e = 0;
            if (idx < used) {
                const uint32_t* p = entry_at(pg, ent, nc, A, idx);
                if (p) e = __ldg(p);
            }
            uint32_t bal = __ballot_sync(FULLM, (e & LIVE) && e_kind(e) == KIND && e_head(e) == head);
            while (bal) {
                const int src = __ffs(bal) - 1;
                bal &= bal - 1;
                patch(H, __shfl_sync(FULLM, e, src), lane);
            }
        }
    }
}

struct F8 {
    uint4 v[8];
};

// A raw head-page's tile, decoded (not inlined: rare).
template <int KIND>
__device__ __noinline__ F8 decode_raw(const Raw r, const uint32_t* pg, const RawPool R, int head, int lane) {
    const uint32_t d = __ldg(pg + 1);
    uint32_t pw[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
    if (d - 1 < (uint32_t)R.ns) {
        const uint8_t* pt = R.data + (size_t)(d - 1) * (size_t)R.plane + (size_t)(2 * head + KIND) * 1024 + lane * 16;
        const uint4 a = __ldg(reinterpret_cast<const uint4*>(pt)), b = __ldg(reinterpret_cast<const uint4*>(pt + 512));
        pw[0] = a.x, pw[1] = a.y, pw[2] = a.z, pw[3] = a.w, pw[4] = b.x, pw[5] = b.y, pw[6] = b.z, pw[7] = b.w;
    }
    const uint32_t cw[8] = {r.cd[0].x, r.cd[0].y, r.cd[0].z, r.cd[0].w, r.cd[1].x, r.cd[1].y, r.cd[1].z, r.cd[1].w};
    const uint32_t L[16] = {r.lo[0].x, r.lo[0].y, r.lo[0].z, r.lo[0].w, r.lo[1].x, r.lo[1].y, r.lo[1].z, r.lo[1].w,
                            r.lo[2].x, r.lo[2].y, r.lo[2].z, r.lo[2].w, r.lo[3].x, r.lo[3].y, r.lo[3].z, r.lo[3].w};
    F8 out;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const uint32_t cb = __funnelshift_l(cw[i], cw[i], 4);
        const uint32_t H0 = (cw[i] & 0x80808080u) | ((cw[i] & 0x07070707u) << 4) | (pw[i] & 0x0F0F0F0Fu);
        const uint32_t H1 = (cb & 0x80808080u) | ((cb & 0x07070707u) << 4) | ((pw[i] >> 4) & 0x0F0F0F0Fu);
        out.v[i] = make_uint4(__byte_perm(L[2 * i], H0, 0x5140), __byte_perm(L[2 * i + 1], H1, 0x5140), __byte_perm(L[2 * i], H0, 0x7362), __byte_perm(L[2 * i + 1], H1, 0x7362));
    }
    return out;
}

// A tile's 64 values a lane as the 8 chunks of bf16 pairs the tensor cores take as A operand: f[i] = (pair 0 of group 2i, pair 0 of
// group 2i + 1, pair 1 of group 2i, pair 1 of group 2i + 1) = a0, a1, a2, a3 of step i.
template <int KIND>
__device__ __forceinline__ void decode_tile(const Raw& r, const uint32_t* hb, const uint4 hd, const uint32_t* pg, const uint32_t* ent, int nc, int A, const RawPool& R, int head, int lane, uint4 (&f)[8], int limit = -1) {
    if (__builtin_expect(hd.x == HEAD_RAW, 0)) {
        const F8 fr = decode_raw<KIND>(r, pg, R, head, lane);
#pragma unroll
        for (int i = 0; i < 8; i++) f[i] = fr.v[i];
        return;
    }
    const uint32_t cw[8] = {r.cd[0].x, r.cd[0].y, r.cd[0].z, r.cd[0].w, r.cd[1].x, r.cd[1].y, r.cd[1].z, r.cd[1].w};
    const uint32_t L[16] = {r.lo[0].x, r.lo[0].y, r.lo[0].z, r.lo[0].w, r.lo[1].x, r.lo[1].y, r.lo[1].z, r.lo[1].w,
                            r.lo[2].x, r.lo[2].y, r.lo[2].z, r.lo[2].w, r.lo[3].x, r.lo[3].y, r.lo[3].z, r.lo[3].w};
    uint32_t H[16];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        H[2 * i] = (cw[i] & 0x87878787u) + hb[KIND == 0 ? i : 2 * i];
        H[2 * i + 1] = (__funnelshift_l(cw[i], cw[i], 4) & 0x87878787u) + hb[KIND == 0 ? i : 2 * i + 1];
    }
    apply_entries<KIND>(H, hd, pg, ent, nc, A, head, lane, limit);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const uint32_t a0 = __byte_perm(L[2 * i], H[2 * i], 0x5140), a1 = __byte_perm(L[2 * i], H[2 * i], 0x7362);
        const uint32_t b0 = __byte_perm(L[2 * i + 1], H[2 * i + 1], 0x5140), b1 = __byte_perm(L[2 * i + 1], H[2 * i + 1], 0x7362);
        f[i] = make_uint4(a0, b0, a1, b1);
    }
}

// ---- writing ------------------------------------------------------------------------------------------------------------
struct WriteParams {
    const uint16_t* key;
    const uint16_t* value;
    int64_t key_stride, value_stride;
    const int64_t* slots;
    uint8_t* pool;
    int64_t page_stride;
    const uint8_t* kbase;
    const uint8_t* vbase;
    uint32_t* status;
    EscapePool esc;
    RawPool raw;
    uint32_t* lost;  // two bits a block, 2 b: it has a void page, 2 b + 1: a raw page (null: none); nlost bits
    int64_t nlost;
    uint32_t ring;  // places of each of the status's two rings
    uint32_t layer_tag;  // (layer + 1) << 24: the owner tag of this layer's pages in the pools
    int N, H, A;
    int lean;  // 0: every run through write_run (the path of a run of any length); else a run of one token through the decode step's path
    int one;   // 1: one stream (the keys alone: no value is read, a page holds no V tile)
};

__shared__ uint32_t g_need;  // an entry of the CTA's page found no place: the page is made raw (make_raw)

// The entries past a head's three inline ones, appended by a warp (all 32 lanes call it, together: obal is the lanes that have one, not 0): the lanes' places in the page's list
// of entries by one atomic on its counter (their numbers are consecutive), the chunks of the escape pool those places fall in taken first (by the first lane, before any lane
// stores), then each lane stores its entry in the arena below A, else in its chunk. An entry that has no place (the pool is out of chunks, or the page's directory is full)
// is dropped and counted (the status's overflow) and the CTA marked (g_need): it makes the page raw before it ends.
// Not inlined: it is rare (a page past its heads' three inline entries) and each call of an inlined one would be a copy of it in
// the write kernel, which has no registers to spare.
__device__ __noinline__ void append_over(const WriteParams& P, uint32_t* pg, uint32_t tag, uint32_t e, bool over, uint32_t obal, int lane) {
    const int A = P.A;
    const EscapePool esc = P.esc;
    uint32_t* status = P.status;
    const int leader = __ffs(obal) - 1;
    const uint32_t n = __popc(obal);
    uint32_t base = 0;
    if (lane == leader) {
        base = atomicAdd(pg, n) & PG_COUNT;
        if (!*(volatile uint32_t*)&g_need && base + n > (uint32_t)A) {  // the chunks these entries fall in, taken before any lane stores
            const uint32_t lo = base > (uint32_t)A ? base - A : 0u, hi = base + n - A;
            for (uint32_t j = lo / CH; j <= (hi - 1) / CH && j < (uint32_t)ND; j++) esc_chunk(esc, pg + 1 + j, tag, tag * 2654435761u + j * 40503u, status);
        }
    }
    __syncwarp();
    base = __shfl_sync(FULLM, base, leader);
    bool lost = false;
    if (over) {
        const uint32_t a = base + __popc(obal & ((1u << lane) - 1u));
        if (a < (uint32_t)A) {
            pg[1 + ND + a] = e;
        } else {
            const uint32_t r = a - A, j = r / CH;
            const uint32_t d = j < (uint32_t)ND ? *(volatile uint32_t*)(pg + 1 + j) : 0u;
            if (d - 1 < (uint32_t)esc.nc) esc.ent[(size_t)(d - 1) * CH + r % CH] = e;
            else lost = true;
        }
    }
    const uint32_t lbal = __ballot_sync(FULLM, lost);
    if (lane == leader) {
        if (lbal) {
            atomicOr(&g_need, 1u);
            atomicAdd(status + ST_OVERFLOW, (uint32_t)__popc(lbal));
        }
        atomicMax(status + ST_PAGE_HW, base + n);
        if (base == 0 && A > 0) atomicAdd(status + ST_ARENA_PAGES, 1u);
        if (base <= (uint32_t)A && base + n > (uint32_t)A) atomicAdd(status + ST_POOL_PAGES, 1u);
    }
}

struct Appender {  // a head-page's entries being appended by a warp
    uint32_t cnt;        // the entries the head-page holds (warp-uniform)
    uint32_t* hdr;       // its header: the count, then three inline entries
    uint32_t* pg;        // the page's `used` word, its directory and its arena
    uint32_t tag;        // this page's owner tag in the pool
    const WriteParams* P;  // (the kernel's parameters: read where an entry needs them, held in no register)
    // Every lane calls it (the votes are the warp's): the lanes with `mine` append e, inline while the head holds fewer than three, else to the page's list of entries (append_over).
    __device__ __forceinline__ void add(uint32_t e, bool mine, int lane) {
        const uint32_t bal = __ballot_sync(FULLM, mine);
        if (!bal) return;
        const uint32_t idx = cnt + (mine ? __popc(bal & ((1u << lane) - 1u)) : 0u);
        const bool over = mine && idx >= 3;
        if (mine && idx < 3) hdr[1 + idx] = e;
        const uint32_t obal = __ballot_sync(FULLM, over);
        if (obal) append_over(*P, pg, tag, e, over, obal, lane);
        if (lane == __ffs(bal) - 1) atomicAdd(P->status + ST_ENTRIES, (uint32_t)__popc(bal));
        cnt += __popc(bal);
    }
};

// A token's 4 values of a group, as the code bits and the exception: value v (bf16 bits), its channel's base byte b.
// Returns the 4-bit code (sign << 3 | offset, 0 for an exception); esc, x: the exception and the byte to XOR in.
__device__ __forceinline__ uint32_t code_of(uint32_t v, uint32_t b, bool& esc, uint32_t& x) {
    const uint32_t e7 = (v >> 8) & 127u;
    const int off = (int)e7 - (int)b;
    esc = off < 0 || off > 7;
    x = esc ? (b ^ e7) : 0u;
    return ((v >> 15) << 3) | (esc ? 0u : (uint32_t)off);
}

// ---- writing a run of tokens into a head-page's two tiles -------------------------------------------------------------------------
// A tile is laid out for the readers' lanes (a lane (g, q) of the decode warp owns words of 16 bytes at lane * 16 of each 512-byte chunk), but the writer's
// lanes divide the work by value, not by owner: a decode step writes one token, and a warp that gave the owners' lanes their values would have 4 to 8
// of its 32 lanes at work through 16 passes. Here a lane takes 4 values of every token of the run (K: a step s and a q, 16 s + 2 q + {0, 1, 8, 9}; V: 4 of
// the 16 channels g + 8 i of a g) and writes the words of the owner lanes they belong to, so a token is one pass over all 32 lanes, a run of 16 sixteen.
// Everything a lane reads (its values, the bases, the words it will write part of) is read first (k_load, v_load: before the kills' barrier), then
// computed and written (k_write, v_write: after it).
struct KPre {
    uint32_t kb;         // the bases of the lane's 4 channels, a byte each
    uint32_t p[2][8][2]; // [nt][g][half]: the lane's keys of token 8 nt + g (2 channels a word), where the run holds it
    uint32_t ow[8];      // [g]: the code word it writes of owner 4 g + q, where the run holds one of its two tokens
};
__device__ __forceinline__ bool k_tok(int nt, int g, int o0, int len) { return 8 * nt + g >= o0 && 8 * nt + g < o0 + len; }
__device__ __forceinline__ void k_load(KPre& r, const uint16_t* __restrict__ key, int64_t stride, const uint8_t* __restrict__ base, const uint8_t* tile, int o0, int len, int lane) {
    const int s = lane >> 2, q = lane & 3;
    r.kb = (uint32_t)base[16 * s + 2 * q] | ((uint32_t)base[16 * s + 2 * q + 1] << 8) | ((uint32_t)base[16 * s + 2 * q + 8] << 16) | ((uint32_t)base[16 * s + 2 * q + 9] << 24);
#pragma unroll
    for (int g = 0; g < 8; g++) {
#pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            const bool act = k_tok(nt, g, o0, len);
            const uint16_t* src = key + (int64_t)(8 * nt + g - o0) * stride + 16 * s + 2 * q;  // (read only where act)
            r.p[nt][g][0] = act ? __ldg(reinterpret_cast<const uint32_t*>(src)) : 0u;
            r.p[nt][g][1] = act ? __ldg(reinterpret_cast<const uint32_t*>(src + 8)) : 0u;
        }
        const bool a0 = k_tok(0, g, o0, len), a1 = k_tok(1, g, o0, len);
        r.ow[g] = (a0 != a1) ? *reinterpret_cast<const uint32_t*>(tile + 2048 + (s >> 2) * 512 + (4 * g + q) * 16 + (s & 3) * 4) : 0u;  // (a half of the word is the other token's)
    }
}
__device__ __forceinline__ void k_write(const KPre& r, uint8_t* tile, int o0, int len, int lane, int head, Appender& ap) {
    const int s = lane >> 2, q = lane & 3;
    const uint32_t cmask[2] = {0x87878787u, 0x78787878u};
#pragma unroll
    for (int g = 0; g < 8; g++) {
        const bool act[2] = {k_tok(0, g, o0, len), k_tok(1, g, o0, len)};
        if (!(act[0] || act[1])) continue;  // (g is the loop's: every lane skips alike)
        const int owner = 4 * g + q;
        uint32_t w = r.ow[g];
        uint32_t ent[2][4];
        bool esc[2][4], any = false;
#pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            const uint32_t v[4] = {r.p[nt][g][0] & 0xFFFFu, r.p[nt][g][0] >> 16, r.p[nt][g][1] & 0xFFFFu, r.p[nt][g][1] >> 16};
            const int i = 2 * s + nt;
            uint32_t lw = 0, cb = 0;
#pragma unroll
            for (int m = 0; m < 4; m++) {
                const uint32_t b = (r.kb >> (8 * m)) & 0xFFu;
                uint32_t x;
                const uint32_t code = code_of(v[m], b, esc[nt][m], x);
                esc[nt][m] &= act[nt];
                any |= esc[nt][m];
                ent[nt][m] = esc[nt][m] ? entry(owner, i, m, x, 8 * nt + g, 0, head) : 0u;
                lw |= (v[m] & 255u) << (8 * m);
                if (nt == 0) cb |= (((code >> 3) << 7) | (code & 7u)) << (8 * m);
                else cb |= (((code >> 3) << 3) << (8 * m)) | ((code & 7u) << (8 * ((m + 3) & 3) + 4));
            }
            if (act[nt]) {
                *reinterpret_cast<uint32_t*>(tile + (i >> 2) * 512 + owner * 16 + (i & 3) * 4) = lw;
                w = (w & ~cmask[nt]) | cb;
            }
        }
        *reinterpret_cast<uint32_t*>(tile + 2048 + (s >> 2) * 512 + owner * 16 + (s & 3) * 4) = w;
        if (__any_sync(FULLM, any)) {
#pragma unroll
            for (int nt = 0; nt < 2; nt++) {
#pragma unroll
                for (int m = 0; m < 4; m++) ap.add(ent[nt][m], esc[nt][m], lane);
            }
        }
    }
}

struct VPre {
    uint32_t bs[4];        // the bases of the lane's 4 channels g + 8 (4 ig + jj)
    uint16_t vv[4][4][4];  // [token group][token of it][jj]: the lane's values (a token group q' holds tokens 2 q' + {0, 1, 8, 9})
    uint4 lw[4];           // [token group]: the 4 low words (4 ig .. 4 ig + 3) it writes bytes of, of owner 4 g + q'
    uint2 cw[4];           // [token group]: the 2 code words (2 ig, 2 ig + 1)
};
__device__ __forceinline__ void v_masks(int qg, int o0, int len, uint32_t& lmask, uint32_t& cmask) {  // the bytes of a low word and the bits of a code word the run holds of token group qg
    lmask = cmask = 0;
#pragma unroll
    for (int m = 0; m < 4; m++) {
        const int t = 2 * qg + (m & 1) + 8 * (m >> 1);
        if (t >= o0 && t < o0 + len) {
            lmask |= 0xFFu << (8 * m);
            cmask |= (0x87u << (8 * m)) | (0x08u << (8 * m)) | (0x70u << (8 * ((m + 3) & 3)));
        }
    }
}
__device__ __forceinline__ void v_load(VPre& r, const uint16_t* __restrict__ value, int64_t stride, const uint8_t* __restrict__ base, const uint8_t* tile, int o0, int len, int lane) {
    const int g = lane & 7, ig = lane >> 3;
#pragma unroll
    for (int jj = 0; jj < 4; jj++) r.bs[jj] = base[g + 8 * (4 * ig + jj)];
#pragma unroll
    for (int qg = 0; qg < 4; qg++) {
        uint32_t lmask, cmask;
        v_masks(qg, o0, len, lmask, cmask);
#pragma unroll
        for (int m = 0; m < 4; m++) {
            const int t = 2 * qg + (m & 1) + 8 * (m >> 1);
            const bool act = t >= o0 && t < o0 + len;
            const uint16_t* row = value + (int64_t)(t - o0) * stride + g + 8 * (4 * ig);  // (read only where act)
#pragma unroll
            for (int jj = 0; jj < 4; jj++) r.vv[qg][m][jj] = act ? __ldg(row + 8 * jj) : (uint16_t)0;
        }
        r.lw[qg] = make_uint4(0, 0, 0, 0);
        r.cw[qg] = make_uint2(0, 0);
        if (lmask && lmask != 0xFFFFFFFFu) r.lw[qg] = *reinterpret_cast<const uint4*>(tile + ig * 512 + (4 * g + qg) * 16);
        if (cmask && cmask != 0xFFFFFFFFu) r.cw[qg] = *reinterpret_cast<const uint2*>(tile + 2048 + (ig >> 1) * 512 + (4 * g + qg) * 16 + 2 * (ig & 1) * 4);
    }
}
__device__ __forceinline__ void v_write(const VPre& r, uint8_t* tile, int o0, int len, int lane, int head, Appender& ap) {
    const int g = lane & 7, ig = lane >> 3;
#pragma unroll
    for (int qg = 0; qg < 4; qg++) {
        uint32_t lmask, cmask;
        v_masks(qg, o0, len, lmask, cmask);
        if (!lmask) continue;  // (qg is the loop's: every lane skips alike)
        const int owner = 4 * g + qg;
        uint32_t lbits[4] = {0, 0, 0, 0}, cb[2] = {0, 0}, ent[4][4];
        bool esc[4][4], any = false;
#pragma unroll
        for (int m = 0; m < 4; m++) {
            const int t = 2 * qg + (m & 1) + 8 * (m >> 1);
            const bool act = t >= o0 && t < o0 + len;
#pragma unroll
            for (int jj = 0; jj < 4; jj++) {
                const uint32_t v = r.vv[qg][m][jj];
                uint32_t x;
                const uint32_t code = code_of(v, r.bs[jj], esc[m][jj], x);
                esc[m][jj] &= act;
                any |= esc[m][jj];
                ent[m][jj] = esc[m][jj] ? entry(owner, 4 * ig + jj, m, x, t, 1, head) : 0u;
                if (act) {
                    lbits[jj] |= (v & 255u) << (8 * m);
                    if ((jj & 1) == 0) cb[jj >> 1] |= (((code >> 3) << 7) | (code & 7u)) << (8 * m);
                    else cb[jj >> 1] |= (((code >> 3) << 3) << (8 * m)) | ((code & 7u) << (8 * ((m + 3) & 3) + 4));
                }
            }
        }
        const uint32_t olw[4] = {r.lw[qg].x, r.lw[qg].y, r.lw[qg].z, r.lw[qg].w}, ocw[2] = {r.cw[qg].x, r.cw[qg].y};
        uint32_t nlw[4], ncw[2];
#pragma unroll
        for (int jj = 0; jj < 4; jj++) nlw[jj] = ((lmask == 0xFFFFFFFFu ? 0u : olw[jj]) & ~lmask) | lbits[jj];
#pragma unroll
        for (int e = 0; e < 2; e++) ncw[e] = ((cmask == 0xFFFFFFFFu ? 0u : ocw[e]) & ~cmask) | cb[e];
        *reinterpret_cast<uint4*>(tile + ig * 512 + owner * 16) = make_uint4(nlw[0], nlw[1], nlw[2], nlw[3]);
        *reinterpret_cast<uint2*>(tile + 2048 + (ig >> 1) * 512 + owner * 16 + 2 * (ig & 1) * 4) = make_uint2(ncw[0], ncw[1]);
        if (__any_sync(FULLM, any)) {
#pragma unroll
            for (int m = 0; m < 4; m++) {
#pragma unroll
                for (int jj = 0; jj < 4; jj++) ap.add(ent[m][jj], esc[m][jj], lane);
            }
        }
    }
}


// ---- raw pages -----------------------------------------------------------------------------------------------------------------------------
// A head-page's K or V tile written raw. CONV: the page was not raw (old tokens decoded, the run's from the inputs); else the run's tokens only.
template <int KIND, bool CONV>
__device__ __forceinline__ void raw_tile(const WriteParams& P, uint8_t* tile, uint8_t* ptile, const uint4 hd, const uint32_t* pg32, int head, int lane, int o0, int len, int i0, int used0) {
    const int g = lane >> 2, q = lane & 3;
    uint4 f[8];
    if (CONV) {
        Raw r;
        raw_load_plain(r, tile, lane);
        uint32_t hb[KIND == 0 ? 8 : 16];
        load_bases<KIND>((KIND == 0 ? P.kbase : P.vbase) + head * D, lane, hb);
        decode_tile<KIND>(r, hb, hd, pg32, P.esc.ent, P.esc.nc, P.A, P.raw, head, lane, f, used0);
    }
    const int64_t stride = KIND == 0 ? P.key_stride : P.value_stride;
    const uint16_t* in = reinterpret_cast<const uint16_t*>(KIND == 0 ? P.key : P.value) + (int64_t)i0 * stride + head * D;
    uint32_t cn[8], cm[8], pn[8], pm[8], ln[16], lm[16];
#pragma unroll
    for (int s = 0; s < 8; s++) cn[s] = cm[s] = pn[s] = pm[s] = 0u;
#pragma unroll
    for (int i = 0; i < 16; i++) ln[i] = lm[i] = 0u;
#pragma unroll
    for (int i = 0; i < 16; i++) {
#pragma unroll
        for (int m = 0; m < 4; m++) {
            int tok, ch;
            coord<KIND>(g, q, i, m, tok, ch);
            const bool run = tok >= o0 && tok < o0 + len;
            if (!CONV && !run) continue;
            uint32_t v;
            if (run) {
                v = in[(int64_t)(tok - o0) * stride + ch];
            } else {
                const uint4& c = f[i >> 1];
                const uint32_t w0 = (i & 1) ? c.y : c.x, w1 = (i & 1) ? c.w : c.z;
                const uint32_t w = m < 2 ? w0 : w1;
                v = (m & 1) ? (w >> 16) : (w & 0xFFFFu);
            }
            const uint32_t hi = v >> 8, sign = hi >> 7, e7 = hi & 127u, off = e7 >> 4, pl = e7 & 15u;
            const int s = i >> 1;
            if ((i & 1) == 0) {
                cn[s] |= ((sign << 7) | off) << (8 * m);
                cm[s] |= 0x87u << (8 * m);
                pn[s] |= pl << (8 * m);
                pm[s] |= 0x0Fu << (8 * m);
            } else {
                cn[s] |= ((sign << 3) << (8 * m)) | (off << (8 * ((m + 3) & 3) + 4));
                cm[s] |= (0x08u << (8 * m)) | (0x70u << (8 * ((m + 3) & 3)));
                pn[s] |= pl << (8 * m + 4);
                pm[s] |= 0xF0u << (8 * m);
            }
            ln[i] |= (v & 0xFFu) << (8 * m);
            lm[i] |= 0xFFu << (8 * m);
        }
    }
#pragma unroll
    for (int s = 0; s < 8; s++) {
        uint32_t* cp = reinterpret_cast<uint32_t*>(tile + 2048 + (s >> 2) * 512 + lane * 16 + (s & 3) * 4);
        uint32_t* pp = reinterpret_cast<uint32_t*>(ptile + (s >> 2) * 512 + lane * 16 + (s & 3) * 4);
        if (cm[s]) *cp = (cm[s] == 0xFFFFFFFFu ? 0u : (*cp & ~cm[s])) | cn[s];
        if (pm[s]) *pp = (pm[s] == 0xFFFFFFFFu ? 0u : (*pp & ~pm[s])) | pn[s];
    }
    if (!CONV) {
#pragma unroll
        for (int i = 0; i < 16; i++) {
            uint32_t* lp = reinterpret_cast<uint32_t*>(tile + (i >> 2) * 512 + lane * 16 + (i & 3) * 4);
            if (lm[i]) *lp = (lm[i] == 0xFFFFFFFFu ? 0u : (*lp & ~lm[i])) | ln[i];
        }
    }
}

// A page that found no slab (or whose slab is not its own) is void; its block goes in the status ring (once while it stays lost).
__device__ __forceinline__ void fail_page(const WriteParams& P, uint32_t* pg32, uint32_t block) {
    pg32[0] = PG_VOID;
    atomicAdd(P.status + ST_VOID, 1u);
    const uint64_t b = 2ull * block;
    if (P.lost && b + 1 < (uint64_t)P.nlost && (atomicOr(P.lost + (b >> 5), 1u << (b & 31)) & (1u << (b & 31)))) return;
    const uint32_t idx = atomicAdd(P.status + ST_RING_HEAD, 1u);
    P.status[ST_RING + idx % P.ring] = block + 1u;
}

// A block that has a raw page goes in the status's raw ring, once while it keeps one (the host keeps such blocks out of the prefix cache, so that they are reused first).
__device__ __forceinline__ void note_raw(const WriteParams& P, uint32_t block) {
    const uint64_t b = 2ull * block + 1;
    if (P.lost && b < (uint64_t)P.nlost && (atomicOr(P.lost + (b >> 5), 1u << (b & 31)) & (1u << (b & 31)))) return;
    const uint32_t idx = atomicAdd(P.status + ST_RAW_RING_HEAD, 1u);
    P.status[ST_RING + P.ring + idx % P.ring] = block + 1u;
}

// The CTA's page made raw (every thread calls it): void where no slab is free.
__device__ __noinline__ void make_raw(const WriteParams& P, uint8_t* pg, uint32_t* pg32, int hb, uint32_t tag, int o0, int len, int i0, uint32_t block, uint32_t used0, const uint32_t* cnt0) {
    __shared__ int s_slab;
    if (threadIdx.x == 0) s_slab = P.raw.ns ? raw_alloc(P.raw, tag, tag * 2654435761u, P.status) : -1;
    __syncthreads();
    const int slab = s_slab;
    if (slab < 0) {
        if (threadIdx.x == 0) fail_page(P, pg32, block);
        __syncthreads();
        return;
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    uint8_t* plane = P.raw.data + (size_t)slab * (size_t)P.raw.plane;
    for (int pr = warp; pr < 2 * P.H; pr += (int)(blockDim.x >> 5)) {  // pr = 2 head + kind
        const int h = pr >> 1;
        if (P.one && (pr & 1)) continue;  // (one stream: no V tile)
        uint8_t* tile = pg + hb + (int64_t)h * head_page(P.one) + (pr & 1) * TILE;
        uint4 hd = *reinterpret_cast<const uint4*>(pg + 16 * h);
        hd.x = cnt0[h];  // the entries before the run
        if (pr & 1) raw_tile<1, true>(P, tile, plane + (size_t)pr * 1024, hd, pg32, h, lane, o0, len, i0, (int)used0);
        else raw_tile<0, true>(P, tile, plane + (size_t)pr * 1024, hd, pg32, h, lane, o0, len, i0, (int)used0);
    }
    __syncthreads();
    for (int j = threadIdx.x; j < ND; j += blockDim.x) {
        const uint32_t d = pg32[1 + j];
        if (d) {
            if (d - 1 < (uint32_t)P.esc.nc && P.esc.owner[d - 1] == tag) esc_free(P.esc, (int)(d - 1), P.status);
            else atomicAdd(P.status + ST_STALE, 1u);
        }
        pg32[1 + j] = 0;
    }
    for (int h = threadIdx.x; h < P.H; h += blockDim.x) *reinterpret_cast<uint4*>(pg + 16 * h) = make_uint4(HEAD_RAW, 0u, 0u, 0u);
    __syncthreads();
    if (threadIdx.x == 0) {
        pg32[1] = (uint32_t)slab + 1u;
        pg32[0] = PG_RAW;
        atomicAdd(P.status + ST_RAW_PAGES, 1u);
        note_raw(P, block);
    }
    __syncthreads();
}

// A run into a raw page (every thread calls it); void if its slab is not its own.
__device__ __noinline__ void raw_run(const WriteParams& P, uint8_t* pg, uint32_t* pg32, int hb, uint32_t tag, int o0, int len, int i0, uint32_t block) {
    const uint32_t d = pg32[1];
    if (d - 1 >= (uint32_t)P.raw.ns || P.raw.owner[d - 1] != tag) {
        if (threadIdx.x == 0) fail_page(P, pg32, block);
        return;
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    uint8_t* plane = P.raw.data + (size_t)(d - 1) * (size_t)P.raw.plane;
    for (int pr = warp; pr < 2 * P.H; pr += (int)(blockDim.x >> 5)) {
        const int h = pr >> 1;
        if (P.one && (pr & 1)) continue;  // (one stream: no V tile)
        uint8_t* tile = pg + hb + (int64_t)h * head_page(P.one) + (pr & 1) * TILE;
        const uint4 hd = make_uint4(HEAD_RAW, 0u, 0u, 0u);
        if (pr & 1) raw_tile<1, false>(P, tile, plane + (size_t)pr * 1024, hd, pg32, h, lane, o0, len, i0, 0);
        else raw_tile<0, false>(P, tile, plane + (size_t)pr * 1024, hd, pg32, h, lane, o0, len, i0, 0);
    }
}

// A block's page cleared as a page's first token clears it, for a block no request holds (a warp a block): its chunks or its slab (by owner tag) back to their pools, its headers, its
// directory and its count zero, and the block's bits in the lost bitmap.
__device__ __forceinline__ void release_body(uint8_t* pool, int64_t page_stride, int64_t nblocks, const int32_t* blocks, int H, EscapePool esc, RawPool raw, uint32_t layer_tag, uint32_t* status,
                                             uint32_t* lost, int64_t nlost) {
    const int64_t block = blocks[blockIdx.x];
    if (block < 0 || block >= nblocks) return;
    uint8_t* pg = pool + block * page_stride;
    uint32_t* pg32 = reinterpret_cast<uint32_t*>(pg + 16 * H);
    const uint32_t tag = layer_tag | (uint32_t)(block & 0xFFFFFF);
    const bool wasraw = (*(volatile uint32_t*)pg32 & PG_RAW) != 0;
    __syncwarp();
    for (int j = threadIdx.x; j < ND; j += 32) {
        const uint32_t d = pg32[1 + j];
        if (d) {  // only what the pools list as this page's
            if (wasraw) {
                if (j == 0 && d - 1 < (uint32_t)raw.ns && raw.owner[d - 1] == tag) raw_free(raw, (int)(d - 1), status);
                else atomicAdd(status + ST_STALE, 1u);
            } else if (d - 1 < (uint32_t)esc.nc && esc.owner[d - 1] == tag) {
                esc_free(esc, (int)(d - 1), status);
            } else {
                atomicAdd(status + ST_STALE, 1u);
            }
        }
        pg32[1 + j] = 0;
    }
    for (int h = threadIdx.x; h < H; h += 32) *reinterpret_cast<uint4*>(pg + 16 * h) = make_uint4(0, 0, 0, 0);
    __syncwarp();
    if (threadIdx.x == 0) {
        pg32[0] = 0;
        if (lost && 2ull * block + 1 < (uint64_t)nlost) atomicAnd(lost + ((2ull * block) >> 5), ~(3u << ((2ull * block) & 31)));
    }
}

__global__ void __launch_bounds__(32) release_kernel(uint8_t* pool, int64_t page_stride, int64_t nblocks, const int32_t* blocks, int H, EscapePool esc, RawPool raw, uint32_t layer_tag, uint32_t* status,
                                                      uint32_t* lost, int64_t nlost) {
    release_body(pool, page_stride, nblocks, blocks, H, esc, raw, layer_tag, status, lost, nlost);
}

// ---- a page moved whole (glyd_gpu_kv_save, glyd_gpu_kv_load: a sequence swapped to the host and back in its 12 bits) --------------------------------------------------------------------------
// A page's image: its bytes (pb, page_bytes), then the entries of the escape pool's chunks its directory names, CH words a chunk at the directory's place (ND places), then
// its raw slab's bytes where it is raw (plane_bytes): image_bytes apart. Only what the page holds is written: its bytes, its chunks, its slab.
__host__ __device__ constexpr int64_t image_bytes(int64_t pb, int HP) { return pb + (int64_t)ND * CH * 4 + plane_bytes(HP); }

// Page pages[j] of the pool into image j (a CTA a page, 128 threads); moved[j], where given: the bytes written. A page whose chunks or slab its pools do not list as its own
// (a page never written, a void one) has PG_VOID set in its image's `used` word: loaded, it is void.
__device__ __forceinline__ void save_body(const uint8_t* pool, int64_t page_stride, int64_t nblocks, const int32_t* pages, int64_t pb, int64_t ib, int HP, EscapePool esc, RawPool raw,
                                          uint32_t layer_tag, uint8_t* img, uint32_t* moved) {
    __shared__ int s_bad, s_n;
    const int j = blockIdx.x;
    const int64_t block = pages[j];
    uint8_t* out = img + (int64_t)j * ib;
    uint32_t* out32 = reinterpret_cast<uint32_t*>(out + 16 * HP);
    if (block < 0 || block >= nblocks) {
        if (threadIdx.x == 0) {
            out32[0] = PG_VOID;
            if (moved) moved[j] = 4u;
        }
        return;
    }
    const uint8_t* pg = pool + block * page_stride;
    const uint32_t* pg32 = reinterpret_cast<const uint32_t*>(pg + 16 * HP);
    const uint32_t tag = layer_tag | (uint32_t)(block & 0xFFFFFF);
    for (int64_t o = threadIdx.x * 16; o < pb; o += 128 * 16) *reinterpret_cast<uint4*>(out + o) = *reinterpret_cast<const uint4*>(pg + o);
    const uint32_t used = pg32[0];
    if (threadIdx.x == 0) {
        s_bad = (used & PG_VOID) != 0;
        s_n = 0;
    }
    __syncthreads();
    uint8_t* extra = out + pb;
    if (used & PG_RAW) {
        const uint32_t d = pg32[1];
        if (d - 1 < (uint32_t)raw.ns && raw.owner[d - 1] == tag) {
            const uint8_t* src = raw.data + (size_t)(d - 1) * (size_t)raw.plane;
            uint8_t* dst = extra + (int64_t)ND * CH * 4;
            for (int64_t o = threadIdx.x * 16; o < raw.plane; o += 128 * 16) *reinterpret_cast<uint4*>(dst + o) = *reinterpret_cast<const uint4*>(src + o);
            if (threadIdx.x == 0) s_n = (int)raw.plane;
        } else if (threadIdx.x == 0) {
            s_bad = 1;
        }
    } else if (!(used & PG_VOID)) {
        // (a chunk is CH words, 16 loads of 16 bytes: a thread a load)
        constexpr int Q = CH * 4 / 16;
        for (int i = threadIdx.x; i < ND * Q; i += 128) {
            const int c = i / Q, q = i % Q;
            const uint32_t d = pg32[1 + c];
            if (!d) continue;
            if (d - 1 < (uint32_t)esc.nc && esc.owner[d - 1] == tag) {
                *reinterpret_cast<uint4*>(extra + (int64_t)c * CH * 4 + q * 16) = *reinterpret_cast<const uint4*>(esc.ent + (size_t)(d - 1) * CH + q * 4);
                if (q == 0) atomicAdd(&s_n, CH * 4);
            } else {
                atomicExch(&s_bad, 1);
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        if (s_bad) out32[0] = used | PG_VOID;
        if (moved) moved[j] = (uint32_t)(pb + s_n);
    }
}

__global__ void __launch_bounds__(128) save_kernel(const uint8_t* pool, int64_t page_stride, int64_t nblocks, const int32_t* pages, int64_t pb, int64_t ib, int HP, EscapePool esc, RawPool raw,
                                                   uint32_t layer_tag, uint8_t* img, uint32_t* moved) {
    save_body(pool, page_stride, nblocks, pages, pb, ib, HP, esc, raw, layer_tag, img, moved);
}

// Image j into the page of block dst[j] (a CTA a page, 128 threads): the page's loans given back first, as a first token's clear gives them (only what the pools list as the
// page's), then its bytes, and for each chunk its image names a chunk of the escape pool taken for this block (its entries, the directory naming it), or for a raw page a slab of
// the raw pool. A page whose image is void, or that finds no chunk or slab, is void, its block named in the status's ring (as glyd_gpu_kv_write names one); a raw one in the raw ring.
__device__ __forceinline__ void load_body(const uint8_t* img, const int32_t* dst, int64_t pb, int64_t ib, uint8_t* pool, int64_t page_stride, int64_t nblocks, int HP, EscapePool esc, RawPool raw,
                                          uint32_t layer_tag, uint32_t* status, uint32_t ring) {
    __shared__ int s_void, s_slab;
    const int j = blockIdx.x;
    const int64_t block = dst[j];
    if (block < 0 || block >= nblocks) return;
    const uint8_t* in = img + (int64_t)j * ib;
    const uint32_t* in32 = reinterpret_cast<const uint32_t*>(in + 16 * HP);
    uint8_t* pg = pool + block * page_stride;
    uint32_t* pg32 = reinterpret_cast<uint32_t*>(pg + 16 * HP);
    const uint32_t tag = layer_tag | (uint32_t)(block & 0xFFFFFF);
    const bool wasraw = (*(volatile uint32_t*)pg32 & PG_RAW) != 0;
    if (threadIdx.x < ND) {
        const uint32_t d = pg32[1 + threadIdx.x];
        if (d) {
            if (wasraw) {
                if (threadIdx.x == 0 && d - 1 < (uint32_t)raw.ns && raw.owner[d - 1] == tag) raw_free(raw, (int)(d - 1), status);
                else atomicAdd(status + ST_STALE, 1u);
            } else if (d - 1 < (uint32_t)esc.nc && esc.owner[d - 1] == tag) {
                esc_free(esc, (int)(d - 1), status);
            } else {
                atomicAdd(status + ST_STALE, 1u);
            }
        }
    }
    __syncthreads();
    for (int64_t o = threadIdx.x * 16; o < pb; o += 128 * 16) *reinterpret_cast<uint4*>(pg + o) = *reinterpret_cast<const uint4*>(in + o);
    const uint32_t used = in32[0];
    if (threadIdx.x == 0) {
        s_void = (used & PG_VOID) != 0;
        s_slab = -1;
        if (!s_void && (used & PG_RAW)) {
            s_slab = raw.ns ? raw_alloc(raw, tag, tag * 2654435761u, status) : -1;
            if (s_slab < 0) s_void = 1;
        }
    }
    __syncthreads();
    const uint8_t* extra = in + pb;
    if (!s_void && (used & PG_RAW)) {
        uint8_t* plane = raw.data + (size_t)s_slab * (size_t)raw.plane;
        for (int64_t o = threadIdx.x * 16; o < raw.plane; o += 128 * 16) *reinterpret_cast<uint4*>(plane + o) = *reinterpret_cast<const uint4*>(extra + (int64_t)ND * CH * 4 + o);
        if (threadIdx.x == 0) pg32[1] = (uint32_t)s_slab + 1u;
    } else if (!s_void) {
        if (threadIdx.x < ND && in32[1 + threadIdx.x]) {
            const int c = esc.nc ? esc_alloc(esc, tag, tag * 2654435761u + threadIdx.x * 40503u, status) : -1;
            pg32[1 + threadIdx.x] = c < 0 ? 0u : (uint32_t)c + 1u;
            if (c < 0) atomicExch(&s_void, 1);
        }
        __syncthreads();
        constexpr int Q = CH * 4 / 16;
        for (int i = threadIdx.x; i < ND * Q; i += 128) {
            const int c = i / Q, q = i % Q;
            const uint32_t d = pg32[1 + c];
            if (d && in32[1 + c]) *reinterpret_cast<uint4*>(esc.ent + (size_t)(d - 1) * CH + q * 4) = *reinterpret_cast<const uint4*>(extra + (int64_t)c * CH * 4 + q * 16);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        if (s_void) {
            // (the chunks it took are this block's: its next clear gives them back)
            pg32[0] = PG_VOID;
            atomicAdd(status + ST_VOID, 1u);
            const uint32_t idx = atomicAdd(status + ST_RING_HEAD, 1u);
            status[ST_RING + idx % ring] = (uint32_t)block + 1u;
        } else if (used & PG_RAW) {
            const uint32_t idx = atomicAdd(status + ST_RAW_RING_HEAD, 1u);
            status[ST_RING + ring + idx % ring] = (uint32_t)block + 1u;
        }
    }
}

__global__ void __launch_bounds__(128) load_kernel(const uint8_t* img, const int32_t* dst, int64_t pb, int64_t ib, uint8_t* pool, int64_t page_stride, int64_t nblocks, int HP, EscapePool esc,
                                                   RawPool raw, uint32_t layer_tag, uint32_t* status, uint32_t ring) {
    load_body(img, dst, pb, ib, pool, page_stride, nblocks, HP, esc, raw, layer_tag, status, ring);
}

// A page's first token clears its headers and its arena and gives its chunks back to the pool before any entry is appended (every thread of the CTA calls it, and it ends in a barrier: a warp
// appends to every head's).
__device__ __forceinline__ void first_token_clear(const WriteParams& P, uint8_t* pg, uint32_t* pg32, uint32_t pg0, uint32_t tag, uint32_t block) {
    const bool wasraw = (pg0 & PG_RAW) != 0;
    for (int j = threadIdx.x; j < ND; j += blockDim.x) {
        const uint32_t d = pg32[1 + j];
        if (d) {  // only what the pool lists as this page's
            if (wasraw) {
                if (j == 0 && d - 1 < (uint32_t)P.raw.ns && P.raw.owner[d - 1] == tag) raw_free(P.raw, (int)(d - 1), P.status);
                else atomicAdd(P.status + ST_STALE, 1u);
            } else if (d - 1 < (uint32_t)P.esc.nc && P.esc.owner[d - 1] == tag) {
                esc_free(P.esc, (int)(d - 1), P.status);
            } else {
                atomicAdd(P.status + ST_STALE, 1u);
            }
            pg32[1 + j] = 0;
        }
    }
    if (threadIdx.x == 0) {
        pg32[0] = 0;
        if (P.lost && (P.layer_tag >> 24) == 1u && 2ull * block + 1 < (uint64_t)P.nlost) atomicAnd(P.lost + ((2ull * block) >> 5), ~(3u << ((2ull * block) & 31)));  // a new life of the block
    }
    for (int h = threadIdx.x; h < P.H; h += blockDim.x) *reinterpret_cast<uint4*>(pg + 16 * h) = make_uint4(0, 0, 0, 0);
    __syncthreads();
}

// A later run kills the entries of the tokens it writes over (a head's own, in its header, the arena and the chunks; a warp calls it for its head, where the head holds entries: cnt != 0).
__device__ __forceinline__ void kill_old(const WriteParams& P, uint32_t* pg32, uint32_t* hdr, const uint4 hv, uint32_t cnt, uint32_t used, int h, int o0, int len, int lane) {
    if (lane == 0) {
        uint32_t e[3] = {hv.y, hv.z, hv.w};
        bool changed = false;
        for (int j = 0; j < 3 && j < (int)cnt; j++) {
            if ((e[j] & LIVE) && e_tok(e[j]) >= o0 && e_tok(e[j]) < o0 + len) {
                e[j] &= ~LIVE;
                changed = true;
            }
        }
        if (changed) *reinterpret_cast<uint4*>(hdr) = make_uint4(cnt, e[0], e[1], e[2]);
    }
    if (cnt > 3) {
        for (uint32_t base = 0; base < used; base += 128) {
            uint32_t* ep[4];
            uint32_t e[4];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const uint32_t idx = base + lane + 32 * j;
                ep[j] = idx < used ? const_cast<uint32_t*>(entry_at(pg32, P.esc.ent, P.esc.nc, P.A, idx)) : nullptr;
                e[j] = ep[j] ? *ep[j] : 0u;
            }
#pragma unroll
            for (int j = 0; j < 4; j++)
                if (ep[j] && (e[j] & LIVE) && e_head(e[j]) == h && e_tok(e[j]) >= o0 && e_tok(e[j]) < o0 + len) *ep[j] = e[j] & ~LIVE;
        }
    }
}

// A run of any length, a CTA (the run's first token's), found from the slots alone: the others return at once. A warp a head (8 at a time). The kernel is a short
// chain of reads of memory no cache holds (the slots, then the page's header, counter and tile words, then the page's other entries (arena, then chunks) of a head that has
// many), so each is made one for all that can be: the slots of the run in one read (a lane a slot), and everything a head's warp needs of its
// page in one go before it kills, computes or writes.
__device__ __noinline__ void write_run(const WriteParams& P) {
    const int i = blockIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t mine = P.slots[i];
    if (mine < 0) return;
    uint8_t* pg = P.pool + (mine >> 4) * P.page_stride;
    uint32_t* pg32 = reinterpret_cast<uint32_t*>(pg + 16 * P.H);
    const uint32_t pg0 = *(volatile uint32_t*)pg32;  // flags and count before this run
    if (threadIdx.x == 0) g_need = 0;
    // lanes 0 .. 15 the slots i .. i + 15 (a run is at most a page), lane 16 the one before
    const int64_t at = lane < 16 ? (i + lane < P.N ? P.slots[i + lane] : -1) : lane == 16 && i > 0 ? P.slots[i - 1] : -1;
    const int64_t before = __shfl_sync(FULLM, at, 16);
    if (i > 0 && before >= 0 && before + 1 == mine && (mine & 15) != 0) return;
    const int o0 = (int)(mine & 15);
    const uint32_t follows = __ballot_sync(FULLM, lane < 16 && at == mine + lane) & 0xFFFFu;  // lane j: slot i + j is slot + j
    const int len = min(__ffs(~follows) - 1, 16 - o0);  // (the run's tokens: the first slot that does not follow)
    const uint32_t tag = P.layer_tag | (uint32_t)((mine >> 4) & 0xFFFFFF);
    const uint32_t block = (uint32_t)(mine >> 4);
    const int hb = hdr_bytes(P.H, P.A);
    __shared__ uint32_t s_used0, s_cnt0[32];  // entries before this run: the page's, each head-page's
    if (o0 == 0) {
        first_token_clear(P, pg, pg32, pg0, tag, block);
    } else {
        if (pg0 & PG_VOID) return;
        if (pg0 & PG_RAW) {
            raw_run(P, pg, pg32, hb, tag, o0, len, i, block);
            return;
        }
    }
    // 8 heads a round, a warp a head: its kills, then (a barrier: an entry a kill reads is not one another warp is still appending) its writes
    for (int h0 = 0; h0 < P.H; h0 += 8) {
        const int h = h0 + warp;
        const bool have = h < P.H;
        uint32_t* hdr = reinterpret_cast<uint32_t*>(pg + 16 * (have ? h : 0));
        uint8_t* tile = pg + hb + (int64_t)(have ? h : 0) * head_page(P.one);
        uint32_t cnt = 0;
        KPre kp;
        VPre vp;
        if (have) {
            // all of it at once: the header (count, three entries), the page's entry counter, the lane's values, bases and words of the K and V tiles
            const uint4 hv = o0 == 0 ? make_uint4(0, 0, 0, 0) : *reinterpret_cast<const uint4*>(hdr);
            const uint32_t used = o0 == 0 ? 0u : min(pg0 & PG_COUNT, (uint32_t)(P.A + ND * CH));
            k_load(kp, P.key + (int64_t)i * P.key_stride + h * D, P.key_stride, P.kbase + h * D, tile, o0, len, lane);
            if (!P.one) v_load(vp, P.value + (int64_t)i * P.value_stride + h * D, P.value_stride, P.vbase + h * D, tile + TILE, o0, len, lane);
            cnt = hv.x;
            if (lane == 0) s_cnt0[h] = cnt;
            if (h0 == 0 && threadIdx.x == 0) s_used0 = used;
            if (cnt != 0) kill_old(P, pg32, hdr, hv, cnt, used, h, o0, len, lane);
        }
        __syncthreads();
        if (have) {
            Appender ap{cnt, hdr, pg32, tag, &P};
            k_write(kp, tile, o0, len, lane, h, ap);
            if (!P.one) v_write(vp, tile + TILE, o0, len, lane, h, ap);
            __syncwarp();
            if (lane == 0) hdr[0] = ap.cnt;
        }
        if (h0 + 8 < P.H) __syncthreads();  // (the next round's kills read the entries this one appended)
    }
    __syncthreads();
    if (*(volatile uint32_t*)&g_need) make_raw(P, pg, pg32, hb, tag, o0, len, i, block, s_used0, s_cnt0);
}

// ---- a decode step's token ---------------------------------------------------------------------------------------------------------
// A token that is the only one of its run (a decode step writes one token a sequence: the slot before it is another page's and the one after it is not the next) goes through this path, which
// does for one token what write_run's loops do for 16 of which one is the run's: what does not wait for its slot (its keys and values, the bases) is read as the slot is, what does (the
// page's counter, its head's header, the words of the tiles it writes part of) together after it, and its codes, low bytes and entries are computed and written. The same bytes as write_run's.
struct KOne {
    uint32_t kb;      // the bases of the lane's 4 channels, a byte each
    uint32_t p0, p1;  // the lane's keys: 2 channels of a word, 16 s + 2 q + {0, 1} and + {8, 9}
    uint32_t ow;      // the code word of the pair of tokens (8 nt + g, g the token's) it writes a half of
};
__device__ __forceinline__ void k_one_in(KOne& r, const uint16_t* __restrict__ key, const uint8_t* __restrict__ base, int lane) {
    const int s = lane >> 2, q = lane & 3;
    r.kb = (uint32_t)base[16 * s + 2 * q] | ((uint32_t)base[16 * s + 2 * q + 1] << 8) | ((uint32_t)base[16 * s + 2 * q + 8] << 16) | ((uint32_t)base[16 * s + 2 * q + 9] << 24);
    r.p0 = __ldg(reinterpret_cast<const uint32_t*>(key + 16 * s + 2 * q));
    r.p1 = __ldg(reinterpret_cast<const uint32_t*>(key + 16 * s + 2 * q + 8));
}
__device__ __forceinline__ void k_one_tile(KOne& r, const uint8_t* tile, int t, int lane) {
    const int s = lane >> 2, q = lane & 3;
    r.ow = *reinterpret_cast<const uint32_t*>(tile + 2048 + (s >> 2) * 512 + (4 * (t & 7) + q) * 16 + (s & 3) * 4);
}
struct KCalc {
    uint32_t lw, cb, ent[4];  // the 4 low bytes (a word), the code bits of the token in the pair's code word, the entries
    bool esc[4], any;         // (any: some lane of the warp has one)
};
__device__ __forceinline__ void k_one_calc(KCalc& c, const KOne& r, int t, int lane, int head) {  // (what needs no word of the page: the token's codes, low bytes and entries)
    const int s = lane >> 2, q = lane & 3, nt = t >> 3, g = t & 7;
    const int owner = 4 * g + q, i = 2 * s + nt;
    const uint32_t v[4] = {r.p0 & 0xFFFFu, r.p0 >> 16, r.p1 & 0xFFFFu, r.p1 >> 16};
    c.lw = 0;
    c.cb = 0;
    bool any = false;
#pragma unroll
    for (int m = 0; m < 4; m++) {
        const uint32_t b = (r.kb >> (8 * m)) & 0xFFu;
        uint32_t x;
        const uint32_t code = code_of(v[m], b, c.esc[m], x);
        any |= c.esc[m];
        c.ent[m] = c.esc[m] ? entry(owner, i, m, x, t, 0, head) : 0u;
        c.lw |= (v[m] & 255u) << (8 * m);
        if (nt == 0) c.cb |= (((code >> 3) << 7) | (code & 7u)) << (8 * m);
        else c.cb |= (((code >> 3) << 3) << (8 * m)) | ((code & 7u) << (8 * ((m + 3) & 3) + 4));
    }
    c.any = any;
}
__device__ __forceinline__ void k_one_store(const KCalc& c, const KOne& r, uint8_t* tile, int t, int lane, Appender& ap) {
    const int s = lane >> 2, q = lane & 3, nt = t >> 3, g = t & 7;
    const int owner = 4 * g + q, i = 2 * s + nt;
    *reinterpret_cast<uint32_t*>(tile + (i >> 2) * 512 + owner * 16 + (i & 3) * 4) = c.lw;
    *reinterpret_cast<uint32_t*>(tile + 2048 + (s >> 2) * 512 + owner * 16 + (s & 3) * 4) = (r.ow & ~(nt ? 0x78787878u : 0x87878787u)) | c.cb;
    if (__any_sync(FULLM, c.any)) {
#pragma unroll
        for (int m = 0; m < 4; m++) ap.add(c.ent[m], c.esc[m], lane);
    }
}

struct VOne {
    uint32_t bs[4];  // the bases of the lane's 4 channels g + 8 (4 ig + jj)
    uint32_t vv[4];  // its values
    uint4 lw;        // the 4 low words (4 ig .. 4 ig + 3) it writes a byte of, of owner 4 g + (the token's group)
    uint2 cw;        // the 2 code words (2 ig, 2 ig + 1)
};
__device__ __forceinline__ void v_one_in(VOne& r, const uint16_t* __restrict__ value, const uint8_t* __restrict__ base, int lane) {
    const int g = lane & 7, ig = lane >> 3;
#pragma unroll
    for (int jj = 0; jj < 4; jj++) {
        r.bs[jj] = base[g + 8 * (4 * ig + jj)];
        r.vv[jj] = __ldg(value + g + 8 * (4 * ig + jj));
    }
}
__device__ __forceinline__ void v_one_tile(VOne& r, const uint8_t* tile, int t, int lane) {  // (tile: the V tile)
    const int g = lane & 7, ig = lane >> 3, owner = 4 * g + ((t & 7) >> 1);
    r.lw = *reinterpret_cast<const uint4*>(tile + ig * 512 + owner * 16);
    r.cw = *reinterpret_cast<const uint2*>(tile + 2048 + (ig >> 1) * 512 + owner * 16 + 2 * (ig & 1) * 4);
}
struct VCalc {
    uint32_t lbits[4], cb[2], ent[4];  // the token's byte of each of the 4 low words, its bits of the 2 code words, the entries
    bool esc[4], any;
};
__device__ __forceinline__ void v_one_calc(VCalc& c, const VOne& r, int t, int lane, int head) {
    const int g = lane & 7, ig = lane >> 3;
    const int owner = 4 * g + ((t & 7) >> 1), m = (t & 1) | ((t >> 3) << 1);  // (token t is slot m of its group of 4: 2 qg + (m & 1) + 8 (m >> 1))
    c.cb[0] = c.cb[1] = 0;
    bool any = false;
#pragma unroll
    for (int jj = 0; jj < 4; jj++) {
        uint32_t x;
        const uint32_t code = code_of(r.vv[jj], r.bs[jj], c.esc[jj], x);
        any |= c.esc[jj];
        c.ent[jj] = c.esc[jj] ? entry(owner, 4 * ig + jj, m, x, t, 1, head) : 0u;
        c.lbits[jj] = (r.vv[jj] & 255u) << (8 * m);
        if ((jj & 1) == 0) c.cb[jj >> 1] |= (((code >> 3) << 7) | (code & 7u)) << (8 * m);
        else c.cb[jj >> 1] |= (((code >> 3) << 3) << (8 * m)) | ((code & 7u) << (8 * ((m + 3) & 3) + 4));
    }
    c.any = any;
}
__device__ __forceinline__ void v_one_store(const VCalc& c, const VOne& r, uint8_t* tile, int t, int lane, Appender& ap) {
    const int g = lane & 7, ig = lane >> 3;
    const int owner = 4 * g + ((t & 7) >> 1), m = (t & 1) | ((t >> 3) << 1);
    const uint32_t lmask = 0xFFu << (8 * m), cmask = (0x87u << (8 * m)) | (0x08u << (8 * m)) | (0x70u << (8 * ((m + 3) & 3)));
    *reinterpret_cast<uint4*>(tile + ig * 512 + owner * 16) = make_uint4((r.lw.x & ~lmask) | c.lbits[0], (r.lw.y & ~lmask) | c.lbits[1], (r.lw.z & ~lmask) | c.lbits[2], (r.lw.w & ~lmask) | c.lbits[3]);
    *reinterpret_cast<uint2*>(tile + 2048 + (ig >> 1) * 512 + owner * 16 + 2 * (ig & 1) * 4) = make_uint2((r.cw.x & ~cmask) | c.cb[0], (r.cw.y & ~cmask) | c.cb[1]);
    if (__any_sync(FULLM, c.any)) {
#pragma unroll
        for (int jj = 0; jj < 4; jj++) ap.add(c.ent[jj], c.esc[jj], lane);
    }
}

// Whether another run of the call names the page of the run that begins at token i (its slot `mine`): a call's runs are written by CTAs at once, so two runs of
// one page would clear, fill and make raw that page together: a chunk or a slab given back twice, or taken and named by no page, and the pools' counts no longer
// what is taken (on an A10, runs of one call into one page left a group's count of raw slabs taken 1 to 5 below zero and of escape chunks 6,137 to 6,240
// below, unsigned: 2^32 less those). Each such run writes nothing; the status counts them (ST_CLASH) and names the page (its block + 1 in ST_CLASH_PAGE), so
// the caller can refuse the call's pass. Every thread of the CTA calls it (a barrier). A call whose tokens of one page follow one another, as glyd_gpu.h asks,
// has none.
__device__ __noinline__ bool run_clashes(const WriteParams& P, int i, int64_t mine) {
    const int64_t page = mine >> 4;
    int found = 0;
    for (int j = threadIdx.x; j < P.N; j += blockDim.x) {
        const int64_t s = P.slots[j];
        if (j == i || s < 0 || (s >> 4) != page) continue;
        const int64_t b = j > 0 ? P.slots[j - 1] : -1;
        found |= !(j > 0 && b >= 0 && b + 1 == s && (s & 15) != 0);  // (token j begins a run of this page)
    }
    const bool clash = __syncthreads_or(found) != 0;
    if (clash && threadIdx.x == 0) {
        atomicAdd(P.status + ST_CLASH, 1u);
        atomicExch(P.status + ST_CLASH_PAGE, (uint32_t)page + 1u);
    }
    return clash;
}

// The write kernel: a CTA a token. A token that is the only one of its run goes through the decode step's path (the rest of this kernel); a longer run's first token runs write_run; the
// run's other tokens return at once. (P.lean 0: write_run for every run.) A row with no slot (a bucket's padding row, whose keys and values nobody made) is not read. A run whose page
// another run of the call names writes nothing (run_clashes).
__device__ __forceinline__ void write_body(const WriteParams& P) {
    const int i = blockIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t mine = P.slots[i];
    if (mine < 0) return;
    const int64_t prv = i > 0 ? P.slots[i - 1] : -1, nxt = i + 1 < P.N ? P.slots[i + 1] : -1;
    KOne ko;
    VOne vo;
    KCalc kc;
    VCalc vc;
    if (warp < P.H) {  // (the first round's head)
        k_one_in(ko, P.key + (int64_t)i * P.key_stride + warp * D, P.kbase + warp * D, lane);
        if (!P.one) v_one_in(vo, P.value + (int64_t)i * P.value_stride + warp * D, P.vbase + warp * D, lane);
    }
    if (i > 0 && prv >= 0 && prv + 1 == mine && (mine & 15) != 0) return;  // (not the first token of its run)
    if (run_clashes(P, i, mine)) return;
    if (!P.lean || (nxt == mine + 1 && (mine & 15) != 15)) {  // (a run of more than one token)
        write_run(P);
        return;
    }
    const int o0 = (int)(mine & 15);
    const uint32_t block = (uint32_t)(mine >> 4), tag = P.layer_tag | (block & 0xFFFFFFu);
    uint8_t* pg = P.pool + (int64_t)block * P.page_stride;
    uint32_t* pg32 = reinterpret_cast<uint32_t*>(pg + 16 * P.H);
    const uint32_t pg0 = *(volatile uint32_t*)pg32;  // flags and count before this token
    if (threadIdx.x == 0) g_need = 0;
    const int hb = hdr_bytes(P.H, P.A);
    __shared__ uint32_t s_used0, s_cnt0[32];  // entries before this token: the page's, each head-page's
    for (int h0 = 0; h0 < P.H; h0 += 8) {
        const int h = h0 + warp;
        const bool have = h < P.H;
        uint32_t* hdr = reinterpret_cast<uint32_t*>(pg + 16 * (have ? h : 0));
        uint8_t* tile = pg + hb + (int64_t)(have ? h : 0) * head_page(P.one);
        uint32_t cnt = 0;
        uint4 hv = make_uint4(0, 0, 0, 0);
        const uint32_t used = o0 == 0 ? 0u : min(pg0 & PG_COUNT, (uint32_t)(P.A + ND * CH));
        if (have) {
            if (h0 > 0) {
                k_one_in(ko, P.key + (int64_t)i * P.key_stride + h * D, P.kbase + h * D, lane);
                if (!P.one) v_one_in(vo, P.value + (int64_t)i * P.value_stride + h * D, P.vbase + h * D, lane);
            }
            if (o0 != 0) hv = *reinterpret_cast<const uint4*>(hdr);
            k_one_tile(ko, tile, o0, lane);
            if (!P.one) v_one_tile(vo, tile + TILE, o0, lane);
            k_one_calc(kc, ko, o0, lane, h);  // (while the page's words are on their way)
            if (!P.one) v_one_calc(vc, vo, o0, lane, h);
        }
        if (h0 == 0) {  // (the page: a first token clears it, any other finds it void or raw, or goes on)
            if (o0 == 0) {
                first_token_clear(P, pg, pg32, pg0, tag, block);
            } else if (pg0 & (PG_VOID | PG_RAW)) {
                if (pg0 & PG_VOID) return;
                raw_run(P, pg, pg32, hb, tag, o0, 1, i, block);
                return;
            }
        }
        if (have) {
            cnt = hv.x;
            if (lane == 0) s_cnt0[h] = cnt;
            if (h0 == 0 && threadIdx.x == 0) s_used0 = used;
            if (cnt != 0) kill_old(P, pg32, hdr, hv, cnt, used, h, o0, 1, lane);
        }
        __syncthreads();
        if (have) {
            Appender ap{cnt, hdr, pg32, tag, &P};
            k_one_store(kc, ko, tile, o0, lane, ap);
            if (!P.one) v_one_store(vc, vo, tile + TILE, o0, lane, ap);
            __syncwarp();
            if (lane == 0 && ap.cnt != cnt) hdr[0] = ap.cnt;
        }
        if (h0 + 8 < P.H) __syncthreads();
    }
    __syncthreads();
    if (*(volatile uint32_t*)&g_need) make_raw(P, pg, pg32, hb, tag, o0, 1, i, block, s_used0, s_cnt0);
}

__global__ void __launch_bounds__(256) write_kernel(const __grid_constant__ WriteParams P) {
    write_body(P);
}

// ---- reading pages back to bf16: out [page][token][head][channel] for the pages listed (one warp a head-page) -----------------
__device__ __forceinline__ void unpack_body(const uint8_t* __restrict__ pool, int64_t page_stride, const int32_t* __restrict__ pages, int n, const uint8_t* __restrict__ kbase,
                                                     const uint8_t* __restrict__ vbase, int H, int A, const uint32_t* __restrict__ ent, int nc, const RawPool R, uint16_t* __restrict__ out_k, uint16_t* __restrict__ out_v, int one) {
    const int wl = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (wl >= n * H) return;
    const int lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
    const int j = wl / H, h = wl % H;
    if (pages[j] < 0) return;  // (a place of the list no page is in: nothing written for it)
    const uint8_t* pg = pool + (int64_t)pages[j] * page_stride;
    const uint4 hd = *reinterpret_cast<const uint4*>(pg + 16 * h);
    const uint32_t* pg32 = reinterpret_cast<const uint32_t*>(pg + 16 * H);
    const uint8_t* tile = pg + hdr_bytes(H, A) + (int64_t)h * head_page(one != 0);
#pragma unroll
    for (int kind = 0; kind < 2; kind++) {
        if (kind == 1 && one) break;  // (one stream: the keys' layout alone)
        uint4 f[8];
        Raw r;
        raw_load(r, tile + kind * TILE, lane);
        if (kind == 0) {
            uint32_t hb[8];
            load_bases<0>(kbase + h * D, lane, hb);
            decode_tile<0>(r, hb, hd, pg32, ent, nc, A, R, h, lane, f);
        } else {
            uint32_t hb[16];
            load_bases<1>(vbase + h * D, lane, hb);
            decode_tile<1>(r, hb, hd, pg32, ent, nc, A, R, h, lane, f);
        }
        uint16_t* out = (kind == 0 ? out_k : out_v) + (int64_t)j * PT * H * D;
#pragma unroll
        for (int i = 0; i < 16; i++) {
            const uint4& c = f[i >> 1];
            const uint32_t w0 = (i & 1) ? c.y : c.x, w1 = (i & 1) ? c.w : c.z;  // group i's two pairs
#pragma unroll
            for (int m = 0; m < 4; m++) {
                int tok, ch;
                if (kind == 0) coord<0>(g, q, i, m, tok, ch);
                else coord<1>(g, q, i, m, tok, ch);
                const uint32_t w = m < 2 ? w0 : w1;
                out[((int64_t)tok * H + h) * D + ch] = (uint16_t)((m & 1) ? (w >> 16) : (w & 0xFFFFu));
            }
        }
    }
}

__global__ void __launch_bounds__(128) unpack_kernel(const uint8_t* __restrict__ pool, int64_t page_stride, const int32_t* __restrict__ pages, int n, const uint8_t* __restrict__ kbase,
                                                     const uint8_t* __restrict__ vbase, int H, int A, const uint32_t* __restrict__ ent, int nc, const RawPool R, uint16_t* __restrict__ out_k, uint16_t* __restrict__ out_v, int one) {
    unpack_body(pool, page_stride, pages, n, kbase, vbase, H, A, ent, nc, R, out_k, out_v, one);
}

// ---- decoding pages into a paged bf16 cache, as vLLM's FlashAttention reads one ----------------------------------------------------
// The pages listed (pages[j]) decoded into the block dst[j] of `out` [blocks][16 tokens][H][2 D] bf16: token t of head h of block b has its keys at ((b 16 + t) H + h) 2 D and its
// values D after (the cache the prefill rows' FlashAttention runs on: a context's pages made bf16 before the step's own tokens are put after it). One warp a head-page, as unpack.
// HALVES 2: a head of 256 channels, held as the page's heads 2 h and 2 h + 1 (its channels 0-127 and 128-255): the head-page 2 h + f's keys at channels 128 f of head h's 256, its values 256 after.
template <int HALVES>
__device__ __forceinline__ void gather_body(const uint8_t* __restrict__ pool, int64_t page_stride, const int32_t* __restrict__ pages, const int32_t* __restrict__ dst, int n,
                                                     const uint8_t* __restrict__ kbase, const uint8_t* __restrict__ vbase, int H, int A, const uint32_t* __restrict__ ent, int nc, const RawPool R,
                                                     uint16_t* __restrict__ out) {
    const int wl = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (wl >= n * H) return;
    const int lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
    const int j = wl / H, h = wl % H;
    if (pages[j] < 0) return;  // (a place of the list no page is in: the plan's lists are as long as the most the rows can need)
    const uint8_t* pg = pool + (int64_t)pages[j] * page_stride;
    const uint4 hd = *reinterpret_cast<const uint4*>(pg + 16 * h);
    const uint32_t* pg32 = reinterpret_cast<const uint32_t*>(pg + 16 * H);
    const uint8_t* tile = pg + hdr_bytes(H, A) + (int64_t)h * HEADPAGE;
    uint16_t* row0 = out + ((int64_t)dst[j] * PT * H + (h - h % HALVES)) * 2 * D + (h % HALVES) * D;  // (token t's row is t H 2 D further)
    {
        uint4 f[8];
        Raw r;
        raw_load(r, tile, lane);
        uint32_t hb[8];
        load_bases<0>(kbase + h * D, lane, hb);
        decode_tile<0>(r, hb, hd, pg32, ent, nc, A, R, h, lane, f);
#pragma unroll
        for (int i = 0; i < 16; i++) {  // keys: group i is token 8 (i & 1) + g, channels 16 (i >> 1) + 2 q + {0, 1} and + {8, 9}: two words
            const uint4& c = f[i >> 1];
            const uint32_t w0 = (i & 1) ? c.y : c.x, w1 = (i & 1) ? c.w : c.z;
            uint16_t* p = row0 + (int64_t)(8 * (i & 1) + g) * H * 2 * D + 16 * (i >> 1) + 2 * q;
            *reinterpret_cast<uint32_t*>(p) = w0;
            *reinterpret_cast<uint32_t*>(p + 8) = w1;
        }
    }
    {
        uint4 f[8];
        Raw r;
        raw_load(r, tile + TILE, lane);
        uint32_t hb[16];
        load_bases<1>(vbase + h * D, lane, hb);
        decode_tile<1>(r, hb, hd, pg32, ent, nc, A, R, h, lane, f);
#pragma unroll
        for (int i = 0; i < 16; i++) {  // values: group i is channel g + 8 i, tokens 2 q + {0, 1} and 8 + 2 q + {0, 1}
            const uint4& c = f[i >> 1];
            const uint32_t w0 = (i & 1) ? c.y : c.x, w1 = (i & 1) ? c.w : c.z;
            uint16_t* p = row0 + HALVES * D + g + 8 * i;
            p[(int64_t)(2 * q) * H * 2 * D] = (uint16_t)(w0 & 0xFFFFu);
            p[(int64_t)(2 * q + 1) * H * 2 * D] = (uint16_t)(w0 >> 16);
            p[(int64_t)(2 * q + 8) * H * 2 * D] = (uint16_t)(w1 & 0xFFFFu);
            p[(int64_t)(2 * q + 9) * H * 2 * D] = (uint16_t)(w1 >> 16);
        }
    }
}

template <int HALVES>
__global__ void __launch_bounds__(128) gather_kernel(const uint8_t* __restrict__ pool, int64_t page_stride, const int32_t* __restrict__ pages, const int32_t* __restrict__ dst, int n,
                                                     const uint8_t* __restrict__ kbase, const uint8_t* __restrict__ vbase, int H, int A, const uint32_t* __restrict__ ent, int nc, const RawPool R,
                                                     uint16_t* __restrict__ out) {
    gather_body<HALVES>(pool, page_stride, pages, dst, n, kbase, vbase, H, A, ent, nc, R, out);
}

// ---- a step's own keys and values into the bf16 staging cache ------------------------------------------------------------------------
// Token t's keys and values (rows of [H][DD] bf16 at key + t key_stride, value + t value_stride) into the slot slots[t] of `out` [blocks][16][H][2 DD]: its keys at (slot H + h) 2 DD and its values DD
// after; a negative slot (or one past the cache) is skipped, so a plan's group takes the tokens of its rows and leaves the others' (one CTA a token, 4-byte words).
template <int DD>
__global__ void __launch_bounds__(128) stage_put_kernel(const uint16_t* __restrict__ key, int64_t key_stride, const uint16_t* __restrict__ value, int64_t value_stride, const int64_t* __restrict__ slots,
                                                        int64_t nslots, int H, uint16_t* __restrict__ out) {
    constexpr int D = DD;
    const int64_t t = blockIdx.x, slot = slots[t];
    if (slot < 0 || slot >= nslots) return;
    const uint32_t* kp = reinterpret_cast<const uint32_t*>(key + t * key_stride);
    const uint32_t* vp = reinterpret_cast<const uint32_t*>(value + t * value_stride);
    uint32_t* row = reinterpret_cast<uint32_t*>(out + slot * H * 2 * D);
    for (int i = threadIdx.x; i < H * D / 2; i += blockDim.x) {
        const int h = i / (D / 2), c = i % (D / 2);  // (head, word of its D channels)
        row[h * D + c] = kp[i];
        row[h * D + D / 2 + c] = vp[i];
    }
}

// ---- decode attention ---------------------------------------------------------------------------------------------------
// One query token a sequence over its pages, G query heads (up to 8) to a kv head, fp32 softmax and accumulation, split over CTAs
// (flash decoding). A CTA is (split, kv head, sequence): NW warps, each the pages split * ppc + w, + NW, ..., one online softmax a
// warp; the CTA's warps are merged through shared memory into one partial (max, sum, output), which goes to a workspace that
// combine_kernel merges over the splits. The scores are S^T = K Q^T (K's tile the A operand,
// the G heads the columns of the 8-column B operand) and the output O^T = V^T P^T (V^T the A operand, P^T the scores' fragments by
// movmatrix): both straight from registers, with P rounded to bf16 for the second product as FlashAttention does.
// A head of 256 channels (HALVES 2) is two of the page's heads of 128 (2 h: channels 0-127, 2 h + 1: 128-255): a page is two warps', each its half's tiles, their
// scores summed through shared memory before the softmax (the pair's two warps hold the same sums, maxima and weights), each warp the output of its half's channels.
struct AttnParams {
    const __nv_bfloat16* q;  // sequence b's queries at q + b * q_stride: [H * G][D HALVES]
    int64_t q_stride;
    const uint8_t* pool;
    int64_t page_stride;
    const uint8_t* kbase;  // [H][D]
    const uint8_t* vbase;
    const int32_t* bt;  // [B][bt_stride]: page index of a sequence's page
    int64_t bt_stride;
    const int32_t* slen;  // [B]
    const uint32_t* ent;  // the escape pool's entries (chunks of CH), where a page's directory names them, and its chunks
    int nc;
    RawPool raw;
    float* ws_ml;         // [B * H][splits][G][2]
    float* ws_o;          // [B * H][splits][G][D HALVES]
    int H, G, A, hdrb, splits;  // splits: the grid's x, the most CTAs a (sequence, kv head) is split over (the partials' stride)
    int HP;                     // the page's heads of D channels: H HALVES
    float sl2;  // scale * log2(e)
    int ppc_min;  // the plan's least pages a CTA
    int pfloor;   // the plan's CTAs a sequence has at least (as far as it has 4 pages for each)
    uint32_t* cnt;  // [B * H] the CTAs of a (sequence, head) that have written their partials (zero between calls): the last merges them (FUSE)
    __nv_bfloat16* out;
    int64_t out_stride;
    const int32_t* first;  // [B] or null: the first token a sequence's query attends to (a sliding window's start)
    int round;             // 1: a score is the bf16 of its dot product, before the scale (a bf16 matmul's output)
    const __nv_bfloat16* kc;  // a bf16 cache in place of the pages (null: the pages): [places][H][D HALVES], a token's place through bt, else modulo cap
    const __nv_bfloat16* vc;
    int cap;
};

// A sequence's tokens attended to: from `first` (0 where there is none, at most the length) to `len`, the page that holds the first one on.
__device__ __forceinline__ int first_of(const int32_t* first, int b, int len) {
    return first ? min(max(first[b], 0), len) : 0;
}

// The plan, from a sequence's own length alone (so a sequence's partition, and its output's bits, do not depend on the batch it is in, nor on a batch padded for a CUDA graph): at
// least ppc_min pages a CTA (16 by default: 4 a warp), at most `splits` CTAs, each the same share of the pages (ceil); a sequence of no tokens has one CTA, whose partial is empty.
// pfloor (1: none): a sequence has at least that many CTAs where its pages are 4 for each (a page a warp), however few ppc_min would give it: a batch of few sequences leaves most of a
// big GPU's SMs without a CTA otherwise. It is the same number for every call whose outputs are compared (the host's, from the batch's size and the GPU's: kv_floor).
constexpr int PPC_MIN = NW * 4;
__device__ __forceinline__ int plan_splits(int np, int splits, int ppc_min, int pfloor) {
    return min(splits, max(max(1, (np + ppc_min - 1) / ppc_min), min(pfloor, (np + NW - 1) / NW)));
}

// The partials of a (sequence, kv head) merged for query head `head`, 128 of its DO channels at a time: a lane 4 channels (a sequence of no tokens gives zeros). The partials' reads are L2's (another
// CTA's, or another kernel's), so they are made in as few waits as there are: the partials' maxima and sums a lane a partial (32 at a time) and the max by shuffles (a max does not depend on its order),
// then the weights and the sums in the partials' order, with the vectors of 16 partials in flight.
template <int DO>
__device__ __forceinline__ void merge_head(const float* __restrict__ mlp, const float* __restrict__ wop, __nv_bfloat16* __restrict__ op, int G, int head, int nparts, int lane) {
#pragma unroll 1
    for (int part = 0; part < DO / D; part++) {
        float M = -INFINITY;
        for (int c = 0; c < nparts; c += 32)
            if (c + lane < nparts) M = fmaxf(M, __ldcg(mlp + ((c + lane) * G + head) * 2));
#pragma unroll
        for (int o = 16; o; o >>= 1) M = fmaxf(M, __shfl_xor_sync(FULLM, M, o));
        float L = 0.f, a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        if (M != -INFINITY) {
            for (int c = 0; c < nparts; c += 32) {
                const int n = min(32, nparts - c);
                float2 ml = make_float2(-INFINITY, 0.f);
                if (lane < n) ml = __ldcg(reinterpret_cast<const float2*>(mlp + ((c + lane) * G + head) * 2));
                const float wl = ex2(ml.x - M), ll = ml.y;  // (lane j: partial c + j's weight and sum)
#pragma unroll 16
                for (int j = 0; j < n; j++) {
                    const float w = __shfl_sync(FULLM, wl, j);
                    L += w * __shfl_sync(FULLM, ll, j);
                    const float4 v = __ldcg(reinterpret_cast<const float4*>(wop + ((size_t)(c + j) * G + head) * DO + part * D + lane * 4));
                    a0 += w * v.x;
                    a1 += w * v.y;
                    a2 += w * v.z;
                    a3 += w * v.w;
                }
            }
        }
        const float inv = L > 0.f ? 1.f / L : 0.f;
        __nv_bfloat16* p = op + (int64_t)head * DO + part * D + lane * 4;
        *reinterpret_cast<__nv_bfloat162*>(p) = __floats2bfloat162_rn(a0 * inv, a1 * inv);
        *reinterpret_cast<__nv_bfloat162*>(p + 2) = __floats2bfloat162_rn(a2 * inv, a3 * inv);
    }
}

struct Buf {  // a page's raw reads
    Raw kr, vr;
    uint4 hd;
    const uint32_t* pg;  // the page's `used` word, directory and arena
};
__device__ __forceinline__ void load_buf(Buf& c, const AttnParams& P, int b, int h, int pg, int lane) {
    const uint8_t* page = P.pool + (int64_t)__ldg(P.bt + (int64_t)b * P.bt_stride + pg) * P.page_stride;
    const uint8_t* kt = page + P.hdrb + (int64_t)h * HEADPAGE;
    raw_load(c.kr, kt, lane);
    raw_load(c.vr, kt + TILE, lane);
    c.hd = ldg_stream<0>(page + 16 * h);
    c.pg = reinterpret_cast<const uint32_t*>(page + 16 * P.HP);
}

// A page of a bf16 cache (AttnParams kc, vc; `blk` the page's block where there is a table): the words of the page's tokens g and g + 8, of the warp's head h and half hf, that a lane's
// fragments take, keys and values alike (words 8 i + q and 8 i + 4 + q of the half's 128 channels). kw is the scores' A fragments as decode_tile<0> gives them; vw holds the values'
// words in the same places, each an 8 x 8 block of tokens by channels that one transpose makes V^T's fragment (decode_tile<1>'s). Every word is read at once, the tokens outside
// the ones attended to as well (their places are in the page's block, or the ring, all the same): their scores are masked and their values zeroed after, as the pages' are.
template <int DO>
__device__ __forceinline__ void bf16_words(const AttnParams& P, int64_t blk, int pg, int h, int hf, int g, int q, uint4 (&kw)[8], uint4 (&vw)[8]) {
    const int64_t t0 = 16 * (int64_t)pg + g;
    const int64_t p0 = P.bt ? blk * 16 + g : t0 % P.cap, p1 = P.bt ? blk * 16 + g + 8 : (t0 + 8) % P.cap;
    const int64_t at = (int64_t)h * DO + hf * D;
    const uint32_t* k0 = reinterpret_cast<const uint32_t*>(P.kc + p0 * P.H * DO + at);
    const uint32_t* k1 = reinterpret_cast<const uint32_t*>(P.kc + p1 * P.H * DO + at);
    const uint32_t* v0 = reinterpret_cast<const uint32_t*>(P.vc + p0 * P.H * DO + at);
    const uint32_t* v1 = reinterpret_cast<const uint32_t*>(P.vc + p1 * P.H * DO + at);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        kw[i] = make_uint4(__ldg(k0 + 8 * i + q), __ldg(k1 + 8 * i + q), __ldg(k0 + 8 * i + 4 + q), __ldg(k1 + 8 * i + 4 + q));
        vw[i] = make_uint4(__ldg(v0 + 8 * i + q), __ldg(v0 + 8 * i + 4 + q), __ldg(v1 + 8 * i + q), __ldg(v1 + 8 * i + 4 + q));
    }
}

// PIPE: the next page's loads are issued before the page in hand is computed (two pages of raw data a warp in registers). FUSE: the last CTA of a (sequence, kv head) to write its partial merges
// them all (the arithmetic of combine_kernel, which a launch of its own does otherwise). HALVES 2 and 4: a head of 256 or 512 channels, a page two or four warps' (AttnParams).
// BF16: the keys and values in a bf16 cache (kc, vc) in place of the pages: an instance of its own, with no page decode in it, that reads a page's words at once (bf16_words) and its next
// page's block a page ahead; the same fragments, so the same arithmetic and bits as the pages'.
// The grid is (H, B, splits): a call's CTAs are dispatched split by split, so those of the first splits (the ones that have pages, where a sequence is short) start before the ones past
// their sequence's plan, which return at once.
template <bool PIPE, bool FUSE, int HALVES, bool BF16 = false>
__device__ __forceinline__ void attn_body(const AttnParams P) {
    constexpr int DO = D * HALVES;   // a query head's channels
    constexpr int NP = NW / HALVES;  // pages a CTA has in hand at once
    const int split = blockIdx.z, h = blockIdx.x, b = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
    const int hf = warp % HALVES, pw = warp / HALVES;  // the warp's half of the channels, and its page's place among the CTA's
    const int hp = h * HALVES + hf;                    // the page's head the warp reads
    const int len = P.slen[b];
    const int fst = first_of(P.first, b, len), ps = fst >> 4;  // (a window: its first token, and the page that holds it)
    const int np = fst < len ? ((len + 15) >> 4) - ps : 0;   // the pages attended to
    const int sb = plan_splits(np, P.splits, P.ppc_min, P.pfloor);  // the CTAs this sequence has: past them, no partial (the combine reads sb of them)
    if (split >= sb) return;
    const int ppc = (np + sb - 1) / sb;
    const int p0 = ps + min(np, split * ppc), p1 = ps + min(np, split * ppc + ppc);
    const int G = P.G;

    uint32_t qb0[8], qb1[8];  // the B operand of the scores: column g = query head g (zero past G), rows the step's channels 2q, 2q + 1 and 2q + 8, 2q + 9
    {
        const uint32_t* qp = reinterpret_cast<const uint32_t*>(P.q + (int64_t)b * P.q_stride + (int64_t)(h * G + (g < G ? g : 0)) * DO + hf * D);
#pragma unroll
        for (int s = 0; s < 8; s++) {
            qb0[s] = g < G ? qp[8 * s + q] : 0u;
            qb1[s] = g < G ? qp[8 * s + q + 4] : 0u;
        }
    }
    uint32_t kb[8], vb[16];
    if constexpr (!BF16) {
        load_bases<0>(P.kbase + hp * D, lane, kb);
        load_bases<1>(P.vbase + hp * D, lane, vb);
    }
    // a lane's softmax state is for heads 2q (index 0) and 2q + 1 (index 1); the lanes of the other g hold the same heads' other tokens
    float m_run0 = -INFINITY, m_run1 = -INFINITY, l_run0 = 0.f, l_run1 = 0.f;
    float o[8][4];  // O^T: channels 16i + g (o[i][0], o[i][1]) and 16i + 8 + g (o[i][2], o[i][3]) of heads 2q, 2q + 1 (of the warp's half)
#pragma unroll
    for (int i = 0; i < 8; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;

    Buf cur, nxt;  // the page in hand, and (PIPE) the next one
    int pg = p0 + pw;
    [[maybe_unused]] int par = 0;  // (HALVES 2: the scores' buffer of this page, one of two in turn)
    [[maybe_unused]] int64_t blk = 0;  // (BF16 with a table: the block of the page in hand)
    if constexpr (BF16) {
        if (P.bt && pg < p1) blk = __ldg(P.bt + (int64_t)b * P.bt_stride + pg);
    } else if constexpr (PIPE) {
        if (pg < p1) load_buf(cur, P, b, hp, pg, lane);
    }
    for (; pg < p1; pg += NP) {
        const int valid = len - 16 * pg, lo = fst - 16 * pg;  // the page's tokens attended to: lo .. valid
        uint4 f[8];
        [[maybe_unused]] uint4 vw[8];  // (BF16: the values' words, transposed into V^T's fragments when the values are taken)
        if constexpr (BF16) {
            // (the next page's block read first, for the next turn: the page's words then wait on no table read)
            const int64_t nblk = P.bt && pg + NP < p1 ? (int64_t)__ldg(P.bt + (int64_t)b * P.bt_stride + pg + NP) : 0;
            bf16_words<DO>(P, blk, pg, h, hf, g, q, f, vw);
            blk = nblk;
        } else {
            if constexpr (PIPE) {
                if (pg + NP < p1) load_buf(nxt, P, b, hp, pg + NP, lane);
            } else {
                load_buf(cur, P, b, hp, pg, lane);
            }
            decode_tile<0>(cur.kr, kb, cur.hd, cur.pg, P.ent, P.nc, P.A, P.raw, hp, lane, f);
        }

        float s[4] = {0.f, 0.f, 0.f, 0.f};  // tokens g (s[0], s[1]) and g + 8 (s[2], s[3]) of heads 2q, 2q + 1
#pragma unroll
        for (int i = 0; i < 8; i++) mma(s, f[i].x, f[i].y, f[i].z, f[i].w, qb0[i], qb1[i]);
        if constexpr (HALVES >= 2) {  // the halves' scores summed: each of a page's warps adds the others' to its own, in the order of the halves, so that all of them hold the same sums bit for bit
            __shared__ float4 s_x[2][NW][32];
            s_x[par][warp][lane] = make_float4(s[0], s[1], s[2], s[3]);
#if defined(__HIP_PLATFORM_AMD__)
            __trap();  // (no named barriers on RDNA: heads of 256 channels and more are CUDA's alone)
#else
            asm volatile("bar.sync %0, %1;" ::"r"(1 + pw), "r"(HALVES * 32) : "memory");  // (the page's warps alone; a buffer is written again two pages on, after the warps have passed the next barrier)
#endif
            if constexpr (HALVES == 2) {
                const float4 x = s_x[par][warp ^ 1][lane];  // (a + b is b + a, bit for bit)
                s[0] += x.x;
                s[1] += x.y;
                s[2] += x.z;
                s[3] += x.w;
            } else {  // 4 halves: (h0 + h1) + (h2 + h3), the same expression in every warp
                const float4 x0 = s_x[par][pw * HALVES][lane], x1 = s_x[par][pw * HALVES + 1][lane], x2 = s_x[par][pw * HALVES + 2][lane], x3 = s_x[par][pw * HALVES + 3][lane];
                s[0] = (x0.x + x1.x) + (x2.x + x3.x);
                s[1] = (x0.y + x1.y) + (x2.y + x3.y);
                s[2] = (x0.z + x1.z) + (x2.z + x3.z);
                s[3] = (x0.w + x1.w) + (x2.w + x3.w);
            }
            par ^= 1;
        }
        if (P.round) {
#pragma unroll
            for (int i = 0; i < 4; i++) s[i] = __bfloat162float(__float2bfloat16(s[i]));
        }
        float x0 = s[0] * P.sl2, x1 = s[1] * P.sl2, x2 = s[2] * P.sl2, x3 = s[3] * P.sl2;
        if (valid < 16 || lo > 0) {
            x0 = g >= lo && g < valid ? x0 : -INFINITY;
            x1 = g >= lo && g < valid ? x1 : -INFINITY;
            x2 = g + 8 >= lo && g + 8 < valid ? x2 : -INFINITY;
            x3 = g + 8 >= lo && g + 8 < valid ? x3 : -INFINITY;
        }
        float a0 = fmaxf(x0, x2), a1 = fmaxf(x1, x3);
        a0 = fmaxf(a0, __shfl_xor_sync(FULLM, a0, 4));
        a1 = fmaxf(a1, __shfl_xor_sync(FULLM, a1, 4));
        a0 = fmaxf(a0, __shfl_xor_sync(FULLM, a0, 8));
        a1 = fmaxf(a1, __shfl_xor_sync(FULLM, a1, 8));
        a0 = fmaxf(a0, __shfl_xor_sync(FULLM, a0, 16));
        a1 = fmaxf(a1, __shfl_xor_sync(FULLM, a1, 16));
        const float mn0 = fmaxf(m_run0, a0), mn1 = fmaxf(m_run1, a1);
        const float al0 = ex2(m_run0 - mn0), al1 = ex2(m_run1 - mn1);
        const float p0f = ex2(x0 - mn0), p1f = ex2(x1 - mn1), p2f = ex2(x2 - mn0), p3f = ex2(x3 - mn1);
        l_run0 = l_run0 * al0 + (p0f + p2f);
        l_run1 = l_run1 * al1 + (p1f + p3f);
        if (__any_sync(FULLM, (mn0 > m_run0) | (mn1 > m_run1))) {  // the output's scale, per head
#pragma unroll
            for (int i = 0; i < 8; i++) {
                o[i][0] *= al0;
                o[i][1] *= al1;
                o[i][2] *= al0;
                o[i][3] *= al1;
            }
        }
        m_run0 = mn0;
        m_run1 = mn1;
        const uint32_t pb0 = transpose8x8(pack_bf16x2(p0f, p1f)), pb1 = transpose8x8(pack_bf16x2(p2f, p3f));  // P^T: tokens 2q, 2q + 1 and 8 + 2q, 9 + 2q of head g

        if constexpr (BF16) {
            // V^T's fragments: f[i].x channel 16 i + g of tokens 2 q, 2 q + 1 (the first in the low half), the transpose of the block of tokens 0-7 by channels 16 i .. 16 i + 7
            // whose row g, channels 2 q and 2 q + 1, the lane holds; .y channels 8 on; .z, .w tokens 8 on
#pragma unroll
            for (int i = 0; i < 8; i++) f[i] = make_uint4(transpose8x8(vw[i].x), transpose8x8(vw[i].y), transpose8x8(vw[i].z), transpose8x8(vw[i].w));
        } else {
            decode_tile<1>(cur.vr, vb, cur.hd, cur.pg, P.ent, P.nc, P.A, P.raw, hp, lane, f);
        }
        if (valid < 16 || lo > 0) {  // the tokens past the sequence's end (and before a window's start) zeroed (a recycled page's bytes there can decode to infinity, which times a probability of 0 is NaN)
            const auto in = [&](int t) { return t >= lo && t < valid; };
            const uint32_t k0 = (in(2 * q) ? 0x0000FFFFu : 0u) | (in(2 * q + 1) ? 0xFFFF0000u : 0u);
            const uint32_t k1 = (in(2 * q + 8) ? 0x0000FFFFu : 0u) | (in(2 * q + 9) ? 0xFFFF0000u : 0u);
#pragma unroll
            for (int i = 0; i < 8; i++) {
                f[i].x &= k0;
                f[i].y &= k0;
                f[i].z &= k1;
                f[i].w &= k1;
            }
        }
#pragma unroll
        for (int i = 0; i < 8; i++) mma(o[i], f[i].x, f[i].y, f[i].z, f[i].w, pb0, pb1);
        if constexpr (PIPE && !BF16) cur = nxt;
    }

    // the sums over the lanes of the same q (the other tokens)
    float l0 = l_run0, l1 = l_run1;
    l0 += __shfl_xor_sync(FULLM, l0, 4);
    l1 += __shfl_xor_sync(FULLM, l1, 4);
    l0 += __shfl_xor_sync(FULLM, l0, 8);
    l1 += __shfl_xor_sync(FULLM, l1, 8);
    l0 += __shfl_xor_sync(FULLM, l0, 16);
    l1 += __shfl_xor_sync(FULLM, l1, 16);
    // the warps' states into shared memory (a warp's lanes of the same q hold the same max and sum: heads 2q and 2q + 1; the output's channels 16 i + g, 16 i + 8 + g of those heads, of its half)
    __shared__ float s_ml[NW][8][2];
    __shared__ float s_o[NW][8][D];
    if (g == 0) {
        s_ml[warp][2 * q][0] = m_run0;
        s_ml[warp][2 * q][1] = l0;
        s_ml[warp][2 * q + 1][0] = m_run1;
        s_ml[warp][2 * q + 1][1] = l1;
    }
#pragma unroll
    for (int i = 0; i < 8; i++) {
        s_o[warp][2 * q][16 * i + g] = o[i][0];
        s_o[warp][2 * q + 1][16 * i + g] = o[i][1];
        s_o[warp][2 * q][16 * i + 8 + g] = o[i][2];
        s_o[warp][2 * q + 1][16 * i + 8 + g] = o[i][3];
    }
    __syncthreads();
    // the CTA's partial: a thread a channel (DO / D of them, D the CTA's threads), a head at a time, over its pages' places in order (a pair's two warps hold the same max and sum); a CTA of no pages
    // has the max -infinity, the sum 0 and the output 0
    const size_t pidx = ((size_t)(b * P.H + h) * P.splits + split) * G;
    for (int head = 0; head < G; head++) {
        float M = s_ml[0][head][0];
#pragma unroll
        for (int w = 1; w < NP; w++) M = fmaxf(M, s_ml[w * HALVES][head][0]);
#pragma unroll
        for (int c = threadIdx.x; c < DO; c += NW * 32) {
            const int half = c / D, ch = c % D;
            float L = 0.f, O = 0.f;
            if (M != -INFINITY) {
#pragma unroll
                for (int w = 0; w < NP; w++) {
                    const float sc = ex2(s_ml[w * HALVES][head][0] - M);
                    L += sc * s_ml[w * HALVES][head][1];
                    O += sc * s_o[w * HALVES + half][head][ch];
                }
            }
            P.ws_o[(pidx + head) * DO + c] = O;
            if (c == 0) {
                P.ws_ml[(pidx + head) * 2] = M;
                P.ws_ml[(pidx + head) * 2 + 1] = L;
            }
        }
    }
    if constexpr (FUSE) {
        __shared__ unsigned s_last;
        __threadfence();  // (the partial is in L2 before the count says so)
        __syncthreads();
        if (threadIdx.x == 0) s_last = atomicAdd(P.cnt + (size_t)b * P.H + h, 1u) == (unsigned)(sb - 1);
        __syncthreads();
        if (!s_last) return;
        __threadfence();
        const size_t bh = (size_t)b * P.H + h;
        for (int head = warp; head < G; head += NW) merge_head<DO>(P.ws_ml + bh * P.splits * G * 2, P.ws_o + bh * P.splits * G * DO, P.out + (int64_t)b * P.out_stride + (int64_t)h * G * DO, G, head, sb, lane);
        if (threadIdx.x == 0) P.cnt[bh] = 0;  // (zero for the next call)
    }
}

template <bool PIPE, bool FUSE, int HALVES, bool BF16 = false>
__global__ void __launch_bounds__(NW * 32, 1) attn_kernel(const AttnParams P) {
    attn_body<PIPE, FUSE, HALVES, BF16>(P);
}

// the partials of a (sequence, kv head) merged: one warp a query head, a lane DO / 32 channels; a sequence of no tokens gives zeros
template <int DO>
__global__ void __launch_bounds__(256) combine_kernel(const float* __restrict__ ml, const float* __restrict__ wo, __nv_bfloat16* __restrict__ out, int64_t out_stride, int H, int G, int splits, int ppc_min,
                                                      int pfloor, const int32_t* __restrict__ slen, const int32_t* __restrict__ first) {
    const int bh = blockIdx.x, head = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (head >= G) return;
    const int b = bh / H, h = bh % H;
    const int len = slen[b], fst = first_of(first, b, len);
    const int nparts = plan_splits(fst < len ? ((len + 15) >> 4) - (fst >> 4) : 0, splits, ppc_min, pfloor);  // (the partials the sequence's CTAs wrote: the attention kernel's plan)
    merge_head<DO>(ml + (size_t)bh * splits * G * 2, wo + (size_t)bh * splits * G * DO, out + (int64_t)b * out_stride + (int64_t)h * G * DO, G, head, nparts, lane);
}

// ---- prompt attention over the pages ----------------------------------------------------------------------------------------------
// The n rows of a prompt chunk of one sequence (positions start .. start + n - 1), each over the keys from its window's first (pos - window + 1; 0 where there is no window) to
// itself: those before `start` read from the sequence's pages (its block table), the chunk's own from its bf16 rows (with no rows, from the pages too: written before the call).
// FlashAttention-2's loop on the tensor cores: a CTA is BM = 16 RG rows of one query head, RG row groups of HALVES warps, warp (rg, hf) its group's 16 rows and the head's channels
// 128 hf .. 128 hf + 127 (the page's head h HALVES + hf). Keys 32 a tile: two pages decoded by the CTA's warps into shared memory (a warp a stream and a page's head, its bases in
// registers for the whole call; K's words as the decode gives them, V's through movmatrix) or 32 of the bf16 rows, both into K and V [32][DO + 8] bf16 (a key's channels
// contiguous; rows 16 bytes apart in the banks, so ldmatrix meets no bank twice). S = Q K^T with the queries' A fragments in registers; a score's shares of the halves summed through
// shared memory in their order (every warp of a row group the same sums); then rounded to bf16 where asked (before the scale: a bf16 matmul's output), scaled, capped (cap
// tanh(x / cap), as transformers), masked, the online softmax in fp32 (seeded by the head's sink where there is one), and O += bf16(P) V into the warp's channels.
// The same kernel reads a bf16 cache in place of the pages (glyd_gpu_kv_prefill_bf16: kc and vc [places][H][DO], a position's place through the block table, or modulo
// cap with none): its tiles are the pages' (the prefix in 32 keys from lo's page on, then the chunk's own from max(start, lo)), each key's values the cache's where it
// is the call's (lo .. hi) and zeros elsewhere, as the pages' decode gives them; so where the cache holds what the pages hold, the outputs are the same bits.
constexpr int PKT = 32;  // keys a tile: two pages
constexpr int PSP = 33;  // a row's floats in the score shares (32 keys and one of padding)
__host__ __device__ constexpr int pf_shared(int halves, int rg) { return 2 * PKT * (D * halves + 8) * 2 + (halves > 1 ? rg * halves * 16 * PSP * 4 : 0); }

struct PrefillParams {
    const __nv_bfloat16* q;  // row t's query heads at q + t q_stride: [H G][DO]
    int64_t q_stride;
    const __nv_bfloat16* key;  // row t's keys at key + t key_stride: [H][DO] (null: the chunk's keys and values are in the pages)
    int64_t key_stride;
    const __nv_bfloat16* value;
    int64_t value_stride;
    const uint8_t* pool;
    int64_t page_stride;
    const uint8_t* kbase;  // [HP][D]
    const uint8_t* vbase;
    const int32_t* bt;  // the sequence's pages: tokens 16 j .. 16 j + 15 in page bt[j]
    const uint32_t* ent;
    int nc;
    RawPool raw;
    const float* sinks;  // [H G] or null
    __nv_bfloat16* out;  // row t's at out + t out_stride: [H G][DO]
    int64_t out_stride;
    const __nv_bfloat16* kc;  // a bf16 cache in place of the pages (null: the pages): [places][H][DO]
    const __nv_bfloat16* vc;
    int cap;  // the cache's places where there is no block table (a position's place: position modulo cap)
    int n, start, window, G, A, hdrb, HP, round;
    float sl2;    // scale * log2(e)
    float icap;   // scale / cap (cap 0: none)
    float capl2;  // cap * log2(e)
};

__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const void* p) { ldmatrix_x4(r, (uint32_t)__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void ldsm4t(uint32_t (&r)[4], const void* p) {
#if defined(__HIP_PLATFORM_AMD__)
    glyd_ldsm4t(r, (uint32_t)__cvta_generic_to_shared(p));
#else
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"((uint32_t)__cvta_generic_to_shared(p)));
#endif
}

// A score of a tile: its dot product (bf16 where asked), scaled and capped, in log2 units; -infinity where the key is not the row's.
__device__ __forceinline__ float pf_score(const PrefillParams& P, float x, bool in) {
    if (P.round) x = __bfloat162float(__float2bfloat16(x));
    const float y = P.icap > 0.f ? P.capl2 * tanhf(x * P.icap) : x * P.sl2;
    return in ? y : -INFINITY;
}

template <int HALVES, int RG>
__device__ __forceinline__ void prefill_body(const PrefillParams P) {
    constexpr int DO = D * HALVES, KS = DO + 8, NWP = RG * HALVES, BM = 16 * RG;
    static_assert(RG % 2 == 0, "the fill: a tile's two pages over the warps of a task in steps of RG / 2");
    extern __shared__ __align__(16) unsigned char pf_sm[];
    __nv_bfloat16* Ks = reinterpret_cast<__nv_bfloat16*>(pf_sm);
    __nv_bfloat16* Vs = Ks + PKT * KS;
    [[maybe_unused]] float* Sp = reinterpret_cast<float*>(Vs + PKT * KS);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3, m4 = lane >> 3;
    const int rg = warp / HALVES, hf = warp % HALVES;
    const int head = blockIdx.y, h = head / P.G, r0 = blockIdx.x * BM;
    const int n = P.n, start = P.start, W = P.window;
    // the CTA's keys: lo (its first row's window) .. hi (past its last row); from the pages up to end1, then (rows given) from the rows, from p1
    const int lo = W > 0 ? max(0, start + r0 - W + 1) : 0, hi = start + min(n, r0 + BM);
    // (a bf16 cache: the chunk's own keys as rows of their own, read from the cache, so that its tiles are the pages' with the rows given)
    const bool rows = P.key || P.kc;
    const int end1 = rows ? start : hi, lo16 = lo & ~15, p1 = max(start, lo);
    const int n0 = lo < end1 ? (end1 - lo16 + PKT - 1) / PKT : 0, n1 = rows ? (hi - p1 + PKT - 1) / PKT : 0;

    // the queries' A fragments: rows ra (g) and rb (g + 8) of the warp's group, its 128 channels, 16 a step (zeros past n)
    const int ra = r0 + rg * 16 + g, rb = ra + 8;
    uint32_t qa[8][4];
    {
        const uint32_t* qx = reinterpret_cast<const uint32_t*>(P.q + (int64_t)min(ra, n - 1) * P.q_stride + (int64_t)head * DO + hf * D);
        const uint32_t* qy = reinterpret_cast<const uint32_t*>(P.q + (int64_t)min(rb, n - 1) * P.q_stride + (int64_t)head * DO + hf * D);
#pragma unroll
        for (int ks = 0; ks < 8; ks++) {
            qa[ks][0] = ra < n ? qx[8 * ks + t4] : 0u;
            qa[ks][1] = rb < n ? qy[8 * ks + t4] : 0u;
            qa[ks][2] = ra < n ? qx[8 * ks + 4 + t4] : 0u;
            qa[ks][3] = rb < n ? qy[8 * ks + 4 + t4] : 0u;
        }
    }
    const int pa = ra < n ? start + ra : -1, pb = rb < n ? start + rb : -1;  // (a row past n takes no key)
    const int la = W > 0 ? pa - W + 1 : 0, lb = W > 0 ? pb - W + 1 : 0;

    // the warp's part of a page fill: a stream (0 K, 1 V) of a page's head, the pages of the tile from slot warp / (2 HALVES) in steps of RG / 2
    const int task = warp % (2 * HALVES), stream = task / HALVES, hp = task % HALVES, hpg = h * HALVES + hp;
    uint32_t hb[16];
    if (P.kc) {
    } else if (stream == 0) {
        load_bases<0>(P.kbase + hpg * D, lane, *reinterpret_cast<uint32_t(*)[8]>(hb));
    } else {
        load_bases<1>(P.vbase + hpg * D, lane, hb);
    }
    // a key's row of the bf16 cache: its block's place, else its place in the ring
    const auto place = [&](int kp) -> int64_t { return P.bt ? (int64_t)__ldg(P.bt + (kp >> 4)) * 16 + (kp & 15) : (int64_t)(kp % P.cap); };

    float acc[16][4];  // O: rows ra (acc[j][0], [1]) and rb ([2], [3]), channels 128 hf + 8 j + 2 t4 and the next
#pragma unroll
    for (int j = 0; j < 16; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    float ma = -INFINITY, mb = -INFINITY, sa = 0.f, sb = 0.f;
    if (P.sinks) {  // (the sink's logit in the softmax's sum: the state as if a key of that score had been seen, with no value)
        ma = mb = P.sinks[head] * 1.4426950408889634f;
        sa = sb = ma == -INFINITY ? 0.f : 1.f;
    }

    // a page's raw reads for the warp's task: the next tile's, made while this one is computed (on an RTX 4080 SUPER, gemma-4-12B's global layer at a prefix of 8,192: 3.47 ms
    // a chunk of 256 rows where reading each page as it is decoded takes 3.64)
    struct Ahead {
        Raw r;
        uint4 hd;
        const uint32_t* pgw;  // the page's `used` word, directory and arena
    };
    constexpr int NPW = 4 / RG;            // a tile's pages a warp fills
    const int slot = warp / (2 * HALVES);  // its first
    Ahead nx[NPW];
    const auto load_page = [&](Ahead& a, int pg) {
        const uint8_t* page = P.pool + (int64_t)__ldg(P.bt + pg) * P.page_stride;
        a.hd = ldg_stream<0>(page + 16 * hpg);
        a.pgw = reinterpret_cast<const uint32_t*>(page + 16 * P.HP);
        raw_load(a.r, page + P.hdrb + (int64_t)hpg * HEADPAGE + stream * TILE, lane);
    };
    if (n0 > 0 && !P.kc) {
#pragma unroll
        for (int u = 0; u < NPW; u++) {
            const int pg = (lo16 >> 4) + slot + u * (RG / 2);
            if (16 * pg < end1) load_page(nx[u], pg);
        }
    }

    for (int it = 0; it < n0 + n1; it++) {
        const bool paged = it < n0;
        const int k0 = paged ? lo16 + PKT * it : p1 + PKT * (it - n0), kend = paged ? end1 : hi;
        __syncthreads();  // (the tile before is read)
        if (P.kc) {
            // (the bf16 cache: a key's values where it is the call's, from its window's first key to the tile's end; zeros elsewhere, as the pages give them)
            constexpr int C8 = DO / 8;
            for (int c = threadIdx.x; c < 2 * PKT * C8; c += 32 * NWP) {
                const int isv = c >= PKT * C8, cc = isv ? c - PKT * C8 : c, j = cc / C8, c8 = cc % C8, kp = k0 + j;
                uint4 v = make_uint4(0, 0, 0, 0);
                if (kp >= lo && kp < kend) v = __ldg(reinterpret_cast<const uint4*>((isv ? P.vc : P.kc) + (place(kp) * P.HP / HALVES + h) * DO) + c8);
                *reinterpret_cast<uint4*>((isv ? Vs : Ks) + j * KS + c8 * 8) = v;
            }
        } else if (paged) {
#pragma unroll
            for (int u = 0; u < NPW; u++) {
                const int p = slot + u * (RG / 2), pg = (k0 >> 4) + p;
                __nv_bfloat16* rows = (stream == 0 ? Ks : Vs) + 16 * p * KS + hp * D;  // the page's 16 keys, the task's 128 channels
                if (16 * pg >= end1) {  // (past the keys the pages hold for this call: zeros)
#pragma unroll
                    for (int i = lane; i < 16 * 16; i += 32) *reinterpret_cast<uint4*>(rows + (i >> 4) * KS + (i & 15) * 8) = make_uint4(0, 0, 0, 0);
                    continue;
                }
                uint4 f[8];
                if (stream == 0) {
                    decode_tile<0>(nx[u].r, hb, nx[u].hd, nx[u].pgw, P.ent, P.nc, P.A, P.raw, hpg, lane, f);
#pragma unroll
                    for (int i = 0; i < 16; i++) {  // group i: key 8 (i & 1) + g, channels 16 (i >> 1) + 2 t4 + {0, 1} and + {8, 9}
                        const uint4& c = f[i >> 1];
                        uint32_t* w = reinterpret_cast<uint32_t*>(rows + (8 * (i & 1) + g) * KS + 16 * (i >> 1) + 2 * t4);
                        w[0] = (i & 1) ? c.y : c.x;
                        w[4] = (i & 1) ? c.w : c.z;
                    }
                } else {
                    decode_tile<1>(nx[u].r, hb, nx[u].hd, nx[u].pgw, P.ent, P.nc, P.A, P.raw, hpg, lane, f);
                    const int tlo = lo - 16 * pg, thi = end1 - 16 * pg;  // the page's keys of this call: tlo .. thi (the others' values zero: a recycled page's bytes can be infinities)
                    if (tlo > 0 || thi < 16) {
                        const auto in = [&](int t) { return t >= tlo && t < thi; };
                        const uint32_t k0m = (in(2 * t4) ? 0x0000FFFFu : 0u) | (in(2 * t4 + 1) ? 0xFFFF0000u : 0u);
                        const uint32_t k1m = (in(2 * t4 + 8) ? 0x0000FFFFu : 0u) | (in(2 * t4 + 9) ? 0xFFFF0000u : 0u);
#pragma unroll
                        for (int i = 0; i < 8; i++) {
                            f[i].x &= k0m;
                            f[i].y &= k0m;
                            f[i].z &= k1m;
                            f[i].w &= k1m;
                        }
                    }
                    // f[i]: channels 16 i + g (x: keys 2 t4, + 1; z: 8 + 2 t4, + 1) and 16 i + 8 + g (y, w); transposed, a word is two channels of one key
                    uint32_t* w0 = reinterpret_cast<uint32_t*>(rows + g * KS + 2 * t4);
                    uint32_t* w8 = reinterpret_cast<uint32_t*>(rows + (8 + g) * KS + 2 * t4);
#pragma unroll
                    for (int i = 0; i < 8; i++) {
                        w0[8 * i] = transpose8x8(f[i].x);
                        w0[8 * i + 4] = transpose8x8(f[i].y);
                        w8[8 * i] = transpose8x8(f[i].z);
                        w8[8 * i + 4] = transpose8x8(f[i].w);
                    }
                }
            }
            if (it + 1 < n0) {  // the next tile's pages on their way while this one is computed
#pragma unroll
                for (int u = 0; u < NPW; u++) {
                    const int pg = ((k0 + PKT) >> 4) + slot + u * (RG / 2);
                    if (16 * pg < end1) load_page(nx[u], pg);
                }
            }
        } else {
            constexpr int C8 = DO / 8;  // a row's 16-byte chunks
            for (int c = threadIdx.x; c < 2 * PKT * C8; c += 32 * NWP) {
                const int isv = c >= PKT * C8, cc = isv ? c - PKT * C8 : c, j = cc / C8, c8 = cc % C8, kp = k0 + j;
                uint4 v = make_uint4(0, 0, 0, 0);
                if (kp < hi) v = __ldg(reinterpret_cast<const uint4*>((isv ? P.value : P.key) + (int64_t)(kp - start) * (isv ? P.value_stride : P.key_stride) + (int64_t)h * DO) + c8);
                *reinterpret_cast<uint4*>((isv ? Vs : Ks) + j * KS + c8 * 8) = v;
            }
        }
        __syncthreads();

        // S [16 rows][32 keys] = Q K^T over the warp's 128 channels: 4 n-tiles of 8 keys
        float s[4][4];
#pragma unroll
        for (int j = 0; j < 4; j++) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
#pragma unroll
        for (int ks = 0; ks < 8; ks++) {
#pragma unroll
            for (int np = 0; np < 2; np++) {
                uint32_t b[4];
                ldsm4(b, Ks + (np * 16 + (m4 >> 1) * 8 + (lane & 7)) * KS + hf * D + ks * 16 + (m4 & 1) * 8);
                mma(s[2 * np], qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], b[0], b[1]);
                mma(s[2 * np + 1], qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], b[2], b[3]);
            }
        }
        if constexpr (HALVES > 1) {  // the halves' shares summed in their order
            float* myS = Sp + (rg * HALVES + hf) * 16 * PSP;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                myS[g * PSP + j * 8 + 2 * t4] = s[j][0];
                myS[g * PSP + j * 8 + 2 * t4 + 1] = s[j][1];
                myS[(g + 8) * PSP + j * 8 + 2 * t4] = s[j][2];
                myS[(g + 8) * PSP + j * 8 + 2 * t4 + 1] = s[j][3];
            }
            __syncthreads();
            const float* gS = Sp + rg * HALVES * 16 * PSP;
#pragma unroll
            for (int j = 0; j < 4; j++) {
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    const int col = j * 8 + 2 * t4 + e;
                    float xa = gS[g * PSP + col], xb = gS[(g + 8) * PSP + col];
#pragma unroll
                    for (int d = 1; d < HALVES; d++) {
                        xa += gS[(d * 16 + g) * PSP + col];
                        xb += gS[(d * 16 + g + 8) * PSP + col];
                    }
                    s[j][e] = xa;
                    s[j][2 + e] = xb;
                }
            }
        }
        // the scores in log2 units, masked; the running maxima and sums (a row's 4 lanes share g)
        float mxa = ma, mxb = mb;
#pragma unroll
        for (int j = 0; j < 4; j++) {
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const int kp = k0 + j * 8 + 2 * t4 + e;
                s[j][e] = pf_score(P, s[j][e], kp >= la && kp <= pa && kp < kend);
                s[j][2 + e] = pf_score(P, s[j][2 + e], kp >= lb && kp <= pb && kp < kend);
                mxa = fmaxf(mxa, s[j][e]);
                mxb = fmaxf(mxb, s[j][2 + e]);
            }
        }
        mxa = fmaxf(mxa, __shfl_xor_sync(FULLM, mxa, 1));
        mxa = fmaxf(mxa, __shfl_xor_sync(FULLM, mxa, 2));
        mxb = fmaxf(mxb, __shfl_xor_sync(FULLM, mxb, 1));
        mxb = fmaxf(mxb, __shfl_xor_sync(FULLM, mxb, 2));
        const float ca = mxa == -INFINITY ? 1.f : ex2(ma - mxa), cb = mxb == -INFINITY ? 1.f : ex2(mb - mxb);
        float ta = 0.f, tb = 0.f;
#pragma unroll
        for (int j = 0; j < 4; j++) {
#pragma unroll
            for (int e = 0; e < 2; e++) {
                s[j][e] = mxa == -INFINITY ? 0.f : ex2(s[j][e] - mxa);
                s[j][2 + e] = mxb == -INFINITY ? 0.f : ex2(s[j][2 + e] - mxb);
                ta += s[j][e];
                tb += s[j][2 + e];
            }
        }
        ta += __shfl_xor_sync(FULLM, ta, 1);
        ta += __shfl_xor_sync(FULLM, ta, 2);
        tb += __shfl_xor_sync(FULLM, tb, 1);
        tb += __shfl_xor_sync(FULLM, tb, 2);
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
        // O += bf16(P) V: the probabilities as the A operand, 16 keys a step; V's rows through ldmatrix.trans
#pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            const uint32_t a0 = pack_bf16x2(s[2 * kk][0], s[2 * kk][1]), a1 = pack_bf16x2(s[2 * kk][2], s[2 * kk][3]);
            const uint32_t a2 = pack_bf16x2(s[2 * kk + 1][0], s[2 * kk + 1][1]), a3 = pack_bf16x2(s[2 * kk + 1][2], s[2 * kk + 1][3]);
#pragma unroll
            for (int np = 0; np < 8; np++) {
                uint32_t b[4];
                ldsm4t(b, Vs + (kk * 16 + (m4 & 1) * 8 + (lane & 7)) * KS + hf * D + np * 16 + (m4 >> 1) * 8);
                mma(acc[2 * np], a0, a1, a2, a3, b[0], b[1]);
                mma(acc[2 * np + 1], a0, a1, a2, a3, b[2], b[3]);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < 16; j++) {
        const int c = head * DO + hf * D + j * 8 + 2 * t4;
        if (ra < n) *reinterpret_cast<__nv_bfloat162*>(P.out + (int64_t)ra * P.out_stride + c) = __floats2bfloat162_rn(sa > 0.f ? acc[j][0] / sa : 0.f, sa > 0.f ? acc[j][1] / sa : 0.f);
        if (rb < n) *reinterpret_cast<__nv_bfloat162*>(P.out + (int64_t)rb * P.out_stride + c) = __floats2bfloat162_rn(sb > 0.f ? acc[j][2] / sb : 0.f, sb > 0.f ? acc[j][3] / sb : 0.f);
    }
}

template <int HALVES, int RG>
__global__ void __launch_bounds__(32 * RG * HALVES, 1) prefill_kernel(const PrefillParams P) {
    prefill_body<HALVES, RG>(P);
}

}  // namespace kvc

__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }
// The same with the approximate reciprocal (2 instructions where the IEEE division is a dozen and a call): 2 ulp, past the bf16 the product is rounded to
// (the tiled kernel's epilogue makes 128 of them a thread, and its code was 17,000 instructions long: an instruction cache that missed 54% of the stalls there).
__device__ __forceinline__ float silu_fast(float x) { return __fdividef(x, 1.f + __expf(-x)); }
__device__ __forceinline__ float gelu_tanh(float x) { return 0.5f * x * (1.f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x))); }

// The experts' product for the pairs of a plan, 16 MT at a time: a block a unit of a hit expert's rows, blockIdx.y
// the hit (past those hit: nothing to do), blockIdx.x the unit; X's row for the i-th pair of the order: token
// order[i] / k (gather) or i. ACT 0: a unit a row block; its rows' sums (+ bias [E, O]) to Y [pairs, O], the
// pairs in the plan's order, or (Y32) times the pair's weight w (bf16, or fp32: wf32) to row j of Y32 [P, O] for
// pair j. ACT 1 (SiLU) and 2 (GELU, tanh): a unit a row block of the gate's half (the first O / 2 rows of the
// expert's) and the same of the up's (the rest): Y [pairs, O / 2] = act(gate) up, each + its bias. The 8 warps
// split K (with the gate: even warps the gate's rows, odd the up's) and add their sums through mma_gemm_kernel's
// mma_steps and warp_sums. Few units (few experts hit) split K over gridDim.z blocks (blockIdx.z's share of the
// steps): each writes its sums to parts ([splits][P][units][64, or with the gate 128] floats, P the pairs), and
// the last of a unit's to finish (last_of; done: a counter a unit) adds them in order and writes the unit out: the
// same result every run.
template <class Fmt, int MT, int ACT, int V = 0>  // V: the step loop's variants, bit 0 Nib::patch_b, bit 1 the m-tiles with pairs alone, bit 2 the L2 prefetch
__device__ __forceinline__ void mma_moe_body(Fmt f, int64_t O, int64_t K, const int* __restrict__ plan, int E, int64_t k, int gather, const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ bias, const void* __restrict__ w, int wf32, __nv_bfloat16* __restrict__ Y, float* __restrict__ Y32, int64_t P, float* __restrict__ parts, int* __restrict__ done, int pf) {
    extern __shared__ __align__(128) float red[];  // [4 warps][16 MT rows][65], then the warps' scratch [8][S2_BYTES]
    __shared__ uint32_t tab[Fmt::kTable ? 256 : 1];
    __shared__ int last;
    if ((int)blockIdx.y >= __ldg(plan)) return;  // the whole block: no expert this far down the hits
    if constexpr (Fmt::kTable) fill_groups(tab);
    __syncthreads();
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    uint32_t* s2 = (uint32_t*)(red + 4 * MT * 16 * 65) + warp * (S2_BYTES / 4);
    if constexpr (!Fmt::kTable) {  // 12-bit: the warp's scratch for a step's many exceptions (decode_x: zero between steps)
        ((uint4*)s2)[2 * lane] = ((uint4*)s2)[2 * lane + 1] = make_uint4(0u, 0u, 0u, 0u);
        __syncwarp();
    }
    const int* order = plan + 2 + 2 * E;
    int e = __ldg(plan + 1 + blockIdx.y), p0 = __ldg(plan + 1 + E + blockIdx.y), p1 = __ldg(plan + 2 + E + blockIdx.y);
    int64_t RB = O / 64, KS = K / 16, chunks = ACT ? 4 : 8, chunk = ACT ? warp >> 1 : warp;
    int64_t rb = blockIdx.x + (ACT && (warp & 1) ? RB / 2 : 0);  // this warp's row block of the expert's
    int64_t k0 = blockIdx.z * KS / gridDim.z, k1 = (blockIdx.z + 1) * KS / gridDim.z;  // this block's steps
    int64_t per = (k1 - k0 + chunks - 1) / chunks, s0 = min(k1, k0 + chunk * per), s1 = min(k1, s0 + per);
    int64_t base = ((int64_t)e * RB + rb) * KS;
    constexpr int W = ACT ? 128 : 64;  // a unit's columns in parts
    // A pair's sums (the i-th of the order; column c of the unit's; ACT: gate a, up b) written out.
    auto finish = [&](int64_t i, int c, float a, float b) {
        if (ACT) {
            int64_t o = blockIdx.x * 64 + c, at = (int64_t)e * O + o;  // the gate's row; the up's O / 2 on
            float gate = a + (bias ? __bfloat162float(bias[at]) : 0.f), up = b + (bias ? __bfloat162float(bias[at + O / 2]) : 0.f);
            Y[i * (O / 2) + o] = __float2bfloat16((ACT == 1 ? silu(gate) : gelu_tanh(gate)) * up);
        } else {
            int64_t o = blockIdx.x * 64 + c;
            float y = a + (bias ? __bfloat162float(bias[(int64_t)e * O + o]) : 0.f);
            if (Y32) {
                int j = __ldg(order + i);
                Y32[(int64_t)j * O + o] = y * (wf32 ? ((const float*)w)[j] : __bfloat162float(((const __nv_bfloat16*)w)[j]));
            } else {
                Y[i * O + o] = __float2bfloat16(y);
            }
        }
    };
    for (int m0 = p0; m0 < p1; m0 += 16 * MT) {
        int n = min(16 * MT, p1 - m0);
        // This thread's rows of X: rows g and g + 8 of each m-tile (none past the pairs).
        const __nv_bfloat16* xr[MT][2];
#pragma unroll
        for (int mt = 0; mt < MT; mt++)
#pragma unroll
            for (int hh = 0; hh < 2; hh++) {
                int r = mt * 16 + g + 8 * hh;
                xr[mt][hh] = r < n ? X + (gather ? (int64_t)(__ldg(order + m0 + r) / k) : (int64_t)(m0 + r)) * K + t * 2 : nullptr;
            }
        float acc[MT][8][4];
#pragma unroll
        for (int mt = 0; mt < MT; mt++)
#pragma unroll
            for (int nn = 0; nn < 8; nn++)
#pragma unroll
                for (int q = 0; q < 4; q++) acc[mt][nn][q] = 0.f;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 800
        constexpr int U = std::is_same_v<Fmt, Nib> && MT == 4 && ACT ? 1 : 0;  // an exception a pass: see Nib::patch
#else
        constexpr int U = 0;
#endif
        if constexpr ((V & 16) && MT == 1 && has_stage<Fmt>::value) {
            constexpr int P = (V & 64) ? 8 : 4;
            if constexpr ((V & 128) != 0)
                mma_steps_staged_s<Fmt, 4>(f, base, s0, s1, xr, lane, (uint8_t*)(red + 4 * MT * 16 * 65) + 8 * S2_BYTES + warp * 8 * ring_slot<Fmt>(), tab, acc);
            else
                mma_steps_staged<Fmt, P>(f, base, s0, s1, xr, lane, (uint8_t*)(red + 4 * MT * 16 * 65) + 8 * S2_BYTES + warp * P * ring_slot<Fmt>(), tab, acc);
        } else {
            mma_steps<Fmt, MT, !Fmt::kTable, U, (V & 1) != 0, (V & 2) != 0, (V & 4) != 0, (V & 8) && MT == 1 ? 2 : 1>(f, base, s0, s1, xr, lane, s2, tab, acc, pf, (n + 15) / 16);
        }
        warp_sums<MT>(acc, red, warp, g, t);
        for (int i = threadIdx.x; i < n * 64; i += 256) {
            int r = i / 64, c = i % 64;
            float v[4];
#pragma unroll
            for (int ww = 0; ww < 4; ww++) v[ww] = red[(ww * MT * 16 + r) * 65 + c];
            float a = ACT ? v[0] + v[2] : v[0] + v[1] + v[2] + v[3], b = ACT ? v[1] + v[3] : 0.f;
            if (gridDim.z == 1) {
                finish(m0 + r, c, a, b);
            } else {
                float* pp = parts + (((int64_t)blockIdx.z * P + m0 + r) * gridDim.x + blockIdx.x) * W + c;
                pp[0] = a;
                if (ACT) pp[64] = b;
            }
        }
        __syncthreads();  // red is reused
    }
    if (gridDim.z > 1 && last_of(done + (int64_t)blockIdx.y * gridDim.x + blockIdx.x, gridDim.z, last))
        for (int i = threadIdx.x; i < (p1 - p0) * 64; i += 256) {
            int r = i / 64, c = i % 64;
            float a = 0.f, b = 0.f;
            for (int z = 0; z < (int)gridDim.z; z++) {
                const float* pp = parts + (((int64_t)z * P + p0 + r) * gridDim.x + blockIdx.x) * W + c;
                a += __ldcg(pp);
                if (ACT) b += __ldcg(pp + 64);
            }
            finish(p0 + r, c, a, b);
        }
}

template <class Fmt, int MT, int ACT, int V = 0>  // V: the step loop's variants, bit 0 Nib::patch_b, bit 1 the m-tiles with pairs alone, bit 2 the L2 prefetch
__global__ void __launch_bounds__(256, MT == 1 ? ((V & 32) ? 3 : 2) : 1) mma_moe_kernel(Fmt f, int64_t O, int64_t K, const int* __restrict__ plan, int E, int64_t k, int gather, const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ bias, const void* __restrict__ w, int wf32, __nv_bfloat16* __restrict__ Y, float* __restrict__ Y32, int64_t P, float* __restrict__ parts, int* __restrict__ done, int pf) {
    mma_moe_body<Fmt, MT, ACT, V>(f, O, K, plan, E, k, gather, X, bias, w, wf32, Y, Y32, P, parts, done, pf);
}

// The experts' product with no gate (the down), a warp a row block over the whole of K: a block 8 row blocks of a hit
// expert (blockIdx.x, blockIdx.y the hit), the warps' sums not added across warps (none in shared memory, no barrier
// but the tiered layout's table), each thread writing its own, 16 MT pairs at a time as mma_moe_kernel does. K = 768 is
// 48 steps a warp where mma_moe_kernel's 8 warps split it 6 each, and a block's start and its sums in shared memory cost
// as much as its steps: 512 blocks of a long loop instead of 4,096 of a short one (Qwen3-30B-A3B's down, 32 row blocks).
// Y [pairs, O] in the plan's order (+ bias [E, O]), or (Y32) times the pair's weight w to row j of Y32 [P, O] for pair j.
template <class Fmt, int MT, int V>
__device__ __forceinline__ void mma_moe_rows_body(Fmt f, int64_t O, int64_t K, const int* __restrict__ plan, int E, int64_t k, int gather, const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ bias, const void* __restrict__ w, int wf32, __nv_bfloat16* __restrict__ Y, float* __restrict__ Y32, int pf) {
    extern __shared__ __align__(128) uint32_t rows_scratch[];  // the warps' scratch [8][S2_BYTES]
    __shared__ uint32_t tab[Fmt::kTable ? 256 : 1];
    if ((int)blockIdx.y >= __ldg(plan)) return;  // the whole block: no expert this far down the hits
    if constexpr (Fmt::kTable) fill_groups(tab);
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    uint32_t* s2 = rows_scratch + warp * (S2_BYTES / 4);
    if constexpr (!Fmt::kTable) {  // 12-bit: the warp's scratch for a step's many exceptions (decode_x: zero between steps)
        ((uint4*)s2)[2 * lane] = ((uint4*)s2)[2 * lane + 1] = make_uint4(0u, 0u, 0u, 0u);
        __syncwarp();
    }
    if constexpr (Fmt::kTable) __syncthreads();
    const int* order = plan + 2 + 2 * E;
    int e = __ldg(plan + 1 + blockIdx.y), p0 = __ldg(plan + 1 + E + blockIdx.y), p1 = __ldg(plan + 2 + E + blockIdx.y);
    int64_t RB = O / 64, KS = K / 16, rb = (int64_t)blockIdx.x * 8 + warp;
    if (rb >= RB) return;
    int64_t base = ((int64_t)e * RB + rb) * KS;
    for (int m0 = p0; m0 < p1; m0 += 16 * MT) {
        int n = min(16 * MT, p1 - m0);
        const __nv_bfloat16* xr[MT][2];
#pragma unroll
        for (int mt = 0; mt < MT; mt++)
#pragma unroll
            for (int hh = 0; hh < 2; hh++) {
                int r = mt * 16 + g + 8 * hh;
                xr[mt][hh] = r < n ? X + (gather ? (int64_t)(__ldg(order + m0 + r) / k) : (int64_t)(m0 + r)) * K + t * 2 : nullptr;
            }
        float acc[MT][8][4];
#pragma unroll
        for (int mt = 0; mt < MT; mt++)
#pragma unroll
            for (int nn = 0; nn < 8; nn++)
#pragma unroll
                for (int q = 0; q < 4; q++) acc[mt][nn][q] = 0.f;
        mma_steps<Fmt, MT, !Fmt::kTable, 0, (V & 1) != 0, (V & 2) != 0, (V & 4) != 0>(f, base, 0, KS, xr, lane, s2, tab, acc, pf, (n + 15) / 16);
#pragma unroll
        for (int mt = 0; mt < MT; mt++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                int r = mt * 16 + g + 8 * h;
                if (r >= n) continue;
                float wj = 0.f;
                int64_t j = 0;
                if (Y32) {
                    j = __ldg(order + m0 + r);
                    wj = wf32 ? ((const float*)w)[j] : __bfloat162float(((const __nv_bfloat16*)w)[j]);
                }
#pragma unroll
                for (int nn = 0; nn < 8; nn++) {
                    int64_t o = rb * 64 + nn * 8 + t * 2;
                    float y0 = acc[mt][nn][2 * h] + (bias ? __bfloat162float(bias[(int64_t)e * O + o]) : 0.f), y1 = acc[mt][nn][2 * h + 1] + (bias ? __bfloat162float(bias[(int64_t)e * O + o + 1]) : 0.f);
                    if (Y32) *(float2*)(Y32 + j * O + o) = make_float2(y0 * wj, y1 * wj);
                    else *(__nv_bfloat162*)(Y + (int64_t)(m0 + r) * O + o) = __floats2bfloat162_rn(y0, y1);
                }
            }
    }
}

template <class Fmt, int MT, int V>
__global__ void __launch_bounds__(256, MT == 1 ? 2 : 1) mma_moe_rows_kernel(Fmt f, int64_t O, int64_t K, const int* __restrict__ plan, int E, int64_t k, int gather, const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ bias, const void* __restrict__ w, int wf32, __nv_bfloat16* __restrict__ Y, float* __restrict__ Y32, int pf) {
    mma_moe_rows_body<Fmt, MT, V>(f, O, K, plan, E, k, gather, X, bias, w, wf32, Y, Y32, pf);
}

// A weighted product's sum: y [T, O] = each token's k rows of y32 [T k, O] added in order, in fp32 (a pair
// routed nowhere adds nothing); 4 columns a thread.
__device__ __forceinline__ void moe_sum_body(const float* __restrict__ y32, const int64_t* __restrict__ ids, int E, int64_t T, int64_t k, int64_t O, __nv_bfloat16* __restrict__ y) {
    int64_t i = (blockIdx.x * (int64_t)blockDim.x + threadIdx.x) * 4;
    if (i >= T * O) return;
    int64_t tok = i / O, c = i - tok * O;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int64_t j = tok * k; j < tok * k + k; j++) {
        int64_t e = __ldg(ids + j);
        if (e < 0 || e >= E) continue;
        float4 u = __ldg((const float4*)(y32 + j * O + c));
        v.x += u.x;
        v.y += u.y;
        v.z += u.z;
        v.w += u.w;
    }
    __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y), hi = __floats2bfloat162_rn(v.z, v.w);
    *(uint2*)(y + i) = make_uint2(*(uint32_t*)&lo, *(uint32_t*)&hi);
}

__global__ void moe_sum_kernel(const float* __restrict__ y32, const int64_t* __restrict__ ids, int E, int64_t T, int64_t k, int64_t O, __nv_bfloat16* __restrict__ y) {
    moe_sum_body(y32, ids, E, T, k, O, y);
}

// ---------------------------------------------------------------------------
// The lossless FP8 (e4m3) weights: glyd_gpu/fp8ref.py is the format (its docstring the layout, byte for byte; gpu/fp8/README.md the choice).
// A matrix [O, K] of bytes (any byte is a value: NaNs, both zeros) is cut into patches of 64 rows by 128 columns (zero bytes past its edges),
// a patch 16 units of 16 x 32, a unit 32 lanes of 16 values, in the order a lane holds the A operand of mma.m16n8k32 with e4m3. A patch is
// packed and unpacked on its own (and each of its row blocks of 16 rows, by the offsets in its head): its core (6 bits a value: sign and mantissa, a
// 2-bit digit), its head (8 words) and its part of var (the values the digit does not name). The pack and the unpack: one CTA of 256 threads a patch,
// a thread a lane of two units (32 values); the 8 KiB patch is staged in shared memory, so that the matrix is read and written in rows.
// Everything between `f8c begin` and `f8c end` is compiled on the CPU, thread by thread, by gpu/fp8/emu_f8.cpp (gpu/fp8/emu_check.py): the
// layout is checked against fp8ref.py there with no GPU.
// f8c begin
#ifndef F8_ALIGN
#define F8_ALIGN(p, n) ((void)0)
#endif
namespace f8c {

constexpr int PR = 64, PC = 128, VALUES = PR * PC;  // a patch: rows, columns, values
constexpr int CORE = VALUES * 6 / 8;                // its core's bytes
constexpr int TAIL = (VALUES - CORE) / 4;           // the words of var a plain patch holds, and the most a coded one's variable part has (less one)
constexpr int HEAD = 8;                             // a head's words
constexpr int MAX_LIT = 1024;                       // a coded patch has fewer literals than this
constexpr int GUARD = 8;                            // zero words after the last patch's part of var
constexpr int NT = 256;                             // threads a CTA: a thread's 32 values are stream bytes 32 tid on (a lane of two units)
constexpr int SCAN_NT = 1024;
static_assert(CORE == 6144 && TAIL == 512 && NT * 32 == VALUES, "the layout of fp8ref.py");

__host__ __device__ constexpr int64_t f8_cdiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

// The pack's arguments: the matrix w [O, K] by rows ld bytes apart, KT patches a row of patches, vec: 16-byte loads allowed.
struct F8Pk {
    const uint8_t* w;
    int64_t ld;
    int O, K, KT, vec;
    int plain;  // layout H: every patch plain (its e4m3 bytes as they are; GLYD_F8H_PLAIN=1, a measurement)
};
// The unpack's: out by rows ld bytes apart; ids (null: the one matrix) the items to take (item i of E: the matrix ids[i] of a stack of E, `per` patches
// each, its rows `item` bytes apart in out); vec: 16-byte stores allowed.
struct F8Un {
    const uint8_t* core;
    const uint32_t* heads;
    const uint32_t* var;
    uint8_t* out;
    const int32_t* ids;
    int64_t ld, item;
    int O, K, KT, per, E, vec;
};

__device__ __forceinline__ uint4 f8_ld16(const void* p) {
    F8_ALIGN(p, 16);
    return __ldcs((const uint4*)p);
}
__device__ __forceinline__ uint2 f8_ld8(const void* p) {
    F8_ALIGN(p, 8);
    return __ldcs((const uint2*)p);
}
__device__ __forceinline__ uint32_t f8_threes(uint32_t d) { return d & (d >> 1) & 0x55555555u; }  // the 2-bit digits of d that are 3: a bit in the low place of each
// The low 2 c bits (c: 0 to 16).
__device__ __forceinline__ uint32_t f8_low2(uint32_t c) { return c >= 16 ? 0xFFFFFFFFu : (1u << (2 * c)) - 1u; }
// The low c nibbles (c: 0 to 8; a funnel shift that clamps), the low c bytes (c: any).
__device__ __forceinline__ uint32_t f8_nibs(uint32_t c) { return __funnelshift_rc(0xFFFFFFFFu, 0u, 32 - 4 * c); }
__device__ __forceinline__ uint32_t f8_bytes(int c) { return c <= 0 ? 0u : c >= 4 ? 0xFFFFFFFFu : (1u << (8 * c)) - 1u; }
// prmt as the PTX instruction is (a selector nibble's top bit: the byte's sign spread), on the selector's low 16 bits as they are (__byte_perm masks them;
// the bits above them are never read, so a selector here may have anything there).
__device__ __forceinline__ uint32_t f8_prmt(uint32_t a, uint32_t b, uint32_t s) {
#if defined(__CUDA_ARCH__)
    uint32_t d;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));
    return d;
#elif defined(__HIP_DEVICE_COMPILE__)
    // (prmt byte by byte: a selector nibble's low 3 bits pick a byte of b:a, its top bit spreads that byte's sign; HIP's __byte_perm has no sign mode)
    const uint64_t x = (uint64_t)b << 32 | a;
    uint32_t d = 0;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const uint32_t n = s >> (4 * i) & 0xFu, v = (uint32_t)(x >> (8 * (n & 7))) & 0xFFu;
        d |= (n & 8 ? (v & 0x80u ? 0xFFu : 0u) : v) << (8 * i);
    }
    return d;
#else
    return __byte_perm(a, b, s);  // (the emulation's: the sign mode of a selector's top bit as prmt's)
#endif
}
// 32 bits of the digits (2 bits each) at s from digit k on; 32 bits of the bit string s from bit p on.
__device__ __forceinline__ uint32_t f8_digits(const uint32_t* s, uint32_t k) { return __funnelshift_r(s[k >> 4], s[(k >> 4) + 1], (k & 15) << 1); }
__device__ __forceinline__ uint32_t f8_bits(const uint32_t* s, uint32_t p) { return __funnelshift_r(s[p >> 5], s[(p >> 5) + 1], p & 31); }
// The 3-bit digits that are 7 among the first n (n: 0 to 16) of the 64 bits lo, hi.
__device__ __forceinline__ uint32_t f8_sevens(uint32_t lo, uint32_t hi, uint32_t n) {
    if (n <= 10) return __popc(lo & (lo >> 1) & (lo >> 2) & 0x09249249u & __funnelshift_rc(0xFFFFFFFFu, 0u, 32 - 3 * n));
    const uint64_t w = (uint64_t)lo | ((uint64_t)hi << 32);
    return (uint32_t)__popcll(w & (w >> 1) & (w >> 2) & 0x1249249249249249ull & ((1ull << (3 * n)) - 1ull));
}
// Where word wc (0 to 31) of row r of a patch is in its 8 KiB of shared memory, as a word index: a row is 32 words, its 8 groups of 4 words in the order
// (group ^ r % 8), so that the 8 rows a half-warp stores or reads a group of do not meet in a bank.
__device__ __forceinline__ int f8_word(int r, int wc) { return r * 32 + ((((wc >> 2) ^ r) & 7) << 2) + (wc & 3); }

// The sum over the CTA's threads before this one, in thread order. tot: 8 words that no other call uses. Every thread calls it; it holds a __syncthreads().
__device__ __forceinline__ uint32_t f8_excl(uint32_t v, uint32_t* tot) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    uint32_t x = v;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const uint32_t y = __shfl_up_sync(0xffffffffu, x, d);
        if (lane >= d) x += y;
    }
    if (lane == 31) tot[w] = x;
    __syncthreads();
    uint32_t b = 0;
    for (int i = 0; i < w; i++) b += tot[i];
    return b + x - v;
}

// A thread's 32 bytes of the patch `patch` as 8 words: row tid / 4, columns 32 (tid % 4) on (0 where they are past the matrix's edge).
__device__ __forceinline__ void f8_load(const F8Pk& P, int64_t patch, int tid, uint32_t (&x)[8]) {
    const int rp = (int)(patch / P.KT), cp = (int)(patch % P.KT);
    const int row = rp * PR + (tid >> 2), col = cp * PC + (tid & 3) * 32;
#pragma unroll
    for (int i = 0; i < 8; i++) x[i] = 0;
    if (row >= P.O) return;
    const uint8_t* s = P.w + (int64_t)row * P.ld + col;
    if (P.vec && col + 32 <= P.K) {
        F8_ALIGN(s, 16);
        const uint4 a = __ldg((const uint4*)s), b = __ldg((const uint4*)(s + 16));
        x[0] = a.x, x[1] = a.y, x[2] = a.z, x[3] = a.w, x[4] = b.x, x[5] = b.y, x[6] = b.z, x[7] = b.w;
    } else {
#pragma unroll
        for (int i = 0; i < 32; i++)
            if (col + i < P.K) x[i >> 2] |= (uint32_t)s[i] << (8 * (i & 3));
    }
}

// Pass 1 of the pack: each patch's head (heads[8 p], its words of var in the first, its base to come from the scan) from the counts of its values'
// exponents, a row block's apart (warps 2 mb and 2 mb + 1 hold row block mb's rows).
__device__ __forceinline__ void f8_head_body(const F8Pk P, uint32_t* heads) {
    __shared__ uint32_t s_cnt[NT / 32][16];
    __shared__ uint32_t s_sum[16];
    __shared__ uint8_t s_sym[16], s_rank[16];
    const int tid = threadIdx.x;
    if (tid < NT / 2) ((uint32_t*)s_cnt)[tid] = 0;
    __syncthreads();
    uint32_t x[8];
    f8_load(P, blockIdx.x, tid, x);
#pragma unroll
    for (int i = 0; i < 8; i++)
#pragma unroll
        for (int k = 0; k < 4; k++) atomicAdd(&s_cnt[tid >> 5][(x[i] >> (8 * k + 3)) & 15], 1u);
    __syncthreads();
    if (tid < 16) {
        uint32_t c = 0;
        for (int w = 0; w < NT / 32; w++) c += s_cnt[w][tid];
        s_sum[tid] = c;
    }
    __syncthreads();
    if (tid < 16) {  // the rank of exponent tid: the exponents before it
        uint32_t r = 0;
        for (int f = 0; f < 16; f++) r += (s_sum[f] > s_sum[tid]) | ((s_sum[f] == s_sum[tid]) & (f > tid));
        s_sym[r] = (uint8_t)tid;
        s_rank[tid] = (uint8_t)r;
    }
    __syncthreads();
    if (tid == 0) {
        uint32_t n1 = 0, n2 = 0, n3 = 0, o2[4], o3[4], o4[4];
        for (int mb = 0; mb < 4; mb++) {
            o2[mb] = n1, o3[mb] = n2, o4[mb] = n3;
            for (int e = 0; e < 16; e++) {
                const uint32_t c = s_cnt[2 * mb][e] + s_cnt[2 * mb + 1][e], r = s_rank[e];
                n1 += r >= 3 ? c : 0u;
                n2 += r >= 6 ? c : 0u;
                n3 += r >= 13 ? c : 0u;
            }
        }
        const uint32_t words = ((n1 + 15) >> 4) + ((3 * n2 + 31) >> 5) + ((n3 + 7) >> 3);
        uint4* h = (uint4*)(heads + (int64_t)blockIdx.x * HEAD);
        if (words >= (uint32_t)TAIL || n3 >= (uint32_t)MAX_LIT) {
            h[0] = make_uint4(TAIL, 0, 0, 1u << 31);
            h[1] = make_uint4(0, 0, 0, 0);
        } else {
            uint32_t s1 = 0, s2 = 0;
            for (int r = 0; r < 8; r++) s1 |= (uint32_t)s_sym[r] << (4 * r);
            for (int r = 0; r < 5; r++) s2 |= (uint32_t)s_sym[8 + r] << (4 * r);
            h[0] = make_uint4(words, s1, s2 | (n3 << 20), n1 | (n2 << 14));
            h[1] = make_uint4(o2[1] | (o2[2] << 16), o2[3] | (o3[1] << 16), o3[2] | (o3[3] << 16), o4[1] | (o4[2] << 10) | (o4[3] << 20));
        }
    }
}

__global__ void __launch_bounds__(NT) f8_head_kernel(const F8Pk P, uint32_t* heads) {
    f8_head_body(P, heads);
}

// Pass 2: heads[8 p] the words of each patch's part of var, to the sum of those before it; the total to *total. One CTA.
__device__ __forceinline__ void f8_scan_body(uint32_t* heads, int64_t patches, uint64_t* total) {
    __shared__ uint32_t s_tot[SCAN_NT / 32];
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    uint64_t carry = 0;
    for (int64_t t0 = 0; t0 < patches; t0 += SCAN_NT) {
        const int64_t i = t0 + tid;
        const uint32_t v = i < patches ? heads[i * HEAD] : 0;
        uint32_t x = v;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const uint32_t y = __shfl_up_sync(0xffffffffu, x, d);
            if (lane >= d) x += y;
        }
        if (lane == 31) s_tot[w] = x;
        __syncthreads();
        uint32_t b = 0, all = 0;
        for (int k = 0; k < SCAN_NT / 32; k++) {
            const uint32_t t = s_tot[k];
            b += k < w ? t : 0u;
            all += t;
        }
        if (i < patches) heads[i * HEAD] = (uint32_t)(carry + b + x - v);
        carry += all;
        __syncthreads();
    }
    if (tid == 0) *total = carry;
}

__global__ void __launch_bounds__(SCAN_NT) f8_scan_kernel(uint32_t* heads, int64_t patches, uint64_t* total) {
    f8_scan_body(heads, patches, total);
}

// A value's place in its lane's core words: its d1 digit's bit (j: 0 to 15), its sign's and its mantissa's in the word of values 8 (j >> 3) on.
__device__ __forceinline__ int f8_d1_bit(int j) { return 4 * (j & 7) + 2 * (j >> 3); }
__device__ __forceinline__ int f8_s_bit(int j) { return (j & 4) ? 8 * (j & 3) + 3 : 8 * (j & 3) + 7; }
__device__ __forceinline__ int f8_m_bit(int j) { return (j & 4) ? 8 * ((j + 3) & 3) + 4 : 8 * (j & 3); }

// Pass 3: each patch's core and part of var (heads from passes 1 and 2). Thread tid: row block mb = tid / 64, lane l = (tid / 2) % 32, units 4 mb + 2 (tid % 2)
// and the next (its 32 stream bytes, 32 tid on).
__device__ __forceinline__ void f8_pack_body(const F8Pk P, const uint32_t* heads, uint8_t* core_out, uint32_t* var, int64_t var_words) {
    __shared__ __align__(16) uint32_t s_tile[VALUES / 4];
    __shared__ uint32_t s_var[TAIL + 8];
    __shared__ uint32_t s_tot[2][NT / 32];
    __shared__ uint8_t s_rank[16];
    const int tid = threadIdx.x, mb = tid >> 6, l = (tid >> 1) & 31, hh = tid & 1, g = l >> 2, t = l & 3;
    const int64_t pat = blockIdx.x;
    const uint4 hd = __ldg((const uint4*)(heads + pat * HEAD));
    const uint32_t base = hd.x, h1 = hd.y, h2 = hd.z, h3 = hd.w;
    const bool plain = (h3 >> 31) != 0;
    const uint32_t n1 = h3 & 0x3FFF, n2 = (h3 >> 14) & 0x3FFF, n3 = h2 >> 20;
    const uint32_t w2 = (n1 + 15) >> 4, w3 = (3 * n2 + 31) >> 5, words = plain ? (uint32_t)TAIL : w2 + w3 + ((n3 + 7) >> 3);
    if (blockIdx.x == 0 && tid < GUARD) var[var_words - GUARD + tid] = 0;
    if ((int64_t)base + words > var_words - GUARD) return;  // (a heads array that is not this matrix's: nothing is written)
    if (tid < 16) {  // the rank of an exponent: its place in the head's list, 13 for any other
        uint32_t r = 13;
        for (int i = 0; i < 13; i++)
            if (((i < 8 ? h1 >> (4 * i) : h2 >> (4 * (i - 8))) & 15) == (uint32_t)tid) r = i;
        s_rank[tid] = (uint8_t)r;
    }
    for (int i = tid; i < TAIL; i += NT) s_var[i] = 0;
    {  // the patch in rows, into shared memory
        uint32_t x[8];
        f8_load(P, pat, tid, x);
        const int row = tid >> 2, wc = (tid & 3) * 8;
        *(uint4*)&s_tile[f8_word(row, wc)] = make_uint4(x[0], x[1], x[2], x[3]);
        *(uint4*)&s_tile[f8_word(row, wc + 4)] = make_uint4(x[4], x[5], x[6], x[7]);
    }
    __syncthreads();
    uint32_t y[2][4];  // unit h (ks = 2 hh + h), value j = 4 q + k: byte k of y[h][q] (the stream's bytes 32 tid on)
#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
        for (int q = 0; q < 4; q++) y[h][q] = s_tile[f8_word(16 * mb + g + 8 * (q & 1), 8 * (2 * hh + h) + t + 4 * (q >> 1))];
    uint8_t* core = core_out + pat * CORE;
    if (plain) {
        if (tid < CORE / 32) {
            *(uint4*)(core + tid * 32) = make_uint4(y[0][0], y[0][1], y[0][2], y[0][3]);
            *(uint4*)(core + tid * 32 + 16) = make_uint4(y[1][0], y[1][1], y[1][2], y[1][3]);
        } else {
            uint32_t* v = var + base + (tid - CORE / 32) * 8;
#pragma unroll
            for (int h = 0; h < 2; h++)
#pragma unroll
                for (int q = 0; q < 4; q++) v[4 * h + q] = y[h][q];
        }
        return;
    }
    uint32_t d1[2] = {0, 0}, sw[2][2] = {{0, 0}, {0, 0}};
    uint32_t c1 = 0, c2 = 0, c3 = 0;
#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
        for (int q = 0; q < 4; q++)
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const int j = 4 * q + k;
                const uint32_t b = (y[h][q] >> (8 * k)) & 0xFF, r = s_rank[(b >> 3) & 15];
                d1[h] |= (r < 3 ? r : 3u) << f8_d1_bit(j);
                sw[h][j >> 3] |= ((b >> 7) << f8_s_bit(j)) | ((b & 7) << f8_m_bit(j));
                c1 += r >= 3, c2 += r >= 6, c3 += r >= 13;
            }
    const uint32_t A = f8_excl(c1 + (c2 << 16), s_tot[0]);  // (the ordinals of the values this thread's are among those that reach each level)
    const uint32_t B = f8_excl(c3, s_tot[1]);
    uint32_t k1 = A & 0xFFFF, k2 = A >> 16, k3 = B;
#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
        for (int q = 0; q < 4; q++)
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const uint32_t e = (y[h][q] >> (8 * k + 3)) & 15, r = s_rank[e];
                if (r >= 3) {
                    atomicOr(&s_var[k1 >> 4], (r < 6 ? r - 3 : 3u) << (2 * (k1 & 15)));
                    k1++;
                    if (r >= 6) {
                        const uint32_t d3 = r < 13 ? r - 6 : 7u, p = 32 * w2 + 3 * k2;
                        atomicOr(&s_var[p >> 5], d3 << (p & 31));
                        if ((p & 31) > 29) atomicOr(&s_var[(p >> 5) + 1], d3 >> (32 - (p & 31)));
                        k2++;
                        if (r >= 13) {
                            atomicOr(&s_var[w2 + w3 + (k3 >> 3)], e << (4 * (k3 & 7)));
                            k3++;
                        }
                    }
                }
            }
    // the core: per unit 32 words of digits, then each lane's 2 words of signs and mantissas
#pragma unroll
    for (int h = 0; h < 2; h++) {
        const int u = 4 * mb + 2 * hh + h;
        *(uint32_t*)(core + u * 384 + l * 4) = d1[h];
        *(uint2*)(core + u * 384 + 128 + l * 8) = make_uint2(sw[h][0], sw[h][1]);
    }
    __syncthreads();
    for (uint32_t i = tid; i < words; i += NT) var[base + i] = s_var[i];
}

__global__ void __launch_bounds__(NT) f8_pack_kernel(const F8Pk P, const uint32_t* heads, uint8_t* core_out, uint32_t* var, int64_t var_words) {
    f8_pack_body(P, heads, core_out, var, var_words);
}

// ---- the decode of a lane of a unit (the unpack's and the fused product's) ----

// The patch's exponents as bytes (exponent e: the byte e << 3): ranks 0-2 and a 0 (lo), 3-5 and the mark 0x80 (l2), 6-9 (l3a), 10-12 and the mark (l3b).
struct F8Tab {
    uint32_t lo, l2, l3a, l3b;
};
__device__ __forceinline__ uint32_t f8_nib4(uint32_t v) {  // v's low 4 nibbles as 4 bytes, each nibble << 3
    uint32_t t = f8_prmt(v, 0, 0x4140);
    t = (t | (t << 4)) & 0x0F0F0F0Fu;
    return t << 3;
}
__device__ __forceinline__ F8Tab f8_tab(uint32_t h1, uint32_t h2) {
    const uint32_t a = f8_nib4(h1), b = f8_nib4(h1 >> 16), c = f8_nib4(h2), d = f8_nib4(h2 >> 16);
    return F8Tab{a & 0x00FFFFFFu, (f8_prmt(a, b, 0x0543) & 0x00FFFFFFu) | 0x80000000u, f8_prmt(b, c, 0x5432), (f8_prmt(c, d, 0x0432) & 0x00FFFFFFu) | 0x80000000u};
}
// 8 digits of 2 bits (v's low 16 bits) as 8 nibbles; 8 digits of 3 bits (v's low 24 bits) as 8 nibbles.
__device__ __forceinline__ uint32_t f8_spread2(uint32_t v) {
    uint32_t y = f8_prmt(v, 0, 0x4140);
    y = (y | (y << 4)) & 0x0F0F0F0Fu;
    return (y | (y << 2)) & 0x33333333u;
}
__device__ __forceinline__ uint32_t f8_spread3(uint32_t v) {
    uint32_t y = (v & 0xFFFu) | ((v << 4) & 0x0FFF0000u);
    y = (y & 0x003F003Fu) | ((y << 2) & 0x3F003F00u);
    return (y & 0x07070707u) | ((y << 1) & 0x70707070u);
}

// The 16 values of a lane of a unit as 4 registers (value 4 q + k in byte k of o[q]: the A operand of mma.m16n8k32 with e4m3) from its d1 word dw, its sign
// and mantissa words s0 and s1, its level-2 digits from its first (b2: 32 bits), its level-3 digits from its first (b3lo, b3hi: 64 bits), and its literals,
// lit(k4) its first (lit(k): the patch's literal k, read only where the lane has one). c1, c2, c3: the digits of levels 2 and 3 and the literals it took.
// Halves a and b: values 0-7 and 8-15. Every step is a few byte permutes and masks a register, whatever the digits (no loop but the literals').
template <class Lit>
__device__ __forceinline__ void f8_lane_unit(const F8Tab& T, uint32_t dw, uint32_t s0, uint32_t s1, uint32_t b2, uint32_t b3lo, uint32_t b3hi, Lit lit, uint32_t k4,
                                             uint32_t (&o)[4], uint32_t& c1, uint32_t& c2, uint32_t& c3) {
    const uint32_t Wa = dw & 0x33333333u, Wb = (dw >> 2) & 0x33333333u;            // the d1 digits, a nibble a value
    const uint32_t Ta = Wa & (Wa >> 1) & 0x11111111u, Tb = Wb & (Wb >> 1) & 0x11111111u;  // the values that spill (digit 3)
    const uint32_t ca = (Ta * 0x11111111u) >> 28, cb = (Tb * 0x11111111u) >> 28;      // how many a half
    const uint32_t xa = f8_spread2(b2), xb = f8_spread2(b2 >> (2 * ca));             // a half's level-2 digits, in the order of its spills
    const uint32_t Ma = xa & (xa >> 1) & 0x11111111u & f8_nibs(ca), Mb = xb & (xb >> 1) & 0x11111111u & f8_nibs(cb);  // the spills that reach level 3
    const uint32_t ma = (Ma * 0x11111111u) >> 28, mb = (Mb * 0x11111111u) >> 28;
    uint32_t sp[4] = {f8_prmt(T.l2, 0, xa), f8_prmt(T.l2, 0, xa >> 16), f8_prmt(T.l2, 0, xb), f8_prmt(T.l2, 0, xb >> 16)};  // the spills' exponents
    c1 = ca + cb;
    c2 = ma + mb;
    c3 = 0;
    if (c2) {  // level 3: the marked spills' exponents (marks 0-7 of the unit from one window, half b's from ma on; past 8, half b's from its own), then each into its spill's place
        const uint32_t y = f8_spread3(b3lo);
        uint32_t L[4], ob = ma;
        L[0] = L[2] = f8_prmt(T.l3a, T.l3b, y), L[1] = L[3] = f8_prmt(T.l3a, T.l3b, y >> 16);
        uint32_t esc = y & (y >> 1) & (y >> 2) & 0x11111111u & f8_nibs(c2 < 8 ? c2 : 8);  // (the marks whose digit is 7, the literals' places)
        if (c2 > 8) {  // (rare)
            const uint32_t yb = f8_spread3(__funnelshift_r(b3lo, b3hi, 3 * ma));
            L[2] = f8_prmt(T.l3a, T.l3b, yb), L[3] = f8_prmt(T.l3a, T.l3b, yb >> 16), ob = 0;
            esc = (y & (y >> 1) & (y >> 2) & 0x11111111u & f8_nibs(ma)) | (yb & (yb >> 1) & (yb >> 2) & 0x11111111u & f8_nibs(mb));
        }
        if (esc) {  // the literals (rare): in order, into the marks' bytes
            const int n[4] = {(int)(ob ? c2 : ma), (int)(ob ? c2 : ma) - 4, ob ? 0 : (int)mb, ob ? 0 : (int)mb - 4};
#pragma unroll
            for (int i = 0; i < 4; i++) {
                uint32_t e = L[i] & f8_bytes(n[i]) & 0x80808080u;
                while (e) {
                    const int bit = __ffs((int)e) - 1;  // 8 byte + 7
                    e &= e - 1;
                    L[i] = (L[i] & ~(0xFFu << (bit - 7))) | (lit(k4 + c3) << (bit - 4));
                    c3++;
                }
            }
            if (ob) L[2] = L[0], L[3] = L[1];
        }
        const uint32_t Ra = Ma * 0x11111110u, Rb = Mb * 0x11111110u + ob * 0x11111111u;  // a spill's marked spills before it (half b: after half a's, in one window)
        sp[0] = f8_prmt(sp[0], f8_prmt(L[0], L[1], Ra), 0x3210 + (Ma << 2));
        sp[1] = f8_prmt(sp[1], f8_prmt(L[0], L[1], Ra >> 16), 0x3210 + (Ma >> 14));
        sp[2] = f8_prmt(sp[2], f8_prmt(L[2], L[3], Rb), 0x3210 + (Mb << 2));
        sp[3] = f8_prmt(sp[3], f8_prmt(L[2], L[3], Rb >> 16), 0x3210 + (Mb >> 14));
    }
    const uint32_t Ra = Ta * 0x11111110u, Rb = Tb * 0x11111110u;  // a value's spills before it in its half
    o[0] = (s0 & 0x87878787u) | f8_prmt(f8_prmt(T.lo, 0, Wa), f8_prmt(sp[0], sp[1], Ra), 0x3210 + (Ta << 2));
    o[1] = (__funnelshift_l(s0, s0, 4) & 0x87878787u) | f8_prmt(f8_prmt(T.lo, 0, Wa >> 16), f8_prmt(sp[0], sp[1], Ra >> 16), 0x3210 + (Ta >> 14));
    o[2] = (s1 & 0x87878787u) | f8_prmt(f8_prmt(T.lo, 0, Wb), f8_prmt(sp[2], sp[3], Rb), 0x3210 + (Tb << 2));
    o[3] = (__funnelshift_l(s1, s1, 4) & 0x87878787u) | f8_prmt(f8_prmt(T.lo, 0, Wb >> 16), f8_prmt(sp[2], sp[3], Rb >> 16), 0x3210 + (Tb >> 14));
}

// The unpack: each patch, or (ids) each patch of the stacked matrices ids[blockIdx.y], into out. Thread tid as the pack's: two lanes of units, its 32 stream bytes.
__device__ __forceinline__ void f8_unpack_body(const F8Un P) {
    __shared__ __align__(16) uint32_t s_tile[VALUES / 4];
    __shared__ uint32_t s_var[TAIL + 8];
    __shared__ uint32_t s_tot[3][NT / 32];
    const int tid = threadIdx.x, mb = tid >> 6, l = (tid >> 1) & 31, hh = tid & 1, g = l >> 2, t = l & 3;
    const int pl = blockIdx.x;
    int64_t pat = pl, item = 0;
    if (P.ids) {
        item = blockIdx.y;
        const int id = P.ids[item];
        if ((unsigned)id >= (unsigned)P.E) return;  // (no such matrix: its place in out is left as it is)
        pat += (int64_t)id * P.per;
    }
    const uint4 hd = __ldg((const uint4*)(P.heads + pat * HEAD));
    const uint32_t base = hd.x, h1 = hd.y, h2 = hd.z, h3 = hd.w;
    const uint8_t* core = P.core + pat * CORE;
    uint32_t o[2][4];  // unit h (ks = 2 hh + h): value 4 q + k at byte k of o[h][q]
    if (h3 >> 31) {  // a plain patch: its stream's bytes
        uint32_t x[8];
        if (tid < CORE / 32) {
            const uint4 a = f8_ld16(core + tid * 32), b = f8_ld16(core + tid * 32 + 16);
            x[0] = a.x, x[1] = a.y, x[2] = a.z, x[3] = a.w, x[4] = b.x, x[5] = b.y, x[6] = b.z, x[7] = b.w;
        } else {
            const uint32_t* v = P.var + base + (tid - CORE / 32) * 8;
#pragma unroll
            for (int i = 0; i < 8; i++) x[i] = __ldg(v + i);
        }
#pragma unroll
        for (int i = 0; i < 8; i++) o[i >> 2][i & 3] = x[i];
    } else {
        const uint32_t n1 = h3 & 0x3FFF, n2 = (h3 >> 14) & 0x3FFF, n3 = h2 >> 20;
        const uint32_t w2 = (n1 + 15) >> 4, w3 = (3 * n2 + 31) >> 5, words = w2 + w3 + ((n3 + 7) >> 3);
        uint32_t dw[2];
        uint2 sw[2];
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int u = 4 * mb + 2 * hh + h;
            dw[h] = __ldg((const uint32_t*)(core + u * 384 + l * 4));
            sw[h] = f8_ld8(core + u * 384 + 128 + l * 8);
        }
        for (uint32_t i = tid; i < words; i += NT) s_var[i] = __ldg(P.var + base + i);
        if (tid < 8) s_var[words + tid] = 0;  // (a read of the digits runs a word or two past them)
        const F8Tab T = f8_tab(h1, h2);
        const uint32_t c1a = __popc(f8_threes(dw[0])), c1b = __popc(f8_threes(dw[1]));
        const uint32_t K1 = f8_excl(c1a + c1b, s_tot[0]);  // (its __syncthreads has s_var in)
        const uint32_t b2[2] = {f8_digits(s_var, K1), f8_digits(s_var, K1 + c1a)};
        const uint32_t c2a = __popc(f8_threes(b2[0]) & f8_low2(c1a)), c2b = __popc(f8_threes(b2[1]) & f8_low2(c1b));
        const uint32_t K2 = f8_excl(c2a + c2b, s_tot[1]);
        const uint32_t p3[2] = {32 * w2 + 3 * K2, 32 * w2 + 3 * (K2 + c2a)};
        uint32_t b3[2][2];
#pragma unroll
        for (int h = 0; h < 2; h++) b3[h][0] = f8_bits(s_var, p3[h]), b3[h][1] = f8_bits(s_var, p3[h] + 32);
        const uint32_t c3a = f8_sevens(b3[0][0], b3[0][1], c2a), c3b = f8_sevens(b3[1][0], b3[1][1], c2b);
        const uint32_t K3 = f8_excl(c3a + c3b, s_tot[2]);
        const uint32_t* sl = s_var + w2 + w3;
        auto lit = [sl](uint32_t k) { return (sl[k >> 3] >> (4 * (k & 7))) & 15u; };
        uint32_t c1, c2, c3;
        f8_lane_unit(T, dw[0], sw[0].x, sw[0].y, b2[0], b3[0][0], b3[0][1], lit, K3, o[0], c1, c2, c3);
        f8_lane_unit(T, dw[1], sw[1].x, sw[1].y, b2[1], b3[1][0], b3[1][1], lit, K3 + c3a, o[1], c1, c2, c3);
    }
    // the patch in rows, through shared memory, to out
#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
        for (int q = 0; q < 4; q++) s_tile[f8_word(16 * mb + g + 8 * (q & 1), 8 * (2 * hh + h) + t + 4 * (q >> 1))] = o[h][q];
    __syncthreads();
    const int row0 = (pl / P.KT) * PR, col0 = (pl % P.KT) * PC;
    uint8_t* dst = P.out + item * P.item + (int64_t)row0 * P.ld + col0;
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int c = tid + i * NT, row = c >> 3, g8 = c & 7;
        if (row0 + row >= P.O) continue;
        const uint4 v = *(const uint4*)&s_tile[row * 32 + (((g8 ^ row) & 7) << 2)];
        uint8_t* d = dst + (int64_t)row * P.ld + g8 * 16;
        if (P.vec && col0 + g8 * 16 + 16 <= P.K) {
            F8_ALIGN(d, 16);
            *(uint4*)d = v;
        } else {
            const uint32_t b[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int k = 0; k < 16; k++)
                if (col0 + g8 * 16 + k < P.K) d[k] = (uint8_t)(b[k >> 2] >> (8 * (k & 3)));
        }
    }
}

__global__ void __launch_bounds__(NT) f8_unpack_kernel(const F8Un P) {
    f8_unpack_body(P);
}

// ---- the fused product: Y [M, N] = X [M, K] W^T for 1 to 64 tokens, W [N, K] in this layout and its scales, W never written out ----
// The weights' bytes go into the tensor cores' A operand as bf16 (W8A16, as vLLM's Marlin FP8: an e4m3 byte's sign, exponent and mantissa in a bf16's
// places, 2^-120 times its value exactly; X bf16, times 2^64 (exact); products and sums in fp32; the scales and 2^56 applied to the sums). A warp takes a row block
// (16 rows) of W, a CTA 8 of them (a row group of 128 rows); a row block of a patch is a "granule": its head, its 4 units' core words and its variable
// part, decoded by f8_lane_unit into 16 registers a lane, then 2 m16n8k16 products a unit and 8 tokens. The work is the row groups' patches in order
// (row group by row group, each across K), cut into as many equal parts as CTAs are resident at once: one wave, no CTA idle at its end. A CTA steps
// through its part a patch at a time; at each step it copies (cp.async, never waited for at once) what the next step reads: X's 128 columns of the
// next patch, each warp's core and variable part of its next granule, and the heads two patches on (which say where the variable parts are). Where a
// row group's K is cut between CTAs, each leaves its sums in parts and the last to finish adds them in the CTAs' order (the same result every run).

#ifndef F8_EXTERN_SMEM
#define F8_EXTERN_SMEM(T, name) extern __shared__ __align__(16) T name[]
#endif

constexpr int F8_BLOCK = 0, F8_TENSOR = 1, F8_MX = 2, F8_DS32 = 3, F8_CHANNEL = 4;  // the scales: fp32 per 128 x 128, one fp32, e8m0 per 1 x 32, e8m0 per 32 x 32,
                                                                       // fp32 a row
constexpr int F8_A8 = 8;  // KIND | F8_A8 (block, tensor, channel): W8A8, X as e4m3 a token and 128 columns (f8_xq_kernel's) into mma.m16n8k32 e4m3
constexpr int LW = 8, LNT = 32 * LW;  // the product's warps a CTA (a row block each), threads
constexpr int CSL = 384;              // words of a row block's core (4 units of 384 bytes): a slot of the warp's two
constexpr int VB = 80;                // words of a slot of the warp's two for a row block's variable part; a longer part is read where it is in var
constexpr int HSL = 3, XSL = 2;       // the CTA's slots of heads (its two rows of patches at a column) and of X (a patch's 128 columns of the tokens)
constexpr int XROW = PC * 2;          // bytes a token's row of an X slot (its 16-byte pieces swizzled by the row's parity: rows g and g + 1 in other banks)
constexpr int LSMEM = 4 * (HSL * 16 + LW * (2 * CSL + 2 * VB));
// (S slots of core, variable parts and X a warp or CTA, heads 2 S - 3 steps ahead: S = 2 as above; 3, one token on an HBM GPU, GLYD_F8_SLOTS=3: two steps' copies in flight, not one)
template <int S> constexpr int f8_hs() { return S == 2 ? HSL : 2 * S - 1; }
template <int S> constexpr int f8_xs() { return S == 2 ? XSL : S; }
template <int S, bool H = false> constexpr int f8_csl() { return H ? 32 * 14 : CSL; }  // (H: layout H's row block, HCSL)
template <int S, bool H = false> constexpr int f8_lsmem() { return 4 * (f8_hs<S>() * 16 + LW * (S * f8_csl<S, H>() + S * VB)); }  // bytes of shared memory before the X slots (XSL slots of M rows of XROW bytes); DENSE: before
                                                                  // the warps' buffers of exponents (LW of EBS bytes), the X slots after them
// Bytes of an X slot of M tokens: bf16 rows of XROW bytes; A8: e4m3 rows of 128 bytes (a row's groups of 32 columns at group ^ (row & 3)), then the M scales.
__host__ __device__ constexpr int f8_xslot(int M, bool a8) { return a8 ? (M * (PC + 4) + 15) & ~15 : M * XROW; }

struct F8Lin {
    const uint8_t* core;
    const uint32_t* heads;
    const uint32_t* var;
    const void* scale;  // F8_BLOCK: float [ceil(N / 128)] rows lds apart; F8_TENSOR: float [1]; F8_MX: uint8 [N] rows lds apart; F8_DS32: uint8 [ceil(N / 32)] rows lds apart;
                        // F8_CHANNEL: float [N]
    const uint16_t* x;  // bf16 [M, K], rows ldx apart
    uint16_t* y;        // bf16 [M, N], rows ldy apart
    const uint8_t* xq;  // F8_A8: X as e4m3 [M][KT 128] in the B fragments' order, and its scales sx [M][KT] (f8_xq_kernel's)
    const float* sx;
    float* parts;       // ctas > 1: [2 ctas][M][128], the sums of a CTA's first and last row group where it holds part of its K
    int* done;          // ctas > 1: a counter a row group of 128 rows, zero (the last CTA leaves it zero)
    int64_t lds, ldx, ldy;
    int M, N, K, KT, RG, ctas, xvec;  // KT: patches across K; RG: row groups (of 128 rows); ctas: the grid; xvec: X's rows 16-byte aligned
    // The experts' product (glyd_gpu_f8_moe; null plan: one matrix): the stack of E matrices [N, K] as one [E N, K] (block scales [E][ceil(N / 128)] rows lds apart), blockIdx.y the
    // plan's slot (moe_route_kernel's: its expert, its pairs' rows of X and Y, M of them at most); gather: X's row of a pair is its token's (order[m] / gk), else the pair's own in the
    // plan's order. f8_slot sets order, and each pointer at the slot's expert and rows.
    const int* plan;
    const int* order;
    int E, gk, gather;
};

// cp.async of 4, 8 and 16 bytes (global to shared, not waited for: the 16 bytes past L1), a commit of the thread's copies so far, a wait for all of them
// (the emulation: copies at once).
__device__ __forceinline__ void f8_cp4(uint32_t* dst, const uint32_t* src) {
#if defined(__CUDA_ARCH__)
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src));
#else
    *dst = *src;
#endif
}
__device__ __forceinline__ void f8_cp8(uint32_t* dst, const uint32_t* src) {
#if defined(__CUDA_ARCH__)
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src));
#else
    dst[0] = src[0], dst[1] = src[1];
#endif
}
__device__ __forceinline__ void f8_cp16(void* dst, const void* src) {
#if defined(__CUDA_ARCH__)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src));
#else
    F8_ALIGN(dst, 16);
    F8_ALIGN(src, 16);
    *(uint4*)dst = *(const uint4*)src;
#endif
}
__device__ __forceinline__ void f8_cp_commit() {
#if defined(__CUDA_ARCH__)
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
__device__ __forceinline__ void f8_cp_wait0() {
#if defined(__CUDA_ARCH__)
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void f8_cp_wait1() {  // all but the newest group in
#if defined(__CUDA_ARCH__)
    asm volatile("cp.async.wait_group 1;\n" ::: "memory");
#endif
}
template <int N>
__device__ __forceinline__ void f8_cp_wait() {
#if defined(__CUDA_ARCH__)
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
#endif
}

// e4m3 bytes 0 and 1 (lo) or 2 and 3 (hi) of o as two bf16 (low half: the first), each 2^-120 times the byte's value: its sign, exponent and mantissa
// in a bf16's places (the byte and its sign spread to 16 bits by prmt, a shift and a mask).
__device__ __forceinline__ uint32_t f8_bf_lo(uint32_t o) { return (f8_prmt(o, 0, 0x9180) << 4) & 0x87F087F0u; }
__device__ __forceinline__ uint32_t f8_bf_hi(uint32_t o) { return (f8_prmt(o, 0, 0xB3A2) << 4) & 0x87F087F0u; }
// Two bf16 of X times 2^64, exactly (for |x| below 2^64): with the weights' 2^-120 the products are 2^-56 times the true ones, in fp32's normal range for
// |x| down to 2^-64 (at 2^-120 alone, activations of about 2^-16 and below gave products in fp32's subnormal range, where the sums lose bits).
__device__ __forceinline__ uint32_t f8_x64(uint32_t v) {
#if defined(__CUDA_ARCH__)
    uint32_t r;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(r) : "r"(v), "r"(0x5F805F80u), "r"(0x80008000u));  // (v 2^64 + -0)
    return r;
#else
    const auto one = [](uint32_t h) { return __float_as_uint(__int_as_float((h & 0xFFFFu) << 16) * 0x1p64f) >> 16; };
    return one(v) | (one(v >> 16) << 16);
#endif
}

// (mma.sync is warp-collective and these asm statements are not volatile, so the compiler may move or merge them: every guard around a call must stay
// uniform across the warp, as each is.)
__device__ __forceinline__ void f8_mma(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__)
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#elif defined(F8_EMU)
    f8_emu_mma(c, a0, a1, a2, a3, b0, b1);
#elif defined(__HIP_DEVICE_COMPILE__)
    const uint32_t a[4] = {a0, a1, a2, a3};
    glyd_mma16816(c, a, b0, b1);
#endif
}

// mma.sync.m16n8k32 with e4m3 A and B (Ada, Hopper and Blackwell): A the decoded registers as they are.
__device__ __forceinline__ void f8_mma8(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#elif defined(F8_EMU)
    f8_emu_mma8(c, a0, a1, a2, a3, b0, b1);
#elif defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    __trap();  // (no e4m3 MMA before compute capability 8.9, nor on AMD: the host never asks for it there, nor where the code loaded is older)
#endif
}
// Two floats as e4m3 bytes (lo: the first), rounded to nearest even, past 448 saturated to it.
__device__ __forceinline__ uint32_t f8_e4m3x2(float lo, float hi) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    uint16_t r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
    return r;
#elif defined(F8_EMU)
    return f8_emu_e4m3(lo) | (f8_emu_e4m3(hi) << 8);
#else
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    __trap();
#endif
    return 0;
#endif
}
// W8A8's X: each token's 128 columns of K (zeros past K, to KT 128) as e4m3 of 448 over their largest magnitude, the bytes of a group of 32 columns in the
// order the B fragments want (4-column pieces i at 2 (i & 3) + (i >> 2)), into xq [M][KT 128]; the scale back (the largest over 448) into sx [M][KT]. A warp a
// token's 128 columns (a lane 4).
__global__ void __launch_bounds__(128) f8_xq_kernel(const uint16_t* x, int64_t ldx, int M, int K, int KT, uint8_t* xq, float* sx) {
    const int w = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (w >= M * KT) return;
    const int m = w / KT, kb = w - m * KT, k = kb * PC + 4 * lane;
    float f[4], mx = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) f[i] = k + i < K ? __int_as_float((uint32_t)x[(int64_t)m * ldx + k + i] << 16) : 0.f, mx = fmaxf(mx, fabsf(f[i]));
#pragma unroll
    for (int d = 16; d >= 1; d >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, d));
    const float inv = mx > 0.f ? 448.f / mx : 0.f;
    const int pc = lane & 7;
    *(uint32_t*)(xq + ((int64_t)m * KT + kb) * PC + (lane >> 3) * 32 + 4 * (2 * (pc & 3) + (pc >> 2))) = f8_e4m3x2(f[0] * inv, f[1] * inv) | (f8_e4m3x2(f[2] * inv, f[3] * inv) << 16);
    if (lane == 0) sx[(int64_t)m * KT + kb] = mx / 448.f;
}

// A float to bf16 bits, rounded to the nearest even (a NaN stays a NaN).
__device__ __forceinline__ uint16_t f8_to_bf16(float v) {
    uint32_t u = __float_as_uint(v);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) return (uint16_t)((u >> 16) | 0x40u);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

// The sum of v over the warp's lanes before this one.
__device__ __forceinline__ uint32_t f8_wexcl(uint32_t v, int lane) {
    uint32_t x = v;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const uint32_t y = __shfl_up_sync(0xffffffffu, x, d);
        if (lane >= d) x += y;
    }
    return x - v;
}

// An e8m0 scale byte as the float 2^(E - 127).
__device__ __forceinline__ float f8_e8m0(uint32_t e) { return __int_as_float(e ? e << 23 : 0x00400000u); }

// Row block mb's place in its patch's variable part (head hA, hB): its first ordinals at levels 2, 3 and the literals (o2, o3, o4), whether it has
// literals (lits), the words of each run it starts at (a2, a3, a4: word w2 + a3 of the patch's part is its level 3's first, w2 + w3 + a4 its literals'),
// and where they are in a copy of it (level 2 at 0, level 3 at v3, the literals at v4; all: its words, with 2 after the level-2 and level-3 runs: a lane's
// reads may touch 2 words past a run, and 3 past level 3's (its second window) where the lane is past the row block's last mark, a value then unused).
struct F8Geo {
    uint32_t w2, w3, o2, o3, o4, lits, a2, a3, a4, v3, v4, all, n1b;  // (n1b: the row block's values that reach level 2)
};
__device__ __forceinline__ F8Geo f8_geo(const uint4& hA, const uint4& hB, int mb) {
    F8Geo G;
    const uint32_t n1 = hA.w & 0x3FFF, n2 = (hA.w >> 14) & 0x3FFF, n3 = hA.z >> 20;
    uint32_t e2, e3, e4;
    G.w2 = (n1 + 15) >> 4, G.w3 = (3 * n2 + 31) >> 5, G.o2 = G.o3 = G.o4 = 0;
    if (mb == 0) e2 = hB.x & 0xFFFF, e3 = hB.y >> 16, e4 = hB.w & 0x3FF;
    else if (mb == 1) G.o2 = hB.x & 0xFFFF, G.o3 = hB.y >> 16, G.o4 = hB.w & 0x3FF, e2 = hB.x >> 16, e3 = hB.z & 0xFFFF, e4 = (hB.w >> 10) & 0x3FF;
    else if (mb == 2) G.o2 = hB.x >> 16, G.o3 = hB.z & 0xFFFF, G.o4 = (hB.w >> 10) & 0x3FF, e2 = hB.y & 0xFFFF, e3 = hB.z >> 16, e4 = hB.w >> 20;
    else G.o2 = hB.y & 0xFFFF, G.o3 = hB.z >> 16, G.o4 = hB.w >> 20, e2 = n1, e3 = n2, e4 = n3;
    const uint32_t z2 = (e2 + 15) >> 4, z3 = (3 * e3 + 31) >> 5, z4 = (e4 + 7) >> 3;
    G.lits = e4 != G.o4, G.n1b = e2 - G.o2;
    G.a2 = G.o2 >> 4, G.a3 = (3 * G.o3) >> 5, G.a4 = G.o4 >> 3;
    G.v3 = z2 - G.a2 + 2, G.v4 = G.v3 + z3 - G.a3 + 2, G.all = G.v4 + z4 - G.a4;
    return G;
}

// A warp's copies of row block mb of patch p for its decode a step or two later: its core (1536 bytes, 3 cp.async of 16 bytes a lane; whatever the
// patch: a plain one's are not read) into cw, and (its head hA, hB in) its variable part, where it fits, into vw (a plain patch's: read at the decode).
__device__ __forceinline__ void f8_core_in(const F8Lin& P, int64_t p, int mb, int lane, uint32_t* cw) {
    const uint4* cs = (const uint4*)(P.core + p * CORE + mb * 4 * 384);
#pragma unroll
    for (int i = 0; i < 3; i++) f8_cp16(cw + 4 * (lane + 32 * i), cs + lane + 32 * i);
}
__device__ __forceinline__ void f8_var_in(const F8Lin& P, int mb, int lane, const uint4& hA, const uint4& hB, uint32_t* vw) {
    if (hA.w >> 31) return;
    const F8Geo G = f8_geo(hA, hB, mb);
    if (G.all > (uint32_t)VB) return;
    const uint32_t* src = P.var + hA.x;
    for (uint32_t i = lane; i < G.all; i += 32) f8_cp4(vw + i, src + (i < G.v3 ? G.a2 + i : i < G.v4 ? G.w2 + G.a3 + (i - G.v3) : G.w2 + G.w3 + G.a4 + (i - G.v4)));
}

// Row block mb of patch p (16 rows x 128 columns: its units 4 mb + ks) into a lane's 16 registers o[ks][q], from its head (hA, hB) and its copies
// (f8_core_in's and f8_var_in's, in: cw its core, vw its variable part where it fit, else var itself is read).
__device__ __forceinline__ void f8_granule(const F8Lin& P, int64_t p, int mb, int lane, const uint4& hA, const uint4& hB, const uint32_t* cw, const uint32_t* vw,
                                           uint32_t (&o)[4][4]) {
    if (hA.w >> 31) {  // plain: the lane's 64 stream bytes (row blocks 0-2 in core, 16-byte aligned; 3 in var, at any word: read by words)
        if (mb < 3) {
            const uint4* s = (const uint4*)(P.core + p * CORE + 2048 * mb + 64 * lane);
#pragma unroll
            for (int ks = 0; ks < 4; ks++) {
                const uint4 v = __ldg(s + ks);
                o[ks][0] = v.x, o[ks][1] = v.y, o[ks][2] = v.z, o[ks][3] = v.w;
            }
        } else {
            const uint32_t* s = P.var + hA.x + 16 * lane;
#pragma unroll
            for (int i = 0; i < 16; i++) o[i >> 2][i & 3] = __ldg(s + i);
        }
        return;
    }
    const F8Geo G = f8_geo(hA, hB, mb);
    const bool in = G.all <= (uint32_t)VB;
    const uint32_t* src = P.var + hA.x;
    const uint32_t* p2 = in ? vw : src + G.a2;                // level 2's digits, from the row block's first word of them
    const uint32_t* p3 = in ? vw + G.v3 : src + G.w2 + G.a3;  // level 3's
    const uint32_t* p4 = in ? vw + G.v4 : src + G.w2 + G.w3 + G.a4;  // the literals'
    uint32_t dw[4], c1[4], t1 = 0;
    uint2 sw[4];
#pragma unroll
    for (int ks = 0; ks < 4; ks++) {
        dw[ks] = cw[ks * 96 + lane];
        sw[ks] = ((const uint2*)(cw + ks * 96 + 32))[lane];
        c1[ks] = __popc(f8_threes(dw[ks]));
        t1 += c1[ks];
    }
    const F8Tab T = f8_tab(hA.y, hA.z);
    uint32_t k1 = G.o2 - 16 * G.a2 + f8_wexcl(t1, lane);  // the lane's first level-2 digit from p2
    uint32_t b2[4], m[4], t2 = 0;
#pragma unroll
    for (int ks = 0; ks < 4; ks++) {
        b2[ks] = f8_digits(p2, k1);
        k1 += c1[ks];
        m[ks] = __popc(f8_threes(b2[ks]) & f8_low2(c1[ks]));
        t2 += m[ks];
    }
    uint32_t q3 = 3 * (G.o3 + f8_wexcl(t2, lane)) - 32 * G.a3;  // the lane's first level-3 digit's bit from p3
    uint32_t b3[4][2], c3[4] = {0, 0, 0, 0};
#pragma unroll
    for (int ks = 0; ks < 4; ks++) {
        b3[ks][0] = f8_bits(p3, q3), b3[ks][1] = f8_bits(p3, q3 + 32);
        q3 += 3 * m[ks];
    }
    uint32_t k4 = 0;
    if (G.lits) {  // (literals are rare: their counts and scan only where the row block has one)
        uint32_t t3 = 0;
#pragma unroll
        for (int ks = 0; ks < 4; ks++) t3 += c3[ks] = f8_sevens(b3[ks][0], b3[ks][1], m[ks]);
        k4 = G.o4 - 8 * G.a4 + f8_wexcl(t3, lane);
    }
    auto lit = [p4](uint32_t k) { return (p4[k >> 3] >> (4 * (k & 7))) & 15u; };
#pragma unroll
    for (int ks = 0; ks < 4; ks++) {
        uint32_t x1, x2, x3;
        f8_lane_unit(T, dw[ks], sw[ks].x, sw[ks].y, b2[ks], b3[ks][0], b3[ks][1], lit, k4, o[ks], x1, x2, x3);
        k4 += c3[ks];
    }
}

// The same granule decoded in two passes (DENSE): first the row block's spills' exponents in stream order, a lane 24 of them (their level-2 digits by a
// byte permute, the level-3 and literal ones of its marks after a warp scan of the marks), into the warp's buffer eb; then each lane's units from them
// (its spills' bytes read from eb at its first spill, then put in place by byte permutes, as f8_lane_unit does). A row block with more spills than eb
// holds, or a lane with more than 8 marks (both rare), and a plain patch: f8_granule.
// between(ks), once unit ks is in o (AHEAD: the products of the granule before, one unit at a time, among the decode's instructions).
constexpr int ES = 24, EB = 32 * ES, EBS = EB + 16;  // spills a lane of the first pass, bytes of a warp's buffer, its stride (the reads past its end land in it)
template <class Between>
__device__ __forceinline__ void f8_granule_dense(const F8Lin& P, int64_t p, int mb, int lane, const uint4& hA, const uint4& hB, const uint32_t* cw,
                                                 const uint32_t* vw, uint32_t* eb, uint32_t (&o)[4][4], Between&& between) {
    auto whole = [&]() {  // (f8_granule's decode)
        f8_granule(P, p, mb, lane, hA, hB, cw, vw, o);
#pragma unroll
        for (int ks = 0; ks < 4; ks++) between(ks);
    };
    if (hA.w >> 31) return whole();
    const F8Geo G = f8_geo(hA, hB, mb);
    if (G.n1b > (uint32_t)EB) return whole();
    const bool in = G.all <= (uint32_t)VB;
    const uint32_t* src = P.var + hA.x;
    const uint32_t* p2 = in ? vw : src + G.a2;
    const uint32_t* p3 = in ? vw + G.v3 : src + G.w2 + G.a3;
    const uint32_t* p4 = in ? vw + G.v4 : src + G.w2 + G.w3 + G.a4;
    const F8Tab T = f8_tab(hA.y, hA.z);
    // pass 1: spills ES lane .. ES lane + v - 1 of the row block
    const uint32_t k0 = ES * lane, v = k0 < G.n1b ? min(G.n1b - k0, (uint32_t)ES) : 0u;
    const uint32_t kd = G.o2 - 16 * G.a2 + (v ? k0 : 0u);  // (its first level-2 digit from p2; a lane with none reads the row block's first)
    const uint32_t* q = p2 + (kd >> 4);
    const uint32_t q0 = q[0], q1 = q[1], q2 = q[2], sh = 2 * (kd & 15);
    const uint32_t lo = __funnelshift_r(q0, q1, sh), hi = __funnelshift_r(q1, q2, sh);
    const uint32_t x0 = f8_spread2(lo), x1 = f8_spread2(lo >> 16), x2 = f8_spread2(hi);  // its 24 digits, a nibble each
    const uint32_t m0 = x0 & (x0 >> 1) & 0x11111111u & f8_nibs(min(v, 8u)), m1 = x1 & (x1 >> 1) & 0x11111111u & f8_nibs(v > 8 ? min(v - 8, 8u) : 0u),
                   m2 = x2 & (x2 >> 1) & 0x11111111u & f8_nibs(v > 16 ? v - 16 : 0u);  // its marks (digit 3)
    const uint32_t cm = __popc(m0) + __popc(m1) + __popc(m2);
    if (__any_sync(0xffffffffu, cm > 8)) return whole();
    const uint32_t c1r = __popc(m0 & 0xFFFFu), c1 = __popc(m0), c2r = c1 + __popc(m1 & 0xFFFFu), c2 = c1 + __popc(m1), c3r = c2 + __popc(m2 & 0xFFFFu);
    const uint32_t mo = f8_wexcl(cm, lane);  // its first mark of the row block's
    const uint32_t y = f8_spread3(f8_bits(p3, 3 * (G.o3 + mo) - 32 * G.a3));  // its marks' level-3 digits (8), a nibble each
    uint32_t L0 = f8_prmt(T.l3a, T.l3b, y), L1 = f8_prmt(T.l3a, T.l3b, y >> 16);  // their exponents (0x80: a literal)
    if (G.lits) {  // (the literals: in order, into the marks' bytes; rare)
        uint32_t e0 = L0 & f8_bytes((int)min(cm, 4u)) & 0x80808080u, e1 = L1 & f8_bytes(cm > 4 ? (int)cm - 4 : 0) & 0x80808080u;
        uint32_t k4 = G.o4 - 8 * G.a4 + f8_wexcl(__popc(e0) + __popc(e1), lane);
        while (e0) {
            const int bit = __ffs((int)e0) - 1;
            e0 &= e0 - 1;
            L0 = (L0 & ~(0xFFu << (bit - 7))) | ((((p4[k4 >> 3] >> (4 * (k4 & 7))) & 15u) << 3) << (bit - 7));
            k4++;
        }
        while (e1) {
            const int bit = __ffs((int)e1) - 1;
            e1 &= e1 - 1;
            L1 = (L1 & ~(0xFFu << (bit - 7))) | ((((p4[k4 >> 3] >> (4 * (k4 & 7))) & 15u) << 3) << (bit - 7));
            k4++;
        }
    }
    // the exponents of its 24 spills: level 2's, a mark's from L by its rank among the lane's marks
    uint32_t e[6] = {f8_prmt(T.l2, 0, x0), f8_prmt(T.l2, 0, x0 >> 16), f8_prmt(T.l2, 0, x1), f8_prmt(T.l2, 0, x1 >> 16), f8_prmt(T.l2, 0, x2), f8_prmt(T.l2, 0, x2 >> 16)};
    const uint32_t mw[6] = {m0 & 0xFFFFu, m0 >> 16, m1 & 0xFFFFu, m1 >> 16, m2 & 0xFFFFu, m2 >> 16}, pre[6] = {0, c1r, c1, c2r, c2, c3r};
#pragma unroll
    for (int i = 0; i < 6; i++) e[i] = f8_prmt(e[i], f8_prmt(L0, L1, mw[i] * 0x1110u + pre[i] * 0x1111u), 0x3210 + (mw[i] << 2));
    __syncwarp();  // (eb: every lane past the granule before)
    uint2* ed = (uint2*)(eb + ES / 4 * lane);
    ed[0] = make_uint2(e[0], e[1]), ed[1] = make_uint2(e[2], e[3]), ed[2] = make_uint2(e[4], e[5]);
    // pass 2: each unit's values from its core words and its spills' bytes in eb
    uint32_t dw[4], c1u[4], t1 = 0;
    uint2 sw[4];
#pragma unroll
    for (int ks = 0; ks < 4; ks++) {
        dw[ks] = cw[ks * 96 + lane];
        sw[ks] = ((const uint2*)(cw + ks * 96 + 32))[lane];
        c1u[ks] = __popc(f8_threes(dw[ks]));
        t1 += c1u[ks];
    }
    uint32_t s = f8_wexcl(t1, lane);  // the lane's first spill of the row block's
    __syncwarp();  // (eb written by every lane)
    const uint8_t* e8 = (const uint8_t*)eb;
#pragma unroll
    for (int ks = 0; ks < 4; ks++) {
        const uint32_t Wa = dw[ks] & 0x33333333u, Wb = (dw[ks] >> 2) & 0x33333333u;
        const uint32_t Ta = Wa & (Wa >> 1) & 0x11111111u, Tb = Wb & (Wb >> 1) & 0x11111111u;
        const uint32_t ca = (Ta * 0x11111111u) >> 28;
        uint32_t sp[4];  // half a's spills' bytes (from s), half b's (from s + ca)
#pragma unroll
        for (int hb = 0; hb < 2; hb++) {
            const uint32_t at = s + (hb ? ca : 0u), r = (at & 3) * 0x1111u + 0x3210u;
            const uint32_t* ew = (const uint32_t*)(e8 + (at & ~3u));
            const uint32_t u0 = ew[0], u1 = ew[1], u2 = ew[2];
            sp[2 * hb] = f8_prmt(u0, u1, r), sp[2 * hb + 1] = f8_prmt(u1, u2, r);
        }
        s += c1u[ks];
        const uint32_t Ra = Ta * 0x11111110u, Rb = Tb * 0x11111110u;  // a value's spills before it in its half
        const uint32_t s0 = sw[ks].x, s1 = sw[ks].y;
        o[ks][0] = (s0 & 0x87878787u) | f8_prmt(f8_prmt(T.lo, 0, Wa), f8_prmt(sp[0], sp[1], Ra), 0x3210 + (Ta << 2));
        o[ks][1] = (__funnelshift_l(s0, s0, 4) & 0x87878787u) | f8_prmt(f8_prmt(T.lo, 0, Wa >> 16), f8_prmt(sp[0], sp[1], Ra >> 16), 0x3210 + (Ta >> 14));
        o[ks][2] = (s1 & 0x87878787u) | f8_prmt(f8_prmt(T.lo, 0, Wb), f8_prmt(sp[2], sp[3], Rb), 0x3210 + (Tb << 2));
        o[ks][3] = (__funnelshift_l(s1, s1, 4) & 0x87878787u) | f8_prmt(f8_prmt(T.lo, 0, Wb >> 16), f8_prmt(sp[2], sp[3], Rb >> 16), 0x3210 + (Tb >> 14));
        between(ks);
    }
}

// ---- layout H, for HBM GPUs: fewer instructions a value, a little more memory ----
// An HBM GPU issues 9 to 12 instructions a byte it reads (an H100 SXM 9-10, an A100 12.5; a 4080 SUPER 35) and the layout above takes about 25 to decode
// (research/dsv4/a100-stalls-2026-10-19): there the product waits on its decode, not on memory. H keeps the patches, row blocks, lanes, the order of values and
// the sign and mantissa words; a value's exponent is a 3-bit code, its rank among the patch's exponents (0-6: the 7 most common; 7: an escape, the exponent a
// 4-bit literal of the patch's literals in var, in the order row block, lane, value). About 7.1 bits a value on FP8 checkpoints (DeepSeek-V4-Flash's dense
// matrices: 2.9% escapes; 8 as released, 6.75 above), decoded by byte permutes: a code selects a byte of the patch's table.
// A row block's core: 14 words a lane, word w of lane l at w * 32 + l: words 2 ks and 2 ks + 1 unit ks's signs and mantissas (as above), words 8-13 its 64 codes
// as 8 groups of 24 bits (group 2 ks + h: values 8 h to 8 h + 7 of unit ks, code i at bit 3 i). A patch's head: x its literals' first word of var (a plain
// patch: its part's), y the table (rank r's exponent at bit 4 r), w bit 31 plain; then x row blocks 1 and 2's first literal (16 bits each, counted in the
// patch), y row block 3's and the patch's count. A plain patch (more than HMAXE escapes, or more than 16 in a lane's 64 values) as above: its stream's bytes, row blocks
// 0-2 in core, 3 in var, and zeros in the core's last HCORE - CORE bytes (the product copies a row block's HGB bytes whole, of any patch: a pack writes every byte).
constexpr int F8_H = 16;           // KIND | F8_H: a product of a matrix in layout H (block scales, the default plan)
constexpr int HLW = 14;            // a lane's words of a row block
constexpr int HGB = 32 * HLW * 4;  // a row block's core bytes
constexpr int HCORE = 4 * HGB;     // a patch's
constexpr int HCSL = 32 * HLW;     // the words of a row block's core: a slot of the product's
constexpr int HMAXE = 1024;        // a patch with more escapes is plain
static_assert(HCORE == 7168 && 3 * 2048 <= HCORE && TAIL * 4 == 2048 && (NT - CORE / 32) * 16 == HCORE - CORE,
              "a plain patch: row blocks 0-2 in core, 3 in var, the core's rest zeros (16 bytes a thread of var's)");

// H's pass 1: a patch's head (its words of var first, for f8_scan_kernel) from the counts of its exponents (warps 2 mb and 2 mb + 1: row block mb's rows).
__device__ __forceinline__ void f8h_head_body(const F8Pk P, uint32_t* heads) {
    __shared__ __align__(16) uint32_t s_tile[VALUES / 4];
    __shared__ uint32_t s_cnt[NT / 32][16];
    __shared__ uint32_t s_sum[16];
    __shared__ uint8_t s_sym[16], s_rank[16];
    const int tid = threadIdx.x, mb = tid >> 6, l = (tid >> 1) & 31, hh = tid & 1, g = l >> 2, t = l & 3;
    if (tid < NT / 2) ((uint32_t*)s_cnt)[tid] = 0;
    __syncthreads();
    uint32_t x[8];
    f8_load(P, blockIdx.x, tid, x);
#pragma unroll
    for (int i = 0; i < 8; i++)
#pragma unroll
        for (int k = 0; k < 4; k++) atomicAdd(&s_cnt[tid >> 5][(x[i] >> (8 * k + 3)) & 15], 1u);
    {
        const int row = tid >> 2, wc = (tid & 3) * 8;
        *(uint4*)&s_tile[f8_word(row, wc)] = make_uint4(x[0], x[1], x[2], x[3]);
        *(uint4*)&s_tile[f8_word(row, wc + 4)] = make_uint4(x[4], x[5], x[6], x[7]);
    }
    __syncthreads();
    if (tid < 16) {
        uint32_t c = 0;
        for (int w = 0; w < NT / 32; w++) c += s_cnt[w][tid];
        s_sum[tid] = c;
    }
    __syncthreads();
    if (tid < 16) {  // (the layout above's ranks)
        uint32_t r = 0;
        for (int f = 0; f < 16; f++) r += (s_sum[f] > s_sum[tid]) | ((s_sum[f] == s_sum[tid]) & (f > tid));
        s_sym[r] = (uint8_t)tid;
        s_rank[tid] = (uint8_t)r;
    }
    __syncthreads();
    uint32_t ne = 0;  // the escapes of this thread's 32 values (the pack's: half its lane's row block), then of its lane's 64
#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
        for (int q = 0; q < 4; q++) {
            const uint32_t v = s_tile[f8_word(16 * mb + g + 8 * (q & 1), 8 * (2 * hh + h) + t + 4 * (q >> 1))];
#pragma unroll
            for (int k = 0; k < 4; k++) ne += s_rank[(v >> (8 * k + 3)) & 15] >= 7;
        }
    ne += __shfl_xor_sync(0xffffffffu, ne, 1);
    const int many = __syncthreads_or(ne > 16);  // (a lane of a row block with more than 16: the decode's window holds 16, so plain)
    if (tid == 0) {
        uint32_t n = 0, o[4];
        for (int mb = 0; mb < 4; mb++) {
            o[mb] = n;
            for (int e = 0; e < 16; e++) n += s_rank[e] >= 7 ? s_cnt[2 * mb][e] + s_cnt[2 * mb + 1][e] : 0u;
        }
        uint4* h = (uint4*)(heads + (int64_t)blockIdx.x * HEAD);
        if (n > (uint32_t)HMAXE || many || P.plain) {
            h[0] = make_uint4(TAIL, 0, 0, 1u << 31);
            h[1] = make_uint4(0, 0, 0, 0);
        } else {
            uint32_t tb = 0;
            for (int r = 0; r < 7; r++) tb |= (uint32_t)s_sym[r] << (4 * r);
            h[0] = make_uint4((n + 7) >> 3, tb, 0, 0);
            h[1] = make_uint4(o[1] | (o[2] << 16), o[3] | (n << 16), 0, 0);
        }
    }
}

__global__ void __launch_bounds__(NT) f8h_head_kernel(const F8Pk P, uint32_t* heads) {
    f8h_head_body(P, heads);
}

// H's pass 3: a patch's core and literals (heads from passes 1 and 2). Thread tid as the pack above: row block mb = tid / 64, lane l = (tid / 2) % 32, units
// 2 (tid % 2) and the next: the lane's values 32 (tid % 2) to 32 (tid % 2) + 31, its 3 words of codes 3 (tid % 2) on.
__device__ __forceinline__ void f8h_pack_body(const F8Pk P, const uint32_t* heads, uint8_t* core_out, uint32_t* var, int64_t var_words) {
    __shared__ __align__(16) uint32_t s_tile[VALUES / 4];
    __shared__ uint32_t s_var[HMAXE / 8 + 8];
    __shared__ uint32_t s_tot[NT / 32];
    __shared__ uint8_t s_rank[16];
    const int tid = threadIdx.x, mb = tid >> 6, l = (tid >> 1) & 31, hh = tid & 1, g = l >> 2, t = l & 3;
    const int64_t pat = blockIdx.x;
    const uint4 hd = __ldg((const uint4*)(heads + pat * HEAD)), hb = __ldg((const uint4*)(heads + pat * HEAD) + 1);
    const bool plain = (hd.w >> 31) != 0;
    const uint32_t base = hd.x, words = plain ? (uint32_t)TAIL : ((hb.y >> 16) + 7) >> 3;
    if (blockIdx.x == 0 && tid < GUARD) var[var_words - GUARD + tid] = 0;
    if ((int64_t)base + words > var_words - GUARD) return;  // (a heads array that is not this matrix's: nothing is written)
    if (tid < 16) {  // the code of an exponent: its rank where it is in the table, else 7
        uint32_t r = 7;
        for (int i = 0; i < 7; i++)
            if (((hd.y >> (4 * i)) & 15) == (uint32_t)tid) r = i;
        s_rank[tid] = (uint8_t)r;
    }
    for (int i = tid; i < HMAXE / 8 + 8; i += NT) s_var[i] = 0;
    {
        uint32_t x[8];
        f8_load(P, pat, tid, x);
        const int row = tid >> 2, wc = (tid & 3) * 8;
        *(uint4*)&s_tile[f8_word(row, wc)] = make_uint4(x[0], x[1], x[2], x[3]);
        *(uint4*)&s_tile[f8_word(row, wc + 4)] = make_uint4(x[4], x[5], x[6], x[7]);
    }
    __syncthreads();
    uint32_t y[2][4];  // unit h (ks = 2 hh + h), value j = 4 q + k: byte k of y[h][q]
#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
        for (int q = 0; q < 4; q++) y[h][q] = s_tile[f8_word(16 * mb + g + 8 * (q & 1), 8 * (2 * hh + h) + t + 4 * (q >> 1))];
    uint8_t* core = core_out + pat * HCORE;
    if (plain) {
        if (tid < CORE / 32) {
            *(uint4*)(core + tid * 32) = make_uint4(y[0][0], y[0][1], y[0][2], y[0][3]);
            *(uint4*)(core + tid * 32 + 16) = make_uint4(y[1][0], y[1][1], y[1][2], y[1][3]);
        } else {
            uint32_t* v = var + base + (tid - CORE / 32) * 8;
#pragma unroll
            for (int h = 0; h < 2; h++)
#pragma unroll
                for (int q = 0; q < 4; q++) v[4 * h + q] = y[h][q];
            *(uint4*)(core + CORE + (tid - CORE / 32) * 16) = make_uint4(0, 0, 0, 0);
        }
        return;
    }
    uint32_t sw[2][2] = {{0, 0}, {0, 0}}, cd[3] = {0, 0, 0}, ne = 0;
#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
        for (int q = 0; q < 4; q++)
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const int j = 4 * q + k, bit = 24 * (2 * h + (j >> 3)) + 3 * (j & 7);
                const uint32_t b = (y[h][q] >> (8 * k)) & 0xFF, r = s_rank[(b >> 3) & 15];
                sw[h][j >> 3] |= ((b >> 7) << f8_s_bit(j)) | ((b & 7) << f8_m_bit(j));
                cd[bit >> 5] |= r << (bit & 31);
                if ((bit & 31) > 29) cd[(bit >> 5) + 1] |= r >> (32 - (bit & 31));
                ne += r == 7;
            }
    uint32_t kl = f8_excl(ne, s_tot);  // (the patch's literals before this thread's; its __syncthreads has s_var zeroed)
#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
        for (int q = 0; q < 4; q++)
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const uint32_t e = (y[h][q] >> (8 * k + 3)) & 15;
                if (s_rank[e] == 7) {
                    atomicOr(&s_var[kl >> 3], e << (4 * (kl & 7)));
                    kl++;
                }
            }
    uint32_t* cw = (uint32_t*)(core + mb * HGB);
#pragma unroll
    for (int h = 0; h < 2; h++) {
        const int ks = 2 * hh + h;
        cw[(2 * ks) * 32 + l] = sw[h][0];
        cw[(2 * ks + 1) * 32 + l] = sw[h][1];
    }
#pragma unroll
    for (int i = 0; i < 3; i++) cw[(8 + 3 * hh + i) * 32 + l] = cd[i];
    __syncthreads();
    for (uint32_t i = tid; i < words; i += NT) var[base + i] = s_var[i];
}

__global__ void __launch_bounds__(NT) f8h_pack_kernel(const F8Pk P, const uint32_t* heads, uint8_t* core_out, uint32_t* var, int64_t var_words) {
    f8h_pack_body(P, heads, core_out, var, var_words);
}

// Row block mb's literals in its patch's (head hB): its first and the next row block's first.
__device__ __forceinline__ void f8h_lits(const uint4& hB, int mb, uint32_t& s, uint32_t& e) {
    const uint32_t a = hB.x & 0xFFFF, b = hB.x >> 16, c = hB.y & 0xFFFF, n = hB.y >> 16;
    s = mb == 0 ? 0u : mb == 1 ? a : mb == 2 ? b : c;
    e = mb == 0 ? a : mb == 1 ? b : mb == 2 ? c : n;
}

// A lane's row block in layout H into o as f8_granule leaves it (value 4 q + k of unit ks at byte k of o[ks][q]): cw its core (word w at w * 32 + lane), lp its literals
// from the row block's first word of them (kb: its first literal's place in that word; read where a lane of the warp has an escape, two words past its last too), tab the
// head's table. Each value: its code's byte of the table (an exponent e as e << 3; an escape 0) by a byte permute; then where the warp has escapes, each lane's
// literals (16 at most, from one window) put in its escapes' bytes in order.
__device__ __forceinline__ void f8h_core_words(const uint32_t* cw, int lane, uint32_t (&sm)[8], uint32_t (&cd)[7]) {  // the lane's 14 words of a core slot
#pragma unroll
    for (int w = 0; w < 8; w++) sm[w] = cw[w * 32 + lane];
#pragma unroll
    for (int w = 0; w < 6; w++) cd[w] = cw[(8 + w) * 32 + lane];
    cd[6] = 0;
}
__device__ __forceinline__ void f8h_granule_regs(const uint32_t (&sm)[8], const uint32_t (&cd)[7], const uint32_t* lp, uint32_t kb, uint32_t tab, int lane,
                                                 uint32_t (&o)[4][4]);
__device__ __forceinline__ void f8h_granule(const uint32_t* cw, const uint32_t* lp, uint32_t kb, uint32_t tab, int lane, uint32_t (&o)[4][4]) {
    uint32_t sm[8], cd[7];
    f8h_core_words(cw, lane, sm, cd);
    f8h_granule_regs(sm, cd, lp, kb, tab, lane, o);
}
__device__ __forceinline__ void f8h_granule_regs(const uint32_t (&sm)[8], const uint32_t (&cd)[7], const uint32_t* lp, uint32_t kb, uint32_t tab, int lane,
                                                 uint32_t (&o)[4][4]) {
    const uint32_t T0 = f8_nib4(tab), T1 = f8_nib4(tab >> 16) & 0x00FFFFFFu;  // ranks 0-3; 4-6 and an escape's 0 (its literal ORed in below)
    uint32_t y[8], f[8], c = 0;
#pragma unroll
    for (int gi = 0; gi < 8; gi++) {  // group gi's 8 codes a nibble each; the escapes' (code 7) bit 3 of their nibble (7 + 1 alone reaches it)
        const int b = 24 * gi;
        y[gi] = f8_spread3(__funnelshift_r(cd[b >> 5], cd[(b >> 5) + 1], b & 31));
        f[gi] = (y[gi] + 0x11111111u) & 0x88888888u;
        c += __popc(f[gi]);
    }
    uint32_t E[4][4];
#pragma unroll
    for (int ks = 0; ks < 4; ks++)
#pragma unroll
        for (int q = 0; q < 4; q++) E[ks][q] = f8_prmt(T0, T1, y[2 * ks + (q >> 1)] >> (16 * (q & 1)));
    if (__any_sync(0xffffffffu, c != 0)) {
        const uint32_t k = kb + f8_wexcl(c, lane);  // the lane's first literal, from lp's first
        // Its literals (16 at most: f8h_head_kernel's) as bytes e << 3, 4 a register, from one window. Byte p of a register: literal pre + the marks below p, pre the
        // lane's marks before the register (kept as pre 0x1111, a nibble a byte; prmt reads a selector's low 16 bits alone). Where no lane has more than 7 (nearly every
        // row block): one permute of LB0, LB1 a register ORed in, an unmarked byte's selector 8 more (prmt's sign mode on a literal: 0), so f * 0x221 + 0x8888 + pre
        // 0x1111 (f's marks 8 a nibble; 8 + 7 at most, nibble for nibble). Else also of LB2, LB3 by the selector less 8, the two OR'd and put in the marked bytes.
        const uint32_t w0 = lp[k >> 3], w1 = lp[(k >> 3) + 1], w2 = lp[(k >> 3) + 2], sh = 4 * (k & 7);
        const uint32_t L0 = __funnelshift_r(w0, w1, sh), L1 = __funnelshift_r(w1, w2, sh);
        const uint32_t LB0 = f8_nib4(L0), LB1 = f8_nib4(L0 >> 16);
        if (!__any_sync(0xffffffffu, c > 7)) {
            uint32_t pre = 0x8888u;
#pragma unroll
            for (int gi = 0; gi < 8; gi++) {
                const uint32_t lo = f[gi] & 0xFFFFu, hi = f[gi] >> 16, p1 = pre + __popc(lo) * 0x1111u;
                E[gi >> 1][2 * (gi & 1)] |= f8_prmt(LB0, LB1, lo * 0x221u + pre);
                E[gi >> 1][2 * (gi & 1) + 1] |= f8_prmt(LB0, LB1, hi * 0x221u + p1);
                pre = p1 + __popc(hi) * 0x1111u;
            }
        } else {
            const uint32_t LB2 = f8_nib4(L1), LB3 = f8_nib4(L1 >> 16);
            auto put = [](uint32_t& e, uint32_t m, uint32_t lit) {  // (lit into e's bytes marked in m, 8 a nibble: M 0xFF there, by prmt's sign mode on 0x80)
                const uint32_t M = f8_prmt(0x80u, 0, 0x4444u + (m >> 1));
                e = (e & ~M) | (lit & M);
            };
            uint32_t pre = 0;
#pragma unroll
            for (int gi = 0; gi < 8; gi++) {
                const uint32_t lo = f[gi] & 0xFFFFu, hi = f[gi] >> 16, p1 = pre + __popc(lo) * 0x1111u, s0 = lo * 0x222u + pre, s1 = hi * 0x222u + p1;
                put(E[gi >> 1][2 * (gi & 1)], lo, f8_prmt(LB0, LB1, s0) | f8_prmt(LB2, LB3, s0 ^ 0x8888u));
                put(E[gi >> 1][2 * (gi & 1) + 1], hi, f8_prmt(LB0, LB1, s1) | f8_prmt(LB2, LB3, s1 ^ 0x8888u));
                pre = p1 + __popc(hi) * 0x1111u;
            }
        }
    }
#pragma unroll
    for (int ks = 0; ks < 4; ks++) {
        const uint32_t s0 = sm[2 * ks], s1 = sm[2 * ks + 1];
        o[ks][0] = (s0 & 0x87878787u) | E[ks][0];
        o[ks][1] = (__funnelshift_l(s0, s0, 4) & 0x87878787u) | E[ks][1];
        o[ks][2] = (s1 & 0x87878787u) | E[ks][2];
        o[ks][3] = (__funnelshift_l(s1, s1, 4) & 0x87878787u) | E[ks][3];
    }
}

// The product's copies of row block mb of patch p for its decode a step or two later: its core (1,792 bytes) into cw; its literals (f8h_words: from its first's word,
// two words past its last: a window reads 3 words) into vw where they fit (else read where they are).
__device__ __forceinline__ void f8h_core_in(const uint8_t* core, int64_t p, int mb, int lane, uint32_t* cw) {
    const uint4* cs = (const uint4*)(core + p * HCORE + mb * HGB);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int c = lane + 32 * i;
        if (c < HGB / 16) f8_cp16(cw + 4 * c, cs + c);
    }
}
__device__ __forceinline__ uint32_t f8h_words(const uint4& hB, int mb, uint32_t& w0) {
    uint32_t s, e;
    f8h_lits(hB, mb, s, e);
    w0 = s >> 3;
    return ((e + 7) >> 3) - w0 + 2;
}
__device__ __forceinline__ void f8h_var_in(const uint32_t* var, int mb, int lane, const uint4& hA, const uint4& hB, uint32_t* vw) {
    if (hA.w >> 31) return;
    uint32_t w0;
    const uint32_t n = f8h_words(hB, mb, w0);
    if (n > (uint32_t)VB) return;
    const uint32_t* src = var + hA.x + w0;
    for (uint32_t i = lane; i < n; i += 32) f8_cp4(vw + i, src + i);
}
__device__ __forceinline__ void f8h_decode(const uint8_t* core, const uint32_t* var, int64_t p, int mb, int lane, const uint4& hA, const uint4& hB, const uint32_t* cw,
                                           const uint32_t* vw, uint32_t (&o)[4][4]) {
    if (hA.w >> 31) {  // plain: the lane's 64 stream bytes (row blocks 0-2 in core, 3 in var)
        const uint32_t* s = mb < 3 ? (const uint32_t*)(core + p * HCORE + 2048 * mb) + 16 * lane : var + hA.x + 16 * lane;
#pragma unroll
        for (int i = 0; i < 16; i++) o[i >> 2][i & 3] = __ldg(s + i);
        return;
    }
    uint32_t s, e, w0;
    f8h_lits(hB, mb, s, e);
    const uint32_t n = f8h_words(hB, mb, w0);
    f8h_granule(cw, n <= (uint32_t)VB ? vw : var + hA.x + w0, s & 7, hA.y, lane, o);
}
static_assert(HCSL == f8_csl<2, true>(), "the product's slots of layout H's cores");

// H's unpack: a patch (or, ids, a patch of the stacked matrices ids[blockIdx.y]) into out; warp mb row block mb, decoded as the product does from core and var.
constexpr int HUNT = 128;
__device__ __forceinline__ void f8h_unpack_body(const F8Un P) {
    __shared__ __align__(16) uint32_t s_tile[VALUES / 4];
    const int tid = threadIdx.x, mb = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int pl = blockIdx.x;
    int64_t pat = pl, item = 0;
    if (P.ids) {
        item = blockIdx.y;
        const int id = P.ids[item];
        if ((unsigned)id >= (unsigned)P.E) return;  // (no such matrix: its place in out is left as it is)
        pat += (int64_t)id * P.per;
    }
    const uint4 hA = __ldg((const uint4*)(P.heads + pat * HEAD)), hB = __ldg((const uint4*)(P.heads + pat * HEAD) + 1);
    const uint8_t* core = P.core + pat * HCORE;
    uint32_t o[4][4];
    if (hA.w >> 31) {  // plain: the lane's 64 stream bytes
        const uint32_t* s = mb < 3 ? (const uint32_t*)(core + 2048 * mb) + 16 * lane : P.var + hA.x + 16 * lane;
#pragma unroll
        for (int i = 0; i < 16; i++) o[i >> 2][i & 3] = __ldg(s + i);
    } else {
        uint32_t s, e;
        f8h_lits(hB, mb, s, e);
        f8h_granule((const uint32_t*)(core + mb * HGB), P.var + hA.x + (s >> 3), s & 7, hA.y, lane, o);
    }
#pragma unroll
    for (int ks = 0; ks < 4; ks++)
#pragma unroll
        for (int q = 0; q < 4; q++) s_tile[f8_word(16 * mb + g + 8 * (q & 1), 8 * ks + t + 4 * (q >> 1))] = o[ks][q];
    __syncthreads();
    const int row0 = (pl / P.KT) * PR, col0 = (pl % P.KT) * PC;
    uint8_t* dst = P.out + item * P.item + (int64_t)row0 * P.ld + col0;
    for (int c = tid; c < PR * 8; c += HUNT) {
        const int row = c >> 3, g8 = c & 7;
        if (row0 + row >= P.O) continue;
        const uint4 v = *(const uint4*)&s_tile[row * 32 + (((g8 ^ row) & 7) << 2)];
        uint8_t* d = dst + (int64_t)row * P.ld + g8 * 16;
        if (P.vec && col0 + g8 * 16 + 16 <= P.K) {
            F8_ALIGN(d, 16);
            *(uint4*)d = v;
        } else {
            const uint32_t b[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int k = 0; k < 16; k++)
                if (col0 + g8 * 16 + k < P.K) d[k] = (uint8_t)(b[k >> 2] >> (8 * (k & 3)));
        }
    }
}

__global__ void __launch_bounds__(HUNT) f8h_unpack_kernel(const F8Un P) {
    f8h_unpack_body(P);
}

// The experts' product's slot blockIdx.y as one matrix's product (P0's pointers at its expert and its rows; done and parts: the slot's own), or false where the plan has no such slot.
template <int SK, bool H = false>
__device__ __forceinline__ bool f8_slot(const F8Lin& P0, F8Lin& P) {
    const int s = blockIdx.y, E = P0.E, *plan = P0.plan;
    if (s >= plan[0]) return false;
    const int e = plan[1 + s], a = plan[1 + E + s], b = plan[2 + E + s];
    if (b <= a) return false;
    const int64_t per = f8_cdiv(P0.N, PR) * P0.KT;  // (an expert's patches)
    P = P0;
    P.core += e * per * (H ? HCORE : CORE);  // (layout H's cores are HCORE bytes a patch)
    P.heads += e * per * HEAD;
    if constexpr (SK == F8_BLOCK) P.scale = (const float*)P0.scale + e * f8_cdiv(P0.N, 128) * P0.lds;
    P.order = P0.gather ? plan + 2 + 2 * E + a : nullptr;
    if (!P0.gather) P.x += (int64_t)a * P0.ldx;
    P.y += (int64_t)a * P0.ldy;
    P.parts += (int64_t)s * 2 * P0.ctas * P0.M * 128;
    P.done += (int64_t)s * P0.RG;
    P.M = b - a;
    return true;
}

// AHEAD (32-64 tokens): a warp decodes the granule of the step after its products' (copies two steps ahead), and of the two warps of an SM's
// quarter (w and w + 4) one does its products first and the other its decode, so that the tensor cores and the rest of the SM work at once.
// KIND | F8_A8: the W8A8 product (block, tensor and channel scales), never AHEAD or DENSE.
// G: the experts' product (glyd_gpu_f8_moe: a slot of the plan a blockIdx.y, f8_slot's).
// KIND | F8_H: W in layout H (block scales, the default plan alone): its row blocks' cores and literals copied and decoded as H has them, the rest as above.
template <int NTL, int KIND, int AHEAD, int DENSE, int G = 0, int S = 2>
__device__ __forceinline__ void f8_linear_body(const F8Lin& P0) {
    constexpr bool H = (KIND & F8_H) != 0;
    constexpr int D = AHEAD ? 1 : 0, SK = KIND & 7, HS = f8_hs<S>(), XS = f8_xs<S>(), LSM = f8_lsmem<S, H>(), CS = f8_csl<S, H>();
    static_assert(S == 2 || (!AHEAD && !DENSE), "three slots: the one-token product's path alone");
    static_assert(!H || (SK == F8_BLOCK && !(KIND & F8_A8) && !AHEAD && !DENSE && S == 2), "layout H: block scales, the default plan");
    constexpr bool A8 = (KIND & F8_A8) != 0, DIRECT = !A8 && (SK == F8_TENSOR || SK == F8_CHANNEL);  // (DIRECT: the sums straight into acc, the scale at the end)
    F8Lin Pg;
    if constexpr (G != 0)
        if (!f8_slot<SK, H>(P0, Pg)) return;  // (the CTA's slot: no expert, or none of its tokens)
    const F8Lin& P = G != 0 ? Pg : P0;
    F8_EXTERN_SMEM(uint32_t, f8_smem);
    __shared__ int s_last;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3, mb = warp & 3, half = warp >> 2;
    const int64_t W = (int64_t)P.RG * P.KT, cta = blockIdx.x;
    const int64_t a0 = cta * W / P.ctas;           // the CTA's part of the work: patches a0 to a0 + n - 1 (patch a: row group a / KT, column a % KT)
    const int n = (int)((cta + 1) * W / P.ctas - a0);
    uint4* hs = (uint4*)f8_smem;  // HSL slots of 4: patch rows 2 rg and 2 rg + 1, hA and hB each
    uint32_t* cw = f8_smem + HS * 16 + warp * (S * CS + S * VB);  // the warp's S slots of core, then its S of variable parts (H: of literals)
    uint32_t* vw = cw + S * CS;
    uint32_t* eb = (uint32_t*)((uint8_t*)f8_smem + LSM + warp * EBS);  // (DENSE: the warp's buffer of exponents)
    uint8_t* xs = (uint8_t*)f8_smem + LSM + (DENSE ? LW * EBS : 0);  // XS slots of M token rows of XROW bytes
    const int XSLOT = f8_xslot(P.M, A8), RP = (int)f8_cdiv(P.N, PR), RB = RP * 4;
    const int nts = (P.M + 7) >> 3;  // the 8-token tiles with tokens in them
    const int kun = (int)f8_cdiv(P.K, 32);
    auto heads_in = [&](int j, int64_t rg, int kp) {  // the heads of step j (row group rg, column kp: the patches of its two rows there) into slot j % HSL
        if (tid < 4) {
            const int64_t rp = 2 * rg + (tid >> 1);
            uint4* d = hs + (j % HS) * 4 + tid;
            if (rp < RP) f8_cp16(d, (const uint4*)(P.heads + (rp * P.KT + kp) * HEAD) + (tid & 1));
            else *d = make_uint4(0, 0, 0, 0);  // (past the matrix: no warp's)
        }
    };
    auto x_in = [&](int j, int kp) {  // step j's 128 columns of X (column kp; tokens < M) into slot j % XSL: in a group of 32 columns, its 8-byte pieces i
                                      // (columns 4i..4i+3) at 2 (i & 3) + (i >> 2); an odd row's groups in pairs swapped
        uint8_t* base = xs + (j % XS) * XSLOT;
        if (A8) {  // the token rows of xq (16 bytes a copy), then their scales
            for (int idx = tid; idx < P.M * 8; idx += LNT) {
                const int mm = idx >> 3, c = idx & 7;
                f8_cp16(base + mm * PC + 32 * ((c >> 1) ^ (mm & 3)) + 16 * (c & 1), P.xq + ((int64_t)mm * P.KT + kp) * PC + 16 * c);
            }
            if (tid < P.M) f8_cp4((uint32_t*)(base + P.M * PC) + tid, (const uint32_t*)(P.sx + (int64_t)tid * P.KT + kp));
        }
        if (!A8)
        for (int idx = tid; idx < P.M * 16; idx += LNT) {
            const int mm = idx >> 4, c8 = idx & 15, k = kp * PC + 8 * c8, i0 = (c8 & 3) * 2, i1 = i0 + 1, sw = (mm & 1) << 6;
            const uint16_t* s = P.x + (int64_t)(G != 0 && P.order ? P.order[mm] / P.gk : mm) * P.ldx + k;
            uint8_t* row = base + mm * XROW;
            uint32_t* d0 = (uint32_t*)(row + (((c8 >> 2) * 64 + 8 * (2 * (i0 & 3) + (i0 >> 2))) ^ sw));
            uint32_t* d1 = (uint32_t*)(row + (((c8 >> 2) * 64 + 8 * (2 * (i1 & 3) + (i1 >> 2))) ^ sw));
            if (P.xvec && k + 8 <= P.K) {
                f8_cp8(d0, (const uint32_t*)s);
                f8_cp8(d1, (const uint32_t*)(s + 4));
            } else {
                uint32_t v[4];
#pragma unroll
                for (int i = 0; i < 4; i++) v[i] = (k + 2 * i < P.K ? (uint32_t)s[2 * i] : 0u) | (k + 2 * i + 1 < P.K ? (uint32_t)s[2 * i + 1] << 16 : 0u);
                d0[0] = v[0], d0[1] = v[1], d1[0] = v[2], d1[1] = v[3];
            }
        }
    };
    auto mine = [&](int64_t rg) { return rg * LW + warp < RB; };  // (the warp's row block of row group rg in the matrix)
    auto core_in = [&](int j, int64_t rg, int kp) {
        if (!mine(rg)) return;
        if constexpr (H) f8h_core_in(P.core, (2 * rg + half) * P.KT + kp, mb, lane, cw + (j % S) * CS);
        else f8_core_in(P, (2 * rg + half) * P.KT + kp, mb, lane, cw + (j % S) * CS);
    };
    auto var_in = [&](int j, int64_t rg) {  // (step j's heads in)
        if (mine(rg)) {
            const uint4* h = hs + (j % HS) * 4 + 2 * half;
            if constexpr (H) f8h_var_in(P.var, mb, lane, h[0], h[1], vw + (j % S) * VB);
            else f8_var_in(P, mb, lane, h[0], h[1], vw + (j % S) * VB);
        }
    };
    auto decode = [&](int j, int64_t rg, int kp, uint32_t (&o)[4][4]) {
        const uint4* h = hs + (j % HS) * 4 + 2 * half;
        if constexpr (H) f8h_decode(P.core, P.var, (2 * rg + half) * P.KT + kp, mb, lane, h[0], h[1], cw + (j % S) * CS, vw + (j % S) * VB, o);
        else if (DENSE) f8_granule_dense(P, (2 * rg + half) * P.KT + kp, mb, lane, h[0], h[1], cw + (j % S) * CS, vw + (j % S) * VB, eb, o, [](int) {});
        else f8_granule(P, (2 * rg + half) * P.KT + kp, mb, lane, h[0], h[1], cw + (j % S) * CS, vw + (j % S) * VB, o);
    };
    float acc[NTL][4];
#pragma unroll
    for (int nt = 0; nt < NTL; nt++) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
    const int g1 = g & 1;
    float tmp[NTL][4];  // (a granule's sums (block scales) or a unit's (mx, ds32), before its scale)
    auto prod_begin = [&]() {
        if (SK == F8_BLOCK || A8)
#pragma unroll
            for (int nt = 0; nt < NTL; nt++) tmp[nt][0] = tmp[nt][1] = tmp[nt][2] = tmp[nt][3] = 0.f;
    };
    auto prod_unit = [&](int j, int64_t rg, int kp, int ks, const uint32_t (&ok)[4]) {  // unit ks's products with step j's X
        if constexpr (A8) {  // the decoded bytes as they are, X's e4m3 8 bytes a lane: one product a unit and 8 tokens
            const uint8_t* xsl = xs + (j % XS) * XSLOT + 8 * t;
            uint2 xb[NTL];
#pragma unroll
            for (int nt = 0; nt < NTL; nt++)
                if (nt < nts) {
                    const int m = min(8 * nt + g, P.M - 1);
                    xb[nt] = *(const uint2*)(xsl + m * PC + 32 * (ks ^ (m & 3)));
                }
#pragma unroll
            for (int nt = 0; nt < NTL; nt++)
                if (nt < nts) f8_mma8(tmp[nt], ok[0], ok[1], ok[2], ok[3], xb[nt].x, xb[nt].y);
            return;
        }
        const uint8_t* xsl = xs + (j % XS) * XSLOT + 16 * t;
        const uint32_t a0_ = f8_bf_lo(ok[0]), a1_ = f8_bf_lo(ok[1]), a2_ = f8_bf_lo(ok[2]), a3_ = f8_bf_lo(ok[3]);
        const uint32_t b0_ = f8_bf_hi(ok[0]), b1_ = f8_bf_hi(ok[1]), b2_ = f8_bf_hi(ok[2]), b3_ = f8_bf_hi(ok[3]);
        if (SK == F8_MX || SK == F8_DS32)
#pragma unroll
            for (int nt = 0; nt < NTL; nt++) tmp[nt][0] = tmp[nt][1] = tmp[nt][2] = tmp[nt][3] = 0.f;
        uint4 xb[NTL];  // (a token past M: any token's row (a row of the slot: its parity need not be g's), its sums not kept)
#pragma unroll
        for (int nt = 0; nt < NTL; nt++)
            if (nt < nts) {
                const uint4 v = *(const uint4*)(xsl + min(8 * nt + g, P.M - 1) * XROW + 64 * (ks ^ g1));
                xb[nt] = make_uint4(f8_x64(v.x), f8_x64(v.y), f8_x64(v.z), f8_x64(v.w));
            }
#pragma unroll
        for (int nt = 0; nt < NTL; nt++)  // (every tile's first product, then every tile's second: a tile's two are NTL products apart)
            if (nt < nts) f8_mma(DIRECT ? acc[nt] : tmp[nt], a0_, a1_, a2_, a3_, xb[nt].x, xb[nt].z);
#pragma unroll
        for (int nt = 0; nt < NTL; nt++)
            if (nt < nts) f8_mma(DIRECT ? acc[nt] : tmp[nt], b0_, b1_, b2_, b3_, xb[nt].y, xb[nt].w);
        if (SK == F8_MX || SK == F8_DS32) {
            const int rb = (int)(rg * LW) + warp;
            const int r0 = min(16 * rb + g, P.N - 1), r1 = min(16 * rb + g + 8, P.N - 1);  // (scale rows: past N, any row's)
            const int ku = min(4 * kp + ks, kun - 1);
            const uint8_t* sc = (const uint8_t*)P.scale;
            const float s0 = f8_e8m0(SK == F8_MX ? sc[(int64_t)r0 * P.lds + ku] : sc[(int64_t)(r0 >> 5) * P.lds + ku]);
            const float s1 = SK == F8_MX ? f8_e8m0(sc[(int64_t)r1 * P.lds + ku]) : s0;
#pragma unroll
            for (int nt = 0; nt < NTL; nt++) {
                acc[nt][0] += tmp[nt][0] * s0, acc[nt][1] += tmp[nt][1] * s0;
                acc[nt][2] += tmp[nt][2] * s1, acc[nt][3] += tmp[nt][3] * s1;
            }
        }
    };
    auto prod_end = [&](int j, int64_t rg, int kp) {
        if constexpr (A8) {  // the granule's sums times each token's scale of step j (and the block's)
            const float* sxs = (const float*)(xs + (j % XS) * XSLOT + P.M * PC);
            const float w = SK == F8_BLOCK ? ((const float*)P.scale)[rg * P.lds + kp] : 1.f;
#pragma unroll
            for (int nt = 0; nt < NTL; nt++) {
                const float x0 = sxs[min(8 * nt + 2 * t, P.M - 1)] * w, x1 = sxs[min(8 * nt + 2 * t + 1, P.M - 1)] * w;
                acc[nt][0] += tmp[nt][0] * x0, acc[nt][1] += tmp[nt][1] * x1, acc[nt][2] += tmp[nt][2] * x0, acc[nt][3] += tmp[nt][3] * x1;
            }
        } else if (SK == F8_BLOCK) {
            const float s = ((const float*)P.scale)[rg * P.lds + kp];
#pragma unroll
            for (int nt = 0; nt < NTL; nt++)
#pragma unroll
                for (int i = 0; i < 4; i++) acc[nt][i] += tmp[nt][i] * s;
        }
    };
    auto product = [&](int j, int64_t rg, int kp, const uint32_t (&o)[4][4]) {  // the granule's products with step j's X into acc, its scales applied
        prod_begin();
#pragma unroll
        for (int ks = 0; ks < 4; ks++) prod_unit(j, rg, kp, ks, o[ks]);
        prod_end(j, rg, kp);
    };
    auto step = [&](int64_t& rg, int& kp) {  // the next patch of the order
        if (++kp == P.KT) kp = 0, rg++;
    };
    const int64_t rg0 = a0 / P.KT;
    const int kp0 = (int)(a0 % P.KT);
    // the steps' places: A = j (products), B = j + 1, X = j + S - 1 (X's copies), C = j + S - 1 + D (core's and variable parts' copies), H = j + 2 S - 2 + D (heads)
    int64_t rgA = rg0, rgB = rg0, rgC, rgH, rgX;
    int kpA = kp0, kpB = kp0, kpC, kpH, kpX;
    step(rgB, kpB);
    rgX = rgB, kpX = kpB;
    for (int i = 2; i < S; i++) step(rgX, kpX);
    rgC = rgX, kpC = kpX;
    if (D) step(rgC, kpC);
    rgH = rgC, kpH = kpC;
    for (int i = 1; i < S; i++) step(rgH, kpH);
    // the first steps' core and heads and X, waited for; then their variable parts (their heads in)
    {
        int64_t r = rg0;
        int k = kp0;
        for (int i = 0; i <= 2 * S - 3 + D && i < n; i++) {
            heads_in(i, r, k);
            if (i <= S - 2 + D) core_in(i, r, k);
            if (i <= S - 2) x_in(i, k);
            step(r, k);
        }
    }
    f8_cp_commit();
    f8_cp_wait0();
    __syncthreads();
    if constexpr (S == 2) {
        var_in(0, rg0);
        if (D && n > 1) var_in(1, rgB);
        f8_cp_commit();
    } else {  // (all the first steps' in before the loop, whose waits leave S - 2 steps' copies in flight)
        int64_t r = rg0;
        int k = kp0;
        for (int i = 0; i <= S - 2 && i < n; i++) {
            var_in(i, r);
            step(r, k);
        }
        f8_cp_commit();
        f8_cp_wait0();
        __syncthreads();
    }
    uint32_t oc[4][4];  // (AHEAD: the granule decoded a step ahead)
    if (D) {
        f8_cp_wait0();
        __syncwarp();  // (a warp's copies are its own)
        if (mine(rg0)) decode(0, rg0, kp0, oc);
    }
    for (int j = 0; j < n; j++) {
        f8_cp_wait<S - 2>();
        __syncthreads();  // step j's copies in for every thread (and step j + S - 1's heads), every warp past step j - 1 (its slots free)
        if (j + S - 1 + D < n) {
            core_in(j + S - 1 + D, rgC, kpC);
            var_in(j + S - 1 + D, rgC);
        }
        if (j + S - 1 < n) x_in(j + S - 1, kpX);
        if (j + 2 * S - 2 + D < n) heads_in(j + 2 * S - 2 + D, rgH, kpH);
        f8_cp_commit();
        if (D) {
            uint32_t on[4][4];
            const bool nx = j + 1 < n && mine(rgB), cur = mine(rgA);
            if (DENSE && nx && cur) {  // the products of unit ks among the decode of the next granule's units
                const uint4* h = hs + ((j + 1) % HS) * 4 + 2 * half;
                prod_begin();
                f8_granule_dense(P, (2 * rgB + half) * P.KT + kpB, mb, lane, h[0], h[1], cw + ((j + 1) % S) * CS, vw + ((j + 1) % S) * VB, eb, on,
                                 [&](int ks) { prod_unit(j, rgA, kpA, ks, oc[ks]); });
                prod_end(j, rgA, kpA);
            } else if (half == 0) {
                if (cur) product(j, rgA, kpA, oc);
                if (nx) decode(j + 1, rgB, kpB, on);
            } else {
                if (nx) decode(j + 1, rgB, kpB, on);
                if (cur) product(j, rgA, kpA, oc);
            }
#pragma unroll
            for (int ks = 0; ks < 4; ks++)
#pragma unroll
                for (int q = 0; q < 4; q++) oc[ks][q] = on[ks][q];
        } else if (mine(rgA)) {
            decode(j, rgA, kpA, oc);
            product(j, rgA, kpA, oc);
        }
        if (kpA == P.KT - 1 || j == n - 1) {  // the end of the CTA's part of row group rgA: its sums out
            const int64_t rg = rgA;
            const int rb = (int)(rg * LW) + warp;
            const float f = (SK == F8_TENSOR ? ((const float*)P.scale)[0] : 1.f) * (A8 ? 1.f : 0x1p56f);  // (W8A16: the bf16 weights' 2^-120 and X's 2^64)
            const int64_t i0 = rg * P.KT;
            const int cf = (int)(((i0 + 1) * P.ctas - 1) / W), cl = (int)(((i0 + P.KT) * P.ctas - 1) / W);  // the CTAs that hold its patches
            const int row0 = 16 * rb + g;
            float rs[2] = {f, f};  // (channel: each of the lane's two rows' scale too)
            if (SK == F8_CHANNEL) rs[0] *= ((const float*)P.scale)[min(row0, P.N - 1)], rs[1] *= ((const float*)P.scale)[min(row0 + 8, P.N - 1)];
            if (cf == cl) {  // all of its K here: Y
                if (rb < RB)
#pragma unroll
                    for (int nt = 0; nt < NTL; nt++)
#pragma unroll
                        for (int i = 0; i < 4; i++) {
                            const int mm = 8 * nt + 2 * t + (i & 1), nn = row0 + 8 * (i >> 1);
                            if (mm < P.M && nn < P.N) P.y[(int64_t)mm * P.ldy + nn] = f8_to_bf16(acc[nt][i] * rs[i >> 1]);
                        }
            } else {  // part of it: the sums into this CTA's first (0) or last (1) slot of parts; the last CTA of the row group adds them
                float* pt = P.parts + (int64_t)(2 * (int)cta + (rg == rg0 ? 0 : 1)) * P.M * 128;
                if (rb < RB)
#pragma unroll
                    for (int nt = 0; nt < NTL; nt++)
#pragma unroll
                        for (int i = 0; i < 4; i++) {
                            const int mm = 8 * nt + 2 * t + (i & 1), r = 16 * warp + g + 8 * (i >> 1);
                            if (mm < P.M) pt[mm * 128 + r] = acc[nt][i] * rs[i >> 1];
                        }
                __threadfence();
                __syncthreads();
                if (tid == 0) {
                    s_last = atomicAdd(P.done + rg, 1) == cl - cf;
                    if (s_last) P.done[rg] = 0;  // ready for the next product
                }
                __syncthreads();
                if (s_last) {
                    __threadfence();
                    const int n0 = (int)rg * 128, wf = (int64_t)cf * W / P.ctas >= i0 ? 0 : 1;  // (CTA cf's slot for it: 0 where its part starts in it; every later CTA's: 0)
                    for (int idx = tid; idx < P.M * 128; idx += LNT) {
                        const int mm = idx >> 7, r = idx & 127;
                        if (n0 + r >= P.N) continue;
                        float s = 0.f;
                        for (int c = cf; c <= cl; c++) s += __ldcg(P.parts + ((int64_t)(2 * c + (c == cf ? wf : 0)) * P.M + mm) * 128 + r);
                        P.y[(int64_t)mm * P.ldy + n0 + r] = f8_to_bf16(s);
                    }
                }
            }
#pragma unroll
            for (int nt = 0; nt < NTL; nt++) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
        }
        step(rgA, kpA);
        step(rgB, kpB);
        step(rgC, kpC);
        step(rgH, kpH);
        step(rgX, kpX);
    }
}

template <int NTL, int KIND, int AHEAD, int DENSE, int G = 0, int S = 2>
__global__ void __launch_bounds__(LNT, 1) f8_linear_kernel(const F8Lin P0) {
    f8_linear_body<NTL, KIND, AHEAD, DENSE, G, S>(P0);
}

// Layout H, one token (block scales; G: the experts' product, a slot's one pair), on f8_linear_kernel's plan: the CTA's patches, the sums of its first and last row
// groups in parts and the last CTA's adds, so its results to the bit. Each warp copies its own patch's head (two steps ahead, to registers) and its core, literals
// and X (a step ahead, to its own slots), so no barrier a step; the steps' places in 32 bits.
constexpr int XW1 = PC * 2 / 4;                    // words of a warp's X slot: one token's 128 columns
constexpr int L1W = 2 * HCSL + 2 * VB + 2 * XW1;   // words of a warp's slots
// C: the CTAs an SM the compiler fits (3: 80 registers a thread; 2: as many as it likes). Being measured on HBM GPUs (GLYD_F8H_CTAS).
template <int G, int C>
__device__ __forceinline__ void f8h_lin1_body(const F8Lin& P) {
    // (the slot's pointers alone as registers; the rest read from the parameters, which f8_slot leaves as they are)
    const uint8_t* core = P.core;
    const uint32_t* heads = P.heads;
    const float* scale = (const float*)P.scale;
    const uint16_t* x = P.x;
    uint16_t* y = P.y;
    float* parts = P.parts;
    int* done = P.done;
    if constexpr (G != 0) {
        F8Lin Pg;
        if (!f8_slot<F8_BLOCK, true>(P, Pg)) return;
        core = Pg.core, heads = Pg.heads, scale = (const float*)Pg.scale, y = Pg.y, parts = Pg.parts, done = Pg.done;
        x = Pg.order ? P.x + (int64_t)(Pg.order[0] / P.gk) * P.ldx : Pg.x;  // (the pair's token's row)
    }
    const uint32_t* var = P.var;
    F8_EXTERN_SMEM(uint32_t, f8_smem);
    __shared__ int s_last;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3, mb = warp & 3, half = warp >> 2, g1 = g & 1;
    uint32_t* cw = f8_smem + warp * L1W;  // two slots of core, then two of literals, then two of X
    uint32_t* vw = cw + 2 * HCSL;
    uint32_t* xw = vw + 2 * VB;
    const int64_t W = (int64_t)P.RG * P.KT, cta = blockIdx.x;
    const int64_t a0 = cta * W / P.ctas;
    const int n = (int)((cta + 1) * W / P.ctas - a0), KT = P.KT, RP = (int)f8_cdiv(P.N, PR), RB = RP * 4;
    const int rg0 = (int)(a0 / KT), kp0 = (int)(a0 % KT);
    auto step = [&](int& r, int& k) {
        if (++k == KT) k = 0, r++;
    };
    auto patch = [&](int r, int k) { return (2 * r + half) * KT + k; };  // the warp's patch at row group r, column k
    auto mine = [&](int r) { return 2 * r + half < RP; };
    auto head = [&](int r, int k, uint4 (&h)[2]) {
        if (mine(r)) {
            const uint4* s = (const uint4*)(heads + (int64_t)patch(r, k) * HEAD);
            h[0] = __ldg(s), h[1] = __ldg(s + 1);
        } else {
            h[0] = h[1] = make_uint4(0, 0, 0, 0);
        }
    };
    auto x_in = [&](int sl, int k) {  // column k's X into slot sl as f8_linear_kernel's x_in has a token's row (lanes 0-15: 8 columns each)
        if (lane >= 16) return;
        const int kk = k * PC + 8 * lane, i0 = (lane & 3) * 2, i1 = i0 + 1;
        uint8_t* row = (uint8_t*)(xw + sl * XW1);
        uint32_t* d0 = (uint32_t*)(row + (lane >> 2) * 64 + 8 * (2 * (i0 & 3) + (i0 >> 2)));
        uint32_t* d1 = (uint32_t*)(row + (lane >> 2) * 64 + 8 * (2 * (i1 & 3) + (i1 >> 2)));
        const uint16_t* s = x + kk;
        if (P.xvec && kk + 8 <= P.K) {
            f8_cp8(d0, (const uint32_t*)s);
            f8_cp8(d1, (const uint32_t*)(s + 4));
        } else {
            uint32_t v[4];
#pragma unroll
            for (int i = 0; i < 4; i++) v[i] = (kk + 2 * i < P.K ? (uint32_t)s[2 * i] : 0u) | (kk + 2 * i + 1 < P.K ? (uint32_t)s[2 * i + 1] << 16 : 0u);
            d0[0] = v[0], d0[1] = v[1], d1[0] = v[2], d1[1] = v[3];
        }
    };
    auto copies = [&](int sl, int r, int k, const uint4 (&h)[2]) {
        if (mine(r)) {
            f8h_core_in(core, patch(r, k), mb, lane, cw + sl * HCSL);
            f8h_var_in(var, mb, lane, h[0], h[1], vw + sl * VB);
        }
        x_in(sl, k);
    };
    // the cursors: A the step decoded (j), C the step copied (j + 1), H the step whose head comes in (j + 2)
    int rgA = rg0, kpA = kp0, rgC = rg0, kpC = kp0, rgH = rg0, kpH = kp0;
    uint4 hc[2], hn[2] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
    head(rgA, kpA, hc);
    step(rgH, kpH);
    if (n > 1) head(rgH, kpH, hn);
    step(rgH, kpH);
    copies(0, rgC, kpC, hc);
    f8_cp_commit();
    step(rgC, kpC);
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int j = 0; j < n; j++) {
        const int sl = j & 1;
        f8_cp_wait0();
        __syncwarp();  // step j's copies in for the warp, and the warp past step j - 1 (slot sl ^ 1 free)
        uint4 h2[2] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
        if (j + 2 < n) head(rgH, kpH, h2);
        if (mine(rgA)) {
            uint32_t o[4][4];
            f8h_decode(core, var, patch(rgA, kpA), mb, lane, hc[0], hc[1], cw + sl * HCSL, vw + sl * VB, o);
            const uint8_t* xsl = (const uint8_t*)(xw + sl * XW1) + 16 * t;
            float tmp[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int ks = 0; ks < 4; ks++) {
                const uint32_t a0_ = f8_bf_lo(o[ks][0]), a1_ = f8_bf_lo(o[ks][1]), a2_ = f8_bf_lo(o[ks][2]), a3_ = f8_bf_lo(o[ks][3]);
                const uint32_t b0_ = f8_bf_hi(o[ks][0]), b1_ = f8_bf_hi(o[ks][1]), b2_ = f8_bf_hi(o[ks][2]), b3_ = f8_bf_hi(o[ks][3]);
                const uint4 v = *(const uint4*)(xsl + 64 * (ks ^ g1));
                const uint4 xb = make_uint4(f8_x64(v.x), f8_x64(v.y), f8_x64(v.z), f8_x64(v.w));
                f8_mma(tmp, a0_, a1_, a2_, a3_, xb.x, xb.z);
                f8_mma(tmp, b0_, b1_, b2_, b3_, xb.y, xb.w);
            }
            const float s = scale[(int64_t)rgA * P.lds + kpA];
#pragma unroll
            for (int i = 0; i < 4; i++) acc[i] += tmp[i] * s;
        }
        if (j + 1 < n) copies(sl ^ 1, rgC, kpC, hn);  // (after the products: see f8h_lin2_kernel)
        f8_cp_commit();
        if (kpA == KT - 1 || j == n - 1) {  // the end of the CTA's part of row group rgA: its sums out, as f8_linear_kernel's
            const int rb = rgA * LW + warp, row0 = 16 * rb + g;
            const float f = 0x1p56f;  // (the bf16 weights' 2^-120 and X's 2^64)
            const int64_t i0 = (int64_t)rgA * KT;
            const int cf = (int)(((i0 + 1) * P.ctas - 1) / W), cl = (int)(((i0 + KT) * P.ctas - 1) / W);
            if (cf == cl) {
                if (rb < RB && t == 0) {
                    if (row0 < P.N) y[row0] = f8_to_bf16(acc[0] * f);
                    if (row0 + 8 < P.N) y[row0 + 8] = f8_to_bf16(acc[2] * f);
                }
            } else {
                float* pt = parts + (int64_t)(2 * (int)cta + (rgA == rg0 ? 0 : 1)) * 128;
                if (rb < RB && t == 0) pt[16 * warp + g] = acc[0] * f, pt[16 * warp + g + 8] = acc[2] * f;
                __threadfence();
                __syncthreads();
                if (tid == 0) {
                    s_last = atomicAdd(done + rgA, 1) == cl - cf;
                    if (s_last) done[rgA] = 0;  // ready for the next product
                }
                __syncthreads();
                if (s_last) {
                    __threadfence();
                    const int n0 = rgA * 128, wf = (int64_t)cf * W / P.ctas >= i0 ? 0 : 1;
                    for (int r = tid; r < 128; r += LNT) {
                        if (n0 + r >= P.N) continue;
                        float sum = 0.f;
                        for (int c = cf; c <= cl; c++) sum += __ldcg(parts + (int64_t)(2 * c + (c == cf ? wf : 0)) * 128 + r);
                        y[n0 + r] = f8_to_bf16(sum);
                    }
                }
            }
            acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
        }
        hc[0] = hn[0], hc[1] = hn[1], hn[0] = h2[0], hn[1] = h2[1];
        step(rgA, kpA);
        step(rgC, kpC);
        step(rgH, kpH);
    }
}

template <int G, int C>
__global__ void __launch_bounds__(LNT, C) f8h_lin1_kernel(const F8Lin P) {
    f8h_lin1_body<G, C>(P);
}

// Layout H, one token, the copies two steps ahead in the same two core slots: a step's core words go to registers first (14 a lane), then its slot takes the step
// after next's core; a step's literals, X and head have one of three small slots. f8h_lin1_kernel's plan and sums (the same digits); its heads come through shared
// memory, so two heads a warp in registers instead of three.
constexpr int SW2 = VB + XW1 + HEAD;      // words of a small slot: a step's literals, X and head
constexpr int CSL2 = 512;                 // words of a core slot: a coded row block's core (HCSL) or a plain one's 2,048 bytes
constexpr int L2W = 2 * CSL2 + 3 * SW2;   // words of a warp's slots
static_assert(CSL2 >= HCSL && CSL2 * 4 == 2048, "a core slot of layout H's one-token kernel");
template <int G, int C>
__global__ void __launch_bounds__(LNT, C) f8h_lin2_kernel(const F8Lin P) {
    const uint8_t* core = P.core;
    const uint32_t* heads = P.heads;
    const float* scale = (const float*)P.scale;
    const uint16_t* x = P.x;
    uint16_t* y = P.y;
    float* parts = P.parts;
    int* done = P.done;
    if constexpr (G != 0) {
        F8Lin Pg;
        if (!f8_slot<F8_BLOCK, true>(P, Pg)) return;
        core = Pg.core, heads = Pg.heads, scale = (const float*)Pg.scale, y = Pg.y, parts = Pg.parts, done = Pg.done;
        x = Pg.order ? P.x + (int64_t)(Pg.order[0] / P.gk) * P.ldx : Pg.x;  // (the pair's token's row)
    }
    const uint32_t* var = P.var;
    F8_EXTERN_SMEM(uint32_t, f8_smem);
    __shared__ int s_last;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3, mb = warp & 3, half = warp >> 2, g1 = g & 1;
    uint32_t* cw = f8_smem + warp * L2W;  // two core slots (step j's at j & 1), then three small slots (step j's at j % 3)
    uint32_t* sw = cw + 2 * CSL2;
    const int64_t W = (int64_t)P.RG * P.KT, cta = blockIdx.x;
    const int64_t a0 = cta * W / P.ctas;
    const int n = (int)((cta + 1) * W / P.ctas - a0), KT = P.KT, RP = (int)f8_cdiv(P.N, PR), RB = RP * 4;
    const int rg0 = (int)(a0 / KT), kp0 = (int)(a0 % KT);
    auto step = [&](int& r, int& k) {
        if (++k == KT) k = 0, r++;
    };
    auto patch = [&](int r, int k) { return (2 * r + half) * KT + k; };  // the warp's patch at row group r, column k
    auto mine = [&](int r) { return 2 * r + half < RP; };
    auto head = [&](int r, int k, uint4 (&h)[2]) {
        if (mine(r)) {
            const uint4* s = (const uint4*)(heads + (int64_t)patch(r, k) * HEAD);
            h[0] = __ldg(s), h[1] = __ldg(s + 1);
        } else {
            h[0] = h[1] = make_uint4(0, 0, 0, 0);
        }
    };
    auto x_in = [&](uint32_t* xw, int k) {  // column k's X into a small slot's X words as f8_linear_kernel's x_in has a token's row (lanes 0-15: 8 columns each)
        if (lane >= 16) return;
        const int kk = k * PC + 8 * lane, i0 = (lane & 3) * 2, i1 = i0 + 1;
        uint8_t* row = (uint8_t*)xw;
        uint32_t* d0 = (uint32_t*)(row + (lane >> 2) * 64 + 8 * (2 * (i0 & 3) + (i0 >> 2)));
        uint32_t* d1 = (uint32_t*)(row + (lane >> 2) * 64 + 8 * (2 * (i1 & 3) + (i1 >> 2)));
        const uint16_t* s = x + kk;
        if (P.xvec && kk + 8 <= P.K) {
            f8_cp8(d0, (const uint32_t*)s);
            f8_cp8(d1, (const uint32_t*)(s + 4));
        } else {
            uint32_t v[4];
#pragma unroll
            for (int i = 0; i < 4; i++) v[i] = (kk + 2 * i < P.K ? (uint32_t)s[2 * i] : 0u) | (kk + 2 * i + 1 < P.K ? (uint32_t)s[2 * i + 1] << 16 : 0u);
            d0[0] = v[0], d0[1] = v[1], d1[0] = v[2], d1[1] = v[3];
        }
    };
    auto copies = [&](int j2, int r, int k, const uint4 (&h)[2]) {  // step j2's: its core to core slot j2 & 1; its literals, X and head to small slot j2 % 3
        uint32_t* s = sw + (j2 % 3) * SW2;
        if (mine(r)) {
            uint32_t* cd = cw + (j2 & 1) * CSL2;
            if (h[0].w >> 31) {  // plain: the lane's 64 stream bytes (row blocks 0-2 in core, 3 in var), 16 bytes q of the lane at (32 q + lane) * 16 of the slot
                const uint32_t* src = (mb < 3 ? (const uint32_t*)(core + patch(r, k) * HCORE + 2048 * mb) : var + h[0].x) + 16 * lane;
                if (mb < 3 || (h[0].x & 3) == 0) {
#pragma unroll
                    for (int q = 0; q < 4; q++) f8_cp16(cd + (32 * q + lane) * 4, src + 4 * q);
                } else {
#pragma unroll
                    for (int i = 0; i < 16; i++) f8_cp4(cd + (32 * (i >> 2) + lane) * 4 + (i & 3), src + i);
                }
            } else {
                f8h_core_in(core, patch(r, k), mb, lane, cd);
                f8h_var_in(var, mb, lane, h[0], h[1], s);
            }
        }
        x_in(s + VB, k);
        if (lane < 2) ((uint4*)(s + VB + XW1))[lane] = lane ? h[1] : h[0];
    };
    // the cursors: A the step decoded (j), C the step copied (j + 2), H the step whose head comes in (j + 3); hcp the head of the step copied next
    int rgA = rg0, kpA = kp0, rgC = rg0, kpC = kp0, rgH = rg0, kpH = kp0;
    uint4 h0[2], hcp[2] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
    head(rgA, kpA, h0);
    copies(0, rgC, kpC, h0);
    f8_cp_commit();
    step(rgC, kpC);
    step(rgH, kpH);
    if (n > 1) {
        head(rgH, kpH, h0);
        copies(1, rgC, kpC, h0);
    }
    f8_cp_commit();  // (a group a step, empty or not: wait_group 1 below counts groups)
    step(rgC, kpC);
    step(rgH, kpH);
    if (n > 2) head(rgH, kpH, hcp);
    step(rgH, kpH);
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int j = 0; j < n; j++) {
        const uint32_t* cs = cw + (j & 1) * CSL2;
        const uint32_t* ss = sw + (j % 3) * SW2;
        f8_cp_wait1();  // step j's copies in (step j + 1's may still be coming)
        __syncwarp();   // for the whole warp
        const uint4 hA = ((const uint4*)(ss + VB + XW1))[0], hB = ((const uint4*)(ss + VB + XW1))[1];
        const bool mn = mine(rgA), plain = (hA.w >> 31) != 0;
        uint32_t sm[8], cd[7];
        uint4 pv[4];
        if (mn) {
            if (plain) {
#pragma unroll
                for (int q = 0; q < 4; q++) pv[q] = ((const uint4*)cs)[32 * q + lane];
            } else {
                f8h_core_words(cs, lane, sm, cd);
            }
        }
        __syncwarp();  // every lane past its reads of the core slot: step j + 2's core may come in
        uint4 hld[2] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
        if (j + 3 < n) head(rgH, kpH, hld);
        const float s = mn ? scale[(int64_t)rgA * P.lds + kpA] : 0.f;
        if (mn) {
            uint32_t o[4][4];
            if (plain) {  // the lane's 64 stream bytes, from the slot
#pragma unroll
                for (int q = 0; q < 4; q++) o[q][0] = pv[q].x, o[q][1] = pv[q].y, o[q][2] = pv[q].z, o[q][3] = pv[q].w;
            } else {
                uint32_t ls, le, w0;
                f8h_lits(hB, mb, ls, le);
                const uint32_t nw = f8h_words(hB, mb, w0);
                f8h_granule_regs(sm, cd, nw <= (uint32_t)VB ? ss : var + hA.x + w0, ls & 7, hA.y, lane, o);
            }
            const uint8_t* xsl = (const uint8_t*)(ss + VB) + 16 * t;
            float tmp[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int ks = 0; ks < 4; ks++) {
                const uint32_t a0_ = f8_bf_lo(o[ks][0]), a1_ = f8_bf_lo(o[ks][1]), a2_ = f8_bf_lo(o[ks][2]), a3_ = f8_bf_lo(o[ks][3]);
                const uint32_t b0_ = f8_bf_hi(o[ks][0]), b1_ = f8_bf_hi(o[ks][1]), b2_ = f8_bf_hi(o[ks][2]), b3_ = f8_bf_hi(o[ks][3]);
                const uint4 v = *(const uint4*)(xsl + 64 * (ks ^ g1));
                const uint4 xb = make_uint4(f8_x64(v.x), f8_x64(v.y), f8_x64(v.z), f8_x64(v.w));
                f8_mma(tmp, a0_, a1_, a2_, a3_, xb.x, xb.z);
                f8_mma(tmp, b0_, b1_, b2_, b3_, xb.y, xb.w);
            }
#pragma unroll
            for (int i = 0; i < 4; i++) acc[i] += tmp[i] * s;
        }
        // step j + 2's copies after the step's products: a full copy queue then holds up nothing but the next wait (session 8: the order and the
        // mechanism alone read at the card's streaming rate; with the copies before the products the warps queued on them instead of computing)
        if (j + 2 < n) copies(j + 2, rgC, kpC, hcp);
        f8_cp_commit();
        if (kpA == KT - 1 || j == n - 1) {  // the end of the CTA's part of row group rgA: its sums out, as f8_linear_kernel's
            const int rb = rgA * LW + warp, row0 = 16 * rb + g;
            const float f = 0x1p56f;  // (the bf16 weights' 2^-120 and X's 2^64)
            const int64_t i0 = (int64_t)rgA * KT;
            const int cf = (int)(((i0 + 1) * P.ctas - 1) / W), cl = (int)(((i0 + KT) * P.ctas - 1) / W);
            if (cf == cl) {
                if (rb < RB && t == 0) {
                    if (row0 < P.N) y[row0] = f8_to_bf16(acc[0] * f);
                    if (row0 + 8 < P.N) y[row0 + 8] = f8_to_bf16(acc[2] * f);
                }
            } else {
                float* pt = parts + (int64_t)(2 * (int)cta + (rgA == rg0 ? 0 : 1)) * 128;
                if (rb < RB && t == 0) pt[16 * warp + g] = acc[0] * f, pt[16 * warp + g + 8] = acc[2] * f;
                __threadfence();
                __syncthreads();
                if (tid == 0) {
                    s_last = atomicAdd(done + rgA, 1) == cl - cf;
                    if (s_last) done[rgA] = 0;  // ready for the next product
                }
                __syncthreads();
                if (s_last) {
                    __threadfence();
                    const int n0 = rgA * 128, wf = (int64_t)cf * W / P.ctas >= i0 ? 0 : 1;
                    for (int r = tid; r < 128; r += LNT) {
                        if (n0 + r >= P.N) continue;
                        float sum = 0.f;
                        for (int c = cf; c <= cl; c++) sum += __ldcg(parts + (int64_t)(2 * c + (c == cf ? wf : 0)) * 128 + r);
                        y[n0 + r] = f8_to_bf16(sum);
                    }
                }
            }
            acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
        }
        hcp[0] = hld[0], hcp[1] = hld[1];
        step(rgA, kpA);
        step(rgC, kpC);
        step(rgH, kpH);
    }
}

// Plain patches alone, lean (research, 2026-10-20): the one-token product on f8h_lin1_kernel's plan for a pack whose every patch is plain
// (GLYD_F8H_PLAIN=1: row blocks 0-2 of patch p at core + p HCORE, row block 3 at var + the head's x), with nothing a step but its copies, its bf16
// conversion and its 8 MMAs: what the step's structure costs. Two core slots and two X slots a warp; a step's words go to registers, then its slots
// take step j + 2's. The sums as f8h_lin1_kernel's, to the bit. GLYD_F8H_ONE=3 (only with plain packs: a coded patch read this way is wrong).
constexpr int BW = 2 * 512 + 2 * XW1;   // words of a warp's slots
template <int G, int C>
__device__ __forceinline__ void f8b_lin1_body(const F8Lin& P) {
    const uint8_t* core = P.core;
    const uint32_t* heads = P.heads;
    const float* scale = (const float*)P.scale;
    const uint16_t* x = P.x;
    uint16_t* y = P.y;
    float* parts = P.parts;
    int* done = P.done;
    if constexpr (G != 0) {
        F8Lin Pg;
        if (!f8_slot<F8_BLOCK, true>(P, Pg)) return;
        core = Pg.core, heads = Pg.heads, scale = (const float*)Pg.scale, y = Pg.y, parts = Pg.parts, done = Pg.done;
        x = Pg.order ? P.x + (int64_t)(Pg.order[0] / P.gk) * P.ldx : Pg.x;
    }
    const uint32_t* var = P.var;
    F8_EXTERN_SMEM(uint32_t, f8_smem);
    __shared__ int s_last;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3, mb = warp & 3, half = warp >> 2, g1 = g & 1;
    uint32_t* cw = f8_smem + warp * BW;  // two core slots (step j's at j & 1), then two X slots
    uint32_t* xw = cw + 2 * 512;
    const int64_t W = (int64_t)P.RG * P.KT, cta = blockIdx.x;
    const int64_t a0 = cta * W / P.ctas;
    const int n = (int)((cta + 1) * W / P.ctas - a0), KT = P.KT, RP = (int)f8_cdiv(P.N, PR), RB = RP * 4;
    const int rg0 = (int)(a0 / KT), kp0 = (int)(a0 % KT);
    auto step = [&](int& r, int& k) {
        if (++k == KT) k = 0, r++;
    };
    auto copies = [&](int sl, int r, int k) {  // step (r, k)'s row block into core slot sl (16 bytes q of the lane at (32 q + lane) * 16), its X into X slot sl
        if (2 * r + half < RP) {
            const int64_t p = (int64_t)(2 * r + half) * KT + k;
            const uint8_t* src = mb < 3 ? core + p * HCORE + mb * 2048 : (const uint8_t*)(var + __ldg(heads + p * HEAD));
            uint32_t* d = cw + sl * 512;
            // copy q: the warp's 512 contiguous bytes (chunk c = 32 q + lane of 16 bytes; lane c / 4's chunk c % 4), put where the decode's read of
            // lane l's chunk q' (position 4 l + ((q' + l / 2) & 3)) has 8 lanes in 8 bank groups
#pragma unroll
            for (int q = 0; q < 4; q++) {
                const int c = 32 * q + lane, l = c >> 2, qq = c & 3;
                f8_cp16(d + (4 * l + ((qq + (l >> 1)) & 3)) * 4, src + c * 16);
            }
        }
        if (lane < 16) {  // column k's X as f8_linear_kernel's x_in has a token's row (lanes 0-15: 8 columns each)
            const int kk = k * PC + 8 * lane, i0 = (lane & 3) * 2, i1 = i0 + 1;
            uint8_t* row = (uint8_t*)(xw + sl * XW1);
            uint32_t* d0 = (uint32_t*)(row + (lane >> 2) * 64 + 8 * (2 * (i0 & 3) + (i0 >> 2)));
            uint32_t* d1 = (uint32_t*)(row + (lane >> 2) * 64 + 8 * (2 * (i1 & 3) + (i1 >> 2)));
            const uint16_t* sx = x + kk;
            if (P.xvec && kk + 8 <= P.K) {
                f8_cp8(d0, (const uint32_t*)sx);
                f8_cp8(d1, (const uint32_t*)(sx + 4));
            } else {
                uint32_t v[4];
#pragma unroll
                for (int i = 0; i < 4; i++) v[i] = (kk + 2 * i < P.K ? (uint32_t)sx[2 * i] : 0u) | (kk + 2 * i + 1 < P.K ? (uint32_t)sx[2 * i + 1] << 16 : 0u);
                d0[0] = v[0], d0[1] = v[1], d1[0] = v[2], d1[1] = v[3];
            }
        }
    };
    int rgA = rg0, kpA = kp0, rgC = rg0, kpC = kp0;  // A the step decoded (j), C the step copied (j + 2)
    copies(0, rgC, kpC);
    f8_cp_commit();
    step(rgC, kpC);
    if (n > 1) copies(1, rgC, kpC);
    f8_cp_commit();
    step(rgC, kpC);
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int j = 0; j < n; j++) {
        const int sl = j & 1;
        f8_cp_wait1();  // step j's copies in (step j + 1's may still be coming)
        __syncwarp();
        const bool mn = 2 * rgA + half < RP;
        uint4 pv[4], xv[4];
        if (mn) {  // the lane's 64 bytes: chunk q' at position 4 lane + ((q' + lane / 2) & 3) (the copies above)
#pragma unroll
            for (int q = 0; q < 4; q++) pv[q] = ((const uint4*)(cw + sl * 512))[4 * lane + ((q + (lane >> 1)) & 3)];
        }
        {
            const uint8_t* xsl = (const uint8_t*)(xw + sl * XW1) + 16 * t;
#pragma unroll
            for (int ks = 0; ks < 4; ks++) xv[ks] = *(const uint4*)(xsl + 64 * (ks ^ g1));
        }
        __syncwarp();  // every lane past its reads: the slots take step j + 2's
        if (j + 2 < n) copies(sl, rgC, kpC);
        f8_cp_commit();
        if (mn) {
            const float s = scale[(int64_t)rgA * P.lds + kpA];
            float tmp[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int ks = 0; ks < 4; ks++) {
                const uint32_t a0_ = f8_bf_lo(pv[ks].x), a1_ = f8_bf_lo(pv[ks].y), a2_ = f8_bf_lo(pv[ks].z), a3_ = f8_bf_lo(pv[ks].w);
                const uint32_t b0_ = f8_bf_hi(pv[ks].x), b1_ = f8_bf_hi(pv[ks].y), b2_ = f8_bf_hi(pv[ks].z), b3_ = f8_bf_hi(pv[ks].w);
                const uint4 xb = make_uint4(f8_x64(xv[ks].x), f8_x64(xv[ks].y), f8_x64(xv[ks].z), f8_x64(xv[ks].w));
                f8_mma(tmp, a0_, a1_, a2_, a3_, xb.x, xb.z);
                f8_mma(tmp, b0_, b1_, b2_, b3_, xb.y, xb.w);
            }
#pragma unroll
            for (int i = 0; i < 4; i++) acc[i] += tmp[i] * s;
        }
        if (kpA == KT - 1 || j == n - 1) {  // the end of the CTA's part of row group rgA: its sums out, as f8_linear_kernel's
            const int rb = rgA * LW + warp, row0 = 16 * rb + g;
            const float f = 0x1p56f;  // (the bf16 weights' 2^-120 and X's 2^64)
            const int64_t i0 = (int64_t)rgA * KT;
            const int cf = (int)(((i0 + 1) * P.ctas - 1) / W), cl = (int)(((i0 + KT) * P.ctas - 1) / W);
            if (cf == cl) {
                if (rb < RB && t == 0) {
                    if (row0 < P.N) y[row0] = f8_to_bf16(acc[0] * f);
                    if (row0 + 8 < P.N) y[row0 + 8] = f8_to_bf16(acc[2] * f);
                }
            } else {
                float* pt = parts + (int64_t)(2 * (int)cta + (rgA == rg0 ? 0 : 1)) * 128;
                if (rb < RB && t == 0) pt[16 * warp + g] = acc[0] * f, pt[16 * warp + g + 8] = acc[2] * f;
                __threadfence();
                __syncthreads();
                if (tid == 0) {
                    s_last = atomicAdd(done + rgA, 1) == cl - cf;
                    if (s_last) done[rgA] = 0;  // ready for the next product
                }
                __syncthreads();
                if (s_last) {
                    __threadfence();
                    const int n0 = rgA * 128, wf = (int64_t)cf * W / P.ctas >= i0 ? 0 : 1;
                    for (int r = tid; r < 128; r += LNT) {
                        if (n0 + r >= P.N) continue;
                        float sum = 0.f;
                        for (int c = cf; c <= cl; c++) sum += __ldcg(parts + (int64_t)(2 * c + (c == cf ? wf : 0)) * 128 + r);
                        y[n0 + r] = f8_to_bf16(sum);
                    }
                }
            }
            acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
        }
        step(rgA, kpA);
        step(rgC, kpC);
    }
}

template <int G, int C>
__global__ void __launch_bounds__(LNT, C) f8b_lin1_kernel(const F8Lin P) {
    f8b_lin1_body<G, C>(P);
}

// Layout H, one token, lean (research, 2026-10-20): f8h_lin2_kernel's work with f8b_lin1_kernel's step. Two core slots (a step's words to registers, then
// the slot takes step j + 2's core) and two small slots (step j + 1's literals, X and head, filled a step ahead); three heads a warp in registers
// (steps j + 1, j + 2, j + 3: the small copies and the core copy need theirs before their step); X copied by all 32 lanes; no slot of three.
// The same plan and sums as f8h_lin1_kernel. GLYD_F8H_ONE=4.
constexpr int S3W = VB + XW1 + HEAD;      // words of a small slot
constexpr int L3W = 2 * CSL2 + 2 * S3W;   // words of a warp's slots
template <int G, int C>
__global__ void __launch_bounds__(LNT, C) f8h_lin3_kernel(const F8Lin P) {
    const uint8_t* core = P.core;
    const uint32_t* heads = P.heads;
    const float* scale = (const float*)P.scale;
    const uint16_t* x = P.x;
    uint16_t* y = P.y;
    float* parts = P.parts;
    int* done = P.done;
    if constexpr (G != 0) {
        F8Lin Pg;
        if (!f8_slot<F8_BLOCK, true>(P, Pg)) return;
        core = Pg.core, heads = Pg.heads, scale = (const float*)Pg.scale, y = Pg.y, parts = Pg.parts, done = Pg.done;
        x = Pg.order ? P.x + (int64_t)(Pg.order[0] / P.gk) * P.ldx : Pg.x;
    }
    const uint32_t* var = P.var;
    F8_EXTERN_SMEM(uint32_t, f8_smem);
    __shared__ int s_last;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3, mb = warp & 3, half = warp >> 2, g1 = g & 1;
    uint32_t* cw = f8_smem + warp * L3W;  // two core slots (step j's at j & 1), then two small slots (step j's at j & 1)
    uint32_t* sw = cw + 2 * CSL2;
    const int64_t W = (int64_t)P.RG * P.KT, cta = blockIdx.x;
    const int64_t a0 = cta * W / P.ctas;
    const int n = (int)((cta + 1) * W / P.ctas - a0), KT = P.KT, RP = (int)f8_cdiv(P.N, PR), RB = RP * 4;
    const int rg0 = (int)(a0 / KT), kp0 = (int)(a0 % KT);
    auto step = [&](int& r, int& k) {
        if (++k == KT) k = 0, r++;
    };
    auto patch = [&](int r, int k) { return (int64_t)(2 * r + half) * KT + k; };
    auto mine = [&](int r) { return 2 * r + half < RP; };
    auto head = [&](int r, int k, uint4 (&h)[2]) {
        if (mine(r)) {
            const uint4* s = (const uint4*)(heads + patch(r, k) * HEAD);
            h[0] = __ldg(s), h[1] = __ldg(s + 1);
        } else {
            h[0] = h[1] = make_uint4(0, 0, 0, 0);
        }
    };
    auto small = [&](int sl, int r, int k, const uint4 (&h)[2]) {  // step (r, k)'s literals, X and head into small slot sl
        uint32_t* s = sw + sl * S3W;
        if (mine(r)) f8h_var_in(var, mb, lane, h[0], h[1], s);
        {   // column k's X as f8_linear_kernel's x_in has a token's row, 8 bytes a lane (lane 2 l + hf: 16-lane l's bytes hf)
            const int l16 = lane >> 1, hf = lane & 1, kk = k * PC + 8 * l16, i = (l16 & 3) * 2 + hf;
            uint32_t* d = (uint32_t*)((uint8_t*)(s + VB) + (l16 >> 2) * 64 + 8 * (2 * (i & 3) + (i >> 2)));
            const uint16_t* sx = x + kk + 4 * hf;
            if (P.xvec && kk + 8 <= P.K) {
                f8_cp8(d, (const uint32_t*)sx);
            } else {
                const int k0 = kk + 4 * hf;
                d[0] = (k0 < P.K ? (uint32_t)sx[0] : 0u) | (k0 + 1 < P.K ? (uint32_t)sx[1] << 16 : 0u);
                d[1] = (k0 + 2 < P.K ? (uint32_t)sx[2] : 0u) | (k0 + 3 < P.K ? (uint32_t)sx[3] << 16 : 0u);
            }
        }
        if (lane < 2) ((uint4*)(s + VB + XW1))[lane] = lane ? h[1] : h[0];
    };
    auto core_in = [&](int sl, int r, int k, const uint4 (&h)[2]) {  // step (r, k)'s core, or a plain patch's bytes, into core slot sl
        if (!mine(r)) return;
        const int64_t p = patch(r, k);
        uint32_t* d = cw + sl * CSL2;
        if (h[0].w >> 31) {
            const uint8_t* src = mb < 3 ? core + p * HCORE + mb * 2048 : (const uint8_t*)(var + h[0].x);
#pragma unroll
            for (int q = 0; q < 4; q++) {
                const int c = 32 * q + lane, l = c >> 2, qq = c & 3;
                f8_cp16(d + (4 * l + ((qq + (l >> 1)) & 3)) * 4, src + c * 16);
            }
        } else {
            f8h_core_in(core, p, mb, lane, d);
        }
    };
    // cursors: A the step decoded (j), B the step whose small parts are copied (j + 1), D the step whose core is copied (j + 2), H the head read (j + 3)
    int rgA = rg0, kpA = kp0, rgB = rg0, kpB = kp0, rgD = rg0, kpD = kp0, rgH = rg0, kpH = kp0;
    uint4 h1[2], h2[2], h3[2] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
    {
        uint4 h0[2];
        head(rgA, kpA, h0);
        small(0, rgB, kpB, h0);
        core_in(0, rgD, kpD, h0);
        f8_cp_commit();
        step(rgB, kpB);
        step(rgD, kpD);
        step(rgH, kpH);
        if (n > 1) head(rgH, kpH, h1); else h1[0] = h1[1] = make_uint4(0, 0, 0, 0);
        step(rgH, kpH);
        if (n > 1) core_in(1, rgD, kpD, h1);
        f8_cp_commit();
        step(rgD, kpD);
        if (n > 2) head(rgH, kpH, h2); else h2[0] = h2[1] = make_uint4(0, 0, 0, 0);
        step(rgH, kpH);
    }
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int j = 0; j < n; j++) {
        const int sl = j & 1;
        const uint32_t* cs = cw + sl * CSL2;
        const uint32_t* ss = sw + sl * S3W;
        f8_cp_wait1();  // step j's small parts and core in (the newest group, step j + 1's core, may still be coming)
        __syncwarp();
        const uint4 hA = ((const uint4*)(ss + VB + XW1))[0], hB = ((const uint4*)(ss + VB + XW1))[1];
        const bool mn = mine(rgA), plain = (hA.w >> 31) != 0;
        uint32_t sm[8], cd[7];
        uint4 pv[4];
        if (mn) {
            if (plain) {
#pragma unroll
                for (int q = 0; q < 4; q++) pv[q] = ((const uint4*)cs)[4 * lane + ((q + (lane >> 1)) & 3)];
            } else {
                f8h_core_words(cs, lane, sm, cd);
            }
        }
        __syncwarp();  // every lane past its reads of the core slot
        if (j + 3 < n) head(rgH, kpH, h3); else h3[0] = h3[1] = make_uint4(0, 0, 0, 0);
        if (j + 1 < n) small(sl ^ 1, rgB, kpB, h1);
        f8_cp_commit();
        if (j + 2 < n) core_in(sl, rgD, kpD, h2);
        f8_cp_commit();
        const float s = mn ? scale[(int64_t)rgA * P.lds + kpA] : 0.f;
        if (mn) {
            uint32_t o[4][4];
            if (plain) {
#pragma unroll
                for (int q = 0; q < 4; q++) o[q][0] = pv[q].x, o[q][1] = pv[q].y, o[q][2] = pv[q].z, o[q][3] = pv[q].w;
            } else {
                uint32_t ls, le, w0;
                f8h_lits(hB, mb, ls, le);
                const uint32_t nw = f8h_words(hB, mb, w0);
                f8h_granule_regs(sm, cd, nw <= (uint32_t)VB ? ss : var + hA.x + w0, ls & 7, hA.y, lane, o);
            }
            const uint8_t* xsl = (const uint8_t*)(ss + VB) + 16 * t;
            float tmp[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int ks = 0; ks < 4; ks++) {
                const uint32_t a0_ = f8_bf_lo(o[ks][0]), a1_ = f8_bf_lo(o[ks][1]), a2_ = f8_bf_lo(o[ks][2]), a3_ = f8_bf_lo(o[ks][3]);
                const uint32_t b0_ = f8_bf_hi(o[ks][0]), b1_ = f8_bf_hi(o[ks][1]), b2_ = f8_bf_hi(o[ks][2]), b3_ = f8_bf_hi(o[ks][3]);
                const uint4 v = *(const uint4*)(xsl + 64 * (ks ^ g1));
                const uint4 xb = make_uint4(f8_x64(v.x), f8_x64(v.y), f8_x64(v.z), f8_x64(v.w));
                f8_mma(tmp, a0_, a1_, a2_, a3_, xb.x, xb.z);
                f8_mma(tmp, b0_, b1_, b2_, b3_, xb.y, xb.w);
            }
#pragma unroll
            for (int i = 0; i < 4; i++) acc[i] += tmp[i] * s;
        }
        if (kpA == KT - 1 || j == n - 1) {  // the end of the CTA's part of row group rgA: its sums out, as f8_linear_kernel's
            const int rb = rgA * LW + warp, row0 = 16 * rb + g;
            const float f = 0x1p56f;  // (the bf16 weights' 2^-120 and X's 2^64)
            const int64_t i0 = (int64_t)rgA * KT;
            const int cf = (int)(((i0 + 1) * P.ctas - 1) / W), cl = (int)(((i0 + KT) * P.ctas - 1) / W);
            if (cf == cl) {
                if (rb < RB && t == 0) {
                    if (row0 < P.N) y[row0] = f8_to_bf16(acc[0] * f);
                    if (row0 + 8 < P.N) y[row0 + 8] = f8_to_bf16(acc[2] * f);
                }
            } else {
                float* pt = parts + (int64_t)(2 * (int)cta + (rgA == rg0 ? 0 : 1)) * 128;
                if (rb < RB && t == 0) pt[16 * warp + g] = acc[0] * f, pt[16 * warp + g + 8] = acc[2] * f;
                __threadfence();
                __syncthreads();
                if (tid == 0) {
                    s_last = atomicAdd(done + rgA, 1) == cl - cf;
                    if (s_last) done[rgA] = 0;  // ready for the next product
                }
                __syncthreads();
                if (s_last) {
                    __threadfence();
                    const int n0 = rgA * 128, wf = (int64_t)cf * W / P.ctas >= i0 ? 0 : 1;
                    for (int r = tid; r < 128; r += LNT) {
                        if (n0 + r >= P.N) continue;
                        float sum = 0.f;
                        for (int c = cf; c <= cl; c++) sum += __ldcg(parts + (int64_t)(2 * c + (c == cf ? wf : 0)) * 128 + r);
                        y[n0 + r] = f8_to_bf16(sum);
                    }
                }
            }
            acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
        }
        h1[0] = h2[0], h1[1] = h2[1], h2[0] = h3[0], h2[1] = h3[1];
        step(rgA, kpA);
        step(rgB, kpB);
        step(rgD, kpD);
        step(rgH, kpH);
    }
}

}  // namespace f8c
// f8c end

// ---------------------------------------------------------------------------
// The prompts' products (R17 step 1: moved unchanged from glyd_gpu.cu, after this header's sections as they were there): split-K's finish, the prompt kernel
// (mma_gemm_big_kernel) and its stream-K form (mma_gemm_sk_kernel), Hopper's TMA and wgmma kernels (mma12_tma_kernel, tiered_tma_kernel, mma12_wgp_kernel),
// the 12-bit layout's staged products (mma12_mid_kernel, mma12_ws_kernel), and the experts' wgmma product (moe_wg_tiles_kernel, moe_wg_gather_kernel,
// mma12_moe_wg_kernel).

// (a lambda forced inline: MSVC's dialect, nvcc's on Windows, takes no attribute there, so the inliner decides)
#ifdef _MSC_VER
#define GLYD_LAMBDA_INLINE
#else
#define GLYD_LAMBDA_INLINE __attribute__((always_inline))
#endif

// Split-K's parts (one [M, O] slice each) added in a fixed order, so the
// result is the same every run, to bf16 with the bias.
__device__ __forceinline__ void finish_body(const float* __restrict__ y32, int64_t split, const __nv_bfloat16* __restrict__ bias, int64_t M, int64_t O, __nv_bfloat16* __restrict__ y) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= M * O) return;
    float t = 0.f;
    for (int64_t z = 0; z < split; z++) t += y32[z * M * O + i];
    y[i] = __float2bfloat16(t + (bias ? __bfloat162float(bias[i % O]) : 0.f));
}

__global__ void finish_kernel(const float* __restrict__ y32, int64_t split, const __nv_bfloat16* __restrict__ bias, int64_t M, int64_t O, __nv_bfloat16* __restrict__ y) {
    finish_body(y32, split, bias, M, O, y);
}

// Y = X W^T for many tokens from the mma layout (a prompt), as a tiled
// GEMM with its work split between warps: a block is TM tokens by RBB row
// blocks of W (64 rows each). K goes in stages of 4 steps (64 columns), NB
// of them in flight in shared memory. PW warps produce: X's tile of a
// stage by cp.async (swizzled for ldmatrix), and W's 4 RBB steps of it
// decoded into B fragments; CW warps consume, each 64 tokens by 64 rows
// on the tensor cores, never waiting on a decode. Named barriers pass a stage's
// buffers from producers to consumers (full) and back (empty). A weight
// is decoded once for TM tokens. Blocks may split K (Y32: their parts,
// added in a fixed order by finish_kernel). A mixture of experts' prompt runs
// it (MOE), and a dense one but on GeForce Ada: there mma_gemm_sk_kernel, its
// blocks by stream-K.
constexpr int BIG_KK = 4;  // steps a stage
template <int CW, int PW, int NB, int RBB, int CR = 64> struct Big {
    static constexpr int THREADS = 32 * (CW + PW);
    static constexpr int TM = CR * CW / RBB;             // tokens a block: its consumers of 64 tokens by CR rows, 64 / CR a row block, by RBB row blocks
    static constexpr int A_BYTES = TM * 64 * 2;          // X's tile a stage
    static constexpr int B_UINT4 = RBB * BIG_KK * 4 * 32;  // W's fragments a stage: [row block][step][4][lane]
    static constexpr int SHARED = NB * (A_BYTES + B_UINT4 * 16);
    static constexpr int STEPS = RBB * BIG_KK / PW;      // W's steps a producer warp decodes a stage
    static constexpr int CHUNKS = TM * 8 / (32 * PW);    // X's 16-byte chunks a producer thread copies a stage
};

// *p + 1 with release and acquire at the GPU's scope (a block's part made visible to the unit's last block, the others' to it): the value before.
__device__ __forceinline__ int atom_inc_acq_rel(int* p) {
#if defined(__HIP_PLATFORM_AMD__)
    return __hip_atomic_fetch_add(p, 1, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
#else
    int before;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;\n" : "=r"(before) : "l"(p) : "memory");
    return before;
#endif
}

// A mixture of experts' layer for mma_gemm_big_kernel (MOE; mma_moe_kernel has the plan and the rest): blockIdx.z a
// hit expert, M its pairs (a block TM of them); X's rows the tokens' (gather) or the pairs'; the output's as
// mma_moe_kernel's: with the gate's activation (act; RBB 2: blockIdx.y a row block of the gate's half, the
// consumers of row block 1 taking the up's, their sums met in shared memory) or times the weights (w, into Y32).
struct MoePairs {
    const int* plan;
    int E;
    int64_t k;
    int gather, act;
    const void* w;
    int wf32;
    int xahead, pf;  // the producers: X's tile of stage j + 1 copied before stage j is decoded (0: of stage j); the W lines of the stages pf ahead of the next asked of L2 (0: not)
    int skip = 1;    // the consumers multiply the m-tiles of 16 tokens that have pairs alone (an expert with 72 pairs has 8 in its tile's second 64: one m-tile of four)
};

template <class Fmt, int CW, int PW, int NB, int RBB, bool MOE>
__device__ __forceinline__ void mma_gemm_big_body(Fmt f, int64_t O, int64_t K, int64_t M, int64_t stages_per_split, const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ Y, float* __restrict__ Y32, MoePairs moe) {
    using C = Big<CW, PW, NB, RBB>;
    int64_t e = 0, p0 = 0;  // MOE: the expert (its rows e O on of W's), its first pair in the plan's order
    const int* order = nullptr;
    if constexpr (MOE) {
        if ((int)blockIdx.z >= __ldg(moe.plan)) return;  // the whole block: no expert this far down the hits
        e = __ldg(moe.plan + 1 + blockIdx.z);
        p0 = __ldg(moe.plan + 1 + moe.E + blockIdx.z);
        M = __ldg(moe.plan + 2 + moe.E + blockIdx.z) - p0;
        if ((int64_t)blockIdx.x * C::TM >= M) return;
        order = moe.plan + 2 + 2 * moe.E;
    }
    extern __shared__ uint4 smem[];  // X's tiles [NB][TM rows][8 16-byte chunks, swizzled], then W's [NB][B_UINT4]
    __shared__ uint32_t tab[Fmt::kTable ? 256 : 1];
    if constexpr (Fmt::kTable) fill_groups(tab);
    __syncthreads();
    const uint32_t a_base = (uint32_t)__cvta_generic_to_shared(smem);
    uint4* Bs = smem + NB * C::A_BYTES / 16;
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int64_t KS = K / 16, RB = O / 64, pair = blockIdx.y;
    int64_t st0 = MOE ? 0 : (int64_t)blockIdx.z * stages_per_split, nst = MOE ? KS / BIG_KK : min(KS / BIG_KK, st0 + stages_per_split) - st0;
    // A block's tile of TM tokens: blockIdx.x; MOE, and every gridDim.x on while the expert has pairs (gridDim.x is
    // sized for T an expert, the most where no token lists an expert twice), the stages' buffers passed on between
    // tiles once every warp is done with them.
    for (int64_t tile = blockIdx.x;; tile += gridDim.x) {
        do {  // a tile's work (a break: this warp is done with it)
            if (warp >= CW) {
                // Producer p: X's chunks id = pt + 32 PW i of a stage's; W's steps STEPS p on of its 4 RBB (row block, step).
                int pt = tid - 32 * CW, p = warp - CW;
                typename Fmt::St st[C::STEPS];
                auto d_load = [&](int64_t j) {
#pragma unroll
                    for (int i = 0; i < C::STEPS; i++) {
                        int u = p * C::STEPS + i;  // of the stage's 4 RBB: row block u / 4, step u % 4
                        int64_t rb = min(pair * RBB + u / BIG_KK, RB - 1), s = min((st0 + j) * BIG_KK + u % BIG_KK, KS - 1);
                        if constexpr (MOE) {
                            if (moe.act) rb = pair + u / BIG_KK * (RB / 2);  // the gate's row block, then the up's
                            f.load(st[i], (e * RB + rb) * KS + s, lane);
                        } else {
                            f.load(st[i], rb * KS + s, lane);
                        }
                    }
                };
                int xr[MOE ? C::CHUNKS : 1];  // MOE: X's row of each of this thread's chunks (-1: past the pairs)
                if constexpr (MOE)
#pragma unroll
                    for (int i = 0; i < C::CHUNKS; i++) {
                        int64_t m = tile * C::TM + ((pt + 32 * PW * i) >> 3);
                        xr[i] = m < M ? (int)(moe.gather ? __ldg(order + p0 + m) / moe.k : p0 + m) : -1;
                    }
                auto copy_x = [&](int64_t j, int b) {  // X's tile of stage j into buffer b
                    uint32_t slot = a_base + (uint32_t)b * C::A_BYTES;
                    int64_t col = (st0 + j) * BIG_KK * 16;
#pragma unroll
                    for (int i = 0; i < C::CHUNKS; i++) {
                        int id = pt + 32 * PW * i, r = id >> 3, c = id & 7;
                        if constexpr (MOE) {
                            cp_async16(slot + r * 128 + ((c ^ (r & 7)) << 4), X + (xr[i] < 0 ? 0 : (int64_t)xr[i]) * K + col + c * 8, xr[i] < 0 ? 0 : 16);
                        } else {
                            int64_t m = tile * C::TM + r;
                            cp_async16(slot + r * 128 + ((c ^ (r & 7)) << 4), X + (m < M ? m : 0) * K + col + c * 8, m < M ? 16 : 0);
                        }
                    }
                };
                auto d_prefetch = [&](int64_t j) {  // this warp's steps of stage j, their lines to L2
#pragma unroll
                    for (int i = 0; i < C::STEPS; i++) {
                        int u = p * C::STEPS + i;
                        int64_t rb = min(pair * RBB + u / BIG_KK, RB - 1), s = min((st0 + j) * BIG_KK + u % BIG_KK, KS - 1);
                        if constexpr (MOE) {
                            if (moe.act) rb = pair + u / BIG_KK * (RB / 2);
                            f.prefetch((e * RB + rb) * KS + s, lane);
                        } else {
                            f.prefetch(rb * KS + s, lane);
                        }
                    }
                };
                const bool ahead = MOE && moe.xahead != 0;  // X's tile a stage ahead (NB >= 3: the buffer the consumers free after stage j + 1 - NB)
                if (MOE && moe.pf)
                    for (int jj = 1; jj <= moe.pf && jj < nst; jj++) d_prefetch(jj);
                d_load(0);
                if (ahead) {
                    copy_x(0, 0);
                    cp_commit();
                }
                for (int64_t j = 0; j < nst; j++) {
                    int b = (int)(j % NB);
                    if (ahead) {
                        if (j + 1 < nst) {
                            int b1 = (int)((j + 1) % NB);
                            if (j + 1 >= NB) bar_sync<C::THREADS>(1 + NB + b1);  // consumers are done with stage j + 1 - NB
                            copy_x(j + 1, b1);
                        }
                    } else {
                        if (j >= NB) bar_sync<C::THREADS>(1 + NB + b);  // consumers are done with stage j - NB
                        copy_x(j, b);
                    }
                    cp_commit();
                    typename Fmt::St cur[C::STEPS];
#pragma unroll
                    for (int i = 0; i < C::STEPS; i++) cur[i] = st[i];
                    if (j + 1 < nst) d_load(j + 1);
                    if (MOE && moe.pf && j + 1 + moe.pf < nst) d_prefetch(j + 1 + moe.pf);
#pragma unroll
                    for (int i = 0; i < C::STEPS; i++) {
                        uint32_t R[16];
                        uint4* slot = Bs + b * C::B_UINT4 + ((p * C::STEPS + i) * 4) * 32;
                        if constexpr (!Fmt::kTable) f.decode_xz(cur[i], lane, (uint32_t*)slot, R);  // (12-bit) many exceptions: through the step's slot, its scratch till then
                        else f.decode(cur[i], lane, (uint32_t*)slot, tab, R);  // (tiered) the step's slot its scratch till then
                        uint4* d = slot + lane;
#pragma unroll
                        for (int q = 0; q < 4; q++) d[q * 32] = make_uint4(R[4 * q], R[4 * q + 1], R[4 * q + 2], R[4 * q + 3]);
                    }
                    if (ahead) cp_wait<1>();  // stage j's tile (stage j + 1's may still come)
                    else cp_wait<0>();
                    bar_arrive<C::THREADS>(1 + b);  // stage j is ready
                }
                break;
            }
            // Consumer: tokens 64 (warp % (CW / RBB)) on of the block's TM, row block warp / (CW / RBB) of its RBB.
            int g = lane >> 2, t = lane & 3, wm = warp % (CW / RBB), rbl = warp / (CW / RBB);
            float acc[4][8][4];
#pragma unroll
            for (int mt = 0; mt < 4; mt++)
#pragma unroll
                for (int nn = 0; nn < 8; nn++)
#pragma unroll
                    for (int q = 0; q < 4; q++) acc[mt][nn][q] = 0.f;
            // ldmatrix rows: this lane's row of the 16 (0-7, then 8-15) and its 16-byte half of k.
            uint32_t a_row = (uint32_t)(wm * 64 + (lane & 7) + ((lane >> 3) & 1) * 8) * 128, a_half = lane >> 4, a_sw = lane & 7;
            // MOE: the m-tiles (of 16 tokens) of this warp's 64 that hold pairs: the others, past the expert's last pair, are not multiplied (the same for every lane of the warp)
            int nmt = 4;
            if constexpr (MOE) {
                if (moe.skip) {
                    const int64_t left = M - (tile * C::TM + wm * 64);
                    nmt = left >= 64 ? 4 : left <= 0 ? 0 : (int)((left + 15) >> 4);
                }
            }
            if constexpr (CW >= 8) {
                // Eight consumer warps (a thread's 128 sums in 168 registers): a step's fragments loaded as it is
                // multiplied, the B ones a pair of n-tiles at a time (each sum's products in the same order).
                for (int64_t j = 0; j < nst; j++) {
                    int bi = (int)(j % NB);
                    bar_sync<C::THREADS>(1 + bi);  // stage j is ready
                    uint32_t slot = a_base + (uint32_t)bi * C::A_BYTES;
                    const uint4* bsrc = Bs + bi * C::B_UINT4 + rbl * BIG_KK * 4 * 32 + lane;
#pragma unroll
                    for (int kk = 0; kk < BIG_KK; kk++) {
                        uint32_t a[4][4];
#pragma unroll
                        for (int mt = 0; mt < 4; mt++)
                            if (mt < nmt) ldmatrix_x4(a[mt], slot + a_row + mt * 16 * 128 + (((2 * kk + a_half) ^ a_sw) << 4));
#pragma unroll
                        for (int q = 0; q < 4; q++) {
                            uint4 b = bsrc[(kk * 4 + q) * 32];
#pragma unroll
                            for (int mt = 0; mt < 4; mt++) {
                                if (mt >= nmt) continue;
                                mma16816(acc[mt][2 * q], a[mt], b.x, b.y);
                                mma16816(acc[mt][2 * q + 1], a[mt], b.z, b.w);
                            }
                        }
                    }
                    if (j + NB < nst) bar_arrive<C::THREADS>(1 + NB + bi);  // done with stage j's buffers
                }
            } else {
                // Fragments double-buffered: a step's are asked for while the step
                // before multiplies; a stage's first, during the last of the one before.
                uint32_t fa[2][4][4];
                uint4 fb[2][4];
                auto frags = [&](int64_t j, int kk, int f) {
                    int bi = (int)(j % NB);
                    uint32_t slot = a_base + (uint32_t)bi * C::A_BYTES;
                    const uint4* bsrc = Bs + bi * C::B_UINT4 + rbl * BIG_KK * 4 * 32 + lane;
#pragma unroll
                    for (int mt = 0; mt < 4; mt++)
                        if (mt < nmt) ldmatrix_x4(fa[f][mt], slot + a_row + mt * 16 * 128 + (((2 * kk + a_half) ^ a_sw) << 4));
#pragma unroll
                    for (int q = 0; q < 4; q++) fb[f][q] = bsrc[(kk * 4 + q) * 32];
                };
                if (nst > 0) {
                    bar_sync<C::THREADS>(1);  // stage 0 is ready
                    frags(0, 0, 0);
                }
                for (int64_t j = 0; j < nst; j++) {
#pragma unroll
                    for (int kk = 0; kk < BIG_KK; kk++) {
                        int f = kk & 1;
                        if (kk + 1 < BIG_KK) frags(j, kk + 1, f ^ 1);
                        else if (j + 1 < nst) {
                            bar_sync<C::THREADS>(1 + (int)((j + 1) % NB));  // stage j + 1 is ready
                            frags(j + 1, 0, f ^ 1);
                        }
#pragma unroll
                        for (int mt = 0; mt < 4; mt++) {
                            if (mt >= nmt) continue;
                            mma16816(acc[mt][0], fa[f][mt], fb[f][0].x, fb[f][0].y);
                            mma16816(acc[mt][1], fa[f][mt], fb[f][0].z, fb[f][0].w);
                            mma16816(acc[mt][2], fa[f][mt], fb[f][1].x, fb[f][1].y);
                            mma16816(acc[mt][3], fa[f][mt], fb[f][1].z, fb[f][1].w);
                            mma16816(acc[mt][4], fa[f][mt], fb[f][2].x, fb[f][2].y);
                            mma16816(acc[mt][5], fa[f][mt], fb[f][2].z, fb[f][2].w);
                            mma16816(acc[mt][6], fa[f][mt], fb[f][3].x, fb[f][3].y);
                            mma16816(acc[mt][7], fa[f][mt], fb[f][3].z, fb[f][3].w);
                        }
                    }
                    if (j + NB < nst) bar_arrive<C::THREADS>(1 + NB + (int)(j % NB));  // done with stage j's buffers
                }
            }
            // C fragments: token rows g and g + 8 of each m-tile, columns 2t and 2t + 1 of each n-tile.
            if constexpr (MOE) {
                if (moe.act) {
                    // The up's sums (row block 1) into shared memory, the stages' buffers once every consumer is done with
                    // them; the gate's warps of the same tokens take theirs, in the same fragment order.
                    float* up = (float*)smem + wm * 128 * 32 + lane;
                    bar_sync<32 * CW>(1 + 2 * NB);
                    if (rbl == 1)
#pragma unroll
                        for (int mt = 0; mt < 4; mt++)
#pragma unroll
                            for (int nn = 0; nn < 8; nn++)
#pragma unroll
                                for (int q = 0; q < 4; q++) up[((mt * 8 + nn) * 4 + q) * 32] = acc[mt][nn][q];
                    bar_sync<32 * CW>(1 + 2 * NB);
                    if (rbl == 1) break;
                    // (the activation chosen once, not at each of the 128 products: SiLU's code and GELU's, one among the other at each, filled the instruction cache)
                    auto epilogue = [&](auto kind) GLYD_LAMBDA_INLINE {
#pragma unroll
                        for (int mt = 0; mt < 4; mt++)
#pragma unroll
                            for (int nn = 0; nn < 8; nn++) {
                                int64_t o = pair * 64 + nn * 8 + t * 2, at = e * O + o;  // the gate's row; the up's O / 2 on
#pragma unroll
                                for (int h = 0; h < 2; h++) {
                                    int64_t m = tile * C::TM + wm * 64 + mt * 16 + g + h * 8;
                                    if (m >= M) continue;
                                    float v[2];
#pragma unroll
                                    for (int c = 0; c < 2; c++) {
                                        float gate = acc[mt][nn][2 * h + c] + (bias ? __bfloat162float(bias[at + c]) : 0.f);
                                        float u = up[((mt * 8 + nn) * 4 + 2 * h + c) * 32] + (bias ? __bfloat162float(bias[at + O / 2 + c]) : 0.f);
                                        v[c] = (decltype(kind)::value == 1 ? silu_fast(gate) : gelu_tanh(gate)) * u;
                                    }
                                    *(__nv_bfloat162*)(Y + (p0 + m) * (O / 2) + o) = __floats2bfloat162_rn(v[0], v[1]);
                                }
                            }
                    };
                    if (moe.act == 1) epilogue(std::integral_constant<int, 1>());
                    else epilogue(std::integral_constant<int, 2>());
                    break;
                }
            }
            int64_t my_rb = pair * RBB + rbl;
            if (my_rb >= RB) break;
            int jr[MOE ? 4 : 1][2];  // MOE with Y32: this thread's 8 pairs' rows of Y32 and their weights, asked for together before the first is stored
            float wr[MOE ? 4 : 1][2];
            if constexpr (MOE) {
                if (Y32) {
#pragma unroll
                    for (int mt = 0; mt < 4; mt++)
#pragma unroll
                        for (int h = 0; h < 2; h++) {
                            int64_t m = tile * C::TM + wm * 64 + mt * 16 + g + h * 8;
                            jr[mt][h] = 0;
                            wr[mt][h] = 0.f;
                            if (m < M) {
                                jr[mt][h] = (int)__ldg(order + p0 + m);
                                wr[mt][h] = moe.wf32 ? __ldg((const float*)moe.w + jr[mt][h]) : __bfloat162float(__ldg((const __nv_bfloat16*)moe.w + jr[mt][h]));
                            }
                        }
                }
            }
#pragma unroll
            for (int mt = 0; mt < 4; mt++)
#pragma unroll
                for (int nn = 0; nn < 8; nn++) {
                    int64_t o = my_rb * 64 + nn * 8 + t * 2;
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        int64_t m = tile * C::TM + wm * 64 + mt * 16 + g + h * 8;
                        if (m >= M) continue;
                        float v0 = acc[mt][nn][2 * h], v1 = acc[mt][nn][2 * h + 1];
                        if constexpr (MOE) {
                            if (bias) {
                                v0 += __bfloat162float(bias[e * O + o]);
                                v1 += __bfloat162float(bias[e * O + o + 1]);
                            }
                            if (Y32) {
                                const int64_t j = jr[mt][h];
                                const float wj = wr[mt][h];
                                *(float2*)(Y32 + j * O + o) = make_float2(v0 * wj, v1 * wj);
                            } else {
                                *(__nv_bfloat162*)(Y + (p0 + m) * O + o) = __floats2bfloat162_rn(v0, v1);
                            }
                        } else if (Y32) {
                            *(float2*)(Y32 + ((int64_t)blockIdx.z * M + m) * O + o) = make_float2(v0, v1);
                        } else {
                            if (bias) {
                                v0 += __bfloat162float(bias[o]);
                                v1 += __bfloat162float(bias[o + 1]);
                            }
                            *(__nv_bfloat162*)(Y + m * O + o) = __floats2bfloat162_rn(v0, v1);
                        }
                    }
                }
        } while (0);
        if (!MOE || (tile + gridDim.x) * C::TM >= M) break;
        __syncthreads();
    }
}

template <class Fmt, int CW, int PW, int NB, int RBB, bool MOE = false>
__global__ void __launch_bounds__((Big<CW, PW, NB, RBB>::THREADS), 1) mma_gemm_big_kernel(Fmt f, int64_t O, int64_t K, int64_t M, int64_t stages_per_split, const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ Y, float* __restrict__ Y32, MoePairs moe) {
    mma_gemm_big_body<Fmt, CW, PW, NB, RBB, MOE>(f, O, K, M, stages_per_split, X, bias, Y, Y32, moe);
}

// A prompt's product on GeForce Ada (not a mixture of experts'): mma_gemm_big_kernel's blocks, TM tokens by RBB row blocks, and its
// producer and consumer warps, as many blocks as the GPU holds at once, each taking an equal share of the units'
// stages in turn (a unit: a tile of tokens by a pair of row blocks; units by pair, then tile), its stages kept in
// flight from one unit to the next (stream-K: no wave part empty, no split of K summed by another kernel). A unit
// that blocks share: each writes its sums to a slot (parts: two a block, for the unit it starts in and the one it
// ends in, [2 gridDim.x][TM 64 RBB] floats in fragment order), and the last of them to finish adds the slots in
// block order (the same sum every run) with the bias (done: a counter a unit, zero before, reset by the last).
// A consumer multiplies 64 tokens by CR rows: 64 (4 consumers, 128 sums a thread) or 32, half a row block (8
// consumers, 64 sums a thread, which with 4 producers fit the 168 registers a thread of 12 warps: two consumers a
// scheduler, not one, each as lean: 2.2 instructions a product, and a weight still decoded once for TM tokens). The
// sums' order is a unit's stages', whatever CR: the same bits for the same TM and RBB. LIVE: the consumers multiply only
// the m-tiles that hold tokens (tiered_mid_run's shape, where it was measured; GeForce Ada's prompts as before, until measured there).
template <class Fmt, int CW, int PW, int NB, int RBB, int CR, bool LIVE>
__device__ __forceinline__ void mma_gemm_sk_body(Fmt f, int64_t O, int64_t K, int64_t M, const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ Y, float* __restrict__ parts, int* __restrict__ done) {
    using C = Big<CW, PW, NB, RBB, CR>;
    constexpr int NN = CR / 8, TW = C::TM / 64, HR = 64 / CR;  // a consumer's n-tiles; consumers along the tokens; a row block's
    constexpr int64_t SLOT = C::TM * 64 * RBB;
    extern __shared__ uint4 smem[];  // X's tiles [NB][TM rows][8 16-byte chunks, swizzled], then W's [NB][B_UINT4]
    __shared__ uint32_t tab[Fmt::kTable ? 256 : 1];
    __shared__ int last;
    if constexpr (Fmt::kTable) fill_groups(tab);
    __syncthreads();
    const uint32_t a_base = (uint32_t)__cvta_generic_to_shared(smem);
    uint4* Bs = smem + NB * C::A_BYTES / 16;
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int64_t KS = K / 16, RB = O / 64, tiles = (M + C::TM - 1) / C::TM, nb = gridDim.x;
    const int S = (int)(KS / BIG_KK);  // stages a unit
    const int64_t total = tiles * ((RB + RBB - 1) / RBB) * S, it0 = blockIdx.x * total / nb, n = (blockIdx.x + 1) * total / nb - it0;
    const int64_t u0 = it0 / S;  // the unit this block starts in, at its stage s0 (then each stage's place kept in turn)
    const int s0 = (int)(it0 - u0 * S);
    if (warp >= CW) {
        // Producer p: X's chunks id = pt + 32 PW i of a stage's; W's steps STEPS p on of its 4 RBB (row block, step).
        int pt = tid - 32 * CW, p = warp - CW, s = s0;
        int64_t tile = u0 % tiles, pair = u0 / tiles;
        typename Fmt::St st[C::STEPS];
        auto d_load = [&](int64_t pair, int s) {
#pragma unroll
            for (int i = 0; i < C::STEPS; i++) {
                int u = p * C::STEPS + i;  // of the stage's 4 RBB: row block u / 4, step u % 4
                f.load(st[i], min(pair * RBB + u / BIG_KK, RB - 1) * KS + s * BIG_KK + u % BIG_KK, lane);
            }
        };
        if (n > 0) d_load(pair, s);
        for (int64_t j = 0; j < n; j++) {
            int ns = s + 1;  // the next stage's place
            int64_t ntile = tile, npair = pair;
            if (ns == S) {
                ns = 0;
                if (++ntile == tiles) ntile = 0, npair++;
            }
            int b = (int)(j % NB);
            if (j >= NB) bar_sync<C::THREADS>(1 + NB + b);  // consumers are done with stage j - NB
            uint32_t slot = a_base + (uint32_t)b * C::A_BYTES;
            int64_t col = (int64_t)s * BIG_KK * 16;
#pragma unroll
            for (int i = 0; i < C::CHUNKS; i++) {
                int id = pt + 32 * PW * i, r = id >> 3, c = id & 7;
                int64_t m = tile * C::TM + r;
                cp_async16(slot + r * 128 + ((c ^ (r & 7)) << 4), X + (m < M ? m : 0) * K + col + c * 8, m < M ? 16 : 0);
            }
            cp_commit();
            typename Fmt::St cur[C::STEPS];
#pragma unroll
            for (int i = 0; i < C::STEPS; i++) cur[i] = st[i];
            if (j + 1 < n) d_load(npair, ns);
#pragma unroll
            for (int i = 0; i < C::STEPS; i++) {
                uint32_t R[16];
                uint4* sl = Bs + b * C::B_UINT4 + ((p * C::STEPS + i) * 4) * 32;
                f.decode(cur[i], lane, (uint32_t*)sl, tab, R);  // (tiered) the step's slot its scratch till then
                uint4* d = sl + lane;
#pragma unroll
                for (int q = 0; q < 4; q++) d[q * 32] = make_uint4(R[4 * q], R[4 * q + 1], R[4 * q + 2], R[4 * q + 3]);
            }
            cp_wait<0>();
            bar_arrive<C::THREADS>(1 + b);  // stage j is ready
            s = ns, tile = ntile, pair = npair;
        }
        return;
    }
    // Consumer: tokens 64 wm on of a unit's TM, rows CR hf on of row block rbl of its RBB (consumers by token slice,
    // then half a row block, then row block).
    int g = lane >> 2, t = lane & 3, wm = warp % TW, hf = warp / TW % HR, rbl = warp / (TW * HR);
    float acc[4][NN][4];
#pragma unroll
    for (int mt = 0; mt < 4; mt++)
#pragma unroll
        for (int nn = 0; nn < NN; nn++)
#pragma unroll
            for (int q = 0; q < 4; q++) acc[mt][nn][q] = 0.f;
    uint32_t a_row = (uint32_t)(wm * 64 + (lane & 7) + ((lane >> 3) & 1) * 8) * 128, a_half = lane >> 4, a_sw = lane & 7;
    uint32_t fa[2][4][4];
    uint4 fb[2][NN / 2];
    auto frags = [&](int64_t j, int kk, int fi) {
        int bi = (int)(j % NB);
        uint32_t slot = a_base + (uint32_t)bi * C::A_BYTES;
        const uint4* bsrc = Bs + bi * C::B_UINT4 + rbl * BIG_KK * 4 * 32 + lane;
#pragma unroll
        for (int mt = 0; mt < 4; mt++) ldmatrix_x4(fa[fi][mt], slot + a_row + mt * 16 * 128 + (((2 * kk + a_half) ^ a_sw) << 4));
#pragma unroll
        for (int q = 0; q < NN / 2; q++) fb[fi][q] = bsrc[(kk * 4 + NN / 2 * hf + q) * 32];
    };
    auto next = [&](int64_t j, int fi) {  // stage j's first fragments, once it is ready
        bar_sync<C::THREADS>(1 + (int)(j % NB));
        frags(j, 0, fi);
    };
    if (n > 0) next(0, 0);
    int s = s0;
    for (int64_t j = 0, u = u0; j < n; j++) {
        bool end = s == S - 1 || j + 1 == n;  // the unit's last stage here: its sums out after it
        // LIVE: the m-tiles (16 tokens) of this warp's 64 that hold tokens of the unit's tile: the others, past M, are not multiplied
        // (their sums, zero, are never stored; the rows that are keep their products and their order)
        const int64_t left = M - u % tiles * C::TM - wm * 64;
        const int nmt = !LIVE || left >= 64 ? 4 : left <= 0 ? 0 : (int)((left + 15) >> 4);
#pragma unroll
        for (int kk = 0; kk < BIG_KK; kk++) {
            int fi = kk & 1;
            if (kk + 1 < BIG_KK) frags(j, kk + 1, fi ^ 1);
            else if (j + 1 < n) next(j + 1, fi ^ 1);
#pragma unroll
            for (int mt = 0; mt < 4; mt++)
                if (mt < nmt)
#pragma unroll
                    for (int q = 0; q < NN / 2; q++) {
                        mma16816(acc[mt][2 * q], fa[fi][mt], fb[fi][q].x, fb[fi][q].y);
                        mma16816(acc[mt][2 * q + 1], fa[fi][mt], fb[fi][q].z, fb[fi][q].w);
                    }
        }
        if (j + NB < n) bar_arrive<C::THREADS>(1 + NB + (int)(j % NB));  // done with stage j's buffers
        if (++s == S) s = 0;
        if (!end) continue;
        // The unit's stages here are done: its blocks, from the one its first stage is in to its last's; a block's
        // slot for it: 2b + 1 for the one it started before it (first, where it did), else 2b.
        int64_t first = block_of_step(u * S, nb, total), fin = block_of_step(u * S + S - 1, nb, total);
        bool out = first == fin;
        if (!out) {
            auto slot_of = [&](int64_t b) { return (float4*)(parts + (2 * b + (b == first && first * total / nb < u * S)) * SLOT) + tid; };
            float4* mine = slot_of(blockIdx.x);
#pragma unroll
            for (int mt = 0; mt < 4; mt++)
#pragma unroll
                for (int nn = 0; nn < NN; nn++) mine[(mt * NN + nn) * 32 * CW] = make_float4(acc[mt][nn][0], acc[mt][nn][1], acc[mt][nn][2], acc[mt][nn][3]);
            __threadfence();
            bar_sync<32 * CW>(1 + 2 * NB);
            if (tid == 0) {
                last = atomicAdd(done + u, 1) == fin - first;
                if (last) done[u] = 0;  // ready for the next product
            }
            bar_sync<32 * CW>(1 + 2 * NB);
            out = last;
            if (out) {  // every block's slot in block order (this one's read back too), NN loads in flight at a time
                __threadfence();
#pragma unroll
                for (int mt = 0; mt < 4; mt++)
#pragma unroll
                    for (int nn = 0; nn < NN; nn++)
#pragma unroll
                        for (int q = 0; q < 4; q++) acc[mt][nn][q] = 0.f;
#pragma unroll
                for (int mt = 0; mt < 4; mt++)  // (more at once: more registers than the kernel has)
                    for (int64_t b = first; b <= fin; b++) {
                        const float4* sl = slot_of(b) + mt * NN * 32 * CW;
#pragma unroll
                        for (int nn = 0; nn < NN; nn++) {
                            float4 w = __ldcg(sl + nn * 32 * CW);
                            acc[mt][nn][0] += w.x;
                            acc[mt][nn][1] += w.y;
                            acc[mt][nn][2] += w.z;
                            acc[mt][nn][3] += w.w;
                        }
                    }
            }
        }
        if (out) {
            int64_t tile = u % tiles, my_rb = u / tiles * RBB + rbl;
#pragma unroll
            for (int mt = 0; mt < 4; mt++)
#pragma unroll
                for (int nn = 0; nn < NN; nn++) {
                    int64_t o = my_rb * 64 + CR * hf + nn * 8 + t * 2;
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        int64_t m = tile * C::TM + wm * 64 + mt * 16 + g + h * 8;
                        if (my_rb >= RB || m >= M) continue;
                        float v0 = acc[mt][nn][2 * h], v1 = acc[mt][nn][2 * h + 1];
                        if (bias) {
                            v0 += __bfloat162float(bias[o]);
                            v1 += __bfloat162float(bias[o + 1]);
                        }
                        *(__nv_bfloat162*)(Y + m * O + o) = __floats2bfloat162_rn(v0, v1);
                    }
                }
        }
#pragma unroll
        for (int mt = 0; mt < 4; mt++)
#pragma unroll
            for (int nn = 0; nn < NN; nn++)
#pragma unroll
                for (int q = 0; q < 4; q++) acc[mt][nn][q] = 0.f;
        u++;
    }
}

template <class Fmt, int CW, int PW, int NB, int RBB, int CR = 64, bool LIVE = false>
__global__ void __launch_bounds__((Big<CW, PW, NB, RBB, CR>::THREADS), 1) mma_gemm_sk_kernel(Fmt f, int64_t O, int64_t K, int64_t M, const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ Y, float* __restrict__ parts, int* __restrict__ done) {
    mma_gemm_sk_body<Fmt, CW, PW, NB, RBB, CR, LIVE>(f, O, K, M, X, bias, Y, parts, done);
}

// Many tokens on Hopper (sm_90a), the 12-bit layout: wgmma, Hopper's
// warpgroup tensor-core instruction, with W as its A operand, decoded
// straight into registers, and the tokens as B, read from shared memory
// (as Machete, a mixed-input GEMM for Hopper, does with 4-bit weights).
// The copies are the TMA's: one lane of a warp of its own hands the copy
// engine a stage at a time, 64 columns of a unit of WG row blocks (64 of
// W's rows each): each row block's 4 compressed steps (a run of 6144
// bytes), their exceptions, and X's tile through a tensor map, laid out in
// wgmma's 128-byte swizzle (a row's 16-byte chunk c at chunk c ^ (row mod
// 8)); each stage lands on an mbarrier, NS of them in flight. WG consumer
// warpgroups, a row block each: a warp decodes its 16 rows of the stage's
// 4 steps (its quarter of each: a word of codes and 8 low bytes a lane)
// into A fragments, then 4 wgmma of 64 rows by NT tokens by
// 16 columns. The work (units by stages) is split evenly over the blocks
// (stream-K, as mma_gemm_kernel's): a unit covered by several blocks is
// summed by the last to finish, in block order.
__device__ __forceinline__ uint64_t sw128_desc(uint32_t addr) {
    // Start address >> 4, stride between 8-row groups 1024 bytes, 128-byte swizzle (K-major).
    return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}

// wgmma is sm_90a's alone (sm_90's PTX has none, nor Blackwell's sm_100 and sm_120): its code and the TMA
// kernel's body compile for sm_90a only (__CUDA_ARCH_FEAT_SM90_ALL), a trap for the other targets, where the
// host never launches that kernel (compute capability 9.0 alone): run anyway (an H100 on a build for sm_90
// without the a, or on compute_80 PTX), it fails loudly rather than leave Y as it was.
__device__ __forceinline__ void wg_fence() {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#endif
}
template <int N> __device__ __forceinline__ void wg_wait() {  // until at most N groups of products are running
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
#endif
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ void mbar_init(uint32_t bar, int count) { asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(bar), "r"(count)); }
__device__ __forceinline__ void mbar_arrive(uint32_t bar) { asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.shared.b64 st, [%0];\n}\n" ::"r"(bar) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
    uint32_t ok;
    do {
#if __CUDA_ARCH__ >= 900
        asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n" : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
#else
        asm volatile("{\n.reg .pred p;\nmbarrier.test_wait.parity.shared.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n" : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
#endif
    } while (!ok);
}
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar), "r"(bytes) : "memory"); }
// bytes (a multiple of 16) from global memory to shared, counted on the mbarrier;
// once: read to be evicted from L2 first (data read once, in a stream).
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar, bool once = false) {
    if (once) {
        uint64_t pol;
        asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n" ::"r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(pol) : "memory");
    } else {
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
    }
}
// A 2-D tile through a tensor map (coordinates innermost first).
__device__ __forceinline__ void tma_2d(uint32_t dst, const CUtensorMap* map, int c0, int c1, uint32_t bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" ::"r"(dst), "l"((uint64_t)map), "r"(c0), "r"(c1), "r"(bar) : "memory");
}

#endif
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
// A stage's 4 wgmma m64nNk16 (64 columns), A (W's 64 rows by 16 columns each) from registers, B (N tokens) from
// shared memory, then their commit: D = A B + (acc ? D : 0) for the first, D += A B for the rest.
__device__ __forceinline__ void wgmma4_rs(float (&d)[8], const uint32_t (&a)[4][4], const uint64_t (&db)[4], int acc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %28, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %24, p, 1, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%12, %13, %14, %15}, %25, 1, 1, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%16, %17, %18, %19}, %26, 1, 1, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%20, %21, %22, %23}, %27, 1, 1, 1, 0;\n"
                 "wgmma.commit_group.sync.aligned;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                 : "r"(a[0][0]), "r"(a[0][1]), "r"(a[0][2]), "r"(a[0][3]), "r"(a[1][0]), "r"(a[1][1]), "r"(a[1][2]), "r"(a[1][3]), "r"(a[2][0]), "r"(a[2][1]), "r"(a[2][2]), "r"(a[2][3]), "r"(a[3][0]), "r"(a[3][1]), "r"(a[3][2]), "r"(a[3][3]), "l"(db[0]), "l"(db[1]), "l"(db[2]), "l"(db[3]), "r"(acc));
}
__device__ __forceinline__ void wgmma4_rs(float (&d)[16], const uint32_t (&a)[4][4], const uint64_t (&db)[4], int acc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %36, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %32, p, 1, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%20, %21, %22, %23}, %33, 1, 1, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%24, %25, %26, %27}, %34, 1, 1, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%28, %29, %30, %31}, %35, 1, 1, 1, 0;\n"
                 "wgmma.commit_group.sync.aligned;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
                 : "r"(a[0][0]), "r"(a[0][1]), "r"(a[0][2]), "r"(a[0][3]), "r"(a[1][0]), "r"(a[1][1]), "r"(a[1][2]), "r"(a[1][3]), "r"(a[2][0]), "r"(a[2][1]), "r"(a[2][2]), "r"(a[2][3]), "r"(a[3][0]), "r"(a[3][1]), "r"(a[3][2]), "r"(a[3][3]), "l"(db[0]), "l"(db[1]), "l"(db[2]), "l"(db[3]), "r"(acc));
}
__device__ __forceinline__ void wgmma4_rs(float (&d)[32], const uint32_t (&a)[4][4], const uint64_t (&db)[4], int acc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %52, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %48, p, 1, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%36, %37, %38, %39}, %49, 1, 1, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%40, %41, %42, %43}, %50, 1, 1, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%44, %45, %46, %47}, %51, 1, 1, 1, 0;\n"
                 "wgmma.commit_group.sync.aligned;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
                 : "r"(a[0][0]), "r"(a[0][1]), "r"(a[0][2]), "r"(a[0][3]), "r"(a[1][0]), "r"(a[1][1]), "r"(a[1][2]), "r"(a[1][3]), "r"(a[2][0]), "r"(a[2][1]), "r"(a[2][2]), "r"(a[2][3]), "r"(a[3][0]), "r"(a[3][1]), "r"(a[3][2]), "r"(a[3][3]), "l"(db[0]), "l"(db[1]), "l"(db[2]), "l"(db[3]), "r"(acc));
}
__device__ __forceinline__ void wgmma4_rs(float (&d)[48], const uint32_t (&a)[4][4], const uint64_t (&db)[4], int acc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %68, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47}, {%48, %49, %50, %51}, %64, p, 1, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47}, {%52, %53, %54, %55}, %65, 1, 1, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47}, {%56, %57, %58, %59}, %66, 1, 1, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47}, {%60, %61, %62, %63}, %67, 1, 1, 1, 0;\n"
                 "wgmma.commit_group.sync.aligned;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47])
                 : "r"(a[0][0]), "r"(a[0][1]), "r"(a[0][2]), "r"(a[0][3]), "r"(a[1][0]), "r"(a[1][1]), "r"(a[1][2]), "r"(a[1][3]), "r"(a[2][0]), "r"(a[2][1]), "r"(a[2][2]), "r"(a[2][3]), "r"(a[3][0]), "r"(a[3][1]), "r"(a[3][2]), "r"(a[3][3]), "l"(db[0]), "l"(db[1]), "l"(db[2]), "l"(db[3]), "r"(acc));
}
__device__ __forceinline__ void wgmma4_rs(float (&d)[56], const uint32_t (&a)[4][4], const uint64_t (&db)[4], int acc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %76, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n112k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55}, {%56, %57, %58, %59}, %72, p, 1, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n112k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55}, {%60, %61, %62, %63}, %73, 1, 1, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n112k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55}, {%64, %65, %66, %67}, %74, 1, 1, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n112k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55}, {%68, %69, %70, %71}, %75, 1, 1, 1, 0;\n"
                 "wgmma.commit_group.sync.aligned;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55])
                 : "r"(a[0][0]), "r"(a[0][1]), "r"(a[0][2]), "r"(a[0][3]), "r"(a[1][0]), "r"(a[1][1]), "r"(a[1][2]), "r"(a[1][3]), "r"(a[2][0]), "r"(a[2][1]), "r"(a[2][2]), "r"(a[2][3]), "r"(a[3][0]), "r"(a[3][1]), "r"(a[3][2]), "r"(a[3][3]), "l"(db[0]), "l"(db[1]), "l"(db[2]), "l"(db[3]), "r"(acc));
}
__device__ __forceinline__ void wgmma4_rs(float (&d)[64], const uint32_t (&a)[4][4], const uint64_t (&db)[4], int acc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %84, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %80, p, 1, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%68, %69, %70, %71}, %81, 1, 1, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%72, %73, %74, %75}, %82, 1, 1, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%76, %77, %78, %79}, %83, 1, 1, 1, 0;\n"
                 "wgmma.commit_group.sync.aligned;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
                 : "r"(a[0][0]), "r"(a[0][1]), "r"(a[0][2]), "r"(a[0][3]), "r"(a[1][0]), "r"(a[1][1]), "r"(a[1][2]), "r"(a[1][3]), "r"(a[2][0]), "r"(a[2][1]), "r"(a[2][2]), "r"(a[2][3]), "r"(a[3][0]), "r"(a[3][1]), "r"(a[3][2]), "r"(a[3][3]), "l"(db[0]), "l"(db[1]), "l"(db[2]), "l"(db[3]), "r"(acc));
}
// The same 4 products with A (W's 64 rows by 16 columns each) from shared memory too, by descriptor (da).
__device__ __forceinline__ void wgmma4_ss(float (&d)[8], const uint64_t (&da)[4], const uint64_t (&db)[4], int acc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %16, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %12, p, 1, 1, 0, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %9, %13, 1, 1, 1, 0, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %10, %14, 1, 1, 1, 0, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %11, %15, 1, 1, 1, 0, 0;\n"
                 "wgmma.commit_group.sync.aligned;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                 : "l"(da[0]), "l"(da[1]), "l"(da[2]), "l"(da[3]), "l"(db[0]), "l"(db[1]), "l"(db[2]), "l"(db[3]), "r"(acc));
}
__device__ __forceinline__ void wgmma4_ss(float (&d)[16], const uint64_t (&da)[4], const uint64_t (&db)[4], int acc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %24, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %20, p, 1, 1, 0, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %17, %21, 1, 1, 1, 0, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %18, %22, 1, 1, 1, 0, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %19, %23, 1, 1, 1, 0, 0;\n"
                 "wgmma.commit_group.sync.aligned;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
                 : "l"(da[0]), "l"(da[1]), "l"(da[2]), "l"(da[3]), "l"(db[0]), "l"(db[1]), "l"(db[2]), "l"(db[3]), "r"(acc));
}
__device__ __forceinline__ void wgmma4_ss(float (&d)[32], const uint64_t (&da)[4], const uint64_t (&db)[4], int acc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %40, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %36, p, 1, 1, 0, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %33, %37, 1, 1, 1, 0, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %34, %38, 1, 1, 1, 0, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %35, %39, 1, 1, 1, 0, 0;\n"
                 "wgmma.commit_group.sync.aligned;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
                 : "l"(da[0]), "l"(da[1]), "l"(da[2]), "l"(da[3]), "l"(db[0]), "l"(db[1]), "l"(db[2]), "l"(db[3]), "r"(acc));
}
#endif

// Shared by the TMA kernel and mma12_mid_kernel. A warp's 16 rows (16w to 16w + 15) of a row block's 4
// steps (a stage), from shared memory into mma's A fragments, a register set a step: of each lane, the
// codes' word w and the low bytes 8 (w mod 2) to 8 (w mod 2) + 7 of half w / 2 (its rows g and g + 8);
// then the stage's run of exceptions, one run for the 4 steps (se: its copy, whose first entry is ea; ea <
// 0: read from global memory; eb: exc_base at the 4 steps and the next). The lanes take the run's entries
// 32 at a time, each in this warp's rows setting its byte in the warp's scratch (xw: a lane's 8 words, 256
// in all, zero between stages), which every lane then XORs into its high bytes and clears: a stage's cost
// grows by a pass per 32 of its exceptions, not by an entry per lane (a few layers' matrices have dozens a
// stage; past the stage's copy, 256 entries in the TMA kernel and 128 in mma12_mid_kernel, they are read
// from global memory).
__device__ __forceinline__ void decode12_rows(const Nib& f, const uint8_t* sp, const uint32_t* se, const int (&eb)[5], int ea, int lane, int w, uint32_t* xw, uint32_t (&A)[4][4]) {
    uint32_t nw[4];
    uint2 sw[4];
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {  // the stage's loads first, 8 in flight at once
        const uint8_t* q = sp + kk * STEP12;
        nw[kk] = *(const uint32_t*)(q + 16 * lane + 4 * w);
        sw[kk] = *(const uint2*)(q + 512 + 512 * (w >> 1) + 16 * lane + 8 * (w & 1));
    }
    uint32_t ew[8];  // step kk's high-byte words 2w and 2w + 1 of each lane: ew[2kk], ew[2kk + 1]
#pragma unroll
    for (int q = 0; q < 8; q++) ew[q] = Nib::high1(nw[q >> 1], f.hb4, q & 1);
    // Entry k is step kk's while eb[kk] <= k < eb[kk + 1]; weight i of lane (x >> 5) mod 32, byte i mod 4 of its
    // word 2 kk + (i / 4 mod 2) here where i / 8 is this warp's.
    if (eb[4] > eb[0]) {
        bool mine = false;
        for (int k = eb[0] + lane; k < eb[4]; k += 32) {
            uint32_t x = ea >= 0 ? se[k - ea] : __ldg(f.exc + k), i = x & 31;
            if ((int)(i >> 3) == w) {
                uint32_t kk = (k >= eb[1]) + (k >= eb[2]) + (k >= eb[3]);
                ((uint8_t*)xw)[(((x >> 5) & 31) * 8 + 2 * kk + ((i >> 2) & 1)) * 4 + (i & 3)] = (uint8_t)(x >> 16);
                mine = true;
            }
        }
        if (__any_sync(FULL, mine)) {
            __syncwarp();
            uint4* xl = (uint4*)(xw + 8 * lane);
            uint4 a = xl[0], b = xl[1];
            ew[0] ^= a.x, ew[1] ^= a.y, ew[2] ^= a.z, ew[3] ^= a.w, ew[4] ^= b.x, ew[5] ^= b.y, ew[6] ^= b.z, ew[7] ^= b.w;
            xl[0] = xl[1] = make_uint4(0u, 0u, 0u, 0u);
            __syncwarp();
        }
    }
    // A fragments: rows g and g + 8 (words 2w and 2w + 1), columns 2t and 8 + 2t (bytes 0-1 and 2-3).
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
        A[kk][0] = __byte_perm(sw[kk].x, ew[2 * kk], 0x5140);
        A[kk][1] = __byte_perm(sw[kk].y, ew[2 * kk + 1], 0x5140);
        A[kk][2] = __byte_perm(sw[kk].x, ew[2 * kk], 0x7362);
        A[kk][3] = __byte_perm(sw[kk].y, ew[2 * kk + 1], 0x7362);
    }
}

// exc_base at a stage's 4 steps and the next, for each of a unit's WG row blocks.
template <int WG>
__device__ __forceinline__ void stage_bounds(const Nib& f, int64_t RB, int64_t KS, int p, int s, int (&e)[5 * WG]) {
#pragma unroll
    for (int r = 0; r < WG; r++)
#pragma unroll
        for (int i = 0; i < 5; i++) e[5 * r + i] = __ldg(f.exc_base + min((int64_t)WG * p + r, RB - 1) * KS + 4 * s + i);
}

// A row block's exceptions for a stage as a copy: na entries from entry a (a multiple of 4), the run
// widened to 16 bytes each way (pack_mma12 pads exc); na = -1 past cap bytes (read from global memory then).
__device__ __forceinline__ void exc_copy(int lo, int hi, int cap, int& a, int& na) {
    a = lo & ~3;
    na = hi > lo ? ((hi + 3) & ~3) - a : 0;
    if (4 * na > cap) na = -1;
}

// A unit's sum out (its WG row blocks, row unit pr, NT tokens m0 on), r in each thread's fragment order (rows
// 16w + g (+ 8) of row block wg, tokens 8jn + 2t (+ 1): r[4jn + 2h + c]): to Y when the block covers the unit,
// else to its slot, and the last of the unit's blocks to finish adds the slots in block order (the same result
// every run). A block's slots: 2b for its first unit, 2b + 1 for its last (only those two may be shared). The WG
// warpgroups' threads (ct), named barrier 1.
template <int NT, int WG>
__device__ __forceinline__ void sum_out12(float (&r)[NT / 2], int64_t p, int64_t pr, int64_t m0, int S, int64_t nb, int64_t U, float* parts, int* done, int& last, int ct, int64_t O, int64_t M, const __nv_bfloat16* bias, __nv_bfloat16* Y) {
    constexpr int R = 64 * WG;
    int wg = ct >> 7, w = (ct >> 5) & 3, g = (ct & 31) >> 2, t = ct & 3;
    int64_t first = block_of_step(p * S, nb, U), fin = block_of_step(p * S + S - 1, nb, U);
    // Block b's slot for the unit: 2b where the unit is b's first (b starts in it: every block of it but the first,
    // which starts in it only at its start), else 2b + 1.
    auto slot = [&](int64_t b) { return parts + (2 * b + (b == first && b * U / nb != p * S)) * (NT * R); };
    if (first != fin) {
        // Slots [NT / 8 float4s][threads]: each thread's r in order, coalesced.
        float4* pp = (float4*)slot(blockIdx.x);
#pragma unroll
        for (int q = 0; q < NT / 8; q++) pp[q * 128 * WG + ct] = make_float4(r[4 * q], r[4 * q + 1], r[4 * q + 2], r[4 * q + 3]);
        // The warpgroups' slot written (the barrier), one thread's count with release and acquire: the others' parts
        // seen by the last, whose threads read them after the second barrier (fences are cumulative; a fence a thread
        // had held each up till its own writes landed).
        bar_sync<128 * WG>(1);
        if (ct == 0) {
            int before = atom_inc_acq_rel(done + p);
            last = before == fin - first;
            if (last) done[p] = 0;  // ready for the next product
        }
        bar_sync<128 * WG>(1);
        if (last) {
#pragma unroll
            for (int i = 0; i < NT / 2; i++) r[i] = 0.f;
            for (int64_t b = first; b <= fin; b++) {
                const float4* bp = (const float4*)slot(b);
#pragma unroll
                for (int q = 0; q < NT / 8; q++) {
                    float4 v = __ldcg(bp + q * 128 * WG + ct);
                    r[4 * q] += v.x;
                    r[4 * q + 1] += v.y;
                    r[4 * q + 2] += v.z;
                    r[4 * q + 3] += v.w;
                }
            }
        }
    }
    if (first == fin || last) {
#pragma unroll
        for (int jn = 0; jn < NT / 8; jn++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                int64_t o = ((int64_t)WG * pr + wg) * 64 + 16 * w + g + 8 * h;
                float bo = bias && o < O ? __bfloat162float(bias[o]) : 0.f;
#pragma unroll
                for (int c = 0; c < 2; c++) {
                    int64_t m = m0 + 8 * jn + 2 * t + c;
                    if (o < O && m < M) Y[m * O + o] = __float2bfloat16(r[4 * jn + 2 * h + c] + bo);
                }
            }
    }
}

template <int NT, int WG> struct Tma12 {
    static constexpr int THREADS = 128 * WG + 32;  // WG consumer warpgroups (a warpgroup's first warp a multiple of 4), the TMA warp
    static constexpr int R = 64 * WG;              // W's rows a unit (a row block a warpgroup)
    static constexpr int XB = NT * 128;            // X's tile a stage: NT tokens by 64 columns
    static constexpr int CB = 4 * (int)STEP12, EB = 1024, BB = 32;  // a row block's steps a stage, its exceptions (up to 256), their bounds
    static constexpr int RBB = CB + EB + BB;
    static constexpr int SLOT = (XB + WG * RBB + 1023) / 1024 * 1024;
    static constexpr int NS = 200 * 1024 / SLOT < 8 ? 200 * 1024 / SLOT : 8;  // stages in flight
    static constexpr int SHARED = NS * SLOT + 1024 + 16 * NS + 4096 * WG;   // alignment, the full and empty barriers, the warps' exceptions' scratch
};

template <int NT, int WG>
__device__ __forceinline__ void mma12_tma_body(const CUtensorMap& xmap, Nib f, int64_t O, int64_t K, int64_t M, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ Y, float* __restrict__ parts, int* __restrict__ done) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    using C = Tma12<NT, WG>;
    // (aligned as the MID and WS bodies declare it, 128: the base is aligned to 1024 by hand below, in the 1024 SHARED leaves for it; a 1024 here pads the
    // block's static memory up to 1024, and tiered_tma's 64-token tile would ask 256 bytes past an H100's 227 KB)
    extern __shared__ __align__(128) uint8_t smem_raw[];  // NS slots [X tile | a row block's steps, exceptions, bounds | the other's], the barriers
    __shared__ int last;
    uint32_t raw = (uint32_t)__cvta_generic_to_shared(smem_raw), base = (raw + 1023) & ~1023u;
    uint8_t* gbase = smem_raw + (base - raw);
    uint32_t full = base + C::NS * C::SLOT, empty = full + 8 * C::NS;
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    // Units: RU row units (WG row blocks) by TC chunks of NT tokens, unit p chunk p / RU and row unit p mod RU; the
    // work, units by stages, split evenly over the blocks. Chunk by chunk: the blocks at work at once share few row
    // units, a chunk's blocks each of theirs at about the same time (read from memory about once, not once a chunk).
    int TC = (int)((M + NT - 1) / NT), RU = (int)((O / 64 + WG - 1) / WG);
    int64_t KS = K / 16, RB = O / 64, U = (int64_t)RU * TC * (K / 64), nb = gridDim.x;
    int S = (int)(K / 64);
    int64_t u0 = blockIdx.x * U / nb;
    int n = (int)((blockIdx.x + 1) * U / nb - u0);  // the block's stages, u0 on: unit u / S, stage u mod S
    int p0 = (int)(u0 / S), s0 = (int)(u0 - (int64_t)p0 * S);
    if (tid == 0) {
        for (int i = 0; i < C::NS; i++) {
            mbar_init(full + 8 * i, 1);
            mbar_init(empty + 8 * i, 4 * WG);
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();
    if (warp >= 4 * WG) {
        // The TMA warp: lane j mod 8 issues stage j. Stages go in batches of 8,
        // lane l's bounds for stage l of the batch loaded while the batch before
        // was issued (a load pends for the whole warp: one register set a lane,
        // reloaded by one lane a stage, would wait out every load).
        int e[5 * WG], en[5 * WG];  // exc_base at the stage's 4 steps and the next, each row block: this batch's, the next's
        auto ahead = [&](int p, int s, int k) {  // the stage k after (p, s)
            s += k;
            stage_bounds<WG>(f, RB, KS, (p + s / S) % RU, s % S, en);
        };
        if (lane < 8 && lane < n) ahead(p0, s0, lane);
        // (unit p's row unit pr and chunk pc kept as p goes: no division a stage, where a stage is short)
        for (int j = 0, p = p0, s = s0, pr = p0 % RU, pc = p0 / RU; j < n; j++) {
            if ((j & 7) == 0) {
#pragma unroll
                for (int i = 0; i < 5 * WG; i++) e[i] = en[i];
                if (lane < 8 && j + 8 + lane < n) ahead(p, s, 8 + lane);
            }
            if (lane == (j & 7)) {
                int sl = j % C::NS;
                uint32_t fb = full + 8 * sl, xs = base + sl * C::SLOT;
                if (j >= C::NS) mbar_wait(empty + 8 * sl, (j / C::NS - 1) & 1);  // the consumers are done with stage j - NS
                uint32_t bytes = C::XB + WG * C::CB;
                int a[WG], na[WG];
#pragma unroll
                for (int r = 0; r < WG; r++) {
                    exc_copy(e[5 * r], e[5 * r + 4], C::EB, a[r], na[r]);
                    bytes += na[r] > 0 ? 4 * na[r] : 0;
                    int* bs = (int*)(gbase + sl * C::SLOT + C::XB + r * C::RBB + C::CB + C::EB);
#pragma unroll
                    for (int i = 0; i < 5; i++) bs[i] = e[5 * r + i];
                    bs[5] = na[r] < 0 ? -1 : a[r];
                }
                mbar_expect_tx(fb, bytes);
                tma_2d(xs, &xmap, (int)(s * 64), pc * NT, fb);
#pragma unroll
                for (int r = 0; r < WG; r++) {
                    uint32_t cs = xs + C::XB + r * C::RBB;
                    bulk_g2s(cs, f.data + (min((int64_t)WG * pr + r, RB - 1) * KS + 4 * s) * STEP12, C::CB, fb, TC == 1);  // (read once but where the chunks share it)
                    if (na[r] > 0) bulk_g2s(cs + C::CB, f.exc + a[r], 4 * na[r], fb);
                }
            }
            __syncwarp();
            if (++s == S) {
                s = 0, p++;
                if (++pr == RU) pr = 0, pc++;
            }
        }
        return;
    }
    // Consumer warpgroup wg: row block WG (p mod RU) + wg of each unit p; warp w its rows 16w to 16w + 15.
    // Two sets of A registers, used in turn: a stage's products may still run while the next decodes.
    int ct = tid, wg = ct >> 7, w = (ct >> 5) & 3;
    uint32_t* xw = (uint32_t*)(gbase + C::NS * C::SLOT + 16 * C::NS) + 256 * warp;  // this warp's exceptions' scratch
    ((uint4*)xw)[2 * lane] = ((uint4*)xw)[2 * lane + 1] = make_uint4(0u, 0u, 0u, 0u);
    __syncwarp();
    float d[NT / 2] = {};  // rows 16w + g (+ 8), tokens 8jn + 2t (+ 1): d[4jn + 2h + c]; set by a unit's first product
    // (zeroed all the same: left unset, CUDA 12.8's ptxas saw it defined inside the loop and serialized every wgmma,
    // its C7515, where CUDA 13's did not; on an H100 SXM the CUDA 12 library's products took a median 4% longer so)
    uint32_t A0[4][4], A1[4][4];
    int held = -1;  // the slot whose products may still be running
    auto release = [&](int sl) {  // the slot is free for stage j + NS (lane 0 arrives, by predicate: no branch among the products)
        __syncwarp();
        asm volatile("{\n.reg .pred p;\nsetp.eq.u32 p, %1, 0;\n@p mbarrier.arrive.shared::cta.b64 _, [%0];\n}\n" ::"r"(empty + 8 * sl), "r"(lane) : "memory");
    };
    auto keep_d = [&]() {  // d stays in its registers while the products run
#pragma unroll
        for (int i = 0; i < NT / 2; i++) asm volatile("" : "+f"(d[i])::"memory");
    };
    auto keep = [&](uint32_t(&A)[4][4]) {  // A is held until its products are done
#pragma unroll
        for (int kk = 0; kk < 4; kk++)
#pragma unroll
            for (int i = 0; i < 4; i++) asm volatile("" : "+r"(A[kk][i])::"memory");
    };
    auto stage = [&](int j, bool first, uint32_t(&A)[4][4], uint32_t(&Aprev)[4][4]) {
        int sl = j % C::NS;
        mbar_wait(full + 8 * sl, (j / C::NS) & 1);
        const uint8_t* sp = gbase + sl * C::SLOT + C::XB + wg * C::RBB;
        const int* bs = (const int*)(sp + C::CB + C::EB);
        int eb[5];
#pragma unroll
        for (int i = 0; i < 5; i++) eb[i] = bs[i];
        decode12_rows(f, sp, (const uint32_t*)(sp + C::CB), eb, bs[5], lane, w, xw, A);
        uint32_t xs = base + sl * C::SLOT;
        uint64_t desc[4];
#pragma unroll
        for (int kk = 0; kk < 4; kk++) desc[kk] = sw128_desc(xs + kk * 32);
        keep_d();
        wg_fence();
        wgmma4_rs(d, A, desc, !first);
        wg_wait<1>();  // the stage before's products are done: its A set and its slot are free
        keep(Aprev);
        keep_d();
        if (held >= 0) release(held);
        held = sl;
    };
    // A unit's stages, then its sum out: no branch touches d while products run. The stages in pairs, an odd
    // one after them: every path gives each set of A registers to the products before it is decoded into
    // again (a pair's second stage skipped inside the loop had ptxas serialize every product, its C7513).
    for (int j = 0, p = p0, s = s0; j < n; p++, s = 0) {
        int len = min(S - s, n - j), k = 0;
        for (; k + 1 < len; k += 2) {
            stage(j + k, k == 0, A0, A1);
            stage(j + k + 1, false, A1, A0);
        }
        if (k < len) stage(j + k, k == 0, A0, A1);
        wg_wait<0>();
        keep(A0);
        keep(A1);
        keep_d();
        if (held >= 0) release(held);
        held = -1;
        sum_out12<NT, WG>(d, p, p % RU, (int64_t)(p / RU) * NT, S, nb, U, parts, done, last, ct, O, M, bias, Y);  // (d, summed in place: the next unit's first product sets it)
        j += len;
    }
#elif defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    __trap();  // sm_90a code, launched from a build without it (or on AMD)
#endif
}

template <int NT, int WG>
__global__ void __launch_bounds__((Tma12<NT, WG>::THREADS), 1) mma12_tma_kernel(const __grid_constant__ CUtensorMap xmap, Nib f, int64_t O, int64_t K, int64_t M, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ Y, float* __restrict__ parts, int* __restrict__ done) {
    mma12_tma_body<NT, WG>(xmap, f, O, K, M, bias, Y, parts, done);
}

// The tiered layout's products of 1 to 64 tokens on Hopper (sm_90a), by the TMA kernel's plan (mma12_tma_kernel). A
// TMA warp copies each stage (64 columns) into a ring of NS slots, landing on an mbarrier: X's tile of NT tokens by its
// tensor map (128-byte swizzle) and, for each of the WG row blocks, its 4 steps' fixed parts (5,120 bytes), the bytes of
// their blocks from 128 before the first one's end to 128 past the last one's start (all that the short form of a
// step's decode reads: its window and its tail), up to BB of them (more: read from global memory instead), and
// block_base at the 4 steps and the next. Consumer warpgroups, a row block each: warp kk decodes step kk (64 rows by 16
// columns) from shared memory (a step with the flag by the general decode, its block from global memory), the 4 warps
// trade the decoded fragments through shared memory (warp w then holds rows 16w to 16w + 15 of all 4 steps: wgmma's A
// registers), and wgmma multiplies them by X's tile. A weight is decoded once for NT tokens, and no global load of the
// decode waits in a consumer: the TMA warp makes them stages ahead. BPS: blocks an SM (the registers' bound). SS: the
// warps put their steps' fragments as one tile of the stage's 64 rows by 64 columns in the 128-byte swizzle of X's
// tile, and wgmma reads A from it as it reads X (no A registers); else the trade (A in registers).
constexpr int TT_BB = 2048;  // a row block's blocks' bytes a stage, at most, in shared memory
template <int NT, int WG, int BPS> struct TmaT {
    static constexpr int THREADS = 128 * WG + 32;  // WG consumer warpgroups, the TMA warp
    static constexpr int R = 64 * WG;              // W's rows a unit (a row block a warpgroup)
    static constexpr int XB = NT * 128;            // X's tile a stage: NT tokens by 64 columns
    static constexpr int CB = 4 * (int)STEP_BYTES, BB = TT_BB, HB = 32;  // a row block's 4 steps' fixed parts, their blocks' bytes, the bounds
    static constexpr int RBB = CB + BB + HB;
    static constexpr int SLOT = (XB + WG * RBB + 1023) / 1024 * 1024;
    static constexpr int XCH = 2 * 16 * 32 * 16;                       // a warpgroup's trade buffers: two (a stage's parity) of [4 steps][4 row groups][32 lanes] uint4, or of the tile [64 rows][128 bytes]
    static constexpr int FIXED = 1024 + WG * XCH + 4 * WG * S2_BYTES;  // the alignment, the trade buffers, the warps' scratch
    static constexpr int ROOM = 228 * 1024 / BPS - 2304;  // (an SM's 228 KB, less a block's 1 KB and its static tab)
    static constexpr int NS0 = (ROOM - FIXED) / (SLOT + 16);
    static constexpr int NS = NS0 < 8 ? NS0 : 8;  // stages in flight
    static constexpr int SHARED = NS * SLOT + 16 * NS + FIXED;
    static_assert(NS >= 2, "a ring of at least two stages");
};

// block_base at a stage's 4 steps and the next, for each of a unit's WG row blocks (bit 31: a step's flag).
template <int WG>
__device__ __forceinline__ void tiered_bounds(const Tiered& f, int64_t RB, int64_t KS, int p, int s, int (&e)[5 * WG]) {
#pragma unroll
    for (int r = 0; r < WG; r++)
#pragma unroll
        for (int i = 0; i < 5; i++) e[5 * r + i] = __ldg(f.block_base + min((int64_t)WG * p + r, RB - 1) * KS + 4 * s + i);
}

// Step w of a stage's row block from its copy in shared memory (sp: the 4 steps' fixed parts, their blocks' bytes from
// bs[5] on, the bounds bs) into a lane's B fragments of its 8 n-tiles, as the step kernel decodes it: the short form, a
// step with the flag by the general decode. bs[5] < 0: the blocks' bytes were not copied (read from global memory).
__device__ __forceinline__ void tiered_step_smem(const Tiered& f, const uint8_t* sp, int w, int lane, uint32_t* s2, const uint32_t* tab, uint32_t (&R)[16]) {
    constexpr int CB = 4 * (int)STEP_BYTES;
    const int* bs = (const int*)(sp + CB + TT_BB);
    const uint8_t* q = sp + w * STEP_BYTES;
    StepH st;
    st.pd[0] = ((const uint32_t*)q)[lane];
    st.pd[1] = ((const uint32_t*)q)[32 + lane];
    uint4 x0 = ((const uint4*)(q + 256))[lane], x1 = ((const uint4*)(q + 256))[32 + lane];
    st.sw[0] = x0.x; st.sw[1] = x0.y; st.sw[2] = x0.z; st.sw[3] = x0.w;
    st.sw[4] = x1.x; st.sw[5] = x1.y; st.sw[6] = x1.z; st.sw[7] = x1.w;
    st.rb = (uint32_t)bs[w];
    st.re = (uint32_t)bs[w + 1];
    const int lo = bs[5];
    const uint32_t b = (st.rb & 0x7FFFFFFFu) >> 2, e = (st.re & 0x7FFFFFFFu) >> 2;
    if (lo >= 0) {  // (blocks' byte i at sp + CB + i - lo)
        const uint8_t* wb = sp + CB;
        st.win = *(const uint32_t*)(wb + (int)(4u * (b + lane) - (uint32_t)lo));
        st.tail = *(const uint32_t*)(wb + (int)(4u * (e - 1 - lane) - (uint32_t)lo));
        st.tail1 = *(const uint32_t*)(wb + (int)(4u * e - (uint32_t)lo));
    } else {
        const uint32_t* bw = (const uint32_t*)f.blocks;
        st.win = __ldg(bw + b + lane);
        st.tail = __ldg(bw + e - 1 - lane);
        st.tail1 = __ldg(bw + e);
    }
    uint32_t ew[8];
    if (st.rb >> 31) exponents(step_of(st), f.ts, f.blocks, lane, s2, tab, ew);  // (inline: a call would serialize every wgmma)
    else exponents_h(st, f.ts, lane, s2, tab, ew);
    pairs(st.sw, ew, R);
}

template <int NT, int WG, int BPS, bool SS>
__device__ __forceinline__ void tiered_tma_body(const CUtensorMap& xmap, Tiered f, int64_t O, int64_t K, int64_t M, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ Y, float* __restrict__ parts, int* __restrict__ done, int bb) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    using C = TmaT<NT, WG, BPS>;
    extern __shared__ __align__(128) uint8_t smem_raw[];  // NS slots [X tile | a row block's steps, blocks' bytes, bounds | the others'], the trade buffers, the barriers, the scratch
    __shared__ int last;
    __shared__ uint32_t tab[256];
    uint32_t raw = (uint32_t)__cvta_generic_to_shared(smem_raw), base = (raw + 1023) & ~1023u;
    uint8_t* gbase = smem_raw + (base - raw);
    uint8_t* more = gbase + C::NS * C::SLOT;  // (1024-byte aligned: the tiles' swizzle)
    uint32_t full = base + C::NS * C::SLOT + WG * C::XCH, empty = full + 8 * C::NS;
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    // Units, stages and their split over the blocks as mma12_tma_kernel's.
    int TC = (int)((M + NT - 1) / NT), RU = (int)((O / 64 + WG - 1) / WG);
    int64_t KS = K / 16, RB = O / 64, U = (int64_t)RU * TC * (K / 64), nb = gridDim.x;
    int S = (int)(K / 64);
    int64_t u0 = blockIdx.x * U / nb;
    int n = (int)((blockIdx.x + 1) * U / nb - u0);
    int p0 = (int)(u0 / S), s0 = (int)(u0 - (int64_t)p0 * S);
    fill_groups(tab);
    if (tid == 0) {
        for (int i = 0; i < C::NS; i++) {
            mbar_init(full + 8 * i, 1);
            mbar_init(empty + 8 * i, 4 * WG);
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();
    if (warp >= 4 * WG) {
        // The TMA warp, as mma12_tma_kernel's: lane j mod 8 issues stage j, the bounds of a batch of 8 loaded while the
        // batch before is issued.
        int e[5 * WG], en[5 * WG];
        auto ahead = [&](int p, int s, int k) {
            s += k;
            tiered_bounds<WG>(f, RB, KS, (p + s / S) % RU, s % S, en);
        };
        if (lane < 8 && lane < n) ahead(p0, s0, lane);
        for (int j = 0, p = p0, s = s0, pr = p0 % RU, pc = p0 / RU; j < n; j++) {
            if ((j & 7) == 0) {
#pragma unroll
                for (int i = 0; i < 5 * WG; i++) e[i] = en[i];
                if (lane < 8 && j + 8 + lane < n) ahead(p, s, 8 + lane);
            }
            if (lane == (j & 7)) {
                int sl = j % C::NS;
                uint32_t fb = full + 8 * sl, xs = base + sl * C::SLOT;
                if (j >= C::NS) mbar_wait(empty + 8 * sl, (j / C::NS - 1) & 1);  // the consumers are done with stage j - NS
                uint32_t bytes = C::XB + WG * C::CB;
                int lo[WG], nw[WG];
#pragma unroll
                for (int r = 0; r < WG; r++) {
                    // The 4 steps' blocks' bytes as their decode reads them: tails from 128 before the first one's end,
                    // windows to 128 past the last one's start (blocks hold 128 bytes before the first and 256 after the last).
                    uint32_t b0 = (uint32_t)e[5 * r] & 0x7FFFFFFFu, e0 = (uint32_t)e[5 * r + 1] & 0x7FFFFFFFu;
                    uint32_t b3 = (uint32_t)e[5 * r + 3] & 0x7FFFFFFFu, e3 = (uint32_t)e[5 * r + 4] & 0x7FFFFFFFu;
                    uint32_t l = min(b0 & ~3u, (e0 & ~3u) - 128u) & ~15u, h = (max((b3 & ~3u) + 128u, (e3 & ~3u) + 4u) + 15u) & ~15u;
                    lo[r] = (int)l;
                    nw[r] = h - l <= (uint32_t)bb ? (int)(h - l) : -1;  // (bb: BB, or less in the checks)
                    bytes += nw[r] > 0 ? (uint32_t)nw[r] : 0u;
                    int* bs = (int*)(gbase + sl * C::SLOT + C::XB + r * C::RBB + C::CB + C::BB);
#pragma unroll
                    for (int i = 0; i < 5; i++) bs[i] = e[5 * r + i];
                    bs[5] = nw[r] < 0 ? -1 : lo[r];
                }
                mbar_expect_tx(fb, bytes);
                tma_2d(xs, &xmap, (int)(s * 64), pc * NT, fb);
#pragma unroll
                for (int r = 0; r < WG; r++) {
                    uint32_t cs = xs + C::XB + r * C::RBB;
                    int64_t rbk = min((int64_t)WG * pr + r, RB - 1);
                    bulk_g2s(cs, f.data + (rbk * KS + 4 * s) * STEP_BYTES, C::CB, fb, TC == 1);  // (read once but where the chunks share it)
                    if (nw[r] > 0) bulk_g2s(cs + C::CB, f.blocks + lo[r], (uint32_t)nw[r], fb, TC == 1);
                }
            }
            __syncwarp();
            if (++s == S) {
                s = 0, p++;
                if (++pr == RU) pr = 0, pc++;
            }
        }
        return;
    }
    // Consumer warpgroup wg: row block WG (p mod RU) + wg of each unit p; warp w decodes step w of each stage and then
    // holds rows 16w to 16w + 15 of the 4. Two sets of A registers, used in turn, as mma12_tma_kernel's.
    int ct = tid, wg = ct >> 7, w = (ct >> 5) & 3;
    uint4* trade = (uint4*)(more + wg * C::XCH);
    uint32_t* s2 = (uint32_t*)(more + WG * C::XCH + 16 * C::NS) + warp * (S2_BYTES / 4);
    float d[NT / 2];  // rows 16w + g (+ 8), tokens 8jn + 2t (+ 1): d[4jn + 2h + c]; set by a unit's first product
#pragma unroll
    for (int i = 0; i < NT / 2; i++) d[i] = 0.f;  // (zeroed all the same, as mma12_tma_kernel's)
    uint32_t A0[4][4], A1[4][4];
    int held = -1;  // the slot whose products may still be running
    auto release = [&](int sl) {
        __syncwarp();
        asm volatile("{\n.reg .pred p;\nsetp.eq.u32 p, %1, 0;\n@p mbarrier.arrive.shared::cta.b64 _, [%0];\n}\n" ::"r"(empty + 8 * sl), "r"(lane) : "memory");
    };
    auto keep_d = [&]() {
#pragma unroll
        for (int i = 0; i < NT / 2; i++) asm volatile("" : "+f"(d[i])::"memory");
    };
    auto keep = [&](uint32_t(&A)[4][4]) {
#pragma unroll
        for (int kk = 0; kk < 4; kk++)
#pragma unroll
            for (int i = 0; i < 4; i++) asm volatile("" : "+r"(A[kk][i])::"memory");
    };
    auto stage = [&](int j, bool first, uint32_t(&A)[4][4], uint32_t(&Aprev)[4][4]) {
        int sl = j % C::NS;
        mbar_wait(full + 8 * sl, (j / C::NS) & 1);
        uint32_t R[16];
        tiered_step_smem(f, gbase + sl * C::SLOT + C::XB + wg * C::RBB, w, lane, s2, tab, R);
        // The trade: this warp's step, rows 16q to 16q + 15 as A fragments (rows g, g + 8 by columns 2t, 2t + 8: n-tiles
        // 2q and 2q + 1) to [w][q]; then [kk][w] of each step kk, once the warpgroup's 4 have put theirs. (Two buffers: a
        // warp puts stage j + 2's after the barrier of j + 1, which every warp passes after taking stage j's.)
        uint4* tb = trade + (j & 1) * 512;
        uint32_t xs = base + sl * C::SLOT;
        uint64_t desc[4];
#pragma unroll
        for (int kk = 0; kk < 4; kk++) desc[kk] = sw128_desc(xs + kk * 32);
        if constexpr (SS) {
            // Row 8n + g's columns 16w + 2t (n-tile n's R[2n]) and 16w + 8 + 2t (R[2n + 1]): 16-byte chunks 2w and 2w + 1 of
            // its 128 bytes, at chunk c ^ (row mod 8) (the 128-byte swizzle; a store of the warp's: 8 rows' 16 bytes, one
            // bank a lane). Then the tile to the tensor cores' proxy, and the warpgroup's 4 steps in it.
            uint8_t* at = (uint8_t*)tb + 4 * (lane & 3);
            const uint32_t g = (uint32_t)lane >> 2, c0 = ((2u * w) ^ g) << 4, c1 = ((2u * w + 1) ^ g) << 4;
#pragma unroll
            for (int nn = 0; nn < 8; nn++) {
                uint8_t* row = at + (8 * nn + g) * 128;
                *(uint32_t*)(row + c0) = R[2 * nn];
                *(uint32_t*)(row + c1) = R[2 * nn + 1];
            }
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            bar_sync<128>(2 + wg);
            uint32_t ab = (uint32_t)__cvta_generic_to_shared(tb);
            uint64_t da[4];
#pragma unroll
            for (int kk = 0; kk < 4; kk++) da[kk] = sw128_desc(ab + kk * 32);
            keep_d();
            wg_fence();
            wgmma4_ss(d, da, desc, !first);
            wg_wait<1>();  // the stage before's products are done: its tile and its slot are free
            keep_d();
        } else {
#pragma unroll
            for (int q = 0; q < 4; q++) tb[(w * 4 + q) * 32 + lane] = make_uint4(R[4 * q], R[4 * q + 2], R[4 * q + 1], R[4 * q + 3]);
            bar_sync<128>(2 + wg);
#pragma unroll
            for (int kk = 0; kk < 4; kk++) {
                uint4 v = tb[(kk * 4 + w) * 32 + lane];
                A[kk][0] = v.x;
                A[kk][1] = v.y;
                A[kk][2] = v.z;
                A[kk][3] = v.w;
            }
            keep_d();
            wg_fence();
            wgmma4_rs(d, A, desc, !first);
            wg_wait<1>();  // the stage before's products are done: its A set and its slot are free
            keep(Aprev);
            keep_d();
        }
        if (held >= 0) release(held);
        held = sl;
    };
    for (int j = 0, p = p0, s = s0; j < n; p++, s = 0) {
        int len = min(S - s, n - j), k = 0;
        for (; k + 1 < len; k += 2) {
            stage(j + k, k == 0, A0, A1);
            stage(j + k + 1, false, A1, A0);
        }
        if (k < len) stage(j + k, k == 0, A0, A1);
        wg_wait<0>();
        if constexpr (!SS) {
            keep(A0);
            keep(A1);
        }
        keep_d();
        if (held >= 0) release(held);
        held = -1;
        sum_out12<NT, WG>(d, p, p % RU, (int64_t)(p / RU) * NT, S, nb, U, parts, done, last, ct, O, M, bias, Y);
        j += len;
    }
#elif defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    __trap();  // sm_90a code, launched from a build without it (or on AMD)
#endif
}

template <int NT, int WG, int BPS, bool SS>
__global__ void __launch_bounds__((TmaT<NT, WG, BPS>::THREADS), BPS) tiered_tma_kernel(const __grid_constant__ CUtensorMap xmap, Tiered f, int64_t O, int64_t K, int64_t M, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ Y, float* __restrict__ parts, int* __restrict__ done, int bb) {
    tiered_tma_body<NT, WG, BPS, SS>(xmap, f, O, K, M, bias, Y, parts, done, bb);
}

// Prompts on Hopper (sm_90a) past 128 tokens, the 12-bit layout: blocks that stay (one an SM, taking tile after tile),
// warp-specialized as the mixed-input GEMMs for Hopper are (CUTLASS 3.x's, Machete's). Lane 0 of a warp of its own
// copies each stage (64 columns) by TMA into a ring of NS stages, each landing on an mbarrier: X's tile of NT tokens
// through a tensor map in wgmma's 128-byte swizzle, W's two row blocks' 4 steps still compact, and their exceptions.
// Two consumer warpgroups, a row block each (a tile 128 of W's rows by NT tokens), decode a step (a k-block of 16
// columns) at a time into wgmma's A registers, a register set a k-block, and hand it to wgmma (64 rows by NT tokens by
// 16 columns) while the two k-blocks before it still multiply: a weight decoded once for NT tokens. The blocks go in
// clusters of CL, a block a row unit (128 rows) of the cluster's tile (CL row units by a chunk of NT tokens), X's tile
// copied once for the cluster (each block copies its CL-th into every block's shared memory: TMA's multicast). Tiles,
// chunks fastest (the clusters at work at once read each row unit's weights from memory once), in whole waves are each
// a cluster's own, summed in its registers and written out; the R left (fewer than a wave) are split by stages over up
// to 3 R clusters (all of them where the tiles are fewer than the clusters): stream-K, a tile covered by several summed
// by the last of them to finish, in cluster order. What bounds it (an H100 SXM, builds for timing alone): with nothing
// decoded (A a constant) Qwen3-8B's gate_up and o took 1.09 / 1.22x cuBLAS's time at 4096 tokens (1.19 / 1.38x with
// each block copying X, in 4 stages); the decode's integer instructions then add about their issue time (gate_up, with
// the stage's loads and exceptions, 0 / 11 / 21 of them a k-block: 1228 / 1310 / 1508 us), no less with some of them
// on the multiply pipe instead or decoded while the other warpgroup's products run: a weight decoded once for 256
// tokens (the most sums a warpgroup's registers hold) costs the tensor cores a quarter to a third more time. So the
// fewest instructions a stage: the exceptions in one pass (4 stages of 256 tokens then fit, not 5, which with a pass
// a step were the slower).
template <int NT, int CL> struct Wgp12 {
    static constexpr int THREADS = 384;  // two consumer warpgroups, then the producer warpgroup (its first warp the TMA's)
    static constexpr int CR = 232;       // a consumer's registers after setmaxnreg (the producers' 40): 64 NT / 128 its sums
    static constexpr int XB = NT * 128;  // X's tile a stage: NT tokens by 64 columns
    static constexpr int CB = 4 * (int)STEP12, EB = 512, BB = 32;  // a row block's steps a stage, its exceptions (up to 128), their bounds
    static constexpr int RBB = CB + EB + BB, WB = 2 * RBB;
    static constexpr int SCRATCH = 8 * 1024;  // the consumer warps' exceptions' scratch, 1 KB each
    // Stages in flight: the X tiles' ring (1024-byte aligned, as the swizzle wants), then W's, the barriers, the scratch.
    static constexpr int NS = (227 * 1024 - SCRATCH - 256) / (XB + WB) < 8 ? (227 * 1024 - SCRATCH - 256) / (XB + WB) : 8;
    static constexpr int SHARED = NS * (XB + WB) + 16 * NS + 16 + SCRATCH;
    static constexpr int SPLIT = 3;  // the split tiles' clusters, at most, for each (past whole waves)
};

// A consumer warp's stage (its 16 rows of a row block's 4 steps at sp, as decode12_rows reads them): the codes' word w
// and 8 low bytes of each step.
__device__ __forceinline__ void stage12(const uint8_t* sp, int lane, int w, uint32_t (&nw)[4], uint2 (&sw)[4]) {
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
        const uint8_t* q = sp + kk * STEP12;
        nw[kk] = *(const uint32_t*)(q + 16 * lane + 4 * w);
        sw[kk] = *(const uint2*)(q + 512 + 512 * (w >> 1) + 16 * lane + 8 * (w & 1));
    }
}

// A stage's exceptions in a warp's rows (bs: the run's bounds at the 4 steps and the next, then ea: the copy's first
// entry, -1 where read from global memory) as bytes to XOR into its high-byte words (xe[2kk], xe[2kk + 1]: step kk's),
// through the warp's scratch (xw: 8 words a lane, zero between stages), decode12_rows's plan: a pass per 32 entries.
__device__ __forceinline__ void stage12_exc(const Nib& f, const uint32_t* se, const int* bs, int lane, int w, uint32_t* xw, uint32_t (&xe)[8]) {
    int eb[5];
#pragma unroll
    for (int i = 0; i < 5; i++) eb[i] = bs[i];
#pragma unroll
    for (int i = 0; i < 8; i++) xe[i] = 0u;
    if (eb[4] > eb[0]) {
        int ea = bs[5];
        bool mine = false;
        for (int k = eb[0] + lane; k < eb[4]; k += 32) {
            uint32_t x = ea >= 0 ? se[k - ea] : __ldg(f.exc + k), i = x & 31;
            if ((int)(i >> 3) == w) {
                uint32_t kk = (k >= eb[1]) + (k >= eb[2]) + (k >= eb[3]);
                ((uint8_t*)xw)[(((x >> 5) & 31) * 8 + 2 * kk + ((i >> 2) & 1)) * 4 + (i & 3)] = (uint8_t)(x >> 16);
                mine = true;
            }
        }
        if (__any_sync(FULL, mine)) {
            __syncwarp();
            uint4* xl = (uint4*)(xw + 8 * lane);
            uint4 a = xl[0], b = xl[1];
            xe[0] = a.x, xe[1] = a.y, xe[2] = a.z, xe[3] = a.w, xe[4] = b.x, xe[5] = b.y, xe[6] = b.z, xe[7] = b.w;
            xl[0] = xl[1] = make_uint4(0u, 0u, 0u, 0u);
            __syncwarp();
        }
    }
}

// Step kk of a stage as a warp's A fragments (rows g and g + 8 of its 16, columns 2t and 8 + 2t), from stage12's.
__device__ __forceinline__ void step12(const Nib& f, uint32_t nw, uint2 sw, uint32_t x0, uint32_t x1, uint32_t (&A)[4]) {
    uint32_t e0 = Nib::high1(nw, f.hb4, 0) ^ x0, e1 = Nib::high1(nw, f.hb4, 1) ^ x1;
    A[0] = __byte_perm(sw.x, e0, 0x5140);
    A[1] = __byte_perm(sw.y, e1, 0x5140);
    A[2] = __byte_perm(sw.x, e0, 0x7362);
    A[3] = __byte_perm(sw.y, e1, 0x7362);
}

#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
// One wgmma m64nNk16 (a k-block), A (W's 64 rows by 16 columns) from registers, B (N tokens) from shared memory, and
// its commit: D = A B + (acc ? D : 0).
__device__ __forceinline__ void wgmma1_rs(float (&d)[96], const uint32_t (&a)[4], uint64_t db, int acc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %101, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n192k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95}, {%96, %97, %98, %99}, %100, p, 1, 1, 0;\n"
                 "wgmma.commit_group.sync.aligned;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(acc));
}
__device__ __forceinline__ void wgmma1_rs(float (&d)[128], const uint32_t (&a)[4], uint64_t db, int acc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %133, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, {%128, %129, %130, %131}, %132, p, 1, 1, 0;\n"
                 "wgmma.commit_group.sync.aligned;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(acc));
}
#endif

// A block's sum out (sum_out12's for the kernel below): its tile's rows (row unit pr: rows 16w + g (+ 8) of row
// block 2 pr + wg), tokens m0 on; r in each thread's fragment order. A whole tile (q < 0), or one the cluster covers
// alone: to Y. Else split tile q of the cluster tiles past whole waves, covered by clusters first to fin (U: the split
// tiles' stages, over nc clusters): to the block's slot (2 (cluster CL + rank) for the cluster's first split tile, + 1
// for its last), and the last of the tile's blocks to finish adds the slots in cluster order (the same every run);
// done: a counter a split tile a rank. nc (mma12_wgp_run's ncs) <= U, as capped there, so every cluster from first to
// fin has a stage of the tile and arrives here: the last finds fin - first before it.
template <int NT, int CL>
__device__ __forceinline__ void sum_out_wgp(float (&r)[NT / 2], int64_t q, int rank, int64_t pr, int64_t m0, int S, int64_t nc, int64_t U, float* parts, int* done, int& last, int ct, int64_t O, int64_t M, const __nv_bfloat16* bias, __nv_bfloat16* Y) {
    constexpr int R = 128;
    int wg = ct >> 7, w = (ct >> 5) & 3, g = (ct & 31) >> 2, t = ct & 3;
    int64_t first = q < 0 ? 0 : block_of_step(q * S, nc, U), fin = q < 0 ? 0 : block_of_step(q * S + S - 1, nc, U);
    auto slot = [&](int64_t c) { return parts + (2 * (c * CL + rank) + (c == first && c * U / nc != q * S)) * (NT * R); };
    if (first != fin) {
        float4* pp = (float4*)slot(blockIdx.x / CL);
#pragma unroll
        for (int i = 0; i < NT / 8; i++) pp[i * 256 + ct] = make_float4(r[4 * i], r[4 * i + 1], r[4 * i + 2], r[4 * i + 3]);
        bar_sync<256>(1);
        if (ct == 0) {
            int before = atom_inc_acq_rel(done + q * CL + rank);
            last = before == fin - first;
            if (last) done[q * CL + rank] = 0;  // ready for the next product
        }
        bar_sync<256>(1);
        if (last) {
#pragma unroll
            for (int i = 0; i < NT / 2; i++) r[i] = 0.f;
            for (int64_t c = first; c <= fin; c++) {
                const float4* bp = (const float4*)slot(c);
#pragma unroll
                for (int i = 0; i < NT / 8; i++) {
                    float4 v = __ldcg(bp + i * 256 + ct);
                    r[4 * i] += v.x;
                    r[4 * i + 1] += v.y;
                    r[4 * i + 2] += v.z;
                    r[4 * i + 3] += v.w;
                }
            }
        }
    }
    if (first == fin || last) {
#pragma unroll
        for (int jn = 0; jn < NT / 8; jn++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                int64_t o = (2 * pr + wg) * 64 + 16 * w + g + 8 * h;
                float bo = bias && o < O ? __bfloat162float(bias[o]) : 0.f;
#pragma unroll
                for (int c = 0; c < 2; c++) {
                    int64_t m = m0 + 8 * jn + 2 * t + c;
                    if (o < O && m < M) Y[m * O + o] = __float2bfloat16(r[4 * jn + 2 * h + c] + bo);
                }
            }
    }
}

#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
// A 2-D tile through a tensor map into the shared memory of each block of the cluster in mask (at dst there, its
// mbarrier at bar there).
__device__ __forceinline__ void tma_2d_mc(uint32_t dst, const CUtensorMap* map, int c0, int c1, uint32_t bar, uint16_t mask) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3}], [%4], %5;\n" ::"r"(dst), "l"((uint64_t)map), "r"(c0), "r"(c1), "r"(bar), "h"(mask) : "memory");
}
#endif

// A cluster's stages in order: its whole tiles c, c + nc, ... below D (cluster tiles: CL row units by a chunk), then its
// share of the split tiles' stages ([z0, z1) of the (T - D) S from tile D on: tile D + z / S, stage z mod S); a block
// its row unit of each.
template <int NT, int CL>
__device__ __forceinline__ void mma12_wgp_body(const CUtensorMap& xmap, Nib f, int64_t O, int64_t K, int64_t M, int64_t D, int ncs, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ Y, float* __restrict__ parts, int* __restrict__ done) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    using C = Wgp12<NT, CL>;
    // NS X tiles, NS W slots [a row block's steps, exceptions, bounds | the other's], the full and empty barriers, the
    // sum out's flag, the scratch. The X tiles 1024-byte aligned: the block's shared memory is its dynamic memory alone.
    extern __shared__ __align__(1024) uint8_t wsmem[];
    uint32_t base = (uint32_t)__cvta_generic_to_shared(wsmem);
    if (base & 1023) __trap();
    uint8_t *smem = wsmem, *gw = smem + C::NS * C::XB;
    uint32_t full = base + C::NS * (C::XB + C::WB), empty = full + 8 * C::NS;
    int& last = *(int*)(smem + C::NS * (C::XB + C::WB) + 16 * C::NS);
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, rank = 0;
    if constexpr (CL > 1) asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rank));
    int TC = (int)((M + NT - 1) / NT), S = (int)(K / 64);
    int64_t KS = K / 16, RB = O / 64, T = ((RB + 1) / 2 + CL - 1) / CL * TC, nc = gridDim.x / CL, c = blockIdx.x / CL;
    int64_t U = (T - D) * S, z0 = c < ncs ? c * U / ncs : 0, z1 = c < ncs ? (c + 1) * U / ncs : 0;
    int64_t whole = c < D ? (D - 1 - c) / nc + 1 : 0;  // the cluster's whole tiles
    int n = (int)(whole * S + z1 - z0);                // and its stages
    if (tid == 0) {
        for (int i = 0; i < C::NS; i++) {
            mbar_init(full + 8 * i, 1);
            mbar_init(empty + 8 * i, 2 * CL);  // each consumer warpgroup of the cluster's blocks: X's tile is theirs too
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    if constexpr (CL > 1) asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
    else __syncthreads();
    if (warp >= 8) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n" ::: "memory");
        if (warp != 8) return;
        // The TMA warp: lane 0 issues each stage; lanes 0-9 meanwhile load the next stage's bounds (row block lane / 5,
        // exc_base at its step lane mod 5 of the 4 and the next), gathered by lane 0 a stage later. A tile's row unit
        // and chunk are worked out once a tile.
        int64_t t = whole ? c : D + z0 / S, rb = 0;
        int s = whole ? 0 : (int)(z0 % S), pu, pc;
        bool w0 = whole > 0;
        auto at = [&]() {
            pu = (int)(t / TC), pc = (int)(t - (int64_t)pu * TC);
            rb = 2 * ((int64_t)pu * CL + rank);  // (row blocks past O: the last one's, their products not written)
        };
        auto bound = [&]() { return lane < 10 ? __ldg(f.exc_base + min(rb + lane / 5, RB - 1) * KS + 4 * s + lane % 5) : 0; };
        at();
        int ev = n > 0 ? bound() : 0;
        for (int j = 0; j < n; j++) {
            int e[10];
#pragma unroll
            for (int i = 0; i < 10; i++) e[i] = __shfl_sync(FULL, ev, i);
            int sl = j % C::NS, s0 = s, pc0 = pc;
            int64_t rb0 = rb;
            if (++s == S) {
                s = 0;
                if (!w0) t++;
                else if ((t += nc) >= D) w0 = false, t = D + z0 / S, s = (int)(z0 % S);
                at();
            }
            if (j + 1 < n) ev = bound();
            if (lane == 0) {
                uint32_t fb = full + 8 * sl, xs = base + sl * C::XB, ws = base + C::NS * C::XB + sl * C::WB;
                if (j >= C::NS) mbar_wait(empty + 8 * sl, (j / C::NS - 1) & 1);  // the cluster's consumers are done with stage j - NS
                uint32_t bytes = C::XB + 2 * C::CB;
                int a[2], na[2];
#pragma unroll
                for (int r = 0; r < 2; r++) {
                    exc_copy(e[5 * r], e[5 * r + 4], C::EB, a[r], na[r]);
                    bytes += na[r] > 0 ? 4 * na[r] : 0;
                    int* bs = (int*)(gw + sl * C::WB + r * C::RBB + C::CB + C::EB);
#pragma unroll
                    for (int i = 0; i < 5; i++) bs[i] = e[5 * r + i];
                    bs[5] = na[r] < 0 ? -1 : a[r];
                }
                mbar_expect_tx(fb, bytes);
                if constexpr (CL > 1) tma_2d_mc(xs + rank * (C::XB / CL), &xmap, s0 * 64, pc0 * NT + rank * (NT / CL), fb, (uint16_t)((1 << CL) - 1));
                else tma_2d(xs, &xmap, s0 * 64, pc0 * NT, fb);
#pragma unroll
                for (int r = 0; r < 2; r++) {
                    uint32_t cs = ws + r * C::RBB;
                    bulk_g2s(cs, f.data + (min(rb0 + r, RB - 1) * KS + 4 * s0) * STEP12, C::CB, fb, TC == 1);  // (read once but where chunks share it)
                    if (na[r] > 0) bulk_g2s(cs + C::CB, f.exc + a[r], 4 * na[r], fb);
                }
            }
            __syncwarp();
        }
        if (lane) return;
        // Every stage's slot released by the cluster's consumers before the block ends (their arrivals and the other
        // blocks' copies land in its shared memory).
        for (int j = max(0, n - C::NS); j < n; j++) mbar_wait(empty + 8 * (j % C::NS), (j / C::NS) & 1);
        return;
    }
    // Consumer warpgroup wg: row block 2 pr + wg of each tile; warp w its rows 16w to 16w + 15.
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(C::CR) : "memory");
    int ct = tid, wg = ct >> 7, w = (ct >> 5) & 3;
    uint32_t* xw = (uint32_t*)(smem + C::NS * (C::XB + C::WB) + 16 * C::NS + 16) + 256 * warp;  // this warp's exceptions' scratch
    ((uint4*)xw)[2 * lane] = ((uint4*)xw)[2 * lane + 1] = make_uint4(0u, 0u, 0u, 0u);
    __syncwarp();
    float d[NT / 2] = {};  // rows 16w + g (+ 8), tokens 8jn + 2t (+ 1): d[4jn + 2h + c]; set by a tile's first product
    // (zeroed all the same, as mma12_tma_kernel's: left unset, CUDA 12.8's ptxas serialized every wgmma, its C7515)
    uint32_t A[4][4];
    int held = -1;  // the slot of the stage before, whose products may still be running
    auto release = [&](int sl) {  // a warpgroup's first warp's lane i arrives on block i's barrier of the cluster, by predicate
        __syncwarp();          // (no branch among the products): its products are the warpgroup's, their reads done
        uint32_t go = w == 0 && lane < CL && sl >= 0, bar = empty + 8 * max(sl, 0);
        if constexpr (CL > 1) asm volatile("{\n.reg .pred p;\n.reg .b32 ra;\nsetp.ne.u32 p, %1, 0;\nmapa.shared::cluster.u32 ra, %0, %2;\n@p mbarrier.arrive.shared::cluster.b64 _, [ra];\n}\n" ::"r"(bar), "r"(go), "r"(lane) : "memory");
        else asm volatile("{\n.reg .pred p;\nsetp.ne.u32 p, %1, 0;\n@p mbarrier.arrive.shared::cta.b64 _, [%0];\n}\n" ::"r"(bar), "r"(go) : "memory");
    };
    auto keep_d = [&]() {  // d in its registers, untouched while the products run
#pragma unroll
        for (int i = 0; i < NT / 2; i++) asm volatile("" : "+f"(d[i])::"memory");
    };
    int64_t t = whole ? c : D + z0 / S;
    int s = whole ? 0 : (int)(z0 % S);
    bool w0 = whole > 0;
    for (int j = 0; j < n;) {
        int s0 = s, e = w0 ? S : min(S, s + (n - j));
        for (; s < e; s++, j++) {
            int sl = j % C::NS;
            mbar_wait(full + 8 * sl, (j / C::NS) & 1);
            const uint8_t* sp = gw + sl * C::WB + wg * C::RBB;
            const int* bs = (const int*)(sp + C::CB + C::EB);
            uint32_t nw[4];
            uint2 sw[4];
            stage12(sp, lane, w, nw, sw);
            uint32_t xe[8];  // the stage's exceptions, before its first product (ptxas keeps branches out from among them)
            stage12_exc(f, (const uint32_t*)(sp + C::CB), bs, lane, w, xw, xe);
            uint64_t xd = sw128_desc(base + sl * C::XB);  // X's tile: k-block kk's 32 bytes on (+2 a k-block)
#pragma unroll
            for (int kk = 0; kk < 4; kk++) {
                wg_wait<2>();  // at most two k-blocks' products running: the A set 4 back is free
                if (kk == 2) release(held), held = sl;  // (and the stage before's products all done: its slot)
                step12(f, nw[kk], sw[kk], xe[2 * kk], xe[2 * kk + 1], A[kk]);
                wg_fence();
                wgmma1_rs(d, A[kk], xd + 2 * kk, s > s0 || kk > 0);
            }
        }
        wg_wait<0>();
        keep_d();
        release(held);
        held = -1;
        int pu = (int)(t / TC);
        int64_t m0 = (t - (int64_t)pu * TC) * NT;
        sum_out_wgp<NT, CL>(d, w0 ? -1 : t - D, rank, (int64_t)pu * CL + rank, m0, S, ncs, U, parts, done, last, ct, O, M, bias, Y);
        s = 0;
        if (!w0) t++;
        else if ((t += nc) >= D) w0 = false, t = D + z0 / S, s = (int)(z0 % S);
    }
#elif defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    __trap();  // sm_90a code, launched from a build without it (or on AMD)
#endif
}

template <int NT, int CL>
__global__ void __launch_bounds__((Wgp12<NT, CL>::THREADS), 1) mma12_wgp_kernel(const __grid_constant__ CUtensorMap xmap, Nib f, int64_t O, int64_t K, int64_t M, int64_t D, int ncs, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ Y, float* __restrict__ parts, int* __restrict__ done) {
    mma12_wgp_body<NT, CL>(xmap, f, O, K, M, D, ncs, bias, Y, parts, done);
}

// Many tokens on Ampere and Ada (sm_86 to sm_89; the A100 its own kernel), the 12-bit layout: mma12_tma_kernel's plan with this
// generation's instructions. A producer warp copies each stage with cp.async, 16 bytes a thread, and the
// stage's mbarrier completes once every lane's copies have landed (cp.async.mbarrier.arrive) and lane 0
// has written the bounds; X's tile is laid out with a row's 16-byte chunk c at chunk c ^ (row mod 8), read
// as B fragments by ldmatrix. The consumer warps decode their 16 rows into A fragments as the TMA
// kernel's do and multiply with mma.sync, 16 rows by 8 tokens by 16 columns. Stream-K and the sum out as
// there. NS stages in flight; WG 4 where a block may hold 153 KB of shared memory (sm_87, sm_90, sm_100; an
// A100 takes a kernel of its own), else 2 (sm_86, sm_89, sm_120).
template <int NT, int WG> struct Mid12 {
    static constexpr int THREADS = 128 * WG + 32;  // WG consumer warpgroups, the producer warp
    static constexpr int NS = 4;
    static constexpr int XB = NT * 128;
    static constexpr int CB = 4 * (int)STEP12, EB = 512, BB = 32;  // a row block's steps a stage, its exceptions (up to 128), their bounds
    static constexpr int RBB = CB + EB + BB;
    static constexpr int SLOT = (XB + WG * RBB + 127) / 128 * 128;
    static constexpr int SHARED = NS * SLOT + 128 + 16 * NS + 4096 * WG;  // alignment, the full and empty barriers, the warps' exceptions' scratch
};

template <int NT, int WG>
__device__ __forceinline__ void mma12_mid_body(Nib f, int64_t O, int64_t K, int64_t M, const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ Y, float* __restrict__ parts, int* __restrict__ done) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using C = Mid12<NT, WG>;
    extern __shared__ __align__(128) uint8_t smem_raw[];  // NS slots [X tile | a row block's steps, exceptions, bounds | ...], the barriers
    __shared__ int last;
    uint32_t raw = (uint32_t)__cvta_generic_to_shared(smem_raw), base = (raw + 127) & ~127u;
    uint8_t* gbase = smem_raw + (base - raw);
    uint32_t full = base + C::NS * C::SLOT, empty = full + 8 * C::NS;
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int64_t KS = K / 16, RB = O / 64, U = (RB + WG - 1) / WG * (K / 64), nb = gridDim.x;
    int S = (int)(K / 64);
    int64_t u0 = blockIdx.x * U / nb;
    int n = (int)((blockIdx.x + 1) * U / nb - u0);
    int p0 = (int)(u0 / S), s0 = (int)(u0 - (int64_t)p0 * S);
    // compute-sanitizer's racecheck (CUDA 12.8's) reports hazards on the slots; none is a race. Three kinds:
    //   WAR: consumer lanes 1 to 31 read a slot, then the producer copies into it again. Their reads come before
    //   lane 0's arrive on "empty" by __syncwarp, which orders the warp's memory operations. The tool does not
    //   carry that order into another lane's arrive (with every lane arriving, "empty" counting 128 WG, it is gone).
    //   RAW and WAW, "invalid memcpy_async synchronization": a consumer reads a copy after its wait on "full", or a
    //   producer lane copies into the padding of a stage's exceptions NS stages on. "full"'s phase cannot complete
    //   before each lane's cp.async.mbarrier.arrive.noinc fires, which is once all of that lane's copies have
    //   landed, and every consumer waits on it before it arrives on "empty" (with cp.async.wait_all and a plain
    //   arrive in its place, the tool says nothing).
    // With both changes the tool reports no hazard, and the bits are the same. An observer that compared every slot
    // with global memory the moment each wait returned found none apart (A10, 2026-11-26). They are not made here
    // (a consumer lane each to arrive, the producer to wait for its copies): that changes the kernel's speed, not its result.
    if (tid == 0)
        for (int i = 0; i < C::NS; i++) {
            mbar_init(full + 8 * i, 33);  // the 32 lanes' copies, lane 0's bounds
            mbar_init(empty + 8 * i, 4 * WG);
        }
    __syncthreads();
    if (warp == 4 * WG) {
        // The producer warp: every lane copies a share of stage j; lane j mod 8 holds its bounds, loaded
        // with its batch of 8 stages while the batch before was copied (as the TMA warp's).
        int e[5 * WG], en[5 * WG];
        auto ahead = [&](int p, int s, int k) {
            s += k;
            stage_bounds<WG>(f, RB, KS, p + s / S, s % S, en);
        };
        if (lane < 8 && lane < n) ahead(p0, s0, lane);
        for (int j = 0, p = p0, s = s0; j < n; j++, s = s + 1 == S ? 0 : s + 1, p += s == 0) {
            if ((j & 7) == 0) {
#pragma unroll
                for (int i = 0; i < 5 * WG; i++) e[i] = en[i];
                if (lane < 8 && j + 8 + lane < n) ahead(p, s, 8 + lane);
            }
            int eb[5 * WG];
#pragma unroll
            for (int i = 0; i < 5 * WG; i++) eb[i] = __shfl_sync(FULL, e[i], j & 7);
            int sl = j % C::NS;
            uint32_t xs = base + sl * C::SLOT;
            if (j >= C::NS) mbar_wait(empty + 8 * sl, (j / C::NS - 1) & 1);  // the consumers are done with stage j - NS
            for (int i = lane; i < NT * 8; i += 32) {  // X's tile; rows past M read as zeros
                int rr = i >> 3, c = i & 7;
                cp_async16(xs + rr * 128 + ((c ^ (rr & 7)) << 4), X + (rr < M ? (int64_t)rr * K : 0) + (int64_t)s * 64 + c * 8, rr < M ? 16 : 0);
            }
#pragma unroll
            for (int r = 0; r < WG; r++) {
                uint32_t cs = xs + C::XB + r * C::RBB;
                const uint8_t* src = f.data + (min((int64_t)WG * p + r, RB - 1) * KS + 4 * s) * STEP12;
                for (int i = lane; i < C::CB / 16; i += 32) cp_async16(cs + 16 * i, src + 16 * i, 16);
                int a, na;
                exc_copy(eb[5 * r], eb[5 * r + 4], C::EB, a, na);
                for (int i = lane; 4 * i < na; i += 32) cp_async16(cs + C::CB + 16 * i, f.exc + a + 4 * i, 16);
                if (lane == 0) {
                    int* bs = (int*)(gbase + sl * C::SLOT + C::XB + r * C::RBB + C::CB + C::EB);
#pragma unroll
                    for (int i = 0; i < 5; i++) bs[i] = eb[5 * r + i];
                    bs[5] = na < 0 ? -1 : a;
                }
            }
            asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" ::"r"(full + 8 * sl) : "memory");
            if (lane == 0) mbar_arrive(full + 8 * sl);
        }
        return;
    }
    // Consumer warpgroup wg: row block WG p + wg of each unit p; warp w its rows 16w to 16w + 15.
    int ct = tid, wg = ct >> 7, w = (ct >> 5) & 3;
    uint32_t* xw = (uint32_t*)(gbase + C::NS * C::SLOT + 16 * C::NS) + 256 * warp;  // this warp's exceptions' scratch
    ((uint4*)xw)[2 * lane] = ((uint4*)xw)[2 * lane + 1] = make_uint4(0u, 0u, 0u, 0u);
    __syncwarp();
    for (int j = 0, p = p0, s = s0; j < n; p++, s = 0) {
        int len = min(S - s, n - j);
        float d[NT / 2];  // n-tile jn's C fragment at d[4jn]: rows g, g + 8 by tokens 8jn + 2t, + 1
#pragma unroll
        for (int i = 0; i < NT / 2; i++) d[i] = 0.f;
        for (int k = 0; k < len; k++) {
            int jj = j + k, sl = jj % C::NS;
            mbar_wait(full + 8 * sl, (jj / C::NS) & 1);
            const uint8_t* sp = gbase + sl * C::SLOT + C::XB + wg * C::RBB;
            const int* bs = (const int*)(sp + C::CB + C::EB);
            int eb[5];
#pragma unroll
            for (int i = 0; i < 5; i++) eb[i] = bs[i];
            uint32_t A[4][4];
            decode12_rows(f, sp, (const uint32_t*)(sp + C::CB), eb, bs[5], lane, w, xw, A);
            uint32_t xs = base + sl * C::SLOT;
#pragma unroll
            for (int kk = 0; kk < 4; kk++)
#pragma unroll
                for (int jp = 0; jp < NT / 16; jp++) {
                    // Tokens 16jp to 16jp + 15, columns 16kk to 16kk + 15: B fragments of n-tiles 2jp and 2jp + 1.
                    int rr = 16 * jp + (lane & 7) + ((lane >> 4) << 3), c = 2 * kk + ((lane >> 3) & 1);
                    uint32_t b[4];
                    ldmatrix_x4(b, xs + rr * 128 + ((c ^ (rr & 7)) << 4));
                    mma16816(&d[8 * jp], A[kk], b[0], b[1]);
                    mma16816(&d[8 * jp + 4], A[kk], b[2], b[3]);
                }
            __syncwarp();
            if (lane == 0) mbar_arrive(empty + 8 * sl);  // the slot is free for stage jj + NS
        }
        sum_out12<NT, WG>(d, p, p, 0, S, nb, U, parts, done, last, ct, O, M, bias, Y);
        j += len;
    }
#elif defined(__HIP_DEVICE_COMPILE__)
    __trap();  // (Ampere's and later NVIDIA GPUs' alone: never launched on AMD)
#endif
}

template <int NT, int WG>
__global__ void __launch_bounds__((Mid12<NT, WG>::THREADS), 1) mma12_mid_kernel(Nib f, int64_t O, int64_t K, int64_t M, const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ Y, float* __restrict__ parts, int* __restrict__ done) {
    mma12_mid_body<NT, WG>(f, O, K, M, X, bias, Y, parts, done);
}

// A 16-byte copy of data read once (W's steps): evicted from L2 first.
__device__ __forceinline__ void cp_async16_once(uint32_t dst, const void* src, uint64_t pol) {
#if defined(__HIP_PLATFORM_AMD__)
    (void)pol;
    cp_async16(dst, src, 16);
#else
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "l"(pol));
#endif
}
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// Many tokens on the A100 (sm_80), the 12-bit layout: mma_gemm_big_kernel's split of the warps, with the
// producers' reads staged. A producer warp a step of W's 4 RBB a stage (64 columns of RBB row blocks): it copies
// its step and the step's exceptions by cp.async NW - 1 stages ahead into its share of a ring (the exceptions' bounds
// loaded 32 stages ahead, a stage a lane), and decodes it from there into B fragments in shared memory; the
// producers copy X's tile a stage ahead of its decode. CW consumer warps multiply, each 16 MT tokens by a row
// block, on the tensor cores; named barriers pass a stage's buffers (NB of them) between the two. Stream-K over
// units (RBB row blocks by TM tokens) and their stages, as mma_gemm_kernel's: a unit covered by several blocks is
// summed by the last to finish, in block order (the same result every run).
template <int CW, int NB, int NW, int RBB, int MT> struct Ws12 {
    static constexpr int PW = 4 * RBB, THREADS = 32 * (CW + PW), PT = 32 * PW, CT = 32 * CW;
    static constexpr int TM = 16 * MT * CW / RBB;     // tokens a unit
    static constexpr int A_BYTES = TM * 128;          // X's tile a stage
    static constexpr int B_UINT4 = RBB * 4 * 4 * 32;  // W's fragments a stage: [row block][step][4][lane]
    static constexpr int DSLOT = A_BYTES + B_UINT4 * 16;
    static constexpr int EB = 128, SB = (int)STEP12 + EB + 16;  // a producer warp's share of a ring slot: its step, the step's exceptions (past EB bytes read from global memory), their bounds
    static constexpr int WSLOT = PW * SB;
    static constexpr int SHARED = NB * DSLOT + NW * WSLOT + 128;
    static constexpr int XQ = TM * 8 / PT;            // X's chunks a producer thread
    // NB >= 3: the producers copy stage j + 1's X tile before they decode stage j, into the slot the consumers
    // free after stage j + 1 - NB, and the consumers ask for stage j + 1 before they free stage j.
    static_assert(NB >= 3 && NW >= 2 && 2 * NB + 2 <= 15 && XQ * PT == TM * 8, "shapes");
};

template <int CW, int NB, int NW, int RBB, int MT>
__device__ __forceinline__ void mma12_ws_body(Nib f, int64_t O, int64_t K, int64_t M, const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ Y, float* __restrict__ parts, int* __restrict__ done) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using C = Ws12<CW, NB, NW, RBB, MT>;
    constexpr int FULL0 = 1, EMPTY0 = 1 + NB, CBAR = 1 + 2 * NB;  // named barriers
    extern __shared__ __align__(128) uint8_t smem_raw[];  // NB slots [X tile | B fragments], then the ring: NW slots [a producer warp's step, exceptions, bounds | ...]
    __shared__ int last;
    uint32_t raw = (uint32_t)__cvta_generic_to_shared(smem_raw), base = (raw + 127) & ~127u;
    uint8_t* gbase = smem_raw + (base - raw);
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int64_t KS = K / 16, RB = O / 64;
    int tiles = (int)((M + C::TM - 1) / C::TM), S = (int)(K / 64);
    int64_t T = (RB + RBB - 1) / RBB * tiles * S, nb = gridDim.x, u0 = blockIdx.x * T / nb;
    int n = (int)((blockIdx.x + 1) * T / nb - u0);  // the block's stages, u0 on: unit (tile of a pair of row blocks) u / S, stage u mod S
    int p0 = (int)(u0 / S), s0 = (int)(u0 - (int64_t)p0 * S);
    if (warp >= CW) {
        // Producer warp pw: step kk of row block pw / 4 of each stage. Commit groups a stage j: X's tile of stage
        // j + 1, then the warp's step of stage j + NW - 1.
        int pt = tid - C::CT, pw = warp - CW, kk = pw & 3, xr = pt >> 3, xc = pt & 7;
        uint64_t pol;
        asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
        const uint32_t wbase = base + NB * C::DSLOT + pw * C::SB;
        const uint8_t* gw = gbase + NB * C::DSLOT + pw * C::SB;
        // The step's exceptions' bounds (exc_base at it and the next): lane i holds them for stage i of the batch of
        // 32 (e0, e1), the next batch's (en0, en1) loaded meanwhile.
        int e0 = 0, e1 = 0, en0 = 0, en1 = 0;
        auto ahead = [&](int p, int s, int k) {  // the stage k after (p, s)
            s += k;
            p += s / S;
            s %= S;
            const int32_t* q = f.exc_base + min((int64_t)(p / tiles) * RBB + (pw >> 2), RB - 1) * KS + 4 * s + kk;
            en0 = __ldg(q);
            en1 = __ldg(q + 1);
        };
        if (lane < n) ahead(p0, s0, lane);
        int cj = 0, cp = p0, cs = s0, csl = 0;  // the next step to copy: stage cj (unit cp, stage cs), slot csl
        const uint8_t* wsrc = f.data + (min((int64_t)(p0 / tiles) * RBB + (pw >> 2), RB - 1) * KS + 4 * s0 + kk) * STEP12 + 16 * lane;
        auto wcopy = [&]() {
            if (cj < n) {
                uint32_t d = wbase + csl * C::WSLOT + 16 * lane;
#pragma unroll
                for (int q = 0; q < (int)STEP12 / 512; q++) cp_async16_once(d + 512 * q, wsrc + 512 * q, pol);
                if ((cj & 31) == 0) {
                    e0 = en0, e1 = en1;
                    if (cj + 32 + lane < n) ahead(cp, cs, 32 + lane);
                }
                int lo = __shfl_sync(FULL, e0, cj & 31), hi = __shfl_sync(FULL, e1, cj & 31), a, na;
                exc_copy(lo, hi, C::EB, a, na);
                if (4 * lane < na) cp_async16(wbase + csl * C::WSLOT + STEP12 + 16 * lane, f.exc + a + 4 * lane, 16);
                if (lane == 0) *(int4*)(gw + csl * C::WSLOT + STEP12 + C::EB) = make_int4(lo, hi, na < 0 ? -1 : a, 0);
                cj++;
                csl = csl + 1 == NW ? 0 : csl + 1;
                if (++cs == S) {
                    cs = 0, cp++;
                    wsrc = f.data + (min((int64_t)(cp / tiles) * RBB + (pw >> 2), RB - 1) * KS + kk) * STEP12 + 16 * lane;
                } else {
                    wsrc += 4 * STEP12;
                }
            }
            asm volatile("cp.async.commit_group;\n" ::);
        };
        int xj = 0, xp = p0, xs = s0, xsl = 0;  // the next X tile to copy: stage xj (unit xp, stage xs), slot xsl; this thread's first row xm (none past M)
        int64_t xm = (int64_t)(p0 % tiles) * C::TM + xr;
        auto xcopy = [&]() {
            if (xj < n) {
                uint32_t d = base + xsl * C::DSLOT + xr * 128 + ((xc ^ (xr & 7)) << 4);
#pragma unroll
                for (int q = 0; q < C::XQ; q++)
                    if (xm + C::PT / 8 * q < M) cp_async16(d + C::PT / 8 * 128 * q, X + (xm + C::PT / 8 * q) * K + (int64_t)xs * 64 + xc * 8, 16);
                xj++;
                xsl = xsl + 1 == NB ? 0 : xsl + 1;
                if (++xs == S) {
                    xs = 0, xp++;
                    xm = (int64_t)(xp % tiles) * C::TM + xr;
                }
            }
            asm volatile("cp.async.commit_group;\n" ::);
        };
#pragma unroll 1
        for (int i = 1 - NW; i < 0; i++) {
            if (i == -1) xcopy();
            else asm volatile("cp.async.commit_group;\n" ::);
            wcopy();
        }
        int wsl = 0, bsl = 0;  // stage j's ring slot, and its slot of fragments
        uint4* bdst = (uint4*)(gbase + C::A_BYTES) + pw * 4 * 32 + lane;
#pragma unroll 1
        for (int j = 0; j < n; j++) {
            cp_async_wait<2 * (NW - 2)>();
            __syncwarp();  // stage j's step has landed
            if (j + 1 < n && j + 1 >= NB) bar_sync<C::THREADS>(EMPTY0 + xsl);  // the consumers are done with stage j + 1 - NB (xsl: stage j + 1's slot)
            xcopy();
            wcopy();
            // The step into B fragments, as Nib::decode (its loads and exceptions from the ring).
            const uint8_t* q = gw + wsl * C::WSLOT;
            uint4 c4 = *(const uint4*)(q + 16 * lane), x0 = *(const uint4*)(q + 512 + 16 * lane), x1 = *(const uint4*)(q + 1024 + 16 * lane);
            uint32_t nbw[4] = {c4.x, c4.y, c4.z, c4.w}, sw[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w}, ew[8];
            f.high(nbw, ew);
            int4 bd = *(const int4*)(q + STEP12 + C::EB);
            const uint32_t *se = (const uint32_t*)(q + STEP12), *ge = f.exc;
            Nib::patch<2>([&](int k) { return bd.z >= 0 ? se[k - bd.z] : __ldg(ge + k); }, bd.x, bd.y, lane, ew);
            uint32_t R[16];
            Nib::pairs(sw, ew, R);
            uint4* d = bdst + bsl * (C::DSLOT / 16);
#pragma unroll
            for (int qq = 0; qq < 4; qq++) d[qq * 32] = make_uint4(R[4 * qq], R[4 * qq + 1], R[4 * qq + 2], R[4 * qq + 3]);
            cp_async_wait<3>();                   // stage j's X tile has landed
            bar_arrive<C::THREADS>(FULL0 + bsl);  // stage j is ready
            wsl = wsl + 1 == NW ? 0 : wsl + 1;
            bsl = bsl + 1 == NB ? 0 : bsl + 1;
        }
        cp_async_wait<0>();
        return;
    }
    // Consumer: tokens 16 MT (warp mod (CW / RBB)) on of the unit's TM, row block warp / (CW / RBB) of its RBB.
    int g = lane >> 2, t = lane & 3, wm = warp % (CW / RBB), rbl = warp / (CW / RBB);
    float acc[MT][8][4];
#pragma unroll
    for (int mt = 0; mt < MT; mt++)
#pragma unroll
        for (int nn = 0; nn < 8; nn++)
#pragma unroll
            for (int q = 0; q < 4; q++) acc[mt][nn][q] = 0.f;
    uint32_t a_row = (uint32_t)(wm * 16 * MT + (lane & 7) + ((lane >> 3) & 1) * 8) * 128, a_half = lane >> 4, a_sw = lane & 7;
    uint32_t fa[2][MT][4];
    uint4 fb[2][4];
    auto frags = [&](int bi, int kk, int fi) {
        uint32_t slot = base + (uint32_t)bi * C::DSLOT;
        const uint4* bsrc = (const uint4*)(gbase + bi * C::DSLOT + C::A_BYTES) + rbl * 4 * 4 * 32 + lane;
#pragma unroll
        for (int mt = 0; mt < MT; mt++) ldmatrix_x4(fa[fi][mt], slot + a_row + mt * 16 * 128 + (((2 * kk + a_half) ^ a_sw) << 4));
#pragma unroll
        for (int q = 0; q < 4; q++) fb[fi][q] = bsrc[(kk * 4 + q) * 32];
    };
    if (n > 0) {
        bar_sync<C::THREADS>(FULL0);
        frags(0, 0, 0);
    }
    int p = p0, s = s0, bsl = 0;
#pragma unroll 1
    for (int j = 0; j < n; j++) {
        int nsl = bsl + 1 == NB ? 0 : bsl + 1;
        bool out = s == S - 1 || j == n - 1;  // the unit's last stage here: its sum out next
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            int fi = kk & 1;
            if (kk + 1 < 4) frags(bsl, kk + 1, fi ^ 1);
            else if (!out) {  // (at a unit's end, after its sum out: the next stage's fragments not held through it)
                bar_sync<C::THREADS>(FULL0 + nsl);  // stage j + 1 is ready
                frags(nsl, 0, fi ^ 1);
            }
#pragma unroll
            for (int mt = 0; mt < MT; mt++) {
                mma16816(acc[mt][0], fa[fi][mt], fb[fi][0].x, fb[fi][0].y);
                mma16816(acc[mt][1], fa[fi][mt], fb[fi][0].z, fb[fi][0].w);
                mma16816(acc[mt][2], fa[fi][mt], fb[fi][1].x, fb[fi][1].y);
                mma16816(acc[mt][3], fa[fi][mt], fb[fi][1].z, fb[fi][1].w);
                mma16816(acc[mt][4], fa[fi][mt], fb[fi][2].x, fb[fi][2].y);
                mma16816(acc[mt][5], fa[fi][mt], fb[fi][2].z, fb[fi][2].w);
                mma16816(acc[mt][6], fa[fi][mt], fb[fi][3].x, fb[fi][3].y);
                mma16816(acc[mt][7], fa[fi][mt], fb[fi][3].z, fb[fi][3].w);
            }
        }
        if (j + NB < n) bar_arrive<C::THREADS>(EMPTY0 + bsl);  // done with stage j's buffers
        bsl = nsl;
        if (out) {
            // Unit p's sum out: to Y where this block covers it, else to its slot (the consumers' accumulators in
            // thread order), the last of its blocks to finish adding the slots in block order.
            int64_t first = block_of_step((int64_t)p * S, nb, T), fin = block_of_step((int64_t)p * S + S - 1, nb, T);
            if (first != fin) {
                float4* pp = (float4*)(parts + ((int64_t)blockIdx.x + p) * (C::TM * 64 * RBB));
#pragma unroll
                for (int mt = 0; mt < MT; mt++)
#pragma unroll
                    for (int nn = 0; nn < 8; nn++) pp[(mt * 8 + nn) * C::CT + tid] = make_float4(acc[mt][nn][0], acc[mt][nn][1], acc[mt][nn][2], acc[mt][nn][3]);
                __threadfence();
                bar_sync<C::CT>(CBAR);
                if (tid == 0) {
                    last = atomicAdd(done + p, 1) == fin - first;
                    if (last) done[p] = 0;  // ready for the next product
                }
                bar_sync<C::CT>(CBAR);
                if (last) {
                    __threadfence();
#pragma unroll
                    for (int mt = 0; mt < MT; mt++)
#pragma unroll
                        for (int nn = 0; nn < 8; nn++)
#pragma unroll
                            for (int q = 0; q < 4; q++) acc[mt][nn][q] = 0.f;
                    for (int64_t b = first; b <= fin; b++) {
                        const float4* bp = (const float4*)(parts + (b + p) * (C::TM * 64 * RBB));
#pragma unroll
                        for (int mt = 0; mt < MT; mt++)
#pragma unroll
                            for (int nn = 0; nn < 8; nn++) {
                                float4 v = __ldcg(bp + (mt * 8 + nn) * C::CT + tid);
                                acc[mt][nn][0] += v.x;
                                acc[mt][nn][1] += v.y;
                                acc[mt][nn][2] += v.z;
                                acc[mt][nn][3] += v.w;
                            }
                    }
                }
            }
            if (first == fin || last) {
                int64_t rb = (int64_t)(p / tiles) * RBB + rbl, m0 = (int64_t)(p % tiles) * C::TM + wm * 16 * MT + g;
                if (rb < RB)
#pragma unroll
                    for (int nn = 0; nn < 8; nn++) {
                        int64_t o = rb * 64 + nn * 8 + t * 2;
                        float b0 = bias ? __bfloat162float(bias[o]) : 0.f, b1 = bias ? __bfloat162float(bias[o + 1]) : 0.f;
#pragma unroll
                        for (int mt = 0; mt < MT; mt++)
#pragma unroll
                            for (int h = 0; h < 2; h++) {
                                int64_t m = m0 + mt * 16 + h * 8;
                                if (m < M) *(__nv_bfloat162*)(Y + m * O + o) = __floats2bfloat162_rn(acc[mt][nn][2 * h] + b0, acc[mt][nn][2 * h + 1] + b1);
                            }
                    }
            }
#pragma unroll
            for (int mt = 0; mt < MT; mt++)
#pragma unroll
                for (int nn = 0; nn < 8; nn++)
#pragma unroll
                    for (int q = 0; q < 4; q++) acc[mt][nn][q] = 0.f;
            if (j + 1 < n) {
                bar_sync<C::THREADS>(FULL0 + bsl);  // stage j + 1 is ready
                frags(bsl, 0, 0);
            }
        }
        if (++s == S) s = 0, p++;
    }
#elif defined(__HIP_DEVICE_COMPILE__)
    __trap();  // (Ampere's and later NVIDIA GPUs' alone: never launched on AMD)
#endif
}

template <int CW, int NB, int NW, int RBB, int MT>
__global__ void __launch_bounds__((Ws12<CW, NB, NW, RBB, MT>::THREADS), 1) mma12_ws_kernel(Nib f, int64_t O, int64_t K, int64_t M, const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ Y, float* __restrict__ parts, int* __restrict__ done) {
    mma12_ws_body<CW, NB, NW, RBB, MT>(f, O, K, M, X, bias, Y, parts, done);
}

// The experts' products on Hopper (sm_90a), the 12-bit layout, by wgmma: mma12_tma_kernel's plan (W decoded into wgmma's A registers
// a stage of 4 steps, the pairs as B through a tensor map, a TMA warp feeding a ring of stages) over the tiles of a plan. A tile is a
// hit expert's chunk of NT pairs (moe_wg_tiles_kernel's record: the expert, the chunk's first row of the order, its pairs); a unit a
// tile by a row unit (two row blocks, one a warpgroup: 128 rows; with the gate, a gate row block and its up's, so that
// SiLU(gate) up is made from both warpgroups' sums), the blocks (one an SM) taking units in turn, each unit's K in order and its
// sums written out when it is done (no K split, no parts). X's rows are the order's: the gate and up product's through
// moe_wg_gather_kernel (a TMA tile is rows in a row), the down's the gate and up's output as it was written (the order). With the
// gate (ACT 1): y [pairs, O / 2] = SiLU(gate) up, + biases [E, O], in the order; without: the order's rows of y [., O] (+ bias),
// or (Y32) times the pair's weight to row j of Y32 [P, O] for pair j, as mma_moe_kernel does.
//
// moe_wg_tiles_kernel: one block, the tiles' records after a header {the records, 0, 0, 0}: a record is {expert, first row of
// the order, pairs (at most NT), 1 where the expert's pairs are one chunk}. The hits are scanned 1,024 at a time.
__device__ __forceinline__ void moe_wg_tiles_body(const int* __restrict__ plan, int E, int NT, int4* __restrict__ tiles) {
    __shared__ int wsum[32];
    const int h = __ldg(plan), lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int base = 0;
    for (int g0 = 0; g0 < h; g0 += 1024) {
        int i = g0 + threadIdx.x, p0 = 0, p1 = 0, v = 0;
        if (i < h) {
            p0 = __ldg(plan + 1 + E + i), p1 = __ldg(plan + 2 + E + i);
            v = (p1 - p0 + NT - 1) / NT;
        }
        int x = v;  // the inclusive scan over the warp, then over the warps' sums
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            int y = __shfl_up_sync(FULL, x, d);
            if (lane >= d) x += y;
        }
        if (lane == 31) wsum[warp] = x;
        __syncthreads();
        if (warp == 0) {
            int z = wsum[lane];
#pragma unroll
            for (int d = 1; d < 32; d <<= 1) {
                int y = __shfl_up_sync(FULL, z, d);
                if (lane >= d) z += y;
            }
            wsum[lane] = z;
        }
        __syncthreads();
        int at = base + (x - v) + (warp ? wsum[warp - 1] : 0), total = wsum[31];
        if (i < h) {
            const int e = __ldg(plan + 1 + i), whole = p1 - p0 <= NT;
            for (int c = 0; c < v; c++) tiles[1 + at + c] = make_int4(e, p0 + c * NT, min(NT, p1 - p0 - c * NT), whole);
        }
        base += total;
        __syncthreads();  // wsum is reused
    }
    if (threadIdx.x == 0) tiles[0] = make_int4(base, 0, 0, 0);
}

__global__ void __launch_bounds__(1024) moe_wg_tiles_kernel(const int* __restrict__ plan, int E, int NT, int4* __restrict__ tiles) {
    moe_wg_tiles_body(plan, E, NT, tiles);
}

// X's rows in the order's: out [pairs, K] row i = x row order[i] / k (the pairs the plan routes: start[h] of them), 16 bytes a thread.
__device__ __forceinline__ void moe_wg_gather_body(const uint4* __restrict__ x, const int* __restrict__ plan, int E, int64_t k, int64_t K8, uint4* __restrict__ out) {
    const int* order = plan + 2 + 2 * E;
    const int64_t pairs = __ldg(plan + 1 + E + __ldg(plan));
    for (int64_t i = blockIdx.x; i < pairs; i += gridDim.x) {
        const uint4* src = x + (int64_t)(__ldg(order + i) / k) * K8;
        for (int64_t c = threadIdx.x; c < K8; c += blockDim.x) out[i * K8 + c] = __ldg(src + c);
    }
}

__global__ void moe_wg_gather_kernel(const uint4* __restrict__ x, const int* __restrict__ plan, int E, int64_t k, int64_t K8, uint4* __restrict__ out) {
    moe_wg_gather_body(x, plan, E, k, K8, out);
}

template <int NT, int ACT> struct Wm12 {
    static constexpr int THREADS = 288;  // two consumer warpgroups (a row block each), then the TMA warp
    static constexpr int XB = NT * 128;  // X's tile a stage: NT pairs by 64 columns
    static constexpr int CB = 4 * (int)STEP12, EB = 512, BB = 64;  // a row block's steps a stage, its exceptions (up to 128), their bounds and the unit's record
    static constexpr int RBB = CB + EB + BB;
    static constexpr int SLOT = (XB + 2 * RBB + 1023) / 1024 * 1024;
    static constexpr int XCH = ACT ? NT * 256 : 0;  // the gate's and the up's sums exchanged: 2 halves of NT / 16 float4s a thread of a warpgroup
    static constexpr int SCR = 8 * 1024;            // the consumer warps' exceptions' scratch, 1 KB each
    static constexpr int NSF = (227 * 1024 - 1024 - SCR - XCH - 256) / SLOT;
    static constexpr int NS = NSF < 8 ? NSF : 8;    // stages in flight
    static constexpr int SHARED = NS * SLOT + 16 * NS + SCR + XCH + 1024;  // the barriers, the scratch, the exchange, the alignment
};

// A unit's sums out of the registers (the kernel's last step; a test's alone, on any GPU: moe_wg_epilogue_test_kernel): d as wgmma leaves
// it, thread ct of the two warpgroups (256 threads, named barrier 1) holding rows 16w + g (+ 8) of its row block and pairs 8jn + 2t (+ 1)
// of the tile as d[4jn + 2h + c]; info: the tile's first row of the order, its pairs, its expert, the unit's row unit.
template <int NT, int ACT>
__device__ __forceinline__ void wm_epilogue(float (&d)[NT / 2], const int4 info, int ct, int64_t O, int E, const int* __restrict__ plan, float4* xch, const __nv_bfloat16* __restrict__ bias, const void* __restrict__ wts, int wf32, __nv_bfloat16* __restrict__ Y, float* __restrict__ Y32) {
    const int wg = ct >> 7, w = (ct >> 5) & 3, g = (ct & 31) >> 2, t = ct & 3;
    const int64_t RB = O / 64;
    const int* order = plan + 2 + 2 * E;
    const int row0 = info.x, np = info.y, ex = info.z, ru = info.w;
    if constexpr (ACT) {
        // The gate (warpgroup 0) and the up (1) of the same rows and pairs sit in the same thread of each: the first half of the
        // pairs is finished by warpgroup 0 (it takes the up's from 1), the second by 1 (the gate's from 0), through xch: [the
        // first half's | the second's][NT / 16 float4s][128 threads]. (d is indexed by constants alone: both halves are
        // read and one chosen.)
        constexpr int HALF = NT / 16;
        const int cl = ct & 127;
#pragma unroll
        for (int jh = 0; jh < HALF; jh++) {
            const float4 lo = make_float4(d[4 * jh], d[4 * jh + 1], d[4 * jh + 2], d[4 * jh + 3]);  // 1 gives the up's of the first half,
            const float4 hi = make_float4(d[4 * (HALF + jh)], d[4 * (HALF + jh) + 1], d[4 * (HALF + jh) + 2], d[4 * (HALF + jh) + 3]);  // 0 the gate's of the second
            xch[((wg ? 0 : HALF) + jh) * 128 + cl] = wg ? lo : hi;
        }
        bar_sync<256>(1);
#pragma unroll
        for (int jh = 0; jh < HALF; jh++) {
            const int jn = wg ? HALF + jh : jh;
            const float mine[4] = {wg ? d[4 * (HALF + jh)] : d[4 * jh], wg ? d[4 * (HALF + jh) + 1] : d[4 * jh + 1], wg ? d[4 * (HALF + jh) + 2] : d[4 * jh + 2], wg ? d[4 * (HALF + jh) + 3] : d[4 * jh + 3]};
            const float4 o4 = xch[((wg ? HALF : 0) + jh) * 128 + cl];  // the other's, for the same rows and pairs
            const float other[4] = {o4.x, o4.y, o4.z, o4.w};
#pragma unroll
            for (int h = 0; h < 2; h++)
#pragma unroll
                for (int c = 0; c < 2; c++) {
                    const int m = 8 * jn + 2 * t + c;
                    if (m < np) {
                        const int64_t o = (int64_t)ru * 64 + 16 * w + g + 8 * h, at = (int64_t)ex * O + o;
                        const float gate = (wg ? other[2 * h + c] : mine[2 * h + c]) + (bias ? __bfloat162float(bias[at]) : 0.f);
                        const float up = (wg ? mine[2 * h + c] : other[2 * h + c]) + (bias ? __bfloat162float(bias[at + O / 2]) : 0.f);
                        Y[(int64_t)(row0 + m) * (O / 2) + o] = __float2bfloat16(silu(gate) * up);
                    }
                }
        }
        bar_sync<256>(1);  // the exchange is free for the next unit
    } else {
        const int64_t rb = 2 * (int64_t)ru + wg;
        if (rb < RB) {
#pragma unroll
            for (int jn = 0; jn < NT / 8; jn++)
#pragma unroll
                for (int c = 0; c < 2; c++) {
                    const int m = 8 * jn + 2 * t + c;
                    if (m < np) {
                        const int64_t i = row0 + m, jp = Y32 ? __ldg(order + i) : 0;
                        const float wj = Y32 ? (wf32 ? ((const float*)wts)[jp] : __bfloat162float(((const __nv_bfloat16*)wts)[jp])) : 1.f;
#pragma unroll
                        for (int h = 0; h < 2; h++) {
                            const int64_t o = rb * 64 + 16 * w + g + 8 * h;
                            const float y = d[4 * jn + 2 * h + c] + (bias ? __bfloat162float(bias[(int64_t)ex * O + o]) : 0.f);
                            if (Y32) Y32[jp * O + o] = y * wj;
                            else Y[i * O + o] = __float2bfloat16(y);
                        }
                    }
                }
        }
    }
}

// The kernel's two loops, as functions of what each stage does (the kernel's: copies and products; moe_wg_walk_test_kernel's: signatures written and
// checked, on any GPU).
// A block's units: block b takes units b, b + nb, ... of RU row units by the tiles; `my` of them.
__device__ __forceinline__ int wm_units(const int4* __restrict__ tiles, int RU, int nb) {
    const int64_t total = (int64_t)RU * __ldg(&tiles[0].x);
    return total > (int64_t)blockIdx.x ? (int)((total - blockIdx.x + nb - 1) / nb) : 0;
}

// The first steps (in the pack's order) of the unit's two row blocks (past O: the last one's, its products not written; with the gate: the
// gate's block ru and the up's RB / 2 + ru), the tile's record (expert, first row of the order, pairs, 1 where the expert's pairs are one chunk).
template <int ACT>
__device__ __forceinline__ int64_t wm_row_block(int64_t RB, int ru, int r) { return ACT ? (r ? RB / 2 + ru : ru) : min((int64_t)2 * ru + r, RB - 1); }

// The TMA warp's loop over a block's stages: lanes 0-9 load the next stage's bounds (row block lane / 5, exc_base at its step lane mod 5 of
// the 4 and the next) while lane 0 issues this one; the next unit's record is loaded a unit ahead. wait_empty(slot, parity): until the
// consumers are done with the stage before in the slot; issue(j, slot, s0, tile record, row unit, the row blocks' first steps, bounds).
template <int ACT, int NS, class Wait, class Issue>
__device__ __forceinline__ void wm_produce(const int4* __restrict__ tiles, const int* __restrict__ exc_base, int64_t RB, int64_t KS, int RU, int S, int nb, int my, int lane, Wait wait_empty, Issue issue) {
    const int n = my * S;
    int uk = 0, s = 0, ru = 0;
    int4 rec = make_int4(0, 0, 0, 0), recn = rec;
    int64_t b0 = 0, b1 = 0;  // the unit's two row blocks' first steps
    auto tile_of = [&](int k, int& row_unit) {  // the tile's index of this block's unit k, its row unit in row_unit
        const int64_t u = blockIdx.x + (int64_t)k * nb, q = u / RU;
        row_unit = (int)(u - q * RU);
        return (int)q;
    };
    if (n > 0) {
        int dummy;
        rec = __ldg(tiles + 1 + tile_of(0, ru));
        if (my > 1) recn = __ldg(tiles + 1 + tile_of(1, dummy));
        b0 = ((int64_t)rec.x * RB + wm_row_block<ACT>(RB, ru, 0)) * KS, b1 = ((int64_t)rec.x * RB + wm_row_block<ACT>(RB, ru, 1)) * KS;
    }
    auto bound = [&]() { return lane < 10 ? __ldg(exc_base + (lane < 5 ? b0 : b1) + 4 * s + lane % 5) : 0; };
    int ev = n > 0 ? bound() : 0;
    for (int j = 0; j < n; j++) {
        int e[10];
#pragma unroll
        for (int i = 0; i < 10; i++) e[i] = __shfl_sync(FULL, ev, i);
        const int sl = j % NS, s0 = s, cru = ru;
        const int4 cur = rec;
        const int64_t cb0 = b0, cb1 = b1;
        if (++s == S) {
            s = 0;
            if (++uk < my) {
                rec = recn;
                tile_of(uk, ru);
                if (uk + 1 < my) {
                    int dummy;
                    recn = __ldg(tiles + 1 + tile_of(uk + 1, dummy));
                }
                b0 = ((int64_t)rec.x * RB + wm_row_block<ACT>(RB, ru, 0)) * KS, b1 = ((int64_t)rec.x * RB + wm_row_block<ACT>(RB, ru, 1)) * KS;
            }
        }
        if (j + 1 < n) ev = bound();
        if (lane == 0) {
            if (j >= NS) wait_empty(sl, (j / NS - 1) & 1);  // the consumers are done with stage j - NS
            issue(j, sl, s0, cur, cru, cb0, cb1, e);
        }
        __syncwarp();
    }
}

// The consumers' loop: a unit's S stages in pairs (stage_a: even ones, with `first` set for a unit's first; stage_b: odd ones), an odd one
// after them, then finish() (every path gives each set of A registers to the products before it is decoded into again: a pair's second
// stage skipped inside the loop had ptxas serialize every product, its C7513).
template <class StageA, class StageB, class Finish>
__device__ __forceinline__ void wm_consume(int my, int S, StageA stage_a, StageB stage_b, Finish finish) {
    for (int j = 0, u = 0; u < my; u++, j += S) {
        int k = 0;
        for (; k + 1 < S; k += 2) {
            stage_a(j + k, k == 0);
            stage_b(j + k + 1);
        }
        if (k < S) stage_a(j + k, k == 0);
        finish();
    }
}

template <int NT, int ACT>
__device__ __forceinline__ void mma12_moe_wg_body(const CUtensorMap& xmap, Nib f, int64_t O, int64_t K, int E, const int* __restrict__ plan, const int4* __restrict__ tiles, int RU, const __nv_bfloat16* __restrict__ bias, const void* __restrict__ wts, int wf32, __nv_bfloat16* __restrict__ Y, float* __restrict__ Y32) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    using C = Wm12<NT, ACT>;
    extern __shared__ __align__(128) uint8_t smem_raw[];  // NS slots [X tile | a row block's steps, exceptions, bounds and record | the other's], the barriers, the scratch, the exchange
    uint32_t raw = (uint32_t)__cvta_generic_to_shared(smem_raw), base = (raw + 1023) & ~1023u;
    uint8_t* gbase = smem_raw + (base - raw);
    uint32_t full = base + C::NS * C::SLOT, empty = full + 8 * C::NS;
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int64_t KS = K / 16, RB = O / 64;
    const int S = (int)(K / 64), nb = gridDim.x;
    // The units: row units by the tiles (a tile's row units together), block b taking units b, b + nb, ...
    const int my = wm_units(tiles, RU, nb);  // this block's units
    if (tid == 0) {
        for (int i = 0; i < C::NS; i++) {
            mbar_init(full + 8 * i, 1);
            mbar_init(empty + 8 * i, 8);  // each consumer warp
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();
    if (warp == 8) {
        wm_produce<ACT, C::NS>(
            tiles, f.exc_base, RB, KS, RU, S, nb, my, lane, [&](int sl, uint32_t parity) { mbar_wait(empty + 8 * sl, parity); },
            [&](int, int sl, int s0, const int4& cur, int cru, int64_t cb0, int64_t cb1, const int (&e)[10]) {
                uint32_t fb = full + 8 * sl, xs = base + sl * C::SLOT, ws = xs + C::XB;
                uint32_t bytes = C::XB + 2 * C::CB;
                int a[2], na[2];
#pragma unroll
                for (int r = 0; r < 2; r++) {
                    exc_copy(e[5 * r], e[5 * r + 4], C::EB, a[r], na[r]);
                    bytes += na[r] > 0 ? 4 * na[r] : 0;
                    int* bs = (int*)(gbase + sl * C::SLOT + C::XB + r * C::RBB + C::CB + C::EB);
#pragma unroll
                    for (int i = 0; i < 5; i++) bs[i] = e[5 * r + i];
                    bs[5] = na[r] < 0 ? -1 : a[r];
                    bs[6] = cur.y, bs[7] = cur.z, bs[8] = cur.x, bs[9] = cru;  // the tile's first row, its pairs, the expert, the row unit
                }
                mbar_expect_tx(fb, bytes);
                tma_2d(xs, &xmap, s0 * 64, cur.y, fb);
#pragma unroll
                for (int r = 0; r < 2; r++) {
                    uint32_t cs = ws + r * C::RBB;
                    bulk_g2s(cs, f.data + ((r ? cb1 : cb0) + 4 * s0) * STEP12, C::CB, fb, cur.w != 0);  // (read once but where the expert's chunks share it)
                    if (na[r] > 0) bulk_g2s(cs + C::CB, f.exc + a[r], 4 * na[r], fb);
                }
            });
        return;
    }
    // Consumer warpgroup wg: row block 2 ru + wg (with the gate: ru, and RB / 2 + ru for the up) of each unit; warp w its rows 16w to 16w + 15.
    int ct = tid, wg = ct >> 7, w = (ct >> 5) & 3;
    uint32_t* xw = (uint32_t*)(gbase + C::NS * C::SLOT + 16 * C::NS) + 256 * warp;  // this warp's exceptions' scratch
    ((uint4*)xw)[2 * lane] = ((uint4*)xw)[2 * lane + 1] = make_uint4(0u, 0u, 0u, 0u);
    __syncwarp();
    float4* xch = (float4*)(gbase + C::NS * C::SLOT + 16 * C::NS + C::SCR);  // the exchange (ACT)
    float d[NT / 2] = {};  // rows 16w + g (+ 8), pairs 8jn + 2t (+ 1): d[4jn + 2h + c]; set by a unit's first product
    // (zeroed all the same: left unset, CUDA 12.8's ptxas saw it defined inside the loop and serialized every wgmma, its C7515)
    uint32_t A0[4][4], A1[4][4];
    int held = -1;  // the slot whose products may still be running
    int4 info = make_int4(0, 0, 0, 0);  // the unit's tile as its last stage had it: first row, pairs, expert, row unit
    auto release = [&](int sl) {  // the slot is free for stage j + NS (lane 0 arrives, by predicate: no branch among the products)
        __syncwarp();
        asm volatile("{\n.reg .pred p;\nsetp.eq.u32 p, %1, 0;\n@p mbarrier.arrive.shared::cta.b64 _, [%0];\n}\n" ::"r"(empty + 8 * sl), "r"(lane) : "memory");
    };
    auto keep_d = [&]() {  // d stays in its registers while the products run
#pragma unroll
        for (int i = 0; i < NT / 2; i++) asm volatile("" : "+f"(d[i])::"memory");
    };
    auto keep = [&](uint32_t(&A)[4][4]) {  // A is held until its products are done
#pragma unroll
        for (int kk = 0; kk < 4; kk++)
#pragma unroll
            for (int i = 0; i < 4; i++) asm volatile("" : "+r"(A[kk][i])::"memory");
    };
    auto stage = [&](int j, bool first, uint32_t(&A)[4][4], uint32_t(&Aprev)[4][4]) {
        int sl = j % C::NS;
        mbar_wait(full + 8 * sl, (j / C::NS) & 1);
        const uint8_t* sp = gbase + sl * C::SLOT + C::XB + wg * C::RBB;
        const int* bs = (const int*)(sp + C::CB + C::EB);
        int eb[5];
#pragma unroll
        for (int i = 0; i < 5; i++) eb[i] = bs[i];
        info = make_int4(bs[6], bs[7], bs[8], bs[9]);
        decode12_rows(f, sp, (const uint32_t*)(sp + C::CB), eb, bs[5], lane, w, xw, A);
        uint32_t xs = base + sl * C::SLOT;
        uint64_t desc[4];
#pragma unroll
        for (int kk = 0; kk < 4; kk++) desc[kk] = sw128_desc(xs + kk * 32);
        keep_d();
        wg_fence();
        wgmma4_rs(d, A, desc, !first);
        wg_wait<1>();  // the stage before's products are done: its A set and its slot are free
        keep(Aprev);
        keep_d();
        if (held >= 0) release(held);
        held = sl;
    };
    wm_consume(
        my, S, [&](int j, bool first) { stage(j, first, A0, A1); }, [&](int j) { stage(j, false, A1, A0); },
        [&]() {
            wg_wait<0>();
            keep(A0);
            keep(A1);
            keep_d();
            if (held >= 0) release(held);
            held = -1;
            wm_epilogue<NT, ACT>(d, info, ct, O, E, plan, xch, bias, wts, wf32, Y, Y32);
        });
#elif defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    __trap();  // sm_90a code, launched from a build without it (or on AMD)
#endif
}

template <int NT, int ACT>
__global__ void __launch_bounds__((Wm12<NT, ACT>::THREADS), 1) mma12_moe_wg_kernel(const __grid_constant__ CUtensorMap xmap, Nib f, int64_t O, int64_t K, int E, const int* __restrict__ plan, const int4* __restrict__ tiles, int RU, const __nv_bfloat16* __restrict__ bias, const void* __restrict__ wts, int wf32, __nv_bfloat16* __restrict__ Y, float* __restrict__ Y32) {
    mma12_moe_wg_body<NT, ACT>(xmap, f, O, K, E, plan, tiles, RU, bias, wts, wf32, Y, Y32);
}
