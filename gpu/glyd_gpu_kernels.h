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
