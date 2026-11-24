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
       ST_RING_HEAD = 12, ST_RAW_RING_HEAD = 13, ST_RING = 16 };

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
__global__ void __launch_bounds__(32) release_kernel(uint8_t* pool, int64_t page_stride, int64_t nblocks, const int32_t* blocks, int H, EscapePool esc, RawPool raw, uint32_t layer_tag, uint32_t* status,
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

// The write kernel: a CTA a token. A token that is the only one of its run goes through the decode step's path (the rest of this kernel); a longer run's first token runs write_run; the
// run's other tokens return at once. (P.lean 0: write_run for every run.)
__global__ void __launch_bounds__(256) write_kernel(const __grid_constant__ WriteParams P) {
    const int i = blockIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t mine = P.slots[i];
    const int64_t prv = i > 0 ? P.slots[i - 1] : -1, nxt = i + 1 < P.N ? P.slots[i + 1] : -1;
    KOne ko;
    VOne vo;
    KCalc kc;
    VCalc vc;
    if (warp < P.H) {  // (the first round's head: its inputs do not wait for the slot)
        k_one_in(ko, P.key + (int64_t)i * P.key_stride + warp * D, P.kbase + warp * D, lane);
        if (!P.one) v_one_in(vo, P.value + (int64_t)i * P.value_stride + warp * D, P.vbase + warp * D, lane);
    }
    if (mine < 0) return;
    if (i > 0 && prv >= 0 && prv + 1 == mine && (mine & 15) != 0) return;  // (not the first token of its run)
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

// ---- reading pages back to bf16: out [page][token][head][channel] for the pages listed (one warp a head-page) -----------------
__global__ void __launch_bounds__(128) unpack_kernel(const uint8_t* __restrict__ pool, int64_t page_stride, const int32_t* __restrict__ pages, int n, const uint8_t* __restrict__ kbase,
                                                     const uint8_t* __restrict__ vbase, int H, int A, const uint32_t* __restrict__ ent, int nc, const RawPool R, uint16_t* __restrict__ out_k, uint16_t* __restrict__ out_v, int one) {
    const int wl = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (wl >= n * H) return;
    const int lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
    const int j = wl / H, h = wl % H;
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

// ---- decoding pages into a paged bf16 cache, as vLLM's FlashAttention reads one ----------------------------------------------------
// The pages listed (pages[j]) decoded into the block dst[j] of `out` [blocks][16 tokens][H][2 D] bf16: token t of head h of block b has its keys at ((b 16 + t) H + h) 2 D and its
// values D after (the cache the prefill rows' FlashAttention runs on: a context's pages made bf16 before the step's own tokens are put after it). One warp a head-page, as unpack.
// HALVES 2: a head of 256 channels, held as the page's heads 2 h and 2 h + 1 (its channels 0-127 and 128-255): the head-page 2 h + f's keys at channels 128 f of head h's 256, its values 256 after.
template <int HALVES>
__global__ void __launch_bounds__(128) gather_kernel(const uint8_t* __restrict__ pool, int64_t page_stride, const int32_t* __restrict__ pages, const int32_t* __restrict__ dst, int n,
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
