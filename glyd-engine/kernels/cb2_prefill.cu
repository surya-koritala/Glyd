// GLM-5.3-Flash's routed experts (the `glyd-hbr-codes/2` pack: two-bit codebook rows with no run scales, src/cpu_cb.rs has the layout and the arithmetic) for the tokens of a prompt chunk, on the tensor cores (included by
// engine.cu after moe_hb.cu, whose tile helpers it uses). One expert is multiplied by all of its chunk's tokens at once, its codes read once and made bf16 in registers on the way to the tiles of shared memory: nothing is
// decoded to memory first.
//
// An expert's bytes in a slot of the ring (src/cb2_prefill.rs) are the pack's: codes [downT 2048 rows x 1024 bytes | up 4096 rows x 1024 bytes] and books [downT 2048 x 4 bf16 | up 4096 x 4 bf16]; row r's weight j is
// book[r][c], c the two bits at 2 (j % 4) of byte j / 4. The weights are bf16 numbers as they are, the tokens' rows x' = R x (bf16) and the activations (bf16) are what the products take, with fp32 sums.
//
// Formats: the up and the down matrix of an expert are each two-bit (cb2), three-bit (cb3) or four-bit (cb4) rows, or trellis rows (B = 5, 6, 7: 2, 3, 4 bits a weight, a row's scale for its table), the kernels being templates on the bits; a ternary row (five codes a byte) is first made two-bit rows by c2_tern_cb2
// (the table's three entries and a zero), so that no kernel reads base-3 digits. A matrix's place in the ring is given by byte offsets (C2Segs), its bytes the pack's: rows of 512 B bytes of codes (B bits a code) and 2^B bf16 entries.
// c2_up (a block: BM = 64 pairs of one expert, 64 neurons = 64 gate rows and the 64 up rows of the same neurons, K = 4,096 in steps of 64): the first product, and then the clamped SwiGLU in the registers, act[row][neuron] =
// bf16(silu(min(g, limit)) * clamp(u, -limit, limit)) with silu(x) = x / (1 + exp(-x)), written to act [hoff + row][2048]. A is the tokens' rows of x' (a pair's token from `tok`), by cp.async; B is the 128 rows' codes, a row a
// thread, 16 bytes (64 codes) a step through registers, one prmt a pair of weights, stored in the layout ldmatrix reads (the rows' chunks XORed by the row's low bits).
// c2_down (BM = 64 pairs, 128 outputs, K = 2,048 neurons): act [row][neuron] times the downT rows (a neuron's row of 4,096 outputs: 32 bytes of codes cover a tile's 128), B in shared memory as [neuron][output] and read
// by ldmatrix.trans; each product is multiplied by its pair's weight and added to yrot [token][4096] (fp32) with an atomic add: a token's pairs come from several experts, so the order of the sums is not fixed (the
// sums differ from run to run at the rounding of fp32).
// The grids are (tiles of the launch, 32); a tile belongs to a segment (an expert's rows, or a part of them) of C2Segs. Three and four bits: a weight is looked up in its row's table in shared memory (the row's entries
// and a thread's row: the rows of the first product are the threads', the second product's are a K-step's 64).

#define C2_HID 4096
#define C2_INT 2048

struct C2Segs {
    int n;
    int begin[16];  // the segment's first pair in the plan (tok, wt)
    int rows[16];   // its pairs
    int hoff[16];   // its first row in act
    int tiles[17];  // prefix of tiles (BM rows each)
    long long ucode[16];  // where the expert's up matrix is: its codes' offset in the codes of the ring ...
    long long dcode[16];  // ... the downT matrix's codes ...
    long long ubook[16];  // ... and the offsets of their tables in the ring's tables
    long long dbook[16];
    int ubits[16];        // the bits of the segment's up and down codes (the kernels of every width at once: c2qx_*)
    int dbits[16];
};

// ------------------------------------------------------------------------------------------------------------- trellis-coded rows (qx_tq_*)
// A row of T weights in the rotated basis (T a multiple of 512) as a tail-biting bitshift trellis at 2 bits a weight: a circular stream of
// 2T bits, most significant first in 32-bit words; weight t's state is the TQ_L bits of the stream from bit 2t on (around the end), its
// value the row's scale times tq_val(state): a multiply-add hash of the state and the sum of its four bytes, centred and of unit variance.
#define TQ_L 14
#define TQ_S (1 << TQ_L)
#define TQ_Q (TQ_S >> 2)          // groups of successors: states that share their 12 high bits
#define TQ_K 0.0067658672f         // 1 / the standard deviation of the sum of four uniform bytes, sqrt(4 (256^2 - 1) / 12)
#define TQ_C 3.4505923f            // 510 times that

__device__ __forceinline__ float tq_val(unsigned s, unsigned a, unsigned b) {
    return fmaf((float)__dp4a(s * a + b, 0x01010101u, 0u), TQ_K, -TQ_C);
}

// The runtime's hash constants (src/cpu_simd.rs TQ_A, TQ_B; a pack of other constants is refused when it is opened), and a state's value before
// its row's scale and TQ_K: the byte sum less 510 (the accumulator's -510 wraps).
#define TQ_HA 1536988885u
#define TQ_HB 2891336453u
__device__ __forceinline__ float tq_v(unsigned s) {
    return (float)(int)__dp4a(s * TQ_HA + TQ_HB, 0x01010101u, 0xFFFFFE02u);
}

// The state of weight t of a row (cr: its T / 16 words).
__device__ __forceinline__ unsigned tq_state(const unsigned* cr, int W, int t) {
    const int p = 2 * t, i = p >> 5;
    return __funnelshift_l(cr[i + 1 == W ? 0 : i + 1], cr[i], p & 31) >> (32 - TQ_L);
}

// Format q at K bits a weight (q3, q4: 3 and 4): weight t's state is the TQ_L bits of the circular stream from bit K t (cr: the row's W = K T / 32 words).
template <int K>
__device__ __forceinline__ unsigned tq_state_k(const unsigned* cr, int W, int t) {
    const int p = K * t, i = p >> 5;
    return __funnelshift_l(cr[i + 1 == W ? 0 : i + 1], cr[i], p & 31) >> (32 - TQ_L);
}

// The value (before the row's scale and TQ_K) of the weight whose state starts at bit p of the words w (its 14 bits within them).
__device__ __forceinline__ float tq_vk(const unsigned* w, int p) {
    return tq_v(__funnelshift_l(w[(p >> 5) + 1], w[p >> 5], p & 31) >> (32 - TQ_L));
}

// two weights of the codes at pair p of w (codes 2p and 2p + 1: bits 4p to 4p + 3), the lower one in the low half: prmt picks the bytes of the book's entries (r01: entries 0 and 1, r23: 2 and 3), the selector from the
// codes (nibbles 2c, 2c + 1 for each weight's code c)
__device__ __forceinline__ uint32_t c2_w2(uint32_t w, int p, uint32_t r01, uint32_t r23) {
    const uint32_t v = (w >> (4 * p)) & 15u;
    return __byte_perm(r01, r23, ((v * 65u) & 0x303u) * 34u + 0x1010u);
}

// 8 weights: codes 8 half .. 8 half + 7 of w
__device__ __forceinline__ uint4 c2_chunk(uint32_t w, int half, uint32_t r01, uint32_t r23) {
    uint4 o;
    o.x = c2_w2(w, 4 * half, r01, r23);
    o.y = c2_w2(w, 4 * half + 1, r01, r23);
    o.z = c2_w2(w, 4 * half + 2, r01, r23);
    o.w = c2_w2(w, 4 * half + 3, r01, r23);
    return o;
}

__device__ __forceinline__ void c2_sts16(uint32_t a, const uint4& v) {
#if defined(__HIP_PLATFORM_AMD__)
    glyd_lds_put16(a, v);
#else
    asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" ::"r"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
#endif
}

// The 64 codes of a thread's chunk of a row: 2 B words of B bits a code, 8 B bytes (16, 24 or 32).
template <int B>
struct C2Cw {
    uint32_t w[2 * B];
};

// B = 5: a trellis row's (src/qx_trellis.rs) 64 weights of a chunk, its words j .. j + 3 and the next (around the row's end) for the windows
template <>
struct C2Cw<5> {
    uint32_t w[5];
};

// a trellis row's (`row`, W words) chunk from word j (a multiple of 4)
__device__ __forceinline__ C2Cw<5> c2_load_tq(const unsigned char* row, int j, int W) {
    C2Cw<5> c;
    const uint4 v = *reinterpret_cast<const uint4*>(row + 4 * j);
    c.w[0] = v.x; c.w[1] = v.y; c.w[2] = v.z; c.w[3] = v.w;
    c.w[4] = reinterpret_cast<const uint32_t*>(row)[j + 4 < W ? j + 4 : j + 4 - W];
    return c;
}

// B = 6, 7: format q at 3 and 4 bits a weight (q3, q4), a chunk's 64 weights in 6 or 8 words and the next (around the row's end)
template <>
struct C2Cw<6> {
    uint32_t w[7];
};
template <>
struct C2Cw<7> {
    uint32_t w[9];
};

// such a row's (`row`, W words) chunk from word j (a multiple of 2 words at 3 bits, 8 at 4: 8- and 16-byte loads)
template <int B>
__device__ __forceinline__ C2Cw<B> c2_load_tqk(const unsigned char* row, int j, int W) {
    constexpr int N = 2 * (B - 3);
    C2Cw<B> c;
    if constexpr (B == 7) {
        const uint4 a = *reinterpret_cast<const uint4*>(row + 4 * j), b = *reinterpret_cast<const uint4*>(row + 4 * j + 16);
        c.w[0] = a.x; c.w[1] = a.y; c.w[2] = a.z; c.w[3] = a.w; c.w[4] = b.x; c.w[5] = b.y; c.w[6] = b.z; c.w[7] = b.w;
    } else {
#pragma unroll
        for (int q = 0; q < N / 2; q++) {
            const uint2 v = *reinterpret_cast<const uint2*>(row + 4 * j + 8 * q);
            c.w[2 * q] = v.x;
            c.w[2 * q + 1] = v.y;
        }
    }
    c.w[N] = reinterpret_cast<const uint32_t*>(row)[j + N < W ? j + N : j + N - W];
    return c;
}

template <int B>
__device__ __forceinline__ C2Cw<B> c2_load(const unsigned char* p) {
    C2Cw<B> c;
    if constexpr (B == 2) {
        const uint4 v = *reinterpret_cast<const uint4*>(p);
        c.w[0] = v.x; c.w[1] = v.y; c.w[2] = v.z; c.w[3] = v.w;
    } else if constexpr (B == 3) {
        const uint2 a = *reinterpret_cast<const uint2*>(p), b = *reinterpret_cast<const uint2*>(p + 8), d = *reinterpret_cast<const uint2*>(p + 16);
        c.w[0] = a.x; c.w[1] = a.y; c.w[2] = b.x; c.w[3] = b.y; c.w[4] = d.x; c.w[5] = d.y;
    } else {
        const uint4 a = *reinterpret_cast<const uint4*>(p), b = *reinterpret_cast<const uint4*>(p + 16);
        c.w[0] = a.x; c.w[1] = a.y; c.w[2] = a.z; c.w[3] = a.w; c.w[4] = b.x; c.w[5] = b.y; c.w[6] = b.z; c.w[7] = b.w;
    }
    return c;
}

// the bits of the codes 8 c .. 8 c + 7 of the chunk (c = 0 to 7): 24 bits at three, a word at four
template <int B>
__device__ __forceinline__ uint32_t c2_group(const C2Cw<B>& k, int c) {
    if constexpr (B == 4) {
        return k.w[c];
    } else {
        switch (c) {
            case 0: return k.w[0] & 0xFFFFFFu;
            case 1: return __funnelshift_r(k.w[0], k.w[1], 24) & 0xFFFFFFu;
            case 2: return __funnelshift_r(k.w[1], k.w[2], 16) & 0xFFFFFFu;
            case 3: return k.w[2] >> 8;
            case 4: return k.w[3] & 0xFFFFFFu;
            case 5: return __funnelshift_r(k.w[3], k.w[4], 24) & 0xFFFFFFu;
            case 6: return __funnelshift_r(k.w[4], k.w[5], 16) & 0xFFFFFFu;
            default: return k.w[5] >> 8;
        }
    }
}

// a row's table in shared memory: ENT bf16 numbers, rows ENT + 2 apart (the odd number of 4-byte words keeps the rows of a warp's threads in different banks)
template <int B>
struct C2Tab {
    static constexpr int ENT = 1 << B;
    static constexpr int STRIDE = ENT + 2;
};

// 8 weights of the codes `bits` (B bits each) looked up in the row's table `tab` (u16 numbers)
template <int B>
__device__ __forceinline__ uint4 c2_lookup8(const unsigned short* tab, uint32_t bits) {
    constexpr uint32_t M = (1u << B) - 1;
    uint32_t v[8];
#pragma unroll
    for (int m = 0; m < 8; m++) v[m] = tab[(bits >> (B * m)) & M];
    return make_uint4(v[0] | v[1] << 16, v[2] | v[3] << 16, v[4] | v[5] << 16, v[6] | v[7] << 16);
}

// the 64 codes of a thread's chunk as 8 chunks of 8 weights into shared memory at row `row` of a tile of ROWLEN elements a row, the thread's chunks c0 .. c0 + 7 of the row; two bits through the registers' table
// (r01, r23), three and four through the row's table in shared memory
template <int ROWLEN, int B>
__device__ __forceinline__ void c2_store(uint32_t base, int row, int c0, const C2Cw<B>& cw, uint32_t r01, uint32_t r23, const unsigned short* tab) {
#pragma unroll
    for (int c = 0; c < 8; c++) {
        uint4 o;
        if constexpr (B == 2) {
            o = c2_chunk(cw.w[c >> 1], c & 1, r01, r23);
        } else if constexpr (B == 5) {
            const float sc = __uint_as_float(r01);
            const uint32_t hi = cw.w[c >> 1], lo = cw.w[(c >> 1) + 1];
            uint32_t p[4];
#pragma unroll
            for (int m = 0; m < 4; m++) {
                const int i = 8 * (c & 1) + 2 * m;
                p[m] = pack_bf16(tq_v(__funnelshift_l(lo, hi, 2 * i) >> (32 - TQ_L)) * sc, tq_v(__funnelshift_l(lo, hi, 2 * i + 2) >> (32 - TQ_L)) * sc);
            }
            o = make_uint4(p[0], p[1], p[2], p[3]);
        } else if constexpr (B >= 6) {
            const float sc = __uint_as_float(r01);
            uint32_t p[4];
#pragma unroll
            for (int m = 0; m < 4; m++) {
                const int i = 8 * c + 2 * m;
                p[m] = pack_bf16(tq_vk(cw.w, (B - 3) * i) * sc, tq_vk(cw.w, (B - 3) * (i + 1)) * sc);
            }
            o = make_uint4(p[0], p[1], p[2], p[3]);
        } else {
            o = c2_lookup8<B>(tab, c2_group<B>(cw, c));
        }
        c2_sts16(base + (uint32_t)(row * ROWLEN + (((c0 + c) ^ (row & 7)) << 3)) * 2u, o);
    }
}

// BM: the pairs a block takes, 64 (4 warps) or 128 (8 warps: the tile's codes made bf16 once for twice the rows, by the first 128 threads; a segment of many pairs)
template <int STAGES, int B, int HID = C2_HID, int INT = C2_INT, int BM = 64>
__device__ __forceinline__ void c2_up_t(const unsigned short* __restrict__ xrb, const int* __restrict__ tok, const unsigned char* __restrict__ rcodes, const unsigned char* __restrict__ rbooks, bf16* __restrict__ act,
                                        const C2Segs& sg, float limit, unsigned short* s_tab) {
    constexpr int BN = 128, BK = 64, WARPS_M = BM / 32, WARPS_N = 2, NTH = WARPS_M * WARPS_N * 32;
    constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N;  // 32 rows, 64 rows of B (32 gate and the same 32 neurons' up)
    constexpr int MT = WM / 16, NTL = WN / 8;
    constexpr int KS = BK / 16;
    constexpr int NK = HID / BK;
    constexpr int ROWB = HID / 8 * (B >= 5 ? B - 3 : B);  // a row's bytes of codes (a trellis row's: B - 3 bits a weight)
    using F = Fill<BK, NTH>;
    constexpr int RPI = F::RPI, PA = BM / RPI;
    constexpr uint32_t STAGE_A = BM * BK * 2, STAGE_B = BN * BK * 2;
    extern __shared__ __align__(128) unsigned char smem_raw[];
    const uint32_t sbase = s_addr(smem_raw);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp / WARPS_N, wn = warp % WARPS_N;
    const int tile = blockIdx.x;
    if (tile >= sg.tiles[sg.n]) return;
    int seg = 0;
    while (seg + 1 < sg.n && tile >= sg.tiles[seg + 1]) seg++;
    const int m0 = (tile - sg.tiles[seg]) * BM;
    const int rows = sg.rows[seg] - m0;
    const int j0 = blockIdx.y * 64;  // the tile's first neuron
    const int lr = F::row(tid), lc = F::chunk(tid);
    const uint32_t soff = (uint32_t)F::soff(tid) * 2;
    uint32_t goffA[PA];
    bool okA[PA];
#pragma unroll
    for (int p = 0; p < PA; p++) {
        const int r = lr + p * RPI;
        okA[p] = r < rows;
        goffA[p] = (uint32_t)tok[sg.begin[seg] + m0 + (okA[p] ? r : 0)] * (uint32_t)(HID * 2) + (uint32_t)lc * 16u;
    }
    const char* xb = reinterpret_cast<const char*>(xrb);
    // B: this thread's row of the tile (rows 64 wn .. 64 wn + 31 are gate neurons j0 + 32 wn .., rows 64 wn + 32 .. 63 the up neurons of the same), the first 128 threads'
    const bool dec = NTH == 128 || tid < 128;
    const int srow = tid & 127;
    const int grow = ((srow & 63) < 32 ? 0 : INT) + j0 + (srow >> 6) * 32 + (srow & 31);
    const unsigned char* crow = rcodes + sg.ucode[seg] + (size_t)grow * ROWB;
    uint32_t r01 = 0, r23 = 0;
    const unsigned short* mytab = nullptr;
    if constexpr (B == 2) {
        const uint2 bk = *reinterpret_cast<const uint2*>(rbooks + sg.ubook[seg] + (size_t)grow * 8);
        r01 = bk.x;
        r23 = bk.y;
    } else if constexpr (B >= 5) {
        r01 = __float_as_uint(b2f(*reinterpret_cast<const bf16*>(rbooks + sg.ubook[seg] + (size_t)grow * 2)) * TQ_K);
    } else {
        constexpr int ENT = C2Tab<B>::ENT, STR = C2Tab<B>::STRIDE;
        const uint32_t* src = reinterpret_cast<const uint32_t*>(rbooks + sg.ubook[seg] + (size_t)grow * ENT * 2);
        uint32_t* dst = reinterpret_cast<uint32_t*>(s_tab + srow * STR);
        if (dec) {
#pragma unroll
            for (int i = 0; i < ENT / 2; i++) dst[i] = src[i];
        }
        mytab = s_tab + srow * STR;
    }
    auto load_a = [&](int st, int kt) {
        const char* xk = xb + (size_t)kt * (BK * 2);
        const uint32_t da = sbase + st * STAGE_A + soff;
#pragma unroll
        for (int p = 0; p < PA; p++) cp16(da + p * RPI * BK * 2, xk + goffA[p], okA[p]);
    };
    auto store_b = [&](int st, const C2Cw<B>& cw) {
        if (dec) c2_store<BK, B>(sbase + STAGES * STAGE_A + st * STAGE_B, srow, 0, cw, r01, r23, mytab);
    };
    auto ldc = [&](int kt) {
        C2Cw<B> c{};
        if constexpr (B == 5) {
            if (dec) c = c2_load_tq(crow, 4 * kt, HID / 16);
        } else if constexpr (B >= 6) {
            if (dec) c = c2_load_tqk<B>(crow, 2 * (B - 3) * kt, HID * (B - 3) / 32);
        } else {
            if (dec) c = c2_load<B>(crow + (size_t)kt * (8 * B));
        }
        return c;
    };
    uint32_t aoff[KS], boff[KS];
#pragma unroll
    for (int ks = 0; ks < KS; ks++) {
        aoff[ks] = (uint32_t)frag_a_off<BK>(wm * WM, lane, ks) * 2;
        boff[ks] = (uint32_t)frag_b_off<BK>(wn * WN, lane, ks) * 2;
    }
    auto load_frags = [&](uint32_t (&af)[MT][4], uint32_t (&bfr)[NTL][2], int st, int ks) {
        const uint32_t a = sbase + st * STAGE_A + aoff[ks];
        const uint32_t b = sbase + STAGES * STAGE_A + st * STAGE_B + boff[ks];
#pragma unroll
        for (int mt = 0; mt < MT; mt++) ldsm4(af[mt], a + mt * 16 * BK * 2);
#pragma unroll
        for (int np = 0; np < NTL / 2; np++) {
            uint32_t t[4];
            ldsm4(t, b + np * 16 * BK * 2);
            bfr[2 * np][0] = t[0];
            bfr[2 * np][1] = t[1];
            bfr[2 * np + 1][0] = t[2];
            bfr[2 * np + 1][1] = t[3];
        }
    };
    if constexpr (B != 2 && B < 5) __syncthreads();  // (the tables)
    C2Cw<B> cw = ldc(0);
#pragma unroll
    for (int st = 0; st < STAGES - 1; st++) {
        load_a(st, st);
        cp_commit();
        store_b(st, cw);
        cw = ldc(st + 1);
    }
    float acc[MT][NTL][4];
#pragma unroll
    for (int i = 0; i < MT; i++)
#pragma unroll
        for (int j = 0; j < NTL; j++) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
    cp_wait<STAGES - 2>();
    __syncthreads();
    uint32_t af[2][MT][4], bfr[2][NTL][2];
    load_frags(af[0], bfr[0], 0, 0);
    int rd = 0, wr = STAGES - 1;
    for (int kt = 0; kt < NK; kt++) {
#pragma unroll
        for (int ks = 0; ks < KS; ks++) {
            const int cur = ks & 1, nxt = cur ^ 1;
            if (ks == KS - 1) {
                cp_wait<STAGES - 2>();
                __syncthreads();
                rd = rd + 1 == STAGES ? 0 : rd + 1;
            }
            load_frags(af[nxt], bfr[nxt], rd, (ks + 1) % KS);
            if (ks == 0) {
                if (kt + STAGES - 1 < NK) {
                    load_a(wr, kt + STAGES - 1);
                    store_b(wr, cw);
                    if (kt + STAGES < NK) cw = ldc(kt + STAGES);
                }
                cp_commit();
                wr = wr + 1 == STAGES ? 0 : wr + 1;
            }
#pragma unroll
            for (int mt = 0; mt < MT; mt++)
#pragma unroll
                for (int nt = 0; nt < NTL; nt++) mma16816(acc[mt][nt], af[cur][mt], bfr[cur][nt][0], bfr[cur][nt][1]);
        }
    }
    cp_wait<0>();
    // the activations: the warp's n-tiles 0 to 3 are its 32 gate neurons, 4 to 7 the same neurons' up
    const int g = lane >> 2, t4 = lane & 3;
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
        const int r0 = wm * WM + mt * 16 + g, r1 = r0 + 8;
#pragma unroll
        for (int nt = 0; nt < NTL / 2; nt++) {
            const int col = j0 + wn * 32 + nt * 8 + 2 * t4;
            float a[4];
#pragma unroll
            for (int q = 0; q < 4; q++) {
                const float gt = fminf(acc[mt][nt][q], limit), up = fminf(fmaxf(acc[mt][nt + NTL / 2][q], -limit), limit);
                a[q] = gt / (1.f + expf(-gt)) * up;
            }
            if (r0 < rows) *reinterpret_cast<uint32_t*>(act + (size_t)(sg.hoff[seg] + m0 + r0) * INT + col) = pack_bf16(a[0], a[1]);
            if (r1 < rows) *reinterpret_cast<uint32_t*>(act + (size_t)(sg.hoff[seg] + m0 + r1) * INT + col) = pack_bf16(a[2], a[3]);
        }
    }
}

// BN: the outputs a block takes, 128 (4 warps) or 256 (8 warps: a step's prologue and the pairs' rows of act read once for twice the outputs; Qwen3.8-Flash-Next's, whose K of 640 is ten steps)
template <int STAGES, int B, int HID = C2_HID, int INT = C2_INT, int BN = 128>
__device__ __forceinline__ void c2_down_t(const bf16* __restrict__ act, const int* __restrict__ tok, const float* __restrict__ wt, const unsigned char* __restrict__ rcodes, const unsigned char* __restrict__ rbooks,
                                          float* __restrict__ yrot, const C2Segs& sg, unsigned short* s_tab) {
    constexpr int BM = 64, BK = 64, WARPS_M = 2, WARPS_N = BN / 64, NTH = WARPS_M * WARPS_N * 32;
    constexpr int Q = NTH / 64;  // threads a row of B (each 64 of its outputs)
    constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N;  // 32 pairs, 64 outputs
    constexpr int MT = WM / 16, NTL = WN / 8;
    constexpr int KS = BK / 16;
    constexpr int NK = INT / BK;
    constexpr int ROWB = HID / 8 * (B >= 5 ? B - 3 : B);
    constexpr int ENT = 1 << (B == 2 || B >= 5 ? 3 : B), STR = ENT + 2;
    constexpr bool TAB = B != 2 && B < 5;  // (three and four bits: tables in shared memory; a trellis row's scale rides as the two-bit rows' entries do)
    using F = Fill<BK, NTH>;
    constexpr int RPI = F::RPI, PA = BM / RPI;
    constexpr uint32_t STAGE_A = BM * BK * 2, STAGE_B = BK * BN * 2;
    extern __shared__ __align__(128) unsigned char smem_raw[];
    // (s_tab: the tables of the K-steps' 64 rows, a set for each stage (three and four bits), the wrapper's)
    const uint32_t sbase = s_addr(smem_raw);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp / WARPS_N, wn = warp % WARPS_N;
    const int tile = blockIdx.x;
    if (tile >= sg.tiles[sg.n]) return;
    int seg = 0;
    while (seg + 1 < sg.n && tile >= sg.tiles[seg + 1]) seg++;
    const int m0 = (tile - sg.tiles[seg]) * BM;
    const int rows = sg.rows[seg] - m0;
    const int n0 = blockIdx.y * BN;  // the tile's first output
    const int mi = lane >> 3, r8 = lane & 7;
    const int lr = F::row(tid), lc = F::chunk(tid);
    const uint32_t soff = (uint32_t)F::soff(tid) * 2;
    uint32_t goffA[PA];
    bool okA[PA];
#pragma unroll
    for (int p = 0; p < PA; p++) {
        const int r = lr + p * RPI;
        okA[p] = r < rows;
        goffA[p] = (uint32_t)(sg.hoff[seg] + m0 + (okA[p] ? r : 0)) * (uint32_t)(INT * 2) + (uint32_t)lc * 16u;
    }
    const char* xb = reinterpret_cast<const char*>(act);
    // B: a thread takes the 64 codes of neuron k = tid % 64 over a Q-th of the tile's outputs (h = tid / 64)
    const int bk_row = tid & 63, bh = tid >> 6;
    const unsigned char* dcodes = rcodes + sg.dcode[seg] + (size_t)n0 * B / 8 + (size_t)bh * (8 * B);
    const unsigned char* dbooks = rbooks + sg.dbook[seg];
    auto ldc = [&](int kt) {
        if constexpr (B == 5) {
            return c2_load_tq(rcodes + sg.dcode[seg] + (size_t)(kt * BK + bk_row) * ROWB, (n0 + 64 * bh) / 16, HID / 16);
        } else if constexpr (B >= 6) {
            return c2_load_tqk<B>(rcodes + sg.dcode[seg] + (size_t)(kt * BK + bk_row) * ROWB, (n0 + 64 * bh) * (B - 3) / 32, HID * (B - 3) / 32);
        } else {
            return c2_load<B>(dcodes + (size_t)(kt * BK + bk_row) * ROWB);
        }
    };
    auto ldb = [&](int kt) {
        if constexpr (B >= 5) {
            return make_uint2(__float_as_uint(b2f(*reinterpret_cast<const bf16*>(dbooks + (size_t)(kt * BK + bk_row) * 2)) * TQ_K), 0u);
        } else {
            return *reinterpret_cast<const uint2*>(dbooks + (size_t)(kt * BK + bk_row) * 8);
        }
    };
    // (three and four bits: the thread's Q-th of its row's table, TW words, for a K-step; kept in registers a step and then put in the stage's tables)
    constexpr int TW = ENT / (2 * Q);
    auto ldt = [&](int kt) {
        uint4 t = make_uint4(0, 0, 0, 0);
        if constexpr (TAB) {
            const uint32_t* p = reinterpret_cast<const uint32_t*>(dbooks + (size_t)(kt * BK + bk_row) * ENT * 2) + bh * TW;
            if constexpr (TW == 1) {
                t.x = p[0];
            } else if constexpr (TW == 2) {
                const uint2 v = *reinterpret_cast<const uint2*>(p);
                t.x = v.x;
                t.y = v.y;
            } else {
                t = *reinterpret_cast<const uint4*>(p);
            }
        }
        return t;
    };
    auto stt = [&](int stage, const uint4& t) {
        if constexpr (TAB) {
            uint32_t* dst = reinterpret_cast<uint32_t*>(s_tab + (stage * 64 + bk_row) * STR) + bh * TW;
            dst[0] = t.x;
            if constexpr (TW >= 2) dst[1] = t.y;
            if constexpr (TW >= 4) {
                dst[2] = t.z;
                dst[3] = t.w;
            }
        }
    };
    auto load_a = [&](int st, int kt) {
        const char* xk = xb + (size_t)kt * (BK * 2);
        const uint32_t da = sbase + st * STAGE_A + soff;
#pragma unroll
        for (int p = 0; p < PA; p++) cp16(da + p * RPI * BK * 2, xk + goffA[p], okA[p]);
    };
    auto store_b = [&](int st, const C2Cw<B>& cw, const uint2& bk) {
        c2_store<BN, B>(sbase + STAGES * STAGE_A + st * STAGE_B, bk_row, bh * 8, cw, bk.x, bk.y, TAB ? reinterpret_cast<const unsigned short*>(s_tab + (st * 64 + bk_row) * STR) : nullptr);
    };
    // A: ldmatrix as the first product's; B is [neuron][output] (BN elements a row, the chunks XORed by the row's low bits): ldmatrix.trans, two n-tiles (16 outputs) a load
    uint32_t aoff[KS], brow[KS], bcol[NTL / 2];
#pragma unroll
    for (int ks = 0; ks < KS; ks++) {
        aoff[ks] = (uint32_t)frag_a_off<BK>(wm * WM, lane, ks) * 2;
        brow[ks] = (uint32_t)((ks * 16 + (mi & 1) * 8 + r8) * BN) * 2;
    }
#pragma unroll
    for (int np = 0; np < NTL / 2; np++) bcol[np] = (uint32_t)((((wn * WN) / 8 + 2 * np + (mi >> 1)) ^ r8) << 3) * 2;
    auto load_frags = [&](uint32_t (&af)[MT][4], uint32_t (&bfr)[NTL][2], int st, int ks) {
        const uint32_t a = sbase + st * STAGE_A + aoff[ks];
        const uint32_t b = sbase + STAGES * STAGE_A + st * STAGE_B + brow[ks];
#pragma unroll
        for (int mt = 0; mt < MT; mt++) ldsm4(af[mt], a + mt * 16 * BK * 2);
#pragma unroll
        for (int np = 0; np < NTL / 2; np++) {
            uint32_t t[4];
            ldsm4t(t, b + bcol[np]);
            bfr[2 * np][0] = t[0];
            bfr[2 * np][1] = t[1];
            bfr[2 * np + 1][0] = t[2];
            bfr[2 * np + 1][1] = t[3];
        }
    };
    // the pipeline of the codes: (cw, bkv) the two-bit step's codes and table, held a step; at three and four bits the tables are one step further on (tb waits in registers a step to be stored, a step before its codes are decoded)
    C2Cw<B> cw = ldc(0);
    uint2 bkv = TAB ? make_uint2(0, 0) : ldb(0);
    uint4 tb = ldt(0);
#pragma unroll
    for (int st = 0; st < STAGES - 1; st++) {
        load_a(st, st);
        cp_commit();
        stt(st, tb);
        if constexpr (TAB) __syncthreads();
        store_b(st, cw, bkv);
        cw = ldc(st + 1);
        if constexpr (!TAB) bkv = ldb(st + 1);
        tb = ldt(st + 1);
    }
    // (the table of step STAGES - 1, whose codes are decoded by the first iteration, is stored now and made visible by the barrier below; tb goes on to step STAGES's)
    stt(STAGES - 1, tb);
    tb = ldt(STAGES);
    float acc[MT][NTL][4];
#pragma unroll
    for (int i = 0; i < MT; i++)
#pragma unroll
        for (int j = 0; j < NTL; j++) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
    cp_wait<STAGES - 2>();
    __syncthreads();
    uint32_t af[2][MT][4], bfr[2][NTL][2];
    load_frags(af[0], bfr[0], 0, 0);
    int rd = 0, wr = STAGES - 1;
    for (int kt = 0; kt < NK; kt++) {
#pragma unroll
        for (int ks = 0; ks < KS; ks++) {
            const int cur = ks & 1, nxt = cur ^ 1;
            if (ks == KS - 1) {
                cp_wait<STAGES - 2>();
                __syncthreads();
                rd = rd + 1 == STAGES ? 0 : rd + 1;
            }
            load_frags(af[nxt], bfr[nxt], rd, (ks + 1) % KS);
            if (ks == 0) {
                if (kt + STAGES - 1 < NK) {
                    load_a(wr, kt + STAGES - 1);
                    // (the table of this step was stored a step ago, behind a barrier; the next step's, held in tb, goes in the stage that step kt just freed)
                    store_b(wr, cw, bkv);
                    if (kt + STAGES < NK) {
                        cw = ldc(kt + STAGES);
                        if constexpr (!TAB) bkv = ldb(kt + STAGES);
                        stt((kt + STAGES) % STAGES, tb);
                        if (kt + STAGES + 1 < NK) tb = ldt(kt + STAGES + 1);
                    }
                }
                cp_commit();
                wr = wr + 1 == STAGES ? 0 : wr + 1;
            }
#pragma unroll
            for (int mt = 0; mt < MT; mt++)
#pragma unroll
                for (int nt = 0; nt < NTL; nt++) mma16816(acc[mt][nt], af[cur][mt], bfr[cur][nt][0], bfr[cur][nt][1]);
        }
    }
    cp_wait<0>();
    const int g = lane >> 2, t4 = lane & 3;
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
        const int r0 = wm * WM + mt * 16 + g, r1 = r0 + 8;
        int t0 = 0, t1 = 0;
        float w0 = 0.f, w1 = 0.f;
        if (r0 < rows) {
            t0 = tok[sg.begin[seg] + m0 + r0];
            w0 = wt[sg.begin[seg] + m0 + r0];
        }
        if (r1 < rows) {
            t1 = tok[sg.begin[seg] + m0 + r1];
            w1 = wt[sg.begin[seg] + m0 + r1];
        }
#pragma unroll
        for (int nt = 0; nt < NTL; nt++) {
            const int col = n0 + wn * WN + nt * 8 + 2 * t4;
            if (r0 < rows) {
                atomicAdd(yrot + (size_t)t0 * HID + col, w0 * acc[mt][nt][0]);
                atomicAdd(yrot + (size_t)t0 * HID + col + 1, w0 * acc[mt][nt][1]);
            }
            if (r1 < rows) {
                atomicAdd(yrot + (size_t)t1 * HID + col, w1 * acc[mt][nt][2]);
                atomicAdd(yrot + (size_t)t1 * HID + col + 1, w1 * acc[mt][nt][3]);
            }
        }
    }
}

// A ternary matrix's rows (820 bytes of five-digit bytes, base 3 with the first code the units digit, and three bf16 entries) as two-bit rows (1024 bytes, four entries, the fourth 0): a thread takes 20 codes (4 bytes in, 5 out),
// 205 of them a row (the last has 16 codes); tables by the first `rows` threads.
extern "C" __global__ void __launch_bounds__(256) c2_tern_cb2(const unsigned char* __restrict__ src, const unsigned char* __restrict__ sbook, unsigned char* __restrict__ dst, unsigned char* __restrict__ dbook, int rows) {
    const long long id = (long long)blockIdx.x * 256 + threadIdx.x;
    if (id < rows) {
        const unsigned short* sb = reinterpret_cast<const unsigned short*>(sbook) + id * 3;
        uint2 o;
        o.x = (uint32_t)sb[0] | (uint32_t)sb[1] << 16;
        o.y = (uint32_t)sb[2];
        *reinterpret_cast<uint2*>(dbook + id * 8) = o;
    }
    if (id >= (long long)rows * 205) return;
    const int row = (int)(id / 205), g = (int)(id % 205);
    const unsigned char* p = src + (size_t)row * 820 + 4 * g;
    uint32_t d[20];
#pragma unroll
    for (int b = 0; b < 4; b++) {
        uint32_t x = p[b];
#pragma unroll
        for (int i = 0; i < 5; i++) {
            const uint32_t q = x / 3u;
            d[5 * b + i] = x - 3u * q;
            x = q;
        }
    }
    unsigned char* q = dst + (size_t)row * 1024 + 5 * g;
#pragma unroll
    for (int o = 0; o < 5; o++) {
        if (5 * g + o < 1024) q[o] = (unsigned char)(d[4 * o] | d[4 * o + 1] << 2 | d[4 * o + 2] << 4 | d[4 * o + 3] << 6);
    }
}

// the tables' shared memory a kernel declares (u16 entries): the first product's 128 rows of the tile (none at two bits: the registers hold them), the second's 64 rows of a step for each stage
#define C2_UP_TAB(B) ((B) == 2 ? 2 : 128 * ((1 << (B)) + 2))
#define C2_DOWN_TAB(B, S) ((B) == 2 ? 2 : (S) * 64 * ((1 << (B)) + 2))

// dynamic shared memory: STAGES * 24,576 bytes (49,152 at two stages, 73,728 at three)
#define C2_UP(B, S, BLK)                                                                                                                                                                                     \
    extern "C" __global__ void __launch_bounds__(128, BLK) c2_up##B##_##S(const unsigned short* xrb, const int* tok, const unsigned char* rcodes, const unsigned char* rbooks, bf16* act, C2Segs sg, float limit) { \
        __shared__ unsigned short s_tab[C2_UP_TAB(B)];                                                                                                                                                       \
        c2_up_t<S, B>(xrb, tok, rcodes, rbooks, act, sg, limit, s_tab);                                                                                                                                      \
    }
#define C2_DOWN(B, S, BLK)                                                                                                                                                                                   \
    extern "C" __global__ void __launch_bounds__(128, BLK) c2_down##B##_##S(const bf16* act, const int* tok, const float* wt, const unsigned char* rcodes, const unsigned char* rbooks, float* yrot, C2Segs sg) { \
        __shared__ unsigned short s_tab[C2_DOWN_TAB(B, S)];                                                                                                                                                  \
        c2_down_t<S, B>(act, tok, wt, rcodes, rbooks, yrot, sg, s_tab);                                                                                                                                      \
    }
C2_UP(2, 2, 2)
C2_UP(2, 3, 1)
C2_UP(3, 2, 2)
C2_UP(3, 3, 1)
C2_UP(4, 2, 2)
C2_UP(4, 3, 1)
C2_DOWN(2, 2, 2)
C2_DOWN(2, 3, 1)
C2_DOWN(3, 2, 2)
C2_DOWN(3, 3, 1)
C2_DOWN(4, 2, 2)
C2_DOWN(4, 3, 1)

// Qwen3.8-Flash-Next's experts (2,560 x 640, the same rows and tables at 2,560 codes a row): the same kernels at those sizes (grid y: 640 / 64 neurons, 2,560 / 128 outputs).
#define C2Q_UP(B, S, BLK)                                                                                                                                                                                    \
    extern "C" __global__ void __launch_bounds__(128, BLK) c2q_up##B##_##S(const unsigned short* xrb, const int* tok, const unsigned char* rcodes, const unsigned char* rbooks, bf16* act, C2Segs sg, float limit) { \
        __shared__ unsigned short s_tab[C2_UP_TAB(B)];                                                                                                                                                       \
        c2_up_t<S, B, 2560, 640>(xrb, tok, rcodes, rbooks, act, sg, limit, s_tab);                                                                                                                           \
    }
#define C2Q_DOWN(B, S, BLK)                                                                                                                                                                                  \
    extern "C" __global__ void __launch_bounds__(128, BLK) c2q_down##B##_##S(const bf16* act, const int* tok, const float* wt, const unsigned char* rcodes, const unsigned char* rbooks, float* yrot, C2Segs sg) { \
        __shared__ unsigned short s_tab[C2_DOWN_TAB(B, S)];                                                                                                                                                  \
        c2_down_t<S, B, 2560, 640>(act, tok, wt, rcodes, rbooks, yrot, sg, s_tab);                                                                                                                           \
    }
C2Q_UP(2, 2, 2)
C2Q_UP(2, 3, 1)
C2Q_UP(3, 2, 2)
C2Q_UP(3, 3, 1)
C2Q_UP(4, 2, 2)
C2Q_UP(4, 3, 1)
C2Q_DOWN(2, 2, 2)
C2Q_DOWN(2, 3, 1)
C2Q_DOWN(3, 2, 2)
C2Q_DOWN(3, 3, 1)
C2Q_DOWN(4, 2, 2)
C2Q_DOWN(4, 3, 1)

// Qwen3.8-Flash-Next's products of every width in one launch (a micro-batch's experts are of two, three and four bits; a launch a width left the products of a short prompt three launches of a few dozen blocks each,
// one after another, each as long as one block's 40 steps): a block takes its segment's width (C2Segs ubits / dbits) and runs that width's kernel, the tables' shared memory the widest's.
#define C2QX_UP(S, BLK)                                                                                                                                                                                      \
    extern "C" __global__ void __launch_bounds__(128, BLK) c2qx_up_##S(const unsigned short* xrb, const int* tok, const unsigned char* rcodes, const unsigned char* rbooks, bf16* act, C2Segs sg, float limit) {   \
        __shared__ unsigned short s_tab[C2_UP_TAB(4)];                                                                                                                                                       \
        const int tile = blockIdx.x;                                                                                                                                                                         \
        if (tile >= sg.tiles[sg.n]) return;                                                                                                                                                                  \
        int seg = 0;                                                                                                                                                                                         \
        while (seg + 1 < sg.n && tile >= sg.tiles[seg + 1]) seg++;                                                                                                                                           \
        switch (sg.ubits[seg]) {                                                                                                                                                                             \
            case 2: c2_up_t<S, 2, 2560, 640>(xrb, tok, rcodes, rbooks, act, sg, limit, s_tab); break;                                                                                                        \
            case 3: c2_up_t<S, 3, 2560, 640>(xrb, tok, rcodes, rbooks, act, sg, limit, s_tab); break;                                                                                                        \
            case 5: c2_up_t<S, 5, 2560, 640>(xrb, tok, rcodes, rbooks, act, sg, limit, s_tab); break;                                                                                                        \
            case 6: c2_up_t<S, 6, 2560, 640>(xrb, tok, rcodes, rbooks, act, sg, limit, s_tab); break;                                                                                                        \
            case 7: c2_up_t<S, 7, 2560, 640>(xrb, tok, rcodes, rbooks, act, sg, limit, s_tab); break;                                                                                                        \
            default: c2_up_t<S, 4, 2560, 640>(xrb, tok, rcodes, rbooks, act, sg, limit, s_tab); break;                                                                                                       \
        }                                                                                                                                                                                                    \
    }
#define C2QX_DOWN(S, BLK)                                                                                                                                                                                    \
    extern "C" __global__ void __launch_bounds__(256, BLK) c2qx_down_##S(const bf16* act, const int* tok, const float* wt, const unsigned char* rcodes, const unsigned char* rbooks, float* yrot, C2Segs sg) {   \
        __shared__ unsigned short s_tab[C2_DOWN_TAB(4, S)];                                                                                                                                                  \
        const int tile = blockIdx.x;                                                                                                                                                                         \
        if (tile >= sg.tiles[sg.n]) return;                                                                                                                                                                  \
        int seg = 0;                                                                                                                                                                                         \
        while (seg + 1 < sg.n && tile >= sg.tiles[seg + 1]) seg++;                                                                                                                                           \
        switch (sg.dbits[seg]) {                                                                                                                                                                             \
            case 2: c2_down_t<S, 2, 2560, 640, 256>(act, tok, wt, rcodes, rbooks, yrot, sg, s_tab); break;                                                                                                   \
            case 3: c2_down_t<S, 3, 2560, 640, 256>(act, tok, wt, rcodes, rbooks, yrot, sg, s_tab); break;                                                                                                   \
            case 5: c2_down_t<S, 5, 2560, 640, 256>(act, tok, wt, rcodes, rbooks, yrot, sg, s_tab); break;                                                                                                   \
            case 6: c2_down_t<S, 6, 2560, 640, 256>(act, tok, wt, rcodes, rbooks, yrot, sg, s_tab); break;                                                                                                   \
            case 7: c2_down_t<S, 7, 2560, 640, 256>(act, tok, wt, rcodes, rbooks, yrot, sg, s_tab); break;                                                                                                   \
            default: c2_down_t<S, 4, 2560, 640, 256>(act, tok, wt, rcodes, rbooks, yrot, sg, s_tab); break;                                                                                                  \
        }                                                                                                                                                                                                    \
    }
C2QX_UP(2, 2)
C2QX_UP(3, 1)
// the same with 128 pairs a block (8 warps): segments of many pairs, the codes made bf16 once for twice the rows
extern "C" __global__ void __launch_bounds__(256, 1) c2qx_up128_2(const unsigned short* xrb, const int* tok, const unsigned char* rcodes, const unsigned char* rbooks, bf16* act, C2Segs sg, float limit) {
    __shared__ unsigned short s_tab[C2_UP_TAB(4)];
    const int tile = blockIdx.x;
    if (tile >= sg.tiles[sg.n]) return;
    int seg = 0;
    while (seg + 1 < sg.n && tile >= sg.tiles[seg + 1]) seg++;
    switch (sg.ubits[seg]) {
        case 2: c2_up_t<2, 2, 2560, 640, 128>(xrb, tok, rcodes, rbooks, act, sg, limit, s_tab); break;
        case 3: c2_up_t<2, 3, 2560, 640, 128>(xrb, tok, rcodes, rbooks, act, sg, limit, s_tab); break;
        case 5: c2_up_t<2, 5, 2560, 640, 128>(xrb, tok, rcodes, rbooks, act, sg, limit, s_tab); break;
        case 6: c2_up_t<2, 6, 2560, 640, 128>(xrb, tok, rcodes, rbooks, act, sg, limit, s_tab); break;
        case 7: c2_up_t<2, 7, 2560, 640, 128>(xrb, tok, rcodes, rbooks, act, sg, limit, s_tab); break;
        default: c2_up_t<2, 4, 2560, 640, 128>(xrb, tok, rcodes, rbooks, act, sg, limit, s_tab); break;
    }
}
C2QX_DOWN(2, 1)
C2QX_DOWN(3, 1)
// the same with 128 outputs a block (4 warps; at two stages 49,152 bytes of dynamic shared memory and its 4,608 of tables, where a block takes 64 KiB at most: src/dev.rs smem64)
extern "C" __global__ void __launch_bounds__(128, 2) c2qx_down4w_2(const bf16* act, const int* tok, const float* wt, const unsigned char* rcodes, const unsigned char* rbooks, float* yrot, C2Segs sg) {
    __shared__ unsigned short s_tab[C2_DOWN_TAB(4, 2)];
    const int tile = blockIdx.x;
    if (tile >= sg.tiles[sg.n]) return;
    int seg = 0;
    while (seg + 1 < sg.n && tile >= sg.tiles[seg + 1]) seg++;
    switch (sg.dbits[seg]) {
        case 2: c2_down_t<2, 2, 2560, 640, 128>(act, tok, wt, rcodes, rbooks, yrot, sg, s_tab); break;
        case 3: c2_down_t<2, 3, 2560, 640, 128>(act, tok, wt, rcodes, rbooks, yrot, sg, s_tab); break;
        case 5: c2_down_t<2, 5, 2560, 640, 128>(act, tok, wt, rcodes, rbooks, yrot, sg, s_tab); break;
        case 6: c2_down_t<2, 6, 2560, 640, 128>(act, tok, wt, rcodes, rbooks, yrot, sg, s_tab); break;
        case 7: c2_down_t<2, 7, 2560, 640, 128>(act, tok, wt, rcodes, rbooks, yrot, sg, s_tab); break;
        default: c2_down_t<2, 4, 2560, 640, 128>(act, tok, wt, rcodes, rbooks, yrot, sg, s_tab); break;
    }
}

// A ternary matrix's rows of `cols` codes (cols / 5 bytes, a multiple of 20 codes) as two-bit rows (cols / 4 bytes, four entries, the fourth 0): c2_tern_cb2's arithmetic at any width, a thread 20 codes.
extern "C" __global__ void __launch_bounds__(256) c2_tern_rows(const unsigned char* __restrict__ src, const unsigned char* __restrict__ sbook, unsigned char* __restrict__ dst, unsigned char* __restrict__ dbook, int rows, int cols) {
    const long long id = (long long)blockIdx.x * 256 + threadIdx.x;
    const int per = cols / 20;
    if (id < rows) {
        const unsigned short* sb = reinterpret_cast<const unsigned short*>(sbook) + id * 3;
        uint2 o;
        o.x = (uint32_t)sb[0] | (uint32_t)sb[1] << 16;
        o.y = (uint32_t)sb[2];
        *reinterpret_cast<uint2*>(dbook + id * 8) = o;
    }
    if (id >= (long long)rows * per) return;
    const int row = (int)(id / per), g = (int)(id % per);
    const unsigned char* p = src + (size_t)row * (cols / 5) + 4 * g;
    uint32_t d[20];
#pragma unroll
    for (int b = 0; b < 4; b++) {
        uint32_t x = p[b];
#pragma unroll
        for (int i = 0; i < 5; i++) {
            const uint32_t q = x / 3u;
            d[5 * b + i] = x - 3u * q;
            x = q;
        }
    }
    unsigned char* q = dst + (size_t)row * (cols / 4) + 5 * g;
#pragma unroll
    for (int o = 0; o < 5; o++) q[o] = (unsigned char)(d[4 * o] | d[4 * o + 1] << 2 | d[4 * o + 2] << 4 | d[4 * o + 3] << 6);
}

// y [count] += x [count] (fp32): the host's share of a call's sums added to the GPU's.
extern "C" __global__ void c2_add_f32(float* __restrict__ y, const float* __restrict__ x, long long count) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) y[i] += x[i];
}
