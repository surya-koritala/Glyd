// The index arithmetic of the dense GEMM's shared-memory tiles, on its own so that a host program can check it (tools/gemm_idx_test.cpp) against the plain per-element formula.
//
// A tile in shared memory is rows of BK bf16 (BK 32: four 16-byte chunks a row, BK 64: eight). The chunks of a row are permuted by a key taken from the row, so that the eight 16-byte
// rows an ldmatrix reads, and the rows a warp's cp.async writes together, fall in different banks.
#pragma once
#if defined(__CUDACC__) || defined(__HIPCC__)
#define GEMM_HD __host__ __device__ __forceinline__
#else
#define GEMM_HD inline
#endif

template <int BK>
GEMM_HD int swz_key(int r) {
    return BK == 32 ? ((r >> 1) & 3) : (r & 7);
}

// the element offset of chunk c (16 bytes) of row r
template <int BK>
GEMM_HD int swzk(int r, int c) {
    return r * BK + ((c ^ swz_key<BK>(r)) << 3);
}

// How a block's NTH threads fill a tile of ROWS rows with cp.async: thread t takes chunk (t % CPR) of row (t / CPR) and then the rows RPI further on, ROWS / RPI times. RPI is a multiple of 8, so the key of
// every row it takes is the first one's, and the shared offset of its pass p is the first pass's plus p * RPI * BK elements.
template <int BK, int NTH>
struct Fill {
    static constexpr int CPR = BK / 8;
    static constexpr int RPI = NTH / CPR;
    static GEMM_HD int row(int tid) { return tid / CPR; }
    static GEMM_HD int chunk(int tid) { return tid % CPR; }
    // the offset, in elements, of pass 0's chunk of thread tid (pass p: + p * RPI * BK)
    static GEMM_HD int soff(int tid) { return swzk<BK>(row(tid), chunk(tid)); }
};

// The ldmatrix addresses of a lane. The A fragment (16 rows x 16 columns of k-step ks) of rows `base + mt * 16`: the lane reads row (mi & 1) * 8 + r8, chunk ks * 2 + (mi >> 1), with mi = lane >> 3
// and r8 = lane & 7; the B fragment (two 8-row groups, np * 16 rows on) reads row (mi >> 1) * 8 + r8, chunk ks * 2 + (mi & 1). The key comes from r8 alone (every row offset added is a multiple of 8), so
// a thread computes one offset a k-step for A and for B, and the rows of its other fragments are constants added to it.
template <int BK>
GEMM_HD int frag_a_off(int base_row, int lane, int ks) {
    const int mi = lane >> 3, r8 = lane & 7;
    return (base_row + (mi & 1) * 8 + r8) * BK + (((ks * 2 + (mi >> 1)) ^ swz_key<BK>(r8)) << 3);
}

template <int BK>
GEMM_HD int frag_b_off(int base_row, int lane, int ks) {
    const int mi = lane >> 3, r8 = lane & 7;
    return (base_row + (mi >> 1) * 8 + r8) * BK + (((ks * 2 + (mi & 1)) ^ swz_key<BK>(r8)) << 3);
}
