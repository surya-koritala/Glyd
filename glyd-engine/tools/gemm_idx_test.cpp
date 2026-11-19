// The GEMM's shared-memory index arithmetic (kernels/gemm_idx.h) against the plain formula, for every thread and lane of every tile shape the kernels use, and that no two chunks of a tile land on one place and
// that the eight rows of an ldmatrix (and a warp's cp.async rows) fall in eight different 16-byte bank groups.
//   c++ -std=c++17 -O1 tools/gemm_idx_test.cpp -o /tmp/gemm_idx_test && /tmp/gemm_idx_test
#include "../kernels/gemm_idx.h"
#include <cstdio>
#include <set>
#include <vector>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { if (fails++ < 10) { printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } } while (0)

template <int BK, int NTH, int ROWS>
void fill() {
    using F = Fill<BK, NTH>;
    static_assert(F::RPI % 8 == 0 && ROWS % F::RPI == 0, "the fill's shape");
    std::set<int> seen;
    for (int t = 0; t < NTH; t++)
        for (int p = 0; p < ROWS / F::RPI; p++) {
            int r = F::row(t) + p * F::RPI, c = F::chunk(t);
            CHECK(swzk<BK>(r, c) == F::soff(t) + p * F::RPI * BK, "BK %d NTH %d thread %d pass %d", BK, NTH, t, p);
            CHECK(seen.insert(swzk<BK>(r, c)).second, "two chunks at one place (BK %d NTH %d)", BK, NTH);
            CHECK(r < ROWS && c < BK / 8, "outside the tile");
        }
    CHECK((int)seen.size() == ROWS * BK / 8, "the tile is not filled: %zu of %d", seen.size(), ROWS * BK / 8);
    // a warp's 32 writes in one pass: 16-byte chunks; the bank group (offset in 16 bytes, mod 8) of each, per 8 consecutive threads reading together is what ldmatrix needs; for the writes: 32 chunks
    // are 512 bytes, 4 wavefronts of 128 bytes at best: each wavefront (8 threads) with no two on one bank group
    for (int w = 0; w < NTH / 32; w++)
        for (int q = 0; q < 4; q++) {
            std::set<int> banks;
            for (int l = q * 8; l < q * 8 + 8; l++) banks.insert((F::soff(w * 32 + l) * 2 / 16) % 8);
            CHECK(banks.size() == 8, "BK %d NTH %d warp %d quarter %d: %zu bank groups", BK, NTH, w, q, banks.size());
        }
}

template <int BK, int WM, int WN, int WARPS_M, int WARPS_N>
void frags() {
    constexpr int MT = WM / 16, NP = WN / 16, KS = BK / 16;
    for (int wm = 0; wm < WARPS_M; wm++)
        for (int wn = 0; wn < WARPS_N; wn++)
            for (int lane = 0; lane < 32; lane++) {
                const int mi = lane >> 3, r8 = lane & 7;
                for (int ks = 0; ks < KS; ks++) {
                    const int a0 = frag_a_off<BK>(wm * WM, lane, ks);
                    for (int mt = 0; mt < MT; mt++)
                        CHECK(a0 + mt * 16 * BK == swzk<BK>(wm * WM + mt * 16 + (mi & 1) * 8 + r8, ks * 2 + (mi >> 1)), "A BK %d lane %d ks %d mt %d", BK, lane, ks, mt);
                    const int b0 = frag_b_off<BK>(wn * WN, lane, ks);
                    for (int np = 0; np < NP; np++)
                        CHECK(b0 + np * 16 * BK == swzk<BK>(wn * WN + np * 16 + (mi >> 1) * 8 + r8, ks * 2 + (mi & 1)), "B BK %d lane %d ks %d np %d", BK, lane, ks, np);
                }
            }
    // each ldmatrix: the eight rows' 16-byte pieces in eight different bank groups
    for (int ks = 0; ks < KS; ks++)
        for (int mt = 0; mt < MT; mt++)
            for (int m = 0; m < 4; m++) {
                std::set<int> banks;
                for (int r8 = 0; r8 < 8; r8++) {
                    const int lane = m * 8 + r8;
                    banks.insert((frag_a_off<BK>(mt * 16, lane, ks) * 2 / 16) % 8);
                }
                CHECK(banks.size() == 8, "A ldmatrix bank groups BK %d ks %d mt %d matrix %d: %zu", BK, ks, mt, m, banks.size());
                banks.clear();
                for (int r8 = 0; r8 < 8; r8++) banks.insert((frag_b_off<BK>(mt * 16, m * 8 + r8, ks) * 2 / 16) % 8);
                CHECK(banks.size() == 8, "B ldmatrix bank groups BK %d ks %d mt %d matrix %d: %zu", BK, ks, mt, m, banks.size());
            }
}

int main() {
    // (BK, threads, rows): the A tile of BM rows and the B tile of BN rows
    fill<32, 256, 128>();  // 128x256 tiles, 8 warps: A 128 rows, B 256 rows
    fill<32, 256, 256>();
    fill<32, 128, 128>();  // 128x128, 4 warps
    fill<32, 128, 64>();   // 128x64, 4 warps
    fill<32, 256, 128>();
    fill<64, 256, 128>();
    fill<64, 256, 256>();
    fill<64, 128, 128>();
    fill<64, 128, 64>();
    frags<32, 64, 64, 2, 4>();  // 128x256: warp tile 64x64
    frags<32, 64, 64, 2, 2>();  // 128x128
    frags<32, 64, 32, 2, 2>();  // 128x64
    frags<32, 64, 64, 4, 2>();  // 256x128
    frags<64, 64, 64, 2, 4>();
    frags<64, 64, 64, 2, 2>();
    if (fails) {
        printf("%d checks failed\n", fails);
        return 1;
    }
    printf("gemm_idx: all index checks pass\n");
    return 0;
}
