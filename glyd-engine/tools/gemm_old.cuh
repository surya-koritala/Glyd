// The dense products' earlier versions and the tiles not chosen, for tools/gemm_bench.cu to time against the engine's (included after kernels/engine.cu, which has the current kernel).
#pragma once
template <int BM, int BN, int BK, int WARPS_M, int WARPS_N, int STAGES, int GROUP_M>
__device__ __forceinline__ void gemm_nt_t(const bf16* __restrict__ X, const bf16* __restrict__ W, bf16* __restrict__ Y, int M, int N, int K) {
    constexpr int NTH = WARPS_M * WARPS_N * 32;
    constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N;
    constexpr int MT = WM / 16, NTL = WN / 8;
    constexpr int CPR = BK / 8;
    extern __shared__ __align__(128) unsigned char smem_raw[];
    bf16* sA = reinterpret_cast<bf16*>(smem_raw);  // [STAGES][BM][BK]
    bf16* sB = sA + STAGES * BM * BK;              // [STAGES][BN][BK]
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp / WARPS_N, wn = warp % WARPS_N;
    const int num_m = (M + BM - 1) / BM, num_n = N / BN;
    const int pid = blockIdx.x;
    const int in_group = GROUP_M * num_n;
    const int first_m = (pid / in_group) * GROUP_M;
    const int gsz = min(num_m - first_m, GROUP_M);
    const int m_tile = first_m + (pid % in_group) % gsz;
    const int n_tile = (pid % in_group) / gsz;
    const int m0 = m_tile * BM, n0 = n_tile * BN;
    const int nk = K / BK;

    auto load_stage = [&](int s, int kt) {
        bf16* a = sA + s * BM * BK;
        bf16* b = sB + s * BN * BK;
        for (int i = tid; i < BM * CPR; i += NTH) {
            int r = i / CPR, c = i % CPR;
            bool ok = m0 + r < M;
            cp16(s_addr(a + swzk<BK>(r, c)), X + (size_t)(ok ? m0 + r : 0) * K + (size_t)kt * BK + c * 8, ok);
        }
        for (int i = tid; i < BN * CPR; i += NTH) {
            int r = i / CPR, c = i % CPR;
            cp16(s_addr(b + swzk<BK>(r, c)), W + (size_t)(n0 + r) * K + (size_t)kt * BK + c * 8, true);
        }
    };
#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < nk) load_stage(s, s);
        cp_commit();
    }
    float acc[MT][NTL][4];
#pragma unroll
    for (int i = 0; i < MT; i++)
#pragma unroll
        for (int j = 0; j < NTL; j++) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
    const int mi = lane >> 3, r8 = lane & 7;

    for (int kt = 0; kt < nk; kt++) {
        cp_wait<STAGES - 2>();
        __syncthreads();
        if (kt + STAGES - 1 < nk) load_stage((kt + STAGES - 1) % STAGES, kt + STAGES - 1);
        cp_commit();
        const bf16* a = sA + (kt % STAGES) * BM * BK;
        const bf16* b = sB + (kt % STAGES) * BN * BK;
#pragma unroll
        for (int ks = 0; ks < BK / 16; ks++) {
            uint32_t af[MT][4], bfr[NTL][2];
#pragma unroll
            for (int mt = 0; mt < MT; mt++) ldsm4(af[mt], s_addr(a + swzk<BK>(wm * WM + mt * 16 + (mi & 1) * 8 + r8, ks * 2 + (mi >> 1))));
#pragma unroll
            for (int np = 0; np < NTL / 2; np++) {
                uint32_t t[4];
                ldsm4(t, s_addr(b + swzk<BK>(wn * WN + np * 16 + (mi >> 1) * 8 + r8, ks * 2 + (mi & 1))));
                bfr[2 * np][0] = t[0];
                bfr[2 * np][1] = t[1];
                bfr[2 * np + 1][0] = t[2];
                bfr[2 * np + 1][1] = t[3];
            }
#pragma unroll
            for (int mt = 0; mt < MT; mt++)
#pragma unroll
                for (int nt = 0; nt < NTL; nt++) mma16816(acc[mt][nt], af[mt], bfr[nt][0], bfr[nt][1]);
        }
    }
    cp_wait<0>();
    const int g = lane >> 2, t4 = lane & 3;
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
        const int r0 = m0 + wm * WM + mt * 16 + g, r1 = r0 + 8;
#pragma unroll
        for (int nt = 0; nt < NTL; nt++) {
            const int col = n0 + wn * WN + nt * 8 + 2 * t4;
            if (r0 < M) *reinterpret_cast<uint32_t*>(Y + (size_t)r0 * N + col) = pack_bf16(acc[mt][nt][0], acc[mt][nt][1]);
            if (r1 < M) *reinterpret_cast<uint32_t*>(Y + (size_t)r1 * N + col) = pack_bf16(acc[mt][nt][2], acc[mt][nt][3]);
        }
    }
}

// The same product with the main loop of a multistage pipeline: the fragments of the next k-step are loaded while the current one's products run (and a stage's wait and barrier come once
// the last k-step of its tile is begun), the stage the loads go to the one finished with a tile ago. BK 32 or 64 (KS = BK / 16 even).
template <int BM, int BN, int BK, int WARPS_M, int WARPS_N, int STAGES, int GROUP_M>
__device__ __forceinline__ void gemm2_nt_t(const bf16* __restrict__ X, const bf16* __restrict__ W, bf16* __restrict__ Y, int M, int N, int K) {
    constexpr int NTH = WARPS_M * WARPS_N * 32;
    constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N;
    constexpr int MT = WM / 16, NTL = WN / 8;
    constexpr int CPR = BK / 8;
    constexpr int KS = BK / 16;
    extern __shared__ __align__(128) unsigned char smem_raw[];
    bf16* sA = reinterpret_cast<bf16*>(smem_raw);
    bf16* sB = sA + STAGES * BM * BK;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp / WARPS_N, wn = warp % WARPS_N;
    const int num_m = (M + BM - 1) / BM, num_n = N / BN;
    const int pid = blockIdx.x;
    const int in_group = GROUP_M * num_n;
    const int first_m = (pid / in_group) * GROUP_M;
    const int gsz = min(num_m - first_m, GROUP_M);
    const int m_tile = first_m + (pid % in_group) % gsz;
    const int n_tile = (pid % in_group) / gsz;
    const int m0 = m_tile * BM, n0 = n_tile * BN;
    const int nk = K / BK;
    const int mi = lane >> 3, r8 = lane & 7;

    auto load_stage = [&](int st, int kt) {
        bf16* a = sA + st * BM * BK;
        bf16* b = sB + st * BN * BK;
#pragma unroll
        for (int i = tid; i < BM * CPR; i += NTH) {
            int r = i / CPR, c = i % CPR;
            bool ok = m0 + r < M;
            cp16(s_addr(a + swzk<BK>(r, c)), X + (size_t)(ok ? m0 + r : 0) * K + (size_t)kt * BK + c * 8, ok);
        }
#pragma unroll
        for (int i = tid; i < BN * CPR; i += NTH) {
            int r = i / CPR, c = i % CPR;
            cp16(s_addr(b + swzk<BK>(r, c)), W + (size_t)(n0 + r) * K + (size_t)kt * BK + c * 8, true);
        }
    };
    auto load_frags = [&](uint32_t (&af)[MT][4], uint32_t (&bfr)[NTL][2], int st, int ks) {
        const bf16* a = sA + st * BM * BK;
        const bf16* b = sB + st * BN * BK;
#pragma unroll
        for (int mt = 0; mt < MT; mt++) ldsm4(af[mt], s_addr(a + swzk<BK>(wm * WM + mt * 16 + (mi & 1) * 8 + r8, ks * 2 + (mi >> 1))));
#pragma unroll
        for (int np = 0; np < NTL / 2; np++) {
            uint32_t t[4];
            ldsm4(t, s_addr(b + swzk<BK>(wn * WN + np * 16 + (mi >> 1) * 8 + r8, ks * 2 + (mi & 1))));
            bfr[2 * np][0] = t[0];
            bfr[2 * np][1] = t[1];
            bfr[2 * np + 1][0] = t[2];
            bfr[2 * np + 1][1] = t[3];
        }
    };
#pragma unroll
    for (int st = 0; st < STAGES - 1; st++) {
        if (st < nk) load_stage(st, st);
        cp_commit();
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
    for (int kt = 0; kt < nk; kt++) {
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
                if (kt + STAGES - 1 < nk) load_stage(wr, kt + STAGES - 1);
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
        const int r0 = m0 + wm * WM + mt * 16 + g, r1 = r0 + 8;
#pragma unroll
        for (int nt = 0; nt < NTL; nt++) {
            const int col = n0 + wn * WN + nt * 8 + 2 * t4;
            if (r0 < M) *reinterpret_cast<uint32_t*>(Y + (size_t)r0 * N + col) = pack_bf16(acc[mt][nt][0], acc[mt][nt][1]);
            if (r1 < M) *reinterpret_cast<uint32_t*>(Y + (size_t)r1 * N + col) = pack_bf16(acc[mt][nt][2], acc[mt][nt][3]);
        }
    }
}


// dynamic shared memory: STAGES * (BM + BN) * BK * 2 bytes
#define GEMM_KERNEL(name, BM, BN, BK, WM_, WN_, ST, GRP, MINB) \
    extern "C" __global__ void __launch_bounds__(WM_ * WN_ * 32, MINB) name(const bf16* X, const bf16* W, bf16* Y, int M, int N, int K) { gemm_nt_t<BM, BN, BK, WM_, WN_, ST, GRP>(X, W, Y, M, N, K); }
GEMM_KERNEL(gemm_128x256_s3, 128, 256, 32, 2, 4, 3, 8, 1)
GEMM_KERNEL(gemm_128x256_s4, 128, 256, 32, 2, 4, 4, 8, 1)
GEMM_KERNEL(gemm_128x128_s3, 128, 128, 32, 2, 2, 3, 8, 2)
GEMM_KERNEL(gemm_128x128_s4, 128, 128, 32, 2, 2, 4, 8, 1)
GEMM_KERNEL(gemm_256x128_s3, 256, 128, 32, 4, 2, 3, 8, 1)
#define GEMM2_KERNEL(name, BM, BN, BK, WM_, WN_, ST, GRP, MINB) \
    extern "C" __global__ void __launch_bounds__(WM_ * WN_ * 32, MINB) name(const bf16* X, const bf16* W, bf16* Y, int M, int N, int K) { gemm2_nt_t<BM, BN, BK, WM_, WN_, ST, GRP>(X, W, Y, M, N, K); }
GEMM2_KERNEL(gemm2_128x128_s3, 128, 128, 32, 2, 2, 3, 8, 2)
GEMM2_KERNEL(gemm2_128x128_s4, 128, 128, 32, 2, 2, 4, 8, 1)
GEMM2_KERNEL(gemm2_128x256_s3, 128, 256, 32, 2, 4, 3, 8, 1)
GEMM2_KERNEL(gemm2_128x256_s4, 128, 256, 32, 2, 4, 4, 8, 1)
GEMM2_KERNEL(gemm2_256x128_s3, 256, 128, 32, 4, 2, 3, 8, 1)
GEMM2_KERNEL(gemm2_128x128_b64_s2, 128, 128, 64, 2, 2, 2, 8, 2)
GEMM2_KERNEL(gemm2_128x128_b64_s3, 128, 128, 64, 2, 2, 3, 8, 1)
GEMM2_KERNEL(gemm2_128x256_b64_s2, 128, 256, 64, 2, 4, 2, 8, 1)



// (kernels/engine.cu writes its kernels out, for gpu/gen_names.py; these experiments are made by a macro)
#define GEMM3_KERNEL(name, BM, BN, BK, WM_, WN_, ST, GRP, MINB, FUSE_) \
    extern "C" __global__ void __launch_bounds__(WM_ * WN_ * 32, MINB) name(const bf16* X, const bf16* W, bf16* Y, int M, int N, int K) { gemm3_nt_t<BM, BN, BK, WM_, WN_, ST, GRP, FUSE_>(X, W, Y, M, N, K); }
GEMM3_KERNEL(gemm3_128x256_s4, 128, 256, 32, 2, 4, 4, 8, 1, false)
GEMM3_KERNEL(gemm3_128x128_s3, 128, 128, 32, 2, 2, 3, 8, 2, false)
GEMM3_KERNEL(gemm3_128x128_b64_s2, 128, 128, 64, 2, 2, 2, 8, 2, false)
GEMM3_KERNEL(gemm3_128x64_s4, 128, 64, 32, 2, 2, 4, 8, 3, false)
GEMM3_KERNEL(gemm3f_128x128_s3, 128, 128, 32, 2, 2, 3, 8, 2, true)
