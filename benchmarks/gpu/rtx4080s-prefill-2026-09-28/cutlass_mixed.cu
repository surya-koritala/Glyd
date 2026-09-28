// CUTLASS 2.x (sm80 API) mixed-input GEMM on an RTX 4080 SUPER: Y [M, N] = X [M, K] (bf16, row-major) W^T, W [N, K]
// as B column-major, W's elements u8 (OpMultiplyAddMixedInputUpcast: converted to bf16 in registers before mma.sync)
// or bf16 (plain), f32 accumulate, bf16 out; tiles as listed. Times: CUDA events, median of 9 after 3.
#include <cstdio>
#include <vector>
#include <algorithm>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"

template <class EB, int TMn, int TNn, int TKn, int WMn, int WNn, int WKn, int ST, class Op>
using G = cutlass::gemm::device::GemmUniversal<
    cutlass::bfloat16_t, cutlass::layout::RowMajor, EB, cutlass::layout::ColumnMajor, cutlass::bfloat16_t, cutlass::layout::RowMajor,
    float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, cutlass::gemm::GemmShape<TMn, TNn, TKn>, cutlass::gemm::GemmShape<WMn, WNn, WKn>,
    cutlass::gemm::GemmShape<16, 8, 16>, cutlass::epilogue::thread::LinearCombination<cutlass::bfloat16_t, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, ST, 8, 16 / sizeof(EB) * 1, Op>;

template <class Gemm, class EB>
float run(const char* name, int M, int N, int K, void* x, void* w, void* y, void* ws) {
    typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K}, 1, {1.f, 0.f}, x, w, y, y, (int64_t)M * K, (int64_t)N * K,
                                  (int64_t)M * N, (int64_t)M * N, K, K, N, N);
    Gemm g;
    if (g.can_implement(args) != cutlass::Status::kSuccess) {
        printf("  %-44s cannot implement\n", name);
        return 0;
    }
    if (g.initialize(args, ws) != cutlass::Status::kSuccess) {
        printf("  %-44s init failed\n", name);
        return 0;
    }
    cudaEvent_t a, b;
    cudaEventCreate(&a), cudaEventCreate(&b);
    std::vector<float> t;
    for (int i = 0; i < 12; i++) {
        cudaEventRecord(a);
        g();
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms;
        cudaEventElapsedTime(&ms, a, b);
        if (i >= 3) t.push_back(ms * 1000);
    }
    std::sort(t.begin(), t.end());
    cudaError_t e = cudaGetLastError();
    printf("  %-44s %8.0f us %6.1f TFLOPS %s\n", name, t[t.size() / 2], 2.0 * M * N * K / t[t.size() / 2] / 1e6, e ? cudaGetErrorString(e) : "");
    return t[t.size() / 2];
}

int main() {
    int shapes[][3] = {{1024, 19456, 2560}, {4096, 19456, 2560}, {1024, 2560, 9728}, {4096, 2560, 9728}};
    void *x, *w, *y, *ws;
    cudaMalloc(&x, 4096ull * 9728 * 2), cudaMalloc(&w, 19456ull * 9728 * 2), cudaMalloc(&y, 4096ull * 19456 * 2), cudaMalloc(&ws, 256 << 20);
    cudaMemset(x, 0x11, 4096ull * 9728 * 2), cudaMemset(w, 0x22, 19456ull * 9728 * 2);
    using Up = cutlass::arch::OpMultiplyAddMixedInputUpcast;
    using Plain = cutlass::arch::OpMultiplyAdd;
    for (auto& s : shapes) {
        int M = s[0], N = s[1], K = s[2];
        printf("M %d N %d K %d\n", M, N, K);
        run<G<uint8_t, 128, 128, 64, 64, 64, 64, 4, Up>, uint8_t>("u8 upcast 128x128x64 w64x64 4st", M, N, K, x, w, y, ws);
        run<G<uint8_t, 128, 256, 64, 64, 64, 64, 3, Up>, uint8_t>("u8 upcast 128x256x64 w64x64 3st", M, N, K, x, w, y, ws);
        run<G<uint8_t, 256, 128, 64, 64, 64, 64, 3, Up>, uint8_t>("u8 upcast 256x128x64 w64x64 3st", M, N, K, x, w, y, ws);
        run<G<uint8_t, 128, 256, 32, 64, 64, 32, 4, Up>, uint8_t>("u8 upcast 128x256x32 w64x64 4st", M, N, K, x, w, y, ws);
        run<G<uint8_t, 256, 128, 32, 64, 64, 32, 4, Up>, uint8_t>("u8 upcast 256x128x32 w64x64 4st", M, N, K, x, w, y, ws);
        run<G<cutlass::bfloat16_t, 128, 256, 32, 64, 64, 32, 3, Plain>, cutlass::bfloat16_t>("bf16 128x256x32 w64x64 3st", M, N, K, x, w, y, ws);
        run<G<cutlass::bfloat16_t, 256, 128, 32, 64, 64, 32, 3, Plain>, cutlass::bfloat16_t>("bf16 256x128x32 w64x64 3st", M, N, K, x, w, y, ws);
    }
    return 0;
}
