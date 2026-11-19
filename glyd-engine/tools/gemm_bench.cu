// The dense bf16 product on its own against cuBLAS: Qwen3-8B's four matrices at the prompt sizes, every tile variant.
//   nvcc -O3 -std=c++17 -arch=sm_89 -lineinfo tools/gemm_bench.cu -lcublas -o gemm_bench
//   ./gemm_bench [M ...]          (default 1024 3500)
#include "../kernels/engine.cu"
#include "gemm_old.cuh"
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <dlfcn.h>
#include <cstring>

// the GPU's SM clock, power and temperature as NVML reports them (loaded by dlopen: nothing to link)
struct Nvml {
    void* dev = nullptr;
    int (*clock)(void*, int, unsigned*) = nullptr;
    int (*power)(void*, unsigned*) = nullptr;
    int (*temp)(void*, int, unsigned*) = nullptr;
    Nvml() {
        void* h = dlopen("libnvidia-ml.so.1", RTLD_NOW);
        if (!h) return;
        auto init = (int (*)())dlsym(h, "nvmlInit_v2");
        auto get = (int (*)(unsigned, void**))dlsym(h, "nvmlDeviceGetHandleByIndex_v2");
        clock = (int (*)(void*, int, unsigned*))dlsym(h, "nvmlDeviceGetClockInfo");
        power = (int (*)(void*, unsigned*))dlsym(h, "nvmlDeviceGetPowerUsage");
        temp = (int (*)(void*, int, unsigned*))dlsym(h, "nvmlDeviceGetTemperature");
        if (!init || !get || init() != 0 || get(0, &dev) != 0) clock = nullptr;
    }
    void read(unsigned& mhz, unsigned& w, unsigned& c) {
        mhz = w = c = 0;
        unsigned mw = 0;
        if (clock && power && temp) {
            clock(dev, 1, &mhz);
            power(dev, &mw);
            temp(dev, 0, &c);
        }
        w = mw / 1000;
    }
};

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

typedef void (*GK)(const bf16*, const bf16*, bf16*, int, int, int);
struct V { const char* name; GK k; int bm, bn, st, th, bk; };

int main(int argc, char** argv) {
    std::vector<int> Ms;
    for (int i = 1; i < argc; i++) Ms.push_back(atoi(argv[i]));
    if (Ms.empty()) Ms = {1024, 3500};
    // GEMM_ALL=1: every kernel; else the ones that won on an L4 (the sweep over M)
    V all[] = {{"v1 128x256 s4", gemm_128x256_s4, 128, 256, 4, 256, 32}, {"v2 128x256 s3", gemm2_128x256_s3, 128, 256, 3, 256, 32}, {"v2 128x128 s3", gemm2_128x128_s3, 128, 128, 3, 128, 32},
              {"v3 128x256 s3", gemm3_128x256_s3, 128, 256, 3, 256, 32}, {"v3 128x256 s4", gemm3_128x256_s4, 128, 256, 4, 256, 32}, {"v3 128x128 s3", gemm3_128x128_s3, 128, 128, 3, 128, 32},
              {"v3 128x128 s4", gemm3_128x128_s4, 128, 128, 4, 128, 32}, {"v3 256x128 s3", gemm3_256x128_s3, 256, 128, 3, 256, 32}, {"v3 128x128 bk64 s2", gemm3_128x128_b64_s2, 128, 128, 2, 128, 64},
              {"v3 128x128 bk64 s3", gemm3_128x128_b64_s3, 128, 128, 3, 128, 64}, {"v3 128x256 bk64 s2", gemm3_128x256_b64_s2, 128, 256, 2, 256, 64}, {"v3 128x64 s4", gemm3_128x64_s4, 128, 64, 4, 128, 32},
              {"v3 256x128 bk64 s2", gemm3_256x128_b64_s2, 256, 128, 2, 256, 64}};
    std::vector<V> vs;
    for (auto& v : all) {
        bool win = !strncmp(v.name, "v3", 2) && strcmp(v.name, "v3 128x64 s4") != 0 && strcmp(v.name, "v3 128x128 s3") != 0 && strcmp(v.name, "v3 128x256 s4") != 0;
        if (win || getenv("GEMM_ALL")) vs.push_back(v);
    }
    // the product with silu(gate) * up in its epilogue (the gate+up shape only): against cuBLAS's product and then the engine's silu_mul
    V fv[] = {{"v3 fused 128x128 s3", gemm3f_128x128_s3, 128, 128, 3, 128, 32}, {"v3 fused 128x256 s3", gemm3f_128x256_s3, 128, 256, 3, 256, 32}, {"v3 fused 128x256 bk64 s2", gemm3f_128x256_b64_s2, 128, 256, 2, 256, 64},
              {"v3 fused 256x128 s3", gemm3f_256x128_s3, 256, 128, 3, 256, 32}};
    for (auto& v : fv) CK(cudaFuncSetAttribute((const void*)v.k, cudaFuncAttributeMaxDynamicSharedMemorySize, (v.bm + v.bn) * v.bk * 2 * v.st));
    for (auto& v : vs) CK(cudaFuncSetAttribute((const void*)v.k, cudaFuncAttributeMaxDynamicSharedMemorySize, (v.bm + v.bn) * v.bk * 2 * v.st));
    struct S { const char* n; int N, K; } shapes[] = {{"gate+up", 24576, 4096}, {"down", 4096, 12288}, {"qkv", 6144, 4096}, {"o", 4096, 4096}};
    cublasHandle_t h;
    cublasCreate(&h);
    Nvml nv;
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int M : Ms) {
        for (auto& s : shapes) {
            size_t xn = (size_t)M * s.K, wn = (size_t)s.N * s.K, yn = (size_t)M * s.N;
            std::vector<bf16> hx(xn), hw(wn);
            for (auto& x : hx) x = __float2bfloat16(nd(rng));
            for (auto& x : hw) x = __float2bfloat16(nd(rng) * 0.05f);
            bf16 *dx, *dw, *dy, *dr;
            CK(cudaMalloc(&dx, xn * 2));
            CK(cudaMalloc(&dw, wn * 2));
            CK(cudaMalloc(&dy, yn * 2));
            CK(cudaMalloc(&dr, yn * 2));
            CK(cudaMemcpy(dx, hx.data(), xn * 2, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dw, hw.data(), wn * 2, cudaMemcpyHostToDevice));
            float al = 1.f, be = 0.f;
            auto run_blas = [&]() { cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, s.N, M, s.K, &al, dw, CUDA_R_16BF, s.K, dx, CUDA_R_16BF, s.K, &be, dr, CUDA_R_16BF, s.N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT); };
            // sustained: the GPU runs the kernel for 0.4 s first (its clock settles at its power cap), then the timed run of at least 0.4 s
            auto timeit = [&](auto f, int) {
                cudaEvent_t a, b;
                cudaEventCreate(&a);
                cudaEventCreate(&b);
                for (int pass = 0; pass < 2; pass++) {
                    int reps = 0;
                    cudaEventRecord(a);
                    float ms = 0;
                    while (ms < (getenv("GEMM_FAST") ? 200.f : 400.f)) {
                        for (int i = 0; i < 4; i++) f();
                        reps += 4;
                        cudaEventRecord(b);
                        cudaEventSynchronize(b);
                        cudaEventElapsedTime(&ms, a, b);
                    }
                    if (pass == 1) return ms / reps;
                }
                return 0.f;
            };
            double flop = 2.0 * M * s.N * s.K;
            float tb = timeit(run_blas, 10);
            std::vector<bf16> ref(yn);
            CK(cudaMemcpy(ref.data(), dr, yn * 2, cudaMemcpyDeviceToHost));
            unsigned mhz, pw, tc;
            nv.read(mhz, pw, tc);
            printf("M=%d %-8s N=%d K=%d: cuBLAS %.3f ms (%.1f TFLOPS) [%u MHz, %u W, %u C]\n", M, s.n, s.N, s.K, tb, flop / (tb * 1e-3) / 1e12, mhz, pw, tc);
            std::vector<bf16> first;
            for (auto& v : vs) {
                int tiles = ((M + v.bm - 1) / v.bm) * (s.N / v.bn);
                auto run = [&]() { v.k<<<tiles, v.th, (v.bm + v.bn) * v.bk * 2 * v.st>>>(dx, dw, dy, M, s.N, s.K); };
                CK(cudaMemset(dy, 0, yn * 2));
                run();
                CK(cudaGetLastError());
                CK(cudaDeviceSynchronize());
                std::vector<bf16> got(yn);
                CK(cudaMemcpy(got.data(), dy, yn * 2, cudaMemcpyDeviceToHost));
                // the same bits as the first variant's (a tile's shape and the loop around it must not change a sum)
                size_t bitdiff = 0;
                if (first.empty()) first = got;
                else for (size_t i = 0; i < yn; i++) bitdiff += memcmp(&got[i], &first[i], 2) != 0;
                double se = 0, sr = 0, mx = 0;
                for (size_t i = 0; i < yn; i++) {
                    double a = __bfloat162float(got[i]), b = __bfloat162float(ref[i]);
                    se += (a - b) * (a - b);
                    sr += b * b;
                    mx = fmax(mx, fabs(a - b));
                }
                float t = timeit(run, 10);
                nv.read(mhz, pw, tc);
                printf("    %-18s %.3f ms (%.1f TFLOPS, %.2fx cuBLAS's time) [%u MHz, %u W, %u C], relative L2 difference from cuBLAS %.2e, largest %.3g, %zu values differ in bits from the first variant's\n", v.name, t, flop / (t * 1e-3) / 1e12, t / tb, mhz, pw, tc, sqrt(se / sr), mx, bitdiff);
            }
            if (s.N == 24576) {
                // the unfused path: cuBLAS then silu_mul over [M, 2I]
                bf16* dm;
                bf16* dref;
                CK(cudaMalloc(&dm, (size_t)M * (s.N / 2) * 2));
                CK(cudaMalloc(&dref, (size_t)M * (s.N / 2) * 2));
                auto run_unfused = [&]() {
                    run_blas();
                    silu_mul<<<dim3(M, (s.N / 2 + 2047) / 2048), 256>>>(dr, dref, s.N / 2);
                };
                run_unfused();
                CK(cudaDeviceSynchronize());
                float tu = timeit(run_unfused, 10);
                nv.read(mhz, pw, tc);
                printf("    cuBLAS + silu_mul   %.3f ms (%.1f TFLOPS) [%u MHz, %u W, %u C]\n", tu, flop / (tu * 1e-3) / 1e12, mhz, pw, tc);
                std::vector<bf16> want((size_t)M * (s.N / 2));
                CK(cudaMemcpy(want.data(), dref, want.size() * 2, cudaMemcpyDeviceToHost));
                for (auto& v : fv) {
                    int tiles = ((M + v.bm - 1) / v.bm) * (s.N / v.bn);
                    auto run = [&]() { v.k<<<tiles, v.th, (v.bm + v.bn) * v.bk * 2 * v.st>>>(dx, dw, dm, M, s.N, s.K); };
                    CK(cudaMemset(dm, 0, want.size() * 2));
                    run();
                    CK(cudaGetLastError());
                    CK(cudaDeviceSynchronize());
                    std::vector<bf16> got(want.size());
                    CK(cudaMemcpy(got.data(), dm, got.size() * 2, cudaMemcpyDeviceToHost));
                    size_t diff = 0;
                    for (size_t i = 0; i < got.size(); i++) diff += memcmp(&got[i], &want[i], 2) != 0;
                    float t = timeit(run, 10);
                    nv.read(mhz, pw, tc);
                    printf("    %-18s %.3f ms (%.1f TFLOPS, %.2fx the unfused time) [%u MHz, %u W, %u C], %zu of %zu values differ from the unfused path's bits\n", v.name, t, flop / (t * 1e-3) / 1e12, t / tu, mhz, pw, tc, diff, got.size());
                }
                cudaFree(dm);
                cudaFree(dref);
            }
            cudaFree(dx);
            cudaFree(dw);
            cudaFree(dy);
            cudaFree(dr);
        }
    }
    return 0;
}
