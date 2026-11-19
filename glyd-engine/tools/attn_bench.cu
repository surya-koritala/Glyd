// The prompt attention kernel on its own: its time and its difference from a plain fp32 attention.
//   nvcc -O3 -std=c++17 -arch=sm_89 -lineinfo tools/attn_bench.cu -o attn_bench
//   ./attn_bench W BN n start [reps] [check]       W: warps (4 or 8), BN: keys a tile (32 or 64), the chunk's n tokens after `start` earlier ones
#include "../kernels/engine.cu"
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

// the reference: one thread a (row, head), two passes, fp32
__global__ void ref_attn(const bf16* q, int qs, const bf16* kv, float* out, int n, int start, int heads, int kvh, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y;
    if (i >= n) return;
    int hk = h / (heads / kvh);
    int pos = start + i;
    float qv[HD];
    for (int d = 0; d < HD; d++) qv[d] = b2f(q[(size_t)i * qs + h * HD + d]);
    float mx = -INFINITY;
    for (int j = 0; j <= pos; j++) {
        const bf16* k = kv + ((size_t)j * kvh + hk) * 256;
        float s = 0;
        for (int d = 0; d < HD; d++) s += qv[d] * b2f(k[d]);
        mx = fmaxf(mx, s * scale);
    }
    float acc[HD];
    for (int d = 0; d < HD; d++) acc[d] = 0;
    float l = 0;
    for (int j = 0; j <= pos; j++) {
        const bf16* k = kv + ((size_t)j * kvh + hk) * 256;
        float s = 0;
        for (int d = 0; d < HD; d++) s += qv[d] * b2f(k[d]);
        float p = expf(s * scale - mx);
        l += p;
        for (int d = 0; d < HD; d++) acc[d] += p * b2f(k[HD + d]);
    }
    for (int d = 0; d < HD; d++) out[(size_t)i * heads * HD + h * HD + d] = acc[d] / l;
}

typedef void (*Kern)(const bf16*, int, const bf16*, bf16*, int, int, int, int, int, float);

int main(int argc, char** argv) {
    if (argc < 5) { printf("usage: attn_bench W BN n start [reps] [check]\n"); return 1; }
    int W = atoi(argv[1]), BN = atoi(argv[2]), n = atoi(argv[3]), start = atoi(argv[4]);
    int reps = argc > 5 ? atoi(argv[5]) : 10, check = argc > 6 ? atoi(argv[6]) : 1;
    const int heads = 32, kvh = 8;
    Kern k = nullptr;
    if (W == 4 && BN == 64) k = prefill_attn_w4_b64;
    if (W == 8 && BN == 64) k = prefill_attn_w8_b64;
    if (W == 4 && BN == 32) k = prefill_attn_w4_b32;
    if (W == 8 && BN == 32) k = prefill_attn_w8_b32;
    if (!k) { printf("no such variant\n"); return 1; }
    size_t smem = (size_t)(16 * W + 4 * BN) * 256;
    CK(cudaFuncSetAttribute((const void*)k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    int kv_len = start + n;
    size_t qn = (size_t)n * heads * HD, kn = (size_t)kv_len * kvh * 256;
    std::vector<bf16> hq(qn), hkv(kn);
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& x : hq) x = __float2bfloat16(nd(rng) * 1.5f);
    for (size_t i = 0; i < kn; i++) hkv[i] = __float2bfloat16(nd(rng) * ((i % 256) < 128 ? 1.5f : 1.0f));
    bf16 *dq, *dkv, *dout;
    float* dref;
    CK(cudaMalloc(&dq, qn * 2));
    CK(cudaMalloc(&dkv, kn * 2));
    CK(cudaMalloc(&dout, qn * 2));
    CK(cudaMalloc(&dref, qn * 4));
    CK(cudaMemcpy(dq, hq.data(), qn * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dkv, hkv.data(), kn * 2, cudaMemcpyHostToDevice));
    CK(cudaMemset(dout, 0, qn * 2));
    float sl2 = (1.f / sqrtf(128.f)) * 1.4426950408889634f;
    dim3 grid((n + 16 * W - 1) / (16 * W), heads), block(32 * W);
    k<<<grid, block, smem>>>(dq, heads * HD, dkv, dout, heads * HD, n, start, heads, kvh, sl2);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    if (check) {
        ref_attn<<<dim3((n + 63) / 64, heads), 64>>>(dq, heads * HD, dkv, dref, n, start, heads, kvh, 1.f / sqrtf(128.f));
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        std::vector<bf16> ho(qn);
        std::vector<float> hr(qn);
        CK(cudaMemcpy(ho.data(), dout, qn * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hr.data(), dref, qn * 4, cudaMemcpyDeviceToHost));
        double se = 0, sr = 0, mxd = 0;
        size_t bad = 0;
        for (size_t i = 0; i < qn; i++) {
            double a = __bfloat162float(ho[i]), b = hr[i];
            if (!(a == a)) bad++;
            double d = fabs(a - b);
            se += d * d;
            sr += b * b;
            mxd = fmax(mxd, d);
        }
        printf("check n=%d start=%d: relative L2 error %.3e, largest difference %.3e, NaN %zu\n", n, start, sqrt(se / sr), mxd, bad);
    }
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int i = 0; i < 3; i++) k<<<grid, block, smem>>>(dq, heads * HD, dkv, dout, heads * HD, n, start, heads, kvh, sl2);
    cudaEventRecord(e0);
    for (int i = 0; i < reps; i++) k<<<grid, block, smem>>>(dq, heads * HD, dkv, dout, heads * HD, n, start, heads, kvh, sl2);
    cudaEventRecord(e1);
    CK(cudaEventSynchronize(e1));
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    ms /= reps;
    double pairs = 0;  // (row, key) pairs: row i sees start + i + 1 keys
    pairs = (double)n * start + (double)n * (n + 1) / 2;
    double flop = 4.0 * pairs * HD * heads;
    printf("W=%d BN=%d n=%d start=%d: %.3f ms, %.1f TFLOPS (a layer; x36 = %.1f ms)\n", W, BN, n, start, ms, flop / (ms * 1e-3) / 1e12, ms * 36);
    return 0;
}
