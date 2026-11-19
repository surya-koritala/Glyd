// stream.cu: how fast a one-token product's weight bytes can be read on this GPU, by access pattern (research, 2026-10-20).
// The product's plain layout: a patch of 8,192 bytes (64 rows x 128 columns, a byte a value), four row blocks of 2,048 bytes; a warp reads one row
// block a step (32 lanes x 64 bytes, as 4 loads of 16 bytes a lane, each load instruction 512 contiguous bytes). The engine's kernel gives CTA c the
// units [c W / ctas, (c + 1) W / ctas) of W = RG x KT units; a unit (r, k) is two patches ((2 r + half) KT + k, half 0 and 1) and warp (half, mb)
// reads row block mb of its patch. The variants below change only the order and the mechanism, not the bytes read.
//   stream4 / stream8   a plain grid-stride read, 4 or 8 loads of 16 bytes a thread in flight (the card's achievable rate)
//   ours1 / ours2       the engine's order (a CTA's contiguous range of units, two streams a CTA 256 KB apart), plain loads, 1 or 2 steps ahead
//   cpasync             the engine's order and mechanism: cp.async into two shared-memory slots a warp, wait_group 1, read back
//   adjacent            the engine's order with the two halves on adjacent patches (one 16 KB span a CTA a step)
//   interleaved         unit j ctas + c at step j (consecutive CTAs read consecutive memory, the whole grid a moving window)
//   wide                interleaved with 16 warps a CTA (32 KB a CTA a step)
//   many                ours1 with 6 CTAs an SM (no shared memory: occupancy by registers alone)
// Prints GB/s per variant (the median of reps) and a checksum (equal across variants).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e_)); exit(1); } } while (0)

__device__ __forceinline__ uint4 ldg16(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ uint32_t x4(uint4 v) { return v.x ^ v.y ^ v.z ^ v.w; }
__device__ __forceinline__ void done(uint32_t acc, uint32_t* out) {
    for (int o = 16; o; o >>= 1) acc ^= __shfl_xor_sync(0xffffffffu, acc, o);
    if ((threadIdx.x & 31) == 0) atomicXor(out, acc);
}

template <int U>
__global__ void __launch_bounds__(256) k_stream(const uint4* __restrict__ a, size_t n16, uint32_t* out) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    uint32_t acc = 0;
    for (; i + (U - 1) * stride < n16; i += U * stride) {
        uint4 v[U];
#pragma unroll
        for (int u = 0; u < U; u++) v[u] = ldg16(a + i + u * stride);
#pragma unroll
        for (int u = 0; u < U; u++) acc ^= x4(v[u]);
    }
    for (; i < n16; i += stride) acc ^= x4(ldg16(a + i));
    done(acc, out);
}

// ORDER 0: the engine's (a CTA's contiguous range; halves KT patches apart). 1: adjacent halves (patch 2 u and 2 u + 1 of a unit u, so a 16 KB span).
// 2: interleaved (unit j ctas + c; halves adjacent). WPC warps a CTA (8: a unit a step; 16: two units a step).
template <int ORDER, int WPC>
__device__ __forceinline__ const uint8_t* unit_ptr(const uint8_t* a, int64_t W, int KT, int ctas, int c, int j, int warp, bool& ok) {
    const int units_per_step = WPC / 8, half = (warp >> 2) & 1, mb = warp & 3, sub = warp >> 3;
    int64_t u;
    if (ORDER == 2) {
        u = ((int64_t)j * ctas + c) * units_per_step + sub;
        ok = u < W;
    } else {
        const int64_t a0 = (int64_t)c * W / ctas, a1 = (int64_t)(c + 1) * W / ctas;
        u = a0 + (int64_t)j * units_per_step + sub;
        ok = u < a1;
    }
    int64_t patch;
    if (ORDER == 0) {
        const int64_t r = u / KT, k = u % KT;
        patch = (2 * r + half) * KT + k;
    } else {
        patch = 2 * u + half;
    }
    return a + patch * 8192 + (int64_t)mb * 2048;
}

// plain loads, PF steps ahead in registers
template <int ORDER, int WPC, int PF>
__global__ void __launch_bounds__(WPC * 32) k_pattern(const uint8_t* __restrict__ a, int64_t W, int KT, int ctas, uint32_t* out) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, c = blockIdx.x;
    const int units_per_step = WPC / 8;
    int n;
    if (ORDER == 2) n = (int)((W - c * units_per_step + (int64_t)ctas * units_per_step - 1) / ((int64_t)ctas * units_per_step));
    else n = (int)(((int64_t)(c + 1) * W / ctas - (int64_t)c * W / ctas + units_per_step - 1) / units_per_step);
    uint4 v[PF + 1][4];
    uint32_t acc = 0;
    auto load = [&](int j, uint4 (&d)[4]) {
        bool ok;
        const uint8_t* p = unit_ptr<ORDER, WPC>(a, W, KT, ctas, c, j, warp, ok);
        if (ok) {
#pragma unroll
            for (int q = 0; q < 4; q++) d[q] = ldg16(p + q * 512 + lane * 16);
        } else {
#pragma unroll
            for (int q = 0; q < 4; q++) d[q] = make_uint4(0, 0, 0, 0);
        }
    };
#pragma unroll
    for (int s = 0; s < PF; s++) load(s, v[s]);
    for (int j = 0; j < n; j++) {
        load(j + PF, v[(j + PF) % (PF + 1)]);
        const uint4 (&cur)[4] = v[j % (PF + 1)];
#pragma unroll
        for (int q = 0; q < 4; q++) acc ^= x4(cur[q]);
    }
    done(acc, out);
}

// the engine's mechanism: cp.async.cg 16 bytes a lane x 4 into one of two 2 KB slots a warp, commit, wait_group 1 (the step before next may be in flight)
__device__ __forceinline__ void cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src));
}
template <int ORDER>
__global__ void __launch_bounds__(256) k_cpasync(const uint8_t* __restrict__ a, int64_t W, int KT, int ctas, uint32_t* out, int slot_bytes) {
    extern __shared__ __align__(16) uint8_t sm[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, c = blockIdx.x;
    uint8_t* mine = sm + (size_t)warp * 2 * slot_bytes;
    const int n = (int)(((int64_t)(c + 1) * W / ctas - (int64_t)c * W / ctas));
    auto issue = [&](int j) {
        bool ok;
        const uint8_t* p = unit_ptr<ORDER, 8>(a, W, KT, ctas, c, j, warp, ok);
        uint8_t* d = mine + (j & 1) * slot_bytes;
        if (ok) {
#pragma unroll
            for (int q = 0; q < 4; q++) cp16(d + (q * 32 + lane) * 16, p + q * 512 + lane * 16);
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    };
    uint32_t acc = 0;
    issue(0);
    for (int j = 0; j < n; j++) {
        if (j + 1 < n) issue(j + 1); else asm volatile("cp.async.commit_group;\n" ::: "memory");
        asm volatile("cp.async.wait_group 1;\n" ::: "memory");
        __syncwarp();
        const uint4* s = (const uint4*)(mine + (j & 1) * slot_bytes);
#pragma unroll
        for (int q = 0; q < 4; q++) acc ^= x4(s[q * 32 + lane]);
        __syncwarp();  // the slot free before the copy after next is issued (the next iteration's issue writes slot (j + 2) & 1 == j & 1)
    }
    done(acc, out);
}

struct Res { const char* name; double gbs; uint32_t sum; };

int main(int argc, char** argv) {
    const int64_t N = argc > 1 ? atoll(argv[1]) : 8192, K = argc > 2 ? atoll(argv[2]) : 4096, reps = argc > 3 ? atoi(argv[3]) : 11;
    const int KT = (int)(K / 128), RG = (int)((N / 64 + 1) / 2);  // units of two patches' row blocks: rows / 64 patches a column, two a unit
    const int64_t patches = (N / 64) * KT, bytes = patches * 8192, W = (int64_t)RG * KT;
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    const int sms = prop.multiProcessorCount;
    int mclk = 0, mbus = 0;
    CK(cudaDeviceGetAttribute(&mclk, cudaDevAttrMemoryClockRate, 0));
    CK(cudaDeviceGetAttribute(&mbus, cudaDevAttrGlobalMemoryBusWidth, 0));
    printf("%s: %d SMs, %.0f GB/s peak (%.1f MB a matrix, %lld patches, W %lld, KT %d)\n", prop.name, sms, 2.0 * mclk * 1e3 * (mbus / 8) / 1e9, bytes / 1e6, (long long)patches, (long long)W, KT);
    // copies of the matrix, used in turn, so a rep never reads what the one before left in L2 (8 x L2 at least)
    const int copies = (int)std::max<int64_t>(2, (8LL * prop.l2CacheSize + bytes - 1) / bytes);
    printf("L2 %d MB: %d copies of the matrix in turn\n", prop.l2CacheSize >> 20, copies);
    std::vector<uint8_t*> as(copies);
    uint32_t* out;
    CK(cudaMalloc(&out, 4));
    {   // pseudo-random bytes, the same in every copy
        std::vector<uint32_t> h(bytes / 4);
        uint32_t s = 1;
        for (auto& w : h) { s = s * 1664525u + 1013904223u; w = s; }
        for (int i = 0; i < copies; i++) {
            CK(cudaMalloc(&as[i], bytes));
            CK(cudaMemcpy(as[i], h.data(), bytes, cudaMemcpyHostToDevice));
        }
    }
    const uint8_t* a = as[0];  // (reassigned a rep)
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    std::vector<Res> rs;
    auto time = [&](const char* name, auto launch) {
        std::vector<float> ts;
        uint32_t sum = 0;
        for (int r = 0; r < reps; r++) {
            a = as[r % copies];
            CK(cudaMemset(out, 0, 4));
            CK(cudaEventRecord(e0));
            launch();
            CK(cudaEventRecord(e1));
            CK(cudaEventSynchronize(e1));
            CK(cudaGetLastError());
            float ms;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            ts.push_back(ms);
            CK(cudaMemcpy(&sum, out, 4, cudaMemcpyDeviceToHost));
        }
        std::sort(ts.begin(), ts.end());
        const double med = ts[ts.size() / 2];
        rs.push_back({name, bytes / med / 1e6, sum});
        printf("%-12s %8.1f us  %6.0f GB/s  (min %.1f)  sum %08x\n", name, med * 1e3, bytes / med / 1e6, ts[0] * 1e3, sum);
    };
    const size_t n16 = bytes / 16;
    time("stream4", [&] { k_stream<4><<<sms * 8, 256>>>((const uint4*)a, n16, out); });
    time("stream8", [&] { k_stream<8><<<sms * 8, 256>>>((const uint4*)a, n16, out); });
    time("stream4x16", [&] { k_stream<4><<<sms * 16, 256>>>((const uint4*)a, n16, out); });
    const int c3 = (int)std::min<int64_t>(sms * 3, W), c6 = (int)std::min<int64_t>(sms * 6, W), c1 = (int)std::min<int64_t>(sms, W);
    time("ours1", [&] { k_pattern<0, 8, 1><<<c3, 256>>>(a, W, KT, c3, out); });
    time("ours2", [&] { k_pattern<0, 8, 2><<<c3, 256>>>(a, W, KT, c3, out); });
    const int slot = 2048;
    CK(cudaFuncSetAttribute(k_cpasync<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, 48 * 1024));
    time("cpasync", [&] { k_cpasync<0><<<c3, 256, 47 * 1024>>>(a, W, KT, c3, out, slot); });   // 47 KB a CTA as the engine's kernel: 3 an SM
    time("cpasync32k", [&] { k_cpasync<0><<<c3, 256, 32 * 1024>>>(a, W, KT, c3, out, slot); });
    time("adjacent", [&] { k_pattern<1, 8, 1><<<c3, 256>>>(a, W, KT, c3, out); });
    time("interleaved", [&] { k_pattern<2, 8, 1><<<c3, 256>>>(a, W, KT, c3, out); });
    time("interleav2", [&] { k_pattern<2, 8, 2><<<c3, 256>>>(a, W, KT, c3, out); });
    const int c3w = (int)std::min<int64_t>(sms * 3, (W + 1) / 2);
    time("wide", [&] { k_pattern<2, 16, 1><<<c3w, 512>>>(a, W, KT, c3w, out); });
    time("many", [&] { k_pattern<0, 8, 1><<<c6, 256>>>(a, W, KT, c6, out); });
    time("many-inter", [&] { k_pattern<2, 8, 1><<<c6, 256>>>(a, W, KT, c6, out); });
    time("one-cta-sm", [&] { k_pattern<0, 8, 2><<<c1, 256>>>(a, W, KT, c1, out); });
    time("cpasync-adj", [&] { k_cpasync<1><<<c3, 256, 47 * 1024>>>(a, W, KT, c3, out, slot); });
    bool same = true;
    for (auto& r : rs) same &= r.sum == rs[0].sum;
    printf("checksums %s\n", same ? "equal" : "DIFFER");
    return same ? 0 : 2;
}
