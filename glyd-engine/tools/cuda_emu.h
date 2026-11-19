// The part of CUDA that kernels/moe.cu and kernels/moe_rows.cu use, on the host, so that the kernels' code can be run and checked without a GPU (tools/moe_emu_test.cpp): a CUDA thread is a std::thread, a block's threads run
// together and the blocks one after another, __syncthreads and a warp's __syncwarp are barriers, a warp's shuffles pass the values through a table apiece. Not CUDA: no memory model, no
// divergence, blocks of a multiple of 32 threads; only enough to run what moe.cu does.
#pragma once
#include <cmath>
#include <cstdlib>
#include <math.h>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __launch_bounds__(...)
#define __shared__ static
#define __align__(n) __attribute__((aligned(n)))
#define __restrict__

struct Dim3 {
    unsigned x = 1, y = 1, z = 1;
};
inline thread_local Dim3 threadIdx, blockIdx;
inline Dim3 blockDim, gridDim;

struct alignas(16) uint4 {
    unsigned x, y, z, w;
};

// A reusable barrier for n threads.
struct Barrier {
    std::mutex m;
    std::condition_variable cv;
    unsigned n = 0, count = 0, gen = 0;
    void wait() {
        std::unique_lock<std::mutex> l(m);
        const unsigned g = gen;
        if (++count == n) {
            count = 0;
            gen++;
            cv.notify_all();
        } else {
            cv.wait(l, [&] { return gen != g; });
        }
    }
};
inline Barrier block_barrier;
inline Barrier warp_barrier[32];
inline uint64_t warp_slot[32][32];

inline void __syncthreads() { block_barrier.wait(); }
inline void __syncwarp(unsigned = 0xffffffffu) { warp_barrier[threadIdx.x >> 5].wait(); }

// Lane l gets the value of lane src(l) of its warp.
template <class T, class F>
T shuffle_from(T v, F src) {
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof(T));
    warp_slot[w][l] = bits;
    warp_barrier[w].wait();
    const uint64_t got = warp_slot[w][src(l)];
    warp_barrier[w].wait();
    T r;
    std::memcpy(&r, &got, sizeof(T));
    return r;
}
template <class T>
T __shfl_sync(unsigned, T v, int from, int = 32) {
    return shuffle_from(v, [&](int) { return from & 31; });
}
template <class T>
T __shfl_xor_sync(unsigned, T v, int m, int = 32) {
    return shuffle_from(v, [&](int l) { return (l ^ m) & 31; });
}
template <class T>
T __shfl_up_sync(unsigned, T v, unsigned d, int = 32) {
    return shuffle_from(v, [&](int l) { return l >= (int)d ? l - (int)d : l; });
}

inline float __uint_as_float(unsigned u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
inline unsigned __float_as_uint(float f) {
    unsigned u;
    std::memcpy(&u, &f, 4);
    return u;
}

// bf16 as the toolkit's __nv_bfloat16: 16 bits, converted to float exactly and from it to nearest, ties to even.
struct bf16 {
    uint16_t x;
};
inline float b2f(bf16 v) { return __uint_as_float((unsigned)v.x << 16); }
inline bf16 f2b(float f) {
    unsigned b = __float_as_uint(f);
    if ((b & 0x7fffffffu) > 0x7f800000u) return bf16{(uint16_t)((b >> 16) | 0x40)};  // NaN stays one
    return bf16{(uint16_t)((b + 0x7fffu + ((b >> 16) & 1)) >> 16)};
}

// A pair of bf16 as the toolkit's __nv_bfloat162 (the first in the low half), and the two operations the sparse experts' decode uses: each half's difference, product, rounded once to bf16.
struct __nv_bfloat162 {
    bf16 x, y;
};
inline __nv_bfloat162 __hsub2(__nv_bfloat162 a, __nv_bfloat162 b) { return {f2b(b2f(a.x) - b2f(b.x)), f2b(b2f(a.y) - b2f(b.y))}; }
inline __nv_bfloat162 __hmul2(__nv_bfloat162 a, __nv_bfloat162 b) { return {f2b(b2f(a.x) * b2f(b.x)), f2b(b2f(a.y) * b2f(b.y))}; }

// The sum over a warp, the same on every lane (engine.cu's).
inline float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Run kernel `k(args)` on a grid of blocks of `block.x` threads.
template <class K, class... A>
void launch(Dim3 grid, Dim3 block, K k, A... a) {
    gridDim = grid;
    blockDim = block;
    const unsigned n = block.x;
    if (n % 32 != 0 || n > 1024) {
        std::printf("emulation: a block of %u threads\n", n);
        std::abort();
    }
    block_barrier.n = n;
    for (unsigned w = 0; w < n / 32; w++) warp_barrier[w].n = 32;
    for (unsigned bz = 0; bz < grid.z; bz++)
        for (unsigned by = 0; by < grid.y; by++)
            for (unsigned bx = 0; bx < grid.x; bx++) {
                std::vector<std::thread> ts;
                for (unsigned t = 0; t < n; t++)
                    ts.emplace_back([=] {
                        threadIdx = Dim3{t, 0, 0};
                        blockIdx = Dim3{bx, by, bz};
                        k(a...);
                    });
                for (auto& t : ts) t.join();
            }
}
