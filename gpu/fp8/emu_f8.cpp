// glyd_gpu.cu's FP8 kernels (the text between `f8c begin` and `f8c end`, put in f8_block.inc by emu.py) run on the CPU, thread for thread: a CTA is 256
// (or 1024) std::threads with a barrier for __syncthreads and for every warp's shuffles, the CTAs one after the other, shared memory a static.
// No GPU is needed; what it checks is the layout (against fp8ref.py) and the kernels' logic and synchronization, not their speed or
// the hardware's rules. Loads by __ldg and __ldcs, and the 16-byte copies, are checked for alignment (F8_ALIGN).
//   python3 emu.py            builds it into a temporary directory and runs the cases (emu.py's own check)
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <mutex>
#include <thread>
#include <vector>

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __noinline__
#define __launch_bounds__(...)
#define __shared__ static
#define __align__(n) __attribute__((aligned(n)))
#define F8_ALIGN(p, n) \
    do { if ((uintptr_t)(p) % (n)) { std::fprintf(stderr, "emu: %p is not %d-byte aligned (line %d)\n", (const void*)(p), (int)(n), __LINE__); std::abort(); } } while (0)

struct dim3 {
    unsigned x, y, z;
    dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
};
struct uint2 { unsigned x, y; };
struct uint4 { unsigned x, y, z, w; };
static uint2 make_uint2(unsigned x, unsigned y) { return {x, y}; }
static uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w) { return {x, y, z, w}; }
static thread_local dim3 threadIdx;
static dim3 blockIdx;  // (the CTAs run one at a time)

class Barrier {
    std::mutex m;
    std::condition_variable cv;
    int n, count = 0;
    unsigned gen = 0;

public:
    explicit Barrier(int n_) : n(n_) {}
    void wait() {
        std::unique_lock<std::mutex> l(m);
        const unsigned g = gen;
        if (++count == n) {
            count = 0;
            gen++;
            cv.notify_all();
        } else {
            cv.wait(l, [&] { return g != gen; });
        }
    }
};
static Barrier* g_cta;
static std::vector<Barrier*> g_warp;
static unsigned g_slot[1024];
static int g_nt;  // (the CTA's threads)

static void __syncthreads() { g_cta->wait(); }
static unsigned __shfl_up_sync(unsigned, unsigned v, int d) {
    const int t = threadIdx.x, lane = t & 31, w = t >> 5;
    g_slot[t] = v;
    g_warp[w]->wait();
    const unsigned r = lane >= d ? g_slot[t - d] : v;
    g_warp[w]->wait();
    return r;
}
static void __syncwarp(unsigned = 0xffffffffu) { g_warp[threadIdx.x >> 5]->wait(); }
static unsigned __shfl_sync(unsigned, unsigned v, int src) {
    const int t = threadIdx.x, w = t >> 5;
    g_slot[t] = v;
    g_warp[w]->wait();
    const unsigned r = g_slot[(w << 5) + (src & 31)];
    g_warp[w]->wait();
    return r;
}
static float __shfl_xor_sync(unsigned, float v, int m) {
    const int t = threadIdx.x, w = t >> 5;
    unsigned u;
    std::memcpy(&u, &v, 4);
    g_slot[t] = u;
    g_warp[w]->wait();
    const unsigned r = g_slot[(w << 5) + ((t & 31) ^ m)];
    g_warp[w]->wait();
    float f;
    std::memcpy(&f, &r, 4);
    return f;
}
static unsigned __shfl_xor_sync(unsigned, unsigned v, int m) {
    const int t = threadIdx.x, w = t >> 5;
    g_slot[t] = v;
    g_warp[w]->wait();
    const unsigned r = g_slot[(w << 5) + ((t & 31) ^ m)];
    g_warp[w]->wait();
    return r;
}
static bool __any_sync(unsigned, bool p) {
    const int t = threadIdx.x, w = t >> 5;
    g_slot[t] = p;
    g_warp[w]->wait();
    bool r = false;
    for (int i = 0; i < 32; i++) r |= g_slot[(w << 5) + i] != 0;
    g_warp[w]->wait();
    return r;
}
static int __syncthreads_or(int p) {  // (every thread's slot, a barrier, the OR, a barrier before the slots are written again)
    const int t = threadIdx.x;
    g_slot[t] = p != 0;
    g_cta->wait();
    int r = 0;
    for (int i = 0; i < g_nt; i++) r |= (int)g_slot[i];
    g_cta->wait();
    return r;
}
static void __threadfence() {}
static unsigned __float_as_uint(float f) { unsigned u; std::memcpy(&u, &f, 4); return u; }
static float __int_as_float(unsigned u) { float f; std::memcpy(&f, &u, 4); return f; }
static int min(int a, int b) { return a < b ? a : b; }
static unsigned min(unsigned a, unsigned b) { return a < b ? a : b; }
// mma.sync.m16n8k16 (bf16 in, fp32 sums): every lane's fragments put down, then each lane's C from them (the sums in k order).
static unsigned g_mma[1024][10];
static float bfv(unsigned bits16) { return __int_as_float((bits16 & 0xFFFFu) << 16); }
static void f8_emu_mma(float (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    unsigned* m = g_mma[tid];
    m[0] = a0, m[1] = a1, m[2] = a2, m[3] = a3, m[4] = b0, m[5] = b1;
    g_warp[w]->wait();
    auto A = [&](int r, int k) {  // A [16 x 16]: lane (r % 8, (k % 8) / 2): a0 (r < 8, k < 8), a1 (r >= 8, k < 8), a2 (r < 8, k >= 8), a3
        const unsigned* q = g_mma[(w << 5) + (r & 7) * 4 + ((k & 7) >> 1)];
        const unsigned v = q[(r >= 8 ? 1 : 0) + (k >= 8 ? 2 : 0)];
        return bfv((k & 1) ? v >> 16 : v);
    };
    auto B = [&](int k, int n) {  // B [16 x 8]: lane (n, (k % 8) / 2): b0 (k < 8), b1
        const unsigned v = g_mma[(w << 5) + n * 4 + ((k & 7) >> 1)][k >= 8 ? 5 : 4];
        return bfv((k & 1) ? v >> 16 : v);
    };
    float r[4];
    for (int i = 0; i < 4; i++) {
        const int row = g + 8 * (i >> 1), col = 2 * t + (i & 1);
        float s = c[i];
        for (int k = 0; k < 16; k++) s += A(row, k) * B(k, col);
        r[i] = s;
    }
    g_warp[w]->wait();
    for (int i = 0; i < 4; i++) c[i] = r[i];
}
static float e4m3v(unsigned b) {  // an e4m3 byte's value (0x7F, 0xFF: NaN)
    b &= 0xFF;
    if ((b & 0x7F) == 0x7F) return __builtin_nanf("");
    const int e = (b >> 3) & 15, m = b & 7;
    const float v = e ? std::ldexp(1.f + m / 8.f, e - 7) : std::ldexp(m / 8.f, -6);
    return (b & 0x80) ? -v : v;
}
// mma.sync.m16n8k32 e4m3: A [16 x 32] (lane (r % 8, (k % 16) / 4): a0 (r < 8, k < 16), a1, a2 (k >= 16), a3), B [32 x 8] (lane (n, (k % 16) / 4): b0 (k < 16), b1).
static void f8_emu_mma8(float (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    unsigned* m = g_mma[tid];
    m[0] = a0, m[1] = a1, m[2] = a2, m[3] = a3, m[4] = b0, m[5] = b1;
    g_warp[w]->wait();
    auto A = [&](int r, int k) { return e4m3v(g_mma[(w << 5) + (r & 7) * 4 + ((k & 15) >> 2)][(r >= 8 ? 1 : 0) + (k >= 16 ? 2 : 0)] >> (8 * (k & 3))); };
    auto B = [&](int k, int n) { return e4m3v(g_mma[(w << 5) + n * 4 + ((k & 15) >> 2)][k >= 16 ? 5 : 4] >> (8 * (k & 3))); };
    float r[4];
    for (int i = 0; i < 4; i++) {
        const int row = g + 8 * (i >> 1), col = 2 * t + (i & 1);
        float sum = c[i];
        for (int k = 0; k < 32; k++) sum += A(row, k) * B(k, col);
        r[i] = sum;
    }
    g_warp[w]->wait();
    for (int i = 0; i < 4; i++) c[i] = r[i];
}
// A float as e4m3 (round to nearest even; past 448: 448, satfinite; NaN: 0x7F).
static unsigned f8_emu_e4m3(float v) {
    if (v != v) return 0x7F;
    const unsigned s = std::signbit(v) ? 0x80u : 0u;
    const double a = std::fabs((double)v);
    if (a >= 448.0) return s | 0x7E;
    unsigned best = 0;
    double bd = 1e300;
    for (unsigned b = 0; b < 0x7F; b++) {  // the nearest of the 127 magnitudes, ties to the even byte
        const double d = std::fabs((double)e4m3v(b) - a);
        if (d < bd || (d == bd && !(b & 1))) bd = d, best = b;
    }
    return s | best;
}
static int __popc(unsigned x) { return __builtin_popcount(x); }
static int __popcll(unsigned long long x) { return __builtin_popcountll(x); }
static int __ffs(int x) { return __builtin_ffs(x); }
static unsigned __byte_perm(unsigned x, unsigned y, unsigned s) {
    const uint64_t v = ((uint64_t)y << 32) | x;
    unsigned r = 0;
    for (int i = 0; i < 4; i++) {
        const unsigned sel = (s >> (4 * i)) & 15;
        unsigned b = (unsigned)(v >> (8 * (sel & 7))) & 0xFF;
        if (sel & 8) b = (b & 0x80) ? 0xFF : 0;
        r |= b << (8 * i);
    }
    return r;
}
static unsigned __funnelshift_r(unsigned lo, unsigned hi, unsigned s) {
    s &= 31;
    return s ? (lo >> s) | (hi << (32 - s)) : lo;
}
static unsigned __funnelshift_rc(unsigned lo, unsigned hi, unsigned s) {
    if (s >= 32) return hi;
    return s ? (lo >> s) | (hi << (32 - s)) : lo;
}
static unsigned __funnelshift_l(unsigned lo, unsigned hi, unsigned s) {
    s &= 31;
    return s ? (hi << s) | (lo >> (32 - s)) : hi;
}
static unsigned __ldg(const unsigned* p) { F8_ALIGN(p, 4); return *p; }
static uint2 __ldg(const uint2* p) { F8_ALIGN(p, 8); return *p; }
static float __ldg(const float* p) { F8_ALIGN(p, 4); return *p; }
static float __ldcg(const float* p) { F8_ALIGN(p, 4); return *p; }
static int atomicAdd(int* p, int v) { return __atomic_fetch_add(p, v, __ATOMIC_RELAXED); }
static uint4 __ldg(const uint4* p) { F8_ALIGN(p, 16); return *p; }
static uint2 __ldcs(const uint2* p) { F8_ALIGN(p, 8); return *p; }
static uint4 __ldcs(const uint4* p) { F8_ALIGN(p, 16); return *p; }
static unsigned atomicAdd(unsigned* p, unsigned v) { return __atomic_fetch_add(p, v, __ATOMIC_RELAXED); }
static unsigned atomicOr(unsigned* p, unsigned v) { return __atomic_fetch_or(p, v, __ATOMIC_RELAXED); }

#define F8_EXTERN_SMEM(T, name) alignas(16) static T name[1 << 16]
#define F8_EMU 1
#include "f8_block.inc"

// One launch: grid.x * grid.y CTAs of nt threads, in order; every thread runs body().
template <class F>
static void launch(dim3 grid, int nt, F body) {
    Barrier cta(nt);
    g_nt = nt;
    g_cta = &cta;
    g_warp.clear();
    for (int w = 0; w < nt / 32; w++) g_warp.push_back(new Barrier(32));
    std::vector<std::thread> th;
    for (int t = 0; t < nt; t++)
        th.emplace_back([&, t] {
            threadIdx = dim3(t, 0, 0);
            for (unsigned by = 0; by < grid.y; by++)
                for (unsigned bx = 0; bx < grid.x; bx++) {
                    if (t == 0) blockIdx = dim3(bx, by, 0);
                    cta.wait();  // (blockIdx is set before any thread reads it)
                    body();
                    cta.wait();  // (and the CTA done by all before the next starts)
                }
        });
    for (auto& x : th) x.join();
    for (auto* b : g_warp) delete b;
    g_warp.clear();
}

static int vec_of(const void* p, int64_t ld, int vec) { return vec >= 0 ? vec : ((uintptr_t)p % 16 == 0 && ld % 16 == 0); }

extern "C" {
// glyd_gpu_f8_linear's kernel (the plan given: ntl, ctas, ahead, dense, a8: X as e4m3, f8_xq_kernel's, for kinds 0, 1 and 4), on the host's memory.
int emu_linear(const uint8_t* core, const uint32_t* heads, const uint32_t* var, int64_t N, int64_t K, int kind, const void* scale, int64_t lds,
               const uint16_t* x, int64_t ldx, int64_t M, uint16_t* y, int64_t ldy, float* parts, int* done, int ntl, int ctas, int ahead, int dense, int a8) {
    const int RG = (int)f8c::f8_cdiv(f8c::f8_cdiv(N, f8c::PR) * 4, f8c::LW), KT = (int)f8c::f8_cdiv(K, f8c::PC);
    a8 = a8 && (kind == 0 || kind == 1 || kind == 4);
    std::vector<uint8_t> xq(a8 ? (size_t)M * KT * f8c::PC + 16 : 16);
    std::vector<float> sx(a8 ? (size_t)M * KT : 1);
    uint8_t* xqa = (uint8_t*)(((uintptr_t)xq.data() + 15) & ~(uintptr_t)15);
    if (a8) launch(dim3((unsigned)f8c::f8_cdiv(M * KT, 4)), 128, [&] { f8c::f8_xq_kernel(x, ldx, (int)M, (int)K, KT, xqa, sx.data()); });
    f8c::F8Lin P{core, heads, var, scale, x, y, xqa, sx.data(), parts, done, lds, ldx, ldy, (int)M, (int)N, (int)K, KT, RG, ctas, ((uintptr_t)x % 16 == 0 && ldx % 8 == 0) ? 1 : 0};
#define EMU_LIN(NT_, KD_, AH_, DN_) launch(dim3((unsigned)ctas), f8c::LNT, [&] { f8c::f8_linear_kernel<NT_, KD_, AH_, DN_>(P); })
#define EMU_KIND(NT_, AH_, DN_) switch (kind) { case 0: EMU_LIN(NT_, 0, AH_, DN_); break; case 1: EMU_LIN(NT_, 1, AH_, DN_); break; case 2: EMU_LIN(NT_, 2, AH_, DN_); break; \
                                                case 3: EMU_LIN(NT_, 3, AH_, DN_); break; default: EMU_LIN(NT_, 4, AH_, DN_); }
#define EMU_A8(NT_) switch (kind) { case 0: EMU_LIN(NT_, 8, 0, 0); break; case 1: EMU_LIN(NT_, 9, 0, 0); break; default: EMU_LIN(NT_, 12, 0, 0); }
#define EMU_DN(NT_, AH_) if (dense) { EMU_KIND(NT_, AH_, 1); } else { EMU_KIND(NT_, AH_, 0); }
    if (a8) {
        switch (ntl) { case 1: EMU_A8(1); break; case 2: EMU_A8(2); break; case 4: EMU_A8(4); break; default: EMU_A8(8); }
        return 0;
    }
    switch (ntl) {
        case 1: EMU_DN(1, 0); break;
        case 2: EMU_DN(2, 0); break;
        case 4: if (ahead) { EMU_DN(4, 1); } else { EMU_DN(4, 0); } break;
        default: if (ahead) { EMU_DN(8, 1); } else { EMU_DN(8, 0); }
    }
    return 0;
}
// glyd_gpu_f8_size's and glyd_gpu_f8_pack's on the host's memory (vec: 1 or 0, or -1: as the library chooses), the size in *words.
int emu_size(const uint8_t* w, int64_t ld, int64_t O, int64_t K, uint32_t* heads, int64_t* words, int vec) {
    const int64_t patches = f8c::f8_cdiv(O, f8c::PR) * f8c::f8_cdiv(K, f8c::PC);
    const f8c::F8Pk P{w, ld, (int)O, (int)K, (int)f8c::f8_cdiv(K, f8c::PC), vec_of(w, ld, vec)};
    uint64_t total = 0;
    launch(dim3((unsigned)patches), f8c::NT, [&] { f8c::f8_head_kernel(P, heads); });
    launch(dim3(1), f8c::SCAN_NT, [&] { f8c::f8_scan_kernel(heads, patches, &total); });
    *words = (int64_t)total + f8c::GUARD;
    return 0;
}
int emu_pack(const uint8_t* w, int64_t ld, int64_t O, int64_t K, const uint32_t* heads, uint8_t* core, uint32_t* var, int64_t var_words, int vec) {
    const int64_t patches = f8c::f8_cdiv(O, f8c::PR) * f8c::f8_cdiv(K, f8c::PC);
    const f8c::F8Pk P{w, ld, (int)O, (int)K, (int)f8c::f8_cdiv(K, f8c::PC), vec_of(w, ld, vec)};
    launch(dim3((unsigned)patches), f8c::NT, [&] { f8c::f8_pack_kernel(P, heads, core, var, var_words); });
    return 0;
}
// glyd_gpu_f8_unpack (ids null) and glyd_gpu_f8_unpack_experts.
int emu_unpack(const uint8_t* core, const uint32_t* heads, const uint32_t* var, int64_t E, int64_t O, int64_t K, const int32_t* ids, int64_t n, uint8_t* out, int64_t ld, int vec) {
    const int64_t patches = f8c::f8_cdiv(O, f8c::PR) * f8c::f8_cdiv(K, f8c::PC);
    const f8c::F8Un P{core, heads, var, out, ids, ld, O * ld, (int)O, (int)K, (int)f8c::f8_cdiv(K, f8c::PC), (int)patches, (int)E, vec_of(out, ld, vec)};
    launch(dim3((unsigned)patches, (unsigned)n), f8c::NT, [&] { f8c::f8_unpack_kernel(P); });
    return 0;
}
}
