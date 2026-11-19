// kernels/nemotron.cu on the host: each CUDA thread a std::thread, a block's __syncthreads() a barrier, a warp's shuffles an exchange behind two barriers of the warp's 32 threads. No GPU, no nvcc:
// the kernels' arithmetic and indexing are checked against tools/nemotron_ref.py before a pod is paid for. Slow (a block at a time, its threads waiting on each other): small shapes.
// What it cannot say: anything of the machine (bank conflicts, occupancy, instruction counts), the existing prompt attention (inline PTX), the library's products.
//
//   c++ -std=c++20 -O1 -pthread -fno-strict-aliasing -o nh_emu tools/nh_emu.cpp          (-fsanitize=address,undefined or -fsanitize=thread on top: indexing past a buffer, a shared-memory access no barrier orders)
//   python3 tools/nemotron_ref.py kernels gen DIR; ./nh_emu DIR/in DIR/out; python3 tools/nemotron_ref.py kernels check DIR
//
// The same files and the same sequence as `glyd-engine nhkernels` (src/kcheck_nh.rs); see tools/nemotron_ref.py for them. NH_EMU_CASES=chunked,recurrent runs those cases alone (the model's width takes
// minutes for a case of single-token launches; `kernels check` then compares the cases that ran).
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// ---- the CUDA a kernel of nemotron.cu uses ---------------------------------------------------------------------------------------------------------------------------------------
#define __global__
#define __device__
#define __forceinline__ inline
#define __launch_bounds__(...)
#define __align__(n) __attribute__((aligned(n)))
#define __shared__ static
#define NH_EMU 1
struct Dim {
    unsigned x = 1, y = 1, z = 1;
};
static Dim blockIdx, blockDim, gridDim;
static thread_local Dim threadIdx;
// A barrier of a mutex and a condition variable (std::barrier's atomics are not known to ThreadSanitizer, which would see a race in every access to shared memory on the two sides of a __syncthreads():
// with this one, `-fsanitize=thread` reports the accesses of two threads that no barrier orders).
struct Bar {
    std::mutex m;
    std::condition_variable cv;
    int n, waiting = 0;
    unsigned gen = 0;
    explicit Bar(int n) : n(n) {}
    void arrive_and_wait() {
        std::unique_lock<std::mutex> l(m);
        const unsigned g = gen;
        if (++waiting == n) {
            waiting = 0;
            gen++;
            cv.notify_all();
        } else {
            cv.wait(l, [&] { return g != gen; });
        }
    }
};
static Bar* g_block = nullptr;
static std::vector<std::unique_ptr<Bar>> g_warp;
static float g_xchg[64][32];  // (a block of up to 64 warps)
static void __syncthreads() { g_block->arrive_and_wait(); }
template <class T>
static T shfl_gather(T v, int lane, int w, int (*pick)(int, int), int arg) {
    static_assert(sizeof(T) == sizeof(float));
    float f;
    std::memcpy(&f, &v, sizeof f);
    g_xchg[w][lane] = f;
    g_warp[w]->arrive_and_wait();
    float r = g_xchg[w][pick(lane, arg)];
    g_warp[w]->arrive_and_wait();
    T out;
    std::memcpy(&out, &r, sizeof r);
    return out;
}
static int pick_xor(int lane, int off) { return lane ^ off; }
static int pick_src(int, int src) { return src & 31; }
template <class T>
static T __shfl_xor_sync(unsigned, T v, int off) { return shfl_gather(v, threadIdx.x & 31, threadIdx.x >> 5, pick_xor, off); }
template <class T>
static T __shfl_sync(unsigned, T v, int src) { return shfl_gather(v, threadIdx.x & 31, threadIdx.x >> 5, pick_src, src); }

struct uint2 { unsigned x, y; };
struct uint4 { unsigned x, y, z, w; };
struct float2 { float x, y; };
struct float4 { float x, y, z, w; };
static uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w) { return {x, y, z, w}; }
static float2 make_float2(float x, float y) { return {x, y}; }
static float __uint_as_float(unsigned u) { float f; std::memcpy(&f, &u, 4); return f; }
template <class A, class B>
static auto min(A a, B b) {  // (CUDA's min takes an int and an unsigned too)
    using C = std::common_type_t<A, B>;
    return (C)a < (C)b ? (C)a : (C)b;
}

struct bf16 { uint16_t bits; };
static float b2f(bf16 x) { return __uint_as_float((unsigned)x.bits << 16); }
static bf16 f2b(float x) {
    if (std::isnan(x)) return bf16{0x7fc0};
    unsigned b; std::memcpy(&b, &x, 4);
    b += 0x7fff + ((b >> 16) & 1);
    return bf16{(uint16_t)(b >> 16)};
}

// (engine.cu's and qwen35.cu's, as they are)
static float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
static float block_sum(float v) {
    __shared__ float sh[32];
    __syncthreads();
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        float t = threadIdx.x < (blockDim.x >> 5) ? sh[threadIdx.x] : 0.f;
        t = warp_sum(t);
        if (threadIdx.x == 0) sh[0] = t;
    }
    __syncthreads();
    return sh[0];
}
static float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = std::fmax(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
static float2 bf2(uint32_t u) { return make_float2(__uint_as_float(u << 16), __uint_as_float(u & 0xffff0000u)); }
static float softplus_f(float x) { return x > 20.f ? x : std::log1p(std::exp(x)); }
static float fmaxf_(float a, float b) { return std::fmax(a, b); }
#define fmaxf fmaxf_
#define expf std::exp
#define exp2f std::exp2
#define rsqrtf(x) (1.0f / std::sqrt(x))
#define fmaf std::fma

#include "../kernels/nemotron.cu"

// ---- launching -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
static void launch(Dim grid, Dim block, const std::function<void()>& kernel) {
    const int nt = (int)(block.x * block.y * block.z);
    gridDim = grid;
    blockDim = block;
    for (unsigned z = 0; z < grid.z; z++)
        for (unsigned y = 0; y < grid.y; y++)
            for (unsigned x = 0; x < grid.x; x++) {
                blockIdx = {x, y, z};
                Bar bar(nt);
                g_block = &bar;
                g_warp.clear();
                for (int w = 0; w < nt / 32; w++) g_warp.push_back(std::make_unique<Bar>(32));
                std::vector<std::thread> ts;
                for (int i = 0; i < nt; i++)
                    ts.emplace_back([&, i] {
                        threadIdx = {(unsigned)i, 0, 0};
                        kernel();
                    });
                for (auto& t : ts) t.join();
            }
}

// ---- the sequence (src/kcheck_nh.rs's) -------------------------------------------------------------------------------------------------------------------------------------------------
static std::string IN, OUT;

template <class T>
static std::vector<T> load(const std::string& name) {
    std::ifstream f(IN + "/" + name, std::ios::binary);
    if (!f) { std::fprintf(stderr, "nh_emu: no %s/%s\n", IN.c_str(), name.c_str()); std::exit(1); }
    std::vector<char> b((std::istreambuf_iterator<char>(f)), {});
    std::vector<T> v(b.size() / sizeof(T));
    std::memcpy(v.data(), b.data(), v.size() * sizeof(T));
    return v;
}
template <class T>
static void save(const std::string& name, const std::vector<T>& v) {
    std::ofstream f(OUT + "/" + name, std::ios::binary);
    f.write((const char*)v.data(), v.size() * sizeof(T));
}
static std::vector<double> nums(const std::string& name) {
    std::ifstream f(IN + "/" + name);
    std::vector<double> v;
    double x;
    while (f >> x) v.push_back(x);
    return v;
}

static void mamba() {
    const auto m = nums("mamba.txt");
    const int nh = (int)m[0], ng = (int)m[1], n = (int)m[2], split = (int)m[5];
    const float eps = (float)m[3], dt_min = (float)m[4];
    const int di = nh * NH_P, cd = di + 2 * ng * NH_N, W = di + cd + nh;
    auto mix = load<bf16>("mix.bin");
    auto cw = load<bf16>("conv_w.bin"), cb = load<bf16>("conv_b.bin"), alog = load<bf16>("a_log.bin"), dd = load<bf16>("d.bin"), dtb = load<bf16>("dt_bias.bin"), nw = load<bf16>("norm_w.bin");
    const auto state0 = load<float>("state0.bin");
    const auto cstate0 = load<bf16>("cstate0.bin");
    const int nch_max = (n + NH_L - 1) / NH_L + 1;
    const char* names[4] = {"chunked", "recurrent", "steps", "mixed"};
    const char* only = std::getenv("NH_EMU_CASES");  // (e.g. chunked,recurrent: the cases of many single-token launches take long on the host)
    std::string ran;
    for (int c = 0; c < 4; c++) {
        if (only && !std::strstr(only, names[c])) continue;
        ran += std::string(ran.empty() ? "" : " ") + names[c];
        std::vector<bf16> conv((size_t)n * cd), y((size_t)n * di), cst = cstate0;
        std::vector<float> state = state0, dtv((size_t)nch_max * NH_L * nh), acum((size_t)nch_max * NH_L * nh), G((size_t)nch_max * ng * NH_L * NH_L), stb((size_t)nch_max * nh * NH_P * NH_N);
        // calls: (first token, tokens, recurrent kernel or the chunked scan)
        struct Call { int t0, len; bool chunked; };
        std::vector<Call> calls;
        const int a = split > 0 && split < n ? split : n;
        if (c == 0) { calls = {{0, a, true}}; if (a < n) calls.push_back({a, n - a, true}); }
        if (c == 1) { calls = {{0, a, false}}; if (a < n) calls.push_back({a, n - a, false}); }
        if (c == 2) for (int t = 0; t < n; t++) calls.push_back({t, 1, false});
        if (c == 3) { calls = {{0, a, true}}; for (int t = a; t < n; t++) calls.push_back({t, 1, false}); }
        for (const Call& k : calls) {
            const bf16* x = mix.data() + (size_t)k.t0 * W + di;       // the convolution's input columns of the rows
            bf16* cv = conv.data() + (size_t)k.t0 * cd;
            launch({(unsigned)((cd + 255) / 256), 1, 1}, {256, 1, 1}, [&] { nh_conv(x, W, cw.data(), cb.data(), cst.data(), cv, k.len, cd); });
            const bf16* dtr = mix.data() + (size_t)k.t0 * W + di + cd;
            bf16* yy = y.data() + (size_t)k.t0 * di;
            if (k.chunked) {
                const int nch = (k.len + NH_L - 1) / NH_L;
                launch({(unsigned)nch, (unsigned)((nh + 127) / 128), 1}, {128, 1, 1}, [&] { nh_ssd_cum(dtr, W, dtb.data(), alog.data(), dtv.data(), acum.data(), k.len, nh, dt_min); });
                launch({(unsigned)nch, (unsigned)ng, 1}, {256, 1, 1}, [&] { nh_ssd_g(cv, cd, G.data(), k.len, nh, ng); });
                launch({(unsigned)nch, (unsigned)nh, 1}, {256, 1, 1}, [&] { nh_ssd_states(cv, cd, dtv.data(), acum.data(), stb.data(), k.len, nh, ng); });
                launch({(unsigned)((nh * NH_P * NH_N + 255) / 256), 1, 1}, {256, 1, 1}, [&] { nh_ssd_pass(acum.data(), stb.data(), state.data(), k.len, nh, nch); });
                launch({(unsigned)nch, (unsigned)nh, 2}, {256, 1, 1}, [&] { nh_ssd_y(cv, cd, dtv.data(), acum.data(), G.data(), stb.data(), dd.data(), yy, k.len, nh, ng); });
            } else {
                launch({(unsigned)nh, (unsigned)(NH_P / 8), 1}, {256, 1, 1}, [&] { nh_ssm(cv, cd, dtr, W, dtb.data(), alog.data(), dd.data(), state.data(), yy, k.len, nh, ng, dt_min); });
            }
        }
        save(std::string(names[c]) + "_conv.bin", conv);
        save(std::string(names[c]) + "_y.bin", y);
        save(std::string(names[c]) + "_state.bin", state);
        save(std::string(names[c]) + "_cstate.bin", cst);
        // the gated norm, on a copy of y (the gate columns of the mix)
        std::vector<bf16> gn = y;
        launch({(unsigned)n, (unsigned)ng, 1}, {256, 1, 1}, [&] { nh_gated_norm(gn.data(), mix.data(), W, nw.data(), di, ng, eps); });
        save(std::string(names[c]) + "_gn.bin", gn);
        std::printf("mamba case %s done\n", names[c]);
    }
    std::ofstream(OUT + "/cases.txt") << ran << "\n";
}

static void attention() {
    const auto m = nums("attn.txt");
    const int nq = (int)m[0], nk = (int)m[1], n = (int)m[2];
    const int S = 4, qs = (nq + 2 * nk) * 128;
    auto qkv = load<bf16>("qkv.bin");
    std::vector<int> pos(n);
    for (int t = 0; t < n; t++) pos[t] = t;
    std::vector<bf16> kv((size_t)n * nk * 256), out((size_t)n * nq * 128);
    launch({(unsigned)n, (unsigned)nk, 1}, {128, 1, 1}, [&] { nh_kv_write(qkv.data(), qs, pos.data(), kv.data(), nq, nk); });
    save("kv.bin", kv);
    std::vector<float> part((size_t)nq * S * NH_PART);
    const float sl2 = (float)(1.0 / std::sqrt(128.0) * 1.4426950408889634);
    for (int t = 0; t < n; t++) {  // (a decode step a row, as the engine runs them)
        const int g = nq / nk;
        launch({1, (unsigned)nk, (unsigned)S}, {(unsigned)(32 * g), 1, 1}, [&] { nh_attn_parts(qkv.data() + (size_t)t * qs, qs, kv.data(), pos.data() + t, 1, nq, nk, sl2, part.data(), S); });
        launch({1, (unsigned)nq, 1}, {128, 1, 1}, [&] { nh_attn_merge(part.data(), S, out.data() + (size_t)t * nq * 128, nq * 128, nq); });
    }
    save("dec.bin", out);
    std::printf("attention done\n");
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: nh_emu IN_DIR OUT_DIR\n"); return 2; }
    IN = argv[1];
    OUT = argv[2];
    std::string mk = "mkdir -p '" + OUT + "'";
    if (std::system(mk.c_str()) != 0) return 1;
    mamba();
    attention();
    return 0;
}
