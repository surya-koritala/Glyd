// kernels/glm.cu on the host: each CUDA thread a std::thread, a block's __syncthreads() a barrier, a warp's shuffles an exchange behind two barriers of the warp's 32 threads. No GPU, no nvcc: the kernels' arithmetic and
// indexing are checked against tools/glm_ref.py before a pod is paid for. Slow (a block at a time, its threads waiting on each other): small shapes (a few heads, streams of 512 columns).
// What it cannot say: anything of the machine (bank conflicts, occupancy, instruction counts, the 99,072 bytes of the chunked rule's shared memory), the engine's convolution (qwen35.cu's conv_silu) and the library's products.
//
//   c++ -std=c++20 -O1 -pthread -fno-strict-aliasing -o glm_emu tools/glm_emu.cpp          (-fsanitize=address,undefined or -fsanitize=thread on top: indexing past a buffer, a shared-memory access no barrier orders)
//   python3 tools/glm_ref.py kernels gen DIR; ./glm_emu DIR/in DIR/out; python3 tools/glm_ref.py kernels check DIR
//
// The same files and the same sequence as `glyd-engine glmkernels` (src/kcheck_glm.rs); see tools/glm_ref.py for them. GLM_EMU_CASES=chunked,recurrent runs those cases alone (`kernels check` then compares the cases that ran).
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

// ---- the CUDA a kernel of glm.cu uses ----------------------------------------------------------------------------------------------------------------------------------------------------
#define __global__
#define __device__
#define __forceinline__ inline
#define __launch_bounds__(...)
#define __align__(n) __attribute__((aligned(n)))
#define __shared__ static
#define GLM_EMU 1
struct Dim {
    unsigned x = 1, y = 1, z = 1;
};
static Dim blockIdx, blockDim, gridDim;
static thread_local Dim threadIdx;
// A barrier of a mutex and a condition variable (std::barrier's atomics are not known to ThreadSanitizer, which would see a race in every access to shared memory on the two sides of a __syncthreads()).
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
static int pick_up(int lane, int off) { return lane >= off ? lane - off : lane; }
template <class T>
static T __shfl_xor_sync(unsigned, T v, int off) { return shfl_gather(v, threadIdx.x & 31, threadIdx.x >> 5, pick_xor, off); }
template <class T>
static T __shfl_up_sync(unsigned, T v, int off) { return shfl_gather(v, threadIdx.x & 31, threadIdx.x >> 5, pick_up, off); }

struct float2 { float x, y; };
struct float4 { float x, y, z, w; };
struct uint4 { unsigned x, y, z, w; };
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
static float2 bf2(uint32_t u) { return make_float2(__uint_as_float(u << 16), __uint_as_float(u & 0xffff0000u)); }
static float fmaxf_(float a, float b) { return std::fmax(a, b); }
#define fmaxf fmaxf_
#define expf std::exp
#define rsqrtf(x) (1.0f / std::sqrt(x))
#define fmaf std::fma
static unsigned __float_as_uint(float f) { unsigned u; std::memcpy(&u, &f, 4); return u; }
static unsigned atomicAdd(unsigned* a, unsigned v) { return __atomic_fetch_add(a, v, __ATOMIC_RELAXED); }
static int atomicAdd(int* a, int v) { return __atomic_fetch_add(a, v, __ATOMIC_RELAXED); }
static float warp_max(float v) {  // (qwen35.cu's)
    for (int o = 16; o > 0; o >>= 1) v = fmaxf_(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

#include "../kernels/residual.cu"
#include "../kernels/glm.cu"

// ---- launching -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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

// ---- the sequence (src/kcheck_glm.rs's) ------------------------------------------------------------------------------------------------------------------------------------------------------
static std::string IN, OUT;

template <class T>
static std::vector<T> load(const std::string& name) {
    std::ifstream f(IN + "/" + name, std::ios::binary);
    if (!f) { std::fprintf(stderr, "glm_emu: no %s/%s\n", IN.c_str(), name.c_str()); std::exit(1); }
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
static unsigned ceil_div(int a, int b) { return (unsigned)((a + b - 1) / b); }

static void delta() {
    const auto m = nums("kda.txt");
    const int H = (int)m[0], n = (int)m[1], split = (int)m[2];
    const float bound = (float)m[3], eps = (float)m[4];
    const int qd = H * GM_D;
    auto xc = load<bf16>("xc.bin"), fg = load<bf16>("fg.bin"), bb = load<bf16>("bb.bin"), og = load<bf16>("og.bin"), ow = load<bf16>("ow.bin");
    const auto dtb = load<float>("dtb.bin"), alog = load<float>("alog.bin"), state0 = load<float>("state0.bin");
    const char* names[4] = {"chunked", "recurrent", "steps", "mixed"};
    const char* only = std::getenv("GLM_EMU_CASES");  // (e.g. chunked,recurrent: the cases of many single-token launches take long on the host)
    std::string ran;
    for (int c = 0; c < 4; c++) {
        if (only && !std::strstr(only, names[c])) continue;
        ran += std::string(ran.empty() ? "" : " ") + names[c];
        std::vector<float> state = state0;
        std::vector<bf16> core((size_t)n * qd), att((size_t)n * qd);
        // calls: (first token, tokens, in chunks or token by token)
        struct Call { int t0, len; bool chunked; };
        std::vector<Call> calls;
        const int a = split > 0 && split < n ? split : n;
        if (c == 0) { calls = {{0, a, true}}; if (a < n) calls.push_back({a, n - a, true}); }
        if (c == 1) { calls = {{0, a, false}}; if (a < n) calls.push_back({a, n - a, false}); }
        if (c == 2) for (int t = 0; t < n; t++) calls.push_back({t, 1, false});
        if (c == 3) { calls = {{0, a, true}}; for (int t = a; t < n; t++) calls.push_back({t, 1, false}); }
        for (const Call& k : calls) {
            const bf16* x = xc.data() + (size_t)k.t0 * 3 * qd;
            const bf16* f = fg.data() + (size_t)k.t0 * qd;
            const bf16* b = bb.data() + (size_t)k.t0 * H;
            bf16* o = core.data() + (size_t)k.t0 * qd;
            if (k.chunked)
                launch({(unsigned)H, 2, 1}, {256, 1, 1}, [&] { glm_kda_chunk(x, f, dtb.data(), alog.data(), b, state.data(), o, k.len, H, bound); });
            else
                launch({(unsigned)H, 1, 1}, {128, 1, 1}, [&] { glm_kda_rec(x, f, dtb.data(), alog.data(), b, state.data(), o, k.len, H, bound); });
        }
        launch({(unsigned)n, (unsigned)H, 1}, {128, 1, 1}, [&] { glm_kda_out(core.data(), og.data(), ow.data(), att.data(), H, eps); });
        save(std::string(names[c]) + "_core.bin", core);
        save(std::string(names[c]) + "_state.bin", state);
        save(std::string(names[c]) + "_att.bin", att);
        std::printf("delta rule case %s done\n", names[c]);
    }
    std::ofstream(OUT + "/cases.txt") << ran << "\n";
}

static void streams() {
    const auto m = nums("hc.txt");
    const int D = (int)m[0], n = (int)m[1], iters = (int)m[4];
    const float eps = (float)m[2], hc_eps = (float)m[3];
    const auto hx = load<bf16>("hx.bin"), hfn = load<bf16>("hfn.bin"), hnw = load<bf16>("hnw.bin"), ho = load<bf16>("ho.bin");
    const auto hbase = load<float>("hbase.bin"), hscale = load<float>("hscale.bin");
    std::vector<bf16> h((size_t)n * D), post = hx, head((size_t)n * D), spread((size_t)n * HC_N * D);
    std::vector<float> hcp((size_t)n * HC_NP);
    // (kernels/residual.cu's streams in transformers' bf16 rounding, as GLM runs them)
    launch({(unsigned)n, 1, 1}, {512, 1, 1}, [&] { hc_mix(hx.data(), hfn.data(), hbase.data(), hscale.data(), hnw.data(), h.data(), hcp.data(), D, eps, hc_eps, iters, 0); });
    launch({(unsigned)n, ceil_div(D, 256), 1}, {256, 1, 1}, [&] { hc_post(post.data(), ho.data(), hcp.data(), D, 1); });
    launch({(unsigned)n, 1, 1}, {512, 1, 1}, [&] { hc_head(hx.data(), nullptr, 0, nullptr, nullptr, hnw.data(), head.data(), D, eps, 0.f, 0, 0); });
    launch({(unsigned)n, ceil_div(D, 256), 1}, {256, 1, 1}, [&] { hc_streams(ho.data(), spread.data(), D); });
    {   // few rows: the split projection and its rest (what a decode step runs)
        const int m8 = std::min(n, 8), KS = HC_N * D / HC_CPB_BF16;
        std::vector<float> part((size_t)m8 * KS * (HC_NP + 1)), hcp_s((size_t)m8 * HC_NP);
        std::vector<bf16> h_s((size_t)m8 * D);
        launch({(unsigned)m8, (unsigned)KS, 1}, {128, 1, 1}, [&] { hc_proj_bf16(hx.data(), hfn.data(), part.data(), D, HC_NP); });
        launch({(unsigned)m8, 1, 1}, {512, 1, 1}, [&] { hc_fin(hx.data(), part.data(), KS, hbase.data(), hscale.data(), hnw.data(), h_s.data(), hcp_s.data(), D, eps, hc_eps, iters, 0, 0); });
        save("hc_params_s.bin", hcp_s);
        save("hc_h_s.bin", h_s);
    }
    save("hc_params.bin", hcp);
    save("hc_h.bin", h);
    save("hc_post.bin", post);
    save("hc_head.bin", head);
    save("hc_streams.bin", spread);
    std::printf("hyper-connection done\n");
}

// The latent attention's kernels (src/kcheck_glm.rs's `mla` sequence): the rows of a sequence in calls, each through the norms into the caches, the pool keys, the absorbed queries, the indexer's scores and choice, the attention and
// the values; the caches and every row's results written for tools/glm_ref.py `kernels check`.
static void mla() {
    const auto m = nums("mla.txt");
    const int NH = (int)m[0], nope = (int)m[1], vd = (int)m[2], KR = (int)m[3], IH = (int)m[4], ID = (int)m[5], kp = (int)m[6], topk = (int)m[7], n = (int)m[8], nc = (int)m[9];
    std::vector<int> calls;
    for (int i = 0; i < nc; i++) calls.push_back((int)m[10 + i]);
    const float eps = (float)m[10 + nc];
    const int tail = m.size() > (size_t)(11 + nc) ? (int)m[11 + nc] : 1;
    const int ksel = topk / kp, Tmax = topk + kp - 1, Pmax = n / kp + 1;
    const auto kvraw = load<bf16>("mla_kvraw.bin"), kvw = load<bf16>("mla_kvw.bin"), ikraw = load<bf16>("mla_ikraw.bin"), knw = load<bf16>("mla_knw.bin"), knb = load<bf16>("mla_knb.bin"), gsraw = load<bf16>("mla_gsraw.bin"),
               ape = load<bf16>("mla_ape.bin"), iq = load<bf16>("mla_iq.bin"), wts = load<bf16>("mla_wts.bin"), q = load<bf16>("mla_q.bin"), wkv = load<bf16>("mla_wkv.bin");
    std::vector<int> pos(n);
    for (int i = 0; i < n; i++) pos[i] = i;
    std::vector<bf16> lat((size_t)n * KR), ikc((size_t)n * ID), gsc((size_t)n * ID), pkc((size_t)Pmax * ID), ov((size_t)n * NH * vd);
    std::vector<float> qa((size_t)n * NH * KR), scores((size_t)n * Pmax), olat((size_t)n * NH * KR), part((size_t)8 * NH * 8 * (KR + 2));
    std::vector<int> sel((size_t)n * Tmax), cnt(n);
    const float scale = std::pow((float)nope, -0.5f), iscale = std::pow((float)ID, -0.5f), wscale = std::pow((float)IH, -0.5f);
    int a = 0;
    for (const int len : calls) {
        const int* ps = pos.data() + a;
        launch({(unsigned)len, 1, 1}, {256, 1, 1}, [&] { glm_rms2(kvraw.data() + (size_t)a * KR, KR, kvw.data(), lat.data(), KR, ps, KR, eps); });
        launch({(unsigned)len, 1, 1}, {128, 1, 1}, [&] { glm_layernorm(ikraw.data() + (size_t)a * ID, ID, knw.data(), knb.data(), ikc.data(), ID, ps, ID, 1e-6f); });
        launch({(unsigned)len, 1, 1}, {128, 1, 1}, [&] { glm_scatter(gsraw.data() + (size_t)a * ID, ID, gsc.data(), ID, ps, ID); });
        launch({ceil_div(len, kp) + 1, 1, 1}, {128, 1, 1}, [&] { glm_kp_keys(ikc.data(), gsc.data(), ape.data(), pkc.data(), ps, len, ID, kp); });
        launch({(unsigned)NH, ceil_div(len, GM_RB), 1}, {256, 1, 1}, [&] { glm_absorb(q.data() + (size_t)a * NH * nope, NH * nope, wkv.data(), qa.data() + (size_t)a * NH * KR, len, NH, nope, vd, KR); });
        launch({ceil_div(Pmax, GM_IDX_W), (unsigned)len, 1}, {256, 1, 1}, [&] {
            glm_idx_score(iq.data() + (size_t)a * IH * ID, IH * ID, wts.data() + (size_t)a * IH, IH, pkc.data(), ps, scores.data(), Pmax, IH, ID, kp, ksel, iscale, wscale);
        });
        launch({(unsigned)len, 1, 1}, {1024, 1, 1}, [&] { glm_select(scores.data(), Pmax, ps, sel.data() + (size_t)a * Tmax, cnt.data() + a, Tmax, kp, ksel, tail); });
        const int splits = len <= 8 ? 8 : 1;  // (the engine's: a few rows split their tokens over blocks)
        launch({(unsigned)NH, (unsigned)len, (unsigned)splits}, {256, 1, 1}, [&] {
            glm_mla_attn(qa.data() + (size_t)a * NH * KR, lat.data(), sel.data() + (size_t)a * Tmax, cnt.data() + a, olat.data() + (size_t)a * NH * KR, part.data(), Tmax, NH, KR, scale, splits);
        });
        if (splits > 1) launch({(unsigned)NH, (unsigned)len, 1}, {256, 1, 1}, [&] { glm_mla_merge(part.data(), olat.data() + (size_t)a * NH * KR, NH, KR, splits); });
        launch({(unsigned)NH, ceil_div(len, GM_VB), 1}, {256, 1, 1}, [&] { glm_expand_v(olat.data() + (size_t)a * NH * KR, wkv.data(), ov.data() + (size_t)a * NH * vd, len, NH, nope, vd, KR); });
        a += len;
    }
    save("mla_lat.bin", lat);
    save("mla_ikc.bin", ikc);
    save("mla_gsc.bin", gsc);
    save("mla_pk.bin", pkc);
    save("mla_qa.bin", qa);
    save("mla_sel.bin", sel);
    save("mla_cnt.bin", cnt);
    save("mla_olat.bin", olat);
    save("mla_ov.bin", ov);
    std::printf("latent attention done: %d rows in %d calls\n", n, nc);
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: glm_emu IN_DIR OUT_DIR\n"); return 2; }
    IN = argv[1];
    OUT = argv[2];
    std::string mk = "mkdir -p '" + OUT + "'";
    if (std::system(mk.c_str()) != 0) return 1;
    if (!std::getenv("GLM_EMU_NO_KDA")) {  // (GLM_EMU_NO_KDA=1: the latent attention's kernels alone)
        streams();
        delta();
    }
    if (std::ifstream(IN + "/mla.txt")) mla();
    return 0;
}
