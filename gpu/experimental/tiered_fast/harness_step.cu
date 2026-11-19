// The step kernels of glyd_gpu.cu on packs from files, timed: the library's own product for up to 64 tokens in both layouts and the variants of
// the tiered layout's (the 8-warp kernel before the step kernel), each against an fp32 product and the first variant's bits, in
// interleaved rounds with the SM clock measured around every timing (an L4 at its power cap changes clock by 10% as it heats), as ns and as SM
// clocks a step (1024 weights) an SM. COLD=1 rotates the weights through copies past the L2. Built twice, to compare with a library before the
// tiered layout's step kernel (-DOLDLIB: its packs, PREFIX.v1.*) or in this one (PREFIX.t.*).
//   nvcc -O3 -lineinfo -std=c++20 --expt-relaxed-constexpr -gencode arch=compute_89,code=sm_89 -I.. harness_step.cu -o harness_step
//   ROUNDS=5 ./harness_step PREFIX [M=1] [ITERS=150] [variants: "n,s,s2,s4,o," or all]
// variants: n (12-bit lib), s (tiered lib), o (tiered, the 8-warp kernel), rn rs (the routes of 17-64 tokens), a trailing comma each;
// FLAGEVERY=1,5,64 adds the tiered lib on the pack with the flag set on every N-th step. PREFIX.{dims,w} the matrix [O, K] as bf16 bits; PREFIX.t.{data,blocks,bb,tiers} the tiered pack,
// PREFIX.nib.{data,exc,eb,hb} the 12-bit one (-DOLDLIB: PREFIX.data, PREFIX.v1.{blocks,bb,tiers} the tiered one).
#include "glyd_gpu.cu"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <functional>
#include <algorithm>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

__global__ void ref_kernel(const __nv_bfloat16* W, const __nv_bfloat16* X, __nv_bfloat16* Y, int64_t O, int64_t K, int64_t M) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= O * M) return;
    int64_t m = i / O, o = i % O;
    float a = 0.f;
    for (int64_t k = 0; k < K; k++) a += __bfloat162float(X[m * K + k]) * __bfloat162float(W[o * K + k]);
    Y[m * O + o] = __float2bfloat16(a);
}

__global__ void clock_probe(long long* out) {  // the SM clock: cycles in 200 us of the global timer
    unsigned long long t0, t1;
    long long c0 = clock64();
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1)); } while (t1 - t0 < 200000ull);
    out[0] = clock64() - c0;
    out[1] = (long long)(t1 - t0);
}
static double sm_ghz() {
    static long long* d = nullptr;
    if (!d) CK(cudaMalloc(&d, 16));
    clock_probe<<<1, 1>>>(d);
    long long h[2];
    CK(cudaMemcpy(h, d, 16, cudaMemcpyDeviceToHost));
    return (double)h[0] / (double)h[1];
}

static std::vector<uint8_t> slurp(const std::string& path, bool must = true) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        if (must) { fprintf(stderr, "cannot read %s\n", path.c_str()); exit(1); }
        return {};
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> v(n);
    if (fread(v.data(), 1, n, f) != (size_t)n) exit(1);
    fclose(f);
    return v;
}

// a host buffer on the GPU, `copies` times (each its own allocation)
static std::vector<void*> to_gpu(const std::vector<uint8_t>& v, int copies) {
    std::vector<void*> out;
    for (int i = 0; i < copies; i++) {
        void* p;
        CK(cudaMalloc(&p, v.size() + 256));
        CK(cudaMemset(p, 0, v.size() + 256));
        CK(cudaMemcpy(p, v.data(), v.size(), cudaMemcpyHostToDevice));
        out.push_back(p);
    }
    return out;
}

struct Pack {
    std::vector<uint8_t> a, b, c;  // tiered: data, blocks, block_base; 12-bit: data, exc, exc_base
    uint32_t t[3];
    uint32_t hb4 = 0;
    std::vector<void*> da, db, dc;
    size_t bytes() const { return a.size() + b.size() + c.size(); }
};

static bool load_tiered(const std::string& prefix, int copies, Pack& p) {
#ifdef OLDLIB
    const std::string tag = ".v1", dtag = "";  // (phase 1's packs: the data array alone is PREFIX.data)
#else
    const std::string tag = ".t", dtag = ".t";
#endif
    p.a = slurp(prefix + dtag + ".data", false);
    if (p.a.empty()) return false;
    p.b = slurp(prefix + tag + ".blocks");
    p.c = slurp(prefix + tag + ".bb");
    auto t = slurp(prefix + tag + ".tiers");
    t.push_back(0);
    sscanf((const char*)t.data(), "%u %u %u", &p.t[0], &p.t[1], &p.t[2]);
    p.da = to_gpu(p.a, copies);
    p.db = to_gpu(p.b, copies);
    p.dc = to_gpu(p.c, copies);
    return true;
}

static bool load_nib(const std::string& prefix, int copies, Pack& p) {
    p.a = slurp(prefix + ".nib.data", false);
    if (p.a.empty()) return false;
    p.b = slurp(prefix + ".nib.exc");
    p.c = slurp(prefix + ".nib.eb");
    auto t = slurp(prefix + ".nib.hb");
    t.push_back(0);
    p.hb4 = (uint32_t)atoi((const char*)t.data()) * 0x01010101u;
    p.da = to_gpu(p.a, copies);
    p.db = to_gpu(p.b, copies);
    p.dc = to_gpu(p.c, copies);
    return true;
}

struct Var {
    std::string name;
    std::function<int(int, const uint16_t*, uint16_t*, void*, size_t, int*, cudaStream_t, size_t*)> launch;
    size_t need = 0;
    void* ws = nullptr;
    int* done = nullptr;
    int copies = 1;
    std::vector<double> clk, ns, ghz;
    std::vector<uint16_t> out;
};

static void prepare(Var& v, int64_t O) {
    CK((cudaError_t)v.launch(0, nullptr, nullptr, nullptr, 0, nullptr, 0, &v.need));
    CK(cudaMalloc(&v.ws, v.need + 16));
    CK(cudaMalloc(&v.done, 4 * (O / 64 + 1024)));
    CK(cudaMemset(v.done, 0, 4 * (O / 64 + 1024)));
}

static void time_once(Var& v, int64_t O, int64_t K, const uint16_t* x, uint16_t* y, int iters) {
    for (int i = 0; i < 3; i++) CK((cudaError_t)v.launch(i % v.copies, x, y, v.ws, v.need, v.done, 0, nullptr));
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    double g0 = sm_ghz();
    cudaEventRecord(a, 0);
    for (int i = 0; i < iters; i++) v.launch(i % v.copies, x, y, v.ws, v.need, v.done, 0, nullptr);
    cudaEventRecord(b, 0);
    cudaEventSynchronize(b);
    double g1 = sm_ghz();
    float ms;
    cudaEventElapsedTime(&ms, a, b);
    double us = ms * 1000.0 / iters, ghz = 0.5 * (g0 + g1);
    int64_t steps = O * K / 1024;
    double ns_step_sm = us * 1000.0 / steps * sm_count(0);
    v.ns.push_back(ns_step_sm);
    v.ghz.push_back(ghz);
    v.clk.push_back(ns_step_sm * ghz);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main(int argc, char** argv) {
    if (argc < 2) return 1;
    std::string prefix = argv[1];
    int64_t M = argc > 2 ? atoi(argv[2]) : 1;
    int iters = argc > 3 ? atoi(argv[3]) : 150;
    std::string which = argc > 4 ? argv[4] : "all";
    auto dims = slurp(prefix + ".dims");
    dims.push_back(0);
    long O, K;
    sscanf((const char*)dims.data(), "%ld %ld", &O, &K);
    auto w = slurp(prefix + ".w");
    int copies = 1;
    if (getenv("COLD")) copies = std::max<int>(1, (int)((256ull << 20) / (O * K * 3 / 2)));
    if (getenv("COPIES")) copies = atoi(getenv("COPIES"));
    auto dw = to_gpu(w, 1);
    uint16_t* d_w = (uint16_t*)dw[0];
    uint16_t *d_x, *d_y, *d_ref;
    CK(cudaMalloc(&d_x, 2 * M * K + 64));
    CK(cudaMalloc(&d_y, 2 * M * O + 64));
    CK(cudaMalloc(&d_ref, 2 * M * O + 64));
    std::vector<uint16_t> hx(M * K);
    srand(7);
    for (auto& v : hx) {  // bf16 of a random number in [-1, 1]
        float f = (rand() / (float)RAND_MAX - 0.5f) * 2.f;
        uint32_t u;
        memcpy(&u, &f, 4);
        v = (uint16_t)(u >> 16);
    }
    CK(cudaMemcpy(d_x, hx.data(), 2 * M * K, cudaMemcpyHostToDevice));
    ref_kernel<<<(O * M + 255) / 256, 256>>>(bf(d_w), bf(d_x), bf(d_ref), O, K, M);
    CK(cudaDeviceSynchronize());
    std::vector<uint16_t> ref(O * M);
    CK(cudaMemcpy(ref.data(), d_ref, 2 * O * M, cudaMemcpyDeviceToHost));
    printf("%s: %ldx%ld, M=%ld, %d SMs, %d weight copies\n", prefix.c_str(), O, K, (long)M, (int)sm_count(0), copies);

    Pack pt, pn;
    bool have_t = load_tiered(prefix, copies, pt), have_nib = load_nib(prefix, copies, pn);
    if (have_t) printf("tiered %.4f bits a weight\n", 8.0 * pt.bytes() / (double)(O * K));
    if (have_nib) printf("12-bit %.4f bits a weight\n", 8.0 * pn.bytes() / (double)(O * K));
    auto FT = [&](int c) { return Tiered{(const uint8_t*)pt.da[c], (const uint8_t*)pt.db[c], (const int32_t*)pt.dc[c], {pt.t[0], pt.t[1], pt.t[2]}}; };
    auto FN = [&](int c) { return Nib{(const uint8_t*)pn.da[c], (const uint32_t*)pn.db[c], (const int32_t*)pn.dc[c], pn.hb4}; };

    std::vector<Var> vars;
    using L = std::function<int(int, const uint16_t*, uint16_t*, void*, size_t, int*, cudaStream_t, size_t*)>;
    auto add = [&](const char* name, L launch) { vars.emplace_back(); vars.back().name = name; vars.back().launch = launch; vars.back().copies = copies; };
    auto has = [&](const char* v) { return which == "all" || ("," + which).find(std::string(",") + v + ",") != std::string::npos; };
    if (has("s") && have_t) add("tiered lib", [&](int c, const uint16_t* x, uint16_t* y, void* ws, size_t wsb, int* done, cudaStream_t cs, size_t* need) { return mma_gemm_run(FT(c), O, K, x, M, nullptr, y, ws, wsb, done, cs, need); });
    if (has("n") && have_nib) add("12-bit lib", [&](int c, const uint16_t* x, uint16_t* y, void* ws, size_t wsb, int* done, cudaStream_t cs, size_t* need) { return mma_gemm_run(FN(c), O, K, x, M, nullptr, y, ws, wsb, done, cs, need); });
#ifndef OLDLIB
    if (M <= 16 && have_t && has("o")) add("tiered, 8 warps", [&](int c, const uint16_t* x, uint16_t* y, void* ws, size_t wsb, int* done, cudaStream_t cs, size_t* need) { return mma_gemm_mt<Tiered, 1, false, false>(FT(c), O, K, x, M, nullptr, y, ws, wsb, done, cs, need); });
#endif
    if (M > 16 && M <= 128) {  // the shipped kernels of each layout at this many tokens (the library's own routes: step, mid, wg)
        if (has("rs") && have_t) add("tiered route", [&](int c, const uint16_t* x, uint16_t* y, void* ws, size_t wsb, int* done, cudaStream_t cs, size_t* need) { return mma_linear_run(FT(c), O, K, x, M, nullptr, y, -1, ws, wsb, done, cs, need); });
        if (has("rn") && have_nib) add("12-bit route", [&](int c, const uint16_t* x, uint16_t* y, void* ws, size_t wsb, int* done, cudaStream_t cs, size_t* need) { return mma_linear_run(FN(c), O, K, x, M, nullptr, y, -1, ws, wsb, done, cs, need); });
    }
#ifndef OLDLIB
    // FLAGEVERY=1,5,64: the tiered pack with the flag set on every N-th step (1: every step), which the general decode takes: the product is the first's
    // bits, whatever the share of them, and the time what such steps cost
    std::vector<Pack> flagged;
    if (const char* fe = getenv("FLAGEVERY")) {
        if (have_t && M <= 16) {
            std::vector<int> every;
            for (const char* q = fe; *q;) {
                every.push_back(atoi(q));
                while (*q && *q != ',') q++;
                if (*q == ',') q++;
            }
            flagged.reserve(every.size());
            for (int e : every) {
                flagged.push_back(pt);
                Pack& pf = flagged.back();
                int32_t* b = (int32_t*)pf.c.data();
                size_t n = pf.c.size() / 4 - 1;
                for (size_t s = 0; s < n; s += e) b[s] = (int32_t)((uint32_t)b[s] | 0x80000000u);
                pf.dc = to_gpu(pf.c, copies);
                std::string nm = "flag every " + std::to_string(e);
                Pack* pp = &pf;
                add(nm.c_str(), [&, pp](int c, const uint16_t* x, uint16_t* y, void* ws, size_t wsb, int* done, cudaStream_t cs, size_t* need) {
                    Tiered f{(const uint8_t*)pp->da[c], (const uint8_t*)pp->db[c], (const int32_t*)pp->dc[c], {pp->t[0], pp->t[1], pp->t[2]}};
                    return mma_gemm_run(f, O, K, x, M, nullptr, y, ws, wsb, done, cs, need); });
            }
        }
    }
#endif
    for (auto& v : vars) prepare(v, O);
    int rounds = getenv("ROUNDS") ? atoi(getenv("ROUNDS")) : 5;
    sm_ghz();
    for (int r = 0; r < rounds; r++)
        for (size_t i = 0; i < vars.size(); i++) time_once(vars[(i + r * 7) % vars.size()], O, K, d_x, d_y, iters);
    std::vector<uint16_t> out1;
    for (auto& v : vars) {
        CK(cudaMemset(d_y, 0, 2 * O * M));
        CK((cudaError_t)v.launch(0, d_x, d_y, v.ws, v.need, v.done, 0, nullptr));
        CK(cudaDeviceSynchronize());
        v.out.resize(O * M);
        CK(cudaMemcpy(v.out.data(), d_y, 2 * O * M, cudaMemcpyDeviceToHost));
        if (&v == &vars[0]) out1 = v.out;
    }
    printf("%-16s %9s %9s %7s   (median of %d rounds, interleaved; clk = ns a step an SM x the SM clock measured around each timing)\n", "variant", "ns/step", "clk/step", "GHz", rounds);
    for (auto& v : vars) {
        double worst = 0, big = 0;
        for (long i = 0; i < O * M; i++) {
            float a, b;
            uint32_t ua = (uint32_t)v.out[i] << 16, ub = (uint32_t)ref[i] << 16;
            memcpy(&a, &ua, 4);
            memcpy(&b, &ub, 4);
            worst = std::max(worst, (double)fabsf(a - b));
            big = std::max(big, (double)fabsf(b));
        }
        printf("%-16s %9.1f %9.1f %7.3f   %s%s\n", v.name.c_str(), median(v.ns), median(v.clk), median(v.ghz), worst / big < 1e-2 ? "ok" : "TOO FAR", v.out == out1 ? ", = the first's bits" : ", differs from the first's");
    }
    return 0;
}
