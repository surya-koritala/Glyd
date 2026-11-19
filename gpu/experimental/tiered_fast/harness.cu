// A scratch harness for the tiered layout's v2 decode (see tiered-fast-design.md). glyd_gpu.cu's own step kernels run the
// v1 tiered layout (Tiered) and the 12-bit layout (Nib); a Tiered2 policy (v2 blocks: tier 3 a dense digit of 4 symbols,
// ranks from 10 on in a list of 3-byte entries) and the option variants of v2.cuh run packs written by pack_v2.py:
// bit-for-bit unpack checks, the products of every variant against a reference and each other, and the time of each, in
// interleaved rounds with the SM clock measured around every timing (an L4 at its power cap changes clock by 10% as it heats),
// as ns and as SM clocks a step (1024 weights) an SM. COLD=1 rotates the weights through copies past the L2.
//   nvcc -O3 -lineinfo -std=c++20 --expt-relaxed-constexpr -gencode arch=compute_89,code=sm_89 -I. harness.cu -o harness
//   ROUNDS=5 ./harness PREFIX [M=1] [ITERS=150] [variants: "l,n,o0u1,..," or all]
// variants: v1 (always), n (12-bit lib kernel), l (v2 lib kernel), oOPTuUNR (v2.cuh's step kernel), a trailing comma each
#include "glyd_gpu.cu"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <functional>
#include <algorithm>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

// ---------------------------------------------------------------------------------------------------------------------
// v2's decode. A step's block: [tier-3 digits of its tier-2 escapes, a byte past 2 T2 bits][the list's entries, 3 bytes
// each: place among the step's tier-1 escapes (bits 0-9), exponent byte (bits 10-17)][tier-2 digits of its tier-1 escapes:
// words of 16, the c-th word back from the end].

__device__ __forceinline__ uint32_t third(uint32_t v) { return (v * 0xAAABu) >> 17; }  // v / 3, v < 2^15

__device__ __forceinline__ void exponents2(const Step& st, Tiers ts, const uint8_t* __restrict__ blocks, int lane, uint32_t* s2, const uint32_t* tab, uint32_t ew[8]) {
    uint32_t n1 = __popc(escapes(st.pd[0])) + __popc(escapes(st.pd[1]));
    uint32_t T1 = __reduce_add_sync(FULL, n1), words = (T1 + 15) >> 4;
    uint32_t b1, T2;
    uint32_t S3 = 0x01010101u * (ts.s3 >> 24);  // the 4th symbol of tier 3 in every byte: digit 3's escape in the table's selectors
    uint32_t o = 8 * (uint32_t)(st.blk & 3);
    if (T1 <= 512) {
        uint32_t up = __shfl_up_sync(FULL, st.tail, 1);
        uint32_t da = (uint32_t)lane < words ? __funnelshift_r(st.tail, lane ? up : st.tail1, 8 * (uint32_t)(st.end & 3)) : 0u;
        uint32_t na = __popc(escapes(da)), tot;
        uint32_t pre = scan(n1 | na << 11, lane, tot);
        b1 = pre & 0x7FF;
        T2 = tot >> 11;
        uint32_t pa = o + 2 * ((pre >> 11) & 0x3FF);
        bool near = o + 2 * T2 + 32 < 992;
        uint32_t d3 = near ? win_bits(st.win, pa) : mem_bits(blocks, st.blk, pa);
        uint32_t g3 = (__reduce_max_sync(FULL, na) + 3) >> 2;
        uint32_t y[4] = {0u, 0u, 0u, 0u}, q4[4];
#pragma unroll
        for (int q = 0; q < 4; q++)
            if (q < (int)g3) y[q] = __byte_perm(ts.s3, S3, tab[(d3 >> (8 * q)) & 0xFF]);
#pragma unroll
        for (int q = 0; q < 4; q++) q4[q] = group(da >> (8 * q), ts.s2, y, tab);
        ((uint4*)s2)[lane] = make_uint4(q4[0], q4[1], q4[2], q4[3]);
    } else {
        // More than 32 words: chunks of 32, each its words' tier-3 digits after the chunks before it.
        uint32_t tot1;
        b1 = scan(n1, lane, tot1);
        uint32_t carry = 0;
        for (uint32_t ch = 0; ch * 32 < words; ch++) {
            uint32_t c = ch * 32 + lane;
            uint32_t da = c < words ? load4(blocks + st.end - 4 * (c + 1)) : 0u;
            uint32_t na = __popc(escapes(da)), tot;
            uint32_t pre = scan(na, lane, tot);
            uint32_t pa = o + 2 * (carry + pre);
            uint32_t d3 = mem_bits(blocks, st.blk, pa);  // (the window holds 128 bytes: tier-3 digits can run past it)
            uint32_t g3 = (__reduce_max_sync(FULL, na) + 3) >> 2;
            uint32_t y[4] = {0u, 0u, 0u, 0u}, q4[4];
#pragma unroll
            for (int q = 0; q < 4; q++)
                if (q < (int)g3) y[q] = __byte_perm(ts.s3, S3, tab[(d3 >> (8 * q)) & 0xFF]);
#pragma unroll
            for (int q = 0; q < 4; q++) q4[q] = group(da >> (8 * q), ts.s2, y, tab);
            ((uint4*)s2)[c] = make_uint4(q4[0], q4[1], q4[2], q4[3]);
            carry += tot;
        }
        T2 = carry;
    }
    // The list: entry i (lane i, i + 32, ...) overwrites its tier-1 escape's byte.
    uint32_t r0 = (2 * T2 + 7) >> 3, nl = third((uint32_t)(st.end - st.blk) - r0 - 4 * words);
    __syncwarp();
    for (uint32_t i = lane; i < nl; i += 32) {
        uint32_t x = load4(blocks + st.blk + r0 + 3 * i);
        ((uint8_t*)s2)[x & 0x3FF] = (uint8_t)(x >> 10);
    }
    __syncwarp();
    // Tier 1: each group's escapes' bytes from the scratch, this lane's from byte b1.
    uint32_t c = b1;
#pragma unroll
    for (int q = 0; q < 8; q++) {
        uint32_t e = tab[(st.pd[q >> 2] >> (8 * (q & 3))) & 0xFF];
        ew[q] = __byte_perm(ts.s1, __funnelshift_r(s2[c >> 2], s2[(c >> 2) + 1], c << 3), e);
        c += e >> 16;
    }
    __syncwarp();
}

struct Tiered2 {
    const uint8_t* data;
    const uint8_t* blocks;
    const int32_t* block_base;
    Tiers ts;
    typedef Step St;
    static constexpr bool kTable = true;
    __device__ __forceinline__ void load(St& st, int64_t step, int lane) const { load_step(st, data, blocks, block_base, step, lane); }
    __device__ __forceinline__ void decode(const St& st, int lane, uint32_t* s2, const uint32_t* tab, uint32_t R[16]) const {
        uint32_t ew[8];
        exponents2(st, ts, blocks, lane, s2, tab, ew);
        pairs(st.sw, ew, R);
    }
};

#include "v2.cuh"

// ---------------------------------------------------------------------------------------------------------------------

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
        CK(cudaMalloc(&p, v.size() + 64));
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

static Pack load_tiered(const std::string& prefix, int version, const std::vector<uint8_t>& data, int copies) {
    Pack p;
    p.a = data;
    p.b = slurp(prefix + ".v" + std::to_string(version) + ".blocks");
    p.c = slurp(prefix + ".v" + std::to_string(version) + ".bb");
    auto t = slurp(prefix + ".v" + std::to_string(version) + ".tiers");
    t.push_back(0);
    sscanf((const char*)t.data(), "%u %u %u", &p.t[0], &p.t[1], &p.t[2]);
    p.da = to_gpu(p.a, copies);
    p.db = to_gpu(p.b, copies);
    p.dc = to_gpu(p.c, copies);
    return p;
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
    CK(cudaMalloc(&v.done, 4 * (O / 64 + 1)));
    CK(cudaMemset(v.done, 0, 4 * (O / 64 + 1)));
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
    auto data = slurp(prefix + ".data");
    // copies of the weights: 1 (hot in L2), or with COLD=1 enough for 256 MB
    int copies = 1;
    if (getenv("COLD")) copies = std::max<int>(1, (int)((256ull << 20) / (O * K * 3 / 2)));
    if (getenv("COPIES")) copies = atoi(getenv("COPIES"));
    auto dw = to_gpu(w, 1);
    uint16_t* d_w = (uint16_t*)dw[0];
    uint16_t *d_x, *d_y, *d_ref, *d_un;
    CK(cudaMalloc(&d_x, 2 * M * K));
    CK(cudaMalloc(&d_y, 2 * M * O));
    CK(cudaMalloc(&d_ref, 2 * M * O));
    CK(cudaMalloc(&d_un, 2 * O * K));
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

    Pack p1 = load_tiered(prefix, 1, data, copies), p2 = load_tiered(prefix, 2, data, copies), pn;
    Pack p2f = p2;  // T2 with the cold flags in block_base (PREFIX.v2.bbf, from pack_v2.py): the same data and blocks
    p2f.c = slurp(prefix + ".v2.bbf", false);
    bool have_flags = !p2f.c.empty();
    if (have_flags && getenv("FLAGEVERY")) {  // a timing and exactness experiment: every N-th step flagged cold (the general decode takes it; the product must not change)
        int every = atoi(getenv("FLAGEVERY"));
        int32_t* b = (int32_t*)p2f.c.data();
        size_t n = p2f.c.size() / 4 - 1;
        for (size_t s = 0; s < n; s++)
            if (every > 0 && s % every == 0) b[s] = (int32_t)((uint32_t)b[s] | 0x80000000u);
        printf("FLAGEVERY=%d: %zu of %zu steps flagged cold\n", every, (n + every - 1) / every, n);
    }
    if (have_flags) p2f.dc = to_gpu(p2f.c, copies);
    bool have_nib = load_nib(prefix, copies, pn);
    auto F1 = [&](int c) { return Tiered{(const uint8_t*)p1.da[c], (const uint8_t*)p1.db[c], (const int32_t*)p1.dc[c], {p1.t[0], p1.t[1], p1.t[2]}}; };
    auto F2 = [&](int c) { return Tiered2{(const uint8_t*)p2.da[c], (const uint8_t*)p2.db[c], (const int32_t*)p2.dc[c], {p2.t[0], p2.t[1], p2.t[2]}}; };
    auto FN = [&](int c) { return Nib{(const uint8_t*)pn.da[c], (const uint32_t*)pn.db[c], (const int32_t*)pn.dc[c], pn.hb4}; };
    auto unpack_check = [&](const char* name, size_t bytes, auto fmt) {
        CK(cudaMemset(d_un, 0, 2 * O * K));
        CK((cudaError_t)mma_unpack_run(fmt, K, 0, O, d_un, 0, 0));
        CK(cudaDeviceSynchronize());
        std::vector<uint16_t> un(O * K);
        CK(cudaMemcpy(un.data(), d_un, 2 * O * K, cudaMemcpyDeviceToHost));
        long bad = 0;
        const uint16_t* wp = (const uint16_t*)w.data();
        for (long i = 0; i < O * K; i++) bad += un[i] != wp[i];
        printf("%-8s %.4f bits a weight, unpack %s (%ld differ)\n", name, 8.0 * bytes / (double)(O * K), bad ? "DIFFERS" : "ok", bad);
    };
    unpack_check("v1", p1.bytes(), F1(0));
    unpack_check("v2", p2.bytes(), F2(0));
    if (have_nib) unpack_check("12-bit", pn.bytes(), FN(0));

    std::vector<Var> vars;
    auto add = [&](const char* name, auto launch) { vars.emplace_back(); vars.back().name = name; vars.back().launch = launch; vars.back().copies = copies; };
    auto has = [&](const char* v) { return which.find(v) != std::string::npos; };
    if (M > 64) {  // a prompt's product: the library's tiled kernel (producers decode, consumers mma.sync), blocks of 128 tokens by two row blocks
        add("v1 big", [&](int c, const uint16_t* x, uint16_t* y, void* ws, size_t wsb, int* done, cudaStream_t cs, size_t* need) { return mma_gemm_grid_run<Tiered, 4, 4, 3, 2>(F1(c), O, K, x, M, nullptr, y, ws, wsb, cs, need); });
        add("v2 big", [&](int c, const uint16_t* x, uint16_t* y, void* ws, size_t wsb, int* done, cudaStream_t cs, size_t* need) { return mma_gemm_grid_run<Tiered2, 4, 4, 3, 2>(F2(c), O, K, x, M, nullptr, y, ws, wsb, cs, need); });
        if (have_nib) add("12-bit big", [&](int c, const uint16_t* x, uint16_t* y, void* ws, size_t wsb, int* done, cudaStream_t cs, size_t* need) { return mma_gemm_grid_run<Nib, 4, 4, 3, 2>(FN(c), O, K, x, M, nullptr, y, ws, wsb, cs, need); });
        goto run;
    }
    add("v1 lib", [&](int c, const uint16_t* x, uint16_t* y, void* ws, size_t wsb, int* done, cudaStream_t cs, size_t* need) { return mma_gemm_run(F1(c), O, K, x, M, nullptr, y, ws, wsb, done, cs, need); });
    if (M > 16 && M <= 128) {  // the shipped kernels of each layout at this many tokens (the library's own routes: step, mid, wg)
        add("v1 route", [&](int c, const uint16_t* x, uint16_t* y, void* ws, size_t wsb, int* done, cudaStream_t cs, size_t* need) { return mma_linear_run(F1(c), O, K, x, M, nullptr, y, -1, ws, wsb, done, cs, need); });
        if (have_nib) add("12-bit route", [&](int c, const uint16_t* x, uint16_t* y, void* ws, size_t wsb, int* done, cudaStream_t cs, size_t* need) { return mma_linear_run(FN(c), O, K, x, M, nullptr, y, -1, ws, wsb, done, cs, need); });
    }
    if (has("n") && have_nib) add("12-bit", [&](int c, const uint16_t* x, uint16_t* y, void* ws, size_t wsb, int* done, cudaStream_t cs, size_t* need) { return mma_gemm_run(FN(c), O, K, x, M, nullptr, y, ws, wsb, done, cs, need); });
    if (has("l")) add("v2 lib", [&](int c, const uint16_t* x, uint16_t* y, void* ws, size_t wsb, int* done, cudaStream_t cs, size_t* need) { return mma_gemm_run(F2(c), O, K, x, M, nullptr, y, ws, wsb, done, cs, need); });
#define W(OPT, UNR, NW) \
    if (has("o" #OPT "u" #UNR "w" #NW ",") || which == "all") add("o" #OPT "u" #UNR "w" #NW, [&](int c, const uint16_t* x, uint16_t* y, void* ws, size_t wsb, int* done, cudaStream_t cs, size_t* need) { \
        Tiered2o<OPT> f{(const uint8_t*)p2.da[c], (const uint8_t*)p2.db[c], (const int32_t*)p2.dc[c], {p2.t[0], p2.t[1], p2.t[2]}}; return step_run<Tiered2o<OPT>, UNR, NW>(f, O, K, x, M, y, ws, wsb, done, cs, need); });
#define B(OPT, UNR, NW, MB) \
    if (has("o" #OPT "u" #UNR "w" #NW "b" #MB ",") || which == "all") add("o" #OPT "u" #UNR "w" #NW "b" #MB, [&](int c, const uint16_t* x, uint16_t* y, void* ws, size_t wsb, int* done, cudaStream_t cs, size_t* need) { \
        Tiered2o<OPT> f{(const uint8_t*)p2.da[c], (const uint8_t*)p2.db[c], (const int32_t*)p2.dc[c], {p2.t[0], p2.t[1], p2.t[2]}}; return step_run<Tiered2o<OPT>, UNR, NW, MB>(f, O, K, x, M, y, ws, wsb, done, cs, need); });
#define SWV(OPT, UNR, NW, MB) \
    if (M <= 8 && (has("o" #OPT "u" #UNR "w" #NW "b" #MB "s,") || which == "all")) add("o" #OPT "u" #UNR "w" #NW "b" #MB "s", [&](int c, const uint16_t* x, uint16_t* y, void* ws, size_t wsb, int* done, cudaStream_t cs, size_t* need) { \
        Tiered2o<OPT> f{(const uint8_t*)p2.da[c], (const uint8_t*)p2.db[c], (const int32_t*)p2.dc[c], {p2.t[0], p2.t[1], p2.t[2]}}; return step_run<Tiered2o<OPT>, UNR, NW, MB, true>(f, O, K, x, M, y, ws, wsb, done, cs, need); });
#define BF(OPT, UNR, NW, MB) \
    if (have_flags && (has("o" #OPT "u" #UNR "w" #NW "b" #MB ",") || which == "all")) add("o" #OPT "u" #UNR "w" #NW "b" #MB, [&](int c, const uint16_t* x, uint16_t* y, void* ws, size_t wsb, int* done, cudaStream_t cs, size_t* need) { \
        Tiered2o<OPT> f{(const uint8_t*)p2f.da[c], (const uint8_t*)p2f.db[c], (const int32_t*)p2f.dc[c], {p2f.t[0], p2f.t[1], p2f.t[2]}}; return step_run<Tiered2o<OPT>, UNR, NW, MB>(f, O, K, x, M, y, ws, wsb, done, cs, need); });
#define SWF(OPT, UNR, NW, MB) \
    if (have_flags && M <= 8 && (has("o" #OPT "u" #UNR "w" #NW "b" #MB "s,") || which == "all")) add("o" #OPT "u" #UNR "w" #NW "b" #MB "s", [&](int c, const uint16_t* x, uint16_t* y, void* ws, size_t wsb, int* done, cudaStream_t cs, size_t* need) { \
        Tiered2o<OPT> f{(const uint8_t*)p2f.da[c], (const uint8_t*)p2f.db[c], (const int32_t*)p2f.dc[c], {p2f.t[0], p2f.t[1], p2f.t[2]}}; return step_run<Tiered2o<OPT>, UNR, NW, MB, true>(f, O, K, x, M, y, ws, wsb, done, cs, need); });
#define V(OPT, UNR) \
    if (has("o" #OPT "u" #UNR ",") || which == "all") add("o" #OPT "u" #UNR, [&](int c, const uint16_t* x, uint16_t* y, void* ws, size_t wsb, int* done, cudaStream_t cs, size_t* need) { \
        Tiered2o<OPT> f{(const uint8_t*)p2.da[c], (const uint8_t*)p2.db[c], (const int32_t*)p2.dc[c], {p2.t[0], p2.t[1], p2.t[2]}}; return step_run<Tiered2o<OPT>, UNR>(f, O, K, x, M, y, ws, wsb, done, cs, need); });
    V(0, 1) V(8, 1) V(9, 1) V(11, 1) V(15, 1) V(8, 2) V(15, 2) V(1, 1) V(2, 1) V(4, 1) V(24, 3) V(27, 3) V(32, 1) V(40, 1) V(72, 1) V(104, 1) V(64, 1) W(8, 1, 4) W(8, 1, 8) W(24, 3, 4) W(56, 3, 4) W(24, 3, 8) W(40, 1, 4) B(24, 3, 4, 4) B(56, 3, 4, 4) B(24, 3, 4, 3) B(24, 3, 8, 2) B(8, 1, 4, 3) SWV(8, 1, 8, 0) SWV(8, 1, 4, 0) SWV(24, 3, 4, 3) SWV(24, 3, 4, 4) SWV(56, 3, 4, 4) SWV(24, 3, 8, 2) SWV(8, 1, 4, 5) SWV(40, 1, 4, 4) V(136, 1) V(137, 1) V(152, 3) B(152, 3, 4, 4) B(152, 3, 4, 3) SWV(152, 3, 4, 4) SWV(152, 3, 4, 3) SWV(136, 1, 4, 5) SWV(136, 1, 8, 0) SWV(184, 3, 4, 4) B(184, 3, 4, 4) W(136, 1, 4) B(136, 1, 4, 5) SWV(136, 1, 4, 0) SWV(136, 1, 4, 6) SWV(152, 3, 8, 2) B(168, 1, 4, 4) B(168, 1, 4, 5) B(168, 1, 4, 6) B(168, 1, 2, 8) B(168, 1, 2, 10) B(168, 1, 2, 12) B(168, 1, 1, 16) B(168, 1, 1, 20) B(168, 1, 1, 24) SWV(168, 1, 4, 4) SWV(168, 1, 4, 5) SWV(168, 1, 4, 6) SWV(168, 1, 2, 8) SWV(168, 1, 2, 10) SWV(168, 1, 2, 12) SWV(168, 1, 1, 16) SWV(168, 1, 1, 20) SWV(168, 1, 1, 24) SWV(232, 1, 4, 5) B(184, 3, 4, 3) SWV(1064, 1, 4, 5) SWV(2088, 1, 4, 5) SWV(4136, 1, 4, 5) SWV(7208, 1, 4, 5) SWV(1064, 1, 4, 6) SWV(7208, 1, 4, 6) SWV(1064, 1, 4, 4) B(1064, 1, 4, 5) B(7208, 1, 4, 5) B(1064, 1, 4, 6) SWV(264, 1, 4, 4) SWV(264, 1, 4, 5) SWV(296, 1, 4, 4) SWV(296, 1, 4, 5) SWV(296, 1, 4, 6) B(264, 1, 4, 4) B(296, 1, 4, 4) B(296, 1, 4, 5) SWV(280, 3, 4, 4) SWV(280, 3, 4, 3) SWV(312, 3, 4, 4) SWV(312, 3, 4, 3) B(280, 3, 4, 4) B(312, 3, 4, 4) B(280, 3, 4, 3) SWV(296, 1, 2, 8) B(296, 1, 2, 8) SWV(296, 1, 8, 0) V(8200, 1) V(8232, 1) B(8232, 1, 4, 4) B(8488, 1, 4, 4) B(8472, 3, 4, 4) SWV(33064, 1, 4, 4) B(33064, 1, 4, 4) SWV(16680, 1, 4, 2) SWV(296, 1, 4, 2) B(16680, 1, 4, 2) B(296, 1, 4, 2) SWV(33048, 3, 4, 4) B(33048, 3, 4, 4) SWV(16664, 3, 4, 2) SWV(280, 3, 4, 2) SWV(33064, 1, 4, 5) SWV(33064, 1, 2, 8) SWV(184, 3, 2, 8) SWV(152, 3, 2, 8) B(152, 3, 2, 8) B(184, 3, 2, 8) SWF(552, 1, 2, 8) SWF(552, 1, 4, 4) SWF(552, 1, 1, 16) SWF(520, 1, 2, 8) SWF(568, 3, 2, 8) SWF(568, 3, 4, 4) SWF(536, 3, 4, 4) BF(552, 1, 2, 8) BF(552, 1, 4, 4) BF(568, 3, 4, 4) BF(568, 3, 2, 8) SWF(552, 1, 3, 5) SWF(16936, 1, 4, 2) SWF(16936, 1, 2, 4) SWF(33320, 1, 4, 4) SWF(33320, 1, 2, 8) SWF(16952, 3, 4, 2) SWF(33336, 3, 4, 4) BF(33320, 1, 4, 4) BF(16936, 1, 4, 2)
run:
    for (auto& v : vars) prepare(v, O);
    int rounds = getenv("ROUNDS") ? atoi(getenv("ROUNDS")) : 5;
    sm_ghz();
    for (int r = 0; r < rounds; r++)
        for (size_t i = 0; i < vars.size(); i++) time_once(vars[(i + r * 7) % vars.size()], O, K, d_x, d_y, iters);
    std::vector<uint16_t> out1;
    for (auto& v : vars) {
        CK((cudaError_t)v.launch(0, d_x, d_y, v.ws, v.need, v.done, 0, nullptr));
        CK(cudaDeviceSynchronize());
        v.out.resize(O * M);
        CK(cudaMemcpy(v.out.data(), d_y, 2 * O * M, cudaMemcpyDeviceToHost));
        if (&v == &vars[0]) out1 = v.out;
    }
    printf("%-10s %9s %9s %7s   (median of %d rounds, interleaved; clk = ns a step an SM x the SM clock measured around each timing)\n", "variant", "ns/step", "clk/step", "GHz", rounds);
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
        printf("%-10s %9.1f %9.1f %7.3f   %s%s\n", v.name.c_str(), median(v.ns), median(v.clk), median(v.ghz), worst / big < 1e-2 ? "ok" : "TOO FAR", v.out == out1 ? ", = v1 bit for bit" : ", differs from v1's");
    }
    return 0;
}
