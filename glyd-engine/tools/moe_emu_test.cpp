// kernels/moe.cu's kernels (the router, the activation between the experts' products, the shared expert's sum) run on the host (tools/cuda_emu.h) against the plain formulas, on shapes of the four models
// (8 of 256 experts, 8 of 128, 8 of 288, 22 of 512) and the edges (k = E, fewer experts than lanes, rows and tokens that are not a multiple of the block's, equal scores); and kernels/moe_rows.cu's, the sparse
// experts' second product (the row layouts at 4, 5 and 6 bits), against the dense sum over every neuron in double: random relu squared rows of about 90% zeros, a pair of all zeros and one of all
// neurons, negative zeros and subnormals, a plan with pairs routed nowhere, one to eight blocks a pair, outputs that are one run, three runs and past a block's 1,024; and over an expert cache's places (the plan
// made of the pairs' places, the experts of a call in waves of at most as many as the places, each wave's parts kept for the last one's sum); and kernels/moe_hb.cu's, the Hummingbird codebook rows (2 and 3 bits), each kernel against
// the formulas in double and the format packed here apart from the engine's packer: the rotation and its inverse (hb_rot_in, hb_sum_rot), the group ids, the first product with relu squared (hb_up2, hb_up3: also the same bits on a
// repeat and for the same experts in other places), the second product over places in waves (hb_down2, hb_down3), and the chain of them against the experts computed in the original basis.
//   c++ -std=c++17 -O1 -fno-strict-aliasing -pthread tools/moe_emu_test.cpp -o /tmp/moe_emu_test && /tmp/moe_emu_test      (with the argument `hb`: the Hummingbird kernels alone)
#include "cuda_emu.h"
#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <random>
#include "../kernels/moe.cu"
#include "../kernels/moe_rows.cu"
#include "../kernels/moe_hb.cu"

static int fails = 0, checks = 0;
#define CHECK(c, ...) \
    do { \
        checks++; \
        if (!(c)) { \
            if (fails++ < 20) { \
                std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
                std::printf(__VA_ARGS__); \
                std::printf("\n"); \
            } \
        } \
    } while (0)

// 64-byte aligned, zeroed memory (the kernels' 16-byte loads).
struct Mem {
    void* p = nullptr;
    explicit Mem(size_t bytes) {
        const size_t n = (bytes + 63) / 64 * 64 + 64;
        if (posix_memalign(&p, 64, n)) std::abort();
        std::memset(p, 0, n);
    }
    ~Mem() { std::free(p); }
    template <class T> T* as() const { return static_cast<T*>(p); }
};

struct Rng {
    std::mt19937_64 g{12345};
    float f(float lo, float hi) { return lo + (hi - lo) * (float)(g() >> 40) / (float)(1 << 24); }
    float n() { return f(-1, 1) + f(-1, 1) + f(-1, 1); }  // (a bell, sigma 1)
};

static uint16_t bits(float x) { return f2b(x).x; }
static float val(uint16_t b) { return __uint_as_float((unsigned)b << 16); }
static float round16(float x) { return val(bits(x)); }
// Places of bf16 between two values.
static int places(uint16_t a, uint16_t b) {
    auto key = [](uint16_t u) { return (u & 0x8000) ? -(int)(u & 0x7fff) : (int)u; };
    return std::abs(key(a) - key(b));
}

// ---- moe_logits: against the sums in double -----------------------------------------------------------------------------------------------
static void logits_case(Rng& r, int T, int E, int H, int xs) {
    Mem xm((size_t)T * xs * 2), wm((size_t)E * H * 2), om((size_t)T * E * 4);
    auto *x = xm.as<uint16_t>(), *w = wm.as<uint16_t>();
    auto* out = om.as<float>();
    for (int t = 0; t < T; t++)
        for (int i = 0; i < xs; i++) x[(size_t)t * xs + i] = bits(r.f(-2, 2));
    for (size_t i = 0; i < (size_t)E * H; i++) w[i] = bits(r.f(-0.1f, 0.1f));
    launch(Dim3{(unsigned)(E + 7) / 8, (unsigned)(T + 7) / 8, 1}, Dim3{256, 1, 1}, moe_logits, (const bf16*)x, xs, (const bf16*)w, out, T, E, H);
    int bad = 0;
    for (int t = 0; t < T; t++)
        for (int e = 0; e < E; e++) {
            double exact = 0, mag = 0;
            for (int i = 0; i < H; i++) {
                const double p = (double)val(x[(size_t)t * xs + i]) * val(w[(size_t)e * H + i]);
                exact += p;
                mag += std::fabs(p);
            }
            bad += std::fabs(out[(size_t)t * E + e] - exact) > 5e-6 * mag + 1e-30;
        }
    CHECK(bad == 0, "moe_logits T %d E %d H %d xs %d: %d of %d logits off", T, E, H, xs, bad, T * E);
}

// ---- moe_select: against a sort -------------------------------------------------------------------------------------------------------------
struct SelCase {
    int T, E, k, flags;
    bool bias, escale, ties;
    float scale;
};

static void select_case(Rng& r, const SelCase& c) {
    const int T = c.T, E = c.E, k = c.k;
    Mem lm((size_t)T * E * 4), bm((size_t)E * 4), sm((size_t)E * 4), im((size_t)T * k * 8), wm((size_t)T * k * 4);
    auto *lg = lm.as<float>(), *bias = bm.as<float>(), *esc = sm.as<float>(), *wts = wm.as<float>();
    auto* ids = im.as<long long>();
    for (size_t i = 0; i < (size_t)T * E; i++) {
        const float v = 1.3f * r.n();
        lg[i] = c.ties ? std::round(v * 4.f) / 4.f : v;  // (quarters: many equal logits, and after bf16's rounding more)
    }
    for (int e = 0; e < E; e++) {
        bias[e] = c.ties ? std::round(r.f(-0.5f, 0.5f) * 8.f) / 8.f : r.f(-0.5f, 0.5f);
        esc[e] = r.f(0.5f, 1.5f);
    }
    const int n = E <= 256 ? 8 : E <= 512 ? 16 : 32;
    auto kernel = n == 8 ? moe_select8 : n == 16 ? moe_select16 : moe_select32;
    launch(Dim3{(unsigned)(T + 3) / 4, 1, 1}, Dim3{128, 1, 1}, kernel, (const float*)lg, c.bias ? (const float*)bias : (const float*)nullptr, c.escale ? (const float*)esc : (const float*)nullptr, ids, wts, T, E, k, c.flags, c.scale);
    int bad_ids = 0, bad_w = 0;
    for (int t = 0; t < T; t++) {
        std::vector<float> s(E), rank(E);
        float mx = -INFINITY;
        for (int e = 0; e < E; e++) {
            s[e] = lg[(size_t)t * E + e];
            if (c.flags & MOE_BF16) s[e] = round16(s[e]);
            mx = std::max(mx, s[e]);
        }
        if (c.flags & MOE_SIGMOID) {
            for (int e = 0; e < E; e++) s[e] = 1.f / (1.f + expf(-s[e]));
        } else {
            float z = 0;
            for (int e = 0; e < E; e++) z += (s[e] = expf(s[e] - mx));
            for (int e = 0; e < E; e++) s[e] /= z;
        }
        for (int e = 0; e < E; e++) rank[e] = ((c.flags & MOE_SIGMOID) && c.bias) ? s[e] + bias[e] : s[e];
        std::vector<int> ord(E);
        std::iota(ord.begin(), ord.end(), 0);
        std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return rank[a] > rank[b]; });  // (stable: equal ones by the smaller expert)
        float tot = 0;
        for (int j = 0; j < k; j++) tot += s[ord[j]];
        for (int j = 0; j < k; j++) {
            const int e = ord[j];
            float w = s[e];
            if (c.flags & MOE_RENORM) w /= (tot + 1e-20f);
            if (c.escale) w *= esc[e];
            w *= c.scale;
            bad_ids += ids[(size_t)t * k + j] != e;
            bad_w += std::fabs(wts[(size_t)t * k + j] - w) > 2e-6 * std::fabs(w) + 1e-30;
        }
    }
    CHECK(bad_ids == 0 && bad_w == 0, "moe_select T %d E %d k %d flags %d bias %d escale %d ties %d: %d ids, %d weights off of %d", T, E, k, c.flags, c.bias, c.escale, c.ties, bad_ids, bad_w, T * k);
}

// ---- moe_act and moe_add: against the formulas ----------------------------------------------------------------------------------------------
static void act_case(Rng& r, int rows, int I, int mode, float lim, bool inplace) {
    const bool gated = mode != MOE_RELU2;
    const int W = gated ? 2 * I : I;
    Mem gm((size_t)rows * W * 2), om((size_t)rows * I * 2);
    auto *g = gm.as<uint16_t>(), *o = (inplace && !gated) ? gm.as<uint16_t>() : om.as<uint16_t>();
    for (size_t i = 0; i < (size_t)rows * W; i++) g[i] = bits(r.f(-15, 15));
    std::vector<uint16_t> in(g, g + (size_t)rows * W);
    launch(Dim3{(unsigned)rows, (unsigned)((I / 8 + 255) / 256), 1}, Dim3{256, 1, 1}, moe_act, (const bf16*)g, (bf16*)o, I, mode, lim);
    int bad = 0;
    for (int t = 0; t < rows; t++)
        for (int i = 0; i < I; i++) {
            float want;
            if (!gated) {
                const float x = std::max(val(in[(size_t)t * I + i]), 0.f);
                want = x * x;
            } else {
                const float a = std::min(val(in[(size_t)t * 2 * I + i]), lim), u = std::min(std::max(val(in[(size_t)t * 2 * I + I + i]), -lim), lim);
                want = (mode == MOE_GELU ? 0.5f * a * (1.f + tanhf(0.7978845608028654f * (a + 0.044715f * a * a * a))) : a / (1.f + expf(-a))) * u;
            }
            bad += places(o[(size_t)t * I + i], bits(want)) > 1;
        }
    CHECK(bad == 0, "moe_act rows %d I %d mode %d lim %g: %d of %d more than a place off", rows, I, mode, lim, bad, rows * I);
}

static void add_case(Rng& r, int rows, int H, bool gated) {
    Mem ym((size_t)rows * H * 2), sm((size_t)rows * H * 2), gm((size_t)rows * 4);
    auto *y = ym.as<uint16_t>(), *s = sm.as<uint16_t>();
    auto* gate = gm.as<float>();
    for (size_t i = 0; i < (size_t)rows * H; i++) {
        y[i] = bits(r.f(-3, 3));
        s[i] = bits(r.f(-3, 3));
    }
    for (int t = 0; t < rows; t++) gate[t] = r.f(-4, 4);
    std::vector<uint16_t> y0(y, y + (size_t)rows * H);
    launch(Dim3{(unsigned)rows, (unsigned)((H / 8 + 255) / 256), 1}, Dim3{256, 1, 1}, moe_add, (bf16*)y, (const bf16*)s, gated ? (const float*)gate : (const float*)nullptr, H);
    int bad = 0;
    for (int t = 0; t < rows; t++) {
        const float g = gated ? 1.f / (1.f + expf(-gate[t])) : 1.f;
        for (int i = 0; i < H; i++) bad += places(y[(size_t)t * H + i], bits(val(y0[(size_t)t * H + i]) + g * val(s[(size_t)t * H + i]))) > 1;
    }
    CHECK(bad == 0, "moe_add rows %d H %d gate %d: %d of %d more than a place off", rows, H, gated, bad, rows * H);
}

// bf16 words of floats (a scale's sign from `neg`).
static uint16_t bits_of(float x) { return f2b(x).x; }
static uint16_t bits_of_scale(float x, bool neg) { return f2b(neg ? -x : x).x; }

// ---- the sparse experts: moe_down_rows and moe_rows_sum ------------------------------------------------------------------------------------------
// The row layout stated here from its doc, apart from glyd-gpu's packer: a row of `runs` runs of 32 codes u (0 to 2^bits - 1): per run four words of low nibbles (pair p, codes 2p and 2p + 1, at bits 4 (p mod 4) and
// 16 + 4 (p mod 4) of word p / 4), then at 5 bits a word of the high bits (pair p at bits p and 16 + p), at 6 bits two words (2 bits, at 2 (p mod 8) and 16 + 2 (p mod 8) of word p / 8), the planes after
// all the runs' nibbles, the row up to 16 bytes.
static std::vector<unsigned char> rows_row(int bits, const std::vector<int>& u, int runs) {
    const int hb = 4 * (bits - 4);
    std::vector<unsigned char> row((size_t)(runs * (16 + hb) + 15) / 16 * 16, 0);
    auto put = [&](size_t at, unsigned w) { std::memcpy(&row[at], &w, 4); };
    for (int k = 0; k < runs; k++) {
        unsigned lo[4] = {0, 0, 0, 0}, hi[2] = {0, 0};
        for (int i = 0; i < 32; i++) {
            const int pr = i / 2, h = i % 2;
            const unsigned c = (unsigned)u[k * 32 + i];
            lo[pr / 4] |= (c & 15u) << (4 * (pr % 4) + 16 * h);
            if (bits == 5) hi[0] |= (c >> 4) << (pr + 16 * h);
            if (bits == 6) hi[pr / 8] |= (c >> 4) << (2 * (pr % 8) + 16 * h);
        }
        for (int q = 0; q < 4; q++) put(16 * k + 4 * q, lo[q]);
        for (int q = 0; q < hb / 4; q++) put(16 * runs + hb * k + 4 * q, hi[q]);
    }
    return row;
}

// The layout's bytes pinned by the literals glyd-gpu's pack.rs test holds (a run of codes -q to q, every second one from the middle of the range, written apart from either packer in an independent script).
static void rows_layout_literals() {
    const unsigned char want4[16] = {0x51, 0xd9, 0x73, 0xfb, 0x62, 0xea, 0x84, 0x1c, 0x73, 0xfb, 0x95, 0x2d, 0x84, 0x1c, 0xa6, 0x3e};
    const unsigned char want5[32] = {0x51, 0xd9, 0x73, 0xfb, 0x51, 0xd9, 0x73, 0xfb, 0x62, 0xea, 0x84, 0x0c, 0x62, 0xea, 0x84, 0x1c, 0xf0, 0xf0, 0xf0, 0x78};
    const unsigned char want6[32] = {0x51, 0xd9, 0x73, 0xfb, 0x51, 0xd9, 0x73, 0xfb, 0x51, 0xd9, 0x73, 0xfb, 0x51, 0xd9, 0x73, 0xfb, 0x00, 0x55, 0x00, 0x55, 0xaa, 0xff, 0xaa, 0xff};
    const unsigned char* want[3] = {want4, want5, want6};
    for (int bits = 4; bits <= 6; bits++) {
        const int q = (1 << (bits - 1)) - 1;
        std::vector<int> u(32);
        for (int i = 0; i < 32; i++) u[i] = (2 * i) % (2 * q + 1) - q + q + 1;
        const std::vector<unsigned char> row = rows_row(bits, u, 1);
        CHECK(row.size() == (bits == 4 ? 16u : 32u) && std::memcmp(row.data(), want[bits - 4], row.size()) == 0, "the row layout's literal bytes at %d bits", bits);
    }
}

// A weight as the kernels decode it: the product of the code (less its bias) and the run's scale, rounded to bf16 once.
static float rows_weight(int bits, int u, uint16_t scale) { return round16((float)(u - (1 << (bits - 1))) * val(scale)); }

// The plan Lib::moe_way makes of ids (an id outside 0 to E - 1 is routed nowhere): [the experts hit, their numbers, their starts, the pairs' total, the pairs by expert]; the pairs in the plan's order.
static std::vector<int> make_plan(const std::vector<long long>& ids, int E, int* plan) {
    const int P = (int)ids.size();
    std::vector<std::vector<int>> by(E);
    for (int j = 0; j < P; j++)
        if (ids[j] >= 0 && ids[j] < E) by[ids[j]].push_back(j);
    int hits = 0, n = 0;
    std::vector<int> order;
    for (int e = 0; e < E; e++)
        if (!by[e].empty()) {
            plan[1 + hits] = e;
            plan[1 + E + hits] = n;
            hits++;
            for (int j : by[e]) order.push_back(j), n++;
        }
    plan[0] = hits;
    plan[1 + E + hits] = n;
    for (int i = 0; i < n; i++) plan[2 + 2 * E + i] = order[i];
    return order;
}

struct RowsCase {
    int bits, E, inter, hidden, T, k, split;
    double zeros;  // the share of an activation row that is zero
    bool special;  // all zeros for one pair and every neuron for another, negative zeros, subnormals; a pair routed nowhere
};

static void rows_case(Rng& r, const RowsCase& c) {
    const int E = c.E, inter = c.inter, hidden = c.hidden, T = c.T, k = c.k, P = T * k, runs = hidden / 32, bits = c.bits;
    const int rstride = (runs * (16 + 4 * (bits - 4)) + 15) / 16 * 16;
    // the experts' codes and scales: row n of expert e at e * inter + n
    std::vector<int> codes((size_t)E * inter * hidden);
    std::vector<uint16_t> scales((size_t)E * inter * runs);
    for (auto& u : codes) u = (int)(r.g() % (1u << bits));
    for (auto& s : scales) s = (r.g() % 20 == 0) ? (uint16_t)0 : bits_of_scale(r.f(0.002f, 0.03f), r.g() % 7 == 0);
    Mem dm((size_t)E * inter * rstride), sm((size_t)E * inter * runs * 2);
    for (int e = 0; e < E; e++)
        for (int n = 0; n < inter; n++) {
            const std::vector<unsigned char> row = rows_row(bits, std::vector<int>(codes.begin() + ((size_t)e * inter + n) * hidden, codes.begin() + ((size_t)e * inter + n + 1) * hidden), runs);
            std::memcpy(dm.as<unsigned char>() + ((size_t)e * inter + n) * rstride, row.data(), rstride);
        }
    std::memcpy(sm.p, scales.data(), scales.size() * 2);
    // the choices: k distinct experts a token (E >= k), some routed nowhere with `special`
    std::vector<long long> ids(P);
    for (int t = 0; t < T; t++) {
        std::vector<int> pool(E);
        std::iota(pool.begin(), pool.end(), 0);
        std::shuffle(pool.begin(), pool.end(), r.g);
        for (int j = 0; j < k; j++) ids[t * k + j] = pool[j];
    }
    if (c.special && P >= 3) ids[P - 2] = -1;
    Mem planm(((size_t)2 + 2 * E + P) * 4), idm((size_t)P * 8), wm((size_t)P * 4), am((size_t)P * inter * 2);
    int* plan = planm.as<int>();
    const std::vector<int> order = make_plan(ids, E, plan);
    const int n = (int)order.size();
    std::memcpy(idm.p, ids.data(), (size_t)P * 8);
    std::vector<float> wts(P);
    for (auto& w : wts) w = r.f(0.1f, 3.f);
    std::memcpy(wm.p, wts.data(), (size_t)P * 4);
    // the activations, a row for each pair in the plan's order: relu squared of something, about `zeros` of them zero
    auto* a = am.as<uint16_t>();
    for (int i = 0; i < n; i++)
        for (int q = 0; q < inter; q++) {
            uint16_t v = 0;
            if ((double)(r.g() % 100000) / 100000.0 >= c.zeros) v = bits_of(r.f(0.01f, 4.f));
            a[(size_t)i * inter + q] = v;
        }
    if (c.special && n >= 2) {
        for (int q = 0; q < inter; q++) {
            a[q] = (q % 5 == 0) ? 0x8000 : 0;                           // the first pair: all zeros, some of them negative
            a[(size_t)(n - 1) * inter + q] = bits_of(r.f(0.01f, 4.f));  // the last: every neuron
        }
        a[(size_t)(n - 1) * inter] = 0x0001;  // (a subnormal: not zero)
        a[(size_t)(n - 1) * inter + inter - 1] = bits_of(3.0e3f);  // (a large one)
    }
    // the kernels
    const int slices = (hidden + 1023) / 1024;
    Mem pm((size_t)P * c.split * hidden * 4), ym((size_t)T * hidden * 2);
    auto* part = pm.as<float>();
    auto kernel = bits == 4 ? moe_down_rows4 : bits == 5 ? moe_down_rows5 : moe_down_rows6;
    launch(Dim3{(unsigned)P, (unsigned)c.split, (unsigned)slices}, Dim3{256, 1, 1}, kernel, (const bf16*)a, (const int*)plan, (const long long*)idm.as<long long>(), dm.as<unsigned char>(), sm.as<unsigned char>(),
           (long long)inter * rstride, (long long)inter * runs * 2, part, E, inter, hidden, c.split);
    launch(Dim3{(unsigned)T, (unsigned)((hidden + 63) / 64), 1}, Dim3{256, 1, 1}, moe_rows_sum, (const float*)part, (const float*)wm.as<float>(), (const long long*)idm.as<long long>(), (bf16*)ym.as<uint16_t>(), E, k, hidden, c.split, (const float*)nullptr);
    // the dense sum over every neuron, in double, for each pair (by the plan's order: its row of the activations) and each token
    std::vector<std::vector<double>> pair_sum(P);
    std::vector<std::vector<double>> pair_mag(P);
    for (int i = 0; i < n; i++) {
        const int j = order[i];
        const int e = (int)ids[j];
        pair_sum[j].assign(hidden, 0.0);
        pair_mag[j].assign(hidden, 0.0);
        for (int q = 0; q < inter; q++) {
            const double av = val(a[(size_t)i * inter + q]);
            if (av == 0.0) continue;  // (zero times a weight: nothing; the dense product adds it, to no effect)
            for (int o = 0; o < hidden; o++) {
                const size_t at = ((size_t)e * inter + q);
                const double w = rows_weight(bits, codes[at * hidden + o], scales[at * runs + o / 32]);
                pair_sum[j][o] += av * w;
                pair_mag[j][o] += std::fabs(av * w);
            }
        }
    }
    int bad_part = 0, bad_y = 0;
    for (int j = 0; j < P; j++) {
        if (pair_sum[j].empty()) continue;
        for (int o = 0; o < hidden; o++) {
            double got = 0;
            for (int s = 0; s < c.split; s++) got += part[((size_t)j * c.split + s) * hidden + o];
            bad_part += std::fabs(got - pair_sum[j][o]) > 3e-6 * pair_mag[j][o] + 1e-30;
        }
    }
    // the outputs: within a place of bf16, or (where the sum is lost in the terms it is made of: fp32 adds in another order) within 4e-6 of the terms' magnitude
    double first_want = 0, first_got = 0, first_mag = 0;
    for (int t = 0; t < T; t++)
        for (int o = 0; o < hidden; o++) {
            double want = 0, mag = 0;
            for (int cc = 0; cc < k; cc++) {
                const int j = t * k + cc;
                if (pair_sum[j].empty()) continue;
                want += pair_sum[j][o] * wts[j];
                mag += pair_mag[j][o] * wts[j];
            }
            const uint16_t got = ym.as<uint16_t>()[(size_t)t * hidden + o];
            const bool off = places(got, bits_of((float)want)) > 1 && std::fabs((double)val(got) - want) > 4e-6 * mag;
            if (off && bad_y++ == 0) first_want = want, first_got = val(got), first_mag = mag;
        }
    CHECK(bad_part == 0 && bad_y == 0, "moe_down_rows%d E %d inter %d hidden %d T %d k %d split %d zeros %.2f special %d: %d parts, %d outputs off (the first: %g against %g of terms %g)", bits, E, inter, hidden, T, k, c.split, c.zeros, c.special, bad_part, bad_y, first_got, first_want, first_mag);
}

// ---- the sparse experts over an expert cache's places (src/offload.rs) -------------------------------------------------------------------------------------
// A layer's experts are not all on the GPU: `slots` places hold the experts a call needs, a call that needs more is computed in waves of at most `slots` experts, and each wave's plan is made of its pairs' places
// (way: the pair's place, -1 for a pair another wave computes). The kernels read the rows at the places, a wave writes the parts of its pairs alone, and the sum, run after each wave as the engine does, is of the
// router's choices: the last wave's is the layer's. The places hold the wave's experts in an order of their own, replacing the earlier wave's.
static void cached_case(Rng& r, int bits, int E, int slots, int inter, int hidden, int T, int k, int split, int least_waves, double zeros) {
    const int P = T * k, runs = hidden / 32;
    const int rstride = (runs * (16 + 4 * (bits - 4)) + 15) / 16 * 16;
    std::vector<int> codes((size_t)E * inter * hidden);
    std::vector<uint16_t> scales((size_t)E * inter * runs);
    for (auto& u : codes) u = (int)(r.g() % (1u << bits));
    for (auto& s : scales) s = bits_of_scale(r.f(0.002f, 0.03f), r.g() % 7 == 0);
    Mem dm((size_t)slots * inter * rstride), sm((size_t)slots * inter * runs * 2);  // (the places)
    auto put_expert = [&](int e, int place) {
        for (int n = 0; n < inter; n++) {
            const std::vector<unsigned char> row = rows_row(bits, std::vector<int>(codes.begin() + ((size_t)e * inter + n) * hidden, codes.begin() + ((size_t)e * inter + n + 1) * hidden), runs);
            std::memcpy(dm.as<unsigned char>() + ((size_t)place * inter + n) * rstride, row.data(), rstride);
        }
        std::memcpy(sm.as<unsigned char>() + (size_t)place * inter * runs * 2, scales.data() + (size_t)e * inter * runs, (size_t)inter * runs * 2);
    };
    // the choices (k distinct experts a token), the experts of the call in the order they first appear, and the waves they are computed in
    std::vector<long long> ids(P);
    std::vector<int> need;
    for (int t = 0; t < T; t++) {
        std::vector<int> pool(E);
        std::iota(pool.begin(), pool.end(), 0);
        std::shuffle(pool.begin(), pool.end(), r.g);
        for (int j = 0; j < k; j++) {
            ids[t * k + j] = pool[j];
            if (std::find(need.begin(), need.end(), pool[j]) == need.end()) need.push_back(pool[j]);
        }
    }
    const int waves = ((int)need.size() + slots - 1) / slots;
    CHECK(waves >= least_waves, "cached sparse experts E %d slots %d T %d k %d: %d experts in %d waves, where at least %d are meant", E, slots, T, k, (int)need.size(), waves, least_waves);
    std::vector<float> wts(P);
    for (auto& w : wts) w = r.f(0.1f, 3.f);
    // a pair's activations (relu squared rows, about `zeros` of them zero)
    std::vector<std::vector<uint16_t>> act(P, std::vector<uint16_t>(inter));
    for (auto& row : act)
        for (auto& v : row) v = ((double)(r.g() % 100000) / 100000.0 >= zeros) ? bits_of(r.f(0.01f, 4.f)) : (uint16_t)0;
    Mem planm(((size_t)2 + 2 * E + P) * 4), idm((size_t)P * 8), waym((size_t)P * 8), wm((size_t)P * 4), am((size_t)P * inter * 2);
    const int slices = (hidden + 1023) / 1024;
    Mem pm((size_t)P * split * hidden * 4), ym((size_t)T * hidden * 2);
    std::memcpy(idm.p, ids.data(), (size_t)P * 8);
    std::memcpy(wm.p, wts.data(), (size_t)P * 4);
    auto* part = pm.as<float>();
    for (size_t i = 0; i < (size_t)P * split * hidden; i++) part[i] = 1e30f;  // (what a pair no wave has computed yet holds)
    auto kernel = bits == 4 ? moe_down_rows4 : bits == 5 ? moe_down_rows5 : moe_down_rows6;
    for (int w = 0; w < waves; w++) {
        const std::vector<int> group(need.begin() + (size_t)w * slots, need.begin() + std::min((size_t)(w + 1) * slots, need.size()));
        std::vector<int> pl(slots), place_of(E, -1);
        std::iota(pl.begin(), pl.end(), 0);
        std::shuffle(pl.begin(), pl.end(), r.g);
        for (size_t q = 0; q < group.size(); q++) {
            place_of[group[q]] = pl[q];
            put_expert(group[q], pl[q]);
        }
        std::vector<long long> way(P);
        for (int j = 0; j < P; j++) way[j] = place_of[ids[j]];
        std::memcpy(waym.p, way.data(), (size_t)P * 8);
        const std::vector<int> order = make_plan(way, E, planm.as<int>());
        for (size_t i = 0; i < order.size(); i++) std::memcpy(am.as<uint16_t>() + i * inter, act[order[i]].data(), (size_t)inter * 2);
        launch(Dim3{(unsigned)P, (unsigned)split, (unsigned)slices}, Dim3{256, 1, 1}, kernel, (const bf16*)am.as<uint16_t>(), (const int*)planm.as<int>(), (const long long*)waym.as<long long>(), dm.as<unsigned char>(),
               sm.as<unsigned char>(), (long long)inter * rstride, (long long)inter * runs * 2, part, E, inter, hidden, split);
        launch(Dim3{(unsigned)T, (unsigned)((hidden + 63) / 64), 1}, Dim3{256, 1, 1}, moe_rows_sum, (const float*)part, (const float*)wm.as<float>(), (const long long*)idm.as<long long>(), (bf16*)ym.as<uint16_t>(), E, k, hidden, split, (const float*)nullptr);
    }
    // the dense sum over every neuron, in double, of each pair by its own expert's weights
    int bad_part = 0, bad_y = 0;
    std::vector<std::vector<double>> pair_sum(P), pair_mag(P);
    for (int j = 0; j < P; j++) {
        pair_sum[j].assign(hidden, 0.0);
        pair_mag[j].assign(hidden, 0.0);
        const int e = (int)ids[j];
        for (int q = 0; q < inter; q++) {
            const double av = val(act[j][q]);
            if (av == 0.0) continue;
            for (int o = 0; o < hidden; o++) {
                const size_t at = ((size_t)e * inter + q);
                const double wv = rows_weight(bits, codes[at * hidden + o], scales[at * runs + o / 32]);
                pair_sum[j][o] += av * wv;
                pair_mag[j][o] += std::fabs(av * wv);
            }
        }
        for (int o = 0; o < hidden; o++) {
            double got = 0;
            for (int s = 0; s < split; s++) got += part[((size_t)j * split + s) * hidden + o];
            bad_part += !(std::fabs(got - pair_sum[j][o]) <= 3e-6 * pair_mag[j][o] + 1e-30);
        }
    }
    for (int t = 0; t < T; t++)
        for (int o = 0; o < hidden; o++) {
            double want = 0, mag = 0;
            for (int cc = 0; cc < k; cc++) {
                want += pair_sum[t * k + cc][o] * wts[t * k + cc];
                mag += pair_mag[t * k + cc][o] * wts[t * k + cc];
            }
            const uint16_t got = ym.as<uint16_t>()[(size_t)t * hidden + o];
            bad_y += !(places(got, bits_of((float)want)) <= 1 || std::fabs((double)val(got) - want) <= 4e-6 * mag);
        }
    CHECK(bad_part == 0 && bad_y == 0, "moe_down_rows%d over places: E %d slots %d inter %d hidden %d T %d k %d split %d, %d waves: %d parts, %d outputs off", bits, E, slots, inter, hidden, T, k, split, waves, bad_part, bad_y);
}

// ---- Hummingbird codebook rows: kernels/moe_hb.cu ---------------------------------------------------------------------------------------------------
// The format stated here from its definition, apart from glyd-gpu's packer: a row of `cols` codes of `bits` (2 or 3) is a little-endian bitstream (code j at stream bits bits j .. bits j + bits - 1, stream bit i
// being bit i mod 8 of byte i div 8), a run of 32 codes is 4 bits bytes; a run has a bf16 scale, a row a codebook of 2^bits bf16 entries; a weight is f32(scale of its run) * f32(codebook[code]).
static void hb_pack_row(int bits, const uint8_t* code, int cols, unsigned char* out) {
    std::memset(out, 0, (size_t)cols * bits / 8);
    for (int j = 0; j < cols; j++)
        for (int b = 0; b < bits; b++)
            if ((code[j] >> b) & 1) {
                const size_t i = (size_t)bits * j + b;
                out[i / 8] |= (unsigned char)(1u << (i % 8));
            }
}

// The packer pinned by bytes worked out apart from it: codes 0 to 7 at 3 bits are the 24-bit integer 0xfac688, and a run of 32 codes at each width (a big-integer shift of each code to its bit, in an independent script).
static void hb_layout_literals() {
    uint8_t c[32];
    unsigned char out[12];
    for (int j = 0; j < 8; j++) c[j] = (uint8_t)j;
    hb_pack_row(3, c, 8, out);
    const unsigned char want38[3] = {0x88, 0xc6, 0xfa};
    CHECK(std::memcmp(out, want38, 3) == 0, "the layout's literal bytes of codes 0 to 7 at 3 bits");
    for (int j = 0; j < 32; j++) c[j] = (uint8_t)((j * 5 + j / 4) % 4);
    hb_pack_row(2, c, 32, out);
    const unsigned char want2[8] = {0xe4, 0x39, 0x4e, 0x93, 0xe4, 0x39, 0x4e, 0x93};
    CHECK(std::memcmp(out, want2, 8) == 0, "the layout's literal bytes of a run at 2 bits");
    for (int j = 0; j < 32; j++) c[j] = (uint8_t)((j * 11 + j / 3) % 8);
    hb_pack_row(3, c, 32, out);
    const unsigned char want3[12] = {0x98, 0x55, 0xf0, 0x72, 0x88, 0x59, 0x05, 0x2f, 0x87, 0x98, 0x55, 0xf0};
    CHECK(std::memcmp(out, want3, 12) == 0, "the layout's literal bytes of a run at 3 bits");
}

// E matrices of `rows` neuron rows of `cols` weights at `bits` (the logical content: the codes, the runs' scales, the rows' codebooks as bf16 words).
struct HbMat {
    int bits, E, rows, cols, R, NCB;
    std::vector<uint8_t> code;  // [E][rows][cols]
    std::vector<uint16_t> sc;   // [E][rows][R]
    std::vector<uint16_t> cbk;  // [E][rows][NCB]
    size_t cbytes() const { return (size_t)rows * cols * bits / 8; }  // an expert's bytes of each array
    size_t sbytes() const { return (size_t)rows * R * 2; }
    size_t bbytes() const { return (size_t)rows * NCB * 2; }
    // the weight of expert e, row n, column j, in double (the product of two bf16: exact)
    double w(int e, int n, int j) const {
        const size_t row = (size_t)e * rows + n;
        return (double)val(sc[row * R + j / 32]) * (double)val(cbk[row * NCB + code[row * cols + j]]);
    }
};

// Random matrices: codes uniform, scales of both signs and some zero, codebook entries of both signs (the codebook is in units of the scale).
static HbMat hb_random(Rng& r, int bits, int E, int rows, int cols) {
    HbMat m;
    m.bits = bits, m.E = E, m.rows = rows, m.cols = cols, m.R = cols / 32, m.NCB = 1 << bits;
    m.code.resize((size_t)E * rows * cols);
    m.sc.resize((size_t)E * rows * m.R);
    m.cbk.resize((size_t)E * rows * m.NCB);
    for (auto& u : m.code) u = (uint8_t)(r.g() % m.NCB);
    for (auto& s : m.sc) s = (r.g() % 20 == 0) ? (uint16_t)0 : bits_of_scale(r.f(0.002f, 0.03f), r.g() % 4 == 0);
    for (auto& c : m.cbk) c = (r.g() % 16 == 0) ? (uint16_t)0 : bits_of(r.f(-3.f, 3.f));
    return m;
}

// `slots` places holding experts of m end to end (an expert at place p is at p times the strides: the experts' own order is the places' when p = e).
struct HbPlaces {
    int bits, rows, cols;
    Mem codes, scales, cb;
    long long cs, ss, bs;
    HbPlaces(const HbMat& m, int slots) : bits(m.bits), rows(m.rows), cols(m.cols), codes(slots * m.cbytes()), scales(slots * m.sbytes()), cb(slots * m.bbytes()), cs((long long)m.cbytes()), ss((long long)m.sbytes()), bs((long long)m.bbytes()) {}
    void put(const HbMat& m, int e, int p) {
        const size_t cstride = (size_t)cols * bits / 8;
        for (int n = 0; n < rows; n++) hb_pack_row(bits, &m.code[((size_t)e * rows + n) * cols], cols, codes.as<unsigned char>() + p * cs + n * cstride);
        std::memcpy(scales.as<unsigned char>() + p * ss, &m.sc[(size_t)e * rows * m.R], (size_t)ss);
        std::memcpy(cb.as<unsigned char>() + p * bs, &m.cbk[(size_t)e * rows * m.NCB], (size_t)bs);
    }
};

static std::vector<signed char> hb_signs(Rng& r, int n) {
    std::vector<signed char> s(n);
    for (auto& v : s) v = (r.g() & 1) ? 1 : -1;
    return s;
}

static void wht64(std::vector<double>& v) {  // the unnormalized Walsh-Hadamard transform in double
    for (size_t h = 1; h < v.size(); h <<= 1)
        for (size_t i = 0; i < v.size(); i++)
            if (!(i & h)) {
                const double a = v[i], b = v[i | h];
                v[i] = a + b;
                v[i | h] = a - b;
            }
}

// ---- hb_group_ids ---------------------------------------------------------------------------------------------------------------------------------
static void hb_group_case(Rng& r, int E, int pairs, int group) {
    Mem gm(E), lm((size_t)E * 2), im((size_t)pairs * 8), om((size_t)pairs * 8);
    auto* go = gm.as<unsigned char>();
    auto* lo = lm.as<uint16_t>();
    auto* ids = im.as<long long>();
    auto* out = om.as<long long>();
    int count[2] = {0, 0};
    for (int e = 0; e < E; e++) {
        const bool absent = r.g() % 4 == 0;  // (an expert in no group: 255)
        const int g = (int)(r.g() % 2);
        go[e] = absent ? 255 : (unsigned char)g;
        lo[e] = absent ? (uint16_t)7 : (uint16_t)count[g]++;
    }
    for (int j = 0; j < pairs; j++) ids[j] = (long long)(r.g() % (E + 6)) - 3;  // (-3 to E + 2)
    if (pairs > 0) ids[0] = -1;
    if (pairs > 1) ids[1] = E;
    if (pairs > 2) ids[2] = E - 1;
    if (pairs > 3) ids[3] = 0;
    std::memset(out, 0x55, (size_t)pairs * 8);
    launch(Dim3{(unsigned)(pairs + 255) / 256, 1, 1}, Dim3{256, 1, 1}, hb_group_ids, (const long long*)ids, (const unsigned char*)go, (const unsigned short*)lo, group, out, pairs, E);
    int bad = 0;
    for (int j = 0; j < pairs; j++) {
        const long long want = (ids[j] >= 0 && ids[j] < E && go[ids[j]] == group) ? (long long)lo[ids[j]] : -1;
        bad += out[j] != want;
    }
    CHECK(bad == 0, "hb_group_ids E %d pairs %d group %d: %d of %d off", E, pairs, group, bad, pairs);
}

// ---- hb_rot_in and hb_sum_rot ---------------------------------------------------------------------------------------------------------------------
// z with an odd mix of magnitudes (two decades either way, zeros, a few thousandfold ones).
static float hb_odd(Rng& r) {
    float v = r.n() * std::pow(10.f, r.f(-2.f, 2.f));
    if (r.g() % 11 == 0) v = 0.f;
    if (r.g() % 97 == 0) v *= 1000.f;
    return v;
}

static void hb_rot_case(Rng& r, int N, int T, int pad) {
    const int zs = N + pad;
    const float scale = (float)(1.0 / std::sqrt((double)N));
    Mem zm((size_t)T * zs * 2), om((size_t)T * N * 4), idm((size_t)T * 8), wm((size_t)T * 4), ym((size_t)T * N * 2);
    auto* z = zm.as<uint16_t>();
    auto* zr = om.as<float>();
    const std::vector<signed char> sg = hb_signs(r, N);
    for (size_t i = 0; i < (size_t)T * zs; i++) z[i] = bits(hb_odd(r));
    launch(Dim3{(unsigned)T, 1, 1}, Dim3{(unsigned)N / 4, 1, 1}, hb_rot_in, (const bf16*)z, zs, (const signed char*)sg.data(), zr, N, scale);
    int bad = 0;
    for (int t = 0; t < T; t++)
        for (int o = 0; o < N; o++) {
            double sum = 0, mag = 0;
            for (int j = 0; j < N; j++) {
                const double term = ((__builtin_popcount(o & j) & 1) ? -1.0 : 1.0) * sg[j] * val(z[(size_t)t * zs + j]);
                sum += term;
                mag += std::fabs(term);
            }
            bad += !(std::fabs(zr[(size_t)t * N + o] - scale * sum) <= 4e-6 * scale * mag + 1e-30);
        }
    CHECK(bad == 0, "hb_rot_in N %d T %d zs %d: %d of %d off", N, T, zs, bad, T * N);
    // the inverse (hb_sum_rot of one pair of weight 1, split 1) returns z within bf16's rounding, or 4e-6 of the row's magnitude
    for (int t = 0; t < T; t++) {
        idm.as<long long>()[t] = 0;
        wm.as<float>()[t] = 1.f;
    }
    launch(Dim3{(unsigned)T, 1, 1}, Dim3{(unsigned)N / 4, 1, 1}, hb_sum_rot, (const float*)zr, (const float*)wm.as<float>(), (const long long*)idm.as<long long>(), (const signed char*)sg.data(), (bf16*)ym.as<uint16_t>(), (const float*)nullptr, 1, 1, N, 1, scale);
    int bad_back = 0;
    for (int t = 0; t < T; t++) {
        double l1 = 0;
        for (int j = 0; j < N; j++) l1 += std::fabs(val(z[(size_t)t * zs + j]));
        for (int o = 0; o < N; o++) {
            const uint16_t got = ym.as<uint16_t>()[(size_t)t * N + o];
            bad_back += !(places(got, z[(size_t)t * zs + o]) <= 1 || std::fabs((double)val(got) - val(z[(size_t)t * zs + o])) <= 4e-6 * l1);
        }
    }
    CHECK(bad_back == 0, "hb_rot_in then hb_sum_rot N %d T %d: %d of %d not z again", N, T, bad_back, T * N);
}

// extra: the host's sums (fp32 [T, hidden], in the rotated basis) are given, and added to the weighted sum before the rotation back.
static void hb_sumrot_case(Rng& r, int hidden, int T, int k, int split, int E, bool invalid, bool with_extra = false) {
    const int P = T * k;
    const float scale = (float)(1.0 / std::sqrt((double)hidden));
    Mem pm((size_t)P * split * hidden * 4), wm((size_t)P * 4), idm((size_t)P * 8), ym((size_t)T * hidden * 2), xm((size_t)T * hidden * 4);
    auto* part = pm.as<float>();
    auto* wts = wm.as<float>();
    auto* ids = idm.as<long long>();
    auto* extra = xm.as<float>();
    const std::vector<signed char> sg = hb_signs(r, hidden);
    for (size_t i = 0; i < (size_t)P * split * hidden; i++) part[i] = hb_odd(r);
    for (size_t i = 0; i < (size_t)T * hidden; i++) extra[i] = hb_odd(r);
    for (int j = 0; j < P; j++) {
        wts[j] = r.f(0.1f, 3.f) * ((r.g() % 5 == 0) ? -1.f : 1.f);
        ids[j] = (long long)(r.g() % E);
        if (invalid && r.g() % 4 == 0) ids[j] = (r.g() & 1) ? -1 : E + (long long)(r.g() % 9);
    }
    if (invalid && P > 1) ids[1] = -1;
    launch(Dim3{(unsigned)T, 1, 1}, Dim3{(unsigned)hidden / 4, 1, 1}, hb_sum_rot, (const float*)part, (const float*)wts, (const long long*)ids, (const signed char*)sg.data(), (bf16*)ym.as<uint16_t>(), with_extra ? (const float*)extra : (const float*)nullptr, E, k, hidden, split, scale);
    int bad = 0;
    for (int t = 0; t < T; t++) {
        std::vector<double> v(hidden, 0.0);
        double mag = 0;
        if (with_extra)
            for (int o = 0; o < hidden; o++) {
                v[o] = extra[(size_t)t * hidden + o];
                mag += std::fabs((double)extra[(size_t)t * hidden + o]);
            }
        for (int c = 0; c < k; c++) {
            const int j = t * k + c;
            if (ids[j] < 0 || ids[j] >= E) continue;
            for (int o = 0; o < hidden; o++) {
                double sum = 0, ms = 0;
                for (int s = 0; s < split; s++) {
                    sum += part[((size_t)j * split + s) * hidden + o];
                    ms += std::fabs(part[((size_t)j * split + s) * hidden + o]);
                }
                v[o] += (double)wts[j] * sum;
                mag += std::fabs((double)wts[j]) * ms;
            }
        }
        wht64(v);
        for (int o = 0; o < hidden; o++) {
            const double want = sg[o] * (double)scale * v[o];
            const uint16_t got = ym.as<uint16_t>()[(size_t)t * hidden + o];
            bad += !(places(got, bits((float)want)) <= 1 || std::fabs((double)val(got) - want) <= 4e-6 * scale * mag);
        }
    }
    CHECK(bad == 0, "hb_sum_rot hidden %d T %d k %d split %d E %d invalid %d extra %d: %d of %d off", hidden, T, k, split, E, invalid, (int)with_extra, bad, T * hidden);
}

// ---- hb_up2 and hb_up3 ----------------------------------------------------------------------------------------------------------------------------
// The activation as the engine rounds it: u to bf16, max(., 0), squared, to bf16 (as bf16 bits; the order of the non-negative ones is the bits').
static uint16_t hb_act_bits(double u) {
    const float h = std::max(val(bits((float)u)), 0.f);
    return bits(h * h);
}

// k distinct experts for each of T tokens (E >= k), the router's choices; with `disjoint` (T k <= E) no expert is chosen by two tokens.
static std::vector<long long> hb_choices(Rng& r, int E, int T, int k, bool disjoint = false) {
    std::vector<long long> ids((size_t)T * k);
    std::vector<int> pool(E);
    std::iota(pool.begin(), pool.end(), 0);
    std::shuffle(pool.begin(), pool.end(), r.g);
    for (int t = 0; t < T; t++) {
        if (!disjoint) std::shuffle(pool.begin(), pool.end(), r.g);
        for (int c = 0; c < k; c++) ids[(size_t)t * k + c] = pool[disjoint ? t * k + c : c];
    }
    return ids;
}

// The first product over E experts' rows [inter, cols] in the places perm (expert e at place perm[e]) of T tokens' k choices (some routed nowhere with `nowhere`), against the products in double, with relu squared rounded as the engine
// rounds it: a result is accepted between the roundings of u - err and u + err, err = 2e-6 of the terms' magnitude (the fp32 sums are in another order than double's). The same call twice, and the same experts in other places, give the same bits.
static void hb_up_case(Rng& r, int nb, int cols, int inter, int E, int T, int k, bool nowhere) {
    const int P = T * k;
    const HbMat m = hb_random(r, nb, E, inter, cols);
    std::vector<float> zr((size_t)T * cols);
    for (auto& v : zr) v = (r.g() % 13 == 0) ? 0.f : r.n() * ((r.g() % 9 == 0) ? 6.f : 1.f);
    std::vector<long long> ids = hb_choices(r, E, T, k);
    if (nowhere && P >= 3) {
        ids[1] = -1;
        ids[P - 1] = E + 5;
    }
    std::vector<int> pa(E), pb(E);
    std::iota(pa.begin(), pa.end(), 0);
    std::shuffle(pa.begin(), pa.end(), r.g);
    pb = pa;
    std::rotate(pb.begin(), pb.begin() + 1, pb.end());  // (the experts one place on)
    int unwritten = 0, stray = 0;
    // the rows of a call: [pair j][inter] as bf16 words, 0xffff for a pair routed nowhere
    auto run = [&](const std::vector<int>& perm) {
        HbPlaces pl(m, E);
        for (int e = 0; e < E; e++) pl.put(m, e, perm[e]);
        std::vector<long long> via(P);
        for (int j = 0; j < P; j++) via[j] = (ids[j] >= 0 && ids[j] < E) ? perm[ids[j]] : -1;
        Mem planm(((size_t)2 + 2 * E + P) * 4), viam((size_t)P * 8), zm((size_t)T * cols * 4), am((size_t)P * inter * 2);
        std::memcpy(viam.p, via.data(), (size_t)P * 8);
        std::memcpy(zm.p, zr.data(), zr.size() * 4);
        std::memset(am.p, 0xff, (size_t)P * inter * 2);
        const std::vector<int> order = make_plan(via, E, planm.as<int>());
        auto kernel = nb == 2 ? hb_up2 : hb_up3;
        launch(Dim3{(unsigned)P, (unsigned)((inter + 63) / 64), 1}, Dim3{256, 1, 1}, kernel, (const float*)zm.as<float>(), (const int*)planm.as<int>(), (const long long*)viam.as<long long>(), pl.codes.as<unsigned char>(), pl.scales.as<unsigned char>(),
               pl.cb.as<unsigned char>(), pl.cs, pl.ss, pl.bs, (bf16*)am.as<uint16_t>(), E, inter, cols, k);
        std::vector<uint16_t> rows((size_t)P * inter, 0xffff);
        const uint16_t* a = am.as<uint16_t>();
        for (size_t i = 0; i < order.size(); i++)
            for (int n = 0; n < inter; n++) {
                unwritten += a[i * inter + n] == 0xffff;
                rows[(size_t)order[i] * inter + n] = a[i * inter + n];
            }
        for (size_t i = order.size(); i < (size_t)P; i++)
            for (int n = 0; n < inter; n++) stray += a[i * inter + n] != 0xffff;
        return rows;
    };
    const std::vector<uint16_t> rows1 = run(pa), rows2 = inter > 128 ? rows1 : run(pa), rows3 = run(pb);  // (the repeat on the shapes of up to 128 neurons: a launch over 2,688 takes seconds of threads)
    int bad = 0, total = 0;
    for (int j = 0; j < P; j++) {
        if (ids[j] < 0 || ids[j] >= E) continue;
        const int e = (int)ids[j], t = j / k;
        for (int n = 0; n < inter; n++) {
            double u = 0, mag = 0;
            for (int c = 0; c < cols; c++) {
                const double term = m.w(e, n, c) * (double)zr[(size_t)t * cols + c];
                u += term;
                mag += std::fabs(term);
            }
            const uint16_t got = rows1[(size_t)j * inter + n];
            bad += !(got >= hb_act_bits(u - 2e-6 * mag) && got <= hb_act_bits(u + 2e-6 * mag));
            total++;
        }
    }
    CHECK(bad == 0 && unwritten == 0 && stray == 0, "hb_up%d cols %d inter %d E %d T %d k %d nowhere %d: %d of %d off, %d rows' entries not written, %d beyond the plan written", nb, cols, inter, E, T, k, nowhere, bad, total, unwritten, stray);
    CHECK(rows1 == rows2, "hb_up%d cols %d inter %d: the same call twice gives other bits", nb, cols, inter);
    CHECK(rows1 == rows3, "hb_up%d cols %d inter %d: the same experts at other places give other bits", nb, cols, inter);
}

// ---- hb_down2 and hb_down3 ------------------------------------------------------------------------------------------------------------------------
// The second product over E experts' rows [inter neurons, hidden outputs] (the transposed down matrices) for T tokens' k choices, against the dense sum over every neuron in double: relu squared rows of `zeros` of them zero, with `special`
// a pair of all zeros (some of them negative) and one of every neuron (a subnormal, a large one) and a pair routed nowhere. `slots` places hold the experts a call needs: a call that needs more is computed in waves of at most `slots`
// experts, each wave's plan made of its pairs' places (way: -1 for a pair another wave computes), the experts in places of an order of their own, the parts of the pairs of the other waves kept.
static void hb_down_case(Rng& r, int nb, int E, int slots, int inter, int hidden, int T, int k, int split, double zeros, bool special, int least_waves) {
    const int P = T * k, slices = (hidden + 1023) / 1024;
    const HbMat m = hb_random(r, nb, E, inter, hidden);
    std::vector<long long> ids = hb_choices(r, E, T, k, least_waves > 1);  // (waves asked for: the experts of the call are as many as the choices)
    if (special && P >= 3) ids[P - 2] = -1;
    std::vector<int> need;  // the experts of the call, in the order they first appear
    int first = -1, last = -1;
    for (int j = 0; j < P; j++)
        if (ids[j] >= 0) {
            if (std::find(need.begin(), need.end(), (int)ids[j]) == need.end()) need.push_back((int)ids[j]);
            if (first < 0) first = j;
            last = j;
        }
    const int waves = ((int)need.size() + slots - 1) / slots;
    CHECK(waves >= least_waves, "hb_down%d E %d slots %d T %d k %d: %d experts in %d waves, where at least %d are meant", nb, E, slots, T, k, (int)need.size(), waves, least_waves);
    std::vector<std::vector<uint16_t>> act(P, std::vector<uint16_t>(inter));  // a pair's activations
    for (auto& row : act)
        for (auto& v : row) v = ((double)(r.g() % 100000) / 100000.0 >= zeros) ? bits_of(r.f(0.01f, 4.f)) : (uint16_t)0;
    if (special && first >= 0 && last != first) {
        for (int q = 0; q < inter; q++) {
            act[first][q] = (q % 5 == 0) ? 0x8000 : 0;
            act[last][q] = bits_of(r.f(0.01f, 4.f));
        }
        act[last][0] = 0x0001;
        act[last][inter - 1] = bits_of(3.0e3f);
    }
    HbPlaces pl(m, slots);
    Mem planm(((size_t)2 + 2 * E + P) * 4), waym((size_t)P * 8), am((size_t)P * inter * 2), pm((size_t)P * split * hidden * 4);
    auto* part = pm.as<float>();
    const size_t per_pair = (size_t)split * hidden;
    for (size_t i = 0; i < (size_t)P * per_pair; i++) part[i] = 1e30f;  // (what a pair no wave has computed yet holds)
    auto kernel = nb == 2 ? hb_down2 : hb_down3;
    int disturbed = 0;
    for (int w = 0; w < waves; w++) {
        const std::vector<int> group(need.begin() + (size_t)w * slots, need.begin() + std::min((size_t)(w + 1) * slots, need.size()));
        std::vector<int> pls(slots), place_of(E, -1);
        std::iota(pls.begin(), pls.end(), 0);
        std::shuffle(pls.begin(), pls.end(), r.g);
        for (size_t q = 0; q < group.size(); q++) {
            place_of[group[q]] = pls[q];
            pl.put(m, group[q], pls[q]);
        }
        std::vector<long long> way(P);
        for (int j = 0; j < P; j++) way[j] = ids[j] >= 0 ? place_of[ids[j]] : -1;
        std::memcpy(waym.p, way.data(), (size_t)P * 8);
        const std::vector<int> order = make_plan(way, E, planm.as<int>());
        for (size_t i = 0; i < order.size(); i++) std::memcpy(am.as<uint16_t>() + i * inter, act[order[i]].data(), (size_t)inter * 2);
        const std::vector<float> before(part, part + (size_t)P * per_pair);
        launch(Dim3{(unsigned)P, (unsigned)split, (unsigned)slices}, Dim3{256, 1, 1}, kernel, (const bf16*)am.as<uint16_t>(), (const int*)planm.as<int>(), (const long long*)waym.as<long long>(), pl.codes.as<unsigned char>(), pl.scales.as<unsigned char>(),
               pl.cb.as<unsigned char>(), pl.cs, pl.ss, pl.bs, part, E, inter, hidden, split);
        std::vector<char> planned(P, 0);
        for (int j : order) planned[j] = 1;
        for (int j = 0; j < P; j++)
            if (!planned[j]) disturbed += std::memcmp(part + j * per_pair, before.data() + j * per_pair, per_pair * 4) != 0;
    }
    int bad = 0, total = 0;
    for (int j = 0; j < P; j++) {
        if (ids[j] < 0) continue;
        const int e = (int)ids[j];
        std::vector<double> sum(hidden, 0.0), mag(hidden, 0.0);
        for (int q = 0; q < inter; q++) {
            const double av = val(act[j][q]);
            if (av == 0.0) continue;  // (zero times a weight: nothing; the dense product adds it, to no effect)
            for (int o = 0; o < hidden; o++) {
                const double term = av * m.w(e, q, o);
                sum[o] += term;
                mag[o] += std::fabs(term);
            }
        }
        for (int o = 0; o < hidden; o++) {
            double got = 0;
            for (int s = 0; s < split; s++) got += part[((size_t)j * split + s) * hidden + o];
            bad += !(std::fabs(got - sum[o]) <= 3e-6 * mag[o] + 1e-30);
            total++;
        }
    }
    CHECK(bad == 0 && disturbed == 0, "hb_down%d E %d slots %d inter %d hidden %d T %d k %d split %d zeros %.2f special %d, %d waves: %d of %d outputs off, %d pairs' parts written by a wave that does not have them", nb, E, slots, inter, hidden, T, k, split, zeros,
          special, waves, bad, total, disturbed);
}

// ---- the whole chain ------------------------------------------------------------------------------------------------------------------------------
// hb_rot_in, the router's choices (made on the host), the plan, hb_up, hb_down and hb_sum_rot, against the experts computed in the original basis in double. The stored weights w' are in the rotated basis (R = scale H diag(signs)), so
// the original basis' weights are R^T w' (formed here with the matrix itself), the expert's first product is <R^T w'up_n, z>, a_n = relu squared of its bf16 rounding (rounded to bf16), the expert's output sum_n a_n R^T w'down_n, and y the
// weighted sum over the token's choices (one routed nowhere, skipped). The kernels round and sum in fp32 in the rotated basis, so an a_n is sometimes rounded the other way (a place of a_n, a fraction of a place of y): an output is
// accepted within 3 places or 2% of its token's rms, and the tokens' rms error must be below 0.4%.
static void hb_chain_case(Rng& r, int nb, int cols, int inter, int E, int T, int k, int split) {
    const int P = T * k;
    const float scale = (float)(1.0 / std::sqrt((double)cols));
    const HbMat up = hb_random(r, nb, E, inter, cols), dn = hb_random(r, nb, E, inter, cols);
    const std::vector<signed char> sg = hb_signs(r, cols);
    Mem zm((size_t)T * cols * 2), zrm((size_t)T * cols * 4), planm(((size_t)2 + 2 * E + P) * 4), idm((size_t)P * 8), wm((size_t)P * 4), am((size_t)P * inter * 2), pm((size_t)P * split * cols * 4), ym((size_t)T * cols * 2);
    auto* z = zm.as<uint16_t>();
    auto* ids = idm.as<long long>();
    auto* wts = wm.as<float>();
    for (size_t i = 0; i < (size_t)T * cols; i++) z[i] = bits(r.n());
    launch(Dim3{(unsigned)T, 1, 1}, Dim3{(unsigned)cols / 4, 1, 1}, hb_rot_in, (const bf16*)z, cols, (const signed char*)sg.data(), zrm.as<float>(), cols, scale);
    // the router on the host: the k best of random scores, softmax weights (times 1.5), one choice then given no expert
    for (int t = 0; t < T; t++) {
        std::vector<float> s(E);
        std::vector<int> ord(E);
        for (auto& v : s) v = r.n();
        std::iota(ord.begin(), ord.end(), 0);
        std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return s[a] > s[b]; });
        float zsum = 0;
        for (int c = 0; c < k; c++) zsum += std::exp(s[ord[c]]);
        for (int c = 0; c < k; c++) {
            ids[t * k + c] = ord[c];
            wts[t * k + c] = 1.5f * std::exp(s[ord[c]]) / zsum;
        }
    }
    if (P >= 3) ids[1] = -1;
    HbPlaces pu(up, E), pd(dn, E);
    for (int e = 0; e < E; e++) {
        pu.put(up, e, e);
        pd.put(dn, e, e);
    }
    const std::vector<int> order = make_plan(std::vector<long long>(ids, ids + P), E, planm.as<int>());
    auto up_kernel = nb == 2 ? hb_up2 : hb_up3;
    auto down_kernel = nb == 2 ? hb_down2 : hb_down3;
    launch(Dim3{(unsigned)P, (unsigned)((inter + 63) / 64), 1}, Dim3{256, 1, 1}, up_kernel, (const float*)zrm.as<float>(), (const int*)planm.as<int>(), (const long long*)ids, pu.codes.as<unsigned char>(), pu.scales.as<unsigned char>(),
           pu.cb.as<unsigned char>(), pu.cs, pu.ss, pu.bs, (bf16*)am.as<uint16_t>(), E, inter, cols, k);
    launch(Dim3{(unsigned)P, (unsigned)split, (unsigned)((cols + 1023) / 1024)}, Dim3{256, 1, 1}, down_kernel, (const bf16*)am.as<uint16_t>(), (const int*)planm.as<int>(), (const long long*)ids, pd.codes.as<unsigned char>(),
           pd.scales.as<unsigned char>(), pd.cb.as<unsigned char>(), pd.cs, pd.ss, pd.bs, pm.as<float>(), E, inter, cols, split);
    launch(Dim3{(unsigned)T, 1, 1}, Dim3{(unsigned)cols / 4, 1, 1}, hb_sum_rot, (const float*)pm.as<float>(), (const float*)wts, (const long long*)ids, (const signed char*)sg.data(), (bf16*)ym.as<uint16_t>(), (const float*)nullptr, E, k, cols, split, scale);
    // R^T w' of expert e's rows, in double: (R^T w')_j = sum_o R[o][j] w'_o, R[o][j] = scale (-1)^popcount(o & j) signs[j]
    auto unrotate = [&](const HbMat& mat, int e) {
        std::vector<double> out((size_t)inter * cols);
        std::vector<double> w(cols);
        for (int n = 0; n < inter; n++) {
            for (int o = 0; o < cols; o++) w[o] = mat.w(e, n, o);
            for (int j = 0; j < cols; j++) {
                double s = 0;
                for (int o = 0; o < cols; o++) s += ((__builtin_popcount(o & j) & 1) ? -w[o] : w[o]);
                out[(size_t)n * cols + j] = (double)scale * sg[j] * s;
            }
        }
        return out;
    };
    std::vector<std::vector<double>> upo(E), dno(E);
    for (int j = 0; j < P; j++)
        if (ids[j] >= 0 && upo[ids[j]].empty()) {
            upo[ids[j]] = unrotate(up, (int)ids[j]);
            dno[ids[j]] = unrotate(dn, (int)ids[j]);
        }
    int bad = 0;
    double err2 = 0, ref2 = 0;
    for (int t = 0; t < T; t++) {
        std::vector<double> want(cols, 0.0);
        for (int c = 0; c < k; c++) {
            const int j = t * k + c;
            if (ids[j] < 0) continue;
            const std::vector<double>&uw = upo[ids[j]], &dw = dno[ids[j]];
            for (int n = 0; n < inter; n++) {
                double u = 0;
                for (int q = 0; q < cols; q++) u += uw[(size_t)n * cols + q] * val(z[(size_t)t * cols + q]);
                const double a = val(hb_act_bits(u));
                if (a == 0.0) continue;
                for (int o = 0; o < cols; o++) want[o] += (double)wts[j] * a * dw[(size_t)n * cols + o];
            }
        }
        double rms = 0;
        for (int o = 0; o < cols; o++) rms += want[o] * want[o];
        rms = std::sqrt(rms / cols);
        for (int o = 0; o < cols; o++) {
            const uint16_t got = ym.as<uint16_t>()[(size_t)t * cols + o];
            bad += !(places(got, bits((float)want[o])) <= 3 || std::fabs((double)val(got) - want[o]) <= 0.02 * rms);
            err2 += ((double)val(got) - want[o]) * ((double)val(got) - want[o]);
            ref2 += want[o] * want[o];
        }
    }
    CHECK(bad == 0 && err2 <= 1.6e-5 * ref2, "hb chain at %d bits: cols %d inter %d E %d T %d k %d split %d: %d of %d outputs off, rms error %.3g of the rms %.3g", nb, cols, inter, E, T, k, split, bad, T * cols, std::sqrt(err2 / (T * cols)),
          std::sqrt(ref2 / (T * cols)));
}

static void hb_tests(Rng& r) {
    hb_layout_literals();
    for (int g : {0, 1}) {
        hb_group_case(r, 40, 300, g);
        hb_group_case(r, 5, 1, g);
        hb_group_case(r, 512, 257, g);
        hb_group_case(r, 22, 256, g);
    }
    for (int N : {128, 256, 512, 1024}) {
        hb_rot_case(r, N, 3, 0);
        hb_rot_case(r, N, 2, 8);
    }
    for (int hidden : {128, 256, 512, 1024}) {
        hb_sumrot_case(r, hidden, 3, 22, 1, 512, false);
        hb_sumrot_case(r, hidden, 2, 22, 8, 512, true);
        hb_sumrot_case(r, hidden, 3, 1, 1, 4, true);
        hb_sumrot_case(r, hidden, 2, 5, 8, 6, true);
        hb_sumrot_case(r, hidden, 3, 22, 8, 512, true, true);
        hb_sumrot_case(r, hidden, 2, 6, 1, 64, false, true);
    }
    // the first product: the widths, a partial tile, a plan with pairs routed nowhere, tokens and places in orders of their own, and the Nemotron shape (2,688 neurons of 1,024)
    for (int nb : {2, 3}) {
        hb_up_case(r, nb, 128, 64, 5, 2, 3, false);
        hb_up_case(r, nb, 256, 96, 6, 3, 2, true);
        hb_up_case(r, nb, 256, 100, 4, 2, 3, true);
        hb_up_case(r, nb, 512, 96, 4, 2, 2, true);
        hb_up_case(r, nb, 1024, 64, 3, 2, 2, false);
        hb_up_case(r, nb, 128, 2688, 4, 2, 2, true);
        hb_up_case(r, nb, 1024, 2688, 4, 1, 3, false);
    }
    // the second product: the shapes of the row layouts' test (one pair or many, 1 to 8 blocks a pair, outputs of one run, three, 1,024, past a block's 1,024), the Nemotron shape, and over places in waves
    for (int nb : {2, 3}) {
        hb_down_case(r, nb, 6, 6, 96, 128, 1, 3, 1, 0.9, false, 1);
        hb_down_case(r, nb, 6, 6, 96, 96, 2, 3, 3, 0.9, true, 1);
        hb_down_case(r, nb, 5, 5, 32, 32, 2, 2, 2, 0.5, true, 1);
        hb_down_case(r, nb, 8, 8, 256, 64, 3, 2, 8, 0.9, true, 1);
        hb_down_case(r, nb, 4, 4, 128, 256, 2, 2, 3, 0.9, true, 1);
        hb_down_case(r, nb, 4, 4, 4096, 1056, 1, 2, 2, 0.9, true, 1);
        hb_down_case(r, nb, 3, 3, 8, 2048, 3, 2, 5, 0.5, true, 1);
        hb_down_case(r, nb, 3, 3, 4096, 32, 1, 3, 7, 0.0, false, 1);
        hb_down_case(r, nb, 2, 2, 64, 3072, 2, 2, 8, 0.7, true, 1);
        hb_down_case(r, nb, 24, 24, 2688, 1024, 1, 22, 4, 0.918, true, 1);
        hb_down_case(r, nb, 8, 8, 96, 128, 2, 3, 2, 0.9, false, 1);
        hb_down_case(r, nb, 16, 4, 96, 96, 3, 4, 3, 0.9, false, 3);
        hb_down_case(r, nb, 12, 3, 64, 64, 4, 3, 1, 0.5, false, 3);
        hb_down_case(r, nb, 10, 5, 32, 1056, 2, 4, 2, 0.9, false, 2);
        hb_down_case(r, nb, 24, 8, 2688, 1024, 1, 22, 4, 0.918, false, 3);
    }
    // the chain, at both widths, on small shapes (a partial tile of neurons, two blocks a pair)
    for (int nb : {2, 3}) {
        hb_chain_case(r, nb, 128, 64, 5, 2, 2, 1);
        hb_chain_case(r, nb, 256, 96, 6, 3, 3, 2);
    }
}

// ---- the sparse experts with some of a call's experts computed on the host (src/offload.rs, the CPU half) -------------------------------------------------------------------------
// The experts of a call that are not in the places are the host's: their pairs are -1 in `way` (the plan leaves them out), the host's weighted sums of them come up as `extra` [T, hidden] (fp32), and the sum, over
// ids = way (a pair that is -1 is no expert: its parts, which nothing wrote, are not read), adds extra before its one rounding. The host's share is one expert to all but one of the call's; with none of the pairs on
// the GPU the products are not launched at all and the sum is bf16 of extra alone.
static void hybrid_case(Rng& r, int bits, int E, int slots, int inter, int hidden, int T, int k, int split, int on_host, double zeros) {
    const int P = T * k, runs = hidden / 32;
    const int rstride = (runs * (16 + 4 * (bits - 4)) + 15) / 16 * 16;
    std::vector<int> codes((size_t)E * inter * hidden);
    std::vector<uint16_t> scales((size_t)E * inter * runs);
    for (auto& u : codes) u = (int)(r.g() % (1u << bits));
    for (auto& s : scales) s = bits_of_scale(r.f(0.002f, 0.03f), r.g() % 7 == 0);
    Mem dm((size_t)slots * inter * rstride), sm((size_t)slots * inter * runs * 2);  // (the places)
    auto put_expert = [&](int e, int place) {
        for (int n = 0; n < inter; n++) {
            const std::vector<unsigned char> row = rows_row(bits, std::vector<int>(codes.begin() + ((size_t)e * inter + n) * hidden, codes.begin() + ((size_t)e * inter + n + 1) * hidden), runs);
            std::memcpy(dm.as<unsigned char>() + ((size_t)place * inter + n) * rstride, row.data(), rstride);
        }
        std::memcpy(sm.as<unsigned char>() + (size_t)place * inter * runs * 2, scales.data() + (size_t)e * inter * runs, (size_t)inter * runs * 2);
    };
    std::vector<long long> ids(P);
    std::vector<int> need;
    for (int t = 0; t < T; t++) {
        std::vector<int> pool(E);
        std::iota(pool.begin(), pool.end(), 0);
        std::shuffle(pool.begin(), pool.end(), r.g);
        for (int j = 0; j < k; j++) {
            ids[t * k + j] = pool[j];
            if (std::find(need.begin(), need.end(), pool[j]) == need.end()) need.push_back(pool[j]);
        }
    }
    const int host_n = std::min(on_host, (int)need.size());
    CHECK((int)need.size() - host_n <= slots, "hybrid sparse experts: %d experts for %d places and %d on the host", (int)need.size(), slots, host_n);
    std::vector<int> place_of(E, -1);
    std::vector<int> pl(slots);
    std::iota(pl.begin(), pl.end(), 0);
    std::shuffle(pl.begin(), pl.end(), r.g);
    for (size_t q = host_n; q < need.size(); q++) {
        place_of[need[q]] = pl[q - host_n];
        put_expert(need[q], pl[q - host_n]);
    }
    std::vector<long long> way(P);
    for (int j = 0; j < P; j++) way[j] = place_of[ids[j]];
    std::vector<float> wts(P);
    for (auto& w : wts) w = r.f(0.1f, 3.f);
    std::vector<std::vector<uint16_t>> act(P, std::vector<uint16_t>(inter));
    for (auto& row : act)
        for (auto& v : row) v = ((double)(r.g() % 100000) / 100000.0 >= zeros) ? bits_of(r.f(0.01f, 4.f)) : (uint16_t)0;
    // every pair's sum over its neurons in double, and the host's tokens' weighted sums of the pairs it computes, as fp32
    std::vector<std::vector<double>> pair_sum(P), pair_mag(P);
    for (int j = 0; j < P; j++) {
        pair_sum[j].assign(hidden, 0.0);
        pair_mag[j].assign(hidden, 0.0);
        const int e = (int)ids[j];
        for (int q = 0; q < inter; q++) {
            const double av = val(act[j][q]);
            if (av == 0.0) continue;
            for (int o = 0; o < hidden; o++) {
                const size_t at = ((size_t)e * inter + q);
                const double wv = rows_weight(bits, codes[at * hidden + o], scales[at * runs + o / 32]);
                pair_sum[j][o] += av * wv;
                pair_mag[j][o] += std::fabs(av * wv);
            }
        }
    }
    Mem xm((size_t)T * hidden * 4);
    auto* extra = xm.as<float>();
    for (int t = 0; t < T; t++)
        for (int o = 0; o < hidden; o++) {
            float acc = 0.f;
            for (int c = 0; c < k; c++)
                if (way[t * k + c] < 0) acc += (float)(pair_sum[t * k + c][o] * wts[t * k + c]);
            extra[(size_t)t * hidden + o] = acc;
        }
    Mem planm(((size_t)2 + 2 * E + P) * 4), idm((size_t)P * 8), waym((size_t)P * 8), wm((size_t)P * 4), am((size_t)P * inter * 2);
    const int slices = (hidden + 1023) / 1024;
    Mem pm((size_t)P * split * hidden * 4), ym((size_t)T * hidden * 2);
    std::memcpy(idm.p, ids.data(), (size_t)P * 8);
    std::memcpy(waym.p, way.data(), (size_t)P * 8);
    std::memcpy(wm.p, wts.data(), (size_t)P * 4);
    auto* part = pm.as<float>();
    for (size_t i = 0; i < (size_t)P * split * hidden; i++) part[i] = 1e30f;  // (what a pair no launch has computed holds: a host pair's, never read)
    const std::vector<int> order = make_plan(way, E, planm.as<int>());
    for (size_t i = 0; i < order.size(); i++) std::memcpy(am.as<uint16_t>() + i * inter, act[order[i]].data(), (size_t)inter * 2);
    auto kernel = bits == 4 ? moe_down_rows4 : bits == 5 ? moe_down_rows5 : moe_down_rows6;
    if (!order.empty())
        launch(Dim3{(unsigned)P, (unsigned)split, (unsigned)slices}, Dim3{256, 1, 1}, kernel, (const bf16*)am.as<uint16_t>(), (const int*)planm.as<int>(), (const long long*)waym.as<long long>(), dm.as<unsigned char>(),
               sm.as<unsigned char>(), (long long)inter * rstride, (long long)inter * runs * 2, part, E, inter, hidden, split);
    launch(Dim3{(unsigned)T, (unsigned)((hidden + 63) / 64), 1}, Dim3{256, 1, 1}, moe_rows_sum, (const float*)part, (const float*)wm.as<float>(), (const long long*)waym.as<long long>(), (bf16*)ym.as<uint16_t>(), E, k, hidden, split, (const float*)extra);
    int bad = 0;
    for (int t = 0; t < T; t++)
        for (int o = 0; o < hidden; o++) {
            double want = extra[(size_t)t * hidden + o], mag = std::fabs((double)extra[(size_t)t * hidden + o]);
            for (int c = 0; c < k; c++) {
                const int j = t * k + c;
                if (way[j] < 0) continue;
                want += pair_sum[j][o] * wts[j];
                mag += pair_mag[j][o] * wts[j];
            }
            const uint16_t got = ym.as<uint16_t>()[(size_t)t * hidden + o];
            bad += !(places(got, bits_of((float)want)) <= 1 || std::fabs((double)val(got) - want) <= 4e-6 * mag);
        }
    CHECK(bad == 0, "moe_rows_sum with the host's sums: %d bits, E %d slots %d inter %d hidden %d T %d k %d split %d, %d of %d pairs' experts on the host, %d of %d GPU pairs: %d outputs off", bits, E, slots, inter, hidden, T, k, split, host_n, (int)need.size(), (int)order.size(), P, bad);
}

int main(int argc, char** argv) {
    Rng r;
    if (argc > 1 && std::strcmp(argv[1], "hb") == 0) {  // (the Hummingbird kernels alone)
        hb_tests(r);
        std::printf("%s: %d checks, %d failed\n", fails ? "FAILED" : "ok", checks, fails);
        return fails ? 1 : 0;
    }
    rows_layout_literals();
    // the sparse experts: the widths, a pair or many, the outputs of one run, three and past a block's 1,024, neurons at the edges of a thread's chunks and past it, a plan with a pair routed nowhere
    for (int bits : {4, 5, 6}) {
        rows_case(r, {bits, 6, 96, 128, 1, 3, 1, 0.9, false});
        rows_case(r, {bits, 6, 96, 96, 2, 3, 3, 0.9, true});
        rows_case(r, {bits, 5, 32, 32, 2, 2, 2, 0.5, true});
        rows_case(r, {bits, 8, 256, 64, 3, 2, 8, 0.9, true});
        rows_case(r, {bits, 4, 4096, 1056, 1, 2, 2, 0.9, true});  // (inter at the most a block lists; outputs past 1,024: a second block of runs)
        rows_case(r, {bits, 24, 2688, 1024, 1, 22, 4, 0.918, true});  // (Nemotron 3 Super's expert: 22 of the layer's choices, its sparsity)
    }
    // the edges of the shapes: one chunk of activations, a full list of 4,096 neurons, outputs past one block's 1,024 (two and three blocks of runs), more blocks a pair than a warp has neurons, odd counts of blocks, and
    // the Nemotron shape at three tokens
    for (int bits : {4, 5, 6}) {
        rows_case(r, {bits, 3, 8, 2048, 3, 2, 5, 0.5, true});
        rows_case(r, {bits, 3, 4096, 32, 1, 3, 7, 0.0, false});
        rows_case(r, {bits, 2, 64, 3072, 2, 2, 8, 0.7, true});
        rows_case(r, {bits, 40, 2688, 1024, 3, 22, 3, 0.92, true});
    }
    rows_case(r, {5, 24, 2688, 1024, 2, 22, 1, 0.0, false});  // (every neuron of every pair)
    rows_case(r, {4, 24, 2688, 1024, 2, 22, 8, 1.0, false});  // (none)
    // over an expert cache's places: the experts of a call all in places (one wave, the places a permutation of the experts), the call's experts more than the places (two, three and four waves: a wave's parts survive
    // the next's, the places are refilled), places for a step's choices and no more, and the Nemotron shape of a decode step with 3 places short of the 22 choices' experts
    for (int bits : {4, 5, 6}) {
        cached_case(r, bits, 8, 8, 96, 128, 2, 3, 2, 1, 0.9);
        cached_case(r, bits, 16, 4, 96, 96, 3, 4, 3, 3, 0.9);
        cached_case(r, bits, 12, 3, 64, 64, 4, 3, 1, 3, 0.5);
        cached_case(r, bits, 10, 5, 32, 1056, 2, 4, 2, 2, 0.9);
        cached_case(r, bits, 64, 22, 2688, 1024, 1, 22, 4, 1, 0.918);
        cached_case(r, bits, 64, 8, 2688, 1024, 1, 22, 4, 3, 0.918);
    }
    // some of a call's experts on the host: the sum adds their sums before its one rounding; a call with all of them there (no product launched) and one with none (extra of zeros)
    for (int bits : {4, 5, 6}) {
        hybrid_case(r, bits, 8, 8, 96, 128, 2, 3, 2, 2, 0.9);
        hybrid_case(r, bits, 16, 4, 96, 96, 3, 4, 3, 9, 0.9);
        hybrid_case(r, bits, 12, 6, 64, 64, 4, 3, 1, 12, 0.5);
        hybrid_case(r, bits, 10, 10, 32, 1056, 2, 4, 2, 0, 0.9);
        hybrid_case(r, bits, 64, 22, 2688, 1024, 1, 22, 4, 7, 0.918);
        hybrid_case(r, bits, 64, 22, 2688, 1024, 1, 22, 4, 22, 0.918);
        hybrid_case(r, bits, 64, 22, 2688, 1024, 1, 22, 4, 21, 0.918);
    }
    // the logits: the four models' expert counts, one token and a few, rows that are not a multiple of 8, a padded x
    for (int E : {1, 5, 8, 100, 128, 256, 288, 512}) {
        logits_case(r, 1, E, 256, 256);
        logits_case(r, 11, E, 64, 80);
    }
    logits_case(r, 3, 288, 2048, 2048);
    logits_case(r, 9, 128, 704, 704);
    // the selection: softmax (Qwen3.6: 8 of 256; Gemma 4: 8 of 128 and its per-expert scale), sigmoid with the bias (GLM-5.3-Flash: 8 of 288, x2.5; Nemotron 3: 22 of 512, x5)
    const int SOFT = MOE_RENORM | MOE_BF16, SIG = MOE_SIGMOID | MOE_RENORM;
    for (bool ties : {false, true}) {
        select_case(r, {1, 256, 8, SOFT, false, false, ties, 1.f});
        select_case(r, {7, 256, 8, SOFT, false, false, ties, 1.f});
        select_case(r, {6, 128, 8, SOFT, false, true, ties, 1.f});
        select_case(r, {5, 128, 8, MOE_RENORM, false, false, ties, 1.f});  // (fp32 logits)
        select_case(r, {9, 288, 8, SIG, true, false, ties, 2.5f});
        select_case(r, {3, 512, 22, SIG, true, false, ties, 5.f});
        select_case(r, {4, 1000, 32, SIG, true, false, ties, 1.f});
        select_case(r, {4, 100, 1, SOFT, false, false, ties, 1.f});
        select_case(r, {4, 20, 20, SIG, true, false, ties, 1.f});  // k = E: every expert, by score
        select_case(r, {4, 5, 5, SOFT, false, false, ties, 1.f});
        select_case(r, {2, 1, 1, SOFT, false, false, ties, 1.f});
        select_case(r, {5, 64, 8, SIG & ~MOE_RENORM, false, false, ties, 1.f});  // (no renormalization)
        select_case(r, {5, 64, 8, 0, false, false, ties, 1.f});
    }
    // the activation: SwiGLU (plain, and GLM's clamp of 10), GELU (Gemma 4), relu squared (Nemotron 3, in place too); sizes of the models and the edges
    for (int I : {8, 64, 512, 704, 2048}) {
        act_case(r, 3, I, MOE_SILU, INFINITY, false);
        act_case(r, 3, I, MOE_SILU, 10.f, false);
        act_case(r, 2, I, MOE_GELU, INFINITY, false);
        act_case(r, 2, I, MOE_RELU2, INFINITY, false);
        act_case(r, 2, I, MOE_RELU2, INFINITY, true);
    }
    act_case(r, 2, 2688, MOE_RELU2, INFINITY, true);
    act_case(r, 5, 5376, MOE_RELU2, INFINITY, false);
    for (int H : {8, 64, 2816, 4096}) {
        add_case(r, 3, H, true);
        add_case(r, 2, H, false);
    }
    hb_tests(r);
    std::printf("%s: %d checks, %d failed\n", fails ? "FAILED" : "ok", checks, fails);
    return fails ? 1 : 0;
}
