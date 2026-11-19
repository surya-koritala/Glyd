// The AMD build's stand-ins for the tensor cores' warp operations (gpu/warp_emu.h: mma.sync m16n8k16, ldmatrix .x4 and .x4.trans, movmatrix .trans as
// shuffles and FMAs) run on an NVIDIA GPU beside the instructions themselves: the moves bit for bit, the products against an fp64 reference within
// fp32's rounding (the tensor cores' sums held to the same). A wrong fragment place gives errors of the size of the values; rounding, of 2^-20 of them.
//   nvcc -O2 -arch=sm_89 -I gpu -o /tmp/warp_emu_test glyd-engine/tools/warp_emu_test.cu && /tmp/warp_emu_test
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>

#define GLYD_LANE() ((int)(threadIdx.x & 31))
#define GLYD_SHFL(v, lane) __shfl_sync(0xffffffffu, v, lane)
#define GLYD_LDS(T, a) (*(T*)__cvta_shared_to_generic(a))
#include "warp_emu.h"

constexpr int TRIALS = 256;  // a warp (a block) each

__device__ uint32_t pk(uint16_t lo, uint16_t hi) { return (uint32_t)lo | (uint32_t)hi << 16; }

// D = C + A B per trial: A [16][16], B [16][8] bf16 bits, C [16][8] floats; the fragments made from the matrices in mma.sync's layout.
__global__ void mma_test(const uint16_t* A, const uint16_t* B, const float* C, float* Dhw, float* Demu) {
    A += blockIdx.x * 256, B += blockIdx.x * 128, C += blockIdx.x * 128, Dhw += blockIdx.x * 128, Demu += blockIdx.x * 128;
    const int l = threadIdx.x, g = l >> 2, t = l & 3;
    const uint32_t a[4] = {pk(A[g * 16 + 2 * t], A[g * 16 + 2 * t + 1]), pk(A[(g + 8) * 16 + 2 * t], A[(g + 8) * 16 + 2 * t + 1]), pk(A[g * 16 + 2 * t + 8], A[g * 16 + 2 * t + 9]),
                           pk(A[(g + 8) * 16 + 2 * t + 8], A[(g + 8) * 16 + 2 * t + 9])};
    const uint32_t b0 = pk(B[(2 * t) * 8 + g], B[(2 * t + 1) * 8 + g]), b1 = pk(B[(2 * t + 8) * 8 + g], B[(2 * t + 9) * 8 + g]);
    const int at[4] = {g * 8 + 2 * t, g * 8 + 2 * t + 1, (g + 8) * 8 + 2 * t, (g + 8) * 8 + 2 * t + 1};
    float h[4], e[4];
    for (int i = 0; i < 4; i++) h[i] = e[i] = C[at[i]];
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(h[0]), "+f"(h[1]), "+f"(h[2]), "+f"(h[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    glyd_mma16816(e, a, b0, b1);
    for (int i = 0; i < 4; i++) Dhw[at[i]] = h[i], Demu[at[i]] = e[i];
}

// ldmatrix .x4 and .x4.trans and movmatrix .trans: 64 rows of 8 b16 in shared memory, each lane's row address a row of its trial's choosing (rows[]);
// out: [trial][kind 0-2][lane][4 words] for the hardware, then the same for the stand-ins.
__global__ void move_test(const uint16_t* tile, const int* rows, const uint32_t* words, uint32_t* hw, uint32_t* emu) {
    __shared__ __align__(16) uint16_t s[64][8];
    for (int i = threadIdx.x; i < 64 * 8; i += 32) s[i / 8][i % 8] = tile[blockIdx.x * 512 + i];
    __syncwarp();
    const int l = threadIdx.x;
    const uint32_t addr = (uint32_t)__cvta_generic_to_shared(&s[rows[blockIdx.x * 32 + l]][0]);
    uint32_t* H = hw + (blockIdx.x * 3 * 32 + l) * 4;
    uint32_t* E = emu + (blockIdx.x * 3 * 32 + l) * 4;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(H[0]), "=r"(H[1]), "=r"(H[2]), "=r"(H[3]) : "r"(addr));
    glyd_ldsm4(E, addr);
    H += 32 * 4, E += 32 * 4;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(H[0]), "=r"(H[1]), "=r"(H[2]), "=r"(H[3]) : "r"(addr));
    glyd_ldsm4t(E, addr);
    H += 32 * 4, E += 32 * 4;
    const uint32_t w = words[blockIdx.x * 32 + l];
    asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(H[0]) : "r"(w));
    E[0] = glyd_transpose8x8(w);
    H[1] = H[2] = H[3] = E[1] = E[2] = E[3] = 0;
}

static uint16_t bf16_of(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1)) >> 16);
}
static double float_of(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

template <class T>
static T* dev(const std::vector<T>& v) {
    T* p;
    cudaMalloc(&p, v.size() * sizeof(T));
    cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return p;
}
template <class T>
static std::vector<T> host(const T* p, size_t n) {
    std::vector<T> v(n);
    cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
    return v;
}

int main() {
    std::mt19937 rng(20261020);
    std::uniform_real_distribution<float> val(-4.f, 4.f);
    std::vector<uint16_t> A(TRIALS * 256), B(TRIALS * 128), tile(TRIALS * 512);
    std::vector<float> C(TRIALS * 128);
    std::vector<int> rows(TRIALS * 32);
    std::vector<uint32_t> words(TRIALS * 32);
    for (auto& x : A) x = bf16_of(val(rng));
    for (auto& x : B) x = bf16_of(val(rng));
    for (auto& x : C) x = val(rng);
    for (auto& x : tile) x = (uint16_t)rng();
    for (auto& x : rows) x = (int)(rng() % 64);
    for (auto& x : words) x = (uint32_t)rng();
    float *dhw, *demu;
    uint32_t *mhw, *memu;
    cudaMalloc(&dhw, TRIALS * 128 * 4), cudaMalloc(&demu, TRIALS * 128 * 4), cudaMalloc(&mhw, TRIALS * 3 * 32 * 16), cudaMalloc(&memu, TRIALS * 3 * 32 * 16);
    mma_test<<<TRIALS, 32>>>(dev(A), dev(B), dev(C), dhw, demu);
    move_test<<<TRIALS, 32>>>(dev(tile), dev(rows), dev(words), mhw, memu);
    if (cudaError_t e = cudaDeviceSynchronize()) {
        printf("FAIL: %s\n", cudaGetErrorString(e));
        return 1;
    }
    const auto Dhw = host(dhw, TRIALS * 128), Demu = host(demu, TRIALS * 128);
    const auto Mhw = host(mhw, TRIALS * 3 * 32 * 4), Memu = host(memu, TRIALS * 3 * 32 * 4);
    int bad = 0;
    double worst_emu = 0, worst_hw = 0;
    for (int tr = 0; tr < TRIALS; tr++)
        for (int m = 0; m < 16; m++)
            for (int n = 0; n < 8; n++) {
                double ref = C[tr * 128 + m * 8 + n], mag = std::fabs(ref);
                for (int k = 0; k < 16; k++) {
                    const double p = float_of(A[tr * 256 + m * 16 + k]) * float_of(B[tr * 128 + k * 8 + n]);
                    ref += p, mag += std::fabs(p);
                }
                const double tol = mag * std::ldexp(1.0, -20), de = std::fabs(Demu[tr * 128 + m * 8 + n] - ref) / mag, dh = std::fabs(Dhw[tr * 128 + m * 8 + n] - ref) / mag;
                worst_emu = std::fmax(worst_emu, de), worst_hw = std::fmax(worst_hw, dh);
                if (de * mag > tol && bad++ < 5) printf("FAIL mma trial %d (%d, %d): stand-in %g, reference %g\n", tr, m, n, Demu[tr * 128 + m * 8 + n], ref);
            }
    static const char* kinds[] = {"ldmatrix", "ldmatrix.trans", "movmatrix"};
    int moved = 0;
    for (size_t i = 0; i < Mhw.size(); i++) {
        if (Mhw[i] == Memu[i]) moved++;
        else if (bad++ < 10) printf("FAIL %s trial %zu lane %zu word %zu: %08x where the instruction gives %08x\n", kinds[i / 128 % 3], i / 384, i / 4 % 32, i % 4, Memu[i], Mhw[i]);
    }
    printf("mma m16n8k16: %d trials, worst error over the terms' magnitude: stand-in %.2e, tensor cores %.2e (bound 2^-20 = %.2e)\n", TRIALS, worst_emu, worst_hw, std::ldexp(1.0, -20));
    printf("ldmatrix .x4, .x4.trans, movmatrix .trans: %d of %zu words as the instructions give\n", moved, Mhw.size());
    printf(bad ? "FAILED\n" : "PASSED\n");
    return bad ? 1 : 0;
}
