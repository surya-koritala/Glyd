// Glyd weights on the GPU: a bf16 tensor held as its sign-and-mantissa
// bytes (8 bits a weight, as they are: noise) and its exponents coded by
// a per-tensor Huffman code (about 2.6 bits a weight where 8 are spent).
//
// Layout. The weights are cut into tiles of 32 lanes x LANE weights, in
// quads of 4 neighbouring weights; in a tile, lane l holds quads l, l +
// 32, l + 64, ... so that at every step the 32 threads of a warp read 128
// neighbouring sign-and-mantissa bytes and write 256 bytes of bf16. Each
// lane's exponents are one bit stream (LSB first), the streams back to
// back; a lane's stream starts at its bit offset (`offs`).
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>

constexpr int LANE = 512;
constexpr int TILE = 32 * LANE;

__device__ __forceinline__ uint32_t exponent_of(uint16_t v) { return (v >> 7) & 0xff; }

// Pass 1: the bits each lane's stream takes.
__global__ void lane_bits_kernel(const uint16_t* __restrict__ w, int64_t n, const uint8_t* __restrict__ len, uint32_t* __restrict__ bits, int64_t lanes) {
    int64_t g = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (g >= lanes) return;
    int64_t base = (g >> 5) * TILE + (g & 31) * 4;
    uint32_t b = 0;
    for (int j = 0; j < LANE / 4; j++) {
        int64_t q = base + (int64_t)j * 128;
        for (int k = 0; k < 4 && q + k < n; k++) b += len[exponent_of(w[q + k])];
        if (q + 4 >= n) break;
    }
    bits[g] = b;
}

// Pass 2: every lane writes its stream from its offset. A lane's first
// and last words may be shared with its neighbours (OR'd in); the words
// between are its own.
__global__ void write_kernel(const uint16_t* __restrict__ w, int64_t n, const uint8_t* __restrict__ len, const uint16_t* __restrict__ code, const int64_t* __restrict__ offs, uint32_t* __restrict__ out, int64_t lanes) {
    int64_t g = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (g >= lanes) return;
    int64_t base = (g >> 5) * TILE + (g & 31) * 4;
    uint64_t pos = (uint64_t)offs[g];
    uint64_t wi = pos >> 5;
    int nb = pos & 31;
    uint64_t acc = 0;
    bool first = true;
    for (int j = 0; j < LANE / 4; j++) {
        int64_t q = base + (int64_t)j * 128;
        for (int k = 0; k < 4 && q + k < n; k++) {
            uint32_t e = exponent_of(w[q + k]);
            acc |= (uint64_t)code[e] << nb;
            nb += len[e];
            if (nb >= 32) {
                if (first) atomicOr(&out[wi], (uint32_t)acc); else out[wi] = (uint32_t)acc;
                first = false;
                acc >>= 32;
                nb -= 32;
                wi++;
            }
        }
        if (q + 4 >= n) break;
    }
    if (nb > 0) atomicOr(&out[wi], (uint32_t)acc);
}

// Decode: the table (2^L entries, each length << 8 | exponent) in shared
// memory; every lane reads its stream through a 64-bit window.
__global__ void decode_kernel(const uint8_t* __restrict__ sm, const uint32_t* __restrict__ stream, const int64_t* __restrict__ offs, const uint16_t* __restrict__ lut, int L, int64_t n, int64_t lanes, uint16_t* __restrict__ out) {
    extern __shared__ uint16_t table[];
    for (int i = threadIdx.x; i < (1 << L); i += blockDim.x) table[i] = lut[i];
    __syncthreads();
    int64_t g = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (g >= lanes) return;
    int64_t base = (g >> 5) * TILE + (g & 31) * 4;
    uint64_t pos = (uint64_t)offs[g];
    uint64_t wi = pos >> 5;
    int sh = pos & 31;
    uint64_t buf = (((uint64_t)stream[wi + 1] << 32) | stream[wi]) >> sh;
    int nb = 64 - sh;
    wi += 2;
    const uint32_t mask = (1u << L) - 1;
    for (int j = 0; j < LANE / 4; j++) {
        int64_t q = base + (int64_t)j * 128;
        if (q >= n) break;
        uint32_t e[4];
#pragma unroll
        for (int k = 0; k < 4; k++) {
            // Two codes of at most 12 bits between refills: a refill
            // leaves at least 33 bits in the window.
            if ((k & 1) == 0 && nb <= 32) {
                buf |= (uint64_t)stream[wi++] << nb;
                nb += 32;
            }
            uint16_t t = table[buf & mask];
            e[k] = t & 0xff;
            buf >>= t >> 8;
            nb -= t >> 8;
        }
        if (q + 4 <= n) {
            uint32_t s4 = *(const uint32_t*)(sm + q);
            uint32_t o[4];
#pragma unroll
            for (int k = 0; k < 4; k++) {
                uint32_t s = (s4 >> (8 * k)) & 0xff;
                o[k] = ((s & 0x80) << 8) | (e[k] << 7) | (s & 0x7f);
            }
            *(uint2*)(out + q) = make_uint2(o[0] | (o[1] << 16), o[2] | (o[3] << 16));
        } else {
            for (int k = 0; q + k < n; k++) {
                uint32_t s = sm[q + k];
                out[q + k] = (uint16_t)(((s & 0x80) << 8) | (e[k] << 7) | (s & 0x7f));
            }
        }
    }
}

static int64_t lanes_for(int64_t n) { return (n + TILE - 1) / TILE * 32; }

torch::Tensor lane_bits(torch::Tensor w, torch::Tensor len) {
    int64_t n = w.numel(), lanes = lanes_for(n);
    auto bits = torch::empty({lanes}, w.options().dtype(torch::kInt32));
    lane_bits_kernel<<<(lanes + 255) / 256, 256>>>((const uint16_t*)w.data_ptr(), n, (const uint8_t*)len.data_ptr(), (uint32_t*)bits.data_ptr(), lanes);
    return bits;
}

void write_codes(torch::Tensor w, torch::Tensor len, torch::Tensor code, torch::Tensor offs, torch::Tensor out) {
    int64_t n = w.numel(), lanes = lanes_for(n);
    write_kernel<<<(lanes + 255) / 256, 256>>>((const uint16_t*)w.data_ptr(), n, (const uint8_t*)len.data_ptr(), (const uint16_t*)code.data_ptr(), (const int64_t*)offs.data_ptr(), (uint32_t*)out.data_ptr(), lanes);
}

void decode(torch::Tensor sm, torch::Tensor stream, torch::Tensor offs, torch::Tensor lut, int64_t L, int64_t n, torch::Tensor out) {
    TORCH_CHECK(out.numel() >= n, "the output holds fewer weights than the tensor");
    int64_t lanes = lanes_for(n);
    int threads = 128;
    decode_kernel<<<(lanes + threads - 1) / threads, threads, (1 << L) * sizeof(uint16_t)>>>((const uint8_t*)sm.data_ptr(), (const uint32_t*)stream.data_ptr(), (const int64_t*)offs.data_ptr(), (const uint16_t*)lut.data_ptr(), (int)L, n, lanes, (uint16_t*)out.data_ptr());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lane_bits", &lane_bits);
    m.def("write_codes", &write_codes);
    m.def("decode", &decode);
}
