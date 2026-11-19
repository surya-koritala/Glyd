// Integer instructions a weight for the decodes of the layouts in README.md: compile (no GPU needed), count SASS.
//
//   nvcc -arch=sm_100a -cubin decode_sass.cu -o decode.cubin && cuobjdump -sass decode.cubin | python sass_count.py
//
// Each kernel decodes 64 weights a thread from a bitstream in global memory with the code table in shared memory,
// fully unrolled. The count (sass_count.py) is of the ALU instructions of the unrolled body, loads, stores, barriers
// and branches left out. Straightforward code, not tuned: an estimate of the cost, not a measured kernel.
#include <cstdint>

// A prefix code of L bits at most, MSB first, in a 64-bit window; table entry = len << 8 | symbol; the window is
// refilled with a 32-bit word when it holds 32 bits or fewer, checked after each group of 4 symbols.
//   MODE 1: FP4 nibbles, a 16-symbol code (the sign inside it), packed 8 to a word as the tensor cores read them
//   MODE 2: FP8: the exponent field by the code, sign and mantissa (4 bits) from a raw plane, packed 4 bytes to a word
template <int L, int MODE>
__device__ __forceinline__ void decode64(const uint32_t* __restrict__ s, const uint32_t* __restrict__ raw, const uint16_t* lut, uint32_t* __restrict__ out) {
    uint64_t win = ((uint64_t)s[0] << 32) | s[1];
    int have = 64, nxt = 2;
    uint32_t o = 0, r = 0;
#pragma unroll
    for (int i = 0; i < 64; i++) {
        uint32_t e = lut[(uint32_t)(win >> (64 - L))];
        uint32_t len = e >> 8, sym = e & 255;
        win <<= len;
        have -= len;
        if (MODE == 1) {
            o |= sym << (4 * (i & 7));
            if ((i & 7) == 7) { out[i >> 3] = o; o = 0; }
        } else {
            if ((i & 7) == 0) r = raw[i >> 3];
            uint32_t sm = (r >> (4 * (i & 7))) & 15;
            o |= (((sm >> 3) << 7) | (sym << 3) | (sm & 7)) << (8 * (i & 3));
            if ((i & 3) == 3) { out[i >> 2] = o; o = 0; }
        }
        if ((i & 3) == 3 && have <= 32) { win |= (uint64_t)s[nxt++] << (32 - have); have += 32; }
    }
}

template <int L, int MODE>
__global__ void decode(const uint32_t* __restrict__ stream, const uint32_t* __restrict__ raw, const uint16_t* __restrict__ lut_g, uint32_t* __restrict__ out) {
    __shared__ uint16_t lut[1 << L];
    for (int i = threadIdx.x; i < (1 << L); i += blockDim.x) lut[i] = lut_g[i];
    __syncthreads();
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    decode64<L, MODE>(stream + (size_t)t * 16, raw + (size_t)t * 8, lut, out + (size_t)t * 16);
}

template __global__ void decode<6, 1>(const uint32_t*, const uint32_t*, const uint16_t*, uint32_t*);  // FP4 nibbles, codes up to 6 bits
template __global__ void decode<8, 2>(const uint32_t*, const uint32_t*, const uint16_t*, uint32_t*);  // FP8 exponent, codes up to 8 bits

// The 2-bit scale layout: 16 scales a 32-bit word as offsets 0-2 from the row's base byte; 3 escapes to a side list.
__global__ void scales(const uint32_t* __restrict__ w, const uint8_t* __restrict__ base, const uint8_t* __restrict__ esc, uint8_t* __restrict__ out) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t x = w[t];
    uint32_t b = base[t >> 3];
    int e = 0;
#pragma unroll
    for (int i = 0; i < 16; i++) {
        uint32_t o = (x >> (2 * i)) & 3;
        uint32_t v = b + o;
        if (o == 3) v = esc[e++];
        out[t * 16 + i] = v;
    }
}
