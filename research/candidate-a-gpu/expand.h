#pragma once
#include <stdint.h>
#ifdef __CUDACC__
#define CA_INLINE __host__ __device__ __forceinline__
#else
#define CA_INLINE inline
#endif

// Four nibbles restore four significance bytes, with the omitted bits set to zero.
CA_INLINE uint32_t candidate_a_expand4(uint32_t value) {
    uint32_t x = value & 0xffffu;
    x = (x | (x << 8)) & 0x00ff00ffu;
    x = (x | (x << 4)) & 0x0f0f0f0fu;
    return (x & 0x01010101u) | ((x & 0x0e0e0e0eu) << 4);
}
#undef CA_INLINE
