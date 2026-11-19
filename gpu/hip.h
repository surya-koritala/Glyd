// AMD GPUs through HIP: the CUDA names the engine's kernels (glyd-engine/kernels/engine.cu, build.rs with GLYD_HIP=1) and this library
// (glyd_gpu.cu, build_lib_hip.sh) use, as HIP's, included in place of CUDA's headers when the compiler is HIP's (hipcc, or clang -x hip).
// The kernels are written for 32-lane warps (lane = threadIdx.x & 31, masks of 32 bits, shuffles over 32 lanes): RDNA 3 and 4 (gfx11,
// gfx12) run them as wave32, which the builds pin (-mno-wavefrontsize64); any other target is refused here, and a device whose waves are not
// 32 wide by the engine (dev.rs). Inline PTX has its AMD form beside it (#if defined(__HIP_PLATFORM_AMD__)).
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__GFX11__) && !defined(__GFX12__)
#error "the AMD build is for RDNA 3 and 4 (gfx11, gfx12: wave32) alone"
#endif

typedef __hip_bfloat16 __nv_bfloat16;
typedef __hip_bfloat162 __nv_bfloat162;

__device__ __forceinline__ __nv_bfloat16 __float2bfloat16_rn(float x) { return __float2bfloat16(x); }
__device__ __forceinline__ __nv_bfloat162 __floats2bfloat162_rn(float lo, float hi) { return __float22bfloat162_rn(make_float2(lo, hi)); }

// The warp functions with a mask: one wave is the warp, every lane in it (HIP's own take 64-bit masks)
#define __shfl_sync(mask, ...) __shfl(__VA_ARGS__)
#define __shfl_xor_sync(mask, ...) __shfl_xor(__VA_ARGS__)
#define __shfl_up_sync(mask, ...) __shfl_up(__VA_ARGS__)
#define __shfl_down_sync(mask, ...) __shfl_down(__VA_ARGS__)
#define __ballot_sync(mask, p) ((unsigned)__ballot(p))
#define __any_sync(mask, p) __any(p)
#define __all_sync(mask, p) __all(p)
#define __syncwarp(...) __syncwarp()
#define __reduce_add_sync(mask, v) __reduce_add_sync((unsigned long long)(mask), v)
#define __reduce_max_sync(mask, v) __reduce_max_sync((unsigned long long)(mask), v)
#define __reduce_min_sync(mask, v) __reduce_min_sync((unsigned long long)(mask), v)
#define __match_any_sync(mask, v) ((unsigned)__match_any_sync((unsigned long long)(mask), v))

// The cache-hinted loads as plain ones (and the read-only load of a bf16, which HIP's __ldg does not take)
#define __ldcs(p) (*(p))
#define __ldcg(p) (*(p))
__device__ __forceinline__ __nv_bfloat16 __ldg(const __nv_bfloat16* p) { return *p; }

#define __trap() __builtin_trap()

// The four-byte dot products (CUDA's __dp4a): the sum of the bytes' products, plus c
__device__ __forceinline__ unsigned __dp4a(unsigned a, unsigned b, unsigned c) {
#pragma unroll
    for (int i = 0; i < 32; i += 8) c += ((a >> i) & 0xFFu) * ((b >> i) & 0xFFu);
    return c;
}
__device__ __forceinline__ int __dp4a(int a, int b, int c) {
#pragma unroll
    for (int i = 0; i < 32; i += 8) c += (int)(signed char)(a >> i) * (int)(signed char)(b >> i);
    return c;
}

// Shared memory by its 32-bit address (CUDA's shared window; AMD's LDS, address space 3)
#define GLYD_LDS(T, a) (*(__attribute__((address_space(3))) T*)(uintptr_t)(uint32_t)(a))
__device__ __forceinline__ uint32_t glyd_lds_addr(const void* p) { return (uint32_t)(uintptr_t)(const __attribute__((address_space(3))) void*)p; }
#define __cvta_generic_to_shared(p) ((size_t)glyd_lds_addr(p))
// 16 bytes into shared memory at its 32-bit address (one ds_write_b128; HIP's uint4 cannot be assigned in address space 3)
__device__ __forceinline__ void glyd_lds_put16(uint32_t a, uint4 v) {
    typedef unsigned u32x4 __attribute__((ext_vector_type(4)));
    GLYD_LDS(u32x4, a) = u32x4{v.x, v.y, v.z, v.w};
}

// The tensor cores' warp operations (mma.sync, ldmatrix, movmatrix) as shuffles and FMAs
#define GLYD_LANE() __lane_id()
#define GLYD_SHFL(v, lane) __shfl(v, lane)
#include "warp_emu.h"

// Hopper's kernel parameters (TMA's tensor maps), for the declarations of its kernels alone: on AMD their bodies trap and the host never
// launches them (glyd_gpu.cu)
#define __grid_constant__
struct CUtensorMap {
    alignas(64) unsigned long long opaque[16];
};

// The CUDA runtime's names the library's host code uses
typedef hipError_t cudaError_t;
typedef hipStream_t cudaStream_t;
typedef hipEvent_t cudaEvent_t;
typedef hipDeviceProp_t cudaDeviceProp;
typedef hipDeviceAttribute_t cudaDeviceAttr;
typedef hipFuncAttributes cudaFuncAttributes;
typedef hipStreamCaptureStatus cudaStreamCaptureStatus;
#define CUDART_VERSION HIP_VERSION
#define cudaSuccess hipSuccess
#define cudaErrorInvalidValue hipErrorInvalidValue
#define cudaErrorNotSupported hipErrorNotSupported
#define cudaErrorInvalidDevice hipErrorInvalidDevice
#define cudaErrorLaunchFailure hipErrorLaunchFailure
#define cudaErrorIllegalState hipErrorIllegalState
#define cudaGetLastError hipGetLastError
#define cudaGetErrorString hipGetErrorString
#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDevAttrComputeCapabilityMajor hipDeviceAttributeComputeCapabilityMajor
#define cudaDevAttrComputeCapabilityMinor hipDeviceAttributeComputeCapabilityMinor
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#define cudaDevAttrMaxSharedMemoryPerBlockOptin hipDeviceAttributeSharedMemPerBlockOptin
template <class F>
static inline hipError_t cudaFuncSetAttribute(F f, hipFuncAttribute a, int v) {
    return hipFuncSetAttribute((const void*)f, a, v);
}
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaFuncAttributePreferredSharedMemoryCarveout hipFuncAttributePreferredSharedMemoryCarveout
#define cudaSharedmemCarveoutMaxShared 100
#define cudaFuncGetAttributes hipFuncGetAttributes
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventRecord hipEventRecord
#define cudaEventDestroy hipEventDestroy
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamIsCapturing hipStreamIsCapturing
#define cudaStreamCaptureStatusNone hipStreamCaptureStatusNone
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
