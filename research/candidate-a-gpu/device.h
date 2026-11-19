#pragma once
#include <stdint.h>
#include <stddef.h>
#include <cuda_runtime.h>

// Experimental ABI. All device pointers are live buffers from pack_rounded, with tag 3.
// The standard tag-2 fallback continues through the existing GPU API.
// Inputs/output must be aligned to 16 bytes. Counters must be zero before each product.
// variant: 0 above 64 tokens selects the block shape as the expanded representation does; 1 or 2 force one (diagnostic).
extern "C" {
int glyd_candidate_a_workspace(int64_t rows, int64_t cols, int64_t tokens, int64_t variant,
                             size_t* workspace_bytes, size_t* counter_bytes);
int glyd_candidate_a_unpack(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base,
                          const uint32_t tiers[3], int64_t rows, int64_t cols, uint16_t* output,
                          cudaStream_t stream);
int glyd_candidate_a_product(const uint8_t* data, const uint8_t* blocks, const int32_t* block_base,
                           const uint32_t tiers[3], int64_t rows, int64_t cols,
                           const uint16_t* input, int64_t tokens, int64_t variant, const uint16_t* bias,
                           uint16_t* output, void* workspace, size_t workspace_bytes,
                           int* counters, cudaStream_t stream);
}
