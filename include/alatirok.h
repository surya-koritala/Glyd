/*
 * Alatirok - High-Performance SIMD Stream Codec
 * C / C++ Language Bindings
 * 
 * Hardware acceleration: AVX-512 / AVX2 / NVIDIA GPU
 * License: Apache-2.0 / MIT
 */

#ifndef ALATIROK_H
#define ALATIROK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Returns a null-terminated version string, e.g. "0.1.0".
 */
const char* alatirok_version(void);

/**
 * Calculates a guaranteed safe upper bound for the destination buffer
 * when compressing an uncompressed source of `src_len` bytes.
 */
size_t alatirok_max_compressed_len(size_t src_len);

/**
 * Compresses `src` of length `src_len` into buffer `dst` of capacity `dst_capacity`
 * using the sequential single-core engine.
 * 
 * Returns:
 *   >= 0 : Number of compressed bytes written to `dst`.
 *     -1 : Destination buffer capacity too small.
 *     -2 : Null pointer passed.
 */
int64_t alatirok_compress(
    const uint8_t* src,
    size_t src_len,
    uint8_t* dst,
    size_t dst_capacity
);

/**
 * Compresses `src` across all available CPU cores in parallel using Rayon.
 * 
 * Returns:
 *   >= 0 : Number of compressed bytes written to `dst`.
 *     -1 : Destination buffer capacity too small.
 *     -2 : Null pointer passed.
 */
int64_t alatirok_compress_parallel(
    const uint8_t* src,
    size_t src_len,
    uint8_t* dst,
    size_t dst_capacity
);

/**
 * Compresses `src` at the max level (format v7: entropy-coded literals and
 * sequences, ratio above zstd -3) using the sequential single-core engine.
 *
 * Returns:
 *   >= 0 : Number of compressed bytes written to `dst`.
 *     -1 : Destination buffer capacity too small.
 *     -2 : Null pointer passed.
 */
int64_t alatirok_compress_max(
    const uint8_t* src,
    size_t src_len,
    uint8_t* dst,
    size_t dst_capacity
);

/**
 * Compresses `src` at the max level (format v7) across all available CPU
 * cores in parallel using Rayon.
 *
 * Returns:
 *   >= 0 : Number of compressed bytes written to `dst`.
 *     -1 : Destination buffer capacity too small.
 *     -2 : Null pointer passed.
 */
int64_t alatirok_compress_max_parallel(
    const uint8_t* src,
    size_t src_len,
    uint8_t* dst,
    size_t dst_capacity
);

/**
 * Decompresses `src` of length `src_len` into buffer `dst` of capacity `dst_capacity`
 * using the sequential single-core engine with checksum verification.
 * 
 * Returns:
 *   >= 0 : Number of uncompressed bytes written to `dst`.
 *     -1 : Output buffer capacity too small.
 *     -2 : Corrupted bitstream or checksum mismatch.
 *     -3 : Null pointer passed.
 */
int64_t alatirok_decompress(
    const uint8_t* src,
    size_t src_len,
    uint8_t* dst,
    size_t dst_capacity
);

/**
 * Decompresses `src` in parallel across all available CPU cores using Rayon
 * with checksum verification.
 * 
 * Returns:
 *   >= 0 : Number of uncompressed bytes written to `dst`.
 *     -1 : Output buffer capacity too small.
 *     -2 : Corrupted bitstream or checksum mismatch.
 *     -3 : Null pointer passed.
 */
int64_t alatirok_decompress_parallel(
    const uint8_t* src,
    size_t src_len,
    uint8_t* dst,
    size_t dst_capacity
);

#ifdef __cplusplus
}
#endif

#endif /* ALATIROK_H */
