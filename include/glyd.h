/*
 * Glyd - High-Performance SIMD Stream Codec
 * C / C++ Language Bindings
 * 
 * Hardware acceleration: AVX-512 / AVX2 / NVIDIA GPU
 * License: BUSL-1.1 (see LICENSE)
 */

#ifndef GLYD_H
#define GLYD_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Returns a null-terminated version string, e.g. "0.1.0".
 */
const char* glyd_version(void);

/**
 * Calculates a guaranteed safe upper bound for the destination buffer
 * when compressing an uncompressed source of `src_len` bytes.
 */
size_t glyd_max_compressed_len(size_t src_len);

/**
 * Compresses `src` of length `src_len` into buffer `dst` of capacity `dst_capacity`
 * using the sequential single-core engine.
 * 
 * Returns:
 *   >= 0 : Number of compressed bytes written to `dst`.
 *     -1 : Destination buffer capacity too small.
 *     -2 : Null pointer passed.
 */
int64_t glyd_compress(
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
int64_t glyd_compress_parallel(
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
int64_t glyd_compress_max(
    const uint8_t* src,
    size_t src_len,
    uint8_t* dst,
    size_t dst_capacity
);

/**
 * Compresses `src` at the ultra level: format v7 on the optimal parse,
 * denser than the max level (ratio above zstd -16 on Silesia), an order of
 * magnitude slower to produce, decoded by the same decoder.
 *
 * Returns:
 *   >= 0 : Number of compressed bytes written to `dst`.
 *     -1 : Destination buffer capacity too small.
 *     -2 : Null pointer passed.
 */
int64_t glyd_compress_ultra(
    const uint8_t* src,
    size_t src_len,
    uint8_t* dst,
    size_t dst_capacity
);

/**
 * Compresses `src` at the ultra level across all available CPU cores.
 *
 * Returns:
 *   >= 0 : Number of compressed bytes written to `dst`.
 *     -1 : Destination buffer capacity too small.
 *     -2 : Null pointer passed.
 */
int64_t glyd_compress_ultra_parallel(
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
int64_t glyd_compress_max_parallel(
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
int64_t glyd_decompress(
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
int64_t glyd_decompress_parallel(
    const uint8_t* src,
    size_t src_len,
    uint8_t* dst,
    size_t dst_capacity
);

/* The fast and turbo levels into a caller's buffer, as glyd_compress. */
int64_t glyd_compress_fast(const uint8_t* src, size_t src_len, uint8_t* dst, size_t dst_capacity);
int64_t glyd_compress_turbo(const uint8_t* src, size_t src_len, uint8_t* dst, size_t dst_capacity);

/* ------------------------------------------------------------------------
 * The allocating API (v0.9.3): one call per operation, the output
 * allocated by the library and released with glyd_free(ptr, len).
 * Levels by number; record mode; base mode; packs; the store. The Python
 * and Go bindings (bindings/) sit on these.
 * ---------------------------------------------------------------------- */

#define GLYD_LEVEL_DEFAULT 0
#define GLYD_LEVEL_FAST 1
#define GLYD_LEVEL_TURBO 2
#define GLYD_LEVEL_MAX 3
#define GLYD_LEVEL_ULTRA 4
#define GLYD_LEVEL_COLD 5

/* Release a buffer handed out by any function below. */
void glyd_free(uint8_t* ptr, size_t len);

/* Compress at a level (GLYD_LEVEL_*), in record mode when records is
 * nonzero, on all cores unless threads is 1. Returns 0 (out, out_len set)
 * or -1 for a bad argument. */
int glyd_compress2(const uint8_t* src, size_t src_len, int level, int records, int threads,
                   uint8_t** out, size_t* out_len);

/* Decompress any Glyd stream (levels, record mode, packs, cold): 0, -1 for
 * a bad argument, -2 for a corrupt stream. */
int glyd_decompress2(const uint8_t* src, size_t src_len, uint8_t** out, size_t* out_len);

/* The decompressed size of a stream, or -1. */
int64_t glyd_decompressed_len(const uint8_t* src, size_t src_len);

/* Base mode: src against base (ultra level when ultra is nonzero); decoding
 * needs the same base. */
int glyd_compress_with_base(const uint8_t* base, size_t base_len, const uint8_t* src, size_t src_len,
                            int ultra, uint8_t** out, size_t* out_len);
int glyd_decompress_with_base(const uint8_t* base, size_t base_len, const uint8_t* src, size_t src_len,
                              uint8_t** out, size_t* out_len);

/* Many small objects as one stream with an index (level: MAX, ULTRA or
 * COLD); one object back by its index. */
int glyd_pack(const uint8_t* const* objs, const size_t* lens, size_t n, int level,
              uint8_t** out, size_t* out_len);
int glyd_unpack_object(const uint8_t* pack, size_t pack_len, size_t i, uint8_t** out, size_t* out_len);
int64_t glyd_pack_len(const uint8_t* pack, size_t pack_len);

/* The store: objects compressed across each other. Metadata at dir; the
 * objects there too, or in s3_url (s3://bucket/prefix, through the AWS
 * CLI) when it is not NULL. These live in libglyd_store (the glyd-store
 * crate, BUSL-1.1), which also carries everything above; libglyd (the
 * glyd crate, Apache-2.0) has everything above and none of these. */
typedef struct GlydStore GlydStore;
GlydStore* glyd_store_open(const char* dir, const char* s3_url);
void glyd_store_close(GlydStore* store);
int64_t glyd_store_put(GlydStore* store, const char* name, const uint8_t* data, size_t len); /* id or -1 */
int glyd_store_get(GlydStore* store, uint32_t id, uint8_t** out, size_t* out_len);         /* 0, -1, -2 */
int64_t glyd_store_id_of(GlydStore* store, const char* name);                                /* id or -1 */
int glyd_store_delete(GlydStore* store, uint32_t id);
int64_t glyd_store_compact(GlydStore* store);                                                /* bytes freed */
int glyd_store_flush(GlydStore* store);
int glyd_store_rebase(GlydStore* store, uint32_t id);
int64_t glyd_store_verify(GlydStore* store);                                                 /* objects that failed */
int glyd_store_stats(GlydStore* store, uint64_t* raw, uint64_t* stored);
int glyd_store_set_level(GlydStore* store, int level);
int64_t glyd_store_count(GlydStore* store);

#ifdef __cplusplus
}
#endif

#endif /* GLYD_H */
