#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "../include/alatirok.h"

int main(void) {
    printf("====================================================\n");
    printf("  Testing Alatirok C ABI Interface (Shared Library)\n");
    printf("====================================================\n");

    const char* version = alatirok_version();
    printf("Alatirok Version: %s\n", version);
    assert(version != NULL && strlen(version) > 0);

    // Create 1 MB test payload with structured text patterns
    const size_t test_size = 1024 * 1024;
    uint8_t* original = (uint8_t*)malloc(test_size);
    assert(original != NULL);

    for (size_t i = 0; i < test_size; ++i) {
        original[i] = (uint8_t)("The quick brown fox jumps over the lazy dog! "[i % 45]);
    }

    size_t max_comp_len = alatirok_max_compressed_len(test_size);
    printf("Uncompressed size: %zu bytes, Max compressed buffer: %zu bytes\n", test_size, max_comp_len);
    assert(max_comp_len >= test_size);

    uint8_t* compressed = (uint8_t*)malloc(max_comp_len);
    uint8_t* restored = (uint8_t*)malloc(test_size);
    assert(compressed != NULL && restored != NULL);

    // Test 1: Single-core compress
    int64_t comp_bytes = alatirok_compress(original, test_size, compressed, max_comp_len);
    printf("1. Single-core compress: written %lld bytes (ratio: %.2fx)\n", 
           (long long)comp_bytes, (double)test_size / (double)comp_bytes);
    assert(comp_bytes > 0);

    // Test 2: Single-core decompress
    memset(restored, 0, test_size);
    int64_t decomp_bytes = alatirok_decompress(compressed, (size_t)comp_bytes, restored, test_size);
    printf("2. Single-core decompress: restored %lld bytes\n", (long long)decomp_bytes);
    assert(decomp_bytes == (int64_t)test_size);
    assert(memcmp(original, restored, test_size) == 0);
    printf("   Single-core verification: PASS\n");

    // Test 3: Multi-core compress
    memset(compressed, 0, max_comp_len);
    int64_t par_comp_bytes = alatirok_compress_parallel(original, test_size, compressed, max_comp_len);
    printf("3. Multi-core compress: written %lld bytes\n", (long long)par_comp_bytes);
    assert(par_comp_bytes > 0);

    // Test 4: Multi-core decompress
    memset(restored, 0, test_size);
    int64_t par_decomp_bytes = alatirok_decompress_parallel(compressed, (size_t)par_comp_bytes, restored, test_size);
    printf("4. Multi-core decompress: restored %lld bytes\n", (long long)par_decomp_bytes);
    fflush(stdout);
    assert(par_decomp_bytes == (int64_t)test_size);
    assert(memcmp(original, restored, test_size) == 0);
    printf("   Multi-core verification: PASS\n");

    // Test 5: Max level compress (format v7) + decompress round trip
    memset(compressed, 0, max_comp_len);
    int64_t max_comp_bytes = alatirok_compress_max(original, test_size, compressed, max_comp_len);
    printf("5. Max-level compress: written %lld bytes (ratio: %.2fx)\n",
           (long long)max_comp_bytes, (double)test_size / (double)max_comp_bytes);
    assert(max_comp_bytes > 0);

    memset(restored, 0, test_size);
    int64_t max_decomp_bytes = alatirok_decompress(compressed, (size_t)max_comp_bytes, restored, test_size);
    printf("   Max-level decompress: restored %lld bytes\n", (long long)max_decomp_bytes);
    assert(max_decomp_bytes == (int64_t)test_size);
    assert(memcmp(original, restored, test_size) == 0);
    printf("   Max-level verification: PASS\n");

    // Test 5b: Max level, parallel compress + decompress round trip
    memset(compressed, 0, max_comp_len);
    int64_t max_par_comp_bytes = alatirok_compress_max_parallel(original, test_size, compressed, max_comp_len);
    printf("5b. Max-level parallel compress: written %lld bytes\n", (long long)max_par_comp_bytes);
    assert(max_par_comp_bytes > 0);

    memset(restored, 0, test_size);
    int64_t max_par_decomp_bytes = alatirok_decompress_parallel(compressed, (size_t)max_par_comp_bytes, restored, test_size);
    printf("    Max-level parallel decompress: restored %lld bytes\n", (long long)max_par_decomp_bytes);
    assert(max_par_decomp_bytes == (int64_t)test_size);
    assert(memcmp(original, restored, test_size) == 0);
    printf("    Max-level parallel verification: PASS\n");

    // Test 7: Error handling with undersized destination
    int64_t err_comp = alatirok_compress(original, test_size, compressed, 10);
    printf("7. Undersized buffer test: error code %lld\n", (long long)err_comp);
    assert(err_comp == -1);

    // Test 8: Null pointer safety
    int64_t err_null = alatirok_compress(NULL, test_size, compressed, max_comp_len);
    printf("8. Null pointer safety test: error code %lld\n", (long long)err_null);
    assert(err_null == -2);

    free(original);
    free(compressed);
    free(restored);

    printf("====================================================\n");
    printf("  All C ABI Tests Passed 100%% Cleanly!\n");
    printf("====================================================\n");
    return 0;
}
