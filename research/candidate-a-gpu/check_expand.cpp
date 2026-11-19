#include "expand.h"
#include <assert.h>
#include <stdio.h>

int main() {
    for (uint32_t x = 0; x < 65536; ++x) {
        uint32_t expected = 0;
        for (unsigned i = 0; i < 4; ++i) {
            uint32_t nibble = (x >> (4*i)) & 15;
            uint32_t original = ((nibble & 14) << 4) | (nibble & 1);
            expected |= original << (8*i);
        }
        assert(candidate_a_expand4(x) == expected);
        assert(candidate_a_expand4(x | 0xabcd0000u) == expected);
    }
    puts("65536 compact significance patterns verified");
}
