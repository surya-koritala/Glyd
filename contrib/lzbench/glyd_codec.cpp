// Glyd in lzbench (https://github.com/inikep/lzbench): the wrapper the
// codec table calls. Levels: 1 default, 2 fast, 3 turbo, 4 --max,
// 5 --ultra. With threads > 1 the parallel entry points are used.
// Appended to bench/misc_codecs.cpp by contrib/lzbench/setup.sh.
#ifndef BENCH_REMOVE_GLYD
#include "glyd.h"

int64_t lzbench_glyd_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    const uint8_t* in = (const uint8_t*)inbuf;
    uint8_t* out = (uint8_t*)outbuf;
    int mt = codec_options->threads > 1;
    int64_t n;
    switch (codec_options->level) {
        case 2: n = glyd_compress_fast(in, insize, out, outsize); break;
        case 3: n = glyd_compress_turbo(in, insize, out, outsize); break;
        case 4: n = mt ? glyd_compress_max_parallel(in, insize, out, outsize) : glyd_compress_max(in, insize, out, outsize); break;
        case 5: n = mt ? glyd_compress_ultra_parallel(in, insize, out, outsize) : glyd_compress_ultra(in, insize, out, outsize); break;
        default: n = mt ? glyd_compress_parallel(in, insize, out, outsize) : glyd_compress(in, insize, out, outsize); break;
    }
    return n < 0 ? 0 : n;
}

int64_t lzbench_glyd_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    int64_t n = codec_options->threads > 1
        ? glyd_decompress_parallel((const uint8_t*)inbuf, insize, (uint8_t*)outbuf, outsize)
        : glyd_decompress((const uint8_t*)inbuf, insize, (uint8_t*)outbuf, outsize);
    return n < 0 ? 0 : n;
}
#endif
