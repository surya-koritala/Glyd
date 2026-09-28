/*
 * A matrix of a model saved by glyd.save_pretrained (glyd-v1: safetensors and
 * glyd.json) decoded on the GPU by the library's C API, and checked against
 * the bf16 checkpoint it was packed from, bit for bit. No Python.
 *
 *   unpack GLYD_DIR BF16_DIR [PACK]
 *
 * GLYD_DIR: the saved model; BF16_DIR: its source checkpoint's directory
 * (model.safetensors, or shards and model.safetensors.index.json); PACK: a
 * pack's name in glyd.json (default: the first). A pack of merged Linears
 * (q, k, v; gate, up) is checked tensor by tensor. Built with the library's
 * release files in LIB (a glyd-gpu download, this file among them: its README
 * has the line for it; or gpu/ and the library built there) and the CUDA
 * toolkit's headers and runtime in CUDA (/usr/local/cuda):
 *
 *   gcc -O2 -I LIB -I CUDA/include unpack.c -o unpack \
 *       -L LIB -lglyd_gpu_cuda13 -L CUDA/lib64 -lcudart -Wl,-rpath,LIB:CUDA/lib64
 *
 * License: BUSL-1.1 (gpu/LICENSE).
 */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime_api.h>
#include "glyd_gpu.h"

static void fail(const char* what, const char* why, ...) {
    va_list a;
    va_start(a, why);
    fprintf(stderr, "unpack: %s: ", what);
    vfprintf(stderr, why, a);
    fputc('\n', stderr);
    va_end(a);
    exit(1);
}

#define CUDA(call) do { cudaError_t e_ = (call); if (e_) fail(#call, "%s", cudaGetErrorString(e_)); } while (0)

/* A file's bytes. */
static int64_t size_of(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f || fseeko(f, 0, SEEK_END)) fail(path, "%s", strerror(errno));
    int64_t n = (int64_t)ftello(f);
    fclose(f);
    return n;
}

/* Bytes [at, at + n) of a file (n < 0: to its end), NUL-terminated. */
static char* read_at(const char* path, int64_t at, int64_t n) {
    if (n < 0) n = size_of(path) - at;
    FILE* f = fopen(path, "rb");
    if (!f) fail(path, "%s", strerror(errno));
    char* b = n >= 0 ? malloc((size_t)n + 1) : NULL;
    if (!b || fseeko(f, at, SEEK_SET) || fread(b, 1, (size_t)n, f) != (size_t)n)
        fail(path, "cannot read %lld bytes at %lld", (long long)n, (long long)at);
    fclose(f);
    b[n] = 0;
    return b;
}

/* JSON, as far as glyd.json and a safetensors header need: objects, arrays, strings without escapes in the keys
 * looked for, numbers. */
static const char* ws(const char* p) {
    while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') p++;
    return p;
}

/* Past the value at p. */
static const char* skip(const char* p) {
    p = ws(p);
    if (*p == '"') {
        for (p++; *p && *p != '"'; p++)
            if (*p == '\\' && p[1]) p++;
        return *p ? p + 1 : p;
    }
    if (*p == '{' || *p == '[') {
        int depth = 0;
        do {
            if (*p == '"') {
                p = skip(p);
                continue;
            }
            depth += (*p == '{' || *p == '[') - (*p == '}' || *p == ']');
            p++;
        } while (*p && depth);
        return p;
    }
    while (*p && *p != ',' && *p != '}' && *p != ']') p++;
    return p;
}

/* The value of key in the object at p, or NULL. */
static const char* member(const char* p, const char* key) {
    if (!p || *(p = ws(p)) != '{') return NULL;
    size_t n = strlen(key);
    for (p = ws(p + 1); *p == '"'; p = ws(p + 1)) {
        const char* k = p + 1;
        const char* v = ws(skip(p));
        if (*v != ':') return NULL;
        v = ws(v + 1);
        if (!strncmp(k, key, n) && k[n] == '"') return v;
        p = ws(skip(v));
        if (*p != ',') return NULL;
    }
    return NULL;
}

/* The string at p into s (cap bytes): 0 where it is none or too long. */
static int string(const char* p, char* s, size_t cap) {
    if (!p || *(p = ws(p)) != '"') return 0;
    const char* e = strchr(p + 1, '"');
    if (!e || (size_t)(e - p) > cap) return 0;
    memcpy(s, p + 1, e - p - 1);
    s[e - p - 1] = 0;
    return 1;
}

/* The numbers of the array at p into v: how many, or -1 where it is none or holds more than most. */
static int numbers(const char* p, uint64_t* v, int most) {
    if (!p || *(p = ws(p)) != '[') return -1;
    int n = 0;
    for (p = ws(p + 1); *p != ']' && n < most; n++) {
        char* e;
        v[n] = strtoull(p, &e, 10);
        if (e == p) return -1;
        p = ws(e);
        if (*p == ',') p = ws(p + 1);
    }
    return *p == ']' ? n : -1;
}

/* Where tensor name of the checkpoint in dir is: its file (model.safetensors, or the shard its index names), where
 * its bytes start there, how many, its shape. Every number taken from the file checked before any use: its header
 * within the file (and safetensors' 100 MB), its dtype the one asked for (this example's: U8, I32 or BF16), its
 * data_offsets in order and within the file's data, their bytes as many as its dtype and shape take, no product
 * past 64 bits. */
typedef struct {
    char file[4096];
    int64_t at, n, shape[8];
    int dims;
} Where;

static Where find(const char* dir, const char* name, const char* dtype) {
    Where w;
    char index[4096], shard[1024], got[16];
    snprintf(index, sizeof index, "%s/model.safetensors.index.json", dir);
    FILE* f = fopen(index, "rb");
    if (f) {
        fclose(f);
        char* j = read_at(index, 0, -1);
        if (!string(member(member(j, "weight_map"), name), shard, sizeof shard) || !*shard || strchr(shard, '/'))
            fail(name, "not in %s (or its shard not a file of that directory)", index);
        snprintf(w.file, sizeof w.file, "%s/%s", dir, shard);
        free(j);
    } else {
        snprintf(w.file, sizeof w.file, "%s/model.safetensors", dir);
    }
    int64_t size = size_of(w.file);
    if (size < 8) fail(w.file, "%lld bytes: not a safetensors file", (long long)size);
    char* h8 = read_at(w.file, 0, 8);
    uint64_t len = 0;
    for (int i = 7; i >= 0; i--) len = len << 8 | (uint8_t)h8[i];
    free(h8);
    if (len > 100000000 || len > (uint64_t)size - 8)
        fail(w.file, "a header of %llu bytes, past its %lld (or safetensors' 100 MB)", (unsigned long long)len,
             (long long)size);
    char* h = read_at(w.file, 8, (int64_t)len);
    const char* t = member(h, name);
    uint64_t off[2], dim[8];
    if (!t) fail(name, "not in %s", w.file);
    w.dims = numbers(member(t, "shape"), dim, 8);
    if (w.dims < 0 || numbers(member(t, "data_offsets"), off, 2) != 2 || !string(member(t, "dtype"), got, sizeof got))
        fail(name, "no shape, data_offsets or dtype in %s's header", w.file);
    if (strcmp(got, dtype)) fail(name, "%s, where this example reads %s", got, dtype);
    uint64_t data = (uint64_t)size - 8 - len, bytes = !strcmp(dtype, "I32") ? 4 : !strcmp(dtype, "BF16") ? 2 : 1;
    if (off[0] > off[1] || off[1] > data)
        fail(name, "data_offsets [%llu, %llu], out of order or past the %llu bytes of %s's data",
             (unsigned long long)off[0], (unsigned long long)off[1], (unsigned long long)data, w.file);
    for (int i = 0; i < w.dims; i++) {
        if (dim[i] && bytes > UINT64_MAX / dim[i]) fail(name, "a shape of more than 2^64 bytes");
        bytes *= dim[i];
        w.shape[i] = dim[i] > INT64_MAX ? -1 : (int64_t)dim[i];
    }
    if (bytes != off[1] - off[0])
        fail(name, "%llu bytes, where its dtype and shape take %llu", (unsigned long long)(off[1] - off[0]),
             (unsigned long long)bytes);
    w.at = 8 + (int64_t)len + (int64_t)off[0];
    w.n = (int64_t)(off[1] - off[0]);
    free(h);
    return w;
}

/* n bytes onto the GPU. */
static void* to_gpu(const void* b, int64_t n) {
    void* d;
    CUDA(cudaMalloc(&d, (size_t)n + 1));
    CUDA(cudaMemcpy(d, b, (size_t)n, cudaMemcpyHostToDevice));
    return d;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s GLYD_DIR BF16_DIR [PACK]\n", argv[0]);
        return 2;
    }
    const char *glyd = argv[1], *orig = argv[2];
    printf("libglyd_gpu: C API %d, CUDA runtime %d\n", glyd_gpu_api_version(), glyd_gpu_cuda_version());
    if (glyd_gpu_api_version() != GLYD_GPU_API_VERSION) fail("libglyd_gpu", "its C API is not this glyd_gpu.h's");

    char path[4096], name[1024], kd[1100], kb[1100], kbb[1100];
    snprintf(path, sizeof path, "%s/glyd.json", glyd);
    char* manifest = read_at(path, 0, -1);
    const char* packs = member(manifest, "packs");
    if (argc > 3) snprintf(name, sizeof name, "%s", argv[3]);
    else if (!packs || !string(ws(packs + 1), name, sizeof name)) fail(path, "no packs");
    const char* pack = member(packs, name);
    uint64_t shape[2], t[3];
    if (!pack) fail(name, "no such pack in glyd.json");
    if (member(pack, "experts")) fail(name, "a mixture of experts' layer: this example takes a Linear's");
    if (numbers(member(pack, "shape"), shape, 2) != 2 || numbers(member(pack, "tiers"), t, 3) != 3)
        fail(name, "no shape or tiers in glyd.json");
    /* W [O, K] as the tiered layout holds it (O a multiple of 64, K of 16), up to 2^40 weights (2 TB in bf16) */
    if (shape[0] < 64 || shape[0] % 64 || shape[1] < 16 || shape[1] % 16 || shape[0] > (1ull << 40) / shape[1])
        fail(name, "[%llu, %llu] in glyd.json: not a tiered pack's shape", (unsigned long long)shape[0],
             (unsigned long long)shape[1]);
    if (t[0] > UINT32_MAX || t[1] > UINT32_MAX || t[2] > UINT32_MAX) fail(name, "tiers past 32 bits in glyd.json");
    int64_t O = (int64_t)shape[0], K = (int64_t)shape[1], n = O * K, steps = n / 1024;
    uint32_t tiers[3] = {(uint32_t)t[0], (uint32_t)t[1], (uint32_t)t[2]};

    /* The pack's buffers (the tiered layout, as glyd.save_pretrained saves every pack), checked against its shape:
     * its steps' digits and bytes, 1280 bytes a step; each step's block within blocks from block_base on (the kernel
     * reads 128 bytes before a block, 256 past the last); then onto the GPU, decoded there. */
    snprintf(kd, sizeof kd, "%s.glyd_data", name);
    snprintf(kb, sizeof kb, "%s.glyd_blocks", name);
    snprintf(kbb, sizeof kbb, "%s.glyd_block_base", name);
    Where wd = find(glyd, kd, "U8"), wb = find(glyd, kb, "U8"), wbb = find(glyd, kbb, "I32");
    if (wd.n != steps * 1280 || wbb.n != (steps + 1) * 4)
        fail(name, "data of %lld bytes, block_base of %lld: not its shape's", (long long)wd.n, (long long)wbb.n);
    char *hd = read_at(wd.file, wd.at, wd.n), *hb = read_at(wb.file, wb.at, wb.n);
    int32_t* base = (int32_t*)read_at(wbb.file, wbb.at, wbb.n);
    for (int64_t s = 0; s <= steps; s++)
        if (base[s] < 128 || (s && base[s] < base[s - 1]) || (int64_t)base[s] + 256 > wb.n)
            fail(name, "block_base[%lld] = %d: its blocks (%lld bytes) do not hold it", (long long)s, base[s],
                 (long long)wb.n);
    void *data = to_gpu(hd, wd.n), *blocks = to_gpu(hb, wb.n), *block_base = to_gpu(base, wbb.n);
    free(hd);
    free(hb);
    free(base);
    uint16_t* out;
    CUDA(cudaMalloc((void**)&out, (size_t)n * 2));
    int r = glyd_gpu_mma_unpack(data, blocks, block_base, tiers, K, 0, O, out, 0, 0);
    if (r) fail("glyd_gpu_mma_unpack", "%s", glyd_gpu_error_string(r));
    CUDA(cudaDeviceSynchronize());
    uint16_t* w = malloc((size_t)n * 2);
    if (!w) fail(name, "no memory");
    CUDA(cudaMemcpy(w, out, (size_t)n * 2, cudaMemcpyDeviceToHost));
    printf("%s: [%lld, %lld], %.2f bits a weight packed, decoded on the GPU\n", name, (long long)O, (long long)K,
           (wd.n + wb.n + wbb.n + 12) * 8.0 / (double)n);

    /* Its tensors (a merged pack's Linears, their rows in turn) against the checkpoint's. */
    const char* x = member(pack, "tensors");
    int64_t row = 0, differ = 0;
    for (x = x && *x == '[' ? ws(x + 1) : ""; *x == '{';) {
        char tn[1024];
        uint64_t ts[2];
        if (!string(member(x, "name"), tn, sizeof tn) || numbers(member(x, "shape"), ts, 2) != 2)
            fail(name, "a tensor without its name or shape in glyd.json");
        if (ts[0] < 1 || ts[0] > (uint64_t)(O - row) || ts[1] != (uint64_t)K)
            fail(tn, "[%llu, %llu] in glyd.json: not the pack's rows from row %lld", (unsigned long long)ts[0],
                 (unsigned long long)ts[1], (long long)row);
        int64_t rows = (int64_t)ts[0], d = 0;
        Where o = find(orig, tn, "BF16");
        if (o.dims != 2 || o.shape[0] != rows || o.shape[1] != K)
            fail(tn, "not [%lld, %lld] in the checkpoint", (long long)rows, (long long)K);
        uint16_t* ref = (uint16_t*)read_at(o.file, o.at, o.n);
        for (int64_t i = 0; i < rows * K; i++) d += ref[i] != w[row * K + i];
        printf("  %s [%lld, %lld]: %s\n", tn, (long long)rows, (long long)K,
               d ? "differs" : "the checkpoint's, bit for bit");
        free(ref);
        row += rows;
        differ += d;
        x = ws(skip(x));
        if (*x == ',') x = ws(x + 1);
    }
    if (row != O) fail(name, "its tensors' rows are not its own");
    if (differ) printf("%lld of %lld weights differ\n", (long long)differ, (long long)n);
    CUDA(cudaFree(out));
    CUDA(cudaFree(data));
    CUDA(cudaFree(blocks));
    CUDA(cudaFree(block_base));
    free(w);
    free(manifest);
    return differ != 0;
}
