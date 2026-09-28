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
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime_api.h>
#include "glyd_gpu.h"

static void fail(const char* what, const char* why) {
    fprintf(stderr, "unpack: %s: %s\n", what, why);
    exit(1);
}

#define CUDA(call) do { cudaError_t e_ = (call); if (e_) fail(#call, cudaGetErrorString(e_)); } while (0)

/* Bytes [at, at + n) of a file (n < 0: to its end), NUL-terminated; *got: how many. */
static char* read_at(const char* path, int64_t at, int64_t n, int64_t* got) {
    FILE* f = fopen(path, "rb");
    if (!f) fail(path, strerror(errno));
    if (n < 0) {
        if (fseeko(f, 0, SEEK_END)) fail(path, strerror(errno));
        n = (int64_t)ftello(f) - at;
    }
    char* b = malloc((size_t)n + 1);
    if (!b || fseeko(f, at, SEEK_SET) || fread(b, 1, (size_t)n, f) != (size_t)n) fail(path, "cannot read it");
    fclose(f);
    b[n] = 0;
    if (got) *got = n;
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

/* Up to most numbers of the array at p into v: how many, or -1 where it is none. */
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
    return n;
}

/* Where tensor name of the checkpoint in dir is: its file (model.safetensors, or the shard its index names), where
 * its bytes start there, how many, its dtype. */
typedef struct {
    char file[4096], dtype[16];
    int64_t at, n;
} Where;

static Where find(const char* dir, const char* name) {
    Where w;
    char index[4096], shard[1024];
    snprintf(index, sizeof index, "%s/model.safetensors.index.json", dir);
    FILE* f = fopen(index, "rb");
    if (f) {
        fclose(f);
        char* j = read_at(index, 0, -1, NULL);
        if (!string(member(member(j, "weight_map"), name), shard, sizeof shard))
            fail(name, "not in the checkpoint's index");
        snprintf(w.file, sizeof w.file, "%s/%s", dir, shard);
        free(j);
    } else {
        snprintf(w.file, sizeof w.file, "%s/model.safetensors", dir);
    }
    char* h8 = read_at(w.file, 0, 8, NULL);
    uint64_t len = 0;
    for (int i = 7; i >= 0; i--) len = len << 8 | (uint8_t)h8[i];
    free(h8);
    char* h = read_at(w.file, 8, (int64_t)len, NULL);
    const char* t = member(h, name);
    uint64_t off[2];
    if (!t || numbers(member(t, "data_offsets"), off, 2) != 2 || !string(member(t, "dtype"), w.dtype, sizeof w.dtype))
        fail(name, "not in the checkpoint");
    w.at = 8 + (int64_t)len + (int64_t)off[0];
    w.n = (int64_t)(off[1] - off[0]);
    free(h);
    return w;
}

/* Tensor name of the checkpoint in dir onto the GPU; *n: its bytes. */
static void* to_gpu(const char* dir, const char* name, int64_t* n) {
    Where w = find(dir, name);
    char* b = read_at(w.file, w.at, w.n, n);
    void* d;
    CUDA(cudaMalloc(&d, (size_t)w.n + 1));
    CUDA(cudaMemcpy(d, b, (size_t)w.n, cudaMemcpyHostToDevice));
    free(b);
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

    char path[4096], name[1024], key[1100];
    snprintf(path, sizeof path, "%s/glyd.json", glyd);
    char* manifest = read_at(path, 0, -1, NULL);
    const char* packs = member(manifest, "packs");
    if (argc > 3) snprintf(name, sizeof name, "%s", argv[3]);
    else if (!packs || !string(ws(packs + 1), name, sizeof name)) fail(path, "no packs");
    const char* pack = member(packs, name);
    uint64_t shape[2], t[3];
    if (!pack) fail(name, "no such pack in glyd.json");
    if (member(pack, "experts")) fail(name, "a mixture of experts' layer: this example takes a Linear's");
    if (numbers(member(pack, "shape"), shape, 2) != 2 || numbers(member(pack, "tiers"), t, 3) != 3)
        fail(name, "no shape or tiers");
    int64_t O = (int64_t)shape[0], K = (int64_t)shape[1], n = O * K;
    uint32_t tiers[3] = {(uint32_t)t[0], (uint32_t)t[1], (uint32_t)t[2]};

    /* The pack's buffers (the tiered layout, as glyd.save_pretrained saves every pack) onto the GPU, decoded there. */
    int64_t nd, nb, nbb;
    snprintf(key, sizeof key, "%s.glyd_data", name);
    void* data = to_gpu(glyd, key, &nd);
    snprintf(key, sizeof key, "%s.glyd_blocks", name);
    void* blocks = to_gpu(glyd, key, &nb);
    snprintf(key, sizeof key, "%s.glyd_block_base", name);
    void* block_base = to_gpu(glyd, key, &nbb);
    if (O % 64 || K % 16 || nd != n / 1024 * 1280 || nbb != (n / 1024 + 1) * 4)
        fail(name, "its buffers are not its shape's");
    uint16_t* out;
    CUDA(cudaMalloc((void**)&out, (size_t)n * 2));
    int r = glyd_gpu_mma_unpack(data, blocks, block_base, tiers, K, 0, O, out, 0, 0);
    if (r) fail("glyd_gpu_mma_unpack", glyd_gpu_error_string(r));
    CUDA(cudaDeviceSynchronize());
    uint16_t* w = malloc((size_t)n * 2);
    if (!w) fail(name, "no memory");
    CUDA(cudaMemcpy(w, out, (size_t)n * 2, cudaMemcpyDeviceToHost));
    printf("%s: [%lld, %lld], %.2f bits a weight packed, decoded on the GPU\n", name, (long long)O, (long long)K,
           (nd + nb + nbb + 12) * 8.0 / (double)n);

    /* Its tensors (a merged pack's Linears, their rows in turn) against the checkpoint's. */
    const char* x = member(pack, "tensors");
    int64_t row = 0, differ = 0;
    for (x = x && *x == '[' ? ws(x + 1) : ""; *x == '{';) {
        char tn[1024];
        uint64_t ts[2];
        if (!string(member(x, "name"), tn, sizeof tn) || numbers(member(x, "shape"), ts, 2) != 2)
            fail(name, "a tensor without its name or shape");
        Where o = find(orig, tn);
        int64_t rows = (int64_t)ts[0], d = 0;
        if (strcmp(o.dtype, "BF16") || (int64_t)ts[1] != K || o.n != rows * K * 2 || row + rows > O)
            fail(tn, "not a bf16 tensor of the pack's rows in the checkpoint");
        uint16_t* ref = (uint16_t*)read_at(o.file, o.at, o.n, NULL);
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
