/* What a C user does with glyd_gpu.h alone: the arrays of a pack as a save holds them (here: raw files test_gpu.py's test_public_header_runs
 * writes), the library of the wheel; compiled, linked and run by that test:
 *   cc -O2 -I include -I CUDA/include public_header_check.c LIBRARY CUDART -Wl,-rpath,... -lm -o check;  check DIR */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime_api.h>
#include "glyd_gpu.h"

#define CUDA(x) do { cudaError_t e_ = (x); if (e_) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)
#define GLYD(x) do { int e_ = (x); if (e_) { fprintf(stderr, "%s:%d %s: %d %s\n", __FILE__, __LINE__, #x, e_, glyd_gpu_error_string(e_)); exit(1); } } while (0)
static const char* dir;

static void* load(const char* name, size_t* n) {
    char path[4096];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    size_t size = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    void* b = malloc(size);
    if (fread(b, 1, size, f) != size) { perror(path); exit(1); }
    fclose(f);
    if (n) *n = size;
    return b;
}
static void* on_device(const char* name) {  /* an array of a save, copied to the device as it is */
    size_t n;
    void* h = load(name, &n);
    void* d;
    CUDA(cudaMalloc(&d, n));
    CUDA(cudaMemcpy(d, h, n, cudaMemcpyHostToDevice));
    free(h);
    return d;
}
static float bf(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }

int main(int argc, char** argv) {
    dir = argc > 1 ? argv[1] : ".";
    long O, K;
    { char path[4096]; snprintf(path, sizeof path, "%s/meta.txt", dir); FILE* f = fopen(path, "r"); if (!f || fscanf(f, "%ld %ld", &O, &K) != 2) { perror(path); return 1; } fclose(f); }
    printf("glyd_gpu_api_version() %d, the header's %d\n", glyd_gpu_api_version(), GLYD_GPU_API_VERSION);
    if (glyd_gpu_api_version() != GLYD_GPU_API_VERSION) return 1;
    size_t wn;
    uint16_t* w = load("w.bf16", &wn);
    /* the arrays of the two packs and their parameters, as the save holds them: tiers, and hb */
    void *m_data = on_device("mma.data"), *m_blocks = on_device("mma.blocks"), *m_base = on_device("mma.block_base");
    uint32_t* tiers = load("mma.tiers", NULL);
    void *q_data = on_device("mma12.data"), *q_exc = on_device("mma12.exc"), *q_base = on_device("mma12.exc_base");
    char* hbtext = load("mma12.hb", NULL);
    uint32_t hb = (uint32_t)atoi(hbtext);
    uint16_t* out;
    CUDA(cudaMalloc((void**)&out, wn));
    for (int layout = 0; layout < 2; layout++) {
        const char* name = layout ? "mma12" : "mma";
        /* unpack: bit for bit */
        CUDA(cudaMemset(out, 0, wn));
        GLYD(layout ? glyd_gpu_mma12_unpack(q_data, q_exc, q_base, &hb, K, 0, O, out, 0, 0) : glyd_gpu_mma_unpack(m_data, m_blocks, m_base, tiers, K, 0, O, out, 0, 0));
        CUDA(cudaDeviceSynchronize());
        uint16_t* back = malloc(wn);
        CUDA(cudaMemcpy(back, out, wn, cudaMemcpyDeviceToHost));
        printf("%-5s unpack: %s\n", name, memcmp(back, w, wn) ? "DIFFERENT" : "bit for bit");
        if (memcmp(back, w, wn)) return 1;
        free(back);
        /* products: the library's own way (GLYD_GPU_AUTO), a few token counts; counters of the macro's size, zero, left zero by each call */
        static const int Ms[] = {1, 3, 40, 200};
        for (int mi = 0; mi < 4; mi++) {
            int M = Ms[mi];
            char fx[64], fy[64];
            snprintf(fx, sizeof fx, "x%d.bf16", M);
            snprintf(fy, sizeof fy, "y%d.f32", M);
            uint16_t* x = on_device(fx);
            float* ref = load(fy, NULL);
            uint16_t* y;
            CUDA(cudaMalloc((void**)&y, (size_t)M * O * 2));
            size_t bytes = 0;
            int r = layout ? glyd_gpu_mma12_linear_workspace(O, K, M, GLYD_GPU_AUTO, &bytes) : glyd_gpu_mma_linear_workspace(O, K, M, GLYD_GPU_AUTO, &bytes);
            if (r == cudaErrorNotSupported) { printf("%-5s M=%-3d: not supported here (the caller unpacks and multiplies)\n", name, M); continue; }
            GLYD(r);
            void* ws = NULL;
            if (bytes) CUDA(cudaMalloc(&ws, bytes));
            size_t ndone = (size_t)GLYD_GPU_DONE_COUNTERS(O, M);
            int* done;
            CUDA(cudaMalloc((void**)&done, ndone * 4));
            CUDA(cudaMemset(done, 0, ndone * 4));
            for (int rep = 0; rep < 2; rep++) {  /* the second call: the counters were left zero */
                CUDA(cudaMemset(y, 0xFF, (size_t)M * O * 2));
                GLYD(layout ? glyd_gpu_mma12_linear(q_data, q_exc, q_base, &hb, O, K, x, M, NULL, y, GLYD_GPU_AUTO, ws, bytes, done, 0)
                            : glyd_gpu_mma_linear(m_data, m_blocks, m_base, tiers, O, K, x, M, NULL, y, GLYD_GPU_AUTO, ws, bytes, done, 0));
                CUDA(cudaDeviceSynchronize());
                uint16_t* got = malloc((size_t)M * O * 2);
                CUDA(cudaMemcpy(got, y, (size_t)M * O * 2, cudaMemcpyDeviceToHost));
                double worst = 0, rms = 0;
                for (long i = 0; i < (long)M * O; i++) rms += (double)ref[i] * ref[i];
                rms = sqrt(rms / ((double)M * O));
                for (long i = 0; i < (long)M * O; i++) worst = fmax(worst, fabs(bf(got[i]) - ref[i]));
                printf("%-5s M=%-3d call %d: %zu workspace bytes, largest difference from fp32 %.5f (rms of the output %.4f)\n", name, M, rep + 1, bytes, worst, rms);
                if (!(worst <= 0.02 * rms + 0.02)) { printf("TOO FAR\n"); return 1; }
                free(got);
            }
            int* counters = malloc(ndone * 4);
            CUDA(cudaMemcpy(counters, done, ndone * 4, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < ndone; i++) if (counters[i]) { printf("a counter was left set\n"); return 1; }
            free(counters);
            cudaFree(done); cudaFree(y); cudaFree(x); if (ws) cudaFree(ws);
            free(ref);
        }
    }
    printf("an argument out of range: %s\n", glyd_gpu_error_string(glyd_gpu_mma_unpack(m_data, m_blocks, m_base, tiers, K, 1, 5, out, 0, 0)));
    printf("ok\n");
    return 0;
}
