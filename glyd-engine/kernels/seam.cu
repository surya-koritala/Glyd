// R9's CUDA spike (src/runtime/cuda.rs): the dense MLP block through the layer-kind seam, in f32 and plainly: a check of the seam, not a fast path.

// y = x / sqrt(mean(x^2) + eps) w, a block a row of d
extern "C" __global__ void __launch_bounds__(256) seam_rms(const float* x, const float* w, float* y, int d, float eps) {
    const float* xr = x + (size_t)blockIdx.x * d;
    __shared__ float part[256];
    float s = 0.f;
    for (int i = threadIdx.x; i < d; i += 256) s += xr[i] * xr[i];
    part[threadIdx.x] = s;
    __syncthreads();
    for (int k = 128; k > 0; k >>= 1) {
        if (threadIdx.x < k) part[threadIdx.x] += part[threadIdx.x + k];
        __syncthreads();
    }
    const float inv = rsqrtf(part[0] / d + eps);
    for (int i = threadIdx.x; i < d; i += 256) y[(size_t)blockIdx.x * d + i] = xr[i] * inv * w[i];
}

// y[n, m] = x[n, k] W[m, k]^T, a thread an output, summed in double as the reference sums
extern "C" __global__ void __launch_bounds__(256) seam_mm(const float* x, const float* W, float* y, int n, int k, int m) {
    const long o = (long)blockIdx.x * 256 + threadIdx.x;
    if (o >= (long)n * m) return;
    const float* xr = x + (size_t)(o / m) * k;
    const float* wr = W + (size_t)(o % m) * k;
    double s = 0.0;
    for (int i = 0; i < k; i++) s += (double)xr[i] * (double)wr[i];
    y[o] = (float)s;
}

// a = silu(g) u
extern "C" __global__ void __launch_bounds__(256) seam_swiglu(const float* g, const float* u, float* a, int n) {
    const long i = (long)blockIdx.x * 256 + threadIdx.x;
    if (i < n) a[i] = g[i] / (1.f + expf(-g[i])) * u[i];
}

// x += o
extern "C" __global__ void __launch_bounds__(256) seam_add(float* x, const float* o, int n) {
    const long i = (long)blockIdx.x * 256 + threadIdx.x;
    if (i < n) x[i] += o[i];
}
