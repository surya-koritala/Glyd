// The gated short convolution's kernels (src/runtime/kinds.rs `ShortConv`, LFM's conv layers; included by engine.cu after nemotron.cu; src/kern/short_conv.rs's table runs them
// for the executor's `Op::ShortConv`). The definition is the CPU reference's (src/runtime/reference/short_conv.rs): the input product's rows are B, C and x, the hidden
// width each; u = B x; each channel of u through its own causal convolution of K taps (the oldest first, the last the token's own u; no bias, no activation); y = C conv(u).
// bf16 in memory, fp32 inside: u is the fp32 product of two bf16 values (exact), the state holds it as it is, the sum is fp32, and y is rounded once.

#define SC_TAPS 16  // the most taps a channel (the lowering refuses more)

// n rows of one sequence, one after another: x [n, xs] the input product's rows (B at column c, C at ch + c, x at 2 ch + c), w [ch, K] bf16 (a channel's taps, the oldest
// first), st fp32 [ch, K - 1] the channel's last K - 1 values of u (the oldest first), read and left holding the last K - 1 of these rows; y [n, ch]. A thread a channel.
__device__ __forceinline__ void sc_conv_of(const bf16* x, int xs, const bf16* w, float* st, bf16* y, int n, int ch, int K) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch || K < 1 || K > SC_TAPS) return;
    const int H = K - 1;
    float wk[SC_TAPS], s[SC_TAPS];
    for (int j = 0; j < K; j++) wk[j] = b2f(w[(size_t)c * K + j]);
    for (int j = 0; j < H; j++) s[j] = st[(size_t)c * H + j];
    for (int t = 0; t < n; t++) {
        const bf16* row = x + (size_t)t * xs;
        const float u = b2f(row[c]) * b2f(row[2 * ch + c]);
        float acc = 0.f;
        for (int j = 0; j < H; j++) acc = fmaf(wk[j], s[j], acc);
        acc = fmaf(wk[H], u, acc);
        y[(size_t)t * ch + c] = f2b(b2f(row[ch + c]) * acc);
        for (int j = 0; j + 1 < H; j++) s[j] = s[j + 1];
        if (H > 0) s[H - 1] = u;
    }
    for (int j = 0; j < H; j++) st[(size_t)c * H + j] = s[j];
}

// (blocks of 256 threads, which the bound says: a thread's taps and history may be held in registers)
extern "C" __global__ void __launch_bounds__(256) sc_conv(const bf16* x, int xs, const bf16* w, float* st, bf16* y, int n, int ch, int K) {
    sc_conv_of(x, xs, w, st, y, n, ch, K);
}

// sc_conv of a step's rows that decode, a token of a sequence each (grid ch / 256 by rows): row r's state at sts[r ss] (a table of addresses; 0: a padding row, whose output
// is zeroed and which reads and writes no state), its input row r of x, its output row r of y. Each row's arithmetic is sc_conv's on that sequence's one token.
extern "C" __global__ void __launch_bounds__(256) sc_conv_rows(const bf16* x, int xs, const bf16* w, const unsigned long long* sts, int ss, bf16* y, int ch, int K) {
    const int r = blockIdx.y;
    float* st = reinterpret_cast<float*>(sts[(size_t)r * ss]);
    if (!st) {
        const int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c < ch) y[(size_t)r * ch + c] = f2b(0.f);
        return;
    }
    sc_conv_of(x + (size_t)r * xs, xs, w, st, y + (size_t)r * ch, 1, ch, K);
}
