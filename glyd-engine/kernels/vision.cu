// The vision tower kind (src/runtime/kinds.rs `VisionDesc`, the Qwen-VL vision transformer; src/vision.rs): the kernels beside the engine's dense products. The math: transformers'
// Qwen4ExpVisionModel (modeling_qwen4_exp.py); the rounding where it rounds (each bf16 tensor it makes), the rest in fp32. Heads of up to 80 channels (a multiple of 8), any number of them.
// Included from engine.cu after the prompt attention's helpers (cp16, ldsm4, ldsm4t, mma16816, ex2, pack_bf16).

// y [n, C] = LayerNorm(x [n, C]) w + b: the mean and the (biased) variance in fp32, eps inside the root. Block a row.
extern "C" __global__ void vis_ln(const bf16* x, const bf16* w, const bf16* b, bf16* y, int C, float eps) {
    const bf16* r = x + (size_t)blockIdx.x * C;
    float s = 0.f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) s += b2f(r[i]);
    const float mean = block_sum(s) / C;
    float q = 0.f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        const float d = b2f(r[i]) - mean;
        q += d * d;
    }
    const float rs = rsqrtf(block_sum(q) / C + eps);
    for (int i = threadIdx.x; i < C; i += blockDim.x) y[(size_t)blockIdx.x * C + i] = f2b((b2f(r[i]) - mean) * rs * b2f(w[i]) + b2f(b[i]));
}

__device__ __forceinline__ float vis_act(float x, int act) {
    if (act == 1) return 0.5f * x * (1.f + tanhf(0.7978845608028654f * (x + 0.044715f * (x * x * x))));  // gelu_pytorch_tanh
    if (act == 2) return x * 0.5f * (1.f + erff(x * 0.7071067811865476f));                               // nn.GELU()
    return x;
}

// y [rows, C] = act(bf16(y + b)) in place (act 0 none, 1 GELU tanh, 2 GELU erf); with x, x = bf16(x + bf16(y + b)) and y is left. total = rows C.
extern "C" __global__ void vis_bias(bf16* y, const bf16* b, bf16* x, int C, int act, long long total) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        const float v = b2f(f2b(b2f(y[i]) + b2f(b[i % C])));
        if (x) x[i] = f2b(b2f(x[i]) + v);
        else y[i] = f2b(vis_act(v, act));
    }
}

// x [n, C] = bf16(bf16(e + b) + bf16(pos)): the patch embedding's bias and the learned position table [side^2, C] resampled bilinearly to the patch's place (4 taps a patch: idx, wt [n, 4],
// the weighted sum in fp32).
extern "C" __global__ void vis_embed(const bf16* e, const bf16* b, const bf16* table, const int* idx, const float* wt, bf16* x, int C) {
    const int t = blockIdx.x;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float p = 0.f;
#pragma unroll
        for (int k = 0; k < 4; k++) p += b2f(table[(size_t)idx[t * 4 + k] * C + c]) * wt[t * 4 + k];
        const size_t i = (size_t)t * C + c;
        x[i] = f2b(b2f(f2b(b2f(e[i]) + b2f(b[c]))) + b2f(f2b(p)));
    }
}

#define VHD 80  // a head's D values (72 for Qwen3.8-Flash-Next's tower) padded with zeros to the mma's 5 steps of 16
#define VS 88   // a row of a tile in shared memory: 176 bytes, so that ldmatrix's eight rows fall on distinct banks

// The heads of qkv [n, 3, H, D] (bias added) into q, k, v [H, n, 80] (zeros past D): q and k turned by the 2D rotary of the patch's (row, column) (pos [n, 2]), the row on the first D / 4
// frequencies of each half and the column on the last D / 4 (rot [positions, D / 4, 2]: the cos and sin of a position on a frequency, fp32), q cos + rotate_half(q) sin in fp32, rounded once.
// Block a patch.
extern "C" __global__ void vis_rope(const bf16* qkv, const int* pos, const float* rot, bf16* q, bf16* k, bf16* v, int n, int H, int D) {
    const int t = blockIdx.x, half = D / 2, q4 = D / 4;
    for (int i = threadIdx.x; i < 3 * H * VHD; i += blockDim.x) {
        const int which = i / (H * VHD), h = (i / VHD) % H, d = i % VHD;
        bf16* out = (which == 0 ? q : which == 1 ? k : v) + ((size_t)h * n + t) * VHD + d;
        if (d >= D) {
            *out = f2b(0.f);
            continue;
        }
        const bf16* src = qkv + (size_t)t * 3 * H * D + which * H * D + h * D;
        if (which == 2) {
            *out = src[d];
            continue;
        }
        const int j = d % half;
        const float* cs = rot + ((size_t)pos[2 * t + (j < q4 ? 0 : 1)] * q4 + j % q4) * 2;
        const float x = b2f(src[d]), r = d < half ? -b2f(src[d + half]) : b2f(src[d - half]);
        *out = f2b(__fadd_rn(__fmul_rn(x, cs[0]), __fmul_rn(r, cs[1])));
    }
}

// The attention of each patch over every patch of its image (no mask): q, k, v [H, n, 80], out [n, H D] (H the grid's y). fp32 softmax and sums, the probabilities rounded to bf16 for
// the values (as FlashAttention does). A block takes 64 query rows of a head (4 warps of 16), the keys 64 at a time, K and V two tiles deep (cp.async), the scores and the output in
// registers (mma.sync m16n8k16). sl2 = the softmax's scale * log2(e). Dynamic shared memory: 320 rows of VS.
extern "C" __global__ void __launch_bounds__(128) vis_attn(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v, bf16* __restrict__ out, int n, float sl2, int D) {
    extern __shared__ __align__(128) unsigned char vis_smem[];
    bf16* sQ = reinterpret_cast<bf16*>(vis_smem);  // [64][VS]
    bf16* sK = sQ + 64 * VS;                       // [2][64][VS]
    bf16* sV = sK + 2 * 64 * VS;                   // [2][64][VS]
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, h = blockIdx.y, q0 = blockIdx.x * 64;
    const size_t hoff = (size_t)h * n * VHD;
    for (int i = tid; i < 64 * 10; i += 128) {
        const int r = i / 10, c = i % 10;
        const bool ok = q0 + r < n;
        cp16(s_addr(sQ + r * VS + c * 8), q + hoff + (size_t)(ok ? q0 + r : 0) * VHD + c * 8, ok);
    }
    auto load_kv = [&](int st, int kb) {
        for (int i = tid; i < 64 * 10; i += 128) {
            const int r = i / 10, c = i % 10, key = kb * 64 + r;
            const bool ok = key < n;
            const size_t off = hoff + (size_t)(ok ? key : 0) * VHD + c * 8;
            cp16(s_addr(sK + (st * 64 + r) * VS + c * 8), k + off, ok);
            cp16(s_addr(sV + (st * 64 + r) * VS + c * 8), v + off, ok);
        }
    };
    const int nblk = (n + 63) / 64;
    load_kv(0, 0);
    cp_commit();
    const int g = lane >> 2, t4 = lane & 3;   // the row of an accumulator and its column pair
    const int mi = lane >> 3, r8 = lane & 7;  // ldmatrix: the matrix a lane's address is for, and its row
    const int wrow = warp * 16;
    uint32_t qa[VHD / 16][4];
    float o[VHD / 8][4];
#pragma unroll
    for (int i = 0; i < VHD / 8; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
    for (int jb = 0; jb < nblk; jb++) {
        if (jb + 1 < nblk) load_kv((jb + 1) & 1, jb + 1);
        cp_commit();
        cp_wait<1>();
        __syncthreads();
        if (jb == 0) {
#pragma unroll
            for (int ks = 0; ks < VHD / 16; ks++) ldsm4(qa[ks], s_addr(sQ + (wrow + (mi & 1) * 8 + r8) * VS + (ks * 2 + (mi >> 1)) * 8));
        }
        const bf16* cK = sK + (jb & 1) * 64 * VS;
        const bf16* cV = sV + (jb & 1) * 64 * VS;
        float s[8][4];
#pragma unroll
        for (int i = 0; i < 8; i++) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f;
#pragma unroll
        for (int ks = 0; ks < VHD / 16; ks++) {
#pragma unroll
            for (int nt2 = 0; nt2 < 4; nt2++) {
                uint32_t b[4];
                ldsm4(b, s_addr(cK + (nt2 * 16 + (mi >> 1) * 8 + r8) * VS + (ks * 2 + (mi & 1)) * 8));
                mma16816(s[nt2 * 2], qa[ks], b[0], b[1]);
                mma16816(s[nt2 * 2 + 1], qa[ks], b[2], b[3]);
            }
        }
        // the scale, the keys past the image's last masked, the new row maxima
        const bool tail = jb * 64 + 64 > n;
        float mx[2] = {m[0], m[1]};
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                float val = s[nt][e] * sl2;
                if (tail && jb * 64 + nt * 8 + 2 * t4 + (e & 1) >= n) val = -INFINITY;
                s[nt][e] = val;
                mx[e >> 1] = fmaxf(mx[e >> 1], val);
            }
        }
#pragma unroll
        for (int i = 0; i < 2; i++) {
            mx[i] = fmaxf(mx[i], __shfl_xor_sync(0xffffffffu, mx[i], 1));
            mx[i] = fmaxf(mx[i], __shfl_xor_sync(0xffffffffu, mx[i], 2));
        }
        float mm[2], corr[2];
#pragma unroll
        for (int i = 0; i < 2; i++) {
            mm[i] = mx[i] == -INFINITY ? 0.f : mx[i];
            corr[i] = ex2(m[i] - mm[i]);
            m[i] = mx[i];
            l[i] *= corr[i];
        }
#pragma unroll
        for (int i = 0; i < VHD / 8; i++) {
            o[i][0] *= corr[0];
            o[i][1] *= corr[0];
            o[i][2] *= corr[1];
            o[i][3] *= corr[1];
        }
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const float p = ex2(s[nt][e] - mm[e >> 1]);
                s[nt][e] = p;
                l[e >> 1] += p;
            }
        }
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            uint32_t pa[4] = {pack_bf16(s[2 * kk][0], s[2 * kk][1]), pack_bf16(s[2 * kk][2], s[2 * kk][3]), pack_bf16(s[2 * kk + 1][0], s[2 * kk + 1][1]), pack_bf16(s[2 * kk + 1][2], s[2 * kk + 1][3])};
#pragma unroll
            for (int nt2 = 0; nt2 < VHD / 16; nt2++) {
                uint32_t b[4];
                ldsm4t(b, s_addr(cV + (kk * 16 + (mi & 1) * 8 + r8) * VS + (nt2 * 2 + (mi >> 1)) * 8));
                mma16816(o[nt2 * 2], pa, b[0], b[1]);
                mma16816(o[nt2 * 2 + 1], pa, b[2], b[3]);
            }
        }
        __syncthreads();
    }
    float inv[2];
#pragma unroll
    for (int i = 0; i < 2; i++) {
        float t = l[i];
        t += __shfl_xor_sync(0xffffffffu, t, 1);
        t += __shfl_xor_sync(0xffffffffu, t, 2);
        inv[i] = 1.f / t;
    }
    const int row0 = q0 + wrow + g, row1 = row0 + 8;
    const size_t os = (size_t)gridDim.y * D;
#pragma unroll
    for (int nt = 0; nt < VHD / 8; nt++) {  // (the D values; the padding's are not written)
        if (nt * 8 >= D) break;
        const int col = h * D + nt * 8 + 2 * t4;
        if (row0 < n) *reinterpret_cast<uint32_t*>(out + (size_t)row0 * os + col) = pack_bf16(o[nt][0] * inv[0], o[nt][1] * inv[0]);
        if (row1 < n) *reinterpret_cast<uint32_t*>(out + (size_t)row1 * os + col) = pack_bf16(o[nt][2] * inv[1], o[nt][3] * inv[1]);
    }
}

// dst [rows, cd] = src [rows, cs] with zeros past cs (cd >= cs): a matrix's columns padded to the products' step.
extern "C" __global__ void vis_pad(const bf16* src, bf16* dst, int cs, int cd, long long total) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        const long long r = i / cd;
        const int c = (int)(i % cd);
        dst[i] = c < cs ? src[r * cs + c] : f2b(0.f);
    }
}

#undef VHD
#undef VS
