// GLM-5.3-Flash's routed experts as codebook rows of 2-bit codes (glyd-hbr-codes/2; src/moe_glm_cb.rs, research/glm53-flash-data.md). A layer's experts are held in the basis of one rotation of the hidden axis,
// R x = 4096^-0.5 H (s x) (H the natural-order Hadamard matrix, s the layer's signs); an expert is two matrices of rows of 4,096 values, each row its codes (2 bits, byte b holds codes 4 b .. 4 b + 3, the first at the
// low bits) and its own four bf16 values (ascending, absolute: a weight is the row's entry the code names): `up` (4,096 rows: gate's 2,048 and up's 2,048) and `downT` (2,048 rows: down transposed, row n the weights
// of neuron n over the 4,096 outputs). A layer's call: x' = R x once (glm_cb_rot_in); for each pair (a token and an expert in a place of the cache) the 4,096 products of up's rows with x' (glm_cb_up); the activation
// silu(min(gate, 10)) clamp(up, -10, 10) of each neuron and the sum over the neurons of activation times row n of downT, the neurons cut into S runs whose partial sums are kept (glm_cb_down); the weighted sum of a
// token's pairs, the runs added, and R^T once, rounded to bf16 (glm_cb_sum_rot). No atomics: every launch gives the same bits every time, and an expert gives the same bits in whatever place it is.

#define GC_H 4096          // the hidden width: the rows' length, the rotation's
#define GC_I 2048          // neurons an expert has
#define GC_ROW 1024        // bytes of the codes of a row
#define GC_UP_AT (2048 * 1024)      // where up's codes start in a place's codes (downT's 2,048 rows first: the order of the pack's files)
#define GC_UP_BOOK (2048 * 8)       // where up's codebooks start in a place's codebooks

// x' = R x for the n rows of x (bf16, row stride xs): xr [n, 4096] fp32, and xrb [n, 4096] bf16 where given. sg: the layer's signs as bytes (+1 or -1). A block a row, 1024 threads.
extern "C" __global__ void __launch_bounds__(1024) glm_cb_rot_in(const bf16* x, int xs, const signed char* sg, float* xr, bf16* xrb) {
    __shared__ float v[GC_H];
    const int t = blockIdx.x, tid = threadIdx.x;
    for (int i = tid; i < GC_H; i += 1024) v[i] = b2f(x[(size_t)t * xs + i]) * (float)sg[i];
    __syncthreads();
    for (int h = 1; h < GC_H; h <<= 1) {
        for (int b = tid; b < GC_H / 2; b += 1024) {
            const int i = (b / h) * 2 * h + (b % h);
            const float a = v[i], c = v[i + h];
            v[i] = a + c;
            v[i + h] = a - c;
        }
        __syncthreads();
    }
    for (int i = tid; i < GC_H; i += 1024) {
        xr[(size_t)t * GC_H + i] = v[i] * 0.015625f;
        if (xrb) xrb[(size_t)t * GC_H + i] = f2b(v[i] * 0.015625f);   // (the host's half reads x' as bf16)
    }
}

// The expert numbers of a layer in the cache's numbers: out[p] = local[ids[p]] (ids int64 [pairs], local int32 [experts], -1 for an expert the layer does not hold).
extern "C" __global__ void glm_cb_local(const long long* ids, const int* local, long long* out, int pairs, int experts) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pairs) return;
    const long long e = ids[p];
    out[p] = (e >= 0 && e < experts) ? (long long)local[e] : -1ll;
}

// One entry of a four-entry codebook by its code.
__device__ __forceinline__ float gc_pick(unsigned c, float e0, float e1, float e2, float e3) {
    const float lo = (c & 1u) ? e1 : e0, hi = (c & 1u) ? e3 : e2;
    return (c & 2u) ? hi : lo;
}

// The product of 16 codes (a 32-bit word of the row's bytes) with 16 values of x' (four float4s).
__device__ __forceinline__ float gc_dot16(unsigned w, const float4* z, float e0, float e1, float e2, float e3) {
    float acc = 0.f;
#pragma unroll
    for (int q = 0; q < 4; q++) {
        const float4 zz = z[q];
        const unsigned b = (w >> (8 * q)) & 255u;
        acc = fmaf(gc_pick(b & 3u, e0, e1, e2, e3), zz.x, acc);
        acc = fmaf(gc_pick((b >> 2) & 3u, e0, e1, e2, e3), zz.y, acc);
        acc = fmaf(gc_pick((b >> 4) & 3u, e0, e1, e2, e3), zz.z, acc);
        acc = fmaf(gc_pick(b >> 6, e0, e1, e2, e3), zz.w, acc);
    }
    return acc;
}

// gu [pairs, 4096] fp32: for each pair p with a place pl = way[p] >= 0, the 4,096 rows of the expert's `up` times x' of the pair's token (p / k). A block 32 rows of a pair (grid 128 x pairs), 8 warps of 4 rows; a lane
// 32 bytes (128 codes) of a row, x' in shared memory with a float4 of padding after each lane's 32.
extern "C" __global__ void __launch_bounds__(256) glm_cb_up(const unsigned char* codes, unsigned long long cstride, const unsigned char* books, unsigned long long bstride, const long long* way, const float* xr, float* gu, int k) {
    const int p = blockIdx.y;
    const long long pl = way[p];
    if (pl < 0) return;
    __shared__ float4 xs[32 * 33];
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const float4* xg = reinterpret_cast<const float4*>(xr + (size_t)(p / k) * GC_H);
    for (int i = tid; i < GC_H / 4; i += 256) xs[(i >> 5) * 33 + (i & 31)] = xg[i];
    __syncthreads();
    const unsigned char* cb = codes + (size_t)pl * cstride + GC_UP_AT;
    const unsigned char* bb = books + (size_t)pl * bstride + GC_UP_BOOK;
    const float4* xv = xs + lane * 33;
    for (int rr = 0; rr < 4; rr++) {
        const int row = blockIdx.x * 32 + w * 4 + rr;
        const uint4* cp = reinterpret_cast<const uint4*>(cb + (size_t)row * GC_ROW + lane * 32);
        const uint4 c0 = cp[0], c1 = cp[1];
        const uint2 bk = *reinterpret_cast<const uint2*>(bb + (size_t)row * 8);
        const float2 b01 = bf2(bk.x), b23 = bf2(bk.y);
        float acc = gc_dot16(c0.x, xv, b01.x, b01.y, b23.x, b23.y);
        acc += gc_dot16(c0.y, xv + 4, b01.x, b01.y, b23.x, b23.y);
        acc += gc_dot16(c0.z, xv + 8, b01.x, b01.y, b23.x, b23.y);
        acc += gc_dot16(c0.w, xv + 12, b01.x, b01.y, b23.x, b23.y);
        acc += gc_dot16(c1.x, xv + 16, b01.x, b01.y, b23.x, b23.y);
        acc += gc_dot16(c1.y, xv + 20, b01.x, b01.y, b23.x, b23.y);
        acc += gc_dot16(c1.z, xv + 24, b01.x, b01.y, b23.x, b23.y);
        acc += gc_dot16(c1.w, xv + 28, b01.x, b01.y, b23.x, b23.y);
        acc = warp_sum(acc);
        if (lane == 0) gu[(size_t)p * GC_H + row] = acc;
    }
}

// part [pairs, S, 4096] fp32: for each pair p with a place, the sum over the neurons n of run s (2,048 / S of them) of act[n] times row n of downT, act[n] = silu(min(gate_n, limit)) clamp(up_n, -limit, limit) from gu.
// A block (output quarter, run, pair), 256 threads: four groups of 64, a group a neuron at a time, a thread four bytes (16 outputs) of the row; the groups' sums are added through shared memory.
extern "C" __global__ void __launch_bounds__(256) glm_cb_down(const unsigned char* codes, unsigned long long cstride, const unsigned char* books, unsigned long long bstride, const long long* way, const float* gu, float* part, int S, float limit) {
    const int os = blockIdx.x, s = blockIdx.y, p = blockIdx.z;
    const long long pl = way[p];
    if (pl < 0) return;
    __shared__ float act[GC_I];
    __shared__ float red[4 * 1024];
    const int tid = threadIdx.x, g = tid >> 6, l = tid & 63;
    const int ns = GC_I / S, n0 = s * ns;
    for (int i = tid; i < ns; i += 256) {
        const float gg = fminf(gu[(size_t)p * GC_H + n0 + i], limit), uu = fminf(fmaxf(gu[(size_t)p * GC_H + GC_I + n0 + i], -limit), limit);
        act[i] = gg / (1.f + expf(-gg)) * uu;
    }
    __syncthreads();
    const unsigned char* cb = codes + (size_t)pl * cstride + (size_t)n0 * GC_ROW + os * 256 + l * 4;
    const unsigned char* bb = books + (size_t)pl * bstride + (size_t)n0 * 8;
    float acc[16];
#pragma unroll
    for (int j = 0; j < 16; j++) acc[j] = 0.f;
#pragma unroll 4
    for (int i = g; i < ns; i += 4) {
        const unsigned w = *reinterpret_cast<const unsigned*>(cb + (size_t)i * GC_ROW);
        const uint2 bk = *reinterpret_cast<const uint2*>(bb + (size_t)i * 8);
        const float2 b01 = bf2(bk.x), b23 = bf2(bk.y);
        const float a = act[i];
        const float e0 = a * b01.x, e1 = a * b01.y, e2 = a * b23.x, e3 = a * b23.y;
#pragma unroll
        for (int j = 0; j < 16; j++) acc[j] += gc_pick((w >> (2 * j)) & 3u, e0, e1, e2, e3);
    }
#pragma unroll
    for (int j = 0; j < 16; j++) red[g * 1024 + l * 16 + j] = acc[j];
    __syncthreads();
    for (int o = tid; o < 1024; o += 256) part[((size_t)p * S + s) * GC_H + os * 1024 + o] = (red[o] + red[1024 + o]) + (red[2048 + o] + red[3072 + o]);
}

// y [n, 4096] bf16 = R^T (extra[t] + sum over the token's k pairs c of wts[t k + c] times the sum over the S runs of part[(t k + c) S + s]), R^T v = sg 4096^-0.5 H v, rounded once. extra: fp32 [n, 4096] (the sums of
// pairs computed elsewhere, in the rotated basis) or 0; the pairs whose skip entry is negative (skip: int64 [pairs] or 0) are left out. A block a token, 1024 threads.
extern "C" __global__ void __launch_bounds__(1024) glm_cb_sum_rot(const float* part, const float* wts, const float* extra, const long long* skip, const signed char* sg, bf16* y, int k, int S) {
    __shared__ float v[GC_H];
    const int t = blockIdx.x, tid = threadIdx.x;
    for (int o = tid; o < GC_H; o += 1024) {
        float acc = extra ? extra[(size_t)t * GC_H + o] : 0.f;
        for (int c = 0; c < k; c++) {
            const size_t p = (size_t)t * k + c;
            if (skip && skip[p] < 0) continue;   // (a pair the host computed is in `extra`)
            float a = 0.f;
            for (int s = 0; s < S; s++) a += part[(p * S + s) * GC_H + o];
            acc = fmaf(wts[p], a, acc);
        }
        v[o] = acc;
    }
    __syncthreads();
    for (int h = 1; h < GC_H; h <<= 1) {
        for (int b = tid; b < GC_H / 2; b += 1024) {
            const int i = (b / h) * 2 * h + (b % h);
            const float a = v[i], c = v[i + h];
            v[i] = a + c;
            v[i + h] = a - c;
        }
        __syncthreads();
    }
    for (int o = tid; o < GC_H; o += 1024) y[(size_t)t * GC_H + o] = f2b(v[o] * (float)sg[o] * 0.015625f);
}

// ---- experts of the other formats, and layers of mixed ones (glyd-hbr-codes/2, glm_pack.py's `pack_rows`) --------------------------------------------------------------------------------------------------
// A row of 4,096 codes is, by format: cb2 1,024 bytes with 4 entries (byte b holds codes 4 b .. 4 b + 3, the low bits first); cb3 1,536 bytes with 8 (codes 8 g .. 8 g + 7 are the 24-bit little-endian integer of bytes
// 3 g .. 3 g + 2); cb4 2,048 bytes with 16 (the low nibble first); tern 820 bytes with 3 (five codes a byte, byte = c0 + 3 c1 + 9 c2 + 27 c3 + 81 c4, the last byte padded with zeros). A place holds an expert as the pack's
// file does: the codes of downT (2,048 rows), then up's (4,096 rows); the codebooks (bf16, `entries` a row) the same way; in a layer of mixed formats the place is as big as the biggest expert and the last byte of its
// codebook bytes says the formats (up's in the low nibble, down's in the high one: 0 cb2, 1 cb3, 2 cb4, 3 tern). glm_cb_upx and glm_cb_downx are glm_cb_up and glm_cb_down for such places: the same arithmetic for
// cb2 (the same bits), a lane's entries held one to a lane (lane i holds entry i) and a code picking its entry by a shuffle for the others.

__device__ __forceinline__ int gc_row_bytes(int f) { return f == 0 ? 1024 : f == 1 ? 1536 : f == 2 ? 2048 : 820; }
__device__ __forceinline__ int gc_book_bytes(int f) { return f == 0 ? 8 : f == 1 ? 16 : f == 2 ? 32 : 6; }

// Entry `lane` of a row's codebook of `k` entries (0 for a lane past them).
__device__ __forceinline__ float gc_entry(const unsigned char* book, int lane, int k) {
    return lane < k ? __uint_as_float(((unsigned)(*reinterpret_cast<const unsigned short*>(book + 2 * lane))) << 16) : 0.f;
}

// x' of the pair's token into shared memory in the layout the lane-wise products read: 32 float4 per lane of 33 (the padding keeps the lanes of a warp in different banks).
__device__ __forceinline__ void gc_load_x4(float4* xs, const float* xrow) {
    const float4* xg = reinterpret_cast<const float4*>(xrow);
    for (int i = threadIdx.x; i < GC_H / 4; i += 256) xs[(i >> 5) * 33 + (i & 31)] = xg[i];
    __syncthreads();
}

// up, cb2: glm_cb_up's rows of the product (a lane 32 bytes of a row).
__device__ __forceinline__ void gc_up_cb2(const unsigned char* cb, const unsigned char* bb, float4* xs, float* guo) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const float4* xv = xs + lane * 33;
    for (int rr = 0; rr < 4; rr++) {
        const int row = blockIdx.x * 32 + w * 4 + rr;
        const uint4* cp = reinterpret_cast<const uint4*>(cb + (size_t)row * GC_ROW + lane * 32);
        const uint4 c0 = cp[0], c1 = cp[1];
        const uint2 bk = *reinterpret_cast<const uint2*>(bb + (size_t)row * 8);
        const float2 b01 = bf2(bk.x), b23 = bf2(bk.y);
        float acc = gc_dot16(c0.x, xv, b01.x, b01.y, b23.x, b23.y);
        acc += gc_dot16(c0.y, xv + 4, b01.x, b01.y, b23.x, b23.y);
        acc += gc_dot16(c0.z, xv + 8, b01.x, b01.y, b23.x, b23.y);
        acc += gc_dot16(c0.w, xv + 12, b01.x, b01.y, b23.x, b23.y);
        acc += gc_dot16(c1.x, xv + 16, b01.x, b01.y, b23.x, b23.y);
        acc += gc_dot16(c1.y, xv + 20, b01.x, b01.y, b23.x, b23.y);
        acc += gc_dot16(c1.z, xv + 24, b01.x, b01.y, b23.x, b23.y);
        acc += gc_dot16(c1.w, xv + 28, b01.x, b01.y, b23.x, b23.y);
        acc = warp_sum(acc);
        if (lane == 0) guo[row] = acc;
    }
}

// up, cb3: a lane 48 bytes (128 codes) of a row, sixteen groups of eight codes.
__device__ __forceinline__ void gc_up_cb3(const unsigned char* cb, const unsigned char* bb, float4* xs, float* guo) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const float4* xv = xs + lane * 33;
    for (int rr = 0; rr < 4; rr++) {
        const int row = blockIdx.x * 32 + w * 4 + rr;
        const uint4* cp = reinterpret_cast<const uint4*>(cb + (size_t)row * 1536 + lane * 48);
        const uint4 a = cp[0], b = cp[1], c = cp[2];
        const unsigned wd[12] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w, c.x, c.y, c.z, c.w};
        const float ent = gc_entry(bb + (size_t)row * 16, lane, 8);
        float acc = 0.f;
#pragma unroll
        for (int g = 0; g < 16; g++) {
            const int bit = 24 * g, wi = bit >> 5, sh = bit & 31;
            unsigned v = wd[wi] >> sh;
            if (sh > 8) v |= wd[wi + 1] << (32 - sh);
            const float4 z0 = xv[2 * g], z1 = xv[2 * g + 1];
            acc = fmaf(__shfl_sync(0xffffffffu, ent, v & 7), z0.x, acc);
            acc = fmaf(__shfl_sync(0xffffffffu, ent, (v >> 3) & 7), z0.y, acc);
            acc = fmaf(__shfl_sync(0xffffffffu, ent, (v >> 6) & 7), z0.z, acc);
            acc = fmaf(__shfl_sync(0xffffffffu, ent, (v >> 9) & 7), z0.w, acc);
            acc = fmaf(__shfl_sync(0xffffffffu, ent, (v >> 12) & 7), z1.x, acc);
            acc = fmaf(__shfl_sync(0xffffffffu, ent, (v >> 15) & 7), z1.y, acc);
            acc = fmaf(__shfl_sync(0xffffffffu, ent, (v >> 18) & 7), z1.z, acc);
            acc = fmaf(__shfl_sync(0xffffffffu, ent, (v >> 21) & 7), z1.w, acc);
        }
        acc = warp_sum(acc);
        if (lane == 0) guo[row] = acc;
    }
}

// up, cb4: a lane 64 bytes (128 codes) of a row, sixteen words of eight nibbles.
__device__ __forceinline__ void gc_up_cb4(const unsigned char* cb, const unsigned char* bb, float4* xs, float* guo) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const float4* xv = xs + lane * 33;
    for (int rr = 0; rr < 4; rr++) {
        const int row = blockIdx.x * 32 + w * 4 + rr;
        const uint4* cp = reinterpret_cast<const uint4*>(cb + (size_t)row * 2048 + lane * 64);
        const uint4 a = cp[0], b = cp[1], c = cp[2], d = cp[3];
        const unsigned wd[16] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w, c.x, c.y, c.z, c.w, d.x, d.y, d.z, d.w};
        const float ent = gc_entry(bb + (size_t)row * 32, lane, 16);
        float acc = 0.f;
#pragma unroll
        for (int g = 0; g < 16; g++) {
            const unsigned v = wd[g];
            const float4 z0 = xv[2 * g], z1 = xv[2 * g + 1];
            acc = fmaf(__shfl_sync(0xffffffffu, ent, v & 15), z0.x, acc);
            acc = fmaf(__shfl_sync(0xffffffffu, ent, (v >> 4) & 15), z0.y, acc);
            acc = fmaf(__shfl_sync(0xffffffffu, ent, (v >> 8) & 15), z0.z, acc);
            acc = fmaf(__shfl_sync(0xffffffffu, ent, (v >> 12) & 15), z0.w, acc);
            acc = fmaf(__shfl_sync(0xffffffffu, ent, (v >> 16) & 15), z1.x, acc);
            acc = fmaf(__shfl_sync(0xffffffffu, ent, (v >> 20) & 15), z1.y, acc);
            acc = fmaf(__shfl_sync(0xffffffffu, ent, (v >> 24) & 15), z1.z, acc);
            acc = fmaf(__shfl_sync(0xffffffffu, ent, v >> 28), z1.w, acc);
        }
        acc = warp_sum(acc);
        if (lane == 0) guo[row] = acc;
    }
}

// The five digits (base 3, the first the lowest) of each byte, three bits each in a short.
__device__ __forceinline__ void gc_tern_table(unsigned short* tab) {
    for (int b = threadIdx.x; b < 256; b += 256) {
        unsigned v = b, o = 0;
#pragma unroll
        for (int j = 0; j < 5; j++) {
            o |= (v % 3u) << (3 * j);
            v /= 3u;
        }
        tab[b] = (unsigned short)o;
    }
}

// up, tern: a lane the bytes lane, lane + 32, ... of a row (five codes each), x' unpadded; 26 rounds for every lane with the shuffles in step (a lane past the row's 820 bytes adds nothing).
__device__ __forceinline__ void gc_up_tern(const unsigned char* cb, const unsigned char* bb, float* xs, const unsigned short* tab, float* guo) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    for (int rr = 0; rr < 4; rr++) {
        const int row = blockIdx.x * 32 + w * 4 + rr;
        const unsigned char* rp = cb + (size_t)row * 820;
        const float ent = gc_entry(bb + (size_t)row * 6, lane, 3);
        float acc = 0.f;
        for (int m = 0; m < 26; m++) {
            const int bi = lane + 32 * m;
            const bool ok = bi < 820;
            const unsigned d = ok ? tab[rp[bi]] : 0u;
            const float* z = xs + 5 * (ok ? bi : 0);
            const float f = ok ? 1.f : 0.f;
            acc = fmaf(__shfl_sync(0xffffffffu, ent, d & 7), z[0] * f, acc);
            acc = fmaf(__shfl_sync(0xffffffffu, ent, (d >> 3) & 7), z[1] * f, acc);
            acc = fmaf(__shfl_sync(0xffffffffu, ent, (d >> 6) & 7), z[2] * f, acc);
            acc = fmaf(__shfl_sync(0xffffffffu, ent, (d >> 9) & 7), z[3] * f, acc);
            acc = fmaf(__shfl_sync(0xffffffffu, ent, (d >> 12) & 7), z[4] * f, acc);
        }
        acc = warp_sum(acc);
        if (lane == 0) guo[row] = acc;
    }
}

// glm_cb_up for places in any of the formats: the formats from the last byte of the place's codebook bytes.
extern "C" __global__ void __launch_bounds__(256) glm_cb_upx(const unsigned char* codes, unsigned long long cstride, const unsigned char* books, unsigned long long bstride, const long long* way, const float* xr, float* gu, int k) {
    const int p = blockIdx.y;
    const long long pl = way[p];
    if (pl < 0) return;
    __shared__ __align__(16) float smem[33 * 32 * 4];
    __shared__ unsigned short tab[256];
    const unsigned char* bplace = books + (size_t)pl * bstride;
    const int fb = bplace[bstride - 1], fu = fb & 15, fd = fb >> 4;
    const unsigned char* cb = codes + (size_t)pl * cstride + (size_t)GC_I * gc_row_bytes(fd);
    const unsigned char* bb = bplace + (size_t)GC_I * gc_book_bytes(fd);
    const float* xrow = xr + (size_t)(p / k) * GC_H;
    float* guo = gu + (size_t)p * GC_H;
    if (fu == 3) {
        for (int i = threadIdx.x; i < GC_H; i += 256) smem[i] = xrow[i];
        for (int i = GC_H + threadIdx.x; i < GC_H + 8; i += 256) smem[i] = 0.f;
        gc_tern_table(tab);
        __syncthreads();
        gc_up_tern(cb, bb, smem, tab, guo);
        return;
    }
    float4* xs = reinterpret_cast<float4*>(smem);
    gc_load_x4(xs, xrow);
    if (fu == 0) gc_up_cb2(cb, bb, xs, guo);
    else if (fu == 1) gc_up_cb3(cb, bb, xs, guo);
    else gc_up_cb4(cb, bb, xs, guo);
}

// The activation of the neurons n0 .. n0 + ns of a pair, into shared memory.
__device__ __forceinline__ void gc_act(float* act, const float* guo, int n0, int ns, float limit) {
    for (int i = threadIdx.x; i < ns; i += 256) {
        const float gg = fminf(guo[n0 + i], limit), uu = fminf(fmaxf(guo[GC_I + n0 + i], -limit), limit);
        act[i] = gg / (1.f + expf(-gg)) * uu;
    }
    __syncthreads();
}

// The groups' sums of a block added through shared memory: `w` outputs a group, to `out` (the slice's first output).
__device__ __forceinline__ void gc_down_store(const float* red, int w, float* out) {
    for (int o = threadIdx.x; o < w; o += 256) out[o] = (red[o] + red[w + o]) + (red[2 * w + o] + red[3 * w + o]);
}

// down, cb2: glm_cb_down's.
__device__ __forceinline__ void gc_down_cb2(const unsigned char* codes, const unsigned char* books, const float* act, float* red, int os, int ns, int n0, float* out) {
    const int tid = threadIdx.x, g = tid >> 6, l = tid & 63;
    const unsigned char* cb = codes + (size_t)n0 * GC_ROW + os * 256 + l * 4;
    const unsigned char* bb = books + (size_t)n0 * 8;
    float acc[16];
#pragma unroll
    for (int j = 0; j < 16; j++) acc[j] = 0.f;
#pragma unroll 4
    for (int i = g; i < ns; i += 4) {
        const unsigned w = *reinterpret_cast<const unsigned*>(cb + (size_t)i * GC_ROW);
        const uint2 bk = *reinterpret_cast<const uint2*>(bb + (size_t)i * 8);
        const float2 b01 = bf2(bk.x), b23 = bf2(bk.y);
        const float a = act[i];
        const float e0 = a * b01.x, e1 = a * b01.y, e2 = a * b23.x, e3 = a * b23.y;
#pragma unroll
        for (int j = 0; j < 16; j++) acc[j] += gc_pick((w >> (2 * j)) & 3u, e0, e1, e2, e3);
    }
#pragma unroll
    for (int j = 0; j < 16; j++) red[g * 1024 + l * 16 + j] = acc[j];
    __syncthreads();
    gc_down_store(red, 1024, out + os * 1024);
}

// down, cb3: a thread 16 outputs of a row of downT (6 bytes), the entries (times the activation) one to a lane and picked by a shuffle.
__device__ __forceinline__ void gc_down_cb3(const unsigned char* codes, const unsigned char* books, const float* act, float* red, int os, int ns, int n0, float* out) {
    const int tid = threadIdx.x, g = tid >> 6, l = tid & 63, lane = tid & 31;
    const unsigned char* cb = codes + (size_t)n0 * 1536 + os * 384 + l * 6;
    const unsigned char* bb = books + (size_t)n0 * 16;
    float acc[16];
#pragma unroll
    for (int j = 0; j < 16; j++) acc[j] = 0.f;
#pragma unroll 2
    for (int i = g; i < ns; i += 4) {
        const unsigned short* sp = reinterpret_cast<const unsigned short*>(cb + (size_t)i * 1536);
        const unsigned long long v = (unsigned long long)sp[0] | ((unsigned long long)sp[1] << 16) | ((unsigned long long)sp[2] << 32);
        const float ent = act[i] * gc_entry(bb + (size_t)i * 16, lane, 8);
#pragma unroll
        for (int j = 0; j < 16; j++) acc[j] += __shfl_sync(0xffffffffu, ent, (int)((v >> (3 * j)) & 7ull));
    }
#pragma unroll
    for (int j = 0; j < 16; j++) red[g * 1024 + l * 16 + j] = acc[j];
    __syncthreads();
    gc_down_store(red, 1024, out + os * 1024);
}

// down, cb4: a thread 8 bytes (16 outputs) of a row.
__device__ __forceinline__ void gc_down_cb4(const unsigned char* codes, const unsigned char* books, const float* act, float* red, int os, int ns, int n0, float* out) {
    const int tid = threadIdx.x, g = tid >> 6, l = tid & 63, lane = tid & 31;
    const unsigned char* cb = codes + (size_t)n0 * 2048 + os * 512 + l * 8;
    const unsigned char* bb = books + (size_t)n0 * 32;
    float acc[16];
#pragma unroll
    for (int j = 0; j < 16; j++) acc[j] = 0.f;
#pragma unroll 2
    for (int i = g; i < ns; i += 4) {
        const uint2 v = *reinterpret_cast<const uint2*>(cb + (size_t)i * 2048);
        const float ent = act[i] * gc_entry(bb + (size_t)i * 32, lane, 16);
#pragma unroll
        for (int j = 0; j < 8; j++) acc[j] += __shfl_sync(0xffffffffu, ent, (int)((v.x >> (4 * j)) & 15u));
#pragma unroll
        for (int j = 0; j < 8; j++) acc[8 + j] += __shfl_sync(0xffffffffu, ent, (int)((v.y >> (4 * j)) & 15u));
    }
#pragma unroll
    for (int j = 0; j < 16; j++) red[g * 1024 + l * 16 + j] = acc[j];
    __syncthreads();
    gc_down_store(red, 1024, out + os * 1024);
}

// down, tern: a thread four bytes (20 outputs) of a row, 64 threads 1,280 outputs, four slices to a row (the last has 256 outputs); the digits from the table; every thread takes part in the shuffles (a thread past the
// row's bytes reads the row's last four and its outputs, past 4,096, are not written).
__device__ __forceinline__ void gc_down_tern(const unsigned char* codes, const unsigned char* books, const float* act, float* red, const unsigned short* tab, int os, int ns, int n0, float* out) {
    const int tid = threadIdx.x, g = tid >> 6, l = tid & 63, lane = tid & 31;
    const int byte0 = os * 256 + l * 4;
    const bool live = byte0 < 820;
    const unsigned char* cb = codes + (size_t)n0 * 820 + (live ? byte0 : 816);
    const unsigned char* bb = books + (size_t)n0 * 6;
    float acc[20];
#pragma unroll
    for (int j = 0; j < 20; j++) acc[j] = 0.f;
#pragma unroll 2
    for (int i = g; i < ns; i += 4) {
        const unsigned v = *reinterpret_cast<const unsigned*>(cb + (size_t)i * 820);
        const float ent = act[i] * gc_entry(bb + (size_t)i * 6, lane, 3);
#pragma unroll
        for (int q = 0; q < 4; q++) {
            const unsigned d = tab[(v >> (8 * q)) & 255u];
#pragma unroll
            for (int j = 0; j < 5; j++) acc[5 * q + j] += __shfl_sync(0xffffffffu, ent, (int)((d >> (3 * j)) & 7u));
        }
    }
#pragma unroll
    for (int j = 0; j < 20; j++) red[g * 1280 + l * 20 + j] = acc[j];
    __syncthreads();
    for (int o = tid; o < 1280; o += 256) {
        const int oo = os * 1280 + o;
        if (oo < GC_H) out[oo] = (red[o] + red[1280 + o]) + (red[2560 + o] + red[3840 + o]);
    }
}

// glm_cb_down for places in any of the formats (downT's, from the high nibble of the place's last codebook byte). Grid [4, S, pairs] as glm_cb_down's: the output slices are 1,024 outputs wide, tern's 1,280.
extern "C" __global__ void __launch_bounds__(256) glm_cb_downx(const unsigned char* codes, unsigned long long cstride, const unsigned char* books, unsigned long long bstride, const long long* way, const float* gu, float* part, int S, float limit) {
    const int os = blockIdx.x, s = blockIdx.y, p = blockIdx.z;
    const long long pl = way[p];
    if (pl < 0) return;
    __shared__ float act[GC_I];
    __shared__ float red[4 * 1280];
    __shared__ unsigned short tab[256];
    const unsigned char* bplace = books + (size_t)pl * bstride;
    const int fd = bplace[bstride - 1] >> 4;
    const int ns = GC_I / S, n0 = s * ns;
    gc_act(act, gu + (size_t)p * GC_H, n0, ns, limit);
    const unsigned char* cplace = codes + (size_t)pl * cstride;
    float* out = part + ((size_t)p * S + s) * GC_H;
    if (fd == 0) gc_down_cb2(cplace, bplace, act, red, os, ns, n0, out);
    else if (fd == 1) gc_down_cb3(cplace, bplace, act, red, os, ns, n0, out);
    else if (fd == 2) gc_down_cb4(cplace, bplace, act, red, os, ns, n0, out);
    else {
        gc_tern_table(tab);
        __syncthreads();
        gc_down_tern(cplace, bplace, act, red, tab, os, ns, n0, out);
    }
}
