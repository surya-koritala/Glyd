// Hummingbird's experts for every mixture of experts (included by engine.cu after moe_hb.cu, whose helpers it uses; docs/design/hummingbird.md, #429): the products of a routed expert over codebook rows, for any width W (the experts' input: the
// hidden size, or the latent) in runs of 32 and the rotation's block b (a power of two from 32 to 1,024 that divides W), a gated expert (SwiGLU, GeGLU, situ: U = 2 I rows in its first matrix, the gate's first) or not (relu squared: U = I).
// A matrix is `rows` neuron rows of W weights in the three arrays of moe_hb.cu (`codes`: 2 or 3 bits a weight, a row W * bits / 8 bytes, a little-endian bitstream, a run of 32 codes 8 or 12 bytes; `scales`: a bf16 for each run, [row][run];
// `cb`: the row's codebook, 2^bits bf16, [row][entry]); a weight is f32(scale of its run) * f32(codebook[code]), exact in fp32. The rows are in the basis of R = (I x H_b) diag(signs) (H_b the natural-order Walsh-Hadamard matrix of order b over
// sqrt b, applied to each run of b values; the signs +1 or -1, one for each of the W values, shared by the layer's experts): the gate_up rows are R w_n, the down_t rows R d_n. A token is rotated once (x' = R x), every expert multiplies in that
// basis (<R w, R x> = <w, x>), and the weighted sum of the experts' outputs is rotated back once (R^T v = signs o (H v / sqrt b)). `scale` is the host's 1 / sqrt(b) as hb_codec.rs computes it in fp32, so that the bits are the codec's.
// A layer's experts are in two groups, one for each bit width (group 0: 2 bits, group 1: 3 bits), each a stack of its experts end to end at a fixed stride a matrix (the strides are the bytes of an expert's matrix in the group): `group_of`
// (u8 [E]) says an expert's group and `local_of` (u16 [E]) its place in it. Every kernel of a product takes the pairs of the plan (Lib::moe_way's: plan [2 + 2 E + P] int32, the pairs sorted by expert; `ids`, the int64 [P] the plan was made of,
// give a pair's expert) and only does the pairs of its own group's experts; a layer with both widths launches the kernel of each.
//
// hbg_rot_in (blocks of max(32, b / 4) threads, a thread four values; grid (tokens, ceil(W / (4 threads)))): xr [token][W] (fp32) = R x of the token's row of x (bf16, `xs` elements apart): the signs, then the butterflies (a, c) -> (a + c, a - c)
// of hbg_wht on each run of b values, then times `scale`. No atomics: the same bits every time.
// hbg_up2, hbg_up3 (a block a pair of the plan and 64 neurons, 8 warps of 8 neurons each): gu [plan row][n] = bf16(<row n of the pair's expert, xr[token]>) for the first matrix's U rows, into the [pairs, U] rows in the plan's order that the library's first
// product writes (so that the family's activation kernel runs on them as it does). The token's row of xr is staged in shared memory (a run of 32 values a 36 floats, 4 of padding, W / 32 * 144 bytes of dynamic shared memory) and a warp takes a neuron at
// a time: lane l has the runs l, l + 32, ... of the row (its codes and scales loaded first, for all of its runs), a run's sum is the 32 products added in order from 0 (fmaf of the codebook entry and xr), the lane's sum adds the runs' sums times
// their scales (fmaf) in the order of the runs, and the 32 lanes' sums are added by an xor-shuffle tree (offsets 1, 2, 4, 8, 16): a row's sum depends on the row alone, not on the place the expert is in nor on the timing. The row's codebook is staged
// in shared memory a warp (the same for the 32 lanes).
// hbg_down2, hbg_down3 (a block a pair of the plan and 128 outputs, 8 warps): part [plan row][o] (fp32, in the rotated basis) = the sum over the I neurons n of h[plan row][n] times down_t row n at o, dense (a gated activation has no zeros): h is the
// activation's rows [pairs, I] (bf16) in the plan's order. A lane takes one run of the block's 128 outputs of one neuron of its step: lane = 4 g + r (g 0 to 7, r 0 to 3), the neuron n = 64 s + 8 warp + g at step s, the run 4 blockIdx.y + r,
// and adds the neuron's row times the activation into 32 fp32 accumulators (c = activation * scale of the run, one fp32 multiply and exact; acc = fmaf(codebook entry, c, acc)), the neurons in the order of s; the 8 neurons of a step's lane groups
// are added by an xor-shuffle tree over the lane bits 2 to 4 (offsets 4, 8, 16), and the 8 warps' sums in the order of the warps through shared memory. A run past the row (W not a multiple of 128) is skipped.
// hbg_inv: inv [pair] = its row in the plan (a pair routed nowhere is not in the plan: its entry is not written).
// hbg_sum_rot (a block of max(32, b / 4) threads for a token's run of b values: grid (tokens, ceil(W / (4 threads)))): v = the sum over the token's choices c in order (an id outside 0 to E - 1: skipped) of wts[c] times the choice's row of part (the
// plan's row inv[t k + c]), fmaf in fp32, then extra [token][o] (fp32, in the rotated basis; none where extra is null) added; then y = bf16(signs o (scale H v)), the butterflies of hbg_rot_in. One rounding to bf16, after R^T. Weights may all be one
// (DeepSeek-V4 folds them into its activation).
// hbg_unrot (a block as hbg_sum_rot's, a plan row and a run): rows [plan row][o] (bf16) = signs o (scale H part[plan row]): each pair's output in the basis of the model, for a family that normalises each expert's output before it sums them (Muse Spark).
//
// The grids: hbg_up{2,3} (pairs, U / 64 rounded up) of 256 threads; hbg_down{2,3} (pairs, ceil(W / 128)) of 256 threads; hbg_inv (pairs / 256 rounded up) of 256; the pairs in the plan's order (a block past the plan's pairs does nothing; a pair routed
// nowhere is not in the plan). Nothing here reads a value but through the lists: a decode step captures into a CUDA graph. It uses moe.cu's MOE_FULL and moe_hb.cu's hb_bf and hb_code.

#define HBG_UP_ROWS 64       // neurons a block of hbg_up takes (8 warps of 8)
#define HBG_DOWN_OUT 128     // outputs a block of hbg_down takes (4 runs of 32)
#define HBG_MAX_RUNS 256     // runs of 32 in a row at most: W up to 8,192
#define HBG_RL 8             // runs a lane of hbg_up has of a row at most (HBG_MAX_RUNS / 32)
#define HBG_XPAD 36          // floats a run of the token's row takes in shared memory: its 32, and 4 of padding (a quarter warp's 16-byte reads of 8 lanes' runs then hit different banks)
#define HBG_DOWN_UNROLL 4    // steps of neurons a warp of hbg_down has loaded before it decodes them

// The Walsh-Hadamard butterflies (unnormalized, natural order) of runs of b values (a power of two from 32 to 1,024; th = b / 4 threads a run, a thread holds 4 consecutive values v[0..3]): (a, c) -> (a + c, a - c) for the index pairs of each bit,
// bit 0 up, the order of hb_wht (and of hb_codec.rs). Bits 0 and 1 in the thread, bits 2 to 6 across the lanes of a warp (xor shuffles, the lanes of a run when th < 32), the higher bits across the warps through sm [4 blockDim.x]. The block has
// max(32, th) threads (a run of b < 128 values is part of a warp, several runs a block), every thread of the block calls it.
__device__ __forceinline__ void hbg_wht(float* v, float* sm, int b) {
    const int tid = threadIdx.x, lane = tid & 31, th = b >> 2;
    float w0 = v[0], w1 = v[1], w2 = v[2], w3 = v[3];
    v[0] = w0 + w1;
    v[1] = w0 - w1;
    v[2] = w2 + w3;
    v[3] = w2 - w3;
    w0 = v[0], w1 = v[1], w2 = v[2], w3 = v[3];
    v[0] = w0 + w2;
    v[1] = w1 + w3;
    v[2] = w0 - w2;
    v[3] = w1 - w3;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
        if (m >= th) break;  // (th is the same for the block)
        const bool hi = (lane & m) != 0;  // (the partner lane is lane ^ m: of the same run; the one with the bit clear holds a)
#pragma unroll
        for (int q = 0; q < 4; q++) {
            const float o = __shfl_xor_sync(MOE_FULL, v[q], m);
            v[q] = hi ? o - v[q] : v[q] + o;
        }
    }
    for (int m = 32; m < th; m <<= 1) {  // (the partner thread is tid ^ m, the same lane of another warp)
#pragma unroll
        for (int q = 0; q < 4; q++) sm[4 * tid + q] = v[q];
        __syncthreads();
#pragma unroll
        for (int q = 0; q < 4; q++) {
            const float o = sm[4 * (tid ^ m) + q];
            v[q] = (tid & m) ? o - v[q] : v[q] + o;
        }
        __syncthreads();
    }
}

// xr [T, W] (fp32) = scale H (signs o x) on each run of b values of each token's row of x [T, xs] (bf16): grid (token, ceil(W / (4 max(32, b / 4)))), max(32, b / 4) threads (a thread's four values 4 threadIdx.x ..: the first column of a thread's
// four is past W only in a block that has more threads than W / 4 values; those threads take part with zeros), signs [W] = +1 or -1.
extern "C" __global__ void __launch_bounds__(256) hbg_rot_in(const bf16* x, int xs, const signed char* signs, float* xr, int W, int b, float scale) {
    __shared__ float sm[1024];
    const size_t t = blockIdx.x;
    const int c = blockIdx.y * 4 * blockDim.x + 4 * threadIdx.x;
    float v[4] = {0.f, 0.f, 0.f, 0.f};
    if (c < W) {
#pragma unroll
        for (int q = 0; q < 4; q++) v[q] = (float)signs[c + q] * b2f(x[t * xs + c + q]);
    }
    hbg_wht(v, sm, b);
    if (c < W) {
        uint4 ov;
        ov.x = __float_as_uint(scale * v[0]);
        ov.y = __float_as_uint(scale * v[1]);
        ov.z = __float_as_uint(scale * v[2]);
        ov.w = __float_as_uint(scale * v[3]);
        *reinterpret_cast<uint4*>(xr + t * W + c) = ov;
    }
}

// One block of hbg_up for BITS (2 or 3) a code. xr: [tokens, W] fp32 (hbg_rot_in's); plan: Lib::moe_way's [2 + 2 E + pairs] int32 made of `ids` (int64 [pairs]); group_of, local_of: the experts' groups and places in them (the block does the pairs whose
// expert is in group BITS - 2); codes, scales, cb: the first matrices' three arrays at the first place, `codes_stride`, `scales_stride`, `cb_stride` bytes between places; gu: [pairs, U] bf16, row i the plan's pair i; k: choices a token (pair j is of
// token j / k). W a multiple of 32 up to 8,192, every row's bytes 4-byte aligned; dynamic shared memory W / 32 * 4 * HBG_XPAD bytes.
template <int BITS>
__device__ __forceinline__ void hbg_up_t(const float* xr, const int* plan, const long long* ids, const unsigned char* group_of, const unsigned short* local_of, const unsigned char* codes, const unsigned char* scales,
                                         const unsigned char* cb, long long codes_stride, long long scales_stride, long long cb_stride, bf16* gu, int E, int U, int W, int k) {
    constexpr int NCB = 1 << BITS;  // entries of a row's codebook
    extern __shared__ __align__(16) float hbg_xs[];
    __shared__ float tab[8][NCB];  // a warp's neuron's codebook
    const int i = blockIdx.x, tile = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int hits = plan[0];
    if (i >= plan[1 + E + hits]) return;  // (the whole block: no pair this far down the plan)
    const int j = plan[2 + 2 * E + i];
    const long long e = ids[j];
    if (e < 0 || e >= E || group_of[e] != BITS - 2) return;  // (the whole block: another group's pair)
    const long long place = local_of[e];
    const int R = W >> 5;  // runs a row
    const float* xp = xr + (size_t)(j / k) * W;
    for (int q = tid; q < R * 8; q += 256) *reinterpret_cast<float4*>(hbg_xs + (q >> 3) * HBG_XPAD + 4 * (q & 7)) = *reinterpret_cast<const float4*>(xp + 4 * q);
    __syncthreads();
    const size_t cstride = (size_t)R * 4 * BITS;  // bytes of a row's codes
    const unsigned char* crow = codes + place * codes_stride;
    const unsigned char* srow = scales + place * scales_stride;
    const unsigned short* brow = reinterpret_cast<const unsigned short*>(cb + place * cb_stride);
    const int n0 = tile * HBG_UP_ROWS + warp * 8;
    for (int u = 0; u < 8; u++) {
        const int n = n0 + u;
        if (n >= U) break;  // (the warp's: n is the same for its lanes)
        unsigned cw[HBG_RL][BITS], sc[HBG_RL];
#pragma unroll
        for (int m = 0; m < HBG_RL; m++) {  // (all the loads first; a run past the row is skipped)
            const int r = lane + 32 * m;
#pragma unroll
            for (int w = 0; w < BITS; w++) cw[m][w] = 0u;
            sc[m] = 0u;
            if (r < R) {
                const unsigned* cp = reinterpret_cast<const unsigned*>(crow + (size_t)n * cstride + 4 * BITS * r);
#pragma unroll
                for (int w = 0; w < BITS; w++) cw[m][w] = cp[w];
                sc[m] = *reinterpret_cast<const unsigned short*>(srow + ((size_t)n * R + r) * 2);
            }
        }
        if (lane < NCB) tab[warp][lane] = hb_bf(brow[(size_t)n * NCB + lane]);
        __syncwarp();
        float acc = 0.f;  // this lane's sum over its runs, in the order of the runs
#pragma unroll
        for (int m = 0; m < HBG_RL; m++) {
            const int r = lane + 32 * m;
            if (r < R) {
                const float* xv = hbg_xs + r * HBG_XPAD;
                float t = 0.f;
#pragma unroll
                for (int q = 0; q < 32; q += 4) {
                    const float4 x4 = *reinterpret_cast<const float4*>(xv + q);
                    t = fmaf(tab[warp][hb_code<BITS>(cw[m], q)], x4.x, t);
                    t = fmaf(tab[warp][hb_code<BITS>(cw[m], q + 1)], x4.y, t);
                    t = fmaf(tab[warp][hb_code<BITS>(cw[m], q + 2)], x4.z, t);
                    t = fmaf(tab[warp][hb_code<BITS>(cw[m], q + 3)], x4.w, t);
                }
                acc = fmaf(hb_bf((unsigned short)sc[m]), t, acc);
            }
        }
        for (int off = 1; off < 32; off <<= 1) acc += __shfl_xor_sync(MOE_FULL, acc, off);  // (every lane has the row's sum)
        if (lane == 0) gu[(size_t)i * U + n] = f2b(acc);
        __syncwarp();  // (the table is read: the next neuron's may be written)
    }
}

extern "C" __global__ void __launch_bounds__(256) hbg_up2(const float* xr, const int* plan, const long long* ids, const unsigned char* group_of, const unsigned short* local_of, const unsigned char* codes, const unsigned char* scales,
                                                          const unsigned char* cb, long long codes_stride, long long scales_stride, long long cb_stride, bf16* gu, int E, int U, int W, int k) {
    hbg_up_t<2>(xr, plan, ids, group_of, local_of, codes, scales, cb, codes_stride, scales_stride, cb_stride, gu, E, U, W, k);
}
extern "C" __global__ void __launch_bounds__(256) hbg_up3(const float* xr, const int* plan, const long long* ids, const unsigned char* group_of, const unsigned short* local_of, const unsigned char* codes, const unsigned char* scales,
                                                          const unsigned char* cb, long long codes_stride, long long scales_stride, long long cb_stride, bf16* gu, int E, int U, int W, int k) {
    hbg_up_t<3>(xr, plan, ids, group_of, local_of, codes, scales, cb, codes_stride, scales_stride, cb_stride, gu, E, U, W, k);
}

// One block of hbg_down for BITS (2 or 3) a code. h: the activation's rows [pairs, I] bf16, row i the plan's pair i; plan, ids, group_of, local_of as hbg_up's; codes, scales, cb: the down_t matrices' three arrays (rows = neurons, I of them, W values a
// row) at the first place and their strides between places; part: [pairs, W] fp32, row i the plan's pair i. W a multiple of 32, every row's bytes 4-byte aligned.
template <int BITS>
__device__ __forceinline__ void hbg_down_t(const bf16* h, const int* plan, const long long* ids, const unsigned char* group_of, const unsigned short* local_of, const unsigned char* codes, const unsigned char* scales,
                                           const unsigned char* cb, long long codes_stride, long long scales_stride, long long cb_stride, float* part, int E, int I, int W) {
    constexpr int NCB = 1 << BITS;  // entries of a row's codebook
    __shared__ float tab[8][HBG_DOWN_UNROLL][8][NCB];  // the codebooks of the neurons a warp has loaded: [warp][step][lane group][entry]
    __shared__ float red[8][HBG_DOWN_OUT];             // the warps' sums
    const int i = blockIdx.x, chunk = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, r = lane & 3;
    const int hits = plan[0];
    if (i >= plan[1 + E + hits]) return;  // (the whole block: no pair this far down the plan)
    const int j = plan[2 + 2 * E + i];
    const long long e = ids[j];
    if (e < 0 || e >= E || group_of[e] != BITS - 2) return;  // (the whole block: another group's pair)
    const long long place = local_of[e];
    const int R = W >> 5;            // runs a row
    const int run = chunk * 4 + r;   // this lane's run of the row
    const bool live = run < R;       // (a lane past the row takes part in the sums with zeros, and loads the codebooks)
    const size_t cstride = (size_t)R * 4 * BITS;
    const unsigned char* drow = codes + place * codes_stride;
    const unsigned char* srow = scales + place * scales_stride;
    const unsigned short* brow = reinterpret_cast<const unsigned short*>(cb + place * cb_stride);
    const bf16* hrow = h + (size_t)i * I;
    float acc[32];
#pragma unroll
    for (int q = 0; q < 32; q++) acc[q] = 0.f;
    for (int s0 = 0; s0 * 64 < I; s0 += HBG_DOWN_UNROLL) {
        unsigned cw[HBG_DOWN_UNROLL][BITS], sc[HBG_DOWN_UNROLL];
        float av[HBG_DOWN_UNROLL];
#pragma unroll
        for (int u = 0; u < HBG_DOWN_UNROLL; u++) {  // (all the loads first)
            const int n = (s0 + u) * 64 + warp * 8 + g;
#pragma unroll
            for (int w = 0; w < BITS; w++) cw[u][w] = 0u;
            sc[u] = 0u;
            av[u] = 0.f;
            if (n < I) {
                av[u] = b2f(hrow[n]);
                if (live) {
                    const unsigned* cp = reinterpret_cast<const unsigned*>(drow + (size_t)n * cstride + 4 * BITS * run);
#pragma unroll
                    for (int w = 0; w < BITS; w++) cw[u][w] = cp[w];
                    sc[u] = *reinterpret_cast<const unsigned short*>(srow + ((size_t)n * R + run) * 2);
                }
                for (int en = r; en < NCB; en += 4) tab[warp][u][g][en] = hb_bf(brow[(size_t)n * NCB + en]);  // (the 4 lanes of a neuron's group load its entries)
            }
        }
        __syncwarp();
#pragma unroll
        for (int u = 0; u < HBG_DOWN_UNROLL; u++) {
            const int n = (s0 + u) * 64 + warp * 8 + g;
            if (n < I && live) {
                const float c = av[u] * hb_bf((unsigned short)sc[u]);  // activation * scale: exact
                const float* tb = tab[warp][u][g];
#pragma unroll
                for (int q = 0; q < 32; q++) acc[q] = fmaf(tb[hb_code<BITS>(cw[u], q)], c, acc[q]);
            }
        }
        __syncwarp();  // (the tables are read: the next steps' may be written)
    }
    // ---- the 8 groups of a warp added by the xor tree (offsets 4, 8, 16), the warps' sums in the warps' order, and the block's outputs written
#pragma unroll
    for (int off = 4; off < 32; off <<= 1) {
#pragma unroll
        for (int q = 0; q < 32; q++) acc[q] += __shfl_xor_sync(MOE_FULL, acc[q], off);
    }
    if (g == 0) {
#pragma unroll
        for (int q = 0; q < 32; q++) red[warp][r * 32 + q] = acc[q];
    }
    __syncthreads();
    if (tid < HBG_DOWN_OUT) {
        const int o = chunk * HBG_DOWN_OUT + tid;
        float sum = red[0][tid];
#pragma unroll
        for (int w = 1; w < 8; w++) sum += red[w][tid];
        if (o < W) part[(size_t)i * W + o] = sum;
    }
}

extern "C" __global__ void __launch_bounds__(256) hbg_down2(const bf16* h, const int* plan, const long long* ids, const unsigned char* group_of, const unsigned short* local_of, const unsigned char* codes, const unsigned char* scales,
                                                            const unsigned char* cb, long long codes_stride, long long scales_stride, long long cb_stride, float* part, int E, int I, int W) {
    hbg_down_t<2>(h, plan, ids, group_of, local_of, codes, scales, cb, codes_stride, scales_stride, cb_stride, part, E, I, W);
}
extern "C" __global__ void __launch_bounds__(256) hbg_down3(const bf16* h, const int* plan, const long long* ids, const unsigned char* group_of, const unsigned short* local_of, const unsigned char* codes, const unsigned char* scales,
                                                            const unsigned char* cb, long long codes_stride, long long scales_stride, long long cb_stride, float* part, int E, int I, int W) {
    hbg_down_t<3>(h, plan, ids, group_of, local_of, codes, scales, cb, codes_stride, scales_stride, cb_stride, part, E, I, W);
}

// inv [pair] = its row in the plan, for the pairs in the plan (the first plan[1 + E + plan[0]] rows: the pairs routed to an expert); a thread a row.
extern "C" __global__ void __launch_bounds__(256) hbg_inv(const int* plan, int* inv, int pairs, int E) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < pairs && r < plan[1 + E + plan[0]]) inv[plan[2 + 2 * E + r]] = r;
}

// y [T, W] (bf16) = for each token signs o (scale H v) on each run of b values, v = the sum over its k choices c (ids[t k + c] an expert; outside 0 to E - 1: none) in order of wts[t k + c] times the plan's row inv[t k + c] of part [pairs, W]
// (fp32, fmaf), then extra [t][o] where extra is not null (one rounding at the end, after the rotation back). Grid (token, ceil(W / (4 max(32, b / 4)))), max(32, b / 4) threads, signs [W] = +1 or -1.
extern "C" __global__ void __launch_bounds__(256) hbg_sum_rot(const float* part, const int* inv, const float* wts, const long long* ids, const signed char* signs, bf16* y, const float* extra, int E, int k, int W, int b, float scale) {
    __shared__ float sm[1024];
    const size_t t = blockIdx.x;
    const int c = blockIdx.y * 4 * blockDim.x + 4 * threadIdx.x;
    float v[4] = {0.f, 0.f, 0.f, 0.f};
    if (c < W) {
        for (int ch = 0; ch < k; ch++) {
            const long long e = ids[t * k + ch];
            if (e < 0 || e >= E) continue;
            const float w = wts[t * k + ch];
            const uint4 pv = *reinterpret_cast<const uint4*>(part + (size_t)inv[t * k + ch] * W + c);
            v[0] = fmaf(w, __uint_as_float(pv.x), v[0]);
            v[1] = fmaf(w, __uint_as_float(pv.y), v[1]);
            v[2] = fmaf(w, __uint_as_float(pv.z), v[2]);
            v[3] = fmaf(w, __uint_as_float(pv.w), v[3]);
        }
        if (extra) {
            const uint4 xv = *reinterpret_cast<const uint4*>(extra + t * W + c);
            v[0] += __uint_as_float(xv.x);
            v[1] += __uint_as_float(xv.y);
            v[2] += __uint_as_float(xv.z);
            v[3] += __uint_as_float(xv.w);
        }
    }
    hbg_wht(v, sm, b);
    if (c < W) {
#pragma unroll
        for (int q = 0; q < 4; q++) y[t * W + c + q] = f2b((float)signs[c + q] * (scale * v[q]));
    }
}

// rows [pairs, W] (bf16) = for each pair of the plan (its row i), signs o (scale H part[i]) on each run of b values: the pair's output in the basis of the model, rounded to bf16 once. Grid (plan row, ceil(W / (4 max(32, b / 4)))), max(32, b / 4)
// threads; a block past the plan's pairs does nothing.
extern "C" __global__ void __launch_bounds__(256) hbg_unrot(const float* part, const int* plan, const signed char* signs, bf16* rows, int E, int W, int b, float scale) {
    __shared__ float sm[1024];
    const int i = blockIdx.x;
    if (i >= plan[1 + E + plan[0]]) return;  // (the whole block)
    const int c = blockIdx.y * 4 * blockDim.x + 4 * threadIdx.x;
    float v[4] = {0.f, 0.f, 0.f, 0.f};
    if (c < W) {
        const uint4 pv = *reinterpret_cast<const uint4*>(part + (size_t)i * W + c);
        v[0] = __uint_as_float(pv.x);
        v[1] = __uint_as_float(pv.y);
        v[2] = __uint_as_float(pv.z);
        v[3] = __uint_as_float(pv.w);
    }
    hbg_wht(v, sm, b);
    if (c < W) {
#pragma unroll
        for (int q = 0; q < 4; q++) rows[(size_t)i * W + c + q] = f2b((float)signs[c + q] * (scale * v[q]));
    }
}
