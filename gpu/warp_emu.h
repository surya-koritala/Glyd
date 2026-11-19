// The tensor cores' warp operations made of shuffles and fp32 FMAs over the same fragment layouts, for GPUs that have none of them (AMD's RDNA:
// no mma.sync, ldmatrix or movmatrix; gpu/hip.h includes this): the kernels that use them run unchanged, correct and slower. The includer
// defines GLYD_LANE() (the lane in a warp of 32), GLYD_SHFL(v, lane) (v of that lane) and GLYD_LDS(T, a) (the T at shared address a, 32 bits), so
// that glyd-engine/tools/warp_emu_test.cu runs this very code on an NVIDIA GPU against the instructions it stands for.
// ponytail: FMAs in place of the tensor cores; RDNA 3's WMMA (v_wmma_f32_16x16x16_bf16, a 16x16x16 tile a wave32) and RDNA 4's take their
// place in the prompt kernels once a card can measure it.
#pragma once

__device__ __forceinline__ float glyd_bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float glyd_bf_hi(uint32_t w) { return __uint_as_float(w & 0xFFFF0000u); }

// mma.sync m16n8k16 (bf16, fp32 sums): c += a (16x16, row) b (16x8, col). Lane (g, t) = (l / 4, l % 4) holds a's rows g and g + 8 at columns
// 2t, 2t + 1 (a[0], a[1]) and 2t + 8, 2t + 9 (a[2], a[3]), b's column g at rows 2t, 2t + 1 (b0) and 2t + 8, 2t + 9 (b1), and c's rows g and
// g + 8 at columns 2t and 2t + 1: its sums take rows g and g + 8 of a from the lanes of its own group and columns 2t and 2t + 1 of b from the
// lanes of groups 2t and 2t + 1, four k at a time.
__device__ __forceinline__ void glyd_mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    const int lane = GLYD_LANE(), t = lane & 3;
#pragma unroll
    for (int q = 0; q < 4; q++) {
        const int la = (lane & ~3) | q, lb = 8 * t + q;  // (k 2q, 2q + 1 and 2q + 8, 2q + 9)
        const uint32_t r0 = GLYD_SHFL(a[0], la), r8 = GLYD_SHFL(a[1], la), r0k = GLYD_SHFL(a[2], la), r8k = GLYD_SHFL(a[3], la);
        const uint32_t c0 = GLYD_SHFL(b0, lb), c0k = GLYD_SHFL(b1, lb), c1 = GLYD_SHFL(b0, lb + 4), c1k = GLYD_SHFL(b1, lb + 4);
        c[0] = fmaf(glyd_bf_hi(r0k), glyd_bf_hi(c0k), fmaf(glyd_bf_lo(r0k), glyd_bf_lo(c0k), fmaf(glyd_bf_hi(r0), glyd_bf_hi(c0), fmaf(glyd_bf_lo(r0), glyd_bf_lo(c0), c[0]))));
        c[1] = fmaf(glyd_bf_hi(r0k), glyd_bf_hi(c1k), fmaf(glyd_bf_lo(r0k), glyd_bf_lo(c1k), fmaf(glyd_bf_hi(r0), glyd_bf_hi(c1), fmaf(glyd_bf_lo(r0), glyd_bf_lo(c1), c[1]))));
        c[2] = fmaf(glyd_bf_hi(r8k), glyd_bf_hi(c0k), fmaf(glyd_bf_lo(r8k), glyd_bf_lo(c0k), fmaf(glyd_bf_hi(r8), glyd_bf_hi(c0), fmaf(glyd_bf_lo(r8), glyd_bf_lo(c0), c[2]))));
        c[3] = fmaf(glyd_bf_hi(r8k), glyd_bf_hi(c1k), fmaf(glyd_bf_lo(r8k), glyd_bf_lo(c1k), fmaf(glyd_bf_hi(r8), glyd_bf_hi(c1), fmaf(glyd_bf_lo(r8), glyd_bf_lo(c1), c[3]))));
    }
}

// ldmatrix .x4 (b16): lane l takes the word at row l / 4, columns 2 (l % 4) and + 1, of each of four 8x8 matrices, matrix j's rows at the
// shared addresses lanes 8 j to 8 j + 7 hold
__device__ __forceinline__ void glyd_ldsm4(uint32_t* r, uint32_t a) {
    const int lane = GLYD_LANE();
#pragma unroll
    for (int j = 0; j < 4; j++) r[j] = GLYD_LDS(uint32_t, GLYD_SHFL(a, 8 * j + (lane >> 2)) + 4 * (lane & 3));
}
// .trans: lane l takes (row 2 (l % 4), column l / 4) and (row 2 (l % 4) + 1, the same column) of each
__device__ __forceinline__ void glyd_ldsm4t(uint32_t* r, uint32_t a) {
    const int lane = GLYD_LANE(), c = 2 * (lane >> 2);
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const uint32_t lo = GLYD_LDS(uint16_t, GLYD_SHFL(a, 8 * j + 2 * (lane & 3)) + c), hi = GLYD_LDS(uint16_t, GLYD_SHFL(a, 8 * j + 2 * (lane & 3) + 1) + c);
        r[j] = lo | hi << 16;
    }
}
// movmatrix .trans (b16): the 8x8 matrix a warp holds (lane l: row l / 4, columns 2 (l % 4) and + 1), transposed
__device__ __forceinline__ uint32_t glyd_transpose8x8(uint32_t w) {
    const int lane = GLYD_LANE(), col = lane >> 2, src = 8 * (lane & 3) + (col >> 1), sh = 16 * (col & 1);
    return ((GLYD_SHFL(w, src) >> sh) & 0xFFFFu) | ((GLYD_SHFL(w, src + 4) >> sh) & 0xFFFFu) << 16;
}
