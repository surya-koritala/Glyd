// A model's layers across the GPUs of one machine (src/dynamic/tp.rs; included by engine.cu last): each GPU holds a slice of the rows of every big matrix, and after
// its product the slices are gathered, so that every GPU has the whole output and runs the small operations itself. The gather writes over the GPUs' links
// (peer access), no host in between.
//
// tp_gather: src [rows, sl] (this GPU's slice of the output, 16-byte vectors: sl8 of them a row) into dst [rows, n sl] (rows ld8 vectors apart) at columns
// rank sl, and to every other GPU's staging; then, once every other GPU's slice is in this GPU's staging, theirs into dst too. tab: the GPUs' staging bases [n],
// then their flag words [n]. A GPU's staging: [2 parities][n senders][TP_BLOCKS blocks][sub16 vectors]; its flags: [n senders][TP_BLOCKS] words, the sequence
// number a sender's block last wrote. ctr [TP_BLOCKS]: this GPU's count of the gathers on this stream that block b took part in, a word a block. A gather of B
// blocks (as few as its vectors take: a decode step's, one) gives block b the same share of the vectors on every GPU, in block b's own region of the staging,
// so a block waits for the same block of the others alone; block b's parity alternates with its own count, so a sender writes a region two of that block's
// gathers later, after the receiver has passed the one between (whose flags it waited for) and so read it.
#define TP_BLOCKS 16

__device__ __forceinline__ unsigned long long tp_ns() {
#if defined(__HIP_PLATFORM_AMD__)
    return (unsigned long long)wall_clock64() * 10;  // (RDNA's wall clock at 100 MHz, as gpu/glyd_gpu.cu's hold_kernel takes it)
#else
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
#endif
}

extern "C" __global__ void __launch_bounds__(256) tp_gather(const uint4* src, int rows, int sl8, uint4* dst, long long ld8, int rank, int n, const unsigned long long* tab, unsigned int* flags, unsigned int* ctr, long long sub16) {
    const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x;
    __shared__ unsigned int seq;
    if (tid == 0) seq = ctr[b] + 1;
    __syncthreads();
    const unsigned int s = seq;
    const long long par = s & 1, total = (long long)rows * sl8, per = (total + nb - 1) / nb, lo = b * per, hi = min(total, lo + per);
    const auto region = [&](int sender) { return ((par * n + sender) * TP_BLOCKS + b) * sub16 - lo; };  // (vector v of block b's share: at region + v)
    for (long long v = lo + tid; v < hi; v += blockDim.x) {
        const uint4 x = src[v];
        const long long r = v / sl8, c = v % sl8;
        dst[r * ld8 + (long long)rank * sl8 + c] = x;
        for (int q = 0; q < n; q++) {
            if (q != rank) reinterpret_cast<uint4*>(tab[q])[region(rank) + v] = x;
        }
    }
    // (the block's writes, then one fence of the system's scope by the thread that tells the others: each thread's own fence cost a decode step's gather
    // most of its 20 us on 8 GPUs)
    __syncthreads();
    if (tid == 0) {
        __threadfence_system();
        for (int q = 0; q < n; q++) {
            if (q != rank) reinterpret_cast<volatile unsigned int*>(tab[n + q])[rank * TP_BLOCKS + b] = s;
        }
        const volatile unsigned int* mine = flags;
        const unsigned long long t0 = tp_ns();
        for (int q = 0; q < n; q++) {
            if (q != rank) {
                while (mine[q * TP_BLOCKS + b] < s) {
                    const unsigned long long t = tp_ns();
                    // (a GPU that has not come to this gather in 10 s: the group is broken, said and stopped)
                    if (t - t0 > 10000000000ull) {
                        DEV_SAY("tp_gather: GPU %d block %d waited 10 s for GPU %d at gather %u (its flag says %u; %d rows of %d vectors)\n", rank, b, q, s, mine[q * TP_BLOCKS + b], rows, sl8);
                        __trap();
                    }
                }
            }
        }
        __threadfence();
        ctr[b] = s;
    }
    __syncthreads();
    const uint4* stage = reinterpret_cast<const uint4*>(tab[rank]);
    for (int q = 0; q < n; q++) {
        if (q == rank) continue;
        for (long long v = lo + tid; v < hi; v += blockDim.x) {
            const long long r = v / sl8, c = v % sl8;
            dst[r * ld8 + (long long)q * sl8 + c] = __ldcg(&stage[region(q) + v]);
        }
    }
}
