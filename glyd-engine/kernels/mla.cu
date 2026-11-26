// The latent attention layer kind's kernels (src/dynamic/mla.rs runs them for every family with one, Kimi-K3 and GLM-5.3-Flash) and the sparse-attention indexer's,
// whose pick DeepSeek-V4's compressed attention takes as well. Included by engine.cu after glm.cu (GM_PER, GM_DYN_FLOATS) and qwen35.cu (warp_max).
// A token's cache is its latent c (kv_lora_rank wide: the key-value down product's, normed) and the key part the heads share (rope channels, Kimi-K3's 64; GLM-5.3's none), which is its key and, through kv_b_proj, its
// value, so the attention runs in the latent space (the absorbed form): the query q_h becomes q_abs = Wk_h^T q_h, the scores are q_abs . c_t, the output is (sum_t p_t c_t) multiplied by Wv_h at the end. An indexer
// (GLM-5.3's) scores the complete pools of kpool tokens (a pool's key a softmax-weighted mean of its tokens' normed keys) for each query, picks the best ksel = topk / kpool, and the query attends to their tokens and
// to the tokens of the incomplete last pool; a query at position q sees the pools whose last token is q at the latest, and where those are no more than ksel it attends to every token up to q.
// A cache is the executor's ring (src/dynamic/pages.rs): position p's row is tab[p / 16] 16 + p % 16 where the sequence has a block table, else p. Every kernel reads the rows' positions from the device (pos),
// launches over bounds that depend on the engine's context and not on a position, and reads no result on the host: a decode step captures into a CUDA graph. tools/glm_ref.py's mla_* functions are the
// reference (checked there against transformers 5.19.0's Glm5NextTextAttention and Glm5NextTextIndexer).

#define GM_NOPE_MAX 256    // a query head's width (nope) at most
#define GM_IDX_W 8         // pools a block of idx_score scores (a warp each)
#define GM_RB 8            // rows a block of mla_absorb takes
#define GM_VB 4            // rows a block of mla_expand_v takes
#define ML_IDX 128         // an indexer head's channels in the release's arithmetic (DeepSeek-V4's: its Hadamard rotation's size)
#define ML_EC 8            // entries a block of idx_score_bf16 takes (a warp each)
#define ML_IH 64           // idx_score_bf16's heads at most

// Position p's row of a cache: through the sequence's block table (16 positions a block) where it has one, else p.
__device__ __forceinline__ size_t ml_place(const int* tab, int p) { return tab ? (size_t)tab[p >> 4] * 16 + (p & 15) : (size_t)p; }

// Row r's block table: tab + r ts where a step's rows that decode each read their own sequence's (src/dynamic/rows.rs `RowTabs`, ts the entries from one row's table to the
// next's), tab itself where ts is 0 (one sequence's rows); none where tab is null. A row whose position is negative is a graph's padding row: it reads and writes no cache row.
__device__ __forceinline__ const int* ml_row_tab(const int* tab, long long ts, int r) { return tab ? tab + (size_t)r * ts : tab; }

__device__ float block_max(float v) {  // (warp_max is qwen35.cu's)
    __shared__ float shm[32];
    __syncthreads();
    v = warp_max(v);
    if ((threadIdx.x & 31) == 0) shm[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        float t = threadIdx.x < (blockDim.x >> 5) ? shm[threadIdx.x] : -INFINITY;
        t = warp_max(t);
        if (threadIdx.x == 0) shm[0] = t;
    }
    __syncthreads();
    return shm[0];
}

// y row (dst ? the row of position dst[i] : i) = w * bf16(x row i * rsqrt(mean(x^2) + eps)): transformers' RMSNorm (the normed value rounded to bf16, times the weight, rounded again). x rows xs apart, y rows ys apart; K
// up to GM_PER * blockDim.x. With dst (the positions of the rows) the result lands in a cache's row for its position (through row i's table, ml_row_tab; a negative position: none). A block a row,
// 256 threads.
extern "C" __global__ void __launch_bounds__(256) rms2_rows(const bf16* x, int xs, const bf16* w, bf16* y, int ys, const int* dst, const int* tab, long long ts, int K, float eps) {
    if (dst && dst[blockIdx.x] < 0) return;
    const bf16* xr = x + (size_t)blockIdx.x * xs;
    bf16* yr = y + (dst ? ml_place(ml_row_tab(tab, ts, blockIdx.x), dst[blockIdx.x]) : (size_t)blockIdx.x) * ys;
    float v[GM_PER], ss = 0.f;
#pragma unroll
    for (int j = 0; j < GM_PER; j++) {
        const int d = threadIdx.x + j * blockDim.x;
        v[j] = d < K ? b2f(xr[d]) : 0.f;
        ss += v[j] * v[j];
    }
    const float r = rsqrtf(block_sum(ss) / K + eps);
#pragma unroll
    for (int j = 0; j < GM_PER; j++) {
        const int d = threadIdx.x + j * blockDim.x;
        if (d < K) yr[d] = f2b(b2f(w[d]) * b2f(f2b(v[j] * r)));
    }
}

// The same function summed by 512 threads, a thread's columns t, t + 512, ... in that order, each square added by an fma: KimiRMSNorm as Kimi-K3's engine sums it (its q latent's norm, its latent
// experts' norm); any width, for K up to 512 ML_COLS a thread's columns read into registers once (a column a round is slower: the row's loads would follow one another). y row i. A block a row, 512 threads.
#define ML_COLS 16
extern "C" __global__ void __launch_bounds__(512) rms2_rows512(const bf16* x, int xs, const bf16* w, bf16* y, int ys, int K, float eps) {
    const bf16* xr = x + (size_t)blockIdx.x * xs;
    bf16* yr = y + (size_t)blockIdx.x * ys;
    const int tid = threadIdx.x;
    if (blockDim.x == 512 && K <= 512 * ML_COLS) {
        float v[ML_COLS];
#pragma unroll
        for (int k = 0; k < ML_COLS; k++) {
            const int d = tid + 512 * k;
            v[k] = d < K ? b2f(xr[d]) : 0.f;
        }
        float ss = 0.f;
#pragma unroll
        for (int k = 0; k < ML_COLS; k++)
            if (tid + 512 * k < K) ss = fmaf(v[k], v[k], ss);
        const float r = rsqrtf(block_sum(ss) / K + eps);
#pragma unroll
        for (int k = 0; k < ML_COLS; k++) {
            const int d = tid + 512 * k;
            if (d < K) yr[d] = f2b(b2f(w[d]) * b2f(f2b(v[k] * r)));
        }
        return;
    }
    float ss = 0.f;
    for (int d = tid; d < K; d += blockDim.x) {
        const float v = b2f(xr[d]);
        ss = fmaf(v, v, ss);
    }
    const float r = rsqrtf(block_sum(ss) / K + eps);
    for (int d = tid; d < K; d += blockDim.x) yr[d] = f2b(b2f(w[d]) * b2f(f2b(b2f(xr[d]) * r)));
}

// nn.LayerNorm over K (<= 2 blockDim.x) channels: fp32, the mean and the biased variance of the row, (x - mean) * rsqrt(var + eps) * w + b, one rounding; y's row of position dst[i] (through row i's table,
// ml_row_tab; a negative position: none) where dst. A block a row, 128 threads.
extern "C" __global__ void __launch_bounds__(128) layernorm_rows(const bf16* x, int xs, const bf16* w, const bf16* b, bf16* y, int ys, const int* dst, const int* tab, long long ts, int K, float eps) {
    if (dst && dst[blockIdx.x] < 0) return;
    const bf16* xr = x + (size_t)blockIdx.x * xs;
    bf16* yr = y + (dst ? ml_place(ml_row_tab(tab, ts, blockIdx.x), dst[blockIdx.x]) : (size_t)blockIdx.x) * ys;
    float v[2], s = 0.f;
#pragma unroll
    for (int j = 0; j < 2; j++) {
        const int d = threadIdx.x + j * blockDim.x;
        v[j] = d < K ? b2f(xr[d]) : 0.f;
        s += v[j];
    }
    const float mu = block_sum(s) / K;
    float q = 0.f;
#pragma unroll
    for (int j = 0; j < 2; j++) {
        const int d = threadIdx.x + j * blockDim.x;
        const float c = d < K ? v[j] - mu : 0.f;
        q += c * c;
    }
    const float r = rsqrtf(block_sum(q) / K + eps);
#pragma unroll
    for (int j = 0; j < 2; j++) {
        const int d = threadIdx.x + j * blockDim.x;
        if (d < K) yr[d] = f2b((v[j] - mu) * r * b2f(w[d]) + b2f(b[d]));
    }
}

// dst's row of position pos[i] (through row i's table, ml_row_tab; rows ds apart; a negative position: none) = src row i (K bf16 each, src rows ss apart): the rows of a call into a cache at their
// positions; pos null: dst's row i (a 12-bit ring's staging rows). A block a row.
extern "C" __global__ void scatter_rows(const bf16* src, int ss, bf16* dst, int ds, const int* pos, const int* tab, long long ts, int K) {
    if (pos && pos[blockIdx.x] < 0) return;
    const bf16* s = src + (size_t)blockIdx.x * ss;
    bf16* d = dst + (pos ? ml_place(ml_row_tab(tab, ts, blockIdx.x), pos[blockIdx.x]) : (size_t)blockIdx.x) * ds;
    for (int k = threadIdx.x; k < K; k += blockDim.x) d[k] = s[k];
}

// The rotary of interleaved pairs in place (DeepSeek's layout, transformers' apply_rotary_pos_emb_interleave): for row i, at position pos[i], and head h, channels
// off + 2j and off + 2j + 1 (pair j < pairs) of x's row (rows xs apart, heads hs apart) turned by pair j's angle at that position, (a, b) -> (a cos - b sin, b cos + a sin)
// in fp32 and rounded once; cos and sin bf16 [positions, pairs] (the executor's rotary tables, transformers' bf16 cos and sin). The latent attention's queries and shared
// key part, and its indexer's queries and keys, before they are cached or scored. Grid (rows, heads), a thread a pair; a row at a negative position (a graph's padding row) is left.
extern "C" __global__ void mla_rope(bf16* x, int xs, int hs, int off, const int* pos, const bf16* cs, const bf16* sn, int pairs) {
    const int j = threadIdx.x;
    if (j >= pairs || pos[blockIdx.x] < 0) return;
    bf16* p = x + (size_t)blockIdx.x * xs + (size_t)blockIdx.y * hs + off + 2 * j;
    const size_t t = (size_t)pos[blockIdx.x] * pairs + j;
    const float c = b2f(cs[t]), s = b2f(sn[t]), a = b2f(p[0]), b = b2f(p[1]);
    p[0] = f2b(a * c - b * s);
    p[1] = f2b(b * c + a * s);
}

// qa [n, NH, KR] fp32 = for each row and head, Wk_h^T q_h: qa[i][h][j] = sum over d < nope of q[i][h nope + d] * wkv[h (nope + vd) + d][j] (wkv = kv_b_proj, bf16 [NH (nope + vd), KR]). A block a head and GM_RB rows (grid NH x
// ceil(n / GM_RB)), 256 threads: KR / 8 threads a group, each 8 columns of a row of the matrix (one 16-byte load), the groups down the head's nope rows by turns, so that a head's 256 rows are read by 4 groups (KR = 512) or
// 32 (KR = 64) at once (a thread a column would read them one after another); the groups' sums are added in order through shared memory. KR a multiple of 64 up to 512.
// A head's query is qhs wide in q (its nope channels first) and KS wide in qa: past KR its channels nope .. nope + KS - KR copied as they are (Kimi-K3's key part every head shares; GLM: qhs = nope, KS = KR).
// wsc: kv_b_proj released in FP8 (GLM-5.3's): wkv its E4M3 values (exact in bf16) and wsc their fp32 scales of 128 x 128 blocks ([ceil(NH (nope + vd) / 128), ceil(KR / 128)]), each value
// times its block's scale once (the fp32 product the reference takes); null: wkv as it is.
extern "C" __global__ void __launch_bounds__(256) mla_absorb(const bf16* q, int qs, int qhs, const bf16* wkv, const float* wsc, float* qa, int n, int NH, int nope, int vd, int KR, int KS) {
    const int h = blockIdx.x, r0 = blockIdx.y * GM_RB, tid = threadIdx.x;
    const int CT = KR / 8, G = 256 / CT, g = tid / CT, ct = tid % CT;
    const bool live = g < G;  // (KR / 8 need not divide 256: the threads past the last whole group take no rows)
    __shared__ float qh[GM_RB][GM_NOPE_MAX];
    __shared__ float red[1024 * GM_RB];  // [G][GM_RB / 2 rows at a time][KR] floats: G KR = 2048
    for (int e = tid; e < GM_RB * nope; e += blockDim.x) {
        const int r = e / nope, d = e % nope;
        qh[r][d] = r0 + r < n ? b2f(q[(size_t)(r0 + r) * qs + h * qhs + d]) : 0.f;
    }
    __syncthreads();
    float acc[GM_RB][8];
#pragma unroll
    for (int r = 0; r < GM_RB; r++)
#pragma unroll
        for (int c = 0; c < 8; c++) acc[r][c] = 0.f;
    const bf16* W = wkv + (size_t)h * (nope + vd) * KR + ct * 8;
    const int gc = (KR + 127) >> 7;
    for (int d = live ? g : nope; d < nope; d += G) {
        const uint4 u = *reinterpret_cast<const uint4*>(W + (size_t)d * KR);
        float2 w0 = bf2(u.x), w1 = bf2(u.y), w2 = bf2(u.z), w3 = bf2(u.w);
        if (wsc) {  // (the 8 columns are one block's: 8 divides 128)
            const float c = wsc[(size_t)((h * (nope + vd) + d) >> 7) * gc + ((ct * 8) >> 7)];
            w0 = make_float2(__fmul_rn(w0.x, c), __fmul_rn(w0.y, c)), w1 = make_float2(__fmul_rn(w1.x, c), __fmul_rn(w1.y, c));
            w2 = make_float2(__fmul_rn(w2.x, c), __fmul_rn(w2.y, c)), w3 = make_float2(__fmul_rn(w3.x, c), __fmul_rn(w3.y, c));
        }
#pragma unroll
        for (int r = 0; r < GM_RB; r++) {
            const float qv = qh[r][d];
            acc[r][0] = fmaf(qv, w0.x, acc[r][0]);
            acc[r][1] = fmaf(qv, w0.y, acc[r][1]);
            acc[r][2] = fmaf(qv, w1.x, acc[r][2]);
            acc[r][3] = fmaf(qv, w1.y, acc[r][3]);
            acc[r][4] = fmaf(qv, w2.x, acc[r][4]);
            acc[r][5] = fmaf(qv, w2.y, acc[r][5]);
            acc[r][6] = fmaf(qv, w3.x, acc[r][6]);
            acc[r][7] = fmaf(qv, w3.y, acc[r][7]);
        }
    }
    // the groups' sums, GM_RB / 2 rows at a time through shared memory ([G][half][KR] floats: 2048 half floats)
#pragma unroll
    for (int part = 0; part < 2; part++) {
        __syncthreads();
        if (live) {
#pragma unroll
            for (int r = 0; r < GM_RB / 2; r++)
#pragma unroll
                for (int c = 0; c < 8; c++) red[(g * (GM_RB / 2) + r) * KR + ct * 8 + c] = acc[part * (GM_RB / 2) + r][c];
        }
        __syncthreads();
        for (int e = tid; e < (GM_RB / 2) * KR; e += blockDim.x) {
            const int r = e / KR, j = e % KR;
            float v = 0.f;
            for (int gg = 0; gg < G; gg++) v += red[(gg * (GM_RB / 2) + r) * KR + j];
            const int row = r0 + part * (GM_RB / 2) + r;
            if (row < n) qa[((size_t)row * NH + h) * KS + j] = v;
        }
    }
    for (int e = tid; e < GM_RB * (KS - KR); e += blockDim.x) {
        const int r = e / (KS - KR), j = e % (KS - KR), row = r0 + r;
        if (row < n) qa[((size_t)row * NH + h) * KS + KR + j] = b2f(q[(size_t)row * qs + h * qhs + nope + j]);
    }
}

// The keys of the complete pools that this call (rows at pos[0] .. pos[n-1], consecutive) completes, from the normed keys ik and the gate scores gs (D wide, a position's row iks and gss apart, through tab) and the
// pooling bias ape [kp, D]: pool p's key = sum over its kp tokens j of softmax_j(gs[kp p + j][c] + ape[j][c]) * ik[kp p + j][c], rounded to bf16, into pk's row of position p pstep + poff (pks apart, through tab: the
// executor keeps it in its last token's key, pstep = kp, poff = kp - 1, which only this block reads, each channel by the thread that writes it; a caller that keeps each pool's key in a row of its own passes pstep 1, poff 0). A block a pool
// from pos[0] / kp on (grid ceil(n / kp) + 1); a pool whose last token is past the call's last row is left (a later call completes it). Grid y: a step's rows that decode, a sequence's each (n 1): row y's
// position at pos + y, its table ml_row_tab's (gridDim.y 1: one sequence's rows); a negative position (a graph's padding row) makes none.
extern "C" __global__ void __launch_bounds__(128) idx_pool(const bf16* ik, int iks, const bf16* gs, int gss, const bf16* ape, bf16* pk, int pks, int pstep, int poff, const int* tab, long long ts, const int* pos,
                                                           int n, int D, int kp) {
    pos += blockIdx.y;
    tab = ml_row_tab(tab, ts, blockIdx.y);
    if (pos[0] < 0) return;
    const int p = pos[0] / kp + blockIdx.x;
    if ((p + 1) * kp - 1 > pos[n - 1]) return;
    bf16* out = pk + ml_place(tab, p * pstep + poff) * pks;
    for (int c = threadIdx.x; c < D; c += blockDim.x) {
        float l[8], mx = -INFINITY;
        for (int j = 0; j < kp; j++) {
            l[j] = b2f(gs[ml_place(tab, p * kp + j) * gss + c]) + b2f(ape[j * D + c]);
            mx = fmaxf(mx, l[j]);
        }
        float sum = 0.f, acc = 0.f;
        for (int j = 0; j < kp; j++) {
            l[j] = expf(l[j] - mx);
            sum += l[j];
        }
        for (int j = 0; j < kp; j++) acc = fmaf(l[j] / sum, b2f(ik[ml_place(tab, p * kp + j) * iks + c]), acc);
        out[c] = f2b(acc);
    }
}

// scores [n, Pmax] fp32: for each row i (a query at pos[i]) and each complete pool p < P_i = (pos[i] + 1) / kp: sum over the indexer's heads h of w_h relu(scale iq[i][h] . pk[p]), w_h = wscale wts[i][h] (iq rows iqs
// apart, IH heads of D channels; pool p's key at pk's row of position p pstep + poff, pks apart, through tab: as idx_pool wrote it). Rows with P_i <= ksel take every pool and need no scores. A block of GM_IDX_W pools
// (a warp each) of a row, grid (ceil(Pmax / GM_IDX_W), n), 256 threads; the lanes are heads (IH > 32: a lane takes several), so a lane's sum is over its head's D channels and a warp_sum adds the heads. Dynamic shared
// memory (IH (D + 1) + IH + GM_IDX_W D) floats. Row i's keys through its table (ml_row_tab).
extern "C" __global__ void __launch_bounds__(256) idx_score(const bf16* iq, int iqs, const bf16* wts, int wsr, const bf16* pk, int pks, int pstep, int poff, const int* tab, long long ts, const int* pos, float* scores,
                                                            int Pmax, int IH, int D, int kp, int ksel, float scale, float wscale) {
    GM_DYN_FLOATS(smi)
    const int i = blockIdx.y, tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    tab = ml_row_tab(tab, ts, i);
    const int P = (pos[i] + 1) / kp;
    if (P <= ksel || (int)blockIdx.x * GM_IDX_W >= P) return;
    float* qs = smi;                      // [IH][D + 1]
    float* wh = qs + IH * (D + 1);        // [IH]
    float* pkw = wh + IH;                 // [GM_IDX_W][D]
    for (int e = tid; e < IH * D; e += blockDim.x) qs[(e / D) * (D + 1) + e % D] = b2f(iq[(size_t)i * iqs + e]);
    for (int h = tid; h < IH; h += blockDim.x) wh[h] = b2f(wts[(size_t)i * wsr + h]) * wscale;
    const int p = blockIdx.x * GM_IDX_W + w;
    if (p < P) {
        const bf16* key = pk + ml_place(tab, p * pstep + poff) * pks;
        for (int d = lane; d < D; d += 32) pkw[w * D + d] = b2f(key[d]);
    }
    __syncthreads();
    if (p >= P) return;
    float s = 0.f;
    for (int h = lane; h < IH; h += 32) {
        float dot = 0.f;
        for (int d = 0; d < D; d++) dot = fmaf(qs[h * (D + 1) + d], pkw[w * D + d], dot);
        s = fmaf(wh[h], fmaxf(dot * scale, 0.f), s);
    }
    s = warp_sum(s);
    if (lane == 0) scores[(size_t)i * Pmax + p] = s;
}

// The indexer's scores in the release's bf16 arithmetic (DeepSeek-V4's): for row t and each entry e of its cache [entries, 128] (entry e's row through tab) it may read (e < (pos + 1) / ratio), sum over heads h of bf16(relu(bf16(q_h . k_e)) w_h), each product and the sum rounded to
// bf16 as the release's bf16 tensors are, with w_h = bf16(wp_h x wscale) (wp [n, IH], the weights' product, bf16); scores [n, emax] fp32 (-inf past the entries). Block (row, ML_EC entries), 8 warps: a warp an
// entry, a lane 4 of its channels, the queries in shared memory; at most ML_IH heads. Row t's entries through its table (ml_row_tab).
extern "C" __global__ void __launch_bounds__(256) idx_score_bf16(const bf16* q, const bf16* wp, float wscale, const bf16* cache, const int* tab, long long ts, const int* pos, float* scores, int IH, int ratio,
                                                                 int emax) {
    const int t = blockIdx.x, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    tab = ml_row_tab(tab, ts, t);
    extern __shared__ float qs[];  // [IH][128] then [IH] weights, then the warps' terms [8][ML_IH]
    const int ne = (pos[t] + 1) / ratio;
    for (int i = threadIdx.x; i < IH * ML_IDX; i += 256) qs[i] = b2f(q[(size_t)t * IH * ML_IDX + i]);
    for (int i = threadIdx.x; i < IH; i += 256) qs[IH * ML_IDX + i] = b2f(f2b(b2f(wp[(size_t)t * IH + i]) * wscale));
    __syncthreads();
    const int e = blockIdx.y * ML_EC + w;
    if (e >= emax) return;
    if (e >= ne) {
        if (lane == 0) scores[(size_t)t * emax + e] = -INFINITY;
        return;
    }
    const uint2 kv = *reinterpret_cast<const uint2*>(cache + ml_place(tab, e) * ML_IDX + lane * 4);
    const float2 k0 = bf2(kv.x), k1 = bf2(kv.y);
    // (every head's dot product a lane's 4 channels at once, then summed across the lanes as warp_sum's butterfly pairs them, each lane keeping half of its heads a stage: lane l ends with
    // heads 2l and 2l + 1, the same additions in the same order, in 62 shuffles where a warp_sum a head took 320, one after another)
    float v[ML_IH];
#pragma unroll
    for (int h = 0; h < ML_IH; h++) {
        const float* qh = qs + h * ML_IDX + lane * 4;
        v[h] = h < IH ? qh[0] * k0.x + qh[1] * k0.y + qh[2] * k1.x + qh[3] * k1.y : 0.f;
    }
#pragma unroll
    for (int m = ML_IH, o = 16; m > 2; m >>= 1, o >>= 1) {
        const bool hi = lane & o;
#pragma unroll
        for (int i = 0; i < m / 2; i++) {
            const float give = hi ? v[i] : v[i + m / 2];
            const float keep = hi ? v[i + m / 2] : v[i];
            v[i] = keep + __shfl_xor_sync(0xffffffffu, give, o);
        }
    }
    // (the heads' terms, then their sum in the heads' order, as one thread took them)
    float* term = qs + IH * ML_IDX + IH + w * ML_IH;
#pragma unroll
    for (int u = 0; u < 2; u++) {
        const int h = 2 * lane + u;
        if (h < IH) term[h] = b2f(f2b(fmaxf(b2f(f2b(v[u])), 0.f) * qs[IH * ML_IDX + h]));
    }
    __syncwarp();
    if (lane == 0) {
        float sum = 0.f;
        for (int h = 0; h < IH; h++) sum += term[h];
        scores[(size_t)t * emax + e] = b2f(f2b(sum));
    }
}

__device__ __forceinline__ unsigned gm_key(float f) {  // (the order of the floats as an unsigned number's: a larger float, a larger key)
    const unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// The indexer's pick, every family's (GLM-5.3's pools, DeepSeek-V4's compressed entries): the entries row t's query reads, the k of largest score (scores [n, emax]) of its ne = min((pos + 1) / ratio, emax) complete ones
// (all where ne <= k), equal scores the earlier entry first, in the entries' order. Each entry is `expand` positions of sel (entry e: e expand .. e expand + expand - 1: a pool's tokens, or 1: the entry itself), then
// where `tail` the (pos + 1) % ratio tokens of the incomplete entry after them; sel [n, sk] int32, their count into cnt [n]. The k-th largest key found bit by bit (32 counts over the row), then the larger ones and
// the equal ones up to k taken in order by a scan. Block a row, 1,024 threads. `nent` (where not null): each row's entries, in place of its complete ones (the candidate blocks' pick, idx_blocks'); `masked`: the
// entries of score -inf are none of the row's candidates (idx_inside's), so a row of no more than k others takes those alone.
extern "C" __global__ void __launch_bounds__(1024) idx_pick(const float* scores, int emax, const int* pos, int ratio, int k, int expand, int tail, int* sel, int sk, int* cnt, const int* nent, int masked) {
    const int t = blockIdx.x, tid = threadIdx.x, P = (pos[t] + 1) / ratio, ne = nent ? min(nent[t], emax) : min(P, emax), tl = tail ? (pos[t] + 1) % ratio : 0;
    const float* sr = scores + (size_t)t * emax;
    int* out = sel + (size_t)t * sk;
    __shared__ int warp_n[32];
    __shared__ int base;
    if (ne <= k && !masked) {
        const int T = ne * expand + tl;
        for (int e = tid; e < T; e += 1024) out[e] = e;
        if (tid == 0) cnt[t] = T;
        return;
    }
    // the k-th largest key: the largest T with at least k keys >= T (masked, with no more than k candidates: every candidate, the keys above -inf's)
    uint32_t T = 0;
    bool every = false;
    if (masked) {
        int c = 0;
        for (int e = tid; e < ne; e += 1024) c += sr[e] > -INFINITY;
        const int m = (int)block_sum((float)c);
        if (m <= k) {
            k = m;
            T = gm_key(-INFINITY);
            every = true;
        }
    }
    for (int b = 31; b >= 0 && !every; b--) {
        const uint32_t try_ = T | (1u << b);
        int c = 0;
        for (int e = tid; e < ne; e += 1024) c += gm_key(sr[e]) >= try_;
        if ((int)block_sum((float)c) >= k) T = try_;
    }
    int gt_n = 0;
    for (int e = tid; e < ne; e += 1024) gt_n += gm_key(sr[e]) > T;
    const int above = (int)block_sum((float)gt_n);
    int ties = k - above;  // the equal keys taken, the earliest first
    if (tid == 0) base = 0;
    __syncthreads();
    for (int e0 = 0; e0 < ne; e0 += 1024) {
        const int e = e0 + tid;
        const uint32_t key = e < ne ? gm_key(sr[e]) : 0u;
        const bool gt = e < ne && key > T, eq = e < ne && key == T;
        // ranks of this chunk's equal keys, then of its taken ones (a scan over the block by warps)
        const unsigned eqm = __ballot_sync(0xffffffffu, eq);
        if ((tid & 31) == 0) warp_n[tid >> 5] = __popc(eqm);
        __syncthreads();
        int eq_before = 0;
        for (int i = 0; i < (tid >> 5); i++) eq_before += warp_n[i];
        eq_before += __popc(eqm & ((1u << (tid & 31)) - 1));
        int eq_all = 0;
        for (int i = 0; i < 32; i++) eq_all += warp_n[i];
        __syncthreads();
        const bool take = gt || (eq && eq_before < ties);
        const unsigned tm = __ballot_sync(0xffffffffu, take);
        if ((tid & 31) == 0) warp_n[tid >> 5] = __popc(tm);
        __syncthreads();
        int before = base;
        for (int i = 0; i < (tid >> 5); i++) before += warp_n[i];
        before += __popc(tm & ((1u << (tid & 31)) - 1));
        if (take)
            for (int j = 0; j < expand; j++) out[(size_t)before * expand + j] = e * expand + j;
        int all = 0;
        for (int i = 0; i < 32; i++) all += warp_n[i];
        __syncthreads();
        if (tid == 0) base += all;
        ties -= min(ties, eq_all);
        __syncthreads();
    }
    for (int u = tid; u < tl; u += 1024) out[k * expand + u] = P * expand + u;
    if (tid == 0) cnt[t] = k * expand + tl;
}

// The candidate source's blocks (`kinds::CompShare::publish`, DeepSeek-V4.1's CSA2): row t's ne = min((pos + 1) / ratio, emax) entries (scores [n, emax]) in blocks of `block`, each block's score its best entry's,
// the block of its newest entry +inf (kept whatever its score) into bsc [n, bmax], their count into nent [n]: idx_pick then keeps the best `top` of them. Block a row, 256 threads.
extern "C" __global__ void __launch_bounds__(256) idx_blocks(const float* scores, int emax, const int* pos, int ratio, int block, float* bsc, int bmax, int* nent) {
    const int t = blockIdx.x, ne = min((pos[t] + 1) / ratio, emax), nb = min((ne + block - 1) / block, bmax);
    const float* sr = scores + (size_t)t * emax;
    for (int b = threadIdx.x; b < nb; b += blockDim.x) {
        float m = -INFINITY;
        for (int e = b * block; e < min((b + 1) * block, ne); e++) m = fmaxf(m, sr[e]);
        bsc[(size_t)t * bmax + b] = b == nb - 1 ? INFINITY : m;
    }
    if (threadIdx.x == 0) nent[t] = nb;
}

// An index source after the candidate source: row t's entries outside the blocks its query was published (cand [n, top], ncand [n] of them, in order: idx_pick's of idx_blocks') get score -inf (scores
// [n, emax]), out of the row's candidates (idx_pick's `masked`). Block a row, 256 threads.
extern "C" __global__ void __launch_bounds__(256) idx_inside(float* scores, int emax, const int* pos, int ratio, int block, const int* cand, int top, const int* ncand) {
    const int t = blockIdx.x, ne = min((pos[t] + 1) / ratio, emax), nc = ncand[t];
    const int* cr = cand + (size_t)t * top;
    for (int e = threadIdx.x; e < ne; e += blockDim.x) {
        const int b = e / block;
        int lo = 0, hi = nc;
        while (lo < hi) {
            const int mid = (lo + hi) / 2;
            if (cr[mid] < b) lo = mid + 1;
            else hi = mid;
        }
        if (lo == nc || cr[lo] != b) scores[(size_t)t * emax + e] = -INFINITY;
    }
}

// olat [n, NH, KR] fp32 = for each row i and head h, softmax over its cnt[i] tokens t of (qa[i][h] . lat[sel[i][t]]) scale, times the latents: sum_t p_t lat[sel[i][t]] (lat [., KS] bf16, a position's row through tab). A
// block a (head, row, split) (grid NH x n x splits), 256 threads; the row's tokens are cut into `splits` runs, a block takes one: a warp four tokens at a time for the scores (bf16 pairs, KR a multiple of 64; the four
// rows' loads are in flight together, where one at a time each waits for the one before), the softmax in fp32 over the block's tokens, then a thread two columns of the weighted sum, four tokens at a time.
// With one split the block writes olat; with more, its unnormalised sum, its largest score and the sum of its exponentials to part [n][NH][splits][KR + 2] for mla_merge. Dynamic shared memory (KS + 2 tile) 4-byte
// words: a block's tokens `tile` at a time (Kimi-K3's dense attention over the whole context).
// The scores are over KS channels (qa's heads KS wide, lat's rows ls >= KS apart), the values the first KR of them (Kimi-K3: the latent's 512 and the shared key part's 64; GLM: KS = KR); sel null: every
// position from 0 (a dense attention, cnt[i] = pos + 1). lat is a ring (ls = KS) or a 12-bit ring's pages unpacked to bf16 (ls its padded width, tab null): the same values in the same order, the same bits.
// Row i's positions through its table (ml_row_tab: a step's rows that decode each through its own sequence's).
extern "C" __global__ void __launch_bounds__(256) mla_attn(const float* qa, const bf16* lat, int ls, const int* tab, long long ts, const int* sel, const int* cnt, float* olat, float* part, int Tmax, int NH, int KR,
                                                           int KS, float scale, int splits, int tile) {
    GM_DYN_FLOATS(sma)
    const int h = blockIdx.x, i = blockIdx.y, sp = blockIdx.z, tid = threadIdx.x, lane = tid & 31, w = tid >> 5, nw = blockDim.x >> 5;
    tab = ml_row_tab(tab, ts, i);
    const int T = cnt[i], per = (T + splits - 1) / splits, t0 = min(T, sp * per), t1 = min(T, t0 + per), m_n = t1 - t0;
    float* qh = sma;                                // [KS]
    float* sc = sma + KS;                           // [tile] a tile's scores, then its weights
    int* idm = reinterpret_cast<int*>(sma + KS + tile);  // [tile] the rows of the tile's tokens (a copy: the rows' loads then wait for no other load)
    const int* ids = sel ? sel + (size_t)i * Tmax + t0 : nullptr;
    for (int j = tid; j < KS; j += blockDim.x) qh[j] = qa[((size_t)i * NH + h) * KS + j];
    // the block's tokens a tile at a time (the most the shared memory holds), the softmax's largest score, its sum and the weighted sum carried over and
    // rescaled where a tile's largest is larger: one tile, the arithmetic of a single pass
    float m = -INFINITY, z = 0.f, a0 = 0.f, a1 = 0.f;
    const int j2 = 2 * tid;
    for (int c0 = 0; c0 < m_n; c0 += tile) {
        const int cn = min(tile, m_n - c0);
        for (int t = tid; t < cn; t += blockDim.x) idm[t] = (int)ml_place(tab, ids ? ids[c0 + t] : t0 + c0 + t);
        __syncthreads();
        for (int tb = w * 4; tb < cn; tb += nw * 4) {
            const bf16* rows[4];
#pragma unroll
            for (int u = 0; u < 4; u++) rows[u] = lat + (size_t)idm[min(tb + u, cn - 1)] * ls;
            float a[4] = {0.f, 0.f, 0.f, 0.f};
            for (int j = 2 * lane; j < KS; j += 64) {
                float2 c[4];
#pragma unroll
                for (int u = 0; u < 4; u++) c[u] = bf2(*reinterpret_cast<const uint32_t*>(rows[u] + j));
                const float q0 = qh[j], q1 = qh[j + 1];
#pragma unroll
                for (int u = 0; u < 4; u++) a[u] = fmaf(q0, c[u].x, fmaf(q1, c[u].y, a[u]));
            }
#pragma unroll
            for (int u = 0; u < 4; u++) {
                const float v = warp_sum(a[u]);
                if (lane == 0 && tb + u < cn) sc[tb + u] = v * scale;
            }
        }
        __syncthreads();
        float mt = -INFINITY;
        for (int t = tid; t < cn; t += blockDim.x) mt = fmaxf(mt, sc[t]);
        mt = fmaxf(m, block_max(mt));
        const float r = m == -INFINITY ? 0.f : expf(m - mt);
        float zt = 0.f;
        for (int t = tid; t < cn; t += blockDim.x) {
            const float e = expf(sc[t] - mt);
            sc[t] = e;
            zt += e;
        }
        z = z * r + block_sum(zt);
        a0 *= r;
        a1 *= r;
        m = mt;
        // the weighted sum: a thread a pair of columns (KR <= 512), eight tokens' loads in flight
        for (int tb = 0; tb < cn; tb += 8) {
            float2 v[8];
            float p[8];
#pragma unroll
            for (int u = 0; u < 8; u++) {
                const bool ok = tb + u < cn;
                p[u] = ok ? sc[tb + u] : 0.f;
                v[u] = j2 < KR ? bf2(*reinterpret_cast<const uint32_t*>(lat + (size_t)idm[ok ? tb + u : cn - 1] * ls + j2)) : make_float2(0.f, 0.f);
            }
#pragma unroll
            for (int u = 0; u < 8; u++) {
                a0 = fmaf(p[u], v[u].x, a0);
                a1 = fmaf(p[u], v[u].y, a1);
            }
        }
        __syncthreads();
    }
    if (splits == 1) {
        const float inv = z > 0.f ? 1.f / z : 0.f;  // (no tokens at all: zeros, as the reference's)
        float* o = olat + ((size_t)i * NH + h) * KR;
        if (j2 < KR) {
            o[j2] = a0 * inv;
            o[j2 + 1] = a1 * inv;
        }
    } else {
        float* o = part + (((size_t)i * NH + h) * splits + sp) * (KR + 2);
        if (j2 < KR) {
            o[j2] = a0;
            o[j2 + 1] = a1;
        }
        if (tid == 0) {
            o[KR] = m;
            o[KR + 1] = z;
        }
    }
}

// olat [n, NH, KR] = the splits of mla_attn merged: with M the largest of the splits' largest scores, sum_s exp(m_s - M) o_s / sum_s exp(m_s - M) z_s (a split without tokens has m = -inf and z = 0). A block a (head,
// row), 256 threads.
extern "C" __global__ void __launch_bounds__(256) mla_merge(const float* part, float* olat, int NH, int KR, int splits) {
    const int h = blockIdx.x, i = blockIdx.y;
    const float* p = part + ((size_t)i * NH + h) * splits * (KR + 2);
    float M = -INFINITY;
    for (int sp = 0; sp < splits; sp++) M = fmaxf(M, p[(size_t)sp * (KR + 2) + KR]);
    if (M == -INFINITY) M = 0.f;  // (no tokens in any split: zeros)
    float den = 0.f;
    for (int sp = 0; sp < splits; sp++) den += expf(p[(size_t)sp * (KR + 2) + KR] - M) * p[(size_t)sp * (KR + 2) + KR + 1];
    for (int j = threadIdx.x; j < KR; j += blockDim.x) {
        float v = 0.f;
        for (int sp = 0; sp < splits; sp++) v += expf(p[(size_t)sp * (KR + 2) + KR] - M) * p[(size_t)sp * (KR + 2) + j];
        olat[((size_t)i * NH + h) * KR + j] = den > 0.f ? v / den : 0.f;
    }
}

// ov [n, NH vd] bf16 = for each row and head, olat [n, NH, KR] times the head's value rows of kv_b_proj: ov[i][h vd + e] = sum_j olat[i][h][j] wkv[h (nope + vd) + nope + e][j], fp32 sums, one rounding. A block a head and
// GM_VB rows (grid NH x ceil(n / GM_VB)), 256 threads: a warp a value channel e at a time, its lanes down the row of the matrix (KR a multiple of 64). Dynamic shared memory GM_VB KR floats.
// wsc: kv_b_proj's fp32 block scales where it is released in FP8 (mla_absorb's), each value times its block's scale once; null: wkv as it is.
extern "C" __global__ void __launch_bounds__(256) mla_expand_v(const float* olat, const bf16* wkv, const float* wsc, bf16* ov, int n, int NH, int nope, int vd, int KR) {
    GM_DYN_FLOATS(smv)
    const int h = blockIdx.x, r0 = blockIdx.y * GM_VB, tid = threadIdx.x, lane = tid & 31, w = tid >> 5, nw = blockDim.x >> 5;
    for (int e = tid; e < GM_VB * KR; e += blockDim.x) {
        const int r = e / KR, j = e % KR;
        smv[e] = r0 + r < n ? olat[((size_t)(r0 + r) * NH + h) * KR + j] : 0.f;
    }
    __syncthreads();
    const int gc = (KR + 127) >> 7;
    for (int e = w; e < vd; e += nw) {
        const bf16* row = wkv + ((size_t)h * (nope + vd) + nope + e) * KR;
        const float* rsc = wsc ? wsc + (size_t)((h * (nope + vd) + nope + e) >> 7) * gc : nullptr;
        float a[GM_VB];
#pragma unroll
        for (int r = 0; r < GM_VB; r++) a[r] = 0.f;
        for (int j = 2 * lane; j < KR; j += 64) {
            float2 c = bf2(*reinterpret_cast<const uint32_t*>(row + j));
            if (rsc) c = make_float2(__fmul_rn(c.x, rsc[j >> 7]), __fmul_rn(c.y, rsc[j >> 7]));  // (j and j + 1 one block's)
#pragma unroll
            for (int r = 0; r < GM_VB; r++) a[r] = fmaf(smv[r * KR + j], c.x, fmaf(smv[r * KR + j + 1], c.y, a[r]));
        }
#pragma unroll
        for (int r = 0; r < GM_VB; r++) {
            const float v = warp_sum(a[r]);
            if (lane == 0 && r0 + r < n) ov[(size_t)(r0 + r) * NH * vd + h * vd + e] = f2b(v);
        }
    }
}

// y [m, N] (rows ys apart) = x [m, K] (rows xs apart) W^T for W released in FP8 (an indexer's products, GLM-5.3's): W its E4M3 values (exact in bf16) and wsc their fp32
// scales of 128 x 128 blocks ([ceil(N / 128), ceil(K / 128)]), each value times its block's scale once (the fp32 product the reference takes), fp32 sums, one rounding.
// small_linear's grid and order of sums (a warp an output, 8 a block; grid m x ceil(N / 8)), 256 threads.
extern "C" __global__ void __launch_bounds__(256) small_linear_f8s(const bf16* x, int xs, const bf16* W, const float* wsc, bf16* y, int ys, int N, int K) {
    const bf16* xr = x + (size_t)blockIdx.x * xs;
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5, gc = (K + 127) >> 7;
    const bool wide = (K & 7) == 0 && ((reinterpret_cast<uintptr_t>(xr) | reinterpret_cast<uintptr_t>(W)) & 15) == 0;
    for (int o = blockIdx.y * nw + w; o < N; o += nw * gridDim.y) {
        const bf16* wr = W + (size_t)o * K;
        const float* rsc = wsc + (size_t)(o >> 7) * gc;
        float s = 0.f;
        if (wide) {
            const uint4* xv = reinterpret_cast<const uint4*>(xr);
            const uint4* wv = reinterpret_cast<const uint4*>(wr);
            for (int k = lane; k < (K >> 3); k += 32) {
                const uint4 a = xv[k], b = wv[k];
                const float c = rsc[k >> 4];  // (columns 8k .. 8k + 7: one block's)
                const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a);
                const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&b);
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const float2 fa = __bfloat1622float2(a2[j]), fb = __bfloat1622float2(b2[j]);
                    s = fmaf(fa.x, __fmul_rn(fb.x, c), s);
                    s = fmaf(fa.y, __fmul_rn(fb.y, c), s);
                }
            }
        } else {
            for (int k = lane; k < K; k += 32) s = fmaf(b2f(xr[k]), __fmul_rn(b2f(wr[k]), rsc[k >> 7]), s);
        }
        s = warp_sum(s);
        if (lane == 0) y[(size_t)blockIdx.x * ys + o] = f2b(s);
    }
}

// o [n, W] = o times sigmoid(g) (g [n, gs] row stride gs), the sigmoid rounded to bf16, the product once: the output gate (Kimi-K3's). A thread a value.
extern "C" __global__ void mla_gate(bf16* o, const bf16* g, int gs, int W, int n) {
    const size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= (size_t)n * W) return;
    const size_t r = e / W, c = e % W;
    const float s = b2f(f2b(1.f / (1.f + expf(-b2f(g[r * gs + c])))));
    o[e] = f2b(b2f(o[e]) * s);
}

// list [maxp] = the blocks of a sequence's first pos[0] / 16 + 1 pages through its table tab, -1 past them: the pages a 12-bit latent ring unpacks for rows whose last position is
// pos[0] (glyd_gpu_kv_unpack skips a negative page), from a position read on the device (a decode step's graph holds it). A thread an entry.
extern "C" __global__ void ml_pages(const int* tab, const int* pos, int maxp, int* list) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < maxp) list[j] = j <= (pos[0] >> 4) ? tab[j] : -1;
}

// cnt [n] = pos[i] + 1: a dense attention's token counts (every position up to the row's own; mla_attn without a selection).
extern "C" __global__ void mla_dense_cnt(const int* pos, int* cnt, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) cnt[i] = pos[i] + 1;
}
