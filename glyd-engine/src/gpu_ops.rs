//! R17 step 0 (#238, glyd-engine/docs/design/one-kernel-build.md) and R1a (#134): the GPU library's operations as one trait, and the dense
//! products as another, so that the engine calls the traits and not the library's table.
//!
//! `GpuOps` is every operation the engine runs of Glyd's GPU library (gpu/glyd_gpu.cu): the packed matrices' products, routes and unpacking,
//! the grouped experts' products, the lossless FP8 format, and the paged lossless KV cache. Its first implementation is the library itself,
//! loaded at run time (`ffi::Lib`, over the C API's table), unchanged. R17 folds the library into the engine's own kernels one family of
//! operations at a time (the mma products and their unpackers, the grouped experts, the paged KV cache, FP8); each family's new
//! implementation is checked bit for bit against the library's on the same inputs, through this trait, before the engine takes it.
//!
//! `Gemm` is the dense product of a matrix decoded to bf16: NVIDIA's cuBLAS where the machine has it (`ffi::Blas`, loaded at run time,
//! never shipped); V8a (#255) adds the libraries that are faster where they are (glyd-engine/docs/design/kernel-libraries.md).
//!
//! The device layer under both is `device::Device`.

use crate::ffi::{KvGeom, Pack, Result, Stream};

/// The GPU library's operations, on the current GPU. Device memory is a `u64` address (the driver's device pointer).
pub trait GpuOps {
    /// The current device's class code (compute capability and class by name, glyd_gpu.h's routes).
    fn gpu(&self) -> Result<i32>;

    /// The route the library takes W [o, k] for `m` tokens on `gpu` (glyd_gpu.h's GLYD_GPU_ROUTE_*), in the 12-bit layout or the tiered one.
    fn route(&self, gpu: i64, twelve: bool, o: i64, k: i64, m: i64) -> Result<i32>;

    /// The workspace bytes of Y [m, o] = X W^T by the current GPU's route for m.
    fn linear_workspace(&self, twelve: bool, o: i64, k: i64, m: i64) -> Result<usize>;

    /// Y [m, rows] = X [m, cols] W^T on `cs` (route: the current GPU's for m). `done`: int32 counters, zero before the first call.
    ///
    /// # Safety
    /// Device addresses of live buffers of the sizes the C API asks for (glyd_gpu.h); `done` and the workspace serve one stream at a time.
    #[allow(clippy::too_many_arguments)]
    unsafe fn linear(&self, w: &Pack, x: u64, m: i64, y: u64, ws: u64, ws_bytes: usize, done: u64, cs: Stream) -> Result<()>;

    /// The lossless FP8 format's size for W [o, k] of E4M3 bytes on the device (rows `ld` bytes apart): its heads into `heads` [patches x 8] and the words its variable part takes (glyd_gpu.h's f8 functions;
    /// waits for the stream). `h`: layout H (glyd_gpu.h's f8h functions, for HBM GPUs) here and in the calls below.
    ///
    /// # Safety
    /// Live device buffers of those sizes; `ws` holds `f8_size_workspace` bytes.
    #[allow(clippy::too_many_arguments)]
    unsafe fn f8_size(&self, h: bool, w: u64, ld: i64, o: i64, k: i64, heads: u64, ws: u64, ws_bytes: usize, cs: Stream) -> Result<i64>;

    /// Layout H's packs for the process: plain (Some(true)), coded (Some(false)) or the GPU's own (None: plain on an HBM GPU); before any pack.
    fn f8h_plain(&self, plain: Option<bool>) -> Result<()>;

    fn f8_size_workspace(&self, o: i64, k: i64) -> Result<usize>;

    /// W's patches into `core` and `var` from the heads `f8_size` made.
    ///
    /// # Safety
    /// Live device buffers of those sizes.
    #[allow(clippy::too_many_arguments)]
    unsafe fn f8_pack(&self, h: bool, w: u64, ld: i64, o: i64, k: i64, heads: u64, core: u64, var: u64, var_words: i64, cs: Stream) -> Result<()>;

    /// W back, bit for bit, into `out` [o, k] (rows `ld` bytes apart).
    ///
    /// # Safety
    /// Live device buffers of those sizes.
    #[allow(clippy::too_many_arguments)]
    unsafe fn f8_unpack(&self, h: bool, core: u64, heads: u64, var: u64, o: i64, k: i64, out: u64, ld: i64, cs: Stream) -> Result<()>;

    fn f8_linear_workspace(&self, n: i64, k: i64, m: i64) -> Result<usize>;

    /// Y [m, n] = X [m, k] W^T for m of 1 to 64 and W in the lossless FP8 format with its scales (`kind`, `scale`, `lds`: glyd_gpu.h's f8_linear), X and Y bf16 with rows k and n elements apart.
    ///
    /// # Safety
    /// Live device buffers of those sizes; `done` ceil(n / 128) zeroed counters.
    #[allow(clippy::too_many_arguments)]
    unsafe fn f8_linear(
        &self,
        h: bool,
        core: u64,
        heads: u64,
        var: u64,
        n: i64,
        k: i64,
        kind: i64,
        scale: u64,
        lds: i64,
        x: u64,
        m: i64,
        y: u64,
        ws: u64,
        ws_bytes: usize,
        done: u64,
        cs: Stream,
    ) -> Result<()>;

    fn f8_moe_workspace(&self, n: i64, k: i64, m: i64, slots: i64) -> Result<usize>;

    /// The experts' product (glyd_gpu.h's f8_moe): `experts` matrices [n, k] packed as one [experts n, k] with their block scales, each pair of `plan` (moe_way's) times its expert's matrix into its row of
    /// y [pairs, n] in the plan's order; x's row of a pair its token's (`gather`: x [tokens, k], `k_` choices a token) or its own in the plan's order (x [pairs, k]). `m`: the most pairs an expert has (1 to 64),
    /// `slots`: the most experts the plan hits.
    ///
    /// # Safety
    /// Live device buffers of those sizes; `done` slots ceil(n / 128) zeroed counters.
    #[allow(clippy::too_many_arguments)]
    unsafe fn f8_moe(
        &self,
        h: bool,
        core: u64,
        heads: u64,
        var: u64,
        experts: i64,
        n: i64,
        k: i64,
        scale: u64,
        x: u64,
        plan: u64,
        k_: i64,
        gather: bool,
        m: i64,
        slots: i64,
        y: u64,
        ws: u64,
        ws_bytes: usize,
        done: u64,
        cs: Stream,
    ) -> Result<()>;

    /// Rows [row0, row0 + rows) of W back to bf16 into `out` [rows, cols] (both multiples of 64).
    ///
    /// # Safety
    /// As `linear`.
    unsafe fn unpack(&self, w: &Pack, row0: i64, rows: i64, out: u64, cs: Stream) -> Result<()>;

    /// The plan (int32 [2 + 2 experts + pairs]) of `pairs` pairs sorted by expert, `ids` (int64 [pairs]) each pair's expert (outside 0 to experts - 1: routed nowhere): the experts hit and where each one's
    /// pairs start in the order, then the pairs by expert, in order within each. A token's k choices are its pairs (pair j: token j / k, choice j mod k). Experts up to 12,288.
    ///
    /// # Safety
    /// As `linear`; `plan` holds 2 + 2 experts + pairs int32.
    unsafe fn moe_way(&self, ids: u64, pairs: i64, experts: i64, plan: u64, cs: Stream) -> Result<()>;

    /// The workspace bytes of `moe`'s product of `experts` experts' matrices [o, cols] for `t` tokens of `k` choices (act 0, 1 or 2; `weighted`: with the choices' weights), on the current GPU.
    #[allow(clippy::too_many_arguments)]
    fn moe_workspace(&self, twelve: bool, experts: i64, o: i64, cols: i64, t: i64, k: i64, act: i64, weighted: bool) -> Result<usize>;

    /// The experts' product over `w`, the experts' matrices [o, cols] stacked as one pack [experts o, cols] (`experts` of them), by the plan of `moe_way`: x [t, cols] (`gather`: a pair takes its token's row) or [t k, cols]
    /// (the pairs in the plan's order). `act` 0: y [t k, o] in the plan's order; 1 (SiLU) or 2 (GELU, tanh): y [t k, o / 2] = act(gate) up, the gate an expert's first o / 2 rows (o a multiple of 128).
    /// `weights` (fp32 [t, k], act 0; 0: none) with `ids` (int64 [t, k]): y [t, o], each token's k rows times their weights, added in fp32. `done`: int32 counters, zero before, left zero.
    ///
    /// `whole`: an expert's rows and the experts of the whole layer where `w` holds a part of it (a GPU's slice of each expert's rows, an expert cache's
    /// places; 0: its own), so the work is cut as the whole layer's call cuts it and each row's sum is the whole layer's, bit for bit.
    ///
    /// # Safety
    /// As `linear`; the workspace is `moe_workspace`'s bytes; the counters are o / 64 (o / 128 with an act) times min(experts, t k).
    #[allow(clippy::too_many_arguments)]
    unsafe fn moe(
        &self,
        w: &Pack,
        experts: i64,
        x: u64,
        t: i64,
        k: i64,
        gather: bool,
        plan: u64,
        act: i64,
        weights: u64,
        ids: u64,
        y: u64,
        ws: u64,
        ws_bytes: usize,
        done: u64,
        cs: Stream,
        whole: (i64, i64),
    ) -> Result<()>;

    /// `moe` over an expert cache's places of a tiered-layout stack (glyd_gpu_mma_moe_placed: each place its own block entries, `stride` bytes of blocks a place); the weights, ids and the rest as `moe`'s.
    ///
    /// # Safety
    /// As `moe`; `w`'s block entries [places][steps an expert + 1] and blocks [places][stride].
    #[allow(clippy::too_many_arguments)]
    unsafe fn moe_placed(
        &self,
        w: &Pack,
        stride: u64,
        experts: i64,
        x: u64,
        t: i64,
        k: i64,
        gather: bool,
        plan: u64,
        act: i64,
        weights: u64,
        ids: u64,
        y: u64,
        ws: u64,
        ws_bytes: usize,
        done: u64,
        cs: Stream,
        whole: (i64, i64),
    ) -> Result<()>;

    /// A page's bytes for `h` heads of `d` channels with an arena of `a` entries.
    fn kv_page_bytes(&self, h: i64, d: i64, a: i64) -> Result<i64>;

    fn kv_escape_bytes(&self, nchunks: i64) -> Result<i64>;

    fn kv_raw_bytes(&self, nslabs: i64, h: i64) -> Result<i64>;

    /// `n` tokens' keys and values [n, H, D] (token t's row at key + t * key_stride elements) into the layer's pool at `slots` [n] (int64).
    ///
    /// # Safety
    /// As `linear`; `status` is uint32 [16 + 2 ring] zeroed once; `lost` is uint32 words of `nlost` bits (two to a block), zeroed once, or 0 and 0.
    #[allow(clippy::too_many_arguments)]
    unsafe fn kv_write(
        &self,
        g: &KvGeom,
        pool: u64,
        kbase: u64,
        vbase: u64,
        layer: i64,
        key: u64,
        key_stride: i64,
        value: u64,
        value_stride: i64,
        slots: u64,
        n: i64,
        status: u64,
        lost: u64,
        nlost: i64,
        ring: i64,
        cs: Stream,
    ) -> Result<()>;

    /// The pages `pages` [n] (int32, block numbers) of the layer's pool decoded into two bf16 tensors, keys `out_k` and values `out_v`, each [n, 16, H, D] (page j's tokens at j * 16 on: NHD, as a paged attention reads a cache of separate keys and values).
    ///
    /// # Safety
    /// As `kv_write`; `pages` is an int32 array of n on the device, `out_k` and `out_v` hold n pages of 16 * H * D values each.
    #[allow(clippy::too_many_arguments)]
    unsafe fn kv_unpack(&self, g: &KvGeom, pool: u64, kbase: u64, vbase: u64, pages: u64, n: i64, out_k: u64, out_v: u64, cs: Stream) -> Result<()>;

    /// The pages `pages` [n] (int32, block numbers) of the layer's pool decoded into the bf16 staging cache `out` [blocks, 16, H, 2 D], page j into block `dst`[j].
    ///
    /// # Safety
    /// As `kv_write`; `pages` and `dst` are int32 arrays of n on the device, `out` holds a block for every dst.
    #[allow(clippy::too_many_arguments)]
    unsafe fn kv_gather(&self, g: &KvGeom, pool: u64, kbase: u64, vbase: u64, pages: u64, dst: u64, n: i64, out: u64, cs: Stream) -> Result<()>;

    /// The most CTAs a (sequence, head) is split over, for `b` sequences of at most `max_pages` pages.
    fn kv_decode_splits(&self, b: i64, h: i64, max_pages: i64) -> Result<i64>;

    /// The decode attention's least CTAs a sequence (1 to 64; 0: the GPU's own), for the whole process.
    fn kv_floor_set(&self, floor: i32) -> Result<()>;

    fn kv_decode_workspace(&self, b: i64, h: i64, g: i64, d: i64, splits: i64) -> Result<usize>;

    /// Attention for `b` sequences: query q [H G, D] at q + i * q_stride, `seq_lens` [b] (int32) tokens in the pages of
    /// `block_table` (row i at + i * bt_stride, int32), out [H G, D] at out + i * out_stride.
    ///
    /// # Safety
    /// As `kv_write`; the workspace is zero when made and left so.
    #[allow(clippy::too_many_arguments)]
    unsafe fn kv_decode(
        &self,
        g: &KvGeom,
        group: i64,
        pool: u64,
        kbase: u64,
        vbase: u64,
        q: u64,
        q_stride: i64,
        block_table: u64,
        bt_stride: i64,
        seq_lens: u64,
        b: i64,
        splits: i64,
        scale: f64,
        out: u64,
        out_stride: i64,
        ws: u64,
        ws_bytes: usize,
        cs: Stream,
    ) -> Result<()>;

    /// `kv_decode` over each sequence's tokens from `first` (int32 [b], a sliding window's start; 0: none, the whole sequence) to its length; `round`: each
    /// score the bf16 of its dot product (a bf16 matmul's, Gemma 4's attention).
    ///
    /// # Safety
    /// As `kv_decode`; `first`, where not 0, holds b int32.
    #[allow(clippy::too_many_arguments)]
    unsafe fn kv_decode_from(
        &self,
        g: &KvGeom,
        group: i64,
        pool: u64,
        kbase: u64,
        vbase: u64,
        q: u64,
        q_stride: i64,
        block_table: u64,
        bt_stride: i64,
        seq_lens: u64,
        first: u64,
        round: bool,
        b: i64,
        splits: i64,
        scale: f64,
        out: u64,
        out_stride: i64,
        ws: u64,
        ws_bytes: usize,
        cs: Stream,
    ) -> Result<()>;

    /// A prompt chunk's attention (glyd_gpu.h's `glyd_gpu_kv_prefill`): `n` rows of one sequence at positions `start` .. `start + n`, queries `q` [n, H G, D] (row t at q + t
    /// q_stride elements), causal over the keys from each row's window (`window` tokens, 0: none) to itself: those before `start` read from the pages `block_table` (int32: the
    /// sequence's row of the table) names, the chunk's own from its bf16 rows `key`, `value` [n, H, D] (0: from the pages too, written before); `round`: each score the bf16 of its dot
    /// product (Gemma 4's), `softcap` 0: none, `sinks` 0 or float [H G] on the device; out [n, H G, D].
    ///
    /// # Safety
    /// As `kv_decode`; `block_table` holds a page for every token before `start + n` that a row's window takes.
    #[allow(clippy::too_many_arguments)]
    unsafe fn kv_prefill(
        &self,
        g: &KvGeom,
        group: i64,
        pool: u64,
        kbase: u64,
        vbase: u64,
        block_table: u64,
        q: u64,
        q_stride: i64,
        rows: Option<(u64, i64, u64, i64)>,
        n: i64,
        start: i64,
        window: i64,
        scale: f64,
        round: bool,
        softcap: f64,
        sinks: u64,
        out: u64,
        out_stride: i64,
        cs: Stream,
    ) -> Result<()>;

    /// `kv_decode_from`'s attention with the keys and values in a bf16 cache `kc`, `vc` [places, H, D] (glyd_gpu.h's `glyd_gpu_kv_decode_bf16`): token p of sequence b at
    /// `block_table[b bt_stride + p / 16] * 16 + p % 16` (0: p modulo `cap`); `b` sequences of `g` query heads a kv head, their lengths and first tokens on the device.
    /// The same kernel, plan and merge: with the same splits and floor, `kv_decode_from`'s bits where the cache holds what the pages hold.
    ///
    /// # Safety
    /// As `kv_decode_from`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn kv_decode_bf16(
        &self,
        q: u64,
        q_stride: i64,
        cache: (u64, u64, u64, i64, i64),
        seqs: (u64, u64, i64, i64),
        heads: (i64, i64, i64),
        scores: (f64, bool),
        out: u64,
        out_stride: i64,
        workspace: (u64, usize),
        cs: Stream,
    ) -> Result<()>;

    /// `kv_prefill`'s attention with the keys and values in a bf16 cache `kc`, `vc` [places, H, D] (glyd_gpu.h's `glyd_gpu_kv_prefill_bf16`): position p's place
    /// `block_table[p / 16] * 16 + p % 16` (0: p modulo `cap`), the chunk's own written there before the call; `g` query heads a kv head. The same kernel and sums:
    /// where the cache holds what the pages hold, `kv_prefill`'s bits.
    ///
    /// # Safety
    /// Live device buffers: q and out [n, H g, D] (rows `q_stride`, `out_stride` apart), the cache's every place a row reads, the table's blocks.
    #[allow(clippy::too_many_arguments)]
    unsafe fn kv_prefill_bf16(
        &self,
        q: u64,
        q_stride: i64,
        cache: (u64, u64, u64, i64),
        rows: (i64, i64, i64),
        heads: (i64, i64, i64),
        scores: (f64, bool, f64),
        out: u64,
        out_stride: i64,
        cs: Stream,
    ) -> Result<()>;
}

/// The dense product of bf16 matrices.
pub trait Gemm {
    /// Y [m, o] (row-major bf16) = X [m, k] (row-major bf16) W^T, W [o, k] row-major bf16: bf16 in and out, fp32 sums.
    ///
    /// # Safety
    /// Device addresses of live buffers of those sizes.
    #[allow(clippy::too_many_arguments)]
    unsafe fn gemm_nt(&self, s: Stream, w: u64, x: u64, y: u64, o: i32, k: i32, m: i32) -> Result<()>;
}

/// V8a's row for the dense product of a matrix decoded to bf16 (glyd-engine/docs/design/kernel-libraries.md), on a GPU of compute capability `cc`:
/// (whether to take cuBLAS where the machine has it, whether Glyd's own product may take it). cuBLAS, but on Ada (8.9) Glyd's own (an RTX 4080 SUPER's
/// 2,000-token Qwen3-8B pass: own 353 ms against vLLM's 357 on cuBLAS, `benchmarks/gpu/rtx4080s-gemm-tiles-2026-11-20`; an A100's Qwen3-8B prompt of
/// 498 / 1,001 / 4,003 tokens 10% / 4% / 2% faster by cuBLAS, `benchmarks/gpu/a100-device-seam-2026-11-24`); Glyd's own on Ampere and
/// Hopper where cuBLAS is not there; neither elsewhere (the fused kernels keep the product). `want` (GLYD_ENGINE_GEMM, a development build's) "blas" or
/// "own" takes that one (another value: neither). A row changes where its A/B on that GPU measures the other faster.
pub fn dense_choice(cc: (i32, i32), want: &str) -> (bool, bool) {
    match want {
        "blas" => (true, false),
        "own" => (false, true),
        "" => (cc != (8, 9), matches!(cc.0, 8 | 9)),
        _ => (false, false),
    }
}

#[cfg(test)]
mod tests {
    /// The dense product's row as the executor took it before the table: cuBLAS off Ada, Glyd's own on Ada and where cuBLAS is not there (Ampere,
    /// Hopper), neither on other GPUs without cuBLAS; GLYD_ENGINE_GEMM takes one.
    #[test]
    fn the_dense_products_row() {
        use super::dense_choice;
        assert_eq!(dense_choice((8, 9), ""), (false, true));
        assert_eq!(dense_choice((8, 0), ""), (true, true));
        assert_eq!(dense_choice((9, 0), ""), (true, true));
        assert_eq!(dense_choice((12, 0), ""), (true, false));
        assert_eq!(dense_choice((8, 9), "blas"), (true, false));
        assert_eq!(dense_choice((12, 0), "own"), (false, true));
    }
}
