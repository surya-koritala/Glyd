//! The engine's own kernels (kernels/engine.cu) with typed launches. Device addresses are `u64`.

use crate::dev::{Dev, Kernel};
use crate::ffi::{Result, Stream};
use crate::launch;

// The kernels' names: those kernels/engine.cu gives them, in every build.
macro_rules! kname {
    ($n:literal) => {
        $n
    };
}

/// Candidates the sampling kernel returns for a row of logits (kernels/engine.cu).
pub const TOPK: usize = 64;

/// Threads of a norm's block: 1024 for a row of 1024 or more (a thread holds at most 8 of its elements), else the row's length up to a warp.
pub(crate) fn norm_threads(k: i32) -> Result<u32> {
    if !(1..=8192).contains(&k) {
        return crate::ffi::err(format!("a norm over {k} elements: not supported"));
    }
    Ok(if k >= 1024 { 1024 } else { (k as u32).div_ceil(32) * 32 })
}

/// A dense product kernel (kernels/engine.cu, gemm3) and its tile: rows of X and rows of W a block takes, its threads and its dynamic shared memory.
#[derive(Clone)]
struct Tile {
    k: Kernel,
    name: &'static str,
    bm: u32,
    bn: u32,
    bk: u32,
    threads: u32,
    smem: u32,
}

/// (kernel, BM, BN, BK, threads, stages): the tiles the products choose among.
const PLAIN: [(&str, u32, u32, u32, u32, u32); 6] = [
    (kname!("gemm3_256x128_s3"), 256, 128, 32, 256, 3),
    (kname!("gemm3_128x256_b64_s2"), 128, 256, 64, 256, 2),
    (kname!("gemm3_128x128_b64_s3"), 128, 128, 64, 128, 3),
    (kname!("gemm3_128x256_s3"), 128, 256, 32, 256, 3),
    (kname!("gemm3_128x128_s4"), 128, 128, 32, 128, 4),
    (kname!("gemm3_256x128_b64_s2"), 256, 128, 64, 256, 2),
];
/// The same with silu(gate) * up in the epilogue, for the gate-and-up product.
const FUSED: [(&str, u32, u32, u32, u32, u32); 3] = [
    (kname!("gemm3f_256x128_s3"), 256, 128, 32, 256, 3),
    (kname!("gemm3f_128x256_b64_s2"), 128, 256, 64, 256, 2),
    (kname!("gemm3f_128x256_s3"), 128, 256, 32, 256, 3),
];

fn tiles(dev: &Dev, list: &[(&'static str, u32, u32, u32, u32, u32)]) -> Result<Vec<Tile>> {
    list.iter()
        // (an AMD build: the tiles whose shared memory fits a block's 64 KiB on RDNA 3 and 4)
        .filter(|&&(_, bm, bn, bk, _, st)| !cfg!(glyd_hip) || (bm + bn) * bk * 2 * st <= 65536)
        .map(|&(name, bm, bn, bk, threads, st)| {
            let smem = (bm + bn) * bk * 2 * st;
            let k = dev.kernel(name)?;
            k.allow_smem(smem)?;
            Ok(Tile { k, name, bm, bn, bk, threads, smem })
        })
        .collect()
}

pub struct Kern {
    plain: Vec<Tile>,
    /// the small tile (64 x 64) for products a big tile leaves the GPU half idle with, and the same with fp32 sums out; the GPU's multiprocessors
    small: Tile,
    small_f32: Tile,
    sms: u32,
    fused: Vec<Tile>,
    copy_row: Kernel,
    copy_rows: Kernel,
    rmsnorm: Kernel,
    rmsnorm_add: Kernel,
    qk_norm_rope: Kernel,
    silu_mul: Kernel,
    argmax: Kernel,
    topk: Kernel,
    /// `topk` with a mask per row (a grammar's): None where the kernels have none.
    topk_masked: Option<Kernel>,
    expsum: Kernel,
    count_diff: Kernel,
    logsumexp: Kernel,
    row_stats: Kernel,
    read_stream: Kernel,
    kv_hist: Kernel,
    kv_outside_total: Kernel,
    kv_leaf_hp: Kernel,
    kv_sweep: Kernel,
    attn_self: Kernel,
    /// The prompt attention's tile (warps a block, keys a step) and its kernel.
    attn: Kernel,
    attn_warps: u32,
    attn_bn: u32,
}

/// The prompt attention's shape (warps a block, keys a step): w4b64 (an AMD build w4b32, whose 48 KiB fit RDNA's 64 KiB a block); a development build tries another with
/// GLYD_ENGINE_ATTN_TILE (w8b64, w4b32 or w8b32).
#[cfg(feature = "dev")]
fn attn_shape() -> Result<(u32, u32)> {
    match std::env::var("GLYD_ENGINE_ATTN_TILE").unwrap_or_else(|_| (if cfg!(glyd_hip) { "w4b32" } else { "w4b64" }).to_string()).as_str() {
        "w4b64" => Ok((4, 64)),
        "w8b64" => Ok((8, 64)),
        "w4b32" => Ok((4, 32)),
        "w8b32" => Ok((8, 32)),
        t => crate::ffi::err(format!("GLYD_ENGINE_ATTN_TILE={t}: w4b64, w8b64, w4b32 or w8b32")),
    }
}

#[cfg(not(feature = "dev"))]
fn attn_shape() -> Result<(u32, u32)> {
    Ok(if cfg!(glyd_hip) { (4, 32) } else { (4, 64) })
}

impl Kern {
    pub fn load(dev: &Dev) -> Result<Kern> {
        let (attn_warps, attn_bn) = attn_shape()?;
        let attn = dev.kernel(match (attn_warps, attn_bn) {
            (4, 64) => kname!("prefill_attn_w4_b64"),
            (8, 64) => kname!("prefill_attn_w8_b64"),
            (4, 32) => kname!("prefill_attn_w4_b32"),
            _ => kname!("prefill_attn_w8_b32"),
        })?;
        attn.allow_smem((16 * attn_warps + 4 * attn_bn) * 256)?;
        let small = tiles(dev, &[(kname!("gemm3_64x64_b64_s3"), 64, 64, 64, 128, 3), (kname!("gemm3o_64x64_b64_s3"), 64, 64, 64, 128, 3)])?;
        Ok(Kern {
            plain: tiles(dev, &PLAIN)?,
            small: small[0].clone(),
            small_f32: small[1].clone(),
            sms: dev.sms.max(1) as u32,
            fused: tiles(dev, &FUSED)?,
            kv_hist: dev.kernel(kname!("kv_hist"))?,
            kv_outside_total: dev.kernel(kname!("kv_outside_total"))?,
            kv_leaf_hp: dev.kernel(kname!("kv_leaf_hp"))?,
            kv_sweep: dev.kernel(kname!("kv_sweep"))?,
            attn_self: dev.kernel(kname!("attn_self"))?,
            attn,
            attn_warps,
            attn_bn,
            copy_row: dev.kernel(kname!("copy_row"))?,
            copy_rows: dev.kernel(kname!("copy_rows"))?,
            rmsnorm: dev.kernel(kname!("rmsnorm"))?,
            rmsnorm_add: dev.kernel(kname!("rmsnorm_add"))?,
            qk_norm_rope: dev.kernel(kname!("qk_norm_rope"))?,
            silu_mul: dev.kernel(kname!("silu_mul"))?,
            argmax: dev.kernel(kname!("argmax_bf16"))?,
            topk: dev.kernel(kname!("topk64_bf16"))?,
            topk_masked: dev.kernel(kname!("topk64_bf16_masked")).ok(),
            expsum: dev.kernel(kname!("expsum_bf16"))?,
            count_diff: dev.kernel(kname!("count_diff_u16"))?,
            logsumexp: dev.kernel(kname!("logsumexp_bf16"))?,
            row_stats: dev.kernel(kname!("row_stats"))?,
            read_stream: dev.kernel(kname!("read_stream"))?,
        })
    }

    /// The tile a dense product of `m` rows by `n` columns and `k` deep takes (an index for `gemm`): the 256x128 tile from 1,536 rows on; below, the 128x256 one (BK 64) for the widest products and
    /// the 128x128 one (BK 64) for the rest; as measured on an L4 with Qwen3-8B's shapes (within 1.00 to 1.09 of cuBLAS's time at 1,024 and 3,500 rows). GLYD_ENGINE_TILE=name forces one.
    pub fn pick_tile(&self, m: usize, n: usize, k: usize) -> usize {
        if let Some(t) = crate::tune!("GLYD_ENGINE_TILE") {
            if let Some(i) = self.plain.iter().position(|x| x.name == t) {
                return i;
            }
        }
        let fits = |t: &Tile| n.is_multiple_of(t.bn as usize) && k.is_multiple_of(t.bk as usize);
        let want = if m >= 1536 {
            // (a long prompt: the big tile, or the smaller one where the big tile's last wave of blocks leaves the multiprocessors more idle: a block a multiprocessor, the work of a wave its tile's rows by columns;
            // the same sums either way. 128 x 128 for the 4,096-column products of a 2,000-token prompt on 80 multiprocessors: 7 waves of the smaller against 4 of the bigger, 6% of the pass)
            let (big, small) = (kname!("gemm3_256x128_s3"), kname!("gemm3_128x128_s4"));
            let cost = |name: &str| {
                self.plain.iter().find(|t| t.name == name && fits(t)).map(|t| {
                    let blocks = m.div_ceil(t.bm as usize) * (n / t.bn.max(1) as usize);
                    blocks.div_ceil(self.sms as usize) as u64 * t.bm as u64 * t.bn as u64
                })
            };
            match (cost(big), cost(small)) {
                (Some(b), Some(sm)) if sm * 100 < b * 97 => small,
                _ => big,
            }
        } else if n >= 16384 {
            kname!("gemm3_128x256_b64_s2")
        } else {
            kname!("gemm3_128x128_b64_s3")
        };
        self.plain.iter().position(|t| t.name == want && fits(t)).or_else(|| self.plain.iter().position(fits)).unwrap_or(0)
    }

    /// `gemm` with the tile `pick_tile` gives, or the small one (64 x 64) where that tile would leave the multiprocessors with fewer than two blocks each and the small one fits: the same sums either way (a
    /// block adds its k steps in the same order whatever its shape).
    ///
    /// # Safety
    /// As `gemm`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_fill(&self, s: Stream, x: u64, w: u64, y: u64, m: i32, n: i32, k: i32) -> Result<()> {
        let t = &self.plain[self.pick_tile(m as usize, n as usize, k as usize)];
        let blocks = (m as u32).div_ceil(t.bm) * (n as u32 / t.bn.max(1));
        if blocks < 2 * self.sms && (n as u32).is_multiple_of(self.small.bn) && (k as u32).is_multiple_of(self.small.bk) {
            return self.launch_tile(&self.small, s, x, w, y, m, n, k);
        }
        self.gemm(s, self.pick_tile(m as usize, n as usize, k as usize), x, w, y, m, n, k)
    }

    /// Y [m, n] fp32 = X [m, k] W^T (bf16 in, the sums as they are: a router's logits) on the small tile; n and k multiples of 64.
    ///
    /// # Safety
    /// As `gemm`, with Y [m, n] fp32.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_f32(&self, s: Stream, x: u64, w: u64, y: u64, m: i32, n: i32, k: i32) -> Result<()> {
        self.launch_tile(&self.small_f32, s, x, w, y, m, n, k)
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_tile(&self, t: &Tile, s: Stream, x: u64, w: u64, y: u64, m: i32, n: i32, k: i32) -> Result<()> {
        if !(n as u32).is_multiple_of(t.bn) || !(k as u32).is_multiple_of(t.bk) {
            return crate::ffi::err(format!("a product with {n} columns and {k} deep is not one the engine runs"));
        }
        let tiles = (m as u32).div_ceil(t.bm) * (n as u32 / t.bn);
        launch!(t.k, [tiles, 1, 1], [t.threads, 1, 1], smem t.smem, s; x, w, y, m, n, k)
    }

    pub fn tile_name(&self, i: usize) -> &'static str {
        self.plain[i].name
    }

    /// Y [m, n] = X [m, k] W^T (bf16 in and out, fp32 sums), W [n, k] row-major: the engine's own product, as cuBLAS's for a decoded matrix. n a multiple of the tile's columns, k of its step.
    ///
    /// # Safety
    /// As `copy_row`: X [m, k], W [n, k] and Y [m, n] live.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm(&self, s: Stream, tile: usize, x: u64, w: u64, y: u64, m: i32, n: i32, k: i32) -> Result<()> {
        let t = &self.plain[tile];
        if !(n as u32).is_multiple_of(t.bn) || !(k as u32).is_multiple_of(t.bk) {
            return crate::ffi::err(format!("a product with {n} columns and {k} deep is not one the engine runs"));
        }
        let tiles = (m as u32).div_ceil(t.bm) * (n as u32 / t.bn);
        launch!(t.k, [tiles, 1, 1], [t.threads, 1, 1], smem t.smem, s; x, w, y, m, n, k)
    }

    /// Y [m, n / 2] = silu(gate) * up of X [m, k] W^T, W [n, k] holding the gate rows first (n / 2) and the up rows after them: the product and `silu_mul` in one kernel, the same bits.
    ///
    /// # Safety
    /// As `gemm`, with Y [m, n / 2].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_silu(&self, s: Stream, x: u64, w: u64, y: u64, m: i32, n: i32, k: i32) -> Result<()> {
        let want = match crate::tune!("GLYD_ENGINE_TILE") {
            Some(t) if self.fused.iter().any(|x| x.name == t) => t,
            _ if m >= 1536 => {
                // (a long prompt: of the two tiles of the same size, the one that leaves the multiprocessors less idle in the last wave of blocks: a block a multiprocessor, the same sums either way)
                let (a, b) = (kname!("gemm3f_256x128_s3"), kname!("gemm3f_128x256_b64_s2"));
                let waves = |name: &str| {
                    self.fused
                        .iter()
                        .find(|t| t.name == name && (n as u32).is_multiple_of(t.bn) && (k as u32).is_multiple_of(t.bk))
                        .map(|t| (m as usize).div_ceil(t.bm as usize).saturating_mul(n as usize / t.bn as usize).div_ceil(self.sms as usize))
                };
                match (waves(a), waves(b)) {
                    (Some(wa), Some(wb)) if wb < wa => b,
                    _ => a,
                }
                .to_string()
            }
            _ => kname!("gemm3f_128x256_b64_s2").to_string(),
        };
        let fits = |t: &Tile| (n as u32).is_multiple_of(t.bn) && (k as u32).is_multiple_of(t.bk);
        let Some(t) = self.fused.iter().find(|t| t.name == want && fits(t)).or_else(|| self.fused.iter().find(|t| fits(t))) else {
            return crate::ffi::err(format!("a fused product with {n} columns and {k} deep is not one the engine runs"));
        };
        let tiles = (m as u32).div_ceil(t.bm) * (n as u32 / t.bn);
        launch!(t.k, [tiles, 1, 1], [t.threads, 1, 1], smem t.smem, s; x, w, y, m, n, k)
    }

    /// For each of the `n` pairs (dst row, src row) at `pairs` (int32 x 2): row `dst` of dst [., k] = row `src` of src [., k]. k a multiple of 8.
    ///
    /// # Safety
    /// As `copy_row`.
    pub unsafe fn copy_rows(&self, s: Stream, src: u64, pairs: u64, n: i32, k: i32, dst: u64) -> Result<()> {
        if k % 8 != 0 {
            return crate::ffi::err(format!("rows of {k} elements: not supported"));
        }
        if n == 0 {
            return Ok(());
        }
        launch!(self.copy_rows, [n as u32, 1, 1], [256, 1, 1], s; src, pairs, k, dst)
    }

    /// dst [k] = row `row` of src [., k].
    ///
    /// # Safety
    /// Live device buffers of the sizes the kernel reads and writes (the kernels' comments in engine.cu).
    pub unsafe fn copy_row(&self, s: Stream, src: u64, row: i32, k: i32, dst: u64) -> Result<()> {
        launch!(self.copy_row, [(k as u32).div_ceil(256), 1, 1], [256, 1, 1], s; src, row, k, dst)
    }

    /// out [rows, k] = rmsnorm(x) * w.
    ///
    /// # Safety
    /// As `copy_row`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn rmsnorm(&self, s: Stream, x: u64, w: u64, out: u64, rows: u32, k: i32, eps: f32) -> Result<()> {
        let t = norm_threads(k)?;
        launch!(self.rmsnorm, [rows, 1, 1], [t, 1, 1], s; x, w, out, k, eps)
    }

    /// x += y (bf16), then out = rmsnorm(x) * w.
    ///
    /// # Safety
    /// As `copy_row`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn rmsnorm_add(&self, s: Stream, x: u64, y: u64, w: u64, out: u64, rows: u32, k: i32, eps: f32) -> Result<()> {
        let t = norm_threads(k)?;
        launch!(self.rmsnorm_add, [rows, 1, 1], [t, 1, 1], s; x, y, w, out, k, eps)
    }

    /// Query and key heads of m qkv rows: norm and rotary embedding.
    ///
    /// # Safety
    /// As `copy_row`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn qk_norm_rope(
        &self,
        s: Stream,
        qkv: u64,
        stride: i32,
        qw: u64,
        kw: u64,
        cos: u64,
        sin: u64,
        pos: u64,
        qo: u64,
        ko: u64,
        m: u32,
        nq: i32,
        nk: i32,
        eps: f32,
    ) -> Result<()> {
        launch!(self.qk_norm_rope, [m, (nq + nk) as u32, 1], [64, 1, 1], s; qkv, stride, qw, kw, cos, sin, pos, qo, ko, nq, nk, eps)
    }

    /// out [rows, inter] = silu(gate) * up of gu [rows, 2 inter].
    ///
    /// # Safety
    /// As `copy_row`.
    pub unsafe fn silu_mul(&self, s: Stream, gu: u64, out: u64, rows: u32, inter: i32) -> Result<()> {
        if inter % 8 != 0 {
            return crate::ffi::err(format!("intermediate size {inter}: not supported"));
        }
        launch!(self.silu_mul, [rows, (inter as u32 / 8).div_ceil(256), 1], [256, 1, 1], s; gu, out, inter)
    }

    /// Causal attention for the `n` tokens of a chunk after `start` earlier ones over the bf16 staging cache `kv` [tokens, kv_heads, 256] (keys then values of a token's head):
    /// q [n, heads, 128] (rows q_stride elements apart) to out [n, heads, 128] (rows out_stride apart).
    ///
    /// # Safety
    /// As `copy_row`; `kv` holds start + n tokens.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn prefill_attn(
        &self,
        s: Stream,
        q: u64,
        q_stride: i32,
        kv: u64,
        out: u64,
        out_stride: i32,
        n: i32,
        start: i32,
        heads: i32,
        kv_heads: i32,
    ) -> Result<()> {
        if heads % kv_heads != 0 {
            return crate::ffi::err(format!("{heads} query heads over {kv_heads} key-value heads"));
        }
        let rows = 16 * self.attn_warps as i32;
        let scale_log2 = (1.0f64 / (128.0f64).sqrt() * std::f64::consts::LOG2_E) as f32;
        let smem = (16 * self.attn_warps + 4 * self.attn_bn) * 256;
        launch!(self.attn, [((n + rows - 1) / rows) as u32, heads as u32, 1], [32 * self.attn_warps, 1, 1], smem smem, s; q, q_stride, kv, out, out_stride, n, start, heads, kv_heads, scale_log2)
    }

    /// The first largest of v logits: its index into `token` (int32) and its value into `value` (f32).
    ///
    /// # Safety
    /// As `copy_row`.
    pub unsafe fn argmax(&self, s: Stream, logits: u64, v: i32, token: u64, value: u64) -> Result<()> {
        self.argmax_rows(s, logits, v, 1, token, value)
    }

    /// The first largest of each of `rows` rows of v logits: row r's index into `token`[r] (int32) and its value into `value`[r] (f32).
    ///
    /// # Safety
    /// As `copy_row`.
    pub unsafe fn argmax_rows(&self, s: Stream, logits: u64, v: i32, rows: u32, token: u64, value: u64) -> Result<()> {
        if v % 8 != 0 {
            return crate::ffi::err(format!("vocabulary {v}: not supported"));
        }
        launch!(self.argmax, [rows, 1, 1], [1024, 1, 1], s; logits, v, token, value)
    }

    /// The `TOPK` largest of each of `rows` rows of v logits, largest first (equal values by the smaller index first), as the host's sampler orders its candidates: row r's token indices into `tok`[r * TOPK ..] (int32) and
    /// their values into `val`[r * TOPK ..] (f32).
    ///
    /// # Safety
    /// As `copy_row`.
    pub unsafe fn topk_rows(&self, s: Stream, logits: u64, v: i32, rows: u32, tok: u64, val: u64) -> Result<()> {
        if v % 8 != 0 || v < TOPK as i32 {
            return crate::ffi::err(format!("vocabulary {v}: not supported"));
        }
        launch!(self.topk, [rows, 1, 1], [1024, 1, 1], s; logits, v, tok, val)
    }

    /// Whether the kernels have `topk_rows_masked`.
    pub fn has_masked_topk(&self) -> bool {
        self.topk_masked.is_some()
    }

    /// `topk_rows` with a mask per row: row r's flags and bits at `mask`[r * `words` ..] (uint32; `words` = 1 + ceil(v / 32): the first word nonzero where the row is held to its bits, a token's bit in word 1 + token / 32,
    /// bit token % 32), a token whose bit is clear ranking below every other (the row as if its logit were -inf). Where fewer than TOPK tokens are allowed the rest of the candidates are tokens that are not (the host
    /// drops them by its own mask). Err where the kernels have no such kernel.
    ///
    /// # Safety
    /// As `topk_rows`; `mask` holds `rows * words` words.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn topk_rows_masked(&self, s: Stream, logits: u64, v: i32, rows: u32, tok: u64, val: u64, mask: u64, words: i32) -> Result<()> {
        let Some(k) = &self.topk_masked else { return crate::ffi::err("these kernels have no masked top-k") };
        if v % 8 != 0 || v < TOPK as i32 {
            return crate::ffi::err(format!("vocabulary {v}: not supported"));
        }
        launch!(k, [rows, 1, 1], [1024, 1, 1], s; logits, v, tok, val, mask, words)
    }

    /// The sum over each of `rows` rows of v logits of exp((x - max) / temperature), the normalizer of a top-p cut over the row's largest logits: row r's into `out`[r] (f64). `maxv`[r] (f32): the row's largest, as `argmax_rows` leaves it.
    /// The exponents are formed in fp32 as the host's sampler forms them; `topkcheck` holds the sum within 2e-6 of the one in fp64.
    ///
    /// # Safety
    /// As `copy_row`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn exp_sum_rows(&self, s: Stream, logits: u64, v: i32, rows: u32, maxv: u64, temperature: f32, out: u64) -> Result<()> {
        if v % 8 != 0 {
            return crate::ffi::err(format!("vocabulary {v}: not supported"));
        }
        launch!(self.expsum, [rows, 1, 1], [1024, 1, 1], s; logits, v, maxv, temperature, out)
    }

    /// How many of the `n` 16-bit values at `a` differ from those at `b`, added to `count` (uint64 [17]: all, then by head and key or value).
    ///
    /// # Safety
    /// As `copy_row`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn count_diff(&self, s: Stream, a: u64, b: u64, n: u64, count: u64, heads: i32, dim: i32) -> Result<()> {
        launch!(self.count_diff, [n.div_ceil(256).clamp(1, 1024) as u32, 1, 1], [256, 1, 1], s; a, b, n, count, heads, dim)
    }

    /// The log-sum-exp of v logits given their maximum, into `lse` (f32).
    ///
    /// # Safety
    /// As `copy_row`.
    pub unsafe fn logsumexp(&self, s: Stream, logits: u64, v: i32, maxv: f32, lse: u64) -> Result<()> {
        launch!(self.logsumexp, [1, 1, 1], [1024, 1, 1], s; logits, v, maxv, lse)
    }

    /// Per row of logits [rows, v]: the first largest into `top1` (int32), the log-sum-exp into `lse` and the target's logit into `tlogit` (f32).
    ///
    /// # Safety
    /// As `copy_row`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn row_stats(&self, s: Stream, logits: u64, v: i32, target: u64, top1: u64, lse: u64, tlogit: u64, rows: u32) -> Result<()> {
        launch!(self.row_stats, [rows, 1, 1], [1024, 1, 1], s; logits, v, target, top1, lse, tlogit)
    }

    /// A streaming read of `n` 16-byte words at `p` (the memory bandwidth a read-only kernel gets); `out` is one u32.
    ///
    /// # Safety
    /// As `copy_row`; `p` holds 16 n bytes.
    pub unsafe fn read_stream(&self, s: Stream, p: u64, n: u64, out: u64, blocks: u32) -> Result<()> {
        launch!(self.read_stream, [blocks, 1, 1], [256, 1, 1], s; p, n, out)
    }

    /// counts [c, 128] (int32) += the exponent bins of the bf16 bits x [t, c] (rows `stride` apart), per channel (c a multiple of 16).
    ///
    /// # Safety
    /// As `copy_row`.
    pub unsafe fn kv_hist(&self, s: Stream, x: u64, stride: i32, t: i32, c: i32, counts: u64) -> Result<()> {
        if c % 16 != 0 {
            return crate::ffi::err(format!("{c} channels: not supported"));
        }
        launch!(self.kv_hist, [(c / 16) as u32, 1, 1], [256, 1, 1], s; x, stride, t, c, counts)
    }

    /// *total (u64) += the values of x [t, c] outside their channels' windows (base [c], uint8).
    ///
    /// # Safety
    /// As `copy_row`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn kv_outside_total(&self, s: Stream, x: u64, stride: i32, t: i32, c: i32, base: u64, total: u64) -> Result<()> {
        let n = (t as u64 * c as u64).div_ceil(256).clamp(1, 4096) as u32;
        launch!(self.kv_outside_total, [n, 1, 1], [256, 1, 1], s; x, stride, t, c, base, total)
    }

    /// hp [page0 + p, heads] (int32) = the values outside the windows in page p's 16 tokens' keys and values, for `npages` pages from row 0 of k and v.
    ///
    /// # Safety
    /// As `copy_row`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn kv_leaf_hp(
        &self,
        s: Stream,
        k: u64,
        ks: i32,
        v: u64,
        vs: i32,
        kbase: u64,
        vbase: u64,
        heads: i32,
        npages: u32,
        page0: i32,
        hp: u64,
    ) -> Result<()> {
        if npages == 0 {
            return Ok(());
        }
        launch!(self.kv_leaf_hp, [npages, heads as u32, 1], [128, 1, 1], s; k, ks, v, vs, kbase, vbase, heads, page0, hp)
    }

    /// The sweep of `m` one-token rows of k and v: into need and exc [rows, layers] at row0 + t, layer.
    ///
    /// # Safety
    /// As `copy_row`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn kv_sweep(
        &self,
        s: Stream,
        k: u64,
        ks: i32,
        v: u64,
        vs: i32,
        kbase: u64,
        vbase: u64,
        heads: i32,
        layer: i32,
        layers: i32,
        row0: i32,
        m: u32,
        need: u64,
        exc: u64,
    ) -> Result<()> {
        launch!(self.kv_sweep, [m, 1, 1], [32 * heads as u32, 1, 1], s; k, ks, v, vs, kbase, vbase, heads, layer, layers, row0, need, exc)
    }

    /// out [m, heads, 128] = the value rows of v [m, kv_heads, 128] (one-token sequences' attention).
    ///
    /// # Safety
    /// As `copy_row`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn attn_self(&self, s: Stream, v: u64, vs: i32, out: u64, os: i32, heads: i32, kv_heads: i32, m: u32) -> Result<()> {
        launch!(self.attn_self, [m, heads as u32, 1], [128, 1, 1], s; v, vs, out, os, heads, kv_heads)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every build looks its kernels up by the names engine.cu gives them (release builds once renamed them gk_N); no two tiles go by one name.
    #[test]
    fn kernel_names_are_the_sources_in_every_build() {
        let tiles: Vec<&str> = PLAIN.iter().chain(FUSED.iter()).map(|t| t.0).collect();
        for n in
            tiles.iter().copied().chain([kname!("copy_row"), kname!("kv_hist"), kname!("prefill_attn_w4_b64"), kname!("topk64_bf16"), kname!("expsum_bf16")])
        {
            assert!(!(n.starts_with("gk_") && n[3..].bytes().all(|b| b.is_ascii_digit())), "{n}");
        }
        assert_eq!(tiles.iter().collect::<std::collections::HashSet<_>>().len(), tiles.len());
    }
}
