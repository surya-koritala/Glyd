//! The engine's own kernels (kernels/engine.cu) with typed launches. Device addresses are `u64`.

pub mod gemma4;
pub mod glm;
pub mod mamba2;
pub mod qwen35;
pub mod qwen4x;

pub(crate) use gemma4::KernG4;
pub use glm::{mla_positions, mla_smem, Hc, KernGlm, GM_CHUNK_SMEM, GM_D, GM_HC, GM_L, GM_NP};
pub use mamba2::{KernMamba2, SsdWs, MAMBA2_CHUNK, MAMBA2_HEAD, MAMBA2_STATE};
pub use qwen35::Kern35;

use crate::dev::{copy_h2d_async, ArgsOf, Buf, ByArgs, Dev, Pinned, Typed};
use crate::ffi::{Result, Stream};
use crate::launch;
use crate::sampler::{Query, Resolved};

/// Candidates the sampling kernel returns for a row of logits (kernels/engine.cu).
pub const TOPK: usize = 64;

/// The bytes of a query's bins (`Kern::draw_bins`): 256 counts (u32), then 256 weights, least keys and greatest keys (u64).
pub const DRAW_BINS_BYTES: usize = 256 * (4 + 8 * 3);

/// A query's bins as `Kern::draw_bins` leaves them (`DRAW_BINS_BYTES` of u32 words).
pub fn bins_of_words(w: &[u32]) -> crate::sampler::Bins {
    let u64s = |at: usize| (0..256).map(|i| w[at + 2 * i] as u64 | (w[at + 2 * i + 1] as u64) << 32).collect::<Vec<u64>>();
    crate::sampler::Bins { count: w[..256].to_vec(), weight: u64s(256), least: u64s(768), most: u64s(1280) }
}

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
    k: ByArgs<GemmArgs>,
    name: &'static str,
    bm: u32,
    bn: u32,
    bk: u32,
    threads: u32,
    smem: u32,
}

/// The arguments every GEMM tile's kernel takes (kernels/engine.cu's gemm3 family): a tile of another signature is not one of these lists.
type GemmArgs = ArgsOf<crate::kt::gemm3_256x128_s3>;

/// A tile's kernel loaded.
type TileLoad = fn(&Dev) -> Result<ByArgs<GemmArgs>>;

fn tile<T: Typed<Args = GemmArgs>>(dev: &Dev) -> Result<ByArgs<GemmArgs>> {
    dev.by_args::<T>()
}

/// A tile of a list: its kernel's name and loader, from the one name (they cannot part), and its shape.
macro_rules! tile {
    ($k:ident, $($shape:expr),+) => {
        (stringify!($k), tile::<crate::kt::$k> as TileLoad, $($shape),+)
    };
}

/// (kernel, BM, BN, BK, threads, stages): the tiles the products choose among.
const PLAIN: [(&str, TileLoad, u32, u32, u32, u32, u32); 6] = [
    tile!(gemm3_256x128_s3, 256, 128, 32, 256, 3),
    tile!(gemm3_128x256_b64_s2, 128, 256, 64, 256, 2),
    tile!(gemm3_128x128_b64_s3, 128, 128, 64, 128, 3),
    tile!(gemm3_128x256_s3, 128, 256, 32, 256, 3),
    tile!(gemm3_128x128_s4, 128, 128, 32, 128, 4),
    tile!(gemm3_256x128_b64_s2, 256, 128, 64, 256, 2),
];
/// The same with silu(gate) * up in the epilogue, for the gate-and-up product.
const FUSED: [(&str, TileLoad, u32, u32, u32, u32, u32); 3] =
    [tile!(gemm3f_256x128_s3, 256, 128, 32, 256, 3), tile!(gemm3f_128x256_b64_s2, 128, 256, 64, 256, 2), tile!(gemm3f_128x256_s3, 128, 256, 32, 256, 3)];

fn tiles(dev: &Dev, list: &[(&'static str, TileLoad, u32, u32, u32, u32, u32)]) -> Result<Vec<Tile>> {
    list.iter()
        // (an AMD build: the tiles whose shared memory fits a block's 64 KiB on RDNA 3 and 4)
        .filter(|&&(_, _, bm, bn, bk, _, st)| !cfg!(glyd_hip) || (bm + bn) * bk * 2 * st <= 65536)
        .map(|&(name, load, bm, bn, bk, threads, st)| {
            let smem = (bm + bn) * bk * 2 * st;
            let k = load(dev)?;
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
    copy_row: crate::kt::copy_row,
    copy_rows: crate::kt::copy_rows,
    parts_rows: crate::kt::parts_rows,
    rmsnorm: crate::kt::rmsnorm,
    rmsnorm_add: crate::kt::rmsnorm_add,
    qk_norm_rope: crate::kt::qk_norm_rope,
    silu_mul: crate::kt::silu_mul,
    argmax: crate::kt::argmax_bf16,
    topk: crate::kt::topk64_bf16,
    gather: crate::kt::gather_bf16,
    /// The row a draw sees (`ProcGpu`), and its candidates made of it; a draw's total, bins and tokens of a row of bf16 logits or of floats (`sampler::View`).
    proc_fill: crate::kt::proc_fill,
    proc_entries: crate::kt::proc_entries,
    topk_f32: crate::kt::topk64_f32,
    draw_total: (crate::kt::draw_total_bf16, crate::kt::draw_total_f32),
    draw_bins: (crate::kt::draw_bins_bf16, crate::kt::draw_bins_f32),
    draw_tokens: (crate::kt::draw_tokens_bf16, crate::kt::draw_tokens_f32),
    draw_weight_sums: crate::kt::draw_weight_sums,
    kv12_rows: crate::kt::kv12_rows,
    row_tables: crate::kt::row_tables,
    swap_copy: crate::kt::swap_copy,
    /// None in a table made without it (a step is then not launched before the one before it is read); `load` always has it.
    step_ids: Option<crate::kt::step_ids>,
    expsum: crate::kt::expsum_bf16,
    logsumexp: crate::kt::logsumexp_bf16,
    read_stream: crate::kt::read_stream,
}

impl Kern {
    pub fn load(dev: &Dev) -> Result<Kern> {
        let small = tiles(dev, &[tile!(gemm3_64x64_b64_s3, 64, 64, 64, 128, 3), tile!(gemm3o_64x64_b64_s3, 64, 64, 64, 128, 3)])?;
        Ok(Kern {
            plain: tiles(dev, &PLAIN)?,
            small: small[0].clone(),
            small_f32: small[1].clone(),
            sms: dev.sms.max(1) as u32,
            fused: tiles(dev, &FUSED)?,
            copy_row: dev.typed()?,
            copy_rows: dev.typed()?,
            parts_rows: dev.typed()?,
            rmsnorm: dev.typed()?,
            rmsnorm_add: dev.typed()?,
            qk_norm_rope: dev.typed()?,
            silu_mul: dev.typed()?,
            argmax: dev.typed()?,
            topk: dev.typed()?,
            gather: dev.typed()?,
            proc_fill: dev.typed()?,
            proc_entries: dev.typed()?,
            topk_f32: dev.typed()?,
            draw_total: (dev.typed()?, dev.typed()?),
            draw_bins: (dev.typed()?, dev.typed()?),
            draw_tokens: (dev.typed()?, dev.typed()?),
            draw_weight_sums: dev.typed()?,
            kv12_rows: dev.typed()?,
            row_tables: dev.typed()?,
            swap_copy: dev.typed()?,
            // (kernels of every build of kernels/engine.cu: one that does not load is this load's error, not a table without it)
            step_ids: Some(dev.typed()?),
            expsum: dev.typed()?,
            logsumexp: dev.typed()?,
            read_stream: dev.typed()?,
        })
    }

    /// The tile a dense product of `m` rows by `n` columns and `k` deep takes (an index for `gemm`): the 256x128 tile from 1,536 rows on; below, the 128x256 one (BK 64) for the widest products and
    /// the 128x128 one (BK 64) for the rest; as measured on an L4 at a dense model's shapes of hidden 4,096 (within 1.00 to 1.09 of cuBLAS's time at 1,024 and 3,500 rows). GLYD_ENGINE_TILE=name forces one.
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
            let (big, small) = ("gemm3_256x128_s3", "gemm3_128x128_s4");
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
            "gemm3_128x256_b64_s2"
        } else {
            "gemm3_128x128_b64_s3"
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
                let (a, b) = ("gemm3f_256x128_s3", "gemm3f_128x256_b64_s2");
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
            _ => "gemm3f_128x256_b64_s2".to_string(),
        };
        let fits = |t: &Tile| (n as u32).is_multiple_of(t.bn) && (k as u32).is_multiple_of(t.bk);
        let Some(t) = self.fused.iter().find(|t| t.name == want && fits(t)).or_else(|| self.fused.iter().find(|t| fits(t))) else {
            return crate::ffi::err(format!("a fused product with {n} columns and {k} deep is not one the engine runs"));
        };
        let tiles = (m as u32).div_ceil(t.bm) * (n as u32 / t.bn);
        launch!(t.k, [tiles, 1, 1], [t.threads, 1, 1], smem t.smem, s; x, w, y, m, n, k)
    }

    /// dst [n, w] (rows ds apart) = n rows of w values of src (rows ss apart): a prompt's part of a fused product's rows copied out (`Op::Parts`).
    ///
    /// # Safety
    /// Live device buffers of those sizes, on stream `s`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn parts_rows(&self, s: Stream, src: u64, ss: usize, dst: u64, ds: usize, w: usize, n: usize) -> Result<()> {
        launch!(self.parts_rows, [((n * w) as u32).div_ceil(256), 1, 1], [256, 1, 1], s; src, ss as i64, dst, ds as i64, w as i32, n as i32)
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

    /// Query and key heads of m qkv rows: norm and rotary embedding. `kcheck` alone launches it (against torch's, tools/kcheck.py): no engine path has since the
    /// attention became the layer kind's operation (src/dynamic/attn.rs, whose kernels norm and turn the heads themselves).
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

    /// Row `row`'s (`v` bf16 logits at `logits`) values at the `n` tokens `idx` (int, each below v) into `out` (f32).
    ///
    /// # Safety
    /// As `copy_row`; `idx` holds n ints below v and `out` room for n f32.
    /// The 12-bit pages' slots (int64 [n]), lengths and windows' first tokens (int32 [n s]) of n rows at positions `pos` (int32 [n]) through block table `tab.0` (0: none; row r's
    /// at tab.0 + r tab.1 entries), as the kv12_rows kernel says.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn kv12_rows(
        &self,
        s: Stream,
        pos: u64,
        n: usize,
        tab: (u64, usize),
        window: usize,
        parts: usize,
        slots: u64,
        lens: u64,
        first: u64,
    ) -> Result<()> {
        launch!(self.kv12_rows, [(n as u32).div_ceil(256), 1, 1], [256, 1, 1], s; pos, n as i32, tab.0, tab.1 as i64, window as i32, parts as i32, slots, lens, first)
    }

    /// The block tables of a pass's `n` decode rows (the row_tables kernel): row r's sequence's (row `slot[r]` of `tabs`, `t` entries a row) to rows r dup .. r dup + dup - 1 of
    /// `out`, the entries its position `pos[r]` reads in a window of `window` tokens (0: none). A row whose slot or position is negative (padding) is left as it is.
    ///
    /// # Safety
    /// `tabs` holds every slot's t entries, `slot` and `pos` n int32 each, `out` n dup t int32.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn row_tables(&self, s: Stream, tabs: u64, t: usize, slot: u64, pos: u64, n: usize, window: usize, dup: usize, out: u64) -> Result<()> {
        let x = (t as u32).div_ceil(256).clamp(1, 8);
        launch!(self.row_tables, [x, n as u32, 1], [256, 1, 1], s; tabs, t as i64, slot, pos, n as i32, window as i32, dup as i32, out)
    }

    /// The `n` copies listed at `items` (three u64 a copy: source, destination, bytes, a multiple of 16 with both ends 16-byte aligned), by `blocks` blocks of 256 threads: a swap's
    /// (src/dynamic/swap.rs), whose host side is pinned memory mapped into the device's addresses.
    ///
    /// # Safety
    /// `items` and every range it lists are live for the kernel's run, device memory or mapped pinned host memory, and no two destinations overlap.
    pub unsafe fn swap_copy(&self, s: Stream, items: u64, n: usize, blocks: u32) -> Result<()> {
        launch!(self.swap_copy, [blocks.max(1), 1, 1], [256, 1, 1], s; items, n as i32)
    }

    /// ids [n] (int64) = src[i] where that is a token, else tok[-1 - src[i]]: a step's token ids, those the step before chose read where it left them (`step_ids`); the same
    /// as int32 into ids32 where it is not 0.
    ///
    /// # Safety
    /// src, ids and ids32 (where not 0) hold n entries on the device; tok every row a negative src names.
    pub unsafe fn step_ids(&self, s: Stream, src: u64, tok: u64, ids: u64, ids32: u64, n: usize) -> Result<()> {
        let Some(k) = &self.step_ids else { return crate::ffi::err("these kernels have no step_ids") };
        if n == 0 {
            return Ok(());
        }
        launch!(k, [(n as u32).div_ceil(256), 1, 1], [256, 1, 1], s; src, tok, ids, ids32, n as i32)
    }

    /// Whether the kernels have `step_ids`.
    pub fn has_step_ids(&self) -> bool {
        self.step_ids.is_some()
    }

    pub unsafe fn gather_row(&self, s: Stream, logits: u64, v: usize, row: usize, idx: u64, n: usize, out: u64) -> Result<()> {
        launch!(self.gather, [(n as u32).div_ceil(256), 1, 1], [256, 1, 1], s; logits + (row * v * 2) as u64, idx, n as i32, out)
    }

    /// Rows of W (`rows` rows of `v` floats at `w`) from the step's bf16 `logits`: row r from logits row `src`[r] (int32 on the device), each token's logit where row r's mask (`words`
    /// u32 a row at `mask`: the first nonzero where the row has one, then token t's bit in word 1 + t / 32) allows it, else -inf (`proc_fill`).
    ///
    /// # Safety
    /// As `copy_row`; `src` and `mask` hold `rows` rows, `w` room for them, every row `src` names one of `logits`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn proc_fill(&self, s: Stream, logits: u64, v: usize, src: u64, mask: u64, words: usize, w: u64, rows: usize) -> Result<()> {
        if !v.is_multiple_of(8) || rows == 0 {
            return crate::ffi::err(format!("the rows a draw sees: a vocabulary of {v}, {rows} rows: not supported"));
        }
        launch!(self.proc_fill, [(v as u32 / 8).div_ceil(256), rows as u32, 1], [256, 1, 1], s; logits, v as i32, src, mask, words as i32, w)
    }

    /// The `n` changed tokens' logits into W after `proc_fill` (`proc_entries`): entry i (token, W's row, flags, said: int32 x4) at `ent`, its bias at `bias`, each W row's (repetition,
    /// presence, frequency, -) at `scal`.
    ///
    /// # Safety
    /// As `proc_fill`; `ent` and `bias` hold n entries, each token below v and each row one of W's, a token once a row.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn proc_entries(
        &self,
        s: Stream,
        logits: u64,
        v: usize,
        src: u64,
        mask: u64,
        words: usize,
        ent: u64,
        bias: u64,
        n: usize,
        scal: u64,
        w: u64,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        launch!(self.proc_entries, [(n as u32).div_ceil(256), 1, 1], [256, 1, 1], s; logits, v as i32, src, mask, words as i32, ent, bias, n as i32, scal, w)
    }

    /// `topk_rows` of rows of floats (W, `rows` of `v`): row r's TOPK into `tok` and `val` at row `dst`[r] (int32 on the device; 0: r).
    ///
    /// # Safety
    /// As `copy_row`; `dst` holds `rows` rows and `tok` and `val` TOPK for every row it names.
    pub unsafe fn topk_f32(&self, s: Stream, w: u64, v: usize, rows: usize, dst: u64, tok: u64, val: u64) -> Result<()> {
        if !v.is_multiple_of(8) || v < TOPK || rows == 0 {
            return crate::ffi::err(format!("the candidates of {rows} rows of {v} floats: not supported"));
        }
        launch!(self.topk_f32, [rows as u32, 1, 1], [1024, 1, 1], s; w, v as i32, dst, tok, val)
    }

    /// A draw's total (`sampler::View::total`) of each of `rows` rows of `v` (bf16 logits, or floats where `f32`) at `rows_at`: row r's sum of its tokens' weights about `maxv`[r]
    /// (f32, its largest) at `temperature` into `out`[r] (u64).
    ///
    /// # Safety
    /// As `copy_row`; `maxv` and `out` hold `rows` values.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn draw_total(&self, s: Stream, rows_at: u64, f32: bool, v: usize, rows: u32, maxv: u64, temperature: f32, out: u64) -> Result<()> {
        match f32 {
            true => launch!(self.draw_total.1, [rows, 1, 1], [1024, 1, 1], s; rows_at, v as i32, maxv, temperature, out),
            false => launch!(self.draw_total.0, [rows, 1, 1], [1024, 1, 1], s; rows_at, v as i32, maxv, temperature, out),
        }
    }

    /// A query's bins of the row of `v` at `row` (bf16 logits, or floats where `f32`), into `out` (`DRAW_BINS_BYTES`: 256 counts, u32, then 256 weights, least and greatest keys, u64).
    ///
    /// # Safety
    /// As `copy_row`; `out` holds DRAW_BINS_BYTES.
    pub unsafe fn draw_bins(&self, s: Stream, row: u64, f32: bool, v: usize, q: &Query, out: u64) -> Result<()> {
        let (mx, t, lo, hi, qmin, shift) = (q.max, q.temperature, q.lo, q.hi, q.floor, q.shift as i32);
        match f32 {
            true => launch!(self.draw_bins.1, [1, 1, 1], [1024, 1, 1], s; row, v as i32, mx, t, lo, hi, qmin, shift, out),
            false => launch!(self.draw_bins.0, [1, 1, 1], [1024, 1, 1], s; row, v as i32, mx, t, lo, hi, qmin, shift, out),
        }
    }

    /// A query's tokens of the row of `v` at `row` (bf16 logits, or floats where `f32`): at most `cap`, token (int32) and logit (f32) into `tok` and `val`, how many there are into `n`
    /// (u32, zeroed here).
    ///
    /// # Safety
    /// As `copy_row`; `tok` and `val` hold `cap` values.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn draw_tokens(&self, s: Stream, row: u64, f32: bool, v: usize, q: &Query, cap: usize, n: u64, tok: u64, val: u64) -> Result<()> {
        crate::dev::memset_async(n, 0, 4, s)?;
        let (mx, t, lo, hi, qmin, c) = (q.max, q.temperature, q.lo, q.hi, q.floor, cap as i32);
        let blocks = (v as u32).div_ceil(256).min(1024);
        match f32 {
            true => launch!(self.draw_tokens.1, [blocks, 1, 1], [256, 1, 1], s; row, v as i32, mx, t, lo, hi, qmin, c, n, tok, val),
            false => launch!(self.draw_tokens.0, [blocks, 1, 1], [256, 1, 1], s; row, v as i32, mx, t, lo, hi, qmin, c, n, tok, val),
        }
    }

    /// Every float from -0 to -32 weighed on the GPU (`draw_weight_sums`): the sum of the weights and of each times its float's bits, mod 2^64, into `out` (two u64, zeroed here).
    ///
    /// # Safety
    /// `out` holds two u64 on the stream's device.
    pub unsafe fn draw_weight_sums(&self, s: Stream, out: u64) -> Result<()> {
        crate::dev::memset_async(out, 0, 16, s)?;
        launch!(self.draw_weight_sums, [1024, 1, 1], [256, 1, 1], s; out)
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

    /// The log-sum-exp of v logits given their maximum, into `lse` (f32).
    ///
    /// # Safety
    /// As `copy_row`.
    pub unsafe fn logsumexp(&self, s: Stream, logits: u64, v: i32, maxv: f32, lse: u64) -> Result<()> {
        launch!(self.logsumexp, [1, 1, 1], [1024, 1, 1], s; logits, v, maxv, lse)
    }

    /// A streaming read of `n` 16-byte words at `p` (the memory bandwidth a read-only kernel gets); `out` is one u32.
    ///
    /// # Safety
    /// As `copy_row`; `p` holds 16 n bytes.
    pub unsafe fn read_stream(&self, s: Stream, p: u64, n: u64, out: u64, blocks: u32) -> Result<()> {
        launch!(self.read_stream, [blocks, 1, 1], [256, 1, 1], s; p, n, out)
    }
}

/// Row `row`'s TOPK candidates (token, logit) from the pinned bytes at `p`: the token indices (int32) from byte `tok` on and the logits (f32) from byte `val` on, each row's TOPK after the row before's.
///
/// # Safety
/// `p` holds TOPK (row + 1) tokens at `tok` and as many logits at `val`, aligned for 4-byte reads.
pub(crate) unsafe fn read_candidates(p: *const u8, tok: usize, val: usize, row: usize) -> Vec<(u32, f32)> {
    (0..TOPK).map(|j| (*(p.add(tok + 4 * (row * TOPK + j)) as *const i32) as u32, *(p.add(val + 4 * (row * TOPK + j)) as *const f32))).collect()
}

/// The rows a step's draws see, made on the GPU (`Sampler::proc`, `Resolved`): for each sampled row whose request changes it (penalties, a bias, tokens taken out), the model's row with
/// the changes put on it, in floats (W, a row of the vocabulary each: `proc_fill`, `proc_entries`), and its candidates made of it (`topk64_f32`) at the row's place among the step's,
/// over the model's own; its sums and buckets are made of W by the caller (`row_of`). The rows' masks, changed tokens and penalties go up in one copy from pinned memory; nothing of
/// the row comes back but its candidates.
pub struct ProcGpu {
    vocab: usize,
    /// A row's mask words: its flag and its V / 32 words of bits.
    words: usize,
    /// W, `rows` rows of the vocabulary's floats.
    w: Buf,
    rows: usize,
    /// The rows' inputs on the device and in pinned memory (u32 words): the head rows, the penalties, the masks, the entries and their biases.
    input: Buf,
    stage: Option<Pinned>,
    /// The head rows the last `queue` made W's rows for, in W's order.
    made: Vec<usize>,
}

/// Where `ProcGpu`'s inputs are, in u32 words, for `n` rows of `words` mask words and `ents` entries: the penalties, the masks, the entries (four words each, 16-byte aligned) and
/// the biases; and the words in all. The head rows are first.
fn proc_layout(n: usize, words: usize, ents: usize) -> (usize, usize, usize, usize, usize) {
    let scal = n.next_multiple_of(4);
    let mask = scal + 4 * n;
    let ent = (mask + words * n).next_multiple_of(4);
    let bias = ent + 4 * ents;
    (scal, mask, ent, bias, bias + ents.max(1))
}

impl ProcGpu {
    pub fn new(vocab: usize) -> ProcGpu {
        ProcGpu { vocab, words: 1 + vocab.div_ceil(32), w: Buf::none(), rows: 0, input: Buf::none(), stage: None, made: Vec::new() }
    }

    /// Room for W's rows of `rows` head rows (`plan::draw_rows_bytes`), made where it has less.
    pub fn reserve(&mut self, rows: usize) -> Result<()> {
        if self.rows < rows {
            self.w = Buf::none();
            self.w = Buf::new(crate::plan::draw_rows_bytes(self.vocab, rows) as usize)?;
            self.rows = rows;
        }
        Ok(())
    }

    /// Whether the kernels take the vocabulary (a multiple of 8, at least TOPK).
    pub fn can(&self) -> bool {
        self.vocab.is_multiple_of(8) && self.vocab >= TOPK
    }

    /// For `rows` (a head row of the step and its `Resolved`): W made of the step's `logits` and the rows' candidates made of W into `tok` and `val` at each head row's place (TOPK a row),
    /// queued on the stream. The room is made first (none is queued where it cannot be had: the caller's candidates are then the model's rows' own).
    ///
    /// # Safety
    /// `logits` holds the head rows named (the vocabulary's bf16 each), `tok` and `val` TOPK for each, all live and on the engine's stream; the stream has run every copy the last call
    /// queued (its pinned bytes are free), and no work it queued that reads W is still to run.
    pub unsafe fn queue(&mut self, kern: &Kern, dev: &Dev, rows: &[(usize, &Resolved)], logits: u64, tok: u64, val: u64) -> Result<()> {
        let (v, n) = (self.vocab, rows.len());
        self.made.clear();
        if n == 0 {
            return Ok(());
        }
        if !self.can() {
            return crate::ffi::err(format!("the rows a draw sees: a vocabulary of {v}: not supported"));
        }
        let ents: usize = rows.iter().map(|r| r.1.entries.len()).sum();
        let (o_scal, o_mask, o_ent, o_bias, total) = proc_layout(n, self.words, ents);
        self.reserve(n)?;
        if self.input.bytes < 4 * total || self.stage.is_none() {
            (self.input, self.stage) = (Buf::none(), None);
            self.input = Buf::new(4 * total)?;
            self.stage = Some(Pinned::new(4 * total)?);
        }
        let stage = self.stage.as_mut().expect("made above");
        let host = stage.ptr;
        let st = &mut stage.as_slice_mut::<u32>()[..total];
        let mut k = 0;
        for (i, &(head, r)) in rows.iter().enumerate() {
            st[i] = head as u32;
            st[o_scal + 4 * i..o_scal + 4 * i + 4].copy_from_slice(&[r.repetition.to_bits(), r.presence.to_bits(), r.frequency.to_bits(), 0]);
            let m = &mut st[o_mask + i * self.words..o_mask + (i + 1) * self.words];
            match &r.words {
                Some(w) if w.len() == self.words - 1 => {
                    m[0] = 1;
                    m[1..].copy_from_slice(w);
                }
                Some(_) => return crate::ffi::err("a row's mask not of the vocabulary"),
                None => m[0] = 0,
            }
            for e in &r.entries {
                let flags = e.bias.is_some() as u32 | (e.seen as u32) << 1 | (e.rescue as u32) << 2;
                st[o_ent + 4 * k..o_ent + 4 * k + 4].copy_from_slice(&[e.token, i as u32, flags, e.said]);
                st[o_bias + k] = e.bias.unwrap_or(0.0).to_bits();
                k += 1;
            }
        }
        let (s, inp) = (dev.stream, self.input.ptr);
        copy_h2d_async(inp, host as *const _, 4 * total, s)?;
        let at = |o: usize| inp + 4 * o as u64;
        kern.proc_fill(s, logits, v, inp, at(o_mask), self.words, self.w.ptr, n)?;
        kern.proc_entries(s, logits, v, inp, at(o_mask), self.words, at(o_ent), at(o_bias), ents, at(o_scal), self.w.ptr)?;
        kern.topk_f32(s, self.w.ptr, v, n, inp, tok, val)?;
        self.made = rows.iter().map(|r| r.0).collect();
        Ok(())
    }

    /// W's row (its device address) for head row `head`, where the last `queue` made one.
    pub fn row_of(&self, head: usize) -> Option<u64> {
        self.made.iter().position(|&h| h == head).map(|i| self.w.ptr + (4 * self.vocab * i) as u64)
    }

    /// The head rows the last `queue` made W's rows for.
    pub fn made(&self) -> &[usize] {
        &self.made
    }
}

/// The 64-row groups of the embedding that `tokens` fall in, each once and in order, and for each batch of `per` groups the pair (destination row, row among the batch's decoded groups) of every token
/// whose group is in it.
pub fn embed_plan(tokens: &[u32], per: usize) -> (Vec<i64>, Vec<Vec<(i32, i32)>>) {
    let mut groups: Vec<i64> = tokens.iter().map(|&t| t as i64 / 64).collect();
    groups.sort_unstable();
    groups.dedup();
    let at: std::collections::HashMap<i64, usize> = groups.iter().enumerate().map(|(i, &g)| (g, i)).collect();
    let mut by_batch: Vec<Vec<(i32, i32)>> = vec![Vec::new(); groups.len().div_ceil(per)];
    for (i, &t) in tokens.iter().enumerate() {
        let g = at[&(t as i64 / 64)];
        by_batch[g / per].push((i as i32, (g % per * 64) as i32 + (t % 64) as i32));
    }
    (groups, by_batch)
}

/// kernels/engine.cu's NextPf: a small kernel's spare blocks (a norm's, the attention's preparation's, an activation's) prefetch into L2 the start of the first-wave
/// blocks of the skinny product that comes next.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct NextPf {
    pub(crate) data: u64,
    pub(crate) scales: u64,
    pub(crate) dstride: i64,
    pub(crate) sstride: i64,
    pub(crate) dhead: i32,
    pub(crate) shead: i32,
    pub(crate) regions: i32,
    pub(crate) shift: i32,
}

/// GLYD_G4_PF: "a,n,g" KiB (unset: the default 128,32,8; 0,0,0 none).
pub(crate) fn pf_kib(v: Option<&str>) -> Result<[usize; 3]> {
    let Some(v) = v.filter(|v| !v.is_empty()) else { return Ok([128, 32, 8]) };
    let k: Vec<Option<usize>> = v.split(',').map(|x| x.trim().parse().ok()).collect();
    match k[..] {
        [Some(a), Some(n), Some(g)] => Ok([a, n, g]),
        _ => crate::ffi::err(format!("the prefetch setting {v}: three numbers of KiB, a,n,g")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// No two tiles go by one name, and each tile's name is a kernel kernels/engine.cu defines (the launch's type is the build's, made from that source; the name
    /// is what a product's choice is logged and overridden by).
    #[test]
    fn the_tiles_names_are_kernels_of_the_source_and_no_two_are_one() {
        const SRC: &str = include_str!("../../kernels/engine.cu");
        let tiles: Vec<&str> = PLAIN.iter().chain(FUSED.iter()).map(|t| t.0).collect();
        for n in &tiles {
            assert!(SRC.contains(&format!(" {n}(const bf16* X, const bf16* W, bf16* Y,")), "{n} is not a product kernel of kernels/engine.cu");
        }
        assert_eq!(tiles.iter().collect::<std::collections::HashSet<_>>().len(), tiles.len());
    }

    #[test]
    fn embedding_groups_once_and_rows_gathered() {
        // tokens 70 and 71 share group 1; 5 and 130 are groups 0 and 2; with two groups a batch: batch 0 holds groups 0 and 1, batch 1 group 2
        let (groups, batches) = embed_plan(&[70, 5, 71, 130, 70], 2);
        assert_eq!(groups, vec![0, 1, 2]);
        assert_eq!(batches[0], vec![(0, 64 + 6), (1, 5), (2, 64 + 7), (4, 64 + 6)]);
        assert_eq!(batches[1], vec![(3, 2)]);
        assert_eq!(embed_plan(&[], 4).1.len(), 0);
    }

    /// A row's candidates are read from where the copies put them: several rows, a step's TOPK of each one after the other from two places, and one row, its tokens and
    /// then its logits from the start.
    #[test]
    fn candidates_are_read_where_the_copies_put_them() {
        const ROWS: usize = 8;
        const TOK_AT: usize = 2048;
        const VAL_AT: usize = 4096;
        let (tok, val) = (TOK_AT / 4, VAL_AT / 4);
        let mut words = vec![0u32; val + ROWS * TOPK];
        for row in 0..ROWS {
            for j in 0..TOPK {
                words[tok + row * TOPK + j] = (1000 * row + j) as u32;
                words[val + row * TOPK + j] = (row as f32 - j as f32 / 8.0).to_bits();
            }
        }
        for row in [0, 1, ROWS - 1] {
            // SAFETY: the buffer holds ROWS rows of TOPK tokens and logits at the offsets, 4-byte aligned.
            let got = unsafe { read_candidates(words.as_ptr() as *const u8, TOK_AT, VAL_AT, row) };
            assert_eq!(got, (0..TOPK).map(|j| ((1000 * row + j) as u32, row as f32 - j as f32 / 8.0)).collect::<Vec<_>>(), "row {row}");
        }
        let mut one = vec![0u32; 2 * TOPK];
        for j in 0..TOPK {
            one[j] = 7 * j as u32;
            one[TOPK + j] = (-(j as f32)).to_bits();
        }
        // SAFETY: as above, one row.
        let got = unsafe { read_candidates(one.as_ptr() as *const u8, 0, 4 * TOPK, 0) };
        assert_eq!(got, (0..TOPK).map(|j| (7 * j as u32, -(j as f32))).collect::<Vec<_>>());
    }
}
