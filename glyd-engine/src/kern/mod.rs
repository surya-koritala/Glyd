//! The engine's own kernels (kernels/engine.cu) with typed launches. Device addresses are `u64`.

pub mod gemma4;
pub mod glm;
pub mod nemotron;
pub mod qwen35;

pub(crate) use gemma4::{pf_kib, G4Pf, KernG4};
pub use glm::{mla_positions, mla_smem, Hc, KernGlm, GM_CHUNK_SMEM, GM_D, GM_HC, GM_L, GM_NP, HC_FEW};
pub use nemotron::{KernNh, SsdWs, NH_L, NH_N, NH_P};
pub use qwen35::Kern35;

use crate::dev::{copy_d2h_async, copy_h2d_async, ArgsOf, Buf, ByArgs, Dev, Pinned, Typed};
use crate::ffi::{Result, Stream};
use crate::launch;
use crate::sampler::{Need, Top};

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
    /// `topk` with a mask per row (a grammar's): None where the kernels have none.
    topk_masked: Option<crate::kt::topk64_bf16_masked>,
    gather: crate::kt::gather_bf16,
    kv12_rows: crate::kt::kv12_rows,
    row_tables: crate::kt::row_tables,
    swap_copy: crate::kt::swap_copy,
    /// None where the kernels have none (a step is then not launched before the one before it is read).
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
            topk_masked: dev.typed::<crate::kt::topk64_bf16_masked>().ok(),
            gather: dev.typed()?,
            kv12_rows: dev.typed()?,
            row_tables: dev.typed()?,
            swap_copy: dev.typed()?,
            step_ids: dev.typed::<crate::kt::step_ids>().ok(),
            expsum: dev.typed()?,
            logsumexp: dev.typed()?,
            read_stream: dev.typed()?,
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

/// The grammar masks of a step's rows on the GPU: `apply` waits for each row's mask (the workers make them beside the step), copies them to the device and, behind the step's work on the stream, makes the step's candidates again with
/// the masked top-k kernel, over the unmasked ones the step made (a graph's, whose kernel is fixed); the host then draws from candidates that hold the allowed tokens (`Sampler::sample_grammar`). No mask for a row: the row is as the step made it.
pub struct MaskGpu {
    dev: Buf,
    stage: Pinned,
    vocab: usize,
    /// A row's words: its flag and its V / 32 words of bits.
    words: usize,
    rows: usize,
}

impl MaskGpu {
    pub fn new(vocab: usize, rows: usize) -> Result<MaskGpu> {
        let words = 1 + vocab.div_ceil(32);
        Ok(MaskGpu { dev: Buf::zeroed(4 * words * rows)?, stage: Pinned::new(4 * words * rows)?, vocab, words, rows })
    }

    /// Whether the masks of rows can be put on the GPU (the kernels have the masked top-k, and take the vocabulary).
    pub fn can(&self, kern: &Kern) -> bool {
        kern.has_masked_topk() && self.vocab.is_multiple_of(8) && self.vocab >= TOPK
    }

    /// For the rows `masks` names (None: no mask): the masked candidates of every row into `tok` and `val` (as many rows as there are masks), after the work queued on the stream. Nothing is done where no row has a mask
    /// or a row's worker died (that row keeps its unmasked candidates, which the host's draw checks against the mask as it always did). Returns whether it made any.
    ///
    /// # Safety
    /// `logits` holds a row of the vocabulary for each of `masks`, `tok` and `val` TOPK of each, all live and on the engine's stream; the stream was synchronized since the last call (the staging bytes are free).
    pub unsafe fn apply(&self, kern: &Kern, dev: &Dev, masks: &[Option<crate::constrain::MaskHandle>], logits: u64, tok: u64, val: u64) -> Result<bool> {
        if masks.is_empty() || masks.len() > self.rows || !self.can(kern) {
            return Ok(false);
        }
        let stage = std::slice::from_raw_parts_mut(self.stage.ptr as *mut u32, self.words * masks.len());
        let mut any = false;
        for (r, m) in masks.iter().enumerate() {
            let row = &mut stage[r * self.words..(r + 1) * self.words];
            row.fill(0);
            if let Some(vob) = m.as_ref().and_then(|m| m.words()).filter(|v| v.len() == self.vocab) {
                row[1..1 + vob.as_slice().len().min(self.words - 1)].copy_from_slice(&vob.as_slice()[..vob.as_slice().len().min(self.words - 1)]);
                row[0] = 1;
                any = true;
            }
        }
        if !any {
            return Ok(false);
        }
        copy_h2d_async(self.dev.ptr, self.stage.ptr as *const _, 4 * self.words * masks.len(), dev.stream)?;
        kern.topk_rows_masked(dev.stream, logits, self.vocab as i32, masks.len() as u32, tok, val, self.dev.ptr, self.words as i32)?;
        Ok(true)
    }
}

/// The GPU's side of the sampler's candidates for an engine with one row of logits (Qwen3.5's and Nemotron 3 Super's, whose `Gen::reply` draws on the host): behind a step that `ask` names (`Sampler::top_need`), `queue` makes the row's
/// 64 largest logits and, for a top-p with no top-k, the sum of its exponentials, and copies them to the host in the step's one synchronize; `read` gives them. A draw then reads 520 bytes in place of the row (131,072 to
/// 248,000 logits to copy, sort and exponentiate on the host). `off` (GLYD_ENGINE_HOST_SAMPLING=1, to compare): none are made and the row is read.
pub struct TopGpu {
    tok: Buf,
    val: Buf,
    sum: Buf,
    stage: Pinned,
    vocab: usize,
    /// What the next step is asked for.
    pub ask: Option<Need>,
    /// What the last `queue` made.
    got: Option<Need>,
    pub off: bool,
    /// The mask of the next step's grammar (`Sampler::mask_handle`), and where it is put on the GPU.
    pub mask: Option<crate::constrain::MaskHandle>,
    masker: Option<MaskGpu>,
}

impl TopGpu {
    pub fn new(vocab: usize) -> Result<TopGpu> {
        Ok(TopGpu {
            tok: Buf::zeroed(4 * TOPK)?,
            val: Buf::zeroed(4 * TOPK)?,
            sum: Buf::zeroed(8)?,
            stage: Pinned::new(8 * TOPK + 8)?,
            vocab,
            ask: None,
            got: None,
            off: crate::tune!("GLYD_ENGINE_HOST_SAMPLING").is_some(),
            mask: None,
            masker: MaskGpu::new(vocab, 1).ok(),
        })
    }

    /// Behind the step's work on the stream: the candidates of the row `ask` names, with the copies to the host (the caller synchronizes). The kernels take a vocabulary of a multiple of 8 and at least TOPK; a smaller one has none.
    ///
    /// # Safety
    /// `logits` is a row of the vocabulary's size and `maxv` its largest (f32), both live and made on the engine's stream.
    pub unsafe fn queue(&mut self, kern: &Kern, dev: &Dev, logits: u64, maxv: u64) -> Result<()> {
        self.got = self.ask.filter(|_| !self.off && self.vocab.is_multiple_of(8) && self.vocab >= TOPK);
        let Some(need) = self.got else { return Ok(()) };
        let (s, host) = (dev.stream, self.stage.ptr as *mut u8);
        kern.topk_rows(s, logits, self.vocab as i32, 1, self.tok.ptr, self.val.ptr)?;
        // (a grammar's mask: the candidates made again with it, the mask put on the row before the top-k)
        if let (Some(m), Some(g)) = (self.mask.take(), &self.masker) {
            g.apply(kern, dev, &[Some(m)], logits, self.tok.ptr, self.val.ptr)?;
        }
        copy_d2h_async(host as *mut _, self.tok.ptr, 4 * TOPK, s)?;
        copy_d2h_async(host.add(4 * TOPK) as *mut _, self.val.ptr, 4 * TOPK, s)?;
        if need.mass {
            kern.exp_sum_rows(s, logits, self.vocab as i32, 1, maxv, need.temperature, self.sum.ptr)?;
            copy_d2h_async(host.add(8 * TOPK) as *mut _, self.sum.ptr, 8, s)?;
        }
        Ok(())
    }

    /// What the last `queue` made (the stream synchronized since): None where it made nothing.
    pub fn read(&self) -> Option<Top> {
        let need = self.got?;
        let p = self.stage.ptr as *const u8;
        // SAFETY: the pinned bytes were copied behind the step (the stream synchronized since): TOPK tokens (int32), TOPK logits (f32), then the sum (f64) where asked.
        unsafe { Some(Top { cands: read_candidates(p, 0, 4 * TOPK, 0), mass: need.mass.then(|| *(p.add(8 * TOPK) as *const f64)) }) }
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

    #[test]
    fn embedding_groups_once_and_rows_gathered() {
        // tokens 70 and 71 share group 1; 5 and 130 are groups 0 and 2; with two groups a batch: batch 0 holds groups 0 and 1, batch 1 group 2
        let (groups, batches) = embed_plan(&[70, 5, 71, 130, 70], 2);
        assert_eq!(groups, vec![0, 1, 2]);
        assert_eq!(batches[0], vec![(0, 64 + 6), (1, 5), (2, 64 + 7), (4, 64 + 6)]);
        assert_eq!(batches[1], vec![(3, 2)]);
        assert_eq!(embed_plan(&[], 4).1.len(), 0);
    }

    /// A row's candidates are read from where the copies put them: several rows, a step's TOPK of each one after the other from two places, and the one row of the
    /// engines' `TopGpu`, its tokens and then its logits from the start.
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
