//! Hummingbird's codec for every mixture of experts (docs/design/hummingbird.md, #429), on the host: the layer's rotation for any width in
//! runs of 128, each row's codebook, nearest and GPTQ rounding, the recipe that gives each expert its bits by use, and the usage file the
//! corpus pass writes. What `nemotron_hbr.py` did for Nemotron 3 Super, for any expert of width W (its input: the hidden size or the latent)
//! and intermediate size I:
//!
//! - `gate_up` [U, W] (U = 2I gated, the gate's rows first; I otherwise): row n is R w_n;
//! - `down_t` [I, W]: row n is R d_n, d_n = down[:, n].
//!
//! R = (I ⊗ H_b) diag(s): H_b the orthonormal Walsh-Hadamard matrix of order b in natural (Sylvester) order on each run of b values, b the
//! largest power of two that divides W (128 to 1,024), s the layer's signs. A row is held as glyd-gpu's codebook rows (`codebook.rs`): codes of
//! 2 or 3 bits, a bf16 scale a run of 32 (the run's absmax), a bf16 codebook of 2^bits entries in units of the scale (Lloyd-Max of the row's
//! run-normalised values). A weight is scale * entry, the f32 product of two bf16 (exact).

use crate::ffi::{err, Error, Result};
use glyd_gpu::codebook::{decode_codebook_rows, pack_codebook_rows, CbRows};
use glyd_gpu::json::{self, Value};

/// Values a scale covers.
pub const RUN: usize = 32;
/// The rotation's smallest and largest block.
pub const MIN_BLOCK: usize = 128;
pub const MAX_BLOCK: usize = 1024;
/// The recipe hb40's share (percent) of a layer's experts at three bits.
pub const HB40_PERCENT: usize = 40;
/// GPTQ's damping (a share of the Hessian's mean diagonal) and the fewest tokens an expert needs for it (the rest are rounded to nearest).
pub const DAMP: f64 = 0.01;
pub const GPTQ_MIN_TOKENS: u64 = 2048;
/// Lloyd-Max's passes.
const LLOYD_ITERS: usize = 25;
/// The usage file's format.
pub const USAGE_FORMAT: &str = "glyd-hb-usage/1";

/// The rotation's block for rows of `w` values: the largest power of two that divides `w`, at most 1,024; None where it is below 128.
pub fn block_of(w: usize) -> Option<usize> {
    if w == 0 {
        return None;
    }
    let b = (1usize << w.trailing_zeros()).min(MAX_BLOCK);
    (b >= MIN_BLOCK).then_some(b)
}

fn fnv1a(s: &[u8]) -> u64 {
    s.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01B3))
}

/// The signs of the MoE layer at the model's block `layer`, for rows of `w` values: +1 or -1 from xorshift64* seeded with the FNV-1a hash of
/// `glyd-hb/<layer>`. A directory saves them; a reader takes the saved ones.
pub fn signs(layer: usize, w: usize) -> Vec<i8> {
    let mut x = fnv1a(format!("glyd-hb/{layer}").as_bytes()) | 1;
    (0..w)
        .map(|_| {
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            if (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 63) == 1 {
                1
            } else {
                -1
            }
        })
        .collect()
}

/// The unscaled Walsh-Hadamard transform of each run of `b` values of `x`, natural order: for h = 1, 2, .. b / 2, the pairs (i, i + h) of each
/// group of 2h become (a + c, a - c) (the order the kernels use, so the sums are the same ones).
fn wht(x: &mut [f32], b: usize) {
    for run in x.chunks_exact_mut(b) {
        let mut h = 1;
        while h < b {
            for g in (0..b).step_by(2 * h) {
                for i in g..g + h {
                    let (a, c) = (run[i], run[i + h]);
                    run[i] = a + c;
                    run[i + h] = a - c;
                }
            }
            h *= 2;
        }
    }
}

/// x <- R x for a row (or rows end to end) of the width of `signs`: the signs, then H_b on each run of `b`, over sqrt b.
pub fn rotate(x: &mut [f32], signs: &[i8], b: usize) {
    let w = signs.len();
    debug_assert!(w > 0 && x.len().is_multiple_of(w) && w.is_multiple_of(b));
    for row in x.chunks_exact_mut(w) {
        row.iter_mut().zip(signs).for_each(|(v, &s)| *v *= s as f32);
        wht(row, b);
    }
    let k = 1.0 / (b as f32).sqrt();
    x.iter_mut().for_each(|v| *v *= k);
}

/// x <- R^T x: H_b on each run of `b` over sqrt b, then the signs.
pub fn unrotate(x: &mut [f32], signs: &[i8], b: usize) {
    let w = signs.len();
    debug_assert!(w > 0 && x.len().is_multiple_of(w) && w.is_multiple_of(b));
    let k = 1.0 / (b as f32).sqrt();
    for row in x.chunks_exact_mut(w) {
        wht(row, b);
        row.iter_mut().zip(signs).for_each(|(v, &s)| *v *= s as f32 * k);
    }
}

pub fn bf16(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return 0x7FC0;
    }
    ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) as u16
}

pub fn f32_of(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// A k-entry codebook for the values `x`: the weighted one-dimensional k-means (Lloyd-Max) from a uniform start on -1..1, 25 passes, sorted,
/// each entry rounded to bf16. `wt`: each value's weight (None: equal).
pub fn lloyd_max(x: &[f32], wt: Option<&[f32]>, k: usize) -> Vec<u16> {
    let mut order: Vec<usize> = (0..x.len()).collect();
    order.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
    let xs: Vec<f64> = order.iter().map(|&i| x[i] as f64).collect();
    let ws: Vec<f64> = order.iter().map(|&i| wt.map_or(1.0, |w| w[i] as f64)).collect();
    let (mut cx, mut cw) = (vec![0.0f64; xs.len() + 1], vec![0.0f64; xs.len() + 1]);
    for i in 0..xs.len() {
        cx[i + 1] = cx[i] + xs[i] * ws[i];
        cw[i + 1] = cw[i] + ws[i];
    }
    let mut c: Vec<f64> = (0..k).map(|j| if k == 1 { 0.0 } else { -1.0 + 2.0 * j as f64 / (k - 1) as f64 }).collect();
    for _ in 0..LLOYD_ITERS {
        // (cluster j holds the sorted values up to each midpoint: bounds[j] .. bounds[j + 1])
        let mut bounds = Vec::with_capacity(k + 1);
        bounds.push(0usize);
        for j in 0..k - 1 {
            let m = (c[j] + c[j + 1]) / 2.0;
            bounds.push(xs.partition_point(|&v| v <= m));
        }
        bounds.push(xs.len());
        for j in 0..k {
            let (lo, hi) = (bounds[j], bounds[j + 1]);
            let den = cw[hi] - cw[lo];
            if hi > lo && den > 0.0 {
                c[j] = (cx[hi] - cx[lo]) / den;
            }
        }
        c.sort_by(f64::total_cmp);
    }
    c.iter().map(|&v| bf16(v as f32)).collect()
}

/// The index of the nearest entry of an ascending codebook (`book`, as f32) to `v`: the number of midpoints below it (a tie goes to the lower entry).
fn nearest(book: &[f32], v: f32) -> u8 {
    let mut i = 0;
    while i + 1 < book.len() && (book[i] + book[i + 1]) / 2.0 < v {
        i += 1;
    }
    i as u8
}

/// Rows coded: each value's code, each run's scale (bf16), each row's codebook (bf16), and the values they make (f32).
#[derive(Clone, Debug, PartialEq)]
pub struct Coded {
    pub rows: usize,
    pub cols: usize,
    pub bits: u32,
    pub codes: Vec<u8>,
    pub scales: Vec<u16>,
    pub books: Vec<u16>,
    pub q: Vec<f32>,
}

impl Coded {
    fn new(rows: usize, cols: usize, bits: u32) -> Coded {
        Coded { rows, cols, bits, codes: vec![0; rows * cols], scales: vec![0; rows * cols / RUN], books: vec![0; rows << bits], q: vec![0.0; rows * cols] }
    }

    /// The rows in glyd-gpu's codebook row layout.
    pub fn pack(&self) -> Result<CbRows> {
        pack_codebook_rows(&self.codes, &self.scales, &self.books, self.bits, self.rows, self.cols)
            .ok_or_else(|| Error(format!("{} x {} rows at {} bits that do not pack (a scale or an entry not finite)", self.rows, self.cols, self.bits)))
    }
}

/// The run's absmax, at least 1e-30.
fn amax(run: &[f32]) -> f32 {
    run.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30)
}

/// A row's codebook from its values: Lloyd-Max of the run-normalised values (each value over its run's absmax), weighted by `wt` where given.
fn fit_book(row: &[f32], wt: Option<&[f32]>, bits: u32) -> Vec<u16> {
    let norm: Vec<f32> = row
        .chunks_exact(RUN)
        .flat_map(|r| {
            let a = amax(r);
            r.iter().map(move |v| v / a)
        })
        .collect();
    lloyd_max(&norm, wt, 1 << bits)
}

/// One run of a row rounded with its codebook: the run's scale (its absmax in bf16) and each value's code; returns the scale and writes codes and values.
fn round_run(run: &[f32], book: &[f32], codes: &mut [u8], q: &mut [f32]) -> u16 {
    let sc = bf16(amax(run));
    let s = f32_of(sc);
    for ((v, c), o) in run.iter().zip(codes.iter_mut()).zip(q.iter_mut()) {
        *c = nearest(book, v / s);
        *o = book[*c as usize] * s;
    }
    sc
}

/// Rows `x` [rows, cols] (already in the rotated basis) rounded to nearest, each with a codebook of its own fitted to it.
pub fn round_nearest(x: &[f32], rows: usize, cols: usize, bits: u32) -> Result<Coded> {
    check_shape(x.len(), rows, cols, bits)?;
    let mut out = Coded::new(rows, cols, bits);
    let k = 1usize << bits;
    for n in 0..rows {
        let row = &x[n * cols..(n + 1) * cols];
        let book = fit_book(row, None, bits);
        let bookf: Vec<f32> = book.iter().map(|&b| f32_of(b)).collect();
        for (r, run) in row.chunks_exact(RUN).enumerate() {
            let at = n * cols + r * RUN;
            out.scales[n * cols / RUN + r] = round_run(run, &bookf, &mut out.codes[at..at + RUN], &mut out.q[at..at + RUN]);
        }
        out.books[n * k..(n + 1) * k].copy_from_slice(&book);
    }
    Ok(out)
}

fn check_shape(len: usize, rows: usize, cols: usize, bits: u32) -> Result<()> {
    if !(2..=3).contains(&bits) || rows == 0 || cols == 0 || !cols.is_multiple_of(RUN) || len != rows * cols {
        return err(format!("{len} values as {rows} rows of {cols} at {bits} bits: rows of runs of {RUN}, 2 or 3 bits"));
    }
    Ok(())
}

// ---- dense linear algebra for GPTQ (float64, row-major n x n) ------------------------------------------------------------------------------------------

/// The lower Cholesky factor L of a symmetric positive definite `a` (a = L L^T); None where it is not positive definite.
fn chol_lower(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0f64; n * n];
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= l[j * n + k] * l[j * n + k];
        }
        if d.is_nan() || d <= 0.0 {
            return None;
        }
        let dj = d.sqrt();
        l[j * n + j] = dj;
        for i in j + 1..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = s / dj;
        }
    }
    Some(l)
}

/// a^-1 from its lower Cholesky factor.
fn inverse_from_chol(l: &[f64], n: usize) -> Vec<f64> {
    // (L^-1 by forward substitution, then a^-1 = L^-T L^-1)
    let mut li = vec![0.0f64; n * n];
    for c in 0..n {
        li[c * n + c] = 1.0 / l[c * n + c];
        for i in c + 1..n {
            let mut s = 0.0;
            for k in c..i {
                s -= l[i * n + k] * li[k * n + c];
            }
            li[i * n + c] = s / l[i * n + i];
        }
    }
    let mut inv = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = 0.0;
            for k in i..n {
                s += li[k * n + i] * li[k * n + j];
            }
            inv[i * n + j] = s;
            inv[j * n + i] = s;
        }
    }
    inv
}

/// The upper factor U of GPTQ: a damped Hessian's inverse as U^T U. None where the damped Hessian, or its inverse, is not positive definite.
fn gptq_factor(h: &mut [f64], n: usize, damp: f64) -> Option<Vec<f64>> {
    let mean = (0..n).map(|i| h[i * n + i]).sum::<f64>() / n as f64;
    for i in 0..n {
        h[i * n + i] += damp * mean;
    }
    let inv = inverse_from_chol(&chol_lower(h, n)?, n);
    let l = chol_lower(&inv, n)?;
    // (U = L^T)
    let mut u = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            u[i * n + j] = l[j * n + i];
        }
    }
    Some(u)
}

fn permute_sym(h: &[f64], n: usize, perm: &[usize]) -> Vec<f64> {
    let mut out = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = h[perm[i] * n + perm[j]];
        }
    }
    out
}

/// GPTQ over neuron rows (act-order, error feedback): `w` [n, cols] the rows (rotated), `h` [n, n] the Hessian of the neurons' activations,
/// E[a a^T]. The neurons are taken in the order of falling H_nn; a whole row (its run scales, its codebook) is rounded a step and the rows
/// after it re-fitted: d_r <- d_r - (delta_i / U_ii) U_ir, U the upper factor of the damped H^-1. Each row's codebook is fitted before the loop
/// on the rows as they were; its run scales are taken from the row at its turn. None where the damped Hessian is not positive definite.
pub fn gptq_neurons(w: &[f32], n: usize, cols: usize, h: &[f64], bits: u32, damp: f64) -> Result<Option<Coded>> {
    check_shape(w.len(), n, cols, bits)?;
    if h.len() != n * n {
        return err(format!("a Hessian of {} values for {n} neurons", h.len()));
    }
    let mut perm: Vec<usize> = (0..n).collect();
    perm.sort_by(|&a, &b| h[b * n + b].total_cmp(&h[a * n + a]));
    let mut hp = permute_sym(h, n, &perm);
    let Some(u) = gptq_factor(&mut hp, n, damp) else { return Ok(None) };
    let mut rows: Vec<f64> = perm.iter().flat_map(|&p| w[p * cols..(p + 1) * cols].iter().map(|&v| v as f64)).collect();
    let k = 1usize << bits;
    let books: Vec<Vec<u16>> = perm.iter().map(|&p| fit_book(&w[p * cols..(p + 1) * cols], None, bits)).collect();
    let mut out = Coded::new(n, cols, bits);
    let mut row = vec![0f32; cols];
    let mut e = vec![0f64; cols];
    for i in 0..n {
        row.iter_mut().zip(&rows[i * cols..(i + 1) * cols]).for_each(|(o, &v)| *o = v as f32);
        let p = perm[i];
        let bookf: Vec<f32> = books[i].iter().map(|&b| f32_of(b)).collect();
        for (r, run) in row.chunks_exact(RUN).enumerate() {
            let at = p * cols + r * RUN;
            out.scales[p * cols / RUN + r] = round_run(run, &bookf, &mut out.codes[at..at + RUN], &mut out.q[at..at + RUN]);
        }
        out.books[p * k..(p + 1) * k].copy_from_slice(&books[i]);
        let uii = u[i * n + i];
        for j in 0..cols {
            e[j] = (rows[i * cols + j] - out.q[p * cols + j] as f64) / uii;
        }
        for r in i + 1..n {
            let f = u[i * n + r];
            if f != 0.0 {
                for j in 0..cols {
                    rows[r * cols + j] -= f * e[j];
                }
            }
        }
    }
    Ok(Some(out))
}

/// GPTQ over the columns (the inputs, act-order by runs of 32): `w` [rows, cols] (rotated), `h` [cols, cols] the Hessian of the rotated inputs,
/// E[x' x'^T]. The runs are taken in the order of their falling sums of H's diagonal; a column of every row is rounded a step, each row with its
/// own codebook (fitted before the loop, its values weighted by H's diagonal) and the run's scale taken at the run's first column; the columns
/// after it are re-fitted. An input no token moves (H_jj = 0) has its weights left out (zero). None where the damped Hessian is not positive definite.
pub fn gptq_columns(w: &[f32], rows: usize, cols: usize, h: &[f64], bits: u32, damp: f64) -> Result<Option<Coded>> {
    check_shape(w.len(), rows, cols, bits)?;
    if h.len() != cols * cols {
        return err(format!("a Hessian of {} values for {cols} inputs", h.len()));
    }
    let runs = cols / RUN;
    let mut h = h.to_vec();
    let mut wt: Vec<f64> = (0..cols).map(|j| h[j * cols + j]).collect();
    let dead: Vec<bool> = wt.iter().map(|&d| d == 0.0).collect();
    for j in 0..cols {
        if dead[j] {
            h[j * cols + j] = 1.0;
        }
    }
    let mut order: Vec<usize> = (0..runs).collect();
    let run_sum = |r: usize| (r * RUN..(r + 1) * RUN).map(|j| h[j * cols + j]).sum::<f64>();
    order.sort_by(|&a, &b| run_sum(b).total_cmp(&run_sum(a)));
    let perm: Vec<usize> = order.iter().flat_map(|&r| r * RUN..(r + 1) * RUN).collect();
    let mut hp = permute_sym(&h, cols, &perm);
    wt = perm.iter().map(|&j| wt[j]).collect();
    let Some(u) = gptq_factor(&mut hp, cols, damp) else { return Ok(None) };
    // (the rows in act-order of their columns, dead inputs zero)
    let mut m: Vec<f64> = (0..rows).flat_map(|r| perm.iter().map(move |&j| (r, j))).map(|(r, j)| if dead[j] { 0.0 } else { w[r * cols + j] as f64 }).collect();
    let mean = wt.iter().sum::<f64>() / cols as f64;
    let wts: Vec<f32> = wt.iter().map(|&x| (x / mean.max(1e-30) + 1e-6) as f32).collect();
    let k = 1usize << bits;
    let mut out = Coded::new(rows, cols, bits);
    let mut books = Vec::with_capacity(rows);
    for r in 0..rows {
        let row: Vec<f32> = m[r * cols..(r + 1) * cols].iter().map(|&v| v as f32).collect();
        let book = fit_book(&row, Some(&wts), bits);
        out.books[r * k..(r + 1) * k].copy_from_slice(&book);
        books.push(book.iter().map(|&b| f32_of(b)).collect::<Vec<f32>>());
    }
    let mut sc = vec![1f32; rows];
    let mut e = vec![0f64; rows];
    for i in 0..cols {
        let (pos, j0) = (i / RUN, perm[i]);
        if i % RUN == 0 {
            for r in 0..rows {
                let run: Vec<f32> = m[r * cols + i..r * cols + i + RUN].iter().map(|&v| v as f32).collect();
                let s = bf16(amax(&run));
                out.scales[r * runs + order[pos]] = s;
                sc[r] = f32_of(s);
            }
        }
        for r in 0..rows {
            let v = m[r * cols + i] as f32;
            let c = nearest(&books[r], v / sc[r]);
            let q = books[r][c as usize] * sc[r];
            out.codes[r * cols + j0] = c;
            out.q[r * cols + j0] = q;
            e[r] = (m[r * cols + i] - q as f64) / u[i * cols + i];
        }
        for j in i + 1..cols {
            let f = u[i * cols + j];
            if f != 0.0 {
                for r in 0..rows {
                    m[r * cols + j] -= e[r] * f;
                }
            }
        }
    }
    Ok(Some(out))
}

// ---- the recipe and the usage file ---------------------------------------------------------------------------------------------------------------------

/// Each expert's bits from its use (`counts`, the corpus pass's choices of it): the `percent` most used of the layer (rounded up) at 3 bits,
/// the rest at 2; ties go to the lower number.
pub fn bits_by_use(counts: &[u64], percent: usize) -> Vec<u32> {
    let threes = (counts.len() * percent).div_ceil(100);
    let mut order: Vec<usize> = (0..counts.len()).collect();
    order.sort_by(|&a, &b| counts[b].cmp(&counts[a]).then(a.cmp(&b)));
    let mut bits = vec![2u32; counts.len()];
    for &e in &order[..threes.min(counts.len())] {
        bits[e] = 3;
    }
    bits
}

/// A MoE layer's use: the model's block it is, and each expert's count of choices.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerUse {
    pub block: usize,
    pub counts: Vec<u64>,
}

/// The corpus pass's counts (`calibrate-experts`): the model, the tokens the pass ran, and each MoE layer's use.
#[derive(Clone, Debug, PartialEq)]
pub struct Usage {
    pub model: String,
    pub tokens: u64,
    pub layers: Vec<LayerUse>,
}

impl Usage {
    pub fn to_json(&self) -> String {
        let layers: Vec<String> = self
            .layers
            .iter()
            .map(|l| format!("{{\"block\":{},\"counts\":[{}]}}", l.block, l.counts.iter().map(u64::to_string).collect::<Vec<_>>().join(",")))
            .collect();
        format!(
            "{{\"format\":\"{USAGE_FORMAT}\",\"model\":{},\"tokens\":{},\"layers\":[{}]}}\n",
            json::to_compact(&Value::from(self.model.as_str())),
            self.tokens,
            layers.join(",")
        )
    }

    pub fn parse(text: &str) -> Result<Usage> {
        let v = json::parse(text).map_err(|e| Error(format!("the usage file: {e}")))?;
        if v.get("format").and_then(Value::as_str) != Some(USAGE_FORMAT) {
            return err(format!("the usage file is not {USAGE_FORMAT}"));
        }
        let mut layers = Vec::new();
        for (i, l) in v.get("layers").and_then(Value::as_array).ok_or_else(|| Error("the usage file has no layers".into()))?.iter().enumerate() {
            let block = l.get("block").and_then(Value::as_u64).ok_or_else(|| Error(format!("the usage file's layer {i} has no block")))? as usize;
            let counts =
                l.get("counts").and_then(Value::as_u64s).filter(|c| !c.is_empty()).ok_or_else(|| Error(format!("the usage file's layer {i} has no counts")))?;
            layers.push(LayerUse { block, counts });
        }
        Ok(Usage {
            model: v.get("model").and_then(Value::as_str).unwrap_or("").to_string(),
            tokens: v.get("tokens").and_then(Value::as_u64).unwrap_or(0),
            layers,
        })
    }

    /// The use of the MoE layer at block `block`.
    pub fn layer(&self, block: usize) -> Option<&LayerUse> {
        self.layers.iter().find(|l| l.block == block)
    }
}

// ---- an expert coded -----------------------------------------------------------------------------------------------------------------------------------

/// An expert's two matrices as codebook rows in its layer's rotated basis.
#[derive(Clone, Debug, PartialEq)]
pub struct ExpertRows {
    pub bits: u32,
    /// [U, W]: the gate's rows then the up's (gated), or the up's.
    pub gate_up: CbRows,
    /// [I, W]: down's columns as rows.
    pub down_t: CbRows,
}

/// A MoE layer's experts coded: the model's block, the rows' width and the rotation's block, its signs, and each expert in the router's order.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerRows {
    pub block: usize,
    pub width: usize,
    pub rot: usize,
    pub inter: usize,
    pub gated: bool,
    pub signs: Vec<i8>,
    pub experts: Vec<ExpertRows>,
}

/// The Hessians of one expert, where the corpus pass kept them: E[x' x'^T] of its rotated inputs [W, W] and E[h h^T] of its activations [I, I],
/// and the tokens they are over.
pub struct ExpertHessians<'a> {
    pub inputs: &'a [f64],
    pub acts: &'a [f64],
    pub tokens: u64,
}

/// One expert coded: `gate_up` [U, W] and `down` [W, I] as the release has them (f32, exact), rotated by `signs` (blocks of `rot`) and rounded to
/// `bits`: by GPTQ where `hess` is given and has enough tokens (and positive definite Hessians), else to nearest. Returns the rows and whether GPTQ made them.
pub fn code_expert(
    gate_up: &[f32],
    down: &[f32],
    width: usize,
    inter: usize,
    signs: &[i8],
    rot: usize,
    bits: u32,
    hess: Option<ExpertHessians>,
) -> Result<(ExpertRows, bool)> {
    if signs.len() != width || !width.is_multiple_of(rot) || inter == 0 || !gate_up.len().is_multiple_of(width) || down.len() != width * inter {
        return err(format!(
            "an expert of {} and {} values for a width of {width} (signs {}, blocks of {rot}) and {inter} neurons",
            gate_up.len(),
            down.len(),
            signs.len()
        ));
    }
    let u = gate_up.len() / width;
    let mut gu = gate_up.to_vec();
    rotate(&mut gu, signs, rot);
    // (down [W, I] as its neurons' rows [I, W], rotated)
    let mut dt = vec![0f32; inter * width];
    for o in 0..width {
        for n in 0..inter {
            dt[n * width + o] = down[o * inter + n];
        }
    }
    rotate(&mut dt, signs, rot);
    let gptq = hess.as_ref().filter(|h| h.tokens >= GPTQ_MIN_TOKENS && h.inputs.len() == width * width && h.acts.len() == inter * inter);
    let (a, b, by_gptq) = match gptq {
        Some(h) => match (gptq_columns(&gu, u, width, h.inputs, bits, DAMP)?, gptq_neurons(&dt, inter, width, h.acts, bits, DAMP)?) {
            (Some(a), Some(b)) => (a, b, true),
            _ => (round_nearest(&gu, u, width, bits)?, round_nearest(&dt, inter, width, bits)?, false),
        },
        None => (round_nearest(&gu, u, width, bits)?, round_nearest(&dt, inter, width, bits)?, false),
    };
    Ok((ExpertRows { bits, gate_up: a.pack()?, down_t: b.pack()? }, by_gptq))
}

/// A matrix of codebook rows decoded to f32 (each weight scale * entry, exact).
pub fn decode(p: &CbRows) -> Result<Vec<f32>> {
    let mut out = vec![0f32; p.rows * p.cols];
    if !decode_codebook_rows(p, &mut out) {
        return err("codebook rows that do not decode");
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn unit(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32
        }
        fn normal(&mut self) -> f32 {
            let (a, b) = (self.unit().max(1e-7), self.unit());
            (-2.0 * a.ln()).sqrt() * (std::f32::consts::TAU * b).cos()
        }
        fn bf(&mut self, s: f32) -> f32 {
            f32_of(bf16(self.normal() * s))
        }
    }

    #[test]
    fn the_rotations_block_by_width() {
        let want = [
            (3584, Some(512)),
            (7168, Some(1024)),
            (4096, Some(1024)),
            (2560, Some(512)),
            (2688, Some(128)),
            (1024, Some(1024)),
            (64, None),
            (96, None),
            (0, None),
        ];
        for (w, b) in want {
            assert_eq!(block_of(w), b, "{w}");
        }
    }

    #[test]
    fn signs_are_drawn_once_a_layer() {
        let (a, b) = (signs(3, 4096), signs(4, 4096));
        assert_eq!(a, signs(3, 4096));
        assert_ne!(a, b);
        assert!(a.iter().all(|&s| s == 1 || s == -1));
        let plus = a.iter().filter(|&&s| s == 1).count();
        assert!((1900..2200).contains(&plus), "{plus}");
        assert_eq!(&signs(3, 128)[..], &a[..128]);
    }

    #[test]
    fn the_rotation_is_the_hadamard_matrix_and_orthonormal() {
        let mut rng = Rng(5);
        // (H_b in natural order: H[i, k] = (-1)^popcount(i & k) / sqrt b, on each run of b)
        let (w, b) = (384, 128);
        let s = signs(1, w);
        let x: Vec<f32> = (0..w).map(|_| rng.normal()).collect();
        let mut y = x.clone();
        rotate(&mut y, &s, b);
        for (i, &got) in y.iter().enumerate() {
            let (run, r) = (i / b, i % b);
            let want: f64 =
                (0..b).map(|k| if (r & k).count_ones() % 2 == 0 { 1.0 } else { -1.0 } * (s[run * b + k] as f64) * x[run * b + k] as f64).sum::<f64>()
                    / (b as f64).sqrt();
            assert!((got as f64 - want).abs() < 1e-4, "{i}: {got} {want}");
        }
        // R^T R = I, and dot products are kept (the experts run in the rotated basis)
        for (w, b) in [(3584, 512), (2688, 128), (4096, 1024)] {
            let s = signs(7, w);
            let x: Vec<f32> = (0..w).map(|_| rng.normal()).collect();
            let v: Vec<f32> = (0..w).map(|_| rng.normal()).collect();
            let (mut rx, mut rv) = (x.clone(), v.clone());
            rotate(&mut rx, &s, b);
            rotate(&mut rv, &s, b);
            let dot = |a: &[f32], c: &[f32]| a.iter().zip(c).map(|(p, q)| *p as f64 * *q as f64).sum::<f64>();
            assert!((dot(&rx, &rv) - dot(&x, &v)).abs() < 1e-3 * w as f64, "{w}");
            let mut back = rx.clone();
            unrotate(&mut back, &s, b);
            assert!(back.iter().zip(&x).all(|(p, q)| (p - q).abs() < 1e-4), "{w}");
        }
    }

    #[test]
    fn lloyd_max_finds_the_levels_of_a_row_that_has_them() {
        let levels = [-0.75f32, -0.25, 0.25, 0.75];
        let mut rng = Rng(9);
        let x: Vec<f32> = (0..1024).map(|_| levels[(rng.next() % 4) as usize]).collect();
        let book = lloyd_max(&x, None, 4);
        assert_eq!(book.iter().map(|&b| f32_of(b)).collect::<Vec<_>>(), levels.to_vec());
        // sorted, bf16, and a weight moves an entry toward the heavy values
        let y: Vec<f32> = (0..512).map(|_| rng.normal() * 0.3).collect();
        let b0 = lloyd_max(&y, None, 8);
        assert!(b0.windows(2).all(|p| f32_of(p[0]) <= f32_of(p[1])));
        let wt: Vec<f32> = y.iter().map(|&v| if v > 0.4 { 100.0 } else { 1.0 }).collect();
        let b1 = lloyd_max(&y, Some(&wt), 8);
        assert_ne!(b0, b1);
    }

    #[test]
    fn rows_rounded_to_nearest_decode_to_their_values_exactly() {
        let mut rng = Rng(3);
        for (bits, cols) in [(2u32, 128usize), (3, 256), (3, 3584)] {
            let rows = 6;
            let x: Vec<f32> = (0..rows * cols).map(|_| rng.bf(0.02)).collect();
            let c = round_nearest(&x, rows, cols, bits).unwrap();
            let p = c.pack().unwrap();
            assert_eq!(decode(&p).unwrap(), c.q, "{bits} {cols}");
            assert_eq!(glyd_gpu::codebook::unpack_codebook_codes(&p), c.codes);
            // the error is a rounding's: each value within half the widest gap of its row's codebook times its run's scale
            let rel = x.iter().zip(&c.q).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>() / x.iter().map(|a| (*a as f64).powi(2)).sum::<f64>();
            assert!(rel < if bits == 3 { 0.05 } else { 0.15 }, "{bits}: {rel}");
            // and a run of zeros is coded (its scale the floor, not a division by zero)
        }
        let mut z = vec![0f32; 64];
        z[40] = 0.5;
        let c = round_nearest(&z, 1, 64, 2).unwrap();
        assert!(c.q[..32].iter().all(|&v| v.abs() < 1e-20) && c.pack().is_ok());
        assert!(round_nearest(&z, 1, 48, 2).is_err() && round_nearest(&z, 1, 64, 4).is_err());
    }

    /// GPTQ over neurons by least squares, float64, one neuron at a time (nemotron_hbr.py's brute_gptq): in act-order the next neuron's current
    /// row is rounded (bf16 absmax scale per run, nearest codebook entry) and the neurons after it are re-fitted on the calibration activations
    /// `a` [t, n] with GPTQ's ridge.
    fn brute_neurons(w: &[f32], n: usize, cols: usize, a: &[f64], t: usize, books: &[Vec<f32>]) -> Vec<f64> {
        let mut h = vec![0f64; n * n];
        for r in 0..t {
            for i in 0..n {
                for j in 0..n {
                    h[i * n + j] += a[r * n + i] * a[r * n + j] / t as f64;
                }
            }
        }
        let lam = DAMP * (0..n).map(|i| h[i * n + i]).sum::<f64>() / n as f64;
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&p, &q| h[q * n + q].total_cmp(&h[p * n + p]));
        let mut wv: Vec<f64> = w.iter().map(|&v| v as f64).collect();
        let mut q = vec![0f64; n * cols];
        for (pos, &i) in order.iter().enumerate() {
            let row: Vec<f32> = wv[i * cols..(i + 1) * cols].iter().map(|&v| v as f32).collect();
            let mut codes = vec![0u8; RUN];
            let mut out = vec![0f32; RUN];
            for (r, run) in row.chunks_exact(RUN).enumerate() {
                round_run(run, &books[i], &mut codes, &mut out);
                for j in 0..RUN {
                    q[i * cols + r * RUN + j] = out[j] as f64;
                }
            }
            let rest = &order[pos + 1..];
            if rest.is_empty() {
                continue;
            }
            // min over D (rest x cols) of |A_rest D + A_i (q_i - w_i)|^2 + t lam |D|^2: (A_r^T A_r + t lam I) D = -A_r^T A_i (q_i - w_i)
            let m = rest.len();
            let mut lhs = vec![0f64; m * m];
            let mut ai = vec![0f64; m];
            for (x, &p) in rest.iter().enumerate() {
                for (y, &r2) in rest.iter().enumerate() {
                    lhs[x * m + y] = (0..t).map(|r| a[r * n + p] * a[r * n + r2]).sum::<f64>() + if x == y { t as f64 * lam } else { 0.0 };
                }
                ai[x] = (0..t).map(|r| a[r * n + p] * a[r * n + i]).sum::<f64>();
            }
            let l = chol_lower(&lhs, m).unwrap();
            let inv = inverse_from_chol(&l, m);
            for j in 0..cols {
                let d = q[i * cols + j] - wv[i * cols + j];
                for (x, &p) in rest.iter().enumerate() {
                    let step: f64 = (0..m).map(|y| inv[x * m + y] * ai[y]).sum::<f64>() * -d;
                    wv[p * cols + j] += step;
                }
            }
        }
        q
    }

    #[test]
    fn gptq_over_neurons_is_the_least_squares_fit_and_beats_nearest() {
        let mut rng = Rng(11);
        let (n, cols, t) = (12usize, 64usize, 300usize);
        for bits in [2u32, 3] {
            let w: Vec<f32> = (0..n * cols).map(|_| rng.bf(0.05)).collect();
            // (activations of relu^2 neurons, one never active: a dead row)
            let mut a = vec![0f64; t * n];
            for r in 0..t {
                let common = rng.normal() as f64 * 0.7;
                for i in 0..n {
                    let v = ((rng.normal() as f64 + common).max(0.0)).powi(2) * if i % 4 == 0 { 4.0 } else { 1.0 };
                    a[r * n + i] = if i == 5 { 0.0 } else { v };
                }
            }
            let mut h = vec![0f64; n * n];
            for r in 0..t {
                for i in 0..n {
                    for j in 0..n {
                        h[i * n + j] += a[r * n + i] * a[r * n + j] / t as f64;
                    }
                }
            }
            let got = gptq_neurons(&w, n, cols, &h, bits, DAMP).unwrap().unwrap();
            let books: Vec<Vec<f32>> = (0..n).map(|i| got.books[i << bits..(i + 1) << bits].iter().map(|&b| f32_of(b)).collect()).collect();
            let want = brute_neurons(&w, n, cols, &a, t, &books);
            let top = want.iter().fold(0f64, |m, v| m.max(v.abs()));
            let diff = got.q.iter().zip(&want).fold(0f64, |m, (g, w)| m.max((*g as f64 - w).abs()));
            assert!(diff < 1e-4 * top, "{bits}: {diff} of {top}");
            assert_eq!(decode(&got.pack().unwrap()).unwrap(), got.q);
            // what it is for: under H its error is below nearest rounding's
            let near = round_nearest(&w, n, cols, bits).unwrap();
            let herr = |q: &[f32]| -> f64 {
                let mut s = 0.0;
                for j in 0..cols {
                    for p in 0..n {
                        for r in 0..n {
                            s += (q[p * cols + j] - w[p * cols + j]) as f64 * h[p * n + r] * (q[r * cols + j] - w[r * cols + j]) as f64;
                        }
                    }
                }
                s
            };
            assert!(herr(&got.q) < herr(&near.q), "{bits}: {} {}", herr(&got.q), herr(&near.q));
        }
        // a Hessian that is no use is said so
        let w = vec![0.01f32; 4 * 32];
        let neg: Vec<f64> = (0..16).map(|i| if i % 5 == 0 { -1.0 } else { 0.0 }).collect();
        assert!(gptq_neurons(&w, 4, 32, &neg, 2, DAMP).unwrap().is_none());
    }

    #[test]
    fn gptq_over_columns_beats_nearest_under_its_hessian() {
        let mut rng = Rng(17);
        let (rows, cols, t) = (16usize, 128usize, 1500usize);
        let w: Vec<f32> = (0..rows * cols).map(|_| rng.bf(0.05)).collect();
        let mut h = vec![0f64; cols * cols];
        let mut x = vec![0f64; cols];
        for _ in 0..t {
            for (j, v) in x.iter_mut().enumerate() {
                *v = rng.normal() as f64 * if j % 17 == 0 { 6.0 } else { 1.0 };
            }
            for i in 0..cols {
                for j in 0..cols {
                    h[i * cols + j] += x[i] * x[j] / t as f64;
                }
            }
        }
        for bits in [2u32, 3] {
            let got = gptq_columns(&w, rows, cols, &h, bits, DAMP).unwrap().unwrap();
            assert_eq!(decode(&got.pack().unwrap()).unwrap(), got.q);
            let near = round_nearest(&w, rows, cols, bits).unwrap();
            let herr = |q: &[f32]| -> f64 {
                let mut s = 0.0;
                for r in 0..rows {
                    for i in 0..cols {
                        for j in 0..cols {
                            s += (q[r * cols + i] - w[r * cols + i]) as f64 * h[i * cols + j] * (q[r * cols + j] - w[r * cols + j]) as f64;
                        }
                    }
                }
                s
            };
            assert!(herr(&got.q) < herr(&near.q), "{bits}: {} {}", herr(&got.q), herr(&near.q));
        }
    }

    #[test]
    fn the_recipe_puts_the_most_used_at_three_bits() {
        let counts = [5u64, 0, 9, 9, 1, 7, 2, 3, 4, 8];
        let bits = bits_by_use(&counts, HB40_PERCENT);
        assert_eq!(bits, vec![2, 2, 3, 3, 2, 3, 2, 2, 2, 3]);
        assert_eq!(bits_by_use(&[1; 896], 40).iter().filter(|&&b| b == 3).count(), 359);
        assert_eq!(bits_by_use(&[0; 512], 40).iter().filter(|&&b| b == 3).count(), 205);
        let u = Usage {
            model: "m".into(),
            tokens: 32768,
            layers: vec![LayerUse { block: 1, counts: vec![3, 0, 2] }, LayerUse { block: 4, counts: vec![1, 1, 1] }],
        };
        assert_eq!(Usage::parse(&u.to_json()).unwrap(), u);
        assert_eq!(u.layer(4).unwrap().counts, vec![1, 1, 1]);
        assert!(Usage::parse("{\"format\":\"other\",\"layers\":[]}").is_err());
    }

    #[test]
    fn an_expert_is_coded_in_its_layers_basis() {
        let mut rng = Rng(21);
        let (width, inter) = (256usize, 64usize);
        let rot = block_of(width).unwrap();
        let s = signs(2, width);
        let gate_up: Vec<f32> = (0..2 * inter * width).map(|_| rng.bf(0.03)).collect();
        let down: Vec<f32> = (0..width * inter).map(|_| rng.bf(0.03)).collect();
        let (e, gptq) = code_expert(&gate_up, &down, width, inter, &s, rot, 3, None).unwrap();
        assert!(!gptq);
        assert_eq!((e.gate_up.rows, e.gate_up.cols, e.down_t.rows, e.down_t.cols), (2 * inter, width, inter, width));
        // the rows decode to the rotated weights, and back to the release's within the rounding
        let mut gu = decode(&e.gate_up).unwrap();
        unrotate(&mut gu, &s, rot);
        let rel =
            |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(p, q)| ((p - q) as f64).powi(2)).sum::<f64>() / b.iter().map(|q| (*q as f64).powi(2)).sum::<f64>();
        assert!(rel(&gu, &gate_up) < 0.05, "{}", rel(&gu, &gate_up));
        let mut dt = decode(&e.down_t).unwrap();
        unrotate(&mut dt, &s, rot);
        let back: Vec<f32> = (0..width * inter).map(|i| dt[(i % inter) * width + i / inter]).collect();
        assert!(rel(&back, &down) < 0.05);
        // the product an expert runs is the release's: <R w, R x> = <w, x>
        let x: Vec<f32> = (0..width).map(|_| rng.normal()).collect();
        let mut xr = x.clone();
        rotate(&mut xr, &s, rot);
        let rows = decode(&e.gate_up).unwrap();
        let got: f64 = rows[..width].iter().zip(&xr).map(|(a, b)| *a as f64 * *b as f64).sum();
        let want: f64 = gu[..width].iter().zip(&x).map(|(a, b)| *a as f64 * *b as f64).sum();
        assert!((got - want).abs() < 1e-3, "{got} {want}");
        assert!(code_expert(&gate_up, &down, width, inter, &s[..128], rot, 3, None).is_err());
    }
}
