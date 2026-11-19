//! GLM-5.3-Flash's routed experts as rows of codebook codes in a rotated basis (Linear GLY-133, GLY-134; the formats measured in research/glm53-flash-data.md section 4).
//!
//! An expert is three matrices: gate and up, `inter` rows of `hidden` values each, and down, `hidden` rows of `inter` values. Every matrix is held in the basis of a rotation R = H diag(s) / sqrt(n) of its rows' axis
//! (H the Walsh-Hadamard matrix in natural order, s a vector of +1 and -1 for the layer: one rotation of 4,096 values for gate and up, one of 2,048 for down), where a row is close to Gaussian, and a row is its codes and a
//! codebook of its own: `entries` bf16 values, ascending, in absolute units (the row's scale is in them; no other scale). So <R w, R z> = <w, z>: the activation is rotated once (z' = R z), gate and up are the
//! products of their rows with z', the activation silu(min(gate, limit)) clamp(up, -limit, limit) is rotated by the down matrix's rotation (a' = R2 a), and the rows of down give the outputs in the plain basis.
//!
//! Formats (the weights' work calls them tern, cb2, cb3, cb4): a row of codes of 1.6, 2, 3 or 4 bits each.
//!   - cb2: 4 entries, 2 bits a code, the codes a little-endian bitstream (code j at stream bits [2 j, 2 j + 2): byte b holds codes 4 b .. 4 b + 3);
//!   - cb3: 8 entries, 3 bits a code (codes 8 g .. 8 g + 7 are the 24-bit little-endian integer of bytes 3 g .. 3 g + 2);
//!   - cb4: 16 entries, 4 bits a code (byte b holds codes 2 b (low nibble) and 2 b + 1);
//!   - tern: 3 entries, five codes a byte as base-3 digits (byte b = sum over i < 5 of code(5 b + i) 3^i; the last byte of a row is padded with zeros).
//!
//! An expert's bytes (a *place*): the codes of gate, of up and of down end to end (rows in order), then the codebooks of gate, up and down (`entries` bf16 a row, rows in order), the whole rounded up to 256 bytes. A group
//! of a layer is the experts of one format, one place after another, in a file `L{layer}.{format}.bin` of a directory with a `manifest.json` (see [`CbManifest`]).

use crate::ffi::{err, Error, Result};
use glyd_gpu::json::{self, Value};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// The format of a group of experts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CbFmt {
    Tern,
    Cb2,
    Cb3,
    Cb4,
}

impl CbFmt {
    pub const ALL: [CbFmt; 4] = [CbFmt::Tern, CbFmt::Cb2, CbFmt::Cb3, CbFmt::Cb4];

    pub fn name(self) -> &'static str {
        match self {
            CbFmt::Tern => "t",
            CbFmt::Cb2 => "a",
            CbFmt::Cb3 => "b",
            CbFmt::Cb4 => "c",
        }
    }

    pub fn parse(s: &str) -> Option<CbFmt> {
        CbFmt::ALL.into_iter().find(|f| f.name() == s)
    }

    /// The codebook's entries a row has.
    pub fn entries(self) -> usize {
        match self {
            CbFmt::Tern => 3,
            CbFmt::Cb2 => 4,
            CbFmt::Cb3 => 8,
            CbFmt::Cb4 => 16,
        }
    }

    /// The bytes of the codes of a row of `cols` values.
    pub fn code_bytes(self, cols: usize) -> usize {
        match self {
            CbFmt::Tern => cols.div_ceil(5),
            CbFmt::Cb2 => cols / 4,
            CbFmt::Cb3 => cols * 3 / 8,
            CbFmt::Cb4 => cols / 2,
        }
    }

    /// The bytes of a row's codebook.
    pub fn book_bytes(self) -> usize {
        self.entries() * 2
    }
}

/// An expert's shape and format: where each part of its place is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CbPlace {
    pub fmt: CbFmt,
    pub inter: usize,
    pub hidden: usize,
}

impl CbPlace {
    pub fn new(fmt: CbFmt, inter: usize, hidden: usize) -> Result<CbPlace> {
        if inter == 0 || hidden == 0 || !inter.is_multiple_of(16) || !hidden.is_multiple_of(16) || !inter.is_power_of_two() || !hidden.is_power_of_two() {
            return err(format!("experts of {inter} neurons over {hidden} values: this format takes sizes that are powers of two, at least 16"));
        }
        Ok(CbPlace { fmt, inter, hidden })
    }

    /// The bytes of the codes of gate (and of up).
    pub fn gate_codes(&self) -> usize {
        self.inter * self.fmt.code_bytes(self.hidden)
    }

    /// The bytes of the codes of down.
    pub fn down_codes(&self) -> usize {
        self.hidden * self.fmt.code_bytes(self.inter)
    }

    /// Where down's codes start.
    pub fn down_at(&self) -> usize {
        2 * self.gate_codes()
    }

    /// Where the codebooks start (the codes' bytes).
    pub fn books_at(&self) -> usize {
        2 * self.gate_codes() + self.down_codes()
    }

    /// Where down's codebooks start, in the place.
    pub fn down_books_at(&self) -> usize {
        self.books_at() + 2 * self.inter * self.fmt.book_bytes()
    }

    /// The bytes of the codebooks of all three matrices.
    pub fn books(&self) -> usize {
        (2 * self.inter + self.hidden) * self.fmt.book_bytes()
    }

    /// The bytes an expert takes in a group's file and in a place of the cache (rounded up to 256).
    pub fn bytes(&self) -> usize {
        (self.books_at() + self.books()).div_ceil(256) * 256
    }
}

// ---- the rotation -------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// splitmix64 (the signs are made from a seed, so that the directory needs to hold no vector).
fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The signs (+1 or -1) of a rotation of `n` values for `seed`.
pub fn signs_of(n: usize, seed: u64) -> Vec<i8> {
    let mut st = seed.wrapping_mul(0xD1B5_4A32_D192_ED03) ^ 0x2545_F491_4F6C_DD1D;
    let mut out = Vec::with_capacity(n);
    let mut bits = 0u64;
    for i in 0..n {
        if i % 64 == 0 {
            bits = splitmix(&mut st);
        }
        out.push(if (bits >> (i % 64)) & 1 == 1 { 1 } else { -1 });
    }
    out
}

/// The seed of a layer's rotation of gate and up (`down`: false) or of down.
pub fn seed_of(layer: usize, down: bool) -> u64 {
    layer as u64 * 8 + if down { 3 } else { 2 }
}

/// R = H diag(s) / sqrt(n) over vectors of `n` values.
pub struct Rot {
    pub n: usize,
    pub signs: Vec<f32>,
}

impl Rot {
    pub fn new(n: usize, seed: u64) -> Rot {
        Rot { n, signs: signs_of(n, seed).into_iter().map(f32::from).collect() }
    }

    /// x := R x.
    pub fn apply(&self, x: &mut [f32]) {
        debug_assert_eq!(x.len(), self.n);
        for (v, s) in x.iter_mut().zip(&self.signs) {
            *v *= *s;
        }
        wht(x);
        let c = (1.0 / (self.n as f64).sqrt()) as f32;
        x.iter_mut().for_each(|v| *v *= c);
    }

    /// x := R^T x (diag(s) H x / sqrt(n)).
    pub fn apply_t(&self, x: &mut [f32]) {
        debug_assert_eq!(x.len(), self.n);
        wht(x);
        let c = (1.0 / (self.n as f64).sqrt()) as f32;
        for (v, s) in x.iter_mut().zip(&self.signs) {
            *v *= *s * c;
        }
    }
}

/// The Walsh-Hadamard transform in natural order, unnormalized, in place.
pub fn wht(x: &mut [f32]) {
    let n = x.len();
    let mut h = 1;
    while h < n {
        for blk in x.chunks_exact_mut(2 * h) {
            let (a, b) = blk.split_at_mut(h);
            for (p, q) in a.iter_mut().zip(b.iter_mut()) {
                let (u, v) = (*p, *q);
                *p = u + v;
                *q = u - v;
            }
        }
        h *= 2;
    }
}

// ---- bf16 ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

pub fn f32_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) as u16
}

pub fn bf16_f32(w: u16) -> f32 {
    f32::from_bits((w as u32) << 16)
}

// ---- the codebooks ------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// The Lloyd-Max levels of a unit Gaussian for `k` levels (the start of every row's codebook).
fn gauss_levels(k: usize) -> &'static [f32] {
    static LEVELS: OnceLock<Vec<(usize, Vec<f32>)>> = OnceLock::new();
    let all = LEVELS.get_or_init(|| {
        [3usize, 4, 8, 16]
            .into_iter()
            .map(|k| {
                let pts: Vec<f64> = (0..40001).map(|i| -8.0 + 16.0 * i as f64 / 40000.0).collect();
                let w: Vec<f64> = pts.iter().map(|x| (-x * x / 2.0).exp()).collect();
                let mut c: Vec<f64> = (0..k).map(|i| -2.0 + 4.0 * i as f64 / (k - 1) as f64).collect();
                for _ in 0..400 {
                    let (mut num, mut den) = (vec![0f64; k], vec![0f64; k]);
                    for (x, w) in pts.iter().zip(&w) {
                        let cell = (0..k - 1).take_while(|&j| (c[j] + c[j + 1]) / 2.0 < *x).count();
                        num[cell] += x * w;
                        den[cell] += w;
                    }
                    for j in 0..k {
                        if den[j] > 0.0 {
                            c[j] = num[j] / den[j];
                        }
                    }
                }
                (k, c.into_iter().map(|v| v as f32).collect())
            })
            .collect()
    });
    &all.iter().find(|(kk, _)| *kk == k).expect("levels for 3, 4, 8 and 16 entries").1
}

/// The codebook of a row `x` (already in the rotated basis) for `k` entries: Lloyd-Max steps (three) from the Gaussian levels on a subsample of the row, in the row's units, as bf16, ascending.
fn fit_row(x: &[f32], k: usize, book: &mut [u16]) {
    let cols = x.len();
    let sig = ((x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / cols as f64).sqrt() as f32).max(1e-30);
    let stride = (cols / if k >= 8 { 2048 } else { 1024 }).max(1);
    let xs: Vec<f32> = x.iter().step_by(stride).map(|v| v / sig).collect();
    let mut c: Vec<f32> = gauss_levels(k).to_vec();
    let (mut num, mut den) = (vec![0f32; k], vec![0u32; k]);
    for _ in 0..3 {
        num.fill(0.0);
        den.fill(0);
        let mids: Vec<f32> = (0..k - 1).map(|j| (c[j] + c[j + 1]) * 0.5).collect();
        for &v in &xs {
            let cell = mids.iter().filter(|&&m| v > m).count();
            num[cell] += v;
            den[cell] += 1;
        }
        for j in 0..k {
            if den[j] > 0 {
                c[j] = num[j] / den[j] as f32;
            }
        }
        c.sort_by(|a, b| a.total_cmp(b));
    }
    for (b, v) in book.iter_mut().zip(&c) {
        *b = f32_bf16(v * sig);
    }
}

/// The code of each value of the row `x` (the nearest entry of `book`, by the midpoints between entries; a value on a midpoint takes the lower), into `codes`.
fn assign_row(x: &[f32], book: &[u16], codes: &mut [u8]) {
    let k = book.len();
    let b: Vec<f32> = book.iter().map(|&w| bf16_f32(w)).collect();
    let mids: Vec<f32> = (0..k - 1).map(|j| (b[j] + b[j + 1]) * 0.5).collect();
    for (c, &v) in codes.iter_mut().zip(x) {
        // (a binary search over at most 15 midpoints: the number below v)
        let (mut lo, mut hi) = (0usize, k - 1);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if mids[mid] < v {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        *c = lo as u8;
    }
}

/// A row's codes (one byte each) as the format's bytes.
fn pack_codes(fmt: CbFmt, codes: &[u8], out: &mut [u8]) {
    match fmt {
        CbFmt::Cb2 => {
            for (o, c) in out.iter_mut().zip(codes.chunks_exact(4)) {
                *o = c[0] | c[1] << 2 | c[2] << 4 | c[3] << 6;
            }
        }
        CbFmt::Cb3 => {
            for (o, c) in out.chunks_exact_mut(3).zip(codes.chunks_exact(8)) {
                let v = c.iter().enumerate().fold(0u32, |a, (j, &x)| a | (x as u32) << (3 * j));
                o.copy_from_slice(&v.to_le_bytes()[..3]);
            }
        }
        CbFmt::Cb4 => {
            for (o, c) in out.iter_mut().zip(codes.chunks_exact(2)) {
                *o = c[0] | c[1] << 4;
            }
        }
        CbFmt::Tern => {
            for (o, c) in out.iter_mut().zip(codes.chunks(5)) {
                *o = c.iter().rev().fold(0u8, |a, &x| a * 3 + x);
            }
        }
    }
}

/// The codes (one byte each, `cols` of them) of a row of the format's bytes.
pub fn unpack_codes(fmt: CbFmt, bytes: &[u8], cols: usize, out: &mut [u8]) {
    match fmt {
        CbFmt::Cb2 => {
            for (c, &b) in out.chunks_exact_mut(4).zip(bytes) {
                c.iter_mut().enumerate().for_each(|(j, o)| *o = b >> (2 * j) & 3);
            }
        }
        CbFmt::Cb3 => {
            for (c, b) in out.chunks_exact_mut(8).zip(bytes.chunks_exact(3)) {
                let v = u32::from_le_bytes([b[0], b[1], b[2], 0]);
                c.iter_mut().enumerate().for_each(|(j, o)| *o = (v >> (3 * j) & 7) as u8);
            }
        }
        CbFmt::Cb4 => {
            for (c, &b) in out.chunks_exact_mut(2).zip(bytes) {
                c[0] = b & 15;
                c[1] = b >> 4;
            }
        }
        CbFmt::Tern => {
            for (i, &b) in bytes.iter().enumerate() {
                let mut v = b;
                for j in 0..5 {
                    if 5 * i + j < cols {
                        out[5 * i + j] = v % 3;
                    }
                    v /= 3;
                }
            }
        }
    }
}

/// A matrix of `rows` rows of `cols` values, already in the rotated basis, as codes and codebooks: `codes` (`rows` rows of `fmt.code_bytes(cols)`) and `books` (`rows` rows of `entries` bf16, little-endian).
pub fn pack_matrix(fmt: CbFmt, w: &[f32], rows: usize, cols: usize, codes: &mut [u8], books: &mut [u8]) {
    let (cb, k) = (fmt.code_bytes(cols), fmt.entries());
    let mut one = vec![0u8; cols.div_ceil(5) * 5];
    let mut book = vec![0u16; k];
    for r in 0..rows {
        let x = &w[r * cols..(r + 1) * cols];
        fit_row(x, k, &mut book);
        assign_row(x, &book, &mut one[..cols]);
        pack_codes(fmt, &one[..if fmt == CbFmt::Tern { one.len() } else { cols }], &mut codes[r * cb..(r + 1) * cb]);
        for (j, e) in book.iter().enumerate() {
            books[(r * k + j) * 2..][..2].copy_from_slice(&e.to_le_bytes());
        }
        one[cols..].fill(0);
    }
}

/// A matrix of codes and codebooks as its values (the rotated basis).
pub fn dequant_matrix(fmt: CbFmt, rows: usize, cols: usize, codes: &[u8], books: &[u8]) -> Vec<f32> {
    let (cb, k) = (fmt.code_bytes(cols), fmt.entries());
    let mut out = vec![0f32; rows * cols];
    let mut one = vec![0u8; cols.div_ceil(5) * 5 + 8];
    for r in 0..rows {
        unpack_codes(fmt, &codes[r * cb..(r + 1) * cb], cols, &mut one);
        let book: Vec<f32> = (0..k).map(|j| bf16_f32(u16::from_le_bytes([books[(r * k + j) * 2], books[(r * k + j) * 2 + 1]]))).collect();
        for (o, &c) in out[r * cols..(r + 1) * cols].iter_mut().zip(&one) {
            *o = book[c as usize];
        }
    }
    out
}

/// An expert from its three bf16 matrices (gate and up [inter, hidden], down [hidden, inter]) as its place's bytes: each row rotated (gate and up by `r1`, down by `r2`), then packed.
pub fn pack_expert(place: &CbPlace, gate: &[u16], up: &[u16], down: &[u16], r1: &Rot, r2: &Rot, out: &mut [u8]) {
    let (inter, hidden, fmt) = (place.inter, place.hidden, place.fmt);
    debug_assert_eq!(out.len(), place.bytes());
    out.fill(0);
    let (codes, books) = out.split_at_mut(place.books_at());
    let (gc, rest) = codes.split_at_mut(place.gate_codes());
    let (uc, dc) = rest.split_at_mut(place.gate_codes());
    let bb = fmt.book_bytes();
    let (gb, rest) = books.split_at_mut(inter * bb);
    let (ub, db) = rest.split_at_mut(inter * bb);
    let db = &mut db[..hidden * bb];
    let rotated = |m: &[u16], rows: usize, cols: usize, r: &Rot| -> Vec<f32> {
        let mut v: Vec<f32> = m.iter().map(|&w| bf16_f32(w)).collect();
        v.chunks_exact_mut(cols).take(rows).for_each(|row| r.apply(row));
        v
    };
    pack_matrix(fmt, &rotated(gate, inter, hidden, r1), inter, hidden, gc, gb);
    pack_matrix(fmt, &rotated(up, inter, hidden, r1), inter, hidden, uc, ub);
    pack_matrix(fmt, &rotated(down, hidden, inter, r2), hidden, inter, dc, db);
}

/// An expert's place as its three matrices in the plain basis (rows rotated back): the reference for what the kernels compute.
pub fn unpack_expert(place: &CbPlace, bytes: &[u8], r1: &Rot, r2: &Rot) -> [Vec<f32>; 3] {
    let (inter, hidden, fmt) = (place.inter, place.hidden, place.fmt);
    let (cb, bb) = (fmt.code_bytes(hidden), fmt.book_bytes());
    let g = place.gate_codes();
    let books = &bytes[place.books_at()..];
    let mut gate = dequant_matrix(fmt, inter, hidden, &bytes[..g], &books[..inter * bb]);
    let mut up = dequant_matrix(fmt, inter, hidden, &bytes[g..2 * g], &books[inter * bb..2 * inter * bb]);
    let mut down = dequant_matrix(fmt, hidden, inter, &bytes[place.down_at()..place.books_at()], &books[2 * inter * bb..(2 * inter + hidden) * bb]);
    let _ = cb;
    gate.chunks_exact_mut(hidden).for_each(|r| r1.apply_t(r));
    up.chunks_exact_mut(hidden).for_each(|r| r1.apply_t(r));
    down.chunks_exact_mut(inter).for_each(|r| r2.apply_t(r));
    [gate, up, down]
}

// ---- the plan and the directory ------------------------------------------------------------------------------------------------------------------------------------------------------

/// What a layer's experts are held as: for each expert of the model, its format, or None where it is pruned (the router never picks it).
pub type LayerPlan = Vec<Option<CbFmt>>;

/// The plan of research/glm-data (`plan_*.json`): `moe_layers` and `table` (a list of format names, or "pruned", for every expert of every such layer). Returns (layer, plan) pairs.
pub fn read_plan(path: &Path) -> Result<Vec<(usize, LayerPlan)>> {
    let text = std::fs::read_to_string(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    let v = json::parse(&text).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    let layers = v.get("moe_layers").and_then(Value::as_u64s).ok_or_else(|| Error(format!("{}: no moe_layers", path.display())))?;
    let table = v.get("table").and_then(Value::as_array).ok_or_else(|| Error(format!("{}: no table", path.display())))?;
    if table.len() != layers.len() {
        return err(format!("{}: {} layers and {} rows of the table", path.display(), layers.len(), table.len()));
    }
    let mut out = Vec::new();
    for (l, row) in layers.iter().zip(table) {
        let row = row.as_array().ok_or_else(|| Error(format!("{}: layer {l}: a row that is no list", path.display())))?;
        let mut plan = Vec::new();
        for x in row {
            let s = x.as_str().ok_or_else(|| Error(format!("{}: layer {l}: a format that is no string", path.display())))?;
            plan.push(if s == "pruned" { None } else { Some(CbFmt::parse(s).ok_or_else(|| Error(format!("{}: layer {l}: the format {s}", path.display())))?) });
        }
        out.push((*l as usize, plan));
    }
    Ok(out)
}

/// One format's experts of a layer: their numbers in the model, ascending (their order in the group's file).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CbGroup {
    pub fmt: CbFmt,
    pub ids: Vec<u16>,
}

/// The groups of a layer by format (those that have experts), in the order of [`CbFmt::ALL`].
pub fn groups_of(plan: &[Option<CbFmt>]) -> Vec<CbGroup> {
    CbFmt::ALL.into_iter().filter_map(|f| {
        let ids: Vec<u16> = plan.iter().enumerate().filter(|(_, p)| **p == Some(f)).map(|(e, _)| e as u16).collect();
        (!ids.is_empty()).then_some(CbGroup { fmt: f, ids })
    }).collect()
}

/// A directory's contents: the shape of the experts, the number of experts of the model, and for each layer its groups.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CbManifest {
    pub hidden: usize,
    pub inter: usize,
    pub experts: usize,
    pub layers: Vec<(usize, Vec<CbGroup>)>,
}

pub const MANIFEST: &str = "manifest.json";

impl CbManifest {
    pub fn to_json(&self) -> String {
        let mut s = format!("{{\"kind\":\"glyd-g/1\",\"hidden\":{},\"inter\":{},\"experts\":{},\"layers\":[\n", self.hidden, self.inter, self.experts);
        for (i, (l, groups)) in self.layers.iter().enumerate() {
            s.push_str(&format!("{{\"layer\":{l},\"groups\":["));
            for (j, g) in groups.iter().enumerate() {
                let ids: Vec<String> = g.ids.iter().map(|x| x.to_string()).collect();
                s.push_str(&format!("{{\"fmt\":\"{}\",\"ids\":[{}]}}{}", g.fmt.name(), ids.join(","), if j + 1 < groups.len() { "," } else { "" }));
            }
            s.push_str(&format!("]}}{}\n", if i + 1 < self.layers.len() { "," } else { "" }));
        }
        s.push_str("]}\n");
        s
    }

    pub fn from_json(text: &str) -> Result<CbManifest> {
        let v = json::parse(text).map_err(|e| Error(format!("{MANIFEST}: {e}")))?;
        let num = |k: &str| v.get(k).and_then(Value::as_u64).map(|x| x as usize).ok_or_else(|| Error(format!("{MANIFEST}: no {k}")));
        if v.get("kind").and_then(Value::as_str) != Some("glyd-g/1") {
            return err(format!("{MANIFEST}: not a directory of this format"));
        }
        let mut layers = Vec::new();
        for l in v.get("layers").and_then(Value::as_array).ok_or_else(|| Error(format!("{MANIFEST}: no layers")))? {
            let layer = l.get("layer").and_then(Value::as_u64).ok_or_else(|| Error(format!("{MANIFEST}: a layer with no number")))? as usize;
            let mut groups = Vec::new();
            for g in l.get("groups").and_then(Value::as_array).ok_or_else(|| Error(format!("{MANIFEST}: layer {layer}: no groups")))? {
                let name = g.get("fmt").and_then(Value::as_str).unwrap_or("");
                let fmt = CbFmt::parse(name).ok_or_else(|| Error(format!("{MANIFEST}: layer {layer}: the format {name}")))?;
                let ids = g.get("ids").and_then(Value::as_u64s).ok_or_else(|| Error(format!("{MANIFEST}: layer {layer}: a group with no ids")))?;
                groups.push(CbGroup { fmt, ids: ids.into_iter().map(|x| x as u16).collect() });
            }
            layers.push((layer, groups));
        }
        Ok(CbManifest { hidden: num("hidden")?, inter: num("inter")?, experts: num("experts")?, layers })
    }

    pub fn read(dir: &Path) -> Result<CbManifest> {
        let p = dir.join(MANIFEST);
        CbManifest::from_json(&std::fs::read_to_string(&p).map_err(|e| Error(format!("{}: {e}", p.display())))?)
    }

    pub fn layer(&self, l: usize) -> Option<&[CbGroup]> {
        self.layers.iter().find(|(x, _)| *x == l).map(|(_, g)| g.as_slice())
    }

    /// The bytes of all the experts of the directory.
    pub fn bytes(&self) -> Result<u64> {
        let mut t = 0u64;
        for (_, groups) in &self.layers {
            for g in groups {
                t += g.ids.len() as u64 * CbPlace::new(g.fmt, self.inter, self.hidden)?.bytes() as u64;
            }
        }
        Ok(t)
    }
}

/// The file of a layer's group.
pub fn group_file(dir: &Path, layer: usize, fmt: CbFmt) -> PathBuf {
    dir.join(format!("L{layer:02}.{}.bin", fmt.name()))
}

/// The routed experts of `plan`'s layers (those of `only` where given) of the model at `snap` (the Hub's FP8 or bf16 tensors), each held as the plan says, into the files of `out` (a manifest beside them; the layers of an
/// earlier run are kept), `threads` experts at a time. A layer's rotations are made from its number (`seed_of`).
pub fn pack_dir(snap: &Path, plan: &[(usize, LayerPlan)], out: &Path, threads: usize, only: Option<&[usize]>) -> Result<()> {
    use std::io::Write;
    let cfg = crate::glm::ConfigGlm::load(snap)?;
    let ck = glyd_gpu::safetensors::Checkpoint::open(snap).map_err(|e| Error(e.to_string()))?;
    std::fs::create_dir_all(out).map_err(|e| Error(format!("{}: {e}", out.display())))?;
    let (hidden, inter, experts) = (cfg.hidden, cfg.experts.inter, cfg.experts.experts);
    let mut manifest = CbManifest::read(out).ok().filter(|m| (m.hidden, m.inter, m.experts) == (hidden, inter, experts)).unwrap_or(CbManifest { hidden, inter, experts, layers: Vec::new() });
    for (l, lp) in plan.iter().filter(|(l, _)| only.is_none_or(|o| o.contains(l))) {
        if lp.len() != experts {
            return err(format!("layer {l}: a plan of {} experts, where the model has {experts}", lp.len()));
        }
        let t0 = std::time::Instant::now();
        let (r1, r2) = (Rot::new(hidden, seed_of(*l, false)), Rot::new(inter, seed_of(*l, true)));
        let groups = groups_of(lp);
        let prefix = format!("{}layers.{l}.mlp.experts", cfg.prefix);
        let mut bytes = 0u64;
        for g in &groups {
            let place = CbPlace::new(g.fmt, inter, hidden)?;
            let path = group_file(out, *l, g.fmt);
            let tmp = path.with_extension("tmp");
            let mut f = std::io::BufWriter::with_capacity(1 << 22, std::fs::File::create(&tmp).map_err(|e| Error(format!("{}: {e}", tmp.display())))?);
            crate::weights::ordered(
                g.ids.len(),
                threads,
                2 * threads.max(1),
                &|k| {
                    let e = g.ids[k] as usize;
                    let read = |what: &str, n: usize| -> Result<Vec<u16>> {
                        let name = format!("{prefix}.{e}.{what}_proj.weight");
                        let mut w = vec![0u16; n];
                        ck.read_bf16(&name, &mut w).map_err(|x| Error(format!("{name}: {x}")))?;
                        Ok(w)
                    };
                    let (gate, up, down) = (read("gate", inter * hidden)?, read("up", inter * hidden)?, read("down", hidden * inter)?);
                    let mut out = vec![0u8; place.bytes()];
                    pack_expert(&place, &gate, &up, &down, &r1, &r2, &mut out);
                    Ok(out)
                },
                |_, b| f.write_all(&b).map_err(|x| Error(format!("{}: {x}", tmp.display()))),
            )?;
            f.flush().map_err(|x| Error(format!("{}: {x}", tmp.display())))?;
            drop(f);
            std::fs::rename(&tmp, &path).map_err(|x| Error(format!("{}: {x}", path.display())))?;
            bytes += (g.ids.len() * place.bytes()) as u64;
        }
        manifest.layers.retain(|(x, _)| x != l);
        manifest.layers.push((*l, groups));
        manifest.layers.sort_by_key(|(x, _)| *x);
        std::fs::write(out.join(MANIFEST), manifest.to_json()).map_err(|e| Error(format!("{}: {e}", out.display())))?;
        println!("layer {l}: {} experts in {} groups, {:.2} GB, {:.1} s", lp.iter().flatten().count(), manifest.layer(*l).map_or(0, |g| g.len()), bytes as f64 / 1e9, t0.elapsed().as_secs_f64());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            splitmix(&mut self.0)
        }

        fn gauss(&mut self) -> f32 {
            let (a, b) = ((self.next() >> 11) as f64 / (1u64 << 53) as f64, (self.next() >> 11) as f64 / (1u64 << 53) as f64);
            ((-2.0 * (a + 1e-300).ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()) as f32
        }
    }

    #[test]
    fn a_rotation_keeps_products_and_goes_back() {
        let mut rng = Rng(7);
        for n in [16usize, 2048, 4096] {
            let r = Rot::new(n, seed_of(5, n == 2048));
            let a: Vec<f32> = (0..n).map(|_| rng.gauss()).collect();
            let b: Vec<f32> = (0..n).map(|_| rng.gauss()).collect();
            let dot = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(p, q)| *p as f64 * *q as f64).sum::<f64>();
            let (mut ra, mut rb) = (a.clone(), b.clone());
            r.apply(&mut ra);
            r.apply(&mut rb);
            assert!((dot(&a, &b) - dot(&ra, &rb)).abs() < 1e-3 * (n as f64).sqrt(), "n {n}");
            let mut back = ra.clone();
            r.apply_t(&mut back);
            assert!(back.iter().zip(&a).all(|(p, q)| (p - q).abs() < 1e-4), "n {n}");
        }
        // (the signs depend on the seed and are balanced)
        let s = signs_of(4096, 42);
        assert!(s != signs_of(4096, 43));
        assert!((s.iter().map(|&x| x as i32).sum::<i32>()).abs() < 400);
    }

    #[test]
    fn codes_survive_their_bytes() {
        let mut rng = Rng(11);
        for fmt in CbFmt::ALL {
            let cols = 4096;
            let k = fmt.entries() as u64;
            let codes: Vec<u8> = (0..cols + 5).map(|i| if i < cols { (rng.next() % k) as u8 } else { 0 }).collect();
            let mut bytes = vec![0u8; fmt.code_bytes(cols)];
            pack_codes(fmt, &codes[..if fmt == CbFmt::Tern { cols.div_ceil(5) * 5 } else { cols }], &mut bytes);
            let mut back = vec![0u8; cols + 8];
            unpack_codes(fmt, &bytes, cols, &mut back);
            assert_eq!(&back[..cols], &codes[..cols], "{}", fmt.name());
        }
        // (a few bytes by hand)
        let mut b = [0u8; 1];
        pack_codes(CbFmt::Cb2, &[1, 2, 3, 0], &mut b);
        assert_eq!(b[0], 1 | 2 << 2 | 3 << 4);
        pack_codes(CbFmt::Tern, &[1, 2, 0, 1, 2], &mut b);
        assert_eq!(b[0], 1 + 2 * 3 + 27 + 2 * 81);
    }

    /// Rows of a Gaussian matrix are held with the error the Lloyd-Max quantizer of the format has (0.343 at four entries, 0.186 at eight, 0.0955 at sixteen, 0.436 at three), as a share of the rows' norm.
    #[test]
    fn the_rows_are_held_with_the_error_of_their_codebook() {
        let mut rng = Rng(3);
        let (rows, cols) = (64usize, 4096usize);
        let w: Vec<f32> = (0..rows * cols).map(|_| rng.gauss() * 0.02).collect();
        for (fmt, want) in [(CbFmt::Tern, 0.436), (CbFmt::Cb2, 0.343), (CbFmt::Cb3, 0.186), (CbFmt::Cb4, 0.0955)] {
            let mut codes = vec![0u8; rows * fmt.code_bytes(cols)];
            let mut books = vec![0u8; rows * fmt.book_bytes()];
            pack_matrix(fmt, &w, rows, cols, &mut codes, &mut books);
            let d = dequant_matrix(fmt, rows, cols, &codes, &books);
            let err = (w.iter().zip(&d).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>() / w.iter().map(|a| (*a as f64).powi(2)).sum::<f64>()).sqrt();
            assert!((err - want).abs() < 0.02, "{}: error {err:.4}, expected about {want}", fmt.name());
            // (the codebook is ascending)
            for r in 0..rows {
                let b: Vec<f32> = (0..fmt.entries()).map(|j| bf16_f32(u16::from_le_bytes([books[(r * fmt.entries() + j) * 2], books[(r * fmt.entries() + j) * 2 + 1]]))).collect();
                assert!(b.windows(2).all(|p| p[0] <= p[1]), "{}", fmt.name());
            }
        }
    }

    #[test]
    fn an_expert_is_its_matrices_to_the_error_of_its_format() {
        let mut rng = Rng(5);
        let (inter, hidden) = (64usize, 128usize);
        let mat = |rng: &mut Rng, n: usize| -> Vec<u16> { (0..n).map(|_| f32_bf16(rng.gauss() * 0.03)).collect() };
        let (gate, up, down) = (mat(&mut rng, inter * hidden), mat(&mut rng, inter * hidden), mat(&mut rng, hidden * inter));
        let (r1, r2) = (Rot::new(hidden, seed_of(3, false)), Rot::new(inter, seed_of(3, true)));
        for fmt in CbFmt::ALL {
            let place = CbPlace::new(fmt, inter, hidden).unwrap();
            let mut bytes = vec![0u8; place.bytes()];
            pack_expert(&place, &gate, &up, &down, &r1, &r2, &mut bytes);
            let [g, u, d] = unpack_expert(&place, &bytes, &r1, &r2);
            for (m, orig) in [(g, &gate), (u, &up), (d, &down)] {
                let e = (m.iter().zip(orig.iter()).map(|(a, b)| ((a - bf16_f32(*b)) as f64).powi(2)).sum::<f64>() / orig.iter().map(|b| (bf16_f32(*b) as f64).powi(2)).sum::<f64>()).sqrt();
                let bound = match fmt {
                    CbFmt::Tern => 0.5,
                    CbFmt::Cb2 => 0.4,
                    CbFmt::Cb3 => 0.25,
                    CbFmt::Cb4 => 0.15,
                };
                assert!(e < bound, "{}: the error of a matrix is {e:.3}", fmt.name());
            }
        }
    }

    #[test]
    fn the_places_have_the_bytes_the_weights_work_counted() {
        // 6,356,992 / 9,568,256 / 12,845,056 / 5,082,316.8 bytes an expert (research/glm53-flash-data.md section 4.1), here with the rows' padding and the place rounded up to 256
        for (fmt, want) in [(CbFmt::Cb2, 6_356_992usize), (CbFmt::Cb3, 9_568_256), (CbFmt::Cb4, 12_845_056)] {
            let p = CbPlace::new(fmt, 2048, 4096).unwrap();
            assert_eq!(p.books_at() + p.books(), want, "{}", fmt.name());
            assert_eq!(p.bytes() % 256, 0);
        }
        let t = CbPlace::new(CbFmt::Tern, 2048, 4096).unwrap();
        assert_eq!(t.books_at() + t.books(), 2 * (2048 * 820 + 2048 * 6) + 4096 * 410 + 4096 * 6);
    }

    #[test]
    fn the_manifest_reads_back() {
        let plan: LayerPlan = vec![Some(CbFmt::Cb2), None, Some(CbFmt::Tern), Some(CbFmt::Cb2), Some(CbFmt::Cb4)];
        let m = CbManifest { hidden: 4096, inter: 2048, experts: 5, layers: vec![(3, groups_of(&plan)), (4, vec![])] };
        let back = CbManifest::from_json(&m.to_json()).unwrap();
        assert_eq!(back, m);
        assert_eq!(back.layer(3).unwrap().len(), 3);
        assert_eq!(back.layer(3).unwrap()[1], CbGroup { fmt: CbFmt::Cb2, ids: vec![0, 3] });
    }
}
