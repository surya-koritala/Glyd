//! The coefficients of a JPEG coded under contexts from the blocks
//! above and to the left, with the binary arithmetic coder of
//! `reflate::coder`.
//!
//! Per block, in raster order within each component: how many of the
//! 7×7 interior coefficients are nonzero (under the neighbours'
//! counts); the interior in zigzag order while any are left — zero or
//! not, the magnitude as an exponent then mantissa bits, the sign —
//! under the position, what is left and the neighbours' values there;
//! then the first row and the first column, each coefficient under a
//! prediction from pixel continuity with the neighbour across that
//! edge: with dequantized coefficients, the pixel row just outside a
//! block is Σ cᵥ·(−1)ᵛ·F[v] and the one just inside its neighbour is
//! Σ cᵥ·F[v], so the coefficient still unknown in that sum is
//! predicted from the rest; then the DC, from the same relation on
//! both edges.

use super::{Frame, Jpeg, ZIGZAG};
use crate::reflate::coder::{Decoder, Encoder as Coder, Prob};

/// The rate of adaptation shrinks with the number of bits seen, down
/// to 1/`LIMIT`: a context that has seen many bits trusts its count.
const LIMIT: usize = 255;
const RECIP: [u32; LIMIT + 1] = {
    let mut t = [0u32; LIMIT + 1];
    let mut n = 0;
    while n <= LIMIT {
        t[n] = 65536 * 2 / (2 * n as u32 + 3);
        n += 1;
    }
    t
};

/// A probability that adapts at 1/(n + 1.5) after n bits.
#[derive(Clone, Copy)]
pub struct Bit {
    p: u16,
    n: u8,
}

impl Default for Bit {
    fn default() -> Self {
        Bit { p: 32768, n: 0 }
    }
}

impl Bit {
    #[cfg(test)]
    fn cost(&self, bit: u32) -> f64 {
        let p = Prob::p(self) as f64 / 65536.0;
        -(if bit != 0 { p } else { 1.0 - p }).log2()
    }
}

impl Prob for Bit {
    #[inline]
    fn p(&self) -> u32 {
        (self.p as u32).clamp(32, 65536 - 32)
    }

    #[inline]
    fn update(&mut self, bit: u32) {
        let r = RECIP[self.n as usize];
        if bit != 0 {
            self.p += (((65535 - self.p as u32) * r) >> 16) as u16;
        } else {
            self.p -= (((self.p as u32) * r) >> 16) as u16;
        }
        if (self.n as usize) < LIMIT {
            self.n += 1;
        }
    }
}

/// In tests, the bits spent per part of the model (count, zero, exp,
/// mant, sign, edge zero, edge exp, edge mant, edge sign, dc) for
/// seeing where they go.
#[cfg(test)]
pub static COST: [std::sync::atomic::AtomicU64; 13] = [const { std::sync::atomic::AtomicU64::new(0) }; 13];

struct Encoder {
    e: Coder,
    slot: usize,
}

impl Encoder {
    fn new() -> Self {
        Encoder { e: Coder::new(), slot: 0 }
    }
    #[inline]
    fn bit(&mut self, m: &mut Bit, bit: u32) {
        #[cfg(test)]
        COST[self.slot].fetch_add((m.cost(bit) * 1000.0) as u64, std::sync::atomic::Ordering::Relaxed);
        self.e.bit(m, bit)
    }
    #[inline]
    fn tree(&mut self, m: &mut [Bit], bits: u32, v: u32) {
        let mut node = 1usize;
        for i in (0..bits).rev() {
            let b = (v >> i) & 1;
            self.bit(&mut m[node], b);
            node = node * 2 + b as usize;
        }
    }
    #[inline]
    fn slot(&mut self, s: usize) {
        self.slot = s;
    }
    fn finish(self) -> Vec<u8> {
        self.e.finish()
    }
}

const MAG_BUCKETS: usize = 12;
const LEFT_BUCKETS: usize = 12;

/// zigzag index of natural index n
const NAT2ZZ: [usize; 64] = {
    let mut t = [0usize; 64];
    let mut k = 0;
    while k < 64 {
        t[ZIGZAG[k]] = k;
        k += 1;
    }
    t
};

/// The interior 7×7 in zigzag order (natural row and column both ≥ 1).
const INTERIOR: [usize; 49] = {
    let mut t = [0usize; 49];
    let (mut k, mut i) = (0usize, 0usize);
    while k < 64 {
        let n = ZIGZAG[k];
        if n / 8 >= 1 && n % 8 >= 1 {
            t[i] = k;
            i += 1;
        }
        k += 1;
    }
    t
};

#[inline]
fn mag_bucket(m: u32) -> usize {
    if m == 0 {
        0
    } else {
        (32 - m.leading_zeros()).min(MAG_BUCKETS as u32 - 1) as usize
    }
}

#[inline]
fn count_bucket(n: u32) -> usize {
    match n {
        0 => 0,
        1..=2 => 1,
        3..=5 => 2,
        6..=9 => 3,
        10..=15 => 4,
        16..=24 => 5,
        25..=38 => 6,
        _ => 7,
    }
}

#[inline]
fn left_bucket(n: u32) -> usize {
    match n {
        0..=3 => n as usize,
        4..=5 => 4,
        6..=8 => 5,
        9..=12 => 6,
        13..=17 => 7,
        18..=24 => 8,
        25..=34 => 9,
        35..=48 => 10,
        _ => 11,
    }
}

/// A prediction's bucket for a context: its magnitude in 9 steps and
/// its sign, 0 for none.
#[inline]
fn pred_bucket(p: i32) -> usize {
    let m = mag_bucket(p.unsigned_abs());
    if p < 0 { MAG_BUCKETS + m } else { m }
}

/// The contexts of one component kind (luma, chroma).
struct Contexts {
    count: Vec<Bit>,
    zero: Vec<Bit>,
    exp: Vec<Bit>,
    mant: Vec<Bit>,
    sign: Vec<Bit>,
    // the edges: per position (14), prediction bucket (18), remaining
    edge_count: Vec<Bit>,
    edge_zero: Vec<Bit>,
    edge_exp: Vec<Bit>,
    edge_mant: Vec<Bit>,
    edge_sign: Vec<Bit>,
    dc_exp: Vec<Bit>,
    dc_mant: Vec<Bit>,
    dc_sign: Vec<Bit>,
}

const EDGE_CTX: usize = 2 * MAG_BUCKETS;

impl Contexts {
    fn new() -> Self {
        Contexts {
            count: vec![Bit::default(); 128 * 256],
            zero: vec![Bit::default(); 64 * LEFT_BUCKETS * MAG_BUCKETS * MAG_BUCKETS],
            exp: vec![Bit::default(); 64 * MAG_BUCKETS * LEFT_BUCKETS * 12],
            mant: vec![Bit::default(); 64 * 12 * MANT],
            sign: vec![Bit::default(); 64 * 3],
            edge_count: vec![Bit::default(); 2 * 8 * 8 * 8],
            edge_zero: vec![Bit::default(); 14 * EDGE_CTX * 8 * 4],
            edge_exp: vec![Bit::default(); 14 * EDGE_CTX * 8 * 4 * 12],
            edge_mant: vec![Bit::default(); 14 * 12 * MANT],
            edge_sign: vec![Bit::default(); 14 * EDGE_CTX],
            dc_exp: vec![Bit::default(); 32 * 13],
            dc_mant: vec![Bit::default(); 32 * 13 * MANT],
            dc_sign: vec![Bit::default(); 32 * 3],
        }
    }
}

fn put_magnitude(e: &mut Encoder, exp: &mut [Bit], mant: &mut [Bit], m: u32) {
    let n = 32 - m.leading_zeros();
    let s = e.slot;
    e.slot(s + 1);
    for i in 1..n {
        e.bit(&mut exp[i as usize], 1);
    }
    if n < 12 {
        e.bit(&mut exp[n as usize], 0);
    }
    e.slot(s + 2);
    let mut node = 1usize;
    for i in (0..n - 1).rev() {
        let b = (m >> i) & 1;
        let depth = (n - 2 - i) as usize;
        if depth < 3 {
            e.bit(&mut mant[(n as usize % 12) * MANT + node], b);
            node = node * 2 + b as usize;
        } else {
            e.bit(&mut mant[(n as usize % 12) * MANT + 8 + depth], b);
        }
    }
}

/// Mantissa contexts per exponent: a tree of the top three bits, then
/// one per bit below.
const MANT: usize = 20;

fn get_magnitude(d: &mut Decoder, exp: &mut [Bit], mant: &mut [Bit]) -> u32 {
    let mut n = 1u32;
    while n < 12 && d.bit(&mut exp[n as usize]) == 1 {
        n += 1;
    }
    let mut m = 1u32;
    let mut node = 1usize;
    for i in (0..n - 1).rev() {
        let depth = (n - 2 - i) as usize;
        let b = if depth < 3 {
            let b = d.bit(&mut mant[(n as usize % 12) * MANT + node]);
            node = node * 2 + b as usize;
            b
        } else {
            d.bit(&mut mant[(n as usize % 12) * MANT + 8 + depth])
        };
        m = (m << 1) | b;
    }
    m
}

/// A signed value: zero or not, magnitude, sign, under the given slices.
fn put_signed(e: &mut Encoder, zero: &mut Bit, exp: &mut [Bit], mant: &mut [Bit], sign: &mut Bit, v: i32) {
    let s = e.slot;
    e.bit(zero, (v != 0) as u32);
    if v != 0 {
        put_magnitude(e, exp, mant, v.unsigned_abs());
        e.slot(s + 3);
        e.bit(sign, (v < 0) as u32);
        e.slot(s);
    }
}

fn get_signed(d: &mut Decoder, zero: &mut Bit, exp: &mut [Bit], mant: &mut [Bit], sign: &mut Bit) -> Option<i32> {
    if d.bit(zero) == 0 {
        return Some(0);
    }
    let m = get_magnitude(d, exp, mant);
    if m > 32767 {
        return None;
    }
    Some(if d.bit(sign) == 1 { -(m as i32) } else { m as i32 })
}

struct Neighbours<'a> {
    left: Option<&'a [i16; 64]>,
    above: Option<&'a [i16; 64]>,
    corner: Option<&'a [i16; 64]>,
}

impl Neighbours<'_> {
    /// The neighbours' magnitudes at `k`, weighted 13:13:6 for left,
    /// above and the corner, out of 32.
    #[inline]
    fn mag(&self, k: usize) -> u32 {
        let l = self.left.map_or(0, |b| b[k].unsigned_abs() as u32);
        let a = self.above.map_or(0, |b| b[k].unsigned_abs() as u32);
        match (self.left.is_some(), self.above.is_some()) {
            (true, true) => {
                let c = self.corner.map_or(0, |b| b[k].unsigned_abs() as u32);
                (13 * l + 13 * a + 6 * c + 16) / 32
            }
            _ => l + a,
        }
    }

    /// The zero flag's context from the neighbours at `k`: the
    /// weighted magnitude and how far left and above disagree.
    #[inline]
    fn mags(&self, k: usize) -> usize {
        let l = self.left.map_or(0, |b| b[k].unsigned_abs() as u32);
        let a = self.above.map_or(0, |b| b[k].unsigned_abs() as u32);
        mag_bucket(self.mag(k)) * MAG_BUCKETS + mag_bucket(l.abs_diff(a)).min(MAG_BUCKETS - 1)
    }

    #[inline]
    fn sign_ctx(&self, k: usize) -> usize {
        let s = |b: Option<&[i16; 64]>| b.map_or(0, |b| b[k].signum() as i32);
        match s(self.left) + s(self.above) {
            x if x > 0 => 1,
            x if x < 0 => 2,
            _ => 0,
        }
    }

    /// The context for the interior count: the neighbours' average
    /// count and how far apart they are.
    fn count(&self) -> usize {
        let c = |b: &[i16; 64]| INTERIOR.iter().filter(|&&k| b[k] != 0).count();
        match (self.left.map(c), self.above.map(c)) {
            (Some(l), Some(a)) => ((l + a + 1) / 2) * 4 + mag_bucket(l.abs_diff(a) as u32 / 3).min(3),
            (Some(n), None) | (None, Some(n)) => 200 + n,
            (None, None) => 250,
        }
    }
}

/// The 1D IDCT weights at the boundary between two blocks — half a
/// pixel past the far side of the neighbour (cos(vπ) = ±1) and half a
/// pixel before the near side of this block (cos 0 = 1) — times C(v),
/// in 1/8192ths: the two sides are predicted to meet there.
const W_OUT: [i64; 8] = [5793, -8192, 8192, -8192, 8192, -8192, 8192, -8192];
const W_IN: [i64; 8] = [5793, 8192, 8192, 8192, 8192, 8192, 8192, 8192];

/// Σ W_OUT[v]·F[v][u]·Q over column `u` of the block above (or, with
/// `row`, W_OUT[u]·F[v][u]·Q over row `v` of the block to the left):
/// the neighbour's pixel value at its far edge, for that frequency.
fn outer_edge(n: &[i16; 64], q: &[u16; 64], line: usize, row: bool) -> i64 {
    let mut sum = 0i64;
    for i in 0..8 {
        let nat = if row { line * 8 + i } else { i * 8 + line };
        let k = NAT2ZZ[nat];
        sum += W_OUT[i] * n[k] as i64 * q[k] as i64;
    }
    sum
}

/// The same at this block's near edge, over the known coefficients
/// (index `skip` is the one being predicted).
fn inner_edge(b: &[i16; 64], q: &[u16; 64], line: usize, row: bool, skip: usize) -> i64 {
    let mut sum = 0i64;
    for i in 0..8 {
        if i == skip {
            continue;
        }
        let nat = if row { line * 8 + i } else { i * 8 + line };
        let k = NAT2ZZ[nat];
        sum += W_IN[i] * b[k] as i64 * q[k] as i64;
    }
    sum
}

/// The prediction of the coefficient at natural index `nat` from the
/// edge it lies on: the first row from the block above (columns
/// continue), the first column from the block to the left (rows
/// continue), in quantized units.
fn edge_prediction(b: &[i16; 64], nb: &Neighbours, q: &[u16; 64], nat: usize) -> Option<i32> {
    let (v, u) = (nat / 8, nat % 8);
    let (neighbour, line, row, skip) = if v == 0 { (nb.above?, u, false, 0) } else { (nb.left?, v, true, 0) };
    let outer = outer_edge(neighbour, q, line, row);
    let inner = inner_edge(b, q, line, row, skip);
    // The unknown is the index-0 term of the inner sum (v = 0 for a
    // column, u = 0 for a row), with weight c₀.
    let k = NAT2ZZ[nat];
    let quant = q[k] as i64;
    Some(((outer - inner) / (W_IN[0] * quant).max(1)) as i32)
}

/// The 1D IDCT: pixel `x` from the eight frequencies, C(u)·cos in
/// 1/8192ths.
const IDCT8: [[i64; 8]; 8] = [
    [5793, 8035, 7568, 6811, 5793, 4551, 3135, 1598],
    [5793, 6811, 3135, -1598, -5793, -8035, -7568, -4551],
    [5793, 4551, -3135, -8035, -5793, 1598, 7568, 6811],
    [5793, 1598, -7568, -4551, 5793, 6811, -3135, -8035],
    [5793, -1598, -7568, 4551, 5793, -6811, -3135, 8035],
    [5793, -4551, -3135, 8035, -5793, -1598, 7568, -6811],
    [5793, -6811, 3135, 1598, -5793, 8035, -7568, 4551],
    [5793, -8035, 7568, -6811, 5793, -4551, 3135, -1598],
];

/// One side's DC prediction: the difference between the neighbour's
/// boundary pixels and this block's (without its DC) along the edge,
/// as the prediction (its mean, in quantized DC units) and how much
/// the eight pixels disagree with each other (the spread, in 1/8192
/// pixel units).
fn dc_side(b: &[i16; 64], n: &[i16; 64], q: &[u16; 64], row: bool) -> (i64, i64) {
    let mut e = [0i64; 8];
    for j in 0..8 {
        for i in 0..8 {
            let nat = if row { j * 8 + i } else { i * 8 + j };
            let k = NAT2ZZ[nat];
            let qq = q[k] as i64;
            e[j] += W_OUT[i] * n[k] as i64 * qq;
            if nat != 0 {
                e[j] -= W_IN[i] * b[k] as i64 * qq;
            }
        }
    }
    let mut spread = 0i64;
    for x in 0..8 {
        let mut p = 0i64;
        for j in 1..8 {
            p += IDCT8[x][j] * e[j];
        }
        spread += (p / 8192).abs();
    }
    (e[0] / (W_IN[0] * q[0] as i64).max(1), spread)
}

/// The DC predicted from both edges, in quantized units, weighted
/// towards the side whose pixels agree more, and the context: how
/// well the better side agrees.
fn dc_prediction(b: &[i16; 64], nb: &Neighbours, q: &[u16; 64]) -> (i32, usize) {
    let unit = 8192 * q[0] as i64;
    let a = nb.above.map(|n| dc_side(b, n, q, false));
    let l = nb.left.map(|n| dc_side(b, n, q, true));
    match (a, l) {
        (Some((pa, sa)), Some((pl, sl))) => {
            let (wa, wl) = (sl + unit, sa + unit);
            let pred = (pa * wa + pl * wl) / (wa + wl);
            (pred as i32, mag_bucket((pa - pl).unsigned_abs() as u32).min(11))
        }
        (Some((p, s)), None) | (None, Some((p, s))) => (p as i32, 12 + mag_bucket((s / unit) as u32).min(11)),
        (None, None) => (0, 24),
    }
}

/// How many of the 7 coefficients of the first row (`dir` 0) or the
/// first column (1) are nonzero.
fn edge_count(b: &[i16; 64], dir: usize) -> usize {
    EDGES[dir * 7..dir * 7 + 7].iter().filter(|&&nat| b[NAT2ZZ[nat]] != 0).count()
}

fn kind(component: usize) -> usize {
    (component != 0) as usize
}

/// The order the edges are coded in: the first row's 1..7, then the
/// first column's 1..7, as natural indices.
const EDGES: [usize; 14] = [1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32, 40, 48, 56];

/// Every component's blocks coded: the model's stream.
pub fn encode(j: &Jpeg) -> Vec<u8> {
    let mut e = Encoder::new();
    let mut ctx = [Contexts::new(), Contexts::new()];
    for (ci, c) in j.frame.components.iter().enumerate() {
        let m = &mut ctx[kind(ci)];
        let q = j.quant[c.tq as usize].as_ref().unwrap();
        let blocks = &j.blocks[ci];
        for y in 0..c.bh {
            for x in 0..c.bw {
                let block = &blocks[y * c.bw + x];
                let nb = Neighbours {
                    left: (x > 0).then(|| &blocks[y * c.bw + x - 1]),
                    above: (y > 0).then(|| &blocks[(y - 1) * c.bw + x]),
                    corner: (x > 0 && y > 0).then(|| &blocks[(y - 1) * c.bw + x - 1]),
                };
                let nz = INTERIOR.iter().filter(|&&k| block[k] != 0).count() as u32;
                e.slot(0);
                e.tree(&mut m.count[nb.count() * 64..], 6, nz);
                let nzb = count_bucket(nz);
                let mut left = nz;
                for &k in INTERIOR.iter() {
                    if left == 0 {
                        break;
                    }
                    let mb = mag_bucket(nb.mag(k));
                    let z = &mut m.zero[(k * LEFT_BUCKETS + left_bucket(left)) * MAG_BUCKETS * MAG_BUCKETS + nb.mags(k)];
                    let v = block[k];
                    e.slot(1);
                    e.bit(z, (v != 0) as u32);
                    if v == 0 {
                        continue;
                    }
                    left -= 1;
                    let base = ((k * MAG_BUCKETS + mb) * LEFT_BUCKETS + left_bucket(left)) * 12;
                    put_magnitude(&mut e, &mut m.exp[base..base + 12], &mut m.mant[k * 12 * MANT..(k + 1) * 12 * MANT], v.unsigned_abs() as u32);
                    let sc = nb.sign_ctx(k);
                    e.slot(4);
                    e.bit(&mut m.sign[k * 3 + sc], (v < 0) as u32);
                }
                // A partial block for the edge predictions: the
                // interior as coded, the edges as they get coded.
                let mut partial = [0i16; 64];
                for &k in INTERIOR.iter() {
                    partial[k] = block[k];
                }
                for dir in 0..2 {
                    let across = if dir == 0 { nb.above } else { nb.left };
                    let count = edge_count(block, dir);
                    let cc = (dir * 8 + nzb) * 8 + across.map_or(0, |b| edge_count(b, dir));
                    e.slot(0);
                    e.tree(&mut m.edge_count[cc * 8..], 3, count as u32);
                    let mut left = count;
                    for (i, &nat) in EDGES.iter().enumerate().skip(dir * 7).take(7) {
                        if left == 0 {
                            break;
                        }
                        let k = NAT2ZZ[nat];
                        let pred = edge_prediction(&partial, &nb, q, nat);
                        let pb = pred.map_or(0, pred_bucket);
                        let v = block[k] as i32;
                        let c = ((i * EDGE_CTX + pb) * 8 + left) * 4 + mag_bucket(nb.mag(k)).min(3);
                        let zero = &mut m.edge_zero[c];
                        let exp = &mut m.edge_exp[c * 12..(c + 1) * 12];
                        let mant = &mut m.edge_mant[i * 12 * MANT..(i + 1) * 12 * MANT];
                        let sign = &mut m.edge_sign[i * EDGE_CTX + pb];
                        e.slot(5);
                        put_signed(&mut e, zero, exp, mant, sign, v);
                        if v != 0 {
                            left -= 1;
                        }
                        partial[k] = block[k];
                    }
                }
                let (pred, dctx) = dc_prediction(&partial, &nb, q);
                let r = block[0] as i32 - pred;
                let base = dctx * 13;
                let (zero, rest) = m.dc_exp[base..base + 13].split_at_mut(1);
                e.slot(9);
                put_signed(&mut e, &mut zero[0], rest, &mut m.dc_mant[dctx * 13 * MANT..(dctx + 1) * 13 * MANT], &mut m.dc_sign[dctx * 3], r);
            }
        }
    }
    e.finish()
}

/// The blocks back from the model's stream, for a frame's layout.
pub fn decode(stream: &[u8], frame: &Frame, quant: &[Option<[u16; 64]>]) -> Option<Vec<Vec<[i16; 64]>>> {
    let mut d = Decoder::new(stream);
    let mut ctx = [Contexts::new(), Contexts::new()];
    let mut out: Vec<Vec<[i16; 64]>> = frame.components.iter().map(|c| vec![[0i16; 64]; c.bw * c.bh]).collect();
    for (ci, c) in frame.components.iter().enumerate() {
        let m = &mut ctx[kind(ci)];
        let q = quant.get(c.tq as usize)?.as_ref()?;
        for y in 0..c.bh {
            for x in 0..c.bw {
                let mut block = [0i16; 64];
                {
                    let blocks = &out[ci];
                    let nb = Neighbours {
                    left: (x > 0).then(|| &blocks[y * c.bw + x - 1]),
                    above: (y > 0).then(|| &blocks[(y - 1) * c.bw + x]),
                    corner: (x > 0 && y > 0).then(|| &blocks[(y - 1) * c.bw + x - 1]),
                };
                    let nz = d.tree(&mut m.count[nb.count() * 64..], 6);
                    let nzb = count_bucket(nz);
                    let mut left = nz;
                    for &k in INTERIOR.iter() {
                        if left == 0 {
                            break;
                        }
                        let mb = mag_bucket(nb.mag(k));
                        let z = &mut m.zero[(k * LEFT_BUCKETS + left_bucket(left)) * MAG_BUCKETS * MAG_BUCKETS + nb.mags(k)];
                        if d.bit(z) == 0 {
                            continue;
                        }
                        left -= 1;
                        let base = ((k * MAG_BUCKETS + mb) * LEFT_BUCKETS + left_bucket(left)) * 12;
                        let mag = get_magnitude(&mut d, &mut m.exp[base..base + 12], &mut m.mant[k * 12 * MANT..(k + 1) * 12 * MANT]);
                        if mag > 32767 {
                            return None;
                        }
                        let sc = nb.sign_ctx(k);
                        let neg = d.bit(&mut m.sign[k * 3 + sc]) == 1;
                        block[k] = if neg { -(mag as i16) } else { mag as i16 };
                    }
                    if left != 0 {
                        return None;
                    }
                    for dir in 0..2 {
                        let across = if dir == 0 { nb.above } else { nb.left };
                        let cc = (dir * 8 + nzb) * 8 + across.map_or(0, |b| edge_count(b, dir));
                        let mut left = d.tree(&mut m.edge_count[cc * 8..], 3) as usize;
                        for (i, &nat) in EDGES.iter().enumerate().skip(dir * 7).take(7) {
                            if left == 0 {
                                break;
                            }
                            let k = NAT2ZZ[nat];
                            let pred = edge_prediction(&block, &nb, q, nat);
                            let pb = pred.map_or(0, pred_bucket);
                            let c = ((i * EDGE_CTX + pb) * 8 + left) * 4 + mag_bucket(nb.mag(k)).min(3);
                            let zero = &mut m.edge_zero[c];
                            let exp = &mut m.edge_exp[c * 12..(c + 1) * 12];
                            let mant = &mut m.edge_mant[i * 12 * MANT..(i + 1) * 12 * MANT];
                            let sign = &mut m.edge_sign[i * EDGE_CTX + pb];
                            let v = get_signed(&mut d, zero, exp, mant, sign)?;
                            if v != 0 {
                                left -= 1;
                            }
                            block[k] = v as i16;
                        }
                        if left != 0 {
                            return None;
                        }
                    }
                    let (pred, dctx) = dc_prediction(&block, &nb, q);
                    let base = dctx * 13;
                    let (zero, rest) = m.dc_exp[base..base + 13].split_at_mut(1);
                    let r = get_signed(&mut d, &mut zero[0], rest, &mut m.dc_mant[dctx * 13 * MANT..(dctx + 1) * 13 * MANT], &mut m.dc_sign[dctx * 3])?;
                    let dc = pred + r;
                    if dc < i16::MIN as i32 || dc > i16::MAX as i32 {
                        return None;
                    }
                    block[0] = dc as i16;
                }
                out[ci][y * c.bw + x] = block;
            }
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::super::{parse, tests::fixture};
    use super::*;

    #[test]
    fn coefficients_round_trip_and_shrink() {
        for name in ["q75-420.jpg", "q90-444.jpg", "q60-422-rst.jpg", "q80-gray.jpg", "q95-opt.jpg"] {
            let data = fixture(name);
            let j = parse(&data).unwrap();
            let stream = encode(&j);
            let back = decode(&stream, &j.frame, &j.quant).unwrap();
            assert!(back == j.blocks, "{name}: coefficients back");
            eprintln!("{name}: {} B -> model {} B ({:.1}%)", data.len(), stream.len(), 100.0 * stream.len() as f64 / data.len() as f64);
            assert!(stream.len() < data.len(), "{name}: smaller");
        }
    }
}
