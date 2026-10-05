//! What a run of a model file's bytes is, and how it is cut into streams of
//! symbols: a layout is the list of streams a unit (an element or a block) gives,
//! each with the context its symbols are coded under, and `split` and `join` are
//! each other's inverse on every possible byte string of whole units.
//!
//! Floats are cut into their fields (a bf16's exponent, sign and 7 mantissa bits;
//! a half's sign and exponent, high and low mantissa), a quantised block into its
//! scales, mins and codes (the 6-bit scales of the K-quants unpacked, the 5- and
//! 6-bit codes put together from the pieces the format scatters them in).

use super::rans::{self, Ctx};

/// Bytes of a unit's chunk the coder works on at a time.
pub(crate) const CHUNK_BYTES: usize = 1 << 20;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Kind {
    /// Any 1-byte element (fp8, int8) or bytes of no known shape.
    Bytes = 1,
    /// Elements of `params[0]` bytes (2, 4 or 8) as byte planes.
    Planes = 2,
    Bf16 = 3,
    F16 = 4,
    Q8_0 = 5,
    Q4_0 = 6,
    Q4_1 = 7,
    Q5_0 = 8,
    Q5_1 = 9,
    Q4K = 10,
    Q5K = 11,
    Q6K = 12,
    Iq4Nl = 13,
    Iq4Xs = 14,
    Mxfp4 = 15,
}

impl Kind {
    pub fn from_code(c: u8) -> Option<Kind> {
        use Kind::*;
        Some(match c {
            1 => Bytes,
            2 => Planes,
            3 => Bf16,
            4 => F16,
            5 => Q8_0,
            6 => Q4_0,
            7 => Q4_1,
            8 => Q5_0,
            9 => Q5_1,
            10 => Q4K,
            11 => Q5K,
            12 => Q6K,
            13 => Iq4Nl,
            14 => Iq4Xs,
            15 => Mxfp4,
            _ => return None,
        })
    }
}

/// A region's coding: its kind, a variant (which of the layouts the kind has) and
/// the variant's parameters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Spec {
    pub kind: Kind,
    pub variant: u8,
    pub params: Vec<u8>,
}

impl Spec {
    pub fn new(kind: Kind) -> Spec {
        Spec { kind, variant: 0, params: Vec::new() }
    }
    pub fn planes(width: u8) -> Spec {
        Spec { kind: Kind::Planes, variant: 0, params: vec![width] }
    }
    pub fn unit_bytes(&self) -> usize {
        match self.kind {
            Kind::Bytes => 1,
            Kind::Planes => self.params.first().copied().unwrap_or(0) as usize,
            Kind::Bf16 | Kind::F16 => 2,
            Kind::Q8_0 => 34,
            Kind::Q4_0 | Kind::Iq4Nl => 18,
            Kind::Q4_1 => 20,
            Kind::Q5_0 => 22,
            Kind::Q5_1 => 24,
            Kind::Q4K => 144,
            Kind::Q5K => 176,
            Kind::Q6K => 210,
            Kind::Iq4Xs => 136,
            Kind::Mxfp4 => 17,
        }
    }
    pub fn chunk_units(&self) -> usize {
        (CHUNK_BYTES / self.unit_bytes().max(1)).max(1)
    }
}

/// How a stream's symbols pick their table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum CtxSpec {
    None,
    /// The symbol of stream `src` at the same position.
    Other(usize),
    /// A class per unit (`aux`), a symbol of this stream `1 << shift` to a class.
    Aux(u8),
    /// A bf16 mantissa: the mantissa before (halved), if `prev`, times 8, and the exponent's
    /// distance from `emode` (0 to 6), if `exp`.
    Bf16 { emode: u8, prev: bool, exp: bool },
    /// Q8_0's codes: whether a code of the extreme magnitude (127) has come in this block of 32.
    Q8Max,
    /// A quantiser puts the extremes (the codes 0 and 15) of every group of `period` weights in
    /// it: bytes of two 4-bit codes, whether each of them has come yet (`period` bytes, 16 or 32).
    Nibbles(u8),
    /// The same for codes of one a symbol: `period` symbols to a group, the codes `lo` and `hi`.
    Extremes { period: u8, lo: u8, hi: u8 },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StreamSpec {
    pub alphabet: usize,
    pub per_unit: usize,
    pub ctx: CtxSpec,
    /// Stored as bits, never coded (a sign).
    pub bits1: bool,
}

fn st(alphabet: usize, per_unit: usize, ctx: CtxSpec) -> StreamSpec {
    StreamSpec { alphabet, per_unit, ctx, bits1: false }
}

impl StreamSpec {
    pub fn nctx(&self) -> usize {
        match &self.ctx {
            CtxSpec::None => 1,
            CtxSpec::Other(_) => 256,
            CtxSpec::Aux(_) => 256,
            CtxSpec::Bf16 { prev, .. } => (if *prev { 64 } else { 1 }) * 8,
            CtxSpec::Q8Max => 3,
            CtxSpec::Nibbles(_) => 64,
            CtxSpec::Extremes { .. } => 12,
        }
    }
}

const QS: usize = 256;

/// The scale of a block as two streams: its high byte, and its low byte under the high one.
fn scale16(v: &mut Vec<StreamSpec>) {
    let hi = v.len();
    v.push(st(256, 1, CtxSpec::None));
    v.push(st(256, 1, CtxSpec::Other(hi)));
}

/// Streams of a spec, in the order they are decoded (a stream's context comes from earlier ones).
pub(crate) fn layout(spec: &Spec) -> Option<Vec<StreamSpec>> {
    // the kinds with variants: bf16 (0, 1) and Q8_0 (0, 1, 2); every other has one
    match spec.kind {
        Kind::Bf16 if spec.variant > 1 => return None,
        Kind::Q8_0 if spec.variant > 2 => return None,
        Kind::Bf16 | Kind::Q8_0 => {}
        _ if spec.variant != 0 => return None,
        _ => {}
    }
    let mut v = Vec::new();
    match spec.kind {
        Kind::Bytes => v.push(st(QS, 1, CtxSpec::None)),
        Kind::Planes => {
            let w = *spec.params.first()? as usize;
            if !(2..=8).contains(&w) {
                return None;
            }
            for _ in 0..w {
                v.push(st(QS, 1, CtxSpec::None));
            }
        }
        Kind::Bf16 => {
            v.push(st(QS, 1, CtxSpec::None));
            match spec.variant {
                0 => v.push(st(QS, 1, CtxSpec::None)), // sign and mantissa as a byte
                1 => {
                    let flags = *spec.params.first()?;
                    let emode = *spec.params.get(1)?;
                    v.push(StreamSpec { alphabet: 2, per_unit: 1, ctx: CtxSpec::None, bits1: true });
                    v.push(st(128, 1, CtxSpec::Bf16 { emode, prev: flags & 1 != 0, exp: flags & 2 != 0 }));
                }
                _ => return None,
            }
        }
        Kind::F16 => {
            v.push(st(64, 1, CtxSpec::None));
            v.push(st(128, 1, CtxSpec::None));
            v.push(st(8, 1, CtxSpec::None));
        }
        Kind::Q8_0 => {
            scale16(&mut v);
            v.push(st(QS, 32, match spec.variant {
                1 => CtxSpec::Aux(5),
                2 => CtxSpec::Q8Max,
                _ => CtxSpec::None,
            }));
        }
        Kind::Q4_0 | Kind::Iq4Nl => {
            scale16(&mut v);
            v.push(st(QS, 16, CtxSpec::Nibbles(16)));
        }
        Kind::Q4_1 => {
            scale16(&mut v);
            scale16(&mut v);
            v.push(st(QS, 16, CtxSpec::Nibbles(16)));
        }
        Kind::Q5_0 => {
            scale16(&mut v);
            v.push(st(32, 32, CtxSpec::Extremes { period: 32, lo: 0, hi: 31 }));
        }
        Kind::Q5_1 => {
            scale16(&mut v);
            scale16(&mut v);
            v.push(st(32, 32, CtxSpec::Extremes { period: 32, lo: 0, hi: 31 }));
        }
        Kind::Q4K => {
            scale16(&mut v);
            scale16(&mut v);
            v.push(st(64, 8, CtxSpec::None));
            v.push(st(64, 8, CtxSpec::Other(4)));
            v.push(st(QS, 128, CtxSpec::Nibbles(32)));
        }
        Kind::Q5K => {
            scale16(&mut v);
            scale16(&mut v);
            v.push(st(64, 8, CtxSpec::None));
            v.push(st(64, 8, CtxSpec::Other(4)));
            v.push(st(32, 256, CtxSpec::Extremes { period: 32, lo: 0, hi: 31 }));
        }
        Kind::Q6K => {
            v.push(st(64, 256, CtxSpec::Extremes { period: 16, lo: 0, hi: 63 }));
            v.push(st(QS, 16, CtxSpec::None));
            scale16(&mut v);
        }
        Kind::Iq4Xs => {
            scale16(&mut v);
            v.push(st(64, 8, CtxSpec::None));
            v.push(st(QS, 128, CtxSpec::Nibbles(32)));
        }
        Kind::Mxfp4 => {
            v.push(st(QS, 1, CtxSpec::None));
            v.push(st(QS, 16, CtxSpec::None));
        }
    }
    Some(v)
}

/// The class of every distinct block scale a Q8_0 variant 1 names (its `params`: the
/// scales as `u16`s, in class order); 255 for any other.
pub(crate) fn scale_classes(params: &[u8]) -> Option<Vec<u8>> {
    if params.len() % 2 != 0 || params.len() / 2 > 255 {
        return None;
    }
    let mut lookup = vec![255u8; 1 << 16];
    for (i, c) in params.chunks_exact(2).enumerate() {
        lookup[u16::from_le_bytes([c[0], c[1]]) as usize] = i as u8;
    }
    Some(lookup)
}

/// The class of each unit, from the streams already known (Q8_0: from the scale's two bytes).
pub(crate) fn aux(spec: &Spec, lookup: &[u8], streams: &[&[u8]]) -> Vec<u8> {
    debug_assert!(spec.kind == Kind::Q8_0);
    let mut out = vec![0u8; streams[0].len()];
    aux_into(lookup, streams[0], streams[1], &mut out);
    out
}

pub(crate) fn aux_into(lookup: &[u8], hi: &[u8], lo: &[u8], out: &mut [u8]) {
    for ((o, &h), &l) in out.iter_mut().zip(hi).zip(lo) {
        *o = lookup[(h as usize) << 8 | l as usize];
    }
}

// ------------------------------------------------------------------ split and join

/// `src` (whole units) cut into the layout's streams. `out[s]` is cleared and filled
/// (the signs of a bf16's variant 1 packed eight to a byte).
pub(crate) fn split(spec: &Spec, src: &[u8], out: &mut [Vec<u8>]) {
    let ub = spec.unit_bytes();
    let n = src.len() / ub;
    debug_assert!(src.len() % ub == 0);
    for o in out.iter_mut() {
        o.clear();
    }
    match spec.kind {
        Kind::Bytes => out[0].extend_from_slice(src),
        Kind::Planes => {
            for (k, o) in out.iter_mut().enumerate() {
                o.extend(src.chunks_exact(ub).map(|e| e[k]));
            }
        }
        Kind::Bf16 => {
            let (e, rest) = out.split_first_mut().unwrap();
            e.reserve(n);
            if spec.variant == 0 {
                let sm = &mut rest[0];
                sm.reserve(n);
                for c in src.chunks_exact(2) {
                    let v = u16::from_le_bytes([c[0], c[1]]);
                    e.push((v >> 7) as u8);
                    sm.push(((v >> 8) & 0x80) as u8 | (v & 0x7F) as u8);
                }
            } else {
                let (s, m) = rest.split_at_mut(1);
                let (s, m) = (&mut s[0], &mut m[0]);
                s.resize(n.div_ceil(8), 0);
                m.reserve(n);
                for (i, c) in src.chunks_exact(2).enumerate() {
                    let v = u16::from_le_bytes([c[0], c[1]]);
                    e.push((v >> 7) as u8);
                    s[i >> 3] |= ((v >> 15) as u8) << (i & 7);
                    m.push((v & 0x7F) as u8);
                }
            }
        }
        Kind::F16 => {
            for c in src.chunks_exact(2) {
                let v = u16::from_le_bytes([c[0], c[1]]);
                out[0].push(((v >> 10) & 0x3F) as u8);
                out[1].push(((v >> 3) & 0x7F) as u8);
                out[2].push((v & 7) as u8);
            }
        }
        Kind::Q8_0 => {
            for b in src.chunks_exact(34) {
                out[0].push(b[1]);
                out[1].push(b[0]);
                out[2].extend_from_slice(&b[2..]);
            }
        }
        Kind::Q4_0 | Kind::Iq4Nl => {
            for b in src.chunks_exact(18) {
                out[0].push(b[1]);
                out[1].push(b[0]);
                out[2].extend_from_slice(&b[2..]);
            }
        }
        Kind::Q4_1 => {
            for b in src.chunks_exact(20) {
                out[0].push(b[1]);
                out[1].push(b[0]);
                out[2].push(b[3]);
                out[3].push(b[2]);
                out[4].extend_from_slice(&b[4..]);
            }
        }
        Kind::Q5_0 => {
            for b in src.chunks_exact(22) {
                out[0].push(b[1]);
                out[1].push(b[0]);
                let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
                let qs = &b[6..22];
                for j in 0..16 {
                    out[2].push((qs[j] & 15) | (((qh >> j) & 1) as u8) << 4);
                }
                for j in 0..16 {
                    out[2].push((qs[j] >> 4) | (((qh >> (j + 16)) & 1) as u8) << 4);
                }
            }
        }
        Kind::Q5_1 => {
            for b in src.chunks_exact(24) {
                out[0].push(b[1]);
                out[1].push(b[0]);
                out[2].push(b[3]);
                out[3].push(b[2]);
                let qh = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
                let qs = &b[8..24];
                for j in 0..16 {
                    out[4].push((qs[j] & 15) | (((qh >> j) & 1) as u8) << 4);
                }
                for j in 0..16 {
                    out[4].push((qs[j] >> 4) | (((qh >> (j + 16)) & 1) as u8) << 4);
                }
            }
        }
        Kind::Q4K | Kind::Q5K => {
            let five = spec.kind == Kind::Q5K;
            let size = ub;
            for b in src.chunks_exact(size) {
                out[0].push(b[1]);
                out[1].push(b[0]);
                out[2].push(b[3]);
                out[3].push(b[2]);
                let s = &b[4..16];
                for j in 0..4 {
                    out[4].push(s[j] & 63);
                }
                for j in 4..8 {
                    out[4].push((s[j + 4] & 15) | ((s[j - 4] >> 6) << 4));
                }
                for j in 0..4 {
                    out[5].push(s[j + 4] & 63);
                }
                for j in 4..8 {
                    out[5].push((s[j + 4] >> 4) | ((s[j] >> 6) << 4));
                }
                if five {
                    let qh = &b[16..48];
                    let qs = &b[48..176];
                    for g in 0..4 {
                        for l in 0..32 {
                            out[6].push((qs[g * 32 + l] & 15) | ((qh[l] >> (2 * g)) & 1) << 4);
                        }
                        for l in 0..32 {
                            out[6].push((qs[g * 32 + l] >> 4) | ((qh[l] >> (2 * g + 1)) & 1) << 4);
                        }
                    }
                } else {
                    out[6].extend_from_slice(&b[16..144]);
                }
            }
        }
        Kind::Q6K => {
            for b in src.chunks_exact(210) {
                let (ql, qh) = (&b[0..128], &b[128..192]);
                for h in 0..2 {
                    let (ql, qh) = (&ql[64 * h..], &qh[32 * h..]);
                    // the four quarters of this half: weights l, l + 32, l + 64, l + 96
                    for q in 0..4 {
                        for l in 0..32 {
                            let low = if q < 2 { ql[l + 32 * (q & 1)] & 15 } else { ql[l + 32 * (q & 1)] >> 4 };
                            out[0].push(low | ((qh[l] >> (2 * q)) & 3) << 4);
                        }
                    }
                }
                out[1].extend_from_slice(&b[192..208]);
                out[2].push(b[209]);
                out[3].push(b[208]);
            }
        }
        Kind::Iq4Xs => {
            for b in src.chunks_exact(136) {
                out[0].push(b[1]);
                out[1].push(b[0]);
                let sh = u16::from_le_bytes([b[2], b[3]]);
                for ib in 0..8 {
                    out[2].push(((b[4 + ib / 2] >> (4 * (ib % 2))) & 15) | (((sh >> (2 * ib)) & 3) as u8) << 4);
                }
                out[3].extend_from_slice(&b[8..136]);
            }
        }
        Kind::Mxfp4 => {
            for b in src.chunks_exact(17) {
                out[0].push(b[0]);
                out[1].extend_from_slice(&b[1..]);
            }
        }
    }
}

/// The streams back to whole units: the inverse of `split`.
pub(crate) fn join(spec: &Spec, st: &[&[u8]], n: usize, dst: &mut [u8]) {
    let ub = spec.unit_bytes();
    debug_assert!(dst.len() == n * ub);
    match spec.kind {
        Kind::Bytes => dst.copy_from_slice(st[0]),
        Kind::Planes => {
            for (i, e) in dst.chunks_exact_mut(ub).enumerate() {
                for k in 0..ub {
                    e[k] = st[k][i];
                }
            }
        }
        Kind::Bf16 => {
            if spec.variant == 0 {
                for ((c, &e), &sm) in dst.chunks_exact_mut(2).zip(st[0]).zip(st[1]) {
                    let sm = sm as u16;
                    let v = (sm & 0x80) << 8 | (e as u16) << 7 | (sm & 0x7F);
                    c.copy_from_slice(&v.to_le_bytes());
                }
            } else {
                for (((c, e), m), &sb) in dst.chunks_mut(16).zip(st[0].chunks(8)).zip(st[2].chunks(8)).zip(st[1]) {
                    for k in 0..e.len() {
                        let v = (((sb >> k) & 1) as u16) << 15 | (e[k] as u16) << 7 | (m[k] & 0x7F) as u16;
                        c[2 * k..2 * k + 2].copy_from_slice(&v.to_le_bytes());
                    }
                }
            }
        }
        Kind::F16 => {
            for (i, c) in dst.chunks_exact_mut(2).enumerate() {
                let v = ((st[0][i] & 0x3F) as u16) << 10 | ((st[1][i] & 0x7F) as u16) << 3 | (st[2][i] & 7) as u16;
                c.copy_from_slice(&v.to_le_bytes());
            }
        }
        Kind::Q8_0 => {
            for (i, b) in dst.chunks_exact_mut(34).enumerate() {
                b[0] = st[1][i];
                b[1] = st[0][i];
                b[2..].copy_from_slice(&st[2][i * 32..i * 32 + 32]);
            }
        }
        Kind::Q4_0 | Kind::Iq4Nl => {
            for (i, b) in dst.chunks_exact_mut(18).enumerate() {
                b[0] = st[1][i];
                b[1] = st[0][i];
                b[2..].copy_from_slice(&st[2][i * 16..i * 16 + 16]);
            }
        }
        Kind::Q4_1 => {
            for (i, b) in dst.chunks_exact_mut(20).enumerate() {
                b[0] = st[1][i];
                b[1] = st[0][i];
                b[2] = st[3][i];
                b[3] = st[2][i];
                b[4..].copy_from_slice(&st[4][i * 16..i * 16 + 16]);
            }
        }
        Kind::Q5_0 => {
            for (i, b) in dst.chunks_exact_mut(22).enumerate() {
                b[0] = st[1][i];
                b[1] = st[0][i];
                let c = &st[2][i * 32..i * 32 + 32];
                let mut qh = 0u32;
                for j in 0..16 {
                    b[6 + j] = (c[j] & 15) | (c[j + 16] & 15) << 4;
                    qh |= ((c[j] >> 4) as u32 & 1) << j | ((c[j + 16] >> 4) as u32 & 1) << (j + 16);
                }
                b[2..6].copy_from_slice(&qh.to_le_bytes());
            }
        }
        Kind::Q5_1 => {
            for (i, b) in dst.chunks_exact_mut(24).enumerate() {
                b[0] = st[1][i];
                b[1] = st[0][i];
                b[2] = st[3][i];
                b[3] = st[2][i];
                let c = &st[4][i * 32..i * 32 + 32];
                let mut qh = 0u32;
                for j in 0..16 {
                    b[8 + j] = (c[j] & 15) | (c[j + 16] & 15) << 4;
                    qh |= ((c[j] >> 4) as u32 & 1) << j | ((c[j + 16] >> 4) as u32 & 1) << (j + 16);
                }
                b[4..8].copy_from_slice(&qh.to_le_bytes());
            }
        }
        Kind::Q4K | Kind::Q5K => {
            let five = spec.kind == Kind::Q5K;
            for (i, b) in dst.chunks_exact_mut(ub).enumerate() {
                b[0] = st[1][i];
                b[1] = st[0][i];
                b[2] = st[3][i];
                b[3] = st[2][i];
                let sc = &st[4][i * 8..i * 8 + 8];
                let mn = &st[5][i * 8..i * 8 + 8];
                for j in 0..4 {
                    b[4 + j] = (sc[j] & 63) | (sc[j + 4] >> 4) << 6;
                    b[8 + j] = (mn[j] & 63) | (mn[j + 4] >> 4) << 6;
                    b[12 + j] = (sc[j + 4] & 15) | (mn[j + 4] & 15) << 4;
                }
                if five {
                    let c = &st[6][i * 256..i * 256 + 256];
                    for l in 0..32 {
                        b[16 + l] = 0;
                    }
                    for g in 0..4 {
                        for l in 0..32 {
                            let (lo, hi) = (c[g * 64 + l], c[g * 64 + 32 + l]);
                            b[48 + g * 32 + l] = (lo & 15) | (hi & 15) << 4;
                            b[16 + l] |= ((lo >> 4) & 1) << (2 * g) | ((hi >> 4) & 1) << (2 * g + 1);
                        }
                    }
                } else {
                    b[16..144].copy_from_slice(&st[6][i * 128..i * 128 + 128]);
                }
            }
        }
        Kind::Q6K => {
            for (i, b) in dst.chunks_exact_mut(210).enumerate() {
                let c = &st[0][i * 256..i * 256 + 256];
                for h in 0..2 {
                    for l in 0..32 {
                        let (q1, q2, q3, q4) = (c[128 * h + l], c[128 * h + 32 + l], c[128 * h + 64 + l], c[128 * h + 96 + l]);
                        b[64 * h + l] = (q1 & 15) | (q3 & 15) << 4;
                        b[64 * h + 32 + l] = (q2 & 15) | (q4 & 15) << 4;
                        b[128 + 32 * h + l] = (q1 >> 4) | (q2 >> 4) << 2 | (q3 >> 4) << 4 | (q4 >> 4) << 6;
                    }
                }
                b[192..208].copy_from_slice(&st[1][i * 16..i * 16 + 16]);
                b[208] = st[3][i];
                b[209] = st[2][i];
            }
        }
        Kind::Iq4Xs => {
            for (i, b) in dst.chunks_exact_mut(136).enumerate() {
                b[0] = st[1][i];
                b[1] = st[0][i];
                let ls = &st[2][i * 8..i * 8 + 8];
                let mut sh = 0u16;
                for k in 0..4 {
                    b[4 + k] = (ls[2 * k] & 15) | (ls[2 * k + 1] & 15) << 4;
                }
                for ib in 0..8 {
                    sh |= ((ls[ib] >> 4) as u16 & 3) << (2 * ib);
                }
                b[2..4].copy_from_slice(&sh.to_le_bytes());
                b[8..].copy_from_slice(&st[3][i * 128..i * 128 + 128]);
            }
        }
        Kind::Mxfp4 => {
            for (i, b) in dst.chunks_exact_mut(17).enumerate() {
                b[0] = st[0][i];
                b[1..].copy_from_slice(&st[1][i * 16..i * 16 + 16]);
            }
        }
    }
}

// ------------------------------------------------------------------ contexts

/// The symbols of each lane of an `n`-symbol stream: the first `LANES - 1` take `n / LANES` each, the last the rest.
pub(crate) fn lane_ranges(n: usize) -> [(usize, usize); rans::LANES] {
    let seg = n / rans::LANES;
    let mut r = [(0, 0); rans::LANES];
    for (l, x) in r.iter_mut().enumerate() {
        *x = (l * seg, if l + 1 == rans::LANES { n } else { (l + 1) * seg });
    }
    r
}

#[inline(always)]
fn erel(e: u8, emode: u8) -> usize {
    (e as i32 - emode as i32 + 3).clamp(0, 6) as usize
}

/// The context of every symbol of stream `s`, as the decoder computes it. `streams` are the
/// symbols of the streams before `s`; `aux` the class of every unit (a Q8_0 variant 1's).
pub(crate) fn contexts(layout: &[StreamSpec], s: usize, streams: &[Vec<u8>], aux: &[u8]) -> Option<Vec<u16>> {
    let n = streams[s].len();
    let syms = &streams[s];
    match &layout[s].ctx {
        CtxSpec::None => None,
        CtxSpec::Other(src) => Some(streams[*src].iter().map(|&x| x as u16).collect()),
        CtxSpec::Aux(shift) => Some((0..n).map(|i| aux[i >> shift] as u16).collect()),
        CtxSpec::Bf16 { emode, prev, exp } => {
            let e = &streams[0];
            let mut out = vec![0u16; n];
            for (a, b) in lane_ranges(n) {
                for i in a..b {
                    let p = if *prev && i > a { (syms[i - 1] >> 1) as usize } else { 0 };
                    let x = if *exp { erel(e[i], *emode) } else { 0 };
                    out[i] = (p << 3 | x) as u16;
                }
            }
            Some(out)
        }
        CtxSpec::Q8Max => Some(simulate(&Q8Ctx, syms)),
        CtxSpec::Nibbles(period) => Some(simulate(&NibbleCtx(*period as usize - 1), syms)),
        CtxSpec::Extremes { period, lo, hi } => Some(simulate(&ExtremeCtx { mask: *period as usize - 1, lo: *lo, hi: *hi }, syms)),
    }
}

/// The contexts a decoder meets: the context function stepped along every lane.
fn simulate<C: Ctx>(c: &C, syms: &[u8]) -> Vec<u16> {
    let mut out = vec![0u16; syms.len()];
    for (a, b) in lane_ranges(syms.len()) {
        let mut st = 0u8;
        for i in a..b {
            out[i] = c.at(i, st) as u16;
            st = c.next(i, syms[i], st);
        }
    }
    out
}

pub(crate) struct OtherCtx<'a>(pub &'a [u8]);
impl Ctx for OtherCtx<'_> {
    #[inline(always)]
    fn at(&self, i: usize, _: u8) -> usize {
        // SAFETY: the other stream has as many symbols as this one.
        debug_assert!(i < self.0.len());
        unsafe { *self.0.get_unchecked(i) as usize }
    }
    #[inline(always)]
    fn next(&self, _: usize, _: u8, _: u8) -> u8 {
        0
    }
}

pub(crate) struct AuxCtx<'a>(pub &'a [u8], pub u8);
impl Ctx for AuxCtx<'_> {
    #[inline(always)]
    fn at(&self, i: usize, _: u8) -> usize {
        // SAFETY: the classes have one entry for every unit, and a stream has `1 << shift` symbols to a unit.
        debug_assert!(i >> self.1 < self.0.len());
        unsafe { *self.0.get_unchecked(i >> self.1) as usize }
    }
    #[inline(always)]
    fn next(&self, _: usize, _: u8, _: u8) -> u8 {
        0
    }
}

/// A bf16 mantissa's context: the exponent's class of each position (`classes`, from
/// `bf16_classes_into`) and, with `pm` = 0x7E, the mantissa before; `pm` = 0 leaves the exponent alone.
pub(crate) struct Bf16Ctx<'a> {
    pub classes: &'a [u8],
    pub pm: u8,
}
impl Ctx for Bf16Ctx<'_> {
    #[inline(always)]
    fn at(&self, i: usize, st: u8) -> usize {
        // SAFETY: the stream and `classes` have one entry for every position (`decode_chunk` makes them so).
        debug_assert!(i < self.classes.len());
        ((st & self.pm) as usize) << 2 | unsafe { *self.classes.get_unchecked(i) } as usize
    }
    #[inline(always)]
    fn next(&self, _: usize, sym: u8, _: u8) -> u8 {
        sym
    }
}

/// How far into its group of `mask + 1` symbols position `i` is, as 0 (early), 1 (late), 2 (the last).
#[inline(always)]
fn group_pos(i: usize, mask: usize) -> usize {
    let p = i & mask;
    (p >= mask - 7) as usize + (p == mask) as usize
}

/// Q8_0: one flag, a code of magnitude 127 seen in the block of 32; the last code of a block that has
/// none is one (the quantiser scales the largest to it). Contexts: seen, last and unseen, other.
pub(crate) struct Q8Ctx;
impl Ctx for Q8Ctx {
    #[inline(always)]
    fn at(&self, i: usize, st: u8) -> usize {
        (st & 1) as usize | (((i & 31 == 31) as usize) & !(st as usize)) << 1
    }
    #[inline(always)]
    fn next(&self, i: usize, sym: u8, st: u8) -> u8 {
        let hit = ((sym as i8).unsigned_abs() >= 127) as u8;
        (st | hit) * (((i + 1) & 31 != 0) as u8)
    }
}

/// Bytes of two 4-bit codes, `mask + 1` bytes to a group: flags for code 0 and 15 seen in the low
/// and the high nibbles, and how late in the group it is.
pub(crate) struct NibbleCtx(pub usize);
impl Ctx for NibbleCtx {
    #[inline(always)]
    fn at(&self, i: usize, st: u8) -> usize {
        (st & 15) as usize | group_pos(i, self.0) << 4
    }
    #[inline(always)]
    fn next(&self, i: usize, sym: u8, st: u8) -> u8 {
        let (lo, hi) = (sym & 15, sym >> 4);
        let f = (lo == 0) as u8 | ((lo == 15) as u8) << 1 | ((hi == 0) as u8) << 2 | ((hi == 15) as u8) << 3;
        (st | f) * (((i + 1) & self.0 != 0) as u8)
    }
}

/// One code a symbol, `mask + 1` symbols to a group: flags for the codes `lo` and `hi` seen, and how
/// late in the group it is.
pub(crate) struct ExtremeCtx {
    pub mask: usize,
    pub lo: u8,
    pub hi: u8,
}
impl Ctx for ExtremeCtx {
    #[inline(always)]
    fn at(&self, i: usize, st: u8) -> usize {
        (st & 3) as usize | group_pos(i, self.mask) << 2
    }
    #[inline(always)]
    fn next(&self, i: usize, sym: u8, st: u8) -> u8 {
        let f = (sym == self.lo) as u8 | ((sym == self.hi) as u8) << 1;
        (st | f) * (((i + 1) & self.mask != 0) as u8)
    }
}

/// The exponent class of each element of a bf16 chunk (0 when the context does not use it).
pub(crate) fn bf16_classes_into(e: &[u8], emode: u8, exp: bool, out: &mut [u8]) {
    if exp {
        for (o, &x) in out.iter_mut().zip(e) {
            *o = erel(x, emode) as u8;
        }
    } else {
        out.fill(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(n: usize, seed: u64) -> Vec<u8> {
        let mut x = seed | 1;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 24) as u8
            })
            .collect()
    }

    fn all_specs() -> Vec<Spec> {
        use Kind::*;
        let mut v = vec![Spec::planes(2), Spec::planes(4), Spec::planes(8)];
        for k in [Bytes, F16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q4K, Q5K, Q6K, Iq4Nl, Iq4Xs, Mxfp4] {
            v.push(Spec::new(k));
        }
        v.push(Spec::new(Bf16));
        v.push(Spec { kind: Bf16, variant: 1, params: vec![3, 120] });
        v.push(Spec { kind: Q8_0, variant: 1, params: vec![1, 0, 2, 0] });
        v
    }

    #[test]
    fn split_join_inverse_on_every_byte_string() {
        for spec in all_specs() {
            let l = layout(&spec).unwrap();
            for units in [0usize, 1, 7, 100] {
                let src = bytes(units * spec.unit_bytes(), 17 + units as u64);
                let mut st = vec![Vec::new(); l.len()];
                split(&spec, &src, &mut st);
                for (s, sp) in l.iter().enumerate() {
                    let want = if sp.bits1 { units.div_ceil(8) } else { units * sp.per_unit };
                    assert_eq!(st[s].len(), want, "{:?} stream {}", spec, s);
                    if !sp.bits1 {
                        assert!(st[s].iter().all(|&x| (x as usize) < sp.alphabet), "{:?} stream {} symbol range", spec, s);
                    }
                }
                let refs: Vec<&[u8]> = st.iter().map(|v| v.as_slice()).collect();
                let mut back = vec![0u8; src.len()];
                join(&spec, &refs, units, &mut back);
                assert_eq!(back, src, "{:?} units {}", spec, units);
            }
        }
    }

    #[test]
    fn layouts_are_refused_when_unknown() {
        assert!(layout(&Spec::planes(1)).is_none());
        assert!(layout(&Spec { kind: Kind::Bf16, variant: 9, params: vec![] }).is_none());
        assert!(Kind::from_code(0).is_none() && Kind::from_code(16).is_none());
    }
}
