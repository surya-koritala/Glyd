//! JPEGs taken apart into their coefficients and put back bit for bit,
//! with nothing outside this crate (`docs/design/jpeg-recoding.md`).
//!
//! This is the first part: a parser that keeps every marker segment as
//! it is and decodes each baseline (sequential Huffman, 8-bit) scan to
//! its blocks' coefficients in zigzag order, and a writer that encodes
//! them back under the same tables with the same restart markers and
//! padding. Progressive, arithmetic-coded, 12-bit, lossless and
//! hierarchical JPEGs are not taken apart (`parse` returns `None`).

pub mod coder;
pub mod model;

use crate::record::{get_varint, put_varint};

/// A frame component: its id and sampling factors, and the block grid
/// it gets — `bw` × `bh` blocks as an interleaved scan lays them out
/// (whole MCUs), of which `cw` × `ch` cover the picture.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Component {
    pub id: u8,
    pub h: u8,
    pub v: u8,
    pub tq: u8,
    pub bw: usize,
    pub bh: usize,
    pub cw: usize,
    pub ch: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    pub width: u16,
    pub height: u16,
    pub components: Vec<Component>,
    pub hmax: u8,
    pub vmax: u8,
    pub mcux: usize,
    pub mcuy: usize,
}

/// A scan's header as read: its components (index into the frame's,
/// DC table, AC table), and the restart interval in force.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Scan {
    pub components: Vec<(usize, u8, u8)>,
    pub restart: u16,
    /// pad bits before a marker that were not all ones: (which marker
    /// counted from the scan's first restart, the bits)
    pub odd_pads: Vec<(u32, u8)>,
}

/// A piece of the file: marker segments kept as they are, or a scan's
/// entropy-coded data, whose coefficients live in `Jpeg::blocks`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Part {
    Bytes(Vec<u8>),
    Scan(Scan),
}

/// A JPEG taken apart.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Jpeg {
    pub frame: Frame,
    pub parts: Vec<Part>,
    /// per component, `bw * bh` blocks of 64 coefficients in zigzag order
    pub blocks: Vec<Vec<[i16; 64]>>,
    /// the quantization tables, in zigzag order, by id
    pub quant: Vec<Option<[u16; 64]>>,
    /// the Huffman tables in force, for the writer: [class][id]
    tables: Vec<Vec<Option<Huffman>>>,
    /// the tables each scan was written with, snapshotted
    scan_tables: Vec<Vec<Vec<Option<Huffman>>>>,
}

pub const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// A Huffman table as the DHT segment gives it: the code lengths'
/// counts and the symbols, plus what decoding and encoding need.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Huffman {
    counts: [u8; 16],
    symbols: Vec<u8>,
    /// (code, length) per symbol value, length 0 when absent
    codes: [(u16, u8); 256],
    /// decoding: the first code and the symbol index of each length
    mincode: [i32; 17],
    maxcode: [i32; 17],
    valptr: [i32; 17],
    /// decoding: by the next 9 bits, `len << 8 | symbol` for a code of
    /// up to 9 bits, 0 for a longer one
    lookup: Vec<u16>,
}

const LOOKUP_BITS: u32 = 9;

impl Huffman {
    fn new(counts: [u8; 16], symbols: Vec<u8>) -> Option<Huffman> {
        let mut codes = [(0u16, 0u8); 256];
        let (mut mincode, mut maxcode, mut valptr) = ([0i32; 17], [-1i32; 17], [0i32; 17]);
        let mut code = 0u32;
        let mut k = 0usize;
        for len in 1..=16usize {
            let n = counts[len - 1] as usize;
            valptr[len] = k as i32;
            mincode[len] = code as i32;
            for _ in 0..n {
                let sym = *symbols.get(k)? as usize;
                if code >= 1 << len {
                    return None;
                }
                codes[sym] = (code as u16, len as u8);
                code += 1;
                k += 1;
            }
            maxcode[len] = if n > 0 { code as i32 - 1 } else { -1 };
            code <<= 1;
        }
        if k != symbols.len() {
            return None;
        }
        let mut lookup = vec![0u16; 1 << LOOKUP_BITS];
        for (sym, &(code, len)) in codes.iter().enumerate() {
            if len == 0 || len as u32 > LOOKUP_BITS {
                continue;
            }
            let first = (code as usize) << (LOOKUP_BITS - len as u32);
            for e in &mut lookup[first..first + (1 << (LOOKUP_BITS - len as u32))] {
                *e = (len as u16) << 8 | sym as u16;
            }
        }
        Some(Huffman { counts, symbols, codes, mincode, maxcode, valptr, lookup })
    }
}

// ---------------------------------------------------------------- reading

fn be16(d: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_be_bytes([*d.get(at)?, *d.get(at + 1)?]))
}

/// Reads entropy-coded data: bits most significant first, a 0x00 after
/// 0xFF dropped, a marker ending the data.
struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
    acc: u32,
    n: u32,
    /// the bits taken from the last partial byte, for the pad check
    hit_marker: bool,
}

impl<'a> Bits<'a> {
    fn new(data: &'a [u8], pos: usize) -> Self {
        Bits { data, pos, acc: 0, n: 0, hit_marker: false }
    }

    #[inline]
    fn fill(&mut self) {
        while self.n <= 24 {
            if self.hit_marker || self.pos >= self.data.len() {
                // Past the data: zeros, which the decoder will not use
                // when the stream is well formed.
                self.acc <<= 8;
                self.n += 8;
                continue;
            }
            let b = self.data[self.pos];
            if b == 0xFF {
                let next = self.data.get(self.pos + 1).copied().unwrap_or(0xFF);
                if next == 0x00 {
                    self.pos += 2;
                } else {
                    self.hit_marker = true;
                    continue;
                }
            } else {
                self.pos += 1;
            }
            self.acc = (self.acc << 8) | b as u32;
            self.n += 8;
        }
    }

    #[inline]
    fn get(&mut self, k: u32) -> u32 {
        if k == 0 {
            return 0;
        }
        if self.n < k {
            self.fill();
        }
        let v = (self.acc >> (self.n - k)) & ((1 << k) - 1);
        self.n -= k;
        v
    }

    #[inline]
    fn decode(&mut self, h: &Huffman) -> Option<u8> {
        if self.n < LOOKUP_BITS {
            self.fill();
        }
        let peek = (self.acc >> (self.n - LOOKUP_BITS)) & ((1 << LOOKUP_BITS) - 1);
        let hit = h.lookup[peek as usize];
        if hit != 0 {
            self.n -= (hit >> 8) as u32;
            return Some(hit as u8);
        }
        let mut code = peek as i32;
        self.n -= LOOKUP_BITS;
        for len in LOOKUP_BITS as usize + 1..=16usize {
            code = (code << 1) | self.get(1) as i32;
            if h.maxcode[len] >= 0 && code <= h.maxcode[len] && code >= h.mincode[len] {
                return h.symbols.get((h.valptr[len] + code - h.mincode[len]) as usize).copied();
            }
        }
        None
    }

    /// To the byte boundary before a marker: the pad bits (their count
    /// is what is left of the byte, at most 7), and the position of the
    /// marker's 0xFF.
    fn align(&mut self) -> (u8, u32, usize) {
        // Bits buffered beyond the current byte were read ahead from
        // whole bytes; give them back.
        let whole = self.n / 8;
        let pad_bits = self.n % 8;
        let pad = (self.acc >> (self.n - pad_bits)) & ((1 << pad_bits) - 1);
        // Un-read `whole` bytes: walk back over them (each was one input
        // byte, or two when stuffed).
        let mut back = whole;
        let mut pos = self.pos;
        while back > 0 {
            if self.hit_marker {
                // Those were zero bytes past the data, not input.
                back -= 1;
                continue;
            }
            pos -= 1;
            if pos > 0 && self.data[pos] == 0x00 && self.data[pos - 1] == 0xFF {
                pos -= 1;
            }
            back -= 1;
        }
        self.pos = pos;
        self.acc = 0;
        self.n = 0;
        self.hit_marker = false;
        (pad as u8, pad_bits, self.pos)
    }
}

#[inline]
fn extend(v: u32, s: u32) -> i16 {
    if s == 0 {
        0
    } else if v < (1 << (s - 1)) {
        (v as i32 - (1 << s) + 1) as i16
    } else {
        v as i16
    }
}

/// Everything from the first byte: `None` when it is not a JPEG this
/// takes apart. The first byte pair must be SOI.
pub fn parse(data: &[u8]) -> Option<Jpeg> {
    parse_with(data, None)
}

/// With `pads`, `data` is a JPEG with every scan's data cut out (as
/// `pack` keeps it): the scans get their pad bits from the list, the
/// blocks stay zero for the model to fill.
fn parse_with(data: &[u8], mut pads: Option<&mut std::vec::IntoIter<Vec<(u32, u8)>>>) -> Option<Jpeg> {
    if data.len() < 4 || data[0] != 0xFF || data[1] != 0xD8 {
        return None;
    }
    let mut parts = Vec::new();
    let mut frame: Option<Frame> = None;
    let mut blocks: Vec<Vec<[i16; 64]>> = Vec::new();
    let mut tables: Vec<Vec<Option<Huffman>>> = vec![vec![None; 4], vec![None; 4]];
    let mut scan_tables = Vec::new();
    let mut quant: Vec<Option<[u16; 64]>> = vec![None; 4];
    let mut restart = 0u16;
    let mut kept_from = 0usize;
    let mut at = 2usize;
    loop {
        if *data.get(at)? != 0xFF {
            return None;
        }
        let marker = *data.get(at + 1)?;
        match marker {
            0xD8 | 0x01 | 0xD0..=0xD7 => {
                // Standalone markers outside a scan: kept.
                at += 2;
                continue;
            }
            0xD9 => {
                // EOI, then anything after it, all kept.
                parts.push(Part::Bytes(data[kept_from..].to_vec()));
                let frame = frame?;
                if frame.components.iter().any(|c| quant[c.tq as usize].is_none()) {
                    return None;
                }
                return Some(Jpeg { frame, parts, blocks, quant, tables, scan_tables });
            }
            _ => {}
        }
        let len = be16(data, at + 2)? as usize;
        if len < 2 {
            return None;
        }
        let body = data.get(at + 4..at + 2 + len)?;
        match marker {
            0xC0 | 0xC1 => {
                // Baseline and extended sequential Huffman.
                if *body.first()? != 8 {
                    return None;
                }
                let height = be16(body, 1)?;
                let width = be16(body, 3)?;
                let n = *body.get(5)? as usize;
                if height == 0 || width == 0 || n == 0 || n > 4 || body.len() != 6 + 3 * n {
                    return None;
                }
                let mut components = Vec::with_capacity(n);
                for c in 0..n {
                    let (id, hv, tq) = (body[6 + 3 * c], body[7 + 3 * c], body[8 + 3 * c]);
                    let (h, v) = (hv >> 4, hv & 15);
                    if !(1..=4).contains(&h) || !(1..=4).contains(&v) || tq > 3 {
                        return None;
                    }
                    components.push(Component { id, h, v, tq, bw: 0, bh: 0, cw: 0, ch: 0 });
                }
                let hmax = components.iter().map(|c| c.h).max()?;
                let vmax = components.iter().map(|c| c.v).max()?;
                let mcux = (width as usize + 8 * hmax as usize - 1) / (8 * hmax as usize);
                let mcuy = (height as usize + 8 * vmax as usize - 1) / (8 * vmax as usize);
                for c in &mut components {
                    c.bw = mcux * c.h as usize;
                    c.bh = mcuy * c.v as usize;
                    let cwidth = (width as usize * c.h as usize + hmax as usize - 1) / hmax as usize;
                    let cheight = (height as usize * c.v as usize + vmax as usize - 1) / vmax as usize;
                    c.cw = (cwidth + 7) / 8;
                    c.ch = (cheight + 7) / 8;
                    blocks.push(vec![[0i16; 64]; c.bw * c.bh]);
                }
                frame = Some(Frame { width, height, components, hmax, vmax, mcux, mcuy });
            }
            0xC2..=0xCF if marker != 0xC4 && marker != 0xC8 && marker != 0xCC => return None,
            0xC4 => {
                let mut p = 0usize;
                while p < body.len() {
                    let (tc, th) = (body[p] >> 4, body[p] & 15);
                    if tc > 1 || th > 3 {
                        return None;
                    }
                    let counts: [u8; 16] = body.get(p + 1..p + 17)?.try_into().unwrap();
                    let n: usize = counts.iter().map(|&c| c as usize).sum();
                    let symbols = body.get(p + 17..p + 17 + n)?.to_vec();
                    tables[tc as usize][th as usize] = Some(Huffman::new(counts, symbols)?);
                    p += 17 + n;
                }
            }
            0xDB => {
                let mut p = 0usize;
                while p < body.len() {
                    let (pq, tq) = (body[p] >> 4, body[p] & 15);
                    if tq > 3 || pq > 1 {
                        return None;
                    }
                    let mut t = [0u16; 64];
                    for i in 0..64 {
                        t[i] = if pq == 0 { *body.get(p + 1 + i)? as u16 } else { be16(body, p + 1 + 2 * i)? };
                        if t[i] == 0 {
                            return None;
                        }
                    }
                    quant[tq as usize] = Some(t);
                    p += 1 + 64 * (1 + pq as usize);
                }
            }
            0xDD => {
                restart = be16(body, 0)?;
            }
            0xDA => {
                let f = frame.as_ref()?;
                let ns = *body.first()? as usize;
                if ns == 0 || ns > 4 || body.len() != 4 + 2 * ns {
                    return None;
                }
                let mut components = Vec::with_capacity(ns);
                for s in 0..ns {
                    let (cs, t) = (body[1 + 2 * s], body[2 + 2 * s]);
                    let ci = f.components.iter().position(|c| c.id == cs)?;
                    let (td, ta) = (t >> 4, t & 15);
                    if td > 3 || ta > 3 || tables[0][td as usize].is_none() || tables[1][ta as usize].is_none() {
                        return None;
                    }
                    components.push((ci, td, ta));
                }
                let (ss, se, ahal) = (body[1 + 2 * ns], body[2 + 2 * ns], body[3 + 2 * ns]);
                if ss != 0 || se != 63 || ahal != 0 {
                    return None;
                }
                // The header (with the SOS segment) kept; the data decoded.
                let data_at = at + 2 + len;
                parts.push(Part::Bytes(data[kept_from..data_at].to_vec()));
                let mut scan = Scan { components, restart, odd_pads: Vec::new() };
                let end = match pads.as_deref_mut() {
                    Some(it) => {
                        scan.odd_pads = it.next()?;
                        data_at
                    }
                    None => decode_scan(data, data_at, f, &tables, &mut scan, &mut blocks)?,
                };
                scan_tables.push(tables.clone());
                parts.push(Part::Scan(scan));
                kept_from = end;
                at = end;
                continue;
            }
            0xDC => return None, // DNL
            _ => {}
        }
        at += 2 + len;
    }
}

/// The blocks of every component (a pointer and a count each),
/// written by bands that own disjoint MCUs.
struct Blocks(Vec<(*mut [i16; 64], usize)>);
// SAFETY: every band writes only the blocks of its own MCUs, and a
// block belongs to one MCU of a scan, so no two bands touch the same
// block; nothing reads the blocks until every band is done.
unsafe impl Sync for Blocks {}

impl Blocks {
    fn of(blocks: &mut [Vec<[i16; 64]>]) -> Blocks {
        Blocks(blocks.iter_mut().map(|b| (b.as_mut_ptr(), b.len())).collect())
    }

    #[inline]
    #[allow(clippy::mut_from_ref)]
    fn get(&self, ci: usize, i: usize) -> &mut [i16; 64] {
        let (p, n) = self.0[ci];
        assert!(i < n);
        // SAFETY: see above; `i` is within the component's blocks.
        unsafe { &mut *p.add(i) }
    }
}

/// MCUs `mcu0..mcu1` of a scan decoded from `at` (the byte after the
/// restart marker before them, or the scan's start): the odd pad
/// bits met, and the position of the marker after the band.
fn decode_band(data: &[u8], at: usize, f: &Frame, tables: &[Vec<Option<Huffman>>], scan: &Scan, l: &Layout, mcu0: usize, mcu1: usize, blocks: &Blocks) -> Option<(Vec<(u32, u8)>, usize)> {
    let mut bits = Bits::new(data, at);
    let restart = scan.restart as usize;
    let mut preds = [0i16; 4];
    let mut markers = if restart > 0 { (mcu0 / restart) as u32 } else { 0 };
    let mut odd = Vec::new();
    for mcu in mcu0..mcu1 {
        if restart > 0 && mcu > mcu0 && mcu % restart == 0 {
            // A restart marker: pad, then RSTn.
            let (pad, pad_bits, pos) = bits.align();
            if pad_bits > 0 && pad != (1 << pad_bits) - 1 {
                odd.push((markers, pad));
            }
            if data.get(pos)? != &0xFF || data.get(pos + 1)? != &(0xD0 + (markers % 8) as u8) {
                return None;
            }
            markers += 1;
            bits = Bits::new(data, pos + 2);
            preds = [0; 4];
        }
        for (k, &(ci, td, ta)) in scan.components.iter().enumerate() {
            let c = &f.components[ci];
            let dc = tables[0][td as usize].as_ref()?;
            let ac = tables[1][ta as usize].as_ref()?;
            let (nh, nv) = l.shape(c);
            for by in 0..nv {
                for bx in 0..nh {
                    let block = blocks.get(ci, l.block(c, mcu, bx, by));
                    let s = bits.decode(dc)? as u32;
                    if s > 11 {
                        return None;
                    }
                    let diff = extend(bits.get(s), s);
                    preds[k] = preds[k].wrapping_add(diff);
                    block[0] = preds[k];
                    let mut i = 1usize;
                    while i < 64 {
                        let rs = bits.decode(ac)?;
                        let (r, s) = ((rs >> 4) as usize, (rs & 15) as u32);
                        if s == 0 {
                            if r == 15 {
                                i += 16;
                                continue;
                            }
                            break;
                        }
                        i += r;
                        if i > 63 || s > 10 {
                            return None;
                        }
                        block[i] = extend(bits.get(s), s);
                        i += 1;
                    }
                }
            }
        }
    }
    let (pad, pad_bits, pos) = bits.align();
    if pad_bits > 0 && pad != (1 << pad_bits) - 1 {
        odd.push((markers, pad));
    }
    if data.get(pos)? != &0xFF {
        return None;
    }
    Some((odd, pos))
}

/// A baseline scan's data from `at`: the blocks filled; where the data
/// ends (the marker after it). With restart intervals the data is cut
/// at its markers into bands decoded on every core.
fn decode_scan(data: &[u8], at: usize, f: &Frame, tables: &[Vec<Option<Huffman>>], scan: &mut Scan, blocks: &mut [Vec<[i16; 64]>]) -> Option<usize> {
    let l = Layout::of(f, scan);
    let restart = scan.restart as usize;
    let bands = if restart > 0 { l.bands(restart) } else { vec![(0, l.total)] };
    let shared = Blocks::of(blocks);
    if bands.len() == 1 {
        let (odd, end) = decode_band(data, at, f, tables, scan, &l, 0, l.total, &shared)?;
        scan.odd_pads = odd;
        return Some(end);
    }
    // Where each band begins: after the restart marker before its
    // first MCU. Markers are the only 0xFF not followed by 0x00.
    let mut starts = vec![at];
    let mut pos = at;
    let mut wanted = bands.iter().skip(1).map(|&(m, _)| m / restart);
    let mut next = wanted.next();
    let mut seen = 0usize;
    while let Some(n) = next {
        let off = data[pos..].iter().position(|&b| b == 0xFF)?;
        pos += off;
        match data.get(pos + 1)? {
            0x00 => pos += 2,
            0xD0..=0xD7 => {
                seen += 1;
                pos += 2;
                if seen == n {
                    starts.push(pos);
                    next = wanted.next();
                }
            }
            _ => return None,
        }
    }
    let results = crate::reflate::each(bands.len(), |i| decode_band(data, starts[i], f, tables, scan, &l, bands[i].0, bands[i].1, &shared));
    let mut end = at;
    for r in results {
        let (odd, pos) = r?;
        scan.odd_pads.extend(odd);
        end = pos;
    }
    Some(end)
}

// ---------------------------------------------------------------- writing

struct BitOut {
    out: Vec<u8>,
    acc: u64,
    n: u32,
}

impl BitOut {
    fn new(out: Vec<u8>) -> Self {
        BitOut { out, acc: 0, n: 0 }
    }

    #[inline]
    fn put(&mut self, v: u32, k: u32) {
        self.acc = (self.acc << k) | (v as u64 & ((1u64 << k) - 1));
        self.n += k;
        if self.n >= 32 {
            self.flush();
        }
    }

    /// Whole bytes out of the accumulator, 0xFF stuffed.
    #[inline]
    fn flush(&mut self) {
        while self.n >= 8 {
            let b = (self.acc >> (self.n - 8)) as u8;
            self.out.push(b);
            if b == 0xFF {
                self.out.push(0x00);
            }
            self.n -= 8;
        }
    }

    /// Pad to the byte with `pad` bits (all ones unless recorded).
    fn pad(&mut self, pad: Option<u8>) {
        self.flush();
        if self.n > 0 {
            let k = 8 - self.n;
            let v = pad.map_or((1u32 << k) - 1, |p| p as u32);
            self.put(v, k);
            self.flush();
        }
    }
}

#[inline]
fn category(v: i16) -> (u32, u32) {
    let a = v.unsigned_abs() as u32;
    let s = 32 - a.leading_zeros();
    let bits = if v < 0 { (v as i32 - 1) as u32 & ((1 << s) - 1) } else { a };
    (s, bits)
}

/// What a band of MCUs encodes to: runs of bits as they are (whole
/// 32-bit words, then a tail), and restart markers, which the join
/// stuffs and places on the byte.
enum Piece {
    Bits(Vec<u32>, u64, u32),
    Marker(u8),
}

/// Bits most significant first into 32-bit words, not stuffed.
struct RawBits {
    words: Vec<u32>,
    acc: u64,
    n: u32,
    total: u64,
    pieces: Vec<Piece>,
}

impl RawBits {
    fn new() -> Self {
        RawBits { words: Vec::new(), acc: 0, n: 0, total: 0, pieces: Vec::new() }
    }

    #[inline]
    fn put(&mut self, v: u32, k: u32) {
        self.acc = (self.acc << k) | (v as u64 & ((1u64 << k) - 1));
        self.n += k;
        self.total += k as u64;
        if self.n >= 32 {
            self.n -= 32;
            self.words.push((self.acc >> self.n) as u32);
        }
    }

    fn take(&mut self) -> Piece {
        let tail = self.acc & ((1u64 << self.n) - 1);
        let p = Piece::Bits(std::mem::take(&mut self.words), tail, self.n);
        self.acc = 0;
        self.n = 0;
        p
    }

    fn finish(mut self) -> Vec<Piece> {
        let bits = self.take();
        self.pieces.push(bits);
        self.pieces
    }

    /// Pad to the byte with `pad` bits (all ones unless recorded);
    /// the band starts on a byte, so its own count tells.
    fn pad(&mut self, pad: Option<u8>) {
        let r = (self.total % 8) as u32;
        if r > 0 {
            let k = 8 - r;
            let v = pad.map_or((1u32 << k) - 1, |p| p as u32);
            self.put(v, k);
        }
    }
}

/// Where a band's bits go: straight into the file, or into pieces for
/// a later join.
trait Sink {
    fn put(&mut self, v: u32, k: u32);
    fn pad(&mut self, pad: Option<u8>);
    fn marker(&mut self, m: u8);
}

impl Sink for BitOut {
    #[inline]
    fn put(&mut self, v: u32, k: u32) {
        BitOut::put(self, v, k)
    }
    fn pad(&mut self, pad: Option<u8>) {
        BitOut::pad(self, pad)
    }
    fn marker(&mut self, m: u8) {
        self.flush();
        debug_assert!(self.n == 0);
        self.out.push(0xFF);
        self.out.push(m);
    }
}

impl Sink for RawBits {
    #[inline]
    fn put(&mut self, v: u32, k: u32) {
        RawBits::put(self, v, k)
    }
    fn pad(&mut self, pad: Option<u8>) {
        RawBits::pad(self, pad)
    }
    fn marker(&mut self, m: u8) {
        let bits = self.take();
        self.pieces.push(bits);
        self.pieces.push(Piece::Marker(m));
    }
}

impl BitOut {
    /// A band's pieces, stuffed, the markers on the byte.
    fn join(&mut self, pieces: Vec<Piece>) {
        for p in pieces {
            match p {
                Piece::Bits(words, tail, n) => {
                    for w in words {
                        self.put(w, 32);
                    }
                    self.put(tail as u32, n);
                }
                Piece::Marker(m) => self.marker(m),
            }
        }
    }
}

/// The layout of a scan: which MCUs, and where each component's
/// blocks sit.
struct Layout {
    single: bool,
    mcux: usize,
    total: usize,
}

impl Layout {
    fn of(f: &Frame, scan: &Scan) -> Layout {
        let single = scan.components.len() == 1;
        let (mcux, mcuy) = if single {
            let c = &f.components[scan.components[0].0];
            (c.cw, c.ch)
        } else {
            (f.mcux, f.mcuy)
        };
        Layout { single, mcux, total: mcux * mcuy }
    }

    /// The block index of block (bx, by) of component `c` in MCU `mcu`.
    #[inline]
    fn block(&self, c: &Component, mcu: usize, bx: usize, by: usize) -> usize {
        let (mx, my) = (mcu % self.mcux, mcu / self.mcux);
        let (x, y) = if self.single { (mx, my) } else { (mx * c.h as usize + bx, my * c.v as usize + by) };
        y * c.bw + x
    }

    #[inline]
    fn shape(&self, c: &Component) -> (usize, usize) {
        if self.single { (1, 1) } else { (c.h as usize, c.v as usize) }
    }

    /// Where the bands of a scan begin and end: about one per core,
    /// at restart intervals when there are any (each band then starts
    /// on a byte with fresh predictors), anywhere otherwise.
    fn bands(&self, restart: usize) -> Vec<(usize, usize)> {
        let n = crate::threads().clamp(1, 16).min(self.total / 2048).max(1);
        let mut cuts: Vec<usize> = (0..=n).map(|i| self.total * i / n).map(|m| if restart > 0 { m / restart * restart } else { m }).collect();
        cuts.dedup();
        cuts.windows(2).map(|w| (w[0], w[1])).collect()
    }
}

/// MCUs `mcu0..mcu1` of a scan encoded into `w`: the bits, and the
/// restart markers between the intervals inside the band and at its
/// end (when the scan goes on). Without restart intervals the DC
/// predictors continue from the MCU before the band.
fn encode_band<S: Sink>(w: &mut S, f: &Frame, tables: &[Vec<Option<Huffman>>], scan: &Scan, blocks: &[Vec<[i16; 64]>], l: &Layout, mcu0: usize, mcu1: usize) -> Option<()> {
    let restart = scan.restart as usize;
    let mut preds = [0i16; 4];
    if restart == 0 && mcu0 > 0 {
        for (k, &(ci, _, _)) in scan.components.iter().enumerate() {
            let c = &f.components[ci];
            let (nh, nv) = l.shape(c);
            preds[k] = blocks[ci][l.block(c, mcu0 - 1, nh - 1, nv - 1)][0];
        }
    }
    let mut markers = if restart > 0 { (mcu0 / restart) as u32 } else { 0 };
    let first = markers;
    let mut odd = scan.odd_pads.iter().skip_while(move |(m, _)| *m < first).peekable();
    for mcu in mcu0..mcu1 {
        if restart > 0 && mcu > mcu0 && mcu % restart == 0 {
            let pad = odd.next_if(|(m, _)| *m == markers).map(|&(_, p)| p);
            w.pad(pad);
            w.marker(0xD0 + (markers % 8) as u8);
            markers += 1;
            preds = [0; 4];
        }
        for (k, &(ci, td, ta)) in scan.components.iter().enumerate() {
            let c = &f.components[ci];
            let dc = tables[0][td as usize].as_ref()?;
            let ac = tables[1][ta as usize].as_ref()?;
            let (nh, nv) = l.shape(c);
            for by in 0..nv {
                for bx in 0..nh {
                    let block = &blocks[ci][l.block(c, mcu, bx, by)];
                    let diff = block[0].wrapping_sub(preds[k]);
                    preds[k] = block[0];
                    let (s, v) = category(diff);
                    let (code, len) = dc.codes[s as usize];
                    if len == 0 {
                        return None;
                    }
                    w.put((code as u32) << s | v, len as u32 + s);
                    let mut run = 0u32;
                    let last = (1..64).rev().find(|&i| block[i] != 0).unwrap_or(0);
                    for i in 1..=last {
                        if block[i] == 0 {
                            run += 1;
                            continue;
                        }
                        while run >= 16 {
                            let (code, len) = ac.codes[0xF0];
                            if len == 0 {
                                return None;
                            }
                            w.put(code as u32, len as u32);
                            run -= 16;
                        }
                        let (s, v) = category(block[i]);
                        let (code, len) = ac.codes[((run << 4) | s) as usize];
                        if len == 0 {
                            return None;
                        }
                        w.put((code as u32) << s | v, len as u32 + s);
                        run = 0;
                    }
                    if last < 63 {
                        let (code, len) = ac.codes[0x00];
                        if len == 0 {
                            return None;
                        }
                        w.put(code as u32, len as u32);
                    }
                }
            }
        }
    }
    if restart > 0 && mcu1 < l.total {
        let pad = odd.next_if(|(m, _)| *m == markers).map(|&(_, p)| p);
        w.pad(pad);
        w.marker(0xD0 + (markers % 8) as u8);
    }
    Some(())
}

/// A scan's data: its bands encoded on every core, joined, the final
/// pad.
fn encode_scan(w: &mut BitOut, f: &Frame, tables: &[Vec<Option<Huffman>>], scan: &Scan, blocks: &[Vec<[i16; 64]>]) -> Option<()> {
    let l = Layout::of(f, scan);
    let bands = l.bands(scan.restart as usize);
    if bands.len() == 1 {
        encode_band(w, f, tables, scan, blocks, &l, 0, l.total)?;
    } else {
        let pieces = crate::reflate::each(bands.len(), |i| {
            let mut raw = RawBits::new();
            encode_band(&mut raw, f, tables, scan, blocks, &l, bands[i].0, bands[i].1).map(|_| raw.finish())
        });
        for p in pieces {
            w.join(p?);
        }
    }
    let markers = if scan.restart > 0 { ((l.total - 1) / scan.restart as usize) as u32 } else { 0 };
    let pad = scan.odd_pads.iter().find(|(m, _)| *m == markers).map(|&(_, p)| p);
    w.pad(pad);
    Some(())
}

/// Glyd's own stream of a JPEG, verified to give the JPEG back:
///
///   "GJPG", the kept bytes (a flag: 0 as they are, 1 compressed with
///   the max level; varint length; the file with every scan's data cut
///   out), the scans (varint count) each with its odd pad bits (varint
///   count, then varint marker index and the bits), the model's
///   streams (varint count, each with a varint length), one per
///   stripe of block rows.
pub fn pack(input: &[u8]) -> Option<Vec<u8>> {
    let j = parse(input)?;
    let mut out = Vec::with_capacity(input.len() * 4 / 5);
    out.extend_from_slice(MAGIC);
    let mut kept = Vec::new();
    for p in &j.parts {
        if let Part::Bytes(b) = p {
            kept.extend_from_slice(b);
        }
    }
    let mut packed = Vec::new();
    crate::compress_max_plain(&kept, &mut packed);
    if packed.len() < kept.len() {
        out.push(1);
        put_varint(&mut out, packed.len() as u64);
        out.extend_from_slice(&packed);
    } else {
        out.push(0);
        put_varint(&mut out, kept.len() as u64);
        out.extend_from_slice(&kept);
    }
    put_varint(&mut out, j.parts.iter().filter(|p| matches!(p, Part::Scan(_))).count() as u64);
    for p in &j.parts {
        if let Part::Scan(s) = p {
            put_varint(&mut out, s.odd_pads.len() as u64);
            for &(marker, bits) in &s.odd_pads {
                put_varint(&mut out, marker as u64);
                out.push(bits);
            }
        }
    }
    let streams = model::encode(&j, stripes_for(&j, input.len()));
    put_varint(&mut out, streams.len() as u64);
    for s in &streams {
        put_varint(&mut out, s.len() as u64);
    }
    for s in &streams {
        out.extend_from_slice(s);
    }
    if unpack(&out)? != input {
        return None;
    }
    Some(out)
}

pub(crate) const MAGIC: &[u8; 4] = b"GJPG";

/// How many stripes a JPEG is coded in: four, so four cores share the
/// work, eight from 10 MB up (a stripe costs a few KB whatever the
/// size, so it pays on a big file), each at least 64 block rows tall;
/// the same on every machine, so the bytes are.
fn stripes_for(j: &Jpeg, len: usize) -> usize {
    let rows = j.frame.components[0].bh;
    let most = if len >= 10 << 20 { 8 } else { 4 };
    (rows / 64).clamp(1, most)
}

/// The JPEG back from `pack`'s stream.
pub fn unpack(stream: &[u8]) -> Option<Vec<u8>> {
    if stream.len() < 5 || &stream[..4] != MAGIC {
        return None;
    }
    let mut pos = 5usize;
    let kept = get_varint(stream, &mut pos).ok()? as usize;
    let headers = stream.get(pos..pos + kept)?;
    pos += kept;
    let headers = match stream[4] {
        0 => std::borrow::Cow::Borrowed(headers),
        1 => std::borrow::Cow::Owned(crate::decompress(headers).ok()?),
        _ => return None,
    };
    let headers = &headers[..];
    let scans = get_varint(stream, &mut pos).ok()? as usize;
    let mut pads = Vec::with_capacity(scans.min(1024));
    for _ in 0..scans {
        let n = get_varint(stream, &mut pos).ok()? as usize;
        let mut list = Vec::with_capacity(n.min(1024));
        for _ in 0..n {
            let marker = get_varint(stream, &mut pos).ok()? as u32;
            list.push((marker, *stream.get(pos)?));
            pos += 1;
        }
        pads.push(list);
    }
    let mut j = parse_with(headers, Some(&mut pads.into_iter()))?;
    let n = get_varint(stream, &mut pos).ok()? as usize;
    let mut lens = Vec::with_capacity(n.min(1024));
    for _ in 0..n {
        lens.push(get_varint(stream, &mut pos).ok()? as usize);
    }
    let mut streams = Vec::with_capacity(n.min(1024));
    for len in lens {
        streams.push(stream.get(pos..pos + len)?);
        pos += len;
    }
    j.blocks = model::decode(&streams, &j.frame, &j.quant)?;
    write(&j)
}

/// The JPEG written back from its parts and coefficients.
pub fn write(j: &Jpeg) -> Option<Vec<u8>> {
    let mut w = BitOut::new(Vec::new());
    let mut scan_i = 0usize;
    for part in &j.parts {
        match part {
            Part::Bytes(b) => w.out.extend_from_slice(b),
            Part::Scan(s) => {
                encode_scan(&mut w, &j.frame, &j.scan_tables[scan_i], s, &j.blocks)?;
                scan_i += 1;
            }
        }
    }
    Some(w.out)
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    pub fn fixture(name: &str) -> Vec<u8> {
        std::fs::read(format!("{}/tests/data/jpeg/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap()
    }

    #[test]
    fn baseline_jpegs_come_back_bit_for_bit() {
        for name in ["q75-420.jpg", "q90-444.jpg", "q60-422-rst.jpg", "q80-gray.jpg", "q95-opt.jpg"] {
            let data = fixture(name);
            let j = parse(&data).unwrap_or_else(|| panic!("{name}: parses"));
            let nonzero: usize = j.blocks.iter().flatten().map(|b| b.iter().filter(|&&c| c != 0).count()).sum();
            assert!(nonzero > 1000, "{name}: coefficients");
            let back = write(&j).unwrap_or_else(|| panic!("{name}: writes"));
            assert!(back == data, "{name}: bit for bit ({} against {} bytes, first difference {:?})", back.len(), data.len(), back.iter().zip(&data).position(|(a, b)| a != b));
        }
        assert!(parse(&fixture("q75-420-prog.jpg")).is_none(), "progressive stays closed for now");
        let mut tail = fixture("q75-420.jpg");
        tail.extend_from_slice(b"bytes after the end");
        assert!(write(&parse(&tail).unwrap()).unwrap() == tail);
    }

    /// `GLYD_JPEG_DIR=dir cargo test --release jpg::tests::photos -- --ignored --nocapture`
    #[test]
    #[ignore = "every .jpg in a directory named by GLYD_JPEG_DIR"]
    fn photos() {
        let Ok(dir) = std::env::var("GLYD_JPEG_DIR") else { return };
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().map_or(true, |e| e != "jpg") {
                continue;
            }
            let data = std::fs::read(&path).unwrap();
            let t = std::time::Instant::now();
            let Some(j) = parse(&data) else {
                eprintln!("{}: not taken apart", path.display());
                continue;
            };
            let parse_s = t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            let back = write(&j).unwrap();
            let write_s = t.elapsed().as_secs_f64();
            assert!(back == data);
            let t = std::time::Instant::now();
            #[cfg(feature = "jpg-stats")]
            for c in model::COST.iter() {
                c.store(0, std::sync::atomic::Ordering::Relaxed);
            }
            let streams = model::encode(&j, 1);
            let stream: Vec<u8> = streams.concat();
            let refs: Vec<&[u8]> = streams.iter().map(|s| &s[..]).collect();
            #[cfg(feature = "jpg-stats")]
            {
                let cost: Vec<String> = ["count", "zero", "exp", "mant", "sign", "e0", "eexp", "emant", "esign", "dc0", "dcexp", "dcmant", "dcsign"].iter().zip(model::COST.iter().zip(model::DECISIONS.iter())).map(|(n, (c, d))| format!("{n} {:.0} KB/{:.1} M", c.load(std::sync::atomic::Ordering::Relaxed) as f64 / 8000.0 / 1000.0, d.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e6)).collect();
                eprintln!("  {}", cost.join(", "));
            }
            let model_s = t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            assert!(model::decode(&refs, &j.frame, &j.quant).unwrap() == j.blocks);
            let decode_s = t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            let lepton = {
                #[cfg(feature = "jpeg")]
                {
                    lepton_jpeg::encode_lepton_verify(&data, &lepton_jpeg::EnabledFeatures::compat_lepton_vector_write(), &lepton_jpeg::SingleThreadPool {}).ok().map(|(l, _)| l.len())
                }
                #[cfg(not(feature = "jpeg"))]
                None::<usize>
            };
            let lepton_s = t.elapsed().as_secs_f64();
            #[cfg(feature = "jpeg")]
            {
                // Lepton at one partition (its smallest) and on its
                // own thread pool (its fastest), each timed through
                // the encode with verify and the decode alone.
                let mut one = lepton_jpeg::EnabledFeatures::compat_lepton_vector_write();
                one.max_partitions = 1;
                let t = std::time::Instant::now();
                let l1 = lepton_jpeg::encode_lepton_verify(&data, &one, &lepton_jpeg::SingleThreadPool {}).ok().map(|(l, _)| l.len());
                let l1_s = t.elapsed().as_secs_f64();
                let t = std::time::Instant::now();
                let l8 = lepton_jpeg::encode_lepton_verify(&data, &lepton_jpeg::EnabledFeatures::compat_lepton_vector_write(), &lepton_jpeg::DEFAULT_THREAD_POOL).ok().map(|(l, _)| l);
                let l8_s = t.elapsed().as_secs_f64();
                let l8_dec = l8.as_ref().map(|l| {
                    let t = std::time::Instant::now();
                    let mut out = Vec::new();
                    lepton_jpeg::decode_lepton(&mut std::io::Cursor::new(l), &mut out, &lepton_jpeg::EnabledFeatures::compat_lepton_vector_read(), &lepton_jpeg::DEFAULT_THREAD_POOL).unwrap();
                    assert!(out == data);
                    t.elapsed().as_secs_f64()
                });
                eprintln!("  lepton 1 partition {:?} B in {l1_s:.2} s; 8 partitions on its pool: {:?} B in {l8_s:.2} s, decode {:?} s", l1, l8.map(|l| l.len()), l8_dec.map(|s| (s * 100.0).round() / 100.0));
            }
            let t = std::time::Instant::now();
            let packed = pack(&data).unwrap();
            let pack_s = t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            assert!(unpack(&packed).unwrap() == data);
            let unpack_s = t.elapsed().as_secs_f64();
            {
                // The read's parts, one stripe set as packed.
                let mut pos = 5usize;
                let kept = get_varint(&packed, &mut pos).unwrap() as usize;
                let headers = &packed[pos..pos + kept];
                pos += kept;
                let t = std::time::Instant::now();
                let headers = if packed[4] == 1 { crate::decompress(headers).unwrap() } else { headers.to_vec() };
                let hdr_s = t.elapsed().as_secs_f64();
                let scans = get_varint(&packed, &mut pos).unwrap() as usize;
                let mut pads = Vec::new();
                for _ in 0..scans {
                    let n = get_varint(&packed, &mut pos).unwrap() as usize;
                    let mut list = Vec::new();
                    for _ in 0..n {
                        let marker = get_varint(&packed, &mut pos).unwrap() as u32;
                        list.push((marker, packed[pos]));
                        pos += 1;
                    }
                    pads.push(list);
                }
                let mut jj = parse_with(&headers, Some(&mut pads.into_iter())).unwrap();
                let n = get_varint(&packed, &mut pos).unwrap() as usize;
                let mut lens = Vec::new();
                for _ in 0..n {
                    lens.push(get_varint(&packed, &mut pos).unwrap() as usize);
                }
                let mut streams: Vec<&[u8]> = Vec::new();
                for len in lens {
                    streams.push(&packed[pos..pos + len]);
                    pos += len;
                }
                let t = std::time::Instant::now();
                jj.blocks = model::decode(&streams, &jj.frame, &jj.quant).unwrap();
                let model_s = t.elapsed().as_secs_f64();
                let t = std::time::Instant::now();
                let out = write(&jj).unwrap();
                let write_s = t.elapsed().as_secs_f64();
                assert!(out == data);
                eprintln!("  read parts: headers {hdr_s:.3} s, model ({} streams) {model_s:.3} s, write {write_s:.3} s", n);
            }
            eprintln!("  pack {} B in {pack_s:.2} s ({:.0} MB/s), unpack {unpack_s:.2} s ({:.0} MB/s); lepton in {lepton_s:.2} s ({:.0} MB/s)", packed.len(), data.len() as f64 / 1e6 / pack_s, data.len() as f64 / 1e6 / unpack_s, data.len() as f64 / 1e6 / lepton_s);
            eprintln!("{}: {} B, {} blocks, parse {parse_s:.3} s, write {write_s:.3} s; model {} B ({:.1}%) in {model_s:.2} s, back in {decode_s:.2} s; lepton {:?} ({:.1}%)", path.display(), data.len(), j.blocks.iter().map(|b| b.len()).sum::<usize>(), stream.len(), 100.0 * stream.len() as f64 / data.len() as f64, lepton, lepton.map_or(0.0, |l| 100.0 * l as f64 / data.len() as f64));
        }
    }
}
