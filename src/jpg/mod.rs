//! JPEGs taken apart into their coefficients and put back bit for bit,
//! with nothing outside this crate (`docs/design/jpeg-recoding.md`).
//!
//! This is the first part: a parser that keeps every marker segment as
//! it is and decodes each baseline (sequential Huffman, 8-bit) scan to
//! its blocks' coefficients in zigzag order, and a writer that encodes
//! them back under the same tables with the same restart markers and
//! padding. Progressive, arithmetic-coded, 12-bit, lossless and
//! hierarchical JPEGs are not taken apart (`parse` returns `None`).

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
}

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
        (k == symbols.len()).then_some(Huffman { counts, symbols, codes, mincode, maxcode, valptr })
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

    fn decode(&mut self, h: &Huffman) -> Option<u8> {
        let mut code = 0i32;
        for len in 1..=16usize {
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
    if data.len() < 4 || data[0] != 0xFF || data[1] != 0xD8 {
        return None;
    }
    let mut parts = Vec::new();
    let mut frame: Option<Frame> = None;
    let mut blocks: Vec<Vec<[i16; 64]>> = Vec::new();
    let mut tables: Vec<Vec<Option<Huffman>>> = vec![vec![None; 4], vec![None; 4]];
    let mut scan_tables = Vec::new();
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
                return Some(Jpeg { frame, parts, blocks, tables, scan_tables });
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
                let end = decode_scan(data, data_at, f, &tables, &mut scan, &mut blocks)?;
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

/// A baseline scan's data from `at`: the blocks filled; where the data
/// ends (the marker after it).
fn decode_scan(data: &[u8], at: usize, f: &Frame, tables: &[Vec<Option<Huffman>>], scan: &mut Scan, blocks: &mut [Vec<[i16; 64]>]) -> Option<usize> {
    let mut bits = Bits::new(data, at);
    let single = scan.components.len() == 1;
    let (mcux, mcuy) = if single {
        let c = &f.components[scan.components[0].0];
        (c.cw, c.ch)
    } else {
        (f.mcux, f.mcuy)
    };
    let total = mcux * mcuy;
    let mut preds = [0i16; 4];
    let mut markers = 0u32;
    let mut mcu = 0usize;
    while mcu < total {
        if scan.restart > 0 && mcu > 0 && mcu % scan.restart as usize == 0 {
            // A restart marker: pad, then RSTn.
            let (pad, pad_bits, pos) = bits.align();
            if pad_bits > 0 && pad != (1 << pad_bits) - 1 {
                scan.odd_pads.push((markers, pad));
            }
            if data.get(pos)? != &0xFF || data.get(pos + 1)? != &(0xD0 + (markers % 8) as u8) {
                return None;
            }
            markers += 1;
            bits = Bits::new(data, pos + 2);
            preds = [0; 4];
        }
        let (mx, my) = (mcu % mcux, mcu / mcux);
        for (k, &(ci, td, ta)) in scan.components.iter().enumerate() {
            let c = &f.components[ci];
            let dc = tables[0][td as usize].as_ref()?;
            let ac = tables[1][ta as usize].as_ref()?;
            let (nh, nv) = if single { (1, 1) } else { (c.h as usize, c.v as usize) };
            for by in 0..nv {
                for bx in 0..nh {
                    let (x, y) = if single { (mx, my) } else { (mx * nh + bx, my * nv + by) };
                    let block = &mut blocks[ci][y * c.bw + x];
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
        mcu += 1;
    }
    let (pad, pad_bits, pos) = bits.align();
    if pad_bits > 0 && pad != (1 << pad_bits) - 1 {
        scan.odd_pads.push((markers, pad));
    }
    if data.get(pos)? != &0xFF {
        return None;
    }
    Some(pos)
}

// ---------------------------------------------------------------- writing

struct BitOut {
    out: Vec<u8>,
    acc: u32,
    n: u32,
}

impl BitOut {
    fn new(out: Vec<u8>) -> Self {
        BitOut { out, acc: 0, n: 0 }
    }

    #[inline]
    fn put(&mut self, v: u32, k: u32) {
        if k == 0 {
            return;
        }
        self.acc = (self.acc << k) | (v & ((1u32 << k) - 1));
        self.n += k;
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
        if self.n > 0 {
            let k = 8 - self.n;
            let v = pad.map_or((1u32 << k) - 1, |p| p as u32);
            self.put(v, k);
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

fn encode_scan(w: &mut BitOut, f: &Frame, tables: &[Vec<Option<Huffman>>], scan: &Scan, blocks: &[Vec<[i16; 64]>]) -> Option<()> {
    let single = scan.components.len() == 1;
    let (mcux, mcuy) = if single {
        let c = &f.components[scan.components[0].0];
        (c.cw, c.ch)
    } else {
        (f.mcux, f.mcuy)
    };
    let total = mcux * mcuy;
    let mut preds = [0i16; 4];
    let mut markers = 0u32;
    let mut odd = scan.odd_pads.iter().peekable();
    for mcu in 0..total {
        if scan.restart > 0 && mcu > 0 && mcu % scan.restart as usize == 0 {
            let pad = odd.next_if(|(m, _)| *m == markers).map(|&(_, p)| p);
            w.pad(pad);
            w.out.push(0xFF);
            w.out.push(0xD0 + (markers % 8) as u8);
            markers += 1;
            preds = [0; 4];
        }
        let (mx, my) = (mcu % mcux, mcu / mcux);
        for (k, &(ci, td, ta)) in scan.components.iter().enumerate() {
            let c = &f.components[ci];
            let dc = tables[0][td as usize].as_ref()?;
            let ac = tables[1][ta as usize].as_ref()?;
            let (nh, nv) = if single { (1, 1) } else { (c.h as usize, c.v as usize) };
            for by in 0..nv {
                for bx in 0..nh {
                    let (x, y) = if single { (mx, my) } else { (mx * nh + bx, my * nv + by) };
                    let block = &blocks[ci][y * c.bw + x];
                    let diff = block[0].wrapping_sub(preds[k]);
                    preds[k] = block[0];
                    let (s, v) = category(diff);
                    let (code, len) = dc.codes[s as usize];
                    if len == 0 {
                        return None;
                    }
                    w.put(code as u32, len as u32);
                    w.put(v, s);
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
                        w.put(code as u32, len as u32);
                        w.put(v, s);
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
    let pad = odd.next_if(|(m, _)| *m == markers).map(|&(_, p)| p);
    w.pad(pad);
    Some(())
}

/// The JPEG's bytes back.
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
mod tests {
    use super::*;

    fn fixture(name: &str) -> Vec<u8> {
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
            eprintln!("{}: {} B, {} blocks, parse {parse_s:.3} s, write {write_s:.3} s, exact {}", path.display(), data.len(), j.blocks.iter().map(|b| b.len()).sum::<usize>(), back == data);
            assert!(back == data);
        }
    }
}
