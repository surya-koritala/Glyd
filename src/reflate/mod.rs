//! Deflate streams taken apart and put back, bit for bit, with nothing
//! outside this crate (`docs/design/deflate-reconstruction.md`).
//!
//! This is the first part: a parser that turns a stream into its blocks
//! — stored bytes, or tokens under fixed or dynamic Huffman codes, with
//! a dynamic block's header kept as the symbols it was written with —
//! and a writer that emits the same blocks as the same bits. Every
//! block also records the bit it started at, for the pieces to be
//! written on every core later.

mod coder;
pub mod trees;
pub mod zlib;

use crate::record::{get_varint, put_varint};

/// A literal byte, or a reference: `len` 3..=258 back `dist` 1..=32768.
/// `len` 259 is 258 coded the long way (code 284 with 31 extra bits),
/// which the standard allows and zlib never writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Token {
    Lit(u8),
    Ref { len: u16, dist: u16 },
}

/// A dynamic block's header as read: the counts and the code-length
/// code, then the symbols (0–15 a length, 16–18 a run) with their
/// extra bits, which the writer sends back as they were.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Header {
    pub hlit: u16,
    pub hdist: u8,
    pub hclen: u8,
    /// lengths of the code-length code, in the order of the stream
    pub clen: [u8; 19],
    pub symbols: Vec<(u8, u8)>,
    /// the literal/length and distance code lengths the symbols expand to
    pub lit_lens: Vec<u8>,
    pub dist_lens: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Kind {
    Stored(Vec<u8>),
    Fixed(Vec<Token>),
    Dynamic(Header, Vec<Token>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block {
    pub last: bool,
    /// where the block's first bit sits, counted from the stream's start
    pub bit_start: u64,
    pub kind: Kind,
}

/// A stream parsed: its blocks, its plain text, and how many bytes of
/// the input it took (the last byte possibly only in part).
pub struct Stream {
    pub blocks: Vec<Block>,
    pub plain: Vec<u8>,
    pub consumed: usize,
}

const LEN_BASE: [u16; 29] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const LEN_EXTRA: [u8; 29] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DIST_BASE: [u16; 30] = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
const DIST_EXTRA: [u8; 30] = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
const CLEN_ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

fn fixed_lit_lens() -> Vec<u8> {
    let mut v = vec![8u8; 288];
    v[144..256].fill(9);
    v[256..280].fill(7);
    v
}

/// The length code's index (0..29) for each length 3..=258.
static LEN_INDEX: [u8; 259] = {
    let mut t = [0u8; 259];
    let mut len = 3usize;
    while len <= 258 {
        let mut i = 28;
        while LEN_BASE[i] as usize > len {
            i -= 1;
        }
        t[len] = i as u8;
        len += 1;
    }
    t
};

/// The distance code for each distance 1..=32768, by its high bits:
/// index by `dist - 1` below 256, else by `(dist - 1) >> 7` at 256..
static DIST_INDEX: [u8; 512] = {
    let mut t = [0u8; 512];
    let mut d = 1usize;
    while d <= 256 {
        let mut i = 29;
        while DIST_BASE[i] as usize > d {
            i -= 1;
        }
        t[d - 1] = i as u8;
        d += 1;
    }
    let mut k = 2usize;
    while k < 256 {
        let d = (k << 7) + 1;
        let mut i = 29;
        while DIST_BASE[i] as usize > d {
            i -= 1;
        }
        t[256 + k] = i as u8;
        k += 1;
    }
    t
};

/// The length code and extra bits of a reference length.
#[inline]
fn len_code(len: u16) -> (u16, u8, u16) {
    if len == 259 {
        return (284, 5, 31);
    }
    let i = LEN_INDEX[len as usize] as usize;
    (257 + i as u16, LEN_EXTRA[i], len - LEN_BASE[i])
}

#[inline]
fn dist_code(dist: u16) -> (u16, u8, u16) {
    let d = dist as usize - 1;
    let i = if d < 256 { DIST_INDEX[d] } else { DIST_INDEX[256 + (d >> 7)] } as usize;
    (i as u16, DIST_EXTRA[i], dist - DIST_BASE[i])
}

// ---------------------------------------------------------------- reading

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    bits: u64,
    n: u32,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        BitReader { data, pos: 0, bits: 0, n: 0 }
    }

    #[inline]
    fn fill(&mut self) {
        while self.n <= 56 {
            let b = if self.pos < self.data.len() { self.data[self.pos] } else { return };
            self.bits |= (b as u64) << self.n;
            self.n += 8;
            self.pos += 1;
        }
    }

    #[inline]
    fn get(&mut self, k: u32) -> Option<u32> {
        if self.n < k {
            self.fill();
            if self.n < k {
                return None;
            }
        }
        let v = (self.bits & ((1u64 << k) - 1)) as u32;
        self.bits >>= k;
        self.n -= k;
        Some(v)
    }

    /// The position of the next unread bit.
    fn bit_position(&self) -> u64 {
        self.pos as u64 * 8 - self.n as u64
    }

    /// To the next byte boundary; the bytes not yet used go back.
    fn align(&mut self) {
        let drop = self.n % 8;
        self.bits >>= drop;
        self.n -= drop;
        let back = (self.n / 8) as usize;
        self.pos -= back;
        self.bits = 0;
        self.n = 0;
    }
}

/// A canonical Huffman code for decoding, bit by bit (RFC 1951 3.2.2).
struct Decoder {
    count: [u16; 16],
    symbol: Vec<u16>,
}

impl Decoder {
    fn new(lens: &[u8]) -> Option<Decoder> {
        let mut count = [0u16; 16];
        for &l in lens {
            count[l as usize] += 1;
        }
        count[0] = 0;
        let mut left = 1i32;
        for len in 1..16 {
            left <<= 1;
            left -= count[len] as i32;
            if left < 0 {
                return None;
            }
        }
        let mut offs = [0u16; 16];
        for len in 1..15 {
            offs[len + 1] = offs[len] + count[len];
        }
        let mut symbol = vec![0u16; lens.len()];
        for (s, &l) in lens.iter().enumerate() {
            if l != 0 {
                symbol[offs[l as usize] as usize] = s as u16;
                offs[l as usize] += 1;
            }
        }
        Some(Decoder { count, symbol })
    }

    #[inline]
    fn decode(&self, r: &mut BitReader) -> Option<u16> {
        let (mut code, mut first, mut index) = (0i32, 0i32, 0i32);
        for len in 1..16 {
            code |= r.get(1)? as i32;
            let count = self.count[len] as i32;
            if code - count < first {
                return Some(self.symbol[(index + (code - first)) as usize]);
            }
            index += count;
            first += count;
            first <<= 1;
            code <<= 1;
        }
        None
    }
}

fn read_tokens(r: &mut BitReader, lit: &Decoder, dist: &Decoder, plain: &mut Vec<u8>) -> Option<Vec<Token>> {
    let mut tokens = Vec::new();
    loop {
        let sym = lit.decode(r)?;
        if sym < 256 {
            plain.push(sym as u8);
            tokens.push(Token::Lit(sym as u8));
        } else if sym == 256 {
            return Some(tokens);
        } else {
            let i = (sym - 257) as usize;
            if i >= 29 {
                return None;
            }
            let extra = r.get(LEN_EXTRA[i] as u32)? as u16;
            let mut len = LEN_BASE[i] + extra;
            if len == 258 && i == 27 {
                len = 259;
            }
            let d = dist.decode(r)? as usize;
            if d >= 30 {
                return None;
            }
            let dist = DIST_BASE[d] + r.get(DIST_EXTRA[d] as u32)? as u16;
            let n = if len == 259 { 258 } else { len } as usize;
            if dist as usize > plain.len() {
                return None;
            }
            let start = plain.len() - dist as usize;
            for k in 0..n {
                let b = plain[start + k];
                plain.push(b);
            }
            tokens.push(Token::Ref { len, dist });
        }
    }
}

fn read_header(r: &mut BitReader) -> Option<Header> {
    let hlit = r.get(5)? as u16 + 257;
    let hdist = r.get(5)? as u8 + 1;
    let hclen = r.get(4)? as u8 + 4;
    let mut clen = [0u8; 19];
    for i in 0..hclen as usize {
        clen[CLEN_ORDER[i]] = r.get(3)? as u8;
    }
    let cl = Decoder::new(&clen)?;
    let total = hlit as usize + hdist as usize;
    let mut lens = Vec::with_capacity(total);
    let mut symbols = Vec::new();
    while lens.len() < total {
        let sym = cl.decode(r)? as u8;
        match sym {
            0..=15 => {
                lens.push(sym);
                symbols.push((sym, 0));
            }
            16 => {
                let prev = *lens.last()?;
                let extra = r.get(2)? as u8;
                lens.extend(std::iter::repeat(prev).take(3 + extra as usize));
                symbols.push((16, extra));
            }
            17 => {
                let extra = r.get(3)? as u8;
                lens.extend(std::iter::repeat(0).take(3 + extra as usize));
                symbols.push((17, extra));
            }
            _ => {
                let extra = r.get(7)? as u8;
                lens.extend(std::iter::repeat(0).take(11 + extra as usize));
                symbols.push((18, extra));
            }
        }
    }
    if lens.len() != total || lens[256] == 0 {
        return None;
    }
    let dist_lens = lens.split_off(hlit as usize);
    Some(Header { hlit, hdist, hclen, clen, symbols, lit_lens: lens, dist_lens })
}

/// `data` from its first byte as a deflate stream, to its final block.
/// `None` when it is not one (or is cut short).
pub fn parse(data: &[u8]) -> Option<Stream> {
    let mut r = BitReader::new(data);
    let mut blocks = Vec::new();
    let mut plain = Vec::new();
    loop {
        let bit_start = r.bit_position();
        let last = r.get(1)? == 1;
        let kind = match r.get(2)? {
            0 => {
                r.align();
                let p = r.pos;
                let len = u16::from_le_bytes(data.get(p..p + 2)?.try_into().unwrap()) as usize;
                let nlen = u16::from_le_bytes(data.get(p + 2..p + 4)?.try_into().unwrap());
                if nlen != !(len as u16) {
                    return None;
                }
                let bytes = data.get(p + 4..p + 4 + len)?.to_vec();
                plain.extend_from_slice(&bytes);
                r.pos = p + 4 + len;
                Kind::Stored(bytes)
            }
            1 => {
                let lit = Decoder::new(&fixed_lit_lens())?;
                let dist = Decoder::new(&[5u8; 32])?;
                Kind::Fixed(read_tokens(&mut r, &lit, &dist, &mut plain)?)
            }
            2 => {
                let header = read_header(&mut r)?;
                let lit = Decoder::new(&header.lit_lens)?;
                let dist = Decoder::new(&header.dist_lens)?;
                let tokens = read_tokens(&mut r, &lit, &dist, &mut plain)?;
                Kind::Dynamic(header, tokens)
            }
            _ => return None,
        };
        blocks.push(Block { last, bit_start, kind });
        if last {
            break;
        }
    }
    let bits = r.bit_position();
    Some(Stream { blocks, plain, consumed: ((bits + 7) / 8) as usize })
}

// ---------------------------------------------------------------- writing

pub struct BitWriter {
    pub out: Vec<u8>,
    bits: u64,
    n: u32,
}

impl BitWriter {
    /// A writer whose first `bit_offset` bits (0..8) are zero: for a
    /// piece of a stream that starts mid-byte.
    pub fn new(bit_offset: u32) -> Self {
        BitWriter { out: Vec::new(), bits: 0, n: bit_offset }
    }

    #[inline]
    pub fn put(&mut self, v: u32, k: u32) {
        self.bits |= (v as u64) << self.n;
        self.n += k;
        while self.n >= 8 {
            self.out.push(self.bits as u8);
            self.bits >>= 8;
            self.n -= 8;
        }
    }

    /// A Huffman code, given reversed so that its most significant bit
    /// goes out first.
    #[inline]
    fn code(&mut self, reversed: u16, len: u8) {
        self.put(reversed as u32, len as u32);
    }

    fn align(&mut self) {
        if self.n > 0 {
            self.out.push(self.bits as u8);
            self.bits = 0;
            self.n = 0;
        }
    }

    /// The bytes, the last padded with zero bits, and how many bits of
    /// that last byte are the stream's (0: it ended on a boundary).
    pub fn finish(mut self) -> (Vec<u8>, u32) {
        let partial = self.n;
        self.align();
        (self.out, partial)
    }
}

/// Canonical codes from lengths (RFC 1951 3.2.2), `(code, len)` per
/// symbol with the code's bits reversed for the writer; a symbol of
/// length 0 has no code.
fn codes(lens: &[u8]) -> Vec<(u16, u8)> {
    let mut count = [0u16; 16];
    for &l in lens {
        count[l as usize] += 1;
    }
    count[0] = 0;
    let mut next = [0u16; 16];
    let mut code = 0u16;
    for bits in 1..16 {
        code = (code + count[bits - 1]) << 1;
        next[bits] = code;
    }
    lens.iter()
        .map(|&l| {
            if l == 0 {
                (0, 0)
            } else {
                let c = next[l as usize];
                next[l as usize] += 1;
                (c.reverse_bits() >> (16 - l), l)
            }
        })
        .collect()
}

fn write_tokens(w: &mut BitWriter, tokens: &[Token], lit: &[(u16, u8)], dist: &[(u16, u8)]) {
    for t in tokens {
        match *t {
            Token::Lit(b) => {
                let (c, l) = lit[b as usize];
                w.code(c, l);
            }
            Token::Ref { len, dist: d } => {
                let (sym, extra, val) = len_code(len);
                let (c, l) = lit[sym as usize];
                w.code(c, l);
                w.put(val as u32, extra as u32);
                let (sym, extra, val) = dist_code(d);
                let (c, l) = dist[sym as usize];
                w.code(c, l);
                w.put(val as u32, extra as u32);
            }
        }
    }
    let (c, l) = lit[256];
    w.code(c, l);
}

/// A block written to `w` as it was read.
pub fn write_block(w: &mut BitWriter, block: &Block) {
    w.put(block.last as u32, 1);
    match &block.kind {
        Kind::Stored(bytes) => {
            w.put(0, 2);
            w.align();
            w.out.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
            w.out.extend_from_slice(&(!(bytes.len() as u16)).to_le_bytes());
            w.out.extend_from_slice(bytes);
        }
        Kind::Fixed(tokens) => {
            w.put(1, 2);
            write_tokens(w, tokens, &codes(&fixed_lit_lens()), &codes(&[5u8; 32]));
        }
        Kind::Dynamic(h, tokens) => {
            w.put(2, 2);
            write_header(w, h);
            write_tokens(w, tokens, &codes(&h.lit_lens), &codes(&h.dist_lens));
        }
    }
}

/// A dynamic block's header as bits, for keeping: the bytes and how
/// many bits of the last are used.
fn header_bits(h: &Header) -> (Vec<u8>, u32) {
    let mut w = BitWriter::new(0);
    write_header(&mut w, h);
    w.finish()
}

fn write_header(w: &mut BitWriter, h: &Header) {
    w.put(h.hlit as u32 - 257, 5);
    w.put(h.hdist as u32 - 1, 5);
    w.put(h.hclen as u32 - 4, 4);
    for i in 0..h.hclen as usize {
        w.put(h.clen[CLEN_ORDER[i]] as u32, 3);
    }
    let cl = codes(&h.clen);
    for &(sym, extra) in &h.symbols {
        let (c, l) = cl[sym as usize];
        w.code(c, l);
        match sym {
            16 => w.put(extra as u32, 2),
            17 => w.put(extra as u32, 3),
            18 => w.put(extra as u32, 7),
            _ => {}
        }
    }
}

/// A stream taken apart: its plain text, and the recipe that gives the
/// stream back from it (`close`).
pub struct Opened {
    pub plain: Vec<u8>,
    pub recipe: Vec<u8>,
    pub consumed: usize,
}

/// `f(i)` for every `i` below `n`, on every core.
fn each<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    if n < 2 {
        return (0..n).map(f).collect();
    }
    let slots: Vec<std::sync::Mutex<Option<T>>> = (0..n).map(|_| std::sync::Mutex::new(None)).collect();
    let _ = crate::par_units::<()>(n, |i| {
        let r = f(i);
        *slots[i].lock().unwrap() = Some(r);
        Ok(())
    });
    slots.into_iter().map(|m| m.into_inner().unwrap().expect("every chunk is made")).collect()
}

/// The plain text a stream is cut at, into chunks of about this much,
/// so that they open and close on every core.
pub const CHUNK: usize = 1 << 20;

/// `data` from its first byte, opened: the recipe holds the level
/// emulated, then the chunks — the blocks cut at block boundaries into
/// runs of about `CHUNK` bytes of plain text, each with its plain
/// length, block count, the bit within a byte its first block starts
/// at, and its corrections to an emulation that first learns the
/// 32 KB before it. `None` when the stream does not parse, or does not
/// come back bit for bit.
pub fn open(data: &[u8]) -> Option<Opened> {
    open_in(data, CHUNK)
}

pub fn open_in(data: &[u8], chunk: usize) -> Option<Opened> {
    let s = parse(data)?;
    let padded = zlib::pad(&s.plain);
    let p = zlib::detect(&padded, &s.blocks);
    // Chunks: (first block, block count, plain start, plain end).
    let mut chunks: Vec<(usize, usize, u32, u32)> = Vec::new();
    let (mut first, mut plain_start, mut at) = (0usize, 0u32, 0u32);
    for (i, b) in s.blocks.iter().enumerate() {
        at += match &b.kind {
            Kind::Stored(x) => x.len() as u32,
            Kind::Fixed(t) | Kind::Dynamic(_, t) => t.iter().map(|t| match t { Token::Lit(_) => 1, Token::Ref { len, .. } => if *len == 259 { 258 } else { *len as u32 } }).sum(),
        };
        if (at - plain_start) as usize >= chunk || i + 1 == s.blocks.len() {
            chunks.push((first, i + 1 - first, plain_start, at));
            first = i + 1;
            plain_start = at;
        }
    }
    let corrections = each(chunks.len(), |i| {
        let (first, n, start, _) = chunks[i];
        zlib::predict(&padded, p, &s.blocks[first..first + n], start)
    });
    let mut recipe = vec![p.level | (p.filtered as u8) << 4 | (p.fixed as u8) << 5, p.mem | p.window << 4];
    put_varint(&mut recipe, chunks.len() as u64);
    for (&(first, n, start, end), c) in chunks.iter().zip(&corrections) {
        put_varint(&mut recipe, (end - start) as u64);
        put_varint(&mut recipe, n as u64);
        recipe.push((s.blocks[first].bit_start % 8) as u8);
        put_varint(&mut recipe, c.len() as u64);
        recipe.extend_from_slice(c);
    }
    let opened = Opened { plain: s.plain, recipe, consumed: s.consumed };
    (close(&opened.plain, &opened.recipe)? == data[..opened.consumed]).then_some(opened)
}

/// The stream back from its plain text and recipe, its chunks on every
/// core.
pub fn close(plain: &[u8], recipe: &[u8]) -> Option<Vec<u8>> {
    let b = *recipe.first()?;
    let m = *recipe.get(1)?;
    let p = zlib::Params { level: b & 0xf, filtered: b & 0x10 != 0, fixed: b & 0x20 != 0, mem: m & 0xf, window: m >> 4 };
    if p.level > 9 || !(1..=9).contains(&p.mem) || !(9..=15).contains(&p.window) {
        return None;
    }
    let mut pos = 2usize;
    let n = get_varint(recipe, &mut pos).ok()? as usize;
    // (plain start, plain end, block count, bit offset, corrections)
    let mut chunks: Vec<(u32, u32, usize, u32, &[u8])> = Vec::with_capacity(n);
    let mut start = 0u32;
    for _ in 0..n {
        let len = get_varint(recipe, &mut pos).ok()? as u32;
        let blocks = get_varint(recipe, &mut pos).ok()? as usize;
        let bit = *recipe.get(pos)? as u32;
        pos += 1;
        let clen = get_varint(recipe, &mut pos).ok()? as usize;
        let c = recipe.get(pos..pos + clen)?;
        pos += clen;
        let end = start.checked_add(len)?;
        if end as usize > plain.len() {
            return None;
        }
        chunks.push((start, end, blocks, bit, c));
        start = end;
    }
    if start as usize != plain.len() {
        return None;
    }
    let padded = zlib::pad(plain);
    let pieces = each(chunks.len(), |i| {
        let (start, end, blocks, bit, c) = chunks[i];
        let blocks = zlib::recreate(&padded, p, c, blocks, start, end)?;
        let mut w = BitWriter::new(bit);
        for b in &blocks {
            write_block(&mut w, b);
        }
        Some(w.finish())
    });
    let pieces: Vec<(Vec<u8>, u32)> = pieces.into_iter().collect::<Option<_>>()?;
    let mut out: Vec<u8> = Vec::with_capacity(pieces.iter().map(|p| p.0.len()).sum());
    let mut partial = 0u32;
    for (bytes, bits) in pieces {
        if partial > 0 && !bytes.is_empty() {
            *out.last_mut().unwrap() |= bytes[0];
            out.extend_from_slice(&bytes[1..]);
        } else {
            out.extend_from_slice(&bytes);
        }
        partial = bits;
    }
    Some(out)
}

/// The stream's bytes back from its blocks.
pub fn write(blocks: &[Block]) -> Vec<u8> {
    let mut w = BitWriter::new(0);
    for b in blocks {
        write_block(&mut w, b);
    }
    w.finish().0
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use std::process::{Command, Stdio};

    pub fn text(n: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(n);
        let mut x = 0x9E3779B97F4A7C15u64;
        while v.len() < n {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            v.extend_from_slice(format!("{} GET /path/{}/item?id={} HTTP/1.0 {} {}\n", x % 251, x % 17, x % 100000, 200 + (x % 3) as u32 * 102, x % 9000).as_bytes());
        }
        v.truncate(n);
        v
    }

    fn noise(n: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(n);
        let mut x = 7u64;
        while v.len() < n {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            v.extend_from_slice(&x.to_le_bytes());
        }
        v.truncate(n);
        v
    }

    /// `cmd` on `input` from a file (a pipe would fill while we were
    /// still writing the input), its standard output.
    fn run(cmd: &str, args: &[&str], input: &[u8]) -> Vec<u8> {
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!("glyd-reflate-{}-{}", std::process::id(), N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)));
        std::fs::write(&path, input).unwrap();
        let args: Vec<String> = args.iter().map(|a| a.replace("{}", path.to_str().unwrap())).collect();
        let out = Command::new(cmd).args(&args).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null()).output().unwrap();
        let _ = std::fs::remove_file(&path);
        assert!(out.status.success(), "{cmd} {args:?}");
        out.stdout
    }

    /// The deflate body of a gzip member: past the header, before the
    /// 8-byte trailer.
    fn gzip_body(gz: &[u8]) -> &[u8] {
        let flg = gz[3];
        let mut p = 10;
        if flg & 4 != 0 {
            p += 2 + u16::from_le_bytes([gz[p], gz[p + 1]]) as usize;
        }
        for bit in [8u8, 16] {
            if flg & bit != 0 {
                while gz[p] != 0 {
                    p += 1;
                }
                p += 1;
            }
        }
        if flg & 2 != 0 {
            p += 2;
        }
        &gz[p..gz.len() - 8]
    }

    pub fn zlib_deflate(plain: &[u8], level: i32, strategy: &str) -> Vec<u8> {
        let script = format!("import sys,zlib\nc=zlib.compressobj({level},zlib.DEFLATED,-15,8,zlib.{strategy})\nd=c.compress(open(sys.argv[1],'rb').read())+c.flush()\nsys.stdout.buffer.write(d)\n");
        run("python3", &["-c", &script, "{}"], plain)
    }

    fn round_trip(stream: &[u8], plain: &[u8], what: &str) -> Stream {
        let s = parse(stream).unwrap_or_else(|| panic!("{what}: parses"));
        assert_eq!(s.consumed, stream.len(), "{what}: consumed");
        assert!(s.plain == plain, "{what}: plain text");
        assert!(write(&s.blocks) == stream, "{what}: bit-exact");
        s
    }

    #[test]
    fn gzip_at_every_level_round_trips() {
        let plain = text(2 << 20);
        for level in ["-1", "-3", "-6", "-9"] {
            let gz = run("gzip", &[level, "-c", "{}"], &plain);
            let s = round_trip(gzip_body(&gz), &plain, &format!("gzip {level}"));
            assert!(s.blocks.len() > 1 && s.blocks.iter().all(|b| matches!(b.kind, Kind::Dynamic(..))), "gzip {level}: dynamic blocks");
            assert_eq!(s.blocks[0].bit_start, 0);
            assert!(s.blocks[1].bit_start > 0);
        }
    }

    /// Whole streams opened and closed through the recipe: gzip at its
    /// levels, zlib with stored and fixed blocks; the recipe small next
    /// to the stream, and preflate's corrections for the same stream.
    #[test]
    fn open_and_close_bit_for_bit() {
        let plain = text(1 << 20);
        let mut streams: Vec<(String, Vec<u8>)> = ["-1", "-6", "-9"].iter().map(|l| (format!("gzip {l}"), gzip_body(&run("gzip", &[l, "-c", "{}"], &plain)).to_vec())).collect();
        for (level, strategy) in [(0, "Z_DEFAULT_STRATEGY"), (3, "Z_DEFAULT_STRATEGY"), (6, "Z_DEFAULT_STRATEGY"), (9, "Z_DEFAULT_STRATEGY"), (6, "Z_FIXED"), (6, "Z_HUFFMAN_ONLY")] {
            streams.push((format!("zlib -{level} {strategy}"), zlib_deflate(&plain, level, strategy)));
        }
        let mut mixed = text(200 << 10);
        mixed.extend_from_slice(&noise(300 << 10));
        mixed.extend_from_slice(&text(200 << 10));
        for (what, stream) in &streams {
            let o = open(stream).unwrap_or_else(|| panic!("{what}: opens"));
            assert!(o.plain == plain && o.consumed == stream.len(), "{what}");
            assert!(close(&o.plain, &o.recipe).unwrap() == *stream, "{what}: closes");
            assert!(o.recipe.len() * 100 < stream.len() || stream.len() < 4096, "{what}: recipe {} B for {} B", o.recipe.len(), stream.len());
            #[cfg(feature = "deflate")]
            {
                let pre = preflate_rs::preflate_whole_deflate_stream(stream, &preflate_rs::PreflateConfig::default()).map(|(r, _)| r.corrections.len());
                eprintln!("{what:28} stream {:7} B  recipe {:6} B ({:.2}%)  preflate {:?}", stream.len(), o.recipe.len(), 100.0 * o.recipe.len() as f64 / stream.len() as f64, pre.ok());
            }
        }
        let stream = zlib_deflate(&mixed, 6, "Z_DEFAULT_STRATEGY");
        let o = open(&stream).unwrap();
        assert!(o.plain == mixed && close(&o.plain, &o.recipe).unwrap() == stream, "stored blocks between compressed ones");
        // In chunks: every chunk's emulation starts from the 32 KB before
        // it, and the pieces join bit for bit; a few hundred bytes more.
        let whole = open_in(&stream, usize::MAX).unwrap();
        let chunked = open_in(&stream, 128 << 10).unwrap();
        assert!(close(&chunked.plain, &chunked.recipe).unwrap() == stream, "chunked stream closes");
        assert!(chunked.recipe.len() < whole.recipe.len() + 8 * 600, "chunked {} against whole {}", chunked.recipe.len(), whole.recipe.len());
        for (what, stream) in &streams[..3] {
            let chunked = open_in(stream, 128 << 10).unwrap();
            assert!(close(&chunked.plain, &chunked.recipe).unwrap() == *stream, "{what} in chunks");
        }
    }

    /// `GLYD_REFLATE_FILE=some.gz cargo test --release reflate::tests::speed -- --ignored --nocapture`:
    /// the gzip's stream opened and closed by reflate and by preflate,
    /// timed, one thread.
    #[test]
    #[ignore = "a timing on a file named by GLYD_REFLATE_FILE"]
    fn speed() {
        let Ok(path) = std::env::var("GLYD_REFLATE_FILE") else { return };
        let gz = std::fs::read(path).unwrap();
        let stream = gzip_body(&gz);
        let t = std::time::Instant::now();
        let s = parse(stream).unwrap();
        let parse_s = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        let padded = zlib::pad(&s.plain);
        let p = zlib::detect(&padded, &s.blocks);
        let detect_s = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        let c = zlib::predict(&padded, p, &s.blocks, 0);
        let predict_s = t.elapsed().as_secs_f64();
        eprintln!("reflate:  parse {parse_s:.2} s, detect {detect_s:.2} s, predict {predict_s:.2} s ({} B, {p:?})", c.len());
        let t = std::time::Instant::now();
        let o = open(stream).unwrap();
        let open_s = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        let back = close(&o.plain, &o.recipe).unwrap();
        let close_s = t.elapsed().as_secs_f64();
        assert!(back == &stream[..o.consumed], "consumed {} of {}, back {}, first difference {:?}", o.consumed, stream.len(), back.len(), back.iter().zip(stream).position(|(a, b)| a != b));
        let mb = o.plain.len() as f64 / 1e6;
        eprintln!("reflate:  open {open_s:.2} s ({:.0} MB/s of content, with the check), close {close_s:.2} s ({:.0} MB/s), recipe {} B, level {} mem {} window {}", mb / open_s, mb / close_s, o.recipe.len(), o.recipe[0] & 0xf, o.recipe[1] & 0xf, o.recipe[1] >> 4);
        #[cfg(feature = "deflate")]
        {
            let t = std::time::Instant::now();
            let config = preflate_rs::PreflateConfig { plain_text_limit: usize::MAX, ..Default::default() };
            let (r, text) = preflate_rs::preflate_whole_deflate_stream(stream, &config).unwrap();
            let open_s = t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            let back = preflate_rs::recreate_whole_deflate_stream(text.text(), &r.corrections).unwrap();
            let close_s = t.elapsed().as_secs_f64();
            assert!(back == &stream[..r.compressed_size]);
            eprintln!("preflate: open {open_s:.2} s ({:.0} MB/s), close {close_s:.2} s ({:.0} MB/s), corrections {} B", mb / open_s, mb / close_s, r.corrections.len());
        }
    }

    #[test]
    fn zlib_levels_strategies_stored_and_fixed_blocks_round_trip() {
        let plain = text(1 << 20);
        for level in 0..=9 {
            round_trip(&zlib_deflate(&plain, level, "Z_DEFAULT_STRATEGY"), &plain, &format!("zlib -{level}"));
        }
        for strategy in ["Z_FILTERED", "Z_HUFFMAN_ONLY", "Z_RLE", "Z_FIXED"] {
            let s = round_trip(&zlib_deflate(&plain, 6, strategy), &plain, strategy);
            if strategy == "Z_FIXED" {
                assert!(s.blocks.iter().all(|b| matches!(b.kind, Kind::Fixed(_))), "fixed blocks");
            }
        }
        // Stored blocks (level 0, and incompressible data at level 6),
        // with their byte alignment, between compressed ones.
        let s = round_trip(&zlib_deflate(&plain, 0, "Z_DEFAULT_STRATEGY"), &plain, "zlib -0");
        assert!(s.blocks.iter().all(|b| matches!(b.kind, Kind::Stored(_))));
        let mut mixed = text(200 << 10);
        mixed.extend_from_slice(&noise(300 << 10));
        mixed.extend_from_slice(&text(200 << 10));
        let s = round_trip(&zlib_deflate(&mixed, 6, "Z_DEFAULT_STRATEGY"), &mixed, "mixed");
        assert!(s.blocks.iter().any(|b| matches!(b.kind, Kind::Stored(_))) && s.blocks.iter().any(|b| matches!(b.kind, Kind::Dynamic(..))));
        // A short stream, and one with bytes after it.
        let small = zlib_deflate(b"hello hello hello hello", 6, "Z_DEFAULT_STRATEGY");
        round_trip(&small, b"hello hello hello hello", "small");
        let mut tail = small.clone();
        tail.extend_from_slice(b"trailer bytes");
        assert_eq!(parse(&tail).unwrap().consumed, small.len());
        assert!(parse(b"\x1f\x8b not a stream").is_none());
        assert!(parse(&small[..small.len() / 2]).is_none());
    }
}

#[cfg(test)]
mod pdf_probe {
    use super::*;

    /// `GLYD_PDF=x.pdf cargo test --release reflate::pdf_probe -- --ignored --nocapture`
    #[test]
    #[ignore = "a probe on a PDF named by GLYD_PDF"]
    fn streams() {
        let Ok(path) = std::env::var("GLYD_PDF") else { return };
        let data = std::fs::read(path).unwrap();
        let mut at = 0usize;
        let mut worst: Vec<(f64, usize, usize, String)> = Vec::new();
        while let Some(i) = data[at..].windows(6).position(|w| w == b"stream") {
            let mut p = at + i + 6;
            if data.get(p) == Some(&b'\r') { p += 1; }
            if data.get(p) == Some(&b'\n') { p += 1; }
            at = p;
            if data.get(p..p + 2).map_or(true, |h| h[0] & 0x0f != 8 || (u16::from_be_bytes([h[0], h[1]]) % 31) != 0) { continue; }
            let Some(s) = parse(&data[p + 2..]) else { continue };
            let padded = zlib::pad(&s.plain);
            let params = zlib::detect(&padded, &s.blocks);
            let c = zlib::predict(&padded, params, &s.blocks, 0);
            let mut desc = format!("{params:?}; blocks:");
            for b in s.blocks.iter().take(12) {
                desc += &match &b.kind { Kind::Stored(x) => format!(" S{}", x.len()), Kind::Fixed(t) => format!(" F{}", t.len()), Kind::Dynamic(_, t) => format!(" D{}", t.len()) };
            }
            // mismatches per block at the detected params
            let mut z = zlib::Zlib::new_at(&padded, params, 0);
            let mut pos = 0u32;
            let mut wrong = Vec::new();
            let mut max_dist = 0u16;
            let mut examples = Vec::new();
            for b in &s.blocks {
                let mut w = 0;
                match &b.kind {
                    Kind::Stored(x) => { pos += x.len() as u32; }
                    Kind::Fixed(t) | Kind::Dynamic(_, t) => for &tok in t {
                        if let Token::Ref { dist, .. } = tok { max_dist = max_dist.max(dist); }
                        let g = z.predict(pos);
                        if g != tok { w += 1; if examples.len() < 5 && w > 100 { examples.push((pos, g, tok)); } }
                        pos = z.commit(pos, tok);
                    }
                }
                wrong.push(w);
            }
            desc += &format!("; max dist {max_dist}; e.g. {examples:?}");
            if wrong.iter().any(|&w| w > 0) && s.plain.len() > 4096 {
                desc += &format!("; wrong per block {:?}", &wrong[..wrong.len().min(12)]);
            }
            worst.push((c.len() as f64 / s.consumed as f64, c.len(), s.plain.len(), desc));
        }
        // The largest streams: where the time goes, on every core.
        let mut at = 0usize;
        let mut biggest: Vec<(usize, usize)> = Vec::new();
        while let Some(i) = data[at..].windows(6).position(|w| w == b"stream") {
            let mut p = at + i + 6;
            if data.get(p) == Some(&b'\r') { p += 1; }
            if data.get(p) == Some(&b'\n') { p += 1; }
            at = p;
            if data.get(p..p + 2).map_or(true, |h| h[0] & 0x0f != 8 || (u16::from_be_bytes([h[0], h[1]]) % 31) != 0) { continue; }
            if let Some(s) = parse(&data[p + 2..]) { biggest.push((s.plain.len(), p + 2)); }
        }
        biggest.sort_by(|a, b| b.cmp(a));
        for &(plain_len, p) in biggest.iter().take(3) {
            let stream = &data[p..];
            let t = std::time::Instant::now();
            let s = parse(stream).unwrap();
            let parse_s = t.elapsed().as_secs_f64();
            let padded = zlib::pad(&s.plain);
            let t = std::time::Instant::now();
            let params = zlib::detect(&padded, &s.blocks);
            let detect_s = t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            let o = open(&stream[..s.consumed]).unwrap();
            let open_s = t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            let back = close(&o.plain, &o.recipe).unwrap();
            let close_s = t.elapsed().as_secs_f64();
            assert!(back == &stream[..s.consumed]);
            eprintln!("stream of {plain_len} B plain: parse {parse_s:.2} s, detect {detect_s:.2} s, open (with the check) {open_s:.2} s, close {close_s:.2} s; {params:?}, recipe {} B", o.recipe.len());
        }
        worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        for (ratio, c, plain, desc) in worst.iter().take(6) {
            eprintln!("{:.1}% of stream: corrections {c} B, plain {plain} B, {desc}", ratio * 100.0);
        }
        eprintln!("{} streams, corrections total {} B", worst.len(), worst.iter().map(|w| w.1).sum::<usize>());
    }
}
