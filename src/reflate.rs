//! Deflate streams taken apart and put back, bit for bit, with nothing
//! outside this crate (`docs/design/deflate-reconstruction.md`).
//!
//! This is the first part: a parser that turns a stream into its blocks
//! — stored bytes, or tokens under fixed or dynamic Huffman codes, with
//! a dynamic block's header kept as the symbols it was written with —
//! and a writer that emits the same blocks as the same bits. Every
//! block also records the bit it started at, for the pieces to be
//! written on every core later.

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

/// The length code and extra bits of a reference length.
fn len_code(len: u16) -> (u16, u8, u16) {
    if len == 259 {
        return (284, 5, 31);
    }
    let i = LEN_BASE.iter().rposition(|&b| b <= len).unwrap();
    (257 + i as u16, LEN_EXTRA[i], len - LEN_BASE[i])
}

fn dist_code(dist: u16) -> (u16, u8, u16) {
    let i = DIST_BASE.iter().rposition(|&b| b <= dist).unwrap();
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

    /// A Huffman code of `len` bits, most significant first.
    #[inline]
    fn code(&mut self, code: u16, len: u8) {
        let mut rev = 0u32;
        for i in 0..len {
            rev |= (((code >> i) & 1) as u32) << (len - 1 - i);
        }
        self.put(rev, len as u32);
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
/// symbol; a symbol of length 0 has no code.
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
                (c, l)
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
            write_tokens(w, tokens, &codes(&h.lit_lens), &codes(&h.dist_lens));
        }
    }
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
mod tests {
    use super::*;
    use std::process::{Command, Stdio};

    fn text(n: usize) -> Vec<u8> {
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

    fn zlib_deflate(plain: &[u8], level: i32, strategy: &str) -> Vec<u8> {
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
