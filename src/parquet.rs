//! Parquet files read page by page: the footer's row groups and
//! column chunks, and each chunk's page headers, so a page's
//! compressed bytes can be opened (snappy: `resnappy`) and its values
//! modeled instead of its LZ tokens. Thrift's compact protocol, the
//! part Parquet's metadata needs; no dependency.

pub const MAGIC: &[u8; 4] = b"PAR1";

/// A column chunk's compression, Parquet's `CompressionCodec`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Codec {
    Uncompressed,
    Snappy,
    Gzip,
    Lzo,
    Brotli,
    Lz4,
    Zstd,
    Lz4Raw,
    Other(i32),
}

impl Codec {
    fn of(v: i32) -> Codec {
        match v {
            0 => Codec::Uncompressed,
            1 => Codec::Snappy,
            2 => Codec::Gzip,
            3 => Codec::Lzo,
            4 => Codec::Brotli,
            5 => Codec::Lz4,
            6 => Codec::Zstd,
            7 => Codec::Lz4Raw,
            v => Codec::Other(v),
        }
    }
}

/// A column chunk: where its pages are and how they are coded.
#[derive(Clone, Debug)]
pub struct Chunk {
    pub row_group: usize,
    pub column: usize,
    pub codec: Codec,
    /// Parquet's physical `Type`: 0 boolean, 1 i32, 2 i64, 3 i96, 4 float, 5 double, 6 byte array, 7 fixed byte array.
    pub physical_type: i32,
    /// The column's greatest definition level: 0 for a required
    /// column of a flat schema, 1 for an optional one; a column under
    /// a group or a list is deeper (its levels are not modeled).
    pub max_def_level: u16,
    pub max_rep_level: u16,
    pub encodings: Vec<i32>,
    pub num_values: i64,
    /// The first page header's offset in the file.
    pub start: usize,
    pub compressed_len: usize,
    pub uncompressed_len: usize,
}

/// One page of a chunk.
#[derive(Clone, Debug)]
pub struct Page {
    pub header_at: usize,
    pub body_at: usize,
    pub compressed_len: usize,
    pub uncompressed_len: usize,
    /// Parquet's `PageType`: 0 data, 1 index, 2 dictionary, 3 data v2.
    pub kind: i32,
    /// The page's values (nulls counted) and their encoding: 0 plain,
    /// 2 plain dictionary, 3 RLE, 5 delta binary packed, 6 delta
    /// length byte array, 7 delta byte array, 8 RLE dictionary, 9 byte
    /// stream split.
    pub num_values: i64,
    pub encoding: i32,
    /// A v2 page's nulls, from its header; a v1 page's are counted from
    /// its definition levels.
    pub num_nulls: i64,
    /// A v2 page's repetition and definition levels, stored plain
    /// before the values; the values alone are compressed.
    pub v2_levels_len: usize,
    pub v2_compressed: bool,
}

impl Page {
    /// The compressed bytes: the whole body, or a v2 page's values.
    pub fn compressed<'a>(&self, data: &'a [u8]) -> &'a [u8] {
        let skip = if self.kind == 3 { self.v2_levels_len } else { 0 };
        &data[self.body_at + skip..self.body_at + self.compressed_len]
    }
}

/// Thrift's compact protocol: a reader over `b` from `p`.
struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Reader<'a> {
    fn byte(&mut self) -> Option<u8> {
        let v = *self.b.get(self.p)?;
        self.p += 1;
        Some(v)
    }

    fn varint(&mut self) -> Option<u64> {
        let (mut v, mut shift) = (0u64, 0u32);
        loop {
            let x = self.byte()?;
            v |= ((x & 0x7f) as u64) << shift;
            if x & 0x80 == 0 {
                return Some(v);
            }
            shift += 7;
            if shift > 63 {
                return None;
            }
        }
    }

    fn zigzag(&mut self) -> Option<i64> {
        let v = self.varint()?;
        Some(((v >> 1) as i64) ^ -((v & 1) as i64))
    }

    fn bytes(&mut self) -> Option<&'a [u8]> {
        let n = self.varint()? as usize;
        let s = self.b.get(self.p..self.p + n)?;
        self.p += n;
        Some(s)
    }

    /// A list's element count and element type.
    fn list(&mut self) -> Option<(usize, u8)> {
        let h = self.byte()?;
        let t = h & 15;
        let n = if h >> 4 == 15 { self.varint()? as usize } else { (h >> 4) as usize };
        Some((n, t))
    }

    fn skip(&mut self, t: u8) -> Option<()> {
        match t {
            1 | 2 => Some(()),
            3 => self.byte().map(|_| ()),
            4..=6 => self.varint().map(|_| ()),
            7 => {
                self.p += 8;
                (self.p <= self.b.len()).then_some(())
            }
            8 => self.bytes().map(|_| ()),
            9 | 10 => {
                let (n, et) = self.list()?;
                for _ in 0..n {
                    self.skip(et)?;
                }
                Some(())
            }
            11 => {
                let n = self.varint()? as usize;
                if n > 0 {
                    let kv = self.byte()?;
                    for _ in 0..n {
                        self.skip(kv >> 4)?;
                        self.skip(kv & 15)?;
                    }
                }
                Some(())
            }
            12 => self.fields(|_, _, _| Some(false)),
            _ => None,
        }
    }

    /// A struct's fields in turn: `f(reader, id, type)` consumes the
    /// value and answers true, or answers false and it is skipped.
    fn fields(&mut self, mut f: impl FnMut(&mut Reader<'a>, i16, u8) -> Option<bool>) -> Option<()> {
        let mut fid = 0i16;
        loop {
            let h = self.byte()?;
            if h == 0 {
                return Some(());
            }
            let (d, t) = ((h >> 4) as i16, h & 15);
            fid = if d != 0 { fid + d } else { self.zigzag()? as i16 };
            if !f(self, fid, t)? {
                self.skip(t)?;
            }
        }
    }
}

pub fn is_parquet(data: &[u8]) -> bool {
    data.len() >= 12 && &data[..4] == MAGIC && &data[data.len() - 4..] == MAGIC
}

/// The file's column chunks in file order, and the writer's name.
pub fn chunks(data: &[u8]) -> Option<(Vec<Chunk>, String)> {
    if !is_parquet(data) {
        return None;
    }
    let n = data.len();
    let flen = u32::from_le_bytes(data[n - 8..n - 4].try_into().unwrap()) as usize;
    let footer = data.get(n.checked_sub(8 + flen)?..n - 8)?;
    let mut r = Reader { b: footer, p: 0 };
    // Each leaf's physical type and its levels: a leaf right under the
    // root has max_def 1 when optional (repetition 1), 0 when required,
    // and max_rep 0; a leaf under a group or a list carries its
    // ancestors' levels, computed the way the format defines them.
    let mut leaf_types: Vec<(i32, u16, u16)> = Vec::new();
    let mut chunks = Vec::new();
    let mut created_by = String::new();
    r.fields(|r, fid, t| {
        match (fid, t) {
            (2, 9) => {
                // schema: elements in depth-first order, each with its
                // children count; a stack of (children left, def, rep)
                // gives every leaf its levels.
                let (n, _) = r.list()?;
                let mut stack: Vec<(i64, u16, u16)> = Vec::new();
                for i in 0..n {
                    let (mut ty, mut repetition, mut children) = (None, 0i64, 0i64);
                    r.fields(|r, fid, t| {
                        match (fid, t) {
                            (1, 5) | (1, 6) => ty = Some(r.zigzag()? as i32),
                            (3, 5) | (3, 6) => repetition = r.zigzag()?,
                            (5, 5) | (5, 6) => children = r.zigzag()?,
                            _ => return Some(false),
                        }
                        Some(true)
                    })?;
                    let (mut def, mut rep) = stack.last().map_or((0, 0), |&(_, d, r)| (d, r));
                    if i > 0 {
                        // OPTIONAL (1) adds a definition level, REPEATED (2) both.
                        if repetition == 1 { def += 1; }
                        if repetition == 2 { def += 1; rep += 1; }
                    }
                    if let Some(top) = stack.last_mut() {
                        top.0 -= 1;
                    }
                    if children > 0 || i == 0 {
                        stack.push((children, def, rep));
                    } else if let Some(ty) = ty {
                        leaf_types.push((ty, def, rep));
                    }
                    while stack.len() > 1 && stack.last().map_or(false, |&(left, _, _)| left <= 0) {
                        stack.pop();
                        if let Some(top) = stack.last_mut() {
                            // A finished group counts as one child of its parent, counted at its push.
                            let _ = top;
                        }
                    }
                }
                Some(true)
            }
            (4, 9) => {
                let (groups, _) = r.list()?;
                for g in 0..groups {
                    r.fields(|r, fid, t| {
                        if fid != 1 || t != 9 {
                            return Some(false);
                        }
                        let (cols, _) = r.list()?;
                        for c in 0..cols {
                            let (ty, max_def, max_rep) = leaf_types.get(c).copied().unwrap_or((-1, u16::MAX, u16::MAX));
                            let mut chunk = Chunk { row_group: g, column: c, codec: Codec::Uncompressed, physical_type: ty, max_def_level: max_def, max_rep_level: max_rep, encodings: Vec::new(), num_values: 0, start: 0, compressed_len: 0, uncompressed_len: 0 };
                            let (mut data_at, mut dict_at) = (0usize, 0usize);
                            r.fields(|r, fid, t| {
                                if fid != 3 || t != 12 {
                                    return Some(false);
                                }
                                r.fields(|r, fid, t| {
                                    match (fid, t) {
                                        (1, 5) => chunk.physical_type = r.zigzag()? as i32,
                                        (2, 9) => {
                                            let (n, _) = r.list()?;
                                            for _ in 0..n {
                                                chunk.encodings.push(r.zigzag()? as i32);
                                            }
                                        }
                                        (4, 5) => chunk.codec = Codec::of(r.zigzag()? as i32),
                                        (5, 6) => chunk.num_values = r.zigzag()?,
                                        (6, 6) => chunk.uncompressed_len = r.zigzag()? as usize,
                                        (7, 6) => chunk.compressed_len = r.zigzag()? as usize,
                                        (9, 6) => data_at = r.zigzag()? as usize,
                                        (11, 6) => dict_at = r.zigzag()? as usize,
                                        _ => return Some(false),
                                    }
                                    Some(true)
                                })?;
                                Some(true)
                            })?;
                            chunk.start = if dict_at > 0 && dict_at < data_at { dict_at } else { data_at };
                            chunks.push(chunk);
                        }
                        Some(true)
                    })?;
                }
                Some(true)
            }
            (6, 8) => {
                created_by = String::from_utf8_lossy(r.bytes()?).into_owned();
                Some(true)
            }
            _ => Some(false),
        }
    })?;
    Some((chunks, created_by))
}

/// A chunk's pages, from its page headers.
pub fn pages(data: &[u8], chunk: &Chunk) -> Option<Vec<Page>> {
    let mut out = Vec::new();
    let mut p = chunk.start;
    let end = chunk.start.checked_add(chunk.compressed_len)?;
    if end > data.len() {
        return None;
    }
    while p < end {
        let mut r = Reader { b: data, p };
        let mut page = Page { header_at: p, body_at: 0, compressed_len: 0, uncompressed_len: 0, kind: -1, num_values: 0, encoding: -1, num_nulls: 0, v2_levels_len: 0, v2_compressed: true };
        r.fields(|r, fid, t| {
            match (fid, t) {
                (1, 5) => page.kind = r.zigzag()? as i32,
                (2, 5) => page.uncompressed_len = r.zigzag()? as usize,
                (3, 5) => page.compressed_len = r.zigzag()? as usize,
                // data_page_header (5) and dictionary_page_header (7): num_values, encoding.
                (5, 12) | (7, 12) => {
                    r.fields(|r, fid, t| {
                        match (fid, t) {
                            (1, 5) | (1, 6) => page.num_values = r.zigzag()?,
                            (2, 5) | (2, 6) => page.encoding = r.zigzag()? as i32,
                            _ => return Some(false),
                        }
                        Some(true)
                    })?;
                }
                (8, 12) => {
                    r.fields(|r, fid, t| {
                        match (fid, t) {
                            (1, 5) | (1, 6) => page.num_values = r.zigzag()?,
                            (2, 5) | (2, 6) => page.num_nulls = r.zigzag()?,
                            (4, 5) | (4, 6) => page.encoding = r.zigzag()? as i32,
                            (5, 5) => page.v2_levels_len += r.zigzag()? as usize,
                            (6, 5) => page.v2_levels_len += r.zigzag()? as usize,
                            (7, 1) => page.v2_compressed = true,
                            (7, 2) => page.v2_compressed = false,
                            _ => return Some(false),
                        }
                        Some(true)
                    })?;
                }
                _ => return Some(false),
            }
            Some(true)
        })?;
        page.body_at = r.p;
        if page.kind < 0 || page.body_at + page.compressed_len > end {
            return None;
        }
        p = page.body_at + page.compressed_len;
        out.push(page);
    }
    Some(out)
}

// ---- The values modeled --------------------------------------------------
//
// A page's values transformed so the LZ and entropy stages see their
// structure: fixed-width values as byte planes (every value's first
// byte, then every second byte, ...), counters and times as deltas
// first, doubles that are decimals as scaled integers, byte arrays as
// their lengths then their bytes. The page's level blocks stay as they
// are ahead of the values. `model` gives the recipe and the modeled
// bytes, `unmodel` the page again; None for a layout the models do not
// cover, which is compressed as it is.

/// The page as it is: the LZ stage does better on it than any model.
const M_RAW: u8 = 0;
const M_PLANES: u8 = 1;
const M_DELTA_PLANES: u8 = 2;
const M_DECIMAL: u8 = 3;
const M_BYTE_ARRAYS: u8 = 4;
/// Dictionary indices (RLE and bit-packed runs) unpacked, as planes.
const M_INDICES: u8 = 5;

// ---- Parquet's RLE / bit-packing hybrid ------------------------------------
//
// Runs of `bit_width`-bit values: a varint header, its low bit 1 for
// `header >> 1` groups of eight bit-packed values, 0 for a value
// repeated `header >> 1` times (the value in ceil(bit_width / 8) bytes).
// `rle_encode` is Arrow's encoder step for step (parquet-cpp, and so
// pyarrow, Spark's native writers and more), so an index page it wrote
// is written again from its values.

fn rle_decode(b: &[u8], bit_width: u32, at_most: usize) -> Option<Vec<u32>> {
    let mut out = Vec::new();
    let mut p = 0usize;
    let value_bytes = ((bit_width + 7) / 8) as usize;
    while p < b.len() {
        let header = get_varint(b, &mut p)?;
        if header & 1 == 1 {
            let groups = (header >> 1) as usize;
            let bytes = groups.checked_mul(bit_width as usize)?;
            let packed = b.get(p..p + bytes)?;
            p += bytes;
            let (mut acc, mut bits) = (0u64, 0u32);
            let (mut q, mut n) = (0usize, groups * 8);
            while n > 0 {
                while bits < bit_width {
                    acc |= (packed[q] as u64) << bits;
                    q += 1;
                    bits += 8;
                }
                out.push((acc & ((1u64 << bit_width) - 1)) as u32);
                acc >>= bit_width;
                bits -= bit_width;
                n -= 1;
            }
        } else {
            let count = (header >> 1) as usize;
            let mut v = 0u32;
            for (k, &x) in b.get(p..p + value_bytes)?.iter().enumerate() {
                v |= (x as u32) << (8 * k);
            }
            p += value_bytes;
            if count > at_most.saturating_sub(out.len()) + 8 {
                return None;
            }
            out.extend(std::iter::repeat(v).take(count));
        }
        if out.len() > at_most + 8 {
            return None;
        }
    }
    Some(out)
}

/// Arrow's `RleEncoder`: eight values buffered; a value seen eight
/// times running becomes a repeated run, else groups of eight are
/// bit-packed into a literal run of up to 504 values; the last group
/// padded with zeros.
fn rle_encode(values: &[u32], bit_width: u32) -> Vec<u8> {
    struct W {
        out: Vec<u8>,
        acc: u64,
        bits: u32,
    }
    impl W {
        fn put(&mut self, v: u64, n: u32) {
            self.acc |= v << self.bits;
            self.bits += n;
            while self.bits >= 8 {
                self.out.push(self.acc as u8);
                self.acc >>= 8;
                self.bits -= 8;
            }
        }
        fn align(&mut self) {
            if self.bits > 0 {
                self.out.push(self.acc as u8);
                self.acc = 0;
                self.bits = 0;
            }
        }
    }
    let value_bytes = (bit_width + 7) / 8;
    let mut w = W { out: Vec::with_capacity(values.len() * bit_width as usize / 8 + 16), acc: 0, bits: 0 };
    let (mut buffered, mut num_buffered) = ([0u32; 8], 0usize);
    let (mut current, mut repeat, mut literal) = (0u32, 0usize, 0usize);
    let mut indicator: Option<usize> = None;
    let flush_literal = |w: &mut W, buffered: &[u32; 8], num_buffered: &mut usize, literal: &mut usize, indicator: &mut Option<usize>, update: bool| {
        if indicator.is_none() {
            w.align();
            *indicator = Some(w.out.len());
            w.out.push(0);
        }
        for &v in &buffered[..*num_buffered] {
            w.put(v as u64, bit_width);
        }
        *num_buffered = 0;
        if update {
            let groups = *literal / 8;
            w.out[indicator.unwrap()] = ((groups << 1) | 1) as u8;
            *indicator = None;
            *literal = 0;
        }
    };
    let flush_repeated = |w: &mut W, current: u32, repeat: &mut usize, num_buffered: &mut usize| {
        w.align();
        put_varint(&mut w.out, (*repeat << 1) as u64);
        w.out.extend_from_slice(&current.to_le_bytes()[..value_bytes as usize]);
        *num_buffered = 0;
        *repeat = 0;
    };
    for &v in values {
        if current == v {
            repeat += 1;
            if repeat > 8 {
                continue;
            }
        } else {
            if repeat >= 8 {
                flush_repeated(&mut w, current, &mut repeat, &mut num_buffered);
            }
            repeat = 1;
            current = v;
        }
        buffered[num_buffered] = v;
        num_buffered += 1;
        if num_buffered == 8 {
            // FlushBufferedValues(false)
            if repeat >= 8 {
                num_buffered = 0;
                if literal != 0 {
                    flush_literal(&mut w, &buffered, &mut num_buffered, &mut literal, &mut indicator, true);
                }
            } else {
                literal += num_buffered;
                let groups = literal / 8;
                flush_literal(&mut w, &buffered, &mut num_buffered, &mut literal, &mut indicator, groups + 1 >= 64);
                repeat = 0;
            }
        }
    }
    // Flush()
    if literal > 0 || repeat > 0 || num_buffered > 0 {
        let all_repeat = literal == 0 && (repeat == num_buffered || num_buffered == 0);
        if repeat > 0 && all_repeat {
            flush_repeated(&mut w, current, &mut repeat, &mut num_buffered);
        } else {
            while num_buffered != 0 && num_buffered < 8 {
                buffered[num_buffered] = 0;
                num_buffered += 1;
            }
            literal += num_buffered;
            flush_literal(&mut w, &buffered, &mut num_buffered, &mut literal, &mut indicator, true);
        }
    }
    w.align();
    w.out
}

/// Why an index page was not modeled: a probe's diagnosis.
pub fn index_page_diagnosis(raw: &[u8], chunk: &Chunk, page: &Page) -> String {
    let Some(at) = values_at(raw, chunk, page) else { return "levels".into() };
    let values = &raw[at..];
    if values.is_empty() { return "empty".into(); }
    let bit_width = values[0] as u32;
    if bit_width == 0 || bit_width > 32 { return format!("bit width {bit_width}"); }
    let Some(count) = present_values(raw, chunk, page) else { return "present".into() };
    let Some(mut decoded) = rle_decode(&values[1..], bit_width, count) else { return "decode".into() };
    if decoded.len() < count { return format!("short: {} of {count}", decoded.len()); }
    decoded.truncate(count);
    let again = rle_encode(&decoded, bit_width);
    if again != values[1..] {
        let at = again.iter().zip(values[1..].iter()).position(|(a, b)| a != b).unwrap_or(again.len().min(values.len() - 1));
        return format!("encode differs at {at} of {} (ours {} B), width {bit_width}", values.len() - 1, again.len());
    }
    "no gain".into()
}

/// A v1 data page's values that are not null: its definition levels
/// decoded and counted; all of them for a required column.
fn present_values(raw: &[u8], chunk: &Chunk, page: &Page) -> Option<usize> {
    if page.kind == 3 {
        return Some((page.num_values - page.num_nulls).max(0) as usize);
    }
    if chunk.max_def_level == 0 {
        return Some(page.num_values.max(0) as usize);
    }
    let mut p = 0usize;
    if chunk.max_rep_level > 0 {
        let n = u32::from_le_bytes(raw.get(p..p + 4)?.try_into().unwrap()) as usize;
        p += 4 + n;
    }
    let n = u32::from_le_bytes(raw.get(p..p + 4)?.try_into().unwrap()) as usize;
    let levels = raw.get(p + 4..p + 4 + n)?;
    let bits = 32 - (chunk.max_def_level as u32).leading_zeros();
    let decoded = rle_decode(levels, bits, page.num_values.max(0) as usize)?;
    Some(decoded.iter().take(page.num_values.max(0) as usize).filter(|&&d| d == chunk.max_def_level as u32).count())
}

fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn get_varint(b: &[u8], p: &mut usize) -> Option<u64> {
    let (mut v, mut shift) = (0u64, 0u32);
    loop {
        let x = *b.get(*p)?;
        *p += 1;
        v |= ((x & 0x7f) as u64) << shift;
        if x & 0x80 == 0 {
            return Some(v);
        }
        shift += 7;
        if shift > 63 {
            return None;
        }
    }
}

/// Order-0 bits of `b`: the coder's cost on a stream without matches.
#[cfg(test)]
fn bits(b: &[u8]) -> f64 {
    let mut h = [0u64; 256];
    for &x in b {
        h[x as usize] += 1;
    }
    let n = b.len() as f64;
    h.iter().filter(|&&c| c > 0).map(|&c| c as f64 * (n / c as f64).log2()).sum()
}

/// What a candidate costs the codec: the max level's own output. An
/// order-0 estimate chose deltas of unsorted times over the raw bytes
/// the LZ stage matched (10% worse); the fast level, with no entropy
/// stage, saw no gain in skewed dictionary indices.
fn cost(b: &[u8]) -> usize {
    let mut out = Vec::with_capacity(b.len() / 2 + 64);
    crate::compress_into_max(b, &mut out);
    out.len()
}

/// The largest power of ten (up to 10^9) dividing every value.
fn common_power(values: &[u8], width: usize) -> u32 {
    let mut d = 9u32;
    'next: while d > 0 {
        let p = 10u64.pow(d);
        for v in values.chunks_exact(width) {
            let mut x = 0u64;
            for (k, &b) in v.iter().enumerate() {
                x |= (b as u64) << (8 * k);
            }
            if x % p != 0 {
                d -= 1;
                continue 'next;
            }
        }
        return d;
    }
    0
}

fn scaled(values: &[u8], width: usize, d: u32, divide: bool) -> Vec<u8> {
    let p = 10u64.pow(d);
    let mut out = Vec::with_capacity(values.len());
    let mask = if width == 8 { u64::MAX } else { (1u64 << (8 * width)) - 1 };
    for v in values.chunks_exact(width) {
        let mut x = 0u64;
        for (k, &b) in v.iter().enumerate() {
            x |= (b as u64) << (8 * k);
        }
        let y = if divide { x / p } else { x.wrapping_mul(p) & mask };
        out.extend_from_slice(&y.to_le_bytes()[..width]);
    }
    out
}

fn planes(values: &[u8], width: usize, out: &mut Vec<u8>) {
    let count = values.len() / width;
    for j in 0..width {
        for i in 0..count {
            out.push(values[i * width + j]);
        }
    }
}

fn unplanes(planes: &[u8], width: usize, out: &mut Vec<u8>) {
    let count = planes.len() / width;
    for i in 0..count {
        for j in 0..width {
            out.push(planes[j * count + i]);
        }
    }
}

fn deltas(values: &[u8], width: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len());
    let mut prev = 0u64;
    for v in values.chunks_exact(width) {
        let mut x = 0u64;
        for (k, &b) in v.iter().enumerate() {
            x |= (b as u64) << (8 * k);
        }
        let d = x.wrapping_sub(prev);
        prev = x;
        out.extend_from_slice(&d.to_le_bytes()[..width]);
    }
    out
}

fn undeltas(d: &[u8], width: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(d.len());
    let mut prev = 0u64;
    for v in d.chunks_exact(width) {
        let mut x = 0u64;
        for (k, &b) in v.iter().enumerate() {
            x |= (b as u64) << (8 * k);
        }
        let mask = if width == 8 { u64::MAX } else { (1u64 << (8 * width)) - 1 };
        let x = prev.wrapping_add(x) & mask;
        prev = x;
        out.extend_from_slice(&x.to_le_bytes()[..width]);
    }
    out
}

/// Doubles that are decimals with `d` places, as scaled integers, when
/// every one comes back to the same bits.
fn decimal_ints(values: &[u8], d: u32) -> Option<Vec<u8>> {
    let scale = 10f64.powi(d as i32);
    let mut out = Vec::with_capacity(values.len());
    for v in values.chunks_exact(8) {
        let x = f64::from_le_bytes(v.try_into().unwrap());
        let s = (x * scale).round();
        if !(s.abs() < 9.0e15) || (s / scale).to_bits() != x.to_bits() {
            return None;
        }
        out.extend_from_slice(&(s as i64).to_le_bytes());
    }
    Some(out)
}

fn undecimal(ints: &[u8], d: u32) -> Vec<u8> {
    let scale = 10f64.powi(d as i32);
    let mut out = Vec::with_capacity(ints.len());
    for v in ints.chunks_exact(8) {
        let s = i64::from_le_bytes(v.try_into().unwrap()) as f64;
        out.extend_from_slice(&(s / scale).to_le_bytes());
    }
    out
}

/// Where a v1 data page's values start: past its level blocks.
fn values_at(raw: &[u8], chunk: &Chunk, page: &Page) -> Option<usize> {
    if page.kind != 0 {
        return Some(0);
    }
    let mut p = 0usize;
    for present in [chunk.max_rep_level > 0, chunk.max_def_level > 0] {
        if present {
            let n = u32::from_le_bytes(raw.get(p..p + 4)?.try_into().unwrap()) as usize;
            p = p.checked_add(4 + n)?;
        }
    }
    (p <= raw.len()).then_some(p)
}

/// The page modeled: its recipe and its modeled bytes.
pub fn model(raw: &[u8], chunk: &Chunk, page: &Page) -> Option<(Vec<u8>, Vec<u8>)> {
    // Plain values only: a data page (v1 or v2) encoded PLAIN, or a
    // dictionary page (PLAIN, or PLAIN_DICTIONARY as older writers
    // name it); a column whose levels the reader could not work out
    // is left alone.
    let plain = match page.kind {
        0 | 3 => page.encoding == 0,
        2 => page.encoding == 0 || page.encoding == 2,
        _ => false,
    };
    let indices = matches!(page.kind, 0 | 3) && matches!(page.encoding, 2 | 8);
    if !(plain || indices) || chunk.max_def_level == u16::MAX {
        return None;
    }
    let at = values_at(raw, chunk, page)?;
    let (prefix, values) = raw.split_at(at);
    if values.is_empty() {
        return None;
    }
    if indices {
        // Dictionary indices: the runs decoded, written again by
        // Arrow's encoder and compared; the indices then as planes of
        // the bytes that hold them.
        let bit_width = values[0] as u32;
        if bit_width == 0 || bit_width > 32 {
            return None;
        }
        let count = present_values(raw, chunk, page)?;
        let mut decoded = rle_decode(&values[1..], bit_width, count)?;
        if decoded.len() < count {
            return None;
        }
        decoded.truncate(count);
        if rle_encode(&decoded, bit_width) != values[1..] {
            return None;
        }
        let width = if bit_width <= 8 { 1 } else if bit_width <= 16 { 2 } else { 4 };
        let mut bytes = Vec::with_capacity(count * width);
        for &v in &decoded {
            bytes.extend_from_slice(&v.to_le_bytes()[..width]);
        }
        let mut p = Vec::with_capacity(bytes.len());
        planes(&bytes, width, &mut p);
        if cost(&p) >= cost(values) {
            return None;
        }
        let mut recipe = Vec::with_capacity(16);
        recipe.push(M_INDICES);
        put_varint(&mut recipe, prefix.len() as u64);
        put_varint(&mut recipe, count as u64);
        recipe.push(width as u8);
        recipe.push(bit_width as u8);
        let mut out = Vec::with_capacity(raw.len());
        out.extend_from_slice(prefix);
        out.extend_from_slice(&p);
        return Some((recipe, out));
    }
    let width = match chunk.physical_type {
        1 | 4 => 4,
        2 | 5 => 8,
        6 => 0,
        _ => return None,
    };
    let mut recipe = Vec::with_capacity(16);
    let mut out = Vec::with_capacity(raw.len() + 16);
    out.extend_from_slice(prefix);
    if width == 0 {
        // Byte arrays: [u32 len][bytes]...: the lengths' planes, then the bytes.
        let (mut p, mut count) = (0usize, 0u64);
        let mut lens = Vec::new();
        let mut bytes = Vec::new();
        while p < values.len() {
            let n = u32::from_le_bytes(values.get(p..p + 4)?.try_into().unwrap()) as usize;
            lens.extend_from_slice(&values[p..p + 4]);
            bytes.extend_from_slice(values.get(p + 4..p + 4 + n)?);
            p += 4 + n;
            count += 1;
        }
        recipe.push(M_BYTE_ARRAYS);
        put_varint(&mut recipe, prefix.len() as u64);
        put_varint(&mut recipe, count);
        planes(&lens, 4, &mut out);
        out.extend_from_slice(&bytes);
        return Some((recipe, out));
    }
    if values.len() % width != 0 {
        return None;
    }
    let is_int = matches!(chunk.physical_type, 1 | 2);
    // The candidates, the page as it is among them, the cheapest by
    // the fast level's output.
    let mut best: (usize, u8, u32, Vec<u8>) = (cost(values), M_RAW, 0, values.to_vec());
    let mut consider = |kind: u8, d: u32, transformed: &[u8]| {
        let mut p = Vec::with_capacity(transformed.len());
        planes(transformed, width, &mut p);
        let c = cost(&p);
        if c < best.0 {
            best = (c, kind, d, p);
        }
    };
    consider(M_PLANES, 0, values);
    if is_int {
        // Counters and times in their unit when every value is a
        // multiple of a power of ten (microseconds that are whole
        // seconds), as planes and as deltas.
        let d = common_power(values, width);
        let base = if d > 0 { scaled(values, width, d, true) } else { values.to_vec() };
        if d > 0 {
            consider(M_PLANES, d, &base);
        }
        consider(M_DELTA_PLANES, d, &deltas(&base, width));
    }
    if chunk.physical_type == 5 {
        for d in 0..=4 {
            if let Some(ints) = decimal_ints(values, d) {
                consider(M_DECIMAL, d, &deltas(&ints, 8));
                break;
            }
        }
    }
    let (_, kind, d, p) = best;
    recipe.push(kind);
    put_varint(&mut recipe, prefix.len() as u64);
    put_varint(&mut recipe, (values.len() / width) as u64);
    recipe.push(width as u8);
    recipe.push(d as u8);
    out.extend_from_slice(&p);
    Some((recipe, out))
}

/// The page's bytes from `model`'s recipe and modeled bytes.
pub fn unmodel(recipe: &[u8], modeled: &[u8]) -> Option<Vec<u8>> {
    let mut p = 0usize;
    let kind = *recipe.get(p)?;
    p += 1;
    let prefix_len = get_varint(recipe, &mut p)? as usize;
    let count = get_varint(recipe, &mut p)? as usize;
    let (prefix, body) = modeled.split_at(prefix_len.min(modeled.len()));
    let mut out = Vec::with_capacity(modeled.len() + 16);
    out.extend_from_slice(prefix);
    if kind == M_BYTE_ARRAYS {
        let (lens, bytes) = body.split_at(count.checked_mul(4)?.min(body.len()));
        let mut l = Vec::with_capacity(lens.len());
        unplanes(lens, 4, &mut l);
        let mut q = 0usize;
        for v in l.chunks_exact(4) {
            let n = u32::from_le_bytes(v.try_into().unwrap()) as usize;
            out.extend_from_slice(v);
            out.extend_from_slice(bytes.get(q..q + n)?);
            q += n;
        }
        return (q == bytes.len()).then_some(out);
    }
    let width = *recipe.get(p)? as usize;
    let d = *recipe.get(p + 1)? as u32;
    if body.len() != count.checked_mul(width)? {
        return None;
    }
    let mut values = Vec::with_capacity(body.len());
    unplanes(body, width, &mut values);
    if kind == M_INDICES {
        let bit_width = d;
        let mut decoded = Vec::with_capacity(count);
        for v in values.chunks_exact(width) {
            let mut x = 0u32;
            for (k, &b) in v.iter().enumerate() {
                x |= (b as u32) << (8 * k);
            }
            decoded.push(x);
        }
        out.push(bit_width as u8);
        out.extend_from_slice(&rle_encode(&decoded, bit_width));
        return Some(out);
    }
    let values = match kind {
        M_RAW => body.to_vec(),
        M_PLANES => {
            if d > 0 { scaled(&values, width, d, false) } else { values }
        }
        M_DELTA_PLANES => {
            let v = undeltas(&values, width);
            if d > 0 { scaled(&v, width, d, false) } else { v }
        }
        M_DECIMAL => undecimal(&undeltas(&values, 8), d),
        _ => return None,
    };
    out.extend_from_slice(&values);
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn models_come_back_exactly() {
        let chunk = |ty: i32| Chunk { row_group: 0, column: 0, codec: Codec::Snappy, physical_type: ty, max_def_level: 1, max_rep_level: 0, encodings: vec![], num_values: 0, start: 0, compressed_len: 0, uncompressed_len: 0 };
        let page = Page { header_at: 0, body_at: 0, compressed_len: 0, uncompressed_len: 0, kind: 0, num_values: 0, encoding: 0, num_nulls: 0, v2_levels_len: 0, v2_compressed: true };
        // A definition-level block of 3 bytes, then values.
        let levels = [3u8, 0, 0, 0, 0x11, 0x22, 0x33];
        let mut ints = levels.to_vec();
        for i in 0..500i64 { ints.extend_from_slice(&(1_700_000_000_000 + i * 60_000).to_le_bytes()); }
        let mut money = levels.to_vec();
        for i in 0..500 { money.extend_from_slice(&((i as f64) * 0.25 + 12.5).to_le_bytes()); }
        let mut texts = levels.to_vec();
        for i in 0..300u32 { let s = format!("name{}", i % 7); texts.extend_from_slice(&(s.len() as u32).to_le_bytes()); texts.extend_from_slice(s.as_bytes()); }
        for (raw, ty, kind) in [(&ints, 2, M_DELTA_PLANES), (&money, 5, M_DECIMAL), (&texts, 6, M_BYTE_ARRAYS)] {
            let (recipe, modeled) = model(raw, &chunk(ty), &page).unwrap();
            assert_eq!(recipe[0], kind, "type {ty}");
            assert_eq!(unmodel(&recipe, &modeled).unwrap(), *raw, "type {ty}");
            if ty != 6 {
                assert!(bits(&modeled) < bits(raw), "type {ty}: {} vs {} bits", bits(&modeled), bits(raw));
            }
        }
        assert!(model(&levels, &chunk(2), &page).is_none(), "no values");
        // Dictionary indices: runs of repeats and literals, a partial last group.
        let mut idx: Vec<u32> = Vec::new();
        for i in 0..700u32 { idx.push(if i % 3 == 0 { 0 } else if i % 50 < 20 { 3 } else { (i * 7) % 11 }); }
        let stream = rle_encode(&idx, 4);
        assert_eq!(rle_decode(&stream, 4, idx.len()).unwrap()[..idx.len()], idx[..]);
        let mut page_bytes = vec![0u8, 0, 0, 0];
        let def = rle_encode(&vec![1u32; 700], 1);
        page_bytes[..4].copy_from_slice(&(def.len() as u32).to_le_bytes());
        page_bytes.extend_from_slice(&def);
        page_bytes.push(4);
        page_bytes.extend_from_slice(&stream);
        let mut pg = page.clone();
        pg.encoding = 8;
        pg.num_values = 700;
        let (recipe, modeled) = model(&page_bytes, &chunk(5), &pg).unwrap();
        assert_eq!(recipe[0], M_INDICES);
        assert_eq!(unmodel(&recipe, &modeled).unwrap(), page_bytes);
    }

    #[test]
    fn the_thrift_reader_walks_fields_and_skips_the_rest() {
        // A struct: field 1 i32 = 300 (zigzag 600), field 2 string "ab",
        // field 4 list<i32> [1, -1], field 5 double, field 6 bool true, stop.
        let b = [0x15, 0xd8, 0x04, 0x18, 2, b'a', b'b', 0x29, 0x25, 2, 1, 0x17, 0, 0, 0, 0, 0, 0, 0, 0, 0x11, 0];
        let mut r = Reader { b: &b, p: 0 };
        let mut seen = Vec::new();
        r.fields(|r, fid, t| {
            if fid == 1 {
                seen.push(r.zigzag()?);
                return Some(true);
            }
            if fid == 4 {
                let (n, _) = r.list()?;
                for _ in 0..n {
                    seen.push(r.zigzag()?);
                }
                return Some(true);
            }
            seen.push(-100 - fid as i64 * (t as i64 + 1));
            Some(false)
        }).unwrap();
        assert_eq!(seen, vec![300, -100 - 2 * 9, 1, -1, -100 - 5 * 8, -100 - 6 * 2]);
        assert_eq!(r.p, b.len());
        assert!(!is_parquet(&b));
    }
}
