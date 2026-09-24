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
    let mut leaf_types: Vec<i32> = Vec::new();
    let mut chunks = Vec::new();
    let mut created_by = String::new();
    r.fields(|r, fid, t| {
        match (fid, t) {
            (2, 9) => {
                // schema: the leaves, those with a physical type, in order.
                let (n, _) = r.list()?;
                for _ in 0..n {
                    let mut ty = None;
                    r.fields(|r, fid, t| {
                        if fid == 1 && (t == 5 || t == 6) {
                            ty = Some(r.zigzag()? as i32);
                            return Some(true);
                        }
                        Some(false)
                    })?;
                    if let Some(ty) = ty {
                        leaf_types.push(ty);
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
                            let mut chunk = Chunk { row_group: g, column: c, codec: Codec::Uncompressed, physical_type: leaf_types.get(c).copied().unwrap_or(-1), encodings: Vec::new(), num_values: 0, start: 0, compressed_len: 0, uncompressed_len: 0 };
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
        let mut page = Page { header_at: p, body_at: 0, compressed_len: 0, uncompressed_len: 0, kind: -1, v2_levels_len: 0, v2_compressed: true };
        r.fields(|r, fid, t| {
            match (fid, t) {
                (1, 5) => page.kind = r.zigzag()? as i32,
                (2, 5) => page.uncompressed_len = r.zigzag()? as usize,
                (3, 5) => page.compressed_len = r.zigzag()? as usize,
                (8, 12) => {
                    r.fields(|r, fid, t| {
                        match (fid, t) {
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

#[cfg(test)]
mod tests {
    use super::*;

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
