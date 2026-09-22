//! Deflate containers opened: gzip, zip (so also .docx, .xlsx, .pptx,
//! .jar, .apk, .odt), zlib streams, PNG, PDF and tar, nested in one
//! another (a tar of gzipped logs, a zip of PNGs, a deck's JPEGs under
//! their deflate entries, a tar.gz of all of it). The deflate streams inside
//! are decoded to their plain text by `preflate`, which also records
//! what it takes to re-encode each bit for bit; the plain text then
//! takes whatever the caller asked for (a level, record mode, the cold
//! level, a base), so a gzipped log costs what the log costs and a
//! spreadsheet what its XML costs. Envelope:
//!
//!   "GLYDDEFL" original_len, recipe_len (varints), the recipe, then
//!   the inner Glyd stream of the plain text (any format: units of a
//!   level, records, cold, a base envelope).
//!
//! The recipe is a list of segments that lay the object end to end:
//! bytes kept verbatim (headers, directories, stored entries, chunks
//! that are not data), a deflate stream (preflate's corrections and
//! the length of its plain text, the next slice of the plain text),
//! or a PNG's image stream (a zlib stream re-cut into its IDAT chunks).
//! An object that does not re-encode bit for bit (checked before it is
//! used) is left as it is.

use crate::record::{get_varint, put_varint};
use preflate_rs::{preflate_whole_deflate_stream, recreate_whole_deflate_stream, PreflateConfig};

pub(crate) const MAGIC: &[u8; 8] = b"GLYDDEFL";
/// Set while an object is compressed closed for the comparison in
/// `wrap`, so the level's own units do not open what they start with
/// (a unit of a container starts with its magic). Process-wide: a
/// concurrent compression in that moment keeps its container closed,
/// which costs bytes, never correctness.
static CLOSED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// The plain text is held in memory; beyond this an object is left alone.
const PLAIN_LIMIT: usize = 8 << 30;
const VERBATIM: u8 = 0;
const DEFLATE: u8 = 1;
const PNG_IMAGE: u8 = 2;
/// A JPEG stored inside a container (a zip of photos): its Lepton
/// stream, in the recipe.
const JPEG: u8 = 3;
/// A JPEG deflated inside a container (an Office document's pictures):
/// the plain text holds its Lepton stream, from which the JPEG and
/// then the deflate stream are recreated.
const DEFLATE_JPEG: u8 = 4;
/// A container deflated inside a container (an Office document's PNG
/// screenshots, a tar.gz of images): the plain text holds the inner
/// container's own plain text, and the recipe a recipe for it, from
/// which the inner container and then the deflate stream are
/// recreated.
const DEFLATE_NESTED: u8 = 5;
/// A container stored as it is inside another (a tar of PNGs, a zip
/// holding a jar): a recipe for it, its plain text in the plain text.
const NESTED: u8 = 6;
/// Containers inside containers are opened this deep.
const MAX_DEPTH: u32 = 4;

/// An object opened: its plain text and the recipe to close it.
pub struct Opened {
    pub plain: Vec<u8>,
    pub recipe: Vec<u8>,
}

/// Whether `input` starts like a container worth opening (a JPEG
/// counts: it is transcoded by `jpeg` through the same hooks).
pub fn is_container(input: &[u8]) -> bool {
    #[cfg(feature = "jpeg")]
    if crate::jpeg::is_jpeg(input) {
        return true;
    }
    is_gzip(input) || is_zip(input) || is_png(input) || is_pdf(input) || is_tar(input) || is_zlib(input)
}

/// A ustar header at the start, with a size that parses.
fn is_tar(input: &[u8]) -> bool {
    input.len() >= 1024 && &input[257..262] == b"ustar" && tar_size(input, 0).is_some()
}

/// The size of the tar entry whose header is at `at`.
fn tar_size(input: &[u8], at: usize) -> Option<usize> {
    let field = input.get(at + 124..at + 136)?;
    let digits: Vec<u8> = field.iter().copied().take_while(|&b| b != 0 && b != b' ').collect();
    if digits.is_empty() || digits.iter().any(|b| !(b'0'..=b'7').contains(b)) {
        return None;
    }
    let mut n = 0usize;
    for &d in &digits {
        n = n.checked_mul(8)?.checked_add((d - b'0') as usize)?;
    }
    Some(n)
}

fn is_pdf(input: &[u8]) -> bool {
    input.len() >= 64 && input.starts_with(b"%PDF-")
}

fn is_gzip(input: &[u8]) -> bool {
    input.len() >= 18 && input[0] == 0x1f && input[1] == 0x8b && input[2] == 8
}

fn is_zip(input: &[u8]) -> bool {
    input.len() >= 30 && input.starts_with(b"PK\x03\x04")
}

fn is_png(input: &[u8]) -> bool {
    input.len() >= 8 + 12 + 12 && input.starts_with(b"\x89PNG\r\n\x1a\n")
}

/// A zlib header: method 8, a window of 32 KB or less, no preset
/// dictionary, the check bits right.
fn is_zlib(input: &[u8]) -> bool {
    input.len() >= 6 + 4 && input[0] & 0x0f == 8 && input[0] >> 4 <= 7 && input[1] & 0x20 == 0 && (input[0] as u32 * 256 + input[1] as u32) % 31 == 0
}

/// The recipe under construction and the plain text it points into.
struct Builder {
    plain: Vec<u8>,
    recipe: Vec<u8>,
    segments: u64,
    /// A verbatim run not yet written, so adjacent ones merge.
    verbatim: (usize, usize),
    config: PreflateConfig,
    /// How many containers this one is inside.
    depth: u32,
    /// Whether anything but verbatim bytes went in (a tar of JPEGs has
    /// no plain text and still opens).
    opened: bool,
}

impl Builder {
    fn at_depth(depth: u32) -> Builder {
        Builder { plain: Vec::new(), recipe: Vec::new(), segments: 0, verbatim: (0, 0), config: PreflateConfig { plain_text_limit: PLAIN_LIMIT, verify_compression: true, ..Default::default() }, depth, opened: false }
    }

    /// `data` opened as a container of its own, when it is one and
    /// this builder is not too deep.
    fn nested(&self, data: &[u8]) -> Option<Opened> {
        if self.depth + 1 >= MAX_DEPTH || !is_container(data) {
            return None;
        }
        #[cfg(feature = "jpeg")]
        if crate::jpeg::is_jpeg(data) {
            return None;
        }
        open_at(data, self.depth + 1)
    }

    fn verbatim(&mut self, input: &[u8], from: usize, to: usize) {
        if from >= to {
            return;
        }
        if self.verbatim.1 == from {
            self.verbatim.1 = to;
        } else {
            self.flush_verbatim(input);
            self.verbatim = (from, to);
        }
    }

    fn flush_verbatim(&mut self, input: &[u8]) {
        let (from, to) = self.verbatim;
        if from < to {
            self.recipe.push(VERBATIM);
            put_varint(&mut self.recipe, (to - from) as u64);
            self.recipe.extend_from_slice(&input[from..to]);
            self.segments += 1;
        }
        self.verbatim = (to, to);
    }

    /// The deflate stream at `at`: its compressed length when it opens.
    /// A stream whose corrections come to more than a quarter of it
    /// (an encoder preflate predicts badly) is not worth opening.
    fn deflate(&mut self, input: &[u8], at: usize) -> Option<usize> {
        let (result, text) = preflate_whole_deflate_stream(input.get(at..)?, &self.config).ok()?;
        if self.plain.len() + text.text().len() > PLAIN_LIMIT || result.corrections.len() * 4 > result.compressed_size {
            return None;
        }
        self.flush_verbatim(input);
        // A JPEG under the deflate: its Lepton stream stands in the
        // plain text for it.
        #[cfg(feature = "jpeg")]
        if crate::jpeg::is_jpeg(text.text()) {
            if let Some(lepton) = crate::jpeg::transcode(text.text()) {
                if lepton.len() + 16 < text.text().len() {
                    self.recipe.push(DEFLATE_JPEG);
                    put_varint(&mut self.recipe, result.corrections.len() as u64);
                    self.recipe.extend_from_slice(&result.corrections);
                    put_varint(&mut self.recipe, lepton.len() as u64);
                    self.plain.extend_from_slice(&lepton);
                    self.segments += 1;
        self.opened = true;
                    self.verbatim = (at + result.compressed_size, at + result.compressed_size);
                    return Some(result.compressed_size);
                }
            }
        }
        // A container under the deflate (a PNG, a tar of logs): opened
        // in a recipe of its own, its plain text standing in the plain
        // text for it.
        if let Some(inner) = self.nested(text.text()) {
            if self.plain.len() + inner.plain.len() <= PLAIN_LIMIT {
                self.recipe.push(DEFLATE_NESTED);
                put_varint(&mut self.recipe, result.corrections.len() as u64);
                self.recipe.extend_from_slice(&result.corrections);
                put_varint(&mut self.recipe, inner.recipe.len() as u64);
                self.recipe.extend_from_slice(&inner.recipe);
                put_varint(&mut self.recipe, inner.plain.len() as u64);
                self.plain.extend_from_slice(&inner.plain);
                self.segments += 1;
        self.opened = true;
                self.verbatim = (at + result.compressed_size, at + result.compressed_size);
                return Some(result.compressed_size);
            }
        }
        self.recipe.push(DEFLATE);
        put_varint(&mut self.recipe, result.corrections.len() as u64);
        self.recipe.extend_from_slice(&result.corrections);
        put_varint(&mut self.recipe, text.text().len() as u64);
        self.plain.extend_from_slice(text.text());
        self.segments += 1;
        self.opened = true;
        self.verbatim = (at + result.compressed_size, at + result.compressed_size);
        Some(result.compressed_size)
    }

    fn finish(mut self, input: &[u8]) -> Option<Opened> {
        self.flush_verbatim(input);
        if !self.opened {
            return None;
        }
        let mut recipe = Vec::with_capacity(self.recipe.len() + 8);
        put_varint(&mut recipe, self.segments);
        recipe.extend_from_slice(&self.recipe);
        Some(Opened { plain: self.plain, recipe })
    }
}

/// The plain text and recipe of a container, when every stream in it
/// re-encodes bit for bit; `None` for anything else.
pub fn open(input: &[u8]) -> Option<Opened> {
    open_at(input, 0)
}

fn open_at(input: &[u8], depth: u32) -> Option<Opened> {
    let b = Builder::at_depth(depth);
    if is_gzip(input) {
        open_gzip(input, b)
    } else if is_zip(input) {
        open_zip(input, b)
    } else if is_png(input) {
        open_png(input, b)
    } else if is_pdf(input) {
        open_pdf(input, b)
    } else if is_tar(input) {
        open_tar(input, b)
    } else if is_zlib(input) {
        open_zlib(input, b)
    } else {
        None
    }
}

/// Entries in order, each a 512-byte header and data padded to 512:
/// the data of a regular file opened in its own way (a gzip member, a
/// picture, an archive), everything else kept.
fn open_tar(input: &[u8], mut b: Builder) -> Option<Opened> {
    let mut at = 0usize;
    while at + 512 <= input.len() && &input[at + 257..at + 262] == b"ustar" {
        let Some(size) = tar_size(input, at) else { break };
        let data = at + 512;
        let end = data.checked_add(size)?;
        if end > input.len() {
            break;
        }
        let padded = data + (size + 511) / 512 * 512;
        b.verbatim(input, at, data);
        if input[at + 156] == b'0' || input[at + 156] == 0 {
            b.stored(input, data, end);
        } else {
            b.verbatim(input, data, end);
        }
        b.verbatim(input, end, padded.min(input.len()));
        at = padded;
    }
    b.verbatim(input, at, input.len());
    b.finish(input)
}

/// The end of the gzip header starting at `at`.
fn gzip_header_end(input: &[u8], at: usize) -> Option<usize> {
    let flg = *input.get(at + 3)?;
    let mut p = at + 10;
    if flg & 4 != 0 {
        let xlen = u16::from_le_bytes(input.get(p..p + 2)?.try_into().unwrap()) as usize;
        p += 2 + xlen;
    }
    for bit in [8u8, 16] {
        if flg & bit != 0 {
            p += input.get(p..)?.iter().position(|&b| b == 0)? + 1;
        }
    }
    if flg & 2 != 0 {
        p += 2;
    }
    if p <= input.len() { Some(p) } else { None }
}

/// Members back to back, each a header, a deflate stream and an 8-byte
/// trailer; whatever follows the last is kept.
fn open_gzip(input: &[u8], mut b: Builder) -> Option<Opened> {
    let mut at = 0usize;
    while is_gzip(&input[at..]) {
        let hend = gzip_header_end(input, at)?;
        b.verbatim(input, at, hend);
        let n = b.deflate(input, hend)?;
        let dend = hend + n;
        input.get(dend..dend + 8)?;
        b.verbatim(input, dend, dend + 8);
        at = dend + 8;
    }
    b.verbatim(input, at, input.len());
    b.finish(input)
}

/// Local entries in order: a header kept, the data opened when it is
/// deflate (method 8), kept when it is stored or anything else; then
/// the central directory and the rest, kept. A data descriptor after
/// an entry is found by looking for the next entry's signature.
fn open_zip(input: &[u8], mut b: Builder) -> Option<Opened> {
    let mut at = 0usize;
    let le16 = |p: usize| input.get(p..p + 2).map(|s| u16::from_le_bytes(s.try_into().unwrap()) as usize);
    let le32 = |p: usize| input.get(p..p + 4).map(|s| u32::from_le_bytes(s.try_into().unwrap()) as usize);
    while input.get(at..at + 4) == Some(b"PK\x03\x04") {
        let (flags, method, mut csize, usize) = (le16(at + 6)?, le16(at + 8)?, le32(at + 18)?, le32(at + 22)?);
        let (nlen, xlen) = (le16(at + 26)?, le16(at + 28)?);
        let data = at + 30 + nlen + xlen;
        if data > input.len() {
            break;
        }
        // A zip64 entry's sizes are in its extra field (id 1), the
        // uncompressed one first when the header has both as 0xFFFFFFFF.
        if csize == 0xFFFF_FFFF {
            let (mut p, xend) = (at + 30 + nlen, data);
            while p + 4 <= xend {
                let (id, len) = (le16(p)?, le16(p + 2)?);
                if id == 1 {
                    let q = p + 4 + if usize == 0xFFFF_FFFF { 8 } else { 0 };
                    csize = input.get(q..q + 8).map(|s| u64::from_le_bytes(s.try_into().unwrap()) as usize)?;
                    break;
                }
                p += 4 + len;
            }
        }
        b.verbatim(input, at, data);
        at = data;
        let mut opened_to = data;
        // An entry whose size is only in a data descriptor after it,
        // and whose stream does not open, ends where the next entry's
        // signature is found (a false find inside the data costs only
        // the opening of what follows, never the bytes).
        let next_signature = |from: usize| (from..input.len().saturating_sub(3)).find(|&p| &input[p..p + 4] == b"PK\x03\x04").unwrap_or(input.len());
        let sized = flags & 8 == 0 && csize != 0xFFFF_FFFF;
        let end = if method == 8 && flags & 1 == 0 {
            match b.deflate(input, data) {
                Some(n) => {
                    opened_to = data + n;
                    data + n
                }
                None if sized => data + csize,
                None => next_signature(data),
            }
        } else if sized {
            data + csize
        } else {
            next_signature(data)
        };
        if end > input.len() {
            break;
        }
        // Stored data (a picture, a nested archive) opened in its own
        // way; a stream that did not open, kept.
        if method == 0 && opened_to == data {
            b.stored(input, data, end);
        } else {
            b.verbatim(input, opened_to, end);
        }
        // The next entry: right here, or past a data descriptor.
        let next = (end..(end + 32).min(input.len().saturating_sub(3))).find(|&p| &input[p..p + 4] == b"PK\x03\x04").unwrap_or(end);
        b.verbatim(input, end, next);
        at = next;
    }
    b.verbatim(input, at, input.len());
    b.finish(input)
}

/// A PDF's streams: every `stream` keyword whose data starts like a
/// zlib stream (a /FlateDecode filter alone; a predictor changes only
/// what the bytes mean, not the stream) and whose stream ends before
/// an `endstream`, opened; everything else kept. Found by scanning,
/// so a /Length given by reference, object streams and cross-reference
/// streams need no parsing.
fn open_pdf(input: &[u8], mut b: Builder) -> Option<Opened> {
    let mut at = 0usize;
    let mut scan = 0usize;
    while scan + 6 <= input.len() {
        let Some(k) = input[scan..].windows(6).position(|w| w == b"stream") else { break };
        let key = scan + k;
        scan = key + 6;
        // `stream` of `endstream`, or not followed by a line end.
        if key >= 3 && &input[key - 3..key] == b"end" {
            continue;
        }
        let data = if input.get(scan..scan + 2) == Some(b"\r\n") { scan + 2 } else if input.get(scan) == Some(&b'\n') { scan + 1 } else { continue };
        if data <= at || !is_zlib(&input[data..]) {
            continue;
        }
        let mut probe = Builder::at_depth(b.depth);
        let Some(n) = probe.deflate(input, data + 2) else { continue };
        let end = data + 2 + n + 4;
        // What follows must be the end of the stream.
        let mut p = end;
        while p < input.len() && matches!(input[p], b'\r' | b'\n' | b' ' | b'\t') {
            p += 1;
        }
        if input.get(p..p + 9) != Some(b"endstream") {
            continue;
        }
        b.verbatim(input, at, data + 2);
        b.deflate(input, data + 2)?;
        b.verbatim(input, data + 2 + n, end);
        at = end;
        scan = p + 9;
    }
    b.verbatim(input, at, input.len());
    b.finish(input)
}

/// A zlib stream: 2-byte header, deflate, 4-byte Adler-32; the rest kept.
fn open_zlib(input: &[u8], mut b: Builder) -> Option<Opened> {
    b.verbatim(input, 0, 2);
    let n = b.deflate(input, 2)?;
    b.verbatim(input, 2 + n, input.len());
    b.finish(input)
}

/// Chunks in order; the IDAT chunks' data, joined, is one zlib stream,
/// opened as a `PNG_IMAGE` segment that remembers how to cut it back
/// into chunks (their lengths and CRCs); every other chunk is kept.
fn open_png(input: &[u8], mut b: Builder) -> Option<Opened> {
    b.png(input, 0, input.len())?;
    b.finish(input)
}

impl Builder {
    /// The PNG at `input[from..to]` as segments; `None` leaves the
    /// builder as it was.
    fn png(&mut self, input: &[u8], from: usize, to: usize) -> Option<()> {
        let be32 = |p: usize| input.get(p..p + 4).filter(|_| p + 4 <= to).map(|s| u32::from_be_bytes(s.try_into().unwrap()) as usize);
        let mut at = from + 8;
        let mut idat: Vec<(usize, usize)> = Vec::new(); // (data start, len)
        let mut image = Vec::new();
        let mut first_idat = None;
        while let (Some(len), Some(kind)) = (be32(at), input.get(at + 4..at + 8)) {
            let chunk_end = at + 12 + len;
            if chunk_end > to {
                break;
            }
            if kind == b"IDAT" {
                if first_idat.is_none() {
                    first_idat = Some(at);
                }
                idat.push((at + 8, len));
                image.extend_from_slice(&input[at + 8..at + 8 + len]);
            } else if first_idat.is_some() {
                break;
            }
            at = chunk_end;
        }
        let start = first_idat?;
        let last_end = idat.last().map(|(s, l)| s + l + 4)?;
        if !is_zlib(&image) {
            return None;
        }
        let (result, text) = preflate_whole_deflate_stream(&image[2..], &self.config).ok()?;
        if 2 + result.compressed_size + 4 != image.len() || self.plain.len() + text.text().len() > PLAIN_LIMIT || result.corrections.len() * 4 > result.compressed_size {
            return None;
        }
        self.verbatim(input, from, start);
        self.flush_verbatim(input);
        self.recipe.push(PNG_IMAGE);
        self.recipe.extend_from_slice(&image[..2]);
        put_varint(&mut self.recipe, result.corrections.len() as u64);
        self.recipe.extend_from_slice(&result.corrections);
        put_varint(&mut self.recipe, text.text().len() as u64);
        self.recipe.extend_from_slice(&image[image.len() - 4..]);
        put_varint(&mut self.recipe, idat.len() as u64);
        for &(s, l) in &idat {
            put_varint(&mut self.recipe, l as u64);
            self.recipe.extend_from_slice(&input[s + l..s + l + 4]);
        }
        self.plain.extend_from_slice(text.text());
        self.segments += 1;
        self.opened = true;
        self.verbatim = (last_end, last_end);
        self.verbatim(input, last_end, to);
        Some(())
    }

    /// A stored entry's data at `input[from..to]`: a JPEG transcoded,
    /// a PNG or gzip member opened, anything else kept.
    fn stored(&mut self, input: &[u8], from: usize, to: usize) {
        let data = &input[from..to];
        #[cfg(feature = "jpeg")]
        if crate::jpeg::is_jpeg(data) {
            if let Some(lepton) = crate::jpeg::transcode(data) {
                if lepton.len() + 16 < data.len() {
                    self.flush_verbatim(input);
                    self.recipe.push(JPEG);
                    put_varint(&mut self.recipe, lepton.len() as u64);
                    self.recipe.extend_from_slice(&lepton);
                    self.segments += 1;
        self.opened = true;
                    self.verbatim = (to, to);
                    return;
                }
            }
        }
        if let Some(inner) = self.nested(data) {
            if self.plain.len() + inner.plain.len() <= PLAIN_LIMIT {
                self.flush_verbatim(input);
                self.recipe.push(NESTED);
                put_varint(&mut self.recipe, inner.recipe.len() as u64);
                self.recipe.extend_from_slice(&inner.recipe);
                put_varint(&mut self.recipe, inner.plain.len() as u64);
                self.plain.extend_from_slice(&inner.plain);
                self.segments += 1;
        self.opened = true;
                self.verbatim = (to, to);
                return;
            }
        }
        self.verbatim(input, from, to);
    }
}

/// The object back from its plain text and recipe.
pub fn close(recipe: &[u8], plain: &[u8]) -> Option<Vec<u8>> {
    let mut pos = 0usize;
    let n = get_varint(recipe, &mut pos).ok()?;
    let mut out = Vec::with_capacity(plain.len() / 4 + recipe.len());
    let mut at = 0usize;
    let take = |pos: &mut usize| -> Option<&[u8]> {
        let len = get_varint(recipe, pos).ok()? as usize;
        let s = recipe.get(*pos..pos.checked_add(len)?)?;
        *pos += len;
        Some(s)
    };
    let fixed = |pos: &mut usize, n: usize| -> Option<&[u8]> {
        let s = recipe.get(*pos..*pos + n)?;
        *pos += n;
        Some(s)
    };
    for _ in 0..n {
        match *recipe.get(pos)? {
            VERBATIM => {
                pos += 1;
                out.extend_from_slice(take(&mut pos)?);
            }
            DEFLATE => {
                pos += 1;
                let corrections = take(&mut pos)?;
                let plen = get_varint(recipe, &mut pos).ok()? as usize;
                let text = plain.get(at..at.checked_add(plen)?)?;
                out.extend_from_slice(&recreate_whole_deflate_stream(text, corrections).ok()?);
                at += plen;
            }
            PNG_IMAGE => {
                pos += 1;
                let header = fixed(&mut pos, 2)?;
                let corrections = take(&mut pos)?;
                let plen = get_varint(recipe, &mut pos).ok()? as usize;
                let adler = fixed(&mut pos, 4)?;
                let text = plain.get(at..at.checked_add(plen)?)?;
                let mut image = header.to_vec();
                image.extend_from_slice(&recreate_whole_deflate_stream(text, corrections).ok()?);
                image.extend_from_slice(adler);
                at += plen;
                let chunks = get_varint(recipe, &mut pos).ok()?;
                let mut off = 0usize;
                for _ in 0..chunks {
                    let len = get_varint(recipe, &mut pos).ok()? as usize;
                    let crc = fixed(&mut pos, 4)?;
                    out.extend_from_slice(&(len as u32).to_be_bytes());
                    out.extend_from_slice(b"IDAT");
                    out.extend_from_slice(image.get(off..off + len)?);
                    out.extend_from_slice(crc);
                    off += len;
                }
                if off != image.len() {
                    return None;
                }
            }
            #[cfg(feature = "jpeg")]
            JPEG => {
                pos += 1;
                out.extend_from_slice(&crate::jpeg::restore(take(&mut pos)?)?);
            }
            DEFLATE_NESTED => {
                pos += 1;
                let corrections = take(&mut pos)?;
                let inner = take(&mut pos)?;
                let plen = get_varint(recipe, &mut pos).ok()? as usize;
                let object = close(inner, plain.get(at..at.checked_add(plen)?)?)?;
                out.extend_from_slice(&recreate_whole_deflate_stream(&object, corrections).ok()?);
                at += plen;
            }
            NESTED => {
                pos += 1;
                let inner = take(&mut pos)?;
                let plen = get_varint(recipe, &mut pos).ok()? as usize;
                out.extend_from_slice(&close(inner, plain.get(at..at.checked_add(plen)?)?)?);
                at += plen;
            }
            #[cfg(feature = "jpeg")]
            DEFLATE_JPEG => {
                pos += 1;
                let corrections = take(&mut pos)?;
                let plen = get_varint(recipe, &mut pos).ok()? as usize;
                let lepton = plain.get(at..at.checked_add(plen)?)?;
                let jpeg = crate::jpeg::restore(lepton)?;
                out.extend_from_slice(&recreate_whole_deflate_stream(&jpeg, corrections).ok()?);
                at += plen;
            }
            _ => return None,
        }
    }
    if at != plain.len() || pos != recipe.len() {
        return None;
    }
    Some(out)
}

/// The envelope's head: everything before the inner stream. The
/// recipe goes in compressed at the max level: the corrections of a
/// container's streams repeat each other (a PDF's duplicate fonts, a
/// jar's thousand small entries).
pub fn envelope(original_len: usize, recipe: &[u8], out: &mut Vec<u8>) {
    let mut packed = Vec::with_capacity(recipe.len() / 2 + 64);
    crate::compress_into_max(recipe, &mut packed);
    out.extend_from_slice(MAGIC);
    put_varint(out, original_len as u64);
    put_varint(out, packed.len() as u64);
    out.extend_from_slice(&packed);
}

/// (original length, recipe, inner stream) of an envelope.
pub(crate) fn parse(compressed: &[u8]) -> Option<(usize, Vec<u8>, &[u8])> {
    if compressed.len() < 10 || &compressed[..8] != MAGIC {
        return None;
    }
    let mut pos = 8usize;
    let original = get_varint(compressed, &mut pos).ok()? as usize;
    let rlen = get_varint(compressed, &mut pos).ok()? as usize;
    let packed = compressed.get(pos..pos.checked_add(rlen)?)?;
    let recipe = crate::decompress(packed).ok()?;
    Some((original, recipe, &compressed[pos + rlen..]))
}

/// `input` compressed as an opened container by `inner` when it is
/// one and that is smaller: the envelope, then `inner` on the plain
/// text. `false` otherwise.
pub(crate) fn wrap(input: &[u8], output: &mut Vec<u8>, inner: impl FnOnce(&[u8], &mut Vec<u8>), closed_level: impl FnOnce(&[u8], &mut Vec<u8>)) -> bool {
    if !is_container(input) || CLOSED.load(std::sync::atomic::Ordering::Relaxed) {
        return false;
    }
    #[cfg(feature = "jpeg")]
    if crate::jpeg::is_jpeg(input) {
        return crate::jpeg::wrap(input, output);
    }
    let Some(opened) = open(input) else { return false };
    // Opened against closed, both at the caller's level: a container
    // can hold repeats of whole streams (a PDF's fonts) that the level
    // finds as it is and a recipe of many corrections would hide.
    let mut wrapped = Vec::with_capacity(opened.plain.len() / 4 + opened.recipe.len() + 32);
    envelope(input.len(), &opened.recipe, &mut wrapped);
    inner(&opened.plain, &mut wrapped);
    let mut closed = Vec::with_capacity(input.len() + 64);
    CLOSED.store(true, std::sync::atomic::Ordering::Relaxed);
    closed_level(input, &mut closed);
    CLOSED.store(false, std::sync::atomic::Ordering::Relaxed);
    if wrapped.len() >= closed.len() {
        return false;
    }
    output.extend_from_slice(&wrapped);
    true
}

/// The original object of an envelope, its inner stream decoded by
/// `inner`; `None` when `compressed` is no envelope.
pub(crate) fn unwrap(compressed: &[u8], inner: impl FnOnce(&[u8]) -> crate::Result<Vec<u8>>) -> Option<crate::Result<Vec<u8>>> {
    #[cfg(feature = "jpeg")]
    if let Some(r) = crate::jpeg::unwrap(compressed) {
        return Some(r);
    }
    let (original, recipe, stream) = parse(compressed)?;
    Some(inner(stream).and_then(|plain| match close(&recipe, &plain) {
        Some(out) if out.len() == original => Ok(out),
        _ => Err(crate::CodecError::CorruptedBitstream("deflate envelope: the object does not close")),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::process::{Command, Stdio};

    fn run(cmd: &str, args: &[&str], stdin: &[u8]) -> Vec<u8> {
        let mut child = Command::new(cmd).args(args).stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().unwrap();
        let mut input = child.stdin.take().unwrap();
        let data = stdin.to_vec();
        let t = std::thread::spawn(move || input.write_all(&data));
        let out = child.wait_with_output().unwrap();
        t.join().unwrap().unwrap();
        out.stdout
    }

    fn gzip(data: &[u8], args: &[&str]) -> Vec<u8> {
        run("gzip", &[args, &["-c"]].concat(), data)
    }

    /// python3 makes the zlib stream, the zip and the PNG.
    fn python(script: &str, stdin: &[u8]) -> Vec<u8> {
        run("python3", &["-c", script], stdin)
    }

    fn text(n: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(n);
        let mut x = 7u64;
        while v.len() < n {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            v.extend_from_slice(format!("2026-09-21T10:{:02}:{:02}Z host{} GET /api/v1/items/{} 200 {}\n", (x >> 8) % 60, (x >> 16) % 60, x % 40, (x >> 24) % 5000, (x >> 40) % 9000).as_bytes());
        }
        v.truncate(n);
        v
    }

    fn round_trip(object: &[u8], what: &str) -> Vec<u8> {
        let opened = open(object).unwrap_or_else(|| panic!("{what}: opens"));
        assert_eq!(close(&opened.recipe, &opened.plain).unwrap(), object, "{what}: closes");
        let mut c = Vec::new();
        crate::compress_into_max(object, &mut c);
        assert!(c.len() <= object.len() + 256, "{what}: {} of {}", c.len(), object.len());
        assert_eq!(crate::decompress(&c).unwrap(), object, "{what}");
        assert_eq!(crate::decompressed_len(&c).unwrap(), object.len());
        let mut dst = vec![0u8; object.len()];
        assert_eq!(crate::decompress_into(&c, &mut dst).unwrap(), object.len());
        assert_eq!(crate::decompress_parallel(&c).unwrap(), object);
        let mut streamed = Vec::new();
        crate::decompress_stream(&c, |b| { streamed.extend_from_slice(b); Ok(()) }).unwrap();
        assert_eq!(streamed, object);
        opened.plain
    }

    #[test]
    fn gzip_objects_open_and_close_bit_for_bit() {
        let plain = text(3 << 20);
        for args in [&["-1"][..], &["-6"], &["-9"], &["-9", "-n"]] {
            let gz = gzip(&plain, args);
            assert_eq!(round_trip(&gz, "gzip"), plain);
            let mut c = Vec::new();
            crate::compress_into_max(&gz, &mut c);
            // Opened when that is smaller (gzip -1 here; the max level is
            // zstd -3's class and gzip -9 beats it on these short random
            // lines, so those stay closed), never larger than the gzip.
            assert_eq!(c.starts_with(MAGIC), args == ["-1"], "{args:?}: {} of {}", c.len(), gz.len());
        }
        // Two members, a trailing remainder, record mode and the cold
        // level on the content.
        let mut two = gzip(&plain[..1 << 20], &["-6"]);
        two.extend(gzip(&plain[1 << 20..], &["-3"]));
        two.extend_from_slice(b"tail bytes");
        assert_eq!(round_trip(&two, "two members"), plain);
        let mut c = Vec::new();
        crate::compress_records_into_max(&two, &mut c);
        assert_eq!(crate::decompress(&c).unwrap(), two);
        let mut c = Vec::new();
        crate::compress_into_cold(&two[..], &mut c);
        assert_eq!(crate::decompress(&c).unwrap(), two);
        // A base that is gzip too: the delta is between the plain texts.
        let mut later = plain.clone();
        later.extend_from_slice(b"one more line\n");
        let (a, b) = (gzip(&plain, &["-6"]), gzip(&later, &["-6"]));
        let mut d = Vec::new();
        crate::compress_with_base(&a, &b, &mut d, false);
        assert!(d.len() < b.len() / 4, "{} of {}", d.len(), b.len());
        assert_eq!(crate::decompress_with_base(&a, &d).unwrap(), b);
        // Not a container: untouched.
        let mut c = Vec::new();
        assert!(!wrap(b"\x1f\x8b\x08 but not really a gzip stream at all", &mut c, |_, _| {}, |_, _| {}));
        assert!(open(&plain).is_none());
    }

    #[test]
    fn zip_zlib_and_png_open_and_close_bit_for_bit() {
        let plain = text(1 << 20);
        // A zip of three entries: deflated, stored, deflated with a data
        // descriptor (as streaming writers make them), and a comment.
        let zip = python(r#"
import sys, zipfile, io
data = sys.stdin.buffer.read()
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w") as z:
    z.writestr("a.log", data[:400000], compress_type=zipfile.ZIP_DEFLATED)
    z.writestr("b.bin", bytes(range(256)) * 100, compress_type=zipfile.ZIP_STORED)
    zi = zipfile.ZipInfo("c.log"); zi.compress_type = zipfile.ZIP_DEFLATED
    with z.open(zi, "w", force_zip64=True) as f:
        f.write(data[400000:])
    zi = zipfile.ZipInfo("d.bin"); zi.compress_type = zipfile.ZIP_STORED
    with z.open(zi, "w", force_zip64=True) as f:
        f.write(bytes(range(256)) * 50)
    z.writestr("e.log", data[:100000], compress_type=zipfile.ZIP_DEFLATED)
    z.comment = b"a comment"
sys.stdout.buffer.write(buf.getvalue())
"#, &plain);
        assert!(is_zip(&zip));
        let opened_plain = round_trip(&zip, "zip");
        assert!(opened_plain.len() >= plain.len() + 100000 - 1, "the three deflated entries opened: {} of {}", opened_plain.len(), plain.len() + 100000);
        // A zlib stream, then a PNG whose image data is that stream in
        // 3 IDAT chunks of odd sizes.
        let zlib = python("import sys, zlib; sys.stdout.buffer.write(zlib.compress(sys.stdin.buffer.read(), 6))", &plain);
        assert!(is_zlib(&zlib));
        assert_eq!(round_trip(&zlib, "zlib"), plain);
        let png = python(r#"
import sys, zlib, struct
raw = sys.stdin.buffer.read()
def chunk(kind, data): return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
z = zlib.compress(raw, 9)
out = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1000, len(raw)//4001, 8, 0, 0, 0, 0)) + chunk(b"tEXt", b"Comment\x00made for a test")
cuts = [0, 7777, 7777 + 100000, len(z)]
for i in range(3): out += chunk(b"IDAT", z[cuts[i]:cuts[i+1]])
out += chunk(b"IEND", b"")
sys.stdout.buffer.write(out)
"#, &plain);
        assert!(is_png(&png));
        assert_eq!(round_trip(&png, "png"), plain);
        // A PDF: two Flate streams (one with a /Length by reference),
        // one uncompressed stream, an object stream.
        let pdf = python(r#"
import sys, zlib
raw = sys.stdin.buffer.read()
a, b, c = raw[:300000], raw[300000:600000], raw[600000:]
parts = [b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n"]
def obj(n, body): parts.append(f"{n} 0 obj\n".encode() + body + b"\nendobj\n")
za = zlib.compress(a, 6)
obj(1, b"<< /Type /Catalog /Pages 2 0 R >>")
obj(2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
obj(3, b"<< /Type /Page /Parent 2 0 R /Contents 4 0 R >>")
obj(4, b"<< /Length " + str(len(za)).encode() + b" /Filter /FlateDecode >>\nstream\r\n" + za + b"\r\nendstream")
zb = zlib.compress(b, 9)
obj(5, b"<< /Length 6 0 R /Filter /FlateDecode >>\nstream\n" + zb + b"\nendstream")
obj(6, str(len(zb)).encode())
obj(7, b"<< /Length " + str(len(c)).encode() + b" >>\nstream\n" + c + b"\nendstream")
zc = zlib.compress(b"8 0 9 20 << /A 1 >> << /B 2 >>", 6)
obj(10, b"<< /Type /ObjStm /N 2 /First 8 /Length " + str(len(zc)).encode() + b" /Filter /FlateDecode >>\nstream\n" + zc + b"\nendstream")
parts.append(b"trailer\n<< /Root 1 0 R >>\n%%EOF\n")
sys.stdout.buffer.write(b"".join(parts))
"#, &plain);
        // A zip with pictures stored (as Office documents hold them): a
        // PNG opened and a JPEG transcoded inside.
        let jpeg = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/tiny.jpg")).unwrap();
        let mut pictures = png.clone();
        pictures.push(0);
        pictures.extend_from_slice(&jpeg);
        let picture_zip = python(&format!(r#"
import sys, zipfile, io
blob = sys.stdin.buffer.read()
png, jpeg = blob[:{}], blob[{}:]
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w") as z:
    z.writestr("word/document.xml", b"<w:document>" + b"<w:p>hello</w:p>" * 2000 + b"</w:document>", compress_type=zipfile.ZIP_DEFLATED)
    z.writestr("word/media/image1.png", png, compress_type=zipfile.ZIP_STORED)
    z.writestr("word/media/image2.jpeg", jpeg, compress_type=zipfile.ZIP_STORED)
    z.writestr("word/media/image3.jpeg", jpeg, compress_type=zipfile.ZIP_DEFLATED)
    z.writestr("word/media/image4.png", png, compress_type=zipfile.ZIP_DEFLATED)
sys.stdout.buffer.write(buf.getvalue())
"#, png.len(), png.len() + 1), &pictures);
        let opened_pictures = round_trip(&picture_zip, "zip with pictures");
        assert!(opened_pictures.len() > plain.len(), "the PNG inside opened: {}", opened_pictures.len());
        let recipe = open(&picture_zip).unwrap().recipe;
        assert!(recipe.windows(1).any(|w| w[0] == JPEG), "the stored JPEG transcoded");
        assert!(recipe.windows(1).any(|w| w[0] == DEFLATE_JPEG), "the deflated JPEG transcoded under its deflate");
        assert!(recipe.windows(1).any(|w| w[0] == DEFLATE_NESTED), "the deflated PNG opened under its deflate");
        assert!(opened_pictures.len() > 2 * plain.len(), "both PNGs' plain text is in: {}", opened_pictures.len());
        // A tar of a gzipped log, the PNG, the JPEG and a text file; then
        // that tar gzipped: everything opened through the layers.
        let mut members = gzip(&plain[..200000], &["-6"]);
        let (gz_len, png_len) = (members.len(), png.len());
        members.extend_from_slice(&png);
        members.extend_from_slice(&jpeg);
        members.extend_from_slice(&plain[..50000]);
        let tar = python(&format!(r#"
import sys, tarfile, io
blob = sys.stdin.buffer.read()
parts = [("log.gz", blob[:{gz}]), ("shot.png", blob[{gz}:{gz}+{png}]), ("photo.jpg", blob[{gz}+{png}:-50000]), ("notes.txt", blob[-50000:])]
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w") as t:
    for name, data in parts:
        info = tarfile.TarInfo(name); info.size = len(data)
        t.addfile(info, io.BytesIO(data))
sys.stdout.buffer.write(buf.getvalue())
"#, gz=gz_len, png=png_len), &members);
        assert!(is_tar(&tar));
        let opened_tar = round_trip(&tar, "tar");
        assert!(opened_tar.len() >= 200000 + plain.len(), "the gzip member and the PNG opened inside the tar: {}", opened_tar.len());
        let targz = gzip(&tar, &["-6"]);
        let opened_targz = round_trip(&targz, "tar.gz");
        assert!(opened_targz.len() >= 200000 + plain.len(), "opened through the gzip: {}", opened_targz.len());
        assert!(is_pdf(&pdf));
        let opened_pdf = round_trip(&pdf, "pdf");
        assert!(opened_pdf.len() >= 600000, "the three Flate streams opened: {}", opened_pdf.len());
        // A truncated PNG and a zip with garbage after an entry still
        // close to what they were.
        round_trip(&png[..png.len() - 7], "truncated png");
        let mut odd = zip.clone();
        odd.extend_from_slice(b"garbage at the end");
        round_trip(&odd, "zip with a tail");
    }
}
