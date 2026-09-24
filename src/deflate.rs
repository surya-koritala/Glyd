//! Containers opened: gzip, zip (so also .docx, .xlsx, .pptx, .jar,
//! .apk, .odt), tar, zlib streams, PNG and PDF, nested in one another
//! four deep, and the JPEGs inside them transcoded (`jpeg`). Every
//! deflate stream is decoded to its content by `preflate`, which also
//! records what it takes to re-encode the stream bit for bit (its
//! corrections). The object becomes one plain text and a recipe. The
//! plain text is the content of every stream, in order, then the side
//! data: every other byte of the object as it was (headers,
//! directories, stored entries, a tar's files), the corrections and the
//! transcoded pictures. The recipe is structure only, segment kinds and
//! lengths. So whatever the caller asked for (a level, record mode, the
//! cold level, a base) sees all of the object and none of the deflate:
//! a gzipped log costs what the log costs (its content is the log, line
//! for line, so record mode takes it), and two versions of an archive
//! share what they have in common, corrections included.
//! Envelope:
//!
//!   "GLYDDEF2" original_len, recipe_len (varints), the recipe
//!   compressed at the max level, then the inner stream of the plain
//!   text (any format: units of a level, records, cold, a base
//!   envelope).
//!
//! The recipe: the segment count and the content's length (varints;
//! the side data follows the content in the plain text), then the
//! segments, each a tag and its fields: 0 kept (length, from the
//! side); 1 deflate (corrections' length, from the side; text
//! length, from the content); 2 a PNG's image stream (2 zlib header
//! bytes, corrections' length, text length, 4 Adler-32 bytes, then a
//! chunk count and per chunk a length and 4 CRC bytes: the recreated
//! stream cut into IDAT chunks); 3 a JPEG (its Lepton stream's length,
//! from the side); 4 a deflate stream holding a JPEG (corrections'
//! length, the Lepton stream's length, both from the side); 5 a
//! deflate stream holding a container (corrections' length, then the
//! inner container: its segment count, its segments' length and
//! segments, its content's and its side's lengths); 6 a container kept
//! inside (the inner container, likewise).
//!
//! Envelopes of v0.12.0 ("GLYDGZIP", gzip members) and v0.13.0
//! ("GLYDDEFL", which kept the kept bytes and the corrections in the
//! recipe) are read. A part of a stream (a unit, a record unit, a trial
//! sample) is never opened on its own (`crate::as_part`).

use crate::record::{get_varint, put_varint};
use preflate_rs::{chunked, preflate_whole_deflate_stream, recreate_whole_deflate_stream, PreflateConfig};
use std::borrow::Cow;
use std::sync::Mutex;

pub(crate) const MAGIC: &[u8; 8] = b"GLYDDEF3";
/// v0.13.1 to v0.13.3: the same recipe, its streams opened by preflate;
/// a base such an envelope was made against opens that way still.
const MAGIC_DEF2: &[u8; 8] = b"GLYDDEF2";
const MAGIC_V013: &[u8; 8] = b"GLYDDEFL";
const MAGIC_V012: &[u8; 8] = b"GLYDGZIP";
/// The plain text is held in memory; beyond this an object is left alone.
const PLAIN_LIMIT: usize = 8 << 30;
/// Containers inside containers are opened this deep.
const MAX_DEPTH: u32 = 4;
const KEEP: u8 = 0;
const DEFLATE: u8 = 1;
const PNG_IMAGE: u8 = 2;
const JPEG: u8 = 3;
const DEFLATE_JPEG: u8 = 4;
const DEFLATE_NESTED: u8 = 5;
const NESTED: u8 = 6;
/// `DEFLATE` and `DEFLATE_NESTED` with the stream's corrections in
/// chunks that open and close on every core (v0.13.3).
const DEFLATE_CHUNKED: u8 = 7;
const DEFLATE_NESTED_CHUNKED: u8 = 8;
/// `DEFLATE` and `DEFLATE_NESTED` opened by Glyd's own reconstruction
/// (`crate::reflate`, v0.13.4): the recipe from the side, the text
/// from the content.
const REFLATE: u8 = 9;
const REFLATE_NESTED: u8 = 10;
/// `PNG_IMAGE` and `DEFLATE_JPEG` opened by `crate::reflate`: the
/// recipe from the side in place of preflate's corrections.
const PNG_REFLATE: u8 = 11;
const REFLATE_JPEG: u8 = 12;
/// A Parquet page's snappy stream written back by `crate::resnappy`:
/// the build (one byte, an index into its `BUILDS`) and the raw page
/// as content; the page's compressed length follows from them.
const SNAPPY_PAGE: u8 = 13;
/// `SNAPPY_PAGE` with the page's values modeled (`crate::parquet::model`):
/// the build, the model's recipe from the side, the modeled bytes as
/// content.
const SNAPPY_MODELED: u8 = 14;
/// `SNAPPY_PAGE` and `SNAPPY_MODELED` for a zstd frame written back by
/// `crate::rezstd`: a Parquet page's, or a whole object's (a `.zst`).
const ZSTD_FRAME: u8 = 15;
const ZSTD_MODELED: u8 = 16;

/// A stream is opened in chunks of this much plain text when it holds
/// at least two of them.
const CHUNK_PLAIN: usize = 8 << 20;

/// An object opened: its plain text and the recipe to close it.
pub struct Opened {
    pub plain: Vec<u8>,
    pub recipe: Vec<u8>,
}

/// Whether `input` starts like a container worth opening (a JPEG
/// counts: it is transcoded by `jpeg` through the same hooks).
pub fn is_container(input: &[u8]) -> bool {
    if crate::jpeg::is_jpeg(input) {
        return true;
    }
    is_gzip(input) || is_zip(input) || is_png(input) || is_pdf(input) || is_tar(input) || is_zlib(input) || crate::parquet::is_parquet(input) || is_zstd(input)
}

/// A zstd frame's magic.
fn is_zstd(input: &[u8]) -> bool {
    input.len() >= 16 && input.starts_with(&[0x28, 0xb5, 0x2f, 0xfd])
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

/// The most a stream of `len` compressed bytes may expand to: real data
/// stays well under 200 times; a bomb stops here.
fn stream_limit(len: usize) -> usize {
    (len.saturating_mul(200)).clamp(256 << 20, PLAIN_LIMIT)
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// What a builder made: finished into an `Opened`, laid into its
/// parent's (a piece opened on a thread of its own), or nested in a
/// segment of its parent (a container inside a container).
struct Parts {
    content: Vec<u8>,
    side: Vec<u8>,
    body: Vec<u8>,
    segments: u64,
}

/// The recipe and plain text under construction. `keep` is the run of
/// the input kept as it is and not yet written: everything before it
/// is written, everything from its end on not yet looked at, so a byte
/// no segment claims is always kept.
/// What opens a deflate stream: Glyd's own reconstruction (every
/// envelope written now), or preflate alone, the way v0.13.1 to
/// v0.13.3 wrote `GLYDDEF2` (a base for their deltas must open to the
/// same plain text).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Engine {
    Reflate,
    Preflate,
}

struct Builder {
    content: Vec<u8>,
    side: Vec<u8>,
    body: Vec<u8>,
    segments: u64,
    keep: (usize, usize),
    depth: u32,
    opened: bool,
    engine: Engine,
}

impl Builder {
    fn new(depth: u32, at: usize, engine: Engine) -> Builder {
        Builder { content: Vec::new(), side: Vec::new(), body: Vec::new(), segments: 0, keep: (at, at), depth, opened: false, engine }
    }

    /// Everything up to `to` not yet looked at, kept.
    fn keep_to(&mut self, to: usize) {
        if to > self.keep.1 {
            self.keep.1 = to;
        }
    }

    fn flush(&mut self, input: &[u8]) {
        let (from, to) = self.keep;
        if from < to {
            self.body.push(KEEP);
            put_varint(&mut self.body, (to - from) as u64);
            self.side.extend_from_slice(&input[from..to]);
            self.segments += 1;
        }
        self.keep = (to, to);
    }

    /// A segment starting at `at`: what comes before it kept, its tag.
    fn segment(&mut self, input: &[u8], tag: u8, at: usize) {
        self.keep_to(at);
        self.flush(input);
        self.body.push(tag);
        self.segments += 1;
        self.opened = true;
    }

    /// A piece of this container opened on its own, standing at
    /// `at..resume` of the input.
    fn lay(&mut self, input: &[u8], parts: Parts, at: usize, resume: usize) {
        self.keep_to(at);
        self.flush(input);
        self.body.extend_from_slice(&parts.body);
        self.content.extend_from_slice(&parts.content);
        self.side.extend_from_slice(&parts.side);
        self.segments += parts.segments;
        self.opened = true;
        self.keep = (resume, resume);
    }

    /// An inner container's fields, after its segment's own: its
    /// segment count and segments, its content's and side's lengths;
    /// its content and side appended to ours.
    fn nest(&mut self, inner: Parts) {
        put_varint(&mut self.body, inner.segments);
        put_varint(&mut self.body, inner.body.len() as u64);
        self.body.extend_from_slice(&inner.body);
        put_varint(&mut self.body, inner.content.len() as u64);
        put_varint(&mut self.body, inner.side.len() as u64);
        self.content.extend_from_slice(&inner.content);
        self.side.extend_from_slice(&inner.side);
    }

    /// `data` opened as a container of its own, when it is one and this
    /// builder is not too deep.
    fn nested(&self, data: &[u8]) -> Option<Parts> {
        if self.depth + 1 >= MAX_DEPTH || !is_container(data) {
            return None;
        }
        if crate::jpeg::is_jpeg(data) {
            return None;
        }
        open_parts(data, self.depth + 1, self.engine)
    }

    /// The deflate stream at `input[at..end]`: its compressed length
    /// when it opens. A stream whose corrections come to more than a
    /// quarter of it (an encoder preflate predicts badly) is not worth
    /// opening. A JPEG under it is transcoded; a container under it,
    /// opened.
    fn deflate(&mut self, input: &[u8], at: usize, end: usize) -> Option<usize> {
        let data = input.get(at..end)?;
        if self.engine == Engine::Reflate {
            if let Some(n) = self.reflate(input, at, data) {
                return Some(n);
            }
        }
        let config = PreflateConfig { plain_text_limit: stream_limit(data.len()), verify_compression: true, ..Default::default() };
        let parsed = chunked::parse(data, &config).ok()?;
        if parsed.plain.text().len() >= 2 * CHUNK_PLAIN {
            if let Some(n) = self.deflate_chunked(input, at, data, parsed) {
                return Some(n);
            }
        }
        let (result, text) = preflate_whole_deflate_stream(data, &config).ok()?;
        let n = result.compressed_size;
        if result.corrections.len() * 4 > n {
            return None;
        }
        let text = text.text();
        if crate::jpeg::is_jpeg(text) {
            if let Some(lepton) = crate::jpeg::transcode(text).filter(|l| l.len() + 16 < text.len()) {
                self.segment(input, DEFLATE_JPEG, at);
                put_varint(&mut self.body, result.corrections.len() as u64);
                put_varint(&mut self.body, lepton.len() as u64);
                self.side.extend_from_slice(&result.corrections);
                self.side.extend_from_slice(&lepton);
                self.keep = (at + n, at + n);
                return Some(n);
            }
        }
        if let Some(inner) = self.nested(text) {
            self.segment(input, DEFLATE_NESTED, at);
            put_varint(&mut self.body, result.corrections.len() as u64);
            self.side.extend_from_slice(&result.corrections);
            self.nest(inner);
            self.keep = (at + n, at + n);
            return Some(n);
        }
        self.segment(input, DEFLATE, at);
        put_varint(&mut self.body, result.corrections.len() as u64);
        put_varint(&mut self.body, text.len() as u64);
        self.side.extend_from_slice(&result.corrections);
        self.content.extend_from_slice(text);
        self.keep = (at + n, at + n);
        Some(n)
    }

    /// The stream opened by Glyd's own reconstruction, checked bit for
    /// bit: a `REFLATE` segment, or `REFLATE_NESTED` when the text is a
    /// container. `None` when it does not open that way, or opens
    /// badly (a recipe over a quarter of the stream), or holds a JPEG
    /// (preflate's road, for now).
    fn reflate(&mut self, input: &[u8], at: usize, data: &[u8]) -> Option<usize> {
        let opened = crate::reflate::open(data)?;
        let n = opened.consumed;
        if opened.recipe.len() * 4 > n || opened.plain.len() > stream_limit(data.len()) {
            return None;
        }
        if crate::jpeg::is_jpeg(&opened.plain) {
            if let Some(lepton) = crate::jpeg::transcode(&opened.plain).filter(|l| l.len() + 16 < opened.plain.len()) {
                self.segment(input, REFLATE_JPEG, at);
                put_varint(&mut self.body, opened.recipe.len() as u64);
                put_varint(&mut self.body, lepton.len() as u64);
                self.side.extend_from_slice(&opened.recipe);
                self.side.extend_from_slice(&lepton);
                self.keep = (at + n, at + n);
                return Some(n);
            }
            return None;
        }
        let inner = self.nested(&opened.plain);
        self.segment(input, if inner.is_some() { REFLATE_NESTED } else { REFLATE }, at);
        put_varint(&mut self.body, opened.recipe.len() as u64);
        self.side.extend_from_slice(&opened.recipe);
        match inner {
            Some(inner) => self.nest(inner),
            None => {
                put_varint(&mut self.body, opened.plain.len() as u64);
                self.content.extend_from_slice(&opened.plain);
            }
        }
        self.keep = (at + n, at + n);
        Some(n)
    }

    /// A large stream opened in chunks, predicted and checked on every
    /// core: a `DEFLATE_CHUNKED` segment, or `DEFLATE_NESTED_CHUNKED`
    /// when the text is a container. `None` when a chunk does not
    /// close, or a JPEG is inside (the whole-stream way takes those).
    fn deflate_chunked(&mut self, input: &[u8], at: usize, data: &[u8], parsed: chunked::Parsed) -> Option<usize> {
        let n = parsed.contents.compressed_size;
        let chunks = chunked::plan(&parsed, CHUNK_PLAIN);
        let corrections = each(chunks.len(), true, |i| chunked::predict(&parsed, &chunks[i]).ok());
        let corrections: Vec<Vec<u8>> = corrections.into_iter().collect::<Option<_>>()?;
        if corrections.iter().map(|c| c.len()).sum::<usize>() * 4 > n {
            return None;
        }
        let params = chunked::parameters(&corrections[0]).ok()?;
        let text = parsed.plain.text();
        let pieces = each(chunks.len(), true, |i| chunked::recreate(&params, text, chunks[i].plain.clone(), &corrections[i], i == 0, chunks[i].bit_start as u32).ok());
        let pieces: Vec<(Vec<u8>, u32)> = pieces.into_iter().collect::<Option<_>>()?;
        if chunked::join(&pieces) != &data[..n] {
            return None;
        }
        if crate::jpeg::is_jpeg(text) {
            return None;
        }
        let inner = self.nested(text);
        self.segment(input, if inner.is_some() { DEFLATE_NESTED_CHUNKED } else { DEFLATE_CHUNKED }, at);
        put_varint(&mut self.body, chunks.len() as u64);
        for (chunk, c) in chunks.iter().zip(&corrections) {
            put_varint(&mut self.body, chunk.plain.len() as u64);
            put_varint(&mut self.body, c.len() as u64);
            self.body.push((chunk.bit_start % 8) as u8);
            self.side.extend_from_slice(c);
        }
        match inner {
            Some(inner) => self.nest(inner),
            None => {
                put_varint(&mut self.body, text.len() as u64);
                self.content.extend_from_slice(text);
            }
        }
        self.keep = (at + n, at + n);
        Some(n)
    }

    /// A stored entry's data at `input[from..to]`: a JPEG transcoded, a
    /// container opened; anything else stays to be kept.
    fn stored(&mut self, input: &[u8], from: usize, to: usize) {
        let data = &input[from..to];
        if crate::jpeg::is_jpeg(data) {
            if let Some(lepton) = crate::jpeg::transcode(data).filter(|l| l.len() + 16 < data.len()) {
                self.segment(input, JPEG, from);
                put_varint(&mut self.body, lepton.len() as u64);
                self.side.extend_from_slice(&lepton);
                self.keep = (to, to);
                return;
            }
        }
        if let Some(inner) = self.nested(data) {
            self.segment(input, NESTED, from);
            self.nest(inner);
            self.keep = (to, to);
        }
    }

    /// The PNG at `input[from..to]`: its IDAT chunks' data, joined, is
    /// one zlib stream, opened as a `PNG_IMAGE` segment that remembers
    /// how to cut it back into chunks; every other chunk stays to be
    /// kept. `None` leaves the builder as it was.
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
                first_idat.get_or_insert(at);
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
        // (tag, corrections or recipe, text)
        let (tag, side, text): (u8, Vec<u8>, Vec<u8>) = match self.engine {
            Engine::Reflate => {
                let o = crate::reflate::open(&image[2..])?;
                if 2 + o.consumed + 4 != image.len() || o.recipe.len() * 4 > o.consumed || o.plain.len() > stream_limit(image.len()) {
                    return None;
                }
                (PNG_REFLATE, o.recipe, o.plain)
            }
            Engine::Preflate => {
                let config = PreflateConfig { plain_text_limit: stream_limit(image.len()), verify_compression: true, ..Default::default() };
                let (result, text) = preflate_whole_deflate_stream(&image[2..], &config).ok()?;
                if 2 + result.compressed_size + 4 != image.len() || result.corrections.len() * 4 > result.compressed_size {
                    return None;
                }
                (PNG_IMAGE, result.corrections, text.text().to_vec())
            }
        };
        self.segment(input, tag, start);
        self.body.extend_from_slice(&image[..2]);
        put_varint(&mut self.body, side.len() as u64);
        put_varint(&mut self.body, text.len() as u64);
        self.body.extend_from_slice(&image[image.len() - 4..]);
        put_varint(&mut self.body, idat.len() as u64);
        for &(s, l) in &idat {
            put_varint(&mut self.body, l as u64);
            self.body.extend_from_slice(&input[s + l..s + l + 4]);
        }
        self.side.extend_from_slice(&side);
        self.content.extend_from_slice(&text);
        self.keep = (last_end, last_end);
        Some(())
    }

    /// Everything up to `to` kept and written: the parts, when anything
    /// opened.
    fn into_parts(mut self, input: &[u8], to: usize) -> Option<Parts> {
        // Nothing opened: nothing to build. (Flushing first copied the
        // whole of a tar with no compressed member, 0.24 s a gigabyte,
        // to then return None.)
        if !self.opened {
            return None;
        }
        self.keep_to(to);
        self.flush(input);
        if self.content.len() + self.side.len() > PLAIN_LIMIT {
            return None;
        }
        Some(Parts { content: self.content, side: self.side, body: self.body, segments: self.segments })
    }
}

/// `f(i)` for every `i < n`, on every core for a container at the top
/// (its entries, streams or segments are independent), in order below.
fn each<T: Send>(n: usize, parallel: bool, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    each_sized(n, parallel, |_| 0, f)
}

fn each_sized<T: Send>(n: usize, parallel: bool, size: impl Fn(usize) -> usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    if !parallel || n < 2 {
        return (0..n).map(f).collect();
    }
    let slots: Vec<Mutex<Option<T>>> = (0..n).map(|_| Mutex::new(None)).collect();
    // The biggest pieces first, so that the long ones do not start
    // last; every piece's own parallel work spawns under this loop's.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(size(i)));
    let _ = crate::par_units::<()>(n, |k| {
        let i = order[k];
        let r = f(i);
        *slots[i].lock().unwrap() = Some(r);
        Ok(())
    });
    slots.into_iter().map(|m| m.into_inner().unwrap().expect("every piece is made")).collect()
}

/// A stored entry at `input[from..to]` opened on its own.
fn stored_parts(input: &[u8], from: usize, to: usize, depth: u32, engine: Engine) -> Option<Parts> {
    let mut b = Builder::new(depth, from, engine);
    b.stored(input, from, to);
    b.into_parts(input, to)
}

/// The deflate stream at `input[from..to]` opened on its own, what
/// follows it up to `to` kept; and where the stream ends.
fn deflate_parts(input: &[u8], from: usize, to: usize, depth: u32, engine: Engine) -> Option<(Parts, usize)> {
    let mut b = Builder::new(depth, from, engine);
    let n = b.deflate(input, from, to)?;
    Some((b.into_parts(input, to)?, from + n))
}

/// The plain text and recipe of a container, when anything in it
/// opens; `None` for anything else.
pub fn open(input: &[u8]) -> Option<Opened> {
    open_with(input, Engine::Reflate)
}

pub fn open_with(input: &[u8], engine: Engine) -> Option<Opened> {
    let parts = open_parts(input, 0, engine)?;
    let mut recipe = Vec::with_capacity(parts.body.len() + 16);
    put_varint(&mut recipe, parts.segments);
    put_varint(&mut recipe, parts.content.len() as u64);
    recipe.extend_from_slice(&parts.body);
    let mut plain = parts.content;
    plain.extend_from_slice(&parts.side);
    Some(Opened { plain, recipe })
}

fn open_parts(input: &[u8], depth: u32, engine: Engine) -> Option<Parts> {
    let b = Builder::new(depth, 0, engine);
    if is_gzip(input) {
        open_gzip(input, b)
    } else if is_zip(input) {
        open_zip(input, b)
    } else if is_png(input) {
        let mut b = b;
        b.png(input, 0, input.len())?;
        b.into_parts(input, input.len())
    } else if is_pdf(input) {
        open_pdf(input, b)
    } else if is_tar(input) {
        open_tar(input, b)
    } else if is_zlib(input) {
        let mut b = b;
        b.deflate(input, 2, input.len())?;
        b.into_parts(input, input.len())
    } else if crate::parquet::is_parquet(input) {
        open_parquet(input, b)
    } else if is_zstd(input) {
        open_zstd(input, b)
    } else {
        None
    }
}

/// A zstd frame standing alone (a `.zst` object) written back by
/// `crate::rezstd`: its content the segment's content; a container
/// under it opened. An object of several frames, or a frame no build
/// made, is kept as it is.
fn open_zstd(input: &[u8], mut b: Builder) -> Option<Parts> {
    let (plain, build) = crate::rezstd::reproduce(input)?;
    let build = crate::rezstd::BUILDS.iter().position(|&x| x == build)? as u8;
    if plain.len() > stream_limit(input.len()) {
        return None;
    }
    b.segment(input, ZSTD_FRAME, 0);
    b.body.push(build);
    put_varint(&mut b.body, plain.len() as u64);
    b.content.extend_from_slice(&plain);
    b.keep = (input.len(), input.len());
    b.into_parts(input, input.len())
}

/// Parquet: every snappy page the reference compressor's port writes
/// back byte for byte becomes a `SNAPPY_PAGE` segment, its raw bytes
/// the content; the headers, the footer and every other page are
/// kept. Pages of the other codecs wait for their reproducers.
fn open_parquet(input: &[u8], mut b: Builder) -> Option<Parts> {
    let (chunks, _) = crate::parquet::chunks(input)?;
    let mut pages: Vec<(usize, usize, usize, crate::parquet::Page)> = Vec::new();
    for (i, c) in chunks.iter().enumerate().filter(|(_, c)| matches!(c.codec, crate::parquet::Codec::Snappy | crate::parquet::Codec::Zstd)) {
        for p in crate::parquet::pages(input, c)? {
            let skip = if p.kind == 3 { p.v2_levels_len } else { 0 };
            pages.push((p.body_at + skip, p.body_at + p.compressed_len, i, p));
        }
    }
    pages.sort_unstable_by_key(|p| p.0);
    // Every page on its own core: reproduced, then its values modeled
    // when a model beats the page as it is.
    let opened: Vec<Option<(bool, u8, Option<(Vec<u8>, Vec<u8>)>, Vec<u8>)>> = each_sized(pages.len(), b.depth == 0, |i| pages[i].1 - pages[i].0, |i| {
        let (at, end, chunk, page) = &pages[i];
        if *end > input.len() || at >= end {
            return None;
        }
        let zstd = chunks[*chunk].codec == crate::parquet::Codec::Zstd;
        let (plain, build) = if zstd {
            let (plain, build) = crate::rezstd::reproduce(&input[*at..*end])?;
            (plain, crate::rezstd::BUILDS.iter().position(|&x| x == build)? as u8)
        } else {
            let (plain, build) = crate::resnappy::reproduce(&input[*at..*end])?;
            (plain, crate::resnappy::BUILDS.iter().position(|&x| x == build)? as u8)
        };
        let modeled = crate::parquet::model(&plain, &chunks[*chunk], page).filter(|(recipe, _)| recipe[0] != 0);
        Some((zstd, build, modeled, plain))
    });
    for (i, (at, end, _, _)) in pages.iter().enumerate() {
        if *at < b.keep.1 {
            continue;
        }
        let Some((zstd, build, modeled, plain)) = &opened[i] else { continue };
        match modeled {
            Some((recipe, m)) => {
                b.segment(input, if *zstd { ZSTD_MODELED } else { SNAPPY_MODELED }, *at);
                b.body.push(*build);
                put_varint(&mut b.body, recipe.len() as u64);
                put_varint(&mut b.body, m.len() as u64);
                b.side.extend_from_slice(recipe);
                b.content.extend_from_slice(m);
            }
            None => {
                b.segment(input, if *zstd { ZSTD_FRAME } else { SNAPPY_PAGE }, *at);
                b.body.push(*build);
                put_varint(&mut b.body, plain.len() as u64);
                b.content.extend_from_slice(plain);
            }
        }
        b.keep = (*end, *end);
    }
    b.into_parts(input, input.len())
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
/// trailer; from the first member that does not open on, kept.
fn open_gzip(input: &[u8], mut b: Builder) -> Option<Parts> {
    let mut at = 0usize;
    while at < input.len() && is_gzip(&input[at..]) {
        let Some(hend) = gzip_header_end(input, at) else { break };
        let Some(n) = b.deflate(input, hend, input.len()) else { break };
        if hend + n + 8 > input.len() {
            break;
        }
        at = hend + n + 8;
    }
    b.into_parts(input, input.len())
}

/// Entries in order, each a 512-byte header and data padded to 512: a
/// regular file that is a container of its own opened (on every core
/// at the top), everything else kept.
fn open_tar(input: &[u8], mut b: Builder) -> Option<Parts> {
    let mut entries: Vec<(usize, usize)> = Vec::new();
    let mut at = 0usize;
    while at + 512 <= input.len() && &input[at + 257..at + 262] == b"ustar" {
        let Some(size) = tar_size(input, at) else { break };
        let data = at + 512;
        let Some(end) = data.checked_add(size).filter(|&e| e <= input.len()) else { break };
        let regular = input[at + 156] == b'0' || input[at + 156] == 0;
        if regular && is_container(&input[data..end]) {
            entries.push((data, end));
        }
        at = data.saturating_add((size + 511) / 512 * 512);
    }
    let depth = b.depth;
    let engine = b.engine;
    let pieces = each_sized(entries.len(), depth == 0, |i| entries[i].1 - entries[i].0, |i| stored_parts(input, entries[i].0, entries[i].1, depth, engine));
    for (&(data, end), piece) in entries.iter().zip(pieces) {
        if let Some(p) = piece {
            b.lay(input, p, data, end);
        }
    }
    b.into_parts(input, input.len())
}

/// Every entry's data from a zip's central directory: (start, end,
/// method, flags) in the order of the data; `None` when there is no
/// directory that parses and agrees with the local headers.
fn zip_entries(input: &[u8]) -> Option<Vec<(usize, usize, usize, usize)>> {
    let le16 = |p: usize| input.get(p..p + 2).map(|s| u16::from_le_bytes(s.try_into().unwrap()) as usize);
    let le32 = |p: usize| input.get(p..p + 4).map(|s| u32::from_le_bytes(s.try_into().unwrap()) as usize);
    let le64 = |p: usize| input.get(p..p + 8).map(|s| u64::from_le_bytes(s.try_into().unwrap()) as usize);
    // The end record sits in the last 64 KB (its comment is at most that).
    let low = input.len().saturating_sub(65536 + 22);
    let eocd = (low..input.len().saturating_sub(21)).rev().find(|&p| &input[p..p + 4] == b"PK\x05\x06")?;
    let (mut count, mut cd) = (le16(eocd + 10)?, le32(eocd + 16)?);
    if count == 0xFFFF || cd == 0xFFFF_FFFF {
        let locator = eocd.checked_sub(20)?;
        if input.get(locator..locator + 4)? != b"PK\x06\x07" {
            return None;
        }
        let record = le64(locator + 8)?;
        if input.get(record..record + 4)? != b"PK\x06\x06" {
            return None;
        }
        count = le64(record + 32)?;
        cd = le64(record + 48)?;
    }
    if count > input.len() / 46 + 1 {
        return None;
    }
    let mut found = Vec::with_capacity(count);
    let mut p = cd;
    for _ in 0..count {
        if input.get(p..p + 4)? != b"PK\x01\x02" {
            return None;
        }
        let (flags, method) = (le16(p + 8)?, le16(p + 10)?);
        let (mut csize, usize, mut offset) = (le32(p + 20)?, le32(p + 24)?, le32(p + 42)?);
        let (nlen, xlen, clen) = (le16(p + 28)?, le16(p + 30)?, le16(p + 32)?);
        // A zip64 entry's sizes and offset are in its extra field (id 1):
        // those that are 0xFFFFFFFF above, in the order uncompressed,
        // compressed, offset.
        if csize == 0xFFFF_FFFF || usize == 0xFFFF_FFFF || offset == 0xFFFF_FFFF {
            let (mut q, xend) = (p + 46 + nlen, p + 46 + nlen + xlen);
            while q + 4 <= xend {
                let (id, len) = (le16(q)?, le16(q + 2)?);
                if id == 1 {
                    let mut f = q + 4;
                    if usize == 0xFFFF_FFFF {
                        f += 8;
                    }
                    if csize == 0xFFFF_FFFF {
                        csize = le64(f)?;
                        f += 8;
                    }
                    if offset == 0xFFFF_FFFF {
                        offset = le64(f)?;
                    }
                    break;
                }
                q += 4 + len;
            }
        }
        found.push((offset, csize, method, flags));
        p += 46 + nlen + xlen + clen;
    }
    found.sort_unstable();
    let mut entries = Vec::with_capacity(found.len());
    let mut last = 0usize;
    for (offset, csize, method, flags) in found {
        if offset < last || input.get(offset..offset + 4)? != b"PK\x03\x04" {
            return None;
        }
        let data = offset + 30 + le16(offset + 26)? + le16(offset + 28)?;
        let end = data.checked_add(csize).filter(|&e| e <= input.len())?;
        entries.push((data, end, method, flags));
        last = end;
    }
    Some(entries)
}

/// A zip: its entries from the central directory, each deflate stream
/// or stored container opened (on every core at the top), everything
/// else kept.
fn open_zip(input: &[u8], mut b: Builder) -> Option<Parts> {
    let Some(entries) = zip_entries(input) else { return open_zip_walk(input, b) };
    let depth = b.depth;
    let engine = b.engine;
    let pieces = each_sized(entries.len(), depth == 0, |i| entries[i].1 - entries[i].0, |i| {
        let (data, end, method, flags) = entries[i];
        match method {
            8 if flags & 1 == 0 => deflate_parts(input, data, end, depth, engine).map(|(p, _)| p),
            0 if is_container(&input[data..end]) => stored_parts(input, data, end, depth, engine),
            _ => None,
        }
    });
    for (&(data, end, _, _), piece) in entries.iter().zip(pieces) {
        if let Some(p) = piece {
            b.lay(input, p, data, end);
        }
    }
    b.into_parts(input, input.len())
}

/// A zip without a usable central directory: the local entries walked
/// in order. An entry whose size is only in a data descriptor after it,
/// and whose stream does not open, ends where the next entry's
/// signature is found.
fn open_zip_walk(input: &[u8], mut b: Builder) -> Option<Parts> {
    let le16 = |p: usize| input.get(p..p + 2).map(|s| u16::from_le_bytes(s.try_into().unwrap()) as usize);
    let le32 = |p: usize| input.get(p..p + 4).map(|s| u32::from_le_bytes(s.try_into().unwrap()) as usize);
    let next_signature = |from: usize| (from..input.len().saturating_sub(3)).find(|&p| &input[p..p + 4] == b"PK\x03\x04").unwrap_or(input.len());
    let mut at = 0usize;
    while input.get(at..at + 4) == Some(b"PK\x03\x04") {
        let (Some(flags), Some(method), Some(mut csize), Some(usize), Some(nlen), Some(xlen)) = (le16(at + 6), le16(at + 8), le32(at + 18), le32(at + 22), le16(at + 26), le16(at + 28)) else { break };
        let data = at + 30 + nlen + xlen;
        if data > input.len() {
            break;
        }
        if csize == 0xFFFF_FFFF {
            let mut p = at + 30 + nlen;
            while p + 4 <= data {
                let (Some(id), Some(len)) = (le16(p), le16(p + 2)) else { break };
                if id == 1 {
                    let q = p + 4 + if usize == 0xFFFF_FFFF { 8 } else { 0 };
                    csize = input.get(q..q + 8).map_or(csize, |s| u64::from_le_bytes(s.try_into().unwrap()) as usize);
                    break;
                }
                p += 4 + len;
            }
        }
        let sized = flags & 8 == 0 && csize != 0xFFFF_FFFF;
        let end = if method == 8 && flags & 1 == 0 {
            match b.deflate(input, data, input.len()) {
                Some(n) => data + n,
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
        if method == 0 {
            b.stored(input, data, end);
        }
        // The next entry: right here, or past a data descriptor.
        at = (end..(end + 32).min(input.len().saturating_sub(3))).find(|&p| &input[p..p + 4] == b"PK\x03\x04").unwrap_or(end);
    }
    b.into_parts(input, input.len())
}

/// A PDF's streams: every `stream` keyword whose data starts like a
/// zlib stream (a /FlateDecode filter; a predictor changes only what
/// the bytes mean) and ends, Adler-32 and white space after it, at its
/// `endstream`, opened (on every core at the top); everything else
/// kept. Found by scanning, so a /Length by reference, object streams
/// and cross-reference streams need no parsing.
fn open_pdf(input: &[u8], mut b: Builder) -> Option<Parts> {
    let mut candidates: Vec<(usize, usize)> = Vec::new();
    let mut scan = 0usize;
    while let Some(k) = find(&input[scan..], b"stream") {
        let key = scan + k;
        scan = key + 6;
        if key >= 3 && &input[key - 3..key] == b"end" {
            continue;
        }
        let data = match (input.get(scan), input.get(scan + 1)) {
            (Some(b'\r'), Some(b'\n')) => scan + 2,
            (Some(b'\n'), _) => scan + 1,
            _ => continue,
        };
        if !is_zlib(&input[data..]) {
            continue;
        }
        let Some(e) = find(&input[data..], b"endstream") else { break };
        candidates.push((data, data + e));
        scan = data + e + 9;
    }
    let depth = b.depth;
    let engine = b.engine;
    let pieces = each_sized(candidates.len(), depth == 0, |i| candidates[i].1 - candidates[i].0, |i| {
        let (data, stop) = candidates[i];
        let (parts, end) = deflate_parts(input, data + 2, stop, depth, engine)?;
        let tail = input.get(end..stop)?;
        (tail.len() >= 4 && tail[4..].iter().all(|c| matches!(c, b'\r' | b'\n' | b' ' | b'\t'))).then_some(parts)
    });
    for (&(data, stop), piece) in candidates.iter().zip(pieces) {
        if let Some(p) = piece {
            b.lay(input, p, data + 2, stop);
        }
    }
    b.into_parts(input, input.len())
}

/// An inner container of a recipe: its segments, content and side.
struct Inner<'a> {
    segments: u64,
    body: &'a [u8],
    content: &'a [u8],
    side: &'a [u8],
}

/// A recipe's segment, parsed; its work not yet done.
enum Seg<'a> {
    Bytes(&'a [u8]),
    Deflate { corrections: &'a [u8], text: &'a [u8] },
    DeflateJpeg { corrections: &'a [u8], lepton: &'a [u8] },
    DeflateNested { corrections: &'a [u8], inner: Inner<'a> },
    /// (plain length, corrections, bit offset) per chunk
    DeflateChunked { chunks: Vec<(usize, &'a [u8], u8)>, text: &'a [u8] },
    DeflateNestedChunked { chunks: Vec<(usize, &'a [u8], u8)>, inner: Inner<'a> },
    Reflate { recipe: &'a [u8], text: &'a [u8] },
    ReflateNested { recipe: &'a [u8], inner: Inner<'a> },
    Snappy { build: u8, text: &'a [u8] },
    SnappyModeled { build: u8, recipe: &'a [u8], text: &'a [u8] },
    Zstd { build: u8, text: &'a [u8] },
    ZstdModeled { build: u8, recipe: &'a [u8], text: &'a [u8] },
    /// `Png` with a recipe in place of the corrections
    PngReflate { header: &'a [u8], recipe: &'a [u8], text: &'a [u8], adler: &'a [u8], chunks: Vec<(usize, &'a [u8])> },
    ReflateJpeg { recipe: &'a [u8], lepton: &'a [u8] },
    Nested(Inner<'a>),
    Png { header: &'a [u8], corrections: &'a [u8], text: &'a [u8], adler: &'a [u8], chunks: Vec<(usize, &'a [u8])> },
    Jpeg(&'a [u8]),
}

/// Reads fields off a recipe body, the content and the side in step.
struct Reader<'a> {
    body: &'a [u8],
    pos: usize,
    content: &'a [u8],
    at: usize,
    side: &'a [u8],
    side_at: usize,
}

impl<'a> Reader<'a> {
    fn varint(&mut self) -> Option<usize> {
        get_varint(self.body, &mut self.pos).ok().map(|v| v as usize)
    }
    fn fixed(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.body.get(self.pos..self.pos.checked_add(n)?)?;
        self.pos += n;
        Some(s)
    }
    fn content(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.content.get(self.at..self.at.checked_add(n)?)?;
        self.at += n;
        Some(s)
    }
    fn side(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.side.get(self.side_at..self.side_at.checked_add(n)?)?;
        self.side_at += n;
        Some(s)
    }
    fn chunks(&mut self) -> Option<Vec<(usize, &'a [u8], u8)>> {
        let k = self.varint()?;
        let mut chunks = Vec::with_capacity(k.min(1 << 16));
        for _ in 0..k {
            let (plain, c) = (self.varint()?, self.varint()?);
            let bit = *self.body.get(self.pos)?;
            self.pos += 1;
            chunks.push((plain, self.side(c)?, bit));
        }
        Some(chunks)
    }
    fn inner(&mut self) -> Option<Inner<'a>> {
        let segments = self.varint()? as u64;
        let len = self.varint()?;
        let body = self.fixed(len)?;
        let (clen, slen) = (self.varint()?, self.varint()?);
        Some(Inner { segments, body, content: self.content(clen)?, side: self.side(slen)? })
    }
}

fn segments<'a>(inner: &Inner<'a>) -> Option<Vec<Seg<'a>>> {
    let mut r = Reader { body: inner.body, pos: 0, content: inner.content, at: 0, side: inner.side, side_at: 0 };
    let mut out = Vec::with_capacity(inner.segments.min(1 << 20) as usize);
    for _ in 0..inner.segments {
        let tag = *r.body.get(r.pos)?;
        r.pos += 1;
        out.push(match tag {
            KEEP => {
                let n = r.varint()?;
                Seg::Bytes(r.side(n)?)
            }
            DEFLATE => {
                let (c, t) = (r.varint()?, r.varint()?);
                Seg::Deflate { corrections: r.side(c)?, text: r.content(t)? }
            }
            PNG_IMAGE | PNG_REFLATE => {
                let header = r.fixed(2)?;
                let (c, t) = (r.varint()?, r.varint()?);
                let adler = r.fixed(4)?;
                let k = r.varint()?;
                let mut chunks = Vec::with_capacity(k.min(1 << 16));
                for _ in 0..k {
                    let len = r.varint()?;
                    chunks.push((len, r.fixed(4)?));
                }
                let (side, text) = (r.side(c)?, r.content(t)?);
                if tag == PNG_IMAGE {
                    Seg::Png { header, corrections: side, text, adler, chunks }
                } else {
                    Seg::PngReflate { header, recipe: side, text, adler, chunks }
                }
            }
            JPEG => {
                let n = r.varint()?;
                Seg::Jpeg(r.side(n)?)
            }
            DEFLATE_JPEG => {
                let (c, l) = (r.varint()?, r.varint()?);
                Seg::DeflateJpeg { corrections: r.side(c)?, lepton: r.side(l)? }
            }
            REFLATE_JPEG => {
                let (c, l) = (r.varint()?, r.varint()?);
                Seg::ReflateJpeg { recipe: r.side(c)?, lepton: r.side(l)? }
            }
            DEFLATE_NESTED => {
                let c = r.varint()?;
                let corrections = r.side(c)?;
                Seg::DeflateNested { corrections, inner: r.inner()? }
            }
            NESTED => Seg::Nested(r.inner()?),
            DEFLATE_CHUNKED => {
                let chunks = r.chunks()?;
                let t = r.varint()?;
                Seg::DeflateChunked { chunks, text: r.content(t)? }
            }
            DEFLATE_NESTED_CHUNKED => {
                let chunks = r.chunks()?;
                Seg::DeflateNestedChunked { chunks, inner: r.inner()? }
            }
            REFLATE => {
                let (c, t) = (r.varint()?, r.varint()?);
                Seg::Reflate { recipe: r.side(c)?, text: r.content(t)? }
            }
            SNAPPY_PAGE => {
                let build = r.fixed(1)?[0];
                let t = r.varint()?;
                Seg::Snappy { build, text: r.content(t)? }
            }
            SNAPPY_MODELED => {
                let build = r.fixed(1)?[0];
                let (c, t) = (r.varint()?, r.varint()?);
                Seg::SnappyModeled { build, recipe: r.side(c)?, text: r.content(t)? }
            }
            ZSTD_FRAME => {
                let build = r.fixed(1)?[0];
                let t = r.varint()?;
                Seg::Zstd { build, text: r.content(t)? }
            }
            ZSTD_MODELED => {
                let build = r.fixed(1)?[0];
                let (c, t) = (r.varint()?, r.varint()?);
                Seg::ZstdModeled { build, recipe: r.side(c)?, text: r.content(t)? }
            }
            REFLATE_NESTED => {
                let c = r.varint()?;
                let recipe = r.side(c)?;
                Seg::ReflateNested { recipe, inner: r.inner()? }
            }
            _ => return None,
        });
    }
    (r.pos == r.body.len() && r.at == r.content.len() && r.side_at == r.side.len()).then_some(out)
}

fn produce<'a>(seg: &Seg<'a>) -> Option<Cow<'a, [u8]>> {
    Some(match seg {
        Seg::Bytes(b) => Cow::Borrowed(*b),
        Seg::Deflate { corrections, text } => Cow::Owned(recreate_whole_deflate_stream(text, corrections).ok()?),
        Seg::DeflateNested { corrections, inner } => Cow::Owned(recreate_whole_deflate_stream(&close_inner(inner, false)?, corrections).ok()?),
        Seg::DeflateChunked { chunks, text } => Cow::Owned(recreate_chunked(chunks, text)?),
        Seg::DeflateNestedChunked { chunks, inner } => Cow::Owned(recreate_chunked(chunks, &close_inner(inner, false)?)?),
        Seg::Reflate { recipe, text } => Cow::Owned(crate::reflate::close(text, recipe)?),
        Seg::Snappy { build, text } => Cow::Owned(crate::resnappy::compress(text, *crate::resnappy::BUILDS.get(*build as usize)?)),
        Seg::SnappyModeled { build, recipe, text } => Cow::Owned(crate::resnappy::compress(&crate::parquet::unmodel(recipe, text)?, *crate::resnappy::BUILDS.get(*build as usize)?)),
        Seg::Zstd { build, text } => Cow::Owned(crate::rezstd::compress(text, *crate::rezstd::BUILDS.get(*build as usize)?)),
        Seg::ZstdModeled { build, recipe, text } => Cow::Owned(crate::rezstd::compress(&crate::parquet::unmodel(recipe, text)?, *crate::rezstd::BUILDS.get(*build as usize)?)),
        Seg::ReflateNested { recipe, inner } => Cow::Owned(crate::reflate::close(&close_inner(inner, false)?, recipe)?),
        Seg::Nested(inner) => Cow::Owned(close_inner(inner, false)?),
        Seg::Png { header, corrections, text, adler, chunks } => Cow::Owned(png_chunks(header, &recreate_whole_deflate_stream(text, corrections).ok()?, adler, chunks)?),
        Seg::PngReflate { header, recipe, text, adler, chunks } => Cow::Owned(png_chunks(header, &crate::reflate::close(text, recipe)?, adler, chunks)?),
        Seg::Jpeg(lepton) => Cow::Owned(crate::jpeg::restore(lepton)?),
        Seg::DeflateJpeg { corrections, lepton } => Cow::Owned(recreate_whole_deflate_stream(&crate::jpeg::restore(lepton)?, corrections).ok()?),
        Seg::ReflateJpeg { recipe, lepton } => Cow::Owned(crate::reflate::close(&crate::jpeg::restore(lepton)?, recipe)?),
    })
}

/// A chunked stream re-created, every chunk on its own core.
fn recreate_chunked(chunks: &[(usize, &[u8], u8)], text: &[u8]) -> Option<Vec<u8>> {
    let params = chunked::parameters(chunks.first()?.1).ok()?;
    let mut starts = Vec::with_capacity(chunks.len());
    let mut at = 0usize;
    for (plain, _, _) in chunks {
        starts.push(at);
        at = at.checked_add(*plain)?;
    }
    if at != text.len() {
        return None;
    }
    let pieces = each(chunks.len(), true, |i| {
        let (plain, corrections, bit) = chunks[i];
        chunked::recreate(&params, text, starts[i]..starts[i] + plain, corrections, i == 0, bit as u32).ok()
    });
    let pieces: Vec<(Vec<u8>, u32)> = pieces.into_iter().collect::<Option<_>>()?;
    Some(chunked::join(&pieces))
}

/// A PNG's IDAT chunks back around its zlib stream.
fn png_chunks(header: &[u8], stream: &[u8], adler: &[u8], chunks: &[(usize, &[u8])]) -> Option<Vec<u8>> {
    let mut image = header.to_vec();
    image.extend_from_slice(stream);
    image.extend_from_slice(adler);
    let mut out = Vec::with_capacity(image.len() + chunks.len() * 12);
    let mut off = 0usize;
    for &(len, crc) in chunks {
        out.extend_from_slice(&(len as u32).to_be_bytes());
        out.extend_from_slice(b"IDAT");
        out.extend_from_slice(image.get(off..off.checked_add(len)?)?);
        out.extend_from_slice(crc);
        off += len;
    }
    (off == image.len()).then_some(out)
}

fn close_inner(inner: &Inner<'_>, parallel: bool) -> Option<Vec<u8>> {
    let segs = segments(inner)?;
    let size = |i: usize| match &segs[i] {
        Seg::Deflate { text, .. } | Seg::DeflateChunked { text, .. } | Seg::Reflate { text, .. } | Seg::Snappy { text, .. } | Seg::SnappyModeled { text, .. } | Seg::Zstd { text, .. } | Seg::ZstdModeled { text, .. } | Seg::Png { text, .. } | Seg::PngReflate { text, .. } => text.len(),
        Seg::DeflateNested { inner, .. } | Seg::DeflateNestedChunked { inner, .. } | Seg::ReflateNested { inner, .. } | Seg::Nested(inner) => inner.content.len(),
        _ => 0,
    };
    let parts = each_sized(segs.len(), parallel, size, |i| produce(&segs[i]));
    let mut out = Vec::with_capacity(parts.iter().map(|p| p.as_ref().map_or(0, |p| p.len())).sum());
    for p in parts {
        out.extend_from_slice(&p?);
    }
    Some(out)
}

/// The object back from its plain text and recipe (the segments'
/// streams recreated on every core).
pub fn close(recipe: &[u8], plain: &[u8]) -> Option<Vec<u8>> {
    let mut pos = 0usize;
    let segments = get_varint(recipe, &mut pos).ok()?;
    let clen = get_varint(recipe, &mut pos).ok()? as usize;
    let inner = Inner { segments, body: &recipe[pos..], content: plain.get(..clen)?, side: plain.get(clen..)? };
    close_inner(&inner, true)
}

/// What the object's deflate streams hold, without re-creating any of
/// them: a gzip's members' text one after the other (what `gunzip`
/// prints), a zlib stream's text, a tar.gz's tar. `None` for an object
/// that is not one stream of content: a zip, a PDF, a PNG, a tar.
fn content_of(inner: &Inner<'_>) -> Option<Vec<u8>> {
    let segs = segments(inner)?;
    let first = segs.iter().find_map(|s| if let Seg::Bytes(b) = s { Some(*b) } else { None })?;
    let gzip_head = first.len() >= 3 && first[0] == 0x1f && first[1] == 0x8b && first[2] == 8;
    let zlib_head = first.len() >= 2 && first[0] & 0x0f == 8 && (u16::from_be_bytes([first[0], first[1]]) % 31 == 0);
    if !(gzip_head || zlib_head) {
        return None;
    }
    let mut out = Vec::new();
    for s in &segs {
        match s {
            Seg::Bytes(_) => {}
            Seg::Deflate { text, .. } | Seg::DeflateChunked { text, .. } | Seg::Reflate { text, .. } | Seg::Zstd { text, .. } => out.extend_from_slice(text),
            Seg::DeflateNested { inner, .. } | Seg::DeflateNestedChunked { inner, .. } | Seg::ReflateNested { inner, .. } => out.extend_from_slice(&close_inner(inner, true)?),
            Seg::DeflateJpeg { lepton, .. } | Seg::ReflateJpeg { lepton, .. } => out.extend_from_slice(&crate::jpeg::restore(lepton)?),
            _ => return None,
        }
    }
    Some(out)
}

fn content_plain(recipe: &[u8], plain: &[u8]) -> Option<Vec<u8>> {
    let mut pos = 0usize;
    let segments = get_varint(recipe, &mut pos).ok()?;
    let clen = get_varint(recipe, &mut pos).ok()? as usize;
    content_of(&Inner { segments, body: &recipe[pos..], content: plain.get(..clen)?, side: plain.get(clen..)? })
}

const NO_CONTENT: crate::CodecError = crate::CodecError::CorruptedBitstream("deflate envelope: the object is not one gzip or zlib stream of content");

/// `content_of` for an envelope, its inner stream decoded by `inner`;
/// `None` when `compressed` is no envelope of this version.
pub(crate) fn content(compressed: &[u8], inner: impl FnOnce(&[u8]) -> crate::Result<Vec<u8>>) -> Option<crate::Result<Vec<u8>>> {
    let (_, recipe, stream, kind) = parse_any(compressed)?;
    if !matches!(kind, Kind::Current | Kind::Def2) {
        return None;
    }
    Some(inner(stream).and_then(|plain| content_plain(&recipe, &plain).ok_or(NO_CONTENT)))
}

/// `content` for a base-mode envelope (see `unwrap_with_base`).
pub(crate) fn content_with_base(base: &[u8], compressed: &[u8], inner: impl FnOnce(&[u8], &[u8]) -> crate::Result<Vec<u8>>) -> Option<crate::Result<Vec<u8>>> {
    let (_, recipe, stream, kind) = parse_any(compressed)?;
    let base_plain = match kind {
        Kind::Current => open(base).map(|o| o.plain),
        Kind::Def2 => open_with(base, Engine::Preflate).map(|o| o.plain),
        _ => return None,
    };
    Some(inner(base_plain.as_deref().unwrap_or(base), stream).and_then(|plain| content_plain(&recipe, &plain).ok_or(NO_CONTENT)))
}

/// The envelope's head: everything before the inner stream. The recipe
/// goes in compressed at the max level.
pub fn envelope(original_len: usize, recipe: &[u8], out: &mut Vec<u8>) {
    let mut packed = Vec::with_capacity(recipe.len() / 2 + 64);
    crate::as_part(|| crate::compress_into_max(recipe, &mut packed));
    out.extend_from_slice(MAGIC);
    put_varint(out, original_len as u64);
    put_varint(out, packed.len() as u64);
    out.extend_from_slice(&packed);
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Current,
    Def2,
    V013,
    V012,
}

/// (original length, recipe, inner stream, kind) of an envelope.
fn parse_any(compressed: &[u8]) -> Option<(usize, Vec<u8>, &[u8], Kind)> {
    if compressed.len() < 10 {
        return None;
    }
    let kind = match &compressed[..8] {
        m if m == MAGIC => Kind::Current,
        m if m == MAGIC_DEF2 => Kind::Def2,
        m if m == MAGIC_V013 => Kind::V013,
        m if m == MAGIC_V012 => Kind::V012,
        _ => return None,
    };
    let mut pos = 8usize;
    let original = get_varint(compressed, &mut pos).ok()? as usize;
    let rlen = get_varint(compressed, &mut pos).ok()? as usize;
    let stored = compressed.get(pos..pos.checked_add(rlen)?)?;
    // v0.12.0 stored its recipe as it was; later ones compress it.
    let recipe = if kind == Kind::V012 { stored.to_vec() } else { crate::decompress(stored).ok()? };
    Some((original, recipe, &compressed[pos + rlen..], kind))
}

/// The original length of an envelope, when `compressed` is one.
pub(crate) fn original_len(compressed: &[u8]) -> Option<usize> {
    if compressed.len() < 10 || ![&MAGIC[..], &MAGIC_DEF2[..], &MAGIC_V013[..], &MAGIC_V012[..]].contains(&&compressed[..8]) {
        return None;
    }
    let mut pos = 8usize;
    get_varint(compressed, &mut pos).ok().map(|v| v as usize)
}

fn close_kind(kind: Kind, recipe: &[u8], plain: &[u8]) -> Option<Vec<u8>> {
    match kind {
        Kind::Current | Kind::Def2 => close(recipe, plain),
        Kind::V013 => legacy::close_v013(recipe, plain),
        Kind::V012 => legacy::close_v012(recipe, plain),
    }
}

/// Whether `b` starts with an envelope of v0.12.0 or v0.13.0: those
/// versions could write one where a block of a stream was due.
pub(crate) fn legacy_magic(b: &[u8]) -> bool {
    b.starts_with(MAGIC_V012) || b.starts_with(MAGIC_V013)
}

/// Such an envelope's head: where its inner stream starts, and how much
/// plain text its recipe takes.
pub(crate) fn embedded_head(b: &[u8]) -> Option<(usize, usize)> {
    let (_, recipe, stream, kind) = parse_any(b)?;
    let need = match kind {
        Kind::V012 => legacy::need_v012(&recipe)?,
        Kind::V013 => legacy::need_v013(&recipe)?,
        Kind::Current | Kind::Def2 => return None,
    };
    Some((b.len() - stream.len(), need))
}

/// Such an envelope closed over its inner stream's plain text.
pub(crate) fn embedded_close(b: &[u8], plain: &[u8]) -> Option<Vec<u8>> {
    let (original, recipe, _, kind) = parse_any(b)?;
    close_kind(kind, &recipe, plain).filter(|o| o.len() == original)
}

/// `input` compressed as an opened container by `inner` when it is one
/// and that is smaller than `closed_level` on it as it is; then the
/// smaller of the two is written and `true` returned. `false`, with
/// nothing written, for anything else. Never on a part of a stream.
pub(crate) fn wrap(input: &[u8], output: &mut Vec<u8>, inner: impl FnOnce(&[u8], &mut Vec<u8>), closed_level: impl FnOnce(&[u8], &mut Vec<u8>)) -> bool {
    if crate::in_part() || !is_container(input) {
        return false;
    }
    if crate::jpeg::is_jpeg(input) {
        return crate::jpeg::wrap(input, output);
    }
    let Some(opened) = open(input) else { return false };
    // Opened against closed, both at the caller's level: a container's
    // own deflate can be as good as the level on its content (an Office
    // sheet at the fast levels).
    let mut wrapped = Vec::with_capacity(opened.plain.len() / 4 + opened.recipe.len() + 32);
    envelope(input.len(), &opened.recipe, &mut wrapped);
    crate::as_part(|| inner(&opened.plain, &mut wrapped));
    drop(opened);
    let mut closed = Vec::with_capacity(input.len() / 2 + 64);
    crate::as_part(|| closed_level(input, &mut closed));
    output.extend_from_slice(if wrapped.len() < closed.len() { &wrapped } else { &closed });
    true
}

/// The original object of an envelope, its inner stream decoded by
/// `inner`; `None` when `compressed` is no envelope.
pub(crate) fn unwrap(compressed: &[u8], inner: impl FnOnce(&[u8]) -> crate::Result<Vec<u8>>) -> Option<crate::Result<Vec<u8>>> {
    if let Some(r) = crate::jpeg::unwrap(compressed) {
        return Some(r);
    }
    let (original, recipe, stream, kind) = parse_any(compressed)?;
    Some(inner(stream).and_then(|plain| match close_kind(kind, &recipe, &plain) {
        Some(out) if out.len() == original => Ok(out),
        _ => Err(crate::CodecError::CorruptedBitstream("deflate envelope: the object does not close")),
    }))
}

/// `unwrap` for a base-mode envelope: `inner` gets the base as its
/// encoder saw it (opened the way of the envelope's version) and the
/// inner stream.
pub(crate) fn unwrap_with_base(base: &[u8], compressed: &[u8], inner: impl FnOnce(&[u8], &[u8]) -> crate::Result<Vec<u8>>) -> Option<crate::Result<Vec<u8>>> {
    let (original, recipe, stream, kind) = parse_any(compressed)?;
    let base_plain = match kind {
        Kind::Current => open(base).map(|o| o.plain),
        Kind::Def2 => open_with(base, Engine::Preflate).map(|o| o.plain),
        Kind::V013 => legacy::base_plain_v013(base),
        Kind::V012 => legacy::base_plain_v012(base),
    };
    Some(inner(base_plain.as_deref().unwrap_or(base), stream).and_then(|plain| match close_kind(kind, &recipe, &plain) {
        Some(out) if out.len() == original => Ok(out),
        _ => Err(crate::CodecError::CorruptedBitstream("deflate envelope: the object does not close")),
    }))
}

/// The envelopes of earlier versions.
mod legacy {
    use super::*;

    /// v0.12.0: per member a header, corrections, the plain length and
    /// an 8-byte trailer, in the recipe; then what followed the members.
    pub(super) fn close_v012(recipe: &[u8], plain: &[u8]) -> Option<Vec<u8>> {
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
        for _ in 0..n {
            let header = take(&mut pos)?;
            let corrections = take(&mut pos)?;
            let plen = get_varint(recipe, &mut pos).ok()? as usize;
            let trailer = recipe.get(pos..pos + 8)?;
            pos += 8;
            let text = plain.get(at..at.checked_add(plen)?)?;
            out.extend_from_slice(header);
            out.extend_from_slice(&recreate_whole_deflate_stream(text, corrections).ok()?);
            out.extend_from_slice(trailer);
            at += plen;
        }
        let tail = take(&mut pos)?;
        if at != plain.len() || pos != recipe.len() {
            return None;
        }
        out.extend_from_slice(tail);
        Some(out)
    }

    /// v0.13.0: the kept bytes (tag 0), corrections and stored JPEGs'
    /// Lepton streams (tag 3) in the recipe; the plain text only what
    /// the streams held.
    pub(super) fn close_v013(recipe: &[u8], plain: &[u8]) -> Option<Vec<u8>> {
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
                KEEP => {
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
                JPEG => {
                    pos += 1;
                    out.extend_from_slice(&crate::jpeg::restore(take(&mut pos)?)?);
                }
                DEFLATE_NESTED => {
                    pos += 1;
                    let corrections = take(&mut pos)?;
                    let inner = take(&mut pos)?;
                    let plen = get_varint(recipe, &mut pos).ok()? as usize;
                    let object = close_v013(inner, plain.get(at..at.checked_add(plen)?)?)?;
                    out.extend_from_slice(&recreate_whole_deflate_stream(&object, corrections).ok()?);
                    at += plen;
                }
                NESTED => {
                    pos += 1;
                    let inner = take(&mut pos)?;
                    let plen = get_varint(recipe, &mut pos).ok()? as usize;
                    out.extend_from_slice(&close_v013(inner, plain.get(at..at.checked_add(plen)?)?)?);
                    at += plen;
                }
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

    /// The plain text a v0.12.0 recipe takes.
    pub(super) fn need_v012(recipe: &[u8]) -> Option<usize> {
        let mut pos = 0usize;
        let n = get_varint(recipe, &mut pos).ok()?;
        let mut need = 0usize;
        let skip = |pos: &mut usize| -> Option<()> {
            let len = get_varint(recipe, pos).ok()? as usize;
            *pos = pos.checked_add(len).filter(|&p| p <= recipe.len())?;
            Some(())
        };
        for _ in 0..n {
            skip(&mut pos)?;
            skip(&mut pos)?;
            need = need.checked_add(get_varint(recipe, &mut pos).ok()? as usize)?;
            pos += 8;
        }
        Some(need)
    }

    /// The plain text a v0.13.0 recipe takes.
    pub(super) fn need_v013(recipe: &[u8]) -> Option<usize> {
        let mut pos = 0usize;
        let n = get_varint(recipe, &mut pos).ok()?;
        let mut need = 0usize;
        let skip = |pos: &mut usize| -> Option<()> {
            let len = get_varint(recipe, pos).ok()? as usize;
            *pos = pos.checked_add(len).filter(|&p| p <= recipe.len())?;
            Some(())
        };
        let mut plain = |pos: &mut usize| -> Option<()> {
            need = need.checked_add(get_varint(recipe, pos).ok()? as usize)?;
            Some(())
        };
        for _ in 0..n {
            let tag = *recipe.get(pos)?;
            pos += 1;
            match tag {
                KEEP | JPEG => skip(&mut pos)?,
                DEFLATE | DEFLATE_JPEG => {
                    skip(&mut pos)?;
                    plain(&mut pos)?;
                }
                PNG_IMAGE => {
                    pos += 2;
                    skip(&mut pos)?;
                    plain(&mut pos)?;
                    pos += 4;
                    let k = get_varint(recipe, &mut pos).ok()?;
                    for _ in 0..k {
                        get_varint(recipe, &mut pos).ok()?;
                        pos += 4;
                    }
                }
                DEFLATE_NESTED => {
                    skip(&mut pos)?;
                    skip(&mut pos)?;
                    plain(&mut pos)?;
                }
                NESTED => {
                    skip(&mut pos)?;
                    plain(&mut pos)?;
                }
                _ => return None,
            }
        }
        Some(need)
    }

    /// A base's plain text as v0.13.0 made it (what its streams held,
    /// nothing kept, stored JPEGs left out), from the base opened now:
    /// the same streams open, so the same bytes come out.
    pub(super) fn base_plain_v013(base: &[u8]) -> Option<Vec<u8>> {
        let parts = open_parts(base, 0, Engine::Preflate)?;
        let mut out = Vec::new();
        collect_v013(&Inner { segments: parts.segments, body: &parts.body, content: &parts.content, side: &parts.side }, &mut out)?;
        Some(out)
    }

    fn collect_v013(inner: &Inner<'_>, out: &mut Vec<u8>) -> Option<()> {
        for seg in segments(inner)? {
            match seg {
                Seg::Deflate { text, .. } | Seg::DeflateChunked { text, .. } | Seg::Reflate { text, .. } | Seg::Snappy { text, .. } | Seg::SnappyModeled { text, .. } | Seg::Zstd { text, .. } | Seg::ZstdModeled { text, .. } | Seg::Png { text, .. } | Seg::PngReflate { text, .. } => out.extend_from_slice(text),
                Seg::DeflateJpeg { lepton, .. } | Seg::ReflateJpeg { lepton, .. } => out.extend_from_slice(lepton),
                Seg::DeflateNested { inner, .. } | Seg::DeflateNestedChunked { inner, .. } | Seg::ReflateNested { inner, .. } | Seg::Nested(inner) => collect_v013(&inner, out)?,
                Seg::Bytes(_) | Seg::Jpeg(_) => {}
            }
        }
        Some(())
    }

    /// A gzip base's plain text as v0.12.0 made it: every member's
    /// content, as it was.
    pub(super) fn base_plain_v012(base: &[u8]) -> Option<Vec<u8>> {
        if !is_gzip(base) {
            return None;
        }
        let parts = open_parts(base, 0, Engine::Preflate)?;
        let mut out = Vec::new();
        for seg in segments(&Inner { segments: parts.segments, body: &parts.body, content: &parts.content, side: &parts.side })? {
            match seg {
                Seg::Deflate { text, .. } => out.extend_from_slice(text),
                Seg::DeflateNested { inner, .. } => out.extend_from_slice(&close_inner(&inner, false)?),
                Seg::DeflateJpeg { lepton, .. } => out.extend_from_slice(&crate::jpeg::restore(lepton)?),
                _ => {}
            }
        }
        Some(out)
    }
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

    /// Whether `plain` holds `text` as one run: the plain text is the
    /// streams' content, then the side data.
    fn holds(plain: &[u8], text: &[u8]) -> bool {
        let probe = &text[..text.len().min(64)];
        plain.windows(probe.len()).enumerate().any(|(i, w)| w == probe && plain.get(i..i + text.len()) == Some(text))
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
            assert!(holds(&round_trip(&gz, "gzip"), &plain));
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
        let opened_two = round_trip(&two, "two members");
        assert!(holds(&opened_two, &plain[..1 << 20]) && holds(&opened_two, &plain[1 << 20..]));
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

    /// A part of a stream is never opened as a container of its own: a
    /// gzip member that starts exactly at a unit boundary of a
    /// multi-unit stream stays in the stream's plain blocks. (Opened, the
    /// unit's envelope sat inside a block stream and the file would not
    /// decode.)
    #[test]
    fn a_unit_is_not_opened_on_its_own() {
        let member = gzip(&text(1 << 20), &["-1"]);
        let mut input = text(8 << 20);
        input.extend_from_slice(&member);
        input.extend_from_slice(&text(7 << 20));
        for f in [crate::compress_parallel_into_max, crate::compress_parallel_into, crate::compress_records_into_max] {
            let mut c = Vec::new();
            f(&input, &mut c);
            assert!(crate::decompress(&c).unwrap() == input);
        }
        let mut c = Vec::new();
        crate::compress_stream(&input, crate::compress_into_max, crate::format::PARALLEL_UNIT_MAX, |u| {
            c.extend_from_slice(u);
            Ok(())
        })
        .unwrap();
        assert!(crate::decompress(&c).unwrap() == input);
    }

    /// A stream holding two chunks or more opens in chunks: the object
    /// closes bit for bit, its content reads, and it takes a base.
    #[test]
    fn large_streams_open_and_close_in_chunks() {
        let plain = text(2 * CHUNK_PLAIN + (1 << 20));
        let gz = gzip(&plain, &["-1"]);
        let opened = open(&gz).unwrap();
        let mut pos = 0usize;
        let segments_n = get_varint(&opened.recipe, &mut pos).unwrap();
        let clen = get_varint(&opened.recipe, &mut pos).unwrap() as usize;
        let inner = Inner { segments: segments_n, body: &opened.recipe[pos..], content: &opened.plain[..clen], side: &opened.plain[clen..] };
        let segs = segments(&inner).unwrap();
        assert!(segs.iter().any(|s| matches!(s, Seg::Reflate { .. })), "opened by reflate");
        assert_eq!(close(&opened.recipe, &opened.plain).unwrap(), gz);
        let mut c = Vec::new();
        crate::compress_into_max(&gz, &mut c);
        assert!(c.starts_with(MAGIC) && c.len() < gz.len());
        assert!(crate::decompress(&c).unwrap() == gz);
        assert!(crate::decompress_parallel(&c).unwrap() == gz);
        assert!(crate::decompress_content(&c).unwrap() == plain);
        let mut later = plain.clone();
        later.extend_from_slice(b"one more line\n");
        let b = gzip(&later, &["-1"]);
        let mut d = Vec::new();
        crate::compress_with_base(&gz, &b, &mut d, false);
        assert!(d.len() < b.len() / 8, "{} of {}", d.len(), b.len());
        assert!(crate::decompress_with_base(&gz, &d).unwrap() == b);
    }

    /// A gzip object's content comes back without its stream being
    /// re-created: what gunzip prints, members one after the other; an
    /// object stored closed has no content view; against a base too.
    #[test]
    fn content_reads_skip_the_recreate() {
        let plain = text(2 << 20);
        let gz = gzip(&plain, &["-1"]);
        let mut c = Vec::new();
        crate::compress_into_max(&gz, &mut c);
        assert!(c.starts_with(MAGIC));
        assert!(crate::decompress_content(&c).unwrap() == plain);
        let mut two = gz.clone();
        two.extend(gzip(&plain[..1 << 20], &["-1"]));
        let mut c = Vec::new();
        crate::compress_into_max(&two, &mut c);
        let mut both = plain.clone();
        both.extend_from_slice(&plain[..1 << 20]);
        assert!(crate::decompress_content(&c).unwrap() == both);
        assert!(crate::decompress_content(&crate::compress(&gz)).is_err());
        let mut later = plain.clone();
        later.extend_from_slice(b"one more line\n");
        let b = gzip(&later, &["-1"]);
        let mut d = Vec::new();
        crate::compress_with_base(&gz, &b, &mut d, false);
        assert!(crate::decompress_content_with_base(&gz, &d).unwrap() == later);
    }

    /// A Parquet file with snappy pages opens into its raw pages and
    /// closes to the same bytes; one with zstd pages is left as it is.
    #[test]
    fn parquet_snappy_pages_open_and_close() {
        let data = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/parquet/tiny.snappy.parquet")).unwrap();
        let o = open(&data).expect("opens");
        assert!(o.recipe.len() < 200, "recipe {} B", o.recipe.len());
        assert!(o.plain.len() > data.len(), "the raw pages outsize the file: {} vs {}", o.plain.len(), data.len());
        assert!(close(&o.recipe, &o.plain).unwrap() == data);
        let mut out = Vec::new();
        crate::compress_into_max(&data, &mut out);
        assert!(crate::decompress(&out).unwrap() == data);
        assert!(out.len() * 10 < data.len() * 9, "opened, the file compresses: {} of {}", out.len(), data.len());
        let zstd = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/parquet/tiny.zstd.parquet")).unwrap();
        let o = open(&zstd).expect("zstd pages open too");
        assert!(close(&o.recipe, &o.plain).unwrap() == zstd);
        let mut out = Vec::new();
        crate::compress_into_max(&zstd, &mut out);
        assert!(crate::decompress(&out).unwrap() == zstd);
        assert!(out.len() * 10 < zstd.len() * 9, "opened, the zstd file compresses: {} of {}", out.len(), zstd.len());
    }

    /// `GLYD_CONTAINER_FILE=x cargo test --release deflate::tests::engines -- --ignored --nocapture`:
    /// the file opened by both engines, the segments and their side
    /// bytes compared.
    #[test]
    #[ignore = "a probe on a file named by GLYD_CONTAINER_FILE"]
    fn engines() {
        let Ok(path) = std::env::var("GLYD_CONTAINER_FILE") else { return };
        let data = std::fs::read(path).unwrap();
        for engine in [Engine::Preflate, Engine::Reflate] {
            let t = std::time::Instant::now();
            let o = open_with(&data, engine).unwrap();
            let open_s = t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            assert!(close(&o.recipe, &o.plain).unwrap() == data);
            let close_s = t.elapsed().as_secs_f64();
            eprintln!("  close {close_s:.2} s");
            let mut pos = 0usize;
            let n = get_varint(&o.recipe, &mut pos).unwrap();
            let clen = get_varint(&o.recipe, &mut pos).unwrap() as usize;
            let inner = Inner { segments: n, body: &o.recipe[pos..], content: &o.plain[..clen], side: &o.plain[clen..] };
            let segs = segments(&inner).unwrap();
            let mut sizes: Vec<(usize, usize)> = Vec::new();
            for s in &segs {
                match s {
                    Seg::Deflate { corrections, text } => sizes.push((corrections.len(), text.len())),
                    Seg::DeflateChunked { chunks, text } => sizes.push((chunks.iter().map(|c| c.1.len()).sum(), text.len())),
                    Seg::Reflate { recipe, text } => sizes.push((recipe.len(), text.len())),
                    _ => {}
                }
            }
            let side: usize = sizes.iter().map(|s| s.0).sum();
            let text: usize = sizes.iter().map(|s| s.1).sum();
            let worst = sizes.iter().max_by_key(|s| s.0).copied();
            let small: usize = sizes.iter().filter(|s| s.1 < 4096).map(|s| s.0).sum();
            eprintln!("{:?}: opened in {open_s:.2} s; {} streams, {} B of text, {} B of corrections ({} B in streams under 4 KB, largest {:?}); content {} B, side {} B", if engine == Engine::Reflate { "reflate" } else { "preflate" }, sizes.len(), text, side, small, worst, clen, o.plain.len() - clen);
        }
    }

    /// The default, fast and turbo levels leave a container as it is: a
    /// read of an opened one re-creates its deflate, which those levels
    /// exist to be faster than. From `--max` up it is opened.
    #[test]
    fn fast_levels_keep_a_container_closed() {
        let gz = gzip(&text(4 << 20), &["-1"]);
        for f in [crate::compress_into, crate::compress_into_fast, crate::compress_into_turbo, crate::compress_parallel_into] {
            let mut c = Vec::new();
            f(&gz, &mut c);
            assert!(parse_any(&c).is_none());
            assert!(crate::decompress(&c).unwrap() == gz);
        }
        let mut c = Vec::new();
        crate::compress_into_max(&gz, &mut c);
        assert!(c.starts_with(MAGIC) && c.len() < gz.len(), "{} of {}", c.len(), gz.len());
    }

    /// Two versions of a tar holding gzip members and plain files (an OS
    /// image's shape): the delta between them is small, the kept bytes
    /// and the corrections being in the plain text base mode sees.
    #[test]
    fn versions_of_a_tar_of_gzips_delta_small() {
        let make = |seed: usize| {
            let mut blob = Vec::new();
            let mut sizes = Vec::new();
            for i in 0..40 {
                let body = text(20_000 + i * 997);
                let member = if i % 2 == 0 { gzip(&body, &["-9"]) } else { body };
                sizes.push(member.len());
                blob.extend_from_slice(&member);
            }
            // The second version changes one member.
            if seed == 1 {
                let i = 20;
                let at: usize = sizes[..i].iter().sum();
                let changed = gzip(&text(20_000 + i * 997 + 50), &["-9"]);
                blob.splice(at..at + sizes[i], changed.iter().copied());
                sizes[i] = changed.len();
            }
            let spec: Vec<String> = sizes.iter().map(|n| n.to_string()).collect();
            python(&format!(r#"
import sys, tarfile, io
blob = sys.stdin.buffer.read()
sizes = [{}]
buf = io.BytesIO()
at = 0
with tarfile.open(fileobj=buf, mode="w") as t:
    for i, n in enumerate(sizes):
        info = tarfile.TarInfo(f"usr/share/doc/p{{i}}/changelog" + (".gz" if i % 2 == 0 else ""))
        info.size = n; info.mtime = 0
        t.addfile(info, io.BytesIO(blob[at:at + n])); at += n
sys.stdout.buffer.write(buf.getvalue())
"#, spec.join(",")), &blob)
        };
        let (a, b) = (make(0), make(1));
        let mut alone = Vec::new();
        crate::compress_into_max(&b, &mut alone);
        let mut d = Vec::new();
        crate::compress_with_base(&a, &b, &mut d, false);
        assert!(d.len() * 8 < alone.len(), "delta {} against alone {}", d.len(), alone.len());
        assert!(crate::decompress_with_base(&a, &d).unwrap() == b);
        assert!(crate::decompress(&alone).unwrap() == b);
    }

    /// Envelopes written by the released v0.12.0 and v0.13.0 CLIs
    /// (`tests/data/legacy/`) still decode, alone and against a base.
    #[test]
    fn earlier_envelopes_still_decode() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/legacy/");
        let read = |f: &str| std::fs::read(format!("{dir}{f}")).unwrap();
        for (glyd, original) in [("log.gz.v0120.glyd", "log.gz"), ("log.gz.v0130.glyd", "log.gz"), ("doc.zip.v0130.glyd", "doc.zip"), ("base.tar.v0130.glyd", "base.tar"), ("log.gz.v0133.glyd", "log.gz"), ("doc.zip.v0133.glyd", "doc.zip"), ("base.tar.v0133.glyd", "base.tar")] {
            assert!(crate::decompress(&read(glyd)).unwrap() == read(original), "{glyd}");
        }
        // v0.14.0's JPEG stream (GJPG): the model's contexts and
        // predictions are the format, so a change to them needs a new
        // stream tag, not a change here.
        let jpeg = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/jpeg/");
        for (glyd, original) in [("q75-420.jpg.v0140.glyd", "q75-420.jpg"), ("q60-422-rst.jpg.v0140.glyd", "q60-422-rst.jpg")] {
            assert!(crate::decompress(&read(glyd)).unwrap() == std::fs::read(format!("{jpeg}{original}")).unwrap(), "{glyd}");
        }
        // v0.13.3's content view too, and its deltas: their bases open
        // the preflate way, as they were made.
        assert!(crate::decompress_content(&read("log.gz.v0133.glyd")).is_ok(), "content of a GLYDDEF2 object");
        for (glyd, base, original) in [("next.gz.v0120.base.glyd", "base.gz", "next.gz"), ("next.gz.v0130.base.glyd", "base.gz", "next.gz"), ("next.tar.v0130.base.glyd", "base.tar", "next.tar"), ("next.gz.v0133.base.glyd", "base.gz", "next.gz"), ("next.tar.v0133.base.glyd", "base.tar", "next.tar")] {
            assert!(crate::decompress_with_base(&read(base), &read(glyd)).unwrap() == read(original), "{glyd}");
        }
    }

    /// Files v0.12.0 and v0.13.0 wrote at the max level with a gzip member
    /// on a unit boundary (an envelope where a block was due, which they
    /// could not read back) read back exactly.
    #[test]
    fn earlier_streams_with_an_embedded_envelope_read_back() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/legacy/");
        let read = |f: &str| std::fs::read(format!("{dir}{f}")).unwrap();
        let filler = |n: usize| b"filler line 0123456789\n".iter().cycle().take(n).copied().collect::<Vec<u8>>();
        let mut input = filler(8 << 20);
        input.extend_from_slice(&read("unit-member.gz"));
        input.extend_from_slice(&filler(64 << 10));
        for f in ["unit.v0120.max.glyd", "unit.v0130.max.glyd"] {
            let c = read(f);
            assert!(crate::decompress(&c).unwrap() == input, "{f}");
            assert!(crate::decompress_parallel(&c).unwrap() == input, "{f}");
            assert_eq!(crate::decompressed_len(&c).unwrap(), input.len());
            let mut streamed = Vec::new();
            crate::decompress_stream(&c, |b| {
                streamed.extend_from_slice(b);
                Ok(())
            })
            .unwrap();
            assert!(streamed == input, "{f}");
        }
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
        assert!(holds(&round_trip(&zlib, "zlib"), &plain));
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
        assert!(holds(&round_trip(&png, "png"), &plain));
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
        assert!(recipe.windows(1).any(|w| w[0] == DEFLATE_JPEG || w[0] == REFLATE_JPEG), "the deflated JPEG transcoded under its deflate");
        assert!(recipe.windows(1).any(|w| w[0] == DEFLATE_NESTED || w[0] == REFLATE_NESTED), "the deflated PNG opened under its deflate");
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
