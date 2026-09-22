//! Deflate containers opened: gzip, zip (so also .docx, .xlsx, .pptx,
//! .jar, .apk, .odt), zlib streams and PNG. The deflate streams inside
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
/// The plain text is held in memory; beyond this an object is left alone.
const PLAIN_LIMIT: usize = 8 << 30;
const VERBATIM: u8 = 0;
const DEFLATE: u8 = 1;
const PNG_IMAGE: u8 = 2;

/// An object opened: its plain text and the recipe to close it.
pub struct Opened {
    pub plain: Vec<u8>,
    pub recipe: Vec<u8>,
}

/// Whether `input` starts like a container worth opening.
pub fn is_container(input: &[u8]) -> bool {
    is_gzip(input) || is_zip(input) || is_png(input) || is_zlib(input)
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
}

impl Builder {
    fn new() -> Builder {
        Builder { plain: Vec::new(), recipe: Vec::new(), segments: 0, verbatim: (0, 0), config: PreflateConfig { plain_text_limit: PLAIN_LIMIT, verify_compression: true, ..Default::default() } }
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
    fn deflate(&mut self, input: &[u8], at: usize) -> Option<usize> {
        let (result, text) = preflate_whole_deflate_stream(input.get(at..)?, &self.config).ok()?;
        if self.plain.len() + text.text().len() > PLAIN_LIMIT {
            return None;
        }
        self.flush_verbatim(input);
        self.recipe.push(DEFLATE);
        put_varint(&mut self.recipe, result.corrections.len() as u64);
        self.recipe.extend_from_slice(&result.corrections);
        put_varint(&mut self.recipe, text.text().len() as u64);
        self.plain.extend_from_slice(text.text());
        self.segments += 1;
        self.verbatim = (at + result.compressed_size, at + result.compressed_size);
        Some(result.compressed_size)
    }

    fn finish(mut self, input: &[u8]) -> Option<Opened> {
        self.flush_verbatim(input);
        if self.segments == 0 || self.plain.is_empty() {
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
    if is_gzip(input) {
        open_gzip(input)
    } else if is_zip(input) {
        open_zip(input)
    } else if is_png(input) {
        open_png(input)
    } else if is_zlib(input) {
        open_zlib(input)
    } else {
        None
    }
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
fn open_gzip(input: &[u8]) -> Option<Opened> {
    let mut b = Builder::new();
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
fn open_zip(input: &[u8]) -> Option<Opened> {
    let mut b = Builder::new();
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
        // Stored data, or a stream that did not open: kept.
        b.verbatim(input, opened_to, end);
        // The next entry: right here, or past a data descriptor.
        let next = (end..(end + 32).min(input.len().saturating_sub(3))).find(|&p| &input[p..p + 4] == b"PK\x03\x04").unwrap_or(end);
        b.verbatim(input, end, next);
        at = next;
    }
    b.verbatim(input, at, input.len());
    b.finish(input)
}

/// A zlib stream: 2-byte header, deflate, 4-byte Adler-32; the rest kept.
fn open_zlib(input: &[u8]) -> Option<Opened> {
    let mut b = Builder::new();
    b.verbatim(input, 0, 2);
    let n = b.deflate(input, 2)?;
    b.verbatim(input, 2 + n, input.len());
    b.finish(input)
}

/// Chunks in order; the IDAT chunks' data, joined, is one zlib stream,
/// opened as a `PNG_IMAGE` segment that remembers how to cut it back
/// into chunks (their lengths and CRCs); every other chunk is kept.
fn open_png(input: &[u8]) -> Option<Opened> {
    let mut b = Builder::new();
    let be32 = |p: usize| input.get(p..p + 4).map(|s| u32::from_be_bytes(s.try_into().unwrap()) as usize);
    let mut at = 8usize;
    let mut idat: Vec<(usize, usize)> = Vec::new(); // (data start, len)
    let mut image = Vec::new();
    let mut first_idat = None;
    while let (Some(len), Some(kind)) = (be32(at), input.get(at + 4..at + 8)) {
        let chunk_end = at + 12 + len;
        if chunk_end > input.len() {
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
    b.verbatim(input, 0, start);
    b.flush_verbatim(input);
    let (result, text) = preflate_whole_deflate_stream(&image[2..], &b.config).ok()?;
    if 2 + result.compressed_size + 4 != image.len() || text.text().len() > PLAIN_LIMIT {
        return None;
    }
    b.recipe.push(PNG_IMAGE);
    b.recipe.extend_from_slice(&image[..2]);
    put_varint(&mut b.recipe, result.corrections.len() as u64);
    b.recipe.extend_from_slice(&result.corrections);
    put_varint(&mut b.recipe, text.text().len() as u64);
    b.recipe.extend_from_slice(&image[image.len() - 4..]);
    put_varint(&mut b.recipe, idat.len() as u64);
    for &(s, l) in &idat {
        put_varint(&mut b.recipe, l as u64);
        b.recipe.extend_from_slice(&input[s + l..s + l + 4]);
    }
    b.plain.extend_from_slice(text.text());
    b.segments += 1;
    b.verbatim = (last_end, last_end);
    b.verbatim(input, last_end, input.len());
    b.finish(input)
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
            _ => return None,
        }
    }
    if at != plain.len() || pos != recipe.len() {
        return None;
    }
    Some(out)
}

/// The envelope's head: everything before the inner stream.
pub fn envelope(original_len: usize, recipe: &[u8], out: &mut Vec<u8>) {
    out.extend_from_slice(MAGIC);
    put_varint(out, original_len as u64);
    put_varint(out, recipe.len() as u64);
    out.extend_from_slice(recipe);
}

/// (original length, recipe, inner stream) of an envelope.
pub(crate) fn parse(compressed: &[u8]) -> Option<(usize, &[u8], &[u8])> {
    if compressed.len() < 10 || &compressed[..8] != MAGIC {
        return None;
    }
    let mut pos = 8usize;
    let original = get_varint(compressed, &mut pos).ok()? as usize;
    let rlen = get_varint(compressed, &mut pos).ok()? as usize;
    let recipe = compressed.get(pos..pos.checked_add(rlen)?)?;
    Some((original, recipe, &compressed[pos + rlen..]))
}

/// `input` compressed as an opened container by `inner` when it is
/// one and that is smaller: the envelope, then `inner` on the plain
/// text. `false` otherwise.
pub(crate) fn wrap(input: &[u8], output: &mut Vec<u8>, inner: impl FnOnce(&[u8], &mut Vec<u8>)) -> bool {
    if !is_container(input) {
        return false;
    }
    let Some(opened) = open(input) else { return false };
    // The container's bytes themselves barely compress, so their length
    // is what the caller's level would give on them; an opened object
    // that costs more than that (a recipe of many corrections on
    // content the level does not beat deflate on) is left closed.
    let mut wrapped = Vec::with_capacity(opened.plain.len() / 4 + opened.recipe.len() + 32);
    envelope(input.len(), &opened.recipe, &mut wrapped);
    inner(&opened.plain, &mut wrapped);
    if wrapped.len() >= input.len() {
        return false;
    }
    output.extend_from_slice(&wrapped);
    true
}

/// The original object of an envelope, its inner stream decoded by
/// `inner`; `None` when `compressed` is no envelope.
pub(crate) fn unwrap(compressed: &[u8], inner: impl FnOnce(&[u8]) -> crate::Result<Vec<u8>>) -> Option<crate::Result<Vec<u8>>> {
    let (original, recipe, stream) = parse(compressed)?;
    Some(inner(stream).and_then(|plain| match close(recipe, &plain) {
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
        assert!(!wrap(b"\x1f\x8b\x08 but not really a gzip stream at all", &mut c, |_, _| {}));
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
        // A truncated PNG and a zip with garbage after an entry still
        // close to what they were.
        round_trip(&png[..png.len() - 7], "truncated png");
        let mut odd = zip.clone();
        odd.extend_from_slice(b"garbage at the end");
        round_trip(&odd, "zip with a tail");
    }
}
