//! Gzip objects opened. The deflate streams inside are decoded to their
//! plain text by `preflate`, which also records what it takes to
//! re-encode each bit for bit; the plain text then takes whatever the
//! caller asked for (a level, record mode, the cold level, a base), so
//! the models see the content, not the deflate. A gzipped log costs
//! what the log costs. Envelope:
//!
//!   "GLYDGZIP" original_len, recipe_len (varints), the recipe, then
//!   the inner Glyd stream of the plain text (any format: units of a
//!   level, records, cold, a base envelope).
//!
//! The recipe: n_members, then per member the gzip header verbatim,
//! preflate's corrections, the member's plain length and its 8-byte
//! trailer verbatim; then whatever followed the last member, verbatim.
//! An object that does not re-encode bit for bit (checked before it is
//! used) is left as it is.

use crate::record::{get_varint, put_varint};
use preflate_rs::{preflate_whole_deflate_stream, recreate_whole_deflate_stream, PreflateConfig};

pub(crate) const GZ_MAGIC: &[u8; 8] = b"GLYDGZIP";
/// A member's plain text is held in memory; beyond this it is left alone.
const PLAIN_LIMIT: usize = 8 << 30;

/// A gzip object opened: its plain text and the recipe to close it.
pub struct Opened {
    pub plain: Vec<u8>,
    pub recipe: Vec<u8>,
}

pub fn is_gzip(input: &[u8]) -> bool {
    input.len() >= 18 && input[0] == 0x1f && input[1] == 0x8b && input[2] == 8
}

/// The end of the gzip header starting at `at`.
fn header_end(input: &[u8], at: usize) -> Option<usize> {
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

/// The plain text and recipe of a gzip object, when every member of it
/// re-encodes bit for bit; `None` for anything else.
pub fn open(input: &[u8]) -> Option<Opened> {
    if !is_gzip(input) {
        return None;
    }
    let config = PreflateConfig { plain_text_limit: PLAIN_LIMIT, verify_compression: true, ..Default::default() };
    let mut plain = Vec::new();
    let mut members: Vec<u8> = Vec::new();
    let mut n = 0u64;
    let mut at = 0usize;
    while is_gzip(&input[at..]) {
        let hend = header_end(input, at)?;
        let (result, text) = preflate_whole_deflate_stream(&input[hend..], &config).ok()?;
        let dend = hend + result.compressed_size;
        let trailer = input.get(dend..dend + 8)?;
        put_varint(&mut members, (hend - at) as u64);
        members.extend_from_slice(&input[at..hend]);
        put_varint(&mut members, result.corrections.len() as u64);
        members.extend_from_slice(&result.corrections);
        put_varint(&mut members, text.text().len() as u64);
        members.extend_from_slice(trailer);
        plain.extend_from_slice(text.text());
        n += 1;
        at = dend + 8;
    }
    let mut recipe = Vec::with_capacity(members.len() + 16);
    put_varint(&mut recipe, n);
    recipe.extend_from_slice(&members);
    put_varint(&mut recipe, (input.len() - at) as u64);
    recipe.extend_from_slice(&input[at..]);
    Some(Opened { plain, recipe })
}

/// The gzip object back from its plain text and recipe.
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

/// The envelope's head: everything before the inner stream.
pub fn envelope(original_len: usize, recipe: &[u8], out: &mut Vec<u8>) {
    out.extend_from_slice(GZ_MAGIC);
    put_varint(out, original_len as u64);
    put_varint(out, recipe.len() as u64);
    out.extend_from_slice(recipe);
}

/// (original length, recipe, inner stream) of an envelope.
pub(crate) fn parse(compressed: &[u8]) -> Option<(usize, &[u8], &[u8])> {
    if compressed.len() < 10 || &compressed[..8] != GZ_MAGIC {
        return None;
    }
    let mut pos = 8usize;
    let original = get_varint(compressed, &mut pos).ok()? as usize;
    let rlen = get_varint(compressed, &mut pos).ok()? as usize;
    let recipe = compressed.get(pos..pos.checked_add(rlen)?)?;
    Some((original, recipe, &compressed[pos + rlen..]))
}

/// `input` compressed as an opened gzip object by `inner` when it is
/// one: the envelope, then `inner` on the plain text. `false` otherwise.
pub(crate) fn wrap(input: &[u8], output: &mut Vec<u8>, inner: impl FnOnce(&[u8], &mut Vec<u8>)) -> bool {
    if !is_gzip(input) {
        return false;
    }
    let Some(opened) = open(input) else { return false };
    // The gzip bytes themselves barely compress, so their length is
    // what the caller's level would give on them; an opened object
    // that costs more than that (a recipe of many corrections on
    // content the level does not beat gzip on) is left closed.
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
        _ => Err(crate::CodecError::CorruptedBitstream("gzip envelope: the object does not close")),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::process::{Command, Stdio};

    fn gzip(data: &[u8], args: &[&str]) -> Vec<u8> {
        let mut child = Command::new("gzip").args(args).arg("-c").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().unwrap();
        let mut stdin = child.stdin.take().unwrap();
        let data = data.to_vec();
        let t = std::thread::spawn(move || stdin.write_all(&data));
        let out = child.wait_with_output().unwrap();
        t.join().unwrap().unwrap();
        out.stdout
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

    #[test]
    fn gzip_objects_open_and_close_bit_for_bit() {
        let plain = text(3 << 20);
        for args in [&["-1"][..], &["-6"], &["-9"], &["-9", "-n"]] {
            let gz = gzip(&plain, args);
            let opened = open(&gz).expect("opens");
            assert_eq!(opened.plain, plain);
            assert_eq!(close(&opened.recipe, &opened.plain).unwrap(), gz);
            let mut c = Vec::new();
            crate::compress_into_max(&gz, &mut c);
            // Opened when that is smaller (gzip -1 here; the max level is
            // zstd -3's class and gzip -9 beats it on these short random
            // lines, so those stay closed), never larger than the gzip.
            assert_eq!(c.starts_with(GZ_MAGIC), args == ["-1"], "{args:?}: {} of {}", c.len(), gz.len());
            assert!(c.len() <= gz.len() + 256, "{args:?}: {} of {}", c.len(), gz.len());
            assert_eq!(crate::decompress(&c).unwrap(), gz);
            assert_eq!(crate::decompressed_len(&c).unwrap(), gz.len());
            let mut dst = vec![0u8; gz.len()];
            assert_eq!(crate::decompress_into(&c, &mut dst).unwrap(), gz.len());
            assert_eq!(dst, gz);
            assert_eq!(crate::decompress_parallel(&c).unwrap(), gz);
            let mut streamed = Vec::new();
            crate::decompress_stream(&c, |b| { streamed.extend_from_slice(b); Ok(()) }).unwrap();
            assert_eq!(streamed, gz);
        }
        // Two members, a trailing remainder, record mode and the cold
        // level on the content.
        let mut two = gzip(&plain[..1 << 20], &["-6"]);
        two.extend(gzip(&plain[1 << 20..], &["-3"]));
        two.extend_from_slice(b"tail bytes");
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
        assert!(d.len() < 4096, "{}", d.len());
        assert_eq!(crate::decompress_with_base(&a, &d).unwrap(), b);
        // Not gzip: untouched.
        let mut c = Vec::new();
        assert!(!wrap(b"\x1f\x8b\x08 but not really a gzip stream at all", &mut c, |_, _| {}));
        assert!(open(&plain).is_none());
    }
}
