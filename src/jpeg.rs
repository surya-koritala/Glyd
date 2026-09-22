//! JPEG objects transcoded losslessly: Lepton (`lepton_jpeg`, the
//! Rust port of Dropbox's) recodes a JPEG's DCT coefficients with an
//! arithmetic coder and a predictor across blocks, about a fifth
//! smaller, and gives the identical JPEG back. Nothing of ours does
//! better on such bytes, so the envelope holds the Lepton stream as it
//! is:
//!
//!   "GLYDJPEG" original_len (varint), the Lepton stream.
//!
//! A JPEG Lepton cannot take (progressive layouts it rejects, more
//! than 16,384 pixels a side) or does not shrink is left as it is.

use crate::record::{get_varint, put_varint};
use lepton_jpeg::{decode_lepton, encode_lepton_verify, EnabledFeatures, SingleThreadPool};

pub(crate) const MAGIC: &[u8; 8] = b"GLYDJPEG";

pub fn is_jpeg(input: &[u8]) -> bool {
    input.len() >= 4 && input[0] == 0xff && input[1] == 0xd8 && input[2] == 0xff
}

/// The Lepton stream of a JPEG, verified to give the JPEG back.
pub fn transcode(input: &[u8]) -> Option<Vec<u8>> {
    if !is_jpeg(input) {
        return None;
    }
    let (out, _) = encode_lepton_verify(input, &EnabledFeatures::compat_lepton_vector_write(), &SingleThreadPool {}).ok()?;
    Some(out)
}

/// The JPEG back from its Lepton stream.
pub fn restore(lepton: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(lepton.len() * 5 / 4 + 1024);
    decode_lepton(&mut std::io::Cursor::new(lepton), &mut out, &EnabledFeatures::compat_lepton_vector_read(), &SingleThreadPool {}).ok()?;
    Some(out)
}

/// `input` as a Lepton envelope when it is a JPEG that shrinks so.
pub(crate) fn wrap(input: &[u8], output: &mut Vec<u8>) -> bool {
    let Some(lepton) = transcode(input) else { return false };
    if lepton.len() + 16 >= input.len() {
        return false;
    }
    output.extend_from_slice(MAGIC);
    put_varint(output, input.len() as u64);
    output.extend_from_slice(&lepton);
    true
}

/// (original length, Lepton stream) of an envelope.
pub(crate) fn parse(compressed: &[u8]) -> Option<(usize, &[u8])> {
    if compressed.len() < 10 || &compressed[..8] != MAGIC {
        return None;
    }
    let mut pos = 8usize;
    let original = get_varint(compressed, &mut pos).ok()? as usize;
    Some((original, &compressed[pos..]))
}

/// The JPEG of an envelope; `None` when `compressed` is no envelope.
pub(crate) fn unwrap(compressed: &[u8]) -> Option<crate::Result<Vec<u8>>> {
    let (original, lepton) = parse(compressed)?;
    Some(match restore(lepton) {
        Some(out) if out.len() == original => Ok(out),
        _ => Err(crate::CodecError::CorruptedBitstream("jpeg envelope: the object does not restore")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_jpeg_transcodes_and_restores_bit_for_bit() {
        let jpeg = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/tiny.jpg")).unwrap();
        assert!(is_jpeg(&jpeg));
        let lepton = transcode(&jpeg).expect("transcodes");
        assert_eq!(restore(&lepton).unwrap(), jpeg);
        let mut c = Vec::new();
        crate::compress_into_max(&jpeg, &mut c);
        assert_eq!(crate::decompress(&c).unwrap(), jpeg);
        assert_eq!(crate::decompressed_len(&c).unwrap(), jpeg.len());
        assert_eq!(crate::decompress_parallel(&c).unwrap(), jpeg);
        let mut c = Vec::new();
        crate::compress_into_cold(&jpeg, &mut c);
        assert_eq!(crate::decompress(&c).unwrap(), jpeg);
        // Not a JPEG: untouched.
        let mut c = Vec::new();
        assert!(!wrap(b"\xff\xd8\xff but not a jpeg", &mut c));
    }
}
