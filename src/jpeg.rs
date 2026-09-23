//! JPEG objects recoded losslessly: the DCT coefficients taken out
//! and coded with a model of neighbouring blocks (`crate::jpg`), about
//! a fifth to a quarter smaller, and the identical JPEG put back. The
//! envelope:
//!
//!   "GLYDJPEG" original_len (varint), the stream.
//!
//! The stream is Glyd's own ("GJPG", `jpg::pack`) since v0.14.0;
//! earlier releases wrote Lepton's (`lepton_jpeg`, 0xCF 0x84), which
//! the `jpeg` feature keeps reading.
//!
//! A JPEG the parser cannot take (progressive, arithmetic-coded,
//! 12-bit, DNL) or that does not shrink is left as it is.

use crate::record::{get_varint, put_varint};

pub(crate) const MAGIC: &[u8; 8] = b"GLYDJPEG";

pub fn is_jpeg(input: &[u8]) -> bool {
    input.len() >= 4 && input[0] == 0xff && input[1] == 0xd8 && input[2] == 0xff
}

/// The stream of a JPEG, verified to give the JPEG back.
pub fn transcode(input: &[u8]) -> Option<Vec<u8>> {
    if !is_jpeg(input) {
        return None;
    }
    crate::jpg::pack(input)
}

/// The JPEG back from its stream, ours or Lepton's.
pub fn restore(stream: &[u8]) -> Option<Vec<u8>> {
    if stream.starts_with(crate::jpg::MAGIC) {
        return crate::jpg::unpack(stream);
    }
    #[cfg(feature = "jpeg")]
    {
        use lepton_jpeg::{decode_lepton, EnabledFeatures, SingleThreadPool};
        let mut out = Vec::with_capacity(stream.len() * 5 / 4 + 1024);
        decode_lepton(&mut std::io::Cursor::new(stream), &mut out, &EnabledFeatures::compat_lepton_vector_read(), &SingleThreadPool {}).ok()?;
        return Some(out);
    }
    #[cfg(not(feature = "jpeg"))]
    None
}

/// `input` as an envelope when it is a JPEG that shrinks so.
pub(crate) fn wrap(input: &[u8], output: &mut Vec<u8>) -> bool {
    let Some(stream) = transcode(input) else { return false };
    if stream.len() + 16 >= input.len() {
        return false;
    }
    output.extend_from_slice(MAGIC);
    put_varint(output, input.len() as u64);
    output.extend_from_slice(&stream);
    true
}

/// (original length, stream) of an envelope.
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
    let (original, stream) = parse(compressed)?;
    Some(match restore(stream) {
        Some(out) if out.len() == original => Ok(out),
        _ => Err(crate::CodecError::CorruptedBitstream("jpeg envelope: the object does not restore")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_jpeg_transcodes_and_restores_bit_for_bit() {
        let jpeg = crate::jpg::tests::fixture("q75-420.jpg");
        let stream = transcode(&jpeg).expect("transcodes");
        assert!(stream.starts_with(crate::jpg::MAGIC));
        assert!(stream.len() < jpeg.len() * 9 / 10, "{} of {}", stream.len(), jpeg.len());
        assert_eq!(restore(&stream).unwrap(), jpeg);
        let mut out = Vec::new();
        assert!(wrap(&jpeg, &mut out));
        assert_eq!(unwrap(&out).unwrap().unwrap(), jpeg);
        assert!(unwrap(b"GLYDJPEGxx").is_some());
    }

    #[cfg(feature = "jpeg")]
    #[test]
    fn a_lepton_stream_still_restores() {
        let jpeg = crate::jpg::tests::fixture("q75-420.jpg");
        let (lepton, _) = lepton_jpeg::encode_lepton_verify(&jpeg, &lepton_jpeg::EnabledFeatures::compat_lepton_vector_write(), &lepton_jpeg::SingleThreadPool {}).unwrap();
        assert_eq!(restore(&lepton).unwrap(), jpeg);
    }
}
