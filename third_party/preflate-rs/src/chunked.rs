/*---------------------------------------------------------------------------------------------
 *  Added in Glyd's copy of preflate-rs. Licensed under the Apache License, Version 2.0,
 *  like the rest of the crate.
 *--------------------------------------------------------------------------------------------*/

//! A deflate stream opened and re-created in pieces that do not depend on
//! each other, so they can be worked on at once.
//!
//! The stream is parsed once, in order (that is inflate, cheap). The
//! blocks are then cut into chunks of about `target` bytes of plain text,
//! each starting at a block boundary. A chunk is predicted, and later
//! re-created, by a predictor of its own that first learns the 32 KB of
//! plain text before the chunk as literals: the same on both sides, so
//! whatever it fails to predict is in the chunk's corrections. A chunk's
//! deflate bits are written from the bit offset its blocks had in the
//! stream, so the pieces join by or-ing the byte they share.

use std::io::Cursor;

use cabac::vp8::{VP8Reader, VP8Writer};

use crate::{
    PreflateConfig, Result,
    cabac_codec::{PredictionDecoderCabac, PredictionEncoderCabac},
    deflate::{deflate_reader::DeflateParser, deflate_token::DeflateTokenBlockType, deflate_writer::DeflateWriter},
    estimator::preflate_parameter_estimator::{TokenPredictorParameters, estimate_preflate_parameters},
    preflate_error::{ExitCode, PreflateError},
    preflate_input::{PlainText, PreflateInput},
    statistical_codec::PredictionEncoder,
    stream_processor::{ReconstructionData, predict_blocks, recreate_blocks},
    token_predictor::TokenPredictor,
};

pub use crate::deflate::deflate_reader::DeflateContents;
pub use crate::deflate::deflate_token::DeflateTokenBlock;

/// The window a chunk's predictor learns before its first block.
pub const PREFIX: usize = 32768;

/// A stream parsed: its blocks, its plain text, and the parameters the
/// encoder that made it seems to have used.
pub struct Parsed {
    pub contents: DeflateContents,
    pub plain: PlainText,
    pub params: TokenPredictorParameters,
    /// where each block's plain text ends
    pub block_plain_ends: Vec<usize>,
}

/// A chunk: which blocks, which plain text, and the bit its deflate
/// starts at.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Chunk {
    pub blocks: std::ops::Range<usize>,
    pub plain: std::ops::Range<usize>,
    pub bit_start: u64,
}

/// The whole of `compressed` parsed, or an error when it does not hold a
/// complete stream.
pub fn parse(compressed: &[u8], config: &PreflateConfig) -> Result<Parsed> {
    let mut parser = DeflateParser::new(config.plain_text_limit);
    let contents = parser.parse(compressed)?;
    if !parser.is_done() {
        return Err(PreflateError::new(ExitCode::ShortRead, "incomplete deflate stream"));
    }
    let plain = parser.detach_plain_text();
    let params = estimate_preflate_parameters(&contents, &plain)?;
    if params.max_chain > config.max_chain_length {
        return Err(PreflateError::new(ExitCode::NoCompressionCandidates, "max_chain above the configured limit"));
    }
    let mut block_plain_ends = Vec::with_capacity(contents.blocks.len());
    let mut at = 0usize;
    for b in &contents.blocks {
        at += match &b.block_type {
            DeflateTokenBlockType::Stored { uncompressed } => uncompressed.len(),
            DeflateTokenBlockType::Huffman { tokens, .. } => tokens
                .iter()
                .map(|t| match t {
                    crate::deflate::deflate_token::DeflateToken::Literal(_) => 1,
                    crate::deflate::deflate_token::DeflateToken::Reference(r) => r.len() as usize,
                })
                .sum(),
        };
        block_plain_ends.push(at);
    }
    Ok(Parsed { contents, plain, params, block_plain_ends })
}

/// The blocks cut into chunks of at least `target` bytes of plain text
/// each (the last may be shorter). One chunk when the stream is small.
pub fn plan(parsed: &Parsed, target: usize) -> Vec<Chunk> {
    let n = parsed.contents.blocks.len();
    let mut chunks = Vec::new();
    let (mut first, mut plain_start) = (0usize, 0usize);
    for i in 0..n {
        let end = parsed.block_plain_ends[i];
        if end - plain_start >= target || i + 1 == n {
            chunks.push(Chunk {
                blocks: first..i + 1,
                plain: plain_start..end,
                bit_start: parsed.contents.block_bit_starts[first],
            });
            first = i + 1;
            plain_start = end;
        }
    }
    chunks
}

/// The plain text a chunk's predictor sees: up to `PREFIX` bytes before
/// it as the dictionary, then the chunk.
fn text_of(plain: &[u8], chunk: &std::ops::Range<usize>) -> PlainText {
    let prefix = chunk.start.min(PREFIX);
    PlainText::with_prefix(plain[chunk.start - prefix..chunk.end].to_vec(), prefix, chunk.start)
}

fn predictor_for(params: &TokenPredictorParameters, text: &PlainText) -> TokenPredictor {
    let mut predictor = TokenPredictor::new(params);
    let mut input = PreflateInput::from_prefix_start(text);
    predictor.seed(&mut input, text.prefix().len() as u32);
    predictor
}

/// A chunk's corrections. The first chunk's carry the parameters, as a
/// whole stream's do.
pub fn predict(parsed: &Parsed, chunk: &Chunk) -> Result<Vec<u8>> {
    let text = text_of(parsed.plain.text(), &chunk.plain);
    let mut predictor = predictor_for(&parsed.params, &text);
    let mut input = PreflateInput::new(&text);
    let mut corrections = Vec::new();
    let mut encoder = PredictionEncoderCabac::new(VP8Writer::new(&mut corrections).unwrap());
    predict_blocks(&parsed.contents.blocks[chunk.blocks.clone()], &mut predictor, &mut encoder, &mut input)?;
    encoder.finish();
    if chunk.blocks.start == 0 {
        return Ok(ReconstructionData { parameters: parsed.params, corrections }.encode());
    }
    Ok(corrections)
}

/// The parameters from the first chunk's corrections.
pub fn parameters(first_corrections: &[u8]) -> Result<TokenPredictorParameters> {
    Ok(ReconstructionData::read(first_corrections)?.parameters)
}

/// A chunk's deflate bits back, from the plain text (`chunk_plain` of
/// `plain`), its corrections (`first`: they carry the parameters) and
/// the bit offset within a byte its blocks started at. The output's
/// first byte has `bit_start % 8` zero bits below the stream's, and the
/// count returned is how many bits of the last byte are the stream's
/// (0: it ends on a byte boundary).
pub fn recreate(params: &TokenPredictorParameters, plain: &[u8], chunk_plain: std::ops::Range<usize>, corrections: &[u8], first: bool, bit_start: u32) -> Result<(Vec<u8>, u32)> {
    let text = text_of(plain, &chunk_plain);
    let mut predictor = predictor_for(params, &text);
    let mut input = PreflateInput::new(&text);
    let read;
    let corrections = if first {
        read = ReconstructionData::read(corrections)?;
        &read.corrections[..]
    } else {
        corrections
    };
    let mut decoder = PredictionDecoderCabac::new(VP8Reader::new(Cursor::new(corrections)).unwrap());
    let mut writer = DeflateWriter::new_at_bit(bit_start % 8);
    recreate_blocks(&mut predictor, &mut decoder, &mut writer, &mut input)?;
    if input.remaining() != 0 {
        return Err(PreflateError::new(ExitCode::RoundtripMismatch, "chunk did not use all its plain text"));
    }
    Ok(writer.finish())
}

/// The pieces `recreate` made, in order, joined into the stream.
pub fn join(pieces: &[(Vec<u8>, u32)]) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::with_capacity(pieces.iter().map(|p| p.0.len()).sum());
    let mut partial = 0u32;
    for (bytes, bits) in pieces {
        if partial > 0 && !bytes.is_empty() {
            *out.last_mut().unwrap() |= bytes[0];
            out.extend_from_slice(&bytes[1..]);
        } else {
            out.extend_from_slice(bytes);
        }
        partial = *bits;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(n: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(n);
        let mut x = 88172645463325252u64;
        while v.len() < n {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let w = (x % 4096) as usize;
            v.extend_from_slice(format!("word{w} line {} value {}\n", x % 1000, x % 7).as_bytes());
        }
        v.truncate(n);
        v
    }

    fn round_trip(compressed: &[u8], target: usize) -> (usize, usize) {
        let parsed = parse(compressed, &PreflateConfig::default()).unwrap();
        let chunks = plan(&parsed, target);
        let corrections: Vec<Vec<u8>> = chunks.iter().map(|c| predict(&parsed, c).unwrap()).collect();
        let params = parameters(&corrections[0]).unwrap();
        let plain = parsed.plain.text();
        let pieces: Vec<(Vec<u8>, u32)> = chunks
            .iter()
            .zip(&corrections)
            .map(|(c, corr)| recreate(&params, plain, c.plain.clone(), corr, c.blocks.start == 0, c.bit_start as u32).unwrap())
            .collect();
        assert_eq!(join(&pieces), compressed);
        (chunks.len(), corrections.iter().map(|c| c.len()).sum())
    }

    #[test]
    fn pieces_join_bit_for_bit() {
        let plain = text(3 << 20);
        for level in [1, 6, 9] {
            let compressed = miniz_oxide::deflate::compress_to_vec(&plain, level);
            let (n, one) = round_trip(&compressed, usize::MAX);
            assert_eq!(n, 1);
            let (n, many) = round_trip(&compressed, 256 << 10);
            assert!(n > 4, "{n} chunks");
            assert!(many < one + n * 400, "level {level}: {many} against {one} in {n} chunks");
        }
    }

    #[test]
    fn stored_blocks_keep_their_alignment() {
        let mut noise = Vec::with_capacity(1 << 20);
        let mut x = 7u64;
        while noise.len() < 1 << 20 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            noise.extend_from_slice(&x.to_le_bytes());
        }
        let mut mixed = text(300 << 10);
        mixed.extend_from_slice(&noise);
        mixed.extend_from_slice(&text(300 << 10));
        let compressed = miniz_oxide::deflate::compress_to_vec(&mixed, 6);
        assert!(round_trip(&compressed, 128 << 10).0 > 2);
    }
}
