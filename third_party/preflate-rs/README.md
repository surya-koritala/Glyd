# preflate

Core DEFLATE analysis and reconstruction engine for the preflate-rs workspace.

This crate takes an existing DEFLATE-compressed bitstream and splits it into two parts:
- The **uncompressed plaintext**
- A compact **corrections blob** that captures everything needed to recreate the original bitstream exactly

Given those two parts, the original DEFLATE stream can be reconstructed bit-for-bit. This enables re-compressing the plaintext with a modern algorithm (Zstd, Brotli, LZMA) while preserving binary-exact fidelity.

[![unsafe forbidden](https://img.shields.io/badge/unsafe-forbidden-success.svg)](https://github.com/rust-secure-code/safety-dance/)

## How It Works

### Analysis pipeline

1. **Parse** (`deflate/`) — Decode the DEFLATE bitstream into a token sequence: literals and length/distance back-references.
2. **Estimate** (`estimator/`) — Analyze the token sequence to fingerprint the original compressor. Identifies hash algorithm, chain depth, nice-length cutoff, window size, and add policy.
3. **Predict** (`token_predictor.rs`) — Re-run compression using the estimated parameters, generating a predicted token sequence.
4. **Encode** (`statistical_codec.rs`, `cabac_codec.rs`) — Record each position where the actual token differs from the prediction, using CABAC arithmetic coding.

### Reconstruction pipeline

Feed the plaintext and corrections back into the predictor. It replays every compression decision, applying corrections where recorded, and writes the resulting tokens back to a DEFLATE bitstream using the original Huffman trees.

## API

### Whole-stream (simple)

```rust
use preflate_rs::{preflate_whole_deflate_stream, recreate_whole_deflate_stream, PreflateConfig};

let config = PreflateConfig::default();

// Analysis: DEFLATE → plaintext + corrections
let (result, plain_text) = preflate_whole_deflate_stream(&compressed_data, &config)?;

// Reconstruction: plaintext + corrections → original DEFLATE
let recreated = recreate_whole_deflate_stream(plain_text.text(), &result.corrections)?;

assert_eq!(compressed_data, recreated);
```

### Streaming (chunked)

For large streams where memory is a concern, use the streaming processors:

```rust
use preflate_rs::{PreflateStreamProcessor, RecreateStreamProcessor, PreflateConfig};

// Analysis
let mut processor = PreflateStreamProcessor::new(&config);
let chunk_result = processor.decompress(&compressed_chunk)?;
// chunk_result.corrections contains the encoded corrections for this chunk
// processor.plain_text() gives access to the decompressed data

// Reconstruction
let mut recreator = RecreateStreamProcessor::new();
let (deflate_output, _blocks) = recreator.recompress(&mut plain_text_reader, &corrections)?;
```

## Configuration

```rust
pub struct PreflateConfig {
    /// Maximum hash chain traversal depth. Higher values improve prediction
    /// accuracy for streams compressed with high chain limits, at the cost
    /// of analysis time. Default: 4096.
    pub max_chain_length: u32,

    /// Maximum plaintext held in memory at once. Default: 128 MB.
    pub plain_text_limit: usize,

    /// Verify round-trip correctness after analysis. Default: true.
    pub verify_compression: bool,
}
```

## Supported Compressors

The estimator detects the following DEFLATE implementations by their token patterns and hash algorithms:

| Compressor | Detection method |
|---|---|
| zlib | Default hash, standard chain behavior |
| zlib-ng | Distinct hash variant |
| libdeflate | 4-byte hash tables |
| miniz / miniz_oxide | Fastest-mode hash function |
| Windows zlib | Built-in PNG/ZIP codec fingerprint |

Unknown compressors still round-trip correctly with higher corrections overhead.

## Key Types

| Type | Description |
|---|---|
| `PreflateStreamChunkResult` | Output of analysis: corrections blob, compressed size, estimated parameters |
| `TokenPredictorParameters` | Compressor fingerprint: hash algorithm, nice_length, max_chain, window_bits, add policy |
| `HashAlgorithm` | Enum of detected compressor families |
| `PlainText` | Decompressed data with sliding-window dictionary support |
| `PreflateError` | Rich error type with detailed exit codes |

## Encoding Format

Parameters are serialized with [`bitcode`](https://crates.io/crates/bitcode). Corrections use CABAC (Context Adaptive Binary Arithmetic Coding), the same codec used in [Lepton JPEG](https://github.com/microsoft/lepton_jpeg_rust) compression. The format is chunked so memory use is bounded regardless of input size.

The following differences from the predicted stream are encoded:

- Block type (uncompressed, static Huffman, dynamic Huffman)
- Token count per block
- Dynamic Huffman tree encoding
- Literal vs. length/distance choice
- Incorrect distance or length (encoded as hop count back through the hash chain)

## Constraints

- **No unsafe code** — `#![forbid(unsafe_code)]`
- Minimum Rust version: **1.89**, Edition **2024**
