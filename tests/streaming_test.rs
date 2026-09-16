use std::io::{Read, Write};
use simd_stream_codec::{AlatirokReader, AlatirokWriter};

#[test]
fn test_streaming_roundtrip_large() {
    let mut original_data = Vec::with_capacity(300_000);
    let sample = b"Streaming compression chunk test for Alatirok engine with cloud & AI workflows. ";
    while original_data.len() < 300_000 {
        original_data.extend_from_slice(sample);
    }

    // Compress via AlatirokWriter
    let mut compressed = Vec::new();
    {
        let mut writer = AlatirokWriter::new(&mut compressed);
        // Write in random-sized chunks to test arbitrary slice boundaries
        let mut offset = 0;
        let chunk_sizes = [13, 1024, 77, 65536, 12, 4096, 33];
        let mut idx = 0;
        while offset < original_data.len() {
            let sz = chunk_sizes[idx % chunk_sizes.len()].min(original_data.len() - offset);
            writer.write_all(&original_data[offset..offset + sz]).unwrap();
            offset += sz;
            idx += 1;
        }
        writer.flush().unwrap();
    }

    assert!(!compressed.is_empty());
    assert!(compressed.len() < original_data.len() / 2);

    // Decompress via AlatirokReader
    let mut reader = AlatirokReader::new(&compressed[..]);
    let mut decompressed = Vec::new();
    let mut buf = [0u8; 512];
    loop {
        let n = reader.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        decompressed.extend_from_slice(&buf[..n]);
    }

    assert_eq!(decompressed.len(), original_data.len());
    assert_eq!(decompressed, original_data);
}
