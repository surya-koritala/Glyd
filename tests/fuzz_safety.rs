use glyd::{
    compress, compress_parallel, decompress, decompress_into_raw, decompress_parallel,
    decompress_parallel_into_raw, fallback,
};
use std::io::Write;

/// G1 requirement from docs/history/GOAL.md: 1,000,000 random mutations of compressed
/// streams, zero panics, zero out-of-bounds. Do not lower this number.
const G1_REQUIRED_MUTATIONS: u64 = 1_000_000;

#[test]
fn test_fallback_parity() {
    let input = b"The quick brown fox jumps over the lazy dog. 1234567890! Repeating pattern: ABCDEFABCDEFABCDEF";
    let mut tokens = Vec::new();
    let mut offsets = Vec::new();
    let mut extras = Vec::new();
    let mut literals = Vec::new();

    let mut table = glyd::finder::new_table();
    fallback::compress_fallback(input, &mut table, &mut tokens, &mut offsets, &mut extras, &mut literals);

    let mut decomp_buf = vec![0u8; input.len()];
    let written = fallback::decompress_fallback(
        &tokens,
        &offsets,
        &extras,
        &literals,
        &mut decomp_buf,
        input.len(),
        glyd::format::MIN_MATCH_LEN,
    )
    .unwrap();
    assert_eq!(written, input.len());
    assert_eq!(&decomp_buf[..written], input);
}

#[test]
fn test_corruption_truncation_safety() {
    let input = b"Structured payload testing memory safety under extreme stream truncations. \
                  {\"id\": 101, \"status\": \"ACTIVE\", \"tokens\": [1,2,3,4,5,6,7,8,9]}";

    let mut max = Vec::new();
    glyd::compress_into_max(input, &mut max);
    for compressed in [compress(input), compress_parallel(input), max] {
        let mut dst = vec![0u8; input.len() + 256];
        for len in 1..compressed.len() {
            let truncated = &compressed[..len];
            // Must fail cleanly on every path, never panic.
            let _ = decompress(truncated);
            let _ = decompress_parallel(truncated);
            let _ = decompress_into_raw(truncated, &mut dst);
            let _ = decompress_parallel_into_raw(truncated, &mut dst);
        }
    }
}

struct Rng(u64);

impl Rng {
    #[inline]
    fn next(&mut self) -> u64 {
        // SplitMix64
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Seed corpus chosen to exercise every encoder path: short input, RLE at
/// offset 1, extended literal runs (v2 escape tokens), high-entropy data that
/// trips the raw-block bypass, periodic small offsets on the pshufb path,
/// and a multi-block stream crossing the 64 KB and 256 KB boundaries.
fn seed_inputs() -> Vec<Vec<u8>> {
    let mut seeds: Vec<Vec<u8>> = Vec::new();
    let mut rng = Rng(0x1234_5678);

    seeds.push(b"Hello, World!".to_vec());
    seeds.push(vec![b'A'; 4096]);

    let mut noise = Vec::with_capacity(3000);
    for _ in 0..3000 {
        noise.push((rng.next() >> 33) as u8);
    }
    seeds.push(noise);

    let mut json = Vec::new();
    while json.len() < 6000 {
        json.extend_from_slice(b"{\"id\":1234,\"level\":\"INFO\",\"msg\":\"token refresh\",\"ok\":true}\n");
    }
    seeds.push(json);

    for period in [1usize, 3, 7, 15] {
        let pattern: Vec<u8> = (0..period).map(|i| b'a' + (i as u8 % 26)).collect();
        let mut buf = Vec::new();
        while buf.len() < 4000 {
            buf.extend_from_slice(&pattern);
        }
        seeds.push(buf);
    }

    let mut big = Vec::new();
    while big.len() < 80_000 {
        big.extend_from_slice(b"EventRecord(id=987654321, metric=123.456, tag='prod')\n");
    }
    seeds.push(big);

    // Many short tokens (a 44-byte match, 3 literals), so the decoders'
    // 32-token wild path runs, and its stores near the block end matter.
    let mut tokens = Vec::new();
    while tokens.len() < 20_000 {
        tokens.extend_from_slice(b"the quick brown fox jumps over the lazy dog ");
        tokens.extend_from_slice(&(rng.next() as u32).to_le_bytes()[..3]);
    }
    seeds.push(tokens);

    seeds
}

/// A v7 block of uncompressed length 0 holding one empty literal-only
/// sequence: the compressor never emits it, but the decoder must take it
/// (it is where a saturating wild-copy margin let a 32-byte store
/// through with a zero-length `dst`).
fn empty_v7_block() -> Vec<u8> {
    use glyd::format::{BlockHeader, FLAG_CHAIN_RESET, HEADER_SIZE, MAGIC, VERSION_V8};
    use glyd::v7_encode::{encode_block, Sequence, Tables};
    let mut payload = Vec::new();
    encode_block(&[Sequence { lit_len: 0, match_len: 0, offset: 0 }], &[], 0, &mut Tables::none(), &mut payload);
    let header = BlockHeader {
        magic: MAGIC,
        version: VERSION_V8,
        flags: FLAG_CHAIN_RESET,
        checksum: glyd::compute_checksum(&[]),
        uncompressed_len: 0,
        token_count: 1,
        token_bytes: payload.len() as u32,
        offset_bytes: 0,
        extras_bytes: 0,
        literal_len: 0,
    };
    let mut out = unsafe { std::slice::from_raw_parts(&header as *const BlockHeader as *const u8, HEADER_SIZE) }.to_vec();
    out.extend_from_slice(&payload);
    out
}

#[test]
fn test_corruption_mutation_fuzz_1m() {
    let seeds = seed_inputs();

    // Compress every seed both sequentially and in parallel so the fuzzer
    // covers chained streams and FLAG_CHAIN_RESET streams alike, at the
    // default level (v6 blocks) and the max level (v7 blocks, flagged).
    let mut streams: Vec<(Vec<u8>, usize)> = Vec::new();
    for s in &seeds {
        streams.push((compress(s), s.len()));
        streams.push((compress_parallel(s), s.len()));
        let (mut m, mut mp) = (Vec::new(), Vec::new());
        glyd::compress_into_max(s, &mut m);
        glyd::compress_parallel_into_max(s, &mut mp);
        streams.push((m, s.len()));
        streams.push((mp, s.len()));
    }
    let empty = empty_v7_block();
    assert_eq!(decompress(&empty).unwrap().len(), 0);
    streams.push((empty, 0));

    let mut rng = Rng(0xDEAD_BEEF_CAFE_F00D);
    let mut mutations: u64 = 0;
    let mut clean_decodes: u64 = 0;

    let max_out = seeds.iter().map(|s| s.len()).max().unwrap() + 4096;
    // Every stream decodes into an exact-size `dst` in front of a 64-byte
    // sentinel: no decoder may write past `dst`, Ok or Err.
    let mut exact = vec![0xEEu8; max_out + 64];
    let mut corrupted: Vec<u8> = Vec::with_capacity(max_out);

    while mutations < G1_REQUIRED_MUTATIONS {
        for (stream, orig_len) in streams.iter() {
            if mutations >= G1_REQUIRED_MUTATIONS {
                break;
            }
            if stream.is_empty() {
                continue;
            }

            corrupted.clear();
            corrupted.extend_from_slice(stream);

            // Apply 1..=4 mutations of a randomly chosen kind.
            let n_mut = 1 + (rng.next() % 4) as usize;
            for _ in 0..n_mut {
                let idx = (rng.next() as usize) % corrupted.len();
                match rng.next() % 4 {
                    0 => corrupted[idx] ^= 1u8 << (rng.next() % 8),
                    1 => corrupted[idx] = (rng.next() >> 24) as u8,
                    // Saturate, which drives length and count fields to extremes.
                    2 => corrupted[idx] = if rng.next() & 1 == 0 { 0x00 } else { 0xFF },
                    _ => {
                        let j = (rng.next() as usize) % corrupted.len();
                        corrupted.swap(idx, j);
                    }
                }
            }

            // Occasionally truncate on top of the mutations.
            if rng.next() % 8 == 0 && corrupted.len() > 1 {
                let new_len = 1 + (rng.next() as usize) % (corrupted.len() - 1);
                corrupted.truncate(new_len);
            }

            // Every mutation goes through the sequential paths. Neither may
            // panic, write out of bounds, or hang.
            if let Ok(out) = decompress(&corrupted) {
                assert!(
                    out.len() <= *orig_len + 4096,
                    "decompress produced {} bytes for a {}-byte original",
                    out.len(),
                    orig_len
                );
                clean_decodes += 1;
            }
            // The Rayon paths carry thread-pool overhead, so sample them.
            let parallel = mutations % 8 == 0;
            let (exact_dst, tail) = exact[..*orig_len + 64].split_at_mut(*orig_len);
            tail.fill(0xEE);
            let _ = decompress_into_raw(&corrupted, exact_dst);
            assert!(tail.iter().all(|&b| b == 0xEE), "decompress_into_raw wrote past dst");
            if parallel {
                let _ = decompress_parallel_into_raw(&corrupted, exact_dst);
                assert!(tail.iter().all(|&b| b == 0xEE), "decompress_parallel_into_raw wrote past dst");
                let _ = decompress_parallel(&corrupted);
            }

            mutations += 1;
        }
    }

    assert_eq!(mutations, G1_REQUIRED_MUTATIONS);

    // Write the G1 status marker that `bench --gates` reads. Without this file
    // the G1 gate reports FAIL instead of auto-passing.
    let marker = format!(
        concat!(
            "{{\n",
            "  \"gate\": \"G1\",\n",
            "  \"status\": \"pass\",\n",
            "  \"mutations\": {},\n",
            "  \"required\": {},\n",
            "  \"streams\": {},\n",
            "  \"clean_decodes\": {},\n",
            "  \"paths\": [\"decompress\", \"decompress_into_raw\", ",
            "\"decompress_parallel\", \"decompress_parallel_into_raw\"]\n",
            "}}\n"
        ),
        mutations,
        G1_REQUIRED_MUTATIONS,
        streams.len(),
        clean_decodes
    );

    let mut f = std::fs::File::create(".g1-status.json").expect("write G1 marker");
    f.write_all(marker.as_bytes()).expect("write G1 marker");

    println!(
        "G1 fuzz: {} mutations over {} streams, {} still decoded cleanly, 0 panics",
        mutations,
        streams.len(),
        clean_decodes
    );
}

/// A flipped byte in one unit of a multi-unit stream is an error, not a
/// hang: the units after it used to wait forever for their turn at the
/// sink (a CI round trip sat for hours on this). Every level's parallel
/// output, the streaming decoder and the whole-buffer one.
#[test]
fn corrupted_unit_fails_fast_in_the_streaming_decoder() {
    let mut x = 11u64;
    let mut data = Vec::with_capacity(24 << 20);
    while data.len() < 24 << 20 {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        data.push(b"the quick brown fox jumps over the lazy dog "[(x % 44) as usize]);
        if x % 7 == 0 { data.push((x >> 20) as u8); }
    }
    for level in [glyd::compress_parallel_into_turbo as fn(&[u8], &mut Vec<u8>), glyd::compress_parallel_into, glyd::compress_parallel_into_max] {
        let mut c = Vec::new();
        level(&data, &mut c);
        for at in [c.len() / 5, c.len() / 2, c.len() - 4000] {
            let mut bad = c.clone();
            bad[at] ^= 0x40;
            let start = std::time::Instant::now();
            let mut got = 0usize;
            let r = glyd::decompress_stream(&bad, |b| { got += b.len(); Ok(()) });
            let r2 = glyd::decompress_parallel(&bad);
            assert!(start.elapsed().as_secs() < 30, "the decoder hung on a corrupted unit");
            match (r, r2) {
                (Ok(()), Ok(d)) => assert!(got == data.len() && d == data, "corruption accepted with wrong bytes"),
                (Ok(()), Err(_)) | (Err(_), Ok(_)) => panic!("the two decoders disagree on a corrupted stream"),
                (Err(_), Err(_)) => {}
            }
        }
    }
}
