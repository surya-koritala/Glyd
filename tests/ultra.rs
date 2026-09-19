//! The ultra level: round trips through the shapes its parse takes
//! (empty, tiny, runs, cross-block matches, the window's edge, raw
//! blocks, dictionaries), determinism, and the parallel path.
use glyd::format::MAX_BLOCK_SIZE;
use glyd::v7_format::LOCAL_WINDOW;

fn rnd(x: &mut u64) -> u64 {
    *x ^= *x << 13;
    *x ^= *x >> 7;
    *x ^= *x << 17;
    *x
}

/// Text-like data: words from a small vocabulary, some noise.
fn wordy(n: usize, seed: u64) -> Vec<u8> {
    let words: Vec<Vec<u8>> = (0..300).map(|i| format!("w{}{} ", i, "abcdefghij".repeat(i % 4)).into_bytes()).collect();
    let mut x = seed;
    let mut v = Vec::with_capacity(n + 64);
    while v.len() < n {
        let r = rnd(&mut x);
        if r % 17 == 0 {
            v.push((r >> 8) as u8);
        } else {
            v.extend_from_slice(&words[(r >> 8) as usize % words.len()]);
        }
    }
    v.truncate(n);
    v
}

fn roundtrip(name: &str, data: &[u8]) -> Vec<u8> {
    let mut c = Vec::new();
    glyd::compress_into_ultra(data, &mut c);
    let d = glyd::decompress(&c).unwrap_or_else(|e| panic!("{name}: decode failed: {e:?}"));
    assert!(d == data, "{name}: round trip mismatch");
    let mut c2 = Vec::new();
    glyd::compress_into_ultra(data, &mut c2);
    assert!(c == c2, "{name}: two compressions differ");
    c
}

#[test]
fn ultra_roundtrip_shapes() {
    roundtrip("empty", &[]);
    roundtrip("one byte", &[42]);
    roundtrip("three bytes", b"abc");
    roundtrip("four equal", &[7; 4]);
    roundtrip("short text", b"the quick brown fox jumps over the lazy dog, the quick brown fox");
    // Runs: every match is a rep at offset 1, the sufficient-length path.
    let c = roundtrip("zeros 1 MB", &vec![0u8; 1 << 20]);
    assert!(c.len() < 2048, "zeros 1 MB: {} bytes", c.len());
    // A period longer than the sufficient length, across blocks.
    let pat: Vec<u8> = (0..1000u32).map(|i| (i * 7 % 251) as u8).collect();
    let mut rep = Vec::new();
    while rep.len() < 3 * MAX_BLOCK_SIZE + 123 {
        rep.extend_from_slice(&pat);
    }
    let c = roundtrip("periodic across blocks", &rep);
    assert!(c.len() < rep.len() / 50, "periodic: {} bytes", c.len());
    // Incompressible: stored raw.
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    let noise: Vec<u8> = (0..MAX_BLOCK_SIZE + 77).map(|_| rnd(&mut x) as u8).collect();
    let c = roundtrip("noise", &noise);
    assert!(c.len() <= noise.len() + 256);
    // Text over several blocks: matches into earlier blocks.
    let text = wordy(5 * MAX_BLOCK_SIZE + 4321, 1);
    let c = roundtrip("wordy 1.3 MB", &text);
    let mut m = Vec::new();
    glyd::compress_into_max(&text, &mut m);
    assert!(c.len() < m.len(), "ultra ({}) should beat max ({}) on text", c.len(), m.len());
}

/// Matches at exactly the local window's edge and just past it: the
/// tree finder offers the first; the second is the long-distance
/// matcher's (its offset needs a far code) and is matched too.
#[test]
fn ultra_window_edge() {
    let w = LOCAL_WINDOW as usize;
    let mut x = 12345u64;
    let block: Vec<u8> = (0..4096).map(|_| rnd(&mut x) as u8).collect();
    for gap in [w - 4096 - 1, w - 4096, w - 4096 + 1, w + 100] {
        let mut data = block.clone();
        let mut y = 777u64 + gap as u64;
        data.extend((0..gap).map(|_| rnd(&mut y) as u8));
        data.extend_from_slice(&block);
        let c = roundtrip(&format!("window gap {gap}"), &data);
        // The repeat straddles a block boundary; the part in the last
        // block (at least 3 KB of the 4 KB) must be matched, the block
        // headers of the raw noise cost ~1 KB.
        assert!(c.len() < data.len() - 2000, "gap {gap}: the repeat should match ({} bytes)", c.len());
    }
}

#[test]
fn ultra_parallel_and_dictionary() {
    let text = wordy(9 * MAX_BLOCK_SIZE + 5, 3);
    let mut p = Vec::new();
    glyd::compress_parallel_into_ultra(&text, &mut p);
    assert_eq!(glyd::decompress(&p).unwrap(), text);

    let dict = glyd::Dict::from_content(&wordy(64 * 1024, 9), &[]);
    let small = wordy(3000, 11);
    let (mut with, mut without) = (Vec::new(), Vec::new());
    glyd::compress_with_dict_ultra(&dict, &small, &mut with);
    glyd::compress_into_ultra(&small, &mut without);
    assert_eq!(glyd::decompress_with_dict(&dict, &with).unwrap(), small);
    assert!(with.len() < without.len(), "dictionary should help: {} vs {}", with.len(), without.len());
}

/// Random structured inputs (runs, repeats at random distances, noise),
/// many of them, in debug builds with the tree's order assertion on.
#[test]
fn ultra_fuzz_roundtrip() {
    let iters: usize = std::env::var("ULTRA_FUZZ").ok().and_then(|v| v.parse().ok()).unwrap_or(60);
    let mut x = 0xD1B5_4A32_D192_ED03u64;
    for i in 0..iters {
        let n = (rnd(&mut x) % 200_000) as usize;
        let mut v = Vec::with_capacity(n);
        while v.len() < n {
            match rnd(&mut x) % 4 {
                0 => {
                    let len = (rnd(&mut x) % 300) as usize;
                    let b = rnd(&mut x) as u8;
                    v.extend(std::iter::repeat(b).take(len));
                }
                1 if !v.is_empty() => {
                    let dist = 1 + (rnd(&mut x) as usize % v.len().min(70_000));
                    let len = 3 + (rnd(&mut x) % 400) as usize;
                    for _ in 0..len {
                        let b = v[v.len() - dist];
                        v.push(b);
                    }
                }
                _ => {
                    let len = (rnd(&mut x) % 64) as usize;
                    v.extend((0..len).map(|_| rnd(&mut x) as u8));
                }
            }
        }
        v.truncate(n);
        roundtrip(&format!("fuzz {i}"), &v);
    }
}

/// Repeats farther back than the local window, at both levels: a 1 MB
/// text repeated 9 MB later, twice, in 24 MB. Only a far match can
/// store a copy in a few KB; the max level's gate must keep its pass
/// on (the text repeats), and every byte comes back.
#[test]
fn far_repeats_round_trip_at_max_and_ultra() {
    let block = wordy(1 << 20, 5);
    let mut data = Vec::new();
    for gap in [21u64, 22] {
        data.extend_from_slice(&block);
        data.extend_from_slice(&wordy(9 << 20, gap));
    }
    data.extend_from_slice(&block);
    let mut alone = Vec::new();
    glyd::compress_into_max(&data[..data.len() - (1 << 20)], &mut alone);
    for (name, f) in [("max", glyd::compress_into_max as fn(&[u8], &mut Vec<u8>)), ("ultra", glyd::compress_into_ultra)] {
        let mut c = Vec::new();
        f(&data, &mut c);
        assert!(glyd::decompress(&c).unwrap() == data, "{name}: round trip mismatch");
        let mut c2 = Vec::new();
        f(&data, &mut c2);
        assert!(c == c2, "{name}: two compressions differ");
        if name == "max" {
            assert!(c.len() < alone.len() + 20_000, "max: the far copy should cost under 20 KB, not {}", c.len() - alone.len());
        }
    }
    // Noise past the window: the pass finds nothing and stops early;
    // the bytes still come back.
    let mut x = 99u64;
    let noise: Vec<u8> = (0..(9 << 20)).map(|_| rnd(&mut x) as u8).collect();
    let mut c = Vec::new();
    glyd::compress_into_max(&noise, &mut c);
    assert!(glyd::decompress(&c).unwrap() == noise);
}
