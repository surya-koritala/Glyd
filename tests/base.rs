//! Base (delta) mode: a new version compressed against the old one comes
//! back byte for byte with that base, costs a small fraction of its
//! plain size, is refused with the wrong base, and survives the shapes
//! versions take (insertions, deletions, drift, a base shorter or longer
//! than the input, tiny and empty inputs, corrupted streams).

fn rnd(x: &mut u64) -> u64 {
    *x ^= *x << 13;
    *x ^= *x >> 7;
    *x ^= *x << 17;
    *x
}

/// Text-like data: words from a vocabulary, some noise, `n` bytes.
fn wordy(n: usize, seed: u64) -> Vec<u8> {
    let words: Vec<Vec<u8>> = (0..500).map(|i| format!("w{}{} ", i, "abcdefghij".repeat(i % 5)).into_bytes()).collect();
    let mut x = seed;
    let mut v = Vec::with_capacity(n + 64);
    while v.len() < n {
        let r = rnd(&mut x);
        if r % 23 == 0 {
            v.push((r >> 8) as u8);
        } else {
            v.extend_from_slice(&words[(r >> 8) as usize % words.len()]);
        }
    }
    v.truncate(n);
    v
}

/// `old` with `edits` random insertions, deletions and replacements.
fn edited(old: &[u8], edits: usize, seed: u64) -> Vec<u8> {
    let mut x = seed;
    let mut v = old.to_vec();
    for _ in 0..edits {
        let at = (rnd(&mut x) as usize) % v.len().max(1);
        let len = (rnd(&mut x) as usize) % 200;
        match rnd(&mut x) % 3 {
            0 => {
                let ins = wordy(len, rnd(&mut x));
                v.splice(at..at, ins);
            }
            1 => {
                let end = (at + len).min(v.len());
                v.drain(at..end);
            }
            _ => {
                let end = (at + len).min(v.len());
                let rep = wordy(end - at, rnd(&mut x));
                v.splice(at..end, rep);
            }
        }
    }
    v
}

fn check(old: &[u8], new: &[u8], name: &str) -> usize {
    let mut c = Vec::new();
    glyd::compress_with_base(old, new, &mut c, false);
    assert!(glyd::needs_base(&c), "{name}: base envelope");
    assert!(glyd::decompress_with_base(old, &c).unwrap() == new, "{name}: round trip");
    assert_eq!(glyd::decompressed_len(&c).unwrap(), new.len(), "{name}: length");
    assert!(glyd::decompress(&c).is_err(), "{name}: decoding without the base must fail");
    c.len()
}

#[test]
fn edited_versions_round_trip_and_shrink() {
    let old = wordy(40 << 20, 1);
    let new = edited(&old, 300, 2);
    let with_base = check(&old, &new, "40 MB, 300 edits");
    let mut alone = Vec::new();
    glyd::compress_parallel_into_max(&new, &mut alone);
    assert!(with_base * 10 < alone.len(), "the delta should be under a tenth of the plain size: {with_base} vs {}", alone.len());
    // The ultra level too, on a smaller pair.
    let old = wordy(3 << 20, 3);
    let new = edited(&old, 40, 4);
    let mut c = Vec::new();
    glyd::compress_with_base(&old, &new, &mut c, true);
    assert!(glyd::decompress_with_base(&old, &c).unwrap() == new, "ultra round trip");
}

#[test]
fn drift_and_size_changes() {
    let old = wordy(50 << 20, 5);
    // Drift: 20 MB inserted at the front (within the slack), then the rest.
    let mut new = wordy(20 << 20, 6);
    new.extend_from_slice(&old[..30 << 20]);
    check(&old, &new, "20 MB inserted at the front");
    // The input longer than the base and reaching past its end.
    let mut longer = old.clone();
    longer.extend_from_slice(&wordy(10 << 20, 7));
    check(&old, &longer, "longer than the base");
    // The input much shorter than the base; the base much shorter than the input.
    check(&old, &old[10 << 20..12 << 20], "a slice of the base");
    check(&old[..100], &old[..5 << 20], "a 100-byte base");
    // Content moved beyond the slack: still exact, just not matched.
    let mut moved = wordy(70 << 20, 8);
    moved.extend_from_slice(&old[..4 << 20]);
    check(&old, &moved, "moved past the slack");
}

#[test]
fn edge_sizes_and_wrong_base() {
    let old = wordy(1 << 20, 9);
    check(&old, &[], "empty input");
    check(&old, b"x", "one byte");
    check(&[], &old[..1000], "empty base");
    check(&[], &[], "both empty");
    let new = edited(&old, 10, 10);
    let mut c = Vec::new();
    glyd::compress_with_base(&old, &new, &mut c, false);
    let other = wordy(1 << 20, 11);
    assert!(glyd::decompress_with_base(&other, &c).is_err(), "another base must be refused");
    let mut shorter = old.clone();
    shorter.pop();
    assert!(glyd::decompress_with_base(&shorter, &c).is_err(), "a truncated base must be refused");
    // Corrupted streams: rejected or, rarely, decoded to the wrong length; never a panic.
    let mut x = 12u64;
    for _ in 0..300 {
        let mut bad = c.clone();
        let i = (rnd(&mut x) as usize) % bad.len();
        bad[i] ^= 1 << (rnd(&mut x) % 8);
        if let Ok(d) = glyd::decompress_with_base(&old, &bad) {
            let _ = d.len();
        }
    }
}
