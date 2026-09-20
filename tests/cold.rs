//! The cold level: context mixing over 32 MB units. Round trips through
//! every decoder, in record mode too; corrupted streams are refused by
//! the unit checksums; empty and tiny inputs.

fn rnd(x: &mut u64) -> u64 {
    *x ^= *x << 13;
    *x ^= *x >> 7;
    *x ^= *x << 17;
    *x
}

/// Log-like lines: a timestamp, a host from a small set, a path, a size.
fn logs(n: usize, seed: u64) -> Vec<u8> {
    let mut x = seed;
    let mut v = Vec::with_capacity(n + 128);
    let mut t = 1_700_000_000u64;
    while v.len() < n {
        let r = rnd(&mut x);
        t += r % 7;
        v.extend_from_slice(format!("host{} - - [{}] \"GET /item/{}/page{}.html HTTP/1.0\" {} {}\n", r % 40, t, (r >> 8) % 5000, (r >> 20) % 9, if r % 11 == 0 { 404 } else { 200 }, (r >> 30) % 100_000).as_bytes());
    }
    v.truncate(n);
    v
}

#[test]
fn round_trips_every_way() {
    let data = logs(33 << 20, 1); // two units
    let mut c = Vec::new();
    glyd::compress_parallel_into_cold(&data, &mut c);
    assert!(c.len() * 8 < data.len(), "{} of {}", c.len(), data.len());
    assert_eq!(glyd::decompressed_len(&c).unwrap(), data.len());
    assert!(glyd::decompress_parallel(&c).unwrap() == data);
    let mut streamed = Vec::new();
    glyd::decompress_stream(&c, |b| { streamed.extend_from_slice(b); Ok(()) }).unwrap();
    assert!(streamed == data);
    // The sequential encoder writes the same bytes; the sequential
    // decoders read them; record mode over the cold level pays on logs.
    let small = &data[..3 << 20];
    let (mut a, mut b) = (Vec::new(), Vec::new());
    glyd::compress_into_cold(small, &mut a);
    glyd::compress_parallel_into_cold(small, &mut b);
    assert!(a == b);
    assert!(glyd::decompress(&a).unwrap() == small);
    let mut dst = vec![0u8; small.len()];
    assert_eq!(glyd::decompress_into(&a, &mut dst).unwrap(), small.len());
    assert!(dst == small);
    let mut r = Vec::new();
    glyd::compress_records_into_cold(small, &mut r);
    assert!(r.len() < a.len(), "columns should pay on logs: {} vs {}", r.len(), a.len());
    assert!(glyd::decompress(&r).unwrap() == small);
}

#[test]
fn edges_and_corruption() {
    for input in [&[][..], b"x", b"abc", &logs(1000, 2)[..]] {
        let mut c = Vec::new();
        glyd::compress_into_cold(input, &mut c);
        assert!(glyd::decompress(&c).unwrap() == input, "round trip of {} bytes", input.len());
    }
    let data = logs(200_000, 3);
    let mut c = Vec::new();
    glyd::compress_into_cold(&data, &mut c);
    let mut x = 4u64;
    let mut refused = 0;
    for _ in 0..40 {
        let mut bad = c.clone();
        let i = 16 + (rnd(&mut x) as usize) % (bad.len() - 16);
        bad[i] ^= 1 << (rnd(&mut x) % 8);
        match glyd::decompress(&bad) {
            Ok(d) => assert!(d == data, "a flipped bit decoded to different bytes unnoticed"),
            Err(_) => refused += 1,
        }
    }
    assert!(refused >= 38, "corrupted streams must be refused: {refused} of 40");
    let mut short = c.clone();
    short.truncate(c.len() - 10);
    assert!(glyd::decompress(&short).is_err());
}
