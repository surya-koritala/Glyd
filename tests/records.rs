//! Record mode through the public API: record-shaped input (delimited
//! lines, JSON lines) takes the envelope and comes back byte for byte,
//! smaller than the plain level; input the transform does not pay on
//! (prose) takes the plain parallel path, with no envelope, and comes
//! back too.

fn rnd(x: &mut u64) -> u64 {
    *x ^= *x << 13;
    *x ^= *x >> 7;
    *x ^= *x << 17;
    *x
}

/// Space-delimited records with integer, few-valued and time columns.
fn delimited(lines: usize) -> Vec<u8> {
    let mut x = 9u64;
    let mut v = Vec::new();
    for i in 0..lines {
        let r = rnd(&mut x);
        v.extend_from_slice(format!("{} host{} 2024-01-15T10:{:02}:{:02} GET /path/{} 200 {}\n", 1_700_000_000 + i as u64 * 3, r % 40, (i / 60) % 60, i % 60, r % 3000, r % 100_000).as_bytes());
    }
    v
}

/// One JSON object per line with typed fields.
fn json_lines(lines: usize) -> Vec<u8> {
    let mut x = 77u64;
    let mut v = Vec::new();
    for i in 0..lines {
        let r = rnd(&mut x);
        v.extend_from_slice(format!("{{\"id\":{},\"type\":\"{}Event\",\"actor\":{{\"login\":\"user{}\",\"id\":{}}},\"payload\":{{\"n\":{}}}}}\n", i, ["Push", "Watch", "Issue", "Fork"][(r % 4) as usize], r % 900, r % 100_000, r % 17).as_bytes());
    }
    v
}

#[test]
fn record_shaped_input_takes_the_envelope() {
    let data = delimited(300_000);
    let (mut rec, mut plain) = (Vec::new(), Vec::new());
    glyd::compress_records_into_max(&data, &mut rec);
    glyd::compress_parallel_into_max(&data, &mut plain);
    assert_eq!(&rec[..8], b"GLYDRECS");
    assert!(glyd::decompress(&rec).unwrap() == data, "sequential decode");
    assert!(glyd::decompress_parallel(&rec).unwrap() == data, "parallel decode");
    let mut exact = vec![0u8; data.len()];
    assert_eq!(glyd::decompress_into(&rec, &mut exact).unwrap(), data.len());
    assert!(exact == data);
    assert!(rec.len() < plain.len(), "the transform should pay: {} vs {}", rec.len(), plain.len());
    let mut ultra = Vec::new();
    glyd::compress_records_into_ultra(&data, &mut ultra);
    assert_eq!(&ultra[..8], b"GLYDRECS");
    assert!(glyd::decompress(&ultra).unwrap() == data);
}

#[test]
fn json_lines_take_the_envelope() {
    let data = json_lines(200_000);
    let (mut rec, mut plain) = (Vec::new(), Vec::new());
    glyd::compress_records_into_max(&data, &mut rec);
    glyd::compress_parallel_into_max(&data, &mut plain);
    assert_eq!(&rec[..8], b"GLYDRECS");
    assert!(rec.len() < plain.len(), "typed fields should pay: {} vs {}", rec.len(), plain.len());
    assert!(glyd::decompress(&rec).unwrap() == data);
    assert!(glyd::decompress_parallel(&rec).unwrap() == data);
}

/// Prose: lines of words, no field structure.
fn prose(lines: usize) -> Vec<u8> {
    let words = ["the", "storage", "bill", "falls", "when", "records", "share", "their", "fields", "across", "lines", "and", "text", "does", "not"];
    let mut x = 3u64;
    let mut v = Vec::new();
    for _ in 0..lines {
        let n = 3 + (rnd(&mut x) % 12) as usize;
        for k in 0..n {
            if k > 0 {
                v.push(b' ');
            }
            v.extend_from_slice(words[(rnd(&mut x) % words.len() as u64) as usize].as_bytes());
        }
        v.push(b'\n');
    }
    v
}

#[test]
fn other_input_takes_the_plain_path() {
    let data = prose(300_000);
    let (mut rec, mut plain) = (Vec::new(), Vec::new());
    glyd::compress_records_into_max(&data, &mut rec);
    glyd::compress_parallel_into_max(&data, &mut plain);
    assert_ne!(&rec[..8], b"GLYDRECS", "prose is not record-shaped");
    assert!(rec == plain, "the plain parallel stream, unit for unit");
    assert!(glyd::decompress(&rec).unwrap() == data);
    // Empty and tiny inputs.
    for n in [0usize, 1, 100] {
        let d = delimited(n);
        let mut c = Vec::new();
        glyd::compress_records_into_max(&d, &mut c);
        assert!(glyd::decompress(&c).unwrap() == d, "{n} lines");
    }
}
