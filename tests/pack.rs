//! Packs: many small objects as one stream with an index. Every object
//! comes back, alone or all at once; record-shaped objects cost far
//! less packed than alone; objects of any bytes, empty ones included;
//! corrupted packs are refused.

fn rnd(x: &mut u64) -> u64 {
    *x ^= *x << 13;
    *x ^= *x >> 7;
    *x ^= *x << 17;
    *x
}

fn events(n: usize, seed: u64) -> Vec<Vec<u8>> {
    let mut x = seed;
    let mut t = 1_700_000_000u64;
    (0..n)
        .map(|_| {
            let mut o = Vec::new();
            for _ in 0..8 {
                let r = rnd(&mut x);
                t += r % 5;
                o.extend_from_slice(format!("{{\"ts\": {}, \"host\": \"h{}\", \"status\": {}, \"bytes\": {}}}\n", t, r % 30, if r % 9 == 0 { 500 } else { 200 }, (r >> 20) % 5000).as_bytes());
            }
            o
        })
        .collect()
}

#[test]
fn objects_come_back_alone_and_together() {
    let objs = events(1000, 1);
    let refs: Vec<&[u8]> = objs.iter().map(|v| v.as_slice()).collect();
    let mut pack = Vec::new();
    glyd::compress_pack(&refs, &mut pack, glyd::compress_into_max);
    let raw: usize = objs.iter().map(|o| o.len()).sum();
    assert!(pack.len() * 8 < raw, "packed events should shrink eightfold: {} of {raw}", pack.len());
    assert_eq!(glyd::pack_len(&pack).unwrap(), 1000);
    assert!(glyd::decompress_pack(&pack).unwrap() == objs);
    for i in [0usize, 1, 499, 999] {
        assert!(glyd::decompress_pack_object(&pack, i).unwrap() == objs[i], "object {i}");
    }
    assert!(glyd::decompress_pack_object(&pack, 1000).is_err());
    // Alone, each object costs far more.
    let alone: usize = objs.iter().map(|o| { let mut c = Vec::new(); glyd::compress_into_max(o, &mut c); c.len() }).sum();
    assert!(pack.len() * 2 < alone, "{} packed vs {alone} alone", pack.len());
}

#[test]
fn any_bytes_and_corruption() {
    let mut x = 5u64;
    let mut objs: Vec<Vec<u8>> = Vec::new();
    for i in 0..300 {
        let len = match i % 4 { 0 => 0, 1 => 1, 2 => (rnd(&mut x) % 5000) as usize, _ => 64 };
        objs.push((0..len).map(|_| (rnd(&mut x) >> 8) as u8).collect());
    }
    let refs: Vec<&[u8]> = objs.iter().map(|v| v.as_slice()).collect();
    for level in [glyd::compress_into_max as fn(&[u8], &mut Vec<u8>), glyd::compress_into_ultra] {
        let mut pack = Vec::new();
        glyd::compress_pack(&refs, &mut pack, level);
        assert!(glyd::decompress_pack(&pack).unwrap() == objs);
        assert!(glyd::decompress_pack_object(&pack, 2).unwrap() == objs[2]);
    }
    let mut pack = Vec::new();
    glyd::compress_pack(&[], &mut pack, glyd::compress_into_max);
    assert!(glyd::decompress_pack(&pack).unwrap().is_empty());
    let mut pack = Vec::new();
    glyd::compress_pack(&refs, &mut pack, glyd::compress_into_max);
    for _ in 0..200 {
        let mut bad = pack.clone();
        let i = (rnd(&mut x) as usize) % bad.len();
        bad[i] ^= 1 << (rnd(&mut x) % 8);
        if let Ok(d) = glyd::decompress_pack(&bad) {
            assert!(d == objs, "a corrupted pack decoded to other bytes unnoticed");
        }
    }
    assert!(glyd::decompress_pack(b"GLYDPACK").is_err());
    assert!(glyd::decompress_pack(&pack[..pack.len() / 2]).is_err());
}
