// Deterministic replay of the G1 fuzz sequence that isolates the exact
// mutation and decode path that corrupts memory.
use simd_stream_codec::{
    compress, compress_parallel, decompress, decompress_into_raw, decompress_parallel,
    decompress_parallel_into_raw,
};
use std::io::Write;

struct Rng(u64);
impl Rng {
    #[inline]
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

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
    seeds
}

fn dump(case: &str, stream_idx: usize, iter: u64, buf: &[u8]) {
    let mut f = std::fs::File::create("/tmp/alk_last_case.txt").unwrap();
    writeln!(f, "iter={} stream_idx={} path={} len={}", iter, stream_idx, case, buf.len()).unwrap();
    f.flush().unwrap();
    std::fs::write("/tmp/alk_last_case.bin", buf).unwrap();
}

fn main() {
    // Optional: only run the named path, to isolate which one corrupts memory.
    let only: Option<String> = std::env::args().nth(1);
    let limit: u64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);

    let seeds = seed_inputs();
    let mut streams: Vec<(Vec<u8>, usize)> = Vec::new();
    for s in &seeds {
        streams.push((compress(s), s.len()));
        streams.push((compress_parallel(s), s.len()));
    }

    let mut rng = Rng(0xDEAD_BEEF_CAFE_F00D);
    let mut mutations: u64 = 0;
    let max_out = seeds.iter().map(|s| s.len()).max().unwrap() + 4096;
    let mut dst = vec![0u8; max_out];
    let mut corrupted: Vec<u8> = Vec::with_capacity(max_out);

    let want = |p: &str| only.as_deref().map_or(true, |o| o == p);

    while mutations < limit {
        for (si, (stream, _orig_len)) in streams.iter().enumerate() {
            if mutations >= limit {
                break;
            }
            if stream.is_empty() {
                continue;
            }
            corrupted.clear();
            corrupted.extend_from_slice(stream);

            let n_mut = 1 + (rng.next() % 4) as usize;
            for _ in 0..n_mut {
                let idx = (rng.next() as usize) % corrupted.len();
                match rng.next() % 4 {
                    0 => corrupted[idx] ^= 1u8 << (rng.next() % 8),
                    1 => corrupted[idx] = (rng.next() >> 24) as u8,
                    2 => corrupted[idx] = if rng.next() & 1 == 0 { 0x00 } else { 0xFF },
                    _ => {
                        let j = (rng.next() as usize) % corrupted.len();
                        corrupted.swap(idx, j);
                    }
                }
            }
            if rng.next() % 8 == 0 && corrupted.len() > 1 {
                let new_len = 1 + (rng.next() as usize) % (corrupted.len() - 1);
                corrupted.truncate(new_len);
            }

            if want("decompress") {
                dump("decompress", si, mutations, &corrupted);
                let _ = decompress(&corrupted);
            }
            if want("decompress_into_raw") {
                dump("decompress_into_raw", si, mutations, &corrupted);
                let _ = decompress_into_raw(&corrupted, &mut dst);
            }
            if mutations % 8 == 0 {
                if want("decompress_parallel") {
                    dump("decompress_parallel", si, mutations, &corrupted);
                    let _ = decompress_parallel(&corrupted);
                }
                if want("decompress_parallel_into_raw") {
                    dump("decompress_parallel_into_raw", si, mutations, &corrupted);
                    let _ = decompress_parallel_into_raw(&corrupted, &mut dst);
                }
            }

            mutations += 1;
            if mutations % 5000 == 0 {
                eprintln!("ok through {} mutations", mutations);
            }
        }
    }
    println!("SURVIVED {} mutations", mutations);
}
