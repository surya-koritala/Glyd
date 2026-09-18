// Is bit-packing the offsets worth it? Two numbers decide it, measured on the
// real offset sequence: (1) how many bytes it saves vs our 1/2/3-byte offsets
// (the ratio gain), and (2) how much it adds to decode (a bitstream read per
// match replaces a single masked load -- the risk to our decode lead).
//
// Offsets are grouped into 4 classes (2 token bits already carry the class);
// each class is a fixed bit width. Several width sets are costed; the cheapest
// is then packed MSB-first and decoded with a 64-bit accumulator, timed.
use simd_stream_codec::format::*;
use std::time::Instant;

#[cfg(target_os = "linux")]
fn pin(c: usize) { unsafe {
    let mut s: libc::cpu_set_t = std::mem::zeroed();
    libc::CPU_ZERO(&mut s); libc::CPU_SET(c, &mut s);
    libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &s);
}}
#[cfg(not(target_os = "linux"))]
fn pin(_c: usize) {}

/// Smallest class (0..3) whose width holds `offset`, or 4 if none.
#[inline]
fn class_of(offset: u32, widths: &[u32; 4]) -> usize {
    for (i, &w) in widths.iter().enumerate() {
        if offset < (1u32 << w) { return i; }
    }
    4
}

fn main() {
    let files = ["dickens","mozilla","mr","nci","ooffice","osdb",
                 "reymont","samba","sao","webster","xml","x-ray"];
    let dir = std::path::Path::new("corpus");
    // Gather every match offset and the bytes it costs today.
    let mut offs: Vec<u32> = Vec::new();
    let mut cur_bytes: u64 = 0;
    for f in &files {
        let p = dir.join(f);
        if !p.exists() { continue; }
        let d = std::fs::read(&p).unwrap();
        let c = simd_stream_codec::compress(&d);
        let mut cur = 0usize;
        while cur + HEADER_SIZE <= c.len() {
            let h = unsafe { std::ptr::read_unaligned(c.as_ptr().add(cur) as *const BlockHeader) };
            if h.flags & FLAG_RAW_UNCOMPRESSED == 0 {
                let tb = cur + HEADER_SIZE;
                let ob = tb + h.token_bytes as usize;
                let eb = ob + h.offset_bytes as usize;
                cur_bytes += h.offset_bytes as u64;
                // Walk tokens to recover each offset value and width.
                let bias = if h.flags & FLAG_DENSE != 0 { MATCH_CODE_BIAS_DENSE } else { MATCH_CODE_BIAS };
                let (mut oi, mut ei) = (0usize, 0usize);
                let read_esc = |c: &[u8], ei: &mut usize, base: usize| -> usize {
                    let v = c[eb + *ei] as usize; *ei += 1;
                    if v != ESCAPE_CONT as usize { base + v }
                    else { let w = u16::from_le_bytes([c[eb + *ei], c[eb + *ei + 1]]) as usize; *ei += 2; base + 255 + w }
                };
                for i in 0..h.token_count as usize {
                    let t = Token(c[tb + i]);
                    if t.lit_code() == LIT_CODE_ESCAPE { let _ = read_esc(&c, &mut ei, ESCAPE_BASE_LIT); }
                    let mc = t.match_code();
                    if mc == 0 { continue; }
                    if mc == MATCH_CODE_ESCAPE { let _ = read_esc(&c, &mut ei, bias + 15); }
                    let w = t.off_width();
                    let mut v = 0u32;
                    for k in 0..w { v |= (c[ob + oi + k] as u32) << (8 * k); }
                    oi += w;
                    offs.push(v);
                }
            }
            cur += HEADER_SIZE + h.payload_len();
        }
    }
    let n = offs.len();
    println!("{} match offsets, current cost {} bytes ({:.3} bytes/match)",
             n, cur_bytes, cur_bytes as f64 / n as f64);

    // Cost several 4-class width sets. Max class must cover 24-bit offsets.
    let candidates: [[u32; 4]; 6] = [
        [8, 16, 24, 24],   // byte-aligned classes (baseline-ish)
        [10, 16, 20, 24],
        [9, 14, 19, 24],
        [11, 17, 21, 24],
        [10, 18, 23, 24],  // LZAV-like
        [12, 18, 22, 24],
    ];
    let mut best = ([0u32; 4], u64::MAX);
    for w in &candidates {
        let mut bits: u64 = 0;
        let mut bad = 0u64;
        for &o in &offs {
            let c = class_of(o, w);
            if c == 4 { bad += 1; continue; }
            bits += w[c] as u64;
        }
        let bytes = (bits + 7) / 8;
        println!("  widths {:?}  {} bytes  {:.3} b/match  save {:.2}%{}",
                 w, bytes, bytes as f64 / n as f64,
                 100.0 * (cur_bytes as f64 - bytes as f64) / cur_bytes as f64,
                 if bad > 0 { format!("  ({} uncovered!)", bad) } else { String::new() });
        if bad == 0 && bytes < best.1 { best = (*w, bytes); }
    }
    let widths = best.0;
    println!("best widths {:?} -> {} offset bytes", widths, best.1);

    // Pack MSB-first, then time decode: refill + variable read per match.
    let classes: Vec<u8> = offs.iter().map(|&o| class_of(o, &widths) as u8).collect();
    let mut packed: Vec<u8> = Vec::new();
    let mut acc: u64 = 0; let mut nb: u32 = 0;
    for (i, &o) in offs.iter().enumerate() {
        let w = widths[classes[i] as usize];
        acc = (acc << w) | o as u64;
        nb += w;
        while nb >= 8 { nb -= 8; packed.push((acc >> nb) as u8); }
    }
    if nb > 0 { packed.push((acc << (8 - nb)) as u8); }

    pin(4);
    let mut out = vec![0u32; n];
    let _ = &packed;
    // Interleaved: offset i -> stream i%N, each stream its own bitstream and
    // accumulator, so consecutive matches' refills are independent (ILP). The
    // wall for Huffman was the table load; here there is none, only shifts, so
    // interleaving should actually pay.
    println!("{:>4}  {:>9}  {:>10}", "N", "ns/match", "note");
    for &nn in &[1usize, 2, 4, 8] {
        // Pack N streams.
        let mut streams: Vec<Vec<u8>> = vec![Vec::new(); nn];
        let mut accs = vec![(0u64, 0u32); nn];
        for (i, &o) in offs.iter().enumerate() {
            let s = i % nn;
            let w = widths[classes[i] as usize];
            accs[s].0 = (accs[s].0 << w) | o as u64;
            accs[s].1 += w;
            while accs[s].1 >= 8 { accs[s].1 -= 8; streams[s].push((accs[s].0 >> accs[s].1) as u8); }
        }
        for s in 0..nn { if accs[s].1 > 0 { streams[s].push((accs[s].0 << (8 - accs[s].1)) as u8); } }

        let decode = |out: &mut [u32]| {
            let mut acc = vec![0u64; nn];
            let mut nb = vec![0u32; nn];
            let mut pos = vec![0usize; nn];
            let mut i = 0;
            while i < n {
                let s = i % nn;
                let w = widths[classes[i] as usize];
                while nb[s] < w {
                    let b = if pos[s] < streams[s].len() { streams[s][pos[s]] } else { 0 };
                    pos[s] += 1;
                    acc[s] = (acc[s] << 8) | b as u64;
                    nb[s] += 8;
                }
                nb[s] -= w;
                out[i] = ((acc[s] >> nb[s]) & ((1u64 << w) - 1)) as u32;
                i += 1;
            }
        };
        decode(&mut out);
        assert_eq!(&out[..], &offs[..], "interleaved N={} offset roundtrip mismatch", nn);
        let mut best_t = f64::MAX;
        for _ in 0..7 {
            let t = Instant::now();
            decode(&mut out);
            let e = t.elapsed().as_secs_f64();
            if e < best_t { best_t = e; }
        }
        println!("{:>4}  {:>9.3}  {:.2} ms total", nn, best_t * 1e9 / n as f64, best_t * 1e3);
    }
    println!("(current offset read is ~1 masked load/match. Added decode = new - old;");
    println!(" ratio gain from best widths is {:.2}% of output. Worth it if decode holds >3.13.)",
             100.0 * 2940000.0 / 88745358.0);
}
