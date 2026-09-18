// The moonshot's feasibility gate: how fast can interleaved rANS DECODE?
//
// Table Huffman topped out at ~3 ns/sym because each symbol is a dependent
// bit-walk plus a table load (huff_speed). rANS replaces the bit-walk with a
// branchless arithmetic state update; N independent states decoded in lockstep
// expose instruction-level parallelism the Huffman loop could not. It still
// does one table load per symbol, so it may hit the same load-throughput wall
// -- which is exactly what this measures, before any encoder/format is built.
//
// 32-bit rANS, 8-bit renormalization, 12-bit frequencies (ryg_rans style,
// Fabian Giesen, public domain technique). Each of N interleaved coders owns
// its own byte buffer, encoded in reverse and read backward at decode.
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

const SCALE_BITS: u32 = 12;
const M: u32 = 1 << SCALE_BITS; // total frequency
const RANS_L: u32 = 1 << 23; // lower bound of the normalized interval

/// Normalize a histogram to frequencies summing to exactly M, every used
/// symbol at least 1.
fn normalize(hist: &[u64; 256]) -> ([u32; 256], [u32; 256]) {
    let total: u64 = hist.iter().sum();
    let mut freq = [0u32; 256];
    let mut used = 0;
    for i in 0..256 {
        if hist[i] > 0 {
            freq[i] = (((hist[i] as u128 * M as u128) / total as u128) as u32).max(1);
            used += 1;
        }
    }
    // Fix the sum to exactly M by adjusting the largest frequencies.
    let mut sum: i64 = freq.iter().map(|&f| f as i64).sum();
    let target = M as i64;
    let _ = used;
    while sum != target {
        // Move one unit onto or off of the current largest frequency.
        let (mut bi, mut bf) = (0usize, 0u32);
        for i in 0..256 {
            if freq[i] > bf { bf = freq[i]; bi = i; }
        }
        if sum < target { freq[bi] += 1; sum += 1; }
        else if freq[bi] > 1 { freq[bi] -= 1; sum -= 1; }
    }
    let mut cum = [0u32; 256];
    let mut c = 0u32;
    for i in 0..256 { cum[i] = c; c += freq[i]; }
    (freq, cum)
}

/// One rANS coder's own byte buffer, encoded in reverse.
fn encode_stream(syms: &[u8], which: &[bool], freq: &[u32; 256], cum: &[u32; 256]) -> Vec<u8> {
    let mut buf = Vec::new();
    let mut state = RANS_L;
    for i in (0..syms.len()).rev() {
        if !which[i] { continue; }
        let s = syms[i] as usize;
        let f = freq[s];
        let x_max = ((RANS_L >> SCALE_BITS) << 8) * f;
        while state >= x_max {
            buf.push((state & 0xff) as u8);
            state >>= 8;
        }
        state = ((state / f) << SCALE_BITS) + (state % f) + cum[s];
    }
    for _ in 0..4 { buf.push((state & 0xff) as u8); state >>= 8; }
    buf
}

struct Decoder<'a> { buf: &'a [u8], pos: usize, state: u32 }
impl<'a> Decoder<'a> {
    fn new(buf: &'a [u8]) -> Self {
        let mut pos = buf.len();
        let mut state = 0u32;
        for _ in 0..4 { pos -= 1; state = (state << 8) | buf[pos] as u32; }
        Decoder { buf, pos, state }
    }
    #[inline(always)]
    fn decode(&mut self, slot2sym: &[u8; M as usize], freq: &[u32; 256], cum: &[u32; 256]) -> u8 {
        let slot = self.state & (M - 1);
        let s = slot2sym[slot as usize];
        self.state = freq[s as usize] * (self.state >> SCALE_BITS) + slot - cum[s as usize];
        while self.state < RANS_L {
            self.pos -= 1;
            self.state = (self.state << 8) | self.buf[self.pos] as u32;
        }
        s
    }
}

fn main() {
    let files = ["dickens","mozilla","mr","nci","ooffice","osdb",
                 "reymont","samba","sao","webster","xml","x-ray"];
    let dir = std::path::Path::new("corpus");
    let mut tokens: Vec<u8> = Vec::new();
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
                tokens.extend_from_slice(&c[tb..tb + h.token_bytes as usize]);
            }
            cur += HEADER_SIZE + h.payload_len();
        }
    }
    let n = tokens.len();
    let mut hist = [0u64; 256];
    for &b in &tokens { hist[b as usize] += 1; }
    let (freq, cum) = normalize(&hist);
    let mut slot2sym = [0u8; M as usize];
    for s in 0..256 {
        for k in 0..freq[s] { slot2sym[(cum[s] + k) as usize] = s as u8; }
    }
    // Entropy achieved by this frequency table.
    let total: f64 = hist.iter().sum::<u64>() as f64;
    let mut bits = 0.0f64;
    for s in 0..256 {
        if hist[s] > 0 {
            let p = freq[s] as f64 / M as f64;
            bits += hist[s] as f64 * -p.log2();
        }
    }
    println!("token stream {} symbols, rANS {:.3} bits/sym ({:.0} bytes, {:.1}%)",
             n, bits / total, bits / 8.0, 100.0 * (bits / 8.0) / n as f64);

    pin(4);
    let mut out = vec![0u8; n];
    let gb = 1024.0f64 * 1024.0 * 1024.0;
    println!("{:>4}  {:>10}  {:>9}  {:>10}", "N", "ms", "ns/sym", "GB/s tok");
    for &nstreams in &[1usize, 2, 4, 8, 16, 32] {
        // Assign symbol i to coder i % nstreams.
        let bufs: Vec<Vec<u8>> = (0..nstreams).map(|j| {
            let which: Vec<bool> = (0..n).map(|i| i % nstreams == j).collect();
            encode_stream(&tokens, &which, &freq, &cum)
        }).collect();
        // Decode round-robin.
        let decode = |out: &mut [u8]| {
            let mut decs: Vec<Decoder> = bufs.iter().map(|b| Decoder::new(b)).collect();
            let mut i = 0;
            while i + nstreams <= n {
                for d in decs.iter_mut() {
                    out[i] = d.decode(&slot2sym, &freq, &cum);
                    i += 1;
                }
            }
            let mut j = 0;
            while i < n { out[i] = decs[j].decode(&slot2sym, &freq, &cum); i += 1; j += 1; }
        };
        decode(&mut out);
        assert_eq!(&out[..], &tokens[..], "rANS N={} roundtrip mismatch", nstreams);
        let mut best = f64::MAX;
        for _ in 0..7 {
            let t = Instant::now();
            decode(&mut out);
            let e = t.elapsed().as_secs_f64();
            if e < best { best = e; }
        }
        println!("{:>4}  {:>10.2}  {:>9.3}  {:>10.3}",
                 nstreams, best * 1e3, best * 1e9 / n as f64, (n as f64 / gb) / best);
    }
    println!("(need ~10+ GB/s on tokens to add little to the 3.130 GB/s output decode)");
}
