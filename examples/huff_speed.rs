// Feasibility gate for entropy coding: how fast can we DECODE Huffman?
//
// Scalar single-stream Huffman decode has a serial dependency: each symbol's
// bit position depends on the previous symbol's length, so the CPU cannot
// start symbol N+1 until N is decoded. huff_probe measured that at ~3.4
// ns/sym, which would sink the decode floor.
//
// The fix is interleaving: split the symbols round-robin into N independent
// bitstreams, each with its own bit accumulator, and decode one from each
// per loop iteration. The N lookups have no dependency between them, so they
// pipeline. This measures ns/sym for N = 1, 2, 4, 8 on the real token stream,
// pinned to one core. If interleaving reaches ~1 ns/sym, entropy coding fits.
use simd_stream_codec::format::*;
use simd_stream_codec::huffman::*;
use std::time::Instant;

#[cfg(target_os = "linux")]
fn pin(c: usize) { unsafe {
    let mut s: libc::cpu_set_t = std::mem::zeroed();
    libc::CPU_ZERO(&mut s); libc::CPU_SET(c, &mut s);
    libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &s);
}}
#[cfg(not(target_os = "linux"))]
fn pin(_c: usize) {}

/// Encode `data` into `n` interleaved MSB-first bitstreams: symbol i goes to
/// stream i % n. Returns the packed bytes of each stream.
fn encode_interleaved(data: &[u8], codes: &[u16; 256], lengths: &[u8; 256], n: usize) -> Vec<Vec<u8>> {
    let mut writers: Vec<(BitWriter, Vec<u8>)> = (0..n).map(|_| (BitWriter::new(), Vec::new())).collect();
    for (i, &b) in data.iter().enumerate() {
        let (w, out) = &mut writers[i % n];
        w.put(codes[b as usize], lengths[b as usize], out);
    }
    writers.into_iter().map(|(mut w, mut out)| { w.finish(&mut out); out }).collect()
}

struct Acc<'a> { src: &'a [u8], pos: usize, bits: u64, nbits: u32 }
impl<'a> Acc<'a> {
    fn new(src: &'a [u8]) -> Self { Acc { src, pos: 0, bits: 0, nbits: 0 } }
    #[inline(always)]
    fn refill(&mut self) {
        while self.nbits <= 56 {
            let byte = if self.pos < self.src.len() { self.src[self.pos] } else { 0 };
            self.pos += 1;
            self.bits = (self.bits << 8) | byte as u64;
            self.nbits += 8;
        }
    }
    #[inline(always)]
    fn decode(&mut self, table: &DecodeTable) -> u8 {
        let idx = ((self.bits >> (self.nbits - TABLE_BITS)) & ((TABLE_SIZE as u64) - 1)) as usize;
        let l = table.len[idx];
        self.nbits -= l as u32;
        table.sym[idx]
    }
}

/// Decode `n_syms` symbols from `n` interleaved streams into `out`.
fn decode_interleaved(streams: &[Vec<u8>], table: &DecodeTable, n_syms: usize, out: &mut [u8]) {
    let n = streams.len();
    let mut accs: Vec<Acc> = streams.iter().map(|s| Acc::new(s)).collect();
    let mut i = 0;
    // Process one symbol from each stream per round; refill all first so the
    // decodes have no inter-stream dependency.
    while i + n <= n_syms {
        for a in accs.iter_mut() { a.refill(); }
        for (s, a) in accs.iter_mut().enumerate() {
            out[i + s] = a.decode(table);
        }
        i += n;
    }
    while i < n_syms {
        let a = &mut accs[i % n];
        a.refill();
        out[i] = a.decode(table);
        i += 1;
    }
}

fn timed(runs: usize, mut op: impl FnMut()) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..runs {
        let t = Instant::now();
        op();
        let e = t.elapsed().as_secs_f64();
        if e < best { best = e; }
    }
    best
}

fn main() {
    let files = ["dickens","mozilla","mr","nci","ooffice","osdb",
                 "reymont","samba","sao","webster","xml","x-ray"];
    let dir = std::path::Path::new("corpus");
    // Gather the whole token stream across all blocks.
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
    let lengths = build_lengths(&hist);
    let codes = build_codes(&lengths);
    let coded_bits: u64 = (0..256).map(|i| hist[i] * lengths[i] as u64).sum();
    println!("token stream {} symbols, huffman {:.3} bits/sym ({} -> {} bytes, {:.1}%)",
             n, coded_bits as f64 / n as f64, n, coded_bits / 8,
             100.0 * (coded_bits as f64 / 8.0) / n as f64);

    pin(4);
    let table = DecodeTable::build(&lengths).unwrap();
    let mut out = vec![0u8; n];
    let gb = 1024.0f64 * 1024.0 * 1024.0;
    // Decode throughput is measured against OUTPUT (token) bytes: that is the
    // rate the codec's decoder would sustain on this stream.
    println!("{:>4}  {:>10}  {:>9}  {:>10}", "N", "ms", "ns/sym", "GB/s tok");
    for &nstreams in &[1usize, 2, 4, 8] {
        let streams = encode_interleaved(&tokens, &codes, &lengths, nstreams);
        // verify round-trip
        decode_interleaved(&streams, &table, n, &mut out);
        assert_eq!(&out[..], &tokens[..], "interleaved N={} roundtrip mismatch", nstreams);
        let t = timed(7, || decode_interleaved(&streams, &table, n, &mut out));
        println!("{:>4}  {:>10.2}  {:>9.3}  {:>10.3}",
                 nstreams, t * 1e3, t * 1e9 / n as f64, (n as f64 / gb) / t);
    }
    // For reference: the whole-corpus decode budget at the T1.4 floor is set by
    // OUTPUT bytes, not token bytes. Tokens are ~7% of output, so token decode
    // has to be several times faster than 3.130 GB/s to be affordable.
    println!("(T1.4 floor is 3.130 GB/s of OUTPUT; tokens are ~7% of output, so");
    println!(" token decode must clear ~10+ GB/s to add little to total decode.)");
}
