// SPIKE (throwaway): how fast can literals be Huffman-decoded on this chip
// with a proper decoder? N interleaved LSB-first bitstreams, packed 2^11
// u16 table (sym | len << 8), branchless 64-bit refill once per 4 symbols,
// 4 symbols per stream per iteration. Measured on the real literal streams
// of every Silesia block (our default parse), best of 5.
use simd_stream_codec::format::*;
use simd_stream_codec::huffman::*;
use std::time::Instant;

const TB: u32 = MAX_CODE_LEN; // table bits

fn reverse_bits(code: u16, len: u8) -> u16 {
    let mut r = 0u16;
    for i in 0..len { r |= ((code >> i) & 1) << (len - 1 - i); }
    r
}

/// Encode symbols round-robin into `n` LSB-first bitstreams. Each stream is
/// padded with 8 zero bytes so the decoder's wide refill never overruns.
fn encode(data: &[u8], lengths: &[u8; 256], n: usize) -> Vec<Vec<u8>> {
    let codes = build_codes(lengths);
    let rev: Vec<u16> = (0..256).map(|s| reverse_bits(codes[s], lengths[s])).collect();
    let mut acc = vec![(0u64, 0u32); n];
    let mut out: Vec<Vec<u8>> = vec![Vec::new(); n];
    for (i, &b) in data.iter().enumerate() {
        let k = i % n;
        let (bits, nb) = &mut acc[k];
        *bits |= (rev[b as usize] as u64) << *nb;
        *nb += lengths[b as usize] as u32;
        while *nb >= 8 { out[k].push(*bits as u8); *bits >>= 8; *nb -= 8; }
    }
    for k in 0..n {
        let (bits, nb) = acc[k];
        if nb > 0 { out[k].push(bits as u8); }
        out[k].extend_from_slice(&[0u8; 8]);
    }
    out
}

/// Packed decode table: entry = sym | (len << 8), indexed by the next TB
/// bits (LSB-first).
fn table(lengths: &[u8; 256]) -> Vec<u16> {
    let codes = build_codes(lengths);
    let mut t = vec![0u16; 1 << TB];
    for s in 0..256 {
        let l = lengths[s];
        if l == 0 { continue; }
        let r = reverse_bits(codes[s], l) as usize;
        let step = 1usize << l;
        let mut i = r;
        while i < (1 << TB) { t[i] = s as u16 | ((l as u16) << 8); i += step; }
    }
    t
}

struct St { p: *const u8, bits: u64, cnt: u32 }

#[inline(always)]
unsafe fn refill(s: &mut St) {
    // Giesen's branchless refill: top up to >= 56 bits.
    s.bits |= std::ptr::read_unaligned(s.p as *const u64) << s.cnt;
    s.p = s.p.add(((63 - s.cnt) >> 3) as usize);
    s.cnt |= 56;
}

#[inline(always)]
unsafe fn sym(s: &mut St, t: *const u16) -> u8 {
    let e = *t.add((s.bits & ((1 << TB) - 1)) as usize);
    let l = (e >> 8) as u32;
    s.bits >>= l;
    s.cnt -= l;
    e as u8
}

/// Decode `n_syms` symbols from N interleaved streams. N is a const so the
/// state stays in registers.
#[inline(never)]
unsafe fn decode<const N: usize>(streams: &[Vec<u8>], t: &[u16], n_syms: usize, out: *mut u8) {
    let t = t.as_ptr();
    let mut st: [St; N] = std::array::from_fn(|k| St { p: streams[k].as_ptr(), bits: 0, cnt: 0 });
    for k in 0..N { refill(&mut st[k]); }
    // Main loop: 4 symbols per stream per iteration (4 * 11 = 44 <= 56).
    let full = n_syms / (4 * N);
    let mut o = out;
    for _ in 0..full {
        for k in 0..N { refill(&mut st[k]); }
        for j in 0..4 {
            for k in 0..N {
                *o.add(j * N + k) = sym(&mut st[k], t);
            }
        }
        o = o.add(4 * N);
    }
    // Tail, one symbol at a time in stream order.
    let done = full * 4 * N;
    for i in done..n_syms {
        let k = i % N;
        refill(&mut st[k]);
        *o = sym(&mut st[k], t);
        o = o.add(1);
    }
}

struct Blk { lit: Vec<u8>, lengths: [u8; 256], streams: Vec<Vec<u8>>, tbl: Vec<u16>, coded: usize }

fn main() {
    let files = ["dickens","mozilla","mr","nci","ooffice","osdb","reymont","samba","sao","webster","xml","x-ray"];
    let mut lits: Vec<Vec<u8>> = Vec::new();
    for f in &files {
        let d = std::fs::read(format!("corpus/{}", f)).unwrap();
        let c = simd_stream_codec::compress(&d);
        let mut cur = 0usize;
        while cur + HEADER_SIZE <= c.len() {
            let h = unsafe { std::ptr::read_unaligned(c.as_ptr().add(cur) as *const BlockHeader) };
            if h.flags & (FLAG_RAW_UNCOMPRESSED | FLAG_DENSE) == 0 {
                let lb = cur + HEADER_SIZE + h.token_bytes as usize + h.offset_bytes as usize + h.extras_bytes as usize;
                lits.push(c[lb..lb + h.literal_len as usize].to_vec());
            }
            cur += HEADER_SIZE + h.payload_len();
        }
    }
    let total: usize = lits.iter().map(|l| l.len()).sum();
    let gb = 1024.0f64 * 1024.0 * 1024.0;
    println!("{} blocks, {} MB of literals", lits.len(), total >> 20);
    let mut out = vec![0u8; lits.iter().map(|l| l.len()).max().unwrap() + 64];

    for n in [1usize, 2, 4, 8, 16] {
        let blks: Vec<Blk> = lits.iter().map(|l| {
            let mut hist = [0u64; 256];
            for &b in l { hist[b as usize] += 1; }
            let lengths = build_lengths(&hist);
            let streams = encode(l, &lengths, n);
            let coded: usize = streams.iter().map(|s| s.len() - 8).sum::<usize>() + 256 / 2;
            Blk { lit: l.clone(), lengths, streams, tbl: table(&lengths), coded }
        }).collect();
        let coded: usize = blks.iter().map(|b| b.coded).sum();
        // correctness on every block
        for b in &blks {
            unsafe { match n { 1 => decode::<1>(&b.streams, &b.tbl, b.lit.len(), out.as_mut_ptr()), 2 => decode::<2>(&b.streams, &b.tbl, b.lit.len(), out.as_mut_ptr()), 4 => decode::<4>(&b.streams, &b.tbl, b.lit.len(), out.as_mut_ptr()), 8 => decode::<8>(&b.streams, &b.tbl, b.lit.len(), out.as_mut_ptr()), _ => decode::<16>(&b.streams, &b.tbl, b.lit.len(), out.as_mut_ptr()) } }
            assert_eq!(&out[..b.lit.len()], &b.lit[..], "mismatch N={}", n);
        }
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t = Instant::now();
            for b in &blks {
                unsafe { match n { 1 => decode::<1>(&b.streams, &b.tbl, b.lit.len(), out.as_mut_ptr()), 2 => decode::<2>(&b.streams, &b.tbl, b.lit.len(), out.as_mut_ptr()), 4 => decode::<4>(&b.streams, &b.tbl, b.lit.len(), out.as_mut_ptr()), 8 => decode::<8>(&b.streams, &b.tbl, b.lit.len(), out.as_mut_ptr()), _ => decode::<16>(&b.streams, &b.tbl, b.lit.len(), out.as_mut_ptr()) } }
            }
            best = best.min(t.elapsed().as_secs_f64());
        }
        // table build cost (decoder side, per block)
        let tb = { let t = Instant::now(); for b in &blks { std::hint::black_box(table(&b.lengths)); } t.elapsed().as_secs_f64() };
        println!("N={:<2} coded {:.1}% of raw  decode {:>6.2} ms  {:.2} ns/sym  {:>5.2} GB/s of literals   (table build {:.1} ms)",
            n, 100.0 * coded as f64 / total as f64, best * 1e3, best * 1e9 / total as f64, total as f64 / gb / best, tb * 1e3);
    }
}
