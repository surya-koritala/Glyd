// What does entropy coding cost and save for real, per block, with the actual
// huffman.rs rather than a Shannon bound? The earlier probe used one corpus-wide
// histogram; the codec will build a table per block, so this measures exactly
// that: per-stream, per-block, coded size including the 128-byte table, taking
// the raw form whenever it is smaller (which is what the encoder will do).
//
// Then it times decoding every coded block back, pinned to one core, so the
// decode cost against the T1.4 floor is a number and not a guess.
use simd_stream_codec::format::*;
use simd_stream_codec::huffman::*;
use std::io::Write;
use std::time::Instant;

#[cfg(target_os = "linux")]
fn pin(c: usize) { unsafe {
    let mut s: libc::cpu_set_t = std::mem::zeroed();
    libc::CPU_ZERO(&mut s); libc::CPU_SET(c, &mut s);
    libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &s);
}}
#[cfg(not(target_os = "linux"))]
fn pin(_c: usize) {}

struct Stream {
    name: &'static str,
    symbols: u64,
    raw_bytes: u64,
    coded_bytes: u64,     // sum over blocks of min(raw, huffman + table)
    blocks_coded: u64,
    blocks_total: u64,
    shannon_bytes: f64,   // per-block Shannon bound, no table cost
    // Encoded blocks kept for the decode timing.
    packed: Vec<(Vec<u8>, Vec<u8>, usize)>, // (lengths packed, bits, n)
}

impl Stream {
    fn new(name: &'static str) -> Self {
        Stream { name, symbols: 0, raw_bytes: 0, coded_bytes: 0, blocks_coded: 0,
                 blocks_total: 0, shannon_bytes: 0.0, packed: Vec::new() }
    }
    fn add_block(&mut self, data: &[u8]) {
        if data.is_empty() { return; }
        let mut hist = [0u64; 256];
        for &b in data { hist[b as usize] += 1; }
        let n = data.len() as u64;
        let mut sh = 0.0f64;
        for &c in &hist {
            if c > 0 { let p = c as f64 / n as f64; sh -= (c as f64) * p.log2(); }
        }
        self.shannon_bytes += sh / 8.0;
        let lengths = build_lengths(&hist);
        let bits: u64 = (0..256).map(|i| hist[i] * lengths[i] as u64).sum();
        let coded = (bits + 7) / 8 + LENGTHS_BYTES as u64;
        self.symbols += n;
        self.raw_bytes += n;
        self.blocks_total += 1;
        if coded < n {
            self.coded_bytes += coded;
            self.blocks_coded += 1;
            let codes = build_codes(&lengths);
            let mut out = Vec::with_capacity(coded as usize + 8);
            let mut w = BitWriter::new();
            for &b in data { w.put(codes[b as usize], lengths[b as usize], &mut out); }
            w.finish(&mut out);
            let mut lens = Vec::new();
            pack_lengths(&lengths, &mut lens);
            self.packed.push((lens, out, data.len()));
        } else {
            self.coded_bytes += n;
        }
    }
    fn saving(&self) -> u64 { self.raw_bytes - self.coded_bytes }
}

fn timed_decode(s: &Stream, runs: usize) -> f64 {
    let max_n = s.packed.iter().map(|p| p.2).max().unwrap_or(0);
    let mut out = vec![0u8; max_n + 64];
    let mut best = f64::MAX;
    for _ in 0..runs {
        let t = Instant::now();
        for (lens, bits, n) in &s.packed {
            let lengths = unpack_lengths(lens);
            let table = DecodeTable::build(&lengths).unwrap();
            decode_into(&table, bits, *n, &mut out).unwrap();
        }
        let e = t.elapsed().as_secs_f64();
        if e < best { best = e; }
    }
    best
}

fn main() {
    let files = ["dickens","mozilla","mr","nci","ooffice","osdb",
                 "reymont","samba","sao","webster","xml","x-ray"];
    let dir = std::path::Path::new("corpus");
    let mut st = [Stream::new("tokens"), Stream::new("offset hi"),
                  Stream::new("offset lo"), Stream::new("literals")];
    let (mut orig, mut comp_total) = (0u64, 0u64);
    let mut blocks = 0u64;

    for f in &files {
        let p = dir.join(f);
        if !p.exists() { continue; }
        let d = std::fs::read(&p).unwrap();
        let c = simd_stream_codec::compress(&d);
        orig += d.len() as u64; comp_total += c.len() as u64;
        let mut cur = 0usize;
        while cur + HEADER_SIZE <= c.len() {
            let h = unsafe { std::ptr::read_unaligned(c.as_ptr().add(cur) as *const BlockHeader) };
            let raw = (h.flags & FLAG_RAW_UNCOMPRESSED) != 0;
            let (tc, oc, ec, ll) = (h.token_count as usize, h.offset_count as usize,
                                    h.extras_count as usize, h.literal_len as usize);
            let pay = if raw { h.uncompressed_len as usize } else { payload_len(tc, oc, ec, ll) };
            if !raw {
                blocks += 1;
                let tb = cur + HEADER_SIZE;
                st[0].add_block(&c[tb..tb + tc]);
                let ob = tb + tc;
                let mut hi = Vec::with_capacity(oc);
                let mut lo = Vec::with_capacity(oc);
                for i in 0..oc {
                    let v = u16::from_le_bytes([c[ob + i * 2], c[ob + i * 2 + 1]]);
                    hi.push((v >> 8) as u8); lo.push((v & 0xFF) as u8);
                }
                st[1].add_block(&hi);
                st[2].add_block(&lo);
                let lb = ob + oc * 2 + ec * 2;
                st[3].add_block(&c[lb..lb + ll]);
            }
            cur += HEADER_SIZE + pay;
        }
    }

    let ratio_now = orig as f64 / comp_total as f64;
    println!("Silesia {} bytes -> {} bytes, ratio {:.5}, {} compressed blocks", orig, comp_total, ratio_now, blocks);
    println!();
    println!("{:<10} {:>11} {:>11} {:>11} {:>8} {:>12} {:>8} {:>9}",
             "stream", "symbols", "raw", "huffman", "save%", "shannon", "capt%", "coded/bl");
    let mut total_save = 0u64;
    for s in &st {
        let sh_save = s.raw_bytes as f64 - s.shannon_bytes;
        println!("{:<10} {:>11} {:>11} {:>11} {:>7.2}% {:>12.0} {:>7.1}% {:>4}/{:<4}",
                 s.name, s.symbols, s.raw_bytes, s.coded_bytes,
                 100.0 * s.saving() as f64 / comp_total as f64,
                 sh_save, 100.0 * s.saving() as f64 / sh_save.max(1.0),
                 s.blocks_coded, s.blocks_total);
        total_save += s.saving();
    }
    println!();
    let scen = [("tokens only", 1usize), ("tokens + offset hi", 2), ("tokens + offsets", 3), ("all four", 4)];
    for (name, k) in scen {
        let save: u64 = st[..k].iter().map(|s| s.saving()).sum();
        let r = orig as f64 / (comp_total - save) as f64;
        println!("{:<20} saves {:>9} bytes = {:>5.2}%  ratio {:.5} -> {:.5} ({})",
                 name, save, 100.0 * save as f64 / comp_total as f64, ratio_now, r,
                 if r >= 2.45 { "T1.2 OK" } else { "T1.2 short" });
    }
    let _ = total_save;

    // Decode cost on one pinned core, best of runs.
    pin(4);
    println!();
    println!("decode cost of coded blocks, best of 5, current decode_into:");
    let mut cum = 0.0f64;
    for s in &st {
        let t = timed_decode(s, 5);
        let syms: usize = s.packed.iter().map(|p| p.2).sum();
        cum += t;
        println!("{:<10} {:>11} syms  {:>8.2} ms  {:>6.2} ns/sym  cumulative {:>7.2} ms",
                 s.name, syms, t * 1e3, t * 1e9 / syms.max(1) as f64, cum * 1e3);
    }
    let gb = 1024.0f64 * 1024.0 * 1024.0;
    let floor_s = (orig as f64 / gb) / 3.130;
    println!("T1.4 floor 3.130 GB/s allows {:.2} ms total for {} bytes", floor_s * 1e3, orig);

    if let Ok(mut fh) = std::fs::OpenOptions::new().create(true).append(true).open("huff_probe_log.txt") {
        let _ = writeln!(fh, "ratio_now={:.5} saves: {}", ratio_now,
            st.iter().map(|s| format!("{}={}", s.name, s.saving())).collect::<Vec<_>>().join(" "));
        let _ = fh.flush();
    }
}
