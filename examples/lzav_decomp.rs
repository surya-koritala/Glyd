// Where does LZAV's 2.45 come from: its byte format, or its parse?
//
// (a) Our exact token stream re-costed in LZAV's stream format 2 (from
//     lzav_write_blk_2 in lzav.h): "our parse, LZAV format".
// (b) LZAV's real output parsed block by block (from lzav_decompress_2):
//     reference count, literal bytes, offset and length distributions, and
//     what that parse would cost in our format: "LZAV parse, our format".
//
// Both numbers are exact byte counts on Silesia, not models.
use simd_stream_codec::format::*;
use simd_stream_codec::huffman;
use std::io::Write;

// LZAV format 2 cost of one (literal run, reference) pair. `mref` is the
// minimum reference length the format is parameterised with; LZAV uses 6, we
// pass 4 when costing our own parse so 4- and 5-byte matches stay legal.
fn lzav_cost(lc: usize, rc: usize, d: usize, mref: usize) -> usize {
    let mut bytes = 0usize;
    let mut d = d;
    if lc != 0 {
        d >>= 2; // two offset bits ride in the literal header
        bytes += if lc < 16 { 1 } else if lc < 16 + 128 { 2 } else {
            let mut n = 1; let mut w = lc - 16; while w > 127 { n += 1; w >>= 7; } n + 1
        };
        bytes += lc;
    }
    if rc == 0 { return bytes; }
    let rcp = rc + 1 - mref;
    let ext = if rcp < 16 { 0 } else if rcp < 16 + 255 { 1 } else { 2 };
    let ob = if d < (1 << 10) { 2 } else if d < (1 << 18) { 3 } else { 4 };
    bytes + ob + ext
}

/// Walk one v5 block's tokens, yielding (literal count, match length, offset).
fn walk_block(c: &[u8], cur: usize, h: &BlockHeader, mut f: impl FnMut(usize, usize, usize)) {
    let tb = cur + HEADER_SIZE;
    let ob = tb + h.token_bytes as usize;
    let eb = ob + h.offset_bytes as usize;
    let bias = if (h.flags & FLAG_DENSE) != 0 { MATCH_CODE_BIAS_DENSE } else { MATCH_CODE_BIAS };
    let (mut oi, mut ei) = (0usize, 0usize);
    let read_esc = |ei: &mut usize, base: usize| -> usize {
        let v = c[eb + *ei] as usize; *ei += 1;
        if v != ESCAPE_CONT as usize { base + v } else {
            let w = u16::from_le_bytes([c[eb + *ei], c[eb + *ei + 1]]) as usize; *ei += 2; base + 255 + w
        }
    };
    for i in 0..h.token_count as usize {
        let t = Token(c[tb + i]);
        let lc = if t.lit_code() == LIT_CODE_ESCAPE { read_esc(&mut ei, ESCAPE_BASE_LIT) } else { t.lit_code() };
        let mc = t.match_code();
        let rc = if mc == 0 { 0 } else if mc == MATCH_CODE_ESCAPE { read_esc(&mut ei, bias + 15) } else { mc + bias };
        let off = if rc > 0 {
            let v = u16::from_le_bytes([c[ob + oi], c[ob + oi + 1]]) as usize | (t.off_hi() << 16);
            oi += OFFSET_BYTES;
            v
        } else { 0 };
        f(lc, rc, off);
    }
}

#[derive(Default)]
struct Parse {
    seq: Vec<(u32, u32, u32)>,
    refs: u64, lit_bytes: u64, match_bytes: u64, lit_runs: u64,
    off_lt1k: u64, off_lt4k: u64, off_lt64k: u64, off_ge64k: u64,
    len_le20: u64, len_le33: u64, len_gt33: u64,
    lit0: u64, lit_le6: u64, lit_le15: u64,
    bytes: u64,
}

impl Parse {
    fn add(&mut self, lc: usize, rc: usize, d: usize) {
        self.seq.push((lc as u32, rc as u32, d as u32));
        self.lit_bytes += lc as u64;
        if lc > 0 { self.lit_runs += 1; }
        if rc == 0 { return; }
        self.refs += 1;
        self.match_bytes += rc as u64;
        if d < 1024 { self.off_lt1k += 1 } else if d < 4096 { self.off_lt4k += 1 }
        else if d < 65536 { self.off_lt64k += 1 } else { self.off_ge64k += 1 }
        if rc <= 20 { self.len_le20 += 1 } else if rc <= 33 { self.len_le33 += 1 } else { self.len_gt33 += 1 }
        if lc == 0 { self.lit0 += 1 } else if lc <= 6 { self.lit_le6 += 1 } else if lc <= 15 { self.lit_le15 += 1 }
    }
    // Cost in our v5 format: token + literals + 1..3 offset bytes, byte escapes.
    fn our_format_cost(lc: usize, rc: usize, d: usize) -> usize {
        let mut b = 1 + lc;
        if lc > 2 { b += if lc - 3 < 255 { 1 } else { 3 }; }
        if rc > 0 {
            b += if d < 256 { 1 } else if d < 65536 { 2 } else { 3 };
            if rc > 19 { b += if rc - 20 < 255 { 1 } else { 3 }; }
        }
        b
    }
}

/// Parse LZAV format-2 output, mirroring lzav_decompress_2.
fn parse_lzav(src: &[u8], out_len: usize) -> Parse {
    let mut p = Parse::default();
    let mref1 = (src[0] & 15) as usize - 1;
    let mut ip = 1usize;
    let (mut cv, mut csh) = (0usize, 0u32);
    let mut bh = src[ip] as usize;
    let ipet = src.len() - 6;
    let ld32 = |i: usize| -> u32 {
        let mut b = [0u8; 4];
        for k in 0..4 { if i + k < src.len() { b[k] = src[i + k]; } }
        u32::from_le_bytes(b)
    };
    let mut pending_lc = 0usize;
    let mut produced = 0usize;
    while ip < ipet {
        if bh & 0x30 == 0 {
            let mut ncv = bh >> 6;
            ip += 1;
            let mut cc = bh & 15;
            if cc != 0 {
                ncv <<= csh;
                ip += cc;
            } else {
                let mut lcw = src[ip] as usize;
                ncv <<= csh;
                ip += 1;
                cc = lcw & 0x7F;
                let mut sh = 7;
                while lcw & 0x80 != 0 {
                    lcw = src[ip] as usize; ip += 1;
                    cc |= (lcw & 0x7F) << sh;
                    if sh == 28 { break; }
                    sh += 7;
                }
                cc += 16;
                ip += cc;
            }
            cv |= ncv;
            csh += 2;
            if ip < src.len() { bh = src[ip] as usize; }
            pending_lc += cc;
            produced += cc;
            continue;
        }
        let bt = (bh >> 4) & 3;
        ip += 1;
        let bt8 = bt * 8;
        let bv = ld32(ip);
        ip += bt;
        let o = (bv & (0xFFFF_FFFFu32 >> (32 - bt8))) as usize;
        let d = (((bh >> 6) | ((o & 0x1F_FFFF) << 2)) << csh) | cv;
        csh = ((3 + (bt != 3) as u32) & 3) as u32;
        cv = o >> 21;
        let mut cc = bh & 15;
        if cc != 0 {
            bh = ((bv >> bt8) & 0xFF) as usize;
            cc += mref1;
        } else {
            bh = ((bv >> bt8) & 0xFF) as usize;
            if bh == 255 {
                cc = 16 + mref1 + 255 + src[ip + 1] as usize;
                bh = src[ip + 2] as usize;
                ip += 2;
            } else {
                cc = 16 + mref1 + bh;
                ip += 1;
                bh = src[ip] as usize;
            }
        }
        p.add(pending_lc, cc, d);
        p.bytes += Parse::our_format_cost(pending_lc, cc, d) as u64;
        pending_lc = 0;
        produced += cc;
    }
    // Whatever literal tail is left (final LZAV_LIT_FIN block).
    if pending_lc > 0 { p.add(pending_lc, 0, 0); p.bytes += Parse::our_format_cost(pending_lc, 0, 0) as u64; }
    if produced != out_len {
        eprintln!("  warning: parsed {} bytes, expected {} (ip {} of {}, ipet {}, refs {}, lit {})", produced, out_len, ip, src.len(), ipet, p.refs, p.lit_bytes);
    }
    p
}

fn main() {
    let files = ["dickens","mozilla","mr","nci","ooffice","osdb",
                 "reymont","samba","sao","webster","xml","x-ray"];
    let dir = std::path::Path::new("corpus");
    let (mut orig, mut ours, mut ours_in_lzav, mut lzav_bytes) = (0u64, 0u64, 0u64, 0u64);
    let mut ourp = Parse::default();
    let mut lzp = Parse::default();

    for f in &files {
        let path = dir.join(f);
        if !path.exists() { continue; }
        let d = std::fs::read(&path).unwrap();
        orig += d.len() as u64;

        // (a) our parse costed in LZAV's format (mref 4 so all our matches are legal)
        let c = simd_stream_codec::compress(&d);
        ours += c.len() as u64;
        let mut cur = 0usize;
        while cur + HEADER_SIZE <= c.len() {
            let h = unsafe { std::ptr::read_unaligned(c.as_ptr().add(cur) as *const BlockHeader) };
            let raw = (h.flags & FLAG_RAW_UNCOMPRESSED) != 0;
            let pay = h.payload_len();
            if raw { ours_in_lzav += pay as u64 + 2; } else {
                walk_block(&c, cur, &h, |lc, rc, off| {
                    ourp.add(lc, rc, off);
                    ours_in_lzav += lzav_cost(lc, rc, off, 4) as u64;
                });
            }
            cur += HEADER_SIZE + pay;
        }

        // (b) LZAV's real parse
        let bound = unsafe { lzav::compress_bound(d.len() as i32) } as usize;
        let mut buf = vec![0u8; bound];
        let n = unsafe { lzav::compress_default(d.as_ptr() as *const _, buf.as_mut_ptr() as *mut _, d.len() as i32, bound as i32) } as usize;
        assert!(n > 0);
        buf.truncate(n);
        lzav_bytes += n as u64;
        let p = parse_lzav(&buf, d.len());
        lzp.refs += p.refs; lzp.lit_bytes += p.lit_bytes; lzp.match_bytes += p.match_bytes;
        lzp.lit_runs += p.lit_runs; lzp.off_lt1k += p.off_lt1k; lzp.off_lt4k += p.off_lt4k;
        lzp.off_lt64k += p.off_lt64k; lzp.off_ge64k += p.off_ge64k; lzp.len_le20 += p.len_le20;
        lzp.len_le33 += p.len_le33; lzp.len_gt33 += p.len_gt33; lzp.lit0 += p.lit0;
        lzp.lit_le6 += p.lit_le6; lzp.lit_le15 += p.lit_le15; lzp.bytes += p.bytes;
        lzp.seq.extend_from_slice(&p.seq);
        println!("{:<8} ours {:>9}  ours-in-lzav-fmt {:>9}  lzav {:>9}  lzav-in-our-fmt {:>9}  refs ours {:>8} lzav {:>8}",
                 f, c.len(), ours_in_lzav, n, p.bytes, ourp.refs, p.refs);
        ourp.refs = 0; // per-file print only; totals recomputed below
    }

    // Recompute our totals (refs zeroed per file above), simplest to re-parse once more.
    let mut tot = Parse::default();
    for f in &files {
        let path = dir.join(f);
        if !path.exists() { continue; }
        let d = std::fs::read(&path).unwrap();
        let c = simd_stream_codec::compress(&d);
        let mut cur = 0usize;
        while cur + HEADER_SIZE <= c.len() {
            let h = unsafe { std::ptr::read_unaligned(c.as_ptr().add(cur) as *const BlockHeader) };
            let raw = (h.flags & FLAG_RAW_UNCOMPRESSED) != 0;
            let pay = h.payload_len();
            if !raw {
                walk_block(&c, cur, &h, |lc, rc, off| tot.add(lc, rc, off));
            }
            cur += HEADER_SIZE + pay;
        }
    }

    let r = |b: u64| orig as f64 / b as f64;
    println!();
    println!("Silesia {} bytes", orig);
    println!("  ours, our format        {:>10}  ratio {:.5}", ours, r(ours));
    println!("  ours, LZAV format       {:>10}  ratio {:.5}   <- format effect on our parse", ours_in_lzav, r(ours_in_lzav));
    println!("  LZAV, LZAV format       {:>10}  ratio {:.5}", lzav_bytes, r(lzav_bytes));
    println!("  LZAV parse, our format  {:>10}  ratio {:.5}   <- parse effect, v5 layout", lzp.bytes, r(lzp.bytes));
    println!();
    let pr = |name: &str, p: &Parse| {
        let refs = p.refs.max(1) as f64;
        println!("{:<6} refs {:>9}  lit bytes {:>9} ({:.1}% of input)  match bytes {:>9}  avg len {:.2}",
                 name, p.refs, p.lit_bytes, 100.0 * p.lit_bytes as f64 / orig as f64, p.match_bytes, p.match_bytes as f64 / refs);
        println!("       offsets <1K {:>5.1}%  <4K {:>5.1}%  <64K {:>5.1}%  >=64K {:>5.1}%",
                 100.0 * p.off_lt1k as f64 / refs, 100.0 * p.off_lt4k as f64 / refs,
                 100.0 * p.off_lt64k as f64 / refs, 100.0 * p.off_ge64k as f64 / refs);
        println!("       len <=20 {:>5.1}%  21..33 {:>5.1}%  >33 {:>5.1}%   lit before ref: 0 {:>5.1}%  1..6 {:>5.1}%  7..15 {:>5.1}%",
                 100.0 * p.len_le20 as f64 / refs, 100.0 * p.len_le33 as f64 / refs, 100.0 * p.len_gt33 as f64 / refs,
                 100.0 * p.lit0 as f64 / refs, 100.0 * p.lit_le6 as f64 / refs, 100.0 * p.lit_le15 as f64 / refs);
    };
    pr("ours", &tot);
    pr("lzav", &lzp);

    format_study(orig, &tot, &lzp);

    if let Ok(mut fh) = std::fs::OpenOptions::new().create(true).append(true).open("lzav_decomp_log.txt") {
        let _ = writeln!(fh, "ours={} ours_in_lzav={} lzav={} lzav_in_ours={} refs ours={} lzav={} lit ours={} lzav={}",
                         ours, ours_in_lzav, lzav_bytes, lzp.bytes, tot.refs, lzp.refs, tot.lit_bytes, lzp.lit_bytes);
    }
}

// ---- candidate stream formats, costed exactly on a parse ----
//
// A token byte splits into an offset-class field, a match-length field and a
// literal-count field. Offsets go to a byte-oriented stream at 1, 2 or 3
// bytes by class. Lengths and counts that do not fit their field escape to a
// side stream, either as u16 (like v3) or as a byte with a 255 continuation
// (like LZAV).
struct Fmt { name: &'static str, off_bits: u32, len_bits: u32, lit_bits: u32, mref: u32, esc16: bool }

fn fmt_cost(f: &Fmt, seq: &[(u32, u32, u32)]) -> u64 {
    let len_direct = (1u32 << f.len_bits) - 1; // last value escapes
    let lit_direct = (1u32 << f.lit_bits) - 1;
    let mut bytes = 0u64;
    for &(lc, rc, d) in seq {
        bytes += 1 + lc as u64; // token + literal bytes
        if lc >= lit_direct {
            let v = lc - lit_direct;
            bytes += if f.esc16 { 2 } else if v < 255 { 1 } else if v < 255 + 255 { 2 } else { 3 };
        }
        if rc == 0 { continue; }
        let rv = rc.saturating_sub(f.mref);
        if rv >= len_direct {
            let v = rv - len_direct;
            bytes += if f.esc16 { 2 } else if v < 255 { 1 } else if v < 255 + 255 { 2 } else { 3 };
        }
        bytes += match f.off_bits {
            0 => 3,
            1 => if d < 65536 { 2 } else { 3 },
            _ => if d < 256 { 1 } else if d < 65536 { 2 } else { 3 },
        };
    }
    bytes
}

fn format_study(orig: u64, ours: &Parse, lzav: &Parse) {
    println!();
    // finer offset histogram on LZAV's parse
    let mut h = [0u64; 6];
    for &(_, rc, d) in &lzav.seq {
        if rc == 0 { continue; }
        let b = if d < 256 { 0 } else if d < 4096 { 1 } else if d < 65536 { 2 } else if d < (1 << 20) { 3 } else if d < (1 << 22) { 4 } else { 5 };
        h[b] += 1;
    }
    let n = lzav.refs.max(1) as f64;
    println!("LZAV offsets: <256 {:.1}%  <4K {:.1}%  <64K {:.1}%  <1M {:.1}%  <4M {:.1}%  >=4M {:.1}%",
             100.0 * h[0] as f64 / n, 100.0 * h[1] as f64 / n, 100.0 * h[2] as f64 / n,
             100.0 * h[3] as f64 / n, 100.0 * h[4] as f64 / n, 100.0 * h[5] as f64 / n);
    let mut lh = [0u64; 5];
    let mut ch = [0u64; 5];
    for &(lc, rc, _) in &lzav.seq {
        if rc > 0 { lh[if rc <= 20 { 0 } else if rc <= 33 { 1 } else if rc <= 270 { 2 } else if rc <= 525 { 3 } else { 4 }] += 1; }
        ch[if lc == 0 { 0 } else if lc <= 2 { 1 } else if lc <= 6 { 2 } else if lc <= 15 { 3 } else { 4 }] += 1;
    }
    println!("LZAV match len: <=20 {:.1}%  21..33 {:.1}%  34..270 {:.1}%  271..525 {:.1}%  >525 {:.1}%",
             100.0 * lh[0] as f64 / n, 100.0 * lh[1] as f64 / n, 100.0 * lh[2] as f64 / n, 100.0 * lh[3] as f64 / n, 100.0 * lh[4] as f64 / n);
    let t = lzav.seq.len().max(1) as f64;
    println!("LZAV lit count: 0 {:.1}%  1..2 {:.1}%  3..6 {:.1}%  7..15 {:.1}%  >15 {:.1}%",
             100.0 * ch[0] as f64 / t, 100.0 * ch[1] as f64 / t, 100.0 * ch[2] as f64 / t, 100.0 * ch[3] as f64 / t, 100.0 * ch[4] as f64 / t);
    println!();
    let fmts = [
        Fmt { name: "v3 as is (3 lit/5 len/u16 off)", off_bits: 0, len_bits: 5, lit_bits: 3, mref: 3, esc16: true },
        Fmt { name: "A: 2 off/4 len/2 lit, esc u16", off_bits: 2, len_bits: 4, lit_bits: 2, mref: 4, esc16: true },
        Fmt { name: "A: 2 off/4 len/2 lit, esc u8+", off_bits: 2, len_bits: 4, lit_bits: 2, mref: 4, esc16: false },
        Fmt { name: "B: 1 off/4 len/3 lit, esc u8+", off_bits: 1, len_bits: 4, lit_bits: 3, mref: 4, esc16: false },
        Fmt { name: "C: 2 off/3 len/3 lit, esc u8+", off_bits: 2, len_bits: 3, lit_bits: 3, mref: 4, esc16: false },
        Fmt { name: "D: 1 off/5 len/2 lit, esc u8+", off_bits: 1, len_bits: 5, lit_bits: 2, mref: 4, esc16: false },
        Fmt { name: "E: 2 off/4 len/2 lit mref6 u8+", off_bits: 2, len_bits: 4, lit_bits: 2, mref: 6, esc16: false },
    ];
    println!("{:<34} {:>12} {:>8}   {:>12} {:>8}", "format", "our parse", "ratio", "LZAV parse", "ratio");
    for f in &fmts {
        let a = fmt_cost(f, &ours.seq);
        let b = fmt_cost(f, &lzav.seq);
        println!("{:<34} {:>12} {:>8.5}   {:>12} {:>8.5}{}", f.name, a, orig as f64 / a as f64, b, orig as f64 / b as f64,
                 if (orig as f64 / b as f64) >= 2.45 { "  T1.2 OK" } else { "" });
    }
    huff_study(orig, ours, lzav, &fmts);
}

// LZAV's own header semantics, split into streams: header bytes to a token
// stream, offset bytes to an offset stream, length escapes to an extras
// stream, literals to a literal stream. Returns (token bytes, other bytes).
fn split_cost(seq: &[(u32, u32, u32)], tokens: &mut Vec<u8>) -> u64 {
    let mut other = 0u64;
    for &(lc, rc, d) in seq {
        let mut d = d as usize;
        if lc > 0 {
            d >>= 2;
            let lc = lc as usize;
            if lc < 16 { tokens.push(lc as u8); } else {
                tokens.push(0);
                let mut w = lc - 16; other += 1; while w > 127 { w >>= 7; other += 1; }
            }
            other += lc as u64;
        }
        if rc == 0 { continue; }
        let rcp = rc as usize + 1 - 6;
        let (bt, ob) = if d < (1 << 10) { (1u8, 1) } else if d < (1 << 18) { (2u8, 2) } else { (3u8, 3) };
        other += ob;
        let hdr = (bt << 4) | (if rcp < 16 { rcp as u8 } else { 0 }) | (((d & 3) as u8) << 6);
        tokens.push(hdr);
        if rcp >= 16 { other += if rcp < 16 + 255 { 1 } else { 2 }; }
    }
    other
}

// Token bytes for a simple format (offset class / len / lit fields).
fn fmt_tokens(f: &Fmt, seq: &[(u32, u32, u32)], tokens: &mut Vec<u8>) {
    let len_direct = (1u32 << f.len_bits) - 1;
    let lit_direct = (1u32 << f.lit_bits) - 1;
    for &(lc, rc, d) in seq {
        let lit = lc.min(lit_direct);
        let len = if rc == 0 { 0 } else { (rc.saturating_sub(f.mref) + 1).min(len_direct) };
        let cls: u32 = if rc == 0 { 0 } else if f.off_bits == 1 { if d < 65536 { 0 } else { 1 } }
                       else { if d < 256 { 0 } else if d < 65536 { 1 } else { 2 } };
        let t = lit | (len << f.lit_bits) | (cls << (f.lit_bits + f.len_bits));
        tokens.push(t as u8);
    }
}

// Per-chunk Huffman size of a token stream with the real huffman.rs, table
// included, raw kept when smaller. Chunk of 16K tokens is about one 256 KB
// input block at LZAV's average match length.
fn huff_bytes(tokens: &[u8]) -> u64 {
    let mut total = 0u64;
    for ch in tokens.chunks(16384) {
        let mut hist = [0u64; 256];
        for &b in ch { hist[b as usize] += 1; }
        let lengths = huffman::build_lengths(&hist);
        let bits: u64 = (0..256).map(|i| hist[i] * lengths[i] as u64).sum();
        let coded = (bits + 7) / 8 + huffman::LENGTHS_BYTES as u64;
        total += coded.min(ch.len() as u64);
    }
    total
}

fn huff_study(orig: u64, ours: &Parse, lzav: &Parse, fmts: &[Fmt]) {
    println!();
    println!("{:<34} {:>10} {:>10} {:>8}   {:>10} {:>8}", "format, LZAV parse", "tokens", "other", "ratio", "huff tok", "ratio");
    let mut tk = Vec::new();
    let other = split_cost(&lzav.seq, &mut tk);
    let raw = tk.len() as u64 + other;
    let hf = huff_bytes(&tk) + other;
    println!("{:<34} {:>10} {:>10} {:>8.5}   {:>10} {:>8.5}{}", "S: LZAV headers, split streams", tk.len(), other,
             orig as f64 / raw as f64, hf, orig as f64 / hf as f64, if orig as f64 / hf as f64 >= 2.45 { "  T1.2 OK" } else { "" });
    for f in fmts.iter().skip(1) {
        let mut tk = Vec::new();
        fmt_tokens(f, &lzav.seq, &mut tk);
        let raw = fmt_cost(f, &lzav.seq);
        let other = raw - tk.len() as u64;
        let hf = huff_bytes(&tk) + other;
        println!("{:<34} {:>10} {:>10} {:>8.5}   {:>10} {:>8.5}{}", f.name, tk.len(), other,
                 orig as f64 / raw as f64, hf, orig as f64 / hf as f64, if orig as f64 / hf as f64 >= 2.45 { "  T1.2 OK" } else { "" });
    }
    let _ = ours;
}
