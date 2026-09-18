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

#[derive(Default)]
struct Parse {
    refs: u64, lit_bytes: u64, match_bytes: u64, lit_runs: u64,
    off_lt1k: u64, off_lt4k: u64, off_lt64k: u64, off_ge64k: u64,
    len_le20: u64, len_le33: u64, len_gt33: u64,
    lit0: u64, lit_le6: u64, lit_le15: u64,
    bytes: u64,
}

impl Parse {
    fn add(&mut self, lc: usize, rc: usize, d: usize) {
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
    // Cost in our v3 format: 1 token + 2 offset + literals, +2 per escape.
    fn our_format_cost(lc: usize, rc: usize) -> usize {
        let mut b = 1 + lc;
        if lc > 6 { b += 2; }
        if rc > 0 { b += 2; if rc > 33 { b += 2; } }
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
        p.bytes += Parse::our_format_cost(pending_lc, cc) as u64;
        pending_lc = 0;
        produced += cc;
    }
    // Whatever literal tail is left (final LZAV_LIT_FIN block).
    if pending_lc > 0 { p.add(pending_lc, 0, 0); p.bytes += Parse::our_format_cost(pending_lc, 0) as u64; }
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
            let (tc, oc, ec, ll) = (h.token_count as usize, h.offset_count as usize,
                                    h.extras_count as usize, h.literal_len as usize);
            let pay = if raw { h.uncompressed_len as usize } else { payload_len(tc, oc, ec, ll) };
            if raw { ours_in_lzav += pay as u64 + 2; } else {
                let tb = cur + HEADER_SIZE;
                let ob = tb + tc;
                let eb = ob + oc * 2;
                let (mut oi, mut ei) = (0usize, 0usize);
                for i in 0..tc {
                    let t = Token(c[tb + i]);
                    let lc = if t.lit_code() == LIT_CODE_ESCAPE {
                        let v = u16::from_le_bytes([c[eb + ei * 2], c[eb + ei * 2 + 1]]) as usize; ei += 1; v
                    } else { t.lit_code() };
                    let mc = t.match_code();
                    let rc = if mc == 0 { 0 } else if mc == MATCH_CODE_ESCAPE {
                        let v = u16::from_le_bytes([c[eb + ei * 2], c[eb + ei * 2 + 1]]) as usize; ei += 1; v
                    } else { mc + MATCH_CODE_BIAS };
                    let off = if rc > 0 {
                        let v = u16::from_le_bytes([c[ob + oi * 2], c[ob + oi * 2 + 1]]) as usize; oi += 1; v
                    } else { 0 };
                    ourp.add(lc, rc, off);
                    ours_in_lzav += lzav_cost(lc, rc, off, 4) as u64;
                }
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
            let (tc, oc, ec, ll) = (h.token_count as usize, h.offset_count as usize,
                                    h.extras_count as usize, h.literal_len as usize);
            let pay = if raw { h.uncompressed_len as usize } else { payload_len(tc, oc, ec, ll) };
            if !raw {
                let tb = cur + HEADER_SIZE; let ob = tb + tc; let eb = ob + oc * 2;
                let (mut oi, mut ei) = (0usize, 0usize);
                for i in 0..tc {
                    let t = Token(c[tb + i]);
                    let lc = if t.lit_code() == LIT_CODE_ESCAPE {
                        let v = u16::from_le_bytes([c[eb + ei * 2], c[eb + ei * 2 + 1]]) as usize; ei += 1; v
                    } else { t.lit_code() };
                    let mc = t.match_code();
                    let rc = if mc == 0 { 0 } else if mc == MATCH_CODE_ESCAPE {
                        let v = u16::from_le_bytes([c[eb + ei * 2], c[eb + ei * 2 + 1]]) as usize; ei += 1; v
                    } else { mc + MATCH_CODE_BIAS };
                    let off = if rc > 0 {
                        let v = u16::from_le_bytes([c[ob + oi * 2], c[ob + oi * 2 + 1]]) as usize; oi += 1; v
                    } else { 0 };
                    tot.add(lc, rc, off);
                }
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
    println!("  LZAV parse, our format  {:>10}  ratio {:.5}   <- parse effect (offsets >=64K costed as if legal)", lzp.bytes, r(lzp.bytes));
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

    if let Ok(mut fh) = std::fs::OpenOptions::new().create(true).append(true).open("lzav_decomp_log.txt") {
        let _ = writeln!(fh, "ours={} ours_in_lzav={} lzav={} lzav_in_ours={} refs ours={} lzav={} lit ours={} lzav={}",
                         ours, ours_in_lzav, lzav_bytes, lzp.bytes, tot.refs, lzp.refs, tot.lit_bytes, lzp.lit_bytes);
    }
}
