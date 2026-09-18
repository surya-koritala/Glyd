// Compression ceiling ladder over Silesia, one core, best of 5.
//
//   memcpy     read + write every byte
//   checksum   the block checksum alone (scalar on arm64)
//   hash_ins   hash every position into a 64 KB table, store position
//   hash_probe hash_ins plus load the candidate and compare 4 bytes
//   greedy     hash_probe plus extend the match and skip over it (LZ4-shape
//              parse, no output): the floor of any single-table greedy finder
//   finder     the real Lzav-port finder, streams emitted, no framing
//   lib        compress_into (finder + checksum + framing)
//   liblz4     same run
use std::time::Instant;
const GB: f64 = 1024.0 * 1024.0 * 1024.0;

#[inline(always)]
fn h4(w: u32) -> usize { (w.wrapping_mul(2654435761) >> 18) as usize } // 14 bits

#[inline(never)]
unsafe fn hash_ins(src: &[u8], table: &mut [u32]) -> usize {
    let p = src.as_ptr();
    let n = src.len().saturating_sub(8);
    let mut i = 0;
    while i < n {
        let w = std::ptr::read_unaligned(p.add(i) as *const u32);
        *table.get_unchecked_mut(h4(w)) = i as u32;
        i += 1;
    }
    table[0] as usize
}

#[inline(never)]
unsafe fn hash_probe(src: &[u8], table: &mut [u32]) -> usize {
    let p = src.as_ptr();
    let n = src.len().saturating_sub(8);
    let mut i = 0;
    let mut hits = 0usize;
    while i < n {
        let w = std::ptr::read_unaligned(p.add(i) as *const u32);
        let h = h4(w);
        let c = *table.get_unchecked(h) as usize;
        *table.get_unchecked_mut(h) = i as u32;
        let cw = std::ptr::read_unaligned(p.add(c) as *const u32);
        hits += (cw == w) as usize;
        i += 1;
    }
    hits
}

#[inline(never)]
unsafe fn greedy(src: &[u8], table: &mut [u32]) -> (usize, usize) {
    let p = src.as_ptr();
    let n = src.len().saturating_sub(12);
    let mut i = 0;
    let (mut tokens, mut mbytes) = (0usize, 0usize);
    let mut step = 1usize; let mut miss = 0usize;
    while i < n {
        let w = std::ptr::read_unaligned(p.add(i) as *const u32);
        let h = h4(w);
        let c = *table.get_unchecked(h) as usize;
        *table.get_unchecked_mut(h) = i as u32;
        let cw = std::ptr::read_unaligned(p.add(c) as *const u32);
        if cw == w && i - c < 65536 && i != c {
            // extend
            let mut l = 4;
            let max = (n - i).min(65535);
            while l + 8 <= max {
                let x = std::ptr::read_unaligned(p.add(i + l) as *const u64);
                let y = std::ptr::read_unaligned(p.add(c + l) as *const u64);
                if x != y { l += ((x ^ y).trailing_zeros() / 8) as usize; break; }
                l += 8;
            }
            tokens += 1; mbytes += l;
            i += l; step = 1; miss = 0;
        } else {
            i += step; miss += 1;
            if miss & 31 == 0 { step += 1; } // LZ4-style acceleration
        }
    }
    (tokens, mbytes)
}

/// find_matches_fast with emit replaced by counting: the parse alone.
#[inline(never)]
unsafe fn fast_noemit<const MIN: usize, const BITS: u32, const H5: bool, const WIN: usize, const RATIO: bool>(src_s: &[u8], block_start: usize, block_len: usize, table: &mut [u32]) -> (usize, usize) {
    let hh = |p: *const u8| -> usize { if H5 { ((std::ptr::read_unaligned(p as *const u64) << 24).wrapping_mul(889523592379u64) >> (64 - BITS)) as usize } else { (std::ptr::read_unaligned(p as *const u32).wrapping_mul(2654435761) >> (32 - BITS)) as usize } };
    let src = src_s.as_ptr();
    let block_end = block_start + block_len;
    let limit = block_end.saturating_sub(8).max(block_start);
    let (mut anchor, mut pos, mut search_nb) = (block_start, block_start, 64u32);
    let (mut ntok, mut outb) = (0usize, 0usize);
    'outer: loop {
        let (cand, mut rc);
        loop {
            if pos >= limit { break 'outer; }
            let h = hh(src.add(pos));
            let c = *table.get_unchecked(h) as usize;
            *table.get_unchecked_mut(h) = pos as u32;
            let step = (search_nb >> 6) as usize; search_nb += 1;
            let d = pos.wrapping_sub(c);
            if d >= 8 && d < WIN {
                let x = std::ptr::read_unaligned(src.add(pos) as *const u64) ^ std::ptr::read_unaligned(src.add(c) as *const u64);
                let len = if x == 0 { 8 } else { (x.trailing_zeros() / 8) as usize };
                if len >= MIN { cand = c; rc = len; break; }
            }
            pos += step;
        }
        search_nb = 64;
        let d = pos - cand;
        let ml = (block_end - pos).min(530);
        if rc == 8 && ml > 8 {
            let (mut a, mut b, mut n, max) = (src.add(pos + 8), src.add(cand + 8), 0usize, ml - 8);
            while n + 8 <= max { let x = std::ptr::read_unaligned(a as *const u64); let y = std::ptr::read_unaligned(b as *const u64); if x != y { n += ((x ^ y).trailing_zeros() / 8) as usize; break; } n += 8; a = a.add(8); b = b.add(8); }
            rc = 8 + n.min(max);
        }
        rc = rc.min(ml);
        let mut lc = pos - anchor; let mut mpos = pos;
        if lc != 0 { let mut room = if RATIO { lc.min(cand) } else { lc.min(cand).min(16) }; let mut bmc = 0usize;
            while room > 0 && *src.add(mpos - 1 - bmc) == *src.add(cand - 1 - bmc) { bmc += 1; room -= 1; }
            rc += bmc; mpos -= bmc; lc -= bmc; }
        ntok += 1; outb += 3 + lc + (lc > 6) as usize + (rc > MIN + 13) as usize; let _ = d;
        pos = mpos + rc; anchor = pos;
        if pos >= 2 && pos < limit { *table.get_unchecked_mut(hh(src.add(pos - 2))) = (pos - 2) as u32; }
        if RATIO { search_nb = 0; } // next probe steps by 0: re-check at pos before accelerating
    }
    (ntok, outb + (block_end - anchor))
}

/// LZ4-structured parse: forward hash one step ahead, immediate re-check
/// after a match. Counting only.
#[inline(never)]
unsafe fn fast_lz4style<const MIN: usize, const BITS: u32>(src_s: &[u8], block_start: usize, block_len: usize, table: &mut [u32]) -> (usize, usize) {
    let hh = |p: *const u8| -> usize { ((std::ptr::read_unaligned(p as *const u64) << 24).wrapping_mul(889523592379u64) >> (64 - BITS)) as usize };
    let src = src_s.as_ptr();
    let block_end = block_start + block_len;
    let limit = block_end.saturating_sub(12).max(block_start);
    let (mut anchor, mut ntok, mut outb) = (block_start, 0usize, 0usize);
    let mut ip = block_start;
    if ip >= limit { return (0, block_end - anchor); }
    *table.get_unchecked_mut(hh(src.add(ip))) = ip as u32;
    ip += 1;
    let mut fwd_h = hh(src.add(ip));
    'outer: loop {
        // search
        let mut cand;
        let mut rc;
        {
            let mut fwd = ip;
            let mut step = 1usize;
            let mut search_nb = 64u32;
            loop {
                let h = fwd_h;
                ip = fwd;
                fwd += step;
                step = (search_nb >> 6) as usize; search_nb += 1;
                if fwd > limit { break 'outer; }
                cand = *table.get_unchecked(h) as usize;
                fwd_h = hh(src.add(fwd));
                *table.get_unchecked_mut(h) = ip as u32;
                let d = ip.wrapping_sub(cand);
                if d >= 8 && d < 131072 {
                    let x = std::ptr::read_unaligned(src.add(ip) as *const u64) ^ std::ptr::read_unaligned(src.add(cand) as *const u64);
                    let len = if x == 0 { 8 } else { (x.trailing_zeros() / 8) as usize };
                    if len >= MIN { rc = len; break; }
                }
            }
        }
        loop {
            // catch up
            let mut mpos = ip; let mut c = cand;
            while mpos > anchor && c > 0 && *src.add(mpos - 1) == *src.add(c - 1) { mpos -= 1; c -= 1; rc += 1; }
            let ml = (block_end - mpos).min(530);
            if rc >= 8 + (ip - mpos) && ml > rc {
                let (mut a, mut b, mut n, max) = (src.add(mpos + rc), src.add(c + rc), 0usize, ml - rc);
                while n + 8 <= max { let x = std::ptr::read_unaligned(a as *const u64); let y = std::ptr::read_unaligned(b as *const u64); if x != y { n += ((x ^ y).trailing_zeros() / 8) as usize; break; } n += 8; a = a.add(8); b = b.add(8); }
                rc += n.min(max);
            }
            rc = rc.min(ml);
            let lc = mpos - anchor;
            ntok += 1; outb += 3 + lc + (lc > 6) as usize + (rc > MIN + 13) as usize;
            ip = mpos + rc; anchor = ip;
            if ip >= limit { break 'outer; }
            *table.get_unchecked_mut(hh(src.add(ip - 2))) = (ip - 2) as u32;
            // immediate re-check at ip
            let h = hh(src.add(ip));
            cand = *table.get_unchecked(h) as usize;
            *table.get_unchecked_mut(h) = ip as u32;
            let d = ip.wrapping_sub(cand);
            if d >= 8 && d < 131072 {
                let x = std::ptr::read_unaligned(src.add(ip) as *const u64) ^ std::ptr::read_unaligned(src.add(cand) as *const u64);
                let len = if x == 0 { 8 } else { (x.trailing_zeros() / 8) as usize };
                if len >= MIN { rc = len; continue; }
            }
            ip += 1;
            fwd_h = hh(src.add(ip));
            break;
        }
    }
    (ntok, outb + (block_end - anchor))
}

fn main() {
    let files = ["dickens","mozilla","mr","nci","ooffice","osdb",
                 "reymont","samba","sao","webster","xml","x-ray"];
    let mut data: Vec<Vec<u8>> = Vec::new();
    for f in &files {
        let p = std::path::Path::new("corpus").join(f);
        if p.exists() { data.push(std::fs::read(p).unwrap()); }
    }
    let total: usize = data.iter().map(|d| d.len()).sum();
    let time = |f: &mut dyn FnMut()| -> f64 {
        let mut best = f64::MAX;
        for _ in 0..5 { let t = Instant::now(); f(); let e = t.elapsed().as_secs_f64(); if e < best { best = e; } }
        best
    };
    let pr = |name: &str, t: f64, extra: String| {
        println!("{:<10} {:>8.1} ms  {:>5.2} ns/byte  {:>6.2} GB/s  {}", name, t * 1e3, t * 1e9 / total as f64, total as f64 / GB / t, extra);
    };
    let mut out = vec![0u8; total + 4096];
    let t_memcpy = time(&mut || { let mut o = 0; for d in &data { out[o..o + d.len()].copy_from_slice(d); o += d.len(); } });
    let t_ck = time(&mut || { let mut s = 0u32; for d in &data { for c in d.chunks(256 * 1024) { s ^= simd_stream_codec::compute_checksum(c); } } std::hint::black_box(s); });
    let mut table = vec![0u32; 1 << 14];
    let t_ins = time(&mut || { for d in &data { table.fill(0); unsafe { hash_ins(d, &mut table); } } });
    let t_probe = time(&mut || { for d in &data { table.fill(0); unsafe { hash_probe(d, &mut table); } } });
    let mut gt = (0, 0);
    let t_greedy = time(&mut || { gt = (0, 0); for d in &data { table.fill(0); let r = unsafe { greedy(d, &mut table) }; gt.0 += r.0; gt.1 += r.1; } });
    // real finder, no framing
    let mut tb = simd_stream_codec::finder::new_table();
    let (mut tk, mut of, mut ex, mut li) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut fout = 0usize;
    let t_finder = time(&mut || {
        fout = 0;
        for d in &data {
            simd_stream_codec::finder::init_table(&mut tb, d);
            let mut s = 0;
            while s < d.len() {
                let l = (d.len() - s).min(256 * 1024);
                tk.clear(); of.clear(); ex.clear(); li.clear();
                simd_stream_codec::fallback::compress_chained_fallback::<simd_stream_codec::finder::Lzav>(d, s, l, &mut tb, &mut tk, &mut of, &mut ex, &mut li);
                fout += tk.len() + of.len() + ex.len() + li.len();
                s += l;
            }
        }
    });
    let mut ft: Box<simd_stream_codec::finder::FastTable> = vec![0u32; simd_stream_codec::finder::FAST_HASH_SIZE].into_boxed_slice().try_into().unwrap();
    let mut fout2 = 0usize;
    let t_fast = time(&mut || {
        fout2 = 0;
        for d in &data {
            ft.fill(0);
            let mut s = 0;
            while s < d.len() {
                let l = (d.len() - s).min(256 * 1024);
                tk.clear(); of.clear(); ex.clear(); li.clear();
                let mut st = simd_stream_codec::finder::Streams::new(5, &mut tk, &mut of, &mut ex, &mut li);
                unsafe { simd_stream_codec::finder::find_matches_fast::<simd_stream_codec::finder::Dense>(d, s, l, &mut ft, &mut st); }
                fout2 += tk.len() + of.len() + ex.len() + li.len();
                s += l;
            }
        }
    });
    let mut fne = (0usize, 0usize);
    let t_fne = time(&mut || { fne = (0, 0); for d in &data { table.fill(0); let mut s = 0; while s < d.len() { let l = (d.len() - s).min(256 * 1024); let r = unsafe { fast_noemit::<7, 14, false, 131072, false>(d, s, l, &mut table) }; fne.0 += r.0; fne.1 += r.1; s += l; } } });
    let mut fne5 = (0usize, 0usize);
    let t_fne5 = time(&mut || { fne5 = (0, 0); for d in &data { table.fill(0); let mut s = 0; while s < d.len() { let l = (d.len() - s).min(256 * 1024); let r = unsafe { fast_noemit::<5, 14, false, 131072, false>(d, s, l, &mut table) }; fne5.0 += r.0; fne5.1 += r.1; s += l; } } });
    let mut fne4 = (0usize, 0usize);
    let t_fne4 = time(&mut || { fne4 = (0, 0); for d in &data { table.fill(0); let mut s = 0; while s < d.len() { let l = (d.len() - s).min(256 * 1024); let r = unsafe { fast_noemit::<4, 14, false, 131072, false>(d, s, l, &mut table) }; fne4.0 += r.0; fne4.1 += r.1; s += l; } } });
    macro_rules! ne { ($name:expr, $m:expr, $b:expr, $h:expr, $w:expr, $r:expr) => {{
        let mut r = (0usize, 0usize);
        let mut tb = vec![0u32; 1 << $b];
        let t = time(&mut || { r = (0, 0); for d in &data { tb.fill(0); let mut s = 0; while s < d.len() { let l = (d.len() - s).min(256 * 1024); let x = unsafe { fast_noemit::<$m, $b, $h, $w, $r>(d, s, l, &mut tb) }; r.0 += x.0; r.1 += x.1; s += l; } } });
        pr($name, t, format!("{} tokens, est ratio {:.3}", r.0, total as f64 / r.1 as f64));
    }}; }
    ne!("m5_h5_14", 5, 14, true, 131072, false);
    ne!("m5_h4_12", 5, 12, false, 131072, false);
    ne!("m5_h5_12", 5, 12, true, 131072, false);
    ne!("m5_h5_13", 5, 13, true, 131072, false);
    ne!("m5_13_ratio", 5, 13, true, 131072, true);
    ne!("m5_12_ratio", 5, 12, true, 131072, true);
    ne!("m5_13_w64k", 5, 13, true, 65536, false);
    ne!("m5_13_w32k", 5, 13, true, 32768, false);
    ne!("m5_14_w64k", 5, 14, true, 65536, false);
    ne!("m5_12_w64k", 5, 12, true, 65536, false);
    ne!("m5_h5_15", 5, 15, true, 131072, false);
    ne!("m5_h5_16", 5, 16, true, 131072, false);
    {
        let mut r = (0usize, 0usize);
        let mut tb = vec![0u32; 1 << 13];
        let t = time(&mut || { r = (0, 0); for d in &data { tb.fill(0); let mut s = 0; while s < d.len() { let l = (d.len() - s).min(256 * 1024); let x = unsafe { fast_lz4style::<5, 13>(d, s, l, &mut tb) }; r.0 += x.0; r.1 += x.1; s += l; } } });
        pr("lz4style_13", t, format!("{} tokens, est ratio {:.3}", r.0, total as f64 / r.1 as f64));
    }
    let mut cbuf = Vec::with_capacity(total);
    let mut clen = 0;
    let t_lib = time(&mut || { clen = 0; for d in &data { cbuf.clear(); simd_stream_codec::compress_into(d, &mut cbuf); clen += cbuf.len(); } });
    let mut clen2 = 0;
    let t_libfast = time(&mut || { clen2 = 0; for d in &data { cbuf.clear(); simd_stream_codec::compress_into_fast(d, &mut cbuf); clen2 += cbuf.len(); } });
    // round-trip check of the fast level
    if std::env::var("NOCHECK").is_err() { for d in &data { cbuf.clear(); simd_stream_codec::compress_into_fast(d, &mut cbuf); assert_eq!(&simd_stream_codec::decompress(&cbuf).unwrap(), d, "fast roundtrip"); } }
    let mut lz = vec![0u8; lz4::block::compress_bound(64 << 20).unwrap()];
    let mut lzlen = 0;
    let t_lz4 = time(&mut || { lzlen = 0; for d in &data { lzlen += lz4::block::compress_to_buffer(d, None, false, &mut lz).unwrap(); } });

    println!("{} MB input", total >> 20);
    pr("memcpy", t_memcpy, String::new());
    pr("checksum", t_ck, String::new());
    pr("hash_ins", t_ins, String::new());
    pr("hash_probe", t_probe, String::new());
    pr("greedy", t_greedy, format!("{} tokens, {:.1}% bytes in matches", gt.0, 100.0 * gt.1 as f64 / total as f64));
    pr("finder", t_finder, format!("ratio {:.3}", total as f64 / fout as f64));
    pr("fast_noemit", t_fne, format!("{} tokens, est ratio {:.3}", fne.0, total as f64 / fne.1 as f64));
    pr("noemit_m5", t_fne5, format!("{} tokens, est ratio {:.3}", fne5.0, total as f64 / fne5.1 as f64));
    pr("noemit_m4", t_fne4, format!("{} tokens, est ratio {:.3}", fne4.0, total as f64 / fne4.1 as f64));
    pr("fast", t_fast, format!("ratio {:.3}", total as f64 / fout2 as f64));
    pr("lib", t_lib, format!("ratio {:.3}", total as f64 / clen as f64));
    pr("lib_fast", t_libfast, format!("ratio {:.3}", total as f64 / clen2 as f64));
    pr("liblz4", t_lz4, format!("ratio {:.3}", total as f64 / lzlen as f64));
}
