// Scratch: the cost of the pieces of a record rebuild.
fn main() {
    let n = 4_000_000usize;
    let mut x = 7u64;
    let mut rnd = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; x };
    let vals: Vec<i64> = (0..n).map(|i| if i % 2 == 0 { 1_000_000 + i as i64 * 3 } else { (rnd() % 1000) as i64 }).collect();
    // varint stream of deltas
    let mut stream = Vec::new();
    let mut last = 0i64;
    for &v in &vals { let mut z = (((v - last) << 1) ^ ((v - last) >> 63)) as u64; last = v; while z >= 128 { stream.push((z & 127) as u8 | 128); z >>= 7; } stream.push(z as u8); }
    let t = std::time::Instant::now();
    let mut pos = 0; let mut acc = 0u64; let mut k = 0;
    while pos < stream.len() {
        let (mut v, mut sh) = (0u64, 0);
        loop { let b = stream[pos]; pos += 1; v |= ((b & 127) as u64) << sh; if b < 128 { break; } sh += 7; }
        acc = acc.wrapping_add(v); k += 1;
    }
    println!("varint decode: {:.1} ns/value ({})", t.elapsed().as_nanos() as f64 / k as f64, acc & 1);
    let mut out: Vec<u8> = Vec::with_capacity(n * 12);
    let t = std::time::Instant::now();
    for &v in &vals { push_int(&mut out, v); out.push(b','); }
    println!("push_int + comma: {:.1} ns/value ({} B)", t.elapsed().as_nanos() as f64 / n as f64, out.len());
    out.clear();
    let t = std::time::Instant::now();
    for &v in &vals { let s = v.to_string(); out.extend_from_slice(s.as_bytes()); out.push(b','); }
    println!("to_string + extend: {:.1} ns/value", t.elapsed().as_nanos() as f64 / n as f64);
    out.clear();
    let t = std::time::Instant::now();
    for &v in &vals { let mut b = itoa_buf(v); out.extend_from_slice(&b.0[b.1..]); out.push(b','); b.1 = 0; }
    println!("buffer itoa + extend: {:.1} ns/value", t.elapsed().as_nanos() as f64 / n as f64);
}
static DIGITS2: [u8; 200] = { let mut t = [0u8; 200]; let mut i = 0; while i < 100 { t[2*i] = b'0' + (i/10) as u8; t[2*i+1] = b'0' + (i%10) as u8; i += 1; } t };
fn push_int(out: &mut Vec<u8>, v: i64) {
    let mut u = v.unsigned_abs();
    out.reserve(21);
    unsafe {
        let base = out.as_mut_ptr().add(out.len());
        let mut p = base;
        if v < 0 { *p = b'-'; p = p.add(1); }
        let digits = if u < 10 { 1 } else if u < 100 { 2 } else if u < 10_000 { if u < 1000 { 3 } else { 4 } } else if u < 100_000_000 { if u < 1_000_000 { if u < 100_000 { 5 } else { 6 } } else if u < 10_000_000 { 7 } else { 8 } } else { 8 + { let mut n = 0; let mut t = u / 100_000_000; while t > 0 { n += 1; t /= 10; } n } };
        let mut i = digits;
        while i >= 2 { let q = u / 100; let r = (u - q * 100) as usize; i -= 2; std::ptr::copy_nonoverlapping(DIGITS2.as_ptr().add(2 * r), p.add(i), 2); u = q; }
        if i == 1 { *p = b'0' + u as u8; }
        out.set_len(out.len() + digits + (v < 0) as usize);
    }
}
fn itoa_buf(v: i64) -> ([u8; 24], usize) {
    let mut b = [0u8; 24]; let mut i = 24; let mut u = v.unsigned_abs();
    loop { i -= 1; b[i] = b'0' + (u % 10) as u8; u /= 10; if u == 0 { break; } }
    if v < 0 { i -= 1; b[i] = b'-'; }
    (b, i)
}
