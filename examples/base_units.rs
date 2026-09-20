// Scratch: per-unit compressed size of a base-mode stream, and the plain size of the same units.
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let old = std::fs::read(&a[1]).unwrap();
    let new = std::fs::read(&a[2]).unwrap();
    let mut c = Vec::new();
    glyd::compress_with_base(&old, &new, &mut c, false);
    // parse the envelope: magic 8, id 8, n varint, then per unit 4 varints
    let mut pos = 16usize;
    let rd = |pos: &mut usize| -> u64 { let (mut v, mut sh) = (0u64, 0); loop { let b = c[*pos]; *pos += 1; v |= ((b & 127) as u64) << sh; if b < 128 { return v; } sh += 7; } };
    let n = rd(&mut pos) as usize;
    let mut worst = Vec::new();
    for i in 0..n {
        let (bs, bl, len, slen) = (rd(&mut pos), rd(&mut pos), rd(&mut pos), rd(&mut pos));
        worst.push((slen, i, bs, bl, len));
    }
    worst.sort_unstable_by(|x, y| y.0.cmp(&x.0));
    let total: u64 = worst.iter().map(|w| w.0).sum();
    println!("{} units, {} B total; the 8 largest:", n, total);
    for (slen, i, bs, bl, len) in worst.iter().take(8) {
        println!("   unit {:>3}: {:>9} B for {} B (region {}..{}, {:.1}%)", i, slen, len, bs, bs + bl, 100.0 * *slen as f64 / *len as f64);
    }
    let small: u64 = worst.iter().rev().take(n / 2).map(|w| w.0).sum();
    println!("   the smaller half of the units together: {} B", small);
}
