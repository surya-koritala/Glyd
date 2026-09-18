// Which 8-bit token layout minimises escapes? Escapes cost ~5 ns each in the
// decoder (dec_ablate: 16 ms of a 66 ms decode at a 25% escape rate). This
// measures the real literal-length, match-length and offset distributions of
// our token stream and costs every plausible split of the 8 token bits:
// escape rate, and the byte cost of the offset classes each split can afford.
use simd_stream_codec::format::*;

fn main() {
    let files = ["dickens","mozilla","mr","nci","ooffice","osdb",
                 "reymont","samba","sao","webster","xml","x-ray"];
    let dir = std::path::Path::new("corpus");
    let mut lit_h = vec![0u64; 70000];
    let mut match_h = vec![0u64; 70000];
    let mut off_bits = [0u64; 25]; // by bits needed
    let (mut ntok, mut nmatch, mut comp_total) = (0u64, 0u64, 0u64);
    for f in &files {
        let p = dir.join(f);
        if !p.exists() { continue; }
        let d = std::fs::read(&p).unwrap();
        let c = simd_stream_codec::compress(&d);
        comp_total += c.len() as u64;
        let mut cur = 0usize;
        while cur + HEADER_SIZE <= c.len() {
            let h = unsafe { std::ptr::read_unaligned(c.as_ptr().add(cur) as *const BlockHeader) };
            if h.flags & FLAG_RAW_UNCOMPRESSED == 0 {
                let bias = if h.flags & FLAG_DENSE != 0 { MATCH_CODE_BIAS_DENSE } else { MATCH_CODE_BIAS };
                let tb = cur + HEADER_SIZE;
                let ob = tb + h.token_bytes as usize;
                let eb = ob + h.offset_bytes as usize;
                let (mut oi, mut ei) = (0usize, 0usize);
                let read_esc = |c: &[u8], ei: &mut usize, base: usize| -> usize {
                    let v = c[eb + *ei] as usize; *ei += 1;
                    if v != ESCAPE_CONT as usize { base + v }
                    else { let w = u16::from_le_bytes([c[eb + *ei], c[eb + *ei + 1]]) as usize; *ei += 2; base + 255 + w }
                };
                for i in 0..h.token_count as usize {
                    let t = Token(c[tb + i]);
                    let lc = if t.lit_code() == LIT_CODE_ESCAPE { read_esc(&c, &mut ei, ESCAPE_BASE_LIT) } else { t.lit_code() };
                    let mc = t.match_code();
                    let rc = if mc == 0 { 0 } else if mc == MATCH_CODE_ESCAPE { read_esc(&c, &mut ei, bias + 15) } else { mc + bias };
                    lit_h[lc.min(69999)] += 1;
                    match_h[rc.min(69999)] += 1;
                    ntok += 1;
                    if rc > 0 {
                        nmatch += 1;
                        let v = u16::from_le_bytes([c[ob + oi], c[ob + oi + 1]]) as usize | (t.off_hi() << 16);
                        oi += OFFSET_BYTES;
                        let bits = if v == 0 { 1 } else { usize::BITS as usize - v.leading_zeros() as usize };
                        off_bits[bits.min(24)] += 1;
                    }
                }
            }
            cur += HEADER_SIZE + h.payload_len();
        }
    }
    let pct = |n: u64| 100.0 * n as f64 / ntok as f64;
    let cum = |h: &Vec<u64>, upto: usize| -> u64 { h[..=upto].iter().sum() };
    println!("{} tokens, {} matches, output {} bytes", ntok, nmatch, comp_total);
    println!("literal runs: 0 {:.1}%  <=2 {:.1}%  <=6 {:.1}%  <=14 {:.1}%  <=30 {:.1}%",
             pct(lit_h[0]), pct(cum(&lit_h, 2)), pct(cum(&lit_h, 6)), pct(cum(&lit_h, 14)), pct(cum(&lit_h, 30)));
    let pm = |n: u64| 100.0 * n as f64 / nmatch as f64;
    let mcum = |upto: usize| -> u64 { match_h[6..=upto].iter().sum() };
    println!("match lens:   <=12 {:.1}%  <=19 {:.1}%  <=20 {:.1}%  <=36 {:.1}%  <=68 {:.1}%",
             pm(mcum(12)), pm(mcum(19)), pm(mcum(20)), pm(mcum(36)), pm(mcum(68)));
    let mut acc = 0u64;
    print!("offset bits: ");
    for b in 1..=24 { acc += off_bits[b]; if b % 4 == 0 || b >= 16 { print!(" <={}b {:.1}%", b, pm(acc)); } }
    println!();

    // Cost each layout: (lit bits, match bits, width bits). Escape = lit >
    // direct max or match > direct max. Width bits give 2^w offset classes
    // (1 bit: 2/3 bytes; 2 bits: 1/2/3 bytes; 0 bits: fixed 3 bytes).
    println!();
    println!("{:<12} {:>9} {:>9} {:>9}   {:>12} {:>8}", "layout", "lit esc", "match esc", "any esc", "offset bytes", "ratio");
    let orig: u64 = 211938580;
    let cur_off_bytes: u64 = nmatch * OFFSET_BYTES as u64;
    for &(lb, mb, wb) in &[(2usize, 4usize, 2usize), (3, 4, 1), (3, 3, 2), (2, 5, 1), (4, 3, 1), (3, 5, 0), (4, 4, 0)] {
        let lit_direct = (1usize << lb) - 2; // last code escapes; codes 0..=direct
        let match_codes = (1usize << mb) - 2; // code 0 = none, last = escape
        let lit_esc: u64 = lit_h.iter().enumerate().filter(|(l, _)| *l > lit_direct).map(|(_, n)| n).sum();
        let m_esc: u64 = match_h.iter().enumerate().filter(|(l, _)| *l > 0 && *l > 5 + match_codes).map(|(_, n)| n).sum();
        // any-escape approximated as union assuming independence within a token
        let any = lit_esc as f64 + m_esc as f64 - (lit_esc as f64 * m_esc as f64 / ntok as f64);
        let off_bytes: u64 = (1..=24).map(|b| off_bits[b] * match wb {
            2 => if b <= 8 { 1 } else if b <= 16 { 2 } else { 3 },
            1 => if b <= 16 { 2 } else { 3 },
            _ => 3,
        }).sum();
        let out = comp_total as i64 - cur_off_bytes as i64 + off_bytes as i64;
        println!("{:<12} {:>8.1}% {:>8.1}% {:>8.1}%   {:>12} {:>8.4}",
                 format!("L{}/M{}/W{}", lb, mb, wb), pct(lit_esc), pct(m_esc), 100.0 * any / ntok as f64, off_bytes, orig as f64 / out as f64);
    }
}
