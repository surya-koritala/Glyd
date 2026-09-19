// Bytes per entropy table under the current encodings and candidate
// compact ones, from the tables the ultra parse would build per block.
use glyd::v7_encode::Sequence;
use glyd::v7_format::{ll_code, ml_code, LL_SYMBOLS, ML_SYMBOLS, OFF_SYMBOLS};
use glyd::v7_ultra::{find_sequences_ultra, UltraState};
use glyd::{huff8, tans};

fn bitlen(c: u32) -> u32 { 32 - c.leading_zeros() }
// Counts: 4-bit width then width-1 mantissa bits.
fn counts_nibble(counts: &[u16]) -> f64 { counts.iter().map(|&c| 4.0 + bitlen(c as u32).saturating_sub(1) as f64).sum::<f64>() / 8.0 }
// Counts: zstd-style, each count in log2(remaining + 1) bits (rounded up).
fn counts_remaining(counts: &[u16]) -> f64 {
    let mut rem = 1024i32; let mut bits = 0f64;
    for &c in counts { let w = bitlen((rem.max(0) + 1) as u32); bits += w as f64; rem -= c as i32; if rem <= 0 { break; } }
    bits / 8.0
}
// Lengths: zero runs as (0, 4-bit run-1), other lengths as 4 bits.
fn lengths_rle(lengths: &[u8; 256]) -> f64 {
    let mut bits = 0f64; let mut i = 0;
    while i < 256 { if lengths[i] == 0 { let mut r = 0; while i < 256 && lengths[i] == 0 && r < 16 { r += 1; i += 1; } bits += 8.0; } else { bits += 4.0; i += 1; } }
    bits / 8.0
}
// Lengths: entropy of the 256 nibbles (a static code's floor).
fn lengths_entropy(lengths: &[u8; 256]) -> f64 {
    let mut h = [0u32; 16]; for &l in lengths { h[l as usize] += 1; }
    h.iter().filter(|&&c| c > 0).map(|&c| c as f64 * (256.0 / c as f64).log2()).sum::<f64>() / 8.0
}

fn main() {
    let mut files: Vec<_> = std::fs::read_dir("corpus").unwrap().filter_map(|e| e.ok()).map(|e| e.path())
        .filter(|p| p.is_file() && !p.file_name().unwrap().to_string_lossy().starts_with("enwik")).collect();
    files.sort();
    let mut st = UltraState::new();
    let (mut n, mut lit_now, mut lit_rle, mut lit_ent, mut seq_now, mut seq_nib, mut seq_rem) = (0f64, 0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
    for p in files {
        let d = std::fs::read(&p).unwrap();
        st.clear(d.len());
        let (mut seqs, mut lits): (Vec<Sequence>, Vec<u8>) = (Vec::new(), Vec::new());
        let mut off = 0;
        while off < d.len() {
            let len = (d.len() - off).min(256 * 1024);
            find_sequences_ultra(&d, off, len, &mut st, [1, 4, 8], &mut seqs, &mut lits);
            let mut hl = [0u64; 256]; for &b in &lits { hl[b as usize] += 1; }
            let lengths = huff8::lengths_for(&hl);
            lit_now += 128.0; lit_rle += lengths_rle(&lengths); lit_ent += lengths_entropy(&lengths);
            let (mut hll, mut hml, mut hoff) = (vec![0u32; LL_SYMBOLS], vec![0u32; ML_SYMBOLS], vec![0u32; OFF_SYMBOLS]);
            let mut r = [1u32, 4, 8];
            for q in &seqs {
                hll[ll_code(q.lit_len).0 as usize] += 1;
                if q.match_len == 0 { continue; }
                hml[ml_code(q.match_len).0 as usize] += 1;
                let c = if q.offset == r[0] { 0 } else if q.offset == r[1] { 1 } else if q.offset == r[2] { 2 } else { 3 + (31 - q.offset.leading_zeros()) as usize };
                hoff[c] += 1;
                r = if q.offset == r[0] { r } else if q.offset == r[1] { [r[1], r[0], r[2]] } else if q.offset == r[2] { [r[2], r[0], r[1]] } else { [q.offset, r[0], r[1]] };
            }
            for (h, ns) in [(&hll, LL_SYMBOLS), (&hml, ML_SYMBOLS), (&hoff, OFF_SYMBOLS)] {
                let counts = tans::normalize(h, ns);
                seq_now += 1.0 + (ns as f64 * 11.0 / 8.0).ceil();
                seq_nib += 1.0 + counts_nibble(&counts).ceil();
                seq_rem += 1.0 + counts_remaining(&counts).ceil();
            }
            n += 1.0;
            off += len;
        }
    }
    println!("per block, {n:.0} blocks: literal table now {:.0} B, zero-RLE nibbles {:.1} B, nibble entropy {:.1} B", lit_now / n, lit_rle / n, lit_ent / n);
    println!("three tANS tables now {:.0} B, nibble-width counts {:.1} B, zstd-style remaining {:.1} B", seq_now / n, seq_nib / n, seq_rem / n);
}
