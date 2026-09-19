// Actual ultra-level output next to the order-0 estimate of its parse
// (literal entropy, code entropies, extra bits), per file: the coder's
// overhead over its own model.
use glyd::v7_encode::Sequence;
use glyd::v7_format::{ll_code, ml_code, LL_SYMBOLS, ML_SYMBOLS, OFF_SYMBOLS};
use glyd::v7_ultra::{find_sequences_ultra, UltraState};

fn ent(h: &[u64]) -> f64 { let t: u64 = h.iter().sum(); h.iter().filter(|&&c| c > 0).map(|&c| c as f64 * ((t as f64) / c as f64).log2()).sum() }

fn main() {
    let mut files: Vec<_> = std::fs::read_dir("corpus").unwrap().filter_map(|e| e.ok()).map(|e| e.path())
        .filter(|p| p.is_file() && !p.file_name().unwrap().to_string_lossy().starts_with("enwik")).collect();
    files.sort();
    let mut st = UltraState::new();
    let (mut ta, mut te) = (0f64, 0f64);
    for p in files {
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        let d = std::fs::read(&p).unwrap();
        let mut out = Vec::new();
        glyd::compress_into_ultra(&d, &mut out);
        st.clear();
        let (mut seqs, mut lits): (Vec<Sequence>, Vec<u8>) = (Vec::new(), Vec::new());
        let (mut est_lit, mut est_seq, mut xbits, mut nblocks) = (0f64, 0f64, 0u64, 0usize);
        let mut off = 0;
        while off < d.len() {
            let len = (d.len() - off).min(256 * 1024);
            find_sequences_ultra(&d, off, len, &mut st, [1, 4, 8], &mut seqs, &mut lits);
            let mut hl = [0u64; 256]; for &b in &lits { hl[b as usize] += 1; }
            let (mut hll, mut hml, mut hoff) = ([0u64; LL_SYMBOLS], [0u64; ML_SYMBOLS], [0u64; OFF_SYMBOLS]);
            let mut r = [1u32, 4, 8];
            for q in &seqs {
                let (c, nb, _) = ll_code(q.lit_len); hll[c as usize] += 1; xbits += nb as u64;
                if q.match_len == 0 { continue; }
                let (c, nb, _) = ml_code(q.match_len); hml[c as usize] += 1; xbits += nb as u64;
                let c = if q.offset == r[0] { 0 } else if q.offset == r[1] { 1 } else if q.offset == r[2] { 2 } else { let k = 31 - q.offset.leading_zeros(); xbits += k as u64; 3 + k as usize };
                hoff[c] += 1;
                r = if q.offset == r[0] { r } else if q.offset == r[1] { [r[1], r[0], r[2]] } else if q.offset == r[2] { [r[2], r[0], r[1]] } else { [q.offset, r[0], r[1]] };
            }
            est_lit += ent(&hl).min(8.0 * lits.len() as f64);
            est_seq += ent(&hll) + ent(&hml) + ent(&hoff);
            nblocks += 1;
            off += len;
        }
        let est = (est_lit + est_seq + xbits as f64) / 8.0;
        println!("{name:12} actual {:9.0} KB  estimate {:9.0} KB  overhead {:+.2}%  ({} blocks, {:.0} B/block)", out.len() as f64 / 1024.0, est / 1024.0, 100.0 * (out.len() as f64 / est - 1.0), nblocks, (out.len() as f64 - est) / nblocks as f64);
        ta += out.len() as f64; te += est;
    }
    println!("total        actual {:9.0} KB  estimate {:9.0} KB  overhead {:+.2}%", ta / 1024.0, te / 1024.0, 100.0 * (ta / te - 1.0));
}
